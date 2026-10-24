//! Resolved axisymmetric rear-cavity, port, finite-baffle, and exterior fields.

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests;

use crate::conventions::StudyMetadata;
use crate::field_export::{
    ExportedSnapshot, FieldExportError, ProjectedCellVector, ProjectionProvenance,
    SpeakerFieldBundle, SpeakerFieldExporter, SpeakerPvdCollection, field_names,
};
use num_complex::Complex64;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use sonium_fem::assembly::{PrebakedAssembler, RegionAssemblyError};
use sonium_fem::boundary::optimal_sigma_max;
use sonium_fem::mesh::{BoundaryType, ElementType, Mesh, PhysicalGroupError, Point};
use sonium_fem::physics::acoustics::pressure::resolved::{
    ResolvedAcousticsError, add_axisymmetric_neumann, axisymmetric_interface_reaction_power,
    axisymmetric_kirchhoff_far_field, build_axisymmetric_pml_assembler,
    helmholtz_resonance_frequency_hz, positive_iomega_axisymmetric_exterior_pml,
};
use sonium_fem::physics::acoustics::pressure::{FluidProperties, thermal_wall_admittance};
use sonium_fem::solver::{
    SolverConfig, SolverError as FemSolverError, SolverType, solve_csr_with_guess,
};
use sonium_solvers::direct::SparseLuFactorization;
use sonium_solvers::{
    AmgConfig, AmgPhysicsProfile, AmgPreconditioner, CsrMatrix, FgmresConfig, IluPreconditioner,
    IlutConfig, IlutPreconditioner, Preconditioner, SolverError as LinearSolverError,
    fgmres_preconditioned,
};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::f64::consts::PI;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

pub const REAR_CAVITY: i32 = 40;
pub const RESOLVED_PORT: i32 = 41;
pub const PHYSICAL_EXTERIOR: i32 = 42;
pub const ACOUSTIC_PML: i32 = 43;

pub const FRONT_DIAPHRAGM: i32 = 140;
pub const REAR_DIAPHRAGM: i32 = 141;
pub const PORT_WALL: i32 = 142;
pub const CAVITY_WALL: i32 = 143;
pub const RIGID_SOLID_WALL: i32 = 144;
pub const OUTER_PML_TRUNCATION: i32 = 145;
pub const SYMMETRY_AXIS: i32 = 146;

/// Node-count cap for the direct-solve fallback in
/// [`ResolvedAcousticModel::direct_solve_fallback`]. Sparse 2D axisymmetric
/// factorizations stay cheap well past this size; the cap only stops a
/// stalled iterative solve on a genuinely large model from turning into an
/// expensive direct attempt instead of a clean `ConvergenceFailure`.
pub const DIRECT_FALLBACK_MAX_NODES: usize = 50_000;

/// Refinement budget for [`ResolvedAcousticModel::direct_solve_fallback`].
const MAX_FALLBACK_REFINEMENT_STEPS: usize = 4;

/// Classical preconditioner used by the resolved-acoustics FGMRES path.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub enum ResolvedPreconditionerKind {
    /// Zero-fill incomplete LU retained for compatibility and learned-ILU
    /// experiments.
    Ilu0,
    /// Pivoted threshold ILU with tighter dropping and additional fill.
    #[default]
    Ilut,
    /// AMG V/F-cycle built for a complex shifted-Laplacian proxy.
    ShiftedLaplacianAmg,
    /// Direct below `direct_solve_max_nodes`, then ILUT until the configured
    /// measured AMG crossover, and shifted-Laplacian AMG above it.
    Auto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedPreconditionerReuse {
    FreshBuild,
    NumericRefactor,
    FullReuse,
    Direct,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedPreconditionerFallback {
    pub source: ResolvedPreconditionerKind,
    pub destination: ResolvedPreconditionerKind,
    pub reason: String,
}

/// Geometry, medium, loss, and mesh controls for the resolved acoustic model.
#[derive(Debug, Clone)]
pub struct ResolvedAcousticConfig {
    pub fluid: FluidProperties,
    pub diaphragm_radius_m: f64,
    pub enclosure_radius_m: f64,
    pub cavity_depth_m: f64,
    pub baffle_radius_m: f64,
    pub baffle_half_thickness_m: f64,
    pub port_inner_radius_m: f64,
    pub port_outer_radius_m: f64,
    pub port_end_correction_m: f64,
    pub radial_pml_start_m: f64,
    pub radial_pml_thickness_m: f64,
    pub upper_pml_start_m: f64,
    pub upper_pml_thickness_m: f64,
    pub lower_pml_start_m: f64,
    pub lower_pml_thickness_m: f64,
    pub cells_per_physical_interval: usize,
    pub pml_cells: usize,
    /// Dimensionless peak coordinate stretch. The dimensional PML profile is
    /// scaled by wavenumber so the assembled coefficients remain affine.
    pub pml_sigma_max: f64,
    /// Classical iterative preconditioner. Tight pivoted ILUT is the default
    /// because ILU(0) stalls on the production PML systems.
    pub iterative_preconditioner: ResolvedPreconditionerKind,
    pub ilut_drop_tolerance: f64,
    pub ilut_fill_factor: usize,
    pub ilut_pivot_tolerance: f64,
    pub ilut_pivoting: bool,
    /// Permit exact reuse of an in-band numeric factor/hierarchy.
    pub preconditioner_full_reuse: bool,
    /// Attempt an ILUT numeric update when exact reuse is unavailable.
    pub preconditioner_numeric_refactorization: bool,
    /// Largest relative frequency step that may reuse an existing factor.
    pub preconditioner_reuse_band: f64,
    /// Use sparse direct factorization at or below this node count.
    pub direct_solve_max_nodes: usize,
    /// FGMRES restart window (default 50). Safe at this size because the
    /// ILUT/AMG preconditioner holds iteration counts low (<=28 measured);
    /// unpreconditioned restart-50 was observed to stall where restart-200
    /// converged (SIM-PERF-20261013.md section 2, gap 4), so do not weaken
    /// the preconditioner without raising this.
    pub fgmres_restart: usize,
    pub fgmres_tolerance: f64,
    pub fgmres_max_iterations: usize,
    /// Complex coefficient multiplying the Helmholtz mass pair in the AMG
    /// proxy `K - shift * k² M`.
    pub shifted_mass_coefficient: Complex64,
    pub amg_config: AmgConfig,
    /// Maximum residual ratio accepted from an optimally scaled one-step AMG
    /// action on the assembled right-hand side. Larger values trigger ILUT
    /// before entering FGMRES.
    pub shifted_amg_max_probe_residual_ratio: f64,
    /// Evidence-qualified node count where Auto may switch from ILUT to AMG.
    /// `usize::MAX` disables the unqualified crossover.
    pub auto_amg_crossover_nodes: usize,
    /// Optional locally reacting cavity-wall specific impedance `p/u_n`.
    pub cavity_wall_specific_impedance_pa_s_per_m: Option<Complex64>,
    /// Treat the enclosure shell (side wall at `enclosure_radius_m` and the
    /// cavity bottom) as solid, making the rear cavity a closed box.
    ///
    /// When `true` (default) the cells beside and behind the enclosure
    /// become solid and the resulting single-owner cavity faces are marked
    /// `CAVITY_WALL` (rigid, or Robin when
    /// `cavity_wall_specific_impedance_pa_s_per_m` is set): the enclosure is
    /// a real box whose only vent is the port, which is the physically
    /// meaningful reading of the geometry. When `false` (legacy opt-out) the
    /// cavity is only a labeled pocket of the continuous exterior air: the
    /// mesh generator marks walls on outer (single-owner) edges only, so the
    /// cavity side and bottom stay open and the enclosure applies no
    /// boxed-compliance load.
    pub seal_enclosure_walls: bool,
    pub far_field_distance_m: f64,
    pub far_field_angle_degrees: f64,
}

impl Default for ResolvedAcousticConfig {
    fn default() -> Self {
        Self {
            fluid: FluidProperties::air_20c(),
            diaphragm_radius_m: 40.0e-3,
            enclosure_radius_m: 80.0e-3,
            cavity_depth_m: 100.0e-3,
            baffle_radius_m: 100.0e-3,
            baffle_half_thickness_m: 4.0e-3,
            port_inner_radius_m: 55.0e-3,
            port_outer_radius_m: 60.0e-3,
            // Annular slit: use the hydraulic half-width scale, not the much
            // larger circular-pipe correction based on outer radius.
            port_end_correction_m: 5.0e-3,
            radial_pml_start_m: 160.0e-3,
            radial_pml_thickness_m: 60.0e-3,
            upper_pml_start_m: 160.0e-3,
            upper_pml_thickness_m: 60.0e-3,
            lower_pml_start_m: -160.0e-3,
            lower_pml_thickness_m: 60.0e-3,
            cells_per_physical_interval: 2,
            pml_cells: 2,
            pml_sigma_max: 18.0,
            iterative_preconditioner: ResolvedPreconditionerKind::Ilut,
            ilut_drop_tolerance: 1.0e-5,
            ilut_fill_factor: 20,
            ilut_pivot_tolerance: 1.0e-10,
            ilut_pivoting: true,
            preconditioner_full_reuse: true,
            preconditioner_numeric_refactorization: false,
            preconditioner_reuse_band: 0.2,
            direct_solve_max_nodes: 1_000,
            fgmres_restart: 50,
            fgmres_tolerance: 1.0e-10,
            fgmres_max_iterations: 1_000,
            shifted_mass_coefficient: Complex64::new(1.0, 0.5),
            amg_config: AmgConfig::for_physics(AmgPhysicsProfile::HelmholtzShiftedLaplacian),
            shifted_amg_max_probe_residual_ratio: 0.95,
            auto_amg_crossover_nodes: usize::MAX,
            cavity_wall_specific_impedance_pa_s_per_m: None,
            seal_enclosure_walls: true,
            far_field_distance_m: 1.0,
            far_field_angle_degrees: 0.0,
        }
    }
}

impl ResolvedAcousticConfig {
    fn ilut_config(&self) -> IlutConfig {
        IlutConfig {
            drop_tolerance: self.ilut_drop_tolerance,
            fill_factor: self.ilut_fill_factor,
            pivot_tolerance: self.ilut_pivot_tolerance,
            pivoting: self.ilut_pivoting,
        }
    }

    fn active_preconditioner(&self, node_count: usize) -> ResolvedPreconditionerKind {
        match self.iterative_preconditioner {
            ResolvedPreconditionerKind::Auto => {
                if node_count >= self.auto_amg_crossover_nodes {
                    ResolvedPreconditionerKind::ShiftedLaplacianAmg
                } else {
                    ResolvedPreconditionerKind::Ilut
                }
            }
            kind => kind,
        }
    }

    /// Size the dimensionless PML stretch from the lowest frequency that the
    /// model must resolve. Higher frequencies then receive stronger physical
    /// attenuation while retaining the same affine coordinate stretch.
    pub fn configure_pml_for_sweep(
        &mut self,
        lowest_frequency_hz: f64,
        target_reflection: f64,
    ) -> Result<(), ResolvedSpeakerAcousticError> {
        if !lowest_frequency_hz.is_finite()
            || lowest_frequency_hz <= 0.0
            || !target_reflection.is_finite()
            || !(0.0..1.0).contains(&target_reflection)
        {
            return Err(ResolvedSpeakerAcousticError::InvalidConfiguration(
                "PML reference frequency must be positive and reflection must lie in (0, 1)",
            ));
        }
        let minimum_thickness = self
            .radial_pml_thickness_m
            .min(self.upper_pml_thickness_m)
            .min(self.lower_pml_thickness_m);
        let wavenumber = 2.0 * PI * lowest_frequency_hz / self.fluid.speed_of_sound;
        self.pml_sigma_max = optimal_sigma_max(2, minimum_thickness, wavenumber, target_reflection);
        Ok(())
    }

    pub fn validate(&self) -> Result<(), ResolvedSpeakerAcousticError> {
        let positive = [
            self.fluid.density,
            self.fluid.speed_of_sound,
            self.fluid.dynamic_viscosity,
            self.fluid.thermal_conductivity,
            self.fluid.specific_heat_cp,
            self.fluid.ratio_of_specific_heats,
            self.fluid.ambient_pressure,
            self.diaphragm_radius_m,
            self.enclosure_radius_m,
            self.cavity_depth_m,
            self.baffle_radius_m,
            self.baffle_half_thickness_m,
            self.port_inner_radius_m,
            self.port_outer_radius_m,
            self.radial_pml_start_m,
            self.radial_pml_thickness_m,
            self.upper_pml_start_m,
            self.upper_pml_thickness_m,
            -self.lower_pml_start_m,
            self.lower_pml_thickness_m,
            self.pml_sigma_max,
            self.far_field_distance_m,
        ];
        if positive
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(ResolvedSpeakerAcousticError::InvalidConfiguration(
                "physical dimensions and medium properties must be finite and positive",
            ));
        }
        if !self.port_end_correction_m.is_finite() || self.port_end_correction_m < 0.0 {
            return Err(ResolvedSpeakerAcousticError::InvalidConfiguration(
                "port end correction must be finite and non-negative",
            ));
        }
        if self.cells_per_physical_interval == 0 || self.pml_cells == 0 {
            return Err(ResolvedSpeakerAcousticError::InvalidConfiguration(
                "every mesh interval needs at least one cell",
            ));
        }
        if !self.preconditioner_reuse_band.is_finite()
            || !(0.0..=1.0).contains(&self.preconditioner_reuse_band)
        {
            return Err(ResolvedSpeakerAcousticError::InvalidConfiguration(
                "the ILU reuse band must lie in [0, 1]",
            ));
        }
        if !self.ilut_drop_tolerance.is_finite()
            || self.ilut_drop_tolerance < 0.0
            || self.ilut_fill_factor == 0
            || !self.ilut_pivot_tolerance.is_finite()
            || self.ilut_pivot_tolerance <= 0.0
        {
            return Err(ResolvedSpeakerAcousticError::InvalidConfiguration(
                "ILUT drop/pivot tolerances must be finite and fill/pivot values positive",
            ));
        }
        if self.fgmres_restart == 0
            || self.fgmres_max_iterations == 0
            || !self.fgmres_tolerance.is_finite()
            || self.fgmres_tolerance <= 0.0
            || self.fgmres_tolerance >= 1.0
        {
            return Err(ResolvedSpeakerAcousticError::InvalidConfiguration(
                "FGMRES restart/iteration limits must be positive and tolerance must lie in (0, 1)",
            ));
        }
        if !complex_is_finite(self.shifted_mass_coefficient)
            || self.shifted_mass_coefficient.re <= 0.0
            || self.shifted_mass_coefficient.im < 0.0
        {
            return Err(ResolvedSpeakerAcousticError::InvalidConfiguration(
                "shifted mass coefficient must have positive real and non-negative imaginary parts",
            ));
        }
        self.amg_config.validate().map_err(|_| {
            ResolvedSpeakerAcousticError::InvalidConfiguration("invalid AMG configuration")
        })?;
        if !self.shifted_amg_max_probe_residual_ratio.is_finite()
            || !(0.0..1.0).contains(&self.shifted_amg_max_probe_residual_ratio)
        {
            return Err(ResolvedSpeakerAcousticError::InvalidConfiguration(
                "shifted AMG probe residual ratio must lie in (0, 1)",
            ));
        }
        if !(self.diaphragm_radius_m < self.port_inner_radius_m
            && self.port_inner_radius_m < self.port_outer_radius_m
            && self.port_outer_radius_m < self.enclosure_radius_m
            && self.enclosure_radius_m <= self.baffle_radius_m
            && self.baffle_radius_m < self.radial_pml_start_m)
        {
            return Err(ResolvedSpeakerAcousticError::InvalidConfiguration(
                "radial dimensions must order diaphragm, port, enclosure, baffle, and PML",
            ));
        }
        let cavity_bottom = -self.cavity_depth_m - self.baffle_half_thickness_m;
        if !(self.lower_pml_start_m < cavity_bottom
            && self.upper_pml_start_m > self.baffle_half_thickness_m)
        {
            return Err(ResolvedSpeakerAcousticError::InvalidConfiguration(
                "the physical exterior must surround the complete enclosure",
            ));
        }
        if !self.far_field_angle_degrees.is_finite()
            || !(0.0..=180.0).contains(&self.far_field_angle_degrees)
        {
            return Err(ResolvedSpeakerAcousticError::InvalidConfiguration(
                "far-field polar angle must lie in [0, 180] degrees",
            ));
        }
        if let Some(impedance) = self.cavity_wall_specific_impedance_pa_s_per_m
            && (!complex_is_finite(impedance) || impedance.norm() == 0.0 || impedance.re < 0.0)
        {
            return Err(ResolvedSpeakerAcousticError::InvalidConfiguration(
                "cavity-wall impedance must be finite, nonzero, and passive",
            ));
        }
        Ok(())
    }

    pub fn port_area_m2(&self) -> f64 {
        PI * (self.port_outer_radius_m.powi(2) - self.port_inner_radius_m.powi(2))
    }

    pub fn cavity_volume_m3(&self) -> f64 {
        PI * self.enclosure_radius_m.powi(2) * self.cavity_depth_m
    }

    pub fn effective_port_length_m(&self) -> f64 {
        2.0 * self.baffle_half_thickness_m + self.port_end_correction_m
    }

    pub fn compact_helmholtz_frequency_hz(&self) -> f64 {
        helmholtz_resonance_frequency_hz(
            self.fluid.speed_of_sound,
            self.cavity_volume_m3(),
            self.port_area_m2(),
            self.effective_port_length_m(),
        )
    }
}

pub struct ResolvedAcousticModel {
    pub config: ResolvedAcousticConfig,
    pub mesh: Mesh,
    affine_assembler: PrebakedAssembler<4>,
    dirichlet_nodes: HashSet<usize>,
}

#[derive(Debug, Clone)]
pub struct ResolvedAcousticSolution {
    pub frequency_hz: f64,
    pub diaphragm_velocity_rms_m_per_s: Complex64,
    pub rear_diaphragm_velocity_rms_m_per_s: Complex64,
    pub pressure_rms_pa: Vec<Complex64>,
    pub cavity_pressure_rms_pa: Complex64,
    pub port_volume_velocity_rms_m3_per_s: Complex64,
    pub far_field_pressure_rms_pa: Complex64,
    pub far_field_spl_db: f64,
    pub driver_input_power_w: f64,
    pub port_thermal_loss_w: f64,
    pub cavity_wall_loss_w: f64,
    pub radiated_power_w: f64,
    pub power_balance_residual_w: f64,
    pub solve_diagnostics: ResolvedSolveDiagnostics,
}

/// A checked, native-DOF linear system for one resolved frequency point.
///
/// Hybrid callers use this value to evaluate candidate guesses against the
/// exact fine operator before returning to the unchanged classical solve.
#[derive(Debug, Clone)]
pub struct ResolvedLinearSystem {
    pub frequency_hz: f64,
    pub front_velocity_rms_m_per_s: Complex64,
    pub rear_velocity_rms_m_per_s: Complex64,
    pub matrix: CsrMatrix<Complex64>,
    pub rhs: Vec<Complex64>,
    pub mesh_fingerprint: String,
    pub mesh_layout_fingerprint: String,
    pub assembly_time_seconds: f64,
}

impl ResolvedLinearSystem {
    fn validate(&self) -> Result<(), ResolvedSpeakerAcousticError> {
        if !self.frequency_hz.is_finite()
            || self.frequency_hz <= 0.0
            || !complex_is_finite(self.front_velocity_rms_m_per_s)
            || !complex_is_finite(self.rear_velocity_rms_m_per_s)
            || self.matrix.num_rows != self.matrix.num_cols
            || self.rhs.len() != self.matrix.num_rows
            || self.matrix.validate_structure().is_err()
            || self
                .matrix
                .values
                .iter()
                .any(|value| !complex_is_finite(*value))
            || self.rhs.iter().any(|value| !complex_is_finite(*value))
        {
            return Err(ResolvedSpeakerAcousticError::InvalidLinearSystem);
        }
        Ok(())
    }

    pub fn exact_relative_residual(
        &self,
        candidate: &[Complex64],
    ) -> Result<f64, ResolvedSpeakerAcousticError> {
        self.validate()?;
        checked_exact_relative_residual(&self.matrix, candidate, &self.rhs)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedGuessKind {
    Zero,
    Continuation,
    External,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedCandidateResiduals {
    pub zero: f64,
    pub continuation: Option<f64>,
    pub external: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedSolveDiagnostics {
    pub iterations: usize,
    pub relative_residual: f64,
    /// Exact residual reached by FGMRES before any sparse-direct fallback.
    /// `None` means the solve selected the direct path before entering FGMRES.
    pub iterative_relative_residual: Option<f64>,
    pub initial_guess_reused: bool,
    pub preconditioner_rebuilt: bool,
    pub direct_fallback_used: bool,
    pub selected_guess_kind: ResolvedGuessKind,
    pub candidate_residuals: ResolvedCandidateResiduals,
    pub ml_rejection_reason: Option<String>,
    pub coarse_time_seconds: f64,
    pub ml_time_seconds: f64,
    pub fine_setup_time_seconds: f64,
    pub fine_iteration_time_seconds: f64,
    pub learned_preconditioner_requested: bool,
    pub learned_preconditioner_active: bool,
    pub learned_preconditioner_probe_ratio: Option<f64>,
    pub learned_preconditioner_rejection_reason: Option<String>,
    pub requested_preconditioner: ResolvedPreconditionerKind,
    pub active_preconditioner: Option<ResolvedPreconditionerKind>,
    pub preconditioner_fallbacks: Vec<ResolvedPreconditionerFallback>,
    pub preconditioner_reuse: ResolvedPreconditionerReuse,
    pub preconditioner_setup_count: usize,
    pub preconditioner_apply_count: u64,
    pub preconditioner_setup_time_seconds: f64,
    pub preconditioner_apply_time_seconds: f64,
    pub preconditioner_nonzeros: usize,
    pub preconditioner_fill_ratio: f64,
    pub preconditioner_estimated_storage_bytes: usize,
    pub preconditioner_regularized_pivots: usize,
}

/// Diagnostics supplied by an opt-in preconditioner factory.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResolvedPreconditionerDiagnostics {
    pub learned_requested: bool,
    pub learned_active: bool,
    pub probe_residual_ratio: Option<f64>,
    pub rejection_reason: Option<String>,
}

/// Result of adapting the continuation-owned ILU into an alternate
/// preconditioner. The baseline ILU remains owned by the continuation, so a
/// learned wrapper can be rebuilt per frequency without forfeiting ILU reuse.
pub struct ResolvedPreconditionerBuild<P> {
    pub preconditioner: P,
    pub diagnostics: ResolvedPreconditionerDiagnostics,
}

#[derive(Debug, Default)]
struct ResolvedApplyTelemetry {
    count: AtomicU64,
    total_ns: AtomicU64,
}

struct ResolvedMeasuredPreconditioner<'a, P> {
    inner: &'a P,
    telemetry: &'a ResolvedApplyTelemetry,
}

impl<P: Preconditioner<Complex64>> Preconditioner<Complex64>
    for ResolvedMeasuredPreconditioner<'_, P>
{
    fn apply_into(&self, residual: &[Complex64], output: &mut [Complex64]) {
        let started = Instant::now();
        self.inner.apply_into(residual, output);
        self.telemetry.count.fetch_add(1, Ordering::Relaxed);
        self.telemetry.total_ns.fetch_add(
            started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
    }
}

#[derive(Clone)]
enum ResolvedCachedPreconditioner {
    Ilu0 {
        factor: IluPreconditioner<Complex64>,
        telemetry: Arc<ResolvedApplyTelemetry>,
    },
    Ilut {
        factor: IlutPreconditioner<Complex64>,
        telemetry: Arc<ResolvedApplyTelemetry>,
    },
    ShiftedLaplacianAmg {
        factor: Arc<AmgPreconditioner<Complex64>>,
        telemetry: Arc<ResolvedApplyTelemetry>,
    },
}

impl ResolvedCachedPreconditioner {
    fn build(
        kind: ResolvedPreconditionerKind,
        matrix: &CsrMatrix<Complex64>,
        shifted_matrix: Option<&CsrMatrix<Complex64>>,
        config: &ResolvedAcousticConfig,
    ) -> Result<(Self, Vec<ResolvedPreconditionerFallback>), LinearSolverError> {
        let telemetry = || Arc::new(ResolvedApplyTelemetry::default());
        match kind {
            ResolvedPreconditionerKind::Ilu0 => Ok((
                Self::Ilu0 {
                    factor: IluPreconditioner::from_csr(matrix),
                    telemetry: telemetry(),
                },
                Vec::new(),
            )),
            ResolvedPreconditionerKind::Ilut => {
                let ilut_config = config.ilut_config();
                match IlutPreconditioner::try_from_csr_with_config(matrix, &ilut_config) {
                    Ok(factor) => Ok((
                        Self::Ilut {
                            factor,
                            telemetry: telemetry(),
                        },
                        Vec::new(),
                    )),
                    Err(error) => Ok((
                        Self::Ilu0 {
                            factor: IluPreconditioner::from_csr(matrix),
                            telemetry: telemetry(),
                        },
                        vec![ResolvedPreconditionerFallback {
                            source: ResolvedPreconditionerKind::Ilut,
                            destination: ResolvedPreconditionerKind::Ilu0,
                            reason: error.to_string(),
                        }],
                    )),
                }
            }
            ResolvedPreconditionerKind::ShiftedLaplacianAmg => {
                let shifted = shifted_matrix.ok_or_else(|| LinearSolverError::OperationFailed {
                    operation: "shifted-Laplacian AMG setup",
                    reason: "shifted operator was not supplied".into(),
                })?;
                match AmgPreconditioner::try_from_csr(shifted, config.amg_config.clone()) {
                    Ok(factor) => Ok((
                        Self::ShiftedLaplacianAmg {
                            factor: Arc::new(factor),
                            telemetry: telemetry(),
                        },
                        Vec::new(),
                    )),
                    Err(error) => {
                        let (fallback, mut fallbacks) =
                            Self::build(ResolvedPreconditionerKind::Ilut, matrix, None, config)?;
                        fallbacks.insert(
                            0,
                            ResolvedPreconditionerFallback {
                                source: ResolvedPreconditionerKind::ShiftedLaplacianAmg,
                                destination: ResolvedPreconditionerKind::Ilut,
                                reason: error.to_string(),
                            },
                        );
                        Ok((fallback, fallbacks))
                    }
                }
            }
            ResolvedPreconditionerKind::Auto => unreachable!("Auto is resolved before setup"),
        }
    }

    fn build_admitted(
        kind: ResolvedPreconditionerKind,
        matrix: &CsrMatrix<Complex64>,
        shifted_matrix: Option<&CsrMatrix<Complex64>>,
        rhs: &[Complex64],
        config: &ResolvedAcousticConfig,
    ) -> Result<(Self, Vec<ResolvedPreconditionerFallback>), LinearSolverError> {
        let (mut preconditioner, mut fallbacks) =
            Self::build(kind, matrix, shifted_matrix, config)?;
        if kind == ResolvedPreconditionerKind::ShiftedLaplacianAmg
            && preconditioner.kind() == ResolvedPreconditionerKind::ShiftedLaplacianAmg
        {
            let ratio = optimally_scaled_preconditioner_probe_ratio(matrix, &preconditioner, rhs);
            if !ratio.is_finite() || ratio > config.shifted_amg_max_probe_residual_ratio {
                let (fallback, mut nested) =
                    Self::build(ResolvedPreconditionerKind::Ilut, matrix, None, config)?;
                fallbacks.push(ResolvedPreconditionerFallback {
                    source: ResolvedPreconditionerKind::ShiftedLaplacianAmg,
                    destination: ResolvedPreconditionerKind::Ilut,
                    reason: format!(
                        "AMG action probe residual ratio {ratio:.6e} exceeds limit {:.6e}",
                        config.shifted_amg_max_probe_residual_ratio
                    ),
                });
                fallbacks.append(&mut nested);
                preconditioner = fallback;
            }
        }
        Ok((preconditioner, fallbacks))
    }

    fn kind(&self) -> ResolvedPreconditionerKind {
        match self {
            Self::Ilu0 { .. } => ResolvedPreconditionerKind::Ilu0,
            Self::Ilut { .. } => ResolvedPreconditionerKind::Ilut,
            Self::ShiftedLaplacianAmg { .. } => ResolvedPreconditionerKind::ShiftedLaplacianAmg,
        }
    }

    fn telemetry(&self) -> &Arc<ResolvedApplyTelemetry> {
        match self {
            Self::Ilu0 { telemetry, .. }
            | Self::Ilut { telemetry, .. }
            | Self::ShiftedLaplacianAmg { telemetry, .. } => telemetry,
        }
    }

    fn storage_diagnostics(&self, matrix_nnz: usize) -> (usize, f64, usize, usize) {
        let (nonzeros, bytes, regularized) = match self {
            Self::Ilu0 { .. } => (matrix_nnz, matrix_nnz * 20, 0),
            Self::Ilut { factor, .. } => {
                let diagnostics = factor.diagnostics();
                (
                    diagnostics.total_nonzeros,
                    diagnostics.estimated_storage_bytes,
                    diagnostics.regularized_pivots,
                )
            }
            Self::ShiftedLaplacianAmg { factor, .. } => {
                let diagnostics = factor.diagnostics();
                let bytes = diagnostics.total_nnz * (std::mem::size_of::<Complex64>() + 4)
                    + diagnostics.level_dofs.iter().sum::<usize>() * std::mem::size_of::<usize>();
                (diagnostics.total_nnz, bytes, 0)
            }
        };
        (
            nonzeros,
            nonzeros as f64 / matrix_nnz.max(1) as f64,
            bytes,
            regularized,
        )
    }
}

impl Preconditioner<Complex64> for ResolvedCachedPreconditioner {
    fn apply_into(&self, residual: &[Complex64], output: &mut [Complex64]) {
        let started = Instant::now();
        match self {
            Self::Ilu0 { factor, .. } => factor.apply_into(residual, output),
            Self::Ilut { factor, .. } => factor.apply_into(residual, output),
            Self::ShiftedLaplacianAmg { factor, .. } => factor.apply_into(residual, output),
        }
        let telemetry = self.telemetry();
        telemetry.count.fetch_add(1, Ordering::Relaxed);
        telemetry.total_ns.fetch_add(
            started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
    }
}

/// Stateful frequency-continuation solver for one resolved acoustic model.
///
/// Threads the previous pressure as the initial guess and shares an ILU
/// factorisation across frequencies inside
/// [`ResolvedAcousticConfig::preconditioner_reuse_band`]. Create one per
/// sweep and call [`Self::solve_frequency_with_drive`] for each point;
/// per-frequency drives are supported, unlike
/// [`ResolvedAcousticModel::solve_sweep`] which applies a uniform velocity.
pub struct ResolvedFrequencyContinuation<'a> {
    model: &'a ResolvedAcousticModel,
    previous_pressure: Option<Vec<Complex64>>,
    preconditioner_frequency_hz: Option<f64>,
    preconditioner: Option<ResolvedCachedPreconditioner>,
    preconditioner_matrix: Option<CsrMatrix<Complex64>>,
}

impl<'a> ResolvedFrequencyContinuation<'a> {
    pub fn new(model: &'a ResolvedAcousticModel) -> Self {
        Self {
            model,
            previous_pressure: None,
            preconditioner_frequency_hz: None,
            preconditioner: None,
            preconditioner_matrix: None,
        }
    }

    /// Solve one frequency with an equal front/rear diaphragm drive, reusing
    /// the warm start and any in-band ILU factorisation from earlier calls.
    pub fn solve_frequency_with_drive(
        &mut self,
        frequency_hz: f64,
        diaphragm_velocity_rms_m_per_s: Complex64,
    ) -> Result<ResolvedAcousticSolution, ResolvedSpeakerAcousticError> {
        self.model
            .solve_frequency_with_surface_velocities_and_continuation(
                frequency_hz,
                diaphragm_velocity_rms_m_per_s,
                diaphragm_velocity_rms_m_per_s,
                Some(self),
            )
    }

    /// Solve one frequency while considering an externally supplied native
    /// nodal pressure alongside the ordinary frequency-continuation guess.
    pub fn solve_frequency_with_external_guess(
        &mut self,
        frequency_hz: f64,
        diaphragm_velocity_rms_m_per_s: Complex64,
        external_guess: Option<&[Complex64]>,
    ) -> Result<ResolvedAcousticSolution, ResolvedSpeakerAcousticError> {
        self.model
            .solve_frequency_with_surface_velocities_and_guesses(
                frequency_hz,
                diaphragm_velocity_rms_m_per_s,
                diaphragm_velocity_rms_m_per_s,
                Some(self),
                external_guess,
            )
    }

    /// The previous certified pressure available to exact-residual candidate
    /// ranking. The slice is read-only and remains owned by the continuation.
    pub fn previous_pressure(&self) -> Option<&[Complex64]> {
        self.previous_pressure.as_deref()
    }

    /// Solve an already assembled system while preserving continuation and
    /// ILU reuse. This avoids reassembly after a hybrid residual gate.
    pub fn solve_assembled_with_external_guess(
        &mut self,
        system: &ResolvedLinearSystem,
        external_guess: Option<&[Complex64]>,
    ) -> Result<ResolvedAcousticSolution, ResolvedSpeakerAcousticError> {
        self.model
            .solve_assembled_with_guesses(system, Some(self), external_guess)
    }

    /// Solve with an adapter built from the continuation-owned ILU.
    ///
    /// The factory is invoked after the ordinary ILU rebuild/reuse decision
    /// and again if a stale reused factor must be rebuilt. This preserves the
    /// classical continuation policy while allowing an opt-in learned wrapper
    /// to vary with frequency or geometry.
    pub fn solve_assembled_with_preconditioner_factory<P, F>(
        &mut self,
        system: &ResolvedLinearSystem,
        external_guess: Option<&[Complex64]>,
        factory: F,
    ) -> Result<ResolvedAcousticSolution, ResolvedSpeakerAcousticError>
    where
        P: Preconditioner<Complex64>,
        F: FnMut(
            &CsrMatrix<Complex64>,
            IluPreconditioner<Complex64>,
        ) -> Result<ResolvedPreconditionerBuild<P>, LinearSolverError>,
    {
        self.model
            .solve_assembled_with_solver(system, Some(self), external_guess, factory)
    }

    /// ILUT-base counterpart of the learned ILU(0) compatibility entry point.
    pub fn solve_assembled_with_ilut_preconditioner_factory<P, F>(
        &mut self,
        system: &ResolvedLinearSystem,
        external_guess: Option<&[Complex64]>,
        factory: F,
    ) -> Result<ResolvedAcousticSolution, ResolvedSpeakerAcousticError>
    where
        P: Preconditioner<Complex64>,
        F: FnMut(
            &CsrMatrix<Complex64>,
            IlutPreconditioner<Complex64>,
        ) -> Result<ResolvedPreconditionerBuild<P>, LinearSolverError>,
    {
        let mut factory = factory;
        self.model.solve_assembled_with_cached_solver(
            system,
            Some(self),
            external_guess,
            ResolvedPreconditionerKind::Ilut,
            move |matrix, preconditioner| {
                let ResolvedCachedPreconditioner::Ilut { factor, .. } = preconditioner else {
                    return Err(LinearSolverError::OperationFailed {
                        operation: "learned ILUT preconditioner factory",
                        reason: "ILUT setup fell back before the learned wrapper was built".into(),
                    });
                };
                factory(matrix, factor)
            },
        )
    }
}

impl ResolvedAcousticModel {
    pub fn build(config: ResolvedAcousticConfig) -> Result<Self, ResolvedSpeakerAcousticError> {
        config.validate()?;
        let mesh = build_conforming_mesh(&config)?;
        let reference_pml = positive_iomega_axisymmetric_exterior_pml(
            config.radial_pml_start_m,
            config.radial_pml_thickness_m,
            config.upper_pml_start_m,
            config.upper_pml_thickness_m,
            config.lower_pml_start_m,
            config.lower_pml_thickness_m,
            1.0,
            config.pml_sigma_max,
        );
        let mut affine_assembler =
            build_axisymmetric_pml_assembler(&mesh, reference_pml, 4, &[PORT_WALL, CAVITY_WALL]);
        let dirichlet_nodes: HashSet<_> = mesh
            .boundaries
            .iter()
            .filter(|boundary| boundary.marker == OUTER_PML_TRUNCATION)
            .flat_map(|boundary| boundary.nodes.iter().copied())
            .collect();
        affine_assembler.apply_dirichlet_nodes(&dirichlet_nodes);
        Ok(Self {
            config,
            mesh,
            affine_assembler,
            dirichlet_nodes,
        })
    }

    pub fn solve_frequency(
        &self,
        frequency_hz: f64,
        diaphragm_velocity_rms_m_per_s: Complex64,
    ) -> Result<ResolvedAcousticSolution, ResolvedSpeakerAcousticError> {
        self.solve_frequency_with_surface_velocities(
            frequency_hz,
            diaphragm_velocity_rms_m_per_s,
            diaphragm_velocity_rms_m_per_s,
        )
    }

    /// Solve with independently prescribed axial velocities on the front and
    /// rear diaphragm faces. A physical zero-thickness diaphragm uses equal
    /// values; the independent form supports one-sided radiation benchmarks.
    pub fn solve_frequency_with_surface_velocities(
        &self,
        frequency_hz: f64,
        front_velocity_rms_m_per_s: Complex64,
        rear_velocity_rms_m_per_s: Complex64,
    ) -> Result<ResolvedAcousticSolution, ResolvedSpeakerAcousticError> {
        self.solve_frequency_with_surface_velocities_and_continuation(
            frequency_hz,
            front_velocity_rms_m_per_s,
            rear_velocity_rms_m_per_s,
            None,
        )
    }

    /// Solve one frequency with an optional native-DOF external guess.
    /// Length and finiteness are checked before any solver setup is done.
    pub fn solve_frequency_with_external_guess(
        &self,
        frequency_hz: f64,
        diaphragm_velocity_rms_m_per_s: Complex64,
        external_guess: Option<&[Complex64]>,
    ) -> Result<ResolvedAcousticSolution, ResolvedSpeakerAcousticError> {
        self.solve_frequency_with_surface_velocities_and_guesses(
            frequency_hz,
            diaphragm_velocity_rms_m_per_s,
            diaphragm_velocity_rms_m_per_s,
            None,
            external_guess,
        )
    }

    fn solve_frequency_with_surface_velocities_and_continuation<'a>(
        &'a self,
        frequency_hz: f64,
        front_velocity_rms_m_per_s: Complex64,
        rear_velocity_rms_m_per_s: Complex64,
        continuation: Option<&mut ResolvedFrequencyContinuation<'a>>,
    ) -> Result<ResolvedAcousticSolution, ResolvedSpeakerAcousticError> {
        self.solve_frequency_with_surface_velocities_and_guesses(
            frequency_hz,
            front_velocity_rms_m_per_s,
            rear_velocity_rms_m_per_s,
            continuation,
            None,
        )
    }

    fn solve_frequency_with_surface_velocities_and_guesses<'a>(
        &'a self,
        frequency_hz: f64,
        front_velocity_rms_m_per_s: Complex64,
        rear_velocity_rms_m_per_s: Complex64,
        continuation: Option<&mut ResolvedFrequencyContinuation<'a>>,
        external_guess: Option<&[Complex64]>,
    ) -> Result<ResolvedAcousticSolution, ResolvedSpeakerAcousticError> {
        let system = self.assemble_frequency_system_with_surface_velocities(
            frequency_hz,
            front_velocity_rms_m_per_s,
            rear_velocity_rms_m_per_s,
        )?;
        self.solve_assembled_with_guesses(&system, continuation, external_guess)
    }

    /// Assemble the exact native system for an equal front/rear diaphragm
    /// drive without solving it.
    pub fn assemble_frequency_system(
        &self,
        frequency_hz: f64,
        diaphragm_velocity_rms_m_per_s: Complex64,
    ) -> Result<ResolvedLinearSystem, ResolvedSpeakerAcousticError> {
        self.assemble_frequency_system_with_surface_velocities(
            frequency_hz,
            diaphragm_velocity_rms_m_per_s,
            diaphragm_velocity_rms_m_per_s,
        )
    }

    pub fn assemble_frequency_system_with_surface_velocities(
        &self,
        frequency_hz: f64,
        front_velocity_rms_m_per_s: Complex64,
        rear_velocity_rms_m_per_s: Complex64,
    ) -> Result<ResolvedLinearSystem, ResolvedSpeakerAcousticError> {
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(ResolvedSpeakerAcousticError::InvalidFrequency);
        }
        if !complex_is_finite(front_velocity_rms_m_per_s)
            || !complex_is_finite(rear_velocity_rms_m_per_s)
        {
            return Err(ResolvedSpeakerAcousticError::InvalidDrive);
        }
        let started = Instant::now();
        let omega = 2.0 * PI * frequency_hz;
        let wavenumber = omega / self.config.fluid.speed_of_sound;
        let port_alpha = thermal_wall_admittance(&self.config.fluid, omega).conj();
        let cavity_alpha = self
            .config
            .cavity_wall_specific_impedance_pa_s_per_m
            .map(|impedance| Complex64::new(0.0, omega * self.config.fluid.density) / impedance);
        let matrix = self.assemble_frequency_matrix(wavenumber, port_alpha, cavity_alpha);

        let mut rhs = vec![Complex64::default(); self.mesh.num_nodes()];
        let i_omega_rho = Complex64::new(0.0, omega * self.config.fluid.density);
        add_axisymmetric_neumann(
            &mut rhs,
            &self.mesh,
            FRONT_DIAPHRAGM,
            i_omega_rho * front_velocity_rms_m_per_s,
        );
        add_axisymmetric_neumann(
            &mut rhs,
            &self.mesh,
            REAR_DIAPHRAGM,
            -i_omega_rho * rear_velocity_rms_m_per_s,
        );
        for &node in &self.dirichlet_nodes {
            rhs[node] = Complex64::default();
        }
        Ok(ResolvedLinearSystem {
            frequency_hz,
            front_velocity_rms_m_per_s,
            rear_velocity_rms_m_per_s,
            matrix,
            rhs,
            mesh_fingerprint: self.mesh_fingerprint(),
            mesh_layout_fingerprint: self.mesh_layout_fingerprint(),
            assembly_time_seconds: started.elapsed().as_secs_f64(),
        })
    }

    /// Solve a previously assembled system after checking that it belongs to
    /// this mesh. This is the no-reassembly entry point used by hybrid code.
    pub fn solve_assembled_with_external_guess(
        &self,
        system: &ResolvedLinearSystem,
        external_guess: Option<&[Complex64]>,
    ) -> Result<ResolvedAcousticSolution, ResolvedSpeakerAcousticError> {
        self.solve_assembled_with_guesses(system, None, external_guess)
    }

    fn solve_assembled_with_guesses<'a>(
        &'a self,
        system: &ResolvedLinearSystem,
        continuation: Option<&mut ResolvedFrequencyContinuation<'a>>,
        external_guess: Option<&[Complex64]>,
    ) -> Result<ResolvedAcousticSolution, ResolvedSpeakerAcousticError> {
        self.solve_assembled_with_cached_solver(
            system,
            continuation,
            external_guess,
            self.config.iterative_preconditioner,
            |_matrix, preconditioner| {
                Ok(ResolvedPreconditionerBuild {
                    preconditioner,
                    diagnostics: ResolvedPreconditionerDiagnostics::default(),
                })
            },
        )
    }

    fn solve_assembled_with_solver<'a, P, F>(
        &'a self,
        system: &ResolvedLinearSystem,
        continuation: Option<&mut ResolvedFrequencyContinuation<'a>>,
        external_guess: Option<&[Complex64]>,
        preconditioner_factory: F,
    ) -> Result<ResolvedAcousticSolution, ResolvedSpeakerAcousticError>
    where
        P: Preconditioner<Complex64>,
        F: FnMut(
            &CsrMatrix<Complex64>,
            IluPreconditioner<Complex64>,
        ) -> Result<ResolvedPreconditionerBuild<P>, LinearSolverError>,
    {
        let mut preconditioner_factory = preconditioner_factory;
        self.solve_assembled_with_cached_solver(
            system,
            continuation,
            external_guess,
            ResolvedPreconditionerKind::Ilu0,
            move |matrix, preconditioner| {
                let ResolvedCachedPreconditioner::Ilu0 { factor: ilu, .. } = preconditioner else {
                    unreachable!("learned preconditioner factory always requests ILU(0)")
                };
                preconditioner_factory(matrix, ilu)
            },
        )
    }

    fn solve_assembled_with_cached_solver<'a, P, F>(
        &'a self,
        system: &ResolvedLinearSystem,
        continuation: Option<&mut ResolvedFrequencyContinuation<'a>>,
        external_guess: Option<&[Complex64]>,
        preconditioner_kind: ResolvedPreconditionerKind,
        preconditioner_factory: F,
    ) -> Result<ResolvedAcousticSolution, ResolvedSpeakerAcousticError>
    where
        P: Preconditioner<Complex64>,
        F: FnMut(
            &CsrMatrix<Complex64>,
            ResolvedCachedPreconditioner,
        ) -> Result<ResolvedPreconditionerBuild<P>, LinearSolverError>,
    {
        system.validate()?;
        if system.mesh_fingerprint != self.mesh_fingerprint() {
            return Err(ResolvedSpeakerAcousticError::MeshFingerprintMismatch);
        }
        let frequency_hz = system.frequency_hz;
        let front_velocity_rms_m_per_s = system.front_velocity_rms_m_per_s;
        let rear_velocity_rms_m_per_s = system.rear_velocity_rms_m_per_s;
        let matrix = &system.matrix;
        let rhs = &system.rhs;
        let omega = 2.0 * PI * frequency_hz;
        let wavenumber = omega / self.config.fluid.speed_of_sound;
        let port_alpha = thermal_wall_admittance(&self.config.fluid, omega).conj();
        let cavity_alpha = self
            .config
            .cavity_wall_specific_impedance_pa_s_per_m
            .map(|impedance| Complex64::new(0.0, omega * self.config.fluid.density) / impedance);
        let (pressure, solve_diagnostics) = self.solve_pressure_with_factory(
            frequency_hz,
            matrix,
            rhs,
            continuation,
            external_guess,
            preconditioner_kind,
            preconditioner_factory,
        )?;
        if pressure.iter().any(|value| !complex_is_finite(*value)) {
            return Err(ResolvedSpeakerAcousticError::NonFiniteSolution);
        }

        let cavity_pressure = volume_average_pressure(&self.mesh, &pressure, REAR_CAVITY);
        let port_volume_velocity =
            port_volume_velocity(&self.mesh, &pressure, omega, self.config.fluid.density);
        let angle = self.config.far_field_angle_degrees.to_radians();
        let observation = [
            self.config.far_field_distance_m * angle.sin(),
            0.0,
            self.config.far_field_distance_m * angle.cos(),
        ];
        let far_field_pressure = axisymmetric_kirchhoff_far_field(
            &self.mesh,
            PHYSICAL_EXTERIOR,
            ACOUSTIC_PML,
            &pressure,
            wavenumber,
            observation,
        )?;
        let radiated_power = axisymmetric_interface_reaction_power(
            &self.mesh,
            PHYSICAL_EXTERIOR,
            ACOUSTIC_PML,
            &pressure,
            omega,
            self.config.fluid.density,
            self.config.fluid.speed_of_sound,
        )?;
        let front_normal_velocity = -front_velocity_rms_m_per_s;
        let rear_normal_velocity = rear_velocity_rms_m_per_s;
        let outward_driver_power =
            boundary_pressure_integral(&self.mesh, &pressure, FRONT_DIAPHRAGM)
                * front_normal_velocity.conj()
                + boundary_pressure_integral(&self.mesh, &pressure, REAR_DIAPHRAGM)
                    * rear_normal_velocity.conj();
        let driver_input_power = -outward_driver_power.re;
        let port_thermal_loss = robin_loss_power(
            &self.mesh,
            &pressure,
            PORT_WALL,
            port_alpha,
            omega,
            self.config.fluid.density,
        );
        let cavity_wall_loss = cavity_alpha.map_or(0.0, |alpha| {
            robin_loss_power(
                &self.mesh,
                &pressure,
                CAVITY_WALL,
                alpha,
                omega,
                self.config.fluid.density,
            )
        });
        let power_balance_residual =
            driver_input_power - port_thermal_loss - cavity_wall_loss - radiated_power;

        Ok(ResolvedAcousticSolution {
            frequency_hz,
            diaphragm_velocity_rms_m_per_s: front_velocity_rms_m_per_s,
            rear_diaphragm_velocity_rms_m_per_s: rear_velocity_rms_m_per_s,
            pressure_rms_pa: pressure,
            cavity_pressure_rms_pa: cavity_pressure,
            port_volume_velocity_rms_m3_per_s: port_volume_velocity,
            far_field_pressure_rms_pa: far_field_pressure,
            far_field_spl_db: spl_db(far_field_pressure.norm()),
            driver_input_power_w: driver_input_power,
            port_thermal_loss_w: port_thermal_loss,
            cavity_wall_loss_w: cavity_wall_loss,
            radiated_power_w: radiated_power,
            power_balance_residual_w: power_balance_residual,
            solve_diagnostics,
        })
    }

    /// Stable SHA-256 over native coordinates and element connectivity.
    pub fn mesh_fingerprint(&self) -> String {
        let mut digest = Sha256::new();
        digest.update(b"sonium-resolved-acoustic-mesh-v1");
        digest.update((self.mesh.dimension as u64).to_le_bytes());
        digest.update((self.mesh.nodes.len() as u64).to_le_bytes());
        for node in &self.mesh.nodes {
            digest.update(node.x.to_le_bytes());
            digest.update(node.y.to_le_bytes());
            digest.update(node.z.to_le_bytes());
        }
        hash_mesh_connectivity(&mut digest, &self.mesh);
        format!("{:x}", digest.finalize())
    }

    /// Stable topology/order fingerprint shared by deformations of the fixed
    /// four-parameter mesh family.
    pub fn mesh_layout_fingerprint(&self) -> String {
        let mut digest = Sha256::new();
        digest.update(b"sonium-resolved-acoustic-layout-v1");
        digest.update((self.mesh.dimension as u64).to_le_bytes());
        digest.update((self.mesh.nodes.len() as u64).to_le_bytes());
        hash_mesh_connectivity(&mut digest, &self.mesh);
        format!("{:x}", digest.finalize())
    }

    fn assemble_frequency_matrix(
        &self,
        wavenumber: f64,
        port_alpha: Complex64,
        cavity_alpha: Option<Complex64>,
    ) -> CsrMatrix<Complex64> {
        let mut boundary_coefficients = HashMap::from([(PORT_WALL as usize, port_alpha)]);
        if let Some(alpha) = cavity_alpha {
            boundary_coefficients.insert(CAVITY_WALL as usize, alpha);
        }
        let k_squared = wavenumber * wavenumber;
        self.affine_assembler.assemble_complex(
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(-k_squared, 0.0),
                Complex64::new(0.0, -k_squared),
            ],
            &boundary_coefficients,
        )
    }

    fn assemble_shifted_laplacian_matrix(
        &self,
        wavenumber: f64,
        port_alpha: Complex64,
        cavity_alpha: Option<Complex64>,
    ) -> CsrMatrix<Complex64> {
        let mut boundary_coefficients = HashMap::from([(PORT_WALL as usize, port_alpha)]);
        if let Some(alpha) = cavity_alpha {
            boundary_coefficients.insert(CAVITY_WALL as usize, alpha);
        }
        let shifted_mass = -self.config.shifted_mass_coefficient * wavenumber * wavenumber;
        self.affine_assembler.assemble_complex(
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 1.0),
                shifted_mass,
                shifted_mass * Complex64::I,
            ],
            &boundary_coefficients,
        )
    }

    fn solve_pressure_with_factory<'a, P, F>(
        &'a self,
        frequency_hz: f64,
        matrix: &CsrMatrix<Complex64>,
        rhs: &[Complex64],
        continuation: Option<&mut ResolvedFrequencyContinuation<'a>>,
        external_guess: Option<&[Complex64]>,
        preconditioner_kind: ResolvedPreconditionerKind,
        mut preconditioner_factory: F,
    ) -> Result<(Vec<Complex64>, ResolvedSolveDiagnostics), ResolvedSpeakerAcousticError>
    where
        P: Preconditioner<Complex64>,
        F: FnMut(
            &CsrMatrix<Complex64>,
            ResolvedCachedPreconditioner,
        ) -> Result<ResolvedPreconditionerBuild<P>, LinearSolverError>,
    {
        let mut local_continuation;
        let continuation = match continuation {
            Some(continuation) => continuation,
            None => {
                local_continuation = ResolvedFrequencyContinuation::new(self);
                &mut local_continuation
            }
        };
        if let Some(guess) = external_guess {
            validate_external_guess(guess, rhs.len())?;
        }
        let continuation_guess = continuation
            .previous_pressure
            .as_deref()
            .filter(|guess| guess.len() == rhs.len());
        let candidate_residuals = ResolvedCandidateResiduals {
            zero: exact_relative_residual(matrix, &vec![Complex64::default(); rhs.len()], rhs),
            continuation: continuation_guess
                .map(|guess| exact_relative_residual(matrix, guess, rhs)),
            external: external_guess.map(|guess| exact_relative_residual(matrix, guess, rhs)),
        };
        let mut selected_guess_kind = ResolvedGuessKind::Zero;
        let mut selected_residual = candidate_residuals.zero;
        let mut initial_guess = None;
        if let Some((guess, residual)) = continuation_guess.zip(candidate_residuals.continuation)
            && residual < selected_residual
        {
            selected_guess_kind = ResolvedGuessKind::Continuation;
            selected_residual = residual;
            initial_guess = Some(guess);
        }
        if let Some((guess, residual)) = external_guess.zip(candidate_residuals.external)
            && residual < selected_residual
        {
            selected_guess_kind = ResolvedGuessKind::External;
            initial_guess = Some(guess);
        }
        let initial_guess_reused = initial_guess.is_some();
        let solve_started = Instant::now();

        if self.mesh.num_nodes() <= self.config.direct_solve_max_nodes {
            let solver = SolverConfig {
                solver_type: SolverType::Direct,
                ..SolverConfig::default()
            };
            let solved = solve_csr_with_guess(matrix, rhs, initial_guess, &solver)?;
            let relative_residual = exact_relative_residual(matrix, &solved.values, rhs);
            continuation.previous_pressure = Some(solved.values.clone());
            return Ok((
                solved.values,
                ResolvedSolveDiagnostics {
                    iterations: solved.iterations,
                    relative_residual,
                    iterative_relative_residual: None,
                    initial_guess_reused,
                    preconditioner_rebuilt: false,
                    direct_fallback_used: false,
                    selected_guess_kind,
                    candidate_residuals,
                    ml_rejection_reason: None,
                    coarse_time_seconds: 0.0,
                    ml_time_seconds: 0.0,
                    fine_setup_time_seconds: 0.0,
                    fine_iteration_time_seconds: solve_started.elapsed().as_secs_f64(),
                    learned_preconditioner_requested: false,
                    learned_preconditioner_active: false,
                    learned_preconditioner_probe_ratio: None,
                    learned_preconditioner_rejection_reason: None,
                    requested_preconditioner: preconditioner_kind,
                    active_preconditioner: None,
                    preconditioner_fallbacks: Vec::new(),
                    preconditioner_reuse: ResolvedPreconditionerReuse::Direct,
                    preconditioner_setup_count: 0,
                    preconditioner_apply_count: 0,
                    preconditioner_setup_time_seconds: 0.0,
                    preconditioner_apply_time_seconds: 0.0,
                    preconditioner_nonzeros: 0,
                    preconditioner_fill_ratio: 0.0,
                    preconditioner_estimated_storage_bytes: 0,
                    preconditioner_regularized_pivots: 0,
                },
            ));
        }

        let setup_started = Instant::now();
        let active_kind = if preconditioner_kind == ResolvedPreconditionerKind::Auto {
            self.config.active_preconditioner(self.mesh.num_nodes())
        } else {
            preconditioner_kind
        };
        let in_band = continuation
            .preconditioner_frequency_hz
            .is_some_and(|reference| {
                (frequency_hz - reference).abs() / frequency_hz.max(reference)
                    <= self.config.preconditioner_reuse_band
            });
        let same_kind = continuation
            .preconditioner
            .as_ref()
            .is_some_and(|cached| cached.kind() == active_kind);
        let mut fallbacks = Vec::new();
        let mut reuse = ResolvedPreconditionerReuse::FreshBuild;
        let mut setup_count = 0usize;
        if same_kind && in_band && self.config.preconditioner_full_reuse {
            reuse = ResolvedPreconditionerReuse::FullReuse;
        } else if same_kind
            && active_kind == ResolvedPreconditionerKind::Ilut
            && self.config.preconditioner_numeric_refactorization
        {
            let refactor = match (
                continuation.preconditioner.as_mut(),
                continuation.preconditioner_matrix.as_ref(),
            ) {
                (
                    Some(ResolvedCachedPreconditioner::Ilut { factor, .. }),
                    Some(previous_matrix),
                ) => factor.try_refactor_fixed_pattern(
                    previous_matrix,
                    matrix,
                    &self.config.ilut_config(),
                ),
                _ => Err(LinearSolverError::OperationFailed {
                    operation: "ILUT fixed-pattern numeric refactorization",
                    reason: "cached factor or reference matrix is unavailable".into(),
                }),
            };
            match refactor {
                Ok(()) => {
                    reuse = ResolvedPreconditionerReuse::NumericRefactor;
                    setup_count = 1;
                }
                Err(error) => fallbacks.push(ResolvedPreconditionerFallback {
                    source: ResolvedPreconditionerKind::Ilut,
                    destination: ResolvedPreconditionerKind::Ilut,
                    reason: format!("numeric refactor failed; rebuilt fully: {error}"),
                }),
            }
        }
        if reuse == ResolvedPreconditionerReuse::FreshBuild {
            let shifted_matrix = (active_kind == ResolvedPreconditionerKind::ShiftedLaplacianAmg)
                .then(|| {
                    let omega = 2.0 * PI * frequency_hz;
                    let wavenumber = omega / self.config.fluid.speed_of_sound;
                    let port_alpha = thermal_wall_admittance(&self.config.fluid, omega).conj();
                    let cavity_alpha =
                        self.config
                            .cavity_wall_specific_impedance_pa_s_per_m
                            .map(|impedance| {
                                Complex64::new(0.0, omega * self.config.fluid.density) / impedance
                            });
                    self.assemble_shifted_laplacian_matrix(wavenumber, port_alpha, cavity_alpha)
                });
            let (cached, mut build_fallbacks) = ResolvedCachedPreconditioner::build_admitted(
                active_kind,
                matrix,
                shifted_matrix.as_ref(),
                rhs,
                &self.config,
            )?;
            continuation.preconditioner = Some(cached);
            fallbacks.append(&mut build_fallbacks);
            setup_count += 1;
        }
        continuation.preconditioner_frequency_hz = Some(frequency_hz);
        continuation.preconditioner_matrix = Some(matrix.clone());
        let mut preconditioner_rebuilt = reuse != ResolvedPreconditionerReuse::FullReuse;
        let mut built = preconditioner_factory(
            matrix,
            continuation
                .preconditioner
                .as_ref()
                .expect("the continuation always has a preconditioner before solving")
                .clone(),
        )?;
        let mut fine_setup_time_seconds = setup_started.elapsed().as_secs_f64();
        let restart = self
            .config
            .fgmres_restart
            .min(self.config.fgmres_max_iterations);
        let config = FgmresConfig {
            max_iterations: self.config.fgmres_max_iterations.div_ceil(restart),
            restart,
            tolerance: self.config.fgmres_tolerance,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 1,
        };
        // Measure the factory product itself. Reading only the continuation's
        // cached factor misses wrapper work because learned adapters own a
        // cloned raw ILU/ILUT factor rather than the cached telemetry object.
        let factory_telemetry = ResolvedApplyTelemetry::default();
        let iteration_started = Instant::now();
        let mut solved = fgmres_preconditioned(
            matrix,
            &ResolvedMeasuredPreconditioner {
                inner: &built.preconditioner,
                telemetry: &factory_telemetry,
            },
            rhs,
            initial_guess,
            &config,
        );
        let mut relative_residual = exact_relative_residual(matrix, &solved.x, rhs);
        if (!solved.converged || relative_residual > config.tolerance)
            && reuse == ResolvedPreconditionerReuse::FullReuse
        {
            let rebuild_started = Instant::now();
            let shifted_matrix = (active_kind == ResolvedPreconditionerKind::ShiftedLaplacianAmg)
                .then(|| {
                    let omega = 2.0 * PI * frequency_hz;
                    self.assemble_shifted_laplacian_matrix(
                        omega / self.config.fluid.speed_of_sound,
                        thermal_wall_admittance(&self.config.fluid, omega).conj(),
                        self.config
                            .cavity_wall_specific_impedance_pa_s_per_m
                            .map(|impedance| {
                                Complex64::new(0.0, omega * self.config.fluid.density) / impedance
                            }),
                    )
                });
            let (cached, mut rebuild_fallbacks) = ResolvedCachedPreconditioner::build_admitted(
                active_kind,
                matrix,
                shifted_matrix.as_ref(),
                rhs,
                &self.config,
            )?;
            fallbacks.push(ResolvedPreconditionerFallback {
                source: active_kind,
                destination: active_kind,
                reason: "reused preconditioner failed convergence; rebuilt fully".into(),
            });
            fallbacks.append(&mut rebuild_fallbacks);
            continuation.preconditioner = Some(cached);
            continuation.preconditioner_frequency_hz = Some(frequency_hz);
            continuation.preconditioner_matrix = Some(matrix.clone());
            preconditioner_rebuilt = true;
            reuse = ResolvedPreconditionerReuse::FreshBuild;
            setup_count += 1;
            built = preconditioner_factory(
                matrix,
                continuation
                    .preconditioner
                    .as_ref()
                    .expect("the rebuilt preconditioner is available")
                    .clone(),
            )?;
            fine_setup_time_seconds += rebuild_started.elapsed().as_secs_f64();
            solved = fgmres_preconditioned(
                matrix,
                &ResolvedMeasuredPreconditioner {
                    inner: &built.preconditioner,
                    telemetry: &factory_telemetry,
                },
                rhs,
                initial_guess,
                &config,
            );
            relative_residual = exact_relative_residual(matrix, &solved.x, rhs);
        }
        let cached = continuation
            .preconditioner
            .as_ref()
            .expect("cached preconditioner remains available after solve");
        let active_preconditioner = cached.kind();
        let apply_count = factory_telemetry.count.load(Ordering::Relaxed);
        let apply_time_seconds = factory_telemetry.total_ns.load(Ordering::Relaxed) as f64 / 1.0e9;
        let (preconditioner_nonzeros, preconditioner_fill_ratio, storage_bytes, regularized) =
            cached.storage_diagnostics(matrix.nnz());
        if !solved.converged
            || !relative_residual.is_finite()
            || relative_residual > config.tolerance
        {
            return self.direct_solve_fallback(
                matrix,
                rhs,
                continuation,
                solved.iterations,
                relative_residual,
                initial_guess_reused,
                preconditioner_rebuilt,
                config.tolerance,
                selected_guess_kind,
                candidate_residuals,
                fine_setup_time_seconds,
                iteration_started.elapsed().as_secs_f64(),
                built.diagnostics,
                preconditioner_kind,
                active_preconditioner,
                fallbacks,
                reuse,
                setup_count,
                apply_count,
                apply_time_seconds,
                preconditioner_nonzeros,
                preconditioner_fill_ratio,
                storage_bytes,
                regularized,
            );
        }
        continuation.previous_pressure = Some(solved.x.clone());
        Ok((
            solved.x,
            ResolvedSolveDiagnostics {
                iterations: solved.iterations,
                relative_residual,
                iterative_relative_residual: Some(relative_residual),
                initial_guess_reused,
                preconditioner_rebuilt,
                direct_fallback_used: false,
                selected_guess_kind,
                candidate_residuals,
                ml_rejection_reason: None,
                coarse_time_seconds: 0.0,
                ml_time_seconds: 0.0,
                fine_setup_time_seconds,
                fine_iteration_time_seconds: iteration_started.elapsed().as_secs_f64(),
                learned_preconditioner_requested: built.diagnostics.learned_requested,
                learned_preconditioner_active: built.diagnostics.learned_active,
                learned_preconditioner_probe_ratio: built.diagnostics.probe_residual_ratio,
                learned_preconditioner_rejection_reason: built.diagnostics.rejection_reason,
                requested_preconditioner: preconditioner_kind,
                active_preconditioner: Some(active_preconditioner),
                preconditioner_fallbacks: fallbacks,
                preconditioner_reuse: reuse,
                preconditioner_setup_count: setup_count,
                preconditioner_apply_count: apply_count,
                preconditioner_setup_time_seconds: fine_setup_time_seconds,
                preconditioner_apply_time_seconds: apply_time_seconds,
                preconditioner_nonzeros,
                preconditioner_fill_ratio,
                preconditioner_estimated_storage_bytes: storage_bytes,
                preconditioner_regularized_pivots: regularized,
            },
        ))
    }

    /// Last-resort sparse-direct solve when the FGMRES+ILU path stalls.
    ///
    /// Low-frequency walled sealed configurations (a compliance-dominated
    /// cavity vented through a micron-scale slit) sit on the cavity/slit
    /// Helmholtz anti-resonance: the assembled matrix mixes PML-stretch
    /// entries ~1e9 with slit-cell entries ~1e-14 and is numerically singular
    /// in f64, so *no* double-precision method — ILU-preconditioned FGMRES,
    /// equilibrated LU, or LU with iterative refinement — can reach the
    /// 1e-10 exact-residual Krylov gate (measured attainable floor ~2e-4 at
    /// 40 Hz on the cells=8 slit mesh; even the coarse direct path solves
    /// this physics at ~1e-5). Up to [`DIRECT_FALLBACK_MAX_NODES`] the
    /// fallback therefore certifies the solve the way the small-model direct
    /// path always has: [`SparseLuFactorization::factorize`] refuses factors
    /// whose pivots reach rounding level (`SparseLuError`), and a bounded
    /// accept-only-if-improving refinement polishes the iterate. The achieved
    /// residual is still measured exactly and reported through
    /// [`ResolvedSolveDiagnostics::relative_residual`], so consumers retain
    /// full visibility. Models beyond the cap, and models whose
    /// factorization hits the pivot guard, still receive
    /// [`ResolvedSpeakerAcousticError::ConvergenceFailure`].
    #[allow(clippy::too_many_arguments)]
    fn direct_solve_fallback<'a>(
        &'a self,
        matrix: &CsrMatrix<Complex64>,
        rhs: &[Complex64],
        continuation: &mut ResolvedFrequencyContinuation<'a>,
        iterative_iterations: usize,
        iterative_residual: f64,
        initial_guess_reused: bool,
        preconditioner_rebuilt: bool,
        tolerance: f64,
        selected_guess_kind: ResolvedGuessKind,
        candidate_residuals: ResolvedCandidateResiduals,
        fine_setup_time_seconds: f64,
        iterative_time_seconds: f64,
        preconditioner_diagnostics: ResolvedPreconditionerDiagnostics,
        requested_preconditioner: ResolvedPreconditionerKind,
        active_preconditioner: ResolvedPreconditionerKind,
        preconditioner_fallbacks: Vec<ResolvedPreconditionerFallback>,
        preconditioner_reuse: ResolvedPreconditionerReuse,
        preconditioner_setup_count: usize,
        preconditioner_apply_count: u64,
        preconditioner_apply_time_seconds: f64,
        preconditioner_nonzeros: usize,
        preconditioner_fill_ratio: f64,
        preconditioner_storage_bytes: usize,
        preconditioner_regularized_pivots: usize,
    ) -> Result<(Vec<Complex64>, ResolvedSolveDiagnostics), ResolvedSpeakerAcousticError> {
        if self.mesh.num_nodes() > DIRECT_FALLBACK_MAX_NODES {
            return Err(ResolvedSpeakerAcousticError::ConvergenceFailure {
                iterations: iterative_iterations,
                relative_residual: iterative_residual,
            });
        }
        let failure =
            |iterations, relative_residual| ResolvedSpeakerAcousticError::ConvergenceFailure {
                iterations,
                relative_residual,
            };
        let factorization = SparseLuFactorization::factorize(matrix)
            .map_err(|_| failure(iterative_iterations, iterative_residual))?;
        let mut x = factorization
            .solve(rhs)
            .map_err(|_| failure(iterative_iterations, iterative_residual))?;
        let mut relative_residual = exact_relative_residual(matrix, &x, rhs);
        // Bounded refinement: each accepted step must strictly improve the
        // exact residual. On the near-singular configs that reach this path
        // the attainable floor sits near eps·cond(A), so the loop typically
        // exits by stagnation after a step or two rather than at `tolerance`.
        let mut refinement_steps = 0;
        while refinement_steps < MAX_FALLBACK_REFINEMENT_STEPS && relative_residual > tolerance {
            let defect: Vec<_> = matrix
                .matvec(&x)
                .iter()
                .zip(rhs)
                .map(|(applied, expected)| expected - applied)
                .collect();
            let correction = factorization
                .solve(&defect)
                .map_err(|_| failure(iterative_iterations, relative_residual))?;
            let candidate: Vec<_> = x
                .iter()
                .zip(correction)
                .map(|(value, delta)| *value + delta)
                .collect();
            let candidate_residual = exact_relative_residual(matrix, &candidate, rhs);
            if !candidate_residual.is_finite() || candidate_residual >= relative_residual {
                break;
            }
            x = candidate;
            relative_residual = candidate_residual;
            refinement_steps += 1;
        }
        continuation.previous_pressure = Some(x.clone());
        Ok((
            x,
            ResolvedSolveDiagnostics {
                iterations: iterative_iterations + refinement_steps,
                relative_residual,
                iterative_relative_residual: Some(iterative_residual),
                initial_guess_reused,
                preconditioner_rebuilt,
                direct_fallback_used: true,
                selected_guess_kind,
                candidate_residuals,
                ml_rejection_reason: None,
                coarse_time_seconds: 0.0,
                ml_time_seconds: 0.0,
                fine_setup_time_seconds,
                fine_iteration_time_seconds: iterative_time_seconds,
                learned_preconditioner_requested: preconditioner_diagnostics.learned_requested,
                learned_preconditioner_active: preconditioner_diagnostics.learned_active,
                learned_preconditioner_probe_ratio: preconditioner_diagnostics.probe_residual_ratio,
                learned_preconditioner_rejection_reason: preconditioner_diagnostics
                    .rejection_reason,
                requested_preconditioner,
                active_preconditioner: Some(active_preconditioner),
                preconditioner_fallbacks,
                preconditioner_reuse,
                preconditioner_setup_count,
                preconditioner_apply_count,
                preconditioner_setup_time_seconds: fine_setup_time_seconds,
                preconditioner_apply_time_seconds,
                preconditioner_nonzeros,
                preconditioner_fill_ratio,
                preconditioner_estimated_storage_bytes: preconditioner_storage_bytes,
                preconditioner_regularized_pivots,
            },
        ))
    }

    /// Condensed piston load impedance `Z = F_opposing / v̄` in N·s/m.
    ///
    /// `F_opposing = ∫_front p dA − ∫_rear p dA` is the axial force the
    /// resolved acoustic domain applies against the diaphragm's +z motion.
    /// The model uses `exp(+iωt)` phasors (matching the distributed
    /// electroacoustic system), so the time-averaged power the diaphragm
    /// delivers into the acoustic domain is `Re(Z·|v̄|²)`, which is
    /// non-negative for this passive domain — for an equal front/rear drive
    /// it equals [`ResolvedAcousticSolution::driver_input_power_w`]. The
    /// model is linear in the drive, so a unit-velocity solve yields the
    /// exact per-frequency condensation.
    pub fn piston_load_impedance(&self, solution: &ResolvedAcousticSolution) -> Complex64 {
        let front =
            boundary_pressure_integral(&self.mesh, &solution.pressure_rms_pa, FRONT_DIAPHRAGM);
        let rear =
            boundary_pressure_integral(&self.mesh, &solution.pressure_rms_pa, REAR_DIAPHRAGM);
        (front - rear) / solution.diaphragm_velocity_rms_m_per_s
    }

    /// Solve an ordered sweep with affine O(nnz) matrix reconstruction,
    /// pressure continuation, and ILU reuse across nearby frequencies.
    pub fn solve_sweep(
        &self,
        frequencies_hz: &[f64],
        diaphragm_velocity_rms_m_per_s: Complex64,
    ) -> Result<Vec<ResolvedAcousticSolution>, ResolvedSpeakerAcousticError> {
        let mut solutions = Vec::with_capacity(frequencies_hz.len());
        let mut continuation = ResolvedFrequencyContinuation::new(self);
        for &frequency_hz in frequencies_hz {
            solutions.push(
                continuation
                    .solve_frequency_with_drive(frequency_hz, diaphragm_velocity_rms_m_per_s)?,
            );
        }
        Ok(solutions)
    }

    /// Solve frequency/drive pairs in contiguous lanes with lane-local
    /// pressure and preconditioner continuation state.
    pub fn solve_driven_sweep_parallel(
        &self,
        frequencies_hz: &[f64],
        diaphragm_velocities_rms_m_per_s: &[Complex64],
    ) -> Result<Vec<ResolvedAcousticSolution>, ResolvedSpeakerAcousticError> {
        if frequencies_hz.len() != diaphragm_velocities_rms_m_per_s.len() {
            return Err(ResolvedSpeakerAcousticError::InvalidConfiguration(
                "frequency and diaphragm-velocity sweeps must have equal length",
            ));
        }
        if frequencies_hz.is_empty() {
            return Ok(Vec::new());
        }
        let estimated_bytes_per_lane = self.mesh.num_nodes().saturating_mul(16).saturating_mul(96);
        let lane_count =
            crate::parallelism::resolved_lane_count(frequencies_hz.len(), estimated_bytes_per_lane);
        let lanes = crate::parallelism::contiguous_lane_ranges(frequencies_hz.len(), lane_count);
        let lane_solutions = lanes
            .into_par_iter()
            .map(|range| {
                let mut continuation = ResolvedFrequencyContinuation::new(self);
                range
                    .map(|index| {
                        continuation.solve_frequency_with_drive(
                            frequencies_hz[index],
                            diaphragm_velocities_rms_m_per_s[index],
                        )
                    })
                    .collect::<Result<Vec<_>, ResolvedSpeakerAcousticError>>()
            })
            .collect::<Result<Vec<_>, ResolvedSpeakerAcousticError>>()?;
        Ok(lane_solutions.into_iter().flatten().collect())
    }

    pub fn export_solution(
        &self,
        solution: &ResolvedAcousticSolution,
        path: impl AsRef<Path>,
    ) -> Result<ExportedSnapshot, ResolvedSpeakerAcousticError> {
        let fields = self.export_fields(solution)?;
        let mut metadata = StudyMetadata::new("resolved finite-baffle enclosure acoustics");
        metadata.frequency_hz = Some(solution.frequency_hz);
        metadata.reference_case = Some("axisymmetric cavity-port-PML".into());
        Ok(SpeakerFieldExporter::new(&self.mesh, metadata, &fields).write_snapshot(path)?)
    }

    pub fn export_sweep(
        &self,
        frequencies_hz: &[f64],
        diaphragm_velocity_rms_m_per_s: Complex64,
        pvd_path: impl AsRef<Path>,
    ) -> Result<Vec<ResolvedAcousticSolution>, ResolvedSpeakerAcousticError> {
        let solutions = self.solve_sweep(frequencies_hz, diaphragm_velocity_rms_m_per_s)?;
        self.export_solutions(&solutions, pvd_path)?;
        Ok(solutions)
    }

    /// Export already solved points whose diaphragm velocity may vary by frequency.
    pub fn export_solutions(
        &self,
        solutions: &[ResolvedAcousticSolution],
        pvd_path: impl AsRef<Path>,
    ) -> Result<(), ResolvedSpeakerAcousticError> {
        let mut collection = SpeakerPvdCollection::new(pvd_path)?;
        for solution in solutions {
            let fields = self.export_fields(solution)?;
            let mut metadata = StudyMetadata::new("resolved finite-baffle enclosure acoustics");
            metadata.frequency_hz = Some(solution.frequency_hz);
            metadata.reference_case = Some("axisymmetric cavity-port-PML".into());
            collection.push(&SpeakerFieldExporter::new(&self.mesh, metadata, &fields))?;
        }
        collection.finish()?;
        Ok(())
    }

    fn export_fields(
        &self,
        solution: &ResolvedAcousticSolution,
    ) -> Result<SpeakerFieldBundle, FieldExportError> {
        let mut fields = SpeakerFieldBundle::new();
        fields.add_point_complex_scalar(
            field_names::ACOUSTIC_PRESSURE,
            solution.pressure_rms_pa.clone(),
        )?;
        fields.add_point_real_scalar(
            field_names::SOUND_PRESSURE_LEVEL,
            solution
                .pressure_rms_pa
                .iter()
                .map(|pressure| spl_db(pressure.norm()))
                .collect(),
        )?;
        let omega = 2.0 * PI * solution.frequency_hz;
        let mut velocity = Vec::with_capacity(self.mesh.num_elements());
        let mut intensity = Vec::with_capacity(self.mesh.num_elements());
        for element in &self.mesh.elements {
            let nodes = element.vertices();
            let gradient = triangle_complex_gradient(&self.mesh, nodes, &solution.pressure_rms_pa);
            let particle_velocity = [
                Complex64::new(0.0, 1.0) * gradient[0] / (omega * self.config.fluid.density),
                Complex64::default(),
                Complex64::new(0.0, 1.0) * gradient[1] / (omega * self.config.fluid.density),
            ];
            let pressure = nodes
                .iter()
                .map(|node| solution.pressure_rms_pa[*node])
                .sum::<Complex64>()
                / nodes.len() as f64;
            velocity.push(particle_velocity);
            intensity.push([
                (pressure * particle_velocity[0].conj()).re,
                0.0,
                (pressure * particle_velocity[2].conj()).re,
            ]);
        }
        fields.add_cell_complex_vector(
            "particle_velocity",
            ProjectedCellVector::new(velocity, ProjectionProvenance::ElementCentroid),
        )?;
        fields.add_cell_real_vector(
            field_names::ACOUSTIC_INTENSITY,
            ProjectedCellVector::new(intensity, ProjectionProvenance::ElementCentroid),
        )?;
        fields.add_cell_real_scalar(
            field_names::REGION_MARKER,
            self.mesh
                .elements
                .iter()
                .map(|element| element.marker as f64)
                .collect(),
        )?;
        Ok(fields)
    }
}

fn build_conforming_mesh(config: &ResolvedAcousticConfig) -> Result<Mesh, PhysicalGroupError> {
    let n = config.cells_per_physical_interval;
    let radial_outer = config.radial_pml_start_m + config.radial_pml_thickness_m;
    let cavity_bottom = -config.cavity_depth_m - config.baffle_half_thickness_m;
    let axial_bottom = config.lower_pml_start_m - config.lower_pml_thickness_m;
    let axial_top = config.upper_pml_start_m + config.upper_pml_thickness_m;
    let radial = segmented_coordinates(&[
        (0.0, config.diaphragm_radius_m, n),
        (config.diaphragm_radius_m, config.port_inner_radius_m, n),
        (config.port_inner_radius_m, config.port_outer_radius_m, n),
        (config.port_outer_radius_m, config.enclosure_radius_m, n),
        (config.enclosure_radius_m, config.baffle_radius_m, n),
        (config.baffle_radius_m, config.radial_pml_start_m, n),
        (config.radial_pml_start_m, radial_outer, config.pml_cells),
    ]);
    let axial = segmented_coordinates(&[
        (axial_bottom, config.lower_pml_start_m, config.pml_cells),
        (config.lower_pml_start_m, cavity_bottom, n),
        (cavity_bottom, -config.baffle_half_thickness_m, n),
        (-config.baffle_half_thickness_m, 0.0, n),
        (0.0, config.baffle_half_thickness_m, n),
        (config.baffle_half_thickness_m, config.upper_pml_start_m, n),
        (config.upper_pml_start_m, axial_top, config.pml_cells),
    ]);
    let nr = radial.len();
    let mut source_points = Vec::with_capacity(radial.len() * axial.len());
    for &z in &axial {
        for &r in &radial {
            source_points.push(Point::new_2d(r, z));
        }
    }
    let mut source_cells = Vec::new();
    for iz in 0..axial.len() - 1 {
        for ir in 0..radial.len() - 1 {
            let centroid_r = 0.5 * (radial[ir] + radial[ir + 1]);
            let centroid_z = 0.5 * (axial[iz] + axial[iz + 1]);
            let Some(marker) = fluid_region(config, centroid_r, centroid_z) else {
                continue;
            };
            let n00 = iz * nr + ir;
            let n10 = n00 + 1;
            let n01 = (iz + 1) * nr + ir;
            let n11 = n01 + 1;
            source_cells.push(([n00, n10, n11], marker));
            source_cells.push(([n00, n11, n01], marker));
        }
    }
    let used: BTreeSet<_> = source_cells
        .iter()
        .flat_map(|(nodes, _)| nodes.iter().copied())
        .collect();
    let mut old_to_new = BTreeMap::new();
    let mut mesh = Mesh::new(2);
    for old in used {
        old_to_new.insert(old, mesh.add_node(source_points[old]));
    }
    for (nodes, marker) in source_cells {
        mesh.add_element_with_marker(
            ElementType::Triangle,
            nodes.iter().map(|node| old_to_new[node]).collect(),
            marker,
        );
    }
    add_external_boundaries(&mut mesh, config, radial_outer, axial_bottom, axial_top);
    for (dimension, tag, name) in [
        (2, REAR_CAVITY, "rear cavity"),
        (2, RESOLVED_PORT, "resolved annular port"),
        (2, PHYSICAL_EXTERIOR, "finite-baffle exterior"),
        (2, ACOUSTIC_PML, "exterior PML"),
        (1, FRONT_DIAPHRAGM, "front diaphragm"),
        (1, REAR_DIAPHRAGM, "rear diaphragm"),
        (1, PORT_WALL, "port thermoviscous wall"),
        (1, CAVITY_WALL, "cavity wall"),
        (1, RIGID_SOLID_WALL, "rigid baffle and enclosure exterior"),
        (1, OUTER_PML_TRUNCATION, "outer PML truncation"),
        (1, SYMMETRY_AXIS, "axis of symmetry"),
    ] {
        mesh.register_physical_group(dimension, tag, name)?;
    }
    Ok(mesh)
}

fn segmented_coordinates(segments: &[(f64, f64, usize)]) -> Vec<f64> {
    let mut coordinates = Vec::new();
    for &(start, end, cells) in segments {
        if coordinates.is_empty() {
            coordinates.push(start);
        }
        for index in 1..=cells {
            coordinates.push(start + (end - start) * index as f64 / cells as f64);
        }
    }
    coordinates
}

fn fluid_region(config: &ResolvedAcousticConfig, r: f64, z: f64) -> Option<i32> {
    if r >= config.radial_pml_start_m
        || z >= config.upper_pml_start_m
        || z <= config.lower_pml_start_m
    {
        return Some(ACOUSTIC_PML);
    }
    let half = config.baffle_half_thickness_m;
    let cavity_bottom = -config.cavity_depth_m - half;
    if z > half || z < cavity_bottom {
        if config.seal_enclosure_walls && z < cavity_bottom && r < config.baffle_radius_m {
            // Solid shadow behind the enclosure bottom panel, extended to the
            // shell's outer radius: a narrower shadow leaves the cavity's
            // bottom corner node shared with an exterior cell, and that
            // point contact leaks the cavity compliance into the exterior.
            return None;
        }
        return Some(PHYSICAL_EXTERIOR);
    }
    if z < -half {
        if r < config.enclosure_radius_m {
            return Some(REAR_CAVITY);
        }
        if config.seal_enclosure_walls && r < config.baffle_radius_m {
            // Solid enclosure shell beside the cavity.
            return None;
        }
        return Some(PHYSICAL_EXTERIOR);
    }
    if r > config.baffle_radius_m {
        return Some(PHYSICAL_EXTERIOR);
    }
    if r > config.port_inner_radius_m && r < config.port_outer_radius_m {
        return Some(RESOLVED_PORT);
    }
    None
}

fn add_external_boundaries(
    mesh: &mut Mesh,
    config: &ResolvedAcousticConfig,
    radial_outer: f64,
    axial_bottom: f64,
    axial_top: f64,
) {
    let mut owners: BTreeMap<(usize, usize), Vec<(usize, usize)>> = BTreeMap::new();
    for (element_index, element) in mesh.elements.iter().enumerate() {
        for (local_index, (a, b)) in element.edges().into_iter().enumerate() {
            owners
                .entry((a.min(b), a.max(b)))
                .or_default()
                .push((element_index, local_index));
        }
    }
    let tolerance = 1.0e-10;
    for ((a, b), adjacent) in owners {
        if adjacent.len() != 1 {
            continue;
        }
        let p0 = &mesh.nodes[a];
        let p1 = &mesh.nodes[b];
        let r = 0.5 * (p0.x + p1.x);
        let z = 0.5 * (p0.y + p1.y);
        let owner = adjacent[0].0;
        let owner_marker = mesh.elements[owner].marker;
        let on_outer = (r - radial_outer).abs() < tolerance
            || (z - axial_bottom).abs() < tolerance
            || (z - axial_top).abs() < tolerance;
        let (boundary_type, marker) = if on_outer {
            (BoundaryType::Dirichlet, OUTER_PML_TRUNCATION)
        } else if r.abs() < tolerance {
            (BoundaryType::Neumann, SYMMETRY_AXIS)
        } else if (z - config.baffle_half_thickness_m).abs() < tolerance
            && r <= config.diaphragm_radius_m + tolerance
        {
            (BoundaryType::Neumann, FRONT_DIAPHRAGM)
        } else if (z + config.baffle_half_thickness_m).abs() < tolerance
            && r <= config.diaphragm_radius_m + tolerance
        {
            (BoundaryType::Neumann, REAR_DIAPHRAGM)
        } else if ((r - config.port_inner_radius_m).abs() < tolerance
            || (r - config.port_outer_radius_m).abs() < tolerance)
            && z >= -config.baffle_half_thickness_m - tolerance
            && z <= config.baffle_half_thickness_m + tolerance
        {
            (BoundaryType::Robin, PORT_WALL)
        } else if owner_marker == REAR_CAVITY {
            (BoundaryType::Robin, CAVITY_WALL)
        } else {
            (BoundaryType::Neumann, RIGID_SOLID_WALL)
        };
        mesh.add_boundary(vec![a, b], boundary_type, marker, owner, adjacent[0].1);
    }
}

fn volume_average_pressure(mesh: &Mesh, pressure: &[Complex64], marker: i32) -> Complex64 {
    let mut integral = Complex64::default();
    let mut volume = 0.0;
    for element in mesh
        .elements
        .iter()
        .filter(|element| element.marker == marker)
    {
        let nodes = element.vertices();
        let area = triangle_area(mesh, nodes);
        let radius = nodes.iter().map(|node| mesh.nodes[*node].x).sum::<f64>() / 3.0;
        let cell_volume = 2.0 * PI * radius * area;
        let cell_pressure = nodes.iter().map(|node| pressure[*node]).sum::<Complex64>() / 3.0;
        integral += cell_pressure * cell_volume;
        volume += cell_volume;
    }
    integral / volume
}

fn port_volume_velocity(
    mesh: &Mesh,
    pressure: &[Complex64],
    omega: f64,
    density: f64,
) -> Complex64 {
    let mut owners: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
    for (element_index, element) in mesh.elements.iter().enumerate() {
        if element.marker != RESOLVED_PORT {
            continue;
        }
        for (a, b) in element.edges() {
            owners
                .entry((a.min(b), a.max(b)))
                .or_default()
                .push(element_index);
        }
    }
    owners
        .into_iter()
        .filter(|((a, b), _)| mesh.nodes[*a].y.abs() < 1.0e-12 && mesh.nodes[*b].y.abs() < 1.0e-12)
        .map(|((a, b), elements)| {
            let uz = elements
                .iter()
                .map(|element| {
                    let gradient = triangle_complex_gradient(
                        mesh,
                        mesh.elements[*element].vertices(),
                        pressure,
                    );
                    Complex64::new(0.0, 1.0) * gradient[1] / (omega * density)
                })
                .sum::<Complex64>()
                / elements.len() as f64;
            let r0 = mesh.nodes[a].x;
            let r1 = mesh.nodes[b].x;
            uz * PI * (r1.powi(2) - r0.powi(2)).abs()
        })
        .sum()
}

fn boundary_pressure_integral(mesh: &Mesh, pressure: &[Complex64], marker: i32) -> Complex64 {
    let gauss = [-1.0_f64 / 3.0_f64.sqrt(), 1.0_f64 / 3.0_f64.sqrt()];
    let mut integral = Complex64::default();
    for boundary in mesh
        .boundaries
        .iter()
        .filter(|boundary| boundary.marker == marker)
    {
        let p0 = &mesh.nodes[boundary.nodes[0]];
        let p1 = &mesh.nodes[boundary.nodes[1]];
        let length = (p1.x - p0.x).hypot(p1.y - p0.y);
        for xi in gauss {
            let shape = [0.5 * (1.0 - xi), 0.5 * (1.0 + xi)];
            let radius = shape[0] * p0.x + shape[1] * p1.x;
            let p = shape[0] * pressure[boundary.nodes[0]] + shape[1] * pressure[boundary.nodes[1]];
            integral += p * 2.0 * PI * radius * 0.5 * length;
        }
    }
    integral
}

fn robin_loss_power(
    mesh: &Mesh,
    pressure: &[Complex64],
    marker: i32,
    alpha: Complex64,
    omega: f64,
    density: f64,
) -> f64 {
    let gauss = [-1.0_f64 / 3.0_f64.sqrt(), 1.0_f64 / 3.0_f64.sqrt()];
    let mut pressure_squared_integral = 0.0;
    for boundary in mesh
        .boundaries
        .iter()
        .filter(|boundary| boundary.marker == marker)
    {
        let p0 = &mesh.nodes[boundary.nodes[0]];
        let p1 = &mesh.nodes[boundary.nodes[1]];
        let length = (p1.x - p0.x).hypot(p1.y - p0.y);
        for xi in gauss {
            let shape = [0.5 * (1.0 - xi), 0.5 * (1.0 + xi)];
            let radius = shape[0] * p0.x + shape[1] * p1.x;
            let p = shape[0] * pressure[boundary.nodes[0]] + shape[1] * pressure[boundary.nodes[1]];
            pressure_squared_integral += p.norm_sqr() * 2.0 * PI * radius * 0.5 * length;
        }
    }
    alpha.im * pressure_squared_integral / (omega * density)
}

fn triangle_area(mesh: &Mesh, nodes: &[usize]) -> f64 {
    let a = &mesh.nodes[nodes[0]];
    let b = &mesh.nodes[nodes[1]];
    let c = &mesh.nodes[nodes[2]];
    0.5 * ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)).abs()
}

fn triangle_complex_gradient(mesh: &Mesh, nodes: &[usize], values: &[Complex64]) -> [Complex64; 2] {
    let a = &mesh.nodes[nodes[0]];
    let b = &mesh.nodes[nodes[1]];
    let c = &mesh.nodes[nodes[2]];
    let twice_area = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
    let gradients = [
        [(b.y - c.y) / twice_area, (c.x - b.x) / twice_area],
        [(c.y - a.y) / twice_area, (a.x - c.x) / twice_area],
        [(a.y - b.y) / twice_area, (b.x - a.x) / twice_area],
    ];
    [
        (0..3).map(|i| values[nodes[i]] * gradients[i][0]).sum(),
        (0..3).map(|i| values[nodes[i]] * gradients[i][1]).sum(),
    ]
}

fn spl_db(pressure_rms_pa: f64) -> f64 {
    20.0 * (pressure_rms_pa.max(1.0e-30) / 20.0e-6).log10()
}

fn complex_is_finite(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

fn hash_mesh_connectivity(digest: &mut Sha256, mesh: &Mesh) {
    digest.update((mesh.elements.len() as u64).to_le_bytes());
    for element in &mesh.elements {
        digest.update((element.element_type as u8).to_le_bytes());
        digest.update(element.marker.to_le_bytes());
        digest.update((element.nodes.len() as u64).to_le_bytes());
        for &node in &element.nodes {
            digest.update((node as u64).to_le_bytes());
        }
    }
    digest.update((mesh.boundaries.len() as u64).to_le_bytes());
    for boundary in &mesh.boundaries {
        digest.update((boundary.boundary_type as u8).to_le_bytes());
        digest.update(boundary.marker.to_le_bytes());
        digest.update((boundary.element_idx as u64).to_le_bytes());
        digest.update((boundary.local_idx as u64).to_le_bytes());
        digest.update((boundary.nodes.len() as u64).to_le_bytes());
        for &node in &boundary.nodes {
            digest.update((node as u64).to_le_bytes());
        }
    }
}

fn exact_relative_residual(
    matrix: &CsrMatrix<Complex64>,
    solution: &[Complex64],
    rhs: &[Complex64],
) -> f64 {
    let applied = matrix.matvec(solution);
    let residual_squared = applied
        .iter()
        .zip(rhs)
        .map(|(actual, expected)| (*actual - *expected).norm_sqr())
        .sum::<f64>();
    let rhs_squared = rhs.iter().map(|value| value.norm_sqr()).sum::<f64>();
    residual_squared.sqrt() / rhs_squared.sqrt().max(f64::MIN_POSITIVE)
}

fn optimally_scaled_preconditioner_probe_ratio<P: Preconditioner<Complex64>>(
    matrix: &CsrMatrix<Complex64>,
    preconditioner: &P,
    rhs: &[Complex64],
) -> f64 {
    let direction = preconditioner.apply(rhs);
    if direction.len() != rhs.len() || direction.iter().any(|value| !complex_is_finite(*value)) {
        return f64::INFINITY;
    }
    let applied = matrix.matvec(&direction);
    let denominator = applied.iter().map(|value| value.norm_sqr()).sum::<f64>();
    let rhs_norm = rhs.iter().map(|value| value.norm_sqr()).sum::<f64>().sqrt();
    if !denominator.is_finite() || denominator <= f64::MIN_POSITIVE || rhs_norm == 0.0 {
        return f64::INFINITY;
    }
    let numerator = applied
        .iter()
        .zip(rhs)
        .map(|(action, expected)| action.conj() * *expected)
        .sum::<Complex64>();
    let scale = numerator / denominator;
    applied
        .iter()
        .zip(rhs)
        .map(|(action, expected)| (*expected - scale * *action).norm_sqr())
        .sum::<f64>()
        .sqrt()
        / rhs_norm.max(f64::MIN_POSITIVE)
}

fn checked_exact_relative_residual(
    matrix: &CsrMatrix<Complex64>,
    solution: &[Complex64],
    rhs: &[Complex64],
) -> Result<f64, ResolvedSpeakerAcousticError> {
    if matrix.num_rows != matrix.num_cols || rhs.len() != matrix.num_rows {
        return Err(ResolvedSpeakerAcousticError::InvalidLinearSystem);
    }
    validate_external_guess(solution, rhs.len())?;
    Ok(exact_relative_residual(matrix, solution, rhs))
}

fn validate_external_guess(
    guess: &[Complex64],
    expected: usize,
) -> Result<(), ResolvedSpeakerAcousticError> {
    if guess.len() != expected {
        return Err(ResolvedSpeakerAcousticError::ExternalGuessLength {
            expected,
            actual: guess.len(),
        });
    }
    if guess.iter().any(|value| !complex_is_finite(*value)) {
        return Err(ResolvedSpeakerAcousticError::NonFiniteExternalGuess);
    }
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum ResolvedSpeakerAcousticError {
    #[error("invalid resolved acoustic model: {0}")]
    InvalidConfiguration(&'static str),
    #[error("frequency must be finite and positive")]
    InvalidFrequency,
    #[error("diaphragm velocity must be finite")]
    InvalidDrive,
    #[error("external guess has length {actual}; expected {expected} native nodal values")]
    ExternalGuessLength { expected: usize, actual: usize },
    #[error("external guess contains a non-finite native nodal value")]
    NonFiniteExternalGuess,
    #[error("assembled linear system is not square or does not match its RHS")]
    InvalidLinearSystem,
    #[error("assembled linear system mesh fingerprint does not match this model")]
    MeshFingerprintMismatch,
    #[error("resolved acoustic solve returned a non-finite field")]
    NonFiniteSolution,
    #[error(
        "resolved acoustic iterative solve did not converge in {iterations} iterations \
         (relative residual {relative_residual:e})"
    )]
    ConvergenceFailure {
        iterations: usize,
        relative_residual: f64,
    },
    #[error(transparent)]
    PhysicalGroup(#[from] PhysicalGroupError),
    #[error(transparent)]
    RegionAssembly(#[from] RegionAssemblyError),
    #[error(transparent)]
    Solver(#[from] FemSolverError),
    #[error(transparent)]
    LinearSolver(#[from] LinearSolverError),
    #[error(transparent)]
    ResolvedAcoustics(#[from] ResolvedAcousticsError),
    #[error(transparent)]
    Export(#[from] FieldExportError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_fem::assembly::{RegionMap, assemble_bilinear_by_region};
    use sonium_fem::basis::PolynomialDegree;
    use sonium_fem::boundary::{FemSystem, apply_homogeneous_dirichlet_complex};
    use sonium_fem::physics::acoustics::pressure::resolved::{
        AxisymmetricPmlHelmholtzForm, add_axisymmetric_robin,
    };
    use sonium_fem::spaces::H1Space;

    #[test]
    fn test_default_fgmres_restart_stays_valid() {
        // SIM-PERF-20261013.md section 2, gap 4: the default restart-50
        // is safe only with the strong ILUT/AMG preconditioner held
        // fixed; this guards the default against a zero restart.
        let config = ResolvedAcousticConfig::default();
        assert!(config.fgmres_restart > 0);
        assert!(config.validate().is_ok());
    }

    fn coarse_model() -> ResolvedAcousticModel {
        ResolvedAcousticModel::build(ResolvedAcousticConfig {
            cells_per_physical_interval: 1,
            pml_cells: 1,
            // Keep the one-cell test PML numerically resolvable. Production
            // meshes use the stronger default stretch with multiple cells.
            pml_sigma_max: 2.0,
            ..ResolvedAcousticConfig::default()
        })
        .unwrap()
    }

    fn directly_assembled_matrix(
        model: &ResolvedAcousticModel,
        frequency_hz: f64,
    ) -> CsrMatrix<Complex64> {
        let omega = 2.0 * PI * frequency_hz;
        let wavenumber = omega / model.config.fluid.speed_of_sound;
        let pml = positive_iomega_axisymmetric_exterior_pml(
            model.config.radial_pml_start_m,
            model.config.radial_pml_thickness_m,
            model.config.upper_pml_start_m,
            model.config.upper_pml_thickness_m,
            model.config.lower_pml_start_m,
            model.config.lower_pml_thickness_m,
            wavenumber,
            model.config.pml_sigma_max,
        );
        let form = AxisymmetricPmlHelmholtzForm::new(Complex64::new(wavenumber, 0.0), pml);
        let mut forms = RegionMap::new();
        for marker in [REAR_CAVITY, RESOLVED_PORT, PHYSICAL_EXTERIOR, ACOUSTIC_PML] {
            forms.insert(marker, form.clone());
        }
        let space = H1Space::new(PolynomialDegree::P1);
        let mut matrix =
            assemble_bilinear_by_region(&forms, &space, &space, &model.mesh, 4).unwrap();
        let port_alpha = thermal_wall_admittance(&model.config.fluid, omega).conj();
        add_axisymmetric_robin(&mut matrix, &model.mesh, PORT_WALL, port_alpha);
        let cavity_alpha = model
            .config
            .cavity_wall_specific_impedance_pa_s_per_m
            .map(|impedance| Complex64::new(0.0, omega * model.config.fluid.density) / impedance);
        if let Some(alpha) = cavity_alpha {
            add_axisymmetric_robin(&mut matrix, &model.mesh, CAVITY_WALL, alpha);
        }
        let mut system = FemSystem::new(matrix, vec![Complex64::default(); model.mesh.num_nodes()]);
        apply_homogeneous_dirichlet_complex(&mut system, &model.mesh, &space, OUTER_PML_TRUNCATION);
        system.matrix.to_csr()
    }

    /// Walled sealed slit configuration at 40 Hz, meshed past
    /// `direct_solve_max_nodes`: the compliance‖slit anti-resonance makes the
    /// assembled matrix numerically singular in f64 (entries span ~1e-14 to
    /// ~1e9), so FGMRES+ILU stalls at its attainable floor (measured ~7e-4)
    /// and the direct fallback must engage. The 1e-10 Krylov gate is
    /// unreachable for *any* f64 method on this matrix — sparse LU with
    /// refinement also floors near 2e-4 — so the certificate is physical,
    /// mirroring the sealed acceptance test in `resolved_coupling`: the
    /// condensed piston load must match the lumped compliance‖slit reference
    /// plus baffled-piston radiation within the same 2% bound, stay passive,
    /// and report its achieved residual through the diagnostics.
    #[test]
    fn walled_sealed_low_frequency_iterative_failure_falls_back_to_direct() {
        let config = ResolvedAcousticConfig {
            port_inner_radius_m: 79.99e-3,
            port_outer_radius_m: 80.0e-3 - 1.0e-9,
            seal_enclosure_walls: true,
            cells_per_physical_interval: 8,
            ..ResolvedAcousticConfig::default()
        };
        let model = ResolvedAcousticModel::build(config.clone()).unwrap();
        assert!(
            model.mesh.num_nodes() > config.direct_solve_max_nodes,
            "the test must exercise the iterative path"
        );
        let drive = Complex64::new(0.01, 0.0);
        let solution = model.solve_frequency(40.0, drive).unwrap();
        let residual = solution.solve_diagnostics.relative_residual;
        assert!(residual.is_finite(), "fallback must report its residual");
        assert!(solution.driver_input_power_w >= 0.0);

        let z_resolved = model.piston_load_impedance(&solution);
        assert!(
            z_resolved.re >= 0.0,
            "condensed load not passive: {z_resolved}"
        );
        let medium = crate::AcousticMedium {
            density_kg_per_m3: model.config.fluid.density,
            sound_speed_m_per_s: model.config.fluid.speed_of_sound,
            ..crate::AcousticMedium::standard_air()
        };
        let slit_width_m = config.port_outer_radius_m - config.port_inner_radius_m;
        let slit_area_m2 = 2.0
            * PI
            * 0.5
            * (config.port_inner_radius_m + config.port_outer_radius_m)
            * slit_width_m;
        let slit = crate::PortSpec {
            radius_m: (slit_area_m2 / PI).sqrt(),
            length_m: 2.0 * config.baffle_half_thickness_m,
            end_correction_m: 0.0,
            resistance_pa_s_per_m3: 0.0,
        };
        let sealed = crate::EnclosureSpec::Vented {
            volume_m3: config.cavity_volume_m3(),
            port: slit,
            leakage_resistance_pa_s_per_m3: 0.0,
            absorption_resistance_pa_s_per_m3: 0.0,
        };
        let piston_area = PI * config.diaphragm_radius_m.powi(2);
        let z_lumped = sealed.acoustic_impedance(40.0, medium).unwrap() * piston_area * piston_area
            + crate::baffled_piston_mechanical_radiation_impedance(
                config.diaphragm_radius_m,
                40.0,
                medium,
            )
            .unwrap();
        let error = (z_resolved - z_lumped).norm() / z_lumped.norm();
        assert!(
            error < 0.02,
            "fallback condensed load {z_resolved} vs lumped {z_lumped} (rel err {error:.3})"
        );
    }

    /// The legacy unsealed geometry remains available as an explicit opt-out:
    /// the cavity side and bottom stay open to the exterior (more fluid
    /// cells, fewer cavity-wall faces than the sealed default) and the model
    /// still solves.
    #[test]
    fn unsealed_opt_out_keeps_cavity_open_to_the_exterior() {
        let sealed = coarse_model();
        let unsealed = ResolvedAcousticModel::build(ResolvedAcousticConfig {
            seal_enclosure_walls: false,
            cells_per_physical_interval: 1,
            pml_cells: 1,
            pml_sigma_max: 2.0,
            ..ResolvedAcousticConfig::default()
        })
        .unwrap();
        assert!(unsealed.mesh.elements.len() > sealed.mesh.elements.len());
        let cavity_faces = |mesh: &Mesh| {
            mesh.boundaries
                .iter()
                .filter(|boundary| boundary.marker == CAVITY_WALL)
                .count()
        };
        assert!(cavity_faces(&unsealed.mesh) < cavity_faces(&sealed.mesh));
        let solution = unsealed
            .solve_frequency(400.0, Complex64::new(0.01, 0.0))
            .unwrap();
        assert!(
            solution
                .pressure_rms_pa
                .iter()
                .all(|value| complex_is_finite(*value))
        );
    }

    #[test]
    fn affine_pml_reconstruction_matches_direct_assembly_across_frequency() {
        let model = coarse_model();
        let probe: Vec<_> = (0..model.mesh.num_nodes())
            .map(|index| Complex64::new(1.0 + index as f64 * 0.01, -0.2))
            .collect();
        for frequency_hz in [250.0, 900.0] {
            let omega = 2.0 * PI * frequency_hz;
            let wavenumber = omega / model.config.fluid.speed_of_sound;
            let port_alpha = thermal_wall_admittance(&model.config.fluid, omega).conj();
            let affine = model.assemble_frequency_matrix(wavenumber, port_alpha, None);
            let direct = directly_assembled_matrix(&model, frequency_hz);
            let affine_action = affine.matvec(&probe);
            let direct_action = direct.matvec(&probe);
            let error = affine_action
                .iter()
                .zip(&direct_action)
                .map(|(actual, expected)| (*actual - *expected).norm_sqr())
                .sum::<f64>()
                .sqrt();
            let reference = direct_action
                .iter()
                .map(|value| value.norm_sqr())
                .sum::<f64>()
                .sqrt();
            assert!(
                error / reference.max(f64::MIN_POSITIVE) < 1.0e-12,
                "affine/direct operator defect at {frequency_hz} Hz = {}",
                error / reference.max(f64::MIN_POSITIVE)
            );
        }
    }

    #[test]
    fn mesh_resolves_every_fluid_region_and_named_boundary() {
        let model = coarse_model();
        for marker in [REAR_CAVITY, RESOLVED_PORT, PHYSICAL_EXTERIOR, ACOUSTIC_PML] {
            assert!(model.mesh.cells_with_marker(marker).next().is_some());
        }
        for marker in [
            FRONT_DIAPHRAGM,
            REAR_DIAPHRAGM,
            PORT_WALL,
            CAVITY_WALL,
            RIGID_SOLID_WALL,
            OUTER_PML_TRUNCATION,
        ] {
            assert!(
                model
                    .mesh
                    .boundaries
                    .iter()
                    .any(|face| face.marker == marker)
            );
        }
        let used: BTreeSet<_> = model
            .mesh
            .elements
            .iter()
            .flat_map(|element| element.vertices().iter().copied())
            .collect();
        assert_eq!(used.len(), model.mesh.num_nodes());
    }

    #[test]
    fn resolved_solution_is_finite_passive_and_power_consistent() {
        let model = coarse_model();
        let solution = model
            .solve_frequency(500.0, Complex64::new(0.01, 0.0))
            .unwrap();
        assert!(
            solution
                .pressure_rms_pa
                .iter()
                .all(|p| complex_is_finite(*p))
        );
        assert!(solution.port_thermal_loss_w >= -1.0e-10);
        let scale = solution.driver_input_power_w.abs().max(1.0e-12);
        assert!(
            solution.radiated_power_w >= -0.01 * scale,
            "radiated power = {}, input scale = {}, ratio = {}",
            solution.radiated_power_w,
            scale,
            solution.radiated_power_w / scale,
        );
        assert!(solution.driver_input_power_w > 0.0);
        assert!(solution.power_balance_residual_w.abs() / scale < 0.35);
        assert!(solution.port_volume_velocity_rms_m3_per_s.norm() > 0.0);
        assert!(solution.cavity_pressure_rms_pa.norm() > 0.0);
    }

    #[test]
    fn resolved_sweep_preserves_frequency_order() {
        let model = coarse_model();
        let frequencies = [400.0, 500.0, 630.0];
        let solutions = model
            .solve_sweep(&frequencies, Complex64::new(0.01, 0.0))
            .unwrap();
        assert_eq!(
            solutions
                .iter()
                .map(|solution| solution.frequency_hz)
                .collect::<Vec<_>>(),
            frequencies
        );
        assert!(
            solutions
                .iter()
                .all(|solution| complex_is_finite(solution.far_field_pressure_rms_pa))
        );
        assert!(!solutions[0].solve_diagnostics.initial_guess_reused);
        assert!(
            solutions[1..]
                .iter()
                .all(|solution| solution.solve_diagnostics.initial_guess_reused)
        );
    }

    #[test]
    fn assembled_system_exposes_checked_exact_residual() {
        let model = coarse_model();
        let drive = Complex64::new(0.01, 0.0);
        let system = model.assemble_frequency_system(500.0, drive).unwrap();
        assert_eq!(system.matrix.num_rows, model.mesh.num_nodes());
        assert_eq!(system.rhs.len(), model.mesh.num_nodes());
        assert_eq!(system.mesh_fingerprint, model.mesh_fingerprint());
        assert_eq!(system.mesh_fingerprint.len(), 64);

        let zero = vec![Complex64::default(); system.rhs.len()];
        assert!((system.exact_relative_residual(&zero).unwrap() - 1.0).abs() < 1.0e-15);
        assert!(matches!(
            system.exact_relative_residual(&zero[..zero.len() - 1]),
            Err(ResolvedSpeakerAcousticError::ExternalGuessLength { .. })
        ));
        let mut non_finite = zero;
        non_finite[0] = Complex64::new(f64::NAN, 0.0);
        assert!(matches!(
            system.exact_relative_residual(&non_finite),
            Err(ResolvedSpeakerAcousticError::NonFiniteExternalGuess)
        ));
    }

    #[test]
    fn external_guess_is_validated_and_preserves_the_converged_solution() {
        let model = coarse_model();
        let drive = Complex64::new(0.01, 0.0);
        let classical = model.solve_frequency(500.0, drive).unwrap();
        let hybrid = model
            .solve_frequency_with_external_guess(500.0, drive, Some(&classical.pressure_rms_pa))
            .unwrap();
        assert_eq!(
            hybrid.solve_diagnostics.selected_guess_kind,
            ResolvedGuessKind::External
        );
        assert!(
            hybrid
                .solve_diagnostics
                .candidate_residuals
                .external
                .unwrap()
                <= classical.solve_diagnostics.relative_residual * 1.01
        );
        let relative_difference = hybrid
            .pressure_rms_pa
            .iter()
            .zip(&classical.pressure_rms_pa)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr())
            .sum::<f64>()
            .sqrt()
            / classical
                .pressure_rms_pa
                .iter()
                .map(|value| value.norm_sqr())
                .sum::<f64>()
                .sqrt();
        assert!(relative_difference < 1.0e-10);
        assert_eq!(
            hybrid.cavity_pressure_rms_pa,
            classical.cavity_pressure_rms_pa
        );
        assert_eq!(
            hybrid.port_volume_velocity_rms_m3_per_s,
            classical.port_volume_velocity_rms_m3_per_s
        );
        assert_eq!(
            hybrid.far_field_pressure_rms_pa,
            classical.far_field_pressure_rms_pa
        );
        assert_eq!(hybrid.far_field_spl_db, classical.far_field_spl_db);
        assert_eq!(hybrid.radiated_power_w, classical.radiated_power_w);
        assert_eq!(
            hybrid.power_balance_residual_w,
            classical.power_balance_residual_w
        );

        assert!(matches!(
            model.solve_frequency_with_external_guess(500.0, drive, Some(&[Complex64::ZERO])),
            Err(ResolvedSpeakerAcousticError::ExternalGuessLength { .. })
        ));
        let mut non_finite = vec![Complex64::ZERO; model.mesh.num_nodes()];
        non_finite[0] = Complex64::new(0.0, f64::INFINITY);
        assert!(matches!(
            model.solve_frequency_with_external_guess(500.0, drive, Some(&non_finite)),
            Err(ResolvedSpeakerAcousticError::NonFiniteExternalGuess)
        ));
    }

    #[test]
    fn iterative_sweep_reuses_near_frequency_preconditioner() {
        let model = ResolvedAcousticModel::build(ResolvedAcousticConfig {
            cells_per_physical_interval: 1,
            pml_cells: 1,
            pml_sigma_max: 2.0,
            direct_solve_max_nodes: 0,
            preconditioner_reuse_band: 0.2,
            ..ResolvedAcousticConfig::default()
        })
        .unwrap();
        let solutions = model
            .solve_sweep(&[400.0, 440.0], Complex64::new(0.01, 0.0))
            .unwrap();
        assert!(solutions[0].solve_diagnostics.preconditioner_rebuilt);
        assert!(solutions[1].solve_diagnostics.initial_guess_reused);
        assert!(!solutions[1].solve_diagnostics.preconditioner_rebuilt);
        assert!(solutions[1].solve_diagnostics.relative_residual <= 1.0e-10);
    }

    #[test]
    fn resolved_acoustics_defaults_to_tight_ilut_but_retains_ilu0_opt_out() {
        assert_eq!(
            ResolvedAcousticConfig::default().iterative_preconditioner,
            ResolvedPreconditionerKind::Ilut
        );
        let matrix = directly_assembled_matrix(&coarse_model(), 500.0);
        let config = ResolvedAcousticConfig::default();
        assert!(matches!(
            ResolvedCachedPreconditioner::build(
                ResolvedPreconditionerKind::Ilu0,
                &matrix,
                None,
                &config
            )
            .unwrap()
            .0,
            ResolvedCachedPreconditioner::Ilu0 { .. }
        ));
        assert!(matches!(
            ResolvedCachedPreconditioner::build(
                ResolvedPreconditionerKind::Ilut,
                &matrix,
                None,
                &config
            )
            .unwrap()
            .0,
            ResolvedCachedPreconditioner::Ilut { .. }
        ));
    }

    #[test]
    fn adaptive_solver_configuration_is_checked() {
        let mut config = ResolvedAcousticConfig::default();
        config.ilut_fill_factor = 0;
        assert!(config.validate().is_err());
        let mut config = ResolvedAcousticConfig::default();
        config.fgmres_max_iterations = 0;
        assert!(config.validate().is_err());
        let mut config = ResolvedAcousticConfig::default();
        config.shifted_mass_coefficient = Complex64::new(0.0, 0.5);
        assert!(config.validate().is_err());
        let mut config = ResolvedAcousticConfig::default();
        config.shifted_amg_max_probe_residual_ratio = 1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn shifted_pml_operator_only_changes_the_mass_pair() {
        let model = coarse_model();
        let k = 3.0;
        let port = Complex64::new(0.2, -0.1);
        let shifted = model.assemble_shifted_laplacian_matrix(k, port, None);
        let coefficient = -model.config.shifted_mass_coefficient * k * k;
        let expected = model.affine_assembler.assemble_complex(
            &[
                Complex64::ONE,
                Complex64::I,
                coefficient,
                coefficient * Complex64::I,
            ],
            &HashMap::from([(PORT_WALL as usize, port)]),
        );
        assert_eq!(shifted.row_ptrs, expected.row_ptrs);
        assert_eq!(shifted.col_indices, expected.col_indices);
        assert_eq!(shifted.values, expected.values);
    }

    #[test]
    fn preconditioner_probe_accepts_an_exact_action() {
        let matrix =
            CsrMatrix::from_triplets(2, 2, vec![(0, 0, Complex64::ONE), (1, 1, Complex64::ONE)]);
        let ratio = optimally_scaled_preconditioner_probe_ratio(
            &matrix,
            &sonium_solvers::IdentityPreconditioner,
            &[Complex64::new(1.0, 2.0), Complex64::new(-3.0, 0.5)],
        );
        assert!(ratio <= 1.0e-15);
    }

    #[test]
    fn shifted_amg_probe_rejection_falls_back_to_ilut() {
        let model = ResolvedAcousticModel::build(ResolvedAcousticConfig {
            cells_per_physical_interval: 1,
            pml_cells: 1,
            pml_sigma_max: 2.0,
            direct_solve_max_nodes: 0,
            iterative_preconditioner: ResolvedPreconditionerKind::ShiftedLaplacianAmg,
            shifted_amg_max_probe_residual_ratio: f64::EPSILON,
            ..ResolvedAcousticConfig::default()
        })
        .unwrap();
        let solution = model
            .solve_frequency(500.0, Complex64::new(0.01, 0.0))
            .unwrap();
        assert_eq!(
            solution.solve_diagnostics.active_preconditioner,
            Some(ResolvedPreconditionerKind::Ilut)
        );
        assert!(
            solution
                .solve_diagnostics
                .preconditioner_fallbacks
                .iter()
                .any(
                    |fallback| fallback.source == ResolvedPreconditionerKind::ShiftedLaplacianAmg
                        && fallback.destination == ResolvedPreconditionerKind::Ilut
                        && fallback.reason.contains("action probe residual ratio")
                )
        );
    }

    #[test]
    fn auto_selection_requires_a_promoted_crossover() {
        let config = ResolvedAcousticConfig::default();
        assert_eq!(
            config.active_preconditioner(usize::MAX - 1),
            ResolvedPreconditionerKind::Ilut
        );
        let config = ResolvedAcousticConfig {
            iterative_preconditioner: ResolvedPreconditionerKind::Auto,
            auto_amg_crossover_nodes: 2_000,
            ..config
        };
        assert_eq!(
            config.active_preconditioner(1_999),
            ResolvedPreconditionerKind::Ilut
        );
        assert_eq!(
            config.active_preconditioner(2_000),
            ResolvedPreconditionerKind::ShiftedLaplacianAmg
        );
    }

    #[test]
    fn iterative_diagnostics_record_reuse_counters_and_storage() {
        let model = ResolvedAcousticModel::build(ResolvedAcousticConfig {
            cells_per_physical_interval: 1,
            pml_cells: 1,
            pml_sigma_max: 2.0,
            direct_solve_max_nodes: 0,
            preconditioner_reuse_band: 1.0,
            ..ResolvedAcousticConfig::default()
        })
        .unwrap();
        let mut continuation = ResolvedFrequencyContinuation::new(&model);
        let first = continuation
            .solve_frequency_with_drive(500.0, Complex64::new(0.01, 0.0))
            .unwrap();
        let second = continuation
            .solve_frequency_with_drive(510.0, Complex64::new(0.01, 0.0))
            .unwrap();
        assert_eq!(
            first.solve_diagnostics.preconditioner_reuse,
            ResolvedPreconditionerReuse::FreshBuild
        );
        assert_eq!(
            second.solve_diagnostics.preconditioner_reuse,
            ResolvedPreconditionerReuse::FullReuse
        );
        assert!(first.solve_diagnostics.preconditioner_apply_count > 0);
        assert!(second.solve_diagnostics.preconditioner_apply_count > 0);
        assert!(first.solve_diagnostics.preconditioner_nonzeros > 0);
        assert!(
            first
                .solve_diagnostics
                .preconditioner_fill_ratio
                .is_finite()
        );
        assert!(
            first
                .solve_diagnostics
                .preconditioner_estimated_storage_bytes
                > 0
        );
    }

    #[test]
    fn opt_in_preconditioner_factory_preserves_solver_contract_and_diagnostics() {
        let model = ResolvedAcousticModel::build(ResolvedAcousticConfig {
            cells_per_physical_interval: 1,
            pml_cells: 1,
            pml_sigma_max: 2.0,
            direct_solve_max_nodes: 0,
            ..ResolvedAcousticConfig::default()
        })
        .unwrap();
        let system = model
            .assemble_frequency_system(500.0, Complex64::new(0.01, 0.0))
            .unwrap();
        let mut continuation = ResolvedFrequencyContinuation::new(&model);
        let solution = continuation
            .solve_assembled_with_preconditioner_factory(&system, None, |_matrix, ilu| {
                Ok(ResolvedPreconditionerBuild {
                    preconditioner: ilu,
                    diagnostics: ResolvedPreconditionerDiagnostics {
                        learned_requested: true,
                        learned_active: false,
                        probe_residual_ratio: Some(1.01),
                        rejection_reason: Some("probe gate rejected test adapter".into()),
                    },
                })
            })
            .unwrap();
        assert!(solution.solve_diagnostics.learned_preconditioner_requested);
        assert!(!solution.solve_diagnostics.learned_preconditioner_active);
        assert_eq!(
            solution
                .solve_diagnostics
                .learned_preconditioner_probe_ratio,
            Some(1.01)
        );
        assert_eq!(
            solution
                .solve_diagnostics
                .learned_preconditioner_rejection_reason
                .as_deref(),
            Some("probe gate rejected test adapter")
        );
        assert!(solution.solve_diagnostics.relative_residual <= 1.0e-10);
        assert!(solution.solve_diagnostics.preconditioner_apply_count > 0);
        assert!(solution.solve_diagnostics.preconditioner_apply_time_seconds > 0.0);
    }

    #[test]
    fn pml_interface_reaction_closes_the_resolved_power_ledger() {
        let frequencies = [40.0, 100.0, 250.0];
        let mut config = ResolvedAcousticConfig {
            cells_per_physical_interval: 10,
            pml_cells: 12,
            radial_pml_thickness_m: 0.20,
            upper_pml_thickness_m: 0.20,
            lower_pml_thickness_m: 0.20,
            ..ResolvedAcousticConfig::default()
        };
        config
            .configure_pml_for_sweep(frequencies[0], 0.01)
            .unwrap();
        let model = ResolvedAcousticModel::build(config).unwrap();

        for solution in model
            .solve_sweep(&frequencies, Complex64::new(0.01, 0.0))
            .unwrap()
        {
            let scale = solution
                .driver_input_power_w
                .abs()
                .max(solution.port_thermal_loss_w.abs())
                .max(solution.radiated_power_w.abs())
                .max(f64::MIN_POSITIVE);
            assert!(solution.driver_input_power_w > 0.0);
            assert!(solution.radiated_power_w > 0.0);
            assert!(
                solution.power_balance_residual_w.abs() / scale < 5.0e-2,
                "reaction-flux power defect at {} Hz = {}",
                solution.frequency_hz,
                solution.power_balance_residual_w / scale
            );
        }
    }

    #[test]
    fn pml_sweep_is_power_balanced_and_stable_under_thickness_refinement() {
        let frequencies = [250.0, 500.0, 900.0];
        let mut base_config = ResolvedAcousticConfig {
            cells_per_physical_interval: 1,
            pml_cells: 2,
            ..ResolvedAcousticConfig::default()
        };
        base_config
            .configure_pml_for_sweep(frequencies[0], 0.25)
            .unwrap();
        let base = ResolvedAcousticModel::build(base_config.clone()).unwrap();

        let refined = ResolvedAcousticModel::build(ResolvedAcousticConfig {
            radial_pml_thickness_m: 0.09,
            upper_pml_thickness_m: 0.09,
            lower_pml_thickness_m: 0.09,
            pml_cells: 3,
            ..base_config
        })
        .unwrap();
        let base_solutions = base
            .solve_sweep(&frequencies, Complex64::new(0.01, 0.0))
            .unwrap();
        let refined_solutions = refined
            .solve_sweep(&frequencies, Complex64::new(0.01, 0.0))
            .unwrap();

        for (base_solution, refined_solution) in base_solutions.iter().zip(&refined_solutions) {
            for solution in [base_solution, refined_solution] {
                let input_scale = solution.driver_input_power_w.abs().max(1.0e-12);
                assert!(solution.driver_input_power_w > 0.0);
                assert!(
                    solution.power_balance_residual_w.abs() / input_scale < 0.35,
                    "power-balance defect at {} Hz = {}",
                    solution.frequency_hz,
                    solution.power_balance_residual_w / input_scale
                );
            }
            let pressure_scale = refined_solution
                .far_field_pressure_rms_pa
                .norm()
                .max(1.0e-12);
            let thickness_defect = (base_solution.far_field_pressure_rms_pa
                - refined_solution.far_field_pressure_rms_pa)
                .norm()
                / pressure_scale;
            assert!(
                thickness_defect < 0.35,
                "PML thickness defect at {} Hz = {thickness_defect}",
                base_solution.frequency_hz
            );
        }
    }

    #[test]
    fn compact_helmholtz_reference_is_bracketed_by_a_resolved_sweep() {
        let model = coarse_model();
        let reference = model.config.compact_helmholtz_frequency_hz();
        let frequencies = [0.75 * reference, reference, 1.25 * reference];
        let flows: Vec<_> = frequencies
            .iter()
            .map(|frequency| {
                model
                    .solve_frequency(*frequency, Complex64::new(0.01, 0.0))
                    .unwrap()
                    .port_volume_velocity_rms_m3_per_s
                    .norm()
            })
            .collect();
        assert!(flows.iter().all(|flow| flow.is_finite() && *flow > 0.0));
        assert!(
            flows
                .windows(2)
                .any(|pair| (pair[1] - pair[0]).abs() > 1.0e-12)
        );
        let ka_cavity = 2.0 * PI * reference * model.config.enclosure_radius_m
            / model.config.fluid.speed_of_sound;
        assert!(
            ka_cavity < 1.0,
            "lumped comparison requires a compact cavity"
        );
    }

    #[test]
    fn kirchhoff_far_field_has_inverse_distance_scaling() {
        let model = coarse_model();
        let solution = model
            .solve_frequency_with_surface_velocities(
                200.0,
                Complex64::new(0.01, 0.0),
                Complex64::default(),
            )
            .unwrap();
        let k = 2.0 * PI * solution.frequency_hz / model.config.fluid.speed_of_sound;
        let p1 = axisymmetric_kirchhoff_far_field(
            &model.mesh,
            PHYSICAL_EXTERIOR,
            ACOUSTIC_PML,
            &solution.pressure_rms_pa,
            k,
            [0.0, 0.0, 2.0],
        )
        .unwrap();
        let p2 = axisymmetric_kirchhoff_far_field(
            &model.mesh,
            PHYSICAL_EXTERIOR,
            ACOUSTIC_PML,
            &solution.pressure_rms_pa,
            k,
            [0.0, 0.0, 4.0],
        )
        .unwrap();
        let distance_ratio = p1.norm() / p2.norm();
        assert!(
            (distance_ratio - 2.0).abs() < 0.08,
            "one/two metre pressure ratio = {distance_ratio}"
        );
        let piston_reference = model.config.fluid.density
            * 2.0
            * PI
            * solution.frequency_hz
            * solution.diaphragm_velocity_rms_m_per_s.norm()
            * model.config.diaphragm_radius_m.powi(2)
            / 4.0;
        let ratio = p1.norm() / piston_reference;
        assert!(
            (0.65..=1.35).contains(&ratio),
            "finite-baffle FEM / compact baffled-piston pressure = {ratio}"
        );
    }

    #[test]
    fn vtu_and_pvd_exports_include_resolved_acoustic_fields() {
        let model = coarse_model();
        let solution = model
            .solve_frequency(400.0, Complex64::new(0.01, 0.0))
            .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let snapshot = model
            .export_solution(&solution, directory.path().join("resolved.vtu"))
            .unwrap();
        let vtu = std::fs::read_to_string(snapshot.vtu_path).unwrap();
        for name in [
            field_names::ACOUSTIC_PRESSURE,
            field_names::SOUND_PRESSURE_LEVEL,
            field_names::ACOUSTIC_INTENSITY,
            "particle_velocity",
            field_names::REGION_MARKER,
        ] {
            assert!(vtu.contains(name), "missing {name}");
        }
        let pvd_path = directory.path().join("resolved.pvd");
        model
            .export_sweep(&[300.0, 600.0], Complex64::new(0.01, 0.0), &pvd_path)
            .unwrap();
        let pvd = std::fs::read_to_string(pvd_path).unwrap();
        assert!(pvd.contains("timestep=\"300\""));
        assert!(pvd.contains("timestep=\"600\""));

        let varying = [
            model
                .solve_frequency(350.0, Complex64::new(0.01, 0.0))
                .unwrap(),
            model
                .solve_frequency(700.0, Complex64::new(0.02, 0.0))
                .unwrap(),
        ];
        let varying_pvd = directory.path().join("resolved-varying.pvd");
        model.export_solutions(&varying, &varying_pvd).unwrap();
        let pvd = std::fs::read_to_string(varying_pvd).unwrap();
        assert!(pvd.contains("timestep=\"350\""));
        assert!(pvd.contains("timestep=\"700\""));
    }

    #[test]
    fn affine_pml_reflection_is_small_and_flat_across_frequency() {
        let frequencies = [100.0, 300.0, 900.0];
        let mut base_config = ResolvedAcousticConfig {
            cells_per_physical_interval: 1,
            pml_cells: 4,
            ..ResolvedAcousticConfig::default()
        };
        base_config
            .configure_pml_for_sweep(frequencies[0], 0.05)
            .unwrap();
        // Reference: double the PML thickness at the same cell size. The
        // physical mesh is untouched, so the round-trip truncation reflection
        // is squared and the far-field difference isolates the PML reflection.
        let base = ResolvedAcousticModel::build(base_config.clone()).unwrap();
        let reference = ResolvedAcousticModel::build(ResolvedAcousticConfig {
            radial_pml_thickness_m: 2.0 * base_config.radial_pml_thickness_m,
            upper_pml_thickness_m: 2.0 * base_config.upper_pml_thickness_m,
            lower_pml_thickness_m: 2.0 * base_config.lower_pml_thickness_m,
            pml_cells: 2 * base_config.pml_cells,
            ..base_config
        })
        .unwrap();
        let drive = Complex64::new(0.01, 0.0);
        let mut reflections = Vec::new();
        for &frequency_hz in &frequencies {
            // One-sided drive: the front diaphragm radiates directly into the
            // PML-terminated exterior, like the baffled-piston benchmark.
            let base_solution = base
                .solve_frequency_with_surface_velocities(frequency_hz, drive, Complex64::default())
                .unwrap();
            let reference_solution = reference
                .solve_frequency_with_surface_velocities(frequency_hz, drive, Complex64::default())
                .unwrap();
            let reflection = (base_solution.far_field_pressure_rms_pa
                - reference_solution.far_field_pressure_rms_pa)
                .norm()
                / reference_solution
                    .far_field_pressure_rms_pa
                    .norm()
                    .max(1.0e-30);
            reflections.push(reflection);
        }
        for (&frequency_hz, &reflection) in frequencies.iter().zip(&reflections) {
            assert!(
                reflection < 0.10,
                "PML reflection at {frequency_hz} Hz = {reflection}"
            );
        }
        let spread = reflections.iter().copied().fold(f64::MIN, f64::max)
            / reflections
                .iter()
                .copied()
                .fold(f64::MAX, f64::min)
                .max(1.0e-30);
        assert!(
            spread < 4.0,
            "PML reflection spread across sweep = {spread}"
        );
    }
}
