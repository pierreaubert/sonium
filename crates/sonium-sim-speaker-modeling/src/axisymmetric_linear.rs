//! Solved axisymmetric linear loudspeaker reference model.
//!
//! This is intentionally a compact golden model: it resolves marked magnetic,
//! moving-structure, and acoustic meridional domains with P1 FEM, transfers the
//! solved bias field into a distributed reciprocal motor coupling, and solves
//! the complete voltage-driven harmonic block system.

use crate::conventions::StudyMetadata;
use crate::field_export::{
    ExportedSnapshot, FieldExportError, ProjectedCellVector, ProjectionProvenance,
    SpeakerFieldBundle, SpeakerFieldExporter, SpeakerPvdCollection, field_names,
};
use crate::power_ledger::{PowerLedger, PowerLedgerError, PowerLedgerTolerance};
use num_complex::Complex64;
use rayon::prelude::*;
use sonium_fem::assembly::{
    RegionAssemblyError, RegionMap, TripletMatrix, assemble_axi_boundary_mass,
    assemble_bilinear_by_region, assemble_bilinear_where, assemble_linear_where,
};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_homogeneous_dirichlet};
use sonium_fem::mesh::{ElementType, Mesh, PhysicalGroupError, Point, rectangular_mesh_triangles};
use sonium_fem::physics::acoustics::electroacoustics::{
    CoupledFrequencyContinuation, CoupledIterativeConfig, CoupledIterativeError,
    CoupledSolveDiagnostics, CoupledSystem, CoupledSystemError, ElectricalSubsystem,
    HarmonicCoupledPoint, harmonic_reciprocity_defect,
};
use sonium_fem::physics::acoustics::pressure::axisymmetric::{AxiLaplacianForm, AxiMassForm};
use sonium_fem::physics::electromagnetics::magnetostatics::{
    AxisymmetricAThetaForm, AzimuthalCurrentDensityForm, MU_0,
};
use sonium_fem::physics::solid::axisymmetric::{
    AxisymmetricElasticityForm, AxisymmetricVectorMassForm,
};
use sonium_fem::physics::solid::elasticity::IsotropicMaterial;
use sonium_fem::physics::solid::elasticity::viscoelastic_freq::{
    ComplexModulusModel, ConstantLossModulus, ModulusPoint, PronyBranch, PronyComplexModulus,
    TabulatedComplexModulus,
};
use sonium_fem::solver::{SolverConfig, SolverError, SolverType, solve_csr};
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};
use std::f64::consts::PI;
use std::path::Path;

pub const MAGNETIC_AIR: i32 = 10;
pub const PERMANENT_MAGNET: i32 = 11;
pub const STEEL_POLE: i32 = 12;
pub const VOICE_COIL: i32 = 13;
pub const AIR_GAP: i32 = 14;
pub const STEEL_YOKE: i32 = 15;
pub const DIAPHRAGM: i32 = 20;
pub const COIL_FORMER: i32 = 21;
pub const SURROUND: i32 = 22;
pub const SPIDER: i32 = 23;
pub const COIL_WINDING: i32 = 24;
pub const ACOUSTIC_FLUID: i32 = 30;
pub const MAGNETIC_AXIS: i32 = 110;
pub const MAGNETIC_OUTER_RADIUS: i32 = 111;
pub const MAGNETIC_LOWER_END: i32 = 112;
pub const MAGNETIC_UPPER_END: i32 = 113;
pub const STRUCTURAL_AXIS: i32 = 120;
pub const STRUCTURAL_FRAME: i32 = 121;
pub const STRUCTURAL_REAR_SURFACE: i32 = 122;
pub const STRUCTURAL_ACOUSTIC_INTERFACE: i32 = 123;
pub const ACOUSTIC_AXIS: i32 = 130;
pub const ACOUSTIC_SIDE_WALL: i32 = 131;
pub const ACOUSTIC_DIAPHRAGM_INTERFACE: i32 = 132;
pub const ACOUSTIC_RADIATION_BOUNDARY: i32 = 133;

/// Cell markers of the five moving-assembly regions, in modeling order.
pub const MOVING_ASSEMBLY_MARKERS: [i32; 5] =
    [DIAPHRAGM, COIL_FORMER, SURROUND, SPIDER, COIL_WINDING];

/// Isotropic material assigned to a moving-assembly region.
///
/// This is the single source of truth for both the stiffness assembly and the
/// post-solve stress recovery: the surround and spider are soft suspensions
/// (MPa range), the former and coil winding are stiff metals (GPa range), and
/// the diaphragm sits between. Returns `None` for markers outside the moving
/// assembly.
pub fn moving_assembly_material(marker: i32) -> Option<IsotropicMaterial> {
    match marker {
        DIAPHRAGM => Some(IsotropicMaterial::new(2.4e9, 0.32)),
        COIL_FORMER => Some(IsotropicMaterial::new(69.0e9, 0.33)),
        SURROUND => Some(IsotropicMaterial::new(5.0e6, 0.49)),
        SPIDER => Some(IsotropicMaterial::new(30.0e6, 0.45)),
        COIL_WINDING => Some(IsotropicMaterial::new(110.0e9, 0.34)),
        _ => None,
    }
}

/// Frequency-dependent modulus assigned to one moving-assembly region.
///
/// Enabling any region modulus replaces the legacy global Rayleigh
/// viscous damping (`η/ω_ref · K`) with per-region structural complex
/// stiffness `K_r · E*_r(f)/E_ref,r` in the solid block. Regions without
/// an explicit entry use a constant-loss modulus at the global
/// structural loss factor, which matches the legacy model at the
/// damping reference frequency.
#[derive(Debug, Clone, PartialEq)]
pub enum RegionModulusSpec {
    /// Constant complex modulus `E·(1 + i·η)`.
    Constant {
        /// Loss factor `η ≥ 0`.
        loss_factor: f64,
    },
    /// Tabulated `(frequency Hz, storage Pa, loss Pa)` points.
    ///
    /// Evaluation is log-frequency interpolation with no extrapolation:
    /// frequencies outside the table fail closed.
    Tabulated {
        /// At least two strictly increasing frequency points.
        points: Vec<(f64, f64, f64)>,
    },
    /// Prony series around an equilibrium modulus.
    Prony {
        /// Equilibrium (long-time) modulus in Pa.
        equilibrium_pa: f64,
        /// `(modulus Pa, relaxation time s)` branches.
        branches: Vec<(f64, f64)>,
    },
}

/// One region modulus assignment, keyed by cell marker.
#[derive(Debug, Clone, PartialEq)]
pub struct RegionModulus {
    /// Moving-assembly cell marker (see `MOVING_ASSEMBLY_MARKERS`).
    pub marker: i32,
    /// Modulus model for the region.
    pub spec: RegionModulusSpec,
}

/// Per-region stiffness triplet with its complex modulus model.
#[derive(Debug)]
pub struct SolidRegionStiffness {
    /// Cell marker of the region.
    pub marker: i32,
    /// Elastic stiffness assembled with the region reference modulus.
    pub stiffness: TripletMatrix<f64>,
    /// Reference Young's modulus the stiffness was assembled with, in Pa.
    pub reference_modulus_pa: f64,
    /// Frequency-dependent complex modulus for the region.
    pub modulus: Box<dyn ComplexModulusModel>,
}

fn build_region_modulus(
    spec: &RegionModulusSpec,
    material: IsotropicMaterial,
) -> Result<Box<dyn ComplexModulusModel>, AxisymmetricLinearError> {
    let invalid = |error: sonium_fem::physics::solid::elasticity::viscoelastic_freq::ViscoelasticError| {
        AxisymmetricLinearError::InvalidRegionModulus(format!("{error}"))
    };
    match spec {
        RegionModulusSpec::Constant { loss_factor } => Ok(Box::new(
            ConstantLossModulus::new(material.youngs_modulus, *loss_factor, material.poisson_ratio)
                .map_err(invalid)?,
        )),
        RegionModulusSpec::Tabulated { points } => Ok(Box::new(
            TabulatedComplexModulus::new(
                points
                    .iter()
                    .map(|&(frequency_hz, storage_pa, loss_pa)| ModulusPoint {
                        frequency_hz,
                        storage_pa,
                        loss_pa,
                    })
                    .collect(),
                material.poisson_ratio,
            )
            .map_err(invalid)?,
        )),
        RegionModulusSpec::Prony { equilibrium_pa, branches } => Ok(Box::new(
            PronyComplexModulus::new(
                *equilibrium_pa,
                branches
                    .iter()
                    .map(|&(modulus_pa, relaxation_time_s)| PronyBranch {
                        modulus_pa,
                        relaxation_time_s,
                    })
                    .collect(),
                material.poisson_ratio,
            )
            .map_err(invalid)?,
        )),
    }
}

#[derive(Debug, Clone)]
pub struct AxisymmetricLinearConfig {
    pub magnetic_radial_cells: usize,
    pub magnetic_axial_cells: usize,
    pub structural_radial_cells: usize,
    pub structural_axial_cells: usize,
    pub acoustic_axial_cells: usize,
    pub diaphragm_radius_m: f64,
    pub diaphragm_thickness_m: f64,
    pub acoustic_height_m: f64,
    pub sound_speed_m_per_s: f64,
    pub fluid_density_kg_per_m3: f64,
    pub coil_resistance_ohm: f64,
    pub coil_inductance_h: f64,
    pub active_turns: f64,
    pub remanence_t: f64,
    pub structural_loss_factor: f64,
    /// Optional bulk acoustic loss used only for controlled sensitivity studies.
    ///
    /// Radiation loss is supplied independently by the outgoing-wave boundary.
    pub acoustic_loss_factor: f64,
    pub damping_reference_hz: f64,
    /// Per-region frequency-dependent moduli; empty keeps legacy damping.
    /// See [`RegionModulusSpec`] for the damping-model switch.
    pub region_moduli: Vec<RegionModulus>,
    pub iterative_solver: CoupledIterativeConfig,
}

impl Default for AxisymmetricLinearConfig {
    fn default() -> Self {
        Self {
            magnetic_radial_cells: 12,
            magnetic_axial_cells: 12,
            structural_radial_cells: 10,
            structural_axial_cells: 2,
            acoustic_axial_cells: 8,
            diaphragm_radius_m: 52.5e-3,
            diaphragm_thickness_m: 1.0e-3,
            acoustic_height_m: 80.0e-3,
            sound_speed_m_per_s: 343.0,
            fluid_density_kg_per_m3: 1.204,
            coil_resistance_ohm: 5.8,
            coil_inductance_h: 0.42e-3,
            active_turns: 75.0,
            remanence_t: 1.2,
            structural_loss_factor: 0.035,
            acoustic_loss_factor: 0.0,
            damping_reference_hz: 1_000.0,
            region_moduli: Vec::new(),
            iterative_solver: CoupledIterativeConfig::default(),
        }
    }
}

impl AxisymmetricLinearConfig {
    pub fn validate(&self) -> Result<(), AxisymmetricLinearError> {
        if self.magnetic_radial_cells < 4
            || self.magnetic_axial_cells < 4
            || self.structural_radial_cells < 5
            || self.structural_axial_cells < 1
            || self.acoustic_axial_cells < 2
        {
            return Err(AxisymmetricLinearError::InvalidConfiguration(
                "mesh resolutions are below the axisymmetric reference minimum",
            ));
        }
        for value in [
            self.diaphragm_radius_m,
            self.diaphragm_thickness_m,
            self.acoustic_height_m,
            self.sound_speed_m_per_s,
            self.fluid_density_kg_per_m3,
            self.coil_resistance_ohm,
            self.coil_inductance_h,
            self.active_turns,
            self.remanence_t,
            self.damping_reference_hz,
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(AxisymmetricLinearError::InvalidConfiguration(
                    "physical dimensions and material values must be finite and positive",
                ));
            }
        }
        for value in [self.structural_loss_factor, self.acoustic_loss_factor] {
            if !value.is_finite() || value < 0.0 {
                return Err(AxisymmetricLinearError::InvalidConfiguration(
                    "loss factors must be finite and non-negative",
                ));
            }
        }
        let mut region_markers = std::collections::HashSet::new();
        for region in &self.region_moduli {
            let Some(material) = moving_assembly_material(region.marker) else {
                return Err(AxisymmetricLinearError::InvalidRegionModulus(format!(
                    "marker {} is not a moving-assembly region",
                    region.marker
                )));
            };
            if !region_markers.insert(region.marker) {
                return Err(AxisymmetricLinearError::InvalidRegionModulus(format!(
                    "duplicate assignment for marker {}",
                    region.marker
                )));
            }
            build_region_modulus(&region.spec, material)?;
        }
        self.iterative_solver.validate()?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MagneticBiasSolution {
    pub vector_potential_atheta: Vec<f64>,
    pub flux_density_by_cell_t: Vec<[f64; 3]>,
    pub flux_density_at_nodes_t: Vec<[f64; 3]>,
    pub distributed_bl_tm: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThieleSmallReference {
    pub moving_mass_kg: f64,
    pub stiffness_n_per_m: f64,
    pub mechanical_resistance_kg_per_s: f64,
    pub force_factor_tm: f64,
    pub resonance_hz: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PowerBalance {
    pub input_w: f64,
    pub copper_loss_w: f64,
    pub mechanical_loss_w: f64,
    pub acoustic_loss_w: f64,
    pub acoustic_operator_loss_w: f64,
    pub residual_w: f64,
}

impl PowerBalance {
    /// Convert the distributed-solve diagnostics to the shared cross-domain
    /// ledger. `acoustic_operator_loss_w` is a second evaluation of the same
    /// fluid loss used by `acoustic_loss_w`, so it is retained as a diagnostic
    /// above but is intentionally not counted twice in the ledger.
    pub fn power_ledger(&self) -> Result<PowerLedger, PowerLedgerError> {
        let mut ledger = PowerLedger::from_components(
            self.input_w,
            self.copper_loss_w,
            0.0,
            self.mechanical_loss_w,
            0.0,
            0.0,
            self.acoustic_loss_w,
            0.0,
        )?;
        // Keep the solver-reported residual as part of the contract rather
        // than silently recomputing it away; this catches omitted terms in a
        // future assembly path.
        ledger.residual_w = self.residual_w;
        Ok(ledger)
    }

    /// Apply the shared closure gate to this harmonic state.
    pub fn validate_power_closure(
        &self,
        tolerance: PowerLedgerTolerance,
    ) -> Result<crate::power_ledger::PowerClosure, PowerLedgerError> {
        self.power_ledger()?.close(tolerance)
    }
}

#[derive(Debug, Clone)]
pub struct AxisymmetricLinearSolution {
    pub frequency_hz: f64,
    pub voltage_rms_v: Complex64,
    pub current_rms_a: Complex64,
    pub impedance_ohm: Complex64,
    pub pressure_rms_pa: Vec<Complex64>,
    pub displacement_rz_m: Vec<[Complex64; 2]>,
    pub velocity_rz_m_per_s: Vec<[Complex64; 2]>,
    pub observation_spl_db: f64,
    pub power: PowerBalance,
    pub solver: CoupledSolveDiagnostics,
}

/// Sweep-wide correctness gates for the distributed linear model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AxisymmetricSweepQa {
    pub max_relative_linear_residual: f64,
    pub max_relative_power_residual: f64,
    pub max_reciprocity_defect: f64,
    pub minimum_impedance_real_ohm: f64,
    pub minimum_dissipated_power_w: f64,
    pub all_points_converged: bool,
}

impl AxisymmetricSweepQa {
    pub fn passes(&self, residual_tolerance: f64, power_tolerance: f64) -> bool {
        self.all_points_converged
            && self.max_relative_linear_residual <= residual_tolerance
            && self.max_relative_power_residual <= power_tolerance
            && self.max_reciprocity_defect <= 1.0e-12
            && self.minimum_impedance_real_ohm >= -1.0e-10
            && self.minimum_dissipated_power_w >= -1.0e-10
    }
}

pub struct AxisymmetricLinearModel {
    pub config: AxisymmetricLinearConfig,
    pub magnetic_mesh: Mesh,
    pub structural_mesh: Mesh,
    pub acoustic_mesh: Mesh,
    pub magnetic_bias: MagneticBiasSolution,
    pub coupled_system: CoupledSystem,
    /// Per-region stiffness for the complex-stiffness path.
    ///
    /// Empty when no region moduli are configured; in that case
    /// `coupled_system` is used unchanged at every frequency.
    pub solid_regions: Vec<SolidRegionStiffness>,
    pub thiele_small: ThieleSmallReference,
    structural_constraints: Vec<usize>,
}

impl AxisymmetricLinearModel {
    pub fn build(config: AxisymmetricLinearConfig) -> Result<Self, AxisymmetricLinearError> {
        config.validate()?;
        let (magnetic_mesh, magnetic_bias) = build_magnetic_bias(&config)?;
        let structural_mesh = build_structural_mesh(&config)?;
        let acoustic_mesh = build_acoustic_mesh(&config)?;
        let scalar_space = H1Space::new(PolynomialDegree::P1);
        let vector_space = VectorH1Space::new(PolynomialDegree::P1, 2);

        let mut elastic_forms = RegionMap::new();
        for marker in MOVING_ASSEMBLY_MARKERS {
            let material = moving_assembly_material(marker)
                .expect("every moving-assembly marker has an assigned material");
            elastic_forms.insert(marker, AxisymmetricElasticityForm::new(material));
        }
        let solid_stiffness = assemble_bilinear_by_region(
            &elastic_forms,
            &vector_space,
            &vector_space,
            &structural_mesh,
            2,
        )?;
        // Per-region triplets for the complex-stiffness path. Assembled
        // only when region moduli are configured, so the legacy model
        // pays no extra assembly or memory.
        let mut solid_regions = Vec::new();
        if !config.region_moduli.is_empty() {
            for marker in MOVING_ASSEMBLY_MARKERS {
                let material = moving_assembly_material(marker)
                    .expect("every moving-assembly marker has an assigned material");
                let form = AxisymmetricElasticityForm::new(material);
                let stiffness = assemble_bilinear_where(
                    &form,
                    &vector_space,
                    &vector_space,
                    &structural_mesh,
                    2,
                    |_, element| element.marker == marker,
                );
                let spec = config
                    .region_moduli
                    .iter()
                    .find(|region| region.marker == marker)
                    .map(|region| region.spec.clone())
                    .unwrap_or(RegionModulusSpec::Constant {
                        loss_factor: config.structural_loss_factor,
                    });
                solid_regions.push(SolidRegionStiffness {
                    marker,
                    stiffness,
                    reference_modulus_pa: material.youngs_modulus,
                    modulus: build_region_modulus(&spec, material)?,
                });
            }
        }
        let mut mass_forms = RegionMap::new();
        mass_forms.insert(DIAPHRAGM, AxisymmetricVectorMassForm::new(1_150.0));
        mass_forms.insert(COIL_FORMER, AxisymmetricVectorMassForm::new(2_700.0));
        mass_forms.insert(SURROUND, AxisymmetricVectorMassForm::new(1_100.0));
        mass_forms.insert(SPIDER, AxisymmetricVectorMassForm::new(900.0));
        mass_forms.insert(COIL_WINDING, AxisymmetricVectorMassForm::new(8_960.0));
        let solid_mass = assemble_bilinear_by_region(
            &mass_forms,
            &vector_space,
            &vector_space,
            &structural_mesh,
            2,
        )?;
        let omega_reference = 2.0 * PI * config.damping_reference_hz;
        let solid_damping = scaled_triplet(
            &solid_stiffness,
            config.structural_loss_factor / omega_reference,
        );

        // Convention unification: the sonium-fem duct forms `AxiLaplacianForm`
        // / `AxiMassForm` (and `assemble_axi_boundary_mass`) are per-radian
        // (`∫ r … dr dz`), while the structural forms, the motor row, and the
        // diaphragm coupling matrix `C` all carry the full `2πr` axisymmetric
        // weight. The monolithic system pairs the fluid block with `C` in
        // both off-diagonal blocks, so the fluid block must share the 2π
        // convention: scaling all three fluid matrices by 2π makes the
        // stage-3 pressure the true physical pressure (previously it ran a
        // factor 2π hot, inflating `Z_compact`, the observation SPL, and the
        // acoustic power terms by the same factor; the coupled results at the
        // diaphragm were only correct because the resolved-coupling ΔZ
        // subtraction cancelled the inflation).
        let mut acoustic_k_forms = RegionMap::new();
        acoustic_k_forms.insert(ACOUSTIC_FLUID, AxiLaplacianForm);
        let fluid_stiffness = scaled_triplet(
            &assemble_bilinear_by_region(
                &acoustic_k_forms,
                &scalar_space,
                &scalar_space,
                &acoustic_mesh,
                2,
            )?,
            2.0 * PI,
        );
        let mut acoustic_m_forms = RegionMap::new();
        acoustic_m_forms.insert(ACOUSTIC_FLUID, AxiMassForm);
        let fluid_mass = scaled_triplet(
            &assemble_bilinear_by_region(
                &acoustic_m_forms,
                &scalar_space,
                &scalar_space,
                &acoustic_mesh,
                2,
            )?,
            2.0 * PI,
        );
        let mut fluid_damping = scaled_triplet(
            &fluid_mass,
            2.0 * config.acoustic_loss_factor * omega_reference
                / config.sound_speed_m_per_s.powi(2),
        );
        // First-order outgoing-wave condition at the truncated axial exterior:
        //   ∂p/∂n = -i k p  (positive-iω convention).
        // In K - k²M + iωD this is D = M_boundary / c. Because it is part of
        // the same pressure/displacement block system, its radiation impedance
        // loads the structure reciprocally rather than being post-processed.
        let radiation_mass = assemble_axi_boundary_mass(
            &acoustic_mesh,
            PolynomialDegree::P1,
            ACOUSTIC_RADIATION_BOUNDARY,
        );
        if radiation_mass.nnz() == 0 {
            return Err(AxisymmetricLinearError::MissingRadiationBoundary);
        }
        for index in 0..radiation_mass.nnz() {
            fluid_damping.push(
                radiation_mass.rows[index],
                radiation_mass.cols[index],
                // `assemble_axi_boundary_mass` is per-radian; lift it to the
                // same 2π-weighted convention as the rest of the fluid block.
                2.0 * PI * radiation_mass.values[index] / config.sound_speed_m_per_s,
            );
        }
        let coupling = assemble_diaphragm_acoustic_coupling(
            &structural_mesh,
            &acoustic_mesh,
            config.structural_radial_cells,
        );
        let motor_coupling = distributed_motor_row(
            &structural_mesh,
            magnetic_bias.distributed_bl_tm,
            COIL_WINDING,
        )?;
        let resistance = scalar_triplet(config.coil_resistance_ohm);
        let inductance = scalar_triplet(config.coil_inductance_h);
        let coupled_system = CoupledSystem {
            n_fluid: scalar_space.num_global_dofs(&acoustic_mesh),
            n_solid: vector_space.num_global_dofs(&structural_mesh),
            fluid_stiffness,
            fluid_mass,
            fluid_damping,
            solid_stiffness,
            solid_mass,
            solid_damping,
            coupling,
            interface_load: None,
            electrical: Some(ElectricalSubsystem {
                n_dofs: 1,
                resistance,
                inductance,
                coupling: motor_coupling,
            }),
        };
        coupled_system.validate()?;
        let structural_constraints =
            structural_constraint_dofs(&structural_mesh, config.diaphragm_radius_m);
        let thiele_small =
            extract_thiele_small(&coupled_system, &structural_mesh, config.diaphragm_radius_m);
        Ok(Self {
            config,
            magnetic_mesh,
            structural_mesh,
            acoustic_mesh,
            magnetic_bias,
            coupled_system,
            solid_regions,
            thiele_small,
            structural_constraints,
        })
    }

    /// Builds the coupled system at one frequency.
    ///
    /// Without region moduli this clones the base system unchanged.
    /// With region moduli the solid block becomes the per-region
    /// complex stiffness `Σ_r K_r·E*_r(f)/E_ref,r`, split into the real
    /// stiffness triplet and the damping triplet `Σ_r K_r·Im/ω` so the
    /// assembled `K + iωC` block equals the complex stiffness exactly.
    /// Allocation happens once per frequency on this path only; the
    /// legacy path never calls it in a hot loop.
    ///
    /// # Errors
    ///
    /// Returns an error when a region modulus fails closed at
    /// `frequency_hz` (for example outside a tabulated range).
    pub fn coupled_system_at(
        &self,
        frequency_hz: f64,
    ) -> Result<CoupledSystem, AxisymmetricLinearError> {
        let base = &self.coupled_system;
        if self.solid_regions.is_empty() {
            return Ok(CoupledSystem {
                n_fluid: base.n_fluid,
                n_solid: base.n_solid,
                fluid_stiffness: base.fluid_stiffness.clone(),
                fluid_mass: base.fluid_mass.clone(),
                fluid_damping: base.fluid_damping.clone(),
                solid_stiffness: base.solid_stiffness.clone(),
                solid_mass: base.solid_mass.clone(),
                solid_damping: base.solid_damping.clone(),
                coupling: base.coupling.clone(),
                interface_load: base.interface_load.clone(),
                electrical: base.electrical.as_ref().map(|electrical| ElectricalSubsystem {
                    n_dofs: electrical.n_dofs,
                    resistance: electrical.resistance.clone(),
                    inductance: electrical.inductance.clone(),
                    coupling: electrical.coupling.clone(),
                }),
            });
        }
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(AxisymmetricLinearError::InvalidConfiguration(
                "frequency must be finite and positive",
            ));
        }
        let omega = 2.0 * PI * frequency_hz;
        let capacity: usize = self.solid_regions.iter().map(|region| region.stiffness.nnz()).sum();
        let mut solid_stiffness =
            TripletMatrix::with_capacity(base.solid_stiffness.num_rows, base.solid_stiffness.num_cols, capacity);
        let mut solid_damping =
            TripletMatrix::with_capacity(base.solid_damping.num_rows, base.solid_damping.num_cols, capacity);
        for region in &self.solid_regions {
            let factor = region
                .modulus
                .stiffness_factor(frequency_hz, region.reference_modulus_pa)
                .map_err(|error| {
                    AxisymmetricLinearError::InvalidRegionModulus(format!(
                        "marker {}: {error}",
                        region.marker
                    ))
                })?;
            let matrix = &region.stiffness;
            for index in 0..matrix.nnz() {
                let value = matrix.values[index];
                solid_stiffness.push(matrix.rows[index], matrix.cols[index], value * factor.re);
                solid_damping.push(matrix.rows[index], matrix.cols[index], value * factor.im / omega);
            }
        }
        Ok(CoupledSystem {
            n_fluid: base.n_fluid,
            n_solid: base.n_solid,
            fluid_stiffness: base.fluid_stiffness.clone(),
            fluid_mass: base.fluid_mass.clone(),
            fluid_damping: base.fluid_damping.clone(),
            solid_stiffness,
            solid_mass: base.solid_mass.clone(),
            solid_damping,
            coupling: base.coupling.clone(),
            interface_load: base.interface_load.clone(),
            electrical: base.electrical.as_ref().map(|electrical| ElectricalSubsystem {
                n_dofs: electrical.n_dofs,
                resistance: electrical.resistance.clone(),
                inductance: electrical.inductance.clone(),
                coupling: electrical.coupling.clone(),
            }),
        })
    }

    /// Solid damping triplet at one frequency for power accounting.
    fn solid_damping_at(
        &self,
        frequency_hz: f64,
    ) -> Result<TripletMatrix<f64>, AxisymmetricLinearError> {
        if self.solid_regions.is_empty() {
            return Ok(self.coupled_system.solid_damping.clone());
        }
        Ok(self.coupled_system_at(frequency_hz)?.solid_damping)
    }

    pub fn solve_frequency(
        &self,
        frequency_hz: f64,
        voltage_rms_v: Complex64,
    ) -> Result<AxisymmetricLinearSolution, AxisymmetricLinearError> {
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 || !complex_is_finite(voltage_rms_v) {
            return Err(AxisymmetricLinearError::InvalidConfiguration(
                "frequency must be positive and voltage must be finite",
            ));
        }
        if !self.solid_regions.is_empty() {
            // Per-frequency system: continuation warm starts do not
            // carry across different solid blocks, so each point owns
            // its solver. Tabulated moduli fail closed here.
            let system = self.coupled_system_at(frequency_hz)?;
            let mut solver = CoupledFrequencyContinuation::new(
                &system,
                self.config.iterative_solver.clone(),
            )?;
            return self.solve_with_continuation(frequency_hz, voltage_rms_v, &mut solver);
        }
        let mut solver = CoupledFrequencyContinuation::new(
            &self.coupled_system,
            self.config.iterative_solver.clone(),
        )?;
        self.solve_with_continuation(frequency_hz, voltage_rms_v, &mut solver)
    }

    /// Solve a frequency sequence with continuation guesses and reusable block factors.
    pub fn solve_sweep(
        &self,
        frequencies_hz: &[f64],
        voltage_rms_v: Complex64,
    ) -> Result<Vec<AxisymmetricLinearSolution>, AxisymmetricLinearError> {
        self.solve_sweep_with_progress(frequencies_hz, voltage_rms_v, |_, _| {})
    }

    /// Solve contiguous frequency lanes in parallel. Each lane owns its
    /// continuation state, while indexed collection restores input order.
    pub fn solve_sweep_parallel(
        &self,
        frequencies_hz: &[f64],
        voltage_rms_v: Complex64,
    ) -> Result<Vec<AxisymmetricLinearSolution>, AxisymmetricLinearError> {
        if frequencies_hz.is_empty() {
            return Err(AxisymmetricLinearError::InvalidConfiguration(
                "a frequency sweep must contain at least one point",
            ));
        }
        if !self.solid_regions.is_empty() {
            // Per-frequency solid blocks: each point solves independently.
            return frequencies_hz
                .iter()
                .map(|&frequency_hz| self.solve_frequency(frequency_hz, voltage_rms_v))
                .collect();
        }
        let state_dofs = self.magnetic_mesh.num_nodes()
            + self.structural_mesh.num_nodes().saturating_mul(2)
            + self.acoustic_mesh.num_nodes();
        let estimated_bytes_per_lane = state_dofs.saturating_mul(16).saturating_mul(64);
        let lane_count =
            crate::parallelism::resolved_lane_count(frequencies_hz.len(), estimated_bytes_per_lane);
        let lanes = crate::parallelism::contiguous_lane_ranges(frequencies_hz.len(), lane_count);
        let lane_solutions = lanes
            .into_par_iter()
            .map(|range| {
                let mut solver = CoupledFrequencyContinuation::new(
                    &self.coupled_system,
                    self.config.iterative_solver.clone(),
                )?;
                frequencies_hz[range]
                    .iter()
                    .map(|&frequency_hz| {
                        self.solve_with_continuation(frequency_hz, voltage_rms_v, &mut solver)
                    })
                    .collect::<Result<Vec<_>, AxisymmetricLinearError>>()
            })
            .collect::<Result<Vec<_>, AxisymmetricLinearError>>()?;
        Ok(lane_solutions.into_iter().flatten().collect())
    }

    /// Solve a frequency sequence and report each completed frequency after
    /// its continuation solve has passed the native error boundary.
    pub fn solve_sweep_with_progress<F>(
        &self,
        frequencies_hz: &[f64],
        voltage_rms_v: Complex64,
        mut on_frequency_complete: F,
    ) -> Result<Vec<AxisymmetricLinearSolution>, AxisymmetricLinearError>
    where
        F: FnMut(usize, f64),
    {
        if frequencies_hz.is_empty() {
            return Err(AxisymmetricLinearError::InvalidConfiguration(
                "a frequency sweep must contain at least one point",
            ));
        }
        if !self.solid_regions.is_empty() {
            let mut solutions = Vec::with_capacity(frequencies_hz.len());
            for (index, &frequency_hz) in frequencies_hz.iter().enumerate() {
                let solution = self.solve_frequency(frequency_hz, voltage_rms_v)?;
                on_frequency_complete(index + 1, frequency_hz);
                solutions.push(solution);
            }
            return Ok(solutions);
        }
        let mut solver = CoupledFrequencyContinuation::new(
            &self.coupled_system,
            self.config.iterative_solver.clone(),
        )?;
        let mut solutions = Vec::with_capacity(frequencies_hz.len());
        for (index, &frequency_hz) in frequencies_hz.iter().enumerate() {
            let solution =
                self.solve_with_continuation(frequency_hz, voltage_rms_v, &mut solver)?;
            on_frequency_complete(index + 1, frequency_hz);
            solutions.push(solution);
        }
        Ok(solutions)
    }

    /// Compare production sweep outputs against convergence, passivity,
    /// reciprocity, and RMS power-closure invariants.
    pub fn audit_sweep(
        &self,
        solutions: &[AxisymmetricLinearSolution],
    ) -> Result<AxisymmetricSweepQa, AxisymmetricLinearError> {
        if solutions.is_empty() {
            return Err(AxisymmetricLinearError::InvalidConfiguration(
                "cannot audit an empty frequency sweep",
            ));
        }
        let mut qa = AxisymmetricSweepQa {
            max_relative_linear_residual: 0.0,
            max_relative_power_residual: 0.0,
            max_reciprocity_defect: 0.0,
            minimum_impedance_real_ohm: f64::INFINITY,
            minimum_dissipated_power_w: f64::INFINITY,
            all_points_converged: true,
        };
        for solution in solutions {
            let point = self.harmonic_point(solution.frequency_hz)?;
            qa.max_relative_linear_residual = qa
                .max_relative_linear_residual
                .max(solution.solver.relative_residual);
            qa.max_relative_power_residual = qa
                .max_relative_power_residual
                .max(solution.power.residual_w.abs() / solution.power.input_w.abs().max(1.0e-15));
            solution
                .power
                .validate_power_closure(PowerLedgerTolerance {
                    absolute_w: 1.0e-8,
                    relative: 1.0e-6,
                })?;
            qa.max_reciprocity_defect = qa
                .max_reciprocity_defect
                .max(harmonic_reciprocity_defect(&self.coupled_system, point)?);
            qa.minimum_impedance_real_ohm =
                qa.minimum_impedance_real_ohm.min(solution.impedance_ohm.re);
            qa.minimum_dissipated_power_w = qa.minimum_dissipated_power_w.min(
                solution
                    .power
                    .copper_loss_w
                    .min(solution.power.mechanical_loss_w)
                    .min(solution.power.acoustic_loss_w)
                    .min(solution.power.acoustic_operator_loss_w),
            );
            qa.all_points_converged &= solution.solver.converged;
        }
        Ok(qa)
    }

    fn solve_with_continuation(
        &self,
        frequency_hz: f64,
        voltage_rms_v: Complex64,
        solver: &mut CoupledFrequencyContinuation<'_>,
    ) -> Result<AxisymmetricLinearSolution, AxisymmetricLinearError> {
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 || !complex_is_finite(voltage_rms_v) {
            return Err(AxisymmetricLinearError::InvalidConfiguration(
                "frequency must be positive and voltage must be finite",
            ));
        }
        let point = self.harmonic_point(frequency_hz)?;
        let mut rhs = vec![Complex64::new(0.0, 0.0); self.coupled_system.total_dofs()];
        rhs[self.coupled_system.n_fluid + self.coupled_system.n_solid] = voltage_rms_v;
        let constraints = self.global_structural_constraints();
        let solved = solver.solve(point, &rhs, &constraints)?;
        self.extract_solution(
            frequency_hz,
            voltage_rms_v,
            &solved.values,
            solved.diagnostics,
        )
    }

    fn extract_solution(
        &self,
        frequency_hz: f64,
        voltage_rms_v: Complex64,
        values: &[Complex64],
        diagnostics: CoupledSolveDiagnostics,
    ) -> Result<AxisymmetricLinearSolution, AxisymmetricLinearError> {
        let omega = 2.0 * PI * frequency_hz;
        let nf = self.coupled_system.n_fluid;
        let ns = self.coupled_system.n_solid;
        let pressure_rms_pa = values[..nf].to_vec();
        let displacement_flat = &values[nf..nf + ns];
        let displacement_rz_m: Vec<_> = displacement_flat
            .as_chunks::<2>()
            .0
            .iter()
            .map(|value| [value[0], value[1]])
            .collect();
        let velocity_rz_m_per_s: Vec<_> = displacement_rz_m
            .iter()
            .map(|value| {
                [
                    Complex64::new(0.0, omega) * value[0],
                    Complex64::new(0.0, omega) * value[1],
                ]
            })
            .collect();
        let current_rms_a = values[nf + ns];
        let observation = pressure_rms_pa
            .iter()
            .zip(&self.acoustic_mesh.nodes)
            .max_by(|(_, a), (_, b)| a.y.total_cmp(&b.y).then_with(|| b.x.total_cmp(&a.x)))
            .map(|(pressure, _)| *pressure)
            .unwrap_or_default();
        let observation_spl_db = 20.0 * (observation.norm().max(1.0e-15) / 20.0e-6).log10();
        let power = self.power_balance(
            voltage_rms_v,
            current_rms_a,
            &pressure_rms_pa,
            displacement_flat,
            omega,
        );
        Ok(AxisymmetricLinearSolution {
            frequency_hz,
            voltage_rms_v,
            current_rms_a,
            impedance_ohm: voltage_rms_v / current_rms_a,
            pressure_rms_pa,
            displacement_rz_m,
            velocity_rz_m_per_s,
            observation_spl_db,
            power,
            solver: diagnostics,
        })
    }

    fn harmonic_point(
        &self,
        frequency_hz: f64,
    ) -> Result<HarmonicCoupledPoint, AxisymmetricLinearError> {
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(AxisymmetricLinearError::InvalidConfiguration(
                "frequency must be finite and positive",
            ));
        }
        Ok(HarmonicCoupledPoint {
            omega_rad_per_s: 2.0 * PI * frequency_hz,
            sound_speed_m_per_s: self.config.sound_speed_m_per_s,
            fluid_density_kg_per_m3: self.config.fluid_density_kg_per_m3,
        })
    }

    fn global_structural_constraints(&self) -> Vec<usize> {
        self.structural_constraints
            .iter()
            .map(|dof| self.coupled_system.n_fluid + dof)
            .collect()
    }

    /// Solve one production harmonic point with the deterministic direct
    /// CPU/f64 reference path.  CUDA evidence producers use this oracle when
    /// the default iterative continuation is not a converged reference for a
    /// particular bounded mesh/frequency cell.
    pub fn solve_frequency_direct_reference(
        &self,
        frequency_hz: f64,
        voltage_rms_v: Complex64,
    ) -> Result<AxisymmetricLinearSolution, AxisymmetricLinearError> {
        let mut matrix = self
            .coupled_system_at(frequency_hz)?
            .assemble_harmonic(self.harmonic_point(frequency_hz)?)?;
        let mut rhs = vec![Complex64::new(0.0, 0.0); self.coupled_system.total_dofs()];
        rhs[self.coupled_system.n_fluid + self.coupled_system.n_solid] = voltage_rms_v;
        apply_complex_constraints(&mut matrix, &mut rhs, &self.global_structural_constraints());
        let config = SolverConfig {
            solver_type: SolverType::Direct,
            ..SolverConfig::default()
        };
        let solved = solve_csr(&matrix.to_csr(), &rhs, &config)?;
        self.extract_solution(
            frequency_hz,
            voltage_rms_v,
            &solved.values,
            CoupledSolveDiagnostics {
                iterations: solved.iterations,
                relative_residual: solved.residual,
                converged: solved.converged,
                initial_guess_reused: false,
                preconditioner_rebuilt: false,
                defect_corrections: 0,
                residual_before_correction: solved.residual,
            },
        )
    }

    /// Install the normalized structural acoustic-interface load distribution
    /// used to inject a condensed external acoustic load (resolved enclosure,
    /// port, and radiation stage) into the solid block.
    ///
    /// The distribution is `2π·∫_Γ r φ_i φ_j ds` over the axial interface
    /// DOFs, divided by the total interface area, so its entries sum to one
    /// and the scale passed to
    /// [`CoupledSystem::assemble_harmonic_with_load`] is directly the piston
    /// load impedance `F / v_mean`. Idempotent; the model solves unchanged
    /// until a nonzero load scale is supplied.
    /// Install the normalized, scalar piston-interface load used by resolved
    /// acoustic coupling.
    ///
    /// This is deliberately a rank-1 closure: the resolved axisymmetric
    /// stage receives only the area-averaged diaphragm velocity and feeds one
    /// complex impedance back through this fixed spatial distribution. It is
    /// not a distributed modal impedance map; use the 3-D FEM/BEM coupling
    /// module when aperture modes or spatially varying loads matter.
    pub fn install_interface_load(&mut self) -> Result<(), AxisymmetricLinearError> {
        let boundary_mass = assemble_axi_boundary_mass(
            &self.structural_mesh,
            PolynomialDegree::P1,
            STRUCTURAL_ACOUSTIC_INTERFACE,
        );
        if boundary_mass.nnz() == 0 {
            return Err(AxisymmetricLinearError::MissingStructuralAcousticInterface);
        }
        let area_m2 = 2.0 * PI * boundary_mass.values.iter().sum::<f64>();
        if !area_m2.is_finite() || area_m2 <= 0.0 {
            return Err(AxisymmetricLinearError::MissingStructuralAcousticInterface);
        }
        let n_solid = self.coupled_system.n_solid;
        let mut load = TripletMatrix::with_capacity(n_solid, n_solid, boundary_mass.nnz());
        for index in 0..boundary_mass.nnz() {
            // Axial (z) vector DOFs only: the interface pressure acts along
            // the piston normal. The 2π factor matches the structural weak
            // form's axisymmetric weighting.
            load.push(
                2 * boundary_mass.rows[index] + 1,
                2 * boundary_mass.cols[index] + 1,
                2.0 * PI * boundary_mass.values[index] / area_m2,
            );
        }
        self.coupled_system.interface_load = Some(load);
        self.coupled_system.validate()?;
        Ok(())
    }

    /// Total piston area of the structural acoustic interface.
    pub fn interface_area_m2(&self) -> Result<f64, AxisymmetricLinearError> {
        let mut area = 0.0;
        for boundary in self
            .structural_mesh
            .boundaries
            .iter()
            .filter(|boundary| boundary.marker == STRUCTURAL_ACOUSTIC_INTERFACE)
        {
            if boundary.nodes.len() != 2 {
                continue;
            }
            let a = &self.structural_mesh.nodes[boundary.nodes[0]];
            let b = &self.structural_mesh.nodes[boundary.nodes[1]];
            area += 2.0 * PI * 0.5 * (a.x + b.x) * (b.x - a.x).hypot(b.y - a.y);
        }
        if !area.is_finite() || area <= 0.0 {
            return Err(AxisymmetricLinearError::MissingStructuralAcousticInterface);
        }
        Ok(area)
    }

    /// Area-averaged axial diaphragm velocity over the structural acoustic
    /// interface (2πr-weighted), the piston velocity that drives the
    /// resolved acoustic stage.
    pub fn area_averaged_diaphragm_velocity(
        &self,
        solution: &AxisymmetricLinearSolution,
    ) -> Result<Complex64, AxisymmetricLinearError> {
        let mut weighted_velocity = Complex64::default();
        let mut area = 0.0;
        for boundary in self
            .structural_mesh
            .boundaries
            .iter()
            .filter(|boundary| boundary.marker == STRUCTURAL_ACOUSTIC_INTERFACE)
        {
            if boundary.nodes.len() != 2 {
                continue;
            }
            let a = &self.structural_mesh.nodes[boundary.nodes[0]];
            let b = &self.structural_mesh.nodes[boundary.nodes[1]];
            let edge_length = (b.x - a.x).hypot(b.y - a.y);
            let radius = 0.5 * (a.x + b.x);
            let surface_area = 2.0 * PI * radius * edge_length;
            let velocity = (solution.velocity_rz_m_per_s[boundary.nodes[0]][1]
                + solution.velocity_rz_m_per_s[boundary.nodes[1]][1])
                * 0.5;
            weighted_velocity += surface_area * velocity;
            area += surface_area;
        }
        if !area.is_finite() || area <= 0.0 {
            return Err(AxisymmetricLinearError::MissingStructuralAcousticInterface);
        }
        Ok(weighted_velocity / area)
    }

    /// Total axial force the compact front fluid domain applies to the
    /// structure, `Σ (Cᵀ p)` over the structural DOFs. A positive value acts
    /// in +z; the load opposing the diaphragm motion is its negation.
    pub fn compact_fluid_interface_force_n(
        &self,
        solution: &AxisymmetricLinearSolution,
    ) -> Complex64 {
        transpose_apply_real(&self.coupled_system.coupling, &solution.pressure_rms_pa)
            .iter()
            .sum()
    }

    /// Solve one frequency with a condensed external interface load scale
    /// `zeta` (piston load impedance `F / v_mean`, exp(+iωt)) through a
    /// caller-owned continuation, so ILU factors and warm starts survive
    /// fixed-point iterations. Requires [`Self::install_interface_load`].
    pub fn solve_with_continuation_and_load(
        &self,
        frequency_hz: f64,
        voltage_rms_v: Complex64,
        solver: &mut CoupledFrequencyContinuation<'_>,
        interface_load_scale: Complex64,
    ) -> Result<AxisymmetricLinearSolution, AxisymmetricLinearError> {
        if self.coupled_system.interface_load.is_none() {
            return Err(AxisymmetricLinearError::InvalidConfiguration(
                "interface load injection requires install_interface_load first",
            ));
        }
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 || !complex_is_finite(voltage_rms_v) {
            return Err(AxisymmetricLinearError::InvalidConfiguration(
                "frequency must be positive and voltage must be finite",
            ));
        }
        let point = self.harmonic_point(frequency_hz)?;
        let mut rhs = vec![Complex64::new(0.0, 0.0); self.coupled_system.total_dofs()];
        rhs[self.coupled_system.n_fluid + self.coupled_system.n_solid] = voltage_rms_v;
        let constraints = self.global_structural_constraints();
        let solved = solver.solve_with_load(point, &rhs, &constraints, interface_load_scale)?;
        self.extract_solution(
            frequency_hz,
            voltage_rms_v,
            &solved.values,
            solved.diagnostics,
        )
    }

    pub fn reduced_mechanical_impedance(&self, frequency_hz: f64) -> Complex64 {
        let omega = 2.0 * PI * frequency_hz;
        Complex64::new(
            self.thiele_small.mechanical_resistance_kg_per_s,
            omega * self.thiele_small.moving_mass_kg - self.thiele_small.stiffness_n_per_m / omega,
        )
    }

    pub fn export_solution(
        &self,
        solution: &AxisymmetricLinearSolution,
        path: impl AsRef<Path>,
    ) -> Result<ExportedSnapshot, AxisymmetricLinearError> {
        let (mesh, fields) = self.combined_export_fields(solution)?;
        let mut metadata = StudyMetadata::new("axisymmetric linear distributed FEM");
        metadata.frequency_hz = Some(solution.frequency_hz);
        metadata.drive_voltage_rms_v = Some(solution.voltage_rms_v.norm());
        metadata.reference_case = Some("axisymmetric_linear_golden_model".to_owned());
        Ok(SpeakerFieldExporter::new(&mesh, metadata, &fields).write_snapshot(path)?)
    }

    /// Solve and export a frequency sweep as VTU frames plus a PVD collection.
    pub fn export_sweep(
        &self,
        frequencies_hz: &[f64],
        voltage_rms_v: Complex64,
        pvd_path: impl AsRef<Path>,
    ) -> Result<Vec<AxisymmetricLinearSolution>, AxisymmetricLinearError> {
        let mut collection = SpeakerPvdCollection::new(pvd_path)?;
        let solutions = self.solve_sweep(frequencies_hz, voltage_rms_v)?;
        for solution in &solutions {
            let frequency_hz = solution.frequency_hz;
            let (mesh, fields) = self.combined_export_fields(solution)?;
            let mut metadata = StudyMetadata::new("axisymmetric linear distributed FEM sweep");
            metadata.frequency_hz = Some(frequency_hz);
            metadata.drive_voltage_rms_v = Some(voltage_rms_v.norm());
            metadata.reference_case = Some("axisymmetric_linear_golden_model".to_owned());
            collection.push(&SpeakerFieldExporter::new(&mesh, metadata, &fields))?;
        }
        collection.finish()?;
        Ok(solutions)
    }

    pub fn combined_export_fields(
        &self,
        solution: &AxisymmetricLinearSolution,
    ) -> Result<(Mesh, SpeakerFieldBundle), AxisymmetricLinearError> {
        let mesh = combine_meshes(&[
            &self.magnetic_mesh,
            &self.structural_mesh,
            &self.acoustic_mesh,
        ])?;
        let nm = self.magnetic_mesh.num_nodes();
        let ns = self.structural_mesh.num_nodes();
        let na = self.acoustic_mesh.num_nodes();
        let cm = self.magnetic_mesh.num_elements();
        let cs = self.structural_mesh.num_elements();
        let ca = self.acoustic_mesh.num_elements();
        let mut fields = SpeakerFieldBundle::new();

        let mut potential = vec![0.0; nm + ns + na];
        potential[..nm].copy_from_slice(&self.magnetic_bias.vector_potential_atheta);
        fields.add_point_real_scalar(field_names::MAGNETIC_POTENTIAL, potential)?;
        let mut pressure = vec![Complex64::new(0.0, 0.0); nm + ns + na];
        pressure[nm + ns..].copy_from_slice(&solution.pressure_rms_pa);
        fields.add_point_complex_scalar(field_names::ACOUSTIC_PRESSURE, pressure)?;
        let mut spl = vec![0.0; nm + ns + na];
        for (target, pressure) in spl[nm + ns..].iter_mut().zip(&solution.pressure_rms_pa) {
            *target = 20.0 * (pressure.norm().max(1.0e-15) / 20.0e-6).log10();
        }
        fields.add_point_real_scalar(field_names::SOUND_PRESSURE_LEVEL, spl)?;
        let mut displacement = vec![[Complex64::new(0.0, 0.0); 3]; nm + ns + na];
        let mut velocity = displacement.clone();
        for node in 0..ns {
            displacement[nm + node] = [
                solution.displacement_rz_m[node][0],
                Complex64::new(0.0, 0.0),
                solution.displacement_rz_m[node][1],
            ];
            velocity[nm + node] = [
                solution.velocity_rz_m_per_s[node][0],
                Complex64::new(0.0, 0.0),
                solution.velocity_rz_m_per_s[node][1],
            ];
        }
        fields.add_point_complex_vector(field_names::DISPLACEMENT, displacement)?;
        fields.add_point_complex_vector(field_names::VELOCITY, velocity)?;

        let mut flux = vec![[0.0; 3]; cm + cs + ca];
        flux[..cm].copy_from_slice(&self.magnetic_bias.flux_density_by_cell_t);
        fields.add_cell_real_vector(
            field_names::MAGNETIC_FLUX_DENSITY,
            ProjectedCellVector::new(flux, ProjectionProvenance::ElementCentroid),
        )?;
        let relative_permeability: Vec<_> = mesh
            .elements
            .iter()
            .map(|element| match element.marker {
                PERMANENT_MAGNET => 1.05,
                STEEL_POLE => 1_000.0,
                _ => 1.0,
            })
            .collect();
        fields.add_cell_real_scalar(
            field_names::RELATIVE_PERMEABILITY,
            relative_permeability.clone(),
        )?;
        fields.add_cell_real_scalar(
            field_names::CONDUCTIVITY,
            mesh.elements
                .iter()
                .map(|element| match element.marker {
                    VOICE_COIL => 5.8e7,
                    STEEL_POLE => 7.0e6,
                    _ => 0.0,
                })
                .collect(),
        )?;
        let mut magnetic_field = vec![[0.0; 3]; cm + cs + ca];
        for (cell, target) in magnetic_field[..cm].iter_mut().enumerate() {
            for (component, target_component) in target.iter_mut().enumerate() {
                *target_component = self.magnetic_bias.flux_density_by_cell_t[cell][component]
                    / (MU_0 * relative_permeability[cell]);
            }
        }
        fields.add_cell_real_vector(
            field_names::MAGNETIC_FIELD_STRENGTH,
            ProjectedCellVector::new(magnetic_field, ProjectionProvenance::ElementCentroid),
        )?;
        let equivalent_current_density = self.config.remanence_t / (MU_0 * 20.0e-3);
        fields.add_cell_real_scalar(
            field_names::SOURCE_CURRENT_DENSITY,
            mesh.elements
                .iter()
                .map(|element| {
                    if element.marker == PERMANENT_MAGNET {
                        equivalent_current_density
                    } else {
                        0.0
                    }
                })
                .collect(),
        )?;
        let mut lorentz = vec![[0.0; 3]; cm + cs + ca];
        for (cell, value) in lorentz[..cm]
            .iter_mut()
            .zip(&self.magnetic_bias.flux_density_by_cell_t)
            .zip(&self.magnetic_mesh.elements)
        {
            let ((target, b), element) = (cell, value);
            if element.marker == VOICE_COIL {
                target[2] = -b[0];
            }
        }
        fields.add_cell_real_vector(
            field_names::LORENTZ_FORCE_DENSITY,
            ProjectedCellVector::new(lorentz, ProjectionProvenance::ElementCentroid),
        )?;
        let stresses = structural_stress_cells(&self.structural_mesh, &solution.displacement_rz_m);
        let mut all_stresses = vec![[Complex64::new(0.0, 0.0); 6]; cm + cs + ca];
        all_stresses[cm..cm + cs].copy_from_slice(&stresses);
        fields.add_cell_complex_symmetric_tensor(field_names::STRESS, all_stresses)?;
        let mut von_mises = vec![Complex64::new(0.0, 0.0); cm + cs + ca];
        for (target, stress) in von_mises[cm..cm + cs].iter_mut().zip(&stresses) {
            *target = complex_von_mises(*stress);
        }
        fields.add_cell_complex_scalar(field_names::VON_MISES_STRESS, von_mises)?;
        let mut acoustic_intensity = vec![[0.0; 3]; cm + cs + ca];
        let omega = 2.0 * PI * solution.frequency_hz;
        for (cell, element) in self.acoustic_mesh.elements.iter().enumerate() {
            let nodes = element.vertices();
            let gradients = triangle_gradients(&self.acoustic_mesh, nodes);
            let pressure = nodes
                .iter()
                .map(|node| solution.pressure_rms_pa[*node])
                .sum::<Complex64>()
                / 3.0;
            let mut gradient = [Complex64::default(); 2];
            for local in 0..3 {
                for component in 0..2 {
                    gradient[component] +=
                        solution.pressure_rms_pa[nodes[local]] * gradients[local][component];
                }
            }
            let velocity = [
                Complex64::new(0.0, 1.0) * gradient[0]
                    / (omega * self.config.fluid_density_kg_per_m3),
                Complex64::new(0.0, 1.0) * gradient[1]
                    / (omega * self.config.fluid_density_kg_per_m3),
            ];
            acoustic_intensity[cm + cs + cell] = [
                (pressure * velocity[0].conj()).re,
                0.0,
                (pressure * velocity[1].conj()).re,
            ];
        }
        fields.add_cell_real_vector(
            field_names::ACOUSTIC_INTENSITY,
            ProjectedCellVector::new(acoustic_intensity, ProjectionProvenance::ElementCentroid),
        )?;
        let material_ids = mesh
            .elements
            .iter()
            .map(|element| element.marker as f64)
            .collect();
        fields.add_cell_real_scalar(field_names::MATERIAL_ID, material_ids)?;
        Ok((mesh, fields))
    }

    fn power_balance(
        &self,
        voltage: Complex64,
        current: Complex64,
        pressure: &[Complex64],
        displacement: &[Complex64],
        omega: f64,
    ) -> PowerBalance {
        let velocity: Vec<_> = displacement
            .iter()
            .map(|value| Complex64::new(0.0, omega) * value)
            .collect();
        let pressure_force = transpose_apply_real(&self.coupled_system.coupling, pressure);
        let fluid_on_structure_w = pressure_force
            .iter()
            .zip(&velocity)
            .map(|(force, velocity)| (*force * velocity.conj()).re)
            .sum::<f64>();
        let input_w = (voltage * current.conj()).re;
        let copper_loss_w = self.config.coil_resistance_ohm * current.norm_sqr();
        let mechanical_loss_w = quadratic_power(
            &self.solid_damping_at(omega / (2.0 * PI)).unwrap_or_else(|_| {
                // Extraction runs only after a successful solve at this
                // frequency, which already evaluated the same damping;
                // reaching this fallback would indicate an internal
                // inconsistency, so mirror the legacy triplet rather
                // than inventing a loss value.
                self.coupled_system.solid_damping.clone()
            }),
            &velocity,
        );
        let acoustic_loss_w = -fluid_on_structure_w;
        let acoustic_operator_loss_w =
            quadratic_power(&self.coupled_system.fluid_damping, pressure)
                / self.config.fluid_density_kg_per_m3;
        PowerBalance {
            input_w,
            copper_loss_w,
            mechanical_loss_w,
            acoustic_loss_w,
            acoustic_operator_loss_w,
            residual_w: input_w - copper_loss_w - mechanical_loss_w - acoustic_loss_w,
        }
    }
}

fn build_magnetic_bias(
    config: &AxisymmetricLinearConfig,
) -> Result<(Mesh, MagneticBiasSolution), AxisymmetricLinearError> {
    let mut mesh = rectangular_mesh_triangles(
        0.0,
        40.0e-3,
        -20.0e-3,
        20.0e-3,
        config.magnetic_radial_cells,
        config.magnetic_axial_cells,
    );
    mark_cells(&mut mesh, |r, z| {
        if r <= 10.0e-3 && z <= 4.0e-3 {
            STEEL_POLE
        } else if (18.0e-3..=30.0e-3).contains(&r) && (-10.0e-3..=10.0e-3).contains(&z) {
            PERMANENT_MAGNET
        } else if (12.0e-3..=16.0e-3).contains(&r) && (-6.0e-3..=6.0e-3).contains(&z) {
            VOICE_COIL
        } else if (10.0e-3..=18.0e-3).contains(&r) && (-10.0e-3..=10.0e-3).contains(&z) {
            AIR_GAP
        } else if (r >= 30.0e-3 && (-15.0e-3..=15.0e-3).contains(&z))
            || (z <= -10.0e-3 && r <= 30.0e-3)
        {
            STEEL_YOKE
        } else {
            MAGNETIC_AIR
        }
    });
    register_groups(
        &mut mesh,
        &[
            (MAGNETIC_AIR, "magnetic air"),
            (PERMANENT_MAGNET, "permanent magnet"),
            (STEEL_POLE, "steel pole"),
            (VOICE_COIL, "voice coil"),
            (AIR_GAP, "air gap"),
            (STEEL_YOKE, "steel yoke"),
        ],
    )?;
    remap_boundary_markers(
        &mut mesh,
        [
            MAGNETIC_AXIS,
            MAGNETIC_OUTER_RADIUS,
            MAGNETIC_LOWER_END,
            MAGNETIC_UPPER_END,
        ],
    );
    register_boundary_groups(
        &mut mesh,
        &[
            (MAGNETIC_AXIS, "magnetic symmetry axis"),
            (MAGNETIC_OUTER_RADIUS, "magnetic exterior"),
            (MAGNETIC_LOWER_END, "magnetic lower exterior"),
            (MAGNETIC_UPPER_END, "magnetic upper exterior"),
        ],
    )?;
    let space = H1Space::new(PolynomialDegree::P1);
    let mut forms = RegionMap::new();
    forms.insert(MAGNETIC_AIR, AxisymmetricAThetaForm::new(MU_0));
    forms.insert(PERMANENT_MAGNET, AxisymmetricAThetaForm::new(1.05 * MU_0));
    forms.insert(STEEL_POLE, AxisymmetricAThetaForm::new(1_000.0 * MU_0));
    forms.insert(VOICE_COIL, AxisymmetricAThetaForm::new(MU_0));
    forms.insert(AIR_GAP, AxisymmetricAThetaForm::new(MU_0));
    forms.insert(STEEL_YOKE, AxisymmetricAThetaForm::new(1_000.0 * MU_0));
    let matrix = assemble_bilinear_by_region(&forms, &space, &space, &mesh, 2)?;
    let equivalent_current_density = config.remanence_t / (MU_0 * 20.0e-3);
    let source = AzimuthalCurrentDensityForm::new(move |_, _| equivalent_current_density);
    let rhs = assemble_linear_where(&source, &space, &mesh, 2, |_, element| {
        element.marker == PERMANENT_MAGNET
    });
    let mut system = FemSystem::new(matrix, rhs);
    for marker in [
        MAGNETIC_AXIS,
        MAGNETIC_OUTER_RADIUS,
        MAGNETIC_LOWER_END,
        MAGNETIC_UPPER_END,
    ] {
        apply_homogeneous_dirichlet(&mut system, &mesh, &space, marker);
    }
    let solver = SolverConfig {
        solver_type: SolverType::Direct,
        ..SolverConfig::default()
    };
    let solution = solve_csr(
        &real_to_complex(&system.matrix).to_csr(),
        &system
            .rhs
            .iter()
            .map(|value| Complex64::new(*value, 0.0))
            .collect::<Vec<_>>(),
        &solver,
    )?;
    let vector_potential_atheta: Vec<_> = solution.values.iter().map(|value| value.re).collect();
    let magnetic_bias = magnetic_bias_from_potential(config, &mesh, vector_potential_atheta)?;
    Ok((mesh, magnetic_bias))
}

pub(crate) fn magnetic_bias_from_potential(
    config: &AxisymmetricLinearConfig,
    mesh: &Mesh,
    vector_potential_atheta: Vec<f64>,
) -> Result<MagneticBiasSolution, AxisymmetricLinearError> {
    let flux_density_by_cell_t = magnetic_flux_by_cell(mesh, &vector_potential_atheta);
    let flux_density_at_nodes_t = cell_vectors_to_nodes(mesh, &flux_density_by_cell_t);
    let (weighted_flux, weighted_radius, weight) = mesh
        .elements
        .iter()
        .zip(&flux_density_by_cell_t)
        .filter(|(element, _)| element.marker == VOICE_COIL)
        .fold(
            (0.0, 0.0, 0.0),
            |(flux, radius_moment, weight), (element, b)| {
                let nodes = element.vertices();
                let (r, _) = element_centroid(mesh, nodes);
                let area = triangle_area(mesh, nodes);
                let radial = [
                    mesh.nodes[nodes[0]].x,
                    mesh.nodes[nodes[1]].x,
                    mesh.nodes[nodes[2]].x,
                ];
                let mean_radius_squared = (radial.iter().map(|r| r * r).sum::<f64>()
                    + radial[0] * radial[1]
                    + radial[0] * radial[2]
                    + radial[1] * radial[2])
                    / 6.0;
                (
                    flux + b[0] * r * area,
                    radius_moment + mean_radius_squared * area,
                    weight + r * area,
                )
            },
        );
    if weight == 0.0 {
        return Err(AxisymmetricLinearError::InvalidConfiguration(
            "magnetic mesh contains no voice-coil cells",
        ));
    }
    let mean_coil_radius = weighted_radius / weight;
    let distributed_bl_tm =
        2.0 * PI * mean_coil_radius * config.active_turns * weighted_flux.abs() / weight;
    Ok(MagneticBiasSolution {
        vector_potential_atheta,
        flux_density_by_cell_t,
        flux_density_at_nodes_t,
        distributed_bl_tm,
    })
}

fn build_structural_mesh(config: &AxisymmetricLinearConfig) -> Result<Mesh, PhysicalGroupError> {
    let mut mesh = rectangular_mesh_triangles(
        0.0,
        config.diaphragm_radius_m,
        -config.diaphragm_thickness_m,
        0.0,
        config.structural_radial_cells,
        config.structural_axial_cells,
    );
    mark_cells(&mut mesh, |r, _| {
        let normalized_radius = r / config.diaphragm_radius_m;
        if normalized_radius < 1.0 / 6.0 || (2.0 / 3.0..5.0 / 6.0).contains(&normalized_radius) {
            DIAPHRAGM
        } else if normalized_radius < 1.0 / 3.0 {
            COIL_WINDING
        } else if normalized_radius < 0.5 {
            COIL_FORMER
        } else if normalized_radius < 2.0 / 3.0 {
            SPIDER
        } else if normalized_radius >= 5.0 / 6.0 {
            SURROUND
        } else {
            DIAPHRAGM
        }
    });
    register_groups(
        &mut mesh,
        &[
            (DIAPHRAGM, "moving diaphragm"),
            (COIL_FORMER, "coil former"),
            (SURROUND, "surround"),
            (SPIDER, "spider"),
            (COIL_WINDING, "voice coil winding"),
        ],
    )?;
    remap_boundary_markers(
        &mut mesh,
        [
            STRUCTURAL_AXIS,
            STRUCTURAL_FRAME,
            STRUCTURAL_REAR_SURFACE,
            STRUCTURAL_ACOUSTIC_INTERFACE,
        ],
    );
    register_boundary_groups(
        &mut mesh,
        &[
            (STRUCTURAL_AXIS, "structural symmetry axis"),
            (STRUCTURAL_FRAME, "fixed frame"),
            (STRUCTURAL_REAR_SURFACE, "rear moving surface"),
            (STRUCTURAL_ACOUSTIC_INTERFACE, "acoustic interface"),
        ],
    )?;
    Ok(mesh)
}

fn build_acoustic_mesh(config: &AxisymmetricLinearConfig) -> Result<Mesh, PhysicalGroupError> {
    let mut mesh = rectangular_mesh_triangles(
        0.0,
        config.diaphragm_radius_m,
        0.0,
        config.acoustic_height_m,
        config.structural_radial_cells,
        config.acoustic_axial_cells,
    );
    for element in &mut mesh.elements {
        element.marker = ACOUSTIC_FLUID;
    }
    register_groups(&mut mesh, &[(ACOUSTIC_FLUID, "acoustic fluid")])?;
    remap_boundary_markers(
        &mut mesh,
        [
            ACOUSTIC_AXIS,
            ACOUSTIC_SIDE_WALL,
            ACOUSTIC_DIAPHRAGM_INTERFACE,
            ACOUSTIC_RADIATION_BOUNDARY,
        ],
    );
    register_boundary_groups(
        &mut mesh,
        &[
            (ACOUSTIC_AXIS, "acoustic symmetry axis"),
            (ACOUSTIC_SIDE_WALL, "acoustic side wall"),
            (ACOUSTIC_DIAPHRAGM_INTERFACE, "diaphragm interface"),
            (ACOUSTIC_RADIATION_BOUNDARY, "acoustic radiation boundary"),
        ],
    )?;
    Ok(mesh)
}

fn assemble_diaphragm_acoustic_coupling(
    structure: &Mesh,
    acoustic: &Mesh,
    radial_cells: usize,
) -> TripletMatrix<f64> {
    let mut coupling = TripletMatrix::new(acoustic.num_nodes(), 2 * structure.num_nodes());
    let structure_top = structure.num_nodes() - (radial_cells + 1);
    let acoustic_bottom = 0;
    let gauss = [
        (-1.0_f64 / 3.0_f64.sqrt(), 1.0),
        (1.0_f64 / 3.0_f64.sqrt(), 1.0),
    ];
    for segment in 0..radial_cells {
        let s_nodes = [structure_top + segment, structure_top + segment + 1];
        let a_nodes = [acoustic_bottom + segment, acoustic_bottom + segment + 1];
        let r0 = structure.nodes[s_nodes[0]].x;
        let r1 = structure.nodes[s_nodes[1]].x;
        for (xi, weight) in gauss {
            let shape = [0.5 * (1.0 - xi), 0.5 * (1.0 + xi)];
            let radius = shape[0] * r0 + shape[1] * r1;
            let jacobian = 0.5 * (r1 - r0);
            for a in 0..2 {
                for s in 0..2 {
                    coupling.push(
                        a_nodes[a],
                        2 * s_nodes[s] + 1,
                        -shape[a] * shape[s] * 2.0 * PI * radius * jacobian * weight,
                    );
                }
            }
        }
    }
    coupling
}

fn distributed_motor_row(
    mesh: &Mesh,
    bl: f64,
    coil_marker: i32,
) -> Result<TripletMatrix<f64>, AxisymmetricLinearError> {
    let mut row = TripletMatrix::new(1, 2 * mesh.num_nodes());
    let mut weights = vec![0.0; mesh.num_nodes()];
    for cell in mesh.cells_with_marker(coil_marker) {
        let element = &mesh.elements[cell];
        let (radius, _) = element_centroid(mesh, element.vertices());
        let axisymmetric_measure = 2.0 * PI * radius * triangle_area(mesh, element.vertices());
        for &node in element.vertices() {
            weights[node] += axisymmetric_measure / element.vertices().len() as f64;
        }
    }
    let sum: f64 = weights.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(AxisymmetricLinearError::MissingMovingCoilRegion);
    }
    for (node, weight) in weights.into_iter().enumerate() {
        if weight > 0.0 {
            row.push(0, 2 * node + 1, bl * weight / sum);
        }
    }
    Ok(row)
}

fn structural_constraint_dofs(mesh: &Mesh, radius: f64) -> Vec<usize> {
    let tolerance = radius * 1.0e-10;
    let mut dofs = Vec::new();
    for (node, point) in mesh.nodes.iter().enumerate() {
        if (point.x - radius).abs() <= tolerance {
            dofs.extend([2 * node, 2 * node + 1]);
        } else if point.x.abs() <= tolerance {
            dofs.push(2 * node);
        }
    }
    dofs
}

fn extract_thiele_small(system: &CoupledSystem, mesh: &Mesh, radius: f64) -> ThieleSmallReference {
    let shape: Vec<_> = mesh
        .nodes
        .iter()
        .flat_map(|point| [0.0, (1.0 - (point.x / radius).powi(2)).max(0.0)])
        .collect();
    let moving_mass_kg = quadratic_real(&system.solid_mass, &shape);
    let stiffness_n_per_m = quadratic_real(&system.solid_stiffness, &shape);
    let mechanical_resistance_kg_per_s = quadratic_real(&system.solid_damping, &shape);
    let force_factor_tm = system
        .electrical
        .as_ref()
        .map(|electrical| {
            electrical
                .coupling
                .values
                .iter()
                .enumerate()
                .map(|(index, value)| value * shape[electrical.coupling.cols[index]])
                .sum()
        })
        .unwrap_or(0.0);
    ThieleSmallReference {
        moving_mass_kg,
        stiffness_n_per_m,
        mechanical_resistance_kg_per_s,
        force_factor_tm,
        resonance_hz: (stiffness_n_per_m / moving_mass_kg).sqrt() / (2.0 * PI),
    }
}

fn mark_cells(mesh: &mut Mesh, marker: impl Fn(f64, f64) -> i32) {
    for index in 0..mesh.elements.len() {
        let (r, z) = element_centroid(mesh, mesh.elements[index].vertices());
        mesh.elements[index].marker = marker(r, z);
    }
}

pub(crate) fn register_groups(
    mesh: &mut Mesh,
    groups: &[(i32, &str)],
) -> Result<(), PhysicalGroupError> {
    for &(marker, name) in groups {
        mesh.register_physical_group(2, marker, name)?;
    }
    Ok(())
}

pub(crate) fn register_boundary_groups(
    mesh: &mut Mesh,
    groups: &[(i32, &str)],
) -> Result<(), PhysicalGroupError> {
    for &(marker, name) in groups {
        mesh.register_physical_group(1, marker, name)?;
    }
    Ok(())
}

pub(crate) fn remap_boundary_markers(mesh: &mut Mesh, replacement: [i32; 4]) {
    for boundary in &mut mesh.boundaries {
        if let Ok(index) = usize::try_from(boundary.marker - 1)
            && let Some(&marker) = replacement.get(index)
        {
            boundary.marker = marker;
        }
    }
}

pub(crate) fn element_centroid(mesh: &Mesh, nodes: &[usize]) -> (f64, f64) {
    let scale = 1.0 / nodes.len() as f64;
    (
        nodes.iter().map(|node| mesh.nodes[*node].x).sum::<f64>() * scale,
        nodes.iter().map(|node| mesh.nodes[*node].y).sum::<f64>() * scale,
    )
}

pub(crate) fn triangle_area(mesh: &Mesh, nodes: &[usize]) -> f64 {
    let [a, b, c] = [
        &mesh.nodes[nodes[0]],
        &mesh.nodes[nodes[1]],
        &mesh.nodes[nodes[2]],
    ];
    0.5 * ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)).abs()
}

pub(crate) fn triangle_gradients(mesh: &Mesh, nodes: &[usize]) -> [[f64; 2]; 3] {
    let [a, b, c] = [
        &mesh.nodes[nodes[0]],
        &mesh.nodes[nodes[1]],
        &mesh.nodes[nodes[2]],
    ];
    let twice_area = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
    [
        [(b.y - c.y) / twice_area, (c.x - b.x) / twice_area],
        [(c.y - a.y) / twice_area, (a.x - c.x) / twice_area],
        [(a.y - b.y) / twice_area, (b.x - a.x) / twice_area],
    ]
}

fn magnetic_flux_by_cell(mesh: &Mesh, potential: &[f64]) -> Vec<[f64; 3]> {
    mesh.elements
        .iter()
        .map(|element| {
            let nodes = element.vertices();
            let gradients = triangle_gradients(mesh, nodes);
            let mut da_dr = 0.0;
            let mut da_dz = 0.0;
            let mut a = 0.0;
            for local in 0..3 {
                da_dr += potential[nodes[local]] * gradients[local][0];
                da_dz += potential[nodes[local]] * gradients[local][1];
                a += potential[nodes[local]] / 3.0;
            }
            let (radius, _) = element_centroid(mesh, nodes);
            [-da_dz, 0.0, da_dr + a / radius.max(1.0e-12)]
        })
        .collect()
}

fn cell_vectors_to_nodes(mesh: &Mesh, values: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let mut result = vec![[0.0; 3]; mesh.num_nodes()];
    let mut counts = vec![0usize; mesh.num_nodes()];
    for (element, value) in mesh.elements.iter().zip(values) {
        for &node in element.vertices() {
            for component in 0..3 {
                result[node][component] += value[component];
            }
            counts[node] += 1;
        }
    }
    for (value, count) in result.iter_mut().zip(counts) {
        if count > 0 {
            for component in value {
                *component /= count as f64;
            }
        }
    }
    result
}

/// Recover the per-cell complex axisymmetric stress phasor
/// `[σrr, σθθ, σzz, 0, 0, τrz]` from a solved RMS displacement field.
///
/// Each cell uses the isotropic material of its own region marker (see
/// [`moving_assembly_material`]); a cell whose marker has no assigned material
/// falls back to the diaphragm material so legacy/export paths never panic.
/// The returned stresses carry the same RMS convention as `displacement`.
pub fn structural_stress_cells(
    mesh: &Mesh,
    displacement: &[[Complex64; 2]],
) -> Vec<[Complex64; 6]> {
    mesh.elements
        .iter()
        .map(|element| {
            let material = moving_assembly_material(element.marker)
                .unwrap_or_else(|| IsotropicMaterial::new(2.4e9, 0.32));
            let lambda = material.lambda();
            let mu = material.mu();
            let nodes = element.vertices();
            let gradients = triangle_gradients(mesh, nodes);
            let mut dur_dr = Complex64::default();
            let mut dur_dz = Complex64::default();
            let mut duz_dr = Complex64::default();
            let mut duz_dz = Complex64::default();
            let mut ur = Complex64::default();
            for local in 0..3 {
                let value = displacement[nodes[local]];
                dur_dr += value[0] * gradients[local][0];
                dur_dz += value[0] * gradients[local][1];
                duz_dr += value[1] * gradients[local][0];
                duz_dz += value[1] * gradients[local][1];
                ur += value[0] / 3.0;
            }
            let (radius, _) = element_centroid(mesh, nodes);
            let hoop = ur / radius.max(1.0e-12);
            let trace = dur_dr + duz_dz + hoop;
            let shear_rz = mu * (dur_dz + duz_dr);
            [
                lambda * trace + 2.0 * mu * dur_dr,
                lambda * trace + 2.0 * mu * hoop,
                lambda * trace + 2.0 * mu * duz_dz,
                Complex64::default(),
                Complex64::default(),
                shear_rz,
            ]
        })
        .collect()
}

/// Complex von Mises equivalent stress of a `[σrr, σθθ, σzz, _, _, τrz]`
/// stress phasor. The modulus of the result is the RMS von Mises magnitude;
/// it equals the amplitude of the von Mises stress oscillation when all
/// stress components share a common phase, and is the convention used by the
/// exported `VON_MISES_STRESS` field.
pub fn complex_von_mises(stress: [Complex64; 6]) -> Complex64 {
    let [srr, stt, szz, _, _, srz] = stress;
    (((srr - stt) * (srr - stt)
        + (stt - szz) * (stt - szz)
        + (szz - srr) * (szz - srr)
        + 6.0 * srz * srz)
        / 2.0)
        .sqrt()
}

/// Phase-agnostic von-Mises envelope for a harmonic complex stress tensor.
///
/// The usual complex von-Mises phasor is exact only when all stress
/// components share a common phase. When components are out of phase, the
/// instantaneous equivalent stress is nonlinear and can exceed
/// `|complex_von_mises|`. This triangle-inequality envelope is conservative
/// for arbitrary component phases and is therefore the value used by safety
/// constraints.
pub fn conservative_von_mises(stress: [Complex64; 6]) -> f64 {
    let [srr, stt, szz, _, _, srz] = stress;
    // Each pairwise difference is itself a complex phasor; taking its norm
    // preserves cancellation before applying the sum-of-squares envelope.
    let invariant_bound = (srr - stt).norm_sqr()
        + (stt - szz).norm_sqr()
        + (szz - srr).norm_sqr()
        + 6.0 * srz.norm_sqr();
    (invariant_bound / 2.0).sqrt()
}

fn combine_meshes(meshes: &[&Mesh]) -> Result<Mesh, PhysicalGroupError> {
    let mut combined = Mesh::new(2);
    for mesh in meshes {
        let offset = combined.num_nodes();
        for point in &mesh.nodes {
            combined.add_node(Point::new_2d(point.x, point.y));
        }
        for element in &mesh.elements {
            combined.add_element_with_marker(
                ElementType::Triangle,
                element
                    .vertices()
                    .iter()
                    .map(|node| offset + node)
                    .collect(),
                element.marker,
            );
        }
        for group in mesh.physical_groups.iter() {
            combined.register_physical_group(group.dimension, group.tag, group.name.clone())?;
        }
    }
    Ok(combined)
}

fn apply_complex_constraints(
    matrix: &mut TripletMatrix<Complex64>,
    rhs: &mut [Complex64],
    dofs: &[usize],
) {
    let constrained: std::collections::HashSet<_> = dofs.iter().copied().collect();
    let mut filtered = TripletMatrix::new(matrix.num_rows, matrix.num_cols);
    for index in 0..matrix.nnz() {
        if !constrained.contains(&matrix.rows[index]) && !constrained.contains(&matrix.cols[index])
        {
            filtered.push(matrix.rows[index], matrix.cols[index], matrix.values[index]);
        }
    }
    for &dof in dofs {
        filtered.push(dof, dof, Complex64::new(1.0, 0.0));
        rhs[dof] = Complex64::new(0.0, 0.0);
    }
    *matrix = filtered;
}

fn scalar_triplet(value: f64) -> TripletMatrix<f64> {
    let mut result = TripletMatrix::new(1, 1);
    result.push(0, 0, value);
    result
}

fn scaled_triplet(matrix: &TripletMatrix<f64>, scale: f64) -> TripletMatrix<f64> {
    let mut result = TripletMatrix::with_capacity(matrix.num_rows, matrix.num_cols, matrix.nnz());
    for index in 0..matrix.nnz() {
        result.push(
            matrix.rows[index],
            matrix.cols[index],
            scale * matrix.values[index],
        );
    }
    result
}

fn real_to_complex(matrix: &TripletMatrix<f64>) -> TripletMatrix<Complex64> {
    let mut result = TripletMatrix::with_capacity(matrix.num_rows, matrix.num_cols, matrix.nnz());
    for index in 0..matrix.nnz() {
        result.push(
            matrix.rows[index],
            matrix.cols[index],
            Complex64::new(matrix.values[index], 0.0),
        );
    }
    result
}

fn quadratic_real(matrix: &TripletMatrix<f64>, vector: &[f64]) -> f64 {
    (0..matrix.nnz())
        .map(|index| vector[matrix.rows[index]] * matrix.values[index] * vector[matrix.cols[index]])
        .sum()
}

fn quadratic_power(matrix: &TripletMatrix<f64>, vector: &[Complex64]) -> f64 {
    (0..matrix.nnz())
        .map(|index| {
            (vector[matrix.rows[index]].conj() * matrix.values[index] * vector[matrix.cols[index]])
                .re
        })
        .sum()
}

fn transpose_apply_real(matrix: &TripletMatrix<f64>, input: &[Complex64]) -> Vec<Complex64> {
    let mut output = vec![Complex64::default(); matrix.num_cols];
    for index in 0..matrix.nnz() {
        output[matrix.cols[index]] += matrix.values[index] * input[matrix.rows[index]];
    }
    output
}

fn complex_is_finite(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

#[derive(Debug, thiserror::Error)]
pub enum AxisymmetricLinearError {
    #[error("invalid axisymmetric model: {0}")]
    InvalidConfiguration(&'static str),
    #[error("invalid region modulus: {0}")]
    InvalidRegionModulus(String),
    #[error("structural mesh has no distributed voice-coil winding region")]
    MissingMovingCoilRegion,
    #[error("structural mesh has no positive-area structural acoustic interface")]
    MissingStructuralAcousticInterface,
    #[error("acoustic radiation boundary is missing from the coupled mesh")]
    MissingRadiationBoundary,
    #[error(transparent)]
    RegionAssembly(#[from] RegionAssemblyError),
    #[error(transparent)]
    PhysicalGroup(#[from] PhysicalGroupError),
    #[error(transparent)]
    CoupledSystem(#[from] CoupledSystemError),
    #[error(transparent)]
    CoupledIterative(#[from] CoupledIterativeError),
    #[error(transparent)]
    Solver(#[from] SolverError),
    #[error(transparent)]
    Export(#[from] FieldExportError),
    #[error(transparent)]
    PowerLedger(#[from] PowerLedgerError),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coarse_model() -> AxisymmetricLinearModel {
        AxisymmetricLinearModel::build(AxisymmetricLinearConfig {
            magnetic_radial_cells: 8,
            magnetic_axial_cells: 8,
            structural_radial_cells: 6,
            structural_axial_cells: 1,
            acoustic_axial_cells: 4,
            ..AxisymmetricLinearConfig::default()
        })
        .unwrap()
    }

    #[test]
    fn region_moduli_wire_complex_stiffness_and_fail_closed() {
        let model = AxisymmetricLinearModel::build(AxisymmetricLinearConfig {
            magnetic_radial_cells: 8,
            magnetic_axial_cells: 8,
            structural_radial_cells: 6,
            structural_axial_cells: 1,
            acoustic_axial_cells: 4,
            region_moduli: vec![RegionModulus {
                marker: SURROUND,
                spec: RegionModulusSpec::Tabulated {
                    points: vec![(50.0, 4.0e6, 1.2e6), (500.0, 6.0e6, 2.4e6), (2_000.0, 9.0e6, 3.6e6)],
                },
            }],
            ..AxisymmetricLinearConfig::default()
        })
        .unwrap();
        assert_eq!(model.solid_regions.len(), MOVING_ASSEMBLY_MARKERS.len());
        let system = model.coupled_system_at(200.0).unwrap();
        // Complex stiffness must change the solid block vs the legacy one.
        let changed = system
            .solid_stiffness
            .values
            .iter()
            .zip(&model.coupled_system.solid_stiffness.values)
            .any(|(actual, legacy)| (actual - legacy).abs() > 1.0e-9);
        assert!(changed, "tabulated surround modulus must change stiffness");
        let solution = model
            .solve_frequency(200.0, Complex64::new(1.0, 0.0))
            .unwrap();
        assert!(solution.impedance_ohm.is_finite());
        // Outside the tabulated range the solve fails closed.
        assert!(model.coupled_system_at(10_000.0).is_err());
        assert!(model.solve_frequency(10_000.0, Complex64::new(1.0, 0.0)).is_err());
    }

    /// Golden-value regression: `build_magnetic_bias` classifies cells with
    /// hard-coded millimetre coordinates (e.g. the STEEL_YOKE region
    /// `r >= 30e-3 || z <= -10e-3`) that assume the default 40×40 mm magnetic
    /// mesh window. Pin the default golden model's small-signal BL so any
    /// change to the classification coordinates, the μ_r assignments, or the
    /// coil BL quadrature shows up as a deliberate, reviewed diff.
    #[test]
    fn golden_model_magnetic_bias_bl_is_pinned() {
        const GOLDEN_BL_TM: f64 = 2.494_862_206_944_29;
        let (_, bias) = build_magnetic_bias(&AxisymmetricLinearConfig::default()).unwrap();
        assert!(
            (bias.distributed_bl_tm - GOLDEN_BL_TM).abs() / GOLDEN_BL_TM < 1.0e-12,
            "golden-model BL changed: {} vs pinned {GOLDEN_BL_TM} T·m — \
             review the magnetic cell classification in build_magnetic_bias",
            bias.distributed_bl_tm
        );
    }

    #[test]
    fn marked_domains_and_region_assemblies_are_complete() {
        let model = coarse_model();
        for marker in [
            MAGNETIC_AIR,
            PERMANENT_MAGNET,
            STEEL_POLE,
            VOICE_COIL,
            AIR_GAP,
        ] {
            assert!(
                model
                    .magnetic_mesh
                    .cells_with_marker(marker)
                    .next()
                    .is_some()
            );
        }
        for marker in [DIAPHRAGM, SURROUND, SPIDER, COIL_FORMER, COIL_WINDING] {
            assert!(
                model
                    .structural_mesh
                    .cells_with_marker(marker)
                    .next()
                    .is_some(),
                "missing moving-assembly marker {marker}"
            );
        }
        for (mesh, groups) in [
            (
                &model.magnetic_mesh,
                &[(2, "air gap"), (1, "magnetic exterior")][..],
            ),
            (
                &model.structural_mesh,
                &[(2, "voice coil winding"), (1, "acoustic interface")][..],
            ),
            (
                &model.acoustic_mesh,
                &[(2, "acoustic fluid"), (1, "diaphragm interface")][..],
            ),
        ] {
            for &(dimension, name) in groups {
                assert!(
                    mesh.physical_groups.by_name(dimension, name).is_some(),
                    "missing named group {name}"
                );
            }
        }
        assert!(model.coupled_system.fluid_stiffness.nnz() > 0);
        assert!(model.coupled_system.solid_stiffness.nnz() > 0);
        assert!(model.magnetic_bias.distributed_bl_tm.is_finite());
        assert!(model.magnetic_bias.distributed_bl_tm > 0.0);
    }

    #[test]
    fn motor_force_and_back_emf_are_distributed_over_structure_dofs() {
        let model = coarse_model();
        let coupling = &model.coupled_system.electrical.as_ref().unwrap().coupling;
        assert!(coupling.nnz() > 2);
        let sum: f64 = coupling.values.iter().sum();
        assert!((sum - model.magnetic_bias.distributed_bl_tm).abs() / sum.abs() < 1.0e-12);
        let coil_nodes: std::collections::BTreeSet<_> = model
            .structural_mesh
            .cells_with_marker(COIL_WINDING)
            .flat_map(|cell| {
                model.structural_mesh.elements[cell]
                    .vertices()
                    .iter()
                    .copied()
            })
            .collect();
        assert!(
            coupling
                .cols
                .iter()
                .all(|dof| dof % 2 == 1 && coil_nodes.contains(&(dof / 2)))
        );
    }

    #[test]
    fn thiele_small_reduction_has_the_expected_resonance() {
        let model = coarse_model();
        let reference = model.thiele_small;
        assert!(reference.moving_mass_kg > 0.0);
        assert!(reference.stiffness_n_per_m > 0.0);
        assert!(reference.resonance_hz > 0.0);
        let impedance = model.reduced_mechanical_impedance(reference.resonance_hz);
        assert!(impedance.im.abs() < 1.0e-9 * impedance.re.abs().max(1.0));
    }

    #[test]
    fn distributed_model_matches_its_lumped_reduction_in_linear_low_frequency_limit() {
        let model = AxisymmetricLinearModel::build(AxisymmetricLinearConfig {
            magnetic_radial_cells: 8,
            magnetic_axial_cells: 8,
            structural_radial_cells: 6,
            structural_axial_cells: 1,
            acoustic_axial_cells: 4,
            // Suppress acoustic back-loading for this structural/electrical
            // reduction check; radiation is validated independently.
            fluid_density_kg_per_m3: 1.0e-9,
            ..AxisymmetricLinearConfig::default()
        })
        .unwrap();
        let frequency_hz = 50.0;
        let omega = 2.0 * PI * frequency_hz;
        let voltage = Complex64::new(1.0, 0.0);
        let distributed = model.solve_frequency(frequency_hz, voltage).unwrap();
        let mechanical = model.reduced_mechanical_impedance(frequency_hz);
        let bl = model.thiele_small.force_factor_tm;
        let lumped_impedance = Complex64::new(
            model.config.coil_resistance_ohm,
            omega * model.config.coil_inductance_h,
        ) + Complex64::new(bl * bl, 0.0) / mechanical;
        let relative_error = (distributed.impedance_ohm - lumped_impedance).norm()
            / lumped_impedance.norm().max(f64::MIN_POSITIVE);
        assert!(
            relative_error < 0.20,
            "distributed/lumped impedance mismatch in the linear LF limit: distributed={}, lumped={}, relative error={relative_error:.3}",
            distributed.impedance_ohm,
            lumped_impedance
        );
    }

    /// The first-order ABC `∂p/∂n = −ikp` is exact for the plane waves the
    /// compact front domain supports at low frequency, so the domain loads
    /// the diaphragm with `Z = ρc·A` (a pure damper) regardless of `kH`.
    /// This pins the unified 2π fluid normalization: before the fluid blocks
    /// were lifted to the structural `2πr` convention, `Z_compact` measured a
    /// factor 2π hot at 40 Hz (the pressure, observation SPL, and acoustic
    /// power terms inherited the same inflation).
    #[test]
    fn compact_fluid_load_matches_first_order_radiation_impedance() {
        let model = coarse_model();
        let solution = model
            .solve_frequency(40.0, Complex64::new(2.83, 0.0))
            .unwrap();
        let velocity = model.area_averaged_diaphragm_velocity(&solution).unwrap();
        let z_compact = -model.compact_fluid_interface_force_n(&solution) / velocity;
        let rho_c_area = model.config.fluid_density_kg_per_m3
            * model.config.sound_speed_m_per_s
            * model.interface_area_m2().unwrap();
        let ratio = z_compact.re / rho_c_area;
        assert!(
            (0.9..=1.1).contains(&ratio),
            "Re(Z_compact) = {} vs rho*c*A = {rho_c_area} (ratio {ratio}); \
             a ratio near 2pi would signal the old per-radian fluid normalization",
            z_compact.re
        );
        assert!(
            z_compact.im.abs() / rho_c_area < 0.2,
            "Z_compact = {z_compact} should be nearly purely resistive at 40 Hz"
        );
    }

    #[test]
    fn voltage_drive_solves_distributed_fields_and_closes_power_balance() {
        let model = coarse_model();
        let solution = model
            .solve_frequency(300.0, Complex64::new(2.83, 0.0))
            .unwrap();
        assert!(solution.current_rms_a.norm() > 0.0);
        assert!(solution.solver.converged);
        assert!(solution.solver.relative_residual < 1.0e-9);
        assert!(
            solution
                .pressure_rms_pa
                .iter()
                .any(|value| value.norm() > 0.0)
        );
        assert!(
            solution
                .displacement_rz_m
                .iter()
                .any(|value| value[1].norm() > 0.0)
        );
        let scale = solution.power.input_w.abs().max(1.0e-12);
        assert!(solution.power.residual_w.abs() / scale < 1.0e-7);
        assert!(solution.power.acoustic_loss_w >= -1.0e-9);
        assert!(
            (solution.power.acoustic_loss_w - solution.power.acoustic_operator_loss_w).abs()
                / solution.power.acoustic_loss_w.abs().max(1.0e-12)
                < 1.0e-6
        );
    }

    #[test]
    fn production_block_solver_matches_direct_reference() {
        let model = coarse_model();
        let iterative = model
            .solve_frequency(350.0, Complex64::new(2.83, 0.0))
            .unwrap();
        let direct = model
            .solve_frequency_direct_reference(350.0, Complex64::new(2.83, 0.0))
            .unwrap();
        let impedance_error = (iterative.impedance_ohm - direct.impedance_ohm).norm()
            / direct.impedance_ohm.norm().max(1.0e-15);
        assert!(
            impedance_error < 1.0e-7,
            "impedance error {impedance_error:e}"
        );
        let pressure_error = iterative
            .pressure_rms_pa
            .iter()
            .zip(&direct.pressure_rms_pa)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr())
            .sum::<f64>()
            .sqrt()
            / direct
                .pressure_rms_pa
                .iter()
                .map(|value| value.norm_sqr())
                .sum::<f64>()
                .sqrt()
                .max(1.0e-15);
        assert!(pressure_error < 1.0e-6, "pressure error {pressure_error:e}");
    }

    #[test]
    fn frequency_sweep_reuses_continuation_and_passes_all_qa_gates() {
        let model = coarse_model();
        let solutions = model
            .solve_sweep(&[200.0, 220.0, 500.0], Complex64::new(2.83, 0.0))
            .unwrap();
        assert!(!solutions[0].solver.initial_guess_reused);
        assert!(solutions[0].solver.preconditioner_rebuilt);
        assert!(solutions[1].solver.initial_guess_reused);
        assert!(!solutions[1].solver.preconditioner_rebuilt);
        assert!(solutions[2].solver.initial_guess_reused);
        assert!(solutions[2].solver.preconditioner_rebuilt);
        let qa = model.audit_sweep(&solutions).unwrap();
        assert!(qa.passes(1.0e-8, 1.0e-6), "sweep QA: {qa:?}");
    }

    #[test]
    fn export_contains_every_primary_and_derived_field() {
        let model = coarse_model();
        let solution = model
            .solve_frequency(500.0, Complex64::new(1.0, 0.0))
            .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let snapshot = model
            .export_solution(&solution, directory.path().join("axisymmetric.vtu"))
            .unwrap();
        let text = std::fs::read_to_string(snapshot.vtu_path).unwrap();
        for name in [
            field_names::MAGNETIC_POTENTIAL,
            field_names::MAGNETIC_FLUX_DENSITY,
            field_names::MAGNETIC_FIELD_STRENGTH,
            field_names::SOURCE_CURRENT_DENSITY,
            field_names::LORENTZ_FORCE_DENSITY,
            field_names::DISPLACEMENT,
            field_names::VELOCITY,
            field_names::STRESS,
            field_names::VON_MISES_STRESS,
            field_names::ACOUSTIC_PRESSURE,
            field_names::SOUND_PRESSURE_LEVEL,
            field_names::ACOUSTIC_INTENSITY,
        ] {
            assert!(text.contains(name), "missing exported field {name}");
        }

        let pvd_path = directory.path().join("axisymmetric_sweep.pvd");
        let sweep = model
            .export_sweep(&[200.0, 400.0], Complex64::new(1.0, 0.0), &pvd_path)
            .unwrap();
        assert_eq!(sweep.len(), 2);
        let pvd = std::fs::read_to_string(pvd_path).unwrap();
        assert!(pvd.contains("timestep=\"200\""));
        assert!(pvd.contains("timestep=\"400\""));
    }

    /// Uniform-strain fixture: a rectangular mesh under the linear field
    /// `u_r = 0, u_z = α·z` has exactly `εzz = α`, so every cell's stress is
    /// the analytic `[λα, λα, (λ+2μ)α, 0, 0, 0]` of its own region material.
    fn strain_fixture(markers: impl Fn(f64) -> i32) -> (Mesh, Vec<[Complex64; 2]>, f64) {
        let mut mesh = rectangular_mesh_triangles(0.005, 0.035, -0.001, 0.0, 6, 2);
        let cell_radii: Vec<f64> = mesh
            .elements
            .iter()
            .map(|element| element_centroid(&mesh, element.vertices()).0)
            .collect();
        for (element, radius) in mesh.elements.iter_mut().zip(cell_radii) {
            element.marker = markers(radius);
        }
        let alpha = 0.25; // du_z/dz
        let displacement: Vec<[Complex64; 2]> = mesh
            .nodes
            .iter()
            .map(|node| {
                [
                    Complex64::new(0.0, 0.0),
                    Complex64::new(alpha * node.y, 0.0),
                ]
            })
            .collect();
        (mesh, displacement, alpha)
    }

    #[test]
    fn stress_recovery_on_uniform_diaphragm_matches_legacy_hardcoded_material() {
        // The pre-fix code hardcoded IsotropicMaterial::new(2.4e9, 0.32) for
        // every cell; on a uniform-diaphragm fixture the per-region path must
        // reproduce that result exactly.
        let (mesh, displacement, alpha) = strain_fixture(|_| DIAPHRAGM);
        let stresses = structural_stress_cells(&mesh, &displacement);
        let legacy = IsotropicMaterial::new(2.4e9, 0.32);
        let (lambda, mu) = (legacy.lambda(), legacy.mu());
        for stress in &stresses {
            let expected = [
                Complex64::new(lambda * alpha, 0.0),
                Complex64::new(lambda * alpha, 0.0),
                Complex64::new((lambda + 2.0 * mu) * alpha, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ];
            for (actual, expected) in stress.iter().zip(expected) {
                assert!(
                    (*actual - expected).norm() <= 1e-12 * expected.norm().max(1.0),
                    "stress {actual:?} vs analytic {expected:?}"
                );
            }
        }
    }

    #[test]
    fn stress_recovery_uses_each_regions_own_material() {
        // Same uniform strain, mixed regions: the per-cell stress must scale
        // with the cell's own Lamé parameters, not the diaphragm's.
        let (mesh, displacement, alpha) =
            strain_fixture(|radius| if radius < 0.02 { DIAPHRAGM } else { SURROUND });
        assert!(mesh.elements.iter().any(|e| e.marker == DIAPHRAGM));
        assert!(mesh.elements.iter().any(|e| e.marker == SURROUND));
        let stresses = structural_stress_cells(&mesh, &displacement);
        for (element, stress) in mesh.elements.iter().zip(&stresses) {
            let material = moving_assembly_material(element.marker).unwrap();
            let (lambda, mu) = (material.lambda(), material.mu());
            let expected_zz = (lambda + 2.0 * mu) * alpha;
            assert!(
                (stress[2] - Complex64::new(expected_zz, 0.0)).norm() / expected_zz < 1e-12,
                "marker {} σzz={:?}, expected {expected_zz}",
                element.marker,
                stress[2]
            );
        }
        // The surround (5 MPa) is ~480× softer than the diaphragm (2.4 GPa);
        // under the same strain its stress must be lower by the exact
        // (λ+2μ) ratio — the hardcoded path made them equal.
        let dia = IsotropicMaterial::new(2.4e9, 0.32);
        let sur = IsotropicMaterial::new(5.0e6, 0.49);
        let ratio = (sur.lambda() + 2.0 * sur.mu()) / (dia.lambda() + 2.0 * dia.mu());
        let dia_zz = stresses[mesh
            .elements
            .iter()
            .position(|e| e.marker == DIAPHRAGM)
            .unwrap()][2]
            .re;
        let sur_zz = stresses[mesh
            .elements
            .iter()
            .position(|e| e.marker == SURROUND)
            .unwrap()][2]
            .re;
        assert!(
            (sur_zz / dia_zz - ratio).abs() / ratio < 1e-12,
            "surround/diaphragm σzz ratio {} vs material ratio {ratio}",
            sur_zz / dia_zz
        );
    }

    #[test]
    fn phase_agnostic_von_mises_envelope_is_conservative() {
        let stress = [
            Complex64::new(3.0, 0.0),
            Complex64::new(0.0, 2.0),
            Complex64::new(-1.0, 1.0),
            Complex64::default(),
            Complex64::default(),
            Complex64::new(0.5, -0.75),
        ];
        let phasor = complex_von_mises(stress).norm();
        let envelope = conservative_von_mises(stress);
        assert!(envelope >= phasor);

        let aligned = stress.map(|value| Complex64::new(value.norm(), 0.0));
        assert!(conservative_von_mises(aligned) >= complex_von_mises(aligned).norm());
    }
}
