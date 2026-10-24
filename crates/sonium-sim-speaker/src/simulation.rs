//! BEM acoustic simulation pipeline for loudspeaker radiation.
//!
//! Runs a frequency sweep with velocity boundary conditions on the
//! driver apertures and evaluates the sound field at measurement points.
//!
//! Key difference from scattering problems: there is no incident field.
//! The RHS comes entirely from the non-zero velocity BCs on the drivers,
//! which `build_tbem_system_with_beta` already handles during assembly.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
#[cfg(feature = "cuda")]
use sonium_compute::Backend;
#[cfg(any(feature = "cuda", feature = "metal"))]
use sonium_compute::DeviceBufferLike;
use sonium_compute::HostMatrix;
#[cfg(any(feature = "cuda", feature = "metal"))]
use sonium_compute::MixedPrecisionPolicy;
#[cfg(any(feature = "cuda", feature = "metal"))]
use std::collections::hash_map::DefaultHasher;
#[cfg(any(feature = "cuda", feature = "metal"))]
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use sonium_bem::core::assembly::hmatrix::{
    HMatrix, apply_tbem_stabilized_exact, apply_tbem_stabilized_exact_hermitian,
    build_hmatrix_stabilized,
};
#[cfg(feature = "cuda")]
use sonium_bem::core::assembly::tbem::CudaTbemAssemblyWorkspace;
use sonium_bem::core::assembly::tbem::{build_tbem_rhs_stabilized, build_tbem_system_stabilized};
use sonium_bem::core::mesh::octree::Octree;
#[cfg(feature = "cuda")]
use sonium_bem::core::postprocess::pressure::build_constant_field_operator;
use sonium_bem::core::postprocess::pressure::compute_scattered_field;
use sonium_bem::core::types::{
    BasisOrder, BoundaryCondition, Cluster, Element, ElementProperty, ElementType, PhysicsParams,
};
use sonium_fem::physics::acoustics::electroacoustics::thermal_coupling::VoiceCoilThermalModel;
use sonium_sim_speaker_modeling::MotorOperatingPoint;

use sonium_solvers::dense::DenseMatrix;
#[cfg(feature = "cuda")]
use sonium_solvers::direct::lu_solve_conj_transpose;
use sonium_solvers::direct::{LuFactorization, lu_factorize, lu_solve};
use sonium_solvers::iterative::{
    BlockGmresConfig, FgmresConfig, GcroDrConfig, GcroDrSubspace,
    block_gmres_preconditioned_with_guess, fgmres_preconditioned, gcro_dr_preconditioned,
    try_block_gmres_preconditioned_with_abort, try_fgmres_preconditioned_with_abort,
};
use sonium_solvers::lanes::{
    LaneConfig, host_memory_bytes, run_lanes_with_worker_budgets, try_plan_lanes,
};
use sonium_solvers::preconditioners::IluPreconditioner;
use sonium_solvers::sparse::CsrMatrix;
use sonium_solvers::traits::LinearOperator;
use std::rc::Rc;

use crate::coupling::{RadiationMatrix, RadiationMatrixDiagnostics};
use crate::mesh_gen::{MeshRegion, PinnedWaveguideMesh, SpeakerMesh};
use crate::model::LoudspeakerModel;
use sonium_sim_speaker_modeling::conventions::{
    bem_impedance_to_modeling, bem_peak_to_modeling_rms, modeling_rms_to_bem_peak,
};
use sonium_sim_speaker_modeling::{
    PowerLedger, PowerLedgerTolerance, ProductReferenceSweep, StructuralAcousticResponse,
    ThermalDriverState, ThermalNode,
};
#[cfg(test)]
use sonium_sim_speaker_modeling::{ThermalNetworkPower, ThermalNetworkState};

const DEFAULT_DIRECT_THRESHOLD: usize = 3000;
pub(crate) const EXTERIOR_TBEM_SIGN: f64 = 1.0;
#[cfg(any(feature = "cuda", feature = "metal"))]
const CUDA_COUPLED_PARITY_TOLERANCE: f64 = 5.0e-3;
#[cfg(any(feature = "cuda", feature = "metal"))]
const GPU_DERIVATIVE_ASSEMBLY_TOLERANCE: f64 = 5.0e-3;
#[cfg(feature = "cuda")]
const CUDA_ADJOINT_PARITY_TOLERANCE: f64 = 5.0e-3;
#[cfg(feature = "cuda")]
const CUDA_ADJOINT_INNER_PRODUCT_TOLERANCE: f64 = 1.0e-10;
/// DOF count at/above which the GPU BEM paths build a block-Jacobi
/// preconditioner (fixed 128-wide blocks) instead of point Jacobi.
///
/// Below this size the host-side explicit block inverses cost more than
/// they save next to diagonal scaling, so the same slot type runs with
/// `block_size = 1`, which reproduces point Jacobi exactly. Raises
/// lower the per-frequency setup cost but weaken the device
/// preconditioner; the 1000-DOF cut follows PERF-REPORT-20261013.md
/// step 4.3.
#[cfg(any(feature = "cuda", feature = "metal"))]
const GPU_BLOCK_JACOBI_MIN_DOFS: usize = 1000;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Result of the BEM simulation across all frequencies and measurement points.
#[derive(Clone)]
pub struct SimulationResult {
    /// Frequency vector (Hz).
    pub frequencies: Vec<f64>,
    /// SPL at each measurement point for each frequency.
    /// Row-major matrix with shape `(n_freqs, n_points)`.
    pub spl: HostMatrix<f64>,
    /// Phase (radians) at each measurement point for each frequency.
    /// Row-major matrix with shape `(n_freqs, n_points)`.
    pub phase: HostMatrix<f64>,
    /// Complex pressure amplitude (peak phasor, `e^{-iωt}` convention) at each
    /// measurement point for each frequency — the values `spl` and `phase`
    /// are derived from. Row-major matrix with shape `(n_freqs, n_points)`.
    /// Retained for complex directivity export
    /// ([`crate::directivity_export`]).
    pub complex_pressure: HostMatrix<Complex64>,
    /// Woofer electrical input impedance after BEM radiation back-loading.
    ///
    /// # Deprecated
    ///
    /// Retained for prescribed-result compatibility only. Producers must
    /// fill [`Self::driver_electrical_impedances_ohm`] (woofer is driver
    /// index 0); metrics and plots prefer the matrix whenever it carries
    /// data.
    #[deprecated(
        note = "retained for prescribed-result compatibility; producers must fill driver_electrical_impedances_ohm (woofer is driver 0)"
    )]
    pub woofer_electrical_impedance_ohm: Vec<Option<Complex64>>,
    /// Tweeter electrical input impedance after BEM radiation back-loading.
    ///
    /// # Deprecated
    ///
    /// Retained for prescribed-result compatibility only. Producers must
    /// fill [`Self::driver_electrical_impedances_ohm`] (tweeter is driver
    /// index 1); metrics and plots prefer the matrix whenever it carries
    /// data.
    #[deprecated(
        note = "retained for prescribed-result compatibility; producers must fill driver_electrical_impedances_ohm (tweeter is driver 1)"
    )]
    pub tweeter_electrical_impedance_ohm: Vec<Option<Complex64>>,
    /// Electrical impedances for every driver, indexed by stable driver
    /// order (`woofer`, `tweeter`, then `additional_drivers`).
    pub driver_electrical_impedances_ohm: Vec<Vec<Option<Complex64>>>,
    /// Radiation-impedance matrix for every frequency when typed coupling
    /// was active, in the modeling crate's convention.
    pub radiation_matrices: Vec<Option<Vec<Vec<Complex64>>>>,
    /// Reciprocity/passivity diagnostics for the typed radiation solve at
    /// each frequency. `None` means the frequency used prescribed legacy
    /// velocities or a backend path without a typed radiation network.
    pub radiation_matrix_diagnostics: Vec<Option<RadiationMatrixDiagnostics>>,
    /// Thermal operating-point provenance for each frequency and driver.
    /// `None` entries mean that driver has no configured thermal model.
    pub thermal_driver_states: Vec<[Option<ThermalDriverState>; 2]>,
    /// Thermal states for every driver in stable driver order.
    pub driver_thermal_states: Vec<Vec<Option<ThermalDriverState>>>,
    /// Product-level harmonic power closure for each frequency. `None` is
    /// used for prescribed-velocity or mixed typed/untyped runs where the
    /// electrical source does not own every acoustic power port.
    pub power_ledgers: Vec<Option<PowerLedger>>,
    /// Optional distributed FEM–BEM reference evidence for the same product
    /// and frequency grid. The native full-order state is intentionally
    /// reduced to validated observables by the modeling crate.
    pub product_reference_sweep: Option<ProductReferenceSweep>,
    /// Frequency-local structural panel/cone response used by acoustic
    /// boundary feedback, when the product enables that contract.
    pub structural_responses: Vec<Option<StructuralAcousticResponse>>,
    /// BEM peak driver velocities in stable driver order per frequency.
    /// Typed solves obtain these from the coupled network; legacy solves
    /// retain their prescribed boundary velocities.
    pub driver_velocities_bem: Vec<Vec<Complex64>>,
    /// Number of expensive uncompressed H-matrix residual applications made
    /// during this result. This is an explicit optimization cost metric.
    pub exact_residual_evaluations: usize,
    /// Backend, precision, validation, and deterministic artifact contract for
    /// this result. Consumers must preserve this record when a result is
    /// reduced, exported, or passed to an optimizer.
    pub provenance: SimulationProvenance,
}

/// Hard bounds for the portable speaker preview payload. The preview is a
/// reduced report, not a solver result; the native golden result remains the
/// source of truth for reanalysis and acceptance gates.
pub const WASM_PREVIEW_MAX_FREQUENCIES: usize = 512;
pub const WASM_PREVIEW_MAX_EVALUATION_POINTS: usize = 4_096;

/// Size-bounded JSON-friendly reduction of a native simulation result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerPreviewArtifact {
    pub frequencies: Vec<f64>,
    pub spl: Vec<Vec<f64>>,
    pub phase: Vec<Vec<f64>>,
    pub provenance: SimulationProvenance,
}

impl SimulationResult {
    /// Downsample a result into the only artifact shape eligible for a WASM
    /// preview. The returned provenance is always `wasm_preview` and the
    /// requested bounds are enforced before allocating the reduced payload.
    pub fn to_wasm_preview(
        &self,
        max_frequencies: usize,
        max_evaluation_points: usize,
    ) -> Result<SpeakerPreviewArtifact, SimulationError> {
        if max_frequencies == 0
            || max_frequencies > WASM_PREVIEW_MAX_FREQUENCIES
            || max_evaluation_points == 0
            || max_evaluation_points > WASM_PREVIEW_MAX_EVALUATION_POINTS
        {
            return Err(SimulationError::new(
                "WASM preview bounds exceed the portable artifact limits",
            ));
        }
        if self.frequencies.len() != self.spl.nrows()
            || self.frequencies.len() != self.phase.nrows()
            || self.spl.ncols() != self.phase.ncols()
        {
            return Err(SimulationError::new(
                "WASM preview requires frequency, SPL, and phase shapes to agree",
            ));
        }
        let frequency_indices = preview_indices(self.frequencies.len(), max_frequencies);
        let point_indices = preview_indices(self.spl.ncols(), max_evaluation_points);
        let frequencies: Vec<f64> = frequency_indices
            .iter()
            .map(|&index| self.frequencies[index])
            .collect();
        let spl = frequency_indices
            .iter()
            .map(|&frequency_index| {
                point_indices
                    .iter()
                    .map(|&point_index| self.spl[[frequency_index, point_index]])
                    .collect()
            })
            .collect();
        let phase = frequency_indices
            .iter()
            .map(|&frequency_index| {
                point_indices
                    .iter()
                    .map(|&point_index| self.phase[[frequency_index, point_index]])
                    .collect()
            })
            .collect();
        Ok(SpeakerPreviewArtifact {
            provenance: SimulationProvenance::wasm_preview(frequencies.len(), point_indices.len()),
            frequencies,
            spl,
            phase,
        })
    }
}

fn preview_indices(length: usize, maximum: usize) -> Vec<usize> {
    if length <= maximum {
        return (0..length).collect();
    }
    (0..maximum)
        .map(|index| index * (length - 1) / (maximum - 1))
        .collect()
}

/// Numerical backend that produced a speaker simulation result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SimulationBackend {
    /// Deterministic f64 CPU reference path.
    Cpu,
    /// CUDA dense path with f32 inner solves and f64 acceptance gates.
    Cuda,
    /// Metal dense path with f32 inner solves and f64 acceptance gates.
    Metal,
    /// Reduced/report-only path intended for a WASM preview.
    WasmPreview,
}

/// Whether a result is eligible for golden validation or is only a preview.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionProfile {
    Preview,
    #[default]
    Golden,
}

/// Arithmetic policy used by the simulation backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrecisionPolicy {
    F64,
    MixedF32InnerF64Gated,
}

/// Outcome of the residual/physical-observable acceptance gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidualGateStatus {
    NotRun,
    Passed,
}

/// Summary of the residual gates applied before a result was accepted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResidualGateReport {
    pub status: ResidualGateStatus,
    pub tolerance: f64,
    pub max_relative_residual: Option<f64>,
    pub max_physical_error: Option<f64>,
}

impl ResidualGateReport {
    fn not_run() -> Self {
        Self {
            status: ResidualGateStatus::NotRun,
            tolerance: 0.0,
            max_relative_residual: None,
            max_physical_error: None,
        }
    }

    fn passed(tolerance: f64) -> Self {
        Self {
            status: ResidualGateStatus::Passed,
            tolerance,
            max_relative_residual: None,
            max_physical_error: None,
        }
    }

    #[cfg(any(feature = "cuda", feature = "metal"))]
    fn passed_with_observations(
        tolerance: f64,
        max_relative_residual: f64,
        max_physical_error: f64,
    ) -> Self {
        Self {
            status: ResidualGateStatus::Passed,
            tolerance,
            max_relative_residual: Some(max_relative_residual),
            max_physical_error: Some(max_physical_error),
        }
    }
}

/// Stable ordering and scalar format for exported simulation artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactOrdering {
    FrequencyMajor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactFloatFormat {
    Ieee754Binary64,
}

/// Deterministic metadata shared by CEA/directivity/config sidecars.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SimulationArtifactMetadata {
    pub schema_version: u32,
    pub frequency_count: usize,
    pub evaluation_point_count: usize,
    pub ordering: ArtifactOrdering,
    pub float_format: ArtifactFloatFormat,
}

/// Typed execution provenance attached to every speaker simulation result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimulationProvenance {
    pub backend: SimulationBackend,
    pub execution_profile: ExecutionProfile,
    pub precision: PrecisionPolicy,
    pub residual_gate: ResidualGateReport,
    pub artifact: SimulationArtifactMetadata,
}

impl SimulationProvenance {
    const ARTIFACT_SCHEMA_VERSION: u32 = 1;

    fn solved(
        backend: SimulationBackend,
        execution_profile: ExecutionProfile,
        precision: PrecisionPolicy,
        residual_gate: ResidualGateReport,
        frequency_count: usize,
        evaluation_point_count: usize,
    ) -> Self {
        Self {
            backend,
            execution_profile,
            precision,
            residual_gate,
            artifact: SimulationArtifactMetadata {
                schema_version: Self::ARTIFACT_SCHEMA_VERSION,
                frequency_count,
                evaluation_point_count,
                ordering: ArtifactOrdering::FrequencyMajor,
                float_format: ArtifactFloatFormat::Ieee754Binary64,
            },
        }
    }

    /// Provenance for synthetic fixtures and derived test data. It is
    /// deliberately marked `not_run` so fixtures cannot be mistaken for a
    /// validated golden solve.
    pub fn synthetic(frequency_count: usize, evaluation_point_count: usize) -> Self {
        Self::solved(
            SimulationBackend::Cpu,
            ExecutionProfile::Preview,
            PrecisionPolicy::F64,
            ResidualGateReport::not_run(),
            frequency_count,
            evaluation_point_count,
        )
    }

    /// Provenance for a bounded reduced report intended for a WASM preview.
    /// It is explicitly preview-only and has no residual gate, so consumers
    /// must not promote it to a native golden or projected-reanalysis result.
    pub fn wasm_preview(frequency_count: usize, evaluation_point_count: usize) -> Self {
        Self::solved(
            SimulationBackend::WasmPreview,
            ExecutionProfile::Preview,
            PrecisionPolicy::F64,
            ResidualGateReport::not_run(),
            frequency_count,
            evaluation_point_count,
        )
    }
}

impl Default for SimulationProvenance {
    fn default() -> Self {
        Self::synthetic(0, 0)
    }
}

/// Execution state of one phase in the accelerator capability matrix.
///
/// Host fallback states are intentionally distinct from unsupported states so
/// callers can reject a performance claim without mistaking an unavailable
/// operation for a slow native implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuCapabilityState {
    Native,
    VendorNative,
    HostSharedFallback,
    HostStagedFallback,
    HostReference,
    Unsupported,
}

/// One configuration-level capability cell. The reason is part of the
/// serialized contract so a caller can distinguish an implemented lane from
/// a deliberate fail-closed boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuCapabilityCell {
    pub state: GpuCapabilityState,
    pub reason: String,
}

impl GpuCapabilityCell {
    fn new(state: GpuCapabilityState, reason: impl Into<String>) -> Self {
        Self {
            state,
            reason: reason.into(),
        }
    }
}

/// High-level speaker configurations are tracked separately from primitive
/// BEM capabilities. This prevents a native pair kernel from being mistaken
/// for support for a product feature whose coupled operator is not present.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuSpeakerConfigurationCapabilities {
    pub prescribed_forward: GpuCapabilityCell,
    pub typed_multi_driver: GpuCapabilityCell,
    pub crossover_phase: GpuCapabilityCell,
    pub coordinate_rotation: GpuCapabilityCell,
    pub body_fitted_projected_reanalysis: GpuCapabilityCell,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuCapabilityMatrix {
    pub regular_pair_assembly_c32: GpuCapabilityState,
    pub regular_pair_assembly_c64: GpuCapabilityState,
    pub near_pair_assembly: GpuCapabilityState,
    pub singular_pair_assembly: GpuCapabilityState,
    pub hypersingular_pair_assembly: GpuCapabilityState,
    pub field_evaluation: GpuCapabilityState,
    pub hmatrix_matvec: GpuCapabilityState,
    pub fmm_matvec: GpuCapabilityState,
    pub coupled_driver_solve: GpuCapabilityState,
    pub derivative_assembly: GpuCapabilityState,
    pub adjoint_solve: GpuCapabilityState,
    pub reduced_order_model: GpuCapabilityState,
    pub cea2034_postprocess: GpuCapabilityState,
    pub speaker_configurations: GpuSpeakerConfigurationCapabilities,
}

impl GpuCapabilityMatrix {
    /// Derive the public capability contract from the backend's machine
    /// readable table.  The high-level speaker phases that do not yet have a
    /// device operator are listed explicitly as unsupported; they are never
    /// silently routed through the CPU path.
    pub fn from_backend<B: sonium_compute::Backend>(backend: &B) -> Self {
        let capabilities = backend.capabilities();
        let state = |state: sonium_compute::CapabilityState| match state {
            sonium_compute::CapabilityState::Native => GpuCapabilityState::Native,
            sonium_compute::CapabilityState::VendorNative => GpuCapabilityState::VendorNative,
            sonium_compute::CapabilityState::HostSharedFallback => {
                GpuCapabilityState::HostSharedFallback
            }
            sonium_compute::CapabilityState::HostStagedFallback => {
                GpuCapabilityState::HostStagedFallback
            }
            sonium_compute::CapabilityState::Unsupported => GpuCapabilityState::Unsupported,
        };
        let prescribed_forward = collapse_gpu_capability_states(&[
            capabilities.bem.regular_pair_c32,
            capabilities.bem.regular_pair_c64,
            capabilities.bem.near_pair.c64,
            capabilities.bem.singular_pair.c64,
            capabilities.bem.hypersingular_pair.c64,
            capabilities.bem.field_evaluation.c64,
        ]);
        Self {
            regular_pair_assembly_c32: state(capabilities.bem.regular_pair_c32),
            regular_pair_assembly_c64: state(capabilities.bem.regular_pair_c64),
            near_pair_assembly: state(capabilities.bem.near_pair.c64),
            singular_pair_assembly: state(capabilities.bem.singular_pair.c64),
            hypersingular_pair_assembly: state(capabilities.bem.hypersingular_pair.c64),
            field_evaluation: state(capabilities.bem.field_evaluation.c64),
            hmatrix_matvec: state(capabilities.bem.hmatrix.c64),
            fmm_matvec: state(capabilities.bem.fmm.c64),
            coupled_driver_solve: GpuCapabilityState::Unsupported,
            derivative_assembly: state(backend.bem_wavenumber_derivative_capability()),
            adjoint_solve: GpuCapabilityState::Unsupported,
            reduced_order_model: GpuCapabilityState::Unsupported,
            cea2034_postprocess: GpuCapabilityState::HostReference,
            speaker_configurations: GpuSpeakerConfigurationCapabilities {
                prescribed_forward: GpuCapabilityCell::new(
                    prescribed_forward,
                    "constant-basis prescribed-driver forward requires native Complex64 regular, near, singular, hypersingular, and field paths; CPU residual gates are required before promotion",
                ),
                typed_multi_driver: GpuCapabilityCell::new(
                    GpuCapabilityState::Unsupported,
                    "backend primitive capabilities do not prove the product typed N-port path",
                ),
                crossover_phase: GpuCapabilityCell::new(
                    GpuCapabilityState::Unsupported,
                    "crossover transfer/phase coupling has no native CUDA product operator",
                ),
                coordinate_rotation: GpuCapabilityCell::new(
                    GpuCapabilityState::Unsupported,
                    "rotated speaker-coordinate dispatch is not a validated accelerator configuration",
                ),
                body_fitted_projected_reanalysis: GpuCapabilityCell::new(
                    GpuCapabilityState::Unsupported,
                    "body-fitted projected reanalysis remains on its dedicated CPU/FEM lane",
                ),
            },
        }
    }
}

/// Collapse the primitive capabilities required by one product-level phase.
/// An aggregate is unsupported if any required primitive is unsupported, and
/// it retains the strongest host-fallback class otherwise. This prevents a
/// single native pair kernel from advertising a complete speaker phase when a
/// required panel route or field path is absent.
fn collapse_gpu_capability_states(
    states: &[sonium_compute::CapabilityState],
) -> GpuCapabilityState {
    if states
        .iter()
        .any(|state| matches!(state, sonium_compute::CapabilityState::Unsupported))
    {
        GpuCapabilityState::Unsupported
    } else if states
        .iter()
        .any(|state| matches!(state, sonium_compute::CapabilityState::HostStagedFallback))
    {
        GpuCapabilityState::HostStagedFallback
    } else if states
        .iter()
        .any(|state| matches!(state, sonium_compute::CapabilityState::HostSharedFallback))
    {
        GpuCapabilityState::HostSharedFallback
    } else if states
        .iter()
        .any(|state| matches!(state, sonium_compute::CapabilityState::VendorNative))
    {
        GpuCapabilityState::VendorNative
    } else {
        GpuCapabilityState::Native
    }
}

/// Collapse the backend capabilities needed by the typed N-port and batched
/// radiation block-GMRES paths into one externally visible state. A host
/// fallback is not promoted to native; the production typed path must either
/// stay fully on the accelerator or reject before allocating/solving its
/// coupled operator.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn gpu_coupled_capability_state<B: sonium_compute::Backend>(backend: &B) -> GpuCapabilityState {
    let capabilities = backend.capabilities();
    let states = [
        capabilities.csr_spmv.c64,
        capabilities.csr_spmm.c64,
        capabilities.blas1.c64,
        capabilities.multi_dot.c64,
        capabilities.multi_dot_strided.c64,
        capabilities.basis_axpy_strided.c64,
    ];
    if states
        .iter()
        .any(|state| matches!(state, sonium_compute::CapabilityState::Unsupported))
    {
        GpuCapabilityState::Unsupported
    } else if states
        .iter()
        .any(|state| matches!(state, sonium_compute::CapabilityState::HostStagedFallback))
    {
        GpuCapabilityState::HostStagedFallback
    } else if states
        .iter()
        .any(|state| matches!(state, sonium_compute::CapabilityState::HostSharedFallback))
    {
        GpuCapabilityState::HostSharedFallback
    } else if states
        .iter()
        .any(|state| matches!(state, sonium_compute::CapabilityState::VendorNative))
    {
        GpuCapabilityState::VendorNative
    } else {
        GpuCapabilityState::Native
    }
}

/// Cache and residency evidence for a bounded accelerator sweep.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuCacheReport {
    pub identity: GpuResidencyIdentity,
    pub geometry_key: u64,
    pub mesh_resident_across_frequencies: bool,
    pub operator_structure_resident: bool,
    pub field_geometry_resident: bool,
    pub assembly_geometry_reused: bool,
    /// Number of CUDA chart workspaces built during the sweep.
    #[serde(default)]
    pub assembly_workspace_builds: usize,
    /// Rebuilds required by changes in the frequency-dependent singular rule.
    #[serde(default)]
    pub assembly_quadrature_schedule_updates: usize,
    pub matrix_value_updates: usize,
    pub preconditioner_updates: usize,
    pub rhs_buffer_reused: bool,
    pub solution_buffer_reused: bool,
    /// The bounded typed-driver network uses resident Complex64 CSR structure
    /// and work buffers when the production model contains only supported
    /// linear drivers. Matrix values and the Jacobi diagonal may still be
    /// updated when frequency-dependent coefficients change.
    pub coupled_operator_structure_resident: bool,
    pub coupled_rhs_buffer_reused: bool,
    pub coupled_solution_buffer_reused: bool,
    /// Resident column-major RHS/solution state for the typed driver's unit
    /// radiation loads.  One block solve replaces the old per-load scalar
    /// loop, so this is the explicit evidence required by the accelerator
    /// workload contract.
    pub coupled_block_rhs_resident: bool,
    pub coupled_block_solution_resident: bool,
    pub coupled_block_rhs_count: usize,
    pub coupled_block_iterations: usize,
    pub coupled_matrix_value_updates: usize,
    pub coupled_preconditioner_updates: usize,
    pub explicit_invalidations: usize,
}

/// Identity of state retained by one accelerator sweep.
///
/// Device buffers are reusable only while every member of this identity is
/// unchanged.  The CUDA speaker entry point scopes its resident workspace to
/// one call, so stale state cannot cross calls; keeping the identity in the
/// report makes that rule explicit for artifact consumers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuResidencyIdentity {
    pub schema_version: u32,
    pub backend: SimulationBackend,
    pub precision: PrecisionPolicy,
    pub device_index: Option<usize>,
    pub geometry_key: u64,
    pub frequency_grid_key: u64,
}

impl GpuResidencyIdentity {
    pub const SCHEMA_VERSION: u32 = 1;

    pub fn is_compatible(
        &self,
        backend: SimulationBackend,
        precision: PrecisionPolicy,
        device_index: Option<usize>,
        geometry_key: u64,
        frequency_grid_key: u64,
    ) -> bool {
        self.schema_version == Self::SCHEMA_VERSION
            && self.backend == backend
            && self.precision == precision
            && self.device_index == device_index
            && self.geometry_key == geometry_key
            && self.frequency_grid_key == frequency_grid_key
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuTimingReport {
    pub assembly_ms: u64,
    pub solve_ms: u64,
    pub residual_gate_ms: u64,
    pub field_ms: u64,
    pub reference_ms: u64,
    pub total_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuTransferReport {
    pub h2d_operations: usize,
    pub h2d_bytes: usize,
    pub d2h_operations: usize,
    pub d2h_bytes: usize,
    pub d2d_operations: usize,
    pub d2d_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuMemoryReport {
    pub total_device_bytes: Option<usize>,
    pub live_allocation_bytes: usize,
    pub allocation_high_water_bytes: usize,
    pub live_staging_bytes: usize,
    pub staging_high_water_bytes: usize,
    /// Live bytes sampled after the CUDA backend and its work buffers have
    /// been dropped. `None` means the backend-specific caller did not retain
    /// a teardown ledger (for example, the adjoint/Metal paths).
    #[serde(default)]
    pub post_teardown_live_allocation_bytes: Option<usize>,
    #[serde(default)]
    pub post_teardown_live_staging_bytes: Option<usize>,
}

/// Failure modes that a native accelerator job must surface as a typed error
/// or an explicit failed QA result.  The list is part of the report contract
/// so a successful timing row cannot hide an unclassified failure path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuFailureMode {
    Allocation,
    DeviceLoss,
    Cancellation,
    NonConvergence,
    ResidualMismatch,
    InvalidGeometry,
    StaleCache,
    InsufficientSampling,
    FailedRomPromotion,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuExecutionReport {
    pub schema_version: u32,
    pub requested_backend: SimulationBackend,
    pub actual_backend: SimulationBackend,
    pub device_index: Option<usize>,
    pub capability_matrix: GpuCapabilityMatrix,
    pub frequencies: usize,
    pub evaluation_points: usize,
    pub active_dofs: usize,
    pub cache: GpuCacheReport,
    pub timings: GpuTimingReport,
    pub transfers: GpuTransferReport,
    pub memory: GpuMemoryReport,
    pub residual_gate: ResidualGateStatus,
    pub max_relative_residual: f64,
    pub max_physical_error: f64,
    /// Maximum CPU/GPU parity error for the common typed-driver power ledger.
    pub max_coupled_power_error: f64,
    /// Maximum CPU/GPU parity error in coupled acoustic observables.
    pub max_coupled_observable_error: f64,
    /// Maximum CPU/GPU parity error in the linear voltage sensitivity
    /// `dv/dV`, the derivative contract for the bounded linear path.
    pub max_coupled_derivative_error: f64,
    /// Maximum relative error of the device pair-channel wavenumber
    /// derivative against the independent CPU f64 derivative assembly.
    pub max_derivative_assembly_error: f64,
    /// Independent reference checks actually executed, separate from capabilities.
    #[serde(default)]
    pub validation: GpuValidationReport,
    /// Whether the timing rows are eligible for a whole-pipeline accelerator
    /// performance claim. The bounded speaker path keeps host-side geometry
    /// setup and independent f64 reference work in the timed envelope, so its
    /// successful timings remain diagnostic evidence until those boundaries
    /// are moved onto the device.
    pub performance_eligible: bool,
    pub failure_modes_fail_closed: Vec<GpuFailureMode>,
}

impl GpuExecutionReport {
    pub const SCHEMA_VERSION: u32 = 3;
}

/// Independent GPU qualification or periodic production cross-checks.
///
/// Every policy retains the host f64 residual gate at every frequency.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum GpuValidationPolicy {
    /// Compare independent CPU f64 solves and fields at every frequency.
    #[default]
    Qualification,
    /// Check the first/last samples, periodic samples, and marginal residuals.
    Sentinel {
        every_k: usize,
        marginal_factor: f64,
    },
}

impl GpuValidationPolicy {
    /// Choose a conservative periodic policy for an already qualified workload.
    pub fn production() -> Self {
        Self::Sentinel {
            every_k: 10,
            marginal_factor: 10.0,
        }
    }

    fn validate(self) -> Result<(), SimulationError> {
        if let Self::Sentinel {
            every_k,
            marginal_factor,
        } = self
            && (every_k == 0 || !marginal_factor.is_finite() || marginal_factor < 1.0)
        {
            return Err(SimulationError::new(
                "GPU sentinel cadence must be positive and marginal_factor must be finite and >= 1",
            ));
        }
        Ok(())
    }

    #[cfg(any(test, feature = "cuda", feature = "metal"))]
    fn reference_due(self, index: usize, count: usize, residual: f64, tolerance: f64) -> bool {
        match self {
            Self::Qualification => true,
            Self::Sentinel {
                every_k,
                marginal_factor,
            } => {
                index == 0
                    || index + 1 == count
                    || index.is_multiple_of(every_k.max(1))
                    || !residual.is_finite()
                    || residual > tolerance / marginal_factor
            }
        }
    }
}

/// Coverage of independent CPU checks in a GPU sweep.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GpuValidationReport {
    pub policy: GpuValidationPolicy,
    pub reference_frequency_indices: Vec<usize>,
    pub derivative_frequency_indices: Vec<usize>,
    /// Frequencies re-solved on CUDA using the independent CPU reference operator.
    #[serde(default)]
    pub corrected_frequency_indices: Vec<usize>,
    /// Largest discrepancy before any explicitly requested reference correction.
    #[serde(default)]
    pub max_uncorrected_physical_error: f64,
}

#[cfg(test)]
mod gpu_validation_tests {
    use super::*;

    #[test]
    fn production_keeps_endpoints_periodic_and_marginal_checks() {
        let policy = GpuValidationPolicy::production();
        assert!(policy.reference_due(0, 24, 1e-9, 1e-6));
        assert!(policy.reference_due(23, 24, 1e-9, 1e-6));
        assert!(policy.reference_due(10, 24, 1e-9, 1e-6));
        assert!(!policy.reference_due(5, 24, 1e-9, 1e-6));
        assert!(policy.reference_due(5, 24, 2e-7, 1e-6));
        assert!(policy.reference_due(5, 24, f64::NAN, 1e-6));
        assert!(GpuValidationPolicy::Qualification.reference_due(5, 24, 0.0, 1e-6));
    }

    #[test]
    fn invalid_policy_rejected_and_old_config_defaults_to_qualification() {
        assert!(
            GpuValidationPolicy::Sentinel {
                every_k: 0,
                marginal_factor: 10.0
            }
            .validate()
            .is_err()
        );
        assert!(
            GpuValidationPolicy::Sentinel {
                every_k: 2,
                marginal_factor: f64::INFINITY
            }
            .validate()
            .is_err()
        );
        let config: SolverConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(config.gpu_validation, GpuValidationPolicy::Qualification);
        assert!(!config.gpu_wavenumber_derivative);
    }
}

/// Acceptance evidence for the bounded product-level CUDA adjoint lane.
///
/// This lane is deliberately narrower than the forward product runner: it
/// accepts prescribed constant-basis driver loads and a field objective, and
/// rejects typed/crossover/structural configurations before allocating a
/// device operator.  The Hermitian CSR structure and solve buffers remain
/// resident across the frequency sweep; CPU f64 factorization and the field
/// inner-product identity are independent acceptance checks.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuAdjointReport {
    pub schema_version: u32,
    pub requested_backend: SimulationBackend,
    pub actual_backend: SimulationBackend,
    pub device_index: usize,
    pub capability_matrix: GpuCapabilityMatrix,
    pub frequencies: usize,
    pub objective_points: usize,
    pub active_dofs: usize,
    pub geometry_key: u64,
    pub operator_structure_resident: bool,
    pub rhs_buffer_reused: bool,
    pub solution_buffer_reused: bool,
    pub matrix_value_updates: usize,
    pub preconditioner_updates: usize,
    pub residual_gate: ResidualGateStatus,
    pub max_relative_residual: f64,
    pub max_cpu_reference_error: f64,
    pub max_inner_product_error: f64,
    pub timings: GpuTimingReport,
    pub transfers: GpuTransferReport,
    pub memory: GpuMemoryReport,
}

#[cfg(feature = "cuda")]
impl GpuAdjointReport {
    pub const SCHEMA_VERSION: u32 = 1;
}

/// Surface-pressure guesses retained between nearby optimization designs.
///
/// `Clone` so batched candidate evaluation can seed one lane-local copy per
/// lane from a shared base design; lanes then update only their own copy.
#[derive(Debug, Default, Clone)]
pub(crate) struct SimulationWarmStart {
    surface_pressure_by_frequency: Vec<Option<Vec<Complex64>>>,
}

#[derive(Debug, Clone)]
pub struct SimulationError {
    message: String,
    cancelled: bool,
}

impl SimulationError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            cancelled: false,
        }
    }

    /// Construct the distinguished error returned when a caller-owned
    /// simulation control requests cancellation.  Keeping cancellation as a
    /// typed property avoids making Python/CLI callers infer terminal state
    /// from a human-readable error string.
    pub fn cancelled() -> Self {
        Self {
            message: "simulation cancelled".to_string(),
            cancelled: true,
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }
}

impl std::fmt::Display for SimulationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for SimulationError {}

/// Thread-safe control and progress state for a running native frequency
/// sweep.
///
/// The solver never invokes Python callbacks from Rayon workers.  Instead it
/// updates atomics at frequency boundaries and polls the cancellation flag at
/// frequency and iterative-solver checkpoints.  A Python adapter can poll
/// [`SimulationControl::progress`] from a separate thread while the native
/// call is executing with the GIL released.
#[derive(Clone, Debug)]
pub struct SimulationControl {
    cancelled: Arc<AtomicBool>,
    completed: Arc<AtomicUsize>,
    total: Arc<AtomicUsize>,
    current_frequency_bits: Arc<AtomicU64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimulationProgress {
    pub completed: usize,
    pub total: usize,
    pub frequency_hz: Option<f64>,
}

impl SimulationControl {
    pub fn new(total: usize) -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            completed: Arc::new(AtomicUsize::new(0)),
            total: Arc::new(AtomicUsize::new(total)),
            current_frequency_bits: Arc::new(AtomicU64::new(f64::NAN.to_bits())),
        }
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    pub fn cancellation_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancelled)
    }

    pub fn progress(&self) -> SimulationProgress {
        let frequency_bits = self.current_frequency_bits.load(Ordering::Acquire);
        SimulationProgress {
            completed: self.completed.load(Ordering::Acquire),
            total: self.total.load(Ordering::Acquire),
            frequency_hz: (!f64::from_bits(frequency_bits).is_nan())
                .then(|| f64::from_bits(frequency_bits)),
        }
    }

    fn report_frequency(&self, completed: usize, frequency_hz: f64) {
        self.completed.fetch_max(completed, Ordering::AcqRel);
        self.current_frequency_bits
            .store(frequency_hz.to_bits(), Ordering::Release);
    }

    fn complete_frequency(&self, frequency_hz: f64) {
        self.completed.fetch_add(1, Ordering::AcqRel);
        self.current_frequency_bits
            .store(frequency_hz.to_bits(), Ordering::Release);
    }

    fn check(&self) -> Result<(), SimulationError> {
        self.is_cancelled()
            .then(SimulationError::cancelled)
            .map_or(Ok(()), Err)
    }
}

/// Policy for the exact (uncompressed, O(N²) memory-free) TBEM residual
/// check after H-matrix solves.
///
/// The compressed-operator residual is always computed and gated against
/// [`SolverConfig::tolerance`]; this policy only controls the expensive
/// exact cross-check that catches compression drift the compressed residual
/// cannot see.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SentinelPolicy {
    /// Run the exact check after every H-matrix solve (legacy behaviour).
    EverySolve,
    /// Run the exact check every `every_k`-th solve, and additionally
    /// whenever the compressed residual is marginal — within
    /// `on_marginal_convergence_factor` of the tolerance gate (i.e.
    /// `residual > tolerance / factor`).
    Sentinel {
        /// Cadence of the periodic exact check, in solves.
        every_k: usize,
        /// Marginal-convergence trigger factor relative to the tolerance.
        on_marginal_convergence_factor: f64,
    },
    /// Never run the exact check.
    Off,
}

impl Default for SentinelPolicy {
    fn default() -> Self {
        SentinelPolicy::Sentinel {
            every_k: 10,
            on_marginal_convergence_factor: 10.0,
        }
    }
}

/// Configuration for the BEM solver within the simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SolverConfig {
    /// Whether this run is a reduced preview or a golden/production solve.
    pub execution_profile: ExecutionProfile,
    /// Independent CPU reference policy for GPU forward sweeps.
    pub gpu_validation: GpuValidationPolicy,
    /// Assemble and validate wavenumber derivatives when explicitly requested.
    /// Forward pressure studies do not need derivative assembly.
    pub gpu_wavenumber_derivative: bool,
    /// For prescribed CUDA sweeps, re-solve a failed physical reference check
    /// using the CPU reference operator uploaded to CUDA. The unchanged f64
    /// residual and physical gates must pass after correction. This adds CPU
    /// assembly work and is reported in `GpuValidationReport`.
    pub gpu_reference_correction: bool,
    /// Burton-Miller beta scale factor.
    ///
    /// The Burton-Miller formulation stabilises exterior resonances, so
    /// no CHIEF interior points are used (SIM-PERF-20261013.md section 2,
    /// gap 5).
    pub beta_scale: f64,
    /// DOF threshold: use direct LU at or below this, FGMRES+ILU above.
    ///
    /// Dense LU is cubic in the number of BEM unknowns, so the default avoids
    /// using it for the largest meshes that were previously inside the direct
    /// range while keeping the current 2932-DOF regression mesh on the
    /// validated LU path.
    pub direct_threshold: usize,
    /// FGMRES tolerance (used when N > direct_threshold).
    pub tolerance: f64,
    /// Maximum FGMRES iterations.
    pub max_iterations: usize,
    /// FGMRES restart parameter.
    ///
    /// Small restarts stall without deflation (measured Oct 2026:
    /// unpreconditioned restart-50 failed where restart-200 converged);
    /// the default 40 is safe because GCRO-DR recycling
    /// (`recycle_subspace_size = 20`) is on by default. Keep recycling
    /// enabled when lowering this (SIM-PERF-20261013.md section 2,
    /// gap 4).
    pub restart: usize,
    /// Diagonal shift for ILU preconditioner.
    pub ilu_shift: f64,
    /// Maximum adjacent frequency spacing, in octaves, for recycling a
    /// previous solution as the next FGMRES initial guess.
    pub max_recycled_guess_octaves: f64,
    /// Number of harmonic Ritz vectors recycled across consecutive frequencies
    /// via GCRO-DR (Parks–de Sturler 2006). `0` keeps the FGMRES + initial-
    /// guess-only path. When `> 0`, the iterative path uses
    /// GCRO-DR with the same ILU preconditioner; values around 15–25 are
    /// typical for acoustic frequency sweeps.
    pub recycle_subspace_size: usize,
    /// DOF threshold for switching from dense assembly to H-matrix assembly.
    /// H-matrix uses ACA compression for far-field blocks, giving O(N log N)
    /// memory and matvec cost. The default leaves a medium-size range for
    /// dense FGMRES+ILU between the direct-LU and H-matrix paths.
    pub hmatrix_threshold: usize,
    /// Relative ACA compression tolerance for H-matrix far-field blocks.
    /// This must not exceed the iterative convergence tolerance.
    pub hmatrix_aca_tolerance: f64,
    /// Re-apply the uncompressed O(N²) TBEM operator after an H-matrix solve.
    ///
    /// Legacy override: `true` forces [`SentinelPolicy::EverySolve`]
    /// regardless of [`SolverConfig::exact_residual_sentinel`]. Intended for
    /// solver development and reference checks. The compressed
    /// operator residual remains checked on every production solve.
    pub verify_exact_hmatrix_residual: bool,
    /// Exact-residual sentinel policy for H-matrix solves (forward and
    /// adjoint). Default: periodic every-10-solves check plus a
    /// marginal-convergence trigger at 10× the tolerance.
    pub exact_residual_sentinel: SentinelPolicy,
    /// Solve independent frequency lanes concurrently with Rayon.
    ///
    /// Each lane retains its own initial guess and GCRO-DR subspace, so
    /// recycling remains available without synchronizing solver state.
    pub parallel_frequency_sweep: bool,
    /// Hard cap on the number of parallel frequency lanes. `None` derives
    /// the lane count from the thread budget and the per-frequency memory
    /// estimate (see `frequency_sweep_memory_fraction`).
    pub max_frequency_lanes: Option<usize>,
    /// Fraction of host RAM that all frequency lanes' live solver state
    /// (operator, preconditioner, Krylov bases) may occupy together.
    /// Lanes are admitted until this budget is exhausted. When host memory
    /// cannot be determined, no memory cap is applied.
    pub frequency_sweep_memory_fraction: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            execution_profile: ExecutionProfile::Golden,
            gpu_validation: GpuValidationPolicy::Qualification,
            gpu_wavenumber_derivative: false,
            gpu_reference_correction: false,
            beta_scale: 4.0,
            direct_threshold: DEFAULT_DIRECT_THRESHOLD,
            tolerance: 1e-6,
            max_iterations: 300,
            restart: 40,
            ilu_shift: 0.1,
            max_recycled_guess_octaves: 0.25,
            recycle_subspace_size: 20,
            hmatrix_threshold: 5_000,
            hmatrix_aca_tolerance: 1e-8,
            verify_exact_hmatrix_residual: false,
            exact_residual_sentinel: SentinelPolicy::default(),
            parallel_frequency_sweep: true,
            max_frequency_lanes: None,
            frequency_sweep_memory_fraction: 0.5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SolverStrategy {
    Direct,
    DenseIterative,
    HMatrixIterative,
}

fn select_solver_strategy(n_dofs: usize, config: &SolverConfig) -> SolverStrategy {
    if n_dofs <= config.direct_threshold {
        SolverStrategy::Direct
    } else if n_dofs > config.hmatrix_threshold {
        SolverStrategy::HMatrixIterative
    } else {
        SolverStrategy::DenseIterative
    }
}

/// Estimated live bytes held by one frequency lane while solving a single
/// frequency point: assembled operator, dense ILU preconditioner (dense
/// iterative path only), Krylov basis (FGMRES restart + GCRO-DR recycled
/// subspace), and solution/RHS/residual scratch.
///
/// Used only for lane-count planning, so it favors simple closed-form
/// bounds over accuracy. No predictive H-matrix byte estimator exists
/// (`HMatrix::memory_bytes` is post-build), so the H-matrix term assumes
/// ACA stores ~`HMATRIX_COMPLEXES_PER_DOF_LOG · n · log2(n)` complexes.
fn estimate_frequency_live_bytes(n_dofs: usize, use_hmatrix: bool, config: &SolverConfig) -> u64 {
    const COMPLEX64_BYTES: u64 = 16;
    const HMATRIX_COMPLEXES_PER_DOF_LOG: u64 = 8;
    let n = n_dofs as u64;
    let operator = if use_hmatrix {
        let log2_n = (usize::BITS - n_dofs.leading_zeros()).max(1) as u64;
        HMATRIX_COMPLEXES_PER_DOF_LOG * n * log2_n * COMPLEX64_BYTES
    } else {
        n * n * COMPLEX64_BYTES
    };
    let dense_preconditioner = if !use_hmatrix && n_dofs > config.direct_threshold {
        n * n * COMPLEX64_BYTES
    } else {
        0
    };
    let krylov_basis =
        (config.restart as u64 + config.recycle_subspace_size as u64) * n * COMPLEX64_BYTES;
    let scratch = 4 * n * COMPLEX64_BYTES;
    operator + dense_preconditioner + krylov_basis + scratch
}

/// Estimated live bytes of one candidate's frequency sweep running a single
/// frequency lane, used for candidate-lane admission control in batched
/// optimization evaluations.
///
/// Re-meshes the given model, so callers should invoke it once per batch on
/// the base design; perturbed candidates share its topology, so their DOF
/// counts are close enough for planning.
pub(crate) fn estimate_frequency_sweep_live_bytes(
    model: &LoudspeakerModel,
    config: &SolverConfig,
) -> u64 {
    let speaker_mesh = super::mesh_gen::generate_loudspeaker_mesh(model);
    let n_dofs = speaker_mesh.triangles.len();
    let use_hmatrix = select_solver_strategy(n_dofs, config) == SolverStrategy::HMatrixIterative;
    estimate_frequency_live_bytes(n_dofs, use_hmatrix, config)
}

fn validate_solver_config(config: &SolverConfig) -> Result<(), SimulationError> {
    config.gpu_validation.validate()?;
    if config.direct_threshold >= config.hmatrix_threshold {
        return Err(SimulationError::new(
            "direct_threshold must be smaller than hmatrix_threshold so the dense iterative range is non-empty",
        ));
    }
    if !config.beta_scale.is_finite() || config.beta_scale <= 0.0 {
        return Err(SimulationError::new(
            "beta_scale must be finite and positive",
        ));
    }
    if !config.tolerance.is_finite() || config.tolerance <= 0.0 {
        return Err(SimulationError::new(
            "solver tolerance must be finite and positive",
        ));
    }
    if config.max_iterations == 0 {
        return Err(SimulationError::new("max_iterations must be at least 1"));
    }
    if config.restart == 0 {
        return Err(SimulationError::new("restart must be at least 1"));
    }
    if !config.ilu_shift.is_finite() || config.ilu_shift < 0.0 {
        return Err(SimulationError::new(
            "ILU shift must be finite and non-negative",
        ));
    }
    if !config.max_recycled_guess_octaves.is_finite() || config.max_recycled_guess_octaves < 0.0 {
        return Err(SimulationError::new(
            "maximum recycled frequency step must be finite and non-negative",
        ));
    }
    if config.recycle_subspace_size >= config.restart && config.recycle_subspace_size != 0 {
        return Err(SimulationError::new(
            "recycle_subspace_size must be smaller than restart",
        ));
    }
    if !config.frequency_sweep_memory_fraction.is_finite()
        || config.frequency_sweep_memory_fraction <= 0.0
        || config.frequency_sweep_memory_fraction > 1.0
    {
        return Err(SimulationError::new(
            "frequency_sweep_memory_fraction must be finite and in (0, 1]",
        ));
    }
    if !config.hmatrix_aca_tolerance.is_finite()
        || config.hmatrix_aca_tolerance <= 0.0
        || config.hmatrix_aca_tolerance > config.tolerance
    {
        return Err(SimulationError::new(
            "H-matrix ACA tolerance must be finite, positive, and no larger than solver tolerance",
        ));
    }
    Ok(())
}

fn validate_evaluation_points(eval_points: &[[f64; 3]]) -> Result<(), SimulationError> {
    if eval_points.is_empty() {
        return Err(SimulationError::new(
            "at least one evaluation point is required",
        ));
    }
    if let Some((index, _)) = eval_points
        .iter()
        .enumerate()
        .find(|(_, point)| point.iter().any(|coordinate| !coordinate.is_finite()))
    {
        return Err(SimulationError::new(format!(
            "evaluation point {index} contains a non-finite coordinate"
        )));
    }
    Ok(())
}

fn validate_field_map_inputs(
    frequency: f64,
    extent: f64,
    resolution: f64,
) -> Result<(), SimulationError> {
    if !frequency.is_finite() || frequency <= 0.0 {
        return Err(SimulationError::new(
            "field-map frequency must be finite and positive",
        ));
    }
    if !extent.is_finite() || extent <= 0.0 {
        return Err(SimulationError::new(
            "field-map extent must be finite and positive",
        ));
    }
    if !resolution.is_finite() || resolution <= 0.0 {
        return Err(SimulationError::new(
            "field-map resolution must be finite and positive",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Dense operator wrapper (for FGMRES)
// ---------------------------------------------------------------------------

struct DenseOperator<'a> {
    matrix: &'a DenseMatrix<Complex64>,
}

impl<'a> LinearOperator<Complex64> for DenseOperator<'a> {
    fn num_rows(&self) -> usize {
        self.matrix.nrows()
    }
    fn num_cols(&self) -> usize {
        self.matrix.ncols()
    }
    fn apply(&self, x: &[Complex64]) -> Vec<Complex64> {
        self.matrix.matvec(x)
    }
    fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        self.matrix.matvec_into(x, y);
    }
    fn apply_block(&self, x: &DenseMatrix<Complex64>, y: &mut DenseMatrix<Complex64>) {
        // One row-major GEMM (A n×n · X n×k → Y n×k) instead of k matvecs.
        self.matrix.matmul_into(x, y);
    }
    fn apply_transpose(&self, x: &[Complex64]) -> Vec<Complex64> {
        let mut y = vec![Complex64::new(0.0, 0.0); self.matrix.ncols()];
        self.matrix.matvec_transpose_into(x, &mut y);
        y
    }
    fn apply_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        self.matrix.matvec_transpose_into(x, y);
    }
}

// ---------------------------------------------------------------------------
// Adaptive solver: LU (small) or FGMRES+ILU (large)
// ---------------------------------------------------------------------------

pub(crate) fn solve_bem_system(
    matrix: &DenseMatrix<Complex64>,
    rhs: &[Complex64],
    x0: Option<&[Complex64]>,
    config: &SolverConfig,
    gcro_dr_subspace: Option<&mut GcroDrSubspace<Complex64>>,
) -> Result<Vec<Complex64>, SimulationError> {
    solve_bem_system_with_preconditioner_control(
        matrix,
        rhs,
        x0,
        config,
        gcro_dr_subspace,
        None,
        None,
    )
}

fn dense_ilu_preconditioner(
    matrix: &DenseMatrix<Complex64>,
    config: &SolverConfig,
) -> IluPreconditioner<Complex64> {
    let mut csr = CsrMatrix::from_dense(matrix, 0.0);
    csr.add_diagonal(Complex64::new(config.ilu_shift, 0.0));
    IluPreconditioner::from_csr(&csr)
}

fn solve_bem_system_with_preconditioner(
    matrix: &DenseMatrix<Complex64>,
    rhs: &[Complex64],
    x0: Option<&[Complex64]>,
    config: &SolverConfig,
    gcro_dr_subspace: Option<&mut GcroDrSubspace<Complex64>>,
    reused_preconditioner: Option<&IluPreconditioner<Complex64>>,
) -> Result<Vec<Complex64>, SimulationError> {
    solve_bem_system_with_preconditioner_control(
        matrix,
        rhs,
        x0,
        config,
        gcro_dr_subspace,
        reused_preconditioner,
        None,
    )
}

fn solve_bem_system_with_preconditioner_control(
    matrix: &DenseMatrix<Complex64>,
    rhs: &[Complex64],
    x0: Option<&[Complex64]>,
    config: &SolverConfig,
    gcro_dr_subspace: Option<&mut GcroDrSubspace<Complex64>>,
    reused_preconditioner: Option<&IluPreconditioner<Complex64>>,
    control: Option<&SimulationControl>,
) -> Result<Vec<Complex64>, SimulationError> {
    if let Some(control) = control {
        control.check()?;
    }
    let n = matrix.nrows();
    if matrix.ncols() != n || rhs.len() != n {
        return Err(SimulationError::new(
            "BEM system dimensions are inconsistent",
        ));
    }
    if !complex_values_are_finite(matrix.as_slice()) || !complex_values_are_finite(rhs) {
        return Err(SimulationError::new(
            "BEM matrix or right-hand side contains non-finite values",
        ));
    }
    if n <= config.direct_threshold {
        let sol = lu_solve(matrix, rhs)
            .map_err(|err| SimulationError::new(format!("LU solve failed: {err}")))?;
        if let Some(control) = control {
            control.check()?;
        }
        let residual = relative_residual(matrix, &sol, rhs);
        if !complex_values_are_finite(&sol) || !residual.is_finite() {
            return Err(SimulationError::new(
                "LU solve produced a non-finite solution or residual",
            ));
        }
        validate_true_residual("LU solve", residual, config)?;
        return Ok(sol);
    }

    let generated_preconditioner;
    let precond = if let Some(preconditioner) = reused_preconditioner {
        preconditioner
    } else {
        generated_preconditioner = dense_ilu_preconditioner(matrix, config);
        &generated_preconditioner
    };
    let operator = DenseOperator { matrix };
    let inner_tolerance = config.tolerance * 0.1;

    let (x_out, iters, converged, label) = if let Some(sub) = gcro_dr_subspace {
        if let Some(control) = control {
            let cfg = FgmresConfig {
                max_iterations: config.max_iterations,
                restart: config.restart,
                tolerance: inner_tolerance,
                print_interval: 0,
                convergence_check_interval: 1,
                exact_residual_check_interval: 1,
            };
            let r = try_fgmres_preconditioned_with_abort(&operator, precond, rhs, x0, &cfg, || {
                control.is_cancelled()
            })
            .map_err(|error| {
                if control.is_cancelled() {
                    SimulationError::cancelled()
                } else {
                    SimulationError::new(format!("controlled FGMRES failed: {error}"))
                }
            })?;
            if control.is_cancelled() {
                return Err(SimulationError::cancelled());
            }
            (r.x, r.iterations, r.converged, "FGMRES+ILU (controlled)")
        } else {
            let cfg = GcroDrConfig {
                max_iterations: config.max_iterations,
                restart: config.restart,
                tolerance: inner_tolerance,
                print_interval: 0,
            };
            let r = gcro_dr_preconditioned(&operator, precond, rhs, x0, sub, &cfg);
            (r.x, r.iterations, r.converged, "GCRO-DR+ILU")
        }
    } else {
        let cfg = FgmresConfig {
            max_iterations: config.max_iterations,
            restart: config.restart,
            tolerance: inner_tolerance,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 1,
        };
        if let Some(control) = control {
            let r = try_fgmres_preconditioned_with_abort(&operator, precond, rhs, x0, &cfg, || {
                control.is_cancelled()
            })
            .map_err(|error| {
                if control.is_cancelled() {
                    SimulationError::cancelled()
                } else {
                    SimulationError::new(format!("controlled FGMRES failed: {error}"))
                }
            })?;
            if control.is_cancelled() {
                return Err(SimulationError::cancelled());
            }
            (r.x, r.iterations, r.converged, "FGMRES+ILU (controlled)")
        } else {
            let r = fgmres_preconditioned(&operator, precond, rhs, x0, &cfg);
            (r.x, r.iterations, r.converged, "FGMRES+ILU")
        }
    };

    if let Some(control) = control {
        control.check()?;
    }

    let residual = relative_residual(matrix, &x_out, rhs);
    eprintln!("  {label}: {iters} iters, converged={converged}, true residual={residual:.3e}",);

    if !converged || !complex_values_are_finite(&x_out) {
        return Err(SimulationError::new(format!(
            "{label} failed to converge: reported converged={converged}, true residual={residual:.3e}, tolerance={:.3e}",
            config.tolerance
        )));
    }
    validate_true_residual(label, residual, config)?;

    Ok(x_out)
}

// ---------------------------------------------------------------------------
// Adjoint solves with forward factor/preconditioner reuse
// ---------------------------------------------------------------------------

/// Solver state retained from a forward BEM solve so the discrete adjoint
/// can reuse it for the conjugate-transpose (Aᴴ) solve at the same frequency.
pub(crate) enum BemSolverState {
    /// Dense LU factors of A; the adjoint reuses them for the Aᴴ triangular
    /// solves instead of factorizing Aᴴ from scratch (PA = LU implies
    /// Aᴴ = UᴴLᴴP).
    Lu(LuFactorization<Complex64>),
    /// ILU preconditioner built for A; reused as a (weaker) preconditioner
    /// for the Aᴴ FGMRES. Shared across nearby frequencies under the same
    /// octave-recycling policy as the production forward sweep.
    Ilu(Rc<IluPreconditioner<Complex64>>),
}

/// ILU recycling slot for an adjoint forward sweep.
///
/// Mirrors the dense-preconditioner policy of `FrequencySweepState`: the ILU
/// is regenerated only when the frequency step exceeds
/// [`SolverConfig::max_recycled_guess_octaves`].
#[derive(Default)]
pub(crate) struct BemAdjointSweepState {
    preconditioner: Option<(f64, Rc<IluPreconditioner<Complex64>>)>,
}

impl BemAdjointSweepState {
    /// Solve `A x = b` like [`solve_bem_system`], additionally retaining the
    /// LU factors or ILU preconditioner for [`solve_bem_adjoint_system`].
    pub(crate) fn solve_forward(
        &mut self,
        matrix: &DenseMatrix<Complex64>,
        rhs: &[Complex64],
        x0: Option<&[Complex64]>,
        frequency_hz: f64,
        config: &SolverConfig,
    ) -> Result<(Vec<Complex64>, BemSolverState), SimulationError> {
        let n = matrix.nrows();
        if matrix.ncols() != n || rhs.len() != n {
            return Err(SimulationError::new(
                "BEM system dimensions are inconsistent",
            ));
        }
        if !complex_values_are_finite(matrix.as_slice()) || !complex_values_are_finite(rhs) {
            return Err(SimulationError::new(
                "BEM matrix or right-hand side contains non-finite values",
            ));
        }
        if n <= config.direct_threshold {
            let factor = lu_factorize(matrix)
                .map_err(|err| SimulationError::new(format!("LU solve failed: {err}")))?;
            let sol = factor
                .solve(rhs)
                .map_err(|err| SimulationError::new(format!("LU solve failed: {err}")))?;
            let residual = relative_residual(matrix, &sol, rhs);
            if !complex_values_are_finite(&sol) || !residual.is_finite() {
                return Err(SimulationError::new(
                    "LU solve produced a non-finite solution or residual",
                ));
            }
            validate_true_residual("LU solve", residual, config)?;
            return Ok((sol, BemSolverState::Lu(factor)));
        }

        let recycle_preconditioner =
            self.preconditioner
                .as_ref()
                .is_some_and(|(preconditioner_frequency, _)| {
                    should_recycle_initial_guess(
                        Some(*preconditioner_frequency),
                        frequency_hz,
                        config.max_recycled_guess_octaves,
                    )
                });
        if !recycle_preconditioner {
            self.preconditioner = Some((
                frequency_hz,
                Rc::new(dense_ilu_preconditioner(matrix, config)),
            ));
        }
        let preconditioner = Rc::clone(
            &self
                .preconditioner
                .as_ref()
                .expect("adjoint sweep preconditioner just built")
                .1,
        );
        let sol = solve_bem_system_with_preconditioner(
            matrix,
            rhs,
            x0,
            config,
            None,
            Some(preconditioner.as_ref()),
        )?;
        Ok((sol, BemSolverState::Ilu(preconditioner)))
    }
}

/// Solve the discrete-adjoint system `Aᴴ λ = rhs` reusing the retained
/// forward solver state.
///
/// The same true-residual gates as the forward solves apply, so a reused
/// factorization or preconditioner can only cost iterations, never
/// correctness.
pub(crate) fn solve_bem_adjoint_system(
    matrix: &DenseMatrix<Complex64>,
    rhs: &[Complex64],
    state: &BemSolverState,
    config: &SolverConfig,
) -> Result<Vec<Complex64>, SimulationError> {
    match state {
        BemSolverState::Lu(factor) => {
            let sol = factor
                .solve_conj_transpose(rhs)
                .map_err(|err| SimulationError::new(format!("LU adjoint solve failed: {err}")))?;
            let residual = relative_residual_adjoint(matrix, &sol, rhs);
            if !complex_values_are_finite(&sol) || !residual.is_finite() {
                return Err(SimulationError::new(
                    "LU adjoint solve produced a non-finite solution or residual",
                ));
            }
            validate_true_residual("LU adjoint solve", residual, config)?;
            Ok(sol)
        }
        BemSolverState::Ilu(preconditioner) => {
            // Solve Aᴴ λ = rhs with the operator's Hermitian applied on the
            // fly (no O(n²) materialization of Aᴴ) and the Hermitian of the
            // forward ILU as the preconditioner — Mᴴ = U⁻ᴴL⁻ᴴ is the
            // mathematically correct reuse of the forward factors for the
            // adjoint operator, not the forward ILU misapplied to Aᴴ.
            let operator = HermitianDenseOperator { matrix };
            let adjoint_preconditioner = AdjointPreconditioner {
                forward: preconditioner.as_ref(),
            };
            let cfg = FgmresConfig {
                max_iterations: config.max_iterations,
                restart: config.restart,
                tolerance: config.tolerance * 0.1,
                print_interval: 0,
                convergence_check_interval: 1,
                exact_residual_check_interval: 1,
            };
            let result = fgmres_preconditioned(&operator, &adjoint_preconditioner, rhs, None, &cfg);
            let sol = result.x;
            let residual = relative_residual_adjoint(matrix, &sol, rhs);
            eprintln!(
                "  FGMRES+ILU^H adjoint: {} iters, converged={}, true residual={residual:.3e}",
                result.iterations, result.converged
            );
            if !result.converged || !complex_values_are_finite(&sol) {
                return Err(SimulationError::new(format!(
                    "FGMRES+ILU^H adjoint failed to converge: reported converged={}, \
                     true residual={residual:.3e}, tolerance={:.3e}",
                    result.converged, config.tolerance
                )));
            }
            validate_true_residual("FGMRES+ILU^H adjoint", residual, config)?;
            Ok(sol)
        }
    }
}

/// Solve the discrete-adjoint system `Aᴴ λ = rhs` on an H-matrix operator.
///
/// Mirrors the forward [`solve_bem_system_hmatrix`] strategy: FGMRES with
/// the Jacobi preconditioner extracted from the H-matrix diagonal blocks,
/// here applied through its Hermitian. The operator runs the parallel
/// field-cluster-scheduled `Aᴴ` matvec (Task B5.1); no dense operator is
/// ever materialized.
///
/// Returns the solution together with its residual against the compressed
/// operator so the caller can feed the exact-residual sentinel policy
/// without a second matvec.
pub(crate) fn solve_bem_adjoint_system_hmatrix(
    hmatrix: &sonium_bem::core::assembly::hmatrix::HMatrix,
    rhs: &[Complex64],
    config: &SolverConfig,
) -> Result<(Vec<Complex64>, f64), SimulationError> {
    use sonium_bem::core::assembly::hmatrix::HMatrixBlockJacobi;

    let forward_preconditioner = HMatrixBlockJacobi::from_hmatrix(hmatrix);
    let adjoint_preconditioner = AdjointPreconditioner {
        forward: &forward_preconditioner,
    };
    let operator = HmatrixAdjointOperator { hmatrix };
    let cfg = FgmresConfig {
        max_iterations: config.max_iterations,
        restart: config.restart,
        tolerance: config.tolerance * 0.1,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 1,
    };
    let result = fgmres_preconditioned(&operator, &adjoint_preconditioner, rhs, None, &cfg);
    let sol = result.x;
    let residual = relative_residual_hmatrix_adjoint(hmatrix, &sol, rhs);
    eprintln!(
        "  FGMRES+Hmatrix adjoint: {} iters, converged={}, true residual={residual:.3e}",
        result.iterations, result.converged
    );
    if !result.converged || !complex_values_are_finite(&sol) {
        return Err(SimulationError::new(format!(
            "FGMRES+Hmatrix adjoint failed to converge: reported converged={}, \
             true residual={residual:.3e}, tolerance={:.3e}",
            result.converged, config.tolerance
        )));
    }
    validate_true_residual("FGMRES+Hmatrix adjoint", residual, config)?;
    Ok((sol, residual))
}

/// `LinearOperator` view of `Aᴴ` over a borrowed dense matrix: applies the
/// conjugate transpose via `matvec_hermitian_into` without materializing
/// `Aᴴ` (the previous adjoint path allocated an `O(n²)` conj-transposed
/// copy per solve).
struct HermitianDenseOperator<'a> {
    matrix: &'a DenseMatrix<Complex64>,
}

impl LinearOperator<Complex64> for HermitianDenseOperator<'_> {
    fn num_rows(&self) -> usize {
        self.matrix.ncols()
    }

    fn num_cols(&self) -> usize {
        self.matrix.nrows()
    }

    fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        self.matrix.matvec_hermitian_into(x, y);
    }

    fn apply_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        // (Aᴴ)ᵀ = conj(A), applied as conj(A · conj(x)).
        let x_conj: Vec<Complex64> = x.iter().map(|v| v.conj()).collect();
        self.matrix.matvec_into(&x_conj, y);
        for value in y.iter_mut() {
            *value = value.conj();
        }
    }
}

/// `LinearOperator` view of `Aᴴ` over a borrowed H-matrix: applies the
/// conjugate transpose through the parallel field-cluster schedule.
struct HmatrixAdjointOperator<'a> {
    hmatrix: &'a sonium_bem::core::assembly::hmatrix::HMatrix,
}

impl LinearOperator<Complex64> for HmatrixAdjointOperator<'_> {
    fn num_rows(&self) -> usize {
        self.hmatrix.num_dofs
    }

    fn num_cols(&self) -> usize {
        self.hmatrix.num_dofs
    }

    fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        self.hmatrix.matvec_hermitian_into(x, y);
    }

    fn apply_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        // (Aᴴ)ᵀ = conj(A), applied as conj(A · conj(x)).
        let x_conj: Vec<Complex64> = x.iter().map(|v| v.conj()).collect();
        self.hmatrix.matvec_into(&x_conj, y);
        for value in y.iter_mut() {
            *value = value.conj();
        }
    }
}

/// Preconditioner adapter applying the Hermitian of a forward
/// preconditioner (`Mᴴ`), so a factorization built for `A` can precondition
/// the adjoint system `Aᴴ λ = g`.
struct AdjointPreconditioner<'a, P: sonium_solvers::traits::Preconditioner<Complex64>> {
    forward: &'a P,
}

impl<P: sonium_solvers::traits::Preconditioner<Complex64>>
    sonium_solvers::traits::Preconditioner<Complex64> for AdjointPreconditioner<'_, P>
{
    fn apply(&self, r: &[Complex64]) -> Vec<Complex64> {
        self.forward.apply_hermitian(r)
    }

    fn apply_into(&self, r: &[Complex64], y: &mut [Complex64]) {
        self.forward.apply_hermitian_into(r, y);
    }
}

/// Conjugate transpose `Aᴴ` of a dense matrix.
#[cfg(test)]
fn hermitian_transpose(matrix: &DenseMatrix<Complex64>) -> DenseMatrix<Complex64> {
    DenseMatrix::from_row_major_vec(
        matrix.ncols(),
        matrix.nrows(),
        (0..matrix.ncols())
            .flat_map(|row| (0..matrix.nrows()).map(move |column| matrix[[column, row]].conj()))
            .collect(),
    )
}

fn relative_residual_adjoint(
    matrix: &DenseMatrix<Complex64>,
    x: &[Complex64],
    rhs: &[Complex64],
) -> f64 {
    let mut ax = vec![Complex64::new(0.0, 0.0); matrix.ncols()];
    matrix.matvec_hermitian_into(x, &mut ax);
    relative_residual_from_applied(&ax, rhs)
}

fn complex_values_are_finite(values: &[Complex64]) -> bool {
    values
        .iter()
        .all(|value| value.re.is_finite() && value.im.is_finite())
}

fn relative_residual(matrix: &DenseMatrix<Complex64>, x: &[Complex64], rhs: &[Complex64]) -> f64 {
    let ax = matrix.matvec(x);
    relative_residual_from_applied(&ax, rhs)
}

fn relative_residual_from_applied(ax: &[Complex64], rhs: &[Complex64]) -> f64 {
    if ax.len() != rhs.len() {
        return f64::NAN;
    }
    let residual_norm = ax
        .iter()
        .zip(rhs.iter())
        .map(|(ax_i, rhs_i)| (*rhs_i - *ax_i).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let rhs_norm = rhs.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

    residual_norm / rhs_norm.max(1e-30)
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn relative_complex_difference(actual: Complex64, reference: Complex64) -> f64 {
    if [actual.re, actual.im, reference.re, reference.im]
        .iter()
        .any(|value| !value.is_finite())
    {
        return f64::INFINITY;
    }
    let numerator = (actual.re - reference.re).hypot(actual.im - reference.im);
    let denominator = reference.re.hypot(reference.im).max(1.0e-30);
    let error = numerator / denominator;
    if error.is_finite() {
        error
    } else {
        f64::INFINITY
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn relative_system_difference(
    actual: &sonium_bem::core::assembly::tbem::TbemSystem,
    reference: &sonium_bem::core::assembly::tbem::TbemSystem,
) -> f64 {
    if actual.num_dofs != reference.num_dofs
        || actual.matrix.nrows() != reference.matrix.nrows()
        || actual.matrix.ncols() != reference.matrix.ncols()
        || actual.rhs.len() != reference.rhs.len()
    {
        return f64::INFINITY;
    }
    let matrix_error = actual
        .matrix
        .as_slice()
        .iter()
        .zip(reference.matrix.as_slice())
        .map(|(actual, reference)| (*actual - *reference).norm_sqr())
        .sum::<f64>()
        .sqrt()
        / reference
            .matrix
            .as_slice()
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(1.0e-30);
    let rhs_error = actual
        .rhs
        .iter()
        .zip(&reference.rhs)
        .map(|(actual, reference)| (*actual - *reference).norm_sqr())
        .sum::<f64>()
        .sqrt()
        / reference
            .rhs
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(1.0e-30);
    matrix_error.max(rhs_error)
}

#[cfg(feature = "cuda")]
fn complex_inner_product(lhs: &[Complex64], rhs: &[Complex64]) -> Complex64 {
    lhs.iter()
        .zip(rhs)
        .map(|(lhs, rhs)| lhs.conj() * *rhs)
        .sum()
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn relative_scalar_difference(actual: f64, reference: f64) -> f64 {
    if !actual.is_finite() || !reference.is_finite() {
        return f64::INFINITY;
    }
    let error = (actual - reference).abs() / reference.abs().max(1.0e-30);
    if error.is_finite() {
        error
    } else {
        f64::INFINITY
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn power_ledger_difference(actual: PowerLedger, reference: PowerLedger) -> f64 {
    [
        relative_scalar_difference(actual.supplied_w, reference.supplied_w),
        relative_scalar_difference(actual.copper_loss_w, reference.copper_loss_w),
        relative_scalar_difference(actual.magnetic_loss_w, reference.magnetic_loss_w),
        relative_scalar_difference(actual.mechanical_loss_w, reference.mechanical_loss_w),
        relative_scalar_difference(actual.structural_loss_w, reference.structural_loss_w),
        relative_scalar_difference(
            actual.internal_acoustic_loss_w,
            reference.internal_acoustic_loss_w,
        ),
        relative_scalar_difference(
            actual.exterior_acoustic_power_w,
            reference.exterior_acoustic_power_w,
        ),
        relative_scalar_difference(actual.thermal_storage_w, reference.thermal_storage_w),
        if actual.residual_w.is_finite()
            && reference.residual_w.is_finite()
            && reference.supplied_w.is_finite()
        {
            let error = (actual.residual_w - reference.residual_w).abs()
                / reference.supplied_w.abs().max(1.0e-12);
            if error.is_finite() {
                error
            } else {
                f64::INFINITY
            }
        } else {
            f64::INFINITY
        },
    ]
    .into_iter()
    .fold(0.0, f64::max)
}

fn validate_true_residual(
    label: &str,
    residual: f64,
    config: &SolverConfig,
) -> Result<(), SimulationError> {
    if !residual.is_finite() || residual > config.tolerance {
        return Err(SimulationError::new(format!(
            "{label} failed true-residual check: residual={residual:.3e}, tolerance={:.3e}",
            config.tolerance
        )));
    }
    Ok(())
}

fn relative_residual_hmatrix(
    hmatrix: &sonium_bem::core::assembly::hmatrix::HMatrix,
    x: &[Complex64],
    rhs: &[Complex64],
) -> f64 {
    let ax = <sonium_bem::core::assembly::hmatrix::HMatrix as LinearOperator<Complex64>>::apply(
        hmatrix, x,
    );
    let residual_norm = ax
        .iter()
        .zip(rhs.iter())
        .map(|(ax_i, rhs_i)| (*rhs_i - *ax_i).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let rhs_norm = rhs.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

    residual_norm / rhs_norm.max(1e-30)
}

/// True residual of the adjoint system `Aᴴ λ = rhs` against the compressed
/// H-matrix operator's Hermitian matvec.
fn relative_residual_hmatrix_adjoint(
    hmatrix: &sonium_bem::core::assembly::hmatrix::HMatrix,
    x: &[Complex64],
    rhs: &[Complex64],
) -> f64 {
    let ax = hmatrix.matvec_hermitian(x);
    relative_residual_from_applied(&ax, rhs)
}

/// Solve the forward system `A p = rhs` on an H-matrix operator.
///
/// Returns the solution together with its residual against the compressed
/// operator so the caller can feed the exact-residual sentinel policy
/// without a second matvec.
pub(crate) fn solve_bem_system_hmatrix(
    hmatrix: &sonium_bem::core::assembly::hmatrix::HMatrix,
    rhs: &[Complex64],
    x0: Option<&[Complex64]>,
    config: &SolverConfig,
    gcro_dr_subspace: Option<&mut GcroDrSubspace<Complex64>>,
) -> Result<(Vec<Complex64>, f64), SimulationError> {
    solve_bem_system_hmatrix_with_control(hmatrix, rhs, x0, config, gcro_dr_subspace, None)
}

fn solve_bem_system_hmatrix_with_control(
    hmatrix: &sonium_bem::core::assembly::hmatrix::HMatrix,
    rhs: &[Complex64],
    x0: Option<&[Complex64]>,
    config: &SolverConfig,
    gcro_dr_subspace: Option<&mut GcroDrSubspace<Complex64>>,
    control: Option<&SimulationControl>,
) -> Result<(Vec<Complex64>, f64), SimulationError> {
    use sonium_bem::core::assembly::hmatrix::HMatrixBlockJacobi;

    if let Some(control) = control {
        control.check()?;
    }

    // Block-Jacobi over the H-matrix diagonal dense blocks (Jacobi
    // fallback for uncovered DOFs); strictly stronger than the old
    // diagonal-only preconditioner whenever a block factors cleanly.
    let precond = HMatrixBlockJacobi::from_hmatrix(hmatrix);
    let inner_tolerance = config.tolerance * 0.1;

    let (x_out, iters, converged, label) = if let Some(sub) = gcro_dr_subspace {
        if let Some(control) = control {
            let cfg = FgmresConfig {
                max_iterations: config.max_iterations,
                restart: config.restart,
                tolerance: inner_tolerance,
                print_interval: 0,
                convergence_check_interval: 1,
                exact_residual_check_interval: 1,
            };
            let r = try_fgmres_preconditioned_with_abort(hmatrix, &precond, rhs, x0, &cfg, || {
                control.is_cancelled()
            })
            .map_err(|error| {
                if control.is_cancelled() {
                    SimulationError::cancelled()
                } else {
                    SimulationError::new(format!("controlled H-matrix FGMRES failed: {error}"))
                }
            })?;
            if control.is_cancelled() {
                return Err(SimulationError::cancelled());
            }
            (
                r.x,
                r.iterations,
                r.converged,
                "FGMRES+Hmatrix (controlled)",
            )
        } else {
            let cfg = GcroDrConfig {
                max_iterations: config.max_iterations,
                restart: config.restart,
                tolerance: inner_tolerance,
                print_interval: 0,
            };
            let r = gcro_dr_preconditioned(hmatrix, &precond, rhs, x0, sub, &cfg);
            (r.x, r.iterations, r.converged, "GCRO-DR+Hmatrix")
        }
    } else {
        let cfg = FgmresConfig {
            max_iterations: config.max_iterations,
            restart: config.restart,
            tolerance: inner_tolerance,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 1,
        };
        if let Some(control) = control {
            let r = try_fgmres_preconditioned_with_abort(hmatrix, &precond, rhs, x0, &cfg, || {
                control.is_cancelled()
            })
            .map_err(|error| {
                if control.is_cancelled() {
                    SimulationError::cancelled()
                } else {
                    SimulationError::new(format!("controlled H-matrix FGMRES failed: {error}"))
                }
            })?;
            if control.is_cancelled() {
                return Err(SimulationError::cancelled());
            }
            (
                r.x,
                r.iterations,
                r.converged,
                "FGMRES+Hmatrix (controlled)",
            )
        } else {
            let r = fgmres_preconditioned(hmatrix, &precond, rhs, x0, &cfg);
            (r.x, r.iterations, r.converged, "FGMRES+Hmatrix")
        }
    };

    if let Some(control) = control {
        control.check()?;
    }

    let residual = relative_residual_hmatrix(hmatrix, &x_out, rhs);
    eprintln!("  {label}: {iters} iters, converged={converged}, true residual={residual:.3e}",);

    if !converged || !complex_values_are_finite(&x_out) {
        return Err(SimulationError::new(format!(
            "{label} failed to converge: reported converged={converged}, true residual={residual:.3e}, tolerance={:.3e}",
            config.tolerance
        )));
    }
    validate_true_residual(label, residual, config)?;

    Ok((x_out, residual))
}

pub(crate) fn build_simulation_hmatrix(
    elements: &[Element],
    nodes: &[[f64; 3]],
    clusters: &[Cluster],
    physics: &PhysicsParams,
    config: &SolverConfig,
) -> HMatrix {
    let n_dofs = elements
        .iter()
        .filter(|element| !element.property.is_evaluation())
        .map(|element| element.dof_addresses.len())
        .sum::<usize>();
    build_hmatrix_stabilized(
        elements,
        nodes,
        clusters,
        physics,
        config.hmatrix_aca_tolerance,
        n_dofs / 4 + 1,
        EXTERIOR_TBEM_SIGN,
        config.beta_scale,
    )
}

pub(crate) fn verify_exact_hmatrix_solution(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    solution: &[Complex64],
    rhs: &[Complex64],
    config: &SolverConfig,
) -> Result<(), SimulationError> {
    let applied = apply_tbem_stabilized_exact(
        elements,
        nodes,
        physics,
        EXTERIOR_TBEM_SIGN,
        config.beta_scale,
        solution,
    )
    .map_err(|message| SimulationError::new(format!("exact TBEM residual failed: {message}")))?;
    let residual = relative_residual_from_applied(&applied, rhs);
    eprintln!("  exact uncompressed TBEM residual={residual:.3e}");
    validate_true_residual("H-matrix exact TBEM operator", residual, config)
}

/// Exact (uncompressed O(N²)) true-residual check for an H-matrix
/// **adjoint** solution: applies the exact `Aᴴ` operator entry-by-entry.
pub(crate) fn verify_exact_hmatrix_adjoint_solution(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    solution: &[Complex64],
    rhs: &[Complex64],
    config: &SolverConfig,
) -> Result<(), SimulationError> {
    let applied = apply_tbem_stabilized_exact_hermitian(
        elements,
        nodes,
        physics,
        EXTERIOR_TBEM_SIGN,
        config.beta_scale,
        solution,
    )
    .map_err(|message| {
        SimulationError::new(format!("exact TBEM adjoint residual failed: {message}"))
    })?;
    let residual = relative_residual_from_applied(&applied, rhs);
    eprintln!("  exact uncompressed TBEM adjoint residual={residual:.3e}");
    validate_true_residual("H-matrix exact TBEM adjoint operator", residual, config)
}

/// Per-sweep counter driving the [`SentinelPolicy`] cadence. One instance
/// per sweep/lane so "every k solves" means every k solves *of that sweep*.
#[derive(Debug, Default)]
pub(crate) struct ExactResidualSentinel {
    solves_seen: usize,
    exact_checks: usize,
}

impl ExactResidualSentinel {
    pub(crate) fn exact_checks(&self) -> usize {
        self.exact_checks
    }

    /// Record one H-matrix solve and decide whether the exact O(N²) check
    /// should run after it. `compressed_residual` is the always-computed
    /// residual against the compressed operator (already gated against the
    /// tolerance by the caller).
    pub(crate) fn should_verify_exact(
        &mut self,
        config: &SolverConfig,
        compressed_residual: f64,
    ) -> bool {
        self.solves_seen += 1;
        // Legacy override: the old boolean flag forces the exact check.
        let should_verify = if config.verify_exact_hmatrix_residual {
            true
        } else {
            match config.exact_residual_sentinel {
                SentinelPolicy::EverySolve => true,
                SentinelPolicy::Off => false,
                SentinelPolicy::Sentinel {
                    every_k,
                    on_marginal_convergence_factor,
                } => {
                    let cadence = every_k > 0 && self.solves_seen.is_multiple_of(every_k);
                    let marginal = compressed_residual.is_finite()
                        && on_marginal_convergence_factor.is_finite()
                        && on_marginal_convergence_factor > 0.0
                        && config.tolerance > 0.0
                        && compressed_residual > config.tolerance / on_marginal_convergence_factor;
                    cadence || marginal
                }
            }
        };
        if should_verify {
            self.exact_checks += 1;
        }
        should_verify
    }
}

struct FrequencySweepState {
    elements: Vec<Element>,
    previous_solution: Option<Vec<Complex64>>,
    previous_frequency: Option<f64>,
    recycling_subspace: Option<GcroDrSubspace<Complex64>>,
    dense_preconditioner: Option<IluPreconditioner<Complex64>>,
    preconditioner_frequency: Option<f64>,
    sentinel: ExactResidualSentinel,
}

impl FrequencySweepState {
    fn new(elements: &[Element], config: &SolverConfig) -> Self {
        Self {
            elements: elements.to_vec(),
            previous_solution: None,
            previous_frequency: None,
            recycling_subspace: (config.recycle_subspace_size > 0)
                .then(|| GcroDrSubspace::new(config.recycle_subspace_size)),
            dense_preconditioner: None,
            preconditioner_frequency: None,
            sentinel: ExactResidualSentinel::default(),
        }
    }
}

struct FrequencySweepOutput {
    spl: Vec<f64>,
    phase: Vec<f64>,
    /// Complex pressures at the evaluation points (input to `spl`/`phase`).
    pressure: Vec<Complex64>,
    woofer_electrical_impedance_ohm: Option<Complex64>,
    tweeter_electrical_impedance_ohm: Option<Complex64>,
    driver_electrical_impedances_ohm: Vec<Option<Complex64>>,
    radiation_matrix: Option<Vec<Vec<Complex64>>>,
    radiation_matrix_diagnostics: Option<RadiationMatrixDiagnostics>,
    thermal_driver_states: [Option<ThermalDriverState>; 2],
    driver_thermal_states: Vec<Option<ThermalDriverState>>,
    power_ledger: Option<PowerLedger>,
    structural_response: Option<StructuralAcousticResponse>,
    driver_velocities_bem: Vec<Complex64>,
    surface_pressure: Vec<Complex64>,
    exact_residual_evaluations: usize,
}

#[allow(clippy::too_many_arguments)]
fn solve_frequency_point(
    frequency_index: usize,
    frequency_count: usize,
    frequency: f64,
    state: &mut FrequencySweepState,
    speaker_mesh: &SpeakerMesh,
    nodes: &[[f64; 3]],
    hmatrix_clusters: Option<&[Cluster]>,
    model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
    config: &SolverConfig,
    use_hmatrix: bool,
    design_point_guess: Option<&[Complex64]>,
    control: Option<&SimulationControl>,
) -> Result<FrequencySweepOutput, SimulationError> {
    if let Some(control) = control {
        control.check()?;
    }
    let frequency_start = Instant::now();
    let exact_residuals_before = state.sentinel.exact_checks();
    let driver_velocities = model.velocities_at(frequency);
    update_driver_bcs(
        &mut state.elements,
        speaker_mesh,
        model,
        frequency,
        Some(&driver_velocities),
    )?;

    let physics = PhysicsParams::new(
        frequency,
        model.simulation.speed_of_sound,
        model.simulation.air_density,
        false,
    );
    let dense_matrix = if use_hmatrix {
        None
    } else {
        Some(
            build_tbem_system_stabilized(
                &state.elements,
                nodes,
                &physics,
                EXTERIOR_TBEM_SIGN,
                config.beta_scale,
            )
            .matrix,
        )
    };
    let hmatrix = if use_hmatrix {
        Some(build_simulation_hmatrix(
            &state.elements,
            nodes,
            hmatrix_clusters.ok_or_else(|| {
                SimulationError::new("H-matrix solver selected without geometry clusters")
            })?,
            &physics,
            config,
        ))
    } else {
        None
    };
    if let Some(matrix) = &dense_matrix
        && matrix.nrows() > config.direct_threshold
        && !should_recycle_initial_guess(
            state.preconditioner_frequency,
            frequency,
            config.max_recycled_guess_octaves,
        )
    {
        state.dense_preconditioner = Some(dense_ilu_preconditioner(matrix, config));
        state.preconditioner_frequency = Some(frequency);
    }

    let recycle_initial_guess = should_recycle_initial_guess(
        state.previous_frequency,
        frequency,
        config.max_recycled_guess_octaves,
    );
    if !recycle_initial_guess && let Some(subspace) = state.recycling_subspace.as_mut() {
        reset_recycling_for_frequency_step(
            state.previous_frequency,
            frequency,
            config.max_recycled_guess_octaves,
            subspace,
        );
    }
    let iteration_guess = if recycle_initial_guess {
        state.previous_solution.clone()
    } else {
        // A design step can change the mesh DOF count; a stale guess of the
        // wrong length would panic inside the iterative solver's matvec.
        let n_dofs = state
            .elements
            .iter()
            .filter(|element| !element.property.is_evaluation())
            .count();
        design_point_guess
            .filter(|guess| guess.len() == n_dofs)
            .map(<[Complex64]>::to_vec)
    };
    let has_typed_driver = model
        .drivers()
        .iter()
        .any(|driver| driver.driver_model.is_some());
    let mut woofer_electrical_impedance_ohm = None;
    let mut tweeter_electrical_impedance_ohm = None;
    let mut driver_electrical_impedances_ohm = vec![None; model.drivers().len()];
    let mut radiation_matrix = None;
    let mut radiation_matrix_diagnostics = None;
    let mut thermal_driver_states = [None; 2];
    let mut driver_thermal_states = vec![None; model.drivers().len()];
    let mut power_ledger = None;
    let mut driver_velocities_bem = driver_velocities.clone();

    let structural_max_iterations = model
        .product_design
        .as_ref()
        .and_then(|product| product.structural_acoustic.as_ref())
        .map_or(1, |feedback| feedback.max_iterations);
    let structural_tolerance = model
        .product_design
        .as_ref()
        .and_then(|product| product.structural_acoustic.as_ref())
        .map(|feedback| feedback.velocity_tolerance);
    let structural_relaxation = model
        .product_design
        .as_ref()
        .and_then(|product| product.structural_acoustic.as_ref())
        .map(|feedback| feedback.under_relaxation);
    let mut structural_velocity: Option<Vec<Complex64>> = None;
    let mut structural_response = None;
    let mut surface_pressure = None;
    for feedback_iteration in 0..structural_max_iterations {
        if let Some(control) = control {
            control.check()?;
        }
        let (pressure, coupling) = if has_typed_driver {
            let (pressure, coupling) = solve_typed_driver_coupling_with_structure_control(
                &mut state.elements,
                speaker_mesh,
                model,
                frequency,
                nodes,
                &physics,
                dense_matrix.as_ref(),
                hmatrix.as_ref(),
                config,
                &mut state.recycling_subspace,
                state.dense_preconditioner.as_ref(),
                iteration_guess.as_deref(),
                &mut state.sentinel,
                structural_velocity.as_deref(),
                control,
            )?;
            (pressure, Some(coupling))
        } else {
            let pressure = solve_bem_with_driver_velocities_and_structure_control(
                &mut state.elements,
                speaker_mesh,
                model,
                frequency,
                &driver_velocities,
                nodes,
                &physics,
                dense_matrix.as_ref(),
                hmatrix.as_ref(),
                config,
                &mut state.recycling_subspace,
                state.dense_preconditioner.as_ref(),
                iteration_guess.as_deref(),
                &mut state.sentinel,
                structural_velocity.as_deref(),
                control,
            )?;
            (pressure, None)
        };
        surface_pressure = Some(pressure.clone());
        if let Some(coupling) = coupling {
            woofer_electrical_impedance_ohm = coupling.woofer_electrical_impedance_ohm;
            tweeter_electrical_impedance_ohm = coupling.tweeter_electrical_impedance_ohm;
            driver_electrical_impedances_ohm = coupling.driver_electrical_impedances_ohm;
            radiation_matrix = Some(coupling.radiation_matrix_all.clone());
            radiation_matrix_diagnostics = Some(coupling.radiation_matrix_diagnostics);
            thermal_driver_states = coupling.thermal_driver_states;
            driver_thermal_states = coupling.driver_thermal_states;
            power_ledger = coupling.power_ledger;
            driver_velocities_bem = coupling.velocities_bem_all.clone();
        }
        let feedback = structural_cabinet_velocity_from_pressure(
            model,
            frequency,
            &state.elements,
            speaker_mesh,
            &pressure,
            structural_velocity.as_deref(),
        )?;
        let Some((next_velocity, response, change)) = feedback else {
            break;
        };
        structural_response = Some(response);
        let relaxation = structural_relaxation.unwrap_or(1.0);
        structural_velocity = Some(if let Some(previous) = structural_velocity.as_ref() {
            previous
                .iter()
                .zip(&next_velocity)
                .map(|(old, new)| *old + relaxation * (*new - *old))
                .collect()
        } else {
            next_velocity
        });
        if change < structural_tolerance.unwrap_or(f64::INFINITY) {
            break;
        }
        if feedback_iteration + 1 == structural_max_iterations {
            return Err(SimulationError::new(format!(
                "structural acoustic feedback did not converge at {frequency:.3} Hz after {} iterations (relative velocity change {change:.3e})",
                structural_max_iterations
            )));
        }
    }
    let surface_pressure = surface_pressure.ok_or_else(|| {
        SimulationError::new("structural/BEM frequency solve produced no pressure")
    })?;

    let surface_velocity = build_surface_velocity(&state.elements);
    let scattered = compute_scattered_field(
        eval_points,
        &state.elements,
        nodes,
        &surface_pressure,
        Some(&surface_velocity),
        &physics,
        BasisOrder::Constant,
    );
    if !complex_values_are_finite(&scattered) {
        return Err(SimulationError::new(format!(
            "field evaluation produced non-finite pressure at {frequency:.3} Hz"
        )));
    }
    let mut spl = Vec::with_capacity(eval_points.len());
    let mut phase = Vec::with_capacity(eval_points.len());
    for (point_index, &pressure) in scattered.iter().enumerate() {
        spl.push(sonium_xem::pressure_to_spl(pressure));
        phase.push(pressure.arg());
        if !spl[point_index].is_finite() || !phase[point_index].is_finite() {
            return Err(SimulationError::new(format!(
                "field evaluation produced non-finite SPL/phase at {frequency:.3} Hz, point {point_index}"
            )));
        }
    }

    state.previous_solution = Some(surface_pressure.clone());
    state.previous_frequency = Some(frequency);
    eprintln!(
        "[{}/{}] f={:.0} Hz, {:.2}s",
        frequency_index + 1,
        frequency_count,
        frequency,
        frequency_start.elapsed().as_secs_f64(),
    );
    Ok(FrequencySweepOutput {
        spl,
        phase,
        pressure: scattered,
        woofer_electrical_impedance_ohm,
        tweeter_electrical_impedance_ohm,
        driver_electrical_impedances_ohm,
        radiation_matrix,
        radiation_matrix_diagnostics,
        thermal_driver_states,
        driver_thermal_states,
        power_ledger,
        structural_response,
        driver_velocities_bem,
        surface_pressure,
        exact_residual_evaluations: state.sentinel.exact_checks() - exact_residuals_before,
    })
}

// ---------------------------------------------------------------------------
// Core simulation
// ---------------------------------------------------------------------------

/// Run the full BEM simulation pipeline.
///
/// For each frequency:
/// 1. Update driver element BCs with frequency-dependent piston velocity
/// 2. Assemble BEM system (matrix + RHS from velocity BCs)
/// 3. Solve with GMRES and solution recycling
/// 4. Evaluate scattered field at the given evaluation points
///
/// Returns SPL and phase at each eval point for each frequency.
pub fn run_simulation(
    model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
) -> Result<SimulationResult, SimulationError> {
    run_simulation_with_config(model, eval_points, &SolverConfig::default())
}

/// Run the BEM simulation with custom solver configuration.
pub fn run_simulation_with_config(
    model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
    solver_config: &SolverConfig,
) -> Result<SimulationResult, SimulationError> {
    run_simulation_with_control(model, eval_points, solver_config, None)
}

/// Run the BEM simulation with caller-owned progress and cancellation state.
///
/// This is intentionally an additive API. Existing callers keep the exact
/// `run_simulation_with_config` behavior, while long-running hosts can create
/// a [`SimulationControl`], poll its atomics from another thread, and call
/// [`SimulationControl::cancel`] without passing Python callbacks into native
/// solver workers.
pub fn run_simulation_with_control(
    model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
    solver_config: &SolverConfig,
    control: Option<&SimulationControl>,
) -> Result<SimulationResult, SimulationError> {
    try_run_simulation_with_config_inner(model, eval_points, solver_config, None, None, control)
}

/// Run the supported dense speaker BEM path on CUDA.
///
/// Regular-pair assembly, sparse GMRES, and exterior field evaluation use the
/// CUDA backend.  The result is accepted only after promoted f64 residual and
/// field-observable gates.  Supported linear typed-driver coupling is solved
/// against the same resident CUDA operator; thermal/nonlinear and
/// H-matrix-sized meshes remain explicitly CPU-only.
#[cfg(feature = "cuda")]
pub fn run_simulation_cuda(
    model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
    solver_config: &SolverConfig,
) -> Result<SimulationResult, SimulationError> {
    run_simulation_cuda_with_report(model, eval_points, solver_config).map(|(result, _)| result)
}

/// Run the CUDA speaker job and return machine-readable residency, transfer,
/// memory, timing, capability, and acceptance-gate evidence.
#[cfg(feature = "cuda")]
pub fn run_simulation_cuda_with_report(
    model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
    solver_config: &SolverConfig,
) -> Result<(SimulationResult, GpuExecutionReport), SimulationError> {
    run_simulation_cuda_on_device_with_control(model, eval_points, solver_config, 0, None)
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn validate_accelerator_speaker_configuration(
    backend_kind: SimulationBackend,
    model: &LoudspeakerModel,
) -> Result<(), SimulationError> {
    if model.product_design.is_some() {
        return Err(SimulationError::new(format!(
            "{backend_kind:?} speaker forward does not support ProductDesign configurations (crossover phase, coordinate rotation, and body-fitted projected reanalysis remain unsupported); use the CPU product path"
        )));
    }
    Ok(())
}

/// Run the CUDA speaker job on an explicitly selected device.
///
/// Device selection is kept inside the native heavy-job entry point so
/// language bindings do not need a Python loop over devices or frequencies.
#[cfg(feature = "cuda")]
pub fn run_simulation_cuda_on_device(
    model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
    solver_config: &SolverConfig,
    cuda_device: usize,
) -> Result<SimulationResult, SimulationError> {
    run_simulation_cuda_on_device_with_control(model, eval_points, solver_config, cuda_device, None)
        .map(|(result, _)| result)
}

/// Controlled CUDA entry point for bindings and bounded QA workloads.
#[cfg(feature = "cuda")]
pub fn run_simulation_cuda_on_device_with_control(
    model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
    solver_config: &SolverConfig,
    cuda_device: usize,
    control: Option<&SimulationControl>,
) -> Result<(SimulationResult, GpuExecutionReport), SimulationError> {
    run_simulation_cuda_on_device_with_control_and_failure(
        model,
        eval_points,
        solver_config,
        cuda_device,
        control,
        None,
    )
}

/// Run the CUDA speaker entry point with one explicitly injected backend
/// failure. This hidden surface exists for QA and driver-shim acceptance
/// tests; normal callers should use [`run_simulation_cuda_on_device`] or
/// [`run_simulation_cuda_on_device_with_control`].
#[doc(hidden)]
#[cfg(feature = "cuda")]
pub fn run_simulation_cuda_on_device_with_failure(
    model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
    solver_config: &SolverConfig,
    cuda_device: usize,
    failure: sonium_compute::CudaFailureInjectionPoint,
) -> Result<(SimulationResult, GpuExecutionReport), SimulationError> {
    run_simulation_cuda_on_device_with_control_and_failure(
        model,
        eval_points,
        solver_config,
        cuda_device,
        None,
        Some(failure),
    )
}

#[cfg(feature = "cuda")]
fn run_simulation_cuda_on_device_with_control_and_failure(
    model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
    solver_config: &SolverConfig,
    cuda_device: usize,
    control: Option<&SimulationControl>,
    failure: Option<sonium_compute::CudaFailureInjectionPoint>,
) -> Result<(SimulationResult, GpuExecutionReport), SimulationError> {
    validate_accelerator_speaker_configuration(SimulationBackend::Cuda, model)?;
    let backend = sonium_compute::CudaBackend::new(cuda_device)
        .map_err(|error| SimulationError::new(format!("CUDA backend unavailable: {error:?}")))?;
    let allocation_ledger = backend.allocation_ledger();
    if let Some(failure) = failure {
        backend.inject_failure(failure);
    }
    let mut assembly_workspace = None;
    let mut quadrature_orders = Vec::new();
    let workspace_builds = Arc::new(AtomicUsize::new(0));
    let build_counter = Arc::clone(&workspace_builds);
    let schedule_updates = Arc::new(AtomicUsize::new(0));
    let schedule_counter = Arc::clone(&schedule_updates);
    let outcome = run_simulation_gpu_with_backend(
        SimulationBackend::Cuda,
        backend,
        model,
        eval_points,
        solver_config,
        control,
        Some(cuda_device),
        move |elements, nodes, physics, beta, backend| {
            let orders: Vec<_> = elements
                .iter()
                .filter(|e| !e.property.is_evaluation())
                .map(|element| {
                    let q = sonium_bem::core::integration::singular::QuadratureParams::for_ka(
                        physics.wave_number * element.area.sqrt().max(1e-10),
                    );
                    (
                        q.edge_gauss_order,
                        q.subelement_gauss_order,
                        q.edge_sections,
                        q.subtriangles_per_section,
                    )
                })
                .collect();
            if assembly_workspace.is_none() || orders != quadrature_orders {
                if assembly_workspace.is_some() {
                    schedule_counter.fetch_add(1, Ordering::Relaxed);
                }
                // Match the CPU's frequency-dependent singular schedule.
                // Reuse geometry while its quadrature orders remain valid.
                assembly_workspace = Some(
                    CudaTbemAssemblyWorkspace::new(elements, nodes, physics, backend).map_err(
                        |error| {
                            SimulationError::new(format!(
                                "CUDA resident TBEM workspace construction failed: {error}"
                            ))
                        },
                    )?,
                );
                quadrature_orders = orders;
                build_counter.fetch_add(1, Ordering::Relaxed);
            }
            let workspace = assembly_workspace.as_mut().ok_or_else(|| {
                SimulationError::new("CUDA assembly workspace was not initialized")
            })?;
            let value = workspace
                .assemble(elements, nodes, physics, beta, backend)
                .map_err(|error| {
                    SimulationError::new(format!("CUDA resident TBEM assembly failed: {error}"))
                })?;
            let derivative = if solver_config.gpu_wavenumber_derivative {
                Some(
                    workspace
                        .assemble_wavenumber_derivative(
                            elements,
                            nodes,
                            physics,
                            beta,
                            physics.burton_miller_beta_wavenumber_derivative()
                                * solver_config.beta_scale,
                            physics.pressure_factor_wavenumber_derivative(),
                            EXTERIOR_TBEM_SIGN,
                            backend,
                        )
                        .map_err(|error| {
                            SimulationError::new(format!(
                                "CUDA resident TBEM derivative assembly failed: {error}"
                            ))
                        })?,
                )
            } else {
                None
            };
            Ok(GpuAssembledSystem {
                value,
                wavenumber_derivative: derivative,
            })
        },
    );
    let post_teardown = allocation_ledger.snapshot();
    match outcome {
        Ok((result, mut report)) => {
            report.cache.assembly_workspace_builds = workspace_builds.load(Ordering::Relaxed);
            report.cache.assembly_quadrature_schedule_updates =
                schedule_updates.load(Ordering::Relaxed);
            report.cache.assembly_geometry_reused = report.cache.assembly_workspace_builds == 1;
            report.memory.post_teardown_live_allocation_bytes =
                Some(post_teardown.live_allocation_bytes);
            report.memory.post_teardown_live_staging_bytes = Some(post_teardown.live_staging_bytes);
            Ok((result, report))
        }
        Err(error) => Err(error),
    }
}

/// Run the bounded product-level CUDA adjoint for a prescribed-driver field
/// objective and return one surface adjoint vector per frequency.
///
/// The objective is the linear field functional
/// `J = <field(eval_points), objective_weights>`.  This entry point is
/// intentionally fail-closed for typed-driver, crossover, and structural
/// product configurations: those paths require a coupled product adjoint and
/// are not silently reduced to the prescribed BEM operator.
#[cfg(feature = "cuda")]
pub fn run_simulation_cuda_adjoint(
    model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
    objective_weights: &[Complex64],
    solver_config: &SolverConfig,
) -> Result<Vec<Vec<Complex64>>, SimulationError> {
    run_simulation_cuda_adjoint_with_report(model, eval_points, objective_weights, solver_config, 0)
        .map(|(adjoints, _)| adjoints)
}

/// Run the bounded prescribed-driver CUDA adjoint and return acceptance and
/// residency evidence alongside the vectors.
#[cfg(feature = "cuda")]
pub fn run_simulation_cuda_adjoint_with_report(
    model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
    objective_weights: &[Complex64],
    solver_config: &SolverConfig,
    cuda_device: usize,
) -> Result<(Vec<Vec<Complex64>>, GpuAdjointReport), SimulationError> {
    model
        .validate()
        .map_err(|message| SimulationError::new(format!("invalid loudspeaker model: {message}")))?;
    validate_evaluation_points(eval_points)?;
    validate_solver_config(solver_config)?;
    if objective_weights.len() != eval_points.len() {
        return Err(SimulationError::new(format!(
            "CUDA adjoint objective has {} weights for {} evaluation points",
            objective_weights.len(),
            eval_points.len()
        )));
    }
    if !complex_values_are_finite(objective_weights)
        || objective_weights
            .iter()
            .all(|weight| weight.norm_sqr() == 0.0)
    {
        return Err(SimulationError::new(
            "CUDA adjoint objective weights must be finite and non-zero",
        ));
    }
    if model
        .drivers()
        .iter()
        .any(|driver| driver.driver_model.is_some())
    {
        return Err(SimulationError::new(
            "CUDA prescribed adjoint does not support typed DriverModel ports; coupled product adjoint remains unsupported",
        ));
    }
    if model.product_design.is_some() {
        return Err(SimulationError::new(
            "CUDA prescribed adjoint does not support crossover product configurations; use the CPU product adjoint",
        ));
    }

    let backend = sonium_compute::CudaBackend::new(cuda_device)
        .map_err(|error| SimulationError::new(format!("CUDA backend unavailable: {error:?}")))?;
    let capabilities = sonium_compute::Backend::capabilities(&backend);
    capabilities
        .require_native_bem_regular_pair_c64()
        .map_err(|error| SimulationError::new(format!("CUDA adjoint unavailable: {error}")))?;
    let adjoint_capability = gpu_coupled_capability_state(&backend);
    if !matches!(
        adjoint_capability,
        GpuCapabilityState::Native | GpuCapabilityState::VendorNative
    ) {
        return Err(SimulationError::new(format!(
            "CUDA adjoint requires native Complex64 CSR/BLAS/reduction support; capability={adjoint_capability:?}"
        )));
    }

    backend.reset_transfer_stats();
    let frequencies = model.simulation.frequencies();
    let speaker_mesh = super::mesh_gen::generate_loudspeaker_mesh(model);
    let (nodes, mut elements) = speaker_mesh_to_bem(&speaker_mesh, model)?;
    let n_dofs = elements
        .iter()
        .filter(|element| !element.property.is_evaluation())
        .count();
    if n_dofs > solver_config.hmatrix_threshold {
        return Err(SimulationError::new(format!(
            "CUDA adjoint dense path is limited to {} DOFs; use the CPU H-matrix adjoint for {} DOFs",
            solver_config.hmatrix_threshold, n_dofs
        )));
    }
    // Keep adjoint Krylov storage across numeric operator updates.
    let krylov_pool = sonium_compute::MemoryPool::new(
        backend,
        sonium_compute::MemoryPoolConfig::default(),
    )
    .map_err(|error| SimulationError::new(format!("CUDA adjoint Krylov pool failed: {error}")))?;
    let backend = krylov_pool.backend();
    let mut workspace = sonium_compute::GmresWorkspace::new(n_dofs);
    let (row_ptr, col_idx) = dense_csr_structure(n_dofs)?;
    let mut assembly_workspace = CudaTbemAssemblyWorkspace::new(
        &elements,
        &nodes,
        &PhysicsParams::new(
            frequencies[0],
            model.simulation.speed_of_sound,
            model.simulation.air_density,
            false,
        ),
        backend,
    )
    .map_err(|error| {
        SimulationError::new(format!(
            "CUDA adjoint resident TBEM workspace construction failed: {error}"
        ))
    })?;
    let geometry_key = assembly_workspace.geometry_key();
    let gpu_block_size = if n_dofs >= GPU_BLOCK_JACOBI_MIN_DOFS {
        sonium_compute::DEFAULT_BLOCK_SIZE
    } else {
        1
    };
    let mut matrix: Option<sonium_compute::PersistentDeviceCsr<_, Complex64>> = None;
    let mut preconditioner: Option<sonium_compute::BlockJacobiBackend<_, Complex64>> = None;
    let mut rhs_device = backend.raw_alloc::<Complex64>(n_dofs).map_err(|error| {
        SimulationError::new(format!("CUDA adjoint RHS allocation failed: {error}"))
    })?;
    let mut solution_device = backend.raw_alloc::<Complex64>(n_dofs).map_err(|error| {
        SimulationError::new(format!("CUDA adjoint solution allocation failed: {error}"))
    })?;

    let total_start = Instant::now();
    let mut assembly_ms = 0;
    let mut solve_ms = 0;
    let mut residual_gate_ms = 0;
    let mut field_ms = 0;
    let mut reference_ms = 0;
    let mut matrix_value_updates = 0;
    let mut preconditioner_updates = 0;
    let mut max_relative_residual = 0.0_f64;
    let mut max_cpu_reference_error = 0.0_f64;
    let mut max_inner_product_error = 0.0_f64;
    let mut adjoints = Vec::with_capacity(frequencies.len());

    for &frequency in &frequencies {
        let velocities = model.velocities_at(frequency);
        update_driver_bcs(
            &mut elements,
            &speaker_mesh,
            model,
            frequency,
            Some(&velocities),
        )?;
        let physics = PhysicsParams::new(
            frequency,
            model.simulation.speed_of_sound,
            model.simulation.air_density,
            false,
        );
        let beta = physics.burton_miller_beta_scaled(solver_config.beta_scale);
        let assembly_start = Instant::now();
        let system = assembly_workspace
            .assemble(&elements, &nodes, &physics, beta, backend)
            .map_err(|error| {
                SimulationError::new(format!(
                    "CUDA adjoint resident TBEM assembly failed at {frequency:.1} Hz: {error}"
                ))
            })?;
        assembly_ms += assembly_start.elapsed().as_millis() as u64;
        if system.matrix.nrows() != n_dofs || system.matrix.ncols() != n_dofs {
            return Err(SimulationError::new(
                "CUDA adjoint assembly returned an inconsistent matrix dimension",
            ));
        }

        let field_start = Instant::now();
        let surface_velocity = build_surface_velocity(&elements);
        let (field_operator, _) = build_constant_field_operator(
            eval_points,
            &elements,
            &nodes,
            Some(&surface_velocity),
            &physics,
        );
        field_ms += field_start.elapsed().as_millis() as u64;
        if field_operator.ncols() != n_dofs {
            return Err(SimulationError::new(format!(
                "CUDA adjoint field operator has {} columns for {} active BEM DOFs",
                field_operator.ncols(),
                n_dofs
            )));
        }
        let mut adjoint_rhs = vec![Complex64::new(0.0, 0.0); n_dofs];
        field_operator.matvec_hermitian_into(objective_weights, &mut adjoint_rhs);
        if !complex_values_are_finite(&adjoint_rhs) {
            return Err(SimulationError::new(format!(
                "CUDA adjoint field RHS is non-finite at {frequency:.1} Hz"
            )));
        }

        let system_matrix = &system.matrix;
        let adjoint_values: Vec<Complex64> = (0..n_dofs)
            .flat_map(|row| (0..n_dofs).map(move |column| system_matrix[[column, row]].conj()))
            .collect();
        if let Some(matrix) = matrix.as_mut() {
            matrix
                .try_copy_values_from_host(backend, &adjoint_values)
                .map_err(|error| {
                    SimulationError::new(format!(
                        "CUDA adjoint CSR value update failed at {frequency:.1} Hz: {error}"
                    ))
                })?;
            matrix_value_updates += 1;
        } else {
            let device_matrix = sonium_compute::DeviceCsr::try_from_host(
                backend,
                n_dofs,
                n_dofs,
                &row_ptr,
                &col_idx,
                &adjoint_values,
            )
            .map_err(|error| {
                SimulationError::new(format!(
                    "CUDA adjoint CSR allocation failed at {frequency:.1} Hz: {error}"
                ))
            })?;
            matrix = Some(
                sonium_compute::PersistentDeviceCsr::try_from_device_csr(backend, device_matrix)
                    .map_err(|error| {
                    SimulationError::new(format!(
                        "CUDA adjoint CSR descriptor setup failed: {error}"
                    ))
                })?,
            );
        }
        if let Some(preconditioner) = preconditioner.as_mut() {
            preconditioner
                .try_update_dense_host(backend, &adjoint_values)
                .map_err(|error| {
                    SimulationError::new(format!(
                        "CUDA adjoint block-Jacobi update failed at {frequency:.1} Hz: {error}"
                    ))
                })?;
            preconditioner_updates += 1;
        } else {
            preconditioner = Some(
                sonium_compute::BlockJacobiBackend::try_from_dense_host(
                    backend,
                    &adjoint_values,
                    n_dofs,
                    gpu_block_size,
                )
                .map_err(|error| {
                    SimulationError::new(format!(
                        "CUDA adjoint block-Jacobi allocation failed: {error}"
                    ))
                })?,
            );
        }

        let solve_start = Instant::now();
        backend
            .try_copy_h2d(&adjoint_rhs, &mut rhs_device)
            .map_err(|error| {
                SimulationError::new(format!(
                    "CUDA adjoint RHS upload failed at {frequency:.1} Hz: {error}"
                ))
            })?;
        let matrix = matrix
            .as_ref()
            .ok_or_else(|| SimulationError::new("CUDA adjoint matrix was not initialized"))?;
        let preconditioner = preconditioner.as_ref().ok_or_else(|| {
            SimulationError::new("CUDA adjoint preconditioner was not initialized")
        })?;
        let status = sonium_compute::gmres::gmres_backend_preconditioned_pooled_with_workspace(
            &krylov_pool,
            &mut workspace,
            matrix,
            preconditioner,
            &rhs_device,
            None,
            &mut solution_device,
            &sonium_compute::GmresBackendConfig::<Complex64> {
                max_iterations: solver_config.max_iterations,
                tolerance: (solver_config.tolerance * 0.1).max(1.0e-12),
            },
        )
        .map_err(|error| {
            SimulationError::new(format!(
                "CUDA adjoint GMRES failed at {frequency:.1} Hz: {error}"
            ))
        })?;
        solve_ms += solve_start.elapsed().as_millis() as u64;
        if !status.converged {
            return Err(SimulationError::new(format!(
                "CUDA adjoint GMRES did not converge at {frequency:.1} Hz: residual={:.3e}",
                status.final_residual
            )));
        }
        let mut gpu_adjoint = vec![Complex64::new(0.0, 0.0); n_dofs];
        backend
            .try_copy_d2h(&solution_device, &mut gpu_adjoint)
            .map_err(|error| {
                SimulationError::new(format!(
                    "CUDA adjoint solution download failed at {frequency:.1} Hz: {error}"
                ))
            })?;
        if !complex_values_are_finite(&gpu_adjoint) {
            return Err(SimulationError::new(format!(
                "CUDA adjoint solution is non-finite at {frequency:.1} Hz"
            )));
        }

        let residual_start = Instant::now();
        let relative_residual =
            relative_residual_adjoint(&system.matrix, &gpu_adjoint, &adjoint_rhs);
        residual_gate_ms += residual_start.elapsed().as_millis() as u64;
        if !relative_residual.is_finite() || relative_residual > solver_config.tolerance {
            return Err(SimulationError::new(format!(
                "CUDA adjoint CPU f64 residual gate failed at {frequency:.1} Hz: residual={relative_residual:.3e}, tolerance={:.3e}",
                solver_config.tolerance
            )));
        }

        let reference_start = Instant::now();
        let cpu_reference =
            lu_solve_conj_transpose(&system.matrix, &adjoint_rhs).map_err(|error| {
                SimulationError::new(format!(
                    "CUDA adjoint CPU f64 reference solve failed at {frequency:.1} Hz: {error}"
                ))
            })?;
        let cpu_reference_error = gpu_adjoint
            .iter()
            .zip(&cpu_reference)
            .map(|(actual, reference)| relative_complex_difference(*actual, *reference))
            .fold(0.0, f64::max);
        let probe: Vec<Complex64> = (0..n_dofs)
            .map(|index| Complex64::new(1.0 + index as f64 / n_dofs.max(1) as f64, -0.25))
            .collect();
        let field_probe = field_operator.matvec(&probe);
        let lhs = complex_inner_product(&field_probe, objective_weights);
        let rhs = complex_inner_product(&probe, &adjoint_rhs);
        let inner_product_error = (lhs - rhs).norm() / lhs.norm().max(rhs.norm()).max(1.0e-30);
        reference_ms += reference_start.elapsed().as_millis() as u64;
        if !cpu_reference_error.is_finite() || cpu_reference_error > CUDA_ADJOINT_PARITY_TOLERANCE {
            return Err(SimulationError::new(format!(
                "CUDA adjoint CPU reference parity failed at {frequency:.1} Hz: error={cpu_reference_error:.3e}, tolerance={CUDA_ADJOINT_PARITY_TOLERANCE:.3e}"
            )));
        }
        if !inner_product_error.is_finite()
            || inner_product_error > CUDA_ADJOINT_INNER_PRODUCT_TOLERANCE
        {
            return Err(SimulationError::new(format!(
                "CUDA adjoint field inner-product identity failed at {frequency:.1} Hz: error={inner_product_error:.3e}, tolerance={CUDA_ADJOINT_INNER_PRODUCT_TOLERANCE:.3e}"
            )));
        }
        max_relative_residual = max_relative_residual.max(relative_residual);
        max_cpu_reference_error = max_cpu_reference_error.max(cpu_reference_error);
        max_inner_product_error = max_inner_product_error.max(inner_product_error);
        adjoints.push(gpu_adjoint);
    }

    let transfer = backend.transfer_stats();
    let memory = backend.memory_stats();
    let mut capability_matrix = GpuCapabilityMatrix::from_backend(backend);
    capability_matrix.adjoint_solve = adjoint_capability;
    let report = GpuAdjointReport {
        schema_version: GpuAdjointReport::SCHEMA_VERSION,
        requested_backend: SimulationBackend::Cuda,
        actual_backend: SimulationBackend::Cuda,
        device_index: cuda_device,
        capability_matrix,
        frequencies: frequencies.len(),
        objective_points: eval_points.len(),
        active_dofs: n_dofs,
        geometry_key,
        operator_structure_resident: true,
        rhs_buffer_reused: frequencies.len() > 1,
        solution_buffer_reused: frequencies.len() > 1,
        matrix_value_updates,
        preconditioner_updates,
        residual_gate: ResidualGateStatus::Passed,
        max_relative_residual,
        max_cpu_reference_error,
        max_inner_product_error,
        timings: GpuTimingReport {
            assembly_ms,
            solve_ms,
            residual_gate_ms,
            field_ms,
            reference_ms,
            total_ms: total_start.elapsed().as_millis() as u64,
        },
        transfers: GpuTransferReport {
            h2d_operations: transfer.h2d_operations,
            h2d_bytes: transfer.h2d_bytes,
            d2h_operations: transfer.d2h_operations,
            d2h_bytes: transfer.d2h_bytes,
            d2d_operations: transfer.d2d_operations,
            d2d_bytes: transfer.d2d_bytes,
        },
        memory: GpuMemoryReport {
            total_device_bytes: backend.total_device_memory_bytes(),
            live_allocation_bytes: memory.live_allocation_bytes,
            allocation_high_water_bytes: memory.allocation_high_water_bytes,
            live_staging_bytes: memory.live_staging_bytes,
            staging_high_water_bytes: memory.staging_high_water_bytes,
            post_teardown_live_allocation_bytes: None,
            post_teardown_live_staging_bytes: None,
        },
    };
    Ok((adjoints, report))
}

/// Run the supported dense speaker BEM path on Apple Metal.
#[cfg(feature = "metal")]
pub fn run_simulation_metal(
    model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
    solver_config: &SolverConfig,
) -> Result<SimulationResult, SimulationError> {
    run_simulation_metal_with_report(model, eval_points, solver_config).map(|(result, _)| result)
}

/// Run the Metal speaker job and return the same explicit evidence contract
/// as the CUDA path.
#[cfg(feature = "metal")]
pub fn run_simulation_metal_with_report(
    model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
    solver_config: &SolverConfig,
) -> Result<(SimulationResult, GpuExecutionReport), SimulationError> {
    validate_accelerator_speaker_configuration(SimulationBackend::Metal, model)?;
    let backend = sonium_compute::MetalBackend::new()
        .map_err(|error| SimulationError::new(format!("Metal backend unavailable: {error:?}")))?;
    run_simulation_gpu_with_backend(
        SimulationBackend::Metal,
        backend,
        model,
        eval_points,
        solver_config,
        None,
        None,
        |elements, nodes, physics, beta, backend| {
            Ok(GpuAssembledSystem {
                value:
                    sonium_bem::core::assembly::tbem::build_tbem_system_stabilized_with_beta_metal(
                        elements,
                        nodes,
                        physics,
                        EXTERIOR_TBEM_SIGN,
                        beta,
                        backend,
                    ),
                wavenumber_derivative: None,
            })
        },
    )
}

#[cfg(any(feature = "cuda", feature = "metal"))]
struct GpuAssembledSystem {
    value: sonium_bem::core::assembly::tbem::TbemSystem,
    wavenumber_derivative: Option<sonium_bem::core::assembly::tbem::TbemSystem>,
}

#[cfg(any(feature = "cuda", feature = "metal"))]
#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
fn run_simulation_gpu_with_backend<B, F>(
    backend_kind: SimulationBackend,
    backend: B,
    model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
    solver_config: &SolverConfig,
    control: Option<&SimulationControl>,
    device_index: Option<usize>,
    mut assemble: F,
) -> Result<(SimulationResult, GpuExecutionReport), SimulationError>
where
    B: sonium_compute::Backend,
    F: FnMut(
        &[Element],
        &[[f64; 3]],
        &PhysicsParams,
        Complex64,
        &B,
    ) -> Result<GpuAssembledSystem, SimulationError>,
{
    if solver_config.gpu_wavenumber_derivative && backend_kind == SimulationBackend::Metal {
        return Err(SimulationError::new(
            "Metal forward wavenumber derivative assembly is unavailable",
        ));
    }
    let capabilities = sonium_compute::Backend::capabilities(&backend);
    capabilities
        .require_native_bem_regular_pair()
        .map_err(|error| {
            SimulationError::new(format!("GPU speaker operation unavailable: {error}"))
        })?;
    if backend_kind == SimulationBackend::Cuda {
        capabilities
            .require_native_bem_regular_pair_c64()
            .map_err(|error| {
                SimulationError::new(format!("CUDA speaker operation unavailable: {error}"))
            })?;
    }
    if let Some(control) = control {
        control.check()?;
    }
    backend.reset_transfer_stats();
    let total_start = Instant::now();
    let mut capability_matrix = GpuCapabilityMatrix::from_backend(&backend);
    if !matches!(
        capability_matrix
            .speaker_configurations
            .prescribed_forward
            .state,
        GpuCapabilityState::Native | GpuCapabilityState::VendorNative
    ) {
        return Err(SimulationError::new(format!(
            "GPU speaker operation unavailable: prescribed forward capability={:?}",
            capability_matrix
                .speaker_configurations
                .prescribed_forward
                .state
        )));
    }
    if backend_kind == SimulationBackend::Cuda
        && solver_config.gpu_wavenumber_derivative
        && !matches!(
            backend.bem_wavenumber_derivative_capability(),
            sonium_compute::CapabilityState::Native | sonium_compute::CapabilityState::VendorNative
        )
    {
        return Err(SimulationError::new(
            "CUDA speaker operation unavailable: native wavenumber-derivative assembly is required",
        ));
    }
    capability_matrix.derivative_assembly = match backend.bem_wavenumber_derivative_capability() {
        sonium_compute::CapabilityState::Native => GpuCapabilityState::Native,
        sonium_compute::CapabilityState::VendorNative => GpuCapabilityState::VendorNative,
        sonium_compute::CapabilityState::HostSharedFallback => {
            GpuCapabilityState::HostSharedFallback
        }
        sonium_compute::CapabilityState::HostStagedFallback => {
            GpuCapabilityState::HostStagedFallback
        }
        sonium_compute::CapabilityState::Unsupported => GpuCapabilityState::Unsupported,
    };
    model
        .validate()
        .map_err(|message| SimulationError::new(format!("invalid loudspeaker model: {message}")))?;
    validate_accelerator_speaker_configuration(backend_kind, model)?;
    validate_evaluation_points(eval_points)?;
    validate_solver_config(solver_config)?;
    let typed_driver_count = model
        .drivers()
        .iter()
        .filter(|driver| driver.driver_model.is_some())
        .count();
    let has_typed_driver = typed_driver_count > 0;
    if has_typed_driver {
        if backend_kind != SimulationBackend::Cuda {
            return Err(SimulationError::new(
                "Metal speaker execution does not support typed driver coupling; use the CPU backend",
            ));
        }
        if typed_driver_count != model.drivers().len() {
            return Err(SimulationError::new(
                "CUDA typed coupling requires every driver port to be typed; mixed prescribed and typed ports remain CPU-only",
            ));
        }
        if model.drivers().iter().any(|driver| {
            driver.driver_model.as_ref().is_some_and(|driver_model| {
                driver_model.thermal_compression_spec.is_some()
                    || driver_model.thermal_network.is_some()
                    || driver_model.nonlinear_motor.is_some()
                    || driver_model.nonlinear_operating_domain_evidence.is_some()
            })
        }) {
            return Err(SimulationError::new(
                "CUDA typed coupling supports only validated linear DriverModel ports; thermal and nonlinear material cells are unsupported",
            ));
        }
        if model
            .product_design
            .as_ref()
            .and_then(|product| product.structural_acoustic.as_ref())
            .is_some()
        {
            return Err(SimulationError::new(
                "CUDA typed coupling does not yet support structural-acoustic cabinet feedback",
            ));
        }
        let coupled_capability = gpu_coupled_capability_state(&backend);
        if !matches!(
            coupled_capability,
            GpuCapabilityState::Native | GpuCapabilityState::VendorNative
        ) {
            return Err(SimulationError::new(format!(
                "CUDA typed coupling requires native Complex64 CSR/BLAS/reduction support; capability={coupled_capability:?}"
            )));
        }
        capability_matrix.coupled_driver_solve = coupled_capability;
        capability_matrix.speaker_configurations.typed_multi_driver = GpuCapabilityCell::new(
            coupled_capability,
            "validated linear typed multi-driver CUDA N-port path; CPU f64 parity and physical gates are required before promotion",
        );
    }

    let speaker_mesh = super::mesh_gen::generate_loudspeaker_mesh(model);
    let (nodes, mut elements) = speaker_mesh_to_bem(&speaker_mesh, model)?;
    let n_dofs = elements
        .iter()
        .filter(|element| !element.property.is_evaluation())
        .count();
    if n_dofs > solver_config.hmatrix_threshold {
        return Err(SimulationError::new(format!(
            "GPU speaker BEM dense path is limited to {} DOFs; use the CPU H-matrix path for {} DOFs",
            solver_config.hmatrix_threshold, n_dofs
        )));
    }

    // Retain Krylov scratch across the frequency sweep. This pool accounts for
    // solver scratch only; existing operator/assembly allocations remain separate.
    let krylov_pool = sonium_compute::MemoryPool::new(
        backend,
        sonium_compute::MemoryPoolConfig::default(),
    )
    .map_err(|error| SimulationError::new(format!("GPU Krylov pool failed: {error}")))?;
    let backend = krylov_pool.backend();
    let mut final_workspace = sonium_compute::GmresWorkspace::<B, Complex64>::new(n_dofs);
    let mut refinement_workspace =
        sonium_compute::gmres_ir::GmresIrWorkspace::<B, Complex64>::new(&krylov_pool);
    let mut metal_workspace =
        sonium_compute::GmresWorkspace::<B, num_complex::Complex32>::new(n_dofs);

    let (row_ptr, col_idx) = dense_csr_structure(n_dofs)?;
    // Block-Jacobi over fixed diagonal blocks at production sizes;
    // block_size = 1 below the cut reproduces point Jacobi exactly.
    let gpu_block_size = if n_dofs >= GPU_BLOCK_JACOBI_MIN_DOFS {
        sonium_compute::DEFAULT_BLOCK_SIZE
    } else {
        1
    };
    let mut matrix32: Option<sonium_compute::PersistentDeviceCsr<B, num_complex::Complex32>> = None;
    let mut preconditioner: Option<sonium_compute::BlockJacobiBackend<B, num_complex::Complex32>> =
        None;
    let mut previous_x32 = if has_typed_driver {
        None
    } else {
        Some(
            backend
                .raw_alloc::<num_complex::Complex32>(n_dofs)
                .map_err(|error| {
                    SimulationError::new(format!(
                        "GPU previous-solution allocation failed: {error}"
                    ))
                })?,
        )
    };
    let mut x32 = if has_typed_driver {
        None
    } else {
        Some(
            backend
                .raw_alloc::<num_complex::Complex32>(n_dofs)
                .map_err(|error| {
                    SimulationError::new(format!("GPU solution allocation failed: {error}"))
                })?,
        )
    };
    let mut rhs32 = if has_typed_driver {
        None
    } else {
        Some(
            backend
                .raw_alloc::<num_complex::Complex32>(n_dofs)
                .map_err(|error| {
                    SimulationError::new(format!("GPU RHS allocation failed: {error}"))
                })?,
        )
    };
    // Typed-driver products use a Complex64 final BEM solve to keep the
    // physical promotion gate independent of the lower-precision radiation
    // probe path. Prescribed-driver workloads retain the mixed Complex32
    // inner solve above.
    let mut typed_matrix64: Option<sonium_compute::PersistentDeviceCsr<B, Complex64>> = None;
    let mut typed_preconditioner64: Option<sonium_compute::BlockJacobiBackend<B, Complex64>> = None;
    let mut typed_rhs64: Option<B::Buffer<Complex64>> = None;
    let mut typed_x64: Option<B::Buffer<Complex64>> = None;
    let mut previous_x64 = if !has_typed_driver && backend_kind == SimulationBackend::Cuda {
        Some(backend.raw_alloc::<Complex64>(n_dofs).map_err(|error| {
            SimulationError::new(format!(
                "GPU refinement warm-start allocation failed: {error}"
            ))
        })?)
    } else {
        None
    };
    // The typed radiation matrix is probed with one unit load per driver.
    // Keep the complete column-major batch and block-GMRES scratch resident
    // so the repeated-load solve does not allocate or upload a scalar RHS per
    // column.  The workspace is scoped to this invocation, together with the
    // geometry/operator identity above.
    let mut typed_block_rhs64: Option<B::Buffer<Complex64>> = None;
    let mut typed_block_x64: Option<B::Buffer<Complex64>> = None;
    let mut typed_block_workspace =
        has_typed_driver.then(|| sonium_compute::BlockGmresWorkspace::<B, Complex64>::new(n_dofs));
    let mut coupled_block_rhs_resident = false;
    let mut coupled_block_solution_resident = false;
    let mut coupled_block_rhs_count = 0;
    let mut coupled_block_iterations = 0;
    // The small coupled network uses Complex64 because CUDA's native sparse
    // and reduction path is Complex64. The large BEM inner solve remains the
    // mixed Complex32/f64-gated path above.
    let mut coupled_workspace =
        sonium_compute::GmresWorkspace::<B, Complex64>::new(model.drivers().len());
    let mut coupled_matrix64: Option<sonium_compute::PersistentDeviceCsr<B, Complex64>> = None;
    // The coupled N-port network is driver-count sized (single digits), so
    // point Jacobi stays: block inverses would buy nothing here.
    let mut coupled_preconditioner: Option<sonium_compute::PointJacobiBackend<B, Complex64>> = None;
    let mut coupled_rhs64: Option<B::Buffer<Complex64>> = None;
    let mut coupled_x64: Option<B::Buffer<Complex64>> = None;
    let mut has_previous = false;
    let boundary_element_count = elements
        .iter()
        .filter(|element| !element.property.is_evaluation())
        .count();
    if !matches!(
        capabilities.bem.field_evaluation.c64,
        sonium_compute::CapabilityState::Native | sonium_compute::CapabilityState::VendorNative
    ) {
        return Err(SimulationError::new(format!(
            "GPU speaker field evaluation is unavailable for Complex64: capability={:?}",
            capabilities.bem.field_evaluation.c64
        )));
    }
    let field_geometry_host = build_speaker_field_geometry(&elements, &nodes)?;
    if field_geometry_host.len() != boundary_element_count {
        return Err(SimulationError::new(
            "GPU speaker field geometry count does not match boundary elements",
        ));
    }
    let mut field_geometry = backend
        .raw_alloc::<sonium_compute::BemFieldElementC64>(field_geometry_host.len())
        .map_err(|error| {
            SimulationError::new(format!("GPU field geometry allocation failed: {error}"))
        })?;
    backend
        .try_copy_h2d(&field_geometry_host, &mut field_geometry)
        .map_err(|error| {
            SimulationError::new(format!("GPU field geometry upload failed: {error}"))
        })?;
    let field_points_host: Vec<_> = eval_points
        .iter()
        .copied()
        .map(sonium_compute::BemFieldPointC64::new)
        .collect();
    let mut field_points = backend
        .raw_alloc::<sonium_compute::BemFieldPointC64>(field_points_host.len())
        .map_err(|error| {
            SimulationError::new(format!("GPU field point allocation failed: {error}"))
        })?;
    backend
        .try_copy_h2d(&field_points_host, &mut field_points)
        .map_err(|error| SimulationError::new(format!("GPU field point upload failed: {error}")))?;
    let mut field_pressure = backend
        .raw_alloc::<Complex64>(boundary_element_count)
        .map_err(|error| {
            SimulationError::new(format!("GPU field pressure allocation failed: {error}"))
        })?;
    let mut field_velocity = backend
        .raw_alloc::<Complex64>(boundary_element_count)
        .map_err(|error| {
            SimulationError::new(format!("GPU field velocity allocation failed: {error}"))
        })?;
    let mut field_output = backend
        .raw_alloc::<Complex64>(eval_points.len())
        .map_err(|error| {
            SimulationError::new(format!("GPU field output allocation failed: {error}"))
        })?;
    let geometry_key = gpu_geometry_key(&nodes, &elements, eval_points);
    let frequencies = model.simulation.frequencies();
    let frequency_count = frequencies.len();
    let precision_policy = PrecisionPolicy::MixedF32InnerF64Gated;
    let frequency_grid_key = gpu_frequency_grid_key(&frequencies);
    let mut matrix_value_updates = 0;
    let mut preconditioner_updates = 0;
    let mut coupled_matrix_value_updates = 0;
    let mut coupled_preconditioner_updates = 0;
    let mut assembly_ms = 0;
    let mut solve_ms = 0;
    let mut residual_gate_ms = 0;
    let mut field_ms = 0;
    let mut reference_ms = 0;
    let mut all_spl =
        HostMatrix::<f64>::zeros(model.simulation.frequencies().len(), eval_points.len());
    let mut all_phase =
        HostMatrix::<f64>::zeros(model.simulation.frequencies().len(), eval_points.len());
    let mut all_complex =
        HostMatrix::<Complex64>::zeros(model.simulation.frequencies().len(), eval_points.len());
    let mut woofer_impedance = vec![None; model.simulation.frequencies().len()];
    let mut tweeter_impedance = vec![None; model.simulation.frequencies().len()];
    let mut driver_electrical_impedances_ohm =
        vec![vec![None; model.drivers().len()]; model.simulation.frequencies().len()];
    let mut radiation_matrices = vec![None; model.simulation.frequencies().len()];
    let mut radiation_matrix_diagnostics = vec![None; model.simulation.frequencies().len()];
    let thermal_driver_states = vec![[None, None]; model.simulation.frequencies().len()];
    let driver_thermal_states =
        vec![vec![None; model.drivers().len()]; model.simulation.frequencies().len()];
    let mut power_ledgers = vec![None; model.simulation.frequencies().len()];
    let structural_responses = vec![None; model.simulation.frequencies().len()];
    let mut driver_velocities_bem = vec![vec![]; model.simulation.frequencies().len()];
    let mut max_relative_residual = 0.0_f64;
    let mut max_physical_error = 0.0_f64;
    let mut max_coupled_power_error = 0.0_f64;
    let mut max_coupled_observable_error = 0.0_f64;
    let mut max_coupled_derivative_error = 0.0_f64;
    let mut max_derivative_assembly_error = 0.0_f64;
    let mut validation = GpuValidationReport {
        policy: solver_config.gpu_validation,
        ..Default::default()
    };
    let gate_policy = MixedPrecisionPolicy {
        outer_tolerance: solver_config.tolerance,
        physical_tolerance: 1.0e-3,
        ..MixedPrecisionPolicy::default()
    };

    for (frequency_index, &frequency) in model.simulation.frequencies().iter().enumerate() {
        if let Some(control) = control {
            control.check()?;
            control.report_frequency(frequency_index, frequency);
        }
        let velocities = model.velocities_at(frequency);
        update_driver_bcs(
            &mut elements,
            &speaker_mesh,
            model,
            frequency,
            Some(&velocities),
        )?;
        let physics = PhysicsParams::new(
            frequency,
            model.simulation.speed_of_sound,
            model.simulation.air_density,
            false,
        );
        let beta = physics.burton_miller_beta_scaled(solver_config.beta_scale);
        let assembly_start = Instant::now();
        let assembled = assemble(&elements, &nodes, &physics, beta, backend)?;
        let derivative_system = assembled.wavenumber_derivative;
        let system = assembled.value;
        assembly_ms += assembly_start.elapsed().as_millis() as u64;
        if system.matrix.nrows() != n_dofs || system.rhs.len() != n_dofs {
            return Err(SimulationError::new(
                "GPU speaker BEM assembly returned inconsistent matrix/RHS dimensions",
            ));
        }

        let values64 = system.matrix.as_slice();
        if let Some(matrix) = matrix32.as_mut() {
            matrix
                .try_copy_values_from_host_cast(backend, values64)
                .map_err(|error| {
                    SimulationError::new(format!(
                        "GPU CSR value update failed at {frequency:.1} Hz: {error}"
                    ))
                })?;
            matrix_value_updates += 1;
        } else {
            let matrix = sonium_compute::DeviceCsr::try_from_host_cast(
                backend, n_dofs, n_dofs, &row_ptr, &col_idx, values64,
            )
            .map_err(|error| {
                SimulationError::new(format!(
                    "GPU f32 CSR allocation failed at {frequency:.1} Hz: {error}"
                ))
            })?;
            matrix32 = Some(
                sonium_compute::PersistentDeviceCsr::try_from_device_csr(backend, matrix)
                    .map_err(|error| {
                        SimulationError::new(format!(
                            "GPU CSR prebake failed at {frequency:.1} Hz: {error}"
                        ))
                    })?,
            );
        }

        if let Some(preconditioner) = preconditioner.as_mut() {
            preconditioner
                .try_update_dense_host_cast(backend, values64)
                .map_err(|error| {
                    SimulationError::new(format!(
                        "GPU block-Jacobi update failed at {frequency:.1} Hz: {error}"
                    ))
                })?;
            preconditioner_updates += 1;
        } else {
            preconditioner = Some(
                sonium_compute::BlockJacobiBackend::<B, num_complex::Complex32>::try_from_dense_host_cast(
                    backend,
                    values64,
                    n_dofs,
                    gpu_block_size,
                )
                .map_err(|error| {
                    SimulationError::new(format!(
                        "GPU block-Jacobi allocation failed at {frequency:.1} Hz: {error}"
                    ))
                })?,
            );
        }
        if has_typed_driver || backend_kind == SimulationBackend::Cuda {
            if let Some(matrix) = typed_matrix64.as_mut() {
                matrix
                    .try_copy_values_from_host(backend, values64)
                    .map_err(|error| {
                        SimulationError::new(format!(
                            "GPU typed Complex64 CSR value update failed at {frequency:.1} Hz: {error}"
                        ))
                    })?;
            } else {
                let matrix = sonium_compute::DeviceCsr::try_from_host(
                    backend, n_dofs, n_dofs, &row_ptr, &col_idx, values64,
                )
                .map_err(|error| {
                    SimulationError::new(format!(
                        "GPU typed Complex64 CSR allocation failed at {frequency:.1} Hz: {error}"
                    ))
                })?;
                typed_matrix64 = Some(
                    sonium_compute::PersistentDeviceCsr::try_from_device_csr(backend, matrix)
                        .map_err(|error| {
                            SimulationError::new(format!(
                                "GPU typed Complex64 CSR prebake failed at {frequency:.1} Hz: {error}"
                            ))
                        })?,
                );
            }
            if let Some(preconditioner) = typed_preconditioner64.as_mut() {
                preconditioner
                    .try_update_dense_host(backend, values64)
                    .map_err(|error| {
                        SimulationError::new(format!(
                            "GPU typed Complex64 block-Jacobi update failed at {frequency:.1} Hz: {error}"
                        ))
                    })?;
            } else {
                typed_preconditioner64 = Some(
                    sonium_compute::BlockJacobiBackend::try_from_dense_host(
                        backend,
                        values64,
                        n_dofs,
                        gpu_block_size,
                    )
                    .map_err(|error| {
                        SimulationError::new(format!(
                            "GPU typed Complex64 block-Jacobi allocation failed at {frequency:.1} Hz: {error}"
                        ))
                    })?,
                );
            }
            if typed_rhs64.is_none() {
                typed_rhs64 = Some(backend.raw_alloc::<Complex64>(n_dofs).map_err(|error| {
                    SimulationError::new(format!(
                        "GPU typed Complex64 RHS allocation failed: {error}"
                    ))
                })?);
            }
            if typed_x64.is_none() {
                typed_x64 = Some(backend.raw_alloc::<Complex64>(n_dofs).map_err(|error| {
                    SimulationError::new(format!(
                        "GPU typed Complex64 solution allocation failed: {error}"
                    ))
                })?);
            }
        }
        let (final_rhs64, gpu_typed_coupling) = if has_typed_driver {
            let driver_count = model.drivers().len();
            let mut z_rad = vec![vec![Complex64::default(); driver_count]; driver_count];
            let block_len = n_dofs.checked_mul(driver_count).ok_or_else(|| {
                SimulationError::new("GPU typed radiation block RHS size overflow")
            })?;
            let mut unit_rhs_block = Vec::with_capacity(block_len);
            for column in 0..driver_count {
                let unit_velocities: Vec<Complex64> = (0..driver_count)
                    .map(|row| Complex64::new((row == column) as u8 as f64, 0.0))
                    .collect();
                update_driver_bcs(
                    &mut elements,
                    &speaker_mesh,
                    model,
                    frequency,
                    Some(&unit_velocities),
                )?;
                let unit_rhs =
                    sonium_bem::core::assembly::tbem::build_tbem_rhs_stabilized_with_beta(
                        &elements,
                        &nodes,
                        &physics,
                        EXTERIOR_TBEM_SIGN,
                        beta,
                    );
                if unit_rhs.len() != n_dofs {
                    return Err(SimulationError::new(format!(
                        "GPU typed radiation unit-load RHS has {} entries for {n_dofs} active DOFs",
                        unit_rhs.len()
                    )));
                }
                unit_rhs_block.extend_from_slice(&unit_rhs);
            }
            if typed_block_rhs64.is_none() {
                typed_block_rhs64 =
                    Some(backend.raw_alloc::<Complex64>(block_len).map_err(|error| {
                        SimulationError::new(format!(
                            "GPU typed block RHS allocation failed: {error}"
                        ))
                    })?);
                typed_block_x64 =
                    Some(backend.raw_alloc::<Complex64>(block_len).map_err(|error| {
                        SimulationError::new(format!(
                            "GPU typed block solution allocation failed: {error}"
                        ))
                    })?);
                coupled_block_rhs_resident = true;
                coupled_block_solution_resident = true;
            }
            let typed_block_rhs = typed_block_rhs64
                .as_mut()
                .ok_or_else(|| SimulationError::new("GPU typed block RHS was not initialized"))?;
            backend
                .try_copy_h2d(&unit_rhs_block, typed_block_rhs)
                .map_err(|error| {
                    SimulationError::new(format!(
                        "GPU typed block radiation RHS upload failed at {frequency:.1} Hz: {error}"
                    ))
                })?;
            let typed_matrix = typed_matrix64.as_ref().ok_or_else(|| {
                SimulationError::new("GPU typed Complex64 matrix was not initialized")
            })?;
            let typed_preconditioner = typed_preconditioner64.as_ref().ok_or_else(|| {
                SimulationError::new("GPU typed Complex64 preconditioner was not initialized")
            })?;
            let typed_block_rhs = typed_block_rhs64
                .as_ref()
                .ok_or_else(|| SimulationError::new("GPU typed block RHS was not initialized"))?;
            let typed_block_solution = typed_block_x64.as_mut().ok_or_else(|| {
                SimulationError::new("GPU typed block solution was not initialized")
            })?;
            let typed_workspace = typed_block_workspace.as_mut().ok_or_else(|| {
                SimulationError::new("GPU typed block workspace was not initialized")
            })?;
            let block_status = sonium_compute::gmres_backend_block_preconditioned_with_workspace(
                backend,
                typed_matrix,
                typed_preconditioner,
                typed_block_rhs,
                None,
                typed_block_solution,
                driver_count,
                &sonium_compute::BlockGmresConfig {
                    max_block_iterations: solver_config.max_iterations.clamp(1, 100),
                    tolerance: (solver_config.tolerance * 0.01).max(1.0e-12),
                },
                typed_workspace,
            )
            .map_err(|error| {
                SimulationError::new(format!(
                    "GPU typed block radiation GMRES failed at {frequency:.1} Hz: {error}"
                ))
            })?;
            if !block_status.converged {
                return Err(SimulationError::new(format!(
                    "GPU typed block radiation GMRES did not converge at {frequency:.1} Hz: residual={:.3e}",
                    block_status.final_residual
                )));
            }
            coupled_block_rhs_count = driver_count;
            coupled_block_iterations = coupled_block_iterations.max(block_status.block_iterations);
            let mut unit_pressure_block = vec![Complex64::new(0.0, 0.0); block_len];
            let typed_block_solution = typed_block_x64.as_ref().ok_or_else(|| {
                SimulationError::new("GPU typed block solution was not initialized")
            })?;
            backend
                .try_copy_d2h(typed_block_solution, &mut unit_pressure_block)
                .map_err(|error| {
                    SimulationError::new(format!(
                        "GPU typed block radiation solution download failed at {frequency:.1} Hz: {error}"
                    ))
                })?;
            for column in 0..driver_count {
                let unit_pressure = &unit_pressure_block[column * n_dofs..(column + 1) * n_dofs];
                let unit_rhs = &unit_rhs_block[column * n_dofs..(column + 1) * n_dofs];
                let unit_residual = relative_residual(&system.matrix, unit_pressure, unit_rhs);
                if !unit_residual.is_finite() || unit_residual > solver_config.tolerance {
                    return Err(SimulationError::new(format!(
                        "GPU typed radiation unit-load residual failed at {frequency:.1} Hz: residual={unit_residual:.3e}, tolerance={:.3e}",
                        solver_config.tolerance
                    )));
                }
                let forces = integrate_driver_axial_forces(&elements, &speaker_mesh, unit_pressure);
                for (row, force) in forces.into_iter().enumerate().take(driver_count) {
                    z_rad[row][column] = bem_impedance_to_modeling(force);
                }
            }
            let coupling = solve_gpu_linear_driver_network(
                &krylov_pool,
                &mut coupled_workspace,
                model,
                frequency,
                &z_rad,
                solver_config,
                &mut coupled_matrix64,
                &mut coupled_preconditioner,
                &mut coupled_rhs64,
                &mut coupled_x64,
                &mut coupled_matrix_value_updates,
                &mut coupled_preconditioner_updates,
            )?;
            update_driver_bcs(
                &mut elements,
                &speaker_mesh,
                model,
                frequency,
                Some(&coupling.velocities_bem_all),
            )?;
            let rhs = sonium_bem::core::assembly::tbem::build_tbem_rhs_stabilized_with_beta(
                &elements,
                &nodes,
                &physics,
                EXTERIOR_TBEM_SIGN,
                beta,
            );
            (rhs, Some(coupling))
        } else {
            (system.rhs.clone(), None)
        };
        let solve_start = Instant::now();
        let mut surface_pressure = vec![Complex64::new(0.0, 0.0); n_dofs];
        let (status_converged, status_final_residual) = if has_typed_driver {
            let rhs64 = typed_rhs64.as_mut().ok_or_else(|| {
                SimulationError::new("GPU typed Complex64 RHS was not initialized")
            })?;
            let typed_matrix = typed_matrix64.as_ref().ok_or_else(|| {
                SimulationError::new("GPU typed Complex64 matrix was not initialized")
            })?;
            let typed_preconditioner = typed_preconditioner64.as_ref().ok_or_else(|| {
                SimulationError::new("GPU typed Complex64 preconditioner was not initialized")
            })?;
            let typed_solution = typed_x64.as_mut().ok_or_else(|| {
                SimulationError::new("GPU typed Complex64 solution was not initialized")
            })?;
            backend.try_copy_h2d(&final_rhs64, rhs64).map_err(|error| {
                SimulationError::new(format!(
                    "GPU typed Complex64 RHS upload failed at {frequency:.1} Hz: {error}"
                ))
            })?;
            let status = sonium_compute::gmres::gmres_backend_preconditioned_pooled_with_workspace(
                &krylov_pool,
                &mut final_workspace,
                typed_matrix,
                typed_preconditioner,
                rhs64,
                None,
                typed_solution,
                &sonium_compute::GmresBackendConfig::<Complex64> {
                    max_iterations: solver_config.max_iterations,
                    // The final typed-driver field is a promoted physical
                    // observable.  Solve it one decade tighter than the
                    // public residual gate so field amplification cannot
                    // turn an otherwise acceptable Krylov residual into a
                    // false physical pass.
                    tolerance: (solver_config.tolerance * 0.01).max(1.0e-12),
                },
            )
            .map_err(|error| {
                SimulationError::new(format!("GPU typed Complex64 GMRES failed: {error}"))
            })?;
            let typed_solution = typed_x64.as_ref().ok_or_else(|| {
                SimulationError::new("GPU typed Complex64 solution was not initialized")
            })?;
            backend
                .try_copy_d2h(typed_solution, &mut surface_pressure)
                .map_err(|error| {
                    SimulationError::new(format!(
                        "GPU typed Complex64 solution download failed: {error}"
                    ))
                })?;
            (status.converged, status.final_residual)
        } else if backend_kind == SimulationBackend::Cuda {
            // Plain f32 GMRES can stagnate above the requested f64 residual.
            // Retain the f32 inner work and refine with the original f64
            // operator; acceptance still uses the independent host residual.
            let rhs = typed_rhs64
                .as_mut()
                .ok_or_else(|| SimulationError::new("GPU refinement RHS missing"))?;
            backend.try_copy_h2d(&final_rhs64, rhs).map_err(|error| {
                SimulationError::new(format!("GPU refinement RHS upload failed: {error}"))
            })?;
            let status = sonium_compute::gmres_ir::gmres_ir_backend_preconditioned_with_workspace(
                &mut refinement_workspace,
                typed_matrix64
                    .as_ref()
                    .ok_or_else(|| SimulationError::new("GPU refinement f64 operator missing"))?,
                matrix32
                    .as_ref()
                    .ok_or_else(|| SimulationError::new("GPU refinement f32 operator missing"))?,
                preconditioner
                    .as_ref()
                    .ok_or_else(|| SimulationError::new("GPU refinement preconditioner missing"))?,
                rhs,
                if has_previous {
                    previous_x64.as_ref()
                } else {
                    None
                },
                typed_x64
                    .as_mut()
                    .ok_or_else(|| SimulationError::new("GPU refinement solution missing"))?,
                &sonium_compute::GmresIrConfig::<Complex64> {
                    max_outer_iterations: solver_config.max_iterations.clamp(1, 40),
                    outer_tolerance: (solver_config.tolerance * 0.1).max(1e-12),
                    max_inner_iterations: solver_config.max_iterations.min(100),
                    inner_tolerance: 1e-3,
                    true_residual_check_period: Some(5),
                    ..Default::default()
                },
            )
            .map_err(|error| {
                SimulationError::new(format!(
                    "GPU speaker mixed-precision refinement failed: {error}"
                ))
            })?;
            backend
                .try_copy_d2h(
                    typed_x64
                        .as_ref()
                        .ok_or_else(|| SimulationError::new("GPU refinement solution missing"))?,
                    &mut surface_pressure,
                )
                .map_err(|error| {
                    SimulationError::new(format!(
                        "GPU refinement solution download failed: {error}"
                    ))
                })?;
            (status.converged, status.final_outer_residual)
        } else {
            let rhs32_host: Vec<num_complex::Complex32> = final_rhs64
                .iter()
                .map(|value| num_complex::Complex32::new(value.re as f32, value.im as f32))
                .collect();
            let rhs32_device = rhs32.as_mut().ok_or_else(|| {
                SimulationError::new("GPU prescribed RHS buffer was not initialized")
            })?;
            backend
                .try_copy_h2d(&rhs32_host, rhs32_device)
                .map_err(|error| {
                    SimulationError::new(format!(
                        "GPU RHS upload failed at {frequency:.1} Hz: {error}"
                    ))
                })?;
            let matrix = matrix32
                .as_ref()
                .ok_or_else(|| SimulationError::new("GPU f32 matrix was not initialized"))?;
            let preconditioner = preconditioner
                .as_ref()
                .ok_or_else(|| SimulationError::new("GPU preconditioner was not initialized"))?;
            let rhs32_device = rhs32.as_ref().ok_or_else(|| {
                SimulationError::new("GPU prescribed RHS buffer was not initialized")
            })?;
            let previous_solution = if has_previous {
                Some(previous_x32.as_ref().ok_or_else(|| {
                    SimulationError::new(
                        "GPU prescribed previous-solution buffer was not initialized",
                    )
                })?)
            } else {
                None
            };
            let solution = x32.as_mut().ok_or_else(|| {
                SimulationError::new("GPU prescribed solution buffer was not initialized")
            })?;
            let status = sonium_compute::gmres::gmres_backend_preconditioned_pooled_with_workspace(
                &krylov_pool,
                &mut metal_workspace,
                matrix,
                preconditioner,
                rhs32_device,
                previous_solution,
                solution,
                &sonium_compute::GmresBackendConfig::<num_complex::Complex32> {
                    max_iterations: solver_config.max_iterations,
                    tolerance: solver_config.tolerance as f32,
                },
            )
            .map_err(|error| SimulationError::new(format!("GPU speaker GMRES failed: {error}")))?;
            let mut surface_pressure32 = vec![num_complex::Complex32::new(0.0, 0.0); n_dofs];
            let solution = x32.as_ref().ok_or_else(|| {
                SimulationError::new("GPU prescribed solution buffer was not initialized")
            })?;
            backend
                .try_copy_d2h(solution, &mut surface_pressure32)
                .map_err(|error| {
                    SimulationError::new(format!(
                        "GPU solution download failed at {frequency:.1} Hz: {error}"
                    ))
                })?;
            for (dst, src) in surface_pressure.iter_mut().zip(surface_pressure32) {
                *dst = Complex64::new(src.re as f64, src.im as f64);
            }
            (status.converged, status.final_residual as f64)
        };
        solve_ms += solve_start.elapsed().as_millis() as u64;
        if !status_converged {
            return Err(SimulationError::new(format!(
                "GPU speaker GMRES did not converge at {frequency:.1} Hz: residual={:.3e}",
                status_final_residual
            )));
        }

        let residual_start = Instant::now();
        // The f64 residual is deliberately reconstructed on the CPU from the
        // original assembled matrix and RHS.  A device f64 copy of the same
        // promoted vector is not an independent acceptance gate.
        let mut relative_residual =
            relative_residual(&system.matrix, &surface_pressure, &final_rhs64);
        residual_gate_ms += residual_start.elapsed().as_millis() as u64;
        if !relative_residual.is_finite() || relative_residual > solver_config.tolerance {
            return Err(SimulationError::new(format!(
                "GPU speaker CPU f64 residual gate failed at {frequency:.1} Hz: residual={relative_residual:.3e}, tolerance={:.3e}",
                solver_config.tolerance
            )));
        }
        let parity_start = Instant::now();
        let reference_due = solver_config.gpu_validation.reference_due(
            frequency_index,
            frequency_count,
            relative_residual,
            solver_config.tolerance,
        );
        if let Some(derivative_system) = derivative_system.as_ref().filter(|_| reference_due) {
            let cpu_derivative =
                sonium_bem::core::assembly::tbem::build_tbem_system_stabilized_wavenumber_derivative(
                    &elements,
                    &nodes,
                    &physics,
                    EXTERIOR_TBEM_SIGN,
                    solver_config.beta_scale,
                );
            let derivative_error = relative_system_difference(derivative_system, &cpu_derivative);
            if !derivative_error.is_finite() || derivative_error > GPU_DERIVATIVE_ASSEMBLY_TOLERANCE
            {
                return Err(SimulationError::new(format!(
                    "GPU BEM wavenumber derivative CPU f64 gate failed at {frequency:.1} Hz: error={derivative_error:.3e}, tolerance={GPU_DERIVATIVE_ASSEMBLY_TOLERANCE:.3e}"
                )));
            }
            max_derivative_assembly_error = max_derivative_assembly_error.max(derivative_error);
            validation
                .derivative_frequency_indices
                .push(frequency_index);
        } else if solver_config.gpu_wavenumber_derivative
            && reference_due
            && (capability_matrix.derivative_assembly == GpuCapabilityState::Native
                || capability_matrix.derivative_assembly == GpuCapabilityState::VendorNative)
        {
            return Err(SimulationError::new(
                "GPU capability matrix advertised derivative assembly without a derivative result",
            ));
        }

        let cpu_typed_reference = if let Some(gpu_coupling) =
            gpu_typed_coupling.as_ref().filter(|_| reference_due)
        {
            let mut cpu_subspace = None;
            let mut cpu_sentinel = ExactResidualSentinel::default();
            let (cpu_pressure, cpu_coupling) = solve_typed_driver_coupling_with_structure_control(
                &mut elements,
                &speaker_mesh,
                model,
                frequency,
                &nodes,
                &physics,
                Some(&system.matrix),
                None,
                solver_config,
                &mut cpu_subspace,
                None,
                None,
                &mut cpu_sentinel,
                None,
                control,
            )?;
            let cpu_surface_velocity = build_surface_velocity(&elements);
            let cpu_field = compute_scattered_field(
                eval_points,
                &elements,
                &nodes,
                &cpu_pressure,
                Some(&cpu_surface_velocity),
                &physics,
                BasisOrder::Constant,
            );
            let voltage = typed_driver_drive_voltages(model, frequency)?;
            let velocity_error = gpu_coupling
                .velocities_bem_all
                .iter()
                .zip(&cpu_coupling.velocities_bem_all)
                .map(|(actual, reference)| relative_complex_difference(*actual, *reference))
                .fold(0.0, f64::max);
            let impedance_error = gpu_coupling
                .driver_electrical_impedances_ohm
                .iter()
                .zip(&cpu_coupling.driver_electrical_impedances_ohm)
                .map(|(actual, reference)| match (actual, reference) {
                    (Some(actual), Some(reference)) => {
                        relative_complex_difference(*actual, *reference)
                    }
                    _ => f64::INFINITY,
                })
                .fold(0.0, f64::max);
            let derivative_error = gpu_coupling
                .velocities_bem_all
                .iter()
                .zip(&cpu_coupling.velocities_bem_all)
                .zip(&voltage)
                .map(|((actual, reference), voltage)| {
                    relative_complex_difference(*actual / *voltage, *reference / *voltage)
                })
                .fold(0.0, f64::max);
            let power_error = match (gpu_coupling.power_ledger, cpu_coupling.power_ledger) {
                (Some(actual), Some(reference)) => power_ledger_difference(actual, reference),
                _ => f64::INFINITY,
            };
            max_coupled_derivative_error = max_coupled_derivative_error.max(derivative_error);
            max_coupled_power_error = max_coupled_power_error.max(power_error);
            max_coupled_observable_error = max_coupled_observable_error.max(velocity_error);
            max_coupled_observable_error = max_coupled_observable_error.max(impedance_error);
            update_driver_bcs(
                &mut elements,
                &speaker_mesh,
                model,
                frequency,
                Some(&gpu_coupling.velocities_bem_all),
            )?;
            Some((cpu_field, cpu_coupling))
        } else {
            None
        };
        reference_ms += parity_start.elapsed().as_millis() as u64;
        let surface_velocity = build_surface_velocity(&elements);
        let field_start = Instant::now();
        let mut gpu_field = evaluate_speaker_field_gpu(
            backend,
            &field_geometry,
            &mut field_pressure,
            &mut field_velocity,
            &field_points,
            &mut field_output,
            &surface_pressure,
            &surface_velocity,
            &physics,
        )?;
        field_ms += field_start.elapsed().as_millis() as u64;
        // Independent f64 host reference: this solves the assembled f64
        // system directly instead of evaluating the same promoted f32
        // solution used by the GPU residual gate.
        if reference_due {
            let reference_start = Instant::now();
            let cpu_system = build_tbem_system_stabilized(
                &elements,
                &nodes,
                &physics,
                EXTERIOR_TBEM_SIGN,
                solver_config.beta_scale,
            );
            let host_reference_solution = lu_solve(
                &cpu_system.matrix,
                if has_typed_driver {
                    &final_rhs64
                } else {
                    &cpu_system.rhs
                },
            )
            .map_err(|error| {
                SimulationError::new(format!(
                    "GPU speaker host f64 reference solve failed at {frequency:.1} Hz: {error}"
                ))
            })?;
            let host_reference_field = compute_scattered_field(
                eval_points,
                &elements,
                &nodes,
                &host_reference_solution,
                Some(&surface_velocity),
                &physics,
                BasisOrder::Constant,
            );
            let mut physical_error = gpu_field
                .iter()
                .zip(&host_reference_field)
                .map(|(observed, reference)| {
                    (observed - reference).norm() / reference.norm().max(1.0e-30)
                })
                .fold(0.0_f64, f64::max);
            validation.max_uncorrected_physical_error = validation
                .max_uncorrected_physical_error
                .max(physical_error);
            if physical_error > gate_policy.physical_tolerance
                && solver_config.gpu_reference_correction
            {
                if backend_kind != SimulationBackend::Cuda
                    || has_typed_driver
                    || solver_config.gpu_wavenumber_derivative
                {
                    return Err(SimulationError::new(
                        "GPU reference correction supports prescribed CUDA forward sweeps only",
                    ));
                }
                let matrix = typed_matrix64
                    .as_mut()
                    .ok_or_else(|| SimulationError::new("CUDA correction matrix missing"))?;
                matrix
                    .try_copy_values_from_host(backend, cpu_system.matrix.as_slice())
                    .map_err(|e| {
                        SimulationError::new(format!("CUDA reference operator upload failed: {e}"))
                    })?;
                matrix_value_updates += 1;
                let preconditioner = typed_preconditioner64.as_mut().ok_or_else(|| {
                    SimulationError::new("CUDA correction preconditioner missing")
                })?;
                preconditioner
                    .try_update_dense_host(backend, cpu_system.matrix.as_slice())
                    .map_err(|e| {
                        SimulationError::new(format!(
                            "CUDA reference preconditioner update failed: {e}"
                        ))
                    })?;
                preconditioner_updates += 1;
                let (corrected, _) = solve_gpu_coupled_rhs(
                    &krylov_pool,
                    &mut final_workspace,
                    matrix,
                    preconditioner,
                    &cpu_system.rhs,
                    typed_rhs64
                        .as_mut()
                        .ok_or_else(|| SimulationError::new("CUDA correction RHS missing"))?,
                    typed_x64
                        .as_mut()
                        .ok_or_else(|| SimulationError::new("CUDA correction solution missing"))?,
                    solver_config,
                )?;
                surface_pressure = corrected;
                relative_residual = crate::simulation::relative_residual(
                    &cpu_system.matrix,
                    &surface_pressure,
                    &cpu_system.rhs,
                );
                gpu_field = evaluate_speaker_field_gpu(
                    backend,
                    &field_geometry,
                    &mut field_pressure,
                    &mut field_velocity,
                    &field_points,
                    &mut field_output,
                    &surface_pressure,
                    &surface_velocity,
                    &physics,
                )?;
                physical_error = gpu_field
                    .iter()
                    .zip(&host_reference_field)
                    .map(|(observed, reference)| {
                        (observed - reference).norm() / reference.norm().max(1e-30)
                    })
                    .fold(0.0_f64, f64::max);
                validation.corrected_frequency_indices.push(frequency_index);
            }
            if let Some((cpu_field, _)) = cpu_typed_reference.as_ref() {
                let coupled_observable_error = gpu_field
                    .iter()
                    .zip(cpu_field)
                    .map(|(actual, reference)| relative_complex_difference(*actual, *reference))
                    .fold(0.0, f64::max);
                max_coupled_observable_error =
                    max_coupled_observable_error.max(coupled_observable_error);
            }
            reference_ms += reference_start.elapsed().as_millis() as u64;
            gate_policy
                .validate_gates(relative_residual, physical_error)
                .map_err(|error| {
                    SimulationError::new(format!(
                        "GPU speaker f64 acceptance gate failed at {frequency:.1} Hz: {error}"
                    ))
                })?;
            validation.reference_frequency_indices.push(frequency_index);
            max_physical_error = max_physical_error.max(physical_error);
        }
        max_relative_residual = max_relative_residual.max(relative_residual);
        if let Some(coupling) = gpu_typed_coupling.as_ref() {
            radiation_matrices[frequency_index] = Some(coupling.radiation_matrix_all.clone());
            radiation_matrix_diagnostics[frequency_index] =
                Some(coupling.radiation_matrix_diagnostics);
            power_ledgers[frequency_index] = coupling.power_ledger;
            driver_velocities_bem[frequency_index] = coupling.velocities_bem_all.clone();
            driver_electrical_impedances_ohm[frequency_index] =
                coupling.driver_electrical_impedances_ohm.clone();
            woofer_impedance[frequency_index] = coupling.woofer_electrical_impedance_ohm;
            tweeter_impedance[frequency_index] = coupling.tweeter_electrical_impedance_ohm;
        }
        all_complex
            .row_mut(frequency_index)
            .copy_from_slice(&gpu_field);
        for (point_index, &pressure) in gpu_field.iter().enumerate() {
            all_spl[[frequency_index, point_index]] = sonium_xem::pressure_to_spl(pressure);
            all_phase[[frequency_index, point_index]] = pressure.arg();
        }
        if !has_typed_driver && backend_kind == SimulationBackend::Cuda {
            backend
                .try_copy_d2d(
                    typed_x64
                        .as_ref()
                        .ok_or_else(|| SimulationError::new("GPU refinement solution missing"))?,
                    previous_x64.as_mut().ok_or_else(|| {
                        SimulationError::new("GPU refinement previous solution missing")
                    })?,
                )
                .map_err(|error| {
                    SimulationError::new(format!("GPU refinement warm-start copy failed: {error}"))
                })?;
        } else if !has_typed_driver {
            let solution = x32.as_ref().ok_or_else(|| {
                SimulationError::new("GPU prescribed solution buffer was not initialized")
            })?;
            let previous_solution = previous_x32.as_mut().ok_or_else(|| {
                SimulationError::new("GPU prescribed previous-solution buffer was not initialized")
            })?;
            backend
                .try_copy_d2d(solution, previous_solution)
                .map_err(|error| {
                    SimulationError::new(format!(
                        "GPU warm-start copy failed at {frequency:.1} Hz: {error}"
                    ))
                })?;
        }
        has_previous = true;
        if let Some(control) = control {
            control.report_frequency(frequency_index + 1, frequency);
            control.check()?;
        }
    }

    if has_typed_driver
        && (max_coupled_power_error > CUDA_COUPLED_PARITY_TOLERANCE
            || max_coupled_observable_error > CUDA_COUPLED_PARITY_TOLERANCE
            || max_coupled_derivative_error > CUDA_COUPLED_PARITY_TOLERANCE)
    {
        return Err(SimulationError::new(format!(
            "CUDA typed coupling CPU parity gate failed: power={:.3e}, observable={:.3e}, derivative={:.3e}, tolerance={:.3e}",
            max_coupled_power_error,
            max_coupled_observable_error,
            max_coupled_derivative_error,
            CUDA_COUPLED_PARITY_TOLERANCE,
        )));
    }

    let transfer = backend.transfer_stats();
    let memory = backend.memory_stats();
    let report = GpuExecutionReport {
        schema_version: GpuExecutionReport::SCHEMA_VERSION,
        requested_backend: backend_kind,
        actual_backend: backend_kind,
        device_index,
        capability_matrix,
        frequencies: frequency_count,
        evaluation_points: eval_points.len(),
        active_dofs: n_dofs,
        cache: GpuCacheReport {
            identity: GpuResidencyIdentity {
                schema_version: GpuResidencyIdentity::SCHEMA_VERSION,
                backend: backend_kind,
                precision: precision_policy,
                device_index,
                geometry_key,
                frequency_grid_key,
            },
            geometry_key,
            mesh_resident_across_frequencies: true,
            operator_structure_resident: true,
            // Geometry, points, output, and the pressure/velocity work
            // buffers are allocated once; only the trace values are updated
            // for each frequency.
            field_geometry_resident: true,
            // CUDA uses CudaTbemAssemblyWorkspace; Metal retains its
            // backend-specific one-shot assembly contract.
            assembly_geometry_reused: backend_kind == SimulationBackend::Cuda,
            assembly_workspace_builds: 0,
            assembly_quadrature_schedule_updates: 0,
            matrix_value_updates,
            preconditioner_updates,
            rhs_buffer_reused: frequency_count > 1,
            solution_buffer_reused: frequency_count > 1,
            coupled_operator_structure_resident: has_typed_driver,
            coupled_rhs_buffer_reused: has_typed_driver && frequency_count > 1,
            coupled_solution_buffer_reused: has_typed_driver && frequency_count > 1,
            coupled_block_rhs_resident,
            coupled_block_solution_resident,
            coupled_block_rhs_count,
            coupled_block_iterations,
            coupled_matrix_value_updates,
            coupled_preconditioner_updates,
            explicit_invalidations: 0,
        },
        timings: GpuTimingReport {
            assembly_ms,
            solve_ms,
            residual_gate_ms,
            field_ms,
            reference_ms,
            total_ms: total_start.elapsed().as_millis() as u64,
        },
        transfers: GpuTransferReport {
            h2d_operations: transfer.h2d_operations,
            h2d_bytes: transfer.h2d_bytes,
            d2h_operations: transfer.d2h_operations,
            d2h_bytes: transfer.d2h_bytes,
            d2d_operations: transfer.d2d_operations,
            d2d_bytes: transfer.d2d_bytes,
        },
        memory: GpuMemoryReport {
            total_device_bytes: backend.total_device_memory_bytes(),
            live_allocation_bytes: memory.live_allocation_bytes,
            allocation_high_water_bytes: memory.allocation_high_water_bytes,
            live_staging_bytes: memory.live_staging_bytes,
            staging_high_water_bytes: memory.staging_high_water_bytes,
            post_teardown_live_allocation_bytes: None,
            post_teardown_live_staging_bytes: None,
        },
        residual_gate: ResidualGateStatus::Passed,
        max_relative_residual,
        max_physical_error,
        max_coupled_power_error,
        max_coupled_observable_error,
        max_coupled_derivative_error,
        max_derivative_assembly_error,
        validation,
        // Assembly geometry and the independent CPU f64 reference remain
        // explicit host boundaries, so this is diagnostic timing evidence,
        // not a whole-pipeline GPU performance claim.
        performance_eligible: false,
        failure_modes_fail_closed: vec![
            GpuFailureMode::Allocation,
            GpuFailureMode::DeviceLoss,
            GpuFailureMode::Cancellation,
            GpuFailureMode::NonConvergence,
            GpuFailureMode::ResidualMismatch,
            GpuFailureMode::InvalidGeometry,
            GpuFailureMode::StaleCache,
            GpuFailureMode::InsufficientSampling,
            GpuFailureMode::FailedRomPromotion,
        ],
    };

    // Compat fill of the deprecated per-driver vectors (the driver matrix
    // below is canonical).
    #[allow(deprecated)]
    let result = SimulationResult {
        frequencies: model.simulation.frequencies(),
        spl: all_spl,
        phase: all_phase,
        complex_pressure: all_complex,
        woofer_electrical_impedance_ohm: woofer_impedance,
        tweeter_electrical_impedance_ohm: tweeter_impedance,
        driver_electrical_impedances_ohm,
        radiation_matrices,
        radiation_matrix_diagnostics,
        thermal_driver_states,
        driver_thermal_states,
        power_ledgers,
        product_reference_sweep: None,
        structural_responses,
        driver_velocities_bem,
        exact_residual_evaluations: 0,
        provenance: SimulationProvenance::solved(
            backend_kind,
            solver_config.execution_profile,
            precision_policy,
            ResidualGateReport::passed_with_observations(
                solver_config.tolerance,
                max_relative_residual,
                max_physical_error,
            ),
            model.simulation.frequencies().len(),
            eval_points.len(),
        ),
    };
    Ok((result, report))
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn dense_csr_structure(n: usize) -> Result<(Vec<i32>, Vec<i32>), SimulationError> {
    i32::try_from(n)
        .map_err(|_| SimulationError::new("GPU dense speaker matrix exceeds i32 dimensions"))?;
    let nnz = n
        .checked_mul(n)
        .ok_or_else(|| SimulationError::new("GPU dense speaker matrix size overflow"))?;
    let row_ptr = (0..=n)
        .map(|row| {
            let offset = row
                .checked_mul(n)
                .ok_or_else(|| SimulationError::new("GPU dense speaker matrix size overflow"))?;
            i32::try_from(offset)
                .map_err(|_| SimulationError::new("GPU dense speaker matrix exceeds i32 nnz"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let col_idx = (0..n)
        .cycle()
        .take(nnz)
        .map(|column| {
            i32::try_from(column)
                .map_err(|_| SimulationError::new("GPU dense speaker matrix exceeds i32 width"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    debug_assert_eq!(
        row_ptr.last().copied(),
        Some(i32::try_from(nnz).unwrap_or(-1))
    );
    Ok((row_ptr, col_idx))
}

/// Solve one typed-driver RHS in native Complex64 precision. CUDA exposes its
/// sparse/reduction primitives natively for Complex64 even on devices where
/// the Complex32 compatibility path is host-staged.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn solve_gpu_coupled_rhs<
    B: sonium_compute::Backend,
    P: sonium_compute::DevicePreconditioner<B, Complex64>,
>(
    pool: &sonium_compute::MemoryPool<B>,
    workspace: &mut sonium_compute::GmresWorkspace<B, Complex64>,
    matrix: &sonium_compute::PersistentDeviceCsr<B, Complex64>,
    preconditioner: &P,
    rhs: &[Complex64],
    rhs64: &mut B::Buffer<Complex64>,
    x64: &mut B::Buffer<Complex64>,
    config: &SolverConfig,
) -> Result<(Vec<Complex64>, f64), SimulationError> {
    let backend = pool.backend();
    if !complex_values_are_finite(rhs) {
        return Err(SimulationError::new(
            "GPU typed-coupling RHS contains a non-finite value",
        ));
    }
    backend
        .try_copy_h2d(rhs, rhs64)
        .map_err(|error| SimulationError::new(format!("GPU coupled RHS upload failed: {error}")))?;
    let status = sonium_compute::gmres::gmres_backend_preconditioned_pooled_with_workspace(
        pool,
        workspace,
        matrix,
        preconditioner,
        rhs64,
        None,
        x64,
        &sonium_compute::GmresBackendConfig::<Complex64> {
            max_iterations: config.max_iterations,
            tolerance: (config.tolerance * 0.1).max(1.0e-12),
        },
    )
    .map_err(|error| SimulationError::new(format!("GPU coupled GMRES failed: {error}")))?;
    if !status.converged {
        return Err(SimulationError::new(format!(
            "GPU coupled GMRES did not converge: residual={:.3e}",
            status.final_residual
        )));
    }
    let mut solution = vec![Complex64::new(0.0, 0.0); rhs.len()];
    backend.try_copy_d2h(x64, &mut solution).map_err(|error| {
        SimulationError::new(format!("GPU coupled solution download failed: {error}"))
    })?;
    if !complex_values_are_finite(&solution) {
        return Err(SimulationError::new(
            "GPU typed-coupling solve produced a non-finite solution",
        ));
    }
    Ok((solution, status.final_residual))
}

/// Solve the supported linear typed-driver N-port on the same accelerator as
/// the BEM operator.  Thermal feedback, nonlinear motor laws, and mixed typed
/// / prescribed ports are admitted by the CPU product path but are rejected by
/// the CUDA capability contract before this function is called.
#[cfg(any(feature = "cuda", feature = "metal"))]
#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
fn solve_gpu_linear_driver_network<B: sonium_compute::Backend>(
    pool: &sonium_compute::MemoryPool<B>,
    workspace: &mut sonium_compute::GmresWorkspace<B, Complex64>,
    model: &LoudspeakerModel,
    frequency: f64,
    z_rad: &[Vec<Complex64>],
    config: &SolverConfig,
    matrix_slot: &mut Option<sonium_compute::PersistentDeviceCsr<B, Complex64>>,
    preconditioner_slot: &mut Option<sonium_compute::PointJacobiBackend<B, Complex64>>,
    rhs_slot: &mut Option<B::Buffer<Complex64>>,
    x_slot: &mut Option<B::Buffer<Complex64>>,
    matrix_value_updates: &mut usize,
    preconditioner_updates: &mut usize,
) -> Result<TypedDriverCoupling, SimulationError> {
    let backend = pool.backend();
    let drivers = model.drivers();
    let driver_count = drivers.len();
    if driver_count == 0
        || z_rad.len() != driver_count
        || z_rad.iter().any(|row| row.len() != driver_count)
        || drivers.iter().any(|driver| driver.driver_model.is_none())
    {
        return Err(SimulationError::new(
            "GPU typed coupling requires a matching all-typed N-port matrix",
        ));
    }
    let parameters: Vec<_> = drivers
        .iter()
        .map(|driver| {
            let driver_model = driver.driver_model.as_ref().ok_or_else(|| {
                SimulationError::new("GPU typed coupling requires every driver to be typed")
            })?;
            driver_model
                .network_parameters_at(frequency)
                .map_err(|error| {
                    SimulationError::new(format!(
                        "GPU driver network parameters failed at {frequency:.3} Hz: {error}"
                    ))
                })
        })
        .collect::<Result<_, _>>()?;
    let voltages = typed_driver_drive_voltages(model, frequency)?;
    let (row_ptr, col_idx) = dense_csr_structure(driver_count)?;
    let mut values = Vec::with_capacity(driver_count * driver_count);
    let mut rhs = Vec::with_capacity(driver_count);
    for row in 0..driver_count {
        let row_parameters = parameters[row];
        for column in 0..driver_count {
            values.push(if row == column {
                row_parameters.intrinsic_mechanical_impedance
                    + z_rad[row][column]
                    + Complex64::new(row_parameters.force_factor_tm.powi(2), 0.0)
                        / row_parameters.coil_impedance_ohm
            } else {
                z_rad[row][column]
            });
        }
        rhs.push(
            Complex64::new(row_parameters.force_factor_tm, 0.0) * voltages[row]
                / row_parameters.coil_impedance_ohm,
        );
    }
    if let Some(matrix) = matrix_slot.as_mut() {
        matrix
            .try_copy_values_from_host(backend, &values)
            .map_err(|error| {
                SimulationError::new(format!("GPU coupled CSR value update failed: {error}"))
            })?;
        *matrix_value_updates += 1;
    } else {
        let matrix = sonium_compute::DeviceCsr::try_from_host(
            backend,
            driver_count,
            driver_count,
            &row_ptr,
            &col_idx,
            &values,
        )
        .map_err(|error| {
            SimulationError::new(format!("GPU coupled CSR allocation failed: {error}"))
        })?;
        *matrix_slot = Some(
            sonium_compute::PersistentDeviceCsr::try_from_device_csr(backend, matrix).map_err(
                |error| SimulationError::new(format!("GPU coupled CSR prebake failed: {error}")),
            )?,
        );
    }
    let diagonal: Vec<Complex64> = (0..driver_count)
        .map(|row| values[row * driver_count + row])
        .collect();
    if let Some(preconditioner) = preconditioner_slot.as_mut() {
        preconditioner
            .try_update_diagonal(backend, &diagonal)
            .map_err(|error| {
                SimulationError::new(format!("GPU coupled Jacobi update failed: {error}"))
            })?;
        *preconditioner_updates += 1;
    } else {
        *preconditioner_slot = Some(
            sonium_compute::PointJacobiBackend::try_from_diagonal(backend, &diagonal).map_err(
                |error| {
                    SimulationError::new(format!("GPU coupled Jacobi allocation failed: {error}"))
                },
            )?,
        );
    }
    if rhs_slot.is_none() {
        *rhs_slot = Some(
            backend
                .raw_alloc::<Complex64>(driver_count)
                .map_err(|error| {
                    SimulationError::new(format!("GPU coupled RHS allocation failed: {error}"))
                })?,
        );
    }
    if x_slot.is_none() {
        *x_slot = Some(
            backend
                .raw_alloc::<Complex64>(driver_count)
                .map_err(|error| {
                    SimulationError::new(format!("GPU coupled solution allocation failed: {error}"))
                })?,
        );
    }
    let (velocities_rms, _) = solve_gpu_coupled_rhs(
        pool,
        workspace,
        matrix_slot
            .as_ref()
            .ok_or_else(|| SimulationError::new("GPU coupled matrix was not initialized"))?,
        preconditioner_slot.as_ref().ok_or_else(|| {
            SimulationError::new("GPU coupled preconditioner was not initialized")
        })?,
        &rhs,
        rhs_slot
            .as_mut()
            .ok_or_else(|| SimulationError::new("GPU coupled RHS was not initialized"))?,
        x_slot
            .as_mut()
            .ok_or_else(|| SimulationError::new("GPU coupled solution was not initialized"))?,
        config,
    )?;
    let electrical_impedances: Vec<Option<Complex64>> = parameters
        .iter()
        .zip(&voltages)
        .zip(&velocities_rms)
        .map(|((parameters, voltage), velocity)| {
            let current = (*voltage - Complex64::new(parameters.force_factor_tm, 0.0) * *velocity)
                / parameters.coil_impedance_ohm;
            (current.norm() > f64::MIN_POSITIVE).then_some(*voltage / current)
        })
        .collect();
    if electrical_impedances.iter().any(Option::is_none) {
        return Err(SimulationError::new(
            "GPU typed coupling produced a vanished voice-coil current",
        ));
    }
    let radiation_diagnostics = diagnose_radiation_matrix_generic(frequency, z_rad)?;
    let mut supplied = 0.0;
    let mut copper = 0.0;
    let mut magnetic = 0.0;
    let mut mechanical = 0.0;
    for ((parameters, driver), (voltage, velocity)) in parameters
        .iter()
        .zip(drivers)
        .zip(voltages.iter().zip(&velocities_rms))
    {
        let current = (*voltage - Complex64::new(parameters.force_factor_tm, 0.0) * *velocity)
            / parameters.coil_impedance_ohm;
        let copper_resistance = driver
            .driver_model
            .as_ref()
            .ok_or_else(|| {
                SimulationError::new("GPU typed coupling requires every driver to be typed")
            })?
            .thiele_small
            .voice_coil_resistance_ohm;
        supplied += (*voltage * current.conj()).re;
        copper += copper_resistance * current.norm_sqr();
        magnetic += (parameters.coil_impedance_ohm.re - copper_resistance) * current.norm_sqr();
        mechanical += parameters.intrinsic_mechanical_impedance.re * velocity.norm_sqr();
    }
    let radiation_power = radiation_matrix_power_generic(z_rad, &velocities_rms);
    let power_ledger = PowerLedger::from_components(
        supplied,
        copper,
        magnetic,
        mechanical,
        0.0,
        0.0,
        radiation_power,
        0.0,
    )
    .map_err(|error| SimulationError::new(format!("GPU coupled power ledger failed: {error}")))?;
    power_ledger
        .close(PowerLedgerTolerance {
            absolute_w: 1.0e-6,
            relative: 1.0e-4,
        })
        .map_err(|error| {
            SimulationError::new(format!("GPU coupled power ledger did not close: {error}"))
        })?;
    let velocities_bem_all: Vec<_> = velocities_rms
        .iter()
        .copied()
        .map(modeling_rms_to_bem_peak)
        .collect();
    let mut velocities_bem = [Complex64::default(); 2];
    for (index, velocity) in velocities_bem_all.iter().copied().take(2).enumerate() {
        velocities_bem[index] = velocity;
    }
    let mut radiation_matrix = [[Complex64::default(); 2]; 2];
    for (row, values) in z_rad.iter().take(2).enumerate() {
        for (column, value) in values.iter().copied().take(2).enumerate() {
            radiation_matrix[row][column] = value;
        }
    }
    Ok(TypedDriverCoupling {
        velocities_bem,
        velocities_bem_all,
        radiation_matrix,
        radiation_matrix_all: z_rad.to_vec(),
        radiation_matrix_diagnostics: radiation_diagnostics,
        thermal_driver_states: [None, None],
        driver_thermal_states: vec![None; driver_count],
        power_ledger: Some(power_ledger),
        woofer_electrical_impedance_ohm: electrical_impedances.first().copied().flatten(),
        tweeter_electrical_impedance_ohm: electrical_impedances.get(1).copied().flatten(),
        driver_electrical_impedances_ohm: electrical_impedances,
    })
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn evaluate_speaker_field_gpu<B: sonium_compute::Backend>(
    backend: &B,
    geometry: &B::Buffer<sonium_compute::BemFieldElementC64>,
    pressure: &mut B::Buffer<Complex64>,
    velocity: &mut B::Buffer<Complex64>,
    points: &B::Buffer<sonium_compute::BemFieldPointC64>,
    output: &mut B::Buffer<Complex64>,
    surface_pressure: &[Complex64],
    surface_velocity: &[Complex64],
    physics: &PhysicsParams,
) -> Result<Vec<Complex64>, SimulationError> {
    if geometry.is_empty()
        || surface_pressure.len() != geometry.len()
        || surface_velocity.len() != geometry.len()
        || points.len() != output.len()
    {
        return Err(SimulationError::new(
            "GPU speaker field evaluation received inconsistent resident geometry",
        ));
    }
    backend
        .try_copy_h2d(surface_pressure, pressure)
        .map_err(|error| {
            SimulationError::new(format!("GPU field pressure update failed: {error}"))
        })?;
    backend
        .try_copy_h2d(surface_velocity, velocity)
        .map_err(|error| {
            SimulationError::new(format!("GPU field velocity update failed: {error}"))
        })?;
    backend
        .try_bem_field_c64_device(
            geometry,
            pressure,
            velocity,
            points,
            output,
            physics.wave_number,
            physics.harmonic_factor,
            physics.pressure_factor,
        )
        .map_err(|error| SimulationError::new(format!("GPU field evaluation failed: {error}")))?;
    backend.try_synchronize().map_err(|error| {
        SimulationError::new(format!("GPU field synchronization failed: {error}"))
    })?;
    let mut host_output = vec![Complex64::default(); output.len()];
    backend
        .try_copy_d2h(output, &mut host_output)
        .map_err(|error| {
            SimulationError::new(format!("GPU field output download failed: {error}"))
        })?;
    if host_output
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(SimulationError::new(
            "GPU field evaluation produced a non-finite result",
        ));
    }
    Ok(host_output)
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn build_speaker_field_geometry(
    elements: &[Element],
    nodes: &[[f64; 3]],
) -> Result<Vec<sonium_compute::BemFieldElementC64>, SimulationError> {
    let boundary_elements: Vec<&Element> = elements
        .iter()
        .filter(|element| !element.property.is_evaluation())
        .collect();
    boundary_elements
        .into_iter()
        .enumerate()
        .map(|(index, element)| {
            let [n0, n1, n2] = element.connectivity.as_slice() else {
                return Err(SimulationError::new(format!(
                    "GPU speaker field evaluation requires triangular elements (element {index})"
                )));
            };
            Ok(sonium_compute::BemFieldElementC64::from_triangle(
                nodes[*n0],
                nodes[*n1],
                nodes[*n2],
                Complex64::default(),
                Complex64::default(),
            ))
        })
        .collect()
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn gpu_geometry_key(nodes: &[[f64; 3]], elements: &[Element], eval_points: &[[f64; 3]]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for node in nodes {
        for coordinate in node {
            coordinate.to_bits().hash(&mut hasher);
        }
    }
    for element in elements {
        element.connectivity.hash(&mut hasher);
        element.element_type.num_nodes().hash(&mut hasher);
        element.property.is_evaluation().hash(&mut hasher);
        for value in element
            .normal
            .iter()
            .chain(element.center.iter())
            .chain(std::iter::once(&element.area))
        {
            value.to_bits().hash(&mut hasher);
        }
        for normal in &element.node_normals {
            for value in normal {
                value.to_bits().hash(&mut hasher);
            }
        }
        element.group.hash(&mut hasher);
        element.dof_addresses.hash(&mut hasher);
    }
    for point in eval_points {
        for coordinate in point {
            coordinate.to_bits().hash(&mut hasher);
        }
    }
    hasher.finish()
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn gpu_frequency_grid_key(frequencies: &[f64]) -> u64 {
    let mut hasher = DefaultHasher::new();
    frequencies.len().hash(&mut hasher);
    for frequency in frequencies {
        frequency.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

/// Run the BEM simulation with custom solver configuration, returning solver
/// failures instead of accepting unconverged acoustic fields.
pub fn try_run_simulation_with_config(
    model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
    solver_config: &SolverConfig,
) -> Result<SimulationResult, SimulationError> {
    try_run_simulation_with_config_inner(model, eval_points, solver_config, None, None, None)
}

pub(crate) fn run_simulation_with_warm_start(
    model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
    solver_config: &SolverConfig,
    warm_start: &mut SimulationWarmStart,
) -> Result<SimulationResult, SimulationError> {
    try_run_simulation_with_config_inner(
        model,
        eval_points,
        solver_config,
        Some(warm_start),
        None,
        None,
    )
}

pub(crate) fn run_simulation_with_warm_start_and_topology(
    model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
    solver_config: &SolverConfig,
    warm_start: &mut SimulationWarmStart,
    topology: &PinnedWaveguideMesh,
) -> Result<SimulationResult, SimulationError> {
    try_run_simulation_with_config_inner(
        model,
        eval_points,
        solver_config,
        Some(warm_start),
        Some(topology),
        None,
    )
}

/// Dense-operator entries per thread above which the frequency sweep fans
/// out over rayon lanes. One frequency point assembles an N×N operator plus
/// preconditioner and an iterative solve, so the grain keeps lane fan-out
/// (pool dispatch plus per-lane operator/ILU/subspace state) amortized;
/// small sweeps stay serial. Starting point, not a calibrated value: tune
/// with benches on the target machine.
pub(crate) const FREQUENCY_SWEEP_ENTRIES_PER_THREAD: usize = 65_536;

/// Machine-adaptive gate for frequency-lane fan-out.
///
/// A fixed frequency count cannot be right on every machine: the point
/// where lane fan-out pays off moves with thread count, clock speed, and
/// Process-wide snapshot of the frequency-lane gate switches.
///
/// `std::env::var` allocates on every call; the switches are therefore read
/// once (first gate evaluation) and cached. Changing them mid-process has
/// no effect, matching the workspace `SONIUM_*` convention.
struct SweepParGate {
    force_serial: bool,
    threshold_override: Option<usize>,
}

static SWEEP_PAR_GATE: std::sync::OnceLock<SweepParGate> = std::sync::OnceLock::new();

fn sweep_par_gate() -> &'static SweepParGate {
    SWEEP_PAR_GATE.get_or_init(|| SweepParGate {
        force_serial: std::env::var("SONIUM_SERIAL")
            .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false),
        threshold_override: std::env::var("SONIUM_SPEAKER_PAR_THRESHOLD")
            .ok()
            .and_then(|value| value.parse::<usize>().ok()),
    })
}

/// Pure lane-gate decision over explicit configuration: fan out when the
/// per-point cost (dense-operator entries, `n_freqs * n_dofs²`) reaches
/// `entries_per_thread * threads`. Hermetic under parallel test threads
/// (no process env, no init-order dependence).
pub fn use_frequency_lanes_with(
    n_freqs: usize,
    n_dofs: usize,
    force_serial: bool,
    threshold_override: Option<usize>,
    threads: usize,
) -> bool {
    let work = n_freqs.saturating_mul(n_dofs).saturating_mul(n_dofs);
    if force_serial {
        return false;
    }
    if let Some(threshold) = threshold_override {
        return work >= threshold;
    }
    if threads < 2 {
        return false;
    }
    work >= FREQUENCY_SWEEP_ENTRIES_PER_THREAD
        .saturating_mul(threads)
        .max(1)
}

/// Whether a frequency sweep fans out across rayon lanes (see
/// [`use_frequency_lanes_with`]). Allocation-free: environment
/// configuration (`SONIUM_SPEAKER_PAR_THRESHOLD` absolute override,
/// `SONIUM_SERIAL=1` process-wide hatch) is read once into
/// [`SWEEP_PAR_GATE`]. Single-threaded processes always run serially.
pub fn use_frequency_lanes(n_freqs: usize, n_dofs: usize) -> bool {
    let gate = sweep_par_gate();
    use_frequency_lanes_with(
        n_freqs,
        n_dofs,
        gate.force_serial,
        gate.threshold_override,
        rayon::current_num_threads(),
    )
}

fn try_run_simulation_with_config_inner(
    model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
    solver_config: &SolverConfig,
    mut warm_start: Option<&mut SimulationWarmStart>,
    pinned_topology: Option<&PinnedWaveguideMesh>,
    control: Option<&SimulationControl>,
) -> Result<SimulationResult, SimulationError> {
    let lowered_model = model
        .with_product_design_geometry()
        .map_err(|message| SimulationError::new(format!("invalid product geometry: {message}")))?;
    let model = &lowered_model;
    model
        .validate()
        .map_err(|message| SimulationError::new(format!("invalid loudspeaker model: {message}")))?;
    validate_evaluation_points(eval_points)?;
    validate_solver_config(solver_config)?;

    let frequencies = model.simulation.frequencies();
    if let Some(control) = control {
        control.check()?;
    }
    let c = model.simulation.speed_of_sound;
    let rho = model.simulation.air_density;

    // Build BEM mesh from speaker mesh
    let speaker_mesh = match pinned_topology {
        Some(topology) => topology.generate(model).map_err(|message| {
            SimulationError::new(format!("pinned candidate mesh failed: {message}"))
        })?,
        None => super::mesh_gen::generate_loudspeaker_mesh(model),
    };
    let (nodes, mut elements) = speaker_mesh_to_bem(&speaker_mesh, model)?;

    let n_eval = eval_points.len();

    let n_dofs = elements.len();
    let strategy = select_solver_strategy(n_dofs, solver_config);
    let use_hmatrix = strategy == SolverStrategy::HMatrixIterative;
    let solver_label = match strategy {
        SolverStrategy::Direct => "LU",
        SolverStrategy::DenseIterative => "FGMRES+ILU",
        SolverStrategy::HMatrixIterative => "Hmatrix+FGMRES",
    };
    eprintln!("{n_dofs} DOFs, solver: {solver_label}");

    // Pre-build octree and clusters for H-matrix path (geometry is fixed)
    let hmatrix_clusters: Option<Vec<Cluster>> = if use_hmatrix {
        let centers: Vec<[f64; 3]> = elements.iter().map(|e| e.center).collect();
        let mut octree = Octree::build(&centers, 32, 8);
        octree.compute_interaction_lists(2.0);
        Some(clusters_from_octree(&elements, &octree))
    } else {
        None
    };

    let n_freqs = frequencies.len();
    let design_point_guesses = warm_start
        .as_ref()
        .map(|warm_start| warm_start.surface_pressure_by_frequency.clone())
        .unwrap_or_default();
    let mut all_spl = HostMatrix::<f64>::zeros(n_freqs, n_eval);
    let mut all_phase = HostMatrix::<f64>::zeros(n_freqs, n_eval);
    let mut all_complex = HostMatrix::<Complex64>::zeros(n_freqs, n_eval);
    let mut woofer_electrical_impedance_ohm = vec![None; n_freqs];
    let mut tweeter_electrical_impedance_ohm = vec![None; n_freqs];
    let mut driver_electrical_impedances_ohm = vec![vec![None; model.drivers().len()]; n_freqs];
    let mut radiation_matrices = vec![None; n_freqs];
    let mut radiation_matrix_diagnostics = vec![None; n_freqs];
    let mut thermal_driver_states = vec![[None, None]; n_freqs];
    let mut driver_thermal_states = vec![vec![None; model.drivers().len()]; n_freqs];
    let mut power_ledgers = vec![None; n_freqs];
    let mut structural_responses = vec![None; n_freqs];
    let mut driver_velocities_bem =
        vec![vec![Complex64::default(); model.drivers().len()]; n_freqs];
    let mut prev_solution: Option<Vec<Complex64>> = None;
    let mut prev_frequency: Option<f64> = None;
    let mut gcro_dr_subspace = (solver_config.recycle_subspace_size > 0)
        .then(|| GcroDrSubspace::<Complex64>::new(solver_config.recycle_subspace_size));
    let mut sentinel = ExactResidualSentinel::default();
    let mut exact_residual_evaluations = 0usize;

    let total_start = Instant::now();

    if !solver_config.parallel_frequency_sweep {
        for (fi, &freq) in frequencies.iter().enumerate() {
            if let Some(control) = control {
                control.check()?;
            }
            let freq_start = Instant::now();
            let exact_residuals_before = sentinel.exact_checks();

            // 1. Seed the BEM boundary condition from the modeling handoff (or
            // legacy product configuration).
            let driver_velocities = model.velocities_at(freq);
            update_driver_bcs(
                &mut elements,
                &speaker_mesh,
                model,
                freq,
                Some(&driver_velocities),
            )?;

            // 2. Assemble the frequency-dependent BEM operator once. Iterations
            // below change only the prescribed velocity RHS.
            let physics = PhysicsParams::new(freq, c, rho, false);
            let dense_matrix = if use_hmatrix {
                None
            } else {
                Some(
                    build_tbem_system_stabilized(
                        &elements,
                        &nodes,
                        &physics,
                        EXTERIOR_TBEM_SIGN,
                        solver_config.beta_scale,
                    )
                    .matrix,
                )
            };
            exact_residual_evaluations += sentinel.exact_checks() - exact_residuals_before;
            let hmatrix = if use_hmatrix {
                let clusters = hmatrix_clusters.as_ref().ok_or_else(|| {
                    SimulationError::new("H-matrix solver selected without geometry clusters")
                })?;
                Some(build_simulation_hmatrix(
                    &elements,
                    &nodes,
                    clusters,
                    &physics,
                    solver_config,
                ))
            } else {
                None
            };

            // 3. Solve, coupling typed driver models to the BEM through the
            // full two-port radiation matrix. Impedance ratios are
            // amplitude-invariant, so BEM peak phasors can load the modeling
            // layer's RMS circuit equations.
            let recycle_initial_guess = should_recycle_initial_guess(
                prev_frequency,
                freq,
                solver_config.max_recycled_guess_octaves,
            );
            if !recycle_initial_guess && let Some(subspace) = gcro_dr_subspace.as_mut() {
                reset_recycling_for_frequency_step(
                    prev_frequency,
                    freq,
                    solver_config.max_recycled_guess_octaves,
                    subspace,
                );
            }
            let iteration_guess = if recycle_initial_guess {
                prev_solution.clone()
            } else {
                // Same cross-design guard as the parallel sweep: drop guesses
                // whose length no longer matches the current mesh DOF count.
                design_point_guesses
                    .get(fi)
                    .and_then(|guess| guess.as_deref())
                    .filter(|guess| guess.len() == n_dofs)
                    .map(<[Complex64]>::to_vec)
            };
            let has_typed_driver = model
                .drivers()
                .iter()
                .any(|driver| driver.driver_model.is_some());
            let structural_max_iterations = model
                .product_design
                .as_ref()
                .and_then(|product| product.structural_acoustic.as_ref())
                .map_or(1, |feedback| feedback.max_iterations);
            let structural_tolerance = model
                .product_design
                .as_ref()
                .and_then(|product| product.structural_acoustic.as_ref())
                .map(|feedback| feedback.velocity_tolerance);
            let structural_relaxation = model
                .product_design
                .as_ref()
                .and_then(|product| product.structural_acoustic.as_ref())
                .map(|feedback| feedback.under_relaxation);
            let mut structural_velocity: Option<Vec<Complex64>> = None;
            let mut structural_response = None;
            let mut surface_pressure = None;
            let mut typed_coupling = None;
            for feedback_iteration in 0..structural_max_iterations {
                let (pressure, coupling) = if has_typed_driver {
                    // The structural velocity is exchanged for every unit-load
                    // column as well as the final network state, so cabinet
                    // radiation changes the full N-port matrix.
                    let (pressure, coupling) = solve_typed_driver_coupling_with_structure_control(
                        &mut elements,
                        &speaker_mesh,
                        model,
                        freq,
                        &nodes,
                        &physics,
                        dense_matrix.as_ref(),
                        hmatrix.as_ref(),
                        solver_config,
                        &mut gcro_dr_subspace,
                        None,
                        iteration_guess.as_deref(),
                        &mut sentinel,
                        structural_velocity.as_deref(),
                        control,
                    )?;
                    (pressure, Some(coupling))
                } else {
                    let pressure = solve_bem_with_driver_velocities_and_structure_control(
                        &mut elements,
                        &speaker_mesh,
                        model,
                        freq,
                        &driver_velocities,
                        &nodes,
                        &physics,
                        dense_matrix.as_ref(),
                        hmatrix.as_ref(),
                        solver_config,
                        &mut gcro_dr_subspace,
                        None,
                        iteration_guess.as_deref(),
                        &mut sentinel,
                        structural_velocity.as_deref(),
                        control,
                    )?;
                    (pressure, None)
                };
                let feedback = structural_cabinet_velocity_from_pressure(
                    model,
                    freq,
                    &elements,
                    &speaker_mesh,
                    &pressure,
                    structural_velocity.as_deref(),
                )?;
                surface_pressure = Some(pressure);
                typed_coupling = coupling;
                let Some((next_velocity, response, change)) = feedback else {
                    break;
                };
                structural_response = Some(response);
                let relaxation = structural_relaxation.unwrap_or(1.0);
                let relaxed_velocity = if let Some(previous) = structural_velocity.as_ref() {
                    previous
                        .iter()
                        .zip(&next_velocity)
                        .map(|(old, new)| *old + relaxation * (*new - *old))
                        .collect()
                } else {
                    next_velocity
                };
                structural_velocity = Some(relaxed_velocity);
                if change < structural_tolerance.unwrap_or(f64::INFINITY) {
                    break;
                }
                if feedback_iteration + 1 == structural_max_iterations {
                    return Err(SimulationError::new(format!(
                        "structural acoustic feedback did not converge at {freq:.3} Hz after {} iterations (relative velocity change {change:.3e})",
                        structural_max_iterations
                    )));
                }
            }
            let surface_pressure = surface_pressure.ok_or_else(|| {
                SimulationError::new("structural/BEM frequency solve produced no pressure")
            })?;
            if let Some(coupling) = typed_coupling {
                woofer_electrical_impedance_ohm[fi] = coupling.woofer_electrical_impedance_ohm;
                tweeter_electrical_impedance_ohm[fi] = coupling.tweeter_electrical_impedance_ohm;
                driver_electrical_impedances_ohm[fi] = coupling.driver_electrical_impedances_ohm;
                radiation_matrices[fi] = Some(coupling.radiation_matrix_all.clone());
                radiation_matrix_diagnostics[fi] = Some(coupling.radiation_matrix_diagnostics);
                thermal_driver_states[fi] = coupling.thermal_driver_states;
                driver_thermal_states[fi] = coupling.driver_thermal_states;
                power_ledgers[fi] = coupling.power_ledger;
                driver_velocities_bem[fi] = coupling.velocities_bem_all.clone();
            }
            if !has_typed_driver {
                driver_velocities_bem[fi] = driver_velocities;
            }
            structural_responses[fi] = structural_response;

            // 4. Evaluate field at measurement points
            let surface_velocity = build_surface_velocity(&elements);

            let p_scattered = compute_scattered_field(
                eval_points,
                &elements,
                &nodes,
                &surface_pressure,
                Some(&surface_velocity),
                &physics,
                BasisOrder::Constant,
            );
            if !complex_values_are_finite(&p_scattered) {
                return Err(SimulationError::new(format!(
                    "field evaluation produced non-finite pressure at {freq:.3} Hz"
                )));
            }
            all_complex.row_mut(fi).copy_from_slice(&p_scattered);

            // The BEM solves for complex pressure amplitudes (peak); dB SPL is
            // referenced to RMS pressure, so use the canonical helper that
            // divides by √2 before the log conversion.
            {
                let spl_row = all_spl.row_mut(fi);
                let phase_row = all_phase.row_mut(fi);
                for i in 0..n_eval {
                    let p = p_scattered[i];
                    spl_row[i] = sonium_xem::pressure_to_spl(p);
                    phase_row[i] = p.arg();
                    if !spl_row[i].is_finite() || !phase_row[i].is_finite() {
                        return Err(SimulationError::new(format!(
                            "field evaluation produced non-finite SPL/phase at {freq:.3} Hz, point {i}"
                        )));
                    }
                }
            }

            let elapsed = freq_start.elapsed().as_secs_f64();
            eprintln!(
                "[{}/{}] f={:.0} Hz, {:.2}s",
                fi + 1,
                frequencies.len(),
                freq,
                elapsed,
            );

            if let Some(warm_start) = warm_start.as_deref_mut() {
                if warm_start.surface_pressure_by_frequency.len() < n_freqs {
                    warm_start
                        .surface_pressure_by_frequency
                        .resize_with(n_freqs, || None);
                }
                warm_start.surface_pressure_by_frequency[fi] = Some(surface_pressure.clone());
            }
            prev_solution = Some(surface_pressure);
            prev_frequency = Some(freq);
            if let Some(control) = control {
                control.report_frequency(fi + 1, freq);
            }
        }
    } else {
        let solve_at = |state: &mut FrequencySweepState,
                        index: usize,
                        frequency: f64,
                        control: Option<&SimulationControl>| {
            solve_frequency_point(
                index,
                n_freqs,
                frequency,
                state,
                &speaker_mesh,
                &nodes,
                hmatrix_clusters.as_deref(),
                model,
                eval_points,
                solver_config,
                use_hmatrix,
                design_point_guesses
                    .get(index)
                    .and_then(|guess| guess.as_deref()),
                control,
            )
        };
        // Contiguous ascending frequency lanes: each lane keeps one
        // FrequencySweepState, so warm starts, GCRO-DR recycling, and ILU
        // reuse compound along the lane instead of being scattered by
        // work-stealing chunk reassignments.
        let thread_budget = solver_config
            .max_frequency_lanes
            .map_or_else(rayon::current_num_threads, |max| {
                max.min(rayon::current_num_threads())
            });
        let memory_budget_bytes = host_memory_bytes().map(|host_bytes| {
            (host_bytes as f64 * solver_config.frequency_sweep_memory_fraction) as u64
        });
        // Small-problem fast path: skip lane fan-out (and per-lane
        // operator/ILU/subspace state) when the sweep is below the
        // machine-adaptive gate. `max_frequency_lanes: None` already
        // derives the lane cap from the thread and memory budgets inside
        // `plan_lanes`; this gate decides whether lanes are worthwhile
        // at all.
        let plan = try_plan_lanes(
            n_freqs,
            |_| estimate_frequency_live_bytes(n_dofs, use_hmatrix, solver_config),
            &LaneConfig {
                max_lanes: if use_frequency_lanes(n_freqs, n_dofs) {
                    solver_config.max_frequency_lanes
                } else {
                    // Small-problem fast path: cap to one lane (serial path
                    // below, no per-lane operator/ILU/subspace state).
                    Some(1)
                },
                thread_budget,
                memory_budget_bytes,
            },
        )
        .map_err(|error| SimulationError::new(format!("frequency lane admission: {error}")))?;
        eprintln!(
            "Frequency sweep: {} lane(s) over {} frequencies",
            plan.num_lanes, n_freqs
        );
        let outputs = if plan.num_lanes <= 1 {
            // Single lane: run serially on the current thread, no pool.
            let mut state = FrequencySweepState::new(&elements, solver_config);
            frequencies
                .iter()
                .enumerate()
                .map(|(index, &frequency)| {
                    let output = solve_at(&mut state, index, frequency, control)?;
                    if let Some(control) = control {
                        control.complete_frequency(frequency);
                    }
                    Ok(output)
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            let cancellation = control.map(SimulationControl::cancellation_flag);
            if rayon::current_thread_index().is_some() {
                // Use the caller's Rayon pool. The shared lane runner chooses
                // serial execution for nested calls to avoid creating a new
                // pool; that would discard this explicit thread budget.
                use rayon::prelude::*;
                let lanes: Vec<_> = plan
                    .chunks
                    .par_iter()
                    .map(|range| {
                        let mut state = FrequencySweepState::new(&elements, solver_config);
                        let mut outputs = Vec::with_capacity(range.len());
                        for index in range.clone() {
                            if cancellation
                                .as_ref()
                                .is_some_and(|flag| flag.load(Ordering::Relaxed))
                            {
                                break;
                            }
                            let output = solve_at(&mut state, index, frequencies[index], control);
                            if output.is_ok()
                                && let Some(control) = control
                            {
                                control.complete_frequency(frequencies[index]);
                            }
                            outputs.push((index, output));
                        }
                        outputs
                    })
                    .collect();
                let mut ordered: Vec<_> = (0..n_freqs).map(|_| None).collect();
                for (index, output) in lanes.into_iter().flatten() {
                    ordered[index] = Some(output);
                }
                if ordered.iter().any(Option::is_none) {
                    return Err(SimulationError::cancelled());
                }
                ordered
                    .into_iter()
                    .map(|output| output.expect("completed lane covers its samples"))
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                // Per-lane worker budgets: each lane's assembly/solve runs
                // in a nested pool of budget/num_lanes threads instead of
                // fanning out to the global pool, so lanes x workers stays
                // within budget past the bandwidth knee (SIM-PERF-20261013.md
                // section 6, recommendation 4).
                let workers_per_lane = (thread_budget / plan.num_lanes).max(1);
                let outcome = run_lanes_with_worker_budgets(
                    &frequencies,
                    &plan,
                    workers_per_lane,
                    || FrequencySweepState::new(&elements, solver_config),
                    &|state, index, &frequency| {
                        let output = solve_at(state, index, frequency, control)?;
                        if let Some(control) = control {
                            control.complete_frequency(frequency);
                        }
                        Ok(output)
                    },
                    cancellation,
                );
                if outcome.cancelled {
                    return Err(SimulationError::cancelled());
                }
                outcome
                    .results
                    .into_iter()
                    .map(|result| result.expect("uncancelled lane sweep covers every frequency"))
                    .collect::<Result<Vec<_>, _>>()?
            }
        };
        for (frequency_index, output) in outputs.into_iter().enumerate() {
            all_spl
                .row_mut(frequency_index)
                .copy_from_slice(&output.spl);
            all_phase
                .row_mut(frequency_index)
                .copy_from_slice(&output.phase);
            all_complex
                .row_mut(frequency_index)
                .copy_from_slice(&output.pressure);
            woofer_electrical_impedance_ohm[frequency_index] =
                output.woofer_electrical_impedance_ohm;
            tweeter_electrical_impedance_ohm[frequency_index] =
                output.tweeter_electrical_impedance_ohm;
            driver_electrical_impedances_ohm[frequency_index] =
                output.driver_electrical_impedances_ohm;
            radiation_matrices[frequency_index] = output.radiation_matrix;
            radiation_matrix_diagnostics[frequency_index] = output.radiation_matrix_diagnostics;
            thermal_driver_states[frequency_index] = output.thermal_driver_states;
            driver_thermal_states[frequency_index] = output.driver_thermal_states;
            power_ledgers[frequency_index] = output.power_ledger;
            structural_responses[frequency_index] = output.structural_response;
            driver_velocities_bem[frequency_index] = output.driver_velocities_bem;
            exact_residual_evaluations += output.exact_residual_evaluations;
            if let Some(warm_start) = warm_start.as_deref_mut() {
                if warm_start.surface_pressure_by_frequency.len() < n_freqs {
                    warm_start
                        .surface_pressure_by_frequency
                        .resize_with(n_freqs, || None);
                }
                warm_start.surface_pressure_by_frequency[frequency_index] =
                    Some(output.surface_pressure);
            }
        }
    }

    if let Some(control) = control {
        control.check()?;
    }
    let total = total_start.elapsed().as_secs_f64();
    eprintln!(
        "Simulation complete: {} frequencies in {:.1}s",
        frequencies.len(),
        total
    );
    let frequency_count = frequencies.len();

    // Compat fill of the deprecated per-driver vectors (the driver matrix
    // below is canonical).
    #[allow(deprecated)]
    let result = SimulationResult {
        frequencies,
        spl: all_spl,
        phase: all_phase,
        complex_pressure: all_complex,
        woofer_electrical_impedance_ohm,
        tweeter_electrical_impedance_ohm,
        driver_electrical_impedances_ohm,
        radiation_matrices,
        radiation_matrix_diagnostics,
        thermal_driver_states,
        driver_thermal_states,
        power_ledgers,
        product_reference_sweep: None,
        structural_responses,
        driver_velocities_bem,
        exact_residual_evaluations,
        provenance: SimulationProvenance::solved(
            SimulationBackend::Cpu,
            solver_config.execution_profile,
            PrecisionPolicy::F64,
            ResidualGateReport::passed(solver_config.tolerance),
            frequency_count,
            n_eval,
        ),
    };
    Ok(result)
}

/// Build a flat cluster list from an octree of element centroids.
pub(crate) fn clusters_from_octree(elements: &[Element], octree: &Octree) -> Vec<Cluster> {
    let leaves = octree.leaves();
    let mut clusters = Vec::with_capacity(leaves.len());
    let leaf_to_cluster: std::collections::HashMap<usize, usize> = leaves
        .iter()
        .filter(|&&i| !octree.nodes[i].element_indices.is_empty())
        .enumerate()
        .map(|(cluster_idx, &leaf_idx)| (leaf_idx, cluster_idx))
        .collect();

    for &leaf_idx in &leaves {
        let node = &octree.nodes[leaf_idx];
        if node.element_indices.is_empty() {
            continue;
        }
        let mut cluster = Cluster::new(node.center);
        cluster.element_indices = node.element_indices.clone();
        cluster.num_elements = node.element_indices.len();
        cluster.element_property = ElementProperty::Surface;
        cluster.radius = node.radius();
        cluster.level = node.level;
        cluster.num_dofs = node
            .element_indices
            .iter()
            .filter(|&&i| !elements[i].property.is_evaluation())
            .count();
        cluster.dofs_per_element = 1;
        clusters.push(cluster);
    }

    for (cluster_idx, &leaf_idx) in leaves.iter().enumerate() {
        if octree.nodes[leaf_idx].element_indices.is_empty() {
            continue;
        }
        let octree_node = &octree.nodes[leaf_idx];
        let near: Vec<usize> = octree_node
            .near_clusters
            .iter()
            .filter_map(|&n| leaf_to_cluster.get(&n).copied())
            .collect();
        let far: Vec<usize> = octree_node
            .far_clusters
            .iter()
            .filter_map(|&f| leaf_to_cluster.get(&f).copied())
            .collect();
        if let Some(c) = clusters.get_mut(cluster_idx) {
            c.near_clusters = near;
            c.far_clusters = far;
        }
    }

    clusters
}

fn should_recycle_initial_guess(
    previous_frequency: Option<f64>,
    frequency: f64,
    max_octave_step: f64,
) -> bool {
    let Some(previous_frequency) = previous_frequency else {
        return false;
    };
    if previous_frequency <= 0.0
        || frequency <= 0.0
        || !previous_frequency.is_finite()
        || !frequency.is_finite()
        || max_octave_step.is_nan()
        || max_octave_step < 0.0
    {
        return false;
    }

    (frequency / previous_frequency).log2().abs() <= max_octave_step + 1e-12
}

fn reset_recycling_for_frequency_step<T: sonium_solvers::traits::ComplexField>(
    previous_frequency: Option<f64>,
    frequency: f64,
    max_octave_step: f64,
    subspace: &mut GcroDrSubspace<T>,
) -> bool {
    if previous_frequency.is_some()
        && !should_recycle_initial_guess(previous_frequency, frequency, max_octave_step)
    {
        subspace.reset();
        true
    } else {
        false
    }
}

// ---------------------------------------------------------------------------
// Mesh conversion: SpeakerMesh → BEM Element/Node arrays
// ---------------------------------------------------------------------------

/// Convert a `SpeakerMesh` to the BEM solver's `Element` + `nodes` representation.
pub(crate) fn speaker_mesh_to_bem(
    mesh: &SpeakerMesh,
    model: &LoudspeakerModel,
) -> Result<(Vec<[f64; 3]>, Vec<Element>), SimulationError> {
    if mesh.region.len() != mesh.triangles.len()
        || mesh.node_normals.len() != mesh.nodes.len()
        || mesh
            .triangles
            .iter()
            .flatten()
            .any(|&node| node >= mesh.nodes.len())
    {
        return Err(SimulationError::new(
            "invalid speaker mesh connectivity or region/normal dimensions",
        ));
    }
    let c = model.simulation.speed_of_sound;
    let rho = model.simulation.air_density;
    let initial_frequency = model.simulation.freq_start;
    let y_cabinet = model
        .materials
        .cabinet
        .try_admittance_at(initial_frequency, rho, c)
        .map_err(SimulationError::new)?;
    let y_waveguide = model
        .materials
        .waveguide
        .try_admittance_at(initial_frequency, rho, c)
        .map_err(SimulationError::new)?;

    let nodes: Vec<[f64; 3]> = mesh.nodes.clone();

    let n_elem = mesh.triangles.len();
    let mut elements = Vec::with_capacity(n_elem);

    for (idx, (tri, &region)) in mesh.triangles.iter().zip(mesh.region.iter()).enumerate() {
        let bc = match region {
            MeshRegion::CabinetWall => BoundaryCondition::VelocityWithAdmittance {
                velocity: vec![Complex64::new(0.0, 0.0)],
                admittance: y_cabinet,
            },
            MeshRegion::Waveguide => BoundaryCondition::VelocityWithAdmittance {
                velocity: vec![Complex64::new(0.0, 0.0)],
                admittance: y_waveguide,
            },
            MeshRegion::WooferAperture => {
                // Initial placeholder — will be updated per frequency with normal projection
                BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)])
            }
            MeshRegion::TweeterAperture => {
                BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)])
            }
            MeshRegion::DriverAperture(_) => {
                BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)])
            }
            MeshRegion::PortAperture => BoundaryCondition::VelocityWithAdmittance {
                velocity: vec![Complex64::new(0.0, 0.0)],
                admittance: model
                    .port_admittance_at(initial_frequency)
                    .unwrap_or_default(),
            },
        };

        let mut elem = Element {
            connectivity: vec![tri[0], tri[1], tri[2]],
            element_type: ElementType::Tri3,
            property: ElementProperty::Surface,
            normal: [0.0; 3],
            node_normals: vec![[0.0; 3]; 3],
            center: [0.0; 3],
            area: 0.0,
            boundary_condition: bc,
            group: 0,
            dof_addresses: vec![idx],
        };

        compute_element_geometry(&mut elem, &nodes, &mesh.node_normals);
        elements.push(elem);
    }

    Ok((nodes, elements))
}

/// Recompute element center, area, and normal from node coordinates.
///
/// If `mesh_node_normals` provides non-zero normals for the element's nodes,
/// they are used for curved BEM integration (e.g. conical driver surfaces).
/// Zero normals indicate a flat surface and leave `elem.node_normals` as zeros,
/// which tells the BEM kernels to skip curved-normal interpolation.
fn compute_element_geometry(
    elem: &mut Element,
    nodes: &[[f64; 3]],
    mesh_node_normals: &[[f64; 3]],
) {
    let n0 = elem.connectivity[0];
    let n1 = elem.connectivity[1];
    let n2 = elem.connectivity[2];

    let p0 = nodes[n0];
    let p1 = nodes[n1];
    let p2 = nodes[n2];

    // Center
    for j in 0..3 {
        elem.center[j] = (p0[j] + p1[j] + p2[j]) / 3.0;
    }

    // Normal and area via cross product
    let v1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let v2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    let cross = [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ];
    let len = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
    elem.area = len / 2.0;
    if len > 1e-15 {
        elem.normal = [cross[0] / len, cross[1] / len, cross[2] / len];
    }

    // Set per-node normals from mesh data. Non-zero normals activate the BEM
    // kernel's curved-surface interpolation path; zeros mean flat (no overhead).
    for (local, &global) in [n0, n1, n2].iter().enumerate() {
        elem.node_normals[local] = mesh_node_normals[global];
    }
}

// ---------------------------------------------------------------------------
// Update driver BCs for a specific frequency
// ---------------------------------------------------------------------------

pub(crate) fn update_driver_bcs(
    elements: &mut [Element],
    speaker_mesh: &SpeakerMesh,
    model: &LoudspeakerModel,
    freq: f64,
    driver_velocities: Option<&[Complex64]>,
) -> Result<(), SimulationError> {
    update_driver_bcs_with_structure(elements, speaker_mesh, model, freq, driver_velocities, None)
}

/// Update acoustic boundary values while retaining a frequency-local
/// structural cabinet velocity exchange. The optional slice is indexed by
/// BEM element and contains normal velocity in the BEM peak convention.
fn update_driver_bcs_with_structure(
    elements: &mut [Element],
    speaker_mesh: &SpeakerMesh,
    model: &LoudspeakerModel,
    freq: f64,
    driver_velocities: Option<&[Complex64]>,
    structural_cabinet_velocity: Option<&[Complex64]>,
) -> Result<(), SimulationError> {
    let model_velocities = model.velocities_at(freq);
    if elements.len() != speaker_mesh.region.len()
        || speaker_mesh.region.iter().any(|region| {
            matches!(region,
            MeshRegion::DriverAperture(index) if *index >= model_velocities.len())
        })
    {
        return Err(SimulationError::new(
            "invalid speaker boundary or driver dimensions",
        ));
    }
    let velocity_for = |index: usize| {
        driver_velocities
            .and_then(|velocities| velocities.get(index).copied())
            .unwrap_or_else(|| model_velocities[index])
    };
    let y_cabinet = model
        .materials
        .cabinet
        .try_admittance_at(
            freq,
            model.simulation.air_density,
            model.simulation.speed_of_sound,
        )
        .map_err(SimulationError::new)?;
    let y_waveguide = model
        .materials
        .waveguide
        .try_admittance_at(
            freq,
            model.simulation.air_density,
            model.simulation.speed_of_sound,
        )
        .map_err(SimulationError::new)?;

    for (i, elem) in elements.iter_mut().enumerate() {
        match speaker_mesh.region[i] {
            MeshRegion::CabinetWall => {
                let velocity = structural_cabinet_velocity
                    .and_then(|velocities| velocities.get(i).copied())
                    .unwrap_or_default();
                elem.boundary_condition = BoundaryCondition::VelocityWithAdmittance {
                    velocity: vec![velocity],
                    admittance: y_cabinet,
                };
            }
            MeshRegion::Waveguide => {
                elem.boundary_condition = BoundaryCondition::VelocityWithAdmittance {
                    velocity: vec![Complex64::new(0.0, 0.0)],
                    admittance: y_waveguide,
                };
            }
            MeshRegion::WooferAperture => {
                // For conical drivers, project piston velocity onto the element normal.
                // The piston moves rigidly along z, so v_n = v_z * n_z.
                let n_z = elem.normal[2];
                elem.boundary_condition = BoundaryCondition::Velocity(vec![velocity_for(0) * n_z]);
            }
            MeshRegion::TweeterAperture => {
                let n_z = elem.normal[2];
                elem.boundary_condition = BoundaryCondition::Velocity(vec![velocity_for(1) * n_z]);
            }
            MeshRegion::DriverAperture(index) => {
                let n_z = elem.normal[2];
                elem.boundary_condition =
                    BoundaryCondition::Velocity(vec![velocity_for(index) * n_z]);
            }
            MeshRegion::PortAperture => {
                elem.boundary_condition = BoundaryCondition::VelocityWithAdmittance {
                    velocity: vec![Complex64::new(0.0, 0.0)],
                    admittance: model.port_admittance_at(freq).unwrap_or_default(),
                };
            }
        }
    }
    Ok(())
}

/// Condense the solved cabinet-wall pressure to the product structural model
/// and expand its normal velocity back to the BEM element layout.
///
/// BEM pressure is a peak `e^{-iωt}` phasor. The product structural contract
/// consumes RMS `e^{+iωt}` pressure, so the conversion is conjugation followed
/// by division by `√2`; the response conversion back to a BEM boundary is
/// performed by [`StructuralAcousticResponse::velocity_bem_peak`].
fn structural_cabinet_velocity_from_pressure(
    model: &LoudspeakerModel,
    frequency: f64,
    elements: &[Element],
    speaker_mesh: &SpeakerMesh,
    pressure: &[Complex64],
    previous_velocity: Option<&[Complex64]>,
) -> Result<Option<(Vec<Complex64>, StructuralAcousticResponse, f64)>, SimulationError> {
    let Some(product) = &model.product_design else {
        return Ok(None);
    };
    if product.structural_acoustic.is_none() {
        return Ok(None);
    }
    if pressure.len() < elements.len() {
        return Err(SimulationError::new(
            "structural feedback pressure layout does not match the BEM mesh",
        ));
    }
    let mut total_area = 0.0;
    let mut pressure_area_integral = Complex64::default();
    for (index, (element, region)) in elements.iter().zip(&speaker_mesh.region).enumerate() {
        if matches!(region, MeshRegion::CabinetWall) {
            total_area += element.area;
            pressure_area_integral += pressure[index] * element.area;
        }
    }
    if !total_area.is_finite() || total_area <= 0.0 {
        return Err(SimulationError::new(
            "structural feedback requires a positive cabinet-wall area",
        ));
    }
    let pressure_bem_peak = pressure_area_integral / total_area;
    let pressure_rms_modeling = pressure_bem_peak.conj() / std::f64::consts::SQRT_2;
    let response = product
        .structural_response_at(frequency, pressure_rms_modeling)
        .map_err(|error| {
            SimulationError::new(format!(
                "structural acoustic response failed at {frequency:.3} Hz: {error}"
            ))
        })?
        .ok_or_else(|| SimulationError::new("structural feedback unexpectedly disappeared"))?;
    let velocity = response.velocity_bem_peak();
    let mut expanded = vec![Complex64::default(); elements.len()];
    for (index, region) in speaker_mesh.region.iter().enumerate() {
        if matches!(region, MeshRegion::CabinetWall) {
            expanded[index] = velocity;
        }
    }
    let change = previous_velocity
        .map(|previous| {
            expanded
                .iter()
                .zip(previous)
                .map(|(current, previous)| (*current - *previous).norm())
                .fold(0.0, f64::max)
                / expanded
                    .iter()
                    .map(|value| value.norm())
                    .fold(0.0, f64::max)
                    .max(1.0e-15)
        })
        .unwrap_or(f64::INFINITY);
    Ok(Some((expanded, response, change)))
}

/// Axial radiation forces on (woofer, tweeter) for one BEM surface-pressure
/// solution, in the BEM e^{−iωt} peak convention.
fn integrate_driver_axial_forces(
    elements: &[Element],
    speaker_mesh: &SpeakerMesh,
    pressure: &[Complex64],
) -> Vec<Complex64> {
    let driver_count = speaker_mesh
        .region
        .iter()
        .filter_map(|region| match region {
            MeshRegion::WooferAperture => Some(0),
            MeshRegion::TweeterAperture => Some(1),
            MeshRegion::DriverAperture(index) => Some(*index),
            MeshRegion::CabinetWall | MeshRegion::Waveguide | MeshRegion::PortAperture => None,
        })
        .max()
        .map_or(2, |index| index + 1)
        .max(2);
    let mut forces = vec![Complex64::default(); driver_count];
    for ((element, region), pressure) in elements
        .iter()
        .zip(&speaker_mesh.region)
        .zip(pressure.iter().copied())
    {
        let driver = match region {
            MeshRegion::WooferAperture => Some(0),
            MeshRegion::TweeterAperture => Some(1),
            MeshRegion::CabinetWall | MeshRegion::Waveguide | MeshRegion::PortAperture => None,
            MeshRegion::DriverAperture(index) => Some(*index),
        };
        if let Some(driver) = driver {
            forces[driver] += pressure * element.area * element.normal[2];
        }
    }
    forces
}

/// Effective scalar mechanical load `force / axial velocity` per driver for
/// one simultaneous velocity state.
///
/// This folds mutual radiation into a drive-dependent scalar and is only
/// kept for diagnostics and regression tests. Production typed-driver
/// coupling uses the full complex 2×2 matrix from
/// [`solve_typed_driver_coupling`], which is drive-independent.
///
/// Convention bridge: sonium-bem phasors use e^{−iωt} (Green's function
/// exp(+ikr)/4πr), while sonium-sim-speaker-modeling uses e^{+iωt} (see its
/// `conventions.rs`). Converting phasors between the two conventions
/// conjugates them, and the impedance ratio transforms as
/// Z_model = conj(F_bem)/conj(v_bem) = conj(Z_bem). Without the conjugation a
/// mass-like radiation load arrives as R−iX instead of R+iX, so radiation mass
/// would be SUBTRACTED from M_ms instead of added.
#[cfg(test)]
fn driver_radiation_impedances(
    elements: &[Element],
    speaker_mesh: &SpeakerMesh,
    pressure: &[Complex64],
    axial_velocities: [Complex64; 2],
) -> [Option<Complex64>; 2] {
    let forces = integrate_driver_axial_forces(elements, speaker_mesh, pressure);
    [0, 1].map(|driver| {
        (axial_velocities[driver].norm() > 1.0e-15).then(|| {
            (forces.get(driver).copied().unwrap_or_default() / axial_velocities[driver]).conj()
        })
    })
}

/// Update the driver velocity BCs, rebuild the right-hand side, and solve
/// against the already-assembled operator. The operator depends on the BC
/// types but not on the prescribed velocity values, so it is assembled once
/// per frequency and reused for every right-hand side (unit loads and the
/// coupled state alike).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn solve_bem_with_driver_velocities(
    elements: &mut [Element],
    speaker_mesh: &SpeakerMesh,
    model: &LoudspeakerModel,
    frequency: f64,
    velocities: &[Complex64],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    dense_matrix: Option<&DenseMatrix<Complex64>>,
    hmatrix: Option<&HMatrix>,
    config: &SolverConfig,
    gcro_dr_subspace: &mut Option<GcroDrSubspace<Complex64>>,
    dense_preconditioner: Option<&IluPreconditioner<Complex64>>,
    guess: Option<&[Complex64]>,
    sentinel: &mut ExactResidualSentinel,
) -> Result<Vec<Complex64>, SimulationError> {
    solve_bem_with_driver_velocities_and_structure(
        elements,
        speaker_mesh,
        model,
        frequency,
        velocities,
        nodes,
        physics,
        dense_matrix,
        hmatrix,
        config,
        gcro_dr_subspace,
        dense_preconditioner,
        guess,
        sentinel,
        None,
    )
}

/// Solve the BEM RHS with a previously exchanged structural cabinet velocity.
///
/// The operator is assembled for the same boundary-condition types; the
/// structural velocity only changes the RHS, which lets the partitioned
/// product coupling reuse the expensive frequency operator and preconditioner.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn solve_bem_with_driver_velocities_and_structure(
    elements: &mut [Element],
    speaker_mesh: &SpeakerMesh,
    model: &LoudspeakerModel,
    frequency: f64,
    velocities: &[Complex64],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    dense_matrix: Option<&DenseMatrix<Complex64>>,
    hmatrix: Option<&HMatrix>,
    config: &SolverConfig,
    gcro_dr_subspace: &mut Option<GcroDrSubspace<Complex64>>,
    dense_preconditioner: Option<&IluPreconditioner<Complex64>>,
    guess: Option<&[Complex64]>,
    sentinel: &mut ExactResidualSentinel,
    structural_cabinet_velocity: Option<&[Complex64]>,
) -> Result<Vec<Complex64>, SimulationError> {
    solve_bem_with_driver_velocities_and_structure_control(
        elements,
        speaker_mesh,
        model,
        frequency,
        velocities,
        nodes,
        physics,
        dense_matrix,
        hmatrix,
        config,
        gcro_dr_subspace,
        dense_preconditioner,
        guess,
        sentinel,
        structural_cabinet_velocity,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn solve_bem_with_driver_velocities_and_structure_control(
    elements: &mut [Element],
    speaker_mesh: &SpeakerMesh,
    model: &LoudspeakerModel,
    frequency: f64,
    velocities: &[Complex64],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    dense_matrix: Option<&DenseMatrix<Complex64>>,
    hmatrix: Option<&HMatrix>,
    config: &SolverConfig,
    gcro_dr_subspace: &mut Option<GcroDrSubspace<Complex64>>,
    dense_preconditioner: Option<&IluPreconditioner<Complex64>>,
    guess: Option<&[Complex64]>,
    sentinel: &mut ExactResidualSentinel,
    structural_cabinet_velocity: Option<&[Complex64]>,
    control: Option<&SimulationControl>,
) -> Result<Vec<Complex64>, SimulationError> {
    if let Some(control) = control {
        control.check()?;
    }
    update_driver_bcs_with_structure(
        elements,
        speaker_mesh,
        model,
        frequency,
        Some(velocities),
        structural_cabinet_velocity,
    )?;
    let rhs = build_tbem_rhs_stabilized(
        elements,
        nodes,
        physics,
        EXTERIOR_TBEM_SIGN,
        config.beta_scale,
    );
    if let Some(hmatrix) = hmatrix {
        let (solution, residual) = solve_bem_system_hmatrix_with_control(
            hmatrix,
            &rhs,
            guess,
            config,
            gcro_dr_subspace.as_mut(),
            control,
        )?;
        if sentinel.should_verify_exact(config, residual) {
            if let Some(control) = control {
                control.check()?;
            }
            verify_exact_hmatrix_solution(elements, nodes, physics, &solution, &rhs, config)?;
        }
        Ok(solution)
    } else {
        let matrix = dense_matrix.ok_or_else(|| {
            SimulationError::new("dense BEM matrix was not assembled for dense solve")
        })?;
        solve_bem_system_with_preconditioner_control(
            matrix,
            &rhs,
            guess,
            config,
            gcro_dr_subspace.as_mut(),
            dense_preconditioner,
            control,
        )
    }
}

/// Coupled driver state from the direct radiation-matrix network solve.
pub(crate) struct TypedDriverCoupling {
    /// Axial driver velocities as BEM e^{−iωt} peak phasors, ready for the
    /// final field solve.
    pub(crate) velocities_bem: [Complex64; 2],
    pub(crate) velocities_bem_all: Vec<Complex64>,
    pub(crate) radiation_matrix: [[Complex64; 2]; 2],
    pub(crate) radiation_matrix_all: Vec<Vec<Complex64>>,
    pub(crate) radiation_matrix_diagnostics: RadiationMatrixDiagnostics,
    pub(crate) thermal_driver_states: [Option<ThermalDriverState>; 2],
    pub(crate) driver_thermal_states: Vec<Option<ThermalDriverState>>,
    pub(crate) power_ledger: Option<PowerLedger>,
    pub(crate) woofer_electrical_impedance_ohm: Option<Complex64>,
    pub(crate) tweeter_electrical_impedance_ohm: Option<Complex64>,
    pub(crate) driver_electrical_impedances_ohm: Vec<Option<Complex64>>,
}

/// Whether the two unit-load solves run as one 2-RHS block GMRES.
///
/// Mirrors the strategy check of `solve_bem_system_with_preconditioner`: the
/// dense iterative path (a dense operator above `direct_threshold`, no
/// H-matrix) with a reusable per-frequency ILU preconditioner. The LU path
/// (small systems), the H-matrix path, and the sequential sweep (which
/// passes no shared preconditioner) keep the per-column solves.
fn use_block_unit_load_solve(
    dense_matrix: Option<&DenseMatrix<Complex64>>,
    hmatrix: Option<&HMatrix>,
    dense_preconditioner: Option<&IluPreconditioner<Complex64>>,
    config: &SolverConfig,
) -> bool {
    hmatrix.is_none()
        && dense_preconditioner.is_some()
        && dense_matrix.is_some_and(|matrix| matrix.nrows() > config.direct_threshold)
}

/// Step 1 of the typed-driver coupling: one unit-velocity solve per driver,
/// yielding the full N×N radiation matrix. Column j holds the forces with
/// driver j moving alone at unit axial velocity. The BEM is linear in the
/// velocity BCs, so superposition makes this matrix exact for the discrete
/// model.
///
/// On the dense iterative path with a shared ILU preconditioner the two
/// unit loads are solved as ONE 2-RHS block GMRES (one shared Krylov basis,
/// the operator applied through multi-vector GEMM). Everywhere else the
/// per-column solves are kept, byte-identical to the previous behavior.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn solve_unit_load_radiation_matrix(
    elements: &mut [Element],
    speaker_mesh: &SpeakerMesh,
    model: &LoudspeakerModel,
    frequency: f64,
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    dense_matrix: Option<&DenseMatrix<Complex64>>,
    hmatrix: Option<&HMatrix>,
    config: &SolverConfig,
    gcro_dr_subspace: &mut Option<GcroDrSubspace<Complex64>>,
    dense_preconditioner: Option<&IluPreconditioner<Complex64>>,
    guess: Option<&[Complex64]>,
    sentinel: &mut ExactResidualSentinel,
    structural_cabinet_velocity: Option<&[Complex64]>,
) -> Result<Vec<Vec<Complex64>>, SimulationError> {
    solve_unit_load_radiation_matrix_control(
        elements,
        speaker_mesh,
        model,
        frequency,
        nodes,
        physics,
        dense_matrix,
        hmatrix,
        config,
        gcro_dr_subspace,
        dense_preconditioner,
        guess,
        sentinel,
        structural_cabinet_velocity,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn solve_unit_load_radiation_matrix_control(
    elements: &mut [Element],
    speaker_mesh: &SpeakerMesh,
    model: &LoudspeakerModel,
    frequency: f64,
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    dense_matrix: Option<&DenseMatrix<Complex64>>,
    hmatrix: Option<&HMatrix>,
    config: &SolverConfig,
    gcro_dr_subspace: &mut Option<GcroDrSubspace<Complex64>>,
    dense_preconditioner: Option<&IluPreconditioner<Complex64>>,
    guess: Option<&[Complex64]>,
    sentinel: &mut ExactResidualSentinel,
    structural_cabinet_velocity: Option<&[Complex64]>,
    control: Option<&SimulationControl>,
) -> Result<Vec<Vec<Complex64>>, SimulationError> {
    if let Some(control) = control {
        control.check()?;
    }
    let driver_count = model.drivers().len();
    if driver_count == 0 {
        return Err(SimulationError::new(
            "typed-driver coupling requires at least one driver",
        ));
    }
    let unit_states: Vec<Vec<Complex64>> = (0..driver_count)
        .map(|column| {
            (0..driver_count)
                .map(|row| Complex64::new((row == column) as u8 as f64, 0.0))
                .collect()
        })
        .collect();
    let mut z_rad = vec![vec![Complex64::default(); driver_count]; driver_count];

    if use_block_unit_load_solve(dense_matrix, hmatrix, dense_preconditioner, config) {
        let matrix = dense_matrix.expect("block unit-load solve needs the dense matrix");
        let preconditioner =
            dense_preconditioner.expect("block unit-load solve needs the shared ILU");
        let n = matrix.nrows();

        // Both unit-load right-hand sides against the shared operator.
        let mut rhs_block = DenseMatrix::from_elem(n, driver_count, Complex64::new(0.0, 0.0));
        for (column, velocities) in unit_states.iter().enumerate() {
            if let Some(control) = control {
                control.check()?;
            }
            update_driver_bcs_with_structure(
                elements,
                speaker_mesh,
                model,
                frequency,
                Some(velocities),
                structural_cabinet_velocity,
            )?;
            let rhs = build_tbem_rhs_stabilized(
                elements,
                nodes,
                physics,
                EXTERIOR_TBEM_SIGN,
                config.beta_scale,
            );
            if !complex_values_are_finite(&rhs) {
                return Err(SimulationError::new(
                    "unit-load right-hand side contains non-finite values",
                ));
            }
            rhs_block.set_column(column, &rhs);
        }

        // The warm-start guess is the previous frequency's coupled surface
        // pressure; the per-column solves each receive it as x0 today, so
        // both block columns start from it as well.
        let x0 = guess.filter(|guess| guess.len() == n).map(|guess| {
            let mut x0 = DenseMatrix::from_elem(n, driver_count, Complex64::new(0.0, 0.0));
            for column in 0..driver_count {
                x0.set_column(column, guess);
            }
            x0
        });
        // Mirror the per-column path: inner tolerance one decade tighter
        // than the hard true-residual gate below.
        let block_config = BlockGmresConfig {
            max_iterations: config.max_iterations,
            restart: config.restart,
            tolerance: config.tolerance * 0.1,
            true_residual_check: true,
            // `exact_residual_sentinel` controls the separate H-matrix
            // compressed-vs-exact validation policy, not inner Krylov
            // checkpoints. Keep the block solver's optional cadence off
            // unless a dedicated inner-solver setting is added here.
            exact_residual_check_interval: 0,
        };
        let operator = DenseOperator { matrix };
        let solution = if let Some(control) = control {
            try_block_gmres_preconditioned_with_abort(
                &operator,
                &rhs_block,
                preconditioner,
                x0.as_ref(),
                &block_config,
                || control.is_cancelled(),
            )
            .map_err(|error| {
                if control.is_cancelled() {
                    SimulationError::cancelled()
                } else {
                    SimulationError::new(format!("controlled BlockGMRES failed: {error}"))
                }
            })?
        } else {
            block_gmres_preconditioned_with_guess(
                &operator,
                &rhs_block,
                preconditioner,
                x0.as_ref(),
                &block_config,
            )
        };
        eprintln!(
            "  BlockGMRES+ILU ({} unit loads): {} block iters, converged={}",
            driver_count, solution.iterations, solution.converged
        );

        for column in 0..driver_count {
            let pressure = solution.x.column_to_vec(column);
            let rhs = rhs_block.column_to_vec(column);
            let residual = relative_residual(matrix, &pressure, &rhs);
            if !solution.converged || !complex_values_are_finite(&pressure) {
                return Err(SimulationError::new(format!(
                    "BlockGMRES+ILU failed to converge for unit load {column}: reported converged={}, true residual={residual:.3e}, tolerance={:.3e}",
                    solution.converged, config.tolerance
                )));
            }
            validate_true_residual("BlockGMRES+ILU unit load", residual, config)?;
            let forces = integrate_driver_axial_forces(elements, speaker_mesh, &pressure);
            // Convention bridge (see driver_radiation_impedances):
            // conjugation converts the BEM e^{−iωt} ratio into the modeling
            // e^{+iωt} convention; the peak/RMS difference cancels in F/v.
            for (row, force) in z_rad.iter_mut().zip(forces.iter().copied()) {
                row[column] = bem_impedance_to_modeling(force);
            }
        }
        return Ok(z_rad);
    }

    for (column, velocities) in unit_states.iter().enumerate() {
        let pressure = solve_bem_with_driver_velocities_and_structure_control(
            elements,
            speaker_mesh,
            model,
            frequency,
            velocities,
            nodes,
            physics,
            dense_matrix,
            hmatrix,
            config,
            gcro_dr_subspace,
            dense_preconditioner,
            guess,
            sentinel,
            structural_cabinet_velocity,
            control,
        )?;
        let forces = integrate_driver_axial_forces(elements, speaker_mesh, &pressure);
        // Convention bridge (see driver_radiation_impedances): conjugation
        // converts the BEM e^{−iωt} ratio into the modeling e^{+iωt}
        // convention; the peak/RMS amplitude difference cancels in F/v.
        for (driver, force) in forces.iter().copied().enumerate().take(driver_count) {
            z_rad[driver][column] = bem_impedance_to_modeling(force);
        }
    }
    Ok(z_rad)
}

/// Couple typed driver models to the BEM through the full complex 2×2
/// mechanical radiation-impedance matrix
///
/// ```text
/// [F_w]   [Z_ww Z_wt] [v_w]
/// [F_t] = [Z_tw Z_tt] [v_t]
/// ```
///
/// This replaces the former scalar `force / velocity` fixed point. One
/// unit-velocity right-hand side per driver is solved against the assembled
/// operator, both driver forces are integrated per right-hand side, and the
/// electrical/mechanical network is then solved directly. That is three BEM
/// solves per frequency instead of up to six, stays well-conditioned when a
/// driver's velocity crosses zero, and exposes reciprocity, passivity, and
/// energy-closure checks on the acoustic load.
///
/// Returns the final surface pressure at the coupled velocity state plus the
/// coupled driver response.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_typed_driver_coupling(
    elements: &mut [Element],
    speaker_mesh: &SpeakerMesh,
    model: &LoudspeakerModel,
    frequency: f64,
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    dense_matrix: Option<&DenseMatrix<Complex64>>,
    hmatrix: Option<&HMatrix>,
    config: &SolverConfig,
    gcro_dr_subspace: &mut Option<GcroDrSubspace<Complex64>>,
    dense_preconditioner: Option<&IluPreconditioner<Complex64>>,
    guess: Option<&[Complex64]>,
    sentinel: &mut ExactResidualSentinel,
) -> Result<(Vec<Complex64>, TypedDriverCoupling), SimulationError> {
    solve_typed_driver_coupling_with_structure(
        elements,
        speaker_mesh,
        model,
        frequency,
        nodes,
        physics,
        dense_matrix,
        hmatrix,
        config,
        gcro_dr_subspace,
        dense_preconditioner,
        guess,
        sentinel,
        None,
    )
}

/// Typed-driver coupling with a partitioned structural cabinet velocity.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_typed_driver_coupling_with_structure(
    elements: &mut [Element],
    speaker_mesh: &SpeakerMesh,
    model: &LoudspeakerModel,
    frequency: f64,
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    dense_matrix: Option<&DenseMatrix<Complex64>>,
    hmatrix: Option<&HMatrix>,
    config: &SolverConfig,
    gcro_dr_subspace: &mut Option<GcroDrSubspace<Complex64>>,
    dense_preconditioner: Option<&IluPreconditioner<Complex64>>,
    guess: Option<&[Complex64]>,
    sentinel: &mut ExactResidualSentinel,
    structural_cabinet_velocity: Option<&[Complex64]>,
) -> Result<(Vec<Complex64>, TypedDriverCoupling), SimulationError> {
    solve_typed_driver_coupling_with_structure_control(
        elements,
        speaker_mesh,
        model,
        frequency,
        nodes,
        physics,
        dense_matrix,
        hmatrix,
        config,
        gcro_dr_subspace,
        dense_preconditioner,
        guess,
        sentinel,
        structural_cabinet_velocity,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn solve_typed_driver_coupling_with_structure_control(
    elements: &mut [Element],
    speaker_mesh: &SpeakerMesh,
    model: &LoudspeakerModel,
    frequency: f64,
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    dense_matrix: Option<&DenseMatrix<Complex64>>,
    hmatrix: Option<&HMatrix>,
    config: &SolverConfig,
    gcro_dr_subspace: &mut Option<GcroDrSubspace<Complex64>>,
    dense_preconditioner: Option<&IluPreconditioner<Complex64>>,
    guess: Option<&[Complex64]>,
    sentinel: &mut ExactResidualSentinel,
    structural_cabinet_velocity: Option<&[Complex64]>,
    control: Option<&SimulationControl>,
) -> Result<(Vec<Complex64>, TypedDriverCoupling), SimulationError> {
    if let Some(control) = control {
        control.check()?;
    }
    // 1. One unit-velocity solve per driver → full 2×2 radiation matrix
    //    (a single 2-RHS block solve on the dense iterative path with a
    //    shared ILU; per-column solves otherwise).
    let z_rad = solve_unit_load_radiation_matrix_control(
        elements,
        speaker_mesh,
        model,
        frequency,
        nodes,
        physics,
        dense_matrix,
        hmatrix,
        config,
        gcro_dr_subspace,
        dense_preconditioner,
        guess,
        sentinel,
        structural_cabinet_velocity,
        control,
    )?;
    // 2. Direct network solve. Use one N-port implementation for every
    // production driver count. The fixed two-driver helper remains below as
    // a compatibility surface for older callers and focused equation tests,
    // but production thermal and power-ledger behavior must not diverge by
    // port count.
    let coupling =
        solve_coupled_driver_network_generic_with_control(model, frequency, &z_rad, control)?;

    // 3. Field solve at the coupled state.
    let surface_pressure = solve_bem_with_driver_velocities_and_structure_control(
        elements,
        speaker_mesh,
        model,
        frequency,
        &coupling.velocities_bem_all,
        nodes,
        physics,
        dense_matrix,
        hmatrix,
        config,
        gcro_dr_subspace,
        dense_preconditioner,
        guess,
        sentinel,
        structural_cabinet_velocity,
        control,
    )?;
    Ok((surface_pressure, coupling))
}

/// Return the RMS electrical source seen by each typed driver.
///
/// A product design owns the crossover transfer functions. The driver model
/// reference voltage remains the per-driver nominal input level, while the
/// product branch transfer supplies the complex gain and phase at the current
/// frequency. Untyped drivers return zero because their velocity is a
/// prescribed boundary condition rather than an electrical unknown.
///
/// Convention note (SIM-PERF-20261013.md section 1): crossover lives at the
/// SOURCE here (drive voltages → velocity BCs, orthogonal to propagation),
/// unlike the room TBEM/FEM `ReceiverOnly` convention, which excites
/// isotropically and applies the pattern to the total field at the receiver.
/// Keep the two apart when comparing speaker and room transfer functions.
fn typed_driver_drive_voltages(
    model: &LoudspeakerModel,
    frequency: f64,
) -> Result<Vec<Complex64>, SimulationError> {
    let drivers = model.drivers();
    let transfers = if let Some(product) = &model.product_design {
        product
            .drive_voltages_at(frequency, Complex64::new(1.0, 0.0))
            .map_err(|error| {
                SimulationError::new(format!(
                    "product crossover drive failed at {frequency:.3} Hz: {error}"
                ))
            })?
    } else {
        vec![Complex64::new(1.0, 0.0); drivers.len()]
    };
    if transfers.len() != drivers.len() {
        return Err(SimulationError::new(format!(
            "product crossover has {} driver drives, expected {}",
            transfers.len(),
            drivers.len()
        )));
    }
    Ok(drivers
        .iter()
        .enumerate()
        .map(|(index, driver)| {
            driver
                .driver_model
                .as_ref()
                .map(|driver_model| {
                    Complex64::new(driver_model.reference_voltage_rms_v, 0.0)
                        * driver_model.drive_transfer_at(frequency)
                        * transfers[index]
                })
                .unwrap_or_default()
        })
        .collect())
}

/// Solve the coupled electrical/mechanical network of the two drivers
/// against a complex 2×2 mechanical radiation-impedance matrix.
///
/// All network quantities use the modeling crate's e^{+iωt} RMS convention.
/// Each modeled driver obeys
///
/// ```text
/// BL_i V_i / Z_e,i = (Z_m,i + BL_i²/Z_e,i) v_i + Σ_j Z_ij v_j
/// ```
///
/// which is linear in the velocities, so the coupled state solves directly.
/// Drivers without a typed model keep their prescribed piston velocity;
/// their mutual radiation load still acts on the modeled driver.
#[cfg(test)]
fn solve_coupled_driver_network(
    model: &LoudspeakerModel,
    frequency: f64,
    z_rad: &[[Complex64; 2]; 2],
) -> Result<TypedDriverCoupling, SimulationError> {
    let radiation_matrix_diagnostics = diagnose_radiation_matrix(frequency, z_rad)?;
    let driver_models = [&model.woofer.driver_model, &model.tweeter.driver_model];
    let thermal_specs = [
        driver_models[0]
            .as_ref()
            .and_then(|driver| driver.thermal_compression_spec),
        driver_models[1]
            .as_ref()
            .and_then(|driver| driver.thermal_compression_spec),
    ];
    let thermal_networks = [
        driver_models[0]
            .as_ref()
            .and_then(|driver| driver.thermal_network),
        driver_models[1]
            .as_ref()
            .and_then(|driver| driver.thermal_network),
    ];
    let prescribed_bem = [
        model.woofer.velocity_at(frequency),
        model.tweeter.velocity_at(frequency),
    ];
    let drive_voltages = typed_driver_drive_voltages(model, frequency)?;

    let thermal_enabled =
        thermal_specs.iter().any(Option::is_some) || thermal_networks.iter().any(Option::is_some);
    let mut hot_resistance = [None; 2];
    // The assembled network uses the previous fixed-point resistance while
    // `hot_resistance` holds the next estimate. Keep the applied value for
    // final loss accounting so the ledger describes the solved state.
    let mut applied_hot_resistance = [None; 2];
    let mut thermal_temperature = [0.0; 2];
    let mut thermal_residual_ohm = [0.0; 2];
    let mut thermal_iteration_count = [0; 2];
    let mut cold_velocity_norm = [None; 2];
    let mut network = [None; 2];
    let mut velocities_rms = [Complex64::default(); 2];
    let mut electrical_impedance_ohm = [None; 2];
    let mut thermal_converged = !thermal_enabled;
    let mut thermal_network_states: [Option<ThermalNetworkState>; 2] = [None, None];
    let mut thermal_network_powers: [Option<ThermalNetworkPower>; 2] = [None, None];

    // Thermal resistance changes are local nonlinearities around an otherwise
    // linear two-port solve. Re-solve the same radiation matrix until each
    // configured coil's fixed point closes; the BEM unit-load work is reused.
    for thermal_iteration in 0..100 {
        network = [None; 2];
        velocities_rms = [Complex64::default(); 2];
        for (driver, driver_model) in driver_models.iter().enumerate() {
            if let Some(driver_model) = driver_model {
                let applied_resistance = hot_resistance[driver]
                    .unwrap_or(driver_model.thiele_small.voice_coil_resistance_ohm);
                let parameters = if let Some(resistance) = hot_resistance[driver] {
                    driver_model.network_parameters_at_with_hot_resistance(frequency, resistance)
                } else {
                    driver_model.network_parameters_at(frequency)
                }
                .map_err(|error| {
                    SimulationError::new(format!(
                        "driver network parameters failed at {frequency:.3} Hz: {error}"
                    ))
                })?;
                if parameters.coil_impedance_ohm.norm() <= f64::MIN_POSITIVE {
                    return Err(SimulationError::new(format!(
                        "driver {driver} coil impedance vanished at {frequency:.3} Hz"
                    )));
                }
                applied_hot_resistance[driver] = Some(applied_resistance);
                network[driver] = Some(parameters);
            } else {
                // Untyped drivers prescribe the BEM peak velocity directly;
                // convert to the modeling RMS convention for the network solve.
                velocities_rms[driver] = bem_peak_to_modeling_rms(prescribed_bem[driver]);
            }
        }

        // Unknown velocities are the modeled drivers'; prescribed mutual
        // contributions move to the right-hand side.
        let unknowns: Vec<usize> = (0..2).filter(|&driver| network[driver].is_some()).collect();
        let diagonal =
            |driver: usize, parameters: &sonium_sim_speaker_modeling::DriverNetworkParameters| {
                parameters.intrinsic_mechanical_impedance
                    + z_rad[driver][driver]
                    + Complex64::new(parameters.force_factor_tm.powi(2), 0.0)
                        / parameters.coil_impedance_ohm
            };
        let right_hand_side =
            |driver: usize, parameters: &sonium_sim_speaker_modeling::DriverNetworkParameters| {
                let voltage = drive_voltages[driver];
                let mut rhs = Complex64::new(parameters.force_factor_tm, 0.0) * voltage
                    / parameters.coil_impedance_ohm;
                for other in 0..2 {
                    if network[other].is_none() {
                        rhs -= z_rad[driver][other] * velocities_rms[other];
                    }
                }
                rhs
            };
        match unknowns.as_slice() {
            [] => {}
            &[driver] => {
                let parameters = network[driver].expect("unknown is modeled");
                let a = diagonal(driver, &parameters);
                if a.norm() <= f64::MIN_POSITIVE {
                    return Err(SimulationError::new(format!(
                        "coupled driver network is singular at {frequency:.3} Hz"
                    )));
                }
                velocities_rms[driver] = right_hand_side(driver, &parameters) / a;
            }
            &[first, second] => {
                let first_parameters = network[first].expect("unknown is modeled");
                let second_parameters = network[second].expect("unknown is modeled");
                let a_ff = diagonal(first, &first_parameters);
                let a_ss = diagonal(second, &second_parameters);
                let determinant = a_ff * a_ss - z_rad[first][second] * z_rad[second][first];
                if determinant.norm() <= f64::MIN_POSITIVE {
                    return Err(SimulationError::new(format!(
                        "coupled two-driver network is singular at {frequency:.3} Hz"
                    )));
                }
                let rhs_f = right_hand_side(first, &first_parameters);
                let rhs_s = right_hand_side(second, &second_parameters);
                velocities_rms[first] = (rhs_f * a_ss - z_rad[first][second] * rhs_s) / determinant;
                velocities_rms[second] =
                    (a_ff * rhs_s - z_rad[second][first] * rhs_f) / determinant;
            }
            _ => {
                return Err(SimulationError::new(format!(
                    "coupled two-driver network supports at most two drivers, got {}",
                    unknowns.len()
                )));
            }
        }

        // Electrical input impedance Z_in = V/I with I = (V − BL v)/Z_e.
        electrical_impedance_ohm = [None; 2];
        for driver in unknowns.iter().copied() {
            let parameters = network[driver].expect("unknown is modeled");
            let voltage = drive_voltages[driver];
            let current = (voltage
                - Complex64::new(parameters.force_factor_tm, 0.0) * velocities_rms[driver])
                / parameters.coil_impedance_ohm;
            if current.norm() <= f64::MIN_POSITIVE {
                return Err(SimulationError::new(format!(
                    "driver {driver} voice-coil current vanished at {frequency:.3} Hz"
                )));
            }
            electrical_impedance_ohm[driver] = Some(voltage / current);
        }

        if !thermal_enabled {
            break;
        }
        thermal_converged = true;
        for driver in 0..2 {
            let parameters = network[driver].expect("thermal driver is modeled");
            let driver_model = driver_models[driver].as_ref().expect("thermal driver");
            let voltage = drive_voltages[driver];
            let current = (voltage
                - Complex64::new(parameters.force_factor_tm, 0.0) * velocities_rms[driver])
                / parameters.coil_impedance_ohm;
            if thermal_iteration == 0 {
                cold_velocity_norm[driver] = Some(velocities_rms[driver].norm());
            }
            let cold_resistance = driver_model.thiele_small.voice_coil_resistance_ohm;
            let resistance = hot_resistance[driver].unwrap_or(cold_resistance);
            let coil_loss = resistance * current.norm_sqr();
            let eddy_loss = (parameters.coil_impedance_ohm.re - resistance) * current.norm_sqr();
            let (temperature, next_resistance) = if let Some(policy) = thermal_networks[driver] {
                let power = policy
                    .power_from_losses(coil_loss, eddy_loss)
                    .map_err(|error| {
                        SimulationError::new(format!(
                            "driver thermal power source failed at {frequency:.3} Hz: {error}"
                        ))
                    })?;
                let state = policy
                    .specification
                    .solve_steady_state(power)
                    .map_err(|error| {
                        SimulationError::new(format!(
                            "driver thermal network failed at {frequency:.3} Hz: {error}"
                        ))
                    })?;
                let temperature = state.temperatures_k[ThermalNode::VoiceCoil as usize];
                let resistance = policy
                    .copper_resistance_at(cold_resistance, temperature)
                    .map_err(|error| {
                        SimulationError::new(format!(
                            "driver hot resistance failed at {frequency:.3} Hz: {error}"
                        ))
                    })?;
                thermal_network_powers[driver] = Some(power);
                thermal_network_states[driver] = Some(state);
                (temperature, resistance)
            } else if let Some(spec) = thermal_specs[driver] {
                let thermal = VoiceCoilThermalModel::simple(
                    cold_resistance,
                    spec.ambient_temperature_k,
                    spec.thermal_resistance_k_per_w,
                    spec.coil_thermal_capacitance_j_per_k,
                );
                let temperature =
                    spec.ambient_temperature_k + coil_loss * spec.thermal_resistance_k_per_w;
                (temperature, thermal.resistance_at(temperature))
            } else {
                continue;
            };
            let residual = (next_resistance - resistance).abs();
            if !temperature.is_finite() || !next_resistance.is_finite() || !residual.is_finite() {
                return Err(SimulationError::new(format!(
                    "thermal driver state became non-finite at {frequency:.3} Hz"
                )));
            }
            hot_resistance[driver] = Some(next_resistance);
            thermal_temperature[driver] = temperature;
            thermal_residual_ohm[driver] = residual;
            thermal_iteration_count[driver] = thermal_iteration + 1;
            if residual > 1.0e-10 * cold_resistance.max(1.0) {
                thermal_converged = false;
            }
        }
        if thermal_converged {
            break;
        }
    }
    if !thermal_converged {
        return Err(SimulationError::new(format!(
            "thermal driver coupling did not converge at {frequency:.3} Hz"
        )));
    }

    diagnose_radiated_power(frequency, z_rad, &velocities_rms)?;

    let mut thermal_driver_states = [None; 2];
    for driver in 0..2 {
        if thermal_specs[driver].is_none() && thermal_networks[driver].is_none() {
            continue;
        }
        let parameters = network[driver].expect("thermal driver is modeled");
        let driver_model = driver_models[driver].as_ref().expect("thermal driver");
        let voltage = drive_voltages[driver];
        let current = (voltage
            - Complex64::new(parameters.force_factor_tm, 0.0) * velocities_rms[driver])
            / parameters.coil_impedance_ohm;
        let resistance = applied_hot_resistance[driver]
            .unwrap_or(driver_model.thiele_small.voice_coil_resistance_ohm);
        let coil_loss = resistance * current.norm_sqr();
        let eddy_loss = (parameters.coil_impedance_ohm.re - resistance) * current.norm_sqr();
        let mechanical_loss =
            parameters.intrinsic_mechanical_impedance.re * velocities_rms[driver].norm_sqr();
        let radiation_force = (0..2)
            .map(|other| z_rad[driver][other] * velocities_rms[other])
            .sum::<Complex64>();
        let radiation_power = (velocities_rms[driver].conj() * radiation_force).re;
        let input_power = (voltage * current.conj()).re;
        let balance = input_power - coil_loss - eddy_loss - mechanical_loss - radiation_power;
        let scale = input_power
            .abs()
            .max(coil_loss.abs())
            .max(eddy_loss.abs())
            .max(mechanical_loss.abs())
            .max(radiation_power.abs())
            .max(1.0e-12);
        if !balance.is_finite() || balance.abs() > 5.0e-7 * scale {
            return Err(SimulationError::new(format!(
                "thermal driver power balance failed at {frequency:.3} Hz: residual={balance:.3e} W, scale={scale:.3e} W"
            )));
        }
        // Read the configured thermal contract here so the final-state path
        // retains the exact policy used by the fixed-point solve.
        let _thermal_time_constant_s = thermal_specs[driver]
            .map(|spec| spec.coil_thermal_capacitance_j_per_k * spec.thermal_resistance_k_per_w)
            .or_else(|| {
                thermal_networks[driver].map(|network| {
                    network.specification.heat_capacity_j_per_k[ThermalNode::VoiceCoil as usize]
                        / network.specification.ambient_conductance_w_per_k
                            [ThermalNode::VoiceCoil as usize]
                })
            });
        let compression_db = cold_velocity_norm[driver]
            .filter(|velocity| *velocity > f64::MIN_POSITIVE)
            .map_or(0.0, |cold_velocity| {
                20.0 * (cold_velocity / velocities_rms[driver].norm().max(f64::MIN_POSITIVE))
                    .log10()
            })
            .max(0.0);
        thermal_driver_states[driver] = Some(ThermalDriverState {
            coil_temperature_k: thermal_temperature[driver],
            hot_resistance_ohm: resistance,
            coil_loss_w: coil_loss,
            eddy_loss_w: eddy_loss,
            input_power_w: input_power,
            mechanical_loss_w: mechanical_loss,
            radiation_power_w: radiation_power,
            power_balance_residual_w: balance,
            compression_db,
            thermal_iteration_count: thermal_iteration_count[driver],
            thermal_residual_ohm: thermal_residual_ohm[driver],
            nonlinear_temperature_feedback_applied: false,
            thermal_network_temperatures_k: thermal_network_states[driver]
                .map(|state| state.temperatures_k),
            thermal_network_power_w: thermal_network_powers[driver],
            thermal_network_residual_w: thermal_network_states[driver]
                .map(|state| state.maximum_balance_residual_w),
        });
    }

    // For a fully typed network, the electrical source must close against
    // copper, intrinsic mechanical, and acoustic radiation power. Prescribed
    // legacy pistons are external power ports, so their mixed-network power
    // is intentionally left to the caller rather than being misreported as a
    // closed product balance.
    let power_ledger = if network.iter().all(Option::is_some) {
        let mut input_power = 0.0;
        let mut copper_power = 0.0;
        let mut magnetic_power = 0.0;
        let mut mechanical_power = 0.0;
        for driver in 0..2 {
            let parameters = network[driver].expect("all drivers are typed");
            let voltage = drive_voltages[driver];
            let current = (voltage
                - Complex64::new(parameters.force_factor_tm, 0.0) * velocities_rms[driver])
                / parameters.coil_impedance_ohm;
            let cold_resistance = driver_models[driver]
                .as_ref()
                .expect("all drivers are typed")
                .thiele_small
                .voice_coil_resistance_ohm;
            let copper_resistance = applied_hot_resistance[driver].unwrap_or(cold_resistance);
            input_power += (voltage * current.conj()).re;
            copper_power += copper_resistance * current.norm_sqr();
            magnetic_power +=
                (parameters.coil_impedance_ohm.re - copper_resistance) * current.norm_sqr();
            mechanical_power +=
                parameters.intrinsic_mechanical_impedance.re * velocities_rms[driver].norm_sqr();
        }
        let radiation_power = radiation_matrix_power(z_rad, &velocities_rms);
        let ledger = PowerLedger::from_components(
            input_power,
            copper_power,
            magnetic_power,
            mechanical_power,
            0.0,
            0.0,
            radiation_power,
            0.0,
        )
        .map_err(|error| {
            SimulationError::new(format!(
                "coupled driver power ledger failed at {frequency:.3} Hz: {error}"
            ))
        })?;
        ledger
            .close(PowerLedgerTolerance::default())
            .map_err(|error| {
                SimulationError::new(format!(
                    "coupled driver power ledger did not close at {frequency:.3} Hz: {error}"
                ))
            })?;
        Some(ledger)
    } else {
        None
    };

    // Back to the BEM convention: conjugated e^{−iωt} phasor at peak amplitude.
    let velocities_bem = velocities_rms.map(modeling_rms_to_bem_peak);
    Ok(TypedDriverCoupling {
        velocities_bem,
        velocities_bem_all: velocities_bem.to_vec(),
        radiation_matrix: *z_rad,
        radiation_matrix_all: z_rad.iter().map(|row| row.to_vec()).collect(),
        radiation_matrix_diagnostics,
        thermal_driver_states,
        driver_thermal_states: thermal_driver_states.to_vec(),
        power_ledger,
        woofer_electrical_impedance_ohm: electrical_impedance_ohm[0],
        tweeter_electrical_impedance_ohm: electrical_impedance_ohm[1],
        driver_electrical_impedances_ohm: electrical_impedance_ohm.to_vec(),
    })
}

/// Generic N-port production network path. The legacy woofer/tweeter helper
/// above remains as a compatibility surface for older callers, but production
/// solves use this implementation for every driver count so thermal iteration,
/// power ledgers, and result artifacts have one contract.
#[cfg(test)]
fn solve_coupled_driver_network_generic(
    model: &LoudspeakerModel,
    frequency: f64,
    z_rad: &[Vec<Complex64>],
) -> Result<TypedDriverCoupling, SimulationError> {
    solve_coupled_driver_network_generic_with_control(model, frequency, z_rad, None)
}

fn solve_coupled_driver_network_generic_with_control(
    model: &LoudspeakerModel,
    frequency: f64,
    z_rad: &[Vec<Complex64>],
    control: Option<&SimulationControl>,
) -> Result<TypedDriverCoupling, SimulationError> {
    if let Some(control) = control {
        control.check()?;
    }
    let drivers = model.drivers();
    let driver_count = drivers.len();
    if driver_count == 0
        || z_rad.len() != driver_count
        || z_rad.iter().any(|row| row.len() != driver_count)
    {
        return Err(SimulationError::new(
            "generic typed-driver coupling requires a matching N-port matrix",
        ));
    }
    let diagnostics = diagnose_radiation_matrix_generic(frequency, z_rad)?;
    let driver_models: Vec<_> = drivers
        .iter()
        .map(|driver| driver.driver_model.as_ref())
        .collect();
    let thermal_specs: Vec<_> = driver_models
        .iter()
        .map(|driver| driver.and_then(|driver| driver.thermal_compression_spec))
        .collect();
    let prescribed_bem: Vec<_> = drivers
        .iter()
        .map(|driver| driver.velocity_at(frequency))
        .collect();
    let drive_voltages = typed_driver_drive_voltages(model, frequency)?;
    let mut hot_resistance = vec![None; driver_count];
    let mut applied_hot_resistance = vec![None; driver_count];
    let mut thermal_temperature = vec![0.0_f64; driver_count];
    let mut thermal_residual_ohm = vec![0.0; driver_count];
    let mut thermal_iteration_count = vec![0; driver_count];
    let mut cold_velocity_norm = vec![None; driver_count];
    let mut network = vec![None; driver_count];
    let mut velocities_rms = vec![Complex64::default(); driver_count];
    let mut electrical_impedances = vec![None; driver_count];
    let thermal_networks: Vec<_> = driver_models
        .iter()
        .map(|driver| driver.and_then(|driver| driver.thermal_network))
        .collect();
    let thermal_enabled =
        thermal_specs.iter().any(Option::is_some) || thermal_networks.iter().any(Option::is_some);
    let mut thermal_converged = !thermal_enabled;
    let mut thermal_network_states = vec![None; driver_count];
    let mut thermal_network_powers = vec![None; driver_count];
    let mut nonlinear_temperature_applied = vec![false; driver_count];

    for thermal_iteration in 0..100 {
        if let Some(control) = control {
            control.check()?;
        }
        network.fill(None);
        velocities_rms.fill(Complex64::default());
        for (driver, driver_model) in driver_models.iter().enumerate() {
            if let Some(control) = control {
                control.check()?;
            }
            if let Some(driver_model) = driver_model {
                let applied_resistance = hot_resistance[driver]
                    .unwrap_or(driver_model.thiele_small.voice_coil_resistance_ohm);
                let parameters = if let Some(resistance) = hot_resistance[driver] {
                    let temperature = thermal_temperature[driver];
                    if driver_model.nonlinear_motor.is_some()
                        && temperature.is_finite()
                        && temperature > 0.0
                    {
                        // Frequency-domain radiation only retains a harmonic
                        // amplitude, not a signed large-signal displacement.
                        // Keep x = 0 here; the thermal iterate still feeds
                        // temperature-dependent BL, Le, and eddy loss into
                        // the loaded product network without inventing a DC
                        // bias. Full x/current large-signal closure remains
                        // the nonlinear time-domain model's responsibility.
                        nonlinear_temperature_applied[driver] = true;
                        driver_model
                            .network_parameters_at_operating_point_with_hot_resistance(
                                MotorOperatingPoint {
                                    displacement_m: 0.0,
                                    current_a: 0.0,
                                    frequency_hz: frequency,
                                    temperature_k: temperature,
                                    flux_density_t: 0.0,
                                },
                                resistance,
                            )
                            .map(|(parameters, _)| parameters)
                    } else {
                        driver_model
                            .network_parameters_at_with_hot_resistance(frequency, resistance)
                    }
                } else {
                    driver_model.network_parameters_at(frequency)
                }
                .map_err(|error| {
                    SimulationError::new(format!(
                        "driver network parameters failed at {frequency:.3} Hz: {error}"
                    ))
                })?;
                if parameters.coil_impedance_ohm.norm() <= f64::MIN_POSITIVE {
                    return Err(SimulationError::new(format!(
                        "driver {driver} coil impedance vanished at {frequency:.3} Hz"
                    )));
                }
                applied_hot_resistance[driver] = Some(applied_resistance);
                network[driver] = Some(parameters);
            } else {
                velocities_rms[driver] = bem_peak_to_modeling_rms(prescribed_bem[driver]);
            }
        }

        let unknowns: Vec<_> = (0..driver_count)
            .filter(|&driver| network[driver].is_some())
            .collect();
        if unknowns.is_empty() {
            return Err(SimulationError::new(
                "generic typed-driver path received no typed drivers",
            ));
        }
        let mut matrix = vec![vec![Complex64::default(); unknowns.len()]; unknowns.len()];
        let mut rhs = vec![Complex64::default(); unknowns.len()];
        for (row, &driver) in unknowns.iter().enumerate() {
            let parameters = network[driver].expect("unknown is modeled");
            let diagonal = parameters.intrinsic_mechanical_impedance
                + z_rad[driver][driver]
                + Complex64::new(parameters.force_factor_tm.powi(2), 0.0)
                    / parameters.coil_impedance_ohm;
            let voltage = drive_voltages[driver];
            rhs[row] = Complex64::new(parameters.force_factor_tm, 0.0) * voltage
                / parameters.coil_impedance_ohm;
            for other in 0..driver_count {
                if network[other].is_none() {
                    rhs[row] -= z_rad[driver][other] * velocities_rms[other];
                }
            }
            for (column, &other) in unknowns.iter().enumerate() {
                matrix[row][column] = if driver == other {
                    diagonal
                } else {
                    z_rad[driver][other]
                };
            }
        }
        let solved = solve_complex_linear_system(matrix, rhs, frequency)?;
        for (row, &driver) in unknowns.iter().enumerate() {
            velocities_rms[driver] = solved[row];
        }

        electrical_impedances.fill(None);
        for &driver in &unknowns {
            let parameters = network[driver].expect("unknown is modeled");
            let voltage = drive_voltages[driver];
            let current = (voltage
                - Complex64::new(parameters.force_factor_tm, 0.0) * velocities_rms[driver])
                / parameters.coil_impedance_ohm;
            if current.norm() <= f64::MIN_POSITIVE {
                return Err(SimulationError::new(format!(
                    "driver {driver} voice-coil current vanished at {frequency:.3} Hz"
                )));
            }
            electrical_impedances[driver] = Some(voltage / current);
        }
        if !thermal_enabled {
            break;
        }

        thermal_converged = true;
        for driver in 0..driver_count {
            if thermal_specs[driver].is_none() && thermal_networks[driver].is_none() {
                continue;
            }
            let parameters = network[driver].expect("thermal driver is modeled");
            let driver_model = driver_models[driver].expect("thermal driver");
            let voltage = drive_voltages[driver];
            let current = (voltage
                - Complex64::new(parameters.force_factor_tm, 0.0) * velocities_rms[driver])
                / parameters.coil_impedance_ohm;
            if thermal_iteration == 0 {
                cold_velocity_norm[driver] = Some(velocities_rms[driver].norm());
            }
            let cold_resistance = driver_model.thiele_small.voice_coil_resistance_ohm;
            let resistance = hot_resistance[driver].unwrap_or(cold_resistance);
            let coil_loss = resistance * current.norm_sqr();
            let eddy_loss = (parameters.coil_impedance_ohm.re - resistance) * current.norm_sqr();
            let (temperature, next_resistance) = if let Some(policy) = thermal_networks[driver] {
                let power = policy
                    .power_from_losses(coil_loss, eddy_loss)
                    .map_err(|error| {
                        SimulationError::new(format!(
                            "driver thermal power source failed at {frequency:.3} Hz: {error}"
                        ))
                    })?;
                let state = policy
                    .specification
                    .solve_steady_state(power)
                    .map_err(|error| {
                        SimulationError::new(format!(
                            "driver thermal network failed at {frequency:.3} Hz: {error}"
                        ))
                    })?;
                let temperature = state.temperatures_k[ThermalNode::VoiceCoil as usize];
                let resistance = policy
                    .copper_resistance_at(cold_resistance, temperature)
                    .map_err(|error| {
                        SimulationError::new(format!(
                            "driver hot resistance failed at {frequency:.3} Hz: {error}"
                        ))
                    })?;
                thermal_network_powers[driver] = Some(power);
                thermal_network_states[driver] = Some(state);
                (temperature, resistance)
            } else if let Some(spec) = thermal_specs[driver] {
                let thermal = VoiceCoilThermalModel::simple(
                    cold_resistance,
                    spec.ambient_temperature_k,
                    spec.thermal_resistance_k_per_w,
                    spec.coil_thermal_capacitance_j_per_k,
                );
                let temperature =
                    spec.ambient_temperature_k + coil_loss * spec.thermal_resistance_k_per_w;
                (temperature, thermal.resistance_at(temperature))
            } else {
                continue;
            };
            let residual = (next_resistance - resistance).abs();
            if !temperature.is_finite() || !next_resistance.is_finite() || !residual.is_finite() {
                return Err(SimulationError::new(format!(
                    "thermal driver state became non-finite at {frequency:.3} Hz"
                )));
            }
            hot_resistance[driver] = Some(next_resistance);
            thermal_temperature[driver] = temperature;
            thermal_residual_ohm[driver] = residual;
            thermal_iteration_count[driver] = thermal_iteration + 1;
            if residual > 1.0e-10 * cold_resistance.max(1.0) {
                thermal_converged = false;
            }
        }
        if thermal_converged {
            // A first thermal solve may converge in resistance immediately
            // (for example, a zero copper coefficient) while still producing
            // a non-reference temperature. Run one more network pass so a
            // nonlinear motor sees that temperature before accepting it.
            let needs_nonlinear_temperature_pass = (0..driver_count).any(|driver| {
                (thermal_specs[driver].is_some() || thermal_networks[driver].is_some())
                    && driver_models[driver].is_some_and(|model| model.nonlinear_motor.is_some())
                    && thermal_temperature[driver].is_finite()
                    && thermal_temperature[driver] > 0.0
                    && !nonlinear_temperature_applied[driver]
            });
            if needs_nonlinear_temperature_pass {
                thermal_converged = false;
                continue;
            }
            break;
        }
    }
    if !thermal_converged {
        return Err(SimulationError::new(format!(
            "thermal driver coupling did not converge at {frequency:.3} Hz"
        )));
    }
    diagnose_radiated_power_generic(frequency, z_rad, &velocities_rms)?;

    let mut driver_thermal_states = vec![None; driver_count];
    for driver in 0..driver_count {
        if thermal_specs[driver].is_none() && thermal_networks[driver].is_none() {
            continue;
        }
        let parameters = network[driver].expect("thermal driver is modeled");
        let driver_model = driver_models[driver].expect("thermal driver");
        let voltage = drive_voltages[driver];
        let current = (voltage
            - Complex64::new(parameters.force_factor_tm, 0.0) * velocities_rms[driver])
            / parameters.coil_impedance_ohm;
        let resistance = applied_hot_resistance[driver]
            .unwrap_or(driver_model.thiele_small.voice_coil_resistance_ohm);
        let coil_loss = resistance * current.norm_sqr();
        let eddy_loss = (parameters.coil_impedance_ohm.re - resistance) * current.norm_sqr();
        let mechanical_loss =
            parameters.intrinsic_mechanical_impedance.re * velocities_rms[driver].norm_sqr();
        let radiation_force = (0..driver_count)
            .map(|other| z_rad[driver][other] * velocities_rms[other])
            .sum::<Complex64>();
        let radiation_power = (velocities_rms[driver].conj() * radiation_force).re;
        let input_power = (voltage * current.conj()).re;
        let balance = input_power - coil_loss - eddy_loss - mechanical_loss - radiation_power;
        let scale = input_power
            .abs()
            .max(coil_loss.abs())
            .max(eddy_loss.abs())
            .max(mechanical_loss.abs())
            .max(radiation_power.abs())
            .max(1.0e-12);
        if !balance.is_finite() || balance.abs() > 5.0e-7 * scale {
            return Err(SimulationError::new(format!(
                "thermal driver power balance failed at {frequency:.3} Hz: residual={balance:.3e} W, scale={scale:.3e} W"
            )));
        }
        let compression_db = cold_velocity_norm[driver]
            .filter(|velocity| *velocity > f64::MIN_POSITIVE)
            .map_or(0.0, |cold_velocity| {
                20.0 * (cold_velocity / velocities_rms[driver].norm().max(f64::MIN_POSITIVE))
                    .log10()
            })
            .max(0.0);
        driver_thermal_states[driver] = Some(ThermalDriverState {
            coil_temperature_k: thermal_temperature[driver],
            hot_resistance_ohm: resistance,
            coil_loss_w: coil_loss,
            eddy_loss_w: eddy_loss,
            input_power_w: input_power,
            mechanical_loss_w: mechanical_loss,
            radiation_power_w: radiation_power,
            power_balance_residual_w: balance,
            compression_db,
            thermal_iteration_count: thermal_iteration_count[driver],
            thermal_residual_ohm: thermal_residual_ohm[driver],
            nonlinear_temperature_feedback_applied: nonlinear_temperature_applied[driver],
            thermal_network_temperatures_k: thermal_network_states[driver]
                .map(|state| state.temperatures_k),
            thermal_network_power_w: thermal_network_powers[driver],
            thermal_network_residual_w: thermal_network_states[driver]
                .map(|state| state.maximum_balance_residual_w),
        });
    }
    let power_ledger = if network.iter().all(Option::is_some) {
        let mut input_power = 0.0;
        let mut copper_power = 0.0;
        let mut magnetic_power = 0.0;
        let mut mechanical_power = 0.0;
        for driver in 0..driver_count {
            let parameters = network[driver].expect("all drivers are typed");
            let voltage = drive_voltages[driver];
            let current = (voltage
                - Complex64::new(parameters.force_factor_tm, 0.0) * velocities_rms[driver])
                / parameters.coil_impedance_ohm;
            let cold_resistance = driver_models[driver]
                .expect("all drivers are typed")
                .thiele_small
                .voice_coil_resistance_ohm;
            let copper_resistance = applied_hot_resistance[driver].unwrap_or(cold_resistance);
            input_power += (voltage * current.conj()).re;
            copper_power += copper_resistance * current.norm_sqr();
            magnetic_power +=
                (parameters.coil_impedance_ohm.re - copper_resistance) * current.norm_sqr();
            mechanical_power +=
                parameters.intrinsic_mechanical_impedance.re * velocities_rms[driver].norm_sqr();
        }
        let radiation_power = radiation_matrix_power_generic(z_rad, &velocities_rms);
        let ledger = PowerLedger::from_components(
            input_power,
            copper_power,
            magnetic_power,
            mechanical_power,
            0.0,
            0.0,
            radiation_power,
            0.0,
        )
        .map_err(|error| {
            SimulationError::new(format!(
                "coupled driver power ledger failed at {frequency:.3} Hz: {error}"
            ))
        })?;
        ledger
            .close(PowerLedgerTolerance::default())
            .map_err(|error| {
                SimulationError::new(format!(
                    "coupled driver power ledger did not close at {frequency:.3} Hz: {error}"
                ))
            })?;
        Some(ledger)
    } else {
        None
    };

    let velocities_bem_all: Vec<_> = velocities_rms
        .into_iter()
        .map(modeling_rms_to_bem_peak)
        .collect();
    let mut velocities_bem = [Complex64::default(); 2];
    for (index, velocity) in velocities_bem_all.iter().copied().take(2).enumerate() {
        velocities_bem[index] = velocity;
    }
    let mut thermal_driver_states = [None; 2];
    for (index, state) in driver_thermal_states.iter().cloned().take(2).enumerate() {
        thermal_driver_states[index] = state;
    }
    let mut radiation_matrix = [[Complex64::default(); 2]; 2];
    for (row, values) in z_rad.iter().take(2).enumerate() {
        for (column, value) in values.iter().copied().take(2).enumerate() {
            radiation_matrix[row][column] = value;
        }
    }
    let mut driver_electrical_impedances_ohm = electrical_impedances;
    let woofer_electrical_impedance_ohm =
        driver_electrical_impedances_ohm.first().copied().flatten();
    let tweeter_electrical_impedance_ohm =
        driver_electrical_impedances_ohm.get(1).copied().flatten();
    Ok(TypedDriverCoupling {
        velocities_bem,
        velocities_bem_all,
        radiation_matrix,
        radiation_matrix_all: z_rad.to_vec(),
        radiation_matrix_diagnostics: diagnostics,
        thermal_driver_states,
        driver_thermal_states,
        power_ledger,
        woofer_electrical_impedance_ohm,
        tweeter_electrical_impedance_ohm,
        driver_electrical_impedances_ohm: std::mem::take(&mut driver_electrical_impedances_ohm),
    })
}

fn solve_complex_linear_system(
    mut matrix: Vec<Vec<Complex64>>,
    mut rhs: Vec<Complex64>,
    frequency: f64,
) -> Result<Vec<Complex64>, SimulationError> {
    let n = rhs.len();
    if n == 0 || matrix.len() != n || matrix.iter().any(|row| row.len() != n) {
        return Err(SimulationError::new(format!(
            "invalid typed-driver network dimensions at {frequency:.3} Hz"
        )));
    }
    for pivot in 0..n {
        let row = (pivot..n)
            .max_by(|&left, &right| {
                matrix[left][pivot]
                    .norm()
                    .total_cmp(&matrix[right][pivot].norm())
            })
            .expect("pivot range is non-empty");
        if matrix[row][pivot].norm() <= f64::MIN_POSITIVE {
            return Err(SimulationError::new(format!(
                "coupled driver network is singular at {frequency:.3} Hz"
            )));
        }
        if row != pivot {
            matrix.swap(row, pivot);
            rhs.swap(row, pivot);
        }
        let pivot_tail = matrix[pivot][(pivot + 1)..n].to_vec();
        let pivot_rhs = rhs[pivot];
        for row in (pivot + 1)..n {
            let factor = matrix[row][pivot] / matrix[pivot][pivot];
            matrix[row][pivot] = Complex64::default();
            for column in (pivot + 1)..n {
                matrix[row][column] -= factor * pivot_tail[column - pivot - 1];
            }
            rhs[row] -= factor * pivot_rhs;
        }
    }
    let mut solution = vec![Complex64::default(); n];
    for row in (0..n).rev() {
        let tail = ((row + 1)..n)
            .map(|column| matrix[row][column] * solution[column])
            .sum::<Complex64>();
        solution[row] = (rhs[row] - tail) / matrix[row][row];
        if !solution[row].re.is_finite() || !solution[row].im.is_finite() {
            return Err(SimulationError::new(format!(
                "coupled driver network produced a non-finite velocity at {frequency:.3} Hz"
            )));
        }
    }
    Ok(solution)
}

/// Reciprocity and passivity gate for the acoustic two-port.
///
/// Constant-element Burton–Miller can have small discretization defects, but
/// a violation beyond the accepted physical tolerance is a coupled-state
/// failure, not a warning: accepting it would allow an active or
/// non-reciprocal load into the driver network.
#[cfg(test)]
fn diagnose_radiation_matrix(
    frequency: f64,
    z_rad: &[[Complex64; 2]; 2],
) -> Result<RadiationMatrixDiagnostics, SimulationError> {
    let matrix =
        RadiationMatrix::new(vec![z_rad[0].to_vec(), z_rad[1].to_vec()]).map_err(|error| {
            SimulationError::new(format!(
                "invalid radiation matrix at {frequency:.3} Hz: {error}"
            ))
        })?;
    matrix.validate(0.10, 0.05).map_err(|error| {
        SimulationError::new(format!(
            "radiation matrix physical gate failed at {frequency:.3} Hz: {error}"
        ))
    })
}

/// Energy-closure diagnostic: the coupled state must radiate, not absorb,
/// acoustic power. With RMS phasors the mean radiated power is Re(vᴴ Z v).
#[cfg(test)]
fn diagnose_radiated_power(
    frequency: f64,
    z_rad: &[[Complex64; 2]; 2],
    velocities_rms: &[Complex64; 2],
) -> Result<(), SimulationError> {
    let matrix =
        RadiationMatrix::new(vec![z_rad[0].to_vec(), z_rad[1].to_vec()]).map_err(|error| {
            SimulationError::new(format!(
                "invalid radiation matrix at {frequency:.3} Hz: {error}"
            ))
        })?;
    let power = matrix
        .active_power_w(velocities_rms)
        .map_err(|error| SimulationError::new(error.to_string()))?;
    let scale = velocities_rms
        .iter()
        .map(|velocity| velocity.norm_sqr())
        .sum::<f64>()
        .max(1.0e-12)
        * matrix
            .values()
            .iter()
            .flatten()
            .map(|value| value.norm())
            .fold(0.0, f64::max)
            .max(1.0e-12);
    if power < -0.05 * scale.max(1.0e-12) {
        return Err(SimulationError::new(format!(
            "coupled drivers absorb {power:.3e} W of acoustic power at {frequency:.3} Hz (scale {scale:.3e}); radiation load is not passive"
        )));
    }
    Ok(())
}

fn diagnose_radiation_matrix_generic(
    frequency: f64,
    z_rad: &[Vec<Complex64>],
) -> Result<RadiationMatrixDiagnostics, SimulationError> {
    let matrix = RadiationMatrix::new(z_rad.to_vec()).map_err(|error| {
        SimulationError::new(format!(
            "invalid radiation matrix at {frequency:.3} Hz: {error}"
        ))
    })?;
    matrix.validate(0.10, 0.05).map_err(|error| {
        SimulationError::new(format!(
            "radiation matrix physical gate failed at {frequency:.3} Hz: {error}"
        ))
    })
}

fn diagnose_radiated_power_generic(
    frequency: f64,
    z_rad: &[Vec<Complex64>],
    velocities_rms: &[Complex64],
) -> Result<(), SimulationError> {
    let matrix = RadiationMatrix::new(z_rad.to_vec()).map_err(|error| {
        SimulationError::new(format!(
            "invalid radiation matrix at {frequency:.3} Hz: {error}"
        ))
    })?;
    let power = matrix
        .active_power_w(velocities_rms)
        .map_err(|error| SimulationError::new(error.to_string()))?;
    let scale = velocities_rms
        .iter()
        .map(|velocity| velocity.norm_sqr())
        .sum::<f64>()
        .max(1.0e-12)
        * matrix
            .values()
            .iter()
            .flatten()
            .map(|value| value.norm())
            .fold(0.0, f64::max)
            .max(1.0e-12);
    if power < -0.05 * scale.max(1.0e-12) {
        return Err(SimulationError::new(format!(
            "coupled drivers absorb {power:.3e} W of acoustic power at {frequency:.3} Hz"
        )));
    }
    Ok(())
}

#[cfg(test)]
fn radiation_matrix_power(z_rad: &[[Complex64; 2]; 2], velocities: &[Complex64; 2]) -> f64 {
    (0..2)
        .flat_map(|i| (0..2).map(move |j| (i, j)))
        .map(|(i, j)| velocities[i].conj() * z_rad[i][j] * velocities[j])
        .sum::<Complex64>()
        .re
}

fn radiation_matrix_power_generic(z_rad: &[Vec<Complex64>], velocities: &[Complex64]) -> f64 {
    (0..velocities.len())
        .flat_map(|i| (0..velocities.len()).map(move |j| (i, j)))
        .map(|(i, j)| velocities[i].conj() * z_rad[i][j] * velocities[j])
        .sum::<Complex64>()
        .re
}

// ---------------------------------------------------------------------------
// Build surface velocity vector from element BCs
// ---------------------------------------------------------------------------

pub(crate) fn build_surface_velocity(elements: &[Element]) -> Vec<Complex64> {
    let n = elements
        .iter()
        .filter(|e| !e.property.is_evaluation())
        .count();
    let mut v = vec![Complex64::new(0.0, 0.0); n];
    let mut idx = 0;
    for elem in elements {
        if elem.property.is_evaluation() {
            continue;
        }
        match &elem.boundary_condition {
            BoundaryCondition::Velocity(vals) => {
                v[idx] = vals.first().copied().unwrap_or(Complex64::new(0.0, 0.0));
            }
            BoundaryCondition::VelocityWithAdmittance { velocity, .. } => {
                v[idx] = velocity
                    .first()
                    .copied()
                    .unwrap_or(Complex64::new(0.0, 0.0));
            }
            _ => {
                v[idx] = Complex64::new(0.0, 0.0);
            }
        }
        idx += 1;
    }
    v
}

// ---------------------------------------------------------------------------
// Field map: single-frequency BEM solve on a grid
// ---------------------------------------------------------------------------

/// Result of a field-map evaluation on a 2D grid.
pub struct FieldMapResult {
    /// Frequency (Hz).
    pub freq: f64,
    /// Grid coordinates (N points).
    pub points: Vec<[f64; 3]>,
    /// SPL (dB re 20 µPa) at each grid point.
    pub spl: Vec<f64>,
    /// Grid dimensions (nx, ny) for the cut plane.
    pub grid_dims: (usize, usize),
}

/// Run BEM at a single frequency and evaluate SPL on a grid in the xz-plane (y = baffle_center_y).
///
/// The grid spans `[-extent, extent]` in x and `[baffle_z - margin, baffle_z + extent]` in z,
/// with `resolution` spacing. Points inside or very close to the cabinet are excluded (NaN).
pub fn run_field_map(
    model: &LoudspeakerModel,
    freq: f64,
    extent: f64,
    resolution: f64,
) -> Result<FieldMapResult, SimulationError> {
    try_run_field_map(model, freq, extent, resolution)
}

pub fn try_run_field_map(
    model: &LoudspeakerModel,
    freq: f64,
    extent: f64,
    resolution: f64,
) -> Result<FieldMapResult, SimulationError> {
    model
        .validate()
        .map_err(|message| SimulationError::new(format!("invalid loudspeaker model: {message}")))?;
    validate_field_map_inputs(freq, extent, resolution)?;

    let c = model.simulation.speed_of_sound;
    let rho = model.simulation.air_density;
    let baffle = model.baffle_center();

    // Build grid in xz-plane at y = baffle_center_y
    let half_w = model.cabinet.width / 2.0;
    let cab_z_back = baffle[2] - model.cabinet.depth;
    let cab_z_front = baffle[2];

    let x_min = -extent;
    let x_max = extent;
    let z_min = baffle[2] - extent * 0.5;
    let z_max = baffle[2] + extent;

    let nx = checked_grid_dimension(x_max - x_min, resolution)?;
    let nz = checked_grid_dimension(z_max - z_min, resolution)?;
    let grid_point_count = nx
        .checked_mul(nz)
        .filter(|&count| count <= 10_000_000)
        .ok_or_else(|| SimulationError::new("field-map grid is too large"))?;

    let mut grid_points = Vec::with_capacity(grid_point_count);
    let mut inside = Vec::with_capacity(grid_point_count);

    for iz in 0..nz {
        let z = z_min + iz as f64 * resolution;
        for ix in 0..nx {
            let x = x_min + ix as f64 * resolution;
            grid_points.push([x, baffle[1], z]);
            // Mark points inside/on the cabinet surface (with small margin)
            let margin = resolution * 0.5;
            let in_box =
                x.abs() < half_w + margin && z > cab_z_back - margin && z < cab_z_front + margin;
            inside.push(in_box);
        }
    }

    // Collect exterior points for BEM evaluation
    let exterior_indices: Vec<usize> = (0..grid_points.len()).filter(|&i| !inside[i]).collect();
    let eval_points: Vec<[f64; 3]> = exterior_indices.iter().map(|&i| grid_points[i]).collect();
    let n_eval = eval_points.len();
    validate_evaluation_points(&eval_points)?;

    // Build BEM mesh & solve
    let speaker_mesh = super::mesh_gen::generate_loudspeaker_mesh(model);
    let (nodes, mut elements) = speaker_mesh_to_bem(&speaker_mesh, model)?;
    update_driver_bcs(&mut elements, &speaker_mesh, model, freq, None)?;

    let physics = PhysicsParams::new(freq, c, rho, false);
    let solver_config = SolverConfig::default();
    validate_solver_config(&solver_config)?;
    let n_dofs = elements
        .iter()
        .filter(|element| !element.property.is_evaluation())
        .map(|element| element.dof_addresses.len())
        .sum::<usize>();
    let strategy = select_solver_strategy(n_dofs, &solver_config);
    let solver_label = match strategy {
        SolverStrategy::Direct => "LU",
        SolverStrategy::DenseIterative => "FGMRES+ILU",
        SolverStrategy::HMatrixIterative => "Hmatrix+FGMRES",
    };
    eprintln!(
        "Field map: {} DOFs ({}), {} eval points, f={:.0} Hz",
        n_dofs, solver_label, n_eval, freq,
    );

    let surface_pressure = if strategy == SolverStrategy::HMatrixIterative {
        let rhs = build_tbem_rhs_stabilized(
            &elements,
            &nodes,
            &physics,
            EXTERIOR_TBEM_SIGN,
            solver_config.beta_scale,
        );
        let centers: Vec<[f64; 3]> = elements.iter().map(|element| element.center).collect();
        let mut octree = Octree::build(&centers, 32, 8);
        octree.compute_interaction_lists(2.0);
        let clusters = clusters_from_octree(&elements, &octree);
        let hmatrix =
            build_simulation_hmatrix(&elements, &nodes, &clusters, &physics, &solver_config);
        let (solution, residual) =
            solve_bem_system_hmatrix(&hmatrix, &rhs, None, &solver_config, None)?;
        let mut sentinel = ExactResidualSentinel::default();
        if sentinel.should_verify_exact(&solver_config, residual) {
            verify_exact_hmatrix_solution(
                &elements,
                &nodes,
                &physics,
                &solution,
                &rhs,
                &solver_config,
            )?;
        }
        solution
    } else {
        let system = build_tbem_system_stabilized(
            &elements,
            &nodes,
            &physics,
            EXTERIOR_TBEM_SIGN,
            solver_config.beta_scale,
        );
        solve_bem_system(&system.matrix, &system.rhs, None, &solver_config, None)?
    };

    let surface_velocity = build_surface_velocity(&elements);
    let p_scattered = compute_scattered_field(
        &eval_points,
        &elements,
        &nodes,
        &surface_pressure,
        Some(&surface_velocity),
        &physics,
        BasisOrder::Constant,
    );
    if !complex_values_are_finite(&p_scattered) {
        return Err(SimulationError::new(
            "field-map evaluation produced non-finite pressure",
        ));
    }

    // Map SPL back onto full grid (NaN for interior points). The BEM
    // returns complex pressure amplitudes; convert to dB SPL via the
    // canonical RMS-referenced helper.
    let mut spl = vec![f64::NAN; grid_points.len()];
    for (eval_i, &grid_i) in exterior_indices.iter().enumerate() {
        let p = p_scattered[eval_i];
        spl[grid_i] = sonium_xem::pressure_to_spl(p);
        if !spl[grid_i].is_finite() {
            return Err(SimulationError::new(format!(
                "field-map evaluation produced non-finite SPL at grid point {grid_i}"
            )));
        }
    }

    Ok(FieldMapResult {
        freq,
        points: grid_points,
        spl,
        grid_dims: (nx, nz),
    })
}

fn checked_grid_dimension(span: f64, resolution: f64) -> Result<usize, SimulationError> {
    let intervals = (span / resolution).ceil();
    if !intervals.is_finite() || intervals < 1.0 || intervals > usize::MAX as f64 - 1.0 {
        return Err(SimulationError::new("field-map grid dimension is invalid"));
    }
    Ok(intervals as usize + 1)
}

/// Write a field map to a Gmsh `.pos` (post-processing view) file.
pub fn write_field_pos(
    result: &FieldMapResult,
    path: impl AsRef<std::path::Path>,
) -> std::io::Result<()> {
    use std::io::Write as W;
    let mut f = std::fs::File::create(path)?;

    writeln!(f, "View \"SPL at {:.0} Hz\" {{", result.freq)?;
    for (i, pt) in result.points.iter().enumerate() {
        let v = result.spl[i];
        if v.is_finite() {
            writeln!(
                f,
                "SP({:.6e},{:.6e},{:.6e}){{{:.4}}};",
                pt[0], pt[1], pt[2], v
            )?;
        }
    }
    writeln!(f, "}};")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::model::*;

    pub(crate) fn tiny_model() -> LoudspeakerModel {
        LoudspeakerModel {
            cabinet: CabinetDimensions {
                height: 0.40,
                width: 0.22,
                depth: 0.28,
            },
            woofer: DriverConfig {
                radius: 0.065,
                x_offset: 0.0,
                y_offset: -0.08,
                piston_velocity: 0.001,
                transfer_function: None,
                driver_model: None,
                cone: None,
            },
            tweeter: DriverConfig {
                radius: 0.013,
                x_offset: 0.0,
                y_offset: 0.10,
                piston_velocity: 0.001,
                transfer_function: None,
                driver_model: None,
                cone: None,
            },
            additional_drivers: Vec::new(),
            product_design: None,
            separator: SeparatorConfig {
                y_position: 0.02,
                thickness: 0.018,
            },
            waveguide: Some(WaveguideConfig {
                horizontal_spline: vec![[0.0, 0.05], [0.5, 0.03], [1.0, 0.013]],
                vertical_spline: vec![[0.0, 0.04], [0.5, 0.025], [1.0, 0.013]],
                depth: 0.025,
            }),
            materials: SpeakerMaterialAssignment::default(),
            simulation: SimulationConfig {
                freq_start: 1000.0,
                freq_end: 1000.0,
                freq_points: 1,
                speed_of_sound: 343.0,
                air_density: 1.21,
                max_mesh_frequency: Some(2000.0), // coarse mesh
            },
        }
    }

    #[test]
    fn gpu_capability_matrix_publishes_native_and_fail_closed_phases() {
        let backend = sonium_compute::CpuBackend::new();
        let matrix = GpuCapabilityMatrix::from_backend(&backend);

        assert_eq!(matrix.regular_pair_assembly_c64, GpuCapabilityState::Native);
        assert_eq!(matrix.field_evaluation, GpuCapabilityState::Native);
        assert_eq!(
            matrix.speaker_configurations.prescribed_forward.state,
            GpuCapabilityState::Native
        );
        assert_eq!(matrix.coupled_driver_solve, GpuCapabilityState::Unsupported);
        assert_eq!(matrix.derivative_assembly, GpuCapabilityState::Unsupported);
        assert_eq!(matrix.adjoint_solve, GpuCapabilityState::Unsupported);
        assert_eq!(matrix.reduced_order_model, GpuCapabilityState::Unsupported);
        assert_eq!(
            matrix.cea2034_postprocess,
            GpuCapabilityState::HostReference
        );
        assert_eq!(
            matrix.speaker_configurations.typed_multi_driver.state,
            GpuCapabilityState::Unsupported
        );
        assert_eq!(
            matrix.speaker_configurations.crossover_phase.state,
            GpuCapabilityState::Unsupported
        );
        assert_eq!(
            matrix.speaker_configurations.coordinate_rotation.state,
            GpuCapabilityState::Unsupported
        );
        assert_eq!(
            matrix
                .speaker_configurations
                .body_fitted_projected_reanalysis
                .state,
            GpuCapabilityState::Unsupported
        );
        serde_json::to_string(&matrix).expect("capability matrix must be serializable");
    }

    #[test]
    fn prescribed_forward_capability_requires_every_device_panel_route() {
        use sonium_compute::CapabilityState;

        assert_eq!(
            collapse_gpu_capability_states(&[
                CapabilityState::Native,
                CapabilityState::Native,
                CapabilityState::Unsupported,
            ]),
            GpuCapabilityState::Unsupported
        );
        assert_eq!(
            collapse_gpu_capability_states(&[
                CapabilityState::Native,
                CapabilityState::VendorNative,
            ]),
            GpuCapabilityState::VendorNative
        );
        assert_eq!(
            collapse_gpu_capability_states(&[
                CapabilityState::Native,
                CapabilityState::HostSharedFallback,
            ]),
            GpuCapabilityState::HostSharedFallback
        );
    }

    #[cfg(any(feature = "cuda", feature = "metal"))]
    #[test]
    fn accelerator_parity_differences_reject_non_finite_values() {
        assert!(
            relative_complex_difference(Complex64::new(f64::NAN, 0.0), Complex64::new(1.0, 0.0))
                .is_infinite(),
            "non-finite coupled complex values must fail closed"
        );
        assert!(
            relative_scalar_difference(f64::NAN, 1.0).is_infinite(),
            "non-finite coupled scalar values must fail closed"
        );
        let mut actual = PowerLedger {
            supplied_w: f64::NAN,
            copper_loss_w: 0.0,
            magnetic_loss_w: 0.0,
            mechanical_loss_w: 0.0,
            structural_loss_w: 0.0,
            internal_acoustic_loss_w: 0.0,
            exterior_acoustic_power_w: 0.0,
            thermal_storage_w: 0.0,
            residual_w: 0.0,
        };
        let reference = PowerLedger {
            supplied_w: 1.0,
            copper_loss_w: 0.0,
            magnetic_loss_w: 0.0,
            mechanical_loss_w: 0.0,
            structural_loss_w: 0.0,
            internal_acoustic_loss_w: 0.0,
            exterior_acoustic_power_w: 0.0,
            thermal_storage_w: 0.0,
            residual_w: 0.0,
        };
        assert!(
            power_ledger_difference(actual, reference).is_infinite(),
            "non-finite power-ledger values must fail closed"
        );
        actual.supplied_w = 1.0;
        actual.thermal_storage_w = f64::NAN;
        assert!(
            power_ledger_difference(actual, reference).is_infinite(),
            "thermal storage must be included in the power-ledger parity gate"
        );
    }

    #[cfg(any(feature = "cuda", feature = "metal"))]
    #[test]
    fn accelerator_product_configuration_is_rejected_before_device_initialization() {
        let mut model = tiny_model();
        model.product_design = Some(sonium_sim_speaker_modeling::ProductDesign {
            cabinet: sonium_sim_speaker_modeling::CabinetDesign::default(),
            drivers: Vec::new(),
            crossover: sonium_sim_speaker_modeling::CrossoverNetwork {
                branches: Vec::new(),
            },
            waveguide: None,
            structural_acoustic: None,
        });

        for backend in [SimulationBackend::Cuda, SimulationBackend::Metal] {
            let error = validate_accelerator_speaker_configuration(backend, &model).unwrap_err();
            assert!(error.to_string().contains("ProductDesign configurations"));
        }

        #[cfg(feature = "cuda")]
        {
            let error = match run_simulation_cuda_on_device(
                &model,
                &[[0.0, 0.0, 1.0]],
                &SolverConfig::default(),
                usize::MAX,
            ) {
                Ok(_) => panic!("ProductDesign CUDA entrypoint unexpectedly succeeded"),
                Err(error) => error,
            };
            assert!(error.to_string().contains("ProductDesign configurations"));
        }
    }

    #[test]
    fn gpu_residency_identity_rejects_backend_geometry_and_grid_changes() {
        let identity = GpuResidencyIdentity {
            schema_version: GpuResidencyIdentity::SCHEMA_VERSION,
            backend: SimulationBackend::Cuda,
            precision: PrecisionPolicy::MixedF32InnerF64Gated,
            device_index: Some(0),
            geometry_key: 11,
            frequency_grid_key: 22,
        };
        assert!(identity.is_compatible(
            SimulationBackend::Cuda,
            PrecisionPolicy::MixedF32InnerF64Gated,
            Some(0),
            11,
            22,
        ));
        assert!(!identity.is_compatible(
            SimulationBackend::Cuda,
            PrecisionPolicy::MixedF32InnerF64Gated,
            Some(0),
            12,
            22,
        ));
        assert!(!identity.is_compatible(
            SimulationBackend::Cuda,
            PrecisionPolicy::MixedF32InnerF64Gated,
            Some(0),
            11,
            23,
        ));
        assert!(!identity.is_compatible(
            SimulationBackend::Metal,
            PrecisionPolicy::MixedF32InnerF64Gated,
            Some(0),
            11,
            22,
        ));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_adjoint_rejects_invalid_objective_before_device_access() {
        let model = tiny_model();
        let points = [[0.0, 0.0, 1.0]];
        let config = SolverConfig::default();
        let error = run_simulation_cuda_adjoint_with_report(&model, &points, &[], &config, 0)
            .expect_err("dimensionally invalid adjoint objective must fail closed");
        assert!(error.to_string().contains("objective has 0 weights"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn native_typed_network_retains_krylov_across_frequency_updates() {
        let backend = match sonium_compute::CudaBackend::new_default() {
            Ok(backend) => backend,
            Err(error) => {
                assert!(
                    std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                    "{error}"
                );
                return;
            }
        };
        let pool =
            sonium_compute::MemoryPool::new(backend, sonium_compute::MemoryPoolConfig::default())
                .unwrap();
        let mut model = tiny_model();
        let study = sonium_sim_speaker_modeling::SpeakerDriverStudy::detailed_frequency_domain_driver();
        let driver = sonium_sim_speaker_modeling::DriverModel::from_study(&study).unwrap();
        model.woofer.radius = study.geometry.diaphragm_radius_m;
        model.tweeter.radius = study.geometry.diaphragm_radius_m;
        model.woofer.driver_model = Some(driver.clone());
        model.tweeter.driver_model = Some(driver);
        let radiation = vec![
            vec![Complex64::new(0.2, 0.1), Complex64::new(0.03, 0.01)],
            vec![Complex64::new(0.03, 0.01), Complex64::new(0.4, 0.3)],
        ];
        let mut workspace = sonium_compute::GmresWorkspace::new(model.drivers().len());
        let (mut matrix, mut preconditioner, mut rhs, mut solution) = (None, None, None, None);
        let (mut matrix_updates, mut preconditioner_updates) = (0, 0);
        let config = SolverConfig::default();
        for (index, frequency) in [1000.0, 1000.0, 1200.0, 1200.0, 1000.0]
            .into_iter()
            .enumerate()
        {
            let before = pool.backend().begin_runtime_stats();
            let actual = solve_gpu_linear_driver_network(
                &pool,
                &mut workspace,
                &model,
                frequency,
                &radiation,
                &config,
                &mut matrix,
                &mut preconditioner,
                &mut rhs,
                &mut solution,
                &mut matrix_updates,
                &mut preconditioner_updates,
            )
            .unwrap();
            let stats = pool.backend().runtime_stats_since(before);
            if index > 0 {
                assert_eq!(stats.allocation_operations, 0, "{stats:?}");
            }
            let expected =
                solve_coupled_driver_network_generic_with_control(&model, frequency, &radiation, None)
                    .unwrap();
            for (actual, expected) in actual
                .velocities_bem_all
                .iter()
                .zip(&expected.velocities_bem_all)
            {
                assert!(relative_complex_difference(*actual, *expected) < 1e-5);
            }
            assert!(workspace.cached_buffer_count().unwrap() > 0);
        }
        assert_eq!(matrix_updates, 4);
        assert_eq!(preconditioner_updates, 4);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn typed_gpu_network_contract_matches_cpu_reference_backend() {
        let mut model = tiny_model();
        let study =
            sonium_sim_speaker_modeling::SpeakerDriverStudy::detailed_frequency_domain_driver();
        let driver = sonium_sim_speaker_modeling::DriverModel::from_study(&study).unwrap();
        model.woofer.radius = study.geometry.diaphragm_radius_m;
        model.tweeter.radius = study.geometry.diaphragm_radius_m;
        model.woofer.driver_model = Some(driver.clone());
        model.tweeter.driver_model = Some(driver);
        let radiation = vec![
            vec![Complex64::new(0.2, 0.1), Complex64::new(0.03, 0.01)],
            vec![Complex64::new(0.03, 0.01), Complex64::new(0.4, 0.3)],
        ];
        let config = SolverConfig::default();
        let pool = sonium_compute::MemoryPool::new(
            sonium_compute::CpuBackend::new(),
            sonium_compute::MemoryPoolConfig {
                // The CPU test backend does not report a device memory capacity.
                max_bytes: Some(32 * 1024 * 1024),
            },
        )
        .unwrap();
        let mut workspace = sonium_compute::GmresWorkspace::new(model.drivers().len());
        let mut matrix = None;
        let mut preconditioner = None;
        let mut rhs = None;
        let mut solution = None;
        let mut matrix_updates = 0;
        let mut preconditioner_updates = 0;
        let accelerator = solve_gpu_linear_driver_network(
            &pool,
            &mut workspace,
            &model,
            1_000.0,
            &radiation,
            &config,
            &mut matrix,
            &mut preconditioner,
            &mut rhs,
            &mut solution,
            &mut matrix_updates,
            &mut preconditioner_updates,
        )
        .unwrap();
        let reference =
            solve_coupled_driver_network_generic_with_control(&model, 1_000.0, &radiation, None)
                .unwrap();
        for (actual, expected) in accelerator
            .velocities_bem_all
            .iter()
            .zip(&reference.velocities_bem_all)
        {
            assert!(relative_complex_difference(*actual, *expected) < 1.0e-5);
        }
        assert!(accelerator.power_ledger.is_some());
        assert!(
            accelerator
                .radiation_matrix_diagnostics
                .reciprocity_defect
                .is_finite()
        );
    }

    #[test]
    fn simulation_control_reports_monotonic_progress_and_cancellation() {
        let control = SimulationControl::new(4);
        assert_eq!(
            control.progress(),
            SimulationProgress {
                completed: 0,
                total: 4,
                frequency_hz: None,
            }
        );
        control.report_frequency(2, 1000.0);
        control.report_frequency(1, 500.0);
        let progress = control.progress();
        assert_eq!(progress.completed, 2);
        assert_eq!(progress.frequency_hz, Some(500.0));
        assert!(!control.is_cancelled());
        control.cancel();
        assert!(control.is_cancelled());
        assert!(control.check().unwrap_err().is_cancelled());
    }

    #[test]
    fn parallel_progress_counts_successes_in_completion_order() {
        let control = SimulationControl::new(4);
        // A high-frequency lane finishing first must not mark lower frequencies done.
        control.complete_frequency(4000.0);
        assert_eq!(control.progress().completed, 1);
        control.complete_frequency(1000.0);
        assert_eq!(control.progress().completed, 2);
    }

    #[test]
    fn pre_cancelled_control_exits_before_native_mesh_work() {
        let model = tiny_model();
        let control = SimulationControl::new(1);
        control.cancel();
        let result = run_simulation_with_control(
            &model,
            &[[0.0, 0.0, 1.0]],
            &SolverConfig::default(),
            Some(&control),
        );
        match result {
            Err(error) => assert!(error.is_cancelled()),
            Ok(_) => panic!("pre-cancelled simulation unexpectedly succeeded"),
        }
        assert_eq!(control.progress().completed, 0);
    }

    #[test]
    fn invalid_speaker_material_is_transactional() {
        let mut model = tiny_model();
        let mesh = crate::mesh_gen::generate_loudspeaker_mesh(&model);
        let (_, mut elements) = speaker_mesh_to_bem(&mesh, &model).unwrap();
        let before = format!("{elements:?}");
        model.materials.waveguide = SpeakerMaterial::ConstantImpedance {
            specific_impedance: 0.0,
        };
        assert!(speaker_mesh_to_bem(&mesh, &model).is_err());
        assert!(update_driver_bcs(&mut elements, &mesh, &model, 100.0, None).is_err());
        assert_eq!(format!("{elements:?}"), before);
        model.materials.waveguide = SpeakerMaterial::hard_plastic();
        let mut malformed = mesh.clone();
        malformed.region.pop();
        assert!(speaker_mesh_to_bem(&malformed, &model).is_err());
        assert!(update_driver_bcs(&mut elements, &malformed, &model, 100.0, None).is_err());
        assert_eq!(format!("{elements:?}"), before);
    }

    #[test]
    fn test_speaker_mesh_to_bem() {
        let model = tiny_model();
        let speaker_mesh = crate::mesh_gen::generate_loudspeaker_mesh(&model);
        let (nodes, elements) = speaker_mesh_to_bem(&speaker_mesh, &model).unwrap();

        assert!(!nodes.is_empty());
        assert_eq!(elements.len(), speaker_mesh.triangles.len());

        // All elements should have valid geometry
        for elem in &elements {
            assert!(elem.area > 0.0, "element has zero area");
            let n = elem.normal;
            let n_len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((n_len - 1.0).abs() < 1e-6, "normal not unit: len={n_len}");
        }
    }

    #[test]
    fn test_surface_velocity_vector() {
        let model = tiny_model();
        let speaker_mesh = crate::mesh_gen::generate_loudspeaker_mesh(&model);
        let (_nodes, mut elements) = speaker_mesh_to_bem(&speaker_mesh, &model).unwrap();

        // BCs are initialized to zero; update for a specific frequency
        update_driver_bcs(&mut elements, &speaker_mesh, &model, 1000.0, None).unwrap();

        let v = build_surface_velocity(&elements);
        assert_eq!(v.len(), elements.len());

        // Most elements should have zero velocity (rigid walls)
        let nonzero_count = v.iter().filter(|&&vi| vi.norm() > 1e-15).count();
        let woofer_count = speaker_mesh
            .region
            .iter()
            .filter(|&&r| r == MeshRegion::WooferAperture)
            .count();
        let tweeter_count = speaker_mesh
            .region
            .iter()
            .filter(|&&r| r == MeshRegion::TweeterAperture)
            .count();

        assert_eq!(nonzero_count, woofer_count + tweeter_count);
    }

    #[test]
    fn test_porous_material_admittance_updates_with_frequency() {
        let mut model = tiny_model();
        model.materials.cabinet = SpeakerMaterial::PorousLayer {
            layer: sonium_bem::materials::PorousLayer::miki(0.05, 12_000.0),
        };
        let speaker_mesh = crate::mesh_gen::generate_loudspeaker_mesh(&model);
        let (_nodes, mut elements) = speaker_mesh_to_bem(&speaker_mesh, &model).unwrap();

        update_driver_bcs(&mut elements, &speaker_mesh, &model, 125.0, None).unwrap();
        let low = first_admittance_for_region(&elements, &speaker_mesh, MeshRegion::CabinetWall);

        update_driver_bcs(&mut elements, &speaker_mesh, &model, 2_000.0, None).unwrap();
        let high = first_admittance_for_region(&elements, &speaker_mesh, MeshRegion::CabinetWall);

        assert!(low.re >= 0.0);
        assert!(high.re >= 0.0);
        assert!((high - low).norm() > 1e-3);
    }

    fn first_admittance_for_region(
        elements: &[Element],
        speaker_mesh: &SpeakerMesh,
        region: MeshRegion,
    ) -> Complex64 {
        elements
            .iter()
            .zip(speaker_mesh.region.iter())
            .find_map(|(elem, &elem_region)| {
                if elem_region != region {
                    return None;
                }
                match elem.boundary_condition {
                    BoundaryCondition::VelocityWithAdmittance { admittance, .. } => {
                        Some(admittance)
                    }
                    _ => None,
                }
            })
            .expect("region should have an admittance boundary condition")
    }

    #[test]
    fn test_relative_residual_exact_solution() {
        let matrix = DenseMatrix::from_row_major_vec(
            2,
            2,
            vec![
                Complex64::new(2.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(3.0, 0.0),
            ],
        );
        let x = vec![Complex64::new(0.5, 0.0), Complex64::new(2.0, 0.0)];
        let rhs = vec![Complex64::new(1.0, 0.0), Complex64::new(6.0, 0.0)];

        assert!(relative_residual(&matrix, &x, &rhs) < 1e-14);
    }

    #[test]
    fn test_true_residual_gate_uses_requested_tolerance() {
        let config = SolverConfig {
            tolerance: 1.0e-6,
            ..SolverConfig::default()
        };
        assert!(validate_true_residual("test solve", 1.0e-6, &config).is_ok());
        assert!(validate_true_residual("test solve", 1.01e-6, &config).is_err());
        assert!(validate_true_residual("test solve", f64::NAN, &config).is_err());
    }

    #[test]
    fn test_default_solver_limits_dense_lu_to_small_systems() {
        assert_eq!(
            SolverConfig::default().direct_threshold,
            DEFAULT_DIRECT_THRESHOLD
        );
    }

    #[test]
    fn test_default_solver_recycling_covers_small_restart() {
        // SIM-PERF-20261013.md section 2, gap 4: small restarts stall
        // without deflation, so the default restart-40 requires GCRO-DR
        // recycling on with room inside the window.
        let config = SolverConfig::default();
        assert!(validate_solver_config(&config).is_ok());
        assert!(config.recycle_subspace_size > 0);
        assert!(config.recycle_subspace_size < config.restart);
    }

    #[test]
    fn test_solver_routing_honors_direct_threshold_before_hmatrix_threshold() {
        let config = SolverConfig::default();
        assert_eq!(
            select_solver_strategy(2_000, &config),
            SolverStrategy::Direct
        );
        assert_eq!(
            select_solver_strategy(4_000, &config),
            SolverStrategy::DenseIterative
        );
        assert_eq!(
            select_solver_strategy(config.hmatrix_threshold + 1, &config),
            SolverStrategy::HMatrixIterative
        );
    }

    #[test]
    fn test_solver_config_rejects_non_finite_or_inconsistent_values() {
        let invalid = [
            SolverConfig {
                tolerance: f64::NAN,
                ..SolverConfig::default()
            },
            SolverConfig {
                tolerance: 0.0,
                ..SolverConfig::default()
            },
            SolverConfig {
                beta_scale: f64::INFINITY,
                ..SolverConfig::default()
            },
            SolverConfig {
                restart: 0,
                ..SolverConfig::default()
            },
            SolverConfig {
                max_iterations: 0,
                ..SolverConfig::default()
            },
            SolverConfig {
                ilu_shift: -1.0,
                ..SolverConfig::default()
            },
            SolverConfig {
                direct_threshold: 5_000,
                hmatrix_threshold: 5_000,
                ..SolverConfig::default()
            },
        ];
        for config in &invalid {
            assert!(validate_solver_config(config).is_err());
        }
    }

    #[test]
    fn test_evaluation_point_validation_rejects_empty_and_non_finite_points() {
        assert!(validate_evaluation_points(&[]).is_err());
        assert!(validate_evaluation_points(&[[0.0, f64::NAN, 1.0]]).is_err());
        assert!(validate_evaluation_points(&[[0.0, 0.0, 1.0]]).is_ok());
    }

    #[test]
    fn test_field_map_input_validation_rejects_invalid_grid() {
        assert!(validate_field_map_inputs(0.0, 1.0, 0.02).is_err());
        assert!(validate_field_map_inputs(1000.0, -1.0, 0.02).is_err());
        assert!(validate_field_map_inputs(1000.0, 1.0, 0.0).is_err());
        assert!(validate_field_map_inputs(1000.0, 1.0, f64::NAN).is_err());
    }

    #[test]
    fn test_initial_guess_recycling_requires_previous_frequency() {
        assert!(!should_recycle_initial_guess(None, 100.0, 0.25));
    }

    #[test]
    fn test_initial_guess_recycling_accepts_close_frequency_step() {
        let quarter_octave = 100.0 * 2f64.powf(0.25);

        assert!(should_recycle_initial_guess(
            Some(100.0),
            quarter_octave,
            0.25
        ));
        assert!(should_recycle_initial_guess(
            Some(quarter_octave),
            100.0,
            0.25
        ));
    }

    #[test]
    fn test_initial_guess_recycling_rejects_large_frequency_jump() {
        assert!(!should_recycle_initial_guess(Some(100.0), 200.0, 0.25));
        assert!(!should_recycle_initial_guess(Some(100.0), 0.0, 0.25));
        assert!(!should_recycle_initial_guess(Some(100.0), 110.0, f64::NAN));
    }

    #[test]
    fn test_large_frequency_jump_resets_gcro_dr_subspace() {
        use sonium_solvers::preconditioners::DiagonalPreconditioner;

        let matrix = DenseMatrix::from_row_major_vec(
            2,
            2,
            vec![
                Complex64::new(2.0, 0.0),
                Complex64::new(0.2, 0.0),
                Complex64::new(0.2, 0.0),
                Complex64::new(3.0, 0.0),
            ],
        );
        let operator = DenseOperator { matrix: &matrix };
        let preconditioner = DiagonalPreconditioner::from_diagonal(&[
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ]);
        let rhs = [Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];
        let mut subspace = GcroDrSubspace::new(1);
        let result = gcro_dr_preconditioned(
            &operator,
            &preconditioner,
            &rhs,
            None,
            &mut subspace,
            &GcroDrConfig {
                max_iterations: 10,
                restart: 2,
                tolerance: 1e-12,
                print_interval: 0,
            },
        );
        assert!(result.converged);
        assert!(subspace.is_populated());

        assert!(reset_recycling_for_frequency_step(
            Some(100.0),
            200.0,
            0.25,
            &mut subspace
        ));
        assert!(!subspace.is_populated());
    }

    #[test]
    fn test_iterative_bem_solve_rejects_unconverged_result() {
        let matrix = DenseMatrix::from_row_major_vec(
            2,
            2,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        );
        let rhs = vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];
        let config = SolverConfig {
            direct_threshold: 0,
            max_iterations: 0,
            restart: 2,
            tolerance: 1e-12,
            ..SolverConfig::default()
        };

        let err = solve_bem_system(&matrix, &rhs, None, &config, None).unwrap_err();
        assert!(
            err.to_string().contains("failed to converge"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_direct_bem_solve_rejects_non_finite_solution() {
        let matrix = DenseMatrix::from_row_major_vec(1, 1, vec![Complex64::new(f64::NAN, 0.0)]);
        let rhs = vec![Complex64::new(1.0, 0.0)];
        let config = SolverConfig {
            direct_threshold: 1,
            ..SolverConfig::default()
        };

        let error = solve_bem_system(&matrix, &rhs, None, &config, None).unwrap_err();
        assert!(error.to_string().contains("non-finite"), "got: {error}");
    }

    fn adjoint_test_system() -> (DenseMatrix<Complex64>, Vec<Complex64>) {
        let matrix = DenseMatrix::from_row_major_vec(
            3,
            3,
            vec![
                Complex64::new(4.0, 1.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, -0.5),
                Complex64::new(1.0, -1.0),
                Complex64::new(3.0, 0.5),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(1.0, 1.0),
                Complex64::new(4.0, -1.0),
            ],
        );
        let rhs = vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(-2.0, 1.0),
            Complex64::new(0.0, -1.0),
        ];
        (matrix, rhs)
    }

    fn assert_adjoint_solution(
        matrix: &DenseMatrix<Complex64>,
        lambda: &[Complex64],
        rhs: &[Complex64],
    ) {
        let mut applied = vec![Complex64::new(0.0, 0.0); matrix.ncols()];
        matrix.matvec_hermitian_into(lambda, &mut applied);
        for (value, expected) in applied.iter().zip(rhs) {
            assert!(
                (*value - *expected).norm() < 1e-8,
                "Aᴴλ = {value:?} differs from rhs {expected:?}"
            );
        }
    }

    #[test]
    fn test_adjoint_solve_reuses_lu_factors() {
        let (matrix, rhs) = adjoint_test_system();
        let config = SolverConfig {
            direct_threshold: 10,
            ..SolverConfig::default()
        };
        let mut sweep = BemAdjointSweepState::default();
        let (forward, state) = sweep
            .solve_forward(&matrix, &rhs, None, 1000.0, &config)
            .unwrap();
        assert!(matches!(state, BemSolverState::Lu(_)));
        let forward_residual = relative_residual(&matrix, &forward, &rhs);
        assert!(
            forward_residual < 1e-10,
            "forward residual {forward_residual:.3e}"
        );

        let lambda = solve_bem_adjoint_system(&matrix, &rhs, &state, &config).unwrap();
        assert_adjoint_solution(&matrix, &lambda, &rhs);
    }

    #[test]
    fn test_adjoint_solve_reuses_ilu_preconditioner() {
        let (matrix, rhs) = adjoint_test_system();
        let config = SolverConfig {
            direct_threshold: 0,
            ..SolverConfig::default()
        };
        let mut sweep = BemAdjointSweepState::default();
        let (_, state_first) = sweep
            .solve_forward(&matrix, &rhs, None, 1000.0, &config)
            .unwrap();
        // A nearby frequency must recycle the same preconditioner (0.25-octave policy).
        let (_, state_second) = sweep
            .solve_forward(&matrix, &rhs, None, 1010.0, &config)
            .unwrap();
        match (&state_first, &state_second) {
            (BemSolverState::Ilu(first), BemSolverState::Ilu(second)) => {
                assert!(
                    Rc::ptr_eq(first, second),
                    "preconditioner should be recycled within 0.25 octaves"
                );
            }
            _ => panic!("expected the iterative path to retain ILU state"),
        }

        let lambda = solve_bem_adjoint_system(&matrix, &rhs, &state_second, &config).unwrap();
        assert_adjoint_solution(&matrix, &lambda, &rhs);
    }

    /// The ILUᴴ adjoint (operator Hermitian + Hermitian of the forward ILU,
    /// no dense Aᴴ materialization) must match the previous path, which
    /// explicitly formed Aᴴ and solved it with the forward ILU.
    #[test]
    fn test_adjoint_solve_ilu_hermitian_matches_materialized_path() {
        let (matrix, rhs) = adjoint_test_system();
        let config = SolverConfig {
            direct_threshold: 0,
            ..SolverConfig::default()
        };
        let mut sweep = BemAdjointSweepState::default();
        let (_, state) = sweep
            .solve_forward(&matrix, &rhs, None, 1000.0, &config)
            .unwrap();
        let BemSolverState::Ilu(preconditioner) = &state else {
            panic!("expected the iterative path to retain ILU state");
        };

        // New path: Aᴴ applied on the fly, preconditioned by ILUᴴ.
        let lambda_new = solve_bem_adjoint_system(&matrix, &rhs, &state, &config).unwrap();

        // Old path: explicitly materialized Aᴴ with the forward ILU.
        let hermitian = hermitian_transpose(&matrix);
        let lambda_old = solve_bem_system_with_preconditioner(
            &hermitian,
            &rhs,
            None,
            &config,
            None,
            Some(preconditioner.as_ref()),
        )
        .unwrap();

        let diff = lambda_new
            .iter()
            .zip(&lambda_old)
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let reference = lambda_old
            .iter()
            .map(|v| v.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(1e-30);
        assert!(
            diff / reference < 1e-4,
            "ILU^H adjoint differs from the materialized-A^H path by {}",
            diff / reference
        );
        assert_adjoint_solution(&matrix, &lambda_new, &rhs);
    }

    /// H-matrix adjoint smoke test: solve Aᴴ λ = rhs through the H-matrix
    /// Hermitian matvec + adjointed Jacobi preconditioner, and validate
    /// against the dense LU conj-transpose reference on the tiny model.
    #[test]
    fn test_adjoint_solve_hmatrix_smoke() {
        let model = tiny_model();
        let speaker_mesh = crate::mesh_gen::generate_loudspeaker_mesh(&model);
        let (nodes, mut elements) = speaker_mesh_to_bem(&speaker_mesh, &model).unwrap();
        update_driver_bcs(&mut elements, &speaker_mesh, &model, 1000.0, None).unwrap();
        let physics = PhysicsParams::new(
            1000.0,
            model.simulation.speed_of_sound,
            model.simulation.air_density,
            false,
        );
        let config = SolverConfig {
            hmatrix_aca_tolerance: 1e-10,
            ..SolverConfig::default()
        };
        let centers: Vec<[f64; 3]> = elements.iter().map(|element| element.center).collect();
        let mut octree = Octree::build(&centers, 32, 8);
        octree.compute_interaction_lists(2.0);
        let clusters = clusters_from_octree(&elements, &octree);
        let hmatrix = build_simulation_hmatrix(&elements, &nodes, &clusters, &physics, &config);
        let rhs = build_tbem_rhs_stabilized(
            &elements,
            &nodes,
            &physics,
            EXTERIOR_TBEM_SIGN,
            config.beta_scale,
        );

        let (lambda, _) = solve_bem_adjoint_system_hmatrix(&hmatrix, &rhs, &config).unwrap();

        // Dense reference: LU conj-transpose solve of the explicitly
        // assembled TBEM operator.
        let dense = sonium_bem::core::assembly::tbem::build_tbem_system_stabilized(
            &elements,
            &nodes,
            &physics,
            EXTERIOR_TBEM_SIGN,
            config.beta_scale,
        );
        let reference = lu_factorize(&dense.matrix)
            .expect("LU of dense TBEM")
            .solve_conj_transpose(&rhs)
            .expect("dense adjoint solve");

        let diff = lambda
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let norm = reference
            .iter()
            .map(|v| v.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(1e-30);
        assert!(
            diff / norm < 1e-4,
            "H-matrix adjoint differs from dense LU reference by {}",
            diff / norm
        );
    }

    /// Sentinel cadence: with `every_k = 3` and a residual far below the
    /// marginal threshold, the exact check fires exactly on solves 3, 6, 9.
    #[test]
    fn test_exact_residual_sentinel_cadence() {
        let config = SolverConfig {
            tolerance: 1e-6,
            exact_residual_sentinel: SentinelPolicy::Sentinel {
                every_k: 3,
                on_marginal_convergence_factor: 10.0,
            },
            ..SolverConfig::default()
        };
        let mut sentinel = ExactResidualSentinel::default();
        // Residual 1e-12 << tolerance / factor = 1e-7: marginal never fires.
        let mut fired_indices = Vec::new();
        for solve in 1..=9 {
            if sentinel.should_verify_exact(&config, 1e-12) {
                fired_indices.push(solve);
            }
        }
        assert_eq!(fired_indices, vec![3, 6, 9]);
    }

    /// Marginal-convergence trigger: a compressed residual within
    /// `factor`× of the tolerance forces the exact check even off cadence;
    /// a comfortably small residual does not.
    #[test]
    fn test_exact_residual_sentinel_marginal_trigger() {
        let config = SolverConfig {
            tolerance: 1e-6,
            exact_residual_sentinel: SentinelPolicy::Sentinel {
                every_k: 1000,
                on_marginal_convergence_factor: 10.0,
            },
            ..SolverConfig::default()
        };
        // tolerance / 5 = 2e-7 > tolerance / factor = 1e-7: marginal fires.
        let mut marginal = ExactResidualSentinel::default();
        assert!(marginal.should_verify_exact(&config, 2e-7));
        // tolerance / 50 = 2e-8 < 1e-7: no cadence hit, no trigger.
        let mut comfortable = ExactResidualSentinel::default();
        assert!(!comfortable.should_verify_exact(&config, 2e-8));
    }

    /// `EverySolve` always fires; `Off` never does — not even on a marginal
    /// residual. The legacy `verify_exact_hmatrix_residual` boolean forces
    /// the check regardless of the policy.
    #[test]
    fn test_exact_residual_sentinel_policy_variants() {
        let base = SolverConfig {
            tolerance: 1e-6,
            ..SolverConfig::default()
        };
        let every = SolverConfig {
            exact_residual_sentinel: SentinelPolicy::EverySolve,
            ..base.clone()
        };
        let mut sentinel = ExactResidualSentinel::default();
        for _ in 0..5 {
            assert!(sentinel.should_verify_exact(&every, 1e-15));
        }

        let off = SolverConfig {
            exact_residual_sentinel: SentinelPolicy::Off,
            ..base.clone()
        };
        let mut sentinel = ExactResidualSentinel::default();
        for _ in 0..5 {
            assert!(!sentinel.should_verify_exact(&off, 9e-7));
        }

        let legacy = SolverConfig {
            verify_exact_hmatrix_residual: true,
            exact_residual_sentinel: SentinelPolicy::Off,
            ..base
        };
        let mut sentinel = ExactResidualSentinel::default();
        assert!(sentinel.should_verify_exact(&legacy, 1e-15));
    }

    /// Hard gate: the exact O(N²) residual check must reject a corrupted
    /// H-matrix solution that still satisfies the compressed operator.
    #[test]
    fn test_exact_residual_sentinel_catches_corrupted_solution() {
        let model = tiny_model();
        let speaker_mesh = crate::mesh_gen::generate_loudspeaker_mesh(&model);
        let (nodes, mut elements) = speaker_mesh_to_bem(&speaker_mesh, &model).unwrap();
        update_driver_bcs(&mut elements, &speaker_mesh, &model, 1000.0, None).unwrap();
        let physics = PhysicsParams::new(
            1000.0,
            model.simulation.speed_of_sound,
            model.simulation.air_density,
            false,
        );
        let config = SolverConfig {
            hmatrix_aca_tolerance: 1e-10,
            ..SolverConfig::default()
        };
        let centers: Vec<[f64; 3]> = elements.iter().map(|element| element.center).collect();
        let mut octree = Octree::build(&centers, 32, 8);
        octree.compute_interaction_lists(2.0);
        let clusters = clusters_from_octree(&elements, &octree);
        let hmatrix = build_simulation_hmatrix(&elements, &nodes, &clusters, &physics, &config);
        let rhs = build_tbem_rhs_stabilized(
            &elements,
            &nodes,
            &physics,
            EXTERIOR_TBEM_SIGN,
            config.beta_scale,
        );

        let (solution, _) = solve_bem_system_hmatrix(&hmatrix, &rhs, None, &config, None).unwrap();
        // Sanity: the converged solution passes the exact check.
        verify_exact_hmatrix_solution(&elements, &nodes, &physics, &solution, &rhs, &config)
            .expect("converged solution must pass the exact residual gate");

        // Corrupt the solution: the exact operator must reject it.
        let corrupted = vec![Complex64::new(0.0, 0.0); solution.len()];
        assert!(
            verify_exact_hmatrix_solution(&elements, &nodes, &physics, &corrupted, &rhs, &config)
                .is_err(),
            "zeroed solution must fail the exact residual gate"
        );
    }

    #[test]
    #[ignore = "production H-matrix/dense comparison; run via `just qa-nightly-regressions`"]
    fn test_simulation_hmatrix_matches_dense_stabilized_operator() {
        let model = tiny_model();
        let speaker_mesh = crate::mesh_gen::generate_loudspeaker_mesh(&model);
        let (nodes, mut elements) = speaker_mesh_to_bem(&speaker_mesh, &model).unwrap();
        update_driver_bcs(&mut elements, &speaker_mesh, &model, 1000.0, None).unwrap();
        let physics = PhysicsParams::new(
            1000.0,
            model.simulation.speed_of_sound,
            model.simulation.air_density,
            false,
        );
        let config = SolverConfig {
            hmatrix_aca_tolerance: 1e-10,
            ..SolverConfig::default()
        };
        let centers: Vec<[f64; 3]> = elements.iter().map(|element| element.center).collect();
        let mut octree = Octree::build(&centers, 32, 8);
        octree.compute_interaction_lists(2.0);
        let clusters = clusters_from_octree(&elements, &octree);
        let hmatrix = build_simulation_hmatrix(&elements, &nodes, &clusters, &physics, &config);
        let dense = sonium_bem::core::assembly::tbem::build_tbem_system_stabilized(
            &elements,
            &nodes,
            &physics,
            1.0,
            config.beta_scale,
        );
        let x: Vec<Complex64> = (0..elements.len())
            .map(|index| Complex64::new(1.0 + index as f64 * 1e-3, -0.2))
            .collect();
        let got = hmatrix.apply(&x);
        let expected = dense.matrix.matvec(&x);
        let error = got
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let reference = expected
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(
            error / reference.max(1e-30) < 1e-7,
            "H-matrix/dense relative matvec error = {}",
            error / reference.max(1e-30)
        );
    }

    #[test]
    fn test_run_simulation_single_freq() {
        let model = tiny_model();
        let eval_points = vec![[0.0, 0.0, 1.14]]; // 1m in front of baffle

        let result = run_simulation(&model, &eval_points).expect("simulation should succeed");

        assert_eq!(result.frequencies.len(), 1);
        assert_eq!(result.spl.shape(), (1, 1));

        // SPL should be finite and reasonable (not NaN/Inf)
        let spl = result.spl[(0, 0)];
        assert!(spl.is_finite(), "SPL is not finite: {spl}");
        // A 1 mm/s piston at 1m should produce something in the 40-100 dB range
        assert!(spl > 20.0 && spl < 120.0, "SPL out of range: {spl} dB");
    }

    #[test]
    fn product_structural_feedback_is_exchanged_and_retained() {
        let mut model = tiny_model();
        model.waveguide = None;
        let mut cabinet = sonium_sim_speaker_modeling::CabinetDesign::default();
        cabinet.port = None;
        model.product_design = Some(sonium_sim_speaker_modeling::ProductDesign {
            cabinet,
            drivers: vec![
                sonium_sim_speaker_modeling::ProductDriver {
                    id: "woofer".into(),
                    radius_m: model.woofer.radius,
                    position_m: [model.woofer.x_offset, model.woofer.y_offset, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    force_factor_tm: 5.0,
                    coil_resistance_ohm: 5.0,
                    coil_reactance_ohm: 1.0,
                    mechanical_resistance_n_s_per_m: 0.5,
                    mechanical_reactance_n_s_per_m: 2.0,
                    crossover_branch: 0,
                },
                sonium_sim_speaker_modeling::ProductDriver {
                    id: "tweeter".into(),
                    radius_m: model.tweeter.radius,
                    position_m: [model.tweeter.x_offset, model.tweeter.y_offset, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    force_factor_tm: 2.0,
                    coil_resistance_ohm: 4.0,
                    coil_reactance_ohm: 0.5,
                    mechanical_resistance_n_s_per_m: 0.3,
                    mechanical_reactance_n_s_per_m: 1.0,
                    crossover_branch: 1,
                },
            ],
            crossover: sonium_sim_speaker_modeling::CrossoverNetwork {
                branches: vec![
                    sonium_sim_speaker_modeling::CrossoverBranch::default(),
                    sonium_sim_speaker_modeling::CrossoverBranch::default(),
                ],
            },
            waveguide: None,
            structural_acoustic: Some(sonium_sim_speaker_modeling::StructuralAcousticFeedback {
                area_m2: 0.10,
                modes: vec![sonium_sim_speaker_modeling::StructuralAcousticMode {
                    natural_frequency_hz: 10_000.0,
                    effective_mass_kg: 10.0,
                    damping_ratio: 0.08,
                    pressure_participation: 1.0,
                    stress_per_displacement_pa_per_m: 1.0e6,
                }],
                max_iterations: 6,
                velocity_tolerance: 1.0e-4,
                under_relaxation: 0.5,
            }),
        });
        let result = run_simulation_with_config(
            &model,
            &[[0.0, 0.0, 1.14]],
            &SolverConfig {
                parallel_frequency_sweep: false,
                ..SolverConfig::default()
            },
        )
        .expect("structural feedback solve should converge");
        let response = result.structural_responses[0]
            .expect("structural response must be retained in the result artifact");
        assert!(response.displacement_rms_m.norm().is_finite());
        assert!(response.velocity_rms_m_per_s.norm() > 0.0);
        assert!(response.absorbed_power_w >= 0.0);
    }

    // Lane-gate selection through the pure helper: hermetic by
    // construction (explicit configuration, no process env, no cache
    // init-order dependence), so it stays race-free however the ambient
    // environment or sibling tests set the gate keys.
    #[test]
    fn frequency_lane_gate() {
        for threads in [1usize, 2, 8] {
            // Tiny sweep stays serial; huge sweep fans out (multi-threaded).
            assert!(!use_frequency_lanes_with(2, 16, false, None, threads));
            assert_eq!(
                use_frequency_lanes_with(usize::MAX / 4096, 4096, false, None, threads),
                threads >= 2
            );
            // Scales with threads, not a fixed count: threads × 256²
            // entries hits exactly grain × threads.
            assert_eq!(
                use_frequency_lanes_with(threads, 256, false, None, threads),
                threads >= 2
            );
            assert!(!use_frequency_lanes_with(
                threads.saturating_sub(1),
                256,
                false,
                None,
                threads
            ));
            // Serial hatch forces serial for any size.
            assert!(!use_frequency_lanes_with(
                usize::MAX / 4096,
                4096,
                true,
                None,
                threads
            ));
            // Threshold override forces lanes for any size.
            assert!(use_frequency_lanes_with(1, 1, false, Some(1), threads));
            assert!(!use_frequency_lanes_with(1, 1, false, Some(2), threads));
        }
    }

    #[test]
    fn parallel_frequency_sweep_matches_serial_order_and_values() {
        let mut model = tiny_model();
        model.simulation.freq_end = 1_100.0;
        model.simulation.freq_points = 2;
        model.simulation.max_mesh_frequency = Some(1_100.0);
        let points = [[0.0, 0.0, 1.14]];
        let serial = run_simulation_with_config(
            &model,
            &points,
            &SolverConfig {
                parallel_frequency_sweep: false,
                ..SolverConfig::default()
            },
        )
        .unwrap();
        let mut warm_start = SimulationWarmStart::default();
        let parallel = run_simulation_with_warm_start(
            &model,
            &points,
            &SolverConfig {
                parallel_frequency_sweep: true,
                ..SolverConfig::default()
            },
            &mut warm_start,
        )
        .unwrap();
        assert_eq!(warm_start.surface_pressure_by_frequency.len(), 2);
        assert!(
            warm_start
                .surface_pressure_by_frequency
                .iter()
                .all(Option::is_some)
        );
        assert_eq!(parallel.frequencies, serial.frequencies);
        for frequency_index in 0..2 {
            assert!(
                (parallel.spl[(frequency_index, 0)] - serial.spl[(frequency_index, 0)]).abs()
                    < 1.0e-10
            );
            assert!(
                (parallel.phase[(frequency_index, 0)] - serial.phase[(frequency_index, 0)]).abs()
                    < 1.0e-10
            );
        }
    }

    #[test]
    fn parallel_frequency_sweep_with_lane_cap_matches_serial() {
        let mut model = tiny_model();
        model.simulation.freq_end = 1_100.0;
        model.simulation.freq_points = 4;
        model.simulation.max_mesh_frequency = Some(1_100.0);
        let points = [[0.0, 0.0, 1.14]];
        let serial = run_simulation_with_config(
            &model,
            &points,
            &SolverConfig {
                parallel_frequency_sweep: false,
                ..SolverConfig::default()
            },
        )
        .unwrap();
        let parallel = run_simulation_with_config(
            &model,
            &points,
            &SolverConfig {
                parallel_frequency_sweep: true,
                max_frequency_lanes: Some(2),
                ..SolverConfig::default()
            },
        )
        .unwrap();
        assert_eq!(parallel.frequencies, serial.frequencies);
        for frequency_index in 0..4 {
            assert!(
                (parallel.spl[(frequency_index, 0)] - serial.spl[(frequency_index, 0)]).abs()
                    < 1.0e-10,
                "lane-capped SPL mismatch at frequency index {frequency_index}"
            );
            assert!(
                (parallel.phase[(frequency_index, 0)] - serial.phase[(frequency_index, 0)]).abs()
                    < 1.0e-10,
                "lane-capped phase mismatch at frequency index {frequency_index}"
            );
        }
    }

    #[test]
    fn frequency_live_bytes_estimate_scales_with_strategy() {
        let config = SolverConfig {
            direct_threshold: 0,
            ..SolverConfig::default()
        };
        let dense = estimate_frequency_live_bytes(1_000, false, &config);
        // Dense operator + dense ILU each cost n² complexes; Krylov and
        // scratch add on top.
        assert!(dense > 2 * 1_000u64.pow(2) * 16);
        let direct = estimate_frequency_live_bytes(1_000, false, &SolverConfig::default());
        assert!(direct < dense, "direct path carries no dense ILU");
        let hmatrix = estimate_frequency_live_bytes(10_000, true, &config);
        assert!(
            hmatrix < estimate_frequency_live_bytes(10_000, false, &config),
            "H-matrix estimate must beat the dense O(n²) estimate"
        );
    }

    /// F2 regression: a design step that changes the mesh DOF count must not
    /// feed the previous design's surface pressures into the iterative solver
    /// (that panicked in the matvec length assert, killing the whole
    /// optimization run). Mismatched guesses are dropped in both the parallel
    /// and serial sweep paths, and the warm-started result must match a cold
    /// run within solver tolerance.
    #[test]
    fn warm_start_survives_design_point_with_different_dof_count() {
        let model_at = |max_mesh_frequency: f64| {
            let mut model = tiny_model();
            model.simulation.freq_start = 500.0;
            model.simulation.freq_end = 500.0;
            model.simulation.max_mesh_frequency = Some(max_mesh_frequency);
            model
        };
        let coarse = model_at(700.0);
        let fine = model_at(800.0);
        let dof_count = |model: &LoudspeakerModel| {
            crate::mesh_gen::generate_loudspeaker_mesh(model)
                .triangles
                .len()
        };
        assert_ne!(
            dof_count(&coarse),
            dof_count(&fine),
            "the two design points must exercise a DOF-count change"
        );
        let points = [[0.0, 0.0, 1.14]];
        // Force the iterative path: the panic happened when FGMRES/GCRO-DR
        // consumed the mismatched initial guess (LU ignores x0).
        for parallel_frequency_sweep in [true, false] {
            let config = SolverConfig {
                direct_threshold: 0,
                parallel_frequency_sweep,
                ..SolverConfig::default()
            };
            let mut warm_start = SimulationWarmStart::default();
            run_simulation_with_warm_start(&coarse, &points, &config, &mut warm_start)
                .expect("first design point should solve");
            let warmed = run_simulation_with_warm_start(&fine, &points, &config, &mut warm_start)
                .expect("warm start across a DOF-count change must not panic");
            let cold = run_simulation_with_config(&fine, &points, &config)
                .expect("cold reference run should solve");
            assert!(
                (warmed.spl[(0, 0)] - cold.spl[(0, 0)]).abs() < 1.0e-2,
                "warm-started SPL {:.6} dB vs cold {:.6} dB (parallel={parallel_frequency_sweep})",
                warmed.spl[(0, 0)],
                cold.spl[(0, 0)]
            );
            assert!(
                (warmed.phase[(0, 0)] - cold.phase[(0, 0)]).abs() < 1.0e-4,
                "warm-started phase vs cold phase (parallel={parallel_frequency_sweep})"
            );
        }
    }

    #[test]
    fn speaker_pressure_converges_under_mesh_refinement() {
        fn pressure_at_mesh_frequency(max_mesh_frequency: f64) -> (usize, Complex64) {
            let mut model = tiny_model();
            model.simulation.freq_start = 500.0;
            model.simulation.freq_end = 500.0;
            model.simulation.max_mesh_frequency = Some(max_mesh_frequency);
            model.tweeter.piston_velocity = 0.0;
            model.waveguide = None;
            let dofs = crate::mesh_gen::generate_loudspeaker_mesh(&model)
                .triangles
                .len();
            let result = run_simulation(&model, &[[0.0, 0.0, 1.14]])
                .expect("mesh-convergence solve should succeed");
            let magnitude = 20.0e-6 * 10.0_f64.powf(result.spl[(0, 0)] / 20.0);
            (dofs, Complex64::from_polar(magnitude, result.phase[(0, 0)]))
        }

        let (coarse_dofs, coarse) = pressure_at_mesh_frequency(750.0);
        let (medium_dofs, medium) = pressure_at_mesh_frequency(1_000.0);
        let (fine_dofs, fine) = pressure_at_mesh_frequency(1_500.0);
        assert!(
            coarse_dofs < medium_dofs && medium_dofs < fine_dofs,
            "refinement must increase BEM resolution: {coarse_dofs}, {medium_dofs}, {fine_dofs}"
        );
        let coarse_step = (medium - coarse).norm();
        let fine_step = (fine - medium).norm();
        assert!(
            fine_step < coarse_step,
            "complex pressure did not converge: coarse step={coarse_step:.3e}, fine step={fine_step:.3e}"
        );
        assert!(
            fine_step / fine.norm().max(f64::MIN_POSITIVE) < 0.20,
            "last mesh-refinement step changed pressure by more than 20%"
        );
    }

    // Fixture reads the deprecated per-driver vectors alongside the matrix.
    #[allow(deprecated)]
    #[test]
    #[ignore = "full typed-driver product solve; run via `just qa-nightly-regressions`"]
    fn typed_driver_is_back_loaded_and_reports_electrical_impedance() {
        let mut model = tiny_model();
        let study =
            sonium_sim_speaker_modeling::SpeakerDriverStudy::detailed_frequency_domain_driver();
        model.woofer.radius = study.geometry.diaphragm_radius_m;
        model.woofer.driver_model =
            Some(sonium_sim_speaker_modeling::DriverModel::from_study(&study).unwrap());
        model.tweeter.piston_velocity = 0.0;
        let result = run_simulation(&model, &[[0.0, 0.0, 1.14]]).unwrap();
        let impedance = result.woofer_electrical_impedance_ohm[0].unwrap();

        assert!(impedance.re > 0.0);
        assert!(impedance.im.is_finite());
        assert_eq!(result.tweeter_electrical_impedance_ohm[0], None);
    }

    fn typed_woofer_model() -> LoudspeakerModel {
        let mut model = tiny_model();
        let study =
            sonium_sim_speaker_modeling::SpeakerDriverStudy::detailed_frequency_domain_driver();
        model.woofer.radius = study.geometry.diaphragm_radius_m;
        model.woofer.driver_model =
            Some(sonium_sim_speaker_modeling::DriverModel::from_study(&study).unwrap());
        model
    }

    #[test]
    fn three_driver_model_solves_through_generic_coupling() {
        // The fixed two-driver helper cannot represent N > 2 (its arrays are
        // `[T; 2]` by construction); a 3-driver model must solve through the
        // generic N-port path and return one velocity per driver, not panic.
        let mut model = typed_woofer_model();
        model.additional_drivers.push(model.woofer.clone());
        assert_eq!(model.drivers().len(), 3);
        let radiation = vec![
            vec![
                Complex64::new(1.0, 0.3),
                Complex64::new(0.1, 0.05),
                Complex64::new(0.02, 0.01),
            ],
            vec![
                Complex64::new(0.1, 0.05),
                Complex64::new(1.2, 0.4),
                Complex64::new(0.02, 0.01),
            ],
            vec![
                Complex64::new(0.02, 0.01),
                Complex64::new(0.02, 0.01),
                Complex64::new(1.1, 0.35),
            ],
        ];
        let coupled = solve_coupled_driver_network_generic(&model, 1_000.0, &radiation)
            .expect("3-driver model must solve through the generic N-port path");
        assert_eq!(coupled.velocities_bem_all.len(), 3);
        assert!(
            coupled
                .velocities_bem_all
                .iter()
                .all(|velocity| velocity.norm().is_finite()),
            "velocities must be finite"
        );
    }

    #[test]
    fn typed_network_thermal_feedback_updates_loaded_state_and_closes_power() {
        let mut model = typed_woofer_model();
        let study =
            sonium_sim_speaker_modeling::SpeakerDriverStudy::detailed_frequency_domain_driver();
        let mut woofer = model.woofer.driver_model.take().unwrap();
        woofer.thermal_compression_spec =
            Some(sonium_sim_speaker_modeling::ThermalCompressionSpec::default());
        let mut tweeter = sonium_sim_speaker_modeling::DriverModel::from_study(&study).unwrap();
        tweeter.thermal_compression_spec =
            Some(sonium_sim_speaker_modeling::ThermalCompressionSpec::default());
        model.woofer.driver_model = Some(woofer.clone());
        model.tweeter.driver_model = Some(tweeter);

        let radiation = [
            [Complex64::new(1.0, 0.3), Complex64::new(0.1, 0.05)],
            [Complex64::new(0.1, 0.05), Complex64::new(1.2, 0.4)],
        ];
        let coupled = solve_coupled_driver_network(&model, 1_000.0, &radiation)
            .expect("thermal coupled network should converge");
        for state in &coupled.thermal_driver_states {
            let state = state
                .as_ref()
                .expect("configured driver must report thermal state");
            assert!(state.hot_resistance_ohm > 0.0);
            assert!(
                state.coil_temperature_k
                    > sonium_sim_speaker_modeling::ThermalCompressionSpec::default()
                        .ambient_temperature_k
            );
            assert!(state.thermal_iteration_count > 0);
            assert!(state.power_balance_residual_w.abs() < 1.0e-8);
        }
        assert!(
            coupled
                .thermal_driver_states
                .iter()
                .all(|state| state.is_some())
        );
        assert!(
            coupled
                .woofer_electrical_impedance_ohm
                .expect("woofer impedance")
                .re
                > woofer.thiele_small.voice_coil_resistance_ohm
        );
    }

    #[test]
    fn thermal_iteration_applies_nonlinear_motor_temperature_law() {
        let study =
            sonium_sim_speaker_modeling::SpeakerDriverStudy::detailed_frequency_domain_driver();
        let thermal = sonium_sim_speaker_modeling::ThermalCompressionSpec {
            thermal_resistance_k_per_w: 1_000.0,
            ..sonium_sim_speaker_modeling::ThermalCompressionSpec::default()
        };
        let mut baseline = typed_woofer_model();
        let mut baseline_woofer = baseline.woofer.driver_model.take().unwrap();
        baseline_woofer.thermal_compression_spec = Some(thermal);
        let mut baseline_tweeter =
            sonium_sim_speaker_modeling::DriverModel::from_study(&study).unwrap();
        baseline_tweeter.thermal_compression_spec = Some(thermal);
        baseline.woofer.driver_model = Some(baseline_woofer.clone());
        baseline.tweeter.driver_model = Some(baseline_tweeter.clone());

        let motor = sonium_sim_speaker_modeling::MotorConstitutiveModel {
            bl_curve: vec![
                sonium_sim_speaker_modeling::NonlinearBlPoint {
                    displacement_m: -0.001,
                    force_factor_tm: 6.0,
                },
                sonium_sim_speaker_modeling::NonlinearBlPoint {
                    displacement_m: 0.0,
                    force_factor_tm: 6.0,
                },
                sonium_sim_speaker_modeling::NonlinearBlPoint {
                    displacement_m: 0.001,
                    force_factor_tm: 6.0,
                },
            ],
            le_surface: vec![sonium_sim_speaker_modeling::NonlinearLePoint {
                displacement_m: 0.0,
                current_a: 0.0,
                frequency_hz: 1_000.0,
                inductance_h: 0.5e-3,
                eddy_resistance_ohm: 0.2,
            }],
            temperature: sonium_sim_speaker_modeling::MotorTemperatureLaw {
                copper_resistance_ohm: baseline_woofer.thiele_small.voice_coil_resistance_ohm,
                bl_per_k: -0.002,
                inductance_per_k: 0.001,
                ..sonium_sim_speaker_modeling::MotorTemperatureLaw::default()
            },
            hysteresis: sonium_sim_speaker_modeling::HysteresisLossModel::default(),
            magnetic: None,
            eddy_current: None,
        };
        let nonlinear_woofer = baseline_woofer.with_nonlinear_motor(motor).unwrap();
        let mut nonlinear = baseline.clone();
        nonlinear.woofer.driver_model = Some(nonlinear_woofer);
        let radiation = vec![
            vec![Complex64::new(1.0, 0.3), Complex64::new(0.1, 0.05)],
            vec![Complex64::new(0.1, 0.05), Complex64::new(1.2, 0.4)],
        ];

        let hot_law =
            solve_coupled_driver_network_generic(&nonlinear, 1_000.0, &radiation).unwrap();
        assert!(
            hot_law.thermal_driver_states[0]
                .as_ref()
                .is_some_and(|state| state.thermal_iteration_count >= 2)
        );
        assert!(
            hot_law.thermal_driver_states[0]
                .as_ref()
                .is_some_and(|state| state.nonlinear_temperature_feedback_applied)
        );
        assert!(
            hot_law.thermal_driver_states[0]
                .as_ref()
                .is_some_and(|state| state.coil_temperature_k > thermal.ambient_temperature_k)
        );
    }

    #[test]
    fn typed_driver_thermal_network_emits_state_and_product_power_ledger() {
        let mut model = typed_woofer_model();
        let study =
            sonium_sim_speaker_modeling::SpeakerDriverStudy::detailed_frequency_domain_driver();
        let mut woofer = model.woofer.driver_model.take().unwrap();
        let mut tweeter = sonium_sim_speaker_modeling::DriverModel::from_study(&study).unwrap();
        woofer.thermal_network = Some(Default::default());
        tweeter.thermal_network = Some(Default::default());
        model.woofer.driver_model = Some(woofer.clone());
        model.tweeter.driver_model = Some(tweeter);

        let radiation = [
            [Complex64::new(1.0, 0.3), Complex64::new(0.1, 0.05)],
            [Complex64::new(0.1, 0.05), Complex64::new(1.2, 0.4)],
        ];
        let coupled = solve_coupled_driver_network(&model, 1_000.0, &radiation)
            .expect("thermal-network coupled solve should converge");
        for state in &coupled.thermal_driver_states {
            let state = state.as_ref().expect("configured driver state");
            let temperatures = state
                .thermal_network_temperatures_k
                .expect("thermal network temperatures");
            assert!(
                temperatures
                    .iter()
                    .all(|temperature| temperature.is_finite())
            );
            assert!(temperatures[0] > 293.15);
            assert!(state.hot_resistance_ohm > woofer.thiele_small.voice_coil_resistance_ohm);
            assert!(state.thermal_network_power_w.is_some());
            assert!(state.thermal_network_residual_w.unwrap() < 1.0e-9);
        }
        let ledger = coupled
            .power_ledger
            .expect("fully typed solve must emit a product power ledger");
        assert!(ledger.supplied_w > 0.0);
        assert!(ledger.residual_w.abs() < 1.0e-9);
        ledger
            .close(PowerLedgerTolerance::default())
            .expect("product power ledger should close");
    }

    #[test]
    fn production_sweep_emits_thermal_network_and_power_ledger_artifacts() {
        let mut model = typed_woofer_model();
        let study =
            sonium_sim_speaker_modeling::SpeakerDriverStudy::detailed_frequency_domain_driver();
        let mut woofer = model.woofer.driver_model.take().unwrap();
        let mut tweeter = sonium_sim_speaker_modeling::DriverModel::from_study(&study).unwrap();
        woofer.thermal_network = Some(Default::default());
        tweeter.thermal_network = Some(Default::default());
        model.woofer.driver_model = Some(woofer);
        model.tweeter.driver_model = Some(tweeter);
        model.simulation.max_mesh_frequency = Some(1_000.0);

        let result = run_simulation(&model, &[[0.0, 0.0, 1.14]])
            .expect("production typed sweep should converge");
        assert_eq!(result.power_ledgers.len(), result.frequencies.len());
        assert_eq!(result.driver_velocities_bem.len(), result.frequencies.len());
        assert_eq!(result.driver_velocities_bem[0].len(), model.drivers().len());
        assert!(
            result.driver_velocities_bem[0]
                .iter()
                .all(|velocity| velocity.re.is_finite() && velocity.im.is_finite())
        );
        let ledger =
            result.power_ledgers[0].expect("production result must expose the typed power ledger");
        ledger
            .close(PowerLedgerTolerance::default())
            .expect("production power ledger should close");
        assert!(result.driver_thermal_states[0].iter().all(|state| {
            state
                .as_ref()
                .and_then(|state| state.thermal_network_temperatures_k)
                .is_some()
        }));
    }

    #[test]
    fn product_crossover_drive_reaches_typed_network_source() {
        let mut model = typed_woofer_model();
        let study =
            sonium_sim_speaker_modeling::SpeakerDriverStudy::detailed_frequency_domain_driver();
        model.tweeter.driver_model =
            Some(sonium_sim_speaker_modeling::DriverModel::from_study(&study).unwrap());

        let mut cabinet = sonium_sim_speaker_modeling::CabinetDesign::default();
        cabinet.port = None;
        let woofer_branch = sonium_sim_speaker_modeling::CrossoverBranch {
            kind: sonium_sim_speaker_modeling::CrossoverKind::LowPass,
            corner_frequency_hz: 1_000.0,
            gain: 0.5,
            delay_s: 50.0e-6,
        };
        let tweeter_branch = sonium_sim_speaker_modeling::CrossoverBranch {
            kind: sonium_sim_speaker_modeling::CrossoverKind::HighPass,
            corner_frequency_hz: 1_000.0,
            gain: 1.2,
            delay_s: 25.0e-6,
        };
        model.product_design = Some(sonium_sim_speaker_modeling::ProductDesign {
            cabinet,
            drivers: vec![
                sonium_sim_speaker_modeling::ProductDriver {
                    id: "woofer".into(),
                    radius_m: model.woofer.radius,
                    position_m: [model.woofer.x_offset, model.woofer.y_offset, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    force_factor_tm: 5.0,
                    coil_resistance_ohm: 5.0,
                    coil_reactance_ohm: 1.0,
                    mechanical_resistance_n_s_per_m: 0.5,
                    mechanical_reactance_n_s_per_m: 2.0,
                    crossover_branch: 0,
                },
                sonium_sim_speaker_modeling::ProductDriver {
                    id: "tweeter".into(),
                    radius_m: model.tweeter.radius,
                    position_m: [model.tweeter.x_offset, model.tweeter.y_offset, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    force_factor_tm: 2.0,
                    coil_resistance_ohm: 4.0,
                    coil_reactance_ohm: 0.5,
                    mechanical_resistance_n_s_per_m: 0.3,
                    mechanical_reactance_n_s_per_m: 1.0,
                    crossover_branch: 1,
                },
            ],
            crossover: sonium_sim_speaker_modeling::CrossoverNetwork {
                branches: vec![woofer_branch, tweeter_branch],
            },
            waveguide: None,
            structural_acoustic: None,
        });

        let frequency = 500.0;
        let drives = typed_driver_drive_voltages(&model, frequency).unwrap();
        let product = model.product_design.as_ref().unwrap();
        let expected_transfers = product.crossover.transfers_at(2, frequency).unwrap();
        for (index, driver) in model.drivers().into_iter().enumerate() {
            let reference = driver
                .driver_model
                .as_ref()
                .unwrap()
                .reference_voltage_rms_v;
            let expected = Complex64::new(reference, 0.0) * expected_transfers[index];
            assert!((drives[index] - expected).norm() < 1.0e-12);
        }

        let radiation = [
            [Complex64::new(1.0, 0.2), Complex64::new(0.05, 0.01)],
            [Complex64::new(0.05, 0.01), Complex64::new(1.2, 0.3)],
        ];
        let coupling = solve_coupled_driver_network(&model, frequency, &radiation).unwrap();
        let woofer = model.woofer.driver_model.as_ref().unwrap();
        let parameters = woofer.network_parameters_at(frequency).unwrap();
        let velocity_rms = coupling.velocities_bem[0].conj() / std::f64::consts::SQRT_2;
        let lhs = Complex64::new(parameters.force_factor_tm, 0.0) * drives[0]
            / parameters.coil_impedance_ohm;
        let rhs = (parameters.intrinsic_mechanical_impedance
            + Complex64::new(parameters.force_factor_tm.powi(2), 0.0)
                / parameters.coil_impedance_ohm)
            * velocity_rms
            + radiation[0][0] * velocity_rms
            + radiation[0][1] * (coupling.velocities_bem[1].conj() / std::f64::consts::SQRT_2);
        assert!((lhs - rhs).norm() < 1.0e-10 * lhs.norm());
    }

    /// B3: on the dense iterative path with a shared ILU preconditioner the
    /// two driver unit loads run as ONE 2-RHS block GMRES. The block path
    /// must reproduce the per-column (FGMRES+ILU) radiation matrix within a
    /// solver-tolerance-scaled bound, and both paths must pass the hard
    /// true-residual gate (each returns `Err` otherwise) and feed the
    /// coupled network solve.
    #[test]
    fn block_unit_load_solve_matches_sequential_z_rad() {
        let mut model = typed_woofer_model();
        // Small mesh: ILU(0) on the dense-pattern CSR is O(n³), which is
        // prohibitive in debug on the default coarse mesh.
        model.simulation.max_mesh_frequency = Some(700.0);
        let frequency = 500.0;
        let speaker_mesh = crate::mesh_gen::generate_loudspeaker_mesh(&model);
        let (nodes, elements) = speaker_mesh_to_bem(&speaker_mesh, &model).unwrap();
        let physics = PhysicsParams::new(
            frequency,
            model.simulation.speed_of_sound,
            model.simulation.air_density,
            false,
        );
        let config = SolverConfig {
            direct_threshold: 0,      // force the dense iterative path
            recycle_subspace_size: 0, // per-column path uses FGMRES (block GCRO-DR is out of scope)
            ..SolverConfig::default()
        };

        // Assemble the shared operator once (BC types matter, values do not).
        let mut assembly_elements = elements.clone();
        update_driver_bcs(
            &mut assembly_elements,
            &speaker_mesh,
            &model,
            frequency,
            None,
        )
        .unwrap();
        let matrix = build_tbem_system_stabilized(
            &assembly_elements,
            &nodes,
            &physics,
            EXTERIOR_TBEM_SIGN,
            config.beta_scale,
        )
        .matrix;
        let preconditioner = dense_ilu_preconditioner(&matrix, &config);

        // The decision predicate: block only on the dense iterative path
        // with a shared preconditioner.
        assert!(use_block_unit_load_solve(
            Some(&matrix),
            None,
            Some(&preconditioner),
            &config
        ));
        assert!(!use_block_unit_load_solve(
            Some(&matrix),
            None,
            None,
            &config
        ));
        let mut direct_only = config.clone();
        direct_only.direct_threshold = matrix.nrows();
        assert!(!use_block_unit_load_solve(
            Some(&matrix),
            None,
            Some(&preconditioner),
            &direct_only
        ));

        let mut block_elements = elements.clone();
        let z_block = solve_unit_load_radiation_matrix(
            &mut block_elements,
            &speaker_mesh,
            &model,
            frequency,
            &nodes,
            &physics,
            Some(&matrix),
            None,
            &config,
            &mut None,
            Some(&preconditioner),
            None,
            &mut ExactResidualSentinel::default(),
            None,
        )
        .expect("block unit-load solve must pass the true-residual gate");

        let mut sequential_elements = elements.clone();
        let z_sequential = solve_unit_load_radiation_matrix(
            &mut sequential_elements,
            &speaker_mesh,
            &model,
            frequency,
            &nodes,
            &physics,
            Some(&matrix),
            None,
            &config,
            &mut None,
            None,
            None,
            &mut ExactResidualSentinel::default(),
            None,
        )
        .expect("sequential unit-load solve must pass the true-residual gate");

        // Parity: relative to the diagonal scale, each entry agrees to
        // within 100× the solve tolerance (two independent iterative
        // solves, each gated at 1e-6).
        let diagonal_scale = z_sequential[0][0]
            .norm()
            .max(z_sequential[1][1].norm())
            .max(1.0e-12);
        for row in 0..2 {
            for column in 0..2 {
                let err = (z_block[row][column] - z_sequential[row][column]).norm();
                assert!(
                    err <= 100.0 * config.tolerance * diagonal_scale,
                    "z_rad[{row}][{column}]: block={:.6e} sequential={:.6e} err={err:.3e}",
                    z_block[row][column],
                    z_sequential[row][column]
                );
            }
        }

        // Both radiation matrices feed the coupled network solve (the hard
        // coupling gate: singular or non-physical networks return Err).
        let z_block_2 = [
            [z_block[0][0], z_block[0][1]],
            [z_block[1][0], z_block[1][1]],
        ];
        let z_sequential_2 = [
            [z_sequential[0][0], z_sequential[0][1]],
            [z_sequential[1][0], z_sequential[1][1]],
        ];
        let coupling_block = solve_coupled_driver_network(&model, frequency, &z_block_2)
            .expect("block z_rad must feed the coupled network");
        let coupling_sequential = solve_coupled_driver_network(&model, frequency, &z_sequential_2)
            .expect("sequential z_rad must feed the coupled network");
        for driver in 0..2 {
            let err = (coupling_block.velocities_bem[driver]
                - coupling_sequential.velocities_bem[driver])
                .norm();
            let scale = coupling_sequential.velocities_bem[driver]
                .norm()
                .max(1.0e-12);
            assert!(
                err <= 100.0 * config.tolerance * scale,
                "coupled velocity driver {driver}: block={:.6e} sequential={:.6e}",
                coupling_block.velocities_bem[driver],
                coupling_sequential.velocities_bem[driver]
            );
        }
        assert_eq!(
            coupling_block.woofer_electrical_impedance_ohm.is_some(),
            coupling_sequential
                .woofer_electrical_impedance_ohm
                .is_some()
        );
    }

    /// With zero mutual terms and a stationary tweeter, the coupled two-port
    /// solve must reduce exactly to the single-driver radiation-load response.
    #[test]
    fn coupled_network_reduces_to_single_driver_without_mutual_terms() {
        let mut model = typed_woofer_model();
        model.tweeter.piston_velocity = 0.0;
        let frequency = 500.0;
        let z_ww = sonium_sim_speaker_modeling::baffled_piston_mechanical_radiation_impedance(
            model.woofer.radius,
            frequency,
            sonium_sim_speaker_modeling::AcousticMedium::standard_air(),
        )
        .unwrap();
        let z_rad = [[z_ww, Complex64::default()], [Complex64::default(), z_ww]];

        let coupling = solve_coupled_driver_network(&model, frequency, &z_rad).unwrap();
        let (expected_velocity_rms, expected_electrical) = model
            .woofer
            .driver_model
            .as_ref()
            .unwrap()
            .response_with_radiation_impedance(frequency, z_ww)
            .unwrap();

        let expected_bem = std::f64::consts::SQRT_2 * expected_velocity_rms.conj();
        assert!(
            (coupling.velocities_bem[0] - expected_bem).norm() < 1.0e-12 * expected_bem.norm(),
            "coupled velocity {} vs single-driver {expected_bem}",
            coupling.velocities_bem[0]
        );
        let electrical = coupling.woofer_electrical_impedance_ohm.unwrap();
        assert!(
            (electrical - expected_electrical).norm() < 1.0e-9 * expected_electrical.norm(),
            "coupled Z_in {electrical} vs single-driver {expected_electrical}"
        );
        // Untyped tweeter keeps its (zero) prescribed velocity.
        assert_eq!(coupling.velocities_bem[1], Complex64::default());
        assert_eq!(coupling.tweeter_electrical_impedance_ohm, None);
    }

    /// The solved velocities must satisfy the two-port network equations
    /// BL·V/Z_e = (Z_m + BL²/Z_e) v + Σ_j Z_ij v_j, including the mutual
    /// radiation terms and a prescribed-velocity untyped tweeter.
    #[test]
    fn coupled_network_satisfies_two_port_equations() {
        let mut model = typed_woofer_model();
        model.tweeter.piston_velocity = 0.002;
        let frequency = 500.0;
        let z_rad = [
            [Complex64::new(0.5, 0.3), Complex64::new(0.02, 0.01)],
            [Complex64::new(0.02, 0.01), Complex64::new(0.05, 0.02)],
        ];

        let coupling = solve_coupled_driver_network(&model, frequency, &z_rad).unwrap();
        let driver_model = model.woofer.driver_model.as_ref().unwrap();
        let parameters = driver_model.network_parameters_at(frequency).unwrap();
        let velocity_rms = coupling.velocities_bem[0].conj() / std::f64::consts::SQRT_2;
        let tweeter_rms =
            Complex64::new(model.tweeter.piston_velocity, 0.0) / std::f64::consts::SQRT_2;

        let voltage = Complex64::new(driver_model.reference_voltage_rms_v, 0.0);
        let lhs = Complex64::new(parameters.force_factor_tm, 0.0) * voltage
            / parameters.coil_impedance_ohm;
        let rhs = (parameters.intrinsic_mechanical_impedance
            + Complex64::new(parameters.force_factor_tm.powi(2), 0.0)
                / parameters.coil_impedance_ohm)
            * velocity_rms
            + z_rad[0][0] * velocity_rms
            + z_rad[0][1] * tweeter_rms;
        assert!(
            (lhs - rhs).norm() < 1.0e-10 * lhs.norm(),
            "two-port equation residual: lhs {lhs}, rhs {rhs}"
        );
        // The mutual term must actually change the woofer velocity relative
        // to the diagonal-only case.
        let z_diagonal = [
            [z_rad[0][0], Complex64::default()],
            [Complex64::default(), z_rad[1][1]],
        ];
        let uncoupled = solve_coupled_driver_network(&model, frequency, &z_diagonal).unwrap();
        assert!(
            (coupling.velocities_bem[0] - uncoupled.velocities_bem[0]).norm() > 0.0,
            "mutual radiation term had no effect on the woofer velocity"
        );
        // Untyped tweeter round-trips its prescribed velocity exactly.
        assert_eq!(
            coupling.velocities_bem[1],
            Complex64::new(model.tweeter.piston_velocity, 0.0)
        );
    }

    #[test]
    fn generic_network_preserves_three_driver_artifacts() {
        let mut model = typed_woofer_model();
        model.tweeter.driver_model = model.woofer.driver_model.clone();
        model.additional_drivers.push(DriverConfig {
            radius: 0.013,
            x_offset: 0.08,
            y_offset: 0.10,
            piston_velocity: 0.001,
            transfer_function: None,
            driver_model: model.woofer.driver_model.clone(),
            cone: None,
        });
        let radiation = vec![
            vec![
                Complex64::new(1.0, 0.2),
                Complex64::new(0.03, 0.01),
                Complex64::new(0.02, 0.01),
            ],
            vec![
                Complex64::new(0.03, 0.01),
                Complex64::new(1.2, 0.2),
                Complex64::new(0.02, 0.01),
            ],
            vec![
                Complex64::new(0.02, 0.01),
                Complex64::new(0.02, 0.01),
                Complex64::new(1.4, 0.2),
            ],
        ];
        let coupling = solve_coupled_driver_network_generic(&model, 500.0, &radiation)
            .expect("generic three-driver network should converge");
        assert_eq!(coupling.velocities_bem_all.len(), 3);
        assert_eq!(coupling.driver_electrical_impedances_ohm.len(), 3);
        assert_eq!(coupling.radiation_matrix_all, radiation);
        assert!(coupling.driver_electrical_impedances_ohm[0].is_some());
        assert!(coupling.driver_electrical_impedances_ohm[1].is_some());
        assert!(coupling.driver_electrical_impedances_ohm[2].is_some());
        assert!(coupling.power_ledger.is_some());
        coupling
            .power_ledger
            .clone()
            .unwrap()
            .close(PowerLedgerTolerance::default())
            .expect("fully typed three-driver solve must close the product power ledger");
    }

    /// The BEM-assembled two-driver radiation matrix must be approximately
    #[test]
    fn generic_three_driver_radiation_rejects_nonreciprocal_and_active_matrices() {
        let reciprocal_but_active = vec![
            vec![
                Complex64::new(-1.0, 0.2),
                Complex64::new(0.03, 0.01),
                Complex64::new(0.02, 0.01),
            ],
            vec![
                Complex64::new(0.03, 0.01),
                Complex64::new(1.2, 0.2),
                Complex64::new(0.02, 0.01),
            ],
            vec![
                Complex64::new(0.02, 0.01),
                Complex64::new(0.02, 0.01),
                Complex64::new(1.4, 0.2),
            ],
        ];
        assert!(diagnose_radiation_matrix_generic(500.0, &reciprocal_but_active).is_err());

        let nonreciprocal = vec![
            vec![
                Complex64::new(1.0, 0.2),
                Complex64::new(0.5, 0.01),
                Complex64::new(0.02, 0.01),
            ],
            vec![
                Complex64::new(0.03, 0.01),
                Complex64::new(1.2, 0.2),
                Complex64::new(0.02, 0.01),
            ],
            vec![
                Complex64::new(0.02, 0.01),
                Complex64::new(0.02, 0.01),
                Complex64::new(1.4, 0.2),
            ],
        ];
        assert!(diagnose_radiation_matrix_generic(500.0, &nonreciprocal).is_err());
    }

    /// The BEM-assembled two-driver radiation matrix must be approximately
    /// reciprocal and passive (Hermitian part positive semidefinite) with
    /// mass-like diagonal reactance in the modeling e^{+iωt} convention.
    ///
    /// Both apertures use the woofer radius and the waveguide is removed:
    /// discrete reciprocity requires both radiators to be comparably
    /// resolved, and the stock waveguide throat (0.013 m) is smaller than
    /// one coarse element.
    #[test]
    fn radiation_matrix_is_reciprocal_and_passive() {
        let mut model = tiny_model();
        model.tweeter.radius = model.woofer.radius;
        model.waveguide = None;
        // Coarse mesh keeps the two debug LU solves at a few seconds.
        model.simulation.max_mesh_frequency = Some(800.0);
        let frequency = 500.0;
        let speaker_mesh = crate::mesh_gen::generate_loudspeaker_mesh(&model);
        let (nodes, mut elements) = speaker_mesh_to_bem(&speaker_mesh, &model).unwrap();
        update_driver_bcs(&mut elements, &speaker_mesh, &model, frequency, None).unwrap();
        let physics = PhysicsParams::new(
            frequency,
            model.simulation.speed_of_sound,
            model.simulation.air_density,
            false,
        );
        let config = SolverConfig::default();
        let system = build_tbem_system_stabilized(
            &elements,
            &nodes,
            &physics,
            EXTERIOR_TBEM_SIGN,
            config.beta_scale,
        );

        let unit_states = [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ];
        let mut gcro_dr_subspace = None;
        let mut sentinel = ExactResidualSentinel::default();
        let mut z_rad = [[Complex64::default(); 2]; 2];
        for (column, velocities) in unit_states.iter().enumerate() {
            let pressure = solve_bem_with_driver_velocities(
                &mut elements,
                &speaker_mesh,
                &model,
                frequency,
                velocities,
                &nodes,
                &physics,
                Some(&system.matrix),
                None,
                &config,
                &mut gcro_dr_subspace,
                None,
                None,
                &mut sentinel,
            )
            .unwrap();
            let forces = integrate_driver_axial_forces(&elements, &speaker_mesh, &pressure);
            z_rad[0][column] = forces[0].conj();
            z_rad[1][column] = forces[1].conj();
        }

        // Diagonals: passive resistance and mass-like reactance (e^{+iωt}).
        for (driver, name) in [(0, "woofer"), (1, "tweeter")] {
            assert!(
                z_rad[driver][driver].re > 0.0,
                "{name} radiation resistance must be positive: {}",
                z_rad[driver][driver]
            );
            assert!(
                z_rad[driver][driver].im > 0.0,
                "{name} radiation reactance must add mass: {}",
                z_rad[driver][driver]
            );
        }
        // Reciprocity: Z_wt ≈ Z_tw (constant-element collocation is not
        // exactly reciprocal; the coarse-mesh error must stay small).
        let mutual_scale = z_rad[0][1].norm().max(z_rad[1][0].norm());
        assert!(mutual_scale > 0.0, "mutual radiation vanished");
        let reciprocity_error = (z_rad[0][1] - z_rad[1][0]).norm() / mutual_scale;
        assert!(
            reciprocity_error < 0.10,
            "reciprocity error {reciprocity_error} (Z_wt={}, Z_tw={})",
            z_rad[0][1],
            z_rad[1][0]
        );
        // Passivity: smallest eigenvalue of the Hermitian part (Z + Zᴴ)/2.
        let h11 = z_rad[0][0].re;
        let h22 = z_rad[1][1].re;
        let h12 = (z_rad[0][1] + z_rad[1][0].conj()) * 0.5;
        let mean = 0.5 * (h11 + h22);
        let spread = (0.25 * (h11 - h22).powi(2) + h12.norm_sqr()).sqrt();
        let smallest_eigenvalue = mean - spread;
        let diagonal_scale = h11.abs().max(h22.abs());
        assert!(
            smallest_eigenvalue > -0.05 * diagonal_scale,
            "Hermitian part not passive: smallest eigenvalue {smallest_eigenvalue}, \
             diagonal scale {diagonal_scale}"
        );
    }

    /// Regression: at mesh frequencies where a driver aperture radius exceeds
    /// 3 element sizes, `add_disk` switches from a center fan to stitched
    /// rings. The stitch call previously inverted the winding, so the annulus
    /// normals pointed −z (into the cabinet) while the center fan kept +z —
    /// corrupting both the velocity BC and the radiation force F = ∫p·n dA.
    #[test]
    fn driver_aperture_normals_point_outward_at_fine_mesh() {
        let model = tiny_model(); // 2000 Hz mesh: radius/h_max > 3 for the woofer
        let speaker_mesh = crate::mesh_gen::generate_loudspeaker_mesh(&model);
        let (_nodes, elements) = speaker_mesh_to_bem(&speaker_mesh, &model).unwrap();
        for (element, region) in elements.iter().zip(&speaker_mesh.region) {
            if matches!(
                region,
                MeshRegion::WooferAperture | MeshRegion::TweeterAperture
            ) {
                assert!(
                    element.normal[2] > 0.99,
                    "aperture element at {:?} has inward/mixed normal {:?}",
                    element.center,
                    element.normal
                );
            }
        }
    }

    /// F1 regression: the BEM radiation load handed to the modeling layer must
    /// be in the modeling crate's e^{+iωt} convention, where a mass-like
    /// reactance is Im(Z) > 0 (radiation mass is ADDED to M_ms). sonium-bem
    /// solves with e^{−iωt} phasors, so the handoff conjugates F/v; before
    /// that fix the reactive sign was flipped.
    ///
    /// Coarse single-frequency check: the tiny model's cabinet baffle is not
    /// the analytical infinite baffle, so only the sign and a generous
    /// magnitude band are asserted.
    #[test]
    fn radiation_load_handoff_matches_baffled_piston_sign() {
        let mut model = tiny_model();
        model.tweeter.piston_velocity = 0.0;
        // Coarse mesh keeps the single debug LU solve at a few seconds; the
        // fine-mesh aperture orientation is pinned separately by
        // `driver_aperture_normals_point_outward_at_fine_mesh`.
        model.simulation.max_mesh_frequency = Some(800.0);
        let frequency = 500.0;
        let velocities = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let speaker_mesh = crate::mesh_gen::generate_loudspeaker_mesh(&model);
        let (nodes, mut elements) = speaker_mesh_to_bem(&speaker_mesh, &model).unwrap();
        update_driver_bcs(
            &mut elements,
            &speaker_mesh,
            &model,
            frequency,
            Some(&velocities),
        )
        .unwrap();
        let physics = PhysicsParams::new(
            frequency,
            model.simulation.speed_of_sound,
            model.simulation.air_density,
            false,
        );
        let config = SolverConfig::default();
        let system = build_tbem_system_stabilized(
            &elements,
            &nodes,
            &physics,
            EXTERIOR_TBEM_SIGN,
            config.beta_scale,
        );
        let pressure = solve_bem_system(&system.matrix, &system.rhs, None, &config, None).unwrap();
        let load = driver_radiation_impedances(&elements, &speaker_mesh, &pressure, velocities)[0]
            .expect("woofer has non-zero velocity");

        let analytical =
            sonium_sim_speaker_modeling::baffled_piston_mechanical_radiation_impedance(
                model.woofer.radius,
                frequency,
                sonium_sim_speaker_modeling::AcousticMedium::standard_air(),
            )
            .unwrap();
        assert!(analytical.re > 0.0 && analytical.im > 0.0);
        assert!(
            load.re > 0.0,
            "radiation resistance must stay positive: {load}"
        );
        assert!(
            load.im > 0.0,
            "radiation reactance must add mass (e^{{+iωt}}: Im > 0): {load}"
        );
        let magnitude_ratio = load.norm() / analytical.norm();
        assert!(
            (1.0 / 3.0..=3.0).contains(&magnitude_ratio),
            "BEM load {load} vs analytical baffled piston {analytical} (ratio {magnitude_ratio})"
        );
    }

    /// F1 regression at the lumped level: adding a mass-like radiation load
    /// must shift the electrical-impedance resonance DOWN (added mass). With
    /// the conjugated (wrong-sign) load the resonance shifts up instead.
    #[test]
    fn radiation_mass_lowers_back_loaded_resonance() {
        let study =
            sonium_sim_speaker_modeling::SpeakerDriverStudy::detailed_frequency_domain_driver();
        let model = sonium_sim_speaker_modeling::DriverModel::from_study(&study).unwrap();
        let medium = sonium_sim_speaker_modeling::AcousticMedium::standard_air();
        let radius = study.geometry.diaphragm_radius_m;

        // Dense sweep around the driver's free-air resonance (~41 Hz).
        let frequencies: Vec<f64> = (0..=200).map(|i| 30.0 + i as f64 * 0.15).collect();
        let resonance_with = |load: &dyn Fn(f64) -> Complex64| -> f64 {
            frequencies
                .iter()
                .map(|&f| {
                    let (_, electrical) =
                        model.response_with_radiation_impedance(f, load(f)).unwrap();
                    (f, electrical.norm())
                })
                .reduce(|best, candidate| {
                    if candidate.1 > best.1 {
                        candidate
                    } else {
                        best
                    }
                })
                .unwrap()
                .0
        };
        let analytical = |f: f64| {
            sonium_sim_speaker_modeling::baffled_piston_mechanical_radiation_impedance(
                radius, f, medium,
            )
            .unwrap()
        };

        let unloaded = resonance_with(&|_| Complex64::new(0.0, 0.0));
        let loaded = resonance_with(&analytical);
        let wrong_sign = resonance_with(&|f| analytical(f).conj());

        assert!(
            loaded < unloaded,
            "added radiation mass must lower the resonance: loaded={loaded} Hz, unloaded={unloaded} Hz"
        );
        assert!(
            wrong_sign > unloaded,
            "conjugated (wrong-sign) load raises the resonance: wrong={wrong_sign} Hz, unloaded={unloaded} Hz"
        );
    }

    #[test]
    fn provenance_marks_synthetic_results_unvalidated_and_serializes_stable_shape() {
        let provenance = SimulationProvenance::synthetic(3, 7);
        assert_eq!(provenance.backend, SimulationBackend::Cpu);
        assert_eq!(provenance.execution_profile, ExecutionProfile::Preview);
        assert_eq!(provenance.precision, PrecisionPolicy::F64);
        assert_eq!(provenance.residual_gate.status, ResidualGateStatus::NotRun);
        assert_eq!(provenance.artifact.schema_version, 1);
        assert_eq!(provenance.artifact.frequency_count, 3);
        assert_eq!(provenance.artifact.evaluation_point_count, 7);
        assert_eq!(
            provenance.artifact.ordering,
            ArtifactOrdering::FrequencyMajor
        );
        let json = serde_json::to_string(&provenance).unwrap();
        assert!(json.contains("frequency_major"));
        assert!(json.contains("not_run"));
    }

    #[test]
    fn wasm_preview_provenance_cannot_be_confused_with_a_golden_solve() {
        let provenance = SimulationProvenance::wasm_preview(4, 12);
        assert_eq!(provenance.backend, SimulationBackend::WasmPreview);
        assert_eq!(provenance.execution_profile, ExecutionProfile::Preview);
        assert_eq!(provenance.precision, PrecisionPolicy::F64);
        assert_eq!(provenance.residual_gate.status, ResidualGateStatus::NotRun);
        let json = serde_json::to_string(&provenance).unwrap();
        assert!(json.contains("wasm_preview"));
        assert!(json.contains("\"preview\""));
    }

    // Fixture fills the deprecated per-driver vectors alongside the matrix.
    #[allow(deprecated)]
    #[test]
    fn wasm_preview_is_bounded_and_reduces_frequency_major_fields() {
        let result = SimulationResult {
            frequencies: vec![10.0, 20.0, 30.0, 40.0],
            spl: HostMatrix::from_row_major_vec(4, 3, (0..12).map(|value| value as f64).collect()),
            phase: HostMatrix::from_row_major_vec(
                4,
                3,
                (0..12).map(|value| -(value as f64)).collect(),
            ),
            complex_pressure: HostMatrix::zeros(4, 3),
            woofer_electrical_impedance_ohm: vec![None; 4],
            tweeter_electrical_impedance_ohm: vec![None; 4],
            driver_electrical_impedances_ohm: vec![vec![None; 2]; 4],
            radiation_matrices: vec![None; 4],
            radiation_matrix_diagnostics: vec![None; 4],
            thermal_driver_states: vec![[None, None]; 4],
            driver_thermal_states: vec![vec![None; 2]; 4],
            power_ledgers: vec![None; 4],
            product_reference_sweep: None,
            structural_responses: Vec::new(),
            driver_velocities_bem: vec![vec![]; 4],
            exact_residual_evaluations: 0,
            provenance: SimulationProvenance::synthetic(4, 3),
        };
        let preview = result.to_wasm_preview(2, 2).unwrap();
        assert_eq!(preview.frequencies, [10.0, 40.0]);
        assert_eq!(preview.spl, vec![vec![0.0, 2.0], vec![9.0, 11.0]]);
        assert_eq!(preview.phase, vec![vec![0.0, -2.0], vec![-9.0, -11.0]]);
        assert_eq!(preview.provenance, SimulationProvenance::wasm_preview(2, 2));
        assert!(
            result
                .to_wasm_preview(WASM_PREVIEW_MAX_FREQUENCIES + 1, 2)
                .is_err()
        );
        let round_trip: SpeakerPreviewArtifact =
            serde_json::from_str(&serde_json::to_string(&preview).unwrap()).unwrap();
        assert_eq!(round_trip, preview);
    }
}
