//! Speaker-driver modeling studies inspired by detailed multiphysics FEM workflows.
//!
//! The crate keeps example-level modeling decisions outside `sonium-fem`: geometry
//! intent, physics activation, coupling choices, and study sequencing live here.
//! Where Sonium already exposes public FEM pieces, the examples adapt into those
//! types instead of duplicating solver logic.
//!
//! Lumped response tables use an electromechanical model with an analytical
//! infinite-baffle rigid-piston radiation load; those tables are not FEM fields.
//! Separate axisymmetric, resolved-acoustic, port-flow and FEM–BEM APIs expose
//! field-based studies within their documented capability and convergence limits.
//! Complex drive and response phasors use RMS values.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use sonium_fem::physics::acoustics::electroacoustics::speaker::{
    AcousticStructureBoundaryCoupling, CoupledSweepPoint, DistributedLorentzCoupling,
    LorentzIntegrationPoint, MagneticBiasState, MagneticPerturbationWorkflow, SpeakerGapModel,
    SpeakerResponse, recommend_speaker_gap_model,
};
use sonium_fem::physics::acoustics::electroacoustics::transducer::{
    ElectrodynamicCoupling, TransducerResponse,
};
use sonium_fem::physics::coupled::lumped_circuit::VoiceCoilParams;
use sonium_fem::physics::electromagnetics::magnetostatics::MagneticMaterial;
use spec_math::Bessel;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

pub mod acoustic_rom;
pub mod axisymmetric_linear;
pub mod backend_evidence;
pub mod cabinet_design;
pub mod calibration;
pub mod conventions;
pub mod coupled_objectives;
pub mod distributed_radiation;
pub mod driver_model;
pub mod eddy_current_bias;
pub mod fem_bem_coupling;
pub mod job_service;
pub mod field_export;
pub mod motor_extraction;
pub mod motor_topology;
pub mod nonlinear_bias;
pub mod nonlinear_motor;
mod parallelism;
pub mod port_flow;
pub mod power_ledger;
pub mod product_acoustic_mesh;
pub mod product_design;
pub mod projected_reanalysis;
pub mod resolved_acoustics;
pub mod resolved_coupling;
pub mod sector_3d;
pub mod structural_acoustic;
pub mod structural_constraints;
pub mod uncertainty;
pub mod waveguide_design;
pub mod workbench;

pub use acoustic_rom::{
    AcousticRom, AcousticRomConfig, RomError, RomFrequencySolution, StaticCorrection,
};
pub use axisymmetric_linear::{
    AxisymmetricLinearConfig, AxisymmetricLinearError, AxisymmetricLinearModel,
    AxisymmetricLinearSolution, MagneticBiasSolution, PowerBalance, RegionModulus,
    RegionModulusSpec, SolidRegionStiffness, ThieleSmallReference,
};
pub use distributed_radiation::{
    BaffledApertureConfig, BreakupPeak, DistributedRadiationError, DistributedRadiationResult,
    InterfaceVelocityTrace, RadiationRegime, axisymmetric_interface_trace,
    build_baffled_aperture_mesh, evaluate_distributed_radiation, radiation_regime,
    separate_breakup_peaks, solve_axisymmetric_radiation,
};
pub use backend_evidence::{
    CoupledBackendEvidence, SpeakerBackendError, SpeakerCapabilityMatrix, SpeakerCapabilityState,
    SpeakerSubsystem, SpeakerSubsystemCapability, run_coupled_reference,
};
#[cfg(feature = "cuda")]
pub use backend_evidence::{
    SpeakerBackendFailureStage, SpeakerCudaFailureEvidence, run_coupled_failure_contract,
    run_production_frequency_step_cuda_contract,
};
pub use cabinet_design::{
    BraceCrossSection, BraceDesign, CabinetDesign, CabinetDesignError, CabinetMaterial,
    DesignVariable, PORT_END_CORRECTION_RADIUS_FACTOR, PanelAxis, PanelId, PortDesign,
    PortPlacement,
};
pub use calibration::{
    CalibrationCoveragePolicy, CalibrationDataOrigin, CalibrationError, CalibrationMetricReport,
    CalibrationObservable, CalibrationReport, CalibrationTolerances, DriverCalibrationData,
    IndependentMeasurementMetadata, MagneticScanPoint, MeasuredBlPoint, MeasuredDisplacementPoint,
    MeasuredEddyCurrentPoint, MeasuredImpedancePoint, MeasuredInductancePoint,
    MeasuredMagneticBhPoint, MeasuredRadiationResponsePoint, MeasuredThermalPoint,
    MeasuredVelocityPoint, correlate_driver_model,
};
pub use conventions::{
    ArtifactBackend, ArtifactExecutionProfile, ArtifactPrecision, ArtifactProvenance,
    ExchangePhasorBridge, ReferenceModel, baseline_reference_models,
};
pub use coupled_objectives::{
    CoupledProductConstraintReport, CoupledProductConstraintSet, CoupledProductLimits,
    CoupledProductMetrics, coupled_product_constraint_labels, coupled_product_constraint_margins,
    to_coupled_product_constraints,
};
pub use driver_model::{
    BlPoint, DRIVER_MODEL_REFERENCE_ARTIFACT_SCHEMA_VERSION, DriverModel,
    DriverModelConvergenceEvidence, DriverModelError, DriverModelExtractionEvidence,
    DriverModelPromotionEvidence, DriverModelReferenceArtifact, DriverModelReferenceExecution,
    DriverModelReferencePoint, DriverModelReferenceTolerance, DriverModelTransferEvidence,
    DriverNetworkParameters, ExtractionResidualEvidence, ImpedancePoint, InductancePoint,
    ThermalDriverState, ThieleSmallParameters, VelocityTransferPoint,
};
pub use eddy_current_bias::{
    SpeakerEddyCurrentConfig, SpeakerEddyCurrentError, SpeakerEddyCurrentPoint,
    SpeakerEddyCurrentSweep,
};
pub use fem_bem_coupling::{
    PRODUCT_COUPLING_MESH_EVIDENCE_SCHEMA_VERSION, ProductCouplingMesh, ProductCouplingMeshError,
    ProductCouplingMeshEvidence, ProductCouplingMeshEvidenceError, ProductInterfaceTrace,
    build_product_coupling_mesh,
};
pub use motor_extraction::MotorParameters;
pub use motor_topology::{
    MotorTopologyBridgeEvidence, MotorTopologyConfig, MotorTopologyDensityEvidence,
    MotorTopologyError, MotorTopologyManufacturabilityReport, MotorTopologyPermeabilityEvidence,
    MotorTopologyProjectionEvidence, MotorTopologyResult, MotorTopologyRobustEvidence,
    MotorTopologyScalarEvidence, SteelPermeability, optimize_motor_topology,
};
pub use nonlinear_bias::{
    NonlinearSpeakerBiasConfig, NonlinearSpeakerBiasCurrentSample,
    NonlinearSpeakerBiasCurrentSweep, NonlinearSpeakerBiasError, NonlinearSpeakerBiasResult,
    SteelPermeabilityMap, default_low_carbon_steel_bh_curve, steel_permeability_map_from_bias,
};
pub use nonlinear_motor::{
    EddyCurrentLossModel, EddyCurrentLossSample, EddyCurrentOperatingValues,
    HysteresisLossFitReport, HysteresisLossModel, HysteresisLossSample,
    MOTOR_OPERATING_DOMAIN_EVIDENCE_SCHEMA_VERSION, MotorConstitutiveModel, MotorMagneticBhPoint,
    MotorMagneticFluxModulationPoint, MotorMagneticLaw, MotorMagneticOperatingValues,
    MotorOperatingDomain, MotorOperatingDomainEvidence, MotorOperatingPoint, MotorOperatingValues,
    MotorReluctanceGeometry, MotorTemperatureLaw, NonlinearBlPoint, NonlinearLePoint,
};
pub use power_ledger::{PowerClosure, PowerLedger, PowerLedgerError, PowerLedgerTolerance};
pub use product_acoustic_mesh::{
    InteriorWall, ProductInteriorMesh, ProductInteriorMeshConfig, ProductInteriorMeshDiagnostics,
    ProductInteriorMeshError, ProductInteriorPatch, ProductInteriorPatchKind, ResolvedPortGeometry,
    build_body_fitted_sealed_product_interior_mesh, build_ported_product_interior_mesh,
    build_sealed_product_interior_mesh,
};
pub use product_design::{
    CrossoverBranch, CrossoverKind, CrossoverNetwork, InterfaceReferenceEvidence,
    MonolithicEvidenceGate, MonolithicProductReferenceEvidence,
    PRODUCT_DESIGN_ARTIFACT_SCHEMA_VERSION, PRODUCT_INDEPENDENT_VALIDATION_SCHEMA_VERSION,
    PRODUCT_MESH_CONVERGENCE_SCHEMA_VERSION, PRODUCT_REFERENCE_ARTIFACT_SCHEMA_VERSION,
    PRODUCT_STRUCTURAL_EVIDENCE_SCHEMA_VERSION, PRODUCT_THERMAL_EVIDENCE_SCHEMA_VERSION,
    ProductBodyFittedMeshLevel, ProductDesign, ProductDesignArtifact, ProductDesignError,
    ProductDriver, ProductIndependentValidationEvidence, ProductMeshConvergenceEvidence,
    ProductMeshConvergenceLevel, ProductReferenceArtifact, ProductReferencePoint,
    ProductReferenceSweep, ProductRomPromotionEvidence, ProductStructuralPromotionGate,
    ProductStructuralReferenceEvidence, ProductThermalPromotionGate,
    ProductThermalReferenceEvidence, ProductValidationObservable, adjacent_mesh_convergence_error,
};
pub use projected_reanalysis::{
    BoundaryLoop, ProjectedReanalysisConfig, ProjectedReanalysisError, ProjectedReanalysisReport,
    ReanalysisLevel, build_body_fitted_mesh, extract_binary_design, extract_steel_boundary,
    loops_are_body_fitted, motor_design_elements, optimize_and_reanalyze,
    reanalyze_physical_design, reanalyze_projected_design,
};
pub use resolved_acoustics::{
    ResolvedAcousticConfig, ResolvedAcousticModel, ResolvedAcousticSolution,
    ResolvedCandidateResiduals, ResolvedFrequencyContinuation, ResolvedGuessKind,
    ResolvedLinearSystem, ResolvedPreconditionerBuild, ResolvedPreconditionerDiagnostics,
    ResolvedPreconditionerFallback, ResolvedPreconditionerKind, ResolvedPreconditionerReuse,
    ResolvedSolveDiagnostics, ResolvedSpeakerAcousticError,
};
pub use resolved_coupling::{
    ResolvedCouplingConfig, ResolvedCouplingError, ResolvedCouplingPoint, ResolvedLoadCoupler,
};
pub use sector_3d::{
    AxisymmetricSectorComparison, Sector3dConfig, Sector3dError, Sector3dModel, Sector3dSolution,
};
pub use structural_acoustic::{
    MeasuredStructuralAcousticPoint, StructuralAcousticFeedback, StructuralAcousticMetricReport,
    StructuralAcousticModalBlock, StructuralAcousticMode, StructuralAcousticReferenceReport,
    StructuralAcousticReferenceTolerances, StructuralAcousticResponse,
    correlate_structural_acoustic_reference,
};
pub use structural_constraints::{
    CabinetConstraintSpec, CabinetPanelEvaluation, CabinetPanelLimits, MovingAssemblyLimits,
    StructuralConstraintError, StructuralConstraintSet, StructuralEvaluation,
    cabinet_panel_response, constraint_labels, constraint_margins, moving_assembly_excursion_m,
    moving_assembly_peak_stress_pa, moving_assembly_region_name, to_solver_constraints,
};
pub use thermal_network::{
    DriverThermalNetworkConfig, THERMAL_NODE_COUNT, ThermalNetworkPower, ThermalNetworkSpec,
    ThermalNetworkState, ThermalNode,
};
pub use uncertainty::{
    SpeakerToleranceSpec, SpeakerUncertaintyCase, UncertaintyError, UncertaintySampling,
    generate_uncertainty_cases,
};
pub use job_service::{JobRecord, JobServiceError, JobState};
pub use waveguide_design::ProductWaveguideDesign;
pub use workbench::{
    FidelityLabel, MaterialReference, ProvenanceEnvelope, WORKBENCH_SCHEMA_VERSION,
    WorkbenchError, WorkbenchProject, WorkbenchStudy, WorkbenchStudyKind,
};

mod distortion_reference;
mod enclosure;
mod large_signal;
mod lumped_nonlinear;
mod port_noise;
mod thermal_network;

pub use distortion_reference::{
    DistortionMetricReport, DistortionReferenceReport, DistortionReferenceTolerances,
    MeasuredDistortionPoint, correlate_distortion_reference,
};
pub use enclosure::{
    EnclosureSpec, PassiveRadiatorSpec, PortSpec, baffled_piston_directivity,
    baffled_piston_far_field_pressure, sound_pressure_level_db,
};
pub use large_signal::{
    LargeSignalAcousticLoad, LargeSignalProvenance, LargeSignalResult, LargeSignalSpec,
    MovingAssemblyBreakupSpec, MovingAssemblyContactSpec, MovingAssemblyFatigueSpec,
    MovingAssemblyNonlinearity, MovingAssemblyViscoelasticSpec, ThermalCompressionDiagnostics,
    ThermalCompressionResult, ThermalCompressionSpec,
};
pub use lumped_nonlinear::{
    AcousticCircuitState, BoxComplianceModel, LumpedNonlinearAcousticSpec,
    LumpedNonlinearCompressionPoint, LumpedNonlinearCompressionSweep, LumpedNonlinearDiagnostics,
    LumpedNonlinearEnergyBalance, PortLossCalibration,
};
pub use port_noise::{
    PORT_NOISE_SCHEMA_VERSION, PortNoiseInput, PortNoiseMethod, PortNoiseObserver,
    PortNoiseObserverResult, PortNoiseResult, PortNoiseSpec, PortNoiseSpectrumPoint,
    analyze_port_noise, analyze_port_noise_input, free_turbulence_power_trend,
    maximum_relative_spectrum_change,
};

/// Validation error for a speaker-modeling input or derived study.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ModelingError {
    #[error("{field} must be finite")]
    NonFinite { field: &'static str },
    #[error("{field} must be {constraint}")]
    OutOfRange {
        field: &'static str,
        constraint: &'static str,
    },
    #[error("frequency_hz must be non-empty, finite, positive, and strictly increasing")]
    InvalidFrequencyGrid,
    #[error("study name must not be blank")]
    BlankStudyName,
    #[error("motor material list must not be empty")]
    EmptyMotorMaterials,
    #[error("study step list must not be empty")]
    EmptyStudySteps,
    #[error("study step names and purposes must not be blank")]
    BlankStudyStep,
    #[error("at least one dynamic study requirement must be enabled")]
    NoDynamicStudyRequested,
    #[error("study symmetry does not match its geometry")]
    SymmetryDoesNotMatchGeometry,
    #[error("study steps do not match the selected dynamic study")]
    StepsDoNotMatchDynamicStudy,
    #[error("steady-state electrothermal iteration did not converge")]
    ThermalIterationDidNotConverge,
    #[error("thermal response power balance did not close")]
    ThermalPowerBalanceDidNotClose,
    #[error("radiation impedance must be finite and passive")]
    InvalidRadiationImpedance,
    #[error("invalid distributed FEM-BEM reference evidence: {0}")]
    InvalidReferenceEvidence(String),
    #[error("large-signal acoustic coupling did not converge at step {step}: residual {residual}")]
    LargeSignalAcousticCouplingDidNotConverge { step: usize, residual: String },
    #[error("Gauss-Legendre node iteration did not converge for order {order}")]
    QuadratureDidNotConverge { order: usize },
    #[error("aeroacoustic post-processing failed: {0}")]
    ExternalAeroacoustics(String),
    #[error("enclosure variant is not supported by this study: {0}")]
    UnsupportedEnclosure(&'static str),
}

fn require_finite(field: &'static str, value: f64) -> Result<(), ModelingError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ModelingError::NonFinite { field })
    }
}

fn require_positive(field: &'static str, value: f64) -> Result<(), ModelingError> {
    require_finite(field, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(ModelingError::OutOfRange {
            field,
            constraint: "greater than zero",
        })
    }
}

fn require_nonnegative(field: &'static str, value: f64) -> Result<(), ModelingError> {
    require_finite(field, value)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(ModelingError::OutOfRange {
            field,
            constraint: "non-negative",
        })
    }
}

fn validate_frequency_grid(frequencies: &[f64]) -> Result<(), ModelingError> {
    if frequencies.is_empty()
        || frequencies
            .iter()
            .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
        || frequencies.windows(2).any(|pair| pair[0] >= pair[1])
    {
        Err(ModelingError::InvalidFrequencyGrid)
    } else {
        Ok(())
    }
}

/// High-level analysis mode for the dynamic driver solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DynamicStudyKind {
    /// Harmonic small-signal solve linearized around the stationary magnetic bias.
    FrequencyDomainPerturbation,
    /// Direct time integration for transients, nonlinearities, or higher harmonics.
    TimeDependent,
}

/// Geometric reduction chosen for the finite-element model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SymmetryModel {
    /// Rotationally symmetric 2D meridian model.
    Axisymmetric2d,
    /// Three-dimensional sector model with periodic boundaries.
    Sector3d { angle_degrees: f64 },
    /// Full three-dimensional model.
    Full3d,
}

/// Acoustic loss model for narrow gaps and small cavities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcousticLossModel {
    /// Lossless pressure acoustics.
    PressureAcoustics,
    /// Effective narrow-region model for thin gaps.
    NarrowRegionAcoustics,
    /// Boundary-layer impedance correction on a pressure-acoustics domain.
    ThermoviscousBoundaryLayerImpedance,
    /// Full thermoviscous acoustics with velocity and temperature variables.
    FullThermoviscousAcoustics,
}

/// Representation used for a moving driver part.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartRepresentation {
    /// Thin shell/surface representation.
    Shell,
    /// Volumetric solid representation.
    Solid,
}

/// Solver family recommended by the modeling layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolverFamily {
    DirectSparse,
    FgmresShiftedLaplacianIlu,
    NewmarkPredictorCorrector,
}

/// Voice-coil parameters used by the example studies.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VoiceCoilSpec {
    pub resistance_ohm: f64,
    pub inductance_h: f64,
    pub bl_tm: f64,
    /// Active conductor turns represented in the air-gap BL integral.
    pub active_turns: f64,
}

impl VoiceCoilSpec {
    pub fn validate(self) -> Result<(), ModelingError> {
        require_positive("voice_coil.resistance_ohm", self.resistance_ohm)?;
        require_nonnegative("voice_coil.inductance_h", self.inductance_h)?;
        require_finite("voice_coil.bl_tm", self.bl_tm)?;
        require_positive("voice_coil.active_turns", self.active_turns)
    }

    /// Checked conversion; fallible instead of panicking on
    /// non-physical parameters.
    pub fn try_as_electrodynamic_coupling(self) -> Result<ElectrodynamicCoupling, ModelingError> {
        self.validate()?;
        Ok(ElectrodynamicCoupling {
            resistance: self.resistance_ohm,
            inductance: self.inductance_h,
            bl: self.bl_tm,
            coil_dofs: vec![(0, 1.0)],
        })
    }

    /// Checked conversion; fallible instead of panicking on
    /// non-physical parameters.
    pub fn try_as_linear_voice_coil(self) -> Result<VoiceCoilParams, ModelingError> {
        self.validate()?;
        Ok(VoiceCoilParams::linear(
            self.resistance_ohm,
            self.inductance_h,
            self.bl_tm,
        ))
    }

    #[deprecated(note = "panics on non-physical parameters; use try_as_electrodynamic_coupling")]
    pub fn as_electrodynamic_coupling(self) -> ElectrodynamicCoupling {
        self.try_as_electrodynamic_coupling()
            .expect("voice-coil parameters must be finite and physical")
    }

    #[deprecated(note = "panics on non-physical parameters; use try_as_linear_voice_coil")]
    pub fn as_linear_voice_coil(self) -> VoiceCoilParams {
        self.try_as_linear_voice_coil()
            .expect("voice-coil parameters must be finite and physical")
    }
}

/// Lumped moving system parameters for quick electrodynamic checks.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MovingSystemSpec {
    pub moving_mass_kg: f64,
    pub mechanical_resistance_kg_per_s: f64,
    pub stiffness_n_per_m: f64,
    /// One-way linear excursion limit Xmax (m). Optional: `None` leaves the
    /// excursion unconstrained. Consumed by the structural-constraint
    /// evaluators in [`crate::structural_constraints`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub xmax_m: Option<f64>,
}

impl MovingSystemSpec {
    pub fn validate(self) -> Result<(), ModelingError> {
        require_positive("moving_system.moving_mass_kg", self.moving_mass_kg)?;
        require_nonnegative(
            "moving_system.mechanical_resistance_kg_per_s",
            self.mechanical_resistance_kg_per_s,
        )?;
        require_positive("moving_system.stiffness_n_per_m", self.stiffness_n_per_m)?;
        if let Some(xmax_m) = self.xmax_m {
            require_positive("moving_system.xmax_m", xmax_m)?;
        }
        Ok(())
    }

    #[deprecated(note = "panics on non-physical parameters; use try_resonance_hz")]
    pub fn resonance_hz(self) -> f64 {
        self.try_resonance_hz()
            .expect("moving-system resonance must be finite and physical")
    }

    pub fn try_resonance_hz(self) -> Result<f64, ModelingError> {
        self.validate()?;
        let resonance_hz =
            (self.stiffness_n_per_m / self.moving_mass_kg).sqrt() / std::f64::consts::TAU;
        require_positive("moving_system.resonance_hz", resonance_hz)?;
        Ok(resonance_hz)
    }

    #[deprecated(note = "panics on non-physical parameters; use try_mechanical_impedance")]
    pub fn mechanical_impedance(self, frequency_hz: f64) -> Complex64 {
        self.try_mechanical_impedance(frequency_hz)
            .expect("mechanical impedance requires physical parameters and positive frequency")
    }

    pub fn try_mechanical_impedance(self, frequency_hz: f64) -> Result<Complex64, ModelingError> {
        self.validate()?;
        require_positive("frequency_hz", frequency_hz)?;
        let omega = std::f64::consts::TAU * frequency_hz;
        require_positive("angular_frequency_rad_per_s", omega)?;
        let impedance = Complex64::new(
            self.mechanical_resistance_kg_per_s,
            omega * self.moving_mass_kg - self.stiffness_n_per_m / omega,
        );
        require_finite("mechanical_impedance.re", impedance.re)?;
        require_finite("mechanical_impedance.im", impedance.im)?;
        Ok(impedance)
    }
}

/// Rotational speaker-driver geometry at the level needed for study selection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DriverGeometry {
    pub voice_coil_radius_m: f64,
    pub diaphragm_radius_m: f64,
    pub diaphragm_half_angle_degrees: f64,
    pub air_gap_m: f64,
    pub has_non_axisymmetric_basket: bool,
    /// Periodic 3D sector angle. `None` means the geometry is axisymmetric.
    pub sector_angle_degrees: Option<f64>,
}

impl DriverGeometry {
    pub fn validate(&self) -> Result<(), ModelingError> {
        require_positive("geometry.voice_coil_radius_m", self.voice_coil_radius_m)?;
        require_positive("geometry.diaphragm_radius_m", self.diaphragm_radius_m)?;
        if self.voice_coil_radius_m > self.diaphragm_radius_m {
            return Err(ModelingError::OutOfRange {
                field: "geometry.voice_coil_radius_m",
                constraint: "no greater than geometry.diaphragm_radius_m",
            });
        }
        require_finite(
            "geometry.diaphragm_half_angle_degrees",
            self.diaphragm_half_angle_degrees,
        )?;
        if !(0.0..90.0).contains(&self.diaphragm_half_angle_degrees) {
            return Err(ModelingError::OutOfRange {
                field: "geometry.diaphragm_half_angle_degrees",
                constraint: "between zero and 90 degrees",
            });
        }
        require_positive("geometry.air_gap_m", self.air_gap_m)?;
        if let Some(angle) = self.sector_angle_degrees {
            require_positive("geometry.sector_angle_degrees", angle)?;
            if !is_proper_divisor_of_360(angle) {
                return Err(ModelingError::OutOfRange {
                    field: "geometry.sector_angle_degrees",
                    constraint: "a proper divisor of 360 degrees smaller than 360",
                });
            }
        }
        Ok(())
    }
}

/// Representation choices for moving driver parts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MovingPartRepresentations {
    pub cone: PartRepresentation,
    pub surround: PartRepresentation,
    pub spider: PartRepresentation,
    pub former: PartRepresentation,
    pub coil: PartRepresentation,
}

impl Default for MovingPartRepresentations {
    fn default() -> Self {
        Self {
            cone: PartRepresentation::Shell,
            surround: PartRepresentation::Shell,
            spider: PartRepresentation::Shell,
            former: PartRepresentation::Shell,
            coil: PartRepresentation::Solid,
        }
    }
}

/// Magnetic-material assignment for the motor cross-section.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MotorMaterialRole {
    AirGap,
    SteelTopPlate,
    SteelPolePiece,
    PermanentMagnet { br_t: f64, direction: [f64; 3] },
    CopperCoil,
}

impl MotorMaterialRole {
    pub fn validate(&self) -> Result<(), ModelingError> {
        if let Self::PermanentMagnet { br_t, direction } = self {
            require_positive("permanent_magnet.br_t", *br_t)?;
            for component in direction {
                require_finite("permanent_magnet.direction", *component)?;
            }
            let norm_squared: f64 = direction
                .iter()
                .map(|component| component * component)
                .sum();
            if norm_squared <= 1.0e-30 {
                return Err(ModelingError::OutOfRange {
                    field: "permanent_magnet.direction",
                    constraint: "a non-zero vector",
                });
            }
        }
        Ok(())
    }

    pub fn as_fem_material(&self) -> MagneticMaterial {
        self.validate()
            .expect("motor material parameters must be finite and physical");
        match self {
            Self::AirGap | Self::CopperCoil => MagneticMaterial::Air,
            Self::SteelTopPlate | Self::SteelPolePiece => MagneticMaterial::Linear { mu_r: 1200.0 },
            Self::PermanentMagnet { br_t, direction } => {
                MagneticMaterial::neodymium(*br_t, *direction)
            }
        }
    }
}

/// A single finite-element study step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StudyStep {
    pub name: String,
    pub purpose: String,
    pub solver: SolverFamily,
}

/// Complete speaker-driver modeling study.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerDriverStudy {
    pub name: String,
    pub geometry: DriverGeometry,
    pub symmetry: SymmetryModel,
    pub dynamic_study: DynamicStudyKind,
    pub acoustic_loss_model: AcousticLossModel,
    pub moving_parts: MovingPartRepresentations,
    pub motor_materials: Vec<MotorMaterialRole>,
    pub voice_coil: VoiceCoilSpec,
    pub moving_system: MovingSystemSpec,
    /// Rear acoustic load. The default is free air for backward compatibility.
    #[serde(default)]
    pub enclosure: EnclosureSpec,
    /// Configured RMS drive voltage used by the default study response.
    pub drive_voltage_rms_v: f64,
    pub acoustic_medium: AcousticMedium,
    pub frequency_hz: Vec<f64>,
    pub steps: Vec<StudyStep>,
}

/// Homogeneous acoustic-medium properties used for loading and FEM sweep assembly.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AcousticMedium {
    pub density_kg_per_m3: f64,
    pub sound_speed_m_per_s: f64,
    pub dynamic_viscosity_pa_s: f64,
    pub thermal_diffusivity_m2_per_s: f64,
}

impl AcousticMedium {
    /// Dry air near 20 °C at one atmosphere.
    pub const fn standard_air() -> Self {
        Self {
            density_kg_per_m3: 1.204,
            sound_speed_m_per_s: 343.0,
            dynamic_viscosity_pa_s: 1.814e-5,
            thermal_diffusivity_m2_per_s: 2.16e-5,
        }
    }

    /// Convert the shared inviscid medium fields to the typed XEM exchange
    /// boundary for a selected frequency.
    ///
    /// Viscous and thermal loss properties remain local extensions used by
    /// speaker-modeling; XEM carries the frequency, density, speed, and
    /// convention metadata shared by BEM/FEM/application consumers.
    pub fn to_xem(self, frequency_hz: f64) -> Result<sonium_xem::AcousticMedium, ModelingError> {
        self.validate()?;
        sonium_xem::AcousticMedium::try_new(
            frequency_hz,
            self.sound_speed_m_per_s,
            self.density_kg_per_m3,
        )
        .map_err(|_| ModelingError::OutOfRange {
            field: "acoustic_medium.xem",
            constraint: "frequency, speed, density, and convention metadata must be valid",
        })
    }

    pub fn validate(self) -> Result<(), ModelingError> {
        require_positive("acoustic_medium.density_kg_per_m3", self.density_kg_per_m3)?;
        require_positive(
            "acoustic_medium.sound_speed_m_per_s",
            self.sound_speed_m_per_s,
        )?;
        require_positive(
            "acoustic_medium.dynamic_viscosity_pa_s",
            self.dynamic_viscosity_pa_s,
        )?;
        require_positive(
            "acoustic_medium.thermal_diffusivity_m2_per_s",
            self.thermal_diffusivity_m2_per_s,
        )
    }
}

impl Default for AcousticMedium {
    fn default() -> Self {
        Self::standard_air()
    }
}

/// Viscous and thermal penetration depths used to pick a speaker-gap acoustic model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundaryLayerSpec {
    pub viscous_delta_m: f64,
    pub thermal_delta_m: f64,
}

impl BoundaryLayerSpec {
    pub fn at_frequency(frequency_hz: f64, medium: AcousticMedium) -> Result<Self, ModelingError> {
        require_positive("frequency_hz", frequency_hz)?;
        medium.validate()?;
        let omega = std::f64::consts::TAU * frequency_hz;
        require_positive("angular_frequency_rad_per_s", omega)?;
        let boundary_layers = Self {
            viscous_delta_m: (2.0 * medium.dynamic_viscosity_pa_s
                / (medium.density_kg_per_m3 * omega))
                .sqrt(),
            thermal_delta_m: (2.0 * medium.thermal_diffusivity_m2_per_s / omega).sqrt(),
        };
        boundary_layers.validate()?;
        Ok(boundary_layers)
    }

    pub fn validate(self) -> Result<(), ModelingError> {
        require_positive("boundary_layers.viscous_delta_m", self.viscous_delta_m)?;
        require_positive("boundary_layers.thermal_delta_m", self.thermal_delta_m)
    }
}

impl Default for BoundaryLayerSpec {
    fn default() -> Self {
        Self::at_frequency(1_000.0, AcousticMedium::standard_air())
            .expect("standard air at 1 kHz is a valid boundary-layer reference")
    }
}

/// Serializable row extracted from FEM's stable speaker-response contract.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ResponseRow {
    pub frequency_hz: f64,
    pub voltage_real_v: f64,
    pub voltage_imag_v: f64,
    pub impedance_real_ohm: f64,
    pub impedance_imag_ohm: f64,
    pub current_real_a: f64,
    pub current_imag_a: f64,
    pub current_abs_a: f64,
    pub bl_force_real_n: f64,
    pub bl_force_imag_n: f64,
    pub bl_force_abs_n: f64,
    pub velocity_real_m_per_s: f64,
    pub velocity_imag_m_per_s: f64,
    pub velocity_abs_m_per_s: f64,
    pub displacement_real_m: f64,
    pub displacement_imag_m: f64,
    pub displacement_abs_m: f64,
    pub input_power_w: f64,
    pub coil_loss_w: f64,
    pub mechanical_damping_power_w: f64,
    pub radiated_power_w: f64,
    #[serde(default)]
    pub enclosure_loss_power_w: f64,
    pub power_balance_error_w: f64,
}

impl ResponseRow {
    pub fn from_response(response: &SpeakerResponse, study: &SpeakerDriverStudy) -> Self {
        let input_power_w = (response.voltage * response.current.conj()).re;
        let coil_loss_w = study.voice_coil.resistance_ohm * response.current.norm_sqr();
        let mechanical_damping_power_w =
            study.moving_system.mechanical_resistance_kg_per_s * response.coil_velocity.norm_sqr();
        let radiation = study
            .mechanical_radiation_impedance(response.frequency_hz)
            .expect("response row requires a valid speaker study");
        let radiated_power_w = radiation.re * response.coil_velocity.norm_sqr();
        let enclosure_loss_power_w = study
            .enclosure_mechanical_impedance(response.frequency_hz)
            .expect("response row requires a valid enclosure")
            .re
            * response.coil_velocity.norm_sqr();
        Self {
            frequency_hz: response.frequency_hz,
            voltage_real_v: response.voltage.re,
            voltage_imag_v: response.voltage.im,
            impedance_real_ohm: response.impedance.re,
            impedance_imag_ohm: response.impedance.im,
            current_real_a: response.current.re,
            current_imag_a: response.current.im,
            current_abs_a: response.current.norm(),
            bl_force_real_n: response.bl_force.re,
            bl_force_imag_n: response.bl_force.im,
            bl_force_abs_n: response.bl_force.norm(),
            velocity_real_m_per_s: response.coil_velocity.re,
            velocity_imag_m_per_s: response.coil_velocity.im,
            velocity_abs_m_per_s: response.coil_velocity.norm(),
            displacement_real_m: response.diaphragm_displacement.re,
            displacement_imag_m: response.diaphragm_displacement.im,
            displacement_abs_m: response.diaphragm_displacement.norm(),
            input_power_w,
            coil_loss_w,
            mechanical_damping_power_w,
            radiated_power_w,
            enclosure_loss_power_w,
            power_balance_error_w: input_power_w
                - coil_loss_w
                - mechanical_damping_power_w
                - radiated_power_w
                - enclosure_loss_power_w,
        }
    }
}

/// Serializable summary of the FEM objects backing a detailed example.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FemExampleSummary {
    pub bias_solve_name: String,
    pub perturbation_solve_name: String,
    pub reuse_bias_tangent: bool,
    pub average_flux_density_t: [f64; 3],
    pub distributed_lorentz_bl_tm: f64,
    /// True because this report has not executed the outstanding end-to-end FEM solve.
    pub magnetic_bias_is_representative: bool,
    /// Gap model selected from standard-air boundary layers at 1 kHz.
    pub gap_model: AcousticLossModel,
    pub acoustic_structure_at_1khz: AcousticCouplingSummary,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AcousticCouplingSummary {
    pub fluid_to_structure_real: f64,
    pub fluid_to_structure_imag: f64,
    pub structure_to_fluid_real: f64,
    pub structure_to_fluid_imag: f64,
}

impl From<AcousticStructureBoundaryCoupling> for AcousticCouplingSummary {
    fn from(coupling: AcousticStructureBoundaryCoupling) -> Self {
        Self {
            fluid_to_structure_real: coupling.fluid_to_structure.re,
            fluid_to_structure_imag: coupling.fluid_to_structure.im,
            structure_to_fluid_real: coupling.structure_to_fluid.re,
            structure_to_fluid_imag: coupling.structure_to_fluid.im,
        }
    }
}

/// JSON-friendly detailed example report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetailedExampleReport {
    pub study: SpeakerDriverStudy,
    pub fem: FemExampleSummary,
    pub response_provenance: ResponseProvenance,
    pub responses: Vec<ResponseRow>,
}

/// Numerical model that produced the response table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResponseProvenance {
    /// Lumped electrical/mechanical parameters with an analytical rigid-piston load.
    LumpedElectromechanicalWithBaffledPistonRadiation,
}

impl SpeakerDriverStudy {
    /// COMSOL-style detailed driver workflow: stationary magnetic bias followed
    /// by a harmonic perturbation solve for small-signal response.
    pub fn detailed_frequency_domain_driver() -> Self {
        let geometry = DriverGeometry {
            voice_coil_radius_m: 12.5e-3,
            diaphragm_radius_m: 52.5e-3,
            diaphragm_half_angle_degrees: 32.0,
            air_gap_m: 0.75e-3,
            has_non_axisymmetric_basket: false,
            sector_angle_degrees: None,
        };

        Self::from_requirements(StudyRequirements {
            name: "detailed-frequency-domain-driver".to_string(),
            geometry,
            voice_coil: VoiceCoilSpec {
                resistance_ohm: 5.8,
                inductance_h: 0.42e-3,
                bl_tm: 6.8,
                active_turns: 75.0,
            },
            moving_system: MovingSystemSpec {
                moving_mass_kg: 14.0e-3,
                mechanical_resistance_kg_per_s: 1.1,
                stiffness_n_per_m: 950.0,
                xmax_m: None,
            },
            harmonic_small_signal: true,
            transient_or_pulse: false,
            nonlinear_large_signal: false,
            frequency_hz: octave_sweep(20.0, 20_000.0, 6),
            voltage_rms_v: 2.83,
            acoustic_medium: AcousticMedium::standard_air(),
        })
    }

    /// Transient example for pulse and nonlinear large-signal studies.
    pub fn transient_large_signal_driver() -> Self {
        let geometry = DriverGeometry {
            voice_coil_radius_m: 12.5e-3,
            diaphragm_radius_m: 52.5e-3,
            diaphragm_half_angle_degrees: 32.0,
            air_gap_m: 0.75e-3,
            has_non_axisymmetric_basket: false,
            sector_angle_degrees: None,
        };

        Self::from_requirements(StudyRequirements {
            name: "transient-large-signal-driver".to_string(),
            geometry,
            voice_coil: VoiceCoilSpec {
                resistance_ohm: 5.8,
                inductance_h: 0.42e-3,
                bl_tm: 6.8,
                active_turns: 75.0,
            },
            moving_system: MovingSystemSpec {
                moving_mass_kg: 14.0e-3,
                mechanical_resistance_kg_per_s: 1.1,
                stiffness_n_per_m: 950.0,
                xmax_m: None,
            },
            harmonic_small_signal: false,
            transient_or_pulse: true,
            nonlinear_large_signal: true,
            frequency_hz: vec![40.0, 80.0, 160.0, 320.0],
            voltage_rms_v: 8.0,
            acoustic_medium: AcousticMedium::standard_air(),
        })
    }

    /// Build a study from high-level modeling requirements.
    ///
    /// # Panics
    ///
    /// Panics when the requirements are invalid. Use [`Self::try_from_requirements`]
    /// when inputs originate outside the program.
    pub fn from_requirements(requirements: StudyRequirements) -> Self {
        Self::try_from_requirements(requirements)
            .expect("invalid speaker-driver study requirements")
    }

    /// Fallible constructor for user-provided or deserialized requirements.
    pub fn try_from_requirements(requirements: StudyRequirements) -> Result<Self, ModelingError> {
        requirements.validate()?;
        let symmetry = choose_symmetry(&requirements.geometry);
        let dynamic_study = choose_dynamic_study(
            requirements.harmonic_small_signal,
            requirements.transient_or_pulse,
            requirements.nonlinear_large_signal,
        );
        let acoustic_loss_model = choose_acoustic_loss_model_at_frequency(
            requirements.geometry.air_gap_m,
            requirements.frequency_hz[0],
            requirements.acoustic_medium,
        )?;
        let moving_parts = MovingPartRepresentations::default();

        let study = Self {
            name: requirements.name,
            geometry: requirements.geometry,
            symmetry,
            dynamic_study,
            acoustic_loss_model,
            moving_parts,
            motor_materials: default_motor_materials(),
            voice_coil: requirements.voice_coil,
            moving_system: requirements.moving_system,
            enclosure: EnclosureSpec::FreeAir,
            drive_voltage_rms_v: requirements.voltage_rms_v,
            acoustic_medium: requirements.acoustic_medium,
            frequency_hz: requirements.frequency_hz,
            steps: build_steps(dynamic_study),
        };
        study.validate()?;
        Ok(study)
    }

    /// Validate all public study inputs before constructing FEM adapters or responses.
    pub fn validate(&self) -> Result<(), ModelingError> {
        if self.name.trim().is_empty() {
            return Err(ModelingError::BlankStudyName);
        }
        self.geometry.validate()?;
        if self.symmetry != try_choose_symmetry(&self.geometry)? {
            return Err(ModelingError::SymmetryDoesNotMatchGeometry);
        }
        self.voice_coil.validate()?;
        self.moving_system.validate()?;
        self.enclosure.validate()?;
        require_nonnegative("drive_voltage_rms_v", self.drive_voltage_rms_v)?;
        self.acoustic_medium.validate()?;
        validate_frequency_grid(&self.frequency_hz)?;
        if self.motor_materials.is_empty() {
            return Err(ModelingError::EmptyMotorMaterials);
        }
        for material in &self.motor_materials {
            material.validate()?;
        }
        if self.steps.is_empty() {
            return Err(ModelingError::EmptyStudySteps);
        }
        if self
            .steps
            .iter()
            .any(|step| step.name.trim().is_empty() || step.purpose.trim().is_empty())
        {
            return Err(ModelingError::BlankStudyStep);
        }
        let expected_dynamic_solver = match self.dynamic_study {
            DynamicStudyKind::FrequencyDomainPerturbation => {
                SolverFamily::FgmresShiftedLaplacianIlu
            }
            DynamicStudyKind::TimeDependent => SolverFamily::NewmarkPredictorCorrector,
        };
        if self.steps[0].solver != SolverFamily::DirectSparse
            || self.steps.get(1).map(|step| step.solver) != Some(expected_dynamic_solver)
        {
            return Err(ModelingError::StepsDoNotMatchDynamicStudy);
        }
        Ok(())
    }

    /// Checked voice-coil coupling; fallible instead of panicking on
    /// non-physical parameters.
    pub fn try_fem_voice_coil(&self) -> Result<ElectrodynamicCoupling, ModelingError> {
        self.voice_coil.try_as_electrodynamic_coupling()
    }

    #[deprecated(note = "panics on non-physical parameters; use try_fem_voice_coil")]
    pub fn fem_voice_coil(&self) -> ElectrodynamicCoupling {
        self.try_fem_voice_coil()
            .expect("voice-coil parameters must be finite and physical")
    }

    #[deprecated(note = "panics on empty study steps; use try_fem_magnetic_workflow")]
    pub fn fem_magnetic_workflow(&self) -> MagneticPerturbationWorkflow {
        self.try_fem_magnetic_workflow()
            .expect("speaker study must contain at least one study step")
    }

    pub fn try_fem_magnetic_workflow(&self) -> Result<MagneticPerturbationWorkflow, ModelingError> {
        self.validate()?;
        let bias_step = self.steps.first().ok_or(ModelingError::EmptyStudySteps)?;
        Ok(MagneticPerturbationWorkflow {
            bias_solve_name: bias_step.name.replace('-', "_"),
            perturbation_solve_name: self
                .steps
                .get(1)
                .map(|step| step.name.replace('-', "_"))
                .unwrap_or_else(|| "dynamic_speaker_response".to_string()),
            reuse_bias_tangent: self.dynamic_study == DynamicStudyKind::FrequencyDomainPerturbation,
        })
    }

    /// Representative bias state for reporting and evidence ladders.
    ///
    /// Validity precondition: the study must pass [`Self::validate`].
    #[track_caller]
    pub fn representative_magnetic_bias_state(&self) -> MagneticBiasState {
        self.validate()
            .expect("representative bias state requires a valid speaker study");
        let b_radial_t = self.representative_gap_flux_density_t();
        MagneticBiasState {
            vector_potential_dofs: vec![0.0, b_radial_t * self.geometry.air_gap_m],
            b_at_quadrature: vec![
                [b_radial_t * 0.98, 0.0, 0.0],
                [b_radial_t, 0.0, 0.0],
                [b_radial_t * 1.02, 0.0, 0.0],
            ],
        }
    }

    /// Representative Lorentz coupling for reporting and evidence ladders.
    ///
    /// Validity precondition: the study must pass [`Self::validate`].
    #[track_caller]
    pub fn representative_distributed_lorentz(&self) -> DistributedLorentzCoupling {
        self.validate()
            .expect("representative Lorentz coupling requires a valid speaker study");
        let turns = self.voice_coil.active_turns;
        let circumference_m = std::f64::consts::TAU * self.geometry.voice_coil_radius_m;
        DistributedLorentzCoupling {
            points: vec![LorentzIntegrationPoint {
                dof_weights: vec![(0, [1.0, 0.0, 0.0])],
                magnetic_flux_density: [0.0, 0.0, self.representative_gap_flux_density_t()],
                conductor_direction: [0.0, 1.0, 0.0],
                jacobian_weight: turns * circumference_m,
            }],
        }
    }

    pub fn fem_gap_model(&self, boundary_layers: BoundaryLayerSpec) -> SpeakerGapModel {
        self.geometry
            .validate()
            .expect("gap-model selection requires valid geometry");
        boundary_layers
            .validate()
            .expect("gap-model selection requires valid boundary layers");
        recommend_speaker_gap_model(
            self.geometry.air_gap_m,
            boundary_layers.viscous_delta_m,
            boundary_layers.thermal_delta_m,
        )
    }

    /// Acoustic-structure boundary coupling at one frequency.
    ///
    /// Validity precondition: the study must pass [`Self::validate`] with
    /// positive `frequency_hz`.
    #[track_caller]
    pub fn fem_acoustic_coupling_at(&self, frequency_hz: f64) -> AcousticStructureBoundaryCoupling {
        self.validate()
            .expect("acoustic coupling requires a valid speaker study");
        require_positive("frequency_hz", frequency_hz)
            .expect("acoustic coupling requires positive frequency");
        CoupledSweepPoint {
            frequency_hz,
            sound_speed: self.acoustic_medium.sound_speed_m_per_s,
            fluid_density: self.acoustic_medium.density_kg_per_m3,
        }
        .coupling()
    }

    pub fn fem_sweep_points(&self) -> Vec<CoupledSweepPoint> {
        self.validate()
            .expect("FEM sweep points require a valid speaker study");
        self.frequency_hz
            .iter()
            .map(|&frequency_hz| CoupledSweepPoint {
                frequency_hz,
                sound_speed: self.acoustic_medium.sound_speed_m_per_s,
                fluid_density: self.acoustic_medium.density_kg_per_m3,
            })
            .collect()
    }

    pub fn blocked_impedance(&self, frequency_hz: f64) -> Complex64 {
        self.voice_coil
            .validate()
            .expect("blocked impedance requires valid voice-coil parameters");
        require_positive("frequency_hz", frequency_hz)
            .expect("blocked impedance requires positive frequency");
        let omega = std::f64::consts::TAU * frequency_hz;
        Complex64::new(
            self.voice_coil.resistance_ohm,
            omega * self.voice_coil.inductance_h,
        )
    }

    #[deprecated(note = "panics on invalid study; use try_small_signal_impedance")]
    pub fn small_signal_impedance(&self, frequency_hz: f64) -> Complex64 {
        self.try_small_signal_impedance(frequency_hz)
            .expect("small-signal impedance requires a valid study and positive frequency")
    }

    pub fn try_small_signal_impedance(
        &self,
        frequency_hz: f64,
    ) -> Result<Complex64, ModelingError> {
        self.validate()?;
        require_positive("frequency_hz", frequency_hz)?;
        let coupling = self.try_fem_voice_coil()?;
        let z_mechanical = self.total_mechanical_impedance(frequency_hz)?;
        Ok(coupling.total_impedance(std::f64::consts::TAU * frequency_hz, z_mechanical))
    }

    /// Complete lumped mechanical load at the coil, including front radiation
    /// and any configured rear enclosure.
    pub fn total_mechanical_impedance(
        &self,
        frequency_hz: f64,
    ) -> Result<Complex64, ModelingError> {
        Ok(self.moving_system.try_mechanical_impedance(frequency_hz)?
            + self.mechanical_radiation_impedance(frequency_hz)?
            + self.enclosure_mechanical_impedance(frequency_hz)?)
    }

    /// Rear enclosure impedance transformed from pressure/volume velocity to
    /// force/diaphragm velocity by the squared piston area.
    pub fn enclosure_mechanical_impedance(
        &self,
        frequency_hz: f64,
    ) -> Result<Complex64, ModelingError> {
        self.geometry.validate()?;
        let area = std::f64::consts::PI * self.geometry.diaphragm_radius_m.powi(2);
        require_positive("piston_area_m2", area)?;
        Ok(self
            .enclosure
            .acoustic_impedance(frequency_hz, self.acoustic_medium)?
            * area
            * area)
    }

    /// Solve the conjugate coil/magnet/gap/cabinet thermal network associated
    /// with this validated driver study.
    pub fn solve_thermal_network(
        &self,
        specification: ThermalNetworkSpec,
        power: ThermalNetworkPower,
    ) -> Result<ThermalNetworkState, ModelingError> {
        self.validate()?;
        specification.solve_steady_state(power)
    }

    /// Mechanical radiation impedance `force / piston velocity` for a rigid
    /// circular piston mounted in an infinite baffle.
    pub fn mechanical_radiation_impedance(
        &self,
        frequency_hz: f64,
    ) -> Result<Complex64, ModelingError> {
        baffled_piston_mechanical_radiation_impedance(
            self.geometry.diaphragm_radius_m,
            frequency_hz,
            self.acoustic_medium,
        )
    }

    #[deprecated(note = "panics on invalid study; use try_small_signal_response_at")]
    pub fn small_signal_response_at(
        &self,
        frequency_hz: f64,
        voltage: Complex64,
    ) -> SpeakerResponse {
        self.try_small_signal_response_at(frequency_hz, voltage)
            .expect("small-signal response requires a valid study, drive, and positive frequency")
    }

    pub fn try_small_signal_response_at(
        &self,
        frequency_hz: f64,
        voltage: Complex64,
    ) -> Result<SpeakerResponse, ModelingError> {
        self.validate()?;
        require_positive("frequency_hz", frequency_hz)?;
        require_finite("voltage.re", voltage.re)?;
        require_finite("voltage.im", voltage.im)?;
        let omega = std::f64::consts::TAU * frequency_hz;
        let coupling = self.try_fem_voice_coil()?;
        let mechanical_impedance = self.total_mechanical_impedance(frequency_hz)?;
        let impedance = coupling.total_impedance(omega, mechanical_impedance);
        let (current, velocity, displacement) =
            coupling.solve_frequency(voltage, omega, mechanical_impedance);
        Ok(SpeakerResponse {
            frequency_hz,
            voltage,
            current,
            impedance,
            bl_force: Complex64::new(coupling.bl, 0.0) * current,
            coil_velocity: velocity,
            diaphragm_displacement: displacement,
        })
    }

    #[deprecated(note = "panics on invalid study; use try_small_signal_response_sweep")]
    pub fn small_signal_response_sweep(&self, voltage_rms: f64) -> Vec<SpeakerResponse> {
        self.try_small_signal_response_sweep(voltage_rms)
            .expect("response sweep requires a valid study and non-negative RMS voltage")
    }

    pub fn try_small_signal_response_sweep(
        &self,
        voltage_rms: f64,
    ) -> Result<Vec<SpeakerResponse>, ModelingError> {
        self.validate()?;
        require_nonnegative("voltage_rms", voltage_rms)?;
        let voltage = Complex64::new(voltage_rms, 0.0);
        self.frequency_hz
            .iter()
            .map(|&frequency_hz| self.try_small_signal_response_at(frequency_hz, voltage))
            .collect()
    }

    /// Convenience sweep at the configured drive voltage.
    ///
    /// Validity precondition: the study must pass [`Self::validate`]; the
    /// fallible [`Self::try_small_signal_response_sweep`] carries the same
    /// computation for callers that need a `Result`.
    #[track_caller]
    pub fn configured_small_signal_response_sweep(&self) -> Vec<SpeakerResponse> {
        self.try_small_signal_response_sweep(self.drive_voltage_rms_v)
            .expect("configured sweep requires a valid study and drive voltage")
    }

    /// Convenience transducer response at one frequency.
    ///
    /// Validity precondition: the study must pass [`Self::validate`]; the
    /// fallible [`Self::try_small_signal_response_at`] carries the same
    /// computation for callers that need a `Result`.
    #[track_caller]
    pub fn transducer_response_at(
        &self,
        frequency_hz: f64,
        voltage: Complex64,
    ) -> TransducerResponse {
        self.validate()
            .expect("transducer response requires a valid speaker study");
        let response = self
            .try_small_signal_response_at(frequency_hz, voltage)
            .expect("transducer response requires positive frequency and finite drive");
        TransducerResponse {
            current: response.current,
            impedance: response.impedance,
            velocity: response.coil_velocity,
            displacement: response.diaphragm_displacement,
            radiated_power: estimate_piston_radiated_power_in_medium(
                self.geometry.diaphragm_radius_m,
                frequency_hz,
                response.coil_velocity.norm(),
                self.acoustic_medium,
            )
            .expect("speaker study must contain a valid piston and acoustic medium"),
            omega: std::f64::consts::TAU * frequency_hz,
        }
    }

    /// Convenience end-to-end example report.
    ///
    /// Validity precondition: the study must pass [`Self::validate`] with
    /// non-negative `voltage_rms`; every fallible step below runs through
    /// its `try_*` variant after that check, so the remaining `expect`s
    /// only fire on internal inconsistency.
    #[track_caller]
    pub fn detailed_example_report(&self, voltage_rms: f64) -> DetailedExampleReport {
        self.validate()
            .expect("detailed report requires a valid speaker study");
        require_nonnegative("voltage_rms", voltage_rms)
            .expect("detailed report requires non-negative RMS voltage");
        let workflow = self
            .try_fem_magnetic_workflow()
            .expect("detailed report requires at least one study step");
        let bias = self.representative_magnetic_bias_state();
        let lorentz = self.representative_distributed_lorentz();
        let velocity_shape = [1.0];
        let acoustic_coupling = self.fem_acoustic_coupling_at(1_000.0);

        DetailedExampleReport {
            study: self.clone(),
            fem: FemExampleSummary {
                bias_solve_name: workflow.bias_solve_name,
                perturbation_solve_name: workflow.perturbation_solve_name,
                reuse_bias_tangent: workflow.reuse_bias_tangent,
                average_flux_density_t: bias.average_flux_density(),
                distributed_lorentz_bl_tm: lorentz.bl_from_velocity_shape(&velocity_shape),
                magnetic_bias_is_representative: true,
                gap_model: acoustic_loss_model_from_fem_gap_model(
                    self.fem_gap_model(BoundaryLayerSpec::default()),
                ),
                acoustic_structure_at_1khz: acoustic_coupling.into(),
            },
            response_provenance:
                ResponseProvenance::LumpedElectromechanicalWithBaffledPistonRadiation,
            responses: self
                .try_small_signal_response_sweep(voltage_rms)
                .expect("detailed report requires a valid study and drive voltage")
                .iter()
                .map(|response| ResponseRow::from_response(response, self))
                .collect(),
        }
    }

    pub fn configured_detailed_example_report(&self) -> DetailedExampleReport {
        self.detailed_example_report(self.drive_voltage_rms_v)
    }

    fn representative_gap_flux_density_t(&self) -> f64 {
        let wire_length_m = self.voice_coil.active_turns
            * std::f64::consts::TAU
            * self.geometry.voice_coil_radius_m;
        self.voice_coil.bl_tm / wire_length_m
    }
}

/// Inputs used to derive a study plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StudyRequirements {
    pub name: String,
    pub geometry: DriverGeometry,
    /// Electrical and force-factor inputs for this specific driver.
    pub voice_coil: VoiceCoilSpec,
    /// Mechanical small-signal inputs for this specific moving assembly.
    pub moving_system: MovingSystemSpec,
    pub harmonic_small_signal: bool,
    pub transient_or_pulse: bool,
    pub nonlinear_large_signal: bool,
    pub frequency_hz: Vec<f64>,
    /// RMS voltage magnitude for the configured response sweep.
    pub voltage_rms_v: f64,
    pub acoustic_medium: AcousticMedium,
}

impl StudyRequirements {
    pub fn validate(&self) -> Result<(), ModelingError> {
        if self.name.trim().is_empty() {
            return Err(ModelingError::BlankStudyName);
        }
        self.geometry.validate()?;
        self.voice_coil.validate()?;
        self.moving_system.validate()?;
        if !self.harmonic_small_signal && !self.transient_or_pulse && !self.nonlinear_large_signal {
            return Err(ModelingError::NoDynamicStudyRequested);
        }
        validate_frequency_grid(&self.frequency_hz)?;
        require_nonnegative("requirements.voltage_rms_v", self.voltage_rms_v)?;
        self.acoustic_medium.validate()
    }
}

pub fn choose_symmetry(geometry: &DriverGeometry) -> SymmetryModel {
    try_choose_symmetry(geometry).expect("symmetry selection requires valid driver geometry")
}

pub fn try_choose_symmetry(geometry: &DriverGeometry) -> Result<SymmetryModel, ModelingError> {
    geometry.validate()?;
    if geometry.has_non_axisymmetric_basket {
        Ok(SymmetryModel::Full3d)
    } else if let Some(angle_degrees) = geometry.sector_angle_degrees {
        Ok(SymmetryModel::Sector3d { angle_degrees })
    } else {
        Ok(SymmetryModel::Axisymmetric2d)
    }
}

pub fn choose_dynamic_study(
    harmonic_small_signal: bool,
    transient_or_pulse: bool,
    nonlinear_large_signal: bool,
) -> DynamicStudyKind {
    try_choose_dynamic_study(
        harmonic_small_signal,
        transient_or_pulse,
        nonlinear_large_signal,
    )
    .expect("at least one dynamic study requirement must be enabled")
}

pub fn try_choose_dynamic_study(
    harmonic_small_signal: bool,
    transient_or_pulse: bool,
    nonlinear_large_signal: bool,
) -> Result<DynamicStudyKind, ModelingError> {
    if !harmonic_small_signal && !transient_or_pulse && !nonlinear_large_signal {
        return Err(ModelingError::NoDynamicStudyRequested);
    }
    if harmonic_small_signal && !transient_or_pulse && !nonlinear_large_signal {
        Ok(DynamicStudyKind::FrequencyDomainPerturbation)
    } else {
        Ok(DynamicStudyKind::TimeDependent)
    }
}

pub fn choose_acoustic_loss_model(air_gap_m: f64) -> AcousticLossModel {
    try_choose_acoustic_loss_model(air_gap_m)
        .expect("acoustic loss model requires a finite, positive gap")
}

pub fn try_choose_acoustic_loss_model(air_gap_m: f64) -> Result<AcousticLossModel, ModelingError> {
    require_positive("air_gap_m", air_gap_m)?;
    Ok(choose_acoustic_loss_model_for_layers(
        air_gap_m,
        BoundaryLayerSpec::default(),
    ))
}

pub fn choose_acoustic_loss_model_at_frequency(
    air_gap_m: f64,
    frequency_hz: f64,
    medium: AcousticMedium,
) -> Result<AcousticLossModel, ModelingError> {
    require_positive("air_gap_m", air_gap_m)?;
    let boundary_layers = BoundaryLayerSpec::at_frequency(frequency_hz, medium)?;
    Ok(choose_acoustic_loss_model_for_layers(
        air_gap_m,
        boundary_layers,
    ))
}

fn choose_acoustic_loss_model_for_layers(
    air_gap_m: f64,
    boundary_layers: BoundaryLayerSpec,
) -> AcousticLossModel {
    let layer = boundary_layers
        .viscous_delta_m
        .max(boundary_layers.thermal_delta_m);
    if air_gap_m <= 6.0 * layer {
        AcousticLossModel::FullThermoviscousAcoustics
    } else if air_gap_m <= 30.0 * layer {
        AcousticLossModel::ThermoviscousBoundaryLayerImpedance
    } else if air_gap_m <= 75.0 * layer {
        AcousticLossModel::NarrowRegionAcoustics
    } else {
        AcousticLossModel::PressureAcoustics
    }
}

pub fn acoustic_loss_model_from_fem_gap_model(model: SpeakerGapModel) -> AcousticLossModel {
    match model {
        SpeakerGapModel::FullThermoviscousRegion => AcousticLossModel::FullThermoviscousAcoustics,
        SpeakerGapModel::BoundaryLayerImpedance => {
            AcousticLossModel::ThermoviscousBoundaryLayerImpedance
        }
        SpeakerGapModel::LumpedNarrowGapImpedance => AcousticLossModel::NarrowRegionAcoustics,
    }
}

pub fn estimate_piston_radiated_power(
    radius_m: f64,
    frequency_hz: f64,
    velocity_abs_m_per_s: f64,
) -> f64 {
    estimate_piston_radiated_power_in_medium(
        radius_m,
        frequency_hz,
        velocity_abs_m_per_s,
        AcousticMedium::standard_air(),
    )
    .expect("piston power inputs must be finite and physical")
}

/// Average radiated power for an RMS piston-velocity phasor.
pub fn estimate_piston_radiated_power_in_medium(
    radius_m: f64,
    frequency_hz: f64,
    velocity_rms_m_per_s: f64,
    medium: AcousticMedium,
) -> Result<f64, ModelingError> {
    require_nonnegative("velocity_rms_m_per_s", velocity_rms_m_per_s)?;
    let radiation = baffled_piston_mechanical_radiation_impedance(radius_m, frequency_hz, medium)?;
    Ok(radiation.re * velocity_rms_m_per_s * velocity_rms_m_per_s)
}

/// Exact real part and numerically integrated reactive part of the mechanical
/// radiation impedance of a uniformly moving circular piston in an infinite baffle.
pub fn baffled_piston_mechanical_radiation_impedance(
    radius_m: f64,
    frequency_hz: f64,
    medium: AcousticMedium,
) -> Result<Complex64, ModelingError> {
    require_positive("radius_m", radius_m)?;
    require_positive("frequency_hz", frequency_hz)?;
    medium.validate()?;

    let area = std::f64::consts::PI * radius_m * radius_m;
    let ka = std::f64::consts::TAU * frequency_hz * radius_m / medium.sound_speed_m_per_s;
    require_positive("piston_area_m2", area)?;
    require_positive("piston_ka", ka)?;
    let (resistance, reactance) = normalized_baffled_piston_impedance(ka)?;
    require_finite("normalized_radiation_resistance", resistance)?;
    require_finite("normalized_radiation_reactance", reactance)?;
    let scale = medium.density_kg_per_m3 * medium.sound_speed_m_per_s * area;
    require_positive("radiation_impedance_scale", scale)?;
    Ok(Complex64::new(resistance, reactance) * scale)
}

fn normalized_baffled_piston_impedance(ka: f64) -> Result<(f64, f64), ModelingError> {
    if ka < 1.0e-3 {
        let ka2 = ka * ka;
        let resistance = ka2 / 2.0 - ka2 * ka2 / 12.0 + ka2 * ka2 * ka2 / 144.0;
        let reactance = (8.0 * ka / 3.0 - 32.0 * ka.powi(3) / 45.0 + 128.0 * ka.powi(5) / 1575.0)
            / std::f64::consts::PI;
        return Ok((resistance, reactance));
    }

    let argument = 2.0 * ka;
    let resistance = 1.0 - argument.bessel_jv(1.0) / ka;
    let reactance = struve_h1(argument)? / ka;
    // The exact R1(ka) is non-negative but oscillates about 1 with overshoot
    // (about 1.13 near ka = 2.7), so only sub-zero truncation noise may be
    // clamped. Capping the correction magnitude keeps the raw value visible
    // to passivity checks instead of hiding a broken quadrature.
    debug_assert!(
        resistance > -1e-9,
        "baffled-piston resistance {resistance} far below zero at ka={ka}"
    );
    Ok((resistance.max(0.0), reactance))
}

/// Whether `angle_degrees` is a proper divisor of 360 (a valid rotational
/// sector angle, strictly below 360). Shared by driver-geometry and sector-3D
/// validation so the two cannot drift apart.
pub fn is_proper_divisor_of_360(angle_degrees: f64) -> bool {
    if !angle_degrees.is_finite() || angle_degrees <= 0.0 || angle_degrees >= 360.0 {
        return false;
    }
    let sectors = 360.0 / angle_degrees;
    (sectors - sectors.round()).abs() <= 1.0e-10
}

/// Integral representation H1(z) = 2z/pi integral_0^(pi/2)
/// cos(theta)^2 sin(z sin(theta)) dtheta.
///
/// Valid for moderate `|argument|` (piston `ka` up to ~50, i.e. argument up
/// to ~100). The Gauss-Legendre order scales with `|argument|` and caps at
/// 512 points, so far-ultrasonic `ka` loses accuracy silently; callers
/// needing high-`ka` radiation should use an asymptotic expansion instead.
fn struve_h1(argument: f64) -> Result<f64, ModelingError> {
    let requested_order = ((argument.abs() / std::f64::consts::PI) * 16.0).ceil() as usize;
    let mut order = requested_order.clamp(32, 512);
    if !order.is_multiple_of(2) {
        order += 1;
    }

    let quadrature = struve_h1_quadrature(order)?;
    let integral = quadrature
        .iter()
        .map(|&(theta, weighted_cosine_squared)| {
            weighted_cosine_squared * (argument * theta.sin()).sin()
        })
        .sum::<f64>();
    Ok(2.0 * argument * integral / std::f64::consts::PI)
}

type StruveQuadrature = Arc<Vec<(f64, f64)>>;

fn struve_h1_quadrature(order: usize) -> Result<StruveQuadrature, ModelingError> {
    static CACHE: OnceLock<Mutex<HashMap<usize, StruveQuadrature>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(quadrature) = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(&order)
    {
        return Ok(Arc::clone(quadrature));
    }

    let mut quadrature = Vec::with_capacity(order);
    for index in 0..order / 2 {
        let mut node = (std::f64::consts::PI * (index as f64 + 0.75) / (order as f64 + 0.5)).cos();
        let mut derivative = 0.0;
        let mut converged = false;
        for _ in 0..16 {
            let (polynomial, previous) = legendre_pair(order, node);
            derivative = order as f64 * (node * polynomial - previous) / (node * node - 1.0);
            if !derivative.is_finite() {
                break;
            }
            let update = polynomial / derivative;
            node -= update;
            if node.is_finite() && update.abs() <= 4.0 * f64::EPSILON {
                converged = true;
                break;
            }
        }
        // Without this guard a stagnant Newton step (near-unit node, flat
        // derivative) would cache a NaN weight and poison every later
        // Struve evaluation at this order.
        if !converged || !node.is_finite() || !derivative.is_finite() {
            return Err(ModelingError::QuadratureDidNotConverge { order });
        }
        let weight = 2.0 / ((1.0 - node * node) * derivative * derivative);
        for signed_node in [node, -node] {
            let theta = std::f64::consts::FRAC_PI_4 * (signed_node + 1.0);
            let mapped_weight = std::f64::consts::FRAC_PI_4 * weight;
            quadrature.push((theta, mapped_weight * theta.cos().powi(2)));
        }
    }
    let quadrature = Arc::new(quadrature);
    let mut cache = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    Ok(Arc::clone(
        cache
            .entry(order)
            .or_insert_with(|| Arc::clone(&quadrature)),
    ))
}

fn legendre_pair(order: usize, x: f64) -> (f64, f64) {
    let mut previous = 1.0;
    let mut current = x;
    if order == 0 {
        return (previous, 0.0);
    }
    for degree in 2..=order {
        let next = ((2 * degree - 1) as f64 * x * current - (degree - 1) as f64 * previous)
            / degree as f64;
        previous = current;
        current = next;
    }
    (current, previous)
}

pub fn build_steps(dynamic_study: DynamicStudyKind) -> Vec<StudyStep> {
    let mut steps = vec![StudyStep {
        name: "stationary-magnetic-bias".to_string(),
        purpose: "Resolve the permanent-magnet bias field before dynamic perturbation".to_string(),
        solver: SolverFamily::DirectSparse,
    }];

    match dynamic_study {
        DynamicStudyKind::FrequencyDomainPerturbation => steps.push(StudyStep {
            name: "frequency-domain-electrovibroacoustic-perturbation".to_string(),
            purpose: "Solve harmonic small-signal current, displacement, and pressure".to_string(),
            solver: SolverFamily::FgmresShiftedLaplacianIlu,
        }),
        DynamicStudyKind::TimeDependent => steps.push(StudyStep {
            name: "time-dependent-electromechanical-response".to_string(),
            purpose: "Integrate pulses, nonlinear BL/Le, or higher-harmonic generation".to_string(),
            solver: SolverFamily::NewmarkPredictorCorrector,
        }),
    }

    steps
}

pub fn default_motor_materials() -> Vec<MotorMaterialRole> {
    vec![
        MotorMaterialRole::AirGap,
        MotorMaterialRole::SteelTopPlate,
        MotorMaterialRole::SteelPolePiece,
        MotorMaterialRole::PermanentMagnet {
            br_t: 1.25,
            direction: [0.0, 1.0, 0.0],
        },
        MotorMaterialRole::CopperCoil,
    ]
}

pub fn octave_sweep(start_hz: f64, stop_hz: f64, points_per_octave: usize) -> Vec<f64> {
    try_octave_sweep(start_hz, stop_hz, points_per_octave).unwrap_or_else(|error| panic!("{error}"))
}

pub fn try_octave_sweep(
    start_hz: f64,
    stop_hz: f64,
    points_per_octave: usize,
) -> Result<Vec<f64>, ModelingError> {
    const MAX_SWEEP_POINTS: f64 = 1_000_000.0;

    require_positive("start_hz", start_hz)?;
    require_positive("stop_hz", stop_hz)?;
    if stop_hz < start_hz {
        return Err(ModelingError::OutOfRange {
            field: "stop_hz",
            constraint: "greater than or equal to start_hz",
        });
    }
    if points_per_octave == 0 {
        return Err(ModelingError::OutOfRange {
            field: "points_per_octave",
            constraint: "greater than zero",
        });
    }

    if start_hz == stop_hz {
        return Ok(vec![start_hz]);
    }

    let interval_count = (stop_hz / start_hz).log2() * points_per_octave as f64;
    if !interval_count.is_finite() || interval_count > MAX_SWEEP_POINTS {
        return Err(ModelingError::OutOfRange {
            field: "octave sweep",
            constraint: "no more than 1,000,000 intervals",
        });
    }
    let full_intervals = interval_count.floor() as usize;
    let mut out = Vec::with_capacity(full_intervals.saturating_add(2));
    for index in 0..=full_intervals {
        let frequency = start_hz * 2.0_f64.powf(index as f64 / points_per_octave as f64);
        if frequency <= stop_hz {
            out.push(frequency);
        }
    }

    let last = out.last_mut().ok_or(ModelingError::InvalidFrequencyGrid)?;
    let endpoint_tolerance = 32.0 * f64::EPSILON * stop_hz;
    if (stop_hz - *last).abs() <= endpoint_tolerance {
        *last = stop_hz;
    } else {
        out.push(stop_hz);
    }
    validate_frequency_grid(&out)?;
    Ok(out)
}

#[cfg(test)]
// Fixture conveniences intentionally use the deprecated infallible
// wrappers on valid inputs; production paths must use `try_*`.
#[allow(deprecated)]
mod tests {
    use super::*;

    #[test]
    fn detailed_driver_uses_axisymmetric_frequency_domain_sequence() {
        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();

        assert_eq!(study.symmetry, SymmetryModel::Axisymmetric2d);
        assert_eq!(
            study.dynamic_study,
            DynamicStudyKind::FrequencyDomainPerturbation
        );
        assert_eq!(study.steps.len(), 2);
        assert_eq!(study.steps[0].name, "stationary-magnetic-bias");
        assert_eq!(
            study.steps[1].name,
            "frequency-domain-electrovibroacoustic-perturbation"
        );
    }

    #[test]
    fn non_axisymmetric_basket_forces_full_3d() {
        let mut study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        study.geometry.has_non_axisymmetric_basket = true;

        assert_eq!(choose_symmetry(&study.geometry), SymmetryModel::Full3d);
    }

    #[test]
    fn sector_modes_use_the_requested_periodic_angle() {
        let mut geometry = SpeakerDriverStudy::detailed_frequency_domain_driver().geometry;
        geometry.sector_angle_degrees = Some(45.0);

        assert_eq!(
            choose_symmetry(&geometry),
            SymmetryModel::Sector3d {
                angle_degrees: 45.0
            }
        );
    }

    #[test]
    fn pulses_and_nonlinearity_choose_time_domain() {
        assert_eq!(
            choose_dynamic_study(false, true, false),
            DynamicStudyKind::TimeDependent
        );
        assert_eq!(
            choose_dynamic_study(false, false, true),
            DynamicStudyKind::TimeDependent
        );
    }

    #[test]
    #[should_panic(expected = "at least one dynamic study requirement must be enabled")]
    fn no_dynamic_requirement_is_not_silently_treated_as_time_domain() {
        let _ = choose_dynamic_study(false, false, false);
    }

    #[test]
    fn coil_is_solid_while_thin_moving_parts_are_shells() {
        let parts = MovingPartRepresentations::default();

        assert_eq!(parts.cone, PartRepresentation::Shell);
        assert_eq!(parts.surround, PartRepresentation::Shell);
        assert_eq!(parts.spider, PartRepresentation::Shell);
        assert_eq!(parts.former, PartRepresentation::Shell);
        assert_eq!(parts.coil, PartRepresentation::Solid);
    }

    #[test]
    fn narrow_gap_recommends_thermoviscous_correction() {
        assert_eq!(
            choose_acoustic_loss_model(0.75e-3),
            AcousticLossModel::ThermoviscousBoundaryLayerImpedance
        );
        assert_eq!(
            choose_acoustic_loss_model(0.2e-3),
            AcousticLossModel::FullThermoviscousAcoustics
        );
    }

    #[test]
    fn adapts_to_existing_fem_electrodynamic_coupling() {
        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        let coupling = study.fem_voice_coil();

        assert_eq!(coupling.resistance, study.voice_coil.resistance_ohm);
        assert_eq!(coupling.inductance, study.voice_coil.inductance_h);
        assert_eq!(coupling.bl, study.voice_coil.bl_tm);
        assert_eq!(
            coupling.force_distribution(),
            vec![(0, study.voice_coil.bl_tm)]
        );
    }

    #[test]
    fn motional_impedance_is_added_to_blocked_impedance() {
        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        let blocked = study.blocked_impedance(100.0);
        let total = study.small_signal_impedance(100.0);

        assert_ne!(blocked, total);
        assert!(total.re > 0.0);
    }

    #[test]
    fn builds_fem_magnetic_workflow_from_study_steps() {
        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        let workflow = study.fem_magnetic_workflow();

        assert_eq!(workflow.bias_solve_name, "stationary_magnetic_bias");
        assert_eq!(
            workflow.perturbation_solve_name,
            "frequency_domain_electrovibroacoustic_perturbation"
        );
        assert!(workflow.reuse_bias_tangent);
    }

    #[test]
    fn malformed_public_study_does_not_index_empty_steps() {
        let mut study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        study.steps.clear();

        assert!(study.try_fem_magnetic_workflow().is_err());
    }

    #[test]
    fn distributed_lorentz_matches_nominal_bl() {
        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        let lorentz = study.representative_distributed_lorentz();
        let force = lorentz.force_vector_per_ampere(1);

        assert!((force[0] - study.voice_coil.bl_tm).abs() < 1.0e-12);
        assert!((lorentz.bl_from_velocity_shape(&[1.0]) - study.voice_coil.bl_tm).abs() < 1.0e-12);
    }

    #[test]
    fn representative_flux_uses_the_configured_active_turn_count() {
        let baseline = SpeakerDriverStudy::detailed_frequency_domain_driver();
        let baseline_flux = baseline
            .representative_magnetic_bias_state()
            .average_flux_density()[0];
        let mut doubled_turns = baseline.clone();
        doubled_turns.voice_coil.active_turns = 2.0 * baseline.voice_coil.active_turns;
        let doubled_turns_flux = doubled_turns
            .representative_magnetic_bias_state()
            .average_flux_density()[0];
        let lorentz_bl = doubled_turns
            .representative_distributed_lorentz()
            .bl_from_velocity_shape(&[1.0]);

        assert!((doubled_turns_flux / baseline_flux - 0.5).abs() < 1.0e-12);
        assert!((lorentz_bl - doubled_turns.voice_coil.bl_tm).abs() < 1.0e-12);
    }

    #[test]
    fn fem_gap_model_maps_back_to_modeling_loss_choice() {
        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        let model = study.fem_gap_model(BoundaryLayerSpec::default());

        assert_eq!(model, SpeakerGapModel::BoundaryLayerImpedance);
        assert_eq!(
            acoustic_loss_model_from_fem_gap_model(model),
            AcousticLossModel::ThermoviscousBoundaryLayerImpedance
        );
    }

    #[test]
    fn acoustic_coupling_uses_frequency_dependent_structure_loading() {
        let mut study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        study.acoustic_medium.density_kg_per_m3 = 1.5;
        study.acoustic_medium.sound_speed_m_per_s = 330.0;
        let low = study.fem_acoustic_coupling_at(100.0);
        let high = study.fem_acoustic_coupling_at(1_000.0);
        let sweep = study.fem_sweep_points();

        assert_eq!(low.fluid_to_structure, high.fluid_to_structure);
        assert!(high.structure_to_fluid.re < low.structure_to_fluid.re);
        assert!(sweep.iter().all(|point| point.fluid_density == 1.5));
        assert!(sweep.iter().all(|point| point.sound_speed == 330.0));
    }

    #[test]
    fn extracts_fem_speaker_response_rows() {
        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        let response = study.small_signal_response_at(100.0, Complex64::new(2.83, 0.0));
        let row = ResponseRow::from_response(&response, &study);

        assert_eq!(row.frequency_hz, 100.0);
        assert!(row.impedance_real_ohm > 0.0);
        assert!(row.current_abs_a > 0.0);
        assert!(row.displacement_abs_m > 0.0);
        assert_eq!(row.voltage_real_v, response.voltage.re);
        assert_eq!(row.voltage_imag_v, response.voltage.im);
        assert_eq!(row.current_real_a, response.current.re);
        assert_eq!(row.current_imag_a, response.current.im);
        assert_eq!(row.bl_force_real_n, response.bl_force.re);
        assert_eq!(row.bl_force_imag_n, response.bl_force.im);
        assert_eq!(row.velocity_real_m_per_s, response.coil_velocity.re);
        assert_eq!(row.velocity_imag_m_per_s, response.coil_velocity.im);
        assert_eq!(row.displacement_real_m, response.diaphragm_displacement.re);
        assert_eq!(row.displacement_imag_m, response.diaphragm_displacement.im);
    }

    #[test]
    fn detailed_report_includes_fem_bridge_and_response_sweep() {
        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        let report = study.detailed_example_report(2.83);

        assert_eq!(report.responses.len(), study.frequency_hz.len());
        assert_eq!(report.fem.bias_solve_name, "stationary_magnetic_bias");
        assert!((report.fem.distributed_lorentz_bl_tm - study.voice_coil.bl_tm).abs() < 1.0e-12);
        assert_eq!(
            report.fem.gap_model,
            AcousticLossModel::ThermoviscousBoundaryLayerImpedance
        );
        assert!(report.fem.magnetic_bias_is_representative);
        assert_eq!(
            report.response_provenance,
            ResponseProvenance::LumpedElectromechanicalWithBaffledPistonRadiation
        );
        assert!(
            report
                .responses
                .iter()
                .all(|row| row.radiated_power_w >= 0.0)
        );
        assert!(
            report
                .responses
                .iter()
                .all(|row| row.power_balance_error_w.abs() < 1.0e-12)
        );
    }

    #[test]
    fn transducer_response_estimates_radiated_power() {
        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        let response = study.transducer_response_at(1_000.0, Complex64::new(2.83, 0.0));

        assert!((response.omega - std::f64::consts::TAU * 1_000.0).abs() < 1.0e-12);
        assert!(response.radiated_power >= 0.0);
        assert!(response.impedance.norm() > 0.0);
    }

    #[test]
    fn octave_sweep_includes_requested_stop_frequency() {
        let frequencies = octave_sweep(20.0, 20_000.0, 6);

        assert_eq!(frequencies.first().copied(), Some(20.0));
        assert_eq!(frequencies.last().copied(), Some(20_000.0));
        assert!(frequencies.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    #[should_panic(expected = "points_per_octave must be greater than zero")]
    fn octave_sweep_rejects_zero_points_per_octave() {
        let _ = octave_sweep(20.0, 20_000.0, 0);
    }

    #[test]
    fn octave_sweep_has_a_fallible_external_input_path() {
        assert!(try_octave_sweep(0.0, 20_000.0, 6).is_err());
        assert!(try_octave_sweep(20.0, 10.0, 6).is_err());
        assert!(try_octave_sweep(20.0, 20_000.0, 0).is_err());
    }

    #[test]
    fn physical_specs_reject_nonphysical_or_nonfinite_values() {
        let mut study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        study.voice_coil.resistance_ohm = 0.0;
        assert!(study.validate().is_err());

        let mut study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        study.moving_system.moving_mass_kg = f64::NAN;
        assert!(study.validate().is_err());

        let mut study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        study.geometry.diaphragm_radius_m = study.geometry.voice_coil_radius_m / 2.0;
        assert!(study.validate().is_err());

        let mut study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        study.frequency_hz = vec![100.0, 100.0, 200.0];
        assert!(study.validate().is_err());

        let mut study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        study.steps[0].name.clear();
        assert!(study.validate().is_err());

        let mut study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        study.symmetry = SymmetryModel::Full3d;
        assert!(study.validate().is_err());

        let mut study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        study.dynamic_study = DynamicStudyKind::TimeDependent;
        assert!(study.validate().is_err());

        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        assert!(study.moving_system.try_mechanical_impedance(0.0).is_err());
        assert!(
            study
                .moving_system
                .try_mechanical_impedance(f64::MAX)
                .is_err()
        );
        let extreme_system = MovingSystemSpec {
            moving_mass_kg: f64::MIN_POSITIVE,
            mechanical_resistance_kg_per_s: 1.0,
            stiffness_n_per_m: f64::MAX,
            xmax_m: None,
        };
        assert!(extreme_system.try_resonance_hz().is_err());
        assert!(
            study
                .try_small_signal_response_at(f64::NAN, Complex64::new(1.0, 0.0))
                .is_err()
        );
        assert!(study.try_small_signal_response_sweep(-1.0).is_err());
        assert!(
            baffled_piston_mechanical_radiation_impedance(
                f64::MAX,
                f64::MAX,
                AcousticMedium::standard_air(),
            )
            .is_err()
        );
        assert!(BoundaryLayerSpec::at_frequency(f64::MAX, AcousticMedium::standard_air()).is_err());
    }

    #[test]
    fn requirements_preserve_drive_motor_and_moving_system_inputs() {
        let baseline = SpeakerDriverStudy::detailed_frequency_domain_driver();
        let geometry = baseline.geometry;
        let voice_coil = VoiceCoilSpec {
            resistance_ohm: 4.2,
            inductance_h: 0.31e-3,
            bl_tm: 7.4,
            active_turns: 82.0,
        };
        let moving_system = MovingSystemSpec {
            moving_mass_kg: 18.0e-3,
            mechanical_resistance_kg_per_s: 1.4,
            stiffness_n_per_m: 1_100.0,
            xmax_m: None,
        };
        let study = SpeakerDriverStudy::try_from_requirements(StudyRequirements {
            name: "custom-drive".to_string(),
            geometry,
            voice_coil,
            moving_system,
            harmonic_small_signal: true,
            transient_or_pulse: false,
            nonlinear_large_signal: false,
            frequency_hz: vec![100.0, 200.0],
            voltage_rms_v: 1.25,
            acoustic_medium: AcousticMedium::standard_air(),
        })
        .expect("valid requirements");

        assert_eq!(study.drive_voltage_rms_v, 1.25);
        assert_eq!(study.voice_coil, voice_coil);
        assert_eq!(study.moving_system, moving_system);
        assert!(study.validate().is_ok());
        assert!(
            study
                .configured_small_signal_response_sweep()
                .iter()
                .all(|response| response.voltage == Complex64::new(1.25, 0.0))
        );
    }

    #[test]
    fn invalid_requirements_are_rejected_fallibly() {
        let geometry = SpeakerDriverStudy::detailed_frequency_domain_driver().geometry;
        let result = SpeakerDriverStudy::try_from_requirements(StudyRequirements {
            name: "invalid".to_string(),
            geometry,
            voice_coil: SpeakerDriverStudy::detailed_frequency_domain_driver().voice_coil,
            moving_system: SpeakerDriverStudy::detailed_frequency_domain_driver().moving_system,
            harmonic_small_signal: false,
            transient_or_pulse: false,
            nonlinear_large_signal: false,
            frequency_hz: vec![100.0, f64::NAN],
            voltage_rms_v: -1.0,
            acoustic_medium: AcousticMedium::standard_air(),
        });

        assert!(result.is_err());
    }

    #[test]
    fn permanent_magnet_direction_is_normalized_for_fem() {
        let material = MotorMaterialRole::PermanentMagnet {
            br_t: 1.25,
            direction: [0.0, 2.0, 0.0],
        }
        .as_fem_material();

        assert_eq!(material.remanent_b(), [0.0, 1.25, 0.0]);
    }

    #[test]
    fn zero_drive_keeps_the_system_impedance_finite() {
        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        let response = study.small_signal_response_at(100.0, Complex64::new(0.0, 0.0));

        assert_eq!(response.current, Complex64::new(0.0, 0.0));
        assert!(response.impedance.re.is_finite());
        assert!(response.impedance.im.is_finite());
        assert_eq!(response.impedance, study.small_signal_impedance(100.0));
    }

    #[test]
    fn acoustic_medium_xem_boundary_preserves_checked_shared_fields() {
        let medium = AcousticMedium::standard_air();
        let xem = medium
            .to_xem(1_000.0)
            .expect("standard air should convert to XEM");
        assert_eq!(xem.frequency.get(), 1_000.0);
        assert_eq!(xem.speed_of_sound.get(), medium.sound_speed_m_per_s);
        assert_eq!(xem.density.get(), medium.density_kg_per_m3);

        assert!(medium.to_xem(0.0).is_err());
        assert!(medium.to_xem(f64::NAN).is_err());
    }

    #[test]
    fn boundary_layers_follow_inverse_square_root_frequency_scaling() {
        let medium = AcousticMedium::standard_air();
        let at_100_hz = BoundaryLayerSpec::at_frequency(100.0, medium)
            .expect("positive frequency and physical medium");
        let at_1khz = BoundaryLayerSpec::at_frequency(1_000.0, medium)
            .expect("positive frequency and physical medium");
        let expected_ratio = 10.0_f64.sqrt();

        assert!(
            (at_100_hz.viscous_delta_m / at_1khz.viscous_delta_m - expected_ratio).abs() < 1.0e-12
        );
        assert!(
            (at_100_hz.thermal_delta_m / at_1khz.thermal_delta_m - expected_ratio).abs() < 1.0e-12
        );
    }

    #[test]
    fn broadband_gap_choice_is_conservative_at_the_lowest_frequency() {
        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();

        assert_eq!(
            study.acoustic_loss_model,
            AcousticLossModel::FullThermoviscousAcoustics
        );
        assert_eq!(
            choose_acoustic_loss_model_at_frequency(
                study.geometry.air_gap_m,
                20.0,
                study.acoustic_medium,
            )
            .expect("valid gap, medium, and frequency"),
            AcousticLossModel::FullThermoviscousAcoustics
        );
        assert!(try_choose_acoustic_loss_model(-1.0).is_err());
    }

    #[test]
    fn baffled_piston_radiation_matches_the_low_frequency_limit() {
        use sonium_fem::physics::acoustics::electroacoustics::lumped_extraction::radiation_impedance_piston_lf;

        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        let frequency_hz = 20.0;
        let exact = study
            .mechanical_radiation_impedance(frequency_hz)
            .expect("valid piston and medium");
        let area = std::f64::consts::PI * study.geometry.diaphragm_radius_m.powi(2);
        let medium = study.acoustic_medium;
        let low_frequency = radiation_impedance_piston_lf(
            study.geometry.diaphragm_radius_m,
            std::f64::consts::TAU * frequency_hz,
            medium.density_kg_per_m3,
            medium.sound_speed_m_per_s,
        ) * area
            * area;

        assert!((exact.re / low_frequency.re - 1.0).abs() < 1.0e-4);
        assert!((exact.im / low_frequency.im - 1.0).abs() < 1.0e-4);
    }

    #[test]
    fn baffled_piston_impedance_matches_ka_one_reference() {
        let (resistance, reactance) =
            normalized_baffled_piston_impedance(1.0).expect("ka=1 converges");

        assert!((resistance - 0.423_275_192_243_126_6).abs() < 1.0e-13);
        assert!(
            (reactance - 0.646_763_728_283_562_2).abs() < 1.0e-10,
            "reactance={reactance:.16}"
        );
    }

    #[test]
    fn struve_quadrature_table_is_reused() {
        let first = struve_h1_quadrature(64).expect("order 64 converges");
        let second = struve_h1_quadrature(64).expect("order 64 converges");
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first.len(), 64);
    }

    #[test]
    fn baffled_piston_load_remains_passive_across_audio_band() {
        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();

        for &frequency_hz in &study.frequency_hz {
            let impedance = study
                .mechanical_radiation_impedance(frequency_hz)
                .expect("valid piston load");
            assert!(impedance.re >= 0.0, "frequency={frequency_hz}");
            assert!(impedance.im > 0.0, "frequency={frequency_hz}");
        }
    }

    #[test]
    fn small_signal_response_includes_passive_radiation_loading() {
        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        let frequency_hz = 100.0;
        let omega = std::f64::consts::TAU * frequency_hz;
        let z_mechanical = study.moving_system.mechanical_impedance(frequency_hz)
            + study
                .mechanical_radiation_impedance(frequency_hz)
                .expect("valid radiation load");
        let expected = study.fem_voice_coil().total_impedance(omega, z_mechanical);

        assert!((study.small_signal_impedance(frequency_hz) - expected).norm() < 1.0e-12);
    }

    #[test]
    fn rms_power_balance_closes_for_loaded_small_signal_response() {
        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        let voltage_rms = Complex64::new(2.83, 0.0);
        let response = study.small_signal_response_at(100.0, voltage_rms);
        let radiation = study
            .mechanical_radiation_impedance(100.0)
            .expect("valid radiation load");
        let input_power = (voltage_rms * response.current.conj()).re;
        let coil_loss = study.voice_coil.resistance_ohm * response.current.norm_sqr();
        let mechanical_loss =
            study.moving_system.mechanical_resistance_kg_per_s * response.coil_velocity.norm_sqr();
        let radiated_power = radiation.re * response.coil_velocity.norm_sqr();

        assert!(radiated_power > 0.0);
        assert!((input_power - coil_loss - mechanical_loss - radiated_power).abs() < 1.0e-12);
    }

    #[test]
    fn sealed_enclosure_adds_compliance_and_preserves_power_balance() {
        let mut study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        study.enclosure = EnclosureSpec::Sealed {
            volume_m3: 20.0e-3,
            leakage_resistance_pa_s_per_m3: 2.0e6,
            absorption_resistance_pa_s_per_m3: 1.0e6,
        };
        assert!(study.validate().is_ok());

        let load = study
            .enclosure_mechanical_impedance(50.0)
            .expect("valid sealed enclosure");
        assert!(load.re > 0.0);
        assert!(load.im < 0.0);
        let response = study.small_signal_response_at(50.0, Complex64::new(2.83, 0.0));
        let row = ResponseRow::from_response(&response, &study);
        assert!(row.enclosure_loss_power_w > 0.0);
        assert!(row.power_balance_error_w.abs() < 1.0e-12);
    }

    #[test]
    fn vented_enclosure_tuning_and_load_are_finite_and_passive() {
        let enclosure = EnclosureSpec::Vented {
            volume_m3: 35.0e-3,
            port: PortSpec {
                radius_m: 35.0e-3,
                length_m: 0.15,
                end_correction_m: 0.055,
                resistance_pa_s_per_m3: 80.0,
            },
            leakage_resistance_pa_s_per_m3: 3.0e6,
            absorption_resistance_pa_s_per_m3: 2.0e6,
        };
        let medium = AcousticMedium::standard_air();
        let tuning = enclosure
            .tuning_frequency_hz(medium)
            .expect("valid vented enclosure")
            .expect("vented enclosure has a tuning frequency");
        assert!((20.0..100.0).contains(&tuning));
        for frequency in [20.0, tuning, 100.0, 1_000.0] {
            let impedance = enclosure
                .acoustic_impedance(frequency, medium)
                .expect("valid passive enclosure load");
            assert!(impedance.re >= 0.0, "frequency={frequency}");
            assert!(impedance.re.is_finite() && impedance.im.is_finite());
        }
    }

    #[test]
    fn piston_directivity_is_normalized_and_far_field_scales_as_inverse_distance() {
        let medium = AcousticMedium::standard_air();
        let radius = 52.5e-3;
        let frequency = 5_000.0;
        let on_axis = baffled_piston_directivity(radius, frequency, 0.0, medium)
            .expect("valid on-axis directivity");
        let off_axis = baffled_piston_directivity(radius, frequency, 60.0, medium)
            .expect("valid off-axis directivity");
        assert!((on_axis - 1.0).abs() < 1.0e-12);
        assert!(off_axis.abs() < on_axis);

        let velocity = Complex64::new(0.1, -0.02);
        let pressure_1m =
            baffled_piston_far_field_pressure(radius, frequency, velocity, 1.0, 0.0, medium)
                .expect("valid far-field pressure");
        let pressure_2m =
            baffled_piston_far_field_pressure(radius, frequency, velocity, 2.0, 0.0, medium)
                .expect("valid far-field pressure");
        assert!((pressure_1m.norm() / pressure_2m.norm() - 2.0).abs() < 1.0e-12);
        assert!(sound_pressure_level_db(pressure_1m.norm()).unwrap() > 0.0);
    }

    #[test]
    fn nonlinear_large_signal_path_executes_and_reports_distortion() {
        let study = SpeakerDriverStudy::transient_large_signal_driver();
        let result = study
            .try_large_signal_response(LargeSignalSpec {
                duration_s: 0.1,
                voltage_rms_v: 4.0,
                ..LargeSignalSpec::default()
            })
            .expect("valid nonlinear transient");

        assert_eq!(result.time_s.len(), result.displacement_m.len());
        assert!(result.peak_displacement_m > 0.0);
        assert!(result.peak_current_a > 0.0);
        assert!(result.displacement_fundamental_peak_m > 0.0);
        assert!(result.displacement_thd > 0.0);
        assert_eq!(
            result.provenance,
            LargeSignalProvenance::LumpedNonlinearElectromechanicalWithoutAcousticLoad
        );
    }

    #[test]
    fn large_signal_sampling_and_duration_are_bounded() {
        let study = SpeakerDriverStudy::transient_large_signal_driver();
        let too_coarse = LargeSignalSpec {
            samples_per_period: 8,
            ..LargeSignalSpec::default()
        };
        let too_short = LargeSignalSpec {
            duration_s: 1.0e-6,
            ..LargeSignalSpec::default()
        };
        assert!(study.try_large_signal_response(too_coarse).is_err());
        assert!(study.try_large_signal_response(too_short).is_err());

        let mut enclosed = study;
        enclosed.enclosure = EnclosureSpec::Sealed {
            volume_m3: 20.0e-3,
            leakage_resistance_pa_s_per_m3: 0.0,
            absorption_resistance_pa_s_per_m3: 0.0,
        };
        let enclosed_result = enclosed
            .try_large_signal_response(LargeSignalSpec::default())
            .expect("M2 supports sealed large-signal enclosures");
        assert_eq!(
            enclosed_result.provenance,
            LargeSignalProvenance::LumpedNonlinearElectromechanicalWithCoupledAcousticCircuit
        );
        assert!(
            enclosed_result
                .box_pressure_pa
                .iter()
                .any(|pressure| pressure.abs() > 0.0)
        );
    }

    #[test]
    fn steady_state_thermal_compression_raises_resistance_and_reduces_velocity() {
        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        let hot = study
            .try_thermal_compressed_response_at(1_000.0, 20.0, ThermalCompressionSpec::default())
            .expect("thermal iteration converges");

        assert!(hot.coil_temperature_k > ThermalCompressionSpec::default().ambient_temperature_k);
        assert!(hot.hot_resistance_ohm > study.voice_coil.resistance_ohm);
        assert!(hot.coil_loss_w > 0.0);
        assert!(hot.compression_db > 0.0);
        assert!(
            hot.diagnostics.thermal_iteration_count > 0,
            "thermal solve must report iteration provenance"
        );
        assert!(hot.diagnostics.radiation_power_w > 0.0);
        assert!(hot.diagnostics.power_balance_residual_w.abs() < 1.0e-10);
        assert!((hot.diagnostics.thermal_time_constant_s - 36.0).abs() < 1.0e-12);
    }

    #[test]
    fn thermal_response_accepts_resolved_passive_load_and_rejects_active_load() {
        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        let frequency_hz = 1_000.0;
        let analytical = study
            .mechanical_radiation_impedance(frequency_hz)
            .expect("analytical load");
        let loaded = study
            .try_thermal_compressed_response_with_radiation_load_at(
                frequency_hz,
                20.0,
                ThermalCompressionSpec::default(),
                analytical * 1.5,
            )
            .expect("passive external load closes");

        assert_eq!(
            loaded.diagnostics.radiation_impedance_n_s_per_m,
            analytical * 1.5
        );
        assert!(loaded.diagnostics.radiation_power_w > 0.0);
        assert!(loaded.diagnostics.input_power_w > loaded.diagnostics.radiation_power_w);
        assert!(
            study
                .try_thermal_compressed_response_with_radiation_load_at(
                    frequency_hz,
                    20.0,
                    ThermalCompressionSpec::default(),
                    Complex64::new(-1.0, analytical.im),
                )
                .is_err()
        );
    }

    #[test]
    fn legacy_study_json_defaults_to_free_air_enclosure() {
        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        let mut value = serde_json::to_value(&study).expect("serialize study");
        value
            .as_object_mut()
            .expect("study is an object")
            .remove("enclosure");
        let decoded: SpeakerDriverStudy =
            serde_json::from_value(value).expect("legacy study remains readable");

        assert_eq!(decoded.enclosure, EnclosureSpec::FreeAir);
        assert!(decoded.validate().is_ok());
    }
}
impl SpeakerDriverStudy {
    /// Return the validated study frequency grid as XEM quantities.
    ///
    /// The serialized study keeps its compatibility `Vec<f64>` field, while
    /// this boundary prevents downstream XEM consumers from accepting a
    /// non-positive, non-finite, duplicate, or descending frequency grid.
    pub fn frequency_quantities(&self) -> Result<Vec<sonium_xem::FrequencyHz>, ModelingError> {
        validate_frequency_grid(&self.frequency_hz)?;
        self.frequency_hz
            .iter()
            .copied()
            .map(|frequency| {
                sonium_xem::FrequencyHz::try_new(frequency)
                    .map_err(|_| ModelingError::InvalidFrequencyGrid)
            })
            .collect()
    }
}

#[cfg(test)]
// Same fixture rationale as `tests` above.
#[allow(deprecated)]
mod typed_frequency_tests {
    use super::SpeakerDriverStudy;

    #[test]
    fn frequency_quantities_preserve_validated_grid() {
        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        let typed = study
            .frequency_quantities()
            .expect("default study frequency grid is valid");
        assert_eq!(typed.len(), study.frequency_hz.len());
        assert_eq!(
            typed.first().expect("non-empty grid").get(),
            study.frequency_hz[0]
        );
        assert_eq!(
            typed.last().expect("non-empty grid").get(),
            *study.frequency_hz.last().unwrap()
        );
    }

    #[test]
    fn frequency_quantities_reject_invalid_ordering() {
        let mut study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        study.frequency_hz.swap(0, 1);
        assert!(study.frequency_quantities().is_err());
    }
}
