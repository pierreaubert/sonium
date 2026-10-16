//! Resumable staged optimization for the bounded five-parameter waveguide preset.
//!
//! Version one deliberately stops at 5 kHz. The preference score and all
//! acceptance decisions produced here are therefore band-limited to
//! 200 Hz–5 kHz and must not be presented as full-band tweeter optimization.

use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::time::Instant;

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use sonium_solvers::LinearSolveConfig;
use sonium_solvers::optimization::Bound;

use super::{
    OBJECTIVE_FAILURE_PENALTY, OptimizationConfig, OptimizationError, OptimizationGradientMethod,
    discrete_adjoint_gradient_for_parameterization,
};
use crate::cea2034_output::{self, SpinoramaMetrics};
use crate::mesh_gen::PinnedWaveguideMesh;
use crate::model::{DriverConfig, LoudspeakerModel};
use crate::simulation::{
    self, ExecutionProfile, SimulationResult, SimulationWarmStart, SolverConfig,
};
use crate::spline::CubicSpline;
use sonium_sim_speaker_modeling::fem_bem_coupling::{
    CylinderFixtureDiscretization, RomPromotionPolicy,
};
use sonium_sim_speaker_modeling::{
    CoupledProductConstraintReport, CoupledProductLimits, CoupledProductMetrics, ProductDesign,
    SpeakerToleranceSpec, SpeakerUncertaintyCase, UncertaintySampling, generate_uncertainty_cases,
};

const PLAN_VERSION: u32 = 1;
const CACHE_IDENTITY_SCHEMA_VERSION: u32 = 1;
const BAND_LABEL: &str = "200 Hz–5 kHz band-limited";
const WAVEGUIDE_PARAMETER_COUNT: usize = 5;
const REFERENCE_PANEL_MODE_ORDER: usize = 3;

/// Deterministic, solver-stable distributed reference fixture used by the
/// staged gate.  The full production BEM mesh remains the candidate's native
/// resolution; this small FEM–BEM fixture is a retained reference contract,
/// and its residual is checked at every candidate frequency.
fn production_reference_discretization() -> CylinderFixtureDiscretization {
    CylinderFixtureDiscretization {
        n_circumference: 12,
        n_box_depth: 3,
        n_cap_radial: 2,
        n_aperture_radial: 2,
        n_port_depth: 2,
    }
}

fn default_unlimited_resource_limit() -> usize {
    usize::MAX
}

/// Optimization parameterization selected by a versioned plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DesignSpace {
    /// Horizontal/vertical rim radius and interior-knot fraction, plus depth.
    WaveguideFive,
    /// Modeling-owned cabinet and driver-placement variables from a
    /// validated [`ProductDesign`].
    Product,
    /// Compatibility marker for the pre-existing all-geometry optimizer.
    LegacyAll,
}

/// Derivative policy is part of the serialized optimization contract. The
/// active five-parameter waveguide path uses a discrete adjoint when it is
/// available and a bounded deterministic derivative-free fallback in the
/// refinement simplex; no unverified gradient is silently used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DerivativePolicy {
    #[default]
    AdjointOrBoundedDerivativeFree,
    /// Product cabinet/placement/crossover variables currently use the
    /// bounded deterministic simplex path until a distributed product
    /// derivative contract is available.
    ProductBoundedDerivativeFree,
}

/// Frequency and mesh resolution for one simulation stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageResolution {
    pub frequency_start_hz: f64,
    pub frequency_end_hz: f64,
    pub frequency_points: usize,
    pub elements_per_wavelength: usize,
}

/// Coarse bounded-search stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoarseStageConfig {
    pub resolution: StageResolution,
    pub max_iterations: usize,
    pub shortlist_size: usize,
    pub distinct_distance: f64,
    pub adjoint_max_dofs: usize,
}

/// High-fidelity bounded Nelder–Mead stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RefinementStageConfig {
    pub resolution: StageResolution,
    pub max_evaluations: usize,
    pub fgmres_tolerance: f64,
    pub aca_tolerance: f64,
}

/// Production validation and mesh-convergence stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidationStageConfig {
    pub frequency_start_hz: f64,
    pub frequency_end_hz: f64,
    pub frequency_points: usize,
    pub production_elements_per_wavelength: usize,
    pub convergence_elements_per_wavelength: usize,
}

/// Gates that must all pass before an optimized model replaces the input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcceptanceCriteria {
    pub minimum_preference_score_improvement: f64,
    pub minimum_on_axis_rms_improvement_db: f64,
    pub minimum_pir_rms_improvement_db: f64,
    pub maximum_mesh_curve_rms_difference_db: f64,
    pub maximum_mesh_score_difference: f64,
    /// Optional product-level physical limits. When present, missing metric
    /// evidence is fail-safe and rejects the candidate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub product_limits: Option<CoupledProductLimits>,
}

/// Optional deterministic robust-evaluation policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobustUncertaintyConfig {
    pub tolerance: SpeakerToleranceSpec,
    pub sampling: UncertaintySampling,
    /// Hard cap on scenario evaluations per candidate and stage.
    pub max_cases: usize,
}

impl RobustUncertaintyConfig {
    fn cases(&self) -> Result<Vec<SpeakerUncertaintyCase>, OptimizationError> {
        if self.max_cases == 0 {
            return Err(OptimizationError::new(
                "robust uncertainty max_cases must be positive",
            ));
        }
        let cases = generate_uncertainty_cases(self.tolerance, self.sampling)
            .map_err(|error| OptimizationError::new(error.to_string()))?;
        if cases.is_empty() || cases.len() > self.max_cases {
            return Err(OptimizationError::new(format!(
                "robust uncertainty sampling produced {} cases, exceeding max_cases {}",
                cases.len(),
                self.max_cases
            )));
        }
        Ok(cases)
    }
}

/// Deterministic hard limits for one staged optimization run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptimizationResourceBudget {
    /// Maximum number of unique candidate records that may invoke the
    /// evaluator. Cache hits do not consume this budget.
    pub max_candidate_evaluations: usize,
    /// Maximum number of uncertainty scenario solves across the whole run.
    /// The limit is relevant only when robust uncertainty is enabled.
    pub max_scenario_evaluations: usize,
    /// Maximum number of frequency points solved across all candidates and
    /// uncertainty scenarios.
    #[serde(default = "default_unlimited_resource_limit")]
    pub max_frequency_evaluations: usize,
    /// Maximum number of frequency-point/RHS work units across the run.
    /// A two-driver candidate consumes two units per frequency point.
    #[serde(default = "default_unlimited_resource_limit")]
    pub max_rhs_evaluations: usize,
}

impl OptimizationResourceBudget {
    fn validate(self) -> Result<(), OptimizationError> {
        if self.max_candidate_evaluations == 0 {
            return Err(OptimizationError::new(
                "resource budget max_candidate_evaluations must be positive",
            ));
        }
        if self.max_scenario_evaluations == 0 {
            return Err(OptimizationError::new(
                "resource budget max_scenario_evaluations must be positive",
            ));
        }
        if self.max_frequency_evaluations == 0 {
            return Err(OptimizationError::new(
                "resource budget max_frequency_evaluations must be positive",
            ));
        }
        if self.max_rhs_evaluations == 0 {
            return Err(OptimizationError::new(
                "resource budget max_rhs_evaluations must be positive",
            ));
        }
        Ok(())
    }
}

/// Version-one staged optimization plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptimizationPlanV1 {
    pub version: u32,
    pub design_space: DesignSpace,
    #[serde(default)]
    pub derivative_policy: DerivativePolicy,
    pub coarse: CoarseStageConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refinement: Option<RefinementStageConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validation: Option<ValidationStageConfig>,
    pub acceptance: AcceptanceCriteria,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub robust_uncertainty: Option<RobustUncertaintyConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resource_budget: Option<OptimizationResourceBudget>,
}

impl OptimizationPlanV1 {
    /// Preview preset: only the inexpensive 1.5–2.8 kHz search.
    pub fn waveguide_preview() -> Self {
        Self {
            version: PLAN_VERSION,
            design_space: DesignSpace::WaveguideFive,
            derivative_policy: DerivativePolicy::default(),
            coarse: CoarseStageConfig {
                resolution: StageResolution {
                    frequency_start_hz: 1_500.0,
                    frequency_end_hz: 2_800.0,
                    frequency_points: 10,
                    elements_per_wavelength: 8,
                },
                max_iterations: 8,
                shortlist_size: 4,
                distinct_distance: 0.10,
                adjoint_max_dofs: 5_000,
            },
            refinement: None,
            validation: None,
            acceptance: AcceptanceCriteria::default(),
            robust_uncertainty: None,
            resource_budget: None,
        }
    }

    /// Preview preset for the modeling-owned cabinet, crossover, and
    /// placement variables.  Product candidates use the direct mesh path;
    /// ported cabinets remain rejected until exterior port coupling lands.
    pub fn product_preview() -> Self {
        let mut plan = Self::waveguide_preview();
        plan.design_space = DesignSpace::Product;
        plan.derivative_policy = DerivativePolicy::ProductBoundedDerivativeFree;
        plan
    }

    /// Balanced preset: coarse search, capped refinement, and production validation.
    pub fn waveguide_balanced() -> Self {
        let mut plan = Self::waveguide_preview();
        plan.refinement = Some(RefinementStageConfig {
            resolution: StageResolution {
                frequency_start_hz: 1_500.0,
                frequency_end_hz: 5_000.0,
                frequency_points: 14,
                elements_per_wavelength: 8,
            },
            max_evaluations: 24,
            fgmres_tolerance: 1.0e-4,
            aca_tolerance: 1.0e-5,
        });
        plan.validation = Some(ValidationStageConfig {
            frequency_start_hz: 200.0,
            frequency_end_hz: 5_000.0,
            frequency_points: 40,
            production_elements_per_wavelength: 8,
            convergence_elements_per_wavelength: 10,
        });
        plan
    }

    /// Validate a deserialized plan before any output or simulations occur.
    pub fn validate(&self) -> Result<(), OptimizationError> {
        if self.version != PLAN_VERSION {
            return Err(OptimizationError::new(format!(
                "unsupported optimization plan version {}; expected {PLAN_VERSION}",
                self.version
            )));
        }
        if !matches!(
            self.design_space,
            DesignSpace::WaveguideFive | DesignSpace::Product
        ) {
            return Err(OptimizationError::new(
                "optimize_with_plan requires a supported product or waveguide design space",
            ));
        }
        match (self.design_space, self.derivative_policy) {
            (DesignSpace::WaveguideFive, DerivativePolicy::AdjointOrBoundedDerivativeFree)
            | (DesignSpace::Product, DerivativePolicy::ProductBoundedDerivativeFree) => {}
            _ => {
                return Err(OptimizationError::new(
                    "derivative policy does not match the selected design space",
                ));
            }
        }
        validate_resolution("coarse", &self.coarse.resolution)?;
        if self.coarse.max_iterations == 0
            || self.coarse.shortlist_size == 0
            || self.coarse.adjoint_max_dofs == 0
            || !self.coarse.distinct_distance.is_finite()
            || self.coarse.distinct_distance < 0.0
        {
            return Err(OptimizationError::new(
                "coarse stage budgets and distinct distance must be positive and finite",
            ));
        }
        if let Some(refinement) = &self.refinement {
            validate_resolution("refinement", &refinement.resolution)?;
            if refinement.max_evaluations < 2 {
                return Err(OptimizationError::new(
                    "refinement max_evaluations must be at least 2",
                ));
            }
            if !refinement.fgmres_tolerance.is_finite()
                || refinement.fgmres_tolerance <= 0.0
                || !refinement.aca_tolerance.is_finite()
                || refinement.aca_tolerance <= 0.0
                || refinement.aca_tolerance > refinement.fgmres_tolerance
            {
                return Err(OptimizationError::new(
                    "refinement tolerances must be positive and ACA must not exceed FGMRES",
                ));
            }
        }
        if let Some(validation) = &self.validation {
            let resolution = StageResolution {
                frequency_start_hz: validation.frequency_start_hz,
                frequency_end_hz: validation.frequency_end_hz,
                frequency_points: validation.frequency_points,
                elements_per_wavelength: validation.production_elements_per_wavelength,
            };
            validate_resolution("validation", &resolution)?;
            if validation.convergence_elements_per_wavelength
                <= validation.production_elements_per_wavelength
            {
                return Err(OptimizationError::new(
                    "mesh-convergence resolution must be finer than production resolution",
                ));
            }
        }
        self.acceptance
            .validate()
            .and_then(|_| {
                self.robust_uncertainty
                    .as_ref()
                    .map_or(Ok(()), |config| config.cases().map(|_| ()))
            })
            .and_then(|_| {
                self.resource_budget
                    .map_or(Ok(()), OptimizationResourceBudget::validate)
            })
    }
}

impl Default for AcceptanceCriteria {
    fn default() -> Self {
        Self {
            minimum_preference_score_improvement: 0.05,
            minimum_on_axis_rms_improvement_db: 0.05,
            minimum_pir_rms_improvement_db: 0.05,
            maximum_mesh_curve_rms_difference_db: 0.25,
            maximum_mesh_score_difference: 0.10,
            product_limits: None,
        }
    }
}

impl AcceptanceCriteria {
    fn validate(&self) -> Result<(), OptimizationError> {
        for (name, value) in [
            (
                "minimum preference-score improvement",
                self.minimum_preference_score_improvement,
            ),
            (
                "minimum on-axis RMS improvement",
                self.minimum_on_axis_rms_improvement_db,
            ),
            (
                "minimum PIR RMS improvement",
                self.minimum_pir_rms_improvement_db,
            ),
            (
                "maximum mesh curve RMS difference",
                self.maximum_mesh_curve_rms_difference_db,
            ),
            (
                "maximum mesh score difference",
                self.maximum_mesh_score_difference,
            ),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(OptimizationError::new(format!(
                    "{name} must be finite and non-negative"
                )));
            }
        }
        if let Some(limits) = &self.product_limits {
            limits
                .validate()
                .map_err(|error| OptimizationError::new(error.to_string()))?;
        }
        Ok(())
    }
}

fn validate_resolution(name: &str, resolution: &StageResolution) -> Result<(), OptimizationError> {
    if !resolution.frequency_start_hz.is_finite()
        || !resolution.frequency_end_hz.is_finite()
        || resolution.frequency_start_hz <= 0.0
        || resolution.frequency_end_hz <= resolution.frequency_start_hz
        || resolution.frequency_points < 2
        || resolution.elements_per_wavelength < 4
    {
        return Err(OptimizationError::new(format!(
            "{name} resolution must have a positive increasing band, at least two frequencies, and at least four elements per wavelength"
        )));
    }
    Ok(())
}

/// Metrics saved for each candidate evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateMetrics {
    pub smooth_objective: f64,
    pub preference_score: f64,
    pub on_axis_rms_db: f64,
    pub pir_rms_db: f64,
    pub maximum_pir_slope_db_per_octave: f64,
    pub on_axis_db: Vec<f64>,
    pub pir_db: Vec<f64>,
    /// Modeling-owned product metrics used by optional acceptance limits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coupled_product_metrics: Option<CoupledProductMetrics>,
}

/// Persistent status for a candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationStatus {
    Completed,
    InvalidGeometry {
        reason: String,
    },
    SimulationFailed {
        reason: String,
    },
    BudgetExhausted {
        resource: BudgetResource,
        limit: usize,
    },
}

/// Resource whose hard limit prevented an evaluation from running.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetResource {
    CandidateEvaluations,
    ScenarioEvaluations,
    FrequencyEvaluations,
    RhsEvaluations,
}

/// One append-only history entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationRecord {
    pub sequence: usize,
    pub stage: String,
    pub mesh_elements_per_wavelength: usize,
    pub normalized_parameters: Vec<f64>,
    pub physical_parameters: Vec<f64>,
    pub status: EvaluationStatus,
    pub metrics: Option<CandidateMetrics>,
    pub elapsed_seconds: f64,
    pub model_identity: String,
    pub plan_identity: String,
    /// Solver/evaluator/coupling/ROM policy identity for this exact result.
    /// Empty values are rejected while loading persisted history, so legacy
    /// records fail closed instead of becoming cache hits.
    #[serde(default)]
    pub cache_identity: String,
    #[serde(default)]
    pub solver_identity: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scenario_results: Vec<ScenarioEvaluationRecord>,
    /// Charged frequency-point work. Budget-exhausted records charge zero.
    #[serde(default)]
    pub frequency_evaluations: usize,
    /// Charged frequency-point/RHS work. Budget-exhausted records charge zero.
    #[serde(default)]
    pub rhs_evaluations: usize,
    /// Expensive exact uncompressed residual applications performed by the
    /// solver for this candidate.
    #[serde(default)]
    pub exact_residual_evaluations: usize,
}

/// Per-scenario result retained in append-only optimization history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioEvaluationRecord {
    pub index: usize,
    pub label: String,
    pub status: EvaluationStatus,
    pub metrics: Option<CandidateMetrics>,
    #[serde(default)]
    pub exact_residual_evaluations: usize,
}

/// Counters emitted with every staged optimization summary.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OptimizationTelemetry {
    pub candidate_evaluations: usize,
    pub scenario_evaluations: usize,
    #[serde(default)]
    pub frequency_evaluations: usize,
    #[serde(default)]
    pub rhs_evaluations: usize,
    #[serde(default)]
    pub exact_residual_evaluations: usize,
    pub gradient_evaluations: usize,
    pub accepted_gradient_steps: usize,
    pub failed_evaluations: usize,
    pub budget_exhausted_evaluations: usize,
    pub candidates_per_hour: f64,
    pub scenario_evaluations_per_hour: f64,
    #[serde(default)]
    pub frequency_evaluations_per_hour: f64,
    #[serde(default)]
    pub rhs_evaluations_per_hour: f64,
    #[serde(default)]
    pub exact_residual_evaluations_per_hour: f64,
    /// O(N²) residual applications are reported explicitly because their
    /// cost is not comparable to a compressed matvec.
    #[serde(default)]
    pub exact_residual_cost_units: usize,
    pub accepted_gradients_per_hour: f64,
    #[serde(default)]
    pub final_l3_validated_preference_improvement: Option<f64>,
    /// Relative bounded Taylor remainder checked before product candidate
    /// evaluations begin. Product optimization is derivative-free, but it
    /// still needs an explicit smoothness/admission record for its physical
    /// design variables.
    #[serde(default)]
    pub product_taylor_error: Option<f64>,
}

/// Acceptance details written even when the best candidate is rejected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationDiagnostics {
    pub preference_score_improvement: f64,
    pub on_axis_rms_improvement_db: f64,
    pub pir_rms_improvement_db: f64,
    pub mesh_curve_rms_difference_db: f64,
    pub mesh_score_difference: f64,
    pub passed_preference_score: bool,
    pub passed_on_axis_rms: bool,
    pub passed_pir_rms: bool,
    pub passed_mesh_curve: bool,
    pub passed_mesh_score: bool,
    pub product_constraint_minimum_margin: Option<f64>,
    pub passed_product_constraints: bool,
}

impl ValidationDiagnostics {
    fn accepted(&self) -> bool {
        self.passed_preference_score
            && self.passed_on_axis_rms
            && self.passed_pir_rms
            && self.passed_mesh_curve
            && self.passed_mesh_score
            && self.passed_product_constraints
    }
}

/// Final staged optimization summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StagedOptimizationSummary {
    pub plan_version: u32,
    pub band_label: String,
    pub model_identity: String,
    pub plan_identity: String,
    pub resumed: bool,
    pub evaluations: usize,
    pub accepted: bool,
    pub preview_only: bool,
    pub best_normalized_parameters: Vec<f64>,
    pub best_metrics: CandidateMetrics,
    pub validation: Option<ValidationDiagnostics>,
    pub elapsed_seconds: f64,
    #[serde(default)]
    pub uncertainty_scenario_count: usize,
    #[serde(default)]
    pub telemetry: OptimizationTelemetry,
    /// Objective values in append-only evaluation order for reproducible
    /// optimization-history gallery plots.
    #[serde(default)]
    pub objective_history: Vec<f64>,
}

/// Returned model and persisted summary.
pub struct StagedOptimizationResult {
    /// Input model for preview/rejection, candidate model only after acceptance.
    pub model: LoudspeakerModel,
    /// Best candidate, including when validation rejected it.
    pub best_candidate: LoudspeakerModel,
    pub summary: StagedOptimizationSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointIdentity {
    model_identity: String,
    plan_identity: String,
    /// Policy identity is part of the checkpoint contract. A history written
    /// before this field existed must fail closed rather than becoming a
    /// cache hit by defaulting to a compatible-looking value.
    #[serde(default)]
    cache_identity: String,
}

#[derive(Debug, Clone)]
struct ParameterBounds {
    design_space: DesignSpace,
    lower: Vec<f64>,
    upper: Vec<f64>,
    initial: Vec<f64>,
}

impl ParameterBounds {
    fn from_model_for_design_space(
        model: &LoudspeakerModel,
        design_space: DesignSpace,
    ) -> Result<Self, OptimizationError> {
        if design_space == DesignSpace::Product {
            let product = model.product_design.as_ref().ok_or_else(|| {
                OptimizationError::new(
                    "product optimization requires model.product_design to be present",
                )
            })?;
            product.validate().map_err(|error| {
                OptimizationError::new(format!("invalid product design: {error}"))
            })?;
            let variables = product.design_variables();
            if variables.is_empty() {
                return Err(OptimizationError::new(
                    "product design exposes no continuous optimization variables",
                ));
            }
            return Ok(Self {
                design_space,
                lower: variables.iter().map(|variable| variable.lower).collect(),
                upper: variables.iter().map(|variable| variable.upper).collect(),
                initial: variables.iter().map(|variable| variable.value).collect(),
            });
        }
        validate_waveguide_driver_inputs(model)?;
        let waveguide = model.waveguide.as_ref().ok_or_else(|| {
            OptimizationError::new("waveguide_five optimization requires a waveguide")
        })?;
        if waveguide.horizontal_spline.len() != 3 || waveguide.vertical_spline.len() != 3 {
            return Err(OptimizationError::new(
                "waveguide_five requires exactly three monotonic knots in each plane",
            ));
        }
        let throat = model.tweeter.radius;
        let horizontal_rim = waveguide.horizontal_spline[0][1];
        let vertical_rim = waveguide.vertical_spline[0][1];
        let horizontal_fraction =
            fraction_from_radius(waveguide.horizontal_spline[1][1], throat, horizontal_rim)?;
        let vertical_fraction =
            fraction_from_radius(waveguide.vertical_spline[1][1], throat, vertical_rim)?;
        let horizontal_fit = model.cabinet.width * 0.5 * 0.8;
        let vertical_fit = model.cabinet.height * 0.5 * 0.5;
        let lower = [
            (horizontal_rim * 0.7).max(1.2 * throat),
            // For the natural cubic spline used by the mesh generator, the
            // interior knot must remain in the middle half of the rim-to-
            // throat span.  Wider bounds can make the spline turn back on
            // itself between knots even though all three knot radii descend.
            0.25,
            (vertical_rim * 0.7).max(1.2 * throat),
            0.25,
            (waveguide.depth * 0.6).max(0.005),
        ];
        let upper = [
            (horizontal_rim * 1.3).min(horizontal_fit),
            0.75,
            (vertical_rim * 1.3).min(vertical_fit),
            0.75,
            (waveguide.depth * 1.4).min(model.cabinet.depth / 3.0),
        ];
        let initial = [
            horizontal_rim,
            horizontal_fraction,
            vertical_rim,
            vertical_fraction,
            waveguide.depth,
        ];
        for index in 0..WAVEGUIDE_PARAMETER_COUNT {
            if !lower[index].is_finite()
                || !upper[index].is_finite()
                || lower[index] > upper[index]
                || initial[index] < lower[index]
                || initial[index] > upper[index]
            {
                return Err(OptimizationError::new(format!(
                    "waveguide_five parameter {index} has invalid bounds or initial value"
                )));
            }
        }
        Ok(Self {
            design_space,
            lower: lower.to_vec(),
            upper: upper.to_vec(),
            initial: initial.to_vec(),
        })
    }

    fn parameter_count(&self) -> usize {
        self.initial.len()
    }

    fn normalize(&self, physical: &[f64]) -> Vec<f64> {
        physical
            .iter()
            .enumerate()
            .map(|(index, value)| {
                (value - self.lower[index]) / (self.upper[index] - self.lower[index])
            })
            .collect()
    }

    fn denormalize(&self, normalized: &[f64]) -> Result<Vec<f64>, OptimizationError> {
        if normalized.len() != self.parameter_count()
            || normalized
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(OptimizationError::new(
                "normalized parameters must be finite values in [0, 1] with the design-space length",
            ));
        }
        Ok(normalized
            .iter()
            .enumerate()
            .map(|(index, value)| {
                self.lower[index] + value * (self.upper[index] - self.lower[index])
            })
            .collect())
    }

    fn initial_normalized(&self) -> Vec<f64> {
        self.normalize(&self.initial)
    }

    fn apply(
        &self,
        normalized: &[f64],
        base: &LoudspeakerModel,
    ) -> Result<LoudspeakerModel, OptimizationError> {
        let physical = self.denormalize(normalized)?;
        if self.design_space == DesignSpace::Product {
            let product = base.product_design.as_ref().ok_or_else(|| {
                OptimizationError::new("product optimization requires product_design")
            })?;
            let product = product.from_design_variables(&physical).map_err(|error| {
                OptimizationError::new(format!("product variables invalid: {error}"))
            })?;
            let mut model = base.clone();
            model.cabinet.width = product.cabinet.baffle_width_m;
            model.cabinet.height = product.cabinet.baffle_height_m;
            model.cabinet.depth = product.cabinet.box_depth_m;
            let mut model_drivers = model.drivers_mut();
            if model_drivers.len() != product.drivers.len() {
                return Err(OptimizationError::new(
                    "product design driver count does not match the speaker model",
                ));
            }
            for (driver, product_driver) in model_drivers.iter_mut().zip(&product.drivers) {
                driver.x_offset = product_driver.position_m[0];
                driver.y_offset = product_driver.position_m[1];
            }
            model.product_design = Some(product);
            model.validate().map_err(|message| {
                OptimizationError::new(format!("product candidate is invalid: {message}"))
            })?;
            return Ok(model);
        }
        let throat = base.tweeter.radius;
        let mut model = base.clone();
        let waveguide = model.waveguide.as_mut().ok_or_else(|| {
            OptimizationError::new("waveguide_five optimization requires a waveguide")
        })?;
        waveguide.horizontal_spline[0][1] = physical[0];
        waveguide.horizontal_spline[1][1] = throat + physical[1] * (physical[0] - throat);
        waveguide.horizontal_spline[2][1] = throat;
        waveguide.vertical_spline[0][1] = physical[2];
        waveguide.vertical_spline[1][1] = throat + physical[3] * (physical[2] - throat);
        waveguide.vertical_spline[2][1] = throat;
        waveguide.depth = physical[4];
        validate_waveguide_geometry(&model)?;
        Ok(model)
    }
}

fn product_design_taylor_error(product: &ProductDesign) -> Result<f64, OptimizationError> {
    let variables = product.design_variables();
    if variables.is_empty() {
        return Err(OptimizationError::new(
            "product derivative check requires at least one design variable",
        ));
    }
    let values = variables
        .iter()
        .map(|variable| variable.value)
        .collect::<Vec<_>>();
    let mut direction = vec![0.0; values.len()];
    direction[0] = 1.0;
    product
        .bounded_taylor_error(&values, &direction, 1.0e-4, |candidate| {
            let voltages = candidate.drive_voltages_at(1_000.0, Complex64::new(1.0, 0.0))?;
            if voltages.is_empty() {
                return Err(sonium_sim_speaker_modeling::ProductDesignError::NoDrivers);
            }
            Ok(voltages.iter().map(|voltage| voltage.norm_sqr()).sum())
        })
        .map_err(|error| OptimizationError::new(format!("product Taylor check failed: {error}")))
}

fn fraction_from_radius(interior: f64, throat: f64, rim: f64) -> Result<f64, OptimizationError> {
    let span = rim - throat;
    if !span.is_finite() || span <= 0.0 {
        return Err(OptimizationError::new(
            "waveguide rim radius must exceed the fixed throat radius",
        ));
    }
    let fraction = (interior - throat) / span;
    if !(0.25..=0.75).contains(&fraction) {
        return Err(OptimizationError::new(format!(
            "initial waveguide interior-knot fraction {fraction:.4} is outside [0.25, 0.75]"
        )));
    }
    Ok(fraction)
}

fn validate_waveguide_driver_inputs(model: &LoudspeakerModel) -> Result<(), OptimizationError> {
    model.validate().map_err(|message| {
        OptimizationError::new(format!("invalid loudspeaker model: {message}"))
    })?;
    for (name, driver) in [("woofer", &model.woofer), ("tweeter", &model.tweeter)] {
        let span = driver_frequency_span(driver).ok_or_else(|| {
            OptimizationError::new(format!(
                "{name} must provide a typed driver model or complex transfer function spanning 200 Hz–5 kHz; flat fallback pistons are not allowed"
            ))
        })?;
        if span.0 > 200.0 || span.1 < 5_000.0 {
            return Err(OptimizationError::new(format!(
                "{name} response spans {:.1}–{:.1} Hz and would clamp outside its data; waveguide mode requires 200–5000 Hz coverage",
                span.0, span.1
            )));
        }
    }
    Ok(())
}

fn driver_frequency_span(driver: &DriverConfig) -> Option<(f64, f64)> {
    if let Some(model) = &driver.driver_model {
        return Some((
            model.velocity_transfer.first()?.frequency_hz,
            model.velocity_transfer.last()?.frequency_hz,
        ));
    }
    let transfer = driver.transfer_function.as_ref()?;
    Some((transfer.first()?.freq, transfer.last()?.freq))
}

/// Densely check spline flare and the generated parametric surface.
fn validate_waveguide_geometry(model: &LoudspeakerModel) -> Result<(), OptimizationError> {
    let waveguide = model.waveguide.as_ref().ok_or_else(|| {
        OptimizationError::new("waveguide_five optimization requires a waveguide")
    })?;
    validate_profile(
        "horizontal",
        &waveguide.horizontal_spline,
        model.tweeter.radius,
    )?;
    validate_profile("vertical", &waveguide.vertical_spline, model.tweeter.radius)?;
    let h =
        CubicSpline::try_from_control_points(&waveguide.horizontal_spline).map_err(|error| {
            OptimizationError::new(format!("invalid horizontal waveguide spline: {error}"))
        })?;
    let v = CubicSpline::try_from_control_points(&waveguide.vertical_spline).map_err(|error| {
        OptimizationError::new(format!("invalid vertical waveguide spline: {error}"))
    })?;
    let rim_h = h
        .try_evaluate(0.0)
        .map_err(|error| OptimizationError::new(format!("invalid spline query: {error}")))?;
    let rim_v = v
        .try_evaluate(0.0)
        .map_err(|error| OptimizationError::new(format!("invalid spline query: {error}")))?;
    if rim_h > model.cabinet.width * 0.5 * 0.8 || rim_v > model.cabinet.height * 0.5 * 0.5 {
        return Err(OptimizationError::new(
            "waveguide rim exceeds the cabinet-fit envelope",
        ));
    }
    let tweeter_to_woofer = (model.tweeter.y_offset - model.woofer.y_offset).abs();
    if tweeter_to_woofer <= rim_v + model.woofer.radius {
        return Err(OptimizationError::new(
            "waveguide envelope intersects the woofer clearance",
        ));
    }
    for sample in 0..=256 {
        let t = sample as f64 / 256.0;
        let query = |value: Result<f64, _>| {
            value.map_err(|error| OptimizationError::new(format!("invalid spline query: {error}")))
        };
        let rh = query(h.try_evaluate(t))?;
        let rv = query(v.try_evaluate(t))?;
        let rh_prime = query(h.try_derivative(t))?;
        let rv_prime = query(v.try_derivative(t))?;
        for azimuth in 0..32 {
            let angle = std::f64::consts::TAU * azimuth as f64 / 32.0;
            let cos = angle.cos();
            let sin = angle.sin();
            let nx = waveguide.depth * rv * cos;
            let ny = waveguide.depth * rh * sin;
            let nz = rh_prime * rv * cos * cos + rv_prime * rh * sin * sin;
            let norm_squared = nx * nx + ny * ny + nz * nz;
            if !norm_squared.is_finite() || norm_squared <= 1.0e-24 {
                return Err(OptimizationError::new(
                    "waveguide profile produces a non-finite or degenerate normal",
                ));
            }
        }
    }
    Ok(())
}

fn validate_profile(name: &str, points: &[[f64; 2]], throat: f64) -> Result<(), OptimizationError> {
    if points.len() != 3 {
        return Err(OptimizationError::new(format!(
            "waveguide_five requires exactly three {name} knots"
        )));
    }
    let spline = CubicSpline::try_from_control_points(points).map_err(|error| {
        OptimizationError::new(format!(
            "{name} waveguide profile is not a valid spline: {error}"
        ))
    })?;
    let query = |value: Result<f64, _>| {
        value.map_err(|error| OptimizationError::new(format!("invalid spline query: {error}")))
    };
    let mut previous = query(spline.try_evaluate(0.0))?;
    for sample in 0..=512 {
        let t = sample as f64 / 512.0;
        let radius = query(spline.try_evaluate(t))?;
        if !radius.is_finite() || radius <= 0.0 || radius < throat - 1.0e-10 {
            return Err(OptimizationError::new(format!(
                "{name} waveguide profile has a non-positive or sub-throat radius"
            )));
        }
        if sample > 0 && radius > previous + 1.0e-10 {
            return Err(OptimizationError::new(format!(
                "{name} waveguide profile is not a monotonic flare"
            )));
        }
        previous = radius;
    }
    Ok(())
}

struct ProductionEvaluator {
    warm_start: SimulationWarmStart,
    reference: LoudspeakerModel,
    envelope: LoudspeakerModel,
    design_space: DesignSpace,
    topologies: HashMap<String, PinnedWaveguideMesh>,
    require_reference_sweep: bool,
}

trait CandidateEvaluator {
    fn evaluate(
        &mut self,
        model: &LoudspeakerModel,
        resolution: &StageResolution,
        solver: &SolverConfig,
    ) -> Result<SimulationResult, OptimizationError>;

    /// Number of independent RHS loads solved at each frequency point.
    /// The production two-driver BEM path uses one RHS per driver.
    fn rhs_count(&self, model: &LoudspeakerModel) -> usize {
        model.drivers().len()
    }

    /// Stable policy identity for persisted evaluation results. This must
    /// include the evaluator backend and any coupling/ROM policy that can
    /// change the returned metrics.
    fn cache_identity(&self) -> String;
}

impl ProductionEvaluator {
    fn new(
        reference: &LoudspeakerModel,
        bounds: &ParameterBounds,
        require_reference_sweep: bool,
    ) -> Result<Self, OptimizationError> {
        let envelope = bounds.apply(&vec![1.0; bounds.parameter_count()], reference)?;
        Ok(Self {
            warm_start: SimulationWarmStart::default(),
            reference: reference.clone(),
            envelope,
            design_space: bounds.design_space,
            topologies: HashMap::new(),
            require_reference_sweep,
        })
    }

    fn attach_reference_sweep(
        &self,
        mut result: SimulationResult,
        candidate: &LoudspeakerModel,
        solver: &SolverConfig,
    ) -> Result<SimulationResult, OptimizationError> {
        if !self.require_reference_sweep {
            return Ok(result);
        }
        let product = candidate.product_design.as_ref().ok_or_else(|| {
            OptimizationError::new(
                "distributed FEM-BEM reference evidence requires model.product_design",
            )
        })?;
        let frequencies = candidate.simulation.frequencies();
        let reference_solver = LinearSolveConfig {
            // The small distributed reference has a denser coupled block
            // than the exterior candidate solve.  Use the validated FEM–BEM
            // reference configuration rather than inheriting a coarse BEM
            // stage's restart/tolerance budget.
            solver: sonium_solvers::LinearSolverType::Auto,
            max_iterations: solver.max_iterations.max(2_000),
            // This retained reference is a convergence/evidence lane; the
            // scalar residual is still carried into acceptance, where a
            // stricter product-specific limit may reject it.
            tolerance: solver.tolerance.max(1.0e-4),
            restart: 30,
            // Product acceptance requires retained residual evidence at every
            // reference frequency; an estimator-only sweep is not enough.
            exact_residual_check_interval: 0,
            ..LinearSolveConfig::default()
        };
        // The distributed fixture currently supports only the centred
        // single-driver product lowering.  On that path retain a ROM
        // promotion report alongside the full-order physical sweep.  A
        // multi-driver product continues through the full-order reference
        // path until its distributed N-port FEM boundary is available.
        let sweep = if product.drivers.len() == 1 {
            let policy = RomPromotionPolicy {
                max_full_order_solves: frequencies.len().max(1),
                ..RomPromotionPolicy::default()
            };
            product.solve_fem_bem_reference_sweep_with_rom(
                production_reference_discretization(),
                candidate.simulation.speed_of_sound,
                candidate.simulation.air_density,
                &frequencies,
                Complex64::new(1.0, 0.0),
                &reference_solver,
                policy,
            )
        } else {
            product.solve_fem_bem_reference_sweep(
                production_reference_discretization(),
                candidate.simulation.speed_of_sound,
                candidate.simulation.air_density,
                &frequencies,
                Complex64::new(1.0, 0.0),
                &reference_solver,
            )
        }
        .map_err(|error| {
            OptimizationError::new(format!(
                "distributed FEM-BEM reference sweep failed: {error}"
            ))
        })?;
        result.product_reference_sweep = Some(sweep);
        Ok(result)
    }
}

impl CandidateEvaluator for ProductionEvaluator {
    fn cache_identity(&self) -> String {
        format!(
            "production-evaluator:v5;backend=cpu-bem-v1;coupling=partitioned-nport-v1;rom=promotion-v1;driver-normal-contract=planar-v1;reference-sweep={};design-space={:?}",
            self.require_reference_sweep, self.design_space
        )
    }

    fn evaluate(
        &mut self,
        model: &LoudspeakerModel,
        resolution: &StageResolution,
        solver: &SolverConfig,
    ) -> Result<SimulationResult, OptimizationError> {
        let mut candidate = model.clone();
        candidate.simulation.freq_start = resolution.frequency_start_hz;
        candidate.simulation.freq_end = resolution.frequency_end_hz;
        candidate.simulation.freq_points = resolution.frequency_points;
        let mesh_frequency =
            resolution.frequency_end_hz * resolution.elements_per_wavelength as f64 / 8.0;
        candidate.simulation.max_mesh_frequency = Some(mesh_frequency);
        if self.design_space == DesignSpace::Product {
            let result = simulation::run_simulation_with_warm_start(
                &candidate,
                &cea2034_output::generate_eval_points(candidate.baffle_center()),
                solver,
                &mut self.warm_start,
            )
            .map_err(OptimizationError::from)?;
            return self.attach_reference_sweep(result, &candidate, solver);
        }
        let mut reference = self.reference.clone();
        reference.simulation = candidate.simulation.clone();
        let mut envelope = self.envelope.clone();
        envelope.simulation = candidate.simulation.clone();
        let topology_key = format!(
            "{}:{}",
            resolution.elements_per_wavelength, resolution.frequency_end_hz
        );
        let topology = match self.topologies.get(&topology_key) {
            Some(topology) => topology.clone(),
            None => {
                let topology = PinnedWaveguideMesh::new(&reference, &envelope)
                    .map_err(OptimizationError::new)?;
                self.topologies.insert(topology_key, topology.clone());
                topology
            }
        };
        let result = simulation::run_simulation_with_warm_start_and_topology(
            &candidate,
            &cea2034_output::generate_eval_points(candidate.baffle_center()),
            solver,
            &mut self.warm_start,
            &topology,
        )
        .map_err(OptimizationError::from)?;
        self.attach_reference_sweep(result, &candidate, solver)
    }
}

struct RunContext<'a> {
    base: &'a LoudspeakerModel,
    bounds: ParameterBounds,
    output: &'a Path,
    identity: CheckpointIdentity,
    records: Vec<EvaluationRecord>,
    cache: HashMap<String, EvaluationRecord>,
    evaluator: Box<dyn CandidateEvaluator>,
    uncertainty_cases: Vec<SpeakerUncertaintyCase>,
    resource_budget: Option<OptimizationResourceBudget>,
    candidate_evaluations: usize,
    scenario_evaluations: usize,
    frequency_evaluations: usize,
    rhs_evaluations: usize,
    exact_residual_evaluations: usize,
    gradient_evaluations: usize,
    accepted_gradient_steps: usize,
}

impl<'a> RunContext<'a> {
    fn evaluate(
        &mut self,
        stage: &str,
        normalized: &[f64],
        resolution: &StageResolution,
        solver: &SolverConfig,
    ) -> CandidateMetrics {
        let solver_identity = solver_cache_identity(solver);
        let cache_key = evaluation_key(
            self.bounds.design_space,
            stage,
            resolution.elements_per_wavelength,
            normalized,
            &self.identity.cache_identity,
            &solver_identity,
        );
        if let Some(record) = self.cache.get(&cache_key) {
            return record
                .metrics
                .clone()
                .unwrap_or_else(|| penalty_metrics(normalized));
        }
        if let Some(limit) = self.resource_budget.and_then(|budget| {
            (self.candidate_evaluations >= budget.max_candidate_evaluations)
                .then_some(budget.max_candidate_evaluations)
        }) {
            let physical = self
                .bounds
                .denormalize(normalized)
                .map(|values| values.to_vec())
                .unwrap_or_default();
            let record = EvaluationRecord {
                sequence: self.records.len(),
                stage: stage.to_string(),
                mesh_elements_per_wavelength: resolution.elements_per_wavelength,
                normalized_parameters: normalized.to_vec(),
                physical_parameters: physical,
                status: EvaluationStatus::BudgetExhausted {
                    resource: BudgetResource::CandidateEvaluations,
                    limit,
                },
                metrics: None,
                elapsed_seconds: 0.0,
                model_identity: self.identity.model_identity.clone(),
                plan_identity: self.identity.plan_identity.clone(),
                cache_identity: self.identity.cache_identity.clone(),
                solver_identity: solver_identity.clone(),
                scenario_results: Vec::new(),
                frequency_evaluations: 0,
                rhs_evaluations: 0,
                exact_residual_evaluations: 0,
            };
            self.store_record(cache_key, record);
            return penalty_metrics(normalized);
        }
        self.candidate_evaluations += 1;
        let started = Instant::now();
        let physical = self
            .bounds
            .denormalize(normalized)
            .map(|values| values.to_vec())
            .unwrap_or_default();
        let (
            status,
            metrics,
            scenario_results,
            frequency_evaluations,
            rhs_evaluations,
            exact_residual_evaluations,
        ) = match self.bounds.apply(normalized, self.base) {
            Err(error) => (
                EvaluationStatus::InvalidGeometry {
                    reason: error.to_string(),
                },
                None,
                Vec::new(),
                0,
                0,
                0,
            ),
            Ok(model) => {
                let scenario_count = self.uncertainty_cases.len().max(1);
                let scenario_evaluations = self.uncertainty_cases.len();
                let rhs_count = self.evaluator.rhs_count(&model);
                let work = resolution
                    .frequency_points
                    .checked_mul(scenario_count)
                    .and_then(|frequencies| {
                        frequencies
                            .checked_mul(rhs_count)
                            .map(|rhs| (frequencies, rhs))
                    });

                if rhs_count == 0 {
                    (
                        EvaluationStatus::SimulationFailed {
                            reason: "candidate evaluator reported zero RHS loads".to_string(),
                        },
                        None,
                        Vec::new(),
                        0,
                        0,
                        0,
                    )
                } else if let Some((resource, limit)) =
                    self.work_budget_exhausted(scenario_evaluations, work)
                {
                    let scenarios = self.budget_scenarios(resource, limit);
                    (
                        EvaluationStatus::BudgetExhausted { resource, limit },
                        None,
                        scenarios,
                        0,
                        0,
                        0,
                    )
                } else if let Some((frequency_work, rhs_work)) = work {
                    self.frequency_evaluations += frequency_work;
                    self.rhs_evaluations += rhs_work;
                    self.scenario_evaluations += scenario_evaluations;
                    let exact_before = self.exact_residual_evaluations;
                    let (status, metrics, scenarios) =
                        self.evaluate_model(&model, normalized, resolution, solver);
                    (
                        status,
                        metrics,
                        scenarios,
                        frequency_work,
                        rhs_work,
                        self.exact_residual_evaluations - exact_before,
                    )
                } else {
                    (
                        EvaluationStatus::SimulationFailed {
                            reason: "candidate work-unit count overflowed usize".to_string(),
                        },
                        None,
                        Vec::new(),
                        0,
                        0,
                        0,
                    )
                }
            }
        };
        let record = EvaluationRecord {
            sequence: self.records.len(),
            stage: stage.to_string(),
            mesh_elements_per_wavelength: resolution.elements_per_wavelength,
            normalized_parameters: normalized.to_vec(),
            physical_parameters: physical,
            status,
            metrics: metrics.clone(),
            elapsed_seconds: started.elapsed().as_secs_f64(),
            model_identity: self.identity.model_identity.clone(),
            plan_identity: self.identity.plan_identity.clone(),
            cache_identity: self.identity.cache_identity.clone(),
            solver_identity,
            scenario_results,
            frequency_evaluations,
            rhs_evaluations,
            exact_residual_evaluations,
        };
        self.store_record(cache_key, record);
        metrics.unwrap_or_else(|| penalty_metrics(normalized))
    }

    fn store_record(&mut self, cache_key: String, record: EvaluationRecord) {
        if let Err(error) = append_record(self.output, &record) {
            log::error!("failed to persist optimization evaluation: {error}");
        }
        self.cache.insert(cache_key, record.clone());
        self.records.push(record);
    }

    fn work_budget_exhausted(
        &self,
        scenario_evaluations: usize,
        work: Option<(usize, usize)>,
    ) -> Option<(BudgetResource, usize)> {
        let budget = self.resource_budget?;
        if scenario_evaluations > 0
            && self
                .scenario_evaluations
                .saturating_add(scenario_evaluations)
                > budget.max_scenario_evaluations
        {
            return Some((
                BudgetResource::ScenarioEvaluations,
                budget.max_scenario_evaluations,
            ));
        }
        let (frequency_work, rhs_work) = work?;
        if self.frequency_evaluations.saturating_add(frequency_work)
            > budget.max_frequency_evaluations
        {
            return Some((
                BudgetResource::FrequencyEvaluations,
                budget.max_frequency_evaluations,
            ));
        }
        if self.rhs_evaluations.saturating_add(rhs_work) > budget.max_rhs_evaluations {
            return Some((BudgetResource::RhsEvaluations, budget.max_rhs_evaluations));
        }
        None
    }

    fn budget_scenarios(
        &self,
        resource: BudgetResource,
        limit: usize,
    ) -> Vec<ScenarioEvaluationRecord> {
        self.uncertainty_cases
            .iter()
            .map(|case| ScenarioEvaluationRecord {
                index: case.index,
                label: case.label.clone(),
                status: EvaluationStatus::BudgetExhausted { resource, limit },
                metrics: None,
                exact_residual_evaluations: 0,
            })
            .collect()
    }

    fn evaluate_model(
        &mut self,
        model: &LoudspeakerModel,
        normalized: &[f64],
        resolution: &StageResolution,
        solver: &SolverConfig,
    ) -> (
        EvaluationStatus,
        Option<CandidateMetrics>,
        Vec<ScenarioEvaluationRecord>,
    ) {
        if self.uncertainty_cases.is_empty() {
            return match self.evaluator.evaluate(model, resolution, solver) {
                Ok(result) => {
                    self.exact_residual_evaluations += result.exact_residual_evaluations;
                    match candidate_metrics(
                        &result,
                        normalized,
                        &self.bounds.initial_normalized(),
                        model.product_design.as_ref(),
                    ) {
                        Ok(metrics) => (EvaluationStatus::Completed, Some(metrics), Vec::new()),
                        Err(error) => (
                            EvaluationStatus::SimulationFailed {
                                reason: error.to_string(),
                            },
                            None,
                            Vec::new(),
                        ),
                    }
                }
                Err(error) => (
                    EvaluationStatus::SimulationFailed {
                        reason: error.to_string(),
                    },
                    None,
                    Vec::new(),
                ),
            };
        }

        let initial = self.bounds.initial_normalized();
        let mut scenarios = Vec::with_capacity(self.uncertainty_cases.len());
        for case in &self.uncertainty_cases {
            let result = apply_uncertainty_case(model, case).and_then(|scenario_model| {
                self.evaluator
                    .evaluate(&scenario_model, resolution, solver)
                    .and_then(|result| {
                        let exact_residual_evaluations = result.exact_residual_evaluations;
                        candidate_metrics(
                            &result,
                            normalized,
                            &initial,
                            scenario_model.product_design.as_ref(),
                        )
                        .map(|mut metrics| {
                            // A uniform measurement offset does not alter the
                            // centered flatness scalars, but it must remain in
                            // the persisted curve evidence used by validation.
                            for value in &mut metrics.on_axis_db {
                                *value += case.measurement_offset_db;
                            }
                            for value in &mut metrics.pir_db {
                                *value += case.measurement_offset_db;
                            }
                            (metrics, exact_residual_evaluations)
                        })
                    })
            });
            match result {
                Ok((metrics, exact_residuals)) => {
                    self.exact_residual_evaluations += exact_residuals;
                    scenarios.push(ScenarioEvaluationRecord {
                        index: case.index,
                        label: case.label.clone(),
                        status: EvaluationStatus::Completed,
                        metrics: Some(metrics),
                        exact_residual_evaluations: exact_residuals,
                    })
                }
                Err(error) => scenarios.push(ScenarioEvaluationRecord {
                    index: case.index,
                    label: case.label.clone(),
                    status: EvaluationStatus::SimulationFailed {
                        reason: error.to_string(),
                    },
                    metrics: None,
                    exact_residual_evaluations: 0,
                }),
            }
        }

        if let Some(failed) = scenarios.iter().find(|scenario| scenario.metrics.is_none()) {
            return (
                EvaluationStatus::SimulationFailed {
                    reason: format!("robust scenario {} ({}) failed", failed.index, failed.label),
                },
                None,
                scenarios,
            );
        }
        match aggregate_robust_metrics(&scenarios) {
            Ok(metrics) => (EvaluationStatus::Completed, Some(metrics), scenarios),
            Err(error) => (
                EvaluationStatus::SimulationFailed {
                    reason: error.to_string(),
                },
                None,
                scenarios,
            ),
        }
    }

    fn telemetry(&self, elapsed_seconds: f64) -> OptimizationTelemetry {
        let elapsed_hours = (elapsed_seconds / 3_600.0).max(f64::MIN_POSITIVE);
        let mut failed_evaluations = 0;
        let mut budget_exhausted_evaluations = 0;
        for record in &self.records {
            match record.status {
                EvaluationStatus::InvalidGeometry { .. }
                | EvaluationStatus::SimulationFailed { .. } => failed_evaluations += 1,
                EvaluationStatus::BudgetExhausted { .. } => budget_exhausted_evaluations += 1,
                EvaluationStatus::Completed => {}
            }
        }
        OptimizationTelemetry {
            candidate_evaluations: self.candidate_evaluations,
            scenario_evaluations: self.scenario_evaluations,
            frequency_evaluations: self.frequency_evaluations,
            rhs_evaluations: self.rhs_evaluations,
            exact_residual_evaluations: self.exact_residual_evaluations,
            gradient_evaluations: self.gradient_evaluations,
            accepted_gradient_steps: self.accepted_gradient_steps,
            failed_evaluations,
            budget_exhausted_evaluations,
            candidates_per_hour: self.candidate_evaluations as f64 / elapsed_hours,
            scenario_evaluations_per_hour: self.scenario_evaluations as f64 / elapsed_hours,
            frequency_evaluations_per_hour: self.frequency_evaluations as f64 / elapsed_hours,
            rhs_evaluations_per_hour: self.rhs_evaluations as f64 / elapsed_hours,
            exact_residual_evaluations_per_hour: self.exact_residual_evaluations as f64
                / elapsed_hours,
            exact_residual_cost_units: self.exact_residual_evaluations,
            accepted_gradients_per_hour: self.accepted_gradient_steps as f64 / elapsed_hours,
            final_l3_validated_preference_improvement: None,
            product_taylor_error: None,
        }
    }
}

/// Execute a versioned staged plan and persist a resumable history.
pub fn optimize_with_plan(
    model: &LoudspeakerModel,
    plan: &OptimizationPlanV1,
    output_dir: &Path,
    resume: bool,
) -> Result<StagedOptimizationResult, OptimizationError> {
    let bounds = ParameterBounds::from_model_for_design_space(model, plan.design_space)?;
    let require_reference_sweep = plan
        .acceptance
        .product_limits
        .as_ref()
        .is_some_and(product_limits_require_reference_evidence);
    let evaluator = ProductionEvaluator::new(model, &bounds, require_reference_sweep)?;
    optimize_with_evaluator(
        model,
        plan,
        output_dir,
        resume,
        Box::new(evaluator),
        true,
        true,
    )
}

fn optimize_with_evaluator(
    model: &LoudspeakerModel,
    plan: &OptimizationPlanV1,
    output_dir: &Path,
    resume: bool,
    evaluator: Box<dyn CandidateEvaluator>,
    write_cea: bool,
    use_adjoint: bool,
) -> Result<StagedOptimizationResult, OptimizationError> {
    let total_started = Instant::now();
    plan.validate()?;
    let uncertainty_cases = plan
        .robust_uncertainty
        .as_ref()
        .map_or_else(|| Ok(Vec::new()), RobustUncertaintyConfig::cases)?;
    let bounds = ParameterBounds::from_model_for_design_space(model, plan.design_space)?;
    let product_taylor_error = if plan.design_space == DesignSpace::Product {
        let product = model.product_design.as_ref().ok_or_else(|| {
            OptimizationError::new("product derivative check requires product_design")
        })?;
        Some(product_design_taylor_error(product)?)
    } else {
        None
    };
    fs::create_dir_all(output_dir).map_err(io_error)?;
    let evaluator_identity = evaluator.cache_identity();
    if evaluator_identity.trim().is_empty() {
        return Err(OptimizationError::new(
            "candidate evaluator returned an empty cache identity",
        ));
    }
    let cache_identity = format!(
        "cache-schema-v{CACHE_IDENTITY_SCHEMA_VERSION};conventions=speaker-conventions-v1;{evaluator_identity}"
    );
    let identity = CheckpointIdentity {
        model_identity: stable_json_identity(model)?,
        plan_identity: stable_json_identity(plan)?,
        cache_identity: cache_identity.clone(),
    };
    prepare_checkpoint(output_dir, plan, &identity, resume)?;
    let records = if resume {
        load_records(output_dir, &identity)?
    } else {
        Vec::new()
    };
    let candidate_evaluations = records
        .iter()
        .filter(|record| !matches!(&record.status, EvaluationStatus::BudgetExhausted { .. }))
        .count();
    let scenario_evaluations = records
        .iter()
        .flat_map(|record| record.scenario_results.iter())
        .filter(|scenario| !matches!(&scenario.status, EvaluationStatus::BudgetExhausted { .. }))
        .count();
    let frequency_evaluations = records
        .iter()
        .map(|record| record.frequency_evaluations)
        .sum();
    let rhs_evaluations = records.iter().map(|record| record.rhs_evaluations).sum();
    let exact_residual_evaluations = records
        .iter()
        .map(|record| record.exact_residual_evaluations)
        .sum();
    let cache = records
        .iter()
        .cloned()
        .map(|record| {
            (
                evaluation_key(
                    plan.design_space,
                    &record.stage,
                    record.mesh_elements_per_wavelength,
                    &record.normalized_parameters,
                    &record.cache_identity,
                    &record.solver_identity,
                ),
                record,
            )
        })
        .collect();
    let mut context = RunContext {
        base: model,
        bounds,
        output: output_dir,
        identity: identity.clone(),
        records,
        cache,
        evaluator,
        uncertainty_cases,
        resource_budget: plan.resource_budget,
        candidate_evaluations,
        scenario_evaluations,
        frequency_evaluations,
        rhs_evaluations,
        exact_residual_evaluations,
        gradient_evaluations: 0,
        accepted_gradient_steps: 0,
    };

    let initial = context.bounds.initial_normalized().to_vec();
    let parameter_count = initial.len();
    let run_adjoint = use_adjoint && plan.design_space == DesignSpace::WaveguideFive;
    let coarse_solver = SolverConfig {
        execution_profile: ExecutionProfile::Preview,
        parallel_frequency_sweep: true,
        ..Default::default()
    };
    let mut coarse_points = vec![initial.clone()];
    // Seed enough of the bounded space to retain four distinct candidates
    // even when the local adjoint path converges immediately.
    for seed in 0..4 {
        let mut point = initial.clone();
        for (dimension, value) in point.iter_mut().enumerate() {
            let phase = ((seed + 1) * (dimension * 2 + 3)) % 11;
            *value = 0.15 + 0.70 * phase as f64 / 10.0;
        }
        coarse_points.push(point);
    }
    let mut adjoint_model = model.clone();
    adjoint_model.simulation.freq_start = plan.coarse.resolution.frequency_start_hz;
    adjoint_model.simulation.freq_end = plan.coarse.resolution.frequency_end_hz;
    adjoint_model.simulation.freq_points = plan.coarse.resolution.frequency_points;
    adjoint_model.simulation.max_mesh_frequency = Some(plan.coarse.resolution.frequency_end_hz);
    let adjoint_eval_points = cea2034_output::generate_eval_points(adjoint_model.baffle_center());
    let adjoint_config = OptimizationConfig {
        max_iterations: plan.coarse.max_iterations,
        weight_flat: 1.0,
        weight_pir: 1.0,
        optim_freq_points: plan.coarse.resolution.frequency_points,
        optim_max_mesh_freq: plan.coarse.resolution.frequency_end_hz,
        gradient_method: OptimizationGradientMethod::DiscreteAdjoint,
        max_candidate_lanes: None,
        adjoint_max_dofs: plan.coarse.adjoint_max_dofs,
        adjoint_hmatrix: true,
        exact_residual_policy: super::ExactResidualPolicy::FinalOnly,
    };
    let adjoint_bounds = (0..parameter_count)
        .map(|_| Bound::both(0.0, 1.0))
        .collect::<Vec<_>>();
    let mut adjoint_point = initial.clone();
    for iteration in 0..plan.coarse.max_iterations {
        let gradient = if run_adjoint {
            discrete_adjoint_gradient_for_parameterization(
                &adjoint_point,
                &adjoint_model,
                &adjoint_eval_points,
                &coarse_solver,
                &adjoint_config,
                &adjoint_bounds,
                |parameters, base| context.bounds.apply(parameters, base),
            )
        } else {
            Err(OptimizationError::new("adjoint disabled by test evaluator"))
        };
        context.gradient_evaluations += 1;
        match gradient {
            Ok(mut gradient) => {
                context.accepted_gradient_steps += 1;
                for (index, value) in gradient.iter_mut().enumerate() {
                    *value += 0.04
                        * (adjoint_point[index] - context.bounds.initial_normalized()[index])
                        / parameter_count as f64;
                }
                let scale = gradient
                    .iter()
                    .map(|value| value.abs())
                    .fold(0.0_f64, f64::max)
                    .max(1.0e-12);
                for index in 0..parameter_count {
                    adjoint_point[index] =
                        (adjoint_point[index] - 0.12 * gradient[index] / scale).clamp(0.0, 1.0);
                }
            }
            Err(error) => {
                log::warn!(
                    "coarse waveguide adjoint iteration {} failed ({error}); using deterministic bounded fallback",
                    iteration + 1
                );
                for (dimension, value) in adjoint_point.iter_mut().enumerate() {
                    let phase = ((iteration + 1) * (dimension * 2 + 3)) % 17;
                    *value = 0.15 + 0.70 * phase as f64 / 16.0;
                }
            }
        }
        coarse_points.push(adjoint_point.clone());
    }
    deduplicate_points(&mut coarse_points);
    let mut coarse_ranked = coarse_points
        .into_iter()
        .map(|point| {
            let metrics =
                context.evaluate("coarse", &point, &plan.coarse.resolution, &coarse_solver);
            (point, metrics)
        })
        .collect::<Vec<_>>();
    coarse_ranked.sort_by(rank_candidates);
    let mut shortlist: Vec<(Vec<f64>, CandidateMetrics)> = Vec::new();
    for (point, metrics) in &coarse_ranked {
        if shortlist
            .iter()
            .all(|(chosen, _)| normalized_distance(point, chosen) >= plan.coarse.distinct_distance)
        {
            shortlist.push((point.clone(), metrics.clone()));
        }
        if shortlist.len() == plan.coarse.shortlist_size {
            break;
        }
    }
    if !shortlist.iter().any(|(point, _)| *point == initial) {
        let metrics = context.evaluate("coarse", &initial, &plan.coarse.resolution, &coarse_solver);
        shortlist.push((initial.clone(), metrics));
    }

    let mut best = shortlist[0].clone();
    if let Some(refinement) = &plan.refinement {
        let refine_solver = SolverConfig {
            execution_profile: ExecutionProfile::Preview,
            tolerance: refinement.fgmres_tolerance,
            hmatrix_aca_tolerance: refinement.aca_tolerance,
            direct_threshold: 1,
            hmatrix_threshold: 2,
            parallel_frequency_sweep: true,
            ..Default::default()
        };
        let mut evaluated = 0usize;
        let mut seeds: Vec<(Vec<f64>, CandidateMetrics)> = Vec::new();
        for (point, _) in &shortlist {
            if evaluated >= refinement.max_evaluations {
                break;
            }
            let metrics =
                context.evaluate("refinement", point, &refinement.resolution, &refine_solver);
            seeds.push((point.clone(), metrics));
            evaluated += 1;
        }
        seeds.sort_by(rank_candidates);
        let center = seeds[0].0.clone();
        let mut simplex = vec![seeds[0].clone()];
        for dimension in 0..parameter_count {
            if evaluated >= refinement.max_evaluations {
                break;
            }
            let mut point = center.clone();
            point[dimension] = if point[dimension] <= 0.85 {
                point[dimension] + 0.15
            } else {
                point[dimension] - 0.15
            };
            let metrics =
                context.evaluate("refinement", &point, &refinement.resolution, &refine_solver);
            simplex.push((point, metrics));
            evaluated += 1;
        }
        while simplex.len() == parameter_count + 1 && evaluated < refinement.max_evaluations {
            simplex.sort_by(|a, b| a.1.smooth_objective.total_cmp(&b.1.smooth_objective));
            let centroid = simplex[..parameter_count].iter().fold(
                vec![0.0; parameter_count],
                |mut sum, (point, _)| {
                    for index in 0..parameter_count {
                        sum[index] += point[index] / parameter_count as f64;
                    }
                    sum
                },
            );
            let worst = &simplex[parameter_count].0;
            let reflected = bounded_affine(&centroid, worst, 1.0);
            let reflected_metrics = context.evaluate(
                "refinement",
                &reflected,
                &refinement.resolution,
                &refine_solver,
            );
            evaluated += 1;
            let replacement = if reflected_metrics.smooth_objective < simplex[0].1.smooth_objective
                && evaluated < refinement.max_evaluations
            {
                let expanded = bounded_affine(&centroid, worst, 2.0);
                let expanded_metrics = context.evaluate(
                    "refinement",
                    &expanded,
                    &refinement.resolution,
                    &refine_solver,
                );
                evaluated += 1;
                if expanded_metrics.smooth_objective < reflected_metrics.smooth_objective {
                    (expanded, expanded_metrics)
                } else {
                    (reflected, reflected_metrics)
                }
            } else if reflected_metrics.smooth_objective
                < simplex[parameter_count - 1].1.smooth_objective
            {
                (reflected, reflected_metrics)
            } else if evaluated < refinement.max_evaluations {
                let contracted = bounded_affine(&centroid, worst, -0.5);
                let contracted_metrics = context.evaluate(
                    "refinement",
                    &contracted,
                    &refinement.resolution,
                    &refine_solver,
                );
                evaluated += 1;
                (contracted, contracted_metrics)
            } else {
                (reflected, reflected_metrics)
            };
            simplex[parameter_count] = replacement;
        }
        simplex.extend(seeds);
        simplex.sort_by(rank_candidates);
        best = simplex[0].clone();
    }

    let best_candidate = context.bounds.apply(&best.0, model)?;
    write_json(
        &output_dir.join("best_candidate_model.json"),
        &best_candidate,
    )?;
    let preview_only = plan.refinement.is_none() || plan.validation.is_none();
    let mut accepted = false;
    let mut validation_diagnostics = None;
    if let Some(validation) = &plan.validation {
        let production_resolution = StageResolution {
            frequency_start_hz: validation.frequency_start_hz,
            frequency_end_hz: validation.frequency_end_hz,
            frequency_points: validation.frequency_points,
            elements_per_wavelength: validation.production_elements_per_wavelength,
        };
        let convergence_resolution = StageResolution {
            elements_per_wavelength: validation.convergence_elements_per_wavelength,
            ..production_resolution.clone()
        };
        let production_solver = SolverConfig {
            execution_profile: ExecutionProfile::Golden,
            parallel_frequency_sweep: true,
            ..Default::default()
        };
        let baseline_metrics = context.evaluate(
            "validation_baseline",
            &initial,
            &production_resolution,
            &production_solver,
        );
        let candidate_metrics = context.evaluate(
            "validation_candidate",
            &best.0,
            &production_resolution,
            &production_solver,
        );
        let convergence_metrics = context.evaluate(
            "validation_convergence",
            &best.0,
            &convergence_resolution,
            &production_solver,
        );
        let diagnostics = acceptance_diagnostics(
            &baseline_metrics,
            &candidate_metrics,
            &convergence_metrics,
            &plan.acceptance,
        );
        accepted = diagnostics.accepted();
        validation_diagnostics = Some(diagnostics);
        best.1 = candidate_metrics;
        if accepted {
            write_json(&output_dir.join("optimised_model.json"), &best_candidate)?;
            if write_cea {
                write_validation_cea(
                    output_dir,
                    model,
                    &best_candidate,
                    &production_resolution,
                    &production_solver,
                    plan.design_space,
                    plan.acceptance
                        .product_limits
                        .as_ref()
                        .is_some_and(product_limits_require_reference_evidence),
                )?;
            }
        }
    }
    let elapsed_seconds = total_started.elapsed().as_secs_f64();
    let mut telemetry = context.telemetry(elapsed_seconds);
    telemetry.final_l3_validated_preference_improvement = validation_diagnostics
        .as_ref()
        .map(|diagnostics| diagnostics.preference_score_improvement);
    telemetry.product_taylor_error = product_taylor_error;
    let objective_history = context
        .records
        .iter()
        .filter_map(|record| {
            record
                .metrics
                .as_ref()
                .map(|metrics| metrics.smooth_objective)
        })
        .collect::<Vec<_>>();
    let summary = StagedOptimizationSummary {
        plan_version: PLAN_VERSION,
        band_label: BAND_LABEL.to_string(),
        model_identity: identity.model_identity,
        plan_identity: identity.plan_identity,
        resumed: resume,
        evaluations: context.records.len(),
        accepted,
        preview_only,
        best_normalized_parameters: best.0.clone(),
        best_metrics: best.1,
        validation: validation_diagnostics,
        elapsed_seconds,
        uncertainty_scenario_count: context.uncertainty_cases.len(),
        telemetry,
        objective_history,
    };
    write_json(&output_dir.join("optimization_summary.json"), &summary)?;
    Ok(StagedOptimizationResult {
        model: if accepted {
            best_candidate.clone()
        } else {
            model.clone()
        },
        best_candidate,
        summary,
    })
}

fn candidate_metrics(
    result: &SimulationResult,
    normalized: &[f64],
    initial: &[f64],
    product: Option<&ProductDesign>,
) -> Result<CandidateMetrics, OptimizationError> {
    let metrics = cea2034_output::compute_spinorama_metrics(result).map_err(|message| {
        OptimizationError::new(format!("spinorama scoring failed: {message}"))
    })?;
    let coupled_product_metrics = match product {
        Some(product) => crate::product_metrics::coupled_product_metrics_from_outputs_with_product(
            result,
            product,
            Some(metrics.preference_score),
            &[],
            &[],
            REFERENCE_PANEL_MODE_ORDER,
        ),
        None => crate::product_metrics::coupled_product_metrics_from_outputs(
            result,
            Some(metrics.preference_score),
            &[],
            &[],
        ),
    }
    .map_err(|error| OptimizationError::new(format!("product metrics failed: {error}")))?;
    metrics_from_curves(
        result,
        &metrics,
        normalized,
        initial,
        Some(coupled_product_metrics),
    )
}

fn product_limits_require_reference_evidence(limits: &CoupledProductLimits) -> bool {
    limits.maximum_reference_relative_residual.is_some()
        || limits.maximum_rom_reduced_residual.is_some()
        || limits.minimum_rom_rank_agreement.is_some()
        || limits.maximum_rom_gradient_relative_error.is_some()
        || limits.maximum_rom_constraint_violation.is_some()
}

fn apply_uncertainty_case(
    model: &LoudspeakerModel,
    case: &SpeakerUncertaintyCase,
) -> Result<LoudspeakerModel, OptimizationError> {
    case.validate()
        .map_err(|error| OptimizationError::new(error.to_string()))?;
    let mut scenario = model.clone();
    let mut drivers: Vec<&mut DriverConfig> =
        Vec::with_capacity(2 + scenario.additional_drivers.len());
    drivers.push(&mut scenario.woofer);
    drivers.push(&mut scenario.tweeter);
    drivers.extend(scenario.additional_drivers.iter_mut());
    for (index, driver) in drivers.into_iter().enumerate() {
        let nominal = driver.driver_model.as_ref().ok_or_else(|| {
            OptimizationError::new(format!(
                "robust uncertainty requires a typed driver model at stable index {index}"
            ))
        })?;
        driver.driver_model = Some(
            case.apply_to_driver_model(nominal)
                .map_err(|error| OptimizationError::new(error.to_string()))?,
        );
        driver.y_offset += case.assembly_offset_m;
    }
    let driver_count = scenario.drivers().len();
    let product_driver_updates = scenario
        .drivers()
        .iter()
        .map(|driver| {
            driver.driver_model.as_ref().map(|model| {
                (
                    model.thiele_small.force_factor_tm,
                    model.thiele_small.voice_coil_resistance_ohm,
                    model.thiele_small.mechanical_resistance_kg_per_s,
                )
            })
        })
        .collect::<Vec<_>>();
    if let Some(product) = scenario.product_design.as_mut() {
        if product.drivers.len() != driver_count {
            return Err(OptimizationError::new(
                "product uncertainty requires product and mesh driver counts to agree",
            ));
        }
        for (product_driver, model_update) in product.drivers.iter_mut().zip(product_driver_updates)
        {
            if let Some((force_factor, resistance, mechanical_resistance)) = model_update {
                // Keep the reduced product reference contract synchronized
                // with the same typed model used by the exterior solve.
                product_driver.force_factor_tm = force_factor;
                product_driver.coil_resistance_ohm = resistance;
                product_driver.mechanical_resistance_n_s_per_m = mechanical_resistance;
            }
            // ProductDriver stores a reference-frequency reactance rather
            // than a full Le(f)/Mms/Cms law; scale the directly represented
            // electrical inductive term until that richer schema is present.
            product_driver.coil_reactance_ohm *= case.inductance_scale;
            product_driver.position_m[1] += case.assembly_offset_m;
        }
    }
    scenario.validate().map_err(|error| {
        OptimizationError::new(format!(
            "uncertainty scenario produced invalid model: {error}"
        ))
    })?;
    Ok(scenario)
}

fn aggregate_robust_metrics(
    scenarios: &[ScenarioEvaluationRecord],
) -> Result<CandidateMetrics, OptimizationError> {
    let successful = scenarios
        .iter()
        .filter_map(|scenario| scenario.metrics.as_ref())
        .collect::<Vec<_>>();
    let first = successful
        .first()
        .ok_or_else(|| OptimizationError::new("robust evaluation has no successful scenarios"))?;
    let mut aggregate = (*first).clone();
    aggregate.smooth_objective = successful
        .iter()
        .map(|metrics| metrics.smooth_objective)
        .fold(f64::NEG_INFINITY, f64::max);
    aggregate.preference_score = successful
        .iter()
        .map(|metrics| metrics.preference_score)
        .fold(f64::INFINITY, f64::min);
    aggregate.on_axis_rms_db = successful
        .iter()
        .map(|metrics| metrics.on_axis_rms_db)
        .fold(f64::NEG_INFINITY, f64::max);
    aggregate.pir_rms_db = successful
        .iter()
        .map(|metrics| metrics.pir_rms_db)
        .fold(f64::NEG_INFINITY, f64::max);
    aggregate.maximum_pir_slope_db_per_octave = successful
        .iter()
        .map(|metrics| metrics.maximum_pir_slope_db_per_octave)
        .fold(f64::NEG_INFINITY, f64::max);
    if let Some(worst_curve) = successful
        .iter()
        .max_by(|left, right| left.on_axis_rms_db.total_cmp(&right.on_axis_rms_db))
    {
        aggregate.on_axis_db = worst_curve.on_axis_db.clone();
    }
    if let Some(worst_curve) = successful
        .iter()
        .max_by(|left, right| left.pir_rms_db.total_cmp(&right.pir_rms_db))
    {
        aggregate.pir_db = worst_curve.pir_db.clone();
    }
    aggregate.coupled_product_metrics = aggregate_product_metrics(
        successful
            .iter()
            .map(|metrics| metrics.coupled_product_metrics.as_ref())
            .collect(),
    );
    Ok(aggregate)
}

fn aggregate_product_metrics(
    metrics: Vec<Option<&CoupledProductMetrics>>,
) -> Option<CoupledProductMetrics> {
    let mut iter = metrics.into_iter();
    let first = (*iter.next().flatten()?).clone();
    let mut aggregate = first;
    for next in iter {
        let next = next?;
        aggregate.cea_preference_score =
            min_optional(aggregate.cea_preference_score, next.cea_preference_score);
        aggregate.minimum_impedance_real_ohm = min_optional(
            aggregate.minimum_impedance_real_ohm,
            next.minimum_impedance_real_ohm,
        );
        aggregate.maximum_reference_relative_residual = max_optional(
            aggregate.maximum_reference_relative_residual,
            next.maximum_reference_relative_residual,
        );
        aggregate.maximum_rom_reduced_residual = max_optional(
            aggregate.maximum_rom_reduced_residual,
            next.maximum_rom_reduced_residual,
        );
        aggregate.minimum_rom_rank_agreement = min_optional(
            aggregate.minimum_rom_rank_agreement,
            next.minimum_rom_rank_agreement,
        );
        aggregate.maximum_rom_gradient_relative_error = max_optional(
            aggregate.maximum_rom_gradient_relative_error,
            next.maximum_rom_gradient_relative_error,
        );
        aggregate.maximum_rom_constraint_violation = max_optional(
            aggregate.maximum_rom_constraint_violation,
            next.maximum_rom_constraint_violation,
        );
        aggregate.peak_excursion_m =
            max_optional(aggregate.peak_excursion_m, next.peak_excursion_m);
        aggregate.maximum_panel_displacement_m = max_optional(
            aggregate.maximum_panel_displacement_m,
            next.maximum_panel_displacement_m,
        );
        aggregate.maximum_panel_bending_stress_pa = max_optional(
            aggregate.maximum_panel_bending_stress_pa,
            next.maximum_panel_bending_stress_pa,
        );
        aggregate.maximum_distortion_fraction = max_optional(
            aggregate.maximum_distortion_fraction,
            next.maximum_distortion_fraction,
        );
        aggregate.thermal_compression_db = max_optional(
            aggregate.thermal_compression_db,
            next.thermal_compression_db,
        );
        aggregate.cost = max_optional(aggregate.cost, next.cost);
        aggregate.manufacturability_penalty = max_optional(
            aggregate.manufacturability_penalty,
            next.manufacturability_penalty,
        );
        for (marker, value) in &next.peak_stress_pa {
            aggregate
                .peak_stress_pa
                .entry(*marker)
                .and_modify(|current| *current = current.max(*value))
                .or_insert(*value);
        }
    }
    Some(aggregate)
}

fn min_optional(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    Some(left?.min(right?))
}

fn max_optional(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    Some(left?.max(right?))
}

fn metrics_from_curves(
    result: &SimulationResult,
    spinorama: &SpinoramaMetrics,
    normalized: &[f64],
    initial: &[f64],
    coupled_product_metrics: Option<CoupledProductMetrics>,
) -> Result<CandidateMetrics, OptimizationError> {
    let on_axis_rms_db = centered_rms(&spinorama.on_axis)?;
    let pir_rms_db = centered_rms(&spinorama.pir)?;
    let maximum_pir_slope_db_per_octave =
        maximum_octave_slope(&result.frequencies, &spinorama.pir)?;
    let slope_penalty = (maximum_pir_slope_db_per_octave - 2.0).max(0.0).powi(2);
    let distance = normalized
        .iter()
        .zip(initial)
        .map(|(value, initial)| (value - initial).powi(2))
        .sum::<f64>()
        / normalized.len().max(1) as f64;
    let smooth_objective =
        on_axis_rms_db.powi(2) + pir_rms_db.powi(2) + 10.0 * slope_penalty + 0.02 * distance;
    Ok(CandidateMetrics {
        smooth_objective,
        preference_score: spinorama.preference_score,
        on_axis_rms_db,
        pir_rms_db,
        maximum_pir_slope_db_per_octave,
        on_axis_db: spinorama.on_axis.clone(),
        pir_db: spinorama.pir.clone(),
        coupled_product_metrics,
    })
}

fn centered_rms(values: &[f64]) -> Result<f64, OptimizationError> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(OptimizationError::new(
            "flatness curve must be non-empty and finite",
        ));
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    Ok((values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt())
}

fn maximum_octave_slope(frequencies: &[f64], values: &[f64]) -> Result<f64, OptimizationError> {
    if frequencies.len() != values.len() || frequencies.len() < 2 {
        return Err(OptimizationError::new(
            "slope curves require matching frequency/value vectors",
        ));
    }
    let mut maximum: f64 = 0.0;
    for index in 1..frequencies.len() {
        let octaves = (frequencies[index] / frequencies[index - 1]).log2();
        if !octaves.is_finite() || octaves <= 0.0 {
            return Err(OptimizationError::new(
                "slope frequencies must be finite and strictly increasing",
            ));
        }
        maximum = maximum.max((values[index] - values[index - 1]).abs() / octaves);
    }
    Ok(maximum)
}

fn penalty_metrics(normalized: &[f64]) -> CandidateMetrics {
    let deterministic = normalized
        .iter()
        .enumerate()
        .map(|(index, value)| (index + 1) as f64 * value.abs())
        .sum::<f64>();
    CandidateMetrics {
        smooth_objective: OBJECTIVE_FAILURE_PENALTY + deterministic,
        preference_score: -OBJECTIVE_FAILURE_PENALTY - deterministic,
        on_axis_rms_db: OBJECTIVE_FAILURE_PENALTY.sqrt(),
        pir_rms_db: OBJECTIVE_FAILURE_PENALTY.sqrt(),
        maximum_pir_slope_db_per_octave: OBJECTIVE_FAILURE_PENALTY.sqrt(),
        on_axis_db: Vec::new(),
        pir_db: Vec::new(),
        coupled_product_metrics: None,
    }
}

fn acceptance_diagnostics(
    baseline: &CandidateMetrics,
    candidate: &CandidateMetrics,
    convergence: &CandidateMetrics,
    criteria: &AcceptanceCriteria,
) -> ValidationDiagnostics {
    let preference_score_improvement = candidate.preference_score - baseline.preference_score;
    let on_axis_rms_improvement_db = baseline.on_axis_rms_db - candidate.on_axis_rms_db;
    let pir_rms_improvement_db = baseline.pir_rms_db - candidate.pir_rms_db;
    let mesh_curve_rms_difference_db = combined_curve_rms(candidate, convergence);
    let mesh_score_difference = (candidate.preference_score - convergence.preference_score).abs();
    let (passed_product_constraints, product_constraint_minimum_margin) =
        match &criteria.product_limits {
            None => (true, None),
            Some(limits) => candidate
                .coupled_product_metrics
                .as_ref()
                .and_then(|metrics| CoupledProductConstraintReport::evaluate(metrics, limits).ok())
                .map_or((false, Some(f64::NEG_INFINITY)), |report| {
                    (report.feasible, report.minimum_margin)
                }),
        };
    ValidationDiagnostics {
        preference_score_improvement,
        on_axis_rms_improvement_db,
        pir_rms_improvement_db,
        mesh_curve_rms_difference_db,
        mesh_score_difference,
        passed_preference_score: preference_score_improvement
            >= criteria.minimum_preference_score_improvement,
        passed_on_axis_rms: on_axis_rms_improvement_db
            >= criteria.minimum_on_axis_rms_improvement_db,
        passed_pir_rms: pir_rms_improvement_db >= criteria.minimum_pir_rms_improvement_db,
        passed_mesh_curve: mesh_curve_rms_difference_db
            <= criteria.maximum_mesh_curve_rms_difference_db,
        passed_mesh_score: mesh_score_difference <= criteria.maximum_mesh_score_difference,
        product_constraint_minimum_margin,
        passed_product_constraints,
    }
}

fn combined_curve_rms(a: &CandidateMetrics, b: &CandidateMetrics) -> f64 {
    let pairs = a
        .on_axis_db
        .iter()
        .zip(&b.on_axis_db)
        .chain(a.pir_db.iter().zip(&b.pir_db))
        .collect::<Vec<_>>();
    if pairs.is_empty()
        || a.on_axis_db.len() != b.on_axis_db.len()
        || a.pir_db.len() != b.pir_db.len()
    {
        return f64::INFINITY;
    }
    (pairs
        .iter()
        .map(|(left, right)| (**left - **right).powi(2))
        .sum::<f64>()
        / pairs.len() as f64)
        .sqrt()
}

fn rank_candidates(
    left: &(Vec<f64>, CandidateMetrics),
    right: &(Vec<f64>, CandidateMetrics),
) -> std::cmp::Ordering {
    right
        .1
        .preference_score
        .total_cmp(&left.1.preference_score)
        .then_with(|| left.1.smooth_objective.total_cmp(&right.1.smooth_objective))
}

fn bounded_affine(centroid: &[f64], worst: &[f64], factor: f64) -> Vec<f64> {
    centroid
        .iter()
        .zip(worst)
        .map(|(center, worst)| (center + factor * (center - worst)).clamp(0.0, 1.0))
        .collect()
}

fn normalized_distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn deduplicate_points(points: &mut Vec<Vec<f64>>) {
    let mut unique = Vec::<Vec<f64>>::new();
    for point in points.drain(..) {
        if unique
            .iter()
            .all(|existing| normalized_distance(existing, &point) > 1.0e-12)
        {
            unique.push(point);
        }
    }
    *points = unique;
}

fn evaluation_key(
    design_space: DesignSpace,
    stage: &str,
    mesh_resolution: usize,
    normalized: &[f64],
    cache_identity: &str,
    solver_identity: &str,
) -> String {
    let parameters = normalized
        .iter()
        .map(|value| format!("{value:.12}"))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{cache_identity}:{solver_identity}:{design_space:?}:{stage}:{mesh_resolution}:{parameters}"
    )
}

/// Serialize every speaker-solver policy that can alter a candidate result.
/// This is intentionally explicit instead of relying on a `Debug` dump of
/// the whole configuration: adding a field to `SolverConfig` must trigger a
/// deliberate cache-schema review rather than silently changing identity.
fn solver_cache_identity(config: &SolverConfig) -> String {
    format!(
        "solver-schema-v1;backend=cpu-bem-v1;profile={:?};beta={:.17e};direct={};tol={:.17e};max_iter={};restart={};ilu_shift={:.17e};recycle_octaves={:.17e};recycle_size={};hmatrix_threshold={};aca_tol={:.17e};verify_exact={};sentinel={:?};parallel={};max_lanes={:?};memory_fraction={:.17e}",
        config.execution_profile,
        config.beta_scale,
        config.direct_threshold,
        config.tolerance,
        config.max_iterations,
        config.restart,
        config.ilu_shift,
        config.max_recycled_guess_octaves,
        config.recycle_subspace_size,
        config.hmatrix_threshold,
        config.hmatrix_aca_tolerance,
        config.verify_exact_hmatrix_residual,
        config.exact_residual_sentinel,
        config.parallel_frequency_sweep,
        config.max_frequency_lanes,
        config.frequency_sweep_memory_fraction,
    )
}

fn stable_json_identity<T: Serialize>(value: &T) -> Result<String, OptimizationError> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        OptimizationError::new(format!("identity serialization failed: {error}"))
    })?;
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    Ok(format!("fnv1a64:{hash:016x}"))
}

fn prepare_checkpoint(
    output: &Path,
    plan: &OptimizationPlanV1,
    identity: &CheckpointIdentity,
    resume: bool,
) -> Result<(), OptimizationError> {
    let checkpoint_path = output.join("optimization_checkpoint.json");
    if resume {
        let saved: CheckpointIdentity =
            serde_json::from_slice(&fs::read(&checkpoint_path).map_err(|error| {
                OptimizationError::new(format!(
                    "resume requires {}: {error}",
                    checkpoint_path.display()
                ))
            })?)
            .map_err(|error| {
                OptimizationError::new(format!("invalid checkpoint identity: {error}"))
            })?;
        if saved.model_identity != identity.model_identity
            || saved.plan_identity != identity.plan_identity
            || saved.cache_identity != identity.cache_identity
        {
            return Err(OptimizationError::new(
                "resume checkpoint model/config/cache identity does not match exactly",
            ));
        }
    } else {
        for name in [
            "optimization_summary.json",
            "best_candidate_model.json",
            "optimised_model.json",
        ] {
            let stale = output.join(name);
            if stale.exists() {
                fs::remove_file(stale).map_err(io_error)?;
            }
        }
        for name in ["baseline", "final"] {
            let stale = output.join(name);
            if stale.exists() {
                fs::remove_dir_all(stale).map_err(io_error)?;
            }
        }
        write_json(&checkpoint_path, identity)?;
        write_json(&output.join("optimization_plan.json"), plan)?;
        let history = output.join("optimization_history.jsonl");
        if history.exists() {
            fs::remove_file(&history).map_err(io_error)?;
        }
    }
    Ok(())
}

fn append_record(output: &Path, record: &EvaluationRecord) -> Result<(), OptimizationError> {
    let path = output.join("optimization_history.jsonl");
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(io_error)?;
    serde_json::to_writer(&mut file, record).map_err(|error| {
        OptimizationError::new(format!("history serialization failed: {error}"))
    })?;
    file.write_all(b"\n").map_err(io_error)?;
    file.sync_all().map_err(io_error)
}

fn load_records(
    output: &Path,
    identity: &CheckpointIdentity,
) -> Result<Vec<EvaluationRecord>, OptimizationError> {
    let path = output.join("optimization_history.jsonl");
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = fs::File::open(path).map_err(io_error)?;
    let mut records = Vec::new();
    for (line_index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(io_error)?;
        if line.trim().is_empty() {
            continue;
        }
        let record: EvaluationRecord = serde_json::from_str(&line).map_err(|error| {
            OptimizationError::new(format!(
                "invalid optimization history at line {}: {error}",
                line_index + 1
            ))
        })?;
        if record.model_identity != identity.model_identity
            || record.plan_identity != identity.plan_identity
            || record.cache_identity != identity.cache_identity
            || record.cache_identity.is_empty()
            || record.solver_identity.is_empty()
        {
            return Err(OptimizationError::new(
                "resume history contains a mismatched or legacy model/config/cache identity",
            ));
        }
        records.push(record);
    }
    Ok(records)
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), OptimizationError> {
    let data = serde_json::to_vec_pretty(value)
        .map_err(|error| OptimizationError::new(format!("JSON serialization failed: {error}")))?;
    fs::write(path, data).map_err(io_error)
}

fn write_validation_cea(
    output: &Path,
    baseline: &LoudspeakerModel,
    candidate: &LoudspeakerModel,
    resolution: &StageResolution,
    solver: &SolverConfig,
    design_space: DesignSpace,
    require_reference_sweep: bool,
) -> Result<(), OptimizationError> {
    let bounds = ParameterBounds::from_model_for_design_space(baseline, design_space)?;
    let mut evaluator = ProductionEvaluator::new(baseline, &bounds, require_reference_sweep)?;
    let baseline_result = evaluator.evaluate(baseline, resolution, solver)?;
    let candidate_result = evaluator.evaluate(candidate, resolution, solver)?;
    cea2034_output::write_cea2034_files(&output.join("baseline"), &baseline_result)
        .map_err(io_error)?;
    cea2034_output::write_cea2034_files(&output.join("final"), &candidate_result)
        .map_err(io_error)?;
    Ok(())
}

fn io_error(error: std::io::Error) -> OptimizationError {
    OptimizationError::new(format!("optimization artifact I/O failed: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        CabinetDimensions, DriverConfig, SeparatorConfig, SimulationConfig,
        SpeakerMaterialAssignment, TransferPoint, WaveguideConfig,
    };
    use sonium_sim_speaker_modeling::{
        CabinetDesign, CrossoverBranch, CrossoverNetwork, DriverModel, ProductDesign,
        ProductDriver, SpeakerDriverStudy, SpeakerUncertaintyCase,
    };
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn transfer() -> Option<Vec<TransferPoint>> {
        Some(vec![
            TransferPoint {
                freq: 100.0,
                magnitude: 0.001,
                phase_deg: 0.0,
            },
            TransferPoint {
                freq: 1_000.0,
                magnitude: 0.001,
                phase_deg: 0.0,
            },
            TransferPoint {
                freq: 6_000.0,
                magnitude: 0.001,
                phase_deg: 0.0,
            },
        ])
    }

    fn sample_model() -> LoudspeakerModel {
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
                transfer_function: transfer(),
                driver_model: None,
                cone: None,
            },
            tweeter: DriverConfig {
                radius: 0.013,
                x_offset: 0.0,
                y_offset: 0.10,
                piston_velocity: 0.001,
                transfer_function: transfer(),
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
                freq_start: 200.0,
                freq_end: 5_000.0,
                freq_points: 10,
                speed_of_sound: 343.0,
                air_density: 1.21,
                max_mesh_frequency: Some(5_000.0),
            },
        }
    }

    #[test]
    fn uncertainty_case_applies_to_all_stable_driver_indices() {
        let mut model = sample_model();
        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        let typed = DriverModel::from_study(&study).unwrap();
        for driver in [&mut model.woofer, &mut model.tweeter] {
            driver.radius = study.geometry.diaphragm_radius_m;
            driver.transfer_function = None;
            driver.driver_model = Some(typed.clone());
        }
        if let Some(waveguide) = model.waveguide.as_mut() {
            waveguide.horizontal_spline.last_mut().unwrap()[1] = study.geometry.diaphragm_radius_m;
            waveguide.vertical_spline.last_mut().unwrap()[1] = study.geometry.diaphragm_radius_m;
        }
        model.additional_drivers.push(DriverConfig {
            radius: study.geometry.diaphragm_radius_m.min(0.010),
            x_offset: 0.08,
            y_offset: 0.10,
            piston_velocity: 0.001,
            transfer_function: None,
            driver_model: Some(typed),
            cone: None,
        });
        let case = SpeakerUncertaintyCase {
            index: 0,
            label: "three-driver-corner".to_owned(),
            force_factor_scale: 1.02,
            inductance_scale: 0.98,
            moving_mass_scale: 1.01,
            compliance_scale: 0.99,
            resistance_scale: 1.03,
            material_scale: 1.02,
            temperature_offset_c: 10.0,
            assembly_offset_m: 0.002,
            measurement_offset_db: 0.0,
        };
        let original_offsets = model
            .drivers()
            .iter()
            .map(|driver| driver.y_offset)
            .collect::<Vec<_>>();
        let scenario = apply_uncertainty_case(&model, &case).unwrap();
        assert_eq!(scenario.drivers().len(), 3);
        for (before, after) in original_offsets.iter().zip(scenario.drivers()) {
            assert!((after.y_offset - before - case.assembly_offset_m).abs() < 1.0e-12);
        }
        assert!((scenario.additional_drivers[0].y_offset - 0.102).abs() < 1.0e-12);
    }

    struct FakeEvaluator {
        calls: Arc<AtomicUsize>,
        fail_call: Option<usize>,
    }

    impl CandidateEvaluator for FakeEvaluator {
        fn cache_identity(&self) -> String {
            "fake-evaluator:v1;backend=synthetic;coupling=none;rom=none".to_string()
        }

        // Fixture fills the deprecated per-driver vectors alongside the matrix.
        #[allow(deprecated)]
        fn evaluate(
            &mut self,
            model: &LoudspeakerModel,
            resolution: &StageResolution,
            _solver: &SolverConfig,
        ) -> Result<SimulationResult, OptimizationError> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail_call == Some(call) {
                return Err(OptimizationError::new("synthetic simulation failure"));
            }
            let frequencies = (0..resolution.frequency_points)
                .map(|index| {
                    let fraction = if resolution.frequency_points <= 1 {
                        0.0
                    } else {
                        index as f64 / (resolution.frequency_points - 1) as f64
                    };
                    (resolution.frequency_start_hz.ln()
                        + fraction
                            * (resolution.frequency_end_hz.ln()
                                - resolution.frequency_start_hz.ln()))
                    .exp()
                })
                .collect::<Vec<_>>();
            let rim = model.waveguide.as_ref().unwrap().horizontal_spline[0][1];
            let curvature = 20.0 * (rim - 0.05).abs();
            let spl = sonium_compute::HostMatrix::from_fn(
                resolution.frequency_points,
                cea2034_output::TOTAL_EVAL_POINTS,
                |frequency, point| {
                    let centered =
                        frequency as f64 - (resolution.frequency_points - 1) as f64 * 0.5;
                    80.0 + curvature * centered.powi(2) / resolution.frequency_points as f64
                        - 0.01 * (point % cea2034_output::ANGLES_PER_PLANE) as f64
                },
            );
            Ok(SimulationResult {
                frequencies,
                phase: sonium_compute::HostMatrix::zeros(
                    resolution.frequency_points,
                    cea2034_output::TOTAL_EVAL_POINTS,
                ),
                complex_pressure: sonium_compute::HostMatrix::zeros(
                    resolution.frequency_points,
                    cea2034_output::TOTAL_EVAL_POINTS,
                ),
                spl,
                woofer_electrical_impedance_ohm: vec![None; resolution.frequency_points],
                tweeter_electrical_impedance_ohm: vec![None; resolution.frequency_points],
                driver_electrical_impedances_ohm: vec![vec![None; 2]; resolution.frequency_points],
                radiation_matrices: vec![None; resolution.frequency_points],
                radiation_matrix_diagnostics: vec![None; resolution.frequency_points],
                thermal_driver_states: vec![[None, None]; resolution.frequency_points],
                driver_thermal_states: vec![vec![None; 2]; resolution.frequency_points],
                power_ledgers: vec![None; resolution.frequency_points],
                product_reference_sweep: None,
                structural_responses: Vec::new(),
                driver_velocities_bem: vec![vec![]; resolution.frequency_points],
                exact_residual_evaluations: 0,
                provenance: simulation::SimulationProvenance::synthetic(
                    resolution.frequency_points,
                    cea2034_output::TOTAL_EVAL_POINTS,
                ),
            })
        }
    }

    #[test]
    fn plan_round_trip_and_presets_match_contract() {
        let balanced = OptimizationPlanV1::waveguide_balanced();
        let json = serde_json::to_string(&balanced).unwrap();
        let decoded: OptimizationPlanV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, balanced);
        assert_eq!(decoded.coarse.resolution.frequency_points, 10);
        assert_eq!(decoded.refinement.as_ref().unwrap().max_evaluations, 24);
        assert_eq!(decoded.validation.as_ref().unwrap().frequency_points, 40);
        assert!(OptimizationPlanV1::waveguide_preview().validation.is_none());
    }

    #[test]
    fn product_plan_declares_its_bounded_derivative_policy() {
        let product = OptimizationPlanV1::product_preview();
        assert_eq!(
            product.derivative_policy,
            DerivativePolicy::ProductBoundedDerivativeFree
        );
        product.validate().unwrap();
        let mut invalid = product;
        invalid.derivative_policy = DerivativePolicy::AdjointOrBoundedDerivativeFree;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn robust_plan_round_trip_and_case_budget_are_explicit() {
        let mut plan = OptimizationPlanV1::waveguide_preview();
        plan.robust_uncertainty = Some(RobustUncertaintyConfig {
            tolerance: SpeakerToleranceSpec::default(),
            sampling: UncertaintySampling::Corners,
            max_cases: 512,
        });
        plan.resource_budget = Some(OptimizationResourceBudget {
            max_candidate_evaluations: 64,
            max_scenario_evaluations: 512,
            max_frequency_evaluations: 10_000,
            max_rhs_evaluations: 20_000,
        });
        let json = serde_json::to_string(&plan).unwrap();
        let decoded: OptimizationPlanV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, plan);
        assert_eq!(decoded.robust_uncertainty.unwrap().max_cases, 512);

        plan.robust_uncertainty.as_mut().unwrap().max_cases = 511;
        assert!(plan.validate().is_err());
        plan.robust_uncertainty.as_mut().unwrap().max_cases = 512;
        plan.resource_budget
            .as_mut()
            .unwrap()
            .max_candidate_evaluations = 0;
        assert!(plan.validate().is_err());
        plan.resource_budget
            .as_mut()
            .unwrap()
            .max_candidate_evaluations = 64;
        plan.resource_budget
            .as_mut()
            .unwrap()
            .max_frequency_evaluations = 0;
        assert!(plan.validate().is_err());
    }

    #[test]
    fn robust_metrics_take_the_deterministic_worst_case() {
        let scenario = |index: usize,
                        label: &str,
                        smooth: f64,
                        preference: f64,
                        on_axis: f64,
                        pir: f64,
                        slope: f64| {
            ScenarioEvaluationRecord {
                index,
                label: label.to_owned(),
                status: EvaluationStatus::Completed,
                metrics: Some(CandidateMetrics {
                    smooth_objective: smooth,
                    preference_score: preference,
                    on_axis_rms_db: on_axis,
                    pir_rms_db: pir,
                    maximum_pir_slope_db_per_octave: slope,
                    on_axis_db: vec![on_axis],
                    pir_db: vec![pir],
                    coupled_product_metrics: None,
                }),
                exact_residual_evaluations: 0,
            }
        };
        let scenarios = vec![
            scenario(0, "nominal", 1.0, 5.0, 1.0, 1.5, 2.0),
            scenario(1, "case-0001", 3.0, 4.0, 2.0, 1.0, 4.0),
        ];
        let aggregate = aggregate_robust_metrics(&scenarios).unwrap();
        assert_eq!(aggregate.smooth_objective, 3.0);
        assert_eq!(aggregate.preference_score, 4.0);
        assert_eq!(aggregate.on_axis_rms_db, 2.0);
        assert_eq!(aggregate.pir_rms_db, 1.5);
        assert_eq!(aggregate.maximum_pir_slope_db_per_octave, 4.0);
        assert_eq!(aggregate.on_axis_db, vec![2.0]);
        assert_eq!(aggregate.pir_db, vec![1.5]);
    }

    #[test]
    fn waveguide_five_round_trip_keeps_throats_and_knot_positions_fixed() {
        let model = sample_model();
        let bounds =
            ParameterBounds::from_model_for_design_space(&model, DesignSpace::WaveguideFive)
                .unwrap();
        let normalized = bounds.initial_normalized();
        let rebuilt = bounds.apply(&normalized, &model).unwrap();
        let original = model.waveguide.unwrap();
        let rebuilt = rebuilt.waveguide.unwrap();
        assert_eq!(rebuilt.horizontal_spline, original.horizontal_spline);
        assert_eq!(rebuilt.vertical_spline, original.vertical_spline);
        assert_eq!(rebuilt.horizontal_spline[2][1], model.tweeter.radius);
        assert_eq!(rebuilt.vertical_spline[2][1], model.tweeter.radius);
        assert_eq!(
            rebuilt
                .horizontal_spline
                .iter()
                .map(|point| point[0])
                .collect::<Vec<_>>(),
            vec![0.0, 0.5, 1.0]
        );
    }

    #[test]
    fn waveguide_five_bounds_keep_natural_splines_monotonic() {
        let model = sample_model();
        let bounds =
            ParameterBounds::from_model_for_design_space(&model, DesignSpace::WaveguideFive)
                .unwrap();

        for normalized in [
            vec![0.0; WAVEGUIDE_PARAMETER_COUNT],
            vec![1.0; WAVEGUIDE_PARAMETER_COUNT],
        ] {
            bounds.apply(&normalized, &model).unwrap();
        }
    }

    #[test]
    fn product_design_bounds_apply_cabinet_and_all_driver_placements() {
        let mut model = sample_model();
        let cabinet = CabinetDesign {
            port: None,
            ..CabinetDesign::default()
        };
        let product = ProductDesign {
            cabinet,
            drivers: vec![
                ProductDriver {
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
                ProductDriver {
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
            crossover: CrossoverNetwork {
                branches: vec![CrossoverBranch::default(), CrossoverBranch::default()],
            },
            waveguide: None,
            structural_acoustic: None,
        };
        model.product_design = Some(product);
        let bounds =
            ParameterBounds::from_model_for_design_space(&model, DesignSpace::Product).unwrap();
        assert_eq!(bounds.parameter_count(), 15);
        let mut normalized = bounds.initial_normalized();
        normalized[5] = (normalized[5] + 0.1).min(1.0);
        let candidate = bounds.apply(&normalized, &model).unwrap();
        let product = candidate.product_design.as_ref().unwrap();
        assert_eq!(product.design_variables().len(), 15);
        assert_eq!(candidate.drivers().len(), 2);
        assert_eq!(candidate.woofer.x_offset, product.drivers[0].position_m[0]);
        assert_eq!(candidate.tweeter.y_offset, product.drivers[1].position_m[1]);
        assert_eq!(candidate.cabinet.width, product.cabinet.baffle_width_m);
    }

    #[test]
    fn pinned_mesh_connectivity_is_invariant_across_design_extremes() {
        let mut model = sample_model();
        model.simulation.freq_end = 700.0;
        model.simulation.freq_points = 2;
        model.simulation.max_mesh_frequency = Some(700.0);
        let bounds =
            ParameterBounds::from_model_for_design_space(&model, DesignSpace::WaveguideFive)
                .unwrap();
        let envelope = bounds.apply(&[1.0, 0.5, 1.0, 0.5, 1.0], &model).unwrap();
        let topology = PinnedWaveguideMesh::new(&model, &envelope).unwrap();
        let lower = bounds.apply(&[0.0, 0.15, 0.0, 0.15, 0.0], &model).unwrap();
        let upper = bounds.apply(&[1.0, 0.85, 1.0, 0.85, 1.0], &model).unwrap();
        let lower_mesh = topology.generate(&lower).unwrap();
        let upper_mesh = topology.generate(&upper).unwrap();
        assert_eq!(lower_mesh.triangles, upper_mesh.triangles);
        assert_eq!(lower_mesh.region, upper_mesh.region);
        assert_eq!(lower_mesh.nodes.len(), upper_mesh.nodes.len());
        assert!(
            lower_mesh
                .node_normals
                .iter()
                .chain(&upper_mesh.node_normals)
                .flatten()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn waveguide_mode_rejects_flat_or_out_of_band_driver_fallbacks() {
        let mut model = sample_model();
        model.tweeter.transfer_function = None;
        assert!(
            ParameterBounds::from_model_for_design_space(&model, DesignSpace::WaveguideFive)
                .is_err()
        );
        model.tweeter.transfer_function = Some(vec![
            TransferPoint {
                freq: 300.0,
                magnitude: 0.001,
                phase_deg: 0.0,
            },
            TransferPoint {
                freq: 4_000.0,
                magnitude: 0.001,
                phase_deg: 0.0,
            },
        ]);
        assert!(
            ParameterBounds::from_model_for_design_space(&model, DesignSpace::WaveguideFive)
                .is_err()
        );
    }

    // Fixture fills the deprecated per-driver vectors alongside the matrix.
    #[allow(deprecated)]
    #[test]
    fn objective_prefers_flatter_on_axis_and_pir_curves() {
        let frequencies = vec![1_500.0, 2_000.0, 3_000.0, 5_000.0];
        let result = SimulationResult {
            frequencies,
            spl: sonium_compute::HostMatrix::zeros(4, 72),
            phase: sonium_compute::HostMatrix::zeros(4, 72),
            complex_pressure: sonium_compute::HostMatrix::zeros(4, 72),
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
            provenance: simulation::SimulationProvenance::synthetic(4, 72),
        };
        let flat = SpinoramaMetrics {
            on_axis: vec![80.0, 80.1, 79.9, 80.0],
            listening_window: vec![],
            early_reflections: vec![],
            sound_power: vec![],
            pir: vec![78.0, 78.1, 77.9, 78.0],
            preference_score: 6.0,
        };
        let rough = SpinoramaMetrics {
            on_axis: vec![76.0, 82.0, 75.0, 84.0],
            pir: vec![74.0, 81.0, 73.0, 82.0],
            preference_score: 4.0,
            ..flat.clone()
        };
        let initial = [0.5; WAVEGUIDE_PARAMETER_COUNT];
        let normalized = vec![0.5; WAVEGUIDE_PARAMETER_COUNT];
        assert!(
            metrics_from_curves(&result, &flat, &normalized, &initial, None)
                .unwrap()
                .smooth_objective
                < metrics_from_curves(&result, &rough, &normalized, &initial, None)
                    .unwrap()
                    .smooth_objective
        );
    }

    #[test]
    fn exact_score_gate_rejects_a_flatness_only_regression() {
        let baseline = CandidateMetrics {
            smooth_objective: 2.0,
            preference_score: 5.0,
            on_axis_rms_db: 1.0,
            pir_rms_db: 1.0,
            maximum_pir_slope_db_per_octave: 1.0,
            on_axis_db: vec![80.0, 80.0],
            pir_db: vec![78.0, 78.0],
            coupled_product_metrics: None,
        };
        let candidate = CandidateMetrics {
            smooth_objective: 1.0,
            preference_score: 4.9,
            on_axis_rms_db: 0.8,
            pir_rms_db: 0.8,
            ..baseline.clone()
        };
        let diagnostics = acceptance_diagnostics(
            &baseline,
            &candidate,
            &candidate,
            &AcceptanceCriteria::default(),
        );
        assert!(!diagnostics.accepted());
        assert!(!diagnostics.passed_preference_score);
    }

    #[test]
    fn product_constraint_gate_rejects_a_candidate_below_the_configured_limit() {
        let metrics = sonium_sim_speaker_modeling::CoupledProductMetrics {
            cea_preference_score: Some(5.0),
            ..Default::default()
        };
        let baseline = CandidateMetrics {
            smooth_objective: 2.0,
            preference_score: 5.0,
            on_axis_rms_db: 1.0,
            pir_rms_db: 1.0,
            maximum_pir_slope_db_per_octave: 1.0,
            on_axis_db: vec![80.0, 80.0],
            pir_db: vec![78.0, 78.0],
            coupled_product_metrics: Some(metrics.clone()),
        };
        let candidate = CandidateMetrics {
            coupled_product_metrics: Some(metrics),
            ..baseline.clone()
        };
        let criteria = AcceptanceCriteria {
            product_limits: Some(sonium_sim_speaker_modeling::CoupledProductLimits {
                minimum_cea_preference_score: Some(6.0),
                ..Default::default()
            }),
            ..Default::default()
        };
        let diagnostics = acceptance_diagnostics(&baseline, &candidate, &candidate, &criteria);
        assert!(!diagnostics.passed_product_constraints);
        assert!(!diagnostics.accepted());
        assert!(diagnostics.product_constraint_minimum_margin.unwrap() < 0.0);
    }

    #[test]
    fn reference_constraint_gate_rejects_missing_candidate_evidence() {
        let metrics = CandidateMetrics {
            smooth_objective: 2.0,
            preference_score: 5.0,
            on_axis_rms_db: 1.0,
            pir_rms_db: 1.0,
            maximum_pir_slope_db_per_octave: 1.0,
            on_axis_db: vec![80.0, 80.0],
            pir_db: vec![78.0, 78.0],
            coupled_product_metrics: Some(Default::default()),
        };
        let criteria = AcceptanceCriteria {
            product_limits: Some(sonium_sim_speaker_modeling::CoupledProductLimits {
                maximum_reference_relative_residual: Some(1.0e-6),
                ..Default::default()
            }),
            ..Default::default()
        };
        let diagnostics = acceptance_diagnostics(&metrics, &metrics, &metrics, &criteria);
        assert!(!diagnostics.passed_product_constraints);
        assert!(
            diagnostics
                .product_constraint_minimum_margin
                .is_some_and(|margin| margin < 0.0)
        );
    }

    #[test]
    fn rom_limits_require_reference_evidence_generation() {
        let limits = CoupledProductLimits {
            minimum_rom_rank_agreement: Some(0.9),
            ..CoupledProductLimits::default()
        };
        assert!(product_limits_require_reference_evidence(&limits));
        assert!(!product_limits_require_reference_evidence(
            &CoupledProductLimits::default()
        ));
    }

    #[test]
    #[ignore = "production ROM validation; run via `just qa-nightly-regressions`"]
    fn production_reference_path_retains_rom_evidence_for_supported_fixture() {
        let mut model = sample_model();
        let cabinet = CabinetDesign {
            port: None,
            ..CabinetDesign::default()
        };
        model.product_design = Some(ProductDesign {
            cabinet,
            drivers: vec![ProductDriver {
                id: "reference-driver".to_owned(),
                radius_m: 0.025,
                position_m: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
                force_factor_tm: 2.0,
                coil_resistance_ohm: 4.0,
                coil_reactance_ohm: 0.5,
                mechanical_resistance_n_s_per_m: 0.3,
                mechanical_reactance_n_s_per_m: 1.0,
                crossover_branch: 0,
            }],
            crossover: CrossoverNetwork {
                branches: vec![CrossoverBranch::default()],
            },
            waveguide: None,
            structural_acoustic: None,
        });
        let evaluator = ProductionEvaluator {
            warm_start: SimulationWarmStart::default(),
            reference: model.clone(),
            envelope: model.clone(),
            design_space: DesignSpace::Product,
            topologies: HashMap::new(),
            require_reference_sweep: true,
        };
        model.simulation.freq_start = 80.0;
        model.simulation.freq_end = 80.0;
        model.simulation.freq_points = 1;
        let resolution = StageResolution {
            frequency_start_hz: 80.0,
            frequency_end_hz: 80.0,
            frequency_points: 1,
            elements_per_wavelength: 8,
        };
        let result = FakeEvaluator {
            calls: Arc::new(AtomicUsize::new(0)),
            fail_call: None,
        }
        .evaluate(&model, &resolution, &SolverConfig::default())
        .unwrap();
        let attached = evaluator
            .attach_reference_sweep(result, &model, &SolverConfig::default())
            .unwrap();
        let sweep = attached.product_reference_sweep.as_ref().unwrap();
        let rom = sweep.rom_promotion.as_ref().unwrap();
        assert_eq!(rom.point_count, 1);
        assert!(rom.maximum_checked_reduced_residual.is_some());
        let metrics = crate::product_metrics::coupled_product_metrics_from_outputs_with_product(
            &attached,
            model.product_design.as_ref().unwrap(),
            None,
            &[],
            &[],
            REFERENCE_PANEL_MODE_ORDER,
        )
        .unwrap();
        assert!(metrics.maximum_rom_reduced_residual.is_some());
    }

    fn fake_checkpoint_identity(plan: &OptimizationPlanV1) -> CheckpointIdentity {
        CheckpointIdentity {
            model_identity: stable_json_identity(&sample_model()).unwrap(),
            plan_identity: stable_json_identity(plan).unwrap(),
            cache_identity: "cache-schema-v1;conventions=speaker-conventions-v1;fake-evaluator:v1;backend=synthetic;coupling=none;rom=none".into(),
        }
    }

    #[test]
    fn checkpoint_identity_requires_exact_model_and_plan() {
        let root = std::env::temp_dir().join(format!(
            "sonium-waveguide-checkpoint-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let plan = OptimizationPlanV1::waveguide_preview();
        let identity = CheckpointIdentity {
            model_identity: stable_json_identity(&sample_model()).unwrap(),
            plan_identity: stable_json_identity(&plan).unwrap(),
            cache_identity: "test-cache-v1".into(),
        };
        prepare_checkpoint(&root, &plan, &identity, false).unwrap();
        let mismatch = CheckpointIdentity {
            model_identity: "different".into(),
            plan_identity: identity.plan_identity.clone(),
            cache_identity: identity.cache_identity.clone(),
        };
        assert!(prepare_checkpoint(&root, &plan, &mismatch, true).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn evaluation_cache_key_includes_solver_and_policy_identity() {
        let mut altered_solver = SolverConfig::default();
        altered_solver.tolerance *= 10.0;
        let normalized = [0.25, 0.75];
        let default_key = evaluation_key(
            DesignSpace::WaveguideFive,
            "coarse",
            8,
            &normalized,
            "cache-policy-a",
            &solver_cache_identity(&SolverConfig::default()),
        );
        let altered_solver_key = evaluation_key(
            DesignSpace::WaveguideFive,
            "coarse",
            8,
            &normalized,
            "cache-policy-a",
            &solver_cache_identity(&altered_solver),
        );
        let altered_policy_key = evaluation_key(
            DesignSpace::WaveguideFive,
            "coarse",
            8,
            &normalized,
            "cache-policy-b",
            &solver_cache_identity(&SolverConfig::default()),
        );
        assert_ne!(default_key, altered_solver_key);
        assert_ne!(default_key, altered_policy_key);
    }

    #[test]
    fn legacy_history_without_cache_identity_is_rejected() {
        let root = std::env::temp_dir().join(format!(
            "sonium-waveguide-legacy-cache-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let plan = OptimizationPlanV1::waveguide_preview();
        let identity = fake_checkpoint_identity(&plan);
        let legacy = serde_json::json!({
            "sequence": 0,
            "stage": "coarse",
            "mesh_elements_per_wavelength": 8,
            "normalized_parameters": [0.5, 0.5, 0.5, 0.5, 0.5],
            "physical_parameters": [],
            "status": "completed",
            "metrics": null,
            "elapsed_seconds": 0.0,
            "model_identity": identity.model_identity,
            "plan_identity": identity.plan_identity
        });
        fs::write(
            root.join("optimization_history.jsonl"),
            format!("{}\n", serde_json::to_string(&legacy).unwrap()),
        )
        .unwrap();
        assert!(load_records(&root, &identity).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn fake_orchestration_respects_budget_caches_failures_and_resumes() {
        let root = std::env::temp_dir().join(format!(
            "sonium-waveguide-orchestration-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let mut plan = OptimizationPlanV1::waveguide_preview();
        plan.coarse.max_iterations = 2;
        plan.coarse.shortlist_size = 2;
        let calls = Arc::new(AtomicUsize::new(0));
        let first = optimize_with_evaluator(
            &sample_model(),
            &plan,
            &root,
            false,
            Box::new(FakeEvaluator {
                calls: Arc::clone(&calls),
                fail_call: Some(1),
            }),
            false,
            false,
        )
        .unwrap();
        assert!(first.summary.preview_only);
        assert!(!first.summary.accepted);
        assert!(first.summary.evaluations <= 1 + 4 + plan.coarse.max_iterations);
        assert!(first.summary.best_metrics.smooth_objective.is_finite());
        let history = load_records(&root, &fake_checkpoint_identity(&plan)).unwrap();
        assert!(
            history
                .iter()
                .any(|record| matches!(record.status, EvaluationStatus::SimulationFailed { .. }))
        );

        let resumed_calls = Arc::new(AtomicUsize::new(0));
        let resumed = optimize_with_evaluator(
            &sample_model(),
            &plan,
            &root,
            true,
            Box::new(FakeEvaluator {
                calls: Arc::clone(&resumed_calls),
                fail_call: None,
            }),
            false,
            false,
        )
        .unwrap();
        assert!(resumed.summary.resumed);
        assert_eq!(resumed_calls.load(Ordering::SeqCst), 0);
        assert_eq!(resumed.summary.evaluations, first.summary.evaluations);
        assert_eq!(
            resumed.summary.telemetry.candidate_evaluations,
            first.summary.telemetry.candidate_evaluations
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resource_budget_is_persisted_and_exhaustion_is_explainable() {
        let root = std::env::temp_dir().join(format!(
            "sonium-waveguide-resource-budget-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let mut plan = OptimizationPlanV1::waveguide_preview();
        plan.coarse.max_iterations = 1;
        plan.coarse.shortlist_size = 2;
        plan.resource_budget = Some(OptimizationResourceBudget {
            max_candidate_evaluations: 1,
            max_scenario_evaluations: 1,
            max_frequency_evaluations: 10,
            max_rhs_evaluations: 20,
        });
        let calls = Arc::new(AtomicUsize::new(0));
        let result = optimize_with_evaluator(
            &sample_model(),
            &plan,
            &root,
            false,
            Box::new(FakeEvaluator {
                calls: Arc::clone(&calls),
                fail_call: None,
            }),
            false,
            false,
        )
        .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(result.summary.telemetry.candidate_evaluations, 1);
        assert_eq!(result.summary.telemetry.frequency_evaluations, 10);
        assert_eq!(result.summary.telemetry.rhs_evaluations, 20);
        assert!(result.summary.telemetry.budget_exhausted_evaluations > 0);
        assert!(result.summary.telemetry.candidates_per_hour.is_finite());
        assert!(
            result
                .summary
                .telemetry
                .frequency_evaluations_per_hour
                .is_finite()
        );
        assert!(
            result
                .summary
                .telemetry
                .rhs_evaluations_per_hour
                .is_finite()
        );
        let history = load_records(&root, &fake_checkpoint_identity(&plan)).unwrap();
        assert!(history.iter().any(|record| matches!(
            record.status,
            EvaluationStatus::BudgetExhausted {
                resource: BudgetResource::CandidateEvaluations,
                limit: 1,
            }
        )));
        let charged = history
            .iter()
            .find(|record| record.frequency_evaluations == 10 && record.rhs_evaluations == 20)
            .unwrap();
        assert_eq!(charged.frequency_evaluations, 10);
        assert_eq!(charged.rhs_evaluations, 20);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn frequency_and_rhs_budgets_exhaust_before_invoking_the_evaluator() {
        let run = |max_frequency_evaluations, max_rhs_evaluations| {
            let root = std::env::temp_dir().join(format!(
                "sonium-waveguide-work-budget-{}-{max_frequency_evaluations}-{max_rhs_evaluations}",
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&root);
            fs::create_dir_all(&root).unwrap();
            let mut plan = OptimizationPlanV1::waveguide_preview();
            plan.coarse.max_iterations = 1;
            plan.coarse.shortlist_size = 1;
            plan.resource_budget = Some(OptimizationResourceBudget {
                max_candidate_evaluations: 8,
                max_scenario_evaluations: 8,
                max_frequency_evaluations,
                max_rhs_evaluations,
            });
            let calls = Arc::new(AtomicUsize::new(0));
            let result = optimize_with_evaluator(
                &sample_model(),
                &plan,
                &root,
                false,
                Box::new(FakeEvaluator {
                    calls: Arc::clone(&calls),
                    fail_call: None,
                }),
                false,
                false,
            )
            .unwrap();
            let history = load_records(&root, &fake_checkpoint_identity(&plan)).unwrap();
            fs::remove_dir_all(root).unwrap();
            (calls.load(Ordering::SeqCst), result, history)
        };

        let (frequency_calls, frequency_result, frequency_history) = run(9, 20);
        assert_eq!(frequency_calls, 0);
        assert!(frequency_history.iter().any(|record| matches!(
            record.status,
            EvaluationStatus::BudgetExhausted {
                resource: BudgetResource::FrequencyEvaluations,
                limit: 9,
            }
        )));
        assert_eq!(frequency_result.summary.telemetry.frequency_evaluations, 0);

        let (rhs_calls, rhs_result, rhs_history) = run(10, 19);
        assert_eq!(rhs_calls, 0);
        assert!(rhs_history.iter().any(|record| matches!(
            record.status,
            EvaluationStatus::BudgetExhausted {
                resource: BudgetResource::RhsEvaluations,
                limit: 19,
            }
        )));
        assert_eq!(rhs_result.summary.telemetry.rhs_evaluations, 0);
    }

    #[test]
    fn work_budget_exhaustion_preserves_every_robust_scenario_identity() {
        let root = std::env::temp_dir().join(format!(
            "sonium-waveguide-robust-work-budget-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let mut plan = OptimizationPlanV1::waveguide_preview();
        plan.coarse.max_iterations = 1;
        plan.coarse.shortlist_size = 1;
        plan.robust_uncertainty = Some(RobustUncertaintyConfig {
            tolerance: SpeakerToleranceSpec::default(),
            sampling: UncertaintySampling::Corners,
            max_cases: 512,
        });
        plan.resource_budget = Some(OptimizationResourceBudget {
            max_candidate_evaluations: 1,
            max_scenario_evaluations: 512,
            max_frequency_evaluations: 5_119,
            max_rhs_evaluations: 10_240,
        });
        let calls = Arc::new(AtomicUsize::new(0));
        let result = optimize_with_evaluator(
            &sample_model(),
            &plan,
            &root,
            false,
            Box::new(FakeEvaluator {
                calls: Arc::clone(&calls),
                fail_call: None,
            }),
            false,
            false,
        )
        .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(result.summary.telemetry.frequency_evaluations, 0);

        let history = load_records(&root, &fake_checkpoint_identity(&plan)).unwrap();
        let exhausted = history
            .iter()
            .find(|record| {
                matches!(
                    record.status,
                    EvaluationStatus::BudgetExhausted {
                        resource: BudgetResource::FrequencyEvaluations,
                        limit: 5_119,
                    }
                ) && record.scenario_results.len() == 512
            })
            .unwrap();
        assert!(
            exhausted
                .scenario_results
                .iter()
                .enumerate()
                .all(|(index, scenario)| {
                    scenario.index == index
                        && matches!(
                            scenario.status,
                            EvaluationStatus::BudgetExhausted {
                                resource: BudgetResource::FrequencyEvaluations,
                                limit: 5_119,
                            }
                        )
                        && scenario.metrics.is_none()
                })
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[ignore = "records workstation timing; not a CI wall-clock assertion"]
    fn m4_waveguide_preview_timing_benchmark() {
        let _plan = OptimizationPlanV1::waveguide_preview();
        let _model = sample_model();
        // Run manually with a production driver model and output directory;
        // this ignored test exists as the stable benchmark entry point.
    }
}
