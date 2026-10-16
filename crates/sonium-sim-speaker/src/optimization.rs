//! Loudspeaker parameter optimisation.
//!
//! Optimises driver positions and waveguide shape for flat on-axis
//! response and smooth radiation pattern (flat PIR with ≤2 dB/oct slope).
//!
//! Uses the augmented Lagrangian method from `sonium-solvers` with
//! L-BFGS-B inner solver. The objective and cache contracts are kept smooth
//! and deterministic so either finite-difference or discrete-adjoint
//! gradients can drive the same optimization problem.

use num_complex::Complex64;
use sonium_bem::core::assembly::tbem::{
    build_tbem_rhs_stabilized, build_tbem_system_stabilized, eval_tbem_rows_stabilized,
};
use sonium_bem::core::mesh::octree::Octree;
use sonium_bem::core::postprocess::pressure::{
    build_constant_field_operator, eval_constant_field_operator_columns,
};
use sonium_bem::core::types::PhysicsParams;
use sonium_compute::HostMatrix;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::lanes::{LaneConfig, host_memory_bytes, run_lanes, try_plan_lanes};
use sonium_solvers::optimization::{
    AugmentedLagrangianConfig, Bound, Constraint, ConstraintKind, InnerSolver, augmented_lagrangian,
};

use crate::cea2034_output::{self, compute_spinorama_metrics};
use crate::model::LoudspeakerModel;
use crate::simulation::{
    self, ExecutionProfile, SentinelPolicy, SimulationError, SimulationProvenance,
    SimulationResult, SolverConfig,
};

mod staged;
pub use staged::*;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Result of the optimisation.
pub struct OptimizationResult {
    /// Optimised model.
    pub model: LoudspeakerModel,
    /// Final objective value on the truncated optimisation band.
    pub objective: f64,
    /// Objective of the final candidate re-evaluated over the full production
    /// band at the production mesh settings (see
    /// [`OptimizationConfig::optim_max_mesh_freq`]).
    pub full_band_objective: f64,
    /// Gradient backend that actually drove the search. The discrete adjoint
    /// falls back to central differences when it is unavailable (DOF limit,
    /// typed driver models); that fallback is reported here instead of only
    /// appearing in the logs.
    pub gradient_method: AppliedGradientMethod,
    /// Number of outer iterations.
    pub iterations: usize,
    /// Whether the optimiser converged.
    pub converged: bool,
}

/// Gradient backend that actually produced the search gradients.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppliedGradientMethod {
    /// Every gradient came from the discrete adjoint.
    DiscreteAdjoint,
    /// Central differences were configured directly.
    CentralDifference,
    /// The discrete adjoint was configured but at least one gradient fell
    /// back to central differences; `reason` is the first adjoint error.
    AdjointWithFiniteDifferenceFallback { reason: String },
}

/// Recoverable optimisation setup, simulation, or solver failure.
#[derive(Debug, Clone)]
pub struct OptimizationError {
    message: String,
}

impl OptimizationError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for OptimizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for OptimizationError {}

impl From<SimulationError> for OptimizationError {
    fn from(error: SimulationError) -> Self {
        Self::new(format!(
            "speaker simulation failed during optimisation: {error}"
        ))
    }
}

/// Configuration for the optimisation.
pub struct OptimizationConfig {
    /// Maximum outer iterations for augmented Lagrangian.
    pub max_iterations: usize,
    /// Weight for the flat on-axis response term.
    pub weight_flat: f64,
    /// Weight for the smooth PIR term.
    pub weight_pir: f64,
    /// Number of frequency points used during optimisation (coarser = faster).
    pub optim_freq_points: usize,
    /// Maximum mesh frequency during optimisation (Hz).
    ///
    /// Optimization truncates its search sweep at this frequency when it is
    /// below the production upper limit. The final candidate is re-evaluated
    /// over the full production band.
    pub optim_max_mesh_freq: f64,
    /// Gradient evaluation method.
    pub gradient_method: OptimizationGradientMethod,
    /// Hard cap on parallel candidate lanes for batched candidate
    /// evaluations (finite-difference gradients).
    ///
    /// `None` derives the candidate-lane count from the thread budget and a
    /// per-candidate memory estimate that reuses
    /// [`SolverConfig::frequency_sweep_memory_fraction`]. `Some(1)` keeps the
    /// serial per-candidate evaluation; each remaining level of parallelism
    /// inside one candidate (the frequency sweep) receives an equal share of
    /// the thread budget.
    pub max_candidate_lanes: Option<usize>,
    /// Largest dense BEM system accepted by the discrete-adjoint path.
    ///
    /// Below this limit the adjoint forward pass assembles and stores dense
    /// operators (O(N²) memory per frequency). Above it the adjoint switches
    /// to the compressed H-matrix operator path (see
    /// [`OptimizationConfig::adjoint_hmatrix`]), which needs no dense
    /// assembly: the forward solve runs FGMRES on the H-matrix and the
    /// adjoint `Aᴴ` solve reuses its Hermitian matvec. Set
    /// `adjoint_hmatrix = false` to keep the legacy behaviour (hard error →
    /// central-difference fallback, reported via
    /// [`OptimizationResult::gradient_method`]). For reference, the bundled
    /// two-way test model meshes to 374 / 5 504 / 38 744 DOFs at
    /// 0.7 / 3 / 8 kHz.
    pub adjoint_max_dofs: usize,
    /// Allow the discrete adjoint to fall back to the H-matrix operator path
    /// when the mesh exceeds [`OptimizationConfig::adjoint_max_dofs`]
    /// (default: true).
    pub adjoint_hmatrix: bool,
    /// Exact uncompressed residual validation policy for H-matrix solves made
    /// while optimizing. `FinalOnly` keeps the normal periodic/marginal
    /// sentinel during the search and forces every exact check on the final
    /// production re-evaluation.
    pub exact_residual_policy: ExactResidualPolicy,
}

/// Controls expensive uncompressed TBEM residual checks during optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactResidualPolicy {
    /// Disable exact H-matrix residual checks for candidate and final solves.
    Off,
    /// Check every H-matrix solve, including all candidate and final solves.
    EveryEvaluation,
    /// Use the solver's normal sentinel for candidates and check every final
    /// production solve.
    FinalOnly,
}

/// Gradient backend for speaker optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationGradientMethod {
    /// One complex-symmetric adjoint solve per frequency. Geometry/operator
    /// derivatives are assembled without additional radiation solves.
    DiscreteAdjoint,
    /// Compatibility/debug path using `2n` full BEM simulations.
    CentralDifference,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            weight_flat: 1.0,
            weight_pir: 1.0,
            optim_freq_points: 50,
            optim_max_mesh_freq: 8_000.0,
            gradient_method: OptimizationGradientMethod::DiscreteAdjoint,
            max_candidate_lanes: None,
            adjoint_max_dofs: 5_000,
            adjoint_hmatrix: true,
            exact_residual_policy: ExactResidualPolicy::FinalOnly,
        }
    }
}

fn optimization_solver_config(policy: ExactResidualPolicy, final_evaluation: bool) -> SolverConfig {
    let mut config = SolverConfig::default();
    config.execution_profile = if final_evaluation {
        ExecutionProfile::Golden
    } else {
        ExecutionProfile::Preview
    };
    config.exact_residual_sentinel = match policy {
        ExactResidualPolicy::Off => SentinelPolicy::Off,
        ExactResidualPolicy::EveryEvaluation => SentinelPolicy::EverySolve,
        ExactResidualPolicy::FinalOnly if final_evaluation => SentinelPolicy::EverySolve,
        ExactResidualPolicy::FinalOnly => config.exact_residual_sentinel,
    };
    config.verify_exact_hmatrix_residual = false;
    config
}

const OBJECTIVE_FAILURE_PENALTY: f64 = 1.0e6;
const CONSTRAINT_FAILURE_PENALTY: f64 = -1.0e3;
const F3_TRANSITION_WIDTH_DB: f64 = 0.35;

fn validate_optimization_setup(
    model: &LoudspeakerModel,
    config: &OptimizationConfig,
) -> Result<(), OptimizationError> {
    model.validate().map_err(|message| {
        OptimizationError::new(format!("invalid loudspeaker model: {message}"))
    })?;
    if model.waveguide.is_none() {
        return Err(OptimizationError::new(
            "loudspeaker optimisation requires a waveguide",
        ));
    }
    if model.simulation.freq_end <= model.simulation.freq_start {
        return Err(OptimizationError::new(
            "optimisation requires a non-zero frequency range",
        ));
    }
    if config.max_iterations == 0 {
        return Err(OptimizationError::new(
            "maximum optimisation iterations must be at least 1",
        ));
    }
    for (name, value) in [
        ("flatness weight", config.weight_flat),
        ("PIR weight", config.weight_pir),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(OptimizationError::new(format!(
                "{name} must be finite and non-negative"
            )));
        }
    }
    if config.weight_flat == 0.0 && config.weight_pir == 0.0 {
        return Err(OptimizationError::new(
            "at least one optimisation objective weight must be positive",
        ));
    }
    if config.optim_freq_points < 2 {
        return Err(OptimizationError::new(
            "optimisation frequency count must be at least 2",
        ));
    }
    if !config.optim_max_mesh_freq.is_finite() || config.optim_max_mesh_freq <= 0.0 {
        return Err(OptimizationError::new(
            "optimisation mesh frequency must be finite and positive",
        ));
    }
    if config.optim_max_mesh_freq <= model.simulation.freq_start {
        return Err(OptimizationError::new(
            "optimisation mesh frequency must exceed the simulation start frequency",
        ));
    }
    if config.max_candidate_lanes == Some(0) {
        return Err(OptimizationError::new(
            "maximum candidate lanes must be at least 1 when set",
        ));
    }
    if config.adjoint_max_dofs == 0 {
        return Err(OptimizationError::new(
            "adjoint DOF limit must be at least one",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Parameter packing
// ---------------------------------------------------------------------------

/// Pack optimisable parameters from a model into a flat vector.
///
/// Layout: `[woofer_y, tweeter_y, optional cone shapes..., wg_h_profile...,
/// wg_v_profile..., wg_depth]`.
///
/// The final control point of each waveguide spline is the tweeter aperture
/// and is intentionally fixed at `tweeter.radius`; including those endpoints
/// as design variables produced invalid perturbed models.
fn pack_parameters(model: &LoudspeakerModel) -> Result<Vec<f64>, OptimizationError> {
    model.validate().map_err(|message| {
        OptimizationError::new(format!("invalid loudspeaker model: {message}"))
    })?;
    let wg = model
        .waveguide
        .as_ref()
        .ok_or_else(|| OptimizationError::new("loudspeaker optimisation requires a waveguide"))?;
    let mut params = Vec::new();
    params.push(model.woofer.y_offset);
    params.push(model.tweeter.y_offset);
    for driver in [&model.woofer, &model.tweeter] {
        if let Some(cone) = driver.cone {
            params.push(cone.cone_depth);
            params.push(cone.dustcap_radius);
        }
    }
    for pt in wg
        .horizontal_spline
        .iter()
        .take(wg.horizontal_spline.len().saturating_sub(1))
    {
        params.push(pt[1]); // radial values only (t is fixed)
    }
    for pt in wg
        .vertical_spline
        .iter()
        .take(wg.vertical_spline.len().saturating_sub(1))
    {
        params.push(pt[1]);
    }
    params.push(wg.depth);
    if params.iter().any(|value| !value.is_finite()) {
        return Err(OptimizationError::new(
            "optimisation parameter vector contains non-finite values",
        ));
    }
    Ok(params)
}

/// Unpack a flat parameter vector back into a model (cloning the base model).
fn unpack_parameters(
    x: &[f64],
    base: &LoudspeakerModel,
) -> Result<LoudspeakerModel, OptimizationError> {
    let base_waveguide = base
        .waveguide
        .as_ref()
        .ok_or_else(|| OptimizationError::new("loudspeaker optimisation requires a waveguide"))?;
    let cone_parameters = [base.woofer.cone, base.tweeter.cone]
        .iter()
        .filter(|cone| cone.is_some())
        .count()
        * 2;
    let expected = 3
        + cone_parameters
        + base_waveguide.horizontal_spline.len().saturating_sub(1)
        + base_waveguide.vertical_spline.len().saturating_sub(1);
    if x.len() != expected {
        return Err(OptimizationError::new(format!(
            "optimisation parameter vector has length {}, expected {expected}",
            x.len()
        )));
    }
    if x.iter().any(|value| !value.is_finite()) {
        return Err(OptimizationError::new(
            "optimisation parameter vector contains non-finite values",
        ));
    }

    let mut model = base.clone();
    let mut parameters = x.iter().copied();
    model.woofer.y_offset = parameters.next().ok_or_else(|| {
        OptimizationError::new("optimisation parameter vector is missing woofer position")
    })?;
    model.tweeter.y_offset = parameters.next().ok_or_else(|| {
        OptimizationError::new("optimisation parameter vector is missing tweeter position")
    })?;
    for driver in [&mut model.woofer, &mut model.tweeter] {
        if let Some(cone) = &mut driver.cone {
            cone.cone_depth = parameters.next().ok_or_else(|| {
                OptimizationError::new("optimisation parameter vector is missing diaphragm depth")
            })?;
            cone.dustcap_radius = parameters.next().ok_or_else(|| {
                OptimizationError::new(
                    "optimisation parameter vector is missing diaphragm cap radius",
                )
            })?;
        }
    }

    let wg = model
        .waveguide
        .as_mut()
        .ok_or_else(|| OptimizationError::new("loudspeaker optimisation requires a waveguide"))?;
    let horizontal_design_points = wg.horizontal_spline.len().saturating_sub(1);
    for pt in wg
        .horizontal_spline
        .iter_mut()
        .take(horizontal_design_points)
    {
        pt[1] = parameters.next().ok_or_else(|| {
            OptimizationError::new("optimisation parameter vector is missing waveguide radius")
        })?;
    }
    let vertical_design_points = wg.vertical_spline.len().saturating_sub(1);
    for pt in wg.vertical_spline.iter_mut().take(vertical_design_points) {
        pt[1] = parameters.next().ok_or_else(|| {
            OptimizationError::new("optimisation parameter vector is missing waveguide radius")
        })?;
    }
    wg.depth = parameters.next().ok_or_else(|| {
        OptimizationError::new("optimisation parameter vector is missing waveguide depth")
    })?;
    if parameters.next().is_some() {
        return Err(OptimizationError::new(
            "optimisation parameter vector contains trailing values",
        ));
    }

    model
        .validate()
        .map_err(|message| OptimizationError::new(format!("invalid optimised model: {message}")))?;
    Ok(model)
}

/// Compute box constraints for the parameter vector.
fn compute_bounds(model: &LoudspeakerModel) -> Result<Vec<Bound<f64>>, OptimizationError> {
    model.validate().map_err(|message| {
        OptimizationError::new(format!("invalid loudspeaker model: {message}"))
    })?;
    let h2 = model.cabinet.height / 2.0;
    let w2 = model.cabinet.width / 2.0;
    let sep = model.separator.y_position;
    let sep_half = model.separator.thickness / 2.0;

    let mut bounds = Vec::new();

    // Woofer y: bottom of cabinet to below separator
    bounds.push(Bound::both(
        -h2 + model.woofer.radius,
        sep - sep_half - model.woofer.radius,
    ));
    // Tweeter y: above separator to top of cabinet
    bounds.push(Bound::both(
        sep + sep_half + model.tweeter.radius,
        h2 - model.tweeter.radius,
    ));
    // Existing procedural cone geometries enlarge the shape-design space
    // without introducing a mesh-specific parameterization.
    for driver in [&model.woofer, &model.tweeter] {
        if driver.cone.is_some() {
            bounds.push(Bound::both(1.0e-4, model.cabinet.depth * 0.5));
            bounds.push(Bound::both(
                (driver.radius * 0.02).max(1.0e-5),
                driver.radius * 0.95,
            ));
        }
    }

    let wg = model
        .waveguide
        .as_ref()
        .ok_or_else(|| OptimizationError::new("loudspeaker optimisation requires a waveguide"))?;
    // Waveguide H spline radii
    for _ in wg
        .horizontal_spline
        .iter()
        .take(wg.horizontal_spline.len().saturating_sub(1))
    {
        bounds.push(Bound::both(model.tweeter.radius, w2 * 0.8));
    }
    // Waveguide V spline radii
    for _ in wg
        .vertical_spline
        .iter()
        .take(wg.vertical_spline.len().saturating_sub(1))
    {
        bounds.push(Bound::both(model.tweeter.radius, h2 * 0.5));
    }
    // Waveguide depth
    bounds.push(Bound::both(0.005, model.cabinet.depth / 3.0));

    for (index, bound) in bounds.iter().enumerate() {
        if let (Some(lower), Some(upper)) = (bound.lower, bound.upper)
            && (!lower.is_finite() || !upper.is_finite() || lower > upper)
        {
            return Err(OptimizationError::new(format!(
                "optimisation bound {index} is empty or non-finite: [{lower}, {upper}]"
            )));
        }
    }
    Ok(bounds)
}

// ---------------------------------------------------------------------------
// Objective and constraints
// ---------------------------------------------------------------------------

/// Compute the combined objective from a simulation result.
///
/// `L_flat` measures on-axis flatness above F3 (the low-frequency −3 dB
/// point). `L_pir` measures flatness and tilt of the CEA-2034 Predicted
/// In-Room curve, which is the proper weighted combination
/// `0.12·LW + 0.44·ER + 0.44·SP` rather than an unweighted angular mean.
fn compute_objective(
    result: &SimulationResult,
    config: &OptimizationConfig,
) -> Result<f64, OptimizationError> {
    let metrics = compute_spinorama_metrics(result).map_err(|message| {
        OptimizationError::new(format!("spinorama scoring failed: {message}"))
    })?;
    let on_axis = &metrics.on_axis;
    let pir = &metrics.pir;

    // L_flat: smoothly weighted mean-square deviation above the −3 dB
    // transition. A hard F3 array index makes an infinitesimal geometry
    // change add or remove a complete frequency bin, which destroys both
    // finite-difference and adjoint gradient checks.
    let f3_weights = smooth_above_f3_weights(on_axis)?;
    let weight_sum = f3_weights.iter().sum::<f64>();
    let mean_above = on_axis
        .iter()
        .zip(&f3_weights)
        .map(|(&spl, &weight)| spl * weight)
        .sum::<f64>()
        / weight_sum;
    let l_flat = on_axis
        .iter()
        .zip(&f3_weights)
        .map(|(&spl, &weight)| weight * (spl - mean_above).powi(2))
        .sum::<f64>()
        / weight_sum;

    // L_pir: mean-square deviation of the (proper, weighted) PIR from
    // its mean, plus a slope penalty when the PIR tilts more than 2 dB
    // per octave (the CEA-2034 preference-score guideline).
    let pir_mean = pir.iter().sum::<f64>() / pir.len().max(1) as f64;
    let l_pir_flat: f64 =
        pir.iter().map(|&s| (s - pir_mean).powi(2)).sum::<f64>() / pir.len().max(1) as f64;

    let l_slope = compute_max_octave_slope(&result.frequencies, pir)?;
    let slope_penalty = (l_slope - 2.0).max(0.0).powi(2);

    let objective =
        config.weight_flat * l_flat + config.weight_pir * (l_pir_flat + 10.0 * slope_penalty);
    if !objective.is_finite() {
        return Err(OptimizationError::new(
            "optimisation objective is non-finite",
        ));
    }
    Ok(objective)
}

/// Smooth membership weights for the response region above the −3 dB point.
///
/// The soft maximum and logistic transition make the score differentiable
/// when a frequency bin crosses the nominal F3 threshold.
fn smooth_above_f3_weights(on_axis: &[f64]) -> Result<Vec<f64>, OptimizationError> {
    if on_axis.is_empty() || on_axis.iter().any(|value| !value.is_finite()) {
        return Err(OptimizationError::new(
            "on-axis response must contain finite values",
        ));
    }
    let max_spl = on_axis.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let beta = 8.0;
    let soft_peak = max_spl
        + on_axis
            .iter()
            .map(|&spl| (beta * (spl - max_spl)).exp())
            .sum::<f64>()
            .ln()
            / beta;
    let threshold = soft_peak - 3.0;
    let weights = on_axis
        .iter()
        .map(|&spl| {
            let argument = ((spl - threshold) / F3_TRANSITION_WIDTH_DB).clamp(-40.0, 40.0);
            1.0 / (1.0 + (-argument).exp())
        })
        .collect::<Vec<_>>();
    if weights.iter().sum::<f64>() <= f64::MIN_POSITIVE {
        return Err(OptimizationError::new(
            "smooth F3 weighting produced no active frequencies",
        ));
    }
    Ok(weights)
}

/// Maximum absolute SPL change per octave across the frequency range.
fn compute_max_octave_slope(freqs: &[f64], spl: &[f64]) -> Result<f64, OptimizationError> {
    if freqs.len() != spl.len() {
        return Err(OptimizationError::new(format!(
            "frequency and SPL vectors have different lengths ({} vs {})",
            freqs.len(),
            spl.len()
        )));
    }
    if freqs.len() < 2 {
        return Err(OptimizationError::new(
            "at least two frequency points are required to compute octave slope",
        ));
    }
    if spl.iter().any(|value| !value.is_finite()) {
        return Err(OptimizationError::new(
            "SPL vector contains non-finite values",
        ));
    }
    let mut max_slope: f64 = 0.0;
    for i in 1..freqs.len() {
        if !freqs[i - 1].is_finite()
            || !freqs[i].is_finite()
            || freqs[i - 1] <= 0.0
            || freqs[i] <= freqs[i - 1]
        {
            return Err(OptimizationError::new(format!(
                "frequencies must be finite, positive, and strictly increasing (invalid pair at indices {} and {i})",
                i - 1
            )));
        }
        let octaves = (freqs[i] / freqs[i - 1]).log2();
        let slope = (spl[i] - spl[i - 1]).abs() / octaves;
        if slope > max_slope {
            max_slope = slope;
        }
    }
    if !max_slope.is_finite() {
        return Err(OptimizationError::new("octave slope is non-finite"));
    }
    Ok(max_slope)
}

/// Compute the PIR slope constraint value: `2.0 - max_slope >= 0`.
///
/// Uses the CEA-2034 weighted Predicted In-Room curve so the constraint
/// matches the published preference-score guideline (≤ 2 dB/oct PIR tilt).
fn pir_slope_constraint(result: &SimulationResult) -> Result<f64, OptimizationError> {
    let metrics = compute_spinorama_metrics(result).map_err(|message| {
        OptimizationError::new(format!("spinorama scoring failed: {message}"))
    })?;
    let max_slope = compute_max_octave_slope(&result.frequencies, &metrics.pir)?;
    Ok(2.0 - max_slope)
}

// ---------------------------------------------------------------------------
// Simulation cache
// ---------------------------------------------------------------------------

use std::sync::{Arc, Mutex};

struct SimCache {
    last_x: Option<Vec<f64>>,
    last_result: Option<Result<SimulationResult, OptimizationError>>,
    computations: usize,
    warm_start: simulation::SimulationWarmStart,
}

impl SimCache {
    fn new() -> Self {
        Self {
            last_x: None,
            last_result: None,
            computations: 0,
            warm_start: simulation::SimulationWarmStart::default(),
        }
    }

    fn get_or_compute(
        &mut self,
        x: &[f64],
        base_model: &LoudspeakerModel,
        eval_points: &[[f64; 3]],
        solver_config: &SolverConfig,
    ) {
        let x_vec: Vec<f64> = x.to_vec();
        if self.last_x.as_ref() == Some(&x_vec) && self.last_result.is_some() {
            return;
        }

        let result = unpack_parameters(x, base_model).and_then(|model| {
            simulation::run_simulation_with_warm_start(
                &model,
                eval_points,
                solver_config,
                &mut self.warm_start,
            )
            .map_err(OptimizationError::from)
        });
        self.last_x = Some(x_vec);
        self.last_result = Some(result);
        self.computations += 1;
    }

    /// Clone of the warm start accumulated by the most recent evaluation,
    /// used to seed lane-local caches for batched candidate evaluations.
    fn warm_start_snapshot(&self) -> simulation::SimulationWarmStart {
        self.warm_start.clone()
    }
}

fn failure_objective(x: &[f64], bounds: &[Bound<f64>]) -> f64 {
    let violation = x
        .iter()
        .zip(bounds)
        .map(|(&value, bound)| {
            let below = bound.lower.map_or(0.0, |lower| (lower - value).max(0.0));
            let above = bound.upper.map_or(0.0, |upper| (value - upper).max(0.0));
            below * below + above * above
        })
        .sum::<f64>();
    OBJECTIVE_FAILURE_PENALTY + 1.0e4 * violation
}

// ---------------------------------------------------------------------------
// Batched candidate evaluation
// ---------------------------------------------------------------------------

/// Evaluate a batch of candidate parameter vectors, returning one objective
/// value per candidate in input order.
///
/// Decomposition policy (scheduler §7.2 — candidate parallelism has top
/// priority for finite-difference/population work):
///
/// - `candidate_lanes = min(n_candidates, thread_budget[, max_candidate_lanes])`;
///   candidate lanes are planned first and each candidate's frequency sweep
///   then receives `max(1, thread_budget / candidate_lanes)` frequency lanes,
///   so the total thread count stays ≈ the budget. Both levels run on
///   dedicated local Rayon pools (`run_lanes` never touches the global
///   pool), so nesting the sweep's lane executor inside a candidate lane is
///   safe.
/// - Memory admission reuses `SolverConfig::frequency_sweep_memory_fraction`:
///   one candidate's live bytes are estimated as one frequency lane's solver
///   state (measured on the base model) times its frequency-lane count, and
///   `plan_lanes` admits candidate lanes until that budget is exhausted.
/// - Lane-local state: each lane owns a `SimulationWarmStart` cloned once
///   from `base_warm_start` and updated only by the candidates that lane
///   evaluated. Warm starts therefore compound along a lane, never across
///   lanes, and no lane ever writes shared state. Warm starts only seed
///   iterative initial guesses, so results are independent of the lane
///   assignment up to solver tolerance.
///
/// A single candidate, `max_candidate_lanes == Some(1)`, or a memory plan
/// that admits only one lane all keep today's serial evaluation: one warm
/// start compounding across the batch and the full thread budget available
/// to each candidate's sweep.
fn evaluate_candidates_batch(
    candidates: &[Vec<f64>],
    base_model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
    solver_config: &SolverConfig,
    config: &OptimizationConfig,
    base_warm_start: Option<&simulation::SimulationWarmStart>,
) -> Vec<Result<f64, OptimizationError>> {
    let evaluate_one = |warm_start: &mut simulation::SimulationWarmStart,
                        x: &[f64],
                        frequency_lanes: Option<usize>|
     -> Result<f64, OptimizationError> {
        let model = unpack_parameters(x, base_model)?;
        let lane_solver_config;
        let solver = match frequency_lanes {
            Some(lanes) => {
                lane_solver_config = SolverConfig {
                    max_frequency_lanes: Some(lanes),
                    ..solver_config.clone()
                };
                &lane_solver_config
            }
            None => solver_config,
        };
        let result =
            simulation::run_simulation_with_warm_start(&model, eval_points, solver, warm_start)?;
        compute_objective(&result, config)
    };
    let evaluate_serial = |candidates: &[Vec<f64>]| -> Vec<Result<f64, OptimizationError>> {
        let mut warm_start = base_warm_start.cloned().unwrap_or_default();
        candidates
            .iter()
            .map(|x| evaluate_one(&mut warm_start, x, None))
            .collect()
    };

    if candidates.len() <= 1 || config.max_candidate_lanes == Some(1) {
        return evaluate_serial(candidates);
    }

    let thread_budget = rayon::current_num_threads();
    let mut candidate_lane_cap = candidates.len().min(thread_budget);
    if let Some(max) = config.max_candidate_lanes {
        candidate_lane_cap = candidate_lane_cap.min(max);
    }
    let frequency_lanes = (thread_budget / candidate_lane_cap).max(1);

    let per_candidate_live_bytes =
        simulation::estimate_frequency_sweep_live_bytes(base_model, solver_config)
            .saturating_mul(frequency_lanes as u64);
    let memory_budget_bytes = host_memory_bytes()
        .map(|bytes| (bytes as f64 * solver_config.frequency_sweep_memory_fraction) as u64);
    let plan = match try_plan_lanes(
        candidates.len(),
        |_| per_candidate_live_bytes,
        &LaneConfig {
            max_lanes: Some(candidate_lane_cap),
            thread_budget,
            memory_budget_bytes,
        },
    ) {
        Ok(plan) => plan,
        Err(error) => {
            return candidates
                .iter()
                .map(|_| {
                    Err(OptimizationError::new(format!(
                        "candidate lane admission: {error}"
                    )))
                })
                .collect();
        }
    };
    if plan.num_lanes <= 1 {
        return evaluate_serial(candidates);
    }

    log::info!(
        "Candidate batch: {} lane(s) over {} candidates, {} frequency lane(s) per candidate",
        plan.num_lanes,
        candidates.len(),
        frequency_lanes,
    );
    let seed = base_warm_start.cloned();
    let outcome = run_lanes(
        candidates,
        &plan,
        || seed.clone().unwrap_or_default(),
        &|warm_start: &mut simulation::SimulationWarmStart, _index, x: &Vec<f64>| {
            evaluate_one(warm_start, x, Some(frequency_lanes))
        },
        None,
    );
    debug_assert!(
        !outcome.cancelled,
        "candidate batch has no cancellation source"
    );
    outcome
        .results
        .into_iter()
        .map(|result| result.expect("uncancelled candidate batch covers every candidate"))
        .collect()
}

/// Central-difference gradient whose `2n` perturbed candidate evaluations
/// are scheduled through [`evaluate_candidates_batch`].
///
/// Step sizes (`h * max(1, |x_i|)`), the plus/minus pairing per parameter,
/// and the failure semantics (a failed candidate evaluation contributes its
/// bound-violation penalty value, exactly like the objective closure) match
/// `gradient_central`; only the evaluation scheduling and the warm-start
/// caching differ.
#[allow(clippy::too_many_arguments)]
fn gradient_central_batched(
    x: &[f64],
    h: f64,
    base_model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
    solver_config: &SolverConfig,
    config: &OptimizationConfig,
    bounds: &[Bound<f64>],
    base_warm_start: Option<&simulation::SimulationWarmStart>,
) -> Vec<f64> {
    let n = x.len();
    let mut points = Vec::with_capacity(2 * n);
    let mut steps = Vec::with_capacity(n);
    let mut x_pert = x.to_vec();
    for i in 0..n {
        let xi_orig = x_pert[i];
        let hi = h * 1.0_f64.max(xi_orig.abs());
        steps.push(hi);
        x_pert[i] = xi_orig + hi;
        points.push(x_pert.clone());
        x_pert[i] = xi_orig - hi;
        points.push(x_pert.clone());
        x_pert[i] = xi_orig;
    }
    let values = evaluate_candidates_batch(
        &points,
        base_model,
        eval_points,
        solver_config,
        config,
        base_warm_start,
    );
    let mut grad = vec![0.0; n];
    for i in 0..n {
        let fp = values[2 * i]
            .as_ref()
            .copied()
            .unwrap_or_else(|_| failure_objective(&points[2 * i], bounds));
        let fm = values[2 * i + 1]
            .as_ref()
            .copied()
            .unwrap_or_else(|_| failure_objective(&points[2 * i + 1], bounds));
        grad[i] = (fp - fm) / (2.0 * steps[i]);
    }
    grad
}

/// Forward BEM operator retained per frequency for the adjoint pass.
enum ForwardOperator {
    /// Dense TBEM matrix; the adjoint reuses the retained LU/ILU state.
    Dense(DenseMatrix<Complex64>),
    /// Compressed H-matrix; no dense assembly is ever materialized. The
    /// adjoint solves `Aᴴ λ = g` through the operator's Hermitian matvec.
    Hmatrix(sonium_bem::core::assembly::hmatrix::HMatrix),
}

impl ForwardOperator {
    /// Apply the forward operator `A x`.
    fn apply(&self, x: &[Complex64]) -> Vec<Complex64> {
        match self {
            ForwardOperator::Dense(matrix) => matrix.matvec(x),
            ForwardOperator::Hmatrix(hmatrix) => hmatrix.matvec(x),
        }
    }
}

struct DenseFrequencyState {
    operator: ForwardOperator,
    rhs: Vec<Complex64>,
    surface_pressure: Vec<Complex64>,
    field_operator: DenseMatrix<Complex64>,
    field_affine: Vec<Complex64>,
    field_pressure: Vec<Complex64>,
    /// Final driver velocities used to build the boundary RHS and field
    /// affine term. Typed-driver models make these geometry-dependent through
    /// the radiation-matrix network.
    driver_velocities: [Complex64; 2],
    /// Radiation matrix used by the typed-driver network, when present.
    radiation_matrix: Option<[[Complex64; 2]; 2]>,
    /// Forward solver factors/preconditioner reused by the dense adjoint Aᴴ
    /// solve. `None` on the H-matrix path, whose adjoint needs no retained
    /// factors (its Jacobi preconditioner is rebuilt from the diagonal).
    solver_state: Option<simulation::BemSolverState>,
}

impl DenseFrequencyState {
    /// Dense operator access for dense-only test paths.
    #[cfg(test)]
    fn dense_matrix(&self) -> &DenseMatrix<Complex64> {
        match &self.operator {
            ForwardOperator::Dense(matrix) => matrix,
            ForwardOperator::Hmatrix(_) => panic!("test path expected a dense operator"),
        }
    }
}

struct DenseForwardEvaluation {
    result: SimulationResult,
    states: Vec<DenseFrequencyState>,
    /// Baseline mesh, retained so shape sensitivities can identify the
    /// elements a parameter perturbation actually moves.
    mesh: crate::mesh_gen::SpeakerMesh,
}

/// Test convenience: dense-only forward evaluation (H-matrix adjoint
/// disabled, matching the pre-B5 behaviour).
#[cfg(test)]
fn dense_forward_evaluation(
    model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
    solver_config: &SolverConfig,
    max_dofs: usize,
) -> Result<DenseForwardEvaluation, OptimizationError> {
    forward_evaluation(model, eval_points, solver_config, max_dofs, false)
}

/// Forward sweep retaining per-frequency operator state for the adjoint.
///
/// Dense below `max_dofs`; above it, either errors (legacy FD-fallback
/// trigger, `allow_hmatrix = false`) or runs the H-matrix operator path
/// (`allow_hmatrix = true`).
// Compatibility scalar fields are populated alongside the canonical driver axis.
#[allow(deprecated)]
fn forward_evaluation(
    model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
    solver_config: &SolverConfig,
    max_dofs: usize,
    allow_hmatrix: bool,
) -> Result<DenseForwardEvaluation, OptimizationError> {
    // This adjoint retains two-port velocity/radiation blocks. Additional
    // drivers must use the generic forward/finite-difference path until its
    // derivative state also carries the complete driver axis.
    if !model.additional_drivers.is_empty() {
        return Err(OptimizationError::new(
            "two-port discrete adjoint does not support additional drivers",
        ));
    }
    let has_typed_driver =
        model.woofer.driver_model.is_some() || model.tweeter.driver_model.is_some();
    let speaker_mesh = crate::mesh_gen::generate_loudspeaker_mesh(model);
    let use_hmatrix = speaker_mesh.triangles.len() > max_dofs;
    if use_hmatrix && !allow_hmatrix {
        return Err(OptimizationError::new(format!(
            "discrete adjoint dense limit exceeded: {} DOFs > {max_dofs} \
             (H-matrix adjoint disabled; enable adjoint_hmatrix or raise adjoint_max_dofs)",
            speaker_mesh.triangles.len()
        )));
    }
    let (nodes, mut elements) = simulation::speaker_mesh_to_bem(&speaker_mesh, model)?;
    // Geometry is fixed across the sweep: build the H-matrix clusters once.
    let hmatrix_clusters = use_hmatrix.then(|| {
        let centers: Vec<[f64; 3]> = elements.iter().map(|element| element.center).collect();
        let mut octree = Octree::build(&centers, 32, 8);
        octree.compute_interaction_lists(2.0);
        simulation::clusters_from_octree(&elements, &octree)
    });
    let frequencies = model.simulation.frequencies();
    let mut spl = HostMatrix::zeros(frequencies.len(), eval_points.len());
    let mut complex_pressure = HostMatrix::zeros(spl.nrows(), spl.ncols());
    let mut phase = HostMatrix::zeros(frequencies.len(), eval_points.len());
    let mut states = Vec::with_capacity(frequencies.len());
    let mut coupling_reports = Vec::with_capacity(frequencies.len());
    let mut previous = None;
    let mut adjoint_sweep = simulation::BemAdjointSweepState::default();
    let mut sentinel = simulation::ExactResidualSentinel::default();

    for (frequency_index, &frequency_hz) in frequencies.iter().enumerate() {
        let mut coupling_report = None;
        simulation::update_driver_bcs(&mut elements, &speaker_mesh, model, frequency_hz, None)?;
        let physics = PhysicsParams::new(
            frequency_hz,
            model.simulation.speed_of_sound,
            model.simulation.air_density,
            false,
        );
        let (operator, rhs, surface_pressure, solver_state, driver_velocities, radiation_matrix) =
            if use_hmatrix {
                let hmatrix = simulation::build_simulation_hmatrix(
                    &elements,
                    &nodes,
                    hmatrix_clusters.as_ref().expect("H-matrix clusters built"),
                    &physics,
                    solver_config,
                );
                if has_typed_driver {
                    let mut coupling_recycling = None;
                    let (surface_pressure, coupling) = simulation::solve_typed_driver_coupling(
                        &mut elements,
                        &speaker_mesh,
                        model,
                        frequency_hz,
                        &nodes,
                        &physics,
                        None,
                        Some(&hmatrix),
                        solver_config,
                        &mut coupling_recycling,
                        None,
                        previous.as_deref(),
                        &mut sentinel,
                    )?;
                    coupling_report = Some(coupling);
                    let coupling = coupling_report.as_ref().expect("coupling retained");
                    simulation::update_driver_bcs(
                        &mut elements,
                        &speaker_mesh,
                        model,
                        frequency_hz,
                        Some(&coupling.velocities_bem),
                    )?;
                    let rhs = build_tbem_rhs_stabilized(
                        &elements,
                        &nodes,
                        &physics,
                        simulation::EXTERIOR_TBEM_SIGN,
                        solver_config.beta_scale,
                    );
                    (
                        ForwardOperator::Hmatrix(hmatrix),
                        rhs,
                        surface_pressure,
                        None,
                        coupling.velocities_bem,
                        Some(coupling.radiation_matrix),
                    )
                } else {
                    let rhs = build_tbem_rhs_stabilized(
                        &elements,
                        &nodes,
                        &physics,
                        simulation::EXTERIOR_TBEM_SIGN,
                        solver_config.beta_scale,
                    );
                    let (surface_pressure, residual) = simulation::solve_bem_system_hmatrix(
                        &hmatrix,
                        &rhs,
                        previous.as_deref(),
                        solver_config,
                        None,
                    )?;
                    if sentinel.should_verify_exact(solver_config, residual) {
                        simulation::verify_exact_hmatrix_solution(
                            &elements,
                            &nodes,
                            &physics,
                            &surface_pressure,
                            &rhs,
                            solver_config,
                        )?;
                    }
                    (
                        ForwardOperator::Hmatrix(hmatrix),
                        rhs,
                        surface_pressure,
                        None,
                        [
                            model.woofer.velocity_at(frequency_hz),
                            model.tweeter.velocity_at(frequency_hz),
                        ],
                        None,
                    )
                }
            } else {
                let system = build_tbem_system_stabilized(
                    &elements,
                    &nodes,
                    &physics,
                    simulation::EXTERIOR_TBEM_SIGN,
                    solver_config.beta_scale,
                );
                if has_typed_driver {
                    let mut coupling_recycling = None;
                    let (_coupled_pressure, coupling) = simulation::solve_typed_driver_coupling(
                        &mut elements,
                        &speaker_mesh,
                        model,
                        frequency_hz,
                        &nodes,
                        &physics,
                        Some(&system.matrix),
                        None,
                        solver_config,
                        &mut coupling_recycling,
                        None,
                        previous.as_deref(),
                        &mut sentinel,
                    )?;
                    coupling_report = Some(coupling);
                    let coupling = coupling_report.as_ref().expect("coupling retained");
                    simulation::update_driver_bcs(
                        &mut elements,
                        &speaker_mesh,
                        model,
                        frequency_hz,
                        Some(&coupling.velocities_bem),
                    )?;
                    let rhs = build_tbem_rhs_stabilized(
                        &elements,
                        &nodes,
                        &physics,
                        simulation::EXTERIOR_TBEM_SIGN,
                        solver_config.beta_scale,
                    );
                    let (surface_pressure, solver_state) = adjoint_sweep.solve_forward(
                        &system.matrix,
                        &rhs,
                        previous.as_deref(),
                        frequency_hz,
                        solver_config,
                    )?;
                    (
                        ForwardOperator::Dense(system.matrix),
                        rhs,
                        surface_pressure,
                        Some(solver_state),
                        coupling.velocities_bem,
                        Some(coupling.radiation_matrix),
                    )
                } else {
                    let (surface_pressure, solver_state) = adjoint_sweep.solve_forward(
                        &system.matrix,
                        &system.rhs,
                        previous.as_deref(),
                        frequency_hz,
                        solver_config,
                    )?;
                    (
                        ForwardOperator::Dense(system.matrix),
                        system.rhs,
                        surface_pressure,
                        Some(solver_state),
                        [
                            model.woofer.velocity_at(frequency_hz),
                            model.tweeter.velocity_at(frequency_hz),
                        ],
                        None,
                    )
                }
            };
        let velocity = simulation::build_surface_velocity(&elements);
        let (field_operator, field_affine) = build_constant_field_operator(
            eval_points,
            &elements,
            &nodes,
            Some(&velocity),
            &physics,
        );
        let mut field_pressure = field_operator.matvec(&surface_pressure);
        for (pressure, affine_pressure) in field_pressure.iter_mut().zip(&field_affine) {
            *pressure += affine_pressure;
        }
        for (point_index, &pressure) in field_pressure.iter().enumerate() {
            spl[(frequency_index, point_index)] = sonium_xem::pressure_to_spl(pressure);
            complex_pressure[(frequency_index, point_index)] = pressure;
            phase[(frequency_index, point_index)] = pressure.arg();
        }
        previous = Some(surface_pressure.clone());
        states.push(DenseFrequencyState {
            operator,
            rhs,
            surface_pressure,
            field_operator,
            field_affine,
            field_pressure,
            driver_velocities,
            radiation_matrix,
            solver_state,
        });
        coupling_reports.push(coupling_report);
    }

    Ok(DenseForwardEvaluation {
        result: SimulationResult {
            frequencies,
            complex_pressure,
            spl,
            phase,
            woofer_electrical_impedance_ohm: coupling_reports
                .iter()
                .map(|report| {
                    report
                        .as_ref()
                        .and_then(|c| c.woofer_electrical_impedance_ohm)
                })
                .collect(),
            tweeter_electrical_impedance_ohm: coupling_reports
                .iter()
                .map(|report| {
                    report
                        .as_ref()
                        .and_then(|c| c.tweeter_electrical_impedance_ohm)
                })
                .collect(),
            driver_electrical_impedances_ohm: coupling_reports
                .iter()
                .map(|report| {
                    report.as_ref().map_or_else(
                        || vec![None; model.drivers().len()],
                        |c| c.driver_electrical_impedances_ohm.clone(),
                    )
                })
                .collect(),
            radiation_matrices: coupling_reports
                .iter()
                .map(|report| report.as_ref().map(|c| c.radiation_matrix_all.clone()))
                .collect(),
            radiation_matrix_diagnostics: coupling_reports
                .iter()
                .map(|report| report.as_ref().map(|c| c.radiation_matrix_diagnostics))
                .collect(),
            thermal_driver_states: coupling_reports
                .iter()
                .map(|report| {
                    report
                        .as_ref()
                        .map_or([None, None], |c| c.thermal_driver_states)
                })
                .collect(),
            driver_thermal_states: coupling_reports
                .iter()
                .map(|report| {
                    report.as_ref().map_or_else(
                        || vec![None; model.drivers().len()],
                        |c| c.driver_thermal_states.clone(),
                    )
                })
                .collect(),
            power_ledgers: coupling_reports
                .iter()
                .map(|report| report.as_ref().and_then(|c| c.power_ledger))
                .collect(),
            product_reference_sweep: None,
            structural_responses: Vec::new(),
            driver_velocities_bem: coupling_reports
                .iter()
                .zip(&states)
                .map(|(report, state)| {
                    report.as_ref().map_or_else(
                        || state.driver_velocities.to_vec(),
                        |c| c.velocities_bem_all.clone(),
                    )
                })
                .collect(),
            exact_residual_evaluations: 0,
            provenance: SimulationProvenance::synthetic(states.len(), eval_points.len()),
        },
        states,
        mesh: speaker_mesh,
    })
}

fn objective_spl_gradient(
    result: &SimulationResult,
    config: &OptimizationConfig,
) -> Result<HostMatrix<f64>, OptimizationError> {
    let mut gradient = HostMatrix::zeros(result.spl.nrows(), result.spl.ncols());
    let step_db = 1.0e-4;
    for frequency in 0..result.spl.nrows() {
        for point in 0..result.spl.ncols() {
            let mut plus = result.clone();
            let mut minus = result.clone();
            plus.spl[(frequency, point)] += step_db;
            minus.spl[(frequency, point)] -= step_db;
            gradient[(frequency, point)] = (compute_objective(&plus, config)?
                - compute_objective(&minus, config)?)
                / (2.0 * step_db);
        }
    }
    Ok(gradient)
}

fn field_objective_state_derivative(
    state: &DenseFrequencyState,
    spl_gradient: &[f64],
) -> Vec<Complex64> {
    let mut field_derivative = vec![Complex64::new(0.0, 0.0); state.field_pressure.len()];
    let spl_scale = 20.0 / std::f64::consts::LN_10;
    for (index, (&pressure, &gradient)) in state.field_pressure.iter().zip(spl_gradient).enumerate()
    {
        field_derivative[index] =
            pressure * (spl_scale * gradient / pressure.norm_sqr().max(1.0e-30));
    }
    let mut state_derivative = vec![Complex64::new(0.0, 0.0); state.field_operator.ncols()];
    state
        .field_operator
        .matvec_hermitian_into(&field_derivative, &mut state_derivative);
    state_derivative
}

struct HeldStateEvaluation {
    result: SimulationResult,
    residuals: Vec<Vec<Complex64>>,
}

/// Mesh elements whose geometry changes under a shape perturbation.
///
/// An element is affected when any of its nodes moved or any of its node
/// normals changed (cone/waveguide normals can tilt without the node moving,
/// e.g. a cone-depth change at a fixed rim). Node indices are stable under
/// the infinitesimal steps used here: the mesh generator is deterministic,
/// and untouched nodes are recomputed through identical arithmetic, so they
/// compare bit-identical. The tolerances sit far below the 0.1 µm-scale
/// perturbation displacements and far above floating-point noise.
fn affected_elements(
    baseline: &crate::mesh_gen::SpeakerMesh,
    perturbed: &crate::mesh_gen::SpeakerMesh,
) -> Vec<usize> {
    const POSITION_TOLERANCE: f64 = 1.0e-12;
    const NORMAL_TOLERANCE: f64 = 1.0e-10;
    if baseline.nodes.len() != perturbed.nodes.len() {
        // Node correspondence is broken; treat everything as affected so the
        // caller degrades to a full (but still correct-order) re-evaluation.
        return (0..perturbed.triangles.len()).collect();
    }
    let mut affected_node = vec![false; baseline.nodes.len()];
    for (index, (before, after)) in baseline.nodes.iter().zip(&perturbed.nodes).enumerate() {
        let position_moved =
            (0..3).any(|axis| (after[axis] - before[axis]).abs() > POSITION_TOLERANCE);
        let normal_changed = (0..3).any(|axis| {
            (perturbed.node_normals[index][axis] - baseline.node_normals[index][axis]).abs()
                > NORMAL_TOLERANCE
        });
        affected_node[index] = position_moved || normal_changed;
    }
    perturbed
        .triangles
        .iter()
        .enumerate()
        .filter_map(|(index, triangle)| {
            triangle
                .iter()
                .any(|&node| affected_node[node])
                .then_some(index)
        })
        .collect()
}

/// Typed-driver held-state evaluation.
///
/// The radiation matrix and the coupled driver velocities are part of the
/// design-dependent BEM right-hand side. Rebuilding only affected TBEM rows
/// is not sufficient to recover the two unit-load solves needed by the
/// network, so typed candidates use a complete perturbed forward assembly to
/// obtain those network quantities. The returned residual and field are still
/// evaluated with the *baseline* surface pressure, which is the contract
/// required by the outer BEM adjoint.
// Compat passthrough of the deprecated per-driver vectors (the driver
// matrix below is canonical).
#[allow(deprecated)]
fn evaluate_typed_geometry_with_held_state(
    model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
    solver_config: &SolverConfig,
    baseline: &DenseForwardEvaluation,
) -> Result<HeldStateEvaluation, OptimizationError> {
    let baseline_state = baseline
        .states
        .first()
        .ok_or_else(|| OptimizationError::new("typed adjoint baseline has no frequencies"))?;
    let (max_dofs, allow_hmatrix) = match &baseline_state.operator {
        ForwardOperator::Dense(_) => (baseline.mesh.triangles.len(), false),
        ForwardOperator::Hmatrix(_) => (baseline.mesh.triangles.len().saturating_sub(1), true),
    };
    let perturbed = forward_evaluation(model, eval_points, solver_config, max_dofs, allow_hmatrix)?;
    if perturbed.mesh.triangles != baseline.mesh.triangles
        || perturbed.result.frequencies != baseline.result.frequencies
    {
        return Err(OptimizationError::new(
            "typed-driver sensitivity changed BEM topology or frequency grid",
        ));
    }

    let mut spl = HostMatrix::zeros(perturbed.result.spl.nrows(), perturbed.result.spl.ncols());
    let mut complex_pressure = HostMatrix::zeros(spl.nrows(), spl.ncols());
    let mut phase = HostMatrix::zeros(
        perturbed.result.phase.nrows(),
        perturbed.result.phase.ncols(),
    );
    let mut residuals = Vec::with_capacity(baseline.states.len());
    for (index, (baseline_state, perturbed_state)) in
        baseline.states.iter().zip(&perturbed.states).enumerate()
    {
        debug_assert!(
            perturbed_state
                .driver_velocities
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        );
        let residual = perturbed_state
            .operator
            .apply(&baseline_state.surface_pressure)
            .into_iter()
            .zip(&perturbed_state.rhs)
            .map(|(lhs, &rhs)| lhs - rhs)
            .collect();
        residuals.push(residual);

        // `field_affine` contains the geometry-dependent contribution of the
        // newly coupled driver velocities. Applying the perturbed operator to
        // the held baseline pressure therefore includes both dC and dv terms.
        let mut field_pressure = perturbed_state
            .field_operator
            .matvec(&baseline_state.surface_pressure);
        for (pressure, affine) in field_pressure.iter_mut().zip(&perturbed_state.field_affine) {
            *pressure += *affine;
        }
        for (point, pressure) in field_pressure.into_iter().enumerate() {
            spl[(index, point)] = sonium_xem::pressure_to_spl(pressure);
            complex_pressure[(index, point)] = pressure;
            phase[(index, point)] = pressure.arg();
        }
    }
    Ok(HeldStateEvaluation {
        result: SimulationResult {
            frequencies: perturbed.result.frequencies,
            complex_pressure,
            spl,
            phase,
            woofer_electrical_impedance_ohm: perturbed.result.woofer_electrical_impedance_ohm,
            tweeter_electrical_impedance_ohm: perturbed.result.tweeter_electrical_impedance_ohm,
            driver_electrical_impedances_ohm: perturbed.result.driver_electrical_impedances_ohm,
            radiation_matrices: perturbed.result.radiation_matrices,
            radiation_matrix_diagnostics: perturbed.result.radiation_matrix_diagnostics,
            thermal_driver_states: perturbed.result.thermal_driver_states,
            driver_thermal_states: perturbed.result.driver_thermal_states,
            power_ledgers: perturbed.result.power_ledgers,
            // The held-state geometry is perturbed; a reference sweep must
            // be recomputed rather than copied across the derivative path.
            product_reference_sweep: None,
            structural_responses: Vec::new(),
            driver_velocities_bem: perturbed.result.driver_velocities_bem,
            exact_residual_evaluations: perturbed.result.exact_residual_evaluations,
            provenance: perturbed.result.provenance,
        },
        residuals,
    })
}

/// Sensitivity evaluation at a perturbed geometry with the forward solution
/// held fixed — LOCAL re-evaluation path.
///
/// Only the TBEM rows/columns and field-operator columns touching the mesh
/// elements the perturbation actually moved (see [`affected_elements`]) are
/// re-assembled; every other entry is provably identical to the baseline and
/// is reused. This replaces the previous O(N²) full re-assembly per
/// perturbation with O(N · N_affected) work per frequency. The full
/// re-assembly reference is retained for tests as
/// `evaluate_perturbed_geometry_with_held_state_full`.
// Prescribed-only results have no electrical model. Keep explicit None values
// in legacy scalar fields as well as the canonical driver-axis field.
#[allow(deprecated)]
fn evaluate_perturbed_geometry_with_held_state(
    model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
    solver_config: &SolverConfig,
    baseline: &DenseForwardEvaluation,
) -> Result<HeldStateEvaluation, OptimizationError> {
    if baseline
        .states
        .iter()
        .any(|state| state.radiation_matrix.is_some())
    {
        return evaluate_typed_geometry_with_held_state(
            model,
            eval_points,
            solver_config,
            baseline,
        );
    }
    let speaker_mesh = crate::mesh_gen::generate_loudspeaker_mesh(model);
    let (nodes, mut elements) = simulation::speaker_mesh_to_bem(&speaker_mesh, model)?;
    if elements.len() != baseline.states[0].surface_pressure.len() {
        return Err(OptimizationError::new(
            "shape perturbation changed BEM topology; reduce the sensitivity step or freeze meshing",
        ));
    }
    if speaker_mesh.triangles != baseline.mesh.triangles {
        // A Delaunay diagonal flip keeps the triangle count but breaks the
        // DOF↔element correspondence the held-state residual relies on.
        // Fail loudly so the caller falls back to central differences
        // instead of differentiating across a connectivity change.
        return Err(OptimizationError::new(
            "shape perturbation flipped baffle triangulation connectivity; \
             reduce the sensitivity step or freeze meshing",
        ));
    }
    let frequencies = model.simulation.frequencies();
    if frequencies != baseline.result.frequencies {
        return Err(OptimizationError::new(
            "shape perturbation changed the optimization frequency grid",
        ));
    }
    let affected = affected_elements(&baseline.mesh, &speaker_mesh);
    let mut is_affected = vec![false; elements.len()];
    for &index in &affected {
        is_affected[index] = true;
    }
    let unaffected: Vec<usize> = (0..elements.len())
        .filter(|&index| !is_affected[index])
        .collect();
    // Baseline elements are rebuilt (cheap, no assembly) so the restricted
    // column evaluations can form exact geometry deltas against the side.
    let (baseline_nodes, mut baseline_elements) =
        simulation::speaker_mesh_to_bem(&baseline.mesh, model)?;

    let mut spl = HostMatrix::zeros(frequencies.len(), eval_points.len());
    let mut complex_pressure = HostMatrix::zeros(spl.nrows(), spl.ncols());
    let mut phase = HostMatrix::zeros(frequencies.len(), eval_points.len());
    let mut residuals = Vec::with_capacity(frequencies.len());
    for (frequency_index, (&frequency_hz, baseline_state)) in
        frequencies.iter().zip(&baseline.states).enumerate()
    {
        simulation::update_driver_bcs(&mut elements, &speaker_mesh, model, frequency_hz, None)?;
        simulation::update_driver_bcs(
            &mut baseline_elements,
            &baseline.mesh,
            model,
            frequency_hz,
            None,
        )?;
        let physics = PhysicsParams::new(
            frequency_hz,
            model.simulation.speed_of_sound,
            model.simulation.air_density,
            false,
        );
        let pressure = &baseline_state.surface_pressure;

        // TBEM residual: full rows for affected elements (their collocation
        // points moved), affected columns only for the remaining rows.
        let side_rows = eval_tbem_rows_stabilized(
            &elements,
            &nodes,
            &physics,
            simulation::EXTERIOR_TBEM_SIGN,
            solver_config.beta_scale,
            &affected,
            None,
        );
        let side_columns = eval_tbem_rows_stabilized(
            &elements,
            &nodes,
            &physics,
            simulation::EXTERIOR_TBEM_SIGN,
            solver_config.beta_scale,
            &unaffected,
            Some(&affected),
        );
        let baseline_columns = eval_tbem_rows_stabilized(
            &baseline_elements,
            &baseline_nodes,
            &physics,
            simulation::EXTERIOR_TBEM_SIGN,
            solver_config.beta_scale,
            &unaffected,
            Some(&affected),
        );

        let mut residual = baseline_state.operator.apply(pressure);
        for (value, &rhs) in residual.iter_mut().zip(&baseline_state.rhs) {
            *value -= rhs;
        }
        for row in &side_rows {
            let mut applied = Complex64::new(0.0, 0.0);
            for &(column, coefficient) in &row.entries {
                applied += coefficient * pressure[column];
            }
            residual[row.row_dof] = applied - row.rhs;
        }
        for (side_row, baseline_row) in side_columns.iter().zip(&baseline_columns) {
            debug_assert_eq!(side_row.row_dof, baseline_row.row_dof);
            debug_assert_eq!(side_row.entries.len(), baseline_row.entries.len());
            let mut applied_delta = Complex64::new(0.0, 0.0);
            for (&(column, side_coeff), &(baseline_column, baseline_coeff)) in
                side_row.entries.iter().zip(&baseline_row.entries)
            {
                debug_assert_eq!(column, baseline_column);
                applied_delta += (side_coeff - baseline_coeff) * pressure[column];
            }
            residual[side_row.row_dof] += applied_delta - (side_row.rhs - baseline_row.rhs);
        }
        residuals.push(residual);

        // Field operator: recompute only the affected source-element columns.
        let velocity = simulation::build_surface_velocity(&elements);
        let baseline_velocity = simulation::build_surface_velocity(&baseline_elements);
        let (side_field_columns, side_affine) = eval_constant_field_operator_columns(
            eval_points,
            &elements,
            &nodes,
            Some(&velocity),
            &physics,
            &affected,
        );
        let (baseline_field_columns, baseline_affine) = eval_constant_field_operator_columns(
            eval_points,
            &baseline_elements,
            &baseline_nodes,
            Some(&baseline_velocity),
            &physics,
            &affected,
        );
        let mut field_pressure = baseline_state.field_pressure.clone();
        for point in 0..eval_points.len() {
            let mut delta = side_affine[point] - baseline_affine[point];
            for (position, &column) in affected.iter().enumerate() {
                delta += (side_field_columns[position][point]
                    - baseline_field_columns[position][point])
                    * pressure[column];
            }
            field_pressure[point] += delta;
        }
        for (point_index, value) in field_pressure.into_iter().enumerate() {
            spl[(frequency_index, point_index)] = sonium_xem::pressure_to_spl(value);
            complex_pressure[(frequency_index, point_index)] = value;
            phase[(frequency_index, point_index)] = value.arg();
        }
    }
    Ok(HeldStateEvaluation {
        result: SimulationResult {
            frequencies,
            complex_pressure,
            spl,
            phase,
            woofer_electrical_impedance_ohm: vec![None; baseline.states.len()],
            tweeter_electrical_impedance_ohm: vec![None; baseline.states.len()],
            driver_electrical_impedances_ohm: vec![
                vec![None; model.drivers().len()];
                baseline.states.len()
            ],
            radiation_matrices: vec![None; baseline.states.len()],
            radiation_matrix_diagnostics: vec![None; baseline.states.len()],
            thermal_driver_states: vec![[None, None]; baseline.states.len()],
            driver_thermal_states: vec![vec![None; model.drivers().len()]; baseline.states.len()],
            power_ledgers: vec![None; baseline.states.len()],
            product_reference_sweep: None,
            structural_responses: Vec::new(),
            driver_velocities_bem: baseline
                .result
                .frequencies
                .iter()
                .map(|&frequency| model.velocities_at(frequency))
                .collect(),
            exact_residual_evaluations: 0,
            provenance: SimulationProvenance::synthetic(baseline.states.len(), eval_points.len()),
        },
        residuals,
    })
}

// Fixture fills the deprecated per-driver vectors alongside the matrix.
#[allow(deprecated)]
/// Full-reassembly reference implementation of
/// [`evaluate_perturbed_geometry_with_held_state`], retained for validation
/// tests and before/after timing. Rebuilds the complete TBEM system and
/// field operator at the perturbed geometry at every frequency.
#[cfg(test)]
fn evaluate_perturbed_geometry_with_held_state_full(
    model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
    solver_config: &SolverConfig,
    baseline: &DenseForwardEvaluation,
) -> Result<HeldStateEvaluation, OptimizationError> {
    if baseline
        .states
        .iter()
        .any(|state| state.radiation_matrix.is_some())
    {
        return evaluate_typed_geometry_with_held_state(
            model,
            eval_points,
            solver_config,
            baseline,
        );
    }
    let speaker_mesh = crate::mesh_gen::generate_loudspeaker_mesh(model);
    let (nodes, mut elements) = simulation::speaker_mesh_to_bem(&speaker_mesh, model)?;
    if elements.len() != baseline.states[0].surface_pressure.len()
        || speaker_mesh.triangles != baseline.mesh.triangles
    {
        return Err(OptimizationError::new(
            "shape perturbation changed BEM topology; reduce the sensitivity step or freeze meshing",
        ));
    }
    let frequencies = model.simulation.frequencies();
    if frequencies != baseline.result.frequencies {
        return Err(OptimizationError::new(
            "shape perturbation changed the optimization frequency grid",
        ));
    }
    let mut spl = HostMatrix::zeros(frequencies.len(), eval_points.len());
    let mut complex_pressure = HostMatrix::zeros(spl.nrows(), spl.ncols());
    let mut phase = HostMatrix::zeros(frequencies.len(), eval_points.len());
    let mut residuals = Vec::with_capacity(frequencies.len());
    for (frequency_index, (&frequency_hz, baseline_state)) in
        frequencies.iter().zip(&baseline.states).enumerate()
    {
        simulation::update_driver_bcs(&mut elements, &speaker_mesh, model, frequency_hz, None)?;
        let physics = PhysicsParams::new(
            frequency_hz,
            model.simulation.speed_of_sound,
            model.simulation.air_density,
            false,
        );
        let system = build_tbem_system_stabilized(
            &elements,
            &nodes,
            &physics,
            simulation::EXTERIOR_TBEM_SIGN,
            solver_config.beta_scale,
        );
        let residual = system
            .matrix
            .matvec(&baseline_state.surface_pressure)
            .into_iter()
            .zip(&system.rhs)
            .map(|(lhs, &rhs)| lhs - rhs)
            .collect();
        residuals.push(residual);
        let velocity = simulation::build_surface_velocity(&elements);
        let (field_operator, affine) = build_constant_field_operator(
            eval_points,
            &elements,
            &nodes,
            Some(&velocity),
            &physics,
        );
        let mut pressure = field_operator.matvec(&baseline_state.surface_pressure);
        for (value, affine_value) in pressure.iter_mut().zip(affine) {
            *value += affine_value;
        }
        for (point_index, value) in pressure.into_iter().enumerate() {
            spl[(frequency_index, point_index)] = sonium_xem::pressure_to_spl(value);
            complex_pressure[(frequency_index, point_index)] = value;
            phase[(frequency_index, point_index)] = value.arg();
        }
    }
    Ok(HeldStateEvaluation {
        result: SimulationResult {
            frequencies,
            complex_pressure,
            spl,
            phase,
            woofer_electrical_impedance_ohm: vec![None; baseline.states.len()],
            tweeter_electrical_impedance_ohm: vec![None; baseline.states.len()],
            driver_electrical_impedances_ohm: vec![
                vec![None; model.drivers().len()];
                baseline.states.len()
            ],
            radiation_matrices: vec![None; baseline.states.len()],
            radiation_matrix_diagnostics: vec![None; baseline.states.len()],
            thermal_driver_states: vec![[None, None]; baseline.states.len()],
            driver_thermal_states: vec![vec![None; model.drivers().len()]; baseline.states.len()],
            power_ledgers: vec![None; baseline.states.len()],
            product_reference_sweep: None,
            structural_responses: Vec::new(),
            driver_velocities_bem: baseline
                .result
                .frequencies
                .iter()
                .map(|&frequency| model.velocities_at(frequency))
                .collect(),
            exact_residual_evaluations: 0,
            provenance: SimulationProvenance::synthetic(baseline.states.len(), eval_points.len()),
        },
        residuals,
    })
}

fn discrete_adjoint_gradient(
    x: &[f64],
    base_model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
    solver_config: &SolverConfig,
    config: &OptimizationConfig,
    bounds: &[Bound<f64>],
) -> Result<Vec<f64>, OptimizationError> {
    discrete_adjoint_gradient_impl(
        x,
        base_model,
        eval_points,
        solver_config,
        config,
        bounds,
        &unpack_parameters,
        evaluate_perturbed_geometry_with_held_state,
    )
}

/// `discrete_adjoint_gradient` with the full-reassembly sensitivity
/// reference, for validation tests and before/after timing.
#[cfg(test)]
fn discrete_adjoint_gradient_reference(
    x: &[f64],
    base_model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
    solver_config: &SolverConfig,
    config: &OptimizationConfig,
    bounds: &[Bound<f64>],
) -> Result<Vec<f64>, OptimizationError> {
    discrete_adjoint_gradient_impl(
        x,
        base_model,
        eval_points,
        solver_config,
        config,
        bounds,
        &unpack_parameters,
        evaluate_perturbed_geometry_with_held_state_full,
    )
}

/// Sensitivity-evaluation backend used by `discrete_adjoint_gradient_impl`:
/// the local partial re-assembly in production, the full-reassembly
/// reference in tests.
type HeldStateEvalFn = fn(
    &LoudspeakerModel,
    &[[f64; 3]],
    &SolverConfig,
    &DenseForwardEvaluation,
) -> Result<HeldStateEvaluation, OptimizationError>;

#[allow(clippy::too_many_arguments)]
fn discrete_adjoint_gradient_impl<F>(
    x: &[f64],
    base_model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
    solver_config: &SolverConfig,
    config: &OptimizationConfig,
    bounds: &[Bound<f64>],
    unpack: &F,
    held_state_eval: HeldStateEvalFn,
) -> Result<Vec<f64>, OptimizationError>
where
    F: Fn(&[f64], &LoudspeakerModel) -> Result<LoudspeakerModel, OptimizationError>,
{
    let model = unpack(x, base_model)?;
    let baseline = forward_evaluation(
        &model,
        eval_points,
        solver_config,
        config.adjoint_max_dofs,
        config.adjoint_hmatrix,
    )?;
    log::info!(
        "discrete speaker adjoint: {} DOFs x {} frequencies",
        baseline
            .states
            .first()
            .map_or(0, |state| state.surface_pressure.len()),
        baseline.states.len()
    );
    let spl_gradient = objective_spl_gradient(&baseline.result, config)?;
    let mut adjoints = Vec::with_capacity(baseline.states.len());
    let mut adjoint_sentinel = simulation::ExactResidualSentinel::default();
    for (frequency_index, state) in baseline.states.iter().enumerate() {
        let state_derivative =
            field_objective_state_derivative(state, spl_gradient.row(frequency_index));
        let adjoint_rhs = state_derivative
            .iter()
            .map(|value| -*value)
            .collect::<Vec<_>>();
        // Reuses the forward LU factors / ILU preconditioner retained per
        // frequency on the dense path; the H-matrix path solves Aᴴ through
        // the operator's Hermitian matvec. Both keep the same
        // true-residual gates.
        let adjoint = match (&state.operator, &state.solver_state) {
            (ForwardOperator::Dense(matrix), Some(solver_state)) => {
                simulation::solve_bem_adjoint_system(
                    matrix,
                    &adjoint_rhs,
                    solver_state,
                    solver_config,
                )
                .map_err(OptimizationError::from)?
            }
            (ForwardOperator::Hmatrix(hmatrix), _) => {
                let (adjoint, residual) = simulation::solve_bem_adjoint_system_hmatrix(
                    hmatrix,
                    &adjoint_rhs,
                    solver_config,
                )
                .map_err(OptimizationError::from)?;
                if adjoint_sentinel.should_verify_exact(solver_config, residual) {
                    // Rebuild the BEM geometry from the retained baseline
                    // mesh: the operator depends on BC *types* (all Velocity
                    // here), not on the prescribed values, so a fresh
                    // conversion with this frequency's BCs applied reproduces
                    // the exact forward operator.
                    let frequency_hz = baseline.result.frequencies[frequency_index];
                    let (exact_nodes, mut exact_elements) =
                        simulation::speaker_mesh_to_bem(&baseline.mesh, &model)?;
                    simulation::update_driver_bcs(
                        &mut exact_elements,
                        &baseline.mesh,
                        &model,
                        frequency_hz,
                        None,
                    )?;
                    let physics = PhysicsParams::new(
                        frequency_hz,
                        model.simulation.speed_of_sound,
                        model.simulation.air_density,
                        false,
                    );
                    simulation::verify_exact_hmatrix_adjoint_solution(
                        &exact_elements,
                        &exact_nodes,
                        &physics,
                        &adjoint,
                        &adjoint_rhs,
                        solver_config,
                    )
                    .map_err(OptimizationError::from)?;
                }
                adjoint
            }
            (ForwardOperator::Dense(_), None) => {
                unreachable!("dense adjoint forward state always retains its solver state")
            }
        };
        adjoints.push(adjoint);
    }

    let baseline_objective = compute_objective(&baseline.result, config)?;
    let baseline_residuals = baseline
        .states
        .iter()
        .map(|state| {
            state
                .operator
                .apply(&state.surface_pressure)
                .into_iter()
                .zip(&state.rhs)
                .map(|(lhs, &rhs)| lhs - rhs)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    // Geometry kernels are smooth but strongly scaled in metres; a 0.1 µm
    // perturbation keeps the procedural mesh topology fixed while remaining
    // comfortably above floating-point cancellation in Green kernels.
    let base_step = 1.0e-7;
    let mut gradient = vec![0.0; x.len()];
    for parameter in 0..x.len() {
        let nominal_step = base_step * x[parameter].abs().max(1.0);
        let lower_room = bounds[parameter]
            .lower
            .map_or(f64::INFINITY, |lower| x[parameter] - lower);
        let upper_room = bounds[parameter]
            .upper
            .map_or(f64::INFINITY, |upper| upper - x[parameter]);
        let use_central = lower_room >= nominal_step && upper_room >= nominal_step;
        let (plus_step, minus_step) = if use_central {
            (nominal_step, nominal_step)
        } else if upper_room >= nominal_step {
            (nominal_step, 0.0)
        } else if lower_room >= nominal_step {
            (0.0, nominal_step)
        } else {
            return Err(OptimizationError::new(format!(
                "parameter {parameter} has no room for an adjoint geometry sensitivity step"
            )));
        };

        let plus = if plus_step > 0.0 {
            let mut perturbed = x.to_vec();
            perturbed[parameter] += plus_step;
            Some(held_state_eval(
                &unpack(&perturbed, base_model)?,
                eval_points,
                solver_config,
                &baseline,
            )?)
        } else {
            None
        };
        let minus = if minus_step > 0.0 {
            let mut perturbed = x.to_vec();
            perturbed[parameter] -= minus_step;
            Some(held_state_eval(
                &unpack(&perturbed, base_model)?,
                eval_points,
                solver_config,
                &baseline,
            )?)
        } else {
            None
        };

        let denominator = plus_step + minus_step;
        let explicit = match (&plus, &minus) {
            (Some(plus), Some(minus)) => {
                (compute_objective(&plus.result, config)?
                    - compute_objective(&minus.result, config)?)
                    / denominator
            }
            (Some(plus), None) => {
                (compute_objective(&plus.result, config)? - baseline_objective) / denominator
            }
            (None, Some(minus)) => {
                (baseline_objective - compute_objective(&minus.result, config)?) / denominator
            }
            (None, None) => unreachable!(),
        };
        let mut sensitivity = explicit;
        for frequency in 0..baseline.states.len() {
            let derivative = match (&plus, &minus) {
                (Some(plus), Some(minus)) => plus.residuals[frequency]
                    .iter()
                    .zip(&minus.residuals[frequency])
                    .map(|(&positive, &negative)| (positive - negative) / denominator)
                    .collect::<Vec<_>>(),
                (Some(plus), None) => plus.residuals[frequency]
                    .iter()
                    .zip(&baseline_residuals[frequency])
                    .map(|(&positive, &base)| (positive - base) / denominator)
                    .collect(),
                (None, Some(minus)) => baseline_residuals[frequency]
                    .iter()
                    .zip(&minus.residuals[frequency])
                    .map(|(&base, &negative)| (base - negative) / denominator)
                    .collect(),
                (None, None) => unreachable!(),
            };
            let contribution = adjoints[frequency]
                .iter()
                .zip(&derivative)
                .map(|(adjoint, residual_derivative)| (adjoint.conj() * *residual_derivative).re)
                .sum::<f64>();
            sensitivity += contribution;
        }
        gradient[parameter] = sensitivity;
    }
    Ok(gradient)
}

/// Reuse the dense discrete-adjoint machinery with a staged parameterization.
///
/// The caller owns packing and bounds; this keeps the legacy public API and
/// parameter layout unchanged while allowing the five normalized waveguide
/// coordinates to drive the same element-mapped adjoint sensitivities.
pub(super) fn discrete_adjoint_gradient_for_parameterization<F>(
    x: &[f64],
    base_model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
    solver_config: &SolverConfig,
    config: &OptimizationConfig,
    bounds: &[Bound<f64>],
    unpack: F,
) -> Result<Vec<f64>, OptimizationError>
where
    F: Fn(&[f64], &LoudspeakerModel) -> Result<LoudspeakerModel, OptimizationError>,
{
    discrete_adjoint_gradient_impl(
        x,
        base_model,
        eval_points,
        solver_config,
        config,
        bounds,
        &unpack,
        evaluate_perturbed_geometry_with_held_state,
    )
}

/// Map the configured gradient backend plus the recorded adjoint fallback
/// reason (if any) to the backend that actually drove the search.
fn applied_gradient_method(
    configured: OptimizationGradientMethod,
    fallback_reason: Option<String>,
) -> AppliedGradientMethod {
    match configured {
        OptimizationGradientMethod::CentralDifference => AppliedGradientMethod::CentralDifference,
        OptimizationGradientMethod::DiscreteAdjoint => match fallback_reason {
            Some(reason) => AppliedGradientMethod::AdjointWithFiniteDifferenceFallback { reason },
            None => AppliedGradientMethod::DiscreteAdjoint,
        },
    }
}

// ---------------------------------------------------------------------------
// Main optimisation entry point
// ---------------------------------------------------------------------------

/// Run loudspeaker optimisation.
pub fn optimize(
    model: &LoudspeakerModel,
    config: &OptimizationConfig,
) -> Result<OptimizationResult, OptimizationError> {
    validate_optimization_setup(model, config)?;

    // Build a model variant with coarser settings for optimisation
    let mut optim_model = model.clone();
    optim_model.simulation.freq_points = config.optim_freq_points;
    optim_model.simulation.freq_end = optim_model
        .simulation
        .freq_end
        .min(config.optim_max_mesh_freq);
    optim_model.simulation.max_mesh_frequency = Some(config.optim_max_mesh_freq);
    optim_model.validate().map_err(|message| {
        OptimizationError::new(format!("invalid optimisation model: {message}"))
    })?;

    let eval_points = cea2034_output::generate_eval_points(optim_model.baffle_center());
    let solver_config = optimization_solver_config(config.exact_residual_policy, false);

    let x0 = pack_parameters(&optim_model)?;
    let bounds = compute_bounds(&optim_model)?;
    for (index, (value, bound)) in x0.iter().zip(bounds.iter()).enumerate() {
        if bound.lower.is_some_and(|lower| *value < lower)
            || bound.upper.is_some_and(|upper| *value > upper)
        {
            return Err(OptimizationError::new(format!(
                "initial optimisation parameter {index}={value} is outside its bounds"
            )));
        }
    }

    // The objective and constraint share one simulation cache. Augmented
    // Lagrangian evaluates them independently at the same x; using separate
    // closures previously doubled every full BEM sweep.
    let cache = Arc::new(Mutex::new(SimCache::new()));
    let failure_bounds = bounds.clone();
    let gradient_bounds = bounds.clone();

    let objective = |x: &[f64]| -> f64 {
        let mut cache = cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        cache.get_or_compute(x, &optim_model, &eval_points, &solver_config);
        match cache.last_result.as_ref() {
            Some(Ok(result)) => compute_objective(result, config)
                .unwrap_or_else(|_| failure_objective(x, &failure_bounds)),
            Some(Err(_)) | None => failure_objective(x, &failure_bounds),
        }
    };

    // Records the first adjoint failure so the silent adjoint→FD fallback is
    // visible in the returned OptimizationResult.
    let adjoint_fallback_reason = Arc::new(Mutex::new(None::<String>));
    let gradient_cache = Arc::clone(&cache);
    let grad_fn = |x: &[f64]| -> Vec<f64> {
        // Finite-difference gradients evaluate 2n independent candidates;
        // schedule them as one batch over candidate lanes, seeded with the
        // warm start of the most recent shared-cache evaluation (the base
        // point x in the augmented-Lagrangian flow). The shared cache itself
        // is left untouched: its warm start keeps referring to x, which is
        // what the next objective/constraint evaluation wants.
        let fd_gradient = |x: &[f64]| -> Vec<f64> {
            let h = f64::EPSILON.powf(1.0 / 3.0);
            let warm_start = gradient_cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .warm_start_snapshot();
            gradient_central_batched(
                x,
                h,
                &optim_model,
                &eval_points,
                &solver_config,
                config,
                &gradient_bounds,
                Some(&warm_start),
            )
        };
        match config.gradient_method {
            OptimizationGradientMethod::DiscreteAdjoint => discrete_adjoint_gradient(
                x,
                &optim_model,
                &eval_points,
                &solver_config,
                config,
                &gradient_bounds,
            )
            .unwrap_or_else(|error| {
                log::warn!("discrete speaker adjoint unavailable ({error}); using central FD");
                let mut reason = adjoint_fallback_reason
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                reason.get_or_insert_with(|| error.to_string());
                fd_gradient(x)
            }),
            OptimizationGradientMethod::CentralDifference => fd_gradient(x),
        }
    };

    // PIR slope constraint: c(x) >= 0
    let constraint = Constraint {
        kind: ConstraintKind::Inequality,
        dim: 1,
        fun: Box::new({
            let optim_model = optim_model.clone();
            let eval_points = eval_points.clone();
            let solver_config = solver_config.clone();
            let cache = Arc::clone(&cache);
            move |x: &[f64]| {
                let mut cache = cache
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                cache.get_or_compute(x, &optim_model, &eval_points, &solver_config);
                let c = match cache.last_result.as_ref() {
                    Some(Ok(result)) => {
                        pir_slope_constraint(result).unwrap_or(CONSTRAINT_FAILURE_PENALTY)
                    }
                    Some(Err(_)) | None => CONSTRAINT_FAILURE_PENALTY,
                };
                vec![c]
            }
        }),
        jac: None,
    };

    let al_config = AugmentedLagrangianConfig {
        max_outer_iterations: config.max_iterations,
        max_inner_iterations: 10, // few inner iterations (expensive objective)
        bounds,
        inner_solver: InnerSolver::Lbfgsb,
        ..Default::default()
    };

    log::info!("Starting optimisation with {} parameters", x0.len());

    let opt = augmented_lagrangian(&objective, &grad_fn, &x0, &[constraint], &al_config)
        .map_err(|error| OptimizationError::new(format!("optimisation solver failed: {error}")))?;
    if !opt.fun.is_finite() || opt.fun >= OBJECTIVE_FAILURE_PENALTY {
        return Err(OptimizationError::new(format!(
            "optimisation ended with an invalid objective ({})",
            opt.fun
        )));
    }
    let final_model = unpack_parameters(&opt.x, model)?;
    let gradient_method = applied_gradient_method(
        config.gradient_method,
        adjoint_fallback_reason
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take(),
    );
    // The search ran on a truncated band and coarse mesh; re-evaluate the
    // final candidate over the full production band so the report includes an
    // objective that is directly comparable to production runs.
    let production_eval_points = cea2034_output::generate_eval_points(final_model.baffle_center());
    let production_solver_config = optimization_solver_config(config.exact_residual_policy, true);
    let production_result = simulation::run_simulation_with_config(
        &final_model,
        &production_eval_points,
        &production_solver_config,
    )
    .map_err(OptimizationError::from)?;
    let full_band_objective = compute_objective(&production_result, config)?;
    log::info!(
        "Optimisation finished: objective={:.6} (search band), full-band={:.6}, gradient_method={gradient_method:?}, converged={}",
        opt.fun,
        full_band_objective,
        opt.converged
    );
    Ok(OptimizationResult {
        model: final_model,
        objective: opt.fun,
        full_band_objective,
        gradient_method,
        iterations: opt.iterations,
        converged: opt.converged,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_port_adjoint_rejects_additional_driver_state() {
        let mut model = simulation::tests::tiny_model();
        model.additional_drivers.push(model.woofer.clone());
        let error = match dense_forward_evaluation(
            &model,
            &[[0.0, 0.0, 1.0]],
            &SolverConfig::default(),
            2_000,
        ) {
            Err(error) => error,
            Ok(_) => panic!("additional driver was silently omitted"),
        };
        assert!(error.to_string().contains("additional drivers"));
    }

    #[test]
    fn optimization_outputs_preserve_complex_pressure() {
        let mut model = simulation::tests::tiny_model();
        model.simulation.max_mesh_frequency = Some(700.0);
        model.simulation.freq_start = 500.0;
        model.simulation.freq_end = 500.0;
        let points = [[0.0, 0.0, 1.0]];
        let config = SolverConfig::default();
        let forward = dense_forward_evaluation(&model, &points, &config, 2_000).unwrap();
        let held = evaluate_perturbed_geometry_with_held_state(&model, &points, &config, &forward)
            .unwrap();
        let full =
            evaluate_perturbed_geometry_with_held_state_full(&model, &points, &config, &forward)
                .unwrap();
        for result in [&forward.result, &held.result, &full.result] {
            assert!(result.complex_pressure[(0, 0)].norm() > 0.0);
            for f in 0..result.frequencies.len() {
                let pressure = result.complex_pressure[(f, 0)];
                assert_eq!(sonium_xem::pressure_to_spl(pressure), result.spl[(f, 0)]);
                assert_eq!(pressure.arg(), result.phase[(f, 0)]);
            }
        }
    }
    use crate::model::*;
    use sonium_solvers::optimization::util::finite_diff::gradient_central;

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
                freq_end: 5000.0,
                freq_points: 10,
                speed_of_sound: 343.0,
                air_density: 1.21,
                max_mesh_frequency: Some(5000.0),
            },
        }
    }

    #[test]
    fn test_pack_unpack_roundtrip() {
        let mut model = sample_model();
        model.woofer.cone = Some(ConeConfig {
            cone_depth: 0.012,
            dustcap_radius: 0.018,
        });
        model.tweeter.cone = Some(ConeConfig {
            cone_depth: 0.003,
            dustcap_radius: 0.005,
        });
        let x = pack_parameters(&model).expect("valid model should pack");
        let restored = unpack_parameters(&x, &model).expect("valid parameters should unpack");

        assert!((restored.woofer.y_offset - model.woofer.y_offset).abs() < 1e-15);
        assert!((restored.tweeter.y_offset - model.tweeter.y_offset).abs() < 1e-15);
        for (restored_driver, model_driver) in [
            (&restored.woofer, &model.woofer),
            (&restored.tweeter, &model.tweeter),
        ] {
            let restored_cone = restored_driver.cone.unwrap();
            let model_cone = model_driver.cone.unwrap();
            assert!((restored_cone.cone_depth - model_cone.cone_depth).abs() < 1e-15);
            assert!((restored_cone.dustcap_radius - model_cone.dustcap_radius).abs() < 1e-15);
        }
        let restored_wg = restored.waveguide.as_ref().unwrap();
        let model_wg = model.waveguide.as_ref().unwrap();
        assert!((restored_wg.depth - model_wg.depth).abs() < 1e-15);
        for (a, b) in restored_wg
            .horizontal_spline
            .iter()
            .zip(model_wg.horizontal_spline.iter())
        {
            assert!((a[1] - b[1]).abs() < 1e-15);
        }
    }

    #[test]
    fn test_bounds_count() {
        let model = sample_model();
        let x = pack_parameters(&model).expect("valid model should pack");
        let bounds = compute_bounds(&model).expect("valid model should have bounds");
        assert_eq!(bounds.len(), x.len());
    }

    #[test]
    fn test_initial_point_within_bounds() {
        let model = sample_model();
        let x = pack_parameters(&model).expect("valid model should pack");
        let bounds = compute_bounds(&model).expect("valid model should have bounds");
        for (i, (xi, b)) in x.iter().zip(bounds.iter()).enumerate() {
            if let Some(lb) = b.lower {
                assert!(*xi >= lb - 1e-10, "param {i}: {xi} < lower bound {lb}");
            }
            if let Some(ub) = b.upper {
                assert!(*xi <= ub + 1e-10, "param {i}: {xi} > upper bound {ub}");
            }
        }
    }

    #[test]
    fn test_max_octave_slope() {
        // Flat response → 0 dB/oct
        let freqs = vec![100.0, 200.0, 400.0, 800.0];
        let spl = vec![80.0, 80.0, 80.0, 80.0];
        assert!(compute_max_octave_slope(&freqs, &spl).unwrap() < 0.01);

        // 3 dB/oct tilt
        let spl_tilt = vec![80.0, 83.0, 86.0, 89.0];
        let slope = compute_max_octave_slope(&freqs, &spl_tilt).unwrap();
        assert!((slope - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_optimization_requires_waveguide() {
        let mut model = sample_model();
        model.waveguide = None;

        assert!(optimize(&model, &OptimizationConfig::default()).is_err());
        assert!(pack_parameters(&model).is_err());
        assert!(compute_bounds(&model).is_err());
    }

    #[test]
    fn test_optimization_config_rejects_invalid_values() {
        let model = sample_model();
        let invalid = [
            OptimizationConfig {
                max_iterations: 0,
                ..OptimizationConfig::default()
            },
            OptimizationConfig {
                optim_freq_points: 0,
                ..OptimizationConfig::default()
            },
            OptimizationConfig {
                optim_freq_points: 1,
                ..OptimizationConfig::default()
            },
            OptimizationConfig {
                weight_flat: f64::NAN,
                ..OptimizationConfig::default()
            },
            OptimizationConfig {
                weight_pir: -1.0,
                ..OptimizationConfig::default()
            },
            OptimizationConfig {
                weight_flat: 0.0,
                weight_pir: 0.0,
                ..OptimizationConfig::default()
            },
            OptimizationConfig {
                optim_max_mesh_freq: f64::INFINITY,
                ..OptimizationConfig::default()
            },
        ];

        for config in &invalid {
            assert!(validate_optimization_setup(&model, config).is_err());
        }
    }

    #[test]
    fn test_optimization_accepts_coarse_search_mesh() {
        let model = sample_model();
        let config = OptimizationConfig {
            optim_max_mesh_freq: model.simulation.freq_end * 0.5,
            ..OptimizationConfig::default()
        };
        assert!(validate_optimization_setup(&model, &config).is_ok());
    }

    #[test]
    fn test_unpack_parameters_rejects_malformed_vectors() {
        let model = sample_model();
        let x = pack_parameters(&model).unwrap();

        assert!(unpack_parameters(&x[..x.len() - 1], &model).is_err());

        let mut too_long = x.clone();
        too_long.push(0.0);
        assert!(unpack_parameters(&too_long, &model).is_err());

        let mut non_finite = x;
        non_finite[0] = f64::NAN;
        assert!(unpack_parameters(&non_finite, &model).is_err());
    }

    #[test]
    fn test_octave_slope_rejects_invalid_grids() {
        assert!(compute_max_octave_slope(&[100.0], &[80.0]).is_err());
        assert!(compute_max_octave_slope(&[100.0, 200.0], &[80.0]).is_err());
        assert!(compute_max_octave_slope(&[100.0, 100.0], &[80.0, 81.0]).is_err());
        assert!(compute_max_octave_slope(&[100.0, 200.0], &[80.0, f64::NAN]).is_err());
    }

    #[test]
    fn test_smooth_f3_weights_follow_rolloff_without_index_jump() {
        let on_axis = [70.0, 76.0, 82.0, 85.0, 85.0]; // rolloff below 400 Hz
        let weights = smooth_above_f3_weights(&on_axis).unwrap();
        assert!(weights[0] < 1.0e-6);
        assert!(weights[1] < 1.0e-4);
        assert!(weights[2] > 0.1 && weights[2] < 0.9);
        assert!(weights[3] > 0.99 && weights[4] > 0.99);

        let mut perturbed = on_axis;
        perturbed[2] += 1.0e-6;
        let perturbed_weights = smooth_above_f3_weights(&perturbed).unwrap();
        let max_change = weights
            .iter()
            .zip(perturbed_weights)
            .map(|(&before, after)| (before - after).abs())
            .fold(0.0, f64::max);
        assert!(
            max_change < 1.0e-5,
            "infinitesimal SPL change must not add/remove a whole F3 bin"
        );
    }

    #[test]
    fn test_failure_penalty_is_finite_and_scales_with_bound_violation() {
        let bounds = vec![Bound::both(0.0, 1.0), Bound::both(-1.0, 1.0)];
        let inside = failure_objective(&[0.5, 0.0], &bounds);
        let outside = failure_objective(&[1.5, 0.0], &bounds);
        assert!(inside.is_finite() && outside.is_finite());
        assert!(inside < outside);
        assert!(outside < 1.0e30);
    }

    #[test]
    fn optimization_exact_residual_policy_reaches_solver_config() {
        assert_eq!(
            optimization_solver_config(ExactResidualPolicy::Off, false).exact_residual_sentinel,
            SentinelPolicy::Off
        );
        assert_eq!(
            optimization_solver_config(ExactResidualPolicy::EveryEvaluation, false)
                .exact_residual_sentinel,
            SentinelPolicy::EverySolve
        );
        assert_eq!(
            optimization_solver_config(ExactResidualPolicy::FinalOnly, false)
                .exact_residual_sentinel,
            SentinelPolicy::default()
        );
        assert_eq!(
            optimization_solver_config(ExactResidualPolicy::FinalOnly, true)
                .exact_residual_sentinel,
            SentinelPolicy::EverySolve
        );
    }

    #[test]
    fn typed_driver_adjoint_forward_retains_geometry_dependent_coupling() {
        let mut model = sample_model();
        model.simulation.freq_start = 500.0;
        model.simulation.freq_end = 700.0;
        model.simulation.freq_points = 2;
        model.simulation.max_mesh_frequency = Some(700.0);
        let study =
            sonium_sim_speaker_modeling::SpeakerDriverStudy::detailed_frequency_domain_driver();
        model.woofer.radius = study.geometry.diaphragm_radius_m;
        model.woofer.driver_model =
            Some(sonium_sim_speaker_modeling::DriverModel::from_study(&study).unwrap());
        model.tweeter.piston_velocity = 0.0;

        let eval_points = cea2034_output::generate_eval_points(model.baffle_center());
        let solver_config = SolverConfig::default();
        let baseline = dense_forward_evaluation(&model, &eval_points, &solver_config, 2_000)
            .expect("typed-driver adjoint forward pass should be supported");
        for row in &baseline.result.driver_electrical_impedances_ohm {
            assert!(
                row[0].is_some(),
                "typed driver impedance must not be discarded"
            );
            assert_eq!(row[1], None, "prescribed driver has no electrical model");
        }
        assert!(
            baseline
                .result
                .radiation_matrices
                .iter()
                .all(Option::is_some)
        );
        assert!(
            baseline
                .result
                .radiation_matrix_diagnostics
                .iter()
                .all(Option::is_some)
        );
        for (row, state) in baseline
            .result
            .driver_velocities_bem
            .iter()
            .zip(&baseline.states)
        {
            assert_eq!(row.as_slice(), &state.driver_velocities);
        }
        for (frequency, state) in baseline.states.iter().enumerate() {
            for (point, &pressure) in state.field_pressure.iter().enumerate() {
                assert_eq!(
                    baseline.result.complex_pressure[(frequency, point)],
                    pressure
                );
            }
        }
        assert!(
            baseline
                .states
                .iter()
                .all(|state| state.radiation_matrix.is_some())
        );

        let x = pack_parameters(&model).unwrap();
        let mut perturbed_x = x.clone();
        perturbed_x[0] += 1.0e-7;
        let perturbed_model = unpack_parameters(&perturbed_x, &model).unwrap();
        let held = evaluate_perturbed_geometry_with_held_state(
            &perturbed_model,
            &eval_points,
            &solver_config,
            &baseline,
        )
        .expect("typed-driver held-state sensitivity should be supported");
        assert!(
            held.residuals
                .iter()
                .flatten()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        );

        let perturbed_forward =
            dense_forward_evaluation(&perturbed_model, &eval_points, &solver_config, 2_000)
                .unwrap();
        let velocity_change = baseline
            .states
            .iter()
            .zip(&perturbed_forward.states)
            .map(|(before, after)| {
                (before.driver_velocities[0] - after.driver_velocities[0]).norm()
            })
            .fold(0.0, f64::max);
        assert!(
            velocity_change > 0.0,
            "geometry perturbation must flow through the typed radiation network"
        );
    }

    #[test]
    fn typed_driver_adjoint_matches_full_simulation_fd() {
        let mut model = sample_model();
        model.simulation.freq_start = 500.0;
        model.simulation.freq_end = 700.0;
        model.simulation.freq_points = 2;
        model.simulation.max_mesh_frequency = Some(700.0);
        let study =
            sonium_sim_speaker_modeling::SpeakerDriverStudy::detailed_frequency_domain_driver();
        model.woofer.radius = study.geometry.diaphragm_radius_m;
        model.woofer.driver_model =
            Some(sonium_sim_speaker_modeling::DriverModel::from_study(&study).unwrap());
        model.tweeter.piston_velocity = 0.0;

        let config = OptimizationConfig {
            optim_freq_points: 2,
            optim_max_mesh_freq: 700.0,
            adjoint_max_dofs: 2_000,
            exact_residual_policy: ExactResidualPolicy::Off,
            ..OptimizationConfig::default()
        };
        let eval_points = cea2034_output::generate_eval_points(model.baffle_center());
        let solver_config = SolverConfig::default();
        let x = pack_parameters(&model).unwrap();
        let bounds = compute_bounds(&model).unwrap();
        let adjoint =
            discrete_adjoint_gradient(&x, &model, &eval_points, &solver_config, &config, &bounds)
                .expect("typed-driver discrete adjoint should be available");

        let objective_at = |parameters: &[f64]| {
            let candidate = unpack_parameters(parameters, &model).unwrap();
            let result =
                simulation::run_simulation_with_config(&candidate, &eval_points, &solver_config)
                    .unwrap();
            compute_objective(&result, &config).unwrap()
        };
        let step = 1.0e-7;
        let mut plus = x.clone();
        let mut minus = x;
        plus[0] += step;
        minus[0] -= step;
        let finite_difference = (objective_at(&plus) - objective_at(&minus)) / (2.0 * step);
        let scale = finite_difference.abs().max(1.0);
        assert!(
            (adjoint[0] - finite_difference).abs() / scale < 5.0e-2,
            "typed-driver adjoint={}, full-simulation FD={finite_difference}",
            adjoint[0]
        );
    }

    /// F3+F5 regression: a full `optimize` run whose adjoint is forced to fail
    /// (adjoint_max_dofs = 1 with the H-matrix adjoint disabled) must report
    /// the FD fallback in the result struct, and must re-evaluate the final
    /// candidate over the full production band.
    /// Two-point splines keep the parameter count (and hence the 2n FD cost)
    /// minimal; still expensive in debug, so run it in release.
    #[test]
    fn optimize_reports_gradient_fallback_and_full_band_objective() {
        let mut model = sample_model();
        model.waveguide = Some(WaveguideConfig {
            horizontal_spline: vec![[0.0, 0.05], [1.0, 0.013]],
            vertical_spline: vec![[0.0, 0.04], [1.0, 0.013]],
            depth: 0.025,
        });
        model.simulation.freq_start = 500.0;
        model.simulation.freq_end = 700.0;
        model.simulation.freq_points = 2;
        model.simulation.max_mesh_frequency = Some(700.0);
        let config = OptimizationConfig {
            max_iterations: 1,
            optim_freq_points: 2,
            optim_max_mesh_freq: 700.0,
            adjoint_max_dofs: 1,    // force the adjoint→FD fallback …
            adjoint_hmatrix: false, // … by refusing both dense and H-matrix adjoints
            ..OptimizationConfig::default()
        };
        let result = optimize(&model, &config).expect("optimization should succeed");
        match &result.gradient_method {
            AppliedGradientMethod::AdjointWithFiniteDifferenceFallback { reason } => {
                assert!(
                    reason.contains("adjoint"),
                    "fallback reason should name the adjoint failure: {reason}"
                );
            }
            other => panic!("expected adjoint→FD fallback to be reported, got {other:?}"),
        }
        assert!(result.objective.is_finite());
        assert!(
            result.full_band_objective.is_finite(),
            "final candidate must be re-evaluated over the full production band"
        );
    }

    #[test]
    fn discrete_bem_adjoint_matches_full_simulation_fd() {
        let mut model = sample_model();
        model.simulation.freq_start = 500.0;
        model.simulation.freq_end = 700.0;
        model.simulation.freq_points = 2;
        model.simulation.max_mesh_frequency = Some(700.0);
        let config = OptimizationConfig {
            optim_freq_points: 2,
            optim_max_mesh_freq: 700.0,
            adjoint_max_dofs: 2_000,
            ..OptimizationConfig::default()
        };
        let eval_points = cea2034_output::generate_eval_points(model.baffle_center());
        let solver_config = SolverConfig {
            direct_threshold: 2_000,
            ..SolverConfig::default()
        };
        let x = pack_parameters(&model).unwrap();
        let bounds = compute_bounds(&model).unwrap();
        let dense_baseline =
            dense_forward_evaluation(&model, &eval_points, &solver_config, 2_000).unwrap();
        let production_baseline =
            simulation::run_simulation_with_config(&model, &eval_points, &solver_config).unwrap();
        let baseline_spl_difference = dense_baseline
            .result
            .spl
            .as_slice()
            .iter()
            .zip(production_baseline.spl.as_slice())
            .map(|(&dense, &production)| (dense - production).abs())
            .fold(0.0, f64::max);
        assert!(
            baseline_spl_difference < 1.0e-10,
            "adjoint forward path differs from production by {baseline_spl_difference} dB"
        );
        let adjoint =
            discrete_adjoint_gradient(&x, &model, &eval_points, &solver_config, &config, &bounds)
                .unwrap();

        let parameter = 0;
        let step = 1.0e-7;
        let objective_at = |parameters: &[f64]| {
            let candidate = unpack_parameters(parameters, &model).unwrap();
            let result =
                simulation::run_simulation_with_config(&candidate, &eval_points, &solver_config)
                    .unwrap();
            compute_objective(&result, &config).unwrap()
        };
        let mut plus = x.clone();
        let mut minus = x.clone();
        plus[parameter] += step;
        minus[parameter] -= step;
        let finite_difference = (objective_at(&plus) - objective_at(&minus)) / (2.0 * step);
        let scale = finite_difference.abs().max(1.0);
        assert!(
            (adjoint[parameter] - finite_difference).abs() / scale < 5.0e-2,
            "adjoint={}, full-simulation FD={finite_difference}",
            adjoint[parameter]
        );
    }

    /// The affected-element map must be local to the parameter being
    /// perturbed and structurally consistent: driver offsets move their
    /// driver (plus baffle-hole neighborhood), waveguide parameters move the
    /// waveguide, and no parameter touches the other driver's aperture.
    #[test]
    fn affected_elements_track_parameter_perturbations() {
        use crate::mesh_gen::MeshRegion;

        let mut model = sample_model();
        model.simulation.freq_start = 500.0;
        model.simulation.freq_end = 700.0;
        model.simulation.freq_points = 2;
        model.simulation.max_mesh_frequency = Some(700.0);
        let baseline_mesh = crate::mesh_gen::generate_loudspeaker_mesh(&model);
        let x = pack_parameters(&model).unwrap();
        // The production adjoint uses a 1e-7 sensitivity step; the baffle
        // Delaunay must not flip diagonals there.
        let step = 1.0e-7;

        let regions_at = |parameter: usize, sign: f64| {
            let mut perturbed = x.clone();
            perturbed[parameter] += sign * step;
            let model = unpack_parameters(&perturbed, &model).unwrap();
            let mesh = crate::mesh_gen::generate_loudspeaker_mesh(&model);
            assert_eq!(
                mesh.triangles, baseline_mesh.triangles,
                "parameter {parameter}: baffle triangulation changed at the sensitivity step"
            );
            let affected = affected_elements(&baseline_mesh, &mesh);
            let mut regions: Vec<MeshRegion> =
                affected.iter().map(|&index| mesh.region[index]).collect();
            regions.sort_by_key(|region| match region {
                MeshRegion::CabinetWall => 0usize,
                MeshRegion::WooferAperture => 1,
                MeshRegion::TweeterAperture => 2,
                MeshRegion::DriverAperture(index) => 3 + *index,
                MeshRegion::Waveguide => usize::MAX,
                MeshRegion::PortAperture => 3,
            });
            regions.dedup();
            (affected, regions)
        };

        for parameter in 0..x.len() {
            let (affected_plus, regions_plus) = regions_at(parameter, 1.0);
            let (affected_minus, _) = regions_at(parameter, -1.0);
            assert!(
                !affected_plus.is_empty(),
                "parameter {parameter} did not move any element"
            );
            assert_eq!(
                affected_plus, affected_minus,
                "parameter {parameter}: affected set must be step-sign independent"
            );
            assert!(
                affected_plus.len() * 10 < baseline_mesh.triangles.len() * 7,
                "parameter {parameter}: {} of {} elements affected — not local",
                affected_plus.len(),
                baseline_mesh.triangles.len()
            );
            // Layout: [woofer_y, tweeter_y, wg shapes..., wg_depth]; no
            // parameter may touch the other driver's aperture.
            match parameter {
                0 => {
                    assert!(regions_plus.contains(&MeshRegion::WooferAperture));
                    assert!(!regions_plus.contains(&MeshRegion::TweeterAperture));
                    assert!(!regions_plus.contains(&MeshRegion::Waveguide));
                }
                1 => {
                    assert!(regions_plus.contains(&MeshRegion::TweeterAperture));
                    assert!(!regions_plus.contains(&MeshRegion::WooferAperture));
                }
                _ => {
                    // Waveguide spline radii and depth. The tweeter aperture
                    // sits at the waveguide throat, so it may legitimately be
                    // affected (throat translation / throat-ring normals).
                    assert!(regions_plus.contains(&MeshRegion::Waveguide));
                    assert!(!regions_plus.contains(&MeshRegion::WooferAperture));
                }
            }
        }
    }

    /// The local sensitivity path must reproduce the full-reassembly
    /// reference for every parameter: identical held-state residuals and
    /// field SPL up to the sub-tolerance node motions it skips.
    #[test]
    fn local_sensitivities_match_full_reassembly_per_parameter() {
        let mut model = sample_model();
        model.simulation.freq_start = 500.0;
        model.simulation.freq_end = 700.0;
        model.simulation.freq_points = 2;
        model.simulation.max_mesh_frequency = Some(700.0);
        let eval_points = cea2034_output::generate_eval_points(model.baffle_center());
        let solver_config = SolverConfig::default();
        let x = pack_parameters(&model).unwrap();
        let baseline =
            dense_forward_evaluation(&model, &eval_points, &solver_config, 2_000).unwrap();

        for parameter in 0..x.len() {
            let mut perturbed = x.clone();
            perturbed[parameter] += 1.0e-7;
            let perturbed_model = unpack_parameters(&perturbed, &model).unwrap();
            let local = evaluate_perturbed_geometry_with_held_state(
                &perturbed_model,
                &eval_points,
                &solver_config,
                &baseline,
            )
            .unwrap();
            let full = evaluate_perturbed_geometry_with_held_state_full(
                &perturbed_model,
                &eval_points,
                &solver_config,
                &baseline,
            )
            .unwrap();

            for frequency in 0..local.residuals.len() {
                let residual_scale = full.residuals[frequency]
                    .iter()
                    .map(|value| value.norm())
                    .fold(1.0, f64::max);
                let max_difference = local.residuals[frequency]
                    .iter()
                    .zip(&full.residuals[frequency])
                    .map(|(local, full)| (*local - *full).norm())
                    .fold(0.0, f64::max);
                assert!(
                    max_difference / residual_scale < 1.0e-6,
                    "parameter {parameter}, frequency {frequency}: residual difference \
                     {max_difference:.3e} (scale {residual_scale:.3e})"
                );
            }
            let spl_difference = local
                .result
                .spl
                .as_slice()
                .iter()
                .zip(full.result.spl.as_slice())
                .map(|(local, full)| (local - full).abs())
                .fold(0.0, f64::max);
            assert!(
                spl_difference < 1.0e-5,
                "parameter {parameter}: SPL difference {spl_difference:.3e} dB"
            );
        }
    }

    /// The local and full-reference sensitivity paths must produce the same
    /// adjoint gradient end to end.
    #[test]
    fn local_adjoint_gradient_matches_full_reference() {
        let mut model = sample_model();
        model.simulation.freq_start = 500.0;
        model.simulation.freq_end = 700.0;
        model.simulation.freq_points = 2;
        model.simulation.max_mesh_frequency = Some(700.0);
        let config = OptimizationConfig {
            optim_freq_points: 2,
            optim_max_mesh_freq: 700.0,
            adjoint_max_dofs: 2_000,
            ..OptimizationConfig::default()
        };
        let eval_points = cea2034_output::generate_eval_points(model.baffle_center());
        let solver_config = SolverConfig::default();
        let x = pack_parameters(&model).unwrap();
        let bounds = compute_bounds(&model).unwrap();
        let local =
            discrete_adjoint_gradient(&x, &model, &eval_points, &solver_config, &config, &bounds)
                .unwrap();
        let reference = discrete_adjoint_gradient_reference(
            &x,
            &model,
            &eval_points,
            &solver_config,
            &config,
            &bounds,
        )
        .unwrap();
        let scale = reference
            .iter()
            .map(|value| value.abs())
            .fold(1.0, f64::max);
        for (parameter, (local, reference)) in local.iter().zip(&reference).enumerate() {
            assert!(
                (local - reference).abs() / scale < 1.0e-6,
                "parameter {parameter}: local={local}, full reference={reference}"
            );
        }
    }

    /// Same adjoint-vs-FD contract as
    /// `discrete_bem_adjoint_matches_full_simulation_fd`, but with
    /// `direct_threshold: 0` so both the forward and the adjoint run on the
    /// FGMRES+ILU path with the reused forward preconditioner.
    #[test]
    fn discrete_bem_adjoint_matches_fd_on_iterative_path() {
        let mut model = sample_model();
        model.simulation.freq_start = 500.0;
        model.simulation.freq_end = 700.0;
        model.simulation.freq_points = 2;
        model.simulation.max_mesh_frequency = Some(700.0);
        let config = OptimizationConfig {
            optim_freq_points: 2,
            optim_max_mesh_freq: 700.0,
            adjoint_max_dofs: 2_000,
            ..OptimizationConfig::default()
        };
        let eval_points = cea2034_output::generate_eval_points(model.baffle_center());
        let solver_config = SolverConfig {
            direct_threshold: 0,
            ..SolverConfig::default()
        };
        let x = pack_parameters(&model).unwrap();
        let bounds = compute_bounds(&model).unwrap();
        let dense_baseline =
            dense_forward_evaluation(&model, &eval_points, &solver_config, 2_000).unwrap();
        let production_baseline =
            simulation::run_simulation_with_config(&model, &eval_points, &solver_config).unwrap();
        let baseline_spl_difference = dense_baseline
            .result
            .spl
            .as_slice()
            .iter()
            .zip(production_baseline.spl.as_slice())
            .map(|(&dense, &production)| (dense - production).abs())
            .fold(0.0, f64::max);
        assert!(
            baseline_spl_difference < 1.0e-3,
            "iterative adjoint forward path differs from production by {baseline_spl_difference} dB"
        );
        let adjoint =
            discrete_adjoint_gradient(&x, &model, &eval_points, &solver_config, &config, &bounds)
                .unwrap();

        let parameter = 0;
        let step = 1.0e-7;
        let objective_at = |parameters: &[f64]| {
            let candidate = unpack_parameters(parameters, &model).unwrap();
            let result =
                simulation::run_simulation_with_config(&candidate, &eval_points, &solver_config)
                    .unwrap();
            compute_objective(&result, &config).unwrap()
        };
        let mut plus = x.clone();
        let mut minus = x.clone();
        plus[parameter] += step;
        minus[parameter] -= step;
        let finite_difference = (objective_at(&plus) - objective_at(&minus)) / (2.0 * step);
        let scale = finite_difference.abs().max(1.0);
        assert!(
            (adjoint[parameter] - finite_difference).abs() / scale < 5.0e-2,
            "iterative-path adjoint={}, full-simulation FD={finite_difference}",
            adjoint[parameter]
        );
    }

    /// Counterpart to `optimize_reports_gradient_fallback_and_full_band_objective`:
    /// with the DOF cap above the test mesh and no typed drivers, the adjoint
    /// gradient path must run to completion (no fallback), and the reported
    /// backend mapping must reflect that.
    ///
    /// This is asserted at the gradient level rather than after a full
    /// `optimize` run on purpose: `compute_bounds` does not encode the
    /// waveguide-rim/tweeter-position coupling, so mid-search candidates can
    /// fail model validation and record a legitimate fallback even when the
    /// adjoint handled every valid point.
    #[test]
    fn discrete_adjoint_gradient_available_above_dof_cap() {
        let mut model = sample_model();
        model.simulation.freq_start = 500.0;
        model.simulation.freq_end = 700.0;
        model.simulation.freq_points = 2;
        model.simulation.max_mesh_frequency = Some(700.0);
        let eval_points = cea2034_output::generate_eval_points(model.baffle_center());
        let solver_config = SolverConfig::default();
        let x = pack_parameters(&model).unwrap();
        let bounds = compute_bounds(&model).unwrap();

        let config = OptimizationConfig {
            optim_freq_points: 2,
            optim_max_mesh_freq: 700.0,
            adjoint_max_dofs: 2_000,
            ..OptimizationConfig::default()
        };
        let gradient =
            discrete_adjoint_gradient(&x, &model, &eval_points, &solver_config, &config, &bounds)
                .expect("adjoint must run when the mesh fits under the DOF cap");
        assert_eq!(gradient.len(), x.len());
        assert!(gradient.iter().all(|value| value.is_finite()));
        assert_eq!(
            applied_gradient_method(config.gradient_method, None),
            AppliedGradientMethod::DiscreteAdjoint
        );

        // Below the cap with the H-matrix adjoint disabled the same call must
        // fail with the gate's reason, which the reporting maps to the FD
        // fallback variant.
        let capped = OptimizationConfig {
            adjoint_max_dofs: 1,
            adjoint_hmatrix: false,
            ..OptimizationConfig::default()
        };
        let error =
            discrete_adjoint_gradient(&x, &model, &eval_points, &solver_config, &capped, &bounds)
                .expect_err("adjoint must be gated by the DOF cap when H-matrix is disabled");
        assert!(
            error.to_string().contains("dense limit exceeded"),
            "unexpected gate reason: {error}"
        );
        match applied_gradient_method(capped.gradient_method, Some(error.to_string())) {
            AppliedGradientMethod::AdjointWithFiniteDifferenceFallback { reason } => {
                assert!(reason.contains("adjoint"));
            }
            other => panic!("expected the fallback variant, got {other:?}"),
        }

        // Below the cap with the H-matrix adjoint enabled (the default) the
        // gradient runs through the compressed operator path instead of
        // falling back: forward and adjoint solves both run FGMRES on the
        // H-matrix (adjoint via its Hermitian matvec).
        let hmatrix_capped = OptimizationConfig {
            optim_freq_points: 2,
            optim_max_mesh_freq: 700.0,
            adjoint_max_dofs: 1,
            adjoint_hmatrix: true,
            ..OptimizationConfig::default()
        };
        let hmatrix_gradient = discrete_adjoint_gradient(
            &x,
            &model,
            &eval_points,
            &solver_config,
            &hmatrix_capped,
            &bounds,
        )
        .expect("H-matrix adjoint must run above the dense DOF cap");
        assert_eq!(hmatrix_gradient.len(), x.len());
        assert!(
            hmatrix_gradient.iter().all(|value| value.is_finite()),
            "H-matrix adjoint gradient must be finite: {hmatrix_gradient:?}"
        );
        // Same physics, coarser operator: the H-matrix gradient must agree
        // with the dense adjoint gradient within a loose band.
        for (dense, compressed) in gradient.iter().zip(&hmatrix_gradient) {
            let scale = dense.abs().max(1.0);
            assert!(
                (dense - compressed).abs() / scale < 5e-2,
                "H-matrix adjoint gradient {compressed} diverges from dense {dense}"
            );
        }
    }

    /// Timing probe for ONE discrete-adjoint gradient evaluation, comparing
    /// the local sensitivities against the full-reassembly reference and the
    /// reused-factor adjoint solves against the previous fresh-solve
    /// behaviour, plus the central-FD fallback cost on the same model.
    ///
    /// Section A uses the default LU route (~724 DOFs at a 1 kHz mesh);
    /// section B forces the FGMRES+ILU route with `direct_threshold: 0` so the
    /// preconditioner-reuse delta is measured as well.
    ///
    /// Run explicitly (release) with:
    /// `cargo test -p sonium-sim-speaker --release adjoint_gradient_timing_probe -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn adjoint_gradient_timing_probe() {
        use std::time::Instant;

        let mut model = sample_model();
        model.waveguide = Some(WaveguideConfig {
            horizontal_spline: vec![[0.0, 0.05], [1.0, 0.013]],
            vertical_spline: vec![[0.0, 0.04], [1.0, 0.013]],
            depth: 0.025,
        });
        model.simulation.freq_start = 500.0;
        model.simulation.freq_end = 900.0;
        model.simulation.freq_points = 3;
        // 1 kHz mesh (724 DOFs): flip-free under the 1e-7 sensitivity step —
        // at 900/1200/1800/2000 Hz the woofer-hole rim sits next to a
        // grid-aligned Delaunay quad whose diagonal flips for woofer_y.
        model.simulation.max_mesh_frequency = Some(1000.0);
        let config = OptimizationConfig {
            optim_freq_points: 3,
            optim_max_mesh_freq: 1000.0,
            adjoint_max_dofs: 10_000,
            ..OptimizationConfig::default()
        };
        let eval_points = cea2034_output::generate_eval_points(model.baffle_center());
        let x = pack_parameters(&model).unwrap();
        let bounds = compute_bounds(&model).unwrap();

        // Previous behaviour: form Aᴴ and solve it from scratch (fresh LU or
        // fresh ILU, depending on the route) at every frequency.
        let previous_adjoint_solves =
            |baseline: &DenseForwardEvaluation,
             solver_config: &SolverConfig,
             config: &OptimizationConfig| {
                let spl_gradient = objective_spl_gradient(&baseline.result, config).unwrap();
                let start = Instant::now();
                for (index, state) in baseline.states.iter().enumerate() {
                    let adjoint_rhs =
                        field_objective_state_derivative(state, spl_gradient.row(index))
                            .iter()
                            .map(|value| -*value)
                            .collect::<Vec<_>>();
                    let matrix = state.dense_matrix();
                    let hermitian = DenseMatrix::from_row_major_vec(
                        matrix.ncols(),
                        matrix.nrows(),
                        (0..matrix.ncols())
                            .flat_map(|row| {
                                (0..matrix.nrows()).map(move |column| matrix[[column, row]].conj())
                            })
                            .collect(),
                    );
                    simulation::solve_bem_system(
                        &hermitian,
                        &adjoint_rhs,
                        None,
                        solver_config,
                        None,
                    )
                    .unwrap();
                }
                start.elapsed().as_secs_f64()
            };
        let reused_adjoint_solves =
            |baseline: &DenseForwardEvaluation,
             solver_config: &SolverConfig,
             config: &OptimizationConfig| {
                let spl_gradient = objective_spl_gradient(&baseline.result, config).unwrap();
                let start = Instant::now();
                for (index, state) in baseline.states.iter().enumerate() {
                    let adjoint_rhs =
                        field_objective_state_derivative(state, spl_gradient.row(index))
                            .iter()
                            .map(|value| -*value)
                            .collect::<Vec<_>>();
                    simulation::solve_bem_adjoint_system(
                        state.dense_matrix(),
                        &adjoint_rhs,
                        state
                            .solver_state
                            .as_ref()
                            .expect("dense forward state retains solver state"),
                        solver_config,
                    )
                    .unwrap();
                }
                start.elapsed().as_secs_f64()
            };

        // --- Section A: direct-LU route ------------------------------------
        let solver_config = SolverConfig::default();
        let start = Instant::now();
        let baseline = dense_forward_evaluation(
            &model,
            &eval_points,
            &solver_config,
            config.adjoint_max_dofs,
        )
        .unwrap();
        let forward_sweep_s = start.elapsed().as_secs_f64();
        let n_dofs = baseline.states[0].surface_pressure.len();
        let reused_lu_s = reused_adjoint_solves(&baseline, &solver_config, &config);

        // Full gradient with LOCAL sensitivities (current implementation) vs
        // the full-reassembly reference (previous behaviour).
        let start = Instant::now();
        let gradient =
            discrete_adjoint_gradient(&x, &model, &eval_points, &solver_config, &config, &bounds)
                .unwrap();
        let local_gradient_s = start.elapsed().as_secs_f64();
        assert!(gradient.iter().all(|value| value.is_finite()));
        let start = Instant::now();
        let reference_gradient = discrete_adjoint_gradient_reference(
            &x,
            &model,
            &eval_points,
            &solver_config,
            &config,
            &bounds,
        )
        .unwrap();
        let reference_gradient_s = start.elapsed().as_secs_f64();
        for (local, reference) in gradient.iter().zip(&reference_gradient) {
            assert!(
                (local - reference).abs() / reference.abs().max(1.0) < 1.0e-6,
                "local={local}, full reference={reference}"
            );
        }

        let fresh_lu_s = previous_adjoint_solves(&baseline, &solver_config, &config);

        // Central-FD fallback cost (what configs above the DOF cap pay).
        let objective = |parameters: &[f64]| {
            let candidate = unpack_parameters(parameters, &model).unwrap();
            let result =
                simulation::run_simulation_with_config(&candidate, &eval_points, &solver_config)
                    .unwrap();
            compute_objective(&result, &config).unwrap()
        };
        let start = Instant::now();
        let fd_gradient = gradient_central(&objective, &x, f64::EPSILON.powf(1.0 / 3.0));
        let central_fd_gradient_s = start.elapsed().as_secs_f64();
        assert!(fd_gradient.iter().all(|value| value.is_finite()));

        eprintln!(
            "adjoint timing probe A (LU route, {n_dofs} DOFs, {} frequencies, {} parameters):",
            baseline.states.len(),
            x.len()
        );
        eprintln!(
            "  gradient, full-reassembly sensitivities: {reference_gradient_s:.3} s (previous behaviour)"
        );
        eprintln!("  gradient, local sensitivities:           {local_gradient_s:.3} s");
        eprintln!(
            "  breakdown: forward sweep {forward_sweep_s:.3} s, adjoint solves (reused LU) {reused_lu_s:.3} s,"
        );
        eprintln!(
            "             local sensitivity share ≈ {:.3} s",
            local_gradient_s - forward_sweep_s - reused_lu_s
        );
        eprintln!(
            "  adjoint solves, fresh Aᴴ LU:               {fresh_lu_s:.3} s (pre-P1 behaviour)"
        );
        eprintln!("  central FD gradient (fallback):            {central_fd_gradient_s:.3} s");

        // --- Section B: FGMRES+ILU route -----------------------------------
        let solver_config = SolverConfig {
            direct_threshold: 0,
            ..SolverConfig::default()
        };
        let baseline = dense_forward_evaluation(
            &model,
            &eval_points,
            &solver_config,
            config.adjoint_max_dofs,
        )
        .unwrap();
        let fresh_ilu_s = previous_adjoint_solves(&baseline, &solver_config, &config);
        let reused_ilu_s = reused_adjoint_solves(&baseline, &solver_config, &config);
        eprintln!("adjoint timing probe B (FGMRES+ILU route, {n_dofs} DOFs):");
        eprintln!("  adjoint solves, fresh ILU each:     {fresh_ilu_s:.3} s (pre-P1 behaviour)");
        eprintln!("  adjoint solves, reused forward ILU: {reused_ilu_s:.3} s");
    }

    /// Larger-mesh companion to `adjoint_gradient_timing_probe`: ~3902 DOFs
    /// (2.5 kHz mesh, FGMRES+ILU route — above the 3000-DOF direct
    /// threshold), where the O(N²) sensitivity re-assembly dominates the old
    /// path. Times only the gradient (reference vs local). The 2.5 kHz mesh
    /// is flip-free under the 1e-7 sensitivity step (2/1.8/1.2/0.9 kHz are
    /// not — woofer-hole Delaunay ties).
    ///
    /// Run explicitly (release) with:
    /// `cargo test -p sonium-sim-speaker --release adjoint_gradient_timing_probe_large -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn adjoint_gradient_timing_probe_large() {
        use std::time::Instant;

        let mut model = sample_model();
        model.waveguide = Some(WaveguideConfig {
            horizontal_spline: vec![[0.0, 0.05], [1.0, 0.013]],
            vertical_spline: vec![[0.0, 0.04], [1.0, 0.013]],
            depth: 0.025,
        });
        model.simulation.freq_start = 500.0;
        model.simulation.freq_end = 900.0;
        model.simulation.freq_points = 3;
        model.simulation.max_mesh_frequency = Some(2500.0); // ~3902 DOFs
        let config = OptimizationConfig {
            optim_freq_points: 3,
            optim_max_mesh_freq: 2500.0,
            adjoint_max_dofs: 10_000,
            ..OptimizationConfig::default()
        };
        let eval_points = cea2034_output::generate_eval_points(model.baffle_center());
        let solver_config = SolverConfig::default();
        let x = pack_parameters(&model).unwrap();
        let bounds = compute_bounds(&model).unwrap();

        let start = Instant::now();
        let baseline = dense_forward_evaluation(
            &model,
            &eval_points,
            &solver_config,
            config.adjoint_max_dofs,
        )
        .unwrap();
        let forward_sweep_s = start.elapsed().as_secs_f64();
        let n_dofs = baseline.states[0].surface_pressure.len();

        let start = Instant::now();
        let gradient =
            discrete_adjoint_gradient(&x, &model, &eval_points, &solver_config, &config, &bounds)
                .unwrap();
        let local_gradient_s = start.elapsed().as_secs_f64();
        assert!(gradient.iter().all(|value| value.is_finite()));

        let start = Instant::now();
        let reference_gradient = discrete_adjoint_gradient_reference(
            &x,
            &model,
            &eval_points,
            &solver_config,
            &config,
            &bounds,
        )
        .unwrap();
        let reference_gradient_s = start.elapsed().as_secs_f64();
        for (local, reference) in gradient.iter().zip(&reference_gradient) {
            assert!(
                (local - reference).abs() / reference.abs().max(1.0) < 1.0e-6,
                "local={local}, full reference={reference}"
            );
        }

        eprintln!(
            "adjoint timing probe large (LU route, {n_dofs} DOFs, {} frequencies, {} parameters):",
            baseline.states.len(),
            x.len()
        );
        eprintln!(
            "  gradient, full-reassembly sensitivities: {reference_gradient_s:.3} s (previous behaviour)"
        );
        eprintln!("  gradient, local sensitivities:           {local_gradient_s:.3} s");
        eprintln!("  forward sweep alone:                     {forward_sweep_s:.3} s");
    }

    // -----------------------------------------------------------------------
    // Batched candidate evaluation
    // -----------------------------------------------------------------------

    /// Small LU-path model shared by the batched-evaluation tests: 3
    /// frequencies and a 1 kHz mesh keep every candidate a direct solve, so
    /// warm starts cannot mask a lane-assignment bug behind iterative
    /// tolerances.
    type BatchTestFixture = (
        LoudspeakerModel,
        Vec<[f64; 3]>,
        SolverConfig,
        Vec<f64>,
        Vec<Bound<f64>>,
    );

    fn batch_test_fixture() -> BatchTestFixture {
        let mut model = sample_model();
        model.simulation.freq_start = 500.0;
        model.simulation.freq_end = 1000.0;
        model.simulation.freq_points = 3;
        model.simulation.max_mesh_frequency = Some(1000.0);
        let eval_points = cea2034_output::generate_eval_points(model.baffle_center());
        let solver_config = SolverConfig::default();
        let x = pack_parameters(&model).unwrap();
        let bounds = compute_bounds(&model).unwrap();
        (model, eval_points, solver_config, x, bounds)
    }

    fn config_with_candidate_lanes(max_candidate_lanes: usize) -> OptimizationConfig {
        OptimizationConfig {
            max_candidate_lanes: Some(max_candidate_lanes),
            ..OptimizationConfig::default()
        }
    }

    fn assert_values_match(
        batched: &[Result<f64, OptimizationError>],
        reference: &[Result<f64, OptimizationError>],
        context: &str,
    ) {
        assert_eq!(batched.len(), reference.len(), "{context}");
        for (index, (batched, reference)) in batched.iter().zip(reference).enumerate() {
            let (Ok(batched), Ok(reference)) = (batched, reference) else {
                panic!("{context}: candidate {index} failed: {batched:?} vs {reference:?}");
            };
            let tolerance = 1.0e-9 * reference.abs().max(1.0);
            assert!(
                (batched - reference).abs() <= tolerance,
                "{context}: candidate {index}: batched={batched}, reference={reference}"
            );
        }
    }

    #[test]
    fn batched_fd_gradient_matches_serial() {
        use std::time::Instant;

        let (model, eval_points, solver_config, x, bounds) = batch_test_fixture();
        let h = f64::EPSILON.powf(1.0 / 3.0);

        // Serial path (max_candidate_lanes = 1): sequential per-parameter
        // evaluation with one compounding warm start — the pre-batch
        // behaviour, kept reachable through the config knob.
        let start = Instant::now();
        let reference = gradient_central_batched(
            &x,
            h,
            &model,
            &eval_points,
            &solver_config,
            &config_with_candidate_lanes(1),
            &bounds,
            None,
        );
        let serial_s = start.elapsed().as_secs_f64();
        assert!(reference.iter().all(|value| value.is_finite()));

        for lanes in [2, 3] {
            let start = Instant::now();
            let gradient = gradient_central_batched(
                &x,
                h,
                &model,
                &eval_points,
                &solver_config,
                &config_with_candidate_lanes(lanes),
                &bounds,
                None,
            );
            let batched_s = start.elapsed().as_secs_f64();
            eprintln!(
                "central FD gradient ({} parameters): serial {serial_s:.2}s vs {lanes} candidate lanes {batched_s:.2}s",
                x.len()
            );
            assert_eq!(gradient.len(), reference.len());
            for (parameter, (batched, expected)) in gradient.iter().zip(&reference).enumerate() {
                let tolerance = 1.0e-9 * expected.abs().max(1.0);
                assert!(
                    (batched - expected).abs() <= tolerance,
                    "parameter {parameter}: batched={batched}, serial={expected}"
                );
            }
        }
    }

    #[test]
    fn batched_candidates_lane_assignment_invariant() {
        let (model, eval_points, solver_config, x, _bounds) = batch_test_fixture();
        let candidates = [1.0e-3, -8.0e-4, 6.0e-4]
            .iter()
            .enumerate()
            .map(|(parameter, &delta)| {
                let mut point = x.clone();
                point[parameter] += delta;
                point
            })
            .collect::<Vec<_>>();

        let single_lane = evaluate_candidates_batch(
            &candidates,
            &model,
            &eval_points,
            &solver_config,
            &config_with_candidate_lanes(1),
            None,
        );
        let two_lanes = evaluate_candidates_batch(
            &candidates,
            &model,
            &eval_points,
            &solver_config,
            &config_with_candidate_lanes(2),
            None,
        );
        assert_values_match(&two_lanes, &single_lane, "two candidate lanes vs one");
    }

    #[test]
    fn batched_candidates_deterministic() {
        let (model, eval_points, solver_config, x, _bounds) = batch_test_fixture();
        let candidates = [1.0e-3, -8.0e-4, 6.0e-4]
            .iter()
            .enumerate()
            .map(|(parameter, &delta)| {
                let mut point = x.clone();
                point[parameter] += delta;
                point
            })
            .collect::<Vec<_>>();
        let config = config_with_candidate_lanes(2);

        // Two identical batched runs must agree bitwise: lanes are contiguous
        // deterministic chunks, every candidate on this fixture is a direct
        // LU solve (no iterative tolerance in play), and lane-local warm
        // starts are seeded from the same (empty) base.
        let first = evaluate_candidates_batch(
            &candidates,
            &model,
            &eval_points,
            &solver_config,
            &config,
            None,
        );
        let second = evaluate_candidates_batch(
            &candidates,
            &model,
            &eval_points,
            &solver_config,
            &config,
            None,
        );
        assert_eq!(first.len(), second.len());
        for (index, (first, second)) in first.iter().zip(&second).enumerate() {
            let (Ok(first), Ok(second)) = (first, second) else {
                panic!("candidate {index} failed: {first:?} vs {second:?}");
            };
            assert_eq!(
                first.to_bits(),
                second.to_bits(),
                "candidate {index}: {first} vs {second}"
            );
        }
    }
}
