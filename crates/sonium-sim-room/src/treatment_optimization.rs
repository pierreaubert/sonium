//! Adjoint optimization of room-boundary treatment.
//!
//! The Helmholtz volume and boundary operators are assembled once. Each design
//! variable scales one boundary mass matrix, so all treatment sensitivities
//! follow from one forward and one adjoint solve without remeshing.
//!
//! [`MultiFrequencyRoomTreatmentProblem`] extends this to a fixed-mesh
//! frequency sweep: per-frequency base operators share one CSR structure and
//! the frequency-independent treatment operators, the production objective
//! combines band-weighted target-level error, seat variance, a minimum
//! useful level, and optional treatment cost, treatment directions are
//! validated for passivity over the full band, and frequencies are evaluated
//! in parallel over contiguous lanes (`sonium_solvers::lanes`) with
//! deterministic results.
//!
//! Known follow-ups (out of scope): patch-level subdivision of boundary
//! groups, continuous density filtering, body-fitted projection, and
//! scenario-pipeline CLI wiring. The optional decay/T60 constraint is a calibrated Eyring
//! reduced-order guard; physical FEM/modal decay calibration remains a
//! separate promotion requirement.

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
#[doc(inline)]
pub use cuda::{CudaMultiFrequencyRoomTreatmentSession, CudaRoomTreatmentSession};

use num_complex::Complex64;
use sonium_fem::assembly::HelmholtzAssembler;
use sonium_fem::optimization::{DensityFilter, DensityFilterError, MmaConstraint, MmaOptimizer};
use sonium_fem::physics::acoustics::pressure::room::metrics::rt60_eyring;
use sonium_fem::solver::{SolverConfig, solve_csr, solve_csr_with_guess};
use sonium_solvers::CsrMatrix;
use sonium_solvers::lanes::{LaneConfig, run_lanes, try_plan_lanes};
use sonium_xem::constants::REFERENCE_PRESSURE;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

/// A sparse linear functional that evaluates pressure at a listener.
#[derive(Debug, Clone)]
pub struct ListenerFunctional {
    /// `(degree_of_freedom, interpolation_weight)` entries.
    pub entries: Vec<(usize, Complex64)>,
}

impl ListenerFunctional {
    /// Construct a nodal listener.
    pub fn at_node(node: usize) -> Self {
        Self {
            entries: vec![(node, Complex64::new(1.0, 0.0))],
        }
    }

    fn evaluate(&self, state: &[Complex64]) -> Complex64 {
        if let Some(&(dof, _)) = self.entries.iter().max_by_key(|&&(dof, _)| dof) {
            assert!(
                dof < state.len(),
                "ListenerFunctional references dof {dof} but state has length {}",
                state.len()
            );
        }
        self.entries
            .iter()
            .map(|&(dof, weight)| weight * state[dof])
            .sum()
    }
}

/// One treatment variable and its exact Helmholtz boundary derivative.
#[derive(Debug, Clone, Copy)]
pub struct TreatmentBoundary {
    /// Physical boundary tag in the `HelmholtzAssembler`.
    pub marker: usize,
    /// Complex coefficient added at unit treatment.
    ///
    /// Under Sonium's `exp(-iωt)` convention a passive local-admittance term
    /// normally has a negative imaginary contribution.
    pub coefficient_direction: Complex64,
    /// Physical panel area in m². When present for all boundaries, the MMA
    /// treatment-fraction constraint is area-weighted
    /// (`Σ t_i·area_i / Σ area_i <= fraction`) instead of a plain mean.
    /// `Option` keeps the field serde/default compatible: absent means
    /// "same weight as every other boundary".
    pub area_m2: Option<f64>,
    /// Cost per m² of the fully-treated panel, used only by the optional
    /// cost objective term `cost_weight · Σ t_i·area_i·cost_i`.
    pub cost_per_m2: Option<f64>,
}

/// Continuous panel-density filter used by the fixed-mesh treatment
/// optimizers.
///
/// Each physical treatment value is a normalized compact-support hat-kernel
/// average of the optimization values at nearby panel centres.  The filter
/// preserves constants, keeps values in `[0, 1]`, and is linear, so objective
/// gradients and linearized constraints can be pulled back through its
/// transpose before the MMA update.  This is a mesh-boundary regularizer; it
/// does not replace body-fitted projection or post-update reanalysis.
#[derive(Debug, Clone)]
pub struct TreatmentDensityFilter {
    /// Compact-support radius in metres.
    pub radius_m: f64,
    filter: DensityFilter,
}

const DENSITY_FILTER_TOLERANCE: f64 = 1.0e-10;

impl TreatmentDensityFilter {
    /// Build a deterministic normalized hat-kernel filter from panel centres.
    ///
    /// The weight map is owned by [`DensityFilter::build_from_positions`];
    /// this adapter only attaches the room error type and the `[0, 1]`
    /// treatment gates in [`Self::apply`].
    pub fn new(positions_m: Vec<[f64; 3]>, radius_m: f64) -> Result<Self, RoomTreatmentError> {
        let filter =
            DensityFilter::build_from_positions(&positions_m, radius_m).map_err(|error| {
                RoomTreatmentError::InvalidProblem(match error {
                    DensityFilterError::EmptyPositions => "density filter requires panel positions",
                    DensityFilterError::InvalidRadius => {
                        "density filter radius must be finite and positive"
                    }
                    DensityFilterError::RadiusTooLarge => "density filter radius is too large",
                    DensityFilterError::NonFinitePosition => {
                        "density filter positions must be finite"
                    }
                    DensityFilterError::NonFiniteDistance => {
                        "density filter position distance is non-finite"
                    }
                    DensityFilterError::EmptySupportRow => {
                        "density filter has an empty support row"
                    }
                })
            })?;
        let adapter = Self { radius_m, filter };
        adapter.validate(adapter.filter.len())?;
        Ok(adapter)
    }

    /// Validate filter dimensions and normalized non-negative rows.
    pub fn validate(&self, treatment_count: usize) -> Result<(), RoomTreatmentError> {
        if treatment_count == 0 || self.filter.len() != treatment_count {
            return Err(RoomTreatmentError::InvalidProblem(
                "density filter count does not match treatment boundaries",
            ));
        }
        if !self.radius_m.is_finite() || self.radius_m <= 0.0 {
            return Err(RoomTreatmentError::InvalidProblem(
                "density filter radius must be finite and positive",
            ));
        }
        for index in 0..self.filter.len() {
            let row = self.filter.row(index);
            if row
                .iter()
                .any(|(_, weight)| !weight.is_finite() || *weight < 0.0)
                || (row.iter().map(|(_, weight)| weight).sum::<f64>() - 1.0).abs()
                    > DENSITY_FILTER_TOLERANCE
            {
                return Err(RoomTreatmentError::InvalidProblem(
                    "density filter rows must be finite, non-negative, and normalized",
                ));
            }
        }
        Ok(())
    }

    /// Apply the filter to optimization densities.
    pub fn apply(&self, design: &[f64]) -> Result<Vec<f64>, RoomTreatmentError> {
        self.validate(design.len())?;
        if design
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(RoomTreatmentError::InvalidProblem(
                "density-filter design must be finite and lie in [0, 1]",
            ));
        }
        let filtered = self.filter.filter_densities(design);
        if filtered
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(RoomTreatmentError::InvalidProblem(
                "density filter produced an invalid treatment",
            ));
        }
        Ok(filtered)
    }

    /// Pull a physical-density gradient back to the optimization variables.
    pub fn pullback_gradient(
        &self,
        filtered_gradient: &[f64],
    ) -> Result<Vec<f64>, RoomTreatmentError> {
        self.validate(filtered_gradient.len())?;
        if filtered_gradient.iter().any(|value| !value.is_finite()) {
            return Err(RoomTreatmentError::InvalidProblem(
                "density-filter gradient must be finite",
            ));
        }
        let design_gradient = self
            .filter
            .filter_sensitivities_transpose(filtered_gradient);
        if design_gradient.iter().any(|value| !value.is_finite()) {
            return Err(RoomTreatmentError::InvalidProblem(
                "density-filter pullback gradient is non-finite",
            ));
        }
        Ok(design_gradient)
    }
}

/// Frequency-local room treatment problem.
pub struct RoomTreatmentProblem {
    base_operator: CsrMatrix<Complex64>,
    treatment_operators: Vec<CsrMatrix<Complex64>>,
    rhs: Vec<Complex64>,
    listeners: Vec<ListenerFunctional>,
    solver: SolverConfig,
}

/// Settings for bounded, volume-constrained treatment optimization.
#[derive(Debug, Clone)]
pub struct RoomTreatmentConfig {
    /// Maximum MMA iterations.
    pub max_iterations: usize,
    /// Stop when the largest treatment change is below this value.
    pub tolerance: f64,
    /// Maximum mean treatment fraction across boundary groups.
    pub treatment_fraction: f64,
    /// Initial treatment fraction for each boundary group.
    pub initial_treatment: f64,
    /// Optional adjacency-based minimum-contrast patch constraint.
    pub spatial_constraint: Option<TreatmentSpatialConstraint>,
    /// Optional continuous panel-density filter.
    pub density_filter: Option<TreatmentDensityFilter>,
}

impl Default for RoomTreatmentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 40,
            tolerance: 1.0e-3,
            treatment_fraction: 0.5,
            initial_treatment: 0.25,
            spatial_constraint: None,
            density_filter: None,
        }
    }
}

/// Result of room-treatment optimization.
#[derive(Debug, Clone)]
pub struct RoomTreatmentResult {
    /// Optimized treatment value per boundary, in input order.
    pub treatments: Vec<f64>,
    /// Seat-to-seat log-pressure variance after each accepted design.
    pub objective_history: Vec<f64>,
    /// Number of completed MMA iterations.
    pub iterations: usize,
    /// Whether the treatment-change criterion was met.
    pub converged: bool,
}

/// Recoverable room-treatment optimization error.
#[derive(Debug)]
pub enum RoomTreatmentError {
    InvalidConfiguration(&'static str),
    InvalidProblem(&'static str),
    /// A treatment direction becomes active (non-passive) somewhere in the
    /// band. Carries frequency/boundary detail.
    NonPassive(String),
    /// The final nonlinear design violates an optional physical-domain
    /// constraint even though the local MMA subproblem was solved.
    ConstraintViolation(String),
    Solve(String),
}

impl fmt::Display for RoomTreatmentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration(message) => {
                write!(formatter, "invalid room-treatment configuration: {message}")
            }
            Self::InvalidProblem(message) => {
                write!(formatter, "invalid room-treatment problem: {message}")
            }
            Self::NonPassive(message) => {
                write!(formatter, "non-passive room-treatment direction: {message}")
            }
            Self::ConstraintViolation(message) => {
                write!(formatter, "room-treatment constraint violated: {message}")
            }
            Self::Solve(message) => write!(formatter, "room-treatment FEM solve failed: {message}"),
        }
    }
}

impl Error for RoomTreatmentError {}

fn validate_treatment_boundaries(
    treatments: &[TreatmentBoundary],
) -> Result<(), RoomTreatmentError> {
    for treatment in treatments {
        if !treatment.coefficient_direction.re.is_finite()
            || !treatment.coefficient_direction.im.is_finite()
        {
            return Err(RoomTreatmentError::InvalidProblem(
                "treatment coefficient direction must be finite",
            ));
        }
        if treatment
            .area_m2
            .is_some_and(|area| !area.is_finite() || area <= 0.0)
        {
            return Err(RoomTreatmentError::InvalidProblem(
                "treatment area must be finite and positive",
            ));
        }
        if treatment
            .cost_per_m2
            .is_some_and(|cost| !cost.is_finite() || cost < 0.0)
        {
            return Err(RoomTreatmentError::InvalidProblem(
                "treatment cost must be finite and non-negative",
            ));
        }
    }
    Ok(())
}

impl RoomTreatmentProblem {
    /// Build a problem from the same preassembled matrices used by room FEM.
    pub fn from_helmholtz(
        assembler: &HelmholtzAssembler,
        wavenumber: Complex64,
        base_boundary_coefficients: &HashMap<usize, Complex64>,
        treatments: &[TreatmentBoundary],
        rhs: Vec<Complex64>,
        listeners: Vec<ListenerFunctional>,
        solver: SolverConfig,
    ) -> Result<Self, RoomTreatmentError> {
        if rhs.len() != assembler.num_rows {
            return Err(RoomTreatmentError::InvalidProblem(
                "RHS size does not match Helmholtz operator",
            ));
        }
        if listeners.len() < 2 {
            return Err(RoomTreatmentError::InvalidProblem(
                "seat-variance optimization requires at least two listeners",
            ));
        }
        if treatments.is_empty() {
            return Err(RoomTreatmentError::InvalidProblem(
                "at least one treatment boundary is required",
            ));
        }
        validate_treatment_boundaries(treatments)?;
        for listener in &listeners {
            if listener.entries.is_empty()
                || listener.entries.iter().any(|&(dof, weight)| {
                    dof >= assembler.num_rows || !weight.re.is_finite() || !weight.im.is_finite()
                })
            {
                return Err(RoomTreatmentError::InvalidProblem(
                    "listener interpolation is empty or invalid",
                ));
            }
        }

        let base_operator = assembler.assemble(wavenumber, base_boundary_coefficients);
        let treatment_operators = treatments
            .iter()
            .map(|treatment| {
                let boundary_values = assembler.boundary_values.get(&treatment.marker).ok_or(
                    RoomTreatmentError::InvalidProblem(
                        "treatment marker has no assembled boundary mass matrix",
                    ),
                )?;
                let values = boundary_values
                    .iter()
                    .map(|&value| treatment.coefficient_direction * value)
                    .collect();
                Ok(CsrMatrix::from_raw_parts(
                    assembler.num_rows,
                    assembler.num_rows,
                    assembler.row_ptrs.clone(),
                    assembler.col_indices.clone(),
                    values,
                ))
            })
            .collect::<Result<Vec<_>, RoomTreatmentError>>()?;

        Ok(Self {
            base_operator,
            treatment_operators,
            rhs,
            listeners,
            solver,
        })
    }

    fn operator(&self, treatments: &[f64]) -> CsrMatrix<Complex64> {
        let mut operator = self.base_operator.clone();
        for (treatment, derivative) in treatments.iter().zip(&self.treatment_operators) {
            for (value, derivative_value) in operator.values.iter_mut().zip(&derivative.values) {
                *value += *treatment * *derivative_value;
            }
        }
        operator
    }

    fn solve(
        &self,
        operator: &CsrMatrix<Complex64>,
        rhs: &[Complex64],
    ) -> Result<Vec<Complex64>, RoomTreatmentError> {
        solve_csr(operator, rhs, &self.solver)
            .map(|solution| solution.values)
            .map_err(|error| RoomTreatmentError::Solve(error.to_string()))
    }
}

const PRESSURE_FLOOR: f64 = 1.0e-24;

/// Legacy seat-variance objective: variance of the natural-log pressure power
/// across listeners, plus its gradient with respect to the state, expressed
/// so that `dJ = 2 Re(state_gradientᴴ dp)` (listener weights map through Lᴴ).
fn seat_variance_objective(
    listeners: &[ListenerFunctional],
    state: &[Complex64],
) -> (f64, Vec<Complex64>) {
    let pressures = listeners
        .iter()
        .map(|listener| listener.evaluate(state))
        .collect::<Vec<_>>();
    let (objective, pressure_gradients) = seat_variance_pressures(&pressures);
    let mut state_gradient = vec![Complex64::new(0.0, 0.0); state.len()];
    for (listener, pressure_gradient) in listeners.iter().zip(pressure_gradients) {
        for &(dof, weight) in &listener.entries {
            state_gradient[dof] += weight.conj() * pressure_gradient;
        }
    }
    (objective, state_gradient)
}

fn seat_variance_pressures(pressures: &[Complex64]) -> (f64, Vec<Complex64>) {
    let log_powers = pressures
        .iter()
        .map(|pressure| (pressure.norm_sqr() + PRESSURE_FLOOR).ln())
        .collect::<Vec<_>>();
    let mean = log_powers.iter().sum::<f64>() / log_powers.len() as f64;
    let residuals = log_powers
        .iter()
        .map(|value| value - mean)
        .collect::<Vec<_>>();
    let objective = residuals
        .iter()
        .map(|residual| residual * residual)
        .sum::<f64>()
        / residuals.len() as f64;
    let pressure_gradients = pressures
        .iter()
        .zip(&residuals)
        .map(|(pressure, residual)| {
            *pressure
                * (2.0 * residual
                    / (pressures.len() as f64 * (pressure.norm_sqr() + PRESSURE_FLOOR)))
        })
        .collect();
    (objective, pressure_gradients)
}

fn conjugate_transpose(matrix: &CsrMatrix<Complex64>) -> CsrMatrix<Complex64> {
    let mut triplets = Vec::with_capacity(matrix.values.len());
    for row in 0..matrix.num_rows {
        for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            triplets.push((matrix.col_indices[index], row, matrix.values[index].conj()));
        }
    }
    CsrMatrix::from_triplets(matrix.num_cols, matrix.num_rows, triplets)
}

fn matvec(matrix: &CsrMatrix<Complex64>, vector: &[Complex64]) -> Vec<Complex64> {
    (0..matrix.num_rows)
        .map(|row| {
            (matrix.row_ptrs[row]..matrix.row_ptrs[row + 1])
                .map(|index| matrix.values[index] * vector[matrix.col_indices[index]])
                .sum()
        })
        .collect()
}

/// Exact per-treatment gradient from a converged forward/adjoint pair:
/// `g_i = 2 Re(λᴴ T_i p)` where `T_i` is the boundary-derivative operator.
fn treatment_gradients(
    treatment_operators: &[CsrMatrix<Complex64>],
    state: &[Complex64],
    adjoint: &[Complex64],
) -> Vec<f64> {
    treatment_operators
        .iter()
        .map(|derivative| {
            let residual_derivative = matvec(derivative, state);
            2.0 * adjoint
                .iter()
                .zip(residual_derivative)
                .map(|(lambda, value)| (lambda.conj() * value).re)
                .sum::<f64>()
        })
        .collect()
}

impl RoomTreatmentProblem {
    /// Evaluate seat variance and its exact discrete-adjoint gradient.
    pub fn objective_gradient(
        &self,
        treatments: &[f64],
    ) -> Result<(f64, Vec<f64>), RoomTreatmentError> {
        if treatments.len() != self.treatment_operators.len()
            || treatments
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(RoomTreatmentError::InvalidProblem(
                "treatment vector has the wrong size or lies outside [0, 1]",
            ));
        }
        let operator = self.operator(treatments);
        let state = self.solve(&operator, &self.rhs)?;
        let (objective, state_gradient) = seat_variance_objective(&self.listeners, &state);
        let adjoint_rhs = state_gradient
            .into_iter()
            .map(|value| -value)
            .collect::<Vec<_>>();
        let adjoint = self.solve(&conjugate_transpose(&operator), &adjoint_rhs)?;
        let gradient = treatment_gradients(&self.treatment_operators, &state, &adjoint);
        Ok((objective, gradient))
    }

    /// Optimize treatment allocation with MMA and a mean-treatment constraint.
    pub fn optimize(
        &self,
        config: &RoomTreatmentConfig,
    ) -> Result<RoomTreatmentResult, RoomTreatmentError> {
        self.optimize_with_evaluator(config, |treatments| self.objective_gradient(treatments))
    }

    fn optimize_with_evaluator(
        &self,
        config: &RoomTreatmentConfig,
        mut evaluate: impl FnMut(&[f64]) -> Result<(f64, Vec<f64>), RoomTreatmentError>,
    ) -> Result<RoomTreatmentResult, RoomTreatmentError> {
        if config.max_iterations == 0
            || !config.tolerance.is_finite()
            || config.tolerance <= 0.0
            || !config.treatment_fraction.is_finite()
            || !(1.0e-3..=1.0).contains(&config.treatment_fraction)
            || !config.initial_treatment.is_finite()
            || !(1.0e-3..=1.0).contains(&config.initial_treatment)
        {
            return Err(RoomTreatmentError::InvalidConfiguration(
                "iterations, tolerance, or treatment fractions are invalid",
            ));
        }
        let count = self.treatment_operators.len();
        if let Some(spatial_constraint) = &config.spatial_constraint {
            spatial_constraint.validate(count)?;
        }
        if let Some(density_filter) = &config.density_filter {
            density_filter.validate(count)?;
        }
        let mut design_treatments =
            vec![config.initial_treatment.min(config.treatment_fraction); count];
        let boundary_weights = vec![1.0; count];
        let mut optimizer = MmaOptimizer::new();
        let mut objective_history = Vec::with_capacity(config.max_iterations);
        let mut converged = false;

        for _ in 0..config.max_iterations {
            let treatments = config.density_filter.as_ref().map_or_else(
                || Ok(design_treatments.clone()),
                |filter| filter.apply(&design_treatments),
            )?;
            let (objective, filtered_gradient) = evaluate(&treatments)?;
            objective_history.push(objective);
            let needs_filtered_update =
                config.density_filter.is_some() || config.spatial_constraint.is_some();
            let next = if needs_filtered_update {
                let gradient = config.density_filter.as_ref().map_or_else(
                    || Ok(filtered_gradient.clone()),
                    |filter| filter.pullback_gradient(&filtered_gradient),
                )?;
                let total_weight = boundary_weights.iter().sum::<f64>();
                let filtered_volume_gradient = boundary_weights
                    .iter()
                    .map(|weight| weight / total_weight)
                    .collect::<Vec<_>>();
                let volume_constraint = MmaConstraint {
                    grad: config.density_filter.as_ref().map_or_else(
                        || Ok(filtered_volume_gradient.clone()),
                        |filter| filter.pullback_gradient(&filtered_volume_gradient),
                    )?,
                    value: treatments.iter().sum::<f64>() / count as f64,
                    bound: config.treatment_fraction,
                };
                let mut constraints = vec![volume_constraint];
                if let Some(spatial_constraint) = &config.spatial_constraint {
                    let filtered_constraints =
                        spatial_constraint.linearized_mma_constraints(&treatments)?;
                    for mut constraint in filtered_constraints {
                        constraint.grad = config.density_filter.as_ref().map_or_else(
                            || Ok(constraint.grad.clone()),
                            |filter| filter.pullback_gradient(&constraint.grad),
                        )?;
                        constraints.push(constraint);
                    }
                }
                optimizer.update_multi(
                    &gradient,
                    &design_treatments,
                    &constraints,
                    &boundary_weights,
                )
            } else {
                optimizer.update(
                    &filtered_gradient,
                    &design_treatments,
                    config.treatment_fraction,
                    &boundary_weights,
                )
            };
            let change = next
                .iter()
                .zip(&design_treatments)
                .map(|(next, current)| (next - current).abs())
                .fold(0.0, f64::max);
            design_treatments = next;
            if change < config.tolerance {
                converged = true;
                break;
            }
        }

        let treatments = if let Some(filter) = &config.density_filter {
            filter.apply(&design_treatments)?
        } else {
            design_treatments
        };
        if let Some(spatial_constraint) = &config.spatial_constraint {
            spatial_constraint.validate_design(&treatments)?;
        }

        Ok(RoomTreatmentResult {
            iterations: objective_history.len(),
            treatments,
            objective_history,
            converged,
        })
    }
}

// ===========================================================================
// Multi-frequency room treatment on a fixed mesh
// ===========================================================================

/// Conversion from log-power ratio to decibels: `L = DB_PER_LN · ln(q/p_ref²)`.
const DB_PER_LN: f64 = 10.0 / std::f64::consts::LN_10;

/// Tolerance on the passivity sign check (imaginary part of a Robin
/// coefficient must stay below this to be called dissipative).
const PASSIVITY_TOLERANCE: f64 = 1.0e-9;

/// One frequency of a multi-frequency treatment problem.
///
/// The base boundary coefficients are the Robin coefficients assembled into
/// the base operator (e.g. rigid-wall thermal admittance); the treatment
/// directions interpolate on top of them, and passivity is checked against
/// `base + t·direction` for `t ∈ [0, 1]`.
#[derive(Debug, Clone)]
pub struct TreatmentFrequencySetup {
    /// Frequency in Hz (used for sorting, reporting, and passivity detail).
    pub frequency_hz: f64,
    /// Complex wavenumber passed to `HelmholtzAssembler::assemble`.
    pub wavenumber: Complex64,
    /// Band weight `w_f` in the aggregated objective. Must be positive.
    pub weight: f64,
    /// Robin coefficients assembled into the base operator at this frequency.
    pub base_boundary_coefficients: HashMap<usize, Complex64>,
    /// Right-hand side at this frequency.
    pub rhs: Vec<Complex64>,
}

/// Which scalar objective the multi-frequency optimizer minimizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreatmentObjectiveKind {
    /// Production objective: band-weighted target-level error + seat
    /// variance + minimum-useful-level penalty (+ optional treatment cost).
    Production,
    /// Legacy objective: seat-to-seat variance of log pressure power only.
    SeatVariance,
}

/// Weights and targets of the multi-frequency objective.
///
/// Per frequency `f`, with seat levels `L_s = 20·log10(|p_s|/p_ref)`:
///
/// ```text
/// J_f = target_weight·mean_s(L_s − L_target_f)²
///     + variance_weight·var_s(L_s)
///     + min_level_weight·mean_s(relu(L_min − L_s)²)
/// J   = Σ_f w_f·J_f + cost_weight·Σ_i t_i·area_i·cost_i
/// ```
///
/// The min-level relu is differentiable almost everywhere; at the kink
/// (`L_s == L_min`) the subgradient 0 (the inactive branch) is used.
#[derive(Debug, Clone)]
pub struct TreatmentObjectiveConfig {
    /// Objective selector.
    pub kind: TreatmentObjectiveKind,
    /// Weight of the per-frequency target-level error term.
    pub target_weight: f64,
    /// Weight of the seat-to-seat level variance term.
    pub variance_weight: f64,
    /// Weight of the minimum-useful-level penalty.
    pub min_level_weight: f64,
    /// Minimum useful seat level `L_min` in dB re `reference_pressure`.
    pub min_level_db: f64,
    /// Per-frequency target level in dB re `reference_pressure`. A single
    /// entry is broadcast to all frequencies.
    pub target_levels_db: Vec<f64>,
    /// Reference pressure for the dB levels (20 µPa for acoustic SPL).
    pub reference_pressure: f64,
    /// Weight of the optional treatment cost term
    /// `Σ t_i·area_i·cost_per_m2_i` (areas/costs default to 1 when absent).
    pub cost_weight: f64,
}

impl Default for TreatmentObjectiveConfig {
    fn default() -> Self {
        Self {
            kind: TreatmentObjectiveKind::Production,
            target_weight: 1.0,
            variance_weight: 0.25,
            min_level_weight: 0.0,
            min_level_db: 40.0,
            target_levels_db: vec![75.0],
            reference_pressure: REFERENCE_PRESSURE,
            cost_weight: 0.0,
        }
    }
}

impl TreatmentObjectiveConfig {
    fn target_level_db(&self, frequency_index: usize) -> f64 {
        if self.target_levels_db.len() == 1 {
            self.target_levels_db[0]
        } else {
            self.target_levels_db[frequency_index]
        }
    }

    fn validate(&self, num_frequencies: usize) -> Result<(), RoomTreatmentError> {
        let weights = [
            self.target_weight,
            self.variance_weight,
            self.min_level_weight,
            self.cost_weight,
        ];
        if weights.iter().any(|w| !w.is_finite() || *w < 0.0)
            || !self.min_level_db.is_finite()
            || !self.reference_pressure.is_finite()
            || self.reference_pressure <= 0.0
        {
            return Err(RoomTreatmentError::InvalidConfiguration(
                "objective weights, minimum level, or reference pressure are invalid",
            ));
        }
        if self.kind == TreatmentObjectiveKind::Production
            && self.target_weight == 0.0
            && self.variance_weight == 0.0
            && self.min_level_weight == 0.0
        {
            return Err(RoomTreatmentError::InvalidConfiguration(
                "the production objective needs at least one positive acoustic weight",
            ));
        }
        if self.target_levels_db.len() != 1 && self.target_levels_db.len() != num_frequencies
            || self.target_levels_db.iter().any(|level| !level.is_finite())
        {
            return Err(RoomTreatmentError::InvalidConfiguration(
                "target levels must have one entry or one per frequency",
            ));
        }
        Ok(())
    }
}

/// Calibrated reduced-order decay constraint for treatment optimization.
///
/// The model predicts one room-wide reverberation time from an affine
/// treatment absorption area:
///
/// ```text
/// A(t) = base_absorption_area_m2
///        + Σ treatment_absorption_area_delta_m2[i] * t[i]
/// RT60(t) = rt60_eyring(volume_m3, total_surface_m2, A(t) / S, 4mV)
/// ```
///
/// Treatment deltas are non-negative absorption-area increments. Validation
/// requires every design in `[0, 1]^n` to remain in the finite, positive
/// Eyring domain. This is a useful optimization guard, not a substitute for
/// calibrating decay against physical FEM/modal or measured-room results.
#[derive(Debug, Clone)]
pub struct TreatmentDecayConstraint {
    /// Room volume in m³.
    pub volume_m3: f64,
    /// Total room surface in m².
    pub total_surface_m2: f64,
    /// Untreated equivalent absorption area in m².
    pub base_absorption_area_m2: f64,
    /// Equivalent absorption-area increase at unit treatment per boundary.
    pub treatment_absorption_area_delta_m2: Vec<f64>,
    /// Eyring air-absorption term `4mV` in m².
    pub air_absorption_4mv: f64,
    /// Optional lower RT60 bound in seconds.
    pub minimum_rt60_s: Option<f64>,
    /// Optional upper RT60 bound in seconds.
    pub maximum_rt60_s: Option<f64>,
}

const RT60_CONSTRAINT_TOLERANCE: f64 = 1.0e-8;

impl TreatmentDecayConstraint {
    /// Construct and validate a calibrated Eyring decay model.
    pub fn new(
        volume_m3: f64,
        total_surface_m2: f64,
        base_absorption_area_m2: f64,
        treatment_absorption_area_delta_m2: Vec<f64>,
        air_absorption_4mv: f64,
        minimum_rt60_s: Option<f64>,
        maximum_rt60_s: Option<f64>,
    ) -> Result<Self, RoomTreatmentError> {
        let constraint = Self {
            volume_m3,
            total_surface_m2,
            base_absorption_area_m2,
            treatment_absorption_area_delta_m2,
            air_absorption_4mv,
            minimum_rt60_s,
            maximum_rt60_s,
        };
        constraint.validate(constraint.treatment_absorption_area_delta_m2.len())?;
        Ok(constraint)
    }

    /// Validate geometry, absorption data, bounds, and the treatment count.
    pub fn validate(&self, treatment_count: usize) -> Result<(), RoomTreatmentError> {
        if treatment_count == 0 || self.treatment_absorption_area_delta_m2.len() != treatment_count
        {
            return Err(RoomTreatmentError::InvalidProblem(
                "decay treatment count does not match treatment boundaries",
            ));
        }
        if !self.volume_m3.is_finite()
            || self.volume_m3 <= 0.0
            || !self.total_surface_m2.is_finite()
            || self.total_surface_m2 <= 0.0
        {
            return Err(RoomTreatmentError::InvalidProblem(
                "decay volume and total surface must be finite and positive",
            ));
        }
        if !self.base_absorption_area_m2.is_finite() || self.base_absorption_area_m2 < 0.0 {
            return Err(RoomTreatmentError::InvalidProblem(
                "base absorption area must be finite and non-negative",
            ));
        }
        if !self.air_absorption_4mv.is_finite() || self.air_absorption_4mv < 0.0 {
            return Err(RoomTreatmentError::InvalidProblem(
                "air absorption must be finite and non-negative",
            ));
        }
        if self
            .treatment_absorption_area_delta_m2
            .iter()
            .any(|delta| !delta.is_finite() || *delta < 0.0)
        {
            return Err(RoomTreatmentError::InvalidProblem(
                "treatment absorption-area deltas must be finite and non-negative",
            ));
        }

        let maximum_absorption = self.base_absorption_area_m2
            + self.treatment_absorption_area_delta_m2.iter().sum::<f64>();
        if !maximum_absorption.is_finite() || maximum_absorption >= self.total_surface_m2 {
            return Err(RoomTreatmentError::InvalidProblem(
                "all treatment designs must have mean absorption below one",
            ));
        }
        if let Some(bound) = self.minimum_rt60_s
            && (!bound.is_finite() || bound <= 0.0)
        {
            return Err(RoomTreatmentError::InvalidProblem(
                "minimum RT60 must be finite and positive",
            ));
        }
        if let Some(bound) = self.maximum_rt60_s
            && (!bound.is_finite() || bound <= 0.0)
        {
            return Err(RoomTreatmentError::InvalidProblem(
                "maximum RT60 must be finite and positive",
            ));
        }
        if let (Some(minimum), Some(maximum)) = (self.minimum_rt60_s, self.maximum_rt60_s)
            && minimum > maximum
        {
            return Err(RoomTreatmentError::InvalidProblem(
                "minimum RT60 cannot exceed maximum RT60",
            ));
        }

        // With non-negative deltas, the endpoints bracket every design in
        // [0, 1]^n. Check both endpoints explicitly so the public Eyring
        // helper's finite-positive contract remains part of this API.
        for absorption_area in [self.base_absorption_area_m2, maximum_absorption] {
            let mean_alpha = absorption_area / self.total_surface_m2;
            let rt60 = rt60_eyring(
                self.volume_m3,
                self.total_surface_m2,
                mean_alpha,
                self.air_absorption_4mv,
            );
            if !rt60.is_finite() || rt60 <= 0.0 {
                return Err(RoomTreatmentError::InvalidProblem(
                    "all treatment designs must have finite positive Eyring RT60",
                ));
            }
        }
        Ok(())
    }

    fn evaluate_unchecked(&self, treatments: &[f64]) -> (f64, Vec<f64>) {
        let absorption_area = self.base_absorption_area_m2
            + treatments
                .iter()
                .zip(&self.treatment_absorption_area_delta_m2)
                .map(|(treatment, delta)| treatment * delta)
                .sum::<f64>();
        let mean_alpha = absorption_area / self.total_surface_m2;
        let one_minus_alpha = 1.0 - mean_alpha;
        let eyring_absorption = -self.total_surface_m2 * one_minus_alpha.ln();
        let denominator = eyring_absorption + self.air_absorption_4mv;
        let rt60 = rt60_eyring(
            self.volume_m3,
            self.total_surface_m2,
            mean_alpha,
            self.air_absorption_4mv,
        );
        let gradient = self
            .treatment_absorption_area_delta_m2
            .iter()
            .map(|delta| -rt60 * delta / (denominator * one_minus_alpha))
            .collect();
        (rt60, gradient)
    }

    /// Predict RT60 and return its analytic treatment gradient.
    pub fn predict_rt60(&self, treatments: &[f64]) -> Result<(f64, Vec<f64>), RoomTreatmentError> {
        self.validate(treatments.len())?;
        if treatments
            .iter()
            .any(|treatment| !treatment.is_finite() || !(0.0..=1.0).contains(treatment))
        {
            return Err(RoomTreatmentError::InvalidProblem(
                "decay treatment values must be finite and lie in [0, 1]",
            ));
        }
        let (rt60, gradient) = self.evaluate_unchecked(treatments);
        if !rt60.is_finite() || rt60 <= 0.0 || gradient.iter().any(|value| !value.is_finite()) {
            return Err(RoomTreatmentError::InvalidProblem(
                "predicted RT60 or gradient is non-finite",
            ));
        }
        Ok((rt60, gradient))
    }

    /// Convert the active RT60 bounds into normalized local MMA constraints.
    ///
    /// The lower bound is represented as `(minimum - RT60) / scale <= 0`;
    /// the upper bound is `RT60 / scale <= maximum / scale`.
    pub fn linearized_mma_constraints(
        &self,
        treatments: &[f64],
    ) -> Result<Vec<MmaConstraint>, RoomTreatmentError> {
        let (rt60, gradient) = self.predict_rt60(treatments)?;
        let scale = self
            .minimum_rt60_s
            .into_iter()
            .chain(self.maximum_rt60_s)
            .fold(1.0_f64, f64::max);
        let mut constraints = Vec::with_capacity(2);
        if let Some(minimum) = self.minimum_rt60_s {
            constraints.push(MmaConstraint {
                grad: gradient.iter().map(|value| -value / scale).collect(),
                value: (minimum - rt60) / scale,
                bound: 0.0,
            });
        }
        if let Some(maximum) = self.maximum_rt60_s {
            constraints.push(MmaConstraint {
                grad: gradient.iter().map(|value| *value / scale).collect(),
                value: rt60 / scale,
                bound: maximum / scale,
            });
        }
        Ok(constraints)
    }

    /// Check the actual nonlinear RT60 against the configured bounds.
    pub fn validate_rt60_bounds(&self, rt60: f64) -> Result<(), RoomTreatmentError> {
        if !rt60.is_finite() || rt60 <= 0.0 {
            return Err(RoomTreatmentError::ConstraintViolation(
                "predicted RT60 is not finite and positive".to_string(),
            ));
        }
        if let Some(minimum) = self.minimum_rt60_s {
            let tolerance = RT60_CONSTRAINT_TOLERANCE * minimum.max(1.0);
            if rt60 + tolerance < minimum {
                return Err(RoomTreatmentError::ConstraintViolation(format!(
                    "predicted RT60 {rt60} s is below minimum {minimum} s"
                )));
            }
        }
        if let Some(maximum) = self.maximum_rt60_s {
            let tolerance = RT60_CONSTRAINT_TOLERANCE * maximum.max(1.0);
            if rt60 > maximum + tolerance {
                return Err(RoomTreatmentError::ConstraintViolation(format!(
                    "predicted RT60 {rt60} s exceeds maximum {maximum} s"
                )));
            }
        }
        Ok(())
    }
}

/// Graph-based spatial treatment constraint that prevents adjacent panels from
/// taking sharply different values.
///
/// This is a linear minimum-contrast patch guard: every adjacent pair obeys
/// `|t_i - t_j| <= maximum_adjacent_difference`. It is useful for rejecting
/// checkerboard-like treatment layouts on a fixed boundary partition while
/// remaining an exact linear MMA constraint. It does not claim to replace
/// continuous density filtering or body-fitted reanalysis.
#[derive(Debug, Clone)]
pub struct TreatmentSpatialConstraint {
    /// Undirected adjacent treatment-boundary pairs.
    pub adjacent_pairs: Vec<(usize, usize)>,
    /// Maximum allowed treatment contrast between adjacent pairs.
    pub maximum_adjacent_difference: f64,
}

impl TreatmentSpatialConstraint {
    /// Construct and validate an explicit adjacency graph.
    pub fn new(
        adjacent_pairs: Vec<(usize, usize)>,
        maximum_adjacent_difference: f64,
        treatment_count: usize,
    ) -> Result<Self, RoomTreatmentError> {
        let constraint = Self {
            adjacent_pairs,
            maximum_adjacent_difference,
        };
        constraint.validate(treatment_count)?;
        Ok(constraint)
    }

    /// Construct adjacency from panel-centre positions and a neighbourhood
    /// radius. Pairs are emitted in deterministic lexicographic order.
    pub fn from_positions(
        positions_m: &[[f64; 3]],
        adjacency_radius_m: f64,
        maximum_adjacent_difference: f64,
    ) -> Result<Self, RoomTreatmentError> {
        if positions_m.is_empty() || positions_m.iter().flatten().any(|value| !value.is_finite()) {
            return Err(RoomTreatmentError::InvalidProblem(
                "spatial treatment positions must be non-empty and finite",
            ));
        }
        if !adjacency_radius_m.is_finite() || adjacency_radius_m <= 0.0 {
            return Err(RoomTreatmentError::InvalidProblem(
                "spatial treatment adjacency radius must be finite and positive",
            ));
        }
        let radius_squared = adjacency_radius_m * adjacency_radius_m;
        if !radius_squared.is_finite() {
            return Err(RoomTreatmentError::InvalidProblem(
                "spatial treatment adjacency radius is too large",
            ));
        }

        let mut adjacent_pairs = Vec::new();
        for first in 0..positions_m.len() {
            for second in (first + 1)..positions_m.len() {
                let delta = [
                    positions_m[first][0] - positions_m[second][0],
                    positions_m[first][1] - positions_m[second][1],
                    positions_m[first][2] - positions_m[second][2],
                ];
                let distance_squared = delta.iter().map(|value| value * value).sum::<f64>();
                if !distance_squared.is_finite() {
                    return Err(RoomTreatmentError::InvalidProblem(
                        "spatial treatment position distance is non-finite",
                    ));
                }
                if distance_squared <= radius_squared {
                    adjacent_pairs.push((first, second));
                }
            }
        }
        if adjacent_pairs.is_empty() {
            return Err(RoomTreatmentError::InvalidProblem(
                "spatial treatment radius produces no adjacent pairs",
            ));
        }
        Self::new(
            adjacent_pairs,
            maximum_adjacent_difference,
            positions_m.len(),
        )
    }

    /// Validate pair indices, uniqueness, and the contrast domain.
    pub fn validate(&self, treatment_count: usize) -> Result<(), RoomTreatmentError> {
        if treatment_count == 0 || self.adjacent_pairs.is_empty() {
            return Err(RoomTreatmentError::InvalidProblem(
                "spatial treatment constraint requires adjacent pairs",
            ));
        }
        if !self.maximum_adjacent_difference.is_finite()
            || !(0.0..=1.0).contains(&self.maximum_adjacent_difference)
        {
            return Err(RoomTreatmentError::InvalidProblem(
                "maximum adjacent treatment difference must be finite and lie in [0, 1]",
            ));
        }
        let mut seen = HashSet::with_capacity(self.adjacent_pairs.len());
        for &(first, second) in &self.adjacent_pairs {
            if first >= treatment_count || second >= treatment_count || first == second {
                return Err(RoomTreatmentError::InvalidProblem(
                    "spatial treatment pair index is invalid",
                ));
            }
            let canonical = if first < second {
                (first, second)
            } else {
                (second, first)
            };
            if !seen.insert(canonical) {
                return Err(RoomTreatmentError::InvalidProblem(
                    "spatial treatment adjacency contains duplicate pairs",
                ));
            }
        }
        Ok(())
    }

    /// Build exact linear MMA constraints for the current treatment design.
    pub fn linearized_mma_constraints(
        &self,
        treatments: &[f64],
    ) -> Result<Vec<MmaConstraint>, RoomTreatmentError> {
        self.validate(treatments.len())?;
        if treatments
            .iter()
            .any(|treatment| !treatment.is_finite() || !(0.0..=1.0).contains(treatment))
        {
            return Err(RoomTreatmentError::InvalidProblem(
                "spatial treatment values must be finite and lie in [0, 1]",
            ));
        }
        let mut constraints = Vec::with_capacity(self.adjacent_pairs.len() * 2);
        for &(first, second) in &self.adjacent_pairs {
            let mut forward_grad = vec![0.0; treatments.len()];
            forward_grad[first] = 1.0;
            forward_grad[second] = -1.0;
            constraints.push(MmaConstraint {
                grad: forward_grad,
                value: treatments[first] - treatments[second],
                bound: self.maximum_adjacent_difference,
            });

            let mut reverse_grad = vec![0.0; treatments.len()];
            reverse_grad[first] = -1.0;
            reverse_grad[second] = 1.0;
            constraints.push(MmaConstraint {
                grad: reverse_grad,
                value: treatments[second] - treatments[first],
                bound: self.maximum_adjacent_difference,
            });
        }
        Ok(constraints)
    }

    /// Reject a final design that violates the nonlinear absolute-difference
    /// form of the graph constraint.
    pub fn validate_design(&self, treatments: &[f64]) -> Result<(), RoomTreatmentError> {
        self.validate(treatments.len())?;
        if treatments
            .iter()
            .any(|treatment| !treatment.is_finite() || !(0.0..=1.0).contains(treatment))
        {
            return Err(RoomTreatmentError::ConstraintViolation(
                "spatial treatment design is outside [0, 1]".to_string(),
            ));
        }
        let tolerance = 1.0e-8 * self.maximum_adjacent_difference.max(1.0);
        for &(first, second) in &self.adjacent_pairs {
            let difference = (treatments[first] - treatments[second]).abs();
            if difference > self.maximum_adjacent_difference + tolerance {
                return Err(RoomTreatmentError::ConstraintViolation(format!(
                    "adjacent treatment pair ({first}, {second}) differs by {difference}, maximum is {}",
                    self.maximum_adjacent_difference
                )));
            }
        }
        Ok(())
    }
}

/// Per-term aggregates of one objective evaluation, so users can see the
/// trade-offs between target error, seat variance, minimum level, and cost.
#[derive(Debug, Clone, Copy, Default)]
pub struct TreatmentObjectiveBreakdown {
    /// Total objective value (sum of the other fields).
    pub total: f64,
    /// Band-weighted target-level error.
    pub target: f64,
    /// Band-weighted seat variance.
    pub variance: f64,
    /// Band-weighted minimum-level penalty.
    pub min_level: f64,
    /// Treatment cost term.
    pub cost: f64,
}

/// Objective value, per-term breakdown, and treatment gradient of one
/// multi-frequency evaluation.
#[derive(Debug, Clone)]
pub struct TreatmentEvaluation {
    /// Total objective `J`.
    pub objective: f64,
    /// Per-term aggregates.
    pub breakdown: TreatmentObjectiveBreakdown,
    /// Total gradient `dJ/dt_i` per treatment boundary, in input order.
    pub gradient: Vec<f64>,
}

/// Settings for multi-frequency treatment optimization.
#[derive(Debug, Clone)]
pub struct MultiFrequencyRoomTreatmentConfig {
    /// Maximum MMA iterations.
    pub max_iterations: usize,
    /// Stop when the largest treatment change is below this value.
    pub tolerance: f64,
    /// Maximum (area-weighted) mean treatment fraction.
    pub treatment_fraction: f64,
    /// Initial treatment fraction for each boundary group.
    pub initial_treatment: f64,
    /// Objective weights and targets.
    pub objective: TreatmentObjectiveConfig,
    /// Cap on the number of frequency lanes. `None` derives the lane count
    /// from `rayon::current_num_threads()`; `Some(1)` is the serial path.
    pub max_frequency_lanes: Option<usize>,
    /// Optional calibrated reduced-order RT60 bounds.
    pub decay_constraint: Option<TreatmentDecayConstraint>,
    /// Optional adjacency-based minimum-contrast patch constraint.
    pub spatial_constraint: Option<TreatmentSpatialConstraint>,
    /// Optional continuous panel-density filter.
    pub density_filter: Option<TreatmentDensityFilter>,
}

impl Default for MultiFrequencyRoomTreatmentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 40,
            tolerance: 1.0e-3,
            treatment_fraction: 0.5,
            initial_treatment: 0.25,
            objective: TreatmentObjectiveConfig::default(),
            max_frequency_lanes: None,
            decay_constraint: None,
            spatial_constraint: None,
            density_filter: None,
        }
    }
}

/// Result of multi-frequency room-treatment optimization.
#[derive(Debug, Clone)]
pub struct MultiFrequencyRoomTreatmentResult {
    /// Optimized treatment value per boundary, in input order.
    pub treatments: Vec<f64>,
    /// Total objective after each evaluated design.
    pub objective_history: Vec<f64>,
    /// Per-term objective breakdown after each evaluated design.
    pub breakdown_history: Vec<TreatmentObjectiveBreakdown>,
    /// Predicted RT60 for each evaluated design when decay constraints are
    /// enabled. It is empty when no decay model was configured and otherwise
    /// has the same length as `objective_history`.
    pub rt60_history: Vec<f64>,
    /// Number of completed MMA iterations.
    pub iterations: usize,
    /// Whether the treatment-change criterion was met.
    pub converged: bool,
}

/// Passivity check for one interpolated Robin coefficient.
///
/// Under Sonium's `exp(-iωt)` convention a passive (dissipative) local
/// admittance contributes a non-positive imaginary part to the Robin
/// coefficient — see `impedance_to_robin_coefficient`
/// (`sonium-fem/.../acoustics/impedance.rs`), where `α = −i·k·ρ₀c₀/Z_s` with
/// `Re(Z_s) ≥ 0` gives `Im(α) ≤ 0`, and the Sommerfeld ABC `α = −ik`. The
/// interpolated coefficient `base + t·direction` is affine in `t`, so it is
/// passive for all `t ∈ [0, 1]` iff both endpoints are passive. The real
/// (reactive) part is unconstrained.
pub fn passive_coefficient_ok(base: Complex64, direction: Complex64) -> bool {
    base.im <= PASSIVITY_TOLERANCE && (base + direction).im <= PASSIVITY_TOLERANCE
}

/// Verify that every treatment direction stays passive over the full band.
///
/// `base_coefficients[f][i]` is the base Robin coefficient of treatment
/// boundary `i` at frequency `f` (use 0 when the marker has no base
/// coefficient). Returns [`RoomTreatmentError::NonPassive`] with
/// frequency/boundary detail on the first violation.
pub fn validate_passivity(
    frequencies_hz: &[f64],
    base_coefficients: &[Vec<Complex64>],
    treatments: &[TreatmentBoundary],
) -> Result<(), RoomTreatmentError> {
    if frequencies_hz.len() != base_coefficients.len() {
        return Err(RoomTreatmentError::InvalidProblem(
            "passivity frequency and coefficient rows differ in length",
        ));
    }
    validate_treatment_boundaries(treatments)?;
    for (frequency_hz, coefficients) in frequencies_hz.iter().zip(base_coefficients) {
        if !frequency_hz.is_finite() || *frequency_hz <= 0.0 {
            return Err(RoomTreatmentError::InvalidProblem(
                "passivity frequencies must be finite and positive",
            ));
        }
        if coefficients.len() != treatments.len() {
            return Err(RoomTreatmentError::InvalidProblem(
                "passivity coefficient row and treatment count differ",
            ));
        }
        for (boundary, treatment) in treatments.iter().enumerate() {
            let base = coefficients[boundary];
            if !base.re.is_finite() || !base.im.is_finite() {
                return Err(RoomTreatmentError::InvalidProblem(
                    "passivity base coefficient must be finite",
                ));
            }
            if !passive_coefficient_ok(base, treatment.coefficient_direction) {
                let direction = treatment.coefficient_direction;
                let at = if base.im > PASSIVITY_TOLERANCE {
                    0.0
                } else {
                    1.0
                };
                return Err(RoomTreatmentError::NonPassive(format!(
                    "marker {} at {frequency_hz} Hz: coefficient {} + t·({}) has a positive \
                     imaginary part ({}) at t={at} under the exp(-iωt) convention",
                    treatment.marker,
                    base,
                    direction,
                    (base + at * direction).im,
                )));
            }
        }
    }
    Ok(())
}

struct FrequencyData {
    frequency_hz: f64,
    weight: f64,
    base_operator: CsrMatrix<Complex64>,
    rhs: Vec<Complex64>,
    /// Base Robin coefficient per treatment boundary (for passivity checks).
    base_coefficients: Vec<Complex64>,
}

#[derive(Clone, Default)]
struct WarmStartCache {
    forward_guess: Vec<Option<Vec<Complex64>>>,
    adjoint_guess: Vec<Option<Vec<Complex64>>>,
}

/// Per-lane solver scratch: one reusable operator matrix plus per-frequency
/// warm starts for the forward and adjoint solves. The cache is shared only
/// for publishing completed solves; each lane still owns its mutable guess
/// vectors, so the solve itself remains lock-free and deterministic.
struct LaneScratch {
    operator: CsrMatrix<Complex64>,
    forward_guess: Vec<Option<Vec<Complex64>>>,
    adjoint_guess: Vec<Option<Vec<Complex64>>>,
    warm_starts: Arc<Mutex<WarmStartCache>>,
}

struct FrequencyEvaluation {
    target: f64,
    variance: f64,
    min_level: f64,
    gradient: Vec<f64>,
}

/// Conservative live-byte estimate for one frequency lane.
///
/// A lane owns the assembled forward CSR values, a conjugate-transpose CSR
/// during the adjoint solve, forward/adjoint vectors and retained warm
/// starts. The iterative solver also allocates Krylov vectors and residual
/// workspaces. The previous estimate counted only one CSR/vector pair and
/// routinely admitted twice as many lanes as the machine could hold.
fn lane_memory_estimate_bytes(nnz: u64, num_dofs: u64) -> u64 {
    nnz.saturating_mul(128)
        .saturating_add(num_dofs.saturating_mul(256))
}

/// Multi-frequency room treatment problem on a fixed mesh.
///
/// The operator for a design `t` at frequency `f` is the affine combination
/// `A_f(t) = base_f + Σ_i t_i·T_i`; all matrices share one CSR structure, so
/// re-assembly per MMA iteration only rewrites the values array. One forward
/// and one conjugate-transpose solve per frequency produce all treatment
/// gradients; frequencies are evaluated in parallel over contiguous lanes
/// ([`run_lanes`]). Completed frequency solves are retained as warm starts for
/// the next objective evaluation (including the next MMA iteration), while
/// every frequency remains independent and lane-count deterministic.
pub struct MultiFrequencyRoomTreatmentProblem {
    frequencies: Vec<FrequencyData>,
    treatment_operators: Vec<CsrMatrix<Complex64>>,
    treatments: Vec<TreatmentBoundary>,
    listeners: Vec<ListenerFunctional>,
    solver: SolverConfig,
    warm_starts: Arc<Mutex<WarmStartCache>>,
}

impl MultiFrequencyRoomTreatmentProblem {
    /// Build a multi-frequency problem from the same preassembled matrices
    /// used by room FEM. `setups` are sorted by `frequency_hz`; the
    /// treatment operators and listener set are shared across frequencies.
    ///
    /// Every treatment direction is validated for passivity over the whole
    /// band at construction (see [`validate_passivity`]).
    pub fn from_helmholtz(
        assembler: &HelmholtzAssembler,
        setups: &[TreatmentFrequencySetup],
        treatments: &[TreatmentBoundary],
        listeners: Vec<ListenerFunctional>,
        solver: SolverConfig,
    ) -> Result<Self, RoomTreatmentError> {
        if setups.is_empty() {
            return Err(RoomTreatmentError::InvalidProblem(
                "at least one frequency is required",
            ));
        }
        if listeners.len() < 2 {
            return Err(RoomTreatmentError::InvalidProblem(
                "seat-variance optimization requires at least two listeners",
            ));
        }
        if treatments.is_empty() {
            return Err(RoomTreatmentError::InvalidProblem(
                "at least one treatment boundary is required",
            ));
        }
        validate_treatment_boundaries(treatments)?;
        for listener in &listeners {
            if listener.entries.is_empty()
                || listener.entries.iter().any(|&(dof, weight)| {
                    dof >= assembler.num_rows || !weight.re.is_finite() || !weight.im.is_finite()
                })
            {
                return Err(RoomTreatmentError::InvalidProblem(
                    "listener interpolation is empty or invalid",
                ));
            }
        }
        for setup in setups {
            if !setup.frequency_hz.is_finite()
                || setup.frequency_hz <= 0.0
                || !setup.weight.is_finite()
                || setup.weight <= 0.0
            {
                return Err(RoomTreatmentError::InvalidProblem(
                    "frequency and band weight must be finite and positive",
                ));
            }
            if setup.rhs.len() != assembler.num_rows {
                return Err(RoomTreatmentError::InvalidProblem(
                    "RHS size does not match Helmholtz operator",
                ));
            }
        }

        let mut ordered = setups.to_vec();
        ordered.sort_by(|a, b| a.frequency_hz.total_cmp(&b.frequency_hz));
        if ordered
            .windows(2)
            .any(|pair| pair[0].frequency_hz == pair[1].frequency_hz)
        {
            return Err(RoomTreatmentError::InvalidProblem(
                "duplicate frequencies are not allowed",
            ));
        }

        let treatment_operators = treatments
            .iter()
            .map(|treatment| {
                let boundary_values = assembler.boundary_values.get(&treatment.marker).ok_or(
                    RoomTreatmentError::InvalidProblem(
                        "treatment marker has no assembled boundary mass matrix",
                    ),
                )?;
                let values = boundary_values
                    .iter()
                    .map(|&value| treatment.coefficient_direction * value)
                    .collect();
                Ok(CsrMatrix::from_raw_parts(
                    assembler.num_rows,
                    assembler.num_rows,
                    assembler.row_ptrs.clone(),
                    assembler.col_indices.clone(),
                    values,
                ))
            })
            .collect::<Result<Vec<_>, RoomTreatmentError>>()?;

        let frequencies = ordered
            .iter()
            .map(|setup| {
                let base_operator =
                    assembler.assemble(setup.wavenumber, &setup.base_boundary_coefficients);
                let base_coefficients = treatments
                    .iter()
                    .map(|treatment| {
                        setup
                            .base_boundary_coefficients
                            .get(&treatment.marker)
                            .copied()
                            .unwrap_or(Complex64::new(0.0, 0.0))
                    })
                    .collect();
                FrequencyData {
                    frequency_hz: setup.frequency_hz,
                    weight: setup.weight,
                    base_operator,
                    rhs: setup.rhs.clone(),
                    base_coefficients,
                }
            })
            .collect::<Vec<_>>();

        let frequencies_hz = frequencies
            .iter()
            .map(|frequency| frequency.frequency_hz)
            .collect::<Vec<_>>();
        let base_coefficients = frequencies
            .iter()
            .map(|frequency| frequency.base_coefficients.clone())
            .collect::<Vec<_>>();
        validate_passivity(&frequencies_hz, &base_coefficients, treatments)?;

        Ok(Self {
            warm_starts: Arc::new(Mutex::new(WarmStartCache {
                forward_guess: vec![None; frequencies.len()],
                adjoint_guess: vec![None; frequencies.len()],
            })),
            frequencies,
            treatment_operators,
            treatments: treatments.to_vec(),
            listeners,
            solver,
        })
    }

    /// Re-run the band-wide passivity validation on the stored data.
    pub fn check_passivity(&self) -> Result<(), RoomTreatmentError> {
        let frequencies_hz = self
            .frequencies
            .iter()
            .map(|frequency| frequency.frequency_hz)
            .collect::<Vec<_>>();
        let base_coefficients = self
            .frequencies
            .iter()
            .map(|frequency| frequency.base_coefficients.clone())
            .collect::<Vec<_>>();
        validate_passivity(&frequencies_hz, &base_coefficients, &self.treatments)
    }

    /// Treatment boundary definitions, in input order.
    pub fn treatment_boundaries(&self) -> &[TreatmentBoundary] {
        &self.treatments
    }

    /// Sorted frequency grid in Hz.
    pub fn frequencies_hz(&self) -> Vec<f64> {
        self.frequencies
            .iter()
            .map(|frequency| frequency.frequency_hz)
            .collect()
    }

    #[cfg(test)]
    fn warm_start_count(&self) -> usize {
        self.warm_starts
            .lock()
            .map(|cache| {
                cache
                    .forward_guess
                    .iter()
                    .chain(cache.adjoint_guess.iter())
                    .filter(|guess| guess.is_some())
                    .count()
            })
            .unwrap_or_default()
    }

    fn validate_treatments(&self, treatments: &[f64]) -> Result<(), RoomTreatmentError> {
        if treatments.len() != self.treatment_operators.len()
            || treatments
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(RoomTreatmentError::InvalidProblem(
                "treatment vector has the wrong size or lies outside [0, 1]",
            ));
        }
        Ok(())
    }

    fn solve_with_scratch(
        &self,
        operator: &CsrMatrix<Complex64>,
        rhs: &[Complex64],
        guess: Option<&[Complex64]>,
    ) -> Result<Vec<Complex64>, RoomTreatmentError> {
        solve_csr_with_guess(operator, rhs, guess, &self.solver)
            .map(|solution| solution.values)
            .map_err(|error| RoomTreatmentError::Solve(error.to_string()))
    }

    /// One frequency: merge operator values, forward solve, objective terms,
    /// adjoint solve, per-treatment gradient.
    fn frequency_evaluation(
        &self,
        frequency_index: usize,
        treatments: &[f64],
        objective: &TreatmentObjectiveConfig,
        scratch: &mut LaneScratch,
    ) -> Result<FrequencyEvaluation, RoomTreatmentError> {
        let frequency = &self.frequencies[frequency_index];
        scratch
            .operator
            .values
            .copy_from_slice(&frequency.base_operator.values);
        for (treatment, derivative) in treatments.iter().zip(&self.treatment_operators) {
            for (value, derivative_value) in
                scratch.operator.values.iter_mut().zip(&derivative.values)
            {
                *value += *treatment * *derivative_value;
            }
        }

        let state = self.solve_with_scratch(
            &scratch.operator,
            &frequency.rhs,
            scratch.forward_guess[frequency_index].as_deref(),
        )?;
        scratch.forward_guess[frequency_index] = Some(state.clone());

        let (target, variance, min_level, state_gradient) = match objective.kind {
            TreatmentObjectiveKind::SeatVariance => {
                let (variance, gradient) = seat_variance_objective(&self.listeners, &state);
                (0.0, variance, 0.0, gradient)
            }
            TreatmentObjectiveKind::Production => {
                let (terms, gradient) = production_objective_and_state_gradient(
                    &self.listeners,
                    &state,
                    objective,
                    objective.target_level_db(frequency_index),
                );
                (terms.0, terms.1, terms.2, gradient)
            }
        };

        let adjoint_operator = conjugate_transpose(&scratch.operator);
        let adjoint_rhs = state_gradient
            .into_iter()
            .map(|value| -value)
            .collect::<Vec<_>>();
        let adjoint = self.solve_with_scratch(
            &adjoint_operator,
            &adjoint_rhs,
            scratch.adjoint_guess[frequency_index].as_deref(),
        )?;
        scratch.adjoint_guess[frequency_index] = Some(adjoint.clone());

        // Publish completed solves for the next call to `evaluate` (usually
        // the next MMA iteration). A poisoned cache cannot affect numerical
        // correctness, so leave it unchanged rather than panicking.
        if let Ok(mut cache) = scratch.warm_starts.lock() {
            cache.forward_guess[frequency_index] = Some(state.clone());
            cache.adjoint_guess[frequency_index] = Some(adjoint.clone());
        }

        let gradient = treatment_gradients(&self.treatment_operators, &state, &adjoint);
        Ok(FrequencyEvaluation {
            target,
            variance,
            min_level,
            gradient,
        })
    }

    /// Evaluate the band-aggregated objective, its per-term breakdown, and
    /// the exact discrete-adjoint gradient at a design point.
    ///
    /// Frequencies are evaluated in parallel over contiguous lanes; the
    /// aggregation runs in fixed frequency order, so results are bitwise
    /// identical for any lane count (warm starts are per-frequency and hence
    /// lane-count independent).
    pub fn evaluate(
        &self,
        treatments: &[f64],
        objective: &TreatmentObjectiveConfig,
        max_frequency_lanes: Option<usize>,
    ) -> Result<TreatmentEvaluation, RoomTreatmentError> {
        self.evaluate_with_cancel(treatments, objective, max_frequency_lanes, None)
    }

    /// Evaluate with an optional cooperative cancellation flag. Set the flag
    /// from a UI/job supervisor to stop a long frequency sweep between lane
    /// items; the default [`Self::evaluate`] preserves the historical
    /// non-cancellable behavior.
    pub fn evaluate_with_cancel(
        &self,
        treatments: &[f64],
        objective: &TreatmentObjectiveConfig,
        max_frequency_lanes: Option<usize>,
        cancel: Option<Arc<AtomicBool>>,
    ) -> Result<TreatmentEvaluation, RoomTreatmentError> {
        self.validate_treatments(treatments)?;
        objective.validate(self.frequencies.len())?;

        let num_frequencies = self.frequencies.len();
        let nnz = self.frequencies[0].base_operator.values.len() as u64;
        let num_dofs = self.frequencies[0].base_operator.num_rows as u64;
        let lane_config = LaneConfig {
            max_lanes: max_frequency_lanes,
            thread_budget: rayon::current_num_threads(),
            memory_budget_bytes: None,
        };
        // Include both CSR operators, retained warm starts, and Krylov
        // workspaces; this is intentionally conservative to avoid OOMs when
        // a memory budget is supplied by a caller.
        let plan = try_plan_lanes(
            num_frequencies,
            |_| lane_memory_estimate_bytes(nnz, num_dofs),
            &lane_config,
        )
        .map_err(|error| RoomTreatmentError::Solve(format!("frequency lane admission: {error}")))?;
        let indices = (0..num_frequencies).collect::<Vec<_>>();
        let template = CsrMatrix::from_raw_parts(
            self.frequencies[0].base_operator.num_rows,
            self.frequencies[0].base_operator.num_cols,
            self.frequencies[0].base_operator.row_ptrs.clone(),
            self.frequencies[0].base_operator.col_indices.clone(),
            vec![Complex64::new(0.0, 0.0); nnz as usize],
        );
        let (forward_guess, adjoint_guess) = self
            .warm_starts
            .lock()
            .map(|cache| (cache.forward_guess.clone(), cache.adjoint_guess.clone()))
            .map_err(|_| {
                RoomTreatmentError::Solve("frequency warm-start cache is poisoned".to_string())
            })?;
        let warm_starts = Arc::clone(&self.warm_starts);
        let outcome = run_lanes(
            &indices,
            &plan,
            || LaneScratch {
                operator: template.clone(),
                forward_guess: forward_guess.clone(),
                adjoint_guess: adjoint_guess.clone(),
                warm_starts: Arc::clone(&warm_starts),
            },
            &|scratch: &mut LaneScratch, _index, &frequency_index: &usize| {
                self.frequency_evaluation(frequency_index, treatments, objective, scratch)
            },
            cancel,
        );
        if outcome.cancelled {
            return Err(RoomTreatmentError::Solve(
                "frequency lane execution was cancelled".to_string(),
            ));
        }
        let mut evaluations = Vec::with_capacity(num_frequencies);
        for result in outcome.results {
            evaluations.push(result.ok_or(RoomTreatmentError::Solve(
                "frequency lane execution dropped a result".to_string(),
            ))??);
        }

        self.aggregate_evaluations(treatments, objective, &evaluations)
    }

    fn aggregate_evaluations(
        &self,
        treatments: &[f64],
        objective: &TreatmentObjectiveConfig,
        evaluations: &[FrequencyEvaluation],
    ) -> Result<TreatmentEvaluation, RoomTreatmentError> {
        let count = self.treatment_operators.len();
        let mut breakdown = TreatmentObjectiveBreakdown::default();
        let mut gradient = vec![0.0; count];
        for (frequency, evaluation) in self.frequencies.iter().zip(evaluations) {
            let weight = frequency.weight;
            breakdown.target += weight * evaluation.target;
            breakdown.variance += weight * evaluation.variance;
            breakdown.min_level += weight * evaluation.min_level;
            for (total, partial) in gradient.iter_mut().zip(&evaluation.gradient) {
                *total += weight * partial;
            }
        }
        if objective.cost_weight > 0.0 {
            for (boundary, total) in gradient.iter_mut().enumerate() {
                let area = boundary_area(self.treatments[boundary]);
                let unit_cost = self.treatments[boundary].cost_per_m2.unwrap_or(1.0);
                let contribution = objective.cost_weight * area * unit_cost;
                breakdown.cost += contribution * treatments[boundary];
                *total += contribution;
            }
        }
        breakdown.total =
            breakdown.target + breakdown.variance + breakdown.min_level + breakdown.cost;
        Ok(TreatmentEvaluation {
            objective: breakdown.total,
            breakdown,
            gradient,
        })
    }

    /// Optimize treatment allocation with MMA and an area-weighted
    /// treatment-fraction constraint (`Σ t_i·area_i / Σ area_i <= fraction`;
    /// falls back to a plain mean when areas are absent).
    pub fn optimize(
        &self,
        config: &MultiFrequencyRoomTreatmentConfig,
    ) -> Result<MultiFrequencyRoomTreatmentResult, RoomTreatmentError> {
        self.optimize_with_evaluator(config, |treatments| {
            self.evaluate(treatments, &config.objective, config.max_frequency_lanes)
        })
    }

    fn optimize_with_evaluator(
        &self,
        config: &MultiFrequencyRoomTreatmentConfig,
        mut evaluate: impl FnMut(&[f64]) -> Result<TreatmentEvaluation, RoomTreatmentError>,
    ) -> Result<MultiFrequencyRoomTreatmentResult, RoomTreatmentError> {
        if config.max_iterations == 0
            || !config.tolerance.is_finite()
            || config.tolerance <= 0.0
            || !config.treatment_fraction.is_finite()
            || !(1.0e-3..=1.0).contains(&config.treatment_fraction)
            || !config.initial_treatment.is_finite()
            || !(1.0e-3..=1.0).contains(&config.initial_treatment)
        {
            return Err(RoomTreatmentError::InvalidConfiguration(
                "iterations, tolerance, or treatment fractions are invalid",
            ));
        }
        config.objective.validate(self.frequencies.len())?;

        let count = self.treatment_operators.len();
        if let Some(decay_constraint) = &config.decay_constraint {
            decay_constraint.validate(count)?;
        }
        if let Some(spatial_constraint) = &config.spatial_constraint {
            spatial_constraint.validate(count)?;
        }
        if let Some(density_filter) = &config.density_filter {
            density_filter.validate(count)?;
        }
        let areas = self
            .treatments
            .iter()
            .map(|&boundary| boundary_area(boundary))
            .collect::<Vec<_>>();
        let total_area = areas.iter().sum::<f64>().max(f64::MIN_POSITIVE);
        let mut design_treatments =
            vec![config.initial_treatment.min(config.treatment_fraction); count];
        let mut optimizer = MmaOptimizer::new();
        let mut objective_history = Vec::with_capacity(config.max_iterations);
        let mut breakdown_history = Vec::with_capacity(config.max_iterations);
        let mut rt60_history = config
            .decay_constraint
            .as_ref()
            .map(|_| Vec::with_capacity(config.max_iterations))
            .unwrap_or_default();
        let mut converged = false;

        for _ in 0..config.max_iterations {
            let treatments = config.density_filter.as_ref().map_or_else(
                || Ok(design_treatments.clone()),
                |filter| filter.apply(&design_treatments),
            )?;
            let evaluation = evaluate(&treatments)?;
            objective_history.push(evaluation.objective);
            breakdown_history.push(evaluation.breakdown);
            let mut decay_constraints = if let Some(decay_constraint) = &config.decay_constraint {
                let (rt60, _) = decay_constraint.predict_rt60(&treatments)?;
                rt60_history.push(rt60);
                decay_constraint.linearized_mma_constraints(&treatments)?
            } else {
                Vec::new()
            };
            let mut spatial_constraints =
                if let Some(spatial_constraint) = &config.spatial_constraint {
                    spatial_constraint.linearized_mma_constraints(&treatments)?
                } else {
                    Vec::new()
                };
            if let Some(density_filter) = &config.density_filter {
                for constraint in decay_constraints
                    .iter_mut()
                    .chain(spatial_constraints.iter_mut())
                {
                    constraint.grad = density_filter.pullback_gradient(&constraint.grad)?;
                }
            }
            let fraction_constraint = MmaConstraint {
                grad: {
                    let filtered_gradient = areas
                        .iter()
                        .map(|area| area / total_area)
                        .collect::<Vec<_>>();
                    config.density_filter.as_ref().map_or_else(
                        || Ok(filtered_gradient.clone()),
                        |filter| filter.pullback_gradient(&filtered_gradient),
                    )?
                },
                value: treatments
                    .iter()
                    .zip(&areas)
                    .map(|(treatment, area)| treatment * area)
                    .sum::<f64>()
                    / total_area,
                bound: config.treatment_fraction,
            };
            let mut constraints =
                Vec::with_capacity(1 + decay_constraints.len() + spatial_constraints.len());
            constraints.push(fraction_constraint);
            constraints.extend(decay_constraints);
            constraints.extend(spatial_constraints);
            let gradient = config.density_filter.as_ref().map_or_else(
                || Ok(evaluation.gradient.clone()),
                |filter| filter.pullback_gradient(&evaluation.gradient),
            )?;
            let next = optimizer.update_multi(&gradient, &design_treatments, &constraints, &areas);
            let change = next
                .iter()
                .zip(&design_treatments)
                .map(|(next, current)| (next - current).abs())
                .fold(0.0, f64::max);
            design_treatments = next;
            if change < config.tolerance {
                converged = true;
                break;
            }
        }

        let treatments = if let Some(filter) = &config.density_filter {
            filter.apply(&design_treatments)?
        } else {
            design_treatments
        };
        if let Some(decay_constraint) = &config.decay_constraint {
            let (rt60, _) = decay_constraint.predict_rt60(&treatments)?;
            decay_constraint.validate_rt60_bounds(rt60)?;
        }
        if let Some(spatial_constraint) = &config.spatial_constraint {
            spatial_constraint.validate_design(&treatments)?;
        }

        Ok(MultiFrequencyRoomTreatmentResult {
            iterations: objective_history.len(),
            treatments,
            objective_history,
            breakdown_history,
            rt60_history,
            converged,
        })
    }
}

/// Physical area of a treatment boundary; absent areas default to 1 so the
/// fraction constraint degenerates to the plain mean.
fn boundary_area(boundary: TreatmentBoundary) -> f64 {
    match boundary.area_m2 {
        Some(area) if area.is_finite() && area > 0.0 => area,
        _ => 1.0,
    }
}

/// Production objective at one frequency: target-level error, seat variance,
/// and minimum-level penalty over `L_s = 20·log10(|p_s|/p_ref)`, plus the
/// state gradient with `dJ_f = 2 Re(state_gradientᴴ dp)`.
///
/// With `q_s = |p_s|² + floor`, `L_s = DB_PER_LN·(ln q_s − ln p_ref²)` and
/// `dL_s/dq_s = DB_PER_LN/q_s`; the per-seat `dJ_f/dL_s` chain through the
/// listener weights exactly like the legacy variance objective. The relu
/// min-level term contributes only where `L_s < L_min`; at the kink the
/// subgradient 0 (inactive branch) is used.
fn production_objective_and_state_gradient(
    listeners: &[ListenerFunctional],
    state: &[Complex64],
    objective: &TreatmentObjectiveConfig,
    target_level_db: f64,
) -> ((f64, f64, f64), Vec<Complex64>) {
    let pressures = listeners
        .iter()
        .map(|listener| listener.evaluate(state))
        .collect::<Vec<_>>();
    let (terms, pressure_gradients) = production_pressures(&pressures, objective, target_level_db);
    let mut state_gradient = vec![Complex64::new(0.0, 0.0); state.len()];
    for (listener, pressure_gradient) in listeners.iter().zip(pressure_gradients) {
        for &(dof, weight) in &listener.entries {
            state_gradient[dof] += weight.conj() * pressure_gradient;
        }
    }
    (terms, state_gradient)
}

fn production_pressures(
    pressures: &[Complex64],
    objective: &TreatmentObjectiveConfig,
    target_level_db: f64,
) -> ((f64, f64, f64), Vec<Complex64>) {
    let seats = pressures.len() as f64;
    let log_reference = 2.0 * objective.reference_pressure.ln();
    let powers = pressures
        .iter()
        .map(|pressure| pressure.norm_sqr() + PRESSURE_FLOOR)
        .collect::<Vec<_>>();
    let levels = powers
        .iter()
        .map(|power| DB_PER_LN * (power.ln() - log_reference))
        .collect::<Vec<_>>();
    let mean_level = levels.iter().sum::<f64>() / seats;

    let mut target = 0.0;
    let mut variance = 0.0;
    let mut min_level = 0.0;
    let mut level_gradients = vec![0.0; pressures.len()];
    for (seat, &level) in levels.iter().enumerate() {
        let error = level - target_level_db;
        target += error * error;
        level_gradients[seat] += 2.0 * objective.target_weight * error / seats;

        let residual = level - mean_level;
        variance += residual * residual;
        level_gradients[seat] += 2.0 * objective.variance_weight * residual / seats;

        let deficit = objective.min_level_db - level;
        if deficit > 0.0 {
            min_level += deficit * deficit;
            level_gradients[seat] -= 2.0 * objective.min_level_weight * deficit / seats;
        }
    }
    target *= objective.target_weight / seats;
    variance *= objective.variance_weight / seats;
    min_level *= objective.min_level_weight / seats;

    let gradients = pressures
        .iter()
        .zip(&powers)
        .zip(&level_gradients)
        .map(|((pressure, &power), &level_gradient)| {
            *pressure * (level_gradient * DB_PER_LN / power)
        })
        .collect();
    ((target, variance, min_level), gradients)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_fem::assembly::{MassMatrix, StiffnessMatrix};
    use sonium_fem::solver::SolverType;

    fn triplet_mass(dim: usize, entries: &[(usize, usize, f64)]) -> MassMatrix {
        let mut matrix = MassMatrix::new(dim);
        for &(row, column, value) in entries {
            matrix.add(row, column, value);
        }
        matrix
    }

    fn problem() -> RoomTreatmentProblem {
        let mut stiffness = StiffnessMatrix::new(3);
        for (row, column, value) in [
            (0, 0, 4.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 1, 3.0),
            (1, 2, -0.5),
            (2, 1, -0.5),
            (2, 2, 2.0),
        ] {
            stiffness.add(row, column, value);
        }
        let mass = triplet_mass(3, &[(0, 0, 0.2), (1, 1, 0.2), (2, 2, 0.2)]);
        let boundaries = vec![
            (1, triplet_mass(3, &[(0, 0, 1.0), (1, 1, 0.2)])),
            (2, triplet_mass(3, &[(1, 1, 0.1), (2, 2, 1.0)])),
        ];
        let assembler = HelmholtzAssembler::from_matrices(&stiffness, &mass, &boundaries);
        RoomTreatmentProblem::from_helmholtz(
            &assembler,
            Complex64::new(1.0, 0.0),
            &HashMap::new(),
            &[
                TreatmentBoundary {
                    marker: 1,
                    coefficient_direction: Complex64::new(0.0, -1.5),
                    area_m2: None,
                    cost_per_m2: None,
                },
                TreatmentBoundary {
                    marker: 2,
                    coefficient_direction: Complex64::new(0.0, -0.8),
                    area_m2: None,
                    cost_per_m2: None,
                },
            ],
            vec![
                Complex64::new(1.0, 0.2),
                Complex64::new(0.1, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            vec![
                ListenerFunctional::at_node(0),
                ListenerFunctional::at_node(1),
                ListenerFunctional::at_node(2),
            ],
            SolverConfig {
                solver_type: SolverType::Direct,
                ..SolverConfig::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn room_treatment_adjoint_matches_finite_difference() {
        let problem = problem();
        let treatments = vec![0.3, 0.4];
        let (_, gradient) = problem.objective_gradient(&treatments).unwrap();
        let step = 1.0e-6;
        for parameter in 0..treatments.len() {
            let mut plus = treatments.clone();
            let mut minus = treatments.clone();
            plus[parameter] += step;
            minus[parameter] -= step;
            let plus_objective = problem.objective_gradient(&plus).unwrap().0;
            let minus_objective = problem.objective_gradient(&minus).unwrap().0;
            let finite_difference = (plus_objective - minus_objective) / (2.0 * step);
            let scale = finite_difference.abs().max(1.0);
            assert!(
                (gradient[parameter] - finite_difference).abs() / scale < 1.0e-6,
                "parameter {parameter}: adjoint={}, FD={finite_difference}",
                gradient[parameter]
            );
        }
    }

    #[test]
    fn room_treatment_optimization_reduces_seat_variance() {
        let problem = problem();
        let initial = vec![0.2, 0.2];
        let initial_objective = problem.objective_gradient(&initial).unwrap().0;
        let result = problem
            .optimize(&RoomTreatmentConfig {
                max_iterations: 12,
                treatment_fraction: 0.6,
                initial_treatment: 0.2,
                ..RoomTreatmentConfig::default()
            })
            .unwrap();
        let final_objective = problem.objective_gradient(&result.treatments).unwrap().0;
        assert!(final_objective < initial_objective);
        assert!(
            result
                .treatments
                .iter()
                .all(|value| (0.0..=1.0).contains(value))
        );
        assert!(result.treatments.iter().sum::<f64>() / 2.0 <= 0.6 + 1.0e-8);
    }

    #[test]
    fn legacy_room_treatment_optimization_respects_spatial_constraint() {
        let problem = problem();
        let spatial = TreatmentSpatialConstraint::new(vec![(0, 1)], 0.05, 2).unwrap();
        let result = problem
            .optimize(&RoomTreatmentConfig {
                max_iterations: 10,
                tolerance: 1.0e-3,
                treatment_fraction: 0.6,
                initial_treatment: 0.2,
                spatial_constraint: Some(spatial.clone()),
                density_filter: None,
            })
            .unwrap();
        spatial.validate_design(&result.treatments).unwrap();
        assert!((result.treatments[0] - result.treatments[1]).abs() <= 0.05 + 1.0e-8);
    }

    #[test]
    fn density_filter_preserves_constants_and_pulls_back_gradients() {
        let filter = TreatmentDensityFilter::new(vec![[0.0, 0.0, 0.0]; 2], 1.0).unwrap();
        let filtered = filter.apply(&[0.2, 0.8]).unwrap();
        assert_eq!(filtered, vec![0.5, 0.5]);
        assert_eq!(filter.apply(&[0.35, 0.35]).unwrap(), vec![0.35, 0.35]);
        assert_eq!(
            filter.pullback_gradient(&[1.0, 3.0]).unwrap(),
            vec![2.0, 2.0]
        );
    }

    #[test]
    fn density_filter_rejects_invalid_geometry_and_dimensions() {
        assert!(TreatmentDensityFilter::new(Vec::new(), 1.0).is_err());
        assert!(TreatmentDensityFilter::new(vec![[f64::NAN, 0.0, 0.0]], 1.0).is_err());
        assert!(TreatmentDensityFilter::new(vec![[0.0, 0.0, 0.0]], 0.0).is_err());

        let filter = TreatmentDensityFilter::new(vec![[0.0, 0.0, 0.0]], 1.0).unwrap();
        assert!(filter.apply(&[0.2, 0.3]).is_err());
        assert!(filter.apply(&[f64::INFINITY]).is_err());
    }

    #[test]
    fn legacy_room_treatment_optimization_returns_filtered_design() {
        let problem = problem();
        let filter = TreatmentDensityFilter::new(vec![[0.0, 0.0, 0.0]; 2], 1.0).unwrap();
        let result = problem
            .optimize(&RoomTreatmentConfig {
                max_iterations: 8,
                treatment_fraction: 0.6,
                initial_treatment: 0.2,
                density_filter: Some(filter),
                ..RoomTreatmentConfig::default()
            })
            .unwrap();
        assert!((result.treatments[0] - result.treatments[1]).abs() <= 1.0e-12);
    }

    // ---------------------------------------------------------------------
    // Multi-frequency problem
    // ---------------------------------------------------------------------

    fn direct_solver() -> SolverConfig {
        SolverConfig {
            solver_type: SolverType::Direct,
            ..SolverConfig::default()
        }
    }

    fn try_multi_frequency_problem(
        directions: &[Complex64],
        areas: &[Option<f64>],
        base: &HashMap<usize, Complex64>,
    ) -> Result<MultiFrequencyRoomTreatmentProblem, RoomTreatmentError> {
        let mut stiffness = StiffnessMatrix::new(3);
        for (row, column, value) in [
            (0, 0, 4.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 1, 3.0),
            (1, 2, -0.5),
            (2, 1, -0.5),
            (2, 2, 2.0),
        ] {
            stiffness.add(row, column, value);
        }
        let mass = triplet_mass(3, &[(0, 0, 0.2), (1, 1, 0.2), (2, 2, 0.2)]);
        let boundaries = vec![
            (1, triplet_mass(3, &[(0, 0, 1.0), (1, 1, 0.2)])),
            (2, triplet_mass(3, &[(1, 1, 0.1), (2, 2, 1.0)])),
        ];
        let assembler = HelmholtzAssembler::from_matrices(&stiffness, &mass, &boundaries);
        let treatments = directions
            .iter()
            .enumerate()
            .map(|(index, &direction)| TreatmentBoundary {
                marker: index + 1,
                coefficient_direction: direction,
                area_m2: areas[index],
                cost_per_m2: Some(10.0 + 5.0 * index as f64),
            })
            .collect::<Vec<_>>();
        let rhs = |imag: f64| {
            vec![
                Complex64::new(1.0, 0.2),
                Complex64::new(0.1, imag),
                Complex64::new(0.0, 0.0),
            ]
        };
        let setups = [
            (63.0, Complex64::new(0.8, 0.0), 1.0, rhs(0.0)),
            (125.0, Complex64::new(1.0, 0.0), 2.0, rhs(0.1)),
            (250.0, Complex64::new(1.25, 0.0), 1.5, rhs(-0.1)),
        ]
        .into_iter()
        .map(
            |(frequency_hz, wavenumber, weight, rhs)| TreatmentFrequencySetup {
                frequency_hz,
                wavenumber,
                weight,
                base_boundary_coefficients: base.clone(),
                rhs,
            },
        )
        .collect::<Vec<_>>();
        MultiFrequencyRoomTreatmentProblem::from_helmholtz(
            &assembler,
            &setups,
            &treatments,
            vec![
                ListenerFunctional::at_node(0),
                ListenerFunctional::at_node(1),
                ListenerFunctional::at_node(2),
            ],
            direct_solver(),
        )
    }

    fn multi_frequency_problem(
        directions: &[Complex64],
        areas: &[Option<f64>],
        base: &HashMap<usize, Complex64>,
    ) -> MultiFrequencyRoomTreatmentProblem {
        try_multi_frequency_problem(directions, areas, base).unwrap()
    }

    fn passive_problem(areas: &[Option<f64>]) -> MultiFrequencyRoomTreatmentProblem {
        multi_frequency_problem(
            &[Complex64::new(0.0, -1.5), Complex64::new(0.0, -0.8)],
            areas,
            &HashMap::new(),
        )
    }

    fn production_objective() -> TreatmentObjectiveConfig {
        TreatmentObjectiveConfig {
            kind: TreatmentObjectiveKind::Production,
            target_weight: 1.0,
            variance_weight: 0.3,
            min_level_weight: 0.2,
            min_level_db: 40.0,
            target_levels_db: vec![-20.0, -10.0, 0.0],
            reference_pressure: 1.0,
            cost_weight: 0.1,
        }
    }

    #[test]
    fn default_objective_uses_acoustic_reference_pressure() {
        assert_eq!(
            TreatmentObjectiveConfig::default().reference_pressure,
            REFERENCE_PRESSURE
        );
    }

    #[test]
    fn multi_frequency_adjoint_matches_finite_difference() {
        let problem = passive_problem(&[Some(2.0), Some(1.0)]);
        let objective = production_objective();
        let treatments = vec![0.3, 0.4];
        let evaluation = problem.evaluate(&treatments, &objective, Some(2)).unwrap();
        let step = 1.0e-6;
        for parameter in 0..treatments.len() {
            let mut plus = treatments.clone();
            let mut minus = treatments.clone();
            plus[parameter] += step;
            minus[parameter] -= step;
            let plus_objective = problem
                .evaluate(&plus, &objective, Some(2))
                .unwrap()
                .objective;
            let minus_objective = problem
                .evaluate(&minus, &objective, Some(2))
                .unwrap()
                .objective;
            let finite_difference = (plus_objective - minus_objective) / (2.0 * step);
            let scale = finite_difference.abs().max(1.0);
            assert!(
                (evaluation.gradient[parameter] - finite_difference).abs() / scale < 1.0e-4,
                "parameter {parameter}: adjoint={}, FD={finite_difference}",
                evaluation.gradient[parameter]
            );
        }
        // Breakdown aggregates are consistent with the total.
        let breakdown = evaluation.breakdown;
        assert!(
            (breakdown.total
                - (breakdown.target + breakdown.variance + breakdown.min_level + breakdown.cost))
                .abs()
                < 1.0e-9
        );
        assert!(breakdown.target > 0.0 && breakdown.min_level > 0.0 && breakdown.cost > 0.0);
    }

    #[test]
    fn production_objective_prefers_target_match_over_uniformly_quiet() {
        let listeners = vec![
            ListenerFunctional::at_node(0),
            ListenerFunctional::at_node(1),
            ListenerFunctional::at_node(2),
        ];
        let objective = TreatmentObjectiveConfig {
            kind: TreatmentObjectiveKind::Production,
            target_weight: 1.0,
            variance_weight: 1.0,
            min_level_weight: 1.0,
            min_level_db: -6.0,
            target_levels_db: vec![-6.0],
            reference_pressure: 1.0,
            cost_weight: 0.0,
        };
        // Seat level exactly at the -6 dB target: |p|² = 10^(-0.6).
        let amplitude = 10.0_f64.powf(-0.3);
        let on_target = vec![
            Complex64::new(amplitude, 0.0),
            Complex64::new(0.0, amplitude),
            Complex64::new(-amplitude, 0.0),
        ];
        let quiet = on_target
            .iter()
            .map(|pressure| pressure * 1.0e-3)
            .collect::<Vec<_>>();
        let total = |state: &[Complex64]| {
            let (terms, _) =
                production_objective_and_state_gradient(&listeners, state, &objective, -6.0);
            terms.0 + terms.1 + terms.2
        };
        // The review's failure mode: a uniformly quiet design must score
        // WORSE than a target-matching design under the production objective.
        assert!(total(&quiet) > total(&on_target) + 100.0);

        // Variance-only cannot tell the two uniform designs apart, and
        // prefers uniform quiet over an uneven loud design.
        let uneven = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.25, 0.0),
        ];
        let (variance_on_target, _) = seat_variance_objective(&listeners, &on_target);
        let (variance_quiet, _) = seat_variance_objective(&listeners, &quiet);
        let (variance_uneven, _) = seat_variance_objective(&listeners, &uneven);
        assert!(variance_on_target < 1.0e-24 && variance_quiet < 1.0e-24);
        assert!(variance_uneven > 1.0e-2);
    }

    #[test]
    fn multi_frequency_variance_mode_matches_single_frequency() {
        let single = problem();
        // Build a one-frequency problem with the same k=1.0 data.
        let mut stiffness = StiffnessMatrix::new(3);
        for (row, column, value) in [
            (0, 0, 4.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 1, 3.0),
            (1, 2, -0.5),
            (2, 1, -0.5),
            (2, 2, 2.0),
        ] {
            stiffness.add(row, column, value);
        }
        let mass = triplet_mass(3, &[(0, 0, 0.2), (1, 1, 0.2), (2, 2, 0.2)]);
        let boundaries = vec![
            (1, triplet_mass(3, &[(0, 0, 1.0), (1, 1, 0.2)])),
            (2, triplet_mass(3, &[(1, 1, 0.1), (2, 2, 1.0)])),
        ];
        let assembler = HelmholtzAssembler::from_matrices(&stiffness, &mass, &boundaries);
        let treatments = [
            TreatmentBoundary {
                marker: 1,
                coefficient_direction: Complex64::new(0.0, -1.5),
                area_m2: None,
                cost_per_m2: None,
            },
            TreatmentBoundary {
                marker: 2,
                coefficient_direction: Complex64::new(0.0, -0.8),
                area_m2: None,
                cost_per_m2: None,
            },
        ];
        let listeners = vec![
            ListenerFunctional::at_node(0),
            ListenerFunctional::at_node(1),
            ListenerFunctional::at_node(2),
        ];
        let setup = TreatmentFrequencySetup {
            frequency_hz: 100.0,
            wavenumber: Complex64::new(1.0, 0.0),
            weight: 1.0,
            base_boundary_coefficients: HashMap::new(),
            rhs: vec![
                Complex64::new(1.0, 0.2),
                Complex64::new(0.1, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        };
        let multi = MultiFrequencyRoomTreatmentProblem::from_helmholtz(
            &assembler,
            &[setup],
            &treatments,
            listeners,
            direct_solver(),
        )
        .unwrap();
        let variance_objective = TreatmentObjectiveConfig {
            kind: TreatmentObjectiveKind::SeatVariance,
            ..TreatmentObjectiveConfig::default()
        };
        let treatments_vec = vec![0.3, 0.4];
        let (single_objective, single_gradient) =
            single.objective_gradient(&treatments_vec).unwrap();
        let multi_evaluation = multi
            .evaluate(&treatments_vec, &variance_objective, Some(1))
            .unwrap();
        assert!(
            (multi_evaluation.objective - single_objective).abs() < 1.0e-12,
            "multi={} single={single_objective}",
            multi_evaluation.objective
        );
        for (multi_gradient, single_gradient) in
            multi_evaluation.gradient.iter().zip(&single_gradient)
        {
            assert!(
                (multi_gradient - single_gradient).abs() < 1.0e-12,
                "multi={multi_gradient} single={single_gradient}"
            );
        }

        // The second evaluation must reuse the completed forward and adjoint
        // vectors rather than rebuilding an empty scratch state. The cache is
        // private, but counting populated entries makes this contract
        // regression-testable without exposing implementation details.
        assert_eq!(multi.warm_start_count(), 2);
        let repeated = multi
            .evaluate(&treatments_vec, &variance_objective, Some(1))
            .unwrap();
        assert_eq!(multi.warm_start_count(), 2);
        assert_eq!(repeated.objective, multi_evaluation.objective);
        assert_eq!(repeated.gradient, multi_evaluation.gradient);

        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        assert!(matches!(
            multi.evaluate_with_cancel(
                &treatments_vec,
                &variance_objective,
                Some(1),
                Some(cancel),
            ),
            Err(RoomTreatmentError::Solve(message)) if message.contains("cancelled")
        ));
    }

    #[test]
    fn lane_memory_estimate_accounts_for_solver_workspace() {
        let estimate = lane_memory_estimate_bytes(1_000, 200);
        // At least two CSR copies, four persistent vectors, and Krylov work
        // buffers must be budgeted; the old 64/64 estimate was 76_800 bytes.
        assert!(estimate >= 128_000 + 51_200);
        assert_eq!(lane_memory_estimate_bytes(u64::MAX, u64::MAX), u64::MAX);
    }

    #[test]
    fn active_direction_is_rejected_with_detail() {
        let error = try_multi_frequency_problem(
            &[Complex64::new(0.0, 0.5), Complex64::new(0.0, -0.8)],
            &[None, None],
            &HashMap::new(),
        );
        let Err(RoomTreatmentError::NonPassive(message)) = error else {
            panic!("an active direction must be rejected")
        };
        assert!(message.contains("marker 1"), "message: {message}");
        assert!(message.contains("63"), "message: {message}");
    }

    #[test]
    fn active_base_coefficient_is_rejected() {
        let mut base = HashMap::new();
        base.insert(1, Complex64::new(0.0, 0.4));
        let error = try_multi_frequency_problem(
            &[Complex64::new(0.0, -1.5), Complex64::new(0.0, -0.8)],
            &[None, None],
            &base,
        );
        assert!(matches!(error, Err(RoomTreatmentError::NonPassive(_))));
    }

    #[test]
    fn passivity_rejects_malformed_rows_without_indexing_or_panicking() {
        let treatment = TreatmentBoundary {
            marker: 1,
            coefficient_direction: Complex64::new(0.0, -1.0),
            area_m2: None,
            cost_per_m2: None,
        };
        assert!(matches!(
            validate_passivity(&[100.0], &[], &[treatment]),
            Err(RoomTreatmentError::InvalidProblem(message))
                if message.contains("rows differ")
        ));
        assert!(matches!(
            validate_passivity(&[100.0], &[vec![]], &[treatment]),
            Err(RoomTreatmentError::InvalidProblem(message))
                if message.contains("treatment count")
        ));
        assert!(matches!(
            validate_passivity(&[f64::NAN], &[vec![Complex64::new(0.0, 0.0)]], &[treatment]),
            Err(RoomTreatmentError::InvalidProblem(message))
                if message.contains("frequencies")
        ));
    }

    #[test]
    fn invalid_treatment_metadata_is_rejected_before_assembly() {
        let Err(invalid_area) = try_multi_frequency_problem(
            &[Complex64::new(0.0, -1.5), Complex64::new(0.0, -0.8)],
            &[Some(f64::NAN), Some(1.0)],
            &HashMap::new(),
        ) else {
            panic!("invalid treatment area must be rejected")
        };
        assert!(matches!(
            invalid_area,
            RoomTreatmentError::InvalidProblem(message)
                if message.contains("area")
        ));

        let Err(invalid_direction) = try_multi_frequency_problem(
            &[Complex64::new(f64::NAN, -1.5), Complex64::new(0.0, -0.8)],
            &[None, None],
            &HashMap::new(),
        ) else {
            panic!("invalid treatment direction must be rejected")
        };
        assert!(matches!(
            invalid_direction,
            RoomTreatmentError::InvalidProblem(message)
                if message.contains("coefficient direction")
        ));
    }

    #[test]
    fn passive_band_wide_direction_is_accepted() {
        let mut base = HashMap::new();
        base.insert(1, Complex64::new(0.1, -0.2));
        let problem = multi_frequency_problem(
            &[Complex64::new(0.2, -1.5), Complex64::new(0.0, -0.8)],
            &[None, None],
            &base,
        );
        assert!(problem.check_passivity().is_ok());
        assert_eq!(problem.frequencies_hz(), vec![63.0, 125.0, 250.0]);
        // Endpoint check helper: affine in t, so endpoints decide.
        assert!(passive_coefficient_ok(
            Complex64::new(0.1, -0.2),
            Complex64::new(0.2, -1.5)
        ));
        assert!(!passive_coefficient_ok(
            Complex64::new(0.0, 0.1),
            Complex64::new(0.0, -1.5)
        ));
    }

    #[test]
    fn lane_counts_give_bitwise_identical_results() {
        let problem = passive_problem(&[Some(2.0), Some(1.0)]);
        let objective = production_objective();
        let treatments = vec![0.3, 0.4];
        let reference = problem.evaluate(&treatments, &objective, Some(1)).unwrap();
        for lanes in [2, 3] {
            let evaluation = problem
                .evaluate(&treatments, &objective, Some(lanes))
                .unwrap();
            // Bitwise: per-frequency work and the fixed-order aggregation do
            // not depend on the lane assignment (direct solver, and warm
            // starts are per-frequency so they are lane-count independent).
            assert_eq!(evaluation.objective, reference.objective, "lanes={lanes}");
            assert_eq!(evaluation.gradient, reference.gradient, "lanes={lanes}");
        }
    }

    #[test]
    fn fixed_mesh_reanalysis_does_not_reuse_stale_design_objective() {
        let problem = passive_problem(&[Some(2.0), Some(1.0)]);
        let objective = production_objective();
        let first = problem.evaluate(&[0.2, 0.2], &objective, Some(2)).unwrap();
        let changed = problem.evaluate(&[0.8, 0.1], &objective, Some(2)).unwrap();
        assert_ne!(first.objective, changed.objective);

        let repeated = problem.evaluate(&[0.8, 0.1], &objective, Some(2)).unwrap();
        assert_eq!(changed.objective, repeated.objective);
        assert_eq!(changed.gradient, repeated.gradient);
    }

    #[test]
    fn area_weighted_fraction_constraint_is_respected() {
        let problem = passive_problem(&[Some(1.0), Some(3.0)]);
        let variance_objective = TreatmentObjectiveConfig {
            kind: TreatmentObjectiveKind::SeatVariance,
            ..TreatmentObjectiveConfig::default()
        };
        let config = MultiFrequencyRoomTreatmentConfig {
            max_iterations: 10,
            treatment_fraction: 0.5,
            initial_treatment: 0.2,
            objective: variance_objective.clone(),
            max_frequency_lanes: Some(2),
            ..MultiFrequencyRoomTreatmentConfig::default()
        };
        let result = problem.optimize(&config).unwrap();
        let weighted_mean = (result.treatments[0] * 1.0 + result.treatments[1] * 3.0) / 4.0;
        assert!(
            weighted_mean <= 0.5 + 1.0e-8,
            "area-weighted fraction {weighted_mean} exceeds the bound"
        );
        assert!(
            result
                .treatments
                .iter()
                .all(|value| (0.0..=1.0).contains(value))
        );
        assert_eq!(result.breakdown_history.len(), result.iterations);

        // Equal-weight fallback when areas are absent.
        let unweighted = passive_problem(&[None, None]);
        let result = unweighted.optimize(&config).unwrap();
        let mean = result.treatments.iter().sum::<f64>() / 2.0;
        assert!(
            mean <= 0.5 + 1.0e-8,
            "mean fraction {mean} exceeds the bound"
        );
    }

    #[test]
    fn multi_frequency_optimization_returns_filtered_design() {
        let problem = passive_problem(&[Some(1.0), Some(1.0)]);
        let filter = TreatmentDensityFilter::new(vec![[0.0, 0.0, 0.0]; 2], 1.0).unwrap();
        let result = problem
            .optimize(&MultiFrequencyRoomTreatmentConfig {
                max_iterations: 8,
                max_frequency_lanes: Some(1),
                density_filter: Some(filter),
                ..MultiFrequencyRoomTreatmentConfig::default()
            })
            .unwrap();
        assert!((result.treatments[0] - result.treatments[1]).abs() <= 1.0e-12);
    }

    #[test]
    fn multi_frequency_optimization_reduces_production_objective() {
        let problem = passive_problem(&[None, None]);
        let objective = production_objective();
        let config = MultiFrequencyRoomTreatmentConfig {
            max_iterations: 10,
            treatment_fraction: 0.6,
            initial_treatment: 0.2,
            objective: objective.clone(),
            max_frequency_lanes: Some(2),
            ..MultiFrequencyRoomTreatmentConfig::default()
        };
        let initial = problem
            .evaluate(&[0.2, 0.2], &objective, Some(2))
            .unwrap()
            .objective;
        let result = problem.optimize(&config).unwrap();
        let final_objective = problem
            .evaluate(&result.treatments, &objective, Some(2))
            .unwrap()
            .objective;
        assert!(
            final_objective < initial,
            "final {final_objective} should be below initial {initial}"
        );
    }

    #[test]
    fn decay_prediction_is_finite_and_decreases_with_absorption() {
        let decay =
            TreatmentDecayConstraint::new(100.0, 150.0, 7.5, vec![2.0, 4.0], 0.2, None, None)
                .unwrap();
        let (untreated, _) = decay.predict_rt60(&[0.0, 0.0]).unwrap();
        let (treated, _) = decay.predict_rt60(&[1.0, 1.0]).unwrap();
        assert!(untreated.is_finite() && untreated > 0.0);
        assert!(treated.is_finite() && treated > 0.0);
        assert!(
            treated < untreated,
            "treated={treated}, untreated={untreated}"
        );
    }

    #[test]
    fn decay_gradient_matches_finite_difference() {
        let decay =
            TreatmentDecayConstraint::new(120.0, 180.0, 9.0, vec![1.5, 3.0], 0.4, None, None)
                .unwrap();
        let treatments = [0.25, 0.65];
        let (_, gradient) = decay.predict_rt60(&treatments).unwrap();
        let step = 1.0e-6;
        for parameter in 0..treatments.len() {
            let mut plus = treatments;
            let mut minus = treatments;
            plus[parameter] += step;
            minus[parameter] -= step;
            let plus_rt60 = decay.predict_rt60(&plus).unwrap().0;
            let minus_rt60 = decay.predict_rt60(&minus).unwrap().0;
            let finite_difference = (plus_rt60 - minus_rt60) / (2.0 * step);
            let scale = finite_difference.abs().max(1.0);
            assert!(
                (gradient[parameter] - finite_difference).abs() / scale < 1.0e-8,
                "parameter {parameter}: analytic={}, FD={finite_difference}",
                gradient[parameter]
            );
        }
    }

    #[test]
    fn decay_constraint_rejects_malformed_physical_domains() {
        assert!(matches!(
            TreatmentDecayConstraint::new(
                f64::NAN,
                150.0,
                5.0,
                vec![1.0],
                0.0,
                None,
                None,
            ),
            Err(RoomTreatmentError::InvalidProblem(message)) if message.contains("volume")
        ));
        assert!(matches!(
            TreatmentDecayConstraint::new(
                100.0,
                150.0,
                5.0,
                vec![-1.0],
                0.0,
                None,
                None,
            ),
            Err(RoomTreatmentError::InvalidProblem(message)) if message.contains("deltas")
        ));
        assert!(matches!(
            TreatmentDecayConstraint::new(
                100.0,
                150.0,
                5.0,
                vec![f64::NAN],
                0.0,
                None,
                None,
            ),
            Err(RoomTreatmentError::InvalidProblem(message)) if message.contains("deltas")
        ));
        assert!(matches!(
            TreatmentDecayConstraint::new(
                100.0,
                150.0,
                140.0,
                vec![10.0],
                0.0,
                None,
                None,
            ),
            Err(RoomTreatmentError::InvalidProblem(message)) if message.contains("mean absorption")
        ));
        assert!(matches!(
            TreatmentDecayConstraint::new(
                100.0,
                150.0,
                0.0,
                vec![0.0],
                0.0,
                None,
                None,
            ),
            Err(RoomTreatmentError::InvalidProblem(message)) if message.contains("finite positive")
        ));

        let valid =
            TreatmentDecayConstraint::new(100.0, 150.0, 5.0, vec![1.0], 0.0, Some(2.0), Some(1.0));
        assert!(matches!(
            valid,
            Err(RoomTreatmentError::InvalidProblem(message))
                if message.contains("cannot exceed")
        ));
    }

    #[test]
    fn decay_mma_constraints_have_correct_signs_and_scaling() {
        let decay = TreatmentDecayConstraint::new(
            100.0,
            150.0,
            5.0,
            vec![2.0, 4.0],
            0.0,
            Some(0.5),
            Some(2.0),
        )
        .unwrap();
        let treatments = [0.2, 0.3];
        let (rt60, gradient) = decay.predict_rt60(&treatments).unwrap();
        let constraints = decay.linearized_mma_constraints(&treatments).unwrap();
        assert_eq!(constraints.len(), 2);
        let scale = 2.0;
        assert_eq!(constraints[0].bound, 0.0);
        assert!((constraints[0].value - (0.5 - rt60) / scale).abs() < 1.0e-14);
        assert_eq!(constraints[1].bound, 1.0);
        assert!((constraints[1].value - rt60 / scale).abs() < 1.0e-14);
        for ((lower, upper), &rt60_gradient) in constraints[0]
            .grad
            .iter()
            .zip(&constraints[1].grad)
            .zip(&gradient)
        {
            assert!((lower + rt60_gradient / scale).abs() < 1.0e-14);
            assert!((upper - rt60_gradient / scale).abs() < 1.0e-14);
        }
    }

    #[test]
    fn decay_bounds_are_checked_after_optimization_and_history_aligns() {
        let problem = passive_problem(&[None, None]);
        let decay = TreatmentDecayConstraint::new(
            100.0,
            150.0,
            5.0,
            vec![5.0, 10.0],
            0.0,
            Some(0.7),
            Some(3.0),
        )
        .unwrap();
        let config = MultiFrequencyRoomTreatmentConfig {
            max_iterations: 8,
            tolerance: 1.0e-3,
            treatment_fraction: 0.6,
            initial_treatment: 0.2,
            objective: TreatmentObjectiveConfig {
                kind: TreatmentObjectiveKind::SeatVariance,
                ..TreatmentObjectiveConfig::default()
            },
            max_frequency_lanes: Some(2),
            decay_constraint: Some(decay.clone()),
            spatial_constraint: None,
            density_filter: None,
        };
        let result = problem.optimize(&config).unwrap();
        let final_rt60 = decay.predict_rt60(&result.treatments).unwrap().0;
        decay.validate_rt60_bounds(final_rt60).unwrap();
        assert_eq!(result.rt60_history.len(), result.objective_history.len());
        assert_eq!(result.rt60_history.len(), result.breakdown_history.len());
        assert!(result.rt60_history.iter().all(|value| value.is_finite()));
        assert!(
            result
                .treatments
                .iter()
                .all(|value| (0.0..=1.0).contains(value))
        );
    }

    #[test]
    fn spatial_constraint_builds_deterministic_position_adjacency() {
        let constraint = TreatmentSpatialConstraint::from_positions(
            &[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0]],
            0.75,
            0.2,
        )
        .unwrap();
        assert_eq!(constraint.adjacent_pairs, vec![(0, 1)]);
        assert_eq!(constraint.maximum_adjacent_difference, 0.2);

        assert!(matches!(
            TreatmentSpatialConstraint::from_positions(
                &[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                0.5,
                0.2,
            ),
            Err(RoomTreatmentError::InvalidProblem(message))
                if message.contains("no adjacent pairs")
        ));
        assert!(matches!(
            TreatmentSpatialConstraint::from_positions(
                &[[f64::NAN, 0.0, 0.0], [0.5, 0.0, 0.0]],
                0.75,
                0.2,
            ),
            Err(RoomTreatmentError::InvalidProblem(message)) if message.contains("positions")
        ));
    }

    #[test]
    fn spatial_constraint_linearization_has_exact_pairwise_signs() {
        let constraint = TreatmentSpatialConstraint::new(vec![(0, 1)], 0.2, 2).unwrap();
        let treatments = [0.7, 0.4];
        let constraints = constraint.linearized_mma_constraints(&treatments).unwrap();
        assert_eq!(constraints.len(), 2);
        assert_eq!(constraints[0].grad, vec![1.0, -1.0]);
        assert_eq!(constraints[1].grad, vec![-1.0, 1.0]);
        assert!((constraints[0].value - 0.3).abs() < 1.0e-14);
        assert!((constraints[1].value + 0.3).abs() < 1.0e-14);
        assert_eq!(constraints[0].bound, 0.2);
        assert_eq!(constraints[1].bound, 0.2);
    }

    #[test]
    fn spatial_constraint_rejects_invalid_graphs_and_duplicates() {
        assert!(matches!(
            TreatmentSpatialConstraint::new(vec![], 0.2, 2),
            Err(RoomTreatmentError::InvalidProblem(message))
                if message.contains("requires adjacent pairs")
        ));
        assert!(matches!(
            TreatmentSpatialConstraint::new(vec![(0, 2)], 0.2, 2),
            Err(RoomTreatmentError::InvalidProblem(message)) if message.contains("pair index")
        ));
        assert!(matches!(
            TreatmentSpatialConstraint::new(vec![(0, 1), (1, 0)], 0.2, 2),
            Err(RoomTreatmentError::InvalidProblem(message)) if message.contains("duplicate")
        ));
        assert!(matches!(
            TreatmentSpatialConstraint::new(vec![(0, 1)], f64::NAN, 2),
            Err(RoomTreatmentError::InvalidProblem(message)) if message.contains("maximum")
        ));
        let constraint = TreatmentSpatialConstraint::new(vec![(0, 1)], 0.2, 2).unwrap();
        assert!(matches!(
            constraint.linearized_mma_constraints(&[0.1]),
            Err(RoomTreatmentError::InvalidProblem(message)) if message.contains("pair")
        ));
    }

    #[test]
    fn spatial_constraint_is_respected_by_multi_frequency_optimization() {
        let problem = passive_problem(&[None, None]);
        let spatial = TreatmentSpatialConstraint::from_positions(
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            1.1,
            0.05,
        )
        .unwrap();
        let config = MultiFrequencyRoomTreatmentConfig {
            max_iterations: 10,
            tolerance: 1.0e-3,
            treatment_fraction: 0.6,
            initial_treatment: 0.2,
            objective: TreatmentObjectiveConfig {
                kind: TreatmentObjectiveKind::SeatVariance,
                ..TreatmentObjectiveConfig::default()
            },
            max_frequency_lanes: Some(2),
            spatial_constraint: Some(spatial.clone()),
            ..MultiFrequencyRoomTreatmentConfig::default()
        };
        let result = problem.optimize(&config).unwrap();
        spatial.validate_design(&result.treatments).unwrap();
        assert!((result.treatments[0] - result.treatments[1]).abs() <= 0.05 + 1.0e-8);
    }

    #[test]
    fn spatial_constraint_final_check_is_fail_closed() {
        let constraint = TreatmentSpatialConstraint::new(vec![(0, 1)], 0.1, 2).unwrap();
        let error = constraint.validate_design(&[0.0, 0.2]).unwrap_err();
        assert!(matches!(
            error,
            RoomTreatmentError::ConstraintViolation(message) if message.contains("pair")
        ));
    }

    #[test]
    fn final_decay_bound_failure_is_fail_closed() {
        let decay =
            TreatmentDecayConstraint::new(100.0, 150.0, 5.0, vec![1.0], 0.0, Some(4.0), None)
                .unwrap();
        assert!(matches!(
            decay.validate_rt60_bounds(decay.predict_rt60(&[0.0]).unwrap().0),
            Err(RoomTreatmentError::ConstraintViolation(message))
                if message.contains("below minimum")
        ));
    }
}
