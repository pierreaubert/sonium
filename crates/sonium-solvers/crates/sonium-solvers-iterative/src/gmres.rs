//! GMRES (Generalized Minimal Residual) solver
//!
//! Implementation of the restarted GMRES algorithm based on Saad & Schultz (1986).
//!
//! GMRES is often the best choice for large non-symmetric systems.
//! It minimizes the residual in a Krylov subspace and has smooth, monotonic
//! convergence behavior.

use super::cgs_orth::{
    FlatBasis, KrylovOpCounts, batched_dots_flat_into, batched_project_flat,
    fused_basis_update_flat,
};
use super::gmres_common::{
    exact_relative_residual, exact_residual_check_due, givens_rotation, solve_upper_triangular,
};
use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use sonium_solvers_core::blas_helpers::vector_norm;
use sonium_solvers_core::traits::{
    AccuracyStatus, ComplexField, IdentityPreconditioner, LinearOperator, Preconditioner,
    SolverError, SolverResultContract, SolverStatus,
};
use std::panic::{AssertUnwindSafe, catch_unwind};

/// GMRES solver configuration
#[derive(Debug, Clone)]
pub struct GmresConfig<R> {
    /// Maximum number of outer iterations (restarts)
    pub max_iterations: usize,
    /// Restart parameter (number of inner iterations before restart)
    pub restart: usize,
    /// Relative tolerance for convergence
    pub tolerance: R,
    /// Print progress every N iterations (0 = no output)
    pub print_interval: usize,
    /// Check convergence every N inner iterations (default 1 = every iteration)
    pub convergence_check_interval: usize,
    /// Verify estimated convergence with an exact residual every N Krylov iterations.
    ///
    /// The exact residual check costs an additional operator application. A
    /// value of 0 preserves the cheap estimated-residual path except for
    /// mandatory safety checks at startup, breakdown, and final exit.
    pub exact_residual_check_interval: usize,
    /// Minimum restart parameter for adaptive restart (None = fixed restart)
    pub min_restart: Option<usize>,
    /// Maximum restart parameter for adaptive restart (None = fixed restart)
    pub max_restart: Option<usize>,
}

impl Default for GmresConfig<f64> {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            restart: 30,
            tolerance: 1e-6,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
            min_restart: None,
            max_restart: None,
        }
    }
}

impl Default for GmresConfig<f32> {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            restart: 30,
            tolerance: 1e-5,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
            min_restart: None,
            max_restart: None,
        }
    }
}

impl<R: Float + FromPrimitive> GmresConfig<R> {
    /// Create config for small problems
    pub fn for_small_problems() -> Self {
        Self {
            max_iterations: 50,
            restart: 50,
            tolerance: R::from_f64(1e-8).unwrap(),
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
            min_restart: None,
            max_restart: None,
        }
    }

    /// Create config with specific restart parameter
    pub fn with_restart(restart: usize) -> Self
    where
        Self: Default,
    {
        Self {
            restart,
            ..Default::default()
        }
    }
}

/// GMRES solver result
#[derive(Debug)]
pub struct GmresSolution<T: ComplexField> {
    /// Solution vector
    pub x: Vec<T>,
    /// Total number of matrix-vector products
    pub iterations: usize,
    /// Number of restarts performed
    pub restarts: usize,
    /// Final relative residual
    pub residual: T::Real,
    /// Cheap projected/Hessenberg residual captured by the iteration.
    /// This is kept separate from the independently recomputed public residual.
    pub estimated_residual: T::Real,
    /// Normwise backward error: `||b - A*x|| / max(||A*x|| + ||b||, 1)`.
    pub backward_error: T::Real,
    /// Whether the requested tolerance was attained by the returned iterate.
    pub attainable_accuracy: AccuracyStatus,
    /// Whether convergence was achieved
    pub converged: bool,
    /// Solver status
    pub status: SolverStatus,
    /// Memory-traffic oriented operation counts for the Krylov solve.
    pub op_counts: KrylovOpCounts,
}

/// Return the common invalid-input result for GMRES-family entry points.
///
/// Several specialized variants (GMRES-DR, GCRO-DR, and recycled GMRES)
/// return [`GmresSolution`] directly, so they cannot surface a construction
/// error.  They must nevertheless reject malformed inputs before invoking an
/// operator or preconditioner, just like standard GMRES does.
pub(crate) fn numerical_error_solution<T: ComplexField>(x: Vec<T>) -> GmresSolution<T> {
    GmresSolution {
        x,
        iterations: 0,
        restarts: 0,
        residual: T::Real::infinity(),
        estimated_residual: T::Real::infinity(),
        backward_error: T::Real::infinity(),
        attainable_accuracy: AccuracyStatus::NotChecked,
        converged: false,
        status: SolverStatus::NumericalError,
        op_counts: KrylovOpCounts::default(),
    }
}

/// Recompute the public accuracy contract from the original operator.
///
/// Krylov variants are allowed to use a projected or preconditioned estimate
/// for their inner loop, but every normal return is finalized against the
/// original `A*x=b` system. The estimate is intentionally preserved in the
/// result for diagnostics and continuation decisions.
pub(crate) fn finalize_gmres_solution<T, A>(
    operator: &A,
    b: &[T],
    tolerance: T::Real,
    mut result: GmresSolution<T>,
) -> GmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    if result.status == SolverStatus::NumericalError {
        return result;
    }
    // The projected residual is an internal estimate, but it is part of the
    // public diagnostic contract. Do not publish a non-finite or negative
    // estimate alongside a finite true residual after an unstable recurrence.
    if !result.estimated_residual.is_finite() || result.estimated_residual < T::Real::zero() {
        result.status = SolverStatus::NumericalError;
        result.converged = false;
        result.residual = T::Real::infinity();
        result.backward_error = T::Real::infinity();
        result.attainable_accuracy = AccuracyStatus::NotChecked;
        return result;
    }
    if result.x.len() != operator.num_cols()
        || b.len() != operator.num_rows()
        || result.x.iter().any(|value| !value.norm().is_finite())
        || b.iter().any(|value| !value.norm().is_finite())
    {
        result.status = SolverStatus::NumericalError;
        result.converged = false;
        result.residual = T::Real::infinity();
        result.backward_error = T::Real::infinity();
        result.attainable_accuracy = AccuracyStatus::NotChecked;
        return result;
    }

    let mut ax = vec![T::zero(); operator.num_rows()];
    operator.apply_into(&result.x, &mut ax);
    result.op_counts.spmv_count += 1;
    let mut residual_norm_sq = T::Real::zero();
    let mut ax_norm_sq = T::Real::zero();
    let mut b_norm_sq = T::Real::zero();
    for ((&rhs, &applied), &value) in b.iter().zip(&ax).zip(&result.x) {
        let residual = rhs - applied;
        residual_norm_sq += residual.norm_sqr();
        ax_norm_sq += applied.norm_sqr();
        b_norm_sq += rhs.norm_sqr();
        if !value.norm().is_finite() || !applied.norm().is_finite() {
            result.attainable_accuracy = AccuracyStatus::NotChecked;
            result.converged = false;
            result.status = SolverStatus::NumericalError;
            result.residual = T::Real::infinity();
            result.backward_error = T::Real::infinity();
            return result;
        }
    }

    if !residual_norm_sq.is_finite() || !ax_norm_sq.is_finite() || !b_norm_sq.is_finite() {
        result.attainable_accuracy = AccuracyStatus::NotChecked;
        result.converged = false;
        result.status = SolverStatus::NumericalError;
        result.residual = T::Real::infinity();
        result.backward_error = T::Real::infinity();
        return result;
    }

    let residual_norm = residual_norm_sq.sqrt();
    result.op_counts.norm_count += 3;
    let b_norm = b_norm_sq.sqrt();
    let true_residual = if b_norm > T::Real::zero() {
        residual_norm / b_norm
    } else {
        residual_norm
    };
    let denominator = (ax_norm_sq.sqrt() + b_norm).max(T::Real::min_positive_value());
    let backward_error = residual_norm / denominator;
    result.residual = true_residual;
    result.backward_error = backward_error;
    result.attainable_accuracy = if true_residual <= tolerance {
        AccuracyStatus::Attained
    } else {
        AccuracyStatus::NotAttained
    };
    result.converged = true_residual <= tolerance;
    if result.converged {
        result.status = SolverStatus::Converged;
    } else if result.status == SolverStatus::Converged {
        result.status = SolverStatus::MaxIterationsReached;
    }
    result
}

pub(crate) fn invalid_gmres_input<T: ComplexField>(
    b: &[T],
    x0: Option<&[T]>,
    max_iterations: usize,
    restart: usize,
    tolerance: T::Real,
) -> Option<GmresSolution<T>> {
    let n = b.len();
    let invalid = max_iterations == 0
        || restart == 0
        || !tolerance.is_finite()
        || tolerance <= T::Real::zero()
        || b.iter().any(|value| !value.norm().is_finite())
        || x0.is_some_and(|guess| {
            guess.len() != n || guess.iter().any(|value| !value.norm().is_finite())
        });

    invalid.then(|| {
        numerical_error_solution(match x0 {
            Some(guess) if guess.len() == n => guess.to_vec(),
            _ => vec![T::zero(); n],
        })
    })
}

/// Solve Ax = b using the restarted GMRES method
///
/// # Arguments
/// * `operator` - Linear operator representing A
/// * `b` - Right-hand side vector
/// * `config` - Solver configuration
///
/// # Returns
/// Solution struct containing x, iteration count, and convergence info
pub fn gmres<T, A>(operator: &A, b: &[T], config: &GmresConfig<T::Real>) -> GmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    finalize_gmres_solution(
        operator,
        b,
        config.tolerance,
        gmres_core(operator, &IdentityPreconditioner, b, None, config),
    )
}

/// Checked GMRES entry point for callers with user-controlled dimensions or
/// numerical data.
///
/// The compatibility [`gmres`] function retains its infallible result-based
/// contract. This variant validates the square operator, RHS, configuration,
/// and adaptive-restart bounds before entering the Krylov loop, contains
/// panics from legacy operator implementations, and promotes a generated
/// `NumericalError` to a typed failure.
pub fn try_gmres<T, A>(
    operator: &A,
    b: &[T],
    config: &GmresConfig<T::Real>,
) -> Result<GmresSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    if operator.num_rows() != operator.num_cols() {
        return Err(SolverError::DimensionMismatch {
            expected: operator.num_cols(),
            got: operator.num_rows(),
        });
    }
    if b.len() != operator.num_cols() {
        return Err(SolverError::DimensionMismatch {
            expected: operator.num_cols(),
            got: b.len(),
        });
    }
    if b.iter().any(|value| !value.norm().is_finite()) {
        return Err(SolverError::OperationFailed {
            operation: "GMRES",
            reason: "right-hand side contains NaN or infinity".to_string(),
        });
    }
    if config.max_iterations == 0 {
        return Err(SolverError::OperationFailed {
            operation: "GMRES configuration",
            reason: "max_iterations must be greater than zero".to_string(),
        });
    }
    if config.restart == 0 {
        return Err(SolverError::OperationFailed {
            operation: "GMRES configuration",
            reason: "restart must be greater than zero".to_string(),
        });
    }
    if !config.tolerance.is_finite() || config.tolerance <= T::Real::zero() {
        return Err(SolverError::OperationFailed {
            operation: "GMRES configuration",
            reason: "tolerance must be finite and positive".to_string(),
        });
    }
    if config.min_restart == Some(0)
        || config.max_restart == Some(0)
        || matches!((config.min_restart, config.max_restart), (Some(min), Some(max)) if min > max)
    {
        return Err(SolverError::OperationFailed {
            operation: "GMRES configuration",
            reason: "adaptive restart bounds must be positive and ordered".to_string(),
        });
    }

    let result =
        catch_unwind(AssertUnwindSafe(|| gmres(operator, b, config))).map_err(|payload| {
            SolverError::OperationFailed {
                operation: "GMRES",
                reason: panic_reason(payload),
            }
        })?;
    if result.status == SolverStatus::NumericalError {
        return Err(SolverError::OperationFailed {
            operation: "GMRES",
            reason: "solver produced a non-finite value or residual".to_string(),
        });
    }
    SolverResultContract::validate_contract(&result)?;
    Ok(result)
}

/// Checked left-preconditioned GMRES entry point.
///
/// This mirrors [`try_gmres`] while also containing panics raised by a
/// user-provided preconditioner.  The legacy [`gmres_preconditioned`] API
/// remains available for compatibility, but callers that accept external
/// operators or preconditioners should use this boundary to receive typed
/// failures before a numerical-error result is published.
pub fn try_gmres_preconditioned<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    config: &GmresConfig<T::Real>,
) -> Result<GmresSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    if operator.num_rows() != operator.num_cols() {
        return Err(SolverError::DimensionMismatch {
            expected: operator.num_cols(),
            got: operator.num_rows(),
        });
    }
    if b.len() != operator.num_cols() {
        return Err(SolverError::DimensionMismatch {
            expected: operator.num_cols(),
            got: b.len(),
        });
    }
    if b.iter().any(|value| !value.norm().is_finite()) {
        return Err(SolverError::OperationFailed {
            operation: "GMRES preconditioned",
            reason: "right-hand side contains NaN or infinity".to_string(),
        });
    }
    if config.max_iterations == 0 {
        return Err(SolverError::OperationFailed {
            operation: "GMRES configuration",
            reason: "max_iterations must be greater than zero".to_string(),
        });
    }
    if config.restart == 0 {
        return Err(SolverError::OperationFailed {
            operation: "GMRES configuration",
            reason: "restart must be greater than zero".to_string(),
        });
    }
    if !config.tolerance.is_finite() || config.tolerance <= T::Real::zero() {
        return Err(SolverError::OperationFailed {
            operation: "GMRES configuration",
            reason: "tolerance must be finite and positive".to_string(),
        });
    }
    if config.min_restart == Some(0)
        || config.max_restart == Some(0)
        || matches!(
            (config.min_restart, config.max_restart),
            (Some(min), Some(max)) if min > max
        )
    {
        return Err(SolverError::OperationFailed {
            operation: "GMRES configuration",
            reason: "adaptive restart bounds must be positive and ordered".to_string(),
        });
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        gmres_preconditioned(operator, precond, b, config)
    }))
    .map_err(|payload| SolverError::OperationFailed {
        operation: "GMRES preconditioned",
        reason: panic_reason(payload),
    })?;
    if result.status == SolverStatus::NumericalError {
        return Err(SolverError::OperationFailed {
            operation: "GMRES preconditioned",
            reason: "solver produced a non-finite value or residual".to_string(),
        });
    }
    SolverResultContract::validate_contract(&result)?;
    Ok(result)
}

/// Checked left-preconditioned GMRES with a caller-provided initial iterate.
///
/// The initial guess is validated before any operator or preconditioner action
/// and panics from either callback are converted to [`SolverError`].
pub fn try_gmres_preconditioned_with_guess<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x0: Option<&[T]>,
    config: &GmresConfig<T::Real>,
) -> Result<GmresSolution<T>, SolverError>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    if operator.num_rows() != operator.num_cols() {
        return Err(SolverError::DimensionMismatch {
            expected: operator.num_cols(),
            got: operator.num_rows(),
        });
    }
    if b.len() != operator.num_cols() {
        return Err(SolverError::DimensionMismatch {
            expected: operator.num_cols(),
            got: b.len(),
        });
    }
    if b.iter().any(|value| !value.norm().is_finite()) {
        return Err(SolverError::OperationFailed {
            operation: "GMRES preconditioned",
            reason: "right-hand side contains NaN or infinity".to_string(),
        });
    }
    if let Some(guess) = x0 {
        if guess.len() != operator.num_cols() {
            return Err(SolverError::DimensionMismatch {
                expected: operator.num_cols(),
                got: guess.len(),
            });
        }
        if guess.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation: "GMRES preconditioned",
                reason: "initial guess contains NaN or infinity".to_string(),
            });
        }
    }
    if config.max_iterations == 0 || config.restart == 0 {
        return Err(SolverError::OperationFailed {
            operation: "GMRES configuration",
            reason: "max_iterations and restart must be greater than zero".to_string(),
        });
    }
    if !config.tolerance.is_finite() || config.tolerance <= T::Real::zero() {
        return Err(SolverError::OperationFailed {
            operation: "GMRES configuration",
            reason: "tolerance must be finite and positive".to_string(),
        });
    }
    if config.min_restart == Some(0)
        || config.max_restart == Some(0)
        || matches!(
            (config.min_restart, config.max_restart),
            (Some(min), Some(max)) if min > max
        )
    {
        return Err(SolverError::OperationFailed {
            operation: "GMRES configuration",
            reason: "adaptive restart bounds must be positive and ordered".to_string(),
        });
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        gmres_preconditioned_with_guess(operator, precond, b, x0, config)
    }))
    .map_err(|payload| SolverError::OperationFailed {
        operation: "GMRES preconditioned",
        reason: panic_reason(payload),
    })?;
    if result.status == SolverStatus::NumericalError {
        return Err(SolverError::OperationFailed {
            operation: "GMRES preconditioned",
            reason: "solver produced a non-finite value or residual".to_string(),
        });
    }
    SolverResultContract::validate_contract(&result)?;
    Ok(result)
}

fn panic_reason(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "GMRES operator panicked".to_string()
    }
}

/// Solve Ax = b using GMRES with an initial guess
pub fn gmres_with_guess<T, A>(
    operator: &A,
    b: &[T],
    x0: Option<&[T]>,
    config: &GmresConfig<T::Real>,
) -> GmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
{
    finalize_gmres_solution(
        operator,
        b,
        config.tolerance,
        gmres_core(operator, &IdentityPreconditioner, b, x0, config),
    )
}

/// GMRES solver with preconditioner
///
/// Solves Ax = b using left preconditioning: M⁻¹Ax = M⁻¹b
pub fn gmres_preconditioned<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    config: &GmresConfig<T::Real>,
) -> GmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    finalize_gmres_solution(
        operator,
        b,
        config.tolerance,
        gmres_core(operator, precond, b, None, config),
    )
}

/// GMRES solver with preconditioner and initial guess
///
/// Solves Ax = b using left preconditioning: M⁻¹Ax = M⁻¹b
/// with an optional initial guess x0.
pub fn gmres_preconditioned_with_guess<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x0: Option<&[T]>,
    config: &GmresConfig<T::Real>,
) -> GmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    finalize_gmres_solution(
        operator,
        b,
        config.tolerance,
        gmres_core(operator, precond, b, x0, config),
    )
}

/// Unified GMRES core: handles unpreconditioned and preconditioned solves.
///
/// For unpreconditioned use, pass [`IdentityPreconditioner`].
fn gmres_core<T, A, P>(
    operator: &A,
    precond: &P,
    b: &[T],
    x0: Option<&[T]>,
    config: &GmresConfig<T::Real>,
) -> GmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    let n = b.len();
    if let Some(invalid) = invalid_gmres_input(
        b,
        x0,
        config.max_iterations,
        config.restart,
        config.tolerance,
    ) {
        return invalid;
    }
    let mut m = config.restart;

    let mut x = match x0 {
        Some(x0) => x0.to_vec(),
        None => vec![T::zero(); n],
    };

    let mut op_buf = vec![T::zero(); n];
    let mut w = vec![T::zero(); n];
    let mut op_counts = KrylovOpCounts::default();
    let tol_threshold = T::Real::from_f64(1e-15).unwrap();

    // Public convergence is measured against the unpreconditioned Ax=b
    // residual. A zero RHS needs an absolute residual check because a relative
    // residual is undefined.
    let raw_true_b_norm = vector_norm(b);
    op_counts.norm_count += 1;
    let rhs_is_near_zero = raw_true_b_norm < tol_threshold;
    let true_b_norm = if rhs_is_near_zero {
        T::Real::one()
    } else {
        raw_true_b_norm
    };

    // Compute preconditioned RHS norm for relative tolerance
    precond.apply_into(b, &mut w);
    op_counts.preconditioner_count += 1;
    let raw_preconditioned_b_norm = vector_norm(&w);
    op_counts.norm_count += 1;
    if !rhs_is_near_zero && raw_preconditioned_b_norm < tol_threshold {
        // Preconditioner destroyed the RHS — cannot proceed meaningfully.
        return GmresSolution {
            x,
            iterations: 0,
            restarts: 0,
            residual: T::Real::one(),
            estimated_residual: T::Real::one(),
            backward_error: T::Real::one(),
            attainable_accuracy: AccuracyStatus::NotAttained,
            converged: false,
            status: SolverStatus::Breakdown,
            op_counts,
        };
    }
    let preconditioned_b_norm = if rhs_is_near_zero {
        T::Real::one()
    } else {
        raw_preconditioned_b_norm
    };

    let mut total_iterations = 0;
    let mut restarts = 0;

    // Allocate for the current restart only; the adaptive-growth arm below
    // resizes every structure on demand. Pre-allocating max_restart upfront
    // would multiply memory by max/restart for every adaptive solve even
    // when the first cycle already converges.
    let alloc_m = m;
    let mut v = FlatBasis::new(alloc_m + 1, n);

    let mut h: sonium_solvers_core::dense::DenseMatrix<T> =
        sonium_solvers_core::dense::DenseMatrix::from_elem(alloc_m + 1, alloc_m, T::zero());
    let mut cs: Vec<T> = Vec::with_capacity(alloc_m);
    let mut sn: Vec<T> = Vec::with_capacity(alloc_m);
    let mut g: Vec<T> = vec![T::zero(); alloc_m + 1];
    let mut coeffs = Vec::with_capacity(alloc_m);
    let mut reorth_coeffs = Vec::with_capacity(alloc_m);

    'outer: for _outer in 0..config.max_iterations {
        // Compute preconditioned residual r = M⁻¹(b - Ax) into w
        operator.apply_into(&x, &mut op_buf);
        op_counts.spmv_count += 1;
        for k in 0..n {
            op_buf[k] = b[k] - op_buf[k];
        }
        let true_rel_residual = vector_norm(&op_buf) / true_b_norm;
        precond.apply_into(&op_buf, &mut w);
        op_counts.preconditioner_count += 1;
        let beta = vector_norm(&w);
        op_counts.norm_count += 1;

        let rel_residual = beta / preconditioned_b_norm;
        if rel_residual < config.tolerance && true_rel_residual < config.tolerance {
            return GmresSolution {
                x,
                iterations: total_iterations,
                restarts,
                residual: true_rel_residual,
                estimated_residual: rel_residual,
                backward_error: true_rel_residual,
                attainable_accuracy: AccuracyStatus::Attained,
                converged: true,
                status: SolverStatus::Converged,
                op_counts,
            };
        }

        // A preconditioner may scale a nonzero residual below the Krylov
        // tolerance without actually solving the original system. Continue
        // with that residual instead of reporting a false breakdown. A zero
        // preconditioned residual cannot seed a Krylov basis, so it remains a
        // genuine breakdown.
        if !beta.is_finite() || beta == T::Real::zero() {
            return GmresSolution {
                x,
                iterations: total_iterations,
                restarts,
                residual: true_rel_residual,
                estimated_residual: rel_residual,
                backward_error: true_rel_residual,
                attainable_accuracy: AccuracyStatus::NotAttained,
                converged: false,
                status: SolverStatus::Breakdown,
                op_counts,
            };
        }

        // Track residual at start of cycle for adaptive restart
        let rel_residual_before_cycle = rel_residual;

        // Initialize v[0] = w / beta
        let inv_beta = T::from_real(T::Real::one() / beta);
        {
            let v0 = v.vector_mut(0);
            for k in 0..n {
                v0[k] = w[k] * inv_beta;
            }
        }

        h.fill(T::zero());
        cs.clear();
        sn.clear();
        g.fill(T::zero());
        g[0] = T::from_real(beta);

        let mut inner_converged = false;
        let dgks_threshold = T::Real::from_f64(0.707).unwrap();
        let breakdown_tol = T::Real::from_f64(1e-20).unwrap();
        let abs_tol = T::Real::from_f64(1e-20).unwrap();

        for j in 0..m {
            total_iterations += 1;

            // w = M⁻¹ * A * v_j (zero allocation via apply_into)
            operator.apply_into(v.vector(j), &mut op_buf);
            op_counts.spmv_count += 1;
            precond.apply_into(&op_buf, &mut w);
            op_counts.preconditioner_count += 1;

            // Classical Gram-Schmidt with DGKS reorthogonalisation, batched.
            //
            // Pass 1 collects every projection coefficient against the basis
            // in parallel (independent reductions over disjoint vectors).
            // Pass 2 fuses all `j+1` axpys into a single parallel walk over
            // rows of `w`, removing the false-serialisation that the per-`i`
            // axpy loop had.
            let w_norm_before = vector_norm(&w);
            op_counts.norm_count += 1;

            batched_dots_flat_into(&v, j + 1, &w, &mut coeffs);
            for (i, &c) in coeffs.iter().enumerate() {
                h[[i, j]] = c;
            }
            batched_project_flat(&v, &coeffs, &mut w);
            op_counts.add_basis_projection::<T>(j + 1, n);

            // DGKS criterion: reorthogonalise if ||w|| < 0.707 * ||w_before||
            let mut w_norm = vector_norm(&w);
            op_counts.norm_count += 1;
            if w_norm < dgks_threshold * w_norm_before {
                batched_dots_flat_into(&v, j + 1, &w, &mut reorth_coeffs);
                for (i, &si) in reorth_coeffs.iter().enumerate() {
                    h[[i, j]] += si;
                }
                batched_project_flat(&v, &reorth_coeffs, &mut w);
                op_counts.add_basis_projection::<T>(j + 1, n);
                w_norm = vector_norm(&w);
                op_counts.norm_count += 1;
            }
            h[[j + 1, j]] = T::from_real(w_norm);

            if w_norm < breakdown_tol {
                inner_converged = true;
            } else {
                let inv_norm = T::from_real(T::Real::one() / w_norm);
                let next_v = v.vector_mut(j + 1);
                for k in 0..n {
                    next_v[k] = w[k] * inv_norm;
                }
            }

            // Apply previous Givens rotations
            for i in 0..j {
                let temp = cs[i].conj() * h[[i, j]] + sn[i].conj() * h[[i + 1, j]];
                h[[i + 1, j]] = T::zero() - sn[i] * h[[i, j]] + cs[i] * h[[i + 1, j]];
                h[[i, j]] = temp;
            }

            let (c, s) = givens_rotation(h[[j, j]], h[[j + 1, j]]);
            cs.push(c);
            sn.push(s);

            h[[j, j]] = c.conj() * h[[j, j]] + s.conj() * h[[j + 1, j]];
            h[[j + 1, j]] = T::zero();

            let temp = c.conj() * g[j] + s.conj() * g[j + 1];
            g[j + 1] = T::zero() - s * g[j] + c * g[j + 1];
            g[j] = temp;

            let abs_residual = g[j + 1].norm();
            let rel_residual = abs_residual / preconditioned_b_norm;

            if config.print_interval > 0 && total_iterations % config.print_interval == 0 {
                log::info!(
                    "GMRES iteration {} (restart {}): relative residual = {:.6e}",
                    total_iterations,
                    restarts,
                    rel_residual.to_f64().unwrap_or(0.0)
                );
            }

            // Lazy convergence checking: only test at specified intervals,
            // the last inner iteration, or on breakdown.
            let should_check =
                j % config.convergence_check_interval.max(1) == 0 || j == m - 1 || inner_converged;
            let converged =
                rel_residual < config.tolerance || (!inner_converged && abs_residual < abs_tol);

            if should_check && converged {
                let y = solve_upper_triangular(&h, &g, j + 1);
                fused_basis_update_flat(&v, &y, &mut x);
                op_counts.add_basis_update::<T>(y.len(), n);

                if exact_residual_check_due(config.exact_residual_check_interval, total_iterations)
                {
                    let actual_residual = exact_relative_residual(
                        operator,
                        &x,
                        b,
                        &mut op_buf,
                        true_b_norm,
                        &mut op_counts,
                    );
                    if actual_residual >= config.tolerance && actual_residual >= abs_tol {
                        restarts += 1;
                        continue 'outer;
                    }

                    return GmresSolution {
                        x,
                        iterations: total_iterations,
                        restarts,
                        residual: actual_residual,
                        estimated_residual: rel_residual,
                        backward_error: actual_residual,
                        attainable_accuracy: AccuracyStatus::Attained,
                        converged: true,
                        status: SolverStatus::Converged,
                        op_counts,
                    };
                }

                let true_rel_residual = exact_relative_residual(
                    operator,
                    &x,
                    b,
                    &mut op_buf,
                    true_b_norm,
                    &mut op_counts,
                );
                if true_rel_residual >= config.tolerance && true_rel_residual >= abs_tol {
                    restarts += 1;
                    continue 'outer;
                }

                return GmresSolution {
                    x,
                    iterations: total_iterations,
                    restarts,
                    residual: true_rel_residual,
                    estimated_residual: rel_residual,
                    backward_error: true_rel_residual,
                    attainable_accuracy: AccuracyStatus::Attained,
                    converged: true,
                    status: SolverStatus::Converged,
                    op_counts,
                };
            }

            // Breakdown with unconverged residual — terminate early.
            if inner_converged {
                let y = solve_upper_triangular(&h, &g, j + 1);
                fused_basis_update_flat(&v, &y, &mut x);
                op_counts.add_basis_update::<T>(y.len(), n);

                // Compute the *actual* residual (not the preconditioned
                // estimate) to guard against pathological preconditioners.
                operator.apply_into(&x, &mut op_buf);
                op_counts.spmv_count += 1;
                let mut actual_residual = T::Real::zero();
                for k in 0..n {
                    let rk = b[k] - op_buf[k];
                    actual_residual += rk.norm_sqr();
                }
                actual_residual = actual_residual.sqrt() / true_b_norm;

                if actual_residual < config.tolerance {
                    return GmresSolution {
                        x,
                        iterations: total_iterations,
                        restarts,
                        residual: actual_residual,
                        estimated_residual: rel_residual,
                        backward_error: actual_residual,
                        attainable_accuracy: AccuracyStatus::Attained,
                        converged: true,
                        status: SolverStatus::Converged,
                        op_counts,
                    };
                }

                return GmresSolution {
                    x,
                    iterations: total_iterations,
                    restarts,
                    residual: actual_residual,
                    estimated_residual: rel_residual,
                    backward_error: actual_residual,
                    attainable_accuracy: AccuracyStatus::NotAttained,
                    converged: false,
                    status: SolverStatus::Breakdown,
                    op_counts,
                };
            }
        }

        // End of restart cycle: solve least-squares and update x
        let y = solve_upper_triangular(&h, &g, m);
        fused_basis_update_flat(&v, &y, &mut x);
        op_counts.add_basis_update::<T>(y.len(), n);

        // The Krylov estimate is a left-preconditioned norm and can be much
        // more pessimistic than the requested ||b-Ax||/||b|| tolerance. Check
        // the true residual at every restart boundary before spending another
        // whole cycle. This also keeps convergence reporting independent of
        // arbitrary preconditioner scaling.
        let true_rel_residual =
            exact_relative_residual(operator, &x, b, &mut op_buf, true_b_norm, &mut op_counts);
        if true_rel_residual < config.tolerance {
            return GmresSolution {
                x,
                iterations: total_iterations,
                restarts,
                residual: true_rel_residual,
                estimated_residual: true_rel_residual,
                backward_error: true_rel_residual,
                attainable_accuracy: AccuracyStatus::Attained,
                converged: true,
                status: SolverStatus::Converged,
                op_counts,
            };
        }

        // Adaptive restart: monitor convergence rate across restarts
        let residual_after_cycle = g[m].norm() / preconditioned_b_norm;
        if let (Some(min_m), Some(max_m)) = (config.min_restart, config.max_restart) {
            let ratio = residual_after_cycle / rel_residual_before_cycle;
            if ratio > T::Real::from_f64(0.9).unwrap() && m < max_m {
                // Stalling: double restart parameter
                m = (m * 2).min(max_m);
                v.resize_vectors(m + 1);
                h = sonium_solvers_core::dense::DenseMatrix::from_elem(m + 1, m, T::zero());
                cs = Vec::with_capacity(m);
                sn = Vec::with_capacity(m);
                g = vec![T::zero(); m + 1];
            } else if ratio < T::Real::from_f64(0.1).unwrap() && m > min_m {
                // Fast convergence: halve restart parameter
                m = (m / 2).max(min_m);
            }
        }

        restarts += 1;
    }

    // Final true residual
    operator.apply_into(&x, &mut op_buf);
    op_counts.spmv_count += 1;
    let mut true_residual = T::Real::zero();
    for k in 0..n {
        let rk = b[k] - op_buf[k];
        true_residual += rk.norm_sqr();
    }
    let rel_residual = true_residual.sqrt() / true_b_norm;
    op_counts.norm_count += 1;

    GmresSolution {
        x,
        iterations: total_iterations,
        restarts,
        residual: rel_residual,
        estimated_residual: rel_residual,
        backward_error: rel_residual,
        attainable_accuracy: AccuracyStatus::NotAttained,
        converged: false,
        status: SolverStatus::MaxIterationsReached,
        op_counts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num_complex::Complex64;
    use sonium_solvers_sparse::CsrMatrix;

    #[test]
    fn direct_gmres_rejects_non_finite_input_before_operator_use() {
        struct PanicOperator;
        impl LinearOperator<f64> for PanicOperator {
            fn num_rows(&self) -> usize {
                1
            }

            fn num_cols(&self) -> usize {
                1
            }

            fn apply_into(&self, _x: &[f64], _y: &mut [f64]) {
                panic!("operator must not be called")
            }

            fn apply_transpose_into(&self, _x: &[f64], _y: &mut [f64]) {
                panic!("operator must not be called")
            }
        }

        let solution = gmres(
            &PanicOperator,
            &[f64::NAN],
            &GmresConfig {
                max_iterations: 4,
                restart: 2,
                tolerance: 1.0e-8,
                ..GmresConfig::default()
            },
        );
        assert_eq!(solution.status, SolverStatus::NumericalError);
        assert!(!solution.converged);
    }

    #[test]
    fn checked_gmres_rejects_bad_dimensions_and_non_finite_rhs() {
        let matrix = CsrMatrix::<f64>::identity(2);
        let config = GmresConfig::default();

        let error = try_gmres(&matrix, &[1.0], &config)
            .expect_err("dimension-mismatched RHS must be rejected");
        assert!(matches!(
            error,
            SolverError::DimensionMismatch {
                expected: 2,
                got: 1
            }
        ));

        let error = try_gmres(&matrix, &[f64::NAN, 1.0], &config)
            .expect_err("non-finite RHS must be rejected");
        assert!(matches!(
            error,
            SolverError::OperationFailed {
                operation: "GMRES",
                ..
            }
        ));
    }

    #[test]
    fn checked_gmres_contains_operator_panics() {
        struct PanicOperator;

        impl LinearOperator<f64> for PanicOperator {
            fn num_rows(&self) -> usize {
                1
            }

            fn num_cols(&self) -> usize {
                1
            }

            fn apply_into(&self, _x: &[f64], _y: &mut [f64]) {
                panic!("checked GMRES operator panic");
            }

            fn apply_transpose_into(&self, _x: &[f64], _y: &mut [f64]) {
                panic!("checked GMRES transpose operator panic");
            }
        }

        let error = try_gmres(&PanicOperator, &[1.0], &GmresConfig::default())
            .expect_err("operator panic must become a typed error");
        match error {
            SolverError::OperationFailed { operation, reason } => {
                assert_eq!(operation, "GMRES");
                assert!(reason.contains("checked GMRES operator panic"));
            }
            other => panic!("expected typed GMRES panic error, got {other:?}"),
        }
    }

    #[test]
    fn checked_gmres_preserves_nonconverged_status() {
        let zero = CsrMatrix::from_triplets(2, 2, Vec::<(usize, usize, f64)>::new());
        let result = try_gmres(&zero, &[1.0, -2.0], &GmresConfig::default())
            .expect("non-convergence is a solver status, not an API panic");
        assert_ne!(result.status, SolverStatus::Converged);
        assert!(!result.converged);
        assert!(result.residual.is_finite());
    }

    #[test]
    fn finalize_gmres_rejects_invalid_projected_residual_metric() {
        let matrix = CsrMatrix::<f64>::identity(1);
        let result = GmresSolution {
            x: vec![1.0],
            iterations: 1,
            restarts: 0,
            residual: 0.0,
            estimated_residual: f64::NAN,
            backward_error: 0.0,
            attainable_accuracy: AccuracyStatus::NotChecked,
            converged: false,
            status: SolverStatus::Breakdown,
            op_counts: KrylovOpCounts::default(),
        };
        let finalized = finalize_gmres_solution(&matrix, &[1.0], 1.0e-12, result);
        assert_eq!(finalized.status, SolverStatus::NumericalError);
        assert!(!finalized.converged);
        assert_eq!(finalized.attainable_accuracy, AccuracyStatus::NotChecked);
        assert!(finalized.residual.is_infinite());
        assert!(finalized.backward_error.is_infinite());
    }

    #[test]
    fn finalize_gmres_rejects_negative_projected_residual_metric() {
        let matrix = CsrMatrix::<f64>::identity(1);
        let result = GmresSolution {
            x: vec![1.0],
            iterations: 1,
            restarts: 0,
            residual: 0.0,
            estimated_residual: -f64::EPSILON,
            backward_error: 0.0,
            attainable_accuracy: AccuracyStatus::NotChecked,
            converged: false,
            status: SolverStatus::Breakdown,
            op_counts: KrylovOpCounts::default(),
        };
        let finalized = finalize_gmres_solution(&matrix, &[1.0], 1.0e-12, result);
        assert_eq!(finalized.status, SolverStatus::NumericalError);
        assert!(!finalized.converged);
        assert_eq!(finalized.attainable_accuracy, AccuracyStatus::NotChecked);
    }

    #[test]
    fn test_gmres_simple() {
        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ];

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];

        let config = GmresConfig {
            max_iterations: 100,
            restart: 10,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let solution = gmres(&a, &b, &config);

        assert!(solution.converged, "GMRES should converge");

        // Verify solution: Ax ≈ b
        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8, "Solution should satisfy Ax = b");
    }

    #[test]
    fn test_gmres_identity() {
        let n = 5;
        let id: CsrMatrix<Complex64> = CsrMatrix::identity(n);
        let b = (1..=n)
            .map(|i| Complex64::new(i as f64, 0.0))
            .collect::<Vec<_>>();

        let config = GmresConfig {
            max_iterations: 10,
            restart: 10,
            tolerance: 1e-12,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let solution = gmres(&id, &b, &config);

        assert!(solution.converged);
        assert!(solution.iterations <= 2);

        let error: f64 = sonium_solvers_core::vec_ops::vec_sub(&solution.x, &b)
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-10);
    }

    #[test]
    fn gmres_does_not_confuse_preconditioned_scaling_with_convergence() {
        struct ScalingPreconditioner;

        impl Preconditioner<f64> for ScalingPreconditioner {
            fn apply_into(&self, r: &[f64], y: &mut [f64]) {
                y[0] = 1.0e12 * r[0];
                y[1] = r[1];
            }
        }

        let identity = CsrMatrix::identity(2);
        let rhs = [1.0_f64, 1.0];
        let initial = [1.0_f64, 0.0];
        let config = GmresConfig {
            max_iterations: 10,
            restart: 2,
            tolerance: 1.0e-8,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let solution = gmres_preconditioned_with_guess(
            &identity,
            &ScalingPreconditioner,
            &rhs,
            Some(&initial),
            &config,
        );

        assert!(solution.converged);
        assert!(solution.residual < config.tolerance);

        assert!((solution.x[0] - 1.0).abs() < 1.0e-12);
        assert!((solution.x[1] - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn test_gmres_exact_residual_frequency_preserves_mandatory_gate() {
        let n = 6;
        let id: CsrMatrix<Complex64> = CsrMatrix::identity(n);
        let b = (1..=n)
            .map(|i| Complex64::new(i as f64, 0.0))
            .collect::<Vec<_>>();

        let base_config = GmresConfig {
            max_iterations: 10,
            restart: 10,
            tolerance: 1e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
            min_restart: None,
            max_restart: None,
        };
        let checked_config = GmresConfig {
            exact_residual_check_interval: 1,
            ..base_config.clone()
        };

        let base = gmres(&id, &b, &base_config);
        let checked = gmres(&id, &b, &checked_config);

        assert!(base.converged);
        assert!(checked.converged);
        assert_eq!(base.iterations, checked.iterations);
        assert_eq!(base.op_counts.exact_residual_checks, 1);
        assert_eq!(checked.op_counts.exact_residual_checks, 1);
        assert_eq!(checked.op_counts.spmv_count, base.op_counts.spmv_count);
    }

    #[test]
    fn test_gmres_f64() {
        let dense = sonium_solvers_core::array![[4.0_f64, 1.0], [1.0, 3.0],];

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![1.0_f64, 2.0];

        let config = GmresConfig {
            max_iterations: 100,
            restart: 10,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let solution = gmres(&a, &b, &config);

        assert!(solution.converged);
        assert!(solution.op_counts.spmv_count >= solution.iterations);
        assert!(solution.op_counts.dot_reductions > 0);
        assert!(solution.op_counts.basis_bytes_touched > 0);

        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt();
        assert_relative_eq!(error, 0.0, epsilon = 1e-8);
    }

    #[test]
    fn test_gmres_restart_behavior() {
        // System that needs multiple restarts: restart=5 on a 20-node system
        // with tight tolerance forces multiple restart cycles.
        let n = 20;
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        for i in 0..n {
            dense[[i, i]] = Complex64::new(4.0, 0.0);
            if i > 0 {
                dense[[i, i - 1]] = Complex64::new(-1.0, 0.0);
            }
            if i < n - 1 {
                dense[[i, i + 1]] = Complex64::new(-1.0, 0.0);
            }
        }

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();

        let config = GmresConfig {
            max_iterations: 100,
            restart: 5,
            tolerance: 1e-12,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let solution = gmres(&a, &b, &config);
        assert!(solution.converged, "GMRES should converge with restarts");
        assert!(
            solution.restarts > 0,
            "Should need at least one restart with restart=5 on 20-node system"
        );

        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8, "Solution error too large: {}", error);
    }

    #[test]
    fn test_gmres_complex() {
        // Complex system with non-trivial imaginary parts
        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 1.0), Complex64::new(1.0, -0.5)],
            [Complex64::new(1.0, 0.5), Complex64::new(3.0, -1.0)],
        ];

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(2.0, 1.0), Complex64::new(1.0, -1.0)];

        let config = GmresConfig {
            max_iterations: 100,
            restart: 10,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let solution = gmres(&a, &b, &config);
        assert!(
            solution.converged,
            "GMRES should converge for complex system"
        );

        let ax = a.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-8, "Complex solution error too large: {}", error);
    }

    #[test]
    fn test_gmres_preconditioned_ilu() {
        // GMRES with ILU preconditioner on a 10-node tridiagonal system
        use sonium_solvers_preconditioners::IluPreconditioner;

        let n = 10;
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        for i in 0..n {
            dense[[i, i]] = Complex64::new(4.0, 0.0);
            if i > 0 {
                dense[[i, i - 1]] = Complex64::new(-1.0, 0.0);
            }
            if i < n - 1 {
                dense[[i, i + 1]] = Complex64::new(-1.0, 0.0);
            }
        }

        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let precond = IluPreconditioner::from_csr(&matrix);
        let b = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();

        let config = GmresConfig {
            max_iterations: 50,
            restart: 10,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let solution = gmres_preconditioned(&matrix, &precond, &b, &config);
        assert!(solution.converged, "Preconditioned GMRES should converge");

        // Verify the actual solution satisfies Ax = b
        let ax = matrix.matvec(&solution.x);
        let error: f64 = (sonium_solvers_core::vec_ops::vec_sub(&ax, &b))
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(
            error < 1e-8,
            "ILU-preconditioned GMRES solution error: {}",
            error
        );
    }

    #[test]
    fn test_gmres_convergence_check_interval_zero() {
        // Regression: convergence_check_interval = 0 must not panic with
        // "attempt to calculate the remainder with a divisor of zero".
        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ];

        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];

        let config = GmresConfig {
            convergence_check_interval: 0,
            ..Default::default()
        };

        let solution = gmres(&a, &b, &config);
        assert!(
            solution.converged,
            "GMRES should converge even with convergence_check_interval=0"
        );
    }

    #[test]
    fn test_gmres_breakdown_not_false_convergence() {
        // Regression: a pathological preconditioner that maps everything to
        // near-zero must not be reported as Converged when the true residual
        // is still large.
        use sonium_solvers_core::traits::{ComplexField, Preconditioner};

        struct ZeroPreconditioner;

        impl<T: ComplexField> Preconditioner<T> for ZeroPreconditioner {
            fn apply(&self, r: &[T]) -> Vec<T> {
                vec![T::zero(); r.len()]
            }
            fn apply_into(&self, _r: &[T], y: &mut [T]) {
                y.fill(T::zero());
            }
        }

        let dense = sonium_solvers_core::array![
            [Complex64::new(4.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ];
        let a = CsrMatrix::from_dense(&dense, 1e-15);
        let b = sonium_solvers_core::array![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];

        let config = GmresConfig {
            max_iterations: 10,
            restart: 5,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let solution = gmres_preconditioned(&a, &ZeroPreconditioner, &b, &config);
        // With a zero preconditioner the true residual is still large,
        // so the solver should NOT claim convergence.
        assert!(
            !solution.converged,
            "Zero preconditioner should not produce false convergence"
        );
        assert!(
            solution.status == SolverStatus::Breakdown
                || solution.status == SolverStatus::MaxIterationsReached,
            "Expected Breakdown or MaxIterationsReached, got {:?}",
            solution.status
        );
    }

    #[test]
    fn test_gmres_restart_accepts_requested_true_residual() {
        struct AnisotropicPreconditioner;

        impl Preconditioner<f64> for AnisotropicPreconditioner {
            fn apply_into(&self, r: &[f64], y: &mut [f64]) {
                y[0] = r[0];
                y[1] = 100.0 * r[1];
            }
        }

        // After one GMRES(1) cycle the true relative residual is about 1.4e-4,
        // while the left-preconditioned relative residual remains about 1e-2.
        // The public tolerance is a true Ax=b accuracy request, so the restart
        // boundary must accept this solution rather than exhaust the budget.
        let a: CsrMatrix<f64> = CsrMatrix::identity(2);
        let b = [1.0, 1.0e-6];
        let config = GmresConfig {
            max_iterations: 10,
            restart: 1,
            tolerance: 1.0e-3,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
            min_restart: None,
            max_restart: None,
            print_interval: 0,
        };

        let solution = gmres_preconditioned(&a, &AnisotropicPreconditioner, &b, &config);

        assert!(
            solution.converged,
            "true residual met the requested tolerance at restart: residual={:.3e}",
            solution.residual
        );
        assert!(solution.residual < config.tolerance);
        assert_eq!(solution.iterations, 1);
    }

    #[test]
    fn test_gmres_inner_convergence_reports_true_residual_with_anisotropic_preconditioner() {
        struct AnisotropicPreconditioner;

        impl Preconditioner<f64> for AnisotropicPreconditioner {
            fn apply_into(&self, r: &[f64], y: &mut [f64]) {
                y[0] = r[0];
                y[1] = 100.0 * r[1];
            }
        }

        let a: CsrMatrix<f64> = CsrMatrix::identity(2);
        let b = [1.0, 1.0];
        let config = GmresConfig {
            max_iterations: 1,
            restart: 2,
            tolerance: 1.0e-1,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
            min_restart: None,
            max_restart: None,
            ..GmresConfig::default()
        };

        let solution = gmres_preconditioned(&a, &AnisotropicPreconditioner, &b, &config);
        let ax = a.matvec(&solution.x);
        let true_residual = sonium_solvers_core::vec_ops::vec_sub(&ax, &b)
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt();

        assert!(!solution.converged);
        assert!(
            solution.status == SolverStatus::MaxIterationsReached
                || solution.status == SolverStatus::Breakdown
        );
        assert!(true_residual / vector_norm(&b) >= config.tolerance);
    }

    #[test]
    fn test_gmres_zero_rhs_corrects_non_solution_initial_guess() {
        let a: CsrMatrix<f64> = CsrMatrix::identity(2);
        let b = [0.0, 0.0];
        let x0 = [1.0, -2.0];

        let solution = gmres_preconditioned_with_guess(
            &a,
            &IdentityPreconditioner,
            &b,
            Some(&x0),
            &GmresConfig::default(),
        );

        assert!(solution.converged, "status={:?}", solution.status);
        assert_eq!(solution.status, SolverStatus::Converged);
        assert!(solution.residual < 1e-12);
        assert!(solution.x.iter().all(|value| value.abs() < 1e-12));
    }

    #[test]
    fn try_gmres_preconditioned_reports_typed_validation_and_success() {
        let matrix = CsrMatrix::<f64>::identity(2);
        let preconditioner = IdentityPreconditioner;
        let config = GmresConfig {
            max_iterations: 4,
            restart: 2,
            tolerance: 1.0e-12,
            ..GmresConfig::default()
        };
        let solution = try_gmres_preconditioned(&matrix, &preconditioner, &[1.0, -2.0], &config)
            .expect("checked preconditioned GMRES should converge");
        assert!(solution.converged);
        assert!(solution.residual < config.tolerance);

        let continuation = try_gmres_preconditioned_with_guess(
            &matrix,
            &preconditioner,
            &[1.0, -2.0],
            Some(&solution.x),
            &config,
        )
        .expect("checked preconditioned continuation should converge");
        assert!(continuation.converged);
        assert_eq!(continuation.iterations, 0);

        let error = try_gmres_preconditioned_with_guess(
            &matrix,
            &preconditioner,
            &[1.0, -2.0],
            Some(&[1.0]),
            &config,
        )
        .expect_err("malformed initial guess must be rejected");
        assert!(matches!(error, SolverError::DimensionMismatch { .. }));

        let bad_config = GmresConfig {
            max_iterations: 0,
            ..config.clone()
        };
        let error = try_gmres_preconditioned(&matrix, &preconditioner, &[1.0, -2.0], &bad_config)
            .expect_err("zero iteration budget must be rejected");
        assert!(matches!(
            error,
            SolverError::OperationFailed {
                operation: "GMRES configuration",
                ..
            }
        ));

        let error = try_gmres_preconditioned(&matrix, &preconditioner, &[f64::NAN, 0.0], &config)
            .expect_err("non-finite RHS must be rejected");
        assert!(matches!(
            error,
            SolverError::OperationFailed {
                operation: "GMRES preconditioned",
                ..
            }
        ));
    }

    #[test]
    fn try_gmres_preconditioned_contains_preconditioner_panics() {
        struct PanickingPreconditioner;
        impl Preconditioner<f64> for PanickingPreconditioner {
            fn apply_into(&self, _r: &[f64], _y: &mut [f64]) {
                panic!("synthetic preconditioner panic");
            }
        }

        let matrix = CsrMatrix::<f64>::identity(2);
        let config = GmresConfig {
            max_iterations: 2,
            restart: 2,
            tolerance: 1.0e-8,
            ..GmresConfig::default()
        };
        let error =
            try_gmres_preconditioned(&matrix, &PanickingPreconditioner, &[1.0, 0.0], &config)
                .expect_err("preconditioner panic must become a typed error");
        assert!(matches!(
            error,
            SolverError::OperationFailed {
                operation: "GMRES preconditioned",
                ..
            }
        ));
    }
}
