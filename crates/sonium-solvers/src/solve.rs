//! High-level linear system solver with automatic preconditioner selection
//!
//! Provides a unified interface for solving `A x = b` with various solver/preconditioner
//! combinations. This consolidates the solver dispatch logic that was previously
//! scattered across FEM and BEM crates.

use crate::direct::SparseCholeskyFactorization;
#[cfg(feature = "cuda")]
use crate::iterative::{
    BackendKrylovConfig, DeviceGmresSession, DeviceInitialGuess, DevicePcgAssumption,
    DevicePcgSession,
};
use crate::iterative::{fgmres_preconditioned, gmres_pipelined, gmres_preconditioned_with_guess};
#[cfg(feature = "cuda")]
use crate::preconditioners::BackendAmgPreconditioner;
use crate::preconditioners::{
    AdditiveSchwarzPreconditioner, AmgConfig, AmgPreconditioner, ChebyshevConfig,
    ChebyshevPreconditioner, DiagonalPreconditioner, IluPreconditioner, SaddlePointBlockAssumption,
    SaddlePointBlockAssumptions, SaddlePointBlockProvenance, SaddlePointConfig,
    SaddlePointPreconditioner, SchurStrategy,
};
use crate::traits::{
    AccuracyStatus, ComplexField, IdentityPreconditioner, LinearOperator, Preconditioner,
    SolverError, SolverFailure, SolverResultContract, SolverStatus,
};
use crate::{CsrMatrix, FgmresConfig, GmresConfig, GmresSolution, SpmvPlanConfig};
use num_complex::{Complex32, Complex64};
use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
#[cfg(feature = "cuda")]
use sonium_compute::CudaBackend;
use sonium_compute::DeviceScalar;
#[cfg(feature = "cuda")]
use sonium_compute::{
    Backend, DevicePreconditioner, IdentityDevicePreconditioner, MemoryPool,
    MemoryPoolConfig, PcgBackendConfig, PersistentDeviceCsr, PointJacobiBackend,
};
use std::any::TypeId;
use std::panic::{AssertUnwindSafe, catch_unwind};
use thiserror::Error;

#[cfg(feature = "cuda")]
mod cuda_session;
#[cfg(feature = "cuda")]
#[doc(inline)]
pub use cuda_session::{CudaLinearSolveSession, CudaLinearSolveStatus, CudaLinearUpdateStatus};

#[cfg(all(test, feature = "cuda"))]
mod cuda_pcg_tests;
#[cfg(all(test, feature = "cuda"))]
mod cuda_session_tests;
#[cfg(all(test, feature = "cuda"))]
mod cuda_session_refresh_tests;
#[cfg(all(test, feature = "cuda"))]
mod cuda_prepared_tests;

/// Default dispatch cutoffs between direct/ILU and iterative/FMM paths.
///
/// Measured on Apple Silicon memory bandwidth (see `sonium-solvers-sys`
/// `runtime.rs`); multi-channel x86 servers saturate differently, so both
/// are env-overridable (`SONIUM_LARGE_PROBLEM_ROWS`,
/// `SONIUM_LARGE_PROBLEM_NNZ`). Existing tests pin the defaults.
const LARGE_PROBLEM_ROWS: usize = 50_000;
const LARGE_PROBLEM_NNZ: usize = 500_000;

fn large_problem_rows() -> usize {
    std::env::var("SONIUM_LARGE_PROBLEM_ROWS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(LARGE_PROBLEM_ROWS)
}

fn large_problem_nnz() -> usize {
    std::env::var("SONIUM_LARGE_PROBLEM_NNZ")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(LARGE_PROBLEM_NNZ)
}
/// Exact positive-definiteness validation is deliberately bounded.  Large
/// production systems should not pay for a second full factorization merely
/// to select an iterative solver; the structural checks below still reject
/// malformed/non-Hermitian inputs, while small dispatch decisions get the
/// same pivot test as the direct SPD path.
const SPD_EXACT_VALIDATION_MAX_ROWS: usize = 1_024;

fn assumption_provenance<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    solver: LinearSolverType,
    assumption: LinearSystemAssumption,
) -> AssumptionProvenance {
    let requires_spd = matches!(
        solver,
        LinearSolverType::GmresAmg
            | LinearSolverType::GmresChebyshev
            | LinearSolverType::Cg
            | LinearSolverType::CgAmg
    );
    if !requires_spd {
        return AssumptionProvenance::NotRequired;
    }
    if matrix.num_rows <= SPD_EXACT_VALIDATION_MAX_ROWS {
        AssumptionProvenance::IndependentlyValidated
    } else if matches!(
        assumption,
        LinearSystemAssumption::PositiveDefinite
            | LinearSystemAssumption::PositiveDefiniteCertified
    ) && has_scalable_spd_certificate(matrix)
    {
        AssumptionProvenance::ScalableCertificate
    } else if matches!(
        assumption,
        LinearSystemAssumption::PositiveDefinite
            | LinearSystemAssumption::PositiveDefiniteCertified
    ) {
        AssumptionProvenance::CallerDeclared
    } else {
        // General large-system requests for SPD-only strategies are
        // rejected before a result can be constructed.
        AssumptionProvenance::NotRequired
    }
}

/// Return whether a validated CSR matrix has a scalable sufficient HPD test.
///
/// A Hermitian matrix with a real positive diagonal that is strictly
/// diagonally dominant is positive definite by the Gershgorin theorem. This
/// helper intentionally provides provenance only; large-system dispatch still
/// requires the caller's explicit [`LinearSystemAssumption::PositiveDefinite`]
/// declaration.
fn has_scalable_spd_certificate<T: ComplexField>(matrix: &CsrMatrix<T>) -> bool {
    let hermitian_tolerance = T::Real::from_f64(1.0e-12).unwrap_or_else(T::Real::epsilon);
    let matrix = coalesced_csr(matrix);
    has_jacobi_scaled_spd_certificate(&matrix)
        || (matrix.num_rows > 0
            && matrix.num_rows == matrix.num_cols
            && (0..matrix.num_rows).all(|row| {
                let mut diagonal = None;
                let mut off_diagonal_sum = T::Real::zero();
                for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
                    let col = matrix.col_indices[index];
                    let value = matrix.values[index];
                    if col == row {
                        diagonal = Some(value);
                    } else {
                        let counterpart = matrix.get(col, row);
                        let scale = T::Real::one() + value.norm().max(counterpart.norm());
                        if (value - counterpart.conj()).norm() > hermitian_tolerance * scale {
                            return false;
                        }
                        off_diagonal_sum += value.norm();
                    }
                }
                diagonal.is_some_and(|value| {
                    value.im().abs() <= T::Real::from_f64(1.0e-12).unwrap_or_else(T::Real::epsilon)
                        && value.re() > off_diagonal_sum
                })
            }))
}

/// Check a symmetric Jacobi-scaled Gershgorin certificate. Congruence by the
/// positive diagonal scaling preserves HPD, so strict dominance after this
/// normalization is a valid scalable sufficient condition even when the raw
/// matrix is badly scaled.
fn has_jacobi_scaled_spd_certificate<T: ComplexField>(matrix: &CsrMatrix<T>) -> bool {
    let tolerance = T::Real::from_f64(1.0e-12).unwrap_or_else(T::Real::epsilon);
    if matrix.num_rows == 0 || matrix.num_rows != matrix.num_cols {
        return false;
    }
    let mut scales = vec![T::Real::zero(); matrix.num_rows];
    for (row, slot) in scales.iter_mut().enumerate() {
        let diagonal = (matrix.row_ptrs[row]..matrix.row_ptrs[row + 1])
            .find(|&index| matrix.col_indices[index] == row)
            .map(|index| matrix.values[index]);
        let Some(diagonal) = diagonal else {
            return false;
        };
        if !diagonal.re().is_finite()
            || !diagonal.im().is_finite()
            || diagonal.im().abs() > tolerance
            || diagonal.re() <= T::Real::zero()
        {
            return false;
        }
        *slot = diagonal.re().sqrt().recip();
    }
    (0..matrix.num_rows).all(|row| {
        let mut diagonal = None;
        let mut off_diagonal_sum = T::Real::zero();
        for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            let col = matrix.col_indices[index];
            if col >= matrix.num_cols {
                return false;
            }
            let value = matrix.values[index];
            if col != row {
                let counterpart = matrix.get(col, row);
                let scale = T::Real::one() + value.norm().max(counterpart.norm());
                if (value - counterpart.conj()).norm() > tolerance * scale {
                    return false;
                }
            }
            let scaled = scales[row] * scales[col] * value.norm();
            if col == row {
                diagonal = Some(scaled);
            } else {
                off_diagonal_sum += scaled;
            }
        }
        diagonal.is_some_and(|value| value > off_diagonal_sum)
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GmresDispatchVariant {
    Standard,
    Pipelined,
}

/// Solver strategy for linear systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearSolverType {
    /// Automatically select a strategy from matrix size and scalar type.
    ///
    /// Small systems keep ILU. Large real/SPD-looking systems prefer AMG with
    /// the FEM profile (Chebyshev smoothing); large complex systems use
    /// flexible GMRES (FGMRES) with additive Schwarz, so the outer iteration
    /// stays correct even if the Schwarz preconditioner becomes non-stationary
    /// (e.g. iterated subdomain solves), unless a caller explicitly requests
    /// another strategy.
    Auto,
    /// GMRES with no preconditioning.
    Gmres,
    /// GMRES with Jacobi (diagonal) preconditioning.
    GmresJacobi,
    /// GMRES with ILU(0) preconditioning (recommended for general use).
    GmresIlu,
    /// GMRES with AMG preconditioning (best for large SPD-like systems).
    GmresAmg,
    /// GMRES with Chebyshev polynomial preconditioning (parallel SPD smoother).
    GmresChebyshev,
    /// GMRES with additive Schwarz domain-decomposition preconditioning.
    ///
    /// Valid only while the Schwarz preconditioner is stationary (today it
    /// is: fixed per-subdomain ILU factorizations). If the Schwarz apply ever
    /// becomes non-stationary, this pairing is the wrong Krylov variant —
    /// use [`LinearSolverType::FgmresSchwarz`] instead.
    GmresSchwarz,
    /// FGMRES (flexible GMRES, right-preconditioned) with additive Schwarz
    /// domain-decomposition preconditioning.
    ///
    /// The prescribed pairing for Helmholtz-type indefinite systems
    /// (FGMRES + shifted-Laplacian): the flexible outer iteration tolerates a
    /// variable/non-stationary preconditioner, and with the current
    /// stationary Schwarz apply it reduces to right-preconditioned GMRES at
    /// the cost of one extra Krylov basis.
    FgmresSchwarz,
    /// CG for real SPD systems; the CUDA route also supports complex HPD systems.
    Cg,
    /// CG with AMG preconditioning; CUDA uses a stationary SPD/HPD backend cycle.
    CgAmg,
}

/// Caller-supplied structural knowledge for solver dispatch.
///
/// `General` is the safe default.  `PositiveDefinite` is an explicit caller
/// assertion that permits SPD/HPD-only strategies on large systems, where a
/// complete factorization-based definiteness proof would be too expensive for
/// automatic dispatch.  The solver still checks dimensions, finite values,
/// diagonal signs, and symmetry/Hermiticity before iterating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LinearSystemAssumption {
    #[default]
    General,
    PositiveDefinite,
    /// Positive-definite/HPD assumption that must also have an independent
    /// scalable certificate when bounded-size exact validation is skipped.
    ///
    /// The historical `PositiveDefinite` variant remains available for
    /// callers that explicitly accept the large-system trust boundary.
    PositiveDefiniteCertified,
}

/// Evidence level for a solver's symmetry/definiteness assumption.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssumptionProvenance {
    /// The selected solver does not require an SPD/HPD assumption.
    NotRequired,
    /// The matrix passed the independent bounded validation path.
    IndependentlyValidated,
    /// Independently validated by a scalable strict diagonal-dominance
    /// certificate (with the Hermitian/positive-diagonal checks performed by
    /// the solver boundary).
    ScalableCertificate,
    /// The caller supplied the assumption for a system too large for the
    /// independent validation path.
    CallerDeclared,
}

/// Backend requested by a high-level solve.
///
/// The high-level facade defaults to [`LinearSolveBackend::Cpu`] for
/// backwards compatibility.  CUDA requests are admitted only for the
/// explicitly supported native CSR/GMRES and PCG paths; callers can opt into an
/// explicitly labelled CPU fallback with [`LinearSolveFallback::AllowCpu`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearSolveBackend {
    Cpu,
    Cuda,
}

/// Policy for an unavailable or unsupported requested backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearSolveFallback {
    /// Return a typed error before entering the solve loop.
    Reject,
    /// Run the CPU facade and mark the result as a CPU fallback.
    AllowCpu,
}

/// Backend provenance carried by every successful high-level result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearSolveBackendProvenance {
    pub requested: LinearSolveBackend,
    pub actual: LinearSolveBackend,
    pub fallback: bool,
}

impl LinearSolveBackendProvenance {
    /// CPU execution provenance for application handoff builders: the result
    /// was requested on and executed by the CPU with no fallback involved.
    pub const fn cpu() -> Self {
        Self {
            requested: LinearSolveBackend::Cpu,
            actual: LinearSolveBackend::Cpu,
            fallback: false,
        }
    }

    #[cfg(feature = "cuda")]
    const fn native(backend: LinearSolveBackend) -> Self {
        Self {
            requested: backend,
            actual: backend,
            fallback: false,
        }
    }

    const fn cpu_fallback(requested: LinearSolveBackend) -> Self {
        Self {
            requested,
            actual: LinearSolveBackend::Cpu,
            fallback: true,
        }
    }
}

/// Configuration for the high-level solver.
#[derive(Debug, Clone)]
pub struct LinearSolveConfig {
    /// Solver type.
    pub solver: LinearSolverType,
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Convergence tolerance (relative residual).
    pub tolerance: f64,
    /// GMRES restart parameter.
    pub restart: usize,
    /// Verify estimated GMRES convergence with a full residual every N Krylov
    /// iterations (0 = disabled).
    pub exact_residual_check_interval: usize,
    /// Verbosity (0=quiet, 1=summary, 2+=detailed).
    pub verbosity: usize,
}

impl Default for LinearSolveConfig {
    fn default() -> Self {
        Self {
            solver: LinearSolverType::Auto,
            max_iterations: 1000,
            tolerance: 1e-10,
            restart: 30,
            exact_residual_check_interval: 0,
            verbosity: 0,
        }
    }
}

impl LinearSolveConfig {
    /// Validate settings before constructing a preconditioner or entering a
    /// Krylov loop. Keeping this check at the public boundary prevents a
    /// malformed restart/tolerance from turning into a misleading breakdown.
    pub fn validate(&self) -> Result<(), LinearSolveError> {
        if self.max_iterations == 0 {
            return Err(LinearSolveError::InvalidConfig(
                "max_iterations must be greater than zero".into(),
            ));
        }
        if self.restart == 0 {
            return Err(LinearSolveError::InvalidConfig(format!(
                "restart must be greater than zero (got {})",
                self.restart
            )));
        }
        if !self.tolerance.is_finite() || self.tolerance <= 0.0 {
            return Err(LinearSolveError::InvalidConfig(
                "tolerance must be finite and greater than zero".into(),
            ));
        }
        Ok(())
    }

    /// Resolve [`LinearSolverType::Auto`] for real systems.
    pub fn resolved_solver_for_real(&self, matrix: &CsrMatrix<f64>) -> LinearSolverType {
        resolve_auto_solver(self.solver, matrix, false)
    }

    /// Resolve [`LinearSolverType::Auto`] for complex systems.
    ///
    /// Large complex systems (the Helmholtz FEM regime) route to
    /// [`LinearSolverType::FgmresSchwarz`]: the flexible outer iteration is
    /// the prescribed partner for a Schwarz/shifted-Laplacian preconditioner
    /// and remains correct if the preconditioner ever becomes non-stationary.
    pub fn resolved_solver_for_complex(&self, matrix: &CsrMatrix<Complex64>) -> LinearSolverType {
        if self.solver != LinearSolverType::Auto {
            return self.solver;
        }
        if is_large_problem(matrix) {
            LinearSolverType::FgmresSchwarz
        } else {
            LinearSolverType::GmresIlu
        }
    }
}

/// Result of a linear system solve.
#[derive(Debug, Clone)]
pub struct LinearSolveResult<T> {
    /// Solution vector.
    pub x: Vec<T>,
    /// Number of iterations.
    pub iterations: usize,
    /// Final relative residual.
    pub residual: f64,
    /// Normwise backward error computed from the original matrix, RHS, and
    /// returned solution. This is independent of the Krylov estimator.
    pub backward_error: f64,
    /// Whether convergence was achieved.
    pub converged: bool,
    /// Canonical solver status for the returned result.
    pub status: SolverStatus,
    /// Whether the requested tolerance was attained by the returned iterate.
    pub attainable_accuracy: AccuracyStatus,
    /// Requested and actual execution backend.  A CPU result produced after
    /// a CUDA request is explicitly marked as a fallback.
    pub backend: LinearSolveBackendProvenance,
    /// Assumption supplied to the dispatch boundary.
    pub assumption: LinearSystemAssumption,
    /// Evidence supporting the assumption used by the selected solver.
    pub assumption_provenance: AssumptionProvenance,
    /// Assumption evidence for the A and Schur blocks of a saddle-point
    /// solve. `None` for ordinary linear systems; the monolithic saddle
    /// matrix itself remains classified as `General`.
    pub saddle_assumptions: Option<SaddlePointAssumptionReport>,
}

impl<T> LinearSolveResult<T> {
    /// Short-form method for the canonical status.
    pub fn status(&self) -> SolverStatus {
        self.solver_status()
    }

    /// Method form of the canonical status for compatibility callers that do
    /// not access result fields directly.
    pub fn solver_status(&self) -> SolverStatus {
        self.status
    }

    /// Method form of the canonical accuracy state for compatibility callers.
    pub fn accuracy_status(&self) -> AccuracyStatus {
        self.attainable_accuracy
    }

    pub fn attainable_accuracy(&self) -> AccuracyStatus {
        self.accuracy_status()
    }
}

impl<T: ComplexField> LinearSolveResult<T> {
    /// Validate the public result/status/accuracy contract before a result is
    /// handed to compatibility callers.
    pub fn validate_contract(&self) -> Result<(), LinearSolveError> {
        if let Some(index) = self.x.iter().position(|value| !value.norm().is_finite()) {
            return Err(LinearSolveError::NonFiniteOutput {
                operation: "linear solve result",
                index,
            });
        }
        if !self.residual.is_finite() || self.residual < 0.0 {
            return Err(LinearSolveError::ContractViolation(
                "residual must be finite and non-negative".into(),
            ));
        }
        if !self.backward_error.is_finite() || self.backward_error < 0.0 {
            return Err(LinearSolveError::ContractViolation(
                "backward error must be finite and non-negative".into(),
            ));
        }

        let backend_matches = if self.backend.fallback {
            self.backend.requested != LinearSolveBackend::Cpu
                && self.backend.actual == LinearSolveBackend::Cpu
        } else {
            self.backend.requested == self.backend.actual
        };
        if !backend_matches {
            return Err(LinearSolveError::ContractViolation(
                "backend provenance must identify native execution or an explicit non-CPU fallback"
                    .into(),
            ));
        }

        let status_matches = self.converged == (self.status == SolverStatus::Converged);
        if !status_matches {
            return Err(LinearSolveError::ContractViolation(
                "converged flag and solver status disagree".into(),
            ));
        }
        let accuracy_matches = match self.status {
            SolverStatus::Converged => self.attainable_accuracy == AccuracyStatus::Attained,
            _ => self.attainable_accuracy != AccuracyStatus::Attained,
        };
        if !accuracy_matches {
            return Err(LinearSolveError::ContractViolation(
                "solver status and attainable accuracy disagree".into(),
            ));
        }
        Ok(())
    }
}

impl<T: ComplexField> SolverResultContract for LinearSolveResult<T> {
    fn solver_status(&self) -> SolverStatus {
        self.status
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        self.attainable_accuracy
    }

    fn validate_contract(&self) -> Result<(), SolverError> {
        LinearSolveResult::validate_contract(self).map_err(|error| SolverError::OperationFailed {
            operation: "linear solve result contract",
            reason: error.to_string(),
        })
    }
}

/// Assumption evidence for the SPD/HPD-only blocks selected by a saddle-point
/// preconditioner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SaddlePointAssumptionReport {
    pub block1: SaddlePointBlockAssumption,
    pub block1_provenance: AssumptionProvenance,
    pub block2: SaddlePointBlockAssumption,
    pub block2_provenance: AssumptionProvenance,
}

/// Error from the linear solver.
#[derive(Debug, Error)]
pub enum LinearSolveError {
    #[error("Solver did not converge in {iterations} iterations (residual = {residual:.2e})")]
    NotConverged { iterations: usize, residual: f64 },
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("{operation} dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        operation: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("Preconditioner construction failed: {0}")]
    PreconditionerFailed(String),
    #[error("non-finite input in {operation} at index {index}")]
    NonFiniteInput {
        operation: &'static str,
        index: usize,
    },
    #[error("non-finite output from {operation} at index {index}")]
    NonFiniteOutput {
        operation: &'static str,
        index: usize,
    },
    #[error("solver result contract violation: {0}")]
    ContractViolation(String),
    #[error("requested backend {backend:?} is unavailable: {reason}")]
    BackendUnavailable {
        backend: LinearSolveBackend,
        reason: String,
    },
    #[error("{backend:?} backend solve failed: {reason}")]
    BackendFailure {
        backend: LinearSolveBackend,
        reason: String,
    },
}

impl LinearSolveError {
    /// Return the canonical status corresponding to this checked failure.
    pub fn status(&self) -> SolverStatus {
        match self {
            Self::NotConverged { .. } => SolverStatus::MaxIterationsReached,
            Self::InvalidConfig(_)
            | Self::DimensionMismatch { .. }
            | Self::PreconditionerFailed(_)
            | Self::NonFiniteInput { .. }
            | Self::NonFiniteOutput { .. }
            | Self::ContractViolation(_)
            | Self::BackendUnavailable { .. }
            | Self::BackendFailure { .. } => SolverStatus::NumericalError,
        }
    }

    /// Return the attainable-accuracy state corresponding to this failure.
    pub fn attainable_accuracy(&self) -> AccuracyStatus {
        match self {
            Self::NotConverged { .. } => AccuracyStatus::NotAttained,
            Self::InvalidConfig(_)
            | Self::DimensionMismatch { .. }
            | Self::PreconditionerFailed(_)
            | Self::NonFiniteInput { .. }
            | Self::NonFiniteOutput { .. }
            | Self::ContractViolation(_)
            | Self::BackendUnavailable { .. }
            | Self::BackendFailure { .. } => AccuracyStatus::NotChecked,
        }
    }

    pub fn solver_status(&self) -> SolverStatus {
        self.status()
    }

    pub fn accuracy_status(&self) -> AccuracyStatus {
        self.attainable_accuracy()
    }
}

impl SolverFailure for LinearSolveError {
    fn error_kind(&self) -> sonium_solvers_core::traits::SolverErrorKind {
        use sonium_solvers_core::traits::SolverErrorKind;
        match self {
            Self::NotConverged { .. } => SolverErrorKind::Convergence,
            Self::DimensionMismatch { .. } => SolverErrorKind::DimensionMismatch,
            _ => SolverErrorKind::OperationFailed,
        }
    }

    fn status(&self) -> SolverStatus {
        LinearSolveError::status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        LinearSolveError::accuracy_status(self)
    }
}

fn validate_linear_inputs<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    rhs: &[T],
    x0: Option<&[T]>,
    config: &LinearSolveConfig,
) -> Result<(), LinearSolveError> {
    config.validate()?;
    validate_csr_structure(matrix)?;
    if matrix.num_rows != matrix.num_cols {
        return Err(LinearSolveError::DimensionMismatch {
            operation: "linear solve matrix",
            expected: matrix.num_rows,
            actual: matrix.num_cols,
        });
    }
    if rhs.len() != matrix.num_rows {
        return Err(LinearSolveError::DimensionMismatch {
            operation: "linear solve rhs",
            expected: matrix.num_rows,
            actual: rhs.len(),
        });
    }
    if let Some(x0) = x0
        && x0.len() != matrix.num_cols
    {
        return Err(LinearSolveError::DimensionMismatch {
            operation: "linear solve initial guess",
            expected: matrix.num_cols,
            actual: x0.len(),
        });
    }
    if let Some((index, _)) = matrix
        .values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.norm().is_finite())
    {
        return Err(LinearSolveError::NonFiniteInput {
            operation: "matrix",
            index,
        });
    }
    if let Some((index, _)) = rhs
        .iter()
        .enumerate()
        .find(|(_, value)| !value.norm().is_finite())
    {
        return Err(LinearSolveError::NonFiniteInput {
            operation: "rhs",
            index,
        });
    }
    if let Some(x0) = x0
        && let Some((index, _)) = x0
            .iter()
            .enumerate()
            .find(|(_, value)| !value.norm().is_finite())
    {
        return Err(LinearSolveError::NonFiniteInput {
            operation: "initial guess",
            index,
        });
    }
    Ok(())
}

/// Construct the preconditioners used by the high-level dispatch through
/// their checked APIs.  The public `from_csr` constructors remain available
/// for compatibility, but checked solver entry points must turn setup
/// failures into the high-level error contract rather than panic.
fn try_diagonal_preconditioner<T: ComplexField>(
    matrix: &CsrMatrix<T>,
) -> Result<DiagonalPreconditioner<T>, LinearSolveError> {
    catch_unwind(AssertUnwindSafe(|| {
        DiagonalPreconditioner::try_from_csr(matrix)
    }))
    .map_err(|_| {
        LinearSolveError::PreconditionerFailed("diagonal preconditioner setup panicked".into())
    })?
    .map_err(|error| LinearSolveError::PreconditionerFailed(error.to_string()))
}

fn try_amg_preconditioner<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    config: AmgConfig,
) -> Result<AmgPreconditioner<T>, LinearSolveError>
where
    T::Real: Sync + Send,
{
    catch_unwind(AssertUnwindSafe(|| {
        AmgPreconditioner::try_from_csr(matrix, config)
    }))
    .map_err(|_| {
        LinearSolveError::PreconditionerFailed("AMG preconditioner setup panicked".into())
    })?
    .map_err(|error| LinearSolveError::PreconditionerFailed(error.to_string()))
}

fn try_chebyshev_preconditioner<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    config: ChebyshevConfig,
) -> Result<ChebyshevPreconditioner<T>, LinearSolveError> {
    catch_unwind(AssertUnwindSafe(|| {
        ChebyshevPreconditioner::try_from_csr(matrix, config)
    }))
    .map_err(|_| {
        LinearSolveError::PreconditionerFailed("Chebyshev preconditioner setup panicked".into())
    })?
    .map_err(|error| LinearSolveError::PreconditionerFailed(error.to_string()))
}

fn try_schwarz_preconditioner<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    overlap: usize,
) -> Result<AdditiveSchwarzPreconditioner<T>, LinearSolveError> {
    catch_unwind(AssertUnwindSafe(|| {
        AdditiveSchwarzPreconditioner::try_from_csr_auto(matrix, overlap)
    }))
    .map_err(|_| {
        LinearSolveError::PreconditionerFailed("Schwarz preconditioner setup panicked".into())
    })?
    .map_err(|error| LinearSolveError::PreconditionerFailed(error.to_string()))
}

/// Validate the public CSR storage before any row range is traversed.
///
/// `CsrMatrix` keeps its arrays public for compatibility, so callers can
/// mutate a structurally invalid value after construction. High-level solve
/// APIs must reject that state before preconditioner or operator setup rather
/// than allowing an out-of-bounds row/column access to panic.
fn validate_csr_structure<T: ComplexField>(matrix: &CsrMatrix<T>) -> Result<(), LinearSolveError> {
    let expected_row_ptrs = matrix.num_rows.saturating_add(1);
    if matrix.row_ptrs.len() != expected_row_ptrs {
        return Err(LinearSolveError::InvalidConfig(format!(
            "invalid CSR structure: row_ptrs has length {}, expected {}",
            matrix.row_ptrs.len(),
            expected_row_ptrs
        )));
    }
    if matrix.col_indices.len() != matrix.values.len() {
        return Err(LinearSolveError::InvalidConfig(format!(
            "invalid CSR structure: col_indices has length {}, values has length {}",
            matrix.col_indices.len(),
            matrix.values.len()
        )));
    }
    if matrix.row_ptrs.first().copied() != Some(0) {
        return Err(LinearSolveError::InvalidConfig(
            "invalid CSR structure: row_ptrs must start at zero".into(),
        ));
    }
    if matrix
        .row_ptrs
        .windows(2)
        .any(|window| window[0] > window[1])
    {
        return Err(LinearSolveError::InvalidConfig(
            "invalid CSR structure: row_ptrs must be monotonically nondecreasing".into(),
        ));
    }
    if matrix.row_ptrs.last().copied() != Some(matrix.values.len()) {
        return Err(LinearSolveError::InvalidConfig(
            "invalid CSR structure: terminal row pointer must equal nnz".into(),
        ));
    }
    if let Some((index, &column)) = matrix
        .col_indices
        .iter()
        .enumerate()
        .find(|&(_, &column)| column >= matrix.num_cols)
    {
        return Err(LinearSolveError::InvalidConfig(format!(
            "invalid CSR structure: column index {column} at position {index} is out of range"
        )));
    }
    Ok(())
}

fn first_non_finite<T: ComplexField>(values: &[T]) -> Option<usize> {
    values.iter().position(|value| !value.norm().is_finite())
}

/// Compute the residual from the original matrix and returned solution.
/// Krylov estimates are useful for scheduling, but this value is the sole
/// convergence value exposed by the high-level solve API.
fn true_relative_residual<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    rhs: &[T],
    x: &[T],
) -> Result<f64, LinearSolveError> {
    if x.len() != matrix.num_cols || rhs.len() != matrix.num_rows {
        return Err(LinearSolveError::InvalidConfig(
            "residual input shape mismatch".into(),
        ));
    }
    if let Some(index) = first_non_finite(x) {
        return Err(LinearSolveError::NonFiniteOutput {
            operation: "solution",
            index,
        });
    }
    let ax = matrix.apply(x);
    if let Some(index) = first_non_finite(&ax) {
        return Err(LinearSolveError::NonFiniteOutput {
            operation: "operator application",
            index,
        });
    }
    let mut residual = Vec::with_capacity(rhs.len());
    for (&b, &ax_i) in rhs.iter().zip(ax.iter()) {
        residual.push(b - ax_i);
    }
    let residual_norm = T::vec_norm(&residual);
    let rhs_norm = T::vec_norm(rhs);
    if !residual_norm.is_finite() || !rhs_norm.is_finite() {
        return Err(LinearSolveError::NonFiniteOutput {
            operation: "residual norm",
            index: 0,
        });
    }
    let relative = if rhs_norm > T::Real::zero() {
        residual_norm / rhs_norm
    } else {
        residual_norm
    };
    relative
        .to_f64()
        .ok_or_else(|| LinearSolveError::InvalidConfig("residual conversion failed".into()))
}

/// Compute a normwise backward error from the original linear system.
///
/// The Frobenius bound `||r||₂ / (||A||_F ||x||₂ + ||b||₂)` is cheap, stable,
/// and valid for real and complex systems. A zero denominator is handled as
/// an exact-zero-system case rather than producing NaN.
fn true_backward_error<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    rhs: &[T],
    x: &[T],
) -> Result<f64, LinearSolveError> {
    if x.len() != matrix.num_cols || rhs.len() != matrix.num_rows {
        return Err(LinearSolveError::InvalidConfig(
            "backward-error input shape mismatch".into(),
        ));
    }
    if let Some(index) = first_non_finite(x) {
        return Err(LinearSolveError::NonFiniteOutput {
            operation: "solution",
            index,
        });
    }
    let ax = matrix.apply(x);
    if let Some(index) = first_non_finite(&ax) {
        return Err(LinearSolveError::NonFiniteOutput {
            operation: "operator application",
            index,
        });
    }
    let residual: Vec<T> = rhs
        .iter()
        .zip(ax.iter())
        .map(|(&b, &ax_i)| b - ax_i)
        .collect();
    let residual_norm = T::vec_norm(&residual);
    let matrix_norm = T::vec_norm(&matrix.values);
    let x_norm = T::vec_norm(x);
    let rhs_norm = T::vec_norm(rhs);
    if !residual_norm.is_finite()
        || !matrix_norm.is_finite()
        || !x_norm.is_finite()
        || !rhs_norm.is_finite()
    {
        return Err(LinearSolveError::NonFiniteOutput {
            operation: "backward error",
            index: 0,
        });
    }
    let denominator = matrix_norm * x_norm + rhs_norm;
    let backward = if denominator > T::Real::zero() {
        residual_norm / denominator
    } else {
        residual_norm
    };
    backward
        .to_f64()
        .ok_or_else(|| LinearSolveError::InvalidConfig("backward-error conversion failed".into()))
}

#[cfg(test)]
mod generated_output_tests {
    use super::*;

    #[test]
    fn true_metrics_reject_non_finite_solution() {
        let matrix = CsrMatrix::identity(1);
        let error = true_relative_residual(&matrix, &[1.0], &[f64::NAN])
            .expect_err("non-finite generated solution must be rejected");
        assert!(matches!(
            error,
            LinearSolveError::NonFiniteOutput {
                operation: "solution",
                index: 0
            }
        ));

        let error = true_backward_error(&matrix, &[1.0], &[f64::INFINITY])
            .expect_err("non-finite generated solution must be rejected");
        assert!(matches!(
            error,
            LinearSolveError::NonFiniteOutput {
                operation: "solution",
                index: 0
            }
        ));
    }
}

/// Solve a real linear system `A x = b`.
pub fn solve_real(
    matrix: &CsrMatrix<f64>,
    rhs: &[f64],
    config: &LinearSolveConfig,
) -> Result<LinearSolveResult<f64>, LinearSolveError> {
    let result = solve_real_with_assumption(matrix, rhs, config, LinearSystemAssumption::General)?;
    result.validate_contract()?;
    Ok(result)
}

/// Solve through an explicitly selected backend.
///
/// CUDA supports CG with identity or AMG, and GMRES with identity, Jacobi, or AMG.
/// CUDA CG+AMG uses a stationary symmetric backend cycle for validated SPD/HPD
/// systems. Setup remains on the host; each call owns a retained device session.
/// Other CUDA requests fail before entering a Krylov loop unless the caller opts
/// into [`LinearSolveFallback::AllowCpu`], in which case the returned result
/// retains the CUDA request and explicitly reports that CPU executed it.
pub fn solve_with_backend<T>(
    matrix: &CsrMatrix<T>,
    rhs: &[T],
    config: &LinearSolveConfig,
    backend: LinearSolveBackend,
    fallback: LinearSolveFallback,
) -> Result<LinearSolveResult<T>, LinearSolveError>
where
    T: ComplexField + DeviceScalar<Real = <T as ComplexField>::Real>,
    <T as ComplexField>::Real: FromPrimitive + Sync + Send,
{
    solve_with_backend_and_assumption(
        matrix,
        rhs,
        config,
        backend,
        fallback,
        LinearSystemAssumption::General,
    )
}

/// Solve on an explicit backend with declared positive-definiteness assumptions.
///
/// Large SPD/HPD systems require an explicit assumption under the same validation
/// policy as CPU solves. Structural and finite-value checks are never bypassed.
///
/// # Errors
/// Returns validation, backend, preconditioner, or nonconvergence errors.
pub fn solve_with_backend_and_assumption<T>(
    matrix: &CsrMatrix<T>,
    rhs: &[T],
    config: &LinearSolveConfig,
    backend: LinearSolveBackend,
    fallback: LinearSolveFallback,
    assumption: LinearSystemAssumption,
) -> Result<LinearSolveResult<T>, LinearSolveError>
where
    T: ComplexField + DeviceScalar<Real = <T as ComplexField>::Real>,
    <T as ComplexField>::Real: FromPrimitive + Sync + Send,
{
    match backend {
        LinearSolveBackend::Cpu => {
            let mut result = solve_with_assumption(matrix, rhs, config, assumption)?;
            result.backend = LinearSolveBackendProvenance::cpu();
            result.validate_contract()?;
            Ok(result)
        }
        LinearSolveBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                match solve_cuda_backend(matrix, rhs, config, assumption) {
                    Ok(mut result) => {
                        result.backend =
                            LinearSolveBackendProvenance::native(LinearSolveBackend::Cuda);
                        result.validate_contract()?;
                        Ok(result)
                    }
                    Err(error @ LinearSolveError::BackendUnavailable { .. })
                        if fallback == LinearSolveFallback::AllowCpu =>
                    {
                        let mut result = solve_with_assumption(matrix, rhs, config, assumption)?;
                        result.backend =
                            LinearSolveBackendProvenance::cpu_fallback(LinearSolveBackend::Cuda);
                        let _ = error;
                        result.validate_contract()?;
                        Ok(result)
                    }
                    Err(error) => Err(error),
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                if fallback == LinearSolveFallback::AllowCpu {
                    let mut result = solve_with_assumption(matrix, rhs, config, assumption)?;
                    result.backend =
                        LinearSolveBackendProvenance::cpu_fallback(LinearSolveBackend::Cuda);
                    result.validate_contract()?;
                    Ok(result)
                } else {
                    Err(LinearSolveError::BackendUnavailable {
                        backend: LinearSolveBackend::Cuda,
                        reason: "sonium-solvers was built without the cuda feature".into(),
                    })
                }
            }
        }
    }
}

/// Explicit-backend real solve convenience wrapper.
pub fn solve_real_with_backend(
    matrix: &CsrMatrix<f64>,
    rhs: &[f64],
    config: &LinearSolveConfig,
    backend: LinearSolveBackend,
    fallback: LinearSolveFallback,
) -> Result<LinearSolveResult<f64>, LinearSolveError> {
    solve_with_backend(matrix, rhs, config, backend, fallback)
}

/// Explicit-backend Complex64 solve convenience wrapper.
pub fn solve_complex_with_backend(
    matrix: &CsrMatrix<Complex64>,
    rhs: &[Complex64],
    config: &LinearSolveConfig,
    backend: LinearSolveBackend,
    fallback: LinearSolveFallback,
) -> Result<LinearSolveResult<Complex64>, LinearSolveError> {
    solve_with_backend(matrix, rhs, config, backend, fallback)
}

#[cfg(feature = "cuda")]
fn solve_cuda_backend<T>(
    matrix: &CsrMatrix<T>,
    rhs: &[T],
    config: &LinearSolveConfig,
    assumption: LinearSystemAssumption,
) -> Result<LinearSolveResult<T>, LinearSolveError>
where
    T: ComplexField + DeviceScalar<Real = <T as ComplexField>::Real>,
    <T as ComplexField>::Real: FromPrimitive + Sync + Send,
{
    cuda_session::validate_strategy(config)?;
    validate_linear_inputs(matrix, rhs, None, config)?;
    if matches!(
        config.solver,
        LinearSolverType::Cg
            | LinearSolverType::CgAmg
            | LinearSolverType::GmresAmg
            | LinearSolverType::GmresChebyshev
    ) {
        validate_spd_system(matrix, config.solver, assumption)?;
    }
    let backend = CudaBackend::new(0).map_err(|error| LinearSolveError::BackendUnavailable {
        backend: LinearSolveBackend::Cuda,
        reason: format!("CUDA initialization failed: {error}"),
    })?;
    let pool = MemoryPool::new(backend, MemoryPoolConfig::default()).map_err(|error| {
        LinearSolveError::BackendFailure {
            backend: LinearSolveBackend::Cuda,
            reason: format!("CUDA memory-pool initialization failed: {error}"),
        }
    })?;
    CudaLinearSolveSession::from_validated(&pool, matrix, config, assumption)?
        .solve_host(rhs, DeviceInitialGuess::Zero)
}

/// Solve a real linear system with an explicit SPD/HPD assumption.
///
/// The assumption is only relevant for large systems and is never used to
/// bypass structural or finite-value validation.  Without it, explicit CG,
/// AMG, and Chebyshev dispatches fail closed once exact definiteness testing
/// would exceed the bounded validation budget.
pub fn solve_real_with_assumption(
    matrix: &CsrMatrix<f64>,
    rhs: &[f64],
    config: &LinearSolveConfig,
    assumption: LinearSystemAssumption,
) -> Result<LinearSolveResult<f64>, LinearSolveError> {
    validate_linear_inputs(matrix, rhs, None, config)?;
    let gmres_config = gmres_config_for_solve(config)?;

    let solver = config.resolved_solver_for_real(matrix);
    let operator = matrix.spmv_plan(spmv_plan_config_for_solve(config));
    let gmres_variant = gmres_variant_for_dispatch(config.solver, matrix, config);
    let result = match solver {
        LinearSolverType::Auto => unreachable!("auto solver should be resolved before dispatch"),
        LinearSolverType::Gmres => {
            let r = run_gmres_dispatch(
                &operator,
                &IdentityPreconditioner,
                rhs,
                None,
                &gmres_config,
                gmres_variant,
            );
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::GmresJacobi => {
            let precond = try_diagonal_preconditioner(matrix)?;
            let r =
                run_gmres_dispatch(&operator, &precond, rhs, None, &gmres_config, gmres_variant);
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::GmresIlu => {
            let precond = IluPreconditioner::try_from_csr(matrix)
                .map_err(|error| LinearSolveError::PreconditionerFailed(error.to_string()))?;
            let r =
                run_gmres_dispatch(&operator, &precond, rhs, None, &gmres_config, gmres_variant);
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::GmresAmg => {
            validate_amg_system(matrix, LinearSolverType::GmresAmg, assumption)?;
            let amg_config = amg_config_for_dispatch(config.solver);
            let precond = try_amg_preconditioner(matrix, amg_config)?;
            let r =
                run_gmres_dispatch(&operator, &precond, rhs, None, &gmres_config, gmres_variant);
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::GmresChebyshev => {
            validate_spd_system(matrix, LinearSolverType::GmresChebyshev, assumption)?;
            let precond = try_chebyshev_preconditioner(matrix, ChebyshevConfig::default())?;
            let r =
                run_gmres_dispatch(&operator, &precond, rhs, None, &gmres_config, gmres_variant);
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::GmresSchwarz => {
            let precond = try_schwarz_preconditioner(matrix, 1)?;
            let r =
                run_gmres_dispatch(&operator, &precond, rhs, None, &gmres_config, gmres_variant);
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::FgmresSchwarz => {
            run_fgmres_schwarz(matrix, &operator, rhs, None, config)?
        }
        LinearSolverType::Cg => {
            validate_spd_system(matrix, LinearSolverType::Cg, assumption)?;
            let cg_config = crate::CgConfig {
                max_iterations: config.max_iterations,
                tolerance: config.tolerance,
                ..Default::default()
            };
            let r = crate::cg(&operator, rhs, &cg_config);
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::CgAmg => {
            validate_spd_system(matrix, LinearSolverType::CgAmg, assumption)?;
            let cg_config = crate::CgConfig {
                max_iterations: config.max_iterations,
                tolerance: config.tolerance,
                ..Default::default()
            };
            let amg_config = amg_config_for_dispatch(config.solver);
            let precond = try_amg_preconditioner(matrix, amg_config)?;
            let r = crate::pcg(&operator, &precond, rhs, &cg_config);
            (r.x, r.iterations, r.residual, r.converged)
        }
    };

    let (x, iterations, _estimated_residual, _solver_converged) = result;
    let residual_f64 = true_relative_residual(matrix, rhs, &x)?;
    let converged = residual_f64 <= config.tolerance;

    if !converged {
        return Err(LinearSolveError::NotConverged {
            iterations,
            residual: residual_f64,
        });
    }

    let result = LinearSolveResult {
        backward_error: true_backward_error(matrix, rhs, &x)?,
        x,
        iterations,
        residual: residual_f64,
        converged,
        status: if converged {
            SolverStatus::Converged
        } else {
            SolverStatus::MaxIterationsReached
        },
        attainable_accuracy: if converged {
            AccuracyStatus::Attained
        } else {
            AccuracyStatus::NotAttained
        },
        assumption,
        assumption_provenance: assumption_provenance(matrix, solver, assumption),
        backend: LinearSolveBackendProvenance::cpu(),
        saddle_assumptions: None,
    };
    result.validate_contract()?;
    Ok(result)
}

/// Solve a complex linear system `A x = b`.
pub fn solve_complex(
    matrix: &CsrMatrix<Complex64>,
    rhs: &[Complex64],
    config: &LinearSolveConfig,
) -> Result<LinearSolveResult<Complex64>, LinearSolveError> {
    validate_linear_inputs(matrix, rhs, None, config)?;
    let gmres_config = gmres_config_for_solve(config)?;

    let solver = config.resolved_solver_for_complex(matrix);
    let assumption = LinearSystemAssumption::General;
    let operator = matrix.spmv_plan(spmv_plan_config_for_solve(config));
    let gmres_variant = gmres_variant_for_dispatch(config.solver, matrix, config);
    let result = match solver {
        LinearSolverType::Auto => unreachable!("auto solver should be resolved before dispatch"),
        LinearSolverType::Gmres => {
            let r = run_gmres_dispatch(
                &operator,
                &IdentityPreconditioner,
                rhs,
                None,
                &gmres_config,
                gmres_variant,
            );
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::GmresIlu => {
            let precond = IluPreconditioner::try_from_csr(matrix)
                .map_err(|error| LinearSolveError::PreconditionerFailed(error.to_string()))?;
            let r =
                run_gmres_dispatch(&operator, &precond, rhs, None, &gmres_config, gmres_variant);
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::GmresJacobi => {
            let precond = try_diagonal_preconditioner(matrix)?;
            let r =
                run_gmres_dispatch(&operator, &precond, rhs, None, &gmres_config, gmres_variant);
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::GmresSchwarz => {
            let precond = try_schwarz_preconditioner(matrix, 1)?;
            let r =
                run_gmres_dispatch(&operator, &precond, rhs, None, &gmres_config, gmres_variant);
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::FgmresSchwarz => {
            run_fgmres_schwarz(matrix, &operator, rhs, None, config)?
        }
        _ => {
            return Err(LinearSolveError::InvalidConfig(format!(
                "{:?} not supported for complex systems",
                solver
            )));
        }
    };

    let (x, iterations, _estimated_residual, _solver_converged) = result;
    let residual_f64 = true_relative_residual(matrix, rhs, &x)?;
    let converged = residual_f64 <= config.tolerance;

    if !converged {
        return Err(LinearSolveError::NotConverged {
            iterations,
            residual: residual_f64,
        });
    }

    let result = LinearSolveResult {
        backward_error: true_backward_error(matrix, rhs, &x)?,
        x,
        iterations,
        residual: residual_f64,
        converged,
        status: if converged {
            SolverStatus::Converged
        } else {
            SolverStatus::MaxIterationsReached
        },
        attainable_accuracy: if converged {
            AccuracyStatus::Attained
        } else {
            AccuracyStatus::NotAttained
        },
        assumption,
        assumption_provenance: assumption_provenance(matrix, solver, assumption),
        backend: LinearSolveBackendProvenance::cpu(),
        saddle_assumptions: None,
    };
    result.validate_contract()?;
    Ok(result)
}

/// Solve a real linear system with an initial guess.
pub fn solve_real_with_guess(
    matrix: &CsrMatrix<f64>,
    rhs: &[f64],
    x0: &[f64],
    config: &LinearSolveConfig,
) -> Result<LinearSolveResult<f64>, LinearSolveError> {
    solve_real_with_guess_and_assumption(matrix, rhs, x0, config, LinearSystemAssumption::General)
}

/// Solve a real linear system with an initial guess and an explicit SPD/HPD
/// assumption.  See [`solve_real_with_assumption`] for the validation rules.
pub fn solve_real_with_guess_and_assumption(
    matrix: &CsrMatrix<f64>,
    rhs: &[f64],
    x0: &[f64],
    config: &LinearSolveConfig,
    assumption: LinearSystemAssumption,
) -> Result<LinearSolveResult<f64>, LinearSolveError> {
    validate_linear_inputs(matrix, rhs, Some(x0), config)?;
    let gmres_config = gmres_config_for_solve(config)?;

    let solver = config.resolved_solver_for_real(matrix);
    let operator = matrix.spmv_plan(spmv_plan_config_for_solve(config));
    let gmres_variant = gmres_variant_for_dispatch(config.solver, matrix, config);
    let result = match solver {
        LinearSolverType::Auto => unreachable!("auto solver should be resolved before dispatch"),
        LinearSolverType::Gmres => {
            let r = run_gmres_dispatch(
                &operator,
                &IdentityPreconditioner,
                rhs,
                Some(x0),
                &gmres_config,
                gmres_variant,
            );
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::GmresIlu => {
            let precond = IluPreconditioner::try_from_csr(matrix)
                .map_err(|error| LinearSolveError::PreconditionerFailed(error.to_string()))?;
            let r = run_gmres_dispatch(
                &operator,
                &precond,
                rhs,
                Some(x0),
                &gmres_config,
                gmres_variant,
            );
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::GmresJacobi => {
            let precond = try_diagonal_preconditioner(matrix)?;
            let r = run_gmres_dispatch(
                &operator,
                &precond,
                rhs,
                Some(x0),
                &gmres_config,
                gmres_variant,
            );
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::GmresAmg => {
            validate_amg_system(matrix, LinearSolverType::GmresAmg, assumption)?;
            let amg_config = amg_config_for_dispatch(config.solver);
            let precond = try_amg_preconditioner(matrix, amg_config)?;
            let r = run_gmres_dispatch(
                &operator,
                &precond,
                rhs,
                Some(x0),
                &gmres_config,
                gmres_variant,
            );
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::GmresChebyshev => {
            validate_spd_system(matrix, LinearSolverType::GmresChebyshev, assumption)?;
            let precond = try_chebyshev_preconditioner(matrix, ChebyshevConfig::default())?;
            let r = run_gmres_dispatch(
                &operator,
                &precond,
                rhs,
                Some(x0),
                &gmres_config,
                gmres_variant,
            );
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::GmresSchwarz => {
            let precond = try_schwarz_preconditioner(matrix, 1)?;
            let r = run_gmres_dispatch(
                &operator,
                &precond,
                rhs,
                Some(x0),
                &gmres_config,
                gmres_variant,
            );
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::FgmresSchwarz => {
            run_fgmres_schwarz(matrix, &operator, rhs, Some(x0), config)?
        }
        LinearSolverType::Cg => {
            validate_spd_system(matrix, LinearSolverType::Cg, assumption)?;
            let cg_config = crate::CgConfig {
                max_iterations: config.max_iterations,
                tolerance: config.tolerance,
                ..Default::default()
            };
            let r = crate::cg_with_guess(&operator, rhs, Some(x0), &cg_config);
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::CgAmg => {
            validate_spd_system(matrix, LinearSolverType::CgAmg, assumption)?;
            let amg_config = amg_config_for_dispatch(config.solver);
            let precond = try_amg_preconditioner(matrix, amg_config)?;
            let cg_config = crate::CgConfig {
                max_iterations: config.max_iterations,
                tolerance: config.tolerance,
                ..Default::default()
            };
            let r = crate::pcg_with_guess(&operator, &precond, rhs, Some(x0), &cg_config);
            (r.x, r.iterations, r.residual, r.converged)
        }
    };

    let (x, iterations, _estimated_residual, _solver_converged) = result;
    let residual_f64 = true_relative_residual(matrix, rhs, &x)?;
    let converged = residual_f64 <= config.tolerance;

    if !converged {
        return Err(LinearSolveError::NotConverged {
            iterations,
            residual: residual_f64,
        });
    }

    let result = LinearSolveResult {
        backward_error: true_backward_error(matrix, rhs, &x)?,
        x,
        iterations,
        residual: residual_f64,
        converged,
        status: if converged {
            SolverStatus::Converged
        } else {
            SolverStatus::MaxIterationsReached
        },
        attainable_accuracy: if converged {
            AccuracyStatus::Attained
        } else {
            AccuracyStatus::NotAttained
        },
        assumption,
        assumption_provenance: assumption_provenance(matrix, solver, assumption),
        backend: LinearSolveBackendProvenance::cpu(),
        saddle_assumptions: None,
    };
    result.validate_contract()?;
    Ok(result)
}

/// Generic solve: works with any `ComplexField` type (f32, f64, Complex32, Complex64).
///
/// Supports GMRES, GMRES+Jacobi, GMRES+ILU, GMRES+AMG, GMRES+Chebyshev,
/// GMRES+Schwarz, FGMRES+Schwarz, CG, and CG+AMG solver strategies. CG and
/// AMG-based strategies require the system to be SPD (or Hermitian
/// positive-definite for complex types); requesting them for a system that
/// violates the necessary diagonal conditions returns
/// [`LinearSolveError::InvalidConfig`].
///
/// # Helmholtz guidance
///
/// For large complex shifted-Laplacian (Helmholtz FEM) systems the prescribed
/// pairing is a *flexible* outer iteration with a Schwarz/shifted-Laplacian
/// preconditioner — [`LinearSolverType::FgmresSchwarz`] — because standard
/// GMRES is only correct while the preconditioner is stationary. `Auto`
/// therefore routes large complex systems to FGMRES+Schwarz. Explicit
/// [`LinearSolverType::GmresSchwarz`] keeps the previous fixed-preconditioner
/// behavior for callers who know their Schwarz apply is stationary.
pub fn solve<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    rhs: &[T],
    config: &LinearSolveConfig,
) -> Result<LinearSolveResult<T>, LinearSolveError>
where
    T::Real: FromPrimitive + Sync + Send,
{
    solve_with_assumption(matrix, rhs, config, LinearSystemAssumption::General)
}

/// Generic solve with an explicit SPD/HPD assumption for large systems.
pub fn solve_with_assumption<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    rhs: &[T],
    config: &LinearSolveConfig,
    assumption: LinearSystemAssumption,
) -> Result<LinearSolveResult<T>, LinearSolveError>
where
    T::Real: FromPrimitive + Sync + Send,
{
    validate_linear_inputs(matrix, rhs, None, config)?;
    let tol = T::Real::from_f64(config.tolerance).ok_or_else(|| {
        LinearSolveError::InvalidConfig("tolerance is not representable by the scalar type".into())
    })?;
    if !tol.is_finite() || tol <= T::Real::zero() {
        return Err(LinearSolveError::InvalidConfig(
            "tolerance is outside the representable scalar range".into(),
        ));
    }
    let gmres_config = gmres_config_for_solve(config)?;
    let cg_config = || crate::CgConfig {
        max_iterations: config.max_iterations,
        tolerance: tol,
        print_interval: 0,
    };

    let solver = resolve_auto_solver(config.solver, matrix, is_complex_scalar::<T>());
    let operator = matrix.spmv_plan(spmv_plan_config_for_solve(config));
    let gmres_variant = gmres_variant_for_dispatch(config.solver, matrix, config);
    let result = match solver {
        LinearSolverType::Auto => unreachable!("auto solver should be resolved before dispatch"),
        LinearSolverType::Gmres => {
            let r = run_gmres_dispatch(
                &operator,
                &IdentityPreconditioner,
                rhs,
                None,
                &gmres_config,
                gmres_variant,
            );
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::GmresJacobi => {
            let precond = try_diagonal_preconditioner(matrix)?;
            let r =
                run_gmres_dispatch(&operator, &precond, rhs, None, &gmres_config, gmres_variant);
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::GmresIlu => {
            let precond = IluPreconditioner::try_from_csr(matrix)
                .map_err(|error| LinearSolveError::PreconditionerFailed(error.to_string()))?;
            let r =
                run_gmres_dispatch(&operator, &precond, rhs, None, &gmres_config, gmres_variant);
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::GmresAmg => {
            // AMG requires SPD/HPD structure; reject complex systems outright
            // instead of silently returning garbage.
            validate_amg_system(matrix, LinearSolverType::GmresAmg, assumption)?;
            let amg_config = amg_config_for_dispatch(config.solver);
            let precond = try_amg_preconditioner(matrix, amg_config)?;
            let r =
                run_gmres_dispatch(&operator, &precond, rhs, None, &gmres_config, gmres_variant);
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::GmresChebyshev => {
            validate_spd_system(matrix, LinearSolverType::GmresChebyshev, assumption)?;
            let precond = try_chebyshev_preconditioner(matrix, ChebyshevConfig::default())?;
            let r =
                run_gmres_dispatch(&operator, &precond, rhs, None, &gmres_config, gmres_variant);
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::GmresSchwarz => {
            let precond = try_schwarz_preconditioner(matrix, 1)?;
            let r =
                run_gmres_dispatch(&operator, &precond, rhs, None, &gmres_config, gmres_variant);
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::FgmresSchwarz => {
            run_fgmres_schwarz(matrix, &operator, rhs, None, config)?
        }
        LinearSolverType::Cg => {
            validate_spd_system(matrix, LinearSolverType::Cg, assumption)?;
            let r = crate::cg(&operator, rhs, &cg_config());
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::CgAmg => {
            validate_amg_system(matrix, LinearSolverType::CgAmg, assumption)?;
            let amg_config = amg_config_for_dispatch(config.solver);
            let precond = try_amg_preconditioner(matrix, amg_config)?;
            let r = crate::pcg(&operator, &precond, rhs, &cg_config());
            (r.x, r.iterations, r.residual, r.converged)
        }
    };

    let (x, iterations, _estimated_residual, _solver_converged) = result;
    let residual_f64 = true_relative_residual(matrix, rhs, &x)?;
    let converged = residual_f64 <= config.tolerance;

    if !converged {
        return Err(LinearSolveError::NotConverged {
            iterations,
            residual: residual_f64,
        });
    }

    let result = LinearSolveResult {
        backward_error: true_backward_error(matrix, rhs, &x)?,
        x,
        iterations,
        residual: residual_f64,
        converged,
        status: if converged {
            SolverStatus::Converged
        } else {
            SolverStatus::MaxIterationsReached
        },
        attainable_accuracy: if converged {
            AccuracyStatus::Attained
        } else {
            AccuracyStatus::NotAttained
        },
        assumption,
        assumption_provenance: assumption_provenance(matrix, solver, assumption),
        backend: LinearSolveBackendProvenance::cpu(),
        saddle_assumptions: None,
    };
    result.validate_contract()?;
    Ok(result)
}

/// Generic solve with initial guess.
pub fn solve_with_guess<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    rhs: &[T],
    x0: &[T],
    config: &LinearSolveConfig,
) -> Result<LinearSolveResult<T>, LinearSolveError>
where
    T::Real: FromPrimitive + Sync + Send,
{
    solve_with_guess_and_assumption(matrix, rhs, x0, config, LinearSystemAssumption::General)
}

/// Generic solve with an initial guess and an explicit SPD/HPD assumption for
/// large systems.
pub fn solve_with_guess_and_assumption<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    rhs: &[T],
    x0: &[T],
    config: &LinearSolveConfig,
    assumption: LinearSystemAssumption,
) -> Result<LinearSolveResult<T>, LinearSolveError>
where
    T::Real: FromPrimitive + Sync + Send,
{
    validate_linear_inputs(matrix, rhs, Some(x0), config)?;
    let tol = T::Real::from_f64(config.tolerance).ok_or_else(|| {
        LinearSolveError::InvalidConfig("tolerance is not representable by the scalar type".into())
    })?;
    if !tol.is_finite() || tol <= T::Real::zero() {
        return Err(LinearSolveError::InvalidConfig(
            "tolerance is outside the representable scalar range".into(),
        ));
    }
    let gmres_config = gmres_config_for_solve(config)?;
    let cg_config = || crate::CgConfig {
        max_iterations: config.max_iterations,
        tolerance: tol,
        print_interval: 0,
    };

    let solver = resolve_auto_solver(config.solver, matrix, is_complex_scalar::<T>());
    let operator = matrix.spmv_plan(spmv_plan_config_for_solve(config));
    let gmres_variant = gmres_variant_for_dispatch(config.solver, matrix, config);
    let result = match solver {
        LinearSolverType::Auto => unreachable!("auto solver should be resolved before dispatch"),
        LinearSolverType::Gmres => {
            let r = run_gmres_dispatch(
                &operator,
                &IdentityPreconditioner,
                rhs,
                Some(x0),
                &gmres_config,
                gmres_variant,
            );
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::GmresIlu => {
            let precond = IluPreconditioner::try_from_csr(matrix)
                .map_err(|error| LinearSolveError::PreconditionerFailed(error.to_string()))?;
            let r = run_gmres_dispatch(
                &operator,
                &precond,
                rhs,
                Some(x0),
                &gmres_config,
                gmres_variant,
            );
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::GmresJacobi => {
            let precond = try_diagonal_preconditioner(matrix)?;
            let r = run_gmres_dispatch(
                &operator,
                &precond,
                rhs,
                Some(x0),
                &gmres_config,
                gmres_variant,
            );
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::GmresAmg => {
            validate_amg_system(matrix, LinearSolverType::GmresAmg, assumption)?;
            let amg_config = amg_config_for_dispatch(config.solver);
            let precond = try_amg_preconditioner(matrix, amg_config)?;
            let r = run_gmres_dispatch(
                &operator,
                &precond,
                rhs,
                Some(x0),
                &gmres_config,
                gmres_variant,
            );
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::GmresChebyshev => {
            validate_spd_system(matrix, LinearSolverType::GmresChebyshev, assumption)?;
            let precond = try_chebyshev_preconditioner(matrix, ChebyshevConfig::default())?;
            let r = run_gmres_dispatch(
                &operator,
                &precond,
                rhs,
                Some(x0),
                &gmres_config,
                gmres_variant,
            );
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::GmresSchwarz => {
            let precond = try_schwarz_preconditioner(matrix, 1)?;
            let r = run_gmres_dispatch(
                &operator,
                &precond,
                rhs,
                Some(x0),
                &gmres_config,
                gmres_variant,
            );
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::FgmresSchwarz => {
            run_fgmres_schwarz(matrix, &operator, rhs, Some(x0), config)?
        }
        LinearSolverType::Cg => {
            validate_spd_system(matrix, LinearSolverType::Cg, assumption)?;
            let r = crate::cg_with_guess(&operator, rhs, Some(x0), &cg_config());
            (r.x, r.iterations, r.residual, r.converged)
        }
        LinearSolverType::CgAmg => {
            validate_amg_system(matrix, LinearSolverType::CgAmg, assumption)?;
            let amg_config = amg_config_for_dispatch(config.solver);
            let precond = try_amg_preconditioner(matrix, amg_config)?;
            let r = crate::pcg_with_guess(&operator, &precond, rhs, Some(x0), &cg_config());
            (r.x, r.iterations, r.residual, r.converged)
        }
    };

    let (x, iterations, _estimated_residual, _solver_converged) = result;
    let residual_f64 = true_relative_residual(matrix, rhs, &x)?;
    let converged = residual_f64 <= config.tolerance;

    if !converged {
        return Err(LinearSolveError::NotConverged {
            iterations,
            residual: residual_f64,
        });
    }

    let result = LinearSolveResult {
        backward_error: true_backward_error(matrix, rhs, &x)?,
        x,
        iterations,
        residual: residual_f64,
        converged,
        status: if converged {
            SolverStatus::Converged
        } else {
            SolverStatus::MaxIterationsReached
        },
        attainable_accuracy: if converged {
            AccuracyStatus::Attained
        } else {
            AccuracyStatus::NotAttained
        },
        assumption,
        assumption_provenance: assumption_provenance(matrix, solver, assumption),
        backend: LinearSolveBackendProvenance::cpu(),
        saddle_assumptions: None,
    };
    result.validate_contract()?;
    Ok(result)
}

/// Solve a 2x2 block saddle-point/field-split system with a dedicated
/// preconditioner.
///
/// Use `saddle_config.block1_size` to specify the leading block size.
/// For example, in Stokes systems this is the velocity block and the
/// trailing block is pressure.
pub fn solve_saddle_point<T: ComplexField + Send + Sync>(
    matrix: &CsrMatrix<T>,
    rhs: &[T],
    config: &LinearSolveConfig,
    saddle_config: SaddlePointConfig,
    schur_approx: Option<&CsrMatrix<T>>,
) -> Result<LinearSolveResult<T>, LinearSolveError>
where
    T::Real: FromPrimitive + Sync + Send,
{
    solve_saddle_point_with_schur_and_guess(
        matrix,
        rhs,
        None,
        config,
        saddle_config,
        SaddlePointBlockAssumptions::default(),
        schur_approx,
    )
}

/// Solve a saddle-point system with explicit SPD/HPD assumptions for the
/// blocks used by AMG preconditioners. Small blocks are independently
/// validated; larger blocks require `PositiveDefinite` declarations.
pub fn solve_saddle_point_with_assumptions<T: ComplexField + Send + Sync>(
    matrix: &CsrMatrix<T>,
    rhs: &[T],
    config: &LinearSolveConfig,
    saddle_config: SaddlePointConfig,
    assumptions: SaddlePointBlockAssumptions,
    schur_approx: Option<&CsrMatrix<T>>,
) -> Result<LinearSolveResult<T>, LinearSolveError>
where
    T::Real: FromPrimitive + Sync + Send,
{
    solve_saddle_point_with_schur_and_guess(
        matrix,
        rhs,
        None,
        config,
        saddle_config,
        assumptions,
        schur_approx,
    )
}

/// Solve a 2x2 block saddle-point/field-split system with an initial guess.
///
/// This routes through GMRES with the same configuration knobs as `solve_real`
/// / `solve_complex`, but uses a field-split preconditioner instead of scalar
/// AMG/ILU/Jacobi choices.
pub fn solve_saddle_point_with_guess<T: ComplexField + Send + Sync>(
    matrix: &CsrMatrix<T>,
    rhs: &[T],
    x0: &[T],
    config: &LinearSolveConfig,
    saddle_config: SaddlePointConfig,
    schur_approx: Option<&CsrMatrix<T>>,
) -> Result<LinearSolveResult<T>, LinearSolveError>
where
    T::Real: FromPrimitive + Sync + Send,
{
    solve_saddle_point_with_schur_and_guess(
        matrix,
        rhs,
        Some(x0),
        config,
        saddle_config,
        SaddlePointBlockAssumptions::default(),
        schur_approx,
    )
}

/// Solve a saddle-point system with an initial guess and explicit SPD/HPD
/// assumptions for the blocks used by AMG preconditioners.
pub fn solve_saddle_point_with_guess_and_assumptions<T: ComplexField + Send + Sync>(
    matrix: &CsrMatrix<T>,
    rhs: &[T],
    x0: &[T],
    config: &LinearSolveConfig,
    saddle_config: SaddlePointConfig,
    assumptions: SaddlePointBlockAssumptions,
    schur_approx: Option<&CsrMatrix<T>>,
) -> Result<LinearSolveResult<T>, LinearSolveError>
where
    T::Real: FromPrimitive + Sync + Send,
{
    solve_saddle_point_with_schur_and_guess(
        matrix,
        rhs,
        Some(x0),
        config,
        saddle_config,
        assumptions,
        schur_approx,
    )
}

fn solve_saddle_point_with_schur_and_guess<T: ComplexField + Send + Sync>(
    matrix: &CsrMatrix<T>,
    rhs: &[T],
    x0: Option<&[T]>,
    config: &LinearSolveConfig,
    saddle_config: SaddlePointConfig,
    assumptions: SaddlePointBlockAssumptions,
    schur_approx: Option<&CsrMatrix<T>>,
) -> Result<LinearSolveResult<T>, LinearSolveError>
where
    T::Real: FromPrimitive + Sync + Send,
{
    config.validate()?;
    if matches!(
        config.solver,
        LinearSolverType::Cg | LinearSolverType::CgAmg
    ) {
        return Err(LinearSolveError::InvalidConfig(format!(
            "solver {:?} is not supported for saddle-point solves; use a GMRES variant",
            config.solver
        )));
    }
    validate_saddle_point_inputs(matrix, rhs, x0, saddle_config.block1_size, schur_approx)?;
    if saddle_config.schur_strategy == SchurStrategy::External && schur_approx.is_none() {
        return Err(LinearSolveError::InvalidConfig(
            "saddle-point solve with SchurStrategy::External requires schur_approx".to_string(),
        ));
    }

    let block_validation = if matches!(
        saddle_config.schur_strategy,
        SchurStrategy::Pcd | SchurStrategy::Lsc
    ) {
        None
    } else {
        Some(
            SaddlePointPreconditioner::validate_checked_block_assumptions(
                matrix,
                &saddle_config,
                assumptions,
                schur_approx,
            )
            .map_err(|error| LinearSolveError::PreconditionerFailed(error.to_string()))?,
        )
    };

    let preconditioner = match saddle_config.schur_strategy {
        SchurStrategy::External => {
            let schur_approx = schur_approx.ok_or_else(|| {
                LinearSolveError::InvalidConfig(
                    "saddle-point solve with SchurStrategy::External requires schur_approx"
                        .to_string(),
                )
            })?;
            SaddlePointPreconditioner::try_from_matrix_with_schur_and_assumptions(
                matrix,
                schur_approx,
                saddle_config,
                assumptions,
            )
            .map_err(|error| LinearSolveError::PreconditionerFailed(error.to_string()))?
        }
        SchurStrategy::Pcd | SchurStrategy::Lsc => {
            return Err(LinearSolveError::InvalidConfig(
                "saddle-point PCD/LSC requires pressure-space or velocity-mass blocks; use the specialized preconditioner constructor"
                    .to_string(),
            ));
        }
        _ => SaddlePointPreconditioner::try_from_matrix_with_assumptions(
            matrix,
            saddle_config,
            assumptions,
        )
        .map_err(|error| LinearSolveError::PreconditionerFailed(error.to_string()))?,
    };

    let gmres_config = gmres_config_for_solve(config)?;
    let operator = matrix.spmv_plan(spmv_plan_config_for_solve(config));
    let gmres_variant = gmres_variant_for_dispatch(config.solver, matrix, config);
    let r = run_gmres_dispatch(
        &operator,
        &preconditioner,
        rhs,
        x0,
        &gmres_config,
        gmres_variant,
    );

    let residual_f64 = true_relative_residual(matrix, rhs, &r.x)?;
    let converged = residual_f64 <= config.tolerance;

    if !converged {
        return Err(LinearSolveError::NotConverged {
            iterations: r.iterations,
            residual: residual_f64,
        });
    }

    let block_validation = block_validation.ok_or_else(|| {
        LinearSolveError::InvalidConfig(
            "specialized saddle-point strategies require specialized block metadata".into(),
        )
    })?;
    let result = LinearSolveResult {
        backward_error: true_backward_error(matrix, rhs, &r.x)?,
        x: r.x,
        iterations: r.iterations,
        residual: residual_f64,
        converged,
        status: if converged {
            SolverStatus::Converged
        } else {
            SolverStatus::MaxIterationsReached
        },
        attainable_accuracy: if converged {
            AccuracyStatus::Attained
        } else {
            AccuracyStatus::NotAttained
        },
        assumption: LinearSystemAssumption::General,
        assumption_provenance: AssumptionProvenance::NotRequired,
        backend: LinearSolveBackendProvenance::cpu(),
        saddle_assumptions: Some(SaddlePointAssumptionReport {
            block1: assumptions.block1,
            block1_provenance: map_saddle_block_provenance(block_validation.block1),
            block2: assumptions.block2,
            block2_provenance: map_saddle_block_provenance(block_validation.block2),
        }),
    };
    result.validate_contract()?;
    Ok(result)
}

fn map_saddle_block_provenance(provenance: SaddlePointBlockProvenance) -> AssumptionProvenance {
    match provenance {
        SaddlePointBlockProvenance::NotRequired => AssumptionProvenance::NotRequired,
        SaddlePointBlockProvenance::IndependentlyValidated => {
            AssumptionProvenance::IndependentlyValidated
        }
        SaddlePointBlockProvenance::CallerDeclared => AssumptionProvenance::CallerDeclared,
    }
}

fn validate_saddle_point_inputs<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    rhs: &[T],
    x0: Option<&[T]>,
    block1_size: usize,
    schur_approx: Option<&CsrMatrix<T>>,
) -> Result<(), LinearSolveError> {
    matrix.validate_structure().map_err(|error| {
        LinearSolveError::InvalidConfig(format!("invalid saddle-point matrix structure: {error}"))
    })?;
    if matrix.num_rows != matrix.num_cols {
        return Err(LinearSolveError::DimensionMismatch {
            operation: "saddle-point matrix",
            expected: matrix.num_rows,
            actual: matrix.num_cols,
        });
    }
    if rhs.len() != matrix.num_rows {
        return Err(LinearSolveError::DimensionMismatch {
            operation: "saddle-point rhs",
            expected: matrix.num_rows,
            actual: rhs.len(),
        });
    }
    match x0 {
        Some(x0) if x0.len() != matrix.num_cols => {
            return Err(LinearSolveError::DimensionMismatch {
                operation: "saddle-point initial guess",
                expected: matrix.num_cols,
                actual: x0.len(),
            });
        }
        _ => {}
    }
    if block1_size == 0 || block1_size >= matrix.num_rows {
        return Err(LinearSolveError::InvalidConfig(format!(
            "saddle-point block1_size must split both fields, got {} for {} dofs",
            block1_size, matrix.num_rows
        )));
    }
    if let Some(schur_approx) = schur_approx {
        schur_approx.validate_structure().map_err(|error| {
            LinearSolveError::InvalidConfig(format!(
                "invalid saddle-point Schur approximation structure: {error}"
            ))
        })?;
        let block2_size = matrix.num_rows - block1_size;
        if schur_approx.num_rows != block2_size || schur_approx.num_cols != block2_size {
            return Err(LinearSolveError::DimensionMismatch {
                operation: "saddle-point Schur approximation",
                expected: block2_size,
                actual: schur_approx.num_rows.max(schur_approx.num_cols),
            });
        }
    }
    if let Some((index, _)) = matrix
        .values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.norm().is_finite())
    {
        return Err(LinearSolveError::NonFiniteInput {
            operation: "matrix",
            index,
        });
    }
    if let Some((index, _)) = rhs
        .iter()
        .enumerate()
        .find(|(_, value)| !value.norm().is_finite())
    {
        return Err(LinearSolveError::NonFiniteInput {
            operation: "rhs",
            index,
        });
    }
    if let Some(x0) = x0
        && let Some((index, _)) = x0
            .iter()
            .enumerate()
            .find(|(_, value)| !value.norm().is_finite())
    {
        return Err(LinearSolveError::NonFiniteInput {
            operation: "initial guess",
            index,
        });
    }
    Ok(())
}

fn resolve_auto_solver<T: ComplexField>(
    solver: LinearSolverType,
    matrix: &CsrMatrix<T>,
    is_complex_scalar: bool,
) -> LinearSolverType {
    if solver != LinearSolverType::Auto {
        return solver;
    }

    if is_large_problem(matrix) {
        if is_complex_scalar {
            // Helmholtz-type large complex systems: flexible outer iteration
            // so a (future) non-stationary Schwarz apply stays correct.
            LinearSolverType::FgmresSchwarz
        } else {
            // Automatic dispatch cannot prove positive definiteness for a
            // large matrix without an expensive factorization.  Use the
            // nonsymmetric-safe ILU path unless the caller explicitly opts
            // into an SPD/HPD solver type.
            LinearSolverType::GmresIlu
        }
    } else {
        LinearSolverType::GmresIlu
    }
}

fn gmres_variant_for_dispatch<T: ComplexField>(
    requested_solver: LinearSolverType,
    matrix: &CsrMatrix<T>,
    config: &LinearSolveConfig,
) -> GmresDispatchVariant {
    if requested_solver == LinearSolverType::Auto
        && is_large_problem(matrix)
        && config.restart >= 16
        && config.max_iterations >= config.restart
    {
        GmresDispatchVariant::Pipelined
    } else {
        GmresDispatchVariant::Standard
    }
}

fn run_gmres_dispatch<T, A, P>(
    operator: &A,
    precond: &P,
    rhs: &[T],
    x0: Option<&[T]>,
    gmres_config: &GmresConfig<T::Real>,
    variant: GmresDispatchVariant,
) -> GmresSolution<T>
where
    T: ComplexField,
    A: LinearOperator<T>,
    P: Preconditioner<T>,
{
    match variant {
        GmresDispatchVariant::Standard => {
            gmres_preconditioned_with_guess(operator, precond, rhs, x0, gmres_config)
        }
        GmresDispatchVariant::Pipelined => {
            gmres_pipelined(operator, precond, rhs, x0, gmres_config)
        }
    }
}

fn gmres_config_for_solve<R>(config: &LinearSolveConfig) -> Result<GmresConfig<R>, LinearSolveError>
where
    R: Float + FromPrimitive,
{
    // The Krylov residual is measured in the preconditioned space, while the
    // public solve result is accepted against the original matrix. Leave a
    // a safety decade so those two termination criteria cannot disagree merely
    // because the preconditioner rescales the residual.
    let internal_tolerance = config.tolerance * 0.1;
    let tolerance = R::from_f64(internal_tolerance).ok_or_else(|| {
        LinearSolveError::InvalidConfig("tolerance not representable scalar type".into())
    })?;
    if !tolerance.is_finite() || tolerance <= R::zero() {
        return Err(LinearSolveError::InvalidConfig(
            "tolerance is outside the representable scalar range".into(),
        ));
    }
    Ok(GmresConfig {
        max_iterations: config.max_iterations,
        tolerance,
        restart: config.restart,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: config.exact_residual_check_interval,
        min_restart: None,
        max_restart: None,
    })
}

fn fgmres_config_for_solve<R>(
    config: &LinearSolveConfig,
) -> Result<FgmresConfig<R>, LinearSolveError>
where
    R: Float + FromPrimitive,
{
    let internal_tolerance = config.tolerance * 0.1;
    let tolerance = R::from_f64(internal_tolerance).ok_or_else(|| {
        LinearSolveError::InvalidConfig("tolerance not representable scalar type".into())
    })?;
    if !tolerance.is_finite() || tolerance <= R::zero() {
        return Err(LinearSolveError::InvalidConfig(
            "tolerance is outside the representable scalar range".into(),
        ));
    }
    Ok(FgmresConfig {
        max_iterations: config.max_iterations,
        tolerance,
        restart: config.restart,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: config.exact_residual_check_interval,
    })
}

/// Run FGMRES with additive Schwarz preconditioning for the
/// [`LinearSolverType::FgmresSchwarz`] dispatch arms. Returns the same
/// `(x, iterations, residual, converged)` tuple shape as the GMRES arms,
/// wrapped in the checked high-level error result.
fn run_fgmres_schwarz<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    operator: &impl LinearOperator<T>,
    rhs: &[T],
    x0: Option<&[T]>,
    config: &LinearSolveConfig,
) -> Result<(Vec<T>, usize, T::Real, bool), LinearSolveError>
where
    T::Real: FromPrimitive,
{
    let precond = try_schwarz_preconditioner(matrix, 1)?;
    let fgmres_config = fgmres_config_for_solve(config)?;
    let r = fgmres_preconditioned(operator, &precond, rhs, x0, &fgmres_config);
    Ok((r.x, r.iterations, r.residual, r.converged))
}

fn is_large_problem<T: ComplexField>(matrix: &CsrMatrix<T>) -> bool {
    matrix.num_rows >= large_problem_rows() || matrix.nnz() >= large_problem_nnz()
}

fn is_complex_scalar<T: ComplexField>() -> bool {
    TypeId::of::<T>() == TypeId::of::<Complex32>() || TypeId::of::<T>() == TypeId::of::<Complex64>()
}

fn spmv_plan_config_for_solve(config: &LinearSolveConfig) -> SpmvPlanConfig {
    SpmvPlanConfig {
        expected_reuse: config.max_iterations.max(config.restart).max(1),
        ..Default::default()
    }
}

fn amg_config_for_dispatch(requested_solver: LinearSolverType) -> AmgConfig {
    if requested_solver == LinearSolverType::Auto {
        AmgConfig::for_muelu_sa_comparison()
    } else {
        AmgConfig::for_fem()
    }
}

/// Validate the SPD/HPD assumptions used by CG, AMG, and Chebyshev.
///
/// The structural scan is cheap and applies to every size.  For bounded-size
/// dispatches we additionally run the same pivot test as sparse Cholesky so a
/// symmetric-indefinite matrix with positive diagonal entries cannot slip
/// into a CG/AMG path.  The bound avoids turning high-level iterative solves
/// for very large PDE systems into an accidental second direct factorization.
/// Return the CSR operator with repeated column entries summed row-wise.
///
/// CSR SpMV treats duplicate columns additively, while several factorization
/// and certificate paths require canonical storage. Keeping this conversion in
/// one place prevents validation from depending on duplicate ordering.
fn coalesced_csr<T: ComplexField>(matrix: &CsrMatrix<T>) -> CsrMatrix<T> {
    let mut row_ptrs = Vec::with_capacity(matrix.num_rows + 1);
    let mut col_indices = Vec::with_capacity(matrix.nnz());
    let mut values = Vec::with_capacity(matrix.nnz());
    row_ptrs.push(0);
    for row in 0..matrix.num_rows {
        let row_output_start = col_indices.len();
        let mut entries: Vec<_> = (matrix.row_ptrs[row]..matrix.row_ptrs[row + 1])
            .map(|index| (matrix.col_indices[index], matrix.values[index]))
            .collect();
        entries.sort_unstable_by_key(|(col, _)| *col);
        for (col, value) in entries {
            if col_indices.len() > row_output_start
                && let Some(last_col) = col_indices.last().copied()
                && last_col == col
            {
                let last = values
                    .last_mut()
                    .expect("coalesced column/value storage stays aligned");
                *last += value;
            } else {
                col_indices.push(col);
                values.push(value);
            }
        }
        row_ptrs.push(col_indices.len());
    }
    CsrMatrix::from_raw_parts(
        matrix.num_rows,
        matrix.num_cols,
        row_ptrs,
        col_indices,
        values,
    )
}

fn validate_spd_system<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    solver: LinearSolverType,
    assumption: LinearSystemAssumption,
) -> Result<(), LinearSolveError> {
    let zero = T::Real::zero();
    let one = T::Real::from_f64(1.0).unwrap_or_else(T::Real::one);

    let coalesced = coalesced_csr(matrix);

    for row in 0..matrix.num_rows {
        let row_start = coalesced.row_ptrs[row];
        let row_end = coalesced.row_ptrs[row + 1];
        let diagonal = (row_start..row_end)
            .find(|&index| coalesced.col_indices[index] == row)
            .map(|index| coalesced.values[index]);
        let Some(diagonal) = diagonal else {
            return Err(LinearSolveError::InvalidConfig(format!(
                "{solver:?} requires an SPD/HPD matrix, but row {row} has no diagonal"
            )));
        };

        if !diagonal.re().is_finite() || !diagonal.im().is_finite() {
            return Err(LinearSolveError::InvalidConfig(format!(
                "{solver:?} received a non-finite diagonal entry at row {row}"
            )));
        }
        let diagonal_tol =
            T::Real::from_f64(1e-12).unwrap_or_else(T::Real::epsilon) * diagonal.norm().max(one);
        if diagonal.im().abs() > diagonal_tol || diagonal.re() <= zero {
            return Err(LinearSolveError::InvalidConfig(format!(
                "{solver:?} requires an SPD/HPD matrix, but diagonal entry {row} is not real and positive"
            )));
        }

        for index in row_start..row_end {
            let col = coalesced.col_indices[index];
            let value = coalesced.values[index];
            if !value.re().is_finite() || !value.im().is_finite() {
                return Err(LinearSolveError::InvalidConfig(format!(
                    "{solver:?} received a non-finite matrix entry at ({row},{col})"
                )));
            }
            if col >= matrix.num_rows {
                return Err(LinearSolveError::InvalidConfig(format!(
                    "{solver:?} received an out-of-range column index {col}"
                )));
            }
            let counterpart_start = coalesced.row_ptrs[col];
            let counterpart_end = coalesced.row_ptrs[col + 1];
            let counterpart = (counterpart_start..counterpart_end)
                .find(|&candidate| coalesced.col_indices[candidate] == row)
                .map(|candidate| coalesced.values[candidate]);
            let Some(counterpart) = counterpart else {
                return Err(LinearSolveError::InvalidConfig(format!(
                    "{solver:?} requires a symmetric/Hermitian sparsity pattern; ({row},{col}) has no counterpart"
                )));
            };
            let difference = value - counterpart.conj();
            let scale = value.norm().max(counterpart.norm()).max(one);
            if difference.norm() > T::Real::from_f64(1e-10).unwrap_or_else(T::Real::epsilon) * scale
            {
                return Err(LinearSolveError::InvalidConfig(format!(
                    "{solver:?} requires a symmetric/Hermitian matrix; entries ({row},{col}) and ({col},{row}) disagree"
                )));
            }
        }
    }

    if matrix.num_rows > SPD_EXACT_VALIDATION_MAX_ROWS
        && !matches!(
            assumption,
            LinearSystemAssumption::PositiveDefinite
                | LinearSystemAssumption::PositiveDefiniteCertified
        )
    {
        return Err(LinearSolveError::InvalidConfig(format!(
            "{solver:?} on a system with {} rows requires an explicit LinearSystemAssumption::PositiveDefinite",
            matrix.num_rows
        )));
    }

    if matrix.num_rows > SPD_EXACT_VALIDATION_MAX_ROWS
        && assumption == LinearSystemAssumption::PositiveDefiniteCertified
        && !has_scalable_spd_certificate(matrix)
    {
        return Err(LinearSolveError::InvalidConfig(format!(
            "{solver:?} on large system requires a scalable positive-definite certificate"
        )));
    }

    if matrix.num_rows <= SPD_EXACT_VALIDATION_MAX_ROWS {
        SparseCholeskyFactorization::factorize_hermitian(&coalesced).map_err(|error| {
            LinearSolveError::InvalidConfig(format!(
                "{solver:?} requires a positive-definite matrix: {error}"
            ))
        })?;
    }
    Ok(())
}

/// Backwards-compatible name used by callers that specifically select AMG.
fn validate_amg_system<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    solver: LinearSolverType,
    assumption: LinearSystemAssumption,
) -> Result<(), LinearSolveError> {
    validate_spd_system(matrix, solver, assumption)
}

#[cfg(test)]
mod finite_input_tests {
    use super::*;
    use half::f16;

    #[test]
    fn low_precision_tolerance_underflow_is_reported_without_panicking() {
        let config = LinearSolveConfig {
            tolerance: 1.0e-7,
            ..Default::default()
        };

        assert!(gmres_config_for_solve::<f16>(&config).is_err());
        assert!(fgmres_config_for_solve::<f16>(&config).is_err());

        let outcome = catch_unwind(AssertUnwindSafe(|| {
            solve::<f16>(&CsrMatrix::identity(1), &[f16::ONE], &config)
        }));
        let error = outcome
            .expect("low-precision tolerance conversion must not panic")
            .expect_err("f16 internal tolerance underflow must be rejected");
        assert!(matches!(error, LinearSolveError::InvalidConfig(_)));
    }

    #[test]
    fn scalable_spd_certificate_accepts_jacobi_scaled_dominance() {
        let strict = CsrMatrix::from_raw_parts(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![3.0, 1.0, 1.0, 3.0],
        );
        assert!(has_scalable_spd_certificate(&strict));
        let scaled = CsrMatrix::from_raw_parts(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![12.0, 1.0, 1.0, 0.75],
        );
        assert!(has_scalable_spd_certificate(&scaled));

        let non_strict = CsrMatrix::from_raw_parts(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0, 1.0, 1.0, 1.0],
        );
        assert!(!has_scalable_spd_certificate(&non_strict));
    }

    #[test]
    fn scalable_spd_certificate_rejects_strictly_dominant_nonsymmetric_matrix() {
        let nonsymmetric = CsrMatrix::from_raw_parts(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![3.0, 1.0, 0.25, 3.0],
        );
        assert!(!has_scalable_spd_certificate(&nonsymmetric));
    }

    #[test]
    fn rejects_non_finite_matrix_values() {
        let mut matrix = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![1.0]);
        matrix.values[0] = f64::NAN;
        let error = solve_real(&matrix, &[1.0], &LinearSolveConfig::default())
            .expect_err("NaN matrix entry must be rejected");
        assert!(matches!(
            error,
            LinearSolveError::NonFiniteInput {
                operation: "matrix",
                ..
            }
        ));
    }

    #[test]
    fn rejects_non_finite_rhs_values() {
        let matrix = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![1.0]);
        let error = solve_real(&matrix, &[f64::INFINITY], &LinearSolveConfig::default())
            .expect_err("infinite RHS must be rejected");
        assert!(matches!(
            error,
            LinearSolveError::NonFiniteInput {
                operation: "rhs",
                ..
            }
        ));
    }

    #[test]
    fn rejects_malformed_csr_structure_before_dispatch() {
        let mut matrix = CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0]);
        matrix.row_ptrs = vec![0, 2, 1];
        let error = solve_real(&matrix, &[1.0, 1.0], &LinearSolveConfig::default())
            .expect_err("non-monotone row pointers must be rejected");
        assert!(
            matches!(error, LinearSolveError::InvalidConfig(message) if message.contains("monotonically"))
        );

        let mut matrix = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![1.0]);
        matrix.col_indices[0] = 1;
        let error = solve_real(&matrix, &[1.0], &LinearSolveConfig::default())
            .expect_err("out-of-range CSR columns must be rejected");
        assert!(
            matches!(error, LinearSolveError::InvalidConfig(message) if message.contains("out of range"))
        );
    }

    #[test]
    fn explicit_backend_preserves_large_spd_assumptions() {
        let n = SPD_EXACT_VALIDATION_MAX_ROWS + 1;
        let matrix = CsrMatrix::<f64>::identity(n);
        let rhs = vec![1.0; n];
        let config = LinearSolveConfig {
            solver: LinearSolverType::Cg,
            ..Default::default()
        };
        assert!(
            solve_with_backend(
                &matrix,
                &rhs,
                &config,
                LinearSolveBackend::Cpu,
                LinearSolveFallback::Reject
            )
            .is_err()
        );
        let result = solve_with_backend_and_assumption(
            &matrix,
            &rhs,
            &config,
            LinearSolveBackend::Cpu,
            LinearSolveFallback::Reject,
            LinearSystemAssumption::PositiveDefiniteCertified,
        )
        .unwrap();
        assert_eq!(
            result.assumption,
            LinearSystemAssumption::PositiveDefiniteCertified
        );
        assert_eq!(
            result.assumption_provenance,
            AssumptionProvenance::ScalableCertificate
        );
        result.validate_contract().unwrap();
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_spd_strategies_reject_indefinite_input_before_device_setup() {
        let matrix = CsrMatrix::from_raw_parts(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0_f64, 2.0, 2.0, 1.0],
        );
        for solver in [
            LinearSolverType::Cg,
            LinearSolverType::CgAmg,
            LinearSolverType::GmresAmg,
        ] {
            let config = LinearSolveConfig {
                solver,
                ..Default::default()
            };
            let error = solve_cuda_backend(
                &matrix,
                &[1.0, 1.0],
                &config,
                LinearSystemAssumption::PositiveDefinite,
            )
            .unwrap_err();
            assert!(
                matches!(error, LinearSolveError::InvalidConfig(_)),
                "{error}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_jacobi_facade_certifies_original_system() {
        if let Err(error) = CudaBackend::new(0) {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                "CUDA is required: {error}"
            );
            eprintln!("SKIP cuda_jacobi_facade_certifies_original_system: {error}");
            return;
        }
        // Duplicate diagonal entries must add before Jacobi setup.
        let matrix = CsrMatrix::from_raw_parts(
            2,
            2,
            vec![0, 3, 5],
            vec![0, 0, 1, 0, 1],
            vec![2.0_f64, 2.0, 1.0, 1.0, 3.0],
        );
        let mut config = LinearSolveConfig {
            solver: LinearSolverType::GmresJacobi,
            max_iterations: 20,
            restart: 2,
            tolerance: 1e-10,
            ..Default::default()
        };
        for solver in [
            LinearSolverType::GmresJacobi,
            LinearSolverType::Cg,
            LinearSolverType::GmresAmg,
            LinearSolverType::CgAmg,
        ] {
            config.solver = solver;
            let result = solve_with_backend(
                &matrix,
                &[6.0, 7.0],
                &config,
                LinearSolveBackend::Cuda,
                LinearSolveFallback::Reject,
            )
            .unwrap();
            assert_eq!(result.backend.actual, LinearSolveBackend::Cuda);
            assert!(!result.backend.fallback);
            assert!(result.residual < config.tolerance);
            assert!((result.x[0] - 1.0).abs() < 1e-9);
            assert!((result.x[1] - 2.0).abs() < 1e-9);
            result.validate_contract().unwrap();
        }
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn explicit_backend_selector_preserves_cpu_fallback_provenance() {
        let matrix = CsrMatrix::identity(2);
        let config = LinearSolveConfig {
            solver: LinearSolverType::Auto,
            ..Default::default()
        };
        let result = solve_with_backend(
            &matrix,
            &[1.0, -2.0],
            &config,
            LinearSolveBackend::Cuda,
            LinearSolveFallback::AllowCpu,
        )
        .expect("unsupported high-level CUDA strategy should use explicit CPU fallback");
        assert_eq!(result.backend.requested, LinearSolveBackend::Cuda);
        assert_eq!(result.backend.actual, LinearSolveBackend::Cpu);
        assert!(result.backend.fallback);
        result
            .validate_contract()
            .expect("fallback result must satisfy the normal result contract");
        let mut invalid = result.clone();
        invalid.backend = LinearSolveBackendProvenance {
            requested: LinearSolveBackend::Cuda,
            actual: LinearSolveBackend::Cpu,
            fallback: false,
        };
        assert!(matches!(
            invalid.validate_contract(),
            Err(LinearSolveError::ContractViolation(message))
                if message.contains("backend provenance")
        ));

        let error = solve_with_backend(
            &matrix,
            &[1.0, -2.0],
            &config,
            LinearSolveBackend::Cuda,
            LinearSolveFallback::Reject,
        )
        .expect_err("unsupported high-level CUDA strategy must fail closed");
        assert!(matches!(
            error,
            LinearSolveError::BackendUnavailable {
                backend: LinearSolveBackend::Cuda,
                ..
            }
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array;

    fn simple_spd_system() -> (CsrMatrix<f64>, Vec<f64>) {
        let a = CsrMatrix::from_dense(&array![[4.0, 1.0], [1.0, 3.0]], 0.0);
        let b = vec![1.0, 2.0];
        (a, b)
    }

    #[test]
    fn test_solve_real_gmres() {
        let (a, b) = simple_spd_system();
        let config = LinearSolveConfig {
            solver: LinearSolverType::Gmres,
            ..Default::default()
        };
        let result = solve_real(&a, &b, &config).unwrap();
        assert!(result.converged);
        // Exact: x = [1/11, 7/11]
        assert!((result.x[0] - 1.0 / 11.0).abs() < 1e-10);
        assert!((result.x[1] - 7.0 / 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_real_gmres_ilu() {
        let (a, b) = simple_spd_system();
        let config = LinearSolveConfig {
            solver: LinearSolverType::GmresIlu,
            ..Default::default()
        };
        let result = solve_real(&a, &b, &config).unwrap();
        assert!(result.converged);
    }

    #[test]
    fn test_auto_solver_prefers_parallel_large_problem_paths() {
        let (small, _) = simple_spd_system();
        let config = LinearSolveConfig::default();
        assert_eq!(
            config.resolved_solver_for_real(&small),
            LinearSolverType::GmresIlu
        );

        let large_real = CsrMatrix::from_triplets(
            LARGE_PROBLEM_ROWS,
            LARGE_PROBLEM_ROWS,
            (0..LARGE_PROBLEM_ROWS).map(|i| (i, i, 1.0_f64)).collect(),
        );
        assert_eq!(
            config.resolved_solver_for_real(&large_real),
            LinearSolverType::GmresIlu
        );

        let mut large_indefinite_triplets = (0..LARGE_PROBLEM_ROWS)
            .map(|i| (i, i, 1.0_f64))
            .collect::<Vec<_>>();
        large_indefinite_triplets.push((0, 1, 2.0));
        large_indefinite_triplets.push((1, 0, 2.0));
        let large_indefinite = CsrMatrix::from_triplets(
            LARGE_PROBLEM_ROWS,
            LARGE_PROBLEM_ROWS,
            large_indefinite_triplets,
        );
        assert_eq!(
            config.resolved_solver_for_real(&large_indefinite),
            LinearSolverType::GmresIlu,
            "automatic large-system dispatch must not assume SPD from size or diagonal signs"
        );

        let large_complex = CsrMatrix::from_triplets(
            LARGE_PROBLEM_ROWS,
            LARGE_PROBLEM_ROWS,
            (0..LARGE_PROBLEM_ROWS)
                .map(|i| (i, i, Complex64::new(1.0, if i == 0 { 0.1 } else { 0.0 })))
                .collect(),
        );
        assert_eq!(
            config.resolved_solver_for_complex(&large_complex),
            LinearSolverType::FgmresSchwarz
        );

        let large_complex64_real_values = CsrMatrix::from_triplets(
            LARGE_PROBLEM_ROWS,
            LARGE_PROBLEM_ROWS,
            (0..LARGE_PROBLEM_ROWS)
                .map(|i| (i, i, Complex64::new(1.0, 0.0)))
                .collect(),
        );
        assert_eq!(
            resolve_auto_solver(
                LinearSolverType::Auto,
                &large_complex64_real_values,
                is_complex_scalar::<Complex64>(),
            ),
            LinearSolverType::FgmresSchwarz
        );

        let large_complex32_real_values = CsrMatrix::from_triplets(
            LARGE_PROBLEM_ROWS,
            LARGE_PROBLEM_ROWS,
            (0..LARGE_PROBLEM_ROWS)
                .map(|i| (i, i, Complex32::new(1.0, 0.0)))
                .collect(),
        );
        assert_eq!(
            resolve_auto_solver(
                LinearSolverType::Auto,
                &large_complex32_real_values,
                is_complex_scalar::<Complex32>(),
            ),
            LinearSolverType::FgmresSchwarz
        );

        assert_eq!(
            gmres_variant_for_dispatch(LinearSolverType::Auto, &small, &config),
            GmresDispatchVariant::Standard
        );
        assert_eq!(
            gmres_variant_for_dispatch(LinearSolverType::Auto, &large_real, &config),
            GmresDispatchVariant::Pipelined
        );
        assert_eq!(
            gmres_variant_for_dispatch(LinearSolverType::GmresAmg, &large_real, &config),
            GmresDispatchVariant::Standard
        );

        let tiny_restart_config = LinearSolveConfig {
            restart: 8,
            ..Default::default()
        };
        assert_eq!(
            gmres_variant_for_dispatch(LinearSolverType::Auto, &large_real, &tiny_restart_config),
            GmresDispatchVariant::Standard
        );
    }

    #[test]
    fn solve_spmv_plan_uses_iteration_reuse_hint() {
        let config = LinearSolveConfig {
            max_iterations: 123,
            restart: 17,
            ..Default::default()
        };
        let plan_config = spmv_plan_config_for_solve(&config);

        assert_eq!(plan_config.expected_reuse, 123);
    }

    #[test]
    fn linear_config_controls_exact_residual_frequency() {
        let config = LinearSolveConfig {
            exact_residual_check_interval: 25,
            ..Default::default()
        };
        let gmres_config: GmresConfig<f64> =
            gmres_config_for_solve(&config).expect("default tolerance is representable");

        assert_eq!(gmres_config.exact_residual_check_interval, 25);
        assert_eq!(
            LinearSolveConfig::default().exact_residual_check_interval,
            0
        );
    }

    #[test]
    fn test_solve_real_cg() {
        let (a, b) = simple_spd_system();
        let config = LinearSolveConfig {
            solver: LinearSolverType::Cg,
            ..Default::default()
        };
        let result = solve_real(&a, &b, &config).unwrap();
        assert!(result.converged);
    }

    #[test]
    fn test_solve_complex_gmres() {
        let a = CsrMatrix::from_dense(
            &array![
                [Complex64::new(4.0, 0.0), Complex64::new(1.0, 0.0)],
                [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)]
            ],
            0.0,
        );
        let b = vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
        let config = LinearSolveConfig {
            solver: LinearSolverType::Gmres,
            ..Default::default()
        };
        let result = solve_complex(&a, &b, &config).unwrap();
        assert!(result.converged);
    }

    #[test]
    fn test_solve_with_guess_cg_uses_warm_start() {
        // Regression: CG solver in solve_with_guess must actually use x0.
        // Previously, Cg, CgAmg, GmresJacobi, and GmresAmg fell through the
        // catch-all `_ => return solve(matrix, rhs, config)` arm, silently
        // discarding the initial guess.
        let (a, b) = simple_spd_system();
        let x0 = vec![0.05_f64, 0.55_f64]; // Close to exact [1/11, 7/11]
        let config = LinearSolveConfig {
            solver: LinearSolverType::Cg,
            max_iterations: 10,
            tolerance: 1e-10,
            ..Default::default()
        };
        let result = solve_real_with_guess(&a, &b, &x0, &config).unwrap();
        assert!(result.converged);
        // CG with warm start should converge in exactly 1 iteration for a 2x2 system
        assert!(
            result.iterations <= 2,
            "CG warm start should iterate ≤ 2 for 2x2 system, got {}",
            result.iterations
        );
    }

    #[test]
    fn test_solve_with_guess_gmres_jacobi_uses_warm_start() {
        let (a, b) = simple_spd_system();
        let x0 = vec![0.05_f64, 0.55_f64];
        let config = LinearSolveConfig {
            solver: LinearSolverType::GmresJacobi,
            max_iterations: 20,
            restart: 30,
            tolerance: 1e-10,
            ..Default::default()
        };
        let result = solve_real_with_guess(&a, &b, &x0, &config).unwrap();
        assert!(result.converged);
        assert!((result.x[0] - 1.0 / 11.0).abs() < 1e-10);
        assert!((result.x[1] - 7.0 / 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_saddle_point_with_external_schur() {
        let matrix = CsrMatrix::from_dense(
            &array![
                [4.0_f64, 1.0, 1.0, 0.0],
                [1.0_f64, 3.0, 0.0, 1.0],
                [1.0_f64, 0.0, 2.0, 0.0],
                [0.0_f64, 1.0, 0.0, 1.5]
            ],
            1.0e-12,
        );
        let rhs = vec![1.0, 2.0, 3.0, 4.0];
        let schur = CsrMatrix::from_dense(&array![[2.0_f64, 0.0], [0.0, 1.5]], 1.0e-12);
        let config = LinearSolveConfig {
            solver: LinearSolverType::Gmres,
            max_iterations: 80,
            tolerance: 1.0e-10,
            ..Default::default()
        };
        let saddle_config = SaddlePointConfig {
            block1_size: 2,
            schur_strategy: SchurStrategy::External,
            block2_config: crate::preconditioners::BlockPreconditionerConfig::Diagonal,
            ..Default::default()
        };
        let result =
            solve_saddle_point(&matrix, &rhs, &config, saddle_config, Some(&schur)).unwrap();

        let residual = matrix.matvec(&result.x);
        for (value, expected) in residual.iter().zip(rhs.iter()) {
            assert!((value - expected).abs() < 1.0e-7);
        }
        let assumptions = result
            .saddle_assumptions
            .expect("saddle solves must report block assumption evidence");
        assert_eq!(
            assumptions.block1_provenance,
            AssumptionProvenance::IndependentlyValidated
        );
        assert_eq!(
            assumptions.block2_provenance,
            AssumptionProvenance::NotRequired
        );
    }

    #[test]
    fn test_saddle_point_large_amg_requires_and_reports_declared_assumption() {
        let block1_size = 1_025;
        let n = block1_size + 1;
        let matrix =
            CsrMatrix::from_triplets(n, n, (0..n).map(|index| (index, index, 1.0_f64)).collect());
        let rhs = vec![0.0_f64; n];
        let config = LinearSolveConfig {
            solver: LinearSolverType::Gmres,
            max_iterations: 10,
            tolerance: 1.0e-10,
            ..Default::default()
        };
        let saddle_config = SaddlePointConfig {
            block1_size,
            block1_config: crate::preconditioners::BlockPreconditionerConfig::Amg(
                AmgConfig::default(),
            ),
            block2_config: crate::preconditioners::BlockPreconditionerConfig::Diagonal,
            ..Default::default()
        };

        let error = solve_saddle_point(&matrix, &rhs, &config, saddle_config.clone(), None)
            .expect_err("large AMG blocks need an explicit definiteness declaration");
        assert!(
            matches!(error, LinearSolveError::PreconditionerFailed(ref message) if message.contains("SaddlePointBlockAssumption::PositiveDefinite")),
            "unexpected error: {error:?}"
        );

        let result = solve_saddle_point_with_assumptions(
            &matrix,
            &rhs,
            &config,
            saddle_config,
            SaddlePointBlockAssumptions {
                block1: SaddlePointBlockAssumption::PositiveDefinite,
                block2: SaddlePointBlockAssumption::General,
            },
            None,
        )
        .expect("caller-declared large SPD block should be accepted");
        let assumptions = result
            .saddle_assumptions
            .expect("saddle solves must report block assumption evidence");
        assert_eq!(
            assumptions.block1_provenance,
            AssumptionProvenance::CallerDeclared
        );
        assert_eq!(
            assumptions.block2_provenance,
            AssumptionProvenance::NotRequired
        );
    }

    #[test]
    fn test_solve_saddle_point_rejects_non_gmres() {
        let matrix = CsrMatrix::from_dense(&array![[4.0_f64, 1.0], [1.0, 3.0]], 1.0e-12);
        let rhs = vec![1.0, 2.0];
        let config = LinearSolveConfig {
            solver: LinearSolverType::Cg,
            ..Default::default()
        };
        let saddle_config = SaddlePointConfig {
            block1_size: 1,
            ..Default::default()
        };

        let err = solve_saddle_point(&matrix, &rhs, &config, saddle_config, None)
            .expect_err("Cg should be rejected for saddle-point solve");
        assert!(
            matches!(err, LinearSolveError::InvalidConfig(_)),
            "unexpected error variant: {err:?}"
        );
    }

    #[test]
    fn test_solve_saddle_point_external_requires_approximation() {
        let matrix = CsrMatrix::from_dense(&array![[4.0_f64, 1.0], [1.0, 3.0]], 1.0e-12);
        let rhs = vec![1.0, 2.0];
        let config = LinearSolveConfig::default();
        let saddle_config = SaddlePointConfig {
            block1_size: 1,
            schur_strategy: SchurStrategy::External,
            ..Default::default()
        };

        let err = solve_saddle_point(&matrix, &rhs, &config, saddle_config, None)
            .expect_err("External Schur strategy requires schur approximation");
        assert!(
            matches!(err, LinearSolveError::InvalidConfig(_)),
            "unexpected error variant: {err:?}"
        );
    }

    #[test]
    fn test_solve_saddle_point_rejects_invalid_block_split() {
        let matrix = CsrMatrix::from_dense(&array![[4.0_f64, 1.0], [1.0, 3.0]], 1.0e-12);
        let rhs = vec![1.0, 2.0];
        let config = LinearSolveConfig::default();
        let saddle_config = SaddlePointConfig {
            block1_size: 2,
            ..Default::default()
        };

        let err = solve_saddle_point(&matrix, &rhs, &config, saddle_config, None)
            .expect_err("block split must leave both fields non-empty");
        assert!(
            matches!(err, LinearSolveError::InvalidConfig(_)),
            "unexpected error variant: {err:?}"
        );
    }

    #[test]
    fn test_solve_saddle_point_rejects_bad_external_schur_shape() {
        let matrix = CsrMatrix::from_dense(
            &array![
                [4.0_f64, 1.0, 1.0],
                [1.0_f64, 3.0, 0.0],
                [1.0_f64, 0.0, 2.0]
            ],
            1.0e-12,
        );
        let rhs = vec![1.0, 2.0, 3.0];
        let schur = CsrMatrix::from_dense(&array![[1.0_f64, 0.0], [0.0, 1.0]], 1.0e-12);
        let config = LinearSolveConfig::default();
        let saddle_config = SaddlePointConfig {
            block1_size: 2,
            schur_strategy: SchurStrategy::External,
            ..Default::default()
        };

        let err = solve_saddle_point(&matrix, &rhs, &config, saddle_config, Some(&schur))
            .expect_err("external Schur approximation must match trailing block shape");
        assert!(
            matches!(
                err,
                LinearSolveError::DimensionMismatch {
                    operation: "saddle-point Schur approximation",
                    expected: 1,
                    actual: 2,
                }
            ),
            "unexpected error variant: {err:?}"
        );
    }

    #[test]
    fn test_solve_saddle_point_rejects_malformed_matrix_structure() {
        let mut matrix = CsrMatrix::from_dense(&array![[4.0_f64, 1.0], [1.0, 3.0]], 1.0e-12);
        matrix.row_ptrs[2] = 3;
        let rhs = vec![1.0, 2.0];
        let config = LinearSolveConfig {
            solver: LinearSolverType::Gmres,
            ..Default::default()
        };
        let saddle_config = SaddlePointConfig {
            block1_size: 1,
            block2_config: crate::preconditioners::BlockPreconditionerConfig::Diagonal,
            ..Default::default()
        };

        let error = solve_saddle_point(&matrix, &rhs, &config, saddle_config, None)
            .expect_err("malformed CSR storage must be rejected before saddle extraction");
        assert!(
            matches!(error, LinearSolveError::InvalidConfig(ref message) if message.contains("matrix structure")),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn test_solve_saddle_point_rejects_malformed_external_schur_structure() {
        let matrix = CsrMatrix::from_dense(&array![[4.0_f64, 1.0], [1.0, 3.0]], 1.0e-12);
        let mut schur = CsrMatrix::from_dense(&array![[1.0_f64]], 1.0e-12);
        schur.row_ptrs[1] = 0;
        let rhs = vec![1.0, 2.0];
        let config = LinearSolveConfig {
            solver: LinearSolverType::Gmres,
            ..Default::default()
        };
        let saddle_config = SaddlePointConfig {
            block1_size: 1,
            schur_strategy: SchurStrategy::External,
            ..Default::default()
        };

        let error = solve_saddle_point(&matrix, &rhs, &config, saddle_config, Some(&schur))
            .expect_err("malformed Schur storage must be rejected before preconditioner setup");
        assert!(
            matches!(error, LinearSolveError::InvalidConfig(ref message) if message.contains("Schur approximation structure")),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn test_solve_saddle_point_rejects_pcd_lsc_without_specialized_blocks() {
        let matrix = CsrMatrix::from_dense(&array![[4.0_f64, 1.0], [1.0, 2.0]], 1.0e-12);
        let rhs = vec![1.0, 2.0];
        let config = LinearSolveConfig::default();

        for schur_strategy in [SchurStrategy::Pcd, SchurStrategy::Lsc] {
            let saddle_config = SaddlePointConfig {
                block1_size: 1,
                schur_strategy,
                ..Default::default()
            };
            let error = solve_saddle_point(&matrix, &rhs, &config, saddle_config, None)
                .expect_err("PCD/LSC without auxiliary blocks must fail during dispatch");
            assert!(matches!(error, LinearSolveError::InvalidConfig(_)));
        }
    }

    #[test]
    fn test_solve_saddle_point_with_guess_rejects_bad_guess_length() {
        let matrix = CsrMatrix::from_dense(&array![[4.0_f64, 1.0], [1.0, 3.0]], 1.0e-12);
        let rhs = vec![1.0, 2.0];
        let x0 = vec![0.0];
        let config = LinearSolveConfig::default();
        let saddle_config = SaddlePointConfig {
            block1_size: 1,
            block2_config: crate::preconditioners::BlockPreconditionerConfig::Diagonal,
            ..Default::default()
        };

        let err = solve_saddle_point_with_guess(&matrix, &rhs, &x0, &config, saddle_config, None)
            .expect_err("initial guess must match matrix dimension");
        assert!(
            matches!(
                err,
                LinearSolveError::DimensionMismatch {
                    operation: "saddle-point initial guess",
                    expected: 2,
                    actual: 1,
                }
            ),
            "unexpected error variant: {err:?}"
        );
    }

    /// Complex diagonal system of size `n` with imaginary content only at
    /// diagonal entry `imag_idx`.
    fn complex_diagonal_system(
        n: usize,
        imag_idx: usize,
    ) -> (CsrMatrix<Complex64>, Vec<Complex64>) {
        let a = CsrMatrix::from_triplets(
            n,
            n,
            (0..n)
                .map(|i| {
                    let imag = if i == imag_idx { 0.5 } else { 0.0 };
                    (i, i, Complex64::new(2.0, imag))
                })
                .collect(),
        );
        let b = vec![Complex64::new(1.0, 0.0); n];
        (a, b)
    }

    /// Real SPD 1D Poisson system (tridiagonal 2/-1), well suited for AMG.
    fn poisson_spd_system(n: usize) -> (CsrMatrix<f64>, Vec<f64>) {
        let mut triplets = Vec::new();
        for i in 0..n {
            triplets.push((i, i, 2.0));
            if i > 0 {
                triplets.push((i, i - 1, -1.0));
            }
            if i + 1 < n {
                triplets.push((i, i + 1, -1.0));
            }
        }
        (CsrMatrix::from_triplets(n, n, triplets), vec![1.0; n])
    }

    #[test]
    fn test_gmres_amg_rejects_complex_system() {
        let (a, b) = complex_diagonal_system(4, 0);
        let config = LinearSolveConfig {
            solver: LinearSolverType::GmresAmg,
            ..Default::default()
        };

        let err = solve(&a, &b, &config).expect_err("GmresAmg must reject complex systems");
        assert!(
            matches!(err, LinearSolveError::InvalidConfig(_)),
            "unexpected error variant: {err:?}"
        );

        let x0 = vec![Complex64::new(0.0, 0.0); 4];
        let err = solve_with_guess(&a, &b, &x0, &config)
            .expect_err("GmresAmg with guess must reject complex systems");
        assert!(
            matches!(err, LinearSolveError::InvalidConfig(_)),
            "unexpected error variant: {err:?}"
        );
    }

    #[test]
    fn test_cg_amg_rejects_complex_system() {
        let (a, b) = complex_diagonal_system(4, 0);
        let config = LinearSolveConfig {
            solver: LinearSolverType::CgAmg,
            ..Default::default()
        };

        let err = solve(&a, &b, &config).expect_err("CgAmg must reject complex systems");
        assert!(
            matches!(err, LinearSolveError::InvalidConfig(_)),
            "unexpected error variant: {err:?}"
        );

        let x0 = vec![Complex64::new(0.0, 0.0); 4];
        let err = solve_with_guess(&a, &b, &x0, &config)
            .expect_err("CgAmg with guess must reject complex systems");
        assert!(
            matches!(err, LinearSolveError::InvalidConfig(_)),
            "unexpected error variant: {err:?}"
        );
    }

    #[test]
    fn test_amg_dispatch_checks_full_diagonal_for_complex_content() {
        // Regression: the previous check sampled only the first 8 CSR values,
        // so complex content deeper in the diagonal was silently accepted and
        // AMG returned garbage. Entry 12 of 16 must now be caught.
        let (a, b) = complex_diagonal_system(16, 12);
        let x0 = vec![Complex64::new(0.0, 0.0); 16];
        for solver in [LinearSolverType::GmresAmg, LinearSolverType::CgAmg] {
            let config = LinearSolveConfig {
                solver,
                ..Default::default()
            };
            let err = solve(&a, &b, &config)
                .expect_err("complex content past the first 8 diagonal entries must error");
            assert!(
                matches!(err, LinearSolveError::InvalidConfig(_)),
                "unexpected error variant for {solver:?}: {err:?}"
            );
            let err = solve_with_guess(&a, &b, &x0, &config)
                .expect_err("complex content past the first 8 diagonal entries must error");
            assert!(
                matches!(err, LinearSolveError::InvalidConfig(_)),
                "unexpected error variant for {solver:?}: {err:?}"
            );
        }
    }

    #[test]
    fn test_cg_amg_rejects_non_positive_real_diagonal() {
        // A non-positive diagonal entry proves the matrix is not SPD, so CG
        // would silently converge to garbage.
        let a = CsrMatrix::from_dense(&array![[4.0, 1.0], [1.0, -3.0]], 0.0);
        let b = vec![1.0, 2.0];
        let config = LinearSolveConfig {
            solver: LinearSolverType::CgAmg,
            ..Default::default()
        };

        let err = solve(&a, &b, &config).expect_err("CgAmg must reject indefinite systems");
        assert!(
            matches!(err, LinearSolveError::InvalidConfig(_)),
            "unexpected error variant: {err:?}"
        );
    }

    #[test]
    fn test_cg_amg_rejects_symmetric_indefinite_positive_diagonal() {
        // Positive diagonal entries and Hermitian structure are not enough:
        // this matrix has eigenvalues 3 and -1.
        let a = CsrMatrix::from_dense(&array![[1.0, 2.0], [2.0, 1.0]], 0.0);
        let b = vec![1.0, 2.0];
        let config = LinearSolveConfig {
            solver: LinearSolverType::CgAmg,
            ..Default::default()
        };

        let error = solve(&a, &b, &config).expect_err("CG/AMG must reject an indefinite matrix");
        assert!(matches!(error, LinearSolveError::InvalidConfig(_)));
    }

    #[test]
    fn test_gmres_amg_real_dispatch_rejects_symmetric_indefinite_positive_diagonal() {
        let matrix = CsrMatrix::from_dense(&array![[1.0, 2.0], [2.0, 1.0]], 0.0);
        let rhs = vec![1.0, 2.0];
        let config = LinearSolveConfig {
            solver: LinearSolverType::GmresAmg,
            ..Default::default()
        };

        let error = solve_real(&matrix, &rhs, &config)
            .expect_err("real GMRES/AMG must reject an indefinite matrix");
        assert!(matches!(error, LinearSolveError::InvalidConfig(_)));
    }

    #[test]
    fn test_gmres_amg_real_guess_dispatch_rejects_symmetric_indefinite_positive_diagonal() {
        let matrix = CsrMatrix::from_dense(&array![[1.0, 2.0], [2.0, 1.0]], 0.0);
        let rhs = vec![1.0, 2.0];
        let x0 = vec![0.0, 0.0];
        let config = LinearSolveConfig {
            solver: LinearSolverType::GmresAmg,
            ..Default::default()
        };

        let error = solve_real_with_guess(&matrix, &rhs, &x0, &config)
            .expect_err("real GMRES/AMG with a guess must reject an indefinite matrix");
        assert!(matches!(error, LinearSolveError::InvalidConfig(_)));
    }

    #[test]
    fn test_amg_dispatch_accepts_real_spd_system() {
        let (a, b) = poisson_spd_system(32);
        for solver in [LinearSolverType::GmresAmg, LinearSolverType::CgAmg] {
            let config = LinearSolveConfig {
                solver,
                max_iterations: 200,
                tolerance: 1e-8,
                ..Default::default()
            };
            let result = solve(&a, &b, &config)
                .unwrap_or_else(|e| panic!("{solver:?} must solve a real SPD system: {e:?}"));
            assert!(result.converged, "{solver:?} did not converge");
            let residual = a.matvec(&result.x);
            for (value, expected) in residual.iter().zip(b.iter()) {
                assert!(
                    (value - expected).abs() < 1.0e-6,
                    "{solver:?} residual too large: {value} vs {expected}"
                );
            }
        }
    }

    /// Complex-symmetric shifted-Laplacian 1D system (Helmholtz-like,
    /// indefinite for k² > 2): diagonal `2 - k² + i·damp`, off-diagonals `-1`.
    fn shifted_laplacian_complex_system(
        n: usize,
        k2: f64,
    ) -> (CsrMatrix<Complex64>, Vec<Complex64>) {
        let diag = Complex64::new(2.0 - k2, 0.05);
        let off = Complex64::new(-1.0, 0.0);
        let mut triplets = Vec::with_capacity(3 * n);
        for i in 0..n {
            triplets.push((i, i, diag));
            if i > 0 {
                triplets.push((i, i - 1, off));
            }
            if i + 1 < n {
                triplets.push((i, i + 1, off));
            }
        }
        let b = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.1).sin(), (i as f64 * 0.07).cos()))
            .collect();
        (CsrMatrix::from_triplets(n, n, triplets), b)
    }

    fn true_relative_residual_complex(
        a: &CsrMatrix<Complex64>,
        x: &[Complex64],
        b: &[Complex64],
    ) -> f64 {
        let ax = a.matvec(x);
        let err: f64 = crate::vec_ops::vec_sub(&ax, b)
            .iter()
            .map(|e| e.norm_sqr())
            .sum::<f64>()
            .sqrt();
        let b_norm: f64 = b.iter().map(|e| e.norm_sqr()).sum::<f64>().sqrt();
        err / b_norm
    }

    #[test]
    fn test_fgmres_schwarz_explicit_complex_indefinite() {
        // Explicit FGMRES+Schwarz on a complex indefinite shifted-Laplacian
        // system (the Helmholtz FEM regime).
        let (a, b) = shifted_laplacian_complex_system(200, 3.0);
        let config = LinearSolveConfig {
            solver: LinearSolverType::FgmresSchwarz,
            max_iterations: 200,
            tolerance: 1e-8,
            restart: 30,
            ..Default::default()
        };
        let result = solve_complex(&a, &b, &config)
            .expect("FgmresSchwarz must solve a complex indefinite system");
        assert!(result.converged);
        assert!(
            result.residual < 1e-8,
            "reported residual too large: {}",
            result.residual
        );
        let rel = true_relative_residual_complex(&a, &result.x, &b);
        assert!(rel < 1e-6, "true relative residual too large: {rel:.3e}");
    }

    #[test]
    fn test_fgmres_schwarz_explicit_real() {
        // The flexible variant is valid for real systems too.
        let (a, b) = poisson_spd_system(64);
        let config = LinearSolveConfig {
            solver: LinearSolverType::FgmresSchwarz,
            max_iterations: 200,
            tolerance: 1e-8,
            ..Default::default()
        };
        let result = solve_real(&a, &b, &config).expect("FgmresSchwarz must solve a real system");
        assert!(result.converged);
        let ax = a.matvec(&result.x);
        for (value, expected) in ax.iter().zip(b.iter()) {
            assert!((value - expected).abs() < 1.0e-6);
        }
    }

    #[test]
    fn test_fgmres_schwarz_explicit_with_guess() {
        let (a, b) = poisson_spd_system(64);
        let config = LinearSolveConfig {
            solver: LinearSolverType::FgmresSchwarz,
            max_iterations: 200,
            tolerance: 1e-8,
            ..Default::default()
        };
        let cold = solve_real(&a, &b, &config).expect("cold solve must converge");
        let warm =
            solve_real_with_guess(&a, &b, &cold.x, &config).expect("warm solve must converge");
        assert!(
            warm.iterations == 0,
            "exact guess should need 0 iterations, got {}",
            warm.iterations
        );
    }

    #[test]
    fn test_auto_large_complex_shifted_laplacian_picks_flexible_path() {
        // Large complex Auto must route to the flexible outer iteration
        // (FGMRES + Schwarz) and actually solve the system.
        let n = LARGE_PROBLEM_ROWS;
        let (a, b) = shifted_laplacian_complex_system(n, 1.5);
        let config = LinearSolveConfig {
            solver: LinearSolverType::Auto,
            max_iterations: 100,
            tolerance: 1e-8,
            restart: 30,
            ..Default::default()
        };

        assert_eq!(
            config.resolved_solver_for_complex(&a),
            LinearSolverType::FgmresSchwarz,
            "large complex Auto must select the flexible Schwarz path"
        );

        let result = solve_complex(&a, &b, &config)
            .expect("Auto (FgmresSchwarz) must solve the large complex system");
        assert!(result.converged);
        // With one subdomain covering the whole (tridiagonal) matrix the
        // local ILU(0) is exact, so convergence must be nearly immediate.
        assert!(
            result.iterations <= 30,
            "iteration count not sane for an exact Schwarz preconditioner: {}",
            result.iterations
        );
        let rel = true_relative_residual_complex(&a, &result.x, &b);
        assert!(rel < 1e-6, "true relative residual too large: {rel:.3e}");
    }

    #[test]
    fn public_solve_rejects_invalid_configuration_before_dispatch() {
        let a = CsrMatrix::identity(2);
        let config = LinearSolveConfig {
            max_iterations: 0,
            ..Default::default()
        };
        let error = solve_real(&a, &[1.0, 2.0], &config).unwrap_err();
        assert!(matches!(error, LinearSolveError::InvalidConfig(_)));

        let config = LinearSolveConfig {
            tolerance: f64::NAN,
            ..Default::default()
        };
        let error = solve_real(&a, &[1.0, 2.0], &config).unwrap_err();
        assert!(matches!(error, LinearSolveError::InvalidConfig(_)));
    }

    #[test]
    fn public_solve_reports_original_matrix_relative_residual() {
        let a = CsrMatrix::identity(3);
        let b = vec![1.0, -2.0, 4.0];
        let config = LinearSolveConfig {
            solver: LinearSolverType::Gmres,
            tolerance: 1e-12,
            ..Default::default()
        };
        let result = solve_real(&a, &b, &config).expect("identity system must converge");
        assert!(result.residual <= 1e-12);
        assert!(result.backward_error <= 1e-12);
        assert_eq!(result.status, SolverStatus::Converged);
        assert_eq!(result.attainable_accuracy, AccuracyStatus::Attained);
        assert_eq!(result.x, b);
    }

    #[test]
    fn linear_solve_errors_expose_canonical_status_and_accuracy() {
        let not_converged = LinearSolveError::NotConverged {
            iterations: 3,
            residual: 1.0,
        };
        assert_eq!(not_converged.status(), SolverStatus::MaxIterationsReached);
        assert_eq!(
            not_converged.attainable_accuracy(),
            AccuracyStatus::NotAttained
        );

        let invalid = LinearSolveError::InvalidConfig("bad tolerance".into());
        assert_eq!(invalid.status(), SolverStatus::NumericalError);
        assert_eq!(invalid.attainable_accuracy(), AccuracyStatus::NotChecked);
    }

    #[test]
    fn cg_rejects_non_symmetric_system_before_iteration() {
        let matrix = CsrMatrix::from_dense(&array![[2.0, 1.0], [0.0, 2.0]], 0.0);
        let config = LinearSolveConfig {
            solver: LinearSolverType::Cg,
            ..Default::default()
        };
        let error = solve_real(&matrix, &[1.0, 1.0], &config).unwrap_err();
        assert!(matches!(error, LinearSolveError::InvalidConfig(_)));
    }
}

#[cfg(test)]
mod checked_dispatch_tests {
    use super::*;
    use crate::array;
    use num_complex::Complex64;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    #[test]
    fn checked_preconditioner_dispatch_never_panics() {
        let matrix = CsrMatrix::from_dense(&array![[4.0, 1.0], [1.0, 3.0]], 0.0);
        let rhs = [1.0, 2.0];
        let solvers = [
            LinearSolverType::GmresJacobi,
            LinearSolverType::GmresAmg,
            LinearSolverType::GmresChebyshev,
            LinearSolverType::GmresSchwarz,
            LinearSolverType::FgmresSchwarz,
            LinearSolverType::CgAmg,
        ];

        for solver in solvers {
            let config = LinearSolveConfig {
                solver,
                max_iterations: 100,
                tolerance: 1e-8,
                ..Default::default()
            };
            let outcome = catch_unwind(AssertUnwindSafe(|| solve_real(&matrix, &rhs, &config)));
            let result = outcome.expect("checked dispatch must not panic");
            let result = result.expect("small SPD system should converge");
            assert!(result.converged, "{solver:?} did not converge");
            result
                .validate_contract()
                .expect("result contract must hold");
        }
    }

    #[test]
    fn checked_amg_setup_failure_is_typed_and_transactional() {
        let scale = f64::MAX;
        let off_diagonal = 0.49 * scale;
        let matrix = CsrMatrix::from_triplets(
            3,
            3,
            vec![
                (0, 0, scale),
                (1, 1, scale),
                (2, 2, scale),
                (0, 1, off_diagonal),
                (1, 0, off_diagonal),
                (0, 2, off_diagonal),
                (2, 0, off_diagonal),
                (1, 2, off_diagonal),
                (2, 1, off_diagonal),
            ],
        );
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            try_amg_preconditioner(&matrix, AmgConfig::default())
        }));
        let result = outcome.expect("checked AMG setup must not panic");
        assert!(
            matches!(result, Err(LinearSolveError::PreconditionerFailed(_))),
            "finite generated-overflow setup must be returned as a typed error"
        );
    }

    #[test]
    fn checked_complex_dispatch_uses_fallible_preconditioners() {
        let matrix = CsrMatrix::from_triplets(
            2,
            2,
            vec![
                (0, 0, Complex64::new(4.0, 0.0)),
                (0, 1, Complex64::new(1.0, 0.0)),
                (1, 0, Complex64::new(1.0, 0.0)),
                (1, 1, Complex64::new(3.0, 0.0)),
            ],
        );
        let rhs = [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
        for solver in [
            LinearSolverType::GmresJacobi,
            LinearSolverType::GmresSchwarz,
            LinearSolverType::FgmresSchwarz,
        ] {
            let config = LinearSolveConfig {
                solver,
                max_iterations: 100,
                tolerance: 1e-8,
                ..Default::default()
            };
            let outcome = catch_unwind(AssertUnwindSafe(|| solve_complex(&matrix, &rhs, &config)));
            let result = outcome
                .expect("checked complex dispatch must not panic")
                .expect("small complex SPD system should converge");
            assert!(result.converged, "{solver:?} did not converge");
            result
                .validate_contract()
                .expect("result contract must hold");
        }
    }
}
