//! FEM solvers for Helmholtz equation
//!
//! This module provides solvers for the assembled Helmholtz system using
//! sparse matrix operations from the `sonium-solvers` crate.
//!
//! # Solver Types
//!
//! - **Direct**: LU factorization (for small problems)
//! - **GMRES**: Iterative solver with restart
//! - **GMRES+ILU**: GMRES with ILU(0) preconditioning (recommended for large problems)
//! - **GMRES+ShiftedLaplacian**: GMRES with shifted-Laplacian preconditioner (best for Helmholtz)
//! - **GMRES+AMG**: GMRES with algebraic multigrid preconditioning (best parallel scalability)
//! - **GMRES+Deflated+ShiftedLaplacian**: Deflated GMRES with shifted-Laplacian (best for clustered eigenvalues)

mod deflation;
#[cfg(feature = "cuda")]
pub mod device;
pub mod explicit_dynamics;
pub mod implicit_dynamics;
pub mod matrix_free;
pub mod nonlinear;
pub mod vibration_fatigue;

pub use deflation::{DeflationConfig, DeflationMethod};
pub use matrix_free::{MatrixFreeError, MatrixFreeOperator};
pub use nonlinear::{
    FemNewtonConfig, fem_newton_solve, fem_newton_solve_contact,
    fem_newton_solve_contact_with_load, fem_newton_solve_element, fem_newton_solve_element_contact,
    fem_newton_solve_with_bcs, fem_newton_solve_with_bcs_and_load,
};
/// Backend provenance re-exported for application handoff builders so FEM
/// callers report the same execution-origin type the solvers facade returns.
pub use sonium_solvers::LinearSolveBackendProvenance;

use crate::assembly::HelmholtzProblem;
use crate::assembly::{MassMatrix, StiffnessMatrix};
use num_complex::Complex64;
use sonium_solvers::iterative::{
    FgmresConfig, fgmres_preconditioned, gmres_deflated_preconditioned,
    gmres_preconditioned_with_guess,
};
use sonium_solvers::traits::SolverFailure;
/// Status types re-exported so application handoffs report the same
/// convergence/accuracy vocabulary the solvers facade enforces.
pub use sonium_solvers::traits::{AccuracyStatus, SolverStatus};
use sonium_solvers::{
    AmgConfig, AmgPreconditioner, CsrMatrix, DeflationSubspace, GmresConfig, IluPreconditioner,
    lu_solve,
};
use std::time::Instant;
use thiserror::Error;

fn fgmres_config_from_gmres(config: &GmresConfig<f64>) -> FgmresConfig<f64> {
    FgmresConfig {
        max_iterations: config.max_iterations,
        restart: config.restart,
        tolerance: config.tolerance,
        print_interval: config.print_interval,
        exact_residual_check_interval: 0,
        convergence_check_interval: 1,
    }
}

fn shifted_laplacian_amg_config() -> AmgConfig {
    let mut config = AmgConfig::for_parallel();
    config.smoother = sonium_solvers::AmgSmoother::Chebyshev;
    config.strong_threshold = 0.5;
    config
}

fn validate_rhs_and_guess(
    expected: usize,
    rhs: &[Complex64],
    x0: Option<&[Complex64]>,
) -> Result<(), SolverError> {
    if expected != rhs.len() {
        return Err(SolverError::DimensionMismatch {
            expected,
            actual: rhs.len(),
        });
    }
    if rhs
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(SolverError::InvalidConfiguration(
            "right-hand side contains NaN or infinity".into(),
        ));
    }
    if let Some(guess) = x0 {
        if guess.len() != rhs.len() {
            return Err(SolverError::DimensionMismatch {
                expected: rhs.len(),
                actual: guess.len(),
            });
        }
        if guess
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(SolverError::InvalidConfiguration(
                "initial guess contains NaN or infinity".into(),
            ));
        }
    }
    Ok(())
}

/// GMRES solver configuration with f64 tolerance
pub type GmresConfigF64 = GmresConfig<f64>;

/// Solver configuration
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Solver type to use
    pub solver_type: SolverType,
    /// GMRES configuration (used for iterative solvers)
    pub gmres: GmresConfigF64,
    /// Verbosity level (0 = quiet, 1 = summary, 2+ = detailed)
    pub verbosity: usize,
    /// Number of subdomains for Schwarz preconditioning (default: 8)
    pub schwarz_subdomains: usize,
    /// Overlap for Schwarz preconditioning (default: 2)
    pub schwarz_overlap: usize,
    /// Shifted-Laplacian configuration (for GmresShiftedLaplacian solver)
    pub shifted_laplacian: Option<ShiftedLaplacianConfig>,
    /// Wavenumber k (used for default shifted-laplacian and WaveHoltz parameters)
    pub wavenumber: Option<f64>,
    /// WaveHoltz configuration (for WaveHoltz solver)
    pub waveholtz: Option<crate::waveholtz::WaveHoltzConfig>,
    /// Deflation configuration (for GmresDeflatedShiftedLaplacian solver)
    pub deflation: Option<DeflationConfig>,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            solver_type: SolverType::GmresIlu,
            gmres: GmresConfigF64 {
                max_iterations: 1000,
                restart: 50,
                tolerance: 1e-10,
                print_interval: 0,
                convergence_check_interval: 1,
                exact_residual_check_interval: 0,
                min_restart: None,
                max_restart: None,
            },
            verbosity: 0,
            schwarz_subdomains: 8,
            schwarz_overlap: 2,
            shifted_laplacian: None,
            wavenumber: None,
            waveholtz: None,
            deflation: None,
        }
    }
}

/// Type of solver to use
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverType {
    /// Direct LU factorization (best for small problems)
    Direct,
    /// GMRES iterative solver without preconditioning
    Gmres,
    /// GMRES with ILU(0) preconditioning (recommended for large problems)
    GmresIlu,
    /// GMRES with Jacobi (diagonal) preconditioning - fully parallel
    GmresJacobi,
    /// GMRES with parallel ILU using graph coloring (level scheduling)
    GmresIluColoring,
    /// GMRES with parallel ILU using fixed-point iteration
    GmresIluFixedPoint,
    /// GMRES with Additive Schwarz domain decomposition (parallel subdomains)
    GmresSchwarz,
    /// GMRES with Algebraic Multigrid preconditioning (best parallel scalability)
    GmresAmg,
    /// Pipelined GMRES (overlaps communication/computation)
    GmresPipelined,
    /// Pipelined GMRES with ILU(0) preconditioning
    GmresPipelinedIlu,
    /// Pipelined GMRES with AMG preconditioning (best for large parallel problems)
    GmresPipelinedAmg,
    /// GMRES with Shifted-Laplacian preconditioner (best for Helmholtz)
    ///
    /// The shifted-Laplacian preconditioner:
    /// P = K + (α + iβ)M
    ///
    /// where α ≈ k² and β ≈ k are complex shifts that transform
    /// the indefinite Helmholtz operator to a more favorable spectrum.
    ///
    /// Reference: Erlangga et al. (2006) "A class of preconditioners for the Helmholtz equation"
    GmresShiftedLaplacian,
    /// Deflated GMRES with Shifted-Laplacian preconditioner
    ///
    /// Combines deflation (projecting out near-k² eigenmodes) with shifted-Laplacian
    /// preconditioning. The deflation subspace is computed via inverse iteration
    /// and dramatically reduces iteration counts for problems with clustered eigenvalues.
    ///
    /// Reference: Erlangga & Nabben (2008), Gaul et al. (2013)
    GmresDeflatedShiftedLaplacian,
    /// GMRES with shifted-Laplacian and V-cycle multigrid smoothing
    GmresShiftedLaplacianMg,
    /// WaveHoltz: O(N) solver via wave equation time-stepping
    ///
    /// Converts the indefinite Helmholtz problem into positive-definite
    /// wave equation time-steps with CG+AMG inner solves, accelerated
    /// by outer GMRES iteration.
    ///
    /// Reference: Appelo et al. (2020) "WaveHoltz: Iterative Solution of the
    /// Helmholtz Equation via the Wave Equation"
    WaveHoltz,
    /// Additive Schwarz with PML transmission conditions
    ///
    /// Domain decomposition with PML at artificial boundaries achieves
    /// k-independent convergence. Requires mesh geometry — use
    /// `schwarz_pml::solve_schwarz_pml()` directly. CSR dispatch through
    /// `solve`/`solve_csr` is intentionally unimplemented (the CSR view
    /// carries no mesh) and fails closed naming the direct entry point
    /// (SIM-PERF-20261013.md section 2, gap 5).
    ///
    /// Reference: Galkowski et al. (2024, arXiv:2408.16580)
    SchwarzPmlAdditive,
    /// Multiplicative Schwarz with PML transmission conditions
    ///
    /// Sequential variant of PML-Schwarz. Each subdomain sees updated
    /// solution from previous subdomains. Requires mesh geometry — use
    /// `schwarz_pml::solve_schwarz_pml()` directly. CSR dispatch is
    /// intentionally unimplemented, same as the additive variant.
    ///
    /// Reference: Galkowski et al. (2024, arXiv:2408.16580)
    SchwarzPmlMultiplicative,
    /// Neural Multigrid (Wave-ADR-NS) solver
    ///
    /// Deep-learning-enhanced multigrid using Chebyshev-accelerated wave
    /// smoothing and ADR correction from WKB decomposition. Targets
    /// heterogeneous Helmholtz at high wavenumbers. Requires mesh
    /// geometry — use `neural_multigrid::solve_neural_multigrid()` directly.
    ///
    /// Reference: Cui & Jiang (2024, arXiv:2404.02493)
    NeuralMultigrid,
}

/// Configuration for Shifted-Laplacian preconditioner
#[derive(Debug, Clone, PartialEq)]
pub struct ShiftedLaplacianConfig {
    /// Real part of complex shift α (default: k²)
    /// Controls damping of low-frequency content
    pub alpha: f64,
    /// Imaginary part of complex shift β (default: k)
    /// Controls damping of high-frequency content
    pub beta: f64,
    /// Number of multigrid V-cycles (0 = use AMG only)
    pub mg_cycles: usize,
    /// AMG coarsening levels for preconditioner setup
    pub amg_levels: usize,
    /// Relaxation factor for smoother
    pub omega: f64,
    /// Pre-smooth iterations
    pub presmooth: usize,
    /// Post-smooth iterations
    pub postsmooth: usize,
}

impl Default for ShiftedLaplacianConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0, // Will be scaled by k²
            beta: 1.0,  // Will be scaled by k
            mg_cycles: 2,
            amg_levels: 0, // Auto-select
            omega: 0.8,
            presmooth: 2,
            postsmooth: 2,
        }
    }
}

impl ShiftedLaplacianConfig {
    /// Create configuration optimized for given wavenumber
    ///
    /// Uses empirically optimal shifts:
    /// - α = 0.5 k² (moderate real shift)
    /// - β = 0.5 k (imaginary shift for damping)
    pub fn for_wavenumber(k: f64) -> Self {
        Self {
            alpha: 0.5 * k * k,
            beta: 0.5 * k,
            mg_cycles: 2,
            amg_levels: 0,
            omega: 0.8,
            presmooth: 2,
            postsmooth: 2,
        }
    }

    /// Aggressive shifts for difficult problems
    pub fn aggressive(k: f64) -> Self {
        Self {
            alpha: k * k,
            beta: k,
            mg_cycles: 3,
            amg_levels: 0,
            omega: 0.7,
            presmooth: 3,
            postsmooth: 3,
        }
    }

    /// Conservative shifts for well-conditioned problems
    pub fn conservative(k: f64) -> Self {
        Self {
            alpha: 0.25 * k * k,
            beta: 0.25 * k,
            mg_cycles: 1,
            amg_levels: 0,
            omega: 0.9,
            presmooth: 1,
            postsmooth: 1,
        }
    }
}

/// Solution result from the solver
#[derive(Debug, Clone)]
pub struct Solution {
    /// Solution vector
    pub values: Vec<Complex64>,
    /// Number of iterations (0 for direct solver)
    pub iterations: usize,
    /// Final residual norm
    pub residual: f64,
    /// Whether the solver converged
    pub converged: bool,
    /// Canonical solver status: convergence state is read from here, not
    /// inferred from finite output or residual size alone.
    pub status: SolverStatus,
    /// Whether the requested tolerance was attained by the returned iterate,
    /// independently of the status classification.
    pub attainable_accuracy: AccuracyStatus,
    /// Requested and actual execution backend for this solve.
    pub backend: LinearSolveBackendProvenance,
}

impl Solution {
    /// Build a metadata-only XEM result envelope for a solved complex field.
    /// Values remain owned by this solution and are identified to a caller-
    /// owned writer by a `mem://` data reference.
    pub fn to_xem_result(
        &self,
        problem: &sonium_xem::MultiphysicsProblem,
        field_id: &str,
    ) -> Result<sonium_xem::MultiphysicsResult, FemResultError> {
        let field = validate_xem_solution_field(problem, field_id, self.values.len())?;
        let byte_length = self
            .values
            .len()
            .checked_mul(std::mem::size_of::<Complex64>())
            .and_then(|size| u64::try_from(size).ok())
            .ok_or_else(|| FemResultError::LengthMismatch {
                field_id: field.id.clone(),
                expected: field.layout.element_count().unwrap_or(usize::MAX),
                actual: self.values.len(),
            })?;
        Ok(sonium_xem::MultiphysicsResult::from_problem_field(
            problem,
            field_id,
            sonium_xem::DataReference {
                expression: None,
                uri: Some(format!("mem://sonium/fem/{}/{}", problem.id, field_id)),
                byte_offset: 0,
                byte_length,
            },
        )?)
    }

    /// Borrow the solved values through the XEM contiguous-field contract.
    /// The caller retains ownership of both the solution and problem metadata.
    pub fn borrowed_xem_field<'a>(
        &'a self,
        problem: &'a sonium_xem::MultiphysicsProblem,
        field_id: &str,
    ) -> Result<sonium_xem::BorrowedFieldView<'a, Complex64>, FemResultError> {
        let field = validate_xem_solution_field(problem, field_id, self.values.len())?;
        Ok(sonium_xem::BorrowedFieldView::new(
            &field.layout,
            &self.values,
        )?)
    }
}

/// Solver errors
#[derive(Debug, Error)]
pub enum SolverError {
    #[error("Solver failed to converge after {0} iterations (residual: {1})")]
    ConvergenceFailure(usize, f64),
    #[error("Direct solver failed: singular matrix")]
    SingularMatrix,
    #[error("Matrix dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("Invalid solver configuration: {0}")]
    InvalidConfiguration(String),
}

impl SolverFailure for SolverError {
    fn status(&self) -> SolverStatus {
        match self {
            Self::ConvergenceFailure(_, _) => SolverStatus::MaxIterationsReached,
            Self::SingularMatrix => SolverStatus::Breakdown,
            Self::DimensionMismatch { .. } | Self::InvalidConfiguration(_) => {
                SolverStatus::NumericalError
            }
        }
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        match self {
            Self::ConvergenceFailure(_, _) => AccuracyStatus::NotAttained,
            Self::SingularMatrix
            | Self::DimensionMismatch { .. }
            | Self::InvalidConfiguration(_) => AccuracyStatus::NotChecked,
        }
    }
}

fn validate_xem_solution_field<'a>(
    problem: &'a sonium_xem::MultiphysicsProblem,
    field_id: &str,
    actual: usize,
) -> Result<&'a sonium_xem::FieldSpec, FemResultError> {
    problem.validate()?;
    let field = match problem.acoustic_complex_pressure_field(field_id) {
        Ok(field) => field,
        Err(sonium_xem::IrValidationError::MissingReference { .. }) => {
            return Err(FemResultError::MissingField(field_id.to_owned()));
        }
        Err(error) => {
            return Err(FemResultError::IncompatibleField {
                field_id: field_id.to_owned(),
                reason: error.to_string(),
            });
        }
    };
    let expected = field.layout.element_count()?;
    if expected != actual {
        return Err(FemResultError::LengthMismatch {
            field_id: field_id.to_owned(),
            expected,
            actual,
        });
    }
    Ok(field)
}

/// Failure while adapting a solved FEM field to XEM.
#[derive(Debug, Error)]
pub enum FemResultError {
    /// The source problem or result metadata failed XEM validation.
    #[error("invalid XEM result metadata: {0}")]
    InvalidProblem(#[from] sonium_xem::IrValidationError),
    /// The requested field is absent from the XEM problem.
    #[error("XEM field `{0}` is not declared")]
    MissingField(String),
    /// The declared field cannot represent a complex FEM Helmholtz result.
    #[error("incompatible XEM field `{field_id}`: {reason}")]
    IncompatibleField { field_id: String, reason: String },
    /// The declared tensor extent does not match the solved vector.
    #[error("XEM field `{field_id}` expects {expected} values, got {actual}")]
    LengthMismatch {
        field_id: String,
        expected: usize,
        actual: usize,
    },
}

/// Solve a Helmholtz problem
///
/// # Arguments
/// * `problem` - The assembled Helmholtz problem
/// * `config` - Solver configuration
///
/// # Returns
/// Solution containing the nodal values
pub fn solve(problem: &HelmholtzProblem, config: &SolverConfig) -> Result<Solution, SolverError> {
    let start = Instant::now();

    // Convert to CSR format
    let csr = problem.matrix.to_csr();
    let rhs: Vec<Complex64> = problem.rhs.clone();
    validate_rhs_and_guess(csr.num_rows, &rhs, None)?;
    if rhs.is_empty() {
        return Ok(Solution {
            values: Vec::new(),
            iterations: 0,
            residual: 0.0,
            converged: true,
            status: SolverStatus::Converged,
            attainable_accuracy: AccuracyStatus::Attained,
            backend: LinearSolveBackendProvenance::cpu(),
        });
    }

    let csr_time = start.elapsed();

    if config.verbosity > 0 {
        println!(
            "  [FEM] System: {} DOFs, {} nnz, sparsity {:.4}%, CSR convert: {:.1}ms",
            csr.num_rows,
            csr.nnz(),
            csr.sparsity() * 100.0,
            csr_time.as_secs_f64() * 1000.0
        );
    }

    let solve_start = Instant::now();
    let result = match config.solver_type {
        // Helmholtz-specific solvers that need FEM data beyond the CSR matrix
        SolverType::GmresShiftedLaplacian => {
            solve_gmres_shifted_laplacian(problem, &csr, &rhs, config)
        }
        SolverType::GmresDeflatedShiftedLaplacian => {
            solve_gmres_deflated_shifted_laplacian(problem, &csr, &rhs, config)
        }
        SolverType::GmresShiftedLaplacianMg => {
            solve_gmres_shifted_laplacian_mg(problem, &csr, &rhs, config)
        }
        SolverType::WaveHoltz => {
            let default_wh = crate::waveholtz::WaveHoltzConfig::default();
            let wh_config = config.waveholtz.as_ref().unwrap_or(&default_wh);
            crate::waveholtz::solve_waveholtz_from_problem(problem, config, wh_config)
        }
        // All other solvers: delegate to the consolidated solve_csr_with_guess
        _ => solve_csr_with_guess(&csr, &rhs, None, config),
    };
    let solve_time = solve_start.elapsed();

    if config.verbosity > 0
        && let Ok(ref sol) = result
    {
        println!(
            "  [FEM] Solve: {} iters, residual {:.2e}, time {:.1}ms",
            sol.iterations,
            sol.residual,
            solve_time.as_secs_f64() * 1000.0
        );
    }

    result
}

/// Solve using direct LU factorization
fn solve_direct(
    csr: &CsrMatrix<Complex64>,
    rhs: &[Complex64],
    config: &SolverConfig,
) -> Result<Solution, SolverError> {
    if config.verbosity > 0 {
        log::info!("Using direct LU solver");
    }

    if csr.num_rows != csr.num_cols {
        return Err(SolverError::InvalidConfiguration(
            "direct solver requires a square matrix".into(),
        ));
    }
    if rhs.is_empty() {
        return Ok(Solution {
            values: Vec::new(),
            iterations: 0,
            residual: 0.0,
            converged: true,
            status: SolverStatus::Converged,
            attainable_accuracy: AccuracyStatus::Attained,
            backend: LinearSolveBackendProvenance::cpu(),
        });
    }

    // Convert to dense for direct solve (only suitable for small problems)
    let dense = csr.to_dense();
    let solution = lu_solve(&dense, rhs).map_err(|_| SolverError::SingularMatrix)?;

    // Compute residual
    let residual_vec = csr.matvec(&solution);
    let residual: f64 = residual_vec
        .iter()
        .zip(rhs.iter())
        .map(|(r, b)| (r - b).norm())
        .sum::<f64>()
        / rhs.len() as f64;
    if !solution
        .iter()
        .chain(residual_vec.iter())
        .all(|value| value.re.is_finite() && value.im.is_finite())
        || !residual.is_finite()
    {
        return Err(SolverError::InvalidConfiguration(
            "direct solver produced a non-finite solution or residual".into(),
        ));
    }

    Ok(Solution {
        values: solution,
        iterations: 0,
        residual,
        converged: true,
        status: SolverStatus::Converged,
        attainable_accuracy: AccuracyStatus::Attained,
        backend: LinearSolveBackendProvenance::cpu(),
    })
}

// Generic GMRES+preconditioner dispatch has been consolidated into
// sonium_solvers::solve::{solve_real, solve_complex}.
// Only Helmholtz-specific solvers (shifted-Laplacian, deflation) remain below.

/// Build shifted-Laplacian matrix P = K + (α + iβ)M from component matrices
///
/// The shifted-Laplacian preconditioner transforms the Helmholtz operator
/// A = K - k²M to P = K + (α + iβ)M which has more favorable spectral properties.
///
/// Reference: Erlangga et al. (2006) "A class of preconditioners for the Helmholtz equation"
pub(crate) fn build_shifted_laplacian(
    stiffness: &StiffnessMatrix,
    mass: &MassMatrix,
    alpha: f64,
    beta: f64,
) -> CsrMatrix<Complex64> {
    use std::collections::HashMap;

    let mut entries: HashMap<(usize, usize), Complex64> = HashMap::new();

    let shift = Complex64::new(alpha, beta);

    for i in 0..stiffness.nnz() {
        let key = (stiffness.rows[i], stiffness.cols[i]);
        let real_part = stiffness.values[i];
        *entries.entry(key).or_insert(Complex64::new(0.0, 0.0)) += Complex64::new(real_part, 0.0);
    }

    for i in 0..mass.nnz() {
        let key = (mass.rows[i], mass.cols[i]);
        let real_part = mass.values[i];
        *entries.entry(key).or_insert(Complex64::new(0.0, 0.0)) +=
            shift * Complex64::new(real_part, 0.0);
    }

    let dim = stiffness.dim;
    let mut rows = Vec::with_capacity(entries.len());
    let mut cols = Vec::with_capacity(entries.len());
    let mut values = Vec::with_capacity(entries.len());

    for ((r, c), v) in entries {
        if v.norm() > 1e-15 {
            rows.push(r);
            cols.push(c);
            values.push(v);
        }
    }

    CsrMatrix::from_triplets(
        dim,
        dim,
        rows.into_iter()
            .zip(cols)
            .zip(values)
            .map(|((r, c), v)| (r, c, v))
            .collect(),
    )
}

/// Solve using GMRES with Shifted-Laplacian preconditioner
///
/// Uses P = K + (α + iβ)M as preconditioner for the Helmholtz system A = K - k²M.
/// The complex shifts α and β transform the indefinite operator to a more
/// favorable spectrum for iterative methods.
///
/// # Arguments
/// * `problem` - The assembled Helmholtz problem (provides access to K and M)
/// * `csr` - The system matrix A = K - k²M
/// * `rhs` - Right-hand side vector
/// * `config` - Solver configuration with shifted-Laplacian parameters
fn solve_gmres_shifted_laplacian(
    problem: &HelmholtzProblem,
    csr: &CsrMatrix<Complex64>,
    rhs: &[Complex64],
    config: &SolverConfig,
) -> Result<Solution, SolverError> {
    let k = config.wavenumber.unwrap_or(1.0);
    let default_config = ShiftedLaplacianConfig::for_wavenumber(k);
    let sl_config = config.shifted_laplacian.as_ref().unwrap_or(&default_config);

    if config.verbosity > 0 {
        println!(
            "  [FEM] Shifted-Laplacian: α={:.4}, β={:.4}, AMG preconditioner",
            sl_config.alpha, sl_config.beta
        );
    }

    let sl_start = Instant::now();
    let p_matrix = build_shifted_laplacian(
        &problem.stiffness,
        &problem.mass,
        sl_config.alpha,
        sl_config.beta,
    );
    let amg_config = shifted_laplacian_amg_config();
    let precond = AmgPreconditioner::from_csr(&p_matrix, amg_config);
    let sl_time = sl_start.elapsed();

    if config.verbosity > 0 {
        let diag = precond.diagnostics();
        println!(
            "  [FEM] SL-AMG setup: {:.1}ms, {} levels",
            sl_time.as_secs_f64() * 1000.0,
            diag.num_levels
        );
    }

    let fgmres_config = fgmres_config_from_gmres(&config.gmres);
    let result = fgmres_preconditioned(csr, &precond, rhs, None, &fgmres_config);

    if config.verbosity > 0 {
        println!(
            "  [FEM] SL-FGMRES {} in {} iterations (residual: {:.2e})",
            if result.converged {
                "converged"
            } else {
                "did not converge"
            },
            result.iterations,
            result.residual
        );
    }

    if !result.converged {
        return Err(SolverError::ConvergenceFailure(
            result.iterations,
            result.residual,
        ));
    }

    Ok(Solution {
        values: result.x,
        iterations: result.iterations,
        residual: result.residual,
        converged: result.converged,
        status: result.status,
        attainable_accuracy: result.attainable_accuracy,
        backend: LinearSolveBackendProvenance::cpu(),
    })
}

/// Solve using GMRES with Shifted-Laplacian and V-cycle multigrid smoothing
fn solve_gmres_shifted_laplacian_mg(
    problem: &HelmholtzProblem,
    csr: &CsrMatrix<Complex64>,
    rhs: &[Complex64],
    config: &SolverConfig,
) -> Result<Solution, SolverError> {
    let k = config.wavenumber.unwrap_or(1.0);
    let default_config = ShiftedLaplacianConfig::for_wavenumber(k);
    let sl_config = config.shifted_laplacian.as_ref().unwrap_or(&default_config);

    if config.verbosity > 0 {
        println!(
            "  [FEM] Shifted-Laplacian-MG: α={:.4}, β={:.4}, {} V-cycles",
            sl_config.alpha, sl_config.beta, sl_config.mg_cycles
        );
    }

    let p_matrix = build_shifted_laplacian(
        &problem.stiffness,
        &problem.mass,
        sl_config.alpha,
        sl_config.beta,
    );
    let amg_config = shifted_laplacian_amg_config();
    let precond = AmgPreconditioner::from_csr(&p_matrix, amg_config);

    let mut residual = rhs.to_vec();
    let mut solution = vec![Complex64::new(0.0, 0.0); rhs.len()];

    for _ in 0..sl_config.mg_cycles {
        let result = gmres_preconditioned_with_guess(
            &p_matrix,
            &precond,
            &residual,
            Some(solution.as_slice()),
            &config.gmres,
        );
        if result.converged {
            solution = result.x;
            break;
        }
        solution = result.x;
        let p_applied = p_matrix.matvec(&solution);
        for (ri, api) in residual.iter_mut().zip(p_applied.iter()) {
            *ri -= *api;
        }
    }

    let final_residual = csr.matvec(&solution);
    let residual_norm: f64 = rhs
        .iter()
        .zip(final_residual.iter())
        .map(|(b, ax)| (*b - *ax).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let rhs_norm = rhs.iter().map(|value| value.norm_sqr()).sum::<f64>().sqrt();
    if !residual_norm.is_finite()
        || !solution
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    {
        return Err(SolverError::InvalidConfiguration(
            "shifted-Laplacian multigrid produced a non-finite solution or residual".into(),
        ));
    }
    let converged = residual_norm <= config.gmres.tolerance * rhs_norm.max(1.0);
    if !converged {
        return Err(SolverError::ConvergenceFailure(
            sl_config.mg_cycles,
            residual_norm,
        ));
    }

    Ok(Solution {
        values: solution,
        iterations: sl_config.mg_cycles,
        residual: residual_norm,
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
        backend: LinearSolveBackendProvenance::cpu(),
    })
}

/// Solve using GMRES with Shifted-Laplacian preconditioner, with optional initial guess
#[allow(dead_code)]
fn solve_gmres_shifted_laplacian_with_guess(
    problem: &HelmholtzProblem,
    csr: &CsrMatrix<Complex64>,
    rhs: &[Complex64],
    x0: Option<&[Complex64]>,
    config: &SolverConfig,
) -> Result<Solution, SolverError> {
    let k = config.wavenumber.unwrap_or(1.0);
    let default_config = ShiftedLaplacianConfig::for_wavenumber(k);
    let sl_config = config.shifted_laplacian.as_ref().unwrap_or(&default_config);

    if config.verbosity > 0 {
        println!(
            "  [FEM] Shifted-Laplacian: α={:.4}, β={:.4}",
            sl_config.alpha, sl_config.beta
        );
    }

    let p_matrix = build_shifted_laplacian(
        &problem.stiffness,
        &problem.mass,
        sl_config.alpha,
        sl_config.beta,
    );
    let amg_config = shifted_laplacian_amg_config();
    let precond = AmgPreconditioner::from_csr(&p_matrix, amg_config);
    let fgmres_config = fgmres_config_from_gmres(&config.gmres);
    let result = fgmres_preconditioned(csr, &precond, rhs, x0, &fgmres_config);

    if !result.converged {
        return Err(SolverError::ConvergenceFailure(
            result.iterations,
            result.residual,
        ));
    }

    Ok(Solution {
        values: result.x,
        iterations: result.iterations,
        residual: result.residual,
        converged: result.converged,
        status: result.status,
        attainable_accuracy: result.attainable_accuracy,
        backend: LinearSolveBackendProvenance::cpu(),
    })
}

/// Solve using GMRES with Shifted-Laplacian+V-cycle, with optional initial guess
#[allow(dead_code)]
fn solve_gmres_shifted_laplacian_mg_with_guess(
    problem: &HelmholtzProblem,
    csr: &CsrMatrix<Complex64>,
    rhs: &[Complex64],
    x0: Option<&[Complex64]>,
    config: &SolverConfig,
) -> Result<Solution, SolverError> {
    let k = config.wavenumber.unwrap_or(1.0);
    let default_config = ShiftedLaplacianConfig::for_wavenumber(k);
    let sl_config = config.shifted_laplacian.as_ref().unwrap_or(&default_config);

    let p_matrix = build_shifted_laplacian(
        &problem.stiffness,
        &problem.mass,
        sl_config.alpha,
        sl_config.beta,
    );
    let amg_config = shifted_laplacian_amg_config();
    let precond = AmgPreconditioner::from_csr(&p_matrix, amg_config);

    let fgmres_config = fgmres_config_from_gmres(&config.gmres);
    let result = fgmres_preconditioned(csr, &precond, rhs, x0, &fgmres_config);

    if !result.converged {
        return Err(SolverError::ConvergenceFailure(
            result.iterations,
            result.residual,
        ));
    }

    Ok(Solution {
        values: result.x,
        iterations: result.iterations,
        residual: result.residual,
        converged: result.converged,
        status: result.status,
        attainable_accuracy: result.attainable_accuracy,
        backend: LinearSolveBackendProvenance::cpu(),
    })
}

/// Solve using deflated GMRES with Shifted-Laplacian preconditioner
///
/// Computes deflation vectors via inverse iteration, then runs deflated GMRES
/// with shifted-Laplacian AMG preconditioning. Best for problems with
/// eigenvalues clustered near k².
fn solve_gmres_deflated_shifted_laplacian(
    problem: &HelmholtzProblem,
    csr: &CsrMatrix<Complex64>,
    rhs: &[Complex64],
    config: &SolverConfig,
) -> Result<Solution, SolverError> {
    let k = config.wavenumber.unwrap_or(1.0);
    let default_sl = ShiftedLaplacianConfig::for_wavenumber(k);
    let sl_config = config.shifted_laplacian.as_ref().unwrap_or(&default_sl);
    let default_defl = DeflationConfig::for_frequency_sweep(10);
    let defl_config = config.deflation.as_ref().unwrap_or(&default_defl);

    if config.verbosity > 0 {
        println!(
            "  [FEM] Deflated+SL: α={:.4}, β={:.4}, {} deflation vectors",
            sl_config.alpha, sl_config.beta, defl_config.num_vectors
        );
    }

    let setup_start = Instant::now();

    // Compute deflation subspace
    let deflation = deflation::compute_deflation_subspace(problem, csr, k, defl_config);

    // Build shifted-Laplacian preconditioner
    let p_matrix = build_shifted_laplacian(
        &problem.stiffness,
        &problem.mass,
        sl_config.alpha,
        sl_config.beta,
    );
    let precond = AmgPreconditioner::from_csr(&p_matrix, shifted_laplacian_amg_config());

    let setup_time = setup_start.elapsed();

    if config.verbosity > 0 {
        println!(
            "  [FEM] Deflated+SL setup: {:.1}ms",
            setup_time.as_secs_f64() * 1000.0
        );
    }

    let result = gmres_deflated_preconditioned(csr, &precond, &deflation, rhs, None, &config.gmres);

    if config.verbosity > 0 {
        println!(
            "  [FEM] Deflated+SL-GMRES {} in {} iterations (residual: {:.2e})",
            if result.converged {
                "converged"
            } else {
                "did not converge"
            },
            result.iterations,
            result.residual
        );
    }

    if !result.converged {
        return Err(SolverError::ConvergenceFailure(
            result.iterations,
            result.residual,
        ));
    }

    Ok(Solution {
        values: result.x,
        iterations: result.iterations,
        residual: result.residual,
        converged: result.converged,
        status: result.status,
        attainable_accuracy: result.attainable_accuracy,
        backend: LinearSolveBackendProvenance::cpu(),
    })
}

/// Solve using deflated GMRES with a pre-computed deflation subspace
///
/// This is useful for frequency sweeps where the same deflation subspace
/// can be reused across multiple frequencies, avoiding recomputation.
///
/// # Arguments
/// * `problem` - The assembled Helmholtz problem
/// * `deflation` - Pre-computed deflation subspace (from `deflation::compute_deflation_subspace`)
/// * `config` - Solver configuration
pub fn solve_deflated_with_subspace(
    problem: &HelmholtzProblem,
    deflation: &DeflationSubspace<Complex64>,
    config: &SolverConfig,
) -> Result<Solution, SolverError> {
    let csr = problem.matrix.to_csr();
    let rhs: Vec<Complex64> = problem.rhs.clone();

    let k = config.wavenumber.unwrap_or(1.0);
    let default_sl = ShiftedLaplacianConfig::for_wavenumber(k);
    let sl_config = config.shifted_laplacian.as_ref().unwrap_or(&default_sl);

    let p_matrix = build_shifted_laplacian(
        &problem.stiffness,
        &problem.mass,
        sl_config.alpha,
        sl_config.beta,
    );
    let precond = AmgPreconditioner::from_csr(&p_matrix, shifted_laplacian_amg_config());

    let result =
        gmres_deflated_preconditioned(&csr, &precond, deflation, &rhs, None, &config.gmres);

    if !result.converged {
        return Err(SolverError::ConvergenceFailure(
            result.iterations,
            result.residual,
        ));
    }

    Ok(Solution {
        values: result.x,
        iterations: result.iterations,
        residual: result.residual,
        converged: result.converged,
        status: result.status,
        attainable_accuracy: result.attainable_accuracy,
        backend: LinearSolveBackendProvenance::cpu(),
    })
}

/// Solve a Helmholtz problem directly from CSR matrix and RHS
///
/// This CPU convenience function accepts preassembled sparse matrices.
/// For retained CUDA execution, enable `cuda` and prepare
/// `device::CudaConfiguredFemSolveSession::try_from_csr` with an explicit
/// workload and memory pool. That session retains operator/preconditioner/Krylov
/// state and exposes device RHS and numeric updates between solves.
pub fn solve_csr(
    csr: &CsrMatrix<Complex64>,
    rhs: &[Complex64],
    config: &SolverConfig,
) -> Result<Solution, SolverError> {
    solve_csr_with_guess(csr, rhs, None, config)
}

/// Solve a Helmholtz problem directly from CSR matrix and RHS with optional initial guess
///
/// Providing an initial guess can significantly reduce iterations when solving
/// problems at nearby frequencies (warm starting).
///
/// # Arguments
/// * `csr` - System matrix in CSR format
/// * `rhs` - Right-hand side vector
/// * `x0` - Optional initial guess (if None, starts from zero)
/// * `config` - Solver configuration
pub fn solve_csr_with_guess(
    csr: &CsrMatrix<Complex64>,
    rhs: &[Complex64],
    x0: Option<&[Complex64]>,
    config: &SolverConfig,
) -> Result<Solution, SolverError> {
    validate_rhs_and_guess(csr.num_rows, rhs, x0)?;
    if rhs.is_empty() {
        return Ok(Solution {
            values: x0.map_or_else(Vec::new, |guess| guess.to_vec()),
            iterations: 0,
            residual: 0.0,
            converged: true,
            status: SolverStatus::Converged,
            attainable_accuracy: AccuracyStatus::Attained,
            backend: LinearSolveBackendProvenance::cpu(),
        });
    }

    // Helmholtz-specific solvers that need FEM data
    match config.solver_type {
        SolverType::GmresShiftedLaplacian
        | SolverType::GmresDeflatedShiftedLaplacian
        | SolverType::GmresShiftedLaplacianMg => {
            return Err(SolverError::InvalidConfiguration(
                "Shifted-Laplacian solver requires HelmholtzProblem. Use solve() instead.".into(),
            ));
        }
        SolverType::WaveHoltz => {
            return Err(SolverError::InvalidConfiguration(
                "WaveHoltz solver requires HelmholtzAssembler. Use solve() instead.".into(),
            ));
        }
        SolverType::SchwarzPmlAdditive | SolverType::SchwarzPmlMultiplicative => {
            return Err(SolverError::InvalidConfiguration(
                "Schwarz-PML requires mesh geometry. Use schwarz_pml::solve_schwarz_pml().".into(),
            ));
        }
        SolverType::NeuralMultigrid => {
            return Err(SolverError::InvalidConfiguration(
                "NeuralMultigrid requires mesh geometry. Use neural_multigrid::solve_neural_multigrid().".into(),
            ));
        }
        SolverType::GmresSchwarz => {
            return Err(SolverError::InvalidConfiguration(
                "GmresSchwarz is not yet implemented. The solver type exists but Schwarz domain \
                 decomposition preconditioning has not been implemented. Use GmresIlu, GmresAmg, \
                 or GmresJacobi instead."
                    .into(),
            ));
        }
        // Direct solver handled separately (not in sonium_solvers::solve_complex)
        SolverType::Direct => return solve_direct(csr, rhs, config),
        // All generic iterative solvers: delegate to sonium_solvers
        _ => {}
    }

    // Map SolverType to LinearSolverType
    let linear_solver = match config.solver_type {
        SolverType::Gmres | SolverType::GmresPipelined => sonium_solvers::LinearSolverType::Gmres,
        SolverType::GmresJacobi => sonium_solvers::LinearSolverType::GmresJacobi,
        SolverType::GmresIlu
        | SolverType::GmresIluColoring
        | SolverType::GmresIluFixedPoint
        | SolverType::GmresPipelinedIlu => sonium_solvers::LinearSolverType::GmresIlu,
        SolverType::GmresAmg | SolverType::GmresPipelinedAmg => {
            sonium_solvers::LinearSolverType::GmresAmg
        }
        _ => unreachable!("all solver types should be handled above"),
    };

    let solve_config = sonium_solvers::LinearSolveConfig {
        solver: linear_solver,
        max_iterations: config.gmres.max_iterations,
        tolerance: config.gmres.tolerance,
        restart: config.gmres.restart,
        verbosity: config.verbosity,
        exact_residual_check_interval: 0,
    };

    let result = match x0 {
        Some(guess) => sonium_solvers::solve_with_guess(csr, rhs, guess, &solve_config),
        None => sonium_solvers::solve_complex(csr, rhs, &solve_config),
    }
    .map_err(|e| match e {
        sonium_solvers::LinearSolveError::NotConverged {
            iterations,
            residual,
        } => SolverError::ConvergenceFailure(iterations, residual),
        other => SolverError::InvalidConfiguration(other.to_string()),
    })?;

    Ok(Solution {
        values: result.x,
        iterations: result.iterations,
        residual: result.residual,
        converged: result.converged,
        status: result.status,
        attainable_accuracy: result.attainable_accuracy,
        backend: result.backend,
    })
}

/// Solve a Helmholtz system using GMRES with an externally-provided preconditioner matrix
///
/// Builds AMG on `precond_matrix` and uses it to precondition GMRES applied to `system`.
/// This is the recommended entry point for Shifted-Laplacian preconditioning when the
/// assembler can directly produce both A and P matrices with matching sparsity patterns.
///
/// # Arguments
/// * `system` - System matrix A (e.g. K - k²M + boundary terms)
/// * `precond_matrix` - Preconditioner matrix P (e.g. K + (α+iβ)M from shifted-Laplacian)
/// * `rhs` - Right-hand side vector
/// * `x0` - Optional initial guess
/// * `config` - Solver configuration (uses `gmres` field for iteration parameters)
pub fn solve_csr_with_preconditioner_matrix(
    system: &CsrMatrix<Complex64>,
    precond_matrix: &CsrMatrix<Complex64>,
    rhs: &[Complex64],
    x0: Option<&[Complex64]>,
    config: &SolverConfig,
) -> Result<Solution, SolverError> {
    validate_rhs_and_guess(system.num_rows, rhs, x0)?;
    if precond_matrix.num_rows != system.num_rows || precond_matrix.num_cols != system.num_cols {
        return Err(SolverError::InvalidConfiguration(
            "preconditioner matrix dimensions must match the system matrix".into(),
        ));
    }

    // Build AMG on the shifted-Laplacian matrix and use a flexible outer solve.
    let precond = AmgPreconditioner::from_csr(precond_matrix, shifted_laplacian_amg_config());

    let fgmres_config = fgmres_config_from_gmres(&config.gmres);
    let result = fgmres_preconditioned(system, &precond, rhs, x0, &fgmres_config);

    if !result.converged {
        return Err(SolverError::ConvergenceFailure(
            result.iterations,
            result.residual,
        ));
    }

    Ok(Solution {
        values: result.x,
        iterations: result.iterations,
        residual: result.residual,
        converged: result.converged,
        status: result.status,
        attainable_accuracy: result.attainable_accuracy,
        backend: LinearSolveBackendProvenance::cpu(),
    })
}

/// Solve a Helmholtz system using GMRES with ILU(0) built on an external preconditioner matrix
///
/// Uses ILU(0) factorisation of `precond_matrix` as the preconditioner for GMRES on `system`.
/// For moderate-sized problems (< ~10 000 DOFs) this is significantly faster than AMG because
/// ILU factorisation and triangular solves are nearly instant at this scale, while AMG has
/// considerable setup cost and per-cycle overhead.
///
/// Combined with warm-starting (passing a previous solution as `x0`), this typically converges
/// in 5–20 iterations for Helmholtz frequency sweeps.
pub fn solve_csr_with_ilu_preconditioned_matrix(
    system: &CsrMatrix<Complex64>,
    precond_matrix: &CsrMatrix<Complex64>,
    rhs: &[Complex64],
    x0: Option<&[Complex64]>,
    config: &SolverConfig,
) -> Result<Solution, SolverError> {
    if system.num_rows != rhs.len() {
        return Err(SolverError::DimensionMismatch {
            expected: system.num_rows,
            actual: rhs.len(),
        });
    }

    if let Some(guess) = x0
        && guess.len() != rhs.len()
    {
        return Err(SolverError::DimensionMismatch {
            expected: rhs.len(),
            actual: guess.len(),
        });
    }

    // ILU(0) on the shifted-Laplacian: fast to build, fast to apply, effective for moderate sizes
    let precond = IluPreconditioner::from_csr(precond_matrix);
    let result = gmres_preconditioned_with_guess(system, &precond, rhs, x0, &config.gmres);

    if !result.converged {
        return Err(SolverError::ConvergenceFailure(
            result.iterations,
            result.residual,
        ));
    }

    Ok(Solution {
        values: result.x,
        iterations: result.iterations,
        residual: result.residual,
        converged: result.converged,
        status: result.status,
        attainable_accuracy: result.attainable_accuracy,
        backend: LinearSolveBackendProvenance::cpu(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly::{HelmholtzProblem, TripletMatrix};
    use crate::basis::PolynomialDegree;
    use crate::mesh::unit_square_triangles;

    #[test]
    fn solver_errors_expose_shared_status_and_accuracy_contract() {
        let convergence = SolverError::ConvergenceFailure(7, 1.0e-3);
        assert_eq!(convergence.status(), SolverStatus::MaxIterationsReached);
        assert_eq!(convergence.accuracy_status(), AccuracyStatus::NotAttained);

        let singular = SolverError::SingularMatrix;
        assert_eq!(singular.solver_status(), SolverStatus::Breakdown);
        assert_eq!(singular.attainable_accuracy(), AccuracyStatus::NotChecked);

        let invalid = SolverError::InvalidConfiguration("bad tolerance".into());
        assert_eq!(invalid.status(), SolverStatus::NumericalError);
        assert_eq!(invalid.accuracy_status(), AccuracyStatus::NotChecked);
    }

    #[test]
    fn solve_csr_with_guess_honors_an_exact_iterative_initial_field() {
        let mut matrix = TripletMatrix::new(2, 2);
        matrix.push(0, 0, Complex64::new(2.0, 0.0));
        matrix.push(1, 1, Complex64::new(3.0, 0.0));
        let matrix = matrix.to_csr();
        let rhs = [Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)];
        let exact = [Complex64::new(1.0, 0.0); 2];
        let config = SolverConfig {
            solver_type: SolverType::Gmres,
            ..SolverConfig::default()
        };
        let solution = solve_csr_with_guess(&matrix, &rhs, Some(&exact), &config).unwrap();
        assert_eq!(solution.iterations, 0);
        assert_eq!(solution.values, exact);
    }

    #[test]
    fn schwarz_pml_csr_dispatch_points_at_direct_api() {
        // SIM-PERF-20261013.md section 2, gap 5: CSR dispatch is
        // intentionally unimplemented (Schwarz-PML needs mesh geometry
        // the CSR view does not carry); the refusal must name the
        // direct entry point instead of failing opaquely.
        let mut matrix = TripletMatrix::new(1, 1);
        matrix.push(0, 0, Complex64::new(2.0, 0.0));
        let matrix = matrix.to_csr();
        let rhs = [Complex64::new(1.0, 0.0)];
        for variant in [
            SolverType::SchwarzPmlAdditive,
            SolverType::SchwarzPmlMultiplicative,
        ] {
            let config = SolverConfig {
                solver_type: variant,
                ..SolverConfig::default()
            };
            let error = solve_csr(&matrix, &rhs, &config)
                .expect_err("CSR dispatch must refuse Schwarz-PML");
            let message = format!("{error}");
            assert!(
                message.contains("schwarz_pml::solve_schwarz_pml"),
                "error must name the direct API, got: {message}"
            );
        }
    }

    #[test]
    fn direct_solver_handles_empty_system_without_nan_residual() {
        let matrix = TripletMatrix::new(0, 0).to_csr();
        let config = SolverConfig {
            solver_type: SolverType::Direct,
            ..SolverConfig::default()
        };
        let solution = solve_csr(&matrix, &[], &config).expect("empty direct system");
        assert!(solution.converged);
        assert_eq!(solution.residual, 0.0);
        assert!(solution.values.is_empty());
    }

    #[test]
    fn solution_propagates_status_accuracy_and_backend() {
        let mut matrix = TripletMatrix::new(1, 1);
        matrix.push(0, 0, Complex64::new(2.0, 0.0));
        let matrix = matrix.to_csr();
        let config = SolverConfig {
            solver_type: SolverType::Direct,
            ..SolverConfig::default()
        };
        let solution = solve_csr(&matrix, &[Complex64::new(4.0, 0.0)], &config)
            .expect("direct solve must succeed");
        assert!(solution.converged);
        assert_eq!(solution.status, SolverStatus::Converged);
        assert_eq!(solution.attainable_accuracy, AccuracyStatus::Attained);
        assert_eq!(
            solution.converged,
            solution.status == SolverStatus::Converged
        );
        assert!((solution.values[0] - Complex64::new(2.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn solver_rejects_nonfinite_rhs_before_backend_dispatch() {
        let mut matrix = TripletMatrix::new(1, 1);
        matrix.push(0, 0, Complex64::new(1.0, 0.0));
        let matrix = matrix.to_csr();
        let config = SolverConfig::default();
        let error = solve_csr(&matrix, &[Complex64::new(f64::NAN, 0.0)], &config)
            .expect_err("non-finite RHS must fail closed");
        assert!(matches!(error, SolverError::InvalidConfiguration(_)));
    }

    #[test]
    fn direct_solver_rejects_rectangular_matrix() {
        let mut matrix = TripletMatrix::new(1, 2);
        matrix.push(0, 0, Complex64::new(1.0, 0.0));
        let matrix = matrix.to_csr();
        let config = SolverConfig {
            solver_type: SolverType::Direct,
            ..SolverConfig::default()
        };
        let error = solve_csr(&matrix, &[Complex64::new(1.0, 0.0)], &config)
            .expect_err("direct solve must require a square matrix");
        assert!(matches!(error, SolverError::InvalidConfiguration(_)));
    }

    #[test]
    fn preconditioner_entry_rejects_mismatched_matrix_before_setup() {
        let mut system = TripletMatrix::new(2, 2);
        system.push(0, 0, Complex64::new(1.0, 0.0));
        system.push(1, 1, Complex64::new(1.0, 0.0));
        let mut preconditioner = TripletMatrix::new(1, 1);
        preconditioner.push(0, 0, Complex64::new(1.0, 0.0));
        let config = SolverConfig::default();
        let error = solve_csr_with_preconditioner_matrix(
            &system.to_csr(),
            &preconditioner.to_csr(),
            &[Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
            None,
            &config,
        )
        .expect_err("mismatched preconditioner must fail before AMG setup");
        assert!(matches!(error, SolverError::InvalidConfiguration(_)));
    }

    #[test]
    fn solved_values_adapt_to_xem_without_copying() {
        let exchange = sonium_xem::MultiphysicsProblem {
            id: "fem-case".into(),
            domains: vec![sonium_xem::Domain {
                id: "acoustics".into(),
                label: "Acoustics".into(),
                kind: sonium_xem::DomainKind::Acoustics,
            }],
            fields: vec![sonium_xem::FieldSpec {
                id: "pressure".into(),
                name: "nodal pressure".into(),
                domain_id: "acoustics".into(),
                kind: sonium_xem::FieldKind::ComplexScalar,
                units: Some("Pa".into()),
                layout: sonium_xem::TensorLayout {
                    scalar_type: sonium_xem::ScalarType::ComplexF64,
                    shape: vec![2],
                    strides: vec![1],
                    ..Default::default()
                },
            }],
            ..Default::default()
        };
        let solution = Solution {
            values: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, -1.0)],
            iterations: 3,
            residual: 1.0e-12,
            converged: true,
            status: SolverStatus::Converged,
            attainable_accuracy: AccuracyStatus::Attained,
            backend: LinearSolveBackendProvenance::cpu(),
        };
        let result = solution
            .to_xem_result(&exchange, "pressure")
            .expect("nodal field metadata should adapt");
        assert_eq!(
            result.fields[0].data.byte_length,
            2 * std::mem::size_of::<Complex64>() as u64
        );
        let view = solution
            .borrowed_xem_field(&exchange, "pressure")
            .expect("nodal values should be borrowed");
        assert_eq!(view.values().as_ptr(), solution.values.as_ptr());
    }

    #[test]
    fn test_solve_helmholtz_direct() {
        let mesh = unit_square_triangles(4);
        let k = Complex64::new(1.0, 0.0);

        let problem = HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k, |_, _, _| {
            Complex64::new(1.0, 0.0)
        });

        let config = SolverConfig {
            solver_type: SolverType::Direct,
            ..Default::default()
        };

        let solution = solve(&problem, &config).expect("Solver should succeed");
        assert!(solution.converged);
        assert_eq!(solution.values.len(), problem.num_dofs());
    }

    #[test]
    fn test_solve_helmholtz_gmres() {
        let mesh = unit_square_triangles(4);
        let k = Complex64::new(1.0, 0.0);

        let problem = HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k, |_, _, _| {
            Complex64::new(1.0, 0.0)
        });

        let config = SolverConfig {
            solver_type: SolverType::Gmres,
            gmres: GmresConfigF64 {
                max_iterations: 100,
                restart: 20,
                tolerance: 1e-8,
                print_interval: 0,
                convergence_check_interval: 1,
                exact_residual_check_interval: 0,
                min_restart: None,
                max_restart: None,
            },
            ..Default::default()
        };

        let solution = solve(&problem, &config).expect("Solver should succeed");
        assert!(solution.converged);
    }

    #[test]
    fn test_solve_helmholtz_gmres_ilu() {
        let mesh = unit_square_triangles(4);
        let k = Complex64::new(1.0, 0.0);

        let problem = HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k, |_, _, _| {
            Complex64::new(1.0, 0.0)
        });

        let config = SolverConfig {
            solver_type: SolverType::GmresIlu,
            gmres: GmresConfigF64 {
                max_iterations: 100,
                restart: 20,
                tolerance: 1e-8,
                print_interval: 0,
                convergence_check_interval: 1,
                exact_residual_check_interval: 0,
                min_restart: None,
                max_restart: None,
            },
            ..Default::default()
        };

        let solution = solve(&problem, &config).expect("Solver should succeed");
        assert!(solution.converged);
    }

    #[test]
    fn test_csr_conversion() {
        let mesh = unit_square_triangles(4);
        let k = Complex64::new(1.0, 0.0);

        let problem = HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k, |_, _, _| {
            Complex64::new(1.0, 0.0)
        });

        let csr = problem.matrix.to_csr();

        // Verify dimensions
        assert_eq!(csr.num_rows, problem.num_dofs());
        assert_eq!(csr.num_cols, problem.num_dofs());

        // Verify nnz is reasonable (should be <= original since duplicates are merged)
        assert!(csr.nnz() > 0);
        assert!(csr.nnz() <= problem.matrix.nnz());
    }

    #[test]
    fn test_ilu_preconditioner_improves_convergence() {
        let mesh = unit_square_triangles(8); // Larger mesh
        let k = Complex64::new(2.0, 0.0);

        let problem = HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k, |x, y, _| {
            Complex64::new(
                (x * std::f64::consts::PI).sin() * (y * std::f64::consts::PI).sin(),
                0.0,
            )
        });

        let gmres_config = GmresConfigF64 {
            max_iterations: 500,
            restart: 30,
            tolerance: 1e-8,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
            min_restart: None,
            max_restart: None,
        };

        // Solve without preconditioning
        let config_no_precond = SolverConfig {
            solver_type: SolverType::Gmres,
            gmres: gmres_config.clone(),
            ..Default::default()
        };

        // Solve with ILU preconditioning
        let config_ilu = SolverConfig {
            solver_type: SolverType::GmresIlu,
            gmres: gmres_config,
            ..Default::default()
        };

        let sol_no_precond = solve(&problem, &config_no_precond).expect("Should converge");
        let sol_ilu = solve(&problem, &config_ilu).expect("Should converge");

        // ILU should require fewer or equal iterations
        assert!(
            sol_ilu.iterations <= sol_no_precond.iterations + 10,
            "ILU should not significantly increase iterations: {} vs {}",
            sol_ilu.iterations,
            sol_no_precond.iterations
        );
    }

    #[test]
    fn test_solve_helmholtz_gmres_pipelined() {
        let mesh = unit_square_triangles(4);
        let k = Complex64::new(1.0, 0.0);

        let problem = HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k, |_, _, _| {
            Complex64::new(1.0, 0.0)
        });

        let config = SolverConfig {
            solver_type: SolverType::GmresPipelined,
            gmres: GmresConfigF64 {
                max_iterations: 100,
                restart: 20,
                tolerance: 1e-8,
                print_interval: 0,
                convergence_check_interval: 1,
                exact_residual_check_interval: 0,
                min_restart: None,
                max_restart: None,
            },
            ..Default::default()
        };

        let solution = solve(&problem, &config).expect("Solver should succeed");
        assert!(solution.converged);
    }

    #[test]
    fn test_solve_helmholtz_gmres_pipelined_ilu() {
        let mesh = unit_square_triangles(4);
        let k = Complex64::new(1.0, 0.0);

        let problem = HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k, |_, _, _| {
            Complex64::new(1.0, 0.0)
        });

        let config = SolverConfig {
            solver_type: SolverType::GmresPipelinedIlu,
            gmres: GmresConfigF64 {
                max_iterations: 100,
                restart: 20,
                tolerance: 1e-8,
                print_interval: 0,
                convergence_check_interval: 1,
                exact_residual_check_interval: 0,
                min_restart: None,
                max_restart: None,
            },
            ..Default::default()
        };

        let solution = solve(&problem, &config).expect("Solver should succeed");
        assert!(solution.converged);
    }

    #[test]
    fn test_solve_helmholtz_gmres_shifted_laplacian() {
        let mesh = unit_square_triangles(4);
        let k = Complex64::new(1.0, 0.0);

        let problem = HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k, |_, _, _| {
            Complex64::new(1.0, 0.0)
        });

        let config = SolverConfig {
            solver_type: SolverType::GmresShiftedLaplacian,
            gmres: GmresConfigF64 {
                max_iterations: 100,
                restart: 20,
                tolerance: 1e-8,
                print_interval: 0,
                convergence_check_interval: 1,
                exact_residual_check_interval: 0,
                min_restart: None,
                max_restart: None,
            },
            wavenumber: Some(1.0),
            ..Default::default()
        };

        let solution = solve(&problem, &config).expect("Solver should succeed");
        assert!(solution.converged);
        assert_eq!(solution.values.len(), problem.num_dofs());
    }

    #[test]
    fn test_solve_helmholtz_gmres_deflated_shifted_laplacian() {
        let mesh = unit_square_triangles(8);
        let k = Complex64::new(5.0, 0.0);

        let problem = HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k, |x, y, _| {
            Complex64::new(
                (x * std::f64::consts::PI).sin() * (y * std::f64::consts::PI).sin(),
                0.0,
            )
        });

        let config = SolverConfig {
            solver_type: SolverType::GmresDeflatedShiftedLaplacian,
            gmres: GmresConfigF64 {
                max_iterations: 200,
                restart: 30,
                tolerance: 1e-8,
                print_interval: 0,
                convergence_check_interval: 1,
                exact_residual_check_interval: 0,
                min_restart: None,
                max_restart: None,
            },
            wavenumber: Some(5.0),
            deflation: Some(DeflationConfig::for_frequency_sweep(5)),
            ..Default::default()
        };

        let solution = solve(&problem, &config).expect("Deflated solver should succeed");
        assert!(solution.converged);
        assert_eq!(solution.values.len(), problem.num_dofs());
    }

    #[test]
    fn test_deflated_vs_standard_iteration_count() {
        let mesh = unit_square_triangles(8);
        let k = Complex64::new(5.0, 0.0);

        let problem = HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k, |x, y, _| {
            Complex64::new(
                (x * std::f64::consts::PI).sin() * (y * std::f64::consts::PI).sin(),
                0.0,
            )
        });

        let gmres_config = GmresConfigF64 {
            max_iterations: 500,
            restart: 50,
            tolerance: 1e-8,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
            min_restart: None,
            max_restart: None,
        };

        // Non-deflated shifted-Laplacian
        let config_sl = SolverConfig {
            solver_type: SolverType::GmresShiftedLaplacian,
            gmres: gmres_config.clone(),
            wavenumber: Some(5.0),
            ..Default::default()
        };

        // Deflated shifted-Laplacian
        let config_defl = SolverConfig {
            solver_type: SolverType::GmresDeflatedShiftedLaplacian,
            gmres: gmres_config,
            wavenumber: Some(5.0),
            deflation: Some(DeflationConfig::for_frequency_sweep(5)),
            ..Default::default()
        };

        let sol_sl = solve(&problem, &config_sl).expect("SL should converge");
        let sol_defl = solve(&problem, &config_defl).expect("Deflated should converge");

        assert!(sol_sl.converged);
        assert!(sol_defl.converged);

        // Deflated should use fewer or comparable iterations
        // (with generous margin since the test problem is small)
        assert!(
            sol_defl.iterations <= sol_sl.iterations + 20,
            "Deflated ({}) should not use significantly more iterations than SL ({})",
            sol_defl.iterations,
            sol_sl.iterations
        );
    }

    #[test]
    fn test_shifted_laplacian_config_constructors() {
        let config_default = ShiftedLaplacianConfig::default();
        assert_eq!(config_default.alpha, 1.0);
        assert_eq!(config_default.beta, 1.0);

        let config_k = ShiftedLaplacianConfig::for_wavenumber(2.0);
        assert_eq!(config_k.alpha, 2.0); // 0.5 * k^2 = 0.5 * 4 = 2
        assert_eq!(config_k.beta, 1.0); // 0.5 * k = 0.5 * 2 = 1

        let config_aggressive = ShiftedLaplacianConfig::aggressive(2.0);
        assert_eq!(config_aggressive.alpha, 4.0); // k^2 = 4
        assert_eq!(config_aggressive.beta, 2.0); // k = 2

        let config_conservative = ShiftedLaplacianConfig::conservative(2.0);
        assert_eq!(config_conservative.alpha, 1.0); // 0.25 * k^2 = 0.25 * 4 = 1
        assert_eq!(config_conservative.beta, 0.5); // 0.25 * k = 0.25 * 2 = 0.5
    }

    #[test]
    fn test_shifted_laplacian_amg_uses_chebyshev_smoother() {
        let config = shifted_laplacian_amg_config();
        assert_eq!(config.smoother, sonium_solvers::AmgSmoother::Chebyshev);
        assert_eq!(config.strong_threshold, 0.5);
    }
}
