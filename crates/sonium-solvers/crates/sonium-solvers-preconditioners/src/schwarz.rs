//! Additive Schwarz Domain Decomposition Preconditioner
//!
//! Implements parallel domain decomposition with overlapping subdomains.
//! Each subdomain is solved independently (embarrassingly parallel), and
//! solutions are combined additively.
//!
//! # Algorithm
//!
//! 1. Partition DOFs into `num_subdomains` groups (contiguous blocks by default)
//! 2. Extend each subdomain by `overlap` layers of neighboring DOFs
//! 3. For each preconditioner apply:
//!    - Extract local subdomain problems in parallel
//!    - Solve each subdomain independently (using local ILU or direct solve)
//!    - Combine solutions with appropriate weighting in overlap regions
//!
//! # Parallelism
//!
//! The subdomain solves are embarrassingly parallel since they operate on
//! independent (though overlapping) portions of the matrix.

use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use sonium_solvers_core::traits::{AdjointCapability, ComplexField, Preconditioner, SolverError};
use sonium_solvers_direct::SparseLuFactorization;
use sonium_solvers_eigen::{LobpcgConfig, lobpcg_generalized};
use sonium_solvers_sparse::CsrMatrix;
use sonium_solvers_sys::topology::MachineTopology;

use rayon::prelude::*;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::ilu::{IluDiagnostics, ilu0_factorize, try_ilu0_factorize};

/// Versioned portable checkpoint format for additive Schwarz.
///
/// Version 1 contains one-level state only; version 2 additionally persists
/// an attached coarse basis and its three sparse-LU factors.  The reader keeps
/// accepting version 1 packets for compatibility.
pub const SCHWARZ_CHECKPOINT_FORMAT_VERSION: u16 = 2;
const SCHWARZ_CHECKPOINT_LEGACY_FORMAT_VERSION: u16 = 1;
const SCHWARZ_CHECKPOINT_MAGIC: &[u8; 8] = b"SSCHWZ01";

fn schwarz_checkpoint_error(message: impl Into<String>) -> SolverError {
    SolverError::OperationFailed {
        operation: "Schwarz checkpoint",
        reason: message.into(),
    }
}

fn schwarz_checkpoint_push_len(output: &mut Vec<u8>, value: usize) -> Result<(), SolverError> {
    let value = u64::try_from(value)
        .map_err(|_| schwarz_checkpoint_error("usize value does not fit checkpoint format"))?;
    output.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn schwarz_checkpoint_push_scalar<T: ComplexField>(
    output: &mut Vec<u8>,
    value: T,
) -> Result<(), SolverError> {
    let re = value
        .re()
        .to_f64()
        .ok_or_else(|| schwarz_checkpoint_error("real scalar cannot be represented as f64"))?;
    let im = value
        .im()
        .to_f64()
        .ok_or_else(|| schwarz_checkpoint_error("imaginary scalar cannot be represented as f64"))?;
    if !re.is_finite() || !im.is_finite() {
        return Err(schwarz_checkpoint_error(
            "checkpoint contains a non-finite scalar",
        ));
    }
    output.extend_from_slice(&re.to_le_bytes());
    output.extend_from_slice(&im.to_le_bytes());
    Ok(())
}

fn schwarz_checkpoint_push_bytes(output: &mut Vec<u8>, bytes: &[u8]) -> Result<(), SolverError> {
    schwarz_checkpoint_push_len(output, bytes.len())?;
    output.extend_from_slice(bytes);
    Ok(())
}

struct SchwarzCheckpointReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> SchwarzCheckpointReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], SolverError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| schwarz_checkpoint_error("checkpoint offset overflow"))?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| schwarz_checkpoint_error("truncated Schwarz checkpoint"))?;
        self.offset = end;
        Ok(bytes)
    }

    fn read_u8(&mut self) -> Result<u8, SolverError> {
        Ok(self.take(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16, SolverError> {
        let mut bytes = [0_u8; 2];
        bytes.copy_from_slice(self.take(2)?);
        Ok(u16::from_le_bytes(bytes))
    }

    fn read_u64(&mut self) -> Result<u64, SolverError> {
        let mut bytes = [0_u8; 8];
        bytes.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_len(&mut self, what: &str, element_bytes: usize) -> Result<usize, SolverError> {
        let value = usize::try_from(self.read_u64()?)
            .map_err(|_| schwarz_checkpoint_error(format!("{what} does not fit usize")))?;
        if element_bytes == 0
            || value > self.bytes.len().saturating_sub(self.offset) / element_bytes
        {
            return Err(schwarz_checkpoint_error(format!(
                "{what} exceeds remaining checkpoint storage"
            )));
        }
        Ok(value)
    }

    fn read_scalar<T: ComplexField>(&mut self) -> Result<T, SolverError> {
        let mut re = [0_u8; 8];
        let mut im = [0_u8; 8];
        re.copy_from_slice(self.take(8)?);
        im.copy_from_slice(self.take(8)?);
        let re = f64::from_le_bytes(re);
        let im = f64::from_le_bytes(im);
        if !re.is_finite() || !im.is_finite() {
            return Err(schwarz_checkpoint_error(
                "checkpoint contains a non-finite scalar",
            ));
        }
        let re = T::Real::from_f64(re)
            .ok_or_else(|| schwarz_checkpoint_error("real scalar conversion failed"))?;
        let im = T::Real::from_f64(im)
            .ok_or_else(|| schwarz_checkpoint_error("imaginary scalar conversion failed"))?;
        if !re.is_finite() || !im.is_finite() {
            return Err(schwarz_checkpoint_error(
                "checkpoint scalar overflows requested real type",
            ));
        }
        Ok(T::from_re_im(re, im))
    }

    fn is_at_end(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

fn schwarz_checkpoint_push_csr<T: ComplexField>(
    output: &mut Vec<u8>,
    matrix: &CsrMatrix<T>,
) -> Result<(), SolverError> {
    matrix.validate_canonical_structure()?;
    schwarz_checkpoint_push_len(output, matrix.num_rows)?;
    schwarz_checkpoint_push_len(output, matrix.num_cols)?;
    schwarz_checkpoint_push_len(output, matrix.row_ptrs.len())?;
    for &row_ptr in &matrix.row_ptrs {
        schwarz_checkpoint_push_len(output, row_ptr)?;
    }
    schwarz_checkpoint_push_len(output, matrix.col_indices.len())?;
    for &column in &matrix.col_indices {
        schwarz_checkpoint_push_len(output, column)?;
    }
    schwarz_checkpoint_push_len(output, matrix.values.len())?;
    for &value in &matrix.values {
        schwarz_checkpoint_push_scalar(output, value)?;
    }
    Ok(())
}

fn schwarz_checkpoint_read_csr<T: ComplexField>(
    reader: &mut SchwarzCheckpointReader<'_>,
    label: &str,
) -> Result<CsrMatrix<T>, SolverError> {
    let rows = usize::try_from(reader.read_u64()?)
        .map_err(|_| schwarz_checkpoint_error(format!("{label} row count does not fit usize")))?;
    let cols = usize::try_from(reader.read_u64()?).map_err(|_| {
        schwarz_checkpoint_error(format!("{label} column count does not fit usize"))
    })?;
    let row_ptr_count = reader.read_len(&format!("{label} row-pointer count"), 8)?;
    let expected_row_ptr_count = rows
        .checked_add(1)
        .ok_or_else(|| schwarz_checkpoint_error(format!("{label} row count overflows")))?;
    if row_ptr_count != expected_row_ptr_count {
        return Err(schwarz_checkpoint_error(format!(
            "{label} row-pointer count does not match row count"
        )));
    }
    let mut row_ptrs = Vec::with_capacity(row_ptr_count);
    for _ in 0..row_ptr_count {
        row_ptrs.push(usize::try_from(reader.read_u64()?).map_err(|_| {
            schwarz_checkpoint_error(format!("{label} row pointer does not fit usize"))
        })?);
    }
    if row_ptrs.first().copied() != Some(0)
        || row_ptrs.windows(2).any(|window| window[0] > window[1])
    {
        return Err(schwarz_checkpoint_error(format!(
            "{label} row pointers are not monotone"
        )));
    }
    let column_count = reader.read_len(&format!("{label} column count"), 8)?;
    let mut col_indices = Vec::with_capacity(column_count);
    for _ in 0..column_count {
        let column = usize::try_from(reader.read_u64()?).map_err(|_| {
            schwarz_checkpoint_error(format!("{label} column index does not fit usize"))
        })?;
        if column >= cols {
            return Err(schwarz_checkpoint_error(format!(
                "{label} column index outside matrix dimensions"
            )));
        }
        col_indices.push(column);
    }
    let value_count = reader.read_len(&format!("{label} value count"), 16)?;
    if value_count != column_count || row_ptrs.last().copied() != Some(value_count) {
        return Err(schwarz_checkpoint_error(format!(
            "{label} CSR column/value counts do not match row pointers"
        )));
    }
    let mut values = Vec::with_capacity(value_count);
    for _ in 0..value_count {
        values.push(reader.read_scalar::<T>()?);
    }
    let matrix = CsrMatrix::try_from_raw_parts(rows, cols, row_ptrs, col_indices, values)?;
    matrix.validate_canonical_structure()?;
    Ok(matrix)
}

/// Outcome of the most recent optional coarse-space setup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchwarzCoarseSetup {
    /// No coarse space was requested; this is the default one-level mode.
    NotRequested,
    /// The requested coarse space was assembled and factored successfully.
    Attached,
    /// Coarse setup was requested but could not produce a usable space, so
    /// the preconditioner is operating in one-level mode.
    FellBackToOneLevel,
}

/// Runtime counters and coarse-space provenance for additive Schwarz.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchwarzRuntimeDiagnostics {
    pub apply_calls: u64,
    pub transpose_calls: u64,
    pub hermitian_calls: u64,
    pub failed_applications: u64,
    pub non_finite_inputs: u64,
    pub non_finite_outputs: u64,
    pub coarse_correction_calls: u64,
    pub coarse_solve_failures: u64,
    pub coarse_setup: SchwarzCoarseSetup,
    pub coarse_dimension: usize,
}

#[derive(Debug, Default)]
struct SchwarzRuntimeState {
    apply_calls: AtomicU64,
    transpose_calls: AtomicU64,
    hermitian_calls: AtomicU64,
    failed_applications: AtomicU64,
    non_finite_inputs: AtomicU64,
    non_finite_outputs: AtomicU64,
    coarse_correction_calls: AtomicU64,
    coarse_solve_failures: AtomicU64,
}

/// Additive Schwarz preconditioner with overlapping subdomains
#[derive(Debug, Clone)]
pub struct AdditiveSchwarzPreconditioner<T: ComplexField> {
    /// Subdomain information
    subdomains: Vec<Subdomain<T>>,
    /// Weights for combining solutions (1/count of subdomains containing each DOF)
    weights: Vec<T>,
    /// Total matrix dimension
    n: usize,
    /// Disjoint core partitions (before overlap extension), kept so a
    /// Nicolaides coarse space can be attached after construction.
    core_partitions: Vec<Vec<usize>>,
    /// Optional coarse space for the two-level variant (attached via
    /// [`AdditiveSchwarzPreconditioner::with_nicolaides_coarse_space`]).
    coarse: Option<Arc<CoarseSpace<T>>>,
    /// Observable status for the most recent optional coarse-space setup.
    coarse_setup: SchwarzCoarseSetup,
    runtime: Arc<SchwarzRuntimeState>,
}

#[cfg(feature = "cuda")]
#[path = "cuda_schwarz.rs"]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::CudaSchwarzPreconditioner;

/// Coarse space for the two-level additive Schwarz variant: an `n × nc`
/// coarse basis `Z` together with the sparse LU factorization of the
/// Galerkin coarse operator `E = ZᵀAZ`.
///
/// Two constructions are available, both opt-in and mutually exclusive
/// (attaching one replaces the other):
///
/// * Nicolaides ([`AdditiveSchwarzPreconditioner::with_nicolaides_coarse_space`]):
///   one constant vector per (disjoint) core subdomain, restricted to that
///   subdomain's support. Because the core partitions are disjoint, the
///   columns of `Z` have exactly one nonzero per row and `range(Z)` is
///   precisely the space of piecewise-constant-per-subdomain modes.
/// * GenEO ([`AdditiveSchwarzPreconditioner::with_geneo_coarse_space`]):
///   spectral modes from per-subdomain generalized eigenproblems, extended
///   by zero and weighted by the partition of unity.
///
/// In both cases `E = ZᵀAZ` is assembled with the Gustavson SpGEMM in
/// [`CsrMatrix::matmul`] and factored with COLAMD sparse LU in
/// [`SparseLuFactorization`]. The normal, transpose, and Hermitian factors
/// are retained so every advertised adjoint action uses the exact coarse
/// operator rather than falling back to an unsupported path.
#[derive(Debug)]
struct CoarseSpace<T: ComplexField> {
    /// `n × nc` coarse basis (`nc` = number of non-empty core subdomains).
    z: CsrMatrix<T>,
    /// Sparse LU factorization of the coarse operator `E = ZᵀAZ`.
    e_lu: SparseLuFactorization<T>,
    e_transpose_lu: SparseLuFactorization<T>,
    e_hermitian_lu: SparseLuFactorization<T>,
}

impl<T: ComplexField> CoarseSpace<T> {
    /// Coarse dimension (number of subdomains that contributed a column).
    fn dim(&self) -> usize {
        self.z.num_cols
    }

    /// Accumulate the coarse correction `y += Z·E⁻¹·Zᵀ·r`.
    ///
    /// `Z` is real-valued for both coarse-space constructions, so the plain
    /// transpose `Zᵀ` equals the adjoint even for complex `T`; the transpose
    /// also matches the Galerkin product `ZᵀAZ` used at setup (the right
    /// convention for complex-symmetric Helmholtz).
    fn apply_add(&self, r: &[T], y: &mut [T]) -> bool {
        self.apply_add_mode(r, y, CoarseSolveMode::Normal)
    }

    fn apply_transpose_add(&self, r: &[T], y: &mut [T]) -> bool {
        self.apply_add_mode(r, y, CoarseSolveMode::Transpose)
    }

    fn apply_hermitian_add(&self, r: &[T], y: &mut [T]) -> bool {
        self.apply_add_mode(r, y, CoarseSolveMode::Hermitian)
    }

    fn apply_add_mode(&self, r: &[T], y: &mut [T], mode: CoarseSolveMode) -> bool {
        let coarse_rhs = match mode {
            CoarseSolveMode::Normal | CoarseSolveMode::Transpose => self.z.matvec_transpose(r),
            CoarseSolveMode::Hermitian => self.z.matvec_hermitian(r),
        };
        let coarse_solution = match mode {
            CoarseSolveMode::Normal => self.e_lu.solve(&coarse_rhs),
            CoarseSolveMode::Transpose => self.e_transpose_lu.solve(&coarse_rhs),
            CoarseSolveMode::Hermitian => self.e_hermitian_lu.solve(&coarse_rhs),
        };
        if let Ok(coarse_solution) = coarse_solution {
            self.z.matvec_add(&coarse_solution, y);
            true
        } else {
            false
        }
    }
}

fn schwarz_checkpoint_push_coarse<T: ComplexField>(
    output: &mut Vec<u8>,
    coarse: &CoarseSpace<T>,
) -> Result<(), SolverError> {
    if coarse.z.num_rows == 0 || coarse.z.num_cols == 0 {
        return Err(schwarz_checkpoint_error(
            "coarse basis must have non-zero dimensions",
        ));
    }
    if coarse.e_lu.dimension() != coarse.z.num_cols
        || coarse.e_transpose_lu.dimension() != coarse.z.num_cols
        || coarse.e_hermitian_lu.dimension() != coarse.z.num_cols
    {
        return Err(schwarz_checkpoint_error(
            "coarse LU dimensions do not match coarse basis",
        ));
    }
    schwarz_checkpoint_push_csr(output, &coarse.z)?;
    for (label, factor) in [
        ("normal", &coarse.e_lu),
        ("transpose", &coarse.e_transpose_lu),
        ("Hermitian", &coarse.e_hermitian_lu),
    ] {
        let bytes = factor.to_checkpoint_bytes().map_err(|error| {
            schwarz_checkpoint_error(format!("coarse {label} LU serialization failed: {error:?}"))
        })?;
        schwarz_checkpoint_push_bytes(output, &bytes)?;
    }
    Ok(())
}

fn schwarz_checkpoint_read_coarse<T: ComplexField>(
    reader: &mut SchwarzCheckpointReader<'_>,
    n: usize,
) -> Result<CoarseSpace<T>, SolverError> {
    let z = schwarz_checkpoint_read_csr::<T>(reader, "coarse basis")?;
    if z.num_rows != n || z.num_cols == 0 {
        return Err(schwarz_checkpoint_error(
            "coarse basis dimensions do not match preconditioner",
        ));
    }
    let mut factors = Vec::with_capacity(3);
    for label in ["normal", "transpose", "Hermitian"] {
        let factor_len = reader.read_len(&format!("coarse {label} LU length"), 1)?;
        let factor_bytes = reader.take(factor_len)?;
        let factor =
            SparseLuFactorization::from_checkpoint_bytes(factor_bytes).map_err(|error| {
                schwarz_checkpoint_error(format!("coarse {label} LU restore failed: {error:?}"))
            })?;
        if factor.dimension() != z.num_cols {
            return Err(schwarz_checkpoint_error(format!(
                "coarse {label} LU dimension does not match coarse basis"
            )));
        }
        factors.push(factor);
    }
    let mut factors = factors.into_iter();
    let e_lu = factors
        .next()
        .ok_or_else(|| schwarz_checkpoint_error("missing normal coarse LU factor"))?;
    let e_transpose_lu = factors
        .next()
        .ok_or_else(|| schwarz_checkpoint_error("missing transpose coarse LU factor"))?;
    let e_hermitian_lu = factors
        .next()
        .ok_or_else(|| schwarz_checkpoint_error("missing Hermitian coarse LU factor"))?;
    Ok(CoarseSpace {
        z,
        e_lu,
        e_transpose_lu,
        e_hermitian_lu,
    })
}

#[derive(Debug, Clone, Copy)]
enum CoarseSolveMode {
    Normal,
    Transpose,
    Hermitian,
}

/// Build a Nicolaides coarse space for `matrix` from disjoint core
/// partitions. Returns `None` (fall back to one-level) when the input is
/// degenerate or the coarse operator `E = ZᵀAZ` is singular.
fn build_nicolaides_coarse_space<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    core_partitions: &[Vec<usize>],
    n: usize,
) -> Option<CoarseSpace<T>> {
    if n == 0 || matrix.num_rows != n {
        return None;
    }
    let partitions: Vec<&Vec<usize>> = core_partitions.iter().filter(|p| !p.is_empty()).collect();
    if partitions.is_empty() {
        return None;
    }
    let nc = partitions.len();

    // Z and Zᵀ share the same support: column j is 1 on core subdomain j.
    let mut z_triplets = Vec::with_capacity(n);
    let mut zt_triplets = Vec::with_capacity(n);
    for (j, core) in partitions.iter().enumerate() {
        for &i in core.iter() {
            z_triplets.push((i, j, T::one()));
            zt_triplets.push((j, i, T::one()));
        }
    }
    let z = CsrMatrix::from_triplets(n, nc, z_triplets);
    let zt = CsrMatrix::from_triplets(nc, n, zt_triplets);

    factorize_coarse_operator(matrix, z, zt)
}

/// Assemble the Galerkin coarse operator `E = ZᵀAZ` via the Gustavson
/// SpGEMM and factor its normal, transpose, and Hermitian forms with COLAMD
/// sparse LU. Returns `None` (fall back to one-level) when any required
/// factorization is singular.
fn transpose_coarse_matrix<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    conjugate: bool,
) -> CsrMatrix<T> {
    let mut triplets = Vec::with_capacity(matrix.values.len());
    for row in 0..matrix.num_rows {
        for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            let value = if conjugate {
                matrix.values[index].conj()
            } else {
                matrix.values[index]
            };
            triplets.push((matrix.col_indices[index], row, value));
        }
    }
    CsrMatrix::from_triplets(matrix.num_cols, matrix.num_rows, triplets)
}

fn factorize_coarse_operator<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    z: CsrMatrix<T>,
    zt: CsrMatrix<T>,
) -> Option<CoarseSpace<T>> {
    let e = zt.matmul(&matrix.matmul(&z));
    let e_lu = SparseLuFactorization::factorize(&e).ok()?;
    let e_transpose = transpose_coarse_matrix(&e, false);
    let e_transpose_lu = SparseLuFactorization::factorize(&e_transpose).ok()?;
    let e_hermitian = transpose_coarse_matrix(&e, true);
    let e_hermitian_lu = SparseLuFactorization::factorize(&e_hermitian).ok()?;

    Some(CoarseSpace {
        z,
        e_lu,
        e_transpose_lu,
        e_hermitian_lu,
    })
}

/// Configuration for the GenEO (Generalized Eigenproblems in the Overlap,
/// Spillane et al. 2014) coarse space; see
/// [`AdditiveSchwarzPreconditioner::with_geneo_coarse_space`] for the exact
/// formulation solved here.
#[derive(Debug, Clone)]
pub struct GeneoConfig {
    /// Eigenvalue threshold τ: keep local eigenpairs with `λ < τ`.
    /// Default `0.4`, a conservative threshold for indefinite Helmholtz.
    pub tau: f64,
    /// Cap on coarse modes kept per subdomain. Also the LOBPCG block size:
    /// the eigensolver computes this many smallest local eigenpairs and the
    /// threshold `tau` keeps the ones below it. Default `16`.
    pub max_modes_per_subdomain: usize,
    /// Relative-residual tolerance for each local LOBPCG eigensolve.
    /// Default `1e-6`.
    pub lobpcg_tolerance: f64,
    /// Iteration cap for each local LOBPCG eigensolve. Default `150`.
    pub lobpcg_max_iterations: usize,
}

impl Default for GeneoConfig {
    fn default() -> Self {
        Self {
            tau: 0.4,
            max_modes_per_subdomain: 16,
            lobpcg_tolerance: 1e-6,
            lobpcg_max_iterations: 150,
        }
    }
}

/// Failure reported by the checked GenEO coarse-space setup API.
///
/// The infallible [`AdditiveSchwarzPreconditioner::with_geneo_coarse_space`]
/// compatibility builder maps these failures to its historical one-level
/// fallback. Callers that need to distinguish invalid input, numerical setup
/// failures, and a singular Galerkin operator should use
/// [`AdditiveSchwarzPreconditioner::try_with_geneo_coarse_space`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GeneoSetupError {
    #[error("invalid GenEO input: {0}")]
    InvalidInput(String),
    #[error("invalid GenEO configuration: {0}")]
    InvalidConfig(String),
    #[error("GenEO produced no eligible coarse modes")]
    EmptyCoarseSpace,
    #[error("GenEO generated non-finite values during {stage} (subdomain: {subdomain:?})")]
    GeneratedNonFinite {
        stage: &'static str,
        subdomain: Option<usize>,
    },
    #[error("GenEO local eigensolver failed for subdomain {subdomain}: {reason}")]
    LocalEigenproblem { subdomain: usize, reason: String },
    #[error("GenEO coarse-factorization failed while building {stage}")]
    CoarseFactorization { stage: &'static str },
    #[error("GenEO setup panicked unexpectedly")]
    Panicked,
}

fn validate_geneo_config(config: &GeneoConfig) -> Result<(), GeneoSetupError> {
    if !config.tau.is_finite() {
        return Err(GeneoSetupError::InvalidConfig("tau must be finite".into()));
    }
    if config.max_modes_per_subdomain == 0 {
        return Err(GeneoSetupError::InvalidConfig(
            "max_modes_per_subdomain must be positive".into(),
        ));
    }
    if !config.lobpcg_tolerance.is_finite() || config.lobpcg_tolerance <= 0.0 {
        return Err(GeneoSetupError::InvalidConfig(
            "lobpcg_tolerance must be finite and positive".into(),
        ));
    }
    if config.lobpcg_max_iterations == 0 {
        return Err(GeneoSetupError::InvalidConfig(
            "lobpcg_max_iterations must be positive".into(),
        ));
    }
    Ok(())
}

/// Build a GenEO coarse space for `matrix`: solve one generalized
/// eigenproblem per (extended) subdomain, keep the eigenvectors with
/// eigenvalue below `config.tau`, and extend them by zero weighted by the
/// partition of unity. Returns `None` (fall back to one-level) when no
/// subdomain contributes a mode or the coarse operator `E = ZᵀAZ` is
/// singular.
///
/// The local eigenproblems are independent and solved in parallel. Nested
/// Rayon work is safe because sonium-compute's SME2 GEMM pack-buffer cache
/// releases its thread-local RefCell borrow before entering `rayon::scope`.
#[allow(dead_code)]
fn build_geneo_coarse_space<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    subdomains: &[Subdomain<T>],
    weights: &[T],
    n: usize,
    config: &GeneoConfig,
) -> Option<CoarseSpace<T>> {
    if n == 0
        || matrix.num_rows != n
        || matrix.num_cols != n
        || subdomains.is_empty()
        || matrix.validate_structure().is_err()
        || matrix.values.iter().any(|value| !value.norm().is_finite())
    {
        return None;
    }

    let local_modes: Vec<Vec<Vec<f64>>> = subdomains
        .par_iter()
        .map(|subdomain| geneo_local_modes(matrix, subdomain, weights, config))
        .collect();
    let nc: usize = local_modes.iter().map(Vec::len).sum();
    if nc == 0 {
        return None;
    }

    // GenEO partition-of-unity extension: each kept local eigenvector v of
    // subdomain i becomes one global coarse column D_i·v extended by zero,
    // where D_i is the inverse-overlap-count weight on the subdomain's
    // support. Z is real-valued; Zᵀ shares its support.
    let mut z_triplets = Vec::new();
    let mut zt_triplets = Vec::new();
    let mut col = 0_usize;
    for (subdomain, modes) in subdomains.iter().zip(&local_modes) {
        for v in modes {
            for (local_idx, &global_idx) in subdomain.global_indices.iter().enumerate() {
                let w = weights[global_idx].re().to_f64().unwrap_or(1.0);
                let value = T::from_real(T::Real::from_f64(w * v[local_idx])?);
                z_triplets.push((global_idx, col, value));
                zt_triplets.push((col, global_idx, value));
            }
            col += 1;
        }
    }
    let z = CsrMatrix::from_triplets(n, nc, z_triplets);
    let zt = CsrMatrix::from_triplets(nc, n, zt_triplets);

    factorize_coarse_operator(matrix, z, zt)
}

fn build_geneo_coarse_space_checked<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    subdomains: &[Subdomain<T>],
    weights: &[T],
    n: usize,
    config: &GeneoConfig,
) -> Result<CoarseSpace<T>, GeneoSetupError> {
    validate_geneo_config(config)?;
    if n == 0 {
        return Err(GeneoSetupError::InvalidInput(
            "matrix must have positive dimension".into(),
        ));
    }
    if matrix.num_rows != n || matrix.num_cols != n {
        return Err(GeneoSetupError::InvalidInput(format!(
            "matrix dimensions {}x{} do not match preconditioner dimension {n}",
            matrix.num_rows, matrix.num_cols
        )));
    }
    matrix
        .validate_structure()
        .map_err(|_| GeneoSetupError::InvalidInput("matrix CSR structure is invalid".into()))?;
    if matrix.values.iter().any(|value| !value.norm().is_finite()) {
        return Err(GeneoSetupError::InvalidInput(
            "matrix values must be finite".into(),
        ));
    }
    let square_safe_limit = f64::MAX.sqrt();
    if matrix.values.iter().any(|value| {
        value
            .re()
            .to_f64()
            .is_some_and(|real| real.abs() > square_safe_limit)
    }) {
        return Err(GeneoSetupError::GeneratedNonFinite {
            stage: "local eigenproblem whitening",
            subdomain: None,
        });
    }
    if subdomains.is_empty() {
        return Err(GeneoSetupError::InvalidInput(
            "at least one subdomain is required".into(),
        ));
    }
    if weights.len() != n {
        return Err(GeneoSetupError::InvalidInput(format!(
            "weight length {} does not match matrix dimension {n}",
            weights.len()
        )));
    }
    for (index, weight) in weights.iter().enumerate() {
        let value = weight.re().to_f64().ok_or_else(|| {
            GeneoSetupError::InvalidInput(format!("weight {index} is not representable as f64"))
        })?;
        if !value.is_finite() || value <= 0.0 {
            return Err(GeneoSetupError::InvalidInput(format!(
                "weight {index} must be finite and positive"
            )));
        }
    }

    // Validate the topology and preflight the only generated sum that can
    // overflow even when all input matrix entries are finite.
    for (subdomain_index, subdomain) in subdomains.iter().enumerate() {
        if subdomain.global_indices.is_empty() {
            return Err(GeneoSetupError::InvalidInput(format!(
                "subdomain {subdomain_index} is empty"
            )));
        }
        let mut global_to_local = vec![usize::MAX; n];
        for (local_index, &global_index) in subdomain.global_indices.iter().enumerate() {
            if global_index >= n || global_to_local[global_index] != usize::MAX {
                return Err(GeneoSetupError::InvalidInput(format!(
                    "subdomain {subdomain_index} contains an invalid or repeated DOF"
                )));
            }
            global_to_local[global_index] = local_index;
        }
        for &global_row in &subdomain.global_indices {
            let mut dropped = 0.0_f64;
            for index in matrix.row_ptrs[global_row]..matrix.row_ptrs[global_row + 1] {
                let global_col = matrix.col_indices[index];
                if global_to_local[global_col] == usize::MAX && global_col != global_row {
                    let value = matrix.values[index].re().to_f64().ok_or(
                        GeneoSetupError::GeneratedNonFinite {
                            stage: "local Neumann assembly",
                            subdomain: Some(subdomain_index),
                        },
                    )?;
                    dropped += value;
                    if !dropped.is_finite() {
                        return Err(GeneoSetupError::GeneratedNonFinite {
                            stage: "local Neumann assembly",
                            subdomain: Some(subdomain_index),
                        });
                    }
                }
            }
        }
    }

    let local_modes: Vec<Vec<Vec<f64>>> = subdomains
        .par_iter()
        .map(|subdomain| geneo_local_modes(matrix, subdomain, weights, config))
        .collect();
    for (subdomain_index, modes) in local_modes.iter().enumerate() {
        if modes.iter().flatten().any(|value| !value.is_finite()) {
            return Err(GeneoSetupError::GeneratedNonFinite {
                stage: "local eigenvector assembly",
                subdomain: Some(subdomain_index),
            });
        }
    }
    let nc: usize = local_modes.iter().map(Vec::len).sum();
    if nc == 0 {
        return Err(GeneoSetupError::EmptyCoarseSpace);
    }

    let mut z_triplets = Vec::new();
    let mut zt_triplets = Vec::new();
    let mut column = 0_usize;
    for (subdomain, modes) in subdomains.iter().zip(&local_modes) {
        for mode in modes {
            if mode.len() != subdomain.global_indices.len() {
                return Err(GeneoSetupError::LocalEigenproblem {
                    subdomain: column,
                    reason: "eigenvector dimension does not match local support".into(),
                });
            }
            for (local_index, &global_index) in subdomain.global_indices.iter().enumerate() {
                let weight = weights[global_index].re().to_f64().ok_or(
                    GeneoSetupError::GeneratedNonFinite {
                        stage: "coarse basis extension",
                        subdomain: None,
                    },
                )?;
                let scaled = weight * mode[local_index];
                if !scaled.is_finite() {
                    return Err(GeneoSetupError::GeneratedNonFinite {
                        stage: "coarse basis extension",
                        subdomain: None,
                    });
                }
                let value = T::from_real(T::Real::from_f64(scaled).ok_or(
                    GeneoSetupError::GeneratedNonFinite {
                        stage: "coarse basis extension",
                        subdomain: None,
                    },
                )?);
                z_triplets.push((global_index, column, value));
                zt_triplets.push((column, global_index, value));
            }
            column += 1;
        }
    }
    let z = CsrMatrix::try_from_triplets(n, nc, z_triplets).map_err(|_| {
        GeneoSetupError::GeneratedNonFinite {
            stage: "coarse basis construction",
            subdomain: None,
        }
    })?;
    let zt = CsrMatrix::try_from_triplets(nc, n, zt_triplets).map_err(|_| {
        GeneoSetupError::GeneratedNonFinite {
            stage: "coarse transpose construction",
            subdomain: None,
        }
    })?;
    factorize_coarse_operator_checked(matrix, z, zt)
}

fn factorize_coarse_operator_checked<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    z: CsrMatrix<T>,
    zt: CsrMatrix<T>,
) -> Result<CoarseSpace<T>, GeneoSetupError> {
    if z.values
        .iter()
        .chain(zt.values.iter())
        .any(|value| !value.norm().is_finite())
    {
        return Err(GeneoSetupError::GeneratedNonFinite {
            stage: "coarse basis construction",
            subdomain: None,
        });
    }
    let e = catch_unwind(AssertUnwindSafe(|| zt.matmul(&matrix.matmul(&z)))).map_err(|_| {
        GeneoSetupError::GeneratedNonFinite {
            stage: "coarse Galerkin assembly",
            subdomain: None,
        }
    })?;
    if e.values.iter().any(|value| !value.norm().is_finite()) {
        return Err(GeneoSetupError::GeneratedNonFinite {
            stage: "coarse Galerkin assembly",
            subdomain: None,
        });
    }
    let e_lu = SparseLuFactorization::factorize(&e)
        .map_err(|_| GeneoSetupError::CoarseFactorization { stage: "normal" })?;
    let e_transpose = transpose_coarse_matrix(&e, false);
    let e_transpose_lu = SparseLuFactorization::factorize(&e_transpose)
        .map_err(|_| GeneoSetupError::CoarseFactorization { stage: "transpose" })?;
    let e_hermitian = transpose_coarse_matrix(&e, true);
    let e_hermitian_lu = SparseLuFactorization::factorize(&e_hermitian)
        .map_err(|_| GeneoSetupError::CoarseFactorization { stage: "Hermitian" })?;
    Ok(CoarseSpace {
        z,
        e_lu,
        e_transpose_lu,
        e_hermitian_lu,
    })
}

/// Solve the GenEO generalized eigenproblem of one subdomain and return the
/// eigenvectors with eigenvalue below `config.tau` (at most
/// `config.max_modes_per_subdomain` of them), B-orthonormal.
///
/// The pencil is
///
/// ```text
/// A_i v = λ B_i v,   B_i = D_i²  (diagonal)
/// ```
///
/// where `A_i` is the real part of the local **Neumann** matrix on the
/// subdomain's overlap support and `D_i` is the diagonal of
/// partition-of-unity weights (inverse overlap counts). The Neumann matrix
/// is assembled algebraically from the *global* matrix as the principal
/// submatrix plus the dropped off-diagonal row entries folded into the
/// diagonal,
///
/// ```text
/// (A_i)_{jj} = A_{jj} + Σ_{k ∉ Ω_i} A_{jk}
/// ```
///
/// which preserves the global row sums — the action of `A` on
/// locally-constant modes — and is the algebraic analogue of natural
/// (do-nothing) boundary conditions on the artificial boundary. For the
/// 5-point Laplacian this puts the constant mode in the kernel of the
/// local stiffness part, which is exactly what GenEO theory needs; the
/// plain principal submatrix (= Dirichlet on the artificial boundary) was
/// tried first and measured *worse than one-level* GMRES on strongly
/// indefinite Helmholtz, because zero-extended Dirichlet modes have kinks
/// at the overlap boundary that poison the coarse correction. Taking `A_i`
/// from the global matrix also keeps the ORAS/HABC impedance modifications
/// of the local solver out of the eigenproblem.
///
/// Why this right-hand side: Spillane's canonical `B_i = D_i A_i D_i`
/// inherits the indefiniteness of the Helmholtz operator, while the
/// eigensolvers require an SPD `B`. `B_i = D_i²` is the lumped
/// weighted-mass choice: SPD by construction (weights are strictly positive
/// on the support), and for finite-difference / mass-lumped FEM
/// discretisations it equals the true weighted mass matrix up to the
/// constant element measure, which only rescales the eigenvalues. The kept
/// modes are the local low-frequency / negative-curvature resonances of
/// `A_i`, which is exactly what a coarse space for indefinite Helmholtz
/// must capture.
///
/// Eigensolver dispatch: below [`GENEO_DENSE_EIGEN_CUTOFF`] local DOFs the
/// pencil is whitened (`C = D_i⁻¹ A_i D_i⁻¹`, symmetric) and solved densely
/// with cyclic Jacobi — LOBPCG stagnates when the block size rivals the
/// problem size (observed spurious Ritz values and no convergence on a
/// 48-DOF block-16 solve), while the O(n³) dense solve is exact and cheap
/// there. Above the cutoff, [`lobpcg_generalized`] computes the
/// `max_modes_per_subdomain` smallest pairs; Ritz pairs whose relative
/// residual stays far above the solver tolerance are rejected so a
/// non-converged eigensolve degrades the coarse space gracefully instead of
/// polluting it.
fn geneo_local_modes<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    subdomain: &Subdomain<T>,
    weights: &[T],
    config: &GeneoConfig,
) -> Vec<Vec<f64>> {
    let local_n = subdomain.global_indices.len();
    if local_n == 0 || config.max_modes_per_subdomain == 0 {
        return Vec::new();
    }

    // Partition-of-unity weights on the support (real, strictly positive).
    let Some(w) = subdomain
        .global_indices
        .iter()
        .map(|&g| weights[g].re().to_f64())
        .collect::<Option<Vec<f64>>>()
    else {
        return Vec::new();
    };

    // A finite global matrix can still produce invalid local data when
    // dropped interface couplings overflow while forming the Neumann row.
    // Compatibility constructors must degrade to one-level Schwarz rather
    // than allowing those values to reach the infallible CSR constructor.
    if w.iter().any(|value| !value.is_finite() || *value <= 0.0) {
        return Vec::new();
    }

    // Extract the local Neumann matrix (real part, symmetric; see above).
    let mut global_to_local = vec![usize::MAX; matrix.num_rows];
    for (local_idx, &global_idx) in subdomain.global_indices.iter().enumerate() {
        global_to_local[global_idx] = local_idx;
    }
    let mut a_triplets = Vec::new();
    for (local_row, &global_row) in subdomain.global_indices.iter().enumerate() {
        let mut dropped = 0.0;
        for idx in matrix.row_ptrs[global_row]..matrix.row_ptrs[global_row + 1] {
            let global_col = matrix.col_indices[idx];
            if global_col >= matrix.num_rows {
                continue;
            }
            let Some(value) = matrix.values[idx].re().to_f64() else {
                return Vec::new();
            };
            if !value.is_finite() {
                return Vec::new();
            }
            if global_to_local[global_col] != usize::MAX {
                a_triplets.push((local_row, global_to_local[global_col], value));
            } else if global_col != global_row {
                dropped += value;
                if !dropped.is_finite() {
                    return Vec::new();
                }
            }
        }
        if dropped != 0.0 {
            a_triplets.push((local_row, local_row, dropped));
        }
    }

    if local_n <= GENEO_DENSE_EIGEN_CUTOFF {
        return geneo_local_modes_dense(local_n, &a_triplets, &w, config);
    }

    let nev = config.max_modes_per_subdomain.min(local_n / 4).max(1);
    let Ok(a) = CsrMatrix::try_from_triplets(local_n, local_n, a_triplets) else {
        return Vec::new();
    };
    let Ok(b) = CsrMatrix::try_from_triplets(
        local_n,
        local_n,
        w.iter()
            .enumerate()
            .map(|(i, &wi)| (i, i, wi * wi))
            .collect(),
    ) else {
        return Vec::new();
    };
    let lobpcg_config = LobpcgConfig {
        num_eigenvalues: nev,
        max_iterations: config.lobpcg_max_iterations,
        tolerance: config.lobpcg_tolerance,
        ..Default::default()
    };
    let result = match lobpcg_generalized(&a, &b, &lobpcg_config) {
        Ok(result) => result,
        // Projected eigensolver failed to converge: contribute no coarse
        // modes from this subdomain. Same graceful-degradation policy as the
        // residual-based rejection below — the coarse space gets weaker but
        // the preconditioner stays valid, and no spurious Ritz pairs pollute
        // it.
        Err(_) => return Vec::new(),
    };

    // Keep the eigenpairs below the GenEO threshold; reject unconverged
    // Ritz pairs (relative residual far above the solver tolerance).
    // Eigenvalues come out sorted smallest-first.
    let residual_cutoff = (config.lobpcg_tolerance * 100.0).max(1e-4);
    (0..result.eigenvalues.len())
        .filter(|&j| {
            result.eigenvalues[j] < config.tau && result.residual_norms[j] < residual_cutoff
        })
        .map(|j| (0..local_n).map(|i| result.eigenvectors[[i, j]]).collect())
        .collect()
}

/// Local problem size below which the GenEO eigenproblem is solved densely
/// (cyclic Jacobi) instead of LOBPCG — see [`geneo_local_modes`].
const GENEO_DENSE_EIGEN_CUTOFF: usize = 256;

/// Dense path for small local eigenproblems: whiten `B = D²` into
/// `C = D⁻¹AD⁻¹` (symmetrised), solve `C u = λ u` with cyclic Jacobi, and
/// map back `v = D⁻¹u` (so `vᵀBv = uᵀu = 1`). Keeps `λ < config.tau`,
/// capped at `config.max_modes_per_subdomain` modes.
fn geneo_local_modes_dense(
    local_n: usize,
    a_triplets: &[(usize, usize, f64)],
    w: &[f64],
    config: &GeneoConfig,
) -> Vec<Vec<f64>> {
    if !w.iter().all(|value| value.is_finite() && *value > 0.0)
        || !a_triplets.iter().all(|(_, _, value)| value.is_finite())
    {
        return Vec::new();
    }
    // Dense, symmetrised, whitened matrix C = D⁻¹·sym(A)·D⁻¹ (row-major).
    let mut a = vec![0.0_f64; local_n * local_n];
    for &(i, j, value) in a_triplets {
        a[i * local_n + j] += value;
    }
    let mut c = vec![0.0_f64; local_n * local_n];
    for i in 0..local_n {
        for j in 0..local_n {
            let denominator = w[i] * w[j];
            if !denominator.is_finite() || denominator <= 0.0 {
                return Vec::new();
            }
            c[i * local_n + j] = 0.5 * (a[i * local_n + j] + a[j * local_n + i]) / denominator;
            if !c[i * local_n + j].is_finite() {
                return Vec::new();
            }
        }
    }

    let mut pairs = jacobi_eigen(&mut c, local_n);
    pairs.sort_by(|lhs, rhs| {
        lhs.0
            .partial_cmp(&rhs.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    pairs
        .into_iter()
        .filter(|(lambda, _)| *lambda < config.tau)
        .take(config.max_modes_per_subdomain)
        .map(|(_, u)| u.iter().zip(w).map(|(&ui, &wi)| ui / wi).collect())
        .collect()
}

/// Cyclic Jacobi eigensolver for a small dense symmetric matrix (row-major,
/// overwritten). Returns all `(eigenvalue, eigenvector)` pairs with
/// Euclidean-normalised eigenvectors. Used for the GenEO local problems
/// below [`GENEO_DENSE_EIGEN_CUTOFF`]: robust where a block Krylov
/// eigensolver stagnates, and O(n³) is affordable at these sizes.
fn jacobi_eigen(a: &mut [f64], n: usize) -> Vec<(f64, Vec<f64>)> {
    if !a.iter().all(|value| value.is_finite()) {
        return Vec::new();
    }
    // vecs[j] is the j-th eigenvector candidate (columns of V).
    let mut vecs = vec![vec![0.0_f64; n]; n];
    for (j, vj) in vecs.iter_mut().enumerate() {
        vj[j] = 1.0;
    }
    let fro = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    if !fro.is_finite() {
        return Vec::new();
    }
    let fro = fro.max(1e-300);
    let tiny = 1e-15 * fro;

    for _sweep in 0..100 {
        let mut off_sq = 0.0_f64;
        for p in 0..n {
            for q in (p + 1)..n {
                off_sq += a[p * n + q] * a[p * n + q];
            }
        }
        if !off_sq.is_finite() {
            return Vec::new();
        }
        if off_sq.sqrt() <= 1e-13 * fro {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() <= tiny {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                if !theta.is_finite() {
                    return Vec::new();
                }
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    1.0 / (theta - (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Rotate rows/columns p and q of A.
                for k in 0..n {
                    if k == p || k == q {
                        continue;
                    }
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[p * n + k] = a[k * n + p];
                    a[k * n + q] = s * akp + c * akq;
                    a[q * n + k] = a[k * n + q];
                }
                a[p * n + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                a[q * n + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;

                // Accumulate the rotation into the eigenvector columns.
                // The index addresses two rows mutably at once; the borrow
                // checker cannot express that disjointness via iterators.
                #[expect(
                    clippy::needless_range_loop,
                    reason = "simultaneous in-place update of vecs[p] and vecs[q]; disjoint mutable row borrows are inexpressible here"
                )]
                for k in 0..n {
                    let vkp = vecs[p][k];
                    let vkq = vecs[q][k];
                    vecs[p][k] = c * vkp - s * vkq;
                    vecs[q][k] = s * vkp + c * vkq;
                }
            }
        }
    }

    if !a.iter().all(|value| value.is_finite())
        || !vecs.iter().flatten().all(|value| value.is_finite())
    {
        return Vec::new();
    }
    (0..n).map(|j| (a[j * n + j], vecs[j].clone())).collect()
}

/// A single subdomain with its local matrix and solver
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Subdomain<T: ComplexField> {
    /// Global indices of DOFs in this subdomain
    global_indices: Vec<usize>,
    /// Local matrix (extracted from global)
    local_values: Vec<T>,
    local_col_indices: Vec<usize>,
    local_row_ptrs: Vec<usize>,
    /// Local ILU factors for solving
    l_values: Vec<T>,
    l_col_indices: Vec<usize>,
    l_row_ptrs: Vec<usize>,
    u_values: Vec<T>,
    u_col_indices: Vec<usize>,
    u_row_ptrs: Vec<usize>,
    /// Pre-computed inverse diagonal with phase-preserving fallback for near-zero pivots
    u_diag_inv: Vec<T>,
    /// Setup diagnostics retained by the compatibility constructor.
    diagnostics: IluDiagnostics<T::Real>,
}

impl<T: ComplexField> AdditiveSchwarzPreconditioner<T> {
    fn validate_csr_input(matrix: &CsrMatrix<T>) -> Result<(), SolverError> {
        matrix.validate_structure()?;
        if matrix.num_rows != matrix.num_cols {
            return Err(SolverError::OperationFailed {
                operation: "Schwarz construction",
                reason: format!(
                    "matrix must be square, got {}x{}",
                    matrix.num_rows, matrix.num_cols
                ),
            });
        }
        Ok(())
    }

    /// Validate caller-provided disjoint core partitions for a checked
    /// Schwarz construction. The infallible compatibility constructors keep
    /// their historical normalize-and-filter behavior; checked callers must
    /// not silently lose ownership or repair malformed topology.
    fn validate_core_partitions(
        matrix: &CsrMatrix<T>,
        partitions: &[Vec<usize>],
    ) -> Result<(), SolverError> {
        let n = matrix.num_rows;
        if n == 0 {
            if partitions.is_empty() {
                return Ok(());
            }
            return Err(SolverError::OperationFailed {
                operation: "Schwarz partition setup",
                reason: "zero-dimensional matrix cannot have core partitions".into(),
            });
        }
        if partitions.is_empty() {
            return Err(SolverError::OperationFailed {
                operation: "Schwarz partition setup",
                reason: "at least one non-empty core partition is required".into(),
            });
        }

        let mut ownership = vec![0_u8; n];
        for (partition_index, partition) in partitions.iter().enumerate() {
            if partition.is_empty() {
                return Err(SolverError::OperationFailed {
                    operation: "Schwarz partition setup",
                    reason: format!("core partition {partition_index} is empty"),
                });
            }

            let mut sorted = partition.clone();
            sorted.sort_unstable();
            for pair in sorted.windows(2) {
                if pair[0] == pair[1] {
                    return Err(SolverError::OperationFailed {
                        operation: "Schwarz partition setup",
                        reason: format!("core partition {partition_index} repeats DOF {}", pair[0]),
                    });
                }
            }
            for &dof in &sorted {
                if dof >= n {
                    return Err(SolverError::OperationFailed {
                        operation: "Schwarz partition setup",
                        reason: format!(
                            "core partition {partition_index} contains out-of-range DOF {dof}"
                        ),
                    });
                }
                ownership[dof] = ownership[dof].saturating_add(1);
                if ownership[dof] > 1 {
                    return Err(SolverError::OperationFailed {
                        operation: "Schwarz partition setup",
                        reason: format!("DOF {dof} belongs to multiple core partitions"),
                    });
                }
            }
        }

        if let Some(uncovered) = ownership.iter().position(|&count| count == 0) {
            return Err(SolverError::OperationFailed {
                operation: "Schwarz partition setup",
                reason: format!("core partitions leave DOF {uncovered} uncovered"),
            });
        }
        Ok(())
    }

    /// Create an Additive Schwarz preconditioner
    ///
    /// # Arguments
    /// * `matrix` - The global sparse matrix
    /// * `num_subdomains` - Number of subdomains (typically = number of threads)
    /// * `overlap` - Number of overlap layers between subdomains
    pub fn from_csr(matrix: &CsrMatrix<T>, num_subdomains: usize, overlap: usize) -> Self {
        Self::from_csr_with_transmission(matrix, num_subdomains, overlap, None)
    }

    pub fn try_from_csr(
        matrix: &CsrMatrix<T>,
        num_subdomains: usize,
        overlap: usize,
    ) -> Result<Self, SolverError> {
        Self::validate_csr_input(matrix)?;
        Self::try_from_csr_with_transmission(matrix, num_subdomains, overlap, None)
    }

    /// Checked automatic constructor using the same bounded subdomain-count
    /// policy as [`Self::from_csr_auto`].
    ///
    /// Unlike the compatibility constructor, malformed CSR input and setup
    /// failures are returned as a typed [`SolverError`] instead of being
    /// allowed to panic during local-factor construction.
    pub fn try_from_csr_auto(matrix: &CsrMatrix<T>, overlap: usize) -> Result<Self, SolverError> {
        let num_subdomains = auto_subdomain_count(matrix.num_rows);
        Self::try_from_csr(matrix, num_subdomains, overlap)
    }

    /// Create an optimized Schwarz preconditioner with first-order absorbing
    /// Robin transmission conditions on artificial subdomain interfaces.
    ///
    /// This is an algebraic ORAS variant for assembled CSR systems: whenever a
    /// local subdomain row drops couplings to DOFs outside the overlap, the
    /// local diagonal receives an `i k` impedance penalty scaled by the dropped
    /// coupling strength. For geometry-exact FEM Robin terms, the assembler
    /// should provide subdomain interface faces and face mass contributions.
    pub fn from_csr_oras(
        matrix: &CsrMatrix<T>,
        num_subdomains: usize,
        overlap: usize,
        wavenumber: f64,
    ) -> Self {
        Self::from_csr_with_transmission(
            matrix,
            num_subdomains,
            overlap,
            Some(TransmissionCondition {
                wavenumber,
                higher_order: false,
            }),
        )
    }

    pub fn try_from_csr_oras(
        matrix: &CsrMatrix<T>,
        num_subdomains: usize,
        overlap: usize,
        wavenumber: f64,
    ) -> Result<Self, SolverError> {
        if !wavenumber.is_finite() || wavenumber <= 0.0 {
            return Err(SolverError::OperationFailed {
                operation: "Schwarz ORAS setup",
                reason: "wavenumber must be finite and positive".into(),
            });
        }
        Self::validate_csr_input(matrix)?;
        Self::try_from_csr_with_transmission(
            matrix,
            num_subdomains,
            overlap,
            Some(TransmissionCondition {
                wavenumber,
                higher_order: false,
            }),
        )
    }

    /// Create an optimized Schwarz preconditioner with a second-order
    /// HABC-style correction on artificial subdomain interfaces.
    pub fn from_csr_habc(
        matrix: &CsrMatrix<T>,
        num_subdomains: usize,
        overlap: usize,
        wavenumber: f64,
    ) -> Self {
        Self::from_csr_with_transmission(
            matrix,
            num_subdomains,
            overlap,
            Some(TransmissionCondition {
                wavenumber,
                higher_order: true,
            }),
        )
    }

    pub fn try_from_csr_habc(
        matrix: &CsrMatrix<T>,
        num_subdomains: usize,
        overlap: usize,
        wavenumber: f64,
    ) -> Result<Self, SolverError> {
        if !wavenumber.is_finite() || wavenumber <= 0.0 {
            return Err(SolverError::OperationFailed {
                operation: "Schwarz HABC setup",
                reason: "wavenumber must be finite and positive".into(),
            });
        }
        Self::validate_csr_input(matrix)?;
        Self::try_from_csr_with_transmission(
            matrix,
            num_subdomains,
            overlap,
            Some(TransmissionCondition {
                wavenumber,
                higher_order: true,
            }),
        )
    }

    fn from_csr_with_transmission(
        matrix: &CsrMatrix<T>,
        num_subdomains: usize,
        overlap: usize,
        transmission: Option<TransmissionCondition>,
    ) -> Self {
        let n = matrix.num_rows;
        let num_subdomains = num_subdomains.max(1).min(n);

        // Step 1: Create initial partition (contiguous blocks)
        let mut partitions = Vec::with_capacity(num_subdomains);
        if n > 0 {
            let base_size = n / num_subdomains;
            let remainder = n % num_subdomains;

            let mut start = 0;
            for i in 0..num_subdomains {
                let size = base_size + if i < remainder { 1 } else { 0 };
                partitions.push((start..start + size).collect());
                start += size;
            }
        }

        Self::from_csr_partitions_with_transmission(matrix, &partitions, overlap, transmission)
    }

    fn try_from_csr_with_transmission(
        matrix: &CsrMatrix<T>,
        num_subdomains: usize,
        overlap: usize,
        transmission: Option<TransmissionCondition>,
    ) -> Result<Self, SolverError> {
        let n = matrix.num_rows;
        let num_subdomains = num_subdomains.max(1).min(n);

        let mut partitions = Vec::with_capacity(num_subdomains);
        if n > 0 {
            let base_size = n / num_subdomains;
            let remainder = n % num_subdomains;

            let mut start = 0;
            for i in 0..num_subdomains {
                let size = base_size + if i < remainder { 1 } else { 0 };
                partitions.push((start..start + size).collect());
                start += size;
            }
        }

        Self::try_from_csr_partitions_with_transmission(matrix, &partitions, overlap, transmission)
    }

    /// Create an Additive Schwarz preconditioner from geometry-provided core
    /// subdomain DOF sets.
    ///
    /// The supplied partitions are extended by the usual graph-overlap layers,
    /// then weighted by overlap count. This is the entry point to use when a
    /// FEM assembler has also decorated the matching artificial interface faces
    /// with Robin/HABC transmission terms.
    pub fn from_csr_partitions(
        matrix: &CsrMatrix<T>,
        partitions: &[Vec<usize>],
        overlap: usize,
    ) -> Self {
        Self::from_csr_partitions_with_transmission(matrix, partitions, overlap, None)
    }

    pub fn try_from_csr_partitions(
        matrix: &CsrMatrix<T>,
        partitions: &[Vec<usize>],
        overlap: usize,
    ) -> Result<Self, SolverError> {
        Self::validate_csr_input(matrix)?;
        Self::try_from_csr_partitions_with_transmission(matrix, partitions, overlap, None)
    }

    /// Create an ORAS preconditioner from geometry-provided core subdomain
    /// DOF sets, with first-order absorbing Robin transmission conditions on
    /// the artificial subdomain interfaces (see [`Self::from_csr_oras`]).
    pub fn from_csr_partitions_oras(
        matrix: &CsrMatrix<T>,
        partitions: &[Vec<usize>],
        overlap: usize,
        wavenumber: f64,
    ) -> Self {
        Self::from_csr_partitions_with_transmission(
            matrix,
            partitions,
            overlap,
            Some(TransmissionCondition {
                wavenumber,
                higher_order: false,
            }),
        )
    }

    /// Checked ORAS construction with a first-order absorbing transmission
    /// condition on artificial interfaces.
    ///
    /// Unlike the historical infallible wrapper, this entry point rejects a
    /// non-finite or non-positive wavenumber before local matrices are built.
    pub fn try_from_csr_partitions_oras(
        matrix: &CsrMatrix<T>,
        partitions: &[Vec<usize>],
        overlap: usize,
        wavenumber: f64,
    ) -> Result<Self, SolverError> {
        if !wavenumber.is_finite() || wavenumber <= 0.0 {
            return Err(SolverError::OperationFailed {
                operation: "Schwarz ORAS setup",
                reason: "wavenumber must be finite and positive".into(),
            });
        }
        Self::validate_csr_input(matrix)?;
        Self::try_from_csr_partitions_with_transmission(
            matrix,
            partitions,
            overlap,
            Some(TransmissionCondition {
                wavenumber,
                higher_order: false,
            }),
        )
    }

    fn from_csr_partitions_with_transmission(
        matrix: &CsrMatrix<T>,
        partitions: &[Vec<usize>],
        overlap: usize,
        transmission: Option<TransmissionCondition>,
    ) -> Self {
        Self::build_from_csr_partitions_with_transmission(
            matrix,
            partitions,
            overlap,
            transmission,
            false,
        )
        .unwrap_or_else(|error| panic!("invalid local Schwarz setup: {error}"))
    }

    fn try_from_csr_partitions_with_transmission(
        matrix: &CsrMatrix<T>,
        partitions: &[Vec<usize>],
        overlap: usize,
        transmission: Option<TransmissionCondition>,
    ) -> Result<Self, SolverError> {
        Self::build_from_csr_partitions_with_transmission(
            matrix,
            partitions,
            overlap,
            transmission,
            true,
        )
    }

    fn build_from_csr_partitions_with_transmission(
        matrix: &CsrMatrix<T>,
        partitions: &[Vec<usize>],
        overlap: usize,
        transmission: Option<TransmissionCondition>,
        checked_local_setup: bool,
    ) -> Result<Self, SolverError> {
        if checked_local_setup {
            Self::validate_core_partitions(matrix, partitions)?;
        }
        let n = matrix.num_rows;
        if n == 0 || partitions.is_empty() {
            return Ok(Self {
                subdomains: Vec::new(),
                weights: Vec::new(),
                n,
                core_partitions: Vec::new(),
                coarse: None,
                coarse_setup: SchwarzCoarseSetup::NotRequested,
                runtime: Arc::new(SchwarzRuntimeState::default()),
            });
        }

        // Step 2: Build adjacency information for overlap extension
        let adjacency = build_adjacency(matrix);

        // Step 3: Clean the core partitions (in-range, sorted, deduplicated),
        // then extend each by overlap layers. Cores whose extension is empty
        // (i.e. empty cores) are dropped from both views.
        let cleaned_cores: Vec<Vec<usize>> = partitions
            .par_iter()
            .map(|partition| {
                let mut core: Vec<usize> =
                    partition.iter().copied().filter(|&idx| idx < n).collect();
                core.sort_unstable();
                core.dedup();
                core
            })
            .collect();
        let (core_partitions, extended_partitions): (Vec<Vec<usize>>, Vec<Vec<usize>>) =
            cleaned_cores
                .into_par_iter()
                .map(|core| {
                    let extended = extend_partition(&core, &adjacency, overlap, n);
                    (core, extended)
                })
                .filter(|(_, extended)| !extended.is_empty())
                .unzip();
        if extended_partitions.is_empty() {
            return Ok(Self {
                subdomains: Vec::new(),
                weights: vec![T::one(); n],
                n,
                core_partitions: Vec::new(),
                coarse: None,
                coarse_setup: SchwarzCoarseSetup::NotRequested,
                runtime: Arc::new(SchwarzRuntimeState::default()),
            });
        }

        // Step 4: Compute weights (inverse of overlap count)
        let mut overlap_count = vec![0usize; n];
        for partition in &extended_partitions {
            for &idx in partition {
                overlap_count[idx] += 1;
            }
        }
        let weights: Vec<T> = overlap_count
            .iter()
            .map(|&c| {
                if c > 0 {
                    T::from_real(T::Real::one() / T::Real::from_usize(c).unwrap())
                } else {
                    T::one()
                }
            })
            .collect();

        // Step 5: Build subdomains with local matrices and ILU factorizations
        let subdomains: Vec<Subdomain<T>> = extended_partitions
            .into_par_iter()
            .map(|indices| build_subdomain(matrix, indices, transmission, checked_local_setup))
            .collect::<Result<_, _>>()?;

        Ok(Self {
            subdomains,
            weights,
            n,
            core_partitions,
            coarse: None,
            coarse_setup: SchwarzCoarseSetup::NotRequested,
            runtime: Arc::new(SchwarzRuntimeState::default()),
        })
    }

    /// Attach a Nicolaides coarse space, turning this into a two-level
    /// additive Schwarz preconditioner.
    ///
    /// The coarse space `Z` holds one constant vector per disjoint core
    /// subdomain, restricted to that subdomain's support, so `range(Z)`
    /// contains exactly the piecewise-constant-per-subdomain modes. The
    /// coarse operator `E = ZᵀAZ` is assembled with the Gustavson SpGEMM
    /// ([`CsrMatrix::matmul`]) and factored once with the COLAMD sparse LU
    /// ([`SparseLuFactorization`]).
    ///
    /// The apply becomes the additive two-level combination
    ///
    /// ```text
    /// M⁻¹r = Σᵢ RᵢᵀBᵢ⁻¹Rᵢr + Z·E⁻¹·Zᵀr
    /// ```
    ///
    /// i.e. the coarse correction uses the same residual `r` as the local
    /// solves and is added after them (plain additive, not hybrid/deflated
    /// sequencing). For SPD problems this keeps `M⁻¹` SPD; for Helmholtz
    /// with ORAS/HABC local solves it removes the global low-frequency
    /// bottleneck, so Krylov iterations stop growing with subdomain count.
    ///
    /// # Scope and limitations
    ///
    /// Constant-per-subdomain (Nicolaides) modes are the right coarse space
    /// for SPD / diffusion-type problems and for *mildly* indefinite
    /// Helmholtz (few negative eigenvalues, sub-wavelength subdomains);
    /// measured GMRES counts are flat vs subdomain count there. For
    /// strongly indefinite Helmholtz (multiple wavelengths per subdomain)
    /// they are the *wrong* coarse space — the correction is neutral to
    /// harmful — and a spectral coarse space (GenEO) or plane-wave coarse
    /// space is required instead; that is future work. Also note `E = ZᵀAZ`
    /// can be exactly singular on undamped indefinite problems, in which
    /// case the coarse space is dropped (see below).
    ///
    /// If the coarse operator is singular (e.g. a subdomain fully decoupled
    /// from the rest of the mesh, or an indefinite `A` whose Galerkin
    /// projection hits a zero pivot) the coarse space is dropped and the
    /// preconditioner silently stays one-level.
    ///
    /// Works on top of every constructor (ASM, ORAS, HABC, auto, explicit
    /// partitions); defaults are unchanged — without this call the
    /// preconditioner remains one-level. Mutually exclusive with
    /// [`Self::with_geneo_coarse_space`]: attaching one coarse space
    /// replaces the other (last builder call wins).
    pub fn with_nicolaides_coarse_space(mut self, matrix: &CsrMatrix<T>) -> Self {
        self.coarse =
            build_nicolaides_coarse_space(matrix, &self.core_partitions, self.n).map(Arc::new);
        self.coarse_setup = if self.coarse.is_some() {
            SchwarzCoarseSetup::Attached
        } else {
            SchwarzCoarseSetup::FellBackToOneLevel
        };
        self
    }

    /// Attach a GenEO (Generalized Eigenproblems in the Overlap, Spillane et
    /// al. 2014) spectral coarse space, turning this into a two-level
    /// additive Schwarz preconditioner.
    ///
    /// Per (extended) subdomain `i` the generalized eigenproblem
    ///
    /// ```text
    /// A_i v = λ B_i v,   B_i = D_i²  (diagonal)
    /// ```
    ///
    /// is solved for the smallest eigenpairs: `A_i` is the real part of the
    /// local Neumann matrix on the subdomain's overlap support — the
    /// principal submatrix of the unmodified global `A` with the dropped
    /// off-diagonal row entries folded into the diagonal (row-sum
    /// preserving, the algebraic analogue of natural boundary conditions on
    /// the artificial boundary) — and `D_i` is the diagonal of
    /// partition-of-unity weights. Eigenvectors with `λ < config.tau` (at
    /// most `config.max_modes_per_subdomain` per subdomain) are kept, and
    /// each becomes one coarse column `D_i·v` extended by zero — the GenEO
    /// partition-of-unity extension. `E = ZᵀAZ` is assembled with the
    /// Gustavson SpGEMM and factored once with the COLAMD sparse LU, exactly
    /// like the Nicolaides path; a singular `E` (or an empty coarse space)
    /// silently drops back to one-level.
    ///
    /// # Formulation notes
    ///
    /// * The right-hand side is the lumped weighted-mass matrix `D_i²`,
    ///   **not** Spillane's `D_i A_i D_i`: the latter is indefinite for
    ///   Helmholtz and the eigensolvers require an SPD `B`. See
    ///   [`geneo_local_modes`] for details, including why the Neumann (not
    ///   Dirichlet) local matrix is essential and when the dense Jacobi
    ///   eigensolve replaces LOBPCG.
    /// * The eigensolvers are `f64`-only, so the eigenproblem is solved on
    ///   the real part of `A` and the (real) eigenvectors are embedded into
    ///   `T` via [`ComplexField::from_real`]. For complex-symmetric
    ///   Helmholtz with real coefficients — the target regime — nothing is
    ///   lost; for genuinely complex `A` the coarse space sees only the
    ///   symmetric real part. No separate `f64` impl block is needed.
    /// * Unlike constant-per-subdomain (Nicolaides) modes, the spectral
    ///   coarse space captures the local resonance modes of strongly
    ///   indefinite Helmholtz (multiple wavelengths per domain); measured
    ///   GMRES counts beat one-level and Nicolaides two-level there and grow
    ///   much more slowly with subdomain count (see the in-file test
    ///   `test_geneo_oras_beats_one_level_strong_helmholtz` for numbers).
    ///
    /// Mutually exclusive with [`Self::with_nicolaides_coarse_space`]:
    /// attaching one coarse space replaces the other (last builder call
    /// wins). Works on top of every constructor; defaults are unchanged —
    /// without this call the preconditioner remains one-level.
    fn attach_geneo_coarse_space(
        &mut self,
        matrix: &CsrMatrix<T>,
        config: &GeneoConfig,
    ) -> Result<(), GeneoSetupError> {
        let coarse = build_geneo_coarse_space_checked(
            matrix,
            &self.subdomains,
            &self.weights,
            self.n,
            config,
        )?;
        self.coarse = Some(Arc::new(coarse));
        self.coarse_setup = SchwarzCoarseSetup::Attached;
        Ok(())
    }

    /// Checked GenEO builder that reports setup failures instead of silently
    /// dropping to one-level Schwarz.
    pub fn try_with_geneo_coarse_space(
        mut self,
        matrix: &CsrMatrix<T>,
        config: &GeneoConfig,
    ) -> Result<Self, GeneoSetupError> {
        let result = catch_unwind(AssertUnwindSafe(|| {
            self.attach_geneo_coarse_space(matrix, config)
        }));
        match result {
            Ok(Ok(())) => Ok(self),
            Ok(Err(error)) => Err(error),
            Err(_) => Err(GeneoSetupError::Panicked),
        }
    }

    /// Attach GenEO coarse space, preserving the historical one-level
    /// fallback for callers that do not need typed setup diagnostics.
    pub fn with_geneo_coarse_space(mut self, matrix: &CsrMatrix<T>, config: &GeneoConfig) -> Self {
        match catch_unwind(AssertUnwindSafe(|| {
            self.attach_geneo_coarse_space(matrix, config)
        })) {
            Ok(Ok(())) => {}
            Ok(Err(_)) | Err(_) => {
                self.coarse = None;
                self.coarse_setup = SchwarzCoarseSetup::FellBackToOneLevel;
            }
        }
        self
    }

    /// Whether a coarse space (two-level variant) is attached.
    pub fn has_coarse_space(&self) -> bool {
        self.coarse.is_some()
    }

    /// Report the outcome of the most recent optional coarse-space setup.
    pub fn coarse_setup(&self) -> SchwarzCoarseSetup {
        self.coarse_setup
    }

    /// Dimension of the coarse space (`0` when one-level).
    pub fn coarse_dimension(&self) -> usize {
        self.coarse.as_ref().map_or(0, |coarse| coarse.dim())
    }

    /// Snapshot runtime application and coarse-solve diagnostics.
    pub fn runtime_diagnostics(&self) -> SchwarzRuntimeDiagnostics {
        let load = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        SchwarzRuntimeDiagnostics {
            apply_calls: load(&self.runtime.apply_calls),
            transpose_calls: load(&self.runtime.transpose_calls),
            hermitian_calls: load(&self.runtime.hermitian_calls),
            failed_applications: load(&self.runtime.failed_applications),
            non_finite_inputs: load(&self.runtime.non_finite_inputs),
            non_finite_outputs: load(&self.runtime.non_finite_outputs),
            coarse_correction_calls: load(&self.runtime.coarse_correction_calls),
            coarse_solve_failures: load(&self.runtime.coarse_solve_failures),
            coarse_setup: self.coarse_setup,
            coarse_dimension: self.coarse_dimension(),
        }
    }

    /// Create with automatic subdomain count based on available threads.
    ///
    /// Two caps apply:
    /// * `MIN_DOFS_PER_SUBDOMAIN` — keep each subdomain large enough that
    ///   per-subdomain ILU setup amortises.
    /// * `MAX_USEFUL_SUBDOMAINS` — beyond ~32 partitions, the
    ///   simultaneous-factorisation memory pressure on many-core x86
    ///   plus the H/h convergence degradation (Schwarz with no coarse
    ///   correction) make adding subdomains a net loss. The plateau
    ///   reflects bandwidth saturation, not core count.
    pub fn from_csr_auto(matrix: &CsrMatrix<T>, overlap: usize) -> Self {
        let num_subdomains = auto_subdomain_count(matrix.num_rows);
        Self::from_csr(matrix, num_subdomains, overlap)
    }

    /// Create ORAS with automatic subdomain count based on available threads.
    pub fn from_csr_oras_auto(matrix: &CsrMatrix<T>, overlap: usize, wavenumber: f64) -> Self {
        let num_subdomains = auto_subdomain_count(matrix.num_rows);
        Self::from_csr_oras(matrix, num_subdomains, overlap, wavenumber)
    }

    /// Checked automatic ORAS constructor using the bounded subdomain-count
    /// policy selected by [`Self::from_csr_oras_auto`].
    pub fn try_from_csr_oras_auto(
        matrix: &CsrMatrix<T>,
        overlap: usize,
        wavenumber: f64,
    ) -> Result<Self, SolverError> {
        let num_subdomains = auto_subdomain_count(matrix.num_rows);
        Self::try_from_csr_oras(matrix, num_subdomains, overlap, wavenumber)
    }

    /// Create HABC-style ORAS with automatic subdomain count based on threads.
    pub fn from_csr_habc_auto(matrix: &CsrMatrix<T>, overlap: usize, wavenumber: f64) -> Self {
        let num_subdomains = auto_subdomain_count(matrix.num_rows);
        Self::from_csr_habc(matrix, num_subdomains, overlap, wavenumber)
    }

    /// Checked automatic HABC constructor using the bounded subdomain-count
    /// policy selected by [`Self::from_csr_habc_auto`].
    pub fn try_from_csr_habc_auto(
        matrix: &CsrMatrix<T>,
        overlap: usize,
        wavenumber: f64,
    ) -> Result<Self, SolverError> {
        let num_subdomains = auto_subdomain_count(matrix.num_rows);
        Self::try_from_csr_habc(matrix, num_subdomains, overlap, wavenumber)
    }

    /// Recommended automatic subdomain count used by `from_csr_auto`.
    pub fn recommended_subdomain_count(n: usize) -> usize {
        auto_subdomain_count(n)
    }

    /// Get statistics about the domain decomposition
    pub fn stats(&self) -> (usize, usize, usize, f64) {
        let num_subdomains = self.subdomains.len();
        let min_size = self
            .subdomains
            .iter()
            .map(|s| s.global_indices.len())
            .min()
            .unwrap_or(0);
        let max_size = self
            .subdomains
            .iter()
            .map(|s| s.global_indices.len())
            .max()
            .unwrap_or(0);
        let avg_size = self
            .subdomains
            .iter()
            .map(|s| s.global_indices.len())
            .sum::<usize>() as f64
            / num_subdomains.max(1) as f64;
        (num_subdomains, min_size, max_size, avg_size)
    }

    /// Return aggregate diagnostics from all local ILU(0) factorizations.
    ///
    /// Checked constructors reject skipped pivots and non-finite local
    /// entries before returning. The compatibility constructors retain the
    /// factors and expose this aggregate so callers can identify a
    /// regularized local solve.
    pub fn diagnostics(&self) -> IluDiagnostics<T::Real> {
        let mut diagnostics: IluDiagnostics<T::Real> = IluDiagnostics {
            skipped_pivots: 0,
            non_finite_entries: 0,
            min_abs_pivot: None,
        };
        for subdomain in &self.subdomains {
            let local = subdomain.diagnostics;
            diagnostics.skipped_pivots += local.skipped_pivots;
            diagnostics.non_finite_entries += local.non_finite_entries;
            diagnostics.min_abs_pivot = match (diagnostics.min_abs_pivot, local.min_abs_pivot) {
                (Some(current), Some(candidate)) => Some(current.min(candidate)),
                (None, candidate) => candidate,
                (current, None) => current,
            };
        }
        diagnostics
    }

    /// Serialize a one-level Schwarz preconditioner into a portable checkpoint.
    ///
    /// The local ILU factors are intentionally rebuilt on restore from the
    /// serialized local CSR matrices.  This keeps the format independent of
    /// the machine's index width and avoids persisting implementation details
    /// of the ILU factor storage.  Version 2 also includes a fully attached
    /// coarse basis and its normal/transpose/Hermitian sparse-LU factors.
    pub fn to_checkpoint_bytes(&self) -> Result<Vec<u8>, SolverError> {
        let coarse_marker = match (self.coarse_setup, self.coarse.as_ref()) {
            (SchwarzCoarseSetup::NotRequested, None) => 0,
            (SchwarzCoarseSetup::FellBackToOneLevel, None) => 1,
            (SchwarzCoarseSetup::Attached, Some(_)) => 2,
            (SchwarzCoarseSetup::Attached, None) => {
                return Err(schwarz_checkpoint_error(
                    "coarse setup is attached but coarse state is missing",
                ));
            }
            (_, Some(_)) => {
                return Err(schwarz_checkpoint_error(
                    "coarse state is present with an incompatible setup marker",
                ));
            }
        };
        if self.n == 0 {
            return Err(schwarz_checkpoint_error(
                "checkpoint cannot contain a zero-dimensional preconditioner",
            ));
        }
        if self.weights.len() != self.n {
            return Err(schwarz_checkpoint_error(format!(
                "weight length {} differs from matrix dimension {}",
                self.weights.len(),
                self.n
            )));
        }

        let mut core_coverage = vec![false; self.n];
        for core in &self.core_partitions {
            if core.is_empty() {
                return Err(schwarz_checkpoint_error(
                    "checkpoint cannot contain an empty core partition",
                ));
            }
            let mut previous = None;
            for &global_index in core {
                if global_index >= self.n {
                    return Err(schwarz_checkpoint_error(
                        "core partition index outside matrix dimension",
                    ));
                }
                if previous.is_some_and(|last| global_index <= last) {
                    return Err(schwarz_checkpoint_error(
                        "core partitions must be strictly increasing",
                    ));
                }
                previous = Some(global_index);
                if core_coverage[global_index] {
                    return Err(schwarz_checkpoint_error(
                        "core partitions contain a duplicate global index",
                    ));
                }
                core_coverage[global_index] = true;
            }
        }
        if core_coverage.iter().any(|covered| !covered) {
            return Err(schwarz_checkpoint_error(
                "core partitions do not cover all matrix DOFs",
            ));
        }

        let mut subdomain_coverage = vec![false; self.n];
        for subdomain in &self.subdomains {
            let local_n = subdomain.global_indices.len();
            if local_n == 0 {
                return Err(schwarz_checkpoint_error(
                    "checkpoint cannot contain an empty subdomain",
                ));
            }
            let expected_row_ptr_count = local_n
                .checked_add(1)
                .ok_or_else(|| schwarz_checkpoint_error("subdomain size overflows row count"))?;
            if subdomain.local_row_ptrs.len() != expected_row_ptr_count
                || subdomain.local_row_ptrs.first().copied() != Some(0)
                || subdomain.local_row_ptrs.last().copied() != Some(subdomain.local_values.len())
                || subdomain.local_col_indices.len() != subdomain.local_values.len()
            {
                return Err(schwarz_checkpoint_error(
                    "subdomain CSR storage has inconsistent lengths",
                ));
            }
            let local_matrix = CsrMatrix::try_from_raw_parts(
                local_n,
                local_n,
                subdomain.local_row_ptrs.clone(),
                subdomain.local_col_indices.clone(),
                subdomain.local_values.clone(),
            )?;
            local_matrix.validate_canonical_structure()?;
            if subdomain.diagnostics.skipped_pivots != 0
                || subdomain.diagnostics.non_finite_entries != 0
            {
                return Err(schwarz_checkpoint_error(
                    "cannot checkpoint a subdomain with non-clean ILU diagnostics",
                ));
            }
            if subdomain
                .u_diag_inv
                .iter()
                .any(|value| !value.re().is_finite() || !value.im().is_finite())
            {
                return Err(schwarz_checkpoint_error(
                    "subdomain ILU factors contain a non-finite diagonal",
                ));
            }
            let mut previous = None;
            for &global_index in &subdomain.global_indices {
                if global_index >= self.n {
                    return Err(schwarz_checkpoint_error(
                        "subdomain index outside matrix dimension",
                    ));
                }
                if previous.is_some_and(|last| global_index <= last) {
                    return Err(schwarz_checkpoint_error(
                        "subdomain indices must be strictly increasing",
                    ));
                }
                previous = Some(global_index);
                subdomain_coverage[global_index] = true;
            }
        }
        if subdomain_coverage.iter().any(|covered| !covered) {
            return Err(schwarz_checkpoint_error(
                "subdomains do not cover all matrix DOFs",
            ));
        }

        let mut output = Vec::new();
        output.extend_from_slice(SCHWARZ_CHECKPOINT_MAGIC);
        output.extend_from_slice(&SCHWARZ_CHECKPOINT_FORMAT_VERSION.to_le_bytes());
        output.push(std::mem::size_of::<T>() as u8);
        output.push(std::mem::size_of::<T::Real>() as u8);
        output.push(u8::from(
            std::mem::size_of::<T>() == 2 * std::mem::size_of::<T::Real>(),
        ));
        output.push(coarse_marker);
        schwarz_checkpoint_push_len(&mut output, self.n)?;
        schwarz_checkpoint_push_len(&mut output, self.weights.len())?;
        for &weight in &self.weights {
            schwarz_checkpoint_push_scalar(&mut output, weight)?;
        }
        schwarz_checkpoint_push_len(&mut output, self.core_partitions.len())?;
        for core in &self.core_partitions {
            schwarz_checkpoint_push_len(&mut output, core.len())?;
            for &global_index in core {
                schwarz_checkpoint_push_len(&mut output, global_index)?;
            }
        }
        schwarz_checkpoint_push_len(&mut output, self.subdomains.len())?;
        for subdomain in &self.subdomains {
            schwarz_checkpoint_push_len(&mut output, subdomain.global_indices.len())?;
            for &global_index in &subdomain.global_indices {
                schwarz_checkpoint_push_len(&mut output, global_index)?;
            }
            schwarz_checkpoint_push_len(&mut output, subdomain.local_row_ptrs.len())?;
            for &row_ptr in &subdomain.local_row_ptrs {
                schwarz_checkpoint_push_len(&mut output, row_ptr)?;
            }
            schwarz_checkpoint_push_len(&mut output, subdomain.local_col_indices.len())?;
            for &column in &subdomain.local_col_indices {
                schwarz_checkpoint_push_len(&mut output, column)?;
            }
            schwarz_checkpoint_push_len(&mut output, subdomain.local_values.len())?;
            for &value in &subdomain.local_values {
                schwarz_checkpoint_push_scalar(&mut output, value)?;
            }
        }
        if let Some(coarse) = self.coarse.as_deref() {
            schwarz_checkpoint_push_coarse(&mut output, coarse)?;
        }
        Ok(output)
    }

    /// Restore a one-level Schwarz preconditioner from a portable checkpoint.
    pub fn from_checkpoint_bytes(bytes: &[u8]) -> Result<Self, SolverError> {
        let mut reader = SchwarzCheckpointReader::new(bytes);
        if reader.take(SCHWARZ_CHECKPOINT_MAGIC.len())? != SCHWARZ_CHECKPOINT_MAGIC {
            return Err(schwarz_checkpoint_error("invalid Schwarz checkpoint magic"));
        }
        let version = reader.read_u16()?;
        if version != SCHWARZ_CHECKPOINT_LEGACY_FORMAT_VERSION
            && version != SCHWARZ_CHECKPOINT_FORMAT_VERSION
        {
            return Err(schwarz_checkpoint_error(
                "unsupported Schwarz checkpoint version",
            ));
        }
        if reader.read_u8()? as usize != std::mem::size_of::<T>()
            || reader.read_u8()? as usize != std::mem::size_of::<T::Real>()
        {
            return Err(schwarz_checkpoint_error(
                "checkpoint scalar metadata does not match requested type",
            ));
        }
        let expected_complex_marker =
            u8::from(std::mem::size_of::<T>() == 2 * std::mem::size_of::<T::Real>());
        if reader.read_u8()? != expected_complex_marker {
            return Err(schwarz_checkpoint_error(
                "checkpoint complex-scalar marker does not match requested type",
            ));
        }
        let coarse_setup = match reader.read_u8()? {
            0 => SchwarzCoarseSetup::NotRequested,
            1 => SchwarzCoarseSetup::FellBackToOneLevel,
            2 if version == SCHWARZ_CHECKPOINT_FORMAT_VERSION => SchwarzCoarseSetup::Attached,
            2 => {
                return Err(schwarz_checkpoint_error(
                    "coarse-space snapshots require checkpoint format v2",
                ));
            }
            _ => {
                return Err(schwarz_checkpoint_error(
                    "invalid coarse-space setup marker",
                ));
            }
        };
        let n = usize::try_from(reader.read_u64()?)
            .map_err(|_| schwarz_checkpoint_error("matrix dimension does not fit usize"))?;
        if n == 0 {
            return Err(schwarz_checkpoint_error(
                "checkpoint cannot contain a zero-dimensional preconditioner",
            ));
        }

        let weight_count = reader.read_len("weight count", 16)?;
        if weight_count != n {
            return Err(schwarz_checkpoint_error(format!(
                "checkpoint contains {weight_count} weights for dimension {n}"
            )));
        }
        let mut weights = Vec::with_capacity(weight_count);
        for _ in 0..weight_count {
            let weight = reader.read_scalar::<T>()?;
            if weight.norm() == T::Real::zero() {
                return Err(schwarz_checkpoint_error(
                    "checkpoint contains a zero partition weight",
                ));
            }
            weights.push(weight);
        }

        let core_count = reader.read_len("core partition count", 8)?;
        if core_count == 0 || core_count > n {
            return Err(schwarz_checkpoint_error(
                "checkpoint core partition count is out of range",
            ));
        }
        let mut core_partitions = Vec::with_capacity(core_count);
        let mut core_coverage = vec![false; n];
        for _ in 0..core_count {
            let core_len = reader.read_len("core partition length", 8)?;
            if core_len == 0 || core_len > n {
                return Err(schwarz_checkpoint_error(
                    "checkpoint contains an invalid core partition length",
                ));
            }
            let mut core = Vec::with_capacity(core_len);
            let mut previous = None;
            for _ in 0..core_len {
                let global_index = usize::try_from(reader.read_u64()?)
                    .map_err(|_| schwarz_checkpoint_error("core index does not fit usize"))?;
                if global_index >= n || previous.is_some_and(|last| global_index <= last) {
                    return Err(schwarz_checkpoint_error(
                        "checkpoint core indices are out of range or unsorted",
                    ));
                }
                previous = Some(global_index);
                if core_coverage[global_index] {
                    return Err(schwarz_checkpoint_error(
                        "checkpoint core partitions contain a duplicate index",
                    ));
                }
                core_coverage[global_index] = true;
                core.push(global_index);
            }
            core_partitions.push(core);
        }
        if core_coverage.iter().any(|covered| !covered) {
            return Err(schwarz_checkpoint_error(
                "checkpoint core partitions do not cover all matrix DOFs",
            ));
        }

        let subdomain_count = reader.read_len("subdomain count", 8)?;
        if subdomain_count == 0 || subdomain_count > n {
            return Err(schwarz_checkpoint_error(
                "checkpoint subdomain count is out of range",
            ));
        }
        let mut subdomains = Vec::with_capacity(subdomain_count);
        let mut subdomain_coverage = vec![false; n];
        for _ in 0..subdomain_count {
            let local_n = reader.read_len("subdomain index count", 8)?;
            if local_n == 0 || local_n > n {
                return Err(schwarz_checkpoint_error(
                    "checkpoint contains an invalid subdomain size",
                ));
            }
            let mut global_indices = Vec::with_capacity(local_n);
            let mut previous = None;
            for _ in 0..local_n {
                let global_index = usize::try_from(reader.read_u64()?)
                    .map_err(|_| schwarz_checkpoint_error("subdomain index does not fit usize"))?;
                if global_index >= n || previous.is_some_and(|last| global_index <= last) {
                    return Err(schwarz_checkpoint_error(
                        "checkpoint subdomain indices are out of range or unsorted",
                    ));
                }
                previous = Some(global_index);
                subdomain_coverage[global_index] = true;
                global_indices.push(global_index);
            }

            let row_ptr_count = reader.read_len("local row-pointer count", 8)?;
            let expected_row_ptr_count = local_n
                .checked_add(1)
                .ok_or_else(|| schwarz_checkpoint_error("subdomain size overflows row count"))?;
            if row_ptr_count != expected_row_ptr_count {
                return Err(schwarz_checkpoint_error(
                    "local row-pointer count does not match subdomain size",
                ));
            }
            let mut local_row_ptrs = Vec::with_capacity(row_ptr_count);
            for _ in 0..row_ptr_count {
                local_row_ptrs.push(usize::try_from(reader.read_u64()?).map_err(|_| {
                    schwarz_checkpoint_error("local row pointer does not fit usize")
                })?);
            }
            if local_row_ptrs.first().copied() != Some(0)
                || local_row_ptrs
                    .windows(2)
                    .any(|window| window[0] > window[1])
            {
                return Err(schwarz_checkpoint_error(
                    "local row pointers are not monotone",
                ));
            }

            let column_count = reader.read_len("local column count", 8)?;
            let mut local_col_indices = Vec::with_capacity(column_count);
            for _ in 0..column_count {
                let column = usize::try_from(reader.read_u64()?)
                    .map_err(|_| schwarz_checkpoint_error("local column does not fit usize"))?;
                if column >= local_n {
                    return Err(schwarz_checkpoint_error(
                        "local column outside subdomain dimension",
                    ));
                }
                local_col_indices.push(column);
            }
            let value_count = reader.read_len("local value count", 16)?;
            if column_count != value_count || local_row_ptrs.last().copied() != Some(value_count) {
                return Err(schwarz_checkpoint_error(
                    "local CSR column/value counts do not match row pointers",
                ));
            }
            let mut local_values = Vec::with_capacity(value_count);
            for _ in 0..value_count {
                local_values.push(reader.read_scalar::<T>()?);
            }
            let local_matrix = CsrMatrix::try_from_raw_parts(
                local_n,
                local_n,
                local_row_ptrs.clone(),
                local_col_indices.clone(),
                local_values.clone(),
            )?;
            local_matrix.validate_canonical_structure()?;
            let factors = try_ilu0_factorize(&local_matrix, "Schwarz checkpoint ILU restore")?;
            let diagnostics = factors.diagnostics;
            subdomains.push(Subdomain {
                global_indices,
                local_values,
                local_col_indices,
                local_row_ptrs,
                l_values: factors.l_values,
                l_col_indices: factors
                    .l_col_indices
                    .iter()
                    .map(|&column| column as usize)
                    .collect(),
                l_row_ptrs: factors.l_row_ptrs,
                u_values: factors.u_values,
                u_col_indices: factors
                    .u_col_indices
                    .iter()
                    .map(|&column| column as usize)
                    .collect(),
                u_row_ptrs: factors.u_row_ptrs,
                u_diag_inv: factors.u_diag_inv,
                diagnostics,
            });
        }
        if subdomain_coverage.iter().any(|covered| !covered) {
            return Err(schwarz_checkpoint_error(
                "checkpoint subdomains do not cover all matrix DOFs",
            ));
        }
        let coarse = if coarse_setup == SchwarzCoarseSetup::Attached {
            Some(Arc::new(schwarz_checkpoint_read_coarse(&mut reader, n)?))
        } else {
            None
        };
        if !reader.is_at_end() {
            return Err(schwarz_checkpoint_error("trailing bytes"));
        }
        Ok(Self {
            subdomains,
            weights,
            n,
            core_partitions,
            coarse,
            coarse_setup,
            runtime: Arc::new(SchwarzRuntimeState::default()),
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct TransmissionCondition {
    wavenumber: f64,
    higher_order: bool,
}

fn auto_subdomain_count(n: usize) -> usize {
    auto_subdomain_count_for_topology(n, rayon::current_num_threads(), &MachineTopology::detect())
}

fn auto_subdomain_count_for_topology(
    n: usize,
    num_threads: usize,
    topology: &MachineTopology,
) -> usize {
    const MIN_DOFS_PER_SUBDOMAIN: usize = 50_000;
    const MAX_USEFUL_SUBDOMAINS: usize = 32;

    let num_threads = num_threads.max(1);
    let by_size = n.div_ceil(MIN_DOFS_PER_SUBDOMAIN).max(1);
    let thread_cap = num_threads.min(MAX_USEFUL_SUBDOMAINS);
    let mut count = thread_cap.min(by_size).max(1);

    let topology_domains = topology
        .numa_nodes
        .len()
        .max(topology.sockets.len())
        .max(1)
        .min(thread_cap)
        .min(by_size);
    if topology_domains > 1 {
        count = count.max(topology_domains);
        let rounded_up = count.div_ceil(topology_domains) * topology_domains;
        if rounded_up <= thread_cap && rounded_up <= by_size {
            count = rounded_up;
        } else {
            let rounded_down = (count / topology_domains) * topology_domains;
            if rounded_down >= topology_domains {
                count = rounded_down;
            }
        }
    }

    count
}

/// Build adjacency list from sparse matrix
fn build_adjacency<T: ComplexField>(matrix: &CsrMatrix<T>) -> Vec<Vec<usize>> {
    let n = matrix.num_rows;
    let mut adjacency = vec![Vec::new(); n];

    for (i, row_adj) in adjacency.iter_mut().enumerate().take(n) {
        for idx in matrix.row_ptrs[i]..matrix.row_ptrs[i + 1] {
            let j = matrix.col_indices[idx];
            if i != j {
                row_adj.push(j);
            }
        }
    }

    adjacency
}

/// Extend a partition by `overlap` layers using adjacency
fn extend_partition(
    partition: &[usize],
    adjacency: &[Vec<usize>],
    overlap: usize,
    n: usize,
) -> Vec<usize> {
    let mut in_partition = vec![false; n];
    for &idx in partition {
        in_partition[idx] = true;
    }

    let mut frontier: Vec<usize> = partition.to_vec();

    for _ in 0..overlap {
        let mut new_frontier = Vec::new();
        for &idx in &frontier {
            for &neighbor in &adjacency[idx] {
                if !in_partition[neighbor] {
                    in_partition[neighbor] = true;
                    new_frontier.push(neighbor);
                }
            }
        }
        frontier = new_frontier;
    }

    // Collect all indices and sort for cache-friendly access
    let mut result: Vec<usize> = (0..n).filter(|&i| in_partition[i]).collect();
    result.sort_unstable();
    result
}

/// Build a subdomain with extracted local matrix and ILU factorization
fn build_subdomain<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    global_indices: Vec<usize>,
    transmission: Option<TransmissionCondition>,
    checked_local_setup: bool,
) -> Result<Subdomain<T>, SolverError> {
    let local_n = global_indices.len();

    // Build global-to-local index mapping
    let mut global_to_local = vec![usize::MAX; matrix.num_rows];
    for (local_idx, &global_idx) in global_indices.iter().enumerate() {
        global_to_local[global_idx] = local_idx;
    }

    // Extract local matrix in CSR format
    let mut local_values = Vec::new();
    let mut local_col_indices = Vec::new();
    let mut local_row_ptrs = vec![0];

    for (local_row, &global_row) in global_indices.iter().enumerate() {
        let row_start = local_values.len();
        let mut diagonal_entry = None;
        let mut dropped_coupling = T::Real::zero();

        for idx in matrix.row_ptrs[global_row]..matrix.row_ptrs[global_row + 1] {
            let global_col = matrix.col_indices[idx];
            if global_col < matrix.num_rows {
                let local_col = global_to_local[global_col];
                if local_col != usize::MAX {
                    if local_col == local_row {
                        diagonal_entry = Some(local_values.len());
                    }
                    local_values.push(matrix.values[idx]);
                    local_col_indices.push(local_col);
                } else {
                    dropped_coupling += matrix.values[idx].norm();
                }
            }
        }

        if let Some(condition) = transmission
            && dropped_coupling > T::Real::zero()
            && condition.wavenumber > 0.0
        {
            let impedance = interface_impedance::<T>(
                dropped_coupling,
                condition.wavenumber,
                condition.higher_order,
            );
            if let Some(diag_idx) = diagonal_entry {
                local_values[diag_idx] += impedance;
            } else {
                let row_end = local_values.len();
                let insert_at = (row_start..row_end)
                    .find(|&idx| local_col_indices[idx] > local_row)
                    .unwrap_or(row_end);
                local_values.insert(insert_at, impedance);
                local_col_indices.insert(insert_at, local_row);
            }
        }

        local_row_ptrs.push(local_values.len());
    }

    // Factor the local matrix with the canonical ILU(0) kernel shared with
    // `IluPreconditioner` and the parallel/hierarchical ILU variants.
    //
    // NOTE: this file historically carried its own ILU(0) copy that ZEROED
    // the multiplier slot on a skipped pivot while the other three copies
    // left the stale matrix entry in the pattern position. That divergence
    // is now unified: the canonical kernel zeroes the slot (the semantic
    // regression-tested by `test_schwarz_small_pivot_skip_zeros_l`, issue
    // #13), so no Schwarz test needed updating for the consolidation.
    let local_matrix = if checked_local_setup {
        CsrMatrix::try_from_raw_parts(
            local_n,
            local_n,
            local_row_ptrs.clone(),
            local_col_indices.clone(),
            local_values.clone(),
        )?
    } else {
        CsrMatrix::from_raw_parts(
            local_n,
            local_n,
            local_row_ptrs.clone(),
            local_col_indices.clone(),
            local_values.clone(),
        )
    };
    let factors = if checked_local_setup {
        try_ilu0_factorize(&local_matrix, "Schwarz local ILU setup")?
    } else {
        ilu0_factorize(&local_matrix)
    };
    let diagnostics = factors.diagnostics;

    Ok(Subdomain {
        global_indices,
        local_values,
        local_col_indices,
        local_row_ptrs,
        l_values: factors.l_values,
        l_col_indices: factors.l_col_indices.iter().map(|&j| j as usize).collect(),
        l_row_ptrs: factors.l_row_ptrs,
        u_values: factors.u_values,
        u_col_indices: factors.u_col_indices.iter().map(|&j| j as usize).collect(),
        u_row_ptrs: factors.u_row_ptrs,
        u_diag_inv: factors.u_diag_inv,
        diagnostics,
    })
}

fn interface_impedance<T: ComplexField>(
    dropped_coupling: T::Real,
    wavenumber: f64,
    higher_order: bool,
) -> T {
    let k = T::Real::from_f64(wavenumber.abs().max(1e-12)).unwrap();
    let strength = dropped_coupling.sqrt();
    let robin = T::from_re_im(T::Real::zero(), k * strength);

    if higher_order {
        let two = T::Real::from_f64(2.0).unwrap();
        robin + T::from_real(dropped_coupling / (two * k))
    } else {
        robin
    }
}

impl<T: ComplexField> Subdomain<T> {
    /// Solve local system in place.
    fn solve_in_place(&self, y: &mut [T]) {
        let n = self.global_indices.len();
        assert_eq!(y.len(), n, "Schwarz local solve buffer length mismatch");

        // Forward substitution: Ly = r
        for i in 0..n {
            for idx in self.l_row_ptrs[i]..self.l_row_ptrs[i + 1] {
                let j = self.l_col_indices[idx];
                let l_ij = self.l_values[idx];
                y[i] -= l_ij * y[j];
            }
        }

        // Backward substitution: Ux = y
        for i in (0..n).rev() {
            for idx in self.u_row_ptrs[i]..self.u_row_ptrs[i + 1] {
                let j = self.u_col_indices[idx];
                let u_ij = self.u_values[idx];
                y[i] -= u_ij * y[j];
            }

            y[i] *= self.u_diag_inv[i];
        }
    }

    /// Solve the transpose or Hermitian local triangular factors in place.
    /// The local factor is `L U`, so the adjoint solve is `UᵀLᵀ` or
    /// `UᴴLᴴ`. Row-wise storage permits the same O(nnz) scatter updates as
    /// the forward solve.
    fn solve_adjoint_in_place(&self, y: &mut [T], hermitian: bool) {
        let n = self.global_indices.len();
        assert_eq!(
            y.len(),
            n,
            "Schwarz local adjoint solve buffer length mismatch"
        );

        // Uᵀ/Uᴴ is lower triangular. Solve in ascending order and scatter
        // each known entry into the later right-hand-side entries.
        for i in 0..n {
            y[i] = if hermitian {
                y[i] * self.u_diag_inv[i].conj()
            } else {
                y[i] * self.u_diag_inv[i]
            };
            for index in self.u_row_ptrs[i]..self.u_row_ptrs[i + 1] {
                let j = self.u_col_indices[index];
                let value = if hermitian {
                    self.u_values[index].conj()
                } else {
                    self.u_values[index]
                };
                y[j] -= value * y[i];
            }
        }

        // Lᵀ/Lᴴ is upper triangular with unit diagonal. Descending row
        // scatter updates complete the solve without constructing CSC data.
        for i in (0..n).rev() {
            for index in self.l_row_ptrs[i]..self.l_row_ptrs[i + 1] {
                let j = self.l_col_indices[index];
                let value = if hermitian {
                    self.l_values[index].conj()
                } else {
                    self.l_values[index]
                };
                y[j] -= value * y[i];
            }
        }
    }
}

impl<T: ComplexField + Send + Sync> Preconditioner<T> for AdditiveSchwarzPreconditioner<T> {
    fn try_apply(&self, r: &[T]) -> Result<Vec<T>, SolverError> {
        if r.len() != self.n {
            self.runtime
                .failed_applications
                .fetch_add(1, Ordering::Relaxed);
            return Err(SolverError::DimensionMismatch {
                expected: self.n,
                got: r.len(),
            });
        }
        if r.iter().any(|value| !value.norm().is_finite()) {
            self.runtime
                .non_finite_inputs
                .fetch_add(1, Ordering::Relaxed);
            self.runtime
                .failed_applications
                .fetch_add(1, Ordering::Relaxed);
            return Err(SolverError::OperationFailed {
                operation: "Schwarz apply",
                reason: "input contains NaN or infinity".into(),
            });
        }
        let applied = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.apply(r)))
        {
            Ok(applied) => applied,
            Err(_) => {
                self.runtime
                    .failed_applications
                    .fetch_add(1, Ordering::Relaxed);
                return Err(SolverError::OperationFailed {
                    operation: "Schwarz apply",
                    reason: "preconditioner application panicked".into(),
                });
            }
        };
        if applied.len() != self.n {
            self.runtime
                .failed_applications
                .fetch_add(1, Ordering::Relaxed);
            return Err(SolverError::DimensionMismatch {
                expected: self.n,
                got: applied.len(),
            });
        }
        if applied.iter().any(|value| !value.norm().is_finite()) {
            self.runtime
                .non_finite_outputs
                .fetch_add(1, Ordering::Relaxed);
            self.runtime
                .failed_applications
                .fetch_add(1, Ordering::Relaxed);
            return Err(SolverError::OperationFailed {
                operation: "Schwarz apply",
                reason: "output contains NaN or infinity".into(),
            });
        }
        Ok(applied)
    }

    fn apply(&self, r: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); self.n];
        self.apply_into(r, &mut y);
        y
    }

    fn apply_into(&self, r: &[T], y: &mut [T]) {
        self.runtime.apply_calls.fetch_add(1, Ordering::Relaxed);
        assert_eq!(r.len(), self.n, "Schwarz RHS length mismatch");
        assert_eq!(y.len(), self.n, "Schwarz output length mismatch");

        if self.n >= 1000 && self.subdomains.len() > 1 {
            self.apply_parallel_into(r, y);
        } else {
            self.apply_sequential_into(r, y);
        }
    }

    fn apply_transpose(&self, r: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); self.n];
        self.apply_transpose_into(r, &mut y);
        y
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn apply_transpose_into(&self, r: &[T], y: &mut [T]) {
        self.runtime.transpose_calls.fetch_add(1, Ordering::Relaxed);
        self.apply_adjoint_into(r, y, false);
    }

    fn apply_hermitian(&self, r: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); self.n];
        self.apply_hermitian_into(r, &mut y);
        y
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn apply_hermitian_into(&self, r: &[T], y: &mut [T]) {
        self.runtime.hermitian_calls.fetch_add(1, Ordering::Relaxed);
        self.apply_adjoint_into(r, y, true);
    }
}

impl<T: ComplexField + Send + Sync> AdditiveSchwarzPreconditioner<T> {
    #[cfg(test)]
    fn apply_sequential(&self, r: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); self.n];
        self.apply_sequential_into(r, &mut y);
        y
    }

    #[cfg(test)]
    fn apply_parallel(&self, r: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); self.n];
        self.apply_parallel_into(r, &mut y);
        y
    }

    fn apply_sequential_into(&self, r: &[T], y: &mut [T]) {
        y.fill(T::zero());

        // Pre-allocate scratch buffer sized to the largest subdomain
        let max_local_n = self
            .subdomains
            .iter()
            .map(|s| s.global_indices.len())
            .max()
            .unwrap_or(0);
        let mut local_solution = vec![T::zero(); max_local_n];

        for subdomain in &self.subdomains {
            let local_n = subdomain.global_indices.len();

            // Gather local RHS into pre-allocated buffer
            for (local_idx, &global_idx) in subdomain.global_indices.iter().enumerate() {
                local_solution[local_idx] = r[global_idx];
            }

            // Solve local system in reusable scratch
            subdomain.solve_in_place(&mut local_solution[..local_n]);

            // Scatter back to global with weights
            for (local_idx, &global_idx) in subdomain.global_indices.iter().enumerate() {
                y[global_idx] += local_solution[local_idx] * self.weights[global_idx];
            }
        }

        // Two-level: add the coarse correction Z·E⁻¹·Zᵀ·r (same residual as
        // the local solves — additive combination, no deflation).
        if let Some(coarse) = &self.coarse {
            self.runtime
                .coarse_correction_calls
                .fetch_add(1, Ordering::Relaxed);
            if !coarse.apply_add(r, y) {
                self.runtime
                    .coarse_solve_failures
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn apply_parallel_into(&self, r: &[T], y: &mut [T]) {
        let r_slice = r;

        let local_solutions: Vec<Vec<T>> = self
            .subdomains
            .par_iter()
            .map(|subdomain| {
                let mut local_solution = subdomain
                    .global_indices
                    .iter()
                    .map(|&i| r_slice[i])
                    .collect::<Vec<_>>();
                subdomain.solve_in_place(&mut local_solution);
                local_solution
            })
            .collect();

        y.fill(T::zero());
        for (subdomain, local_solution) in self.subdomains.iter().zip(&local_solutions) {
            for (local_idx, &global_idx) in subdomain.global_indices.iter().enumerate() {
                y[global_idx] += local_solution[local_idx] * self.weights[global_idx];
            }
        }

        // Two-level: add the coarse correction Z·E⁻¹·Zᵀ·r (same residual as
        // the local solves — additive combination, no deflation).
        if let Some(coarse) = &self.coarse {
            self.runtime
                .coarse_correction_calls
                .fetch_add(1, Ordering::Relaxed);
            if !coarse.apply_add(r, y) {
                self.runtime
                    .coarse_solve_failures
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn apply_adjoint_into(&self, r: &[T], y: &mut [T], hermitian: bool) {
        assert_eq!(r.len(), self.n, "Schwarz adjoint RHS length mismatch");
        assert_eq!(y.len(), self.n, "Schwarz adjoint output length mismatch");
        if self.n >= 1000 && self.subdomains.len() > 1 {
            let local_solutions: Vec<Vec<T>> = self
                .subdomains
                .par_iter()
                .map(|subdomain| {
                    let mut local_solution = subdomain
                        .global_indices
                        .iter()
                        .map(|&index| r[index] * self.weights[index])
                        .collect::<Vec<_>>();
                    subdomain.solve_adjoint_in_place(&mut local_solution, hermitian);
                    local_solution
                })
                .collect();

            y.fill(T::zero());
            for (subdomain, local_solution) in self.subdomains.iter().zip(&local_solutions) {
                for (local_idx, &global_idx) in subdomain.global_indices.iter().enumerate() {
                    y[global_idx] += local_solution[local_idx];
                }
            }
        } else {
            y.fill(T::zero());
            let max_local_n = self
                .subdomains
                .iter()
                .map(|subdomain| subdomain.global_indices.len())
                .max()
                .unwrap_or(0);
            let mut local_solution = vec![T::zero(); max_local_n];

            for subdomain in &self.subdomains {
                let local_n = subdomain.global_indices.len();
                for (local_idx, &global_idx) in subdomain.global_indices.iter().enumerate() {
                    local_solution[local_idx] = r[global_idx] * self.weights[global_idx];
                }
                subdomain.solve_adjoint_in_place(&mut local_solution[..local_n], hermitian);
                for (local_idx, &global_idx) in subdomain.global_indices.iter().enumerate() {
                    y[global_idx] += local_solution[local_idx];
                }
            }
        }
        if let Some(coarse) = &self.coarse {
            self.runtime
                .coarse_correction_calls
                .fetch_add(1, Ordering::Relaxed);
            let applied = if hermitian {
                coarse.apply_hermitian_add(r, y)
            } else {
                coarse.apply_transpose_add(r, y)
            };
            if !applied {
                self.runtime
                    .coarse_solve_failures
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

// =============================================================================
// Double-sweep multiplicative Schwarz (sweeping-style) preconditioner.
// SIM-PERF-20261013.md section 2, gap 2: the missing sweeping /
// double-sweep / source-transfer preconditioner for high-k Helmholtz.
//
// A `SweepingSchwarzPreconditioner` holds the same ORAS subdomain factors
// as [`AdditiveSchwarzPreconditioner`] (first-order absorbing Robin
// transmission on artificial interfaces, ILU(0) local solves) but applies
// them as ordered forward/backward sweeps over coordinate layers instead
// of an additive combination. Each layer corrects the current residual
// with a restricted multiplicative update, so information propagates
// across the whole domain in one double sweep; the additive variant
// needs one iteration per subdomain diameter for the same transport.
// Cost per apply is two subdomain-solve passes plus roughly one global
// SpMV of residual updates.
//
// Scope notes, read before extending:
// - Transmission is first-order ABC (the existing ORAS machinery).
//   PML transmission, which defines the Engquist-Ying sweeping
//   preconditioner proper, is not implemented: there is no PML
//   infrastructure to build on yet.
// - Adjoint actions report `Unsupported` (the trait default); exact
//   transpose sweeps would run the layer order backwards through
//   `Subdomain::solve_adjoint_in_place`, which exists but is not
//   wired here yet.
// =============================================================================

/// Ordered layer of a [`SweepingSchwarzPreconditioner`]: an extended
/// (overlapped) subdomain with its local factors plus the owned-core
/// DOFs this layer is responsible for correcting.
#[derive(Debug, Clone)]
struct SweepLayer<T: ComplexField> {
    subdomain: Subdomain<T>,
    /// Owned (core) DOFs in the caller-provided layer order.
    owned_global: Vec<usize>,
    /// Local row of each owned DOF inside the extended subdomain.
    owned_local: Vec<usize>,
}

/// Double-sweep multiplicative Schwarz preconditioner over ordered
/// coordinate layers with ORAS transmission.
///
/// The operator is a fixed linear map (fixed layer order, fixed local
/// factors), so it is safe for standard (non-flexible) GMRES. It is
/// not symmetric in general; use the Hermitian/transpose actions only
/// once they are implemented (currently `Unsupported`).
#[derive(Debug, Clone)]
pub struct SweepingSchwarzPreconditioner<T: ComplexField> {
    layers: Vec<SweepLayer<T>>,
    matrix: CsrMatrix<T>,
    n: usize,
    wavenumber: f64,
}

impl<T: ComplexField> SweepingSchwarzPreconditioner<T> {
    /// Build sweeping layers from explicit ordered core-DOF sets.
    ///
    /// Every DOF must be owned by exactly one layer (checked, same
    /// rule as the additive partitions); each layer is extended by
    /// `overlap` adjacency rings and factored with first-order
    /// absorbing transmission at `wavenumber`.
    pub fn try_from_csr_layers(
        matrix: &CsrMatrix<T>,
        layers: &[Vec<usize>],
        overlap: usize,
        wavenumber: f64,
    ) -> Result<Self, SolverError> {
        if !wavenumber.is_finite() || wavenumber <= 0.0 {
            return Err(SolverError::OperationFailed {
                operation: "Sweeping Schwarz setup",
                reason: "wavenumber must be finite and positive".into(),
            });
        }
        AdditiveSchwarzPreconditioner::<T>::validate_core_partitions(matrix, layers)?;
        let n = matrix.num_rows;
        let adjacency = build_adjacency(matrix);
        let transmission = Some(TransmissionCondition {
            wavenumber,
            higher_order: false,
        });
        let mut built = Vec::with_capacity(layers.len());
        for layer in layers {
            let mut core: Vec<usize> = layer.iter().copied().filter(|&idx| idx < n).collect();
            core.sort_unstable();
            core.dedup();
            let extended = extend_partition(&core, &adjacency, overlap, n);
            let subdomain = build_subdomain(matrix, extended, transmission, true)?;
            let mut owned_global = Vec::with_capacity(core.len());
            let mut owned_local = Vec::with_capacity(core.len());
            for &global in &core {
                let local = subdomain
                    .global_indices
                    .binary_search(&global)
                    .map_err(|_| SolverError::OperationFailed {
                        operation: "Sweeping Schwarz setup",
                        reason: format!("owned DOF {global} missing from its extended layer"),
                    })?;
                owned_global.push(global);
                owned_local.push(local);
            }
            built.push(SweepLayer {
                subdomain,
                owned_global,
                owned_local,
            });
        }
        Ok(Self {
            layers: built,
            matrix: matrix.clone(),
            n,
            wavenumber,
        })
    }

    /// Build sweeping layers from explicit ordered core-DOF sets,
    /// panicking only on invalid input (use [`Self::try_from_csr_layers`]
    /// when the layers come from an untrusted source).
    pub fn from_csr_layers(
        matrix: &CsrMatrix<T>,
        layers: &[Vec<usize>],
        overlap: usize,
        wavenumber: f64,
    ) -> Self {
        Self::try_from_csr_layers(matrix, layers, overlap, wavenumber)
            .expect("sweeping Schwarz setup requires a partition covering every DOF exactly once")
    }

    /// Number of sweep layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Wavenumber used for the interface transmission condition.
    pub fn wavenumber(&self) -> f64 {
        self.wavenumber
    }

    /// One multiplicative correction pass over `order`: for each layer,
    /// form the current residual on the extended set, solve locally,
    /// and add the owned part into `y`.
    fn sweep_into(
        &self,
        r: &[T],
        y: &mut [T],
        order: impl Iterator<Item = usize>,
        scratch: &mut [T],
    ) {
        let row_ptrs = &self.matrix.row_ptrs;
        let col_indices = &self.matrix.col_indices;
        let values = &self.matrix.values;
        for s in order {
            let layer = &self.layers[s];
            let ext = &layer.subdomain.global_indices;
            for (j, &global) in ext.iter().enumerate() {
                let mut acc = r[global];
                for idx in row_ptrs[global]..row_ptrs[global + 1] {
                    let col = col_indices[idx];
                    if col < self.n {
                        acc -= values[idx] * y[col];
                    }
                }
                scratch[j] = acc;
            }
            layer.subdomain.solve_in_place(&mut scratch[..ext.len()]);
            for (o, &local) in layer.owned_local.iter().enumerate() {
                y[layer.owned_global[o]] += scratch[local];
            }
        }
    }
}

impl<T: ComplexField> Preconditioner<T> for SweepingSchwarzPreconditioner<T> {
    fn apply(&self, r: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); self.n];
        self.apply_into(r, &mut y);
        y
    }

    fn apply_into(&self, r: &[T], y: &mut [T]) {
        assert_eq!(r.len(), self.n, "Sweeping Schwarz RHS length mismatch");
        assert_eq!(y.len(), self.n, "Sweeping Schwarz output length mismatch");
        y.fill(T::zero());
        if self.layers.is_empty() {
            return;
        }
        let max_ext = self
            .layers
            .iter()
            .map(|layer| layer.subdomain.global_indices.len())
            .max()
            .unwrap_or(0);
        let mut scratch = vec![T::zero(); max_ext];
        let count = self.layers.len();
        // Double sweep: forward pass transports information along the
        // layer order, backward pass transports it back.
        self.sweep_into(r, y, 0..count, &mut scratch);
        self.sweep_into(r, y, (0..count).rev(), &mut scratch);
    }

    fn try_apply(&self, r: &[T]) -> Result<Vec<T>, SolverError> {
        if r.len() != self.n {
            return Err(SolverError::DimensionMismatch {
                expected: self.n,
                got: r.len(),
            });
        }
        if r.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation: "Sweeping Schwarz apply",
                reason: "input contains NaN or infinity".into(),
            });
        }
        let applied = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.apply(r)))
        {
            Ok(applied) => applied,
            Err(_) => {
                return Err(SolverError::OperationFailed {
                    operation: "Sweeping Schwarz apply",
                    reason: "preconditioner application panicked".into(),
                });
            }
        };
        if applied.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation: "Sweeping Schwarz apply",
                reason: "output contains NaN or infinity".into(),
            });
        }
        Ok(applied)
    }
}

/// Partition DOFs into `num_layers` contiguous slabs along a
/// coordinate axis, for use as sweeping layers.
///
/// `coords[i]` is the position of DOF `i`; DOFs are ordered by
/// `coords[i][axis]` and split into contiguous slabs. Returns an
/// error for an empty coordinate set, `axis >= 3`, or
/// `num_layers == 0`; more layers than DOFs are capped at one DOF
/// per layer.
pub fn coordinate_layers(
    coords: &[[f64; 3]],
    num_layers: usize,
    axis: usize,
) -> Result<Vec<Vec<usize>>, SolverError> {
    if coords.is_empty() {
        return Err(SolverError::OperationFailed {
            operation: "Sweeping Schwarz layering",
            reason: "coordinate set is empty".into(),
        });
    }
    if axis >= 3 {
        return Err(SolverError::OperationFailed {
            operation: "Sweeping Schwarz layering",
            reason: format!("axis {axis} out of range (must be 0, 1, or 2)"),
        });
    }
    if num_layers == 0 {
        return Err(SolverError::OperationFailed {
            operation: "Sweeping Schwarz layering",
            reason: "at least one layer is required".into(),
        });
    }
    let n = coords.len();
    let count = num_layers.min(n);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        coords[a][axis]
            .partial_cmp(&coords[b][axis])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut layers = Vec::with_capacity(count);
    for (k, chunk) in order.chunks(n.div_ceil(count)).enumerate() {
        if k >= count {
            break;
        }
        layers.push(chunk.to_vec());
    }
    Ok(layers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use sonium_solvers_iterative::{GmresConfig, gmres_preconditioned};
    use sonium_solvers_sys::topology::{NumaNode, SocketInfo};

    /// 5-point `-Δ` on an `m × m` grid with Dirichlet BCs (unit spacing).
    fn poisson_2d(m: usize) -> CsrMatrix<Complex64> {
        helmholtz_2d(m, 0.0)
    }

    /// 5-point `-Δ - k²` on an `m × m` grid (unit spacing). With `k²` inside
    /// the `(0, 8)` spectrum of the discrete `-Δ` this is an indefinite
    /// Helmholtz operator.
    fn helmholtz_2d(m: usize, k: f64) -> CsrMatrix<Complex64> {
        let n = m * m;
        let diag = 4.0 - k * k;
        let mut triplets = Vec::with_capacity(5 * n);
        for iy in 0..m {
            for ix in 0..m {
                let i = iy * m + ix;
                triplets.push((i, i, Complex64::new(diag, 0.0)));
                if ix > 0 {
                    triplets.push((i, i - 1, Complex64::new(-1.0, 0.0)));
                }
                if ix + 1 < m {
                    triplets.push((i, i + 1, Complex64::new(-1.0, 0.0)));
                }
                if iy > 0 {
                    triplets.push((i, i - m, Complex64::new(-1.0, 0.0)));
                }
                if iy + 1 < m {
                    triplets.push((i, i + m, Complex64::new(-1.0, 0.0)));
                }
            }
        }
        CsrMatrix::from_triplets(n, n, triplets)
    }

    fn test_gmres_config() -> GmresConfig<f64> {
        GmresConfig {
            max_iterations: 500,
            restart: 80,
            tolerance: 1e-8,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        }
    }

    fn create_test_matrix() -> CsrMatrix<Complex64> {
        let n = 20;
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));

        // Create a 2D Laplacian-like matrix (5-point stencil pattern)
        for i in 0..n {
            dense[[i, i]] = Complex64::new(4.0, 0.0);
            if i > 0 {
                dense[[i, i - 1]] = Complex64::new(-1.0, 0.0);
            }
            if i < n - 1 {
                dense[[i, i + 1]] = Complex64::new(-1.0, 0.0);
            }
            // Add some off-diagonal connections for more interesting structure
            if i >= 5 {
                dense[[i, i - 5]] = Complex64::new(-0.5, 0.0);
            }
            if i < n - 5 {
                dense[[i, i + 5]] = Complex64::new(-0.5, 0.0);
            }
        }

        CsrMatrix::from_dense(&dense, 1e-15)
    }

    fn create_diagonal_matrix(n: usize, value: Complex64) -> CsrMatrix<Complex64> {
        let triplets = (0..n).map(|i| (i, i, value)).collect();
        CsrMatrix::from_triplets(n, n, triplets)
    }

    fn synthetic_topology(nodes: usize, sockets: usize) -> MachineTopology {
        let cpus_per_node = 2;
        let numa_nodes = (0..nodes)
            .map(|id| NumaNode {
                id,
                logical_cpus: ((id * cpus_per_node)..((id + 1) * cpus_per_node)).collect(),
                memory_bytes: Some(64 << 30),
            })
            .collect::<Vec<_>>();
        let socket_infos = (0..sockets)
            .map(|id| {
                let node_ids = (0..nodes)
                    .filter(|node_id| node_id % sockets == id)
                    .collect::<Vec<_>>();
                let logical_cpus = node_ids
                    .iter()
                    .flat_map(|&node_id| numa_nodes[node_id].logical_cpus.iter().copied())
                    .collect::<Vec<_>>();
                SocketInfo {
                    id,
                    logical_cpus,
                    numa_node_ids: node_ids,
                }
            })
            .collect::<Vec<_>>();

        MachineTopology {
            total_logical_cpus: nodes * cpus_per_node,
            numa_nodes,
            sockets: socket_infos,
            notes: Vec::new(),
        }
    }

    #[test]
    fn auto_subdomain_count_balances_across_topology_domains() {
        let topology = synthetic_topology(4, 2);
        let count = auto_subdomain_count_for_topology(500_000, 10, &topology);

        assert_eq!(count, 8);
        assert_eq!(count % topology.numa_nodes.len(), 0);
    }

    #[test]
    fn auto_subdomain_count_keeps_small_matrices_single_domain() {
        let topology = synthetic_topology(4, 2);
        let count = auto_subdomain_count_for_topology(10_000, 10, &topology);

        assert_eq!(count, 1);
    }

    #[test]
    fn test_schwarz_basic() {
        let matrix = create_test_matrix();
        let precond = AdditiveSchwarzPreconditioner::from_csr(&matrix, 4, 1);

        let r = (0..20)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();
        let result = precond.apply(&r);

        assert_eq!(result.len(), 20);
        assert!(result.iter().all(|x| x.norm() < 100.0));
    }

    #[test]
    fn test_oras_adds_impedance_on_artificial_interface() {
        let matrix = create_test_matrix();
        let precond = AdditiveSchwarzPreconditioner::from_csr_oras(&matrix, 2, 0, 3.0);
        let first_subdomain = &precond.subdomains[0];
        let local_row = first_subdomain
            .global_indices
            .iter()
            .position(|&idx| idx == 9)
            .expect("first partition should contain the interface row");

        let mut diagonal = None;
        for idx in
            first_subdomain.local_row_ptrs[local_row]..first_subdomain.local_row_ptrs[local_row + 1]
        {
            if first_subdomain.local_col_indices[idx] == local_row {
                diagonal = Some(first_subdomain.local_values[idx]);
                break;
            }
        }

        let diagonal = diagonal.expect("interface row should keep a diagonal entry");
        assert!(
            diagonal.im > 0.0,
            "ORAS should add positive imaginary impedance, got {diagonal:?}"
        );
    }

    #[test]
    fn test_oras_apply_is_finite() {
        let matrix = create_test_matrix();
        let precond = AdditiveSchwarzPreconditioner::from_csr_oras(&matrix, 4, 1, 2.5);

        let r = (0..20)
            .map(|i| Complex64::new((i as f64).cos(), (i as f64).sin()))
            .collect::<Vec<_>>();
        let result = precond.apply(&r);

        assert_eq!(result.len(), 20);
        assert!(result.iter().all(|x| x.re.is_finite() && x.im.is_finite()));
    }

    #[test]
    fn test_schwarz_from_geometry_partitions() {
        let matrix = create_test_matrix();
        let partitions = vec![(0..10).collect(), (10..20).collect()];
        let precond = AdditiveSchwarzPreconditioner::from_csr_partitions(&matrix, &partitions, 1);

        let (num_subdomains, min_size, max_size, avg_size) = precond.stats();
        assert_eq!(num_subdomains, 2);
        assert!(min_size > 0);
        assert!(max_size >= min_size);
        assert!(avg_size >= 10.0);

        let r = vec![Complex64::new(1.0, 0.0); 20];
        let result = precond.apply(&r);
        assert_eq!(result.len(), 20);
        assert!(result.iter().all(|x| x.re.is_finite() && x.im.is_finite()));
    }

    #[test]
    fn test_schwarz_parallel_apply_matches_sequential() {
        let n = 1200;
        let matrix = create_diagonal_matrix(n, Complex64::new(3.0, 0.0));
        let precond = AdditiveSchwarzPreconditioner::from_csr(&matrix, 8, 1);

        let r = (0..n)
            .map(|i| Complex64::new((i as f64 + 1.0).sin(), (i as f64).cos()))
            .collect::<Vec<_>>();
        let sequential = precond.apply_sequential(&r);
        let parallel = precond.apply_parallel(&r);

        assert_eq!(parallel.len(), n);
        for (seq, par) in sequential.iter().zip(&parallel) {
            assert!((*seq - *par).norm() < 1e-12);
        }
    }

    #[test]
    fn test_schwarz_stats() {
        let matrix = create_test_matrix();
        let precond = AdditiveSchwarzPreconditioner::from_csr(&matrix, 4, 2);

        let (num_subdomains, min_size, max_size, avg_size) = precond.stats();

        assert_eq!(num_subdomains, 4);
        assert!(min_size > 0);
        assert!(max_size >= min_size);
        assert!(avg_size > 0.0);
        // With overlap, subdomains should be larger than n/4
        assert!(avg_size > 5.0);
    }

    #[test]
    fn test_schwarz_with_gmres() {
        let matrix = create_test_matrix();
        let precond = AdditiveSchwarzPreconditioner::from_csr(&matrix, 4, 2);

        let b = (0..20)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();

        let config = GmresConfig {
            max_iterations: 100,
            restart: 20,
            tolerance: 1e-8,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let sol = gmres_preconditioned(&matrix, &precond, &b, &config);
        assert!(sol.converged, "GMRES with Schwarz should converge");
    }

    #[test]
    fn test_schwarz_overlap_effect() {
        let matrix = create_test_matrix();

        // Test with different overlap sizes
        let precond_no_overlap = AdditiveSchwarzPreconditioner::from_csr(&matrix, 4, 0);
        let precond_overlap_1 = AdditiveSchwarzPreconditioner::from_csr(&matrix, 4, 1);
        let precond_overlap_2 = AdditiveSchwarzPreconditioner::from_csr(&matrix, 4, 2);

        let b = (0..20)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();

        let config = GmresConfig {
            max_iterations: 100,
            restart: 20,
            tolerance: 1e-8,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let sol_no_overlap = gmres_preconditioned(&matrix, &precond_no_overlap, &b, &config);
        let sol_overlap_1 = gmres_preconditioned(&matrix, &precond_overlap_1, &b, &config);
        let sol_overlap_2 = gmres_preconditioned(&matrix, &precond_overlap_2, &b, &config);

        // All should converge
        assert!(sol_no_overlap.converged);
        assert!(sol_overlap_1.converged);
        assert!(sol_overlap_2.converged);

        // More overlap should generally help (or at least not hurt much)
        // Note: This may not always hold for small test problems
    }

    #[test]
    fn test_schwarz_small_pivot_skip_zeros_l() {
        // Regression test for issue #13: when a pivot is too small and skipped,
        // the original A entry must NOT be copied into L. Otherwise L is
        // corrupted and the forward substitution is wrong.
        let n = 4;
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        // Row 0: well-conditioned diagonal
        dense[[0, 0]] = Complex64::new(4.0, 0.0);
        dense[[0, 1]] = Complex64::new(-1.0, 0.0);
        // Row 1: diagonal is tiny, off-diagonal is large
        // The pivot u_11 will be tiny, so elimination of A[2,1] using row 1
        // should be skipped and L[2,1] must be zero.
        dense[[1, 0]] = Complex64::new(-1.0, 0.0);
        dense[[1, 1]] = Complex64::new(1e-15, 0.0); // tiny pivot
        dense[[1, 2]] = Complex64::new(-1.0, 0.0);
        // Row 2: has entry in column 1 (below the tiny pivot)
        dense[[2, 1]] = Complex64::new(-1.0, 0.0);
        dense[[2, 2]] = Complex64::new(4.0, 0.0);
        dense[[2, 3]] = Complex64::new(-1.0, 0.0);
        // Row 3
        dense[[3, 2]] = Complex64::new(-1.0, 0.0);
        dense[[3, 3]] = Complex64::new(4.0, 0.0);

        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let precond = AdditiveSchwarzPreconditioner::from_csr(&matrix, 1, 0);

        // Single subdomain: inspect its L factor directly
        let sub = &precond.subdomains[0];
        // Find L entry for row 2, column 1
        let mut l_21 = None;
        for idx in sub.l_row_ptrs[2]..sub.l_row_ptrs[3] {
            if sub.l_col_indices[idx] == 1 {
                l_21 = Some(sub.l_values[idx]);
                break;
            }
        }

        // Because the pivot u_11 was skipped, L[2,1] should be zero,
        // not the original A[2,1] = -1.0.
        assert!(
            l_21.is_none_or(|v| v.norm() < 1e-12),
            "L[2,1] should be zero after skipped pivot, got {:?}",
            l_21
        );
    }

    #[test]
    fn test_schwarz_local_ilu_matches_canonical_ilu() {
        // The Schwarz subdomain factorization must produce the same ILU(0)
        // factors as the canonical `IluPreconditioner` on the same matrix.
        // The first matrix exercises the pivot-skip path (row 1 loses its
        // diagonal to the CSR threshold), where the historical copies
        // diverged (Schwarz zeroed the skipped L entry, ilu.rs left the
        // stale value); the second is a well-conditioned stencil matrix.
        let mut skip_dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(4, 4, Complex64::new(0.0, 0.0));
        skip_dense[[0, 0]] = Complex64::new(4.0, 0.0);
        skip_dense[[0, 1]] = Complex64::new(-1.0, 0.0);
        skip_dense[[1, 0]] = Complex64::new(-1.0, 0.0);
        skip_dense[[1, 1]] = Complex64::new(1e-15, 0.0); // dropped by from_dense
        skip_dense[[1, 2]] = Complex64::new(-1.0, 0.0);
        skip_dense[[2, 1]] = Complex64::new(-1.0, 0.0);
        skip_dense[[2, 2]] = Complex64::new(4.0, 0.0);
        skip_dense[[2, 3]] = Complex64::new(-1.0, 0.0);
        skip_dense[[3, 2]] = Complex64::new(-1.0, 0.0);
        skip_dense[[3, 3]] = Complex64::new(4.0, 0.0);
        let skip_matrix = CsrMatrix::from_dense(&skip_dense, 1e-15);

        for matrix in [skip_matrix, create_test_matrix()] {
            let n = matrix.num_rows;
            // One subdomain, no overlap: the local matrix is the global
            // matrix and the apply reduces to the local ILU solve (weight 1).
            let schwarz = AdditiveSchwarzPreconditioner::from_csr(&matrix, 1, 0);
            let ilu = crate::IluPreconditioner::from_csr(&matrix);

            for trial in 0..3 {
                let r: Vec<Complex64> = (0..n)
                    .map(|i| {
                        Complex64::new(
                            ((i + trial * n) as f64 * 0.7).sin(),
                            ((i * (trial + 1)) as f64 * 1.3).cos(),
                        )
                    })
                    .collect();
                let x_schwarz = schwarz.apply(&r);
                let x_ilu = ilu.apply(&r);
                for i in 0..n {
                    assert!(
                        (x_schwarz[i] - x_ilu[i]).norm() < 1e-12,
                        "Schwarz local ILU diverges from canonical ILU at [{i}] \
                         (trial {trial}): schwarz={:?} ilu={:?}",
                        x_schwarz[i],
                        x_ilu[i]
                    );
                }
            }
        }
    }

    #[test]
    fn test_schwarz_zero_pivot_preserves_complex_phase() {
        // Regression test: near-zero pivot fallback must preserve the complex
        // phase, not replace the entry with a purely real inverse.
        let n = 3;
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        dense[[0, 0]] = Complex64::new(4.0, 0.0);
        // Near-zero purely imaginary diagonal -- phase MUST be preserved
        dense[[1, 1]] = Complex64::new(0.0, 1e-12);
        dense[[2, 2]] = Complex64::new(5.0, 0.0);

        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let precond = AdditiveSchwarzPreconditioner::from_csr(&matrix, 1, 0);

        // Single subdomain: check that u_diag_inv preserves phase
        let sub = &precond.subdomains[0];
        let inv_1 = sub.u_diag_inv[1];

        // For a purely imaginary diagonal 0 + i*1e-12, the inverse should be
        // approximately 0 - i*1e12 (purely imaginary, negative).
        assert!(
            inv_1.norm().is_finite(),
            "Inverse diagonal is not finite: {:?}",
            inv_1
        );
        assert!(
            inv_1.re.abs() < 1.0,
            "Real part too large -- phase lost? re={}, im={}",
            inv_1.re,
            inv_1.im
        );
        assert!(
            inv_1.im.abs() > 1e10,
            "Imag part too small -- phase lost? re={}, im={}",
            inv_1.re,
            inv_1.im
        );

        // Apply preconditioner to e_1 and verify the result is also purely imaginary
        let b = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let result = precond.apply(&b);
        let r1 = result[1];
        assert!(
            r1.re.abs() < 1.0,
            "Preconditioner result real part too large: re={}, im={}",
            r1.re,
            r1.im
        );
        assert!(
            r1.im.abs() > 1e10,
            "Preconditioner result imag part too small: re={}, im={}",
            r1.re,
            r1.im
        );
    }

    #[test]
    fn test_two_level_coarse_operator_nonsingular_on_poisson() {
        let matrix = poisson_2d(16);
        let precond = AdditiveSchwarzPreconditioner::from_csr(&matrix, 4, 1)
            .with_nicolaides_coarse_space(&matrix);

        // A is SPD and Z has full column rank (disjoint supports), so
        // E = ZᵀAZ is SPD and the sparse LU factorization must succeed.
        assert!(precond.has_coarse_space(), "coarse space should attach");
        assert_eq!(precond.coarse_dimension(), 4);
        assert_eq!(precond.coarse_setup(), SchwarzCoarseSetup::Attached);

        // One-level default stays unchanged: no coarse space unless requested.
        let one_level = AdditiveSchwarzPreconditioner::from_csr(&matrix, 4, 1);
        assert!(!one_level.has_coarse_space());
        assert_eq!(one_level.coarse_dimension(), 0);
        assert_eq!(one_level.coarse_setup(), SchwarzCoarseSetup::NotRequested);
    }

    #[test]
    fn test_two_level_coarse_correction_exact_for_piecewise_constant_modes() {
        let m = 16;
        let n = m * m;
        let matrix = poisson_2d(m);
        let precond = AdditiveSchwarzPreconditioner::from_csr(&matrix, 4, 1)
            .with_nicolaides_coarse_space(&matrix);
        let coarse = precond.coarse.as_ref().expect("coarse space attached");
        let nc = coarse.dim();

        // Any mode in range(Z): piecewise constant per core subdomain.
        let c: Vec<Complex64> = (0..nc)
            .map(|j| Complex64::new(j as f64 + 1.0, 0.5 * j as f64))
            .collect();
        let x = coarse.z.matvec(&c);
        let r = matrix.matvec(&x);

        // Galerkin identity: E = ZᵀAZ ⟹ Z·E⁻¹·Zᵀ·(A·Z·c) = Z·c exactly
        // (up to sparse LU roundoff).
        let mut y = vec![Complex64::new(0.0, 0.0); n];
        coarse.apply_add(&r, &mut y);

        let err = y
            .iter()
            .zip(&x)
            .map(|(yi, xi)| (*yi - *xi).norm())
            .fold(0.0_f64, f64::max);
        assert!(
            err < 1e-8,
            "coarse correction must recover constant-in-subdomain modes exactly, err={err:e}"
        );
    }

    #[test]
    fn two_level_coarse_adjoint_actions_are_exact() {
        let matrix = poisson_2d(8);
        let preconditioner = AdditiveSchwarzPreconditioner::from_csr(&matrix, 4, 1)
            .with_nicolaides_coarse_space(&matrix);
        assert_eq!(
            preconditioner.transpose_capability(),
            AdjointCapability::Exact
        );
        assert_eq!(
            preconditioner.hermitian_capability(),
            AdjointCapability::Exact
        );

        let x: Vec<Complex64> = (0..matrix.num_rows)
            .map(|i| Complex64::new((i as f64 + 1.0) * 0.13, (i as f64 - 3.0) * 0.07))
            .collect();
        let y: Vec<Complex64> = (0..matrix.num_rows)
            .map(|i| Complex64::new((i as f64 + 2.0) * -0.11, (i as f64 + 1.0) * 0.05))
            .collect();
        let bx = preconditioner.apply(&x);
        let bty = preconditioner.apply_transpose(&y);
        let bhy = preconditioner.apply_hermitian(&y);
        let transpose_lhs: Complex64 = y.iter().zip(&bx).map(|(yi, bxi)| *yi * *bxi).sum();
        let transpose_rhs: Complex64 = bty.iter().zip(&x).map(|(btyi, xi)| *btyi * *xi).sum();
        let hermitian_lhs: Complex64 = y.iter().zip(&bx).map(|(yi, bxi)| yi.conj() * *bxi).sum();
        let hermitian_rhs: Complex64 = bhy.iter().zip(&x).map(|(bhyi, xi)| bhyi.conj() * *xi).sum();
        assert!((transpose_lhs - transpose_rhs).norm() < 1e-8);
        assert!((hermitian_lhs - hermitian_rhs).norm() < 1e-8);
    }

    /// Disjoint square `p × p` block partition of an `m × m` grid.
    fn block_partitions(m: usize, p: usize) -> Vec<Vec<usize>> {
        assert_eq!(m % p, 0);
        let block = m / p;
        let mut partitions = Vec::with_capacity(p * p);
        for by in 0..p {
            for bx in 0..p {
                let mut core = Vec::with_capacity(block * block);
                for iy in (by * block)..((by + 1) * block) {
                    for ix in (bx * block)..((bx + 1) * block) {
                        core.push(iy * m + ix);
                    }
                }
                partitions.push(core);
            }
        }
        partitions
    }

    fn grid_coords_2d(m: usize) -> Vec<[f64; 3]> {
        let h = 1.0 / (m as f64 + 1.0);
        (0..m * m)
            .map(|i| {
                let ix = i % m;
                let iy = i / m;
                [(ix as f64 + 1.0) * h, (iy as f64 + 1.0) * h, 0.0]
            })
            .collect()
    }

    #[test]
    fn test_coordinate_layers_validation() {
        let coords = grid_coords_2d(8);
        assert!(coordinate_layers(&[], 2, 0).is_err());
        assert!(coordinate_layers(&coords, 0, 0).is_err());
        assert!(coordinate_layers(&coords, 2, 3).is_err());
        let layers = coordinate_layers(&coords, 4, 0).expect("valid layering");
        assert_eq!(layers.len(), 4);
        let mut covered: Vec<usize> = layers.iter().flatten().copied().collect();
        covered.sort_unstable();
        assert_eq!(covered, (0..64).collect::<Vec<_>>());
        // Slabs along x: first layer holds the lowest-x columns.
        assert!(layers[0].iter().all(|&i| i % 8 < 2));
    }

    #[test]
    fn test_sweeping_setup_rejects_bad_input() {
        let matrix = helmholtz_2d(12, 0.12);
        assert!(SweepingSchwarzPreconditioner::try_from_csr_layers(&matrix, &[], 1, 0.12).is_err());
        let layers = coordinate_layers(&grid_coords_2d(12), 3, 0).unwrap();
        assert!(
            SweepingSchwarzPreconditioner::try_from_csr_layers(&matrix, &layers, 1, 0.0).is_err()
        );
        assert!(
            SweepingSchwarzPreconditioner::try_from_csr_layers(&matrix, &layers, 1, f64::NAN)
                .is_err()
        );
        let overlapping = vec![(0..100).collect::<Vec<_>>(), (50..144).collect::<Vec<_>>()];
        assert!(
            SweepingSchwarzPreconditioner::try_from_csr_layers(&matrix, &overlapping, 1, 0.12)
                .is_err()
        );
        let sweep = SweepingSchwarzPreconditioner::try_from_csr_layers(&matrix, &layers, 1, 0.12)
            .expect("valid sweeping setup");
        assert_eq!(sweep.num_layers(), 3);
        assert_eq!(sweep.wavenumber(), 0.12);
    }

    #[test]
    fn test_sweeping_gmres_beats_additive_oras_helmholtz() {
        // 2D Helmholtz -Δ - k² on a 48x48 grid (same operator as the
        // two-level ORAS test above), 4 x-slabs vs 2x2 blocks, overlap 1,
        // first-order ABC transmission on both sides, GMRES restart 50.
        // Measured GMRES iterations (restart 50, tol 1e-8):
        //   additive ORAS (2x2 blocks)   72
        //   sweeping double sweep        37   <- ~2x fewer
        // Larger release probe (96x96, 9216 DOFs, k=0.3, 8 slabs,
        // restart 100): sweeping 900 iters / 3.49 s, additive ORAS
        // 2300 iters / 8.43 s, BlockJacobi+ILUT not converged in
        // 200000 iters / 823 s. The gap widens with problem size;
        // plain ILUT fails outright on this indefinite operator.
        // 3D probe (32^3, 32768 DOFs, k=0.5, 8 slabs, restart 100):
        // sweeping 200 iters / 2.19 s, additive ORAS 500 / 3.40 s,
        // BlockJacobi+ILUT 600 / 4.94 s — sweeping wins iterations
        // and wall-clock in 3D as well.
        // 2D scaling trend (m=128, 16384 DOFs, k=0.3, restart 100,
        // counts reproduced over two runs): sweeping 10800 / 73 s,
        // additive ORAS 24600 / 133 s, BlockJacobi+ILU did not finish
        // within 280 s. Iteration counts grow for both Schwarz
        // variants without PML/coarse correction — expected, and the
        // reason PML transmission is the documented follow-up.
        // The sweeping count must not exceed the additive count: one
        // double sweep transports information across all four slabs,
        // while the additive combination needs one iteration per slab
        // diameter.
        let m = 48;
        let n = m * m;
        let k = 0.12;
        let matrix = helmholtz_2d(m, k);
        let b: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.7).sin(), (i as f64 * 1.3).cos()))
            .collect();
        let config = GmresConfig {
            max_iterations: 500,
            restart: 50,
            tolerance: 1e-8,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };
        let layers = coordinate_layers(&grid_coords_2d(m), 4, 0).expect("layers");
        let sweeping = SweepingSchwarzPreconditioner::try_from_csr_layers(&matrix, &layers, 1, k)
            .expect("sweeping setup");
        let blocks = block_partitions(m, 2);
        let additive =
            AdditiveSchwarzPreconditioner::from_csr_partitions_oras(&matrix, &blocks, 1, k);
        let sol_sweep = gmres_preconditioned(&matrix, &sweeping, &b, &config);
        let sol_add = gmres_preconditioned(&matrix, &additive, &b, &config);
        assert!(sol_add.converged, "additive ORAS GMRES should converge");
        assert!(sol_sweep.converged, "sweeping GMRES should converge");
        assert!(
            sol_sweep.iterations <= sol_add.iterations,
            "sweeping ({} iters) should need no more iterations than additive ORAS ({} iters)",
            sol_sweep.iterations,
            sol_add.iterations
        );
    }

    #[test]
    fn test_two_level_gmres_flat_vs_subdomain_count_poisson() {
        // 2D Poisson on a 64x64 grid, square block subdomains. Measured
        // GMRES iterations (restart 80, tol 1e-8):
        //   subdomains   4   16   64
        //   one-level   54   56   60   <- grows with subdomain count
        //   two-level   46   46   38   <- flat (even decreasing)
        let m = 64; // n = 4096
        let n = m * m;
        let matrix = poisson_2d(m);
        let b: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.7).sin(), 0.0))
            .collect();
        let config = test_gmres_config();

        let mut one_level_iters = Vec::new();
        let mut two_level_iters = Vec::new();
        for p in [2_usize, 4, 8] {
            let partitions = block_partitions(m, p);
            let one_level =
                AdditiveSchwarzPreconditioner::from_csr_partitions(&matrix, &partitions, 1);
            let two_level =
                AdditiveSchwarzPreconditioner::from_csr_partitions(&matrix, &partitions, 1)
                    .with_nicolaides_coarse_space(&matrix);
            assert!(two_level.has_coarse_space());

            let sol_one = gmres_preconditioned(&matrix, &one_level, &b, &config);
            let sol_two = gmres_preconditioned(&matrix, &two_level, &b, &config);
            assert!(sol_one.converged, "one-level GMRES should converge");
            assert!(sol_two.converged, "two-level GMRES should converge");
            one_level_iters.push(sol_one.iterations);
            two_level_iters.push(sol_two.iterations);
        }

        // One-level iterations grow as the subdomain count goes 4 -> 64 ...
        assert!(
            one_level_iters[2] > one_level_iters[0],
            "one-level iterations should grow with subdomain count: {one_level_iters:?}"
        );
        // ... while the two-level count stays nearly flat ...
        assert!(
            two_level_iters[2] as f64 <= 1.2 * two_level_iters[0].max(1) as f64,
            "two-level iterations should stay nearly flat: {two_level_iters:?}"
        );
        // ... and clearly beats one-level at the highest subdomain count.
        assert!(
            two_level_iters[2] < one_level_iters[2],
            "two-level should need fewer iterations at 64 subdomains: \
             one-level={one_level_iters:?}, two-level={two_level_iters:?}"
        );
    }

    #[test]
    fn test_two_level_oras_gmres_flat_vs_subdomain_count_helmholtz() {
        // 2D Helmholtz -Δ - k² on a 48x48 grid, k = 0.12: k² = 0.0144 sits
        // just above the smallest eigenvalue of -Δ (≈ 0.0082), so the
        // operator is genuinely indefinite (one negative eigenvalue) while
        // still mild enough that constant-per-subdomain (Nicolaides) coarse
        // modes capture the offending smooth mode. Measured GMRES iterations
        // (restart 50, tol 1e-8):
        //   subdomains   4   16   64
        //   one-level   72   62   81   <- grows again at 64 subdomains
        //   two-level   50   50   46   <- flat, and fewer throughout
        //
        // NOTE: for stronger indefiniteness (more negative eigenvalues,
        // multiple wavelengths per subdomain) the Nicolaides coarse space is
        // the WRONG space for wave problems — measured two-level counts are
        // then neutral-to-worse than one-level. That regime needs a spectral
        // (GenEO) or plane-wave coarse space; see the doc comment on
        // `with_nicolaides_coarse_space`.
        let m = 48;
        let n = m * m;
        let k = 0.12;
        let matrix = helmholtz_2d(m, k);
        let b: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.7).sin(), (i as f64 * 1.3).cos()))
            .collect();
        let config = GmresConfig {
            max_iterations: 80,
            restart: 50,
            tolerance: 1e-8,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let mut one_level_iters = Vec::new();
        let mut two_level_iters = Vec::new();
        for p in [2_usize, 4, 8] {
            let partitions = block_partitions(m, p);
            let one_level =
                AdditiveSchwarzPreconditioner::from_csr_partitions_oras(&matrix, &partitions, 1, k);
            let two_level =
                AdditiveSchwarzPreconditioner::from_csr_partitions_oras(&matrix, &partitions, 1, k)
                    .with_nicolaides_coarse_space(&matrix);
            assert!(two_level.has_coarse_space());

            let sol_one = gmres_preconditioned(&matrix, &one_level, &b, &config);
            let sol_two = gmres_preconditioned(&matrix, &two_level, &b, &config);
            assert!(sol_one.converged, "one-level ORAS GMRES should converge");
            assert!(sol_two.converged, "two-level ORAS GMRES should converge");
            one_level_iters.push(sol_one.iterations);
            two_level_iters.push(sol_two.iterations);
        }

        // One-level iterations grow again as subdomains go 16 -> 64 ...
        assert!(
            one_level_iters[2] > one_level_iters[1],
            "one-level iterations should grow at high subdomain count: {one_level_iters:?}"
        );
        // ... while the two-level count stays nearly flat ...
        assert!(
            two_level_iters[2] as f64 <= 1.2 * two_level_iters[0].max(1) as f64,
            "two-level iterations should stay nearly flat: {two_level_iters:?}"
        );
        // ... and needs fewer iterations than one-level at every count.
        for (two, one) in two_level_iters.iter().zip(&one_level_iters) {
            assert!(
                two < one,
                "two-level should beat one-level at every subdomain count: \
                 one-level={one_level_iters:?}, two-level={two_level_iters:?}"
            );
        }
    }

    #[test]
    fn test_two_level_drops_singular_coarse_operator_and_still_converges() {
        // Undamped Helmholtz at k = 0.5 on a 32x32 grid: E = ZᵀAZ samples
        // the smooth modes where -Δ - k² is negative, and comes out singular
        // for this subdomain count. The coarse space must be dropped
        // (one-level fallback) and GMRES must still converge.
        let m = 32;
        let n = m * m;
        let k = 0.5;
        let matrix = helmholtz_2d(m, k);
        let partitions = block_partitions(m, 2);

        let precond =
            AdditiveSchwarzPreconditioner::from_csr_partitions_oras(&matrix, &partitions, 1, k)
                .with_nicolaides_coarse_space(&matrix);
        assert!(
            !precond.has_coarse_space(),
            "singular E should drop the coarse space (one-level fallback)"
        );
        assert_eq!(precond.coarse_dimension(), 0);

        let b: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.7).sin(), (i as f64 * 1.3).cos()))
            .collect();
        let config = GmresConfig {
            max_iterations: 80,
            restart: 50,
            tolerance: 1e-8,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };
        let sol = gmres_preconditioned(&matrix, &precond, &b, &config);
        assert!(
            sol.converged,
            "GMRES should still converge after the one-level fallback"
        );
    }

    /// Extract column `j` of a coarse basis as a dense vector.
    fn coarse_column<T: ComplexField>(coarse: &CoarseSpace<T>, j: usize) -> Vec<T> {
        let mut e = vec![T::zero(); coarse.dim()];
        e[j] = T::one();
        coarse.z.matvec(&e)
    }

    #[test]
    fn test_geneo_coarse_space_correctness_poisson() {
        // 2D Poisson (SPD), 4 subdomains: the GenEO coarse space must
        // attach, E = ZᵀAZ must be nonsingular, Z must have full column
        // rank with a well-conditioned Gram matrix ZᵀZ, and the coarse
        // correction must recover any mode in range(Z) exactly (Galerkin
        // identity Z·E⁻¹·Zᵀ·A·Z·c = Z·c).
        let m = 16;
        let n = m * m;
        let matrix = poisson_2d(m);
        let partitions = block_partitions(m, 2);
        let precond = AdditiveSchwarzPreconditioner::from_csr_partitions(&matrix, &partitions, 1)
            .with_geneo_coarse_space(&matrix, &GeneoConfig::default());

        assert!(
            precond.has_coarse_space(),
            "GenEO coarse space should attach on Poisson"
        );
        // At least the (near-)constant Neumann mode per subdomain.
        assert!(precond.coarse_dimension() >= partitions.len());
        let coarse = precond.coarse.as_ref().expect("coarse space attached");
        let nc = coarse.dim();

        // Gram matrix G = ZᵀZ: diagonal is 1 (the local eigenvectors are
        // B_i-orthonormal and the GenEO extension is Z = D_i·v, so
        // ||z_j||² = vᵀD_i²v = 1); measured off-diagonal row sums are
        // below 0.55 here, i.e. Gershgorin bounds eigenvalues in
        // (0.45, 1.55) — ZᵀZ is well conditioned.
        let cols: Vec<Vec<Complex64>> = (0..nc).map(|j| coarse_column(coarse, j)).collect();
        let mut max_offdiag_rowsum = 0.0_f64;
        for (j, zj) in cols.iter().enumerate() {
            let mut rowsum = 0.0_f64;
            for (l, zl) in cols.iter().enumerate() {
                let g = Complex64::vec_dot(zj, zl);
                if j == l {
                    assert!(
                        (g.re - 1.0).abs() < 1e-6 && g.im.abs() < 1e-12,
                        "ZᵀZ diagonal should be 1, got {g:?} at column {j}"
                    );
                } else {
                    rowsum += g.norm();
                }
            }
            max_offdiag_rowsum = max_offdiag_rowsum.max(rowsum);
        }
        assert!(
            max_offdiag_rowsum < 0.9,
            "ZᵀZ should be well conditioned (Gershgorin), max off-diagonal row sum =              {max_offdiag_rowsum}"
        );

        // Galerkin identity on a random coarse mode.
        let c: Vec<Complex64> = (0..nc)
            .map(|j| Complex64::new(j as f64 + 1.0, 0.5 * j as f64))
            .collect();
        let x = coarse.z.matvec(&c);
        let r = matrix.matvec(&x);
        let mut y = vec![Complex64::new(0.0, 0.0); n];
        coarse.apply_add(&r, &mut y);
        let err = y
            .iter()
            .zip(&x)
            .map(|(yi, xi)| (*yi - *xi).norm())
            .fold(0.0_f64, f64::max);
        assert!(
            err < 1e-8,
            "GenEO coarse correction must recover range(Z) modes exactly, err={err:e}"
        );
    }

    #[test]
    fn test_geneo_and_nicolaides_are_mutually_exclusive() {
        // Last builder call wins: attaching one coarse space replaces the
        // other (documented behaviour — composing the two would double-
        // count the constant modes and make E singular).
        let m = 16;
        let matrix = poisson_2d(m);
        let partitions = block_partitions(m, 2);
        let geneo_config = GeneoConfig {
            tau: 0.5,
            ..Default::default()
        };

        let nic_dim = AdditiveSchwarzPreconditioner::from_csr_partitions(&matrix, &partitions, 1)
            .with_nicolaides_coarse_space(&matrix)
            .coarse_dimension();
        let geneo_dim = AdditiveSchwarzPreconditioner::from_csr_partitions(&matrix, &partitions, 1)
            .with_geneo_coarse_space(&matrix, &geneo_config)
            .coarse_dimension();
        assert_eq!(nic_dim, 4);
        assert!(geneo_dim > nic_dim, "GenEO keeps more than the constants");

        let nic_then_geneo =
            AdditiveSchwarzPreconditioner::from_csr_partitions(&matrix, &partitions, 1)
                .with_nicolaides_coarse_space(&matrix)
                .with_geneo_coarse_space(&matrix, &geneo_config);
        assert_eq!(nic_then_geneo.coarse_dimension(), geneo_dim);

        let geneo_then_nic =
            AdditiveSchwarzPreconditioner::from_csr_partitions(&matrix, &partitions, 1)
                .with_geneo_coarse_space(&matrix, &geneo_config)
                .with_nicolaides_coarse_space(&matrix);
        assert_eq!(geneo_then_nic.coarse_dimension(), nic_dim);
    }

    #[test]
    fn test_geneo_empty_coarse_space_falls_back_to_one_level() {
        // A threshold no local eigenvalue can pass leaves the coarse space
        // empty: the preconditioner must silently stay one-level.
        let m = 16;
        let matrix = poisson_2d(m);
        let partitions = block_partitions(m, 2);
        let config = GeneoConfig {
            tau: -1e9,
            ..Default::default()
        };
        let precond = AdditiveSchwarzPreconditioner::from_csr_partitions(&matrix, &partitions, 1)
            .with_geneo_coarse_space(&matrix, &config);
        assert!(!precond.has_coarse_space());
        assert_eq!(precond.coarse_dimension(), 0);
        assert_eq!(
            precond.coarse_setup(),
            SchwarzCoarseSetup::FellBackToOneLevel
        );
    }

    #[test]
    fn schwarz_exposes_aggregate_setup_diagnostics() {
        let matrix = poisson_2d(6);
        let preconditioner = AdditiveSchwarzPreconditioner::from_csr(&matrix, 3, 1);
        let diagnostics = preconditioner.diagnostics();

        assert_eq!(diagnostics.skipped_pivots, 0);
        assert_eq!(diagnostics.non_finite_entries, 0);
        assert!(diagnostics.min_abs_pivot.is_some());
        assert_eq!(
            preconditioner.coarse_setup(),
            SchwarzCoarseSetup::NotRequested
        );
    }

    #[test]
    fn checked_oras_rejects_nonphysical_wavenumbers() {
        let matrix = poisson_2d(4);
        let partitions = block_partitions(4, 2);
        for wavenumber in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                AdditiveSchwarzPreconditioner::try_from_csr_partitions_oras(
                    &matrix,
                    &partitions,
                    1,
                    wavenumber
                ),
                Err(SolverError::OperationFailed {
                    operation: "Schwarz ORAS setup",
                    ..
                })
            ));
        }
    }

    #[test]
    fn test_geneo_oras_beats_one_level_strong_helmholtz() {
        // 2D Helmholtz -Δ - k² on a 48x48 grid, k = 0.3: k² = 0.09 sits
        // above ~10 eigenvalues of the discrete -Δ — the strongly
        // indefinite regime where the Nicolaides coarse space is neutral-
        // to-worse (its constants cannot represent the resonances).
        // Measured GMRES iterations (restart 80, tol 1e-8):
        //   subdomains        4    16    64
        //   one-level ORAS  150   152   160   <- grows with subdomain count
        //   Nicolaides      148   149   143   <- neutral (wrong coarse space)
        //   GenEO            27    36    60   <- beats both, bounded growth
        //   GenEO dim        63   104   136
        // (At the milder k = 0.2 the same trend holds: 76/77/86 one-level,
        // 101/101/65 Nicolaides, 71/27/38 GenEO.)
        let m = 48;
        let n = m * m;
        let k = 0.3;
        let matrix = helmholtz_2d(m, k);
        let b: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.7).sin(), (i as f64 * 1.3).cos()))
            .collect();
        let config = GmresConfig {
            max_iterations: 300,
            restart: 80,
            tolerance: 1e-8,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let mut one_level_iters = Vec::new();
        let mut nicolaides_iters = Vec::new();
        let mut geneo_iters = Vec::new();
        for p in [2_usize, 4, 8] {
            let partitions = block_partitions(m, p);
            let one_level =
                AdditiveSchwarzPreconditioner::from_csr_partitions_oras(&matrix, &partitions, 1, k);
            let nicolaides =
                AdditiveSchwarzPreconditioner::from_csr_partitions_oras(&matrix, &partitions, 1, k)
                    .with_nicolaides_coarse_space(&matrix);
            let geneo =
                AdditiveSchwarzPreconditioner::from_csr_partitions_oras(&matrix, &partitions, 1, k)
                    .with_geneo_coarse_space(&matrix, &GeneoConfig::default());
            assert!(geneo.has_coarse_space(), "GenEO coarse space should attach");

            let sol_one = gmres_preconditioned(&matrix, &one_level, &b, &config);
            let sol_nic = gmres_preconditioned(&matrix, &nicolaides, &b, &config);
            let sol_gen = gmres_preconditioned(&matrix, &geneo, &b, &config);
            assert!(sol_one.converged, "one-level ORAS GMRES should converge");
            assert!(
                sol_nic.converged,
                "Nicolaides two-level GMRES should converge"
            );
            assert!(sol_gen.converged, "GenEO two-level GMRES should converge");
            one_level_iters.push(sol_one.iterations);
            nicolaides_iters.push(sol_nic.iterations);
            geneo_iters.push(sol_gen.iterations);
        }

        // GenEO beats one-level AND Nicolaides at every subdomain count.
        for ((&g_it, &one), &nic) in geneo_iters
            .iter()
            .zip(&one_level_iters)
            .zip(&nicolaides_iters)
        {
            assert!(
                g_it < one && g_it < nic,
                "GenEO should beat one-level and Nicolaides:                  one-level={one_level_iters:?}, nicolaides={nicolaides_iters:?},                  geneo={geneo_iters:?}"
            );
        }
        // One-level iterations grow with subdomain count ...
        assert!(
            one_level_iters[2] > one_level_iters[0],
            "one-level iterations should grow with subdomain count: {one_level_iters:?}"
        );
        // ... while GenEO stays flat (measured: it even decreases).
        //
        // The conservative default threshold excludes unstable positive
        // local modes on the finest partition while retaining the
        // indefinite modes that matter to the global solve. This gives
        // bounded growth as the number of subdomains increases.
        assert!(
            geneo_iters[2] as f64 <= 2.5 * geneo_iters[0].max(1) as f64,
            "GenEO iterations should grow much more slowly than one-level:              one-level={one_level_iters:?}, geneo={geneo_iters:?}"
        );
    }
}
