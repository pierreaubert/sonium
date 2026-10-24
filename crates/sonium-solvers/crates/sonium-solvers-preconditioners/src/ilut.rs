//! ILUT/ILUTP preconditioner
//!
//! Incomplete LU factorization with threshold dropping (Saad's ILUT).
//! More flexible than ILU(0): controls fill-in via a drop tolerance and fill factor,
//! allowing better approximation of the true LU factors at the cost of more storage.
//!
//! With [`IlutConfig::pivoting`] (on by default) this is ILUTP: column partial
//! pivoting in the spirit of Saad's ILUTP and ILUPACK. Each step pivots on the
//! largest-magnitude row entry among the not-yet-pivoted columns, so `L U ≈ A P`
//! and solves apply the permutation (`x = P (U⁻¹ L⁻¹ b)`). Pivoting prevents the
//! multiplier blow-up unpivoted ILUT suffers on indefinite systems or matrices
//! with structurally small diagonals. Pivot thresholds are row-norm-relative
//! (`pivot_tolerance * ||A(k,:)||₂`), consistent with the ILU(0) kernel in
//! `ilu.rs`.

use crate::ilu::{IluReordering, finish_scaled_norm, permute_symmetric, update_scaled_norm};
use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use sonium_solvers_core::traits::{ComplexField, Preconditioner, SolverError};
use sonium_solvers_direct::sparse_lu::colamd_ordering;
use sonium_solvers_sparse::CsrMatrix;
use std::collections::BTreeSet;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
#[doc(inline)]
pub use cuda::CudaIlutPreconditioner;

/// Version of the portable ILUT/ILUTP factor checkpoint format.
pub const ILUT_CHECKPOINT_FORMAT_VERSION: u16 = 1;
const ILUT_CHECKPOINT_MAGIC: &[u8; 8] = b"SILTCP01";

fn ilut_checkpoint_error(message: impl Into<String>) -> SolverError {
    SolverError::OperationFailed {
        operation: "ILUT checkpoint",
        reason: message.into(),
    }
}

fn ilut_checkpoint_push_u64(output: &mut Vec<u8>, value: usize) -> Result<(), SolverError> {
    output.extend_from_slice(
        &u64::try_from(value)
            .map_err(|_| ilut_checkpoint_error("usize value does not fit checkpoint format"))?
            .to_le_bytes(),
    );
    Ok(())
}

fn ilut_checkpoint_push_usize_vec(
    output: &mut Vec<u8>,
    values: &[usize],
) -> Result<(), SolverError> {
    ilut_checkpoint_push_u64(output, values.len())?;
    for &value in values {
        ilut_checkpoint_push_u64(output, value)?;
    }
    Ok(())
}

fn ilut_checkpoint_push_u32_vec(output: &mut Vec<u8>, values: &[u32]) -> Result<(), SolverError> {
    ilut_checkpoint_push_u64(output, values.len())?;
    for &value in values {
        output.extend_from_slice(&value.to_le_bytes());
    }
    Ok(())
}

fn ilut_checkpoint_push_values<T: ComplexField>(
    output: &mut Vec<u8>,
    values: &[T],
) -> Result<(), SolverError> {
    ilut_checkpoint_push_u64(output, values.len())?;
    for &value in values {
        let re = value
            .re()
            .to_f64()
            .ok_or_else(|| ilut_checkpoint_error("real factor value cannot convert to f64"))?;
        let im = value
            .im()
            .to_f64()
            .ok_or_else(|| ilut_checkpoint_error("imaginary factor value cannot convert to f64"))?;
        if !re.is_finite() || !im.is_finite() {
            return Err(ilut_checkpoint_error("factor contains non-finite value"));
        }
        output.extend_from_slice(&re.to_le_bytes());
        output.extend_from_slice(&im.to_le_bytes());
    }
    Ok(())
}

struct IlutCheckpointReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> IlutCheckpointReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N], SolverError> {
        let end = self
            .offset
            .checked_add(N)
            .ok_or_else(|| ilut_checkpoint_error("checkpoint offset overflow"))?;
        let slice = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| ilut_checkpoint_error("truncated checkpoint"))?;
        let mut value = [0_u8; N];
        value.copy_from_slice(slice);
        self.offset = end;
        Ok(value)
    }

    fn read_u8(&mut self) -> Result<u8, SolverError> {
        Ok(self.take::<1>()?[0])
    }

    fn read_u16(&mut self) -> Result<u16, SolverError> {
        Ok(u16::from_le_bytes(self.take::<2>()?))
    }

    fn read_u64(&mut self) -> Result<u64, SolverError> {
        Ok(u64::from_le_bytes(self.take::<8>()?))
    }

    fn read_f64(&mut self) -> Result<f64, SolverError> {
        let value = f64::from_le_bytes(self.take::<8>()?);
        if !value.is_finite() {
            return Err(ilut_checkpoint_error(
                "checkpoint contains non-finite value",
            ));
        }
        Ok(value)
    }

    fn read_len(&mut self, element_bytes: usize) -> Result<usize, SolverError> {
        let len = usize::try_from(self.read_u64()?)
            .map_err(|_| ilut_checkpoint_error("checkpoint length does not fit usize"))?;
        if len > self.bytes.len().saturating_sub(self.offset) / element_bytes {
            return Err(ilut_checkpoint_error(
                "checkpoint vector length exceeds payload",
            ));
        }
        Ok(len)
    }

    fn read_usize_vec(&mut self) -> Result<Vec<usize>, SolverError> {
        let len = self.read_len(8)?;
        (0..len)
            .map(|_| {
                usize::try_from(self.read_u64()?)
                    .map_err(|_| ilut_checkpoint_error("checkpoint index does not fit usize"))
            })
            .collect()
    }

    fn read_u32_vec(&mut self) -> Result<Vec<u32>, SolverError> {
        let len = self.read_len(4)?;
        (0..len)
            .map(|_| {
                let bytes = self.take::<4>()?;
                Ok(u32::from_le_bytes(bytes))
            })
            .collect()
    }

    fn read_values<T: ComplexField>(&mut self) -> Result<Vec<T>, SolverError> {
        let len = self.read_len(16)?;
        (0..len)
            .map(|_| {
                let re = T::Real::from_f64(self.read_f64()?)
                    .ok_or_else(|| ilut_checkpoint_error("real factor value cannot decode"))?;
                let im = T::Real::from_f64(self.read_f64()?)
                    .ok_or_else(|| ilut_checkpoint_error("imaginary factor value cannot decode"))?;
                Ok(T::from_re_im(re, im))
            })
            .collect()
    }

    fn is_at_end(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

fn ilut_validate_ptrs(
    name: &str,
    ptrs: &[usize],
    entries: usize,
    n: usize,
) -> Result<(), SolverError> {
    let expected = n
        .checked_add(1)
        .ok_or_else(|| ilut_checkpoint_error(format!("{name} pointer length overflows")))?;
    if ptrs.len() != expected || ptrs.first().copied() != Some(0) {
        return Err(ilut_checkpoint_error(format!(
            "{name} pointers must have length {expected} and start at zero"
        )));
    }
    if ptrs.windows(2).any(|window| window[0] > window[1]) {
        return Err(ilut_checkpoint_error(format!(
            "{name} pointers are not monotone"
        )));
    }
    if ptrs.last().copied() != Some(entries) {
        return Err(ilut_checkpoint_error(format!(
            "{name} pointer terminus does not match entry count"
        )));
    }
    Ok(())
}

fn ilut_validate_permutation(permutation: &[usize], n: usize) -> Result<(), SolverError> {
    if permutation.len() != n || permutation.iter().any(|&index| index >= n) {
        return Err(ilut_checkpoint_error(
            "permutation has invalid length or index",
        ));
    }
    let mut seen = vec![false; n];
    for &index in permutation {
        if std::mem::replace(&mut seen[index], true) {
            return Err(ilut_checkpoint_error(
                "permutation contains duplicate index",
            ));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn ilut_validate_checkpoint_state<T: ComplexField>(
    n: usize,
    l_row_ptrs: &[usize],
    l_col_indices: &[u32],
    l_values: &[T],
    u_row_ptrs: &[usize],
    u_col_indices: &[u32],
    u_values: &[T],
    u_diag_inv: &[T],
    perm: Option<&[usize]>,
    col_perm: Option<&[usize]>,
    regularized_pivots: &[usize],
) -> Result<(), SolverError> {
    ilut_validate_ptrs("L", l_row_ptrs, l_values.len(), n)?;
    ilut_validate_ptrs("U", u_row_ptrs, u_values.len(), n)?;
    if l_col_indices.len() != l_values.len() || u_col_indices.len() != u_values.len() {
        return Err(ilut_checkpoint_error("factor index/value lengths differ"));
    }
    if u_diag_inv.len() != n {
        return Err(ilut_checkpoint_error(
            "diagonal inverse length does not match dimension",
        ));
    }
    for row in 0..n {
        for &column in &l_col_indices[l_row_ptrs[row]..l_row_ptrs[row + 1]] {
            if column as usize >= row {
                return Err(ilut_checkpoint_error("L contains non-strictly-lower entry"));
            }
        }
        for &column in &u_col_indices[u_row_ptrs[row]..u_row_ptrs[row + 1]] {
            if column as usize <= row || column as usize >= n {
                return Err(ilut_checkpoint_error("U contains invalid upper entry"));
            }
        }
    }
    if l_col_indices.iter().any(|&column| column as usize >= n)
        || l_values
            .iter()
            .chain(u_values)
            .chain(u_diag_inv)
            .any(|value| !value.norm().is_finite())
    {
        return Err(ilut_checkpoint_error(
            "factor contains invalid or non-finite value",
        ));
    }
    if let Some(permutation) = perm {
        ilut_validate_permutation(permutation, n)?;
    }
    if let Some(permutation) = col_perm {
        ilut_validate_permutation(permutation, n)?;
    }
    if regularized_pivots.iter().any(|&pivot| pivot >= n)
        || regularized_pivots
            .windows(2)
            .any(|window| window[0] >= window[1])
    {
        return Err(ilut_checkpoint_error("regularized pivot list is invalid"));
    }
    Ok(())
}

/// Keep only the `max_keep` largest-magnitude entries of `entries`.
///
/// Selection (quickselect via `select_nth_unstable_by`) instead of the
/// previous full sort: O(len) expected vs O(len log len). Semantics are
/// identical to the old stable sort-by-magnitude-descending + truncate:
/// magnitude ties are broken by original position, and the survivors keep
/// their original relative order (callers re-sort by column index for CSR
/// storage anyway).
fn keep_largest<T: ComplexField>(entries: &mut Vec<(usize, T)>, max_keep: usize) {
    if entries.len() <= max_keep {
        return;
    }
    if max_keep == 0 {
        entries.clear();
        return;
    }
    // Total order (magnitude descending, original position ascending) makes
    // the selected set match stable-sort + truncate exactly, ties included.
    let mut keyed: Vec<(T::Real, usize)> = entries
        .iter()
        .enumerate()
        .map(|(pos, e)| (e.1.norm(), pos))
        .collect();
    keyed.select_nth_unstable_by(max_keep - 1, |a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    keyed.truncate(max_keep);
    let mut keep: Vec<usize> = keyed.into_iter().map(|(_, pos)| pos).collect();
    keep.sort_unstable();
    *entries = keep.into_iter().map(|pos| entries[pos]).collect();
}

/// Configuration for ILUT preconditioner
#[derive(Debug, Clone)]
pub struct IlutConfig {
    /// Entries with magnitude < drop_tolerance * ||row|| are dropped
    pub drop_tolerance: f64,
    /// Maximum fill per row as a multiple of original nnz per row
    pub fill_factor: usize,
    /// Relative pivot threshold: a pivot `u_kk` is treated as numerically
    /// zero when `|u_kk| < pivot_tolerance * ||A(k,:)||₂` (with an absolute
    /// floor of 1e-30 for matrices scaled near machine epsilon).
    ///
    /// This mirrors the row-norm-relative construction of the ILU(0) kernel
    /// in `ilu.rs` (`1e-2 * ||A(k,:)||₂`); ILUT keeps the historical, much
    /// gentler default factor of 1e-12, but the threshold now *scales with
    /// the matrix*. The previous absolute semantics misfired on matrices
    /// whose entries span large magnitudes: on a row of norm ~1e6 a pivot
    /// of 1e-9 passed the old absolute 1e-12 test and produced multipliers
    /// of order 1e15, while on a row of norm ~1e-6 a fine 1e-13 pivot was
    /// "phase-rescaled" up to 1e-12, silently manufacturing a wrong U.
    pub pivot_tolerance: f64,
    /// ILUTP column pivoting (Saad's ILUTP / ILUPACK style).
    ///
    /// When enabled, each factorization step picks the largest-magnitude
    /// entry of the current row among the not-yet-pivoted columns (generated
    /// fill included) as the pivot, recording a column permutation `P` such
    /// that `L U ≈ A P`. Triangular solves apply the permutation:
    /// `x = P (U⁻¹ L⁻¹ b)`. This is the standard robust-ILU behaviour and
    /// prevents the multiplier blow-up that unpivoted ILUT suffers on
    /// indefinite or structurally small-diagonal matrices (shifted
    /// Laplacians, Helmholtz, saddle-point blocks).
    ///
    /// On diagonally dominant rows the diagonal is already the row maximum,
    /// so no swap happens and the factors are identical to unpivoted ILUT;
    /// the flag only changes behaviour where pivoting is actually needed.
    /// It is therefore on by default; set it to `false` for the historical
    /// unpivoted ILUT.
    pub pivoting: bool,
}

impl IlutConfig {
    /// Create a new ILUT configuration with specified parameters.
    ///
    /// Pivoting ([`IlutConfig::pivoting`]) defaults to on; use
    /// `IlutConfig { pivoting: false, .. }` or struct-update syntax to
    /// disable it.
    pub fn new(drop_tolerance: f64, fill_factor: usize, pivot_tolerance: f64) -> Self {
        Self {
            drop_tolerance,
            fill_factor,
            pivot_tolerance,
            pivoting: true,
        }
    }
}

impl Default for IlutConfig {
    fn default() -> Self {
        Self {
            drop_tolerance: 1e-4,
            fill_factor: 10,
            pivot_tolerance: 1e-12,
            pivoting: true,
        }
    }
}

/// ILUT preconditioner
///
/// Computes an incomplete LU factorization where fill-in is controlled by
/// a drop tolerance and a maximum fill factor per row. This produces a
/// better approximation than ILU(0) at the cost of more memory.
#[derive(Debug, Clone)]
pub struct IlutPreconditioner<T: ComplexField> {
    /// Lower triangular factor (unit diagonal, stored in CSR)
    l_values: Vec<T>,
    l_col_indices: Vec<u32>,
    l_row_ptrs: Vec<usize>,
    /// Upper triangular factor (off-diagonal only, j > i, stored in CSR)
    u_values: Vec<T>,
    u_col_indices: Vec<u32>,
    u_row_ptrs: Vec<usize>,
    /// Pre-computed inverse diagonal of U (avoids inv() per row in apply)
    u_diag_inv: Vec<T>,
    /// Matrix dimension
    n: usize,
    /// Fill-reducing permutation applied before factorization,
    /// `perm[new_index] = old_index`. `None` for natural ordering.
    perm: Option<Vec<usize>>,
    /// ILUTP column permutation, `col_perm[step] = original column label`:
    /// the factors satisfy `L U ≈ A P` with `(A P)[:, s] = A[:, col_perm[s]]`,
    /// so solves finish with the scatter `x[col_perm[s]] = x'[s]`.
    /// `None` when pivoting was disabled or the permutation is the identity.
    col_perm: Option<Vec<usize>>,
    /// Elimination steps whose pivots had to be inserted or scaled because
    /// the requested threshold could not be satisfied. Compatibility
    /// constructors retain this regularization; checked constructors reject
    /// it so callers cannot mistake a fabricated pivot for a valid factor.
    regularized_pivots: Vec<usize>,
    /// Number of non-finite values generated while building the factors.
    /// Checked constructors reject non-zero values; compatibility constructors
    /// retain the factor and expose the diagnostic to their callers.
    non_finite_entries: usize,
}

/// Aggregate setup diagnostics retained by both compatibility and checked
/// ILUT constructors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IlutDiagnostics {
    /// Number of elimination steps that required pivot regularization.
    pub regularized_pivots: usize,
    /// Number of non-finite values generated while building the factors.
    pub non_finite_entries: usize,
    /// Strictly lower-triangular factor entries.
    pub l_nonzeros: usize,
    /// Upper-triangular factor entries, including the implicit diagonal.
    pub u_nonzeros: usize,
    /// Total factor entries (`L + U`, including the U diagonal).
    pub total_nonzeros: usize,
    /// Conservative byte count for numeric values, CSR indices/pointers,
    /// diagonal inverses, permutations, and pivot diagnostics.
    pub estimated_storage_bytes: usize,
}

impl<T: ComplexField> IlutPreconditioner<T> {
    /// Create ILUT preconditioner from a CSR matrix with default configuration
    pub fn from_csr(matrix: &CsrMatrix<T>) -> Self {
        Self::from_csr_with_config(matrix, &IlutConfig::default())
    }

    /// Fallible ILUT construction with the default configuration.
    ///
    /// Unlike [`Self::from_csr`], this rejects malformed input, invalid
    /// configuration, non-finite matrix values, and any pivot that required
    /// regularization during incomplete factorization.
    pub fn try_from_csr(matrix: &CsrMatrix<T>) -> Result<Self, SolverError> {
        Self::try_from_csr_with_config(matrix, &IlutConfig::default())
    }

    /// Create ILUT preconditioner from a CSR matrix with custom configuration
    pub fn from_csr_with_config(matrix: &CsrMatrix<T>, config: &IlutConfig) -> Self {
        Self::factorize_with_recovery(matrix, config, None)
    }

    /// Fallible ILUT construction with explicit setup diagnostics.
    pub fn try_from_csr_with_config(
        matrix: &CsrMatrix<T>,
        config: &IlutConfig,
    ) -> Result<Self, SolverError> {
        Self::validate_setup_input(matrix, config)?;
        Self::checked_factorize_with_recovery(matrix, config, None)
    }

    /// Create ILUT preconditioner with a fill-reducing reordering applied
    /// before factorization (symmetric permutation `P A Pᵀ`).
    ///
    /// With `IluReordering::None` this is identical to
    /// [`Self::from_csr_with_config`].
    pub fn from_csr_with_reordering(
        matrix: &CsrMatrix<T>,
        config: &IlutConfig,
        reordering: IluReordering,
    ) -> Self {
        match reordering {
            IluReordering::None => Self::from_csr_with_config(matrix, config),
            IluReordering::Colamd => {
                let perm = colamd_ordering(matrix);
                let permuted = permute_symmetric(matrix, &perm);
                Self::factorize(&permuted, config, Some(perm))
            }
        }
    }

    /// Fallible ILUT construction with optional fill-reducing reordering.
    pub fn try_from_csr_with_reordering(
        matrix: &CsrMatrix<T>,
        config: &IlutConfig,
        reordering: IluReordering,
    ) -> Result<Self, SolverError> {
        Self::validate_setup_input(matrix, config)?;
        match reordering {
            IluReordering::None => Self::checked_factorize(matrix, config, None),
            IluReordering::Colamd => {
                let perm = colamd_ordering(matrix);
                let permuted = permute_symmetric(matrix, &perm);
                Self::checked_factorize(&permuted, config, Some(perm))
            }
        }
    }

    /// Elimination steps whose pivots required compatibility regularization.
    /// Checked constructors reject any non-empty list; compatibility callers
    /// can inspect it to distinguish a fabricated pivot from a clean factor.
    pub fn regularized_pivots(&self) -> &[usize] {
        &self.regularized_pivots
    }

    /// Return aggregate setup diagnostics for this ILUT factor.
    pub fn diagnostics(&self) -> IlutDiagnostics {
        let l_nonzeros = self.l_values.len();
        let u_nonzeros = self.u_values.len() + self.n;
        IlutDiagnostics {
            regularized_pivots: self.regularized_pivots.len(),
            non_finite_entries: self.non_finite_entries,
            l_nonzeros,
            u_nonzeros,
            total_nonzeros: l_nonzeros + u_nonzeros,
            estimated_storage_bytes: self.estimated_storage_bytes(),
        }
    }

    /// Estimate storage retained by this factorization.
    pub fn estimated_storage_bytes(&self) -> usize {
        use std::mem::size_of;
        (self.l_values.len() + self.u_values.len() + self.u_diag_inv.len()) * size_of::<T>()
            + (self.l_col_indices.len() + self.u_col_indices.len()) * size_of::<u32>()
            + (self.l_row_ptrs.len() + self.u_row_ptrs.len()) * size_of::<usize>()
            + self
                .perm
                .as_ref()
                .map_or(0, |values| values.len() * size_of::<usize>())
            + self
                .col_perm
                .as_ref()
                .map_or(0, |values| values.len() * size_of::<usize>())
            + self.regularized_pivots.len() * size_of::<usize>()
    }

    /// Recompute numeric values while retaining this factor's established
    /// permutations and sparse factor pattern.
    pub fn try_refactor_fixed_pattern(
        &mut self,
        previous_matrix: &CsrMatrix<T>,
        updated_matrix: &CsrMatrix<T>,
        config: &IlutConfig,
    ) -> Result<(), SolverError> {
        Self::validate_setup_input(previous_matrix, config)?;
        Self::validate_setup_input(updated_matrix, config)?;
        if previous_matrix.num_rows != updated_matrix.num_rows
            || previous_matrix.num_cols != updated_matrix.num_cols
            || previous_matrix.row_ptrs != updated_matrix.row_ptrs
            || previous_matrix.col_indices != updated_matrix.col_indices
        {
            return Err(SolverError::OperationFailed {
                operation: "ILUT fixed-pattern numeric refactorization",
                reason: "updated matrix CSR structure does not match the reference matrix".into(),
            });
        }
        if self.n != previous_matrix.num_rows {
            return Err(SolverError::OperationFailed {
                operation: "ILUT fixed-pattern numeric refactorization",
                reason: "reference matrix dimension does not match the existing factor".into(),
            });
        }

        let candidate = match self.perm.as_deref() {
            None => Self::checked_factorize_with_recovery(updated_matrix, config, None)?,
            Some(perm) => {
                let permuted = permute_symmetric(updated_matrix, perm);
                Self::checked_factorize(&permuted, config, Some(perm.to_vec()))?
            }
        };
        if self.l_row_ptrs != candidate.l_row_ptrs
            || self.l_col_indices != candidate.l_col_indices
            || self.u_row_ptrs != candidate.u_row_ptrs
            || self.u_col_indices != candidate.u_col_indices
            || self.perm != candidate.perm
            || self.col_perm != candidate.col_perm
        {
            return Err(SolverError::OperationFailed {
                operation: "ILUT fixed-pattern numeric refactorization",
                reason: "numeric update changed the established factor pattern or permutation"
                    .into(),
            });
        }
        self.l_values.copy_from_slice(&candidate.l_values);
        self.u_values.copy_from_slice(&candidate.u_values);
        self.u_diag_inv.copy_from_slice(&candidate.u_diag_inv);
        self.regularized_pivots = candidate.regularized_pivots;
        self.non_finite_entries = candidate.non_finite_entries;
        Ok(())
    }

    /// Number of non-finite values generated while building the ILUT factors.
    pub fn non_finite_entries(&self) -> usize {
        self.non_finite_entries
    }

    /// Return the dimension of the factorized local system.
    pub(crate) fn dimension(&self) -> usize {
        self.n
    }

    /// Serialize the complete ILUT/ILUTP factor and setup state.
    ///
    /// Numeric values use little-endian `f64` real/imaginary pairs, while
    /// scalar sizes and all structural vectors are checked on restore. The
    /// checkpoint contains the row permutation used for COLAMD and the
    /// column permutation introduced by ILUTP pivoting, so a restored factor
    /// has exactly the same solve semantics as the original.
    pub fn to_checkpoint_bytes(&self) -> Result<Vec<u8>, SolverError> {
        ilut_validate_checkpoint_state(
            self.n,
            &self.l_row_ptrs,
            &self.l_col_indices,
            &self.l_values,
            &self.u_row_ptrs,
            &self.u_col_indices,
            &self.u_values,
            &self.u_diag_inv,
            self.perm.as_deref(),
            self.col_perm.as_deref(),
            &self.regularized_pivots,
        )?;

        let mut output = Vec::new();
        output.extend_from_slice(ILUT_CHECKPOINT_MAGIC);
        output.extend_from_slice(&ILUT_CHECKPOINT_FORMAT_VERSION.to_le_bytes());
        let mut flags = 0_u8;
        if self.perm.is_some() {
            flags |= 1;
        }
        if self.col_perm.is_some() {
            flags |= 2;
        }
        output.push(flags);
        output.push(std::mem::size_of::<T>() as u8);
        output.push(std::mem::size_of::<T::Real>() as u8);
        ilut_checkpoint_push_u64(&mut output, self.n)?;
        ilut_checkpoint_push_usize_vec(&mut output, &self.l_row_ptrs)?;
        ilut_checkpoint_push_u32_vec(&mut output, &self.l_col_indices)?;
        ilut_checkpoint_push_values(&mut output, &self.l_values)?;
        ilut_checkpoint_push_usize_vec(&mut output, &self.u_row_ptrs)?;
        ilut_checkpoint_push_u32_vec(&mut output, &self.u_col_indices)?;
        ilut_checkpoint_push_values(&mut output, &self.u_values)?;
        ilut_checkpoint_push_values(&mut output, &self.u_diag_inv)?;
        if let Some(permutation) = self.perm.as_deref() {
            ilut_checkpoint_push_usize_vec(&mut output, permutation)?;
        }
        if let Some(permutation) = self.col_perm.as_deref() {
            ilut_checkpoint_push_usize_vec(&mut output, permutation)?;
        }
        ilut_checkpoint_push_usize_vec(&mut output, &self.regularized_pivots)?;
        ilut_checkpoint_push_u64(&mut output, self.non_finite_entries)?;
        Ok(output)
    }

    /// Restore an ILUT/ILUTP factor produced by [`Self::to_checkpoint_bytes`].
    pub fn from_checkpoint_bytes(bytes: &[u8]) -> Result<Self, SolverError> {
        let mut reader = IlutCheckpointReader::new(bytes);
        if &reader.take::<8>()? != ILUT_CHECKPOINT_MAGIC {
            return Err(ilut_checkpoint_error("bad magic"));
        }
        if reader.read_u16()? != ILUT_CHECKPOINT_FORMAT_VERSION {
            return Err(ilut_checkpoint_error("unsupported format version"));
        }
        let flags = reader.read_u8()?;
        if flags & !0b11 != 0 {
            return Err(ilut_checkpoint_error("unknown checkpoint flags"));
        }
        let scalar_size = reader.read_u8()?;
        let real_size = reader.read_u8()?;
        if scalar_size as usize != std::mem::size_of::<T>()
            || real_size as usize != std::mem::size_of::<T::Real>()
        {
            return Err(ilut_checkpoint_error(
                "checkpoint scalar family does not match restore type",
            ));
        }
        let n = usize::try_from(reader.read_u64()?)
            .map_err(|_| ilut_checkpoint_error("dimension does not fit usize"))?;
        let l_row_ptrs = reader.read_usize_vec()?;
        let l_col_indices = reader.read_u32_vec()?;
        let l_values = reader.read_values::<T>()?;
        let u_row_ptrs = reader.read_usize_vec()?;
        let u_col_indices = reader.read_u32_vec()?;
        let u_values = reader.read_values::<T>()?;
        let u_diag_inv = reader.read_values::<T>()?;
        let perm = if flags & 1 != 0 {
            Some(reader.read_usize_vec()?)
        } else {
            None
        };
        let col_perm = if flags & 2 != 0 {
            Some(reader.read_usize_vec()?)
        } else {
            None
        };
        let regularized_pivots = reader.read_usize_vec()?;
        let non_finite_entries = usize::try_from(reader.read_u64()?)
            .map_err(|_| ilut_checkpoint_error("non-finite count does not fit usize"))?;
        if !reader.is_at_end() {
            return Err(ilut_checkpoint_error("trailing bytes"));
        }

        ilut_validate_checkpoint_state(
            n,
            &l_row_ptrs,
            &l_col_indices,
            &l_values,
            &u_row_ptrs,
            &u_col_indices,
            &u_values,
            &u_diag_inv,
            perm.as_deref(),
            col_perm.as_deref(),
            &regularized_pivots,
        )?;
        Ok(Self {
            l_values,
            l_col_indices,
            l_row_ptrs,
            u_values,
            u_col_indices,
            u_row_ptrs,
            u_diag_inv,
            n,
            perm,
            col_perm,
            regularized_pivots,
            non_finite_entries,
        })
    }

    fn validate_setup_input(matrix: &CsrMatrix<T>, config: &IlutConfig) -> Result<(), SolverError> {
        matrix.validate_canonical_structure()?;
        if matrix.num_rows != matrix.num_cols {
            return Err(SolverError::OperationFailed {
                operation: "ILUT setup",
                reason: format!(
                    "matrix must be square, got {}x{}",
                    matrix.num_rows, matrix.num_cols
                ),
            });
        }
        if !config.drop_tolerance.is_finite() || config.drop_tolerance < 0.0 {
            return Err(SolverError::OperationFailed {
                operation: "ILUT setup",
                reason: "drop_tolerance must be finite and non-negative".to_string(),
            });
        }
        if config.fill_factor == 0 {
            return Err(SolverError::OperationFailed {
                operation: "ILUT setup",
                reason: "fill_factor must be greater than zero".to_string(),
            });
        }
        let max_row_nnz = matrix
            .row_ptrs
            .windows(2)
            .map(|window| window[1].saturating_sub(window[0]))
            .max()
            .unwrap_or(0);
        if config.fill_factor.checked_mul(max_row_nnz.max(1)).is_none() {
            return Err(SolverError::OperationFailed {
                operation: "ILUT setup",
                reason: format!(
                    "fill_factor {} overflows the per-row fill budget for rows with up to {} entries",
                    config.fill_factor, max_row_nnz
                ),
            });
        }
        if !config.pivot_tolerance.is_finite() || config.pivot_tolerance <= 0.0 {
            return Err(SolverError::OperationFailed {
                operation: "ILUT setup",
                reason: "pivot_tolerance must be finite and positive".to_string(),
            });
        }
        if matrix.values.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation: "ILUT setup",
                reason: "matrix contains NaN or infinity".to_string(),
            });
        }
        CsrMatrix::try_from_raw_parts(
            matrix.num_rows,
            matrix.num_cols,
            matrix.row_ptrs.clone(),
            matrix.col_indices.clone(),
            matrix.values.clone(),
        )
        .map(|_| ())
    }

    fn checked_factorize(
        matrix: &CsrMatrix<T>,
        config: &IlutConfig,
        perm: Option<Vec<usize>>,
    ) -> Result<Self, SolverError> {
        let preconditioner = Self::factorize(matrix, config, perm);
        if preconditioner.non_finite_entries != 0 {
            Err(SolverError::OperationFailed {
                operation: "ILUT setup",
                reason: format!(
                    "{} generated non-finite factor value(s)",
                    preconditioner.non_finite_entries
                ),
            })
        } else if preconditioner.regularized_pivots.is_empty() {
            Ok(preconditioner)
        } else {
            Err(SolverError::OperationFailed {
                operation: "ILUT setup",
                reason: format!(
                    "{} pivot(s) required regularization at elimination steps {:?}",
                    preconditioner.regularized_pivots.len(),
                    preconditioner.regularized_pivots
                ),
            })
        }
    }

    /// ILUT factorization of `matrix`, optionally recording the permutation
    /// that was used to produce it.
    ///
    /// With `config.pivoting` enabled this is ILUTP: step `i` pivots on the
    /// largest-magnitude entry of the current row among the columns not yet
    /// pivoted (fill included), producing `L U ≈ A P`. The work array is
    /// keyed by *original* column label throughout; U rows are stored by
    /// label and remapped to step indices once the permutation is complete
    /// (every column is pivoted exactly once, so the map is total).
    fn factorize(matrix: &CsrMatrix<T>, config: &IlutConfig, perm: Option<Vec<usize>>) -> Self {
        let n = matrix.num_rows;
        let drop_tol = T::Real::from_f64(config.drop_tolerance).unwrap();
        let fill_factor = config.fill_factor;
        let pivoting = config.pivoting;

        // Relative pivot threshold, mirroring the ILU(0) kernel in ilu.rs:
        // tol_k = pivot_tolerance * max(||A(k,:)||₂, 1e-30). The absolute
        // floor keeps the threshold meaningful for matrices whose entries
        // sit near machine epsilon.
        let abs_floor = T::Real::from_f64(1e-30).unwrap();
        let pivot_tol_rel = T::Real::from_f64(config.pivot_tolerance).unwrap();
        let row_norms: Vec<T::Real> = (0..n)
            .map(|i| {
                let mut scale = T::Real::zero();
                let mut sum_squares = T::Real::one();
                for idx in matrix.row_ptrs[i]..matrix.row_ptrs[i + 1] {
                    update_scaled_norm(&mut scale, &mut sum_squares, matrix.values[idx].norm());
                }
                finish_scaled_norm(scale, sum_squares)
            })
            .collect();
        let pivot_tols: Vec<T::Real> = row_norms
            .iter()
            .map(|&rn| pivot_tol_rel * rn.max(abs_floor))
            .collect();

        // L and U stored row-by-row
        let mut l_values: Vec<T> = Vec::new();
        let mut l_col_indices: Vec<u32> = Vec::new();
        let mut l_row_ptrs: Vec<usize> = vec![0];

        let mut u_values: Vec<T> = Vec::new();
        let mut u_col_indices: Vec<u32> = Vec::new();
        let mut u_row_ptrs: Vec<usize> = vec![0];
        let mut u_diag: Vec<T> = vec![T::one(); n];
        let mut regularized_pivots: Vec<usize> = Vec::new();

        // Column pivoting state: col_step[label] is the elimination step at
        // which column `label` was chosen as pivot (usize::MAX while still
        // unpivoted); pcol[step] = label is its inverse. With pivoting
        // disabled step i always pivots on column i and both reduce to the
        // identity, reproducing the historical unpivoted ILUT exactly.
        let mut col_step = vec![usize::MAX; n];
        let mut pcol: Vec<usize> = Vec::with_capacity(n);

        // Work array: w[j] holds the running value for column j in the current row
        // We use a dense workspace indexed by column and a list of active columns.
        let mut w = vec![T::zero(); n];
        let mut active = Vec::with_capacity(n);
        let mut active_flag = vec![false; n];

        for i in 0..n {
            // ----------------------------------------------------------
            // Step 1: Copy row i of A into the work array w
            // ----------------------------------------------------------
            active.clear();
            for idx in matrix.row_ptrs[i]..matrix.row_ptrs[i + 1] {
                let j = matrix.col_indices[idx];
                w[j] = matrix.values[idx];
                if !active_flag[j] {
                    active_flag[j] = true;
                    active.push(j);
                }
            }

            // Compute the row norm for dropping: ||a_i||_2
            let mut row_norm_scale = T::Real::zero();
            let mut row_norm_sum_squares = T::Real::one();
            for &j in &active {
                update_scaled_norm(&mut row_norm_scale, &mut row_norm_sum_squares, w[j].norm());
            }
            let row_norm = finish_scaled_norm(row_norm_scale, row_norm_sum_squares);

            // Number of non-zeros in the original row
            let nnz_row = matrix.row_ptrs[i + 1] - matrix.row_ptrs[i];
            let max_fill = fill_factor * nnz_row.max(1);

            // ----------------------------------------------------------
            // Step 2: Elimination - iterate over pivoted columns in
            // increasing step order.
            // ----------------------------------------------------------
            // Keep an ordered worklist because eliminating step k can create
            // a new active entry in a column pivoted at step k' with
            // k < k' < i. That generated lower entry must itself be
            // eliminated before the completed row is stored in L.
            //
            // Invariant: U row k only references columns that were unpivoted
            // at step k, so updates from step k only touch columns with
            // step > k (or unpivoted) and the worklist stays well-ordered.
            let mut lower_work: BTreeSet<usize> = active
                .iter()
                .filter_map(|&j| (col_step[j] != usize::MAX).then_some(col_step[j]))
                .collect();

            while let Some(k) = lower_work.pop_first() {
                let label = pcol[k];
                // w[label] = w[label] / u_kk
                let u_kk = u_diag[k];
                if u_kk.norm() <= pivot_tols[k] {
                    // Skip near-zero pivots (relative to the row the pivot
                    // came from — see pivot_tols above). `<=`, not `<`: a
                    // pivot regularized to the floor must still be treated
                    // as numerically zero here, otherwise the floor value
                    // itself breeds multipliers of magnitude ~1/tol.
                    w[label] = T::zero();
                    active_flag[label] = false;
                    continue;
                }
                w[label] *= u_kk.inv();

                // Drop test: if |w[label]| < drop_tol * ||row_i||, set to 0
                if w[label].norm() < drop_tol * row_norm {
                    w[label] = T::zero();
                    active_flag[label] = false;
                    continue;
                }

                // Update: for each entry in U row k (stored keyed by
                // original column label; all such labels were unpivoted at
                // step k)
                let w_k = w[label];
                let u_row_start = u_row_ptrs[k];
                let u_row_end = u_row_ptrs[k + 1];
                for u_idx in u_row_start..u_row_end {
                    let j = u_col_indices[u_idx] as usize;
                    let u_kj = u_values[u_idx];
                    w[j] -= w_k * u_kj;
                    if !active_flag[j] {
                        active_flag[j] = true;
                        active.push(j);
                    }
                    if col_step[j] != usize::MAX {
                        lower_work.insert(col_step[j]);
                    }
                }
            }

            // ----------------------------------------------------------
            // Step 3: Pivot selection.
            // ----------------------------------------------------------
            // ILUTP: pivot on the largest-magnitude active entry among the
            // not-yet-pivoted columns (generated fill included) — column
            // partial pivoting. Unpivoted ILUT: always column i.
            let pivot_label = if pivoting {
                let mut best: Option<(usize, T::Real)> = None;
                for &j in &active {
                    if !active_flag[j] || col_step[j] != usize::MAX {
                        continue;
                    }
                    let mag = w[j].norm();
                    if best.is_none_or(|(_, best_mag)| mag > best_mag) {
                        best = Some((j, mag));
                    }
                }
                match best {
                    Some((j, _)) => j,
                    // No active candidate (empty or fully dropped row):
                    // fall back to any unpivoted column, preferring i, so
                    // that pcol stays a genuine permutation. The pivot value
                    // is inserted by the regularization below.
                    None => {
                        if col_step[i] == usize::MAX {
                            i
                        } else {
                            (0..n)
                                .find(|&j| col_step[j] == usize::MAX)
                                .expect("step i < n always leaves an unpivoted column")
                        }
                    }
                }
            } else {
                i
            };
            col_step[pivot_label] = i;
            pcol.push(pivot_label);

            // ----------------------------------------------------------
            // Step 4: Apply dropping rules and store L and U entries
            // ----------------------------------------------------------
            // Separate into L part (columns pivoted at steps < i; the work
            // array holds their multipliers) and U part (unpivoted columns,
            // keyed by label for now).
            let mut l_entries: Vec<(usize, T)> = Vec::new();
            let mut u_entries: Vec<(usize, T)> = Vec::new();

            for &j in &active {
                if !active_flag[j] {
                    // Already dropped
                    continue;
                }
                let val = w[j];
                if j == pivot_label {
                    // Never drop the pivot
                    u_entries.push((j, val));
                } else if val.norm() < drop_tol * row_norm {
                    // Drop small entries
                } else if col_step[j] != usize::MAX {
                    // Multiplier for step col_step[j] (< i by the invariant)
                    l_entries.push((col_step[j], val));
                } else {
                    u_entries.push((j, val));
                }
            }

            // Keep at most max_fill largest entries in L (selection, not a
            // full sort — see keep_largest)
            if l_entries.len() > max_fill {
                keep_largest(&mut l_entries, max_fill);
            }
            // Sort L entries by step index for CSR storage
            l_entries.sort_by_key(|&(k, _)| k);

            // Keep at most max_fill largest entries in U (always keep pivot)
            if u_entries.len() > max_fill {
                // Separate pivot from off-pivot entries
                let pivot_entry = u_entries.iter().find(|&&(j, _)| j == pivot_label).copied();
                let mut off_pivot: Vec<(usize, T)> = u_entries
                    .iter()
                    .filter(|&&(j, _)| j != pivot_label)
                    .copied()
                    .collect();
                keep_largest(&mut off_pivot, max_fill.saturating_sub(1));
                u_entries.clear();
                if let Some(d) = pivot_entry {
                    u_entries.push(d);
                }
                u_entries.extend(off_pivot);
            }
            // Sort U entries by column label (the pivot-insert below uses a
            // binary-search insertion point; the pivoted path is re-sorted
            // by step index during the post-factorization remap anyway)
            u_entries.sort_by_key(|&(j, _)| j);

            // Ensure we have a pivot entry in U
            let has_pivot = u_entries.iter().any(|&(j, _)| j == pivot_label);
            if !has_pivot {
                regularized_pivots.push(i);
                // Insert a small pivot to avoid singular U (scaled by the
                // row norm so the regularization matches the matrix scale)
                let pivot = T::from_real(pivot_tols[i]);
                // Find insertion position to maintain sorted order
                let pos = u_entries.partition_point(|&(j, _)| j < pivot_label);
                u_entries.insert(pos, (pivot_label, pivot));
            }

            // Enforce minimum pivot (preserve complex phase), relative to
            // the row norm — a small-but-legitimate pivot is left alone.
            for entry in &mut u_entries {
                if entry.0 == pivot_label && entry.1.norm() < pivot_tols[i] {
                    regularized_pivots.push(i);
                    let norm = entry.1.norm();
                    if norm > abs_floor {
                        entry.1 *= T::from_real(pivot_tols[i] / norm);
                    } else {
                        // Original value is effectively zero; fall back to real
                        entry.1 = T::from_real(pivot_tols[i]);
                    }
                }
            }

            // Store diagonal of U for fast access
            for &(j, val) in &u_entries {
                if j == pivot_label {
                    u_diag[i] = val;
                    break;
                }
            }

            // Store L row (keyed by step index)
            for &(k, val) in &l_entries {
                l_values.push(val);
                l_col_indices.push(k as u32);
            }
            l_row_ptrs.push(l_values.len());

            // Store U row (off-pivot only, keyed by original column label
            // for now; diagonal stored in u_diag). Remapped to step indices
            // after the loop, once the permutation is complete.
            for &(j, val) in &u_entries {
                if j != pivot_label {
                    u_values.push(val);
                    u_col_indices.push(j as u32);
                }
            }
            u_row_ptrs.push(u_values.len());

            // ----------------------------------------------------------
            // Reset workspace for next row
            // ----------------------------------------------------------
            for &j in &active {
                w[j] = T::zero();
                active_flag[j] = false;
            }
        }

        // With pivoting, U column labels refer to original columns; remap
        // them to step indices (step > row, as required by the backward
        // substitution) now that col_step is a complete permutation.
        if pivoting {
            let mut new_u_values: Vec<T> = Vec::with_capacity(u_values.len());
            let mut new_u_col_indices: Vec<u32> = Vec::with_capacity(u_values.len());
            let mut new_u_row_ptrs: Vec<usize> = vec![0];
            let mut row_entries: Vec<(usize, T)> = Vec::new();
            for i in 0..n {
                row_entries.clear();
                for idx in u_row_ptrs[i]..u_row_ptrs[i + 1] {
                    row_entries.push((col_step[u_col_indices[idx] as usize], u_values[idx]));
                }
                row_entries.sort_by_key(|&(j, _)| j);
                for &(j, v) in &row_entries {
                    new_u_col_indices.push(j as u32);
                    new_u_values.push(v);
                }
                new_u_row_ptrs.push(new_u_values.len());
            }
            u_values = new_u_values;
            u_col_indices = new_u_col_indices;
            u_row_ptrs = new_u_row_ptrs;
        }

        let u_diag_inv: Vec<T> = u_diag
            .iter()
            .map(|&d| {
                if d.norm() > T::Real::from_f64(1e-20).unwrap() {
                    d.inv()
                } else {
                    T::one()
                }
            })
            .collect();

        regularized_pivots.sort_unstable();
        regularized_pivots.dedup();
        let non_finite_entries = l_values
            .iter()
            .chain(&u_values)
            .chain(&u_diag)
            .chain(&u_diag_inv)
            .filter(|value| !value.norm().is_finite())
            .count();
        let col_perm = if pivoting && pcol.iter().enumerate().any(|(s, &l)| l != s) {
            Some(pcol)
        } else {
            None
        };

        Self {
            l_values,
            l_col_indices,
            l_row_ptrs,
            u_values,
            u_col_indices,
            u_row_ptrs,
            u_diag_inv,
            n,
            perm,
            col_perm,
            regularized_pivots,
            non_finite_entries,
        }
    }

    /// Retry small pivoted systems without dropping entries when an ILUTP
    /// setup had to manufacture pivots.  A dropped near-null pivot can make a
    /// compatibility preconditioner numerically useless even though the
    /// matrix itself is perfectly solvable.  The bounded retry keeps the
    /// safety net out of large production factorizations; checked callers still
    /// receive the regularization diagnostic when the retry cannot help.
    fn factorize_with_recovery(
        matrix: &CsrMatrix<T>,
        config: &IlutConfig,
        perm: Option<Vec<usize>>,
    ) -> Self {
        let primary = Self::factorize(matrix, config, perm.clone());
        if !config.pivoting || primary.regularized_pivots.is_empty() || matrix.num_rows > 512 {
            return primary;
        }

        let retry_config = IlutConfig {
            drop_tolerance: 0.0,
            fill_factor: config.fill_factor.max(matrix.num_rows),
            ..config.clone()
        };
        let retry = Self::factorize(matrix, &retry_config, perm);
        if retry.regularized_pivots.len() < primary.regularized_pivots.len() {
            retry
        } else {
            primary
        }
    }

    fn checked_factorize_with_recovery(
        matrix: &CsrMatrix<T>,
        config: &IlutConfig,
        perm: Option<Vec<usize>>,
    ) -> Result<Self, SolverError> {
        let preconditioner = Self::factorize_with_recovery(matrix, config, perm);
        if preconditioner.non_finite_entries > 0 {
            Err(SolverError::OperationFailed {
                operation: "ILUT setup",
                reason: format!(
                    "{} generated non-finite factor value(s)",
                    preconditioner.non_finite_entries
                ),
            })
        } else if preconditioner.regularized_pivots.is_empty() {
            Ok(preconditioner)
        } else {
            Err(SolverError::OperationFailed {
                operation: "ILUT setup",
                reason: format!(
                    "{} pivot(s) required regularization at elimination steps {:?}",
                    preconditioner.regularized_pivots.len(),
                    preconditioner.regularized_pivots
                ),
            })
        }
    }
}

impl<T: ComplexField> Preconditioner<T> for IlutPreconditioner<T> {
    fn apply(&self, r: &[T]) -> Vec<T> {
        let mut x = r.to_vec();
        self.apply_into(r, &mut x);
        x
    }

    fn apply_into(&self, r: &[T], y: &mut [T]) {
        match &self.perm {
            None => {
                y.copy_from_slice(r);
                self.solve_in_place(y);
            }
            Some(perm) => {
                // Gather into permuted ordering, solve, scatter back.
                let mut s: Vec<T> = perm.iter().map(|&old| r[old]).collect();
                self.solve_in_place(&mut s);
                for (new_idx, &old_idx) in perm.iter().enumerate() {
                    y[old_idx] = s[new_idx];
                }
            }
        }
    }

    fn apply_transpose(&self, r: &[T]) -> Vec<T> {
        let mut x = r.to_vec();
        self.apply_transpose_into(r, &mut x);
        x
    }

    fn transpose_capability(&self) -> sonium_solvers_core::traits::AdjointCapability {
        sonium_solvers_core::traits::AdjointCapability::Exact
    }

    fn apply_transpose_into(&self, r: &[T], y: &mut [T]) {
        self.apply_adjoint_into(r, y, false);
    }

    fn apply_hermitian(&self, r: &[T]) -> Vec<T> {
        let mut x = r.to_vec();
        self.apply_hermitian_into(r, &mut x);
        x
    }

    fn hermitian_capability(&self) -> sonium_solvers_core::traits::AdjointCapability {
        sonium_solvers_core::traits::AdjointCapability::Exact
    }

    fn apply_hermitian_into(&self, r: &[T], y: &mut [T]) {
        self.apply_adjoint_into(r, y, true);
    }
}

impl<T: ComplexField> IlutPreconditioner<T> {
    #[inline]
    pub(crate) fn solve_in_place(&self, s: &mut [T]) {
        assert_eq!(s.len(), self.n, "ILUT solve buffer length mismatch");
        let l_vals = self.l_values.as_slice();
        let l_cols = self.l_col_indices.as_slice();
        let l_ptrs = self.l_row_ptrs.as_slice();
        let u_vals = self.u_values.as_slice();
        let u_cols = self.u_col_indices.as_slice();
        let u_ptrs = self.u_row_ptrs.as_slice();

        // Forward substitution: Ly = r (L has unit diagonal)
        for i in 0..self.n {
            let mut value = unsafe { *s.get_unchecked(i) };
            for idx in l_ptrs[i]..l_ptrs[i + 1] {
                unsafe {
                    let j = *l_cols.get_unchecked(idx) as usize;
                    value -= *l_vals.get_unchecked(idx) * *s.get_unchecked(j);
                }
            }
            unsafe {
                *s.get_unchecked_mut(i) = value;
            }
        }

        // Backward substitution: Ux = y
        // U stores only off-diagonal entries (j > i); diagonal handled via u_diag.
        for i in (0..self.n).rev() {
            let mut value = unsafe { *s.get_unchecked(i) };
            for idx in u_ptrs[i]..u_ptrs[i + 1] {
                unsafe {
                    let j = *u_cols.get_unchecked(idx) as usize;
                    value -= *u_vals.get_unchecked(idx) * *s.get_unchecked(j);
                }
            }
            unsafe {
                *s.get_unchecked_mut(i) = value * *self.u_diag_inv.get_unchecked(i);
            }
        }

        // ILUTP: the factors satisfy L U ≈ A P, so the solution in step
        // numbering must be scattered back: x[col_perm[step]] = x'[step].
        if let Some(pcol) = &self.col_perm {
            let solved = s.to_vec();
            for (step, &label) in pcol.iter().enumerate() {
                s[label] = solved[step];
            }
        }
    }

    /// Solve the transpose or Hermitian triangular factors in place.
    ///
    /// The forward factor is `L U`; its transpose/adjoint solve is therefore
    /// `Uᵀ Lᵀ` / `Uᴴ Lᴴ`. The factors are stored by rows, so each triangular
    /// solve uses scatter updates and preserves the same O(nnz) work as the
    /// forward substitution. This is also used by block-Jacobi's exact
    /// block-diagonal adjoint path.
    pub(crate) fn solve_adjoint_in_place(&self, s: &mut [T], hermitian: bool) {
        assert_eq!(s.len(), self.n, "ILUT adjoint solve buffer length mismatch");

        // Solve Uᵀ y = r or Uᴴ y = r. U is stored by rows with off-diagonal
        // entries j > i. Once y_i is known, its row contributes to all later
        // right-hand-side entries of the transposed lower-triangular solve.
        for i in 0..self.n {
            s[i] = if hermitian {
                s[i] * self.u_diag_inv[i].conj()
            } else {
                s[i] * self.u_diag_inv[i]
            };
            for index in self.u_row_ptrs[i]..self.u_row_ptrs[i + 1] {
                let j = self.u_col_indices[index] as usize;
                let value = if hermitian {
                    self.u_values[index].conj()
                } else {
                    self.u_values[index]
                };
                s[j] -= value * s[i];
            }
        }

        // Solve Lᵀ y = r or Lᴴ y = r. L has unit diagonal and contains only
        // entries j < i, so descending row scatter updates implement the
        // upper-triangular solve directly.
        for i in (0..self.n).rev() {
            for index in self.l_row_ptrs[i]..self.l_row_ptrs[i + 1] {
                let j = self.l_col_indices[index] as usize;
                let value = if hermitian {
                    self.l_values[index].conj()
                } else {
                    self.l_values[index]
                };
                s[j] -= value * s[i];
            }
        }
    }

    fn apply_adjoint_into(&self, r: &[T], y: &mut [T], hermitian: bool) {
        assert_eq!(r.len(), self.n, "ILUT adjoint RHS length mismatch");
        assert_eq!(y.len(), self.n, "ILUT adjoint output length mismatch");

        // Forward ILUTP scatters factor-order values with col_perm. Its
        // transpose/adjoint therefore gathers with the same labels before
        // solving the triangular factors.
        let mut s = if let Some(col_perm) = &self.col_perm {
            col_perm.iter().map(|&label| r[label]).collect()
        } else {
            r.to_vec()
        };
        self.solve_adjoint_in_place(&mut s, hermitian);

        // The row permutation used during factorization is real. Its
        // transpose is the same scatter used by the forward path.
        if let Some(perm) = &self.perm {
            for (new_index, &old_index) in perm.iter().enumerate() {
                y[old_index] = s[new_index];
            }
        } else {
            y.copy_from_slice(&s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use sonium_solvers_iterative::{GmresConfig, gmres, gmres_preconditioned};

    /// Helper: create a Complex64 value
    fn c(re: f64) -> Complex64 {
        Complex64::new(re, 0.0)
    }

    #[test]
    fn checked_ilut_accepts_a_well_pivoted_matrix() {
        let dense = sonium_solvers_core::array![[c(4.0), c(1.0)], [c(1.0), c(3.0)]];
        let matrix = CsrMatrix::from_dense(&dense, 1e-15);

        assert!(IlutPreconditioner::try_from_csr(&matrix).is_ok());
    }

    #[test]
    fn checked_ilut_norms_do_not_overflow_on_large_finite_diagonal() {
        let matrix =
            CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![f64::MAX, f64::MAX]);

        let preconditioner = IlutPreconditioner::try_from_csr(&matrix)
            .expect("large finite diagonal must remain representable");
        assert!(preconditioner.regularized_pivots().is_empty());
        assert_eq!(preconditioner.non_finite_entries(), 0);
    }

    #[test]
    fn ilut_exposes_regularized_pivot_diagnostics() {
        let dense = sonium_solvers_core::array![[c(4.0), c(1.0)], [c(1.0), c(3.0)]];
        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let preconditioner = IlutPreconditioner::from_csr(&matrix);

        assert!(preconditioner.regularized_pivots().is_empty());
        assert_eq!(
            preconditioner.diagnostics(),
            IlutDiagnostics {
                regularized_pivots: 0,
                non_finite_entries: 0,
                l_nonzeros: 1,
                u_nonzeros: 3,
                total_nonzeros: 4,
                estimated_storage_bytes: preconditioner.estimated_storage_bytes(),
            }
        );
    }

    #[test]
    fn fixed_pattern_numeric_refactorization_updates_values_and_preserves_storage() {
        let first = CsrMatrix::from_dense(
            &sonium_solvers_core::array![[c(4.0), c(1.0)], [c(1.0), c(3.0)]],
            1e-15,
        );
        let second = CsrMatrix::from_dense(
            &sonium_solvers_core::array![[c(5.0), c(1.0)], [c(1.0), c(4.0)]],
            1e-15,
        );
        let config = IlutConfig::default();
        let mut factor = IlutPreconditioner::try_from_csr_with_config(&first, &config).unwrap();
        let before = factor.diagnostics();
        factor
            .try_refactor_fixed_pattern(&first, &second, &config)
            .unwrap();
        let after = factor.diagnostics();
        assert_eq!(before.total_nonzeros, after.total_nonzeros);
        assert_eq!(
            before.estimated_storage_bytes,
            after.estimated_storage_bytes
        );
        let applied = factor.apply(&[c(6.0), c(5.0)]);
        assert!((applied[0] - c(1.0)).norm() < 1.0e-12);
        assert!((applied[1] - c(1.0)).norm() < 1.0e-12);
    }

    #[test]
    fn fixed_pattern_numeric_refactorization_rejects_structure_changes() {
        let first = CsrMatrix::from_dense(
            &sonium_solvers_core::array![[c(4.0), c(1.0)], [c(1.0), c(3.0)]],
            1e-15,
        );
        let changed = CsrMatrix::from_dense(
            &sonium_solvers_core::array![[c(4.0), c(0.0)], [c(1.0), c(3.0)]],
            1e-15,
        );
        let config = IlutConfig::default();
        let mut factor = IlutPreconditioner::try_from_csr_with_config(&first, &config).unwrap();
        let error = factor
            .try_refactor_fixed_pattern(&first, &changed, &config)
            .unwrap_err();
        assert!(error.to_string().contains("CSR structure"));
    }

    #[test]
    fn checked_ilut_rejects_singular_pivot_regularization() {
        let dense = sonium_solvers_core::array![[c(1.0), c(0.0)], [c(0.0), c(0.0)]];
        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let error = IlutPreconditioner::try_from_csr(&matrix)
            .expect_err("singular ILUT setup must report a pivot diagnostic");

        assert!(matches!(
            error,
            sonium_solvers_core::traits::SolverError::OperationFailed { .. }
        ));
        assert!(error.to_string().contains("regularization"));
    }

    #[test]
    fn checked_ilut_rejects_generated_non_finite_factors() {
        let matrix = CsrMatrix::from_raw_parts(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1e-100_f64, 1e154, 1e154, 1e100],
        );
        let config = IlutConfig {
            drop_tolerance: 0.0,
            pivot_tolerance: 1e-300,
            pivoting: false,
            ..IlutConfig::default()
        };

        let compatibility = IlutPreconditioner::from_csr_with_config(&matrix, &config);
        assert!(compatibility.non_finite_entries() > 0);

        let error = IlutPreconditioner::try_from_csr_with_config(&matrix, &config)
            .expect_err("checked ILUT must reject generated non-finite factors");
        assert!(matches!(
            error,
            sonium_solvers_core::traits::SolverError::OperationFailed { .. }
        ));
        assert!(error.to_string().contains("generated non-finite factor"));
    }

    #[test]
    fn checked_ilut_rejects_invalid_setup_input() {
        let mut matrix = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![1.0]);
        matrix.values[0] = f64::NAN;
        let error = IlutPreconditioner::try_from_csr(&matrix)
            .expect_err("non-finite matrix values must be rejected");
        assert!(error.to_string().contains("NaN or infinity"));

        let dense = sonium_solvers_core::array![[c(1.0)]];
        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let config = IlutConfig {
            fill_factor: 0,
            ..IlutConfig::default()
        };
        let error = IlutPreconditioner::try_from_csr_with_config(&matrix, &config)
            .expect_err("zero fill factor must be rejected");
        assert!(error.to_string().contains("fill_factor"));
    }

    #[test]
    fn checked_ilut_rejects_fill_budget_overflow_before_factorization() {
        let matrix = CsrMatrix::from_raw_parts(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![4.0, 1.0, 1.0, 3.0],
        );
        let config = IlutConfig {
            fill_factor: usize::MAX,
            ..IlutConfig::default()
        };

        let error = IlutPreconditioner::try_from_csr_with_config(&matrix, &config)
            .expect_err("fill-budget overflow must be rejected before factorization");
        assert!(matches!(
            error,
            sonium_solvers_core::traits::SolverError::OperationFailed { .. }
        ));
        assert!(
            error
                .to_string()
                .contains("overflows the per-row fill budget")
        );
    }

    #[test]
    fn test_ilut_diagonal_matrix() {
        // A diagonal matrix should factor exactly: L = I, U = D
        let n = 5;
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        for i in 0..n {
            dense[[i, i]] = Complex64::new((i + 1) as f64 * 2.0, 0.0);
        }

        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let precond = IlutPreconditioner::from_csr(&matrix);

        // Applying preconditioner to b should give D^{-1} * b exactly
        let b = (0..n)
            .map(|i| Complex64::new((i + 1) as f64, 0.0))
            .collect::<Vec<_>>();
        let result = precond.apply(&b);

        for i in 0..n {
            let expected = b[i] * dense[[i, i]].inv();
            assert!(
                (result[i] - expected).norm() < 1e-12,
                "Diagonal ILUT failed at row {}: got {:?}, expected {:?}",
                i,
                result[i],
                expected,
            );
        }
    }

    #[test]
    fn test_ilut_tridiagonal_spd() {
        // Tridiagonal SPD matrix
        let dense = sonium_solvers_core::array![
            [c(4.0), c(-1.0), c(0.0)],
            [c(-1.0), c(4.0), c(-1.0)],
            [c(0.0), c(-1.0), c(4.0)],
        ];

        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let config = IlutConfig {
            drop_tolerance: 1e-10,
            fill_factor: 10,
            pivot_tolerance: 1e-12,
            pivoting: false,
        };
        let precond = IlutPreconditioner::from_csr_with_config(&matrix, &config);

        let r = sonium_solvers_core::array![c(1.0), c(2.0), c(3.0)];
        let result = precond.apply(&r);

        // Verify that A * result ~ r (since with low drop tolerance, ILUT ~ exact LU)
        let check = matrix.matvec(&result);
        for i in 0..3 {
            assert!(
                (check[i] - r[i]).norm() < 1e-10,
                "ILUT tridiagonal SPD failed at row {}: got {:?}, expected {:?}",
                i,
                check[i],
                r[i],
            );
        }
    }

    #[test]
    fn test_ilut_exact_lu_small_system() {
        // For a small system with generous fill, ILUT should match exact LU
        let dense = sonium_solvers_core::array![
            [c(10.0), c(-1.0), c(2.0), c(0.0)],
            [c(-1.0), c(11.0), c(-1.0), c(3.0)],
            [c(2.0), c(-1.0), c(10.0), c(-1.0)],
            [c(0.0), c(3.0), c(-1.0), c(8.0)],
        ];

        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let config = IlutConfig {
            drop_tolerance: 0.0, // keep everything
            fill_factor: 100,    // allow lots of fill
            pivot_tolerance: 1e-15,
            pivoting: false,
        };
        let precond = IlutPreconditioner::from_csr_with_config(&matrix, &config);

        // Apply to several right-hand sides and verify A * M^{-1} r ~ r
        for k in 0..3 {
            let b = (0..4)
                .map(|i| Complex64::new(((i + k) as f64).sin(), 0.0))
                .collect::<Vec<_>>();
            let x = precond.apply(&b);
            let residual = sonium_solvers_core::vec_ops::vec_sub(&matrix.matvec(&x), &b);
            let res_norm: f64 = residual.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
            assert!(
                res_norm < 1e-10,
                "ILUT exact LU test failed for RHS {}: residual norm = {}",
                k,
                res_norm,
            );
        }
    }

    #[test]
    fn test_ilut_exact_fill_processes_generated_lower_entry() {
        // Eliminating column 0 in row 2 generates a non-zero in column 1,
        // which is still below the diagonal. Exact-fill ILUT must process
        // that generated entry before storing the row in L.
        let dense = sonium_solvers_core::array![
            [c(4.0), c(1.0), c(1.0)],
            [c(1.0), c(4.0), c(0.0)],
            [c(1.0), c(0.0), c(4.0)],
        ];
        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let config = IlutConfig {
            drop_tolerance: 0.0,
            fill_factor: 100,
            pivot_tolerance: 1e-15,
            pivoting: false,
        };
        let precond = IlutPreconditioner::from_csr_with_config(&matrix, &config);

        for b in [
            sonium_solvers_core::array![c(1.0), c(0.0), c(0.0)],
            sonium_solvers_core::array![c(0.0), c(1.0), c(0.0)],
            sonium_solvers_core::array![c(0.0), c(0.0), c(1.0)],
        ] {
            let x = precond.apply(&b);
            let residual = sonium_solvers_core::vec_ops::vec_sub(&matrix.matvec(&x), &b);
            let residual_norm = residual.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
            assert!(
                residual_norm < 1e-12,
                "exact-fill ILUT missed generated lower fill: residual={residual_norm:.3e}",
            );
        }
    }

    #[test]
    fn test_ilut_exact_fill_complex_irregular_graph() {
        let z = |re, im| Complex64::new(re, im);
        let dense = sonium_solvers_core::array![
            [
                z(8.0, 0.5),
                z(1.0, 0.1),
                z(1.5, -0.2),
                z(0.0, 0.0),
                z(0.0, 0.0)
            ],
            [
                z(1.0, 0.1),
                z(7.0, 0.4),
                z(0.0, 0.0),
                z(1.2, 0.2),
                z(0.0, 0.0)
            ],
            [
                z(1.5, -0.2),
                z(0.0, 0.0),
                z(9.0, 0.3),
                z(0.8, 0.1),
                z(1.1, -0.1)
            ],
            [
                z(0.0, 0.0),
                z(1.2, 0.2),
                z(0.8, 0.1),
                z(8.0, 0.6),
                z(0.7, 0.0)
            ],
            [
                z(0.0, 0.0),
                z(0.0, 0.0),
                z(1.1, -0.1),
                z(0.7, 0.0),
                z(7.5, 0.2)
            ],
        ];
        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let precond = IlutPreconditioner::from_csr_with_config(
            &matrix,
            &IlutConfig {
                drop_tolerance: 0.0,
                fill_factor: 100,
                pivot_tolerance: 1e-15,
                pivoting: false,
            },
        );
        let b = sonium_solvers_core::array![
            z(1.0, -0.5),
            z(-0.2, 0.7),
            z(0.3, 0.1),
            z(1.2, -0.4),
            z(-0.8, 0.2)
        ];
        let x = precond.apply(&b);
        let residual = sonium_solvers_core::vec_ops::vec_sub(&matrix.matvec(&x), &b);
        let residual_norm = residual.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        assert!(
            residual_norm < 1e-11,
            "complex irregular exact-fill ILUT residual={residual_norm:.3e}",
        );
    }

    #[test]
    fn test_ilut_preconditioned_residual_reduced() {
        // Verify that applying the ILUT preconditioner reduces the residual
        // compared to the unpreconditioned residual
        let n = 20;
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));

        // Build a diagonally dominant matrix with some off-diagonal fill
        for i in 0..n {
            dense[[i, i]] = Complex64::new(10.0, 0.0);
            if i > 0 {
                dense[[i, i - 1]] = Complex64::new(-2.0, 0.0);
            }
            if i < n - 1 {
                dense[[i, i + 1]] = Complex64::new(-1.0, 0.0);
            }
            if i > 1 {
                dense[[i, i - 2]] = Complex64::new(0.5, 0.0);
            }
            if i < n - 2 {
                dense[[i, i + 2]] = Complex64::new(0.3, 0.0);
            }
        }

        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let precond = IlutPreconditioner::from_csr(&matrix);

        // True solution
        let x_true = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.3).sin(), 0.0))
            .collect::<Vec<_>>();
        let b = matrix.matvec(&x_true);

        // Unpreconditioned residual for initial guess x0 = 0
        let r0 = b.clone();
        let r0_norm: f64 = r0.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

        // After one preconditioner application: x1 = M^{-1} r0
        let x1 = precond.apply(&r0);
        let r1 = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x1));
        let r1_norm: f64 = r1.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

        assert!(
            r1_norm < r0_norm,
            "ILUT should reduce residual: ||r1|| = {}, ||r0|| = {}",
            r1_norm,
            r0_norm,
        );
    }

    #[test]
    fn test_ilut_with_gmres() {
        // Larger system where preconditioning helps
        let n = 10;
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));

        // Tridiagonal matrix
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
        let precond = IlutPreconditioner::from_csr(&matrix);

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

        // Solve without preconditioning
        let sol_no_precond = gmres(&matrix, &b, &config);

        // Solve with preconditioning
        let sol_precond = gmres_preconditioned(&matrix, &precond, &b, &config);

        // Both should converge
        assert!(sol_no_precond.converged);
        assert!(sol_precond.converged);

        // Preconditioned should need fewer iterations (usually)
        assert!(
            sol_precond.iterations <= sol_no_precond.iterations + 5,
            "Preconditioning should not hurt: {} vs {}",
            sol_precond.iterations,
            sol_no_precond.iterations,
        );
    }

    #[test]
    fn test_ilut_drop_tolerance_effect() {
        // Verify that higher drop tolerance produces fewer non-zeros
        let n = 10;
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));

        for i in 0..n {
            dense[[i, i]] = Complex64::new(10.0, 0.0);
            if i > 0 {
                dense[[i, i - 1]] = Complex64::new(-1.0, 0.0);
            }
            if i < n - 1 {
                dense[[i, i + 1]] = Complex64::new(-1.0, 0.0);
            }
        }

        let matrix = CsrMatrix::from_dense(&dense, 1e-15);

        let config_tight = IlutConfig {
            drop_tolerance: 1e-12,
            fill_factor: 100,
            pivot_tolerance: 1e-15,
            pivoting: false,
        };
        let precond_tight = IlutPreconditioner::from_csr_with_config(&matrix, &config_tight);

        let config_loose = IlutConfig {
            drop_tolerance: 0.5,
            fill_factor: 100,
            pivot_tolerance: 1e-15,
            pivoting: false,
        };
        let precond_loose = IlutPreconditioner::from_csr_with_config(&matrix, &config_loose);

        let nnz_tight = precond_tight.l_values.len() + precond_tight.u_values.len();
        let nnz_loose = precond_loose.l_values.len() + precond_loose.u_values.len();

        assert!(
            nnz_loose <= nnz_tight,
            "Higher drop tolerance should produce fewer or equal non-zeros: loose={}, tight={}",
            nnz_loose,
            nnz_tight,
        );
    }

    #[test]
    fn test_ilut_fill_factor_zero() {
        // Regression: fill_factor = 0 must not cause usize underflow in
        // off_diag.truncate(max_fill - 1).  With fill_factor = 0 the
        // preconditioner should keep only the diagonal of U (and nothing in L).
        let dense = sonium_solvers_core::array![
            [c(4.0), c(-1.0), c(0.0)],
            [c(-1.0), c(4.0), c(-1.0)],
            [c(0.0), c(-1.0), c(4.0)],
        ];

        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let config = IlutConfig {
            drop_tolerance: 1e-10,
            fill_factor: 0,
            pivot_tolerance: 1e-12,
            pivoting: false,
        };

        // Should not panic.
        let precond = IlutPreconditioner::from_csr_with_config(&matrix, &config);

        // With fill_factor = 0, L should be empty (no off-diagonal fills kept)
        // and U off-diagonal should also be empty (diagonal is stored in u_diag_inv).
        assert_eq!(
            precond.l_values.len(),
            0,
            "L should be empty with fill_factor=0"
        );
        assert_eq!(
            precond.u_values.len(),
            0,
            "U off-diagonal should be empty with fill_factor=0"
        );
        assert_eq!(
            precond.u_diag_inv.len(),
            3,
            "Diagonal should still be stored"
        );
    }

    #[test]
    fn test_keep_largest_matches_stable_sort_truncate() {
        // The quickselect truncation must select exactly the same entries as
        // the previous stable full-sort-by-magnitude + truncate, including
        // magnitude ties (broken by original position).
        let reference = |entries: &[(usize, Complex64)], max_keep: usize| {
            let mut sorted: Vec<(usize, (usize, Complex64))> =
                entries.iter().copied().enumerate().collect();
            sorted.sort_by(|a, b| {
                b.1.1
                    .norm()
                    .partial_cmp(&a.1.1.norm())
                    .unwrap_or(std::cmp::Ordering::Equal)
                    // stable sort keeps original order on ties
                    .then(a.0.cmp(&b.0))
            });
            sorted.truncate(max_keep.min(sorted.len()));
            let mut kept: Vec<(usize, Complex64)> = sorted.into_iter().map(|(_, e)| e).collect();
            kept.sort_by_key(|&(j, _)| j);
            kept
        };

        // Deterministic pseudo-random magnitudes with plenty of exact ties.
        let mut state: u64 = 0xDEADBEEFCAFEF00D;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        for (len, max_keep) in [(1, 0), (5, 2), (17, 4), (64, 7), (100, 50), (33, 33)] {
            let entries: Vec<(usize, Complex64)> = (0..len)
                .map(|pos| {
                    // small value set -> many equal magnitudes; column indices
                    // are unique, as they always are within an ILUT row
                    let re = (next() % 5) as f64;
                    let im = (next() % 3) as f64;
                    (pos, Complex64::new(re, im))
                })
                .collect();

            let mut selected = entries.clone();
            keep_largest(&mut selected, max_keep);
            selected.sort_by_key(|&(j, _)| j);

            let expected = reference(&entries, max_keep);

            assert_eq!(selected.len(), expected.len(), "len={len} keep={max_keep}");
            for (sel, exp) in selected.iter().zip(&expected) {
                assert_eq!(sel, exp, "len={len} keep={max_keep}: selection mismatch");
            }
        }

        // max_keep = 0 must empty the vector (no underflow).
        let mut entries = vec![(0usize, Complex64::new(1.0, 0.0))];
        keep_largest(&mut entries, 0);
        assert!(entries.is_empty());
    }

    #[test]
    fn test_ilut_colamd_reordering_reduces_fill_arrowhead() {
        // Arrowhead matrix with the hub numbered first: eliminating the hub
        // fills the whole leaf block, so natural-order ILUT generates fill up
        // to the per-row cap. With COLAMD reordering the hub is ordered last,
        // the exact LU has zero fill, and ILUT stores only A's nonzeros.
        //
        // Measured on n = 40, fill_factor = 2, drop_tolerance = 1e-12:
        //   natural ordering:   nnz(L) + nnz(U) = 300
        //   COLAMD reordering:  nnz(L) + nnz(U) = 78  (= nnz(A) - n, zero fill)
        let n = 40;
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        dense[[0, 0]] = Complex64::new(n as f64, 0.0);
        for i in 1..n {
            dense[[i, i]] = Complex64::new(2.0, 0.0);
            dense[[0, i]] = Complex64::new(-1.0, 0.0);
            dense[[i, 0]] = Complex64::new(-1.0, 0.0);
        }
        let matrix = CsrMatrix::from_dense(&dense, 1e-15);

        let config = IlutConfig {
            drop_tolerance: 1e-12,
            fill_factor: 2,
            pivot_tolerance: 1e-15,
            pivoting: false,
        };
        let natural = IlutPreconditioner::from_csr_with_config(&matrix, &config);
        let reordered =
            IlutPreconditioner::from_csr_with_reordering(&matrix, &config, IluReordering::Colamd);

        let nnz = |pc: &IlutPreconditioner<Complex64>| pc.l_values.len() + pc.u_values.len();
        let nnz_natural = nnz(&natural);
        let nnz_reordered = nnz(&reordered);
        eprintln!(
            "ILUT arrowhead n={n}: factor nnz natural={nnz_natural}, \
             colamd-reordered={nnz_reordered}"
        );

        // Hub ordered last => zero fill => factors carry exactly the
        // off-diagonal nonzeros of A (2 per leaf).
        assert_eq!(
            nnz_reordered,
            2 * (n - 1),
            "reordered ILUT should have no fill on the arrowhead"
        );
        assert!(
            nnz_natural > 2 * nnz_reordered,
            "natural ordering should produce much more fill: {nnz_natural} vs {nnz_reordered}"
        );

        // The reordered factorization is exact: A * M^{-1} b ≈ b.
        let b: Vec<_> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.7).sin(), 0.0))
            .collect();
        let x = reordered.apply(&b);
        let residual = sonium_solvers_core::vec_ops::vec_sub(&matrix.matvec(&x), &b);
        let res_norm = residual.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        assert!(
            res_norm < 1e-10,
            "reordered ILUT should be exact on the arrowhead, residual={res_norm:.3e}"
        );

        // apply_into must scatter back to the original ordering (match apply).
        let mut y = vec![Complex64::new(0.0, 0.0); n];
        reordered.apply_into(&b, &mut y);
        for i in 0..n {
            assert!(
                (x[i] - y[i]).norm() < 1e-14,
                "apply/apply_into mismatch at {i}"
            );
        }
    }

    #[test]
    fn test_ilut_relative_pivot_threshold_scales_up_with_matrix() {
        // Regression: the pivot threshold must scale with the matrix. Row
        // norms here are ~1e6 and the (0,0) pivot is 1e-9 — negligible
        // relative to its row, yet ABOVE the old absolute 1e-12 threshold.
        //
        // Old behaviour: elimination proceeds with multiplier
        // |a_10 / u_00| = 1e6 / 1e-9 = 1e15, manufacturing a garbage L.
        // New behaviour: tol_0 = 1e-12 * ||A(0,:)||₂ ≈ 1e-6 > 1e-9, so the
        // pivot is regularized to the floor and the elimination against it
        // is skipped (multiplier zeroed).
        let dense = sonium_solvers_core::array![
            [c(1e-9), c(1e6), c(0.0)],
            [c(1e6), c(1e6), c(0.0)],
            [c(0.0), c(0.0), c(1e6)],
        ];
        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let config = IlutConfig {
            drop_tolerance: 1e-10,
            fill_factor: 10,
            pivot_tolerance: 1e-12,
            pivoting: false, // isolate the threshold semantics from pivoting
        };
        let precond = IlutPreconditioner::from_csr_with_config(&matrix, &config);

        // With the old absolute 1e-12 threshold this multiplier would be 1e15.
        for &v in &precond.l_values {
            assert!(
                v.norm() <= 1.0,
                "multiplier blew up: |L| = {:.3e} (old absolute threshold would give 1e15)",
                v.norm(),
            );
        }
        // The preconditioner application stays bounded at the matrix scale.
        let b = sonium_solvers_core::array![c(1.0), c(1.0), c(1.0)];
        let x = precond.apply(&b);
        assert!(x.iter().all(|v| v.norm().is_finite()));
        assert!(
            x.iter().all(|v| v.norm() < 1e6),
            "apply produced values above the matrix scale: {x:?}"
        );
    }

    #[test]
    fn test_ilut_relative_pivot_threshold_scales_down_with_matrix() {
        // Other direction of the same regression: on a matrix scaled ~1e-6
        // a pivot of 1e-13 is legitimately small (1e-7 relative to its row).
        // The relative threshold 1e-12 * ||row|| = 1e-25 leaves it alone.
        // The old absolute 1e-12 threshold "phase-rescaled" it up to 1e-12,
        // silently manufacturing a wrong U (inverse off by a factor of 10).
        let dense = sonium_solvers_core::array![[c(1e-6), c(0.0)], [c(0.0), c(1e-13)]];
        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let config = IlutConfig {
            drop_tolerance: 1e-10,
            fill_factor: 10,
            pivot_tolerance: 1e-12,
            pivoting: false,
        };
        let precond = IlutPreconditioner::from_csr_with_config(&matrix, &config);

        // Diagonal matrix: apply(e_1) must be exactly 1 / 1e-13 = 1e13.
        // With the old absolute threshold it would be 1e12.
        let b = sonium_solvers_core::array![c(0.0), c(1.0)];
        let x = precond.apply(&b);
        assert!(
            ((x[1].re - 1e13) / 1e13).abs() < 1e-9,
            "pivot was rescaled by the threshold: got {:?}, expected 1e13",
            x[1],
        );
    }

    #[test]
    fn test_ilutp_permutation_reconstruction() {
        // ILUTP with zero dropping and generous fill is the exact LU of
        // A P. Verify by dense reconstruction from the stored factors:
        // ||L U - A P|| ≈ 0, with (A P)[:, s] = A[:, col_perm[s]].
        let n = 5;
        let dense = sonium_solvers_core::array![
            [c(1e-8), c(2.0), c(1.0), c(0.0), c(0.0)],
            [c(1.0), c(0.5), c(0.0), c(2.0), c(0.0)],
            [c(0.0), c(1.0), c(1e-8), c(1.0), c(1.0)],
            [c(2.0), c(0.0), c(1.0), c(1.0), c(0.0)],
            [c(0.0), c(0.0), c(1.0), c(0.0), c(1e-8)],
        ];
        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let config = IlutConfig {
            drop_tolerance: 0.0,
            fill_factor: 100,
            pivot_tolerance: 1e-15,
            pivoting: true,
        };
        let precond = IlutPreconditioner::from_csr_with_config(&matrix, &config);

        // Pivoting must actually have engaged on this matrix (tiny
        // diagonals at rows 0, 2, 4 force swaps).
        let pcol = precond
            .col_perm
            .as_ref()
            .expect("pivoting must produce a non-identity permutation here");
        assert!(pcol.iter().enumerate().any(|(s, &l)| s != l));
        let mut sorted = pcol.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n).collect::<Vec<_>>(), "not a permutation");

        // Reconstruct dense L (unit diagonal; columns keyed by step) and U
        // (diagonal via 1/u_diag_inv; off-diagonal keyed by step).
        let mut l = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        for (i, row) in l.iter_mut().enumerate() {
            row[i] = c(1.0);
        }
        for i in 0..n {
            for idx in precond.l_row_ptrs[i]..precond.l_row_ptrs[i + 1] {
                l[i][precond.l_col_indices[idx] as usize] = precond.l_values[idx];
            }
        }
        let mut u = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        for (i, row) in u.iter_mut().enumerate() {
            row[i] = precond.u_diag_inv[i].inv();
        }
        for i in 0..n {
            for idx in precond.u_row_ptrs[i]..precond.u_row_ptrs[i + 1] {
                u[i][precond.u_col_indices[idx] as usize] = precond.u_values[idx];
            }
        }

        // max |L U - A P|
        let mut max_err = 0.0f64;
        for i in 0..n {
            for s in 0..n {
                let mut lu = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    lu += l[i][k] * u[k][s];
                }
                let ap = dense[[i, pcol[s]]];
                max_err = max_err.max((lu - ap).norm());
            }
        }
        assert!(
            max_err < 1e-9,
            "ILUTP reconstruction residual ||LU - AP|| = {max_err:.3e}"
        );

        // End-to-end: A * M⁻¹ b ≈ b for a non-trivial right-hand side.
        let b: Vec<_> = (0..n)
            .map(|i| Complex64::new((i as f64 * 1.3).sin(), 0.0))
            .collect();
        let x = precond.apply(&b);
        let residual = sonium_solvers_core::vec_ops::vec_sub(&matrix.matvec(&x), &b);
        let res_norm = residual.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        assert!(
            res_norm < 1e-9,
            "ILUTP apply is inconsistent with A: residual={res_norm:.3e}"
        );
    }

    #[test]
    fn ilut_checkpoint_round_trip_preserves_pivoted_factor_semantics() {
        let dense = sonium_solvers_core::array![
            [c(1e-8), c(2.0), c(1.0), c(0.0), c(0.0)],
            [c(1.0), c(0.5), c(0.0), c(2.0), c(0.0)],
            [c(0.0), c(1.0), c(1e-8), c(1.0), c(1.0)],
            [c(2.0), c(0.0), c(1.0), c(1.0), c(0.0)],
            [c(0.0), c(0.0), c(1.0), c(0.0), c(1e-8)],
        ];
        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let config = IlutConfig {
            drop_tolerance: 0.0,
            fill_factor: 100,
            pivot_tolerance: 1e-15,
            pivoting: true,
        };
        let source =
            IlutPreconditioner::from_csr_with_reordering(&matrix, &config, IluReordering::Colamd);
        let bytes = source
            .to_checkpoint_bytes()
            .expect("serialize ILUTP factor");
        let restored = IlutPreconditioner::<Complex64>::from_checkpoint_bytes(&bytes)
            .expect("restore ILUTP factor");
        let rhs: Vec<_> = (0..5)
            .map(|index| Complex64::new((index as f64 * 0.7).sin(), (index as f64 * 0.2).cos()))
            .collect();

        assert_eq!(source.perm, restored.perm);
        assert_eq!(source.col_perm, restored.col_perm);
        assert_eq!(source.diagnostics(), restored.diagnostics());
        for (expected, actual) in source.apply(&rhs).iter().zip(restored.apply(&rhs)) {
            assert!((expected - actual).norm() < 1e-14);
        }
    }

    #[test]
    fn ilut_checkpoint_rejects_corruption_and_trailing_bytes() {
        let matrix = CsrMatrix::from_raw_parts(
            3,
            3,
            vec![0, 2, 5, 7],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![5.0, 1.0, 2.0, 4.0, 1.0, 1.0, 3.0],
        );
        let source = IlutPreconditioner::try_from_csr(&matrix).expect("well-pivoted ILUT");
        let bytes = source.to_checkpoint_bytes().expect("serialize ILUT factor");

        let mut bad_magic = bytes.clone();
        bad_magic[0] ^= 0xff;
        assert!(IlutPreconditioner::<f64>::from_checkpoint_bytes(&bad_magic).is_err());

        let mut trailing = bytes;
        trailing.push(0);
        assert!(IlutPreconditioner::<f64>::from_checkpoint_bytes(&trailing).is_err());
    }

    #[test]
    fn test_ilutp_gmres_tiny_diagonal_nonsymmetric() {
        // Structurally tiny diagonal requiring permutation: diag = 1e-12,
        // superdiagonal = 2, subdiagonal = 1. Unpivoted ILUT regularizes
        // every pivot to the floor and degenerates to a diagonal scaling;
        // ILUTP pivots onto the superdiagonal and is (with zero dropping)
        // the exact factorization.
        let n = 30;
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        for i in 0..n {
            dense[[i, i]] = c(1e-12);
            if i + 1 < n {
                dense[[i, i + 1]] = c(2.0);
            }
            if i > 0 {
                dense[[i, i - 1]] = c(1.0);
            }
        }
        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let b: Vec<_> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.7).sin(), 0.0))
            .collect();

        let gmres_cfg = GmresConfig {
            max_iterations: 100,
            restart: 30,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };
        let ilut_cfg = IlutConfig {
            drop_tolerance: 0.0,
            fill_factor: 10,
            pivot_tolerance: 1e-12,
            pivoting: false,
        };
        let ilut = IlutPreconditioner::from_csr_with_config(&matrix, &ilut_cfg);
        let ilutp = IlutPreconditioner::from_csr_with_config(
            &matrix,
            &IlutConfig {
                pivoting: true,
                ..ilut_cfg.clone()
            },
        );

        let sol_nopiv = gmres_preconditioned(&matrix, &ilut, &b, &gmres_cfg);
        let sol_piv = gmres_preconditioned(&matrix, &ilutp, &b, &gmres_cfg);
        eprintln!(
            "tiny-diagonal n={n}: GMRES+ILUT converged={} iters={}, \
             GMRES+ILUTP converged={} iters={}",
            sol_nopiv.converged, sol_nopiv.iterations, sol_piv.converged, sol_piv.iterations,
        );

        // ILUTP is the exact factorization here: immediate convergence.
        assert!(sol_piv.converged, "GMRES+ILUTP must converge");
        assert!(
            sol_piv.iterations <= 5,
            "ILUTP should be near-exact, got {} iterations",
            sol_piv.iterations,
        );
        // Unpivoted ILUT stalls or needs far more iterations.
        assert!(
            !sol_nopiv.converged || sol_nopiv.iterations > 5 * sol_piv.iterations.max(1),
            "unpivoted ILUT should struggle: converged={}, iters={} vs ILUTP iters={}",
            sol_nopiv.converged,
            sol_nopiv.iterations,
            sol_piv.iterations,
        );
    }

    #[test]
    fn test_ilutp_gmres_indefinite_shifted_laplacian() {
        // Indefinite 2D shifted Laplacian on a 12x12 grid (5-point stencil,
        // lexicographic numbering): diag = 4 - k² with k² = 3.999999, i.e.
        // 1e-6. The eigenvalues straddle zero (indefinite) and the diagonal
        // is tiny relative to the row norm (||row||₂ ≈ 2). The 2D stencil
        // generates fill that ILUT must drop, and with multipliers of
        // magnitude ~1/1e-6 the unpivoted factorization amplifies the
        // dropped-fill error until the preconditioner is useless; ILUTP
        // pivots onto the O(1) off-diagonal entries, keeping multipliers
        // bounded.
        //
        // Measured (debug, this test): GMRES+ILUT does NOT converge in
        // 100x30 iterations; GMRES+ILUTP converges in 11.
        let g = 12;
        let n = g * g;
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        for row in 0..g {
            for col in 0..g {
                let i = row * g + col;
                dense[[i, i]] = c(1e-6);
                if col > 0 {
                    dense[[i, i - 1]] = c(-1.0);
                }
                if col + 1 < g {
                    dense[[i, i + 1]] = c(-1.0);
                }
                if row > 0 {
                    dense[[i, i - g]] = c(-1.0);
                }
                if row + 1 < g {
                    dense[[i, i + g]] = c(-1.0);
                }
            }
        }
        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let b: Vec<_> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.3).cos(), 0.0))
            .collect();

        let gmres_cfg = GmresConfig {
            max_iterations: 100,
            restart: 30,
            tolerance: 1e-8,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };
        let ilut_cfg = IlutConfig {
            drop_tolerance: 1e-2,
            fill_factor: 2,
            pivot_tolerance: 1e-12,
            pivoting: false,
        };
        let ilut = IlutPreconditioner::from_csr_with_config(&matrix, &ilut_cfg);
        let ilutp = IlutPreconditioner::from_csr_with_config(
            &matrix,
            &IlutConfig {
                pivoting: true,
                ..ilut_cfg.clone()
            },
        );

        let sol_nopiv = gmres_preconditioned(&matrix, &ilut, &b, &gmres_cfg);
        let sol_piv = gmres_preconditioned(&matrix, &ilutp, &b, &gmres_cfg);
        eprintln!(
            "2D shifted Laplacian {g}x{g}: GMRES+ILUT converged={} iters={}, \
             GMRES+ILUTP converged={} iters={}",
            sol_nopiv.converged, sol_nopiv.iterations, sol_piv.converged, sol_piv.iterations,
        );
        assert!(sol_piv.converged, "GMRES+ILUTP must converge");
        assert!(
            sol_piv.iterations <= 20,
            "ILUTP should converge quickly, got {} iterations",
            sol_piv.iterations,
        );
        // Unpivoted ILUT stalls or needs far more iterations.
        assert!(
            !sol_nopiv.converged || sol_nopiv.iterations >= 3 * sol_piv.iterations,
            "unpivoted ILUT should struggle: converged={}, iters={} vs ILUTP iters={}",
            sol_nopiv.converged,
            sol_nopiv.iterations,
            sol_piv.iterations,
        );
    }

    #[test]
    fn test_ilutp_default_config_pivots() {
        // Pivoting is on by default (robust choice; a no-op on diagonally
        // dominant rows) and the permutation is exposed as identity-free
        // `None` when no swap was needed.
        assert!(IlutConfig::default().pivoting);
        assert!(IlutConfig::new(1e-4, 10, 1e-12).pivoting);

        // Diagonally dominant matrix: pivoting never swaps, col_perm = None.
        let dense = sonium_solvers_core::array![
            [c(4.0), c(-1.0), c(0.0)],
            [c(-1.0), c(4.0), c(-1.0)],
            [c(0.0), c(-1.0), c(4.0)],
        ];
        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let precond = IlutPreconditioner::from_csr(&matrix);
        assert!(
            precond.col_perm.is_none(),
            "no swap should be needed on a diagonally dominant matrix"
        );

        // And the factorization matches the unpivoted one bit-for-bit.
        let unpivoted = IlutPreconditioner::from_csr_with_config(
            &matrix,
            &IlutConfig {
                pivoting: false,
                ..IlutConfig::default()
            },
        );
        let b = sonium_solvers_core::array![c(1.0), c(2.0), c(3.0)];
        assert_eq!(precond.apply(&b), unpivoted.apply(&b));
    }
}
