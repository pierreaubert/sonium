//! ILU(0) preconditioner
//!
//! Incomplete LU factorization with no fill-in.
//! Effective for many sparse systems, especially those arising from discretization.

use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use sonium_solvers_core::traits::{AdjointCapability, ComplexField, Preconditioner, SolverError};
use sonium_solvers_sparse::CsrMatrix;

/// Canonical COLAMD ordering, shared with the sparse LU factorization.
///
/// This re-export replaces the former private copy in this module (deleted);
/// sibling modules (`ilut`) keep importing it from here.
use sonium_solvers_direct::sparse_lu::colamd_ordering;

pub(crate) fn update_scaled_norm<R: Float>(scale: &mut R, sum_squares: &mut R, magnitude: R) {
    if !magnitude.is_finite() {
        *scale = magnitude;
        *sum_squares = R::one();
        return;
    }
    if magnitude == R::zero() || !scale.is_finite() {
        return;
    }
    if *scale < magnitude {
        let ratio = *scale / magnitude;
        *sum_squares = R::one() + *sum_squares * ratio * ratio;
        *scale = magnitude;
    } else {
        let ratio = magnitude / *scale;
        *sum_squares = *sum_squares + ratio * ratio;
    }
}

pub(crate) fn merge_scaled_norm<R: Float>(
    scale: &mut R,
    sum_squares: &mut R,
    other_scale: R,
    other_sum_squares: R,
) {
    if other_scale == R::zero() {
        return;
    }
    if !other_scale.is_finite() || !other_sum_squares.is_finite() {
        *scale = other_scale;
        *sum_squares = other_sum_squares;
        return;
    }
    if !scale.is_finite() {
        return;
    }
    if *scale < other_scale {
        let ratio = *scale / other_scale;
        *sum_squares = other_sum_squares + *sum_squares * ratio * ratio;
        *scale = other_scale;
    } else {
        let ratio = other_scale / *scale;
        *sum_squares = *sum_squares + other_sum_squares * ratio * ratio;
    }
}

pub(crate) fn finish_scaled_norm<R: Float>(scale: R, sum_squares: R) -> R {
    if scale == R::zero() {
        R::zero()
    } else {
        scale * sum_squares.sqrt()
    }
}

/// Fill-reducing reordering applied to the matrix before ILU factorization.
///
/// The reordering is a symmetric permutation `P A Pᵀ`: the factorization runs
/// on the permuted matrix and the triangular solves map back to the original
/// ordering, so the preconditioner still applies to vectors in the original
/// DOF numbering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IluReordering {
    /// Natural ordering (default, previous behavior).
    #[default]
    None,
    /// COLAMD-style approximate minimum degree ordering on the column
    /// intersection graph. Fill-reducing: hub vertices are ordered late, so
    /// ILU(0)/ILUT drop far less fill on matrices with a pathological
    /// natural numbering (arrowhead, star graphs, ...).
    Colamd,
}

/// Configuration for [`IluPreconditioner`].
#[derive(Debug, Clone)]
pub struct IluConfig {
    /// Fill-reducing reordering applied before factorization (default: none).
    pub reordering: IluReordering,
    /// Relative pivot threshold against the source row norm.
    ///
    /// A pivot below `pivot_tolerance * ||row||` is reported as skipped by
    /// checked construction. The default is `1e-2`.
    pub pivot_tolerance: f64,
}

impl Default for IluConfig {
    fn default() -> Self {
        Self {
            reordering: IluReordering::None,
            pivot_tolerance: 1e-2,
        }
    }
}

/// Symmetric permutation `P A Pᵀ` of a CSR matrix, `perm[new_index] = old_index`.
///
/// Each output row is sorted by column index (required by the ILU kernels,
/// which binary-search rows).
pub(crate) fn permute_symmetric<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    perm: &[usize],
) -> CsrMatrix<T> {
    let n = matrix.num_rows;
    debug_assert_eq!(perm.len(), n);
    let mut inv_perm = vec![0usize; n];
    for (new_idx, &old_idx) in perm.iter().enumerate() {
        inv_perm[old_idx] = new_idx;
    }

    let mut row_ptrs = vec![0usize];
    let mut col_indices = Vec::with_capacity(matrix.values.len());
    let mut values = Vec::with_capacity(matrix.values.len());
    let mut row_entries: Vec<(usize, T)> = Vec::new();

    for &old_row in &perm[..n] {
        row_entries.clear();
        for idx in matrix.row_ptrs[old_row]..matrix.row_ptrs[old_row + 1] {
            row_entries.push((inv_perm[matrix.col_indices[idx]], matrix.values[idx]));
        }
        row_entries.sort_by_key(|&(j, _)| j);
        for &(j, v) in &row_entries {
            col_indices.push(j);
            values.push(v);
        }
        row_ptrs.push(values.len());
    }

    CsrMatrix::from_raw_parts(n, matrix.num_cols, row_ptrs, col_indices, values)
}

/// Canonical ILU(0) factors of a CSR matrix.
///
/// Shared by every ILU-flavoured preconditioner in this crate: the sequential
/// [`IluPreconditioner`], the parallel variants in `ilu_parallel`, the
/// per-block factorizations of `ilu_hierarchical`, and the local subdomain
/// solves of `schwarz`. Produced by [`ilu0_factorize`].
#[derive(Debug, Clone)]
pub(crate) struct Ilu0Factors<T: ComplexField> {
    /// Lower triangular factor (unit diagonal; strictly lower entries j < i).
    pub(crate) l_values: Vec<T>,
    pub(crate) l_col_indices: Vec<u32>,
    pub(crate) l_row_ptrs: Vec<usize>,
    /// Upper triangular factor, off-diagonal only (j > i).
    pub(crate) u_values: Vec<T>,
    pub(crate) u_col_indices: Vec<u32>,
    pub(crate) u_row_ptrs: Vec<usize>,
    /// Exact diagonal retained for factor reconstruction and diagnostics.
    /// Reconstructing it as `inv(inv(diagonal))` can overflow for a valid
    /// finite pivot near the scalar maximum.
    pub(crate) u_diag: Vec<T>,
    /// Pre-computed inverse diagonal of U (phase-preserving fallback for
    /// near-zero pivots).
    pub(crate) u_diag_inv: Vec<T>,
    /// Matrix dimension.
    pub(crate) n: usize,
    /// Setup diagnostics retained so callers can choose a strict policy.
    pub(crate) diagnostics: IluDiagnostics<T::Real>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IluDiagnostics<R: Float> {
    pub skipped_pivots: usize,
    pub non_finite_entries: usize,
    pub min_abs_pivot: Option<R>,
}

/// Version of the portable ILU factor checkpoint format.
pub const ILU_CHECKPOINT_FORMAT_VERSION: u16 = 1;
const ILU_CHECKPOINT_MAGIC: &[u8; 8] = b"SILUCP01";

pub(crate) fn ilu_checkpoint_error(message: impl Into<String>) -> SolverError {
    SolverError::OperationFailed {
        operation: "ILU checkpoint",
        reason: message.into(),
    }
}

pub(crate) fn checkpoint_push_u64(output: &mut Vec<u8>, value: usize) -> Result<(), SolverError> {
    let value = u64::try_from(value)
        .map_err(|_| ilu_checkpoint_error("usize value does not fit checkpoint format"))?;
    output.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

pub(crate) fn checkpoint_push_usize_vec(
    output: &mut Vec<u8>,
    values: &[usize],
) -> Result<(), SolverError> {
    checkpoint_push_u64(output, values.len())?;
    for &value in values {
        checkpoint_push_u64(output, value)?;
    }
    Ok(())
}

pub(crate) fn checkpoint_push_u32_vec(
    output: &mut Vec<u8>,
    values: &[u32],
) -> Result<(), SolverError> {
    checkpoint_push_u64(output, values.len())?;
    for &value in values {
        output.extend_from_slice(&value.to_le_bytes());
    }
    Ok(())
}

pub(crate) fn checkpoint_push_values<T: ComplexField>(
    output: &mut Vec<u8>,
    values: &[T],
) -> Result<(), SolverError> {
    checkpoint_push_u64(output, values.len())?;
    for value in values {
        let re = value
            .re()
            .to_f64()
            .ok_or_else(|| ilu_checkpoint_error("real factor value cannot convert to f64"))?;
        let im = value
            .im()
            .to_f64()
            .ok_or_else(|| ilu_checkpoint_error("imaginary factor value cannot convert to f64"))?;
        if !re.is_finite() || !im.is_finite() {
            return Err(ilu_checkpoint_error("factor contains non-finite value"));
        }
        output.extend_from_slice(&re.to_le_bytes());
        output.extend_from_slice(&im.to_le_bytes());
    }
    Ok(())
}

pub(crate) fn checkpoint_push_real<R: ToPrimitive>(
    output: &mut Vec<u8>,
    value: R,
) -> Result<(), SolverError> {
    let value = value
        .to_f64()
        .ok_or_else(|| ilu_checkpoint_error("diagnostic value cannot convert to f64"))?;
    if !value.is_finite() {
        return Err(ilu_checkpoint_error("diagnostic value is non-finite"));
    }
    output.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

pub(crate) struct IluCheckpointReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> IluCheckpointReader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    pub(crate) fn take<const N: usize>(&mut self) -> Result<[u8; N], SolverError> {
        let end = self
            .offset
            .checked_add(N)
            .ok_or_else(|| ilu_checkpoint_error("checkpoint offset overflow"))?;
        let slice = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| ilu_checkpoint_error("truncated checkpoint"))?;
        let mut result = [0_u8; N];
        result.copy_from_slice(slice);
        self.offset = end;
        Ok(result)
    }

    pub(crate) fn read_u8(&mut self) -> Result<u8, SolverError> {
        Ok(self.take::<1>()?[0])
    }

    pub(crate) fn read_u16(&mut self) -> Result<u16, SolverError> {
        Ok(u16::from_le_bytes(self.take::<2>()?))
    }

    pub(crate) fn read_u32(&mut self) -> Result<u32, SolverError> {
        Ok(u32::from_le_bytes(self.take::<4>()?))
    }

    pub(crate) fn read_u64(&mut self) -> Result<u64, SolverError> {
        Ok(u64::from_le_bytes(self.take::<8>()?))
    }

    pub(crate) fn read_f64(&mut self) -> Result<f64, SolverError> {
        let value = f64::from_le_bytes(self.take::<8>()?);
        if !value.is_finite() {
            return Err(ilu_checkpoint_error("checkpoint contains non-finite value"));
        }
        Ok(value)
    }

    pub(crate) fn read_len(&mut self, element_bytes: usize) -> Result<usize, SolverError> {
        let len = usize::try_from(self.read_u64()?)
            .map_err(|_| ilu_checkpoint_error("checkpoint length does not fit usize"))?;
        if len > self.bytes.len().saturating_sub(self.offset) / element_bytes {
            return Err(ilu_checkpoint_error(
                "checkpoint vector length exceeds payload",
            ));
        }
        Ok(len)
    }

    pub(crate) fn read_usize_vec(&mut self) -> Result<Vec<usize>, SolverError> {
        let len = self.read_len(8)?;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(
                usize::try_from(self.read_u64()?)
                    .map_err(|_| ilu_checkpoint_error("checkpoint index does not fit usize"))?,
            );
        }
        Ok(values)
    }

    pub(crate) fn read_u32_vec(&mut self) -> Result<Vec<u32>, SolverError> {
        let len = self.read_len(4)?;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(self.read_u32()?);
        }
        Ok(values)
    }

    pub(crate) fn read_values<T: ComplexField>(&mut self) -> Result<Vec<T>, SolverError> {
        let len = self.read_len(16)?;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            let re = T::Real::from_f64(self.read_f64()?)
                .ok_or_else(|| ilu_checkpoint_error("real factor value cannot decode"))?;
            let im = T::Real::from_f64(self.read_f64()?)
                .ok_or_else(|| ilu_checkpoint_error("imaginary factor value cannot decode"))?;
            values.push(T::from_re_im(re, im));
        }
        Ok(values)
    }

    pub(crate) fn read_real<R: FromPrimitive>(&mut self) -> Result<R, SolverError> {
        R::from_f64(self.read_f64()?)
            .ok_or_else(|| ilu_checkpoint_error("diagnostic value cannot decode"))
    }

    pub(crate) fn is_at_end(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

fn validate_checkpoint_ptrs(
    name: &str,
    ptrs: &[usize],
    entry_count: usize,
    n: usize,
) -> Result<(), SolverError> {
    let expected = n
        .checked_add(1)
        .ok_or_else(|| ilu_checkpoint_error(format!("{name} pointer length overflows")))?;
    if ptrs.len() != expected || ptrs.first().copied() != Some(0) {
        return Err(ilu_checkpoint_error(format!(
            "{name} pointers must have length {expected} and start at zero"
        )));
    }
    if ptrs.windows(2).any(|window| window[0] > window[1]) {
        return Err(ilu_checkpoint_error(format!(
            "{name} pointers are not monotone"
        )));
    }
    if ptrs.last().copied() != Some(entry_count) {
        return Err(ilu_checkpoint_error(format!(
            "{name} pointer terminus does not match entry count"
        )));
    }
    Ok(())
}

fn validate_checkpoint_permutation(permutation: &[usize], n: usize) -> Result<(), SolverError> {
    if permutation.len() != n || permutation.iter().any(|&index| index >= n) {
        return Err(ilu_checkpoint_error(
            "permutation has invalid length or index",
        ));
    }
    let mut seen = vec![false; n];
    for &index in permutation {
        if std::mem::replace(&mut seen[index], true) {
            return Err(ilu_checkpoint_error("permutation contains duplicate index"));
        }
    }
    Ok(())
}

pub(crate) fn validate_checkpoint_factors<T: ComplexField>(
    factors: &Ilu0Factors<T>,
    perm: Option<&[usize]>,
) -> Result<(), SolverError> {
    let n = factors.n;
    validate_checkpoint_ptrs("L", &factors.l_row_ptrs, factors.l_values.len(), n)?;
    validate_checkpoint_ptrs("U", &factors.u_row_ptrs, factors.u_values.len(), n)?;
    if factors.l_col_indices.len() != factors.l_values.len()
        || factors.u_col_indices.len() != factors.u_values.len()
        || factors.u_diag.len() != n
        || factors.u_diag_inv.len() != n
    {
        return Err(ilu_checkpoint_error(
            "factor vector lengths are inconsistent",
        ));
    }
    for &column in &factors.l_col_indices {
        if column as usize >= n {
            return Err(ilu_checkpoint_error("L column index is out of range"));
        }
    }
    for &column in &factors.u_col_indices {
        if column as usize >= n {
            return Err(ilu_checkpoint_error("U column index is out of range"));
        }
    }
    for row in 0..n {
        for &column in &factors.l_col_indices[factors.l_row_ptrs[row]..factors.l_row_ptrs[row + 1]]
        {
            if column as usize >= row {
                return Err(ilu_checkpoint_error(
                    "L contains a non-strictly-lower entry",
                ));
            }
        }
        for &column in &factors.u_col_indices[factors.u_row_ptrs[row]..factors.u_row_ptrs[row + 1]]
        {
            if column as usize <= row {
                return Err(ilu_checkpoint_error(
                    "U contains a non-strictly-upper entry",
                ));
            }
        }
    }
    if factors
        .l_values
        .iter()
        .chain(factors.u_values.iter())
        .chain(factors.u_diag.iter())
        .chain(factors.u_diag_inv.iter())
        .any(|value| !value.norm().is_finite())
    {
        return Err(ilu_checkpoint_error("factor contains non-finite value"));
    }
    if let Some(permutation) = perm {
        validate_checkpoint_permutation(permutation, n)?;
    }
    if factors
        .diagnostics
        .min_abs_pivot
        .is_some_and(|value| !value.is_finite() || value < T::Real::zero())
    {
        return Err(ilu_checkpoint_error(
            "minimum pivot diagnostic is non-finite",
        ));
    }
    Ok(())
}

/// Canonical ILU(0) factorization (no fill-in) of a CSR matrix.
///
/// Single implementation of the kernel that used to be copy-pasted into
/// `ilu`, `ilu_parallel` (twice), `ilu_hierarchical`, and `schwarz` with
/// subtly divergent pivot-skip behaviour. The unified semantics are:
///
/// - Row norms of the *original* matrix set a relative pivot threshold:
///   elimination of column `k` is skipped when `|u_kk| < 1e-2 *
///   max(||A(k,:)||₂, 1e-30)`, or when row `k` has no diagonal entry in the
///   pattern at all. This prevents garbage multipliers (`|L_ik| >> 1`) on
///   indefinite systems (Helmholtz, Maxwell) with legitimately tiny
///   diagonals.
/// - On a skipped pivot the would-be multiplier slot is **zeroed**, so the
///   stale matrix entry cannot leak into `L` and corrupt the forward
///   substitution. (Historical divergence: `schwarz` zeroed the entry while
///   the other copies left the stale value in the pattern position; the
///   zeroing semantic is the one regression-tested against issue #13 and is
///   now canonical.)
/// - A tiny final diagonal `u_ii` is inverted with the phase-preserving
///   fallback: scale by the row norm and keep the complex phase of `u_ii`
///   instead of substituting a purely real inverse.
///
/// Requires rows sorted by column index (guaranteed by `CsrMatrix`
/// constructors) so the U-row lookup can binary-search.
pub(crate) fn ilu0_factorize<T: ComplexField>(matrix: &CsrMatrix<T>) -> Ilu0Factors<T> {
    ilu0_factorize_with_tolerance(matrix, 1e-2)
}

fn ilu0_factorize_with_tolerance<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    pivot_tolerance: f64,
) -> Ilu0Factors<T> {
    let n = matrix.num_rows;
    let mut diagnostics: IluDiagnostics<T::Real> = IluDiagnostics {
        skipped_pivots: 0,
        non_finite_entries: matrix
            .values
            .iter()
            .filter(|value| !value.norm().is_finite())
            .count(),
        min_abs_pivot: None,
    };

    let col_indices = &matrix.col_indices;
    let row_ptrs = &matrix.row_ptrs;

    let mut diag_indices = vec![usize::MAX; n];
    for i in 0..n {
        #[allow(clippy::needless_range_loop)]
        for idx in row_ptrs[i]..row_ptrs[i + 1] {
            if col_indices[idx] == i {
                diag_indices[i] = idx;
                break;
            }
        }
    }

    // Compute original row norms for the relative pivot threshold.
    //
    // For indefinite matrices (Helmholtz, Maxwell), the diagonal can be
    // legitimately near-zero or purely imaginary. A fixed absolute threshold
    // is meaningless: on a matrix with entries ~1e6, a pivot of 1.0 is
    // effectively zero. We use a relative threshold against the row norm.
    diagnostics.skipped_pivots += diag_indices
        .iter()
        .filter(|&&index| index == usize::MAX)
        .count();

    let row_norms: Vec<T::Real> = (0..n)
        .map(|i| {
            let mut scale = T::Real::zero();
            let mut sum_squares = T::Real::one();
            for idx in row_ptrs[i]..row_ptrs[i + 1] {
                update_scaled_norm(&mut scale, &mut sum_squares, matrix.values[idx].norm());
            }
            finish_scaled_norm(scale, sum_squares)
        })
        .collect();

    // Relative drop threshold: |pivot| < tol × ||row|| → skip this row
    // 1e-2 captures genuinely tiny pivots without dropping useful ones.
    // A separate absolute floor handles matrices with entries near machine
    // epsilon.
    let pivot_tol_rel = T::Real::from_f64(pivot_tolerance).unwrap();
    let pivot_tol_abs = T::Real::from_f64(1e-30).unwrap();

    let mut values = matrix.values.clone();

    for i in 0..n {
        for idx in row_ptrs[i]..row_ptrs[i + 1] {
            let k = col_indices[idx];
            if k >= i {
                break;
            }

            let u_kk_idx = diag_indices[k];
            if u_kk_idx == usize::MAX {
                // No diagonal in row k: skip the elimination and zero the
                // would-be multiplier so the stale entry does not land in L.
                values[idx] = T::zero();
                diagnostics.skipped_pivots += 1;
                continue;
            }

            let u_kk = values[u_kk_idx];
            let u_kk_norm = u_kk.norm();
            diagnostics.min_abs_pivot = Some(
                diagnostics
                    .min_abs_pivot
                    .map_or(u_kk_norm, |current: T::Real| current.min(u_kk_norm)),
            );

            // Relative threshold: skip elimination if pivot is tiny
            // compared with the row it came from. This prevents
            // garbage multipliers (|L_ik| >> 1) from destroying the
            // factorization quality. The multiplier slot is zeroed so the
            // stale entry does not land in L.
            let tol = pivot_tol_rel * row_norms[k].max(pivot_tol_abs);
            if u_kk_norm < tol {
                values[idx] = T::zero();
                diagnostics.skipped_pivots += 1;
                continue;
            }

            let l_ik = values[idx] * u_kk.inv();
            values[idx] = l_ik;

            for j_idx in row_ptrs[i]..row_ptrs[i + 1] {
                let j = col_indices[j_idx];
                if j <= k {
                    continue;
                }

                let u_kj_idx = diag_indices[k] + 1;
                if u_kj_idx < row_ptrs[k + 1] && col_indices[u_kj_idx] == j {
                    values[j_idx] = values[j_idx] - l_ik * values[u_kj_idx];
                } else {
                    // Binary search in the sorted column list of row k
                    let cols_k = &col_indices[(row_ptrs[k] + 1)..row_ptrs[k + 1]];
                    if let Ok(pos) = cols_k.binary_search(&j) {
                        let search_idx = row_ptrs[k] + 1 + pos;
                        values[j_idx] = values[j_idx] - l_ik * values[search_idx];
                    }
                }
            }
        }
    }

    let mut l_values = Vec::new();
    let mut l_col_indices: Vec<u32> = Vec::new();
    let mut l_row_ptrs = vec![0];

    let mut u_values = Vec::new();
    let mut u_col_indices: Vec<u32> = Vec::new();
    let mut u_row_ptrs = vec![0];
    let mut u_diag = vec![T::one(); n];

    for i in 0..n {
        for idx in row_ptrs[i]..row_ptrs[i + 1] {
            let j = col_indices[idx];
            let val = values[idx];

            if j < i {
                l_values.push(val);
                l_col_indices.push(j as u32);
            } else if j == i {
                u_diag[i] = val;
            } else {
                // j > i only — off-diagonal U entries
                u_values.push(val);
                u_col_indices.push(j as u32);
            }
        }
        l_row_ptrs.push(l_values.len());
        u_row_ptrs.push(u_values.len());
    }

    // The diagonal used by the triangular solve is the final pivot after
    // elimination, not merely the pivot observed while processing a lower
    // entry.  Count zero/tiny/non-finite final pivots as setup diagnostics so
    // checked constructors cannot silently accept a factor whose inverse was
    // fabricated by the compatibility fallback below.
    diagnostics.non_finite_entries = values
        .iter()
        .filter(|value| !value.norm().is_finite())
        .count();
    for (i, &diagonal) in u_diag.iter().enumerate() {
        let diagonal_norm = diagonal.norm();
        diagnostics.min_abs_pivot = Some(
            diagnostics
                .min_abs_pivot
                .map_or(diagonal_norm, |current: T::Real| current.min(diagonal_norm)),
        );
        let row_scale = row_norms[i].max(pivot_tol_abs);
        let pivot_limit = pivot_tol_rel * row_scale;
        if !diagonal_norm.is_finite() || diagonal_norm <= pivot_limit.max(pivot_tol_abs) {
            diagnostics.skipped_pivots += 1;
        }
    }

    let u_diag_inv: Vec<T> = u_diag
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            let d_norm: T::Real = d.norm();
            if d_norm > T::Real::from_f64(1e-20).unwrap() {
                d.inv()
            } else {
                // Fallback: near-zero diagonal. Use row norm as scale so the
                // inverse has the right magnitude. Preserve the complex phase.
                let fallback: T::Real = row_norms[i].max(T::Real::from_f64(1.0).unwrap());
                if d_norm > T::Real::from_f64(1e-30).unwrap() {
                    let scale: T::Real = T::Real::one() / (d_norm * fallback);
                    d.conj() * T::from_real(scale)
                } else {
                    // d is effectively zero; no phase to preserve
                    T::from_real(fallback).inv()
                }
            }
        })
        .collect();

    Ilu0Factors {
        l_values,
        l_col_indices,
        l_row_ptrs,
        u_values,
        u_col_indices,
        u_row_ptrs,
        u_diag,
        u_diag_inv,
        n,
        diagnostics,
    }
}

/// Checked ILU(0) setup for consumers that cannot safely retain the
/// compatibility fallback used by [`ilu0_factorize`].
pub(crate) fn try_ilu0_factorize<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    operation: &'static str,
) -> Result<Ilu0Factors<T>, SolverError> {
    matrix.validate_canonical_structure()?;
    let factors = ilu0_factorize(matrix);
    let diagnostics = factors.diagnostics;

    if diagnostics.non_finite_entries != 0 {
        return Err(SolverError::OperationFailed {
            operation,
            reason: format!(
                "{} non-finite matrix entries",
                diagnostics.non_finite_entries
            ),
        });
    }
    if diagnostics.skipped_pivots != 0 {
        return Err(SolverError::OperationFailed {
            operation,
            reason: format!(
                "{} pivots were below the configured relative threshold",
                diagnostics.skipped_pivots
            ),
        });
    }

    Ok(factors)
}

impl<T: ComplexField> Ilu0Factors<T> {
    /// Forward/backward substitution in-place on a mutable slice.
    #[inline]
    pub(crate) fn solve_in_place(&self, s: &mut [T]) {
        let l_vals = self.l_values.as_slice();
        let l_cols = self.l_col_indices.as_slice();
        let l_ptrs = self.l_row_ptrs.as_slice();
        let u_vals = self.u_values.as_slice();
        let u_cols = self.u_col_indices.as_slice();
        let u_ptrs = self.u_row_ptrs.as_slice();

        // Forward substitution: Ly = r (L has unit diagonal)
        for i in 0..self.n {
            let mut sum = T::zero();
            for idx in l_ptrs[i]..l_ptrs[i + 1] {
                unsafe {
                    let j = *l_cols.get_unchecked(idx) as usize;
                    sum += *l_vals.get_unchecked(idx) * *s.get_unchecked(j);
                }
            }
            s[i] -= sum;
        }

        // Backward substitution: Ux = y
        // U stores only off-diagonal entries (j > i); diagonal handled via u_diag_inv.
        for i in (0..self.n).rev() {
            let mut sum = T::zero();
            for idx in u_ptrs[i]..u_ptrs[i + 1] {
                unsafe {
                    let j = *u_cols.get_unchecked(idx) as usize;
                    sum += *u_vals.get_unchecked(idx) * *s.get_unchecked(j);
                }
            }
            s[i] -= sum;
            s[i] *= self.u_diag_inv[i];
        }
    }

    /// Adjoint (Hermitian) solve in-place: `s ← M⁻ᴴ s = L⁻ᴴ (U⁻ᴴ s)` using
    /// the stored factors — no refactorization.
    ///
    /// Conjugation is applied to the stored entries on the fly and the
    /// pre-computed `u_diag_inv` is conjugated (cheap, exact) rather than
    /// recomputed. Both sweeps use the CSR row layout with scatter
    /// accumulation, since column access is unavailable:
    ///
    /// - `Uᴴ z = r`: `Uᴴ` is lower triangular. Forward sweep — once `z_i`
    ///   is known, scatter `-conj(u_ij) · z_i` into the rows `j > i` of
    ///   `Uᴴ` (which are the row-`i` entries of `U`).
    /// - `Lᴴ y = z`: `Lᴴ` is unit upper triangular. Backward sweep with the
    ///   symmetric scatter into `j < i`.
    #[inline]
    pub(crate) fn solve_hermitian_in_place(&self, s: &mut [T]) {
        let l_vals = self.l_values.as_slice();
        let l_cols = self.l_col_indices.as_slice();
        let l_ptrs = self.l_row_ptrs.as_slice();
        let u_vals = self.u_values.as_slice();
        let u_cols = self.u_col_indices.as_slice();
        let u_ptrs = self.u_row_ptrs.as_slice();

        // Forward sweep: Uᴴ z = r (lower triangular, scatter into j > i).
        for i in 0..self.n {
            // s[i] now holds r_i - Σ_{k<i} conj(U_ki) z_k; the diagonal of
            // Uᴴ is conj(u_ii), and conj(u_ii)⁻¹ = conj(u_diag_inv[i]).
            s[i] *= self.u_diag_inv[i].conj();
            for idx in u_ptrs[i]..u_ptrs[i + 1] {
                unsafe {
                    let j = *u_cols.get_unchecked(idx) as usize;
                    let z_i = *s.get_unchecked(i);
                    *s.get_unchecked_mut(j) -= u_vals.get_unchecked(idx).conj() * z_i;
                }
            }
        }

        // Backward sweep: Lᴴ y = z (unit upper triangular, scatter into j < i).
        for i in (0..self.n).rev() {
            for idx in l_ptrs[i]..l_ptrs[i + 1] {
                unsafe {
                    let j = *l_cols.get_unchecked(idx) as usize;
                    let y_i = *s.get_unchecked(i);
                    *s.get_unchecked_mut(j) -= l_vals.get_unchecked(idx).conj() * y_i;
                }
            }
        }
    }

    /// Algebraic-transpose solve in-place: `s ← M⁻ᵀ s = L⁻ᵀ (U⁻ᵀ s)`.
    /// Identical sweep structure to [`Self::solve_hermitian_in_place`]
    /// without the conjugation.
    #[inline]
    pub(crate) fn solve_transpose_in_place(&self, s: &mut [T]) {
        let l_vals = self.l_values.as_slice();
        let l_cols = self.l_col_indices.as_slice();
        let l_ptrs = self.l_row_ptrs.as_slice();
        let u_vals = self.u_values.as_slice();
        let u_cols = self.u_col_indices.as_slice();
        let u_ptrs = self.u_row_ptrs.as_slice();

        for i in 0..self.n {
            s[i] *= self.u_diag_inv[i];
            for idx in u_ptrs[i]..u_ptrs[i + 1] {
                unsafe {
                    let j = *u_cols.get_unchecked(idx) as usize;
                    let z_i = *s.get_unchecked(i);
                    *s.get_unchecked_mut(j) -= *u_vals.get_unchecked(idx) * z_i;
                }
            }
        }

        for i in (0..self.n).rev() {
            for idx in l_ptrs[i]..l_ptrs[i + 1] {
                unsafe {
                    let j = *l_cols.get_unchecked(idx) as usize;
                    let y_i = *s.get_unchecked(i);
                    *s.get_unchecked_mut(j) -= *l_vals.get_unchecked(idx) * y_i;
                }
            }
        }
    }
}

/// Row-level schedule for one triangular factor: `rows[starts[l]..starts[l + 1]]`
/// are the rows of level `l`. Every dependency of a level-`l` row lives in
/// an earlier level, so rows within a level can be solved concurrently and
/// each row sum keeps its serial accumulation order (bit-identical results
/// at any thread count and on any architecture).
fn compute_level_schedule(
    col_indices: &[u32],
    row_ptrs: &[usize],
    n: usize,
    reverse: bool,
) -> (Vec<usize>, Vec<usize>) {
    let mut level_of = vec![0usize; n];
    let visit = |i: usize, level_of: &mut [usize]| {
        let mut deepest = 0usize;
        for &j in &col_indices[row_ptrs[i]..row_ptrs[i + 1]] {
            deepest = deepest.max(level_of[j as usize] + 1);
        }
        level_of[i] = deepest;
    };
    if reverse {
        for i in (0..n).rev() {
            visit(i, &mut level_of);
        }
    } else {
        for i in 0..n {
            visit(i, &mut level_of);
        }
    }
    let level_count = level_of.iter().copied().max().map_or(0, |m| m + 1);
    let mut starts = vec![0usize; level_count + 1];
    for &level in &level_of {
        starts[level + 1] += 1;
    }
    for level in 0..level_count {
        starts[level + 1] += starts[level];
    }
    let mut rows = vec![0usize; n];
    let mut cursor = starts[..level_count].to_vec();
    for (row, &level) in level_of.iter().enumerate() {
        rows[cursor[level]] = row;
        cursor[level] += 1;
    }
    (rows, starts)
}

/// ILU(0) preconditioner
///
/// Computes an incomplete LU factorization where the sparsity pattern
/// of L and U is the same as the original matrix.
///
/// Applies run serially on one thread and through a level-scheduled
/// wavefront on multiple threads; both paths accumulate each row sum in
/// the same order, so results are bit-identical at any thread count.
#[derive(Debug, Clone)]
pub struct IluPreconditioner<T: ComplexField> {
    /// Canonical ILU(0) factors of the (possibly permuted) matrix.
    factors: Ilu0Factors<T>,
    /// Fill-reducing permutation applied before factorization,
    /// `perm[new_index] = old_index`. `None` for natural ordering.
    perm: Option<Vec<usize>>,
    /// Forward (L) level offsets over the factored ordering
    /// (length = levels + 1); only the level count gates the wavefront.
    forward_level_starts: Vec<usize>,
    /// Backward (U) level offsets over the factored ordering
    /// (length = levels + 1); only the level count gates the wavefront.
    backward_level_starts: Vec<usize>,
}

impl<T: ComplexField> IluPreconditioner<T> {
    /// Assemble a preconditioner from factors, deriving the level
    /// schedules used by the parallel triangular apply.
    fn assemble(factors: Ilu0Factors<T>, perm: Option<Vec<usize>>) -> Self {
        let (_, forward_level_starts) = compute_level_schedule(
            &factors.l_col_indices,
            &factors.l_row_ptrs,
            factors.n,
            false,
        );
        let (_, backward_level_starts) =
            compute_level_schedule(&factors.u_col_indices, &factors.u_row_ptrs, factors.n, true);
        Self {
            factors,
            perm,
            forward_level_starts,
            backward_level_starts,
        }
    }
}

impl<T: ComplexField> IluPreconditioner<T> {
    pub(crate) fn from_factors(
        factors: Ilu0Factors<T>,
        perm: Option<Vec<usize>>,
    ) -> Result<Self, sonium_solvers_core::traits::SolverError> {
        validate_checkpoint_factors(&factors, perm.as_deref())?;
        Ok(Self::assemble(factors, perm))
    }

    pub(crate) fn into_factors(self) -> (Ilu0Factors<T>, Option<Vec<usize>>) {
        (self.factors, self.perm)
    }

    /// Borrow the canonical ILU(0) factors (device upload path).
    pub(crate) fn factors(&self) -> &Ilu0Factors<T> {
        &self.factors
    }

    /// Borrow the fill-reducing permutation, if one was applied.
    pub(crate) fn permutation(&self) -> Option<&[usize]> {
        self.perm.as_deref()
    }

    /// Construct ILU(0) and reject non-finite input or skipped pivots rather
    /// than silently fabricating a fallback inverse diagonal.
    pub fn try_from_csr(
        matrix: &CsrMatrix<T>,
    ) -> Result<Self, sonium_solvers_core::traits::SolverError> {
        Self::try_from_csr_with_config(matrix, &IluConfig::default())
    }

    pub fn try_from_csr_with_config(
        matrix: &CsrMatrix<T>,
        config: &IluConfig,
    ) -> Result<Self, sonium_solvers_core::traits::SolverError> {
        if !config.pivot_tolerance.is_finite() || !(0.0..=1.0).contains(&config.pivot_tolerance) {
            return Err(sonium_solvers_core::traits::SolverError::OperationFailed {
                operation: "ILU setup",
                reason: "pivot_tolerance must be finite and in [0, 1]".into(),
            });
        }
        matrix.validate_canonical_structure()?;
        let preconditioner = Self::from_csr_with_config(matrix, config);
        let diagnostics = preconditioner.diagnostics();
        if diagnostics.non_finite_entries != 0 {
            return Err(sonium_solvers_core::traits::SolverError::OperationFailed {
                operation: "ILU setup",
                reason: format!(
                    "{} non-finite matrix entries",
                    diagnostics.non_finite_entries
                ),
            });
        }
        if diagnostics.skipped_pivots != 0 {
            return Err(sonium_solvers_core::traits::SolverError::OperationFailed {
                operation: "ILU setup",
                reason: format!(
                    "{} pivots were below the configured relative threshold",
                    diagnostics.skipped_pivots
                ),
            });
        }
        Ok(preconditioner)
    }

    pub fn diagnostics(&self) -> &IluDiagnostics<T::Real> {
        &self.factors.diagnostics
    }

    /// Serialize the factor and setup state into a portable checkpoint.
    ///
    /// Numeric values are stored as little-endian `f64` real/imaginary pairs,
    /// with scalar sizes recorded so a checkpoint cannot silently be restored
    /// into a different scalar family.
    pub fn to_checkpoint_bytes(&self) -> Result<Vec<u8>, SolverError> {
        validate_checkpoint_factors(&self.factors, self.perm.as_deref())?;

        let mut output = Vec::new();
        output.extend_from_slice(ILU_CHECKPOINT_MAGIC);
        output.extend_from_slice(&ILU_CHECKPOINT_FORMAT_VERSION.to_le_bytes());
        let flags = u8::from(self.perm.is_some());
        output.push(flags);
        output.push(std::mem::size_of::<T>() as u8);
        output.push(std::mem::size_of::<T::Real>() as u8);
        checkpoint_push_u64(&mut output, self.factors.n)?;
        checkpoint_push_usize_vec(&mut output, &self.factors.l_row_ptrs)?;
        checkpoint_push_u32_vec(&mut output, &self.factors.l_col_indices)?;
        checkpoint_push_values(&mut output, &self.factors.l_values)?;
        checkpoint_push_usize_vec(&mut output, &self.factors.u_row_ptrs)?;
        checkpoint_push_u32_vec(&mut output, &self.factors.u_col_indices)?;
        checkpoint_push_values(&mut output, &self.factors.u_values)?;
        checkpoint_push_values(&mut output, &self.factors.u_diag)?;
        checkpoint_push_values(&mut output, &self.factors.u_diag_inv)?;
        if let Some(permutation) = self.perm.as_deref() {
            checkpoint_push_usize_vec(&mut output, permutation)?;
        }
        checkpoint_push_u64(&mut output, self.factors.diagnostics.skipped_pivots)?;
        checkpoint_push_u64(&mut output, self.factors.diagnostics.non_finite_entries)?;
        match self.factors.diagnostics.min_abs_pivot {
            Some(value) => {
                output.push(1);
                checkpoint_push_real(&mut output, value)?;
            }
            None => output.push(0),
        }
        Ok(output)
    }

    /// Restore an ILU factor produced by [`Self::to_checkpoint_bytes`].
    pub fn from_checkpoint_bytes(bytes: &[u8]) -> Result<Self, SolverError> {
        let mut reader = IluCheckpointReader::new(bytes);
        if &reader.take::<8>()? != ILU_CHECKPOINT_MAGIC {
            return Err(ilu_checkpoint_error("bad magic"));
        }
        let version = reader.read_u16()?;
        if version != ILU_CHECKPOINT_FORMAT_VERSION {
            return Err(ilu_checkpoint_error(format!(
                "unsupported format version {version}"
            )));
        }
        let flags = reader.read_u8()?;
        if flags & !1 != 0 {
            return Err(ilu_checkpoint_error("unknown checkpoint flags"));
        }
        let scalar_size = reader.read_u8()?;
        let real_size = reader.read_u8()?;
        if scalar_size as usize != std::mem::size_of::<T>()
            || real_size as usize != std::mem::size_of::<T::Real>()
        {
            return Err(ilu_checkpoint_error(
                "checkpoint scalar family does not match restore type",
            ));
        }
        let n = usize::try_from(reader.read_u64()?)
            .map_err(|_| ilu_checkpoint_error("dimension does not fit usize"))?;
        let l_row_ptrs = reader.read_usize_vec()?;
        let l_col_indices = reader.read_u32_vec()?;
        let l_values = reader.read_values::<T>()?;
        let u_row_ptrs = reader.read_usize_vec()?;
        let u_col_indices = reader.read_u32_vec()?;
        let u_values = reader.read_values::<T>()?;
        let u_diag = reader.read_values::<T>()?;
        let u_diag_inv = reader.read_values::<T>()?;
        let perm = if flags & 1 != 0 {
            Some(reader.read_usize_vec()?)
        } else {
            None
        };
        let skipped_pivots = usize::try_from(reader.read_u64()?)
            .map_err(|_| ilu_checkpoint_error("skipped-pivot count does not fit usize"))?;
        let non_finite_entries = usize::try_from(reader.read_u64()?)
            .map_err(|_| ilu_checkpoint_error("non-finite count does not fit usize"))?;
        let min_abs_pivot = match reader.read_u8()? {
            0 => None,
            1 => Some(reader.read_real::<T::Real>()?),
            _ => return Err(ilu_checkpoint_error("invalid minimum-pivot presence flag")),
        };
        if !reader.is_at_end() {
            return Err(ilu_checkpoint_error("trailing bytes"));
        }

        let factors = Ilu0Factors {
            l_values,
            l_col_indices,
            l_row_ptrs,
            u_values,
            u_col_indices,
            u_row_ptrs,
            u_diag,
            u_diag_inv,
            n,
            diagnostics: IluDiagnostics {
                skipped_pivots,
                non_finite_entries,
                min_abs_pivot,
            },
        };
        validate_checkpoint_factors(&factors, perm.as_deref())?;
        Ok(Self::assemble(factors, perm))
    }

    /// Create ILU(0) preconditioner from a CSR matrix
    pub fn from_csr(matrix: &CsrMatrix<T>) -> Self {
        Self::from_csr_with_reordering(matrix, IluReordering::None)
    }

    /// Create ILU(0) preconditioner with an explicit reordering choice.
    pub fn from_csr_with_reordering(matrix: &CsrMatrix<T>, reordering: IluReordering) -> Self {
        Self::from_csr_with_config(
            matrix,
            &IluConfig {
                reordering,
                ..IluConfig::default()
            },
        )
    }

    /// Create ILU(0) preconditioner from a CSR matrix with custom configuration.
    ///
    /// With `IluReordering::Colamd` the matrix is symmetrically permuted
    /// (`P A Pᵀ`) with a fill-reducing ordering before factorization; the
    /// triangular solves map results back to the original numbering, so
    /// `apply` still takes and returns vectors in the original ordering.
    pub fn from_csr_with_config(matrix: &CsrMatrix<T>, config: &IluConfig) -> Self {
        let pivot_tolerance = if config.pivot_tolerance.is_finite()
            && (0.0..=1.0).contains(&config.pivot_tolerance)
        {
            config.pivot_tolerance
        } else {
            IluConfig::default().pivot_tolerance
        };
        match config.reordering {
            IluReordering::None => Self::factorize(matrix, None, pivot_tolerance),
            IluReordering::Colamd => {
                let perm = colamd_ordering(matrix);
                let permuted = permute_symmetric(matrix, &perm);
                Self::factorize(&permuted, Some(perm), pivot_tolerance)
            }
        }
    }

    /// ILU(0) factorization of `matrix`, optionally recording the permutation
    /// that was used to produce it.
    fn factorize(matrix: &CsrMatrix<T>, perm: Option<Vec<usize>>, pivot_tolerance: f64) -> Self {
        Self::assemble(ilu0_factorize_with_tolerance(matrix, pivot_tolerance), perm)
    }
}

impl<T: ComplexField + Send + Sync> Preconditioner<T> for IluPreconditioner<T> {
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

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn apply_transpose_into(&self, r: &[T], y: &mut [T]) {
        self.apply_adjoint_into(r, y, Ilu0Factors::solve_transpose_in_place);
    }

    /// `y = M⁻ᴴ-applied r` — i.e. `y = L⁻ᴴ (U⁻ᴴ r)` with the same stored
    /// factors as the forward apply. This makes the forward ILU
    /// factorization reusable as a preconditioner for the adjoint system
    /// `Aᴴ λ = g`: `Mᴴ ≈ A⁻ᴴ` whenever `M ≈ A⁻¹`.
    fn apply_hermitian(&self, r: &[T]) -> Vec<T> {
        let mut x = r.to_vec();
        self.apply_hermitian_into(r, &mut x);
        x
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn apply_hermitian_into(&self, r: &[T], y: &mut [T]) {
        self.apply_adjoint_into(r, y, Ilu0Factors::solve_hermitian_in_place);
    }
}

/// Smallest matrix dimension eligible for the level-wavefront apply.
///
/// Measured on a chiplet x86 (Threadripper PRO 3995WX) with 2D FEM
/// saddle systems: the wavefront loses to the serial kernel at every
/// tested size (12k, 50k, 120k rows) and thread count, because freshly
/// written dependency values cross die boundaries at full coherence
/// latency while the serial kernel stays cache-resident. The gate
/// therefore only admits very large, DRAM-resident systems, where the
/// balance can flip (and uniform-latency architectures, e.g. ARM
/// servers, where cross-core latency is lower). Validated sizes keep
/// the bit-identical serial kernel.
const LEVEL_WAVEFRONT_MIN_N: usize = 1_048_576;
/// Minimum average rows per level for wavefront synchronization to pay.
const LEVEL_WAVEFRONT_MIN_AVG_WIDTH: usize = 32;

impl<T: ComplexField + Send + Sync> IluPreconditioner<T> {
    /// Forward/backward substitution in-place on a mutable slice.
    ///
    /// With multiple threads and a wide enough schedule this uses a
    /// level-scheduled wavefront; otherwise it uses the serial factor
    /// kernel. Both accumulate each row sum in CSR order, so results are
    /// bit-identical at any thread count.
    #[inline]
    fn solve_in_place(&self, s: &mut [T]) {
        if self.use_level_wavefront() {
            self.solve_level_wavefront(s);
        } else {
            self.factors.solve_in_place(s);
        }
    }

    /// Whether the level wavefront should replace the serial kernel.
    fn use_level_wavefront(&self) -> bool {
        let n = self.factors.n;
        if n < LEVEL_WAVEFRONT_MIN_N || rayon::current_num_threads() < 2 {
            return false;
        }
        let forward_levels = self.forward_level_starts.len().saturating_sub(1);
        let backward_levels = self.backward_level_starts.len().saturating_sub(1);
        forward_levels > 0
            && backward_levels > 0
            && forward_levels * LEVEL_WAVEFRONT_MIN_AVG_WIDTH <= n
            && backward_levels * LEVEL_WAVEFRONT_MIN_AVG_WIDTH <= n
    }

    /// Level-scheduled forward and backward substitution.
    fn solve_level_wavefront(&self, s: &mut [T]) {
        self.run_sliced_sweep(
            &self.factors.l_values,
            &self.factors.l_col_indices,
            &self.factors.l_row_ptrs,
            None,
            false,
            s,
        );
        self.run_sliced_sweep(
            &self.factors.u_values,
            &self.factors.u_col_indices,
            &self.factors.u_row_ptrs,
            Some(&self.factors.u_diag_inv),
            true,
            s,
        );
    }

    /// One triangular sweep in natural row order, sliced into one
    /// contiguous row range per worker. A worker solves its slice in
    /// dependency order (in-slice dependencies are satisfied by that
    /// order, and contiguous ranges keep the worker's data cache-local)
    /// and spins only on cross-slice dependencies, tracked by per-worker
    /// progress counters. Dependencies always point into earlier slices
    /// (forward) or later slices (backward), so waits form a DAG and
    /// cannot deadlock; one fork/join covers the whole sweep.
    #[allow(clippy::too_many_arguments)]
    fn run_sliced_sweep(
        &self,
        values: &[T],
        col_indices: &[u32],
        row_ptrs: &[usize],
        diag_inv: Option<&[T]>,
        reverse: bool,
        s: &mut [T],
    ) {
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Shared mutable access to the solution buffer. A row reads only
        // rows at earlier positions (earlier levels) and writes only its
        // own entry; progress counters make those writes visible
        // (release/acquire) before any dependent read.
        #[derive(Clone, Copy)]
        struct SharedSlice<U>(*mut U);
        unsafe impl<U: Send> Send for SharedSlice<U> {}
        unsafe impl<U: Send> Sync for SharedSlice<U> {}
        impl<U: Copy> SharedSlice<U> {
            fn get(&self, i: usize) -> U {
                unsafe { *self.0.add(i) }
            }
            fn set(&self, i: usize, value: U) {
                unsafe { *self.0.add(i) = value }
            }
        }

        let n = s.len();
        if n == 0 {
            return;
        }
        let shared = SharedSlice(s.as_mut_ptr());
        let solve_row = |i: usize| {
            let mut sum = T::zero();
            for idx in row_ptrs[i]..row_ptrs[i + 1] {
                let j = col_indices[idx] as usize;
                sum += values[idx] * shared.get(j);
            }
            match diag_inv {
                None => shared.set(i, shared.get(i) - sum),
                Some(diag) => shared.set(i, (shared.get(i) - sum) * diag[i]),
            }
        };
        let workers = rayon::current_num_threads().max(1).min(n);
        if workers <= 1 {
            if reverse {
                for i in (0..n).rev() {
                    solve_row(i);
                }
            } else {
                for i in 0..n {
                    solve_row(i);
                }
            }
            return;
        }
        let slice = n.div_ceil(workers);
        // Frontier counters, one padded slot per worker. Forward:
        // progress[w] is the next unsolved row of slice w. Backward:
        // progress[w] is the lowest unsolved row of slice w. Publishing
        // every PROGRESS_PUBLISH_INTERVAL rows is safe (waiting slightly
        // longer never breaks a dependency) and keeps the counters from
        // becoming a coherence bottleneck of their own.
        #[repr(align(128))]
        struct ProgressSlot(AtomicUsize);
        const PROGRESS_PUBLISH_INTERVAL: usize = 32;
        let progress: Vec<ProgressSlot> = (0..workers)
            .map(|w| {
                let lo = w * slice;
                let hi = ((w + 1) * slice).min(n);
                ProgressSlot(AtomicUsize::new(if reverse { hi } else { lo }))
            })
            .collect();
        // Bounded spin: yields only after a generous cap so an
        // oversubscribed pool cannot starve the worker being awaited.
        let wait_for = |slot: &AtomicUsize, ready: &dyn Fn(usize) -> bool| {
            let mut spins = 0u32;
            while !ready(slot.load(Ordering::Acquire)) {
                std::hint::spin_loop();
                spins += 1;
                if spins >= 1_000_000 {
                    std::thread::yield_now();
                    spins = 0;
                }
            }
        };

        rayon::scope(|scope| {
            for w in 0..workers {
                let progress = &progress;
                let wait_for = &wait_for;
                scope.spawn(move |_| {
                    let lo = w * slice;
                    let hi = ((w + 1) * slice).min(n);
                    // Locally cached frontiers: a remote counter is only
                    // re-read when a dependency reaches past the cached
                    // value (dependencies advance monotonically through
                    // the slice), so satisfied checks stay local. The
                    // confirmation is fused into the gather: each
                    // cross-slice value is waited on immediately before
                    // it is read.
                    if reverse {
                        let mut known: Vec<usize> = (0..workers)
                            .map(|owner| ((owner + 1) * slice).min(n))
                            .collect();
                        for (done, i) in (lo..hi).rev().enumerate() {
                            let mut sum = T::zero();
                            for idx in row_ptrs[i]..row_ptrs[i + 1] {
                                let j = col_indices[idx] as usize;
                                if j >= hi && known[j / slice] > j {
                                    let owner = j / slice;
                                    wait_for(&progress[owner].0, &|frontier| frontier <= j);
                                    known[owner] = progress[owner].0.load(Ordering::Acquire);
                                }
                                sum += values[idx] * shared.get(j);
                            }
                            match diag_inv {
                                None => shared.set(i, shared.get(i) - sum),
                                Some(diag) => shared.set(i, (shared.get(i) - sum) * diag[i]),
                            }
                            if (done + 1) % PROGRESS_PUBLISH_INTERVAL == 0 || done + 1 == hi - lo {
                                progress[w].0.store(i, Ordering::Release);
                            }
                        }
                    } else {
                        let mut known: Vec<usize> =
                            (0..workers).map(|owner| owner * slice).collect();
                        for (done, i) in (lo..hi).enumerate() {
                            let mut sum = T::zero();
                            for idx in row_ptrs[i]..row_ptrs[i + 1] {
                                let j = col_indices[idx] as usize;
                                if j < lo && known[j / slice] <= j {
                                    let owner = j / slice;
                                    wait_for(&progress[owner].0, &|frontier| frontier > j);
                                    known[owner] = progress[owner].0.load(Ordering::Acquire);
                                }
                                sum += values[idx] * shared.get(j);
                            }
                            match diag_inv {
                                None => shared.set(i, shared.get(i) - sum),
                                Some(diag) => shared.set(i, (shared.get(i) - sum) * diag[i]),
                            }
                            if (done + 1) % PROGRESS_PUBLISH_INTERVAL == 0 || done + 1 == hi - lo {
                                progress[w].0.store(i + 1, Ordering::Release);
                            }
                        }
                    }
                });
            }
        });
    }
}

impl<T: ComplexField> IluPreconditioner<T> {
    /// Shared driver for the transpose/Hermitian applies: gather into the
    /// (optional) permuted ordering, run the adjoint factor solve, scatter
    /// back. The permutation is real, so `Mᴴ = Pᵀ (LU)⁻ᴴ P` reuses exactly
    /// the same gather/scatter as the forward apply.
    #[inline]
    fn apply_adjoint_into(&self, r: &[T], y: &mut [T], solve: impl Fn(&Ilu0Factors<T>, &mut [T])) {
        match &self.perm {
            None => {
                y.copy_from_slice(r);
                solve(&self.factors, y);
            }
            Some(perm) => {
                let mut s: Vec<T> = perm.iter().map(|&old| r[old]).collect();
                solve(&self.factors, &mut s);
                for (new_idx, &old_idx) in perm.iter().enumerate() {
                    y[old_idx] = s[new_idx];
                }
            }
        }
    }
}

#[cfg(test)]
mod strict_setup_tests {
    use super::*;

    #[test]
    fn strict_setup_rejects_missing_diagonal() {
        let matrix = CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 1], vec![1], vec![1.0]);
        let error = IluPreconditioner::try_from_csr(&matrix)
            .expect_err("strict ILU setup must reject a missing diagonal");
        assert!(matches!(
            error,
            sonium_solvers_core::traits::SolverError::OperationFailed { .. }
        ));
    }

    #[test]
    fn strict_setup_rejects_present_zero_final_pivot() {
        let matrix = CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![0.0, 1.0]);
        let error = IluPreconditioner::try_from_csr(&matrix)
            .expect_err("strict ILU setup must reject a zero diagonal");
        assert!(matches!(
            error,
            sonium_solvers_core::traits::SolverError::OperationFailed { ref reason, .. }
                if reason.contains("pivots")
        ));
    }

    #[test]
    fn strict_setup_rejects_pivot_zeroed_by_elimination() {
        let matrix = CsrMatrix::from_raw_parts(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0, 1.0, 1.0, 1.0],
        );
        let error = IluPreconditioner::try_from_csr(&matrix)
            .expect_err("strict ILU setup must reject a zero Schur pivot");
        assert!(matches!(
            error,
            sonium_solvers_core::traits::SolverError::OperationFailed { ref reason, .. }
                if reason.contains("pivots")
        ));
    }

    #[test]
    fn strict_setup_rejects_invalid_pivot_tolerance() {
        let matrix = CsrMatrix::<f64>::identity(2);
        let config = IluConfig {
            pivot_tolerance: f64::NAN,
            ..IluConfig::default()
        };
        let error = IluPreconditioner::try_from_csr_with_config(&matrix, &config)
            .expect_err("non-finite pivot tolerance must be rejected");
        assert!(matches!(
            error,
            sonium_solvers_core::traits::SolverError::OperationFailed { ref reason, .. }
                if reason.contains("pivot_tolerance")
        ));
    }

    #[test]
    fn pivot_tolerance_is_reported_in_diagnostics_policy() {
        let matrix = CsrMatrix::from_raw_parts(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0e-6, 1.0, 1.0, 1.0],
        );
        let strict = IluPreconditioner::from_csr(&matrix);
        assert!(strict.diagnostics().skipped_pivots > 0);

        let relaxed = IluPreconditioner::from_csr_with_config(
            &matrix,
            &IluConfig {
                pivot_tolerance: 0.0,
                ..IluConfig::default()
            },
        );
        assert_eq!(relaxed.diagnostics().skipped_pivots, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use sonium_solvers_iterative::{GmresConfig, gmres, gmres_preconditioned};

    #[test]
    fn test_ilu_preconditioner() {
        // Simple tridiagonal matrix
        let dense = sonium_solvers_core::array![
            [
                Complex64::new(4.0, 0.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(0.0, 0.0)
            ],
            [
                Complex64::new(-1.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(-1.0, 0.0)
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(4.0, 0.0)
            ],
        ];

        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let precond = IluPreconditioner::from_csr(&matrix);

        let r = sonium_solvers_core::array![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0)
        ];

        let result = precond.apply(&r);

        // Verify that M*result ≈ r (where M = L*U)
        // For ILU(0) on this matrix, M should be close to A
        let check = matrix.matvec(&result);
        for i in 0..3 {
            // Allow some error since ILU is approximate
            assert!(
                (check[i] - r[i]).norm() < 0.5,
                "ILU should approximately invert: got {:?} expected {:?}",
                check[i],
                r[i]
            );
        }
    }

    #[test]
    fn test_ilu_with_gmres() {
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

        // Solve without preconditioning
        let sol_no_precond = gmres(&matrix, &b, &config);

        // Solve with preconditioning
        let sol_precond = gmres_preconditioned(&matrix, &precond, &b, &config);

        // Both should converge
        assert!(sol_no_precond.converged);
        assert!(sol_precond.converged);

        // Preconditioned should need fewer iterations (usually)
        // For this well-conditioned problem, difference may be small
        assert!(
            sol_precond.iterations <= sol_no_precond.iterations + 5,
            "Preconditioning should not hurt: {} vs {}",
            sol_precond.iterations,
            sol_no_precond.iterations
        );
    }

    #[test]
    fn test_ilu_larger_system() {
        // 50-node pentadiagonal system, verify GMRES converges faster with ILU
        let n = 50;
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        for i in 0..n {
            dense[[i, i]] = Complex64::new(6.0, 0.0);
            if i > 0 {
                dense[[i, i - 1]] = Complex64::new(-1.0, 0.0);
            }
            if i < n - 1 {
                dense[[i, i + 1]] = Complex64::new(-1.0, 0.0);
            }
            if i >= 2 {
                dense[[i, i - 2]] = Complex64::new(-0.5, 0.0);
            }
            if i < n - 2 {
                dense[[i, i + 2]] = Complex64::new(-0.5, 0.0);
            }
        }

        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let precond = IluPreconditioner::from_csr(&matrix);
        let b = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.3).sin(), 0.0))
            .collect::<Vec<_>>();

        let config = GmresConfig {
            max_iterations: 200,
            restart: 20,
            tolerance: 1e-10,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let sol_no_precond = gmres(&matrix, &b, &config);
        let sol_precond = gmres_preconditioned(&matrix, &precond, &b, &config);

        assert!(
            sol_no_precond.converged,
            "Unpreconditioned GMRES should converge"
        );
        assert!(
            sol_precond.converged,
            "ILU-preconditioned GMRES should converge"
        );

        // ILU should help or at least not significantly hurt
        assert!(
            sol_precond.iterations <= sol_no_precond.iterations + 5,
            "ILU preconditioning should not hurt: {} vs {} iterations",
            sol_precond.iterations,
            sol_no_precond.iterations
        );
    }

    #[test]
    fn test_ilu_diagonal_matrix() {
        // Diagonal matrix: ILU should give exact inverse on the diagonal
        let n = 5;
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        for i in 0..n {
            dense[[i, i]] = Complex64::new((i + 1) as f64, 0.0);
        }

        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let precond = IluPreconditioner::from_csr(&matrix);

        // Applying preconditioner to a vector should give element-wise division
        let r = sonium_solvers_core::array![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
        ];
        let result = precond.apply(&r);

        // Each result[i] should be r[i] / diag[i] = i / (i+1) ... wait: r[i]/(i+1)
        for i in 0..n {
            let expected = (i + 1) as f64 / (i + 1) as f64; // r[i] = i+1, diag = i+1 => 1.0
            approx::assert_relative_eq!(result[i].re, expected, epsilon = 1e-10);
            approx::assert_relative_eq!(result[i].im, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_ilu_nearly_zero_pivot_complex() {
        // Regression test for the near-zero pivot issue: indefinite matrices
        // (Helmholtz, Maxwell) can have near-zero or purely imaginary diagonals.
        // The old shift strategy (add 1e-21 to a 1e-20 pivot) was numerically
        // meaningless, producing garbage multipliers that destroyed the preconditioner.
        // The new relative drop threshold gracefully skips such rows.
        let n = 5;
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));

        // Diagonal: [well-conditioned, purely imaginary 1e-10, well-conditioned,
        //            near-zero real 1e-15, well-conditioned]
        dense[[0, 0]] = Complex64::new(4.0, 0.0);
        dense[[1, 1]] = Complex64::new(0.0, 1e-10); // Purely imaginary — old shift was useless
        dense[[2, 2]] = Complex64::new(5.0, 0.0);
        dense[[3, 3]] = Complex64::new(1e-15, 0.0); // Near-zero real
        dense[[4, 4]] = Complex64::new(4.0, 0.0);

        // Off-diagonals to make it non-trivial
        for i in 0..n - 1 {
            dense[[i, i + 1]] = Complex64::new(-0.5, 0.0);
            dense[[i + 1, i]] = Complex64::new(-0.5, 0.0);
        }

        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let precond = IluPreconditioner::from_csr(&matrix);

        let b = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();
        let result = precond.apply(&b);

        // The preconditioner should produce finite results without NaN/inf
        for i in 0..n {
            assert!(
                result[i].norm().is_finite(),
                "Result at row {} is not finite: {:?}",
                i,
                result[i]
            );
        }
    }

    #[test]
    fn test_ilu_zero_pivot_preserves_complex_phase() {
        // Regression test for issue #11: near-zero pivot fallback must preserve
        // the complex phase, not replace the entry with a purely real inverse.
        //
        // Old code: T::from_real(fallback).inv()  ->  purely real, wrong phase
        // New code: d.conj() * T::from_real(scale)  ->  preserves phase of d
        //
        // Use a purely diagonal matrix so apply() is just element-wise scaling
        // and we can directly verify the phase of the inverse diagonal entry.
        let n = 3;
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        dense[[0, 0]] = Complex64::new(4.0, 0.0);
        // Near-zero purely imaginary diagonal -- phase MUST be preserved
        dense[[1, 1]] = Complex64::new(0.0, 1e-12);
        dense[[2, 2]] = Complex64::new(5.0, 0.0);

        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let precond = IluPreconditioner::from_csr(&matrix);

        // Apply to e_1 (the unit vector hitting the problematic row)
        let b = sonium_solvers_core::array![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let result = precond.apply(&b);

        // For a diagonal matrix, result[1] == b[1] * (u_diag[1])^{-1}
        // u_diag[1] = 0 + i*1e-12, so inverse should be 0 - i*1e12
        let r1 = result[1];
        assert!(
            r1.norm().is_finite(),
            "Preconditioner produced non-finite value: {:?}",
            r1
        );
        // Phase preservation: if d = i*epsilon, the inverse should be ~ -i/epsilon,
        // so the real part should be near zero and imag part large and negative.
        assert!(
            r1.re.abs() < 1.0,
            "Real part too large -- phase lost? re={}, im={}",
            r1.re,
            r1.im
        );
        assert!(
            r1.im.abs() > 1e10,
            "Imag part too small -- phase lost? re={}, im={}",
            r1.re,
            r1.im
        );
    }

    #[test]
    fn test_ilu_indefinite_helmholtz_like() {
        // Simulate a small Helmholtz-like system where diagonal entries can be
        // small relative to off-diagonals (k² → large, diagonal becomes small).
        let n = 10;
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));

        for i in 0..n {
            // Diagonal: oscillating real part near zero, significant imaginary part
            let re = (i as f64 * 0.5).sin() * 0.01; // ~0 range
            let im = (i as f64 * 0.3).sin();
            dense[[i, i]] = Complex64::new(re, im);

            if i > 0 {
                dense[[i, i - 1]] = Complex64::new(-1.0, 0.0);
            }
            if i < n - 1 {
                dense[[i, i + 1]] = Complex64::new(-1.0, 0.0);
            }
        }

        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let precond = IluPreconditioner::from_csr(&matrix);

        let b = vec![Complex64::new(1.0, 0.0); n];
        let result = precond.apply(&b);

        // All results should be finite and non-zero
        assert!(
            result.iter().all(|v| v.norm().is_finite()),
            "Preconditioner produced non-finite result"
        );
        assert!(
            result.iter().any(|v| v.norm() > 1e-10),
            "Preconditioner produced all-zero result"
        );

        // With the preconditioned system, GMRES should handle it
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

        // Even if the system doesn't converge (it's nearly singular), the
        // preconditioner itself should not have produced garbage
        for v in sol.x.iter() {
            assert!(v.norm().is_finite(), "Solution contains non-finite value");
        }
    }

    #[test]
    fn test_ilu_colamd_reordering_arrowhead() {
        // Arrowhead matrix: node 0 (the hub) is coupled to every leaf.
        // In the natural ordering, eliminating the hub first would fill the
        // entire leaf block; ILU(0) drops all of that fill, so M = L*U is a
        // poor approximation of A. A fill-reducing ordering places the hub
        // last, where the exact LU has zero fill and ILU(0) becomes exact.
        //
        // Measured on n = 40 (hub diag = n, leaf diag = 2, coupling = -1):
        //   natural ordering:   ||A M^{-1} b - b|| ≈ 2.0e-1
        //   COLAMD reordering:  ||A M^{-1} b - b|| ≈ 5.2e-16 (machine precision)
        // (ILU(0) has fixed storage fill; what the reordering reduces is the
        // dropped fill, i.e. the factorization error. See the ILUT test in
        // ilut.rs for a storage-fill measurement.)
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

        let natural = IluPreconditioner::from_csr(&matrix);
        let reordered = IluPreconditioner::from_csr_with_config(
            &matrix,
            &IluConfig {
                reordering: IluReordering::Colamd,
                ..IluConfig::default()
            },
        );

        let b: Vec<_> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.7).sin(), 0.0))
            .collect();

        let factorization_error = |pc: &IluPreconditioner<Complex64>| {
            let x = pc.apply(&b);
            let r = sonium_solvers_core::vec_ops::vec_sub(&matrix.matvec(&x), &b);
            r.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt()
        };
        let err_natural = factorization_error(&natural);
        let err_reordered = factorization_error(&reordered);
        eprintln!(
            "ILU(0) arrowhead n={n}: factorization error natural={err_natural:.3e}, \
             colamd-reordered={err_reordered:.3e}"
        );

        // With the hub ordered last, ILU(0) is the exact factorization.
        assert!(
            err_reordered < 1e-10,
            "reordered ILU(0) should be exact on the arrowhead, got {err_reordered:.3e}"
        );
        assert!(
            err_natural > 1e3 * err_reordered.max(1e-300),
            "natural ordering should have much larger factorization error: \
             natural={err_natural:.3e}, reordered={err_reordered:.3e}"
        );

        // The permutation must be a genuine permutation of 0..n.
        let perm = reordered.perm.as_ref().expect("reordering stores a perm");
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n).collect::<Vec<_>>());

        // apply_into must scatter back to the original ordering (match apply).
        let mut y = vec![Complex64::new(0.0, 0.0); n];
        reordered.apply_into(&b, &mut y);
        let x = reordered.apply(&b);
        for i in 0..n {
            assert!(
                (x[i] - y[i]).norm() < 1e-14,
                "apply/apply_into mismatch at {i}"
            );
        }

        // GMRES with the exact (reordered) preconditioner converges at once.
        let config = GmresConfig {
            max_iterations: 50,
            restart: 10,
            tolerance: 1e-12,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };
        let sol = gmres_preconditioned(&matrix, &reordered, &b, &config);
        assert!(sol.converged, "reordered ILU(0) GMRES should converge");
        assert!(
            sol.iterations <= 2,
            "exact preconditioner should converge in <=2 iterations, got {}",
            sol.iterations
        );
    }

    #[test]
    fn test_ilu_reordering_none_matches_default() {
        // IluReordering::None must be bit-identical to the default path.
        let matrix = {
            let n = 8;
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
            CsrMatrix::from_dense(&dense, 1e-15)
        };
        let default = IluPreconditioner::from_csr(&matrix);
        let explicit = IluPreconditioner::from_csr_with_reordering(&matrix, IluReordering::None);
        assert!(explicit.perm.is_none());

        let b: Vec<_> = (0..8).map(|i| Complex64::new(i as f64, 0.0)).collect();
        let x_default = default.apply(&b);
        let x_explicit = explicit.apply(&b);
        for i in 0..8 {
            assert_eq!(x_default[i], x_explicit[i]);
        }
    }

    /// Build a small well-conditioned non-Hermitian complex matrix: no
    /// pivot fallback triggers, and the phases are non-aligned so any
    /// conjugation mistake is visible.
    fn make_adjoint_test_matrix(n: usize) -> sonium_solvers_core::dense::DenseMatrix<Complex64> {
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        for i in 0..n {
            dense[[i, i]] = Complex64::new(4.0 + 0.1 * i as f64, 0.3);
            if i > 0 {
                dense[[i, i - 1]] = Complex64::new(-1.0, 0.2 * (i as f64).sin());
            }
            if i < n - 1 {
                dense[[i, i + 1]] = Complex64::new(-0.8, -0.1 * (i as f64).cos());
            }
            if i + 2 < n {
                // Extra fill candidate so ILU(0) is *not* the exact
                // factorization — the reference is M = L·U, not A.
                dense[[i, i + 2]] = Complex64::new(0.3, 0.15);
            }
            if i >= 2 {
                dense[[i, i - 2]] = Complex64::new(-0.2, 0.05);
            }
        }
        dense
    }

    /// Reconstruct the dense `M = L·U` implied by the stored ILU(0)
    /// factors. Requires a well-conditioned matrix (the `u_diag_inv`
    /// fallback must not have triggered, so `u_diag = 1/u_diag_inv`).
    fn dense_factors_m(
        precond: &IluPreconditioner<Complex64>,
        n: usize,
    ) -> sonium_solvers_core::dense::DenseMatrix<Complex64> {
        let f = &precond.factors;
        let mut l = sonium_solvers_core::dense::DenseMatrix::<Complex64>::eye(n);
        for i in 0..n {
            for idx in f.l_row_ptrs[i]..f.l_row_ptrs[i + 1] {
                l[[i, f.l_col_indices[idx] as usize]] = f.l_values[idx];
            }
        }
        let mut u = sonium_solvers_core::dense::DenseMatrix::<Complex64>::zeros(n, n);
        for i in 0..n {
            u[[i, i]] = f.u_diag_inv[i].inv();
            for idx in f.u_row_ptrs[i]..f.u_row_ptrs[i + 1] {
                u[[i, f.u_col_indices[idx] as usize]] = f.u_values[idx];
            }
        }
        l.matmul(&u)
    }

    fn assert_vec_close(actual: &[Complex64], expected: &[Complex64], tol: f64, what: &str) {
        let diff = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let reference = expected
            .iter()
            .map(|v| v.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(1e-30);
        assert!(
            diff / reference < tol,
            "{what}: relative error {:.3e} exceeds {tol:.1e}",
            diff / reference
        );
    }

    #[test]
    fn test_ilu_apply_hermitian_and_transpose_vs_explicit() {
        let n = 6;
        let dense = make_adjoint_test_matrix(n);
        let matrix = CsrMatrix::from_dense(&dense, 1e-15);

        let r: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((0.7 * (i + 1) as f64).sin(), (0.3 * (i + 1) as f64).cos()))
            .collect();

        for precond in [
            IluPreconditioner::from_csr(&matrix),
            IluPreconditioner::from_csr_with_reordering(&matrix, IluReordering::Colamd),
        ] {
            // Effective operator the preconditioner inverts, in original
            // DOF ordering: M_eff = Pᵀ (L U) P (identity permutation when
            // no reordering). Note P is real, so M_effᴴ = Pᵀ (LU)ᴴ P.
            let m_perm = dense_factors_m(&precond, n);
            let mut m_eff = sonium_solvers_core::dense::DenseMatrix::<Complex64>::zeros(n, n);
            match &precond.perm {
                None => m_eff = m_perm,
                Some(perm) => {
                    let mut inv_perm = vec![0usize; n];
                    for (new_idx, &old_idx) in perm.iter().enumerate() {
                        inv_perm[old_idx] = new_idx;
                    }
                    for a in 0..n {
                        for b in 0..n {
                            m_eff[[a, b]] = m_perm[[inv_perm[a], inv_perm[b]]];
                        }
                    }
                }
            }

            // Hermitian: reference is the dense solve of M_effᴴ y = r.
            let lu = sonium_solvers_direct::lu_factorize(&m_eff).expect("LU of M_eff");
            let expected_h = lu.solve_conj_transpose(&r).expect("solve M_effᴴ");
            let got_h = precond.apply_hermitian(&r);
            assert_vec_close(&got_h, &expected_h, 1e-10, "apply_hermitian vs M⁻ᴴ");

            // Transpose: reference is the dense solve of M_effᵀ y = r.
            let expected_t = lu.solve_transpose(&r).expect("solve M_effᵀ");
            let got_t = precond.apply_transpose(&r);
            assert_vec_close(&got_t, &expected_t, 1e-10, "apply_transpose vs M⁻ᵀ");

            // _into variants agree with the allocating ones.
            let mut got_h_into = vec![Complex64::new(0.0, 0.0); n];
            precond.apply_hermitian_into(&r, &mut got_h_into);
            assert_vec_close(&got_h_into, &got_h, 1e-14, "apply_hermitian_into");

            // Algebraic identity: M⁻ᴴ r == conj(M⁻ᵀ conj(r)).
            let r_conj: Vec<Complex64> = r.iter().map(|v| v.conj()).collect();
            let mut via_transpose = precond.apply_transpose(&r_conj);
            for v in via_transpose.iter_mut() {
                *v = v.conj();
            }
            assert_vec_close(
                &got_h,
                &via_transpose,
                1e-12,
                "hermitian == conj(transpose(conj))",
            );
        }
    }

    #[test]
    fn test_ilu_hermitian_exact_on_tridiagonal() {
        // ILU(0) is the exact factorization of a tridiagonal matrix, so
        // apply_hermitian must recover the dense solve of Aᴴ x = r.
        let n = 8;
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        for i in 0..n {
            dense[[i, i]] = Complex64::new(4.0, 0.5);
            if i > 0 {
                dense[[i, i - 1]] = Complex64::new(-1.0, 0.2);
            }
            if i < n - 1 {
                dense[[i, i + 1]] = Complex64::new(-1.2, -0.1);
            }
        }
        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let precond = IluPreconditioner::from_csr(&matrix);

        let r: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), (0.5 * i as f64).cos()))
            .collect();

        // Reference: solve Aᴴ x = r by dense LU on the conj-transposed matrix.
        let mut ah = sonium_solvers_core::dense::DenseMatrix::<Complex64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                ah[[i, j]] = dense[[j, i]].conj();
            }
        }
        let lu = sonium_solvers_direct::lu_factorize(&ah).expect("LU of Aᴴ");
        let expected = lu.solve(&r).expect("dense solve of Aᴴ x = r");

        let got = precond.apply_hermitian(&r);
        assert_vec_close(&got, &expected, 1e-8, "exact ILUᴴ vs dense Aᴴ solve");
    }

    /// Toy Stokes saddle system on an `nx` by `ny` grid: velocity unknown
    /// per node (5-point diffusion) plus one pressure per node, with a
    /// discrete gradient coupling. Mirrors the NS cavity sparsity (wide
    /// dependency graph, empty pressure diagonal block).
    fn saddle_test_matrix(nx: usize, ny: usize) -> CsrMatrix<f64> {
        let nodes = nx * ny;
        let n = 2 * nodes;
        let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        let node = |i: usize, j: usize| j * nx + i;
        for j in 0..ny {
            for i in 0..nx {
                let u = node(i, j);
                let p = nodes + u;
                rows[u].push((u, 4.0));
                if i > 0 {
                    rows[u].push((node(i - 1, j), -1.0));
                    rows[u].push((p, 0.5));
                    rows[p].push((node(i - 1, j), -0.5));
                }
                if i + 1 < nx {
                    rows[u].push((node(i + 1, j), -1.0));
                    rows[u].push((p, -0.5));
                    rows[p].push((node(i + 1, j), 0.5));
                }
                if j > 0 {
                    rows[u].push((node(i, j - 1), -1.0));
                }
                if j + 1 < ny {
                    rows[u].push((node(i, j + 1), -1.0));
                }
                rows[p].push((p, 0.0));
                rows[u].sort_by_key(|&(c, _)| c);
                rows[p].sort_by_key(|&(c, _)| c);
            }
        }
        let mut row_ptrs = Vec::with_capacity(n + 1);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        row_ptrs.push(0);
        for row in &rows {
            for &(c, v) in row {
                col_indices.push(c);
                values.push(v);
            }
            row_ptrs.push(col_indices.len());
        }
        CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values)
    }

    fn saddle_rhs(n: usize) -> Vec<f64> {
        // Deterministic xorshift values in [-1, 1).
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % 2000) as f64 / 1000.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn level_schedule_respects_dependencies() {
        let matrix = saddle_test_matrix(16, 16);
        let precond = IluPreconditioner::from_csr(&matrix);
        let factors = &precond.factors;
        let level_of = |col_indices: &[u32], row_ptrs: &[usize], reverse: bool| {
            let (rows, starts) = compute_level_schedule(col_indices, row_ptrs, factors.n, reverse);
            // The schedule covers every row exactly once.
            assert_eq!(rows.len(), factors.n);
            let mut level = vec![usize::MAX; rows.len()];
            for (l, w) in starts.windows(2).enumerate() {
                for &row in &rows[w[0]..w[1]] {
                    level[row] = l;
                }
            }
            level
        };
        let fwd = level_of(&factors.l_col_indices, &factors.l_row_ptrs, false);
        for i in 0..factors.n {
            for &j in &factors.l_col_indices[factors.l_row_ptrs[i]..factors.l_row_ptrs[i + 1]] {
                assert!(
                    fwd[j as usize] < fwd[i],
                    "forward dependency {j} -> {i} not in an earlier level"
                );
            }
        }
        let bwd = level_of(&factors.u_col_indices, &factors.u_row_ptrs, true);
        for i in 0..factors.n {
            for &j in &factors.u_col_indices[factors.u_row_ptrs[i]..factors.u_row_ptrs[i + 1]] {
                assert!(
                    bwd[j as usize] < bwd[i],
                    "backward dependency {j} -> {i} not in an earlier level"
                );
            }
        }
    }

    #[test]
    fn level_wavefront_apply_is_bit_identical_across_thread_counts() {
        let matrix = saddle_test_matrix(48, 48);
        let precond = IluPreconditioner::from_csr(&matrix);
        let rhs = saddle_rhs(matrix.num_rows);
        eprintln!(
            "saddle 48x48: n={}, forward levels={}, backward levels={}",
            matrix.num_rows,
            precond.forward_level_starts.len() - 1,
            precond.backward_level_starts.len() - 1,
        );
        // Reference: the serial factor kernel.
        let mut serial = rhs.clone();
        precond.factors.solve_in_place(&mut serial);
        for threads in [2usize, 4, 8] {
            // Drive the wavefront directly so the heuristic gate cannot
            // silently substitute the serial path.
            let mut parallel = rhs.clone();
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("worker pool")
                .install(|| precond.solve_level_wavefront(&mut parallel));
            assert_eq!(parallel.len(), serial.len());
            for (index, (got, want)) in parallel.iter().zip(&serial).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "wavefront result differs from serial at index {index} ({threads} threads)"
                );
            }
        }
        // The public apply must agree with the serial kernel at one thread.
        let applied = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("one-thread pool")
            .install(|| precond.apply(&rhs));
        for (index, (got, want)) in applied.iter().zip(&serial).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "public apply differs from serial at index {index}"
            );
        }
    }
}
