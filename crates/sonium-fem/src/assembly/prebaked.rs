//! Pre-baked CSR assembler for efficient parametric re-assembly
//!
//! Stores N component matrices sharing a single CSR sparsity pattern.
//! Enables O(nnz) re-assembly as a linear combination of components —
//! critical for frequency sweeps (acoustics, Maxwell) where only coefficients change.
//!
//! Generalizes the original `HelmholtzAssembler` (which was `PrebakedAssembler<2>`
//! with components = [stiffness, mass]).

use crate::parallelism::{POINTWISE_VALUES_PER_THREAD, collect_indexed};
use num_complex::Complex64;
use sonium_solvers::CsrMatrix;
use std::collections::HashMap;

use super::triplet::TripletMatrix;

/// Validation failure returned by PrebakedAssembler::try_from_triplets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrebakedAssemblyError {
    /// A component or boundary matrix is not square.
    NonSquare {
        /// Matrix label (component-N or boundary-TAG).
        matrix: String,
        /// Number of rows supplied.
        rows: usize,
        /// Number of columns supplied.
        cols: usize,
    },
    /// Components do not use one common square dimension.
    DimensionMismatch {
        /// Matrix label.
        matrix: String,
        /// Expected number of rows/columns.
        expected: usize,
        /// Supplied number of rows/columns.
        actual_rows: usize,
        /// Supplied number of columns.
        actual_cols: usize,
    },
    /// The triplet storage vectors have inconsistent lengths.
    StorageLengthMismatch {
        /// Matrix label.
        matrix: String,
        /// Number of rows/columns stored.
        rows: usize,
        /// Number of values stored.
        values: usize,
    },
    /// A triplet references a row or column outside the matrix.
    IndexOutOfBounds {
        /// Matrix label.
        matrix: String,
        /// Triplet position.
        triplet: usize,
        /// Row index.
        row: usize,
        /// Column index.
        col: usize,
        /// Matrix dimension.
        dimension: usize,
    },
    /// A matrix contains a NaN or an infinity.
    NonFiniteValue {
        /// Matrix label.
        matrix: String,
        /// Triplet position.
        triplet: usize,
    },
    /// A boundary tag occurs more than once.
    DuplicateBoundaryTag {
        /// Repeated boundary marker.
        tag: usize,
    },
}

impl std::fmt::Display for PrebakedAssemblyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonSquare { matrix, rows, cols } => {
                write!(formatter, "{matrix} is not square ({rows}x{cols})")
            }
            Self::DimensionMismatch {
                matrix,
                expected,
                actual_rows,
                actual_cols,
            } => write!(
                formatter,
                "{matrix} has dimensions {actual_rows}x{actual_cols}, expected {expected}x{expected}"
            ),
            Self::StorageLengthMismatch {
                matrix,
                rows,
                values,
            } => write!(
                formatter,
                "{matrix} stores {rows} row/column entries but {values} values"
            ),
            Self::IndexOutOfBounds {
                matrix,
                triplet,
                row,
                col,
                dimension,
            } => write!(
                formatter,
                "{matrix} triplet {triplet} ({row}, {col}) exceeds dimension {dimension}"
            ),
            Self::NonFiniteValue { matrix, triplet } => {
                write!(formatter, "{matrix} triplet {triplet} is non-finite")
            }
            Self::DuplicateBoundaryTag { tag } => {
                write!(formatter, "boundary tag {tag} occurs more than once")
            }
        }
    }
}

impl std::error::Error for PrebakedAssemblyError {}

fn validate_triplet_matrix(
    matrix: &TripletMatrix<f64>,
    label: &str,
    expected: usize,
) -> Result<(), PrebakedAssemblyError> {
    if matrix.num_rows != matrix.num_cols {
        return Err(PrebakedAssemblyError::NonSquare {
            matrix: label.to_string(),
            rows: matrix.num_rows,
            cols: matrix.num_cols,
        });
    }
    if matrix.num_rows != expected || matrix.num_cols != expected {
        return Err(PrebakedAssemblyError::DimensionMismatch {
            matrix: label.to_string(),
            expected,
            actual_rows: matrix.num_rows,
            actual_cols: matrix.num_cols,
        });
    }
    if matrix.rows.len() != matrix.cols.len() || matrix.rows.len() != matrix.values.len() {
        return Err(PrebakedAssemblyError::StorageLengthMismatch {
            matrix: label.to_string(),
            rows: matrix.rows.len().min(matrix.cols.len()),
            values: matrix.values.len(),
        });
    }
    for (triplet, ((&row, &col), &value)) in matrix
        .rows
        .iter()
        .zip(&matrix.cols)
        .zip(&matrix.values)
        .enumerate()
    {
        if row >= expected || col >= expected {
            return Err(PrebakedAssemblyError::IndexOutOfBounds {
                matrix: label.to_string(),
                triplet,
                row,
                col,
                dimension: expected,
            });
        }
        if !value.is_finite() {
            return Err(PrebakedAssemblyError::NonFiniteValue {
                matrix: label.to_string(),
                triplet,
            });
        }
    }
    Ok(())
}

/// Pre-baked CSR assembler storing `N` component matrices on a shared sparsity pattern.
///
/// # Examples
///
/// **Helmholtz** (N=2): components = [K, M], result = K - k²M
/// **Maxwell curl-curl** (N=2): components = [curl_curl, mass], result = curl_curl - ω²εM
/// **Elasticity** (N=1): single stiffness matrix (no frequency sweep needed)
pub struct PrebakedAssembler<const N: usize> {
    /// Number of rows (= number of columns, square system)
    pub num_rows: usize,
    /// CSR row pointers (shared across all components)
    pub row_ptrs: Vec<usize>,
    /// CSR column indices (shared across all components)
    pub col_indices: Vec<usize>,
    /// One value array per component, all aligned to the shared CSR pattern
    pub components: [Vec<f64>; N],
    /// Tagged boundary matrices (same CSR pattern, keyed by boundary marker)
    pub boundary_components: HashMap<usize, Vec<f64>>,
}

impl<const N: usize> PrebakedAssembler<N> {
    /// Fallibly build a pre-baked assembler from component and boundary triplets.
    ///
    /// The legacy from_triplets constructor predates checked FEM assembly and
    /// remains available for compatibility. New callers should use this
    /// method so malformed deserialized or generated matrices are rejected
    /// before the CSR merge.
    pub fn try_from_triplets(
        triplets: [TripletMatrix<f64>; N],
        boundary_triplets: &[(usize, TripletMatrix<f64>)],
    ) -> Result<Self, PrebakedAssemblyError> {
        if N == 0 {
            return Err(PrebakedAssemblyError::DimensionMismatch {
                matrix: "components".to_string(),
                expected: 1,
                actual_rows: 0,
                actual_cols: 0,
            });
        }

        let expected = triplets[0].num_rows;
        for (index, matrix) in triplets.iter().enumerate() {
            validate_triplet_matrix(matrix, &format!("component-{index}"), expected)?;
        }
        for (tag, matrix) in boundary_triplets {
            validate_triplet_matrix(matrix, &format!("boundary-{tag}"), expected)?;
        }

        let mut tags = std::collections::HashSet::with_capacity(boundary_triplets.len());
        for (tag, _) in boundary_triplets {
            if !tags.insert(*tag) {
                return Err(PrebakedAssemblyError::DuplicateBoundaryTag { tag: *tag });
            }
        }

        Ok(Self::from_triplets(triplets, boundary_triplets))
    }

    /// Build from N `TripletMatrix<f64>` that share the same DOF space.
    ///
    /// The triplet matrices may have different sparsity patterns; the assembler
    /// merges them into a single CSR pattern with separate value arrays.
    pub fn from_triplets(
        triplets: [TripletMatrix<f64>; N],
        boundary_triplets: &[(usize, TripletMatrix<f64>)],
    ) -> Self {
        let num_rows = triplets[0].num_rows;
        debug_assert!(triplets.iter().all(|t| t.num_rows == num_rows));

        // Collect all (row, col) pairs with source identification
        struct Entry {
            row: usize,
            col: usize,
            source: i32, // 0..N-1 for components, N+tag for boundaries
            value: f64,
        }

        let total_nnz: usize = triplets.iter().map(|t| t.nnz()).sum::<usize>()
            + boundary_triplets
                .iter()
                .map(|(_, t)| t.nnz())
                .sum::<usize>();

        let mut entries = Vec::with_capacity(total_nnz);

        for (source_idx, triplet) in triplets.iter().enumerate() {
            for i in 0..triplet.nnz() {
                entries.push(Entry {
                    row: triplet.rows[i],
                    col: triplet.cols[i],
                    source: source_idx as i32,
                    value: triplet.values[i],
                });
            }
        }

        let boundary_tags: Vec<usize> = boundary_triplets.iter().map(|(tag, _)| *tag).collect();
        for (bt_idx, (_, triplet)) in boundary_triplets.iter().enumerate() {
            for i in 0..triplet.nnz() {
                entries.push(Entry {
                    row: triplet.rows[i],
                    col: triplet.cols[i],
                    source: (N + bt_idx) as i32,
                    value: triplet.values[i],
                });
            }
        }

        // Sort by (row, col)
        entries.sort_unstable_by(|a, b| a.row.cmp(&b.row).then(a.col.cmp(&b.col)));

        // Build CSR structure with merged sparsity
        let mut row_ptrs = vec![0usize; num_rows + 1];
        let mut col_indices = Vec::with_capacity(total_nnz);
        let mut comp_values: [Vec<f64>; N] = std::array::from_fn(|_| Vec::with_capacity(total_nnz));
        let mut bound_values: Vec<Vec<f64>> = (0..boundary_tags.len())
            .map(|_| Vec::with_capacity(total_nnz / 4))
            .collect();

        let mut prev_row = usize::MAX;
        let mut prev_col = usize::MAX;

        for entry in &entries {
            if entry.row != prev_row || entry.col != prev_col {
                // New structural position
                col_indices.push(entry.col);
                for cv in &mut comp_values {
                    cv.push(0.0);
                }
                for bv in &mut bound_values {
                    bv.push(0.0);
                }
                // Fill row_ptrs
                if prev_row != usize::MAX {
                    for rp in &mut row_ptrs[(prev_row + 1)..=(entry.row)] {
                        *rp = col_indices.len() - 1;
                    }
                } else {
                    for rp in &mut row_ptrs[..=(entry.row)] {
                        *rp = 0;
                    }
                }
                prev_row = entry.row;
                prev_col = entry.col;
            }

            // Accumulate value into the right component
            let pos = col_indices.len() - 1;
            let src = entry.source;
            if src >= 0 && (src as usize) < N {
                comp_values[src as usize][pos] += entry.value;
            } else if src >= N as i32 {
                let bt_idx = (src as usize) - N;
                bound_values[bt_idx][pos] += entry.value;
            }
        }

        // Fill remaining row_ptrs
        let final_nnz = col_indices.len();
        if prev_row != usize::MAX {
            for rp in &mut row_ptrs[(prev_row + 1)..] {
                *rp = final_nnz;
            }
        }

        let boundary_components: HashMap<usize, Vec<f64>> =
            boundary_tags.into_iter().zip(bound_values).collect();

        Self {
            num_rows,
            row_ptrs,
            col_indices,
            components: comp_values,
            boundary_components,
        }
    }

    /// Number of non-zeros in the merged CSR pattern.
    pub fn nnz(&self) -> usize {
        self.col_indices.len()
    }

    /// Assemble with complex coefficients: `result[i] = sum_k coeffs[k] * component_k[i]`
    /// plus boundary terms.
    pub fn assemble_complex(
        &self,
        coeffs: &[Complex64; N],
        boundary_coeffs: &HashMap<usize, Complex64>,
    ) -> CsrMatrix<Complex64> {
        let mut out = CsrMatrix::from_raw_parts(
            self.num_rows,
            self.num_rows,
            self.row_ptrs.clone(),
            self.col_indices.clone(),
            vec![Complex64::new(0.0, 0.0); self.nnz()],
        );
        self.assemble_complex_into(coeffs, boundary_coeffs, &mut out);
        out
    }

    /// Assemble with complex coefficients into an existing CSR matrix.
    ///
    /// See [`Self::assemble_real_into`] for the buffer-reuse contract.
    ///
    /// # Panics
    /// Panics when `out` does not carry this assembler's sparsity pattern.
    pub fn assemble_complex_into(
        &self,
        coeffs: &[Complex64; N],
        boundary_coeffs: &HashMap<usize, Complex64>,
        out: &mut CsrMatrix<Complex64>,
    ) {
        assert_eq!(out.num_rows, self.num_rows, "prebaked row count mismatch");
        assert_eq!(
            out.num_cols, self.num_rows,
            "prebaked column count mismatch"
        );
        assert_eq!(out.row_ptrs, self.row_ptrs, "prebaked row pattern mismatch");
        assert_eq!(
            out.col_indices, self.col_indices,
            "prebaked column pattern mismatch"
        );
        let values: Vec<Complex64> =
            collect_indexed(self.nnz(), POINTWISE_VALUES_PER_THREAD, |i| {
                let mut val = Complex64::new(0.0, 0.0);
                for (k, coeff) in coeffs.iter().enumerate() {
                    val += *coeff * self.components[k][i];
                }
                for (&tag, &coeff) in boundary_coeffs {
                    if let Some(bv) = self.boundary_components.get(&tag) {
                        val += coeff * bv[i];
                    }
                }
                val
            });
        out.values.copy_from_slice(&values);
    }

    /// Assemble with real coefficients: `result[i] = sum_k coeffs[k] * component_k[i]`.
    pub fn assemble_real(&self, coeffs: &[f64; N]) -> CsrMatrix<f64> {
        let mut out = CsrMatrix::from_raw_parts(
            self.num_rows,
            self.num_rows,
            self.row_ptrs.clone(),
            self.col_indices.clone(),
            vec![0.0; self.nnz()],
        );
        self.assemble_real_into(coeffs, &mut out);
        out
    }

    /// Assemble with real coefficients into an existing CSR matrix.
    ///
    /// Frequency sweeps reuse the caller's CSR buffers: the shared
    /// sparsity (`row_ptrs`/`col_indices`) is neither cloned nor
    /// reallocated, only the values are recomputed in place.
    ///
    /// # Panics
    /// Panics when `out` does not carry this assembler's sparsity pattern.
    pub fn assemble_real_into(&self, coeffs: &[f64; N], out: &mut CsrMatrix<f64>) {
        assert_eq!(out.num_rows, self.num_rows, "prebaked row count mismatch");
        assert_eq!(
            out.num_cols, self.num_rows,
            "prebaked column count mismatch"
        );
        assert_eq!(out.row_ptrs, self.row_ptrs, "prebaked row pattern mismatch");
        assert_eq!(
            out.col_indices, self.col_indices,
            "prebaked column pattern mismatch"
        );
        let values = collect_indexed(self.nnz(), POINTWISE_VALUES_PER_THREAD, |i| {
            let mut val = 0.0;
            for (k, coeff) in coeffs.iter().enumerate() {
                val += *coeff * self.components[k][i];
            }
            val
        });
        out.values.copy_from_slice(&values);
    }

    /// Apply Dirichlet boundary conditions by zeroing rows AND columns of
    /// constrained DOFs, setting the diagonal to 1 in component 0.
    ///
    /// This preserves matrix symmetry: both `A[i,j]` and `A[j,i]` are zeroed
    /// when `i` is a Dirichlet node. The diagonal `A[i,i]` is set to 1 in
    /// component 0 and 0 in all other components.
    pub fn apply_dirichlet_nodes(&mut self, nodes: &std::collections::HashSet<usize>) {
        if nodes.is_empty() {
            return;
        }

        for row in 0..self.num_rows {
            let row_start = self.row_ptrs[row];
            let row_end = self.row_ptrs[row + 1];
            let is_constrained_row = nodes.contains(&row);

            for pos in row_start..row_end {
                let col = self.col_indices[pos];
                let is_constrained_col = nodes.contains(&col);

                if is_constrained_row {
                    // Zero entire row, set diagonal to identity in component 0
                    let is_diag = col == row;
                    for k in 0..N {
                        self.components[k][pos] = if is_diag && k == 0 { 1.0 } else { 0.0 };
                    }
                    for bv in self.boundary_components.values_mut() {
                        bv[pos] = 0.0;
                    }
                } else if is_constrained_col {
                    // Zero column entries in non-constrained rows
                    for k in 0..N {
                        self.components[k][pos] = 0.0;
                    }
                    for bv in self.boundary_components.values_mut() {
                        bv[pos] = 0.0;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly::TripletMatrix;

    /// Build a 3-DOF Laplacian: K = [[1,-1,0],[-1,2,-1],[0,-1,1]]
    fn laplacian_3dof() -> PrebakedAssembler<2> {
        let mut k = TripletMatrix::new(3, 3);
        k.push(0, 0, 1.0);
        k.push(0, 1, -1.0);
        k.push(1, 0, -1.0);
        k.push(1, 1, 2.0);
        k.push(1, 2, -1.0);
        k.push(2, 1, -1.0);
        k.push(2, 2, 1.0);

        let mut m = TripletMatrix::new(3, 3);
        m.push(0, 0, 1.0);
        m.push(1, 1, 1.0);
        m.push(2, 2, 1.0);

        PrebakedAssembler::from_triplets([k, m], &[])
    }

    #[test]
    fn assemble_real_into_reuses_csr_buffers() {
        let assembler = laplacian_3dof();
        let mut out = assembler.assemble_real(&[1.0, 0.0]);
        let row_ptr_addr = out.row_ptrs.as_ptr();
        let col_addr = out.col_indices.as_ptr();
        let values_addr = out.values.as_ptr();

        assembler.assemble_real_into(&[1.0, 10.0], &mut out);
        assert_eq!(out.row_ptrs.as_ptr(), row_ptr_addr);
        assert_eq!(out.col_indices.as_ptr(), col_addr);
        assert_eq!(out.values.as_ptr(), values_addr);
        // K + 10*I on the diagonal.
        let reference = assembler.assemble_real(&[1.0, 10.0]);
        assert_eq!(out.values, reference.values);
        assert_eq!(out.values[0], 11.0);
    }

    #[test]
    fn test_apply_dirichlet_zeros_columns() {
        let mut assembler = laplacian_3dof();

        // Constrain DOF 0
        let nodes = std::collections::HashSet::from([0usize]);
        assembler.apply_dirichlet_nodes(&nodes);

        // Assemble K (coeffs = [1, 0])
        let csr = assembler.assemble_real(&[1.0, 0.0]);

        // Row 0 should be [1, 0, 0] (identity)
        let row0: Vec<(usize, f64)> = csr.row_entries(0).collect();
        for &(col, val) in &row0 {
            if col == 0 {
                assert!((val - 1.0).abs() < 1e-14, "K[0,0] = {val}, expected 1.0");
            } else {
                assert!(val.abs() < 1e-14, "K[0,{col}] = {val}, expected 0.0");
            }
        }

        // Row 1: K[1,0] should be zeroed (column elimination)
        let row1: Vec<(usize, f64)> = csr.row_entries(1).collect();
        for &(col, val) in &row1 {
            if col == 0 {
                assert!(
                    val.abs() < 1e-14,
                    "K[1,0] = {val}, expected 0.0 (column zeroed)"
                );
            }
        }
    }

    #[test]
    fn test_apply_dirichlet_preserves_symmetry() {
        let mut assembler = laplacian_3dof();

        let nodes = std::collections::HashSet::from([0usize, 2]);
        assembler.apply_dirichlet_nodes(&nodes);

        let csr = assembler.assemble_real(&[1.0, 0.0]);

        // Check A[i,j] == A[j,i] for all pairs
        for i in 0..3 {
            for &(j, val_ij) in &csr.row_entries(i).collect::<Vec<_>>() {
                let val_ji = csr
                    .row_entries(j)
                    .find(|&(col, _)| col == i)
                    .map(|(_, v)| v)
                    .unwrap_or(0.0);
                assert!(
                    (val_ij - val_ji).abs() < 1e-14,
                    "Symmetry broken: A[{i},{j}] = {val_ij}, A[{j},{i}] = {val_ji}"
                );
            }
        }
    }
}
