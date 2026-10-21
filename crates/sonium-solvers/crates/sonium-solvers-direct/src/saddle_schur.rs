//! Exact block-Schur solver for saddle-point systems.
//!
//! Solves systems of the form
//!
//! ```text
//! [ A   B^T ] [u]   [f]
//! [ B   -C  ] [p] = [g]
//! ```
//!
//! by eliminating the primal block: with `S = C + B·A⁻¹·Bᵀ`,
//!
//! ```text
//! S·p = B·A⁻¹·f − g,     u = A⁻¹·(f − Bᵀ·p).
//! ```
//!
//! The factorization costs one sparse LU of `A`, one triangular
//! solve per constraint column to assemble `S`, and one sparse LU
//! of `S`. It is therefore the right tool when the constraint block
//! is small or moderate (up to a few hundred rows), when the same
//! blocks are re-solved with many right-hand sides, or when an
//! exact Schur complement is needed as a building block (for
//! example to validate an approximate Schur preconditioner). For
//! large constraint blocks the assembled `S` is generally dense and
//! the monolithic sparse LU or a block preconditioner is cheaper.
//!
//! The transpose is the algebraic transpose (no conjugation), which
//! matches the real-valued saddle systems produced by mixed finite
//! element discretizations. Works with all `ComplexField` types.

use num_traits::{FromPrimitive, Zero};
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::ComplexField;
use sonium_solvers_sparse::CsrMatrix;
use thiserror::Error;

use super::SparseLuError;
use super::sparse_lu::SparseLuFactorization;

/// Errors produced by the block-Schur saddle solver.
#[derive(Error, Debug)]
pub enum SaddleSchurError {
    /// A block that must be square is not.
    #[error("saddle block `{block}` must be square ({rows} x {cols})")]
    NotSquare {
        /// Which block failed the check (`A` or `C`).
        block: &'static str,
        /// Row count of the block.
        rows: usize,
        /// Column count of the block.
        cols: usize,
    },
    /// Block or right-hand-side dimensions do not agree.
    #[error("saddle dimension mismatch: {0}")]
    DimensionMismatch(String),
    /// The primal block or the Schur complement failed to factorize.
    #[error("sparse LU failed inside the Schur solver: {0}")]
    Lu(#[from] SparseLuError),
}

/// Exact block-Schur factorization of a saddle-point system.
///
/// Build with [`SaddlePointSchurSolver::factorize`], then solve any
/// number of right-hand sides with
/// [`SaddlePointSchurSolver::solve`] or
/// [`SaddlePointSchurSolver::solve_monolithic`].
pub struct SaddlePointSchurSolver<T: ComplexField> {
    primal_factor: SparseLuFactorization<T>,
    schur_factor: SparseLuFactorization<T>,
    constraint: CsrMatrix<T>,
    schur: CsrMatrix<T>,
    primal_dimension: usize,
    constraint_dimension: usize,
}

impl<T: ComplexField + FromPrimitive> SaddlePointSchurSolver<T> {
    /// Factorize the saddle system `[A Bᵀ; B −C]`.
    ///
    /// `constraint_c` is the (usually zero or small) stabilization
    /// block `C`; pass `None` for the common unstabilized case. The
    /// primal block `A` must be square and nonsingular.
    ///
    /// # Errors
    ///
    /// Returns [`SaddleSchurError::NotSquare`] if `A` or `C` is not
    /// square, [`SaddleSchurError::DimensionMismatch`] if the block
    /// shapes do not conform, and [`SaddleSchurError::Lu`] if `A`
    /// or the assembled Schur complement is singular.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_solvers_direct::SaddlePointSchurSolver;
    /// use sonium_solvers_sparse::CsrMatrix;
    ///
    /// // [2 0 | 1]        [f]   [3]
    /// // [0 2 | 1]  with  [g] = [3]
    /// // [1 1 | 0]
    /// let a = CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0_f64, 2.0]);
    /// let b = CsrMatrix::from_raw_parts(1, 2, vec![0, 2], vec![0, 1], vec![1.0_f64, 1.0]);
    /// let solver = SaddlePointSchurSolver::factorize(&a, &b, None).unwrap();
    /// let (u, p) = solver.solve(&[3.0, 3.0], &[3.0]).unwrap();
    /// assert!((u[0] - 1.5).abs() < 1.0e-12);
    /// assert!(p[0].abs() < 1.0e-12);
    /// ```
    pub fn factorize(
        primal: &CsrMatrix<T>,
        constraint: &CsrMatrix<T>,
        stabilization: Option<&CsrMatrix<T>>,
    ) -> Result<Self, SaddleSchurError> {
        let n = primal.num_rows;
        if primal.num_cols != n {
            return Err(SaddleSchurError::NotSquare {
                block: "A",
                rows: primal.num_rows,
                cols: primal.num_cols,
            });
        }
        let m = constraint.num_rows;
        if constraint.num_cols != n {
            return Err(SaddleSchurError::DimensionMismatch(format!(
                "constraint block B is {m} x {} but A is {n} x {n}",
                constraint.num_cols
            )));
        }
        if let Some(stabilization) = stabilization
            && (stabilization.num_rows != m || stabilization.num_cols != m)
        {
            return Err(SaddleSchurError::NotSquare {
                block: "C",
                rows: stabilization.num_rows,
                cols: stabilization.num_cols,
            });
        }

        let primal_factor = SparseLuFactorization::factorize(primal)?;

        // Assemble S = C + B·A⁻¹·Bᵀ column by column: column j is
        // B·(A⁻¹·(Bᵀ e_j)) + C e_j, and Bᵀ e_j is row j of B.
        let mut schur_dense = DenseMatrix::<T>::zeros(m, m);
        for j in 0..m {
            let mut column_rhs = vec![T::zero(); n];
            for index in constraint.row_ptrs[j]..constraint.row_ptrs[j + 1] {
                column_rhs[constraint.col_indices[index]] = constraint.values[index];
            }
            let eliminated = primal_factor.solve(&column_rhs)?;
            let schur_column = constraint.matvec(&eliminated);
            for (i, value) in schur_column.into_iter().enumerate() {
                schur_dense[[i, j]] = value;
            }
            if let Some(stabilization) = stabilization {
                for index in stabilization.row_ptrs[j]..stabilization.row_ptrs[j + 1] {
                    schur_dense[[stabilization.col_indices[index], j]] += stabilization.values[index];
                }
            }
        }
        let schur = CsrMatrix::from_dense(&schur_dense, T::Real::zero());
        let schur_factor = SparseLuFactorization::factorize(&schur)?;

        Ok(Self {
            primal_factor,
            schur_factor,
            constraint: constraint.clone(),
            schur,
            primal_dimension: n,
            constraint_dimension: m,
        })
    }

    /// Dimensions `(primal, constraint)` of the factorized system.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.primal_dimension, self.constraint_dimension)
    }

    /// The assembled Schur complement `S = C + B·A⁻¹·Bᵀ`.
    pub fn schur_complement(&self) -> &CsrMatrix<T> {
        &self.schur
    }

    /// Solve for one right-hand-side pair `(f, g)`.
    ///
    /// Returns the primal and constraint parts `(u, p)` of the
    /// solution.
    ///
    /// # Errors
    ///
    /// Returns [`SaddleSchurError::DimensionMismatch`] when a
    /// right-hand side has the wrong length and
    /// [`SaddleSchurError::Lu`] when a triangular solve breaks down.
    pub fn solve(&self, primal_rhs: &[T], constraint_rhs: &[T]) -> Result<(Vec<T>, Vec<T>), SaddleSchurError> {
        if primal_rhs.len() != self.primal_dimension {
            return Err(SaddleSchurError::DimensionMismatch(format!(
                "primal right-hand side has length {} but A is {} x {}",
                primal_rhs.len(),
                self.primal_dimension,
                self.primal_dimension
            )));
        }
        if constraint_rhs.len() != self.constraint_dimension {
            return Err(SaddleSchurError::DimensionMismatch(format!(
                "constraint right-hand side has length {} but B has {} rows",
                constraint_rhs.len(),
                self.constraint_dimension
            )));
        }
        // p = S⁻¹ (B·A⁻¹·f − g); u = A⁻¹ (f − Bᵀ·p).
        let eliminated_rhs = self.primal_factor.solve(primal_rhs)?;
        let mut schur_rhs = self.constraint.matvec(&eliminated_rhs);
        for (value, rhs) in schur_rhs.iter_mut().zip(constraint_rhs.iter()) {
            *value -= *rhs;
        }
        let constraint_solution = self.schur_factor.solve(&schur_rhs)?;
        let coupling = self.constraint.matvec_transpose(&constraint_solution);
        let primal_residual: Vec<T> = primal_rhs
            .iter()
            .zip(coupling.iter())
            .map(|(rhs, value)| *rhs - *value)
            .collect();
        let primal_solution = self.primal_factor.solve(&primal_residual)?;
        Ok((primal_solution, constraint_solution))
    }

    /// Solve with a stacked right-hand side `[f; g]`.
    ///
    /// # Errors
    ///
    /// Returns [`SaddleSchurError::DimensionMismatch`] when the
    /// stacked length is not `primal + constraint`, and
    /// [`SaddleSchurError::Lu`] when a triangular solve breaks down.
    pub fn solve_monolithic(&self, rhs: &[T]) -> Result<Vec<T>, SaddleSchurError> {
        let total = self.primal_dimension + self.constraint_dimension;
        if rhs.len() != total {
            return Err(SaddleSchurError::DimensionMismatch(format!(
                "stacked right-hand side has length {} but the system is {total} x {total}",
                rhs.len()
            )));
        }
        let (primal, constraint) = self.solve(
            &rhs[..self.primal_dimension],
            &rhs[self.primal_dimension..],
        )?;
        let mut solution = primal;
        solution.extend_from_slice(&constraint);
        Ok(solution)
    }
}

/// Blocks of a monolithic saddle system: `(A, B, C)` in solver
/// convention (the monolithic `(2,2)` block is `-C`).
pub type SaddleBlocks<T> = (CsrMatrix<T>, CsrMatrix<T>, CsrMatrix<T>);

/// Split a monolithic saddle matrix into its `(A, B, C)` blocks.
///
/// `primal_size` is the number of primal rows/columns. The returned
/// stabilization block follows the solver convention: the
/// monolithic `(2,2)` block is `-C`, so the returned `C` is its
/// negation (a zero block stays zero).
///
/// # Errors
///
/// Returns [`SaddleSchurError::NotSquare`] if the matrix is not
/// square and [`SaddleSchurError::DimensionMismatch`] if
/// `primal_size` is not strictly between zero and the dimension.
pub fn saddle_blocks_from_monolithic<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    primal_size: usize,
) -> Result<SaddleBlocks<T>, SaddleSchurError> {
    if matrix.num_rows != matrix.num_cols {
        return Err(SaddleSchurError::NotSquare {
            block: "monolithic",
            rows: matrix.num_rows,
            cols: matrix.num_cols,
        });
    }
    let total = matrix.num_rows;
    if primal_size == 0 || primal_size >= total {
        return Err(SaddleSchurError::DimensionMismatch(format!(
            "primal block size {primal_size} must be between 1 and {}",
            total - 1
        )));
    }
    let extract = |row_range: std::ops::Range<usize>,
                   col_offset: usize,
                   rows: usize,
                   cols: usize,
                   negate: bool|
     -> CsrMatrix<T> {
        let mut row_ptrs = Vec::with_capacity(rows + 1);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        row_ptrs.push(0);
        for row in row_range {
            for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
                let col = matrix.col_indices[index];
                if col >= col_offset && col < col_offset + cols {
                    col_indices.push(col - col_offset);
                    values.push(if negate {
                        -matrix.values[index]
                    } else {
                        matrix.values[index]
                    });
                }
            }
            row_ptrs.push(col_indices.len());
        }
        CsrMatrix::from_raw_parts(rows, cols, row_ptrs, col_indices, values)
    };
    let primal = extract(0..primal_size, 0, primal_size, primal_size, false);
    let constraint = extract(
        primal_size..total,
        0,
        total - primal_size,
        primal_size,
        false,
    );
    let stabilization = extract(
        primal_size..total,
        primal_size,
        total - primal_size,
        total - primal_size,
        true,
    );
    Ok((primal, constraint, stabilization))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_solvers_core::dense::DenseMatrix;

    fn dense(rows: &[&[f64]]) -> DenseMatrix<f64> {
        let mut matrix = DenseMatrix::zeros(rows.len(), rows[0].len());
        for (i, row) in rows.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                matrix[[i, j]] = *value;
            }
        }
        matrix
    }

    fn dense_saddle() -> (CsrMatrix<f64>, CsrMatrix<f64>, Vec<f64>, Vec<f64>) {
        // A: 3x3 SPD-ish; B: 2x3 constraint; C = 0.
        let primal = CsrMatrix::from_dense(
            &dense(&[
                &[4.0, 1.0, 0.0],
                &[1.0, 3.0, 1.0],
                &[0.0, 1.0, 2.0],
            ]),
            1.0e-15,
        );
        let constraint = CsrMatrix::from_dense(
            &dense(&[&[1.0, 0.0, -1.0], &[0.0, 2.0, 1.0]]),
            1.0e-15,
        );
        let u_exact = vec![1.0, -2.0, 0.5];
        let p_exact = vec![3.0, -1.0];
        // f = A u + Bᵀ p; g = B u.
        let mut f = primal.matvec(&u_exact);
        let bt_p = constraint.matvec_transpose(&p_exact);
        for (value, coupling) in f.iter_mut().zip(&bt_p) {
            *value += *coupling;
        }
        let g = constraint.matvec(&u_exact);
        (primal, constraint, {
            let mut stacked = f.clone();
            stacked.extend_from_slice(&g);
            stacked
        }, {
            let mut exact = u_exact;
            exact.extend_from_slice(&p_exact);
            exact
        })
    }

    #[test]
    fn schur_solve_matches_exact_solution() {
        let (primal, constraint, rhs, exact) = dense_saddle();
        let solver =
            SaddlePointSchurSolver::factorize(&primal, &constraint, None).expect("factorize");
        assert_eq!(solver.dimensions(), (3, 2));
        let solution = solver.solve_monolithic(&rhs).expect("solve");
        for (actual, expected) in solution.iter().zip(&exact) {
            assert!((actual - expected).abs() < 1.0e-10, "{actual} vs {expected}");
        }
        let (u, p) = solver.solve(&rhs[..3], &rhs[3..]).expect("split solve");
        for (actual, expected) in u.iter().zip(&exact[..3]) {
            assert!((actual - expected).abs() < 1.0e-12);
        }
        for (actual, expected) in p.iter().zip(&exact[3..]) {
            assert!((actual - expected).abs() < 1.0e-12);
        }
    }

    #[test]
    fn monolithic_split_matches_monolithic_lu() {
        let (primal, constraint, rhs, exact) = dense_saddle();
        // Assemble the monolithic [A Bᵀ; B 0] system densely.
        let mut monolithic = DenseMatrix::zeros(5, 5);
        for i in 0..3 {
            for index in primal.row_ptrs[i]..primal.row_ptrs[i + 1] {
                monolithic[[i, primal.col_indices[index]]] = primal.values[index];
            }
        }
        for i in 0..2 {
            for index in constraint.row_ptrs[i]..constraint.row_ptrs[i + 1] {
                let j = constraint.col_indices[index];
                monolithic[[3 + i, j]] = constraint.values[index];
                monolithic[[j, 3 + i]] = constraint.values[index];
            }
        }
        let monolithic = CsrMatrix::from_dense(&monolithic, 1.0e-15);
        let (a, b, c) = saddle_blocks_from_monolithic(&monolithic, 3).expect("split blocks");
        assert_eq!(a.values, primal.values);
        assert_eq!(b.values, constraint.values);
        assert!(c.values.iter().all(|value| *value == 0.0));

        let solver = SaddlePointSchurSolver::factorize(&a, &b, Some(&c)).expect("factorize");
        let solution = solver.solve_monolithic(&rhs).expect("solve");
        let reference = SparseLuFactorization::factorize(&monolithic)
            .expect("monolithic factorize")
            .solve(&rhs)
            .expect("monolithic solve");
        for (actual, expected) in solution.iter().zip(&reference) {
            assert!((actual - expected).abs() < 1.0e-10);
        }
        for (actual, expected) in solution.iter().zip(&exact) {
            assert!((actual - expected).abs() < 1.0e-10);
        }
        // The assembled Schur complement is B·A⁻¹·Bᵀ here (C = 0).
        assert_eq!(solver.schur_complement().num_rows, 2);
    }

    #[test]
    fn stabilization_block_enters_the_schur_complement() {
        let (primal, constraint, rhs, _exact) = dense_saddle();
        let stabilization = CsrMatrix::from_dense(
            &dense(&[&[0.25, 0.0], &[0.0, 0.5]]),
            1.0e-15,
        );
        let solver =
            SaddlePointSchurSolver::factorize(&primal, &constraint, Some(&stabilization))
                .expect("factorize");
        // Solve and verify against the monolithic system [A Bᵀ; B −C].
        let solution = solver.solve_monolithic(&rhs).expect("solve");
        let mut monolithic = DenseMatrix::zeros(5, 5);
        for i in 0..3 {
            for index in primal.row_ptrs[i]..primal.row_ptrs[i + 1] {
                monolithic[[i, primal.col_indices[index]]] = primal.values[index];
            }
        }
        for i in 0..2 {
            for index in constraint.row_ptrs[i]..constraint.row_ptrs[i + 1] {
                let j = constraint.col_indices[index];
                monolithic[[3 + i, j]] = constraint.values[index];
                monolithic[[j, 3 + i]] = constraint.values[index];
            }
        }
        monolithic[[3, 3]] = -0.25;
        monolithic[[4, 4]] = -0.5;
        let monolithic = CsrMatrix::from_dense(&monolithic, 1.0e-15);
        let residual = monolithic.matvec(&solution);
        for (actual, expected) in residual.iter().zip(&rhs) {
            assert!((actual - expected).abs() < 1.0e-10);
        }
    }

    #[test]
    fn rejects_bad_shapes_and_singular_primal() {
        let (primal, constraint, _rhs, _exact) = dense_saddle();
        // B with the wrong column count.
        let bad_constraint = CsrMatrix::from_dense(&dense(&[&[1.0, 0.0]]), 1.0e-15);
        let Err(error) = SaddlePointSchurSolver::factorize(&primal, &bad_constraint, None) else {
            panic!("mismatched B must be rejected");
        };
        assert!(matches!(error, SaddleSchurError::DimensionMismatch(_)));

        // Singular A (zero row).
        let singular = CsrMatrix::from_dense(
            &dense(&[&[1.0, 0.0, 0.0], &[0.0, 0.0, 0.0], &[0.0, 0.0, 1.0]]),
            1.0e-15,
        );
        let Err(error) = SaddlePointSchurSolver::factorize(&singular, &constraint, None) else {
            panic!("singular A must be rejected");
        };
        assert!(matches!(error, SaddleSchurError::Lu(_)));

        // Wrong right-hand-side length.
        let solver =
            SaddlePointSchurSolver::factorize(&primal, &constraint, None).expect("factorize");
        let Err(error) = solver.solve(&[1.0], &[1.0, 2.0]) else {
            panic!("short primal rhs must be rejected");
        };
        assert!(matches!(error, SaddleSchurError::DimensionMismatch(_)));
    }
}
