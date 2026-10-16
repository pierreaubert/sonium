//! Small, deterministic factor/solve property matrix.
//!
//! The fixtures are intentionally independent of the direct implementations:
//! each test starts from a known solution and recomputes `A*x` as its oracle.
//! The same checks exercise dense/sparse, real/complex, single/block RHS, and
//! symmetric permutation invariance.

use num_complex::{Complex32, Complex64};
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::{
    DenseDirectConfig, DenseDirectMethod, DirectFactorization, MatrixClass, Ordering,
    SparseDirectConfig, SparseDirectMethod, SparseLuFactorization, factorize_dense,
    factorize_sparse,
};
use sonium_solvers::sparse::CsrMatrix;

fn dense_matvec<T: Copy + std::ops::AddAssign + std::ops::Mul<Output = T> + Default>(
    matrix: &[Vec<T>],
    x: &[T],
) -> Vec<T> {
    matrix
        .iter()
        .map(|row| {
            let mut value = T::default();
            for (&a, &b) in row.iter().zip(x) {
                value += a * b;
            }
            value
        })
        .collect()
}

fn real_matrix() -> Vec<Vec<f64>> {
    vec![
        vec![4.0, 1.0, 0.5],
        vec![2.0, 5.0, 1.0],
        vec![0.25, 1.5, 3.0],
    ]
}

fn real_spd_matrix() -> Vec<Vec<f64>> {
    vec![
        vec![4.0, 1.0, 0.0],
        vec![1.0, 3.0, 1.0],
        vec![0.0, 1.0, 2.0],
    ]
}

fn csr_from_dense<T: Copy + sonium_solvers::ComplexField>(matrix: &[Vec<T>]) -> CsrMatrix<T>
where
    T: num_traits::Zero + PartialEq,
{
    let triplets = matrix
        .iter()
        .enumerate()
        .flat_map(|(row, values)| {
            values
                .iter()
                .enumerate()
                .filter_map(move |(col, &value)| (value != T::zero()).then_some((row, col, value)))
        })
        .collect();
    CsrMatrix::from_triplets(matrix.len(), matrix[0].len(), triplets)
}

fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {index}: actual={actual}, expected={expected}"
        );
    }
}

#[test]
fn dense_and_sparse_lu_match_known_solution_and_residual() {
    let matrix = real_matrix();
    let expected = [1.0, -0.5, 2.0];
    let rhs = dense_matvec(&matrix, &expected);
    let dense = DenseMatrix::from_row_major_vec(3, 3, matrix.iter().flatten().copied().collect());
    let dense_factor = factorize_dense(
        &dense,
        &DenseDirectConfig {
            matrix_class: MatrixClass::General,
            method: DenseDirectMethod::Lu,
            ..Default::default()
        },
    )
    .expect("dense LU setup");
    let dense_solution = dense_factor.solve(&rhs).expect("dense LU solve");

    let csr = csr_from_dense(&matrix);
    let sparse_factor = factorize_sparse(
        &csr,
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .expect("sparse LU setup");
    let sparse_solution = sparse_factor.solve(&rhs).expect("sparse LU solve");

    assert_close(&dense_solution, &expected, 1e-12);
    assert_close(&sparse_solution, &expected, 1e-12);
    assert_close(&dense_matvec(&matrix, &dense_solution), &rhs, 1e-12);
    assert_close(&dense_matvec(&matrix, &sparse_solution), &rhs, 1e-12);
}

#[test]
fn dense_and_sparse_cholesky_match_known_spd_solution() {
    let matrix = real_spd_matrix();
    let expected = [1.0, -2.0, 0.75];
    let rhs = dense_matvec(&matrix, &expected);
    let dense = DenseMatrix::from_row_major_vec(3, 3, matrix.iter().flatten().copied().collect());
    let dense_factor = factorize_dense(
        &dense,
        &DenseDirectConfig {
            matrix_class: MatrixClass::SymmetricPositiveDefinite,
            method: DenseDirectMethod::Cholesky,
            ..Default::default()
        },
    )
    .expect("dense Cholesky setup");
    let dense_solution = dense_factor.solve(&rhs).expect("dense Cholesky solve");

    let csr = csr_from_dense(&matrix);
    let sparse_factor = factorize_sparse(
        &csr,
        &SparseDirectConfig {
            matrix_class: MatrixClass::SymmetricPositiveDefinite,
            method: SparseDirectMethod::SimplicialCholesky,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .expect("sparse Cholesky setup");
    let sparse_solution = sparse_factor.solve(&rhs).expect("sparse Cholesky solve");

    assert_close(&dense_solution, &expected, 1e-12);
    assert_close(&sparse_solution, &expected, 1e-12);
}

#[test]
fn complex_lu_preserves_conjugate_free_solution_and_residual() {
    let matrix = vec![
        vec![Complex64::new(3.0, 1.0), Complex64::new(1.0, -2.0)],
        vec![Complex64::new(2.0, 0.5), Complex64::new(4.0, -1.0)],
    ];
    let expected = [Complex64::new(0.5, -1.0), Complex64::new(2.0, 0.25)];
    let rhs = dense_matvec(&matrix, &expected);
    let dense = DenseMatrix::from_row_major_vec(2, 2, matrix.iter().flatten().copied().collect());
    let factor = factorize_dense(
        &dense,
        &DenseDirectConfig {
            matrix_class: MatrixClass::General,
            method: DenseDirectMethod::Lu,
            ..Default::default()
        },
    )
    .expect("complex dense LU setup");
    let solution = factor.solve(&rhs).expect("complex dense LU solve");
    for (actual, expected) in solution.iter().zip(expected) {
        assert!((*actual - expected).norm() <= 1e-12);
    }
    let residual = dense_matvec(&matrix, &solution)
        .iter()
        .zip(rhs)
        .map(|(actual, expected)| (*actual - expected).norm())
        .fold(0.0, f64::max);
    assert!(residual <= 1e-12, "complex residual={residual}");
}

#[test]
fn complex_hermitian_positive_definite_dense_sparse_cholesky_match() {
    let matrix = vec![
        vec![
            Complex64::new(4.0, 0.0),
            Complex64::new(2.0, -2.0),
            Complex64::new(1.0, 0.5),
        ],
        vec![
            Complex64::new(2.0, 2.0),
            Complex64::new(6.0, 0.0),
            Complex64::new(-0.75, -0.75),
        ],
        vec![
            Complex64::new(1.0, -0.5),
            Complex64::new(-0.75, 0.75),
            Complex64::new(3.375, 0.0),
        ],
    ];
    let expected = [
        Complex64::new(0.5, -1.0),
        Complex64::new(1.25, 0.25),
        Complex64::new(-0.75, 0.5),
    ];
    let rhs = dense_matvec(&matrix, &expected);
    let dense = DenseMatrix::from_row_major_vec(3, 3, matrix.iter().flatten().copied().collect());
    let dense_factor = factorize_dense(
        &dense,
        &DenseDirectConfig {
            matrix_class: MatrixClass::HermitianPositiveDefinite,
            method: DenseDirectMethod::Cholesky,
            ..Default::default()
        },
    )
    .expect("complex dense Cholesky setup");
    let dense_solution = dense_factor
        .solve(&rhs)
        .expect("complex dense Cholesky solve");

    let csr = csr_from_dense(&matrix);
    let sparse_factor = factorize_sparse(
        &csr,
        &SparseDirectConfig {
            matrix_class: MatrixClass::HermitianPositiveDefinite,
            method: SparseDirectMethod::SimplicialCholesky,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .expect("complex sparse Cholesky setup");
    let sparse_solution = sparse_factor
        .solve(&rhs)
        .expect("complex sparse Cholesky solve");

    let solution_error = dense_solution
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - expected).norm())
        .fold(0.0, f64::max);
    let dense_sparse_error = dense_solution
        .iter()
        .zip(&sparse_solution)
        .map(|(dense, sparse)| (*dense - *sparse).norm())
        .fold(0.0, f64::max);
    let residual = dense_matvec(&matrix, &sparse_solution)
        .iter()
        .zip(&rhs)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0, f64::max);
    assert!(
        solution_error <= 1e-12,
        "complex HPD solution error={solution_error}"
    );
    assert!(
        dense_sparse_error <= 1e-12,
        "complex HPD dense/sparse error={dense_sparse_error}"
    );
    assert!(residual <= 1e-12, "complex HPD residual={residual}");

    let second_expected = [
        Complex64::new(-1.0, 0.25),
        Complex64::new(0.25, -0.5),
        Complex64::new(0.75, 1.0),
    ];
    let second_rhs = dense_matvec(&matrix, &second_expected);
    let block_rhs: Vec<_> = (0..3).flat_map(|row| [rhs[row], second_rhs[row]]).collect();
    let mut block_solution = vec![Complex64::new(f64::NAN, f64::NAN); 6];
    dense_factor
        .solve_many_into(&block_rhs, &mut block_solution, 2)
        .expect("complex HPD dense block solve");
    let expected_block: Vec<_> = (0..3)
        .flat_map(|row| [expected[row], second_expected[row]])
        .collect();
    let block_error = block_solution
        .iter()
        .zip(expected_block)
        .map(|(actual, expected)| (*actual - expected).norm())
        .fold(0.0, f64::max);
    assert!(
        block_error <= 1e-12,
        "complex HPD block error={block_error}"
    );
}

#[test]
fn complex_hermitian_indefinite_dense_sparse_ldlt_match() {
    let matrix = vec![
        vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 1.0)],
        vec![Complex64::new(1.0, -1.0), Complex64::new(0.0, 0.0)],
    ];
    let expected = [Complex64::new(0.75, -0.25), Complex64::new(-1.0, 0.5)];
    let rhs = dense_matvec(&matrix, &expected);
    let dense = DenseMatrix::from_row_major_vec(2, 2, matrix.iter().flatten().copied().collect());
    let dense_factor = factorize_dense(
        &dense,
        &DenseDirectConfig {
            matrix_class: MatrixClass::HermitianIndefinite,
            method: DenseDirectMethod::Ldlt,
            ..Default::default()
        },
    )
    .expect("complex dense LDLT setup");
    let dense_solution = dense_factor.solve(&rhs).expect("complex dense LDLT solve");

    let sparse_factor = factorize_sparse(
        &csr_from_dense(&matrix),
        &SparseDirectConfig {
            matrix_class: MatrixClass::HermitianIndefinite,
            method: SparseDirectMethod::MultifrontalLdlt,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .expect("complex sparse LDLT setup");
    let sparse_solution = sparse_factor
        .solve(&rhs)
        .expect("complex sparse LDLT solve");
    let second_expected = [Complex64::new(-0.25, 0.5), Complex64::new(0.5, -1.0)];
    let second_rhs = dense_matvec(&matrix, &second_expected);
    let block_rhs: Vec<_> = (0..2).flat_map(|row| [rhs[row], second_rhs[row]]).collect();
    let mut block_solution = vec![Complex64::new(f64::NAN, f64::NAN); 4];
    sparse_factor
        .solve_many_into(&block_rhs, &mut block_solution, 2)
        .expect("complex sparse LDLT block solve");
    let expected_block: Vec<_> = (0..2)
        .flat_map(|row| [expected[row], second_expected[row]])
        .collect();
    let block_error = block_solution
        .iter()
        .zip(expected_block)
        .map(|(actual, expected)| (*actual - expected).norm())
        .fold(0.0, f64::max);
    let dense_error = dense_solution
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - expected).norm())
        .fold(0.0, f64::max);
    let sparse_error = sparse_solution
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - expected).norm())
        .fold(0.0, f64::max);
    let residual = dense_matvec(&matrix, &sparse_solution)
        .iter()
        .zip(&rhs)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0, f64::max);
    assert!(
        dense_error <= 1e-12,
        "complex LDLT dense error={dense_error}"
    );
    assert!(
        sparse_error <= 1e-12,
        "complex LDLT sparse error={sparse_error}"
    );
    assert!(
        block_error <= 1e-12,
        "complex LDLT sparse block error={block_error}"
    );
    assert!(residual <= 1e-12, "complex LDLT residual={residual}");
}

#[test]
fn complex32_dense_sparse_lu_match_known_solution() {
    let matrix = vec![
        vec![Complex32::new(2.0, 0.5), Complex32::new(0.5, -0.25)],
        vec![Complex32::new(1.0, -0.75), Complex32::new(3.0, 0.25)],
    ];
    let expected = [Complex32::new(0.5, -0.25), Complex32::new(-1.0, 0.75)];
    let rhs = dense_matvec(&matrix, &expected);
    let dense = DenseMatrix::from_row_major_vec(2, 2, matrix.iter().flatten().copied().collect());
    let dense_factor = factorize_dense(
        &dense,
        &DenseDirectConfig {
            matrix_class: MatrixClass::General,
            method: DenseDirectMethod::Lu,
            ..Default::default()
        },
    )
    .expect("Complex32 dense LU setup");
    let dense_solution = dense_factor.solve(&rhs).expect("Complex32 dense LU solve");
    let sparse_factor = factorize_sparse(
        &csr_from_dense(&matrix),
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .expect("Complex32 sparse LU setup");
    let sparse_solution = sparse_factor
        .solve(&rhs)
        .expect("Complex32 sparse LU solve");
    let dense_error = dense_solution
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - expected).norm() as f64)
        .fold(0.0, f64::max);
    let sparse_error = sparse_solution
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - expected).norm() as f64)
        .fold(0.0, f64::max);
    assert!(dense_error <= 1e-5, "Complex32 dense error={dense_error}");
    assert!(
        sparse_error <= 1e-5,
        "Complex32 sparse error={sparse_error}"
    );
}

#[test]
fn dense_block_rhs_matches_single_rhs_solves() {
    let matrix = real_matrix();
    let dense = DenseMatrix::from_row_major_vec(3, 3, matrix.iter().flatten().copied().collect());
    let factor = factorize_dense(
        &dense,
        &DenseDirectConfig {
            matrix_class: MatrixClass::General,
            method: DenseDirectMethod::Lu,
            ..Default::default()
        },
    )
    .expect("dense LU setup");
    let x0 = [1.0, -0.5, 2.0];
    let x1 = [-2.0, 0.25, 1.5];
    let b0 = dense_matvec(&matrix, &x0);
    let b1 = dense_matvec(&matrix, &x1);
    let block_rhs = [b0[0], b1[0], b0[1], b1[1], b0[2], b1[2]];
    let mut block_solution = vec![f64::NAN; 6];
    factor
        .solve_many_into(&block_rhs, &mut block_solution, 2)
        .expect("dense block solve");
    assert_close(
        &block_solution,
        &[x0[0], x1[0], x0[1], x1[1], x0[2], x1[2]],
        1e-12,
    );
    assert_close(&factor.solve(&b0).expect("single RHS 0"), &x0, 1e-12);
    assert_close(&factor.solve(&b1).expect("single RHS 1"), &x1, 1e-12);
}

#[test]
fn symmetric_row_column_permutation_preserves_solution() {
    let matrix = real_spd_matrix();
    let expected = [1.0, -2.0, 0.75];
    let rhs = dense_matvec(&matrix, &expected);
    let permutation = [2usize, 0, 1];
    let permuted_matrix: Vec<Vec<f64>> = permutation
        .iter()
        .map(|&row| permutation.iter().map(|&col| matrix[row][col]).collect())
        .collect();
    let permuted_rhs: Vec<f64> = permutation.iter().map(|&index| rhs[index]).collect();
    let csr = csr_from_dense(&permuted_matrix);
    let factor = factorize_sparse(
        &csr,
        &SparseDirectConfig {
            matrix_class: MatrixClass::SymmetricPositiveDefinite,
            method: SparseDirectMethod::SimplicialCholesky,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .expect("permuted sparse Cholesky setup");
    let solution = factor.solve(&permuted_rhs).expect("permuted solve");
    let expected_permuted: Vec<f64> = permutation.iter().map(|&index| expected[index]).collect();
    assert_close(&solution, &expected_permuted, 1e-12);
}
fn seeded_general_matrix(seed: u64) -> Vec<Vec<f64>> {
    let n = 5;
    let mut state = seed;
    let mut matrix = vec![vec![0.0; n]; n];
    for row in 0..n {
        let mut row_sum = 0.0;
        for column in 0..n {
            if row == column {
                continue;
            }
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let value = ((state >> 11) as f64 / (1_u64 << 53) as f64) - 0.5;
            matrix[row][column] = value;
            row_sum += value.abs();
        }
        matrix[row][row] = 2.0 + row_sum;
    }
    matrix
}

#[test]
fn seeded_general_factor_solve_reconstructs_dense_sparse_and_block_rhs() {
    for seed in 0..8_u64 {
        let matrix = seeded_general_matrix(0xFAC7_0A ^ seed);
        let expected: Vec<_> = (0..matrix.len())
            .map(|index| (index as f64 + 1.0) * (1.0 - 0.1 * seed as f64))
            .collect();
        let rhs = dense_matvec(&matrix, &expected);
        let dense = DenseMatrix::from_row_major_vec(
            matrix.len(),
            matrix.len(),
            matrix.iter().flatten().copied().collect(),
        );
        let dense_factor = factorize_dense(
            &dense,
            &DenseDirectConfig {
                matrix_class: MatrixClass::General,
                method: DenseDirectMethod::Lu,
                ..Default::default()
            },
        )
        .expect("seeded dense LU setup");
        let sparse_factor = factorize_sparse(
            &csr_from_dense(&matrix),
            &SparseDirectConfig {
                matrix_class: MatrixClass::General,
                method: SparseDirectMethod::SimplicialLu,
                ordering: Ordering::Auto,
                ..Default::default()
            },
            None,
        )
        .expect("seeded sparse LU setup");

        let dense_solution = dense_factor.solve(&rhs).expect("seeded dense solve");
        let sparse_solution = sparse_factor.solve(&rhs).expect("seeded sparse solve");
        assert_close(&dense_solution, &expected, 2.0e-11);
        assert_close(&sparse_solution, &expected, 2.0e-11);

        let second_expected: Vec<_> = expected.iter().map(|value| -0.5 * value).collect();
        let second_rhs = dense_matvec(&matrix, &second_expected);
        let mut block_rhs = Vec::with_capacity(2 * matrix.len());
        for row in 0..matrix.len() {
            block_rhs.push(rhs[row]);
            block_rhs.push(second_rhs[row]);
        }
        let mut block_solution = vec![f64::NAN; block_rhs.len()];
        dense_factor
            .solve_many_into(&block_rhs, &mut block_solution, 2)
            .expect("seeded dense block solve");
        let expected_block: Vec<_> = (0..matrix.len())
            .flat_map(|row| [expected[row], second_expected[row]])
            .collect();
        assert_close(&block_solution, &expected_block, 2.0e-11);
    }
}

#[test]
fn sparse_lu_orderings_and_symbolic_value_reuse_preserve_pivoted_solution() {
    // The zero diagonal forces row pivoting; this keeps the property meaningful
    // for both sparse LU backends even though their ordering policy is internal.
    let matrix = vec![
        vec![0.0, 2.0, 0.0, 1.0],
        vec![1.0, 0.0, 3.0, 0.0],
        vec![4.0, 5.0, 6.0, 0.0],
        vec![0.0, 1.0, 2.0, 3.0],
    ];
    let expected = [1.0, -2.0, 0.5, 3.0];
    let rhs = dense_matvec(&matrix, &expected);
    let csr = csr_from_dense(&matrix);

    for method in [
        SparseDirectMethod::SimplicialLu,
        SparseDirectMethod::SupernodalLu,
    ] {
        let ordering = Ordering::Auto;
        let factor = factorize_sparse(
            &csr,
            &SparseDirectConfig {
                matrix_class: MatrixClass::General,
                method,
                ordering,
                ..Default::default()
            },
            None,
        )
        .unwrap_or_else(|error| panic!("ordering {ordering:?} factorization failed: {error}"));
        let solution = factor
            .solve(&rhs)
            .unwrap_or_else(|error| panic!("ordering {ordering:?} solve failed: {error}"));
        assert_close(&solution, &expected, 1.0e-11);
        let residual = dense_matvec(&matrix, &solution)
            .iter()
            .zip(&rhs)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            residual <= 1.0e-11,
            "ordering {ordering:?} residual={residual}"
        );
    }

    let symbolic = sonium_solvers::direct::SparseLuFactorization::analyze_supernodal(&csr)
        .expect("symbolic analysis");
    let updated = CsrMatrix::from_raw_parts(
        4,
        4,
        vec![0, 2, 4, 7, 10],
        vec![1, 3, 0, 2, 0, 1, 2, 1, 2, 3],
        vec![2.0, 1.0, 1.0, 3.0, 4.0, 5.25, 6.0, 1.0, 2.0, 3.5],
    );
    let updated_expected = [1.0, -2.0, 0.5, 3.0];
    let updated_rhs = dense_matvec(
        &[
            vec![0.0, 2.0, 0.0, 1.0],
            vec![1.0, 0.0, 3.0, 0.0],
            vec![4.0, 5.25, 6.0, 0.0],
            vec![0.0, 1.0, 2.0, 3.5],
        ],
        &updated_expected,
    );
    let reused = sonium_solvers::direct::SparseLuFactorization::factorize_supernodal_with_symbolic(
        &updated, &symbolic, 1.0,
    )
    .expect("numeric factorization with reused symbolic pattern");
    let solution = reused.solve(&updated_rhs).expect("reused-symbolic solve");
    assert_close(&solution, &updated_expected, 1.0e-11);
}

#[test]
fn sparse_lu_preprocessing_modes_and_csr_policy_are_checked() {
    let matrix = vec![
        vec![0.0, 2.0, 0.0, 1.0],
        vec![1.0, 0.0, 3.0, 0.0],
        vec![4.0, 5.0, 6.0, 0.0],
        vec![0.0, 1.0, 2.0, 3.0],
    ];
    let expected = [1.0, -2.0, 0.5, 3.0];
    let csr = csr_from_dense(&matrix);
    let rhs = dense_matvec(&matrix, &expected);

    for supernodal in [false, true] {
        for equilibrate in [false, true] {
            let factor = if supernodal {
                SparseLuFactorization::factorize_supernodal_with_preprocessing(
                    &csr,
                    1.0,
                    equilibrate,
                )
                .unwrap_or_else(|error| {
                    panic!("supernodal preprocessing={equilibrate} factorization failed: {error}")
                })
            } else {
                SparseLuFactorization::factorize_with_preprocessing(&csr, 1.0, equilibrate)
                    .unwrap_or_else(|error| {
                        panic!(
                            "simplicial preprocessing={equilibrate} factorization failed: {error}"
                        )
                    })
            };
            let solution = factor.solve(&rhs).unwrap_or_else(|error| {
                panic!("supernodal={supernodal} preprocessing={equilibrate} solve failed: {error}")
            });
            assert_close(&solution, &expected, 1.0e-10);
            let residual = dense_matvec(&matrix, &solution)
                .iter()
                .zip(&rhs)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            assert!(
                residual <= 1.0e-10,
                "supernodal={supernodal} preprocessing={equilibrate} residual={residual}"
            );
        }
    }

    let duplicate = CsrMatrix::from_raw_parts(2, 2, vec![0, 2, 2], vec![0, 0], vec![1.0, 2.0]);
    let duplicate_error = SparseLuFactorization::factorize(&duplicate)
        .err()
        .expect("duplicate CSR columns must be rejected");
    assert!(matches!(
        duplicate_error,
        sonium_solvers::direct::SparseLuError::InvalidCsrStructure(_)
    ));

    let unsorted = CsrMatrix::from_raw_parts(2, 2, vec![0, 2, 2], vec![1, 0], vec![1.0, 2.0]);
    let unsorted_error = SparseLuFactorization::factorize(&unsorted)
        .err()
        .expect("unsorted CSR columns must be rejected");
    assert!(matches!(
        unsorted_error,
        sonium_solvers::direct::SparseLuError::InvalidCsrStructure(_)
    ));
}

#[test]
fn complex_sparse_lu_matches_dense_and_symmetric_permutation() {
    let matrix = vec![
        vec![
            Complex64::new(5.0, 0.5),
            Complex64::new(1.0, -0.25),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.5, 0.1),
        ],
        vec![
            Complex64::new(0.25, 0.2),
            Complex64::new(4.5, -0.75),
            Complex64::new(1.0, 0.3),
            Complex64::new(0.0, 0.0),
        ],
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.5, -0.2),
            Complex64::new(3.5, 0.4),
            Complex64::new(0.75, -0.1),
        ],
        vec![
            Complex64::new(0.1, -0.3),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.25, 0.15),
            Complex64::new(4.0, 0.6),
        ],
    ];
    let expected = vec![
        Complex64::new(0.5, -0.75),
        Complex64::new(-1.0, 0.25),
        Complex64::new(1.25, 0.5),
        Complex64::new(0.75, -0.2),
    ];
    let rhs = dense_matvec(&matrix, &expected);
    let dense = DenseMatrix::from_row_major_vec(4, 4, matrix.iter().flatten().copied().collect());
    let dense_factor = factorize_dense(
        &dense,
        &DenseDirectConfig {
            matrix_class: MatrixClass::General,
            method: DenseDirectMethod::Lu,
            ..Default::default()
        },
    )
    .expect("complex dense LU setup");
    let sparse_factor = factorize_sparse(
        &csr_from_dense(&matrix),
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .expect("complex sparse LU setup");

    let dense_solution = dense_factor.solve(&rhs).expect("complex dense solve");
    let sparse_solution = sparse_factor.solve(&rhs).expect("complex sparse solve");
    for (actual, expected) in dense_solution.iter().zip(&expected) {
        assert!((*actual - *expected).norm() <= 1.0e-12);
    }
    for (actual, expected) in sparse_solution.iter().zip(&expected) {
        assert!((*actual - *expected).norm() <= 1.0e-12);
    }

    let permutation = [2usize, 0, 3, 1];
    let permuted_matrix: Vec<Vec<_>> = permutation
        .iter()
        .map(|&row| {
            permutation
                .iter()
                .map(|&column| matrix[row][column])
                .collect()
        })
        .collect();
    let permuted_rhs: Vec<_> = permutation.iter().map(|&row| rhs[row]).collect();
    let permuted_expected: Vec<_> = permutation.iter().map(|&row| expected[row]).collect();
    let permuted_factor = factorize_sparse(
        &csr_from_dense(&permuted_matrix),
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .expect("permuted complex sparse LU setup");
    let permuted_solution = permuted_factor
        .solve(&permuted_rhs)
        .expect("permuted complex sparse solve");
    for (actual, expected) in permuted_solution.iter().zip(&permuted_expected) {
        assert!((*actual - *expected).norm() <= 1.0e-12);
    }
}

#[test]
fn f32_sparse_refinement_reports_monotone_value_update_residual() {
    // Factor the original matrix, then solve a same-pattern value update. The
    // first correction must be measured against the updated operator, and the
    // checked result must expose finite diagnostics rather than only a vector.
    let original = CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![1.0_f32, 0.2, 0.2, 1.1, 0.3, 0.3, 0.9],
    );
    let updated = CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![1.001_f32, 0.2, 0.2, 1.099, 0.3, 0.3, 0.901],
    );
    let expected = [1.0_f32, -2.0, 0.5];
    let rhs = updated.matvec(&expected);
    let factor = SparseLuFactorization::factorize(&original).expect("f32 factorization");
    let result = factor
        .solve_refined(&updated, &rhs, 1.0e-6, 6)
        .expect("same-pattern f32 value update refinement");

    assert!(result.initial_relative_residual.is_finite());
    assert!(result.relative_residual.is_finite());
    assert!(result.relative_residual <= result.initial_relative_residual + 1.0e-6);
    assert!(result.x.iter().all(|value| value.is_finite()));
    assert!(result.relative_residual <= 1.0e-5);
}

#[test]
fn f32_dense_and_sparse_lu_reconstruct_known_solution() {
    let matrix = vec![
        vec![3.0_f32, 1.0, -0.5],
        vec![0.25, 2.5, 1.0],
        vec![1.0, -0.25, 4.0],
    ];
    let expected = [1.0_f32, -2.0, 0.75];
    let rhs = dense_matvec(&matrix, &expected);
    let dense = DenseMatrix::from_row_major_vec(3, 3, matrix.iter().flatten().copied().collect());
    let dense_factor = factorize_dense(
        &dense,
        &DenseDirectConfig {
            matrix_class: MatrixClass::General,
            method: DenseDirectMethod::Lu,
            ..Default::default()
        },
    )
    .expect("f32 dense LU setup");
    let sparse_factor = factorize_sparse(
        &csr_from_dense(&matrix),
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .expect("f32 sparse LU setup");

    let dense_solution = dense_factor.solve(&rhs).expect("f32 dense LU solve");
    let sparse_solution = sparse_factor.solve(&rhs).expect("f32 sparse LU solve");
    for solution in [&dense_solution, &sparse_solution] {
        for (actual, expected) in solution.iter().zip(expected) {
            assert!(
                (*actual - expected).abs() <= 2.0e-5,
                "f32 solution mismatch: actual={actual}, expected={expected}"
            );
        }
        let residual = dense_matvec(&matrix, solution)
            .iter()
            .zip(&rhs)
            .map(|(actual, expected)| (*actual - *expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(residual <= 2.0e-5, "f32 residual={residual}");
    }
}

#[test]
fn sparse_lu_block_rhs_matches_single_rhs_solves() {
    let matrix = real_matrix();
    let x0 = [1.0, -0.5, 2.0];
    let x1 = [-2.0, 0.25, 1.5];
    let b0 = dense_matvec(&matrix, &x0);
    let b1 = dense_matvec(&matrix, &x1);
    let factor =
        SparseLuFactorization::factorize(&csr_from_dense(&matrix)).expect("sparse block-LU setup");

    let mut block_rhs = Vec::with_capacity(2 * matrix.len());
    for row in 0..matrix.len() {
        block_rhs.push(b0[row]);
        block_rhs.push(b1[row]);
    }
    let block_solution = factor
        .solve_many(&block_rhs, 2)
        .expect("sparse block-LU solve");
    let solution0 = factor.solve(&b0).expect("sparse single RHS 0");
    let solution1 = factor.solve(&b1).expect("sparse single RHS 1");
    let expected: Vec<_> = (0..matrix.len())
        .flat_map(|row| [solution0[row], solution1[row]])
        .collect();
    assert_close(&block_solution, &expected, 1.0e-12);
}

#[test]
fn sparse_lu_row_and_column_permutation_preserves_general_solution() {
    let matrix = real_matrix();
    let expected = [1.0, -2.0, 0.75];
    let rhs = dense_matvec(&matrix, &expected);
    let row_permutation = [2_usize, 0, 1];
    let column_permutation = [1_usize, 2, 0];
    let permuted_matrix: Vec<Vec<_>> = row_permutation
        .iter()
        .map(|&row| {
            column_permutation
                .iter()
                .map(|&column| matrix[row][column])
                .collect()
        })
        .collect();
    let permuted_rhs: Vec<_> = row_permutation.iter().map(|&row| rhs[row]).collect();
    let permuted_expected: Vec<_> = column_permutation
        .iter()
        .map(|&column| expected[column])
        .collect();
    let factor = factorize_sparse(
        &csr_from_dense(&permuted_matrix),
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .expect("permuted sparse LU setup");
    let solution = factor
        .solve(&permuted_rhs)
        .expect("permuted sparse LU solve");
    assert_close(&solution, &permuted_expected, 1.0e-12);
    let residual = dense_matvec(&permuted_matrix, &solution)
        .iter()
        .zip(&permuted_rhs)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    assert!(residual <= 1.0e-12, "permuted residual={residual}");
}

#[test]
fn dense_and_sparse_lu_reject_exactly_singular_matrix() {
    let matrix = vec![vec![1.0_f64, 2.0], vec![2.0, 4.0]];
    let dense = DenseMatrix::from_row_major_vec(2, 2, matrix.iter().flatten().copied().collect());
    assert!(
        factorize_dense(
            &dense,
            &DenseDirectConfig {
                matrix_class: MatrixClass::General,
                method: DenseDirectMethod::Lu,
                ..Default::default()
            },
        )
        .is_err(),
        "dense LU must reject a rank-deficient matrix"
    );
    assert!(
        factorize_sparse(
            &csr_from_dense(&matrix),
            &SparseDirectConfig {
                matrix_class: MatrixClass::General,
                method: SparseDirectMethod::SimplicialLu,
                ordering: Ordering::Auto,
                ..Default::default()
            },
            None,
        )
        .is_err(),
        "sparse LU must reject a rank-deficient matrix"
    );
}

#[test]
fn near_singular_dense_and_sparse_lu_report_finite_residual() {
    let matrix = vec![vec![1.0_f64, 1.0], vec![1.0, 1.0 + 1.0e-10]];
    let expected = [2.0, -1.0];
    let rhs = dense_matvec(&matrix, &expected);
    let dense = DenseMatrix::from_row_major_vec(2, 2, matrix.iter().flatten().copied().collect());
    let dense_factor = factorize_dense(
        &dense,
        &DenseDirectConfig {
            matrix_class: MatrixClass::General,
            method: DenseDirectMethod::Lu,
            ..Default::default()
        },
    )
    .expect("near-singular dense LU should be admitted");
    let sparse_factor = factorize_sparse(
        &csr_from_dense(&matrix),
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .expect("near-singular sparse LU should be admitted");
    for solution in [
        dense_factor.solve(&rhs).expect("near-singular dense solve"),
        sparse_factor
            .solve(&rhs)
            .expect("near-singular sparse solve"),
    ] {
        assert!(solution.iter().all(|value| value.is_finite()));
        let residual = dense_matvec(&matrix, &solution)
            .iter()
            .zip(&rhs)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(residual <= 1.0e-10, "near-singular residual={residual}");
    }
}

fn seeded_complex_hpd_matrix(seed: u64) -> Vec<Vec<Complex64>> {
    let n = 5;
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0
    };
    let mut lower = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for (row, entries) in lower.iter_mut().enumerate() {
        for entry in entries.iter_mut().take(row + 1) {
            *entry = Complex64::new(0.15 * next(), 0.15 * next());
        }
        entries[row] += Complex64::new(1.5 + 0.25 * next().abs(), 0.0);
    }
    let mut matrix = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for row in 0..n {
        for column in 0..n {
            let limit = (row.min(column)) + 1;
            matrix[row][column] = (0..limit)
                .map(|k| lower[row][k] * lower[column][k].conj())
                .sum();
            if row == column {
                matrix[row][column] += Complex64::new(0.5, 0.0);
            }
        }
    }
    matrix
}

#[test]
fn seeded_complex_hpd_factor_solve_matches_dense_sparse() {
    let mut max_solution_error = 0.0_f64;
    let mut max_residual = 0.0_f64;
    for seed in 0..8_u64 {
        let matrix = seeded_complex_hpd_matrix(0xC0DE_0000_u64.wrapping_add(seed));
        let expected: Vec<_> = (0..matrix.len())
            .map(|index| Complex64::new(0.5 + index as f64, -0.25 * index as f64))
            .collect();
        let rhs = dense_matvec(&matrix, &expected);
        let dense = DenseMatrix::from_row_major_vec(
            matrix.len(),
            matrix.len(),
            matrix.iter().flatten().copied().collect(),
        );
        let dense_factor = factorize_dense(
            &dense,
            &DenseDirectConfig {
                matrix_class: MatrixClass::HermitianPositiveDefinite,
                method: DenseDirectMethod::Cholesky,
                ..Default::default()
            },
        )
        .expect("seeded complex HPD dense factorization");
        let sparse_factor = factorize_sparse(
            &csr_from_dense(&matrix),
            &SparseDirectConfig {
                matrix_class: MatrixClass::HermitianPositiveDefinite,
                method: SparseDirectMethod::SimplicialCholesky,
                ordering: Ordering::Auto,
                ..Default::default()
            },
            None,
        )
        .expect("seeded complex HPD sparse factorization");
        let dense_solution = dense_factor.solve(&rhs).expect("seeded dense solve");
        let sparse_solution = sparse_factor.solve(&rhs).expect("seeded sparse solve");
        max_solution_error = max_solution_error.max(
            dense_solution
                .iter()
                .zip(&sparse_solution)
                .map(|(dense, sparse)| (*dense - *sparse).norm())
                .fold(0.0, f64::max),
        );
        max_residual = max_residual.max(
            dense_matvec(&matrix, &sparse_solution)
                .iter()
                .zip(&rhs)
                .map(|(actual, expected)| (*actual - *expected).norm())
                .fold(0.0, f64::max),
        );
    }
    assert!(
        max_solution_error <= 1.0e-11,
        "dense/sparse error={max_solution_error}"
    );
    assert!(
        max_residual <= 1.0e-11,
        "seeded HPD residual={max_residual}"
    );
}

#[test]
fn near_singular_lu_matrix_bank_preserves_finite_block_residuals() {
    let expected_columns = [[2.0_f64, -1.0_f64], [-0.5_f64, 1.25_f64]];
    for delta in [1.0e-6_f64, 1.0e-8, 1.0e-10] {
        let matrix = vec![vec![1.0, 1.0], vec![1.0, 1.0 + delta]];
        let dense = DenseMatrix::from_row_major_vec(2, 2, vec![1.0, 1.0, 1.0, 1.0 + delta]);
        let rhs_columns: Vec<Vec<f64>> = expected_columns
            .iter()
            .map(|column| dense_matvec(&matrix, column))
            .collect();
        let block_rhs: Vec<f64> = (0..2)
            .flat_map(|row| [rhs_columns[0][row], rhs_columns[1][row]])
            .collect();

        let dense_factor = factorize_dense(
            &dense,
            &DenseDirectConfig {
                matrix_class: MatrixClass::General,
                method: DenseDirectMethod::Lu,
                ..Default::default()
            },
        )
        .expect("near-singular dense LU setup");
        for rhs in &rhs_columns {
            let solution = dense_factor.solve(rhs).expect("near-singular dense solve");
            let residual = dense_matvec(&matrix, &solution)
                .iter()
                .zip(rhs)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            assert!(residual.is_finite() && residual <= 1.0e-8);
        }

        for (method, equilibrate) in [
            (SparseDirectMethod::SimplicialLu, false),
            (SparseDirectMethod::SupernodalLu, false),
            (SparseDirectMethod::SimplicialLu, true),
            (SparseDirectMethod::SupernodalLu, true),
        ] {
            let factor = factorize_sparse(
                &csr_from_dense(&matrix),
                &SparseDirectConfig {
                    matrix_class: MatrixClass::General,
                    method,
                    equilibrate,
                    ..Default::default()
                },
                None,
            )
            .unwrap_or_else(|error| panic!("near-singular factor setup failed: {error}"));
            let mut block_solution = vec![f64::NAN; block_rhs.len()];
            factor
                .solve_many_into(&block_rhs, &mut block_solution, 2)
                .expect("near-singular block solve");
            assert!(block_solution.iter().all(|value| value.is_finite()));
            for column in 0..2 {
                let solution = block_solution
                    .iter()
                    .skip(column)
                    .step_by(2)
                    .copied()
                    .collect::<Vec<_>>();
                let residual = dense_matvec(&matrix, &solution)
                    .iter()
                    .zip(&rhs_columns[column])
                    .map(|(actual, expected)| (actual - expected).abs())
                    .fold(0.0, f64::max);
                assert!(
                    residual.is_finite() && residual <= 1.0e-8,
                    "delta={delta:e} method={method:?} equilibrate={equilibrate} residual={residual:e}"
                );
            }
        }
    }
}
