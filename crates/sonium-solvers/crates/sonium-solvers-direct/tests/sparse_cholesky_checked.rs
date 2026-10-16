use num_complex::Complex64;
use sonium_solvers_core::traits::{AccuracyStatus, SolverStatus};
use sonium_solvers_core::traits::{AdjointCapability, Preconditioner};
use sonium_solvers_direct::{CholeskyOrdering, SparseCholeskyError, SparseCholeskyFactorization};
use sonium_solvers_sparse::CsrMatrix;

fn diagonal(value: f64) -> CsrMatrix<f64> {
    CsrMatrix::from_triplets(1, 1, vec![(0, 0, value)])
}

fn complex_hpd_fixture() -> (CsrMatrix<Complex64>, Vec<Complex64>) {
    let matrix = CsrMatrix::from_triplets(
        4,
        4,
        vec![
            (0, 0, Complex64::new(4.0, 0.0)),
            (0, 1, Complex64::new(1.0, 0.5)),
            (0, 3, Complex64::new(0.2, -0.1)),
            (1, 0, Complex64::new(1.0, -0.5)),
            (1, 1, Complex64::new(5.0, 0.0)),
            (1, 2, Complex64::new(0.75, -0.2)),
            (2, 1, Complex64::new(0.75, 0.2)),
            (2, 2, Complex64::new(4.5, 0.0)),
            (2, 3, Complex64::new(1.0, 0.3)),
            (3, 0, Complex64::new(0.2, 0.1)),
            (3, 2, Complex64::new(1.0, -0.3)),
            (3, 3, Complex64::new(3.5, 0.0)),
        ],
    );
    let expected = vec![
        Complex64::new(0.5, -0.75),
        Complex64::new(-1.0, 0.25),
        Complex64::new(1.25, 0.5),
        Complex64::new(0.75, -0.2),
    ];
    (matrix, expected)
}

#[test]
fn complex_hermitian_cholesky_orderings_match_independent_oracle() {
    let (matrix, expected) = complex_hpd_fixture();
    let rhs = matrix.matvec(&expected);

    for ordering in [
        CholeskyOrdering::Natural,
        CholeskyOrdering::Colamd,
        CholeskyOrdering::NestedDissection,
    ] {
        let factor = SparseCholeskyFactorization::factorize_with_ordering(&matrix, ordering, true)
            .unwrap_or_else(|error| panic!("Complex64 Cholesky failed for {ordering:?}: {error}"));
        let solution = factor.solve(&rhs).expect("Complex64 Cholesky solve");
        let solution_error = solution
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (*actual - *expected).norm())
            .fold(0.0, f64::max);
        let residual = matrix
            .matvec(&solution)
            .iter()
            .zip(&rhs)
            .map(|(actual, expected)| (*actual - *expected).norm())
            .fold(0.0, f64::max);
        assert!(
            solution_error < 1.0e-12,
            "{ordering:?}: solution error {solution_error:e}"
        );
        assert!(residual < 1.0e-12, "{ordering:?}: residual {residual:e}");
    }
}

#[test]
fn factorization_rejects_non_finite_matrix_entries() {
    let mut matrix = diagonal(1.0);
    matrix.values[0] = f64::NAN;

    assert!(matches!(
        SparseCholeskyFactorization::factorize(&matrix),
        Err(SparseCholeskyError::NonFiniteMatrix { row: 0, col: 0 })
    ));
    assert!(matches!(
        SparseCholeskyFactorization::factorize_supernodal(&matrix),
        Err(SparseCholeskyError::NonFiniteMatrix { row: 0, col: 0 })
    ));
}

#[test]
fn factorization_reports_generated_overflow() {
    let matrix = CsrMatrix::from_triplets(
        2,
        2,
        vec![
            (0, 0, 1.0e300),
            (0, 1, f64::MAX),
            (1, 0, f64::MAX),
            (1, 1, f64::MAX),
        ],
    );

    assert!(matches!(
        SparseCholeskyFactorization::factorize_with_ordering(
            &matrix,
            CholeskyOrdering::Natural,
            false,
        ),
        Err(SparseCholeskyError::NumericalBreakdown { .. })
    ));
    assert!(matches!(
        SparseCholeskyFactorization::factorize_supernodal_with_ordering(
            &matrix,
            CholeskyOrdering::Natural,
            false,
        ),
        Err(SparseCholeskyError::NumericalBreakdown { .. })
    ));
}

#[test]
fn solve_rejects_non_finite_rhs_without_mutating_destination() {
    let factor = SparseCholeskyFactorization::factorize(&diagonal(1.0)).unwrap();
    let mut destination = [42.0];

    assert!(matches!(
        factor.solve_into(&[f64::INFINITY], &mut destination),
        Err(SparseCholeskyError::NonFiniteRhs(0))
    ));
    assert_eq!(destination, [42.0]);
}

#[test]
fn solve_reports_finite_rhs_overflow_without_mutating_destination() {
    for factor in [
        SparseCholeskyFactorization::factorize(&diagonal(f64::MIN_POSITIVE)).unwrap(),
        SparseCholeskyFactorization::factorize_supernodal(&diagonal(f64::MIN_POSITIVE)).unwrap(),
    ] {
        let mut destination = [42.0];
        assert!(matches!(
            factor.solve_into(&[f64::MAX], &mut destination),
            Err(SparseCholeskyError::NumericalBreakdown {
                stage: "forward substitution",
                index: 0,
            })
        ));
        assert_eq!(destination, [42.0]);
    }
}

#[test]
fn solve_distinguishes_rhs_and_solution_dimensions() {
    let factor = SparseCholeskyFactorization::factorize(&diagonal(1.0)).unwrap();

    let mut destination = [42.0];
    let rhs_error = factor.solve_into(&[], &mut destination).unwrap_err();
    assert!(matches!(
        rhs_error,
        SparseCholeskyError::DimensionMismatch {
            expected: 1,
            actual: 0
        }
    ));
    assert_eq!(rhs_error.status(), SolverStatus::NumericalError);
    assert_eq!(rhs_error.accuracy_status(), AccuracyStatus::NotChecked);
    assert_eq!(destination, [42.0]);

    let mut destination = [42.0; 0];
    let solution_error = factor.solve_into(&[1.0], &mut destination).unwrap_err();
    assert!(matches!(
        solution_error,
        SparseCholeskyError::SolutionDimensionMismatch {
            expected: 1,
            actual: 0
        }
    ));
    assert_eq!(solution_error.status(), SolverStatus::NumericalError);
    assert_eq!(solution_error.accuracy_status(), AccuracyStatus::NotChecked);
}

#[test]
fn sparse_cholesky_multi_rhs_is_checked_and_row_major() {
    let matrix = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 2.0_f64), (1, 1, 4.0_f64)]);
    let factor = SparseCholeskyFactorization::factorize(&matrix).unwrap();
    let mut destination = [42.0; 4];
    factor
        .solve_many_into(&[2.0, 8.0, 4.0, 12.0], &mut destination, 2)
        .unwrap();
    for (actual, expected) in destination.iter().zip([1.0, 4.0, 1.0, 3.0]) {
        assert!((actual - expected).abs() < 1.0e-12);
    }

    let mut rejected = [42.0; 4];
    assert!(matches!(
        factor.solve_many_into(&[f64::NAN; 4], &mut rejected, 2),
        Err(SparseCholeskyError::NonFiniteRhs(0))
    ));
    assert_eq!(rejected, [42.0; 4]);

    assert!(matches!(
        factor.solve_many_into(&[], &mut [], usize::MAX),
        Err(SparseCholeskyError::SizeOverflow {
            dimension: 2,
            nrhs: usize::MAX,
        })
    ));
}

#[test]
fn complex_hpd_cholesky_exposes_exact_hermitian_preconditioner_action() {
    let (matrix, expected) = complex_hpd_fixture();
    let rhs = matrix.matvec(&expected);
    let factor = SparseCholeskyFactorization::factorize_hermitian(&matrix).unwrap();

    assert_eq!(factor.hermitian_capability(), AdjointCapability::Exact);
    assert_eq!(
        factor.transpose_capability(),
        AdjointCapability::Unsupported
    );
    let applied = factor.try_apply_hermitian(&rhs).unwrap();
    for (actual, expected) in applied.iter().zip(&expected) {
        assert!((*actual - *expected).norm() < 1e-11);
    }
    assert!(factor.try_apply_transpose(&rhs).is_err());
}

#[test]
fn real_spd_cholesky_exposes_exact_transpose_and_hermitian_actions() {
    let matrix = CsrMatrix::from_triplets(
        3,
        3,
        vec![
            (0, 0, 4.0_f64),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 3.0),
            (1, 2, 0.5),
            (2, 1, 0.5),
            (2, 2, 2.0),
        ],
    );
    let expected = [1.0, -0.5, 2.0];
    let rhs = matrix.matvec(&expected);
    let factor = SparseCholeskyFactorization::factorize(&matrix).unwrap();

    assert_eq!(factor.transpose_capability(), AdjointCapability::Exact);
    assert_eq!(factor.hermitian_capability(), AdjointCapability::Exact);
    let transpose = factor.try_apply_transpose(&rhs).unwrap();
    let hermitian = factor.try_apply_hermitian(&rhs).unwrap();
    for ((actual_transpose, actual_hermitian), expected) in
        transpose.iter().zip(&hermitian).zip(expected)
    {
        assert!((*actual_transpose - expected).abs() < 1.0e-12);
        assert!((*actual_hermitian - expected).abs() < 1.0e-12);
    }
}

#[test]
fn wave_dispatch_gate_is_bitwise_identical() {
    // Same contract as the sparse-LU gate test: dispatch-every-wave and
    // serialize-every-wave must agree bit-for-bit (disjoint in-wave
    // writes), and the block path must match the single-RHS path.
    let n = 1200;
    let mut triplets = Vec::with_capacity(3 * n);
    for row in 0..n {
        triplets.push((row, row, 4.0));
        if row > 0 {
            triplets.push((row, row - 1, -1.0));
        }
        if row + 1 < n {
            triplets.push((row, row + 1, -1.0));
        }
    }
    let matrix = CsrMatrix::from_triplets(n, n, triplets);
    let expected: Vec<f64> = (0..n).map(|i| (i as f64 * 0.013).sin()).collect();
    let rhs = matrix.matvec(&expected);
    let factor = SparseCholeskyFactorization::factorize(&matrix).expect("SPD factorization");

    const GATE: &str = "SONIUM_SPARSE_PAR_MIN_NNZ";
    // SAFETY: the gate only routes work between bitwise-identical paths, so
    // any value a concurrent test thread observes keeps its solve correct;
    // the var is restored below and no other test depends on its value.
    let ambient = std::env::var(GATE).ok();
    unsafe { std::env::set_var(GATE, "0") };
    let dispatched = factor.solve(&rhs).expect("dispatched solve");
    unsafe { std::env::set_var(GATE, &usize::MAX.to_string()) };
    let serial = factor.solve(&rhs).expect("serial solve");
    assert_eq!(dispatched.len(), serial.len());
    for (index, (&a, &b)) in dispatched.iter().zip(&serial).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "gate changed solution bits at index {index}"
        );
    }
    let block: Vec<f64> = rhs.iter().flat_map(|&v| [v, 2.0 * v]).collect();
    let mut output = vec![0.0; block.len()];
    factor
        .solve_many_into(&block, &mut output, 2)
        .expect("serial block solve");
    let doubled: Vec<f64> = rhs.iter().map(|v| 2.0 * v).collect();
    let single2 = factor.solve(&doubled).expect("scaled solve");
    for row in 0..n {
        assert_eq!(output[row * 2].to_bits(), serial[row].to_bits());
        assert_eq!(output[row * 2 + 1].to_bits(), single2[row].to_bits());
    }
    match ambient {
        Some(value) => unsafe { std::env::set_var(GATE, value) },
        None => unsafe { std::env::remove_var(GATE) },
    }
}
