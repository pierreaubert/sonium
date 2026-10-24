use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::{AccuracyStatus, SolverFailure, SolverStatus};
use sonium_solvers_direct::PanelKind;
use sonium_solvers_direct::multifrontal_blr::FrontError;
use sonium_solvers_direct::multifrontal_blr::front::{FactoredFront, PivotKind};
use sonium_solvers_direct::multifrontal_blr::panel::Panel;
use sonium_solvers_direct::{
    BlrConfig, BlrError, LdltBlrConfig, LdltConfig, MultifrontalBlrFactorization,
    MultifrontalLdltFactorization, MultifrontalSymbolic, PivotStrategy,
};
use sonium_solvers_sparse::CsrMatrix;

fn dense_csr(values: Vec<f64>, n: usize) -> CsrMatrix<f64> {
    let dense = DenseMatrix::from_row_major_vec(n, n, values);
    CsrMatrix::from_dense(&dense, 0.0)
}

#[test]
fn factorization_rejects_non_finite_matrix_and_configuration() {
    let mut non_finite = dense_csr(vec![1.0], 1);
    non_finite.values[0] = f64::NAN;
    assert!(matches!(
        MultifrontalBlrFactorization::factorize(&non_finite),
        Err(BlrError::NonFiniteMatrix { index: 0 })
    ));
    assert!(matches!(
        MultifrontalLdltFactorization::factorize(&non_finite),
        Err(BlrError::NonFiniteMatrix { index: 0 })
    ));
    assert!(matches!(
        MultifrontalSymbolic::analyze(&non_finite, &BlrConfig::default()),
        Err(BlrError::NonFiniteMatrix { index: 0 })
    ));

    let identity = dense_csr(vec![1.0], 1);
    let blr_config = BlrConfig {
        epsilon: f64::NAN,
        ..BlrConfig::default()
    };
    assert!(matches!(
        MultifrontalBlrFactorization::factorize_with_config(&identity, &blr_config),
        Err(BlrError::InvalidConfiguration { field: "epsilon" })
    ));

    let ldlt_config = LdltConfig {
        pivot_tol_multiplier: f64::INFINITY,
        ..LdltConfig::default()
    };
    assert!(matches!(
        MultifrontalLdltFactorization::factorize_with_config(&identity, &ldlt_config),
        Err(BlrError::InvalidConfiguration {
            field: "pivot_tol_multiplier"
        })
    ));
}

#[test]
fn blr_rejects_configuration_that_overflows_scalar_precision() {
    let matrix = CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f32, 2.0]);
    let config = BlrConfig {
        epsilon: f64::MAX,
        ..BlrConfig::default()
    };
    assert!(matches!(
        MultifrontalBlrFactorization::<f32>::factorize_with_config(&matrix, &config),
        Err(BlrError::InvalidConfiguration { field: "epsilon" })
    ));

    let config = BlrConfig {
        pivot_tol_multiplier: f64::MAX,
        ..BlrConfig::default()
    };
    assert!(matches!(
        MultifrontalBlrFactorization::<f32>::factorize_with_config(&matrix, &config),
        Err(BlrError::InvalidConfiguration {
            field: "pivot_tol_multiplier"
        })
    ));
}

#[test]
fn ldlt_rejects_configuration_that_overflows_scalar_precision() {
    let matrix = CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f32, 2.0]);
    let config = LdltConfig {
        pivot_tol_multiplier: f64::MAX,
        ..LdltConfig::default()
    };
    assert!(matches!(
        MultifrontalLdltFactorization::<f32>::factorize_with_config(&matrix, &config),
        Err(BlrError::InvalidConfiguration {
            field: "pivot_tol_multiplier"
        })
    ));

    let config = LdltConfig {
        blr: Some(LdltBlrConfig {
            epsilon: f64::MAX,
            ..LdltBlrConfig::default()
        }),
        ..LdltConfig::default()
    };
    assert!(matches!(
        MultifrontalLdltFactorization::<f32>::factorize_with_config(&matrix, &config),
        Err(BlrError::InvalidConfiguration {
            field: "blr.epsilon"
        })
    ));
}

#[test]
fn checked_solves_reject_bad_rhs_before_traversal() {
    let matrix = dense_csr(vec![2.0, 0.0, 0.0, 3.0], 2);
    let blr = MultifrontalBlrFactorization::factorize(&matrix).unwrap();
    let ldlt = MultifrontalLdltFactorization::factorize(&matrix).unwrap();

    for result in [blr.try_solve(&[1.0]), ldlt.try_solve(&[1.0])] {
        assert!(matches!(
            result,
            Err(BlrError::DimensionMismatch {
                expected: 2,
                actual: 1
            })
        ));
    }
    for result in [
        blr.try_solve(&[f64::INFINITY, 1.0]),
        ldlt.try_solve(&[f64::INFINITY, 1.0]),
    ] {
        assert!(matches!(result, Err(BlrError::NonFiniteRhs { index: 0 })));
    }
}

#[test]
fn finite_factorization_overflow_is_typed() {
    let max = f64::MAX;
    let matrix = dense_csr(vec![max, max, max, -max], 2);
    let blr_config = BlrConfig {
        pivot_tol_multiplier: 0.0,
        ..BlrConfig::default()
    };
    assert!(matches!(
        MultifrontalBlrFactorization::factorize_with_config(&matrix, &blr_config),
        Err(BlrError::FrontNumericalBreakdown {
            stage: "frontal factorization",
            ..
        })
    ));
    assert!(matches!(
        MultifrontalLdltFactorization::factorize(&matrix),
        Err(BlrError::FrontNumericalBreakdown {
            stage: "symmetric frontal factorization",
            ..
        })
    ));
}

#[test]
fn finite_rhs_overflow_is_typed() {
    let matrix = dense_csr(vec![f64::MIN_POSITIVE], 1);
    let blr = MultifrontalBlrFactorization::factorize(&matrix).unwrap();
    let ldlt = MultifrontalLdltFactorization::factorize(&matrix).unwrap();

    assert!(matches!(
        blr.try_solve(&[f64::MAX]),
        Err(BlrError::FrontNumericalBreakdown {
            stage: "front back substitution",
            ..
        })
    ));
    assert!(matches!(
        ldlt.try_solve(&[f64::MAX]),
        Err(BlrError::FrontNumericalBreakdown {
            stage: "D one-by-one solve",
            ..
        })
    ));
}

#[test]
fn checked_two_by_two_pivot_solves_do_not_recompute_through_expect() {
    let matrix = dense_csr(vec![0.0, 1.0, 1.0, 0.0], 2);
    let blr_config = BlrConfig {
        pivot_strategy: PivotStrategy::BunchKaufman,
        pivot_tol_multiplier: 0.0,
        ..BlrConfig::default()
    };
    let blr = MultifrontalBlrFactorization::factorize_with_config(&matrix, &blr_config).unwrap();
    let ldlt = MultifrontalLdltFactorization::factorize(&matrix).unwrap();

    let blr_solution = blr.try_solve(&[1.0, 2.0]).unwrap();
    let ldlt_solution = ldlt.try_solve(&[1.0, 2.0]).unwrap();
    for solution in [blr_solution, ldlt_solution] {
        assert!((solution[0] - 2.0).abs() < 1e-12);
        assert!((solution[1] - 1.0).abs() < 1e-12);
    }
}

#[test]
fn public_front_apply_wrappers_report_bad_vector_dimensions() {
    let zero = DenseMatrix::<f64>::zeros(1, 1);
    let front = FactoredFront {
        nfs: 1,
        ncb: 1,
        fs_elim_indices: vec![0],
        f11_lu: DenseMatrix::from_row_major_vec(1, 1, vec![1.0]),
        pivot_kinds: vec![PivotKind::OneByOne],
        pivots: vec![0],
        col_pivots: vec![0],
        l21: Panel::from_dense(&zero, PanelKind::Blr, 1, 0.0),
        u12: Panel::from_dense(&zero, PanelKind::Blr, 1, 0.0),
        cb_indices: vec![1],
        cb_panel: Panel::from_dense(&zero, PanelKind::Blr, 1, 0.0),
        ndelayed: 0,
        delayed_row_indices: Vec::new(),
        delayed_col_indices: Vec::new(),
        delayed: None,
    };

    let mut rhs = Vec::new();
    assert!(matches!(
        front.try_forward_apply(&mut rhs),
        Err(FrontError::Invalid("front vector length mismatch"))
    ));
    assert!(matches!(
        front.try_backward_apply(&mut rhs),
        Err(FrontError::Invalid("front vector length mismatch"))
    ));
}

#[test]
fn front_error_exposes_checked_failure_contract() {
    let invalid = FrontError::Invalid("front vector length mismatch");
    assert_eq!(
        invalid.error_kind(),
        sonium_solvers_core::traits::SolverErrorKind::OperationFailed
    );
    assert_eq!(invalid.status(), SolverStatus::NumericalError);
    assert_eq!(invalid.solver_status(), SolverStatus::NumericalError);
    assert_eq!(invalid.accuracy_status(), AccuracyStatus::NotChecked);
    assert_eq!(invalid.attainable_accuracy(), AccuracyStatus::NotChecked);

    let numerical = FrontError::Numerical {
        stage: "front back substitution",
        index: 3,
    };
    assert_eq!(
        numerical.error_kind(),
        sonium_solvers_core::traits::SolverErrorKind::Breakdown
    );
    assert_eq!(numerical.status(), SolverStatus::Breakdown);
    assert_eq!(numerical.accuracy_status(), AccuracyStatus::NotChecked);
}
