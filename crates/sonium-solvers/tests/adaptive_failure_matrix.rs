//! Failure-exit coverage for adaptive and continuation Krylov entry points.
//!
//! These methods historically had strong success-path coverage but no common
//! assertion that a zero operator with a nonzero RHS remains typed and
//! non-converged instead of publishing a false success or panicking.

#![expect(
    deprecated,
    reason = "the failure matrix covers the deprecated recycled-GMRES entry point alongside its replacement"
)]
use std::panic::{AssertUnwindSafe, catch_unwind};

use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::iterative::try_gmres_recycled_preconditioned;
use sonium_solvers::traits::Preconditioner;
use sonium_solvers::{
    BlockFgmresConfig, ComplexField, CsrMatrix, DeflationSubspace, GcroDrConfig, GcroDrSubspace,
    GmresConfig, GmresDrConfig, IdentityPreconditioner, RecycledSubspace, SolverResultContract,
    SolverStatus, try_block_fgmres_preconditioned, try_gcro_dr, try_gmres_deflated,
    try_gmres_dr_preconditioned, try_gmres_recycled,
};

fn zero_operator() -> CsrMatrix<f64> {
    CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![0.0, 0.0])
}

fn gmres_config() -> GmresConfig<f64> {
    GmresConfig {
        max_iterations: 1,
        restart: 2,
        tolerance: 1.0e-12,
        ..GmresConfig::default()
    }
}

fn assert_non_success<T: ComplexField>(
    result: Result<sonium_solvers::GmresSolution<T>, sonium_solvers::SolverError>,
) {
    match result {
        Ok(solution) => {
            assert_ne!(solution.status, SolverStatus::Converged);
            assert_ne!(
                solution.attainable_accuracy,
                sonium_solvers::AccuracyStatus::Attained
            );
            solution
                .validate_contract()
                .expect("non-success result remains contract-valid");
        }
        Err(_) => {}
    }
}

#[test]
fn adaptive_krylov_failure_matrix_is_panic_free_and_non_success() {
    let operator = zero_operator();
    let rhs = [1.0, -2.0];
    let preconditioner = IdentityPreconditioner;
    let config = gmres_config();

    let gmres_dr = catch_unwind(AssertUnwindSafe(|| {
        try_gmres_dr_preconditioned(
            &operator,
            &preconditioner,
            &rhs,
            None,
            &GmresDrConfig {
                gmres: config.clone(),
                num_deflation_vectors: 1,
            },
        )
    }))
    .expect("GMRES-DR failure path must not panic");
    assert_non_success(gmres_dr);

    let mut gcro_subspace = GcroDrSubspace::new(1);
    let gcro = catch_unwind(AssertUnwindSafe(|| {
        try_gcro_dr(
            &operator,
            &rhs,
            None,
            &mut gcro_subspace,
            &GcroDrConfig::from_gmres(&config),
        )
    }))
    .expect("GCRO-DR failure path must not panic");
    assert_non_success(gcro);

    let mut recycled_subspace = RecycledSubspace::new(1);
    let recycled = catch_unwind(AssertUnwindSafe(|| {
        try_gmres_recycled(&operator, &rhs, None, &mut recycled_subspace, &config)
    }))
    .expect("recycled GMRES failure path must not panic");
    assert_non_success(recycled);

    let deflation = DeflationSubspace::try_new(Vec::new(), &operator)
        .expect("empty deflation space is a valid zero-rank fixture");
    let deflated = catch_unwind(AssertUnwindSafe(|| {
        try_gmres_deflated(&operator, &deflation, &rhs, None, &config)
    }))
    .expect("deflated GMRES failure path must not panic");
    assert_non_success(deflated);

    let block_rhs = DenseMatrix::from_row_major_vec(2, 2, vec![1.0, 0.0, 0.0, -2.0]);
    let block = catch_unwind(AssertUnwindSafe(|| {
        try_block_fgmres_preconditioned(
            &operator,
            &block_rhs,
            &preconditioner,
            None,
            &BlockFgmresConfig {
                max_iterations: 1,
                restart: 1,
                tolerance: 1.0e-12,
                true_residual_check: true,
                exact_residual_check_interval: 0,
            },
        )
    }))
    .expect("block FGMRES failure path must not panic");
    match block {
        Ok(solution) => {
            assert!(!solution.converged);
            solution
                .validate_contract()
                .expect("block failure result contract");
        }
        Err(_) => {}
    }
}

struct NonFinitePreconditioner;

impl Preconditioner<f64> for NonFinitePreconditioner {
    fn apply_into(&self, input: &[f64], output: &mut [f64]) {
        output.copy_from_slice(input);
        output[0] = f64::NAN;
    }

    fn apply_transpose_into(&self, input: &[f64], output: &mut [f64]) {
        self.apply_into(input, output);
    }
}

#[test]
#[allow(deprecated)]
fn recycled_failure_does_not_mutate_caller_state() {
    let operator = CsrMatrix::identity(2);
    let config = GmresConfig {
        max_iterations: 4,
        restart: 2,
        tolerance: 1.0e-10,
        ..GmresConfig::default()
    };
    let mut subspace = RecycledSubspace::new(1);
    let seed = try_gmres_recycled(&operator, &[1.0, 2.0], None, &mut subspace, &config)
        .expect("identity seed solve");
    assert!(seed.converged);
    let before = subspace
        .to_checkpoint_bytes()
        .expect("serialize seed state");

    let error = try_gmres_recycled_preconditioned(
        &operator,
        &NonFinitePreconditioner,
        &[1.0, 2.0],
        None,
        &mut subspace,
        &config,
    )
    .expect_err("non-finite preconditioner output must be typed");
    assert!(matches!(
        error,
        sonium_solvers::SolverError::OperationFailed { .. }
    ));
    let after = subspace
        .to_checkpoint_bytes()
        .expect("serialize failure state");
    assert_eq!(before, after, "typed failure must preserve recycled state");
}
