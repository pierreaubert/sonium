//! Prepared FGMRES retains variable actions and enforces explicit update policies.

// Rust guideline compliant 2026-02-21
use super::*;
use sonium_compute::{BackendError, BackendResult, CudaBuffer, CudaFailureInjectionPoint};
use std::sync::atomic::{AtomicUsize, Ordering};

struct ChangingScale(AtomicUsize);
impl<T: DeviceScalar> DevicePreconditioner<CudaBackend, T> for ChangingScale {
    fn apply(&self, backend: &CudaBackend, rhs: &CudaBuffer<T>, out: &mut CudaBuffer<T>) {
        self.try_apply(backend, rhs, out).unwrap();
    }
    fn try_apply(
        &self,
        backend: &CudaBackend,
        rhs: &CudaBuffer<T>,
        out: &mut CudaBuffer<T>,
    ) -> BackendResult<()> {
        let factor = if self.0.fetch_add(1, Ordering::Relaxed).is_multiple_of(2) {
            0.5
        } else {
            2.0
        };
        backend.try_copy_d2d(rhs, out)?;
        backend.try_scale(T::from_real(T::Real::from_f64(factor).unwrap()), out)
    }
}

fn all_scalar<T>(pool: &MemoryPool<CudaBackend>, tolerance: f64)
where
    T: ComplexField + DeviceScalar<Real = <T as ComplexField>::Real>,
    <T as ComplexField>::Real: FromPrimitive + Sync + Send,
{
    let scalar = |x| {
        <T as ComplexField>::from_real(
            <<T as ComplexField>::Real as FromPrimitive>::from_f64(x).unwrap(),
        )
    };
    let mut entries = Vec::new();
    for i in 0..12 {
        entries.push((i, i, scalar(if i % 2 == 0 { 3.0 } else { -4.0 })));
        if i + 1 < 12 {
            entries.push((i, i + 1, scalar(0.3)));
            entries.push((i + 1, i, scalar(-0.2)));
        }
    }
    let matrix = CsrMatrix::from_triplets(12, 12, entries);
    let controls = BackendKrylovConfig {
        max_iterations: 200,
        restart: 8,
        tolerance: <<T as ComplexField>::Real as FromPrimitive>::from_f64(tolerance).unwrap(),
        require_native: true,
        exact_residual_check_interval: 0,
    };
    let mut session = CudaLinearSolveSession::try_new_flexible(pool, &matrix, &controls, |_, _| {
        Ok(ChangingScale(AtomicUsize::new(0)))
    })
    .unwrap();
    assert_eq!(session.requested_strategy(), None);
    assert_eq!(session.selected_strategy(), None);
    let exact: Vec<_> = (0..12).map(|i| scalar(0.25 + (i as f64).sin())).collect();
    let rhs = matrix.apply(&exact);
    session.solve_host(&rhs, DeviceInitialGuess::Zero).unwrap();
    let before = pool.backend().begin_runtime_stats();
    let result = session.solve_host(&rhs, DeviceInitialGuess::Zero).unwrap();
    assert_eq!(
        pool.backend()
            .runtime_stats_since(before)
            .allocation_operations,
        0
    );
    assert_eq!(
        result.assumption_provenance,
        AssumptionProvenance::NotRequired
    );
    for (a, b) in result.x.iter().zip(&exact) {
        assert!((*a - *b).norm().to_f64().unwrap() < tolerance * 10.0);
    }
    let mut changed = matrix.clone();
    changed.values.iter_mut().for_each(|v| *v *= scalar(1.125));
    assert!(session.try_update_matrix(&changed).is_err());
    assert!(session.solution_device().is_none());
    session.solve_host(&rhs, DeviceInitialGuess::Zero).unwrap();
    let before = pool.backend().begin_runtime_stats();
    session.try_update_matrix_lagged(&changed).unwrap();
    assert_eq!(
        pool.backend()
            .runtime_stats_since(before)
            .allocation_operations,
        0
    );
    assert!(session.solution_device().is_none());
    session
        .solve_host(&changed.apply(&exact), DeviceInitialGuess::PreviousSolution)
        .unwrap();
    session
        .try_update_matrix_with_preconditioner(&matrix, |_, _| {
            Ok(ChangingScale(AtomicUsize::new(0)))
        })
        .unwrap();
    session
        .solve_host(&rhs, DeviceInitialGuess::PreviousSolution)
        .unwrap();
}

fn pool() -> Option<MemoryPool<CudaBackend>> {
    match CudaBackend::new(0) {
        Ok(b) => Some(MemoryPool::new(b, MemoryPoolConfig::with_cap(1 << 26)).unwrap()),
        Err(e) => {
            assert!(
                std::env::var_os("SONIUM_REQUIRE_CUDA").is_none(),
                "CUDA required: {e}"
            );
            None
        }
    }
}

#[test]
fn native_cuda_prepared_fgmres_variable_action_all_scalars() {
    let Some(pool) = pool() else {
        return;
    };
    all_scalar::<f32>(&pool, 2e-5);
    all_scalar::<f64>(&pool, 1e-10);
    all_scalar::<Complex32>(&pool, 2e-5);
    all_scalar::<Complex64>(&pool, 1e-10);
}

#[test]
fn native_cuda_prepared_policy_rejection_and_recovery() {
    let Some(pool) = pool() else {
        return;
    };
    let matrix = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 2.0_f64), (1, 1, -3.0)]);
    let controls = BackendKrylovConfig {
        max_iterations: 20,
        restart: 2,
        tolerance: 1e-10,
        require_native: true,
        exact_residual_check_interval: 0,
    };
    let mut calls = 0;
    let mut invalid = controls.clone();
    invalid.require_native = false;
    assert!(
        CudaLinearSolveSession::try_new_flexible(&pool, &matrix, &invalid, |_, _| {
            calls += 1;
            Ok(IdentityDevicePreconditioner)
        })
        .is_err()
    );
    assert_eq!(calls, 0);
    let mut session =
        CudaLinearSolveSession::try_new_flexible(&pool, &matrix, &controls, |_, _| {
            Ok(IdentityDevicePreconditioner)
        })
        .unwrap();
    session
        .solve_host(&[2.0, -3.0], DeviceInitialGuess::Zero)
        .unwrap();
    assert!(
        session
            .try_update_matrix_with_preconditioner::<IdentityDevicePreconditioner>(
                &matrix,
                |_, _| Err(BackendError::InvalidArgument {
                    operation: "test_prepare",
                    reason: "setup failed".into()
                })
            )
            .is_err()
    );
    assert!(session.solution_device().is_none());
    session
        .solve_host(&[2.0, -3.0], DeviceInitialGuess::Zero)
        .unwrap();
    pool.backend()
        .inject_failure(CudaFailureInjectionPoint::TransferH2D);
    assert!(session.try_update_matrix_lagged(&matrix).is_err());
    pool.backend().recover_after_error().unwrap();
    assert!(
        session
            .solve_host(&[2.0, -3.0], DeviceInitialGuess::Zero)
            .is_err()
    );
    session
        .try_update_matrix_with_preconditioner(&matrix, |_, _| Ok(IdentityDevicePreconditioner))
        .unwrap();
    assert!(
        session
            .solve_host(&[2.0, -3.0], DeviceInitialGuess::PreviousSolution)
            .is_err()
    );
    session
        .solve_host(&[2.0, -3.0], DeviceInitialGuess::Zero)
        .unwrap();
    let changed = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 2.0_f64), (0, 1, 1.0), (1, 1, -3.0)]);
    assert!(
        session
            .try_update_matrix_with_preconditioner(&changed, |_, _| {
                calls += 1;
                Ok(IdentityDevicePreconditioner)
            })
            .is_err()
    );
    assert_eq!(calls, 0);
}
