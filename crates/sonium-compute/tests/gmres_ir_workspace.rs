//! Repeated mixed-precision solves retain scratch and certify the original system.

// Rust guideline compliant 2026-02-21
use num_complex::Complex64;
use sonium_compute::gmres::{BackendCsr, IdentityBackendPreconditioner};
use sonium_compute::gmres_ir::{
    GmresIrConfig, GmresIrWorkspace, gmres_ir_backend_preconditioned_with_workspace,
};
use sonium_compute::{Backend, CpuBackend, LowerPrecision, MemoryPool, MemoryPoolConfig};

struct RejectPreconditioner;

impl<B: Backend, T: sonium_compute::DeviceScalar> sonium_compute::DevicePreconditioner<B, T>
    for RejectPreconditioner
{
    fn apply(&self, _: &B, _: &B::Buffer<T>, _: &mut B::Buffer<T>) {
        panic!("checked solver must use try_apply");
    }
    fn try_apply(
        &self,
        backend: &B,
        _: &B::Buffer<T>,
        output: &mut B::Buffer<T>,
    ) -> sonium_compute::BackendResult<()> {
        use sonium_compute::DeviceBufferLike;
        backend.try_copy_h2d(
            &vec![T::from_real(<T::Real as num_traits::Float>::nan()); output.len()],
            output,
        )?;
        Err(sonium_compute::BackendError::OperationFailed {
            operation: "test_preconditioner",
            reason: "injected failure after scratch acquisition".into(),
        })
    }
}

fn repeated<B: Backend, T: LowerPrecision<Real = f64>>(
    pool: &MemoryPool<B>,
    diagonal: T,
    off: T,
    allocations: impl Fn() -> usize,
) {
    let backend = pool.backend();
    let mut workspace = GmresIrWorkspace::<B, T>::new(pool);
    // A dimension change must release old cached vectors and build matching storage.
    for n in [12_usize, 7] {
        let mut rows = vec![0];
        let mut columns = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            for j in i.saturating_sub(1)..=(i + 1).min(n - 1) {
                columns.push(j as i32);
                values.push(if i == j { diagonal } else { off });
            }
            rows.push(values.len() as i32);
        }
        let a = BackendCsr::try_from_host(backend, n, n, &rows, &columns, &values).unwrap();
        let lower: Vec<_> = values.iter().map(|v| v.cast_to_lower()).collect();
        let low = BackendCsr::try_from_host(backend, n, n, &rows, &columns, &lower).unwrap();
        let a = sonium_compute::PrebakedSpmv::try_from_csr(backend, &a).unwrap();
        let low = sonium_compute::PrebakedSpmv::try_from_csr(backend, &low).unwrap();
        let mut rhs = pool.try_alloc::<T>(n).unwrap();
        let mut x = pool.try_alloc::<T>(n).unwrap();
        let mut guess = pool.try_alloc::<T>(n).unwrap();
        let config = GmresIrConfig::<T> {
            max_outer_iterations: 80,
            restart_size: Some(2),
            max_inner_iterations: 2,
            outer_tolerance: 1e-10,
            true_residual_check_period: Some(1),
            ..Default::default()
        };
        for trial in 0..3 {
            let exact: Vec<_> = (0..n)
                .map(|i| T::from_real((i as f64 * 0.3).sin() + 0.2 + trial as f64 * 0.1))
                .collect();
            let b: Vec<_> = (0..n)
                .map(|i| {
                    (rows[i] as usize..rows[i + 1] as usize).fold(T::zero(), |sum, k| {
                        sum + values[k] * exact[columns[k] as usize]
                    })
                })
                .collect();
            backend.try_copy_h2d(&b, rhs.raw_mut()).unwrap();
            if trial == 0 {
                assert!(
                    gmres_ir_backend_preconditioned_with_workspace(
                        &mut workspace,
                        &a,
                        &low,
                        &RejectPreconditioner,
                        rhs.raw(),
                        None,
                        x.raw_mut(),
                        &config,
                    )
                    .is_err()
                );
            }
            let before = allocations();
            let status = gmres_ir_backend_preconditioned_with_workspace(
                &mut workspace,
                &a,
                &low,
                &IdentityBackendPreconditioner,
                rhs.raw(),
                None,
                x.raw_mut(),
                &config,
            )
            .unwrap();
            assert!(status.converged, "{status:?}");
            assert!(status.restart_cycles > 1);
            assert!(!status.true_residual_checks.is_empty());
            if trial > 0 {
                assert_eq!(allocations(), before, "repeated solve grew device scratch");
            }
            let mut actual = vec![T::zero(); n];
            backend.try_copy_d2h(x.raw(), &mut actual).unwrap();
            assert!(
                actual
                    .iter()
                    .zip(&exact)
                    .all(|(a, e)| (*a - *e).magnitude() < 1e-8)
            );
            backend.try_copy_d2d(x.raw(), guess.raw_mut()).unwrap();
            let status = gmres_ir_backend_preconditioned_with_workspace(
                &mut workspace,
                &a,
                &low,
                &IdentityBackendPreconditioner,
                rhs.raw(),
                Some(guess.raw()),
                x.raw_mut(),
                &config,
            )
            .unwrap();
            assert!(status.converged && status.outer_iterations == 0);
            let invalid = GmresIrConfig::<T> {
                max_outer_iterations: 0,
                ..config.clone()
            };
            assert!(
                gmres_ir_backend_preconditioned_with_workspace(
                    &mut workspace,
                    &a,
                    &low,
                    &IdentityBackendPreconditioner,
                    rhs.raw(),
                    None,
                    x.raw_mut(),
                    &invalid,
                )
                .is_err()
            );
        }
    }
    drop(workspace);
    assert_eq!(pool.bytes_in_use(), 0, "workspace did not return its quota");
}

#[test]
fn cpu_ir_workspace_restarts_changed_rhs_and_dimension() {
    let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(1 << 24)).unwrap();
    repeated(&pool, 3.0, -0.7, || pool.bytes_in_use());
    repeated(
        &pool,
        Complex64::new(3., 0.4),
        Complex64::new(-0.7, 0.1),
        || pool.bytes_in_use(),
    );
}

#[cfg(feature = "cuda")]
#[test]
fn native_ir_workspace_restarts_changed_rhs_and_dimension() {
    let backend = match sonium_compute::CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            assert!(std::env::var_os("SONIUM_REQUIRE_CUDA").is_none(), "{error}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 24)).unwrap();
    repeated(&pool, 3.0, -0.7, || {
        pool.backend().runtime_stats().allocation_operations
    });
    repeated(
        &pool,
        Complex64::new(3., 0.4),
        Complex64::new(-0.7, 0.1),
        || pool.backend().runtime_stats().allocation_operations,
    );
}

#[test]
fn ir_workspace_quota_failure_returns_retained_storage_on_drop() {
    let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(48)).unwrap();
    let backend = pool.backend();
    let a = BackendCsr::try_from_host(backend, 2, 2, &[0, 1, 2], &[0, 1], &[2_f64, 3.]).unwrap();
    let low = BackendCsr::try_from_host(backend, 2, 2, &[0, 1, 2], &[0, 1], &[2_f32, 3.]).unwrap();
    let mut rhs = pool.try_alloc::<f64>(2).unwrap();
    let mut x = pool.try_alloc::<f64>(2).unwrap();
    backend.try_copy_h2d(&[1., 2.], rhs.raw_mut()).unwrap();
    let before = pool.bytes_in_use();
    let mut workspace = GmresIrWorkspace::new(&pool);
    let error = gmres_ir_backend_preconditioned_with_workspace(
        &mut workspace,
        &a,
        &low,
        &IdentityBackendPreconditioner,
        rhs.raw(),
        None,
        x.raw_mut(),
        &GmresIrConfig::default(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        sonium_compute::gmres_ir::GmresIrError::Allocation { .. }
    ));
    assert!(pool.bytes_in_use() <= 48);
    drop(workspace);
    assert_eq!(pool.bytes_in_use(), before);
}
