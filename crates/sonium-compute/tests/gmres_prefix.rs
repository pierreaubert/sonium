//! Reserved Krylov capacity must not determine coefficient transfer volume.

// Rust guideline compliant 2026-02-21
use num_complex::{Complex32, Complex64};
use num_traits::{Float, FromPrimitive};
use sonium_compute::gmres::{
    GmresBackendConfig, GmresWorkspace, IdentityBackendPreconditioner,
    gmres_backend_preconditioned_pooled_with_workspace,
};
use sonium_compute::{
    Backend, CpuBackend, DeviceCsr, DeviceScalar, MemoryPool, MemoryPoolConfig, PrebakedSpmv,
};

fn check<B: Backend, T: DeviceScalar>(
    pool: &MemoryPool<B>,
    tolerance: f64,
    counters: impl Fn() -> Option<[usize; 3]>,
) {
    let backend = pool.backend();
    let real = |v| T::Real::from_f64(v).unwrap();
    let scalar = |v| T::from_real(real(v));
    let n = 257;
    let rows: Vec<_> = (0..=n as i32).collect();
    let columns: Vec<_> = (0..n as i32).collect();
    let diagonal: Vec<_> = (0..n).map(|i| scalar(2.0 + (i % 2) as f64)).collect();
    let matrix = DeviceCsr::try_from_host(backend, n, n, &rows, &columns, &diagonal).unwrap();
    let a = PrebakedSpmv::try_from_csr(backend, &matrix).unwrap();
    let mut b = backend.raw_alloc::<T>(n).unwrap();
    let mut output = backend.raw_alloc::<T>(n).unwrap();
    let mut workspace = GmresWorkspace::new(n);
    for (trial, budget) in [64, 4, 64].into_iter().enumerate() {
        let expected: Vec<_> = (0..n)
            .map(|i| scalar(1.0 + (i % 5) as f64 * 0.01 + trial as f64 * 0.1))
            .collect();
        let rhs: Vec<_> = expected
            .iter()
            .zip(&diagonal)
            .map(|(&x, &a)| x * a)
            .collect();
        backend.try_copy_h2d(&rhs, &mut b).unwrap();
        let before = counters();
        let status = gmres_backend_preconditioned_pooled_with_workspace(
            pool,
            &mut workspace,
            &a,
            &IdentityBackendPreconditioner,
            &b,
            None,
            &mut output,
            &GmresBackendConfig {
                max_iterations: budget,
                tolerance: real(tolerance),
            },
        )
        .unwrap();
        assert!(status.converged, "{status:?}");
        assert!(status.iterations <= 3);
        if let (Some(before), Some(after)) = (before, counters()) {
            if trial > 0 {
                assert_eq!(after[0], before[0]);
            }
            assert!(
                after[1] - before[1] <= 256,
                "coefficient upload scales with capacity: {before:?} -> {after:?}"
            );
            assert!(
                after[2] - before[2] <= 16 * std::mem::size_of::<T>(),
                "projection download scales with capacity: {before:?} -> {after:?}"
            );
        }
        let mut actual = vec![scalar(0.0); n];
        backend.try_copy_d2h(&output, &mut actual).unwrap();
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).norm_sqr().sqrt() < real(30.0 * tolerance));
        }
    }
}

#[test]
fn cpu_workspace_accepts_smaller_iteration_budget_all_scalars() {
    let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(1 << 24)).unwrap();
    check::<_, f32>(&pool, 1e-5, || None);
    check::<_, f64>(&pool, 1e-10, || None);
    check::<_, Complex32>(&pool, 1e-5, || None);
    check::<_, Complex64>(&pool, 1e-10, || None);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_prefix_transfers_ignore_reserved_capacity_all_scalars() {
    let backend = match sonium_compute::CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            assert!(
                std::env::var_os("SONIUM_REQUIRE_CUDA").is_none(),
                "CUDA required: {error}"
            );
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 24)).unwrap();
    let counters = || {
        let s = pool.backend().runtime_stats();
        Some([
            s.allocation_operations,
            s.transfer.h2d_bytes,
            s.transfer.d2h_bytes,
        ])
    };
    check::<_, f32>(&pool, 1e-5, counters);
    check::<_, f64>(&pool, 1e-10, counters);
    check::<_, Complex32>(&pool, 1e-5, counters);
    check::<_, Complex64>(&pool, 1e-10, counters);
}
