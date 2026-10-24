//! Persistent CSR residuals retain descriptors and propagate checked failures.

// Rust guideline compliant 2026-02-21
use num_complex::{Complex32, Complex64};
use num_traits::{FromPrimitive, ToPrimitive};
use sonium_compute::{
    Backend, CpuBackend, DeviceOperator, DeviceScalar, HostCsrRef, MemoryPool, MemoryPoolConfig,
    PersistentDeviceCsr,
};

fn check<B: Backend, T: DeviceScalar + PartialEq>(pool: &MemoryPool<B>, values: [T; 4], tolerance: f64) {
    let backend = pool.backend();
    let scalar = |v| T::from_real(T::Real::from_f64(v).unwrap());
    let mut matrix = PersistentDeviceCsr::try_from_host_csr_ref(
        backend,
        HostCsrRef::try_new(2, 3, &[0, 2, 4], &[0, 2, 1, 2], &values).unwrap(),
    )
    .unwrap();
    let mut x = pool.try_alloc(3).unwrap();
    let mut rhs = pool.try_alloc(2).unwrap();
    let mut out = pool.try_alloc(2).unwrap();
    let xv = [scalar(1.0), scalar(-2.0), scalar(0.25)];
    let bv = [scalar(0.2), scalar(-0.3)];
    backend.try_copy_h2d(&xv, x.raw_mut()).unwrap();
    backend.try_copy_h2d(&bv, rhs.raw_mut()).unwrap();
    matrix
        .try_residual(backend, x.raw(), rhs.raw(), out.raw_mut())
        .unwrap();
    for factor in [scalar(1.0), scalar(-0.5)] {
        let v = values.map(|a| a * factor);
        matrix.try_copy_values_from_host(backend, &v).unwrap();
        let before = backend.transfer_stats();
        let memory = backend.memory_stats();
        DeviceOperator::try_residual(&matrix, backend, x.raw(), rhs.raw(), out.raw_mut()).unwrap();
        assert_eq!(backend.transfer_stats().h2d_bytes, before.h2d_bytes);
        assert_eq!(backend.transfer_stats().d2h_bytes, before.d2h_bytes);
        assert_eq!(backend.memory_stats(), memory);
        let expected = [
            bv[0] - (v[0] * xv[0] + v[1] * xv[2]),
            bv[1] - (v[2] * xv[1] + v[3] * xv[2]),
        ];
        let mut actual = [T::zero(); 2];
        backend.try_copy_d2h(out.raw(), &mut actual).unwrap();
        for (a, e) in actual.into_iter().zip(expected) {
            assert!((a - e).magnitude().to_f64().unwrap() < tolerance);
        }
    }
    let bad = pool.try_alloc::<T>(1).unwrap();
    let mut before = [T::zero(); 2];
    backend.try_copy_d2h(out.raw(), &mut before).unwrap();
    assert!(
        DeviceOperator::try_residual(&matrix, backend, x.raw(), bad.raw(), out.raw_mut()).is_err()
    );
    assert!(
        DeviceOperator::try_residual(&matrix, backend, bad.raw(), rhs.raw(), out.raw_mut())
            .is_err()
    );
    let mut actual = [T::zero(); 2];
    backend.try_copy_d2h(out.raw(), &mut actual).unwrap();
    assert_eq!(actual, before);
}

fn all_scalars<B: Backend>(pool: &MemoryPool<B>) {
    check(pool, [2.0_f32, 0.5, -0.25, 3.0], 1e-6);
    check(pool, [2.0_f64, 0.5, -0.25, 3.0], 1e-14);
    check(
        pool,
        [
            Complex32::new(2.0, 0.3),
            Complex32::new(0.5, -0.1),
            Complex32::new(-0.25, 0.7),
            Complex32::new(3.0, -0.8),
        ],
        1e-6,
    );
    check(
        pool,
        [
            Complex64::new(2.0, 0.3),
            Complex64::new(0.5, -0.1),
            Complex64::new(-0.25, 0.7),
            Complex64::new(3.0, -0.8),
        ],
        1e-14,
    );
}

#[test]
fn cpu_persistent_residual_all_scalars_and_numeric_updates() {
    let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(1 << 20)).unwrap();
    all_scalars(&pool);
}

#[cfg(feature = "cuda")]
#[test]
fn native_persistent_residual_all_scalars_and_checked_recovery() {
    use sonium_compute::{CudaBackend, CudaFailureInjectionPoint};
    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            assert!(
                std::env::var_os("SONIUM_REQUIRE_CUDA").is_none(),
                "CUDA required: {e}"
            );
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 24)).unwrap();
    all_scalars(&pool);
    let backend = pool.backend();
    let matrix = PersistentDeviceCsr::try_from_host_diagonal(backend, &[2.0_f64; 3]).unwrap();
    let mut x = pool.try_alloc(3).unwrap();
    let mut rhs = pool.try_alloc(3).unwrap();
    let mut out = pool.try_alloc(3).unwrap();
    backend.try_copy_h2d(&[3.0; 3], x.raw_mut()).unwrap();
    backend.try_copy_h2d(&[2.0; 3], rhs.raw_mut()).unwrap();
    matrix
        .try_residual(backend, x.raw(), rhs.raw(), out.raw_mut())
        .unwrap();
    let before = backend.begin_runtime_stats();
    DeviceOperator::try_residual(&matrix, backend, x.raw(), rhs.raw(), out.raw_mut()).unwrap();
    let stats = backend.runtime_stats_since(before);
    assert_eq!(stats.kernel_launches, 1);
    assert_eq!(stats.library_operations, 1);
    assert_eq!(stats.allocation_operations, 0);
    assert_eq!(stats.transfer.h2d_bytes, 0);
    assert_eq!(stats.transfer.d2h_bytes, 0);
    backend.inject_failure(CudaFailureInjectionPoint::Launch);
    assert!(
        DeviceOperator::try_residual(&matrix, backend, x.raw(), rhs.raw(), out.raw_mut()).is_err()
    );
    backend.recover_after_error().unwrap();
    DeviceOperator::try_residual(&matrix, backend, x.raw(), rhs.raw(), out.raw_mut()).unwrap();
    let mut actual = [0.0; 3];
    backend.try_copy_d2h(out.raw(), &mut actual).unwrap();
    assert_eq!(actual, [-4.0; 3]);
}
