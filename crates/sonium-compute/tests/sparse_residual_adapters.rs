//! Sparse adapter residuals preserve complex actions and retained CUDA storage.

// Rust guideline compliant 2026-02-21
use num_complex::{Complex32, Complex64};
use num_traits::{FromPrimitive, ToPrimitive};
use sonium_compute::{
    AdjointDeviceCsr, Backend, CpuBackend, DeviceCsr, DeviceOperator, DeviceScalar, HostCsrRef,
    MemoryPool, MemoryPoolConfig, PrebakedSpmv,
};

fn check<B: Backend, T: DeviceScalar + PartialEq>(
    pool: &MemoryPool<B>,
    values: [T; 4],
    tolerance: f64,
) {
    let backend = pool.backend();
    let scalar = |v| T::from_real(T::Real::from_f64(v).unwrap());
    let host = HostCsrRef::try_new(2, 2, &[0, 2, 4], &[0, 1, 0, 1], &values).unwrap();
    let matrix = DeviceCsr::try_from_host_csr_ref(backend, host).unwrap();
    let pooled = DeviceCsr::try_from_pool(pool, 2, 2, &[0, 2, 4], &[0, 1, 0, 1], &values).unwrap();
    let pair = AdjointDeviceCsr::try_from_host_csr_ref(backend, host).unwrap();
    let prebaked = PrebakedSpmv::try_from_csr(backend, &matrix).unwrap();
    let pooled_prebaked = PrebakedSpmv::try_from_pooled(backend, &pooled).unwrap();
    let paired_prebaked = pair.try_prebake(backend).unwrap();
    let hermitian = PrebakedSpmv::try_from_csr(backend, pair.hermitian()).unwrap();
    let xv = [scalar(0.25), scalar(-2.0)];
    let bv = [scalar(0.3), scalar(-0.7)];
    let forward = [
        bv[0] - (values[0] * xv[0] + values[1] * xv[1]),
        bv[1] - (values[2] * xv[0] + values[3] * xv[1]),
    ];
    let adjoint = [
        bv[0] - (values[0].conj_mul(xv[0]) + values[2].conj_mul(xv[1])),
        bv[1] - (values[1].conj_mul(xv[0]) + values[3].conj_mul(xv[1])),
    ];
    let mut x = pool.try_alloc(2).unwrap();
    let mut rhs = pool.try_alloc(2).unwrap();
    let mut out = pool.try_alloc(2).unwrap();
    let bad = pool.try_alloc::<T>(1).unwrap();
    backend.try_copy_h2d(&xv, x.raw_mut()).unwrap();
    backend.try_copy_h2d(&bv, rhs.raw_mut()).unwrap();
    let operators = [
        (&matrix as &dyn DeviceOperator<B, T>, false, forward),
        (&pooled, false, forward),
        (&pair, false, forward),
        (&prebaked, true, forward),
        (&pooled_prebaked, true, forward),
        (&paired_prebaked, true, forward),
        (&hermitian, true, adjoint),
    ];
    for (operator, retained, expected) in operators {
        operator
            .try_residual(backend, x.raw(), rhs.raw(), out.raw_mut())
            .unwrap();
        let memory = backend.memory_stats();
        let transfer = backend.transfer_stats();
        operator
            .try_residual(backend, x.raw(), rhs.raw(), out.raw_mut())
            .unwrap();
        if retained {
            assert_eq!(backend.memory_stats(), memory);
            assert_eq!(backend.transfer_stats().h2d_bytes, transfer.h2d_bytes);
            assert_eq!(backend.transfer_stats().d2h_bytes, transfer.d2h_bytes);
        }
        let mut actual = [T::zero(); 2];
        backend.try_copy_d2h(out.raw(), &mut actual).unwrap();
        for (a, e) in actual.into_iter().zip(expected) {
            assert!((a - e).magnitude().to_f64().unwrap() < tolerance);
        }
        let before = actual;
        assert!(
            operator
                .try_residual(backend, x.raw(), bad.raw(), out.raw_mut())
                .is_err()
        );
        assert!(
            operator
                .try_residual(backend, bad.raw(), rhs.raw(), out.raw_mut())
                .is_err()
        );
        backend.try_copy_d2h(out.raw(), &mut actual).unwrap();
        assert_eq!(before, actual);
    }
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
fn cpu_sparse_residual_adapters() {
    let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(1 << 24)).unwrap();
    all_scalars(&pool);
}

#[cfg(feature = "cuda")]
#[test]
fn native_sparse_residual_adapters() {
    use sonium_compute::{CudaBackend, CudaFailureInjectionPoint};
    let backend = match CudaBackend::new(0) {
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
    all_scalars(&pool);
    let backend = pool.backend();
    let matrix = AdjointDeviceCsr::try_from_host_csr_ref(
        backend,
        HostCsrRef::try_new(2, 2, &[0, 1, 2], &[0, 1], &[2.0_f64, 3.0]).unwrap(),
    )
    .unwrap();
    let retained = matrix.try_prebake(backend).unwrap();
    let forward = PrebakedSpmv::try_from_csr(backend, matrix.forward()).unwrap();
    let mut x = pool.try_alloc(2).unwrap();
    let mut rhs = pool.try_alloc(2).unwrap();
    let mut out = pool.try_alloc(2).unwrap();
    backend.try_copy_h2d(&[1.0; 2], x.raw_mut()).unwrap();
    backend.try_copy_h2d(&[4.0; 2], rhs.raw_mut()).unwrap();
    for (op, persistent) in [
        (&matrix as &dyn DeviceOperator<CudaBackend, f64>, false),
        (&retained, true),
        (&forward, true),
    ] {
        op.try_residual(backend, x.raw(), rhs.raw(), out.raw_mut())
            .unwrap();
        let before = backend.begin_runtime_stats();
        op.try_residual(backend, x.raw(), rhs.raw(), out.raw_mut())
            .unwrap();
        let stats = backend.runtime_stats_since(before);
        assert_eq!(stats.kernel_launches, 1);
        assert_eq!(stats.library_operations, 1);
        if persistent {
            assert_eq!(stats.allocation_operations, 0);
            assert_eq!(stats.transfer.h2d_bytes, 0);
            assert_eq!(stats.transfer.d2h_bytes, 0);
        }
        backend.inject_failure(CudaFailureInjectionPoint::Launch);
        assert!(
            op.try_residual(backend, x.raw(), rhs.raw(), out.raw_mut())
                .is_err()
        );
        backend.recover_after_error().unwrap();
        op.try_residual(backend, x.raw(), rhs.raw(), out.raw_mut())
            .unwrap();
        let mut actual = [0.0; 2];
        backend.try_copy_d2h(out.raw(), &mut actual).unwrap();
        assert_eq!(actual, [2.0, 1.0]);
    }
}
