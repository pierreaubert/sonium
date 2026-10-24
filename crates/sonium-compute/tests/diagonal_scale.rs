//! Native diagonal scaling overwrites scratch without staging or allocation.

// Rust guideline compliant 2026-02-21
use sonium_compute::{Backend, CpuBackend, MemoryPool, MemoryPoolConfig};

#[test]
fn unsupported_native_update_preserves_output() {
    let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(1 << 24)).unwrap();
    let b = pool.try_alloc::<f64>(3).unwrap();
    let mut x = pool.try_alloc::<f64>(3).unwrap();
    pool.backend()
        .try_copy_h2d(&[1., 2., 3.], x.raw_mut())
        .unwrap();
    assert!(
        !pool
            .backend()
            .try_diagonal_scale_native(b.raw(), b.raw(), x.raw_mut(), 0.5)
            .unwrap()
    );
    let mut actual = [0.; 3];
    pool.backend().try_copy_d2h(x.raw(), &mut actual).unwrap();
    assert_eq!(actual, [1., 2., 3.]);
    assert!(
        pool.backend()
            .try_diagonal_scale_native(b.raw(), b.raw(), x.raw_mut(), f64::NAN)
            .is_err()
    );
}

#[cfg(feature = "cuda")]
#[test]
fn native_diagonal_scale_all_scalars_and_tails() {
    use num_complex::{Complex32, Complex64};
    use num_traits::{FromPrimitive, ToPrimitive};
    use sonium_compute::{CudaBackend, DeviceScalar};

    fn check<T: DeviceScalar>(pool: &MemoryPool<CudaBackend>, values: [T; 4], tolerance: f64) {
        let backend = pool.backend();
        for n in [0, 1, 257, 4099] {
            let mut inverse = pool.try_alloc::<T>(n).unwrap();
            let mut b = pool.try_alloc::<T>(n).unwrap();
            let mut x = pool.try_alloc::<T>(n).unwrap();
            backend
                .try_copy_h2d(&vec![values[0]; n], inverse.raw_mut())
                .unwrap();
            backend
                .try_copy_h2d(&vec![values[1]; n], b.raw_mut())
                .unwrap();
            for damping in [0.0, -0.3, 0.75, 1.0] {
                backend
                    .try_copy_h2d(
                        &vec![T::from_real(T::Real::from_f64(f64::NAN).unwrap()); n],
                        x.raw_mut(),
                    )
                    .unwrap();
                let omega = T::Real::from_f64(damping).unwrap();
                let before = backend.runtime_stats();
                assert!(
                    backend
                        .try_diagonal_scale_native(inverse.raw(), b.raw(), x.raw_mut(), omega)
                        .unwrap()
                );
                let after = backend.runtime_stats();
                assert_eq!(after.allocation_operations, before.allocation_operations);
                assert_eq!(after.transfer, before.transfer);
                assert_eq!(
                    after.kernel_launches - before.kernel_launches,
                    usize::from(n > 0)
                );
                let mut actual = vec![T::zero(); n];
                backend.try_copy_d2h(x.raw(), &mut actual).unwrap();
                let expected = T::from_real(omega) * (values[0] * values[1]);
                for value in actual {
                    assert!((value - expected).magnitude().to_f64().unwrap() < tolerance);
                }
            }
            let bad = pool.try_alloc::<T>(n + 1).unwrap();
            let before = backend.runtime_stats();
            assert!(
                backend
                    .try_diagonal_scale_native(
                        bad.raw(),
                        b.raw(),
                        x.raw_mut(),
                        T::Real::from_f64(1.).unwrap()
                    )
                    .is_err()
            );
            assert_eq!(
                backend.runtime_stats().kernel_launches,
                before.kernel_launches
            );
        }
    }

    let backend = match CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            assert!(std::env::var_os("SONIUM_REQUIRE_CUDA").is_none(), "{error}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 24)).unwrap();
    check(&pool, [0.3_f32, 2.1, -0.4, 0.7], 1e-6);
    check(&pool, [0.3_f64, 2.1, -0.4, 0.7], 1e-14);
    check(
        &pool,
        [
            Complex32::new(0.3, -0.7),
            Complex32::new(2.1, 0.4),
            Complex32::new(-0.4, 0.8),
            Complex32::new(0.7, -0.2),
        ],
        1e-6,
    );
    check(
        &pool,
        [
            Complex64::new(0.3, -0.7),
            Complex64::new(2.1, 0.4),
            Complex64::new(-0.4, 0.8),
            Complex64::new(0.7, -0.2),
        ],
        1e-14,
    );
}

#[cfg(feature = "cuda")]
#[test]
fn native_diagonal_scale_rejections_and_launch_failure_preserve_output() {
    use sonium_compute::{CudaBackend, CudaFailureInjectionPoint};
    let backend = match CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            assert!(std::env::var_os("SONIUM_REQUIRE_CUDA").is_none(), "{error}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
    let backend = pool.backend();
    let mut b = pool.try_alloc::<f64>(3).unwrap();
    let mut x = pool.try_alloc::<f64>(3).unwrap();
    backend.try_copy_h2d(&[2.; 3], b.raw_mut()).unwrap();
    backend.try_copy_h2d(&[7.; 3], x.raw_mut()).unwrap();
    for omega in [f64::NAN, f64::INFINITY] {
        assert!(
            backend
                .try_diagonal_scale_native(b.raw(), b.raw(), x.raw_mut(), omega)
                .is_err()
        );
    }
    let foreign = MemoryPool::new(
        CudaBackend::new(0).unwrap(),
        MemoryPoolConfig::with_cap(1 << 20),
    )
    .unwrap();
    let other = foreign.try_alloc::<f64>(3).unwrap();
    assert!(
        backend
            .try_diagonal_scale_native(other.raw(), b.raw(), x.raw_mut(), 1.)
            .is_err()
    );
    backend.inject_failure(CudaFailureInjectionPoint::Launch);
    assert!(
        backend
            .try_diagonal_scale_native(b.raw(), b.raw(), x.raw_mut(), 1.)
            .is_err()
    );
    backend.recover_after_error().unwrap();
    let mut actual = [0.; 3];
    backend.try_copy_d2h(x.raw(), &mut actual).unwrap();
    assert_eq!(actual, [7.; 3]);
    assert!(
        backend
            .try_diagonal_scale_native(b.raw(), b.raw(), x.raw_mut(), 1.)
            .unwrap()
    );
}
