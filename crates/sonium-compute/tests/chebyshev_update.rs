//! Fused recurrence parity, overwrite semantics, residency, and checked failures.

// Rust guideline compliant 2026-02-21
use sonium_compute::{Backend, CpuBackend, MemoryPool, MemoryPoolConfig};

#[test]
fn unavailable_recurrence_preserves_both_outputs() {
    let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(1 << 20)).unwrap();
    let backend = pool.backend();
    let b = pool.try_alloc::<f64>(3).unwrap();
    let mut d = pool.try_alloc::<f64>(3).unwrap();
    let mut x = pool.try_alloc::<f64>(3).unwrap();
    backend.try_copy_h2d(&[4.; 3], d.raw_mut()).unwrap();
    backend.try_copy_h2d(&[7.; 3], x.raw_mut()).unwrap();
    assert!(
        !backend
            .try_chebyshev_update_native(
                b.raw(),
                b.raw(),
                b.raw(),
                d.raw_mut(),
                x.raw_mut(),
                0.5,
                0.0,
            )
            .unwrap()
    );
    for (alpha, beta) in [(f64::NAN, 0.), (1., f64::INFINITY)] {
        assert!(
            backend
                .try_chebyshev_update_native(
                    b.raw(),
                    b.raw(),
                    b.raw(),
                    d.raw_mut(),
                    x.raw_mut(),
                    alpha,
                    beta,
                )
                .is_err()
        );
    }
    let bad = pool.try_alloc::<f64>(4).unwrap();
    assert!(
        backend
            .try_chebyshev_update_native(
                bad.raw(),
                b.raw(),
                b.raw(),
                d.raw_mut(),
                x.raw_mut(),
                1.,
                0.,
            )
            .is_err()
    );
    let mut values = [0.; 3];
    backend.try_copy_d2h(d.raw(), &mut values).unwrap();
    assert_eq!(values, [4.; 3]);
    backend.try_copy_d2h(x.raw(), &mut values).unwrap();
    assert_eq!(values, [7.; 3]);
}

#[cfg(feature = "cuda")]
fn cuda_pool() -> Option<MemoryPool<sonium_compute::CudaBackend>> {
    let backend = match sonium_compute::CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            assert!(std::env::var_os("SONIUM_REQUIRE_CUDA").is_none(), "{error}");
            eprintln!("SKIP native Chebyshev update: {error}");
            return None;
        }
    };
    Some(MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 24)).unwrap())
}

#[cfg(feature = "cuda")]
#[test]
fn native_recurrence_all_scalars_tails_and_overwrite() {
    use num_complex::{Complex32, Complex64};
    use num_traits::{FromPrimitive, ToPrimitive};
    use sonium_compute::{CudaBackend, DeviceScalar};

    fn check<T: DeviceScalar>(pool: &MemoryPool<CudaBackend>, seeds: [T; 4], tolerance: f64) {
        let backend = pool.backend();
        for n in [0, 1, 257, 4099] {
            let host: Vec<Vec<T>> = seeds
                .iter()
                .map(|&seed| {
                    (0..n)
                        .map(|i| {
                            seed + T::from_real(T::Real::from_f64((i % 7) as f64 * 0.01).unwrap())
                        })
                        .collect()
                })
                .collect();
            let mut inverse = pool.try_alloc::<T>(n).unwrap();
            let mut b = pool.try_alloc::<T>(n).unwrap();
            let mut ax = pool.try_alloc::<T>(n).unwrap();
            let mut d = pool.try_alloc::<T>(n).unwrap();
            let mut x = pool.try_alloc::<T>(n).unwrap();
            backend.try_copy_h2d(&host[0], inverse.raw_mut()).unwrap();
            backend.try_copy_h2d(&host[1], b.raw_mut()).unwrap();
            backend.try_copy_h2d(&host[2], ax.raw_mut()).unwrap();
            backend.try_copy_h2d(&host[3], x.raw_mut()).unwrap();
            backend
                .try_copy_h2d(
                    &vec![T::from_real(T::Real::from_f64(f64::NAN).unwrap()); n],
                    d.raw_mut(),
                )
                .unwrap();
            let mut expected_d = vec![T::zero(); n];
            let mut expected_x = host[3].clone();
            for (alpha, beta) in [(0.8, 0.0), (0.2, 0.6), (-0.3, -0.2), (0., 1.)] {
                let alpha = T::Real::from_f64(alpha).unwrap();
                let beta = T::Real::from_f64(beta).unwrap();
                let before = backend.runtime_stats();
                assert!(
                    backend
                        .try_chebyshev_update_native(
                            inverse.raw(),
                            b.raw(),
                            ax.raw(),
                            d.raw_mut(),
                            x.raw_mut(),
                            alpha,
                            beta,
                        )
                        .unwrap()
                );
                let after = backend.runtime_stats();
                assert_eq!(before.transfer, after.transfer);
                assert_eq!(before.allocation_operations, after.allocation_operations);
                assert_eq!(
                    after.kernel_launches - before.kernel_launches,
                    usize::from(n > 0)
                );
                for i in 0..n {
                    expected_d[i] = T::from_real(beta) * expected_d[i]
                        + T::from_real(alpha) * (host[0][i] * (host[1][i] - host[2][i]));
                    expected_x[i] = expected_x[i] + expected_d[i];
                }
                let mut actual = vec![T::zero(); n];
                for (device, expected) in [(d.raw(), &expected_d), (x.raw(), &expected_x)] {
                    backend.try_copy_d2h(device, &mut actual).unwrap();
                    for (a, e) in actual.iter().zip(expected) {
                        assert!(
                            (*a - *e).magnitude().to_f64().unwrap()
                                < tolerance * (1. + e.magnitude().to_f64().unwrap())
                        );
                    }
                }
            }
        }
    }
    let Some(pool) = cuda_pool() else { return };
    check(&pool, [0.3_f32, 2.1, -0.4, 0.7], 2e-6);
    check(&pool, [0.3_f64, 2.1, -0.4, 0.7], 2e-14);
    check(
        &pool,
        [
            Complex32::new(0.3, -0.7),
            Complex32::new(2.1, 0.4),
            Complex32::new(-0.4, 0.8),
            Complex32::new(0.7, -0.2),
        ],
        2e-6,
    );
    check(
        &pool,
        [
            Complex64::new(0.3, -0.7),
            Complex64::new(2.1, 0.4),
            Complex64::new(-0.4, 0.8),
            Complex64::new(0.7, -0.2),
        ],
        2e-14,
    );
}

#[cfg(feature = "cuda")]
#[test]
fn native_recurrence_rejections_and_failure_preserve_outputs() {
    use sonium_compute::CudaFailureInjectionPoint;
    let Some(pool) = cuda_pool() else { return };
    let backend = pool.backend();
    let mut b = pool.try_alloc::<f64>(3).unwrap();
    let mut d = pool.try_alloc::<f64>(3).unwrap();
    let mut x = pool.try_alloc::<f64>(3).unwrap();
    backend.try_copy_h2d(&[2.; 3], b.raw_mut()).unwrap();
    backend.try_copy_h2d(&[4.; 3], d.raw_mut()).unwrap();
    backend.try_copy_h2d(&[7.; 3], x.raw_mut()).unwrap();
    let bad = pool.try_alloc::<f64>(4).unwrap();
    let foreign = cuda_pool().unwrap();
    let other = foreign.try_alloc::<f64>(3).unwrap();
    for invalid in [bad.raw(), other.raw()] {
        assert!(
            backend
                .try_chebyshev_update_native(
                    invalid,
                    b.raw(),
                    b.raw(),
                    d.raw_mut(),
                    x.raw_mut(),
                    1.,
                    0.
                )
                .is_err()
        );
    }
    for (alpha, beta) in [(f64::NAN, 0.), (1., f64::INFINITY)] {
        assert!(
            backend
                .try_chebyshev_update_native(
                    b.raw(),
                    b.raw(),
                    b.raw(),
                    d.raw_mut(),
                    x.raw_mut(),
                    alpha,
                    beta
                )
                .is_err()
        );
    }
    backend.inject_failure(CudaFailureInjectionPoint::Launch);
    assert!(
        backend
            .try_chebyshev_update_native(
                b.raw(),
                b.raw(),
                b.raw(),
                d.raw_mut(),
                x.raw_mut(),
                1.,
                0.
            )
            .is_err()
    );
    backend.recover_after_error().unwrap();
    let mut actual = [0.; 3];
    backend.try_copy_d2h(d.raw(), &mut actual).unwrap();
    assert_eq!(actual, [4.; 3]);
    backend.try_copy_d2h(x.raw(), &mut actual).unwrap();
    assert_eq!(actual, [7.; 3]);
    assert!(
        backend
            .try_chebyshev_update_native(
                b.raw(),
                b.raw(),
                b.raw(),
                d.raw_mut(),
                x.raw_mut(),
                1.,
                0.
            )
            .unwrap()
    );
    backend.try_copy_d2h(d.raw(), &mut actual).unwrap();
    assert_eq!(actual, [0.; 3]);
    backend.try_copy_d2h(x.raw(), &mut actual).unwrap();
    assert_eq!(actual, [7.; 3]);
}
