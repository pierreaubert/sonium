//! Native fused Jacobi arithmetic and residency contracts.

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
            .try_jacobi_update_native(b.raw(), b.raw(), b.raw(), x.raw_mut(), 0.5)
            .unwrap()
    );
    let mut actual = [0.; 3];
    pool.backend().try_copy_d2h(x.raw(), &mut actual).unwrap();
    assert_eq!(actual, [1., 2., 3.]);
    assert!(
        pool.backend()
            .try_jacobi_update_native(b.raw(), b.raw(), b.raw(), x.raw_mut(), f64::NAN)
            .is_err()
    );
}

#[cfg(feature = "cuda")]
#[test]
fn native_jacobi_all_scalars_and_tails() {
    use num_complex::{Complex32, Complex64};
    use num_traits::{FromPrimitive, ToPrimitive};
    use sonium_compute::{CudaBackend, DeviceScalar};

    fn check<T: DeviceScalar>(pool: &MemoryPool<CudaBackend>, values: [T; 4], tolerance: f64) {
        let backend = pool.backend();
        for n in [0, 1, 257, 4099] {
            let mut inverse = pool.try_alloc::<T>(n).unwrap();
            let mut b = pool.try_alloc::<T>(n).unwrap();
            let mut ax = pool.try_alloc::<T>(n).unwrap();
            let mut x = pool.try_alloc::<T>(n).unwrap();
            backend
                .try_copy_h2d(&vec![values[0]; n], inverse.raw_mut())
                .unwrap();
            backend
                .try_copy_h2d(&vec![values[1]; n], b.raw_mut())
                .unwrap();
            backend
                .try_copy_h2d(&vec![values[2]; n], ax.raw_mut())
                .unwrap();
            for damping in [0.0, -0.3, 0.75, 1.0] {
                backend
                    .try_copy_h2d(&vec![values[3]; n], x.raw_mut())
                    .unwrap();
                let omega = T::Real::from_f64(damping).unwrap();
                let before = backend.runtime_stats();
                assert!(
                    backend
                        .try_jacobi_update_native(
                            inverse.raw(),
                            b.raw(),
                            ax.raw(),
                            x.raw_mut(),
                            omega
                        )
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
                let expected =
                    values[3] + T::from_real(omega) * (values[0] * (values[1] - values[2]));
                for value in actual {
                    assert!((value - expected).magnitude().to_f64().unwrap() < tolerance);
                }
            }
            let bad = pool.try_alloc::<T>(n + 1).unwrap();
            let before = backend.runtime_stats();
            assert!(
                backend
                    .try_jacobi_update_native(
                        bad.raw(),
                        b.raw(),
                        ax.raw(),
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
fn native_jacobi_rejections_and_launch_failure_preserve_output() {
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
                .try_jacobi_update_native(b.raw(), b.raw(), b.raw(), x.raw_mut(), omega)
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
            .try_jacobi_update_native(other.raw(), b.raw(), b.raw(), x.raw_mut(), 1.)
            .is_err()
    );
    backend.inject_failure(CudaFailureInjectionPoint::Launch);
    assert!(
        backend
            .try_jacobi_update_native(b.raw(), b.raw(), b.raw(), x.raw_mut(), 1.)
            .is_err()
    );
    backend.recover_after_error().unwrap();
    let mut actual = [0.; 3];
    backend.try_copy_d2h(x.raw(), &mut actual).unwrap();
    assert_eq!(actual, [7.; 3]);
    assert!(
        backend
            .try_jacobi_update_native(b.raw(), b.raw(), b.raw(), x.raw_mut(), 1.)
            .unwrap()
    );
}

/// Compares only the post-SpMV update; excludes setup and matrix action.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "manual native CUDA timing diagnostic"]
fn native_jacobi_update_timing() {
    use num_complex::Complex64 as C;
    use sonium_compute::{CudaBackend, PersistentDeviceCsr};
    use std::time::Instant;
    let pool = MemoryPool::new(
        CudaBackend::new(0).unwrap(),
        MemoryPoolConfig::with_cap(1 << 28),
    )
    .unwrap();
    let backend = pool.backend();
    for n in [1024, 65536] {
        let diagonal =
            PersistentDeviceCsr::try_from_host_diagonal(backend, &vec![C::new(0.3, -0.1); n])
                .unwrap();
        let mut rhs = pool.try_alloc::<C>(n).unwrap();
        let mut ax = pool.try_alloc::<C>(n).unwrap();
        let mut residual = pool.try_alloc::<C>(n).unwrap();
        let mut delta = pool.try_alloc::<C>(n).unwrap();
        let mut x = pool.try_alloc::<C>(n).unwrap();
        backend
            .try_copy_h2d(&vec![C::new(1., 0.2); n], rhs.raw_mut())
            .unwrap();
        backend
            .try_copy_h2d(&vec![C::new(0.4, -0.3); n], ax.raw_mut())
            .unwrap();
        // Alternate order to reduce systematic cache/clock bias. Restoring ax
        // is outside each timed interval because ordinary AMG freshly computes it.
        for trial in 0..7 {
            for fused in if trial % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let mut elapsed = 0.;
                for iteration in 0..110 {
                    backend.try_copy_d2d(ax.raw(), residual.raw_mut()).unwrap();
                    backend
                        .try_copy_h2d(&vec![C::new(0.7, -0.2); n], x.raw_mut())
                        .unwrap();
                    backend.try_synchronize().unwrap();
                    let start = Instant::now();
                    if fused {
                        assert!(
                            backend
                                .try_jacobi_update_native(
                                    &diagonal.matrix().values,
                                    rhs.raw(),
                                    residual.raw(),
                                    x.raw_mut(),
                                    0.75,
                                )
                                .unwrap()
                        );
                    } else {
                        backend
                            .try_scale(C::new(-1., 0.), residual.raw_mut())
                            .unwrap();
                        backend
                            .try_axpy(C::new(1., 0.), rhs.raw(), residual.raw_mut())
                            .unwrap();
                        diagonal
                            .try_apply(backend, residual.raw(), delta.raw_mut())
                            .unwrap();
                        backend
                            .try_axpy(C::new(0.75, 0.), delta.raw(), x.raw_mut())
                            .unwrap();
                    }
                    backend.try_synchronize().unwrap();
                    if iteration >= 10 {
                        elapsed += start.elapsed().as_secs_f64();
                    }
                }
                let mut actual = vec![C::default(); n];
                backend.try_copy_d2h(x.raw(), &mut actual).unwrap();
                let expected = C::new(0.7, -0.2)
                    + 0.75 * C::new(0.3, -0.1) * (C::new(1., 0.2) - C::new(0.4, -0.3));
                assert!(
                    actual
                        .iter()
                        .all(|value| (*value - expected).norm() < 1e-14)
                );
                println!(
                    "jacobi_update n={n} trial={trial} fused={fused} mean_us={:.4}",
                    elapsed * 1e4
                );
            }
        }
    }
}
