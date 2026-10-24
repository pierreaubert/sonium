//! Broadcast-diagonal block smoother arithmetic and checked admission.

// Rust guideline compliant 2026-02-21
use sonium_compute::{Backend, CpuBackend, MemoryPool, MemoryPoolConfig};

#[test]
fn unavailable_block_smoothers_validate_without_mutation() {
    let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(1 << 20)).unwrap();
    let b = pool.try_alloc::<f64>(9).unwrap();
    let diagonal = pool.try_alloc::<f64>(3).unwrap();
    let mut x = pool.try_alloc::<f64>(9).unwrap();
    let mut d = pool.try_alloc::<f64>(9).unwrap();
    let backend = pool.backend();
    backend.try_copy_h2d(&[7.; 9], x.raw_mut()).unwrap();
    backend.try_copy_h2d(&[4.; 9], d.raw_mut()).unwrap();
    assert!(
        !backend
            .try_diagonal_scale_colmajor_native(3, diagonal.raw(), b.raw(), x.raw_mut(), 1.)
            .unwrap()
    );
    assert!(
        !backend
            .try_chebyshev_update_colmajor_native(
                3,
                diagonal.raw(),
                b.raw(),
                b.raw(),
                d.raw_mut(),
                x.raw_mut(),
                1.,
                0.
            )
            .unwrap()
    );
    for width in [2, usize::MAX] {
        assert!(
            backend
                .try_diagonal_scale_colmajor_native(width, diagonal.raw(), b.raw(), x.raw_mut(), 1.)
                .is_err()
        );
        assert!(
            backend
                .try_chebyshev_update_colmajor_native(
                    width,
                    diagonal.raw(),
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
    let mut actual = [0.; 9];
    backend.try_copy_d2h(x.raw(), &mut actual).unwrap();
    assert_eq!(actual, [7.; 9]);
    backend.try_copy_d2h(d.raw(), &mut actual).unwrap();
    assert_eq!(actual, [4.; 9]);
}

#[cfg(feature = "cuda")]
#[test]
fn native_block_smoothers_all_scalars_shapes_and_residency() {
    use num_complex::{Complex32, Complex64};
    use num_traits::{FromPrimitive, ToPrimitive};
    use sonium_compute::{CudaBackend, DeviceScalar};
    fn check<T: DeviceScalar>(pool: &MemoryPool<CudaBackend>, seed: T, tolerance: f64) {
        let backend = pool.backend();
        let real = |v| T::Real::from_f64(v).unwrap();
        let scalar = |v| T::from_real(real(v));
        for rows in [0, 1, 257] {
            for width in [0, 1, 3] {
                let len = rows * width;
                let inverse: Vec<_> = (0..rows)
                    .map(|i| seed + scalar(0.01 * (i % 7) as f64))
                    .collect();
                let rhs: Vec<_> = (0..len).map(|i| seed + scalar((i as f64).sin())).collect();
                let action: Vec<_> = (0..len).map(|i| scalar((i as f64).cos())).collect();
                let mut diagonal = pool.try_alloc::<T>(rows).unwrap();
                let mut b = pool.try_alloc::<T>(len).unwrap();
                let mut ax = pool.try_alloc::<T>(len).unwrap();
                let mut x = pool.try_alloc::<T>(len).unwrap();
                let mut d = pool.try_alloc::<T>(len).unwrap();
                backend.try_copy_h2d(&inverse, diagonal.raw_mut()).unwrap();
                backend.try_copy_h2d(&rhs, b.raw_mut()).unwrap();
                backend.try_copy_h2d(&action, ax.raw_mut()).unwrap();
                backend
                    .try_copy_h2d(&vec![scalar(f64::NAN); len], x.raw_mut())
                    .unwrap();
                backend
                    .try_copy_h2d(&vec![scalar(f64::NAN); len], d.raw_mut())
                    .unwrap();
                let before = backend.runtime_stats();
                assert!(
                    backend
                        .try_diagonal_scale_colmajor_native(
                            width,
                            diagonal.raw(),
                            b.raw(),
                            x.raw_mut(),
                            real(0.7)
                        )
                        .unwrap()
                );
                let mut expected_x: Vec<_> = (0..len)
                    .map(|i| scalar(0.7) * inverse[i % rows] * rhs[i])
                    .collect();
                let mut expected_d = vec![T::zero(); len];
                for beta in [0., 0.3] {
                    assert!(
                        backend
                            .try_chebyshev_update_colmajor_native(
                                width,
                                diagonal.raw(),
                                b.raw(),
                                ax.raw(),
                                d.raw_mut(),
                                x.raw_mut(),
                                real(0.4),
                                real(beta)
                            )
                            .unwrap()
                    );
                    for i in 0..len {
                        expected_d[i] = scalar(beta) * expected_d[i]
                            + scalar(0.4) * inverse[i % rows] * (rhs[i] - action[i]);
                        expected_x[i] = expected_x[i] + expected_d[i];
                    }
                }
                let after = backend.runtime_stats();
                assert_eq!(after.transfer, before.transfer);
                assert_eq!(after.allocation_operations, before.allocation_operations);
                assert_eq!(
                    after.kernel_launches - before.kernel_launches,
                    3 * usize::from(len > 0)
                );
                let mut actual = vec![T::zero(); len];
                for (device, expected) in [(x.raw(), expected_x), (d.raw(), expected_d)] {
                    backend.try_copy_d2h(device, &mut actual).unwrap();
                    for (a, e) in actual.iter().zip(expected) {
                        assert!(
                            (*a - e).magnitude().to_f64().unwrap()
                                < tolerance * (1. + e.magnitude().to_f64().unwrap())
                        );
                    }
                }
            }
        }
    }
    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            assert!(std::env::var_os("SONIUM_REQUIRE_CUDA").is_none(), "{e}");
            eprintln!("SKIP CUDA block smoother: {e}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 24)).unwrap();
    check(&pool, 0.3_f32, 2e-6);
    check(&pool, 0.3_f64, 2e-14);
    check(&pool, Complex32::new(0.3, -0.2), 2e-6);
    check(&pool, Complex64::new(0.3, -0.2), 2e-14);
}
