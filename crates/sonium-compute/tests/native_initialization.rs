//! Native finite validation and zeroing avoid bulk initialization transfers.

// Rust guideline compliant 2026-02-21
use sonium_compute::{Backend, CpuBackend};

#[test]
fn unsupported_hooks_decline_without_mutation() {
    let backend = CpuBackend::new();
    let mut values = backend.raw_alloc::<f64>(2).unwrap();
    backend
        .try_copy_h2d(&[3.0, f64::INFINITY], &mut values)
        .unwrap();
    assert_eq!(backend.try_all_finite_native(&values).unwrap(), None);
    assert!(!backend.try_zero_native(&mut values).unwrap());
    let mut host = [0.0; 2];
    backend.try_copy_d2h(&values, &mut host).unwrap();
    assert_eq!(host, [3.0, f64::INFINITY]);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_validation_and_zero_cover_all_scalars_and_tails() {
    use num_complex::{Complex32, Complex64};
    use sonium_compute::{CudaBackend, DeviceScalar};
    fn check<T: DeviceScalar + PartialEq>(backend: &CudaBackend, finite: T, bad: T) {
        for n in [0, 1, 257] {
            let mut host = vec![finite; n];
            let mut values = backend.raw_alloc::<T>(n).unwrap();
            backend.try_copy_h2d(&host, &mut values).unwrap();
            assert_eq!(backend.try_all_finite_native(&values).unwrap(), Some(true));
            let before = backend.runtime_stats();
            assert_eq!(backend.try_all_finite_native(&values).unwrap(), Some(true));
            let after = backend.runtime_stats();
            assert_eq!(after.allocation_operations, before.allocation_operations);
            assert_eq!(after.transfer.h2d_bytes, before.transfer.h2d_bytes);
            assert!(after.transfer.d2h_bytes - before.transfer.d2h_bytes <= 4);
            if n > 0 {
                host[n - 1] = bad;
                backend.try_copy_h2d(&host, &mut values).unwrap();
                assert_eq!(backend.try_all_finite_native(&values).unwrap(), Some(false));
            }
            let before = backend.runtime_stats();
            assert!(backend.try_zero_native(&mut values).unwrap());
            let after = backend.runtime_stats();
            assert_eq!(after.allocation_operations, before.allocation_operations);
            assert_eq!(after.transfer.h2d_bytes, before.transfer.h2d_bytes);
            assert_eq!(after.transfer.d2h_bytes, before.transfer.d2h_bytes);
            backend.try_copy_d2h(&values, &mut host).unwrap();
            assert!(host.iter().all(|v| *v == T::zero()));
        }
    }
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
    check(&backend, f32::MAX, f32::NAN);
    check(&backend, f64::MAX, f64::INFINITY);
    check(
        &backend,
        Complex32::new(1.0, 2.0),
        Complex32::new(1.0, f32::NAN),
    );
    check(
        &backend,
        Complex64::new(1.0, 2.0),
        Complex64::new(f64::NEG_INFINITY, 1.0),
    );

    let foreign = CudaBackend::new(0).unwrap();
    let mut values = foreign.raw_alloc::<f64>(1).unwrap();
    foreign.try_copy_h2d(&[7.0], &mut values).unwrap();
    assert!(backend.try_all_finite_native(&values).is_err());
    assert!(backend.try_zero_native(&mut values).is_err());
    let mut host = [0.0];
    foreign.try_copy_d2h(&values, &mut host).unwrap();
    assert_eq!(host, [7.0]);
}
