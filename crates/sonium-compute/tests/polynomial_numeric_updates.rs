//! Owned polynomial preconditioners retain storage across numeric refreshes.

// Rust guideline compliant 2026-02-21
use num_complex::{Complex32, Complex64};
use num_traits::{Float, FromPrimitive};
use sonium_compute::{
    Backend, ChebyshevBackend, CpuBackend, DeviceCsr, DevicePreconditioner, DeviceScalar,
    PersistentDeviceCsr, PolynomialJacobiBackend, PreconditionerFailureInjectionPoint,
};

fn check<B: Backend, T: DeviceScalar>(
    backend: &B,
    counters: impl Fn() -> Option<[usize; 3]>,
    inject: impl Fn(),
    recover: impl Fn(),
    degree: usize,
) {
    let scalar = |v| T::from_real(T::Real::from_f64(v).unwrap());
    let n = 17;
    let rows: Vec<_> = (0..=n as i32).collect();
    let columns: Vec<_> = (0..n as i32).collect();
    let initial = vec![scalar(2.0); n];
    let source = DeviceCsr::try_from_host(backend, n, n, &rows, &columns, &initial).unwrap();
    let mut polynomial = PolynomialJacobiBackend::try_from_owned_csr(
        backend,
        PersistentDeviceCsr::try_clone_from_device_csr(backend, &source).unwrap(),
        &initial,
        degree,
        scalar(0.7),
    )
    .unwrap();
    let mut chebyshev = ChebyshevBackend::try_from_owned_csr_host(
        backend,
        PersistentDeviceCsr::try_clone_from_device_csr(backend, &source).unwrap(),
        &rows,
        &columns,
        &initial,
        degree,
        T::Real::from_f64(3.0).unwrap(),
    )
    .unwrap();
    let mut rhs = backend.raw_alloc::<T>(2 * n).unwrap();
    let mut out = backend.raw_alloc::<T>(2 * n).unwrap();
    let mut expected = backend.raw_alloc::<T>(2 * n).unwrap();
    backend
        .try_copy_h2d(&vec![scalar(6.0); 2 * n], &mut rhs)
        .unwrap();
    for trial in 0..3 {
        let diagonal: Vec<_> = (0..n)
            .map(|i| scalar(2.0 + trial as f64 + (i % 3) as f64))
            .collect();
        let matrix = DeviceCsr::try_from_host(backend, n, n, &rows, &columns, &diagonal).unwrap();
        let fresh_poly =
            PolynomialJacobiBackend::try_new(backend, &matrix, &diagonal, degree, scalar(0.7))
                .unwrap();
        let fresh_cheb = ChebyshevBackend::try_from_csr_host(
            backend,
            &matrix,
            &rows,
            &columns,
            &diagonal,
            degree,
            T::Real::from_f64(3.0).unwrap(),
        )
        .unwrap();
        let before = counters();
        polynomial
            .try_update_numeric(backend, &matrix.values, &diagonal)
            .unwrap();
        chebyshev
            .try_update_from_csr_host(backend, &matrix.values, &rows, &columns, &diagonal)
            .unwrap();
        polynomial
            .try_apply_colmajor(backend, 2, &rhs, &mut out)
            .unwrap();
        chebyshev
            .try_apply_colmajor(backend, 2, &rhs, &mut out)
            .unwrap();
        if let (Some(before), Some(after)) = (before, counters()) {
            if trial > 0 {
                assert_eq!(
                    after[0], before[0],
                    "numeric update replaced device storage"
                );
            }
            assert_eq!(after[1] - before[1], 2 * n * std::mem::size_of::<T>());
            // First sparse analysis may validate the matrix once on device.
            if trial > 0 {
                assert_eq!(after[2], before[2]);
            }
        }
        for (retained, fresh) in [
            (
                &polynomial as &dyn DevicePreconditioner<B, T>,
                &fresh_poly as &dyn DevicePreconditioner<B, T>,
            ),
            (
                &chebyshev as &dyn DevicePreconditioner<B, T>,
                &fresh_cheb as &dyn DevicePreconditioner<B, T>,
            ),
        ] {
            retained
                .try_apply_colmajor(backend, 2, &rhs, &mut out)
                .unwrap();
            fresh
                .try_apply_colmajor(backend, 2, &rhs, &mut expected)
                .unwrap();
            let mut actual = vec![T::zero(); 2 * n];
            let mut reference = actual.clone();
            backend.try_copy_d2h(&out, &mut actual).unwrap();
            backend.try_copy_d2h(&expected, &mut reference).unwrap();
            for (a, b) in actual.iter().zip(reference) {
                assert!(
                    (*a - b).norm_sqr().sqrt()
                        < T::Real::epsilon() * T::Real::from_f64(100.0).unwrap()
                );
            }
        }
    }
    // Validation failure preserves the previously usable operator.
    assert!(
        polynomial
            .try_update_numeric(backend, &source.values, &vec![T::zero(); n])
            .is_err()
    );
    polynomial
        .try_apply_colmajor(backend, 2, &rhs, &mut out)
        .unwrap();
    // Inject after matrix replacement: neither polynomial may consume a mixed state.
    inject();
    assert!(
        polynomial
            .try_update_numeric(backend, &source.values, &initial)
            .is_err()
    );
    recover();
    backend
        .try_copy_h2d(&vec![scalar(91.0); 2 * n], &mut out)
        .unwrap();
    assert!(
        polynomial
            .try_apply_colmajor(backend, 2, &rhs, &mut out)
            .is_err()
    );
    let mut unchanged = vec![T::zero(); 2 * n];
    backend.try_copy_d2h(&out, &mut unchanged).unwrap();
    assert!(
        unchanged
            .iter()
            .all(|&v| (v - scalar(91.0)).norm_sqr() == T::Real::from_f64(0.0).unwrap())
    );
    polynomial
        .try_update_numeric(backend, &source.values, &initial)
        .unwrap();
    polynomial
        .try_apply_colmajor(backend, 2, &rhs, &mut out)
        .unwrap();
    inject();
    assert!(
        chebyshev
            .try_update_from_csr_host(backend, &source.values, &rows, &columns, &initial)
            .is_err()
    );
    recover();
    assert!(
        chebyshev
            .try_apply_colmajor(backend, 2, &rhs, &mut out)
            .is_err()
    );
    chebyshev
        .try_update_from_csr_host(backend, &source.values, &rows, &columns, &initial)
        .unwrap();
    chebyshev
        .try_apply_colmajor(backend, 2, &rhs, &mut out)
        .unwrap();
}

#[test]
fn cpu_owned_polynomials_refresh_and_recover() {
    let backend = CpuBackend::new();
    let inject = || {
        CpuBackend::inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::DiagonalUpload,
        )
    };
    check::<_, f32>(&backend, || None, inject, || {}, 3);
    check::<_, f64>(&backend, || None, inject, || {}, 3);
    check::<_, Complex32>(&backend, || None, inject, || {}, 3);
    check::<_, Complex64>(&backend, || None, inject, || {}, 3);
    check::<_, Complex64>(&backend, || None, inject, || {}, 1);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_owned_polynomials_refresh_without_allocations() {
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
    let counters = || {
        let s = backend.runtime_stats();
        Some([
            s.allocation_operations,
            s.transfer.h2d_bytes,
            s.transfer.d2h_bytes,
        ])
    };
    let inject = || {
        backend.inject_preconditioner_failure(PreconditionerFailureInjectionPoint::DiagonalUpload)
    };
    let recover = || backend.recover_after_error().unwrap();
    check::<_, f32>(&backend, counters, inject, recover, 3);
    check::<_, f64>(&backend, counters, inject, recover, 3);
    check::<_, Complex32>(&backend, counters, inject, recover, 3);
    check::<_, Complex64>(&backend, counters, inject, recover, 3);
    check::<_, Complex64>(&backend, counters, inject, recover, 1);
}
