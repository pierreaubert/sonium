//! Direct reconstruction and orthogonality checks for device initial QR.

// Rust guideline compliant 2026-02-21
use super::*;
use num_complex::{Complex32, Complex64};

fn check<B: Backend, T: DeviceScalar>(backend: &B, phase: T, tolerance: f64) {
    let real = |x| T::Real::from_f64(x).unwrap();
    let n = 17;
    let count = 3;
    let values: Vec<_> = (0..n * count)
        .map(|i| {
            let row = i % n;
            let col = i / n;
            phase
                * T::from_real(real(if row == col {
                    1.0
                } else {
                    ((row + col * 3) % 7) as f64 * 0.01
                }))
        })
        .collect();
    let mut input = backend.raw_alloc::<T>(values.len()).unwrap();
    let mut q = backend.raw_alloc::<T>(values.len()).unwrap();
    let mut column = backend.raw_alloc::<T>(n).unwrap();
    let mut r = Vec::new();
    let mut coefficients = Vec::new();
    backend.try_copy_h2d(&values, &mut input).unwrap();
    for _ in 0..2 {
        let rank = qr_device_block_into(
            backend,
            &input,
            &mut q,
            &mut column,
            n,
            count,
            real(tolerance),
            &mut r,
            &mut coefficients,
        )
        .unwrap();
        assert_eq!(rank, count);
        let mut basis = vec![T::zero(); values.len()];
        backend.try_copy_d2h(&q, &mut basis).unwrap();
        for col in 0..count {
            for row in 0..n {
                let reconstructed = (0..count).fold(T::zero(), |sum, j| {
                    sum + basis[j * n + row] * r[j * count + col]
                });
                assert!(
                    (reconstructed - values[col * n + row]).norm_sqr().sqrt() < real(tolerance)
                );
            }
            for other in 0..count {
                let dot = (0..n).fold(T::zero(), |sum, row| {
                    sum + basis[col * n + row].conj_mul(basis[other * n + row])
                });
                let expected = if col == other { T::one() } else { T::zero() };
                assert!((dot - expected).norm_sqr().sqrt() < real(tolerance));
            }
        }
    }
    // Reusing nonzero Q storage must not preserve stale rejected columns.
    backend
        .try_copy_h2d(&vec![T::zero(); values.len()], &mut input)
        .unwrap();
    assert_eq!(
        qr_device_block_into(
            backend,
            &input,
            &mut q,
            &mut column,
            n,
            count,
            real(tolerance),
            &mut r,
            &mut coefficients
        )
        .unwrap(),
        0
    );
    let mut basis = vec![T::one(); values.len()];
    backend.try_copy_d2h(&q, &mut basis).unwrap();
    assert!(
        basis
            .iter()
            .all(|value| value.norm_sqr() == T::Real::zero())
    );
    assert!(r.iter().all(|value| value.norm_sqr() == T::Real::zero()));
}

fn all_scalars<B: Backend>(backend: &B) {
    check(backend, 1.0_f32, 1e-5);
    check(backend, 1.0_f64, 1e-11);
    check(backend, Complex32::new(1.0, 0.3), 1e-5);
    check(backend, Complex64::new(1.0, 0.3), 1e-11);
}

#[test]
fn cpu_initial_qr_reconstructs_and_clears_all_scalars() {
    all_scalars(&crate::CpuBackend::new());
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_initial_qr_reconstructs_and_clears_all_scalars() {
    let backend = match crate::CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            assert!(
                std::env::var_os("SONIUM_REQUIRE_CUDA").is_none(),
                "CUDA required: {error}"
            );
            return;
        }
    };
    all_scalars(&backend);
}
