//! Native factor/permutation parity, storage reuse, refresh and recovery.

// Rust guideline compliant 2026-02-21
use super::*;
use crate::ilu::permute_symmetric;
use num_traits::{FromPrimitive, ToPrimitive};
use sonium_solvers_core::traits::Preconditioner;

fn backend() -> Option<CudaBackend> {
    match CudaBackend::new(0) {
        Ok(backend) => Some(backend),
        Err(error) => {
            assert!(
                std::env::var_os("SONIUM_REQUIRE_CUDA").is_none(),
                "CUDA required: {error}"
            );
            None
        }
    }
}

fn scalar<T: ComplexField>(re: f64, im: f64) -> T {
    T::from_re_im(
        T::Real::from_f64(re).unwrap(),
        T::Real::from_f64(im).unwrap(),
    )
}

fn check<T>(backend: &CudaBackend, tolerance: f64)
where
    T: ComplexField + DeviceScalar<Real = <T as ComplexField>::Real>,
{
    let n = 13;
    for pivoting in [false, true] {
        for reorder in [false, true] {
            let mut entries = Vec::new();
            for row in 0..n {
                for (offset, re, im) in [(0, 5.0, 0.1), (1, -0.3, 0.02), (4, -0.15, -0.01)] {
                    let column = (row + offset + usize::from(pivoting)) % n;
                    entries.push((row, column, scalar::<T>(re, im)));
                }
            }
            let matrix = CsrMatrix::from_triplets(n, n, entries);
            let config = IlutConfig {
                pivoting,
                drop_tolerance: 1e-3,
                fill_factor: 3,
                ..Default::default()
            };
            let make = |matrix: &CsrMatrix<T>| {
                if reorder {
                    // Non-involutive ordering makes reversed permutation conventions detectable.
                    let perm: Vec<_> = (0..n).map(|i| (i * 5 + 2) % n).collect();
                    let reordered = permute_symmetric(matrix, &perm);
                    IlutPreconditioner::checked_factorize(&reordered, &config, Some(perm)).unwrap()
                } else {
                    IlutPreconditioner::try_from_csr_with_config(matrix, &config).unwrap()
                }
            };
            let host = make(&matrix);
            assert_eq!(host.perm.is_some(), reorder);
            if pivoting {
                assert!(host.col_perm.is_some(), "fixture must pivot");
            }
            let mut device = CudaIlutPreconditioner::try_from_host(backend, &host).unwrap();
            let mut rhs = backend.raw_alloc(n).unwrap();
            let mut output = backend.raw_alloc(n).unwrap();
            let apply = |device: &CudaIlutPreconditioner<T>,
                         rhs: &CudaBuffer<T>,
                         output: &mut CudaBuffer<T>,
                         host: &IlutPreconditioner<T>,
                         input: &[T]| {
                let expected = host.apply(input);
                let before = backend.begin_runtime_stats();
                device.try_apply(backend, rhs, output).unwrap();
                backend.try_synchronize().unwrap();
                let stats = backend.runtime_stats_since(before);
                assert_eq!(stats.allocation_operations, 0);
                assert_eq!(stats.transfer.h2d_bytes, 0);
                assert_eq!(stats.transfer.d2h_bytes, 0);
                let mut actual = vec![<T as DeviceScalar>::zero(); n];
                backend.try_copy_d2h(output, &mut actual).unwrap();
                for (a, e) in actual.iter().zip(expected) {
                    assert!(
                        (*a - e).norm().to_f64().unwrap()
                            < tolerance * (1.0 + e.norm().to_f64().unwrap())
                    );
                }
            };
            let mut input = Vec::new();
            for pass in 0..3 {
                input = (0..n)
                    .map(|i| scalar::<T>((i + pass + 1) as f64 * 0.2, 0.1))
                    .collect();
                backend.try_copy_h2d(&input, &mut rhs).unwrap();
                apply(&device, &rhs, &mut output, &host, &input);
            }
            let mut scaled = matrix.clone();
            for value in &mut scaled.values {
                *value *= scalar::<T>(2.0, 0.0);
            }
            let refreshed = make(&scaled);
            let before = backend.begin_runtime_stats();
            assert!(device.try_refresh_from_host(backend, &refreshed).unwrap());
            assert_eq!(backend.runtime_stats_since(before).allocation_operations, 0);
            apply(&device, &rhs, &mut output, &refreshed, &input);

            let mut incompatible = refreshed.clone();
            incompatible.perm = Some((0..n).rev().collect());
            let before = backend.begin_runtime_stats();
            assert!(
                !device
                    .try_refresh_from_host(backend, &incompatible)
                    .unwrap()
            );
            assert_eq!(backend.runtime_stats_since(before).transfer.h2d_bytes, 0);
            apply(&device, &rhs, &mut output, &refreshed, &input);

            let mut different_pattern = refreshed.clone();
            assert!(!different_pattern.l_values.is_empty());
            different_pattern.l_values.clear();
            different_pattern.l_col_indices.clear();
            different_pattern.l_row_ptrs.fill(0);
            let before = backend.begin_runtime_stats();
            assert!(
                !device
                    .try_refresh_from_host(backend, &different_pattern)
                    .unwrap()
            );
            assert_eq!(backend.runtime_stats_since(before).transfer.h2d_bytes, 0);
            apply(&device, &rhs, &mut output, &refreshed, &input);

            backend.inject_preconditioner_failure(
                sonium_compute::PreconditionerFailureInjectionPoint::DiagonalUpload,
            );
            assert!(device.try_refresh_from_host(backend, &refreshed).is_err());
            backend.recover_after_error().unwrap();
            assert!(device.try_apply(backend, &rhs, &mut output).is_err());
            assert!(device.try_refresh_from_host(backend, &refreshed).unwrap());
            apply(&device, &rhs, &mut output, &refreshed, &input);

            let mut before_output = vec![<T as DeviceScalar>::zero(); n];
            backend.try_copy_d2h(&output, &mut before_output).unwrap();
            backend.inject_failure(sonium_compute::CudaFailureInjectionPoint::Launch);
            assert!(device.try_apply(backend, &rhs, &mut output).is_err());
            backend.recover_after_error().unwrap();
            let mut after_output = before_output.clone();
            backend.try_copy_d2h(&output, &mut after_output).unwrap();
            assert_eq!(before_output, after_output);
            apply(&device, &rhs, &mut output, &refreshed, &input);
        }
    }
}

#[test]
fn native_ilut_preserves_pivots_ordering_and_refresh_all_scalars() {
    let Some(backend) = backend() else { return };
    check::<f32>(&backend, 3e-5);
    check::<f64>(&backend, 1e-11);
    check::<num_complex::Complex32>(&backend, 3e-5);
    check::<num_complex::Complex64>(&backend, 1e-11);
}

#[test]
fn native_ilut_rejects_invalid_factors_before_upload() {
    let Some(backend) = backend() else { return };
    let matrix = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 2.0), (1, 1, 3.0)]);
    let host = IlutPreconditioner::try_from_csr(&matrix).unwrap();
    for case in 0..5 {
        let mut invalid = host.clone();
        match case {
            0 => invalid.regularized_pivots.push(0),
            1 => invalid.u_diag_inv[0] = f64::NAN,
            2 => invalid.u_diag_inv[0] = 0.0,
            3 => invalid.perm = Some(vec![0, 0]),
            _ => invalid.col_perm = Some(vec![2, 0]),
        }
        let before = backend.begin_runtime_stats();
        assert!(CudaIlutPreconditioner::try_from_host(&backend, &invalid).is_err());
        let stats = backend.runtime_stats_since(before);
        assert_eq!(stats.allocation_operations, 0);
        assert_eq!(stats.transfer.h2d_bytes, 0);
    }
    let config = IlutConfig {
        drop_tolerance: f64::NAN,
        ..Default::default()
    };
    assert!(CudaIlutPreconditioner::try_from_csr_with_config(&backend, &matrix, &config).is_err());
}
