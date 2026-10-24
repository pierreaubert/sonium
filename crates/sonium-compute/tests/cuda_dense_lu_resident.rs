//! Device-input LU refresh preserves storage and rejects invalid factors.
#![cfg(feature = "cuda")]

// Rust guideline compliant 2026-02-21
use num_complex::Complex64;
use sonium_compute::{Backend, CudaBackend, CudaFailureInjectionPoint, Transpose};

#[test]
fn native_dense_lu_refresh_actions_and_failure_recovery() {
    let backend = match CudaBackend::new_default() {
        Ok(backend) => backend,
        Err(error) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").is_err(),
                "CUDA required: {error}"
            );
            return;
        }
    };
    let c = Complex64::new;
    let mut matrix = backend.raw_alloc::<Complex64>(4).unwrap();
    let mut rhs = backend.raw_alloc::<Complex64>(2).unwrap();
    let mut out = backend.raw_alloc::<Complex64>(2).unwrap();
    let original = [c(4.0, 1.0), c(1.0, -0.2), c(-0.3, 0.4), c(3.0, -0.5)];
    backend.try_copy_h2d(&original, &mut matrix).unwrap();
    let mut lu = backend.dense_lu_c64_from_device(2, &matrix).unwrap();
    let expected = [c(0.2, -0.1), c(-0.4, 0.3)];
    for pass in 0..3 {
        let mut a = original;
        a[0] += c(pass as f64, 0.0);
        backend.try_copy_h2d(&a, &mut matrix).unwrap();
        let before = backend.begin_runtime_stats();
        backend
            .dense_lu_c64_refactor_device(&mut lu, &matrix)
            .unwrap();
        let stats = backend.runtime_stats_since(before);
        assert_eq!(stats.allocation_operations, 0);
        assert_eq!(stats.transfer.h2d_bytes, 0);
        assert_eq!(stats.transfer.d2h_bytes, 8);
        for operation in [Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
            let b: Vec<_> = (0..2)
                .map(|i| {
                    (0..2)
                        .map(|j| {
                            let entry = match operation {
                                Transpose::NoTrans => a[i + 2 * j],
                                Transpose::Trans => a[j + 2 * i],
                                Transpose::ConjTrans => a[j + 2 * i].conj(),
                            };
                            entry * expected[j]
                        })
                        .sum::<Complex64>()
                })
                .collect();
            backend.try_copy_h2d(&b, &mut rhs).unwrap();
            backend
                .dense_lu_c64_solve(&lu, &rhs, &mut out, operation)
                .unwrap();
            let mut actual = [c(0.0, 0.0); 2];
            backend.try_copy_d2h(&out, &mut actual).unwrap();
            for i in 0..2 {
                assert!((actual[i] - expected[i]).norm() < 1e-12);
            }
        }
    }
    // A singular refresh mutates factor storage. It must never expose stale LU.
    backend
        .try_copy_h2d(&[c(0.0, 0.0); 4], &mut matrix)
        .unwrap();
    assert!(
        backend
            .dense_lu_c64_refactor_device(&mut lu, &matrix)
            .is_err()
    );
    let sentinel = [c(17.0, 2.0); 2];
    backend.try_copy_h2d(&sentinel, &mut out).unwrap();
    assert!(
        backend
            .dense_lu_c64_solve(&lu, &rhs, &mut out, Transpose::NoTrans)
            .is_err()
    );
    let mut actual = [c(0.0, 0.0); 2];
    backend.try_copy_d2h(&out, &mut actual).unwrap();
    assert_eq!(actual, sentinel);
    backend.try_copy_h2d(&original, &mut matrix).unwrap();
    backend
        .dense_lu_c64_refactor_device(&mut lu, &matrix)
        .unwrap();
    backend.inject_failure(CudaFailureInjectionPoint::TransferD2D);
    assert!(
        backend
            .dense_lu_c64_refactor_device(&mut lu, &matrix)
            .is_err()
    );
    backend.recover_after_error().unwrap();
    assert!(
        backend
            .dense_lu_c64_solve(&lu, &rhs, &mut out, Transpose::NoTrans)
            .is_err()
    );
    backend
        .dense_lu_c64_refactor_device(&mut lu, &matrix)
        .unwrap();
    backend
        .dense_lu_c64_solve(&lu, &rhs, &mut out, Transpose::NoTrans)
        .unwrap();
    // Rejected input before mutation leaves the current factors usable.
    backend
        .try_copy_h2d(&[c(f64::NAN, 0.0); 4], &mut matrix)
        .unwrap();
    assert!(
        backend
            .dense_lu_c64_refactor_device(&mut lu, &matrix)
            .is_err()
    );
    backend
        .dense_lu_c64_solve(&lu, &rhs, &mut out, Transpose::NoTrans)
        .unwrap();
}
