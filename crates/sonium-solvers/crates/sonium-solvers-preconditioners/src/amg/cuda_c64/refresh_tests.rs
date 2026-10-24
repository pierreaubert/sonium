//! Retained complex coarse-factor refresh and failure recovery.

// Rust guideline compliant 2026-02-21
use super::*;
use sonium_compute::{
    DevicePreconditioner, MemoryPool, MemoryPoolConfig, PreconditionerFailureInjectionPoint,
};

#[test]
fn native_c64_coarse_numeric_refresh() {
    let backend = match CudaBackend::new_default() {
        Ok(backend) => backend,
        Err(error) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                "{error}"
            );
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 28)).unwrap();
    let backend = pool.backend();
    let n = 48;
    let mut entries = Vec::new();
    for i in 0..n {
        entries.push((i, i, Complex64::new(3.5, 0.2)));
        if i + 1 < n {
            entries.push((i, i + 1, Complex64::new(-1.0, 0.1)));
            entries.push((i + 1, i, Complex64::new(-0.8, -0.1)));
        }
    }
    let original = CsrMatrix::from_triplets(n, n, entries);
    let mut rhs = pool.try_alloc::<Complex64>(n).unwrap();
    let mut out = pool.try_alloc::<Complex64>(n).unwrap();
    let mut reference = pool.try_alloc::<Complex64>(n).unwrap();
    backend
        .try_copy_h2d(&vec![Complex64::new(1.0, 0.2); n], rhs.raw_mut())
        .unwrap();
    for coarse_size in [n, 4] {
        let config = AmgConfig {
            coarse_size,
            ..Default::default()
        };
        let mut retained =
            BackendAmgPreconditioner::<CudaBackend, Complex64>::try_from_csr_with_cuda_coarse_lu(
                backend,
                &original,
                config.clone(),
            )
            .unwrap();
        for scale in [1.125, 0.75] {
            let mut matrix = original.clone();
            for value in &mut matrix.values {
                *value *= scale;
            }
            let host = AmgPreconditioner::try_from_csr(&matrix, config.clone()).unwrap();
            let fresh = BackendAmgPreconditioner::<CudaBackend, Complex64>::try_from_csr_with_cuda_coarse_lu(backend, &matrix, config.clone()).unwrap();
            let before = backend.begin_runtime_stats();
            assert!(retained.try_refresh_from_host_amg(backend, &host).unwrap());
            let stats = backend.runtime_stats_since(before);
            assert_eq!(stats.allocation_operations, 0);
            // cuSOLVER returns one status integer; matrix data only uploads.
            assert_eq!(stats.transfer.d2h_bytes, 4);
            assert!(stats.transfer.h2d_bytes > 0);
            retained
                .try_apply(backend, rhs.raw(), out.raw_mut())
                .unwrap();
            fresh
                .try_apply(backend, rhs.raw(), reference.raw_mut())
                .unwrap();
            let mut actual = vec![Complex64::default(); n];
            let mut expected = actual.clone();
            backend.try_copy_d2h(out.raw(), &mut actual).unwrap();
            backend
                .try_copy_d2h(reference.raw(), &mut expected)
                .unwrap();
            for (a, b) in actual.iter().zip(&expected) {
                assert!((*a - *b).norm() < 1e-11);
            }
            let restored = BackendAmgPreconditioner::<CudaBackend, Complex64>::try_from_checkpoint_bytes_with_cuda_coarse_lu(backend, &retained.to_checkpoint_bytes().unwrap()).unwrap();
            assert!(restored.diagnostics().has_coarse_direct_solver);
            restored
                .try_apply(backend, rhs.raw(), reference.raw_mut())
                .unwrap();
            backend
                .try_copy_d2h(reference.raw(), &mut expected)
                .unwrap();
            for (a, b) in actual.iter().zip(&expected) {
                assert!((*a - *b).norm() < 1e-11);
            }
            eprintln!(
                "coarse refresh levels={} scale={scale} allocations={} h2d={} d2h={}",
                host.levels.len(),
                stats.allocation_operations,
                stats.transfer.h2d_bytes,
                stats.transfer.d2h_bytes
            );
            backend.inject_preconditioner_failure(PreconditionerFailureInjectionPoint::CoarseSolve);
            assert!(retained.try_refresh_from_host_amg(backend, &host).is_err());
            assert!(retained.to_checkpoint_bytes().is_err());
            assert!(
                retained
                    .try_apply(backend, rhs.raw(), out.raw_mut())
                    .is_err()
            );
            backend.recover_after_error().unwrap();
            assert!(retained.try_refresh_from_host_amg(backend, &host).unwrap());
            retained
                .try_apply(backend, rhs.raw(), out.raw_mut())
                .unwrap();
        }
    }
}

#[test]
fn native_c64_host_refactor_validation_and_singular_recovery() {
    let backend = match CudaBackend::new_default() {
        Ok(backend) => backend,
        Err(error) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                "{error}"
            );
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
    let backend = pool.backend();
    let values = [
        Complex64::new(2.0, 0.0),
        Complex64::default(),
        Complex64::default(),
        Complex64::new(4.0, 0.0),
    ];
    let mut lu = backend.dense_lu_c64_from_column_major(2, &values).unwrap();
    let mut rhs = pool.try_alloc::<Complex64>(2).unwrap();
    let mut out = pool.try_alloc::<Complex64>(2).unwrap();
    backend
        .try_copy_h2d(
            &[Complex64::new(2.0, 0.0), Complex64::new(8.0, 0.0)],
            rhs.raw_mut(),
        )
        .unwrap();
    let mut invalid = values;
    invalid[0].re = f64::NAN;
    for bad in [&invalid[..], &values[..3]] {
        assert!(
            backend
                .dense_lu_c64_refactor_column_major(&mut lu, bad)
                .is_err()
        );
        backend
            .dense_lu_c64_solve(&lu, rhs.raw(), out.raw_mut(), Transpose::NoTrans)
            .unwrap();
    }
    assert!(
        backend
            .dense_lu_c64_refactor_column_major(&mut lu, &[Complex64::default(); 4])
            .is_err()
    );
    assert!(
        backend
            .dense_lu_c64_solve(&lu, rhs.raw(), out.raw_mut(), Transpose::NoTrans)
            .is_err()
    );
    backend
        .dense_lu_c64_refactor_column_major(&mut lu, &values)
        .unwrap();
    backend
        .dense_lu_c64_solve(&lu, rhs.raw(), out.raw_mut(), Transpose::NoTrans)
        .unwrap();
    let mut actual = [Complex64::default(); 2];
    backend.try_copy_d2h(out.raw(), &mut actual).unwrap();
    assert!((actual[0] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
    assert!((actual[1] - Complex64::new(2.0, 0.0)).norm() < 1e-12);
}
