//! Fresh-hierarchy parity and failed-update recovery for retained AMG.

// Rust guideline compliant 2026-02-21
use super::*;
use sonium_compute::{
    Backend, CpuBackend, DevicePreconditioner, MemoryPool, MemoryPoolConfig,
    PreconditionerFailureInjectionPoint,
};

fn matrix(n: usize, shift: f64) -> CsrMatrix<f64> {
    let mut entries = Vec::new();
    for i in 0..n {
        entries.push((i, i, 3.0 + shift + 0.01 * i as f64));
        if i + 1 < n {
            entries.extend([(i, i + 1, -1.0), (i + 1, i, -1.0)]);
        }
    }
    CsrMatrix::from_triplets(n, n, entries)
}

fn check<B: Backend>(
    pool: &MemoryPool<B>,
    allocations: impl Fn() -> usize,
    inject: impl Fn(),
    recover: impl Fn(),
) {
    let backend = pool.backend();
    let n = 48;
    let mut rhs = pool.try_alloc::<f64>(2 * n).unwrap();
    let mut actual = pool.try_alloc::<f64>(2 * n).unwrap();
    let mut expected = pool.try_alloc::<f64>(2 * n).unwrap();
    backend
        .try_copy_h2d(
            &(0..2 * n).map(|i| (i as f64).sin()).collect::<Vec<_>>(),
            rhs.raw_mut(),
        )
        .unwrap();
    let mut single_rhs = pool.try_alloc::<f64>(n).unwrap();
    let mut single_output = pool.try_alloc::<f64>(n).unwrap();
    backend
        .try_copy_h2d(&vec![1.0; n], single_rhs.raw_mut())
        .unwrap();
    for smoother in [
        AmgSmoother::Jacobi,
        AmgSmoother::L1Jacobi,
        AmgSmoother::Chebyshev,
    ] {
        let config = AmgConfig {
            smoother,
            coarse_size: 2,
            ..Default::default()
        };
        let initial = AmgPreconditioner::try_from_csr(&matrix(n, 0.0), config.clone()).unwrap();
        assert!(initial.levels.len() >= 3);
        let mut retained = BackendAmgPreconditioner::try_from_host_amg(backend, &initial).unwrap();
        for pass in 1..=3 {
            let host =
                AmgPreconditioner::try_from_csr(&matrix(n, 0.01 * pass as f64), config.clone())
                    .unwrap();
            let fresh = BackendAmgPreconditioner::try_from_host_amg(backend, &host).unwrap();
            let before = allocations();
            assert!(retained.try_refresh_from_host_amg(backend, &host).unwrap());
            assert_eq!(allocations(), before, "refresh allocated device storage");
            retained
                .try_apply_colmajor(backend, 2, rhs.raw(), actual.raw_mut())
                .unwrap();
            fresh
                .try_apply_colmajor(backend, 2, rhs.raw(), expected.raw_mut())
                .unwrap();
            let mut a = vec![0.0; 2 * n];
            let mut b = a.clone();
            backend.try_copy_d2h(actual.raw(), &mut a).unwrap();
            backend.try_copy_d2h(expected.raw(), &mut b).unwrap();
            for (a, b) in a.iter().zip(b) {
                assert!((a - b).abs() < 1e-12 * (1.0 + b.abs()));
            }
            let restored = BackendAmgPreconditioner::<B, f64>::try_from_checkpoint_bytes(
                backend,
                &retained.to_checkpoint_bytes().unwrap(),
            )
            .unwrap();
            restored
                .try_apply_colmajor(backend, 2, rhs.raw(), expected.raw_mut())
                .unwrap();
            backend.try_copy_d2h(expected.raw(), &mut a).unwrap();
            let mut original = vec![0.0; 2 * n];
            backend.try_copy_d2h(actual.raw(), &mut original).unwrap();
            for (a, b) in a.iter().zip(original) {
                assert!((a - b).abs() < 1e-12 * (1.0 + b.abs()));
            }
        }
        let changed = AmgPreconditioner::try_from_csr(&matrix(n + 1, 0.0), config.clone()).unwrap();
        let checkpoint = retained.to_checkpoint_bytes().unwrap();
        assert!(
            !retained
                .try_refresh_from_host_amg(backend, &changed)
                .unwrap()
        );
        assert_eq!(checkpoint, retained.to_checkpoint_bytes().unwrap());
        let mut invalid = AmgPreconditioner::<f64>::from_checkpoint_bytes(
            &initial.to_checkpoint_bytes().unwrap(),
        )
        .unwrap();
        invalid.levels[1].matrix.values[0] = f64::NAN;
        assert!(
            retained
                .try_refresh_from_host_amg(backend, &invalid)
                .is_err()
        );
        assert_eq!(checkpoint, retained.to_checkpoint_bytes().unwrap());
        inject();
        assert!(
            retained
                .try_refresh_from_host_amg(backend, &initial)
                .is_err()
        );
        recover();
        backend
            .try_copy_h2d(&vec![42.0; 2 * n], actual.raw_mut())
            .unwrap();
        assert!(retained.to_checkpoint_bytes().is_err());
        backend
            .try_copy_h2d(&vec![42.0; n], single_output.raw_mut())
            .unwrap();
        assert!(
            retained
                .try_apply(backend, single_rhs.raw(), single_output.raw_mut())
                .is_err()
        );
        let mut single_sentinel = vec![0.0; n];
        backend
            .try_copy_d2h(single_output.raw(), &mut single_sentinel)
            .unwrap();
        assert!(single_sentinel.iter().all(|v| *v == 42.0));
        assert!(
            retained
                .try_apply_colmajor(backend, 2, rhs.raw(), actual.raw_mut())
                .is_err()
        );
        let mut sentinel = vec![0.0; 2 * n];
        backend.try_copy_d2h(actual.raw(), &mut sentinel).unwrap();
        assert!(sentinel.iter().all(|v| *v == 42.0));
        assert!(
            retained
                .try_refresh_from_host_amg(backend, &initial)
                .unwrap()
        );
        retained
            .try_apply_colmajor(backend, 2, rhs.raw(), actual.raw_mut())
            .unwrap();
    }
}

#[test]
fn cpu_amg_numeric_refresh() {
    let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(1 << 26)).unwrap();
    check(
        &pool,
        || 0,
        || {
            CpuBackend::inject_preconditioner_failure(
                PreconditionerFailureInjectionPoint::DiagonalUpload,
            )
        },
        || {},
    );
}

#[cfg(feature = "cuda")]
#[test]
fn native_amg_numeric_refresh() {
    let backend = match sonium_compute::CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            assert_ne!(
                std::env::var("SONIUM_REQUIRE_CUDA").as_deref(),
                Ok("1"),
                "{error}"
            );
            eprintln!("SKIP CUDA AMG refresh: {error}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 26)).unwrap();
    complex_check(&pool, || {
        pool.backend().runtime_stats().allocation_operations
    });
    check(
        &pool,
        || pool.backend().runtime_stats().allocation_operations,
        || {
            pool.backend()
                .inject_preconditioner_failure(PreconditionerFailureInjectionPoint::DiagonalUpload)
        },
        || pool.backend().recover_after_error().unwrap(),
    );
}

fn complex_check<B: Backend>(pool: &MemoryPool<B>, allocations: impl Fn() -> usize) {
    use num_complex::Complex64 as C;
    let n = 32;
    let backend = pool.backend();
    let mut rhs = pool.try_alloc::<C>(n).unwrap();
    let mut out = pool.try_alloc::<C>(n).unwrap();
    let mut expected = pool.try_alloc::<C>(n).unwrap();
    backend
        .try_copy_h2d(&vec![C::new(1.0, 0.5); n], rhs.raw_mut())
        .unwrap();
    let host = |shift| {
        let mut entries = Vec::new();
        for i in 0..n {
            entries.push((i, i, C::new(3.5 + shift, 0.3)));
            if i + 1 < n {
                entries.extend([
                    (i, i + 1, C::new(-1.0, 0.1)),
                    (i + 1, i, C::new(-1.0, -0.1)),
                ]);
            }
        }
        AmgPreconditioner::try_from_csr(
            &CsrMatrix::from_triplets(n, n, entries),
            AmgConfig {
                smoother: AmgSmoother::Chebyshev,
                coarse_size: 2,
                ..Default::default()
            },
        )
        .unwrap()
    };
    let initial = host(0.0);
    let changed = host(0.01);
    assert!(initial.levels.len() >= 3);
    let mut retained = BackendAmgPreconditioner::try_from_host_amg(backend, &initial).unwrap();
    let fresh = BackendAmgPreconditioner::try_from_host_amg(backend, &changed).unwrap();
    let before = allocations();
    assert!(
        retained
            .try_refresh_from_host_amg(backend, &changed)
            .unwrap()
    );
    assert_eq!(allocations(), before);
    retained
        .try_apply(backend, rhs.raw(), out.raw_mut())
        .unwrap();
    fresh
        .try_apply(backend, rhs.raw(), expected.raw_mut())
        .unwrap();
    let mut a = vec![C::default(); n];
    let mut b = a.clone();
    backend.try_copy_d2h(out.raw(), &mut a).unwrap();
    backend.try_copy_d2h(expected.raw(), &mut b).unwrap();
    for (a, b) in a.iter().zip(b) {
        assert!((*a - b).norm() < 1e-12 * (1.0 + b.norm()));
    }
}

#[test]
fn cpu_complex_amg_numeric_refresh() {
    let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(1 << 26)).unwrap();
    complex_check(&pool, || 0);
}
