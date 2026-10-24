//! Restart state and quota boundaries independent of convergence acceleration.

// Rust guideline compliant 2026-02-21
use super::*;

fn fixture(cap: usize) -> (MemoryPool<CpuBackend>, DeviceCsr<CpuBackend, f64>) {
    let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(cap)).unwrap();
    let rows: Vec<_> = (0..=8).collect();
    let cols: Vec<_> = (0..8).collect();
    let values: Vec<_> = (0..8).map(|i| 1.0 + 0.15 * i as f64).collect();
    let matrix = DeviceCsr::try_from_host(pool.backend(), 8, 8, &rows, &cols, &values).unwrap();
    (pool, matrix)
}

fn config(max_iterations: usize) -> BackendKrylovConfig<f64> {
    BackendKrylovConfig {
        restart: 1,
        max_iterations,
        tolerance: 1e-9,
        require_native: true,
        exact_residual_check_interval: 0,
    }
}

#[test]
fn cached_block_restart_never_overrides_the_callers_initial_guess() {
    let (pool, matrix) = fixture(1 << 20);
    let mut rhs = pool.try_alloc(16).unwrap();
    let mut output = pool.try_alloc(16).unwrap();
    let mut fresh_output = pool.try_alloc(16).unwrap();
    let mut initial = pool.try_alloc(16).unwrap();
    let old: Vec<_> = (0..16).map(|i| ((37 * i + 3) % 17) as f64 / 17.0).collect();
    pool.backend().try_copy_h2d(&old, rhs.raw_mut()).unwrap();
    let mut reused = BackendKrylovWorkspace::with_pool(8, &pool);
    let status = try_block_gmres_backend(
        pool.backend(),
        &mut reused,
        &matrix,
        &IdentityDevicePreconditioner,
        rhs.raw(),
        None,
        output.raw_mut(),
        2,
        &config(100),
        None,
    )
    .unwrap();
    assert!(status.converged && status.restarts > 0);
    assert!(reused.block_restart.is_some());
    let changed: Vec<_> = (0..16).map(|i| ((13 * i + 7) % 19) as f64 / 9.0).collect();
    pool.backend()
        .try_copy_h2d(&changed, rhs.raw_mut())
        .unwrap();
    pool.backend()
        .try_copy_h2d(&[0.25; 16], initial.raw_mut())
        .unwrap();
    for x0 in [None, Some(initial.raw())] {
        let mut fresh = BackendKrylovWorkspace::with_pool(8, &pool);
        let a = try_block_gmres_backend(
            pool.backend(),
            &mut reused,
            &matrix,
            &IdentityDevicePreconditioner,
            rhs.raw(),
            x0,
            output.raw_mut(),
            2,
            &config(1),
            None,
        )
        .unwrap();
        let b = try_block_gmres_backend(
            pool.backend(),
            &mut fresh,
            &matrix,
            &IdentityDevicePreconditioner,
            rhs.raw(),
            x0,
            fresh_output.raw_mut(),
            2,
            &config(1),
            None,
        )
        .unwrap();
        assert!(!a.converged && !b.converged);
        assert_eq!(a.restarts, 0);
        assert!(fresh.block_restart.is_none());
        let mut actual = [0.0; 16];
        let mut reference = [0.0; 16];
        pool.backend()
            .try_copy_d2h(output.raw(), &mut actual)
            .unwrap();
        pool.backend()
            .try_copy_d2h(fresh_output.raw(), &mut reference)
            .unwrap();
        assert!(
            actual
                .iter()
                .zip(reference)
                .all(|(a, b)| (a - b).abs() < 1e-13)
        );
        assert!((a.true_residual - b.true_residual).abs() < 1e-13);
    }
}

#[test]
fn pooled_restart_quota_is_checked_only_when_another_cycle_is_needed() {
    // Measure one cycle including its scalar-recovery scratch. The constrained
    // run admits that same cycle but misses the complete restart by one byte.
    let cycle_bytes = {
        let (pool, matrix) = fixture(1 << 20);
        let mut rhs = pool.try_alloc(16).unwrap();
        let mut output = pool.try_alloc(16).unwrap();
        let host: Vec<_> = (0..16).map(|i| ((37 * i + 3) % 17) as f64 / 17.0).collect();
        pool.backend().try_copy_h2d(&host, rhs.raw_mut()).unwrap();
        let mut workspace = BackendKrylovWorkspace::with_pool(8, &pool);
        let status = try_block_gmres_backend(
            pool.backend(),
            &mut workspace,
            &matrix,
            &IdentityDevicePreconditioner,
            rhs.raw(),
            None,
            output.raw_mut(),
            2,
            &config(1),
            None,
        )
        .unwrap();
        assert!(!status.converged);
        assert!(workspace.block_restart.is_none());
        pool.bytes_in_use()
    };
    let (pool, matrix) = fixture(cycle_bytes + 16 * 8 - 1);
    let mut rhs = pool.try_alloc(16).unwrap();
    let mut output = pool.try_alloc(16).unwrap();
    let host: Vec<_> = (0..16).map(|i| ((37 * i + 3) % 17) as f64 / 17.0).collect();
    pool.backend().try_copy_h2d(&host, rhs.raw_mut()).unwrap();
    let mut workspace = BackendKrylovWorkspace::with_pool(8, &pool);
    let error = try_block_gmres_backend(
        pool.backend(),
        &mut workspace,
        &matrix,
        &IdentityDevicePreconditioner,
        rhs.raw(),
        None,
        output.raw_mut(),
        2,
        &config(100),
        None,
    )
    .unwrap_err();
    assert!(
        matches!(
            error,
            BackendKrylovError::Block {
                source: BlockGmresError::Allocation {
                    label: "block restart initial",
                    ..
                },
                ..
            }
        ),
        "{error:?}"
    );
    assert!(workspace.block_restart.is_none());
    assert_eq!(pool.bytes_in_use(), cycle_bytes);
    let last = try_block_gmres_backend(
        pool.backend(),
        &mut workspace,
        &matrix,
        &IdentityDevicePreconditioner,
        rhs.raw(),
        None,
        output.raw_mut(),
        2,
        &config(1),
        None,
    )
    .unwrap();
    assert!(!last.converged);
    assert_eq!(last.restarts, 0);
    assert!(workspace.block_restart.is_none());
    assert_eq!(pool.bytes_in_use(), cycle_bytes);
}
