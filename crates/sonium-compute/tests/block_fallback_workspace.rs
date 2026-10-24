//! Scalar recovery inside block GMRES retains its device scratch.

// Rust guideline compliant 2026-02-21
use num_complex::{Complex32, Complex64};
use num_traits::{Float, FromPrimitive};
use sonium_compute::gmres::IdentityBackendPreconditioner;
use sonium_compute::{
    Backend, BlockGmresConfig, BlockGmresWorkspace, CpuBackend, DeviceCsr, DeviceScalar,
    MemoryPool, MemoryPoolConfig, PrebakedSpmv, gmres_backend_block_preconditioned_with_workspace,
};

fn repeated<B: Backend, T: DeviceScalar + std::ops::AddAssign + PartialEq>(
    pool: &MemoryPool<B>,
    tolerance: f64,
    counters: impl Fn() -> [usize; 3],
) {
    let backend = pool.backend();
    let real = |value| T::Real::from_f64(value).unwrap();
    let scalar = |value| T::from_real(real(value));
    // Enough rows to distinguish bulk transfers from compact Krylov data.
    let n = 257;
    let rows: Vec<_> = (0..=n as i32).collect();
    let columns: Vec<_> = (0..n as i32).collect();
    let diagonal: Vec<_> = (0..n).map(|i| scalar((i % 4 + 2) as f64)).collect();
    let matrix = DeviceCsr::try_from_host(backend, n, n, &rows, &columns, &diagonal).unwrap();
    let a = PrebakedSpmv::try_from_csr(backend, &matrix).unwrap();
    let mut rhs = backend.raw_alloc::<T>(n * 3).unwrap();
    let mut output = backend.raw_alloc::<T>(n * 3).unwrap();
    let mut workspace = BlockGmresWorkspace::with_fallback_pool(n, pool);
    let config = BlockGmresConfig {
        max_block_iterations: 8,
        tolerance: real(tolerance),
    };
    for trial in 0..3 {
        // Dependent columns force scalar recovery; the zero column is deflated.
        let scale = 1.0 + trial as f64 * 0.25;
        let expected: Vec<T> = (0..n * 3)
            .map(|i| {
                if i < n * 2 {
                    scalar((1.0 + (i % n % 7) as f64 * 0.05) * (i / n + 1) as f64 * scale)
                } else {
                    scalar(0.0)
                }
            })
            .collect();
        let b: Vec<T> = expected
            .iter()
            .enumerate()
            .map(|(i, &x)| x * diagonal[i % n])
            .collect();
        backend.try_copy_h2d(&b, &mut rhs).unwrap();
        let before = counters();
        let status = gmres_backend_block_preconditioned_with_workspace(
            backend,
            &a,
            &IdentityBackendPreconditioner,
            &rhs,
            None,
            &mut output,
            3,
            &config,
            &mut workspace,
        )
        .unwrap();
        assert!(status.converged, "{status:?}");
        assert!(workspace.fallback_cached_bytes().unwrap() > 0);
        let after = counters();
        if trial > 0 {
            assert_eq!(after[0], before[0], "warm fallback allocated");
        }
        // Validation and zero initialization are native on CUDA. Only compact
        // status words and scalar Krylov coefficients may cross the boundary.
        let scalar_bytes = std::mem::size_of::<T>();
        assert!(
            after[1] - before[1] <= 256 * scalar_bytes + 8,
            "recovery uploaded full columns: {before:?} -> {after:?}"
        );
        assert!(
            after[2] - before[2] <= 256 * scalar_bytes + 8,
            "recovery downloaded full columns: {before:?} -> {after:?}"
        );
        let mut actual = vec![scalar(0.0); n * 3];
        backend.try_copy_d2h(&output, &mut actual).unwrap();
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).norm_sqr().sqrt() < real(30.0 * tolerance));
        }
    }
    let mut guess = backend.raw_alloc::<T>(n * 3).unwrap();
    backend.try_copy_d2d(&output, &mut guess).unwrap();
    let before = counters();
    let warm = gmres_backend_block_preconditioned_with_workspace(
        backend,
        &a,
        &IdentityBackendPreconditioner,
        &rhs,
        Some(&guess),
        &mut output,
        3,
        &config,
        &mut workspace,
    )
    .unwrap();
    assert!(warm.converged);
    assert_eq!(warm.block_iterations, 0);
    let after = counters();
    assert_eq!(after[0], before[0], "exact warm start allocated");
    assert_eq!(
        after[1], before[1],
        "exact warm start reuploaded its solution"
    );
    let mut expected_output = vec![T::zero(); n * 3];
    backend.try_copy_d2h(&output, &mut expected_output).unwrap();
    let mut original_rhs = vec![T::zero(); n * 3];
    backend.try_copy_d2h(&rhs, &mut original_rhs).unwrap();
    let mut invalid = original_rhs.clone();
    invalid[n * 3 - 1] = T::from_real(T::Real::nan());
    backend.try_copy_h2d(&invalid, &mut rhs).unwrap();
    let error = gmres_backend_block_preconditioned_with_workspace(
        backend,
        &a,
        &IdentityBackendPreconditioner,
        &rhs,
        None,
        &mut output,
        3,
        &config,
        &mut workspace,
    )
    .unwrap_err();
    assert!(
        matches!(error, sonium_compute::BlockGmresError::InvalidArgument(message) if message.contains("RHS"))
    );
    backend.try_copy_h2d(&original_rhs, &mut rhs).unwrap();
    backend.try_copy_h2d(&invalid, &mut guess).unwrap();
    let error = gmres_backend_block_preconditioned_with_workspace(
        backend,
        &a,
        &IdentityBackendPreconditioner,
        &rhs,
        Some(&guess),
        &mut output,
        3,
        &config,
        &mut workspace,
    )
    .unwrap_err();
    assert!(
        matches!(error, sonium_compute::BlockGmresError::InvalidArgument(message) if message.contains("initial guess"))
    );
    let mut preserved = vec![T::zero(); n * 3];
    backend.try_copy_d2h(&output, &mut preserved).unwrap();
    assert_eq!(
        preserved, expected_output,
        "invalid input changed published output"
    );
    workspace.clear();
    assert_eq!(workspace.device_buffer_count(), 0);
    assert_eq!(workspace.device_capacity(), 0);
    assert_eq!(workspace.fallback_cached_bytes().unwrap(), 0);
    assert_eq!(pool.bytes_in_use(), 0);
}

#[test]
fn cpu_dependent_blocks_reuse_scalar_workspace_all_scalars() {
    let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(1 << 20)).unwrap();
    repeated::<_, f32>(&pool, 1e-5, || [0; 3]);
    repeated::<_, f64>(&pool, 1e-10, || [0; 3]);
    repeated::<_, Complex32>(&pool, 1e-5, || [0; 3]);
    repeated::<_, Complex64>(&pool, 1e-10, || [0; 3]);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_dependent_blocks_reuse_scalar_workspace_all_scalars() {
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
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
    let allocations = || {
        let stats = pool.backend().runtime_stats();
        [
            stats.allocation_operations,
            stats.transfer.h2d_bytes,
            stats.transfer.d2h_bytes,
        ]
    };
    repeated::<_, f32>(&pool, 1e-5, allocations);
    repeated::<_, f64>(&pool, 1e-10, allocations);
    repeated::<_, Complex32>(&pool, 1e-5, allocations);
    repeated::<_, Complex64>(&pool, 1e-10, allocations);
}

#[test]
fn fallback_pool_rejects_wrong_backend_and_reports_quota_failure() {
    let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(32)).unwrap();
    let backend = pool.backend();
    let a = DeviceCsr::try_from_host(backend, 2, 2, &[0, 1, 2], &[0, 1], &[2.0_f64, 3.0]).unwrap();
    let mut rhs = backend.raw_alloc::<f64>(4).unwrap();
    let mut out = backend.raw_alloc::<f64>(4).unwrap();
    backend
        .try_copy_h2d(&[2.0, 6.0, 4.0, 12.0], &mut rhs)
        .unwrap();
    backend.try_copy_h2d(&[7.0; 4], &mut out).unwrap();
    let mut workspace = BlockGmresWorkspace::with_fallback_pool(2, &pool);
    let config = BlockGmresConfig {
        max_block_iterations: 4,
        tolerance: 1e-10,
    };
    let error = gmres_backend_block_preconditioned_with_workspace(
        &CpuBackend::new(),
        &a,
        &IdentityBackendPreconditioner,
        &rhs,
        None,
        &mut out,
        2,
        &config,
        &mut workspace,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        sonium_compute::BlockGmresError::InvalidArgument(_)
    ));
    let mut untouched = [0.0; 4];
    backend.try_copy_d2h(&out, &mut untouched).unwrap();
    assert_eq!(untouched, [7.0; 4]);
    let error = gmres_backend_block_preconditioned_with_workspace(
        backend,
        &a,
        &IdentityBackendPreconditioner,
        &rhs,
        None,
        &mut out,
        2,
        &config,
        &mut workspace,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        sonium_compute::BlockGmresError::Allocation { .. }
    ));
    workspace.clear();
    assert_eq!(pool.bytes_in_use(), 0);
}
