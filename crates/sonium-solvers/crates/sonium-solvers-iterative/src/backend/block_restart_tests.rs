//! Multiple-RHS restart regression and retained workspace checks.

// Rust guideline compliant 2026-02-21
use super::*;
use num_traits::{FromPrimitive, ToPrimitive};
use sonium_compute::{CpuBackend, DeviceCsr, IdentityDevicePreconditioner, MemoryPoolConfig};

mod guards;

#[test]
fn block_restart_preserves_every_rhs() {
    let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
    let n = 8;
    let columns: Vec<i32> = (0..n as i32).collect();
    let rows: Vec<i32> = (0..=n as i32).collect();
    let diagonal: Vec<f64> = (0..n).map(|i| 1.0 + 0.15 * i as f64).collect();
    let matrix =
        DeviceCsr::try_from_host(pool.backend(), n, n, &rows, &columns, &diagonal).unwrap();
    let exact: Vec<_> = (0..2 * n)
        .map(|i| ((37 * i + 3) % 17) as f64 / 17.0)
        .collect();
    let host_rhs: Vec<_> = exact
        .iter()
        .enumerate()
        .map(|(i, x)| diagonal[i % n] * x)
        .collect();
    let mut rhs = pool.try_alloc(2 * n).unwrap();
    let mut output = pool.try_alloc(2 * n).unwrap();
    pool.backend()
        .try_copy_h2d(&host_rhs, rhs.raw_mut())
        .unwrap();
    let mut workspace = BackendKrylovWorkspace::new(n);
    let status = try_block_gmres_backend(
        pool.backend(),
        &mut workspace,
        &matrix,
        &IdentityDevicePreconditioner,
        rhs.raw(),
        None,
        output.raw_mut(),
        2,
        &BackendKrylovConfig {
            restart: 1,
            max_iterations: 100,
            tolerance: 1e-9,
            require_native: true,
            exact_residual_check_interval: 0,
        },
        None,
    )
    .unwrap();
    assert!(status.converged, "{status:?}");
    assert!(status.restarts > 0);
    let mut actual = vec![0.0; 2 * n];
    pool.backend()
        .try_copy_d2h(output.raw(), &mut actual)
        .unwrap();
    assert!(actual.iter().zip(exact).all(|(a, x)| (a - x).abs() < 1e-8));
}

fn retained<B: Backend, T: DeviceScalar + std::ops::AddAssign>(
    backend: B,
    value: impl Fn(f64, f64) -> T,
    tolerance: f64,
    mut observe: impl FnMut(&B, bool, usize),
) {
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 24)).unwrap();
    let backend = pool.backend();
    let n = 8;
    let rows: Vec<_> = (0..=n).collect();
    let cols: Vec<_> = (0..n).collect();
    let diagonal: Vec<_> = (0..n)
        .map(|i| value(1.0 + 0.15 * i as f64, 0.02 * i as f64))
        .collect();
    let matrix = sonium_compute::PersistentDeviceCsr::try_from_host_csr_ref(
        backend,
        sonium_compute::HostCsrRef::new(n, n, &rows, &cols, &diagonal),
    )
    .unwrap();
    let mut workspace = BackendKrylovWorkspace::with_pool(n, &pool);
    let config = BackendKrylovConfig {
        restart: 1,
        max_iterations: 100,
        tolerance: T::Real::from_f64(tolerance).unwrap(),
        require_native: true,
        exact_residual_check_interval: 0,
    };
    for flexible in [false, true] {
        for width in [2, 3, 2] {
            let mut rhs = pool.try_alloc(n * width).unwrap();
            let mut output = pool.try_alloc(n * width).unwrap();
            let mut first_capacity = None;
            for pass in 0..6 {
                let variant = pass / 2;
                let exact: Vec<_> = (0..n * width)
                    .map(|i| {
                        value(
                            ((37 * i + 3 + 5 * variant) % 17) as f64 / 17.0,
                            ((13 * i + 7 + 3 * variant) % 11) as f64 / 13.0,
                        )
                    })
                    .collect();
                let host: Vec<_> = exact
                    .iter()
                    .enumerate()
                    .map(|(i, x)| diagonal[i % n] * *x)
                    .collect();
                backend.try_copy_h2d(&host, rhs.raw_mut()).unwrap();
                backend
                    .try_copy_h2d(&vec![value(123.0, -7.0); n * width], output.raw_mut())
                    .unwrap();
                observe(backend, true, pass);
                let solve = if flexible {
                    try_block_fgmres_backend
                } else {
                    try_block_gmres_backend
                };
                let status = solve(
                    backend,
                    &mut workspace,
                    &matrix,
                    &IdentityDevicePreconditioner,
                    rhs.raw(),
                    None,
                    output.raw_mut(),
                    width,
                    &config,
                    None,
                )
                .unwrap();
                observe(backend, false, pass);
                assert!(status.converged && status.restarts > 0, "{status:?}");
                assert_eq!(
                    workspace.block_restart.as_ref().unwrap().raw().len(),
                    n * width
                );
                assert!(matches!(
                    workspace.block_restart,
                    Some(BlockRestart::Pooled(_))
                ));
                let mut actual = vec![T::zero(); n * width];
                backend.try_copy_d2h(output.raw(), &mut actual).unwrap();
                for column in 0..width {
                    let mut error = 0.0;
                    let mut norm = 0.0;
                    for (row, diagonal_value) in diagonal.iter().enumerate() {
                        let i = row + n * column;
                        error += (*diagonal_value * actual[i] - host[i])
                            .magnitude()
                            .to_f64()
                            .unwrap()
                            .powi(2);
                        norm += host[i].magnitude().to_f64().unwrap().powi(2);
                        assert!(
                            (actual[i] - exact[i]).magnitude().to_f64().unwrap() < 40.0 * tolerance
                        );
                    }
                    assert!((error / norm).sqrt() < 2.0 * tolerance);
                }
                let capacity = workspace.block_metrics();
                if pass % 2 == 1 {
                    assert_eq!(Some(capacity), first_capacity);
                }
                first_capacity = Some(capacity);
                eprintln!(
                    "block restart scalar={} flexible={flexible} width={width} pass={pass} iterations={} restarts={} residual={:?}",
                    std::any::type_name::<T>(),
                    status.iterations,
                    status.restarts,
                    status.true_residual
                );
            }
        }
    }
    workspace.clear().unwrap();
    assert!(workspace.block_restart.is_none());
    assert_eq!(workspace.block_metrics(), (0, 0));
    assert_eq!(pool.bytes_in_use(), 0);
}

#[test]
fn pooled_block_restarts_all_scalars_and_width_changes() {
    retained(CpuBackend, |r, _| r, 1e-9, |_, _, _| {});
    retained(CpuBackend, |r, _| r as f32, 2e-5, |_, _, _| {});
    retained(CpuBackend, num_complex::Complex64::new, 1e-9, |_, _, _| {});
    retained(
        CpuBackend,
        |r, i| num_complex::Complex32::new(r as f32, i as f32),
        2e-5,
        |_, _, _| {},
    );
}

#[cfg(feature = "cuda")]
#[test]
fn native_pooled_block_restarts_reuse_allocations() {
    use sonium_compute::CudaBackend;
    fn run<T: DeviceScalar + std::ops::AddAssign>(value: impl Fn(f64, f64) -> T, tolerance: f64) {
        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend,
            Err(error) => {
                assert!(
                    std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                    "CUDA required: {error}"
                );
                return;
            }
        };
        let mut before = None;
        retained(backend, value, tolerance, |backend, start, pass| {
            if start {
                before = Some(backend.begin_runtime_stats());
            } else {
                let stats = backend.runtime_stats_since(before.take().unwrap());
                if pass % 2 == 1 {
                    assert_eq!(stats.allocation_operations, 0, "{stats:?}");
                }
            }
        });
    }
    run(|r, _| r, 1e-9);
    run(|r, _| r as f32, 2e-5);
    run(num_complex::Complex64::new, 1e-9);
    run(|r, i| num_complex::Complex32::new(r as f32, i as f32), 2e-5);
}
