//! Native residual update parity, residency and failure contracts.

// Rust guideline compliant 2026-02-21
use super::*;
use crate::MemoryPoolConfig;

fn backend() -> Option<CudaBackend> {
    match CudaBackend::new(0) {
        Ok(b) => Some(b),
        Err(error) => {
            assert!(
                std::env::var_os("SONIUM_REQUIRE_CUDA").is_none(),
                "CUDA required: {error}"
            );
            None
        }
    }
}

#[test]
fn native_residual_update_all_scalars_and_tails() {
    fn check<T: DeviceScalar>(pool: &MemoryPool<CudaBackend>, values: [T; 4], tolerance: f64) {
        use num_traits::ToPrimitive;
        let backend = pool.backend();
        for n in [0, 1, 257, 4099] {
            let rhs: Vec<_> = (0..n).map(|i| values[i % 4]).collect();
            let ax: Vec<_> = (0..n).map(|i| values[(i + 1) % 4]).collect();
            let mut b = pool.try_alloc(n).unwrap();
            let mut r = pool.try_alloc(n).unwrap();
            backend.try_copy_h2d(&rhs, b.raw_mut()).unwrap();
            backend.try_copy_h2d(&ax, r.raw_mut()).unwrap();
            let before = backend.begin_runtime_stats();
            assert!(
                backend
                    .try_residual_update_native(b.raw(), r.raw_mut())
                    .unwrap()
            );
            let stats = backend.runtime_stats_since(before);
            assert_eq!(stats.allocation_operations, 0);
            assert_eq!(stats.transfer.h2d_bytes, 0);
            assert_eq!(stats.transfer.d2h_bytes, 0);
            assert_eq!(stats.kernel_launches, usize::from(n > 0));
            assert_eq!(stats.library_operations, 0);
            let mut actual = vec![T::zero(); n];
            backend.try_copy_d2h(r.raw(), &mut actual).unwrap();
            for ((actual, b), ax) in actual.into_iter().zip(rhs).zip(ax) {
                let expected = b - ax;
                if expected.is_finite() {
                    assert!(
                        (actual - expected).magnitude().to_f64().unwrap()
                            <= tolerance * (1.0 + expected.magnitude().to_f64().unwrap())
                    );
                } else {
                    assert!(!actual.is_finite());
                }
            }
        }
    }
    let Some(backend) = backend() else { return };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 24)).unwrap();
    check(&pool, [0.3_f32, -2.1, 0.0, f32::NAN], 1e-6);
    check(&pool, [0.3_f64, -2.1, 0.0, f64::INFINITY], 1e-14);
    check(
        &pool,
        [
            Complex32::new(0.3, -0.7),
            Complex32::new(-2.1, 0.4),
            Complex32::new(0.0, 0.0),
            Complex32::new(1.0, f32::NAN),
        ],
        1e-6,
    );
    check(
        &pool,
        [
            Complex64::new(0.3, -0.7),
            Complex64::new(-2.1, 0.4),
            Complex64::new(0.0, 0.0),
            Complex64::new(f64::INFINITY, 1.0),
        ],
        1e-14,
    );
}

#[test]
fn native_residual_update_rejections_recovery_and_missing_module() {
    let Some(backend) = backend() else { return };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
    let b = pool.backend();
    let mut rhs = pool.try_alloc::<f64>(3).unwrap();
    let mut output = pool.try_alloc::<f64>(3).unwrap();
    b.try_copy_h2d(&[2.; 3], rhs.raw_mut()).unwrap();
    b.try_copy_h2d(&[7.; 3], output.raw_mut()).unwrap();
    let short = pool.try_alloc::<f64>(2).unwrap();
    let foreign_pool = MemoryPool::new(
        CudaBackend::new(0).unwrap(),
        MemoryPoolConfig::with_cap(1 << 20),
    )
    .unwrap();
    let foreign = foreign_pool.try_alloc::<f64>(3).unwrap();
    let before = b.begin_runtime_stats();
    assert!(
        b.try_residual_update_native(short.raw(), output.raw_mut())
            .is_err()
    );
    assert!(
        b.try_residual_update_native(foreign.raw(), output.raw_mut())
            .is_err()
    );
    assert_eq!(b.runtime_stats_since(before).kernel_launches, 0);
    b.inject_failure(CudaFailureInjectionPoint::Launch);
    assert!(
        b.try_residual_update_native(rhs.raw(), output.raw_mut())
            .is_err()
    );
    b.recover_after_error().unwrap();
    let mut actual = [0.; 3];
    b.try_copy_d2h(output.raw(), &mut actual).unwrap();
    assert_eq!(actual, [7.; 3]);
    assert!(
        b.try_residual_update_native(rhs.raw(), output.raw_mut())
            .unwrap()
    );
    b.try_copy_d2h(output.raw(), &mut actual).unwrap();
    assert_eq!(actual, [-5.; 3]);

    let mut unavailable = CudaBackend::new(0).unwrap();
    unavailable.local_kernels = None;
    let pool = MemoryPool::new(unavailable, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
    let mut rhs = pool.try_alloc::<f64>(3).unwrap();
    let mut output = pool.try_alloc::<f64>(3).unwrap();
    pool.backend()
        .try_copy_h2d(&[2.; 3], rhs.raw_mut())
        .unwrap();
    pool.backend()
        .try_copy_h2d(&[7.; 3], output.raw_mut())
        .unwrap();
    let before = pool.backend().begin_runtime_stats();
    assert!(
        !pool
            .backend()
            .try_residual_update_native(rhs.raw(), output.raw_mut())
            .unwrap()
    );
    let stats = pool.backend().runtime_stats_since(before);
    assert_eq!(stats.kernel_launches, 0);
    assert_eq!(stats.allocation_operations, 0);
    assert_eq!(stats.transfer.h2d_bytes, 0);
    assert_eq!(stats.transfer.d2h_bytes, 0);
    pool.backend()
        .try_copy_d2h(output.raw(), &mut actual)
        .unwrap();
    assert_eq!(actual, [7.; 3]);
}
