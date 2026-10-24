//! Regression coverage for variable device preconditioners and saved Z storage.

// Rust guideline compliant 2026-02-21
use super::*;
use num_traits::{FromPrimitive, ToPrimitive};
use sonium_compute::{CpuBackend, DeviceCsr, MemoryPoolConfig};
use std::sync::atomic::AtomicUsize;

struct ChangingScale(AtomicUsize);

impl<B: Backend, T: DeviceScalar> DevicePreconditioner<B, T> for ChangingScale {
    fn apply(&self, backend: &B, r: &B::Buffer<T>, z: &mut B::Buffer<T>) {
        self.try_apply(backend, r, z).unwrap();
    }

    fn try_apply(
        &self,
        backend: &B,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> sonium_compute::BackendResult<()> {
        let step = self.0.fetch_add(1, Ordering::Relaxed);
        let factor = [0.5, 2.0, 0.75, 1.25][step % 4];
        backend.try_copy_d2d(r, z)?;
        backend.try_scale(T::from_real(T::Real::from_f64(factor).unwrap()), z)
    }
}

fn exercise<B: Backend, T: DeviceScalar>(
    backend: B,
    value: impl Fn(f64, f64) -> T,
    tolerance: f64,
) {
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 22)).unwrap();
    let backend = pool.backend();
    // Nonsymmetric, genuinely complex for complex scalar instantiations.
    let values = [
        value(4., 0.4),
        value(1., -0.2),
        value(0.3, 0.1),
        value(3., -0.3),
        value(1., 0.2),
        value(0.2, -0.1),
        value(2., 0.3),
    ];
    let rows = [0, 2, 5, 7];
    let cols = [0, 1, 0, 1, 2, 1, 2];
    let matrix = DeviceCsr::try_from_host(backend, 3, 3, &rows, &cols, &values).unwrap();
    let mut rhs = pool.try_alloc::<T>(3).unwrap();
    let mut out = pool.try_alloc::<T>(3).unwrap();
    let mut workspace = BackendKrylovWorkspace::new(3);
    let preconditioner = ChangingScale(AtomicUsize::new(0));
    for restart in [3, 1] {
        let config = BackendKrylovConfig {
            restart,
            max_iterations: 100,
            tolerance: T::Real::from_f64(tolerance).unwrap(),
            require_native: true,
            exact_residual_check_interval: 0,
        };
        let mut retained = None;
        for repeat in 0..3 {
            let exact = [
                value(1. + repeat as f64, 0.2),
                value(-2., 0.7),
                value(0.5, -0.3),
            ];
            let mut b = [T::zero(); 3];
            for row in 0..3 {
                for index in rows[row] as usize..rows[row + 1] as usize {
                    b[row] = b[row] + values[index] * exact[cols[index] as usize];
                }
            }
            backend.try_copy_h2d(&b, rhs.raw_mut()).unwrap();
            let calls_before = preconditioner.0.load(Ordering::Relaxed);
            let status = try_fgmres_backend_preconditioned(
                backend,
                &pool,
                &mut workspace,
                &matrix,
                &preconditioner,
                rhs.raw(),
                None,
                out.raw_mut(),
                &config,
                None,
            )
            .unwrap();
            assert!(status.converged, "{status:?}");
            assert_eq!(
                preconditioner.0.load(Ordering::Relaxed) - calls_before,
                status.iterations
            );
            if restart == 1 {
                assert!(status.restarts > 0);
            }
            let mut actual = [T::zero(); 3];
            backend.try_copy_d2h(out.raw(), &mut actual).unwrap();
            for (a, x) in actual.into_iter().zip(exact) {
                assert!((a - x).magnitude().to_f64().unwrap() < 40. * tolerance);
            }
            let capacity = workspace.scalar_metrics();
            if let Some(previous) = retained {
                assert_eq!(capacity, previous);
            }
            retained = Some(capacity);
        }
    }
    let mut session = DeviceGmresSession::try_new_flexible(&pool, 3).unwrap();
    let config = BackendKrylovConfig {
        restart: 3,
        max_iterations: 30,
        tolerance: T::Real::from_f64(tolerance).unwrap(),
        require_native: true,
        exact_residual_check_interval: 0,
    };
    let first = session
        .solve(
            &matrix,
            &preconditioner,
            rhs.raw(),
            DeviceInitialGuess::Zero,
            &config,
            None,
        )
        .unwrap();
    assert!(first.converged);
    assert_eq!(first.method, BackendKrylovMethod::Fgmres);
    let warm = session
        .solve(
            &matrix,
            &preconditioner,
            rhs.raw(),
            DeviceInitialGuess::PreviousSolution,
            &config,
            None,
        )
        .unwrap();
    assert!(warm.converged);
    assert_eq!(warm.iterations, 0);
    let cancel = CancellationToken::new();
    cancel.cancel();
    assert!(
        session
            .solve(
                &matrix,
                &preconditioner,
                rhs.raw(),
                DeviceInitialGuess::Zero,
                &config,
                Some(&cancel)
            )
            .is_err()
    );
    assert!(session.solution().is_none());
    drop(session);
    workspace.clear().unwrap();
    drop(out);
    drop(rhs);
    assert_eq!(pool.bytes_in_use(), 0);
}

#[test]
fn changing_preconditioner_all_scalars_and_restarts() {
    exercise(CpuBackend, |r, _| r as f32, 2e-6);
    exercise(CpuBackend, |r, _| r, 1e-11);
    exercise(
        CpuBackend,
        |r, i| num_complex::Complex32::new(r as f32, i as f32),
        2e-6,
    );
    exercise(CpuBackend, num_complex::Complex64::new, 1e-11);
}

#[cfg(feature = "cuda")]
#[test]
fn native_changing_preconditioner() {
    let backend = match sonium_compute::cuda::CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                "CUDA required: {error}"
            );
            eprintln!("SKIP native flexible GMRES: {error}");
            return;
        }
    };
    exercise(backend, num_complex::Complex64::new, 1e-11);
}

#[test]
fn flexible_quota_failure_releases_saved_vectors() {
    for cap in [64, 128, 192, 256] {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(cap)).unwrap();
        let matrix = DeviceCsr::try_from_host(
            pool.backend(),
            3,
            3,
            &[0, 1, 2, 3],
            &[0, 1, 2],
            &[2.0_f64, 3.0, 5.0],
        )
        .unwrap();
        let mut rhs = pool.try_alloc::<f64>(3).unwrap();
        let mut out = pool.try_alloc::<f64>(3).unwrap();
        pool.backend()
            .try_copy_h2d(&[1., 1., 1.], rhs.raw_mut())
            .unwrap();
        let mut workspace = BackendKrylovWorkspace::new(3);
        let config = BackendKrylovConfig {
            restart: 3,
            max_iterations: 3,
            tolerance: 1e-12,
            require_native: true,
            exact_residual_check_interval: 0,
        };
        let result = try_fgmres_backend_preconditioned(
            pool.backend(),
            &pool,
            &mut workspace,
            &matrix,
            &ChangingScale(AtomicUsize::new(0)),
            rhs.raw(),
            None,
            out.raw_mut(),
            &config,
            None,
        );
        assert!(result.is_err(), "quota {cap} should be insufficient");
        workspace.clear().unwrap();
        drop(out);
        drop(rhs);
        assert_eq!(pool.bytes_in_use(), 0);
    }
}
