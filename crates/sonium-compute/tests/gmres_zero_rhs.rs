//! Homogeneous GMRES systems use absolute residuals for every scalar type.

// Rust guideline compliant 2026-02-21
use num_complex::{Complex32, Complex64};
use num_traits::FromPrimitive;
use sonium_compute::gmres::{
    GmresBackendConfig, GmresWorkspace, IdentityBackendPreconditioner,
    fgmres_backend_preconditioned_pooled_with_workspace,
    gmres_backend_preconditioned_pooled_with_workspace,
};
use sonium_compute::{
    Backend, CpuBackend, DeviceCsr, DeviceScalar, MemoryPool, MemoryPoolConfig, PrebakedSpmv,
};

fn check<B: Backend, T: DeviceScalar>(pool: &MemoryPool<B>, tolerance: f64) {
    let backend = pool.backend();
    let real = |v| T::Real::from_f64(v).unwrap();
    let scalar = |v| T::from_real(real(v));
    let matrix = DeviceCsr::try_from_host(
        backend,
        3,
        3,
        &[0, 1, 2, 3],
        &[0, 1, 2],
        &[scalar(2.0), scalar(3.0), scalar(4.0)],
    )
    .unwrap();
    let a = PrebakedSpmv::try_from_csr(backend, &matrix).unwrap();
    let mut rhs = pool.try_alloc::<T>(3).unwrap();
    let mut initial = pool.try_alloc::<T>(3).unwrap();
    let mut out = pool.try_alloc::<T>(3).unwrap();
    backend
        .try_copy_h2d(&[scalar(0.0); 3], rhs.raw_mut())
        .unwrap();
    backend
        .try_copy_h2d(&[scalar(1.0), scalar(-2.0), scalar(0.5)], initial.raw_mut())
        .unwrap();
    for flexible in [false, true] {
        let mut workspace = GmresWorkspace::new(3);
        for warm in [false, true] {
            let solve = if flexible {
                fgmres_backend_preconditioned_pooled_with_workspace
            } else {
                gmres_backend_preconditioned_pooled_with_workspace
            };
            let status = solve(
                pool,
                &mut workspace,
                &a,
                &IdentityBackendPreconditioner,
                rhs.raw(),
                warm.then(|| initial.raw()),
                out.raw_mut(),
                &GmresBackendConfig {
                    max_iterations: 8,
                    tolerance: real(tolerance),
                },
            )
            .unwrap();
            assert!(status.converged, "{status:?}");
            assert!(status.final_residual <= real(tolerance));
            assert_eq!(status.iterations == 0, !warm);
            let mut actual = [scalar(1.0); 3];
            backend.try_copy_d2h(out.raw(), &mut actual).unwrap();
            // The smallest diagonal is 2, so ||x|| <= ||A*x|| / 2.
            assert!(
                actual
                    .iter()
                    .all(|v| v.norm_sqr() <= real(tolerance * tolerance))
            );
        }
    }
}

#[test]
fn cpu_homogeneous_gmres_and_fgmres_all_scalars() {
    let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(1 << 20)).unwrap();
    check::<_, f32>(&pool, 1e-5);
    check::<_, f64>(&pool, 1e-10);
    check::<_, Complex32>(&pool, 1e-5);
    check::<_, Complex64>(&pool, 1e-10);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_homogeneous_gmres_and_fgmres_all_scalars() {
    let backend = match sonium_compute::CudaBackend::new_default() {
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
    check::<_, f32>(&pool, 1e-5);
    check::<_, f64>(&pool, 1e-10);
    check::<_, Complex32>(&pool, 1e-5);
    check::<_, Complex64>(&pool, 1e-10);
}
