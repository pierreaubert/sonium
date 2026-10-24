//! Native regression for a Q3 Helmholtz system requiring auxiliary preconditioning.

// Rust guideline compliant 2026-02-21
use super::*;
use crate::assembly::assemble_bilinear_checked;
use crate::mesh::unit_cube_hexahedra;
use sonium_compute::cuda::{CudaBackend, fem_hex::CudaHexPartialC64};
use sonium_compute::{Backend, MemoryPool, MemoryPoolConfig};
use sonium_solvers::iterative::{BackendKrylovConfig, DeviceGmresSession, DeviceInitialGuess};
use sonium_solvers::{AmgConfig, AmgSmoother, BackendAmgPreconditioner};

#[test]
fn native_q3_lor_retains_solve_for_original_indefinite_operator() {
    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                "CUDA required: {e}"
            );
            eprintln!("SKIP CUDA Q3 auxiliary solve: {e}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 29)).unwrap();
    let mesh = unit_cube_hexahedra(4);
    let space = H1Space::new(PolynomialDegree::P3);
    let form = HelmholtzForm::from_real(5.0);
    let original = assemble_bilinear_checked(&form, &space, &space, &mesh, 4)
        .unwrap()
        .to_csr();
    let host = prepare_hex_helmholtz(&mesh, &space, &form, 4).unwrap();
    let operator = CudaHexPartialC64::try_new(pool.clone(), &host, 1).unwrap();
    let auxiliary =
        prepare_hex_lor_shifted_laplacian(&mesh, &space, Complex64::new(25.0, 25.0), &[]).unwrap();
    let preconditioner =
        BackendAmgPreconditioner::<CudaBackend, Complex64>::try_from_csr_with_cuda_coarse_lu(
            pool.backend(),
            &auxiliary,
            AmgConfig {
                coarse_size: 16,
                smoother: AmgSmoother::Jacobi,
                jacobi_weight: 0.6,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(preconditioner.num_levels() > 1);
    let n = original.num_rows;
    let mut rhs = pool.try_alloc::<Complex64>(n).unwrap();
    let mut session = DeviceGmresSession::try_new_flexible(&pool, n).unwrap();
    let config = BackendKrylovConfig {
        max_iterations: 600,
        restart: 60,
        tolerance: 1e-9,
        require_native: true,
        exact_residual_check_interval: 0,
    };
    let apply = |x: &[Complex64]| -> Vec<Complex64> {
        (0..n)
            .map(|row| {
                (original.row_ptrs[row]..original.row_ptrs[row + 1])
                    .map(|j| original.values[j] * x[original.col_indices[j]])
                    .sum()
            })
            .collect()
    };
    for source in 0..2 {
        let exact: Vec<_> = (0..n)
            .map(|i| {
                Complex64::new(
                    (0.13 * i as f64 + source as f64).sin(),
                    (0.07 * i as f64).cos(),
                )
            })
            .collect();
        let b = apply(&exact);
        pool.backend().try_copy_h2d(&b, rhs.raw_mut()).unwrap();
        let before = pool.backend().runtime_stats();
        let status = session
            .solve(
                &operator,
                &preconditioner,
                rhs.raw(),
                if source == 0 {
                    DeviceInitialGuess::Zero
                } else {
                    DeviceInitialGuess::PreviousSolution
                },
                &config,
                None,
            )
            .unwrap();
        assert!(status.converged, "{status:?}");
        if source > 0 {
            assert_eq!(
                pool.backend().runtime_stats().allocation_operations,
                before.allocation_operations
            );
        }
        let mut x = vec![Complex64::default(); n];
        pool.backend()
            .try_copy_d2h(session.solution().unwrap(), &mut x)
            .unwrap();
        let residual = apply(&x)
            .iter()
            .zip(&b)
            .map(|(a, b)| (*a - *b).norm_sqr())
            .sum::<f64>();
        let norm = b.iter().map(|v| v.norm_sqr()).sum::<f64>();
        assert!((residual / norm).sqrt() < 2e-9);
    }
}
