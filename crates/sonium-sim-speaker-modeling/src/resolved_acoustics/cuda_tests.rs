//! Native ILUT/FGMRES qualification on the resolved axisymmetric PML producer.

// Rust guideline compliant 2026-02-21
use super::*;
use sonium_compute::{CudaBackend, MemoryPool, MemoryPoolConfig};
use sonium_solvers::CudaLinearSolveSession;
use sonium_solvers::iterative::{BackendKrylovConfig, DeviceInitialGuess};
use sonium_solvers::preconditioners::CudaIlutPreconditioner;

#[test]
fn native_resolved_pml_retained_fgmres_ilut() {
    let backend = match CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            assert!(
                std::env::var_os("SONIUM_REQUIRE_CUDA").is_none(),
                "CUDA required: {error}"
            );
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 29)).unwrap();
    for config in [
        ResolvedAcousticConfig {
            cells_per_physical_interval: 1,
            pml_cells: 1,
            pml_sigma_max: 2.0,
            direct_solve_max_nodes: usize::MAX,
            ..Default::default()
        },
        ResolvedAcousticConfig {
            direct_solve_max_nodes: usize::MAX,
            ..Default::default()
        },
    ] {
        // CPU reference uses direct LU; native preparation explicitly preserves
        // the model's ILUT drop/fill/pivot configuration and FGMRES controls.
        let model = ResolvedAcousticModel::build(config).unwrap();
        let ilut_config = model.config.ilut_config();
        assert!(ilut_config.pivoting);
        let controls = BackendKrylovConfig {
            max_iterations: model.config.fgmres_max_iterations,
            restart: model.config.fgmres_restart,
            tolerance: model.config.fgmres_tolerance,
            exact_residual_check_interval: 1,
            require_native: true,
        };
        let first = model
            .assemble_frequency_system(400.0, Complex64::new(0.01, 0.0))
            .unwrap();
        let mut session = CudaLinearSolveSession::try_new_flexible(
            &pool,
            &first.matrix,
            &controls,
            |backend, matrix| {
                CudaIlutPreconditioner::try_from_csr_with_config(backend, matrix, &ilut_config)
            },
        )
        .unwrap();
        assert!(session.selected_strategy().is_none());
        let mut warm = false;
        for frequency in [400.0, 450.0] {
            for (drive_index, drive) in [Complex64::new(0.01, 0.0), Complex64::new(0.008, 0.002)]
                .into_iter()
                .enumerate()
            {
                let system = model.assemble_frequency_system(frequency, drive).unwrap();
                if frequency != 400.0 && drive_index == 0 {
                    session
                        .try_update_matrix_with_preconditioner(&system.matrix, |backend, matrix| {
                            CudaIlutPreconditioner::try_from_csr_with_config(
                                backend,
                                matrix,
                                &ilut_config,
                            )
                        })
                        .unwrap();
                }
                let expected = model
                    .solve_assembled_with_external_guess(&system, None)
                    .unwrap();
                let before = pool.backend().begin_runtime_stats();
                let solved = session
                    .solve_host(
                        &system.rhs,
                        if warm {
                            DeviceInitialGuess::PreviousSolution
                        } else {
                            DeviceInitialGuess::Zero
                        },
                    )
                    .unwrap();
                if warm {
                    assert_eq!(
                        pool.backend()
                            .runtime_stats_since(before)
                            .allocation_operations,
                        0
                    );
                }
                assert!(solved.converged);
                assert!(session.solution_device().is_some());
                let residual = exact_relative_residual(&system.matrix, &solved.x, &system.rhs);
                assert!(residual < 1.1 * controls.tolerance, "residual {residual}");
                let error = solved
                    .x
                    .iter()
                    .zip(&expected.pressure_rms_pa)
                    .map(|(a, b)| (*a - *b).norm_sqr())
                    .sum::<f64>()
                    .sqrt();
                let norm = expected
                    .pressure_rms_pa
                    .iter()
                    .map(|v| v.norm_sqr())
                    .sum::<f64>()
                    .sqrt();
                assert!(error / norm < 1e-6, "relative field error {}", error / norm);
                eprintln!(
                    "resolved native ILUT: nodes={} frequency={frequency} drive={drive_index} iterations={} residual={residual}",
                    model.mesh.num_nodes(),
                    solved.iterations
                );
                warm = true;
            }
        }
        let system = model
            .assemble_frequency_system(450.0, Complex64::new(0.01, 0.0))
            .unwrap();
        pool.backend()
            .inject_failure(sonium_compute::CudaFailureInjectionPoint::TransferD2D);
        assert!(
            session
                .solve_host(&system.rhs, DeviceInitialGuess::PreviousSolution)
                .is_err()
        );
        assert!(session.solution_device().is_none());
        pool.backend().recover_after_error().unwrap();
        assert!(
            session
                .solve_host(&system.rhs, DeviceInitialGuess::Zero)
                .unwrap()
                .converged
        );
    }
}
