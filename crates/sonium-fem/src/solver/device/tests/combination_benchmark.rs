//! Controlled timing of numeric updates through complete retained FEM solves.

// Rust guideline compliant 2026-02-21
use super::*;
use std::time::Instant;

#[test]
#[ignore = "native CUDA performance experiment; run on an otherwise idle GPU"]
fn native_fused_combination_solve_benchmark() {
    run_benchmark(false, false);
}

#[test]
#[ignore = "native CUDA performance experiment; run on an otherwise idle GPU"]
fn native_csr_preprocessing_solve_benchmark() {
    run_benchmark(true, false);
}

#[test]
#[ignore = "native CUDA performance experiment; run on an otherwise idle GPU"]
fn native_fused_jacobi_solve_benchmark() {
    run_benchmark(true, true);
}

fn run_benchmark(preprocessing: bool, fused_smoother: bool) {
    for cells in [8, 24] {
        let mut mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, cells, cells);
        mark(&mut mesh);
        for degree in [PolynomialDegree::P1, PolynomialDegree::P2] {
            let space = H1Space::new(degree);
            let make = |k| {
                let matrix =
                    assemble_bilinear_checked(&HelmholtzForm::new(k), &space, &space, &mesh, 4)
                        .unwrap();
                let mut system = FemSystem::new(
                    matrix,
                    vec![Complex64::default(); space.num_global_dofs(&mesh)],
                );
                try_apply_homogeneous_dirichlet_complex(&mut system, &mesh, &space, 1).unwrap();
                system
            };
            let stiffness = make(Complex64::new(0.0, 0.0)).matrix.to_csr();
            let at_one = make(Complex64::new(1.0, 0.0)).matrix.to_csr();
            assert_eq!(stiffness.row_ptrs, at_one.row_ptrs);
            assert_eq!(stiffness.col_indices, at_one.col_indices);
            let mut mass = stiffness.clone();
            for (m, a) in mass.values.iter_mut().zip(&at_one.values) {
                *m -= *a;
            }
            let initial = make(Complex64::new(4.0, 0.2));
            let exact: Vec<_> = (0..stiffness.num_rows)
                .map(|i| {
                    if initial.dirichlet_dofs.contains(&i) {
                        Complex64::default()
                    } else {
                        Complex64::new((i as f64 * 0.2).sin(), (i as f64 * 0.1).cos())
                    }
                })
                .collect();
            let k_rhs = matvec(&stiffness, &exact, FemAction::Forward);
            let m_rhs = matvec(&mass, &exact, FemAction::Forward);
            let auxiliary = make(Complex64::new(0.0, 4.0)).matrix.to_csr();
            let pools: Vec<_> = [false, true]
                .into_iter()
                .map(|fused| {
                    MemoryPool::new(
                        CudaBackend::new_default()
                            .expect("native CUDA benchmark requires GPU")
                            .with_fused_numeric_combinations(
                                fused && !preprocessing && !fused_smoother,
                            )
                            .try_with_csr_preprocessing(fused_smoother || (fused && preprocessing))
                            .unwrap(),
                        MemoryPoolConfig::with_cap(1 << 30),
                    )
                    .unwrap()
                })
                .collect();
            let mut preparations = Vec::new();
            let mut states: Vec<_> = pools
                .iter()
                .enumerate()
                .map(|(mode, pool)| {
                    let backend = pool.backend();
                    let start = Instant::now();
                    let workload = if degree == PolynomialDegree::P1 {
                        FemWorkload::native_cuda(FemScalar::Complex64, FemAction::Forward)
                    } else {
                        FemWorkload::native_cuda_p2(FemScalar::Complex64, FemAction::Forward)
                    };
                    let session = CudaFemSolveSession::try_assemble(
                        pool,
                        workload,
                        FemDeviceMethod::Fgmres,
                        || Ok(make(Complex64::new(4.0, 0.2))),
                        |_| {
                            let mut preconditioner = BackendAmgPreconditioner::<
                                CudaBackend,
                                Complex64,
                            >::try_from_csr_with_cuda_coarse_lu(
                                backend,
                                &auxiliary,
                                AmgConfig {
                                    coarse_size: 8,
                                    smoother: AmgSmoother::Jacobi,
                                    jacobi_weight: 0.6,
                                    ..Default::default()
                                },
                            )?;
                            preconditioner.set_fused_jacobi_enabled(fused_smoother && mode == 1);
                            Ok(preconditioner)
                        },
                    )
                    .unwrap();
                    let upload = |v: &[Complex64]| {
                        let mut buffer = pool.try_alloc::<Complex64>(v.len()).unwrap();
                        backend.try_copy_h2d(v, buffer.raw_mut()).unwrap();
                        buffer
                    };
                    let buffers = (
                        upload(&stiffness.values),
                        upload(&mass.values),
                        upload(&k_rhs),
                        upload(&m_rhs),
                    );
                    backend.try_synchronize().unwrap();
                    preparations.push(start.elapsed().as_secs_f64());
                    (session, buffers)
                })
                .collect();
            let mut records = [Vec::new(), Vec::new()];
            // Two complete sweep warmups, then nine samples; alternate mode order.
            for sample in 0..11 {
                for mode in if sample % 2 == 0 { [0, 1] } else { [1, 0] } {
                    let backend = pools[mode].backend();
                    let (session, (kv, mv, kr, mr)) = &mut states[mode];
                    for real_k in [4.0, 4.2, 3.8] {
                        let k = Complex64::new(real_k, 0.2);
                        let coefficient = -(k * k);
                        backend.try_synchronize().unwrap();
                        let before = backend.begin_runtime_stats();
                        let start = Instant::now();
                        session
                            .try_combine_system_lagged(
                                (Complex64::new(1.0, 0.0), kv.raw()),
                                &[(coefficient, mv.raw())],
                                (Complex64::new(1.0, 0.0), kr.raw()),
                                &[(coefficient, mr.raw())],
                            )
                            .unwrap();
                        let update_seconds = start.elapsed().as_secs_f64();
                        let update_stats = backend.runtime_stats_since(before);
                        let status = session
                            .solve(DeviceInitialGuess::Zero, &controls())
                            .unwrap();
                        backend.try_synchronize().unwrap();
                        let total_seconds = start.elapsed().as_secs_f64();
                        let stats = backend.runtime_stats_since(before);
                        assert!(status.converged && status.iterations > 0);
                        assert_eq!(update_stats.transfer.h2d_bytes, 0);
                        assert_eq!(
                            update_stats.transfer.d2d_bytes,
                            if mode == 1 && !preprocessing && !fused_smoother {
                                0
                            } else {
                                16 * (stiffness.values.len() + exact.len())
                            }
                        );
                        if sample >= 2 {
                            assert_eq!(stats.allocation_operations, 0);
                        }
                        let mut solution = vec![Complex64::default(); exact.len()];
                        backend
                            .try_copy_d2h(session.solution().unwrap(), &mut solution)
                            .unwrap();
                        // Independent physical assembly and residual evaluation stay outside timing.
                        let reference = make(k).matrix.to_csr();
                        let rhs = matvec(&reference, &exact, FemAction::Forward);
                        let action = matvec(&reference, &solution, FemAction::Forward);
                        let residual = action
                            .iter()
                            .zip(&rhs)
                            .map(|(a, b)| (*a - *b).norm_sqr())
                            .sum::<f64>()
                            .sqrt()
                            / rhs.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
                        let error = solution
                            .iter()
                            .zip(&exact)
                            .map(|(a, b)| (*a - *b).norm_sqr())
                            .sum::<f64>()
                            .sqrt()
                            / exact.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
                        assert!(
                            residual < 1.1e-9 && error < 1e-7,
                            "residual={residual}, error={error}"
                        );
                        if sample >= 2 {
                            records[mode].push(serde_json::json!({
                                "sample": sample-2, "k_real": real_k,
                                "update_seconds": update_seconds, "total_seconds": total_seconds,
                                "iterations": status.iterations, "true_residual": residual, "relative_error": error,
                                "update_d2d_bytes": update_stats.transfer.d2d_bytes,
                                "allocations": stats.allocation_operations,
                            }));
                        }
                    }
                }
            }
            for (legacy, fused) in records[0].iter().zip(&records[1]) {
                assert_eq!(legacy["iterations"], fused["iterations"]);
            }
            println!(
                "{} {}",
                if fused_smoother {
                    "FUSED_JACOBI_FEM_BENCH"
                } else if preprocessing {
                    "CSR_PREPROCESS_FEM_BENCH"
                } else {
                    "FUSED_FEM_BENCH"
                },
                serde_json::json!({
                    "cells_per_axis": cells, "degree": format!("{degree:?}"),
                    "dofs": exact.len(), "nnz": stiffness.values.len(),
                    "preparation_seconds": preparations,
                    "legacy": records[0], "fused": records[1],
                    "timed_window": "resident numeric update plus zero-start FGMRES and synchronization; excludes initial setup, host reference and output download"
                })
            );
        }
    }
}
