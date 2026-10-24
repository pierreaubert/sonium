//! Native compressed BEM production solves and retained-state checks.

// Rust guideline compliant 2026-02-21
use super::*;
use sonium_compute::MemoryPoolConfig;

fn cuda_pool() -> Option<MemoryPool<CudaBackend>> {
    match CudaBackend::new_default() {
        Ok(backend) => Some(MemoryPool::new(backend, MemoryPoolConfig::default()).unwrap()),
        Err(error) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").is_err(),
                "CUDA required: {error}"
            );
            None
        }
    }
}

fn problem(subdivisions: usize, boundary: BoundaryConditionType, bm: bool) -> BemProblem {
    BemProblem {
        mesh: generate_icosphere_mesh(1.0, subdivisions),
        physics: PhysicsParams::new(343.0 / (2.0 * PI), 343.0, 1.21, false),
        incident_field: IncidentField::plane_wave_z(),
        bc_type: boundary,
        use_burton_miller: bm,
    }
}

fn solver() -> BemSolver {
    BemSolver {
        solver_method: SolverMethod::Gmres,
        assembly_method: AssemblyMethod::Hmatrix,
        tolerance: 1e-9,
        gmres_restart: 60,
        max_iterations: 600,
        hmatrix_aca_tolerance: 1e-6,
        compressed_exact_residual_dof_limit: 2000,
        compressed_exact_residual_tolerance: 1e-7,
        ..BemSolver::default()
    }
}

#[test]
fn hmatrix_cluster_inverse_respects_spatial_numbering_and_rejects_bad_blocks() {
    use crate::core::assembly::hmatrix::DenseBlock;
    let c = Complex64::new;
    let mut matrix = DenseMatrix::zeros(2, 2);
    matrix[[0, 0]] = c(3.0, 0.2);
    matrix[[0, 1]] = c(0.5, -0.1);
    matrix[[1, 0]] = c(-0.3, 0.4);
    matrix[[1, 1]] = c(2.0, -0.2);
    let mut host = HMatrix {
        num_dofs: 4,
        dense_blocks: vec![
            DenseBlock {
                source_cluster: 0,
                field_cluster: 0,
                matrix: matrix.clone(),
            },
            DenseBlock {
                source_cluster: 1,
                field_cluster: 1,
                matrix: matrix.clone(),
            },
        ],
        low_rank_blocks: Vec::new(),
        cluster_dof_indices: vec![vec![2, 0], vec![3, 1]],
        partitions: Vec::new(),
        transpose_dense_order: Vec::new(),
        transpose_low_rank_order: Vec::new(),
        transpose_partitions: Vec::new(),
        memory_bytes: 0,
        compression_ratio: 1.0,
    };
    let inverse = inverse_blocks(&host, CudaHMatrixPreconditioner::ClusterBlockJacobi).unwrap();
    let expected = [c(0.5, -0.2), c(-0.1, 0.3), c(1.0, 0.5), c(0.7, -0.4)];
    let mut rhs = [c(0.0, 0.0); 4];
    for dofs in &host.cluster_dof_indices {
        for i in 0..2 {
            for j in 0..2 {
                rhs[dofs[i]] += matrix[[i, j]] * expected[dofs[j]];
            }
        }
    }
    for (row, expected) in expected.iter().enumerate() {
        let actual: Complex64 = (inverse.row_ptrs[row]..inverse.row_ptrs[row + 1])
            .map(|offset| inverse.values[offset] * rhs[inverse.col_indices[offset]])
            .sum();
        assert!((actual - expected).norm() < 1e-12);
    }
    host.cluster_dof_indices[1][0] = 2;
    assert!(inverse_blocks(&host, CudaHMatrixPreconditioner::ClusterBlockJacobi).is_err());
    host.cluster_dof_indices[1][0] = 3;
    host.dense_blocks[0].matrix = DenseMatrix::zeros(2, 2);
    assert!(inverse_blocks(&host, CudaHMatrixPreconditioner::ClusterBlockJacobi).is_err());
    assert!(inverse_blocks(&host, CudaHMatrixPreconditioner::Jacobi).is_err());
    host.cluster_dof_indices[0][0] = 99;
    assert!(inverse_blocks(&host, CudaHMatrixPreconditioner::ClusterBlockJacobi).is_err());
}

#[test]
fn native_hmatrix_small_matches_independent_dense_surface_and_fields() {
    let Some(pool) = cuda_pool() else { return };
    for boundary in [
        BoundaryConditionType::Rigid,
        BoundaryConditionType::Impedance,
    ] {
        for bm in [false, true] {
            let problem = problem(1, boundary, bm);
            let reference = BemSolver::default().solve(&problem).unwrap();
            for preconditioner in [
                CudaHMatrixPreconditioner::Jacobi,
                CudaHMatrixPreconditioner::ClusterBlockJacobi,
            ] {
                let mut session = solver()
                    .prepare_cuda_hmatrix(&problem, &pool, preconditioner)
                    .unwrap();
                let report = session
                    .solve_incident(&problem.incident_field, DeviceInitialGuess::Zero)
                    .unwrap();
                assert!(report.exact_relative_residual.unwrap() < 1e-7);
                let result = session.download_solution().unwrap();
                let error = result
                    .surface_pressure
                    .iter()
                    .zip(&reference.surface_pressure)
                    .map(|(a, b)| (*a - *b).norm_sqr())
                    .sum::<f64>()
                    .sqrt();
                let norm = reference
                    .surface_pressure
                    .iter()
                    .map(|v| v.norm_sqr())
                    .sum::<f64>()
                    .sqrt();
                assert!(error / norm < 1e-7, "surface error {}", error / norm);
                let points = [[0.0, 0.0, 2.0], [2.0, 0.0, 0.0], [0.5, -2.0, 0.25]];
                session.prepare_field_points(&points).unwrap();
                // Warm once; subsequent evaluation must reuse geometry and all scratch.
                session.evaluate_pressure_device().unwrap();
                let before = pool.backend().begin_runtime_stats();
                let device = session.evaluate_pressure_device().unwrap();
                let stats = pool.backend().runtime_stats_since(before);
                assert_eq!(stats.allocation_operations, 0, "{stats:?}");
                assert_eq!(stats.transfer.h2d_bytes, 0, "{stats:?}");
                assert!(stats.transfer.d2h_bytes <= 16, "{stats:?}");
                let mut field = vec![Complex64::default(); points.len()];
                pool.backend().try_copy_d2h(device, &mut field).unwrap();
                for (point, actual) in points.into_iter().zip(field) {
                    let expected = reference.evaluate_pressure(&point);
                    assert!(
                        (actual - expected).norm() < 1e-7 * expected.norm().max(1.0),
                        "CUDA field {actual:?}, reference {expected:?}"
                    );
                    assert!(
                        (result.evaluate_pressure(&point) - expected).norm()
                            < 1e-7 * expected.norm().max(1.0)
                    );
                }
            }
        }
    }
}

#[test]
fn native_hmatrix_resident_fields_follow_sources_and_recover() {
    let Some(pool) = cuda_pool() else { return };
    let mut problem = problem(1, BoundaryConditionType::Impedance, true);
    let mut session = solver()
        .prepare_cuda_hmatrix(
            &problem,
            &pool,
            CudaHMatrixPreconditioner::ClusterBlockJacobi,
        )
        .unwrap();
    let points = [[0.0, 0.0, 2.0], [2.0, -0.2, 0.7]];
    assert!(session.evaluate_pressure_device().is_err());
    session.prepare_field_points(&points).unwrap();
    assert!(session.evaluate_pressure_device().is_err());
    let c = Complex64::new;
    for incident in [
        IncidentField::MultiplePlaneWaves(vec![
            ([1.0, 0.0, 0.0], c(0.5, 0.3)),
            ([0.0, -1.0, 0.0], c(-0.2, 0.7)),
        ]),
        IncidentField::PointSource {
            position: [3.0, -0.3, 0.5],
            strength: c(0.3, 0.4),
        },
        IncidentField::MultiplePointSources(Vec::new()),
    ] {
        problem.incident_field = incident;
        let reference = BemSolver::default().solve(&problem).unwrap();
        session
            .solve_incident(&problem.incident_field, DeviceInitialGuess::Zero)
            .unwrap();
        session.evaluate_pressure_device().unwrap();
        let before = pool.backend().begin_runtime_stats();
        let device = session.evaluate_pressure_device().unwrap();
        let stats = pool.backend().runtime_stats_since(before);
        assert_eq!(stats.allocation_operations, 0, "{stats:?}");
        assert_eq!(stats.transfer.h2d_bytes, 0, "{stats:?}");
        assert!(stats.transfer.d2h_bytes <= 16, "{stats:?}");
        let mut actual = vec![c(0.0, 0.0); points.len()];
        pool.backend().try_copy_d2h(device, &mut actual).unwrap();
        for (point, actual) in points.iter().zip(actual) {
            let expected = reference.evaluate_pressure(point);
            assert!(
                (actual - expected).norm() < 1e-7 * expected.norm().max(1.0),
                "{actual:?} != {expected:?}"
            );
        }
        eprintln!(
            "resident fields: alloc={} h2d={} d2h={}",
            stats.allocation_operations, stats.transfer.h2d_bytes, stats.transfer.d2h_bytes
        );
    }
    pool.backend()
        .inject_failure(sonium_compute::CudaFailureInjectionPoint::Launch);
    assert!(session.evaluate_pressure_device().is_err());
    pool.backend().recover_after_error().unwrap();
    session.evaluate_pressure_device().unwrap();
    assert!(
        session
            .prepare_field_points(&[[f64::NAN, 0.0, 0.0]])
            .is_err()
    );
    assert!(session.evaluate_pressure_device().is_err());
    assert!(
        session
            .prepare_field_points(&[problem.mesh.nodes[0]])
            .is_err()
    );
    assert!(session.evaluate_pressure_device().is_err());
    session.prepare_field_points(&[]).unwrap();
    assert!(session.evaluate_pressure_device().unwrap().is_empty());
    session.prepare_field_points(&points).unwrap();
    let invalid = IncidentField::PlaneWave {
        direction: [f64::NAN, 0.0, 0.0],
        amplitude: c(1.0, 0.0),
    };
    assert!(
        session
            .solve_incident(&invalid, DeviceInitialGuess::Zero)
            .is_err()
    );
    assert!(session.evaluate_pressure_device().is_err());
    session
        .solve_incident(&IncidentField::plane_wave_z(), DeviceInitialGuess::Zero)
        .unwrap();
    session.evaluate_pressure_device().unwrap();
}

#[test]
fn native_hmatrix_large_retains_compression_and_certifies_repeated_incidents() {
    let Some(pool) = cuda_pool() else { return };
    // 1,280 panels exceed the dense CUDA producer's 1,000-DOF limit.
    let problem = problem(3, BoundaryConditionType::Rigid, true);
    let mut session = solver()
        .prepare_cuda_hmatrix(
            &problem,
            &pool,
            CudaHMatrixPreconditioner::ClusterBlockJacobi,
        )
        .unwrap();
    assert!(session.rhs.len() > 1000);
    assert!(session.low_rank_blocks() > 0);
    assert!(
        session.compression_ratio() > 1.0,
        "ratio={}",
        session.compression_ratio()
    );
    assert!(session.solution_device().is_none());
    for pass in 0..3 {
        let incident =
            IncidentField::plane_wave([0.2 + pass as f64 * 0.1, 0.3, 1.0], 1.0 + pass as f64 * 0.2);
        let before = pool.backend().begin_runtime_stats();
        let report = session
            .solve_incident(
                &incident,
                if pass == 0 {
                    DeviceInitialGuess::Zero
                } else {
                    DeviceInitialGuess::PreviousSolution
                },
            )
            .unwrap();
        let stats = pool.backend().runtime_stats_since(before);
        assert!(report.compressed_relative_residual < 1e-9);
        assert!(report.exact_relative_residual.unwrap() < 1e-7);
        if pass > 0 {
            assert_eq!(stats.allocation_operations, 0, "{stats:?}");
        }
        eprintln!(
            "H-matrix n={} blocks={} ratio={} pass={pass} iterations={} compressed={} exact={} alloc={} h2d={} d2h={}",
            session.rhs.len(),
            session.low_rank_blocks(),
            session.compression_ratio(),
            report.iterations,
            report.compressed_relative_residual,
            report.exact_relative_residual.unwrap(),
            stats.allocation_operations,
            stats.transfer.h2d_bytes,
            stats.transfer.d2h_bytes
        );
    }
    // Invalid sources must unpublish both output and warm-start eligibility.
    let invalid = IncidentField::PlaneWave {
        direction: [f64::NAN, 0.0, 1.0],
        amplitude: Complex64::new(1.0, 0.0),
    };
    assert!(
        session
            .solve_incident(&invalid, DeviceInitialGuess::Zero)
            .is_err()
    );
    assert!(session.solution_device().is_none());
    assert!(session.download_solution().is_err());
    assert!(
        session
            .solve_incident(
                &problem.incident_field,
                DeviceInitialGuess::PreviousSolution
            )
            .is_err()
    );
    session
        .solve_incident(&problem.incident_field, DeviceInitialGuess::Zero)
        .unwrap();
    // Isolate the normal resident solve from optional host exact certification.
    session.exact_tolerance = None;
    let before = pool.backend().begin_runtime_stats();
    let report = session
        .solve_incident(&problem.incident_field, DeviceInitialGuess::Zero)
        .unwrap();
    let stats = pool.backend().runtime_stats_since(before);
    assert!(report.exact_relative_residual.is_none());
    assert_eq!(stats.allocation_operations, 0);
    assert!(
        stats.transfer.d2h_bytes < session.rhs.len() * size_of::<Complex64>(),
        "bulk download: {stats:?}"
    );
    eprintln!(
        "H-matrix resident solve iterations={} alloc={} h2d={} d2h={}",
        report.iterations,
        stats.allocation_operations,
        stats.transfer.h2d_bytes,
        stats.transfer.d2h_bytes
    );
    let points = [[0.0, 0.0, 2.0], [2.0, -0.2, 0.7]];
    session.prepare_field_points(&points).unwrap();
    session.evaluate_pressure_device().unwrap();
    let before = pool.backend().begin_runtime_stats();
    let device = session.evaluate_pressure_device().unwrap();
    let stats = pool.backend().runtime_stats_since(before);
    assert_eq!(stats.allocation_operations, 0, "{stats:?}");
    assert_eq!(stats.transfer.h2d_bytes, 0, "{stats:?}");
    assert!(stats.transfer.d2h_bytes <= 16, "{stats:?}");
    let mut actual = vec![Complex64::default(); points.len()];
    pool.backend().try_copy_d2h(device, &mut actual).unwrap();
    let reference = session.download_solution().unwrap();
    for (point, actual) in points.iter().zip(actual) {
        let expected = reference.evaluate_pressure(point);
        assert!((actual - expected).norm() < 1e-10 * expected.norm().max(1.0));
    }
    eprintln!(
        "large resident fields: alloc={} h2d={} d2h={}",
        stats.allocation_operations, stats.transfer.h2d_bytes, stats.transfer.d2h_bytes
    );
}

#[test]
fn native_hmatrix_failure_recovery_and_zero_sources() {
    let Some(pool) = cuda_pool() else { return };
    let problem = problem(1, BoundaryConditionType::Rigid, true);
    let mut session = solver()
        .prepare_cuda_hmatrix(
            &problem,
            &pool,
            CudaHMatrixPreconditioner::ClusterBlockJacobi,
        )
        .unwrap();
    session
        .solve_incident(&problem.incident_field, DeviceInitialGuess::Zero)
        .unwrap();
    session.exact_tolerance = Some(0.0);
    assert!(matches!(
        session.solve_incident(&problem.incident_field, DeviceInitialGuess::Zero),
        Err(BemError::ResidualCheckFailed { .. })
    ));
    assert!(session.solution_device().is_none());
    assert!(
        session
            .solve_incident(
                &problem.incident_field,
                DeviceInitialGuess::PreviousSolution
            )
            .is_err()
    );
    session.exact_tolerance = None;
    pool.backend()
        .inject_failure(sonium_compute::CudaFailureInjectionPoint::Launch);
    assert!(
        session
            .solve_incident(&problem.incident_field, DeviceInitialGuess::Zero)
            .is_err()
    );
    assert!(session.solution_device().is_none());
    pool.backend().recover_after_error().unwrap();
    session
        .solve_incident(&problem.incident_field, DeviceInitialGuess::Zero)
        .unwrap();
    let zero = IncidentField::MultiplePlaneWaves(Vec::new());
    let report = session
        .solve_incident(&zero, DeviceInitialGuess::PreviousSolution)
        .unwrap();
    assert!(report.compressed_relative_residual < 1e-9);
    let solution = session.download_solution().unwrap();
    let residual = apply_tbem_stabilized_exact(
        &solution.elements,
        &solution.nodes,
        &solution.physics,
        1.0,
        session.beta_scale,
        &solution.surface_pressure,
    )
    .unwrap();
    assert!(residual.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt() < 1e-9);
    session
        .solve_incident(&zero, DeviceInitialGuess::Zero)
        .unwrap();
    assert!(
        session
            .download_solution()
            .unwrap()
            .surface_pressure
            .iter()
            .all(|v| v.norm() < 1e-12)
    );
}
