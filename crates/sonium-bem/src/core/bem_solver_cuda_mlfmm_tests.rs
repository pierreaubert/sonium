//! Public MLFMM scattering parity, admission and retained-source lifecycle.

// Rust guideline compliant 2026-02-21
use super::*;
use sonium_compute::{CudaFailureInjectionPoint, MemoryPoolConfig};

fn pool() -> Option<MemoryPool<CudaBackend>> {
    match CudaBackend::new_default() {
        Ok(b) => Some(MemoryPool::new(b, MemoryPoolConfig::default()).unwrap()),
        Err(e) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                "{e}"
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
        assembly_method: AssemblyMethod::Mlfmm,
        tolerance: 1e-9,
        gmres_restart: 10,
        max_iterations: 300,
        compressed_exact_residual_dof_limit: 1000,
        compressed_exact_residual_tolerance: 1e-6,
        ..Default::default()
    }
}
fn check(session: &CudaMlfmmBemSession<'_>, problem: &BemProblem, tolerance: f64) {
    let reference = BemSolver::default().solve(problem).unwrap();
    let result = session.download_solution().unwrap();
    let norm = reference
        .surface_pressure
        .iter()
        .map(|v| v.norm_sqr())
        .sum::<f64>()
        .sqrt();
    let error = result
        .surface_pressure
        .iter()
        .zip(&reference.surface_pressure)
        .map(|(a, b)| (*a - *b).norm_sqr())
        .sum::<f64>()
        .sqrt()
        / norm;
    assert!(error < tolerance, "surface {error}");
    for point in [[0.0, 0.0, 2.0], [2.0, 0.0, 0.0], [0.5, -2.0, 0.25]] {
        let expected = reference.evaluate_pressure(&point);
        let actual = result.evaluate_pressure(&point);
        assert!(
            (actual - expected).norm() < tolerance * expected.norm().max(1.0),
            "field {actual:?} vs {expected:?}"
        );
    }
}

#[test]
fn native_public_mlfmm_small_surface_and_fields() {
    let Some(pool) = pool() else {
        return;
    };
    for boundary in [
        BoundaryConditionType::Rigid,
        BoundaryConditionType::Impedance,
    ] {
        for bm in [false, true] {
            for preconditioner in [
                CudaMlfmmPreconditioner::Jacobi,
                CudaMlfmmPreconditioner::LeafBlockJacobi,
            ] {
                let mut problem = problem(1, boundary, bm);
                let mut session = solver()
                    .prepare_cuda_mlfmm(&problem, &pool, preconditioner)
                    .unwrap();
                for incident in [
                    IncidentField::plane_wave_z(),
                    IncidentField::PointSource {
                        position: [0.0, 0.0, 3.0],
                        strength: Complex64::new(1.0, 0.3),
                    },
                ] {
                    problem.incident_field = incident;
                    let report = session
                        .solve_incident(&problem.incident_field, DeviceInitialGuess::Zero)
                        .unwrap();
                    assert!(report.exact_relative_residual.unwrap() < 1e-7);
                    check(&session, &problem, 1e-7);
                    let before = pool.backend().begin_runtime_stats();
                    let repeat = session
                        .solve_incident(&problem.incident_field, DeviceInitialGuess::Zero)
                        .unwrap();
                    assert!(repeat.mlfmm_relative_residual < 1e-9);
                    assert_eq!(
                        pool.backend()
                            .runtime_stats_since(before)
                            .allocation_operations,
                        0
                    );
                    assert_eq!(
                        session
                            .solve_incident(
                                &problem.incident_field,
                                DeviceInitialGuess::PreviousSolution
                            )
                            .unwrap()
                            .iterations,
                        0
                    );
                }
            }
        }
    }
}

#[test]
fn native_public_mlfmm_far_incident_reuse() {
    let Some(pool) = pool() else {
        return;
    };
    let mut problem = problem(2, BoundaryConditionType::Rigid, true);
    let mut session = solver()
        .prepare_cuda_mlfmm_with_leaf_size(
            &problem,
            &pool,
            CudaMlfmmPreconditioner::LeafBlockJacobi,
            4,
        )
        .unwrap();
    assert!(session.far_translations() > 0);
    for (index, incident) in [
        IncidentField::plane_wave_z(),
        IncidentField::MultiplePointSources(vec![
            ([0.0, 0.0, 3.0], Complex64::new(1.0, 0.0)),
            ([2.0, 0.0, 0.0], Complex64::new(0.2, -0.1)),
        ]),
    ]
    .into_iter()
    .enumerate()
    {
        problem.incident_field = incident;
        let report = session
            .solve_incident(&problem.incident_field, DeviceInitialGuess::Zero)
            .unwrap();
        assert!(report.exact_relative_residual.unwrap() < 1e-6);
        check(&session, &problem, 1e-5);
        // Isolate resident execution after independent exact certification.
        session.exact_tolerance = None;
        let before = pool.backend().begin_runtime_stats();
        let repeated = session
            .solve_incident(&problem.incident_field, DeviceInitialGuess::Zero)
            .unwrap();
        let stats = pool.backend().runtime_stats_since(before);
        assert!(repeated.exact_relative_residual.is_none());
        assert_eq!(stats.allocation_operations, 0, "{stats:?}");
        eprintln!(
            "public MLFMM n={} far={} source_case={index} iterations={} residual={} h2d={} d2h={} allocations={}",
            session.rhs.len(),
            session.far_translations(),
            repeated.iterations,
            repeated.mlfmm_relative_residual,
            stats.transfer.h2d_bytes,
            stats.transfer.d2h_bytes,
            stats.allocation_operations
        );
        session.exact_tolerance = Some(1e-6);
    }
}

#[test]
fn native_public_mlfmm_rejection_and_recovery() {
    let Some(pool) = pool() else {
        return;
    };
    let mut problem = problem(1, BoundaryConditionType::Rigid, true);
    for assembly in [
        AssemblyMethod::Tbem,
        AssemblyMethod::Hmatrix,
        AssemblyMethod::DirectionalMlfmm,
    ] {
        let mut invalid = solver();
        invalid.assembly_method = assembly;
        let before = pool.backend().begin_runtime_stats();
        assert!(
            invalid
                .prepare_cuda_mlfmm(&problem, &pool, CudaMlfmmPreconditioner::Jacobi)
                .is_err()
        );
        assert_eq!(
            pool.backend()
                .runtime_stats_since(before)
                .allocation_operations,
            0
        );
    }
    let mut session = solver()
        .prepare_cuda_mlfmm(&problem, &pool, CudaMlfmmPreconditioner::Jacobi)
        .unwrap();
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
    let invalid = IncidentField::PointSource {
        position: [f64::NAN, 0.0, 3.0],
        strength: Complex64::new(1.0, 0.0),
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
    pool.backend()
        .inject_failure(CudaFailureInjectionPoint::Launch);
    assert!(
        session
            .solve_incident(&problem.incident_field, DeviceInitialGuess::Zero)
            .is_err()
    );
    pool.backend().recover_after_error().unwrap();
    session
        .solve_incident(&problem.incident_field, DeviceInitialGuess::Zero)
        .unwrap();
    session.exact_tolerance = Some(0.0);
    assert!(matches!(
        session.solve_incident(&problem.incident_field, DeviceInitialGuess::Zero),
        Err(BemError::ResidualCheckFailed { .. })
    ));
    assert!(session.solution_device().is_none());
    session.exact_tolerance = Some(1e-6);
    session
        .solve_incident(&problem.incident_field, DeviceInitialGuess::Zero)
        .unwrap();
    problem.incident_field = IncidentField::PlaneWave {
        direction: [0.0, 0.0, 1.0],
        amplitude: Complex64::default(),
    };
    let zero = session
        .solve_incident(&problem.incident_field, DeviceInitialGuess::Zero)
        .unwrap();
    assert_eq!(zero.iterations, 0);
    assert_eq!(zero.exact_relative_residual, Some(0.0));
}
