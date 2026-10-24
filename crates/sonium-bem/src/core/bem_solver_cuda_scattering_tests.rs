//! Native representation dispatch, parity, retention and recovery.

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
fn check(session: &CudaScatteringSession<'_>, problem: &BemProblem, tolerance: f64) {
    check_solution(session.download_solution().unwrap(), problem, tolerance);
}

fn check_solution(result: BemSolution, problem: &BemProblem, tolerance: f64) {
    let reference = BemSolver::default().solve(problem).unwrap();
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

fn configured(method: AssemblyMethod) -> BemSolver {
    let mut config = solver();
    config.assembly_method = method;
    config.solver_method = if matches!(method, AssemblyMethod::Tbem | AssemblyMethod::Auto) {
        SolverMethod::Direct
    } else {
        SolverMethod::Gmres
    };
    config.hmatrix_aca_tolerance = 1e-10;
    config
}

#[test]
fn native_scattering_dispatch_parity_retention_recovery() {
    let Some(pool) = pool() else {
        return;
    };
    for method in [
        AssemblyMethod::Auto,
        AssemblyMethod::Tbem,
        AssemblyMethod::Hmatrix,
        AssemblyMethod::Mlfmm,
    ] {
        for preconditioner in [
            CudaScatteringPreconditioner::Jacobi,
            CudaScatteringPreconditioner::SpatialBlockJacobi,
        ] {
            let mut problem = problem(1, BoundaryConditionType::Rigid, true);
            let mut session = configured(method)
                .prepare_cuda_scattering(
                    &problem,
                    &pool,
                    CudaScatteringOptions {
                        preconditioner,
                        ..Default::default()
                    },
                )
                .unwrap();
            assert_eq!(session.assembly_method(), method.resolve(80));
            assert!(
                session
                    .solve_incident(
                        &problem.incident_field,
                        DeviceInitialGuess::PreviousSolution
                    )
                    .is_err()
            );
            for incident in [
                IncidentField::plane_wave_z(),
                IncidentField::PointSource {
                    position: [0.0, 0.0, 3.0],
                    strength: Complex64::new(1.0, 0.3),
                },
            ] {
                problem.incident_field = incident;
                session
                    .solve_incident(&problem.incident_field, DeviceInitialGuess::Zero)
                    .unwrap();
                check(&session, &problem, 1e-7);
                let before = pool.backend().begin_runtime_stats();
                let repeat = session
                    .solve_incident(
                        &problem.incident_field,
                        DeviceInitialGuess::PreviousSolution,
                    )
                    .unwrap();
                match repeat {
                    CudaScatteringReport::DenseGmres(r) => {
                        assert_eq!(r.iterations, 0);
                    }
                    CudaScatteringReport::Dense(r) => {
                        assert!(r.factorization_reused);
                        assert!(r.relative_residual < 1e-9);
                    }
                    CudaScatteringReport::Hmatrix(r) => {
                        assert_eq!(r.iterations, 0);
                        assert!(r.exact_relative_residual.unwrap() < 1e-7);
                    }
                    CudaScatteringReport::Mlfmm(r) => {
                        assert_eq!(r.iterations, 0);
                        assert!(r.exact_relative_residual.unwrap() < 1e-7);
                    }
                }
                assert_eq!(
                    pool.backend()
                        .runtime_stats_since(before)
                        .allocation_operations,
                    0
                );
            }
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
            assert!(session.solution_device().is_none());
            pool.backend().recover_after_error().unwrap();
            session
                .solve_incident(&problem.incident_field, DeviceInitialGuess::Zero)
                .unwrap();
            check(&session, &problem, 1e-7);
        }
    }
}

#[test]
fn native_scattering_dispatch_far_mlfmm() {
    let Some(pool) = pool() else {
        return;
    };
    let problem = problem(2, BoundaryConditionType::Rigid, true);
    let mut session = configured(AssemblyMethod::Mlfmm)
        .prepare_cuda_scattering(
            &problem,
            &pool,
            CudaScatteringOptions {
                mlfmm_target_elements_per_leaf: 4,
                ..Default::default()
            },
        )
        .unwrap();
    let Session::Mlfmm(inner) = &session.inner else {
        panic!("wrong representation");
    };
    assert!(inner.far_translations() > 0);
    session
        .solve_incident(&problem.incident_field, DeviceInitialGuess::Zero)
        .unwrap();
    check(&session, &problem, 1e-5);
    let before = pool.backend().begin_runtime_stats();
    session
        .solve_incident(&problem.incident_field, DeviceInitialGuess::Zero)
        .unwrap();
    assert_eq!(
        pool.backend()
            .runtime_stats_since(before)
            .allocation_operations,
        0
    );
}

#[test]
fn native_scattering_dispatch_rejects_before_allocation() {
    let Some(pool) = pool() else {
        return;
    };
    let problem = problem(1, BoundaryConditionType::Rigid, true);
    for method in [
        AssemblyMethod::Slfmm,
        AssemblyMethod::DirectionalMlfmm,
        AssemblyMethod::Tbem,
        AssemblyMethod::Hmatrix,
        AssemblyMethod::Mlfmm,
    ] {
        let mut config = configured(method);
        if method == AssemblyMethod::Tbem {
            config.solver_method = SolverMethod::Gmres;
        }
        if matches!(method, AssemblyMethod::Hmatrix | AssemblyMethod::Mlfmm) {
            config.solver_method = SolverMethod::Direct;
        }
        let before = pool.backend().begin_runtime_stats();
        assert!(
            config
                .prepare_cuda_scattering(&problem, &pool, Default::default())
                .is_err()
        );
        assert_eq!(
            pool.backend()
                .runtime_stats_since(before)
                .allocation_operations,
            0
        );
    }
}

#[test]
fn native_scattering_dispatch_auto_compression_threshold() {
    let Some(pool) = pool() else {
        return;
    };
    let problem = problem(3, BoundaryConditionType::Rigid, true);
    assert_eq!(problem.mesh.elements.len(), AUTO_TBEM_DOF_LIMIT);
    let mut config = configured(AssemblyMethod::Auto);
    config.solver_method = SolverMethod::Gmres;
    config.gmres_restart = 60;
    config.max_iterations = 600;
    config.compressed_exact_residual_dof_limit = 2000;
    let mut session = config
        .prepare_cuda_scattering(&problem, &pool, Default::default())
        .unwrap();
    assert_eq!(session.assembly_method(), AssemblyMethod::Hmatrix);
    let Session::Hmatrix(inner) = &session.inner else {
        panic!("wrong representation");
    };
    assert!(inner.low_rank_blocks() > 0);
    let report = session
        .solve_incident(&problem.incident_field, DeviceInitialGuess::Zero)
        .unwrap();
    let CudaScatteringReport::Hmatrix(report) = report else {
        panic!("wrong report");
    };
    assert!(report.exact_relative_residual.unwrap() < 1e-6);
    check(&session, &problem, 1e-5);
    let before = pool.backend().begin_runtime_stats();
    session
        .solve_incident(&problem.incident_field, DeviceInitialGuess::Zero)
        .unwrap();
    assert_eq!(
        pool.backend()
            .runtime_stats_since(before)
            .allocation_operations,
        0
    );

    // Auto selects compressed CUDA below the CPU crossover as well. Direct
    // remains unsupported there and must reject without changing the linear solver.
    let mut gap = problem.clone();
    gap.mesh.elements.truncate(1000);
    config.solver_method = SolverMethod::Direct;
    let before = pool.backend().begin_runtime_stats();
    assert!(
        config
            .prepare_cuda_scattering(&gap, &pool, Default::default())
            .is_err()
    );
    assert_eq!(
        pool.backend()
            .runtime_stats_since(before)
            .allocation_operations,
        0
    );
}

#[test]
fn native_public_tbem_gmres_dispatch() {
    let Some(pool) = pool() else {
        return;
    };
    for method in [AssemblyMethod::Tbem, AssemblyMethod::Auto] {
        let mut config = configured(method);
        config.solver_method = SolverMethod::Gmres;
        let mut problem = problem(1, BoundaryConditionType::Rigid, true);
        let mut session = config
            .prepare_cuda_scattering(
                &problem,
                &pool,
                CudaScatteringOptions {
                    preconditioner: CudaScatteringPreconditioner::Jacobi,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(session.assembly_method(), AssemblyMethod::Tbem);
        for incident in [
            IncidentField::plane_wave_z(),
            IncidentField::PointSource {
                position: [0.0, 0.0, 3.0],
                strength: Complex64::new(1.0, 0.3),
            },
        ] {
            problem.incident_field = incident;
            let CudaScatteringReport::DenseGmres(report) = session
                .solve_incident(&problem.incident_field, DeviceInitialGuess::Zero)
                .unwrap()
            else {
                panic!("wrong representation");
            };
            assert!(report.converged);
            check(&session, &problem, 1e-7);
            let before = pool.backend().begin_runtime_stats();
            let CudaScatteringReport::DenseGmres(report) = session
                .solve_incident(
                    &problem.incident_field,
                    DeviceInitialGuess::PreviousSolution,
                )
                .unwrap()
            else {
                panic!("wrong representation");
            };
            assert_eq!(report.iterations, 0);
            assert_eq!(
                pool.backend()
                    .runtime_stats_since(before)
                    .allocation_operations,
                0
            );
        }
    }
}

#[test]
fn native_public_tbem_gmres_numeric_and_recovery() {
    let Some(pool) = pool() else {
        return;
    };
    let mut config = configured(AssemblyMethod::Tbem);
    config.solver_method = SolverMethod::Gmres;
    for boundary in [
        BoundaryConditionType::Rigid,
        BoundaryConditionType::Impedance,
    ] {
        for bm in [false, true] {
            let mut problem = problem(1, boundary, bm);
            let mut session = config.prepare_cuda_gmres_jacobi(&problem, &pool).unwrap();
            assert!(
                session
                    .solve_device(&problem, DeviceInitialGuess::PreviousSolution)
                    .is_err()
            );
            for pass in 0..3 {
                problem.physics = PhysicsParams::new(
                    (1.0 + pass as f64 * 0.01) * 343.0 / (2.0 * PI),
                    343.0,
                    1.21,
                    false,
                );
                problem.incident_field = if pass == 1 {
                    IncidentField::PointSource {
                        position: [0.0, 0.0, 3.0],
                        strength: Complex64::new(1.0, 0.3),
                    }
                } else {
                    IncidentField::plane_wave_z()
                };
                let before = pool.backend().begin_runtime_stats();
                let report = session
                    .solve_device(
                        &problem,
                        if pass == 0 {
                            DeviceInitialGuess::Zero
                        } else {
                            DeviceInitialGuess::PreviousSolution
                        },
                    )
                    .unwrap();
                assert!(report.converged);
                if pass > 0 {
                    assert_eq!(
                        pool.backend()
                            .runtime_stats_since(before)
                            .allocation_operations,
                        0
                    );
                }
                check_solution(session.download_solution().unwrap(), &problem, 1e-7);
            }
            let mut invalid = problem.clone();
            invalid.incident_field = IncidentField::PointSource {
                position: [f64::NAN, 0.0, 3.0],
                strength: Complex64::new(1.0, 0.0),
            };
            assert!(
                session
                    .solve_device(&invalid, DeviceInitialGuess::Zero)
                    .is_err()
            );
            assert!(session.solution_device().is_none());
            assert!(session.download_solution().is_err());
            assert!(
                session
                    .solve_device(&problem, DeviceInitialGuess::PreviousSolution)
                    .is_err()
            );
            pool.backend()
                .inject_failure(CudaFailureInjectionPoint::Launch);
            assert!(
                session
                    .solve_device(&problem, DeviceInitialGuess::Zero)
                    .is_err()
            );
            assert!(session.solution_device().is_none());
            pool.backend().recover_after_error().unwrap();
            session
                .solve_device(&problem, DeviceInitialGuess::Zero)
                .unwrap();
            check_solution(session.download_solution().unwrap(), &problem, 1e-7);
        }
    }
}

#[test]
fn cuda_auto_respects_dense_ceiling_and_explicit_methods() {
    for n in [0, 80, 999, 1000, 1279, 1280, 5000] {
        assert_eq!(
            AssemblyMethod::Auto.resolve_cuda(n),
            if n < 1000 {
                AssemblyMethod::Tbem
            } else {
                AssemblyMethod::Hmatrix
            }
        );
        // CPU crossover remains a separate, empirically selected policy.
        assert_eq!(
            AssemblyMethod::Auto.resolve(n),
            if n < AUTO_TBEM_DOF_LIMIT {
                AssemblyMethod::Tbem
            } else {
                AssemblyMethod::Hmatrix
            }
        );
        for method in [
            AssemblyMethod::Tbem,
            AssemblyMethod::Hmatrix,
            AssemblyMethod::Slfmm,
            AssemblyMethod::Mlfmm,
            AssemblyMethod::DirectionalMlfmm,
        ] {
            assert_eq!(method.resolve_cuda(n), method);
        }
    }
}

#[test]
fn native_scattering_dispatch_cuda_auto_gap() {
    let Some(pool) = pool() else { return };
    let mut problem = problem(1, BoundaryConditionType::Rigid, true);
    // Closed Fibonacci sphere: 502 vertices give exactly 1000 triangular faces.
    problem.mesh = generate_sphere_mesh(1.0, 11, 50);
    assert_eq!(problem.mesh.elements.len(), 1000);
    let mut config = configured(AssemblyMethod::Auto);
    let before = pool.backend().begin_runtime_stats();
    assert!(
        config
            .prepare_cuda_scattering(&problem, &pool, Default::default())
            .is_err()
    );
    assert_eq!(
        pool.backend()
            .runtime_stats_since(before)
            .allocation_operations,
        0
    );
    assert_eq!(config.solver_method, SolverMethod::Direct);
    config.solver_method = SolverMethod::Gmres;
    config.gmres_restart = 60;
    config.max_iterations = 600;
    config.compressed_exact_residual_dof_limit = 2000;
    let mut session = config
        .prepare_cuda_scattering(&problem, &pool, Default::default())
        .unwrap();
    assert_eq!(session.assembly_method(), AssemblyMethod::Hmatrix);
    let Session::Hmatrix(inner) = &session.inner else {
        panic!("wrong representation");
    };
    assert!(inner.low_rank_blocks() > 0);
    for index in 0..2 {
        let before = pool.backend().begin_runtime_stats();
        let CudaScatteringReport::Hmatrix(report) = session
            .solve_incident(&problem.incident_field, DeviceInitialGuess::Zero)
            .unwrap()
        else {
            panic!("wrong report");
        };
        assert!(report.exact_relative_residual.unwrap() < 1e-6);
        if index > 0 {
            assert_eq!(
                pool.backend()
                    .runtime_stats_since(before)
                    .allocation_operations,
                0
            );
        }
    }
    check(&session, &problem, 1e-5);
}

#[test]
fn native_scattering_preflight_rejects_poisoned_context() {
    for method in [
        AssemblyMethod::Tbem,
        AssemblyMethod::Hmatrix,
        AssemblyMethod::Mlfmm,
    ] {
        let Some(pool) = pool() else { return };
        let backend = pool.backend();
        let problem = problem(1, BoundaryConditionType::Rigid, true);
        let mut config = configured(method);
        config.solver_method = SolverMethod::Gmres;
        // A terminal error must be propagated before any host hierarchy/device setup.
        backend.inject_failure(CudaFailureInjectionPoint::Synchronization);
        let original = backend.try_synchronize().unwrap_err().to_string();
        let before = backend.begin_runtime_stats();
        let error = config
            .prepare_cuda_scattering(
                &problem,
                &pool,
                CudaScatteringOptions {
                    preconditioner: CudaScatteringPreconditioner::Jacobi,
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains(&original), "{error}");
        let stats = backend.runtime_stats_since(before);
        assert_eq!(stats.allocation_operations, 0);
        assert_eq!(stats.kernel_launches + stats.library_operations, 0);
        assert_eq!(
            stats.transfer.h2d_bytes + stats.transfer.d2h_bytes + stats.transfer.d2d_bytes,
            0
        );
        assert_eq!(
            backend.pending_error().unwrap().unwrap().to_string(),
            original
        );
        assert!(backend.recover_after_error().is_err());
    }
}
