//! Physical boundary-treatment parity, adjoint gradients and MMA reuse.

// Rust guideline compliant 2026-02-21
use super::*;
use sonium_compute::{CudaFailureInjectionPoint, MemoryPoolConfig};
use sonium_fem::assembly::{assemble_boundary_mass, assemble_mass, assemble_stiffness};
use sonium_fem::backend::FemScalar;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::solver::SolverType;

fn problem(solver_type: SolverType) -> RoomTreatmentProblem {
    let mut mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 4, 4);
    for face in &mut mesh.boundaries {
        face.marker = if face.nodes.iter().all(|&i| mesh.nodes[i].x == 0.0) {
            1
        } else {
            2
        };
    }
    let degree = PolynomialDegree::P1;
    let stiffness = assemble_stiffness(&mesh, degree);
    let mass = assemble_mass(&mesh, degree);
    let boundaries = [
        (1, assemble_boundary_mass(&mesh, degree, 1)),
        (2, assemble_boundary_mass(&mesh, degree, 2)),
    ];
    let assembler = HelmholtzAssembler::from_matrices(&stiffness, &mass, &boundaries);
    let mut rhs = vec![Complex64::default(); mesh.num_nodes()];
    rhs[12] = Complex64::new(1.0, 0.2);
    let mut solver = SolverConfig {
        solver_type,
        schwarz_subdomains: 3,
        schwarz_overlap: 1,
        ..Default::default()
    };
    solver.gmres.tolerance = 1e-11;
    solver.gmres.restart = 30;
    RoomTreatmentProblem::from_helmholtz(
        &assembler,
        Complex64::new(2.0, 0.2),
        &HashMap::from([
            (1, Complex64::new(0.0, -0.4)),
            (2, Complex64::new(0.0, -0.3)),
        ]),
        &[
            TreatmentBoundary {
                marker: 1,
                coefficient_direction: Complex64::new(0.0, -0.8),
                area_m2: None,
                cost_per_m2: None,
            },
            TreatmentBoundary {
                marker: 2,
                coefficient_direction: Complex64::new(0.0, -0.5),
                area_m2: None,
                cost_per_m2: None,
            },
        ],
        rhs,
        vec![
            ListenerFunctional {
                entries: vec![
                    (0, Complex64::new(0.7, 0.2)),
                    (1, Complex64::new(0.3, -0.1)),
                    (0, Complex64::new(0.1, -0.2)),
                ],
            },
            ListenerFunctional::at_node(9),
            ListenerFunctional::at_node(24),
        ],
        solver,
    )
    .unwrap()
}

fn field_addresses(state: &CudaTreatmentState<'_>) -> [usize; 2] {
    [
        state.forward.solver.session().solution().unwrap() as *const _ as usize,
        state.adjoint.solver.session().solution().unwrap() as *const _ as usize,
    ]
}

#[test]
fn native_cuda_room_treatment_mma_and_adjoint_reuse() {
    let backend = match CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                "CUDA required: {error}"
            );
            eprintln!("SKIP room treatment CUDA: {error}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 29)).unwrap();
    let reference = problem(SolverType::Direct);
    let workload = FemWorkload::native_cuda(FemScalar::Complex64, FemAction::Forward);
    for solver in [SolverType::GmresIlu, SolverType::GmresSchwarz] {
        let problem = problem(solver);
        let mut session =
            CudaRoomTreatmentSession::try_prepare(&problem, &pool, workload, &[0.3, 0.4]).unwrap();
        for (pass, design) in [[0.3, 0.4], [0.35, 0.2], [0.1, 0.7]]
            .into_iter()
            .enumerate()
        {
            let before = pool.backend().begin_runtime_stats();
            session.state.forward.update(&design).unwrap();
            let update = pool.backend().runtime_stats_since(before);
            assert_eq!(update.transfer.h2d_bytes, 0, "{update:?}");
            assert_eq!(update.transfer.d2h_bytes, 8, "{update:?}");
            if pass > 0 {
                assert_eq!(update.allocation_operations, 0);
            }
            let before = pool.backend().begin_runtime_stats();
            session.state.adjoint.update(&design).unwrap();
            let update = pool.backend().runtime_stats_since(before);
            assert_eq!(update.transfer.h2d_bytes, 0, "{update:?}");
            assert_eq!(update.transfer.d2h_bytes, 8, "{update:?}");
            if pass > 0 {
                assert_eq!(update.allocation_operations, 0);
            }
            let expected = reference.objective_gradient(&design).unwrap();
            let before = pool.backend().begin_runtime_stats();
            let actual = session.objective_gradient(&design).unwrap();
            let stats = pool.backend().runtime_stats_since(before);
            if pass > 0 {
                assert_eq!(stats.allocation_operations, 0, "{stats:?}");
            }
            // Only listener vectors, treatment scalars and compact controls transfer.
            assert!(stats.transfer.h2d_bytes > 0 && stats.transfer.d2h_bytes > 0);
            let before = pool.backend().begin_runtime_stats();
            let objective = session.state.prepare_adjoint_rhs().unwrap();
            let listener_stats = pool.backend().runtime_stats_since(before);
            let listener_bytes = problem.listeners.len() * std::mem::size_of::<Complex64>();
            assert_eq!(
                listener_stats.transfer.h2d_bytes, listener_bytes,
                "{listener_stats:?}"
            );
            assert_eq!(
                listener_stats.transfer.d2h_bytes, listener_bytes,
                "{listener_stats:?}"
            );
            assert_eq!(listener_stats.allocation_operations, 0);
            assert!((objective - actual.0).abs() < 1e-12);
            let before = pool.backend().begin_runtime_stats();
            let gradient = session.state.treatment_gradient().unwrap();
            let gradient_stats = pool.backend().runtime_stats_since(before);
            assert_eq!(gradient_stats.transfer.h2d_bytes, 0, "{gradient_stats:?}");
            assert!(
                gradient_stats.transfer.d2h_bytes
                    <= gradient.len() * std::mem::size_of::<Complex64>(),
                "{gradient_stats:?}"
            );
            assert_eq!(gradient_stats.allocation_operations, 0);
            for (a, b) in gradient.iter().zip(&actual.1) {
                assert!((a - b).abs() < 1e-12);
            }
            assert!((actual.0 - expected.0).abs() < 1e-8);
            for (a, e) in actual.1.iter().zip(&expected.1) {
                assert!((a - e).abs() < 1e-7 * (1.0 + e.abs()));
            }
            for parameter in 0..2 {
                let mut plus = design;
                let mut minus = design;
                plus[parameter] += 1e-5;
                minus[parameter] -= 1e-5;
                let derivative = (reference.objective_gradient(&plus).unwrap().0
                    - reference.objective_gradient(&minus).unwrap().0)
                    / 2e-5;
                assert!((actual.1[parameter] - derivative).abs() < 1e-6 * (1.0 + derivative.abs()));
            }
        }
        pool.backend()
            .inject_failure(CudaFailureInjectionPoint::TransferD2D);
        assert!(session.objective_gradient(&[0.2, 0.4]).is_err());
        pool.backend().recover_after_error().unwrap();
        let repaired = session.objective_gradient(&[0.2, 0.4]).unwrap();
        let expected = reference.objective_gradient(&[0.2, 0.4]).unwrap();
        assert!((repaired.0 - expected.0).abs() < 1e-8);
        let fields = field_addresses(&session.state);
        pool.backend()
            .inject_failure(CudaFailureInjectionPoint::TransferH2D);
        assert!(session.try_refresh_preconditioners(&[0.6, 0.1]).is_err());
        pool.backend().recover_after_error().unwrap();
        assert_eq!(field_addresses(&session.state), fields);
        session.try_refresh_preconditioners(&[0.6, 0.1]).unwrap();
        assert!(session.state.forward.solver.session().solution().is_none());
        assert!(session.state.adjoint.solver.session().solution().is_none());
        let before = pool.backend().begin_runtime_stats();
        let refreshed = session.objective_gradient(&[0.6, 0.1]).unwrap();
        assert_eq!(
            pool.backend()
                .runtime_stats_since(before)
                .allocation_operations,
            0
        );
        assert_eq!(field_addresses(&session.state), fields);
        let expected = reference.objective_gradient(&[0.6, 0.1]).unwrap();
        assert!((refreshed.0 - expected.0).abs() < 1e-8);
        for (a, b) in refreshed.1.iter().zip(&expected.1) {
            assert!((a - b).abs() < 1e-7 * (1.0 + b.abs()));
        }
        let config = RoomTreatmentConfig {
            max_iterations: 4,
            tolerance: 1e-8,
            ..Default::default()
        };
        let expected = reference.optimize(&config).unwrap();
        let actual = session.optimize(&config).unwrap();
        assert_eq!(actual.iterations, expected.iterations);
        for (a, e) in actual
            .objective_history
            .iter()
            .zip(&expected.objective_history)
        {
            assert!((a - e).abs() < 1e-7);
        }
        for (a, e) in actual.treatments.iter().zip(&expected.treatments) {
            assert!((a - e).abs() < 1e-6);
        }
    }
    let valid = problem(SolverType::GmresIlu);
    let wrong_physics = FemWorkload {
        physics: FemPhysics::Stokes,
        ..workload
    };
    let before = pool.backend().begin_runtime_stats();
    assert!(
        CudaRoomTreatmentSession::try_prepare(&valid, &pool, wrong_physics, &[0.3, 0.4]).is_err()
    );
    assert_eq!(
        pool.backend()
            .runtime_stats_since(before)
            .allocation_operations,
        0
    );
    let unsupported = problem(SolverType::Direct);
    let before = pool.backend().begin_runtime_stats();
    assert!(
        CudaRoomTreatmentSession::try_prepare(&unsupported, &pool, workload, &[0.3, 0.4]).is_err()
    );
    assert_eq!(
        pool.backend()
            .runtime_stats_since(before)
            .allocation_operations,
        0
    );
}

fn band_problem(solver_type: SolverType) -> MultiFrequencyRoomTreatmentProblem {
    band_problem_on_grid(solver_type, 4)
}

fn band_problem_on_grid(
    solver_type: SolverType,
    subdivisions: usize,
) -> MultiFrequencyRoomTreatmentProblem {
    let mut mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, subdivisions, subdivisions);
    for face in &mut mesh.boundaries {
        face.marker = if face.nodes.iter().all(|&i| mesh.nodes[i].x == 0.0) {
            1
        } else {
            2
        };
    }
    let degree = PolynomialDegree::P1;
    let assembler = HelmholtzAssembler::from_matrices(
        &assemble_stiffness(&mesh, degree),
        &assemble_mass(&mesh, degree),
        &[
            (1, assemble_boundary_mass(&mesh, degree, 1)),
            (2, assemble_boundary_mass(&mesh, degree, 2)),
        ],
    );
    let local = problem(solver_type);
    let setups = [(250.0, 2.5, 0.5), (63.0, 1.5, 1.0), (125.0, 2.0, 2.0)]
        .into_iter()
        .map(|(hz, k, weight)| {
            let mut rhs = vec![Complex64::default(); mesh.num_nodes()];
            rhs[mesh.num_nodes() / 2] = Complex64::new(1.0, 0.2) * Complex64::new(k, 0.2);
            TreatmentFrequencySetup {
                frequency_hz: hz,
                wavenumber: Complex64::new(k, 0.2),
                weight,
                base_boundary_coefficients: HashMap::from([
                    (1, Complex64::new(0.0, -0.4)),
                    (2, Complex64::new(0.0, -0.3)),
                ]),
                rhs,
            }
        })
        .collect::<Vec<_>>();
    MultiFrequencyRoomTreatmentProblem::from_helmholtz(
        &assembler,
        &setups,
        &[
            TreatmentBoundary {
                marker: 1,
                coefficient_direction: Complex64::new(0.0, -0.8),
                area_m2: Some(2.0),
                cost_per_m2: Some(3.0),
            },
            TreatmentBoundary {
                marker: 2,
                coefficient_direction: Complex64::new(0.0, -0.5),
                area_m2: Some(1.5),
                cost_per_m2: Some(7.0),
            },
        ],
        local.listeners,
        local.solver,
    )
    .unwrap()
}

#[test]
fn native_cuda_band_treatment_production_and_mma() {
    let backend = match CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                "CUDA required: {error}"
            );
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 29)).unwrap();
    let reference = band_problem(SolverType::Direct);
    assert_eq!(reference.frequencies_hz(), vec![63.0, 125.0, 250.0]);
    let objective = TreatmentObjectiveConfig {
        target_levels_db: vec![0.0, -3.0, -6.0],
        reference_pressure: 1.0,
        min_level_weight: 0.3,
        min_level_db: 10.0,
        cost_weight: 0.2,
        ..Default::default()
    };
    let workload = FemWorkload::native_cuda(FemScalar::Complex64, FemAction::Forward);
    for solver in [SolverType::GmresIlu, SolverType::GmresSchwarz] {
        let problem = band_problem(solver);
        let mut session = CudaMultiFrequencyRoomTreatmentSession::try_prepare(
            &problem,
            &pool,
            workload,
            &[0.3, 0.4],
        )
        .unwrap();
        for kind in [
            TreatmentObjectiveKind::Production,
            TreatmentObjectiveKind::SeatVariance,
        ] {
            let objective = TreatmentObjectiveConfig {
                kind,
                ..objective.clone()
            };
            for (pass, design) in [[0.3, 0.4], [0.1, 0.7], [0.35, 0.2]]
                .into_iter()
                .enumerate()
            {
                let expected = reference.evaluate(&design, &objective, Some(2)).unwrap();
                let before = pool.backend().begin_runtime_stats();
                let actual = session.evaluate(&design, &objective).unwrap();
                if pass > 0 {
                    assert_eq!(
                        pool.backend()
                            .runtime_stats_since(before)
                            .allocation_operations,
                        0
                    );
                }
                for (a, b) in [
                    (actual.objective, expected.objective),
                    (actual.breakdown.target, expected.breakdown.target),
                    (actual.breakdown.variance, expected.breakdown.variance),
                    (actual.breakdown.min_level, expected.breakdown.min_level),
                    (actual.breakdown.cost, expected.breakdown.cost),
                ] {
                    assert!(
                        (a - b).abs() < 1e-7 * (1.0 + b.abs()),
                        "{solver:?}/{kind:?}: {a} != {b}"
                    );
                }
                for parameter in 0..2 {
                    assert!(
                        (actual.gradient[parameter] - expected.gradient[parameter]).abs()
                            < 1e-6 * (1.0 + expected.gradient[parameter].abs())
                    );
                    let mut plus = design;
                    let mut minus = design;
                    plus[parameter] += 1e-5;
                    minus[parameter] -= 1e-5;
                    let fd = (reference
                        .evaluate(&plus, &objective, Some(1))
                        .unwrap()
                        .objective
                        - reference
                            .evaluate(&minus, &objective, Some(1))
                            .unwrap()
                            .objective)
                        / 2e-5;
                    assert!((actual.gradient[parameter] - fd).abs() < 2e-5 * (1.0 + fd.abs()));
                }
            }
        }
        let cancel = AtomicBool::new(true);
        let before = pool.backend().begin_runtime_stats();
        assert!(
            session
                .evaluate_with_cancel(&[0.3, 0.4], &objective, Some(&cancel))
                .is_err()
        );
        assert_eq!(
            pool.backend()
                .runtime_stats_since(before)
                .allocation_operations,
            0
        );
        assert_eq!(
            pool.backend()
                .runtime_stats_since(before)
                .transfer
                .h2d_bytes,
            0
        );
        cancel.store(false, std::sync::atomic::Ordering::Relaxed);
        session
            .evaluate_with_cancel(&[0.3, 0.4], &objective, Some(&cancel))
            .unwrap();
        pool.backend()
            .inject_failure(CudaFailureInjectionPoint::TransferD2D);
        assert!(session.evaluate(&[0.2, 0.4], &objective).is_err());
        pool.backend().recover_after_error().unwrap();
        let repaired = session.evaluate(&[0.2, 0.4], &objective).unwrap();
        let expected = reference
            .evaluate(&[0.2, 0.4], &objective, Some(1))
            .unwrap();
        assert!(
            (repaired.objective - expected.objective).abs()
                < 1e-7 * (1.0 + expected.objective.abs())
        );
        let fields: Vec<_> = session.frequencies.iter().map(field_addresses).collect();
        pool.backend()
            .inject_failure(CudaFailureInjectionPoint::TransferH2D);
        assert!(session.try_refresh_preconditioners(&[0.6, 0.1]).is_err());
        pool.backend().recover_after_error().unwrap();
        assert_eq!(
            session
                .frequencies
                .iter()
                .map(field_addresses)
                .collect::<Vec<_>>(),
            fields
        );
        session.try_refresh_preconditioners(&[0.6, 0.1]).unwrap();
        for state in &session.frequencies {
            assert!(state.forward.solver.session().solution().is_none());
            assert!(state.adjoint.solver.session().solution().is_none());
        }
        let before = pool.backend().begin_runtime_stats();
        let refreshed = session.evaluate(&[0.6, 0.1], &objective).unwrap();
        assert_eq!(
            pool.backend()
                .runtime_stats_since(before)
                .allocation_operations,
            0
        );
        assert_eq!(
            session
                .frequencies
                .iter()
                .map(field_addresses)
                .collect::<Vec<_>>(),
            fields
        );
        let expected = reference
            .evaluate(&[0.6, 0.1], &objective, Some(1))
            .unwrap();
        assert!(
            (refreshed.objective - expected.objective).abs()
                < 1e-7 * (1.0 + expected.objective.abs())
        );
        for (a, b) in refreshed.gradient.iter().zip(&expected.gradient) {
            assert!((a - b).abs() < 1e-6 * (1.0 + b.abs()));
        }
        let config = MultiFrequencyRoomTreatmentConfig {
            max_iterations: 3,
            tolerance: 1e-8,
            objective: objective.clone(),
            max_frequency_lanes: Some(2),
            density_filter: Some(
                TreatmentDensityFilter::new(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 1.5).unwrap(),
            ),
            ..Default::default()
        };
        let actual = session.optimize(&config).unwrap();
        let expected = reference.optimize(&config).unwrap();
        assert_eq!(actual.iterations, expected.iterations);
        for (a, b) in actual
            .objective_history
            .iter()
            .zip(&expected.objective_history)
        {
            assert!((a - b).abs() < 1e-6 * (1.0 + b.abs()));
        }
        for (a, b) in actual.treatments.iter().zip(&expected.treatments) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}

#[test]
fn native_cuda_body_fitted_treatment_handoff() {
    use crate::treatment_projection::{
        BodyFittedTreatmentProjection, optimize_and_promote_body_fitted_treatment,
        optimize_and_promote_body_fitted_treatment_cuda, promote_projected_treatment_cuda,
        reanalyze_projected_treatment, reanalyze_projected_treatment_cuda,
    };
    let backend = match CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                "CUDA required: {error}"
            );
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 29)).unwrap();
    let source_problem = band_problem_on_grid(SolverType::GmresIlu, 4);
    let target_problem = band_problem_on_grid(SolverType::GmresSchwarz, 5);
    let source_reference = band_problem_on_grid(SolverType::Direct, 4);
    let target_reference = band_problem_on_grid(SolverType::Direct, 5);
    assert_ne!(
        source_problem.frequencies[0].base_operator.num_rows,
        target_problem.frequencies[0].base_operator.num_rows
    );
    let workload = FemWorkload::native_cuda(FemScalar::Complex64, FemAction::Forward);
    let mut source = CudaMultiFrequencyRoomTreatmentSession::try_prepare(
        &source_problem,
        &pool,
        workload,
        &[0.25, 0.25],
    )
    .unwrap();
    let mut target = CudaMultiFrequencyRoomTreatmentSession::try_prepare(
        &target_problem,
        &pool,
        workload,
        &[0.25, 0.25],
    )
    .unwrap();
    let projection = BodyFittedTreatmentProjection::new(
        &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        &[[0.25, 0.0, 0.0], [0.75, 0.0, 0.0]],
        1.0,
    )
    .unwrap();
    let objective = TreatmentObjectiveConfig {
        reference_pressure: 1.0,
        target_levels_db: vec![0.0],
        cost_weight: 0.5,
        ..Default::default()
    };
    let baseline = [0.2, 0.2];
    target.evaluate(&baseline, &objective).unwrap();
    let expected = reanalyze_projected_treatment(
        &projection,
        &[0.0, 1.0],
        &target_reference,
        &baseline,
        &objective,
        Some(1),
    )
    .unwrap();
    let before = pool.backend().begin_runtime_stats();
    let actual = reanalyze_projected_treatment_cuda(
        &projection,
        &[0.0, 1.0],
        &mut target,
        &baseline,
        &objective,
    )
    .unwrap();
    assert_eq!(
        pool.backend()
            .runtime_stats_since(before)
            .allocation_operations,
        0
    );
    assert_eq!(actual.target_treatments, vec![0.25, 0.75]);
    assert!(
        (actual.baseline.objective - expected.baseline.objective).abs()
            < 1e-7 * (1.0 + expected.baseline.objective.abs())
    );
    assert!(
        (actual.projected.objective - expected.projected.objective).abs()
            < 1e-7 * (1.0 + expected.projected.objective.abs())
    );
    for (a, b) in actual
        .projected
        .gradient
        .iter()
        .zip(&expected.projected.gradient)
    {
        assert!((a - b).abs() < 1e-6 * (1.0 + b.abs()));
    }
    // Select explicitly different target objectives to exercise the rejection gate.
    let mut candidates = [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
        .into_iter()
        .map(|design| {
            let mapped = projection.project(&design).unwrap();
            let value = target_reference
                .evaluate(&mapped, &objective, Some(1))
                .unwrap()
                .objective;
            (value, design, mapped)
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
    assert!(candidates[3].0 - candidates[0].0 > 1e-4);
    assert!(
        promote_projected_treatment_cuda(
            &projection,
            &candidates[3].1,
            &mut target,
            &candidates[0].2,
            &objective,
            0.0
        )
        .is_err()
    );
    let wrong =
        BodyFittedTreatmentProjection::new(&[[0.0, 0.0, 0.0]], &[[0.0, 0.0, 0.0]], 1.0).unwrap();
    let before = pool.backend().begin_runtime_stats();
    assert!(
        reanalyze_projected_treatment_cuda(&wrong, &[0.5], &mut target, &[0.5], &objective)
            .is_err()
    );
    assert_eq!(
        pool.backend()
            .runtime_stats_since(before)
            .transfer
            .h2d_bytes,
        0
    );
    assert_eq!(
        pool.backend()
            .runtime_stats_since(before)
            .allocation_operations,
        0
    );
    pool.backend()
        .inject_failure(CudaFailureInjectionPoint::TransferD2D);
    assert!(
        promote_projected_treatment_cuda(
            &projection,
            &[0.0, 1.0],
            &mut target,
            &baseline,
            &objective,
            1e6
        )
        .is_err()
    );
    pool.backend().recover_after_error().unwrap();
    reanalyze_projected_treatment_cuda(
        &projection,
        &[0.0, 1.0],
        &mut target,
        &baseline,
        &objective,
    )
    .unwrap();
    let strict = MultiFrequencyRoomTreatmentConfig {
        max_iterations: 1,
        tolerance: 1e-14,
        objective: objective.clone(),
        ..Default::default()
    };
    let error = optimize_and_promote_body_fitted_treatment_cuda(
        &mut source,
        &strict,
        &projection,
        &mut target,
        &baseline,
        &objective,
        1e6,
    )
    .unwrap_err();
    assert!(error.to_string().contains("did not converge"), "{error}");
    // A permissive optimizer stopping tolerance is explicit for this handoff smoke test.
    let config = MultiFrequencyRoomTreatmentConfig {
        max_iterations: 2,
        tolerance: 1.0,
        objective: objective.clone(),
        ..Default::default()
    };
    let reference_source = source_reference.optimize(&config).unwrap();
    assert!(reference_source.converged);
    let matching_baseline = projection.project(&reference_source.treatments).unwrap();
    let expected = optimize_and_promote_body_fitted_treatment(
        &source_reference,
        &config,
        &projection,
        &target_reference,
        &matching_baseline,
        &objective,
        Some(1),
        1e-6,
    )
    .unwrap();
    let actual = optimize_and_promote_body_fitted_treatment_cuda(
        &mut source,
        &config,
        &projection,
        &mut target,
        &matching_baseline,
        &objective,
        1e-6,
    )
    .unwrap();
    assert!(actual.source.converged);
    for (a, b) in actual
        .source
        .treatments
        .iter()
        .zip(&expected.source.treatments)
    {
        assert!((a - b).abs() < 1e-6);
    }
    assert!(
        (actual.promotion.evaluation.objective - expected.promotion.evaluation.objective).abs()
            < 1e-6
    );
    assert!(actual.promotion.objective_delta <= 1e-6);
}
