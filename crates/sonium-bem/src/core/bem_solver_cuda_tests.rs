//! Public-producer parity and residency for the dense CUDA BEM session.

// Rust guideline compliant 2026-02-21
use super::*;

#[test]
fn native_public_cuda_session_matches_cpu_fields_and_reuses_factors() {
    let backend = match CudaBackend::new_default() {
        Ok(backend) => backend,
        Err(error) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").is_err(),
                "CUDA required: {error}"
            );
            return;
        }
    };
    let solver = BemSolver {
        tolerance: 1e-9,
        ..BemSolver::default()
    };
    for boundary in [
        BoundaryConditionType::Rigid,
        BoundaryConditionType::Impedance,
    ] {
        for burton_miller in [false, true] {
            let mut problem = BemProblem {
                mesh: generate_icosphere_mesh(1.0, 1),
                physics: PhysicsParams::new(343.0 / (2.0 * PI), 343.0, 1.21, false),
                incident_field: IncidentField::plane_wave_z(),
                bc_type: boundary,
                use_burton_miller: burton_miller,
            };
            let mut session = solver.prepare_cuda(&problem, &backend).unwrap();
            assert!(session.solution_device().is_none());
            for pass in 0..3 {
                problem.physics = PhysicsParams::new(
                    (1.0 + pass.min(1) as f64 * 0.01) * 343.0 / (2.0 * PI),
                    343.0,
                    1.21,
                    false,
                );
                problem.incident_field =
                    IncidentField::plane_wave([0.2, 0.3, 1.0], 1.0 + pass as f64 * 0.1);
                let cpu = solver.solve(&problem).unwrap();
                let before = backend.begin_runtime_stats();
                let report = session.solve_device(&problem).unwrap();
                let stats = backend.runtime_stats_since(before);
                assert_eq!(report.factorization_reused, pass == 2);
                if pass > 0 {
                    assert_eq!(stats.allocation_operations, 0, "{stats:?}");
                    assert_eq!(
                        stats.transfer.h2d_bytes,
                        10 * 80 * size_of::<Complex64>() + 6 * size_of::<f64>()
                    );
                    assert!(
                        stats.transfer.d2h_bytes <= 64,
                        "bulk output in solve: {stats:?}"
                    );
                }
                let gpu = session.download_solution().unwrap();
                let error = cpu
                    .surface_pressure
                    .iter()
                    .zip(&gpu.surface_pressure)
                    .map(|(a, b)| (*a - *b).norm_sqr())
                    .sum::<f64>()
                    .sqrt();
                let reference = cpu
                    .surface_pressure
                    .iter()
                    .map(|x| x.norm_sqr())
                    .sum::<f64>()
                    .sqrt();
                assert!(
                    error / reference < 1e-9,
                    "surface error {}",
                    error / reference
                );
                for point in [[0.0, 0.0, 2.0], [2.0, 0.0, 0.0], [0.5, -2.0, 0.25]] {
                    let expected = cpu.evaluate_pressure(&point);
                    let actual = gpu.evaluate_pressure(&point);
                    assert!((expected - actual).norm() < 1e-9 * expected.norm().max(1.0));
                }
                eprintln!(
                    "public CUDA BEM boundary={boundary:?} BM={burton_miller} pass={pass} residual={} surface_error={} allocations={} h2d={} d2h={}",
                    report.relative_residual,
                    error / reference,
                    stats.allocation_operations,
                    stats.transfer.h2d_bytes,
                    stats.transfer.d2h_bytes
                );
            }
            // Fail before publication, then recover the retained session.
            let saved = problem.mesh.nodes[0];
            problem.mesh.nodes[0][0] += 0.001;
            assert!(session.solve_device(&problem).is_err());
            assert!(session.solution_device().is_none());
            assert!(session.download_solution().is_err());
            problem.mesh.nodes[0] = saved;
            session.solve_device(&problem).unwrap();
            assert!(session.solution_device().is_some());
            let saved_frequency = problem.physics.clone();
            problem.physics = PhysicsParams::new(10000.0, 343.0, 1.21, false);
            assert!(
                session.solve_device(&problem).is_err(),
                "changed quadrature must require rebuild"
            );
            assert!(session.solution_device().is_none());
            problem.physics = saved_frequency;
            session.solve_device(&problem).unwrap();
            let compressed = solver.clone().with_assembly_method(AssemblyMethod::Hmatrix);
            let before = backend.begin_runtime_stats();
            assert!(compressed.prepare_cuda(&problem, &backend).is_err());
            assert_eq!(backend.runtime_stats_since(before).allocation_operations, 0);
            let higher_order = solver.clone().with_basis_order(BasisOrder::Linear);
            assert!(higher_order.prepare_cuda(&problem, &backend).is_err());
        }
    }
}

#[test]
fn native_incident_rhs_all_sources_conventions_and_invalid_inputs() {
    let backend = match CudaBackend::new_default() {
        Ok(backend) => backend,
        Err(error) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").is_err(),
                "CUDA required: {error}"
            );
            return;
        }
    };
    let n = 131;
    let mut centers: Vec<_> = (0..n).map(|i| [0.01 * i as f64, 0.3, -0.2]).collect();
    centers[0] = [0.0, 0.0, 0.0];
    centers[1] = [1e-11, 0.0, 0.0];
    let normals = vec![[0.2, -0.3, 0.5]; n];
    let geometry: Vec<_> = centers
        .iter()
        .zip(&normals)
        .flat_map(|(x, n)| x.iter().chain(n).copied())
        .collect();
    let mut points = backend.raw_alloc::<f64>(6 * n).unwrap();
    backend.try_copy_h2d(&geometry, &mut points).unwrap();
    let mut rhs = backend.raw_alloc::<Complex64>(n).unwrap();
    let c = Complex64::new;
    let fields = [
        IncidentField::PlaneWave {
            direction: [0.2, -0.4, 0.7],
            amplitude: c(0.7, -0.2),
        },
        IncidentField::PointSource {
            position: [0.0, 0.0, 0.0],
            strength: c(0.3, 0.4),
        },
        IncidentField::MultiplePlaneWaves(vec![
            ([1.0, 0.0, 0.0], c(0.5, 0.3)),
            ([0.0, -1.0, 0.0], c(-0.2, 0.7)),
        ]),
        IncidentField::MultiplePointSources(vec![
            ([0.0, 0.0, 0.0], c(0.3, 0.4)),
            ([2.0, -0.3, 0.5], c(-0.2, 0.7)),
        ]),
        IncidentField::MultiplePointSources(Vec::new()),
    ];
    for field in fields {
        let packed = pack_incident_sources(&field);
        let mut sources = backend.raw_alloc::<f64>(packed.len()).unwrap();
        backend.try_copy_h2d(&packed, &mut sources).unwrap();
        for harmonic in [-1.0, 1.0] {
            for tau in [-1.0, 1.0] {
                let mut physics = PhysicsParams::new(85.0, 343.0, 1.21, false);
                physics.harmonic_factor = harmonic;
                physics.tau = tau;
                let beta = c(0.13, -0.21);
                let expected = field.compute_rhs_with_beta(&centers, &normals, &physics, beta);
                let before = backend.begin_runtime_stats();
                backend
                    .try_bem_incident_rhs_c64_device(
                        &points,
                        &sources,
                        packed.len() / 6,
                        physics.wave_number,
                        harmonic,
                        physics.gamma(),
                        beta * tau,
                        &mut rhs,
                    )
                    .unwrap();
                let stats = backend.runtime_stats_since(before);
                assert_eq!(stats.transfer.h2d_bytes, 0);
                assert!(stats.transfer.d2h_bytes <= 12);
                let mut actual = vec![c(0.0, 0.0); n];
                backend.try_copy_d2h(&rhs, &mut actual).unwrap();
                for (a, e) in actual.iter().zip(&expected) {
                    assert!(
                        (*a - *e).norm() < 1e-12 * e.norm().max(1.0),
                        "actual={a} expected={e}"
                    );
                }
            }
        }
    }
    let mut bad_source = backend.raw_alloc::<f64>(6).unwrap();
    backend
        .try_copy_h2d(&[0.0, 0.0, 1.0, 1.0, 0.0, 2.0], &mut bad_source)
        .unwrap();
    assert!(
        backend
            .try_bem_incident_rhs_c64_device(
                &points,
                &bad_source,
                1,
                1.0,
                1.0,
                1.0,
                c(0.0, 0.2),
                &mut rhs
            )
            .is_err()
    );
    backend
        .try_copy_h2d(&[f64::NAN, 0.0, 1.0, 1.0, 0.0, 0.0], &mut bad_source)
        .unwrap();
    assert!(
        backend
            .try_bem_incident_rhs_c64_device(
                &points,
                &bad_source,
                1,
                1.0,
                1.0,
                1.0,
                c(0.0, 0.2),
                &mut rhs
            )
            .is_err()
    );
    assert!(
        backend
            .try_bem_incident_rhs_c64_device(
                &points,
                &bad_source,
                2,
                1.0,
                1.0,
                1.0,
                c(0.0, 0.2),
                &mut rhs
            )
            .is_err()
    );
}
