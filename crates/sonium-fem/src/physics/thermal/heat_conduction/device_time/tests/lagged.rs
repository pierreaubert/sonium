//! Physical parity and failure isolation for explicitly lagged thermal AMG.

// Rust guideline compliant 2026-02-21
use super::*;

#[test]
fn native_thermal_lagged_updates_preserve_diffusion_and_residency() {
    let Some(pool) = pool() else { return };
    for (dimension, mesh) in [
        (2, rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 6, 6)),
        (3, unit_cube_tetrahedra(3)),
    ] {
        let (mass, stiffness, initial) = fixture(mesh, PolynomialDegree::P2, dimension);
        let scheme = ThermalTimeStepper::backward_euler(0.001);
        let mut session = scheme
            .prepare_cuda(
                &pool,
                &mass,
                &stiffness,
                &initial,
                DevicePcgAssumption::PositiveDefinite,
            )
            .unwrap();
        let zeros = vec![0.0; initial.len()];
        let mut source = pool.try_alloc(initial.len()).unwrap();
        pool.backend()
            .try_copy_h2d(&zeros, source.raw_mut())
            .unwrap();
        let mut cpu = initial;
        for (i, (dt, theta)) in [
            (0.001, 1.0),
            (0.0005, 0.5),
            (0.002, 0.75),
            (0.00001, 0.0),
            (0.001, 1.0),
        ]
        .into_iter()
        .enumerate()
        {
            let scheme = ThermalTimeStepper::theta(dt, theta);
            let committed = download(&pool, session.temperature());
            let before = pool.backend().begin_runtime_stats();
            session
                .try_update_scheme_lagged(scheme, DevicePcgAssumption::PositiveDefinite)
                .unwrap();
            let stats = pool.backend().runtime_stats_since(before);
            assert_eq!(stats.transfer.h2d_bytes, 0, "{stats:?}");
            if i > 0 {
                assert_eq!(stats.allocation_operations, 0, "{stats:?}");
            }
            assert_eq!(download(&pool, session.temperature()), committed);
            cpu = scheme
                .step(
                    &cpu,
                    &zeros,
                    Some(&zeros),
                    &mass,
                    &stiffness,
                    &LinearSolveConfig {
                        solver: LinearSolverType::GmresIlu,
                        tolerance: 1e-12,
                        ..Default::default()
                    },
                )
                .unwrap()
                .x;
            let before = pool.backend().begin_runtime_stats();
            let status = session
                .step_device(source.raw(), Some(source.raw()), &controls())
                .unwrap();
            let stats = pool.backend().runtime_stats_since(before);
            assert!(status.converged, "{status:?}");
            assert_eq!(stats.transfer.h2d_bytes, 0);
            if i > 0 {
                assert_eq!(stats.allocation_operations, 0);
            }
            let error = download(&pool, session.temperature())
                .iter()
                .zip(&cpu)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(error < 2e-8, "{dimension}D step {i}: {error}");
            assert_eq!(session.steps_completed(), i + 1);
            eprintln!(
                "thermal lagged {dimension}D step={i} iterations={} residual={} cpu_error={error}",
                status.iterations, status.true_residual
            );
        }
    }
}

#[test]
fn native_thermal_lagged_failure_needs_complete_repair() {
    let Some(pool) = pool() else { return };
    let mass = CsrMatrix::from_triplets(1, 1, vec![(0, 0, 2.0)]);
    let stiffness = CsrMatrix::from_triplets(1, 1, vec![(0, 0, 3.0)]);
    let scheme = ThermalTimeStepper::backward_euler(0.1);
    let mut session = scheme
        .prepare_cuda(
            &pool,
            &mass,
            &stiffness,
            &[2.0],
            DevicePcgAssumption::PositiveDefinite,
        )
        .unwrap();
    let mut source = pool.try_alloc(1).unwrap();
    pool.backend()
        .try_copy_h2d(&[0.0], source.raw_mut())
        .unwrap();
    assert!(
        session
            .step_device(source.raw(), None, &controls())
            .unwrap()
            .converged
    );
    let committed = download(&pool, session.temperature());
    for point in [
        CudaFailureInjectionPoint::TransferD2D,
        CudaFailureInjectionPoint::Launch,
    ] {
        pool.backend().inject_failure(point);
        assert!(
            session
                .try_update_scheme_lagged(scheme, DevicePcgAssumption::PositiveDefinite)
                .is_err(),
            "{point:?}"
        );
        pool.backend().recover_after_error().unwrap();
        assert_eq!(download(&pool, session.temperature()), committed);
        assert_eq!(session.steps_completed(), 1);
        assert!(
            session
                .step_device(source.raw(), None, &controls())
                .is_err()
        );
        assert!(
            session
                .try_update_scheme_lagged(scheme, DevicePcgAssumption::PositiveDefinite)
                .is_err()
        );
        session
            .try_update_scheme(scheme, DevicePcgAssumption::PositiveDefinite)
            .unwrap();
    }
    // A finite coefficient can still overflow the combined matrix on-device.
    assert!(
        session
            .try_update_scheme_lagged(
                ThermalTimeStepper::backward_euler(1e-308),
                DevicePcgAssumption::PositiveDefinite
            )
            .is_err()
    );
    assert_eq!(download(&pool, session.temperature()), committed);
    assert!(!session.numeric_ready);
    session
        .try_update_scheme(scheme, DevicePcgAssumption::PositiveDefinite)
        .unwrap();
    assert!(
        session
            .step_device(source.raw(), None, &controls())
            .unwrap()
            .converged
    );
    assert!((download(&pool, session.temperature())[0] - committed[0] * 20.0 / 23.0).abs() < 1e-10);
}

#[test]
#[ignore = "controlled release-mode CUDA timing experiment"]
fn profile_thermal_lagged_amg() {
    let Some(pool) = pool() else { return };
    for (dimension, mesh) in [
        (2, rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 32, 32)),
        (3, unit_cube_tetrahedra(8)),
    ] {
        let (mass, stiffness, initial) = fixture(mesh, PolynomialDegree::P2, dimension);
        let zeros = vec![0.0; initial.len()];
        let mut source = pool.try_alloc(initial.len()).unwrap();
        pool.backend()
            .try_copy_h2d(&zeros, source.raw_mut())
            .unwrap();
        let schemes = [
            (0.001, 1.0),
            (0.0008, 1.0),
            (0.0012, 0.5),
            (0.0009, 0.75),
            (0.0011, 1.0),
        ];
        let mut cpu = initial.clone();
        let mut references = Vec::new();
        for (dt, theta) in schemes {
            cpu = ThermalTimeStepper::theta(dt, theta)
                .step(
                    &cpu,
                    &zeros,
                    Some(&zeros),
                    &mass,
                    &stiffness,
                    &LinearSolveConfig {
                        solver: LinearSolverType::GmresIlu,
                        tolerance: 1e-12,
                        ..Default::default()
                    },
                )
                .unwrap()
                .x;
            references.push(cpu.clone());
        }
        // Identical trajectories with ABBA ordering; cold setup is reported separately.
        for (run, lagged) in [false, true, true, false].into_iter().enumerate() {
            pool.backend().try_synchronize().unwrap();
            let start = std::time::Instant::now();
            let mut session = ThermalTimeStepper::backward_euler(0.001)
                .prepare_cuda(
                    &pool,
                    &mass,
                    &stiffness,
                    &initial,
                    DevicePcgAssumption::PositiveDefinite,
                )
                .unwrap();
            pool.backend().try_synchronize().unwrap();
            let setup_ms = start.elapsed().as_secs_f64() * 1e3;
            for (step, (dt, theta)) in schemes.into_iter().enumerate() {
                let scheme = ThermalTimeStepper::theta(dt, theta);
                let before = pool.backend().begin_runtime_stats();
                let start = std::time::Instant::now();
                let rebuilt = if lagged {
                    session
                        .try_update_scheme_lagged(scheme, DevicePcgAssumption::PositiveDefinite)
                        .unwrap();
                    false
                } else {
                    session
                        .try_update_scheme(scheme, DevicePcgAssumption::PositiveDefinite)
                        .unwrap()
                };
                let update_ms = start.elapsed().as_secs_f64() * 1e3;
                let update_stats = pool.backend().runtime_stats_since(before);
                let start = std::time::Instant::now();
                let status = session
                    .step_device(source.raw(), Some(source.raw()), &controls())
                    .unwrap();
                let solve_ms = start.elapsed().as_secs_f64() * 1e3;
                let stats = pool.backend().runtime_stats_since(before);
                assert!(status.converged, "{status:?}");
                let error = download(&pool, session.temperature())
                    .iter()
                    .zip(&references[step])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);
                assert!(error < 2e-8, "{error}");
                if lagged {
                    assert_eq!(stats.transfer.h2d_bytes, 0);
                    if step > 0 {
                        assert_eq!(stats.allocation_operations, 0);
                    }
                }
                eprintln!(
                    "THERMAL_PROFILE {}",
                    serde_json::json!({
                        "dimension": dimension, "dofs": initial.len(), "run": run, "lagged": lagged,
                        "step": step, "dt": dt, "theta": theta, "setup_ms": setup_ms,
                        "update_ms": update_ms, "solve_ms": solve_ms, "iterations": status.iterations,
                        "true_residual": status.true_residual, "cpu_error": error,
                        "preconditioner_rebuilt": rebuilt, "update_h2d_bytes": update_stats.transfer.h2d_bytes,
                        "update_d2h_bytes": update_stats.transfer.d2h_bytes,
                        "total_h2d_bytes": stats.transfer.h2d_bytes, "total_d2h_bytes": stats.transfer.d2h_bytes,
                        "total_allocations": stats.allocation_operations
                    })
                );
            }
        }
    }
}
