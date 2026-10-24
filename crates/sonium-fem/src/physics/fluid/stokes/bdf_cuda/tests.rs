//! Production BDF parity, cache invalidation and failure recovery on CUDA.

// Rust guideline compliant 2026-02-21
use super::*;
use crate::basis::PolynomialDegree;
use crate::mesh::generators::rectangular_mesh_triangles;
use crate::physics::fluid::stokes::{
    navier_stokes::{LinearSolverBackend, solve_bdf_step_cached},
    time_stepping::{BdfConfig, BdfState},
    traction::PressureTraction,
};
use crate::spaces::{FunctionSpace, H1Space, VectorH1Space};

fn cuda_available() -> bool {
    match CudaBackend::new(0) {
        Ok(_) => true,
        Err(error) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                "CUDA required: {error}"
            );
            eprintln!("SKIP retained BDF CUDA: {error}");
            false
        }
    }
}

fn compare(a: &BdfState, b: &BdfState) {
    for (name, a, b) in [
        ("velocity", &a.velocity, &b.velocity),
        ("pressure", &a.pressure, &b.pressure),
        ("history", &a.velocity_prev, &b.velocity_prev),
    ] {
        let error = a
            .iter()
            .zip(b)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale = a.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
        assert!(
            error / scale < 2e-6,
            "{name} relative error {}",
            error / scale
        );
    }
    assert_eq!(a.step, b.step);
    assert_eq!(a.time, b.time);
}

#[test]
fn native_bdf_cuda_schur_reuses_probes_and_advances_history() {
    if !cuda_available() {
        return;
    }
    let mut mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.1, 6, 3);
    for face in &mut mesh.boundaries {
        face.marker = if face.nodes.iter().all(|&i| mesh.nodes[i].x == 0.0) {
            1
        } else if face.nodes.iter().all(|&i| mesh.nodes[i].x == 1.0) {
            2
        } else {
            3
        };
    }
    let v = VectorH1Space::new(PolynomialDegree::P2, 2);
    let p = H1Space::new(PolynomialDegree::P1);
    let nv = v.num_global_dofs(&mesh);
    let np = p.num_global_dofs(&mesh);
    let boundaries = v.boundary_dofs(&mesh, 3);
    let bc = |_: f64, _: f64, _: f64| vec![0.0, 0.0];
    let bdf = BdfConfig::bdf2(1e-3, 3);
    for kind in [
        BackendPressureSchurPreconditionerKind::Polynomial,
        BackendPressureSchurPreconditionerKind::Chebyshev,
        BackendPressureSchurPreconditionerKind::Amg,
    ] {
        let config = NavierStokesConfig {
            mu: 1.85e-5,
            rho: 1.2,
            use_residual_stabilisation: false,
            use_direct_solver: false,
            linear_solver_backend: LinearSolverBackend::Cuda,
            linear_solver_backend_preconditioner: BackendPreconditionerKind::JacobiSchur,
            linear_solver_schur_preconditioner: kind,
            linear_solver_max_iterations: 500,
            linear_solver_tolerance: 1e-11,
            ..Default::default()
        };
        let cpu = NavierStokesConfig {
            use_direct_solver: true,
            linear_solver_backend: LinearSolverBackend::Cpu,
            ..config.clone()
        };
        let mut workspace = BdfWorkspace::new(&mesh, &v, &p);
        let reference_workspace = BdfWorkspace::new(&mesh, &v, &p);
        let tractions = [
            PressureTraction {
                marker: 1,
                pressure_pa: 1.0,
            },
            PressureTraction {
                marker: 2,
                pressure_pa: 0.0,
            },
        ];
        let mut reference = BdfState::from_rest(nv, np);
        let mut state = BdfState::from_rest(nv, np);
        for probe in 0..2 {
            let probe_tractions = [
                PressureTraction {
                    marker: 1,
                    pressure_pa: if probe == 0 { 1.0 } else { 0.5 },
                },
                PressureTraction {
                    marker: 2,
                    pressure_pa: 0.0,
                },
            ];
            reference = BdfState::from_rest(nv, np);
            state = BdfState::from_rest(nv, np);
            assert!(solve_bdf_step_cached(
                &mesh,
                &v,
                &p,
                &cpu,
                &bdf,
                &mut reference,
                &bc,
                &boundaries,
                &probe_tractions,
                &reference_workspace
            ));
            assert!(solve_bdf_step_cached(
                &mesh,
                &v,
                &p,
                &config,
                &bdf,
                &mut state,
                &bc,
                &boundaries,
                &probe_tractions,
                &workspace
            ));
            let telemetry = workspace.last_linear_solve_telemetry().unwrap();
            assert_eq!(
                telemetry.resolved_linear_solver_backend,
                LinearSolverBackend::Cuda
            );
            assert_eq!(telemetry.setup_reused, probe > 0);
            compare(&reference, &state);
            let cache = workspace.cuda_solver_cache.borrow();
            let cache = cache.as_ref().unwrap();
            assert_eq!(cache.setup_builds, 1);
            assert_eq!(cache.matrix_uploads, 1);
            if probe > 0 {
                assert_eq!(cache.last_allocations, 0);
            }
            eprintln!(
                "BDF {kind:?} probe={probe} iterations={} allocations={} reused={}",
                telemetry.iterations, cache.last_allocations, telemetry.setup_reused
            );
        }
        for step in 1..3 {
            assert!(solve_bdf_step_cached(
                &mesh,
                &v,
                &p,
                &cpu,
                &bdf,
                &mut reference,
                &bc,
                &boundaries,
                &tractions,
                &reference_workspace
            ));
            assert!(solve_bdf_step_cached(
                &mesh,
                &v,
                &p,
                &config,
                &bdf,
                &mut state,
                &bc,
                &boundaries,
                &tractions,
                &workspace
            ));
            let telemetry = workspace.last_linear_solve_telemetry().unwrap();
            assert_eq!(
                telemetry.resolved_linear_solver_backend,
                LinearSolverBackend::Cuda
            );
            compare(&reference, &state);
            let cache = workspace.cuda_solver_cache.borrow();
            let cache = cache.as_ref().unwrap();
            eprintln!(
                "BDF refresh {kind:?} step={step} device_builds={} allocations={}",
                cache.setup_builds, cache.last_allocations
            );
            assert_eq!(cache.setup_builds, 1);
            assert_eq!(cache.last_allocations, 0);
            assert_eq!(cache.matrix_uploads, step + 1);
            eprintln!(
                "BDF {kind:?} step={step} iterations={} setups={} uploads={}",
                telemetry.iterations, cache.setup_builds, cache.matrix_uploads
            );
        }
        // A changed control must rebuild setup even when the physical CSR is identical.
        let (matrix, rhs) = workspace.last_linear_system().unwrap();
        let schur = workspace
            .cuda_solver_cache
            .borrow()
            .as_ref()
            .unwrap()
            .schur
            .clone();
        let mut changed = config.clone();
        changed.linear_solver_polynomial_degree = 2;
        let zero = vec![0.0; matrix.num_rows];
        let (_, _, reused) = solve_cached(
            &workspace,
            &changed,
            &matrix,
            &rhs,
            &zero,
            nv,
            schur.as_ref(),
        )
        .unwrap();
        assert!(!reused);
        assert_eq!(
            workspace
                .cuda_solver_cache
                .borrow()
                .as_ref()
                .unwrap()
                .setup_builds,
            2
        );
        assert_eq!(
            workspace
                .cuda_solver_cache
                .borrow()
                .as_ref()
                .unwrap()
                .matrix_uploads,
            3
        );
        if kind == BackendPressureSchurPreconditionerKind::Amg {
            let (reference, _, _) = solve_cached(
                &workspace,
                &changed,
                &matrix,
                &rhs,
                &zero,
                nv,
                schur.as_ref(),
            )
            .unwrap();
            let mut velocity_changed = matrix.clone();
            for row in 0..nv {
                for j in velocity_changed.row_ptrs[row]..velocity_changed.row_ptrs[row + 1] {
                    if velocity_changed.col_indices[j] < nv
                        && velocity_changed.col_indices[j] != row
                    {
                        velocity_changed.values[j] *= 1.001;
                    }
                }
            }
            assert_ne!(velocity_changed.values, matrix.values);
            let mut changed_rhs = vec![0.0; matrix.num_rows];
            for (row, value) in changed_rhs.iter_mut().enumerate() {
                *value = (velocity_changed.row_ptrs[row]..velocity_changed.row_ptrs[row + 1])
                    .map(|j| {
                        velocity_changed.values[j] * reference[velocity_changed.col_indices[j]]
                    })
                    .sum();
            }
            let (solution, _, reused) = solve_cached(
                &workspace,
                &changed,
                &velocity_changed,
                &changed_rhs,
                &zero,
                nv,
                schur.as_ref(),
            )
            .unwrap();
            assert!(!reused);
            let error = solution
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            let norm = reference.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(error / norm < 2e-6);
            {
                let cache = workspace.cuda_solver_cache.borrow();
                let cache = cache.as_ref().unwrap();
                assert_eq!(cache.setup_builds, 2);
                assert_eq!(cache.last_allocations, 0);
            }
            eprintln!(
                "BDF AMG velocity-only refresh: setups=2 allocations=0 relative_error={}",
                error / norm
            );
            // Changed pressure values must refresh the hierarchy, preserving
            // device storage when every level pattern matches.
            let mut updated_schur = schur.clone().unwrap();
            for value in &mut updated_schur.values {
                *value *= 1.1;
            }
            let (_, _, reused) = solve_cached(
                &workspace,
                &changed,
                &matrix,
                &rhs,
                &zero,
                nv,
                Some(&updated_schur),
            )
            .unwrap();
            assert!(!reused);
            assert_eq!(
                workspace
                    .cuda_solver_cache
                    .borrow()
                    .as_ref()
                    .unwrap()
                    .setup_builds,
                2
            );
            let (_, _, reused) = solve_cached(
                &workspace,
                &changed,
                &matrix,
                &rhs,
                &zero,
                nv,
                Some(&updated_schur),
            )
            .unwrap();
            assert!(reused);
            assert_eq!(
                workspace
                    .cuda_solver_cache
                    .borrow()
                    .as_ref()
                    .unwrap()
                    .last_allocations,
                0
            );
        }
        if kind == BackendPressureSchurPreconditionerKind::Amg {
            // Changed pressure CSR structure requires full device setup.
            let original = schur.as_ref().unwrap();
            let mut entries = Vec::new();
            for row in 0..original.num_rows {
                for j in original.row_ptrs[row]..original.row_ptrs[row + 1] {
                    if original.col_indices[j] == row {
                        entries.push((row, row, original.values[j]));
                    }
                }
            }
            let diagonal_pressure =
                CsrMatrix::from_triplets(original.num_rows, original.num_cols, entries);
            assert!(!same_pattern(&diagonal_pressure, original));
            solve_cached(
                &workspace,
                &changed,
                &matrix,
                &rhs,
                &zero,
                nv,
                Some(&diagonal_pressure),
            )
            .unwrap();
            assert_eq!(
                workspace
                    .cuda_solver_cache
                    .borrow()
                    .as_ref()
                    .unwrap()
                    .setup_builds,
                3
            );
            solve_cached(
                &workspace,
                &changed,
                &matrix,
                &rhs,
                &zero,
                nv,
                schur.as_ref(),
            )
            .unwrap();
        }
        if kind == BackendPressureSchurPreconditionerKind::Amg {
            workspace
                .cuda_solver_cache
                .borrow()
                .as_ref()
                .unwrap()
                .pool
                .backend()
                .inject_preconditioner_failure(
                    sonium_compute::PreconditionerFailureInjectionPoint::DiagonalUpload,
                );
            let mut failed_pressure = schur.clone().unwrap();
            for value in &mut failed_pressure.values {
                *value *= 1.2;
            }
            assert!(
                solve_cached(
                    &workspace,
                    &changed,
                    &matrix,
                    &rhs,
                    &zero,
                    nv,
                    Some(&failed_pressure)
                )
                .is_err()
            );
            assert!(workspace.cuda_solver_cache.borrow().is_none());
            solve_cached(
                &workspace,
                &changed,
                &matrix,
                &rhs,
                &zero,
                nv,
                schur.as_ref(),
            )
            .unwrap();
        }
        // Any interrupted transfer evicts the cache; a later call creates clean state.
        workspace
            .cuda_solver_cache
            .borrow()
            .as_ref()
            .unwrap()
            .pool
            .backend()
            .inject_failure(sonium_compute::CudaFailureInjectionPoint::TransferH2D);
        assert!(
            solve_cached(
                &workspace,
                &changed,
                &matrix,
                &rhs,
                &zero,
                nv,
                schur.as_ref()
            )
            .is_err()
        );
        assert!(workspace.cuda_solver_cache.borrow().is_none());
        assert!(
            solve_cached(
                &workspace,
                &changed,
                &matrix,
                &rhs,
                &zero,
                nv,
                schur.as_ref()
            )
            .is_ok()
        );
        // Preserve the public driver's existing CPU recovery policy, including
        // truthful backend telemetry and an empty failed CUDA cache.
        let recovery = NavierStokesConfig {
            linear_solver_cpu_preconditioner:
                crate::backend_selection::LinearSolverCpuPreconditionerKind::JacobiSchur,
            ..config.clone()
        };
        workspace
            .cuda_solver_cache
            .borrow()
            .as_ref()
            .unwrap()
            .pool
            .backend()
            .inject_failure(sonium_compute::CudaFailureInjectionPoint::TransferH2D);
        let mut recovered = BdfState::from_rest(nv, np);
        assert!(solve_bdf_step_cached(
            &mesh,
            &v,
            &p,
            &recovery,
            &bdf,
            &mut recovered,
            &bc,
            &boundaries,
            &tractions,
            &workspace
        ));
        assert_eq!(
            workspace
                .last_linear_solve_telemetry()
                .unwrap()
                .resolved_linear_solver_backend,
            LinearSolverBackend::Cpu
        );
        assert!(workspace.cuda_solver_cache.borrow().is_none());
        let mut retried = BdfState::from_rest(nv, np);
        assert!(solve_bdf_step_cached(
            &mesh,
            &v,
            &p,
            &recovery,
            &bdf,
            &mut retried,
            &bc,
            &boundaries,
            &tractions,
            &workspace
        ));
        assert_eq!(
            workspace
                .last_linear_solve_telemetry()
                .unwrap()
                .resolved_linear_solver_backend,
            LinearSolverBackend::Cuda
        );
        compare(&recovered, &retried);
        workspace.set_pressure_gauge(0, 0.0);
        assert!(workspace.cuda_solver_cache.borrow().is_none());
    }
}

#[test]
fn native_bdf_cuda_polynomial_and_chebyshev_control_invalidation() {
    if !cuda_available() {
        return;
    }
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 1, 1);
    let v = VectorH1Space::new(PolynomialDegree::P2, 2);
    let p = H1Space::new(PolynomialDegree::P1);
    // This algebra fixture isolates outer preconditioner dispatch and its keys;
    // the separate BDF test supplies production saddle-point evidence.
    let workspace = BdfWorkspace::new(&mesh, &v, &p);
    let matrix = CsrMatrix::from_triplets(
        3,
        3,
        vec![
            (0, 0, 4.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 3.0),
            (2, 2, 2.0),
        ],
    );
    let rhs = [6.0, 7.0, 6.0];
    let zero = [0.0; 3];
    for kind in [
        BackendPreconditionerKind::PolynomialJacobi,
        BackendPreconditionerKind::Chebyshev,
    ] {
        let mut config = NavierStokesConfig {
            linear_solver_backend_preconditioner: kind,
            linear_solver_tolerance: 1e-12,
            linear_solver_max_iterations: 10,
            ..Default::default()
        };
        for pass in 0..3 {
            let (x, _, reused) =
                solve_cached(&workspace, &config, &matrix, &rhs, &zero, 2, None).unwrap();
            for (a, b) in x.iter().zip([1.0, 2.0, 3.0]) {
                assert!((a - b).abs() < 1e-10);
            }
            assert_eq!(reused, pass == 1);
            if pass == 1 {
                config.linear_solver_chebyshev_degree += 1;
                config.linear_solver_polynomial_degree += 1;
            }
        }
    }
    // Changed coefficients retain polynomial storage and refresh both diagonal
    // scaling and the Chebyshev spectral estimate. Compare with fresh setup.
    for kind in [
        BackendPreconditionerKind::PolynomialJacobi,
        BackendPreconditionerKind::Chebyshev,
    ] {
        workspace.cuda_solver_cache.borrow_mut().take();
        let config = NavierStokesConfig {
            linear_solver_backend_preconditioner: kind,
            linear_solver_tolerance: 1e-12,
            linear_solver_max_iterations: 10,
            ..Default::default()
        };
        solve_cached(&workspace, &config, &matrix, &rhs, &zero, 2, None).unwrap();
        let mut changed = matrix.clone();
        for pass in 1..=3 {
            changed.values = vec![4.0 + pass as f64, 0.5, 0.5, 3.0, 2.0];
            let rhs = [5.0 + pass as f64, 6.5, 6.0];
            let (x, _, reused) =
                solve_cached(&workspace, &config, &changed, &rhs, &zero, 2, None).unwrap();
            assert!(
                !reused,
                "numeric setup changed even though storage was retained"
            );
            let cache = workspace.cuda_solver_cache.borrow();
            let cache = cache.as_ref().unwrap();
            assert_eq!(cache.setup_builds, 1);
            assert_eq!(cache.matrix_uploads, pass + 1);
            assert_eq!(cache.last_allocations, 0);
            let fresh = BdfWorkspace::new(&mesh, &v, &p);
            let (reference, _, _) =
                solve_cached(&fresh, &config, &changed, &rhs, &zero, 2, None).unwrap();
            for ((a, b), exact) in x.iter().zip(reference).zip([1.0, 2.0, 3.0]) {
                assert!((a - b).abs() < 1e-10);
                assert!((a - exact).abs() < 1e-10);
            }
            eprintln!("BDF numeric refresh {kind:?} pass={pass} allocations=0 setups=1");
        }
        workspace
            .cuda_solver_cache
            .borrow()
            .as_ref()
            .unwrap()
            .pool
            .backend()
            .inject_failure(sonium_compute::CudaFailureInjectionPoint::TransferD2D);
        changed.values[0] += 1.0;
        let rhs = [9.0, 6.5, 6.0];
        assert!(solve_cached(&workspace, &config, &changed, &rhs, &zero, 2, None).is_err());
        assert!(workspace.cuda_solver_cache.borrow().is_none());
        let (x, _, reused) =
            solve_cached(&workspace, &config, &changed, &rhs, &zero, 2, None).unwrap();
        assert!(!reused);
        for (a, b) in x.iter().zip([1.0, 2.0, 3.0]) {
            assert!((a - b).abs() < 1e-10);
        }
    }
    let larger = CsrMatrix::from_triplets(
        4,
        4,
        vec![(0, 0, 2.0), (1, 1, 3.0), (2, 2, 4.0), (3, 3, 5.0)],
    );
    let (x, _, reused) = solve_cached(
        &workspace,
        &NavierStokesConfig::default(),
        &larger,
        &[2.0, 3.0, 4.0, 5.0],
        &[0.0; 4],
        2,
        None,
    )
    .unwrap();
    assert!(!reused);
    assert!(x.iter().all(|v| (v - 1.0).abs() < 1e-10));
    assert_eq!(
        workspace
            .cuda_solver_cache
            .borrow()
            .as_ref()
            .unwrap()
            .matrix_uploads,
        1
    );
    let config = NavierStokesConfig {
        linear_solver_max_iterations: 1,
        linear_solver_tolerance: 1e-15,
        ..Default::default()
    };
    assert!(solve_cached(&workspace, &config, &matrix, &rhs, &zero, 2, None).is_err());
    assert!(workspace.cuda_solver_cache.borrow().is_none());
}

#[test]
fn native_bdf_cuda_multilevel_pressure_refresh() {
    if !cuda_available() {
        return;
    }
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 1, 1);
    let v = VectorH1Space::new(PolynomialDegree::P2, 2);
    let p = H1Space::new(PolynomialDegree::P1);
    let workspace = BdfWorkspace::new(&mesh, &v, &p);
    let nv = 8;
    let np = 128;
    let n = nv + np;
    let mut entries = Vec::new();
    for row in 0..np {
        entries.push((row, row, 4.0));
        if row + 1 < np {
            entries.extend([(row, row + 1, -1.0), (row + 1, row, -1.0)]);
        }
    }
    let original = CsrMatrix::from_triplets(np, np, entries);
    let matrix = CsrMatrix::from_triplets(n, n, (0..n).map(|i| (i, i, 2.0)).collect());
    let config = NavierStokesConfig {
        linear_solver_backend_preconditioner: BackendPreconditionerKind::JacobiSchur,
        linear_solver_schur_preconditioner: BackendPressureSchurPreconditionerKind::Amg,
        linear_solver_tolerance: 1e-11,
        linear_solver_max_iterations: 500,
        ..Default::default()
    };
    // Algebra fixture isolates hierarchy lifecycle; physical BDF parity is covered above.
    let rhs = vec![2.0; n];
    let guess = vec![0.0; n];
    for pass in 0..3 {
        let mut pressure = original.clone();
        if pass == 1 {
            for value in &mut pressure.values {
                *value *= 1.1;
            }
        }
        if pass == 2 {
            for row in 0..np {
                for j in pressure.row_ptrs[row]..pressure.row_ptrs[row + 1] {
                    if pressure.col_indices[j] != row {
                        pressure.values[j] = 0.0;
                    }
                }
            }
        }
        assert!(same_pattern(&pressure, &original));
        let (x, _, _) = solve_cached(
            &workspace,
            &config,
            &matrix,
            &rhs,
            &guess,
            nv,
            Some(&pressure),
        )
        .unwrap();
        assert!(x.iter().all(|v| (v - 1.0).abs() < 1e-8));
        let cache = workspace.cuda_solver_cache.borrow();
        let cache = cache.as_ref().unwrap();
        assert_eq!(cache.setup_builds, if pass == 2 { 2 } else { 1 });
        if pass == 1 {
            assert_eq!(cache.last_allocations, 0);
        }
        eprintln!(
            "BDF multilevel AMG pass={pass} device_builds={} allocations={}",
            cache.setup_builds, cache.last_allocations
        );
    }
}

#[test]
fn native_bdf_cuda_pressure_only_multilevel_physical_parity() {
    if !cuda_available() {
        return;
    }
    let nx = std::env::var("SONIUM_BDF_COMPARE_NX")
        .ok()
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(24);
    assert!((12..=48).contains(&nx));
    let mut mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.1, nx, nx / 2);
    for face in &mut mesh.boundaries {
        face.marker = if face.nodes.iter().all(|&i| mesh.nodes[i].x == 0.0) {
            1
        } else if face.nodes.iter().all(|&i| mesh.nodes[i].x == 1.0) {
            2
        } else {
            3
        };
    }
    let v = VectorH1Space::new(PolynomialDegree::P2, 2);
    let p = H1Space::new(PolynomialDegree::P1);
    let nv = v.num_global_dofs(&mesh);
    let np = p.num_global_dofs(&mesh);
    let boundaries = v.boundary_dofs(&mesh, 3);
    let bc = |_: f64, _: f64, _: f64| vec![0.0, 0.0];
    let config = NavierStokesConfig {
        mu: 1.85e-5,
        rho: 1.2,
        use_residual_stabilisation: false,
        use_direct_solver: false,
        linear_solver_backend: LinearSolverBackend::Cuda,
        linear_solver_backend_preconditioner: BackendPreconditionerKind::JacobiSchur,
        linear_solver_schur_preconditioner: BackendPressureSchurPreconditionerKind::Amg,
        linear_solver_bdf_pressure_only_amg: true,
        linear_solver_max_iterations: 1000,
        linear_solver_tolerance: 1e-11,
        ..Default::default()
    };
    let cpu = NavierStokesConfig {
        use_direct_solver: true,
        linear_solver_backend: LinearSolverBackend::Cpu,
        ..config.clone()
    };
    let workspace = BdfWorkspace::new(&mesh, &v, &p);
    let cpu_workspace = BdfWorkspace::new(&mesh, &v, &p);
    let bdf = BdfConfig::bdf2(1e-3, 2);
    let tractions = [
        PressureTraction {
            marker: 1,
            pressure_pa: 1.0,
        },
        PressureTraction {
            marker: 2,
            pressure_pa: 0.0,
        },
    ];
    let mut state = BdfState::from_rest(nv, np);
    let mut reference = BdfState::from_rest(nv, np);
    for step in 0..2 {
        assert!(solve_bdf_step_cached(
            &mesh,
            &v,
            &p,
            &cpu,
            &bdf,
            &mut reference,
            &bc,
            &boundaries,
            &tractions,
            &cpu_workspace
        ));
        assert!(solve_bdf_step_cached(
            &mesh,
            &v,
            &p,
            &config,
            &bdf,
            &mut state,
            &bc,
            &boundaries,
            &tractions,
            &workspace
        ));
        compare(&reference, &state);
        let telemetry = workspace.last_linear_solve_telemetry().unwrap();
        assert_eq!(
            telemetry.resolved_linear_solver_backend,
            LinearSolverBackend::Cuda
        );
        let cache = workspace.cuda_solver_cache.borrow();
        let cache = cache.as_ref().unwrap();
        let Preconditioner::Schur {
            pressure: Some(pressure),
            ..
        } = &cache.preconditioner
        else {
            panic!("expected pressure AMG");
        };
        let diagnostics = pressure.diagnostics();
        assert_eq!(diagnostics.level_dofs[0], np);
        assert!(diagnostics.num_levels >= 2);
        eprintln!(
            "Pressure-only BDF step={step} nv={nv} np={np} levels={:?} iterations={} allocations={} device_builds={}",
            diagnostics.level_dofs,
            telemetry.iterations,
            cache.last_allocations,
            cache.setup_builds
        );
    }
    // Layout changes invalidate the retained preconditioner even for identical matrices.
    let (matrix, rhs) = workspace.last_linear_system().unwrap();
    let (schur, builds) = {
        let cache = workspace.cuda_solver_cache.borrow();
        let cache = cache.as_ref().unwrap();
        (cache.schur.clone(), cache.setup_builds)
    };
    let embedded = NavierStokesConfig {
        linear_solver_bdf_pressure_only_amg: false,
        ..config.clone()
    };
    solve_cached(
        &workspace,
        &embedded,
        &matrix,
        &rhs,
        &vec![0.0; matrix.num_rows],
        nv,
        schur.as_ref(),
    )
    .unwrap();
    {
        let cache = workspace.cuda_solver_cache.borrow();
        let cache = cache.as_ref().unwrap();
        assert_eq!(cache.setup_builds, builds + 1);
        let Preconditioner::Schur {
            pressure: Some(pressure),
            ..
        } = &cache.preconditioner
        else {
            panic!("expected pressure AMG");
        };
        assert_eq!(pressure.diagnostics().level_dofs[0], nv + np);
    }
    if std::env::var("SONIUM_BDF_PRESSURE_COMPARE").as_deref() == Ok("1") {
        compare_pressure_layouts(&workspace, &config, nv);
    }
}

#[test]
fn native_pressure_only_amg_block_views_and_residency() {
    if !cuda_available() {
        return;
    }
    let pool = MemoryPool::new(CudaBackend::new(0).unwrap(), MemoryPoolConfig::default()).unwrap();
    let backend = pool.backend();
    let nv = 64;
    let np = 325;
    let n = nv + np;
    let mut entries = Vec::new();
    for row in 0..np {
        entries.push((row, row, 4.0));
        if row + 1 < np {
            entries.extend([(row, row + 1, -1.0), (row + 1, row, -1.0)]);
        }
    }
    let matrix = CsrMatrix::from_triplets(np, np, entries);
    let action = PressureAmg::new(&pool, n, nv, &matrix, true).unwrap();
    let reference = sonium_solvers::preconditioners::BackendAmgPreconditioner::try_from_csr(
        backend,
        &matrix,
        sonium_solvers::AmgConfig {
            smoother: sonium_solvers::AmgSmoother::Chebyshev,
            num_pre_smooth: 2,
            num_post_smooth: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let mut rhs = pool.try_alloc::<f64>(2 * n).unwrap();
    let mut out = pool.try_alloc::<f64>(2 * n).unwrap();
    let mut pressure_rhs = pool.try_alloc::<f64>(2 * np).unwrap();
    let mut pressure_out = pool.try_alloc::<f64>(2 * np).unwrap();
    let values: Vec<_> = (0..2 * n).map(|i| (i as f64).sin()).collect();
    let mut extracted = values[nv..n].to_vec();
    extracted.extend_from_slice(&values[n + nv..2 * n]);
    backend.try_copy_h2d(&values, rhs.raw_mut()).unwrap();
    backend
        .try_copy_h2d(&extracted, pressure_rhs.raw_mut())
        .unwrap();
    reference
        .try_apply_colmajor(backend, 2, pressure_rhs.raw(), pressure_out.raw_mut())
        .unwrap();
    let mut expected = vec![0.0; 2 * np];
    backend
        .try_copy_d2h(pressure_out.raw(), &mut expected)
        .unwrap();
    for pass in 0..3 {
        backend
            .try_copy_h2d(&vec![f64::NAN; 2 * n], out.raw_mut())
            .unwrap();
        let before = backend.begin_runtime_stats();
        action
            .try_apply_colmajor(backend, 2, rhs.raw(), out.raw_mut())
            .unwrap();
        let stats = backend.runtime_stats_since(before);
        if pass > 0 {
            assert_eq!(stats.allocation_operations, 0);
            assert_eq!(stats.transfer.h2d_bytes, 0);
            assert_eq!(stats.transfer.d2h_bytes, 0);
        }
        let mut actual = vec![0.0; 2 * n];
        backend.try_copy_d2h(out.raw(), &mut actual).unwrap();
        for col in 0..2 {
            assert!(actual[col * n..col * n + nv].iter().all(|v| *v == 0.0));
            for (a, b) in actual[col * n + nv..(col + 1) * n]
                .iter()
                .zip(&expected[col * np..(col + 1) * np])
            {
                assert!((a - b).abs() < 1e-12 * (1.0 + b.abs()));
            }
        }
    }
}

fn compare_pressure_layouts(workspace: &BdfWorkspace, config: &NavierStokesConfig, nv: usize) {
    use sonium_compute::preconditioner::embed_pressure_schur_with_velocity_identity;
    use sonium_solvers::{AmgConfig, AmgSmoother, preconditioners::BackendAmgPreconditioner};
    let (matrix, rhs) = workspace.last_linear_system().unwrap();
    let mut cache = workspace.cuda_solver_cache.borrow_mut();
    let current = cache.as_mut().unwrap();
    let schur = current.schur.clone().unwrap();
    let mut baseline = BdfCudaSolver::new(&matrix, Some(&schur), config, nv).unwrap();
    let backend = baseline.pool.backend();
    let (rows, columns, values) =
        embed_pressure_schur_with_velocity_identity(matrix.num_rows, nv, schur.as_host_csr_ref());
    let embedded =
        CsrMatrix::from_raw_parts(matrix.num_rows, matrix.num_rows, rows, columns, values);
    let old_amg = BackendAmgPreconditioner::try_from_csr(
        backend,
        &embedded,
        AmgConfig {
            smoother: AmgSmoother::Chebyshev,
            num_pre_smooth: 2,
            num_post_smooth: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let old_levels = old_amg.diagnostics().level_dofs;
    let schur_config = SaddlePointJacobiSchurBackendConfig {
        velocity_block_size: nv,
        velocity_degree: config.linear_solver_polynomial_degree.max(1),
        velocity_damping: config.linear_solver_polynomial_damping,
        pressure_degree: config.linear_solver_schur_polynomial_degree.max(1),
        pressure_damping: config.linear_solver_schur_polynomial_damping,
        pressure_kind: PressureSchurBackendKind::Polynomial,
        pressure_chebyshev_degree: config.linear_solver_schur_chebyshev_degree.max(1),
        pressure_chebyshev_eigenvalue_ratio: config
            .linear_solver_schur_chebyshev_eigenvalue_ratio
            .max(1.0),
    };
    let action = SaddlePointJacobiSchurBackend::try_from_host_csr(
        backend,
        matrix.as_host_csr_ref(),
        schur.as_host_csr_ref(),
        schur_config,
    )
    .unwrap()
    .with_owned_pressure_preconditioner(Box::new(old_amg));
    baseline.preconditioner = Preconditioner::Schur {
        action: Box::new(action),
        pressure: None,
    };
    let restricted =
        PressureAmg::new_restricted(&current.pool, matrix.num_rows, nv, &schur).unwrap();
    let action = SaddlePointJacobiSchurBackend::try_from_host_csr(
        current.pool.backend(),
        matrix.as_host_csr_ref(),
        schur.as_host_csr_ref(),
        schur_config,
    )
    .unwrap()
    .with_owned_pressure_preconditioner(Box::new(restricted.clone()));
    current.preconditioner = Preconditioner::Schur {
        action: Box::new(action),
        pressure: Some(restricted),
    };
    let zero = vec![0.0; matrix.num_rows];
    let mut timings = [Vec::new(), Vec::new()];
    let mut iterations = [0; 2];
    for pass in 0..6 {
        let mut results = Vec::new();
        // Alternate order to reduce ordering and clock-ramp bias.
        for index in if pass % 2 == 0 { [0, 1] } else { [1, 0] } {
            let solver = if index == 0 {
                &mut baseline
            } else {
                &mut *current
            };
            let start = std::time::Instant::now();
            let (x, it, _) = solver
                .solve(&matrix, Some(&schur), config, nv, &rhs, &zero)
                .unwrap();
            let elapsed = start.elapsed().as_secs_f64();
            if pass > 0 {
                assert_eq!(solver.last_allocations, 0);
                timings[index].push(elapsed);
            }
            iterations[index] = it;
            results.push(x);
        }
        let err = results[0]
            .iter()
            .zip(&results[1])
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let norm = results[0].iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(err / norm < 2e-6);
    }
    for values in &mut timings {
        values.sort_by(f64::total_cmp);
    }
    eprintln!(
        "PRESSURE_LAYOUT_COMPARE old_levels={old_levels:?} iterations={iterations:?} median_seconds=[{},{}] speedup={} samples={timings:?}",
        timings[0][2],
        timings[1][2],
        timings[0][2] / timings[1][2]
    );
}
