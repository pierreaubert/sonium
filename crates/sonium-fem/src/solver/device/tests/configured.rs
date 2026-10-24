//! Configuration-preserving public FEM CUDA preparation and reuse.

// Rust guideline compliant 2026-02-21
use super::*;
use crate::solver::{SolverConfig, SolverType};
use sonium_solvers::{AdditiveSchwarzPreconditioner, Preconditioner};

#[test]
fn native_configured_fem_strategies_and_schwarz_settings() {
    let Some(pool) = pool() else { return };
    let backend = pool.backend();
    let mut mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 4, 4);
    mark(&mut mesh);
    for degree in [PolynomialDegree::P1, PolynomialDegree::P2] {
        let space = H1Space::new(degree);
        let make = || -> Result<_, FemDeviceSolveError> {
            // Coercive complex Helmholtz supports all five requested strategies.
            let matrix = assemble_bilinear_checked(
                &HelmholtzForm::new(Complex64::new(1.0, 4.0)),
                &space,
                &space,
                &mesh,
                4,
            )?;
            let mut system = FemSystem::new(
                matrix,
                vec![Complex64::default(); space.num_global_dofs(&mesh)],
            );
            try_apply_homogeneous_dirichlet_complex(&mut system, &mesh, &space, 1)?;
            Ok(system)
        };
        let reference = make().unwrap().matrix.to_csr();
        let workload = if degree == PolynomialDegree::P1 {
            FemWorkload::native_cuda(FemScalar::Complex64, FemAction::Forward)
        } else {
            FemWorkload::native_cuda_p2(FemScalar::Complex64, FemAction::Forward)
        };
        for (solver_type, domains, overlap) in [
            (SolverType::Gmres, 8, 2),
            (SolverType::GmresJacobi, 8, 2),
            (SolverType::GmresIlu, 8, 2),
            (SolverType::GmresAmg, 8, 2),
            (SolverType::GmresSchwarz, 3, 0),
            (SolverType::GmresSchwarz, 8, 2),
        ] {
            let mut config = SolverConfig {
                solver_type,
                schwarz_subdomains: domains,
                schwarz_overlap: overlap,
                ..Default::default()
            };
            config.gmres.restart = 60;
            config.gmres.tolerance = 1e-9;
            config.gmres.exact_residual_check_interval = 7;
            let mut session =
                CudaConfiguredFemSolveSession::try_assemble(&pool, workload, &config, make)
                    .unwrap();
            assert_eq!(session.config().solver_type, solver_type);
            let n = reference.num_rows;
            let mut rhs = pool.try_alloc::<Complex64>(n).unwrap();
            let mut output = pool.try_alloc::<Complex64>(n).unwrap();
            if solver_type == SolverType::GmresSchwarz {
                let host =
                    AdditiveSchwarzPreconditioner::try_from_csr(&reference, domains, overlap)
                        .unwrap();
                let input: Vec<_> = (0..n)
                    .map(|i| Complex64::new((i as f64 * 0.2).sin(), 0.2))
                    .collect();
                let expected = Preconditioner::apply(&host, &input);
                backend.try_copy_h2d(&input, rhs.raw_mut()).unwrap();
                let before = backend.begin_runtime_stats();
                session
                    .session()
                    .preconditioner
                    .try_apply(backend, rhs.raw(), output.raw_mut())
                    .unwrap();
                let stats = backend.runtime_stats_since(before);
                assert_eq!(stats.allocation_operations, 0);
                assert_eq!(stats.transfer.h2d_bytes, 0);
                assert_eq!(stats.transfer.d2h_bytes, 0);
                let mut actual = vec![Complex64::default(); n];
                backend.try_copy_d2h(output.raw(), &mut actual).unwrap();
                for (a, e) in actual.iter().zip(expected) {
                    assert!((*a - e).norm() < 1e-10 * (1.0 + e.norm()));
                }
            }
            for pass in 0..3 {
                let exact: Vec<_> = (0..n)
                    .map(|i| Complex64::new((i as f64 * 0.1 + pass as f64).sin(), 0.1))
                    .collect();
                let load = matvec(&reference, &exact, FemAction::Forward);
                backend.try_copy_h2d(&load, rhs.raw_mut()).unwrap();
                session.session_mut().try_update_rhs(rhs.raw()).unwrap();
                let before = backend.begin_runtime_stats();
                let status = session
                    .solve(if pass == 0 {
                        DeviceInitialGuess::Zero
                    } else {
                        DeviceInitialGuess::PreviousSolution
                    })
                    .unwrap();
                assert!(status.converged, "{solver_type:?}: {status:?}");
                if pass > 0 {
                    assert_eq!(backend.runtime_stats_since(before).allocation_operations, 0);
                }
                let mut actual = vec![Complex64::default(); n];
                backend
                    .try_copy_d2h(session.session().solution().unwrap(), &mut actual)
                    .unwrap();
                let residual = matvec(&reference, &actual, FemAction::Forward)
                    .iter()
                    .zip(&load)
                    .map(|(a, b)| (*a - *b).norm_sqr())
                    .sum::<f64>()
                    .sqrt();
                let norm = load.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
                assert!(residual / norm < 1.1e-9);
                assert!(
                    actual
                        .iter()
                        .zip(&exact)
                        .all(|(a, e)| (*a - *e).norm() < 1e-7)
                );
                if pass == 1 {
                    let field = session.session().solution().unwrap() as *const _;
                    for malformed in [0, 1, 2] {
                        let mut invalid = reference.clone();
                        match malformed {
                            0 => invalid.num_cols += 1,
                            1 => {
                                invalid.values.pop();
                            }
                            _ => invalid.values[0] = Complex64::new(f64::NAN, 0.0),
                        }
                        let before = backend.begin_runtime_stats();
                        assert!(session.try_prepare_preconditioner(&invalid).is_err());
                        let stats = backend.runtime_stats_since(before);
                        assert_eq!(stats.allocation_operations, 0);
                        assert_eq!(stats.transfer.h2d_bytes, 0);
                        assert_eq!(session.session().solution().unwrap() as *const _, field);
                    }
                    session.try_refresh_preconditioner(&reference).unwrap();
                    assert!(session.session().solution().is_none());
                    assert!(
                        session
                            .solve(DeviceInitialGuess::PreviousSolution)
                            .unwrap()
                            .converged
                    );
                    assert_eq!(session.session().solution().unwrap() as *const _, field);
                }
            }
        }
    }
}

#[test]
fn native_configured_fem_rejection_precedes_assembly() {
    let Some(pool) = pool() else { return };
    let mut configs = Vec::new();
    for solver_type in [
        SolverType::Direct,
        SolverType::GmresPipelined,
        SolverType::GmresIluFixedPoint,
        SolverType::GmresIluColoring,
        SolverType::GmresShiftedLaplacian,
    ] {
        configs.push(SolverConfig {
            solver_type,
            ..Default::default()
        });
    }
    let mut adaptive = SolverConfig::default();
    adaptive.gmres.min_restart = Some(10);
    configs.push(adaptive);
    let mut invalid = SolverConfig::default();
    invalid.gmres.tolerance = f64::NAN;
    configs.push(invalid);
    configs.push(SolverConfig {
        solver_type: SolverType::GmresSchwarz,
        schwarz_subdomains: 0,
        ..Default::default()
    });
    for config in configs {
        let before = pool.backend().begin_runtime_stats();
        let result = CudaConfiguredFemSolveSession::try_assemble(
            &pool,
            FemWorkload::native_cuda(FemScalar::Complex64, FemAction::Forward),
            &config,
            || panic!("unsupported configuration must reject before assembly"),
        );
        assert!(result.is_err());
        let stats = pool.backend().runtime_stats_since(before);
        assert_eq!(stats.allocation_operations, 0);
        assert_eq!(stats.transfer.h2d_bytes, 0);
    }
}

#[test]
fn native_configured_fem_csr_actions_and_reuse() {
    let Some(pool) = pool() else { return };
    let backend = pool.backend();
    let mut mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 4, 4);
    mark(&mut mesh);
    for degree in [PolynomialDegree::P1, PolynomialDegree::P2] {
        let space = H1Space::new(degree);
        let matrix = assemble_bilinear_checked(
            &HelmholtzForm::new(Complex64::new(1.0, 4.0)),
            &space,
            &space,
            &mesh,
            4,
        )
        .unwrap();
        let mut system = FemSystem::new(
            matrix,
            vec![Complex64::default(); space.num_global_dofs(&mesh)],
        );
        try_apply_homogeneous_dirichlet_complex(&mut system, &mesh, &space, 1).unwrap();
        // A nonsymmetric perturbation distinguishes transpose from forward;
        // complex Helmholtz values distinguish Hermitian from transpose.
        system.matrix.push(6, 7, Complex64::new(0.03, 0.02));
        let matrix = system.matrix.to_csr();
        let n = matrix.num_rows;
        for action in [
            FemAction::Forward,
            FemAction::Transpose,
            FemAction::Hermitian,
        ] {
            let workload = if degree == PolynomialDegree::P1 {
                FemWorkload::native_cuda(FemScalar::Complex64, action)
            } else {
                FemWorkload::native_cuda_p2(FemScalar::Complex64, action)
            };
            for solver_type in [SolverType::GmresIlu, SolverType::GmresSchwarz] {
                let mut config = SolverConfig {
                    solver_type,
                    schwarz_subdomains: 3,
                    schwarz_overlap: 1,
                    ..Default::default()
                };
                config.gmres.restart = 60;
                config.gmres.tolerance = 1e-9;
                let exact: Vec<_> = (0..n)
                    .map(|i| Complex64::new((0.2 * i as f64).sin(), 0.1))
                    .collect();
                let load = matvec(&matrix, &exact, action);
                let mut csr = CudaConfiguredFemSolveSession::try_from_csr(
                    &pool, workload, &config, &matrix, &load,
                )
                .unwrap();
                let mut assembled =
                    CudaConfiguredFemSolveSession::try_assemble(&pool, workload, &config, || {
                        let mut value = FemSystem::new(system.matrix.clone(), load.clone());
                        value.dirichlet_dofs = system.dirichlet_dofs.clone();
                        Ok(value)
                    })
                    .unwrap();
                assert_eq!(csr.session().pattern(), assembled.session().pattern());
                assert!(assembled.solve(DeviceInitialGuess::Zero).unwrap().converged);
                let mut expected = vec![Complex64::default(); n];
                backend
                    .try_copy_d2h(assembled.session().solution().unwrap(), &mut expected)
                    .unwrap();
                let mut rhs = pool.try_alloc(n).unwrap();
                for pass in 0..3 {
                    let factor = Complex64::new(1.0 + 0.1 * pass as f64, 0.05 * pass as f64);
                    let load: Vec<_> = load.iter().map(|v| *v * factor).collect();
                    backend.try_copy_h2d(&load, rhs.raw_mut()).unwrap();
                    csr.session_mut().try_update_rhs(rhs.raw()).unwrap();
                    let before = backend.begin_runtime_stats();
                    let status = csr
                        .solve(if pass == 0 {
                            DeviceInitialGuess::Zero
                        } else {
                            DeviceInitialGuess::PreviousSolution
                        })
                        .unwrap();
                    assert!(
                        status.converged,
                        "{degree:?} {action:?} {solver_type:?}: {status:?}"
                    );
                    if pass > 0 {
                        assert_eq!(backend.runtime_stats_since(before).allocation_operations, 0);
                    }
                    let mut actual = vec![Complex64::default(); n];
                    backend
                        .try_copy_d2h(csr.session().solution().unwrap(), &mut actual)
                        .unwrap();
                    for ((actual, exact), reference) in actual.iter().zip(&exact).zip(&expected) {
                        assert!((*actual - *exact * factor).norm() < 1e-7);
                        assert!((*actual - *reference * factor).norm() < 1e-7);
                    }
                    let residual = matvec(&matrix, &actual, action)
                        .iter()
                        .zip(&load)
                        .map(|(a, b)| (*a - *b).norm_sqr())
                        .sum::<f64>()
                        .sqrt();
                    let norm = load.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
                    assert!(residual / norm < 1.1e-9);
                }
                backend.inject_failure(CudaFailureInjectionPoint::TransferD2D);
                assert!(csr.session_mut().try_update_rhs(rhs.raw()).is_err());
                assert!(csr.session().solution().is_none());
                backend.recover_after_error().unwrap();
                assert!(csr.solve(DeviceInitialGuess::Zero).is_err());
                csr.session_mut().try_update_rhs(rhs.raw()).unwrap();
                assert!(csr.solve(DeviceInitialGuess::Zero).unwrap().converged);
            }
        }
    }
}

#[test]
fn native_fem_csr_rejects_malformed_input_before_setup() {
    let Some(pool) = pool() else { return };
    let backend = pool.backend();
    let valid = CsrMatrix::from_triplets(
        2,
        2,
        vec![
            (0, 0, Complex64::new(2.0, 0.0)),
            (0, 1, Complex64::new(0.1, 0.2)),
            (1, 0, Complex64::new(0.3, 0.1)),
            (1, 1, Complex64::new(3.0, 0.0)),
        ],
    );
    for case in 0..12 {
        let mut matrix = valid.clone();
        let mut rhs = vec![Complex64::new(1.0, 0.0); 2];
        match case {
            0 => matrix.row_ptrs.clear(),
            1 => matrix.row_ptrs[0] = 1,
            2 => matrix.row_ptrs[1] = 5,
            3 => matrix.col_indices[0] = 2,
            4 => {
                matrix.values.pop();
            }
            5 => matrix.col_indices.swap(0, 1),
            6 => matrix.col_indices[1] = matrix.col_indices[0],
            7 => matrix.values[0] = Complex64::new(f64::NAN, 0.0),
            8 => {
                rhs.pop();
            }
            9 => rhs[0] = Complex64::new(0.0, f64::INFINITY),
            10 => matrix.num_rows = usize::MAX,
            _ => matrix.num_cols = 3,
        }
        for action in [
            FemAction::Forward,
            FemAction::Transpose,
            FemAction::Hermitian,
        ] {
            let before = backend.begin_runtime_stats();
            let result = CudaFemSolveSession::<
                Complex64,
                sonium_compute::IdentityDevicePreconditioner,
            >::try_from_csr(
                &pool,
                FemWorkload::native_cuda(FemScalar::Complex64, action),
                FemDeviceMethod::Gmres,
                &matrix,
                &rhs,
                |_| panic!("invalid input must reject before preconditioner setup"),
            );
            assert!(result.is_err(), "case {case}");
            let stats = backend.runtime_stats_since(before);
            assert_eq!(stats.allocation_operations, 0);
            assert_eq!(stats.transfer.h2d_bytes, 0);
        }
    }
}
