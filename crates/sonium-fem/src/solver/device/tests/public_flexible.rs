//! Assembled pressure sweeps exercise the public retained flexible solver.

// Rust guideline compliant 2026-02-21
use super::*;
use sonium_solvers::CudaLinearSolveSession;

#[test]
fn native_public_flexible_helmholtz_sweep_p1_p2() {
    let Some(pool) = pool() else {
        return;
    };
    let mut mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 6, 6);
    mark(&mut mesh);
    let amg_config = AmgConfig {
        coarse_size: 8,
        smoother: AmgSmoother::Jacobi,
        jacobi_weight: 0.6,
        ..Default::default()
    };
    for degree in [PolynomialDegree::P1, PolynomialDegree::P2] {
        let space = H1Space::new(degree);
        let assemble = |k| {
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
        let initial = assemble(Complex64::new(4.0, 0.2));
        let initial_matrix = initial.matrix.to_csr();
        let auxiliary = assemble(Complex64::new(0.0, 4.0)).matrix.to_csr();
        let mut session = CudaLinearSolveSession::try_new_flexible(&pool, &initial_matrix, &controls(), |backend, physical| {
            assert_eq!(physical.num_rows, auxiliary.num_rows);
            BackendAmgPreconditioner::<CudaBackend, Complex64>::try_from_csr_with_cuda_coarse_lu(backend, &auxiliary, amg_config.clone())
        }).unwrap();
        let exact: Vec<_> = (0..initial_matrix.num_rows)
            .map(|i| {
                if initial.dirichlet_dofs.contains(&i) {
                    Complex64::default()
                } else {
                    Complex64::new((i as f64 * 0.2).sin(), (i as f64 * 0.1).cos())
                }
            })
            .collect();
        for (index, k) in [4.0, 4.2, 3.8].into_iter().enumerate() {
            let matrix = assemble(Complex64::new(k, 0.2)).matrix.to_csr();
            if index == 1 {
                let before = pool.backend().begin_runtime_stats();
                session.try_update_matrix_lagged(&matrix).unwrap();
                let stats = pool.backend().runtime_stats_since(before);
                assert_eq!(stats.allocation_operations, 0);
                assert_eq!(stats.transfer.d2h_bytes, 0);
                assert!(session.solution_device().is_none());
            } else if index == 2 {
                let auxiliary = assemble(Complex64::new(0.0, 4.5)).matrix.to_csr();
                session.try_update_matrix_with_preconditioner(&matrix, |backend, _| {
                    BackendAmgPreconditioner::<CudaBackend, Complex64>::try_from_csr_with_cuda_coarse_lu(backend, &auxiliary, amg_config.clone())
                }).unwrap();
            }
            // Independent host CSR action supplies a manufactured discrete load
            // for the assembled volume and constrained boundary problem.
            let rhs = matvec(&matrix, &exact, FemAction::Forward);
            session
                .solve_host(
                    &rhs,
                    if index == 0 {
                        DeviceInitialGuess::Zero
                    } else {
                        DeviceInitialGuess::PreviousSolution
                    },
                )
                .unwrap();
            let before = pool.backend().begin_runtime_stats();
            let result = session.solve_host(&rhs, DeviceInitialGuess::Zero).unwrap();
            let stats = pool.backend().runtime_stats_since(before);
            assert_eq!(stats.allocation_operations, 0);
            assert!(result.residual < 1e-9 && !result.backend.fallback);
            let error = result
                .x
                .iter()
                .zip(&exact)
                .map(|(a, b)| (*a - *b).norm_sqr())
                .sum::<f64>()
                .sqrt()
                / exact.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
            assert!(error < 1e-7, "{degree:?} k={k} error={error}");
            eprintln!(
                "Public FEM flexible degree={degree:?} k={k} iterations={} residual={} error={error} warm_allocations={}",
                result.iterations, result.residual, stats.allocation_operations
            );
        }
    }
}

#[test]
fn native_resident_helmholtz_combinations_and_adjoints() {
    resident_helmholtz_combinations_and_adjoints(false, false);
}

#[test]
fn native_preprocessed_helmholtz_combinations_and_adjoints() {
    resident_helmholtz_combinations_and_adjoints(true, false);
}

#[test]
fn native_fused_jacobi_helmholtz_combinations_and_adjoints() {
    resident_helmholtz_combinations_and_adjoints(true, true);
}

fn resident_helmholtz_combinations_and_adjoints(preprocessing: bool, fused_jacobi: bool) {
    let Some(pool) = pool_with_preprocessing(preprocessing) else {
        return;
    };
    let backend = pool.backend();
    let mut mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 6, 6);
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
        // K includes constraint identity, while M = K - A(1) has zero constrained rows.
        let stiffness = make(Complex64::new(0.0, 0.0)).matrix.to_csr();
        let at_one = make(Complex64::new(1.0, 0.0)).matrix.to_csr();
        assert_eq!(stiffness.row_ptrs, at_one.row_ptrs);
        assert_eq!(stiffness.col_indices, at_one.col_indices);
        let mut mass = stiffness.clone();
        for (m, a) in mass.values.iter_mut().zip(&at_one.values) {
            *m -= *a;
        }
        let auxiliary = make(Complex64::new(0.0, 4.0)).matrix.to_csr();
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
        for action in [
            FemAction::Forward,
            FemAction::Transpose,
            FemAction::Hermitian,
        ] {
            let orient = |a: &CsrMatrix<Complex64>| {
                let mut entries = Vec::with_capacity(a.values.len());
                for r in 0..a.num_rows {
                    for j in a.row_ptrs[r]..a.row_ptrs[r + 1] {
                        let c = a.col_indices[j];
                        let v = a.values[j];
                        entries.push(match action {
                            FemAction::Forward => (r, c, v),
                            FemAction::Transpose => (c, r, v),
                            FemAction::Hermitian => (c, r, v.conj()),
                        });
                    }
                }
                CsrMatrix::from_triplets(a.num_rows, a.num_cols, entries)
            };
            let k_action = orient(&stiffness);
            let m_action = orient(&mass);
            let auxiliary_action = orient(&auxiliary);
            let workload = if degree == PolynomialDegree::P1 {
                FemWorkload::native_cuda(FemScalar::Complex64, action)
            } else {
                FemWorkload::native_cuda_p2(FemScalar::Complex64, action)
            };
            let mut session = CudaFemSolveSession::try_assemble(
                &pool, workload, FemDeviceMethod::Fgmres,
                || Ok(make(Complex64::new(4.0, 0.2))),
                |_| BackendAmgPreconditioner::<CudaBackend, Complex64>::try_from_csr_with_cuda_coarse_lu(
                    backend, &auxiliary_action, AmgConfig { coarse_size: 8, smoother: AmgSmoother::Jacobi, jacobi_weight: 0.6, ..Default::default() },
                ).map(|mut preconditioner| {
                    preconditioner.set_fused_jacobi_enabled(fused_jacobi);
                    preconditioner
                }),
            ).unwrap();
            assert_eq!(
                session.pattern(),
                (
                    k_action.row_ptrs.as_slice(),
                    k_action.col_indices.as_slice()
                )
            );
            assert_eq!(k_action.row_ptrs, m_action.row_ptrs);
            assert_eq!(k_action.col_indices, m_action.col_indices);
            let upload = |values: &[Complex64]| {
                let mut buffer = pool.try_alloc::<Complex64>(values.len()).unwrap();
                backend.try_copy_h2d(values, buffer.raw_mut()).unwrap();
                buffer
            };
            let k_values = upload(&k_action.values);
            let m_values = upload(&m_action.values);
            let k_rhs = upload(&matvec(&stiffness, &exact, action));
            let m_rhs = upload(&matvec(&mass, &exact, action));
            for (index, real_k) in [4.0, 4.2, 3.8].into_iter().enumerate() {
                let k = Complex64::new(real_k, 0.2);
                let coefficient = match action {
                    FemAction::Hermitian => -(k * k).conj(),
                    _ => -(k * k),
                };
                let before = backend.begin_runtime_stats();
                session
                    .try_combine_system_lagged(
                        (Complex64::new(1.0, 0.0), k_values.raw()),
                        &[(coefficient, m_values.raw())],
                        (Complex64::new(1.0, 0.0), k_rhs.raw()),
                        &[(coefficient, m_rhs.raw())],
                    )
                    .unwrap();
                assert!(session.solution().is_none());
                let update_stats = backend.runtime_stats_since(before);
                assert_eq!(update_stats.transfer.h2d_bytes, 0, "{update_stats:?}");
                assert_eq!(update_stats.transfer.d2d_bytes, 0, "{update_stats:?}");
                assert!(update_stats.transfer.d2h_bytes <= 32, "{update_stats:?}");
                // A zero start forces iterations at every frequency; it cannot
                // conceal a bad update behind the unchanged manufactured solution.
                let status = session
                    .solve(DeviceInitialGuess::Zero, &controls())
                    .unwrap();
                assert!(
                    status.converged && status.iterations > 0,
                    "{degree:?}/{action:?}: {status:?}"
                );
                let stats = backend.runtime_stats_since(before);
                // FGMRES retains bulk vectors but exchanges compact Krylov
                // control coefficients; measure these separately from updates.
                eprintln!(
                    "resident combination {degree:?}/{action:?} k={k}: iterations={} h2d={} d2h={}",
                    status.iterations, stats.transfer.h2d_bytes, stats.transfer.d2h_bytes
                );
                if index > 0 {
                    assert_eq!(stats.allocation_operations, 0);
                }
                let mut solution = vec![Complex64::default(); exact.len()];
                backend
                    .try_copy_d2h(session.solution().unwrap(), &mut solution)
                    .unwrap();
                // Independent assembly is used only after the resident solve.
                let reference = make(k).matrix.to_csr();
                let rhs = matvec(&reference, &exact, action);
                let residual = matvec(&reference, &solution, action)
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
                    residual < 1.1e-9,
                    "{degree:?}/{action:?} k={k}: residual {residual}"
                );
                assert!(error < 1e-7, "{degree:?}/{action:?} k={k}: error {error}");
            }
        }
    }
}

#[test]
fn native_standard_gmres_helmholtz_ilu_and_configured_schwarz() {
    use sonium_solvers::preconditioners::{
        AdditiveSchwarzPreconditioner, CudaIluPreconditioner, CudaSchwarzPreconditioner,
    };

    let Some(pool) = pool() else { return };
    let mut mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 6, 6);
    mark(&mut mesh);
    for degree in [PolynomialDegree::P1, PolynomialDegree::P2] {
        let space = H1Space::new(degree);
        let make = || -> Result<_, FemDeviceSolveError> {
            let matrix = assemble_bilinear_checked(
                &HelmholtzForm::new(Complex64::new(4.0, 0.2)),
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
            let ilu = CudaFemSolveSession::try_assemble(
                &pool,
                workload,
                FemDeviceMethod::Gmres,
                make,
                |matrix| CudaIluPreconditioner::try_from_csr(pool.backend(), matrix),
            )
            .unwrap();
            check(ilu, &reference, action);
            // Preserve FEM's default eight subdomains and two overlap layers,
            // rather than using the facade's automatic partition/overlap policy.
            let schwarz = CudaFemSolveSession::try_assemble(
                &pool,
                workload,
                FemDeviceMethod::Gmres,
                make,
                |matrix| {
                    let host = AdditiveSchwarzPreconditioner::try_from_csr(matrix, 8, 2).unwrap();
                    CudaSchwarzPreconditioner::try_from_host(pool.backend(), &host)
                },
            )
            .unwrap();
            check(schwarz, &reference, action);
        }
    }
}
