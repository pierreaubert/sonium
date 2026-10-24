//! Production assembly and independent residual checks for retained FEM solves.

// Rust guideline compliant 2026-02-21
use super::*;
use crate::boundary::generic::{
    try_apply_homogeneous_dirichlet, try_apply_homogeneous_dirichlet_complex,
};
use crate::physics::{
    acoustics::pressure::HelmholtzForm,
    solid::elasticity::{ElasticityForm, IsotropicMaterial},
};
use crate::{
    assembly::assemble_bilinear_checked,
    basis::PolynomialDegree,
    mesh::{Mesh, generators::rectangular_mesh_triangles, unit_cube_tetrahedra},
    spaces::{FunctionSpace, H1Space, VectorH1Space},
};
use sonium_compute::{CudaFailureInjectionPoint, IdentityDevicePreconditioner, MemoryPoolConfig};
use sonium_solvers::{AmgConfig, AmgSmoother, preconditioners::BackendAmgPreconditioner};

mod combination_benchmark;
mod configured;
mod electrostatic;
mod hexahedral;
mod magnetic;
mod public_flexible;
mod stokes;
mod tetrahedral;
mod thermal;

fn pool() -> Option<MemoryPool<CudaBackend>> {
    pool_with_preprocessing(false)
}

fn pool_with_preprocessing(preprocessing: bool) -> Option<MemoryPool<CudaBackend>> {
    match CudaBackend::new(0) {
        Ok(backend) => Some(
            MemoryPool::new(
                backend
                    .with_fused_numeric_combinations(true)
                    .try_with_csr_preprocessing(preprocessing)
                    .unwrap(),
                MemoryPoolConfig::with_cap(1 << 29),
            )
            .unwrap(),
        ),
        Err(error) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                "CUDA required: {error}"
            );
            eprintln!("SKIP generic FEM session: {error}");
            None
        }
    }
}

fn controls() -> BackendKrylovConfig<f64> {
    BackendKrylovConfig {
        max_iterations: 800,
        restart: 60,
        tolerance: 1e-9,
        require_native: true,
        exact_residual_check_interval: 0,
    }
}

fn mark(mesh: &mut Mesh) {
    for face in &mut mesh.boundaries {
        face.marker = if face.nodes.iter().all(|&node| mesh.nodes[node].x == 0.0) {
            1
        } else {
            2
        };
    }
}

fn matvec<T: ComplexField>(a: &CsrMatrix<T>, x: &[T], action: FemAction) -> Vec<T> {
    let mut out = vec![T::zero(); a.num_rows];
    for row in 0..a.num_rows {
        for j in a.row_ptrs[row]..a.row_ptrs[row + 1] {
            let col = a.col_indices[j];
            match action {
                FemAction::Forward => out[row] += a.values[j] * x[col],
                FemAction::Transpose => out[col] += a.values[j] * x[row],
                FemAction::Hermitian => out[col] += a.values[j].conj() * x[row],
            }
        }
    }
    out
}

fn check<T, P>(
    mut session: CudaFemSolveSession<'_, T, P>,
    reference: &CsrMatrix<T>,
    action: FemAction,
) where
    T: DeviceScalar<Real = f64> + ComplexField<Real = f64> + Default,
    P: DevicePreconditioner<CudaBackend, T>,
{
    let b = session.pool.backend();
    let n = reference.num_rows;
    let mut rhs = session.pool.try_alloc::<T>(n).unwrap();
    let mut values = session.pool.try_alloc::<T>(session.columns.len()).unwrap();
    // Build replacement values from the independent forward CSR, respecting the
    // session's documented action ordering rather than reusing its device matrix.
    let (rows, columns) = session.pattern();
    let canonical_values: Vec<_> = (0..n)
        .flat_map(|row| {
            (rows[row]..rows[row + 1]).map(move |j| {
                let column = columns[j];
                let (r, c) = if action == FemAction::Forward {
                    (row, column)
                } else {
                    (column, row)
                };
                let v = (reference.row_ptrs[r]..reference.row_ptrs[r + 1])
                    .find(|&k| reference.col_indices[k] == c)
                    .map(|k| reference.values[k])
                    .unwrap();
                if action == FemAction::Hermitian {
                    ComplexField::conj(&v)
                } else {
                    v
                }
            })
        })
        .collect();
    for pass in 0..3 {
        let scale = 1.0 + 0.1 * pass as f64;
        let exact: Vec<_> = (0..n)
            .map(|i| <T as DeviceScalar>::from_real((0.1 * i as f64 + pass as f64).sin()))
            .collect();
        let host_rhs: Vec<_> = matvec(reference, &exact, action)
            .into_iter()
            .map(|v| v * <T as DeviceScalar>::from_real(scale))
            .collect();
        let host_values: Vec<_> = canonical_values
            .iter()
            .map(|&v| v * <T as DeviceScalar>::from_real(scale))
            .collect();
        b.try_copy_h2d(&host_rhs, rhs.raw_mut()).unwrap();
        b.try_copy_h2d(&host_values, values.raw_mut()).unwrap();
        let before = b.begin_runtime_stats();
        session.try_update_values_lagged(values.raw()).unwrap();
        session.try_update_rhs(rhs.raw()).unwrap();
        let updates = b.runtime_stats_since(before);
        assert_eq!(updates.transfer.h2d_bytes, 0);
        assert_eq!(updates.transfer.d2h_bytes, 8); // Two finite-validation status words.
        if pass > 0 {
            assert_eq!(updates.allocation_operations, 0);
        }
        assert!(session.solution().is_none());
        let before = b.begin_runtime_stats();
        let status = session
            .solve(
                if pass == 0 {
                    DeviceInitialGuess::Zero
                } else {
                    DeviceInitialGuess::PreviousSolution
                },
                &controls(),
            )
            .unwrap();
        let stats = b.runtime_stats_since(before);
        assert!(status.converged, "{status:?}");
        if pass > 0 {
            assert_eq!(stats.allocation_operations, 0, "{stats:?}");
        }
        let mut actual = vec![<T as DeviceScalar>::zero(); n];
        b.try_copy_d2h(session.solution().unwrap(), &mut actual)
            .unwrap();
        let error = actual
            .iter()
            .zip(&exact)
            .map(|(&a, &e)| <T as DeviceScalar>::norm_sqr(&(a - e)))
            .sum::<f64>()
            .sqrt();
        let residual = matvec(reference, &actual, action)
            .into_iter()
            .zip(&host_rhs)
            .map(|(ax, &rhs)| {
                <T as DeviceScalar>::norm_sqr(&(ax * <T as DeviceScalar>::from_real(scale) - rhs))
            })
            .sum::<f64>()
            .sqrt();
        let norm = host_rhs
            .iter()
            .map(|&v| <T as DeviceScalar>::norm_sqr(&v))
            .sum::<f64>()
            .sqrt();
        assert!(error < 2e-6 * (n as f64).sqrt(), "error {error}");
        assert!(residual / norm < 1.1e-9);
        eprintln!(
            "generic FEM {:?} action={action:?} pass={pass} iterations={} allocations={} residual={}",
            session.admission.workload.physics,
            status.iterations,
            stats.allocation_operations,
            residual / norm
        );
    }
    // An interrupted numeric copy cannot leave a publishable or usable operator.
    b.inject_failure(CudaFailureInjectionPoint::TransferD2D);
    assert!(session.try_update_values_lagged(values.raw()).is_err());
    assert!(session.solution().is_none());
    b.recover_after_error().unwrap();
    assert!(
        session
            .solve(DeviceInitialGuess::Zero, &controls())
            .is_err()
    );
    session.try_update_values_lagged(values.raw()).unwrap();
    assert!(
        session
            .solve(DeviceInitialGuess::PreviousSolution, &controls())
            .is_err()
    );
    assert!(
        session
            .solve(DeviceInitialGuess::Zero, &controls())
            .unwrap()
            .converged
    );
    b.inject_failure(CudaFailureInjectionPoint::TransferD2D);
    assert!(session.try_update_rhs(rhs.raw()).is_err());
    assert!(session.solution().is_none());
    b.recover_after_error().unwrap();
    session.try_update_values_lagged(values.raw()).unwrap();
    assert!(
        session
            .solve(DeviceInitialGuess::Zero, &controls())
            .is_err()
    );
    session.try_update_rhs(rhs.raw()).unwrap();
    assert!(
        session
            .solve(DeviceInitialGuess::Zero, &controls())
            .unwrap()
            .converged
    );
    let short = BackendKrylovConfig {
        max_iterations: 1,
        restart: 1,
        tolerance: 1e-14,
        ..controls()
    };
    let stopped = session.solve(DeviceInitialGuess::Zero, &short);
    assert!(stopped.is_err() || !stopped.unwrap().converged);
    assert!(session.solution().is_none());
    let mut bad = controls();
    bad.tolerance = f64::NAN;
    assert!(
        session
            .solve(DeviceInitialGuess::PreviousSolution, &bad)
            .is_err()
    );
    assert!(session.solution().is_none());
}

#[test]
fn native_generic_elasticity_pcg_amg_reuses_system() {
    let Some(pool) = pool() else {
        return;
    };
    let mut mesh = unit_cube_tetrahedra(3);
    mark(&mut mesh);
    let space = VectorH1Space::new(PolynomialDegree::P1, 3);
    let form = ElasticityForm::new_3d(IsotropicMaterial::new(3.0, 0.25));
    for action in [
        FemAction::Forward,
        FemAction::Transpose,
        FemAction::Hermitian,
    ] {
        let assemble = || -> Result<_, FemDeviceSolveError> {
            let matrix = assemble_bilinear_checked(&form, &space, &space, &mesh, 2)?;
            let mut system = FemSystem::new(matrix, vec![0.0; space.num_global_dofs(&mesh)]);
            try_apply_homogeneous_dirichlet(&mut system, &mesh, &space, 1)?;
            Ok(system)
        };
        let reference = assemble().unwrap().matrix.to_csr();
        let session = CudaFemSolveSession::try_assemble(
            &pool,
            FemWorkload::native_cuda_structural(FemScalar::F64, action),
            FemDeviceMethod::Pcg(DevicePcgAssumption::PositiveDefinite),
            assemble,
            |a| {
                BackendAmgPreconditioner::try_from_csr(
                    pool.backend(),
                    a,
                    AmgConfig {
                        coarse_size: 8,
                        ..AmgConfig::for_backend_pcg()
                    },
                )
            },
        )
        .unwrap();
        check(session, &reference, action);
    }
}

#[test]
fn native_generic_helmholtz_fgmres_shifted_amg_reuses_system() {
    let Some(pool) = pool() else {
        return;
    };
    let mut mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 6, 6);
    mark(&mut mesh);
    for degree in [PolynomialDegree::P1, PolynomialDegree::P2] {
        let space = H1Space::new(degree);
        let make = |k| -> Result<_, FemDeviceSolveError> {
            let matrix =
                assemble_bilinear_checked(&HelmholtzForm::new(k), &space, &space, &mesh, 4)?;
            let mut system = FemSystem::new(
                matrix,
                vec![Complex64::default(); space.num_global_dofs(&mesh)],
            );
            try_apply_homogeneous_dirichlet_complex(&mut system, &mesh, &space, 1)?;
            Ok(system)
        };
        let auxiliary = make(Complex64::new(0.0, 4.0)).unwrap().matrix.to_csr();
        for action in [
            FemAction::Forward,
            FemAction::Transpose,
            FemAction::Hermitian,
        ] {
            let reference = make(Complex64::new(4.0, 0.2)).unwrap().matrix.to_csr();
            let workload = if degree == PolynomialDegree::P1 {
                FemWorkload::native_cuda(FemScalar::Complex64, action)
            } else {
                FemWorkload::native_cuda_p2(FemScalar::Complex64, action)
            };
            let session = CudaFemSolveSession::try_assemble(
                &pool,
                workload,
                FemDeviceMethod::Fgmres,
                || make(Complex64::new(4.0, 0.2)),
                |_| {
                    BackendAmgPreconditioner::<CudaBackend, Complex64>::try_from_csr_with_cuda_coarse_lu(
                        pool.backend(),
                        &auxiliary,
                        AmgConfig {
                            coarse_size: 8,
                            smoother: AmgSmoother::Jacobi,
                            jacobi_weight: 0.6,
                            ..Default::default()
                        },
                    )
                },
            )
            .unwrap();
            check(session, &reference, action);
        }
    }
}

#[test]
fn native_generic_admission_precedes_assembly() {
    let Some(pool) = pool() else {
        return;
    };
    let method = FemDeviceMethod::Pcg(DevicePcgAssumption::PositiveDefinite);
    let mut unsupported = FemWorkload::native_cuda(FemScalar::F64, FemAction::Forward);
    unsupported.mode = FemAssemblyMode::PartialAssembly;
    for workload in [
        unsupported,
        FemWorkload::native_cuda(FemScalar::Complex64, FemAction::Forward),
        FemWorkload::native_cuda_stokes(FemAction::Forward),
    ] {
        let before = pool.backend().begin_runtime_stats();
        let result = CudaFemSolveSession::<f64, IdentityDevicePreconditioner>::try_assemble(
            &pool,
            workload,
            method,
            || panic!("rejected workload assembled"),
            |_| panic!("rejected workload prepared"),
        );
        assert!(result.is_err());
        let stats = pool.backend().runtime_stats_since(before);
        assert_eq!(stats.allocation_operations, 0);
        assert_eq!(stats.kernel_launches, 0);
    }
}

#[test]
fn native_generic_nonsymmetric_action_contract() {
    let Some(pool) = pool() else {
        return;
    };
    // Synthetic algebra fixture distinguishes transpose from Hermitian; this
    // does not claim production Maxwell discretization/preconditioner coverage.
    for (action, workload) in [
        FemAction::Forward,
        FemAction::Transpose,
        FemAction::Hermitian,
    ]
    .into_iter()
    .flat_map(|action| {
        [
            FemWorkload::native_cuda_maxwell(FemScalar::Complex64, action),
            FemWorkload::native_cuda(FemScalar::Complex64, action),
            FemWorkload::native_cuda_p2(FemScalar::Complex64, action),
        ]
        .into_iter()
        .map(move |workload| (action, workload))
    }) {
        let mut matrix = crate::assembly::TripletMatrix::new(2, 2);
        matrix.push(0, 0, Complex64::new(3.0, 0.2));
        matrix.push(0, 1, Complex64::new(0.5, -0.7));
        matrix.push(1, 0, Complex64::new(-0.3, 0.1));
        matrix.push(1, 1, Complex64::new(4.0, -0.4));
        let reference = matrix.to_csr();
        let exact = [Complex64::new(0.3, -0.4), Complex64::new(0.5, 0.2)];
        let system = FemSystem::new(matrix, matvec(&reference, &exact, action));
        let mut session = CudaFemSolveSession::try_assemble(
            &pool,
            workload,
            FemDeviceMethod::Fgmres,
            || Ok(system),
            |_| Ok(IdentityDevicePreconditioner),
        )
        .unwrap();
        assert!(
            session
                .solve(DeviceInitialGuess::Zero, &controls())
                .unwrap()
                .converged
        );
        let mut initial = [Complex64::default(); 2];
        pool.backend()
            .try_copy_d2h(session.solution().unwrap(), &mut initial)
            .unwrap();
        for (actual, expected) in initial.iter().zip(exact) {
            assert!((*actual - expected).norm() < 1e-12);
        }
        session.replace_preconditioner(IdentityDevicePreconditioner);
        assert!(session.solution().is_none());
        let reused = session
            .solve(DeviceInitialGuess::PreviousSolution, &controls())
            .unwrap();
        assert!(reused.converged);
        assert_eq!(reused.iterations, 0);
        check(session, &reference, action);
    }
}

#[test]
fn native_generic_malformed_system_rejected_before_upload() {
    let Some(pool) = pool() else {
        return;
    };
    for case in 0..4 {
        let mut matrix = crate::assembly::TripletMatrix::new(2, if case == 0 { 3 } else { 2 });
        matrix.push(0, 0, if case == 1 { f64::NAN } else { 1.0 });
        matrix.push(1, 1, 1.0);
        let mut system = FemSystem::new(matrix, vec![0.0; 2]);
        if case == 2 {
            system.rhs[1] = f64::INFINITY;
        }
        if case == 3 {
            system.dirichlet_dofs.insert(2);
        }
        let before = pool.backend().begin_runtime_stats();
        let result = CudaFemSolveSession::<f64, IdentityDevicePreconditioner>::try_assemble(
            &pool,
            FemWorkload::native_cuda(FemScalar::F64, FemAction::Forward),
            FemDeviceMethod::Fgmres,
            || Ok(system),
            |_| panic!("invalid system prepared"),
        );
        assert!(result.is_err());
        assert_eq!(
            pool.backend()
                .runtime_stats_since(before)
                .allocation_operations,
            0
        );
    }
}
