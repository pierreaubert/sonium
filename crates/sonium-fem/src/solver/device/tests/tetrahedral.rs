//! Public admission and retained solves for assembled tetrahedral pressure systems.

// Rust guideline compliant 2026-02-21
use super::*;
use crate::backend::FemOrder;

#[test]
fn native_generic_tetrahedral_helmholtz_admitted_actions() {
    let Some(pool) = pool() else {
        return;
    };
    let mut mesh = unit_cube_tetrahedra(2);
    mark(&mut mesh);
    for (degree, order) in [
        (PolynomialDegree::P1, FemOrder::P1),
        (PolynomialDegree::P2, FemOrder::P2),
    ] {
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
        let reference = make(Complex64::new(4.0, 0.2)).unwrap().matrix.to_csr();
        for action in [
            FemAction::Forward,
            FemAction::Transpose,
            FemAction::Hermitian,
        ] {
            let session = CudaFemSolveSession::try_assemble(
                &pool,
                FemWorkload::native_cuda_tet(order, action),
                FemDeviceMethod::Fgmres,
                || make(Complex64::new(4.0, 0.2)),
                |_| BackendAmgPreconditioner::<CudaBackend, Complex64>::try_from_csr_with_cuda_coarse_lu(
                    pool.backend(), &auxiliary,
                    AmgConfig { coarse_size: 8, smoother: AmgSmoother::Jacobi, jacobi_weight: 0.6, ..Default::default() },
                ),
            ).unwrap();
            // Shared fixture independently checks the selected action, changed
            // values/RHS, warm allocation, certification and failure recovery.
            check(session, &reference, action);
            eprintln!("generic tetrahedral {degree:?} {action:?} passed");
        }
    }
}

#[test]
fn native_tetrahedral_rejection_precedes_callbacks() {
    use crate::backend::{FemAssemblyMode, FemSpace};
    let Some(pool) = pool() else {
        return;
    };
    let base = FemWorkload::native_cuda_tet(FemOrder::P1, FemAction::Forward);
    for workload in [
        FemWorkload {
            scalar: FemScalar::F64,
            ..base
        },
        FemWorkload {
            order: FemOrder::P2OrHigher,
            ..base
        },
        FemWorkload {
            mode: FemAssemblyMode::MatrixFree,
            ..base
        },
        FemWorkload {
            mode: FemAssemblyMode::PartialAssembly,
            ..base
        },
        FemWorkload {
            space: FemSpace::Hcurl,
            ..base
        },
    ] {
        let before = pool.backend().begin_runtime_stats();
        let result = CudaFemSolveSession::<Complex64, IdentityDevicePreconditioner>::try_assemble(
            &pool,
            workload,
            FemDeviceMethod::Fgmres,
            || panic!("unsupported tetrahedral request assembled"),
            |_| panic!("unsupported tetrahedral request prepared"),
        );
        assert!(matches!(result, Err(FemDeviceSolveError::Admission(_))));
        let stats = pool.backend().runtime_stats_since(before);
        assert_eq!(stats.allocation_operations + stats.kernel_launches, 0);
        assert_eq!(stats.transfer.h2d_bytes + stats.transfer.d2h_bytes, 0);
    }
}
