//! Assembled high-order hex solves through the shared admission contract.

// Rust guideline compliant 2026-02-21
use super::*;
use crate::backend::{FemAssemblyMode, FemOrder, FemSpace};
use crate::mesh::unit_cube_hexahedra;
use crate::physics::acoustics::pressure::device_hex::prepare_hex_lor_shifted_laplacian;

#[test]
fn native_generic_hexahedral_helmholtz_lor_admitted_actions() {
    let Some(pool) = pool() else {
        return;
    };
    let mut mesh = unit_cube_hexahedra(2);
    mark(&mut mesh);
    // Preserve boundary markers while making the shared-node cells nonorthogonal.
    for point in &mut mesh.nodes {
        point.x += 0.12 * point.y;
        point.z += 0.08 * point.x;
    }
    for (degree, order) in [
        (PolynomialDegree::P1, FemOrder::P1),
        (PolynomialDegree::P2, FemOrder::P2),
        (PolynomialDegree::P3, FemOrder::P3),
    ] {
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
        let auxiliary = prepare_hex_lor_shifted_laplacian(
            &mesh,
            &space,
            Complex64::new(16.0, 0.0),
            &space.boundary_dofs(&mesh, 1),
        )
        .unwrap();
        assert_eq!(auxiliary.num_rows, reference.num_rows);
        for action in [
            FemAction::Forward,
            FemAction::Transpose,
            FemAction::Hermitian,
        ] {
            let session = CudaFemSolveSession::try_assemble(
                &pool,
                FemWorkload::native_cuda_hex(order, action),
                FemDeviceMethod::Fgmres,
                make,
                |_| {
                    BackendAmgPreconditioner::<CudaBackend, Complex64>::try_from_csr_with_cuda_coarse_lu(
                        pool.backend(),
                        &auxiliary,
                        AmgConfig {
                            coarse_size: 8,
                            smoother: AmgSmoother::Jacobi,
                            jacobi_weight: 0.6,
                            ..AmgConfig::for_aggregation()
                        },
                    )
                },
            )
            .unwrap();
            check(session, &reference, action);
            eprintln!("generic hexahedral {degree:?} {action:?} passed");
        }
    }
}

#[test]
fn native_hexahedral_rejection_precedes_callbacks() {
    let Some(pool) = pool() else {
        return;
    };
    let base = FemWorkload::native_cuda_hex(FemOrder::P3, FemAction::Forward);
    for workload in [
        FemWorkload {
            scalar: FemScalar::F64,
            ..base
        },
        FemWorkload {
            scalar: FemScalar::Complex32,
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
            || panic!("unsupported hexahedral request assembled"),
            |_| panic!("unsupported hexahedral request prepared"),
        );
        assert!(matches!(result, Err(FemDeviceSolveError::Admission(_))));
        let stats = pool.backend().runtime_stats_since(before);
        assert_eq!(stats.allocation_operations + stats.kernel_launches, 0);
        assert_eq!(stats.transfer.h2d_bytes + stats.transfer.d2h_bytes, 0);
    }
}
