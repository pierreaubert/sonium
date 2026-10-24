//! Physical heat-conduction and retained-solver checks for scalar H1 systems.

// Rust guideline compliant 2026-02-21
use super::*;
use crate::assembly::assemble_linear_checked;
use crate::backend::{FemElement, FemOrder};
use crate::boundary::generic::try_apply_dirichlet_values;
use crate::physics::thermal::heat_conduction::{ConductionForm, HeatSourceForm};

#[test]
fn native_generic_thermal_pcg_amg_admitted_actions() {
    let Some(pool) = pool() else { return };
    for (element, mut mesh) in [
        (
            FemElement::Tri3,
            rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 6, 6),
        ),
        (FemElement::Tetrahedron, unit_cube_tetrahedra(3)),
    ] {
        // Insulated sides and prescribed temperatures at x=0 and x=1.
        // P1 uses T=1+x; P2 uses T=1+x(1-x) with volumetric heat Q=2k.
        for face in &mut mesh.boundaries {
            face.marker = if face.nodes.iter().all(|&i| mesh.nodes[i].x == 0.0)
                || face.nodes.iter().all(|&i| mesh.nodes[i].x == 1.0)
            {
                1
            } else {
                2
            };
        }
        for (degree, order) in [
            (PolynomialDegree::P1, FemOrder::P1),
            (PolynomialDegree::P2, FemOrder::P2),
        ] {
            let space = H1Space::new(degree);
            let temperature = |x: f64, _: f64, _: f64| {
                if order == FemOrder::P1 {
                    1.0 + x
                } else {
                    1.0 + x * (1.0 - x)
                }
            };
            let assemble = || -> Result<_, FemDeviceSolveError> {
                let matrix = assemble_bilinear_checked(
                    &ConductionForm::from_conductivity(2.1),
                    &space,
                    &space,
                    &mesh,
                    4,
                )?;
                let rhs = assemble_linear_checked(
                    &HeatSourceForm {
                        source_fn: |_, _, _| if order == FemOrder::P1 { 0.0 } else { 4.2 },
                    },
                    &space,
                    &mesh,
                    4,
                )?;
                let mut system = FemSystem::new(matrix, rhs);
                try_apply_dirichlet_values(&mut system, &mesh, &space, 1, &temperature, None)?;
                Ok(system)
            };
            let reference = assemble().unwrap();
            let csr = reference.matrix.to_csr();
            let coordinates = space.dof_coordinates(&mesh).unwrap();
            for action in [
                FemAction::Forward,
                FemAction::Transpose,
                FemAction::Hermitian,
            ] {
                let mut session = CudaFemSolveSession::try_assemble(
                    &pool,
                    FemWorkload::native_cuda_heat_conduction(element, order, action),
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
                let status = session
                    .solve(DeviceInitialGuess::Zero, &controls())
                    .unwrap();
                assert!(status.converged);
                let mut actual = vec![0.0; coordinates.len()];
                pool.backend()
                    .try_copy_d2h(session.solution().unwrap(), &mut actual)
                    .unwrap();
                let error = actual
                    .iter()
                    .zip(&coordinates)
                    .map(|(t, p)| (t - temperature(p[0], p[1], p[2])).abs())
                    .fold(0.0, f64::max);
                assert!(error < 1e-7, "{element:?} {degree:?} {action:?}: {error}");
                let residual = matvec(&csr, &actual, action)
                    .iter()
                    .zip(&reference.rhs)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let norm = reference.rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
                assert!(residual / norm < 1.1e-9);
                eprintln!(
                    "thermal {element:?} {degree:?} {action:?}: temperature error={error}, residual={}",
                    residual / norm
                );
                // Independently generated action RHS and values exercise reuse,
                // publication, zero-allocation warm solves and failure recovery.
                check(session, &csr, action);
            }
        }
    }
}

#[test]
fn native_thermal_rejection_precedes_assembly() {
    use crate::backend::{FemAssemblyMode, FemSpace};
    let Some(pool) = pool() else { return };
    let base = FemWorkload::native_cuda_heat_conduction(
        FemElement::Tri3,
        FemOrder::P1,
        FemAction::Forward,
    );
    for workload in [
        FemWorkload {
            scalar: FemScalar::Complex64,
            ..base
        },
        FemWorkload {
            space: FemSpace::Hcurl,
            ..base
        },
        FemWorkload {
            order: FemOrder::P3,
            ..base
        },
        FemWorkload {
            order: FemOrder::P2OrHigher,
            ..base
        },
        FemWorkload {
            element: FemElement::Hexahedron,
            ..base
        },
        FemWorkload {
            mode: FemAssemblyMode::PartialAssembly,
            ..base
        },
        FemWorkload {
            mode: FemAssemblyMode::MatrixFree,
            ..base
        },
    ] {
        let before = pool.backend().begin_runtime_stats();
        let result = CudaFemSolveSession::<f64, IdentityDevicePreconditioner>::try_assemble(
            &pool,
            workload,
            FemDeviceMethod::Fgmres,
            || panic!("unsupported thermal request assembled"),
            |_| panic!("unsupported thermal request prepared"),
        );
        assert!(matches!(result, Err(FemDeviceSolveError::Admission(_))));
        let stats = pool.backend().runtime_stats_since(before);
        assert_eq!(stats.allocation_operations + stats.kernel_launches, 0);
        assert_eq!(stats.transfer.h2d_bytes + stats.transfer.d2h_bytes, 0);
    }
}
