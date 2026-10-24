//! Layered scalar magnetostatics through retained CUDA PCG and symmetric AMG.

// Rust guideline compliant 2026-02-21
use super::*;
use crate::assembly::assemble_linear_checked;
use crate::backend::{FemAssemblyMode, FemElement, FemOrder, FemSpace};
use crate::boundary::generic::try_apply_dirichlet_values;
use crate::forms::{BilinearForm, LinearForm, LocalMatrix, LocalVector};
use crate::physics::electromagnetics::magnetostatics::{
    CudaMagneticCentroidField, MagnetSourceForm, PermeabilityForm, compute_b_field,
};
use crate::spaces::BasisTabulation;

// Permeability uses common reference units; potential is scaled by the same
// reference permeability, so recovered flux retains the remanence units.
const MU: [f64; 2] = [2.0, 5.0];
const REMANENCE: [f64; 2] = [0.4, 1.2];
const END_POTENTIAL: f64 = 0.1;
fn region(x: f64) -> usize {
    usize::from(x >= 0.5)
}

struct LayeredPermeability;
impl BilinearForm<f64> for LayeredPermeability {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        matrix: &mut LocalMatrix<f64>,
    ) {
        PermeabilityForm::from_mu(MU[region(test.physical_coords[0])])
            .integrate_point(test, trial, weight, matrix);
    }
}
struct LayeredRemanence(f64);
impl LinearForm<f64> for LayeredRemanence {
    fn integrate_point(&self, test: &BasisTabulation, weight: f64, rhs: &mut LocalVector<f64>) {
        MagnetSourceForm {
            br: [
                self.0 * REMANENCE[region(test.physical_coords[0])],
                0.0,
                0.0,
            ],
        }
        .integrate_point(test, weight, rhs);
    }
}
fn exact_flux(amplitude: f64) -> f64 {
    (0.5 * amplitude * (REMANENCE[0] / MU[0] + REMANENCE[1] / MU[1]) - END_POTENTIAL)
        / (0.5 / MU[0] + 0.5 / MU[1])
}
fn exact_potential(x: f64, amplitude: f64) -> f64 {
    let b = exact_flux(amplitude);
    x.min(0.5) * (amplitude * REMANENCE[0] - b) / MU[0]
        + (x - 0.5).max(0.0) * (amplitude * REMANENCE[1] - b) / MU[1]
}

#[test]
fn native_magnetic_scalar_layered_potential_flux_and_reuse() {
    let Some(pool) = pool() else { return };
    for (element, mut mesh) in [
        (
            FemElement::Tri3,
            rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 6, 6),
        ),
        (FemElement::Tetrahedron, unit_cube_tetrahedra(4)),
    ] {
        // Both meshes align with the material interface x=1/2. Prescribe the
        // end potentials; natural conditions on all side walls impose B.n=0.
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
            let assemble = |amplitude| -> Result<_, FemDeviceSolveError> {
                let matrix =
                    assemble_bilinear_checked(&LayeredPermeability, &space, &space, &mesh, 4)?;
                let rhs = assemble_linear_checked(&LayeredRemanence(amplitude), &space, &mesh, 4)?;
                let mut system = FemSystem::new(matrix, rhs);
                try_apply_dirichlet_values(
                    &mut system,
                    &mesh,
                    &space,
                    1,
                    &|x, _, _| END_POTENTIAL * x,
                    None,
                )?;
                Ok(system)
            };
            let reference = assemble(1.0).unwrap();
            let csr = reference.matrix.to_csr();
            let coordinates = space.dof_coordinates(&mesh).unwrap();
            for action in [
                FemAction::Forward,
                FemAction::Transpose,
                FemAction::Hermitian,
            ] {
                let mut session = CudaFemSolveSession::try_assemble(
                    &pool,
                    FemWorkload::native_cuda_magnetic_scalar_potential(element, order, action),
                    FemDeviceMethod::Pcg(DevicePcgAssumption::PositiveDefinite),
                    || assemble(1.0),
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
                let dimension = if element == FemElement::Tri3 { 2 } else { 3 };
                let regions: Vec<_> = mesh
                    .elements
                    .iter()
                    .map(|e| {
                        region(
                            e.vertices().iter().map(|&i| mesh.nodes[i].x).sum::<f64>()
                                / e.vertices().len() as f64,
                        )
                    })
                    .collect();
                let permeability: Vec<_> = regions.iter().map(|&r| MU[r]).collect();
                let remanence: Vec<_> = regions.iter().map(|&r| [REMANENCE[r], 0.0, 0.0]).collect();
                let mut fields = CudaMagneticCentroidField::try_new(
                    &pool,
                    &mesh,
                    &space,
                    &permeability,
                    &remanence,
                )
                .unwrap();
                assert_eq!(fields.spatial_dimension(), dimension);
                let mut remanence_device = pool.try_alloc(mesh.elements.len() * dimension).unwrap();
                let mut rhs_device = pool.try_alloc(coordinates.len()).unwrap();
                for (i, amplitude) in [1.0, 1.5, 0.8].into_iter().enumerate() {
                    let physical = assemble(amplitude).unwrap();
                    pool.backend()
                        .try_copy_h2d(&physical.rhs, rhs_device.raw_mut())
                        .unwrap();
                    let before = pool.backend().begin_runtime_stats();
                    session.try_update_rhs(rhs_device.raw()).unwrap();
                    let status = session
                        .solve(
                            if i == 0 {
                                DeviceInitialGuess::Zero
                            } else {
                                DeviceInitialGuess::PreviousSolution
                            },
                            &controls(),
                        )
                        .unwrap();
                    let stats = pool.backend().runtime_stats_since(before);
                    assert!(status.converged, "{status:?}");
                    assert_eq!(stats.transfer.h2d_bytes, 0);
                    if i > 0 {
                        assert_eq!(stats.allocation_operations, 0, "{stats:?}");
                    }
                    let br: Vec<_> = remanence
                        .iter()
                        .flat_map(|v| v[..dimension].iter().map(|b| amplitude * b))
                        .collect();
                    pool.backend()
                        .try_copy_h2d(&br, remanence_device.raw_mut())
                        .unwrap();
                    let before = pool.backend().begin_runtime_stats();
                    fields.try_update_remanence(remanence_device.raw()).unwrap();
                    fields.evaluate(session.solution().unwrap()).unwrap();
                    let stats = pool.backend().runtime_stats_since(before);
                    assert_eq!(stats.transfer.h2d_bytes, 0, "{stats:?}");
                    assert_eq!(stats.transfer.d2h_bytes, 12, "{stats:?}");
                    if i > 0 {
                        assert_eq!(stats.allocation_operations, 0, "{stats:?}");
                    }
                    let mut gpu_field = vec![0.0; br.len()];
                    pool.backend()
                        .try_copy_d2h(fields.field().unwrap(), &mut gpu_field)
                        .unwrap();
                    let mut actual = vec![0.0; coordinates.len()];
                    pool.backend()
                        .try_copy_d2h(session.solution().unwrap(), &mut actual)
                        .unwrap();
                    let potential_error = actual
                        .iter()
                        .zip(&coordinates)
                        .map(|(v, p)| (v - exact_potential(p[0], amplitude)).abs())
                        .fold(0.0, f64::max);
                    assert!(
                        potential_error < 1e-7,
                        "{element:?} {order:?}: {potential_error}"
                    );
                    let mut flux_error = 0.0_f64;
                    let mut field_parity_error = 0.0_f64;
                    for e in 0..mesh.elements.len() {
                        let q = if element == FemElement::Tri3 {
                            [1.0 / 3.0, 1.0 / 3.0, 0.0]
                        } else {
                            [0.25; 3]
                        };
                        let tab = space.tabulate(&mesh, e, q[0], q[1], q[2]);
                        let r = region(tab.physical_coords[0]);
                        let values: Vec<_> = space
                            .local_to_global(&mesh, e)
                            .iter()
                            .map(|&j| actual[j])
                            .collect();
                        let b = compute_b_field(
                            &values,
                            &tab,
                            MU[r],
                            &[amplitude * REMANENCE[r], 0.0, 0.0],
                        );
                        for (d, b) in b.into_iter().enumerate() {
                            let device_b = gpu_field[e * dimension + d];
                            field_parity_error = field_parity_error.max((device_b - b).abs());
                            flux_error = flux_error.max(
                                (device_b - if d == 0 { exact_flux(amplitude) } else { 0.0 }).abs(),
                            );
                            flux_error = flux_error
                                .max((b - if d == 0 { exact_flux(amplitude) } else { 0.0 }).abs());
                        }
                    }
                    assert!(field_parity_error < 1e-12, "{field_parity_error}");
                    assert!(flux_error < 1e-6, "{element:?} {order:?}: {flux_error}");
                    let residual = matvec(&csr, &actual, action)
                        .iter()
                        .zip(&physical.rhs)
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt()
                        / physical.rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
                    assert!(residual < 1.1e-9, "{residual}");
                    eprintln!(
                        "magnetic {element:?} {order:?} {action:?} amplitude={amplitude}: potential_error={potential_error} flux_error={flux_error} residual={residual} field_parity_error={field_parity_error}"
                    );
                }
                check(session, &csr, action);
            }
        }
    }
}

#[test]
fn native_magnetic_scalar_rejection_precedes_assembly() {
    let Some(pool) = pool() else { return };
    let base = FemWorkload::native_cuda_magnetic_scalar_potential(
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
            scalar: FemScalar::F32,
            ..base
        },
        FemWorkload {
            space: FemSpace::Hcurl,
            ..base
        },
        FemWorkload {
            space: FemSpace::Hdiv,
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
            || panic!("unsupported magnetic request assembled"),
            |_| panic!("unsupported magnetic request prepared"),
        );
        assert!(matches!(result, Err(FemDeviceSolveError::Admission(_))));
        let stats = pool.backend().runtime_stats_since(before);
        assert_eq!(stats.allocation_operations + stats.kernel_launches, 0);
        assert_eq!(stats.transfer.h2d_bytes + stats.transfer.d2h_bytes, 0);
    }
}
