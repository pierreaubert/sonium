//! Parallel-plate potential and electric-field parity through retained CUDA solves.

// Rust guideline compliant 2026-02-21
use super::*;
use crate::backend::{FemElement, FemOrder};
use crate::boundary::generic::try_apply_dirichlet_values;
use crate::forms::LaplacianForm;
use crate::forms::VariableLaplacianForm;
use crate::physics::electromagnetics::electrostatic::{
    CudaElectrostaticCentroidField, CudaElectrostaticObservables, EPSILON_0,
    ElectrostaticSolvedField,
};

#[test]
fn native_electrostatic_capacitor_potential_field_and_reuse() {
    let Some(pool) = pool() else { return };
    let backend = pool.backend();
    for (element, mut mesh) in [
        (
            FemElement::Tri3,
            rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 4, 4),
        ),
        (FemElement::Tetrahedron, unit_cube_tetrahedra(3)),
    ] {
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
            let coordinates = space.dof_coordinates(&mesh).unwrap();
            let n = coordinates.len();
            // Homogeneous positive permittivity: divide the charge-free PDE by
            // epsilon to avoid imposing an absolute AMG scale near 1e-12.
            let assemble = |voltage| -> Result<FemSystem<f64>, FemDeviceSolveError> {
                let matrix = assemble_bilinear_checked(&LaplacianForm, &space, &space, &mesh, 4)?;
                let mut system = FemSystem::new(matrix, vec![0.0; n]);
                try_apply_dirichlet_values(
                    &mut system,
                    &mesh,
                    &space,
                    1,
                    &|x, _, _| voltage * x,
                    None,
                )?;
                Ok(system)
            };
            let unit: CsrMatrix<f64> =
                assemble_bilinear_checked(&LaplacianForm, &space, &space, &mesh, 4)
                    .unwrap()
                    .to_csr();
            for action in [
                FemAction::Forward,
                FemAction::Transpose,
                FemAction::Hermitian,
            ] {
                let mut session = CudaFemSolveSession::try_assemble(
                    &pool,
                    FemWorkload::native_cuda_electrostatic_potential(element, order, action),
                    FemDeviceMethod::Pcg(DevicePcgAssumption::PositiveDefinite),
                    || assemble(1.0),
                    |a| {
                        BackendAmgPreconditioner::try_from_csr(
                            backend,
                            a,
                            AmgConfig {
                                coarse_size: 8,
                                ..AmgConfig::for_backend_pcg()
                            },
                        )
                    },
                )
                .unwrap();
                let mut field =
                    CudaElectrostaticCentroidField::try_new(&pool, &mesh, &space).unwrap();
                for permittivity in [0.0, -1.0, f64::NAN, f64::INFINITY] {
                    let before = backend.begin_runtime_stats();
                    assert!(
                        CudaElectrostaticObservables::try_new(&pool, &unit, permittivity).is_err()
                    );
                    let stats = backend.runtime_stats_since(before);
                    assert_eq!(stats.allocation_operations, 0);
                    assert_eq!(stats.transfer.h2d_bytes + stats.transfer.d2h_bytes, 0);
                }
                let mut integrals =
                    CudaElectrostaticObservables::try_new(&pool, &unit, EPSILON_0).unwrap();
                let dimension = field.spatial_dimension();
                let mut rhs = pool.try_alloc::<f64>(n).unwrap();
                for (index, voltage) in [1.0, 2.0, -0.5].into_iter().enumerate() {
                    let physical = assemble(voltage).unwrap();
                    backend.try_copy_h2d(&physical.rhs, rhs.raw_mut()).unwrap();
                    let before = backend.begin_runtime_stats();
                    session.try_update_rhs(rhs.raw()).unwrap();
                    let status = session
                        .solve(
                            if index == 0 {
                                DeviceInitialGuess::Zero
                            } else {
                                DeviceInitialGuess::PreviousSolution
                            },
                            &controls(),
                        )
                        .unwrap();
                    assert!(status.converged, "{element:?}/{order:?}: {status:?}");
                    let stats = backend.runtime_stats_since(before);
                    assert_eq!(stats.transfer.h2d_bytes, 0);
                    if index > 0 {
                        assert_eq!(stats.allocation_operations, 0);
                    }
                    let before = backend.begin_runtime_stats();
                    let native_observables = integrals
                        .evaluate(session.solution().unwrap(), voltage)
                        .unwrap();
                    let stats = backend.runtime_stats_since(before);
                    assert_eq!(stats.transfer.h2d_bytes, 0);
                    assert!(stats.transfer.d2h_bytes <= 64, "{stats:?}");
                    if index > 0 {
                        assert_eq!(stats.allocation_operations, 0);
                    }
                    let before = backend.begin_runtime_stats();
                    field.evaluate(session.solution().unwrap()).unwrap();
                    let stats = backend.runtime_stats_since(before);
                    assert_eq!(stats.transfer.h2d_bytes, 0);
                    assert!(stats.transfer.d2h_bytes <= 16);
                    if index > 0 {
                        assert_eq!(stats.allocation_operations, 0);
                    }
                    let mut potential = vec![0.0; n];
                    let mut electric = vec![0.0; mesh.elements.len() * dimension];
                    backend
                        .try_copy_d2h(session.solution().unwrap(), &mut potential)
                        .unwrap();
                    backend
                        .try_copy_d2h(field.field().unwrap(), &mut electric)
                        .unwrap();
                    for (&v, p) in potential.iter().zip(&coordinates) {
                        assert!((v - voltage * p[0]).abs() < 1e-7);
                    }
                    let cpu = ElectrostaticSolvedField::new(&mesh, &space, &potential, EPSILON_0)
                        .unwrap();
                    let reference = if dimension == 2 {
                        [1.0 / 3.0, 1.0 / 3.0, 0.0]
                    } else {
                        [0.25; 3]
                    };
                    for e in 0..mesh.elements.len() {
                        let expected = cpu
                            .electric_field_at_reference(
                                e,
                                reference[0],
                                reference[1],
                                reference[2],
                            )
                            .unwrap();
                        for d in 0..dimension {
                            assert!((electric[e * dimension + d] - expected[d]).abs() < 1e-11);
                            assert!(
                                (electric[e * dimension + d] - if d == 0 { -voltage } else { 0.0 })
                                    .abs()
                                    < 1e-6
                            );
                        }
                    }
                    // Unit area/gap (unit depth for 2D). CPU postprocessing is
                    // an independent parity reference after native evaluation.
                    let observables = cpu.capacitor_observables(&unit, voltage).unwrap();
                    assert!((observables.capacitance_f / EPSILON_0 - 1.0).abs() < 1e-7);
                    assert!((native_observables.capacitance_f / EPSILON_0 - 1.0).abs() < 1e-7);
                    for (native, reference) in [
                        (native_observables.energy_j, observables.energy_j),
                        (native_observables.capacitance_f, observables.capacitance_f),
                        (native_observables.charge_c, observables.charge_c),
                    ] {
                        assert!((native / reference - 1.0).abs() < 1e-10);
                    }
                    let csr = physical.matrix.to_csr();
                    let residual = matvec(&csr, &potential, action)
                        .iter()
                        .zip(&physical.rhs)
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt()
                        / physical.rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
                    assert!(residual < 1.1e-9, "{residual}");
                }
                let short = pool.try_alloc::<f64>(n - 1).unwrap();
                assert!(integrals.evaluate(short.raw(), 1.0).is_err());
                for voltage in [0.0, f64::NAN, f64::INFINITY, f64::MAX, f64::MIN_POSITIVE] {
                    let before = backend.begin_runtime_stats();
                    assert!(
                        integrals
                            .evaluate(session.solution().unwrap(), voltage)
                            .is_err()
                    );
                    let stats = backend.runtime_stats_since(before);
                    assert_eq!(stats.allocation_operations, 0);
                    assert_eq!(stats.kernel_launches + stats.library_operations, 0);
                    assert_eq!(stats.transfer.h2d_bytes + stats.transfer.d2h_bytes, 0);
                }
                let mut bad = pool.try_alloc::<f64>(n).unwrap();
                backend
                    .try_copy_h2d(&vec![f64::NAN; n], bad.raw_mut())
                    .unwrap();
                assert!(integrals.evaluate(bad.raw(), 1.0).is_err());
                backend
                    .try_copy_h2d(&vec![f64::MAX; n], bad.raw_mut())
                    .unwrap();
                assert!(integrals.evaluate(bad.raw(), 1.0).is_err());
                let mut negative = unit.clone();
                negative.values.iter_mut().for_each(|v| *v = -*v);
                let mut nonphysical =
                    CudaElectrostaticObservables::try_new(&pool, &negative, EPSILON_0).unwrap();
                assert!(
                    nonphysical
                        .evaluate(session.solution().unwrap(), -0.5)
                        .is_err()
                );
                backend.inject_failure(sonium_compute::CudaFailureInjectionPoint::Launch);
                assert!(
                    integrals
                        .evaluate(session.solution().unwrap(), 1.0)
                        .is_err()
                );
                backend.recover_after_error().unwrap();
                assert!(
                    integrals
                        .evaluate(session.solution().unwrap(), -0.5)
                        .unwrap()
                        .charge_c
                        < 0.0
                );
                assert!(field.evaluate(short.raw()).is_err());
                assert!(field.field().is_none());
                field.evaluate(session.solution().unwrap()).unwrap();
                backend.inject_failure(sonium_compute::CudaFailureInjectionPoint::Launch);
                assert!(field.evaluate(session.solution().unwrap()).is_err());
                assert!(field.field().is_none());
                backend.recover_after_error().unwrap();
                field.evaluate(session.solution().unwrap()).unwrap();
                assert!(field.field().is_some());
            }
        }
    }
}

#[test]
fn native_electrostatic_layered_material_updates() {
    layered_material_updates(false);
}

#[test]
fn native_electrostatic_resident_material_combinations() {
    layered_material_updates(true);
}

fn layered_material_updates(combine: bool) {
    let Some(pool) = pool() else { return };
    let backend = pool.backend();
    for (element, mut mesh) in [
        (
            FemElement::Tri3,
            rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 4, 4),
        ),
        (FemElement::Tetrahedron, unit_cube_tetrahedra(2)),
    ] {
        // The material interface x=1/2 conforms to both meshes.
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
            let coordinates = space.dof_coordinates(&mesh).unwrap();
            let n = coordinates.len();
            let assemble = |contrast: f64, voltage: f64| -> Result<_, FemDeviceSolveError> {
                // Normalize the solve by epsilon_0, retaining positive relative weights.
                let form =
                    VariableLaplacianForm::new(move |x, _, _| if x < 0.5 { 1.0 } else { contrast });
                let matrix = assemble_bilinear_checked(&form, &space, &space, &mesh, 4)?;
                let mut energy_matrix = matrix.to_csr();
                energy_matrix
                    .values
                    .iter_mut()
                    .for_each(|v| *v *= EPSILON_0);
                let mut system = FemSystem::new(matrix, vec![0.0; n]);
                try_apply_dirichlet_values(
                    &mut system,
                    &mesh,
                    &space,
                    1,
                    &|x, _, _| voltage * x,
                    None,
                )?;
                Ok((system, energy_matrix))
            };
            let (initial, energy_initial) = assemble(2.0, 1.0).unwrap();
            // Affine bases include constraint identity only in the base, and
            // retain the corresponding material-dependent Dirichlet lifting.
            let (base_system, base_energy) = assemble(1.0, 1.0).unwrap();
            let base_matrix = base_system.matrix.to_csr();
            let second_matrix = initial.matrix.to_csr();
            assert_eq!(base_matrix.row_ptrs, second_matrix.row_ptrs);
            assert_eq!(base_matrix.col_indices, second_matrix.col_indices);
            assert_eq!(base_energy.row_ptrs, energy_initial.row_ptrs);
            assert_eq!(base_energy.col_indices, energy_initial.col_indices);
            let difference =
                |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a - b).collect::<Vec<_>>();
            let upload = |values: &[f64]| {
                let mut buffer = pool.try_alloc::<f64>(values.len()).unwrap();
                backend.try_copy_h2d(values, buffer.raw_mut()).unwrap();
                buffer
            };
            let matrix_base = upload(&base_matrix.values);
            let matrix_delta = upload(&difference(&second_matrix.values, &base_matrix.values));
            let rhs_base = upload(&base_system.rhs);
            let rhs_delta = upload(&difference(&initial.rhs, &base_system.rhs));
            let energy_base = upload(&base_energy.values);
            let energy_delta = upload(&difference(&energy_initial.values, &base_energy.values));
            let mut session = CudaFemSolveSession::try_assemble(
                &pool,
                FemWorkload::native_cuda_electrostatic_potential(
                    element,
                    order,
                    FemAction::Forward,
                ),
                FemDeviceMethod::Pcg(DevicePcgAssumption::PositiveDefinite),
                || Ok(initial),
                |a| {
                    BackendAmgPreconditioner::try_from_csr(
                        backend,
                        a,
                        AmgConfig {
                            coarse_size: 8,
                            ..AmgConfig::for_backend_pcg()
                        },
                    )
                },
            )
            .unwrap();
            let mut integrals =
                CudaElectrostaticObservables::try_from_stiffness(&pool, &energy_initial).unwrap();
            let mut rhs = pool.try_alloc::<f64>(n).unwrap();
            let mut solve_values = pool.try_alloc::<f64>(session.pattern().1.len()).unwrap();
            let mut energy_values = pool.try_alloc::<f64>(energy_initial.values.len()).unwrap();
            for (index, (contrast, voltage)) in [(2.0, 1.0), (4.0, -2.0), (0.5, 0.75)]
                .into_iter()
                .enumerate()
            {
                let (system, energy_matrix) = assemble(contrast, voltage).unwrap();
                let csr = system.matrix.to_csr();
                assert_eq!(
                    session.pattern(),
                    (csr.row_ptrs.as_slice(), csr.col_indices.as_slice())
                );
                assert_eq!(energy_matrix.row_ptrs, energy_initial.row_ptrs);
                assert_eq!(energy_matrix.col_indices, energy_initial.col_indices);
                if !combine {
                    // Direct replacement retains its explicit host upload boundary.
                    backend
                        .try_copy_h2d(&csr.values, solve_values.raw_mut())
                        .unwrap();
                    backend.try_copy_h2d(&system.rhs, rhs.raw_mut()).unwrap();
                    backend
                        .try_copy_h2d(&energy_matrix.values, energy_values.raw_mut())
                        .unwrap();
                }
                let before = backend.begin_runtime_stats();
                if combine {
                    session
                        .try_combine_system_lagged(
                            (1.0, matrix_base.raw()),
                            &[(contrast - 1.0, matrix_delta.raw())],
                            (voltage, rhs_base.raw()),
                            &[(voltage * (contrast - 1.0), rhs_delta.raw())],
                        )
                        .unwrap();
                } else {
                    session
                        .try_update_values_lagged(solve_values.raw())
                        .unwrap();
                    session.try_update_rhs(rhs.raw()).unwrap();
                }
                if combine {
                    let update_stats = backend.runtime_stats_since(before);
                    assert_eq!(update_stats.transfer.d2d_bytes, 0, "{update_stats:?}");
                }
                let status = session
                    .solve(
                        if index == 0 {
                            DeviceInitialGuess::Zero
                        } else {
                            DeviceInitialGuess::PreviousSolution
                        },
                        &controls(),
                    )
                    .unwrap();
                assert!(status.converged, "{element:?}/{order:?}: {status:?}");
                let stats = backend.runtime_stats_since(before);
                assert_eq!(stats.transfer.h2d_bytes, 0);
                if index > 0 {
                    assert_eq!(stats.allocation_operations, 0);
                }
                let before = backend.begin_runtime_stats();
                if combine {
                    integrals
                        .try_combine_values(
                            1.0,
                            energy_base.raw(),
                            &[(contrast - 1.0, energy_delta.raw())],
                        )
                        .unwrap();
                } else {
                    integrals.try_update_values(energy_values.raw()).unwrap();
                }
                let output = integrals
                    .evaluate(session.solution().unwrap(), voltage)
                    .unwrap();
                let stats = backend.runtime_stats_since(before);
                assert_eq!(stats.transfer.h2d_bytes, 0);
                assert!(stats.transfer.d2h_bytes <= 80, "{stats:?}");
                if index > 0 {
                    assert_eq!(stats.allocation_operations, 0);
                }
                // Two dielectrics in series: C/A = epsilon_0 / (d1 + d2/contrast).
                let resistance = 0.5 + 0.5 / contrast;
                let capacitance = EPSILON_0 / resistance;
                assert!((output.capacitance_f / capacitance - 1.0).abs() < 1e-7);
                assert!(
                    (output.energy_j / (0.5 * capacitance * voltage * voltage) - 1.0).abs() < 1e-7
                );
                assert!((output.charge_c / (capacitance * voltage) - 1.0).abs() < 1e-7);
                let mut potential = vec![0.0; n];
                backend
                    .try_copy_d2h(session.solution().unwrap(), &mut potential)
                    .unwrap();
                for (&v, p) in potential.iter().zip(&coordinates) {
                    let expected =
                        voltage * (p[0].min(0.5) + (p[0] - 0.5).max(0.0) / contrast) / resistance;
                    assert!((v - expected).abs() < 1e-7, "{v} != {expected}");
                }
                let residual = matvec(&csr, &potential, FemAction::Forward)
                    .iter()
                    .zip(&system.rhs)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt()
                    / system.rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
                assert!(residual < 1.1e-9, "{residual}");
                let energy_cpu = 0.5
                    * potential
                        .iter()
                        .zip(matvec(&energy_matrix, &potential, FemAction::Forward))
                        .map(|(v, kv)| v * kv)
                        .sum::<f64>();
                assert!((output.energy_j / energy_cpu - 1.0).abs() < 1e-10);
            }
            if combine {
                // Invalid combinations cannot publish or solve a partially updated system.
                for coefficient in [f64::NAN, f64::MAX] {
                    assert!(
                        session
                            .try_combine_system_lagged(
                                (coefficient, matrix_base.raw()),
                                &[],
                                (1.0, rhs_base.raw()),
                                &[],
                            )
                            .is_err()
                    );
                    assert!(session.solution().is_none());
                    assert!(
                        session
                            .solve(DeviceInitialGuess::Zero, &controls())
                            .is_err()
                    );
                    assert!(
                        integrals
                            .try_combine_values(coefficient, matrix_base.raw(), &[])
                            .is_err()
                    );
                }
                backend.inject_failure(CudaFailureInjectionPoint::TransferD2D);
                assert!(
                    session
                        .try_combine_system_lagged(
                            (1.0, matrix_base.raw()),
                            &[],
                            (1.0, rhs_base.raw()),
                            &[],
                        )
                        .is_err()
                );
                backend.recover_after_error().unwrap();
                assert!(
                    session
                        .solve(DeviceInitialGuess::Zero, &controls())
                        .is_err()
                );
                session
                    .try_combine_system_lagged(
                        (1.0, matrix_base.raw()),
                        &[(-0.5, matrix_delta.raw())],
                        (0.75, rhs_base.raw()),
                        &[(-0.375, rhs_delta.raw())],
                    )
                    .unwrap();
                assert!(
                    session
                        .solve(DeviceInitialGuess::Zero, &controls())
                        .unwrap()
                        .converged
                );
                integrals
                    .try_combine_values(1.0, energy_base.raw(), &[(-0.5, energy_delta.raw())])
                    .unwrap();
                assert!(
                    integrals
                        .evaluate(session.solution().unwrap(), 0.75)
                        .unwrap()
                        .energy_j
                        > 0.0
                );
                // The following direct-copy failure cases use a complete replacement.
                let (_, final_energy) = assemble(0.5, 0.75).unwrap();
                backend
                    .try_copy_h2d(&final_energy.values, energy_values.raw_mut())
                    .unwrap();
            }
            // Every update failure blocks stale energy evaluation until replacement.
            let short = pool.try_alloc::<f64>(energy_values.len() - 1).unwrap();
            assert!(integrals.try_update_values(short.raw()).is_err());
            assert!(
                integrals
                    .evaluate(session.solution().unwrap(), 0.75)
                    .is_err()
            );
            integrals.try_update_values(energy_values.raw()).unwrap();
            let mut bad = pool.try_alloc::<f64>(energy_values.len()).unwrap();
            backend
                .try_copy_h2d(&vec![f64::NAN; energy_values.len()], bad.raw_mut())
                .unwrap();
            assert!(integrals.try_update_values(bad.raw()).is_err());
            assert!(
                integrals
                    .evaluate(session.solution().unwrap(), 0.75)
                    .is_err()
            );
            integrals.try_update_values(energy_values.raw()).unwrap();
            {
                backend.inject_failure(CudaFailureInjectionPoint::TransferD2D);
                assert!(integrals.try_update_values(energy_values.raw()).is_err());
                backend.recover_after_error().unwrap();
                assert!(
                    integrals
                        .evaluate(session.solution().unwrap(), 0.75)
                        .is_err()
                );
                integrals.try_update_values(energy_values.raw()).unwrap();
                assert!(
                    integrals
                        .evaluate(session.solution().unwrap(), 0.75)
                        .unwrap()
                        .energy_j
                        > 0.0
                );
            }
        }
    }
    // Synchronization failure is terminal: the poisoned context cannot be repaired
    // by replacing matrix values. Exercise this only after the reusable cases.
    let matrix = CsrMatrix::from_triplets(
        2,
        2,
        vec![(0, 0, 1.0), (0, 1, -1.0), (1, 0, -1.0), (1, 1, 1.0)],
    );
    let mut integrals = CudaElectrostaticObservables::try_from_stiffness(&pool, &matrix).unwrap();
    let mut values = pool.try_alloc::<f64>(4).unwrap();
    let mut potential = pool.try_alloc::<f64>(2).unwrap();
    backend
        .try_copy_h2d(&matrix.values, values.raw_mut())
        .unwrap();
    backend
        .try_copy_h2d(&[0.0, 1.0], potential.raw_mut())
        .unwrap();
    backend.inject_failure(CudaFailureInjectionPoint::Synchronization);
    assert!(integrals.try_update_values(values.raw()).is_err());
    assert!(backend.recover_after_error().is_err());
    assert!(integrals.evaluate(potential.raw(), 1.0).is_err());
    assert!(integrals.try_update_values(values.raw()).is_err());
}
