//! Physical Taylor–Hood flow through the retained assembled CUDA session.

// Rust guideline compliant 2026-02-21
use super::*;
use crate::assembly::{assemble_linear_checked, assemble_mixed_checked};
use crate::boundary::generic::try_apply_homogeneous_dirichlet_dofs;
use crate::forms::MassForm;
use crate::physics::fluid::stokes::forms::{
    StokesBodyForceForm, StokesDivergenceForm, StokesViscousForm,
};
use sonium_compute::preconditioner::{
    PressureSchurBackendKind, SaddlePointJacobiSchurBackend, SaddlePointJacobiSchurBackendConfig,
};

#[test]
fn native_generic_stokes_poiseuille_schur_reuses_rhs() {
    let Some(pool) = pool() else { return };
    let backend = pool.backend();
    // The pinned saddle system amplifies residual error in pressure. Certify
    // more tightly than the scalar fixtures to meet the physical pressure gate.
    let controls = BackendKrylovConfig {
        tolerance: 1e-11,
        ..controls()
    };
    for subdivisions in [3, 6] {
        let mut mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, subdivisions, subdivisions);
        for face in &mut mesh.boundaries {
            face.marker = 1;
        }
        let velocity = VectorH1Space::new(PolynomialDegree::P2, 2);
        let pressure = H1Space::new(PolynomialDegree::P1);
        let mixed = assemble_mixed_checked(
            &StokesViscousForm::new(1.0),
            &StokesDivergenceForm,
            &velocity,
            &pressure,
            &mesh,
            4,
        )
        .unwrap();
        let nv = mixed.n_dofs_0;
        let np = mixed.n_dofs_1;
        let n = nv + np;
        let divergence = mixed.block_10.to_csr();
        let mut exact: Vec<_> = velocity
            .dof_coordinates(&mesh)
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, p)| if i % 2 == 0 { p[1] * (1.0 - p[1]) } else { 0.0 })
            .collect();
        exact.extend(
            pressure
                .dof_coordinates(&mesh)
                .unwrap()
                .iter()
                .map(|p| 1.0 - p[0]),
        );
        // -Delta u + grad p = (1, 0), div u = 0. Assemble the physical
        // load independently; do not manufacture the RHS by multiplying A*x.
        let mut rhs = assemble_linear_checked(
            &StokesBodyForceForm::new(|_| vec![1.0, 0.0]),
            &velocity,
            &mesh,
            4,
        )
        .unwrap();
        rhs.resize(n, 0.0);
        let mut system = FemSystem::new(mixed.to_monolithic(), rhs);
        let mut prescribed = velocity.boundary_dofs(&mesh, 1);
        prescribed.push(nv); // Pin the first pressure DOF to its analytic value.
        let mut lift = vec![0.0; n];
        for &i in &prescribed {
            lift[i] = exact[i];
        }
        let correction = matvec(&system.matrix.to_csr(), &lift, FemAction::Forward);
        for (rhs, correction) in system.rhs.iter_mut().zip(correction) {
            *rhs -= correction;
        }
        try_apply_homogeneous_dirichlet_dofs(&mut system, &prescribed).unwrap();
        for &i in &prescribed {
            system.rhs[i] = exact[i];
        }
        let reference = system.matrix.to_csr();
        let base_rhs = system.rhs.clone();
        // For [A B; B^T 0], the Schur complement is negative. Approximate it
        // with -M_p / mu, preserving the positive identity of the pressure pin.
        let mut mass =
            assemble_bilinear_checked(&MassForm, &pressure, &pressure, &mesh, 2).unwrap();
        for value in &mut mass.values {
            *value = -*value;
        }
        let mut schur = FemSystem::new(mass, vec![0.0; np]);
        try_apply_homogeneous_dirichlet_dofs(&mut schur, &[0]).unwrap();
        let schur = schur.matrix.to_csr();
        for action in [
            FemAction::Forward,
            FemAction::Transpose,
            FemAction::Hermitian,
        ] {
            let mut session = CudaFemSolveSession::try_assemble(
                &pool,
                FemWorkload::native_cuda_stokes(action),
                FemDeviceMethod::Fgmres,
                || {
                    let mut assembled = FemSystem::new(system.matrix.clone(), system.rhs.clone());
                    assembled.dirichlet_dofs.clone_from(&system.dirichlet_dofs);
                    Ok(assembled)
                },
                |a| {
                    // The returned action owns both matrices, independent of
                    // the temporary canonical CSR supplied by the session.
                    SaddlePointJacobiSchurBackend::try_from_host_csr(
                        backend,
                        a.as_host_csr_ref(),
                        schur.as_host_csr_ref(),
                        SaddlePointJacobiSchurBackendConfig {
                            velocity_block_size: nv,
                            velocity_degree: 4,
                            velocity_damping: 0.6,
                            pressure_degree: 4,
                            pressure_damping: 0.6,
                            pressure_kind: PressureSchurBackendKind::Polynomial,
                            pressure_chebyshev_degree: 4,
                            pressure_chebyshev_eigenvalue_ratio: 30.0,
                        },
                    )
                },
            )
            .unwrap();
            let mut device_rhs = pool.try_alloc::<f64>(n).unwrap();
            let mut preconditioned = pool.try_alloc::<f64>(n).unwrap();
            backend
                .try_copy_h2d(&base_rhs, device_rhs.raw_mut())
                .unwrap();
            session
                .preconditioner
                .try_apply(backend, device_rhs.raw(), preconditioned.raw_mut())
                .unwrap();
            let before = backend.begin_runtime_stats();
            session
                .preconditioner
                .try_apply(backend, device_rhs.raw(), preconditioned.raw_mut())
                .unwrap();
            let stats = backend.runtime_stats_since(before);
            assert_eq!(stats.allocation_operations, 0);
            assert_eq!(stats.transfer.h2d_bytes, 0);
            assert_eq!(stats.transfer.d2h_bytes, 0);
            for (pass, amplitude) in [1.0, -0.5, 1.5].into_iter().enumerate() {
                let rhs: Vec<_> = base_rhs.iter().map(|v| amplitude * v).collect();
                backend.try_copy_h2d(&rhs, device_rhs.raw_mut()).unwrap();
                let before = backend.begin_runtime_stats();
                session.try_update_rhs(device_rhs.raw()).unwrap();
                let updates = backend.runtime_stats_since(before);
                if pass > 0 {
                    assert_eq!(updates.allocation_operations, 0);
                }
                assert_eq!(updates.transfer.h2d_bytes, 0);
                assert_eq!(updates.transfer.d2h_bytes, 4);
                let before = backend.begin_runtime_stats();
                let status = session
                    .solve(
                        if pass == 0 {
                            DeviceInitialGuess::Zero
                        } else {
                            DeviceInitialGuess::PreviousSolution
                        },
                        &controls,
                    )
                    .unwrap();
                let stats = backend.runtime_stats_since(before);
                assert!(status.converged, "{status:?}");
                if pass > 0 {
                    assert_eq!(stats.allocation_operations, 0, "{stats:?}");
                }
                let mut actual = vec![0.0; n];
                backend
                    .try_copy_d2h(session.solution().unwrap(), &mut actual)
                    .unwrap();
                let error = |range: std::ops::Range<usize>| {
                    range
                        .map(|i| (actual[i] - amplitude * exact[i]).abs())
                        .fold(0.0_f64, f64::max)
                };
                let velocity_error = error(0..nv);
                let pressure_error = error(nv..n);
                assert!(velocity_error < 2e-7, "velocity {velocity_error}");
                assert!(pressure_error < 1e-7, "pressure {pressure_error}");
                assert!((actual[nv] - amplitude * exact[nv]).abs() < 1e-9);
                let mut div = vec![0.0; np];
                for (row, value) in div.iter_mut().enumerate() {
                    for j in divergence.row_ptrs[row]..divergence.row_ptrs[row + 1] {
                        *value += divergence.values[j] * actual[divergence.col_indices[j]];
                    }
                }
                let divergence_error = div.into_iter().map(f64::abs).fold(0.0_f64, f64::max);
                assert!(divergence_error < 1e-8, "divergence {divergence_error}");
                let residual = matvec(&reference, &actual, action)
                    .iter()
                    .zip(&rhs)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt()
                    / rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
                assert!(residual < 1.1e-11, "residual {residual}");
                eprintln!(
                    "Stokes n={n} action={action:?} pass={pass} iterations={} allocations={} residual={residual} velocity={velocity_error} pressure={pressure_error} divergence={divergence_error}",
                    status.iterations, stats.allocation_operations
                );
            }
        }
    }
}
