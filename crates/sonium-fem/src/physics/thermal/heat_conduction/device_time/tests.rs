//! CPU-step, analytical diffusion, source-weighting and failure parity.

// Rust guideline compliant 2026-02-21
use super::*;
use crate::physics::thermal::heat_conduction::{ConductionForm, HeatCapacityForm, ThermalMaterial};
use crate::{
    assembly::assemble_bilinear_checked,
    basis::PolynomialDegree,
    mesh::{Mesh, generators::rectangular_mesh_triangles, unit_cube_tetrahedra},
    spaces::{FunctionSpace, H1Space},
};
use sonium_compute::{CudaFailureInjectionPoint, MemoryPoolConfig};
use sonium_solvers::{LinearSolveConfig, LinearSolverType};

fn pool() -> Option<MemoryPool<CudaBackend>> {
    match CudaBackend::new_default() {
        Ok(b) => Some(MemoryPool::new(b, MemoryPoolConfig::with_cap(1 << 29)).unwrap()),
        Err(e) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                "{e}"
            );
            None
        }
    }
}
fn controls() -> PcgBackendConfig<f64> {
    PcgBackendConfig {
        max_iterations: 300,
        tolerance: 1e-10,
    }
}
fn download(pool: &MemoryPool<CudaBackend>, source: &CudaBuffer<f64>) -> Vec<f64> {
    let mut values = vec![0.0; source.len()];
    pool.backend().try_copy_d2h(source, &mut values).unwrap();
    values
}
fn fixture(
    mut mesh: Mesh,
    degree: PolynomialDegree,
    dimension: usize,
) -> (CsrMatrix<f64>, CsrMatrix<f64>, Vec<f64>) {
    for face in &mut mesh.boundaries {
        face.marker = 1;
    }
    let space = H1Space::new(degree);
    let coords = space.dof_coordinates(&mesh).unwrap();
    let boundary: std::collections::HashSet<_> =
        space.boundary_dofs(&mesh, 1).into_iter().collect();
    let free: Vec<_> = (0..coords.len())
        .filter(|i| !boundary.contains(i))
        .collect();
    let mut map = vec![None; coords.len()];
    for (i, &dof) in free.iter().enumerate() {
        map[dof] = Some(i);
    }
    let material = ThermalMaterial::new(0.1, 1.0, 1.0);
    let mass =
        assemble_bilinear_checked(&HeatCapacityForm::new(&material), &space, &space, &mesh, 4)
            .unwrap();
    let stiffness =
        assemble_bilinear_checked(&ConductionForm::new(&material), &space, &space, &mesh, 4)
            .unwrap();
    let reduce = |a: crate::assembly::TripletMatrix<f64>| {
        let entries = a
            .rows
            .iter()
            .zip(&a.cols)
            .zip(&a.values)
            .filter_map(|((&r, &c), &v)| Some((map[r]?, map[c]?, v)))
            .collect();
        CsrMatrix::from_triplets(free.len(), free.len(), entries)
    };
    let initial = free
        .iter()
        .map(|&i| {
            coords[i][..dimension]
                .iter()
                .map(|x| (std::f64::consts::PI * x).sin())
                .product()
        })
        .collect();
    (reduce(mass), reduce(stiffness), initial)
}

#[test]
fn native_thermal_time_physical_history_and_reuse() {
    let Some(pool) = pool() else { return };
    for (dimension, mesh) in [
        (2, rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 6, 6)),
        (3, unit_cube_tetrahedra(3)),
    ] {
        for degree in [PolynomialDegree::P1, PolynomialDegree::P2] {
            let (mass, stiffness, initial) = fixture(mesh.clone(), degree, dimension);
            let n = initial.len();
            let mut source = pool.try_alloc(n).unwrap();
            pool.backend()
                .try_copy_h2d(&vec![0.0; n], source.raw_mut())
                .unwrap();
            for theta in [0.5, 0.75, 1.0] {
                let stepper = ThermalTimeStepper::theta(0.0025, theta);
                let mut session = stepper
                    .prepare_cuda(
                        &pool,
                        &mass,
                        &stiffness,
                        &initial,
                        DevicePcgAssumption::PositiveDefinite,
                    )
                    .unwrap();
                let mut cpu = initial.clone();
                for step in 0..8 {
                    let reference = stepper
                        .step(
                            &cpu,
                            &vec![0.0; n],
                            Some(&vec![0.0; n]),
                            &mass,
                            &stiffness,
                            &LinearSolveConfig {
                                solver: LinearSolverType::GmresIlu,
                                tolerance: 1e-12,
                                ..Default::default()
                            },
                        )
                        .unwrap();
                    cpu = reference.x;
                    let before = pool.backend().begin_runtime_stats();
                    let status = session
                        .step_device(source.raw(), Some(source.raw()), &controls())
                        .unwrap();
                    let stats = pool.backend().runtime_stats_since(before);
                    assert!(status.converged, "{status:?}");
                    if step > 0 {
                        assert_eq!(stats.allocation_operations, 0, "{stats:?}");
                    }
                    assert_eq!(stats.transfer.h2d_bytes, 0, "{stats:?}");
                    let actual = download(&pool, session.temperature());
                    let error = actual
                        .iter()
                        .zip(&cpu)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f64::max);
                    assert!(
                        error < 2e-8,
                        "{dimension}D {degree:?} theta={theta}: {error}"
                    );
                    assert_eq!(session.steps_completed(), step + 1);
                }
                let actual = download(&pool, session.temperature());
                let decay = (-0.1 * dimension as f64 * std::f64::consts::PI.powi(2) * 0.02).exp();
                let physical_error = actual
                    .iter()
                    .zip(&initial)
                    .map(|(a, u)| (a - u * decay).abs())
                    .fold(0.0, f64::max);
                assert!(
                    physical_error < 0.025,
                    "{dimension}D {degree:?}: {physical_error}"
                );
                eprintln!(
                    "thermal time {dimension}D {degree:?} theta={theta} n={n}: physical error={physical_error}"
                );
            }
        }
    }
}

#[test]
fn native_thermal_time_source_weights_and_recovery() {
    let Some(pool) = pool() else { return };
    let m = CsrMatrix::from_triplets(1, 1, vec![(0, 0, 2.0)]);
    let k = CsrMatrix::from_triplets(1, 1, vec![(0, 0, 3.0)]);
    let mut next = pool.try_alloc(1).unwrap();
    let mut current = pool.try_alloc(1).unwrap();
    for theta in [0.0, 0.5, 0.75, 1.0] {
        let stepper = ThermalTimeStepper::theta(0.1, theta);
        let mut session = stepper
            .prepare_cuda(&pool, &m, &k, &[2.0], DevicePcgAssumption::PositiveDefinite)
            .unwrap();
        for i in 0..5 {
            // T=2+t is exact for every theta when f(t)=M+K*T(t).
            pool.backend()
                .try_copy_h2d(&[8.0 + 0.3 * i as f64], current.raw_mut())
                .unwrap();
            pool.backend()
                .try_copy_h2d(&[8.0 + 0.3 * (i + 1) as f64], next.raw_mut())
                .unwrap();
            assert!(
                session
                    .step_device(next.raw(), Some(current.raw()), &controls())
                    .unwrap()
                    .converged
            );
            assert!(
                (download(&pool, session.temperature())[0] - (2.0 + 0.1 * (i + 1) as f64)).abs()
                    < 1e-10
            );
        }
        let committed = download(&pool, session.temperature());
        pool.backend()
            .try_copy_h2d(&[f64::NAN], next.raw_mut())
            .unwrap();
        assert!(session.step_device(next.raw(), None, &controls()).is_err());
        assert_eq!(session.steps_completed(), 5);
        assert_eq!(download(&pool, session.temperature()), committed);
        pool.backend().try_copy_h2d(&[0.0], next.raw_mut()).unwrap();
        pool.backend()
            .inject_failure(CudaFailureInjectionPoint::Launch);
        assert!(session.step_device(next.raw(), None, &controls()).is_err());
        pool.backend().recover_after_error().unwrap();
        assert_eq!(download(&pool, session.temperature()), committed);
        assert!(
            session
                .step_device(next.raw(), None, &controls())
                .unwrap()
                .converged
        );
        assert_eq!(session.steps_completed(), 6);
    }
}

#[test]
fn native_thermal_time_rejects_inputs_and_preserves_failed_history() {
    let Some(pool) = pool() else { return };
    let (mass, stiffness, initial) = fixture(
        rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 6, 6),
        PolynomialDegree::P2,
        2,
    );
    for stepper in [
        ThermalTimeStepper::backward_euler(0.0),
        ThermalTimeStepper::backward_euler(f64::NAN),
        ThermalTimeStepper {
            dt: 0.01,
            scheme: super::super::ThermalTimeScheme::Theta(1.1),
        },
    ] {
        let before = pool.backend().begin_runtime_stats();
        assert!(
            stepper
                .prepare_cuda(
                    &pool,
                    &mass,
                    &stiffness,
                    &initial,
                    DevicePcgAssumption::PositiveDefinite
                )
                .is_err()
        );
        assert_eq!(
            pool.backend()
                .runtime_stats_since(before)
                .allocation_operations,
            0
        );
    }
    let stepper = ThermalTimeStepper::crank_nicolson(0.01);
    let mut malformed = mass.clone();
    malformed.row_ptrs[0] = 1;
    assert!(
        stepper
            .prepare_cuda(
                &pool,
                &malformed,
                &stiffness,
                &initial,
                DevicePcgAssumption::PositiveDefinite
            )
            .is_err()
    );
    let mut session = stepper
        .prepare_cuda(
            &pool,
            &mass,
            &stiffness,
            &initial,
            DevicePcgAssumption::PositiveDefinite,
        )
        .unwrap();
    let mut source = pool.try_alloc(initial.len()).unwrap();
    pool.backend()
        .try_copy_h2d(&vec![0.0; initial.len()], source.raw_mut())
        .unwrap();
    let stopped = session.step_device(
        source.raw(),
        None,
        &PcgBackendConfig {
            max_iterations: 1,
            tolerance: 1e-15,
        },
    );
    assert!(stopped.is_err() || !stopped.unwrap().converged);
    assert_eq!(session.steps_completed(), 0);
    assert_eq!(download(&pool, session.temperature()), initial);
    let short = pool.try_alloc::<f64>(1).unwrap();
    let other = MemoryPool::new(
        CudaBackend::new_default().unwrap(),
        MemoryPoolConfig::with_cap(1 << 20),
    )
    .unwrap();
    let foreign = other.try_alloc::<f64>(initial.len()).unwrap();
    for invalid in [short.raw(), foreign.raw()] {
        let before = pool.backend().begin_runtime_stats();
        assert!(session.step_device(invalid, None, &controls()).is_err());
        let stats = pool.backend().runtime_stats_since(before);
        assert_eq!(stats.allocation_operations + stats.kernel_launches, 0);
        assert_eq!(stats.transfer.h2d_bytes + stats.transfer.d2h_bytes, 0);
        assert_eq!(session.steps_completed(), 0);
    }
    assert!(
        session
            .step_device(source.raw(), None, &controls())
            .unwrap()
            .converged
    );
    let committed = download(&pool, session.temperature());
    pool.backend()
        .inject_failure(CudaFailureInjectionPoint::TransferD2D);
    assert!(
        session
            .step_device(source.raw(), None, &controls())
            .is_err()
    );
    pool.backend().recover_after_error().unwrap();
    assert_eq!(download(&pool, session.temperature()), committed);
    assert_eq!(session.steps_completed(), 1);
    assert!(
        session
            .step_device(source.raw(), None, &controls())
            .unwrap()
            .converged
    );
    assert_eq!(session.steps_completed(), 2);
}

#[test]
fn native_thermal_time_scheme_refresh_preserves_physical_history() {
    let Some(pool) = pool() else { return };
    for (dimension, mesh) in [
        (2, rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 6, 6)),
        (3, unit_cube_tetrahedra(3)),
    ] {
        let (mass, stiffness, initial) = fixture(mesh, PolynomialDegree::P2, dimension);
        let mut session = ThermalTimeStepper::backward_euler(0.001)
            .prepare_cuda(
                &pool,
                &mass,
                &stiffness,
                &initial,
                DevicePcgAssumption::PositiveDefinite,
            )
            .unwrap();
        let zeros = vec![0.0; initial.len()];
        let mut source = pool.try_alloc(initial.len()).unwrap();
        pool.backend()
            .try_copy_h2d(&zeros, source.raw_mut())
            .unwrap();
        let mut cpu = initial;
        for (i, (dt, theta)) in [
            (0.001, 1.0),
            (0.0005, 0.5),
            (0.002, 0.75),
            (0.00001, 0.0),
            (0.001, 1.0),
        ]
        .into_iter()
        .enumerate()
        {
            let scheme = ThermalTimeStepper::theta(dt, theta);
            let committed = download(&pool, session.temperature());
            let rebuilt = session
                .try_update_scheme(scheme, DevicePcgAssumption::PositiveDefinite)
                .unwrap();
            assert_eq!(download(&pool, session.temperature()), committed);
            assert_eq!(session.steps_completed(), i);
            // A second identical hierarchy update must refresh, not reallocate.
            let before = pool.backend().begin_runtime_stats();
            assert!(
                !session
                    .try_update_scheme(scheme, DevicePcgAssumption::PositiveDefinite)
                    .unwrap()
            );
            let update = pool.backend().runtime_stats_since(before);
            assert_eq!(update.allocation_operations, 0, "{update:?}");
            cpu = scheme
                .step(
                    &cpu,
                    &zeros,
                    Some(&zeros),
                    &mass,
                    &stiffness,
                    &LinearSolveConfig {
                        solver: LinearSolverType::GmresIlu,
                        tolerance: 1e-12,
                        ..Default::default()
                    },
                )
                .unwrap()
                .x;
            let before = pool.backend().begin_runtime_stats();
            let status = session
                .step_device(source.raw(), Some(source.raw()), &controls())
                .unwrap();
            let stats = pool.backend().runtime_stats_since(before);
            assert!(status.converged, "{status:?}");
            assert_eq!(stats.transfer.h2d_bytes, 0);
            if i > 0 {
                assert_eq!(stats.allocation_operations, 0, "{stats:?}");
            }
            let error = download(&pool, session.temperature())
                .iter()
                .zip(&cpu)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(error < 2e-8, "{dimension}D step {i}: {error}");
            eprintln!(
                "thermal scheme refresh {dimension}D step={i} dt={dt} theta={theta} rebuilt={rebuilt} error={error} residual={}",
                status.true_residual
            );
        }
    }
}

#[test]
fn native_thermal_time_scheme_refresh_failure_requires_repair() {
    let Some(pool) = pool() else { return };
    let mass = CsrMatrix::from_triplets(1, 1, vec![(0, 0, 2.0)]);
    let stiffness = CsrMatrix::from_triplets(1, 1, vec![(0, 0, 3.0)]);
    let old = ThermalTimeStepper::backward_euler(0.1);
    let new = ThermalTimeStepper::crank_nicolson(0.05);
    let mut session = old
        .prepare_cuda(
            &pool,
            &mass,
            &stiffness,
            &[2.0],
            DevicePcgAssumption::PositiveDefinite,
        )
        .unwrap();
    let mut source = pool.try_alloc(1).unwrap();
    pool.backend()
        .try_copy_h2d(&[0.0], source.raw_mut())
        .unwrap();
    assert!(
        session
            .step_device(source.raw(), None, &controls())
            .unwrap()
            .converged
    );
    let committed = download(&pool, session.temperature());
    for invalid in [
        ThermalTimeStepper::backward_euler(0.0),
        ThermalTimeStepper::backward_euler(f64::NAN),
    ] {
        let before = pool.backend().begin_runtime_stats();
        assert!(
            session
                .try_update_scheme(invalid, DevicePcgAssumption::PositiveDefinite)
                .is_err()
        );
        let stats = pool.backend().runtime_stats_since(before);
        assert_eq!(
            stats.allocation_operations
                + stats.kernel_launches
                + stats.transfer.h2d_bytes
                + stats.transfer.d2h_bytes,
            0
        );
        assert!(session.numeric_ready);
    }
    for point in [
        CudaFailureInjectionPoint::TransferD2D,
        CudaFailureInjectionPoint::TransferH2D,
    ] {
        pool.backend().inject_failure(point);
        assert!(
            session
                .try_update_scheme(new, DevicePcgAssumption::PositiveDefinite)
                .is_err()
        );
        pool.backend().recover_after_error().unwrap();
        assert_eq!(download(&pool, session.temperature()), committed);
        assert_eq!(session.steps_completed(), 1);
        let before = pool.backend().begin_runtime_stats();
        assert!(
            session
                .step_device(source.raw(), None, &controls())
                .is_err()
        );
        let stats = pool.backend().runtime_stats_since(before);
        assert_eq!(
            stats.allocation_operations
                + stats.kernel_launches
                + stats.transfer.h2d_bytes
                + stats.transfer.d2h_bytes,
            0
        );
        session
            .try_update_scheme(new, DevicePcgAssumption::PositiveDefinite)
            .unwrap();
    }
    assert!(
        session
            .step_device(source.raw(), None, &controls())
            .unwrap()
            .converged
    );
    let expected = committed[0] * (2.0 / 0.05 - 1.5) / (2.0 / 0.05 + 1.5);
    assert!((download(&pool, session.temperature())[0] - expected).abs() < 1e-10);
    assert_eq!(session.steps_completed(), 2);
}

mod lagged;
