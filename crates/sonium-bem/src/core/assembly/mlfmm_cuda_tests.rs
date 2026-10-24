//! Native retained solves against CPU MLFMM and independent dense TBEM references.

// Rust guideline compliant 2026-02-21
use super::super::{build_cluster_tree, build_mlfmm_system_stabilized_checked};
use super::*;
use crate::core::assembly::tbem::build_tbem_system_stabilized;
use crate::core::mesh::generators::generate_icosphere_mesh;
use crate::core::types::{BoundaryCondition, PhysicsParams};
use sonium_compute::{CudaFailureInjectionPoint, MemoryPoolConfig};

fn pool() -> Option<MemoryPool<CudaBackend>> {
    match CudaBackend::new_default() {
        Ok(b) => Some(MemoryPool::new(b, MemoryPoolConfig::default()).unwrap()),
        Err(e) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                "{e}"
            );
            None
        }
    }
}

fn fixture(
    subdivisions: usize,
    target: usize,
) -> (MlfmmSystem, sonium_solvers::dense::DenseMatrix<Complex64>) {
    let mesh = generate_icosphere_mesh(1.0, subdivisions);
    let mut elements = mesh.elements.clone();
    for (i, e) in elements.iter_mut().enumerate() {
        e.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::default()]);
        e.dof_addresses = vec![i];
    }
    let physics = PhysicsParams::new(343.0 / (2.0 * std::f64::consts::PI), 343.0, 1.21, false);
    let tree = build_cluster_tree(&elements, target, &physics);
    let dense = build_tbem_system_stabilized(&elements, &mesh.nodes, &physics, 1.0, 1.0).matrix;
    (
        build_mlfmm_system_stabilized_checked(&elements, &mesh.nodes, tree, &physics, 1.0, 1.0)
            .unwrap(),
        dense,
    )
}

fn controls() -> BackendKrylovConfig<f64> {
    BackendKrylovConfig {
        restart: 10,
        max_iterations: 200,
        tolerance: 1e-9,
        require_native: true,
        exact_residual_check_interval: 0,
    }
}

#[test]
fn mlfmm_cuda_preconditioner_rejects_invalid_self_blocks() {
    let (system, _) = fixture(0, 40);
    let index = system
        .near_matrix
        .iter()
        .position(|b| b.source_cluster == b.field_cluster)
        .unwrap();
    for kind in [
        CudaMlfmmPreconditioner::Jacobi,
        CudaMlfmmPreconditioner::LeafBlockJacobi,
    ] {
        let mut missing = system.clone();
        missing.near_matrix.remove(index);
        assert!(inverse_blocks(&missing, Transpose::NoTrans, kind).is_err());
        let mut duplicate = system.clone();
        duplicate
            .near_matrix
            .push(duplicate.near_matrix[index].clone());
        assert!(inverse_blocks(&duplicate, Transpose::NoTrans, kind).is_err());
        let mut singular = system.clone();
        singular.near_matrix[index]
            .coefficients
            .fill(Complex64::default());
        assert!(inverse_blocks(&singular, Transpose::NoTrans, kind).is_err());
        let mut nonfinite = system.clone();
        nonfinite.near_matrix[index].coefficients[[0, 0]] = Complex64::new(f64::NAN, 0.0);
        assert!(inverse_blocks(&nonfinite, Transpose::ConjTrans, kind).is_err());
    }
}

fn apply(system: &MlfmmSystem, action: Transpose, x: &[Complex64]) -> Vec<Complex64> {
    match action {
        Transpose::NoTrans => system.matvec(x),
        Transpose::Trans => system.matvec_transpose(x),
        Transpose::ConjTrans => system
            .matvec_transpose(&x.iter().map(|v| v.conj()).collect::<Vec<_>>())
            .into_iter()
            .map(|v| v.conj())
            .collect(),
    }
}

#[test]
fn native_mlfmm_retained_preconditioned_solves() {
    let Some(pool) = pool() else {
        return;
    };
    let (system, dense) = fixture(2, 4);
    assert!(
        system.d_matrices.iter().any(|level| !level.is_empty()),
        "fixture must exercise far translations"
    );
    let n = system.num_dofs;
    let mut rhs = pool.try_alloc(n).unwrap();
    for preconditioner in [
        CudaMlfmmPreconditioner::Jacobi,
        CudaMlfmmPreconditioner::LeafBlockJacobi,
    ] {
        for action in [Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
            let mut session = system
                .try_prepare_cuda_solve(&pool, action, preconditioner, controls())
                .unwrap();
            for pass in 0..3 {
                let exact: Vec<_> = (0..n)
                    .map(|i| {
                        Complex64::new(
                            (0.17 * i as f64 + pass as f64).sin(),
                            (0.11 * i as f64 - pass as f64).cos(),
                        )
                    })
                    .collect();
                // Independent TBEM forcing tests the physical approximation,
                // not just an inverse crime against the uploaded operator.
                let host: Vec<_> = (0..n)
                    .map(|i| {
                        (0..n)
                            .map(|j| {
                                let a = match action {
                                    Transpose::NoTrans => dense[[i, j]],
                                    Transpose::Trans => dense[[j, i]],
                                    Transpose::ConjTrans => dense[[j, i]].conj(),
                                };
                                a * exact[j]
                            })
                            .sum()
                    })
                    .collect();
                pool.backend().try_copy_h2d(&host, rhs.raw_mut()).unwrap();
                let before = pool.backend().begin_runtime_stats();
                let report = session
                    .solve_device(rhs.raw(), DeviceInitialGuess::Zero)
                    .unwrap();
                let stats = pool.backend().runtime_stats_since(before);
                if pass > 0 {
                    assert_eq!(stats.allocation_operations, 0, "{stats:?}");
                }
                let mut x = vec![Complex64::default(); n];
                pool.backend()
                    .try_copy_d2h(session.solution_device().unwrap(), &mut x)
                    .unwrap();
                let norm = host.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
                let residual = apply(&system, action, &x)
                    .iter()
                    .zip(&host)
                    .map(|(a, b)| (*a - *b).norm_sqr())
                    .sum::<f64>()
                    .sqrt()
                    / norm;
                assert!(residual < 2e-9, "{residual}");
                let physical = (0..n)
                    .map(|i| {
                        let ax: Complex64 = (0..n)
                            .map(|j| {
                                let a = match action {
                                    Transpose::NoTrans => dense[[i, j]],
                                    Transpose::Trans => dense[[j, i]],
                                    Transpose::ConjTrans => dense[[j, i]].conj(),
                                };
                                a * x[j]
                            })
                            .sum();
                        (ax - host[i]).norm_sqr()
                    })
                    .sum::<f64>()
                    .sqrt()
                    / norm;
                assert!(physical < 1e-5, "MLFMM truncation residual {physical}");
                eprintln!(
                    "MLFMM n={n} action={action:?} preconditioner={preconditioner:?} pass={pass} iterations={} residual={residual:e} physical={physical:e} allocations={} h2d={} d2h={}",
                    report.iterations,
                    stats.allocation_operations,
                    stats.transfer.h2d_bytes,
                    stats.transfer.d2h_bytes
                );
                assert_eq!(
                    session
                        .solve_device(rhs.raw(), DeviceInitialGuess::PreviousSolution)
                        .unwrap()
                        .iterations,
                    0
                );
            }
        }
    }
}

#[test]
fn native_mlfmm_session_failure_and_recovery() {
    let Some(pool) = pool() else {
        return;
    };
    let (system, _) = fixture(0, 40);
    let n = system.num_dofs;
    let mut session = system
        .try_prepare_cuda_solve(
            &pool,
            Transpose::ConjTrans,
            CudaMlfmmPreconditioner::LeafBlockJacobi,
            controls(),
        )
        .unwrap();
    let mut rhs = pool.try_alloc(n).unwrap();
    let host = vec![Complex64::new(1.0, 0.3); n];
    pool.backend().try_copy_h2d(&host, rhs.raw_mut()).unwrap();
    session
        .solve_device(rhs.raw(), DeviceInitialGuess::Zero)
        .unwrap();
    let short = pool.try_alloc(n - 1).unwrap();
    let foreign = MemoryPool::new(
        CudaBackend::new_default().unwrap(),
        MemoryPoolConfig::default(),
    )
    .unwrap();
    let foreign_rhs = foreign.try_alloc(n).unwrap();
    for bad in [short.raw(), foreign_rhs.raw()] {
        let before = pool.backend().begin_runtime_stats();
        assert!(session.solve_device(bad, DeviceInitialGuess::Zero).is_err());
        assert_eq!(
            pool.backend().runtime_stats_since(before).kernel_launches,
            0
        );
        assert!(session.solution_device().is_none());
        assert!(
            session
                .solve_device(rhs.raw(), DeviceInitialGuess::PreviousSolution)
                .is_err()
        );
        session
            .solve_device(rhs.raw(), DeviceInitialGuess::Zero)
            .unwrap();
    }
    pool.backend()
        .try_copy_h2d(&vec![Complex64::new(f64::NAN, 0.0); n], rhs.raw_mut())
        .unwrap();
    assert!(
        session
            .solve_device(rhs.raw(), DeviceInitialGuess::Zero)
            .is_err()
    );
    pool.backend().try_copy_h2d(&host, rhs.raw_mut()).unwrap();
    pool.backend()
        .inject_failure(CudaFailureInjectionPoint::Launch);
    assert!(
        session
            .solve_device(rhs.raw(), DeviceInitialGuess::Zero)
            .is_err()
    );
    assert!(session.solution_device().is_none());
    pool.backend().recover_after_error().unwrap();
    session
        .solve_device(rhs.raw(), DeviceInitialGuess::Zero)
        .unwrap();
    pool.backend()
        .try_copy_h2d(&vec![Complex64::default(); n], rhs.raw_mut())
        .unwrap();
    assert_eq!(
        session
            .solve_device(rhs.raw(), DeviceInitialGuess::Zero)
            .unwrap()
            .iterations,
        0
    );
    let mut bad_controls = controls();
    bad_controls.require_native = false;
    assert!(
        system
            .try_prepare_cuda_solve(
                &pool,
                Transpose::NoTrans,
                CudaMlfmmPreconditioner::Jacobi,
                bad_controls
            )
            .is_err()
    );
    let mut limited = controls();
    limited.max_iterations = 1;
    limited.restart = 1;
    let mut nonconverged = system
        .try_prepare_cuda_solve(
            &pool,
            Transpose::NoTrans,
            CudaMlfmmPreconditioner::Jacobi,
            limited,
        )
        .unwrap();
    let varying: Vec<_> = (0..n)
        .map(|i| Complex64::new((0.7 * i as f64).sin(), (0.2 * i as f64).cos()))
        .collect();
    pool.backend()
        .try_copy_h2d(&varying, rhs.raw_mut())
        .unwrap();
    assert!(
        nonconverged
            .solve_device(rhs.raw(), DeviceInitialGuess::Zero)
            .is_err()
    );
    assert!(nonconverged.solution_device().is_none());
    assert!(
        nonconverged
            .solve_device(rhs.raw(), DeviceInitialGuess::PreviousSolution)
            .is_err()
    );
}
