//! Independent dense and exact-operator references for compressed linear sessions.

// Rust guideline compliant 2026-02-21
use super::*;
use sonium_compute::{CudaFailureInjectionPoint, MemoryPoolConfig};

fn pool() -> Option<MemoryPool<CudaBackend>> {
    match CudaBackend::new_default() {
        Ok(backend) => Some(MemoryPool::new(backend, MemoryPoolConfig::default()).unwrap()),
        Err(error) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                "CUDA required: {error}"
            );
            None
        }
    }
}

fn problem(subdivisions: usize, boundary: BoundaryConditionType, bm: bool) -> BemProblem {
    BemProblem {
        mesh: generate_icosphere_mesh(1.0, subdivisions),
        physics: PhysicsParams::new(343.0 / (2.0 * PI), 343.0, 1.21, false),
        incident_field: IncidentField::plane_wave_z(),
        bc_type: boundary,
        use_burton_miller: bm,
    }
}

fn solver() -> BemSolver {
    BemSolver {
        solver_method: SolverMethod::Gmres,
        assembly_method: AssemblyMethod::Hmatrix,
        tolerance: 1e-9,
        gmres_restart: 40,
        max_iterations: 400,
        hmatrix_aca_tolerance: 1e-7,
        compressed_exact_residual_dof_limit: 2000,
        compressed_exact_residual_tolerance: 2e-7,
        ..Default::default()
    }
}

fn dense(problem: &BemProblem) -> DenseMatrix<Complex64> {
    let solver = BemSolver::default();
    let elements = solver.prepare_elements(problem).unwrap();
    let beta = if problem.use_burton_miller {
        PhysicsParams::optimal_beta_scale(problem.ka())
    } else {
        0.0
    };
    let AssemblyResult::Dense(matrix, _) = solver
        .assemble_system_stabilized(&elements, &problem.mesh.nodes, &problem.physics, 1.0, beta)
        .unwrap()
    else {
        panic!("dense reference expected")
    };
    matrix
}

fn action(
    matrix: &DenseMatrix<Complex64>,
    x: &[Complex64],
    kind: CudaHMatrixAction,
) -> Vec<Complex64> {
    (0..x.len())
        .map(|i| {
            (0..x.len())
                .map(|j| {
                    (match kind {
                        CudaHMatrixAction::Forward => matrix[[i, j]],
                        CudaHMatrixAction::Transpose => matrix[[j, i]],
                        CudaHMatrixAction::Hermitian => matrix[[j, i]].conj(),
                    }) * x[j]
                })
                .sum()
        })
        .collect()
}

fn check(mut session: CudaHMatrixLinearSession<'_>, matrix: &DenseMatrix<Complex64>) {
    let pool = session.inner.pool;
    let n = session.num_dofs();
    let mut rhs = pool.try_alloc(n).unwrap();
    assert!(session.solution_device().is_none());
    for pass in 0..3 {
        let exact: Vec<_> = (0..n)
            .map(|i| {
                Complex64::new(
                    (0.17 * i as f64 + pass as f64).sin(),
                    (0.11 * i as f64 - pass as f64).cos(),
                )
            })
            .collect();
        let host_rhs = action(matrix, &exact, session.action());
        pool.backend()
            .try_copy_h2d(&host_rhs, rhs.raw_mut())
            .unwrap();
        let before = pool.backend().begin_runtime_stats();
        let report = session
            .solve_device(
                rhs.raw(),
                if pass == 0 {
                    DeviceInitialGuess::Zero
                } else {
                    DeviceInitialGuess::PreviousSolution
                },
            )
            .unwrap();
        let stats = pool.backend().runtime_stats_since(before);
        if pass > 0 {
            assert_eq!(stats.allocation_operations, 0, "{stats:?}");
        }
        assert!(report.compressed_relative_residual < 1e-9);
        assert!(report.exact_relative_residual.is_none_or(|r| r < 2e-7));
        let mut actual = vec![Complex64::default(); n];
        pool.backend()
            .try_copy_d2h(session.solution_device().unwrap(), &mut actual)
            .unwrap();
        let ax = action(matrix, &actual, session.action());
        let residual = ax
            .iter()
            .zip(&host_rhs)
            .map(|(a, b)| (*a - *b).norm_sqr())
            .sum::<f64>()
            .sqrt()
            / host_rhs.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        let error = actual
            .iter()
            .zip(&exact)
            .map(|(a, b)| (*a - *b).norm_sqr())
            .sum::<f64>()
            .sqrt()
            / exact.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        assert!(residual < 2e-7, "{residual}");
        assert!(error < 1e-6, "{error}");
        eprintln!(
            "linear H-matrix n={n} action={:?} pass={pass} blocks={} iterations={} residual={residual} error={error} alloc={} h2d={} d2h={}",
            session.action(),
            session.low_rank_blocks(),
            report.iterations,
            stats.allocation_operations,
            stats.transfer.h2d_bytes,
            stats.transfer.d2h_bytes
        );
    }
}

#[test]
fn native_hmatrix_linear_all_actions_match_dense_reference() {
    let Some(pool) = pool() else { return };
    for boundary in [
        BoundaryConditionType::Rigid,
        BoundaryConditionType::Impedance,
    ] {
        for bm in [false, true] {
            let problem = problem(1, boundary, bm);
            let reference = dense(&problem);
            for preconditioner in [
                CudaHMatrixPreconditioner::Jacobi,
                CudaHMatrixPreconditioner::ClusterBlockJacobi,
            ] {
                for action in [
                    CudaHMatrixAction::Forward,
                    CudaHMatrixAction::Transpose,
                    CudaHMatrixAction::Hermitian,
                ] {
                    let session = solver()
                        .prepare_cuda_hmatrix_linear(&problem, &pool, preconditioner, action)
                        .unwrap();
                    assert_eq!(session.action(), action);
                    check(session, &reference);
                }
            }
        }
    }
}

#[test]
fn native_hmatrix_linear_compressed_adjoint_without_host_certification() {
    let Some(pool) = pool() else { return };
    let problem = problem(2, BoundaryConditionType::Impedance, true);
    let reference = dense(&problem);
    let mut solver = solver();
    solver.compressed_exact_residual_dof_limit = 0;
    for action in [CudaHMatrixAction::Transpose, CudaHMatrixAction::Hermitian] {
        let session = solver
            .prepare_cuda_hmatrix_linear(
                &problem,
                &pool,
                CudaHMatrixPreconditioner::ClusterBlockJacobi,
                action,
            )
            .unwrap();
        assert!(session.low_rank_blocks() > 0);
        check(session, &reference);
    }
}

#[test]
fn native_hmatrix_linear_one_block_oriented_inverse() {
    let Some(pool) = pool() else { return };
    let problem = problem(0, BoundaryConditionType::Impedance, true);
    let reference = dense(&problem);
    for kind in [
        CudaHMatrixAction::Forward,
        CudaHMatrixAction::Transpose,
        CudaHMatrixAction::Hermitian,
    ] {
        let mut session = solver()
            .prepare_cuda_hmatrix_linear(
                &problem,
                &pool,
                CudaHMatrixPreconditioner::ClusterBlockJacobi,
                kind,
            )
            .unwrap();
        // Twenty panels fit in one self-block: the oriented inverse is exact.
        let n = session.num_dofs();
        assert_eq!(n, 20);
        let exact: Vec<_> = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), (0.7 * i as f64).cos()))
            .collect();
        let mut rhs = pool.try_alloc(n).unwrap();
        pool.backend()
            .try_copy_h2d(&action(&reference, &exact, kind), rhs.raw_mut())
            .unwrap();
        let report = session
            .solve_device(rhs.raw(), DeviceInitialGuess::Zero)
            .unwrap();
        assert_eq!(
            report.iterations, 1,
            "oriented block inverse must solve in one step: {kind:?}"
        );
        let mut actual = vec![Complex64::default(); n];
        pool.backend()
            .try_copy_d2h(session.solution_device().unwrap(), &mut actual)
            .unwrap();
        assert!(
            actual
                .iter()
                .zip(&exact)
                .all(|(a, b)| (*a - *b).norm() < 1e-10)
        );
        let before = pool.backend().begin_runtime_stats();
        let warm = session
            .solve_device(rhs.raw(), DeviceInitialGuess::PreviousSolution)
            .unwrap();
        assert_eq!(warm.iterations, 0);
        assert_eq!(
            pool.backend()
                .runtime_stats_since(before)
                .allocation_operations,
            0
        );
    }
}

#[test]
fn native_hmatrix_linear_rejects_bad_rhs_and_unpublishes_failed_solves() {
    let Some(pool) = pool() else { return };
    let problem = problem(1, BoundaryConditionType::Rigid, true);
    let reference = dense(&problem);
    let mut session = solver()
        .prepare_cuda_hmatrix_linear(
            &problem,
            &pool,
            CudaHMatrixPreconditioner::ClusterBlockJacobi,
            CudaHMatrixAction::Hermitian,
        )
        .unwrap();
    let n = session.num_dofs();
    let mut rhs = pool.try_alloc(n).unwrap();
    let host = action(
        &reference,
        &vec![Complex64::new(0.2, -0.3); n],
        session.action(),
    );
    pool.backend().try_copy_h2d(&host, rhs.raw_mut()).unwrap();
    session
        .solve_device(rhs.raw(), DeviceInitialGuess::Zero)
        .unwrap();
    let wrong_size = pool.try_alloc::<Complex64>(n - 1).unwrap();
    let foreign_pool = MemoryPool::new(
        CudaBackend::new_default().unwrap(),
        MemoryPoolConfig::default(),
    )
    .unwrap();
    let foreign = foreign_pool.try_alloc::<Complex64>(n).unwrap();
    for bad in [wrong_size.raw(), foreign.raw()] {
        let before = pool.backend().begin_runtime_stats();
        assert!(session.solve_device(bad, DeviceInitialGuess::Zero).is_err());
        let stats = pool.backend().runtime_stats_since(before);
        assert_eq!(
            stats.kernel_launches
                + stats.allocation_operations
                + stats.transfer.h2d_bytes
                + stats.transfer.d2h_bytes,
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
    assert!(session.solution_device().is_none());
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
    session.inner.exact_tolerance = Some(0.0);
    assert!(matches!(
        session.solve_device(rhs.raw(), DeviceInitialGuess::Zero),
        Err(BemError::ResidualCheckFailed { .. })
    ));
    assert!(session.solution_device().is_none());
    session.inner.exact_tolerance = None;
    session.inner.controls.max_iterations = 1;
    assert!(
        session
            .solve_device(rhs.raw(), DeviceInitialGuess::Zero)
            .is_err()
    );
    assert!(session.solution_device().is_none());
    session.inner.controls.max_iterations = 400;
    pool.backend()
        .try_copy_h2d(&vec![Complex64::default(); n], rhs.raw_mut())
        .unwrap();
    let report = session
        .solve_device(rhs.raw(), DeviceInitialGuess::Zero)
        .unwrap();
    assert_eq!(report.iterations, 0);
    assert_eq!(report.compressed_relative_residual, 0.0);
}

#[path = "bem_solver_cuda_hmatrix_block_tests.rs"]
mod block;
