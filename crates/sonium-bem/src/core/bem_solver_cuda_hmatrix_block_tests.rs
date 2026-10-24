//! Production block solves against independently assembled dense references.

// Rust guideline compliant 2026-02-21
use super::*;

fn check_block(
    session: &mut CudaHMatrixLinearSession<'_>,
    matrix: &DenseMatrix<Complex64>,
    width: usize,
    dependent: bool,
) {
    let pool = session.inner.pool;
    let n = session.num_dofs();
    let mut rhs = pool.try_alloc(n * width).unwrap();
    for variant in 0..2 {
        let exact: Vec<_> = (0..n * width)
            .map(|i| {
                let col = i / n;
                if dependent && col == 2 {
                    return Complex64::default();
                }
                let phase = if dependent { 0.0 } else { 0.31 * col as f64 };
                Complex64::new(
                    (0.17 * (i % n) as f64 + phase + variant as f64).sin(),
                    (0.13 * (i % n) as f64 - phase + variant as f64).cos(),
                )
            })
            .collect();
        let host: Vec<_> = exact
            .chunks_exact(n)
            .flat_map(|x| action(matrix, x, session.action()))
            .collect();
        pool.backend().try_copy_h2d(&host, rhs.raw_mut()).unwrap();
        for repeat in 0..2 {
            let before = pool.backend().begin_runtime_stats();
            let report = session
                .solve_block_device(rhs.raw(), width, DeviceInitialGuess::Zero)
                .unwrap();
            let stats = pool.backend().runtime_stats_since(before);
            if repeat == 1 {
                assert_eq!(stats.allocation_operations, 0, "{stats:?}");
            }
            assert!(
                report
                    .compressed_relative_residuals
                    .iter()
                    .all(|r| *r <= 1e-9)
            );
            let mut got = vec![Complex64::default(); n * width];
            pool.backend()
                .try_copy_d2h(session.block_solution_device().unwrap(), &mut got)
                .unwrap();
            for (x, b) in got.chunks_exact(n).zip(host.chunks_exact(n)) {
                let ax = action(matrix, x, session.action());
                let norm = b.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
                let residual = ax
                    .iter()
                    .zip(b)
                    .map(|(a, b)| (*a - *b).norm_sqr())
                    .sum::<f64>()
                    .sqrt()
                    / if norm > 0.0 { norm } else { 1.0 };
                assert!(residual < 3e-7, "{residual}");
            }
            eprintln!(
                "BEM block n={n} action={:?} width={width} dependent={dependent} variant={variant} repeat={repeat} iterations={} restarts={} allocations={}",
                session.action(),
                report.iterations,
                report.restarts,
                stats.allocation_operations
            );
        }
        let warm = session
            .solve_block_device(rhs.raw(), width, DeviceInitialGuess::PreviousSolution)
            .unwrap();
        assert_eq!(warm.iterations, 0);
    }
}

#[test]
fn native_hmatrix_block_sessions_orientations_and_preconditioners() {
    let Some(pool) = pool() else {
        return;
    };
    let problem = problem(1, BoundaryConditionType::Impedance, true);
    let matrix = dense(&problem);
    let mut solver = solver();
    solver.gmres_restart = 3;
    for preconditioner in [
        CudaHMatrixPreconditioner::Jacobi,
        CudaHMatrixPreconditioner::ClusterBlockJacobi,
    ] {
        for kind in [
            CudaHMatrixAction::Forward,
            CudaHMatrixAction::Transpose,
            CudaHMatrixAction::Hermitian,
        ] {
            let mut session = solver
                .prepare_cuda_hmatrix_block(&problem, &pool, preconditioner, kind)
                .unwrap();
            check_block(&mut session, &matrix, 2, false);
            check_block(&mut session, &matrix, 3, true);
        }
    }
}

#[test]
fn native_hmatrix_block_low_rank_resident_session() {
    let Some(pool) = pool() else {
        return;
    };
    let problem = problem(2, BoundaryConditionType::Rigid, true);
    let matrix = dense(&problem);
    let mut solver = solver();
    solver.compressed_exact_residual_dof_limit = 0;
    solver.gmres_restart = 5;
    let mut session = solver
        .prepare_cuda_hmatrix_linear(
            &problem,
            &pool,
            CudaHMatrixPreconditioner::ClusterBlockJacobi,
            CudaHMatrixAction::Hermitian,
        )
        .unwrap();
    assert!(session.low_rank_blocks() > 0);
    check_block(&mut session, &matrix, 2, false);
}

#[test]
fn native_hmatrix_block_failure_publication_and_recovery() {
    let Some(pool) = pool() else {
        return;
    };
    let problem = problem(1, BoundaryConditionType::Rigid, true);
    let mut session = solver()
        .prepare_cuda_hmatrix_linear(
            &problem,
            &pool,
            CudaHMatrixPreconditioner::Jacobi,
            CudaHMatrixAction::Forward,
        )
        .unwrap();
    let n = session.num_dofs();
    let mut rhs = pool.try_alloc(n * 2).unwrap();
    let host: Vec<_> = (0..n * 2)
        .map(|i| Complex64::new((i as f64 * 0.17).sin(), (i as f64 * 0.11).cos()))
        .collect();
    pool.backend().try_copy_h2d(&host, rhs.raw_mut()).unwrap();
    session
        .solve_block_device(rhs.raw(), 2, DeviceInitialGuess::Zero)
        .unwrap();
    for width in [0, 1, usize::MAX] {
        assert!(
            session
                .solve_block_device(rhs.raw(), width, DeviceInitialGuess::Zero)
                .is_err()
        );
        assert!(session.block_solution_device().is_none());
        assert!(
            session
                .solve_block_device(rhs.raw(), 2, DeviceInitialGuess::PreviousSolution)
                .is_err()
        );
        session
            .solve_block_device(rhs.raw(), 2, DeviceInitialGuess::Zero)
            .unwrap();
    }
    let foreign = MemoryPool::new(
        CudaBackend::new_default().unwrap(),
        MemoryPoolConfig::default(),
    )
    .unwrap();
    let foreign_rhs = foreign.try_alloc(n * 2).unwrap();
    assert!(
        session
            .solve_block_device(foreign_rhs.raw(), 2, DeviceInitialGuess::Zero)
            .is_err()
    );
    pool.backend()
        .try_copy_h2d(&vec![Complex64::new(f64::NAN, 0.0); n * 2], rhs.raw_mut())
        .unwrap();
    assert!(
        session
            .solve_block_device(rhs.raw(), 2, DeviceInitialGuess::Zero)
            .is_err()
    );
    pool.backend().try_copy_h2d(&host, rhs.raw_mut()).unwrap();
    pool.backend()
        .inject_failure(CudaFailureInjectionPoint::Launch);
    assert!(
        session
            .solve_block_device(rhs.raw(), 2, DeviceInitialGuess::Zero)
            .is_err()
    );
    assert!(session.block_solution_device().is_none());
    pool.backend().recover_after_error().unwrap();
    session
        .solve_block_device(rhs.raw(), 2, DeviceInitialGuess::Zero)
        .unwrap();
    session.inner.exact_tolerance = Some(0.0);
    assert!(matches!(
        session.solve_block_device(rhs.raw(), 2, DeviceInitialGuess::Zero),
        Err(BemError::ResidualCheckFailed { .. })
    ));
    assert!(session.block_solution_device().is_none());
    session.inner.exact_tolerance = None;
    session.inner.controls.max_iterations = 1;
    assert!(
        session
            .solve_block_device(rhs.raw(), 2, DeviceInitialGuess::Zero)
            .is_err()
    );
    assert!(session.block_solution_device().is_none());
    session.inner.controls.max_iterations = 400;
    pool.backend()
        .try_copy_h2d(&vec![Complex64::default(); n * 2], rhs.raw_mut())
        .unwrap();
    let zero = session
        .solve_block_device(rhs.raw(), 2, DeviceInitialGuess::Zero)
        .unwrap();
    assert_eq!(zero.iterations, 0);
    assert_eq!(zero.compressed_relative_residuals, vec![0.0; 2]);
    let mut scalar_rhs = pool.try_alloc(n).unwrap();
    pool.backend()
        .try_copy_h2d(&host[..n], scalar_rhs.raw_mut())
        .unwrap();
    session
        .solve_device(scalar_rhs.raw(), DeviceInitialGuess::Zero)
        .unwrap();
    assert!(session.block_solution_device().is_none());
    assert!(session.solution_device().is_some());
    assert!(
        session
            .solve_block_device(rhs.raw(), 2, DeviceInitialGuess::PreviousSolution)
            .is_err()
    );
    assert!(session.solution_device().is_none());
}

#[test]
fn native_hmatrix_block_preparation_rejects_poisoned_context() {
    let Some(pool) = pool() else { return };
    let backend = pool.backend();
    backend.inject_failure(CudaFailureInjectionPoint::Synchronization);
    let original = backend.try_synchronize().unwrap_err().to_string();
    let before = backend.begin_runtime_stats();
    let error = solver()
        .prepare_cuda_hmatrix_block(
            &problem(1, BoundaryConditionType::Rigid, true),
            &pool,
            CudaHMatrixPreconditioner::ClusterBlockJacobi,
            CudaHMatrixAction::Hermitian,
        )
        .unwrap_err();
    assert!(error.to_string().contains(&original), "{error}");
    let stats = backend.runtime_stats_since(before);
    assert_eq!(stats.allocation_operations, 0);
    assert_eq!(stats.kernel_launches + stats.library_operations, 0);
    assert_eq!(
        stats.transfer.h2d_bytes + stats.transfer.d2h_bytes + stats.transfer.d2d_bytes,
        0
    );
    assert!(backend.recover_after_error().is_err());
}
