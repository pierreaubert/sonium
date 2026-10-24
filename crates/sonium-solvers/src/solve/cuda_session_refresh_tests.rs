//! Public numeric refresh retains storage and rejects partially updated systems.

// Rust guideline compliant 2026-02-21
use super::*;
use sonium_compute::{CudaFailureInjectionPoint, PreconditionerFailureInjectionPoint};

fn pool() -> Option<MemoryPool<CudaBackend>> {
    match CudaBackend::new(0) {
        Ok(backend) => Some(MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 28)).unwrap()),
        Err(error) => {
            assert!(
                std::env::var_os("SONIUM_REQUIRE_CUDA").is_none(),
                "CUDA required: {error}"
            );
            None
        }
    }
}

fn tridiagonal<T>(n: usize, complex: bool) -> CsrMatrix<T>
where
    T: ComplexField,
    T::Real: FromPrimitive,
{
    let scalar = |re, im| {
        T::from_re_im(
            T::Real::from_f64(re).unwrap(),
            T::Real::from_f64(if complex { im } else { 0.0 }).unwrap(),
        )
    };
    let mut entries = Vec::new();
    for i in 0..n {
        entries.push((i, i, scalar(4.0, 0.0)));
        if i + 1 < n {
            entries.push((i, i + 1, scalar(-1.0, 0.2)));
            entries.push((i + 1, i, scalar(-1.0, -0.2)));
        }
    }
    CsrMatrix::from_triplets(n, n, entries)
}

fn all_strategies<T>(pool: &MemoryPool<CudaBackend>, complex: bool, tolerance: f64)
where
    T: ComplexField + DeviceScalar<Real = <T as ComplexField>::Real>,
    <T as ComplexField>::Real: FromPrimitive + Sync + Send,
{
    let original = tridiagonal::<T>(96, complex);
    let exact: Vec<_> = (0..96)
        .map(|i| {
            <T as ComplexField>::from_real(
                <<T as ComplexField>::Real as FromPrimitive>::from_f64((i as f64 * 0.2).sin())
                    .unwrap(),
            )
        })
        .collect();
    let backend = pool.backend();
    for solver in [
        LinearSolverType::Auto,
        LinearSolverType::Gmres,
        LinearSolverType::GmresJacobi,
        LinearSolverType::GmresIlu,
        LinearSolverType::GmresSchwarz,
        LinearSolverType::FgmresSchwarz,
        LinearSolverType::GmresAmg,
        LinearSolverType::GmresChebyshev,
        LinearSolverType::Cg,
        LinearSolverType::CgAmg,
    ] {
        let original = if solver == LinearSolverType::GmresChebyshev {
            tridiagonal::<T>(96, false)
        } else {
            original.clone()
        };
        let config = LinearSolveConfig {
            solver,
            tolerance,
            max_iterations: 200,
            restart: 40,
            ..Default::default()
        };
        let mut session = CudaLinearSolveSession::try_new(
            pool,
            &original,
            &config,
            LinearSystemAssumption::General,
        )
        .unwrap();
        session
            .solve_host(&original.apply(&exact), DeviceInitialGuess::Zero)
            .unwrap();
        for scale in [1.125, 0.75] {
            let mut changed = original.clone();
            let factor = <T as ComplexField>::from_real(
                <<T as ComplexField>::Real as FromPrimitive>::from_f64(scale).unwrap(),
            );
            changed.values.iter_mut().for_each(|v| *v *= factor);
            let rhs = changed.apply(&exact);
            let before = backend.begin_runtime_stats();
            let update = session.try_update_matrix(&changed).unwrap();
            let stats = backend.runtime_stats_since(before);
            assert!(!update.preconditioner_rebuilt, "{solver:?}");
            assert_eq!(stats.allocation_operations, 0, "{solver:?}");
            assert_eq!(stats.transfer.d2h_bytes, 0);
            assert!(stats.transfer.h2d_bytes > 0);
            assert!(session.solution_device().is_none());
            let warm = session
                .solve_host(&rhs, DeviceInitialGuess::PreviousSolution)
                .unwrap();
            assert!(warm.residual <= tolerance);
            let before = backend.begin_runtime_stats();
            let actual = session.solve_host(&rhs, DeviceInitialGuess::Zero).unwrap();
            assert_eq!(backend.runtime_stats_since(before).allocation_operations, 0);
            let mut fresh = CudaLinearSolveSession::try_new(
                pool,
                &changed,
                &config,
                LinearSystemAssumption::General,
            )
            .unwrap();
            let expected = fresh.solve_host(&rhs, DeviceInitialGuess::Zero).unwrap();
            assert_eq!(actual.iterations, expected.iterations, "{solver:?}");
            eprintln!(
                "CUDA refresh scalar={} solver={solver:?} scale={scale} rebuilt={} alloc={} h2d={} d2h={} iterations={} residual={}",
                std::any::type_name::<T>(),
                update.preconditioner_rebuilt,
                stats.allocation_operations,
                stats.transfer.h2d_bytes,
                stats.transfer.d2h_bytes,
                actual.iterations,
                actual.residual,
            );
            for ((a, b), x) in actual.x.iter().zip(expected.x).zip(&exact) {
                assert!((*a - b).norm().to_f64().unwrap() < tolerance * 10.0);
                assert!((*a - *x).norm().to_f64().unwrap() < tolerance * 10.0);
            }
        }
    }
}

#[test]
fn native_cuda_public_numeric_refresh_all_strategies_and_scalars() {
    let Some(pool) = pool() else {
        return;
    };
    all_strategies::<f32>(&pool, false, 2e-5);
    all_strategies::<f64>(&pool, false, 1e-10);
    all_strategies::<Complex32>(&pool, true, 2e-5);
    all_strategies::<Complex64>(&pool, true, 1e-10);
}

#[test]
fn native_cuda_public_chebyshev_rejects_nonreal_coefficients_before_upload() {
    let Some(pool) = pool() else {
        return;
    };
    let matrix = tridiagonal::<Complex64>(32, false);
    let config = LinearSolveConfig {
        solver: LinearSolverType::GmresChebyshev,
        tolerance: 1e-10,
        ..Default::default()
    };
    let nonreal = tridiagonal::<Complex64>(32, true);
    let before = pool.backend().begin_runtime_stats();
    assert!(
        CudaLinearSolveSession::try_new(&pool, &nonreal, &config, LinearSystemAssumption::General,)
            .is_err()
    );
    let stats = pool.backend().runtime_stats_since(before);
    assert_eq!(stats.allocation_operations, 0);
    assert_eq!(stats.transfer.h2d_bytes, 0);
    let mut session =
        CudaLinearSolveSession::try_new(&pool, &matrix, &config, LinearSystemAssumption::General)
            .unwrap();
    let exact = vec![Complex64::new(1.0, 0.5); 32];
    let rhs = matrix.apply(&exact);
    session.solve_host(&rhs, DeviceInitialGuess::Zero).unwrap();
    let before = pool.backend().begin_runtime_stats();
    assert!(session.try_update_matrix(&nonreal).is_err());
    let stats = pool.backend().runtime_stats_since(before);
    assert_eq!(stats.allocation_operations, 0);
    assert_eq!(stats.transfer.h2d_bytes, 0);
    assert!(session.solution_device().is_none());
    assert!(
        session
            .solve_host(&rhs, DeviceInitialGuess::PreviousSolution)
            .is_err()
    );
    let recovered = session.solve_host(&rhs, DeviceInitialGuess::Zero).unwrap();
    assert!(recovered.residual <= config.tolerance);
    for (actual, expected) in recovered.x.iter().zip(exact) {
        assert!((*actual - expected).norm() < 1e-8);
    }
    let one_shot = solve_with_backend_and_assumption(
        &matrix,
        &rhs,
        &config,
        LinearSolveBackend::Cuda,
        LinearSolveFallback::Reject,
        LinearSystemAssumption::General,
    )
    .unwrap();
    assert!(one_shot.residual <= config.tolerance);
    assert!(!one_shot.backend.fallback);
}

#[test]
fn native_cuda_public_numeric_refresh_rebuild_and_recovery() {
    let Some(pool) = pool() else {
        return;
    };
    let original = tridiagonal::<f64>(128, false);
    let backend = pool.backend();
    for solver in [
        LinearSolverType::GmresJacobi,
        LinearSolverType::GmresIlu,
        LinearSolverType::GmresSchwarz,
        LinearSolverType::FgmresSchwarz,
        LinearSolverType::CgAmg,
        LinearSolverType::GmresChebyshev,
    ] {
        let config = LinearSolveConfig {
            solver,
            tolerance: 1e-10,
            ..Default::default()
        };
        let mut session = CudaLinearSolveSession::try_new(
            &pool,
            &original,
            &config,
            LinearSystemAssumption::General,
        )
        .unwrap();
        let rhs = original.apply(&vec![1.0; 128]);
        session.solve_host(&rhs, DeviceInitialGuess::Zero).unwrap();
        let mut changed = original.clone();
        changed.values.iter_mut().for_each(|v| *v *= 1.125);
        for failure in 0..2 {
            match failure {
                0 => backend.inject_failure(CudaFailureInjectionPoint::TransferH2D),
                _ => backend.inject_preconditioner_failure(
                    PreconditionerFailureInjectionPoint::DiagonalUpload,
                ),
            }
            assert!(session.try_update_matrix(&changed).is_err());
            assert!(session.solution_device().is_none());
            backend.recover_after_error().unwrap();
            assert!(session.solve_host(&rhs, DeviceInitialGuess::Zero).is_err());
            session.try_update_matrix(&original).unwrap();
            assert!(
                session
                    .solve_host(&rhs, DeviceInitialGuess::PreviousSolution)
                    .is_err()
            );
            let result = session.solve_host(&rhs, DeviceInitialGuess::Zero).unwrap();
            assert!(result.x.iter().all(|x| (x - 1.0).abs() < 1e-8));
        }
        let different_pattern =
            CsrMatrix::from_triplets(128, 128, (0..128).map(|i| (i, i, 4.0)).collect());
        let before = backend.begin_runtime_stats();
        assert!(session.try_update_matrix(&different_pattern).is_err());
        assert_eq!(backend.runtime_stats_since(before).transfer.h2d_bytes, 0);
        assert!(session.solution_device().is_none());
        session.solve_host(&rhs, DeviceInitialGuess::Zero).unwrap();
        let mut invalid = original.clone();
        invalid.values[0] = f64::NAN;
        assert!(session.try_update_matrix(&invalid).is_err());
        session.solve_host(&rhs, DeviceInitialGuess::Zero).unwrap();
        if matches!(
            solver,
            LinearSolverType::CgAmg | LinearSolverType::GmresChebyshev
        ) {
            let mut indefinite = original.clone();
            indefinite.values[0] = -4.0;
            let before = backend.begin_runtime_stats();
            assert!(session.try_update_matrix(&indefinite).is_err());
            assert_eq!(backend.runtime_stats_since(before).transfer.h2d_bytes, 0);
            session.solve_host(&rhs, DeviceInitialGuess::Zero).unwrap();
        }
        // Fixed physical sparsity, but removing strong edges changes AMG levels.
        let mut diagonal = original.clone();
        for row in 0..128 {
            for j in diagonal.row_ptrs[row]..diagonal.row_ptrs[row + 1] {
                if diagonal.col_indices[j] != row {
                    diagonal.values[j] = 0.0;
                }
            }
        }
        let update = session.try_update_matrix(&diagonal).unwrap();
        assert_eq!(
            update.preconditioner_rebuilt,
            solver == LinearSolverType::CgAmg
        );
        assert!(session.solution_device().is_none());
        let result = session
            .solve_host(&vec![4.0; 128], DeviceInitialGuess::PreviousSolution)
            .unwrap();
        assert!(result.x.iter().all(|x| (x - 1.0).abs() < 1e-8));
        let mut fresh = CudaLinearSolveSession::try_new(
            &pool,
            &diagonal,
            &config,
            LinearSystemAssumption::General,
        )
        .unwrap();
        let expected = fresh
            .solve_host(&vec![4.0; 128], DeviceInitialGuess::Zero)
            .unwrap();
        let actual = session
            .solve_host(&vec![4.0; 128], DeviceInitialGuess::Zero)
            .unwrap();
        assert_eq!(actual.iterations, expected.iterations);
    }
}

#[test]
fn native_cuda_public_numeric_refresh_canonical_duplicate_layout() {
    let Some(pool) = pool() else {
        return;
    };
    let matrix = CsrMatrix::from_triplets(
        2,
        2,
        vec![(0, 0, 4.0_f64), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)],
    );
    let config = LinearSolveConfig {
        solver: LinearSolverType::GmresJacobi,
        tolerance: 1e-11,
        ..Default::default()
    };
    let mut session =
        CudaLinearSolveSession::try_new(&pool, &matrix, &config, LinearSystemAssumption::General)
            .unwrap();
    session
        .solve_host(&[6.0, 7.0], DeviceInitialGuess::Zero)
        .unwrap();
    let duplicates = CsrMatrix {
        num_rows: 2,
        num_cols: 2,
        row_ptrs: vec![0, 3, 6],
        col_indices: vec![1, 0, 0, 1, 0, 1],
        values: vec![2.0, 3.0, 5.0, 2.0, 2.0, 4.0],
    };
    let before = pool.backend().begin_runtime_stats();
    let update = session.try_update_matrix(&duplicates).unwrap();
    assert!(!update.preconditioner_rebuilt);
    assert_eq!(
        pool.backend()
            .runtime_stats_since(before)
            .allocation_operations,
        0
    );
    assert!(session.solution_device().is_none());
    let result = session
        .solve_host(&[12.0, 14.0], DeviceInitialGuess::PreviousSolution)
        .unwrap();
    assert!((result.x[0] - 1.0).abs() < 1e-10 && (result.x[1] - 2.0).abs() < 1e-10);
}

#[test]
fn native_cuda_public_numeric_refresh_terminal_sync_requires_recreate() {
    let Some(failed_pool) = pool() else {
        return;
    };
    let original = tridiagonal::<f64>(96, false);
    let config = LinearSolveConfig {
        solver: LinearSolverType::CgAmg,
        tolerance: 1e-10,
        ..Default::default()
    };
    let mut session = CudaLinearSolveSession::try_new(
        &failed_pool,
        &original,
        &config,
        LinearSystemAssumption::General,
    )
    .unwrap();
    let rhs = original.apply(&vec![1.0; 96]);
    session.solve_host(&rhs, DeviceInitialGuess::Zero).unwrap();
    let mut changed = original.clone();
    changed.values.iter_mut().for_each(|v| *v *= 1.125);
    failed_pool
        .backend()
        .inject_failure(CudaFailureInjectionPoint::Synchronization);
    assert!(session.try_update_matrix(&changed).is_err());
    assert!(session.solution_device().is_none());
    // Synchronization failure is terminal in the backend contract. Retrying
    // uploads cannot make this session usable, even if all copies were issued.
    assert!(failed_pool.backend().recover_after_error().is_err());
    assert!(session.try_update_matrix(&original).is_err());
    let before = failed_pool.backend().begin_runtime_stats();
    assert!(session.solve_host(&rhs, DeviceInitialGuess::Zero).is_err());
    let stats = failed_pool.backend().runtime_stats_since(before);
    assert_eq!(stats.transfer.h2d_bytes, 0);
    assert_eq!(stats.kernel_launches, 0);
    assert_eq!(stats.library_operations, 0);
    let fresh_pool = pool().unwrap();
    let mut fresh = CudaLinearSolveSession::try_new(
        &fresh_pool,
        &changed,
        &config,
        LinearSystemAssumption::General,
    )
    .unwrap();
    let result = fresh
        .solve_host(&changed.apply(&vec![1.0; 96]), DeviceInitialGuess::Zero)
        .unwrap();
    assert!(result.x.iter().all(|x| (x - 1.0).abs() < 1e-8));
}
