//! Native public CUDA sessions retain setup across host and device right-hand sides.

// Rust guideline compliant 2026-02-21
use super::*;
use sonium_compute::CudaFailureInjectionPoint;

fn check_scalar<T>(pool: &MemoryPool<CudaBackend>, complex: bool, tolerance: f64)
where
    T: ComplexField + DeviceScalar<Real = <T as ComplexField>::Real>,
    <T as ComplexField>::Real: FromPrimitive + Sync + Send,
{
    let scalar = |re, im| {
        <T as ComplexField>::from_re_im(
            <<T as ComplexField>::Real as FromPrimitive>::from_f64(re).unwrap(),
            <<T as ComplexField>::Real as FromPrimitive>::from_f64(if complex { im } else { 0.0 })
                .unwrap(),
        )
    };
    let n = 96;
    let mut entries = Vec::new();
    for i in 0..n {
        entries.push((i, i, scalar(4.0 + (i % 3) as f64, 0.0)));
        if i + 1 < n {
            entries.push((i, i + 1, scalar(-1.0, 0.3)));
            entries.push((i + 1, i, scalar(-1.0, -0.3)));
        }
    }
    let matrix = CsrMatrix::from_triplets(n, n, entries);
    let exact: Vec<_> = (0..n)
        .map(|i| scalar((i as f64 * 0.2).sin(), (i as f64 * 0.3).cos()))
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
        // The facade's Chebyshev contract admits real SPD coefficients even
        // when vectors use a complex scalar representation.
        let mut matrix = matrix.clone();
        if solver == LinearSolverType::GmresChebyshev {
            for value in &mut matrix.values {
                *value = <T as ComplexField>::from_real(value.re());
            }
        }
        let rhs = matrix.apply(&exact);
        let config = LinearSolveConfig {
            solver,
            tolerance,
            max_iterations: 200,
            restart: 40,
            ..Default::default()
        };
        let mut session = CudaLinearSolveSession::try_new(
            pool,
            &matrix,
            &config,
            LinearSystemAssumption::General,
        )
        .unwrap();
        if solver == LinearSolverType::Auto {
            assert_eq!(session.requested_strategy(), Some(LinearSolverType::Auto));
            assert_eq!(
                session.selected_strategy(),
                Some(resolve_auto_solver(
                    solver,
                    &matrix,
                    is_complex_scalar::<T>(),
                ))
            );
            let one_shot = solve_with_backend_and_assumption(
                &matrix,
                &rhs,
                &config,
                LinearSolveBackend::Cuda,
                LinearSolveFallback::Reject,
                LinearSystemAssumption::General,
            )
            .unwrap();
            assert_eq!(one_shot.backend.actual, LinearSolveBackend::Cuda);
            assert!(!one_shot.backend.fallback);
            assert!(one_shot.residual <= tolerance);
        }
        assert!(session.solution_device().is_none());
        assert!(
            session
                .solve_host(&rhs, DeviceInitialGuess::PreviousSolution)
                .is_err()
        );
        session.solve_host(&rhs, DeviceInitialGuess::Zero).unwrap();
        let changed: Vec<_> = rhs.iter().map(|&v| v * scalar(1.3, 0.1)).collect();
        let before = backend.begin_runtime_stats();
        let result = session
            .solve_host(&changed, DeviceInitialGuess::PreviousSolution)
            .unwrap();
        let stats = backend.runtime_stats_since(before);
        assert_eq!(stats.allocation_operations, 0, "{solver:?}");
        result.validate_contract().unwrap();
        assert!(result.residual <= tolerance);
        assert!(!result.backend.fallback);
        for (actual, expected) in result.x.iter().zip(&exact) {
            assert!(
                (*actual - *expected * scalar(1.3, 0.1))
                    .norm()
                    .to_f64()
                    .unwrap()
                    < tolerance * 10.0
            );
        }
        let mut device_rhs = pool.try_alloc::<T>(n).unwrap();
        backend
            .try_copy_h2d(&changed, device_rhs.raw_mut())
            .unwrap();
        let before = backend.begin_runtime_stats();
        let status = session
            .solve_device(device_rhs.raw(), DeviceInitialGuess::Zero)
            .unwrap();
        assert!(status.residual <= tolerance);
        let device_stats = backend.runtime_stats_since(before);
        assert_eq!(device_stats.allocation_operations, 0, "{solver:?}");
        // The same zero-start host solve differs only by the explicit RHS and
        // solution transfers. Small algorithm control traffic is identical.
        let before = backend.begin_runtime_stats();
        let host = session
            .solve_host(&changed, DeviceInitialGuess::Zero)
            .unwrap();
        let host_stats = backend.runtime_stats_since(before);
        assert_eq!(host.iterations, status.iterations);
        let vector_bytes = n * std::mem::size_of::<T>();
        assert_eq!(
            host_stats.transfer.h2d_bytes,
            device_stats.transfer.h2d_bytes + vector_bytes
        );
        assert_eq!(
            host_stats.transfer.d2h_bytes,
            device_stats.transfer.d2h_bytes + vector_bytes
        );
        assert!(session.solve_host(&[], DeviceInitialGuess::Zero).is_err());
        assert!(session.solution_device().is_none());
        assert!(
            session
                .solve_device(device_rhs.raw(), DeviceInitialGuess::PreviousSolution)
                .is_err()
        );
        session
            .solve_device(device_rhs.raw(), DeviceInitialGuess::Zero)
            .unwrap();
        let bad = pool.try_alloc::<T>(1).unwrap();
        assert!(
            session
                .solve_device(bad.raw(), DeviceInitialGuess::Zero)
                .is_err()
        );
        assert!(session.solution_device().is_none());
        session.solve_host(&rhs, DeviceInitialGuess::Zero).unwrap();
        backend.inject_failure(CudaFailureInjectionPoint::Launch);
        assert!(
            session
                .solve_device(device_rhs.raw(), DeviceInitialGuess::Zero)
                .is_err()
        );
        assert!(session.solution_device().is_none());
        backend.recover_after_error().unwrap();
        assert!(
            session
                .solve_device(device_rhs.raw(), DeviceInitialGuess::PreviousSolution)
                .is_err()
        );
        session.solve_host(&rhs, DeviceInitialGuess::Zero).unwrap();
        let zero = vec![<T as DeviceScalar>::zero(); n];
        let result = session.solve_host(&zero, DeviceInitialGuess::Zero).unwrap();
        assert_eq!(result.residual, 0.0);
    }
}

#[test]
fn native_cuda_public_retained_sessions_all_strategies_and_scalars() {
    let backend = match CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            assert!(
                std::env::var_os("SONIUM_REQUIRE_CUDA").is_none(),
                "CUDA required: {error}"
            );
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 28)).unwrap();
    check_scalar::<f32>(&pool, false, 2e-5);
    check_scalar::<f64>(&pool, false, 1e-10);
    check_scalar::<Complex32>(&pool, true, 2e-5);
    check_scalar::<Complex64>(&pool, true, 1e-10);
}

fn check_auto_large<T>(pool: &MemoryPool<CudaBackend>, tolerance: f64)
where
    T: ComplexField + DeviceScalar<Real = <T as ComplexField>::Real>,
    <T as ComplexField>::Real: FromPrimitive + Sync + Send,
{
    let n = large_problem_rows().max(4);
    assert!(
        n <= 100_000,
        "native Auto fixture requires a row cutoff <= 100000"
    );
    // Signed diagonal entries prove that large real Auto does not assume SPD.
    // Real coefficients in complex storage prove selection follows scalar type.
    let mut matrix = CsrMatrix::from_triplets(
        n,
        n,
        (0..n)
            .map(|i| {
                (
                    i,
                    i,
                    <T as ComplexField>::from_real(
                        <<T as ComplexField>::Real as FromPrimitive>::from_f64(if i % 2 == 0 {
                            2.0
                        } else {
                            -3.0
                        })
                        .unwrap(),
                    ),
                )
            })
            .collect(),
    );
    let config = LinearSolveConfig {
        solver: LinearSolverType::Auto,
        tolerance,
        max_iterations: 20,
        restart: 8,
        ..Default::default()
    };
    let expected = if is_complex_scalar::<T>() {
        LinearSolverType::FgmresSchwarz
    } else {
        LinearSolverType::GmresIlu
    };
    assert_eq!(
        resolve_auto_solver(config.solver, &matrix, is_complex_scalar::<T>()),
        expected
    );
    let mut session =
        CudaLinearSolveSession::try_new(pool, &matrix, &config, LinearSystemAssumption::General)
            .unwrap();
    assert_eq!(session.requested_strategy(), Some(LinearSolverType::Auto));
    assert_eq!(session.selected_strategy(), Some(expected));
    let exact = vec![
        <T as ComplexField>::from_re_im(
            <<T as ComplexField>::Real as FromPrimitive>::from_f64(1.0).unwrap(),
            <<T as ComplexField>::Real as FromPrimitive>::from_f64(0.25).unwrap(),
        );
        n
    ];
    let rhs = matrix.apply(&exact);
    let solution = session.solve_host(&rhs, DeviceInitialGuess::Zero).unwrap();
    assert!(solution.residual <= tolerance);
    for (actual, expected) in solution.x.iter().zip(&exact) {
        assert!((*actual - *expected).norm().to_f64().unwrap() < tolerance * 10.0);
    }
    matrix.values.iter_mut().for_each(|value| {
        *value *= <T as ComplexField>::from_real(
            <<T as ComplexField>::Real as FromPrimitive>::from_f64(1.25).unwrap(),
        )
    });
    let before = pool.backend().begin_runtime_stats();
    let update = session.try_update_matrix(&matrix).unwrap();
    assert!(!update.preconditioner_rebuilt);
    assert_eq!(
        pool.backend()
            .runtime_stats_since(before)
            .allocation_operations,
        0
    );
    assert_eq!(session.selected_strategy(), Some(expected));
    assert_eq!(session.requested_strategy(), Some(LinearSolverType::Auto));
    let rhs = matrix.apply(&exact);
    let before = pool.backend().begin_runtime_stats();
    let solution = session
        .solve_host(&rhs, DeviceInitialGuess::PreviousSolution)
        .unwrap();
    assert!(solution.residual <= tolerance);
    assert_eq!(
        pool.backend()
            .runtime_stats_since(before)
            .allocation_operations,
        0
    );
}

#[test]
fn native_cuda_public_auto_large_real_and_complex_selection() {
    let backend = match CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            assert!(
                std::env::var_os("SONIUM_REQUIRE_CUDA").is_none(),
                "CUDA required: {error}"
            );
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 29)).unwrap();
    check_auto_large::<f32>(&pool, 2e-5);
    check_auto_large::<f64>(&pool, 1e-10);
    check_auto_large::<Complex32>(&pool, 2e-5);
    check_auto_large::<Complex64>(&pool, 1e-10);
}

#[test]
fn native_cuda_public_session_admission_and_failed_publication() {
    let backend = match CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            assert!(
                std::env::var_os("SONIUM_REQUIRE_CUDA").is_none(),
                "CUDA required: {error}"
            );
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 24)).unwrap();
    // Retain additive duplicates to exercise canonical upload and independent
    // certification against the original public CSR representation.
    let matrix = CsrMatrix {
        num_rows: 2,
        num_cols: 2,
        row_ptrs: vec![0, 3, 6],
        col_indices: vec![0, 0, 1, 0, 1, 1],
        values: vec![1.0_f64, 3.0, 1.0, 1.0, 1.0, 2.0],
    };
    let config = LinearSolveConfig {
        solver: LinearSolverType::GmresJacobi,
        tolerance: 1e-11,
        ..Default::default()
    };
    let mut session =
        CudaLinearSolveSession::try_new(&pool, &matrix, &config, LinearSystemAssumption::General)
            .unwrap();
    let result = session
        .solve_host(&[6.0, 7.0], DeviceInitialGuess::Zero)
        .unwrap();
    assert!((result.x[0] - 1.0).abs() < 1e-10 && (result.x[1] - 2.0).abs() < 1e-10);
    assert!(
        session
            .solve_host(&[f64::NAN, 1.0], DeviceInitialGuess::Zero)
            .is_err()
    );
    assert!(session.solution_device().is_none());
    assert!(
        session
            .solve_host(&[6.0, 7.0], DeviceInitialGuess::PreviousSolution)
            .is_err()
    );
    session
        .solve_host(&[6.0, 7.0], DeviceInitialGuess::Zero)
        .unwrap();
    let foreign_pool = MemoryPool::new(
        CudaBackend::new(0).unwrap(),
        MemoryPoolConfig::with_cap(1 << 20),
    )
    .unwrap();
    let mut foreign_rhs = foreign_pool.try_alloc::<f64>(2).unwrap();
    foreign_pool
        .backend()
        .try_copy_h2d(&[6.0, 7.0], foreign_rhs.raw_mut())
        .unwrap();
    let before_foreign = pool.backend().begin_runtime_stats();
    assert!(
        session
            .solve_device(foreign_rhs.raw(), DeviceInitialGuess::Zero)
            .is_err()
    );
    let rejected = pool.backend().runtime_stats_since(before_foreign);
    assert_eq!(rejected.kernel_launches, 0);
    assert_eq!(rejected.library_operations, 0);
    assert_eq!(rejected.transfer.d2d_bytes, 0);
    assert_eq!(rejected.allocation_operations, 0);
    assert!(session.solution_device().is_none());
    pool.backend().recover_after_error().unwrap();
    session
        .solve_host(&[6.0, 7.0], DeviceInitialGuess::Zero)
        .unwrap();
    let mut malformed = matrix.clone();
    malformed.num_rows = usize::MAX;
    assert!(
        CudaLinearSolveSession::try_new(
            &pool,
            &malformed,
            &config,
            LinearSystemAssumption::General
        )
        .is_err()
    );
    let indefinite = CsrMatrix::from_triplets(2, 2, vec![(0, 0, -1.0_f64), (1, 1, 2.0)]);
    let cg = LinearSolveConfig {
        solver: LinearSolverType::CgAmg,
        ..config.clone()
    };
    assert!(
        CudaLinearSolveSession::try_new(
            &pool,
            &indefinite,
            &cg,
            LinearSystemAssumption::PositiveDefinite
        )
        .is_err()
    );
    let limited = LinearSolveConfig {
        solver: LinearSolverType::Gmres,
        max_iterations: 1,
        restart: 1,
        ..config
    };
    let mut session =
        CudaLinearSolveSession::try_new(&pool, &matrix, &limited, LinearSystemAssumption::General)
            .unwrap();
    assert!(matches!(
        session.solve_host(&[6.0, 7.0], DeviceInitialGuess::Zero),
        Err(LinearSolveError::NotConverged { .. })
    ));
    assert!(session.solution_device().is_none());
}
