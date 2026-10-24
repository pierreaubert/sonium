//! Public CUDA CG+AMG dispatch and retained-session checks across scalar types.

// Rust guideline compliant 2026-02-21
use super::*;

fn check_scalar<T>(complex: bool, tolerance: f64)
where
    T: ComplexField + DeviceScalar<Real = <T as ComplexField>::Real>,
    <T as ComplexField>::Real: FromPrimitive + Sync + Send,
{
    let scalar = |re: f64, im: f64| {
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
            entries.extend([
                (i, i + 1, scalar(-1.0, 0.3)),
                (i + 1, i, scalar(-1.0, -0.3)),
            ]);
        }
    }
    let matrix = CsrMatrix::from_triplets(n, n, entries);
    let exact: Vec<_> = (0..n)
        .map(|i| scalar((i as f64 * 0.2).sin(), (i as f64 * 0.3).cos()))
        .collect();
    let rhs: Vec<_> = (0..n)
        .map(|row| {
            (matrix.row_ptrs[row]..matrix.row_ptrs[row + 1])
                .fold(<T as DeviceScalar>::zero(), |sum, j| {
                    sum + matrix.values[j] * exact[matrix.col_indices[j]]
                })
        })
        .collect();
    let config = LinearSolveConfig {
        solver: LinearSolverType::CgAmg,
        tolerance,
        max_iterations: 200,
        ..Default::default()
    };
    let result = solve_with_backend(
        &matrix,
        &rhs,
        &config,
        LinearSolveBackend::Cuda,
        LinearSolveFallback::Reject,
    )
    .unwrap();
    assert_eq!(result.backend.actual, LinearSolveBackend::Cuda);
    assert!(!result.backend.fallback && result.converged);
    assert_eq!(
        result.assumption_provenance,
        AssumptionProvenance::IndependentlyValidated
    );
    assert!(result.residual <= tolerance);
    result.validate_contract().unwrap();
    for (actual, expected) in result.x.iter().zip(&exact) {
        assert!(
            <T as ComplexField>::norm(&(*actual - *expected))
                .to_f64()
                .unwrap()
                < 10.0 * tolerance
        );
    }

    let pool = MemoryPool::new(
        CudaBackend::new(0).unwrap(),
        MemoryPoolConfig::with_cap(1 << 27),
    )
    .unwrap();
    let backend = pool.backend();
    let before = backend.begin_runtime_stats();
    crate::iterative::validate_backend_pcg::<_, T>(backend, true).unwrap();
    let admission = backend.runtime_stats_since(before);
    assert_eq!(admission.allocation_operations, 0);
    assert_eq!(admission.kernel_launches + admission.library_operations, 0);
    assert_eq!(admission.transfer.h2d_bytes + admission.transfer.d2h_bytes, 0);
    let operator =
        PersistentDeviceCsr::try_from_host_csr_ref(backend, matrix.as_host_csr_ref()).unwrap();
    let amg =
        BackendAmgPreconditioner::try_from_csr(backend, &matrix, AmgConfig::for_backend_pcg())
            .unwrap();
    assert!(amg.num_levels() > 1);
    let mut session =
        DevicePcgSession::try_new(&pool, n, DevicePcgAssumption::PositiveDefinite).unwrap();
    let mut device_rhs = pool.try_alloc::<T>(n).unwrap();
    let controls = PcgBackendConfig {
        max_iterations: 200,
        tolerance: <<T as ComplexField>::Real as FromPrimitive>::from_f64(tolerance).unwrap(),
    };
    for update in 0..2 {
        let factor = scalar(1.0 + update as f64 * 0.3, update as f64 * 0.1);
        let changed: Vec<_> = rhs.iter().map(|&v| v * factor).collect();
        backend
            .try_copy_h2d(&changed, device_rhs.raw_mut())
            .unwrap();
        let before = backend.runtime_stats();
        let status = session
            .solve(
                &operator,
                &amg,
                device_rhs.raw(),
                if update == 0 {
                    DeviceInitialGuess::Zero
                } else {
                    DeviceInitialGuess::PreviousSolution
                },
                &controls,
            )
            .unwrap();
        assert!(status.converged);
        let after = backend.runtime_stats();
        if update > 0 {
            assert_eq!(after.allocation_operations, before.allocation_operations);
        }
        assert_eq!(amg.diagnostics().runtime.fallback_count, 0);
        let mut actual = vec![<T as DeviceScalar>::zero(); n];
        backend
            .try_copy_d2h(session.solution().unwrap(), &mut actual)
            .unwrap();
        assert!(true_relative_residual(&matrix, &changed, &actual).unwrap() <= tolerance);
    }
}

#[test]
fn cuda_cg_amg_rejects_complex_symmetric_nonhermitian_input() {
    let matrix = CsrMatrix::from_triplets(
        2,
        2,
        vec![
            (0, 0, Complex64::new(4.0, 0.0)),
            (1, 1, Complex64::new(4.0, 0.0)),
            (0, 1, Complex64::new(-1.0, 0.3)),
            (1, 0, Complex64::new(-1.0, 0.3)),
        ],
    );
    let config = LinearSolveConfig {
        solver: LinearSolverType::CgAmg,
        ..Default::default()
    };
    let error = solve_with_backend(
        &matrix,
        &[Complex64::new(1.0, 0.0); 2],
        &config,
        LinearSolveBackend::Cuda,
        LinearSolveFallback::Reject,
    )
    .unwrap_err();
    assert!(
        matches!(error, LinearSolveError::InvalidConfig(_)),
        "{error}"
    );
}

#[test]
fn native_cuda_cg_amg_facade_and_persistent_sessions() {
    if let Err(error) = CudaBackend::new(0) {
        assert!(
            std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
            "CUDA required: {error}"
        );
        eprintln!("SKIP CUDA CG+AMG: {error}");
        return;
    }
    check_scalar::<f32>(false, 2e-5);
    check_scalar::<f64>(false, 1e-10);
    check_scalar::<Complex32>(true, 2e-5);
    check_scalar::<Complex64>(true, 1e-10);
}
