#[cfg(feature = "cuda")]
use num_complex::Complex32;
use num_complex::Complex64;
use sonium_compute::{Backend, CpuBackend, DevicePreconditioner};
use sonium_solvers_preconditioners::{
    AdditiveSchwarzPreconditioner, AmgConfig, AmgPreconditioner, AmgSmoother,
    BackendAmgPreconditioner, BlockJacobiPreconditioner, ChebyshevConfig, ChebyshevError,
    ChebyshevPreconditioner, GeneoConfig, GeneoSetupError, HiluPreconditioner, IluPreconditioner,
    IlutConfig, IlutPreconditioner, MaterialSocConfig, SaddlePointConfig,
    SaddlePointPreconditioner, SchurStrategy, SchwarzCoarseSetup,
};
use sonium_solvers_sparse::CsrMatrix;

#[test]
fn geneo_generated_non_finite_local_row_falls_back_without_panic() {
    let matrix = CsrMatrix::from_triplets(
        3,
        3,
        vec![
            (0, 0, 1.0_f64),
            (0, 1, f64::MAX),
            (0, 2, f64::MAX),
            (1, 1, 1.0),
            (2, 2, 1.0),
        ],
    );
    let partitions = vec![vec![0], vec![1, 2]];
    let setup = std::panic::catch_unwind(|| {
        AdditiveSchwarzPreconditioner::from_csr_partitions(&matrix, &partitions, 1)
            .with_geneo_coarse_space(&matrix, &GeneoConfig::default())
    });
    let preconditioner =
        setup.expect("finite input may overflow local GenEO assembly but must not panic");
    assert_eq!(
        preconditioner.coarse_setup(),
        SchwarzCoarseSetup::FellBackToOneLevel
    );
    assert!(!preconditioner.has_coarse_space());
}

#[test]
fn checked_geneo_setup_attaches_and_reports_typed_failures() {
    let matrix = CsrMatrix::from_triplets(
        2,
        2,
        vec![(0, 0, 2.0_f64), (0, 1, -1.0), (1, 0, -1.0), (1, 1, 2.0)],
    );
    let partitions = vec![vec![0], vec![1]];
    let config = GeneoConfig {
        tau: 2.0,
        ..Default::default()
    };
    let preconditioner =
        AdditiveSchwarzPreconditioner::try_from_csr_partitions(&matrix, &partitions, 0)
            .expect("valid Schwarz setup")
            .try_with_geneo_coarse_space(&matrix, &config)
            .expect("checked GenEO setup should attach");
    assert_eq!(preconditioner.coarse_setup(), SchwarzCoarseSetup::Attached);
    assert!(preconditioner.coarse_dimension() > 0);

    let invalid = GeneoConfig {
        tau: f64::NAN,
        ..Default::default()
    };
    let error = AdditiveSchwarzPreconditioner::try_from_csr_partitions(&matrix, &partitions, 0)
        .expect("valid Schwarz setup")
        .try_with_geneo_coarse_space(&matrix, &invalid)
        .expect_err("NaN GenEO configuration must be rejected");
    assert!(matches!(error, GeneoSetupError::InvalidConfig(_)));

    let empty = GeneoConfig {
        tau: -1.0e9,
        ..Default::default()
    };
    let error = AdditiveSchwarzPreconditioner::try_from_csr_partitions(&matrix, &partitions, 0)
        .expect("valid Schwarz setup")
        .try_with_geneo_coarse_space(&matrix, &empty)
        .expect_err("an empty checked coarse space must be explicit");
    assert!(matches!(error, GeneoSetupError::EmptyCoarseSpace));
}

#[test]
fn checked_geneo_setup_reports_generated_non_finite_values() {
    let matrix = CsrMatrix::from_triplets(
        3,
        3,
        vec![
            (0, 0, 1.0_f64),
            (0, 1, f64::MAX),
            (0, 2, f64::MAX),
            (1, 1, 1.0),
            (2, 2, 1.0),
        ],
    );
    let error =
        AdditiveSchwarzPreconditioner::from_csr_partitions(&matrix, &[vec![0], vec![1, 2]], 1)
            .try_with_geneo_coarse_space(&matrix, &GeneoConfig::default())
            .expect_err("overflowed local GenEO assembly must be typed");
    assert!(
        matches!(error, GeneoSetupError::GeneratedNonFinite { .. }),
        "unexpected checked GenEO error: {error:?}"
    );
}

#[test]
fn checked_chebyshev_rejects_overflowed_spectral_estimate() {
    let scale = f64::MAX;
    let off_diagonal = 0.49 * scale;
    let matrix = CsrMatrix::from_triplets(
        3,
        3,
        vec![
            (0, 0, scale),
            (1, 1, scale),
            (2, 2, scale),
            (0, 1, off_diagonal),
            (1, 0, off_diagonal),
            (0, 2, off_diagonal),
            (2, 0, off_diagonal),
            (1, 2, off_diagonal),
            (2, 1, off_diagonal),
        ],
    );
    let error = match ChebyshevPreconditioner::try_from_csr(&matrix, ChebyshevConfig::default()) {
        Ok(_) => panic!("overflowed spectral estimate must be rejected"),
        Err(error) => error,
    };
    assert_eq!(
        error,
        ChebyshevError::InvalidConfiguration {
            reason: "estimated spectral interval is not finite or usable",
        }
    );
}

#[test]
fn checked_amg_rejects_overflowed_generated_hierarchy() {
    let scale = f64::MAX;
    let off_diagonal = 0.49 * scale;
    let matrix = CsrMatrix::from_triplets(
        3,
        3,
        vec![
            (0, 0, scale),
            (1, 1, scale),
            (2, 2, scale),
            (0, 1, off_diagonal),
            (1, 0, off_diagonal),
            (0, 2, off_diagonal),
            (2, 0, off_diagonal),
            (1, 2, off_diagonal),
            (2, 1, off_diagonal),
        ],
    );

    let error = AmgPreconditioner::try_from_csr(&matrix, AmgConfig::default())
        .expect_err("checked AMG setup must reject generated non-finite values");
    assert!(matches!(
        error,
        sonium_solvers_core::traits::SolverError::OperationFailed { .. }
    ));
}

#[test]
fn backend_amg_exposes_setup_diagnostics() {
    let n = 9;
    let mut triplets = Vec::with_capacity(3 * n - 2);
    for row in 0..n {
        triplets.push((row, row, 2.0_f64));
        if row > 0 {
            triplets.push((row, row - 1, -1.0));
        }
        if row + 1 < n {
            triplets.push((row, row + 1, -1.0));
        }
    }
    let matrix = CsrMatrix::from_triplets(n, n, triplets);
    let backend = CpuBackend::new();
    let amg = BackendAmgPreconditioner::<CpuBackend>::try_from_csr(
        &backend,
        &matrix,
        AmgConfig {
            coarse_size: 2,
            max_levels: 8,
            ..AmgConfig::default()
        },
    )
    .expect("CPU backend AMG setup should succeed");

    let diagnostics = amg.diagnostics();
    assert_eq!(diagnostics.num_levels, amg.num_levels());
    assert!(diagnostics.num_levels >= 2);
    assert_eq!(diagnostics.level_dofs.len(), diagnostics.num_levels);
    assert_eq!(diagnostics.level_nnz.len(), diagnostics.num_levels);
    assert_eq!(diagnostics.finest_nnz, diagnostics.level_nnz[0]);
    assert_eq!(
        diagnostics.total_nnz,
        diagnostics.level_nnz.iter().sum::<usize>()
    );
    assert!(diagnostics.grid_complexity >= 1.0);
    assert!(diagnostics.operator_complexity >= 1.0);
    assert!(!diagnostics.has_coarse_direct_solver);
    assert_eq!(diagnostics.apply_count, 0);
    assert_eq!(diagnostics.total_apply_ms, 0.0);
    assert_eq!(diagnostics.mean_apply_ms, 0.0);

    let expected_workspace = diagnostics
        .level_dofs
        .iter()
        .enumerate()
        .map(|(level, &dofs)| {
            5 * dofs
                + diagnostics
                    .level_dofs
                    .get(level + 1)
                    .map_or(0, |coarse_dofs| 2 * coarse_dofs)
        })
        .sum::<usize>();
    assert_eq!(
        diagnostics.workspace_elements,
        expected_workspace + 2 * diagnostics.level_dofs[0]
    );

    let mut rhs = backend.raw_alloc::<f64>(n).expect("RHS allocation");
    let mut output = backend.raw_alloc::<f64>(n).expect("output allocation");
    backend.copy_h2d(&vec![1.0; n], &mut rhs);
    DevicePreconditioner::apply(&amg, &backend, &rhs, &mut output);
    let runtime = amg.diagnostics();
    assert_eq!(runtime.apply_count, 1);
    assert!(runtime.total_apply_ms.is_finite());
    assert!(runtime.mean_apply_ms.is_finite());
    assert_eq!(runtime.total_apply_ms, runtime.mean_apply_ms);

    // Checked failures must not publish partial output, while still recording
    // the first failing stage in the normalized runtime diagnostics.
    let bad_rhs = backend.raw_alloc::<f64>(n - 1).expect("bad RHS allocation");
    let mut guarded_output = backend
        .raw_alloc::<f64>(n)
        .expect("guarded output allocation");
    backend.copy_h2d(&vec![7.0; n], &mut guarded_output);
    let error = DevicePreconditioner::try_apply(&amg, &backend, &bad_rhs, &mut guarded_output)
        .expect_err("wrong-sized backend AMG RHS must fail");
    assert!(matches!(
        error,
        sonium_compute::BackendError::InvalidArgument { .. }
    ));
    let mut guarded_host = vec![0.0; n];
    backend.copy_d2h(&guarded_output, &mut guarded_host);
    assert_eq!(guarded_host, vec![7.0; n]);
    let runtime_after_failure = amg.diagnostics().runtime;
    assert_eq!(runtime_after_failure.apply_count, 1);
    assert_eq!(runtime_after_failure.apply_failures, 1);
    assert_eq!(
        runtime_after_failure.first_failure_stage,
        Some(sonium_compute::DevicePreconditionerFailureStage::Unknown)
    );

    let checkpoint = amg.to_checkpoint_bytes().expect("backend AMG checkpoint");
    let restored =
        BackendAmgPreconditioner::<CpuBackend>::try_from_checkpoint_bytes(&backend, &checkpoint)
            .expect("backend AMG checkpoint restore");
    assert_eq!(
        restored
            .to_checkpoint_bytes()
            .expect("restored backend AMG checkpoint"),
        checkpoint
    );
    let mut restored_output = backend
        .raw_alloc::<f64>(n)
        .expect("restored output allocation");
    DevicePreconditioner::try_apply(&restored, &backend, &rhs, &mut restored_output)
        .expect("restored backend AMG application");
    let mut source_values = vec![0.0; n];
    let mut restored_values = vec![0.0; n];
    backend.copy_d2h(&output, &mut source_values);
    backend.copy_d2h(&restored_output, &mut restored_values);
    let action_error = source_values
        .iter()
        .zip(&restored_values)
        .map(|(source, restored)| (source - restored).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        action_error < 1.0e-12,
        "backend AMG action error {action_error}"
    );

    let mut bad_magic = checkpoint.clone();
    bad_magic[0] ^= 0xff;
    assert!(
        BackendAmgPreconditioner::<CpuBackend>::try_from_checkpoint_bytes(&backend, &bad_magic)
            .is_err()
    );
    let mut bad_version = checkpoint.clone();
    bad_version[8] ^= 1;
    assert!(
        BackendAmgPreconditioner::<CpuBackend>::try_from_checkpoint_bytes(&backend, &bad_version)
            .is_err()
    );
    let scalar_offset = checkpoint
        .windows(std::mem::size_of_val(b"f64"))
        .position(|window| window == b"f64")
        .expect("backend checkpoint scalar tag") as usize;
    let mut bad_scalar = checkpoint.clone();
    bad_scalar[scalar_offset + 2] = b'2';
    assert!(
        BackendAmgPreconditioner::<CpuBackend>::try_from_checkpoint_bytes(&backend, &bad_scalar)
            .is_err()
    );
    let backend_name = std::any::type_name::<CpuBackend>().as_bytes();
    let backend_offset = checkpoint
        .windows(backend_name.len())
        .position(|window| window == backend_name)
        .expect("backend checkpoint backend tag");
    let mut bad_backend = checkpoint.clone();
    bad_backend[backend_offset] ^= 1;
    assert!(
        BackendAmgPreconditioner::<CpuBackend>::try_from_checkpoint_bytes(&backend, &bad_backend)
            .is_err()
    );
    let mut bad_checksum = checkpoint.clone();
    let last = bad_checksum.len() - 1;
    bad_checksum[last] ^= 1;
    assert!(
        BackendAmgPreconditioner::<CpuBackend>::try_from_checkpoint_bytes(&backend, &bad_checksum)
            .is_err()
    );
    let mut trailing = checkpoint;
    trailing.push(0);
    assert!(
        BackendAmgPreconditioner::<CpuBackend>::try_from_checkpoint_bytes(&backend, &trailing)
            .is_err()
    );
}

#[test]
fn backend_amg_faults_are_typed_transactional_and_retryable() {
    let n = 9;
    let mut triplets = Vec::with_capacity(3 * n - 2);
    for row in 0..n {
        triplets.push((row, row, 2.0_f64));
        if row > 0 {
            triplets.push((row, row - 1, -1.0));
        }
        if row + 1 < n {
            triplets.push((row, row + 1, -1.0));
        }
    }
    let matrix = CsrMatrix::from_triplets(n, n, triplets);
    let backend = CpuBackend::new();
    let config = AmgConfig {
        coarse_size: 2,
        max_levels: 8,
        num_pre_smooth: 1,
        num_post_smooth: 1,
        smoother: AmgSmoother::Jacobi,
        ..AmgConfig::default()
    };

    for point in [
        sonium_compute::PreconditionerFailureInjectionPoint::HierarchyUpload,
        sonium_compute::PreconditionerFailureInjectionPoint::DiagonalUpload,
        sonium_compute::PreconditionerFailureInjectionPoint::WorkspaceAllocation,
    ] {
        sonium_compute::CpuBackend::inject_preconditioner_failure(point);
        let error =
            BackendAmgPreconditioner::<CpuBackend>::try_from_csr(&backend, &matrix, config.clone())
                .err()
                .expect("injected AMG setup fault must be returned");
        assert!(matches!(
            error,
            sonium_compute::BackendError::Transfer { .. }
                | sonium_compute::BackendError::Allocation { .. }
        ));
    }

    let amg = BackendAmgPreconditioner::<CpuBackend>::try_from_csr(&backend, &matrix, config)
        .expect("AMG setup must remain retryable after injected failures");
    let mut rhs = backend.raw_alloc::<f64>(n).expect("RHS allocation");
    let mut output = backend.raw_alloc::<f64>(n).expect("output allocation");
    backend.copy_h2d(&vec![1.0; n], &mut rhs);
    backend.copy_h2d(&vec![37.0; n], &mut output);

    sonium_compute::CpuBackend::inject_preconditioner_failure(
        sonium_compute::PreconditionerFailureInjectionPoint::Smoother,
    );
    let error = DevicePreconditioner::try_apply(&amg, &backend, &rhs, &mut output)
        .expect_err("injected smoother fault must be returned");
    assert!(matches!(error, sonium_compute::BackendError::Launch { .. }));
    let mut output_host = vec![0.0; n];
    backend.copy_d2h(&output, &mut output_host);
    assert_eq!(output_host, vec![37.0; n]);
    assert_eq!(amg.diagnostics().runtime.apply_failures, 1);

    DevicePreconditioner::try_apply(&amg, &backend, &rhs, &mut output)
        .expect("a failed smoother setup/action must not poison the retry");
    assert_eq!(amg.diagnostics().runtime.apply_count, 1);
    sonium_compute::CpuBackend::clear_preconditioner_failure();
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_amg_checkpoint_rebuilds_native_coarse_state_without_action_staging() {
    let cuda = match sonium_compute::CudaBackend::new_default() {
        Ok(backend) => backend,
        Err(error) if std::env::var("SONIUM_REQUIRE_CUDA").is_ok() => {
            panic!("CUDA required by SONIUM_REQUIRE_CUDA but unavailable: {error}")
        }
        Err(error) => {
            eprintln!("CUDA unavailable, skipping CUDA AMG checkpoint replay: {error}");
            return;
        }
    };

    let n = 9;
    let mut triplets = Vec::with_capacity(3 * n - 2);
    for row in 0..n {
        triplets.push((row, row, Complex32::new(2.0, 0.0)));
        if row > 0 {
            triplets.push((row, row - 1, Complex32::new(-1.0, 0.0)));
        }
        if row + 1 < n {
            triplets.push((row, row + 1, Complex32::new(-1.0, 0.0)));
        }
    }
    let matrix = CsrMatrix::from_triplets(n, n, triplets);
    let config = AmgConfig {
        coarse_size: 2,
        max_levels: 8,
        num_pre_smooth: 1,
        num_post_smooth: 1,
        smoother: AmgSmoother::Jacobi,
        ..AmgConfig::default()
    };
    cuda.inject_preconditioner_failure(
        sonium_compute::PreconditionerFailureInjectionPoint::CoarseSolve,
    );
    let error = BackendAmgPreconditioner::<sonium_compute::CudaBackend, Complex32>::try_from_csr_with_cuda_coarse_lu(
        &cuda,
        &matrix,
        config.clone(),
    )
    .err()
    .expect("native CUDA coarse-solve setup fault must be returned");
    assert!(matches!(
        error,
        sonium_compute::BackendError::OperationFailed { .. }
    ));
    cuda.recover_after_error()
        .expect("recoverable native CUDA coarse-solve fault");
    let fresh = BackendAmgPreconditioner::<sonium_compute::CudaBackend, Complex32>::try_from_csr_with_cuda_coarse_lu(
        &cuda,
        &matrix,
        config,
    )
    .expect("native CUDA AMG setup with coarse LU");
    let checkpoint = fresh.to_checkpoint_bytes().expect("CUDA AMG checkpoint");
    let restored = BackendAmgPreconditioner::<sonium_compute::CudaBackend, Complex32>::try_from_checkpoint_bytes_with_cuda_coarse_lu(
        &cuda,
        &checkpoint,
    )
    .expect("native CUDA AMG checkpoint restore with coarse LU");
    assert_eq!(
        restored.to_checkpoint_bytes().expect("restored checkpoint"),
        checkpoint
    );
    // Restore publishes precisely the same resident hierarchy/capability
    // description before either instance has observed an action. Runtime
    // counters and timing deliberately diverge once the two instances run.
    assert_eq!(fresh.diagnostics(), restored.diagnostics());
    assert_eq!(fresh.forward_capability(), restored.forward_capability());
    assert_eq!(
        fresh.transpose_capability(),
        restored.transpose_capability()
    );
    assert_eq!(
        fresh.hermitian_capability(),
        restored.hermitian_capability()
    );

    let rhs_host = (0..n)
        .map(|index| Complex32::new(index as f32 + 1.0, -0.25 * index as f32))
        .collect::<Vec<_>>();
    let block_rhs_host = rhs_host
        .iter()
        .copied()
        .chain(rhs_host.iter().rev().map(|value| value.conj()))
        .collect::<Vec<_>>();
    let mut rhs = cuda.raw_alloc::<Complex32>(n).expect("CUDA RHS allocation");
    let mut block_rhs = cuda
        .raw_alloc::<Complex32>(2 * n)
        .expect("CUDA block RHS allocation");
    cuda.copy_h2d(&rhs_host, &mut rhs);
    cuda.copy_h2d(&block_rhs_host, &mut block_rhs);
    let mut fresh_out = cuda
        .raw_alloc::<Complex32>(n)
        .expect("CUDA fresh output allocation");
    let mut restored_out = cuda
        .raw_alloc::<Complex32>(n)
        .expect("CUDA restored output allocation");
    let mut fresh_block_out = cuda
        .raw_alloc::<Complex32>(2 * n)
        .expect("CUDA fresh block output allocation");
    let mut restored_block_out = cuda
        .raw_alloc::<Complex32>(2 * n)
        .expect("CUDA restored block output allocation");

    let before = cuda.runtime_stats();
    DevicePreconditioner::try_apply(&fresh, &cuda, &rhs, &mut fresh_out)
        .expect("fresh CUDA AMG apply");
    DevicePreconditioner::try_apply(&restored, &cuda, &rhs, &mut restored_out)
        .expect("restored CUDA AMG apply");
    DevicePreconditioner::try_apply_colmajor(&fresh, &cuda, 2, &block_rhs, &mut fresh_block_out)
        .expect("fresh CUDA AMG block apply");
    DevicePreconditioner::try_apply_colmajor(
        &restored,
        &cuda,
        2,
        &block_rhs,
        &mut restored_block_out,
    )
    .expect("restored CUDA AMG block apply");
    cuda.try_synchronize()
        .expect("CUDA AMG action synchronization");
    let action_stats = cuda.runtime_stats_since(before);
    assert_eq!(action_stats.transfer.h2d_operations, 0);
    assert_eq!(action_stats.transfer.d2h_operations, 0);

    let mut fresh_host = vec![Complex32::new(0.0, 0.0); n];
    let mut restored_host = vec![Complex32::new(0.0, 0.0); n];
    let mut fresh_block_host = vec![Complex32::new(0.0, 0.0); 2 * n];
    let mut restored_block_host = vec![Complex32::new(0.0, 0.0); 2 * n];
    cuda.copy_d2h(&fresh_out, &mut fresh_host);
    cuda.copy_d2h(&restored_out, &mut restored_host);
    cuda.copy_d2h(&fresh_block_out, &mut fresh_block_host);
    cuda.copy_d2h(&restored_block_out, &mut restored_block_host);
    for (fresh_value, restored_value) in fresh_host.iter().zip(&restored_host) {
        assert!((*fresh_value - *restored_value).norm() < 3.0e-4);
    }
    for (fresh_value, restored_value) in fresh_block_host.iter().zip(&restored_block_host) {
        assert!((*fresh_value - *restored_value).norm() < 3.0e-4);
    }
    assert_eq!(fresh.diagnostics().runtime, restored.diagnostics().runtime);
}

fn complex_inner(x: &[Complex64], y: &[Complex64]) -> Complex64 {
    x.iter().zip(y).map(|(lhs, rhs)| lhs.conj() * rhs).sum()
}

fn complex_bilinear(x: &[Complex64], y: &[Complex64]) -> Complex64 {
    x.iter().zip(y).map(|(lhs, rhs)| *lhs * *rhs).sum()
}

#[test]
fn block_jacobi_complex_transpose_and_hermitian_actions_are_adjoint() {
    use sonium_solvers_core::traits::{AdjointCapability, Preconditioner};

    let matrix = CsrMatrix::from_raw_parts(
        4,
        4,
        vec![0, 2, 4, 6, 8],
        vec![0, 1, 0, 1, 2, 3, 2, 3],
        vec![
            Complex64::new(4.0, 1.0),
            Complex64::new(-1.0, 0.2),
            Complex64::new(0.5, -0.3),
            Complex64::new(3.0, 0.5),
            Complex64::new(2.0, -0.4),
            Complex64::new(-0.2, 0.1),
            Complex64::new(0.7, 0.2),
            Complex64::new(5.0, -0.1),
        ],
    );
    let preconditioner = BlockJacobiPreconditioner::from_csr(
        &matrix,
        2,
        IlutConfig {
            drop_tolerance: 0.0,
            fill_factor: 4,
            pivot_tolerance: 1.0e-12,
            ..IlutConfig::default()
        },
    );
    assert_eq!(
        preconditioner.transpose_capability(),
        AdjointCapability::Exact
    );
    assert_eq!(
        preconditioner.hermitian_capability(),
        AdjointCapability::Exact
    );

    let x = vec![
        Complex64::new(1.0, -0.5),
        Complex64::new(-2.0, 0.25),
        Complex64::new(0.75, 1.5),
        Complex64::new(1.25, -0.75),
    ];
    let y = vec![
        Complex64::new(-0.5, 0.75),
        Complex64::new(1.5, -1.0),
        Complex64::new(0.25, 0.5),
        Complex64::new(-1.25, 0.2),
    ];
    let forward_y = preconditioner.apply(&y);
    let transpose_x = preconditioner.apply_transpose(&x);
    let hermitian_y = preconditioner.apply_hermitian(&y);

    let transpose_error =
        (complex_bilinear(&transpose_x, &y) - complex_bilinear(&x, &forward_y)).norm();
    let hermitian_error =
        (complex_inner(&preconditioner.apply(&x), &y) - complex_inner(&x, &hermitian_y)).norm();
    assert!(
        transpose_error < 1.0e-11,
        "transpose error {transpose_error}"
    );
    assert!(
        hermitian_error < 1.0e-11,
        "Hermitian error {hermitian_error}"
    );
}

fn malformed_csr() -> CsrMatrix<f64> {
    let mut matrix = CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0, 3.0]);
    matrix.row_ptrs[2] = 1;
    matrix
}

#[test]
fn composed_compatibility_constructors_expose_aggregate_setup_diagnostics() {
    let identity = CsrMatrix::<f64>::identity(4);

    let block = BlockJacobiPreconditioner::from_csr(&identity, 2, IlutConfig::default());
    let block_diagnostics = block.diagnostics();
    assert_eq!(block_diagnostics.regularized_pivots, 0);
    assert_eq!(block_diagnostics.blocks_with_regularized_pivots, 0);
    assert_eq!(block_diagnostics.non_finite_entries, 0);

    let schwarz = AdditiveSchwarzPreconditioner::from_csr(&identity, 2, 1);
    let schwarz_diagnostics = schwarz.diagnostics();
    assert_eq!(schwarz_diagnostics.skipped_pivots, 0);
    assert_eq!(schwarz_diagnostics.non_finite_entries, 0);
    assert!(
        schwarz_diagnostics
            .min_abs_pivot
            .is_some_and(|pivot| pivot.is_finite() && pivot > 0.0)
    );

    let generated_non_finite = CsrMatrix::from_raw_parts(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![1e-100_f64, 1e154, 1e154, 1e100],
    );
    let config = IlutConfig {
        drop_tolerance: 0.0,
        pivot_tolerance: 1e-300,
        pivoting: false,
        ..IlutConfig::default()
    };
    let compatibility =
        BlockJacobiPreconditioner::from_csr(&generated_non_finite, 1, config.clone());
    assert!(compatibility.diagnostics().non_finite_entries > 0);
    assert!(BlockJacobiPreconditioner::try_from_csr(&generated_non_finite, 1, config).is_err());
}

#[test]
fn checked_ilu_families_reject_public_csr_mutation() {
    let matrix = malformed_csr();
    assert!(IluPreconditioner::try_from_csr(&matrix).is_err());
    assert!(IlutPreconditioner::try_from_csr(&matrix).is_err());
    assert!(HiluPreconditioner::try_from_csr(&matrix).is_err());
}

#[test]
fn checked_block_jacobi_rejects_public_csr_mutation() {
    let matrix = malformed_csr();
    assert!(BlockJacobiPreconditioner::try_from_csr(&matrix, 2, IlutConfig::default()).is_err());
}

#[test]
fn checked_block_jacobi_rejects_rectangular_matrix() {
    let matrix = CsrMatrix::from_raw_parts(1, 2, vec![0, 1], vec![1], vec![1.0]);
    assert!(BlockJacobiPreconditioner::try_from_csr(&matrix, 1, IlutConfig::default()).is_err());
}

#[test]
fn checked_amg_rejects_public_csr_mutation() {
    let matrix = malformed_csr();
    assert!(AmgPreconditioner::try_from_csr(&matrix, AmgConfig::default()).is_err());
}

#[test]
fn checked_chebyshev_rejects_public_csr_mutation_without_panicking() {
    let matrix = malformed_csr();
    assert!(matches!(
        ChebyshevPreconditioner::try_from_csr(&matrix, ChebyshevConfig::default()),
        Err(ChebyshevError::InvalidCsrStructure)
    ));
}

#[test]
fn checked_amg_rejects_rectangular_empty_and_invalid_configuration() {
    let rectangular = CsrMatrix::from_raw_parts(1, 2, vec![0, 1], vec![1], vec![1.0]);
    assert!(AmgPreconditioner::try_from_csr(&rectangular, AmgConfig::default()).is_err());

    let empty: CsrMatrix<f64> = CsrMatrix::from_raw_parts(0, 0, vec![0], Vec::new(), Vec::new());
    assert!(AmgPreconditioner::try_from_csr(&empty, AmgConfig::default()).is_err());

    let matrix: CsrMatrix<f64> = CsrMatrix::identity(2);
    let mut invalid = AmgConfig::default();
    invalid.max_levels = 0;
    assert!(AmgPreconditioner::try_from_csr(&matrix, invalid).is_err());

    let mut invalid = AmgConfig::default();
    invalid.strong_threshold = f64::NAN;
    assert!(AmgPreconditioner::try_from_csr(&matrix, invalid).is_err());

    let mut invalid = AmgConfig::default();
    invalid.block_size = 3;
    assert!(AmgPreconditioner::try_from_csr(&matrix, invalid).is_err());
}

#[test]
fn checked_amg_rejects_bad_material_soc_and_accepts_valid_setup() {
    let matrix: CsrMatrix<f64> = CsrMatrix::identity(2);
    let mut invalid = AmgConfig::default();
    invalid.material_soc = Some(MaterialSocConfig {
        tensors: vec![vec![1.0], vec![1.0]],
        dim: 2,
    });
    assert!(AmgPreconditioner::try_from_csr(&matrix, invalid).is_err());

    let amg = AmgPreconditioner::try_from_csr(&matrix, AmgConfig::default())
        .expect("checked AMG should accept a finite square identity matrix");
    assert!(amg.num_levels() >= 1);
}

#[test]
fn checked_schwarz_rejects_public_csr_mutation() {
    let matrix = malformed_csr();
    assert!(AdditiveSchwarzPreconditioner::try_from_csr(&matrix, 2, 1).is_err());
    assert!(AdditiveSchwarzPreconditioner::try_from_csr_oras(&matrix, 2, 1, 1.0).is_err());
    assert!(AdditiveSchwarzPreconditioner::try_from_csr_habc(&matrix, 2, 1, 1.0).is_err());
}

#[test]
fn checked_schwarz_rejects_rectangular_matrix() {
    let matrix = CsrMatrix::from_raw_parts(1, 2, vec![0, 1], vec![1], vec![1.0]);
    assert!(AdditiveSchwarzPreconditioner::try_from_csr(&matrix, 1, 0).is_err());
}

#[test]
fn checked_schwarz_rejects_skipped_local_ilu_pivot() {
    let matrix = CsrMatrix::from_raw_parts(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![1e-6, 1.0, 1.0, 2.0],
    );

    let error = AdditiveSchwarzPreconditioner::try_from_csr(&matrix, 1, 0)
        .expect_err("checked Schwarz must reject skipped local ILU pivots");
    assert!(matches!(
        error,
        sonium_solvers_core::traits::SolverError::OperationFailed { .. }
    ));
}

#[test]
fn checked_transmission_schwarz_rejects_skipped_local_ilu_pivot() {
    let matrix = CsrMatrix::from_raw_parts(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![1e-6, 1.0, 1.0, 2.0],
    );

    assert!(AdditiveSchwarzPreconditioner::try_from_csr_oras(&matrix, 1, 0, 1.0).is_err());
    assert!(AdditiveSchwarzPreconditioner::try_from_csr_habc(&matrix, 1, 0, 1.0).is_err());
}

#[test]
fn checked_oras_rejects_non_finite_generated_local_setup() {
    let matrix = CsrMatrix::from_raw_parts(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(2.0, 0.0),
        ],
    );

    assert!(
        AdditiveSchwarzPreconditioner::try_from_csr_oras(&matrix, 2, 0, f64::INFINITY).is_err()
    );
}

#[test]
fn checked_saddle_point_rejects_malformed_csr_and_invalid_block_split() {
    let matrix = malformed_csr();
    let config = SaddlePointConfig {
        block1_size: 1,
        ..Default::default()
    };
    assert!(SaddlePointPreconditioner::try_from_matrix(&matrix, config).is_err());

    let matrix = CsrMatrix::from_raw_parts(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![2.0, 0.0, 0.0, 2.0],
    );
    let config = SaddlePointConfig {
        block1_size: 2,
        ..Default::default()
    };
    assert!(SaddlePointPreconditioner::try_from_matrix(&matrix, config).is_err());
}

#[test]
fn checked_saddle_point_rejects_non_finite_matrix_values() {
    let mut matrix = CsrMatrix::from_raw_parts(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![2.0, 0.0, 0.0, 2.0],
    );
    matrix.values[1] = f64::NAN;
    let config = SaddlePointConfig {
        block1_size: 1,
        ..Default::default()
    };

    let error = SaddlePointPreconditioner::try_from_matrix(&matrix, config)
        .err()
        .expect("checked saddle setup must reject non-finite matrix values");
    assert!(matches!(
        error,
        sonium_solvers_core::traits::SolverError::OperationFailed { .. }
    ));
}

#[test]
fn checked_saddle_point_external_schur_rejects_bad_shape_and_storage() {
    let matrix = CsrMatrix::from_raw_parts(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![2.0, 0.0, 0.0, 2.0],
    );
    let bad_shape = CsrMatrix::from_raw_parts(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![1.0, 0.0, 0.0, 1.0],
    );
    let config = SaddlePointConfig {
        block1_size: 1,
        schur_strategy: SchurStrategy::External,
        ..Default::default()
    };
    assert!(
        SaddlePointPreconditioner::try_from_matrix_with_schur(&matrix, &bad_shape, config).is_err()
    );

    let mut malformed = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![1.0]);
    malformed.row_ptrs[1] = 0;
    let config = SaddlePointConfig {
        block1_size: 1,
        schur_strategy: SchurStrategy::External,
        ..Default::default()
    };
    assert!(
        SaddlePointPreconditioner::try_from_matrix_with_schur(&matrix, &malformed, config).is_err()
    );
}

#[test]
fn checked_saddle_point_external_schur_rejects_non_finite_values() {
    let matrix = CsrMatrix::from_raw_parts(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![2.0, 0.0, 0.0, 2.0],
    );
    let mut schur = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![1.0]);
    schur.values[0] = f64::INFINITY;
    let config = SaddlePointConfig {
        block1_size: 1,
        schur_strategy: SchurStrategy::External,
        ..Default::default()
    };

    let error = SaddlePointPreconditioner::try_from_matrix_with_schur(&matrix, &schur, config)
        .err()
        .expect("checked external Schur setup must reject non-finite values");
    assert!(matches!(
        error,
        sonium_solvers_core::traits::SolverError::OperationFailed { .. }
    ));
}

#[test]
fn checked_saddle_point_rejects_skipped_schur_ilu_pivot() {
    let matrix = CsrMatrix::from_raw_parts(
        4,
        4,
        vec![0, 1, 2, 4, 6],
        vec![0, 1, 2, 3, 2, 3],
        vec![1.0, 1.0, 1e-6, 1.0, 1.0, 2.0],
    );
    let config = SaddlePointConfig {
        block1_size: 2,
        block1_config: sonium_solvers_preconditioners::BlockPreconditionerConfig::None,
        block2_config: sonium_solvers_preconditioners::BlockPreconditionerConfig::Ilut(
            IlutConfig {
                pivot_tolerance: 0.01,
                pivoting: false,
                ..Default::default()
            },
        ),
        ..Default::default()
    };

    let error = SaddlePointPreconditioner::try_from_matrix(&matrix, config)
        .err()
        .expect("checked saddle setup must reject skipped Schur ILUT pivots");
    assert!(matches!(
        error,
        sonium_solvers_core::traits::SolverError::OperationFailed { .. }
    ));
}
#[test]
fn checked_schwarz_rejects_malformed_core_partition_topology() {
    let matrix = CsrMatrix::<f64>::identity(4);
    let malformed = [
        ("empty", vec![vec![], vec![0, 1, 2, 3]]),
        ("out-of-range", vec![vec![0, 1], vec![2, 4]]),
        ("duplicate-within-core", vec![vec![0, 0], vec![1, 2, 3]]),
        ("overlap", vec![vec![0, 1], vec![1, 2, 3]]),
        ("uncovered", vec![vec![0, 1], vec![2]]),
    ];

    for (name, partitions) in malformed {
        assert!(
            AdditiveSchwarzPreconditioner::try_from_csr_partitions(&matrix, &partitions, 1,)
                .is_err(),
            "malformed {name} partition topology was accepted"
        );
        assert!(
            AdditiveSchwarzPreconditioner::try_from_csr_partitions_oras(
                &matrix,
                &partitions,
                1,
                1.0,
            )
            .is_err(),
            "malformed {name} ORAS partition topology was accepted"
        );
    }

    let valid = vec![vec![1, 0], vec![3, 2]];
    assert!(AdditiveSchwarzPreconditioner::try_from_csr_partitions(&matrix, &valid, 1).is_ok());
    assert!(
        AdditiveSchwarzPreconditioner::try_from_csr_partitions_oras(&matrix, &valid, 1, 1.0)
            .is_ok()
    );
}

#[test]
fn checked_schwarz_auto_rejects_malformed_csr() {
    let mut matrix = CsrMatrix::<f64>::identity(2);
    matrix.row_ptrs = vec![0, 2, 1];
    let error = AdditiveSchwarzPreconditioner::try_from_csr_auto(&matrix, 1)
        .expect_err("automatic checked Schwarz setup must reject malformed CSR");
    assert!(matches!(
        error,
        sonium_solvers_core::traits::SolverError::OperationFailed { .. }
            | sonium_solvers_core::traits::SolverError::DimensionMismatch { .. }
    ));
    assert!(AdditiveSchwarzPreconditioner::try_from_csr_oras_auto(&matrix, 1, 1.0).is_err());
    assert!(AdditiveSchwarzPreconditioner::try_from_csr_habc_auto(&matrix, 1, 1.0).is_err());

    let valid = CsrMatrix::<f64>::identity(2);
    assert!(AdditiveSchwarzPreconditioner::try_from_csr_oras_auto(&valid, 0, 1.0).is_ok());
    assert!(AdditiveSchwarzPreconditioner::try_from_csr_habc_auto(&valid, 0, 1.0).is_ok());
    assert!(AdditiveSchwarzPreconditioner::try_from_csr_oras_auto(&valid, 0, f64::NAN).is_err());
    assert!(
        AdditiveSchwarzPreconditioner::try_from_csr_habc_auto(&valid, 0, f64::INFINITY).is_err()
    );
}

#[test]
fn checked_block_jacobi_auto_rejects_malformed_csr() {
    let mut malformed = CsrMatrix::<f64>::identity(4);
    malformed.row_ptrs = vec![0, 2, 1, 3, 4];
    assert!(
        BlockJacobiPreconditioner::try_from_csr_auto(&malformed, IlutConfig::default()).is_err()
    );

    let valid = CsrMatrix::<f64>::identity(4);
    assert!(BlockJacobiPreconditioner::try_from_csr_auto(&valid, IlutConfig::default()).is_ok());
}
