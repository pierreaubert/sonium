//! Ownership and checked admission for retained saddle-point preconditioning.

// Rust guideline compliant 2026-02-21
use sonium_compute::preconditioner::{
    PressureSchurBackendKind, SaddlePointJacobiSchurBackend, SaddlePointJacobiSchurBackendConfig,
};
use sonium_compute::{
    Backend, CpuBackend, DevicePreconditioner, HostCsrRef, MemoryPool, MemoryPoolConfig,
};

fn config() -> SaddlePointJacobiSchurBackendConfig {
    SaddlePointJacobiSchurBackendConfig {
        velocity_block_size: 2,
        velocity_degree: 1,
        velocity_damping: 1.0,
        pressure_degree: 1,
        pressure_damping: 1.0,
        pressure_kind: PressureSchurBackendKind::Polynomial,
        pressure_chebyshev_degree: 3,
        pressure_chebyshev_eigenvalue_ratio: 30.0,
    }
}

fn owned(backend: &CpuBackend) -> SaddlePointJacobiSchurBackend<'static, CpuBackend> {
    let rows = vec![0, 2, 4, 5];
    let columns = vec![0, 2, 1, 2, 2];
    let values = vec![4.0, 1.0, 4.0, 1.0, 0.1];
    SaddlePointJacobiSchurBackend::try_from_host_csr(
        backend,
        HostCsrRef::try_new(3, 3, &rows, &columns, &values).unwrap(),
        HostCsrRef::try_new(1, 1, &[0, 1], &[0], &[1.0]).unwrap(),
        config(),
    )
    .unwrap()
}

#[test]
fn saddle_owned_survives_host_inputs_and_reuses_action() {
    let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(1 << 20)).unwrap();
    let backend = pool.backend();
    let preconditioner = owned(backend);
    let mut rhs = pool.try_alloc::<f64>(3).unwrap();
    let mut result = pool.try_alloc::<f64>(3).unwrap();
    for amplitude in [1.0, -2.0, 0.5] {
        backend
            .try_copy_h2d(&[amplitude; 3], rhs.raw_mut())
            .unwrap();
        preconditioner
            .try_apply(backend, rhs.raw(), result.raw_mut())
            .unwrap();
        let mut host = [0.0; 3];
        backend.try_copy_d2h(result.raw(), &mut host).unwrap();
        assert_eq!(host, [0.0, 0.0, amplitude]);
    }
}

#[test]
fn saddle_owned_rejects_invalid_inputs() {
    let backend = CpuBackend::new();
    for case in 0..8 {
        let mut config = config();
        let mut values = [4.0, 1.0, 4.0, 1.0, 0.1];
        let mut schur = [1.0];
        match case {
            0 => config.velocity_block_size = 4,
            1 => config.pressure_kind = PressureSchurBackendKind::External,
            2 => config.velocity_damping = f64::NAN,
            3 => config.pressure_damping = f64::INFINITY,
            4 => config.pressure_chebyshev_eigenvalue_ratio = f64::NAN,
            5 => values[1] = f64::NAN,
            6 => schur[0] = f64::INFINITY,
            7 => config.velocity_block_size = 1,
            _ => unreachable!(),
        }
        let result = SaddlePointJacobiSchurBackend::try_from_host_csr(
            &backend,
            HostCsrRef::try_new(3, 3, &[0, 2, 4, 5], &[0, 2, 1, 2, 2], &values).unwrap(),
            HostCsrRef::try_new(1, 1, &[0, 1], &[0], &schur).unwrap(),
            config,
        );
        assert!(result.is_err(), "case {case}");
    }
}

#[test]
fn saddle_owned_external_pressure_action_masks_scalar_and_block_results() {
    let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(1 << 20)).unwrap();
    let backend = pool.backend();
    let pressure =
        sonium_compute::PointJacobiBackend::try_from_diagonal(backend, &[1.0, 1.0, 2.0]).unwrap();
    let preconditioner = owned(backend).with_owned_pressure_preconditioner(Box::new(pressure));
    let mut rhs = pool.try_alloc::<f64>(6).unwrap();
    let mut result = pool.try_alloc::<f64>(6).unwrap();
    backend
        .try_copy_h2d(&[1.0, 1.0, 1.0, -2.0, -2.0, -2.0], rhs.raw_mut())
        .unwrap();
    preconditioner
        .try_apply_colmajor(backend, 2, rhs.raw(), result.raw_mut())
        .unwrap();
    let mut host = [0.0; 6];
    backend.try_copy_d2h(result.raw(), &mut host).unwrap();
    assert_eq!(host, [0.125, 0.125, 0.5, -0.25, -0.25, -1.0]);
    let mut single = pool.try_alloc::<f64>(3).unwrap();
    let mut output = pool.try_alloc::<f64>(3).unwrap();
    backend.try_copy_h2d(&[1.0; 3], single.raw_mut()).unwrap();
    preconditioner
        .try_apply(backend, single.raw(), output.raw_mut())
        .unwrap();
    let mut actual = [0.0; 3];
    backend.try_copy_d2h(output.raw(), &mut actual).unwrap();
    assert_eq!(actual, [0.125, 0.125, 0.5]);
}

fn refresh_parity<B: Backend>(backend: B) {
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 24)).unwrap();
    let backend = pool.backend();
    let base = [4.0, 0.5, 1.0, 0.5, 3.0, 2.0, 2.0, 0.2, 0.2, 3.0];
    let pressure = [2.0, 0.2, 0.2, 3.0];
    fn matrix(values: &[f64]) -> HostCsrRef<'_, f64> {
        HostCsrRef::try_new(
            4,
            4,
            &[0, 3, 6, 8, 10],
            &[0, 1, 2, 0, 1, 3, 2, 3, 2, 3],
            values,
        )
        .unwrap()
    }
    fn schur(values: &[f64]) -> HostCsrRef<'_, f64> {
        HostCsrRef::try_new(2, 2, &[0, 2, 4], &[0, 1, 0, 1], values).unwrap()
    }
    for kind in [
        PressureSchurBackendKind::Polynomial,
        PressureSchurBackendKind::Chebyshev,
    ] {
        let config = SaddlePointJacobiSchurBackendConfig {
            velocity_degree: 3,
            pressure_degree: 3,
            pressure_kind: kind,
            ..config()
        };
        let mut action = SaddlePointJacobiSchurBackend::try_from_host_csr(
            backend,
            matrix(&base),
            schur(&pressure),
            config,
        )
        .unwrap();
        for pass in 1..=3 {
            let mut values = base;
            values[0] += pass as f64;
            values[2] *= pass as f64 + 0.5;
            let pressure = [2.0, 0.1 * pass as f64, 0.1 * pass as f64, 3.0 + pass as f64];
            action
                .try_update_numeric_host(backend, matrix(&values), schur(&pressure))
                .unwrap();
            let fresh = SaddlePointJacobiSchurBackend::try_from_host_csr(
                backend,
                matrix(&values),
                schur(&pressure),
                config,
            )
            .unwrap();
            // Invalid input must preserve the last usable numeric state.
            let mut invalid = values;
            invalid[0] = f64::NAN;
            assert!(
                action
                    .try_update_numeric_host(backend, matrix(&invalid), schur(&pressure))
                    .is_err()
            );
            let wrong_columns = [1, 0, 2, 0, 1, 3, 2, 3, 2, 3];
            let wrong = HostCsrRef {
                col_indices: &wrong_columns,
                ..matrix(&values)
            };
            assert!(
                action
                    .try_update_numeric_host(backend, wrong, schur(&pressure))
                    .is_err()
            );
            for count in [1, 2] {
                let mut rhs = pool.try_alloc::<f64>(4 * count).unwrap();
                let mut actual = pool.try_alloc::<f64>(4 * count).unwrap();
                let mut expected = pool.try_alloc::<f64>(4 * count).unwrap();
                let host: Vec<_> = (0..4 * count).map(|i| i as f64 - 2.0).collect();
                backend.try_copy_h2d(&host, rhs.raw_mut()).unwrap();
                action
                    .try_apply_colmajor(backend, count, rhs.raw(), actual.raw_mut())
                    .unwrap();
                fresh
                    .try_apply_colmajor(backend, count, rhs.raw(), expected.raw_mut())
                    .unwrap();
                let mut a = vec![0.0; host.len()];
                let mut b = a.clone();
                backend.try_copy_d2h(actual.raw(), &mut a).unwrap();
                backend.try_copy_d2h(expected.raw(), &mut b).unwrap();
                for (a, b) in a.iter().zip(b) {
                    assert!((a - b).abs() < 1e-12);
                }
            }
        }
    }
}

#[test]
fn saddle_numeric_refresh_matches_fresh_scalar_and_block() {
    refresh_parity(CpuBackend::new());
}

#[cfg(feature = "cuda")]
#[test]
fn native_saddle_numeric_refresh_and_failed_upload_recovery() {
    use sonium_compute::{CudaBackend, PreconditionerFailureInjectionPoint};
    let backend = match CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            assert_ne!(
                std::env::var("SONIUM_REQUIRE_CUDA").as_deref(),
                Ok("1"),
                "{error}"
            );
            eprintln!("SKIP CUDA Schur numeric refresh: {error}");
            return;
        }
    };
    refresh_parity(backend);
    external_refresh(CudaBackend::new(0).unwrap());
    let pool = MemoryPool::new(
        CudaBackend::new(0).unwrap(),
        MemoryPoolConfig::with_cap(1 << 24),
    )
    .unwrap();
    let backend = pool.backend();
    let matrix = HostCsrRef::try_new(
        3,
        3,
        &[0, 2, 4, 5],
        &[0, 2, 1, 2, 2],
        &[4.0, 1.0, 4.0, 1.0, 0.1],
    )
    .unwrap();
    let schur = HostCsrRef::try_new(1, 1, &[0, 1], &[0], &[1.0]).unwrap();
    let mut action =
        SaddlePointJacobiSchurBackend::try_from_host_csr(backend, matrix, schur, config()).unwrap();
    let mut rhs = pool.try_alloc::<f64>(3).unwrap();
    let mut output = pool.try_alloc::<f64>(3).unwrap();
    backend.try_copy_h2d(&[1.0; 3], rhs.raw_mut()).unwrap();
    backend.try_copy_h2d(&[42.0; 3], output.raw_mut()).unwrap();
    let matrix = HostCsrRef {
        values: &[8.0, 2.0, 4.0, 1.0, 0.1],
        ..matrix
    };
    let schur = HostCsrRef {
        values: &[2.0],
        ..schur
    };
    backend.inject_preconditioner_failure(PreconditionerFailureInjectionPoint::DiagonalUpload);
    assert!(
        action
            .try_update_numeric_host(backend, matrix, schur)
            .is_err()
    );
    backend.recover_after_error().unwrap();
    assert!(
        action
            .try_apply(backend, rhs.raw(), output.raw_mut())
            .is_err()
    );
    assert!(
        action
            .try_apply_colmajor(backend, 1, rhs.raw(), output.raw_mut())
            .is_err()
    );
    let mut host = [0.0; 3];
    backend.try_copy_d2h(output.raw(), &mut host).unwrap();
    assert_eq!(host, [42.0; 3]);
    // Retry the old pressure matrix after the failed upload of the new one.
    // Recovery must restore pressure buffers even though the host snapshot matches.
    let old_pressure = HostCsrRef {
        values: &[1.0],
        ..schur
    };
    action
        .try_update_numeric_host(backend, matrix, old_pressure)
        .unwrap();
    action
        .try_apply(backend, rhs.raw(), output.raw_mut())
        .unwrap();
    backend.try_copy_d2h(output.raw(), &mut host).unwrap();
    assert_eq!(host, [-0.125, 0.0, 1.0]);
    let before = backend.begin_runtime_stats();
    action
        .try_update_numeric_host(backend, matrix, schur)
        .unwrap();
    assert_eq!(backend.runtime_stats_since(before).allocation_operations, 0);
    action
        .try_apply(backend, rhs.raw(), output.raw_mut())
        .unwrap();
    backend.try_copy_d2h(output.raw(), &mut host).unwrap();
    assert_eq!(host, [0.0, 0.125, 0.5]);
}

fn external_refresh<B: Backend>(backend: B) {
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 24)).unwrap();
    let backend = pool.backend();
    let matrix = HostCsrRef::try_new(
        3,
        3,
        &[0, 2, 4, 5],
        &[0, 2, 1, 2, 2],
        &[4.0, 1.0, 4.0, 1.0, 0.1],
    )
    .unwrap();
    let schur = HostCsrRef::try_new(1, 1, &[0, 1], &[0], &[2.0]).unwrap();
    let pressure =
        sonium_compute::PointJacobiBackend::try_from_diagonal(backend, &[1.0, 1.0, 2.0]).unwrap();
    let mut action =
        SaddlePointJacobiSchurBackend::try_from_host_csr(backend, matrix, schur, config())
            .unwrap()
            .with_owned_pressure_preconditioner(Box::new(pressure));
    let changed = HostCsrRef {
        values: &[8.0, 2.0, 4.0, 1.0, 0.1],
        ..matrix
    };
    action
        .try_update_numeric_host(backend, changed, schur)
        .unwrap();
    let wrong_pressure = HostCsrRef {
        values: &[3.0],
        ..schur
    };
    assert!(
        action
            .try_update_numeric_host(backend, changed, wrong_pressure)
            .is_err()
    );
    let mut rhs = pool.try_alloc::<f64>(6).unwrap();
    let mut output = pool.try_alloc::<f64>(6).unwrap();
    backend
        .try_copy_h2d(&[1.0, 1.0, 1.0, -2.0, -2.0, -2.0], rhs.raw_mut())
        .unwrap();
    action
        .try_apply_colmajor(backend, 2, rhs.raw(), output.raw_mut())
        .unwrap();
    let mut host = [0.0; 6];
    backend.try_copy_d2h(output.raw(), &mut host).unwrap();
    assert_eq!(host, [0.0, 0.125, 0.5, 0.0, -0.25, -1.0]);
    let replacement =
        sonium_compute::PointJacobiBackend::try_from_diagonal(backend, &[1.0, 1.0, 3.0]).unwrap();
    action
        .try_update_numeric_host_with_pressure(
            backend,
            changed,
            wrong_pressure,
            Box::new(replacement),
        )
        .unwrap();
    action
        .try_apply_colmajor(backend, 2, rhs.raw(), output.raw_mut())
        .unwrap();
    backend.try_copy_d2h(output.raw(), &mut host).unwrap();
    for (a, b) in host.iter().zip([
        1.0 / 24.0,
        1.0 / 6.0,
        1.0 / 3.0,
        -1.0 / 12.0,
        -1.0 / 3.0,
        -2.0 / 3.0,
    ]) {
        assert!((a - b).abs() < 1e-14);
    }
}

#[test]
fn external_pressure_survives_velocity_refresh_and_rejects_pressure_change() {
    external_refresh(CpuBackend::new());
}
