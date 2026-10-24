//! Focused runtime parity for production FEM RHS dispatches.
//!
//! This deliberately stays a two-frequency/two-source fixture. It exercises
//! the real solver paths while keeping the regression separate from QA
//! binaries and production-scale room simulations.

use sonium_sim_room::backend_parity::compare_backend_outputs;
#[cfg(feature = "cuda")]
use sonium_sim_room::fem_runner::run_fem_gpu_report_with_failure_injection;
#[cfg(any(feature = "cuda", feature = "metal"))]
use sonium_sim_room::fem_runner::run_fem_with_mesh_report;
#[cfg(any(feature = "cuda", feature = "metal"))]
use sonium_sim_room::fem_runner::{
    FemBackendExecutionError, FemBackendFailureDiagnostics, FemPhysicalEvidenceStatus,
    FemRunCancellation, GpuPrecision, GpuPrecond,
};
use sonium_sim_room::fem_runner::{FemMeshConfig, run_fem_with_mesh};
#[cfg(feature = "cuda")]
use sonium_sim_room::fem_runner::{run_fem_gpu_report, run_fem_gpu_report_cancellation};
#[cfg(feature = "metal")]
use sonium_sim_room::fem_runner::{run_fem_metal_report, run_fem_metal_report_cancellation};
use sonium_xem::{
    BoundaryConfig, Point3D, RectangularRoom, RoomGeometry, RoomSimulation, Source, SurfaceConfig,
};

fn fixture() -> RoomSimulation {
    let mut simulation = RoomSimulation::new(
        RoomGeometry::Rectangular(RectangularRoom::new(1.6, 1.5, 1.4)),
        vec![
            Source::omnidirectional(Point3D::new(0.4, 0.35, 0.35), 1.0)
                .with_name("left".to_string()),
            Source::omnidirectional(Point3D::new(0.5, 0.9, 0.45), 0.7)
                .with_name("right".to_string()),
        ],
        vec![Point3D::new(1.2, 1.1, 1.0), Point3D::new(1.0, 0.55, 1.05)],
    );
    let wall = SurfaceConfig::Admittance {
        real: 1.0,
        imag: 0.0,
    };
    simulation.boundaries = BoundaryConfig {
        floor: wall.clone(),
        ceiling: wall.clone(),
        walls: wall,
        front_wall: None,
        back_wall: None,
        left_wall: None,
        right_wall: None,
    };
    simulation.frequencies = vec![200.0];
    simulation
}

#[test]
fn fem_block_rhs_matches_per_source_dispatch() {
    let simulation = fixture();
    let mut per_source_config = FemMeshConfig::default();
    per_source_config.fine_radius = 0.4;
    per_source_config.block_source_solve = false;
    let mut block_config = per_source_config.clone();
    block_config.block_source_solve = true;

    let per_source = run_fem_with_mesh(&simulation, &per_source_config).unwrap();
    let block = run_fem_with_mesh(&simulation, &block_config).unwrap();
    let report = compare_backend_outputs(
        "fem-per-source",
        &per_source,
        "fem-block-rhs",
        &block,
        0.0,
        1.0e-12,
    )
    .unwrap();

    assert_eq!(report.sample_count, 4);
    report.require_within(2.0e-6, 2.0e-5).unwrap();
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn accelerator_fixture() -> sonium_xem::RoomSimulation {
    sonium_xem::RoomSimulation::with_frequencies(
        sonium_xem::RoomGeometry::Rectangular(sonium_xem::RectangularRoom::new(0.9, 0.8, 0.7)),
        vec![
            sonium_xem::Source::omnidirectional(sonium_xem::Point3D::new(0.68, 0.53, 0.42), 1.0)
                .with_name("source-a".into()),
            sonium_xem::Source::omnidirectional(sonium_xem::Point3D::new(0.59, 0.24, 0.51), 1.0)
                .with_name("source-b".into()),
        ],
        vec![
            sonium_xem::Point3D::new(0.68, 0.53, 0.42),
            sonium_xem::Point3D::new(0.59, 0.24, 0.51),
        ],
        80.0,
        120.0,
        2,
    )
}

#[cfg(feature = "cuda")]
fn accelerator_resident_fixture() -> sonium_xem::RoomSimulation {
    let mut simulation = accelerator_fixture();
    // Keep the mesh, sources, and listener layout shape-matched with the
    // existing production CUDA parity lane while extending the sweep to four
    // frequencies.  This makes the warm-state counters observable beyond the
    // first cached solve without introducing a synthetic operator.
    simulation.frequencies = vec![80.0, 100.0, 120.0, 140.0];
    simulation
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn assert_physical_parity(
    cpu: &sonium_sim_room::fem_runner::FemRunDiagnostics,
    accelerator: &sonium_sim_room::fem_runner::FemRunDiagnostics,
) {
    assert_eq!(
        cpu.frequency_stats.len(),
        accelerator.frequency_stats.len(),
        "CPU and accelerator frequency diagnostics differ"
    );
    assert_eq!(
        accelerator.physical_evidence_status,
        FemPhysicalEvidenceStatus::Complete
    );
    for (index, (cpu_stats, accelerator_stats)) in cpu
        .frequency_stats
        .iter()
        .zip(&accelerator.frequency_stats)
        .enumerate()
    {
        assert_eq!(cpu_stats.frequency_hz, accelerator_stats.frequency_hz);
        assert!(cpu_stats.converged && accelerator_stats.converged);
        assert!(cpu_stats.max_residual.is_finite());
        assert!(accelerator_stats.max_residual.is_finite());
        assert!(accelerator_stats.max_residual <= 1.0e-4);
        let transfer_stats = accelerator_stats
            .transfer_stats
            .expect("accelerator frequency must publish transfer telemetry");
        assert!(transfer_stats.d2h_bytes > 0);

        let cpu_work = cpu_stats.operator_work.unwrap();
        let accelerator_work = accelerator_stats.operator_work.unwrap();
        let cpu_power = cpu_stats.physical_power.unwrap();
        let accelerator_power = accelerator_stats.physical_power.unwrap();
        let close = |label: &str, left: f64, right: f64| {
            let scale = left.abs().max(right.abs()).max(1.0e-12);
            assert!(
                (left - right).abs() / scale <= 0.25,
                "{label} parity failed at frequency {index}: CPU={left:?}, accelerator={right:?}"
            );
        };
        close(
            "operator source work",
            cpu_work.source_work,
            accelerator_work.source_work,
        );
        close(
            "operator volume damping work",
            cpu_work.volume_damping_work,
            accelerator_work.volume_damping_work,
        );
        close(
            "operator boundary work",
            cpu_work.boundary_work,
            accelerator_work.boundary_work,
        );
        // Balance residuals are cancellation remainders. Comparing two values
        // that are both near zero with the ordinary relative-error helper is
        // ill-conditioned (and can reject a smaller accelerator remainder).
        // Scale their difference by the physical work terms instead.
        let operator_scale = cpu_work
            .source_work
            .abs()
            .max(cpu_work.volume_damping_work.abs())
            .max(cpu_work.boundary_work.abs())
            .max(accelerator_work.source_work.abs())
            .max(accelerator_work.volume_damping_work.abs())
            .max(accelerator_work.boundary_work.abs())
            .max(1.0e-1);
        assert!(
            (cpu_work.balance_residual - accelerator_work.balance_residual).abs()
                <= 1.0e-6 * operator_scale,
            "operator balance residual parity failed at frequency {index}: CPU={:?}, accelerator={:?}, work scale={operator_scale:?}",
            cpu_work.balance_residual,
            accelerator_work.balance_residual,
        );
        close(
            "physical source power",
            cpu_power.source_power_w,
            accelerator_power.source_power_w,
        );
        close(
            "physical volume damping power",
            cpu_power.volume_damping_power_w,
            accelerator_power.volume_damping_power_w,
        );
        close(
            "physical boundary power",
            cpu_power.boundary_power_w,
            accelerator_power.boundary_power_w,
        );
        let power_scale = cpu_power
            .source_power_w
            .abs()
            .max(cpu_power.volume_damping_power_w.abs())
            .max(cpu_power.boundary_power_w.abs())
            .max(accelerator_power.source_power_w.abs())
            .max(accelerator_power.volume_damping_power_w.abs())
            .max(accelerator_power.boundary_power_w.abs())
            .max(1.0e-1);
        assert!(
            (cpu_power.balance_residual_w - accelerator_power.balance_residual_w).abs()
                <= 1.0e-6 * power_scale,
            "physical balance residual parity failed at frequency {index}: CPU={:?}, accelerator={:?}, power scale={power_scale:?}",
            cpu_power.balance_residual_w,
            accelerator_power.balance_residual_w,
        );

        let cpu_flux = cpu_stats.boundary_flux.as_ref().unwrap();
        let accelerator_flux = accelerator_stats.boundary_flux.as_ref().unwrap();
        assert_eq!(cpu_flux.samples.len(), accelerator_flux.samples.len());
        for (cpu_sample, accelerator_sample) in
            cpu_flux.samples.iter().zip(&accelerator_flux.samples)
        {
            assert_eq!(cpu_sample.boundary_tag, accelerator_sample.boundary_tag);
            close(
                "boundary active power",
                cpu_sample.active_power_w,
                accelerator_sample.active_power_w,
            );
        }
        close(
            "boundary total power",
            cpu_flux.total_power_w,
            accelerator_flux.total_power_w,
        );
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn assert_reciprocal_output(output: &sonium_sim_room::SimulationOutput) {
    assert_eq!(output.pressures.len(), 2);
    assert!(output.pressures.iter().all(|source| source.len() == 2));
    for frequency_index in 0..output.frequencies.len() {
        for source_index in 0..2 {
            for listener_index in (source_index + 1)..2 {
                let left = output.pressures[source_index][listener_index][frequency_index];
                let right = output.pressures[listener_index][source_index][frequency_index];
                let scale = left.norm().max(right.norm()).max(1.0e-12);
                assert!(
                    (left - right).norm() / scale <= 2.0e-4,
                    "reciprocity failed at frequency index {frequency_index}: {left:?} vs {right:?}"
                );
            }
        }
    }
}

#[cfg(feature = "cuda")]
fn write_sr1_evidence(
    path: &std::path::Path,
    workload: &str,
    diagnostics: &sonium_sim_room::fem_runner::FemRunDiagnostics,
    parity: &sonium_sim_room::backend_parity::BackendParityReport,
) {
    let Some(execution) = diagnostics.execution.as_ref() else {
        panic!("CUDA SR1 evidence requires execution provenance");
    };
    let frequency_stats = diagnostics
        .frequency_stats
        .iter()
        .map(|stats| {
            let transfer = stats
                .transfer_stats
                .expect("CUDA SR1 evidence requires per-frequency transfers");
            serde_json::json!({
                "frequency_hz": stats.frequency_hz,
                "iterations": stats.iterations,
                "inner_iterations": stats.inner_iterations,
                "retries": stats.retries,
                "converged": stats.converged,
                "max_residual": stats.max_residual,
                "preconditioner_age": stats.preconditioner_age,
                "preconditioner_rebuild_reason": stats.preconditioner_rebuild_reason,
                "fallback_stage": stats.fallback_stage,
                "block_solve": stats.block_solve,
                "transfer": {
                    "h2d_operations": transfer.h2d_operations,
                    "h2d_bytes": transfer.h2d_bytes,
                    "d2h_operations": transfer.d2h_operations,
                    "d2h_bytes": transfer.d2h_bytes,
                    "d2d_operations": transfer.d2d_operations,
                    "d2d_bytes": transfer.d2d_bytes,
                },
                "allocation_operations": stats.allocation_operations,
                "allocation_high_water_bytes": stats.allocation_high_water_bytes,
                "synchronization_calls": stats.synchronization_calls,
            })
        })
        .collect::<Vec<_>>();
    let artifact = serde_json::json!({
        "schema_version": 1,
        "artifact": "sonium-sim-room.sr1-cuda",
        "status": "passed",
        "workload": workload,
        "backend": {
            "requested": execution.requested_backend,
            "actual": execution.actual_backend,
            "precision": execution.precision,
            "fallback_status": execution.fallback_status,
            "host_architecture": execution.host_architecture,
        },
        "memory": {
            "estimated_peak_bytes": execution.estimated_peak_memory_bytes,
            "device_free_bytes": execution.device_free_memory_bytes,
            "device_total_bytes": execution.device_total_memory_bytes,
        },
        "setup_transfer": {
            "h2d_operations": execution.setup_transfer_stats.h2d_operations,
            "h2d_bytes": execution.setup_transfer_stats.h2d_bytes,
            "d2h_operations": execution.setup_transfer_stats.d2h_operations,
            "d2h_bytes": execution.setup_transfer_stats.d2h_bytes,
            "d2d_operations": execution.setup_transfer_stats.d2d_operations,
            "d2d_bytes": execution.setup_transfer_stats.d2d_bytes,
        },
        "total_transfer": {
            "h2d_operations": execution.total_transfer_stats.h2d_operations,
            "h2d_bytes": execution.total_transfer_stats.h2d_bytes,
            "d2h_operations": execution.total_transfer_stats.d2h_operations,
            "d2h_bytes": execution.total_transfer_stats.d2h_bytes,
            "d2d_operations": execution.total_transfer_stats.d2d_operations,
            "d2d_bytes": execution.total_transfer_stats.d2d_bytes,
        },
        "parity": {
            "reference_backend": parity.reference_backend,
            "candidate_backend": parity.candidate_backend,
            "sample_count": parity.sample_count,
            "rms_relative_error": parity.rms_relative_error,
            "max_relative_error": parity.max_relative_error,
        },
        "cancellation": {
            "status": "frequency-boundary cooperative",
            "pre_setup_test": "cuda_cancellation_rejects_before_device_setup",
            "mid_frequency_interrupt": "not-exercised",
        },
        "frequency_stats": frequency_stats,
    });
    std::fs::create_dir_all(path.parent().expect("SR1 artifact has a parent directory"))
        .expect("SR1 artifact directory creates");
    std::fs::write(
        path,
        serde_json::to_vec_pretty(&artifact).expect("SR1 artifact serializes"),
    )
    .expect("SR1 artifact writes");
}

#[cfg(feature = "cuda")]
fn write_sr1_cancellation_evidence(
    path: &std::path::Path,
    backend: &str,
    reason: &str,
    frequency_index: usize,
    failure: &FemBackendFailureDiagnostics,
    context_reacquired: bool,
    success_artifact_published: bool,
) {
    let artifact = serde_json::json!({
        "schema_version": 1,
        "artifact": "sonium-sim-room.sr1-cuda-cancellation",
        "status": "passed",
        "backend": {
            "requested": "cuda",
            "actual": backend,
            "precision": "f64-outer (IR)",
            "fallback_status": "none",
        },
        "cancellation": {
            "stage": "mid-frequency",
            "frequency_index": frequency_index,
            "typed_error": "Cancelled",
            "backend": backend,
            "reason": reason,
            "partial_result_published": false,
            "success_artifact_published": success_artifact_published,
        },
        "failure_diagnostics": {
            "backend": failure.backend,
            "stage": failure.stage.label(),
            "operation": failure.operation,
            "detail": failure.detail,
            "frequency_index": failure.frequency_index,
            "source_index": failure.source_index,
            "partial_result_published": failure.partial_result_published,
            "resources_released_on_unwind": failure.resources_released_on_unwind,
            "context_recovery": failure.context_recovery.label(),
        },
        "cleanup": {
            "owned_resources_drop": "Rust error-unwind Drop",
            "context_reacquired": context_reacquired,
            "device_allocation_leak_counter": "not-exposed by sonium-compute",
            "proof": "error-unwind-drop-and-context-reacquisition",
        },
    });
    let parent = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    std::fs::create_dir_all(parent).expect("SR1 cancellation artifact directory creates");
    std::fs::write(
        path,
        serde_json::to_vec_pretty(&artifact).expect("SR1 cancellation artifact serializes"),
    )
    .expect("SR1 cancellation artifact writes");
}

#[cfg(feature = "cuda")]
fn write_sr1_resident_recovery_evidence(
    path: &std::path::Path,
    backend: &str,
    reason: &str,
    failure: &FemBackendFailureDiagnostics,
    context_reacquired: bool,
    success_artifact_published: bool,
    retry: &sonium_sim_room::fem_runner::FemRunDiagnostics,
) {
    let execution = retry
        .execution
        .as_ref()
        .expect("SR1 resident recovery requires retry execution provenance");
    let retry_frequency_stats = retry
        .frequency_stats
        .iter()
        .map(|stats| {
            let transfer = stats
                .transfer_stats
                .expect("SR1 resident recovery requires per-frequency transfers");
            serde_json::json!({
                "frequency_hz": stats.frequency_hz,
                "converged": stats.converged,
                "max_residual": stats.max_residual,
                "preconditioner_age": stats.preconditioner_age,
                "preconditioner_rebuild_reason": stats.preconditioner_rebuild_reason,
                "transfer": {
                    "h2d_operations": transfer.h2d_operations,
                    "h2d_bytes": transfer.h2d_bytes,
                    "d2h_operations": transfer.d2h_operations,
                    "d2h_bytes": transfer.d2h_bytes,
                    "d2d_operations": transfer.d2d_operations,
                    "d2d_bytes": transfer.d2d_bytes,
                },
                "allocation_operations": stats.allocation_operations,
                "allocation_high_water_bytes": stats.allocation_high_water_bytes,
                "synchronization_calls": stats.synchronization_calls,
            })
        })
        .collect::<Vec<_>>();
    let high_water = retry
        .frequency_stats
        .iter()
        .map(|stats| stats.allocation_high_water_bytes)
        .max()
        .expect("SR1 resident recovery requires allocation telemetry");
    let artifact = serde_json::json!({
        "schema_version": 1,
        "artifact": "sonium-sim-room.sr1-cuda-resident-recovery",
        "status": "passed",
        "backend": {
            "requested": "cuda",
            "actual": backend,
            "precision": "f64-outer (IR)",
            "fallback_status": "none",
        },
        "cancellation": {
            "stage": "mid-frequency",
            "frequency_index": 2,
            "committed_frequency_count": 2,
            "typed_error": "Cancelled",
            "backend": backend,
            "reason": reason,
            "partial_result_published": false,
            "success_artifact_published": success_artifact_published,
        },
        "failure_diagnostics": {
            "backend": failure.backend,
            "stage": failure.stage.label(),
            "operation": failure.operation,
            "detail": failure.detail,
            "frequency_index": failure.frequency_index,
            "source_index": failure.source_index,
            "partial_result_published": failure.partial_result_published,
            "resources_released_on_unwind": failure.resources_released_on_unwind,
            "context_recovery": failure.context_recovery.label(),
        },
        "cleanup": {
            "owned_resources_drop": "Rust error-unwind Drop",
            "context_reacquired": context_reacquired,
            "device_allocation_leak_counter": "not-exposed by sonium-compute",
            "proof": "error-unwind-drop-and-context-reacquisition",
        },
        "retry": {
            "mode": "fresh-context",
            "context_recreated": true,
            "success": true,
            "backend": {
                "requested": execution.requested_backend,
                "actual": execution.actual_backend,
                "precision": execution.precision,
                "fallback_status": execution.fallback_status,
            },
            "frequency_count": retry.frequency_stats.len(),
            "all_frequencies_converged": retry.frequency_stats.iter().all(|stats| stats.converged),
            "preconditioner_ages": retry
                .frequency_stats
                .iter()
                .map(|stats| stats.preconditioner_age)
                .collect::<Vec<_>>(),
            "preconditioner_rebuild_reasons": retry
                .frequency_stats
                .iter()
                .map(|stats| stats.preconditioner_rebuild_reason.clone())
                .collect::<Vec<_>>(),
            "allocation_high_water_bytes": high_water,
            "frequency_stats": retry_frequency_stats,
        },
    });
    let parent = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    std::fs::create_dir_all(parent).expect("SR1 resident recovery artifact directory creates");
    std::fs::write(
        path,
        serde_json::to_vec_pretty(&artifact).expect("SR1 resident recovery artifact serializes"),
    )
    .expect("SR1 resident recovery artifact writes");
}

#[cfg(feature = "cuda")]
fn write_sr1_failure_evidence(
    path: &std::path::Path,
    point: sonium_compute::CudaFailureInjectionPoint,
    failure: &FemBackendFailureDiagnostics,
    retry: &sonium_sim_room::fem_runner::FemRunDiagnostics,
    retry_context_recreated: bool,
    retry_artifact_published: bool,
) {
    let execution = retry
        .execution
        .as_ref()
        .expect("SR1 failure evidence requires retry execution provenance");
    let high_water = retry
        .frequency_stats
        .iter()
        .map(|stats| stats.allocation_high_water_bytes)
        .max()
        .expect("SR1 failure evidence requires retry frequency telemetry");
    let (point_label, expected_operation, expected_recovery) = match point {
        sonium_compute::CudaFailureInjectionPoint::Launch => {
            ("launch", "cuda_launch", "synchronize-before-reuse")
        }
        sonium_compute::CudaFailureInjectionPoint::DeviceLoss => {
            ("device-loss", "cuda_device_loss", "recreate")
        }
        other => panic!("SR1 failure evidence only accepts launch/device-loss, got {other:?}"),
    };
    let artifact = serde_json::json!({
        "schema_version": 1,
        "artifact": "sonium-sim-room.sr1-cuda-failure-injection",
        "status": "passed",
        "injection": {
            "point": point_label,
            "one_shot": true,
            "native_backend": "cuda",
            "expected_operation": expected_operation,
        },
        "failure_diagnostics": {
            "backend": failure.backend,
            "stage": failure.stage.label(),
            "operation": failure.operation,
            "detail": failure.detail,
            "partial_result_published": failure.partial_result_published,
            "resources_released_on_unwind": failure.resources_released_on_unwind,
            "context_recovery": failure.context_recovery.label(),
        },
        "publication": {
            "partial_result_published": false,
            "success_artifact_published_before_retry": false,
            "retry_artifact_published": retry_artifact_published,
        },
        "unwind": {
            "owned_resources_drop": "Rust error-unwind Drop",
            "live_allocation_after_unwind": "not-observable after backend ownership ends",
            "allocation_ledger": "sonium-compute backend ledger releases owned buffers on Drop",
        },
        "recovery": {
            "required": expected_recovery,
            "retry_mode": "recreated-context",
            "context_recreated": retry_context_recreated,
            "retry_success": true,
        },
        "retry_success": {
            "backend": {
                "requested": execution.requested_backend,
                "actual": execution.actual_backend,
                "fallback_status": execution.fallback_status,
                "precision": execution.precision,
            },
            "frequency_count": retry.frequency_stats.len(),
            "all_frequencies_converged": retry.frequency_stats.iter().all(|stats| stats.converged),
            "allocation_high_water_bytes": high_water,
            "estimated_peak_memory_bytes": execution.estimated_peak_memory_bytes,
        },
    });
    let parent = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    std::fs::create_dir_all(parent).expect("SR1 failure artifact directory creates");
    std::fs::write(
        path,
        serde_json::to_vec_pretty(&artifact).expect("SR1 failure artifact serializes"),
    )
    .expect("SR1 failure artifact writes");
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn accelerator_mesh_config() -> FemMeshConfig {
    FemMeshConfig {
        fine_radius: 0.2,
        modal_damping_ratio: 0.02,
        block_source_solve: false,
        rebuild_precond_every: 2,
        ..FemMeshConfig::default()
    }
}

#[cfg(feature = "cuda")]
fn accelerator_resident_mesh_config() -> FemMeshConfig {
    let mut config = accelerator_mesh_config();
    // The production CUDA loop keeps the assembled basis/operator buffers
    // resident and refreshes values in place.  Rebuild the AMG preconditioner
    // only once so the three subsequent frequencies exercise that cache.
    config.rebuild_precond_every = 4;
    config
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_fem_matches_cpu_backend_with_checked_parity_budget() {
    let _backend = match sonium_compute::CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            if std::env::var_os("SONIUM_SIM_ROOM_REQUIRE_CUDA").is_some() {
                panic!("CUDA is required for this evidence lane but device setup failed: {error}");
            }
            eprintln!("no CUDA device — skipping: {error}");
            return;
        }
    };

    let simulation = accelerator_fixture();
    let mesh_config = accelerator_mesh_config();
    let (cpu, cpu_diagnostics) = run_fem_with_mesh_report(&simulation, &mesh_config).unwrap();
    let (cuda, diagnostics) = match run_fem_gpu_report(
        &simulation,
        &mesh_config,
        GpuPrecision::F64Outer,
        GpuPrecond::Auto.resolve_for_cuda(),
        0,
    ) {
        Ok(result) => result,
        Err(error) => {
            assert!(
                error.downcast_ref::<FemBackendExecutionError>().is_some(),
                "CUDA setup/unsupported failures must remain typed: {error:?}"
            );
            if std::env::var_os("SONIUM_SIM_ROOM_REQUIRE_CUDA").is_some() {
                panic!("CUDA is required for this evidence lane but the FEM run failed: {error}");
            }
            eprintln!("native CUDA FEM unavailable — skipping: {error}");
            return;
        }
    };
    assert!(diagnostics.validate().is_ok());
    assert_physical_parity(&cpu_diagnostics, &diagnostics);
    assert_reciprocal_output(&cpu);
    assert_reciprocal_output(&cuda);
    let execution = diagnostics
        .execution
        .as_ref()
        .expect("CUDA run must publish execution provenance");
    assert_eq!(execution.requested_backend, "cuda");
    assert_eq!(execution.actual_backend, "cuda");
    assert_eq!(execution.fallback_status, "none");
    assert_eq!(execution.precision, "f64-outer (IR)");
    assert!(execution.estimated_peak_memory_bytes.is_some());
    assert!(execution.device_total_memory_bytes.is_some());
    assert!(execution.setup_transfer_stats.h2d_bytes > 0);
    assert!(execution.total_transfer_stats.d2h_bytes > 0);
    assert_eq!(diagnostics.frequency_stats.len(), 2);
    assert!(
        diagnostics
            .frequency_stats
            .iter()
            .all(|stats| stats.transfer_stats.is_some())
    );
    assert!(
        diagnostics
            .frequency_stats
            .iter()
            .all(|stats| stats.allocation_high_water_bytes > 0)
    );

    let report =
        compare_backend_outputs("cpu-f64", &cpu, "cuda-f64-outer", &cuda, 0.0, 1.0e-12).unwrap();
    assert_eq!(report.sample_count, 8);
    report.require_within(0.25, 0.25).unwrap();
    if let Some(path) = std::env::var_os("SONIUM_SIM_ROOM_SR1_ARTIFACT") {
        write_sr1_evidence(
            std::path::Path::new(&path),
            "f64-outer",
            &diagnostics,
            &report,
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_fem_block_rhs_publishes_physical_ledgers() {
    check_cuda_block_physical_ledgers(GpuPrecond::PointJacobi, "SONIUM_SIM_ROOM_SR1_ARTIFACT", 6.0);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_fem_amg_block_rhs_publishes_physical_ledgers() {
    check_cuda_block_physical_ledgers(GpuPrecond::Amg, "SONIUM_SIM_ROOM_AMG_BLOCK_ARTIFACT", 6.0);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_fem_amg_block_refined_mesh_publishes_physical_ledgers() {
    check_cuda_block_physical_ledgers(
        GpuPrecond::Amg,
        "SONIUM_SIM_ROOM_AMG_BLOCK_REFINED_ARTIFACT",
        12.0,
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_fem_polynomial_block_rhs_publishes_physical_ledgers() {
    check_cuda_block_physical_ledgers(
        GpuPrecond::PolynomialJacobi { degree: 3 },
        "SONIUM_SIM_ROOM_POLYNOMIAL_BLOCK_ARTIFACT",
        6.0,
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_fem_chebyshev_block_rhs_publishes_physical_ledgers() {
    check_cuda_block_physical_ledgers(
        GpuPrecond::Chebyshev { degree: 3 },
        "SONIUM_SIM_ROOM_CHEBYSHEV_BLOCK_ARTIFACT",
        6.0,
    );
}

#[cfg(feature = "cuda")]
fn check_cuda_block_physical_ledgers(
    preconditioner: GpuPrecond,
    artifact_env: &str,
    resolution: f64,
) {
    let _ = env_logger::builder().is_test(true).try_init();
    let _backend = match sonium_compute::CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            if std::env::var_os("SONIUM_SIM_ROOM_REQUIRE_CUDA").is_some() {
                panic!("CUDA is required for this evidence lane but device setup failed: {error}");
            }
            eprintln!("no CUDA device — skipping: {error}");
            return;
        }
    };
    // Exercise retained block scratch across numeric updates and a scheduled
    // preconditioner rebuild. CPU parity checks every source and frequency.
    let simulation = accelerator_resident_fixture();
    let mut mesh_config = accelerator_mesh_config();
    mesh_config.elements_per_wavelength = resolution;
    let (cpu, cpu_diagnostics) = run_fem_with_mesh_report(&simulation, &mesh_config).unwrap();
    let (cuda, diagnostics) = match run_fem_gpu_report(
        &simulation,
        &mesh_config,
        GpuPrecision::F32Pure,
        preconditioner,
        0,
    ) {
        Ok(result) => result,
        Err(error) => {
            assert!(
                error.downcast_ref::<FemBackendExecutionError>().is_some(),
                "CUDA setup/unsupported failures must remain typed: {error:?}"
            );
            if std::env::var_os("SONIUM_SIM_ROOM_REQUIRE_CUDA").is_some() {
                panic!("CUDA is required for this evidence lane but the FEM run failed: {error}");
            }
            eprintln!("native CUDA block-RHS FEM unavailable — skipping: {error}");
            return;
        }
    };
    assert_eq!(
        diagnostics.execution.as_ref().unwrap().precision,
        "f32-pure"
    );
    assert_eq!(
        diagnostics.execution.as_ref().unwrap().fallback_status,
        "none"
    );
    assert_physical_parity(&cpu_diagnostics, &diagnostics);
    assert_reciprocal_output(&cuda);
    assert_eq!(diagnostics.frequency_stats.len(), 4);
    assert!(
        diagnostics.frequency_stats.iter().all(|stats| matches!(
            stats.block_solve,
            sonium_sim_room::fem_runner::FemBlockSolveOutcome::Accepted { .. }
        )),
        "block results must pass certification without producer scalar recovery: {:?}",
        diagnostics
            .frequency_stats
            .iter()
            .map(|stats| &stats.block_solve)
            .collect::<Vec<_>>()
    );
    assert!(
        diagnostics
            .frequency_stats
            .iter()
            .all(|stats| stats.max_residual <= 1.0e-6)
    );
    if preconditioner == GpuPrecond::PointJacobi {
        assert!(
            diagnostics.frequency_stats.iter().any(|stats| matches!(
                stats.block_solve,
                sonium_sim_room::fem_runner::FemBlockSolveOutcome::Accepted {
                    correction_steps: 1..
                }
            )),
            "fixture must exercise promoted block correction"
        );
        let cold_allocations = diagnostics.frequency_stats[0].allocation_operations;
        assert!(
            diagnostics.frequency_stats[1..]
                .iter()
                // Allow one extra Krylov vector and two reduction pointer buffers
                // as convergence demand grows; Jacobi scratch must remain cached.
                .all(|stats| stats.allocation_operations <= 3
                    && stats.allocation_operations < cold_allocations),
            "warm block solves must reuse scratch: {:?}",
            diagnostics
                .frequency_stats
                .iter()
                .map(|stats| stats.allocation_operations)
                .collect::<Vec<_>>()
        );
    }
    if matches!(preconditioner, GpuPrecond::PolynomialJacobi { .. } | GpuPrecond::Chebyshev { .. }) {
        let allowance = if matches!(preconditioner, GpuPrecond::PolynomialJacobi { .. }) { 3 } else { 0 };
        assert!(diagnostics.frequency_stats[1..].iter().all(|stats| stats.allocation_operations <= allowance),
            "numeric polynomial refresh must retain device storage: {:?}",
            diagnostics.frequency_stats.iter().map(|stats| stats.allocation_operations).collect::<Vec<_>>());
    }
    if preconditioner == GpuPrecond::Amg {
        for index in [1, 3] {
            let stats = &diagnostics.frequency_stats[index];
            assert_eq!(stats.preconditioner_age, 1);
            assert_eq!(stats.preconditioner_rebuild_reason, "cached");
            assert_eq!(stats.allocation_operations, 0, "cached AMG/block solve must reuse device storage");
        }
    }
    assert!(
        diagnostics
            .frequency_stats
            .iter()
            .all(|stats| stats.allocation_high_water_bytes > 0)
    );
    let report =
        compare_backend_outputs("cpu-f64", &cpu, "cuda-f32-block-rhs", &cuda, 0.0, 1.0e-12)
            .unwrap();
    report.require_within(2.0e-2, 5.0e-2).unwrap();
    if let Some(path) = std::env::var_os(artifact_env) {
        write_sr1_evidence(
            std::path::Path::new(&path),
            "f32-pure-block-rhs",
            &diagnostics,
            &report,
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_fem_f32_scalar_four_frequency_reuses_krylov_workspace() {
    if let Err(error) = sonium_compute::CudaBackend::new(0) {
        assert!(
            std::env::var_os("SONIUM_SIM_ROOM_REQUIRE_CUDA").is_none(),
            "CUDA is required for this evidence lane: {error}"
        );
        eprintln!("no CUDA device — skipping: {error}");
        return;
    }
    let simulation = accelerator_resident_fixture();
    let mesh_config = accelerator_resident_mesh_config();
    assert!(!mesh_config.block_source_solve);
    let (cpu, cpu_diagnostics) = run_fem_with_mesh_report(&simulation, &mesh_config).unwrap();
    let (cuda, diagnostics) = run_fem_gpu_report(
        &simulation,
        &mesh_config,
        GpuPrecision::F32Pure,
        GpuPrecond::Auto.resolve_for_cuda(),
        0,
    )
    .expect("persistent scalar FP32 room sweep must converge");
    diagnostics.validate().unwrap();
    assert_physical_parity(&cpu_diagnostics, &diagnostics);
    assert_reciprocal_output(&cuda);
    assert_eq!(diagnostics.frequency_stats.len(), 4);
    assert!(
        diagnostics
            .frequency_stats
            .iter()
            .all(|stats| stats.converged)
    );
    assert_eq!(
        diagnostics
            .frequency_stats
            .iter()
            .map(|stats| stats.preconditioner_age)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
    let report = compare_backend_outputs(
        "cpu-f64",
        &cpu,
        "cuda-f32-persistent-scalar",
        &cuda,
        0.0,
        1e-12,
    )
    .unwrap();
    assert_eq!(report.sample_count, 16);
    report.require_within(0.25, 0.25).unwrap();
    assert!(
        diagnostics.frequency_stats.iter().all(|stats| stats.max_residual <= 1.0e-6),
        "f32 promotion must certify the requested solve accuracy"
    );
    // The first frequency warms this fixture's largest Krylov demand.
    // Matrix/RHS descriptors and scratch survive all numeric updates.
    assert!(
        diagnostics.frequency_stats.iter().skip(1).all(|stats| stats.allocation_operations == 0),
        "scalar f32 correction must retain its sparse execution state"
    );
    if let Some(path) = std::env::var_os("SONIUM_SIM_ROOM_F32_RESIDENT_ARTIFACT") {
        write_sr1_evidence(
            std::path::Path::new(&path),
            "f32-resident-certified-4-frequency",
            &diagnostics,
            &report,
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_fem_four_frequency_resident_state_reuses_preconditioner() {
    let _backend = match sonium_compute::CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            if std::env::var_os("SONIUM_SIM_ROOM_REQUIRE_CUDA").is_some() {
                panic!("CUDA is required for this evidence lane but device setup failed: {error}");
            }
            eprintln!("no CUDA device — skipping: {error}");
            return;
        }
    };

    let simulation = accelerator_resident_fixture();
    let mesh_config = accelerator_resident_mesh_config();
    let (cpu, cpu_diagnostics) = run_fem_with_mesh_report(&simulation, &mesh_config).unwrap();
    let (cuda, diagnostics) = run_fem_gpu_report(
        &simulation,
        &mesh_config,
        GpuPrecision::F64Outer,
        GpuPrecond::Auto.resolve_for_cuda(),
        0,
    )
    .expect("native four-frequency CUDA resident run must complete");

    diagnostics
        .validate()
        .expect("resident CUDA diagnostics must remain valid");
    assert_physical_parity(&cpu_diagnostics, &diagnostics);
    assert_reciprocal_output(&cpu);
    assert_reciprocal_output(&cuda);
    assert_eq!(diagnostics.frequency_stats.len(), 4);
    let ages = diagnostics
        .frequency_stats
        .iter()
        .map(|stats| stats.preconditioner_age)
        .collect::<Vec<_>>();
    assert_eq!(ages, vec![0, 1, 2, 3]);
    let rebuild_reasons = diagnostics
        .frequency_stats
        .iter()
        .map(|stats| stats.preconditioner_rebuild_reason.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        rebuild_reasons,
        vec!["scheduled", "cached", "cached", "cached"]
    );
    assert!(diagnostics.frequency_stats.iter().all(|stats| {
        stats.converged
            && stats
                .transfer_stats
                .is_some_and(|transfer| transfer.d2h_bytes > 0 && transfer.d2d_operations > 0)
            && stats.allocation_high_water_bytes > 0
    }));

    // This fixed fixture warms the largest Krylov demand at its first frequency.
    // Numeric updates must reuse both matrix/RHS descriptors and Krylov scratch.
    assert!(
        diagnostics.frequency_stats.iter().skip(1).all(|stats| stats.allocation_operations == 0),
        "mixed-precision scratch should be retained across frequencies: {:?}",
        diagnostics.frequency_stats.iter().map(|stats| stats.allocation_operations).collect::<Vec<_>>()
    );

    let report = compare_backend_outputs(
        "cpu-f64",
        &cpu,
        "cuda-f64-resident-reuse",
        &cuda,
        0.0,
        1.0e-12,
    )
    .unwrap();
    assert_eq!(report.sample_count, 16);
    report.require_within(0.25, 0.25).unwrap();
    if let Some(path) = std::env::var_os("SONIUM_SIM_ROOM_SR1_RESIDENT_ARTIFACT") {
        write_sr1_evidence(
            std::path::Path::new(&path),
            "f64-resident-reuse-4-frequency",
            &diagnostics,
            &report,
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_cancellation_rejects_before_device_setup() {
    let cancellation = FemRunCancellation::new();
    cancellation.cancel("cancel before CUDA setup");
    let error = run_fem_gpu_report_cancellation(
        &accelerator_fixture(),
        &accelerator_mesh_config(),
        GpuPrecision::F64Outer,
        GpuPrecond::Auto,
        0,
        &cancellation,
    )
    .unwrap_err();
    let typed = error
        .downcast_ref::<FemBackendExecutionError>()
        .expect("CUDA cancellation must retain typed backend provenance");
    assert!(matches!(
        typed,
        FemBackendExecutionError::Cancelled { backend, reason }
            if backend == "cuda" && reason == "cancel before CUDA setup"
    ));
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_mid_frequency_cancellation_is_typed_and_does_not_publish_partial_result() {
    assert_eq!(
        std::env::var("SONIUM_SIM_ROOM_SR1_CANCEL_FREQUENCY").as_deref(),
        Ok("0"),
        "the stable SR1 runner must select the deterministic mid-frequency seam"
    );
    let success_artifact = std::env::var_os("SONIUM_SIM_ROOM_SR1_CANCEL_SUCCESS_ARTIFACT")
        .map(std::path::PathBuf::from);
    if let Some(path) = &success_artifact {
        let _ = std::fs::remove_file(path);
    }

    let cancellation = FemRunCancellation::new();
    let error = run_fem_gpu_report_cancellation(
        &accelerator_fixture(),
        &accelerator_mesh_config(),
        GpuPrecision::F64Outer,
        GpuPrecond::Auto,
        0,
        &cancellation,
    )
    .unwrap_err();
    let typed = error
        .downcast_ref::<FemBackendExecutionError>()
        .expect("mid-frequency CUDA cancellation must retain typed backend provenance");
    let (backend, reason) = match typed {
        FemBackendExecutionError::Cancelled { backend, reason } => (backend, reason),
        other => panic!("mid-frequency cancellation returned an untyped error: {other:?}"),
    };
    let failure = error
        .downcast_ref::<FemBackendFailureDiagnostics>()
        .expect("mid-frequency CUDA cancellation must publish structured diagnostics");
    assert_eq!(failure.backend, "cuda");
    assert_eq!(failure.stage.label(), "cancellation");
    assert!(!failure.partial_result_published);
    assert!(failure.resources_released_on_unwind);
    assert_eq!(backend, "cuda");
    assert!(reason.contains("mid-frequency cancellation injected at frequency 0"));

    // `run_fem_gpu_report_cancellation` returns before constructing a
    // SimulationOutput. Reacquiring the device context after error unwinding
    // verifies backend reuse; allocation lifetime is owned by Rust Drop, while
    // sonium-compute does not expose a low-level leak counter.
    let context_reacquired = sonium_compute::CudaBackend::new(0).is_ok();
    assert!(
        context_reacquired,
        "CUDA context was not reusable after cancellation"
    );
    let success_artifact_published = success_artifact.as_ref().is_some_and(|path| path.exists());
    assert!(
        !success_artifact_published,
        "cancellation must not publish a partial success artifact"
    );

    if let Some(path) = std::env::var_os("SONIUM_SIM_ROOM_SR1_CANCEL_ARTIFACT") {
        write_sr1_cancellation_evidence(
            std::path::Path::new(&path),
            backend,
            reason,
            0,
            failure,
            context_reacquired,
            success_artifact_published,
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_resident_mid_frequency_cancellation_retries_after_warm_state() {
    assert_eq!(
        std::env::var("SONIUM_SIM_ROOM_SR1_RESIDENT_CANCEL_FREQUENCY").as_deref(),
        Ok("2"),
        "the SR1 breadth runner must cancel after two committed warm frequencies"
    );
    let success_artifact = std::env::var_os("SONIUM_SIM_ROOM_SR1_RESIDENT_CANCEL_SUCCESS_ARTIFACT")
        .map(std::path::PathBuf::from);
    if let Some(path) = &success_artifact {
        let _ = std::fs::remove_file(path);
    }

    let cancellation = FemRunCancellation::new();
    let error = run_fem_gpu_report_cancellation(
        &accelerator_resident_fixture(),
        &accelerator_resident_mesh_config(),
        GpuPrecision::F64Outer,
        GpuPrecond::Auto,
        0,
        &cancellation,
    )
    .expect_err("the injected warm-state cancellation must stop publication");
    let typed = error
        .downcast_ref::<FemBackendExecutionError>()
        .expect("warm-state cancellation must retain typed backend provenance");
    let (backend, reason) = match typed {
        FemBackendExecutionError::Cancelled { backend, reason } => (backend, reason),
        other => panic!("warm-state cancellation returned an untyped error: {other:?}"),
    };
    let failure = error
        .downcast_ref::<FemBackendFailureDiagnostics>()
        .expect("warm-state cancellation must publish structured diagnostics");
    assert_eq!(failure.backend, "cuda");
    assert_eq!(failure.stage.label(), "cancellation");
    assert!(!failure.partial_result_published);
    assert!(failure.resources_released_on_unwind);
    assert_eq!(backend, "cuda");
    assert!(reason.contains("mid-frequency cancellation injected at frequency 2"));
    let context_reacquired = sonium_compute::CudaBackend::new(0).is_ok();
    assert!(
        context_reacquired,
        "CUDA context must be reusable after warm cancellation"
    );
    let success_artifact_published = success_artifact.as_ref().is_some_and(|path| path.exists());
    assert!(!success_artifact_published);

    // A cancellation after committed frequencies must still unwind all
    // resident state.  Recreate the runner and complete the same production
    // four-frequency sweep to prove retry, rather than merely context setup.
    let (_output, retry) = run_fem_gpu_report(
        &accelerator_resident_fixture(),
        &accelerator_resident_mesh_config(),
        GpuPrecision::F64Outer,
        GpuPrecond::Auto,
        0,
    )
    .expect("fresh native CUDA context must retry the resident sweep");
    retry
        .validate()
        .expect("resident retry diagnostics must remain valid");
    let retry_execution = retry
        .execution
        .as_ref()
        .expect("retry execution provenance");
    assert_eq!(retry_execution.requested_backend, "cuda");
    assert_eq!(retry_execution.actual_backend, "cuda");
    assert_eq!(retry_execution.fallback_status, "none");
    assert_eq!(retry.frequency_stats.len(), 4);
    assert!(retry.frequency_stats.iter().all(|stats| stats.converged));
    if let Some(path) = std::env::var_os("SONIUM_SIM_ROOM_SR1_RESIDENT_CANCEL_ARTIFACT") {
        write_sr1_resident_recovery_evidence(
            std::path::Path::new(&path),
            backend,
            reason,
            failure,
            context_reacquired,
            success_artifact_published,
            &retry,
        );
    }
}

#[cfg(feature = "cuda")]
fn assert_cuda_failure_injection_lane(
    point: sonium_compute::CudaFailureInjectionPoint,
    expected_recovery: &str,
) {
    let artifact_path = std::env::var_os("SONIUM_SIM_ROOM_SR1_FAILURE_ARTIFACT")
        .map(std::path::PathBuf::from)
        .expect("the stable SR1 runner must provide a failure artifact path");
    let retry_artifact_path = std::env::var_os("SONIUM_SIM_ROOM_SR1_FAILURE_RETRY_ARTIFACT")
        .map(std::path::PathBuf::from)
        .expect("the stable SR1 runner must provide a retry artifact path");
    let _ = std::fs::remove_file(&artifact_path);
    let _ = std::fs::remove_file(&retry_artifact_path);

    // Prove this is a live native-CUDA lane before arming the one-shot shim.
    let backend = sonium_compute::CudaBackend::new(0).unwrap_or_else(|error| {
        panic!("native CUDA is required for SR1 failure injection: {error}")
    });
    drop(backend);

    let error = run_fem_gpu_report_with_failure_injection(
        &accelerator_fixture(),
        &accelerator_mesh_config(),
        GpuPrecision::F64Outer,
        GpuPrecond::Auto,
        0,
        point,
    )
    .unwrap_err();
    let failure = error
        .downcast_ref::<FemBackendFailureDiagnostics>()
        .expect("native CUDA injection must publish structured diagnostics");
    assert_eq!(failure.backend, "cuda");
    let expected_stage = match point {
        sonium_compute::CudaFailureInjectionPoint::Launch => "launch",
        sonium_compute::CudaFailureInjectionPoint::DeviceLoss => "setup",
        other => panic!("SR1 failure evidence only accepts launch/device-loss, got {other:?}"),
    };
    assert_eq!(failure.stage.label(), expected_stage);
    assert_eq!(failure.context_recovery.label(), expected_recovery);
    assert!(!failure.partial_result_published);
    assert!(failure.resources_released_on_unwind);
    assert!(failure.detail.contains("injected CUDA"));
    assert!(
        failure.operation == "cuda_launch" || failure.operation == "cuda_device_loss",
        "unexpected injected operation: {}",
        failure.operation
    );
    assert!(
        !artifact_path.exists(),
        "a failed run must not publish its success artifact"
    );
    assert!(
        !retry_artifact_path.exists(),
        "a failed run must not publish its retry artifact before recovery"
    );

    // Both launch and device-loss dispositions are safely retried by
    // recreating the runner-owned backend. This is the only allowed recovery
    // path for device loss; launch failures may also synchronize/reuse the
    // old context, but the room runner has already unwound it here.
    let (_output, retry) = run_fem_gpu_report(
        &accelerator_fixture(),
        &accelerator_mesh_config(),
        GpuPrecision::F64Outer,
        GpuPrecond::Auto,
        0,
    )
    .expect("fresh native CUDA context must retry successfully");
    retry
        .validate()
        .expect("retry diagnostics must retain valid CUDA provenance");
    assert_eq!(retry.execution.as_ref().unwrap().actual_backend, "cuda");
    assert_eq!(retry.execution.as_ref().unwrap().fallback_status, "none");
    assert!(retry.frequency_stats.iter().all(|stats| stats.converged));
    std::fs::write(
        &retry_artifact_path,
        b"{\"status\":\"published-after-retry\"}\n",
    )
    .expect("retry success marker writes after successful return");
    assert!(retry_artifact_path.exists());
    write_sr1_failure_evidence(&artifact_path, point, failure, &retry, true, true);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_injected_launch_failure_is_typed_unwound_and_retryable() {
    assert_eq!(
        std::env::var("SONIUM_SIM_ROOM_SR1_FAILURE_POINT").as_deref(),
        Ok("launch"),
        "the stable SR1 runner must select launch injection"
    );
    assert_cuda_failure_injection_lane(
        sonium_compute::CudaFailureInjectionPoint::Launch,
        "synchronize-before-reuse",
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_injected_device_loss_requires_recreated_context_and_retries() {
    assert_eq!(
        std::env::var("SONIUM_SIM_ROOM_SR1_FAILURE_POINT").as_deref(),
        Ok("device-loss"),
        "the stable SR1 runner must select device-loss injection"
    );
    assert_cuda_failure_injection_lane(
        sonium_compute::CudaFailureInjectionPoint::DeviceLoss,
        "recreate",
    );
}

#[cfg(feature = "metal")]
#[test]
fn metal_cancellation_rejects_before_device_setup() {
    let cancellation = FemRunCancellation::new();
    cancellation.cancel("cancel before Metal setup");
    let error = run_fem_metal_report_cancellation(
        &accelerator_fixture(),
        &accelerator_mesh_config(),
        GpuPrecision::F64Outer,
        GpuPrecond::Chebyshev { degree: 4 },
        &cancellation,
    )
    .unwrap_err();
    assert!(error.to_string().contains("cancel before Metal setup"));
}

#[cfg(feature = "metal")]
#[test]
fn metal_fem_matches_cpu_backend_with_checked_parity_budget() {
    let _backend = match sonium_compute::MetalBackend::new() {
        Ok(backend) => backend,
        Err(error) => {
            if std::env::var_os("SONIUM_SIM_ROOM_REQUIRE_METAL").is_some() {
                panic!(
                    "Metal is required for this evidence lane but device setup failed: {error:?}"
                );
            }
            eprintln!("no Metal device — skipping: {error:?}");
            return;
        }
    };

    let simulation = accelerator_fixture();
    let mesh_config = accelerator_mesh_config();

    let (cpu, cpu_diagnostics) = run_fem_with_mesh_report(&simulation, &mesh_config).unwrap();
    let (metal, diagnostics) = match run_fem_metal_report(
        &simulation,
        &mesh_config,
        GpuPrecision::F64Outer,
        GpuPrecond::Chebyshev { degree: 4 },
    ) {
        Ok(result) => result,
        Err(error) => {
            assert!(
                error.downcast_ref::<FemBackendExecutionError>().is_some(),
                "Metal setup/unsupported failures must remain typed: {error:?}"
            );
            if std::env::var_os("SONIUM_SIM_ROOM_REQUIRE_METAL").is_some() {
                panic!("Metal is required for this evidence lane but the FEM run failed: {error}");
            }
            eprintln!("native Metal FEM unavailable — skipping: {error}");
            return;
        }
    };
    assert!(diagnostics.validate().is_ok());
    assert_physical_parity(&cpu_diagnostics, &diagnostics);
    assert_reciprocal_output(&cpu);
    assert_reciprocal_output(&metal);
    let execution = diagnostics
        .execution
        .as_ref()
        .expect("Metal run must publish execution provenance");
    assert_eq!(execution.requested_backend, "metal");
    assert_eq!(execution.actual_backend, "metal");
    assert_eq!(execution.fallback_status, "none");
    assert_eq!(execution.precision, "f64-outer (IR)");
    assert!(execution.estimated_peak_memory_bytes.is_some());
    assert!(execution.setup_transfer_stats.h2d_bytes > 0);
    assert!(execution.total_transfer_stats.d2h_bytes > 0);
    assert_eq!(diagnostics.frequency_stats.len(), 2);
    assert!(
        diagnostics
            .frequency_stats
            .iter()
            .all(|stats| stats.transfer_stats.is_some())
    );

    let report =
        compare_backend_outputs("cpu-f64", &cpu, "metal-f64-outer", &metal, 0.0, 1.0e-12).unwrap();
    assert_eq!(report.sample_count, 8);
    report.require_within(0.25, 0.25).unwrap();
}
