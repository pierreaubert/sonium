use super::fem_mesh_config::FemMeshConfig;
use super::fem_preconditioner_kind::FemPreconditionerKind;
use super::fem_run_diagnostics::{FemPhysicalEvidenceStatus, FemRunDiagnostics};
use super::shifted_laplacian_shift::ShiftedLaplacianShift;
use super::types::{FemFrequencyStats, FemTransferStats};
use crate::energy_validation::{
    FemBoundaryFluxEvidence, FemBoundaryFluxSample, FemOperatorWorkEvidence,
    FemPhysicalPowerEvidence,
};

fn valid_fem_run_diagnostics() -> FemRunDiagnostics {
    let cfg = FemMeshConfig {
        preconditioner: FemPreconditionerKind::Ilut,
        modal_damping_ratio: 0.01,
        shifted_laplacian_shift: ShiftedLaplacianShift::K2 {
            alpha: 0.4,
            beta: 0.6,
        },
        ..FemMeshConfig::default()
    };
    let mut diagnostics = FemRunDiagnostics::new("cpu", &cfg);
    diagnostics.record_frequency(FemFrequencyStats {
        block_solve: Default::default(),
        frequency_hz: 50.0,
        symmetry_mode: None,
        matrix_ms: 1.0,
        preconditioner_ms: 2.0,
        rhs_ms: 3.0,
        solve_ms: 4.0,
        iterations: 10,
        inner_iterations: 2,
        retries: 1,
        preconditioner_age: 0,
        preconditioner_rebuild_reason: "test".to_string(),
        fallback_stage: 0,
        max_residual: 1.0e-7,
        converged: true,
        operator_work: None,
        physical_power: None,
        boundary_flux: None,
        transfer_stats: None,
        allocation_operations: 0,
        allocation_high_water_bytes: 0,
        synchronization_calls: 0,
    });
    diagnostics
}

#[test]
fn accepts_cuda_backend_diagnostics_with_per_frequency_evidence() {
    let mut diagnostics = valid_fem_run_diagnostics();
    diagnostics.backend = "cuda".into();
    let execution = diagnostics.execution.as_mut().unwrap();
    execution.requested_backend = "cuda".into();
    execution.actual_backend = "cuda".into();
    execution.precision = "f64-outer (IR)".into();
    execution.estimated_peak_memory_bytes = Some(1024);
    execution.setup_transfer_stats = FemTransferStats {
        h2d_operations: 1,
        h2d_bytes: 64,
        ..FemTransferStats::default()
    };
    execution.total_transfer_stats = FemTransferStats {
        d2h_operations: 1,
        d2h_bytes: 64,
        ..FemTransferStats::default()
    };
    diagnostics.frequency_stats[0].transfer_stats = Some(FemTransferStats {
        d2h_operations: 1,
        d2h_bytes: 64,
        ..FemTransferStats::default()
    });
    diagnostics.frequency_stats[0].allocation_high_water_bytes = 1024;
    diagnostics.frequency_stats[0].synchronization_calls = 1;
    diagnostics.frequency_stats[0].preconditioner_rebuild_reason = "scheduled".into();
    diagnostics.frequency_stats[0].preconditioner_age = 0;
    diagnostics.frequency_stats[0].fallback_stage = 1;
    diagnostics.frequency_stats[0].retries = 1;
    diagnostics.retry_count = 1;
    diagnostics.total_iterations = diagnostics.frequency_stats[0].iterations;
    diagnostics.total_inner_iterations = diagnostics.frequency_stats[0].inner_iterations;
    diagnostics.nonconverged_count = 0;
    diagnostics.max_residual = diagnostics.frequency_stats[0].max_residual;
    assert!(diagnostics.validate().is_ok());
    assert!(diagnostics.require_converged().is_ok());
}

#[test]
fn rejects_accelerator_diagnostics_without_residency_evidence() {
    let mut diagnostics = valid_fem_run_diagnostics();
    {
        let execution = diagnostics.execution.as_mut().unwrap();
        execution.requested_backend = "cuda".into();
        execution.actual_backend = "cuda".into();
    }

    let error = diagnostics.validate().unwrap_err();
    assert!(error.contains("peak-memory"));

    diagnostics
        .execution
        .as_mut()
        .unwrap()
        .estimated_peak_memory_bytes = Some(1024);
    let error = diagnostics.validate().unwrap_err();
    assert!(error.contains("transfer provenance"));
}

#[test]
fn rejects_backend_mismatch_without_explicit_fallback() {
    let mut diagnostics = valid_fem_run_diagnostics();
    diagnostics.execution.as_mut().unwrap().actual_backend = "cuda".into();

    let error = diagnostics.validate().unwrap_err();
    assert!(error.contains("fallback provenance"));
}

#[test]
fn transfer_deltas_saturate_when_backend_counters_reset() {
    let earlier = FemTransferStats {
        h2d_operations: 10,
        h2d_bytes: 100,
        d2h_operations: 10,
        d2h_bytes: 100,
        d2d_operations: 10,
        d2d_bytes: 100,
    };
    let later = FemTransferStats {
        h2d_operations: 2,
        h2d_bytes: 20,
        d2h_operations: 12,
        d2h_bytes: 120,
        d2d_operations: 12,
        d2d_bytes: 120,
    };

    assert_eq!(
        later.saturating_sub(earlier),
        FemTransferStats {
            d2h_operations: 2,
            d2h_bytes: 20,
            d2d_operations: 2,
            d2d_bytes: 20,
            ..FemTransferStats::default()
        }
    );
}

#[test]
fn accepts_consistent_converged_stats() {
    let diagnostics = valid_fem_run_diagnostics();

    assert!(diagnostics.validate().is_ok());
    assert!(diagnostics.require_converged().is_ok());
}

#[test]
fn rejects_nonconverged_stats() {
    let mut diagnostics = valid_fem_run_diagnostics();
    diagnostics.frequency_stats[0].converged = false;
    diagnostics.nonconverged_count = 1;

    assert!(diagnostics.validate().is_ok());
    let error = diagnostics.require_converged().unwrap_err();
    assert!(error.contains("nonconverged"));
}

#[test]
fn rejects_invalid_metrics_and_empty_stats() {
    let mut invalid = valid_fem_run_diagnostics();
    invalid.frequency_stats[0].solve_ms = f64::NAN;
    assert!(invalid.validate().is_err());

    let mut negative = valid_fem_run_diagnostics();
    negative.frequency_stats[0].rhs_ms = -1.0;
    assert!(negative.validate().is_err());

    let empty = FemRunDiagnostics::new("cpu", &FemMeshConfig::default());
    assert!(empty.validate().is_err());
}

#[test]
fn rejects_inconsistent_aggregate_counters() {
    let mut diagnostics = valid_fem_run_diagnostics();
    diagnostics.total_iterations += 1;

    let error = diagnostics.validate().unwrap_err();
    assert!(error.contains("aggregate counters"));
}

#[test]
fn rejects_physical_power_without_operator_work() {
    let mut diagnostics = valid_fem_run_diagnostics();
    diagnostics.frequency_stats[0].physical_power = Some(FemPhysicalPowerEvidence {
        source_power_w: 1.0,
        volume_damping_power_w: 0.25,
        boundary_power_w: 0.75,
        balance_residual_w: 0.0,
    });

    let error = diagnostics.validate().unwrap_err();
    assert!(error.contains("requires operator-work"));
}

#[test]
fn rejects_boundary_flux_without_matching_physical_power() {
    let mut diagnostics = valid_fem_run_diagnostics();
    diagnostics.frequency_stats[0].boundary_flux = Some(FemBoundaryFluxEvidence {
        samples: vec![FemBoundaryFluxSample {
            boundary_tag: 7,
            active_power_w: 1.0,
        }],
        total_power_w: 1.0,
    });

    let error = diagnostics.validate().unwrap_err();
    assert!(error.contains("requires physical-power"));
}

#[test]
fn rejects_boundary_flux_that_disagrees_with_physical_power() {
    let mut diagnostics = valid_fem_run_diagnostics();
    diagnostics.frequency_stats[0].operator_work = Some(FemOperatorWorkEvidence {
        source_work: 1.0,
        volume_damping_work: 0.0,
        boundary_work: 1.0,
        balance_residual: 0.0,
    });
    diagnostics.frequency_stats[0].physical_power = Some(FemPhysicalPowerEvidence {
        source_power_w: 1.0,
        volume_damping_power_w: 0.0,
        boundary_power_w: 0.25,
        balance_residual_w: 0.75,
    });
    diagnostics.frequency_stats[0].boundary_flux = Some(FemBoundaryFluxEvidence {
        samples: vec![FemBoundaryFluxSample {
            boundary_tag: 7,
            active_power_w: 1.0,
        }],
        total_power_w: 1.0,
    });

    let error = diagnostics.validate().unwrap_err();
    assert!(error.contains("disagrees"));
}

#[test]
fn labels_reduced_domain_and_device_energy_evidence_gaps() {
    let mut reduced = valid_fem_run_diagnostics();
    reduced.mark_reduced_domain_physical_evidence_unavailable();
    for stats in &mut reduced.frequency_stats {
        stats.operator_work = None;
        stats.physical_power = None;
        stats.boundary_flux = None;
    }
    assert_eq!(
        reduced.physical_evidence_status,
        FemPhysicalEvidenceStatus::UnavailableReducedDomain
    );
    assert!(reduced.validate().is_ok());

    let mut device = valid_fem_run_diagnostics();
    device.mark_device_physical_evidence_unavailable();
    for stats in &mut device.frequency_stats {
        stats.operator_work = None;
        stats.physical_power = None;
        stats.boundary_flux = None;
    }
    assert_eq!(
        device.physical_evidence_status,
        FemPhysicalEvidenceStatus::UnavailableDeviceBackend
    );
    assert!(device.validate().is_ok());
}

#[test]
fn rejects_incomplete_physical_evidence_status() {
    let mut diagnostics = valid_fem_run_diagnostics();
    diagnostics.frequency_stats[0].operator_work = Some(FemOperatorWorkEvidence {
        source_work: 1.0,
        volume_damping_work: 0.0,
        boundary_work: 1.0,
        balance_residual: 0.0,
    });
    diagnostics.frequency_stats[0].physical_power = None;
    diagnostics.frequency_stats[0].boundary_flux = None;
    assert!(
        diagnostics
            .validate()
            .unwrap_err()
            .contains("operator-work, physical-power, and boundary-flux")
    );
}
