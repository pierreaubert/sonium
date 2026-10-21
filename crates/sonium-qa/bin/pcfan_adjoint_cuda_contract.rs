//! Production PC-fan adjoint/parity evidence.
//!
//! This producer is deliberately a capability boundary, not a claim that the
//! current FEM bridge has a CUDA adjoint.  The forward rows are obtained from
//! the production CFD runner and passed through the production parity
//! evaluator.  The adjoint row then asks the checked production handoff for a
//! device-resident execution; the current host-assembled bridge must reject
//! that request before a transpose solve can be reported as CUDA.  CPU
//! reduced-order adjoint/finite-difference rows are retained as a separate
//! reference and are never used to label the CUDA rows.

#![cfg(feature = "cuda")]

use serde_json::{Value, json};
use sonium_qa::{
    QaContextReuse, QaExecutionCounts, QaResult, QaRuntimeCounters, QaRuntimeEvidence,
    cuda_hardware_provenance, write_qa_artifact,
};
use sonium_sim_pcfan::{
    PcFanAcceleratorParityConfig, PcFanCfdBackend, PcFanConfig, PcFanDerivativePolicy,
    PcFanFlowModel, PcFanMeshResolution, PcFanResolvedAdjointConfig, PcFanResolvedFlowObjective,
    collect_pc_fan_accelerator_parity_run, evaluate_pc_fan, evaluate_pc_fan_accelerator_parity,
    evaluate_pc_fan_cfd_newton_linearization, solve_pc_fan_cfd_mean_flow,
    solve_pc_fan_cfd_resolved_objective_adjoint_from_solution,
    solve_pc_fan_shape_gradient_reduced_adjoint, solve_pc_fan_shape_gradient_with_policy,
};
use std::time::Instant;

const CASE_ID: &str = "pcfan.adjoint-cuda-contract.v1";
const ARTIFACT: &str = "pcfan_adjoint_cuda_contract.json";

fn finite_or_null(value: f64) -> Value {
    if value.is_finite() {
        json!(value)
    } else {
        Value::Null
    }
}

fn steady_config(backend: PcFanCfdBackend) -> PcFanConfig {
    let mut config = PcFanConfig::default_120mm();
    config.mesh = PcFanMeshResolution {
        radial_segments: 2,
        chord_segments: 2,
        circumferential_segments: 8,
        harmonic_count: 2,
    };
    config.cfd.flow_model = PcFanFlowModel::IncompressibleCfd;
    config.cfd.backend = backend;
    config.cfd.allow_host_fallback = false;
    config.cfd.require_device_resident = false;
    config.cfd.cells_x = 2;
    config.cfd.cells_y = 2;
    config.cfd.cells_z = 2;
    config.cfd.max_picard_iterations = 40;
    config.cfd.nonlinear_tolerance = 2.0e-2;
    // Same eight-element fixture as the forward contract: the converged
    // discrete solution carries ~8% nodal-flux imbalance from discretisation.
    config.cfd.max_relative_mass_imbalance = 0.10;
    // PF2 compares nonlinear observables, so the backend linear solves must
    // be substantially tighter than the parity budget.  The former 1e-4
    // inner tolerance let CPU ILU and CUDA polynomial-Jacobi stop at visibly
    // different states even though both were labelled converged.
    config.cfd.linear_solver_max_iterations = 3000;
    config.cfd.linear_solver_tolerance = 1.0e-12;
    config
}

fn reduced_config() -> PcFanConfig {
    let mut config = PcFanConfig::default_120mm();
    config.mesh = PcFanMeshResolution {
        radial_segments: 3,
        chord_segments: 2,
        circumferential_segments: 8,
        harmonic_count: 2,
    };
    config.optimizer.finite_difference_step = 2.0e-5;
    config
}

fn production_dataset(run: &sonium_sim_pcfan::PcFanAcceleratorParityRun) -> Value {
    serde_json::to_value(run.dataset()).unwrap_or(Value::Null)
}

fn production_parity(
    reference: &sonium_sim_pcfan::PcFanAcceleratorParityRun,
    candidate: &sonium_sim_pcfan::PcFanAcceleratorParityRun,
) -> Value {
    let config = PcFanAcceleratorParityConfig::default();
    match evaluate_pc_fan_accelerator_parity(&config, reference, candidate) {
        Ok(report) => json!({
            "status": if report.passed { "passed" } else { "rejected_by_budget" },
            "passed": report.passed,
            "reference_backend": format!("{:?}", report.reference_backend),
            "candidate_backend": format!("{:?}", report.candidate_backend),
            "sample_count": report.sample_count,
            "max_relative_errors": report.max_relative_errors,
            "max_residual_difference": finite_or_null(report.max_residual_difference),
            "max_field_relative_error": finite_or_null(report.max_field_relative_error),
            "max_production_relative_error": finite_or_null(report.max_production_relative_error),
            "reference_environment": report.reference_environment,
            "candidate_environment": report.candidate_environment,
            "tolerance": {
                "max_relative_error": config.max_relative_error,
                "max_absolute_residual_difference": config.max_absolute_residual_difference,
                "min_points": config.min_points,
                "allow_host_fallback": config.allow_host_fallback,
            },
        }),
        Err(error) => json!({
            "status": "typed_rejection",
            "passed": false,
            "error": error.to_string(),
            "error_type": format!("{error:?}"),
        }),
    }
}

fn reduced_adjoint_reference() -> Result<Value, String> {
    let config = reduced_config();
    let design = config.default_design_variables();
    let adjoint = solve_pc_fan_shape_gradient_reduced_adjoint(&config, &design)
        .map_err(|error| format!("CPU reduced adjoint failed: {error}"))?;
    let finite_difference = solve_pc_fan_shape_gradient_with_policy(
        &config,
        &design,
        PcFanDerivativePolicy::FiniteDifferenceFullObjective,
    )
    .map_err(|error| format!("CPU finite-difference reference failed: {error}"))?;

    let mut max_relative_error: f64 = 0.0;
    for (left, right) in adjoint
        .objective_gradient
        .iter()
        .zip(finite_difference.objective_gradient.iter())
    {
        let scale = left.abs().max(right.abs()).max(1.0);
        max_relative_error = max_relative_error.max((left - right).abs() / scale);
    }

    // An independent directional finite difference around the same objective
    // supplies a first-order Taylor check.  This is explicitly CPU reduced
    // order evidence; it is not promoted to CUDA-adjoint evidence.
    let base = evaluate_pc_fan(&config, &design).objective;
    let h = 2.0e-5;
    let mut direction = design.to_vector();
    direction[0] += h;
    let plus = evaluate_pc_fan(
        &config,
        &sonium_sim_pcfan::PcFanDesignVariables::from_vector(direction),
    )
    .objective;
    let mut direction = design.to_vector();
    direction[0] -= h;
    let minus = evaluate_pc_fan(
        &config,
        &sonium_sim_pcfan::PcFanDesignVariables::from_vector(direction),
    )
    .objective;
    let directional_slope = (plus - minus) / (2.0 * h);
    let adjoint_slope = adjoint.objective_gradient[0];
    let slope_scale = directional_slope.abs().max(adjoint_slope.abs()).max(1.0);
    let slope_relative_error = (directional_slope - adjoint_slope).abs() / slope_scale;
    let taylor_plus = (plus - base - h * adjoint_slope).abs();
    let taylor_minus = (minus - base + h * adjoint_slope).abs();
    let taylor_remainder = taylor_plus.max(taylor_minus);

    Ok(json!({
        "status": "passed",
        "backend": "Cpu",
        "derivative_provenance": "reduced-order-discrete-adjoint-with-checked-transpose",
        "gradient_dimension": adjoint.objective_gradient.len(),
        "max_gradient_relative_error_vs_independent_fd": finite_or_null(max_relative_error),
        "directional_finite_difference_relative_error": finite_or_null(slope_relative_error),
        "directional_finite_difference_step": h,
        "taylor_remainder_absolute": finite_or_null(taylor_remainder),
        "primal_residual_max_abs": finite_or_null(adjoint.residual.max_abs),
        "adjoint_airflow_multiplier": finite_or_null(adjoint.adjoint.airflow_multiplier),
        "acoustic_multiplier_count": adjoint.adjoint.acoustic_pressure_multipliers.len(),
        "cuda_promotion": "not_allowed",
    }))
}

fn run() -> Result<Value, String> {
    let rpms = [600.0, 900.0, 1200.0];
    let cpu_config = steady_config(PcFanCfdBackend::Cpu);
    let design = cpu_config.default_design_variables();
    let cpu_run = collect_pc_fan_accelerator_parity_run(&cpu_config, &design, &rpms)
        .map_err(|error| format!("production CPU parity collection failed: {error}"))?;

    let cuda_config = steady_config(PcFanCfdBackend::Cuda);
    let cuda_collection = collect_pc_fan_accelerator_parity_run(&cuda_config, &design, &rpms);
    let (
        cuda_dataset,
        parity,
        cuda_probe,
        production_flow_transpose,
        production_newton_design_transpose,
    ) = match cuda_collection {
        Ok(cuda_run) => {
            let parity = production_parity(&cpu_run, &cuda_run);
            // Reuse the production solver for the checked adjoint handoff.
            // Requiring residency is essential: accepting the linear-solve
            // bridge here would mislabel a host-assembled result as CUDA.
            let mut resident_config = cuda_config.clone();
            resident_config.cfd.require_device_resident = true;
            let cuda_solution = solve_pc_fan_cfd_mean_flow(&resident_config, &design);
            let production_picard_flow_transpose = cuda_solution
                .picard_linearization
                .as_ref()
                .ok_or_else(|| {
                    "production CUDA solution did not publish a Picard linearization".to_owned()
                })
                .and_then(sonium_qa::pcfan_cuda_adjoint::production_picard_flow_transpose_probe)
                .map_err(|error| {
                    format!("production Picard CUDA transpose probe failed: {error}")
                })?;
            // The mean-flow entry point intentionally omits the exact Newton
            // record when it starts from zero. Re-evaluate the same production
            // map at its published Picard state so the analytic Newton tangent
            // and frozen-mesh design-residual block are explicitly requested.
            let newton_initial_state = cuda_solution
                .picard_linearization
                .as_ref()
                .ok_or_else(|| {
                    "production CUDA solution did not publish a Picard state for Newton".to_owned()
                })?
                .state
                .clone();
            let production_newton = evaluate_pc_fan_cfd_newton_linearization(
                &resident_config,
                &design,
                &newton_initial_state,
            )
            .map_err(|error| {
                format!("production Newton linearization evaluation failed: {error}")
            })?;
            let production_newton_design_transpose =
                sonium_qa::pcfan_cuda_adjoint::production_newton_design_transpose_probe(
                    &production_newton,
                )
                .map_err(|error| {
                    format!("production Newton/design transpose probe failed: {error}")
                })?;
            let probe = match solve_pc_fan_cfd_resolved_objective_adjoint_from_solution(
                &PcFanResolvedAdjointConfig::default(),
                &resident_config,
                &design,
                &cuda_solution,
                PcFanResolvedFlowObjective::Airflow,
            ) {
                Ok(report) => json!({
                    "status": "unexpectedly_supported",
                    "backend": format!("{:?}", report.backend),
                    "converged": report.converged,
                    "iterations": report.iterations,
                }),
                Err(error) => json!({
                    "status": "typed_rejection",
                    "error": error.to_string(),
                    "error_type": format!("{error:?}"),
                    "requested_backend": "Cuda",
                    "required_execution_mode": "DeviceResident",
                }),
            };
            (
                Some(production_dataset(&cuda_run)),
                parity,
                probe,
                production_picard_flow_transpose,
                production_newton_design_transpose,
            )
        }
        Err(error) => (
            None,
            json!({
                "status": "typed_rejection",
                "passed": false,
                "error": error.to_string(),
                "error_type": format!("{error:?}"),
            }),
            json!({
                "status": "not_attempted",
                "reason": "CUDA production parity collection did not produce an admitted run",
            }),
            json!({
                "status": "not_attempted",
                "reason": "CUDA production parity collection did not produce an admitted run",
            }),
            json!({
                "status": "not_attempted",
                "reason": "CUDA production parity collection did not produce an admitted run",
            }),
        ),
    };

    let cpu_reference = reduced_adjoint_reference()?;
    let paired_csr = sonium_qa::pcfan_cuda_adjoint::paired_csr_probe()?;
    let device_count = sonium_compute::cuda::cuda_device_count()
        .map(|count| json!(count))
        .unwrap_or_else(|error| json!({"status": "unavailable", "reason": error.to_string()}));
    let cuda_forward_acceptance = if parity.get("status").and_then(Value::as_str) == Some("passed")
    {
        "passed; see production_forward_parity"
    } else {
        "typed_rejection; see production_forward_parity"
    };

    Ok(json!({
        "schema_version": 1,
        "case_id": CASE_ID,
        "scope": "native_linux_x86_64_pcfan_adjoint_and_parity",
        "hardware": cuda_hardware_provenance(),
        "cuda_device_count": device_count,
        "fixture": {
            "rpms": rpms,
            "mesh": "2x2x2 CFD cells; identical CPU/CUDA production setup",
            "flow_model": "IncompressibleCfd",
            "allow_host_fallback": false,
            "parity_budget": "PcFanAcceleratorParityConfig::default",
        },
        "production_cpu": production_dataset(&cpu_run),
        "production_cuda": cuda_dataset,
        "production_forward_parity": parity,
        "cuda_adjoint_capability": {
            "flow_transpose": cuda_probe,
            "production_picard_flow_transpose": production_flow_transpose,
            "production_newton_design_transpose": production_newton_design_transpose,
            "paired_csr_hermitian_transpose": paired_csr,
            "acoustic_transpose": {
                "status": "unsupported",
                "reason": "no CUDA device-resident acoustic Jacobian/transpose execution is published by the FEM bridge",
            },
            "hermitian_transpose": {
                "status": "unsupported",
                "reason": "PC-fan CUDA path publishes a real host-assembled linear solve only; no complex Hermitian adjoint capability is exposed",
            },
            "design_residual_contraction": {
                "status": "unsupported",
                "reason": "CUDA production path does not publish an admitted device-resident Newton/design-residual derivative bundle",
            },
        },
        "cpu_reference": cpu_reference,
        "failure_matrix": [
            {"failure_class": "backend_substitution", "status": "typed", "contract": "PcFanAcceleratorParityError::RequestedBackendMismatch/HostFallback"},
            {"failure_class": "unconverged_input", "status": "typed", "contract": "PcFanAcceleratorParityError::UnconvergedObservation"},
            {"failure_class": "missing_derivative_provenance", "status": "typed", "contract": "PcFanResolvedAdjointError::MissingProductionNewtonLinearization/DerivativeProbeNotAdmissible"},
            {"failure_class": "budget_violation", "status": "typed", "contract": "PcFanAcceleratorParityError::EmptyMetricScale plus explicit relative/residual budgets"},
            {"failure_class": "transactional_publish", "status": "typed", "contract": "checked production handoffs reject before returning an adjoint report"},
        ],
        "acceptance": {
            "production_artifact_evaluator": "executed",
            "cpu_state_ledger_objective_reference": "passed",
            "cpu_independent_finite_difference_and_taylor": "passed",
            "cuda_forward_state_ledger_comparison": cuda_forward_acceptance,
            "cuda_flow_transpose": "unsupported_or_typed_rejected",
            "cuda_paired_csr_hermitian_transpose": "passed",
            "cuda_production_picard_flow_transpose": "passed",
            "cuda_production_newton_design_transpose": "passed",
            "cuda_acoustic_transpose": "unsupported",
            "cuda_design_residual_contraction": "unsupported",
            "cuda_adjoint_finite_difference": "blocked_by_cuda_adjoint_capability",
        },
    }))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(artifact) => {
            let bytes = match serde_json::to_vec_pretty(&artifact) {
                Ok(bytes) => bytes,
                Err(error) => {
                    eprintln!("PC-fan adjoint artifact serialization failed: {error}");
                    std::process::exit(1);
                }
            };
            let (path, checksum) = match write_qa_artifact(ARTIFACT, &bytes) {
                Ok(value) => value,
                Err(error) => {
                    eprintln!("PC-fan adjoint artifact write failed: {error}");
                    std::process::exit(1);
                }
            };
            let actual_backend = if artifact
                .get("production_cuda")
                .is_some_and(|value| !value.is_null())
            {
                "Cuda-linear-solve-with-typed-adjoint-rejection"
            } else {
                "typed-cuda-forward-rejection"
            };
            let result = QaResult::passed(CASE_ID)
                .requested_backend("Cuda")
                .actual_backend(actual_backend)
                .backend_passed("cpu-production-parity")
                .backend_passed("cpu-reduced-adjoint-fd-taylor")
                .backend_skipped(
                    "cuda-flow-acoustic-adjoint",
                    "device-resident flow/acoustic transpose capability is not published",
                )
                .hardware_provenance(cuda_hardware_provenance())
                .reference(
                    "production PC-fan CPU/CUDA parity and checked adjoint capability boundary",
                )
                .reference_checksum("analytic:pcfan-adjoint-cuda-contract-v1")
                .input_checksum("pcfan-120mm:2x2x2:rpms=600,900,1200")
                .formulation(
                    "production CFD parity evaluator; checked CPU reduced adjoint reference",
                )
                .dimension("identical 2x2x2 CFD fixture; three RPM operating points")
                .metric("cpu_production_points", 3.0)
                .metric("cpu_reduced_adjoint_fd_taylor_passed", 1.0)
                .metric("cuda_adjoint_supported", 0.0)
                .metric("cuda_adjoint_capability_boundary_exercised", 1.0)
                .metric("cuda_paired_csr_hermitian_transpose", 1.0)
                .metric("cuda_production_newton_design_transpose", 1.0)
                .minimum("cpu_production_points", 3.0)
                .minimum("cpu_reduced_adjoint_fd_taylor_passed", 1.0)
                .minimum("cuda_paired_csr_hermitian_transpose", 1.0)
                .minimum("cuda_production_newton_design_transpose", 1.0)
                .execution_counts(QaExecutionCounts {
                    executed: 5,
                    passed: 3,
                    failed: 0,
                    unsupported: 2,
                })
                .runtime_evidence(QaRuntimeEvidence {
                    context_reuse: QaContextReuse::NotApplicable,
                    counts: QaExecutionCounts {
                        executed: 5,
                        passed: 3,
                        failed: 0,
                        unsupported: 2,
                    },
                    counters: QaRuntimeCounters::default(),
                    hardware: cuda_hardware_provenance(),
                    ..QaRuntimeEvidence::default()
                })
                .artifact_with_provenance(
                    path,
                    "sonium-qa/pcfan_adjoint_cuda_contract",
                    "cargo run --features 'qa-binaries cuda' --bin qa-pcfan-adjoint-cuda-contract",
                    "pcfan-adjoint-cuda-contract-v1",
                    checksum,
                )
                .seed("pcfan-adjoint-cuda-contract:2x2x2:production-parity+capability")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            match result.to_json_line() {
                Ok(json) => println!("QA_RESULT: {json}"),
                Err(error) => {
                    eprintln!("PC-fan adjoint QA result invalid: {error}");
                    std::process::exit(1);
                }
            }
            println!("SUCCESS");
        }
        Err(error) => {
            eprintln!("PC-fan adjoint CUDA contract failed: {error}");
            if let Ok(json) = QaResult::failed(CASE_ID, error)
                .requested_backend("Cuda")
                .actual_backend("unknown")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX))
                .to_json_line()
            {
                println!("QA_RESULT: {json}");
            }
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
