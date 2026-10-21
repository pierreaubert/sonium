//! Production resolved-flow backend parity gate.
//!
//! This target deliberately obtains both datasets by executing the bounded
//! PC-fan CFD runner. It must not be changed to construct observations by
//! hand: backend parity is only meaningful when actual dispatch, convergence,
//! and conservation provenance come from the production path.

use sonium_qa::QaResult;
use sonium_sim_pcfan::{
    PcFanAcceleratorParityConfig, PcFanCfdBackend, PcFanConfig, PcFanFlowModel,
    collect_pc_fan_accelerator_parity_run, evaluate_pc_fan_accelerator_parity,
};

const CASE_ID: &str = "pcfan.accelerator-parity-reference.v1";

fn run() -> Result<String, String> {
    let rpms = [600.0, 900.0, 1200.0];
    let design_config = {
        let mut config = PcFanConfig::default_120mm();
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfd;
        config.cfd.cells_x = 2;
        config.cfd.cells_y = 2;
        config.cfd.cells_z = 2;
        // Compare the identical first linearized CFD step. Nonlinear Picard
        // convergence is covered independently; allowing later iterates here
        // conflates backend parity with nonlinear stopping-path differences.
        config.cfd.max_picard_iterations = 1;
        config.cfd.nonlinear_tolerance = 2.0;
        // Both backends must solve the identical linearized step to near
        // machine precision: at the default 1e-4 inner tolerance the CPU
        // ILUT-Schur and CUDA polynomial-Jacobi iterates stop at visibly
        // different states and swamp the 5e-3 parity budget.
        config.cfd.linear_solver_tolerance = 1.0e-12;
        config.cfd.max_relative_mass_imbalance = 10.0;
        config.cfd.max_relative_momentum_imbalance = 10.0;
        config.cfd.max_relative_pressure_closure = 10.0;
        config.cfd.max_relative_energy_imbalance = 10.0;
        config.cfd.allow_host_fallback = false;
        config
    };
    let design = design_config.default_design_variables();

    let mut reference_config = design_config.clone();
    reference_config.cfd.backend = PcFanCfdBackend::Cpu;
    let reference = collect_pc_fan_accelerator_parity_run(&reference_config, &design, &rpms)
        .map_err(|error| error.to_string())?;

    let mut candidate_config = design_config;
    candidate_config.cfd.backend = PcFanCfdBackend::Cuda;
    let candidate = collect_pc_fan_accelerator_parity_run(&candidate_config, &design, &rpms)
        .map_err(|error| error.to_string())?;

    let parity_config = PcFanAcceleratorParityConfig {
        // The two production preconditioners stop at the same configured
        // linear residual but differ by about 0.3% in the retained fields.
        max_relative_error: 5.0e-3,
        ..Default::default()
    };
    let report = evaluate_pc_fan_accelerator_parity(&parity_config, &reference, &candidate)
        .map_err(|error| error.to_string())?;
    if !report.passed {
        return Err(format!(
            "production accelerator parity failed: observation errors={:?}, residual difference={:.3e}, field error={:.3e}, conservation error={:.3e}",
            report.max_relative_errors,
            report.max_residual_difference,
            report.max_field_relative_error,
            report.max_production_relative_error,
        ));
    }
    Ok(format!(
        "production CPU/CUDA parity passed for {} RPM points",
        report.sample_count
    ))
}

fn main() {
    match run() {
        Ok(message) => {
            let mut result = QaResult::passed(CASE_ID);
            result.message = Some(message);
            println!(
                "{}",
                result.to_json_line().expect("QA result serialization")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "{}",
                QaResult::failed(CASE_ID, error)
                    .to_json_line()
                    .expect("QA result serialization")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
