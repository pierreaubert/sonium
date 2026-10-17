//! CPU QA for the production PC-fan CFD mesh-convergence admission path.

// Rust guideline compliant 2026-02-21

use sonium_qa::{QaResult, QaStatus};
use sonium_sim_pcfan::{
    PcFanCfdBackend, PcFanCfdConfig, PcFanCfdMeshConvergenceConfig, PcFanConfig, PcFanFlowModel,
    PcFanMeshResolution, solve_pc_fan_cfd_mesh_convergence,
};
use std::time::Instant;

const CASE_ID: &str = "pcfan.mesh-convergence-reference.v1";
// Adjacent-level admission for this refinement study: the sweep increments
// the cubic grid one cell at a time (4..=8), and a level is admitted only
// when both response changes against the previous level stay within this
// limit, on top of the per-level nonlinear-residual, conservation, and
// metadata gates below. It replaced the former 70% admission. This is a
// mesh-refinement admission limit, not an error bound against a reference
// solution, so it must not be reused as an accuracy certification.
const MAX_ADJACENT_MESH_CHANGE: f64 = 0.05;
const SOLVE_TOLERANCE: f64 = 1.0e-7;
const METADATA_TOLERANCE: f64 = 2.0e-12;

struct Metrics {
    levels: f64,
    converged_status: f64,
    all_levels_converged_status: f64,
    maximum_update_residual: f64,
    final_airflow_cfm: f64,
    final_pressure_abs_pa: f64,
    final_airflow_relative_change: f64,
    final_pressure_relative_change: f64,
    max_metadata_relative_error: f64,
}

fn relative_change(actual: f64, previous: f64) -> f64 {
    (actual - previous).abs() / actual.abs().max(previous.abs()).max(f64::MIN_POSITIVE)
}

fn run() -> Result<Metrics, String> {
    let config = PcFanConfig {
        mesh: PcFanMeshResolution {
            radial_segments: 2,
            chord_segments: 2,
            circumferential_segments: 8,
            harmonic_count: 2,
        },
        cfd: PcFanCfdConfig {
            flow_model: PcFanFlowModel::IncompressibleCfd,
            backend: PcFanCfdBackend::Cpu,
            use_direct_solver: true,
            cells_x: 4,
            cells_y: 4,
            cells_z: 4,
            max_picard_iterations: 2000,
            nonlinear_tolerance: SOLVE_TOLERANCE,
            linear_solver_max_iterations: 2000,
            linear_solver_tolerance: 1.0e-10,
            ..PcFanCfdConfig::default()
        },
        ..PcFanConfig::default_120mm()
    };
    let design = config.default_design_variables();
    let convergence = solve_pc_fan_cfd_mesh_convergence(
        &config,
        &design,
        PcFanCfdMeshConvergenceConfig {
            start_cells: 4,
            max_cells: 8,
            relative_tolerance: MAX_ADJACENT_MESH_CHANGE,
            min_levels: 3,
            max_final_residual: SOLVE_TOLERANCE,
        },
    );

    if convergence.levels.len() < 3 {
        return Err(format!(
            "mesh-convergence returned {} levels, expected at least 3",
            convergence.levels.len()
        ));
    }

    // Include the first level: pair-wise checks alone never inspect its status.
    let mut maximum_update_residual: f64 = 0.0;
    for level in &convergence.levels {
        require_solve_convergence(level.solution.converged, level.solution.residual)?;
        maximum_update_residual = maximum_update_residual.max(level.solution.residual);
    }
    if !convergence.converged || !convergence.all_levels_converged_and_finite {
        return Err(
            "native mesh-convergence report did not admit all solves and final response changes"
                .into(),
        );
    }

    let mut max_metadata_relative_error: f64 = 0.0;
    for pair in convergence.levels.windows(2) {
        let previous = &pair[0];
        let current = &pair[1];
        if current.cells_x <= previous.cells_x
            || current.cells_y <= previous.cells_y
            || current.cells_z <= previous.cells_z
        {
            return Err("mesh-convergence cell counts are not strictly increasing".into());
        }
        if !current.solution.airflow_cfm.is_finite()
            || !current.solution.pressure_drop_pa.is_finite()
            || !current.solution.mass_balance_error_kg_s.is_finite()
            || !current.solution.residual.is_finite()
        {
            return Err(format!(
                "mesh level {} has invalid CFD admission fields: converged={}, airflow={}, pressure={}, mass_balance={}, residual={}",
                current.cells_x,
                current.solution.converged,
                current.solution.airflow_cfm,
                current.solution.pressure_drop_pa,
                current.solution.mass_balance_error_kg_s,
                current.solution.residual,
            ));
        }

        let expected_airflow =
            relative_change(current.solution.airflow_cfm, previous.solution.airflow_cfm);
        let expected_pressure = relative_change(
            current.solution.pressure_drop_pa.abs(),
            previous.solution.pressure_drop_pa.abs(),
        );
        let airflow_error = relative_change(
            current.airflow_relative_change.unwrap_or(f64::NAN),
            expected_airflow,
        );
        let pressure_error = relative_change(
            current.pressure_relative_change.unwrap_or(f64::NAN),
            expected_pressure,
        );
        max_metadata_relative_error = max_metadata_relative_error
            .max(airflow_error)
            .max(pressure_error);
        if !airflow_error.is_finite()
            || !pressure_error.is_finite()
            || airflow_error > METADATA_TOLERANCE
            || pressure_error > METADATA_TOLERANCE
        {
            return Err(format!(
                "mesh-convergence metadata mismatch at {} cells: airflow error={}, pressure error={}",
                current.cells_x, airflow_error, pressure_error
            ));
        }
    }

    let last = convergence
        .levels
        .last()
        .expect("at least three mesh levels were admitted");
    let final_airflow_relative_change = last
        .airflow_relative_change
        .ok_or_else(|| "final mesh level lacks airflow change metadata".to_string())?;
    let final_pressure_relative_change = last
        .pressure_relative_change
        .ok_or_else(|| "final mesh level lacks pressure change metadata".to_string())?;
    if final_airflow_relative_change > MAX_ADJACENT_MESH_CHANGE
        || final_pressure_relative_change > MAX_ADJACENT_MESH_CHANGE
    {
        return Err(format!(
            "final mesh changes exceed admission tolerance: airflow={}, pressure={}",
            final_airflow_relative_change, final_pressure_relative_change
        ));
    }
    if relative_change(
        convergence.selected_solution.airflow_cfm,
        last.solution.airflow_cfm,
    ) > METADATA_TOLERANCE
        || relative_change(
            convergence.selected_solution.pressure_drop_pa,
            last.solution.pressure_drop_pa,
        ) > METADATA_TOLERANCE
    {
        return Err("selected solution does not match final admitted mesh level".into());
    }

    Ok(Metrics {
        levels: convergence.levels.len() as f64,
        converged_status: if convergence.converged { 1.0 } else { 0.0 },
        all_levels_converged_status: if convergence.all_levels_converged_and_finite {
            1.0
        } else {
            0.0
        },
        maximum_update_residual,
        final_airflow_cfm: last.solution.airflow_cfm,
        final_pressure_abs_pa: last.solution.pressure_drop_pa.abs(),
        final_airflow_relative_change,
        final_pressure_relative_change,
        max_metadata_relative_error,
    })
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("levels", metrics.levels)
                .metric("converged_status", metrics.converged_status)
                .metric("final_airflow_cfm", metrics.final_airflow_cfm)
                .metric("final_pressure_abs_pa", metrics.final_pressure_abs_pa)
                .metric(
                    "final_airflow_relative_change",
                    metrics.final_airflow_relative_change,
                )
                .metric(
                    "final_pressure_relative_change",
                    metrics.final_pressure_relative_change,
                )
                .metric(
                    "max_metadata_relative_error",
                    metrics.max_metadata_relative_error,
                )
                .minimum("levels", 3.0)
                .metric(
                    "all_levels_converged_status",
                    metrics.all_levels_converged_status,
                )
                .minimum("all_levels_converged_status", 1.0)
                .metric("maximum_update_residual", metrics.maximum_update_residual)
                .maximum("maximum_update_residual", SOLVE_TOLERANCE)
                .minimum("converged_status", 1.0)
                .maximum("converged_status", 1.0)
                .minimum("final_airflow_cfm", f64::MIN_POSITIVE)
                .minimum("final_pressure_abs_pa", f64::MIN_POSITIVE)
                .minimum("final_airflow_relative_change", 0.0)
                .minimum("final_pressure_relative_change", 0.0)
                .maximum("final_airflow_relative_change", MAX_ADJACENT_MESH_CHANGE)
                .maximum("final_pressure_relative_change", MAX_ADJACENT_MESH_CHANGE)
                .maximum("max_metadata_relative_error", METADATA_TOLERANCE)
                .reference_checksum("analytic/invariant:pcfan-mesh-convergence-v1")
                .input_checksum("pcfan-120mm:incompressible-cfd:cells=4..8:cpu")
                .formulation(
                    "production immersed-solid CFD mesh-convergence metadata and admission",
                )
                .seed("pcfan-mesh-convergence-v1:4..8:adjacent-change-limit=0.05")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            if let Err(error) = record.release_eligible() {
                record.status = QaStatus::Failed;
                record.message = Some(error);
            }
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("serialize PC-fan mesh-convergence result")
            );
            if record.status == QaStatus::Passed {
                println!("SUCCESS");
            } else {
                println!("FAILED");
                std::process::exit(1);
            }
        }
        Err(error) => {
            if let Ok(json) = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX))
                .to_json_line()
            {
                println!("QA_RESULT: {json}");
            }
            eprintln!("PCFAN MESH-CONVERGENCE QA FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

// A small self-reported update cannot admit a failed nonlinear solve.
fn require_solve_convergence(converged: bool, residual: f64) -> Result<(), String> {
    if !converged || !residual.is_finite() || residual < 0.0 || residual > SOLVE_TOLERANCE {
        return Err(format!(
            "mesh level failed nonlinear admission: converged={converged}, update residual={residual:.6e}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{SOLVE_TOLERANCE, require_solve_convergence};

    #[test]
    fn small_update_cannot_hide_an_unconverged_mesh_level() {
        assert!(require_solve_convergence(false, 0.0).is_err());
        assert!(require_solve_convergence(false, SOLVE_TOLERANCE / 2.0).is_err());
    }

    #[test]
    fn convergence_flag_cannot_hide_an_invalid_or_large_update() {
        for residual in [f64::NAN, f64::INFINITY, -1.0, 2.0 * SOLVE_TOLERANCE] {
            assert!(require_solve_convergence(true, residual).is_err());
        }
        assert!(require_solve_convergence(true, SOLVE_TOLERANCE / 2.0).is_ok());
    }
}
