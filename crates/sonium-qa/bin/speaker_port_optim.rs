//! QA gate for the PORT-OPTIM bounded optimisation (Step 10).
//!
//! Runs the CI-cheap bounded flow from `sonium_sim_speaker_modeling`: a full
//! Tier-0 duct screen plus AL-BOBYQA refinement, a full Tier-1 spline-linear
//! re-rank of the quota, and a capped Tier-2 leg (straight tube plus the
//! re-ranked winner at small/nominal/`Vmax` drives plus one equal-airflow
//! rescale run). Asserts the feasibility constraints and the
//! baseline-comparison logic, then emits a deterministic evidence summary.
//!
//! Release-mode gate: the Tier-1 re-rank and the capped Tier-2 leg need
//! optimized FEM solves (a few minutes total). The full campaign
//! (production-fidelity Tier-2 verification, Pareto front, sensitivity
//! legs) stays an ignored/explicit test like the existing runtime ladder
//! and is never implied by this binary's SUCCESS.

use std::time::Instant;

use sonium_qa::QaResult;
use sonium_sim_speaker_modeling::port_flow::{
    DEFAULT_AIRFLOW_MIN_RATIO, DEFAULT_COMPRESSION_MAX_DB, DEFAULT_TUNING_TOLERANCE_HZ,
    STEP10_CASE_ID, STEP10_MIN_TIER0_FEASIBLE, Step10GateConfig, Step10GateReport, run_step10_gate,
};

/// Numeric gate evidence extracted from the bounded-gate report.
struct GateNumbers {
    f_target_hz: f64,
    tier0_screened: usize,
    tier0_feasible: usize,
    best_grid_j: f64,
    refined_j: f64,
    tier1_quota: usize,
    tier1_feasible: usize,
    winner_is_straight: bool,
    tier2_runs: usize,
    airflow_ratio_tier2: f64,
    compression_db_vmax: f64,
    tuning_drift_hz: f64,
    d_thd_rel_nominal: f64,
    d_noise_db_nominal: f64,
    match_residual_pct: f64,
    sc2_improves: bool,
    sc2_pareto_outcome: String,
}

fn extract(report: &Step10GateReport) -> Result<GateNumbers, String> {
    let tier1 = report
        .tier1
        .as_ref()
        .ok_or_else(|| "bounded gate must run Tier 1".to_string())?;
    let tier2 = report
        .tier2
        .as_ref()
        .ok_or_else(|| "bounded gate must run Tier 2".to_string())?;
    let sc1 = report
        .criterion_1
        .as_ref()
        .ok_or_else(|| "bounded gate must report SC1".to_string())?;
    let sc2_pareto = report
        .criterion_2_pareto
        .as_ref()
        .ok_or_else(|| "bounded gate must report SC2".to_string())?;
    if !report.passed {
        return Err(format!("bounded gate did not pass: {}", report.detail));
    }
    if !sc1.passed {
        return Err(format!("SC1 failed: {}", sc1.detail));
    }
    if !sc2_pareto.passed {
        return Err(format!("SC2 failed: {}", sc2_pareto.detail));
    }
    let vmax = tier2
        .equal_voltage
        .last()
        .ok_or_else(|| "bounded gate Tier-2 needs an equal-voltage Vmax level".to_string())?;
    let compression_db_vmax = tier2
        .winner
        .last()
        .and_then(|metrics| metrics.compression_db)
        .ok_or_else(|| "bounded gate Tier-2 misses Vmax compression".to_string())?;
    let nominal = tier2
        .equal_airflow
        .first()
        .ok_or_else(|| "bounded gate Tier-2 needs the equal-airflow nominal level".to_string())?;
    Ok(GateNumbers {
        f_target_hz: report.f_target_hz,
        tier0_screened: report.tier0.screen.ranking.len(),
        tier0_feasible: report.tier0.feasible_count,
        best_grid_j: report.tier0.best_grid_j,
        refined_j: report.tier0.refined_j.unwrap_or(f64::NAN),
        tier1_quota: report.tier0.quota.len(),
        tier1_feasible: tier1
            .rerank
            .ranking
            .iter()
            .filter(|row| row.feasible)
            .count(),
        winner_is_straight: tier1.winner_is_straight,
        tier2_runs: tier2.straight.len() + tier2.winner.len() + 1,
        airflow_ratio_tier2: vmax.airflow_ratio,
        compression_db_vmax,
        tuning_drift_hz: (tier1.winner_f0_hz - report.f_target_hz).abs(),
        d_thd_rel_nominal: nominal.d_thd_rel,
        d_noise_db_nominal: nominal.d_noise_db,
        match_residual_pct: nominal.match_residual_pct,
        sc2_improves: nominal.d_thd_rel < 0.0 && nominal.d_noise_db < 0.0,
        sc2_pareto_outcome: sc2_pareto.outcome.clone(),
    })
}

fn run() -> Result<(Step10GateReport, GateNumbers), String> {
    let mut config = Step10GateConfig::bounded();
    if let Ok(value) = std::env::var("SPEAKER_PORT_OPTIM_STAGE_PARALLELISM") {
        config.stage_parallelism = value.parse::<usize>().map_err(|_| {
            format!(
                "SPEAKER_PORT_OPTIM_STAGE_PARALLELISM must be a positive integer, got {value:?}"
            )
        })?;
    }
    let report = run_step10_gate(&config).map_err(|error| error.to_string())?;
    report.validate().map_err(|error| error.to_string())?;
    let numbers = extract(&report)?;
    Ok((report, numbers))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((report, numbers)) => {
            let result = QaResult::passed(STEP10_CASE_ID)
                .reference("frozen Step-0 reference case; straight-tube baseline at equal voltage and equal airflow")
                .reference_checksum("analytic/deterministic-fixture:speaker-port-optim-bounded-v1")
                .dimension("N=5 spline port (2 free radii), 121-point Tier-0 grid, quota of 6, 7 capped Tier-2 runs")
                .formulation("Tier-0 duct screen + AL-BOBYQA refine, Tier-1 spline-linear re-rank, capped Tier-2 two-profile leg with equal-airflow rescale")
                .tolerance("Tier-0 feasible >= 6 incl straight; Tier-1 straight feasible; Tier-2 all healthy; SC1 pass; SC2 pareto-consistent; Q ratio >= 0.95; compression <= 1 dB; tuning <= 2 Hz")
                .metric("f_target_hz", numbers.f_target_hz)
                .metric("tier0_screened", numbers.tier0_screened as f64)
                .metric("tier0_feasible", numbers.tier0_feasible as f64)
                .metric("best_grid_j", numbers.best_grid_j)
                .metric("refined_j", numbers.refined_j)
                .metric("tier1_quota", numbers.tier1_quota as f64)
                .metric("tier1_feasible", numbers.tier1_feasible as f64)
                .metric(
                    "winner_is_straight",
                    f64::from(u8::from(numbers.winner_is_straight)),
                )
                .metric("tier2_runs", numbers.tier2_runs as f64)
                .metric("airflow_ratio_tier2", numbers.airflow_ratio_tier2)
                .metric("compression_db_vmax", numbers.compression_db_vmax)
                .metric("tuning_drift_hz", numbers.tuning_drift_hz)
                .metric("d_thd_rel_nominal", numbers.d_thd_rel_nominal)
                .metric("d_noise_db_nominal", numbers.d_noise_db_nominal)
                .metric("match_residual_pct", numbers.match_residual_pct)
                .metric(
                    "sc2_improves",
                    f64::from(u8::from(numbers.sc2_improves)),
                )
                .minimum("tier0_feasible", STEP10_MIN_TIER0_FEASIBLE as f64)
                .minimum("tier1_quota", 1.0)
                .minimum("tier1_feasible", 1.0)
                .minimum("tier2_runs", 7.0)
                .minimum("airflow_ratio_tier2", DEFAULT_AIRFLOW_MIN_RATIO)
                .maximum("compression_db_vmax", DEFAULT_COMPRESSION_MAX_DB)
                .maximum("tuning_drift_hz", DEFAULT_TUNING_TOLERANCE_HZ)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            println!("GATE_DETAIL: {}", report.detail);
            println!(
                "SC1: {}",
                report
                    .criterion_1
                    .as_ref()
                    .map_or("skipped", |v| v.detail.as_str())
            );
            println!(
                "SC2[pareto:{}]: {}",
                numbers.sc2_pareto_outcome,
                report
                    .criterion_2_pareto
                    .as_ref()
                    .map_or("skipped", |v| v.detail.as_str())
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(STEP10_CASE_ID, error);
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn case_id_matches_the_manifest_entry() {
        assert_eq!(STEP10_CASE_ID, "speaker.port-optim-bounded.v1");
    }

    #[test]
    fn bounded_config_validates_without_solver_work() {
        Step10GateConfig::bounded()
            .validate()
            .expect("bounded config must validate");
    }
}
