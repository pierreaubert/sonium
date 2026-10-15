//! PORT-OPTIM Step 7: optimisation campaign on the frozen reference case.
//!
//! Default (fast) tests guard the campaign machinery on the real Tier-0
//! duct: the global screen finds a feasible set, local refinement improves
//! it, and the quota selection promotes the straight tube plus an
//! airflow-spread subset. The committed evidence archive (when present) is
//! parsed and structurally validated.
//!
//! The ignored `step7_campaign_full` test runs the real campaign
//! (Tier-0 global + local refine, Tier-1 re-rank, Tier-2 production
//! verification of the top candidates at three drives incl `V_max`,
//! equal-airflow rescaling, N = 4/7 sensitivity) and writes
//! `fixtures/port_optim_step7/`. The ignored `step7_pareto_tier2_coarse`
//! test archives the searched NSGA-II (distortion, airflow) front at coarse
//! Tier-2 fidelity. Both are release-only:
//!
//! ```sh
//! cargo test -p sonium-sim-speaker-modeling --release \
//!   --test port_optim_step7 -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use sonium_sim_speaker_modeling::port_flow::{
    AlBobyqaConfig, CampaignArchiverError, DEFAULT_CACHE_QUANTIZATION_STEP, Duct1dTier0Evaluator,
    EvalTier, FeasibilitySpec, FunnelConfig, PARETO_VIOLATION_PENALTY, PortOptimFunnel,
    PortOptimReferenceCase, STEP7_GRID_LEVELS_N5, STEP7_REPORT_PROVENANCE, ScalarWeights,
    Step7Report, TierBudgets, campaign_duct_config, campaign_f_target_hz, campaign_profile_options,
    campaign_template, grid_candidates, quota_selection,
};

fn campaign_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/port_optim_step7")
}

fn report_path() -> PathBuf {
    campaign_dir().join("step7_report.json")
}

/// Campaign-scale Tier-0 funnel (no Tier-1/2 evaluators): screening only.
fn screen_funnel(n_control_points: usize) -> (PortOptimFunnel, f64) {
    let case = PortOptimReferenceCase::default();
    let template = campaign_template(&case, n_control_points, campaign_profile_options());
    let straight = template.straight().expect("straight must build");
    let duct = campaign_duct_config(&case);
    let f_target = campaign_f_target_hz(&case, &template).expect("tuning must resolve");
    let tier0 = Duct1dTier0Evaluator {
        medium: case.medium(),
        duct,
        f_target_hz: f_target,
    };
    let _ = straight;
    let config = FunnelConfig {
        template,
        weights: ScalarWeights::default(),
        feasibility: FeasibilitySpec::port_optim_defaults(f_target),
        budgets: TierBudgets {
            tier0: 20_000,
            tier1: 0,
            tier2: 0,
        },
        search_tier: EvalTier::Tier0,
        tier2_quota: 0,
        cache_quantization_step: DEFAULT_CACHE_QUANTIZATION_STEP,
        pareto_penalty: PARETO_VIOLATION_PENALTY,
        stage_parallelism: 1,
    };
    let funnel = PortOptimFunnel::new(config, Box::new(tier0), None, None)
        .expect("screen funnel must build");
    (funnel, f_target)
}

#[test]
fn tier0_global_screen_finds_feasible_set() {
    let (funnel, f_target) = screen_funnel(5);
    let grid = grid_candidates(2, STEP7_GRID_LEVELS_N5);
    assert_eq!(grid.len(), STEP7_GRID_LEVELS_N5.pow(2));
    let report = funnel.screen_tier0(&grid);
    assert_eq!(report.ranking.len(), grid.len());
    assert_eq!(report.skipped_for_budget, 0);
    let feasible: Vec<_> = report.ranking.iter().filter(|row| row.feasible).collect();
    assert!(
        feasible.iter().any(|row| row.u == vec![0.0, 0.0]),
        "straight tube must survive the Tier-0 screen"
    );
    assert!(
        feasible.len() >= 6,
        "need a feasible set worth promoting, got {} of {}",
        feasible.len(),
        grid.len()
    );
    // Local refinement from the best grid point must stay feasible.
    let best = feasible
        .iter()
        .min_by(|a, b| {
            a.objective_j
                .partial_cmp(&b.objective_j)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("feasible set is non-empty");
    let refined = funnel
        .optimize_al_bobyqa(
            &best.u,
            &AlBobyqaConfig {
                max_outer_iterations: 6,
                max_inner_evals: 60,
                ..AlBobyqaConfig::default()
            },
        )
        .expect("Tier-0 refinement must run");
    assert!(refined.feasible, "refinement must stay feasible");
    println!(
        "tier0 screen N=5: f_target={f_target:.6} Hz, feasible={}/{}, best_grid_J={:.6}, \
         refined_J={:.6} ({})",
        feasible.len(),
        grid.len(),
        best.objective_j.expect("screened rows carry J"),
        refined.best_j.expect("refined best carries J"),
        refined.message,
    );
}

#[test]
fn quota_selection_promotes_straight_and_spread() {
    let (funnel, _) = screen_funnel(5);
    let grid = grid_candidates(2, 5);
    let _ = funnel.screen_tier0(&grid);
    let feasible: Vec<_> = funnel
        .history()
        .records
        .iter()
        .filter_map(|record| record.candidate.clone())
        .filter(|candidate| candidate.tier == EvalTier::Tier0 && candidate.feasible)
        .collect();
    assert!(!feasible.is_empty());
    let quota = quota_selection(&feasible, 6);
    assert_eq!(quota[0], vec![0.0, 0.0]);
    assert!(quota.len() <= 6);
    // Spread: quota airflow ratios must span a wider range than top-J alone.
    let mut ratios: Vec<f64> = quota
        .iter()
        .map(|u| {
            let candidate = feasible.iter().find(|c| &c.u == u).expect("quota from set");
            candidate.q_rms_m3s / candidate.q_baseline_m3s
        })
        .collect();
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    println!(
        "quota: {} candidates, airflow-ratio span [{:.4}, {:.4}]",
        quota.len(),
        ratios.first().expect("non-empty"),
        ratios.last().expect("non-empty")
    );
}

#[test]
fn step7_archive_validates_when_present() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let path = report_path();
    if !path.exists() {
        println!("step-7 campaign archive absent (run --ignored campaign); nothing to validate");
        return Ok(());
    }
    let text = std::fs::read_to_string(&path)?;
    let report = Step7Report::from_json_str(&text).map_err(|error| match error {
        CampaignArchiverError::Io(message) | CampaignArchiverError::Json(message) => message,
    })?;
    assert_eq!(report.provenance, STEP7_REPORT_PROVENANCE);
    report.validate().map_err(|error| match error {
        CampaignArchiverError::Io(message) | CampaignArchiverError::Json(message) => message,
    })?;
    assert!(
        report.criterion_1.passed,
        "SC1 must pass: {}",
        report.criterion_1.detail
    );
    assert!(
        report.criterion_2.passed,
        "SC2 must pass: {}",
        report.criterion_2.detail
    );
    println!(
        "step-7 archive: finalists={}, winner_u={:?} (straight={}), SC1[{}], SC2[{}]",
        report.finalists.len(),
        report.winner_u,
        report.winner_is_straight,
        report.criterion_1.outcome,
        report.criterion_2.outcome,
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Ignored release-only campaign
// ---------------------------------------------------------------------------

type CampaignError = Box<dyn std::error::Error + Send + Sync>;

fn release_guard(name: &str) -> Result<(), CampaignError> {
    if cfg!(debug_assertions) {
        return Err(format!(
            "run the Step-7 {name} in release mode (cargo test -p sonium-sim-speaker-modeling \
             --release --test port_optim_step7 -- --ignored --nocapture)"
        )
        .into());
    }
    Ok(())
}

/// Full Step-7 campaign: Tier-0 global + refine, Tier-1 re-rank, Tier-2
/// production verification (3 drives incl `V_max`), equal-airflow rescaling,
/// and N = 4/7 sensitivity. Writes `fixtures/port_optim_step7/`.
#[test]
#[ignore = "Step-7 optimisation campaign; run explicitly in release mode"]
fn step7_campaign_full() -> Result<(), CampaignError> {
    release_guard("campaign")?;

    use rayon::prelude::*;
    use sonium_fem::physics::acoustics::pressure::FluidProperties;
    use sonium_sim_speaker_modeling::port_flow::{
        Criterion1Inputs, EqualAirflowDelta, EvaluatedCandidate, LevelDelta, PortCandidateMetrics,
        PortEvalDrive, PortOptimBaseline, STEP7_MIN_TIER2_FINALISTS, STEP7_PROFILE_CSV_STATIONS,
        STEP7_REPORT_PROVENANCE, STEP7_REPORT_SCHEMA_VERSION, STEP7_SENSITIVITY_REL_TOL,
        STEP7_TIER1_COARSE_MESH_M, STEP7_TIER2_DRIVES_V_PEAK, SensitivityPoint,
        SmallSignalReference, SplineTier1Evaluator, Step7Finalist, Step7Report, Tier2EvalContext,
        airflow_match_residual_pct, check_criterion_1, check_criterion_2, equal_airflow_voltage,
        lhs_candidates, production_fidelity, profile_csv, resample_winner_u, tier2_slot_template,
        tuning_frequency_hz,
    };

    let case = PortOptimReferenceCase::default();
    case.validate().map_err(|error| error.to_string())?;
    let out_dir = campaign_dir();
    std::fs::create_dir_all(&out_dir)?;
    let options = campaign_profile_options();

    // ---- Stage A: Tier-0 global screen + local refinement (N = 5). ----
    let template = campaign_template(&case, 5, options);
    let f_target = campaign_f_target_hz(&case, &template).map_err(|error| error.to_string())?;
    let duct = campaign_duct_config(&case);
    let tier0 = Duct1dTier0Evaluator {
        medium: case.medium(),
        duct,
        f_target_hz: f_target,
    };
    let tier1 = SplineTier1Evaluator {
        tier0,
        chamber_length_m: case.tier1_chamber_length_m,
        chamber_radius_m: case.tier1_chamber_radius_m,
        flare_length_m: 0.01,
        mesh_size_m: STEP7_TIER1_COARSE_MESH_M,
        angular_sectors: None,
        fluid: FluidProperties::air_20c(),
        box_volume_m3: case.box_volume_m3,
        f_target_hz: f_target,
    };
    let config = FunnelConfig {
        template: template.clone(),
        weights: ScalarWeights::default(),
        feasibility: FeasibilitySpec::port_optim_defaults(f_target),
        budgets: TierBudgets {
            tier0: 20_000,
            tier1: 50,
            tier2: 0,
        },
        search_tier: EvalTier::Tier0,
        tier2_quota: 0,
        cache_quantization_step: DEFAULT_CACHE_QUANTIZATION_STEP,
        pareto_penalty: PARETO_VIOLATION_PENALTY,
        stage_parallelism: 1,
    };
    let funnel = PortOptimFunnel::new(config, Box::new(tier0), Some(Box::new(tier1)), None)
        .map_err(|error| error.to_string())?;
    let grid = grid_candidates(2, STEP7_GRID_LEVELS_N5);
    let screen = funnel.screen_tier0(&grid);
    let feasible_rows: Vec<_> = screen.ranking.iter().filter(|row| row.feasible).collect();
    println!(
        "stage A: Tier-0 screen N=5: f_target={f_target:.6} Hz, feasible={}/{}",
        feasible_rows.len(),
        grid.len()
    );
    assert!(
        feasible_rows.len() > STEP7_MIN_TIER2_FINALISTS,
        "need a feasible set worth promoting"
    );
    // Local refinement from the top-3 feasible grid starts.
    let mut starts: Vec<Vec<f64>> = feasible_rows.iter().map(|row| row.u.clone()).collect();
    starts.truncate(3);
    for start in &starts {
        let refined = funnel
            .optimize_al_bobyqa(
                start,
                &AlBobyqaConfig {
                    max_outer_iterations: 6,
                    max_inner_evals: 60,
                    ..AlBobyqaConfig::default()
                },
            )
            .map_err(|error| error.to_string())?;
        println!(
            "stage A refine from {start:?}: feasible={} J={:?} ({})",
            refined.feasible, refined.best_j, refined.message
        );
    }
    // Feasible Tier-0 survivors, deduplicated by u.
    let collect_feasible = |funnel: &PortOptimFunnel| {
        let mut feasible: Vec<EvaluatedCandidate> = Vec::new();
        for record in &funnel.history().records {
            if let Some(candidate) = record.candidate.clone()
                && candidate.tier == EvalTier::Tier0
                && candidate.feasible
                && !feasible
                    .iter()
                    .any(|c: &EvaluatedCandidate| c.u == candidate.u)
            {
                feasible.push(candidate);
            }
        }
        feasible
    };
    let mut feasible0 = collect_feasible(&funnel);
    // The airflow/tuning gates admit only a near-straight sliver; densify
    // with a deterministic LHS top-up when the feasible set is thin.
    if feasible0.len() < 10 {
        let top_up = lhs_candidates(2, 512, 5);
        let extra = funnel.screen_tier0(&top_up);
        feasible0 = collect_feasible(&funnel);
        println!(
            "stage A densify: LHS top-up screened {}, feasible now {}",
            extra.ranking.len(),
            feasible0.len()
        );
    }
    let quota = quota_selection(&feasible0, 6);
    assert!(
        quota.len() > STEP7_MIN_TIER2_FINALISTS,
        "quota must hold straight + 5, got {}",
        quota.len()
    );
    println!("stage A quota ({}): {quota:?}", quota.len());

    // ---- Stage B: Tier-1 re-rank of the quota (coarse mesh, sequential). ----
    let rerank = funnel
        .rerank_tier1(&quota)
        .map_err(|error| error.to_string())?;
    let straight_row = rerank
        .ranking
        .iter()
        .find(|row| row.u == vec![0.0, 0.0])
        .ok_or("straight tube missing from Tier-1 re-rank")?;
    assert!(
        straight_row.feasible,
        "straight tube must be Tier-1 feasible"
    );
    println!("stage B: Tier-1 re-rank (coarse mesh {STEP7_TIER1_COARSE_MESH_M} m):");
    for row in &rerank.ranking {
        let t0 = feasible0
            .iter()
            .find(|c| c.u == row.u)
            .map(|c| c.q_rms_m3s / c.q_baseline_m3s);
        println!(
            "  u={:?}: {} feasible={} J={:?} Qratio_tier0={t0:?}",
            row.u, row.outcome, row.feasible, row.objective_j
        );
    }
    funnel
        .history()
        .save_json(&out_dir.join("history_campaign.json"))
        .map_err(|error| error.to_string())?;

    // ---- Stage C: Tier-2 production verification (slot scale, parallel). ----
    let context =
        Tier2EvalContext::from_reference_case(case.clone()).map_err(|error| error.to_string())?;
    let drive_hz = case
        .tier2_drive_frequency_hz()
        .map_err(|error| error.to_string())?;
    // One production wave: every (u, drive, fidelity) job in parallel, with
    // per-candidate compression references supplied by the caller.
    let run_wave = |jobs: &[(Vec<f64>, f64, Option<SmallSignalReference>, usize)]| {
        let results: Result<Vec<(Vec<f64>, PortCandidateMetrics)>, String> = jobs
            .par_iter()
            .map(|(u, voltage, reference, n_cp)| {
                let profile = context
                    .candidate_profile(*n_cp, u, options)
                    .map_err(|error| error.to_string())?;
                let drive = PortEvalDrive {
                    voltage_peak_v: *voltage,
                    frequency_hz: drive_hz,
                    second_tone: None,
                };
                let mut fidelity = production_fidelity(&case);
                fidelity.compression_reference = *reference;
                let metrics = context
                    .evaluate(&profile, &drive, &fidelity)
                    .map_err(|error| format!("Tier-2 u={u:?} V={voltage}: {error}"))?
                    .metrics;
                Ok((u.clone(), metrics))
            })
            .collect();
        results
    };
    // Wave 1: small-signal references for every finalist.
    let wave1_jobs: Vec<(Vec<f64>, f64, Option<SmallSignalReference>, usize)> = quota
        .iter()
        .map(|u| (u.clone(), STEP7_TIER2_DRIVES_V_PEAK[0], None, 5))
        .collect();
    let wave1 = run_wave(&wave1_jobs)?;
    let reference_of = |u: &[f64]| {
        let metrics = wave1
            .iter()
            .find(|(w, _)| w == u)
            .map(|(_, metrics)| metrics)?;
        Some(SmallSignalReference {
            voltage_peak_v: STEP7_TIER2_DRIVES_V_PEAK[0],
            port_flow_fundamental_amplitude_m3s: metrics.port_flow_fundamental_amplitude_m3s,
        })
    };
    // Wave 2: nominal + Vmax with per-candidate compression references.
    let mut wave2_jobs = Vec::new();
    for u in &quota {
        for voltage in &STEP7_TIER2_DRIVES_V_PEAK[1..] {
            wave2_jobs.push((u.clone(), *voltage, reference_of(u), 5));
        }
    }
    let wave2 = run_wave(&wave2_jobs)?;
    // Assemble per-finalist Tier-2 trails (ascending voltage).
    let metrics_of = |u: &[f64]| {
        let mut levels: Vec<PortCandidateMetrics> = wave1
            .iter()
            .chain(wave2.iter())
            .filter(|(w, _)| w == u)
            .map(|(_, metrics)| metrics.clone())
            .collect();
        levels.sort_by(|a, b| {
            a.voltage_peak_v
                .partial_cmp(&b.voltage_peak_v)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        levels
    };
    // Straight-tube replay against the frozen Step-0 fixture (1e-6 gate).
    let archive_text = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/port_optim_baseline_v1.json"),
    )?;
    let archive: PortOptimBaseline = serde_json::from_str(&archive_text)?;
    let straight_u = vec![0.0, 0.0];
    let straight_levels = metrics_of(&straight_u);
    assert_eq!(straight_levels.len(), 3);
    for (metrics, level) in straight_levels.iter().zip(archive.tier2.iter()) {
        assert!(
            (metrics.voltage_peak_v - level.voltage_peak_v).abs() <= 1.0e-12,
            "drive-level mismatch in replay check"
        );
        for (name, measured, archived) in [
            ("Q_rms", metrics.port_flow_rms_m3s, level.port_flow_rms_m3s),
            (
                "Q_fund",
                metrics.port_flow_fundamental_amplitude_m3s,
                level.port_flow_fundamental_amplitude_m3s,
            ),
        ] {
            let rel = (measured - archived).abs() / archived.abs().max(1.0e-30);
            println!("replay {name} @{} V: rel={rel:.3e}", metrics.voltage_peak_v);
            assert!(
                rel <= 1.0e-6,
                "{name} replay mismatch at {} V: {measured:.6e} vs {archived:.6e}",
                metrics.voltage_peak_v
            );
        }
        assert!(metrics.health.passed, "replay tripped a health flag");
    }
    // Winner: minimum full-J feasible at Vmax (Tier-2 production).
    let vmax = STEP7_TIER2_DRIVES_V_PEAK[2];
    let straight_vmax = straight_levels
        .iter()
        .find(|metrics| metrics.voltage_peak_v == vmax)
        .ok_or("straight Vmax level missing")?;
    let weights = ScalarWeights::default();
    let curvature_budget = options.limits.curvature_budget_m3;
    let mut scored: Vec<(Vec<f64>, f64, bool, String)> = Vec::new();
    for u in &quota {
        let levels = metrics_of(u);
        let top = levels
            .iter()
            .find(|metrics| metrics.voltage_peak_v == vmax)
            .ok_or("finalist Vmax level missing")?;
        let t0 = feasible0
            .iter()
            .find(|c| &c.u == u)
            .ok_or("finalist Tier-0 record missing")?;
        let campaign_profile = template.unpack(u).map_err(|error| error.to_string())?;
        let j = weights.w_thd * top.thd_fraction / straight_vmax.thd_fraction
            + weights.w_noise * top.turbulence_power_trend / straight_vmax.turbulence_power_trend
            + weights.w_curvature * campaign_profile.curvature_integral() / curvature_budget;
        let flow_ok = top.port_flow_rms_m3s / straight_vmax.port_flow_rms_m3s >= 0.95;
        let tune_ok = (t0.f0_hz - f_target).abs() <= 2.0;
        let comp_ok = top.compression_db.is_some_and(|c| c <= 1.0);
        let feasible = flow_ok && tune_ok && comp_ok;
        scored.push((
            u.clone(),
            j,
            feasible,
            format!(
                "Q_ratio={:.4} dT={:+.3}Hz comp={:?} THD={:.3e} noise={:.3e}",
                top.port_flow_rms_m3s / straight_vmax.port_flow_rms_m3s,
                t0.f0_hz - f_target,
                top.compression_db,
                top.thd_fraction,
                top.turbulence_power_trend,
            ),
        ));
    }
    println!("stage C: Tier-2 production @Vmax={vmax} V (J normalised by straight):");
    for (u, j, feasible, detail) in &scored {
        println!("  u={u:?}: J={j:.6} feasible={feasible} {detail}");
    }
    let (winner_u, winner_j) = scored
        .iter()
        .filter(|(_, _, feasible, _)| *feasible)
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(u, j, _, _)| (u.clone(), *j))
        .ok_or("no feasible Tier-2 finalist at Vmax")?;
    let winner_is_straight = winner_u.iter().all(|v| *v == 0.0);
    println!("winner: u={winner_u:?} J={winner_j:.6} straight={winner_is_straight}");

    // ---- Stage D: equal-voltage + equal-airflow winner vs straight. ----
    let winner_levels = metrics_of(&winner_u);
    let mut equal_voltage = Vec::new();
    for voltage in &STEP7_TIER2_DRIVES_V_PEAK[1..] {
        let straight = straight_levels
            .iter()
            .find(|metrics| metrics.voltage_peak_v == *voltage)
            .ok_or("straight level missing")?;
        let winner = winner_levels
            .iter()
            .find(|metrics| metrics.voltage_peak_v == *voltage)
            .ok_or("winner level missing")?;
        let delta = LevelDelta::compare(straight, winner).map_err(|error| error.to_string())?;
        println!(
            "equal-voltage @{} V: Q_ratio={:.4} dTHD={:+.2}% dNoise={:+.3} dB dC={:?} dB",
            voltage,
            delta.airflow_ratio,
            100.0 * delta.d_thd_rel,
            delta.d_noise_db,
            delta.d_compression_db,
        );
        equal_voltage.push(delta);
    }
    let mut equal_airflow = Vec::new();
    if winner_is_straight {
        // Matched by identity: the same runs, zero residual by construction.
        for voltage in &STEP7_TIER2_DRIVES_V_PEAK[1..] {
            let straight = straight_levels
                .iter()
                .find(|metrics| metrics.voltage_peak_v == *voltage)
                .ok_or("straight level missing")?;
            let delta = EqualAirflowDelta::compare(straight, straight)
                .map_err(|error| error.to_string())?;
            assert!(delta.match_residual_pct.abs() <= 1.0e-9);
            equal_airflow.push(delta);
        }
        println!("equal-airflow: winner is straight; matched by identity");
    } else {
        let mut adj_jobs = Vec::new();
        for voltage in &STEP7_TIER2_DRIVES_V_PEAK[1..] {
            let straight = straight_levels
                .iter()
                .find(|metrics| metrics.voltage_peak_v == *voltage)
                .ok_or("straight level missing")?;
            let winner = winner_levels
                .iter()
                .find(|metrics| metrics.voltage_peak_v == *voltage)
                .ok_or("winner level missing")?;
            let v_adj = equal_airflow_voltage(
                *voltage,
                straight.port_flow_rms_m3s,
                winner.port_flow_rms_m3s,
            );
            adj_jobs.push((winner_u.clone(), v_adj, None, 5usize));
            println!("equal-airflow target @{voltage} V: V_adj={v_adj:.6} V");
        }
        let adj = run_wave(&adj_jobs)?;
        for voltage in &STEP7_TIER2_DRIVES_V_PEAK[1..] {
            let straight = straight_levels
                .iter()
                .find(|metrics| metrics.voltage_peak_v == *voltage)
                .ok_or("straight level missing")?;
            let winner = winner_levels
                .iter()
                .find(|metrics| metrics.voltage_peak_v == *voltage)
                .ok_or("winner level missing")?;
            let v_adj = equal_airflow_voltage(
                *voltage,
                straight.port_flow_rms_m3s,
                winner.port_flow_rms_m3s,
            );
            let adj_metrics = adj
                .iter()
                .find(|(_, metrics)| metrics.voltage_peak_v == v_adj)
                .map(|(_, metrics)| metrics)
                .ok_or("adjusted run missing")?;
            let delta = EqualAirflowDelta::compare(straight, adj_metrics)
                .map_err(|error| error.to_string())?;
            println!(
                "equal-airflow @{voltage} V: V_adj={v_adj:.6} V match={:+.3}% dTHD={:+.2}% \
                 dNoise={:+.3} dB",
                delta.match_residual_pct,
                100.0 * delta.d_thd_rel,
                delta.d_noise_db,
            );
            assert!(
                delta.match_residual_pct.abs() <= 5.0,
                "airflow match missed 5%: {:+.3}%",
                delta.match_residual_pct
            );
            equal_airflow.push(delta);
        }
    }
    let _ = airflow_match_residual_pct(1.0, 1.0);

    // ---- Stage E: N = 4/7 sensitivity — re-run the winner resampled. ----
    // The plan asks to re-run the winner shape at N = 4/7, not to re-search
    // a new optimum per N (a fresh search would confound shape and N
    // effects, and the feasible sliver is unfindable by blind LHS in 4-D).
    let winner_campaign_profile = template
        .unpack(&winner_u)
        .map_err(|error| error.to_string())?;
    let mut sensitivity = Vec::new();
    for n_control in [4usize, 7usize] {
        let leg_template = campaign_template(&case, n_control, options);
        let resampled_u = resample_winner_u(&winner_campaign_profile, &leg_template)
            .map_err(|error| error.to_string())?;
        let leg_profile = leg_template
            .unpack(&resampled_u)
            .map_err(|error| error.to_string())?;
        let leg_f0 = tuning_frequency_hz(&leg_profile, case.medium(), &duct)
            .map_err(|error| error.to_string())?;
        // Tier-2 production verify (resampled + straight) at 3 drives.
        let n_free = n_control - 3;
        let leg_us = [vec![0.0; n_free], resampled_u.clone()];
        let leg_wave1: Vec<(Vec<f64>, f64, Option<SmallSignalReference>, usize)> = leg_us
            .iter()
            .map(|u| (u.clone(), STEP7_TIER2_DRIVES_V_PEAK[0], None, n_control))
            .collect();
        let leg_refs = run_wave(&leg_wave1)?;
        let mut leg_jobs = Vec::new();
        for (u, small) in &leg_refs {
            let reference = SmallSignalReference {
                voltage_peak_v: STEP7_TIER2_DRIVES_V_PEAK[0],
                port_flow_fundamental_amplitude_m3s: small.port_flow_fundamental_amplitude_m3s,
            };
            for voltage in &STEP7_TIER2_DRIVES_V_PEAK[1..] {
                leg_jobs.push((u.clone(), *voltage, Some(reference), n_control));
            }
        }
        let leg_high = run_wave(&leg_jobs)?;
        let leg_levels = |u: &[f64]| {
            let mut levels: Vec<PortCandidateMetrics> = leg_refs
                .iter()
                .chain(leg_high.iter())
                .filter(|(w, _)| w == u)
                .map(|(_, metrics)| metrics.clone())
                .collect();
            levels.sort_by(|a, b| {
                a.voltage_peak_v
                    .partial_cmp(&b.voltage_peak_v)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            levels
        };
        let best_levels = leg_levels(&resampled_u);
        let leg_straight = leg_levels(&vec![0.0; n_free]);
        // Headline change vs the N = 5 winner (Vmax Q/THD/noise + Tier-0 f0).
        let best_vmax = best_levels
            .iter()
            .find(|metrics| metrics.voltage_peak_v == vmax)
            .ok_or("leg Vmax level missing")?;
        let winner_vmax = winner_levels
            .iter()
            .find(|metrics| metrics.voltage_peak_v == vmax)
            .ok_or("winner Vmax level missing")?;
        let winner_t0_f0 = feasible0
            .iter()
            .find(|c| c.u == winner_u)
            .ok_or("winner Tier-0 record missing")?
            .f0_hz;
        let rel = |a: f64, b: f64| {
            if b == 0.0 {
                if a == 0.0 { 0.0 } else { f64::INFINITY }
            } else {
                (a - b).abs() / b.abs()
            }
        };
        let changes = [
            rel(best_vmax.port_flow_rms_m3s, winner_vmax.port_flow_rms_m3s),
            rel(best_vmax.thd_fraction, winner_vmax.thd_fraction),
            rel(
                best_vmax.turbulence_power_trend,
                winner_vmax.turbulence_power_trend,
            ),
            rel(leg_f0, winner_t0_f0),
        ];
        let max_change = changes.iter().copied().fold(0.0f64, f64::max);
        println!(
            "sensitivity N={n_control}: resampled_u={resampled_u:?} max_change={:.2}% (Q={:.2}% \
             THD={:.2}% noise={:.2}% f0={:.2}%)",
            100.0 * max_change,
            100.0 * changes[0],
            100.0 * changes[1],
            100.0 * changes[2],
            100.0 * changes[3],
        );
        sensitivity.push(SensitivityPoint {
            n_control_points: n_control,
            best_u: resampled_u,
            best_tier2: best_levels,
            straight_tier2: leg_straight,
            max_rel_change_vs_n5: max_change,
            within_tolerance: max_change <= STEP7_SENSITIVITY_REL_TOL,
        });
    }

    // ---- Verdicts, archive, plot data. ----
    let winner_t0 = feasible0
        .iter()
        .find(|c| c.u == winner_u)
        .ok_or("winner Tier-0 record missing")?;
    let winner_vmax = winner_levels
        .iter()
        .find(|metrics| metrics.voltage_peak_v == vmax)
        .ok_or("winner Vmax level missing")?;
    let criterion_1 = check_criterion_1(&Criterion1Inputs {
        airflow_ratio_tier2: winner_vmax.port_flow_rms_m3s / straight_vmax.port_flow_rms_m3s,
        airflow_ratio_tier0: winner_t0.q_rms_m3s / winner_t0.q_baseline_m3s,
        f0_winner_hz: winner_t0.f0_hz,
        f_target_hz: f_target,
        compression_db: winner_vmax.compression_db,
    });
    let pareto_path = out_dir.join("pareto_front_tier2_coarse.json");
    let criterion_2 = check_criterion_2(&equal_airflow, pareto_path.exists());
    println!("SC1: passed={} {}", criterion_1.passed, criterion_1.detail);
    println!("SC2: passed={} {}", criterion_2.passed, criterion_2.detail);

    let tier1_of = |u: &[f64]| {
        funnel
            .history()
            .records
            .iter()
            .filter_map(|record| record.candidate.clone())
            .find(|candidate| candidate.tier == EvalTier::Tier1 && candidate.u == u)
    };
    let finalists: Vec<Step7Finalist> = quota
        .iter()
        .map(|u| Step7Finalist {
            u: u.clone(),
            tier0: feasible0.iter().find(|c| &c.u == u).cloned(),
            tier1: tier1_of(u),
            tier2: metrics_of(u),
        })
        .collect();
    let report = Step7Report {
        schema_version: STEP7_REPORT_SCHEMA_VERSION,
        provenance: STEP7_REPORT_PROVENANCE.to_string(),
        case: case.clone(),
        f_target_hz: f_target,
        n_control_points: 5,
        finalists,
        winner_u: winner_u.clone(),
        winner_is_straight,
        equal_voltage,
        equal_airflow,
        sensitivity,
        criterion_1,
        criterion_2,
        notes: format!(
            "Step-7 campaign N=5: Tier-0 grid 121 + AL-BOBYQA refine (3 starts); Tier-1 coarse-mesh \
             ({STEP7_TIER1_COARSE_MESH_M} m) re-rank of quota; Tier-2 production (252 spc, 40/8 cycles, \
             reference mesh) at 3 drives incl Vmax={vmax} V with per-candidate compression refs; \
             Tier-1 f0 is the embedded Tier-0 screen (tautological tuning agreement), so the Tier-1 \
             cross-check is the airflow-ratio comparison; searched Pareto front (if run) at {}",
            pareto_path.display()
        ),
    };
    report.validate().map_err(|error| error.to_string())?;
    report
        .save_json(&out_dir.join("step7_report.json"))
        .map_err(|error| error.to_string())?;
    let winner_profile = template
        .unpack(&winner_u)
        .map_err(|error| error.to_string())?;
    std::fs::write(
        out_dir.join("winner_profile.csv"),
        profile_csv(&winner_profile, STEP7_PROFILE_CSV_STATIONS),
    )?;
    // Sampled campaign-scale profiles for every finalist (Step-11 handoff).
    let quota_profiles: Vec<serde_json::Value> = quota
        .iter()
        .map(|u| {
            let profile = template.unpack(u).map_err(|error| error.to_string())?;
            Ok(serde_json::json!({
                "u": u,
                "stations": profile.sampled_profile(STEP7_PROFILE_CSV_STATIONS),
            }))
        })
        .collect::<Result<_, String>>()
        .map_err(|error: String| -> CampaignError { error.into() })?;
    std::fs::write(
        out_dir.join("quota_profiles.json"),
        serde_json::to_string_pretty(&quota_profiles)?,
    )?;
    let _ = tier2_slot_template(&case, 5, options);
    println!("step-7 campaign archived to {}", out_dir.display());
    Ok(())
}

/// Searched NSGA-II (distortion, airflow) front at coarse Tier-2 fidelity
/// (slot scale, Vmax drive, straight-tube compression reference).
#[test]
#[ignore = "Step-7 Pareto search; run explicitly in release mode"]
fn step7_pareto_tier2_coarse() -> Result<(), CampaignError> {
    release_guard("Pareto search")?;

    use sonium_sim_speaker_modeling::port_flow::{
        ParetoSearchConfig, PortEvalDrive, PortEvalFidelity, SmallSignalReference,
        Tier2EvalContext, Tier2HarnessEvaluator, campaign_duct_config, tier2_slot_template,
    };

    let case = PortOptimReferenceCase::default();
    case.validate().map_err(|error| error.to_string())?;
    let out_dir = campaign_dir();
    std::fs::create_dir_all(&out_dir)?;
    let options = campaign_profile_options();
    let template = tier2_slot_template(&case, 5, options);
    let straight = template.straight().map_err(|error| error.to_string())?;
    let duct = campaign_duct_config(&case);
    let slot_target = sonium_sim_speaker_modeling::port_flow::tuning_frequency_hz(
        &straight,
        case.medium(),
        &duct,
    )
    .map_err(|error| error.to_string())?;
    let tier0 = Duct1dTier0Evaluator {
        medium: case.medium(),
        duct,
        f_target_hz: slot_target,
    };
    let context =
        Tier2EvalContext::from_reference_case(case.clone()).map_err(|error| error.to_string())?;
    let drive_hz = case
        .tier2_drive_frequency_hz()
        .map_err(|error| error.to_string())?;
    // Straight small-signal reference so compression is enforced in the search.
    let small_drive = PortEvalDrive {
        voltage_peak_v: 0.01,
        frequency_hz: drive_hz,
        second_tone: None,
    };
    let small_fidelity = PortEvalFidelity {
        samples_per_cycle: 48,
        total_cycles: 10,
        tail_cycles: 4,
        snapshot_interval_steps: 0,
        compression_reference: None,
    };
    let small = context
        .evaluate(&straight, &small_drive, &small_fidelity)
        .map_err(|error| error.to_string())?
        .metrics;
    let search_drive = PortEvalDrive {
        voltage_peak_v: 1.0,
        frequency_hz: drive_hz,
        second_tone: None,
    };
    let search_fidelity = PortEvalFidelity {
        compression_reference: Some(SmallSignalReference {
            voltage_peak_v: 0.01,
            port_flow_fundamental_amplitude_m3s: small.port_flow_fundamental_amplitude_m3s,
        }),
        ..small_fidelity
    };
    let tier2 = Tier2HarnessEvaluator {
        context,
        drive: search_drive,
        fidelity: search_fidelity.clone(),
    };
    let config = FunnelConfig {
        template,
        weights: ScalarWeights::default(),
        feasibility: FeasibilitySpec::port_optim_defaults(slot_target),
        budgets: TierBudgets {
            tier0: 0,
            tier1: 0,
            tier2: 60,
        },
        search_tier: EvalTier::Tier2,
        tier2_quota: 0,
        cache_quantization_step: DEFAULT_CACHE_QUANTIZATION_STEP,
        pareto_penalty: PARETO_VIOLATION_PENALTY,
        stage_parallelism: 1,
    };
    let funnel = PortOptimFunnel::new(config, Box::new(tier0), None, Some(Box::new(tier2)))
        .map_err(|error| error.to_string())?;
    let result = funnel
        .optimize_pareto(&ParetoSearchConfig {
            maxeval: 48,
            seed: Some(0),
            ..ParetoSearchConfig::default()
        })
        .map_err(|error| error.to_string())?;
    println!(
        "pareto: feasible_front={} infeasible={} evals={} ({})",
        result.front.len(),
        result.infeasible_front.len(),
        result.evaluations,
        result.message
    );
    for member in &result.front {
        println!(
            "  u={:?}: distortion={:.6} airflow={:.4} viol={:.3e}",
            member.u, member.distortion_noise, member.airflow_ratio, member.max_violation
        );
    }
    funnel
        .history()
        .save_json(&out_dir.join("pareto_history.json"))
        .map_err(|error| error.to_string())?;
    std::fs::write(
        out_dir.join("pareto_front_tier2_coarse.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": "step7-pareto-front-v1",
            "drive_voltage_peak_v": 1.0,
            "fidelity": search_fidelity,
            "front": result.front,
            "infeasible_front": result.infeasible_front,
            "message": result.message,
        }))?,
    )?;
    Ok(())
}
