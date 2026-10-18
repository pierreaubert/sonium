//! PORT-OPTIM Step 10: bounded optimisation gate (Tier 0/1 full, Tier 2 capped).
//!
//! This module runs the CI-cheap bounded optimisation that backs the
//! `qa-speaker-port-optim` QA gate ([`STEP10_CASE_ID`]): a full Tier-0 duct
//! screen plus AL-BOBYQA refinement, a full Tier-1 spline-linear re-rank of
//! the quota, and a capped Tier-2 leg (straight tube plus the re-ranked
//! winner at small/nominal/`Vmax` drives, coarse screening fidelity, one
//! equal-airflow rescale run). It asserts the feasibility constraints and
//! the baseline-comparison logic (`LevelDelta`, `EqualAirflowDelta`,
//! [`check_criterion_1`], [`check_criterion_2`]) and emits a deterministic
//! JSON evidence summary ([`Step10GateReport`]).
//!
//! The full campaign (global LHS search, Pareto front, production-fidelity
//! Tier-2 verification, sensitivity legs) stays an ignored/explicit test
//! like the existing runtime ladder; see `tests/port_optim_step7.rs`,
//! `tests/port_optim_step8.rs`, `tests/port_optim_step9.rs`, and the ignored
//! `step10_bounded_gate_explicit` test in `tests/port_optim_step10.rs`.
//!
//! Validated envelope: headline claims stay at the 2D/meridian tier with the
//! Tier-1 oracle cross-check. A null result (winner == straight, or no
//! equal-airflow improvement) with archived evidence is a successful gate
//! outcome; only the machinery verdicts must pass.

use std::path::Path;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::baseline::PortOptimReferenceCase;
use super::campaign::{
    CampaignArchiverError, Criterion1Inputs, EqualAirflowDelta, GateVerdict, LevelDelta,
    STEP7_GRID_LEVELS_N5, STEP7_TIER1_COARSE_MESH_M, STEP7_TIER2_DRIVES_V_PEAK,
    campaign_duct_config, campaign_f_target_hz, campaign_profile_options, check_criterion_1,
    check_criterion_2, coarse_screening_fidelity, equal_airflow_voltage, grid_candidates,
    production_fidelity, quota_selection_with_straight,
};
use super::error::PortFlowError;
use super::evaluate::{
    PortCandidateMetrics, PortEvalDrive, PortEvalFidelity, SmallSignalReference, Tier2EvalContext,
};
use super::optimize::{
    AlBobyqaConfig, BudgetUsage, DEFAULT_CACHE_QUANTIZATION_STEP, Duct1dTier0Evaluator, EvalTier,
    EvaluatedCandidate, FeasibilitySpec, FunnelConfig, PARETO_VIOLATION_PENALTY, PortOptimError,
    PortOptimFunnel, RankedCandidate, ScalarWeights, SplineTier1Evaluator, StageReport,
    Tier1Evaluator, TierBudgets, campaign_template, tier2_slot_template,
};

/// QA case id for the bounded port-optim gate (mirrored in
/// `crates/sonium-qa/validation-manifest.toml`).
pub const STEP10_CASE_ID: &str = "speaker.port-optim-bounded.v1";

/// Schema version for [`Step10GateReport`] JSON.
pub const STEP10_REPORT_SCHEMA_VERSION: u32 = 1;

/// Provenance string for [`Step10GateReport`] JSON.
pub const STEP10_REPORT_PROVENANCE: &str = "port-optim-step10-bounded-gate.v1";

/// Minimum Tier-0 feasible count on the `N = 5` campaign grid (the
/// airflow/tuning gates admit only a near-straight sliver; the frozen
/// reference case yields exactly 6, so this pins the fixture too).
pub const STEP10_MIN_TIER0_FEASIBLE: usize = 6;

/// Tier-2 fidelity behind the bounded gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Step10Tier2Fidelity {
    /// Skip Tier 2 entirely (Tier-0/1 evidence only; deterministic and fast).
    Skip,
    /// Coarse screening fidelity ([`coarse_screening_fidelity`]): the CI-cheap
    /// default behind the QA gate.
    Capped,
    /// Production fidelity ([`production_fidelity`]): explicit runs only, like
    /// the runtime ladder.
    Production,
}

/// Bounded-gate configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Step10GateConfig {
    /// Control-point count `N` (`N - 3` free radii; campaign default 5).
    pub n_control_points: usize,
    /// Full-factorial Tier-0 grid levels per free variable (campaign: 11).
    pub grid_levels: usize,
    /// Tier-1 re-rank quota (campaign: 6, straight first).
    pub quota: usize,
    /// Run the Tier-1 spline-linear re-rank of the quota.
    pub run_tier1: bool,
    /// Tier-2 leg behind the gate.
    pub tier2: Step10Tier2Fidelity,
    /// Max candidates solved concurrently per stage/wave (`1` = serial).
    ///
    /// Applies to the Tier-0 screen, the Tier-1 re-rank, and the Tier-2
    /// verification waves. Rankings and Tier-2 metrics match the serial
    /// path; only per-record wall times vary run to run.
    pub stage_parallelism: usize,
    /// Flare cap as a multiple of the bore radius (`Some`, e.g. `1.4`,
    /// enables the flare search space at campaign and slot scales).
    #[serde(default)]
    pub flare_ratio: Option<f64>,
    /// Port-length search range in m (enables the length variable).
    #[serde(default)]
    pub length_range_m: Option<(f64, f64)>,
    /// Pin the inlet knot to the bore (cabinet-fixed inlet opening).
    #[serde(default)]
    pub pin_interior_mouth: bool,
    /// Pin the exhaust knot to the bore (cabinet-fixed exhaust opening).
    #[serde(default)]
    pub pin_exterior_mouth: bool,
}

impl Step10GateConfig {
    /// QA-gate default: Tier 0/1 full, Tier 2 capped.
    pub fn bounded() -> Self {
        Self {
            n_control_points: 5,
            grid_levels: STEP7_GRID_LEVELS_N5,
            quota: 6,
            run_tier1: true,
            tier2: Step10Tier2Fidelity::Capped,
            stage_parallelism: 1,
            flare_ratio: None,
            length_range_m: None,
            pin_interior_mouth: false,
            pin_exterior_mouth: false,
        }
    }

    /// Fixed-mouths flare search: 1.4·bore flare cap, both mouths pinned
    /// to the bore, port length fixed, and a 5-level grid (125 Tier-0
    /// points in the resulting 3-variable interior-knot space).
    pub fn flare_search() -> Self {
        Self {
            grid_levels: 5,
            flare_ratio: Some(1.4),
            pin_interior_mouth: true,
            pin_exterior_mouth: true,
            ..Self::bounded()
        }
    }

    /// Fast deterministic Tier-0-only flow for default unit tests.
    pub fn tier0_only() -> Self {
        Self {
            run_tier1: false,
            tier2: Step10Tier2Fidelity::Skip,
            ..Self::bounded()
        }
    }

    /// Explicit full-fidelity flow: Tier 0/1 plus production Tier 2.
    pub fn production() -> Self {
        Self {
            tier2: Step10Tier2Fidelity::Production,
            ..Self::bounded()
        }
    }

    /// Reject an inconsistent gate configuration before any solve.
    pub fn validate(&self) -> Result<(), PortOptimError> {
        let bad = |message: String| PortOptimError::InvalidConfig(message);
        if self.n_control_points < 4 {
            return Err(bad(format!(
                "Step-10 gate needs N >= 4 control points, got {}",
                self.n_control_points
            )));
        }
        if self.grid_levels < 2 {
            return Err(bad(format!(
                "Step-10 gate needs grid_levels >= 2, got {}",
                self.grid_levels
            )));
        }
        if self.quota == 0 {
            return Err(bad("Step-10 gate needs quota >= 1".to_string()));
        }
        if self.stage_parallelism == 0 {
            return Err(bad("Step-10 gate needs stage_parallelism >= 1".to_string()));
        }
        if self.tier2 != Step10Tier2Fidelity::Skip && !self.run_tier1 {
            return Err(bad(
                "Step-10 Tier-2 verification needs the Tier-1 re-ranked winner".to_string(),
            ));
        }
        if let Some(ratio) = self.flare_ratio
            && (!ratio.is_finite() || ratio <= 1.0)
        {
            return Err(bad(format!(
                "Step-10 flare ratio must be finite and above 1, got {ratio}"
            )));
        }
        if let Some((min_m, max_m)) = self.length_range_m
            && (!min_m.is_finite() || !max_m.is_finite() || min_m <= 0.0 || max_m <= min_m)
        {
            return Err(bad(format!(
                "Step-10 length range ({min_m}, {max_m}) must satisfy 0 < min < max"
            )));
        }
        Ok(())
    }
}

/// Tier-0 stage evidence: full screen, refinement, and quota.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step10Tier0Stage {
    /// Full-grid Tier-0 screen (rank order).
    pub screen: StageReport,
    /// Feasible rows in the screen.
    pub feasible_count: usize,
    /// Whether the straight tube (`u = 0`) screened feasible.
    pub straight_feasible: bool,
    /// Best feasible grid point (minimum `J`).
    pub best_grid_u: Vec<f64>,
    /// Objective at the best feasible grid point.
    pub best_grid_j: f64,
    /// Refined best point (`None` when refinement found no feasible point).
    pub refined_u: Option<Vec<f64>>,
    /// Objective at the refined best point.
    pub refined_j: Option<f64>,
    /// Whether refinement stayed feasible.
    pub refined_feasible: bool,
    /// Quota promoted to Tier 1 (straight first).
    pub quota: Vec<Vec<f64>>,
}

/// Tier-1 stage evidence: re-rank of the quota and winner selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step10Tier1Stage {
    /// Tier-1 re-rank of the quota (rank order).
    pub rerank: StageReport,
    /// Whether the straight tube re-ranked feasible.
    pub straight_feasible: bool,
    /// Straight-tube Tier-1 airflow self-ratio (baseline sanity: exactly 1).
    pub straight_q_ratio: f64,
    /// Tier-0 airflow ratio of the winner (SC1 Tier-0 input).
    pub winner_q_ratio_tier0: f64,
    /// Tier-0 tuning of the winner in Hz (SC1 tuning input).
    pub winner_f0_hz: f64,
    /// Winner `u`: minimum-`J` feasible non-straight quota member, else straight.
    pub winner_u: Vec<f64>,
    /// Whether the winner is the straight tube.
    pub winner_is_straight: bool,
}

/// Tier-2 stage evidence: capped two-profile leg plus equal-airflow rescale.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step10Tier2Stage {
    /// Fidelity label (`spc/cycles/tail`, capped or production).
    pub fidelity_label: String,
    /// Drive levels in V peak (small, nominal, Vmax).
    pub drives_v_peak: Vec<f64>,
    /// Straight-tube metrics per drive (ascending voltage).
    pub straight: Vec<PortCandidateMetrics>,
    /// Winner metrics per drive (ascending voltage).
    pub winner: Vec<PortCandidateMetrics>,
    /// Winner verification run at the rescaled equal-airflow drive.
    pub airflow_match: PortCandidateMetrics,
    /// Rescaled winner drive in V peak.
    pub airflow_match_voltage_v: f64,
    /// Equal-voltage winner-vs-straight deltas per drive.
    pub equal_voltage: Vec<LevelDelta>,
    /// Equal-airflow winner-vs-straight delta (nominal level).
    pub equal_airflow: Vec<EqualAirflowDelta>,
    /// Whether every Tier-2 run passed its health flags.
    pub all_healthy: bool,
    /// Whether every Tier-2 run completed its configured steps.
    pub steps_complete: bool,
    /// Configured BDF steps per run.
    pub steps_expected: usize,
}

/// Deterministic bounded-gate evidence summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step10GateReport {
    /// Must equal [`STEP10_REPORT_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Model lineage; must equal [`STEP10_REPORT_PROVENANCE`].
    pub provenance: String,
    /// Frozen reference case the gate ran on.
    pub case: PortOptimReferenceCase,
    /// Campaign tuning target in Hz (Tier-0 duct tuning of straight).
    pub f_target_hz: f64,
    /// Gate configuration behind this report.
    pub config: Step10GateConfig,
    /// Funnel budget consumption (Tier-0 screen/refine and Tier-1 re-rank).
    /// The Tier-2 leg runs through the harness directly, not the funnel, and
    /// is counted by the Tier-2 stage levels instead.
    pub budget: BudgetUsage,
    /// Tier-0 screen/refine/quota evidence (always present).
    pub tier0: Step10Tier0Stage,
    /// Tier-1 re-rank evidence (`None` when skipped).
    pub tier1: Option<Step10Tier1Stage>,
    /// Tier-2 verification evidence (`None` when skipped).
    pub tier2: Option<Step10Tier2Stage>,
    /// Success Criterion 1 verdict (`None` without Tier 2).
    pub criterion_1: Option<GateVerdict>,
    /// Success Criterion 2 verdict without a Pareto archive (`None` w/o Tier 2).
    pub criterion_2_no_pareto: Option<GateVerdict>,
    /// Success Criterion 2 verdict with a Pareto archive (`None` w/o Tier 2).
    pub criterion_2_pareto: Option<GateVerdict>,
    /// Whether the bounded gate passes (machinery + feasibility verdicts).
    pub passed: bool,
    /// One-line deterministic evidence summary.
    pub detail: String,
}

impl Step10GateReport {
    /// Serialize the report to pretty JSON (deterministic field order).
    pub fn to_json_string(&self) -> Result<String, CampaignArchiverError> {
        serde_json::to_string_pretty(self)
            .map_err(|error| CampaignArchiverError::Json(error.to_string()))
    }

    /// Write the report to a JSON file.
    pub fn save_json(&self, path: &Path) -> Result<(), CampaignArchiverError> {
        std::fs::write(path, self.to_json_string()?)
            .map_err(|error| CampaignArchiverError::Io(error.to_string()))
    }

    /// Parse report JSON, rejecting schema or provenance drift loudly.
    pub fn from_json_str(raw: &str) -> Result<Self, CampaignArchiverError> {
        let report: Step10GateReport = serde_json::from_str(raw)
            .map_err(|error| CampaignArchiverError::Json(error.to_string()))?;
        if report.schema_version != STEP10_REPORT_SCHEMA_VERSION {
            return Err(CampaignArchiverError::Json(format!(
                "unsupported Step-10 report schema {}, expected {STEP10_REPORT_SCHEMA_VERSION}",
                report.schema_version
            )));
        }
        if report.provenance != STEP10_REPORT_PROVENANCE {
            return Err(CampaignArchiverError::Json(format!(
                "Step-10 report provenance drift: {}",
                report.provenance
            )));
        }
        Ok(report)
    }

    /// Structural validation: stage coherence, drive coverage incl Vmax,
    /// verdict presence exactly when Tier 2 ran.
    ///
    /// A failing gate (`passed == false`) is still valid evidence; this only
    /// rejects structurally incoherent reports.
    pub fn validate(&self) -> Result<(), CampaignArchiverError> {
        let bad = |message: String| CampaignArchiverError::Json(message);
        let ranked_feasible = self
            .tier0
            .screen
            .ranking
            .iter()
            .filter(|row| row.feasible)
            .count();
        if ranked_feasible != self.tier0.feasible_count {
            return Err(bad(format!(
                "Tier-0 feasible count {} disagrees with {} feasible screen rows",
                self.tier0.feasible_count, ranked_feasible
            )));
        }
        let Some(first) = self.tier0.quota.first() else {
            return Err(bad("Step-10 quota is empty".to_string()));
        };
        let straight_u = gate_template(&self.config, &self.case).straight_u();
        if *first != straight_u {
            return Err(bad(format!(
                "Step-10 quota must lead with the straight tube, got {first:?}"
            )));
        }
        if self.config.run_tier1 != self.tier1.is_some() {
            return Err(bad(
                "Step-10 Tier-1 evidence presence disagrees with the config".to_string(),
            ));
        }
        if let Some(tier1) = &self.tier1
            && !tier1.winner_is_straight
            && !self.tier0.quota.contains(&tier1.winner_u)
        {
            return Err(bad(format!(
                "Step-10 winner {:?} is outside the quota",
                tier1.winner_u
            )));
        }
        let want_tier2 = self.config.tier2 != Step10Tier2Fidelity::Skip;
        if want_tier2 != self.tier2.is_some() {
            return Err(bad(
                "Step-10 Tier-2 evidence presence disagrees with the config".to_string(),
            ));
        }
        if want_tier2
            != (self.criterion_1.is_some()
                && self.criterion_2_no_pareto.is_some()
                && self.criterion_2_pareto.is_some())
        {
            return Err(bad(
                "Step-10 verdict presence disagrees with the Tier-2 config".to_string(),
            ));
        }
        if let Some(tier2) = &self.tier2 {
            let vmax = STEP7_TIER2_DRIVES_V_PEAK
                .iter()
                .copied()
                .fold(0.0f64, f64::max);
            if tier2.drives_v_peak.len() < 2
                || !tier2
                    .drives_v_peak
                    .iter()
                    .any(|v| (*v - vmax).abs() <= 1.0e-12)
            {
                return Err(bad(format!(
                    "Step-10 Tier-2 needs 2+ drives incl Vmax={vmax} V, got {:?}",
                    tier2.drives_v_peak
                )));
            }
            if tier2.straight.len() != tier2.drives_v_peak.len()
                || tier2.winner.len() != tier2.drives_v_peak.len()
            {
                return Err(bad(
                    "Step-10 Tier-2 level count disagrees with drives".to_string()
                ));
            }
            if tier2.equal_voltage.len() != tier2.drives_v_peak.len() {
                return Err(bad(
                    "Step-10 needs one equal-voltage delta per drive".to_string()
                ));
            }
            if tier2.equal_airflow.len() != 1 {
                return Err(bad(format!(
                    "Step-10 needs exactly one equal-airflow level, got {}",
                    tier2.equal_airflow.len()
                )));
            }
        }
        Ok(())
    }
}

/// Campaign template behind a gate config (flare/length applied).
fn gate_template(
    config: &Step10GateConfig,
    case: &PortOptimReferenceCase,
) -> super::optimize::ProfileTemplate {
    let mut template = campaign_template(case, config.n_control_points, campaign_profile_options());
    template.flare_max_radius_m = config
        .flare_ratio
        .map(|ratio| ratio * case.port_radius_r0_m);
    template.length_range_m = config.length_range_m;
    template.pin_interior_mouth = config.pin_interior_mouth;
    template.pin_exterior_mouth = config.pin_exterior_mouth;
    template
}

fn min_j_feasible(rows: &[RankedCandidate]) -> Option<&RankedCandidate> {
    rows.iter()
        .filter(|row| row.feasible && row.objective_j.is_some())
        .min_by(|a, b| {
            a.objective_j
                .partial_cmp(&b.objective_j)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

/// Deduplicated feasible Tier-0 candidates from the funnel history.
fn feasible_tier0(funnel: &PortOptimFunnel) -> Vec<EvaluatedCandidate> {
    let mut feasible = Vec::new();
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
}

fn fidelity_label(fidelity: &PortEvalFidelity, capped: bool) -> String {
    format!(
        "{}:{}spc/{}cyc/{}tail",
        if capped { "capped" } else { "production" },
        fidelity.samples_per_cycle,
        fidelity.total_cycles,
        fidelity.tail_cycles,
    )
}

/// One independent Tier-2 verification solve: profile, drive level, and
/// optional small-signal compression reference.
type Tier2GateJob<'a> = (
    &'a super::profile::SplinePortProfile,
    f64,
    Option<SmallSignalReference>,
);

/// One Tier-2 verification solve runner (shared across wave jobs).
type Tier2GateRunner<'a> = &'a (
        dyn Fn(
    &super::profile::SplinePortProfile,
    f64,
    Option<SmallSignalReference>,
) -> Result<PortCandidateMetrics, PortFlowError>
            + Sync
    );

/// Run one independent Tier-2 verification wave, serially or in bounded
/// parallel chunks. Results keep job order, so parallel waves reproduce the
/// serial metrics exactly (Tier-2 metrics carry no wall times).
fn run_tier2_wave(
    run: Tier2GateRunner<'_>,
    jobs: &[Tier2GateJob<'_>],
    width: usize,
) -> Result<Vec<PortCandidateMetrics>, PortFlowError> {
    if width <= 1 {
        return jobs.iter().map(|job| run(job.0, job.1, job.2)).collect();
    }
    let mut out = Vec::with_capacity(jobs.len());
    for chunk in jobs.chunks(width) {
        let mut partial: Vec<PortCandidateMetrics> = chunk
            .par_iter()
            .map(|job| run(job.0, job.1, job.2))
            .collect::<Result<Vec<_>, _>>()?;
        out.append(&mut partial);
    }
    Ok(out)
}

/// Run the bounded Step-10 gate: Tier-0 screen + refine, optional Tier-1
/// re-rank, optional Tier-2 capped/production leg, then the SC1/SC2 verdicts.
///
/// Deterministic for a fixed config: the same inputs reproduce the same
/// report up to linear-solver roundoff (Tier-2 metrics), and exactly for
/// Tier 0/1. The JSON summary excludes wall times so it is byte-stable for
/// the Tier-0-only config.
pub fn run_step10_gate(config: &Step10GateConfig) -> Result<Step10GateReport, PortOptimError> {
    config.validate()?;
    let case = PortOptimReferenceCase::default();
    case.validate()?;
    let options = campaign_profile_options();
    let template = gate_template(config, &case);
    let straight_u = template.straight_u();
    let f_target = campaign_f_target_hz(&case, &template)?;
    let tier0 = Duct1dTier0Evaluator {
        medium: case.medium(),
        duct: campaign_duct_config(&case),
        f_target_hz: f_target,
    };
    let tier1: Option<Box<dyn Tier1Evaluator>> = if config.run_tier1 {
        Some(Box::new(SplineTier1Evaluator {
            tier0,
            chamber_length_m: case.tier1_chamber_length_m,
            chamber_radius_m: case.tier1_chamber_radius_m,
            flare_length_m: 0.01,
            mesh_size_m: STEP7_TIER1_COARSE_MESH_M,
            angular_sectors: None,
            fluid: sonium_fem::physics::acoustics::pressure::FluidProperties::air_20c(),
            box_volume_m3: case.box_volume_m3,
            f_target_hz: f_target,
        }))
    } else {
        None
    };
    let funnel = PortOptimFunnel::new(
        FunnelConfig {
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
            stage_parallelism: config.stage_parallelism,
        },
        Box::new(tier0),
        tier1,
        None,
    )?;

    // ---- Tier 0: full screen + refinement + quota. ----
    let bounds = template.variable_bounds();
    let mut grid: Vec<Vec<f64>> = grid_candidates(template.n_free(), config.grid_levels)
        .iter()
        .map(|point| {
            point
                .iter()
                .zip(&bounds)
                .map(|(g, bound)| {
                    let lo = bound.lower.unwrap_or(0.0);
                    lo + g * (bound.upper.unwrap_or(1.0) - lo)
                })
                .collect()
        })
        .collect();
    if !grid.contains(&straight_u) {
        grid.push(straight_u.clone());
    }
    let screen = funnel.screen_tier0(&grid);
    let straight_row = screen
        .ranking
        .iter()
        .find(|row| row.u == straight_u)
        .ok_or_else(|| {
            PortOptimError::InvalidConfig("straight tube missing from Tier-0 screen".to_string())
        })?;
    let straight_feasible = straight_row.feasible;
    let best_grid = min_j_feasible(&screen.ranking).ok_or_else(|| {
        PortOptimError::InvalidConfig("Tier-0 screen found no feasible candidate".to_string())
    })?;
    let refined = funnel.optimize_al_bobyqa(
        &best_grid.u,
        &AlBobyqaConfig {
            max_outer_iterations: 6,
            max_inner_evals: 60,
            ..AlBobyqaConfig::default()
        },
    )?;
    let feasible0 = feasible_tier0(&funnel);
    let quota = quota_selection_with_straight(&feasible0, config.quota, &straight_u);
    if quota.is_empty() {
        return Err(PortOptimError::InvalidConfig(
            "Tier-0 quota selection came back empty".to_string(),
        ));
    }
    let tier0_stage = Step10Tier0Stage {
        feasible_count: screen.ranking.iter().filter(|row| row.feasible).count(),
        straight_feasible,
        best_grid_u: best_grid.u.clone(),
        best_grid_j: best_grid.objective_j.unwrap_or(f64::NAN),
        refined_u: refined.best_u.clone(),
        refined_j: refined.best_j,
        refined_feasible: refined.feasible,
        quota: quota.clone(),
        screen,
    };

    // ---- Tier 1: re-rank the quota, select the winner. ----
    let mut tier1_stage: Option<Step10Tier1Stage> = None;
    if config.run_tier1 {
        let rerank = funnel.rerank_tier1(&quota)?;
        let straight_tier1 = rerank
            .ranking
            .iter()
            .find(|row| row.u == straight_u)
            .ok_or_else(|| {
                PortOptimError::InvalidConfig(
                    "straight tube missing from Tier-1 re-rank".to_string(),
                )
            })?;
        let winner_row = rerank
            .ranking
            .iter()
            .filter(|row| row.feasible && row.u != straight_u && row.objective_j.is_some())
            .min_by(|a, b| {
                a.objective_j
                    .partial_cmp(&b.objective_j)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        let winner_u = winner_row.map_or_else(|| straight_u.clone(), |row| row.u.clone());
        let winner_is_straight = winner_u == straight_u;
        let winner_tier0 = feasible0.iter().find(|c| c.u == winner_u).ok_or_else(|| {
            PortOptimError::InvalidConfig(format!("winner {winner_u:?} has no Tier-0 record"))
        })?;
        let straight_q_ratio = funnel
            .history()
            .records
            .iter()
            .filter_map(|record| record.candidate.clone())
            .find(|c| c.tier == EvalTier::Tier1 && c.u == straight_u)
            .map(|c| c.q_rms_m3s / c.q_baseline_m3s)
            .unwrap_or(f64::NAN);
        tier1_stage = Some(Step10Tier1Stage {
            straight_feasible: straight_tier1.feasible,
            straight_q_ratio,
            winner_q_ratio_tier0: winner_tier0.q_rms_m3s / winner_tier0.q_baseline_m3s,
            winner_f0_hz: winner_tier0.f0_hz,
            winner_u,
            winner_is_straight,
            rerank,
        });
    }

    // ---- Tier 2: capped two-profile leg + equal-airflow rescale. ----
    let mut tier2_stage: Option<Step10Tier2Stage> = None;
    let mut criterion_1: Option<GateVerdict> = None;
    let mut criterion_2_no_pareto: Option<GateVerdict> = None;
    let mut criterion_2_pareto: Option<GateVerdict> = None;
    if config.tier2 != Step10Tier2Fidelity::Skip {
        let tier1 = tier1_stage.as_ref().ok_or_else(|| {
            PortOptimError::InvalidConfig("Tier-2 leg needs the Tier-1 winner".to_string())
        })?;
        let capped = config.tier2 == Step10Tier2Fidelity::Capped;
        let context = Tier2EvalContext::from_reference_case(case.clone())?;
        let drive_hz = case.tier2_drive_frequency_hz()?;
        let fidelity = if capped {
            coarse_screening_fidelity()
        } else {
            production_fidelity(&case)
        };
        let mut slot_template = tier2_slot_template(&case, config.n_control_points, options);
        slot_template.flare_max_radius_m = config
            .flare_ratio
            .map(|ratio| ratio * slot_template.bore_radius_m);
        slot_template.pin_interior_mouth = config.pin_interior_mouth;
        slot_template.pin_exterior_mouth = config.pin_exterior_mouth;
        let straight_profile = slot_template.straight()?;
        let winner_radii = &tier1.winner_u[..slot_template.n_free().min(tier1.winner_u.len())];
        let winner_profile = slot_template.unpack(winner_radii)?;
        let drives: Vec<f64> = STEP7_TIER2_DRIVES_V_PEAK.to_vec();
        let run = |profile: &super::profile::SplinePortProfile,
                   voltage: f64,
                   reference: Option<SmallSignalReference>|
         -> Result<PortCandidateMetrics, PortFlowError> {
            let drive = PortEvalDrive {
                voltage_peak_v: voltage,
                frequency_hz: drive_hz,
                second_tone: None,
            };
            let mut fidelity = fidelity.clone();
            fidelity.compression_reference = reference;
            Ok(context.evaluate(profile, &drive, &fidelity)?.metrics)
        };
        // Wave 1: small-signal references for both profiles.
        let small = drives[0];
        let wave1 = run_tier2_wave(
            &run,
            &[
                (&straight_profile, small, None),
                (&winner_profile, small, None),
            ],
            config.stage_parallelism,
        )?;
        let mut wave1 = wave1.into_iter();
        let (straight_small, winner_small) = (
            wave1.next().expect("wave 1 runs two jobs"),
            wave1.next().expect("wave 1 runs two jobs"),
        );
        let reference_of = |metrics: &PortCandidateMetrics| SmallSignalReference {
            voltage_peak_v: small,
            port_flow_fundamental_amplitude_m3s: metrics.port_flow_fundamental_amplitude_m3s,
        };
        let straight_ref = reference_of(&straight_small);
        let winner_ref = reference_of(&winner_small);
        // Wave 2: nominal + Vmax with per-profile compression references.
        let wave2_jobs: Vec<Tier2GateJob<'_>> = drives[1..]
            .iter()
            .flat_map(|voltage| {
                [
                    (&straight_profile, *voltage, Some(straight_ref)),
                    (&winner_profile, *voltage, Some(winner_ref)),
                ]
            })
            .collect();
        let wave2 = run_tier2_wave(&run, &wave2_jobs, config.stage_parallelism)?;
        let mut straight = vec![straight_small];
        let mut winner = vec![winner_small];
        let (pairs, remainder) = wave2.as_chunks::<2>();
        assert!(
            remainder.is_empty(),
            "wave-2 jobs come in straight/winner pairs"
        );
        for [straight_metrics, winner_metrics] in pairs {
            straight.push(straight_metrics.clone());
            winner.push(winner_metrics.clone());
        }
        // Equal-airflow rescale at the nominal level + verification run.
        let nominal = drives[1];
        let v_winner = equal_airflow_voltage(
            nominal,
            straight[1].port_flow_rms_m3s,
            winner[1].port_flow_rms_m3s,
        );
        let airflow_match = run(&winner_profile, v_winner, Some(winner_ref))?;
        let mut equal_voltage = Vec::with_capacity(drives.len());
        for (s, w) in straight.iter().zip(winner.iter()) {
            equal_voltage.push(LevelDelta::compare(s, w)?);
        }
        let equal_airflow = vec![EqualAirflowDelta::compare(&straight[1], &airflow_match)?];
        let all_healthy = straight
            .iter()
            .chain(winner.iter())
            .chain(std::iter::once(&airflow_match))
            .all(|m| m.health.passed);
        let steps_expected = fidelity.samples_per_cycle * fidelity.total_cycles;
        let steps_complete = straight
            .iter()
            .chain(winner.iter())
            .chain(std::iter::once(&airflow_match))
            .all(|m| m.steps_completed == steps_expected);
        let vmax_level = straight.len() - 1;
        criterion_1 = Some(check_criterion_1(&Criterion1Inputs {
            airflow_ratio_tier2: equal_voltage[vmax_level].airflow_ratio,
            airflow_ratio_tier0: tier1.winner_q_ratio_tier0,
            f0_winner_hz: tier1.winner_f0_hz,
            f_target_hz: f_target,
            compression_db: winner[vmax_level].compression_db,
        }));
        criterion_2_no_pareto = Some(check_criterion_2(&equal_airflow, false));
        criterion_2_pareto = Some(check_criterion_2(&equal_airflow, true));
        tier2_stage = Some(Step10Tier2Stage {
            fidelity_label: fidelity_label(&fidelity, capped),
            drives_v_peak: drives,
            straight,
            winner,
            airflow_match,
            airflow_match_voltage_v: v_winner,
            equal_voltage,
            equal_airflow,
            all_healthy,
            steps_complete,
            steps_expected,
        });
    }

    // ---- Verdicts. ----
    let t0_ok = tier0_stage.straight_feasible
        && tier0_stage.feasible_count >= STEP10_MIN_TIER0_FEASIBLE
        && tier0_stage.refined_feasible
        && tier0_stage.quota.first().is_some_and(|u| u == &straight_u);
    let t1_ok = tier1_stage.as_ref().is_none_or(|t| t.straight_feasible);
    let improves = tier2_stage.as_ref().is_some_and(|t| {
        !t.equal_airflow.is_empty()
            && t.equal_airflow
                .iter()
                .all(|level| level.d_thd_rel < 0.0 && level.d_noise_db < 0.0)
    });
    let sc2_consistent = match (criterion_2_no_pareto.as_ref(), criterion_2_pareto.as_ref()) {
        (None, None) => true,
        (Some(plain), Some(pareto)) => {
            let want_plain = if improves { "improvement" } else { "fail" };
            let want_pareto = if improves {
                "improvement"
            } else {
                "null-documented"
            };
            plain.passed == improves
                && plain.outcome == want_plain
                && pareto.passed
                && pareto.outcome == want_pareto
        }
        _ => false,
    };
    let t2_ok = match (&tier2_stage, &criterion_1) {
        (None, None) => true,
        (Some(tier2), Some(sc1)) => {
            tier2.all_healthy && tier2.steps_complete && sc1.passed && sc2_consistent
        }
        _ => false,
    };
    let passed = t0_ok && t1_ok && t2_ok;
    let sc1_label = criterion_1
        .as_ref()
        .map_or("skipped", |v| if v.passed { "pass" } else { "FAIL" });
    let sc2_label = criterion_2_pareto.as_ref().map_or("skipped", |v| {
        if v.passed { v.outcome.as_str() } else { "FAIL" }
    });
    let detail = format!(
        "step-10 bounded gate: Tier-0 {}/{} feasible (straight {}), refine {} J={:.6} -> {:.6}, \
         quota {}{}, SC1[{sc1_label}], SC2[{sc2_label}]",
        tier0_stage.feasible_count,
        tier0_stage.screen.ranking.len(),
        if tier0_stage.straight_feasible {
            "feasible"
        } else {
            "INFEASIBLE"
        },
        if tier0_stage.refined_feasible {
            "feasible"
        } else {
            "INFEASIBLE"
        },
        tier0_stage.best_grid_j,
        tier0_stage.refined_j.unwrap_or(f64::NAN),
        tier0_stage.quota.len(),
        tier1_stage.as_ref().map_or_else(
            || " (Tier-1 skipped)".to_string(),
            |t| format!(
                ", Tier-1 straight {} winner {:?}{}",
                if t.straight_feasible {
                    "feasible"
                } else {
                    "INFEASIBLE"
                },
                t.winner_u,
                tier2_stage.as_ref().map_or_else(
                    || " (Tier-2 skipped)".to_string(),
                    |t2| format!(
                        ", Tier-2 {} healthy={} steps={}",
                        t2.fidelity_label, t2.all_healthy, t2.steps_complete
                    )
                )
            )
        ),
    );

    Ok(Step10GateReport {
        schema_version: STEP10_REPORT_SCHEMA_VERSION,
        provenance: STEP10_REPORT_PROVENANCE.to_string(),
        case,
        f_target_hz: f_target,
        config: config.clone(),
        budget: funnel.usage(),
        tier0: tier0_stage,
        tier1: tier1_stage,
        tier2: tier2_stage,
        criterion_1,
        criterion_2_no_pareto,
        criterion_2_pareto,
        passed,
        detail,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flare_length_gate_tier0_screen_explores_signed_space() {
        let config = Step10GateConfig {
            run_tier1: false,
            tier2: Step10Tier2Fidelity::Skip,
            ..Step10GateConfig::flare_search()
        };
        let report = run_step10_gate(&config).expect("flare gate Tier-0 screen must run");
        report.validate().expect("flare gate report must validate");
        assert!(
            report.tier0.straight_feasible,
            "straight tube must screen feasible"
        );
        assert!(
            report
                .tier0
                .screen
                .ranking
                .iter()
                .any(|row| row.u.len() == 3 && row.u.iter().any(|v| *v < 0.0)),
            "screen must explore negative (flare) interior radii"
        );
        assert_eq!(
            report.tier0.quota.first().map(Vec::as_slice),
            Some(gate_template(&config, &report.case).straight_u().as_slice()),
            "quota must lead with the template straight vector"
        );
    }

    #[test]
    fn tier2_without_tier1_is_rejected_before_any_solve() {
        let config = Step10GateConfig {
            run_tier1: false,
            tier2: Step10Tier2Fidelity::Capped,
            ..Step10GateConfig::bounded()
        };
        let error = config
            .validate()
            .expect_err("must reject Tier-2 without Tier-1");
        assert!(matches!(error, PortOptimError::InvalidConfig(_)));
        let error = run_step10_gate(&config).expect_err("runner must reject Tier-2 without Tier-1");
        assert!(matches!(error, PortOptimError::InvalidConfig(_)));
    }

    #[test]
    fn degenerate_configs_are_rejected() {
        for config in [
            Step10GateConfig {
                n_control_points: 3,
                ..Step10GateConfig::bounded()
            },
            Step10GateConfig {
                grid_levels: 1,
                ..Step10GateConfig::bounded()
            },
            Step10GateConfig {
                quota: 0,
                ..Step10GateConfig::bounded()
            },
        ] {
            assert!(
                config.validate().is_err(),
                "degenerate config must not validate: {config:?}"
            );
        }
    }

    #[test]
    fn report_json_rejects_schema_and_provenance_drift() {
        let report = Step10GateReport {
            schema_version: STEP10_REPORT_SCHEMA_VERSION,
            provenance: STEP10_REPORT_PROVENANCE.to_string(),
            case: PortOptimReferenceCase::default(),
            f_target_hz: 92.0,
            config: Step10GateConfig::tier0_only(),
            budget: BudgetUsage::default(),
            tier0: Step10Tier0Stage {
                screen: StageReport {
                    ranking: Vec::new(),
                    skipped_for_budget: 0,
                },
                feasible_count: 0,
                straight_feasible: false,
                best_grid_u: Vec::new(),
                best_grid_j: 0.0,
                refined_u: None,
                refined_j: None,
                refined_feasible: false,
                quota: Vec::new(),
            },
            tier1: None,
            tier2: None,
            criterion_1: None,
            criterion_2_no_pareto: None,
            criterion_2_pareto: None,
            passed: false,
            detail: String::new(),
        };
        let json = report
            .to_json_string()
            .expect("synthetic report serializes");
        let parsed = Step10GateReport::from_json_str(&json).expect("round-trip parses");
        assert_eq!(parsed.provenance, STEP10_REPORT_PROVENANCE);
        let drifted = json.replace(
            STEP10_REPORT_PROVENANCE,
            "port-optim-step10-bounded-gate.v0",
        );
        assert!(Step10GateReport::from_json_str(&drifted).is_err());
        let bad_schema = json.replace("\"schema_version\": 1", "\"schema_version\": 99");
        assert!(Step10GateReport::from_json_str(&bad_schema).is_err());
    }
}
