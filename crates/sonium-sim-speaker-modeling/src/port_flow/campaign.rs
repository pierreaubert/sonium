//! Step-7 optimisation-campaign machinery (PORT-OPTIM.md Step 7).
//!
//! This module holds the reusable pieces of the reference-case campaign: Tier
//! 0/1/2 configuration builders on the frozen [`PortOptimReferenceCase`],
//! deterministic global-search samplers (full-factorial grid, Latin
//! hypercube), Tier-2 quota selection (best-`J` plus airflow spread),
//! equal-airflow rescaling, the [`Step7Report`] evidence archive with its
//! schema/provenance gate, and the Success-Criteria 1–2 predicates.
//!
//! # Two scales, one `u`
//!
//! Tier 0/1 run at the campaign (round-tube) scale (`R0 = 25 mm`) while Tier
//! 2 runs on the slot surrogate (`bore = 10 mm`). The normalised search
//! variables `u in [0, 1]^(N-3)` are scale-free fractions of the throat span,
//! so the campaign promotes candidates across tiers by promoting the `u`
//! vector and rebuilding the profile at the target tier's template
//! ([`campaign_template`](super::optimize::campaign_template) vs
//! [`tier2_slot_template`](super::optimize::tier2_slot_template)). The two
//! profiles share the same fractional dip shape, not the same absolute radii.
//!
//! # What runs where
//!
//! The expensive solves live in the ignored `port_optim_step7` campaign test
//! (release-only, like the Step-0 baseline regeneration). This module stays
//! solver-free except for the cheap Tier 0 duct tuning used to derive the
//! campaign target; everything else here is configuration, sampling,
//! selection, comparison, and archiving logic covered by fast unit tests.
//!
//! Validated envelope: campaign bookkeeping only. Headline numbers always
//! come from the archived tier evidence behind them (Steps 7–9); the verdict
//! predicates never invent a measurement they were not given.

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::baseline::PortOptimReferenceCase;
use super::duct1d::{DUCT1D_DEFAULT_SEGMENTS, Duct1dConfig, tuning_frequency_hz};
use super::error::PortFlowError;
use super::evaluate::{PortCandidateMetrics, PortEvalFidelity};
use super::optimize::{
    DEFAULT_AIRFLOW_MIN_RATIO, DEFAULT_COMPRESSION_MAX_DB, DEFAULT_TUNING_TOLERANCE_HZ,
    EvaluatedCandidate, ProfileTemplate,
};
use super::profile::{ProfileOptions, SplinePortProfile};

/// Archive schema version for [`Step7Report`].
pub const STEP7_REPORT_SCHEMA_VERSION: u32 = 1;

/// Model lineage recorded in every [`Step7Report`].
pub const STEP7_REPORT_PROVENANCE: &str =
    "PortOptimStep7Campaign[Tier0Duct,Tier1SplineLRF,Tier2MeridianSlot]";

/// Minimum Tier-2 finalists (Step 7: "at least the best 5").
pub const STEP7_MIN_TIER2_FINALISTS: usize = 5;

/// Default full-factorial grid levels per free variable (N = 5: 11² = 121).
pub const STEP7_GRID_LEVELS_N5: usize = 11;

/// Default Latin-hypercube sample count for N != 5 global search.
pub const STEP7_LHS_COUNT: usize = 256;

/// Coarse Tier-1 mesh size in m for survivor re-ranking (reference: 0.01 m).
///
/// The re-rank only needs relative ordering plus the Tier-0/Tier-1 tuning
/// cross-check; absolute Tier-1 claims stay on the reference mesh (Step 0).
pub const STEP7_TIER1_COARSE_MESH_M: f64 = 0.02;

/// Sensitivity pass band: N = 4/7 headline metrics within 5% of N = 5.
pub const STEP7_SENSITIVITY_REL_TOL: f64 = 0.05;

/// Profile CSV station count (`x`, `r(x)` rows).
pub const STEP7_PROFILE_CSV_STATIONS: usize = 201;

/// Tier-2 drive levels for finalist verification (V peak).
///
/// Nominal plus `V_max`, with the small-signal level retained so every
/// candidate carries its own compression reference.
pub const STEP7_TIER2_DRIVES_V_PEAK: [f64; 3] = [0.01, 0.1, 1.0];

// ---------------------------------------------------------------------------
// Campaign configuration builders
// ---------------------------------------------------------------------------

/// Campaign Tier-0 duct configuration: 128 segments, lossless, box load.
pub fn campaign_duct_config(case: &PortOptimReferenceCase) -> Duct1dConfig {
    Duct1dConfig {
        n_segments: DUCT1D_DEFAULT_SEGMENTS,
        thermoviscous: false,
        box_volume_m3: case.box_volume_m3,
    }
}

/// Campaign tuning target: Tier-0 duct tuning of the straight campaign tube.
///
/// Matches the lumped `f_target` to the Step-3 agreement (< 0.1%); the duct
/// value is used so feasibility verdicts are self-consistent with the Tier-0
/// evaluator that judges them.
pub fn campaign_f_target_hz(
    case: &PortOptimReferenceCase,
    template: &ProfileTemplate,
) -> Result<f64, PortFlowError> {
    let straight = template.straight()?;
    tuning_frequency_hz(&straight, case.medium(), &campaign_duct_config(case))
}

/// Campaign profile options: plan defaults (uniform knots, slope ≤ 0.5,
/// shape-preserving PCHIP).
pub fn campaign_profile_options() -> ProfileOptions {
    ProfileOptions::default()
}

/// Coarse Tier-2 screening fidelity: 48 samples/cycle, 10 cycles, 4-cycle
/// tail, no snapshots, no compression reference.
pub fn coarse_screening_fidelity() -> PortEvalFidelity {
    PortEvalFidelity {
        samples_per_cycle: 48,
        total_cycles: 10,
        tail_cycles: 4,
        snapshot_interval_steps: 0,
        compression_reference: None,
    }
}

/// Production Tier-2 fidelity from the frozen reference case (252
/// samples/cycle, 40 cycles, 8-cycle tail).
pub fn production_fidelity(case: &PortOptimReferenceCase) -> PortEvalFidelity {
    PortEvalFidelity::from_reference_case(case)
}

// ---------------------------------------------------------------------------
// Deterministic global-search samplers
// ---------------------------------------------------------------------------

/// Full-factorial grid over the unit box: `levels` points per free variable.
///
/// Returns `levels^n_free` candidates in lexicographic order (first variable
/// slowest). Empty for `n_free == 0` (nothing to search) or `levels < 2`.
pub fn grid_candidates(n_free: usize, levels: usize) -> Vec<Vec<f64>> {
    if n_free == 0 || levels < 2 {
        return Vec::new();
    }
    let count = levels.pow(n_free as u32);
    let mut out = Vec::with_capacity(count);
    for index in 0..count {
        let mut point = vec![0.0; n_free];
        let mut rest = index;
        for var in (0..n_free).rev() {
            let level = rest % levels;
            rest /= levels;
            point[var] = level as f64 / (levels - 1) as f64;
        }
        out.push(point);
    }
    out
}

/// Tiny xorshift64* PRNG: deterministic LHS without a new dependency.
#[derive(Debug, Clone)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        // Multiplicative mix (never the identity map): adjacent seeds must
        // diverge, and the state must be nonzero for xorshift to advance.
        let mixed = seed
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(0xbf58_476d_1ce4_e5b9);
        Self {
            state: if mixed == 0 { 1 } else { mixed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform double in `[0, 1)`.
    fn next_unit(&mut self) -> f64 {
        const SCALE: f64 = (1u64 << 53) as f64;
        ((self.next_u64() >> 11) as f64) / SCALE
    }
}

/// Latin-hypercube samples over the unit box: `count` points, one per
/// stratum per dimension, deterministically shuffled by `seed`.
///
/// Returns an empty vector for `n_free == 0` or `count == 0`.
pub fn lhs_candidates(n_free: usize, count: usize, seed: u64) -> Vec<Vec<f64>> {
    if n_free == 0 || count == 0 {
        return Vec::new();
    }
    let mut rng = XorShift64::new(seed);
    // One permutation of the strata per dimension (Fisher–Yates).
    let mut columns: Vec<Vec<usize>> = Vec::with_capacity(n_free);
    for _ in 0..n_free {
        let mut strata: Vec<usize> = (0..count).collect();
        for i in (1..count).rev() {
            let j = (rng.next_unit() * (i + 1) as f64) as usize;
            strata.swap(i, j.min(i));
        }
        columns.push(strata);
    }
    (0..count)
        .map(|row| {
            columns
                .iter()
                .map(|strata| (strata[row] as f64 + rng.next_unit()) / count as f64)
                .collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tier-2 quota selection: best-J plus airflow spread
// ---------------------------------------------------------------------------

/// Resample a winner profile onto a new control-point count.
///
/// Evaluates the source profile's `r(x)` at the target template's free-knot
/// positions (knots `2..N-1`) and converts back to normalised `u`. This is
/// the Step-7 sensitivity leg ("re-run the winner with `n_control_points` 4
/// and 7"): the same shape re-represented, so any metric change measures
/// control-point dependence, not a different optimum. Requires matching duct
/// lengths; the result is validated through [`ProfileTemplate::unpack`], so
/// an unrepresentable resampling fails loudly instead of silently clipping.
pub fn resample_winner_u(
    source: &SplinePortProfile,
    target: &ProfileTemplate,
) -> Result<Vec<f64>, PortFlowError> {
    if (source.length_m() - target.length_m).abs() > 1.0e-12 * target.length_m.max(1.0e-12) {
        return Err(PortFlowError::InvalidConfig(format!(
            "resampling needs matching duct lengths, got {} m vs {} m",
            source.length_m(),
            target.length_m
        )));
    }
    let knots = target.straight()?.knots().to_vec();
    let span = target.bore_radius_m - target.throat_radius_m;
    if !span.is_finite() || span <= 0.0 {
        return Err(PortFlowError::InvalidConfig(
            "resampling needs a positive target throat span".to_string(),
        ));
    }
    let u: Vec<f64> = knots
        .iter()
        .skip(2)
        .take(target.n_free())
        .map(|x| (target.bore_radius_m - source.local_radius_m(*x)) / span)
        .collect();
    // Validate through the template (bounds, slope, curvature, dense grid).
    let _ = target.unpack(&u)?;
    Ok(u)
}

/// Select Tier-2 quota candidates from feasible Tier-0/1 survivors.
///
/// Tier 0/1 rank by the curvature regulariser alone (THD/noise are
/// unmeasurable below Tier 2), so "top by `J`" clusters near the straight
/// tube. The quota instead takes the straight tube, the best-`J` non-straight
/// survivor, then fills the remaining slots with survivors evenly spaced by
/// airflow ratio — mapping the distortion/airflow trade-off the null-result
/// Pareto front needs. Returns at most `quota` vectors; fewer when fewer
/// survivors exist. Returns an empty vector for `quota == 0`.
pub fn quota_selection(feasible: &[EvaluatedCandidate], quota: usize) -> Vec<Vec<f64>> {
    let straight = feasible
        .first()
        .map_or_else(Vec::new, |c| vec![0.0; c.u.len()]);
    quota_selection_with_straight(feasible, quota, &straight)
}

/// Quota selection with an explicit straight-tube vector.
///
/// Same policy as [`quota_selection`] (straight first, then best `J`,
/// then airflow spread), but the straight vector is supplied by the
/// caller so templates whose straight point is not all-zero (length
/// variable at its nominal coordinate) still lead with the straight tube.
pub fn quota_selection_with_straight(
    feasible: &[EvaluatedCandidate],
    quota: usize,
    straight_u: &[f64],
) -> Vec<Vec<f64>> {
    if quota == 0 || feasible.is_empty() {
        return Vec::new();
    }
    let is_straight = |c: &EvaluatedCandidate| c.u == straight_u;
    let contains = |picked: &[Vec<f64>], u: &[f64]| picked.iter().any(|p| p == u);
    let mut picked: Vec<Vec<f64>> = Vec::new();

    if let Some(straight) = feasible.iter().find(|c| is_straight(c)) {
        picked.push(straight.u.clone());
    }
    let mut by_j: Vec<&EvaluatedCandidate> = feasible.iter().collect();
    by_j.sort_by(|a, b| {
        a.objective_j
            .partial_cmp(&b.objective_j)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for candidate in &by_j {
        if picked.len() >= quota {
            break;
        }
        if !contains(&picked, &candidate.u) {
            picked.push(candidate.u.clone());
            break;
        }
    }
    // Fill by airflow-ratio spread over the remainder.
    let mut rest: Vec<&EvaluatedCandidate> = feasible
        .iter()
        .filter(|c| !contains(&picked, &c.u))
        .collect();
    rest.sort_by(|a, b| {
        let ra = a.q_rms_m3s / a.q_baseline_m3s;
        let rb = b.q_rms_m3s / b.q_baseline_m3s;
        ra.partial_cmp(&rb).unwrap_or(std::cmp::Ordering::Equal)
    });
    if picked.len() < quota && !rest.is_empty() {
        let slots_left = quota - picked.len();
        if slots_left >= rest.len() {
            picked.extend(rest.iter().map(|c| c.u.clone()));
        } else {
            // Evenly spaced indices across the airflow-sorted remainder.
            let mut indices = Vec::with_capacity(slots_left);
            for k in 0..slots_left {
                let idx = k * (rest.len() - 1) / (slots_left - 1).max(1);
                if !indices.contains(&idx) {
                    indices.push(idx);
                }
            }
            for idx in indices {
                if picked.len() >= quota {
                    break;
                }
                picked.push(rest[idx].u.clone());
            }
        }
    }
    picked
}

// ---------------------------------------------------------------------------
// Equal-voltage and equal-airflow comparison
// ---------------------------------------------------------------------------

/// Rescaled winner drive matching the straight tube's flow at `v_ref`.
///
/// The coupled system is near-linear in drive (baseline THD ~ 1e-5), so one
/// linear step lands within a few percent; the verification run at the
/// returned voltage measures the residual honestly instead of assuming it.
/// Returns `v_ref` unchanged when either flow is non-positive (caller must
/// then report the match as unmeasured, never as matched).
pub fn equal_airflow_voltage(v_ref: f64, q_straight_ref: f64, q_winner_ref: f64) -> f64 {
    if q_straight_ref <= 0.0
        || !q_straight_ref.is_finite()
        || q_winner_ref <= 0.0
        || !q_winner_ref.is_finite()
        || !v_ref.is_finite()
        || v_ref <= 0.0
    {
        return v_ref;
    }
    v_ref * q_straight_ref / q_winner_ref
}

/// Relative airflow-match residual in percent: `100·(q_w − q_s)/q_s`.
pub fn airflow_match_residual_pct(q_winner_adj: f64, q_straight_ref: f64) -> f64 {
    if q_straight_ref <= 0.0 || !q_straight_ref.is_finite() || !q_winner_adj.is_finite() {
        return f64::NAN;
    }
    100.0 * (q_winner_adj - q_straight_ref) / q_straight_ref
}

/// Winner-vs-straight deltas at one equal-voltage drive level (Tier 2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LevelDelta {
    /// Drive level (V peak) shared by both runs.
    pub voltage_peak_v: f64,
    /// Straight-tube port-flow RMS in m³/s.
    pub q_straight_m3s: f64,
    /// Winner port-flow RMS in m³/s.
    pub q_winner_m3s: f64,
    /// Airflow ratio `Q_w/Q_s` (constraint: ≥ 0.95).
    pub airflow_ratio: f64,
    /// Absolute THD difference `THD_w − THD_s` (fraction points).
    pub d_thd_abs: f64,
    /// Relative THD difference `(THD_w − THD_s)/THD_s` (fraction).
    pub d_thd_rel: f64,
    /// Noise-proxy ratio `N_w/N_s`.
    pub d_noise_ratio: f64,
    /// Noise-proxy difference `10·log10(N_w/N_s)` in dB.
    pub d_noise_db: f64,
    /// Compression difference `C_w − C_s` in dB (`None` when unmeasured).
    pub d_compression_db: Option<f64>,
    /// Bulk-Mach ratio `M_w/M_s`.
    pub d_mach_ratio: f64,
}

impl LevelDelta {
    /// Compare two Tier-2 metric sets at the same drive level.
    ///
    /// Returns [`PortFlowError::InvalidConfig`] when the levels differ or a
    /// required observable is non-positive where a ratio needs it.
    pub fn compare(
        straight: &PortCandidateMetrics,
        winner: &PortCandidateMetrics,
    ) -> Result<Self, PortFlowError> {
        if straight.voltage_peak_v != winner.voltage_peak_v {
            return Err(PortFlowError::InvalidConfig(format!(
                "equal-voltage comparison needs one drive level, got {} V vs {} V",
                straight.voltage_peak_v, winner.voltage_peak_v
            )));
        }
        for (name, value) in [
            ("straight Q_rms", straight.port_flow_rms_m3s),
            ("winner Q_rms", winner.port_flow_rms_m3s),
            ("straight noise proxy", straight.turbulence_power_trend),
            ("winner noise proxy", winner.turbulence_power_trend),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(PortFlowError::InvalidConfig(format!(
                    "level comparison needs finite positive {name}, got {value}"
                )));
            }
        }
        if !straight.thd_fraction.is_finite() || !winner.thd_fraction.is_finite() {
            return Err(PortFlowError::InvalidConfig(
                "level comparison needs finite THD on both sides".to_string(),
            ));
        }
        let d_compression_db = match (straight.compression_db, winner.compression_db) {
            (Some(s), Some(w)) => Some(w - s),
            _ => None,
        };
        let d_thd_rel = if straight.thd_fraction > 0.0 {
            (winner.thd_fraction - straight.thd_fraction) / straight.thd_fraction
        } else if winner.thd_fraction == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
        Ok(Self {
            voltage_peak_v: straight.voltage_peak_v,
            q_straight_m3s: straight.port_flow_rms_m3s,
            q_winner_m3s: winner.port_flow_rms_m3s,
            airflow_ratio: winner.port_flow_rms_m3s / straight.port_flow_rms_m3s,
            d_thd_abs: winner.thd_fraction - straight.thd_fraction,
            d_thd_rel,
            d_noise_ratio: winner.turbulence_power_trend / straight.turbulence_power_trend,
            d_noise_db: 10.0
                * (winner.turbulence_power_trend / straight.turbulence_power_trend).log10(),
            d_compression_db,
            d_mach_ratio: winner.peak_mach_bulk / straight.peak_mach_bulk,
        })
    }
}

/// Winner-vs-straight comparison at matched airflow (Tier 2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EqualAirflowDelta {
    /// Straight-tube reference drive (V peak).
    pub v_straight_v: f64,
    /// Winner verification drive (V peak, rescaled).
    pub v_winner_v: f64,
    /// Airflow-match residual in percent (see [`airflow_match_residual_pct`]).
    pub match_residual_pct: f64,
    /// Relative THD difference at matched airflow (fraction).
    pub d_thd_rel: f64,
    /// Noise-proxy difference at matched airflow in dB.
    pub d_noise_db: f64,
}

impl EqualAirflowDelta {
    /// Compare the straight reference run against the rescaled winner run.
    ///
    /// The two runs carry different drives by construction; what must match
    /// (within the reported residual) is the port-flow RMS.
    pub fn compare(
        straight_ref: &PortCandidateMetrics,
        winner_adj: &PortCandidateMetrics,
    ) -> Result<Self, PortFlowError> {
        for (name, value) in [
            ("straight Q_rms", straight_ref.port_flow_rms_m3s),
            ("winner Q_rms", winner_adj.port_flow_rms_m3s),
            ("straight noise proxy", straight_ref.turbulence_power_trend),
            ("winner noise proxy", winner_adj.turbulence_power_trend),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(PortFlowError::InvalidConfig(format!(
                    "equal-airflow comparison needs finite positive {name}, got {value}"
                )));
            }
        }
        if !straight_ref.thd_fraction.is_finite() || !winner_adj.thd_fraction.is_finite() {
            return Err(PortFlowError::InvalidConfig(
                "equal-airflow comparison needs finite THD on both sides".to_string(),
            ));
        }
        let d_thd_rel = if straight_ref.thd_fraction > 0.0 {
            (winner_adj.thd_fraction - straight_ref.thd_fraction) / straight_ref.thd_fraction
        } else if winner_adj.thd_fraction == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
        Ok(Self {
            v_straight_v: straight_ref.voltage_peak_v,
            v_winner_v: winner_adj.voltage_peak_v,
            match_residual_pct: airflow_match_residual_pct(
                winner_adj.port_flow_rms_m3s,
                straight_ref.port_flow_rms_m3s,
            ),
            d_thd_rel,
            d_noise_db: 10.0
                * (winner_adj.turbulence_power_trend / straight_ref.turbulence_power_trend).log10(),
        })
    }
}

// ---------------------------------------------------------------------------
// Success-criteria verdicts
// ---------------------------------------------------------------------------

/// Structured gate verdict: pass/fail plus the measured evidence behind it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateVerdict {
    /// Whether the criterion is met.
    pub passed: bool,
    /// Short outcome label (`"improvement"`, `"null-documented"`, `"fail"`, …).
    pub outcome: String,
    /// Human-readable evidence summary with the measured numbers.
    pub detail: String,
}

/// Success Criterion 1 inputs: feasibility of the reported winner.
#[derive(Debug, Clone)]
pub struct Criterion1Inputs {
    /// Winner airflow ratio vs straight at equal drive (Tier 2, Vmax).
    pub airflow_ratio_tier2: f64,
    /// Winner airflow ratio vs straight at `f_target` (Tier 0).
    pub airflow_ratio_tier0: f64,
    /// Winner tuning (Tier 0 campaign scale) in Hz.
    pub f0_winner_hz: f64,
    /// Campaign tuning target in Hz.
    pub f_target_hz: f64,
    /// Winner compression at Vmax in dB (`None` unmeasured → fail).
    pub compression_db: Option<f64>,
}

/// Check Success Criterion 1: airflow ≥ 95% (Tier 0 and Tier 2), tuning
/// within ±2 Hz, compression ≤ 1 dB at Vmax.
pub fn check_criterion_1(inputs: &Criterion1Inputs) -> GateVerdict {
    let tuning_ok = (inputs.f0_winner_hz - inputs.f_target_hz).abs() <= DEFAULT_TUNING_TOLERANCE_HZ;
    let flow0_ok = inputs.airflow_ratio_tier0 >= DEFAULT_AIRFLOW_MIN_RATIO;
    let flow2_ok = inputs.airflow_ratio_tier2 >= DEFAULT_AIRFLOW_MIN_RATIO;
    let compression_ok = inputs
        .compression_db
        .is_some_and(|c| c <= DEFAULT_COMPRESSION_MAX_DB);
    let passed = tuning_ok && flow0_ok && flow2_ok && compression_ok;
    GateVerdict {
        passed,
        outcome: if passed { "feasible" } else { "fail" }.to_string(),
        detail: format!(
            "SC1: Q_ratio_tier0={:.4} (>=0.95: {flow0_ok}), Q_ratio_tier2={:.4} (>=0.95: {flow2_ok}), \
             |f0-f_target|={:.3} Hz (<=2: {tuning_ok}), compression={} (<=1 dB: {compression_ok})",
            inputs.airflow_ratio_tier0,
            inputs.airflow_ratio_tier2,
            (inputs.f0_winner_hz - inputs.f_target_hz).abs(),
            inputs
                .compression_db
                .map_or("unmeasured".to_string(), |c| format!("{c:.4} dB")),
        ),
    }
}

/// Check Success Criterion 2 at equal airflow: lower THD and lower noise
/// proxy than the straight tube at every reported level, or a documented
/// null result (still `passed`, with outcome `"null-documented"`).
///
/// A null result requires the Pareto evidence to exist: the caller passes
/// `pareto_archived = true` once the (distortion, airflow) front is on disk.
/// Without it, a non-improvement is a plain failure, not a null result.
pub fn check_criterion_2(levels: &[EqualAirflowDelta], pareto_archived: bool) -> GateVerdict {
    if levels.is_empty() {
        return GateVerdict {
            passed: false,
            outcome: "fail".to_string(),
            detail: "SC2: no equal-airflow levels reported".to_string(),
        };
    }
    let improves = levels
        .iter()
        .all(|level| level.d_thd_rel < 0.0 && level.d_noise_db < 0.0);
    if improves {
        let detail = levels
            .iter()
            .map(|level| {
                format!(
                    "@V_s={} V: dTHD={:+.2}%, dNoise={:+.3} dB (match {:+.2}%)",
                    level.v_straight_v,
                    100.0 * level.d_thd_rel,
                    level.d_noise_db,
                    level.match_residual_pct,
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        GateVerdict {
            passed: true,
            outcome: "improvement".to_string(),
            detail: format!("SC2: winner beats straight at equal airflow: {detail}"),
        }
    } else if pareto_archived {
        let detail = levels
            .iter()
            .map(|level| {
                format!(
                    "@V_s={} V: dTHD={:+.2}%, dNoise={:+.3} dB (match {:+.2}%)",
                    level.v_straight_v,
                    100.0 * level.d_thd_rel,
                    level.d_noise_db,
                    level.match_residual_pct,
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        GateVerdict {
            passed: true,
            outcome: "null-documented".to_string(),
            detail: format!(
                "SC2: no equal-airflow improvement under inward-only constraint \
                 (documented null result, Pareto front archived): {detail}"
            ),
        }
    } else {
        GateVerdict {
            passed: false,
            outcome: "fail".to_string(),
            detail: "SC2: no equal-airflow improvement and no Pareto front archived".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Evidence archive
// ---------------------------------------------------------------------------

/// One Tier-2-verified finalist with its full tier trail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step7Finalist {
    /// Normalised free radii `u` (scale-free across tiers).
    pub u: Vec<f64>,
    /// Tier-0 campaign-scale observables (`None` when never screened).
    pub tier0: Option<EvaluatedCandidate>,
    /// Tier-1 campaign-scale observables (`None` when never re-ranked).
    pub tier1: Option<EvaluatedCandidate>,
    /// Tier-2 production metrics per drive level (ascending voltage).
    pub tier2: Vec<PortCandidateMetrics>,
}

/// Control-point-count sensitivity point (N = 4 / 7 vs 5).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityPoint {
    /// Control-point count of this leg.
    pub n_control_points: usize,
    /// Resampled winner `u` at this N (same shape, new knots).
    pub best_u: Vec<f64>,
    /// Tier-2 production metrics of the best point per drive level.
    pub best_tier2: Vec<PortCandidateMetrics>,
    /// Tier-2 production metrics of the straight tube per drive level.
    pub straight_tier2: Vec<PortCandidateMetrics>,
    /// Largest relative headline-metric change vs the N = 5 winner.
    pub max_rel_change_vs_n5: f64,
    /// Whether the leg stays inside the 5% band.
    pub within_tolerance: bool,
}

/// Complete Step-7 campaign evidence archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step7Report {
    /// Must equal [`STEP7_REPORT_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Model lineage; must equal [`STEP7_REPORT_PROVENANCE`].
    pub provenance: String,
    /// Frozen reference case the campaign ran on.
    pub case: PortOptimReferenceCase,
    /// Campaign tuning target in Hz (Tier-0 duct tuning of straight).
    pub f_target_hz: f64,
    /// Primary control-point count (5).
    pub n_control_points: usize,
    /// Tier-2-verified finalists (at least [`STEP7_MIN_TIER2_FINALISTS`]).
    pub finalists: Vec<Step7Finalist>,
    /// Winning `u` (minimum full-`J` feasible at Vmax Tier 2).
    pub winner_u: Vec<f64>,
    /// Whether the winner is the straight tube (null-result shape signal).
    pub winner_is_straight: bool,
    /// Equal-voltage winner-vs-straight deltas per drive level.
    pub equal_voltage: Vec<LevelDelta>,
    /// Equal-airflow winner-vs-straight deltas per reference level.
    pub equal_airflow: Vec<EqualAirflowDelta>,
    /// N = 4/7 sensitivity legs.
    pub sensitivity: Vec<SensitivityPoint>,
    /// Success Criterion 1 verdict.
    pub criterion_1: GateVerdict,
    /// Success Criterion 2 verdict.
    pub criterion_2: GateVerdict,
    /// Free-form campaign notes (fidelities, Tier-1 mesh, Pareto location).
    pub notes: String,
}

/// Archive error: IO, JSON, or schema/provenance drift.
#[derive(Debug, thiserror::Error)]
pub enum CampaignArchiverError {
    /// File IO failure.
    #[error("campaign archive IO error: {0}")]
    Io(String),
    /// JSON failure.
    #[error("campaign archive JSON error: {0}")]
    Json(String),
}

impl Step7Report {
    /// Serialize the report to pretty JSON.
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
        let report: Step7Report = serde_json::from_str(raw)
            .map_err(|error| CampaignArchiverError::Json(error.to_string()))?;
        if report.schema_version != STEP7_REPORT_SCHEMA_VERSION {
            return Err(CampaignArchiverError::Json(format!(
                "unsupported Step-7 report schema {}, expected {STEP7_REPORT_SCHEMA_VERSION}",
                report.schema_version
            )));
        }
        if report.provenance != STEP7_REPORT_PROVENANCE {
            return Err(CampaignArchiverError::Json(format!(
                "Step-7 report provenance drift: {}",
                report.provenance
            )));
        }
        Ok(report)
    }

    /// Structural validation: finalist count, drive coverage incl Vmax,
    /// verdict coherence with the archived deltas.
    pub fn validate(&self) -> Result<(), CampaignArchiverError> {
        let bad = |message: String| CampaignArchiverError::Json(message);
        if self.finalists.len() < STEP7_MIN_TIER2_FINALISTS {
            return Err(bad(format!(
                "Step 7 needs at least {STEP7_MIN_TIER2_FINALISTS} Tier-2 finalists, got {}",
                self.finalists.len()
            )));
        }
        let vmax = STEP7_TIER2_DRIVES_V_PEAK
            .iter()
            .copied()
            .fold(0.0f64, f64::max);
        for finalist in &self.finalists {
            if finalist.tier2.len() < 2 {
                return Err(bad(format!(
                    "finalist {:?} needs Tier-2 metrics at 2+ drives, got {}",
                    finalist.u,
                    finalist.tier2.len()
                )));
            }
            if !finalist
                .tier2
                .iter()
                .any(|metrics| (metrics.voltage_peak_v - vmax).abs() <= 1.0e-12)
            {
                return Err(bad(format!(
                    "finalist {:?} misses the Vmax={vmax} V Tier-2 level",
                    finalist.u
                )));
            }
        }
        if self.equal_voltage.len() < 2 {
            return Err(bad(format!(
                "Step 7 needs equal-voltage deltas at 2+ levels, got {}",
                self.equal_voltage.len()
            )));
        }
        if self.equal_airflow.is_empty() {
            return Err(bad(
                "Step 7 needs at least one equal-airflow level".to_string()
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Profile plot data
// ---------------------------------------------------------------------------

/// Sampled `(x, r(x))` plot data as CSV text with a header row.
pub fn profile_csv(profile: &SplinePortProfile, stations: usize) -> String {
    let mut text = String::from("x_m,r_m\n");
    for station in profile.sampled_profile(stations.max(2)) {
        text.push_str(&format!("{:.9e},{:.9e}\n", station[0], station[1]));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::super::evaluate::PortEvalHealth;
    use super::super::optimize::{ConstraintValue, EvalTier};
    use super::*;

    fn candidate(u: Vec<f64>, j: f64, q: f64, q_ref: f64) -> EvaluatedCandidate {
        EvaluatedCandidate {
            u,
            tier: EvalTier::Tier0,
            f0_hz: 92.0,
            q_rms_m3s: q,
            q_baseline_m3s: q_ref,
            thd: None,
            noise_proxy: None,
            compression_db: None,
            peak_mach: None,
            separation_fraction: None,
            curvature_m3: 0.1,
            max_slope: 0.01,
            constraints: vec![ConstraintValue {
                name: "airflow".to_string(),
                value: 0.1,
                enforced: true,
            }],
            feasible: true,
            max_violation: 0.0,
            objective_j: j,
            objective_partial: true,
            cached: false,
            wall_time_ms: 1.0,
        }
    }

    fn metrics(voltage: f64, q: f64, thd: f64, noise: f64, mach: f64) -> PortCandidateMetrics {
        PortCandidateMetrics {
            schema_version: 1,
            provenance: "test".to_string(),
            profile_normalized: vec![0.0],
            n_control_points: 5,
            voltage_peak_v: voltage,
            frequency_hz: 40.0,
            second_tone_frequency_hz: None,
            dt_s: 1.0e-4,
            steps_completed: 100,
            tail_samples: 40,
            n_tail_snapshots: 0,
            port_flow_rms_m3s: q,
            port_flow_fundamental_amplitude_m3s: q,
            box_pressure_amplitude_pa: 1.0,
            peak_excursion_m: 1.0e-4,
            f0_estimate_hz: 90.0,
            thd_fraction: thd,
            imd_fraction: None,
            compression_db: Some(0.01),
            peak_mach_bulk: mach,
            peak_mach_field: None,
            separation_fraction: None,
            turbulence_power_trend: noise,
            curvature_integral_m3: 0.1,
            energy_physical_relative_closure: 0.001,
            energy_algebraic_relative_closure: 1.0e-12,
            energy_interface_relative_residual: 1.0e-12,
            energy_scale_j: 1.0e-6,
            health: PortEvalHealth {
                dissipations_nonnegative: true,
                max_accepted_linear_relative_residual: 1.0e-12,
                linear_residual_ok: true,
                cap_continuity_relative_error: 1.0e-6,
                box_conservation_relative_residual: 1.0e-9,
                passed: true,
            },
        }
    }

    #[test]
    fn grid_sampler_covers_corners_lexicographically() {
        let grid = grid_candidates(2, 3);
        assert_eq!(grid.len(), 9);
        assert_eq!(grid[0], vec![0.0, 0.0]);
        assert_eq!(grid[8], vec![1.0, 1.0]);
        assert_eq!(grid[4], vec![0.5, 0.5]);
        assert!(grid_candidates(0, 3).is_empty());
        assert!(grid_candidates(2, 1).is_empty());
    }

    #[test]
    fn lhs_sampler_stratifies_and_reproduces() {
        let first = lhs_candidates(2, 8, 42);
        let second = lhs_candidates(2, 8, 42);
        assert_eq!(first, second, "LHS must be seed-deterministic");
        assert_eq!(first.len(), 8);
        for dim in 0..2 {
            let mut strata: Vec<usize> = first.iter().map(|p| (p[dim] * 8.0) as usize).collect();
            strata.sort_unstable();
            assert_eq!(strata, (0..8).collect::<Vec<_>>(), "one sample per stratum");
        }
        let other = lhs_candidates(2, 8, 43);
        assert_ne!(first, other, "seed must matter");
        assert!(lhs_candidates(0, 8, 1).is_empty());
    }

    #[test]
    fn quota_selection_keeps_straight_best_and_spread() {
        let feasible = vec![
            candidate(vec![0.0, 0.0], 0.0, 1.0e-3, 1.0e-3),
            candidate(vec![0.1, 0.1], 0.05, 0.99e-3, 1.0e-3),
            candidate(vec![0.3, 0.3], 0.2, 0.97e-3, 1.0e-3),
            candidate(vec![0.5, 0.5], 0.4, 0.95e-3, 1.0e-3),
            candidate(vec![0.2, 0.4], 0.3, 0.96e-3, 1.0e-3),
        ];
        let quota = quota_selection(&feasible, 4);
        assert_eq!(quota.len(), 4);
        assert_eq!(quota[0], vec![0.0, 0.0], "straight always promoted");
        assert!(
            quota.contains(&vec![0.1, 0.1]),
            "best-J non-straight promoted"
        );
        assert!(quota_selection(&feasible, 0).is_empty());
        assert!(quota_selection(&[], 4).is_empty());
    }

    #[test]
    fn equal_airflow_rescaling_matches_flows() {
        let v = equal_airflow_voltage(1.0, 1.0e-3, 0.8e-3);
        assert!((v - 1.25).abs() < 1.0e-12);
        assert!((airflow_match_residual_pct(1.0e-3, 1.0e-3)).abs() < 1.0e-12);
        assert!((airflow_match_residual_pct(1.02e-3, 1.0e-3) - 2.0).abs() < 1.0e-12);
        assert_eq!(equal_airflow_voltage(1.0, 0.0, 0.8e-3), 1.0);
    }

    #[test]
    fn level_delta_reports_improvement_signs() {
        let straight = metrics(1.0, 1.0e-3, 0.02, 1.0e-6, 0.01);
        let mut winner = metrics(1.0, 0.98e-3, 0.015, 0.8e-6, 0.011);
        winner.compression_db = Some(0.02);
        let delta = LevelDelta::compare(&straight, &winner).expect("levels must compare");
        assert!((delta.airflow_ratio - 0.98).abs() < 1.0e-12);
        assert!(delta.d_thd_rel < 0.0);
        assert!(delta.d_noise_db < 0.0);
        assert!((delta.d_compression_db.expect("compression measured") - 0.01).abs() < 1.0e-12);
        let mismatch = LevelDelta::compare(&straight, &metrics(0.1, 1.0e-3, 0.02, 1.0e-6, 0.01));
        assert!(mismatch.is_err(), "drive mismatch must fail loudly");
    }

    #[test]
    fn equal_airflow_delta_compares_across_drives() {
        let straight = metrics(1.0, 1.0e-3, 0.02, 1.0e-6, 0.01);
        let winner = metrics(1.25, 1.001e-3, 0.021, 1.1e-6, 0.012);
        let delta = EqualAirflowDelta::compare(&straight, &winner).expect("must compare");
        assert!((delta.match_residual_pct - 0.1).abs() < 1.0e-9);
        assert!(delta.d_thd_rel > 0.0, "rescaled winner distorts more");
        assert!(delta.d_noise_db > 0.0);
    }

    #[test]
    fn criterion_1_gates_all_four_conditions() {
        let good = Criterion1Inputs {
            airflow_ratio_tier2: 0.99,
            airflow_ratio_tier0: 0.98,
            f0_winner_hz: 93.0,
            f_target_hz: 92.0,
            compression_db: Some(0.1),
        };
        assert!(check_criterion_1(&good).passed);
        let bad_flow = Criterion1Inputs {
            airflow_ratio_tier2: 0.90,
            ..good.clone()
        };
        assert!(!check_criterion_1(&bad_flow).passed);
        let bad_tune = Criterion1Inputs {
            f0_winner_hz: 95.0,
            ..good.clone()
        };
        assert!(!check_criterion_1(&bad_tune).passed);
        let unmeasured = Criterion1Inputs {
            compression_db: None,
            ..good.clone()
        };
        assert!(!check_criterion_1(&unmeasured).passed);
    }

    #[test]
    fn criterion_2_accepts_improvement_or_documented_null() {
        let improved = vec![EqualAirflowDelta {
            v_straight_v: 1.0,
            v_winner_v: 1.1,
            match_residual_pct: 0.5,
            d_thd_rel: -0.1,
            d_noise_db: -0.5,
        }];
        let verdict = check_criterion_2(&improved, false);
        assert!(verdict.passed);
        assert_eq!(verdict.outcome, "improvement");
        let regressed = vec![EqualAirflowDelta {
            d_thd_rel: 0.05,
            d_noise_db: 0.2,
            ..improved[0].clone()
        }];
        let null = check_criterion_2(&regressed, true);
        assert!(null.passed);
        assert_eq!(null.outcome, "null-documented");
        let bare = check_criterion_2(&regressed, false);
        assert!(!bare.passed, "null without Pareto evidence must fail");
        assert!(!check_criterion_2(&[], true).passed);
    }

    #[test]
    fn report_round_trips_and_rejects_drift() {
        let report = Step7Report {
            schema_version: STEP7_REPORT_SCHEMA_VERSION,
            provenance: STEP7_REPORT_PROVENANCE.to_string(),
            case: PortOptimReferenceCase::default(),
            f_target_hz: 92.0,
            n_control_points: 5,
            finalists: vec![Step7Finalist {
                u: vec![0.0, 0.0],
                tier0: None,
                tier1: None,
                tier2: vec![
                    metrics(0.1, 1.0e-4, 0.01, 1.0e-8, 0.001),
                    metrics(1.0, 1.0e-3, 0.02, 1.0e-6, 0.01),
                ],
            }],
            winner_u: vec![0.0, 0.0],
            winner_is_straight: true,
            equal_voltage: vec![],
            equal_airflow: vec![],
            sensitivity: vec![],
            criterion_1: GateVerdict {
                passed: true,
                outcome: "feasible".to_string(),
                detail: "test".to_string(),
            },
            criterion_2: GateVerdict {
                passed: true,
                outcome: "null-documented".to_string(),
                detail: "test".to_string(),
            },
            notes: "unit-test report".to_string(),
        };
        let json = report.to_json_string().expect("must serialize");
        let parsed = Step7Report::from_json_str(&json).expect("must parse");
        assert_eq!(parsed.winner_u, vec![0.0, 0.0]);
        let mut value: serde_json::Value = serde_json::from_str(&json).expect("JSON");
        value["provenance"] = serde_json::Value::from("forged");
        let err = Step7Report::from_json_str(&serde_json::to_string(&value).expect("render"))
            .expect_err("provenance drift must fail");
        assert!(matches!(err, CampaignArchiverError::Json(_)));
        // One finalist cannot satisfy the structural gate.
        assert!(report.validate().is_err());
    }

    #[test]
    fn resample_winner_preserves_shape_across_n() {
        use super::super::optimize::ProfileTemplate;

        let options = ProfileOptions::default();
        let source_template = ProfileTemplate {
            n_control_points: 5,
            length_m: 0.1,
            bore_radius_m: 0.025,
            throat_radius_m: 0.015,
            flare_max_radius_m: None,
            length_range_m: None,
            pin_exterior_mouth: false,
            pin_interior_mouth: false,
            options,
        };
        let source = source_template
            .unpack(&[0.02, 0.03])
            .expect("gentle dip must build");
        for n_control in [4usize, 7usize] {
            let target = ProfileTemplate {
                n_control_points: n_control,
                ..source_template.clone()
            };
            let u = resample_winner_u(&source, &target).expect("resample must validate");
            assert_eq!(u.len(), n_control - 3);
            let rebuilt = target.unpack(&u).expect("resampled must build");
            for station in [0.0, 0.025, 0.05, 0.075, 0.1] {
                let rel = (rebuilt.local_radius_m(station) - source.local_radius_m(station)).abs()
                    / source.local_radius_m(station);
                assert!(
                    rel < 0.02,
                    "N={n_control} resampling drifts {rel:.4} at x={station}"
                );
            }
        }
        let straight_u = resample_winner_u(
            &source_template.straight().expect("straight builds"),
            &ProfileTemplate {
                n_control_points: 7,
                ..source_template.clone()
            },
        )
        .expect("straight resamples");
        assert!(straight_u.iter().all(|v| v.abs() < 1.0e-12));
        let wrong_length = ProfileTemplate {
            length_m: 0.2,
            ..source_template.clone()
        };
        assert!(resample_winner_u(&source, &wrong_length).is_err());
    }

    #[test]
    fn profile_csv_has_header_and_stations() {
        let profile = SplinePortProfile::straight(5, 0.1, 0.025, 0.015, ProfileOptions::default())
            .expect("straight must build");
        let csv = profile_csv(&profile, 11);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 12);
        assert_eq!(lines[0], "x_m,r_m");
        let first: Vec<f64> = lines[1]
            .split(',')
            .map(|v| v.parse().expect("numeric"))
            .collect();
        assert!((first[0] - 0.0).abs() < 1.0e-12);
        assert!((first[1] - 0.025).abs() < 1.0e-12);
    }
}
