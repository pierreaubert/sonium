//! Objective, constraints, and optimiser wiring for port-shape search (PORT-OPTIM.md Step 6).
//!
//! [`PortOptimFunnel`] owns the tiered evaluation funnel: Tier 0 (lumped/1D
//! duct) screens the full candidate set, Tier 1 (linear viscothermal)
//! re-ranks the survivors, and Tier 2 (nonlinear coupled Navier–Stokes)
//! evaluates only the configured quota per iteration. Every evaluation flows
//! through one choke point ([`PortOptimFunnel::evaluate`]) that applies, in
//! order: candidate caching keyed by quantised `u`, rejection-by-construction
//! (illegal profiles consume no solver budget), per-tier budget accounting,
//! and JSON history recording.
//!
//! # Variables and constraints
//!
//! The search variables are the normalised free radii `u in [0, 1]^(N-3)`
//! (plan decision D3); [`ProfileTemplate::unpack`] builds the candidate
//! profile and [`pack_u`] recovers the variables. Bounds are the unit box in
//! the [`sonium_solvers_optimization`](sonium_solvers_optimization) [`Bound`](sonium_solvers_optimization::Bound)
//! type ([`free_variable_bounds`]). The five constraints — airflow, tuning,
//! compression, slope, curvature — use the solver-wide `c(u) >= 0` feasible
//! convention ([`Constraint`](sonium_solvers_optimization::Constraint) with
//! [`ConstraintKind::Inequality`](sonium_solvers_optimization::ConstraintKind)).
//! Slope and curvature come from the profile geometry (no solve); airflow and
//! tuning come from the evaluating tier; compression is only measurable at
//! Tier 2, so below Tier 2 it is recorded as unenforced and verified at the
//! Tier 2 quota stage instead of guessed.
//!
//! # Objective
//!
//! The scalarised search objective is
//! `J = w_THD·(THD/THD_ref) + w_N·(N/N_ref) + w_k·(κ/κ_max)`, normalised by the
//! straight-tube baseline at the evaluating tier so the weights are
//! scale-free. Tiers that cannot measure a term (Tier 0/1 have no THD or
//! noise) rank by the available terms and flag the value partial
//! ([`EvaluatedCandidate::objective_partial`]); the full `J` is only claimed
//! at Tier 2. Pareto mode returns the (distortion-plus-noise, airflow) front
//! instead of a scalar optimum.
//!
//! # Engines
//!
//! * Default: hand-rolled augmented-Lagrangian outer loop with BOBYQA inner
//!   solves ([`PortOptimFunnel::optimize_al_bobyqa`]), derivative-free as plan
//!   decision D7 requires (the bundled AL only offers gradient inner solvers).
//! * Constrained Bayesian optimisation when the Tier 2 budget is below
//!   [`CBO_TIER2_BUDGET_THRESHOLD`] ([`PortOptimFunnel::optimize_cbo`]).
//! * NSGA-II for Pareto mode ([`PortOptimFunnel::optimize_pareto`]).
//!
//! [`EngineChoice::Auto`] selects between the scalarised engines by the budget
//! rule above.
//!
//! # Validated envelope
//!
//! Optimiser wiring only. Claims about a profile still need the tier evidence
//! behind them (Steps 7–9); this module never reports a candidate feasible
//! without evaluating the enforced constraints. The frozen campaign winner and
//! its evidence links are handed to downstream consumers through
//! [`super::handoff`].
//!
//! # Inward-only limitation
//!
//! The search variables are the normalised free radii `u in [0, 1]^(N-3)`
//! with `r = R0 - u·(R0 - r_min)`, so the funnel can only propose profiles
//! inside the original bore (`r(x) <= R0`). The campaign outcome is honestly a
//! documented SC2 null result under this constraint; outward flare is a
//! follow-up decision, not a silent bound change.
//!
//! # Equal-airflow reporting rule
//!
//! Airflow is a hard constraint (≥ 95% of the straight tube at equal voltage),
//! not an objective, and every winner-vs-straight distortion/noise comparison
//! is reported at matched airflow (rescaled drive plus verification run) next
//! to the equal-voltage deltas. Success Criterion 2 judges the equal-airflow
//! levels only; see [`super::campaign::EqualAirflowDelta`].

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sonium_fem::physics::acoustics::pressure::FluidProperties;
use sonium_solvers_optimization::{
    BayesOptConstraint, BayesOptConstraintFn, BobyqaConfig, Bound, ConstrainedBayesOptConfig,
    Constraint, ConstraintKind, NsgaConfig, NsgaVariant, bobyqa, constrained_bayesian_optimization,
    nsga2,
};

use super::baseline::PortOptimReferenceCase;
use super::duct1d::{Duct1dConfig, solve_duct_1d, tuning_frequency_hz};
use super::error::PortFlowError;
use super::evaluate::{
    PortCandidateMetrics, PortEvalDrive, PortEvalFidelity, Tier2EvalContext, Tier2EvalOutcome,
};
use super::geometry3d::{SplineRoundPortGeometry, build_spline_round_port_mesh};
use super::linear_reference::{
    SplinePortLinearReferenceConfig, solve_spline_port_linear_reference,
};
use super::profile::{PROFILE_VALIDATION_GRID_POINTS, ProfileOptions, SplinePortProfile};
use crate::AcousticMedium;

/// Schema version for [`OptimHistory`].
pub const PORT_OPTIM_HISTORY_SCHEMA_VERSION: u32 = 1;

/// Model lineage recorded in every [`OptimHistory`].
pub const PORT_OPTIM_HISTORY_PROVENANCE: &str =
    "PortOptimStep6Funnel[AugmentedLagrangianBOBYQA,cBO,NSGA-II]";

/// Default cache quantisation step in normalised `u` units.
///
/// Two candidates whose coordinates agree to 1e-9 share a cache entry; the
/// physics tiers cannot resolve finer differences, and the shared entry keeps
/// optimiser re-evaluations (objective plus one call per constraint) free.
pub const DEFAULT_CACHE_QUANTIZATION_STEP: f64 = 1.0e-9;

/// Tier 2 budget below which [`EngineChoice::Auto`] selects constrained
/// Bayesian optimisation over BOBYQA plus augmented Lagrangian (plan D7).
pub const CBO_TIER2_BUDGET_THRESHOLD: usize = 100;

/// Scalarised objective assigned to rejected or failed candidates.
///
/// Exceeds any plausible evaluated merit (`J` is baseline-normalised O(1)),
/// so optimisers move away from illegal regions without any special casing.
pub const REJECTED_OBJECTIVE: f64 = 1.0e6;

/// Additive penalty per unit of constraint violation in Pareto objectives.
pub const PARETO_VIOLATION_PENALTY: f64 = 1.0e3;

/// Constraint names in evaluation order: airflow, tuning, compression, slope,
/// curvature. [`PortOptimFunnel::search_constraints`] exposes the enforced
/// subset at the search tier in this order.
pub const CONSTRAINT_NAMES: [&str; 5] = ["airflow", "tuning", "compression", "slope", "curvature"];

/// Default tuning tolerance in Hz (Success Criterion 1: ±2 Hz).
pub const DEFAULT_TUNING_TOLERANCE_HZ: f64 = 2.0;

/// Default minimum airflow ratio vs the straight tube at equal voltage.
pub const DEFAULT_AIRFLOW_MIN_RATIO: f64 = 0.95;

/// Default maximum port-output compression in dB at `V_max`.
pub const DEFAULT_COMPRESSION_MAX_DB: f64 = 1.0;

// ---------------------------------------------------------------------------
// Tiers, templates, pack/unpack
// ---------------------------------------------------------------------------

/// Physics tier of one funnel evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvalTier {
    /// Lumped/1D variable-area duct screening (milliseconds).
    Tier0,
    /// Linear axisymmetric viscothermal re-rank (seconds to minutes).
    Tier1,
    /// Nonlinear coupled Navier–Stokes quota evaluation (minutes).
    Tier2,
}

/// Fixed geometry behind the normalised search variables `u in [0, 1]^(N-3)`.
///
/// Axial knots are fixed (plan D3); only the free radii vary. The inlet,
/// outlet, and straight-start knots are pinned to `R0` by
/// [`SplinePortProfile::from_normalized`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileTemplate {
    /// Control-point count `N` (needs `N >= 3`; optimisation needs `N >= 4`
    /// so at least one variable is free).
    pub n_control_points: usize,
    /// Port length `L` in m.
    pub length_m: f64,
    /// Straight-bore radius `R0` in m.
    pub bore_radius_m: f64,
    /// Minimum admissible radius `r_min` in m (default `0.6·R0`).
    pub throat_radius_m: f64,
    /// Knot spacing, shape limits, and spline kind.
    pub options: ProfileOptions,
}

impl ProfileTemplate {
    /// Number of free search variables (`N - 3`, saturating at zero).
    pub fn n_free(&self) -> usize {
        self.n_control_points.saturating_sub(3)
    }

    /// Build the straight-tube profile (`u = 0`), validating the template.
    pub fn straight(&self) -> Result<SplinePortProfile, PortFlowError> {
        SplinePortProfile::straight(
            self.n_control_points,
            self.length_m,
            self.bore_radius_m,
            self.throat_radius_m,
            self.options,
        )
    }

    /// Build a candidate profile from normalised variables (`unpack`).
    ///
    /// Rejects out-of-range, non-finite, or wrong-length `u`, and any profile
    /// violating the shape limits, with [`PortFlowError::InvalidConfig`]
    /// before any solve can consume it.
    pub fn unpack(&self, u: &[f64]) -> Result<SplinePortProfile, PortFlowError> {
        SplinePortProfile::from_normalized(
            self.n_control_points,
            self.length_m,
            self.bore_radius_m,
            self.throat_radius_m,
            u,
            self.options,
        )
    }
}

/// Recover the normalised search variables from a profile (`pack`).
pub fn pack_u(profile: &SplinePortProfile) -> Vec<f64> {
    profile.normalized()
}

/// Unit-box bounds for the free variables in solver types.
pub fn free_variable_bounds(n_free: usize) -> Vec<Bound<f64>> {
    vec![Bound::both(0.0, 1.0); n_free]
}

/// Maximum wall slope `|r'(x)|` over the validation grid (dimensionless).
fn max_profile_slope(profile: &SplinePortProfile) -> f64 {
    let grid = PROFILE_VALIDATION_GRID_POINTS;
    let mut max = 0.0f64;
    for i in 0..grid {
        let x = profile.length_m() * i as f64 / (grid - 1) as f64;
        max = max.max(profile.slope(x).abs());
    }
    max
}

// ---------------------------------------------------------------------------
// Tier observables and evaluator traits
// ---------------------------------------------------------------------------

/// Tier 0 observables: tuning, equal-voltage flow, shape audits.
///
/// `q_rms_m3s` may be per-unit-drive (the duct solve uses a unit box
/// pressure): only the ratio against the straight baseline enters the
/// airflow constraint, so the drive scale cancels for the linear tiers.
#[derive(Debug, Clone)]
pub struct Tier0Obs {
    /// Compact Tier-0 tuning estimate in Hz.
    pub f0_hz: f64,
    /// Port-flow RMS at `f_target` in m³/s (per-unit-drive allowed).
    pub q_rms_m3s: f64,
    /// Profile curvature budget `∫(r'')² dx` in m⁻³.
    pub curvature_m3: f64,
    /// Maximum wall slope (dimensionless).
    pub max_slope: f64,
    /// Plane-wave margin `f_c/f` at `f_target` (validity audit).
    pub plane_wave_margin: f64,
    /// Analytic distortion term (toy evaluators only; `None` for [`solve_duct_1d`]).
    pub thd_proxy: Option<f64>,
    /// Analytic noise term (toy evaluators only; `None` for [`solve_duct_1d`]).
    pub noise_proxy: Option<f64>,
}

/// Tier 1 observables: Tier-0 tuning plus the viscothermal transfer.
#[derive(Debug, Clone)]
pub struct Tier1Obs {
    /// Compact Tier-0 tuning estimate of the same profile in Hz.
    pub f0_hz: f64,
    /// Port-flow magnitude at `f_target` in m³/s (unit-pressure drive; ratios only).
    pub q_rms_m3s: f64,
    /// Complex port impedance `[re, im]` in Pa·s/m³.
    pub z_port_pa_s_per_m3: [f64; 2],
    /// Complex box-to-port flow ratio `[re, im]`.
    pub flow_ratio: [f64; 2],
    /// True relative residual of the reference solve (audit).
    pub relative_residual: f64,
    /// Strong/weak exterior-flux relative difference (audit).
    pub flux_relative_difference: f64,
    /// Analytic distortion term (toy evaluators only; `None` for the FEM reference).
    pub thd_proxy: Option<f64>,
    /// Analytic noise term (toy evaluators only; `None` for the FEM reference).
    pub noise_proxy: Option<f64>,
}

/// Tier 0 evaluator: one profile in, screening observables out.
pub trait Tier0Evaluator: Send + Sync {
    /// Evaluate the profile; all returned scalars must be finite.
    fn eval_tier0(&self, profile: &SplinePortProfile) -> Result<Tier0Obs, PortFlowError>;
}

/// Tier 1 evaluator: one profile in, viscothermal observables out.
pub trait Tier1Evaluator: Send + Sync {
    /// Evaluate the profile; all returned scalars must be finite.
    fn eval_tier1(&self, profile: &SplinePortProfile) -> Result<Tier1Obs, PortFlowError>;
}

/// Tier 2 evaluator: one profile in, nonlinear metrics out.
pub trait Tier2Evaluator: Send + Sync {
    /// Evaluate the profile, returning the full Step-5 metric set.
    fn eval_tier2(
        &self,
        profile: &SplinePortProfile,
    ) -> Result<PortCandidateMetrics, PortFlowError>;
}

// ---------------------------------------------------------------------------
// Production evaluators
// ---------------------------------------------------------------------------

/// Tier 0 evaluator backed by the variable-area duct model.
#[derive(Debug, Clone, Copy)]
pub struct Duct1dTier0Evaluator {
    /// Acoustic medium (density, sound speed, viscosity, diffusivity).
    pub medium: AcousticMedium,
    /// Transfer-matrix configuration (segments, losses, box volume).
    pub duct: Duct1dConfig,
    /// Campaign tuning target in Hz (flow-evaluation frequency).
    pub f_target_hz: f64,
}

impl Tier0Evaluator for Duct1dTier0Evaluator {
    fn eval_tier0(&self, profile: &SplinePortProfile) -> Result<Tier0Obs, PortFlowError> {
        let f0_hz = tuning_frequency_hz(profile, self.medium, &self.duct)?;
        let point = solve_duct_1d(profile, self.medium, self.f_target_hz, &self.duct)?;
        Ok(Tier0Obs {
            f0_hz,
            q_rms_m3s: point.inlet_volume_velocity_rms_m3_per_s.norm(),
            curvature_m3: profile.curvature_integral(),
            max_slope: max_profile_slope(profile),
            plane_wave_margin: point.plane_wave_margin,
            thd_proxy: None,
            noise_proxy: None,
        })
    }
}

/// Tier 1 evaluator backed by the spline linear reference (Step 4 oracle).
///
/// Each evaluation rebuilds the 3D round-port mesh for the candidate profile
/// and runs the P2 viscothermal reference at `f_target`; tuning comes from
/// the embedded Tier-0 screen of the same profile.
pub struct SplineTier1Evaluator {
    /// Tier-0 screen reused for the tuning estimate.
    pub tier0: Duct1dTier0Evaluator,
    /// Axial length of each end-effect chamber in m.
    pub chamber_length_m: f64,
    /// Chamber radius in m (at least the bore radius).
    pub chamber_radius_m: f64,
    /// Axial extent of each smooth chamber-to-duct transition in m.
    pub flare_length_m: f64,
    /// Target mesh cell size in m.
    pub mesh_size_m: f64,
    /// Explicit circumferential sector count (`None` for automatic).
    pub angular_sectors: Option<usize>,
    /// Equilibrium thermophysical properties for the LRF constitutive law.
    pub fluid: FluidProperties,
    /// Lumped sealed-box volume in m³ (compliance transfer).
    pub box_volume_m3: f64,
    /// Campaign tuning target in Hz (reference frequency).
    pub f_target_hz: f64,
}

impl Tier1Evaluator for SplineTier1Evaluator {
    fn eval_tier1(&self, profile: &SplinePortProfile) -> Result<Tier1Obs, PortFlowError> {
        let f0_hz = tuning_frequency_hz(profile, self.tier0.medium, &self.tier0.duct)?;
        let geometry = SplineRoundPortGeometry {
            duct_length_m: profile.length_m(),
            bore_radius_m: profile.bore_radius_m(),
            chamber_length_m: self.chamber_length_m,
            chamber_radius_m: self.chamber_radius_m,
            flare_length_m: self.flare_length_m,
            mesh_size_m: self.mesh_size_m,
            angular_sectors: self.angular_sectors,
            profile: profile.clone(),
        };
        let mesh = build_spline_round_port_mesh(&geometry)?;
        let config = SplinePortLinearReferenceConfig {
            geometry,
            fluid: self.fluid.clone(),
            box_volume_m3: self.box_volume_m3,
        };
        let transfer = solve_spline_port_linear_reference(&mesh, &config, self.f_target_hz)?;
        Ok(Tier1Obs {
            f0_hz,
            q_rms_m3s: transfer.port_volume_velocity_rms_m3_per_s.norm(),
            z_port_pa_s_per_m3: [
                transfer.port_impedance_pa_s_per_m3.re,
                transfer.port_impedance_pa_s_per_m3.im,
            ],
            flow_ratio: [
                transfer.box_to_port_flow_ratio.re,
                transfer.box_to_port_flow_ratio.im,
            ],
            relative_residual: transfer.relative_residual,
            flux_relative_difference: transfer.strong_weak_exterior_flux_relative_difference,
            thd_proxy: None,
            noise_proxy: None,
        })
    }
}

/// Tier 2 evaluator backed by the Step-5 harness at a fixed drive/fidelity.
pub struct Tier2HarnessEvaluator {
    /// Frozen Tier-2 envelope with the shared baseline mesh.
    pub context: Tier2EvalContext,
    /// Terminal-voltage drive specification.
    pub drive: PortEvalDrive,
    /// Time resolution, run length, tail, and snapshot policy.
    pub fidelity: PortEvalFidelity,
}

impl Tier2HarnessEvaluator {
    /// Evaluate one profile, returning metrics plus the evidence meshes.
    pub fn evaluate_with_evidence(
        &self,
        profile: &SplinePortProfile,
    ) -> Result<Tier2EvalOutcome, PortFlowError> {
        self.context.evaluate(profile, &self.drive, &self.fidelity)
    }
}

impl Tier2Evaluator for Tier2HarnessEvaluator {
    fn eval_tier2(
        &self,
        profile: &SplinePortProfile,
    ) -> Result<PortCandidateMetrics, PortFlowError> {
        Ok(self.evaluate_with_evidence(profile)?.metrics)
    }
}

// ---------------------------------------------------------------------------
// Weights, feasibility, budgets
// ---------------------------------------------------------------------------

/// Scalarisation weights for `J = w_THD·THD + w_N·N + w_k·κ`.
///
/// Every term is normalised (distortion and noise by the straight-tube
/// baseline at the evaluating tier, curvature by the shape budget), so the
/// weights are scale-free trade-offs, not unit conversions.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScalarWeights {
    /// Weight on the normalised THD term.
    pub w_thd: f64,
    /// Weight on the normalised noise-proxy term.
    pub w_noise: f64,
    /// Weight on the normalised curvature regulariser.
    pub w_curvature: f64,
}

impl Default for ScalarWeights {
    /// Equal distortion/noise weight with a light curvature regulariser.
    fn default() -> Self {
        Self {
            w_thd: 1.0,
            w_noise: 1.0,
            w_curvature: 0.1,
        }
    }
}

impl ScalarWeights {
    /// Reject non-finite, negative, or all-zero weights.
    pub fn validate(&self) -> Result<(), PortFlowError> {
        for (name, value) in [
            ("w_thd", self.w_thd),
            ("w_noise", self.w_noise),
            ("w_curvature", self.w_curvature),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(PortFlowError::InvalidConfig(format!(
                    "port-optim weight {name} must be finite and non-negative, got {value}"
                )));
            }
        }
        if self.w_thd == 0.0 && self.w_noise == 0.0 && self.w_curvature == 0.0 {
            return Err(PortFlowError::InvalidConfig(
                "port-optim needs at least one positive scalarisation weight".to_string(),
            ));
        }
        Ok(())
    }
}

/// Feasibility envelope: hard constraints shared by every tier.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FeasibilitySpec {
    /// Campaign tuning target in Hz.
    pub f_target_hz: f64,
    /// Tuning tolerance in Hz (default ±2 Hz).
    pub tuning_tol_hz: f64,
    /// Minimum airflow ratio vs the straight tube at equal voltage.
    pub airflow_min_ratio: f64,
    /// Maximum port-output compression in dB (Tier 2 only).
    pub compression_max_db: f64,
}

impl FeasibilitySpec {
    /// Plan-default envelope around a campaign tuning target.
    pub fn port_optim_defaults(f_target_hz: f64) -> Self {
        Self {
            f_target_hz,
            tuning_tol_hz: DEFAULT_TUNING_TOLERANCE_HZ,
            airflow_min_ratio: DEFAULT_AIRFLOW_MIN_RATIO,
            compression_max_db: DEFAULT_COMPRESSION_MAX_DB,
        }
    }

    /// Reject non-finite or non-positive envelope values.
    pub fn validate(&self) -> Result<(), PortFlowError> {
        for (name, value) in [
            ("f_target_hz", self.f_target_hz),
            ("tuning_tol_hz", self.tuning_tol_hz),
            ("airflow_min_ratio", self.airflow_min_ratio),
            ("compression_max_db", self.compression_max_db),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(PortFlowError::InvalidConfig(format!(
                    "port-optim feasibility field {name} must be finite and positive, got {value}"
                )));
            }
        }
        Ok(())
    }
}

/// Maximum uncached solver evaluations per tier.
///
/// Rejected-by-construction candidates and cache hits never touch these
/// counters; failed solves (divergence, inverted mesh) consume budget because
/// the solver ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TierBudgets {
    /// Tier 0 screening budget.
    pub tier0: usize,
    /// Tier 1 re-rank budget.
    pub tier1: usize,
    /// Tier 2 quota budget (also drives [`EngineChoice::Auto`]).
    pub tier2: usize,
}

impl Default for TierBudgets {
    fn default() -> Self {
        Self {
            tier0: 10_000,
            tier1: 500,
            tier2: 200,
        }
    }
}

/// Consumed budget plus funnel accounting (cache hits, rejections, failures).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct BudgetUsage {
    /// Uncached Tier 0 evaluations performed.
    pub tier0_used: usize,
    /// Uncached Tier 1 evaluations performed.
    pub tier1_used: usize,
    /// Uncached Tier 2 evaluations performed.
    pub tier2_used: usize,
    /// Evaluations served from the candidate cache (budget-free).
    pub cache_hits: usize,
    /// Candidates rejected by construction (budget-free).
    pub rejected_by_construction: usize,
    /// Solver evaluations that ran and failed (budget-consuming).
    pub failed_evals: usize,
}

impl BudgetUsage {
    /// Budget consumed at one tier.
    pub fn used(&self, tier: EvalTier) -> usize {
        match tier {
            EvalTier::Tier0 => self.tier0_used,
            EvalTier::Tier1 => self.tier1_used,
            EvalTier::Tier2 => self.tier2_used,
        }
    }
}

impl TierBudgets {
    /// Budget limit at one tier.
    pub fn limit(&self, tier: EvalTier) -> usize {
        match tier {
            EvalTier::Tier0 => self.tier0,
            EvalTier::Tier1 => self.tier1,
            EvalTier::Tier2 => self.tier2,
        }
    }
}

// ---------------------------------------------------------------------------
// Constraints, candidates, outcomes
// ---------------------------------------------------------------------------

/// One funnel constraint value in the solver convention (`c >= 0` feasible).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstraintValue {
    /// Constraint name (see [`CONSTRAINT_NAMES`]).
    pub name: String,
    /// Constraint value (`>= 0` feasible). Unenforced entries carry `0.0` as
    /// an explicit placeholder, never a guessed value.
    pub value: f64,
    /// Whether this constraint gates feasibility at the evaluating tier.
    pub enforced: bool,
}

/// Fully reduced candidate observables at one tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluatedCandidate {
    /// Normalised free radii `u`.
    pub u: Vec<f64>,
    /// Tier that produced these observables.
    pub tier: EvalTier,
    /// Tuning estimate in Hz.
    pub f0_hz: f64,
    /// Port-flow RMS at `f_target` in m³/s.
    pub q_rms_m3s: f64,
    /// Straight-tube flow at the same tier, drive, and fidelity in m³/s.
    pub q_baseline_m3s: f64,
    /// THD fraction (`None` below Tier 2 for production evaluators).
    pub thd: Option<f64>,
    /// Noise proxy (`U⁸/L³` trend at Tier 2; `None` below Tier 2).
    pub noise_proxy: Option<f64>,
    /// Port-output compression in dB (`None` unless a reference was supplied).
    pub compression_db: Option<f64>,
    /// Peak Mach number, bulk velocity over throat (`None` below Tier 2).
    pub peak_mach: Option<f64>,
    /// Profile curvature budget `∫(r'')² dx` in m⁻³.
    pub curvature_m3: f64,
    /// Maximum wall slope (dimensionless).
    pub max_slope: f64,
    /// The five plan constraints in [`CONSTRAINT_NAMES`] order.
    pub constraints: Vec<ConstraintValue>,
    /// Whether every enforced constraint holds (`c >= 0`).
    pub feasible: bool,
    /// Largest enforced violation (`max(-c, 0)`; 0 when feasible).
    pub max_violation: f64,
    /// Scalarised objective `J` (baseline-normalised terms).
    pub objective_j: f64,
    /// Whether a positively weighted term was unavailable or unnormalised at
    /// this tier, so `J` must not be compared across tiers.
    pub objective_partial: bool,
    /// Whether this record was served from the candidate cache.
    pub cached: bool,
    /// Solver wall time in ms (0 for cache hits).
    pub wall_time_ms: f64,
}

/// Outcome of one funnel evaluation.
///
/// Only [`FunnelOutcome::Evaluated`] consumes solver budget. Rejections name
/// the construction check that failed; failures name the solver error. Both
/// are recorded in the history and are never reported feasible.
#[derive(Debug, Clone)]
pub enum FunnelOutcome {
    /// The candidate evaluated; see [`EvaluatedCandidate`].
    Evaluated(EvaluatedCandidate),
    /// Rejected by construction (illegal `u` or profile); budget-free.
    Rejected {
        /// Normalised variables as submitted.
        u: Vec<f64>,
        /// Tier the candidate was submitted to.
        tier: EvalTier,
        /// Construction check that failed.
        reason: String,
    },
    /// The solver ran and failed (divergence, inverted mesh, non-finite
    /// observables); budget-consuming.
    Failed {
        /// Normalised variables as submitted.
        u: Vec<f64>,
        /// Tier the candidate was submitted to.
        tier: EvalTier,
        /// Solver error that ended the run.
        error: String,
    },
}

impl FunnelOutcome {
    /// The evaluated candidate, if the run produced observables.
    pub fn candidate(&self) -> Option<&EvaluatedCandidate> {
        match self {
            FunnelOutcome::Evaluated(candidate) => Some(candidate),
            FunnelOutcome::Rejected { .. } | FunnelOutcome::Failed { .. } => None,
        }
    }

    /// Whether the outcome is an evaluated feasible candidate.
    pub fn is_feasible(&self) -> bool {
        self.candidate().is_some_and(|c| c.feasible)
    }
}

/// One ranked row of a funnel stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedCandidate {
    /// 1-based rank within the stage (feasible by `J`, then by violation,
    /// then failed, then rejected).
    pub rank: usize,
    /// Normalised free radii `u`.
    pub u: Vec<f64>,
    /// Tier that produced this row.
    pub tier: EvalTier,
    /// Outcome kind: `"evaluated"`, `"rejected"`, or `"failed"`.
    pub outcome: String,
    /// Whether the row is an evaluated feasible candidate.
    pub feasible: bool,
    /// Largest enforced violation (evaluated rows only; `None` unmeasured).
    pub max_violation: Option<f64>,
    /// Scalarised objective (evaluated rows only).
    pub objective_j: Option<f64>,
    /// Whether the objective is partial (Tier 0/1 ranking terms only).
    pub objective_partial: bool,
    /// Whether the row was served from the candidate cache.
    pub cached: bool,
    /// Rejection or failure reason (unmeasured rows only).
    pub reason: Option<String>,
}

/// One funnel stage: the ranked rows plus the budget-skipped remainder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageReport {
    /// Ranked candidates in rank order.
    pub ranking: Vec<RankedCandidate>,
    /// Candidates never submitted because the tier budget ran out.
    pub skipped_for_budget: usize,
}

/// Full funnel product: per-stage rankings, winners, and budget evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunnelReport {
    /// Tier 0 screen of the full candidate set.
    pub tier0: StageReport,
    /// Tier 1 re-rank of the Tier 0 survivors (`None` without a Tier 1 evaluator).
    pub tier1: Option<StageReport>,
    /// Tier 2 quota evaluation (`None` without a Tier 2 evaluator or quota).
    pub tier2: Option<StageReport>,
    /// Feasible rows of the deepest evaluated stage, in rank order.
    pub winners: Vec<RankedCandidate>,
    /// Budget consumption behind this report.
    pub usage: BudgetUsage,
}

// ---------------------------------------------------------------------------
// History (JSON)
// ---------------------------------------------------------------------------

/// Outcome kind recorded per history entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HistoryOutcomeKind {
    /// The candidate evaluated (feasibility is on the candidate record).
    Evaluated,
    /// Rejected by construction (budget-free).
    Rejected,
    /// The solver ran and failed (budget-consuming).
    Failed,
}

/// One recorded funnel evaluation: profile, tier, metrics, wall time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryRecord {
    /// 1-based sequence number within the funnel.
    pub seq: usize,
    /// Tier the candidate was submitted to.
    pub tier: EvalTier,
    /// Normalised variables as submitted.
    pub u: Vec<f64>,
    /// Outcome kind.
    pub outcome: HistoryOutcomeKind,
    /// Reduced observables (evaluated records only).
    pub candidate: Option<EvaluatedCandidate>,
    /// Rejection or failure reason (unmeasured records only).
    pub reason: Option<String>,
    /// Whether the record was served from the candidate cache.
    pub cached: bool,
    /// Solver wall time in ms (0 for cache hits and rejections).
    pub wall_time_ms: f64,
}

/// Complete reproducible evaluation history of one funnel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimHistory {
    /// Must equal [`PORT_OPTIM_HISTORY_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Model lineage; must equal [`PORT_OPTIM_HISTORY_PROVENANCE`].
    pub provenance: String,
    /// Scalarisation weights in force for the recorded `J` values.
    pub weights: ScalarWeights,
    /// Feasibility envelope in force for the recorded verdicts.
    pub feasibility: FeasibilitySpec,
    /// Tier budgets in force.
    pub budgets: TierBudgets,
    /// Every funnel evaluation in submission order, including cache hits,
    /// rejections, and failures.
    pub records: Vec<HistoryRecord>,
    /// Budget consumption after the last record.
    pub usage: BudgetUsage,
}

impl OptimHistory {
    /// Serialize the history to pretty JSON.
    pub fn to_json_string(&self) -> Result<String, PortOptimError> {
        serde_json::to_string_pretty(self)
            .map_err(|error| PortOptimError::History(error.to_string()))
    }

    /// Write the history to a JSON file.
    pub fn save_json(&self, path: &Path) -> Result<(), PortOptimError> {
        std::fs::write(path, self.to_json_string()?)
            .map_err(|error| PortOptimError::History(error.to_string()))
    }

    /// Parse history JSON, rejecting schema or provenance drift loudly.
    pub fn from_json_str(raw: &str) -> Result<Self, PortOptimError> {
        let history: OptimHistory = serde_json::from_str(raw)
            .map_err(|error| PortOptimError::History(error.to_string()))?;
        if history.schema_version != PORT_OPTIM_HISTORY_SCHEMA_VERSION {
            return Err(PortOptimError::History(format!(
                "unsupported port-optim history schema {}, expected {PORT_OPTIM_HISTORY_SCHEMA_VERSION}",
                history.schema_version
            )));
        }
        if history.provenance != PORT_OPTIM_HISTORY_PROVENANCE {
            return Err(PortOptimError::History(format!(
                "port-optim history provenance drift: {}",
                history.provenance
            )));
        }
        Ok(history)
    }
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Funnel and optimiser-driver failures.
#[derive(Debug, thiserror::Error)]
pub enum PortOptimError {
    /// A tier budget ran out; no further uncached evaluations at that tier.
    #[error("tier {tier:?} budget exhausted ({used}/{limit} evaluations consumed)")]
    BudgetExhausted {
        /// Tier whose budget ran out.
        tier: EvalTier,
        /// Evaluations consumed at that tier.
        used: usize,
        /// Budget limit at that tier.
        limit: usize,
    },
    /// Invalid funnel, search, or template configuration.
    #[error("invalid port-optim configuration: {0}")]
    InvalidConfig(String),
    /// A tier evaluator failed where the funnel needs a value (straight-tube
    /// baseline, template validation).
    #[error("port-optim evaluation failed: {0}")]
    Eval(#[from] PortFlowError),
    /// An optimiser driver (BOBYQA, cBO, NSGA-II) failed.
    #[error("port-optim driver failed: {0}")]
    Driver(String),
    /// History JSON serialisation, IO, or schema failure.
    #[error("port-optim history error: {0}")]
    History(String),
}

// ---------------------------------------------------------------------------
// Funnel
// ---------------------------------------------------------------------------

/// Static funnel configuration: template, weights, envelope, budgets, tiers.
#[derive(Debug, Clone)]
pub struct FunnelConfig {
    /// Fixed geometry behind the search variables.
    pub template: ProfileTemplate,
    /// Scalarisation weights for `J`.
    pub weights: ScalarWeights,
    /// Feasibility envelope shared by every tier.
    pub feasibility: FeasibilitySpec,
    /// Per-tier solver budgets.
    pub budgets: TierBudgets,
    /// Tier the optimiser drivers search at (cheap tiers for search,
    /// expensive tiers for quota verification).
    pub search_tier: EvalTier,
    /// Tier 2 evaluations admitted per funnel run (top-ranked first).
    pub tier2_quota: usize,
    /// Cache quantisation step in `u` units (see [`DEFAULT_CACHE_QUANTIZATION_STEP`]).
    pub cache_quantization_step: f64,
    /// Additive Pareto penalty per unit of constraint violation.
    pub pareto_penalty: f64,
    /// Max candidates solved concurrently inside one stage (`1` = serial).
    ///
    /// Values below 1 behave as 1. Parallel stages replay history records,
    /// cache inserts, and budget accounting in candidate order, so rankings
    /// and histories match the serial path (baseline records land before the
    /// stage records; per-record wall times still vary run to run).
    pub stage_parallelism: usize,
}

/// Straight-tube normalisation references at one tier.
#[derive(Debug, Clone, Copy)]
struct BaselineObs {
    /// Straight-tube flow in m³/s (airflow denominator).
    q_ref: f64,
    /// Straight-tube THD (objective denominator, when the tier measures it).
    thd_ref: Option<f64>,
    /// Straight-tube noise proxy (objective denominator, when measured).
    noise_ref: Option<f64>,
}

/// Cache key: tier plus quantised `u`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    tier: EvalTier,
    q: Vec<i64>,
}

/// Mutable funnel state behind the lock: cache, baselines, history.
struct FunnelState {
    cache: HashMap<CacheKey, EvaluatedCandidate>,
    baselines: HashMap<EvalTier, BaselineObs>,
    history: OptimHistory,
    /// Best scalarised objective seen at the search tier (neutral value
    /// returned by driver closures once the budget is exhausted).
    best_search_j: f64,
    /// Whether any driver closure hit budget exhaustion mid-run.
    budget_exhausted: bool,
}

/// One candidate's fate under the serial admission order.
#[derive(Debug, Clone)]
enum StageDisposition {
    /// Solve in a worker, then stamp/cache/record in replay order.
    Solve { key: CacheKey },
    /// Serve from a pre-existing (or earlier-admitted) cache entry.
    CacheHit { key: CacheKey },
    /// Straight-tube candidate served through the baseline insert.
    CacheHitThroughBaseline { key: CacheKey },
    /// Budget-free construction rejection with the serial message.
    Rejected { reason: String },
}

/// Serial-order admission plan for a parallel stage.
struct StageAdmissionPlan {
    /// Per-candidate dispositions in candidate order (a prefix when the
    /// budget runs out; the tail is skipped exactly like the serial loop).
    dispositions: Vec<StageDisposition>,
    /// Unpacked profiles to solve, in admission (= candidate) order.
    admitted: Vec<SplinePortProfile>,
    /// Whether any disposition needs the straight-tube baseline.
    needs_baseline: bool,
}

/// Baseline pre-resolution outcome for a parallel stage.
enum StageBaseline {
    /// Budget ran out before the baseline could resolve (state unchanged).
    Exhausted,
    /// The straight tube itself is broken (rejection or solver failure).
    Broken(String),
}

/// Tiered evaluation funnel with caching, budgets, and history.
///
/// All methods take `&self`; shared state lives behind a mutex so optimiser
/// closures (`Fn + Sync`) can evaluate through a shared reference.
pub struct PortOptimFunnel {
    config: FunnelConfig,
    tier0: Box<dyn Tier0Evaluator>,
    tier1: Option<Box<dyn Tier1Evaluator>>,
    tier2: Option<Box<dyn Tier2Evaluator>>,
    state: Mutex<FunnelState>,
}

impl PortOptimFunnel {
    /// Build a funnel, validating the template, weights, envelope, and tier
    /// availability up front (the straight profile must construct).
    pub fn new(
        config: FunnelConfig,
        tier0: Box<dyn Tier0Evaluator>,
        tier1: Option<Box<dyn Tier1Evaluator>>,
        tier2: Option<Box<dyn Tier2Evaluator>>,
    ) -> Result<Self, PortOptimError> {
        let mut config = config;
        config.stage_parallelism = config.stage_parallelism.max(1);
        config.template.straight().map_err(PortOptimError::Eval)?;
        config.weights.validate().map_err(PortOptimError::Eval)?;
        config
            .feasibility
            .validate()
            .map_err(PortOptimError::Eval)?;
        if !config.cache_quantization_step.is_finite() || config.cache_quantization_step <= 0.0 {
            return Err(PortOptimError::InvalidConfig(format!(
                "cache quantisation step must be finite and positive, got {}",
                config.cache_quantization_step
            )));
        }
        if !config.pareto_penalty.is_finite() || config.pareto_penalty < 0.0 {
            return Err(PortOptimError::InvalidConfig(format!(
                "Pareto penalty must be finite and non-negative, got {}",
                config.pareto_penalty
            )));
        }
        match config.search_tier {
            EvalTier::Tier0 => {}
            EvalTier::Tier1 if tier1.is_none() => {
                return Err(PortOptimError::InvalidConfig(
                    "search tier Tier1 needs a Tier 1 evaluator".to_string(),
                ));
            }
            EvalTier::Tier2 if tier2.is_none() => {
                return Err(PortOptimError::InvalidConfig(
                    "search tier Tier2 needs a Tier 2 evaluator".to_string(),
                ));
            }
            EvalTier::Tier1 | EvalTier::Tier2 => {}
        }
        let history = OptimHistory {
            schema_version: PORT_OPTIM_HISTORY_SCHEMA_VERSION,
            provenance: PORT_OPTIM_HISTORY_PROVENANCE.to_string(),
            weights: config.weights,
            feasibility: config.feasibility,
            budgets: config.budgets,
            records: Vec::new(),
            usage: BudgetUsage::default(),
        };
        Ok(Self {
            config,
            tier0,
            tier1,
            tier2,
            state: Mutex::new(FunnelState {
                cache: HashMap::new(),
                baselines: HashMap::new(),
                history,
                best_search_j: f64::INFINITY,
                budget_exhausted: false,
            }),
        })
    }

    /// The static funnel configuration.
    pub fn config(&self) -> &FunnelConfig {
        &self.config
    }

    /// Number of free search variables (`N - 3`).
    pub fn n_free(&self) -> usize {
        self.config.template.n_free()
    }

    /// Unit-box bounds for the free variables in solver types.
    pub fn bounds(&self) -> Vec<Bound<f64>> {
        free_variable_bounds(self.n_free())
    }

    /// Current budget consumption and funnel accounting.
    pub fn usage(&self) -> BudgetUsage {
        self.lock_state().history.usage
    }

    /// Whether any driver closure hit budget exhaustion mid-run.
    pub fn budget_exhausted(&self) -> bool {
        self.lock_state().budget_exhausted
    }

    /// Remaining uncached evaluations at one tier (saturating).
    pub fn budget_remaining(&self, tier: EvalTier) -> usize {
        let usage = self.usage();
        self.config
            .budgets
            .limit(tier)
            .saturating_sub(usage.used(tier))
    }

    /// Clone the complete evaluation history.
    pub fn history(&self) -> OptimHistory {
        self.lock_state().history.clone()
    }

    /// Serialize the complete evaluation history to pretty JSON.
    pub fn history_json(&self) -> Result<String, PortOptimError> {
        self.history().to_json_string()
    }

    /// Write the complete evaluation history to a JSON file.
    pub fn save_history_json(&self, path: &Path) -> Result<(), PortOptimError> {
        self.history().save_json(path)
    }

    /// Evaluate one candidate at one tier: the single funnel choke point.
    ///
    /// Applies the cache first (budget-free), then rejection-by-construction
    /// (budget-free), then the per-tier budget, then the tier evaluator.
    /// Every call appends one [`HistoryRecord`], including cache hits.
    pub fn evaluate(&self, tier: EvalTier, u: &[f64]) -> Result<FunnelOutcome, PortOptimError> {
        self.evaluate_inner(tier, u, false)
    }

    /// Evaluate with the baseline-recursion guard: the straight-tube baseline
    /// run skips baseline resolution and normalises against itself.
    fn evaluate_inner(
        &self,
        tier: EvalTier,
        u: &[f64],
        is_baseline: bool,
    ) -> Result<FunnelOutcome, PortOptimError> {
        if let Err(reason) = self.validate_u(u) {
            return self.record_rejection(tier, u, reason);
        }

        let key = self.quantize(u, tier);
        if let Some(outcome) = self.cache_hit(tier, u, &key) {
            return Ok(outcome);
        }

        let profile = match self.config.template.unpack(u) {
            Ok(profile) => profile,
            Err(error) => return self.record_rejection(tier, u, error.to_string()),
        };

        // Straight-tube normalisation references (cached after first use).
        let baseline = if is_baseline {
            None
        } else {
            Some(self.ensure_baseline(tier)?)
        };
        // The baseline run may have populated this exact entry (a straight
        // candidate shares the baseline key); re-check before spending budget.
        if !is_baseline && let Some(outcome) = self.cache_hit(tier, u, &key) {
            return Ok(outcome);
        }

        // Reserve budget before the solver runs; failed solves consume it.
        {
            let mut state = self.lock_state();
            let used = state.history.usage.used(tier);
            let limit = self.config.budgets.limit(tier);
            if used >= limit {
                return Err(PortOptimError::BudgetExhausted { tier, used, limit });
            }
            match tier {
                EvalTier::Tier0 => state.history.usage.tier0_used += 1,
                EvalTier::Tier1 => state.history.usage.tier1_used += 1,
                EvalTier::Tier2 => state.history.usage.tier2_used += 1,
            }
        }

        let started = Instant::now();
        let evaluated = self.run_tier(tier, &profile, baseline);
        let wall_time_ms = started.elapsed().as_secs_f64() * 1000.0;
        match evaluated {
            Ok(candidate) => Ok(self.record_evaluated(tier, u, key, candidate, wall_time_ms)),
            Err(error) => Ok(self.record_failed(tier, u, wall_time_ms, error.to_string())),
        }
    }

    /// Stamp, cache, and record one solved candidate (serial and parallel
    /// stages share this so their histories stay identical).
    fn record_evaluated(
        &self,
        tier: EvalTier,
        u: &[f64],
        key: CacheKey,
        mut candidate: EvaluatedCandidate,
        wall_time_ms: f64,
    ) -> FunnelOutcome {
        candidate.u = u.to_vec();
        candidate.tier = tier;
        candidate.cached = false;
        candidate.wall_time_ms = wall_time_ms;
        if tier == self.config.search_tier && candidate.objective_j.is_finite() {
            let mut state = self.lock_state();
            state.best_search_j = state.best_search_j.min(candidate.objective_j);
        }
        self.lock_state().cache.insert(key, candidate.clone());
        self.push_record(HistoryRecord {
            seq: 0,
            tier,
            u: u.to_vec(),
            outcome: HistoryOutcomeKind::Evaluated,
            candidate: Some(candidate.clone()),
            reason: None,
            cached: false,
            wall_time_ms,
        });
        FunnelOutcome::Evaluated(candidate)
    }

    /// Record one failed solve (budget was already consumed by reservation).
    fn record_failed(
        &self,
        tier: EvalTier,
        u: &[f64],
        wall_time_ms: f64,
        reason: String,
    ) -> FunnelOutcome {
        self.lock_state().history.usage.failed_evals += 1;
        self.push_record(HistoryRecord {
            seq: 0,
            tier,
            u: u.to_vec(),
            outcome: HistoryOutcomeKind::Failed,
            candidate: None,
            reason: Some(reason.clone()),
            cached: false,
            wall_time_ms,
        });
        FunnelOutcome::Failed {
            u: u.to_vec(),
            tier,
            error: reason,
        }
    }

    /// Validate search-box membership, returning the rejection reason.
    ///
    /// Shared by the serial path and the parallel admission simulation so
    /// both reject the same candidates with identical messages.
    fn validate_u(&self, u: &[f64]) -> Result<(), String> {
        let n_free = self.n_free();
        if u.len() != n_free {
            return Err(format!("expected {n_free} free variables, got {}", u.len()));
        }
        if let Some((index, value)) = u
            .iter()
            .enumerate()
            .find(|(_, v)| !v.is_finite() || **v < 0.0 || **v > 1.0)
        {
            return Err(format!(
                "u[{index}] = {value} outside the [0, 1] search box"
            ));
        }
        Ok(())
    }

    /// Serve a cache hit (budget-free), recording it in the history.
    fn cache_hit(&self, tier: EvalTier, u: &[f64], key: &CacheKey) -> Option<FunnelOutcome> {
        let mut hit = self.lock_state().cache.get(key).cloned()?;
        hit.cached = true;
        hit.wall_time_ms = 0.0;
        let outcome = FunnelOutcome::Evaluated(hit.clone());
        self.push_record(HistoryRecord {
            seq: 0,
            tier,
            u: u.to_vec(),
            outcome: HistoryOutcomeKind::Evaluated,
            candidate: Some(hit),
            reason: None,
            cached: true,
            wall_time_ms: 0.0,
        });
        self.lock_state().history.usage.cache_hits += 1;
        Some(outcome)
    }

    /// Record a budget-free rejection and return it as the outcome.
    fn record_rejection(
        &self,
        tier: EvalTier,
        u: &[f64],
        reason: String,
    ) -> Result<FunnelOutcome, PortOptimError> {
        self.lock_state().history.usage.rejected_by_construction += 1;
        self.push_record(HistoryRecord {
            seq: 0,
            tier,
            u: u.to_vec(),
            outcome: HistoryOutcomeKind::Rejected,
            candidate: None,
            reason: Some(reason.clone()),
            cached: false,
            wall_time_ms: 0.0,
        });
        Ok(FunnelOutcome::Rejected {
            u: u.to_vec(),
            tier,
            reason,
        })
    }

    /// Evaluate the straight tube at one tier once; cache the references.
    fn ensure_baseline(&self, tier: EvalTier) -> Result<BaselineObs, PortOptimError> {
        if let Some(baseline) = self.lock_state().baselines.get(&tier).copied() {
            return Ok(baseline);
        }
        let straight = vec![0.0; self.n_free()];
        let outcome = self.evaluate_inner(tier, &straight, true)?;
        let candidate = match outcome {
            FunnelOutcome::Evaluated(candidate) => candidate,
            FunnelOutcome::Rejected { reason, .. } => {
                let error = PortFlowError::InvalidConfig(format!(
                    "straight-tube baseline rejected at {tier:?}: {reason}"
                ));
                return Err(PortOptimError::Eval(error));
            }
            FunnelOutcome::Failed { error, .. } => {
                let error = PortFlowError::InvalidConfig(format!(
                    "straight-tube baseline failed at {tier:?}: {error}"
                ));
                return Err(PortOptimError::Eval(error));
            }
        };
        if !(candidate.q_rms_m3s.is_finite() && candidate.q_rms_m3s > 0.0) {
            let error = PortFlowError::InvalidConfig(format!(
                "straight-tube baseline flow at {tier:?} must be finite and positive, got {}",
                candidate.q_rms_m3s
            ));
            return Err(PortOptimError::Eval(error));
        }
        let baseline = BaselineObs {
            q_ref: candidate.q_rms_m3s,
            thd_ref: candidate.thd,
            noise_ref: candidate.noise_proxy,
        };
        self.lock_state().baselines.insert(tier, baseline);
        Ok(baseline)
    }

    /// Run the tier evaluator and reduce its observables to a candidate.
    fn run_tier(
        &self,
        tier: EvalTier,
        profile: &SplinePortProfile,
        baseline: Option<BaselineObs>,
    ) -> Result<EvaluatedCandidate, PortFlowError> {
        match tier {
            EvalTier::Tier0 => {
                let obs = self.tier0.eval_tier0(profile)?;
                self.reduce_candidate(
                    tier,
                    obs.f0_hz,
                    obs.q_rms_m3s,
                    obs.thd_proxy,
                    obs.noise_proxy,
                    None,
                    None,
                    obs.curvature_m3,
                    obs.max_slope,
                    baseline,
                )
            }
            EvalTier::Tier1 => {
                let tier1 = self.tier1.as_ref().ok_or_else(|| {
                    PortFlowError::InvalidConfig(
                        "Tier 1 evaluation needs a Tier 1 evaluator".to_string(),
                    )
                })?;
                let obs = tier1.eval_tier1(profile)?;
                self.reduce_candidate(
                    tier,
                    obs.f0_hz,
                    obs.q_rms_m3s,
                    obs.thd_proxy,
                    obs.noise_proxy,
                    None,
                    None,
                    profile.curvature_integral(),
                    max_profile_slope(profile),
                    baseline,
                )
            }
            EvalTier::Tier2 => {
                let tier2 = self.tier2.as_ref().ok_or_else(|| {
                    PortFlowError::InvalidConfig(
                        "Tier 2 evaluation needs a Tier 2 evaluator".to_string(),
                    )
                })?;
                let metrics = tier2.eval_tier2(profile)?;
                self.reduce_candidate(
                    tier,
                    metrics.f0_estimate_hz,
                    metrics.port_flow_rms_m3s,
                    Some(metrics.thd_fraction),
                    Some(metrics.turbulence_power_trend),
                    metrics.compression_db,
                    Some(metrics.peak_mach_bulk),
                    metrics.curvature_integral_m3,
                    max_profile_slope(profile),
                    baseline,
                )
            }
        }
    }

    /// Reduce tier observables to constraints, feasibility, and `J`.
    #[allow(clippy::too_many_arguments)]
    fn reduce_candidate(
        &self,
        tier: EvalTier,
        f0_hz: f64,
        q_rms_m3s: f64,
        thd: Option<f64>,
        noise_proxy: Option<f64>,
        compression_db: Option<f64>,
        peak_mach: Option<f64>,
        curvature_m3: f64,
        max_slope: f64,
        baseline: Option<BaselineObs>,
    ) -> Result<EvaluatedCandidate, PortFlowError> {
        for (name, value) in [
            ("f0_hz", f0_hz),
            ("q_rms_m3s", q_rms_m3s),
            ("curvature_m3", curvature_m3),
            ("max_slope", max_slope),
        ] {
            if !value.is_finite() {
                return Err(PortFlowError::InvalidConfig(format!(
                    "tier {tier:?} observable {name} must be finite, got {value}"
                )));
            }
        }
        for (name, value) in [
            ("thd", thd),
            ("noise_proxy", noise_proxy),
            ("compression_db", compression_db),
            ("peak_mach", peak_mach),
        ] {
            if let Some(v) = value
                && !v.is_finite()
            {
                return Err(PortFlowError::InvalidConfig(format!(
                    "tier {tier:?} observable {name} must be finite, got {v}"
                )));
            }
        }
        if q_rms_m3s < 0.0 {
            return Err(PortFlowError::InvalidConfig(format!(
                "tier {tier:?} flow RMS must be non-negative, got {q_rms_m3s}"
            )));
        }
        // The baseline run normalises against itself; every other run uses
        // the cached straight-tube references.
        let refs = baseline.unwrap_or(BaselineObs {
            q_ref: q_rms_m3s,
            thd_ref: thd,
            noise_ref: noise_proxy,
        });
        if !(refs.q_ref.is_finite() && refs.q_ref > 0.0) {
            return Err(PortFlowError::InvalidConfig(format!(
                "tier {tier:?} airflow baseline must be finite and positive, got {}",
                refs.q_ref
            )));
        }
        let spec = &self.config.feasibility;
        let limits = self.config.template.options.limits;
        let c_airflow = q_rms_m3s / refs.q_ref - spec.airflow_min_ratio;
        let c_tuning = 1.0 - (f0_hz - spec.f_target_hz).abs() / spec.tuning_tol_hz;
        let (c_compression, compression_enforced) = match compression_db {
            Some(db) => (1.0 - db / spec.compression_max_db, true),
            None => (0.0, false),
        };
        let c_slope = 1.0 - max_slope / limits.slope_max;
        let c_curvature = 1.0 - curvature_m3 / limits.curvature_budget_m3;
        let constraints = vec![
            ConstraintValue {
                name: CONSTRAINT_NAMES[0].to_string(),
                value: c_airflow,
                enforced: true,
            },
            ConstraintValue {
                name: CONSTRAINT_NAMES[1].to_string(),
                value: c_tuning,
                enforced: true,
            },
            ConstraintValue {
                name: CONSTRAINT_NAMES[2].to_string(),
                value: c_compression,
                enforced: compression_enforced,
            },
            ConstraintValue {
                name: CONSTRAINT_NAMES[3].to_string(),
                value: c_slope,
                enforced: true,
            },
            ConstraintValue {
                name: CONSTRAINT_NAMES[4].to_string(),
                value: c_curvature,
                enforced: true,
            },
        ];
        let mut feasible = true;
        let mut max_violation = 0.0f64;
        for c in &constraints {
            if c.enforced {
                if c.value < 0.0 {
                    feasible = false;
                }
                max_violation = max_violation.max(-c.value).max(0.0);
            }
        }
        let (objective_j, objective_partial) = self.scalar_objective(
            thd,
            refs.thd_ref,
            noise_proxy,
            refs.noise_ref,
            curvature_m3,
            limits.curvature_budget_m3,
        );
        Ok(EvaluatedCandidate {
            u: Vec::new(),
            tier,
            f0_hz,
            q_rms_m3s,
            q_baseline_m3s: refs.q_ref,
            thd,
            noise_proxy,
            compression_db,
            peak_mach,
            curvature_m3,
            max_slope,
            constraints,
            feasible,
            max_violation,
            objective_j,
            objective_partial,
            cached: false,
            wall_time_ms: 0.0,
        })
    }

    /// Scalarised objective with baseline-normalised terms.
    fn scalar_objective(
        &self,
        thd: Option<f64>,
        thd_ref: Option<f64>,
        noise: Option<f64>,
        noise_ref: Option<f64>,
        curvature_m3: f64,
        curvature_budget_m3: f64,
    ) -> (f64, bool) {
        let weights = &self.config.weights;
        let mut j = 0.0;
        let mut partial = false;
        for (weight, value, reference) in [
            (weights.w_thd, thd, thd_ref),
            (weights.w_noise, noise, noise_ref),
        ] {
            if weight == 0.0 {
                continue;
            }
            match (value, reference) {
                (Some(v), Some(r)) if r > 0.0 => j += weight * v / r,
                (Some(v), _) => {
                    // Unnormalised fallback (straight baseline missed the
                    // term); usable but flagged, never silently rescaled.
                    j += weight * v;
                    partial = true;
                }
                (None, _) => partial = true,
            }
        }
        if weights.w_curvature != 0.0 {
            j += weights.w_curvature * curvature_m3 / curvature_budget_m3;
        }
        (j, partial)
    }

    /// Quantise `u` to the cache grid at one tier.
    fn quantize(&self, u: &[f64], tier: EvalTier) -> CacheKey {
        let step = self.config.cache_quantization_step;
        CacheKey {
            tier,
            q: u.iter().map(|v| (v / step).round() as i64).collect(),
        }
    }

    /// Append a history record with the next sequence number.
    fn push_record(&self, mut record: HistoryRecord) {
        let mut state = self.lock_state();
        record.seq = state.history.records.len() + 1;
        state.history.records.push(record);
    }

    /// Lock the funnel state, recovering from a poisoned mutex.
    fn lock_state(&self) -> MutexGuard<'_, FunnelState> {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    /// Names of the enforced search constraints in evaluation order.
    ///
    /// Compression joins the list only when the search tier is Tier 2; below
    /// Tier 2 it is unverifiable and checked at the quota stage instead.
    pub fn search_constraint_names(&self) -> Vec<String> {
        let compression_enforced = self.config.search_tier == EvalTier::Tier2;
        CONSTRAINT_NAMES
            .iter()
            .filter(|name| **name != "compression" || compression_enforced)
            .map(|name| name.to_string())
            .collect()
    }

    /// Evaluate one search point: enforced constraints plus the objective.
    ///
    /// Returns `(constraints, j, feasible)`. Rejected or failed points report
    /// `J = REJECTED_OBJECTIVE` with every constraint at `-1` (infeasible by
    /// construction, never silently feasible). Budget exhaustion flags the
    /// funnel and returns the best `J` seen so far with the constraints of the
    /// best cached point, stalling the driver honestly instead of erroring
    /// through an infallible closure.
    pub fn search_point(&self, u: &[f64]) -> (Vec<f64>, f64, bool) {
        let names = self.search_constraint_names();
        match self.evaluate(self.config.search_tier, u) {
            Ok(FunnelOutcome::Evaluated(candidate)) => {
                let values = names
                    .iter()
                    .map(|name| {
                        candidate
                            .constraints
                            .iter()
                            .find(|c| &c.name == name)
                            .map_or(-1.0, |c| c.value)
                    })
                    .collect();
                (values, candidate.objective_j, candidate.feasible)
            }
            Ok(FunnelOutcome::Rejected { .. } | FunnelOutcome::Failed { .. }) => {
                (vec![-1.0; names.len()], REJECTED_OBJECTIVE, false)
            }
            Err(PortOptimError::BudgetExhausted { .. }) => {
                let state = self.lock_state();
                let neutral = if state.best_search_j.is_finite() {
                    state.best_search_j
                } else {
                    REJECTED_OBJECTIVE
                };
                drop(state);
                self.lock_state().budget_exhausted = true;
                (vec![-1.0; names.len()], neutral, false)
            }
            Err(_) => (vec![-1.0; names.len()], REJECTED_OBJECTIVE, false),
        }
    }

    /// Search constraints as solver types over a shared funnel.
    ///
    /// Each constraint is one inequality output in
    /// [`PortOptimFunnel::search_constraint_names`] order, evaluated through
    /// the funnel (cache, budget, history) at the search tier. Unknown points
    /// (rejected, failed, budget-exhausted) evaluate to `-1`: infeasible
    /// unless proven otherwise.
    pub fn search_constraints(self: &Arc<Self>) -> Vec<Constraint<f64>> {
        self.search_constraint_names()
            .into_iter()
            .enumerate()
            .map(|(index, _)| {
                let funnel = Arc::clone(self);
                Constraint {
                    kind: ConstraintKind::Inequality,
                    dim: 1,
                    fun: Box::new(move |u: &[f64]| vec![funnel.search_point(u).0[index]]),
                    jac: None,
                }
            })
            .collect()
    }

    /// Augmented-Lagrangian merit at one search point (BOBYQA inner objective).
    ///
    /// `merit = J + Σ μ/2·shifted²` over violated shifted constraints
    /// (`shifted = c - λ/μ < 0`), matching the bundled AL inequality handling
    /// but minimised derivative-free.
    pub fn al_merit(&self, u: &[f64], lambda: &[f64], mu: f64) -> f64 {
        let (constraints, j, _) = self.search_point(u);
        let mut merit = j;
        for (c, lambda_i) in constraints.iter().zip(lambda.iter()) {
            let shifted = c - lambda_i / mu;
            if shifted < 0.0 {
                merit += 0.5 * mu * shifted * shifted;
            }
        }
        merit
    }

    /// Distortion-plus-noise Pareto point at one search point.
    ///
    /// Returns `(distortion_noise, airflow_ratio, feasible, max_violation)`.
    /// The distortion axis carries the positively weighted THD/noise terms
    /// (missing terms contribute 0); the airflow axis is the straight-tube
    /// ratio (maximised, hence minimised negated by the driver).
    pub fn pareto_point(&self, u: &[f64]) -> (f64, f64, bool, f64) {
        match self.evaluate(self.config.search_tier, u) {
            Ok(FunnelOutcome::Evaluated(candidate)) => {
                let weights = &self.config.weights;
                let mut distortion = 0.0;
                if weights.w_thd != 0.0
                    && let (Some(v), Some(r)) = (
                        candidate.thd,
                        self.baseline_term(self.config.search_tier, true),
                    )
                    && r > 0.0
                {
                    distortion += weights.w_thd * v / r;
                }
                if weights.w_noise != 0.0
                    && let (Some(v), Some(r)) = (
                        candidate.noise_proxy,
                        self.baseline_term(self.config.search_tier, false),
                    )
                    && r > 0.0
                {
                    distortion += weights.w_noise * v / r;
                }
                let airflow = candidate.q_rms_m3s / candidate.q_baseline_m3s;
                (
                    distortion,
                    airflow,
                    candidate.feasible,
                    candidate.max_violation,
                )
            }
            Ok(FunnelOutcome::Rejected { .. } | FunnelOutcome::Failed { .. }) => {
                (REJECTED_OBJECTIVE, 0.0, false, 1.0)
            }
            Err(_) => {
                self.lock_state().budget_exhausted = true;
                (REJECTED_OBJECTIVE, 0.0, false, 1.0)
            }
        }
    }

    /// Cached straight-tube THD (`thd = true`) or noise reference at one tier.
    fn baseline_term(&self, tier: EvalTier, thd: bool) -> Option<f64> {
        self.lock_state()
            .baselines
            .get(&tier)
            .copied()
            .and_then(|b| if thd { b.thd_ref } else { b.noise_ref })
    }

    /// Check a driver start point before an optimiser sees it.
    fn check_search_start(&self, x0: &[f64]) -> Result<(), PortOptimError> {
        let n_free = self.n_free();
        if n_free == 0 {
            return Err(PortOptimError::InvalidConfig(
                "optimisation needs at least one free variable (n_control_points >= 4)".to_string(),
            ));
        }
        if x0.len() != n_free {
            return Err(PortOptimError::InvalidConfig(format!(
                "start point has {} variables, expected {n_free}",
                x0.len()
            )));
        }
        if let Some((index, value)) = x0
            .iter()
            .enumerate()
            .find(|(_, v)| !v.is_finite() || **v < 0.0 || **v > 1.0)
        {
            return Err(PortOptimError::InvalidConfig(format!(
                "start point x0[{index}] = {value} outside the [0, 1] search box"
            )));
        }
        Ok(())
    }

    /// Best feasible search-tier candidate seen so far (cache scan).
    fn best_feasible_cached(&self) -> Option<EvaluatedCandidate> {
        self.lock_state()
            .cache
            .values()
            .filter(|c| c.tier == self.config.search_tier && c.feasible)
            .min_by(|a, b| {
                a.objective_j
                    .partial_cmp(&b.objective_j)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
    }
}

// ---------------------------------------------------------------------------
// Scalarised drivers
// ---------------------------------------------------------------------------

/// Scalarised-search engine selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EngineChoice {
    /// BOBYQA plus augmented Lagrangian, or cBO when the Tier 2 budget is
    /// below [`CBO_TIER2_BUDGET_THRESHOLD`] (plan D7).
    #[default]
    Auto,
    /// Augmented-Lagrangian outer loop with BOBYQA inner solves.
    AugmentedLagrangianBobyqa,
    /// Constrained Bayesian optimisation (small evaluation budgets).
    ConstrainedBayesOpt,
}

/// Engine that ran a scalarised search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EngineUsed {
    /// Augmented-Lagrangian outer loop with BOBYQA inner solves.
    AugmentedLagrangianBobyqa,
    /// Constrained Bayesian optimisation.
    ConstrainedBayesOpt,
}

/// Augmented-Lagrangian plus BOBYQA search configuration.
#[derive(Debug, Clone)]
pub struct AlBobyqaConfig {
    /// Outer multiplier-update iterations.
    pub max_outer_iterations: usize,
    /// BOBYQA evaluations per outer iteration (capped by remaining budget).
    pub max_inner_evals: usize,
    /// Initial BOBYQA trust-region radius in `u` units.
    pub rhobeg: f64,
    /// Final BOBYQA trust-region radius (inner convergence).
    pub rhoend: f64,
    /// Constraint tolerance for the stall/early-exit test.
    pub constraint_tolerance: f64,
    /// Initial penalty parameter.
    pub initial_penalty: f64,
    /// Penalty growth factor when the violation stalls.
    pub penalty_factor: f64,
}

impl Default for AlBobyqaConfig {
    fn default() -> Self {
        Self {
            max_outer_iterations: 12,
            max_inner_evals: 150,
            rhobeg: 0.2,
            rhoend: 1.0e-4,
            constraint_tolerance: 1.0e-3,
            initial_penalty: 10.0,
            penalty_factor: 5.0,
        }
    }
}

/// Constrained-Bayesian-optimisation search configuration.
#[derive(Debug, Clone)]
pub struct CboSearchConfig {
    /// Maximum objective evaluations (capped by remaining budget).
    pub maxeval: usize,
    /// Initial design samples (`0` uses the crate default).
    pub initial_samples: usize,
    /// RNG seed for deterministic runs.
    pub seed: Option<u64>,
}

impl Default for CboSearchConfig {
    fn default() -> Self {
        Self {
            maxeval: 60,
            initial_samples: 0,
            seed: Some(0),
        }
    }
}

/// Scalarised-search configuration: engine plus per-engine settings.
#[derive(Debug, Clone, Default)]
pub struct ScalarSearchConfig {
    /// Engine selection (resolves [`EngineChoice::Auto`] by budget).
    pub engine: EngineChoice,
    /// BOBYQA plus augmented-Lagrangian settings.
    pub al_bobyqa: AlBobyqaConfig,
    /// Constrained-Bayesian-optimisation settings.
    pub cbo: CboSearchConfig,
}

/// Scalarised-search product: best feasible point plus run evidence.
#[derive(Debug, Clone)]
pub struct ScalarizedResult {
    /// Engine that ran the search.
    pub engine: EngineUsed,
    /// Best feasible point (`None` when no evaluated point is feasible).
    pub best_u: Option<Vec<f64>>,
    /// Scalarised objective at the best feasible point.
    pub best_j: Option<f64>,
    /// Whether a feasible point was found (infeasible points are never
    /// returned as feasible).
    pub feasible: bool,
    /// History records appended by this run.
    pub evaluations: usize,
    /// Whether the search stalled on budget exhaustion.
    pub budget_exhausted: bool,
    /// Human-readable termination summary.
    pub message: String,
}

impl PortOptimFunnel {
    /// Resolve [`EngineChoice::Auto`] by the Tier 2 budget rule.
    pub fn resolve_engine(&self, engine: EngineChoice) -> EngineUsed {
        match engine {
            EngineChoice::AugmentedLagrangianBobyqa => EngineUsed::AugmentedLagrangianBobyqa,
            EngineChoice::ConstrainedBayesOpt => EngineUsed::ConstrainedBayesOpt,
            EngineChoice::Auto if self.config.budgets.tier2 < CBO_TIER2_BUDGET_THRESHOLD => {
                EngineUsed::ConstrainedBayesOpt
            }
            EngineChoice::Auto => EngineUsed::AugmentedLagrangianBobyqa,
        }
    }

    /// Run a scalarised search with the configured engine.
    ///
    /// Takes a shared funnel because the cBO path erases its constraint
    /// closures into `'static` Arcs; the AL-BOBYQA path borrows through it.
    pub fn optimize_scalarized(
        self: &Arc<Self>,
        x0: &[f64],
        config: &ScalarSearchConfig,
    ) -> Result<ScalarizedResult, PortOptimError> {
        match self.resolve_engine(config.engine) {
            EngineUsed::AugmentedLagrangianBobyqa => self.optimize_al_bobyqa(x0, &config.al_bobyqa),
            EngineUsed::ConstrainedBayesOpt => self.optimize_cbo(x0, &config.cbo),
        }
    }

    /// Scalarised search via an augmented-Lagrangian outer loop with BOBYQA
    /// inner solves (the default engine).
    ///
    /// Multipliers update as `λ <- max(0, λ - μ·c)` per outer iteration and
    /// the penalty grows while the violation stalls. The reported best is
    /// the minimum-`J` feasible point seen; when nothing is feasible the
    /// result carries `feasible = false` and no best point.
    pub fn optimize_al_bobyqa(
        &self,
        x0: &[f64],
        config: &AlBobyqaConfig,
    ) -> Result<ScalarizedResult, PortOptimError> {
        self.check_search_start(x0)?;
        if config.max_outer_iterations == 0 || config.max_inner_evals == 0 {
            return Err(PortOptimError::InvalidConfig(
                "AL-BOBYQA needs at least one outer iteration and inner evaluation".to_string(),
            ));
        }
        for (name, value) in [
            ("rhobeg", config.rhobeg),
            ("rhoend", config.rhoend),
            ("constraint_tolerance", config.constraint_tolerance),
            ("initial_penalty", config.initial_penalty),
            ("penalty_factor", config.penalty_factor),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(PortOptimError::InvalidConfig(format!(
                    "AL-BOBYQA field {name} must be finite and positive, got {value}"
                )));
            }
        }
        let history_before = self.lock_state().history.records.len();
        let n_constraints = self.search_constraint_names().len();
        let mut x = x0.to_vec();
        let mut lambda = vec![0.0; n_constraints];
        let mut mu = config.initial_penalty;
        let mut prev_violation = f64::INFINITY;
        let mut best_j = f64::INFINITY;
        let mut stall = 0usize;
        let mut message = String::from("outer iterations exhausted");

        // Seed the baseline references (and the best-J tracker) before the
        // first inner solve so merit values are normalised from the start.
        let _ = self.search_point(&x);

        for _outer in 0..config.max_outer_iterations {
            if self.budget_exhausted() {
                message = String::from("search-tier budget exhausted");
                break;
            }
            let remaining = self.budget_remaining(self.config.search_tier).max(1);
            let inner_evals = config.max_inner_evals.min(remaining);
            let merit = |point: &[f64]| self.al_merit(point, &lambda, mu);
            let bobyqa_config = BobyqaConfig {
                bounds: self.bounds(),
                npt: None,
                rhobeg: Some(config.rhobeg),
                rhoend: config.rhoend,
                max_fun_evals: inner_evals,
                max_iterations: inner_evals,
                gradient_tolerance: 1.0e-8,
            };
            match bobyqa(&merit, &x, &bobyqa_config) {
                Ok(result) => x = result.x,
                Err(error) => {
                    message = format!("BOBYQA inner solve failed: {error}");
                    break;
                }
            }
            let (constraints, _, _) = self.search_point(&x);
            let violation = constraints
                .iter()
                .map(|c| (-c).max(0.0))
                .fold(0.0f64, f64::max);
            if violation > 0.25 * prev_violation {
                mu *= config.penalty_factor;
            }
            prev_violation = violation;
            for (lambda_i, c) in lambda.iter_mut().zip(constraints.iter()) {
                *lambda_i = (*lambda_i - mu * c).max(0.0);
            }
            if let Some(best) = self.best_feasible_cached()
                && best.objective_j < best_j - 1.0e-12 * best_j.abs().max(1.0)
            {
                best_j = best.objective_j;
                stall = 0;
            } else {
                stall += 1;
            }
            // Feasible, tight, and stalled: further outer iterations only
            // burn budget on an already converged point.
            if violation <= config.constraint_tolerance && best_j.is_finite() && stall >= 3 {
                message = String::from("feasible and stalled (converged)");
                break;
            }
        }

        let best = self.best_feasible_cached();
        let evaluations = self.lock_state().history.records.len() - history_before;
        let budget_exhausted = self.budget_exhausted();
        match best {
            Some(candidate) => Ok(ScalarizedResult {
                engine: EngineUsed::AugmentedLagrangianBobyqa,
                best_u: Some(candidate.u),
                best_j: Some(candidate.objective_j),
                feasible: true,
                evaluations,
                budget_exhausted,
                message,
            }),
            None => Ok(ScalarizedResult {
                engine: EngineUsed::AugmentedLagrangianBobyqa,
                best_u: None,
                best_j: None,
                feasible: false,
                evaluations,
                budget_exhausted,
                message: format!("no feasible point found ({message})"),
            }),
        }
    }

    /// Scalarised search via constrained Bayesian optimisation (small budgets).
    ///
    /// Funnel constraints negate into the `g(x) <= 0` cBO convention. The
    /// reported best is re-verified through the funnel: the cBO incumbent is
    /// only returned when the funnel agrees it is feasible.
    pub fn optimize_cbo(
        self: &Arc<Self>,
        x0: &[f64],
        config: &CboSearchConfig,
    ) -> Result<ScalarizedResult, PortOptimError> {
        self.check_search_start(x0)?;
        if config.maxeval == 0 {
            return Err(PortOptimError::InvalidConfig(
                "cBO needs at least one evaluation".to_string(),
            ));
        }
        let history_before = self.lock_state().history.records.len();
        let remaining = self.budget_remaining(self.config.search_tier).max(1);
        let maxeval = config.maxeval.min(remaining);
        let objective_funnel = Arc::clone(self);
        let objective = move |point: &[f64]| objective_funnel.search_point(point).1;
        let n_constraints = self.search_constraint_names().len();
        let mut constraints = Vec::with_capacity(n_constraints);
        for index in 0..n_constraints {
            let funnel = Arc::clone(self);
            let fun: BayesOptConstraintFn =
                Arc::new(move |point: &[f64]| -funnel.search_point(point).0[index]);
            constraints.push(BayesOptConstraint { fun });
        }
        let cbo_config = ConstrainedBayesOptConfig {
            bounds: vec![(0.0, 1.0); self.n_free()],
            x0: Some(x0.to_vec()),
            initial_samples: config.initial_samples,
            maxeval,
            seed: config.seed,
            // No final COBYLA polish: every true-function evaluation must
            // stay visible in the funnel budget and history.
            polish_fraction: 0.0,
            ..ConstrainedBayesOptConfig::default()
        };
        let report = constrained_bayesian_optimization(&objective, &constraints, cbo_config)
            .map_err(|error| PortOptimError::Driver(error.to_string()))?;
        // Re-verify the incumbent through the funnel (cache hit, budget-free)
        // and prefer the best feasible point actually seen.
        let _ = self.search_point(&report.x);
        let best = self.best_feasible_cached();
        let evaluations = self.lock_state().history.records.len() - history_before;
        let budget_exhausted = self.budget_exhausted();
        match best {
            Some(candidate) => Ok(ScalarizedResult {
                engine: EngineUsed::ConstrainedBayesOpt,
                best_u: Some(candidate.u),
                best_j: Some(candidate.objective_j),
                feasible: true,
                evaluations,
                budget_exhausted,
                message: report.message,
            }),
            None => Ok(ScalarizedResult {
                engine: EngineUsed::ConstrainedBayesOpt,
                best_u: None,
                best_j: None,
                feasible: false,
                evaluations,
                budget_exhausted,
                message: format!("no feasible point found ({})", report.message),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Pareto driver (NSGA-II)
// ---------------------------------------------------------------------------

/// Pareto-search configuration: NSGA-II settings at the search tier.
#[derive(Debug, Clone)]
pub struct ParetoSearchConfig {
    /// Maximum objective evaluations (capped by remaining budget).
    pub maxeval: usize,
    /// Population size (`0` uses the crate default).
    pub population_size: usize,
    /// Optional initial individual (clipped to the unit box).
    pub x0: Option<Vec<f64>>,
    /// RNG seed for deterministic runs.
    pub seed: Option<u64>,
}

impl Default for ParetoSearchConfig {
    fn default() -> Self {
        Self {
            maxeval: 500,
            population_size: 0,
            x0: None,
            seed: Some(0),
        }
    }
}

/// One Pareto-front member with funnel-verified feasibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoCandidate {
    /// Normalised free radii `u`.
    pub u: Vec<f64>,
    /// Distortion-plus-noise axis (weighted THD/noise ratios, minimised).
    pub distortion_noise: f64,
    /// Airflow ratio vs the straight tube (maximised).
    pub airflow_ratio: f64,
    /// Whether every enforced constraint holds.
    pub feasible: bool,
    /// Largest enforced violation (0 when feasible).
    pub max_violation: f64,
}

/// Pareto-search product: feasible front plus infeasible evidence.
#[derive(Debug, Clone)]
pub struct ParetoResult {
    /// Feasible non-dominated front, sorted by distortion ascending.
    pub front: Vec<ParetoCandidate>,
    /// Penalised-front members the funnel verifies as infeasible (evidence
    /// that the search explored the boundary, never reported as winners).
    pub infeasible_front: Vec<ParetoCandidate>,
    /// History records appended by this run.
    pub evaluations: usize,
    /// Whether the search stalled on budget exhaustion.
    pub budget_exhausted: bool,
    /// Human-readable termination summary.
    pub message: String,
}

impl PortOptimFunnel {
    /// Pareto search via NSGA-II: the (distortion-plus-noise, airflow) front.
    ///
    /// Both minimised objectives carry the additive penalty
    /// `pareto_penalty · max_violation`, steering the population toward the
    /// feasible region; the reported front is re-verified through the funnel
    /// and split into feasible winners and infeasible evidence.
    pub fn optimize_pareto(
        &self,
        config: &ParetoSearchConfig,
    ) -> Result<ParetoResult, PortOptimError> {
        if self.n_free() == 0 {
            return Err(PortOptimError::InvalidConfig(
                "optimisation needs at least one free variable (n_control_points >= 4)".to_string(),
            ));
        }
        if config.maxeval == 0 {
            return Err(PortOptimError::InvalidConfig(
                "NSGA-II needs at least one evaluation".to_string(),
            ));
        }
        if let Some(x0) = &config.x0 {
            self.check_search_start(x0)?;
        }
        let history_before = self.lock_state().history.records.len();
        let remaining = self.budget_remaining(self.config.search_tier).max(1);
        let maxeval = config.maxeval.min(remaining);
        let penalty = self.config.pareto_penalty;
        let objectives = |point: &[f64]| {
            let (distortion, airflow, _, violation) = self.pareto_point(point);
            let finite_distortion = distortion.is_finite();
            let finite_airflow = airflow.is_finite();
            if !finite_distortion || !finite_airflow {
                return vec![REJECTED_OBJECTIVE, REJECTED_OBJECTIVE];
            }
            vec![
                distortion + penalty * violation,
                -airflow + penalty * violation,
            ]
        };
        let nsga_config = NsgaConfig {
            bounds: vec![(0.0, 1.0); self.n_free()],
            x0: config.x0.clone(),
            population_size: config.population_size,
            maxeval,
            crossover_prob: 0.9,
            mutation_prob: None,
            eta_c: 15.0,
            eta_m: 20.0,
            variant: NsgaVariant::Nsga2,
            reference_partitions: 0,
            seed: config.seed,
        };
        let report = nsga2(&objectives, nsga_config)
            .map_err(|error| PortOptimError::Driver(error.to_string()))?;
        let mut front = Vec::new();
        let mut infeasible_front = Vec::new();
        for member in &report.pareto_front {
            // Re-verify through the funnel (cache hit, budget-free).
            let (distortion, airflow, feasible, violation) = self.pareto_point(&member.x);
            let candidate = ParetoCandidate {
                u: member.x.clone(),
                distortion_noise: distortion,
                airflow_ratio: airflow,
                feasible,
                max_violation: violation,
            };
            if feasible {
                front.push(candidate);
            } else {
                infeasible_front.push(candidate);
            }
        }
        front.sort_by(|a, b| {
            a.distortion_noise
                .partial_cmp(&b.distortion_noise)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let evaluations = self.lock_state().history.records.len() - history_before;
        let message = if front.is_empty() {
            format!(
                "no feasible front member ({} infeasible, {})",
                infeasible_front.len(),
                report.message
            )
        } else {
            report.message.clone()
        };
        Ok(ParetoResult {
            front,
            infeasible_front,
            evaluations,
            budget_exhausted: self.budget_exhausted(),
            message,
        })
    }
}

// ---------------------------------------------------------------------------
// Funnel stages
// ---------------------------------------------------------------------------

impl PortOptimFunnel {
    /// Rank one stage's outcomes: feasible by `J`, then by violation, then
    /// failed, then rejected. Ranks are 1-based in output order.
    fn rank_outcomes(
        tier: EvalTier,
        outcomes: Vec<(Vec<f64>, FunnelOutcome)>,
    ) -> Vec<RankedCandidate> {
        // (class, violation, j): class 0 feasible, 1 infeasible, 2 failed, 3 rejected.
        let mut rows: Vec<RankedCandidate> = outcomes
            .into_iter()
            .map(|(u, outcome)| match outcome {
                FunnelOutcome::Evaluated(candidate) => RankedCandidate {
                    rank: 0,
                    u,
                    tier,
                    outcome: "evaluated".to_string(),
                    feasible: candidate.feasible,
                    max_violation: Some(candidate.max_violation),
                    objective_j: Some(candidate.objective_j),
                    objective_partial: candidate.objective_partial,
                    cached: candidate.cached,
                    reason: None,
                },
                FunnelOutcome::Rejected { reason, .. } => RankedCandidate {
                    rank: 0,
                    u,
                    tier,
                    outcome: "rejected".to_string(),
                    feasible: false,
                    max_violation: None,
                    objective_j: None,
                    objective_partial: false,
                    cached: false,
                    reason: Some(reason),
                },
                FunnelOutcome::Failed { error, .. } => RankedCandidate {
                    rank: 0,
                    u,
                    tier,
                    outcome: "failed".to_string(),
                    feasible: false,
                    max_violation: None,
                    objective_j: None,
                    objective_partial: false,
                    cached: false,
                    reason: Some(error),
                },
            })
            .collect();
        rows.sort_by(|a, b| {
            let class = |row: &RankedCandidate| match row.outcome.as_str() {
                "evaluated" if row.feasible => 0,
                "evaluated" => 1,
                "failed" => 2,
                _ => 3,
            };
            class(a)
                .cmp(&class(b))
                .then(
                    a.max_violation
                        .partial_cmp(&b.max_violation)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
                .then(
                    a.objective_j
                        .partial_cmp(&b.objective_j)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        });
        for (index, row) in rows.iter_mut().enumerate() {
            row.rank = index + 1;
        }
        rows
    }

    /// Run one stage over explicit candidates, stopping at budget exhaustion.
    fn run_stage(&self, tier: EvalTier, candidates: &[Vec<f64>]) -> StageReport {
        if self.config.stage_parallelism <= 1 || candidates.len() <= 1 {
            return self.run_stage_serial(tier, candidates);
        }
        self.run_stage_parallel(tier, candidates)
    }

    /// Legacy serial stage loop (also the parallel-path fallback).
    fn run_stage_serial(&self, tier: EvalTier, candidates: &[Vec<f64>]) -> StageReport {
        let mut outcomes = Vec::with_capacity(candidates.len());
        let mut skipped_for_budget = 0usize;
        for u in candidates {
            match self.evaluate(tier, u) {
                Ok(outcome) => outcomes.push((u.clone(), outcome)),
                Err(PortOptimError::BudgetExhausted { .. }) => {
                    skipped_for_budget = candidates.len() - outcomes.len();
                    break;
                }
                Err(error) => outcomes.push((
                    u.clone(),
                    FunnelOutcome::Failed {
                        u: u.clone(),
                        tier,
                        error: error.to_string(),
                    },
                )),
            }
        }
        StageReport {
            ranking: Self::rank_outcomes(tier, outcomes),
            skipped_for_budget,
        }
    }

    /// Parallel stage: simulate serial admission, solve the admitted set in
    /// bounded parallel chunks, then replay cache inserts, history records,
    /// and budget accounting in candidate order.
    ///
    /// Rankings match the serial path exactly; histories match except that
    /// baseline records land before the stage records and per-record wall
    /// times vary run to run (as they do between any two serial runs).
    fn run_stage_parallel(&self, tier: EvalTier, candidates: &[Vec<f64>]) -> StageReport {
        let plan = self.simulate_stage_admission(tier, candidates);
        // Resolve the straight-tube baseline up front (serially, exactly
        // once) so workers can reduce without shared state.
        let baseline = match self.ensure_baseline_for_plan(tier, &plan) {
            Ok(baseline) => baseline,
            Err(StageBaseline::Exhausted) => {
                // Reservation-free failure: no state changed, exact fallback.
                return self.run_stage_serial(tier, candidates);
            }
            Err(StageBaseline::Broken(reason)) => {
                return self.replay_broken_baseline(tier, candidates, &plan, reason);
            }
        };
        // Atomically reserve the admitted budget; a concurrent stage that
        // moved the counters since the simulation forces exact fallback.
        {
            let mut state = self.lock_state();
            let used = state.history.usage.used(tier);
            let limit = self.config.budgets.limit(tier);
            if used + plan.admitted.len() > limit {
                drop(state);
                return self.run_stage_serial(tier, candidates);
            }
            match tier {
                EvalTier::Tier0 => state.history.usage.tier0_used += plan.admitted.len(),
                EvalTier::Tier1 => state.history.usage.tier1_used += plan.admitted.len(),
                EvalTier::Tier2 => state.history.usage.tier2_used += plan.admitted.len(),
            }
        }
        // Bounded in-flight solves: each chunk runs on the shared rayon pool
        // (nested FEM assembly work-steals with the candidates), and results
        // collect in admission order. Without needy candidates there is
        // nothing to solve and the replay below only serves hits/rejections.
        let mut solved: Vec<(f64, Result<EvaluatedCandidate, PortFlowError>)> =
            Vec::with_capacity(plan.admitted.len());
        if let Some(baseline) = baseline {
            let width = self.config.stage_parallelism.max(1);
            for chunk in plan.admitted.chunks(width) {
                let mut partial: Vec<(f64, Result<EvaluatedCandidate, PortFlowError>)> = chunk
                    .par_iter()
                    .map(|profile: &SplinePortProfile| {
                        let started = Instant::now();
                        let outcome = self.run_tier(tier, profile, Some(baseline));
                        (started.elapsed().as_secs_f64() * 1000.0, outcome)
                    })
                    .collect();
                solved.append(&mut partial);
            }
        }
        // Ordered replay: identical records to the serial loop.
        let mut outcomes = Vec::with_capacity(plan.dispositions.len());
        let mut solved_iter = solved.into_iter();
        for (index, disposition) in plan.dispositions.iter().enumerate() {
            let u = &candidates[index];
            match disposition {
                StageDisposition::Solve { key } => {
                    let (wall_time_ms, outcome) = solved_iter
                        .next()
                        .expect("one worker result per admitted candidate");
                    match outcome {
                        Ok(candidate) => outcomes.push((
                            u.clone(),
                            self.record_evaluated(tier, u, key.clone(), candidate, wall_time_ms),
                        )),
                        Err(error) => outcomes.push((
                            u.clone(),
                            self.record_failed(tier, u, wall_time_ms, error.to_string()),
                        )),
                    }
                }
                StageDisposition::CacheHit { key }
                | StageDisposition::CacheHitThroughBaseline { key } => {
                    let outcome = self
                        .cache_hit(tier, u, key)
                        .expect("simulated hit must be served from cache");
                    outcomes.push((u.clone(), outcome));
                }
                StageDisposition::Rejected { reason } => {
                    let outcome = self
                        .record_rejection(tier, u, reason.clone())
                        .expect("rejection recording cannot fail");
                    outcomes.push((u.clone(), outcome));
                }
            }
        }
        StageReport {
            ranking: Self::rank_outcomes(tier, outcomes),
            skipped_for_budget: candidates.len() - plan.dispositions.len(),
        }
    }

    /// Replay a stage whose straight-tube baseline is broken: rejections and
    /// pre-existing cache hits are served exactly as the serial loop would;
    /// every needy candidate fails with the identical baseline error and
    /// burns one budget unit, matching serial budget accounting.
    ///
    /// The serial loop would additionally re-run (and re-record) the failing
    /// straight-tube solve per needy candidate; the parallel path records the
    /// single real attempt. A broken straight tube means the campaign has no
    /// normalisation reference, so the stage is degenerate either way.
    fn replay_broken_baseline(
        &self,
        tier: EvalTier,
        candidates: &[Vec<f64>],
        plan: &StageAdmissionPlan,
        reason: String,
    ) -> StageReport {
        let mut outcomes = Vec::with_capacity(plan.dispositions.len());
        // The failed pre-resolve already consumed one unit; the remaining
        // needy candidates each burn one more, exactly like serial retries.
        let mut extra_budget = plan
            .dispositions
            .iter()
            .filter(|d| matches!(d, StageDisposition::Solve { .. }))
            .count()
            .saturating_sub(1);
        // The straight-tube candidate (a simulated hit through the baseline
        // insert) also fails: serially its baseline resolution fails first.
        extra_budget += plan
            .dispositions
            .iter()
            .filter(|d| matches!(d, StageDisposition::CacheHitThroughBaseline { .. }))
            .count();
        {
            let mut state = self.lock_state();
            match tier {
                EvalTier::Tier0 => state.history.usage.tier0_used += extra_budget,
                EvalTier::Tier1 => state.history.usage.tier1_used += extra_budget,
                EvalTier::Tier2 => state.history.usage.tier2_used += extra_budget,
            }
        }
        for (index, disposition) in plan.dispositions.iter().enumerate() {
            let u = &candidates[index];
            match disposition {
                StageDisposition::Solve { .. }
                | StageDisposition::CacheHitThroughBaseline { .. } => {
                    outcomes.push((
                        u.clone(),
                        FunnelOutcome::Failed {
                            u: u.clone(),
                            tier,
                            error: reason.clone(),
                        },
                    ));
                }
                StageDisposition::CacheHit { key } => {
                    let outcome = self
                        .cache_hit(tier, u, key)
                        .expect("simulated hit must be served from cache");
                    outcomes.push((u.clone(), outcome));
                }
                StageDisposition::Rejected { reason } => {
                    let outcome = self
                        .record_rejection(tier, u, reason.clone())
                        .expect("rejection recording cannot fail");
                    outcomes.push((u.clone(), outcome));
                }
            }
        }
        StageReport {
            ranking: Self::rank_outcomes(tier, outcomes),
            skipped_for_budget: candidates.len() - plan.dispositions.len(),
        }
    }

    /// Pure serial-order admission simulation: decide each candidate's fate
    /// (solve / cache hit / rejection / budget skip) from one state snapshot,
    /// without running any solves.
    ///
    /// The walk mirrors [`Self::evaluate_inner`] exactly — validate, cache,
    /// unpack, baseline, re-check, reserve — so the parallel replay admits
    /// precisely the candidates the serial loop would submit.
    fn simulate_stage_admission(
        &self,
        tier: EvalTier,
        candidates: &[Vec<f64>],
    ) -> StageAdmissionPlan {
        let straight = vec![0.0; self.n_free()];
        let straight_key = self.quantize(&straight, tier);
        let (mut simulated_cache, baseline_cached, mut simulated_used) = {
            let state = self.lock_state();
            (
                state.cache.keys().cloned().collect::<HashSet<_>>(),
                state.baselines.contains_key(&tier),
                state.history.usage.used(tier),
            )
        };
        let limit = self.config.budgets.limit(tier);
        let mut simulated_baseline = baseline_cached;
        let mut dispositions = Vec::with_capacity(candidates.len());
        let mut admitted = Vec::new();
        for u in candidates {
            if let Err(reason) = self.validate_u(u) {
                dispositions.push(StageDisposition::Rejected { reason });
                continue;
            }
            let key = self.quantize(u, tier);
            if simulated_cache.contains(&key) {
                dispositions.push(StageDisposition::CacheHit { key });
                continue;
            }
            let profile = match self.config.template.unpack(u) {
                Ok(profile) => profile,
                Err(error) => {
                    dispositions.push(StageDisposition::Rejected {
                        reason: error.to_string(),
                    });
                    continue;
                }
            };
            if !simulated_baseline {
                if simulated_used >= limit {
                    break;
                }
                simulated_used += 1;
                simulated_baseline = true;
                simulated_cache.insert(straight_key.clone());
                if key == straight_key {
                    dispositions.push(StageDisposition::CacheHitThroughBaseline { key });
                    continue;
                }
            }
            if simulated_used >= limit {
                break;
            }
            simulated_used += 1;
            simulated_cache.insert(key.clone());
            admitted.push(profile);
            dispositions.push(StageDisposition::Solve { key });
        }
        let needs_baseline = dispositions.iter().any(|disposition| {
            matches!(
                disposition,
                StageDisposition::Solve { .. } | StageDisposition::CacheHitThroughBaseline { .. }
            )
        });
        StageAdmissionPlan {
            dispositions,
            admitted,
            needs_baseline,
        }
    }

    /// Resolve the straight-tube baseline for an admission plan (exactly once,
    /// serially), unless no disposition needs it.
    fn ensure_baseline_for_plan(
        &self,
        tier: EvalTier,
        plan: &StageAdmissionPlan,
    ) -> Result<Option<BaselineObs>, StageBaseline> {
        if !plan.needs_baseline {
            return Ok(None);
        }
        match self.ensure_baseline(tier) {
            Ok(baseline) => Ok(Some(baseline)),
            Err(PortOptimError::BudgetExhausted { .. }) => Err(StageBaseline::Exhausted),
            Err(error) => Err(StageBaseline::Broken(error.to_string())),
        }
    }

    /// Tier 0 screen of the full candidate set.
    pub fn screen_tier0(&self, candidates: &[Vec<f64>]) -> StageReport {
        self.run_stage(EvalTier::Tier0, candidates)
    }

    /// Tier 1 re-rank of explicit candidates (usually the Tier 0 survivors).
    pub fn rerank_tier1(&self, candidates: &[Vec<f64>]) -> Result<StageReport, PortOptimError> {
        if self.tier1.is_none() {
            return Err(PortOptimError::InvalidConfig(
                "Tier 1 re-rank needs a Tier 1 evaluator".to_string(),
            ));
        }
        Ok(self.run_stage(EvalTier::Tier1, candidates))
    }

    /// Tier 2 evaluation of the top-`tier2_quota` ranked candidates.
    pub fn evaluate_tier2_quota(
        &self,
        ranked: &[RankedCandidate],
    ) -> Result<StageReport, PortOptimError> {
        if self.tier2.is_none() {
            return Err(PortOptimError::InvalidConfig(
                "Tier 2 quota evaluation needs a Tier 2 evaluator".to_string(),
            ));
        }
        let admitted: Vec<Vec<f64>> = ranked
            .iter()
            .filter(|row| row.outcome == "evaluated")
            .take(self.config.tier2_quota)
            .map(|row| row.u.clone())
            .collect();
        Ok(self.run_stage(EvalTier::Tier2, &admitted))
    }

    /// Run the full funnel: Tier 0 screen, Tier 1 re-rank of the survivors,
    /// Tier 2 quota on the top-ranked.
    ///
    /// Only evaluated rows promote: Tier 1 sees the Tier 0 feasible rows in
    /// rank order, and Tier 2 sees the top-`tier2_quota` evaluated rows of
    /// the deepest available ranking. Empty survivor sets propagate as empty
    /// (but present) stages with a complete history, never as invented rows.
    pub fn run_funnel(&self, initial: &[Vec<f64>]) -> Result<FunnelReport, PortOptimError> {
        let tier0 = self.screen_tier0(initial);
        let survivors0: Vec<Vec<f64>> = tier0
            .ranking
            .iter()
            .filter(|row| row.feasible)
            .map(|row| row.u.clone())
            .collect();
        let tier1 = match &self.tier1 {
            Some(_) => Some(self.rerank_tier1(&survivors0)?),
            None => None,
        };
        let base_ranking = tier1
            .as_ref()
            .map_or(&tier0.ranking, |stage| &stage.ranking);
        let tier2 = match &self.tier2 {
            Some(_) if self.config.tier2_quota > 0 => {
                Some(self.evaluate_tier2_quota(base_ranking)?)
            }
            _ => None,
        };
        let deepest = tier2.as_ref().map_or(base_ranking, |stage| &stage.ranking);
        let winners: Vec<RankedCandidate> =
            deepest.iter().filter(|row| row.feasible).cloned().collect();
        Ok(FunnelReport {
            tier0,
            tier1,
            tier2,
            winners,
            usage: self.usage(),
        })
    }
}

// ---------------------------------------------------------------------------
// Reference-case helpers
// ---------------------------------------------------------------------------

/// Build the campaign-scale profile template from the Step-0 reference case
/// (`R0`, `L`, `r_min = 0.6·R0`).
pub fn campaign_template(
    case: &PortOptimReferenceCase,
    n_control_points: usize,
    options: ProfileOptions,
) -> ProfileTemplate {
    ProfileTemplate {
        n_control_points,
        length_m: case.port_length_l_m,
        bore_radius_m: case.port_radius_r0_m,
        throat_radius_m: 0.6 * case.port_radius_r0_m,
        options,
    }
}

/// Build the Tier-2 slot-scale profile template from the Step-0 reference
/// case (`bore = tier2_duct_height/2`, `r_min = 0.6·bore`).
pub fn tier2_slot_template(
    case: &PortOptimReferenceCase,
    n_control_points: usize,
    options: ProfileOptions,
) -> ProfileTemplate {
    ProfileTemplate {
        n_control_points,
        length_m: case.tier2_duct_length_m,
        bore_radius_m: 0.5 * case.tier2_duct_height_m,
        throat_radius_m: 0.3 * case.tier2_duct_height_m,
        options,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::super::baseline::PortOptimReferenceCase;
    use super::super::duct1d::DUCT1D_DEFAULT_SEGMENTS;
    use super::super::evaluate::{
        PortEvalHealth, TIER2_EVAL_PROVENANCE, TIER2_EVAL_SCHEMA_VERSION,
    };
    use super::super::profile::{ProfileSplineKind, ShapeLimits};
    use super::*;

    const L: f64 = 0.1;
    const R0: f64 = 0.025;
    const RMIN: f64 = 0.015;
    const F_TARGET: f64 = 60.0;
    /// Known toy optimum: THD/noise proxies minimise here.
    const TOY_OPTIMUM: [f64; 2] = [0.3, 0.6];

    fn relaxed_options() -> ProfileOptions {
        ProfileOptions {
            limits: ShapeLimits {
                slope_max: 1.0,
                curvature_budget_m3: 1.0e4,
            },
            kind: ProfileSplineKind::ShapePreserving,
            ..ProfileOptions::default()
        }
    }

    fn toy_template() -> ProfileTemplate {
        ProfileTemplate {
            n_control_points: 5,
            length_m: L,
            bore_radius_m: R0,
            throat_radius_m: RMIN,
            options: relaxed_options(),
        }
    }

    fn toy_config() -> FunnelConfig {
        FunnelConfig {
            template: toy_template(),
            weights: ScalarWeights::default(),
            feasibility: FeasibilitySpec::port_optim_defaults(F_TARGET),
            budgets: TierBudgets::default(),
            search_tier: EvalTier::Tier0,
            tier2_quota: 2,
            cache_quantization_step: DEFAULT_CACHE_QUANTIZATION_STEP,
            pareto_penalty: PARETO_VIOLATION_PENALTY,
            stage_parallelism: 1,
        }
    }

    fn bowl(u: &[f64]) -> f64 {
        (u[0] - TOY_OPTIMUM[0]).powi(2) + (u[1] - TOY_OPTIMUM[1]).powi(2)
    }

    /// Analytic Tier 0 stand-in with a known optimum and a mild flow loss.
    struct ToyTier0 {
        calls: AtomicUsize,
        /// Constant tuning offset in Hz (tests the all-infeasible funnel).
        tuning_offset_hz: f64,
    }

    impl ToyTier0 {
        fn maps(u: &[f64], tuning_offset_hz: f64) -> (f64, f64, f64, f64) {
            let flow_loss = 0.02 * (u[0] + u[1]);
            let q = 1.0e-4 * (1.0 - flow_loss);
            let f0 = F_TARGET - 1.5 * (u[0] + u[1]) + tuning_offset_hz;
            let thd = 0.02 * bowl(u) + 0.001;
            let noise = 0.5 * bowl(u) + 0.05;
            (q, f0, thd, noise)
        }
    }

    impl Tier0Evaluator for ToyTier0 {
        fn eval_tier0(&self, profile: &SplinePortProfile) -> Result<Tier0Obs, PortFlowError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let u = pack_u(profile);
            let (q, f0, thd, noise) = Self::maps(&u, self.tuning_offset_hz);
            Ok(Tier0Obs {
                f0_hz: f0,
                q_rms_m3s: q,
                curvature_m3: profile.curvature_integral(),
                max_slope: max_profile_slope(profile),
                plane_wave_margin: 25.0,
                thd_proxy: Some(thd),
                noise_proxy: Some(noise),
            })
        }
    }

    /// Analytic Tier 1 stand-in: Tier 0 maps with a small systematic shift.
    struct ToyTier1 {
        calls: AtomicUsize,
    }

    impl Tier1Evaluator for ToyTier1 {
        fn eval_tier1(&self, profile: &SplinePortProfile) -> Result<Tier1Obs, PortFlowError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let u = pack_u(profile);
            let (q, f0, thd, noise) = ToyTier0::maps(&u, 0.0);
            Ok(Tier1Obs {
                f0_hz: f0 + 0.1 * (u[0] - u[1]),
                q_rms_m3s: q * 0.99,
                z_port_pa_s_per_m3: [100.0, 500.0],
                flow_ratio: [0.9, 0.1],
                relative_residual: 1.0e-9,
                flux_relative_difference: 1.0e-3,
                thd_proxy: Some(1.1 * thd),
                noise_proxy: Some(1.1 * noise),
            })
        }
    }

    /// Analytic Tier 2 stand-in returning a full metric set, including a
    /// compression map that fails deep dips (> 1 dB past u-sum 1.8).
    struct ToyTier2 {
        calls: AtomicUsize,
    }

    impl Tier2Evaluator for ToyTier2 {
        fn eval_tier2(
            &self,
            profile: &SplinePortProfile,
        ) -> Result<PortCandidateMetrics, PortFlowError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let u = pack_u(profile);
            let (q, f0, thd, noise) = ToyTier0::maps(&u, 0.0);
            Ok(PortCandidateMetrics {
                schema_version: TIER2_EVAL_SCHEMA_VERSION,
                provenance: TIER2_EVAL_PROVENANCE.to_string(),
                profile_normalized: u.clone(),
                n_control_points: 5,
                voltage_peak_v: 0.1,
                frequency_hz: F_TARGET,
                second_tone_frequency_hz: None,
                dt_s: 1.0 / (F_TARGET * 48.0),
                steps_completed: 480,
                tail_samples: 192,
                n_tail_snapshots: 0,
                port_flow_rms_m3s: q,
                port_flow_fundamental_amplitude_m3s: q,
                box_pressure_amplitude_pa: 1.0,
                peak_excursion_m: 1.0e-4,
                f0_estimate_hz: f0,
                thd_fraction: thd,
                imd_fraction: None,
                compression_db: Some(0.1 + 0.5 * (u[0] + u[1])),
                peak_mach_bulk: 0.01,
                peak_mach_field: None,
                separation_fraction: None,
                turbulence_power_trend: noise,
                curvature_integral_m3: profile.curvature_integral(),
                energy_physical_relative_closure: 1.0e-3,
                energy_algebraic_relative_closure: 1.0e-12,
                energy_interface_relative_residual: 1.0e-12,
                energy_scale_j: 1.0e-6,
                health: PortEvalHealth {
                    dissipations_nonnegative: true,
                    max_accepted_linear_relative_residual: 1.0e-12,
                    linear_residual_ok: true,
                    cap_continuity_relative_error: 1.0e-9,
                    box_conservation_relative_residual: 1.0e-12,
                    passed: true,
                },
            })
        }
    }

    struct ToyFunnel {
        funnel: Arc<PortOptimFunnel>,
        tier0_calls: Arc<ToyTier0>,
        tier1_calls: Arc<ToyTier1>,
        tier2_calls: Arc<ToyTier2>,
    }

    // The toy evaluators must be shared with the funnel for call counting,
    // but the funnel takes `Box<dyn _>`; hold them in an `Arc` each and move
    // a second `Arc` (as the trait object) into the funnel.
    impl ToyTier0 {
        fn shared(offset: f64) -> (Arc<Self>, Box<dyn Tier0Evaluator>) {
            let owned = Arc::new(Self {
                calls: AtomicUsize::new(0),
                tuning_offset_hz: offset,
            });
            (Arc::clone(&owned), Box::new(SharedToyTier0(owned)))
        }
    }

    struct SharedToyTier0(Arc<ToyTier0>);

    impl Tier0Evaluator for SharedToyTier0 {
        fn eval_tier0(&self, profile: &SplinePortProfile) -> Result<Tier0Obs, PortFlowError> {
            self.0.eval_tier0(profile)
        }
    }

    fn toy_funnel(tuning_offset_hz: f64) -> ToyFunnel {
        toy_funnel_with_config(tuning_offset_hz, toy_config(), false)
    }

    /// Toy funnel with an explicit config; `failing` swaps in a Tier 0
    /// evaluator that errors for `u[0] > 0.9` (failure-path coverage).
    fn toy_funnel_with_config(
        tuning_offset_hz: f64,
        config: FunnelConfig,
        failing: bool,
    ) -> ToyFunnel {
        let (tier0_calls, tier0) = ToyTier0::shared(tuning_offset_hz);
        let tier0: Box<dyn Tier0Evaluator> = if failing {
            Box::new(FailingToyTier0(Arc::clone(&tier0_calls)))
        } else {
            tier0
        };
        let tier1_calls = Arc::new(ToyTier1 {
            calls: AtomicUsize::new(0),
        });
        let tier2_calls = Arc::new(ToyTier2 {
            calls: AtomicUsize::new(0),
        });
        struct T1(Arc<ToyTier1>);
        impl Tier1Evaluator for T1 {
            fn eval_tier1(&self, p: &SplinePortProfile) -> Result<Tier1Obs, PortFlowError> {
                self.0.eval_tier1(p)
            }
        }
        struct T2(Arc<ToyTier2>);
        impl Tier2Evaluator for T2 {
            fn eval_tier2(
                &self,
                p: &SplinePortProfile,
            ) -> Result<PortCandidateMetrics, PortFlowError> {
                self.0.eval_tier2(p)
            }
        }
        let funnel = PortOptimFunnel::new(
            config,
            tier0,
            Some(Box::new(T1(Arc::clone(&tier1_calls)))),
            Some(Box::new(T2(Arc::clone(&tier2_calls)))),
        )
        .expect("toy funnel must build");
        ToyFunnel {
            funnel: Arc::new(funnel),
            tier0_calls,
            tier1_calls,
            tier2_calls,
        }
    }

    /// Tier 0 evaluator that fails deterministically for `u[0] > 0.9`,
    /// covering the solver-failure replay path.
    struct FailingToyTier0(Arc<ToyTier0>);

    impl Tier0Evaluator for FailingToyTier0 {
        fn eval_tier0(&self, profile: &SplinePortProfile) -> Result<Tier0Obs, PortFlowError> {
            let u = pack_u(profile);
            if u[0] > 0.9 {
                return Err(PortFlowError::InvalidConfig(format!(
                    "toy failure for u[0] = {}",
                    u[0]
                )));
            }
            self.0.eval_tier0(profile)
        }
    }

    /// Candidate list exercising every disposition: solves, duplicates,
    /// a mid-list straight tube, rejections, and a solver failure.
    fn stage_equivalence_candidates() -> Vec<Vec<f64>> {
        vec![
            vec![0.25, 0.75],
            vec![0.25, 0.75],
            vec![0.5, 0.5],
            vec![0.0, 0.0],
            vec![1.5, 0.5],
            vec![0.1],
            vec![0.95, 0.05],
            vec![0.3, 0.6],
            vec![0.7, 0.2],
            vec![0.5, 0.5],
        ]
    }

    /// History records with run-dependent fields stripped, as a sorted
    /// multiset (parallel replay resolves the baseline before the stage, so
    /// baseline records land earlier than in the serial loop).
    fn normalized_records(history: &OptimHistory) -> Vec<String> {
        let mut rows: Vec<String> = history
            .records
            .iter()
            .map(|record| {
                let mut flat = record.clone();
                flat.seq = 0;
                flat.wall_time_ms = 0.0;
                if let Some(candidate) = flat.candidate.as_mut() {
                    candidate.wall_time_ms = 0.0;
                }
                format!("{flat:?}")
            })
            .collect();
        rows.sort();
        rows
    }

    fn assert_stage_equivalent(serial: &ToyFunnel, parallel: &ToyFunnel) {
        let serial_history = serial.funnel.history();
        let parallel_history = parallel.funnel.history();
        assert_eq!(
            serial_history.usage, parallel_history.usage,
            "budget accounting must match"
        );
        assert_eq!(
            normalized_records(&serial_history),
            normalized_records(&parallel_history),
            "history records must match up to order and wall time"
        );
        assert_eq!(
            serial.tier0_calls.calls.load(Ordering::SeqCst),
            parallel.tier0_calls.calls.load(Ordering::SeqCst),
            "both paths must run the same Tier-0 solves"
        );
        assert_eq!(
            serial.tier1_calls.calls.load(Ordering::SeqCst),
            parallel.tier1_calls.calls.load(Ordering::SeqCst),
            "both paths must run the same Tier-1 solves"
        );
        assert_eq!(
            serial.tier2_calls.calls.load(Ordering::SeqCst),
            parallel.tier2_calls.calls.load(Ordering::SeqCst),
            "both paths must run the same Tier-2 solves"
        );
    }

    #[test]
    fn parallel_stage_matches_serial_tier0() {
        let candidates = stage_equivalence_candidates();
        let serial = toy_funnel_with_config(0.0, toy_config(), true);
        let serial_report = serial.funnel.screen_tier0(&candidates);
        let mut parallel_config = toy_config();
        parallel_config.stage_parallelism = 3;
        let parallel = toy_funnel_with_config(0.0, parallel_config, true);
        let parallel_report = parallel.funnel.screen_tier0(&candidates);
        assert_eq!(
            format!("{serial_report:?}"),
            format!("{parallel_report:?}"),
            "parallel ranking must match serial exactly"
        );
        assert_stage_equivalent(&serial, &parallel);
    }

    #[test]
    fn parallel_stage_matches_serial_under_budget_pressure() {
        let candidates = stage_equivalence_candidates();
        let mut budgets = TierBudgets::default();
        budgets.tier0 = 4;
        let mut serial_config = toy_config();
        serial_config.budgets = budgets;
        let serial = toy_funnel_with_config(0.0, serial_config, true);
        let serial_report = serial.funnel.screen_tier0(&candidates);
        assert!(
            serial_report.skipped_for_budget > 0,
            "budget must bind for this test"
        );
        let mut parallel_config = toy_config();
        parallel_config.budgets = budgets;
        parallel_config.stage_parallelism = 4;
        let parallel = toy_funnel_with_config(0.0, parallel_config, true);
        let parallel_report = parallel.funnel.screen_tier0(&candidates);
        assert_eq!(
            format!("{serial_report:?}"),
            format!("{parallel_report:?}"),
            "parallel ranking must match serial exactly under budget pressure"
        );
        assert_stage_equivalent(&serial, &parallel);
    }

    #[test]
    fn parallel_stage_is_deterministic_across_runs() {
        let candidates = stage_equivalence_candidates();
        let mut config = toy_config();
        config.stage_parallelism = 4;
        let first = toy_funnel_with_config(0.0, config.clone(), true);
        let first_report = first.funnel.screen_tier0(&candidates);
        let second = toy_funnel_with_config(0.0, config, true);
        let second_report = second.funnel.screen_tier0(&candidates);
        assert_eq!(
            format!("{first_report:?}"),
            format!("{second_report:?}"),
            "parallel rankings must be identical across runs"
        );
        // Ordered comparison (not the multiset): replay order is fixed.
        let strip = |history: OptimHistory| {
            history
                .records
                .iter()
                .map(|record| {
                    let mut flat = record.clone();
                    flat.wall_time_ms = 0.0;
                    if let Some(candidate) = flat.candidate.as_mut() {
                        candidate.wall_time_ms = 0.0;
                    }
                    format!("{flat:?}")
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(
            strip(first.funnel.history()),
            strip(second.funnel.history()),
            "parallel histories must be identical across runs up to wall time"
        );
    }

    #[test]
    fn parallel_rerank_and_quota_match_serial() {
        let candidates = stage_equivalence_candidates();
        let mut quota: Vec<Vec<f64>> = candidates
            .iter()
            .filter(|u| u.len() == 2)
            .cloned()
            .collect();
        quota.push(vec![2.0, 0.5]);
        let serial = toy_funnel_with_config(0.0, toy_config(), true);
        let serial_rerank = serial
            .funnel
            .rerank_tier1(&quota)
            .expect("serial rerank must run");
        let serial_quota = serial
            .funnel
            .evaluate_tier2_quota(&serial_rerank.ranking)
            .expect("serial quota must run");
        let mut parallel_config = toy_config();
        parallel_config.stage_parallelism = 2;
        let parallel = toy_funnel_with_config(0.0, parallel_config, true);
        let parallel_rerank = parallel
            .funnel
            .rerank_tier1(&quota)
            .expect("parallel rerank must run");
        let parallel_quota = parallel
            .funnel
            .evaluate_tier2_quota(&parallel_rerank.ranking)
            .expect("parallel quota must run");
        assert_eq!(
            format!("{serial_rerank:?}"),
            format!("{parallel_rerank:?}"),
            "parallel Tier-1 rerank must match serial exactly"
        );
        assert_eq!(
            format!("{serial_quota:?}"),
            format!("{parallel_quota:?}"),
            "parallel Tier-2 quota must match serial exactly"
        );
        assert_stage_equivalent(&serial, &parallel);
    }

    #[test]
    fn pack_unpack_round_trip() {
        let template = toy_template();
        let u = vec![0.25, 0.75];
        let profile = template.unpack(&u).expect("toy u must unpack");
        let back = pack_u(&profile);
        assert_eq!(back.len(), u.len());
        for (got, want) in back.iter().zip(u.iter()) {
            assert!((got - want).abs() < 1.0e-12, "pack/unpack must round-trip");
        }
    }

    #[test]
    fn search_bounds_and_constraints_use_solver_types() {
        let toy = toy_funnel(0.0);
        let bounds = toy.funnel.bounds();
        assert_eq!(bounds.len(), 2);
        for bound in &bounds {
            assert_eq!(bound.lower, Some(0.0));
            assert_eq!(bound.upper, Some(1.0));
        }
        // Tier 0 search: compression is unverifiable and excluded.
        assert_eq!(
            toy.funnel.search_constraint_names(),
            vec!["airflow", "tuning", "slope", "curvature"]
        );
        let constraints = toy.funnel.search_constraints();
        assert_eq!(constraints.len(), 4);
        for constraint in &constraints {
            assert_eq!(constraint.kind, ConstraintKind::Inequality);
            assert_eq!(constraint.dim, 1);
            assert!(constraint.jac.is_none());
        }
        // The straight tube satisfies every enforced Tier 0 constraint.
        for constraint in &constraints {
            let value = (constraint.fun)(&[0.0, 0.0]);
            assert_eq!(value.len(), 1);
            assert!(
                value[0] >= 0.0,
                "straight tube must satisfy every search constraint"
            );
        }
    }

    #[test]
    fn constraint_values_match_definitions() {
        let toy = toy_funnel(0.0);
        let u = vec![0.2, 0.4];
        let outcome = toy
            .funnel
            .evaluate(EvalTier::Tier0, &u)
            .expect("toy u must evaluate");
        let FunnelOutcome::Evaluated(candidate) = outcome else {
            panic!("toy u must produce observables");
        };
        let profile = toy_template().unpack(&u).expect("toy u must unpack");
        let (q, f0, _, _) = ToyTier0::maps(&u, 0.0);
        let (q_ref, _, _, _) = ToyTier0::maps(&[0.0, 0.0], 0.0);
        let spec = FeasibilitySpec::port_optim_defaults(F_TARGET);
        let limits = relaxed_options().limits;
        let expected = [
            q / q_ref - spec.airflow_min_ratio,
            1.0 - (f0 - spec.f_target_hz).abs() / spec.tuning_tol_hz,
            0.0, // compression placeholder below Tier 2
            1.0 - max_profile_slope(&profile) / limits.slope_max,
            1.0 - profile.curvature_integral() / limits.curvature_budget_m3,
        ];
        assert_eq!(candidate.constraints.len(), 5);
        for (got, want) in candidate.constraints.iter().zip(expected.iter()) {
            assert!(
                (got.value - want).abs() < 1.0e-12,
                "constraint {} = {} vs {want}",
                got.name,
                got.value
            );
        }
        assert!(
            !candidate.constraints[2].enforced,
            "compression unenforced below Tier 2"
        );
        assert!(candidate.feasible, "mild dip must be Tier-0 feasible");
        assert!(
            !candidate.objective_partial,
            "toy Tier 0 measures every term"
        );
    }

    #[test]
    fn toy_tier0_optimum_recovery() {
        let toy = toy_funnel(0.0);
        let config = ScalarSearchConfig {
            engine: EngineChoice::AugmentedLagrangianBobyqa,
            ..ScalarSearchConfig::default()
        };
        let result = toy
            .funnel
            .optimize_al_bobyqa(&[0.05, 0.05], &config.al_bobyqa)
            .expect("toy search must run");
        assert_eq!(result.engine, EngineUsed::AugmentedLagrangianBobyqa);
        assert!(result.feasible, "toy search must find a feasible point");
        assert!(!result.budget_exhausted);
        let best = result.best_u.expect("feasible search must return a point");
        for (got, want) in best.iter().zip(TOY_OPTIMUM.iter()) {
            assert!(
                (got - want).abs() < 0.08,
                "recovered {best:?} must approach the known optimum {TOY_OPTIMUM:?}"
            );
        }
        let (_, straight_j, _) = toy.funnel.search_point(&[0.0, 0.0]);
        let best_j = result.best_j.expect("feasible search must return J");
        assert!(
            best_j < 0.5 * straight_j,
            "best J = {best_j} must beat the straight tube J = {straight_j}"
        );
    }

    #[test]
    fn engine_auto_selects_cbo_for_small_tier2_budget() {
        let small = toy_funnel(0.0);
        assert_eq!(
            small.funnel.config().budgets.tier2,
            TierBudgets::default().tier2
        );
        let mut config = toy_config();
        config.budgets.tier2 = CBO_TIER2_BUDGET_THRESHOLD - 1;
        let (calls, tier0) = ToyTier0::shared(0.0);
        let funnel = Arc::new(
            PortOptimFunnel::new(config, tier0, None, None).expect("cBO funnel must build"),
        );
        assert_eq!(
            funnel.resolve_engine(EngineChoice::Auto),
            EngineUsed::ConstrainedBayesOpt
        );
        assert_eq!(
            small.funnel.resolve_engine(EngineChoice::Auto),
            EngineUsed::AugmentedLagrangianBobyqa
        );
        let before = calls.calls.load(Ordering::SeqCst);
        let result = funnel
            .optimize_cbo(
                &[0.05, 0.05],
                &CboSearchConfig {
                    maxeval: 25,
                    seed: Some(0),
                    ..CboSearchConfig::default()
                },
            )
            .expect("toy cBO must run");
        assert_eq!(result.engine, EngineUsed::ConstrainedBayesOpt);
        assert!(result.feasible, "toy cBO must find a feasible point");
        let (_, straight_j, _) = funnel.search_point(&[0.0, 0.0]);
        let best_j = result.best_j.expect("feasible cBO must return J");
        assert!(best_j < straight_j, "cBO must improve on the straight tube");
        // Objective plus 4 constraints share each funnel evaluation via cache.
        let solver_evals = calls.calls.load(Ordering::SeqCst) - before;
        assert!(
            solver_evals <= 26,
            "cBO must not exceed its evaluation budget (used {solver_evals})"
        );
    }

    #[test]
    fn infeasible_never_feasible() {
        // A +8 Hz tuning offset makes every candidate infeasible, including
        // the straight tube (which is always airflow-feasible by construction;
        // the -3 Hz maximum drift cannot reach the +8 Hz offset).
        let toy = toy_funnel(8.0);
        let initial = vec![
            vec![0.0, 0.0],
            vec![0.3, 0.6],
            vec![0.5, 0.5],
            vec![0.9, 0.1],
        ];
        let report = toy.funnel.run_funnel(&initial).expect("funnel must run");
        assert!(report.winners.is_empty(), "no winner may exist");
        assert!(
            report
                .tier1
                .as_ref()
                .expect("Tier 1 stage must exist")
                .ranking
                .is_empty()
        );
        assert!(
            report
                .tier2
                .as_ref()
                .expect("Tier 2 stage must exist")
                .ranking
                .is_empty()
        );
        for row in &report.tier0.ranking {
            assert!(!row.feasible, "no Tier 0 row may be feasible");
        }
        let scalar = toy
            .funnel
            .optimize_al_bobyqa(&[0.1, 0.1], &AlBobyqaConfig::default())
            .expect("search must run");
        assert!(!scalar.feasible);
        assert!(scalar.best_u.is_none(), "no feasible point may be returned");
        let pareto = toy
            .funnel
            .optimize_pareto(&ParetoSearchConfig {
                maxeval: 120,
                seed: Some(0),
                ..ParetoSearchConfig::default()
            })
            .expect("Pareto search must run");
        assert!(pareto.front.is_empty(), "no feasible front may exist");
        // Property: every feasible flag in the history implies non-negative
        // enforced constraints.
        for record in &toy.funnel.history().records {
            if let Some(candidate) = &record.candidate {
                assert!(
                    !candidate.feasible,
                    "history must hold no feasible candidate"
                );
                assert!(candidate.max_violation > 0.0);
            }
        }
    }

    #[test]
    fn cache_no_rerun() {
        let toy = toy_funnel(0.0);
        let u = vec![0.25, 0.75];
        for tier in [EvalTier::Tier0, EvalTier::Tier1, EvalTier::Tier2] {
            for _ in 0..3 {
                let outcome = toy.funnel.evaluate(tier, &u).expect("toy u must evaluate");
                assert!(outcome.candidate().is_some());
            }
        }
        // Per tier: straight baseline + u run once; the repeats are cache hits.
        assert_eq!(toy.tier0_calls.calls.load(Ordering::SeqCst), 2);
        assert_eq!(toy.tier1_calls.calls.load(Ordering::SeqCst), 2);
        assert_eq!(toy.tier2_calls.calls.load(Ordering::SeqCst), 2);
        let usage = toy.funnel.usage();
        assert_eq!(usage.tier0_used, 2);
        assert_eq!(usage.tier1_used, 2);
        assert_eq!(usage.tier2_used, 2);
        assert_eq!(usage.cache_hits, 6);
        // A sub-quantisation perturbation shares the entry.
        let perturbed = vec![u[0] + 1.0e-13, u[1] - 1.0e-13];
        let outcome = toy
            .funnel
            .evaluate(EvalTier::Tier0, &perturbed)
            .expect("perturbed u must evaluate");
        assert!(
            outcome.candidate().is_some_and(|c| c.cached),
            "sub-quantisation perturbation must hit the cache"
        );
        assert_eq!(toy.tier0_calls.calls.load(Ordering::SeqCst), 2);
        assert_eq!(toy.funnel.usage().cache_hits, 7);
    }

    #[test]
    fn rejected_by_construction_consumes_no_budget() {
        let toy = toy_funnel(0.0);
        // NaN, out-of-range, and wrong-length variables.
        for bad in [
            vec![f64::NAN, 0.5],
            vec![0.5, 1.5],
            vec![-0.1, 0.5],
            vec![0.5],
            vec![0.1, 0.2, 0.3],
        ] {
            let outcome = toy
                .funnel
                .evaluate(EvalTier::Tier0, &bad)
                .expect("rejection must not error");
            assert!(
                matches!(outcome, FunnelOutcome::Rejected { .. }),
                "illegal u = {bad:?} must be rejected by construction"
            );
        }
        // A deep dip under a tight slope limit fails profile construction.
        let mut tight = toy_config();
        tight.template.options.limits.slope_max = 0.05;
        let (_, tier0) = ToyTier0::shared(0.0);
        let tight_funnel =
            PortOptimFunnel::new(tight, tier0, None, None).expect("tight funnel must build");
        let outcome = tight_funnel
            .evaluate(EvalTier::Tier0, &[0.9, 0.9])
            .expect("rejection must not error");
        assert!(matches!(outcome, FunnelOutcome::Rejected { .. }));
        assert_eq!(tight_funnel.usage().tier0_used, 0);
        assert_eq!(tight_funnel.usage().rejected_by_construction, 1);

        let usage = toy.funnel.usage();
        assert_eq!(usage.tier0_used, 0);
        assert_eq!(usage.rejected_by_construction, 5);
        for record in &toy.funnel.history().records {
            assert_eq!(record.outcome, HistoryOutcomeKind::Rejected);
            assert_eq!(record.wall_time_ms, 0.0);
        }
    }

    #[test]
    fn budget_exhaustion_skips_and_flags() {
        let mut config = toy_config();
        config.budgets.tier0 = 3;
        let (_, tier0) = ToyTier0::shared(0.0);
        let funnel =
            PortOptimFunnel::new(config, tier0, None, None).expect("budget funnel must build");
        let candidates = (0..6)
            .map(|i| vec![0.1 * i as f64, 0.05 * i as f64])
            .collect::<Vec<_>>();
        let stage = funnel.screen_tier0(&candidates);
        // Straight (baseline + cache hit) + 2 candidates consume the budget.
        assert_eq!(funnel.usage().tier0_used, 3);
        assert_eq!(stage.ranking.len(), 3);
        assert_eq!(stage.skipped_for_budget, 3);
        let err = funnel
            .evaluate(EvalTier::Tier0, &[0.11, 0.12])
            .expect_err("exhausted budget must error");
        assert!(matches!(err, PortOptimError::BudgetExhausted { .. }));
    }

    #[test]
    fn pareto_mode_returns_feasible_front() {
        let toy = toy_funnel(0.0);
        let result = toy
            .funnel
            .optimize_pareto(&ParetoSearchConfig {
                maxeval: 300,
                seed: Some(0),
                ..ParetoSearchConfig::default()
            })
            .expect("Pareto search must run");
        assert!(!result.budget_exhausted);
        assert!(!result.front.is_empty(), "Pareto front must be non-empty");
        assert!(
            result.front.len() >= 2,
            "front must span the distortion/airflow trade-off"
        );
        for member in &result.front {
            assert!(member.feasible, "front members must be feasible");
            assert!(member.distortion_noise.is_finite());
            assert!(member.airflow_ratio.is_finite());
            assert!(member.max_violation <= 0.0);
        }
        for pair in result.front.windows(2) {
            assert!(
                pair[0].distortion_noise <= pair[1].distortion_noise,
                "front must sort by distortion ascending"
            );
        }
        let best_distortion = result
            .front
            .iter()
            .map(|m| m.distortion_noise)
            .fold(f64::INFINITY, f64::min);
        assert!(
            best_distortion < 1.0,
            "front must improve on the straight tube (distortion 2.0)"
        );
        for member in &result.infeasible_front {
            assert!(!member.feasible);
        }
    }

    #[test]
    fn compression_enforced_at_tier2() {
        let toy = toy_funnel(0.0);
        // u-sum 1.8 -> compression 1.0 dB boundary; past it, infeasible.
        let u = vec![0.95, 0.95];
        let outcome = toy
            .funnel
            .evaluate(EvalTier::Tier2, &u)
            .expect("toy Tier 2 must evaluate");
        let FunnelOutcome::Evaluated(candidate) = outcome else {
            panic!("toy Tier 2 must produce observables");
        };
        let compression = candidate
            .constraints
            .iter()
            .find(|c| c.name == "compression")
            .expect("compression constraint must exist");
        assert!(compression.enforced, "compression enforced at Tier 2");
        assert!(compression.value < 0.0, "deep dip must violate compression");
        assert!(!candidate.feasible);
        // A Tier 2 search exposes all five constraints in solver types.
        let mut config = toy_config();
        config.search_tier = EvalTier::Tier2;
        let (_, tier0) = ToyTier0::shared(0.0);
        struct T2(Arc<ToyTier2>);
        impl Tier2Evaluator for T2 {
            fn eval_tier2(
                &self,
                p: &SplinePortProfile,
            ) -> Result<PortCandidateMetrics, PortFlowError> {
                self.0.eval_tier2(p)
            }
        }
        let tier2_calls = Arc::new(ToyTier2 {
            calls: AtomicUsize::new(0),
        });
        let funnel = Arc::new(
            PortOptimFunnel::new(
                config,
                tier0,
                None,
                Some(Box::new(T2(Arc::clone(&tier2_calls)))),
            )
            .expect("Tier 2 search funnel must build"),
        );
        assert_eq!(
            funnel.search_constraint_names(),
            vec!["airflow", "tuning", "compression", "slope", "curvature"]
        );
        assert_eq!(funnel.search_constraints().len(), 5);
    }

    fn campaign_case() -> PortOptimReferenceCase {
        PortOptimReferenceCase::default()
    }

    fn campaign_duct() -> Duct1dConfig {
        Duct1dConfig {
            n_segments: DUCT1D_DEFAULT_SEGMENTS,
            thermoviscous: false,
            box_volume_m3: campaign_case().box_volume_m3,
        }
    }

    /// Real Tier 0 screen: the straight tube is exactly feasible by
    /// construction (ratio 1.0, tuning on target).
    #[test]
    fn real_tier0_straight_tube_is_feasible() {
        let case = campaign_case();
        let template = campaign_template(&case, 5, relaxed_options());
        let straight = template.straight().expect("straight must build");
        let evaluator = Duct1dTier0Evaluator {
            medium: case.medium(),
            duct: campaign_duct(),
            f_target_hz: tuning_frequency_hz(&straight, case.medium(), &campaign_duct())
                .expect("straight tuning must resolve"),
        };
        let config = FunnelConfig {
            template,
            weights: ScalarWeights::default(),
            feasibility: FeasibilitySpec::port_optim_defaults(evaluator.f_target_hz),
            budgets: TierBudgets::default(),
            search_tier: EvalTier::Tier0,
            tier2_quota: 0,
            cache_quantization_step: DEFAULT_CACHE_QUANTIZATION_STEP,
            pareto_penalty: PARETO_VIOLATION_PENALTY,
            stage_parallelism: 1,
        };
        let funnel = PortOptimFunnel::new(config, Box::new(evaluator), None, None)
            .expect("real Tier 0 funnel must build");
        let outcome = funnel
            .evaluate(EvalTier::Tier0, &[0.0, 0.0])
            .expect("straight must evaluate");
        let FunnelOutcome::Evaluated(candidate) = outcome else {
            panic!("straight must produce observables");
        };
        assert!(candidate.feasible);
        assert_eq!(candidate.q_rms_m3s, candidate.q_baseline_m3s);
        assert!(
            (candidate.f0_hz - evaluator.f_target_hz).abs() < 1.0e-9,
            "straight tuning must equal the campaign target"
        );
        assert!(
            candidate.objective_partial,
            "duct Tier 0 measures no THD/noise"
        );
        // A deep dip loses flow and drifts tuning: honestly recorded.
        let dipped = funnel
            .evaluate(EvalTier::Tier0, &[0.9, 0.9])
            .expect("dipped must evaluate");
        let FunnelOutcome::Evaluated(dipped) = dipped else {
            panic!("dipped must produce observables");
        };
        assert!(
            dipped.q_rms_m3s < candidate.q_rms_m3s,
            "dipped port must flow less than the straight tube"
        );
    }

    /// End-to-end dry run: real Tier 0 screen, toy Tier 1 re-rank, toy Tier 2
    /// quota, ranked lists, and a complete history file.
    #[test]
    fn dry_run_tier0_tier1_tier2_ranked_list() {
        let case = campaign_case();
        let template = campaign_template(&case, 5, relaxed_options());
        let straight = template.straight().expect("straight must build");
        let f_target = tuning_frequency_hz(&straight, case.medium(), &campaign_duct())
            .expect("straight tuning must resolve");
        let tier0 = Duct1dTier0Evaluator {
            medium: case.medium(),
            duct: campaign_duct(),
            f_target_hz: f_target,
        };
        let config = FunnelConfig {
            template,
            weights: ScalarWeights::default(),
            feasibility: FeasibilitySpec::port_optim_defaults(f_target),
            budgets: TierBudgets::default(),
            search_tier: EvalTier::Tier0,
            tier2_quota: 2,
            cache_quantization_step: DEFAULT_CACHE_QUANTIZATION_STEP,
            pareto_penalty: PARETO_VIOLATION_PENALTY,
            stage_parallelism: 1,
        };
        // NOTE: the toy Tier 1/2 maps are centred on F_TARGET (60 Hz), not on
        // the campaign duct tuning, so a mixed real/toy funnel would misjudge
        // tuning at the toy tiers. The dry run screens with the real duct and
        // re-ranks the survivors through a second toy funnel sharing the same
        // candidates; both histories are archived.
        let screen = PortOptimFunnel::new(config, Box::new(tier0), None, None)
            .expect("screen funnel must build");
        let initial = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.15],
            vec![0.3, 0.6],
            vec![0.5, 0.2],
            vec![0.9, 0.9],
        ];
        let tier0_report = screen.screen_tier0(&initial);
        assert_eq!(tier0_report.ranking.len(), initial.len());
        assert_eq!(tier0_report.skipped_for_budget, 0);
        assert!(
            tier0_report.ranking.iter().any(|row| row.feasible),
            "straight tube must survive the real Tier 0 screen"
        );
        for row in &tier0_report.ranking {
            assert!(row.objective_j.is_some(), "screened rows carry J");
        }

        // Re-rank the evaluated survivors with the toy Tier 1/2 funnel.
        let toy = toy_funnel(0.0);
        let survivors: Vec<Vec<f64>> = tier0_report
            .ranking
            .iter()
            .filter(|row| row.outcome == "evaluated")
            .map(|row| row.u.clone())
            .collect();
        let reranked = toy
            .funnel
            .rerank_tier1(&survivors)
            .expect("toy Tier 1 must re-rank");
        assert_eq!(reranked.ranking.len(), survivors.len());
        let quota = toy
            .funnel
            .evaluate_tier2_quota(&reranked.ranking)
            .expect("toy Tier 2 must run the quota");
        assert!(
            quota.ranking.len() <= toy.funnel.config().tier2_quota,
            "Tier 2 quota must cap evaluations"
        );
        assert!(
            quota.ranking.iter().any(|row| row.feasible),
            "quota must admit a feasible candidate"
        );
        // Per-tier budgets: Tier 2 ran at most quota + straight baseline.
        assert!(toy.funnel.usage().tier2_used <= 3);

        // Both histories serialize and round-trip with schema intact.
        for funnel in [&screen, &*toy.funnel] {
            let json = funnel.history_json().expect("history must serialize");
            let parsed = OptimHistory::from_json_str(&json).expect("history must parse");
            assert_eq!(parsed.schema_version, PORT_OPTIM_HISTORY_SCHEMA_VERSION);
            assert_eq!(parsed.provenance, PORT_OPTIM_HISTORY_PROVENANCE);
            assert_eq!(parsed.records.len(), funnel.history().records.len());
            assert!(parsed.records.iter().all(|r| r.seq >= 1));
        }
    }

    #[test]
    fn history_json_rejects_schema_drift() {
        let toy = toy_funnel(0.0);
        toy.funnel
            .evaluate(EvalTier::Tier0, &[0.1, 0.2])
            .expect("toy u must evaluate");
        let json = toy.funnel.history_json().expect("history must serialize");
        let mut value: serde_json::Value =
            serde_json::from_str(&json).expect("history must be JSON");
        value["schema_version"] = serde_json::Value::from(999u64);
        let err =
            OptimHistory::from_json_str(&serde_json::to_string(&value).expect("JSON must render"))
                .expect_err("schema drift must fail loudly");
        assert!(matches!(err, PortOptimError::History(_)));
        let mut value: serde_json::Value =
            serde_json::from_str(&json).expect("history must be JSON");
        value["provenance"] = serde_json::Value::from("forged");
        let err =
            OptimHistory::from_json_str(&serde_json::to_string(&value).expect("JSON must render"))
                .expect_err("provenance drift must fail loudly");
        assert!(matches!(err, PortOptimError::History(_)));
    }

    /// Real Tier 2 quota at coarse screening fidelity (slot scale): the
    /// straight tube promotes and stays feasible; the quota caps the solves.
    #[test]
    fn real_tier2_quota_coarse_slot_scale() {
        let case = campaign_case();
        let template = tier2_slot_template(&case, 5, ProfileOptions::default());
        let straight = template.straight().expect("straight must build");
        let duct = Duct1dConfig {
            n_segments: DUCT1D_DEFAULT_SEGMENTS,
            thermoviscous: false,
            box_volume_m3: case.box_volume_m3,
        };
        let f_target =
            tuning_frequency_hz(&straight, case.medium(), &duct).expect("slot tuning must resolve");
        let tier0 = Duct1dTier0Evaluator {
            medium: case.medium(),
            duct,
            f_target_hz: f_target,
        };
        let context =
            Tier2EvalContext::from_reference_case(case.clone()).expect("Tier 2 context must build");
        let drive = PortEvalDrive {
            voltage_peak_v: 0.1,
            frequency_hz: case
                .tier2_drive_frequency_hz()
                .expect("Tier 2 drive frequency must resolve"),
            second_tone: None,
        };
        let fidelity = PortEvalFidelity {
            samples_per_cycle: 48,
            total_cycles: 10,
            tail_cycles: 4,
            snapshot_interval_steps: 0,
            compression_reference: None,
        };
        let tier2 = Tier2HarnessEvaluator {
            context,
            drive,
            fidelity,
        };
        let config = FunnelConfig {
            template,
            weights: ScalarWeights::default(),
            feasibility: FeasibilitySpec::port_optim_defaults(f_target),
            budgets: TierBudgets::default(),
            search_tier: EvalTier::Tier0,
            tier2_quota: 2,
            cache_quantization_step: DEFAULT_CACHE_QUANTIZATION_STEP,
            pareto_penalty: PARETO_VIOLATION_PENALTY,
            stage_parallelism: 1,
        };
        let funnel = PortOptimFunnel::new(config, Box::new(tier0), None, Some(Box::new(tier2)))
            .expect("slot funnel must build");
        let report = funnel
            .run_funnel(&[vec![0.0, 0.0], vec![0.1, 0.15]])
            .expect("slot funnel must run");
        let tier2 = report.tier2.expect("Tier 2 stage must exist");
        assert_eq!(tier2.ranking.len(), 2, "quota of 2 admits both rows");
        let straight_row = tier2
            .ranking
            .iter()
            .find(|row| row.u == vec![0.0, 0.0])
            .expect("straight row must exist");
        assert_eq!(straight_row.outcome, "evaluated");
        assert!(
            straight_row.feasible,
            "straight tube must stay Tier-2 feasible"
        );
        assert!(
            !straight_row.objective_partial,
            "Tier 2 measures every objective term"
        );
        assert_eq!(funnel.usage().tier2_used, 2, "straight baseline + dip");
        assert!(!report.winners.is_empty());
    }

    /// Real Tier 1 oracle wiring (ignored: one P2 reference solve).
    #[test]
    #[ignore]
    fn real_tier1_oracle_evaluates_straight_tube() {
        use sonium_fem::physics::acoustics::pressure::FluidProperties;

        let case = campaign_case();
        let template = campaign_template(&case, 5, ProfileOptions::default());
        let straight = template.straight().expect("straight must build");
        let duct = campaign_duct();
        let f_target = tuning_frequency_hz(&straight, case.medium(), &duct)
            .expect("straight tuning must resolve");
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
            mesh_size_m: 0.02,
            angular_sectors: None,
            fluid: FluidProperties::air_20c(),
            box_volume_m3: case.box_volume_m3,
            f_target_hz: f_target,
        };
        let config = FunnelConfig {
            template,
            weights: ScalarWeights::default(),
            feasibility: FeasibilitySpec::port_optim_defaults(f_target),
            budgets: TierBudgets::default(),
            search_tier: EvalTier::Tier0,
            tier2_quota: 0,
            cache_quantization_step: DEFAULT_CACHE_QUANTIZATION_STEP,
            pareto_penalty: PARETO_VIOLATION_PENALTY,
            stage_parallelism: 1,
        };
        let funnel = PortOptimFunnel::new(config, Box::new(tier0), Some(Box::new(tier1)), None)
            .expect("Tier 1 funnel must build");
        let outcome = funnel
            .evaluate(EvalTier::Tier1, &[0.0, 0.0])
            .expect("Tier 1 straight must evaluate");
        let FunnelOutcome::Evaluated(candidate) = outcome else {
            panic!("Tier 1 straight must produce observables");
        };
        assert!(candidate.feasible, "straight tube must be Tier-1 feasible");
        assert!(candidate.q_rms_m3s > 0.0);
    }
}
