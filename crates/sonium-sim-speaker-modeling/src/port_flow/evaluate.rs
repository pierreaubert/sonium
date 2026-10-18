//! Tier 2 candidate evaluation harness: one spline profile in, metrics out (PORT-OPTIM.md Step 5).
//!
//! [`Tier2EvalContext`] freezes the Step-0 Tier-2 envelope (slot-surrogate
//! meridian mesh, lumped box/driver, medium, Taylor–Hood P2–P1 BDF2 with
//! residual stabilisation off) and morphs one shared straight baseline mesh
//! onto each candidate profile (plan decision D6).
//! [`Tier2EvalContext::evaluate`] is a pure function of
//! (profile, drive, fidelity): it assembles the [`CoupledPortConfig`], calls
//! [`run_coupled_port`], and reduces the settled tail to
//! [`PortCandidateMetrics`] — `Q_rms`, Tier-0 `f0` estimate, THD, IMD,
//! compression, peak Mach numbers, separation fraction, the `U⁸/L³`
//! turbulence-trend noise proxy, curvature budget, energy-closure diagnostics,
//! and solver health flags.
//!
//! # Slot-scale profiles
//!
//! Tier 2 runs on the 2D slot surrogate, so candidate profiles live at the
//! slot scale: bore radius `tier2_duct_height/2` (0.01 m), not the campaign
//! round-tube `R0` (0.025 m). [`Tier2EvalContext::candidate_profile`] builds
//! envelope-consistent profiles from normalised `u` vectors; profiles built
//! at any other scale are rejected before any solve. The straight slot-scale
//! profile morphs to the bit-identical Step-0 baseline mesh (morphing with
//! `u = 0` is the identity map), so a straight evaluation at production
//! fidelity reproduces the frozen Step-0 Tier-2 archive.
//!
//! # Snapshots
//!
//! Field snapshots are opt-in (`snapshot_interval_steps = 0` disables them)
//! and reserved for finalist evaluation feeding [`analyze_port_noise`](crate::analyze_port_noise).
//! The field peak Mach number and the separation fraction need snapshots and
//! are `None` without them; every other metric is available on every run.
//!
//! # Error contract
//!
//! Invalid profiles, drives, and fidelity settings are rejected with
//! [`PortFlowError::InvalidConfig`] before any solve consumes budget. A
//! morph that inverts an element returns [`PortFlowError::Mesh`]. Divergence,
//! non-finite state, and unanalyzable tails (zero fundamental, non-finite
//! derived scalars) return [`PortFlowError::SolverDiverged`]. Metrics are
//! total on success: every scalar field is finite, and `Option` fields are
//! `None` only when their input was not requested (single-tone IMD,
//! missing compression reference, disabled snapshots) — never a silent zero.
//!
//! Validated envelope: 2D meridian/slot evidence at the evaluated fidelity,
//! plus a Tier-0 tuning estimate. Not a 3D claim; the nightly 3D gates stay
//! untouched. Single- or two-tone harmonic drive only.

use std::f64::consts::{PI, TAU};

use serde::{Deserialize, Serialize};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::Mesh;
use sonium_fem::physics::fluid::stokes::navier_stokes::{
    BackendPreconditionerKind, BackendPressureSchurPreconditionerKind, LinearSolverBackend,
    LinearSolverCpuPreconditionerKind, NavierStokesConfig,
};
use sonium_fem::physics::fluid::stokes::time_stepping::BdfConfig;
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};

use super::analysis::{dft_amplitude, imd_fraction, thd_fraction};
use super::baseline::{PortOptimReferenceCase, compression_db, duct_bulk_mach_number};
use super::coupled::{
    CoupledPortConfig, CoupledPortResult, PortFlowFieldSnapshot, run_coupled_port,
};
use super::duct1d::{DUCT1D_DEFAULT_SEGMENTS, Duct1dConfig, tuning_frequency_hz};
use super::error::PortFlowError;
use super::geometry::build_slot_port_mesh;
use super::profile::{PROFILE_VALIDATION_GRID_POINTS, ProfileOptions, SplinePortProfile};
use super::spline_geometry::{
    DEFAULT_MORPH_SMOOTHING_PASSES, SplinePortGeometry, min_triangle_area_m2,
};
use crate::free_turbulence_power_trend;

/// Schema version for [`PortCandidateMetrics`].
pub const TIER2_EVAL_SCHEMA_VERSION: u32 = 1;

/// Model lineage recorded in every [`PortCandidateMetrics`].
pub const TIER2_EVAL_PROVENANCE: &str =
    "Tier2MeridianSlot[CoupledP2P1BDF2,NoStabilisation,AffineBoxClosure]";

/// Minimum BDF samples per drive cycle accepted by the harness.
///
/// The Nyquist audit against the analysed harmonics is the real guard; this
/// floor only rejects degenerate DFT tails.
pub const TIER2_MIN_SAMPLES_PER_CYCLE: usize = 8;

/// Acceptance bound for the per-step accepted linear-solve true residual.
///
/// Direct-solve telemetry lands at roundoff scale; this bound carries wide
/// margin and only trips on a genuinely suspicious linear solve.
pub const TIER2_LINEAR_RESIDUAL_OK_MAX: f64 = 1.0e-6;

/// Default throat fraction `r_min = 0.6·R0` used by [`Tier2EvalContext::candidate_profile`].
pub const TIER2_THROAT_FRACTION_OF_BORE: f64 = 0.6;

/// Relative scale below which negative dissipation counts as roundoff.
///
/// Matches the `qa-speaker-port-nonlinear` 2D leg and the Step-0 baseline
/// regeneration audit.
const DISSIPATION_NEGATIVITY_RELATIVE_TOLERANCE: f64 = 1.0e-12;

/// Snapshots whose instantaneous bulk flow falls below this fraction of the
/// tail maximum sit on a flow-reversal instant and carry no separation
/// information, so they are skipped rather than counted.
const SEPARATION_ZERO_FLOW_RELATIVE_TOLERANCE: f64 = 1.0e-6;

/// Axial tolerance around `[0, Ld]` for the duct-band DOF mask.
///
/// Covers float roundoff only; jittered interior nodes near the step faces
/// stay on their built side.
const DUCT_MASK_TOLERANCE_M: f64 = 1.0e-12;

/// Harmonics (including the fundamental) in the THD reduction.
const THD_HARMONICS: usize = 5;

/// Second harmonic tone for a two-tone IMD evaluation.
///
/// The secondary frequency is an exact rational multiple `f2 = f1·num/den`
/// with `num > den`, so the settled tail always holds an integer number of
/// common periods and every IMD product lands on a DFT bin.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SecondTone {
    /// Numerator of the `f2/f1` ratio (must exceed [`Self::denominator`]).
    pub numerator: u32,
    /// Denominator of the `f2/f1` ratio (at least 1).
    pub denominator: u32,
    /// Secondary-tone peak as a fraction of the primary-tone peak.
    pub amplitude_ratio: f64,
}

impl SecondTone {
    /// Secondary-tone frequency in Hz for a primary tone at `f1_hz`.
    pub fn frequency_hz(&self, f1_hz: f64) -> f64 {
        f1_hz * (f64::from(self.numerator) / f64::from(self.denominator))
    }
}

/// Terminal-voltage drive specification for one Tier 2 evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PortEvalDrive {
    /// Primary-tone peak voltage in V (peak of the `f1` sine, not RMS).
    pub voltage_peak_v: f64,
    /// Primary-tone frequency in Hz (THD fundamental, timestep base).
    pub frequency_hz: f64,
    /// Optional second tone; `None` selects a single-tone sine drive.
    pub second_tone: Option<SecondTone>,
}

/// Small-signal reference for the compression reduction.
///
/// Compression is `20·log10((V/V0)/(Q/Q0))` against this reference; without
/// one the harness reports `compression_db: None` rather than inventing a
/// baseline.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SmallSignalReference {
    /// Reference peak voltage in V.
    pub voltage_peak_v: f64,
    /// Reference port-flow fundamental amplitude in m³/s.
    pub port_flow_fundamental_amplitude_m3s: f64,
}

impl SmallSignalReference {
    /// Reject a non-positive or non-finite reference before any solve.
    pub fn validate(&self) -> Result<(), PortFlowError> {
        if !(self.voltage_peak_v.is_finite() && self.voltage_peak_v > 0.0) {
            return Err(PortFlowError::InvalidConfig(format!(
                "compression reference voltage must be finite and positive, got {}",
                self.voltage_peak_v
            )));
        }
        if !(self.port_flow_fundamental_amplitude_m3s.is_finite()
            && self.port_flow_fundamental_amplitude_m3s > 0.0)
        {
            return Err(PortFlowError::InvalidConfig(format!(
                "compression reference flow must be finite and positive, got {}",
                self.port_flow_fundamental_amplitude_m3s
            )));
        }
        Ok(())
    }
}

/// Time resolution, run length, tail selection, and snapshot policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PortEvalFidelity {
    /// BDF samples per primary-tone cycle (sets `dt = 1/(f1·spc)`).
    pub samples_per_cycle: usize,
    /// Total drive cycles simulated (transient plus settled tail).
    pub total_cycles: usize,
    /// Settled tail cycles analysed (must be `< total_cycles`).
    pub tail_cycles: usize,
    /// Snapshot stride in BDF steps (`0` disables snapshots).
    ///
    /// A nonzero stride must not exceed the tail length, which guarantees at
    /// least one snapshot lands in the analysed tail.
    pub snapshot_interval_steps: usize,
    /// Small-signal reference for compression (`None` skips compression).
    pub compression_reference: Option<SmallSignalReference>,
}

impl PortEvalFidelity {
    /// Production Step-0 fidelity: 252 samples/cycle, 40 cycles, 8-cycle tail.
    pub fn from_reference_case(case: &PortOptimReferenceCase) -> Self {
        Self {
            samples_per_cycle: case.tier2_samples_per_cycle,
            total_cycles: case.tier2_total_cycles,
            tail_cycles: case.tier2_tail_cycles,
            snapshot_interval_steps: 0,
            compression_reference: None,
        }
    }

    /// Reject an inconsistent fidelity before any solve.
    pub fn validate(&self) -> Result<(), PortFlowError> {
        if self.samples_per_cycle < TIER2_MIN_SAMPLES_PER_CYCLE {
            return Err(PortFlowError::InvalidConfig(format!(
                "Tier 2 needs at least {TIER2_MIN_SAMPLES_PER_CYCLE} samples per cycle, got {}",
                self.samples_per_cycle
            )));
        }
        if self.tail_cycles == 0 || self.tail_cycles >= self.total_cycles {
            return Err(PortFlowError::InvalidConfig(format!(
                "Tier 2 needs 0 < tail_cycles < total_cycles, got tail {} of total {}",
                self.tail_cycles, self.total_cycles
            )));
        }
        let tail_samples = self
            .tail_cycles
            .checked_mul(self.samples_per_cycle)
            .ok_or_else(|| {
                PortFlowError::InvalidConfig("Tier 2 tail sample count overflow".to_string())
            })?;
        self.total_cycles
            .checked_mul(self.samples_per_cycle)
            .ok_or_else(|| {
                PortFlowError::InvalidConfig("Tier 2 total sample count overflow".to_string())
            })?;
        if self.snapshot_interval_steps > tail_samples {
            return Err(PortFlowError::InvalidConfig(format!(
                "Tier 2 snapshot stride {} exceeds the {tail_samples}-sample tail; \
                 no snapshot would land in the analysed window",
                self.snapshot_interval_steps
            )));
        }
        if let Some(reference) = &self.compression_reference {
            reference.validate()?;
        }
        Ok(())
    }
}

/// Validated (drive, fidelity) pair with the derived run parameters.
#[derive(Debug, Clone, Copy)]
struct ValidatedRun {
    dt_s: f64,
    num_steps: usize,
    tail_samples: usize,
    second_tone_hz: Option<f64>,
}

impl PortEvalDrive {
    /// Highest analysed frequency: 5th primary harmonic, plus the 5th
    /// secondary harmonic and the `f2 + 2·f1` IMD product for two-tone runs.
    fn highest_analysis_hz(&self, second_tone_hz: Option<f64>) -> f64 {
        let harmonics = THD_HARMONICS as f64;
        let primary = harmonics * self.frequency_hz;
        second_tone_hz.map_or(primary, |f2| {
            primary
                .max(harmonics * f2)
                .max(f2 + 2.0 * self.frequency_hz)
        })
    }

    /// Validate the drive against the fidelity and derive run parameters.
    fn validate_run(&self, fidelity: &PortEvalFidelity) -> Result<ValidatedRun, PortFlowError> {
        fidelity.validate()?;
        if !(self.voltage_peak_v.is_finite() && self.voltage_peak_v > 0.0) {
            return Err(PortFlowError::InvalidConfig(format!(
                "Tier 2 drive voltage must be finite and positive, got {}",
                self.voltage_peak_v
            )));
        }
        if !(self.frequency_hz.is_finite() && self.frequency_hz > 0.0) {
            return Err(PortFlowError::InvalidConfig(format!(
                "Tier 2 drive frequency must be finite and positive, got {}",
                self.frequency_hz
            )));
        }
        let second_tone_hz = match &self.second_tone {
            None => None,
            Some(tone) => {
                if tone.numerator == 0 || tone.denominator == 0 {
                    return Err(PortFlowError::InvalidConfig(format!(
                        "Tier 2 second-tone ratio must have positive numerator/denominator, got {}/{}",
                        tone.numerator, tone.denominator
                    )));
                }
                if tone.numerator <= tone.denominator {
                    return Err(PortFlowError::InvalidConfig(format!(
                        "Tier 2 second tone must sit above the primary (f2 > f1), got ratio {}/{}",
                        tone.numerator, tone.denominator
                    )));
                }
                if !(tone.amplitude_ratio.is_finite() && tone.amplitude_ratio > 0.0) {
                    return Err(PortFlowError::InvalidConfig(format!(
                        "Tier 2 second-tone amplitude ratio must be finite and positive, got {}",
                        tone.amplitude_ratio
                    )));
                }
                // u128 arithmetic: validation must never panic, even on
                // absurd cycle counts that later fail allocation-free checks.
                let aligned = (fidelity.tail_cycles as u128) * (u128::from(tone.numerator))
                    % (u128::from(tone.denominator));
                if aligned != 0 {
                    return Err(PortFlowError::InvalidConfig(format!(
                        "Tier 2 two-tone tail holds {} primary cycles, not an integer number of \
                         {}/{} common periods; IMD products would leak between DFT bins",
                        fidelity.tail_cycles, tone.numerator, tone.denominator
                    )));
                }
                let f2 = tone.frequency_hz(self.frequency_hz);
                if !(f2.is_finite() && f2 > 0.0) {
                    return Err(PortFlowError::InvalidConfig(format!(
                        "Tier 2 second-tone frequency is non-finite: {f2}"
                    )));
                }
                Some(f2)
            }
        };
        let dt_s = 1.0 / (self.frequency_hz * fidelity.samples_per_cycle as f64);
        if !(dt_s.is_finite() && dt_s > 0.0) {
            return Err(PortFlowError::InvalidConfig(format!(
                "Tier 2 timestep is non-finite: {dt_s}"
            )));
        }
        let highest = self.highest_analysis_hz(second_tone_hz);
        let nyquist_hz = 0.5 / dt_s;
        if highest >= nyquist_hz {
            return Err(PortFlowError::InvalidConfig(format!(
                "Tier 2 cannot analyse {highest:.3e} Hz at dt = {dt_s:.3e} s: \
                 highest analysed frequency must sit below Nyquist {nyquist_hz:.3e} Hz"
            )));
        }
        let num_steps = fidelity.total_cycles * fidelity.samples_per_cycle;
        let tail_samples = fidelity.tail_cycles * fidelity.samples_per_cycle;
        Ok(ValidatedRun {
            dt_s,
            num_steps,
            tail_samples,
            second_tone_hz,
        })
    }
}

/// Solver health evidence for one Tier 2 evaluation.
///
/// Energy-closure and cap-continuity bounds are imposed by the Step-9
/// promotion ladder at production fidelity, not here: coarse screening runs
/// legitimately close worse, so this struct reports the measured values and
/// only flags the fidelity-independent properties (nonnegative physical
/// dissipation channels, sane linear-solve telemetry).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PortEvalHealth {
    /// Every tail energy step has finite channels and nonnegative physical
    /// dissipation (within [`DISSIPATION_NEGATIVITY_RELATIVE_TOLERANCE`]).
    pub dissipations_nonnegative: bool,
    /// Worst accepted affine-solve true relative residual over the tail.
    pub max_accepted_linear_relative_residual: f64,
    /// Whether the worst linear residual respects [`TIER2_LINEAR_RESIDUAL_OK_MAX`].
    pub linear_residual_ok: bool,
    /// `max|q_ext + q_box,out|/max|q_ext|` over the tail (time domain).
    pub cap_continuity_relative_error: f64,
    /// `max|box-conservation residual|/max|p_box|` over the tail.
    pub box_conservation_relative_residual: f64,
    /// Aggregate: every boolean flag in this struct holds.
    pub passed: bool,
}

/// Reduced observables for one evaluated port candidate.
///
/// All scalar fields are finite on success. `Option` fields are `None` only
/// when their input was not requested — single-tone IMD, missing compression
/// reference, disabled snapshots — never a silent zero.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PortCandidateMetrics {
    /// Must equal [`TIER2_EVAL_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Model lineage; must equal [`TIER2_EVAL_PROVENANCE`].
    pub provenance: String,
    /// Candidate identity: normalised free radii `u` (knots `2..N-1`).
    pub profile_normalized: Vec<f64>,
    /// Control-point count `N` of the evaluated profile.
    pub n_control_points: usize,
    /// Primary-tone peak drive voltage in V.
    pub voltage_peak_v: f64,
    /// Primary-tone drive frequency in Hz.
    pub frequency_hz: f64,
    /// Secondary-tone frequency in Hz (`None` for single-tone runs).
    pub second_tone_frequency_hz: Option<f64>,
    /// BDF timestep in s.
    pub dt_s: f64,
    /// Completed BDF steps (always the configured total on success).
    pub steps_completed: usize,
    /// Analysed settled-tail samples.
    pub tail_samples: usize,
    /// Retained snapshots landing in the analysed tail (0 when disabled).
    pub n_tail_snapshots: usize,
    /// Time-domain port-flow RMS over the tail in m³/s.
    pub port_flow_rms_m3s: f64,
    /// Fundamental port-flow amplitude over the tail in m³/s.
    pub port_flow_fundamental_amplitude_m3s: f64,
    /// Fundamental box-pressure amplitude over the tail in Pa.
    pub box_pressure_amplitude_pa: f64,
    /// Peak diaphragm excursion over the tail in m.
    pub peak_excursion_m: f64,
    /// Tier-0 compact tuning estimate of the evaluated profile in Hz.
    ///
    /// Screens relative tuning drift between candidates; the absolute tuning
    /// constraint is enforced at Tier 0/1 on campaign-scale profiles.
    pub f0_estimate_hz: f64,
    /// Total harmonic distortion over [`THD_HARMONICS`] harmonics (fraction).
    pub thd_fraction: f64,
    /// Two-tone IMD around `f2` (fraction; `None` for single-tone runs).
    pub imd_fraction: Option<f64>,
    /// Port-output compression against the fidelity reference in dB.
    pub compression_db: Option<f64>,
    /// Bulk-velocity Mach number `max|q|/(S_min·c)` over the tail.
    ///
    /// A lower bound on the true peak Mach: the centreline velocity of the
    /// duct profile exceeds the bulk value.
    pub peak_mach_bulk: f64,
    /// Field peak Mach `max|u|/c` over tail snapshots (`None` when disabled).
    pub peak_mach_field: Option<f64>,
    /// Mean reversed-axial-flow DOF fraction in the duct band over usable
    /// tail snapshots (`None` without snapshots, or when every tail snapshot
    /// sits on a flow-reversal instant).
    pub separation_fraction: Option<f64>,
    /// Free-turbulence `U⁸/L³` noise-proxy trend from the fundamental throat velocity.
    pub turbulence_power_trend: f64,
    /// Profile curvature budget `∫(r'')² dx` in m⁻³.
    pub curvature_integral_m3: f64,
    /// Physical-only energy residual over the tail (relative).
    pub energy_physical_relative_closure: f64,
    /// Full-ledger energy residual over the tail (relative).
    pub energy_algebraic_relative_closure: f64,
    /// Cumulative interface energy mismatch over the tail (relative).
    pub energy_interface_relative_residual: f64,
    /// Energy normalisation scale over the tail in J.
    pub energy_scale_j: f64,
    /// Solver health evidence.
    pub health: PortEvalHealth,
}

/// Complete Tier 2 evaluation product: reduced metrics plus the evidence.
#[derive(Debug, Clone)]
pub struct Tier2EvalOutcome {
    /// Reduced candidate observables.
    pub metrics: PortCandidateMetrics,
    /// Morphed mesh the candidate ran on (straight candidates: baseline clone).
    pub mesh: Mesh,
    /// Retained field snapshots with their timestamps (empty when disabled).
    pub field_snapshots: Vec<PortFlowFieldSnapshot>,
}

/// Linear-solver device for Tier-2 evaluations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tier2SolverDevice {
    /// Legacy CPU path: sparse direct solve (the frozen-campaign default).
    #[default]
    Cpu,
    /// GPU path: backend GMRES with a Jacobi-Schur/Chebyshev block
    /// preconditioner, mirroring the proven 3D spot-check configuration.
    /// Requires a `cuda`-feature build plus a CUDA driver; the harness
    /// fails loudly when a solve resolves to the wrong device.
    Cuda,
}

/// Linear-solver settings for Tier-2 evaluations.
///
/// The CPU device ignores the iteration fields (direct solve); they apply
/// to the iterative GPU backend path only.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Tier2SolverSettings {
    /// Device that must resolve every linear solve.
    pub device: Tier2SolverDevice,
    /// Maximum backend GMRES iterations per solve (GPU path).
    pub max_iterations: usize,
    /// Relative residual tolerance per solve (GPU path).
    pub tolerance: f64,
}

impl Default for Tier2SolverSettings {
    fn default() -> Self {
        Self {
            device: Tier2SolverDevice::Cpu,
            max_iterations: 5_000,
            tolerance: 1.0e-8,
        }
    }
}

/// Frozen Tier-2 evaluation envelope with one shared straight baseline mesh.
///
/// The reference tube, chambers, mesh size, lumped box/driver, and medium all
/// come from the Step-0 [`PortOptimReferenceCase`]; every candidate profile
/// morphs the same baseline mesh, so the objective stays continuous across
/// the optimisation loop (plan decision D6).
#[derive(Debug, Clone)]
pub struct Tier2EvalContext {
    case: PortOptimReferenceCase,
    duct_length_m: f64,
    bore_radius_m: f64,
    chamber_length_m: f64,
    chamber_radius_m: f64,
    mesh_size_m: f64,
    smoothing_passes: usize,
    baseline_mesh: Mesh,
    solver: Tier2SolverSettings,
}

impl Tier2EvalContext {
    /// Build the production context: Step-0 Tier-2 slot envelope.
    ///
    /// The straight reference is the frozen slot surrogate
    /// (`bore = tier2_duct_height/2`, chambers halved the same way) meshed
    /// with the Step-0 deterministic builder, so straight evaluations replay
    /// the baseline campaign bit-near-identically.
    ///
    /// # Errors
    ///
    /// Returns [`PortFlowError::InvalidConfig`] for an invalid reference case
    /// or [`PortFlowError::Mesh`] when the baseline mesh fails to build.
    pub fn from_reference_case(case: PortOptimReferenceCase) -> Result<Self, PortFlowError> {
        case.validate()?;
        let baseline_mesh = build_slot_port_mesh(&case.slot_geometry())?;
        Ok(Self {
            duct_length_m: case.tier2_duct_length_m,
            bore_radius_m: 0.5 * case.tier2_duct_height_m,
            chamber_length_m: case.tier2_chamber_length_m,
            chamber_radius_m: 0.5 * case.tier2_chamber_height_m,
            mesh_size_m: case.tier2_mesh_size_m,
            smoothing_passes: DEFAULT_MORPH_SMOOTHING_PASSES,
            case,
            baseline_mesh,
            solver: Tier2SolverSettings::default(),
        })
    }

    /// Rebuild the baseline mesh at a coarser (or finer) cell size.
    ///
    /// Screening and unit-test use only: production evidence always runs at
    /// the reference mesh size. Consumes and returns the context so the
    /// override stays visible at the call site.
    ///
    /// # Errors
    ///
    /// Returns [`PortFlowError::InvalidConfig`] for a non-positive mesh size
    /// or [`PortFlowError::Mesh`] when the rebuilt mesh fails.
    pub fn with_mesh_size_m(mut self, mesh_size_m: f64) -> Result<Self, PortFlowError> {
        if !(mesh_size_m.is_finite() && mesh_size_m > 0.0) {
            return Err(PortFlowError::InvalidConfig(format!(
                "Tier 2 mesh size override must be finite and positive, got {mesh_size_m}"
            )));
        }
        let mut slot = self.case.slot_geometry();
        slot.mesh_size_m = mesh_size_m;
        self.baseline_mesh = build_slot_port_mesh(&slot)?;
        self.mesh_size_m = mesh_size_m;
        Ok(self)
    }

    /// Select the linear-solver device/settings for evaluations.
    ///
    /// Consumes and returns the context so the override stays visible at
    /// the call site. The CPU device (default) keeps the frozen direct
    /// solver; CUDA switches to the iterative backend path and makes
    /// [`Self::evaluate`] fail loudly if any solve resolves off-device.
    pub fn with_solver_settings(mut self, solver: Tier2SolverSettings) -> Self {
        self.solver = solver;
        self
    }

    /// Linear-solver settings evaluations will use.
    pub fn solver_settings(&self) -> Tier2SolverSettings {
        self.solver
    }

    /// The frozen reference case backing this context.
    pub fn case(&self) -> &PortOptimReferenceCase {
        &self.case
    }

    /// The shared straight baseline mesh every candidate morphs from.
    pub fn baseline_mesh(&self) -> &Mesh {
        &self.baseline_mesh
    }

    /// Reference duct length in m (candidate profiles must match it).
    pub fn duct_length_m(&self) -> f64 {
        self.duct_length_m
    }

    /// Reference bore radius in m (candidate profiles must match it).
    pub fn bore_radius_m(&self) -> f64 {
        self.bore_radius_m
    }

    /// Laplacian smoothing passes applied to the duct interior after morphing.
    pub fn smoothing_passes(&self) -> usize {
        self.smoothing_passes
    }

    /// Build an envelope-consistent candidate profile from normalised radii.
    ///
    /// Uses the context duct length and slot-scale bore with the plan-default
    /// throat `r_min = 0.6·R0`, so optimiser `u` vectors cannot drift off the
    /// Tier-2 envelope.
    ///
    /// # Errors
    ///
    /// Returns [`PortFlowError::InvalidConfig`] for an illegal `u` vector
    /// (wrong length, out of `[0, 1]`, or violating the shape limits).
    pub fn candidate_profile(
        &self,
        n_control_points: usize,
        u: &[f64],
        options: ProfileOptions,
    ) -> Result<SplinePortProfile, PortFlowError> {
        SplinePortProfile::from_normalized(
            n_control_points,
            self.duct_length_m,
            self.bore_radius_m,
            TIER2_THROAT_FRACTION_OF_BORE * self.bore_radius_m,
            u,
            options,
        )
    }

    /// Pair a candidate profile with the context reference envelope.
    pub fn candidate_geometry(&self, profile: &SplinePortProfile) -> SplinePortGeometry {
        SplinePortGeometry {
            duct_length_m: self.duct_length_m,
            bore_radius_m: self.bore_radius_m,
            chamber_length_m: self.chamber_length_m,
            chamber_radius_m: self.chamber_radius_m,
            mesh_size_m: self.mesh_size_m,
            profile: profile.clone(),
        }
    }

    /// Morph the shared baseline onto a candidate profile, audited.
    ///
    /// # Errors
    ///
    /// Returns [`PortFlowError::InvalidConfig`] when the profile is illegal
    /// or off-envelope, or [`PortFlowError::Mesh`] when the morphed mesh
    /// inverts or degenerates an element.
    pub fn morphed_mesh(&self, profile: &SplinePortProfile) -> Result<Mesh, PortFlowError> {
        profile.validate()?;
        let mesh = self
            .candidate_geometry(profile)
            .morph_baseline_mesh(&self.baseline_mesh, self.smoothing_passes)?;
        let min_area = min_triangle_area_m2(&mesh);
        if min_area.is_nan() || min_area <= 0.0 {
            return Err(PortFlowError::Mesh(format!(
                "Tier 2 morphed mesh failed the pre-solve positivity audit \
                 (min signed area {min_area:.3e} m²)"
            )));
        }
        Ok(mesh)
    }

    /// Assemble the coupled BDF2 configuration for a drive/fidelity pair.
    ///
    /// Mirrors [`PortOptimReferenceCase::coupled_config`] (same driver,
    /// medium-derived Navier–Stokes parameters, stabilisation off,
    /// `under_relaxation = 1.0`, zero exterior gauge) at the requested
    /// timestep, length, drive, and snapshot stride.
    ///
    /// # Errors
    ///
    /// Returns [`PortFlowError::InvalidConfig`] for an inconsistent
    /// drive/fidelity pair.
    pub fn coupled_config(
        &self,
        drive: &PortEvalDrive,
        fidelity: &PortEvalFidelity,
    ) -> Result<CoupledPortConfig, PortFlowError> {
        let run = drive.validate_run(fidelity)?;
        let voltage_peak_v = drive.voltage_peak_v;
        let f1_hz = drive.frequency_hz;
        let second = run.second_tone_hz.map(|f2| {
            let ratio = drive.second_tone.map_or(0.0, |tone| tone.amplitude_ratio);
            (f2, ratio)
        });
        // CPU device keeps the frozen Navier-Stokes defaults exactly
        // (sparse direct solve). CUDA switches to the iterative backend
        // path with the 3D spot-check block preconditioner.
        let mut ns_config = NavierStokesConfig {
            mu: self.case.dynamic_viscosity_pa_s,
            rho: self.case.density_kg_per_m3,
            use_residual_stabilisation: false,
            // Zero body force through `solve_bdf_step_cached`: the
            // force quadrature is identical every step, so caching
            // skips two full assemblies per step after the first.
            cache_body_force_assembly: true,
            ..Default::default()
        };
        if self.solver.device == Tier2SolverDevice::Cuda {
            ns_config.use_direct_solver = false;
            ns_config.linear_solver_backend = LinearSolverBackend::Cuda;
            ns_config.linear_solver_backend_preconditioner = BackendPreconditionerKind::JacobiSchur;
            ns_config.linear_solver_schur_preconditioner =
                BackendPressureSchurPreconditionerKind::Chebyshev;
            ns_config.linear_solver_cpu_preconditioner =
                LinearSolverCpuPreconditionerKind::JacobiSchur;
            ns_config.linear_solver_max_iterations = self.solver.max_iterations;
            ns_config.linear_solver_tolerance = self.solver.tolerance;
        }
        Ok(CoupledPortConfig {
            driver: self.case.driver(),
            ns_config,
            bdf: BdfConfig::bdf2(run.dt_s, run.num_steps)
                .with_output(fidelity.snapshot_interval_steps),
            voltage_fn: Box::new(move |time_s| {
                let primary = voltage_peak_v * (TAU * f1_hz * time_s).sin();
                second.map_or(primary, |(f2_hz, ratio)| {
                    primary + ratio * voltage_peak_v * (TAU * f2_hz * time_s).sin()
                })
            }),
            under_relaxation: 1.0,
            exterior_pressure_pa: 0.0,
        })
    }

    /// Evaluate one candidate: morph, solve, reduce.
    ///
    /// A pure function of (profile, drive, fidelity): the same inputs
    /// reproduce the same metrics up to linear-solver roundoff.
    ///
    /// # Errors
    ///
    /// Returns [`PortFlowError::InvalidConfig`] for illegal or off-envelope
    /// inputs (before any solve), [`PortFlowError::Mesh`] for an inverted
    /// morphed mesh, or [`PortFlowError::SolverDiverged`] for a diverged,
    /// non-finite, or unanalyzable run.
    pub fn evaluate(
        &self,
        profile: &SplinePortProfile,
        drive: &PortEvalDrive,
        fidelity: &PortEvalFidelity,
    ) -> Result<Tier2EvalOutcome, PortFlowError> {
        let run = drive.validate_run(fidelity)?;
        // The normalized identity must be finite: a zero-span profile
        // (r_min == R0 with free radii) has no defined `u` coordinates, so
        // it cannot be identified in metrics or optimiser history.
        if !profile.normalized().iter().all(|u| u.is_finite()) {
            return Err(PortFlowError::InvalidConfig(
                "Tier 2 candidate profile has a zero throat span, so its normalized identity is undefined"
                    .to_string(),
            ));
        }
        // Fail fast on geometry-derived quantities before the expensive solve.
        let f0_estimate_hz = self.tier0_tuning_hz(profile)?;
        let curvature_integral_m3 = profile.curvature_integral();
        if !curvature_integral_m3.is_finite() {
            return Err(PortFlowError::SolverDiverged {
                step: 0,
                time_s: 0.0,
            });
        }
        let throat_area_m2 = self.min_profile_area_m2(profile)?;
        let mesh = self.morphed_mesh(profile)?;
        let config = self.coupled_config(drive, fidelity)?;
        let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let p_space = H1Space::new(PolynomialDegree::P1);
        let result = run_coupled_port(&mesh, &v_space, &p_space, &config)?;
        // A CUDA request must never silently report CPU numbers: every
        // linear solve has to resolve to the requested device.
        if self.solver.device == Tier2SolverDevice::Cuda
            && result.linear_solves.iter().any(|solve| {
                !matches!(
                    solve.resolved_linear_solver_backend,
                    LinearSolverBackend::Cuda
                )
            })
        {
            return Err(PortFlowError::InvalidConfig(
                "Tier 2 requested the CUDA device but a linear solve fell back to CPU; \
                 build with the cuda feature and provide a CUDA driver"
                    .to_string(),
            ));
        }
        let reduction = TailReduction {
            result: &result,
            mesh: &mesh,
            profile,
            drive,
            fidelity,
            run: &run,
            f0_estimate_hz,
            curvature_integral_m3,
            throat_area_m2,
        };
        let metrics = self.reduce_tail(&reduction)?;
        let field_snapshots = if fidelity.snapshot_interval_steps == 0 {
            Vec::new()
        } else {
            result.field_snapshots.clone()
        };
        Ok(Tier2EvalOutcome {
            metrics,
            mesh,
            field_snapshots,
        })
    }

    /// Tier-0 compact tuning of the evaluated profile with the box compliance.
    fn tier0_tuning_hz(&self, profile: &SplinePortProfile) -> Result<f64, PortFlowError> {
        profile.validate()?;
        let config = Duct1dConfig {
            n_segments: DUCT1D_DEFAULT_SEGMENTS,
            thermoviscous: false,
            box_volume_m3: self.case.box_volume_m3,
        };
        tuning_frequency_hz(profile, self.case.medium(), &config)
    }

    /// Minimum duct cross-section `π·min(r)²` over the dense validation grid.
    fn min_profile_area_m2(&self, profile: &SplinePortProfile) -> Result<f64, PortFlowError> {
        let min_radius = profile
            .sampled_profile(PROFILE_VALIDATION_GRID_POINTS)
            .iter()
            .map(|station| station[1])
            .fold(f64::INFINITY, f64::min);
        if !(min_radius.is_finite() && min_radius > 0.0) {
            return Err(PortFlowError::SolverDiverged {
                step: 0,
                time_s: 0.0,
            });
        }
        Ok(PI * min_radius * min_radius)
    }

    /// Reduce the settled tail of a coupled run to candidate metrics.
    fn reduce_tail(
        &self,
        input: &TailReduction<'_>,
    ) -> Result<PortCandidateMetrics, PortFlowError> {
        let result = input.result;
        let run = input.run;
        let num_steps = run.num_steps;
        let last_time = result.time_s.last().copied().unwrap_or(0.0);
        let diverged = |step: usize| PortFlowError::SolverDiverged {
            step,
            time_s: last_time,
        };
        if result.steps_completed != num_steps {
            return Err(diverged(result.steps_completed));
        }
        // Every recorded series must hold exactly one entry per BDF step;
        // a short series means an incomplete run whose tail is misaligned.
        for length in [
            result.time_s.len(),
            result.port_flow_m3s.len(),
            result.exterior_cap_flow_m3s.len(),
            result.box_cap_outward_flow_m3s.len(),
            result.diaphragm_volume_velocity_m3s.len(),
            result.box_pressure_pa.len(),
            result.displacement_m.len(),
            result.current_a.len(),
            result.box_conservation_residual_pa.len(),
            result.energy_steps.len(),
            result.accepted_linear_relative_residual.len(),
        ] {
            if length != num_steps {
                return Err(PortFlowError::SolverDiverged {
                    step: length.min(num_steps.saturating_sub(1)),
                    time_s: last_time,
                });
            }
        }
        let tail_start = num_steps - run.tail_samples;
        let tail_range = tail_start..num_steps;
        for series in [
            &result.port_flow_m3s[tail_range.clone()],
            &result.box_cap_outward_flow_m3s[tail_range.clone()],
            &result.box_pressure_pa[tail_range.clone()],
            &result.displacement_m[tail_range.clone()],
            &result.box_conservation_residual_pa[tail_range.clone()],
            &result.accepted_linear_relative_residual[tail_range.clone()],
        ] {
            if series.iter().any(|value| !value.is_finite()) {
                return Err(diverged(num_steps - 1));
            }
        }
        if result.energy_steps[tail_range.clone()]
            .iter()
            .any(|step| !step.is_finite())
        {
            return Err(diverged(num_steps - 1));
        }

        let dt_s = run.dt_s;
        let f1_hz = input.drive.frequency_hz;
        let flow_tail = &result.port_flow_m3s[tail_range.clone()];
        let port_flow_rms_m3s =
            (flow_tail.iter().map(|q| q * q).sum::<f64>() / flow_tail.len() as f64).sqrt();
        let max_abs_flow = flow_tail.iter().map(|q| q.abs()).fold(0.0, f64::max);
        let port_flow_fundamental_amplitude_m3s = dft_amplitude(flow_tail, dt_s, f1_hz);
        // A silent (zero-flow) or non-finite tail cannot form metrics;
        // report divergence rather than a silent zero.
        for value in [port_flow_rms_m3s, port_flow_fundamental_amplitude_m3s] {
            if !(value.is_finite() && value > 0.0) {
                return Err(PortFlowError::SolverDiverged {
                    step: num_steps - 1,
                    time_s: last_time,
                });
            }
        }
        let box_pressure_amplitude_pa =
            dft_amplitude(&result.box_pressure_pa[tail_range.clone()], dt_s, f1_hz);
        let peak_excursion_m = result.displacement_m[tail_range.clone()]
            .iter()
            .map(|x| x.abs())
            .fold(0.0, f64::max);
        if !(box_pressure_amplitude_pa.is_finite() && peak_excursion_m.is_finite()) {
            return Err(diverged(num_steps - 1));
        }
        let thd = thd_fraction(flow_tail, dt_s, f1_hz, THD_HARMONICS);
        if !(thd.is_finite() && thd >= 0.0) {
            return Err(diverged(num_steps - 1));
        }
        let imd = match run.second_tone_hz {
            None => None,
            Some(f2_hz) => {
                if dft_amplitude(flow_tail, dt_s, f2_hz) <= 0.0 {
                    return Err(PortFlowError::SolverDiverged {
                        step: num_steps - 1,
                        time_s: last_time,
                    });
                }
                let value = imd_fraction(flow_tail, dt_s, f1_hz, f2_hz);
                if !(value.is_finite() && value >= 0.0) {
                    return Err(diverged(num_steps - 1));
                }
                Some(value)
            }
        };
        let compression = match &input.fidelity.compression_reference {
            None => None,
            Some(reference) => {
                let value = compression_db(
                    input.drive.voltage_peak_v,
                    port_flow_fundamental_amplitude_m3s,
                    reference.voltage_peak_v,
                    reference.port_flow_fundamental_amplitude_m3s,
                );
                if !value.is_finite() {
                    return Err(diverged(num_steps - 1));
                }
                Some(value)
            }
        };

        let medium = self.case.medium();
        let peak_mach_bulk = duct_bulk_mach_number(
            max_abs_flow,
            input.throat_area_m2,
            medium.sound_speed_m_per_s,
        );
        if !peak_mach_bulk.is_finite() {
            return Err(diverged(num_steps - 1));
        }
        let throat_velocity = port_flow_fundamental_amplitude_m3s / input.throat_area_m2;
        let turbulence_power_trend =
            free_turbulence_power_trend(throat_velocity, self.duct_length_m).map_err(|error| {
                PortFlowError::InvalidConfig(format!(
                    "Tier 2 turbulence proxy rejected its inputs: {error}"
                ))
            })?;

        let energy = result.energy_summary(tail_range.clone());
        if !energy.is_finite() {
            return Err(diverged(num_steps - 1));
        }
        let dissipation_tolerance =
            DISSIPATION_NEGATIVITY_RELATIVE_TOLERANCE * energy.energy_scale_j;
        let dissipations_nonnegative = result.energy_steps[tail_range.clone()]
            .iter()
            .all(|step| step.physical_dissipations_nonnegative(dissipation_tolerance));

        let box_tail = &result.box_cap_outward_flow_m3s[tail_range.clone()];
        let max_cap_mismatch = flow_tail
            .iter()
            .zip(box_tail.iter())
            .map(|(exterior, box_outward)| (exterior + box_outward).abs())
            .fold(0.0, f64::max);
        let cap_continuity_relative_error = max_cap_mismatch / max_abs_flow;
        let conservation_tail = &result.box_conservation_residual_pa[tail_range.clone()];
        let max_conservation = conservation_tail
            .iter()
            .map(|value| value.abs())
            .fold(0.0, f64::max);
        let max_abs_pressure = result.box_pressure_pa[tail_range.clone()]
            .iter()
            .map(|value| value.abs())
            .fold(0.0, f64::max);
        // Box pressure is driven, so its tail maximum is positive in
        // practice; when it is not, report the raw residual (still finite)
        // rather than dividing by zero.
        let box_conservation_relative_residual = if max_abs_pressure > 0.0 {
            max_conservation / max_abs_pressure
        } else {
            max_conservation
        };
        if !(cap_continuity_relative_error.is_finite()
            && box_conservation_relative_residual.is_finite())
        {
            return Err(diverged(num_steps - 1));
        }
        let max_accepted_linear_relative_residual = result.accepted_linear_relative_residual
            [tail_range.clone()]
        .iter()
        .copied()
        .fold(0.0, f64::max);
        let linear_residual_ok =
            max_accepted_linear_relative_residual <= TIER2_LINEAR_RESIDUAL_OK_MAX;

        let tail_start_time_s = result.time_s[tail_start];
        let (n_tail_snapshots, peak_mach_field, separation_fraction) =
            snapshot_field_proxies(SnapshotProxyInputs {
                mesh: input.mesh,
                duct_length_m: self.duct_length_m,
                snapshots: &result.field_snapshots,
                tail_start_time_s,
                port_flow: &result.port_flow_m3s,
                dt_s,
                max_abs_flow,
                sound_speed_m_per_s: medium.sound_speed_m_per_s,
            })?;

        let health = PortEvalHealth {
            dissipations_nonnegative,
            max_accepted_linear_relative_residual,
            linear_residual_ok,
            cap_continuity_relative_error,
            box_conservation_relative_residual,
            passed: dissipations_nonnegative && linear_residual_ok,
        };
        Ok(PortCandidateMetrics {
            schema_version: TIER2_EVAL_SCHEMA_VERSION,
            provenance: TIER2_EVAL_PROVENANCE.to_string(),
            profile_normalized: input.profile.normalized(),
            n_control_points: input.profile.n_control_points(),
            voltage_peak_v: input.drive.voltage_peak_v,
            frequency_hz: f1_hz,
            second_tone_frequency_hz: run.second_tone_hz,
            dt_s,
            steps_completed: num_steps,
            tail_samples: run.tail_samples,
            n_tail_snapshots,
            port_flow_rms_m3s,
            port_flow_fundamental_amplitude_m3s,
            box_pressure_amplitude_pa,
            peak_excursion_m,
            f0_estimate_hz: input.f0_estimate_hz,
            thd_fraction: thd,
            imd_fraction: imd,
            compression_db: compression,
            peak_mach_bulk,
            peak_mach_field,
            separation_fraction,
            turbulence_power_trend,
            curvature_integral_m3: input.curvature_integral_m3,
            energy_physical_relative_closure: energy.physical_relative_closure,
            energy_algebraic_relative_closure: energy.algebraic_relative_closure,
            energy_interface_relative_residual: energy.interface_relative_residual,
            energy_scale_j: energy.energy_scale_j,
            health,
        })
    }
}

/// Inputs for the snapshot-only field proxies.
struct SnapshotProxyInputs<'a> {
    mesh: &'a Mesh,
    duct_length_m: f64,
    snapshots: &'a [PortFlowFieldSnapshot],
    tail_start_time_s: f64,
    port_flow: &'a [f64],
    dt_s: f64,
    max_abs_flow: f64,
    sound_speed_m_per_s: f64,
}

/// Field peak Mach and duct separation fraction from tail snapshots.
///
/// Returns `(n_tail_snapshots, peak_mach_field, separation_fraction)`. The
/// separation fraction is the mean, over usable tail snapshots, of the
/// fraction of duct-band velocity DOFs whose axial component opposes the
/// instantaneous bulk flow. Snapshots on flow-reversal instants are skipped;
/// with no usable snapshot the fraction is `None`.
fn snapshot_field_proxies(
    inputs: SnapshotProxyInputs<'_>,
) -> Result<(usize, Option<f64>, Option<f64>), PortFlowError> {
    let tail: Vec<&PortFlowFieldSnapshot> = inputs
        .snapshots
        .iter()
        .filter(|snapshot| snapshot.time_s >= inputs.tail_start_time_s)
        .collect();
    if tail.is_empty() {
        return Ok((0, None, None));
    }
    // Scalar P2 coordinates share the vector space's scalar ordering, so entry
    // `i` locates velocity pair `(2i, 2i + 1)`.
    let scalar_space = H1Space::new(PolynomialDegree::P2);
    let coordinates = scalar_space.dof_coordinates(inputs.mesh).ok_or_else(|| {
        PortFlowError::InvalidConfig(
            "Tier 2 snapshot analysis needs P2 scalar DOF coordinates".to_string(),
        )
    })?;
    let n_scalar = coordinates.len();
    for snapshot in &tail {
        if snapshot.velocity_dofs_m_per_s.len() != 2 * n_scalar {
            return Err(PortFlowError::InvalidConfig(format!(
                "Tier 2 snapshot at t = {} s has {} velocity DOFs, expected {}",
                snapshot.time_s,
                snapshot.velocity_dofs_m_per_s.len(),
                2 * n_scalar
            )));
        }
        if snapshot
            .velocity_dofs_m_per_s
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(PortFlowError::SolverDiverged {
                step: inputs.port_flow.len().saturating_sub(1),
                time_s: snapshot.time_s,
            });
        }
    }
    let mut peak_speed = 0.0f64;
    for snapshot in &tail {
        // Lengths were validated above, so every pair index is in bounds.
        for scalar in 0..n_scalar {
            let speed = snapshot.velocity_dofs_m_per_s[2 * scalar]
                .hypot(snapshot.velocity_dofs_m_per_s[2 * scalar + 1]);
            peak_speed = peak_speed.max(speed);
        }
    }
    let peak_mach_field = peak_speed / inputs.sound_speed_m_per_s;
    if !peak_mach_field.is_finite() {
        return Err(PortFlowError::SolverDiverged {
            step: inputs.port_flow.len().saturating_sub(1),
            time_s: tail.last().map_or(0.0, |snapshot| snapshot.time_s),
        });
    }
    let duct: Vec<bool> = coordinates
        .iter()
        .map(|coordinate| {
            coordinate[0] >= -DUCT_MASK_TOLERANCE_M
                && coordinate[0] <= inputs.duct_length_m + DUCT_MASK_TOLERANCE_M
        })
        .collect();
    let duct_count = duct.iter().filter(|inside| **inside).count();
    if duct_count == 0 || inputs.port_flow.is_empty() {
        return Ok((tail.len(), Some(peak_mach_field), None));
    }
    let zero_flow = SEPARATION_ZERO_FLOW_RELATIVE_TOLERANCE * inputs.max_abs_flow;
    let mut fractions = Vec::new();
    for snapshot in &tail {
        let step = (snapshot.time_s / inputs.dt_s).round() as usize;
        let index = step.saturating_sub(1).min(inputs.port_flow.len() - 1);
        let bulk = inputs.port_flow[index];
        if bulk.abs() < zero_flow {
            continue;
        }
        let mut reversed = 0usize;
        for (scalar, inside) in duct.iter().enumerate() {
            if !inside {
                continue;
            }
            if snapshot.velocity_dofs_m_per_s[2 * scalar] * bulk < 0.0 {
                reversed += 1;
            }
        }
        fractions.push(reversed as f64 / duct_count as f64);
    }
    let separation = if fractions.is_empty() {
        None
    } else {
        Some(fractions.iter().sum::<f64>() / fractions.len() as f64)
    };
    Ok((tail.len(), Some(peak_mach_field), separation))
}

/// Bundled inputs for [`Tier2EvalContext::reduce_tail`].
struct TailReduction<'a> {
    result: &'a CoupledPortResult,
    mesh: &'a Mesh,
    profile: &'a SplinePortProfile,
    drive: &'a PortEvalDrive,
    fidelity: &'a PortEvalFidelity,
    run: &'a ValidatedRun,
    f0_estimate_hz: f64,
    curvature_integral_m3: f64,
    throat_area_m2: f64,
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::port_flow::baseline::PortOptimBaseline;
    use crate::port_flow::coupled::CoupledEnergyStep;

    use super::*;

    fn test_context() -> Tier2EvalContext {
        Tier2EvalContext::from_reference_case(PortOptimReferenceCase::default())
            .expect("test context must build")
    }

    fn straight_profile(context: &Tier2EvalContext) -> SplinePortProfile {
        context
            .candidate_profile(5, &[0.0, 0.0], ProfileOptions::default())
            .expect("straight test profile must build")
    }

    fn coarse_fidelity() -> PortEvalFidelity {
        PortEvalFidelity {
            samples_per_cycle: 48,
            total_cycles: 10,
            tail_cycles: 4,
            snapshot_interval_steps: 0,
            compression_reference: None,
        }
    }

    fn coarse_drive(context: &Tier2EvalContext) -> PortEvalDrive {
        PortEvalDrive {
            voltage_peak_v: 0.1,
            frequency_hz: context
                .case()
                .tier2_drive_frequency_hz()
                .expect("test drive frequency must resolve"),
            second_tone: None,
        }
    }

    fn assert_close(name: &str, a: f64, b: f64, rel: f64) {
        let scale = a.abs().max(b.abs()).max(1.0e-30);
        assert!(
            (a - b).abs() <= rel * scale,
            "{name}: {a:.6e} vs {b:.6e} exceeds relative tolerance {rel:.1e}"
        );
    }

    fn assert_opt_close(name: &str, a: Option<f64>, b: Option<f64>, rel: f64) {
        match (a, b) {
            (None, None) => {}
            (Some(x), Some(y)) => assert_close(name, x, y, rel),
            _ => panic!("{name}: option mismatch {a:?} vs {b:?}"),
        }
    }

    /// Compare the run-dependent physics observables of two metric sets.
    fn assert_physics_close(a: &PortCandidateMetrics, b: &PortCandidateMetrics, rel: f64) {
        assert_close(
            "port_flow_rms_m3s",
            a.port_flow_rms_m3s,
            b.port_flow_rms_m3s,
            rel,
        );
        assert_close(
            "port_flow_fundamental_amplitude_m3s",
            a.port_flow_fundamental_amplitude_m3s,
            b.port_flow_fundamental_amplitude_m3s,
            rel,
        );
        assert_close(
            "box_pressure_amplitude_pa",
            a.box_pressure_amplitude_pa,
            b.box_pressure_amplitude_pa,
            rel,
        );
        assert_close(
            "peak_excursion_m",
            a.peak_excursion_m,
            b.peak_excursion_m,
            rel,
        );
        assert_close("thd_fraction", a.thd_fraction, b.thd_fraction, rel);
        assert_close("peak_mach_bulk", a.peak_mach_bulk, b.peak_mach_bulk, rel);
        assert_close(
            "turbulence_power_trend",
            a.turbulence_power_trend,
            b.turbulence_power_trend,
            rel,
        );
        assert_close(
            "energy_physical_relative_closure",
            a.energy_physical_relative_closure,
            b.energy_physical_relative_closure,
            rel,
        );
        assert_close(
            "energy_algebraic_relative_closure",
            a.energy_algebraic_relative_closure,
            b.energy_algebraic_relative_closure,
            rel,
        );
        assert_close(
            "energy_interface_relative_residual",
            a.energy_interface_relative_residual,
            b.energy_interface_relative_residual,
            rel,
        );
        assert_close("energy_scale_j", a.energy_scale_j, b.energy_scale_j, rel);
        assert_close(
            "cap_continuity_relative_error",
            a.health.cap_continuity_relative_error,
            b.health.cap_continuity_relative_error,
            rel,
        );
        assert_close(
            "box_conservation_relative_residual",
            a.health.box_conservation_relative_residual,
            b.health.box_conservation_relative_residual,
            rel,
        );
        assert_close(
            "max_accepted_linear_relative_residual",
            a.health.max_accepted_linear_relative_residual,
            b.health.max_accepted_linear_relative_residual,
            rel,
        );
        assert_opt_close("imd_fraction", a.imd_fraction, b.imd_fraction, rel);
        assert_opt_close("compression_db", a.compression_db, b.compression_db, rel);
        assert_opt_close("peak_mach_field", a.peak_mach_field, b.peak_mach_field, rel);
        assert_opt_close(
            "separation_fraction",
            a.separation_fraction,
            b.separation_fraction,
            rel,
        );
        assert_eq!(
            a.health.dissipations_nonnegative, b.health.dissipations_nonnegative,
            "dissipation flag mismatch"
        );
        assert_eq!(
            a.health.linear_residual_ok, b.health.linear_residual_ok,
            "linear-residual flag mismatch"
        );
        assert_eq!(a.health.passed, b.health.passed, "health pass mismatch");
    }

    /// Compare two metric sets from identical inputs field by field.
    fn assert_metrics_replica(a: &PortCandidateMetrics, b: &PortCandidateMetrics, rel: f64) {
        assert_eq!(a.schema_version, TIER2_EVAL_SCHEMA_VERSION);
        assert_eq!(a.provenance, TIER2_EVAL_PROVENANCE);
        assert_eq!(a.profile_normalized, b.profile_normalized);
        assert_eq!(a.n_control_points, b.n_control_points);
        assert_eq!(a.voltage_peak_v, b.voltage_peak_v);
        assert_eq!(a.frequency_hz, b.frequency_hz);
        assert_eq!(a.second_tone_frequency_hz, b.second_tone_frequency_hz);
        assert_eq!(a.dt_s, b.dt_s);
        assert_eq!(a.steps_completed, b.steps_completed);
        assert_eq!(a.tail_samples, b.tail_samples);
        assert_eq!(a.n_tail_snapshots, b.n_tail_snapshots);
        assert_eq!(a.f0_estimate_hz, b.f0_estimate_hz);
        assert_eq!(a.curvature_integral_m3, b.curvature_integral_m3);
        assert_physics_close(a, b, rel);
    }

    #[test]
    fn straight_config_matches_baseline_coupled_config() {
        let context = test_context();
        let case = PortOptimReferenceCase::default();
        let voltage_peak_v = 0.1;
        let drive = PortEvalDrive {
            voltage_peak_v,
            frequency_hz: case.tier2_drive_frequency_hz().unwrap(),
            second_tone: None,
        };
        let fidelity = PortEvalFidelity::from_reference_case(&case);
        let eval_config = context.coupled_config(&drive, &fidelity).unwrap();
        let base_config = case.coupled_config(voltage_peak_v).unwrap();
        assert_eq!(eval_config.bdf.dt, base_config.bdf.dt);
        assert_eq!(eval_config.bdf.num_steps, base_config.bdf.num_steps);
        assert_eq!(eval_config.bdf.order, 2);
        assert_eq!(eval_config.bdf.order, base_config.bdf.order);
        assert_eq!(eval_config.bdf.output_interval, 0);
        assert_eq!(eval_config.ns_config.mu, base_config.ns_config.mu);
        assert_eq!(eval_config.ns_config.rho, base_config.ns_config.rho);
        assert!(!eval_config.ns_config.use_residual_stabilisation);
        assert_eq!(eval_config.under_relaxation, 1.0);
        assert_eq!(eval_config.exterior_pressure_pa, 0.0);
        assert_eq!(eval_config.driver.mms_kg, base_config.driver.mms_kg);
        assert_eq!(
            eval_config.driver.rms_kg_per_s,
            base_config.driver.rms_kg_per_s
        );
        assert_eq!(
            eval_config.driver.kms_n_per_m,
            base_config.driver.kms_n_per_m
        );
        assert_eq!(eval_config.driver.sd_m2, base_config.driver.sd_m2);
        assert_eq!(
            eval_config.driver.box_volume_m3,
            base_config.driver.box_volume_m3
        );
        for time_s in [
            0.0,
            eval_config.bdf.dt / 2.0,
            1.7 * eval_config.bdf.dt,
            0.01,
            0.1,
        ] {
            assert_eq!(
                (eval_config.voltage_fn)(time_s),
                (base_config.voltage_fn)(time_s),
                "drive waveform mismatch at t = {time_s}"
            );
        }
    }

    #[test]
    fn straight_baseline_mesh_matches_slot_fixture() {
        let context = test_context();
        let slot =
            build_slot_port_mesh(&PortOptimReferenceCase::default().slot_geometry()).unwrap();
        assert_eq!(
            format!("{:?}", context.baseline_mesh()),
            format!("{slot:?}"),
            "Tier-2 baseline mesh drifted from the Step-0 slot fixture"
        );
    }

    #[test]
    fn candidate_profile_lives_on_the_slot_envelope() {
        let context = test_context();
        let profile = context
            .candidate_profile(5, &[0.2, 0.6], ProfileOptions::default())
            .unwrap();
        assert_eq!(profile.length_m(), context.duct_length_m());
        assert_eq!(profile.bore_radius_m(), context.bore_radius_m());
        assert_eq!(
            profile.throat_radius_m(),
            TIER2_THROAT_FRACTION_OF_BORE * context.bore_radius_m()
        );
        // Normalised radii round-trip through r = R0 - u·span.
        for (measured, expected) in profile.normalized().iter().zip([0.2, 0.6]) {
            assert_close("normalized round trip", *measured, expected, 1.0e-12);
        }
        assert!(
            context
                .candidate_profile(5, &[0.2], ProfileOptions::default())
                .is_err()
        );
        assert!(
            context
                .candidate_profile(5, &[0.2, 1.5], ProfileOptions::default())
                .is_err()
        );
    }

    #[test]
    fn tier0_estimate_uses_the_screening_configuration() {
        let context = test_context();
        let profile = straight_profile(&context);
        let via_helper = context.tier0_tuning_hz(&profile).unwrap();
        let direct = tuning_frequency_hz(
            &profile,
            context.case().medium(),
            &Duct1dConfig {
                n_segments: DUCT1D_DEFAULT_SEGMENTS,
                thermoviscous: false,
                box_volume_m3: context.case().box_volume_m3,
            },
        )
        .unwrap();
        assert_eq!(via_helper, direct);
        assert!(via_helper.is_finite() && via_helper > 0.0);
        println!("slot-scale straight Tier-0 tuning estimate: {via_helper:.6} Hz");
    }

    #[test]
    fn morph_is_identity_for_straight_and_moves_for_dipped() {
        let context = test_context();
        let straight = context.morphed_mesh(&straight_profile(&context)).unwrap();
        assert_eq!(
            format!("{straight:?}"),
            format!("{:?}", context.baseline_mesh()),
            "straight morph must return the baseline clone untouched"
        );
        let dipped_profile = context
            .candidate_profile(5, &[0.3, 0.5], ProfileOptions::default())
            .unwrap();
        let dipped = context.morphed_mesh(&dipped_profile).unwrap();
        assert_eq!(dipped.nodes.len(), context.baseline_mesh().nodes.len());
        assert_eq!(
            dipped.elements.len(),
            context.baseline_mesh().elements.len()
        );
        assert_eq!(
            dipped.boundaries.len(),
            context.baseline_mesh().boundaries.len()
        );
        let max_dy = dipped
            .nodes
            .iter()
            .zip(context.baseline_mesh().nodes.iter())
            .map(|(moved, base)| (moved.y - base.y).abs())
            .fold(0.0, f64::max);
        assert!(
            max_dy > 1.0e-12,
            "dipped morph left every node in place (max |dy| = {max_dy:.3e})"
        );
        println!("dipped morph max |dy| = {max_dy:.6e} m");
    }

    /// Synthetic settled tail with known content: fundamental plus 2% second
    /// harmonic in the port flow, pure sines elsewhere, exact cap continuity.
    fn synthetic_result(
        f1_hz: f64,
        samples_per_cycle: usize,
        total_cycles: usize,
    ) -> CoupledPortResult {
        let n = total_cycles * samples_per_cycle;
        let dt = 1.0 / (f1_hz * samples_per_cycle as f64);
        let amplitude = 1.0e-4;
        let time_s: Vec<f64> = (1..=n).map(|step| step as f64 * dt).collect();
        let port_flow_m3s: Vec<f64> = time_s
            .iter()
            .map(|time| {
                amplitude * (TAU * f1_hz * time).sin()
                    + 0.02 * amplitude * (TAU * 2.0 * f1_hz * time).sin()
            })
            .collect();
        CoupledPortResult {
            time_s,
            port_flow_m3s: port_flow_m3s.clone(),
            exterior_cap_flow_m3s: port_flow_m3s.clone(),
            box_cap_outward_flow_m3s: port_flow_m3s.iter().map(|q| -q).collect(),
            diaphragm_volume_velocity_m3s: vec![0.0; n],
            box_pressure_pa: (1..=n)
                .map(|step| 10.0 * (TAU * f1_hz * step as f64 * dt).sin())
                .collect(),
            displacement_m: (1..=n)
                .map(|step| 1.0e-4 * (TAU * f1_hz * step as f64 * dt).sin())
                .collect(),
            current_a: vec![0.0; n],
            box_conservation_residual_pa: vec![0.0; n],
            energy_steps: vec![CoupledEnergyStep::default(); n],
            linear_solves: Vec::new(),
            accepted_linear_relative_residual: vec![1.0e-13; n],
            field_snapshots: Vec::new(),
            final_flow_state: None,
            steps_completed: n,
        }
    }

    #[test]
    fn reduce_synthetic_tail_matches_closed_form() {
        let context = test_context();
        let profile = straight_profile(&context);
        let f1_hz = 50.0;
        let (samples_per_cycle, total_cycles, tail_cycles) = (64, 6, 2);
        let result = synthetic_result(f1_hz, samples_per_cycle, total_cycles);
        let drive = PortEvalDrive {
            voltage_peak_v: 0.1,
            frequency_hz: f1_hz,
            second_tone: None,
        };
        let fidelity = PortEvalFidelity {
            samples_per_cycle,
            total_cycles,
            tail_cycles,
            snapshot_interval_steps: 0,
            compression_reference: Some(SmallSignalReference {
                voltage_peak_v: 0.05,
                port_flow_fundamental_amplitude_m3s: 0.5e-4,
            }),
        };
        let run = drive.validate_run(&fidelity).unwrap();
        let reduction = TailReduction {
            result: &result,
            mesh: context.baseline_mesh(),
            profile: &profile,
            drive: &drive,
            fidelity: &fidelity,
            run: &run,
            f0_estimate_hz: context.tier0_tuning_hz(&profile).unwrap(),
            curvature_integral_m3: profile.curvature_integral(),
            throat_area_m2: context.min_profile_area_m2(&profile).unwrap(),
        };
        let metrics = context.reduce_tail(&reduction).unwrap();

        assert_eq!(metrics.schema_version, TIER2_EVAL_SCHEMA_VERSION);
        assert_eq!(metrics.provenance, TIER2_EVAL_PROVENANCE);
        assert_eq!(metrics.profile_normalized, vec![0.0, 0.0]);
        assert_eq!(metrics.n_control_points, 5);
        assert_eq!(metrics.tail_samples, tail_cycles * samples_per_cycle);
        assert_eq!(metrics.steps_completed, total_cycles * samples_per_cycle);
        assert_eq!(metrics.n_tail_snapshots, 0);
        assert_eq!(metrics.imd_fraction, None);
        assert_eq!(metrics.peak_mach_field, None);
        assert_eq!(metrics.separation_fraction, None);

        // RMS of A·sin + 0.02A·sin(2ωt) over complete cycles.
        let expected_rms = 1.0e-4 * (1.0 + 0.02f64.powi(2)).sqrt() / 2.0f64.sqrt();
        assert_close(
            "synthetic rms",
            metrics.port_flow_rms_m3s,
            expected_rms,
            1.0e-9,
        );
        assert_close(
            "synthetic fundamental",
            metrics.port_flow_fundamental_amplitude_m3s,
            1.0e-4,
            1.0e-9,
        );
        assert_close("synthetic thd", metrics.thd_fraction, 0.02, 1.0e-6);
        assert_close(
            "synthetic box pressure",
            metrics.box_pressure_amplitude_pa,
            10.0,
            1.0e-9,
        );
        assert!(
            metrics.peak_excursion_m <= 1.0e-4 && metrics.peak_excursion_m > 0.999e-4,
            "sampled excursion peak {} misses the 1e-4 sine peak",
            metrics.peak_excursion_m
        );
        // Equal-voltage doubling against the half-amplitude reference.
        // Absolute check: compression is exactly 0 dB in closed form, so a
        // relative tolerance has no scale.
        assert!(
            metrics.compression_db.unwrap().abs() < 1.0e-9,
            "synthetic compression should vanish, got {} dB",
            metrics.compression_db.unwrap()
        );
        // Bulk Mach applies the throat area and sound speed to the tail peak.
        let tail_peak = result.port_flow_m3s[result.port_flow_m3s.len() - metrics.tail_samples..]
            .iter()
            .map(|q| q.abs())
            .fold(0.0, f64::max);
        let expected_mach = tail_peak / (PI * 0.01f64.powi(2)) / 343.0;
        assert_close(
            "synthetic mach",
            metrics.peak_mach_bulk,
            expected_mach,
            1.0e-12,
        );
        assert!(metrics.curvature_integral_m3.abs() < 1.0e-12);
        assert!(metrics.turbulence_power_trend.is_finite() && metrics.turbulence_power_trend > 0.0);
        assert_eq!(metrics.energy_scale_j, 1.0e-18);
        assert_eq!(metrics.energy_physical_relative_closure, 0.0);
        assert_eq!(metrics.energy_algebraic_relative_closure, 0.0);
        assert_eq!(metrics.energy_interface_relative_residual, 0.0);
        assert_eq!(metrics.health.cap_continuity_relative_error, 0.0);
        assert_eq!(metrics.health.box_conservation_relative_residual, 0.0);
        assert!(metrics.health.passed);

        // Metrics must survive the Step-6 history JSON round trip exactly.
        let json = serde_json::to_string(&metrics).unwrap();
        let decoded: PortCandidateMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, metrics);
    }

    #[test]
    fn invalid_inputs_are_rejected_before_any_solve() {
        let context = test_context();
        let straight = straight_profile(&context);
        let drive = coarse_drive(&context);
        let fidelity = coarse_fidelity();
        let invalid_profiles = [
            SplinePortProfile::from_normalized(
                5,
                0.2,
                context.bore_radius_m(),
                0.6 * context.bore_radius_m(),
                &[0.0, 0.0],
                ProfileOptions::default(),
            )
            .unwrap(),
            SplinePortProfile::from_normalized(
                5,
                context.duct_length_m(),
                0.025,
                0.015,
                &[0.0, 0.0],
                ProfileOptions::default(),
            )
            .unwrap(),
            // Zero throat span: a legal straight profile whose normalized
            // identity is undefined (0/0), so the harness rejects it.
            SplinePortProfile::from_normalized(
                5,
                context.duct_length_m(),
                context.bore_radius_m(),
                context.bore_radius_m(),
                &[0.0, 0.0],
                ProfileOptions::default(),
            )
            .unwrap(),
        ];
        for profile in &invalid_profiles {
            assert!(
                matches!(
                    context.evaluate(profile, &drive, &fidelity),
                    Err(PortFlowError::InvalidConfig(_))
                ),
                "off-envelope profile was not rejected pre-solve"
            );
        }
        let invalid_drives = [
            PortEvalDrive {
                voltage_peak_v: 0.0,
                ..drive
            },
            PortEvalDrive {
                voltage_peak_v: -1.0,
                ..drive
            },
            PortEvalDrive {
                voltage_peak_v: f64::NAN,
                ..drive
            },
            PortEvalDrive {
                frequency_hz: 0.0,
                ..drive
            },
            PortEvalDrive {
                frequency_hz: f64::NAN,
                ..drive
            },
            PortEvalDrive {
                second_tone: Some(SecondTone {
                    numerator: 5,
                    denominator: 5,
                    amplitude_ratio: 0.5,
                }),
                ..drive
            },
            PortEvalDrive {
                second_tone: Some(SecondTone {
                    numerator: 4,
                    denominator: 5,
                    amplitude_ratio: 0.5,
                }),
                ..drive
            },
            PortEvalDrive {
                second_tone: Some(SecondTone {
                    numerator: 0,
                    denominator: 4,
                    amplitude_ratio: 0.5,
                }),
                ..drive
            },
            PortEvalDrive {
                second_tone: Some(SecondTone {
                    numerator: 5,
                    denominator: 4,
                    amplitude_ratio: 0.0,
                }),
                ..drive
            },
        ];
        for bad_drive in &invalid_drives {
            assert!(
                matches!(
                    context.evaluate(&straight, bad_drive, &fidelity),
                    Err(PortFlowError::InvalidConfig(_))
                ),
                "invalid drive {bad_drive:?} was not rejected pre-solve"
            );
        }
        // Two-tone tail must hold integer common periods: 3 cycles of a 5/4
        // ratio leaves a fractional common period.
        let misaligned = PortEvalFidelity {
            tail_cycles: 3,
            ..fidelity.clone()
        };
        let two_tone = PortEvalDrive {
            second_tone: Some(SecondTone {
                numerator: 5,
                denominator: 4,
                amplitude_ratio: 0.5,
            }),
            ..drive
        };
        assert!(matches!(
            context.evaluate(&straight, &two_tone, &misaligned),
            Err(PortFlowError::InvalidConfig(_))
        ));
        let invalid_fidelities = [
            PortEvalFidelity {
                samples_per_cycle: 4,
                ..fidelity.clone()
            },
            PortEvalFidelity {
                tail_cycles: 0,
                ..fidelity.clone()
            },
            PortEvalFidelity {
                tail_cycles: 10,
                ..fidelity.clone()
            },
            PortEvalFidelity {
                tail_cycles: 11,
                ..fidelity.clone()
            },
            PortEvalFidelity {
                snapshot_interval_steps: fidelity.tail_cycles * fidelity.samples_per_cycle + 1,
                ..fidelity.clone()
            },
            PortEvalFidelity {
                compression_reference: Some(SmallSignalReference {
                    voltage_peak_v: 0.05,
                    port_flow_fundamental_amplitude_m3s: 0.0,
                }),
                ..fidelity.clone()
            },
            PortEvalFidelity {
                compression_reference: Some(SmallSignalReference {
                    voltage_peak_v: -0.05,
                    port_flow_fundamental_amplitude_m3s: 0.5e-4,
                }),
                ..fidelity.clone()
            },
        ];
        for bad_fidelity in &invalid_fidelities {
            assert!(
                matches!(
                    context.evaluate(&straight, &drive, bad_fidelity),
                    Err(PortFlowError::InvalidConfig(_))
                ),
                "invalid fidelity {bad_fidelity:?} was not rejected pre-solve"
            );
        }
        // Nyquist: 8 samples/cycle at ~159 Hz cannot analyse the 5th harmonic.
        let nyquist = PortEvalFidelity {
            samples_per_cycle: 8,
            ..fidelity.clone()
        };
        match context.evaluate(&straight, &drive, &nyquist) {
            Err(PortFlowError::InvalidConfig(message)) => {
                assert!(message.contains("Nyquist"), "unexpected message: {message}");
            }
            other => panic!("expected a Nyquist InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn straight_harness_matches_direct_coupled_run() {
        let context = test_context();
        let profile = straight_profile(&context);
        let drive = coarse_drive(&context);
        let fidelity = coarse_fidelity();
        let outcome = context.evaluate(&profile, &drive, &fidelity).unwrap();
        assert!(outcome.field_snapshots.is_empty());
        assert_eq!(outcome.metrics.imd_fraction, None);
        assert_eq!(outcome.metrics.compression_db, None);

        let config = context.coupled_config(&drive, &fidelity).unwrap();
        let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let p_space = H1Space::new(PolynomialDegree::P1);
        let direct =
            run_coupled_port(context.baseline_mesh(), &v_space, &p_space, &config).unwrap();
        let run = drive.validate_run(&fidelity).unwrap();
        let reduction = TailReduction {
            result: &direct,
            mesh: context.baseline_mesh(),
            profile: &profile,
            drive: &drive,
            fidelity: &fidelity,
            run: &run,
            f0_estimate_hz: outcome.metrics.f0_estimate_hz,
            curvature_integral_m3: outcome.metrics.curvature_integral_m3,
            throat_area_m2: context.min_profile_area_m2(&profile).unwrap(),
        };
        let expected = context.reduce_tail(&reduction).unwrap();
        assert_metrics_replica(&outcome.metrics, &expected, 1.0e-9);
        println!(
            "straight coarse: Q_rms={:.6e} m3/s, Q_fund={:.6e} m3/s, THD={:.6}%, \
             Mach_bulk={:.6e}, cap={:.3e}, box_cons={:.3e}, lin_res={:.3e}, \
             energy_phys={:.3e}, energy_alg={:.3e}, energy_iface={:.3e}, health={}",
            outcome.metrics.port_flow_rms_m3s,
            outcome.metrics.port_flow_fundamental_amplitude_m3s,
            100.0 * outcome.metrics.thd_fraction,
            outcome.metrics.peak_mach_bulk,
            outcome.metrics.health.cap_continuity_relative_error,
            outcome.metrics.health.box_conservation_relative_residual,
            outcome.metrics.health.max_accepted_linear_relative_residual,
            outcome.metrics.energy_physical_relative_closure,
            outcome.metrics.energy_algebraic_relative_closure,
            outcome.metrics.energy_interface_relative_residual,
            outcome.metrics.health.passed,
        );
        assert!(
            outcome.metrics.health.passed,
            "coarse straight run tripped a health flag: {:?}",
            outcome.metrics.health
        );
    }

    #[test]
    fn halved_dt_changes_metrics_within_tolerance() {
        let context = test_context();
        let profile = straight_profile(&context);
        let drive = coarse_drive(&context);
        let coarse = coarse_fidelity();
        let fine = PortEvalFidelity {
            samples_per_cycle: 2 * coarse.samples_per_cycle,
            ..coarse.clone()
        };
        let a = context.evaluate(&profile, &drive, &coarse).unwrap().metrics;
        let b = context.evaluate(&profile, &drive, &fine).unwrap().metrics;
        let rel_diff = |x: f64, y: f64| (x - y).abs() / x.abs().max(y.abs()).max(1.0e-30);
        println!(
            "halved-dt deltas: Q_rms={:.3e}, Q_fund={:.3e}, Mach={:.3e}, THD {:.3e} vs {:.3e}",
            rel_diff(a.port_flow_rms_m3s, b.port_flow_rms_m3s),
            rel_diff(
                a.port_flow_fundamental_amplitude_m3s,
                b.port_flow_fundamental_amplitude_m3s
            ),
            rel_diff(a.peak_mach_bulk, b.peak_mach_bulk),
            a.thd_fraction,
            b.thd_fraction,
        );
        // Coarse screening tolerance; the production 2% gate belongs to the
        // Step-9 promotion ladder, not to this 10-cycle smoke run. THD is
        // asserted small in absolute terms rather than relatively: near zero
        // distortion a relative comparison has no scale, and 48 samples/cycle
        // only marginally resolves the 5th harmonic (finer sampling reports
        // the larger, better-resolved value).
        assert_close(
            "halved-dt Q_rms",
            a.port_flow_rms_m3s,
            b.port_flow_rms_m3s,
            0.05,
        );
        assert_close(
            "halved-dt Q_fundamental",
            a.port_flow_fundamental_amplitude_m3s,
            b.port_flow_fundamental_amplitude_m3s,
            0.05,
        );
        assert_close(
            "halved-dt box pressure",
            a.box_pressure_amplitude_pa,
            b.box_pressure_amplitude_pa,
            0.05,
        );
        assert_close("halved-dt Mach", a.peak_mach_bulk, b.peak_mach_bulk, 0.05);
        assert!(
            a.thd_fraction < 0.01 && b.thd_fraction < 0.01,
            "low-drive THD should stay small on both grids, got {} and {}",
            a.thd_fraction,
            b.thd_fraction
        );
    }

    #[test]
    fn two_tone_imd_and_snapshots_populated() {
        let context = test_context();
        let profile = straight_profile(&context);
        let drive = PortEvalDrive {
            second_tone: Some(SecondTone {
                numerator: 5,
                denominator: 4,
                amplitude_ratio: 0.5,
            }),
            ..coarse_drive(&context)
        };
        let fidelity = PortEvalFidelity {
            snapshot_interval_steps: coarse_fidelity().samples_per_cycle,
            ..coarse_fidelity()
        };
        let outcome = context.evaluate(&profile, &drive, &fidelity).unwrap();
        let metrics = &outcome.metrics;
        let expected_f2 = 1.25 * drive.frequency_hz;
        assert_eq!(metrics.second_tone_frequency_hz, Some(expected_f2));
        let imd = metrics.imd_fraction.expect("two-tone run must report IMD");
        assert!(imd.is_finite() && imd >= 0.0, "IMD = {imd}");
        assert_eq!(outcome.field_snapshots.len(), 10);
        assert!(
            metrics.n_tail_snapshots >= 1,
            "no snapshot landed in the analysed tail"
        );
        let field_mach = metrics
            .peak_mach_field
            .expect("snapshots must yield a field Mach");
        assert!(field_mach.is_finite() && field_mach > 0.0);
        if let Some(separation) = metrics.separation_fraction {
            assert!(
                (0.0..=1.0).contains(&separation),
                "separation fraction out of range: {separation}"
            );
        }
        println!(
            "two-tone: f2={expected_f2:.3} Hz, IMD={imd:.6e}, tail_snaps={}, \
             Mach_field={field_mach:.6e}, separation={:?}",
            metrics.n_tail_snapshots, metrics.separation_fraction,
        );
    }

    #[test]
    fn dipped_candidate_evaluates_deterministically() {
        let context = test_context();
        let profile = context
            .candidate_profile(5, &[0.3, 0.5], ProfileOptions::default())
            .unwrap();
        let drive = coarse_drive(&context);
        let fidelity = coarse_fidelity();
        let first = context
            .evaluate(&profile, &drive, &fidelity)
            .unwrap()
            .metrics;
        let second = context
            .evaluate(&profile, &drive, &fidelity)
            .unwrap()
            .metrics;
        assert_metrics_replica(&first, &second, 1.0e-9);
        println!(
            "dipped u=[0.3, 0.5]: Q_rms={:.6e} m3/s, THD={:.6}%, f0={:.6} Hz, health={}",
            first.port_flow_rms_m3s,
            100.0 * first.thd_fraction,
            first.f0_estimate_hz,
            first.health.passed,
        );
    }

    /// Straight production evaluation against the frozen Step-0 archive.
    ///
    /// Replays one Tier-2 baseline level (lowest drive, 252 samples/cycle, 40
    /// cycles) through the harness and compares against
    /// `fixtures/port_optim_baseline_v1.json`. Release-only: debug builds are
    /// rejected rather than run for many minutes.
    #[test]
    #[ignore = "production-fidelity Tier-2 replay; run explicitly in release mode"]
    fn straight_production_eval_matches_archived_baseline()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if cfg!(debug_assertions) {
            return Err("run the production Tier-2 replay in release mode \
                (cargo test -p sonium-sim-speaker-modeling --release \
                --lib port_flow::evaluate -- --ignored --nocapture)"
                .into());
        }
        let case = PortOptimReferenceCase::default();
        let context = Tier2EvalContext::from_reference_case(case.clone())?;
        let profile = context.candidate_profile(5, &[0.0, 0.0], ProfileOptions::default())?;
        let voltage_peak_v = case.tier2_drives_v_peak[0];
        let drive = PortEvalDrive {
            voltage_peak_v,
            frequency_hz: case.tier2_drive_frequency_hz()?,
            second_tone: None,
        };
        let fidelity = PortEvalFidelity::from_reference_case(&case);
        let metrics = context.evaluate(&profile, &drive, &fidelity)?.metrics;

        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/port_optim_baseline_v1.json");
        let baseline: PortOptimBaseline = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
        let level = baseline
            .tier2
            .first()
            .ok_or("archive has no Tier 2 levels")?;
        assert_eq!(level.voltage_peak_v, voltage_peak_v);
        assert_eq!(metrics.steps_completed, case.num_steps());

        // Archive Mach/trend use the slot area; recompute archive-style from
        // the harness fundamental for an apples-to-apples comparison.
        let medium = case.medium();
        let slot_mach = duct_bulk_mach_number(
            metrics.port_flow_fundamental_amplitude_m3s,
            case.tier2_slot_area_m2(),
            medium.sound_speed_m_per_s,
        );
        let slot_trend = free_turbulence_power_trend(
            metrics.port_flow_fundamental_amplitude_m3s / case.tier2_slot_area_m2(),
            case.tier2_duct_length_m,
        )
        .map_err(|error| format!("turbulence proxy failed: {error}"))?;
        for (name, measured, archived) in [
            (
                "Q_fundamental",
                metrics.port_flow_fundamental_amplitude_m3s,
                level.port_flow_fundamental_amplitude_m3s,
            ),
            ("Q_rms", metrics.port_flow_rms_m3s, level.port_flow_rms_m3s),
            (
                "p_box",
                metrics.box_pressure_amplitude_pa,
                level.box_pressure_amplitude_pa,
            ),
            (
                "excursion",
                metrics.peak_excursion_m,
                level.peak_excursion_m,
            ),
            ("THD", metrics.thd_fraction, level.thd_fraction),
            ("slot_mach", slot_mach, level.duct_bulk_mach),
            ("slot_trend", slot_trend, level.turbulence_power_trend),
            (
                "energy_physical",
                metrics.energy_physical_relative_closure,
                level.energy_physical_relative_closure,
            ),
            (
                "energy_algebraic",
                metrics.energy_algebraic_relative_closure,
                level.energy_algebraic_relative_closure,
            ),
            (
                "energy_interface",
                metrics.energy_interface_relative_residual,
                level.energy_interface_relative_residual,
            ),
            ("energy_scale", metrics.energy_scale_j, level.energy_scale_j),
        ] {
            let scale = measured.abs().max(archived.abs()).max(1.0e-30);
            let rel = (measured - archived).abs() / scale;
            println!(
                "archive replay {name}: harness={measured:.6e} archive={archived:.6e} rel={rel:.3e}"
            );
            assert!(
                rel <= 1.0e-6,
                "{name} replay mismatch: {measured:.6e} vs archived {archived:.6e}"
            );
        }
        assert!(metrics.health.passed, "replay tripped a health flag");
        Ok(())
    }
}
