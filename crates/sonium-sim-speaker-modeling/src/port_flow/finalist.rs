//! Step-8 finalist machinery: snapshot noise comparison + nightly 3D spot-check
//! (PORT-OPTIM.md Step 8).
//!
//! This module holds the reusable pieces of the finalist leg: snapshot-enabled
//! Tier-2 fidelity, the 1 m on-axis [`PortNoiseSpec`](crate::PortNoiseSpec)
//! with automatic Lighthill/FW–H selection by `ka`, the winner-vs-straight
//! integrated-noise-floor table, the campaign-scale spline round-port geometry
//! for the 3D spot-check, the 3D coupled spot-check runner, the 10%
//! agreement gate, and the [`Step8Report`] evidence archive with its
//! schema/provenance gate.
//!
//! # What runs where
//!
//! The expensive solves live in the ignored `port_optim_step8` test
//! (release-only, like the Step-7 campaign). This module stays solver-free
//! except for [`run_3d_spot_check`], which the ignored test calls; everything
//! else here is configuration, comparison, and archiving logic covered by fast
//! unit tests.
//!
//! # Why ratios, not absolute flows, cross the 2D/3D boundary
//!
//! Tier 2 runs on the 2D slot surrogate (unit-depth flows, 10 mm bore) while
//! the spot-check runs the 3D round port at campaign scale (25 mm bore), so
//! absolute `Q_rms` values are incomparable across tiers. The transfer check
//! therefore compares winner/straight *ratios*: `Q_w/Q_s` in 3D against
//! `Q_w/Q_s` in 2D at the same drive voltage, and likewise for THD. A
//! relative disagreement above [`STEP8_SPOTCHECK_DISAGREE_REL_TOL`] on either
//! ratio returns [`SpotVerdict::Investigate`]; the verdict is recorded, never
//! used to adjust a claim.
//!
//! Validated envelope: 2D snapshot-noise evidence at the evaluated fidelity,
//! plus nightly-tier 3D spot evidence. The 3D leg reuses the nightly iterative
//! solver path (`SPEAKER_PORT_3D=1` family) and must not be reported as a
//! passed gate from any 2D result.

use std::f64::consts::TAU;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::Mesh;
use sonium_fem::physics::fluid::stokes::navier_stokes::{
    BackendPreconditionerKind, BackendPressureSchurPreconditionerKind, LinearSolverBackend,
    LinearSolverCpuPreconditionerKind, NavierStokesConfig,
};
use sonium_fem::physics::fluid::stokes::time_stepping::BdfConfig;
use sonium_fem::spaces::{H1Space, VectorH1Space};

use super::analysis::thd_fraction;
use super::baseline::PortOptimReferenceCase;
use super::campaign::{CampaignArchiverError, STEP7_TIER2_DRIVES_V_PEAK, production_fidelity};
use super::coupled::{CoupledPortConfig, PortFlowFieldSnapshot, run_coupled_port};
use super::error::PortFlowError;
use super::evaluate::PortEvalFidelity;
use super::geometry3d::{SplineRoundPortGeometry, build_spline_round_port_mesh};
use super::optimize::{ProfileTemplate, campaign_template};
use super::profile::SplinePortProfile;
use crate::{
    AcousticMedium, PortNoiseMethod, PortNoiseObserver, PortNoiseObserverResult, PortNoiseResult,
    PortNoiseSpec, analyze_port_noise,
};

/// Archive schema version for [`Step8Report`].
pub const STEP8_REPORT_SCHEMA_VERSION: u32 = 1;

/// Model lineage recorded in every [`Step8Report`].
pub const STEP8_REPORT_PROVENANCE: &str =
    "PortOptimStep8Finalist[Tier2SnapshotNoise,Round3DSpotCheckNightly]";

/// Model lineage recorded in every [`SpotCheck3dRun`].
pub const STEP8_SPOTCHECK_PROVENANCE: &str =
    "Tier2Round3D[CoupledP2P1BDF2,NoStabilisation,AffineBoxClosure,IterativeJacobiSchur]+nightly";

/// Tier label carried by every 3D spot-check record.
pub const STEP8_SPOTCHECK_TIER: &str = "nightly";

/// On-axis observer distance from the exterior mouth in m (plan default).
pub const STEP8_NOISE_OBSERVER_DISTANCE_M: f64 = 1.0;

/// Field snapshots retained per drive cycle for the noise DFT.
pub const STEP8_NOISE_SNAPSHOTS_PER_CYCLE: usize = 12;

/// Harmonics of the drive analysed for the noise floor (`f1..=5·f1`).
pub const STEP8_NOISE_HARMONICS: usize = 5;

/// 3D-vs-2D ratio agreement tolerance: above 10% investigates, never adjusts.
pub const STEP8_SPOTCHECK_DISAGREE_REL_TOL: f64 = 0.10;

/// Minimum noise snapshots: the analyzer needs four equally spaced fields.
pub const STEP8_MIN_NOISE_SNAPSHOTS: usize = 4;

/// Chamber-to-duct smoothstep extent for the 3D spot-check in m.
///
/// Mirrors the Step-7 Tier-1 evaluator flare; the spline path rejects sharp
/// shoulders, so this must stay positive.
pub const STEP8_SPOTCHECK_FLARE_LENGTH_M: f64 = 0.01;

/// Drive voltages for the 3D spot-check: one low, one high (V peak).
pub const STEP8_SPOTCHECK_DRIVES_V_PEAK: [f64; 2] =
    [STEP7_TIER2_DRIVES_V_PEAK[0], STEP7_TIER2_DRIVES_V_PEAK[2]];

// ---------------------------------------------------------------------------
// Snapshot fidelity and noise specification
// ---------------------------------------------------------------------------

/// Snapshot stride giving [`STEP8_NOISE_SNAPSHOTS_PER_CYCLE`] per cycle.
///
/// Requires an exact division so every snapshot lands on the same
/// cycle phase and every analysed harmonic lands on a DFT bin.
pub fn noise_snapshot_stride(samples_per_cycle: usize) -> Result<usize, PortFlowError> {
    if samples_per_cycle < STEP8_NOISE_SNAPSHOTS_PER_CYCLE
        || !samples_per_cycle.is_multiple_of(STEP8_NOISE_SNAPSHOTS_PER_CYCLE)
    {
        return Err(PortFlowError::InvalidConfig(format!(
            "Step-8 noise needs samples_per_cycle divisible by {}, got {samples_per_cycle}",
            STEP8_NOISE_SNAPSHOTS_PER_CYCLE
        )));
    }
    Ok(samples_per_cycle / STEP8_NOISE_SNAPSHOTS_PER_CYCLE)
}

/// Production Tier-2 fidelity with snapshot output for noise post-processing.
///
/// Same 252 samples/cycle, 40/8-cycle envelope as the Step-7 evidence, plus a
/// stride-21 snapshot stream (12 per cycle, 96 in the tail).
pub fn noise_snapshot_fidelity(
    case: &PortOptimReferenceCase,
) -> Result<PortEvalFidelity, PortFlowError> {
    let mut fidelity = production_fidelity(case);
    fidelity.snapshot_interval_steps = noise_snapshot_stride(fidelity.samples_per_cycle)?;
    Ok(fidelity)
}

/// On-axis observer 1 m downstream of the Tier-2 exterior mouth.
pub fn noise_observer_1m_on_axis(case: &PortOptimReferenceCase) -> PortNoiseObserver {
    PortNoiseObserver {
        position_m: [
            case.tier2_duct_length_m
                + case.tier2_chamber_length_m
                + STEP8_NOISE_OBSERVER_DISTANCE_M,
            0.0,
            0.0,
        ],
    }
}

/// 1 m on-axis noise spec with automatic Lighthill/FW–H selection by `ka`.
///
/// Frequencies are the drive harmonics `f1..=5·f1` that sit strictly below the
/// snapshot Nyquist rate; at least the fundamental must fit. The source length
/// is the duct length (the compactness scale), and the out-of-plane depth is
/// the caller's 2D extrusion depth (unit depth for the slot surrogate).
pub fn finalist_noise_spec(
    drive_frequency_hz: f64,
    snapshot_interval_s: f64,
    observer: PortNoiseObserver,
    characteristic_source_length_m: f64,
    out_of_plane_depth_m: f64,
) -> Result<PortNoiseSpec, PortFlowError> {
    for (name, value) in [
        ("drive_frequency_hz", drive_frequency_hz),
        ("snapshot_interval_s", snapshot_interval_s),
        (
            "characteristic_source_length_m",
            characteristic_source_length_m,
        ),
        ("out_of_plane_depth_m", out_of_plane_depth_m),
    ] {
        if !(value.is_finite() && value > 0.0) {
            return Err(PortFlowError::InvalidConfig(format!(
                "Step-8 noise spec needs finite positive {name}, got {value}"
            )));
        }
    }
    if observer.position_m.iter().any(|value| !value.is_finite()) {
        return Err(PortFlowError::InvalidConfig(
            "Step-8 noise spec needs a finite observer position".to_string(),
        ));
    }
    let nyquist_hz = 0.5 / snapshot_interval_s;
    let frequencies_hz: Vec<f64> = (1..=STEP8_NOISE_HARMONICS)
        .map(|harmonic| harmonic as f64 * drive_frequency_hz)
        .take_while(|frequency| *frequency < nyquist_hz)
        .collect();
    if frequencies_hz.is_empty() {
        return Err(PortFlowError::InvalidConfig(format!(
            "Step-8 noise spec cannot analyse f1={drive_frequency_hz} Hz \
             at snapshot interval {snapshot_interval_s:.3e} s (Nyquist {nyquist_hz:.3e} Hz)"
        )));
    }
    Ok(PortNoiseSpec {
        method: PortNoiseMethod::Automatic,
        frequencies_hz,
        observers: vec![observer],
        characteristic_source_length_m,
        compact_ka_max: PortNoiseSpec::default().compact_ka_max,
        out_of_plane_depth_m,
        reference_pressure_pa: PortNoiseSpec::default().reference_pressure_pa,
    })
}

/// Tail snapshots (`time_s >= tail_start_time_s`) in run order.
pub fn tail_snapshots(
    snapshots: &[PortFlowFieldSnapshot],
    tail_start_time_s: f64,
) -> Vec<&PortFlowFieldSnapshot> {
    snapshots
        .iter()
        .filter(|snapshot| snapshot.time_s >= tail_start_time_s)
        .collect()
}

/// Select the settled-tail snapshot window from a full-run stream.
///
/// Takes the last `tail_samples / snapshot_interval_steps` snapshots (exact
/// division required) and verifies uniform `stride·dt` spacing plus alignment
/// of the final snapshot with the run end (`steps_completed·dt`). This pins
/// the window without assuming the time origin of the series.
pub fn tail_snapshot_window(
    snapshots: &[PortFlowFieldSnapshot],
    tail_samples: usize,
    snapshot_interval_steps: usize,
    dt_s: f64,
    steps_completed: usize,
) -> Result<Vec<PortFlowFieldSnapshot>, PortFlowError> {
    if snapshot_interval_steps == 0 {
        return Err(PortFlowError::InvalidConfig(
            "Step-8 noise needs snapshots enabled (nonzero stride)".to_string(),
        ));
    }
    if !(dt_s.is_finite() && dt_s > 0.0) {
        return Err(PortFlowError::InvalidConfig(format!(
            "Step-8 noise needs a finite positive timestep, got {dt_s}"
        )));
    }
    if !tail_samples.is_multiple_of(snapshot_interval_steps) {
        return Err(PortFlowError::InvalidConfig(format!(
            "Step-8 noise needs tail_samples divisible by the snapshot stride, \
             got tail {tail_samples} stride {snapshot_interval_steps}"
        )));
    }
    let n_tail = tail_samples / snapshot_interval_steps;
    if n_tail < STEP8_MIN_NOISE_SNAPSHOTS {
        return Err(PortFlowError::InvalidConfig(format!(
            "Step-8 noise needs at least {STEP8_MIN_NOISE_SNAPSHOTS} tail snapshots, got {n_tail}"
        )));
    }
    if snapshots.len() < n_tail {
        return Err(PortFlowError::InvalidConfig(format!(
            "Step-8 noise needs {n_tail} tail snapshots, the run retained {}",
            snapshots.len()
        )));
    }
    let window: Vec<PortFlowFieldSnapshot> = snapshots[snapshots.len() - n_tail..].to_vec();
    let expected_interval = snapshot_interval_steps as f64 * dt_s;
    for pair in window.windows(2) {
        let gap = pair[1].time_s - pair[0].time_s;
        if (gap - expected_interval).abs() > 1.0e-9 * expected_interval {
            return Err(PortFlowError::InvalidConfig(format!(
                "Step-8 noise tail window is not uniformly spaced: gap {gap:.6e} s, \
                 expected {expected_interval:.6e} s"
            )));
        }
    }
    let expected_end = steps_completed as f64 * dt_s;
    let last = window.last().map_or(f64::NAN, |snapshot| snapshot.time_s);
    if (last - expected_end).abs() > 1.0e-9 * expected_end.max(dt_s) {
        return Err(PortFlowError::InvalidConfig(format!(
            "Step-8 noise tail window ends at {last:.6e} s, run end is {expected_end:.6e} s"
        )));
    }
    Ok(window)
}

/// Run [`analyze_port_noise`] over settled-tail snapshots of one Tier-2 run.
///
/// The velocity/pressure spaces match the Tier-2 solve (P2/P1); the
/// [`PortNoiseResult`] carries the analyzer provenance intact.
pub fn analyze_tail_noise(
    mesh: &Mesh,
    tail: &[PortFlowFieldSnapshot],
    medium: AcousticMedium,
    spec: &PortNoiseSpec,
) -> Result<PortNoiseResult, PortFlowError> {
    if tail.len() < STEP8_MIN_NOISE_SNAPSHOTS {
        return Err(PortFlowError::InvalidConfig(format!(
            "Step-8 noise needs at least {STEP8_MIN_NOISE_SNAPSHOTS} tail snapshots, got {}",
            tail.len()
        )));
    }
    let velocity_space = H1Space::new(PolynomialDegree::P2);
    let pressure_space = H1Space::new(PolynomialDegree::P1);
    analyze_port_noise(mesh, &velocity_space, &pressure_space, tail, medium, spec).map_err(
        |error| {
            PortFlowError::InvalidConfig(format!(
                "Step-8 noise analysis rejected its inputs: {error}"
            ))
        },
    )
}

// ---------------------------------------------------------------------------
// Winner-vs-straight noise levels
// ---------------------------------------------------------------------------

/// Winner-vs-straight noise comparison at one drive level.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FinalistNoiseLevel {
    /// Drive level (V peak) shared by both runs.
    pub voltage_peak_v: f64,
    /// Tail snapshots analysed (same count both sides).
    pub n_snapshots: usize,
    /// Snapshot sampling interval in s (measured, not assumed).
    pub snapshot_interval_s: f64,
    /// Straight-tube observer result (spectra with per-point method/`ka`).
    pub straight: PortNoiseObserverResult,
    /// Winner observer result (spectra with per-point method/`ka`).
    pub winner: PortNoiseObserverResult,
    /// Integrated-floor difference `winner − straight` in dB.
    pub delta_integrated_db: f64,
    /// Analyzer provenance behind the straight result.
    pub straight_provenance: String,
    /// Analyzer provenance behind the winner result.
    pub winner_provenance: String,
}

impl FinalistNoiseLevel {
    /// Compare two single-observer noise results at one drive level.
    pub fn compare(
        voltage_peak_v: f64,
        straight: &PortNoiseResult,
        winner: &PortNoiseResult,
    ) -> Result<Self, PortFlowError> {
        if !(voltage_peak_v.is_finite() && voltage_peak_v > 0.0) {
            return Err(PortFlowError::InvalidConfig(format!(
                "noise comparison needs a finite positive drive voltage, got {voltage_peak_v}"
            )));
        }
        for (name, result) in [("straight", straight), ("winner", winner)] {
            if result.observers.len() != 1 {
                return Err(PortFlowError::InvalidConfig(format!(
                    "noise comparison needs single-observer {name} results, got {}",
                    result.observers.len()
                )));
            }
            if result.sample_count < STEP8_MIN_NOISE_SNAPSHOTS {
                return Err(PortFlowError::InvalidConfig(format!(
                    "noise comparison needs {STEP8_MIN_NOISE_SNAPSHOTS}+ {name} snapshots, got {}",
                    result.sample_count
                )));
            }
            if !(result.sample_interval_s.is_finite() && result.sample_interval_s > 0.0) {
                return Err(PortFlowError::InvalidConfig(format!(
                    "noise comparison needs a finite positive {name} sample interval"
                )));
            }
        }
        if straight.sample_count != winner.sample_count {
            return Err(PortFlowError::InvalidConfig(format!(
                "noise comparison needs matched snapshot counts, got {} vs {}",
                straight.sample_count, winner.sample_count
            )));
        }
        let straight_obs = &straight.observers[0];
        let winner_obs = &winner.observers[0];
        if straight_obs.spectrum.len() != winner_obs.spectrum.len() {
            return Err(PortFlowError::InvalidConfig(format!(
                "noise comparison needs matched spectra, got {} vs {} points",
                straight_obs.spectrum.len(),
                winner_obs.spectrum.len()
            )));
        }
        if !(straight_obs.integrated_noise_floor_db.is_finite()
            && winner_obs.integrated_noise_floor_db.is_finite())
        {
            return Err(PortFlowError::InvalidConfig(
                "noise comparison needs finite integrated floors on both sides".to_string(),
            ));
        }
        Ok(Self {
            voltage_peak_v,
            n_snapshots: straight.sample_count,
            snapshot_interval_s: straight.sample_interval_s,
            straight: straight_obs.clone(),
            winner: winner_obs.clone(),
            delta_integrated_db: winner_obs.integrated_noise_floor_db
                - straight_obs.integrated_noise_floor_db,
            straight_provenance: straight.provenance.clone(),
            winner_provenance: winner.provenance.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// Nightly 3D spot-check
// ---------------------------------------------------------------------------

/// Linear-solve device for one 3D spot-check run. Execution-only: both
/// devices solve the same physics to the same tolerance.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpotCheck3dDevice {
    /// CPU FGMRES + Jacobi/Schur (default; always available).
    #[default]
    Cpu,
    /// CUDA backend GMRES + Jacobi/Schur with a Chebyshev pressure
    /// block. Requires a `cuda`-feature build plus a CUDA driver; the
    /// run fails loudly if any solve falls back to CPU.
    Cuda,
}

/// Time resolution, run length, and mesh for one 3D spot-check run.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpotCheck3dFidelity {
    /// Target cell size in m.
    pub mesh_size_m: f64,
    /// BDF samples per drive cycle (sets `dt = 1/(f·spc)`).
    pub samples_per_cycle: usize,
    /// Transient cycles discarded before the analysed tail.
    pub discarded_cycles: usize,
    /// Settled tail cycles analysed.
    pub tail_cycles: usize,
    /// Linear-solve device ([`SpotCheck3dDevice::Cpu`] by default).
    pub device: SpotCheck3dDevice,
}

impl SpotCheck3dFidelity {
    /// Cheap smoke leg: 0.02 m mesh, 32 samples/cycle, 2 + 2 cycles.
    ///
    /// Mirrors the nightly `SPEAKER_PORT_REDUCED_3D=smoke` tier; the spot-check
    /// is a transfer sanity check, not a convergence claim.
    pub fn smoke() -> Self {
        Self {
            mesh_size_m: 0.02,
            samples_per_cycle: 32,
            discarded_cycles: 2,
            tail_cycles: 2,
            device: SpotCheck3dDevice::Cpu,
        }
    }

    /// Reject an inconsistent fidelity before any solve.
    pub fn validate(&self) -> Result<(), PortFlowError> {
        if !(self.mesh_size_m.is_finite() && self.mesh_size_m > 0.0) {
            return Err(PortFlowError::InvalidConfig(format!(
                "3D spot-check mesh size must be finite and positive, got {}",
                self.mesh_size_m
            )));
        }
        if self.samples_per_cycle < STEP8_NOISE_SNAPSHOTS_PER_CYCLE {
            return Err(PortFlowError::InvalidConfig(format!(
                "3D spot-check needs at least {} samples per cycle, got {}",
                STEP8_NOISE_SNAPSHOTS_PER_CYCLE, self.samples_per_cycle
            )));
        }
        if self.tail_cycles == 0 {
            return Err(PortFlowError::InvalidConfig(
                "3D spot-check needs a non-empty analysed tail".to_string(),
            ));
        }
        self.discarded_cycles
            .checked_add(self.tail_cycles)
            .and_then(|cycles| cycles.checked_mul(self.samples_per_cycle))
            .ok_or_else(|| {
                PortFlowError::InvalidConfig("3D spot-check step count overflow".to_string())
            })?;
        Ok(())
    }
}

/// 3D drive frequency: half the campaign tuning, mirroring the Tier-2
/// convention (`tier2_drive_frequency_hz` = slot tuning / 2).
pub fn spot_drive_frequency_hz(f_target_hz: f64) -> Result<f64, PortFlowError> {
    if !(f_target_hz.is_finite() && f_target_hz > 0.0) {
        return Err(PortFlowError::InvalidConfig(format!(
            "3D spot-check needs a finite positive campaign tuning, got {f_target_hz}"
        )));
    }
    Ok(0.5 * f_target_hz)
}

/// Campaign-scale profile template for the 3D spot-check (N = 5).
pub fn spotcheck_template(case: &PortOptimReferenceCase) -> ProfileTemplate {
    campaign_template(case, 5, super::campaign::campaign_profile_options())
}

/// Revolve a campaign-scale meridian profile into a 3D round-port geometry.
///
/// Chambers come from the Tier-1 envelope; the profile length/bore must match
/// the campaign duct or the geometry validation rejects the run before any
/// solve (slot-scale Tier-2 profiles are off-envelope here).
pub fn spline_3d_geometry(
    case: &PortOptimReferenceCase,
    profile: &SplinePortProfile,
    mesh_size_m: f64,
) -> Result<SplineRoundPortGeometry, PortFlowError> {
    case.validate()?;
    if !(mesh_size_m.is_finite() && mesh_size_m > 0.0) {
        return Err(PortFlowError::InvalidConfig(format!(
            "3D spot-check mesh size must be finite and positive, got {mesh_size_m}"
        )));
    }
    Ok(SplineRoundPortGeometry {
        duct_length_m: case.port_length_l_m,
        bore_radius_m: case.port_radius_r0_m,
        chamber_length_m: case.tier1_chamber_length_m,
        chamber_radius_m: case.tier1_chamber_radius_m,
        flare_length_m: STEP8_SPOTCHECK_FLARE_LENGTH_M,
        mesh_size_m,
        angular_sectors: None,
        profile: profile.clone(),
    })
}

/// Reduced observables of one 3D spot-check run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpotCheck3dRun {
    /// Must equal [`STEP8_SPOTCHECK_PROVENANCE`].
    pub provenance: String,
    /// Always [`STEP8_SPOTCHECK_TIER`].
    pub tier: String,
    /// Candidate identity: normalised free radii `u`.
    pub profile_normalized: Vec<f64>,
    /// Drive voltage in V peak.
    pub voltage_peak_v: f64,
    /// Drive frequency in Hz.
    pub drive_frequency_hz: f64,
    /// BDF timestep in s.
    pub dt_s: f64,
    /// Completed BDF steps (always the configured total on success).
    pub steps_completed: usize,
    /// Time-domain port-flow RMS over the tail in m³/s.
    pub port_flow_rms_m3s: f64,
    /// THD over 5 harmonics (fraction).
    pub thd_fraction: f64,
    /// `max|q_ext + q_box,out|/max|q_ext|` over the tail (time domain).
    pub cap_continuity_relative_error: f64,
    /// Mesh node count.
    pub mesh_nodes: usize,
    /// Mesh tetrahedron count.
    pub mesh_tetrahedra: usize,
}

/// Run one 3D coupled spot-check: revolve, mesh, solve, reduce.
///
/// Taylor–Hood P2–P1 BDF2 with residual stabilisation off and the nightly
/// iterative Jacobi–Schur saddle solver, mirroring the `SPEAKER_PORT_3D=1`
/// reduced-3D leg. A pure function of (profile, voltage, frequency, fidelity).
/// The linear-solve device comes from [`SpotCheck3dFidelity::device`]
/// (CPU by default; CUDA selects the tuned backend Chebyshev–Schur path).
pub fn run_3d_spot_check(
    case: &PortOptimReferenceCase,
    profile: &SplinePortProfile,
    voltage_peak_v: f64,
    drive_frequency_hz: f64,
    fidelity: &SpotCheck3dFidelity,
) -> Result<SpotCheck3dRun, PortFlowError> {
    fidelity.validate()?;
    if !(voltage_peak_v.is_finite() && voltage_peak_v > 0.0) {
        return Err(PortFlowError::InvalidConfig(format!(
            "3D spot-check drive voltage must be finite and positive, got {voltage_peak_v}"
        )));
    }
    if !(drive_frequency_hz.is_finite() && drive_frequency_hz > 0.0) {
        return Err(PortFlowError::InvalidConfig(format!(
            "3D spot-check drive frequency must be finite and positive, got {drive_frequency_hz}"
        )));
    }
    if !profile.normalized().iter().all(|u| u.is_finite()) {
        return Err(PortFlowError::InvalidConfig(
            "3D spot-check profile has a zero throat span, so its normalized identity is undefined"
                .to_string(),
        ));
    }
    let geometry = spline_3d_geometry(case, profile, fidelity.mesh_size_m)?;
    let mesh = build_spline_round_port_mesh(&geometry)?;
    let dt_s = 1.0 / (drive_frequency_hz * fidelity.samples_per_cycle as f64);
    if !(dt_s.is_finite() && dt_s > 0.0) {
        return Err(PortFlowError::InvalidConfig(format!(
            "3D spot-check timestep is non-finite: {dt_s}"
        )));
    }
    // The 5th drive harmonic must sit below Nyquist for the THD reduction.
    if 5.0 * drive_frequency_hz >= 0.5 / dt_s {
        return Err(PortFlowError::InvalidConfig(format!(
            "3D spot-check cannot analyse the 5th harmonic of {drive_frequency_hz} Hz \
             at dt = {dt_s:.3e} s"
        )));
    }
    let num_steps = (fidelity.discarded_cycles + fidelity.tail_cycles) * fidelity.samples_per_cycle;
    let tail_samples = fidelity.tail_cycles * fidelity.samples_per_cycle;
    let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
    let pressure_space = H1Space::new(PolynomialDegree::P1);
    let frequency_hz = drive_frequency_hz;
    // The CUDA device uses the backend Jacobi/Schur solver with a
    // Chebyshev pressure block: measured fastest on the 47.8k-DOF
    // parity probe (16.9s vs 24.0s diagonal-Schur and 26.6s CPU
    // JacobiSchur at tighter 1e-10 tolerance). The CPU-only knobs are
    // ignored on the backend path and vice versa.
    let (linear_solver_backend, backend_preconditioner, schur_preconditioner) =
        match fidelity.device {
            SpotCheck3dDevice::Cpu => (
                LinearSolverBackend::Cpu,
                BackendPreconditionerKind::PolynomialJacobi,
                BackendPressureSchurPreconditionerKind::Polynomial,
            ),
            SpotCheck3dDevice::Cuda => (
                LinearSolverBackend::Cuda,
                BackendPreconditionerKind::JacobiSchur,
                BackendPressureSchurPreconditionerKind::Chebyshev,
            ),
        };
    let config = CoupledPortConfig {
        driver: case.driver(),
        ns_config: NavierStokesConfig {
            mu: case.dynamic_viscosity_pa_s,
            rho: case.density_kg_per_m3,
            use_residual_stabilisation: false,
            // Zero body force through `solve_bdf_step_cached`: the
            // force quadrature is identical every step, so caching
            // skips two full assemblies per step after the first.
            cache_body_force_assembly: true,
            use_direct_solver: false,
            linear_solver_backend,
            linear_solver_backend_preconditioner: backend_preconditioner,
            linear_solver_schur_preconditioner: schur_preconditioner,
            linear_solver_cpu_preconditioner: LinearSolverCpuPreconditionerKind::JacobiSchur,
            linear_solver_max_iterations: 5000,
            linear_solver_tolerance: 1.0e-8,
            ..Default::default()
        },
        bdf: BdfConfig::bdf2(dt_s, num_steps),
        voltage_fn: Box::new(move |time_s| voltage_peak_v * (TAU * frequency_hz * time_s).sin()),
        under_relaxation: 1.0,
        exterior_pressure_pa: 0.0,
    };
    let result = run_coupled_port(&mesh, &velocity_space, &pressure_space, &config)?;
    // A CUDA run must never silently report CPU numbers: every linear
    // solve has to resolve to the requested device.
    if matches!(fidelity.device, SpotCheck3dDevice::Cuda)
        && result.linear_solves.iter().any(|solve| {
            !matches!(
                solve.resolved_linear_solver_backend,
                LinearSolverBackend::Cuda
            )
        })
    {
        return Err(PortFlowError::InvalidConfig(
            "3D spot-check requested the CUDA device but a linear solve fell back to CPU; \
             build with the cuda feature and provide a CUDA driver"
                .to_string(),
        ));
    }
    let last_time = result.time_s.last().copied().unwrap_or(0.0);
    let diverged = |step: usize| PortFlowError::SolverDiverged {
        step,
        time_s: last_time,
    };
    if result.steps_completed != num_steps
        || result.port_flow_m3s.len() != num_steps
        || result.box_cap_outward_flow_m3s.len() != num_steps
    {
        return Err(diverged(result.steps_completed.min(num_steps)));
    }
    let tail = &result.port_flow_m3s[num_steps - tail_samples..];
    if tail.iter().any(|value| !value.is_finite()) {
        return Err(diverged(num_steps - 1));
    }
    let port_flow_rms_m3s = (tail.iter().map(|q| q * q).sum::<f64>() / tail.len() as f64).sqrt();
    let max_abs_flow = tail.iter().map(|q| q.abs()).fold(0.0, f64::max);
    if !(port_flow_rms_m3s.is_finite() && port_flow_rms_m3s > 0.0 && max_abs_flow > 0.0) {
        return Err(diverged(num_steps - 1));
    }
    let thd = thd_fraction(tail, dt_s, drive_frequency_hz, 5);
    if !(thd.is_finite() && thd >= 0.0) {
        return Err(diverged(num_steps - 1));
    }
    let box_tail = &result.box_cap_outward_flow_m3s[num_steps - tail_samples..];
    let max_mismatch = tail
        .iter()
        .zip(box_tail.iter())
        .map(|(exterior, box_outward)| (exterior + box_outward).abs())
        .fold(0.0, f64::max);
    let cap_continuity_relative_error = max_mismatch / max_abs_flow;
    if !cap_continuity_relative_error.is_finite() {
        return Err(diverged(num_steps - 1));
    }
    if !result
        .energy_summary(num_steps - tail_samples..num_steps)
        .is_finite()
    {
        return Err(diverged(num_steps - 1));
    }
    Ok(SpotCheck3dRun {
        provenance: STEP8_SPOTCHECK_PROVENANCE.to_string(),
        tier: STEP8_SPOTCHECK_TIER.to_string(),
        profile_normalized: profile.normalized(),
        voltage_peak_v,
        drive_frequency_hz,
        dt_s,
        steps_completed: num_steps,
        port_flow_rms_m3s,
        thd_fraction: thd,
        cap_continuity_relative_error,
        mesh_nodes: mesh.num_nodes(),
        mesh_tetrahedra: mesh.elements.len(),
    })
}

// ---------------------------------------------------------------------------
// 3D-vs-2D agreement gate
// ---------------------------------------------------------------------------

/// Agreement verdict for one spot-check level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpotVerdict {
    /// Both ratios agree within [`STEP8_SPOTCHECK_DISAGREE_REL_TOL`].
    Agree,
    /// A ratio disagrees above tolerance: investigate, never adjust the claim.
    Investigate,
}

/// Relative disagreement `|a3 − a2|/|a2|` of a 3D ratio against its 2D twin.
pub fn relative_ratio_disagreement(ratio_2d: f64, ratio_3d: f64) -> Result<f64, PortFlowError> {
    if !(ratio_2d.is_finite() && ratio_2d > 0.0) {
        return Err(PortFlowError::InvalidConfig(format!(
            "spot-check agreement needs a finite positive 2D ratio, got {ratio_2d}"
        )));
    }
    if !ratio_3d.is_finite() {
        return Err(PortFlowError::InvalidConfig(format!(
            "spot-check agreement needs a finite 3D ratio, got {ratio_3d}"
        )));
    }
    Ok((ratio_3d - ratio_2d).abs() / ratio_2d)
}

/// 3D-vs-2D transfer check at one drive level (winner/straight ratios).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpotCheck3dLevel {
    /// Drive level (V peak) shared by all four runs behind this level.
    pub voltage_peak_v: f64,
    /// 3D drive frequency in Hz (half the campaign tuning).
    pub drive_frequency_hz: f64,
    /// Winner/straight `Q_rms` ratio in 2D.
    pub q_ratio_2d: f64,
    /// Winner/straight `Q_rms` ratio in 3D.
    pub q_ratio_3d: f64,
    /// Relative disagreement of the airflow ratios.
    pub q_disagreement_rel: f64,
    /// Winner/straight THD ratio in 2D.
    pub thd_ratio_2d: f64,
    /// Winner/straight THD ratio in 3D.
    pub thd_ratio_3d: f64,
    /// Relative disagreement of the THD ratios.
    pub thd_disagreement_rel: f64,
    /// Agreement verdict (recomputed by [`Step8Report::validate`]).
    pub verdict: SpotVerdict,
    /// Always [`STEP8_SPOTCHECK_TIER`].
    pub tier: String,
}

impl SpotCheck3dLevel {
    /// Compare 2D and 3D winner/straight ratios at one drive level.
    pub fn compare(
        voltage_peak_v: f64,
        drive_frequency_hz: f64,
        q_ratio_2d: f64,
        q_ratio_3d: f64,
        thd_ratio_2d: f64,
        thd_ratio_3d: f64,
    ) -> Result<Self, PortFlowError> {
        if !(voltage_peak_v.is_finite() && voltage_peak_v > 0.0) {
            return Err(PortFlowError::InvalidConfig(format!(
                "spot-check comparison needs a finite positive drive voltage, got {voltage_peak_v}"
            )));
        }
        if !(drive_frequency_hz.is_finite() && drive_frequency_hz > 0.0) {
            return Err(PortFlowError::InvalidConfig(format!(
                "spot-check comparison needs a finite positive drive frequency, got {drive_frequency_hz}"
            )));
        }
        let q_disagreement_rel = relative_ratio_disagreement(q_ratio_2d, q_ratio_3d)?;
        let thd_disagreement_rel = relative_ratio_disagreement(thd_ratio_2d, thd_ratio_3d)?;
        let verdict = if q_disagreement_rel > STEP8_SPOTCHECK_DISAGREE_REL_TOL
            || thd_disagreement_rel > STEP8_SPOTCHECK_DISAGREE_REL_TOL
        {
            SpotVerdict::Investigate
        } else {
            SpotVerdict::Agree
        };
        Ok(Self {
            voltage_peak_v,
            drive_frequency_hz,
            q_ratio_2d,
            q_ratio_3d,
            q_disagreement_rel,
            thd_ratio_2d,
            thd_ratio_3d,
            thd_disagreement_rel,
            verdict,
            tier: STEP8_SPOTCHECK_TIER.to_string(),
        })
    }

    /// Recompute the verdict from the archived disagreements.
    pub fn recomputed_verdict(&self) -> SpotVerdict {
        if self.q_disagreement_rel > STEP8_SPOTCHECK_DISAGREE_REL_TOL
            || self.thd_disagreement_rel > STEP8_SPOTCHECK_DISAGREE_REL_TOL
        {
            SpotVerdict::Investigate
        } else {
            SpotVerdict::Agree
        }
    }
}

// ---------------------------------------------------------------------------
// Step-8 evidence archive
// ---------------------------------------------------------------------------

/// Step-8 evidence archive: noise table plus nightly 3D spot-checks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Step8Report {
    /// Must equal [`STEP8_REPORT_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Model lineage; must equal [`STEP8_REPORT_PROVENANCE`].
    pub provenance: String,
    /// Frozen reference case the finalists ran on.
    pub case: PortOptimReferenceCase,
    /// Campaign tuning target in Hz (Tier-0 duct tuning of straight).
    pub f_target_hz: f64,
    /// Primary control-point count (5).
    pub n_control_points: usize,
    /// Winning `u` carried over from the Step-7 archive.
    pub winner_u: Vec<f64>,
    /// Winner-vs-straight noise levels (integrated floor dB table).
    pub noise: Vec<FinalistNoiseLevel>,
    /// 3D spot-check runs (straight + winner per level).
    pub spot3d_runs: Vec<SpotCheck3dRun>,
    /// 3D-vs-2D agreement levels (nightly tier).
    pub spot3d: Vec<SpotCheck3dLevel>,
    /// Free-form leg notes (fidelities, snapshot counts, verdict reading).
    pub notes: String,
}

impl Step8Report {
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
        let report: Step8Report = serde_json::from_str(raw)
            .map_err(|error| CampaignArchiverError::Json(error.to_string()))?;
        if report.schema_version != STEP8_REPORT_SCHEMA_VERSION {
            return Err(CampaignArchiverError::Json(format!(
                "unsupported Step-8 report schema {}, expected {STEP8_REPORT_SCHEMA_VERSION}",
                report.schema_version
            )));
        }
        if report.provenance != STEP8_REPORT_PROVENANCE {
            return Err(CampaignArchiverError::Json(format!(
                "Step-8 report provenance drift: {}",
                report.provenance
            )));
        }
        Ok(report)
    }

    /// Structural validation: noise coverage, nightly labels, verdict coherence.
    pub fn validate(&self) -> Result<(), CampaignArchiverError> {
        let bad = |message: String| CampaignArchiverError::Json(message);
        if self.noise.is_empty() {
            return Err(bad("Step 8 needs at least one noise level".to_string()));
        }
        for level in &self.noise {
            if !(level.voltage_peak_v.is_finite() && level.voltage_peak_v > 0.0) {
                return Err(bad(format!(
                    "noise level has a non-positive drive voltage {}",
                    level.voltage_peak_v
                )));
            }
            if !level.delta_integrated_db.is_finite() {
                return Err(bad(format!(
                    "noise level at {} V has a non-finite integrated delta",
                    level.voltage_peak_v
                )));
            }
            if level.straight.spectrum.is_empty() || level.winner.spectrum.is_empty() {
                return Err(bad(format!(
                    "noise level at {} V has an empty spectrum",
                    level.voltage_peak_v
                )));
            }
        }
        if self.spot3d.is_empty() {
            return Err(bad(
                "Step 8 needs at least one 3D spot-check level".to_string()
            ));
        }
        for level in &self.spot3d {
            if level.tier != STEP8_SPOTCHECK_TIER {
                return Err(bad(format!(
                    "3D spot-check at {} V lost its nightly tier label: {}",
                    level.voltage_peak_v, level.tier
                )));
            }
            if level.verdict != level.recomputed_verdict() {
                return Err(bad(format!(
                    "3D spot-check at {} V has a stale verdict: archived {:?}, \
                     disagreements q={:.4} thd={:.4} recompute to {:?}",
                    level.voltage_peak_v,
                    level.verdict,
                    level.q_disagreement_rel,
                    level.thd_disagreement_rel,
                    level.recomputed_verdict()
                )));
            }
            let runs: Vec<_> = self
                .spot3d_runs
                .iter()
                .filter(|run| run.voltage_peak_v == level.voltage_peak_v)
                .collect();
            if runs.len() < 2 {
                return Err(bad(format!(
                    "3D spot-check at {} V needs straight + winner runs, got {}",
                    level.voltage_peak_v,
                    runs.len()
                )));
            }
            for run in runs {
                if run.tier != STEP8_SPOTCHECK_TIER {
                    return Err(bad(format!(
                        "3D spot-check run at {} V lost its nightly tier label",
                        run.voltage_peak_v
                    )));
                }
                if run.provenance != STEP8_SPOTCHECK_PROVENANCE {
                    return Err(bad(format!(
                        "3D spot-check run at {} V has provenance drift: {}",
                        run.voltage_peak_v, run.provenance
                    )));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_stride_needs_exact_cycles() {
        assert_eq!(noise_snapshot_stride(252).unwrap(), 21);
        assert_eq!(noise_snapshot_stride(48).unwrap(), 4);
        assert!(noise_snapshot_stride(50).is_err());
        assert!(noise_snapshot_stride(8).is_err());
        assert!(noise_snapshot_stride(0).is_err());
    }

    #[test]
    fn production_fidelity_carries_snapshots() {
        let case = PortOptimReferenceCase::default();
        let fidelity = noise_snapshot_fidelity(&case).unwrap();
        assert_eq!(fidelity.snapshot_interval_steps, 21);
        assert_eq!(fidelity.samples_per_cycle, case.tier2_samples_per_cycle);
        assert_eq!(fidelity.total_cycles, case.tier2_total_cycles);
        assert_eq!(fidelity.tail_cycles, case.tier2_tail_cycles);
        fidelity.validate().unwrap();
    }

    #[test]
    fn noise_spec_selects_harmonics_below_nyquist() {
        let case = PortOptimReferenceCase::default();
        let drive_hz = case.tier2_drive_frequency_hz().unwrap();
        let dt = 1.0 / (drive_hz * 252.0);
        let spec = finalist_noise_spec(
            drive_hz,
            21.0 * dt,
            noise_observer_1m_on_axis(&case),
            case.tier2_duct_length_m,
            1.0,
        )
        .unwrap();
        assert_eq!(spec.method, PortNoiseMethod::Automatic);
        assert_eq!(spec.frequencies_hz.len(), STEP8_NOISE_HARMONICS);
        for (index, frequency) in spec.frequencies_hz.iter().enumerate() {
            assert!((frequency - (index + 1) as f64 * drive_hz).abs() <= 1.0e-9);
        }
        let mouth_x = case.tier2_duct_length_m + case.tier2_chamber_length_m;
        assert_eq!(spec.observers.len(), 1);
        assert!((spec.observers[0].position_m[0] - (mouth_x + 1.0)).abs() <= 1.0e-12);
        assert_eq!(spec.observers[0].position_m[1], 0.0);
        assert_eq!(spec.observers[0].position_m[2], 0.0);
        spec.validate(21.0 * dt).unwrap();
        // `ka` at f1 is compact (Lighthill) while 2·f1 already trips FW–H.
        let ka = |f: f64| TAU * f * case.tier2_duct_length_m / case.sound_speed_m_per_s;
        assert!(ka(drive_hz) <= spec.compact_ka_max);
        assert!(ka(2.0 * drive_hz) > spec.compact_ka_max);
    }

    #[test]
    fn noise_spec_rejects_unanalysable_drive() {
        let case = PortOptimReferenceCase::default();
        let observer = noise_observer_1m_on_axis(&case);
        assert!(finalist_noise_spec(1000.0, 0.01, observer, 0.1, 1.0).is_err());
        assert!(finalist_noise_spec(100.0, 0.0, observer, 0.1, 1.0).is_err());
        assert!(finalist_noise_spec(100.0, 0.001, observer, 0.1, 0.0).is_err());
    }

    #[test]
    fn tail_filter_keeps_run_order() {
        let snapshots: Vec<PortFlowFieldSnapshot> = (0..10)
            .map(|step| PortFlowFieldSnapshot {
                time_s: step as f64 * 0.1,
                velocity_dofs_m_per_s: Vec::new(),
                pressure_dofs_pa: Vec::new(),
            })
            .collect();
        let tail = tail_snapshots(&snapshots, 0.5);
        assert_eq!(tail.len(), 5);
        assert!((tail[0].time_s - 0.5).abs() <= 1.0e-12);
    }

    fn window_stream(dt: f64, stride: usize, steps: usize) -> Vec<PortFlowFieldSnapshot> {
        (1..=steps)
            .filter(|step| step % stride == 0)
            .map(|step| PortFlowFieldSnapshot {
                time_s: step as f64 * dt,
                velocity_dofs_m_per_s: Vec::new(),
                pressure_dofs_pa: Vec::new(),
            })
            .collect()
    }

    #[test]
    fn tail_window_selects_run_end() {
        let dt = 2.5e-5;
        let stream = window_stream(dt, 21, 10_080);
        let window = tail_snapshot_window(&stream, 2016, 21, dt, 10_080).unwrap();
        assert_eq!(window.len(), 96);
        assert!((window[0].time_s - (10_080 - 2016 + 21) as f64 * dt).abs() <= 1.0e-15);
        assert!((window[95].time_s - 10_080.0 * dt).abs() <= 1.0e-15);
    }

    #[test]
    fn tail_window_rejects_misaligned_streams() {
        let dt = 2.5e-5;
        let stream = window_stream(dt, 21, 10_080);
        assert!(tail_snapshot_window(&stream, 2016, 0, dt, 10_080).is_err());
        assert!(tail_snapshot_window(&stream, 2017, 21, dt, 10_080).is_err());
        assert!(tail_snapshot_window(&stream, 42, 21, dt, 10_080).is_err());
        assert!(tail_snapshot_window(&stream[..10], 2016, 21, dt, 10_080).is_err());
        let mut gapped = stream.clone();
        gapped[400].time_s += dt;
        assert!(tail_snapshot_window(&gapped, 2016, 21, dt, 10_080).is_err());
        assert!(tail_snapshot_window(&stream, 2016, 21, dt, 9000).is_err());
    }

    #[test]
    fn agreement_gate_flags_ten_percent_boundary() {
        let agree = SpotCheck3dLevel::compare(1.0, 46.0, 1.0, 1.099, 1.0, 0.901).unwrap();
        assert_eq!(agree.verdict, SpotVerdict::Agree);
        let q_bad = SpotCheck3dLevel::compare(1.0, 46.0, 1.0, 1.101, 1.0, 1.0).unwrap();
        assert_eq!(q_bad.verdict, SpotVerdict::Investigate);
        let thd_bad = SpotCheck3dLevel::compare(1.0, 46.0, 1.0, 1.0, 1.0, 0.899).unwrap();
        assert_eq!(thd_bad.verdict, SpotVerdict::Investigate);
        assert_eq!(agree.recomputed_verdict(), SpotVerdict::Agree);
        assert_eq!(q_bad.recomputed_verdict(), SpotVerdict::Investigate);
        assert!(relative_ratio_disagreement(0.0, 1.0).is_err());
        assert!(relative_ratio_disagreement(1.0, f64::NAN).is_err());
        assert!(SpotCheck3dLevel::compare(0.0, 46.0, 1.0, 1.0, 1.0, 1.0).is_err());
    }

    #[test]
    fn spotcheck_fidelity_validates() {
        SpotCheck3dFidelity::smoke().validate().unwrap();
        let bad = SpotCheck3dFidelity {
            mesh_size_m: 0.0,
            ..SpotCheck3dFidelity::smoke()
        };
        assert!(bad.validate().is_err());
        let bad = SpotCheck3dFidelity {
            samples_per_cycle: 8,
            ..SpotCheck3dFidelity::smoke()
        };
        assert!(bad.validate().is_err());
        let bad = SpotCheck3dFidelity {
            tail_cycles: 0,
            ..SpotCheck3dFidelity::smoke()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn spot_drive_mirrors_tier2_convention() {
        assert!((spot_drive_frequency_hz(92.0).unwrap() - 46.0).abs() <= 1.0e-12);
        assert!(spot_drive_frequency_hz(0.0).is_err());
    }

    #[test]
    fn spline_3d_geometry_rejects_off_envelope_profiles() {
        let case = PortOptimReferenceCase::default();
        let template = spotcheck_template(&case);
        let straight = template.straight().unwrap();
        let geometry = spline_3d_geometry(&case, &straight, 0.05).unwrap();
        assert!((geometry.local_radius_m(0.05).unwrap() - case.port_radius_r0_m).abs() <= 1.0e-12);
        // Slot-scale Tier-2 profiles must fail loudly, never silently revolve.
        let slot = SplinePortProfile::from_normalized(
            5,
            case.tier2_duct_length_m,
            0.5 * case.tier2_duct_height_m,
            0.6 * 0.5 * case.tier2_duct_height_m,
            &[0.0, 0.0],
            super::super::campaign::campaign_profile_options(),
        )
        .unwrap();
        let off_envelope = spline_3d_geometry(&case, &slot, 0.05).unwrap();
        assert!(off_envelope.local_radius_m(0.05).is_err());
    }
}
