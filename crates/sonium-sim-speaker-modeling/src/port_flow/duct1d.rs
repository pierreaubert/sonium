//! Tier 0 variable-area duct model for port-shape screening (PORT-OPTIM.md Step 3).
//!
//! [`solve_duct_1d`] evaluates a [`SplinePortProfile`] with a segmented
//! plane-wave transfer-matrix cascade: every segment carries its local
//! characteristic impedance `Z_c = rho·c/S(x)` and propagation constant,
//! pressure `p` and volume velocity `Q` are continuous across area steps by
//! construction of the cascade, the exterior mouth sees the full complex
//! baffled-piston radiation impedance (never a pressure-release `p = 0`
//! idealisation), and the box presents the lumped compliance `C = V/(rho·c²)`.
//!
//! # Conventions
//!
//! * Axial coordinate `x ∈ [0, L]` shared with [`SplinePortProfile`]: `x = 0`
//!   is the box/interior end (the `MARKER_BOX_CAP` side of the duct mesh),
//!   `x = L` is the exterior/radiating end (`MARKER_EXTERIOR_CAP` side).
//! * Time-harmonic phasors use the engineering `exp(+i·ω·t)` convention with
//!   RMS amplitudes, matching [`PortSpec::acoustic_impedance`](crate::PortSpec).
//!   Acoustic inertance therefore has positive imaginary part (`Z_M = +i·ω·M`).
//! * Volume velocity is positive from the interior (box) toward the exterior.
//!   Input power at the box side is `Re(conj(p)·Q)`; radiated power is the
//!   same product at the mouth, i.e. `Re(Z_rad)·|Q_L|²`.
//!
//! # Transfer matrix
//!
//! Segment `j` of length `dx` with mid-station area `S_j` has, with complex
//! propagation constant `Γ_j` (`Γ = i·k` lossless, Kirchhoff complex below
//! when thermoviscous losses are enabled),
//!
//! ```text
//! [p_j]   [cosh(Γ·dx)  Z_c·sinh(Γ·dx)] [p_{j+1}]
//! [Q_j] = [sinh(Γ·dx)/Z_c  cosh(Γ·dx)] [Q_{j+1}]
//! ```
//!
//! and `[p_0; Q_0] = T_1·T_2·…·T_n·[p_L; Q_L]`. The cascade is second-order
//! accurate in `dx` (midpoint areas); `n = 128` keeps segmentation error below
//! 1e-5 in the port band (see the refinement test).
//!
//! # End-correction model (termination-aware, profile-dependent)
//!
//! This model deliberately does **not** use a fixed `1.46·r` total. Each mouth
//! carries the termination model of its own side, evaluated at a
//! profile-derived effective radius:
//!
//! * Exterior (baffle side): the full baffled-piston radiation impedance
//!   [`crate::baffled_piston_mechanical_radiation_impedance`] evaluated at the
//!   effective mouth radius `ā_out`, converted to acoustic ohms with the
//!   effective mouth area. Its reactance tends to the flanged value
//!   `8·ā/(3·π) ≈ 0.85·ā` as `ka → 0`; at finite `ka` the exact
//!   Bessel/Struve form is used, so no separate exterior length is added.
//! * Interior (box side): a lumped series inertance `M_in = rho·δ_in/S(0)`
//!   with `δ_in = 0.61·ā_in` (thin-wall unflanged value 0.6133 of
//!   Levine–Schwinger, rounded to the cabinet-design `0.61` free-side
//!   convention so the uniform limit matches the calibrated lumped model).
//!
//! The effective radii are area-equivalent means over the acoustic near field:
//! `ā = sqrt(mean(r²))` over one local radius of duct depth from each mouth
//! (capped at `L/4` for very short ducts). A straight tube reproduces
//! `δ_in = 0.61·R0` and the lumped radiation load bit-near-exactly, while a
//! profile that dips just inside a mouth shrinks that mouth's correction.
//!
//! Limits of this model (plan risk R4): the near-mouth averaging is a
//! heuristic bridge, exact only in the uniform limit; wall slope and curvature
//! enter only through the averaged radius, not through a horn-radiation
//! model; flange geometry and wall thickness are not represented; the
//! thermoviscous flag adds wide-tube duct loss but no mouth near-field loss.
//! Tier 1 (resolved viscothermal) re-ranks every finalist, so Tier 0 is never
//! used alone for a final claim.
//!
//! # Tuning convention
//!
//! [`tuning_frequency_hz`] reports the compact Helmholtz evaluation
//! `f0 = 1/(2·π·sqrt(M_total·C))` with
//! `M_total = rho·∫dx/S(x) + M_in + X_rad(f0)/ω0` (the radiation mass is
//! iterated to self-consistency). This keeps Tier-0 tuning exactly consistent
//! with the calibrated lumped infrastructure it screens for: on the straight
//! tube it matches [`EnclosureSpec::tuning_frequency_hz`](crate::EnclosureSpec)
//! to ~0.01%. The transfer matrix itself retains real distributed
//! (neck-compliance) effects of order `(k·L)²` — about 0.6% on the tuning
//! zero and up to ~3.5% on `Z_port` at the 150 Hz band edge for the reference
//! geometry — which is why the tuning screen uses the compact reduction while
//! [`Duct1dPoint::port_impedance_pa_s_per_m3`] reports the distributed value.
//!
//! # Thermoviscous option
//!
//! With [`Duct1dConfig::thermoviscous`] enabled, each segment propagates with
//! the Kirchhoff wide-tube constant
//! `Γ = i·k·(1 + (1-i)·(δ_v + (γ-1)·δ_t)/(2·r))`, where
//! `δ_v = sqrt(2·μ/(rho·ω))`, `δ_t = sqrt(2·α_th/ω)`, and `γ = 1.4` (dry air;
//! [`AcousticMedium`](crate::AcousticMedium) carries no ratio of specific
//! heats). `Z_c` keeps its lossless real value (first-order correction
//! neglected, `O(δ/r) ~ 1%`). Valid while `δ/r ≪ 1`; the audit field
//! [`Duct1dPoint::max_boundary_layer_ratio`] reports the worst segment.
//!
//! # Plane-wave validity
//!
//! The model resolves only the plane-wave mode: evanescent higher-order modes
//! excited at area steps are neglected. It is valid while the port band sits
//! far below the first circular-duct cutoff `f_c = 1.84·c/(2·π·R)` (≈4.0 kHz
//! at `R0 = 25 mm`, a > 25× margin over the 150 Hz band edge); every result
//! carries [`Duct1dPoint::plane_wave_margin`] (`f_c/f`) so callers can audit
//! this. Tier 0 cannot see flow separation or noise by construction.
//!
//! Validated envelope: linear plane-wave screening only. No flow solve, no
//! distortion or noise claim; the low-drive oracle remains the Tier 1
//! viscothermal reference.

use std::f64::consts::{PI, TAU};

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use super::error::PortFlowError;
use super::profile::SplinePortProfile;
use crate::{AcousticMedium, baffled_piston_mechanical_radiation_impedance};

// Serde note: `Complex64` intentionally has no field-level `Serialize` derive
// here, the same precedent as `ComplexPortTransfer` in `linear_reference.rs`;
// Tier 0 points are screening values, and any archived history serializes the
// profile plus the scalar observables it needs.

/// Default number of transfer-matrix segments (second-order midpoint areas).
pub const DUCT1D_DEFAULT_SEGMENTS: usize = 128;

/// Interior (box-side) end-correction factor: `δ_in = 0.61·ā_in`.
///
/// Thin-wall unflanged value (Levine–Schwinger 0.6133) rounded to the
/// cabinet-design free-side convention; see the module docs.
pub const INTERIOR_END_CORRECTION_FACTOR: f64 = 0.61;

/// First zero of `J₁′` (≈ 1.84), setting the first higher-order circular-duct
/// cutoff `f_c = ROOT·c/(2·π·R)`.
pub const FIRST_HIGHER_MODE_ROOT: f64 = 1.8412;

/// Ratio of specific heats used by the thermoviscous propagation constant.
///
/// Dry-air value; [`AcousticMedium`] carries density, sound speed, viscosity,
/// and thermal diffusivity but no `γ`, and Tier 0 targets air only.
const SPECIFIC_HEAT_RATIO_AIR: f64 = 1.4;

/// Trapezoid stations used for each near-mouth effective-radius average.
const MOUTH_ZONE_STATIONS: usize = 33;

/// Uniform stations (plus all knots) sampled for the cutoff radius audit.
const CUTOFF_SAMPLE_STATIONS: usize = 65;

/// Self-consistency iterations for the radiation mass in the tuning evaluation.
const TUNING_RADIATION_ITERATIONS: usize = 8;

/// Which duct mouth an effective radius or correction belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DuctEnd {
    /// Box/interior mouth at `x = 0` (the `MARKER_BOX_CAP` side).
    Interior,
    /// Exterior/radiating mouth at `x = L` (the `MARKER_EXTERIOR_CAP` side).
    Exterior,
}

/// Configuration of one Tier 0 duct evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Duct1dConfig {
    /// Number of transfer-matrix segments (`≥ 1`; 128 by convention).
    pub n_segments: usize,
    /// Enable Kirchhoff wide-tube thermoviscous propagation (default off:
    /// lossless, matching the lumped screening infrastructure).
    pub thermoviscous: bool,
    /// Enclosure volume in m³, setting the box compliance `C = V/(rho·c²)`.
    pub box_volume_m3: f64,
}

impl Duct1dConfig {
    /// Validate the configuration before any solve consumes it.
    pub fn validate(&self) -> Result<(), PortFlowError> {
        if self.n_segments < 1 {
            return Err(PortFlowError::InvalidConfig(format!(
                "Tier 0 duct needs at least 1 segment, got {}",
                self.n_segments
            )));
        }
        if !(self.box_volume_m3.is_finite() && self.box_volume_m3 > 0.0) {
            return Err(PortFlowError::InvalidConfig(format!(
                "Tier 0 box volume must be finite and positive, got {}",
                self.box_volume_m3
            )));
        }
        Ok(())
    }
}

/// One-frequency Tier 0 result: impedances, phasors, powers, and audits.
///
/// Phasors are RMS with the `exp(+i·ω·t)` convention; `Q` is positive from
/// the box toward the exterior. The drive is a unit box pressure
/// ([`Duct1dPoint::inlet_pressure_rms_pa`] `= 1 + 0i`).
#[derive(Debug, Clone)]
pub struct Duct1dPoint {
    /// Harmonic frequency in Hz.
    pub frequency_hz: f64,
    /// Distributed port-branch impedance `p_box/Q_0` in Pa·s/m³.
    ///
    /// Full transfer-matrix cascade with the interior series mass and the
    /// radiation-impedance termination.
    pub port_impedance_pa_s_per_m3: Complex64,
    /// Compact port-branch impedance `Z_rad + i·ω·(M_duct + M_in)`.
    ///
    /// Same physical parameters, lumped reduction; agrees with
    /// [`PortSpec::acoustic_impedance`](crate::PortSpec) on the straight tube
    /// across the port band (< 0.1%).
    pub compact_port_impedance_pa_s_per_m3: Complex64,
    /// Duct-cascade input impedance before the interior series mass.
    pub duct_inlet_impedance_pa_s_per_m3: Complex64,
    /// Exterior radiation load `Z_rad` in Pa·s/m³ (baffled piston at `ā_out`).
    pub radiation_impedance_pa_s_per_m3: Complex64,
    /// Interior series impedance `i·ω·M_in` in Pa·s/m³ (lossless by model).
    pub interior_series_impedance_pa_s_per_m3: Complex64,
    /// Box-to-port flow ratio `Q_port/Q_diaphragm = 1/(1 + i·ω·C·Z_port)`.
    pub box_to_port_flow_ratio: Complex64,
    /// Box-side pressure phasor in Pa RMS (the unit drive, `1 + 0i`).
    pub inlet_pressure_rms_pa: Complex64,
    /// Duct-inlet volume velocity in m³/s RMS, positive box-to-exterior.
    pub inlet_volume_velocity_rms_m3_per_s: Complex64,
    /// Mouth pressure phasor in Pa RMS (`Z_rad·Q_L`, never forced to zero).
    pub outlet_pressure_rms_pa: Complex64,
    /// Mouth volume velocity in m³/s RMS, positive box-to-exterior.
    pub outlet_volume_velocity_rms_m3_per_s: Complex64,
    /// Acoustic power entering the port branch from the box, in W.
    pub input_power_w: f64,
    /// Acoustic power radiated at the mouth, in W.
    pub radiated_power_w: f64,
    /// Interior end correction `δ_in = 0.61·ā_in` in m.
    pub interior_end_correction_m: f64,
    /// Exterior equivalent length `X_rad·S_eff/(ω·rho)` in m (reporting only;
    /// the solve uses the full complex `Z_rad`, not this length).
    pub exterior_end_correction_m: f64,
    /// Effective interior mouth radius `ā_in` in m.
    pub effective_interior_radius_m: f64,
    /// Effective exterior mouth radius `ā_out` in m.
    pub effective_exterior_radius_m: f64,
    /// Duct acoustic inertance `rho·∫₀ᴸ dx/S(x)` in kg/m⁴.
    pub duct_inertance_kg_per_m4: f64,
    /// First higher-order circular-duct cutoff `f_c` in Hz.
    pub cutoff_frequency_hz: f64,
    /// Plane-wave margin `f_c/f` (must be ≫ 1 for the model to apply).
    pub plane_wave_margin: f64,
    /// Worst-segment viscous boundary-layer ratio `δ_v/r` (validity audit;
    /// reported whether or not thermoviscous propagation is enabled).
    pub max_boundary_layer_ratio: f64,
}

fn invalid(message: impl Into<String>) -> PortFlowError {
    PortFlowError::InvalidConfig(message.into())
}

fn check_medium(medium: AcousticMedium) -> Result<(), PortFlowError> {
    medium
        .validate()
        .map_err(|error| invalid(format!("Tier 0 acoustic medium invalid: {error}")))
}

fn check_frequency(frequency_hz: f64) -> Result<(), PortFlowError> {
    if !(frequency_hz.is_finite() && frequency_hz > 0.0) {
        return Err(invalid(format!(
            "Tier 0 frequency must be finite and positive, got {frequency_hz}"
        )));
    }
    Ok(())
}

fn check_segments(n_segments: usize) -> Result<(), PortFlowError> {
    if n_segments < 1 {
        return Err(invalid(format!(
            "Tier 0 duct needs at least 1 segment, got {n_segments}"
        )));
    }
    Ok(())
}

fn complex_is_finite(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

/// Lumped box compliance `C = V/(rho·c²)` in m⁵/N (m³/Pa).
fn box_compliance_m5_per_n(
    box_volume_m3: f64,
    medium: AcousticMedium,
) -> Result<f64, PortFlowError> {
    if !(box_volume_m3.is_finite() && box_volume_m3 > 0.0) {
        return Err(invalid(format!(
            "Tier 0 box volume must be finite and positive, got {box_volume_m3}"
        )));
    }
    let compliance =
        box_volume_m3 / (medium.density_kg_per_m3 * medium.sound_speed_m_per_s.powi(2));
    if !(compliance.is_finite() && compliance > 0.0) {
        return Err(invalid(format!(
            "Tier 0 box compliance must be finite and positive, got {compliance}"
        )));
    }
    Ok(compliance)
}

/// Effective mouth radius: area-equivalent mean over the near-mouth zone.
///
/// `ā = sqrt(mean(r²))` over one local mouth radius of duct depth, capped at
/// `L/4` for very short ducts. Exact (`R0`) on the straight tube.
pub fn effective_mouth_radius_m(profile: &SplinePortProfile, end: DuctEnd) -> f64 {
    let length = profile.length_m();
    let mouth_radius = match end {
        DuctEnd::Interior => profile.local_radius_m(0.0),
        DuctEnd::Exterior => profile.local_radius_m(length),
    };
    // Profile validation guarantees positive radii and length, so the zone
    // below is always positive; the L/4 cap only binds for stubby ducts.
    let zone = mouth_radius.min(0.25 * length);
    let start = match end {
        DuctEnd::Interior => 0.0,
        DuctEnd::Exterior => length - zone,
    };
    let mut sum = 0.0;
    for i in 0..MOUTH_ZONE_STATIONS {
        let x = start + zone * i as f64 / (MOUTH_ZONE_STATIONS - 1) as f64;
        let r = profile.local_radius_m(x);
        let weight = if i == 0 || i == MOUTH_ZONE_STATIONS - 1 {
            0.5
        } else {
            1.0
        };
        sum += weight * r * r;
    }
    (sum / (MOUTH_ZONE_STATIONS - 1) as f64).sqrt()
}

/// Interior end correction `δ_in = 0.61·ā_in` in m (profile-dependent).
pub fn interior_end_correction_m(profile: &SplinePortProfile) -> f64 {
    INTERIOR_END_CORRECTION_FACTOR * effective_mouth_radius_m(profile, DuctEnd::Interior)
}

/// Duct acoustic inertance `rho·∫₀ᴸ dx/S(x)` in kg/m⁴ (midpoint rule).
pub fn duct_inertance_kg_per_m4(
    profile: &SplinePortProfile,
    density_kg_per_m3: f64,
    n_segments: usize,
) -> Result<f64, PortFlowError> {
    profile.validate()?;
    check_segments(n_segments)?;
    if !(density_kg_per_m3.is_finite() && density_kg_per_m3 > 0.0) {
        return Err(invalid(format!(
            "Tier 0 density must be finite and positive, got {density_kg_per_m3}"
        )));
    }
    let length = profile.length_m();
    let dx = length / n_segments as f64;
    let mut integral = 0.0;
    for j in 0..n_segments {
        let x_mid = (j as f64 + 0.5) * dx;
        integral += dx / profile.area_m2(x_mid);
    }
    let inertance = density_kg_per_m3 * integral;
    if !(inertance.is_finite() && inertance > 0.0) {
        return Err(invalid(format!(
            "Tier 0 duct inertance must be finite and positive, got {inertance}"
        )));
    }
    Ok(inertance)
}

/// Fidelity tier selected by frequency against the duct cutoff.
///
/// The one-dimensional plane-wave tier is used only sufficiently below
/// the first higher-order cutoff; at and above half the cutoff the
/// resolved or modal tier is required, because evanescent higher-order
/// modes still change the local field near discontinuities even when
/// they carry no far-field power.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PortFidelity {
    /// One-dimensional transfer-matrix tier (Tier 0/1).
    OneDimensional,
    /// Resolved or modal tier (Tier 2).
    ResolvedModal,
}

/// Selects the port fidelity for `frequency_hz` against `cutoff_hz`.
///
/// Returns [`PortFidelity::OneDimensional`] only when the frequency is
/// below half the cutoff (plane-wave margin `f_c/f > 2`); otherwise the
/// resolved/modal tier is required. Both inputs must be finite and
/// positive; invalid inputs fail closed to the resolved tier by
/// returning an error from the caller's validation, so this selector
/// itself documents the threshold in one place.
///
/// # Panics
///
/// Panics only on a programming error: non-finite or non-positive
/// inputs are contract violations caught by caller validation.
#[must_use]
pub fn select_port_fidelity(frequency_hz: f64, cutoff_hz: f64) -> PortFidelity {
    assert!(
        frequency_hz.is_finite() && frequency_hz > 0.0 && cutoff_hz.is_finite() && cutoff_hz > 0.0,
        "port fidelity inputs must be finite and positive"
    );
    if frequency_hz * 2.0 < cutoff_hz {
        PortFidelity::OneDimensional
    } else {
        PortFidelity::ResolvedModal
    }
}

/// First higher-order circular-duct cutoff `f_c = 1.8412·c/(2·π·R)` in Hz.
pub fn cutoff_frequency_hz(medium: AcousticMedium, radius_m: f64) -> Result<f64, PortFlowError> {
    check_medium(medium)?;
    if !(radius_m.is_finite() && radius_m > 0.0) {
        return Err(invalid(format!(
            "Tier 0 cutoff radius must be finite and positive, got {radius_m}"
        )));
    }
    let cutoff = FIRST_HIGHER_MODE_ROOT * medium.sound_speed_m_per_s / (TAU * radius_m);
    if !(cutoff.is_finite() && cutoff > 0.0) {
        return Err(invalid(format!(
            "Tier 0 cutoff frequency must be finite and positive, got {cutoff}"
        )));
    }
    Ok(cutoff)
}

/// Largest profile radius over knots plus a uniform sample (cutoff audit).
fn max_profile_radius_m(profile: &SplinePortProfile) -> f64 {
    let length = profile.length_m();
    let mut max = profile.radii().iter().copied().fold(0.0f64, f64::max);
    for i in 0..CUTOFF_SAMPLE_STATIONS {
        let x = length * i as f64 / (CUTOFF_SAMPLE_STATIONS - 1) as f64;
        max = max.max(profile.local_radius_m(x));
    }
    max
}

/// Exterior radiation load: baffled piston at `ā_out`, in acoustic ohms.
fn radiation_impedance(
    effective_radius_m: f64,
    frequency_hz: f64,
    medium: AcousticMedium,
) -> Result<Complex64, PortFlowError> {
    let mechanical =
        baffled_piston_mechanical_radiation_impedance(effective_radius_m, frequency_hz, medium)
            .map_err(|error| invalid(format!("Tier 0 radiation impedance failed: {error}")))?;
    let area = PI * effective_radius_m * effective_radius_m;
    let acoustic = mechanical / (area * area);
    if !complex_is_finite(acoustic) {
        return Err(invalid(format!(
            "Tier 0 radiation impedance must be finite, got {acoustic:?}"
        )));
    }
    Ok(acoustic)
}

/// Row-major complex 2×2 transfer matrix `[A, B, C, D]`.
type Abcd = [Complex64; 4];

fn abcd_identity() -> Abcd {
    [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ]
}

fn abcd_multiply(left: &Abcd, right: &Abcd) -> Abcd {
    [
        left[0] * right[0] + left[1] * right[2],
        left[0] * right[1] + left[1] * right[3],
        left[2] * right[0] + left[3] * right[2],
        left[2] * right[1] + left[3] * right[3],
    ]
}

/// Cascade the segmented duct into one ABCD matrix plus audit quantities.
///
/// Returns `(total, duct_inertance, max_boundary_layer_ratio)`.
fn cascade_duct(
    profile: &SplinePortProfile,
    medium: AcousticMedium,
    frequency_hz: f64,
    n_segments: usize,
    thermoviscous: bool,
) -> Result<(Abcd, f64, f64), PortFlowError> {
    let omega = TAU * frequency_hz;
    let wavenumber = omega / medium.sound_speed_m_per_s;
    let length = profile.length_m();
    let dx = length / n_segments as f64;
    // Boundary-layer depths are frequency-global; only the local radius varies.
    let viscous_depth_m =
        (2.0 * medium.dynamic_viscosity_pa_s / (medium.density_kg_per_m3 * omega)).sqrt();
    let thermal_depth_m = (2.0 * medium.thermal_diffusivity_m2_per_s / omega).sqrt();
    let loss_depth_m = viscous_depth_m + (SPECIFIC_HEAT_RATIO_AIR - 1.0) * thermal_depth_m;

    let mut total = abcd_identity();
    let mut inertance_integral = 0.0;
    let mut max_layer_ratio = 0.0f64;
    let one = Complex64::new(1.0, 0.0);
    let eye = Complex64::new(0.0, 1.0);
    for j in 0..n_segments {
        let x_mid = (j as f64 + 0.5) * dx;
        let radius = profile.local_radius_m(x_mid);
        let area = PI * radius * radius;
        let characteristic = medium.density_kg_per_m3 * medium.sound_speed_m_per_s / area;
        // Kirchhoff wide-tube propagation; lossless Γ = i·k when disabled.
        let beta = if thermoviscous {
            loss_depth_m / (2.0 * radius)
        } else {
            0.0
        };
        let gamma = eye * wavenumber * (one + (one - eye) * Complex64::new(beta, 0.0));
        let gd = gamma * dx;
        let cosh = gd.cosh();
        let sinh = gd.sinh();
        let segment = [
            cosh,
            Complex64::new(characteristic, 0.0) * sinh,
            sinh / Complex64::new(characteristic, 0.0),
            cosh,
        ];
        total = abcd_multiply(&total, &segment);
        inertance_integral += dx / area;
        max_layer_ratio = max_layer_ratio.max(viscous_depth_m / radius);
    }
    for entry in total {
        if !complex_is_finite(entry) {
            return Err(invalid(
                "Tier 0 duct cascade produced a non-finite transfer matrix",
            ));
        }
    }
    let inertance = medium.density_kg_per_m3 * inertance_integral;
    if !(inertance.is_finite() && inertance > 0.0) {
        return Err(invalid(format!(
            "Tier 0 duct inertance must be finite and positive, got {inertance}"
        )));
    }
    Ok((total, inertance, max_layer_ratio))
}

/// Compact (lumped-reduction) port impedance `Z_rad + i·ω·(M_duct + M_in)`.
///
/// Same physical parameters as the transfer matrix — profile inertance,
/// profile-dependent interior mass, baffled-piston radiation at `ā_out` —
/// without distributed effects. This is the quantity that reproduces
/// [`PortSpec::acoustic_impedance`](crate::PortSpec) on the straight tube.
pub fn compact_port_impedance_pa_s_per_m3(
    profile: &SplinePortProfile,
    medium: AcousticMedium,
    frequency_hz: f64,
    n_segments: usize,
) -> Result<Complex64, PortFlowError> {
    profile.validate()?;
    check_medium(medium)?;
    check_frequency(frequency_hz)?;
    check_segments(n_segments)?;
    let duct_inertance = duct_inertance_kg_per_m4(profile, medium.density_kg_per_m3, n_segments)?;
    let mouth_area = profile.area_m2(0.0);
    let interior_mass = medium.density_kg_per_m3 * interior_end_correction_m(profile) / mouth_area;
    let exterior_radius = effective_mouth_radius_m(profile, DuctEnd::Exterior);
    let radiation = radiation_impedance(exterior_radius, frequency_hz, medium)?;
    let impedance =
        radiation + Complex64::new(0.0, TAU * frequency_hz * (duct_inertance + interior_mass));
    if !complex_is_finite(impedance) {
        return Err(invalid("Tier 0 compact port impedance must be finite"));
    }
    Ok(impedance)
}

/// Tier 0 tuning frequency: compact Helmholtz evaluation in Hz.
///
/// `f0 = 1/(2·π·sqrt(M_total·C))` with
/// `M_total = rho·∫dx/S + M_in + X_rad(f0)/ω0`; the radiation mass is
/// iterated to self-consistency from the duct-plus-interior seed. Matches
/// [`EnclosureSpec::tuning_frequency_hz`](crate::EnclosureSpec) on the
/// straight tube to ~0.01% (see the module docs for why tuning uses the
/// compact reduction while impedance stays distributed).
pub fn tuning_frequency_hz(
    profile: &SplinePortProfile,
    medium: AcousticMedium,
    config: &Duct1dConfig,
) -> Result<f64, PortFlowError> {
    profile.validate()?;
    check_medium(medium)?;
    config.validate()?;
    let compliance = box_compliance_m5_per_n(config.box_volume_m3, medium)?;
    let duct_inertance =
        duct_inertance_kg_per_m4(profile, medium.density_kg_per_m3, config.n_segments)?;
    let mouth_area = profile.area_m2(0.0);
    let interior_mass = medium.density_kg_per_m3 * interior_end_correction_m(profile) / mouth_area;
    let fixed_mass = duct_inertance + interior_mass;
    let exterior_radius = effective_mouth_radius_m(profile, DuctEnd::Exterior);

    let mut frequency = 1.0 / (TAU * (fixed_mass * compliance).sqrt());
    for _ in 0..TUNING_RADIATION_ITERATIONS {
        check_frequency(frequency)?;
        let radiation = radiation_impedance(exterior_radius, frequency, medium)?;
        let radiation_mass = radiation.im / (TAU * frequency);
        frequency = 1.0 / (TAU * ((fixed_mass + radiation_mass) * compliance).sqrt());
    }
    check_frequency(frequency)?;
    Ok(frequency)
}

/// Evaluate the Tier 0 duct at one frequency.
///
/// Cascades the segmented transfer matrix with the radiation-impedance
/// termination, prepends the interior series mass, and drives the branch
/// with a unit box pressure. Returns impedances, phasors, powers, and the
/// plane-wave / boundary-layer audits in [`Duct1dPoint`].
pub fn solve_duct_1d(
    profile: &SplinePortProfile,
    medium: AcousticMedium,
    frequency_hz: f64,
    config: &Duct1dConfig,
) -> Result<Duct1dPoint, PortFlowError> {
    profile.validate()?;
    check_medium(medium)?;
    check_frequency(frequency_hz)?;
    config.validate()?;

    let omega = TAU * frequency_hz;
    let compliance = box_compliance_m5_per_n(config.box_volume_m3, medium)?;
    let interior_radius = effective_mouth_radius_m(profile, DuctEnd::Interior);
    let exterior_radius = effective_mouth_radius_m(profile, DuctEnd::Exterior);
    let end_correction = interior_end_correction_m(profile);
    let mouth_area = profile.area_m2(0.0);
    let interior_mass = medium.density_kg_per_m3 * end_correction / mouth_area;
    let interior_series = Complex64::new(0.0, omega * interior_mass);
    let radiation = radiation_impedance(exterior_radius, frequency_hz, medium)?;

    let (total, duct_inertance, max_layer_ratio) = cascade_duct(
        profile,
        medium,
        frequency_hz,
        config.n_segments,
        config.thermoviscous,
    )?;
    // Termination p_L = Z_rad·Q_L: Z_duct_in = (A·Z + B)/(C·Z + D).
    let duct_inlet = (total[0] * radiation + total[1]) / (total[2] * radiation + total[3]);
    if !complex_is_finite(duct_inlet) {
        return Err(invalid("Tier 0 duct inlet impedance must be finite"));
    }
    let port_impedance = interior_series + duct_inlet;
    let compact =
        compact_port_impedance_pa_s_per_m3(profile, medium, frequency_hz, config.n_segments)?;

    // Unit box-pressure drive; T⁻¹ = [[D, -B], [-C, A]] (det = 1).
    let inlet_pressure = Complex64::new(1.0, 0.0);
    let inlet_flow = inlet_pressure / port_impedance;
    let duct_pressure = inlet_pressure - interior_series * inlet_flow;
    let outlet_pressure = total[3] * duct_pressure - total[1] * inlet_flow;
    let outlet_flow = -total[2] * duct_pressure + total[0] * inlet_flow;
    for (name, value) in [
        ("port impedance", port_impedance),
        ("inlet volume velocity", inlet_flow),
        ("outlet pressure", outlet_pressure),
        ("outlet volume velocity", outlet_flow),
    ] {
        if !complex_is_finite(value) {
            return Err(invalid(format!(
                "Tier 0 {name} must be finite, got {value:?}"
            )));
        }
    }

    let input_power = (inlet_pressure.conj() * inlet_flow).re;
    let radiated_power = (outlet_pressure.conj() * outlet_flow).re;
    let flow_ratio = Complex64::new(1.0, 0.0)
        / (Complex64::new(1.0, 0.0) + Complex64::new(0.0, omega * compliance) * port_impedance);
    if !complex_is_finite(flow_ratio) {
        return Err(invalid("Tier 0 box-to-port flow ratio must be finite"));
    }
    let exterior_area = PI * exterior_radius * exterior_radius;
    let exterior_length = radiation.im * exterior_area / (omega * medium.density_kg_per_m3);
    let cutoff = cutoff_frequency_hz(medium, max_profile_radius_m(profile))?;

    Ok(Duct1dPoint {
        frequency_hz,
        port_impedance_pa_s_per_m3: port_impedance,
        compact_port_impedance_pa_s_per_m3: compact,
        duct_inlet_impedance_pa_s_per_m3: duct_inlet,
        radiation_impedance_pa_s_per_m3: radiation,
        interior_series_impedance_pa_s_per_m3: interior_series,
        box_to_port_flow_ratio: flow_ratio,
        inlet_pressure_rms_pa: inlet_pressure,
        inlet_volume_velocity_rms_m3_per_s: inlet_flow,
        outlet_pressure_rms_pa: outlet_pressure,
        outlet_volume_velocity_rms_m3_per_s: outlet_flow,
        input_power_w: input_power,
        radiated_power_w: radiated_power,
        interior_end_correction_m: end_correction,
        exterior_end_correction_m: exterior_length,
        effective_interior_radius_m: interior_radius,
        effective_exterior_radius_m: exterior_radius,
        duct_inertance_kg_per_m4: duct_inertance,
        cutoff_frequency_hz: cutoff,
        plane_wave_margin: cutoff / frequency_hz,
        max_boundary_layer_ratio: max_layer_ratio,
    })
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::super::profile::ProfileOptions;
    use super::*;
    use crate::{EnclosureSpec, PortSpec};

    const R0: f64 = 0.025;
    const L: f64 = 0.1;
    const RMIN: f64 = 0.015;
    const VBOX: f64 = 0.005;

    /// Step-0 baseline medium magnitudes (NOT `standard_air`, by fixture).
    fn medium() -> AcousticMedium {
        AcousticMedium {
            density_kg_per_m3: 1.2,
            sound_speed_m_per_s: 343.0,
            dynamic_viscosity_pa_s: 1.85e-5,
            thermal_diffusivity_m2_per_s: 2.2e-5,
        }
    }

    fn config() -> Duct1dConfig {
        Duct1dConfig {
            n_segments: DUCT1D_DEFAULT_SEGMENTS,
            thermoviscous: false,
            box_volume_m3: VBOX,
        }
    }

    fn straight() -> SplinePortProfile {
        SplinePortProfile::straight(5, L, R0, RMIN, ProfileOptions::default())
            .expect("straight test profile must construct")
    }

    fn dipped() -> SplinePortProfile {
        // u = 0.7 keeps the wall slope (≈0.42) inside the 0.5 limit; u = 1.0
        // would hit ≈0.6 and be rejected by construction.
        SplinePortProfile::from_normalized(5, L, R0, RMIN, &[0.7, 0.7], ProfileOptions::default())
            .expect("dipped test profile must construct")
    }

    /// Profile whose wall already slopes inside both near-mouth zones.
    ///
    /// `from_normalized` pins two inlet knots to `R0` (straight start), which
    /// legitimately leaves the interior correction of coarse-uniform profiles
    /// at the straight value; this absolute-radii polygon probes the
    /// correction's geometric response at both mouths instead.
    fn mouth_tapered() -> SplinePortProfile {
        SplinePortProfile::try_from_radii(
            &[R0, 0.9 * R0, 0.8 * R0, 0.9 * R0, R0],
            L,
            R0,
            RMIN,
            ProfileOptions::default(),
        )
        .expect("tapered test profile must construct")
    }

    fn lumped_port() -> PortSpec {
        PortSpec {
            radius_m: R0,
            length_m: L,
            end_correction_m: 1.46 * R0,
            resistance_pa_s_per_m3: 0.0,
        }
    }

    fn lumped_enclosure() -> EnclosureSpec {
        EnclosureSpec::Vented {
            volume_m3: VBOX,
            port: lumped_port(),
            leakage_resistance_pa_s_per_m3: 0.0,
            absorption_resistance_pa_s_per_m3: 0.0,
        }
    }

    #[test]
    fn uniform_compact_impedance_matches_lumped_across_band() {
        let profile = straight();
        let medium = medium();
        for frequency in [30.0, 60.0, 92.6, 120.0, 150.0] {
            let tier0 = compact_port_impedance_pa_s_per_m3(
                &profile,
                medium,
                frequency,
                DUCT1D_DEFAULT_SEGMENTS,
            )
            .expect("compact impedance must evaluate");
            let lumped = lumped_port()
                .acoustic_impedance(frequency, medium)
                .expect("lumped impedance must evaluate");
            let error = (tier0 - lumped).norm() / lumped.norm();
            assert!(
                error < 1.0e-3,
                "compact vs lumped at {frequency} Hz: rel err {error}"
            );
        }
    }

    #[test]
    fn uniform_distributed_impedance_matches_lumped_in_compact_limit() {
        let profile = straight();
        let medium = medium();
        let config = config();
        for frequency in [2.0, 5.0, 10.0] {
            let point = solve_duct_1d(&profile, medium, frequency, &config)
                .expect("duct solve must succeed");
            let lumped = lumped_port()
                .acoustic_impedance(frequency, medium)
                .expect("lumped impedance must evaluate");
            let error = (point.port_impedance_pa_s_per_m3 - lumped).norm() / lumped.norm();
            assert!(
                error < 1.0e-3,
                "distributed vs lumped at {frequency} Hz: rel err {error}"
            );
        }
    }

    #[test]
    fn distributed_deviation_grows_like_kl_squared_in_band() {
        // Real neck-compliance physics, not implementation error: the transfer
        // matrix keeps O((k·L)²) distributed effects the lumped model drops.
        let profile = straight();
        let medium = medium();
        let config = config();
        let mut previous = 0.0;
        for frequency in [30.0, 60.0, 92.6, 120.0, 150.0] {
            let point = solve_duct_1d(&profile, medium, frequency, &config)
                .expect("duct solve must succeed");
            let lumped = lumped_port()
                .acoustic_impedance(frequency, medium)
                .expect("lumped impedance must evaluate");
            let error = (point.port_impedance_pa_s_per_m3 - lumped).norm() / lumped.norm();
            assert!(
                error > previous,
                "distributed deviation must grow with f: {error} <= {previous} at {frequency} Hz"
            );
            previous = error;
        }
        assert!(
            previous < 0.10,
            "band-edge distributed deviation must stay bounded: {previous}"
        );
        assert!(
            previous > 5.0e-3,
            "model must not be secretly lumped: band-edge deviation {previous}"
        );
    }

    #[test]
    fn uniform_tuning_matches_lumped() {
        let profile = straight();
        let medium = medium();
        let tier0 = tuning_frequency_hz(&profile, medium, &config()).expect("tuning must solve");
        let lumped = lumped_enclosure()
            .tuning_frequency_hz(medium)
            .expect("lumped tuning must solve")
            .expect("vented enclosure has a tuning");
        assert!(
            (lumped - 92.59).abs() < 0.93,
            "lumped sanity anchor moved: {lumped}"
        );
        let error = (tier0 - lumped) / lumped;
        assert!(error.abs() < 1.0e-3, "tuning rel err {error}");
    }

    #[test]
    fn dipped_profile_tunes_below_straight() {
        let medium = medium();
        let config = config();
        let straight_f0 =
            tuning_frequency_hz(&straight(), medium, &config).expect("tuning must solve");
        let dipped_f0 = tuning_frequency_hz(&dipped(), medium, &config).expect("tuning must solve");
        assert!(
            dipped_f0 < straight_f0,
            "throat mass must lower tuning: dipped {dipped_f0} vs straight {straight_f0}"
        );
    }

    #[test]
    fn halved_area_doubles_inertance() {
        // Uniform tube at R0/√2 via absolute radii (no end pinning there).
        let half_area_radius = R0 / 2.0f64.sqrt();
        let narrow = SplinePortProfile::try_from_radii(
            &[half_area_radius; 5],
            L,
            R0,
            RMIN,
            ProfileOptions::default(),
        )
        .expect("uniform narrow tube must construct");
        let straight_inertance =
            duct_inertance_kg_per_m4(&straight(), 1.2, DUCT1D_DEFAULT_SEGMENTS)
                .expect("inertance must evaluate");
        let analytic = 1.2 * L / (PI * R0 * R0);
        assert!(
            (straight_inertance - analytic).abs() / analytic < 1.0e-12,
            "straight inertance must equal rho·L/S"
        );
        let narrow_inertance = duct_inertance_kg_per_m4(&narrow, 1.2, DUCT1D_DEFAULT_SEGMENTS)
            .expect("inertance must evaluate");
        let ratio = narrow_inertance / straight_inertance;
        assert!(
            (ratio - 2.0).abs() < 1.0e-9,
            "halved area must double ∫dx/S inertance: ratio {ratio}"
        );
    }

    #[test]
    fn inertance_matches_independent_quadrature() {
        let helper = duct_inertance_kg_per_m4(&dipped(), 1.2, DUCT1D_DEFAULT_SEGMENTS)
            .expect("inertance must evaluate");
        // Independent dense trapezoid (different rule, different resolution).
        let profile = dipped();
        let n = 2001;
        let mut sum = 0.0;
        for i in 0..n {
            let x = L * i as f64 / (n - 1) as f64;
            let weight = if i == 0 || i == n - 1 { 0.5 } else { 1.0 };
            sum += weight / profile.area_m2(x);
        }
        let independent = 1.2 * sum * L / (n - 1) as f64;
        let error = (helper - independent).abs() / independent;
        assert!(error < 1.0e-6, "inertance quadrature rel err {error}");
    }

    #[test]
    fn lossless_power_balance_holds() {
        let medium = medium();
        let config = config();
        for profile in [straight(), dipped()] {
            for frequency in [30.0, 92.6, 150.0] {
                let point = solve_duct_1d(&profile, medium, frequency, &config)
                    .expect("duct solve must succeed");
                assert!(
                    point.input_power_w > 0.0,
                    "passive lossy termination must absorb power"
                );
                let imbalance =
                    (point.input_power_w - point.radiated_power_w).abs() / point.input_power_w;
                assert!(
                    imbalance < 1.0e-9,
                    "lossless power imbalance {imbalance} at {frequency} Hz"
                );
            }
        }
    }

    #[test]
    fn thermoviscous_adds_dissipation() {
        let medium = medium();
        let lossless = config();
        let lossy = Duct1dConfig {
            thermoviscous: true,
            ..lossless
        };
        let profile = dipped();
        let plain =
            solve_duct_1d(&profile, medium, 92.6, &lossless).expect("lossless solve must succeed");
        let lossy_point =
            solve_duct_1d(&profile, medium, 92.6, &lossy).expect("lossy solve must succeed");
        assert!(
            lossy_point.port_impedance_pa_s_per_m3.re > plain.port_impedance_pa_s_per_m3.re,
            "thermoviscous loss must raise Re(Z_port)"
        );
        // Wide-tube duct resistance (~ωM·δ/r) exceeds the ka² radiation
        // resistance by several times at 92.6 Hz in a 25 mm port, so most of
        // the input power dissipates in the duct (port Q of order tens, the
        // known viscothermal regime) — the small number is the per-length
        // attenuation αL ~ 1e-3, not the circuit dissipation ratio.
        let dissipated =
            (lossy_point.input_power_w - lossy_point.radiated_power_w) / lossy_point.input_power_w;
        assert!(
            dissipated > 0.05 && dissipated < 1.0,
            "dissipated fraction must be substantial but passive: {dissipated}"
        );
        let layer_ratio = lossy_point.max_boundary_layer_ratio;
        assert!(
            layer_ratio > 0.0 && layer_ratio < 0.05,
            "wide-tube audit out of range: {layer_ratio}"
        );
    }

    #[test]
    fn screening_eval_is_sub_10ms() {
        // Typical Tier-0 screen: a band sweep plus a tuning evaluation.
        let profile = dipped();
        let medium = medium();
        let config = config();
        let start = Instant::now();
        for k in 0..21 {
            let frequency = 30.0 + 6.0 * k as f64;
            let _ = solve_duct_1d(&profile, medium, frequency, &config)
                .expect("duct solve must succeed");
        }
        let _ = tuning_frequency_hz(&profile, medium, &config).expect("tuning must solve");
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_secs_f64() < 0.010,
            "Tier 0 screen took {elapsed:?}, budget is 10 ms"
        );
    }

    #[test]
    fn port_band_is_far_below_cutoff() {
        let medium = medium();
        let cutoff = cutoff_frequency_hz(medium, R0).expect("cutoff must evaluate");
        assert!(
            (cutoff - 4020.0).abs() / 4020.0 < 0.01,
            "cutoff anchor moved: {cutoff}"
        );
        assert!(
            cutoff / 150.0 > 20.0,
            "plane-wave margin at the 150 Hz band edge must exceed 20×"
        );
        let point =
            solve_duct_1d(&straight(), medium, 150.0, &config()).expect("duct solve must succeed");
        assert!(
            (point.plane_wave_margin - cutoff / 150.0).abs() < 1.0e-9,
            "point must carry a consistent plane-wave margin"
        );
    }

    #[test]
    fn end_correction_is_profile_dependent() {
        let straight_correction = interior_end_correction_m(&straight());
        assert!(
            (straight_correction - 0.61 * R0).abs() / (0.61 * R0) < 1.0e-12,
            "straight tube must recover 0.61·R0, got {straight_correction}"
        );
        let straight_mouth = effective_mouth_radius_m(&straight(), DuctEnd::Exterior);
        assert!(
            (straight_mouth - R0).abs() / R0 < 1.0e-12,
            "straight effective mouth must equal R0, got {straight_mouth}"
        );
        // The tapered wall sits below R0 throughout both near-mouth zones,
        // so both corrections must shrink by a real (not roundoff) amount.
        let tapered = mouth_tapered();
        let tapered_correction = interior_end_correction_m(&tapered);
        assert!(
            tapered_correction < straight_correction * (1.0 - 1.0e-6),
            "tapered interior correction must shrink: {tapered_correction} vs {straight_correction}"
        );
        let tapered_mouth = effective_mouth_radius_m(&tapered, DuctEnd::Exterior);
        assert!(
            tapered_mouth < R0 * (1.0 - 1.0e-6),
            "tapered effective mouth must shrink: {tapered_mouth} vs {R0}"
        );
        // The pinned-end optimizer profile still varies its exterior mouth:
        // only its inlet transition sits outside the one-radius zone.
        let dipped_mouth = effective_mouth_radius_m(&dipped(), DuctEnd::Exterior);
        assert!(
            dipped_mouth < R0 * (1.0 - 1.0e-6),
            "dipped effective exterior mouth must shrink: {dipped_mouth} vs {R0}"
        );
        // Total termination must not be a fixed 1.46·r on a varying profile.
        let medium = medium();
        let config = config();
        let total = |profile: &SplinePortProfile| {
            let point =
                solve_duct_1d(profile, medium, 92.6, &config).expect("duct solve must succeed");
            point.interior_end_correction_m + point.exterior_end_correction_m
        };
        let straight_total = total(&straight());
        let tapered_total = total(&tapered);
        assert!(
            tapered_total < straight_total * (1.0 - 1.0e-6),
            "total end correction must vary with profile: {tapered_total} vs {straight_total}"
        );
    }

    #[test]
    fn radiation_termination_is_not_pressure_release() {
        let point =
            solve_duct_1d(&straight(), medium(), 92.6, &config()).expect("duct solve must succeed");
        let radiation = point.radiation_impedance_pa_s_per_m3;
        assert!(
            radiation.re > 0.0,
            "radiation resistance must be positive, got {}",
            radiation.re
        );
        assert!(
            radiation.im > 0.0,
            "baffled-piston reactance must be mass-like, got {}",
            radiation.im
        );
        // Pressure release would pin the mouth pressure to zero.
        assert!(
            point.outlet_pressure_rms_pa.norm() > 1.0e-6 * point.inlet_pressure_rms_pa.norm(),
            "mouth pressure must be nonzero under radiation loading"
        );
    }

    #[test]
    fn box_compliance_transfer_has_correct_limits() {
        let medium = medium();
        let config = config();
        let profile = straight();
        // |H - 1| ≈ (f/f0)², so 1 Hz sits ~1e-4 off unity.
        let low = solve_duct_1d(&profile, medium, 1.0, &config).expect("duct solve must succeed");
        assert!(
            (low.box_to_port_flow_ratio - Complex64::new(1.0, 0.0)).norm() < 1.0e-3,
            "far below tuning all diaphragm flow must reach the port"
        );
        let f0 = tuning_frequency_hz(&profile, medium, &config).expect("tuning must solve");
        let resonant =
            solve_duct_1d(&profile, medium, f0, &config).expect("duct solve must succeed");
        assert!(
            resonant.box_to_port_flow_ratio.norm() > 10.0,
            "resonant amplification must exceed 10×, got {}",
            resonant.box_to_port_flow_ratio.norm()
        );
    }

    #[test]
    fn segment_refinement_converges() {
        let medium = medium();
        let profile = dipped();
        let impedance = |n: usize| {
            let config = Duct1dConfig {
                n_segments: n,
                ..config()
            };
            solve_duct_1d(&profile, medium, 150.0, &config)
                .expect("duct solve must succeed")
                .port_impedance_pa_s_per_m3
        };
        let fine = impedance(2048);
        let coarse = impedance(64);
        let medium_grid = impedance(256);
        let coarse_error = (coarse - fine).norm() / fine.norm();
        let medium_error = (medium_grid - fine).norm() / fine.norm();
        assert!(
            coarse_error < 1.0e-3,
            "coarse grid must already be close: {coarse_error}"
        );
        let rate = coarse_error / medium_error;
        assert!(
            (4.0..64.0).contains(&rate),
            "refinement must converge ~2nd order (4× finer ⇒ ~16× smaller): rate {rate}"
        );
    }

    #[test]
    fn distributed_resonance_stays_near_compact_tuning() {
        // Internal consistency: the transfer-matrix Im(Y) = 0 zero (which keeps
        // neck compliance) sits within ~1% of the compact tuning screen.
        let medium = medium();
        let config = config();
        let profile = straight();
        let f0 = tuning_frequency_hz(&profile, medium, &config).expect("tuning must solve");
        let compliance = box_compliance_m5_per_n(VBOX, medium).expect("compliance valid");
        let imag_admittance = |frequency: f64| {
            let point = solve_duct_1d(&profile, medium, frequency, &config)
                .expect("duct solve must succeed");
            (Complex64::new(0.0, TAU * frequency * compliance)
                + Complex64::new(1.0, 0.0) / point.port_impedance_pa_s_per_m3)
                .im
        };
        let mut low = 0.5 * f0;
        let mut high = 1.5 * f0;
        assert!(
            imag_admittance(low) < 0.0 && imag_admittance(high) > 0.0,
            "distributed resonance must bracket the compact tuning"
        );
        for _ in 0..60 {
            let mid = 0.5 * (low + high);
            if imag_admittance(mid) > 0.0 {
                high = mid;
            } else {
                low = mid;
            }
        }
        let distributed = 0.5 * (low + high);
        let gap = (distributed - f0) / f0;
        assert!(
            gap.abs() < 0.02,
            "distributed/compact tuning gap must be O((kL)²): {gap}"
        );
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        let medium = medium();
        let profile = straight();
        let config = config();
        for bad_frequency in [0.0, -3.0, f64::NAN, f64::INFINITY] {
            assert!(
                solve_duct_1d(&profile, medium, bad_frequency, &config).is_err(),
                "frequency {bad_frequency} must be rejected"
            );
        }
        let bad_segments = Duct1dConfig {
            n_segments: 0,
            ..config
        };
        assert!(solve_duct_1d(&profile, medium, 92.6, &bad_segments).is_err());
        assert!(tuning_frequency_hz(&profile, medium, &bad_segments).is_err());
        for bad_volume in [0.0, -1.0, f64::NAN] {
            let bad_box = Duct1dConfig {
                box_volume_m3: bad_volume,
                ..config
            };
            assert!(
                solve_duct_1d(&profile, medium, 92.6, &bad_box).is_err(),
                "box volume {bad_volume} must be rejected"
            );
        }
        let bad_medium = AcousticMedium {
            density_kg_per_m3: 0.0,
            ..medium
        };
        assert!(solve_duct_1d(&profile, bad_medium, 92.6, &config).is_err());
        assert!(cutoff_frequency_hz(medium, 0.0).is_err());
        assert!(duct_inertance_kg_per_m4(&profile, 1.2, 0).is_err());
    }
}
