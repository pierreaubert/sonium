//! Frozen Step-0 reference case and baseline archive.
//!
//! [`PortOptimReferenceCase`] freezes the enclosure, driver, straight-tube
//! port, drive levels, medium, mesh sizes, and timestep of the port-shape
//! optimisation campaign (PORT-OPTIM.md Step 0). All magnitudes reuse the
//! `qa-speaker-port-nonlinear` fixtures: Tier 2 replays the verified 2D slot
//! leg bit-identically (same geometry, mesh, dt, cycles, drives, and drive
//! rule of half the surrogate tuning), while Tier 1 runs the round-tube
//! linear reference on the straight tube at `[0.75, 1.0, 1.25] x f_target`.
//! [`PortOptimBaseline`] archives the Tier 1/2 observables plus the low-drive
//! gate evidence; `fixtures/port_optim_baseline_v1.json` stores the frozen
//! run. [`check_baseline_gates`] re-verifies an archive against the frozen
//! fixture and the existing 2D low-drive gates (lumped limit, THD, energy).
//!
//! Validated envelope: Tier 2 numbers are 2D slot evidence at the converged
//! 252-samples-per-cycle timestep. Tier 1 is the bounded linear reference,
//! not a nonlinear 3D claim. A same-geometry Tier 1 to Tier 2 linear gate
//! arrives with the spline-parameterized reference (Step 4/5), not here.

// Rust guideline compliant 2026-02-21

use std::f64::consts::{LN_2, PI, TAU};

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use sonium_fem::physics::acoustics::pressure::FluidProperties;
use sonium_fem::physics::coupled::lumped_circuit::VoiceCoilParams;
use sonium_fem::physics::fluid::stokes::navier_stokes::NavierStokesConfig;
use sonium_fem::physics::fluid::stokes::time_stepping::BdfConfig;

use super::box_driver::BoxDriverModel;
use super::coupled::CoupledPortConfig;
use super::error::PortFlowError;
use super::geometry::SlotPortGeometry;
use super::geometry3d::RoundPortGeometry;
use super::linear_reference::RoundPortLinearReferenceConfig;
use crate::cabinet_design::PORT_END_CORRECTION_RADIUS_FACTOR;
use crate::{AcousticMedium, EnclosureSpec, PortSpec};

/// Archive schema version for [`PortOptimReferenceCase`] and [`PortOptimBaseline`].
///
/// Bump when a field is added, removed, or redefined; the gate checker
/// rejects any archive whose version differs so stale baselines fail loudly.
pub const PORT_OPTIM_BASELINE_SCHEMA_VERSION: u32 = 1;

/// Provenance recorded in every baseline archive.
pub const PORT_OPTIM_BASELINE_PROVENANCE: &str =
    "PortOptimStep0StraightTube[v1,SlotTier2,RoundLinearTier1]";

/// Maximum low-drive lumped-limit relative error.
///
/// Matches the `qa-speaker-port-nonlinear` 2D leg
/// (`LUMPED_LIMIT_MAX_RELATIVE_ERROR`). Tightening this below the measured
/// ~9.9% Tier 2 error fails the gate.
pub const BASELINE_MAX_LUMPED_LIMIT_RELATIVE_ERROR: f64 = 0.10;

/// Maximum low-drive THD fraction.
///
/// Matches the `qa-speaker-port-nonlinear` 2D leg
/// (`LINEAR_THD_MAX_FRACTION`).
pub const BASELINE_MAX_LINEAR_THD_FRACTION: f64 = 0.01;

/// Maximum physical energy closure.
///
/// Matches the `qa-speaker-port-nonlinear` 2D leg
/// (`COUPLED_ENERGY_PHYSICAL_CLOSURE_MAX`).
pub const BASELINE_MAX_ENERGY_PHYSICAL_CLOSURE: f64 = 0.04;

/// Maximum algebraic energy closure.
///
/// Matches the `qa-speaker-port-nonlinear` 2D leg tight roundoff-scale bound.
pub const BASELINE_MAX_ENERGY_ALGEBRAIC_CLOSURE: f64 = 1.0e-8;

/// Maximum interface energy residual.
///
/// Matches the `qa-speaker-port-nonlinear` 2D leg tight roundoff-scale bound.
pub const BASELINE_MAX_ENERGY_INTERFACE_RESIDUAL: f64 = 1.0e-8;

/// Maximum Tier 1 true relative residual.
///
/// Matches the linear-reference solver's own acceptance bound.
pub const BASELINE_MAX_TIER1_RELATIVE_RESIDUAL: f64 = 1.0e-8;

/// Maximum Tier 1 strong/weak flux relative difference.
///
/// The straight sharp-shoulder tube measures ~2.5e-3 (the re-entrant corner
/// singularity slows strong/weak gradient agreement); the existing
/// straight-tube reference test only requires this audit to be finite, and
/// the 1e-6 bound belongs to the flared-geometry coupled gate, not to the
/// straight-tube oracle. Step 0 records the value and gates it here with
/// ~4x margin; profile refinement in later steps must not regress it.
pub const BASELINE_MAX_TIER1_FLUX_RELATIVE_DIFFERENCE: f64 = 1.0e-2;

/// Frozen Step-0 reference configuration (schema v1).
///
/// Every field is plain data so the fixture serializes to JSON; builders
/// below convert it into solver inputs. `Default` is the frozen campaign
/// reference; any deviation is fixture drift and fails [`check_baseline_gates`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PortOptimReferenceCase {
    /// Must equal [`PORT_OPTIM_BASELINE_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Ambient medium density (kg/m3).
    pub density_kg_per_m3: f64,
    /// Ambient medium sound speed (m/s).
    pub sound_speed_m_per_s: f64,
    /// Ambient medium dynamic viscosity (Pa s).
    pub dynamic_viscosity_pa_s: f64,
    /// Ambient medium thermal diffusivity (m2/s).
    pub thermal_diffusivity_m2_per_s: f64,
    /// Enclosure volume (m3).
    pub box_volume_m3: f64,
    /// Driver moving mass Mms (kg).
    pub driver_mms_kg: f64,
    /// Driver mechanical resistance Rms (kg/s).
    pub driver_rms_kg_per_s: f64,
    /// Driver mechanical stiffness Kms (N/m).
    pub driver_kms_n_per_m: f64,
    /// Voice-coil DC resistance (ohm).
    pub voice_coil_resistance_ohm: f64,
    /// Voice-coil inductance (H).
    pub voice_coil_inductance_h: f64,
    /// Voice-coil force factor Bl (N/A).
    pub voice_coil_bl_n_per_a: f64,
    /// Effective diaphragm area Sd (m2).
    pub diaphragm_area_m2: f64,
    /// Diaphragm excursion limit Xmax (m).
    pub xmax_m: f64,
    /// Straight-tube port radius R0 (m).
    pub port_radius_r0_m: f64,
    /// Straight-tube port length L (m).
    pub port_length_l_m: f64,
    /// Lumped end correction as a multiple of R0 (both terminations).
    pub end_correction_radius_factor: f64,
    /// Tier 1 round-chamber axial length (m).
    pub tier1_chamber_length_m: f64,
    /// Tier 1 round-chamber radius (m).
    pub tier1_chamber_radius_m: f64,
    /// Tier 1 mesh size (m).
    pub tier1_mesh_size_m: f64,
    /// Tier 1 frequencies as multiples of `f_target`.
    pub tier1_frequency_factors: [f64; 3],
    /// Tier 2 slot duct length (m).
    pub tier2_duct_length_m: f64,
    /// Tier 2 slot duct height (m, area per unit depth).
    pub tier2_duct_height_m: f64,
    /// Tier 2 slot chamber length (m).
    pub tier2_chamber_length_m: f64,
    /// Tier 2 slot chamber height (m).
    pub tier2_chamber_height_m: f64,
    /// Tier 2 mesh size (m).
    pub tier2_mesh_size_m: f64,
    /// Tier 2 BDF samples per drive cycle.
    pub tier2_samples_per_cycle: usize,
    /// Tier 2 total drive cycles per level.
    pub tier2_total_cycles: usize,
    /// Tier 2 settled tail cycles analyzed per level.
    pub tier2_tail_cycles: usize,
    /// Tier 2 drive levels (V peak).
    pub tier2_drives_v_peak: [f64; 3],
}

impl Default for PortOptimReferenceCase {
    fn default() -> Self {
        Self {
            schema_version: PORT_OPTIM_BASELINE_SCHEMA_VERSION,
            // QA 2D leg medium (RHO, SOUND_SPEED, MU, alpha).
            density_kg_per_m3: 1.2,
            sound_speed_m_per_s: 343.0,
            dynamic_viscosity_pa_s: 1.85e-5,
            thermal_diffusivity_m2_per_s: 2.2e-5,
            // QA 2D leg enclosure and driver (BOX_VOLUME_M3, fixture_driver).
            box_volume_m3: 0.005,
            driver_mms_kg: 0.02,
            driver_rms_kg_per_s: 1.0,
            driver_kms_n_per_m: 4000.0,
            voice_coil_resistance_ohm: 6.0,
            voice_coil_inductance_h: 0.5e-3,
            voice_coil_bl_n_per_a: 10.0,
            diaphragm_area_m2: 0.01,
            xmax_m: 0.005,
            // Straight reference tube: 3D-leg radius, 2D-leg duct length.
            port_radius_r0_m: 0.025,
            port_length_l_m: 0.1,
            end_correction_radius_factor: PORT_END_CORRECTION_RADIUS_FACTOR,
            // Tier 1: QA exact-leg chambers at the fine mesh, straight (flare 0).
            tier1_chamber_length_m: 0.05,
            tier1_chamber_radius_m: 0.08,
            tier1_mesh_size_m: 0.01,
            tier1_frequency_factors: [0.75, 1.0, 1.25],
            // Tier 2: exact QA 2D-leg slot mesh and verified timestep.
            tier2_duct_length_m: 0.1,
            tier2_duct_height_m: 0.02,
            tier2_chamber_length_m: 0.05,
            tier2_chamber_height_m: 0.08,
            tier2_mesh_size_m: 0.01,
            tier2_samples_per_cycle: 252,
            tier2_total_cycles: 40,
            tier2_tail_cycles: 8,
            tier2_drives_v_peak: [0.01, 0.1, 1.0],
        }
    }
}

impl PortOptimReferenceCase {
    /// Validate every fixture field for positivity and finiteness.
    ///
    /// # Errors
    ///
    /// Returns [`PortFlowError::InvalidConfig`] for a version mismatch or a
    /// non-finite/non-positive field.
    pub fn validate(&self) -> Result<(), PortFlowError> {
        if self.schema_version != PORT_OPTIM_BASELINE_SCHEMA_VERSION {
            return Err(PortFlowError::InvalidConfig(format!(
                "unsupported port-optim baseline schema {}, expected {PORT_OPTIM_BASELINE_SCHEMA_VERSION}",
                self.schema_version
            )));
        }
        for (name, value) in [
            ("density_kg_per_m3", self.density_kg_per_m3),
            ("sound_speed_m_per_s", self.sound_speed_m_per_s),
            ("dynamic_viscosity_pa_s", self.dynamic_viscosity_pa_s),
            (
                "thermal_diffusivity_m2_per_s",
                self.thermal_diffusivity_m2_per_s,
            ),
            ("box_volume_m3", self.box_volume_m3),
            ("driver_mms_kg", self.driver_mms_kg),
            ("driver_rms_kg_per_s", self.driver_rms_kg_per_s),
            ("driver_kms_n_per_m", self.driver_kms_n_per_m),
            ("voice_coil_resistance_ohm", self.voice_coil_resistance_ohm),
            ("voice_coil_inductance_h", self.voice_coil_inductance_h),
            ("voice_coil_bl_n_per_a", self.voice_coil_bl_n_per_a),
            ("diaphragm_area_m2", self.diaphragm_area_m2),
            ("xmax_m", self.xmax_m),
            ("port_radius_r0_m", self.port_radius_r0_m),
            ("port_length_l_m", self.port_length_l_m),
            (
                "end_correction_radius_factor",
                self.end_correction_radius_factor,
            ),
            ("tier1_chamber_length_m", self.tier1_chamber_length_m),
            ("tier1_chamber_radius_m", self.tier1_chamber_radius_m),
            ("tier1_mesh_size_m", self.tier1_mesh_size_m),
            ("tier2_duct_length_m", self.tier2_duct_length_m),
            ("tier2_duct_height_m", self.tier2_duct_height_m),
            ("tier2_chamber_length_m", self.tier2_chamber_length_m),
            ("tier2_chamber_height_m", self.tier2_chamber_height_m),
            ("tier2_mesh_size_m", self.tier2_mesh_size_m),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(PortFlowError::InvalidConfig(format!(
                    "port-optim reference field {name} must be finite and positive, got {value}"
                )));
            }
        }
        for (index, factor) in self.tier1_frequency_factors.iter().enumerate() {
            if !(factor.is_finite() && *factor > 0.0) {
                return Err(PortFlowError::InvalidConfig(format!(
                    "tier1_frequency_factors[{index}] must be finite and positive, got {factor}"
                )));
            }
        }
        for (index, drive) in self.tier2_drives_v_peak.iter().enumerate() {
            if !(drive.is_finite() && *drive > 0.0) {
                return Err(PortFlowError::InvalidConfig(format!(
                    "tier2_drives_v_peak[{index}] must be finite and positive, got {drive}"
                )));
            }
        }
        if self.tier2_samples_per_cycle < 4
            || self.tier2_tail_cycles == 0
            || self.tier2_tail_cycles >= self.tier2_total_cycles
        {
            return Err(PortFlowError::InvalidConfig(format!(
                "invalid Tier 2 sampling: {} samples/cycle, {} tail of {} total cycles",
                self.tier2_samples_per_cycle, self.tier2_tail_cycles, self.tier2_total_cycles
            )));
        }
        Ok(())
    }

    /// Ambient medium shared by every tier.
    pub fn medium(&self) -> AcousticMedium {
        AcousticMedium {
            density_kg_per_m3: self.density_kg_per_m3,
            sound_speed_m_per_s: self.sound_speed_m_per_s,
            dynamic_viscosity_pa_s: self.dynamic_viscosity_pa_s,
            thermal_diffusivity_m2_per_s: self.thermal_diffusivity_m2_per_s,
        }
    }

    /// Lumped straight-tube port (R0, L, termination-aware end correction).
    pub fn port_spec(&self) -> PortSpec {
        PortSpec {
            radius_m: self.port_radius_r0_m,
            length_m: self.port_length_l_m,
            end_correction_m: self.end_correction_radius_factor * self.port_radius_r0_m,
            resistance_pa_s_per_m3: 0.0,
        }
    }

    /// Vented-box enclosure holding the straight tube.
    pub fn enclosure_spec(&self) -> EnclosureSpec {
        EnclosureSpec::Vented {
            volume_m3: self.box_volume_m3,
            port: self.port_spec(),
            leakage_resistance_pa_s_per_m3: 0.0,
            absorption_resistance_pa_s_per_m3: 0.0,
        }
    }

    /// Lumped box/driver model (linear suspension, linear voice coil).
    pub fn driver(&self) -> BoxDriverModel {
        BoxDriverModel {
            mms_kg: self.driver_mms_kg,
            rms_kg_per_s: self.driver_rms_kg_per_s,
            kms_n_per_m: self.driver_kms_n_per_m,
            voice_coil: VoiceCoilParams::linear(
                self.voice_coil_resistance_ohm,
                self.voice_coil_inductance_h,
                self.voice_coil_bl_n_per_a,
            ),
            compliance_force_fn: None,
            compliance_potential_fn: None,
            sd_m2: self.diaphragm_area_m2,
            box_volume_m3: Some(self.box_volume_m3),
            medium_density_kg_per_m3: self.density_kg_per_m3,
            medium_sound_speed_m_per_s: self.sound_speed_m_per_s,
            beta: 0.25,
            gamma: 0.5,
        }
    }

    /// Campaign tuning target: Helmholtz tuning of the straight tube.
    ///
    /// # Errors
    ///
    /// Returns [`PortFlowError::InvalidConfig`] when the fixture is invalid
    /// or the vented enclosure yields no tuning frequency.
    pub fn f_target_hz(&self) -> Result<f64, PortFlowError> {
        self.validate()?;
        self.enclosure_spec()
            .tuning_frequency_hz(self.medium())
            .map_err(|error| PortFlowError::InvalidConfig(error.to_string()))?
            .ok_or_else(|| {
                PortFlowError::InvalidConfig(
                    "vented reference enclosure produced no tuning frequency".to_string(),
                )
            })
    }

    /// Tier 2 drive frequency: half the slot surrogate's own tuning.
    ///
    /// This mirrors the `qa-speaker-port-nonlinear` 2D drive rule exactly,
    /// so Tier 2 replays the verified fixture bit-identically. It differs
    /// from [`PortOptimReferenceCase::f_target_hz`], the round-tube campaign
    /// tuning, because the slot surrogate has its own lumped area.
    ///
    /// # Errors
    ///
    /// Returns [`PortFlowError::InvalidConfig`] when the fixture is invalid
    /// or the slot surrogate yields no tuning frequency.
    pub fn tier2_drive_frequency_hz(&self) -> Result<f64, PortFlowError> {
        self.validate()?;
        let enclosure = EnclosureSpec::Vented {
            volume_m3: self.box_volume_m3,
            port: self.tier2_lumped_port(),
            leakage_resistance_pa_s_per_m3: 0.0,
            absorption_resistance_pa_s_per_m3: 0.0,
        };
        let tuning = enclosure
            .tuning_frequency_hz(self.medium())
            .map_err(|error| PortFlowError::InvalidConfig(error.to_string()))?
            .ok_or_else(|| {
                PortFlowError::InvalidConfig(
                    "Tier 2 slot surrogate produced no tuning frequency".to_string(),
                )
            })?;
        Ok(tuning / 2.0)
    }

    /// Tier 1 analysis frequencies in Hz.
    ///
    /// # Errors
    ///
    /// Returns [`PortFlowError::InvalidConfig`] when the fixture is invalid.
    pub fn tier1_frequencies_hz(&self) -> Result<[f64; 3], PortFlowError> {
        let target = self.f_target_hz()?;
        Ok(self.tier1_frequency_factors.map(|factor| factor * target))
    }

    /// Tier 2 slot surrogate geometry (deterministic hash-jittered mesh).
    pub fn slot_geometry(&self) -> SlotPortGeometry {
        SlotPortGeometry {
            duct_length_m: self.tier2_duct_length_m,
            duct_height_m: self.tier2_duct_height_m,
            chamber_length_m: self.tier2_chamber_length_m,
            chamber_height_m: self.tier2_chamber_height_m,
            mesh_size_m: self.tier2_mesh_size_m,
        }
    }

    /// Tier 1 straight round-tube geometry (zero flare).
    pub fn round_geometry(&self) -> RoundPortGeometry {
        RoundPortGeometry {
            duct_length_m: self.port_length_l_m,
            duct_radius_m: self.port_radius_r0_m,
            chamber_length_m: self.tier1_chamber_length_m,
            chamber_radius_m: self.tier1_chamber_radius_m,
            flare_radius_m: 0.0,
            mesh_size_m: self.tier1_mesh_size_m,
        }
    }

    /// Tier 1 thermophysical fluid properties from the fixture medium.
    pub fn linear_reference_fluid(&self) -> FluidProperties {
        FluidProperties {
            density: self.density_kg_per_m3,
            speed_of_sound: self.sound_speed_m_per_s,
            dynamic_viscosity: self.dynamic_viscosity_pa_s,
            ..FluidProperties::air_20c()
        }
    }

    /// Tier 1 linear-reference configuration for the straight tube.
    pub fn linear_reference_config(&self) -> RoundPortLinearReferenceConfig {
        RoundPortLinearReferenceConfig {
            geometry: self.round_geometry(),
            fluid: self.linear_reference_fluid(),
            box_volume_m3: self.box_volume_m3,
        }
    }

    /// Tier 2 timestep: one drive period over the sample count.
    ///
    /// # Errors
    ///
    /// Returns [`PortFlowError::InvalidConfig`] when the fixture is invalid.
    pub fn dt_s(&self) -> Result<f64, PortFlowError> {
        Ok(1.0 / (self.tier2_drive_frequency_hz()? * self.tier2_samples_per_cycle as f64))
    }

    /// Tier 2 BDF steps per drive level.
    pub fn num_steps(&self) -> usize {
        self.tier2_total_cycles * self.tier2_samples_per_cycle
    }

    /// Tier 2 coupled configuration at one drive level (BDF2, no stabilisation).
    ///
    /// # Errors
    ///
    /// Returns [`PortFlowError::InvalidConfig`] for a non-positive drive
    /// voltage or an invalid fixture.
    pub fn coupled_config(&self, voltage_peak_v: f64) -> Result<CoupledPortConfig, PortFlowError> {
        if !(voltage_peak_v.is_finite() && voltage_peak_v > 0.0) {
            return Err(PortFlowError::InvalidConfig(format!(
                "Tier 2 drive voltage must be finite and positive, got {voltage_peak_v}"
            )));
        }
        let frequency_hz = self.tier2_drive_frequency_hz()?;
        let dt = 1.0 / (frequency_hz * self.tier2_samples_per_cycle as f64);
        Ok(CoupledPortConfig {
            driver: self.driver(),
            ns_config: NavierStokesConfig {
                mu: self.dynamic_viscosity_pa_s,
                rho: self.density_kg_per_m3,
                use_residual_stabilisation: false,
                // Zero body force through `solve_bdf_step_cached`: the
                // force quadrature is identical every step, so caching
                // skips two full assemblies per step after the first.
                cache_body_force_assembly: true,
                ..Default::default()
            },
            bdf: BdfConfig::bdf2(dt, self.num_steps()),
            voltage_fn: Box::new(move |time| voltage_peak_v * (TAU * frequency_hz * time).sin()),
            under_relaxation: 1.0,
            exterior_pressure_pa: 0.0,
        })
    }

    /// Tier 2 slot duct area per unit out-of-plane depth (m2).
    pub fn tier2_slot_area_m2(&self) -> f64 {
        self.tier2_duct_height_m
    }

    /// Lumped port matching the Tier 2 slot surrogate.
    ///
    /// The round-port radius is area-equivalent to the slot
    /// (`S = h x 1 m`), and the end correction is the 2D slot value
    /// `4 ln 2 / pi x h` used by the QA 2D fixture.
    pub fn tier2_lumped_port(&self) -> PortSpec {
        PortSpec {
            radius_m: (self.tier2_slot_area_m2() / PI).sqrt(),
            length_m: self.tier2_duct_length_m,
            end_correction_m: 4.0 * LN_2 / PI * self.tier2_duct_height_m,
            resistance_pa_s_per_m3: 0.0,
        }
    }

    /// Lumped small-signal port-flow amplitude at one Tier 2 drive level.
    ///
    /// Couples the voice-coil electrical impedance, the driver mechanical
    /// impedance loaded by the slot-surrogate enclosure, and the port/end
    /// impedance divider. This is the independent low-drive oracle for the
    /// Tier 2 lumped-limit gate.
    ///
    /// # Errors
    ///
    /// Returns [`PortFlowError::InvalidConfig`] for a non-positive drive
    /// voltage, an invalid fixture, or a lumped impedance failure.
    pub fn lumped_port_flow_amplitude(&self, voltage_peak_v: f64) -> Result<f64, PortFlowError> {
        if !(voltage_peak_v.is_finite() && voltage_peak_v > 0.0) {
            return Err(PortFlowError::InvalidConfig(format!(
                "lumped reference drive voltage must be finite and positive, got {voltage_peak_v}"
            )));
        }
        self.validate()?;
        let medium = self.medium();
        let frequency_hz = self.tier2_drive_frequency_hz()?;
        let enclosure = EnclosureSpec::Vented {
            volume_m3: self.box_volume_m3,
            port: self.tier2_lumped_port(),
            leakage_resistance_pa_s_per_m3: 0.0,
            absorption_resistance_pa_s_per_m3: 0.0,
        };
        let enclosure_impedance = enclosure
            .acoustic_impedance(frequency_hz, medium)
            .map_err(|error| PortFlowError::InvalidConfig(error.to_string()))?;
        let port_impedance = self
            .tier2_lumped_port()
            .acoustic_impedance(frequency_hz, medium)
            .map_err(|error| PortFlowError::InvalidConfig(error.to_string()))?;
        let omega = TAU * frequency_hz;
        let electrical_impedance = Complex64::new(
            self.voice_coil_resistance_ohm,
            omega * self.voice_coil_inductance_h,
        );
        let mechanical_impedance = Complex64::new(
            self.driver_rms_kg_per_s,
            omega * self.driver_mms_kg - self.driver_kms_n_per_m / omega,
        );
        let total_mechanical_impedance =
            mechanical_impedance + enclosure_impedance * self.diaphragm_area_m2.powi(2);
        let diaphragm_velocity = self.voice_coil_bl_n_per_a * voltage_peak_v
            / (electrical_impedance * total_mechanical_impedance
                + self.voice_coil_bl_n_per_a.powi(2));
        Ok(
            (diaphragm_velocity * self.diaphragm_area_m2 * enclosure_impedance / port_impedance)
                .norm(),
        )
    }
}

/// Pack a phasor as `[re, im]` for JSON archives.
pub fn complex_pair(value: Complex64) -> [f64; 2] {
    [value.re, value.im]
}

/// Duct bulk-velocity Mach number from a port-flow amplitude.
///
/// Uses the area-averaged velocity `Q/S` over the speed of sound; the
/// centerline Mach number is higher, so this is a conservative low-drive
/// indicator, not a peak-Mach claim.
pub fn duct_bulk_mach_number(
    port_flow_amplitude_m3s: f64,
    duct_area_m2: f64,
    sound_speed_m_per_s: f64,
) -> f64 {
    (port_flow_amplitude_m3s / duct_area_m2).abs() / sound_speed_m_per_s
}

/// Port-output compression of one level against a small-signal reference.
///
/// `20 log10((V/V0)/(Q/Q0))` in dB; zero when output tracks the drive.
pub fn compression_db(
    voltage_peak_v: f64,
    flow_amplitude_m3s: f64,
    reference_voltage_peak_v: f64,
    reference_flow_amplitude_m3s: f64,
) -> f64 {
    20.0 * ((voltage_peak_v / reference_voltage_peak_v)
        / (flow_amplitude_m3s / reference_flow_amplitude_m3s))
        .log10()
}

/// Tier 1 straight-tube impedance point at one frequency.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tier1BaselinePoint {
    /// Harmonic frequency (Hz).
    pub frequency_hz: f64,
    /// `(P_box - P_ext)/Q_port` RMS phasor `[re, im]` (Pa s/m3).
    pub port_impedance_pa_s_per_m3: [f64; 2],
    /// `Q_port/Q_diaphragm` transfer `[re, im]`.
    pub box_to_port_flow_ratio: [f64; 2],
    /// Port volume-velocity RMS phasor `[re, im]` (m3/s).
    pub port_volume_velocity_rms_m3_per_s: [f64; 2],
    /// True relative residual of the reference solve.
    pub relative_residual: f64,
    /// Strong/weak exterior-flux relative difference.
    pub flux_relative_difference: f64,
    /// Scalar pressure unknowns in the reference solve.
    pub scalar_dofs: usize,
}

/// Tier 2 straight-slot measurements at one drive level.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tier2BaselineLevel {
    /// Applied terminal voltage (V peak).
    pub voltage_peak_v: f64,
    /// Completed BDF steps (must equal the fixture step count).
    pub steps_completed: usize,
    /// Fundamental port-flow amplitude from the settled tail (m3/s).
    pub port_flow_fundamental_amplitude_m3s: f64,
    /// Time-domain port-flow RMS over the settled tail (m3/s).
    pub port_flow_rms_m3s: f64,
    /// Fundamental box-pressure amplitude from the settled tail (Pa).
    pub box_pressure_amplitude_pa: f64,
    /// Peak diaphragm excursion over the settled tail (m).
    pub peak_excursion_m: f64,
    /// Total harmonic distortion over 5 harmonics (fraction).
    pub thd_fraction: f64,
    /// Port-output compression against the lowest drive (dB).
    pub compression_db: f64,
    /// Duct bulk-velocity Mach number at the fundamental amplitude.
    pub duct_bulk_mach: f64,
    /// Free-turbulence `U^8/L^3` noise-proxy trend value.
    pub turbulence_power_trend: f64,
    /// Physical-only energy residual over the tail (relative).
    pub energy_physical_relative_closure: f64,
    /// Full-ledger energy residual over the tail (relative).
    pub energy_algebraic_relative_closure: f64,
    /// Cumulative interface energy mismatch over the tail (relative).
    pub energy_interface_relative_residual: f64,
    /// Energy normalization scale over the tail (J).
    pub energy_scale_j: f64,
}

/// Low-drive lumped-limit evidence at the lowest Tier 2 drive.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LumpedLimitBaseline {
    /// `|Q_coupled - Q_lumped|/|Q_lumped|` at the lowest drive.
    pub relative_error: f64,
    /// Coupled fundamental port-flow amplitude (m3/s).
    pub coupled_flow_m3s: f64,
    /// Lumped reference port-flow amplitude (m3/s).
    pub reference_flow_m3s: f64,
}

/// Frozen Step-0 straight-tube baseline archive (schema v1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PortOptimBaseline {
    /// Must equal [`PORT_OPTIM_BASELINE_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Model lineage; must equal [`PORT_OPTIM_BASELINE_PROVENANCE`].
    pub provenance: String,
    /// Reference configuration the baseline was measured with.
    pub case: PortOptimReferenceCase,
    /// Campaign tuning target used for the Tier 1 sweep (Hz).
    pub f_target_hz: f64,
    /// Tier 1 impedance points, ascending frequency.
    pub tier1: Vec<Tier1BaselinePoint>,
    /// Tier 2 drive levels, ascending voltage.
    pub tier2: Vec<Tier2BaselineLevel>,
    /// Low-drive lumped-limit evidence.
    pub lumped_limit: LumpedLimitBaseline,
    /// Worst compression across Tier 2 levels (dB).
    pub max_compression_db: f64,
}

/// Worst-case gate quantities of a checked baseline archive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaselineGateReport {
    /// Lumped-limit relative error at the lowest drive.
    pub lumped_limit_relative_error: f64,
    /// THD fraction at the lowest drive.
    pub linear_thd_fraction: f64,
    /// Worst physical energy closure across levels.
    pub max_energy_physical_closure: f64,
    /// Worst algebraic energy closure across levels.
    pub max_energy_algebraic_closure: f64,
    /// Worst interface energy residual across levels.
    pub max_energy_interface_residual: f64,
    /// Worst Tier 1 true relative residual.
    pub max_tier1_relative_residual: f64,
    /// Worst Tier 1 flux relative difference.
    pub max_tier1_flux_relative_difference: f64,
    /// Worst Tier 2 compression.
    pub max_compression_db: f64,
}

fn gate_error(message: String) -> PortFlowError {
    PortFlowError::InvalidConfig(format!("port-optim baseline gate failed: {message}"))
}

/// Check a baseline archive against the frozen fixture and gates.
///
/// Verifies the schema version, provenance, exact fixture match, Tier 1
/// audit bounds, Tier 2 step counts and energy bounds, compression
/// monotonicity, and the low-drive lumped-limit and THD gates.
///
/// # Errors
///
/// Returns [`PortFlowError::InvalidConfig`] naming the first violated check.
pub fn check_baseline_gates(
    baseline: &PortOptimBaseline,
) -> Result<BaselineGateReport, PortFlowError> {
    if baseline.schema_version != PORT_OPTIM_BASELINE_SCHEMA_VERSION {
        return Err(gate_error(format!(
            "schema version is {}, expected {PORT_OPTIM_BASELINE_SCHEMA_VERSION}",
            baseline.schema_version
        )));
    }
    if baseline.provenance != PORT_OPTIM_BASELINE_PROVENANCE {
        return Err(gate_error(format!(
            "provenance is {:?}, expected {PORT_OPTIM_BASELINE_PROVENANCE:?}",
            baseline.provenance
        )));
    }
    if baseline.case != PortOptimReferenceCase::default() {
        return Err(gate_error(
            "reference case differs from the frozen Step-0 fixture".to_string(),
        ));
    }
    let case = PortOptimReferenceCase::default();
    let expected_target = case.f_target_hz()?;
    if !baseline.f_target_hz.is_finite()
        || (baseline.f_target_hz - expected_target).abs() > 1.0e-12 * expected_target
    {
        return Err(gate_error(format!(
            "f_target_hz is {}, expected {expected_target}",
            baseline.f_target_hz
        )));
    }

    let expected_frequencies = case.tier1_frequencies_hz()?;
    if baseline.tier1.len() != expected_frequencies.len() {
        return Err(gate_error(format!(
            "Tier 1 has {} points, expected {}",
            baseline.tier1.len(),
            expected_frequencies.len()
        )));
    }
    let mut max_tier1_residual = 0.0_f64;
    let mut max_tier1_flux = 0.0_f64;
    for (point, expected_frequency) in baseline.tier1.iter().zip(expected_frequencies.iter()) {
        if !point.frequency_hz.is_finite()
            || (point.frequency_hz - expected_frequency).abs() > 1.0e-12 * expected_frequency
        {
            return Err(gate_error(format!(
                "Tier 1 frequency is {}, expected {expected_frequency}",
                point.frequency_hz
            )));
        }
        for (name, pair) in [
            (
                "port_impedance_pa_s_per_m3",
                point.port_impedance_pa_s_per_m3,
            ),
            ("box_to_port_flow_ratio", point.box_to_port_flow_ratio),
            (
                "port_volume_velocity_rms_m3_per_s",
                point.port_volume_velocity_rms_m3_per_s,
            ),
        ] {
            if !pair.iter().all(|value| value.is_finite()) {
                return Err(gate_error(format!(
                    "Tier 1 {name} at {} Hz is non-finite: {pair:?}",
                    point.frequency_hz
                )));
            }
        }
        let impedance_norm = Complex64::new(
            point.port_impedance_pa_s_per_m3[0],
            point.port_impedance_pa_s_per_m3[1],
        )
        .norm();
        if impedance_norm <= 0.0 {
            return Err(gate_error(format!(
                "Tier 1 impedance at {} Hz has non-positive norm",
                point.frequency_hz
            )));
        }
        if !point.relative_residual.is_finite()
            || point.relative_residual > BASELINE_MAX_TIER1_RELATIVE_RESIDUAL
        {
            return Err(gate_error(format!(
                "Tier 1 residual at {} Hz is {}, bound is {BASELINE_MAX_TIER1_RELATIVE_RESIDUAL:e}",
                point.frequency_hz, point.relative_residual
            )));
        }
        if !point.flux_relative_difference.is_finite()
            || point.flux_relative_difference > BASELINE_MAX_TIER1_FLUX_RELATIVE_DIFFERENCE
        {
            return Err(gate_error(format!(
                "Tier 1 flux difference at {} Hz is {}, bound is {BASELINE_MAX_TIER1_FLUX_RELATIVE_DIFFERENCE:e}",
                point.frequency_hz, point.flux_relative_difference
            )));
        }
        if point.scalar_dofs == 0 {
            return Err(gate_error(format!(
                "Tier 1 point at {} Hz records zero scalar DOFs",
                point.frequency_hz
            )));
        }
        max_tier1_residual = max_tier1_residual.max(point.relative_residual);
        max_tier1_flux = max_tier1_flux.max(point.flux_relative_difference);
    }

    if baseline.tier2.len() != case.tier2_drives_v_peak.len() {
        return Err(gate_error(format!(
            "Tier 2 has {} levels, expected {}",
            baseline.tier2.len(),
            case.tier2_drives_v_peak.len()
        )));
    }
    let expected_steps = case.num_steps();
    let mut max_physical = 0.0_f64;
    let mut max_algebraic = 0.0_f64;
    let mut max_interface = 0.0_f64;
    let mut max_compression = f64::NEG_INFINITY;
    let mut previous_compression: Option<f64> = None;
    for (level, expected_drive) in baseline.tier2.iter().zip(case.tier2_drives_v_peak.iter()) {
        if level.voltage_peak_v != *expected_drive {
            return Err(gate_error(format!(
                "Tier 2 drive is {} V peak, expected {expected_drive} V peak",
                level.voltage_peak_v
            )));
        }
        if level.steps_completed != expected_steps {
            return Err(gate_error(format!(
                "Tier 2 at {} V peak completed {} steps, expected {expected_steps}",
                level.voltage_peak_v, level.steps_completed
            )));
        }
        for (name, value) in [
            (
                "port_flow_fundamental_amplitude_m3s",
                level.port_flow_fundamental_amplitude_m3s,
            ),
            ("port_flow_rms_m3s", level.port_flow_rms_m3s),
            ("box_pressure_amplitude_pa", level.box_pressure_amplitude_pa),
            ("peak_excursion_m", level.peak_excursion_m),
            ("thd_fraction", level.thd_fraction),
            ("compression_db", level.compression_db),
            ("duct_bulk_mach", level.duct_bulk_mach),
            ("turbulence_power_trend", level.turbulence_power_trend),
            (
                "energy_physical_relative_closure",
                level.energy_physical_relative_closure,
            ),
            (
                "energy_algebraic_relative_closure",
                level.energy_algebraic_relative_closure,
            ),
            (
                "energy_interface_relative_residual",
                level.energy_interface_relative_residual,
            ),
            ("energy_scale_j", level.energy_scale_j),
        ] {
            if !value.is_finite() {
                return Err(gate_error(format!(
                    "Tier 2 {name} at {} V peak is non-finite: {value}",
                    level.voltage_peak_v
                )));
            }
        }
        if level.port_flow_fundamental_amplitude_m3s <= 0.0
            || level.port_flow_rms_m3s <= 0.0
            || level.energy_scale_j <= 0.0
            || level.duct_bulk_mach < 0.0
            || level.turbulence_power_trend < 0.0
            || level.thd_fraction < 0.0
        {
            return Err(gate_error(format!(
                "Tier 2 at {} V peak has an invalid-sign observable",
                level.voltage_peak_v
            )));
        }
        if level.energy_physical_relative_closure > BASELINE_MAX_ENERGY_PHYSICAL_CLOSURE
            || level.energy_algebraic_relative_closure > BASELINE_MAX_ENERGY_ALGEBRAIC_CLOSURE
            || level.energy_interface_relative_residual > BASELINE_MAX_ENERGY_INTERFACE_RESIDUAL
        {
            return Err(gate_error(format!(
                "Tier 2 energy closure at {} V peak exceeds bounds: physical={}, algebraic={}, interface={}",
                level.voltage_peak_v,
                level.energy_physical_relative_closure,
                level.energy_algebraic_relative_closure,
                level.energy_interface_relative_residual
            )));
        }
        if let Some(previous) = previous_compression
            && level.compression_db < previous
        {
            return Err(gate_error(format!(
                "Tier 2 compression decreases from {previous} dB to {} dB",
                level.compression_db
            )));
        }
        previous_compression = Some(level.compression_db);
        max_physical = max_physical.max(level.energy_physical_relative_closure);
        max_algebraic = max_algebraic.max(level.energy_algebraic_relative_closure);
        max_interface = max_interface.max(level.energy_interface_relative_residual);
        max_compression = max_compression.max(level.compression_db);
    }

    let linear_thd = baseline
        .tier2
        .first()
        .map(|level| level.thd_fraction)
        .unwrap_or(f64::NAN);
    if !linear_thd.is_finite() || linear_thd > BASELINE_MAX_LINEAR_THD_FRACTION {
        return Err(gate_error(format!(
            "low-drive THD is {linear_thd}, bound is {BASELINE_MAX_LINEAR_THD_FRACTION}"
        )));
    }
    if !baseline.lumped_limit.relative_error.is_finite()
        || baseline.lumped_limit.relative_error > BASELINE_MAX_LUMPED_LIMIT_RELATIVE_ERROR
        || !baseline.lumped_limit.coupled_flow_m3s.is_finite()
        || !baseline.lumped_limit.reference_flow_m3s.is_finite()
        || baseline.lumped_limit.reference_flow_m3s <= 0.0
    {
        return Err(gate_error(format!(
            "lumped-limit evidence is invalid: {:?}",
            baseline.lumped_limit
        )));
    }
    if !baseline.max_compression_db.is_finite()
        || (baseline.max_compression_db - max_compression).abs() > 1.0e-12
    {
        return Err(gate_error(format!(
            "max_compression_db is {}, Tier 2 worst is {max_compression}",
            baseline.max_compression_db
        )));
    }

    Ok(BaselineGateReport {
        lumped_limit_relative_error: baseline.lumped_limit.relative_error,
        linear_thd_fraction: linear_thd,
        max_energy_physical_closure: max_physical,
        max_energy_algebraic_closure: max_algebraic,
        max_energy_interface_residual: max_interface,
        max_tier1_relative_residual: max_tier1_residual,
        max_tier1_flux_relative_difference: max_tier1_flux,
        max_compression_db: baseline.max_compression_db,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_default_validates_and_round_trips() {
        let case = PortOptimReferenceCase::default();
        case.validate().unwrap();
        let json = serde_json::to_string(&case).unwrap();
        let decoded: PortOptimReferenceCase = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, case);
    }

    #[test]
    fn tuning_targets_match_closed_form_helmholtz() {
        let case = PortOptimReferenceCase::default();
        // Closed form f0 = c/(2 pi) sqrt(S/(V L_eff)) evaluated directly from
        // the lumped specs, independent of the tuning-frequency API wiring.
        let medium = case.medium();
        let area = PI * case.port_radius_r0_m.powi(2);
        let effective_length =
            case.port_length_l_m + case.end_correction_radius_factor * case.port_radius_r0_m;
        let expected_target = medium.sound_speed_m_per_s / TAU
            * (area / (case.box_volume_m3 * effective_length)).sqrt();
        let target = case.f_target_hz().unwrap();
        assert!(
            (target - expected_target).abs() < 1.0e-12 * expected_target,
            "f_target {target} Hz differs from closed form {expected_target}"
        );
        // Approximately 92.6 Hz for R0 = 25 mm, L = 0.1 m, V = 5 L. The band
        // below guards against gross fixture drift, not formula precision.
        assert!(
            (90.0..95.0).contains(&target),
            "f_target {target} Hz left the 90-95 Hz fixture band"
        );

        let slot_area = case.tier2_slot_area_m2();
        let slot_effective = case.tier2_duct_length_m + 4.0 * LN_2 / PI * case.tier2_duct_height_m;
        let expected_drive = medium.sound_speed_m_per_s / TAU
            * (slot_area / (case.box_volume_m3 * slot_effective)).sqrt()
            / 2.0;
        let drive = case.tier2_drive_frequency_hz().unwrap();
        assert!(
            (drive - expected_drive).abs() < 1.0e-12 * expected_drive,
            "Tier 2 drive {drive} Hz differs from closed form {expected_drive}"
        );
        // Approximately 159.2 Hz: the QA 2D fixture point (f0_slot / 2).
        assert!(
            (155.0..163.0).contains(&drive),
            "Tier 2 drive {drive} Hz left the 155-163 Hz fixture band"
        );
    }

    #[test]
    fn timestep_divides_drive_period() {
        let case = PortOptimReferenceCase::default();
        let frequency_hz = case.tier2_drive_frequency_hz().unwrap();
        let dt = case.dt_s().unwrap();
        let cycle_error = (dt * frequency_hz * case.tier2_samples_per_cycle as f64 - 1.0).abs();
        assert!(cycle_error < 1.0e-10, "cycle error {cycle_error:e}");
        assert_eq!(case.num_steps(), 40 * 252);
        let config = case.coupled_config(0.1).unwrap();
        assert_eq!(config.bdf.dt, dt);
        assert_eq!(config.bdf.num_steps, case.num_steps());
        assert_eq!(config.bdf.order, 2);
        assert!(!config.ns_config.use_residual_stabilisation);
    }

    #[test]
    fn lumped_reference_is_positive_and_scales_linearly() {
        let case = PortOptimReferenceCase::default();
        let low = case.lumped_port_flow_amplitude(0.01).unwrap();
        let high = case.lumped_port_flow_amplitude(0.1).unwrap();
        assert!(low > 0.0, "lumped reference must be positive, got {low}");
        // The lumped oracle is linear in drive; any deviation is a wiring bug.
        assert!(
            (high / low - 10.0).abs() < 1.0e-12,
            "lumped reference is not linear: {low:e} vs {high:e}"
        );
    }

    #[test]
    fn slot_end_correction_matches_qa_fixture() {
        let case = PortOptimReferenceCase::default();
        let port = case.tier2_lumped_port();
        let expected = 4.0 * LN_2 / PI * case.tier2_duct_height_m;
        assert_eq!(port.end_correction_m, expected);
        assert_eq!(port.length_m, case.tier2_duct_length_m);
    }

    #[test]
    fn invalid_fixture_fields_are_rejected() {
        let case = PortOptimReferenceCase {
            schema_version: 999,
            ..Default::default()
        };
        assert!(case.validate().is_err());
        let case = PortOptimReferenceCase {
            port_radius_r0_m: -0.025,
            ..Default::default()
        };
        assert!(case.validate().is_err());
        let mut case = PortOptimReferenceCase::default();
        case.tier2_drives_v_peak[0] = 0.0;
        assert!(case.validate().is_err());
        let mut case = PortOptimReferenceCase::default();
        case.tier2_tail_cycles = case.tier2_total_cycles;
        assert!(case.validate().is_err());
    }

    fn passing_baseline() -> PortOptimBaseline {
        let case = PortOptimReferenceCase::default();
        let target = case.f_target_hz().unwrap();
        let tier1 = case
            .tier1_frequencies_hz()
            .unwrap()
            .iter()
            .map(|frequency| Tier1BaselinePoint {
                frequency_hz: *frequency,
                port_impedance_pa_s_per_m3: [100.0, 500.0],
                box_to_port_flow_ratio: [0.9, -0.1],
                port_volume_velocity_rms_m3_per_s: [1.0e-4, 2.0e-5],
                relative_residual: 1.0e-10,
                flux_relative_difference: 1.0e-9,
                scalar_dofs: 1000,
            })
            .collect();
        let tier2 = case
            .tier2_drives_v_peak
            .iter()
            .enumerate()
            .map(|(index, drive)| Tier2BaselineLevel {
                voltage_peak_v: *drive,
                steps_completed: case.num_steps(),
                port_flow_fundamental_amplitude_m3s: 1.0e-4,
                port_flow_rms_m3s: 7.0e-5,
                box_pressure_amplitude_pa: 10.0,
                peak_excursion_m: 1.0e-4,
                thd_fraction: 0.001,
                compression_db: index as f64 * 0.1,
                duct_bulk_mach: 0.001,
                turbulence_power_trend: 1.0e-30,
                energy_physical_relative_closure: 0.01,
                energy_algebraic_relative_closure: 1.0e-12,
                energy_interface_relative_residual: 1.0e-12,
                energy_scale_j: 1.0e-6,
            })
            .collect();
        PortOptimBaseline {
            schema_version: PORT_OPTIM_BASELINE_SCHEMA_VERSION,
            provenance: PORT_OPTIM_BASELINE_PROVENANCE.to_string(),
            case,
            f_target_hz: target,
            tier1,
            tier2,
            lumped_limit: LumpedLimitBaseline {
                relative_error: 0.05,
                coupled_flow_m3s: 1.0e-4,
                reference_flow_m3s: 1.05e-4,
            },
            max_compression_db: 0.2,
        }
    }

    #[test]
    fn gate_checker_accepts_a_passing_archive() {
        let report = check_baseline_gates(&passing_baseline()).unwrap();
        assert_eq!(report.lumped_limit_relative_error, 0.05);
        assert_eq!(report.max_compression_db, 0.2);
    }

    #[test]
    fn gate_checker_rejects_schema_provenance_and_fixture_drift() {
        let mut baseline = passing_baseline();
        baseline.schema_version = 999;
        assert!(check_baseline_gates(&baseline).is_err());
        let mut baseline = passing_baseline();
        baseline.provenance = "forged".to_string();
        assert!(check_baseline_gates(&baseline).is_err());
        let mut baseline = passing_baseline();
        baseline.case.port_radius_r0_m = 0.03;
        assert!(check_baseline_gates(&baseline).is_err());
        let mut baseline = passing_baseline();
        baseline.f_target_hz *= 1.5;
        assert!(check_baseline_gates(&baseline).is_err());
    }

    #[test]
    fn gate_checker_rejects_tier_violations() {
        let mut baseline = passing_baseline();
        baseline.tier1.clear();
        assert!(check_baseline_gates(&baseline).is_err());
        let mut baseline = passing_baseline();
        baseline.tier1[0].relative_residual = 1.0e-6;
        assert!(check_baseline_gates(&baseline).is_err());
        let mut baseline = passing_baseline();
        baseline.tier2[0].thd_fraction = 0.02;
        assert!(check_baseline_gates(&baseline).is_err());
        let mut baseline = passing_baseline();
        baseline.tier2[2].compression_db = -1.0;
        assert!(check_baseline_gates(&baseline).is_err());
        let mut baseline = passing_baseline();
        baseline.lumped_limit.relative_error = 0.2;
        assert!(check_baseline_gates(&baseline).is_err());
        let mut baseline = passing_baseline();
        baseline.max_compression_db = 99.0;
        assert!(check_baseline_gates(&baseline).is_err());
    }
}
