//! Three-phase winding layout, excitation and terminal quantities.
//!
//! Conventions match the reference motor solver this module replicates:
//! phase currents are imposed in the rotor frame so that `gamma = 0` is
//! the q-axis once the per-topology d-axis offset `daxis_deg` has been
//! calibrated (see [`calibrate_daxis_deg`]), the coil current density in
//! a slot is `direction * I_coil * turns / A_copper_slot` so every slot
//! carries exactly `turns * I_coil` ampere-turns, and phase flux linkage
//! is the source-adjoint sum `L_stack / n_parallel * sum_slots
//! direction * turns * mean(A_z over the slot copper)`.

// Rust guideline compliant 2026-02-21

use super::az::AzQuadraturePoint;

/// One of the three phase channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Phase {
    /// Phase A (reference channel).
    A,
    /// Phase B (-120 electrical degrees).
    B,
    /// Phase C (+120 electrical degrees).
    C,
}

impl Phase {
    /// All phases in channel order.
    pub const ALL: [Phase; 3] = [Phase::A, Phase::B, Phase::C];

    /// Channel index into `[A, B, C]` arrays.
    pub fn index(self) -> usize {
        match self {
            Phase::A => 0,
            Phase::B => 1,
            Phase::C => 2,
        }
    }

    /// Electrical offset of the channel, in radians.
    pub fn offset_rad(self) -> f64 {
        match self {
            Phase::A => 0.0,
            Phase::B => -2.0 * std::f64::consts::PI / 3.0,
            Phase::C => 2.0 * std::f64::consts::PI / 3.0,
        }
    }
}

/// One slot's winding assignment: phase and current direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotWinding {
    /// Phase channel feeding the slot.
    pub phase: Phase,
    /// Current direction along `+z` for positive phase current (`+1`/`-1`).
    pub direction: i8,
}

/// A stranded three-phase winding on a slotted stator.
#[derive(Debug, Clone, PartialEq)]
pub struct Winding {
    /// Per-slot `(phase, direction)`, in slot order around the stator.
    pub layout: Vec<SlotWinding>,
    /// Series turns per slot side (wires per slot in the reference CAD).
    pub turns_per_slot: f64,
    /// Parallel paths per phase; divides coil current and flux linkage.
    pub n_parallel: f64,
    /// Pole pairs of the machine.
    pub pole_pairs: usize,
}

impl Winding {
    /// Validates the layout and counts.
    ///
    /// # Errors
    ///
    /// Returns [`WindingError`] for an empty layout, a direction other
    /// than `+1`/`-1`, or non-positive turns/parallel/pole-pair counts.
    pub fn validate(&self) -> Result<(), WindingError> {
        if self.layout.is_empty() {
            return Err(WindingError::EmptyLayout);
        }
        if self.layout.iter().any(|slot| slot.direction.abs() != 1) {
            return Err(WindingError::InvalidDirection);
        }
        if !self.turns_per_slot.is_finite() || self.turns_per_slot <= 0.0 {
            return Err(WindingError::InvalidTurns);
        }
        if !self.n_parallel.is_finite() || self.n_parallel <= 0.0 {
            return Err(WindingError::InvalidParallelPaths);
        }
        if self.pole_pairs == 0 {
            return Err(WindingError::InvalidPolePairs);
        }
        Ok(())
    }

    /// Balanced sinusoidal phase currents (peak amperes) at a rotor angle.
    ///
    /// `i_peak_branch_a` is the peak current in one parallel branch
    /// (the reference solver passes `I_rms * sqrt(2) / n_parallel`).
    /// The electrical angle is
    /// `theta_mech * pole_pairs + gamma_deg + daxis_deg`.
    pub fn phase_currents_a(
        &self,
        rotor_angle_deg: f64,
        gamma_deg: f64,
        daxis_deg: f64,
        i_peak_branch_a: f64,
    ) -> [f64; 3] {
        let theta_e =
            (rotor_angle_deg * self.pole_pairs as f64 + gamma_deg + daxis_deg).to_radians();
        Phase::ALL.map(|phase| i_peak_branch_a * (theta_e + phase.offset_rad()).cos())
    }

    /// Balanced sinusoidal phase voltages (peak volts) at a rotor angle.
    pub fn phase_voltages_v(
        &self,
        rotor_angle_deg: f64,
        v_delta_deg: f64,
        daxis_deg: f64,
        v_peak_v: f64,
    ) -> [f64; 3] {
        let theta_e =
            (rotor_angle_deg * self.pole_pairs as f64 + v_delta_deg + daxis_deg).to_radians();
        Phase::ALL.map(|phase| v_peak_v * (theta_e + phase.offset_rad()).cos())
    }

    /// Axial current density of one slot for given phase currents.
    ///
    /// `J_z = direction * I_branch * turns / A_copper_slot`, so the
    /// density integrated over the slot copper is exactly
    /// `turns * I_branch` ampere-turns.
    pub fn slot_current_density_a_per_m2(
        &self,
        slot: usize,
        phase_currents_a: &[f64; 3],
        slot_copper_area_m2: f64,
    ) -> f64 {
        let winding = self.layout[slot];
        f64::from(winding.direction) * phase_currents_a[winding.phase.index()] * self.turns_per_slot
            / slot_copper_area_m2.max(1.0e-30)
    }

    /// Phase flux linkages from per-slot mean `A_z` over the slot copper.
    ///
    /// `psi_ph = L_stack / n_parallel * sum_slots direction * turns *
    /// mean_A(slot)`, summed over the slots of each phase. For a sector
    /// model the caller passes slot means replicated over the full
    /// layout (or multiplies by the sector count).
    ///
    /// # Errors
    ///
    /// Returns [`WindingError::SlotCountMismatch`] when the number of
    /// slot means differs from the layout length.
    pub fn phase_flux_linkages_wb_turn(
        &self,
        slot_mean_a_wb_per_m: &[f64],
        stack_length_m: f64,
    ) -> Result<[f64; 3], WindingError> {
        if slot_mean_a_wb_per_m.len() != self.layout.len() {
            return Err(WindingError::SlotCountMismatch {
                expected: self.layout.len(),
                actual: slot_mean_a_wb_per_m.len(),
            });
        }
        let scale = stack_length_m * self.turns_per_slot / self.n_parallel;
        let mut psi = [0.0; 3];
        for (slot, mean_a) in slot_mean_a_wb_per_m.iter().enumerate() {
            let winding = self.layout[slot];
            psi[winding.phase.index()] += f64::from(winding.direction) * mean_a;
        }
        Ok(psi.map(|value| value * scale))
    }
}

/// Mean `A_z` over the quadrature points of one marker region.
///
/// Returns `None` when the marker carries no quadrature weight.
pub fn region_mean_a_wb_per_m(state: &[AzQuadraturePoint], marker: i32) -> Option<f64> {
    let (mut integral, mut area) = (0.0, 0.0);
    for point in state.iter().filter(|point| point.marker == marker) {
        integral += point.vector_potential_wb_per_m * point.integration_weight_m2;
        area += point.integration_weight_m2;
    }
    (area > 0.0).then_some(integral / area)
}

/// Amplitude-invariant Park transform `abc -> dq` at electrical angle.
pub fn park(abc: [f64; 3], theta_e_rad: f64) -> [f64; 2] {
    // 2*pi/3 as a literal, matching the reference implementation so the
    // transforms compare bit-for-bit across codes.
    const TWO_PI_OVER_3: f64 = 2.094_395_102_393_195;
    let d = (2.0 / 3.0)
        * (abc[0] * theta_e_rad.cos()
            + abc[1] * (theta_e_rad - TWO_PI_OVER_3).cos()
            + abc[2] * (theta_e_rad + TWO_PI_OVER_3).cos());
    let q = -(2.0 / 3.0)
        * (abc[0] * theta_e_rad.sin()
            + abc[1] * (theta_e_rad - TWO_PI_OVER_3).sin()
            + abc[2] * (theta_e_rad + TWO_PI_OVER_3).sin());
    [d, q]
}

/// Inverse Park transform `dq -> abc` at electrical angle.
pub fn inverse_park(dq: [f64; 2], theta_e_rad: f64) -> [f64; 3] {
    const TWO_PI_OVER_3: f64 = 2.094_395_102_393_195;
    [
        dq[0] * theta_e_rad.cos() - dq[1] * theta_e_rad.sin(),
        dq[0] * (theta_e_rad - TWO_PI_OVER_3).cos() - dq[1] * (theta_e_rad - TWO_PI_OVER_3).sin(),
        dq[0] * (theta_e_rad + TWO_PI_OVER_3).cos() - dq[1] * (theta_e_rad + TWO_PI_OVER_3).sin(),
    ]
}

/// Step voltages and step-mean currents for one phase over a window.
///
/// For the step ending at frame `k`,
/// `v = R * (i_k + i_{k-1}) / 2 + (psi_k - psi_{k-1}) / dt_k` — the exact
/// mean EMF over the step by Faraday's law. The window is closed
/// periodically unless `previous` supplies the frame before frame 0.
/// Returns `(voltages, mean_currents)`, the pairing under which a
/// lossless inductor exchanges exactly zero energy over a period.
///
/// # Errors
///
/// Returns [`WindingError::FrameCountMismatch`] when the series lengths
/// disagree or fewer than two frames are given without `previous`.
pub fn step_voltage_series(
    psi_wb_turn: &[f64],
    current_a: &[f64],
    resistance_ohm: f64,
    dt_s: &[f64],
    previous: Option<(f64, f64)>,
) -> Result<(Vec<f64>, Vec<f64>), WindingError> {
    let frames = psi_wb_turn.len();
    if frames == 0 || current_a.len() != frames || dt_s.len() != frames {
        return Err(WindingError::FrameCountMismatch);
    }
    if previous.is_none() && frames < 2 {
        return Err(WindingError::FrameCountMismatch);
    }
    if dt_s.iter().any(|dt| !dt.is_finite() || *dt <= 0.0) {
        return Err(WindingError::InvalidTimeStep);
    }
    let mut voltages = Vec::with_capacity(frames);
    let mut mean_currents = Vec::with_capacity(frames);
    for frame in 0..frames {
        let (psi_prev, i_prev) = if frame == 0 {
            previous.unwrap_or((psi_wb_turn[frames - 1], current_a[frames - 1]))
        } else {
            (psi_wb_turn[frame - 1], current_a[frame - 1])
        };
        let i_mid = 0.5 * (current_a[frame] + i_prev);
        voltages.push(resistance_ohm * i_mid + (psi_wb_turn[frame] - psi_prev) / dt_s[frame]);
        mean_currents.push(i_mid);
    }
    Ok((voltages, mean_currents))
}

/// Calibrates the d-axis offset from a no-load `psi_A` rotor sweep.
///
/// `theta_star` is the mechanical rotor angle of maximum `psi_A`
/// (parabolically refined); the offset is
/// `(90 - theta_star * pole_pairs) mod 360`, so that `gamma = 0` lands
/// on the q-axis. Calibrate only on a converged solve: an unconverged
/// no-load sweep misplaces the axis by degrees of electrical angle.
///
/// # Errors
///
/// Returns [`WindingError::FrameCountMismatch`] for mismatched or
/// too-short series.
pub fn calibrate_daxis_deg(
    rotor_angles_deg: &[f64],
    psi_a_wb_turn: &[f64],
    pole_pairs: usize,
) -> Result<f64, WindingError> {
    if pole_pairs == 0 {
        return Err(WindingError::InvalidPolePairs);
    }
    if rotor_angles_deg.len() != psi_a_wb_turn.len() || psi_a_wb_turn.len() < 3 {
        return Err(WindingError::FrameCountMismatch);
    }
    let frames = psi_a_wb_turn.len();
    let peak = (0..frames)
        .max_by(|&a, &b| psi_a_wb_turn[a].total_cmp(&psi_a_wb_turn[b]))
        .expect("non-empty series");
    // Parabolic refinement on the periodic neighbours.
    let (y0, y1, y2) = (
        psi_a_wb_turn[(peak + frames - 1) % frames],
        psi_a_wb_turn[peak],
        psi_a_wb_turn[(peak + 1) % frames],
    );
    let step = if peak + 1 < frames {
        rotor_angles_deg[peak + 1] - rotor_angles_deg[peak]
    } else {
        rotor_angles_deg[1] - rotor_angles_deg[0]
    };
    let denominator = y0 - 2.0 * y1 + y2;
    let offset = if denominator.abs() > 0.0 {
        0.5 * (y0 - y2) / denominator
    } else {
        0.0
    }
    .clamp(-1.0, 1.0);
    let theta_star = rotor_angles_deg[peak] + offset * step;
    Ok((90.0 - theta_star * pole_pairs as f64).rem_euclid(360.0))
}

/// Invalid winding data or series shape.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WindingError {
    /// The slot layout was empty.
    #[error("winding layout must not be empty")]
    EmptyLayout,
    /// A slot direction was not `+1` or `-1`.
    #[error("slot direction must be +1 or -1")]
    InvalidDirection,
    /// Turns per slot were non-positive or non-finite.
    #[error("turns per slot must be finite and positive")]
    InvalidTurns,
    /// Parallel paths were non-positive or non-finite.
    #[error("parallel paths must be finite and positive")]
    InvalidParallelPaths,
    /// Pole-pair count was zero.
    #[error("pole-pair count must be positive")]
    InvalidPolePairs,
    /// Slot-mean count disagreed with the layout.
    #[error("slot means have {actual} entries; layout has {expected}")]
    SlotCountMismatch {
        /// Expected slot count.
        expected: usize,
        /// Actual entry count.
        actual: usize,
    },
    /// Series lengths disagreed or were too short.
    #[error("series lengths must agree and cover at least two frames")]
    FrameCountMismatch,
    /// A time step was non-positive or non-finite.
    #[error("time steps must be finite and positive")]
    InvalidTimeStep,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn winding_12s() -> Winding {
        // The GEO_30MM layout from the reference geometry.
        let raw = [
            (Phase::A, 1),
            (Phase::A, -1),
            (Phase::C, -1),
            (Phase::C, 1),
            (Phase::B, 1),
            (Phase::B, -1),
            (Phase::A, -1),
            (Phase::A, 1),
            (Phase::C, 1),
            (Phase::C, -1),
            (Phase::B, -1),
            (Phase::B, 1),
        ];
        Winding {
            layout: raw
                .iter()
                .map(|&(phase, direction)| SlotWinding { phase, direction })
                .collect(),
            turns_per_slot: 6.0,
            n_parallel: 1.0,
            pole_pairs: 7,
        }
    }

    #[test]
    fn currents_sum_to_zero_and_peak_on_phase_a() {
        let winding = winding_12s();
        let currents = winding.phase_currents_a(0.0, 0.0, 0.0, 84.852_813_742_385_7);
        assert!(currents.iter().sum::<f64>().abs() < 1.0e-9);
        assert!((currents[0] - 84.852_813_742_385_7).abs() < 1.0e-12);
    }

    #[test]
    fn slot_density_integrates_to_turns_times_current() {
        let winding = winding_12s();
        let area = 6.0e-6;
        let currents = [37.0, -11.0, -26.0];
        for slot in 0..12 {
            let density = winding.slot_current_density_a_per_m2(slot, &currents, area);
            let layout = winding.layout[slot];
            let expected = f64::from(layout.direction) * currents[layout.phase.index()] * 6.0;
            assert!((density * area - expected).abs() < 1.0e-9, "slot {slot}");
        }
    }

    #[test]
    fn park_round_trips() {
        let abc = [3.0, -1.0, -2.0];
        let dq = park(abc, 0.7);
        let back = inverse_park(dq, 0.7);
        for (a, b) in abc.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1.0e-12);
        }
    }

    #[test]
    fn step_voltage_of_pure_inductor_exchanges_zero_energy() {
        let frames = 360;
        let dt: Vec<f64> = vec![1.0e-4; frames];
        let psi: Vec<f64> = (0..frames)
            .map(|k| 0.05 * (2.0 * PI * k as f64 / frames as f64).cos())
            .collect();
        let current: Vec<f64> = psi.iter().map(|p| p / 0.001).collect();
        let (voltage, mean_current) = step_voltage_series(&psi, &current, 0.0, &dt, None).unwrap();
        let energy: f64 = voltage
            .iter()
            .zip(&mean_current)
            .zip(&dt)
            .map(|((v, i), d)| v * i * d)
            .sum();
        assert!(energy.abs() < 1.0e-12, "net energy {energy:e}");
    }

    #[test]
    fn daxis_calibration_recovers_known_offset() {
        // psi_A = cos(p*theta - phi0) peaks at theta* = phi0/p; choose
        // theta* = 30/7 deg for p = 7, the 12s14p reference value, so
        // daxis = 90 - 30 = 60 deg.
        let frames = 720;
        let period_mech = 360.0 / 7.0;
        let angles: Vec<f64> = (0..frames)
            .map(|k| period_mech * k as f64 / frames as f64)
            .collect();
        let theta_star = 30.0 / 7.0;
        let psi: Vec<f64> = angles
            .iter()
            .map(|theta| (7.0 * (theta - theta_star)).to_radians().cos())
            .collect();
        let daxis = calibrate_daxis_deg(&angles, &psi, 7).unwrap();
        assert!((daxis - 60.0).abs() < 0.05, "daxis {daxis}");
    }
}
