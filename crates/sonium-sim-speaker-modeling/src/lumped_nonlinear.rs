//! Fast coupled nonlinear driver/box/port time-domain model.
//!
//! This is the M2 design path: it keeps application-level enclosure choices in
//! the modeling crate while reusing the FEM crate's audited voice-coil circuit
//! step. Diaphragm velocity is positive into the box and port volume velocity
//! is positive from the box to the exterior.

use serde::{Deserialize, Serialize};
use sonium_fem::physics::coupled::lumped_circuit::{
    CircuitState, VoiceCoilParams, circuit_step_audited,
};

use crate::port_flow::analysis::CompressionPoint;
use crate::{
    AcousticMedium, EnclosureSpec, LargeSignalSpec, ModelingError, MovingAssemblyNonlinearity,
    PortSpec, SpeakerDriverStudy, require_finite, require_nonnegative, require_positive,
};

/// Box-compliance law used by the fast acoustic circuit.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum BoxComplianceModel {
    /// Constant small-signal compliance `V/(rho*c^2)`.
    Linear,
    /// Adiabatic tangent bulk modulus `rho*c^2 + gamma*p`.
    #[default]
    Adiabatic,
}

/// Controls for the M2 acoustic circuit.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LumpedNonlinearAcousticSpec {
    /// Combined inlet/outlet minor-loss coefficient.
    pub port_loss_coefficient: f64,
    pub box_compliance_model: BoxComplianceModel,
    /// Heat-capacity ratio used by the adiabatic law.
    pub heat_capacity_ratio: f64,
    pub max_coupling_iterations: usize,
    pub coupling_tolerance: f64,
}

impl Default for LumpedNonlinearAcousticSpec {
    fn default() -> Self {
        Self {
            port_loss_coefficient: 1.5,
            box_compliance_model: BoxComplianceModel::Adiabatic,
            heat_capacity_ratio: 1.4,
            max_coupling_iterations: 32,
            coupling_tolerance: 1.0e-10,
        }
    }
}

impl LumpedNonlinearAcousticSpec {
    pub fn validate(self) -> Result<(), ModelingError> {
        require_nonnegative(
            "large_signal.acoustic.port_loss_coefficient",
            self.port_loss_coefficient,
        )?;
        require_positive(
            "large_signal.acoustic.heat_capacity_ratio",
            self.heat_capacity_ratio,
        )?;
        if self.heat_capacity_ratio <= 1.0 {
            return Err(ModelingError::OutOfRange {
                field: "large_signal.acoustic.heat_capacity_ratio",
                constraint: "greater than one",
            });
        }
        if self.max_coupling_iterations == 0 || self.max_coupling_iterations > 1_000 {
            return Err(ModelingError::OutOfRange {
                field: "large_signal.acoustic.max_coupling_iterations",
                constraint: "between 1 and 1000",
            });
        }
        require_positive(
            "large_signal.acoustic.coupling_tolerance",
            self.coupling_tolerance,
        )
    }
}

/// Acoustic state advanced alongside the electrical/mechanical state.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct AcousticCircuitState {
    pub box_pressure_pa: f64,
    pub port_flow_m3s: f64,
}

/// Whole-run signed energy ledger for the fast path.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct LumpedNonlinearEnergyBalance {
    pub source_j: f64,
    pub copper_loss_j: f64,
    pub mechanical_damping_j: f64,
    pub box_loss_j: f64,
    pub port_loss_j: f64,
    pub initial_stored_j: f64,
    pub final_stored_j: f64,
    pub numerical_residual_j: f64,
    pub relative_closure: f64,
}

/// Coupling diagnostics retained with persisted M2 results.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct LumpedNonlinearDiagnostics {
    pub maximum_coupling_iterations: usize,
    pub maximum_coupling_residual: f64,
    pub energy: LumpedNonlinearEnergyBalance,
}

/// One M2 drive-level result.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LumpedNonlinearCompressionPoint {
    pub voltage_rms_v: f64,
    pub port_flow_amplitude_m3s: f64,
    pub box_pressure_amplitude_pa: f64,
    pub estimated_port_spl_db: f64,
    pub compression_db: f64,
    pub acoustic_thd: f64,
    pub acoustic_imd: f64,
    pub peak_excursion_m: f64,
}

/// Fast M2 voltage sweep used by optimizers and studio charts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LumpedNonlinearCompressionSweep {
    pub provenance: String,
    pub frequency_hz: f64,
    pub port_loss_coefficient: f64,
    pub points: Vec<LumpedNonlinearCompressionPoint>,
    pub max_spl_estimate_db: Option<f64>,
}

/// Result of fitting the M2 minor-loss coefficient to an M1 sweep.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PortLossCalibration {
    pub calibrated_loss_coefficient: f64,
    pub rms_relative_flow_error: f64,
    pub maximum_relative_flow_error: f64,
    pub evaluations: usize,
    pub accepted: bool,
}

impl SpeakerDriverStudy {
    /// Run the M2 fast path over increasing positive RMS drive levels.
    pub fn try_lumped_nonlinear_compression_sweep(
        &self,
        base_spec: LargeSignalSpec,
        voltages_rms_v: &[f64],
    ) -> Result<LumpedNonlinearCompressionSweep, ModelingError> {
        if !matches!(self.enclosure, EnclosureSpec::Vented { .. }) {
            return Err(ModelingError::OutOfRange {
                field: "large_signal.enclosure",
                constraint: "vented for a port-compression sweep",
            });
        }
        validate_voltages(voltages_rms_v)?;
        let mut raw = Vec::with_capacity(voltages_rms_v.len());
        for &voltage in voltages_rms_v {
            let mut spec = base_spec;
            spec.voltage_rms_v = voltage;
            let result = self.try_large_signal_response(spec)?;
            let spl = result
                .estimated_port_spl_db
                .ok_or(ModelingError::OutOfRange {
                    field: "large_signal.port_flow",
                    constraint: "positive for a vented compression sweep",
                })?;
            raw.push((
                voltage,
                result.port_flow_fundamental_peak_m3s,
                result.acoustic_pressure_fundamental_peak_pa,
                spl,
                result.acoustic_pressure_thd,
                result.acoustic_pressure_imd,
                result.peak_displacement_m,
            ));
        }
        let reference = *raw
            .iter()
            .min_by(|left, right| left.0.total_cmp(&right.0))
            .expect("validated non-empty voltages");
        let mut max_spl_estimate_db: Option<f64> = None;
        let points = raw
            .into_iter()
            .map(|(voltage, flow, pressure, spl, thd, imd, excursion)| {
                let linear_flow = reference.1 * voltage / reference.0;
                let compression_db = 20.0 * (linear_flow / flow.max(f64::MIN_POSITIVE)).log10();
                let excursion_ok = self
                    .moving_system
                    .xmax_m
                    .is_none_or(|xmax| excursion <= xmax);
                if compression_db <= 3.0 && excursion_ok {
                    max_spl_estimate_db =
                        Some(max_spl_estimate_db.map_or(spl, |current| current.max(spl)));
                }
                LumpedNonlinearCompressionPoint {
                    voltage_rms_v: voltage,
                    port_flow_amplitude_m3s: flow,
                    box_pressure_amplitude_pa: pressure,
                    estimated_port_spl_db: spl,
                    compression_db,
                    acoustic_thd: thd,
                    acoustic_imd: imd,
                    peak_excursion_m: excursion,
                }
            })
            .collect();
        Ok(LumpedNonlinearCompressionSweep {
            provenance: "LumpedNonlinearAcousticCircuit[Driver,Box,QuasiSteadyPort]".into(),
            frequency_hz: base_spec.frequency_hz,
            port_loss_coefficient: base_spec.acoustic.port_loss_coefficient,
            points,
            max_spl_estimate_db,
        })
    }

    /// Fit `K` to matched M1 flow-amplitude evidence using bounded golden search.
    /// The returned calibration is accepted only when every matched flow point
    /// is within the M2 requirement of 10 percent.
    pub fn calibrate_lumped_port_loss_against_m1(
        &self,
        base_spec: LargeSignalSpec,
        resolved_points: &[CompressionPoint],
        coefficient_bounds: (f64, f64),
    ) -> Result<PortLossCalibration, ModelingError> {
        if resolved_points.len() < 2 {
            return Err(ModelingError::OutOfRange {
                field: "large_signal.m1_calibration_points",
                constraint: "at least two",
            });
        }
        let voltages: Vec<f64> = resolved_points
            .iter()
            .map(|point| point.voltage_rms_v)
            .collect();
        validate_voltages(&voltages)?;
        let (mut lower, mut upper) = coefficient_bounds;
        require_nonnegative("large_signal.calibration.lower_k", lower)?;
        require_positive("large_signal.calibration.upper_k", upper)?;
        if lower >= upper {
            return Err(ModelingError::OutOfRange {
                field: "large_signal.calibration.coefficient_bounds",
                constraint: "strictly increasing",
            });
        }
        let phi = (5.0_f64.sqrt() - 1.0) / 2.0;
        let mut left = upper - phi * (upper - lower);
        let mut right = lower + phi * (upper - lower);
        let mut left_error = calibration_error(self, base_spec, &voltages, resolved_points, left)?;
        let mut right_error =
            calibration_error(self, base_spec, &voltages, resolved_points, right)?;
        let mut evaluations = 2;
        for _ in 0..24 {
            if left_error.0 <= right_error.0 {
                upper = right;
                right = left;
                right_error = left_error;
                left = upper - phi * (upper - lower);
                left_error = calibration_error(self, base_spec, &voltages, resolved_points, left)?;
            } else {
                lower = left;
                left = right;
                left_error = right_error;
                right = lower + phi * (upper - lower);
                right_error =
                    calibration_error(self, base_spec, &voltages, resolved_points, right)?;
            }
            evaluations += 1;
        }
        let (coefficient, error) = if left_error.0 <= right_error.0 {
            (left, left_error)
        } else {
            (right, right_error)
        };
        Ok(PortLossCalibration {
            calibrated_loss_coefficient: coefficient,
            rms_relative_flow_error: error.0,
            maximum_relative_flow_error: error.1,
            evaluations,
            accepted: error.1 < 0.10,
        })
    }
}

fn calibration_error(
    study: &SpeakerDriverStudy,
    mut spec: LargeSignalSpec,
    voltages: &[f64],
    resolved: &[CompressionPoint],
    coefficient: f64,
) -> Result<(f64, f64), ModelingError> {
    spec.acoustic.port_loss_coefficient = coefficient;
    let sweep = study.try_lumped_nonlinear_compression_sweep(spec, voltages)?;
    let errors: Vec<f64> = sweep
        .points
        .iter()
        .zip(resolved)
        .map(|(lumped, resolved)| {
            (lumped.port_flow_amplitude_m3s - resolved.port_flow_amplitude_m3s).abs()
                / resolved
                    .port_flow_amplitude_m3s
                    .abs()
                    .max(f64::MIN_POSITIVE)
        })
        .collect();
    let rms = (errors.iter().map(|error| error.powi(2)).sum::<f64>() / errors.len() as f64).sqrt();
    let maximum = errors.into_iter().fold(0.0_f64, f64::max);
    Ok((rms, maximum))
}

fn validate_voltages(voltages: &[f64]) -> Result<(), ModelingError> {
    if voltages.is_empty()
        || voltages
            .iter()
            .any(|voltage| !voltage.is_finite() || *voltage <= 0.0)
        || voltages.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(ModelingError::OutOfRange {
            field: "large_signal.voltages_rms_v",
            constraint: "non-empty, finite, positive, and strictly increasing",
        });
    }
    Ok(())
}

pub(crate) struct LumpedNonlinearRunConfig {
    pub mms_kg: f64,
    pub rms_kg_per_s: f64,
    pub kms_n_per_m: f64,
    pub sd_m2: f64,
    pub voice_coil: VoiceCoilParams,
    pub assembly: MovingAssemblyNonlinearity,
    pub enclosure: EnclosureSpec,
    pub medium: AcousticMedium,
    pub acoustic: LumpedNonlinearAcousticSpec,
    pub harmonic_front_damping_n_s_per_m: f64,
    pub harmonic_front_mass_kg: f64,
    pub voltage_fn: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    pub dt: f64,
    pub num_steps: usize,
}

pub(crate) struct LumpedNonlinearRunResult {
    pub time_s: Vec<f64>,
    pub displacement_m: Vec<f64>,
    pub velocity_m_per_s: Vec<f64>,
    pub current_a: Vec<f64>,
    pub force_n: Vec<f64>,
    pub box_pressure_pa: Vec<f64>,
    pub port_flow_m3s: Vec<f64>,
    pub diagnostics: LumpedNonlinearDiagnostics,
}

#[derive(Clone, Copy)]
struct AcousticParameters {
    volume_m3: f64,
    box_conductance_m3_per_pa_s: f64,
    port: Option<PortParameters>,
}

#[derive(Clone, Copy)]
struct PortParameters {
    inertance_pa_s2_per_m3: f64,
    linear_resistance_pa_s_per_m3: f64,
    nonlinear_resistance_scale: f64,
}

impl AcousticParameters {
    fn from_enclosure(
        enclosure: &EnclosureSpec,
        medium: AcousticMedium,
        frequency_hz: f64,
        loss_coefficient: f64,
    ) -> Result<Option<Self>, ModelingError> {
        let (volume_m3, leakage, absorption, port) = match enclosure {
            EnclosureSpec::FreeAir => return Ok(None),
            EnclosureSpec::Sealed {
                volume_m3,
                leakage_resistance_pa_s_per_m3,
                absorption_resistance_pa_s_per_m3,
            } => (
                *volume_m3,
                *leakage_resistance_pa_s_per_m3,
                *absorption_resistance_pa_s_per_m3,
                None,
            ),
            EnclosureSpec::Vented {
                volume_m3,
                port,
                leakage_resistance_pa_s_per_m3,
                absorption_resistance_pa_s_per_m3,
            } => (
                *volume_m3,
                *leakage_resistance_pa_s_per_m3,
                *absorption_resistance_pa_s_per_m3,
                Some(*port),
            ),
            // The lumped nonlinear circuit has no passive-radiator or
            // bandpass front-chamber branch; fail closed instead of
            // silently degrading either variant to a sealed box.
            EnclosureSpec::PassiveRadiator { .. } => {
                return Err(ModelingError::UnsupportedEnclosure(
                    "passive radiator in lumped nonlinear study",
                ));
            }
            EnclosureSpec::Bandpass { .. } => {
                return Err(ModelingError::UnsupportedEnclosure(
                    "bandpass in lumped nonlinear study",
                ));
            }
        };
        let conductance = [leakage, absorption]
            .into_iter()
            .filter(|resistance| *resistance > 0.0)
            .map(|resistance| 1.0 / resistance)
            .sum();
        let port = port
            .map(|port| port_parameters(port, medium, frequency_hz, loss_coefficient))
            .transpose()?;
        Ok(Some(Self {
            volume_m3,
            box_conductance_m3_per_pa_s: conductance,
            port,
        }))
    }
}

fn port_parameters(
    port: PortSpec,
    medium: AcousticMedium,
    frequency_hz: f64,
    loss_coefficient: f64,
) -> Result<PortParameters, ModelingError> {
    let area = port.area_m2()?;
    let effective_length = port.length_m + port.end_correction_m;
    require_positive("enclosure.port.effective_length_m", effective_length)?;
    let inertance = medium.density_kg_per_m3 * effective_length / area;
    let linear_resistance = port.acoustic_impedance(frequency_hz, medium)?.re;
    Ok(PortParameters {
        inertance_pa_s2_per_m3: inertance,
        linear_resistance_pa_s_per_m3: linear_resistance,
        nonlinear_resistance_scale: medium.density_kg_per_m3 * loss_coefficient / area.powi(2),
    })
}

pub(crate) fn run_lumped_nonlinear(
    config: LumpedNonlinearRunConfig,
    frequency_hz: f64,
) -> Result<LumpedNonlinearRunResult, ModelingError> {
    validate_run_config(&config)?;
    let acoustic = AcousticParameters::from_enclosure(
        &config.enclosure,
        config.medium,
        frequency_hz,
        config.acoustic.port_loss_coefficient,
    )?;
    let dt = config.dt;
    let beta = 0.25;
    let gamma = 0.5;
    let effective_mass = config.mms_kg + config.harmonic_front_mass_kg;
    let effective_damping = config.rms_kg_per_s + config.harmonic_front_damping_n_s_per_m;
    let effective_operator =
        effective_mass + gamma * dt * effective_damping + beta * dt.powi(2) * config.kms_n_per_m;

    let capacity = config.num_steps + 1;
    let mut time_s = Vec::with_capacity(capacity);
    let mut displacement_m = Vec::with_capacity(capacity);
    let mut velocity_m_per_s = Vec::with_capacity(capacity);
    let mut current_a = Vec::with_capacity(capacity);
    let mut force_n = Vec::with_capacity(capacity);
    let mut box_pressure_pa = Vec::with_capacity(capacity);
    let mut port_flow_m3s = Vec::with_capacity(capacity);

    let mut x = 0.0;
    let mut v = 0.0;
    let mut acceleration = 0.0;
    let mut circuit = CircuitState::at_rest();
    let mut acoustic_state = AcousticCircuitState::default();
    push_state(
        0.0,
        x,
        v,
        0.0,
        0.0,
        acoustic_state,
        &mut time_s,
        &mut displacement_m,
        &mut velocity_m_per_s,
        &mut current_a,
        &mut force_n,
        &mut box_pressure_pa,
        &mut port_flow_m3s,
    );

    let mut energy = LumpedNonlinearEnergyBalance {
        initial_stored_j: stored_energy(&config, acoustic, x, v, circuit.current, acoustic_state),
        ..Default::default()
    };
    let mut max_iterations = 0;
    let mut max_residual = 0.0_f64;

    for step in 0..config.num_steps {
        let t_next = (step + 1) as f64 * dt;
        let x_pred = x + dt * v + (0.5 - beta) * dt.powi(2) * acceleration;
        let v_pred = v + (1.0 - gamma) * dt * acceleration;
        let voltage = (config.voltage_fn)(t_next);
        let circuit_audit = circuit_step_audited(
            &config.voice_coil,
            &mut circuit,
            voltage,
            x_pred,
            v_pred,
            dt,
        );
        let spring_pred = config.kms_n_per_m * x_pred + config.assembly.nonlinear_force(x_pred);
        let nonlinear_damping_pred = config.assembly.nonlinear_damping_force(x_pred, v_pred);

        let mut a_next = acceleration;
        let mut next_acoustic = acoustic_state;
        let mut converged = acoustic.is_none();
        let mut accepted_residual = 0.0;
        let mut iterations = 0;
        for iteration in 1..=config.acoustic.max_coupling_iterations {
            iterations = iteration;
            let v_next = v_pred + gamma * dt * a_next;
            let candidate_acoustic = if let Some(parameters) = acoustic {
                advance_acoustic(
                    parameters,
                    config.medium,
                    config.acoustic,
                    acoustic_state,
                    next_acoustic,
                    config.sd_m2 * v,
                    config.sd_m2 * v_next,
                    dt,
                )?
            } else {
                AcousticCircuitState::default()
            };
            let candidate_acceleration = (circuit_audit.force_em_n
                - effective_damping * v_pred
                - nonlinear_damping_pred
                - spring_pred
                - config.sd_m2 * candidate_acoustic.box_pressure_pa)
                / effective_operator;
            accepted_residual = relative_change(candidate_acceleration, a_next)
                .max(relative_change(
                    candidate_acoustic.box_pressure_pa,
                    next_acoustic.box_pressure_pa,
                ))
                .max(relative_change(
                    candidate_acoustic.port_flow_m3s,
                    next_acoustic.port_flow_m3s,
                ));
            a_next = candidate_acceleration;
            next_acoustic = candidate_acoustic;
            if accepted_residual <= config.acoustic.coupling_tolerance {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(ModelingError::LargeSignalAcousticCouplingDidNotConverge {
                step,
                residual: format!("{accepted_residual:e}"),
            });
        }
        max_iterations = max_iterations.max(iterations);
        max_residual = max_residual.max(accepted_residual);

        x = x_pred + beta * dt.powi(2) * a_next;
        v = v_pred + gamma * dt * a_next;
        acceleration = a_next;
        acoustic_state = next_acoustic;
        ensure_finite_state(
            x,
            v,
            circuit.current,
            circuit_audit.force_em_n,
            acoustic_state,
        )?;

        energy.source_j += circuit_audit.source_energy_j(dt);
        energy.copper_loss_j += circuit_audit.copper_loss_j(dt);
        energy.mechanical_damping_j += dt * effective_damping * v.powi(2);
        if let Some(parameters) = acoustic {
            energy.box_loss_j += dt
                * parameters.box_conductance_m3_per_pa_s
                * acoustic_state.box_pressure_pa.powi(2);
            if let Some(port) = parameters.port {
                let resistance = port.linear_resistance_pa_s_per_m3
                    + port.nonlinear_resistance_scale * acoustic_state.port_flow_m3s.abs();
                energy.port_loss_j += dt * resistance * acoustic_state.port_flow_m3s.powi(2);
            }
        }
        push_state(
            t_next,
            x,
            v,
            circuit.current,
            circuit_audit.force_em_n,
            acoustic_state,
            &mut time_s,
            &mut displacement_m,
            &mut velocity_m_per_s,
            &mut current_a,
            &mut force_n,
            &mut box_pressure_pa,
            &mut port_flow_m3s,
        );
    }

    energy.final_stored_j = stored_energy(&config, acoustic, x, v, circuit.current, acoustic_state);
    energy.numerical_residual_j = energy.source_j
        - energy.copper_loss_j
        - energy.mechanical_damping_j
        - energy.box_loss_j
        - energy.port_loss_j
        - (energy.final_stored_j - energy.initial_stored_j);
    let energy_scale = energy
        .source_j
        .abs()
        .max((energy.final_stored_j - energy.initial_stored_j).abs())
        .max(
            energy.copper_loss_j
                + energy.mechanical_damping_j
                + energy.box_loss_j
                + energy.port_loss_j,
        )
        .max(f64::MIN_POSITIVE);
    energy.relative_closure = energy.numerical_residual_j.abs() / energy_scale;

    Ok(LumpedNonlinearRunResult {
        time_s,
        displacement_m,
        velocity_m_per_s,
        current_a,
        force_n,
        box_pressure_pa,
        port_flow_m3s,
        diagnostics: LumpedNonlinearDiagnostics {
            maximum_coupling_iterations: max_iterations,
            maximum_coupling_residual: max_residual,
            energy,
        },
    })
}

fn advance_acoustic(
    parameters: AcousticParameters,
    medium: AcousticMedium,
    spec: LumpedNonlinearAcousticSpec,
    old: AcousticCircuitState,
    iterate: AcousticCircuitState,
    old_diaphragm_flow_m3s: f64,
    new_diaphragm_flow_m3s: f64,
    dt: f64,
) -> Result<AcousticCircuitState, ModelingError> {
    let linear_bulk = medium.density_kg_per_m3 * medium.sound_speed_m_per_s.powi(2);
    let bulk = match spec.box_compliance_model {
        BoxComplianceModel::Linear => linear_bulk,
        BoxComplianceModel::Adiabatic => {
            linear_bulk + spec.heat_capacity_ratio * iterate.box_pressure_pa
        }
    };
    if !bulk.is_finite() || bulk <= 0.0 {
        return Err(ModelingError::OutOfRange {
            field: "large_signal.box_pressure_pa",
            constraint: "must preserve positive tangent bulk modulus",
        });
    }
    let box_rate = bulk / parameters.volume_m3;
    let half_dt = 0.5 * dt;
    let pressure_diagonal = 1.0 + half_dt * box_rate * parameters.box_conductance_m3_per_pa_s;
    let pressure_rhs = old.box_pressure_pa
        + half_dt
            * box_rate
            * (old_diaphragm_flow_m3s + new_diaphragm_flow_m3s
                - old.port_flow_m3s
                - parameters.box_conductance_m3_per_pa_s * old.box_pressure_pa);
    let Some(port) = parameters.port else {
        return Ok(AcousticCircuitState {
            box_pressure_pa: pressure_rhs / pressure_diagonal,
            port_flow_m3s: 0.0,
        });
    };
    let port_resistance = port.linear_resistance_pa_s_per_m3
        + port.nonlinear_resistance_scale * iterate.port_flow_m3s.abs();
    let old_port_resistance = port.linear_resistance_pa_s_per_m3
        + port.nonlinear_resistance_scale * old.port_flow_m3s.abs();
    let flow_diagonal = 1.0 + half_dt * port_resistance / port.inertance_pa_s2_per_m3;
    let pressure_to_flow = half_dt / port.inertance_pa_s2_per_m3;
    let flow_to_pressure = half_dt * box_rate;
    let flow_rhs = old.port_flow_m3s
        + pressure_to_flow * (old.box_pressure_pa - old_port_resistance * old.port_flow_m3s);
    let determinant = pressure_diagonal * flow_diagonal + flow_to_pressure * pressure_to_flow;
    require_positive("large_signal.acoustic_step_determinant", determinant)?;
    Ok(AcousticCircuitState {
        box_pressure_pa: (pressure_rhs * flow_diagonal - flow_to_pressure * flow_rhs) / determinant,
        port_flow_m3s: (pressure_diagonal * flow_rhs + pressure_to_flow * pressure_rhs)
            / determinant,
    })
}

fn validate_run_config(config: &LumpedNonlinearRunConfig) -> Result<(), ModelingError> {
    require_positive("large_signal.mms_kg", config.mms_kg)?;
    require_nonnegative("large_signal.rms_kg_per_s", config.rms_kg_per_s)?;
    require_positive("large_signal.kms_n_per_m", config.kms_n_per_m)?;
    require_positive("large_signal.sd_m2", config.sd_m2)?;
    require_nonnegative(
        "large_signal.harmonic_front_damping_n_s_per_m",
        config.harmonic_front_damping_n_s_per_m,
    )?;
    require_finite(
        "large_signal.harmonic_front_mass_kg",
        config.harmonic_front_mass_kg,
    )?;
    if config.mms_kg + config.harmonic_front_mass_kg <= 0.0 {
        return Err(ModelingError::OutOfRange {
            field: "large_signal.effective_mass_kg",
            constraint: "greater than zero",
        });
    }
    require_positive("large_signal.dt", config.dt)?;
    if config.num_steps == 0 {
        return Err(ModelingError::OutOfRange {
            field: "large_signal.num_steps",
            constraint: "greater than zero",
        });
    }
    config.enclosure.validate()?;
    config.medium.validate()?;
    config.acoustic.validate()?;
    config.assembly.validate()
}

fn stored_energy(
    config: &LumpedNonlinearRunConfig,
    acoustic: Option<AcousticParameters>,
    displacement: f64,
    velocity: f64,
    current: f64,
    state: AcousticCircuitState,
) -> f64 {
    let effective_mass = config.mms_kg + config.harmonic_front_mass_kg;
    let mut energy = 0.5 * effective_mass * velocity.powi(2)
        + 0.5 * config.kms_n_per_m * displacement.powi(2)
        + config.assembly.potential_energy(displacement)
        + 0.5 * config.voice_coil.le_at(displacement) * current.powi(2);
    if let Some(parameters) = acoustic {
        let linear_bulk =
            config.medium.density_kg_per_m3 * config.medium.sound_speed_m_per_s.powi(2);
        let pressure = state.box_pressure_pa;
        energy += match config.acoustic.box_compliance_model {
            BoxComplianceModel::Linear => {
                0.5 * parameters.volume_m3 * pressure.powi(2) / linear_bulk
            }
            BoxComplianceModel::Adiabatic => {
                let gamma = config.acoustic.heat_capacity_ratio;
                let ratio = (linear_bulk + gamma * pressure) / linear_bulk;
                parameters.volume_m3 / gamma.powi(2) * (gamma * pressure - linear_bulk * ratio.ln())
            }
        };
        if let Some(port) = parameters.port {
            energy += 0.5 * port.inertance_pa_s2_per_m3 * state.port_flow_m3s.powi(2);
        }
    }
    energy
}

fn relative_change(next: f64, previous: f64) -> f64 {
    (next - previous).abs() / next.abs().max(previous.abs()).max(1.0)
}

#[allow(clippy::too_many_arguments)]
fn push_state(
    time: f64,
    displacement: f64,
    velocity: f64,
    current: f64,
    force: f64,
    acoustic: AcousticCircuitState,
    time_s: &mut Vec<f64>,
    displacement_m: &mut Vec<f64>,
    velocity_m_per_s: &mut Vec<f64>,
    current_a: &mut Vec<f64>,
    force_n: &mut Vec<f64>,
    box_pressure_pa: &mut Vec<f64>,
    port_flow_m3s: &mut Vec<f64>,
) {
    time_s.push(time);
    displacement_m.push(displacement);
    velocity_m_per_s.push(velocity);
    current_a.push(current);
    force_n.push(force);
    box_pressure_pa.push(acoustic.box_pressure_pa);
    port_flow_m3s.push(acoustic.port_flow_m3s);
}

fn ensure_finite_state(
    displacement: f64,
    velocity: f64,
    current: f64,
    force: f64,
    acoustic: AcousticCircuitState,
) -> Result<(), ModelingError> {
    if [
        displacement,
        velocity,
        current,
        force,
        acoustic.box_pressure_pa,
        acoustic.port_flow_m3s,
    ]
    .into_iter()
    .all(f64::is_finite)
    {
        Ok(())
    } else {
        Err(ModelingError::NonFinite {
            field: "large_signal.response",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::port_flow::analysis::dft_phasor;

    fn linear_spec(frequency_hz: f64) -> LargeSignalSpec {
        LargeSignalSpec {
            frequency_hz,
            voltage_rms_v: 0.01,
            duration_s: 24.0 / frequency_hz,
            samples_per_period: 256,
            bl_drop_fraction: 0.0,
            inductance_gradient_h_per_m: 0.0,
            suspension_cubic_n_per_m3: 0.0,
            acoustic: LumpedNonlinearAcousticSpec {
                port_loss_coefficient: 0.0,
                box_compliance_model: BoxComplianceModel::Linear,
                ..LumpedNonlinearAcousticSpec::default()
            },
            ..LargeSignalSpec::default()
        }
    }

    fn vented_enclosure() -> EnclosureSpec {
        EnclosureSpec::Vented {
            volume_m3: 0.02,
            port: PortSpec {
                radius_m: 0.025,
                length_m: 0.12,
                end_correction_m: 0.03,
                resistance_pa_s_per_m3: 25.0,
            },
            leakage_resistance_pa_s_per_m3: 0.0,
            absorption_resistance_pa_s_per_m3: 0.0,
        }
    }

    #[test]
    fn sealed_and_vented_small_signal_impedance_match_frequency_domain() {
        let frequency_hz = 60.0;
        let spec = linear_spec(frequency_hz);
        for enclosure in [
            EnclosureSpec::Sealed {
                volume_m3: 0.02,
                leakage_resistance_pa_s_per_m3: 0.0,
                absorption_resistance_pa_s_per_m3: 0.0,
            },
            vented_enclosure(),
        ] {
            let mut study = SpeakerDriverStudy::transient_large_signal_driver();
            study.enclosure = enclosure.clone();
            let result = study.try_large_signal_response(spec).unwrap();
            let tail_samples = 8 * spec.samples_per_period;
            let start = result.time_s.len() - tail_samples;
            let pressure = dft_phasor(
                &result.box_pressure_pa[start..],
                spec.time_step_s().unwrap(),
                frequency_hz,
            );
            let diaphragm_flow = dft_phasor(
                &result.diaphragm_flow_m3s[start..],
                spec.time_step_s().unwrap(),
                frequency_hz,
            );
            // `dft_phasor` represents real signals as Re{X exp(-iωt)}, while
            // `EnclosureSpec::acoustic_impedance` uses the crate's historical
            // exp(+iωt) frequency-domain convention.
            let numerical = (pressure / diaphragm_flow).conj();
            let exact = enclosure
                .acoustic_impedance(frequency_hz, study.acoustic_medium)
                .unwrap();
            let relative = (numerical - exact).norm() / exact.norm();
            assert!(
                relative < 0.02,
                "{enclosure:?}: numerical={numerical:?}, exact={exact:?}, relative={relative:e}"
            );
        }
    }

    #[test]
    fn linear_acoustic_limit_is_harmonic_free_and_energy_is_bounded() {
        let mut study = SpeakerDriverStudy::transient_large_signal_driver();
        study.enclosure = vented_enclosure();
        let result = study.try_large_signal_response(linear_spec(60.0)).unwrap();
        assert!(result.acoustic_pressure_thd < 1.0e-4);
        let energy = result.acoustic_diagnostics.unwrap().energy;
        assert!(energy.copper_loss_j >= 0.0);
        assert!(energy.mechanical_damping_j >= 0.0);
        assert!(energy.port_loss_j >= 0.0);
        assert!(energy.relative_closure < 0.10, "energy={energy:?}");
    }

    #[test]
    fn port_loss_calibration_recovers_synthetic_m1_curve() {
        let mut study = SpeakerDriverStudy::transient_large_signal_driver();
        study.enclosure = vented_enclosure();
        let voltages = [0.05, 0.5, 2.0];
        let mut reference_spec = linear_spec(45.0);
        reference_spec.duration_s = 12.0 / reference_spec.frequency_hz;
        reference_spec.samples_per_period = 96;
        reference_spec.acoustic.box_compliance_model = BoxComplianceModel::Adiabatic;
        reference_spec.acoustic.port_loss_coefficient = 4.0;
        let reference = study
            .try_lumped_nonlinear_compression_sweep(reference_spec, &voltages)
            .unwrap();
        let resolved: Vec<CompressionPoint> = reference
            .points
            .iter()
            .map(|point| CompressionPoint {
                voltage_rms_v: point.voltage_rms_v,
                port_flow_amplitude_m3s: point.port_flow_amplitude_m3s,
                estimated_port_spl_db: point.estimated_port_spl_db,
                compression_db: point.compression_db,
                thd: point.acoustic_thd,
            })
            .collect();
        let mut initial = reference_spec;
        initial.acoustic.port_loss_coefficient = 1.0;
        let calibration = study
            .calibrate_lumped_port_loss_against_m1(initial, &resolved, (0.0, 8.0))
            .unwrap();
        assert!(calibration.accepted, "{calibration:?}");
        assert!(calibration.maximum_relative_flow_error < 0.01);
        assert!((calibration.calibrated_loss_coefficient - 4.0).abs() < 0.1);
    }
}
