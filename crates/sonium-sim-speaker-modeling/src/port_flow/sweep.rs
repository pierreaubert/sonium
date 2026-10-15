//! Drive-level compression sweeps for resolved nonlinear port flow.

use std::cmp::Ordering;

use sonium_fem::mesh::Mesh;
use sonium_fem::spaces::{H1Space, VectorH1Space};

use super::analysis::{CompressionPoint, dft_amplitude, port_spl_db, thd_fraction};
use super::coupled::{CoupledPortConfig, run_coupled_port};
use super::error::PortFlowError;

const PROVENANCE: &str = "ResolvedIncompressiblePortFlow2D[LumpedBoxDriver,NoFrontRadiation]";

/// Results of a drive-level compression sweep at one frequency.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PortCompressionSweep {
    /// Model lineage and limitations for persisted results.
    pub provenance: String,
    /// Drive frequency shared by every point (Hz).
    pub frequency_hz: f64,
    /// Per-drive port output and distortion measurements.
    pub points: Vec<CompressionPoint>,
    /// Highest estimated port SPL satisfying the 3 dB compression gate and,
    /// when supplied, the diaphragm-excursion gate.
    pub max_spl_estimate_db: Option<f64>,
}

struct RawPoint {
    voltage_rms_v: f64,
    port_flow_amplitude_m3s: f64,
    estimated_port_spl_db: f64,
    thd: f64,
    peak_excursion_m: f64,
}

/// Run the coupled port model at each RMS drive level and form a compression
/// curve from the steady-state tail of every record.
///
/// Complete fundamental periods contained in the last 20% of each run are
/// analyzed (at least two periods are required). The lowest positive drive establishes
/// the small-signal reference, extrapolated as `20*log10(V/V0)`. A fresh
/// [`CoupledPortConfig`] is requested for each voltage because it owns boxed
/// closures and is intentionally not `Clone`.
pub fn run_compression_sweep(
    mesh: &Mesh,
    v_space: &VectorH1Space,
    p_space: &H1Space,
    config_factory: impl Fn(f64) -> CoupledPortConfig,
    frequency_hz: f64,
    voltages_rms: &[f64],
    xmax_m: Option<f64>,
) -> Result<PortCompressionSweep, PortFlowError> {
    validate_inputs(frequency_hz, voltages_rms, xmax_m)?;

    let mut raw_points = Vec::with_capacity(voltages_rms.len());
    for &voltage_rms_v in voltages_rms {
        let config = config_factory(voltage_rms_v);
        let dt = config.bdf.dt;
        if !dt.is_finite() || dt <= 0.0 {
            return Err(PortFlowError::InvalidConfig(format!(
                "compression sweep config for {voltage_rms_v} V has invalid dt {dt}"
            )));
        }
        validate_analysis_band(frequency_hz, dt, voltage_rms_v)?;
        let rho = config.ns_config.rho;
        if !rho.is_finite() || rho <= 0.0 {
            return Err(PortFlowError::InvalidConfig(format!(
                "compression sweep config for {voltage_rms_v} V has invalid density {rho}"
            )));
        }

        let result = run_coupled_port(mesh, v_space, p_space, &config)?;
        let sample_count = result.port_flow_m3s.len();
        if sample_count < 10
            || result.displacement_m.len() != sample_count
            || result.time_s.len() != sample_count
            || result.box_pressure_pa.len() != sample_count
            || result.current_a.len() != sample_count
        {
            return Err(PortFlowError::InvalidConfig(format!(
                "compression sweep at {voltage_rms_v} V returned insufficient or inconsistent samples: flow {}, displacement {}, time {}, box pressure {}, current {}",
                result.port_flow_m3s.len(),
                result.displacement_m.len(),
                result.time_s.len(),
                result.box_pressure_pa.len(),
                result.current_a.len()
            )));
        }

        let tail_start = steady_state_tail_start(sample_count, dt, frequency_hz)?;
        let flow_tail = &result.port_flow_m3s[tail_start..];
        let displacement_tail = &result.displacement_m[tail_start..];
        let port_flow_amplitude_m3s = dft_amplitude(flow_tail, dt, frequency_hz);
        let estimated_port_spl_db = port_spl_db(port_flow_amplitude_m3s, frequency_hz, rho, 1.0);
        let thd = thd_fraction(flow_tail, dt, frequency_hz, 5);
        let peak_excursion_m = displacement_tail
            .iter()
            .map(|x| x.abs())
            .fold(0.0, f64::max);
        if !port_flow_amplitude_m3s.is_finite()
            || !estimated_port_spl_db.is_finite()
            || !thd.is_finite()
            || !peak_excursion_m.is_finite()
        {
            return Err(PortFlowError::InvalidConfig(format!(
                "compression sweep analysis at {voltage_rms_v} V produced a non-finite result"
            )));
        }
        raw_points.push(RawPoint {
            voltage_rms_v,
            port_flow_amplitude_m3s,
            estimated_port_spl_db,
            thd,
            peak_excursion_m,
        });
    }

    Ok(finalize_sweep(frequency_hz, raw_points, xmax_m))
}

fn validate_analysis_band(
    frequency_hz: f64,
    dt: f64,
    voltage_rms_v: f64,
) -> Result<(), PortFlowError> {
    let highest_harmonic_hz = 5.0 * frequency_hz;
    let nyquist_hz = 0.5 / dt;
    if highest_harmonic_hz.partial_cmp(&nyquist_hz) != Some(Ordering::Less) {
        return Err(PortFlowError::InvalidConfig(format!(
            "compression sweep config for {voltage_rms_v} V cannot analyze five harmonics: highest harmonic {highest_harmonic_hz} Hz must be below Nyquist {nyquist_hz} Hz for dt {dt}"
        )));
    }
    Ok(())
}

fn steady_state_tail_start(
    sample_count: usize,
    dt: f64,
    frequency_hz: f64,
) -> Result<usize, PortFlowError> {
    let samples_per_period_exact = 1.0 / (dt * frequency_hz);
    let samples_per_period = samples_per_period_exact.round() as usize;
    if samples_per_period == 0
        || (samples_per_period_exact - samples_per_period as f64).abs()
            > 1.0e-10 * samples_per_period_exact
    {
        return Err(PortFlowError::InvalidConfig(format!(
            "compression sweep requires an integer number of samples per period, got {samples_per_period_exact} for dt {dt} and frequency {frequency_hz}"
        )));
    }
    let requested_tail_samples = sample_count.div_ceil(5);
    let complete_periods = requested_tail_samples / samples_per_period;
    if complete_periods < 2 {
        return Err(PortFlowError::InvalidConfig(format!(
            "compression sweep last-20% tail contains only {complete_periods} complete periods; at least 2 are required"
        )));
    }
    Ok(sample_count - complete_periods * samples_per_period)
}

fn validate_inputs(
    frequency_hz: f64,
    voltages_rms: &[f64],
    xmax_m: Option<f64>,
) -> Result<(), PortFlowError> {
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        return Err(PortFlowError::InvalidConfig(format!(
            "compression sweep frequency must be finite and positive, got {frequency_hz}"
        )));
    }
    if voltages_rms.is_empty() {
        return Err(PortFlowError::InvalidConfig(
            "compression sweep requires at least one voltage".to_string(),
        ));
    }
    if let Some((index, voltage)) = voltages_rms
        .iter()
        .enumerate()
        .find(|(_, voltage)| !voltage.is_finite() || **voltage <= 0.0)
    {
        return Err(PortFlowError::InvalidConfig(format!(
            "compression sweep voltage {index} must be finite and positive, got {voltage}"
        )));
    }
    if let Some(xmax) = xmax_m
        && (!xmax.is_finite() || xmax <= 0.0)
    {
        return Err(PortFlowError::InvalidConfig(format!(
            "compression sweep xmax must be finite and positive, got {xmax}"
        )));
    }
    Ok(())
}

fn finalize_sweep(
    frequency_hz: f64,
    raw_points: Vec<RawPoint>,
    xmax_m: Option<f64>,
) -> PortCompressionSweep {
    let reference = raw_points
        .iter()
        .min_by(|left, right| left.voltage_rms_v.total_cmp(&right.voltage_rms_v))
        .expect("validated non-empty sweep");
    let reference_voltage = reference.voltage_rms_v;
    let reference_spl = reference.estimated_port_spl_db;

    let mut max_spl_estimate_db: Option<f64> = None;
    let points = raw_points
        .into_iter()
        .map(|raw| {
            let linear_spl = reference_spl + 20.0 * (raw.voltage_rms_v / reference_voltage).log10();
            let compression_db = linear_spl - raw.estimated_port_spl_db;
            let excursion_ok = xmax_m.is_none_or(|xmax| raw.peak_excursion_m <= xmax);
            if compression_db <= 3.0 && excursion_ok {
                max_spl_estimate_db = Some(
                    max_spl_estimate_db.map_or(raw.estimated_port_spl_db, |current| {
                        current.max(raw.estimated_port_spl_db)
                    }),
                );
            }
            CompressionPoint {
                voltage_rms_v: raw.voltage_rms_v,
                port_flow_amplitude_m3s: raw.port_flow_amplitude_m3s,
                estimated_port_spl_db: raw.estimated_port_spl_db,
                compression_db,
                thd: raw.thd,
            }
        })
        .collect();

    PortCompressionSweep {
        provenance: PROVENANCE.to_string(),
        frequency_hz,
        points,
        max_spl_estimate_db,
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::TAU;

    use sonium_fem::basis::PolynomialDegree;
    use sonium_fem::physics::coupled::lumped_circuit::VoiceCoilParams;
    use sonium_fem::physics::fluid::stokes::navier_stokes::NavierStokesConfig;
    use sonium_fem::physics::fluid::stokes::time_stepping::BdfConfig;
    use sonium_fem::spaces::{H1Space, VectorH1Space};

    use super::*;
    use crate::port_flow::box_driver::BoxDriverModel;
    use crate::port_flow::geometry::{SlotPortGeometry, build_slot_port_mesh};

    fn slot_fixture_coarse() -> (Mesh, VectorH1Space, H1Space) {
        let mesh = build_slot_port_mesh(&SlotPortGeometry {
            duct_length_m: 0.1,
            duct_height_m: 0.02,
            chamber_length_m: 0.05,
            chamber_height_m: 0.08,
            mesh_size_m: 0.01,
        })
        .unwrap();
        (
            mesh,
            VectorH1Space::new(PolynomialDegree::P2, 2),
            H1Space::new(PolynomialDegree::P1),
        )
    }

    fn coupled_fixture_with_voltage(voltage_rms_v: f64) -> CoupledPortConfig {
        let frequency_hz = 40.0;
        let dt = 5e-4;
        CoupledPortConfig {
            driver: BoxDriverModel {
                mms_kg: 0.02,
                rms_kg_per_s: 1.0,
                kms_n_per_m: 4000.0,
                voice_coil: VoiceCoilParams::linear(6.0, 0.5e-3, 10.0),
                compliance_force_fn: None,
                compliance_potential_fn: None,
                sd_m2: 0.01,
                box_volume_m3: Some(0.005),
                medium_density_kg_per_m3: 1.2,
                medium_sound_speed_m_per_s: 343.0,
                beta: 0.25,
                gamma: 0.5,
            },
            ns_config: NavierStokesConfig {
                mu: 1.85e-5,
                rho: 1.2,
                use_residual_stabilisation: false,
                // Zero body force through `solve_bdf_step_cached`: the
                // force quadrature is identical every step, so caching
                // skips two full assemblies per step after the first.
                cache_body_force_assembly: true,
                ..Default::default()
            },
            // Ten periods makes the last 20% exactly two complete periods,
            // the minimum spectrally valid steady-state tail.
            bdf: BdfConfig::bdf2(dt, (10.0 / frequency_hz / dt).round() as usize),
            voltage_fn: Box::new(move |time_s| {
                voltage_rms_v * 2.0_f64.sqrt() * (TAU * frequency_hz * time_s).sin()
            }),
            under_relaxation: 1.0,
            exterior_pressure_pa: 0.0,
        }
    }

    #[test]
    fn compression_curve_is_monotonic_and_labels_provenance() {
        let (mesh, v_space, p_space) = slot_fixture_coarse();
        let sweep = run_compression_sweep(
            &mesh,
            &v_space,
            &p_space,
            coupled_fixture_with_voltage,
            40.0,
            &[0.05, 0.5, 2.0],
            Some(0.005),
        )
        .unwrap();
        assert!(
            sweep
                .provenance
                .contains("ResolvedIncompressiblePortFlow2D")
        );
        assert_eq!(sweep.points.len(), 3);
        assert!(
            sweep.points[0].compression_db.abs() < 0.05,
            "lowest level defines the reference"
        );
        assert!(sweep.points[1].port_flow_amplitude_m3s > sweep.points[0].port_flow_amplitude_m3s);
        assert!(sweep.points[2].thd >= sweep.points[0].thd);
    }

    #[test]
    fn compression_reference_uses_lowest_drive_and_xmax_limits_max_spl() {
        let sweep = finalize_sweep(
            40.0,
            vec![
                RawPoint {
                    voltage_rms_v: 2.0,
                    port_flow_amplitude_m3s: 2.0,
                    estimated_port_spl_db: 105.0,
                    thd: 0.1,
                    peak_excursion_m: 0.006,
                },
                RawPoint {
                    voltage_rms_v: 0.5,
                    port_flow_amplitude_m3s: 0.5,
                    estimated_port_spl_db: 95.0,
                    thd: 0.01,
                    peak_excursion_m: 0.002,
                },
            ],
            Some(0.005),
        );
        assert!((sweep.points[1].compression_db).abs() < f64::EPSILON);
        let expected = 95.0 + 20.0 * (2.0_f64 / 0.5).log10() - 105.0;
        assert!((sweep.points[0].compression_db - expected).abs() < 1e-12);
        assert_eq!(sweep.max_spl_estimate_db, Some(95.0));
    }

    #[test]
    fn invalid_sweep_inputs_return_errors() {
        assert!(matches!(
            validate_inputs(40.0, &[], None),
            Err(PortFlowError::InvalidConfig(_))
        ));
        assert!(matches!(
            validate_inputs(40.0, &[0.1, f64::NAN], None),
            Err(PortFlowError::InvalidConfig(_))
        ));
        assert!(matches!(
            validate_inputs(40.0, &[0.0], None),
            Err(PortFlowError::InvalidConfig(_))
        ));
        assert!(matches!(
            validate_inputs(40.0, &[0.1], Some(-1.0)),
            Err(PortFlowError::InvalidConfig(_))
        ));
    }

    #[test]
    fn public_sweep_rejects_empty_and_insufficient_runs_without_panicking() {
        let (mesh, v_space, p_space) = slot_fixture_coarse();
        let empty = run_compression_sweep(
            &mesh,
            &v_space,
            &p_space,
            |_| panic!("factory must not run for invalid input"),
            40.0,
            &[],
            None,
        );
        assert!(matches!(empty, Err(PortFlowError::InvalidConfig(_))));

        let insufficient = run_compression_sweep(
            &mesh,
            &v_space,
            &p_space,
            |voltage| {
                let mut config = coupled_fixture_with_voltage(voltage);
                config.bdf.num_steps = 0;
                config
            },
            40.0,
            &[0.1],
            None,
        );
        assert!(matches!(insufficient, Err(PortFlowError::InvalidConfig(_))));
    }

    #[test]
    fn public_sweep_rejects_five_harmonics_at_or_above_nyquist() {
        let (mesh, v_space, p_space) = slot_fixture_coarse();
        for dt in [1.0 / 400.0, 1.0 / 320.0] {
            let result = run_compression_sweep(
                &mesh,
                &v_space,
                &p_space,
                |voltage| {
                    let mut config = coupled_fixture_with_voltage(voltage);
                    config.bdf.dt = dt;
                    config.bdf.num_steps = 0;
                    config
                },
                40.0,
                &[0.1],
                None,
            );
            match result {
                Err(PortFlowError::InvalidConfig(message)) => {
                    assert!(message.contains("Nyquist"), "{message}");
                }
                _ => panic!("expected Nyquist InvalidConfig for dt {dt}"),
            }
        }
    }

    #[test]
    fn tail_start_selects_complete_periods_within_last_twenty_percent() {
        assert_eq!(steady_state_tail_start(500, 5.0e-4, 40.0).unwrap(), 400);
        assert!(steady_state_tail_start(150, 5.0e-4, 40.0).is_err());
        assert!(steady_state_tail_start(500, 4.9e-4, 40.0).is_err());
    }

    #[test]
    fn complete_cycle_sweep_tail_reports_negligible_thd_for_pure_sine() {
        let dt = 5.0e-4;
        let frequency_hz = 40.0;
        let sample_count = 500;
        let signal: Vec<f64> = (1..=sample_count)
            .map(|sample| (TAU * frequency_hz * sample as f64 * dt).sin())
            .collect();
        let tail_start = steady_state_tail_start(sample_count, dt, frequency_hz).unwrap();
        let thd = thd_fraction(&signal[tail_start..], dt, frequency_hz, 5);
        assert!(thd < 1.0e-12, "pure-sine sweep tail THD = {thd:e}");
    }
}
