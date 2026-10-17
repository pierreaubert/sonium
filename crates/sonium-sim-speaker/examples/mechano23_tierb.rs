//! Tier B comparison for the open-source XMechanik Mechano23.
//!
//! This example makes three independent calculations on one frequency grid:
//!
//! 1. **Sonium prediction** — rigid-piston BEM radiation from the published
//!    cabinet and driver geometry. Piston velocities come from the drivers'
//!    lumped electromechanical parameters, the woofer's vented rear load, and
//!    the exact published crossover network loaded by measured in-cabinet
//!    driver impedances from the public VituixCAD project.
//! 2. **VituixCAD-project reference** — the crossover transfer functions are
//!    applied to XMechanik's measured per-driver polar FRDs and summed as
//!    complex pressures. This reproduces the project calculation from its
//!    published inputs; it does not execute the Windows VituixCAD binary.
//! 3. **ASR reference** — the NFS-derived horizontal/vertical SPL planes in
//!    `fixtures/tierb/mechano23`, aggregated with the same CEA-2034 groups
//!    and spherical weights as the Sonium metrics.
//!
//! The BEM uses peak phasors (`e^{-iωt}`); circuit and lumped calculations
//! use RMS phasors (`e^{+jωt}`) and are conjugated and scaled by √2 at the
//! transfer-function boundary. No measured curve is fitted to another. For
//! each curve the output reports the broadband sensitivity offset separately
//! and then the shape error after removing only that one offset.
//!
//! Usage:
//! `cargo run --release -p sonium-sim-speaker --example mechano23_tierb -- [out-dir] [spinorama-dir] [vituix-dir]`
//!
//! The frequency grid can be reduced for smoke runs without recompiling via
//! `MECHANO23_FREQ_START_HZ`, `MECHANO23_FREQ_POINTS`, `MECHANO23_FREQ_END_HZ`,
//! and `MECHANO23_MAX_MESH_HZ` (defaults: 21 points, 50 Hz–8 kHz, mesh sized
//! at λ/8 for 8 kHz). Solver probes can set `MECHANO23_DENSE=1` to force the
//! dense FGMRES+ILU path instead of H-matrix assembly, plus
//! `MECHANO23_TOLERANCE`, `MECHANO23_MAX_LANES`, `MECHANO23_RESTART`,
//! `MECHANO23_RECYCLE`, `MECHANO23_ACA_TOLERANCE`, and
//! `MECHANO23_MAX_ITERATIONS`. With the crate's `cuda` feature,
//! `MECHANO23_CUDA=1` runs the dense CUDA solver instead of the CPU
//! paths.
//!
//! Measured cost (128-core host, October 2026): the 4 kHz grid
//! (`MECHANO23_FREQ_POINTS=17 MECHANO23_FREQ_END_HZ=4000
//! MECHANO23_MAX_MESH_HZ=4000`, 6,494 DOFs) completes in ~9 minutes, while
//! the default 8 kHz mesh (25,014 DOFs) did not finish a full sweep in
//! 2 hours. See `fixtures/tierb/README.md` for the recorded metrics and
//! the solver-cost boundary.
//!
//! `vituix-dir` defaults to `$SONIUM_MECHANO23_VXP_DIR` and then
//! `tmp/mechano_vituix/Mechano-23`. See `fixtures/tierb/README.md` for the
//! public archive link, hashes, geometry, and licensing.

// Rust guideline compliant 2026-02-21

use num_complex::Complex64;
use sonium_roomeq::cea2034::try_cea2034;
use sonium_sim_speaker::cea2034_output;
use sonium_sim_speaker::model::{
    CabinetDimensions, DriverConfig, LoudspeakerModel, SeparatorConfig, SimulationConfig,
    TransferPoint, WaveguideConfig,
};
use sonium_sim_speaker::simulation::{run_simulation_with_config, SolverConfig};
use sonium_sim_speaker_modeling::{
    baffled_piston_mechanical_radiation_impedance, AcousticMedium, EnclosureSpec, MovingSystemSpec,
    PortSpec,
};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

type ExampleResult<T> = Result<T, Box<dyn std::error::Error>>;

const DRIVE_V_RMS: f64 = 2.83;
const SOURCE_RESISTANCE_OHM: f64 = 0.001;
const CABINET_HEIGHT_M: f64 = 0.290;
const CABINET_WIDTH_M: f64 = 0.174;
const CABINET_DEPTH_M: f64 = 0.263;
const WOOFER_Y_M: f64 = -0.056;
const TWEETER_Y_M: f64 = 0.079;
const WOOFER_SD_M2: f64 = 87.0e-4;
const TWEETER_SD_M2: f64 = 5.7e-4;
const COMPARISON_MIN_HZ: f64 = 100.0;
const COMPARISON_MAX_HZ: f64 = 8_000.0;

fn woofer_radius_m() -> f64 {
    (WOOFER_SD_M2 / std::f64::consts::PI).sqrt()
}

fn tweeter_radius_m() -> f64 {
    (TWEETER_SD_M2 / std::f64::consts::PI).sqrt()
}

/// Frequency/magnitude/phase table used for impedances and FRDs.
#[derive(Debug, Clone)]
struct ResponseTable {
    frequencies_hz: Vec<f64>,
    magnitudes: Vec<f64>,
    phases_deg: Vec<f64>,
}

impl ResponseTable {
    fn complex_at(&self, frequency_hz: f64) -> Complex64 {
        let index = self
            .frequencies_hz
            .partition_point(|&value| value <= frequency_hz);
        if index == 0 {
            return Self::point(self.magnitudes[0], self.phases_deg[0]);
        }
        if index >= self.frequencies_hz.len() {
            let last = self.frequencies_hz.len() - 1;
            return Self::point(self.magnitudes[last], self.phases_deg[last]);
        }
        let f0 = self.frequencies_hz[index - 1];
        let f1 = self.frequencies_hz[index];
        let weight = (frequency_hz.ln() - f0.ln()) / (f1.ln() - f0.ln());
        let a = Self::point(self.magnitudes[index - 1], self.phases_deg[index - 1]);
        let b = Self::point(self.magnitudes[index], self.phases_deg[index]);
        a + (b - a) * weight
    }

    fn impedance_at(&self, frequency_hz: f64) -> Complex64 {
        self.complex_at(frequency_hz)
    }

    fn pressure_at(&self, frequency_hz: f64) -> Complex64 {
        // FRD magnitudes are dB values, not linear pressure.
        let index = self
            .frequencies_hz
            .partition_point(|&value| value <= frequency_hz);
        if index == 0 {
            return Self::db_point(self.magnitudes[0], self.phases_deg[0]);
        }
        if index >= self.frequencies_hz.len() {
            let last = self.frequencies_hz.len() - 1;
            return Self::db_point(self.magnitudes[last], self.phases_deg[last]);
        }
        let f0 = self.frequencies_hz[index - 1];
        let f1 = self.frequencies_hz[index];
        let weight = (frequency_hz.ln() - f0.ln()) / (f1.ln() - f0.ln());
        let a = Self::db_point(self.magnitudes[index - 1], self.phases_deg[index - 1]);
        let b = Self::db_point(self.magnitudes[index], self.phases_deg[index]);
        a + (b - a) * weight
    }

    fn point(magnitude: f64, phase_deg: f64) -> Complex64 {
        Complex64::from_polar(magnitude, phase_deg.to_radians())
    }

    fn db_point(db: f64, phase_deg: f64) -> Complex64 {
        Complex64::from_polar(10f64.powf(db / 20.0), phase_deg.to_radians())
    }
}

fn parse_response_table(path: &Path) -> ExampleResult<ResponseTable> {
    let text = std::fs::read_to_string(path)?;
    let mut frequencies_hz = Vec::new();
    let mut magnitudes = Vec::new();
    let mut phases_deg = Vec::new();
    for line in text.lines() {
        let fields: Vec<&str> = line
            .trim()
            .trim_start_matches('*')
            .split(|c: char| c == '\t' || c == ',' || c.is_whitespace())
            .filter(|field| !field.is_empty())
            .collect();
        if fields.len() < 3 {
            continue;
        }
        let Ok(frequency) = fields[0].parse::<f64>() else {
            continue;
        };
        let Ok(magnitude) = fields[1].parse::<f64>() else {
            continue;
        };
        let Ok(phase) = fields[2].parse::<f64>() else {
            continue;
        };
        if frequency > 0.0 && magnitude.is_finite() && phase.is_finite() {
            frequencies_hz.push(frequency);
            magnitudes.push(magnitude);
            phases_deg.push(phase);
        }
    }
    if frequencies_hz.len() < 3 {
        return Err(format!("response table {} has too few numeric rows", path.display()).into());
    }
    Ok(ResponseTable {
        frequencies_hz,
        magnitudes,
        phases_deg,
    })
}

fn capacitor(frequency_hz: f64, capacitance_f: f64, esr_ohm: f64) -> Complex64 {
    Complex64::new(
        esr_ohm,
        -1.0 / (std::f64::consts::TAU * frequency_hz * capacitance_f),
    )
}

fn inductor(frequency_hz: f64, inductance_h: f64, dcr_ohm: f64) -> Complex64 {
    Complex64::new(dcr_ohm, std::f64::consts::TAU * frequency_hz * inductance_h)
}

fn parallel(first: Complex64, second: Complex64) -> Complex64 {
    first * second / (first + second)
}

fn solve_linear<const N: usize>(
    mut matrix: [[Complex64; N]; N],
    mut rhs: [Complex64; N],
) -> ExampleResult<[Complex64; N]> {
    for column in 0..N {
        let pivot = (column..N)
            .max_by(|&a, &b| {
                matrix[a][column]
                    .norm()
                    .total_cmp(&matrix[b][column].norm())
            })
            .ok_or("empty circuit matrix")?;
        if matrix[pivot][column].norm() < 1.0e-14 {
            return Err("singular crossover circuit matrix".into());
        }
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        let divisor = matrix[column][column];
        for value in &mut matrix[column] {
            *value /= divisor;
        }
        rhs[column] /= divisor;
        for row in 0..N {
            if row == column {
                continue;
            }
            let factor = matrix[row][column];
            if factor.norm() == 0.0 {
                continue;
            }
            let pivot_row = matrix[column];
            for (offset, value) in matrix[row].iter_mut().skip(column).enumerate() {
                *value -= factor * pivot_row[column + offset];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    Ok(rhs)
}

/// Exact Mechano23 crossover transfers from the public VituixCAD project.
///
/// Component values and DCR/ESR are the project values also listed in the
/// ASR review. The woofer shunt is a 27 µF series capacitor feeding a node
/// loaded in parallel by 8.2 Ω, 18 µF, and 1.5 mH. The 0.82 µF capacitor
/// bypasses both woofer series inductors.
fn crossover_transfers(
    frequency_hz: f64,
    woofer_impedance: Complex64,
    tweeter_impedance: Complex64,
) -> ExampleResult<(Complex64, Complex64)> {
    let source = Complex64::new(DRIVE_V_RMS, 0.0);
    let source_y = 1.0 / SOURCE_RESISTANCE_OHM;

    // Woofer nodes: [input, after first inductor, driver, shunt-node].
    let l1 = inductor(frequency_hz, 0.33e-3, 0.258);
    let l2 = inductor(frequency_hz, 0.56e-3, 0.51);
    let y_input_mid = 1.0 / l1 + 1.0 / Complex64::new(12.0, 0.0);
    let y_input_driver = 1.0 / capacitor(frequency_hz, 0.82e-6, 0.01);
    let y_mid_driver = 1.0 / l2;
    let y_mid_ground = 1.0 / capacitor(frequency_hz, 6.8e-6, 0.01);
    let y_driver_shunt = 1.0 / capacitor(frequency_hz, 27.0e-6, 0.01);
    let shunt_load = parallel(
        Complex64::new(8.2, 0.0),
        parallel(
            capacitor(frequency_hz, 18.0e-6, 0.01),
            inductor(frequency_hz, 1.5e-3, 0.94),
        ),
    );
    let y_shunt_ground = 1.0 / shunt_load;
    let y_driver_ground = 1.0 / woofer_impedance;
    let mut woofer_matrix = [[Complex64::new(0.0, 0.0); 4]; 4];
    let mut woofer_rhs = [Complex64::new(0.0, 0.0); 4];
    woofer_matrix[0][0] = source_y + y_input_mid + y_input_driver;
    woofer_matrix[0][1] = -y_input_mid;
    woofer_matrix[0][2] = -y_input_driver;
    woofer_matrix[1][0] = -y_input_mid;
    woofer_matrix[1][1] = y_input_mid + y_mid_driver + y_mid_ground;
    woofer_matrix[1][2] = -y_mid_driver;
    woofer_matrix[2][0] = -y_input_driver;
    woofer_matrix[2][1] = -y_mid_driver;
    woofer_matrix[2][2] = y_input_driver + y_mid_driver + y_driver_shunt + y_driver_ground;
    woofer_matrix[2][3] = -y_driver_shunt;
    woofer_matrix[3][2] = -y_driver_shunt;
    woofer_matrix[3][3] = y_driver_shunt + y_shunt_ground;
    woofer_rhs[0] = source * source_y;
    let woofer_nodes = solve_linear(woofer_matrix, woofer_rhs)?;

    // Tweeter nodes: [input, after 3.3 µF, shunt node, driver].
    let y_input_mid = 1.0 / capacitor(frequency_hz, 3.3e-6, 0.01);
    let y_mid_shunt = 1.0 / Complex64::new(15.0, 0.0);
    let y_shunt_ground = 1.0 / inductor(frequency_hz, 0.39e-3, 0.42);
    let y_shunt_driver = 1.0 / capacitor(frequency_hz, 2.0e-6, 0.01);
    let y_driver_ground = 1.0 / tweeter_impedance;
    let mut tweeter_matrix = [[Complex64::new(0.0, 0.0); 4]; 4];
    let mut tweeter_rhs = [Complex64::new(0.0, 0.0); 4];
    tweeter_matrix[0][0] = source_y + y_input_mid;
    tweeter_matrix[0][1] = -y_input_mid;
    tweeter_matrix[1][0] = -y_input_mid;
    tweeter_matrix[1][1] = y_input_mid + y_mid_shunt;
    tweeter_matrix[1][2] = -y_mid_shunt;
    tweeter_matrix[2][1] = -y_mid_shunt;
    tweeter_matrix[2][2] = y_mid_shunt + y_shunt_ground + y_shunt_driver;
    tweeter_matrix[2][3] = -y_shunt_driver;
    tweeter_matrix[3][2] = -y_shunt_driver;
    tweeter_matrix[3][3] = y_shunt_driver + y_driver_ground;
    tweeter_rhs[0] = source * source_y;
    let tweeter_nodes = solve_linear(tweeter_matrix, tweeter_rhs)?;

    Ok((woofer_nodes[2] / source, tweeter_nodes[3] / source))
}

struct DriverTransfers {
    woofer_velocity_rms: Vec<Complex64>,
    tweeter_velocity_rms: Vec<Complex64>,
    woofer_terminal_transfer: Vec<Complex64>,
    tweeter_terminal_transfer: Vec<Complex64>,
}

fn driver_transfers(
    frequencies_hz: &[f64],
    woofer_impedance: &ResponseTable,
    tweeter_impedance: &ResponseTable,
) -> ExampleResult<DriverTransfers> {
    let medium = AcousticMedium::standard_air();
    // SB13PFCR25-4 manufacturer data: Fs 44 Hz, Mms 10.5 g,
    // Cms 1.25 mm/N, Rms 1.1 kg/s, BL 5.3 T·m.
    let woofer_moving = MovingSystemSpec {
        moving_mass_kg: 10.5e-3,
        mechanical_resistance_kg_per_s: 1.1,
        stiffness_n_per_m: 1.0 / 1.25e-3,
        xmax_m: Some(4.5e-3),
    };
    let woofer_enclosure = EnclosureSpec::Vented {
        volume_m3: 7.96e-3,
        port: PortSpec {
            radius_m: 0.0175,
            length_m: 0.145,
            end_correction_m: 0.030,
            resistance_pa_s_per_m3: 0.0,
        },
        leakage_resistance_pa_s_per_m3: 0.0,
        absorption_resistance_pa_s_per_m3: 0.0,
    };
    // H2606/920000 manufacturer data: Fs 1030 Hz, Mms 0.4 g, Rms 1.3 kg/s,
    // BL 3.3 T·m. Stiffness follows directly from Fs and Mms.
    let tweeter_stiffness = 0.4e-3 * (std::f64::consts::TAU * 1030.0).powi(2);
    let tweeter_moving = MovingSystemSpec {
        moving_mass_kg: 0.4e-3,
        mechanical_resistance_kg_per_s: 1.3,
        stiffness_n_per_m: tweeter_stiffness,
        xmax_m: Some(0.2e-3),
    };

    let mut result = DriverTransfers {
        woofer_velocity_rms: Vec::with_capacity(frequencies_hz.len()),
        tweeter_velocity_rms: Vec::with_capacity(frequencies_hz.len()),
        woofer_terminal_transfer: Vec::with_capacity(frequencies_hz.len()),
        tweeter_terminal_transfer: Vec::with_capacity(frequencies_hz.len()),
    };
    for &frequency_hz in frequencies_hz {
        let woofer_z = woofer_impedance.impedance_at(frequency_hz);
        let tweeter_z = tweeter_impedance.impedance_at(frequency_hz);
        let (woofer_transfer, tweeter_transfer) =
            crossover_transfers(frequency_hz, woofer_z, tweeter_z)?;
        let woofer_voltage = Complex64::new(DRIVE_V_RMS, 0.0) * woofer_transfer;
        let tweeter_voltage = Complex64::new(DRIVE_V_RMS, 0.0) * tweeter_transfer;
        let woofer_current = woofer_voltage / woofer_z;
        let tweeter_current = tweeter_voltage / tweeter_z;

        let woofer_mechanical = woofer_moving.try_mechanical_impedance(frequency_hz)?
            + woofer_enclosure.acoustic_impedance(frequency_hz, medium)? * WOOFER_SD_M2.powi(2)
            + baffled_piston_mechanical_radiation_impedance(
                woofer_radius_m(),
                frequency_hz,
                medium,
            )?;
        let tweeter_mechanical = tweeter_moving.try_mechanical_impedance(frequency_hz)?
            + baffled_piston_mechanical_radiation_impedance(
                tweeter_radius_m(),
                frequency_hz,
                medium,
            )?;
        result
            .woofer_velocity_rms
            .push(5.3 * woofer_current / woofer_mechanical);
        result
            .tweeter_velocity_rms
            .push(3.3 * tweeter_current / tweeter_mechanical);
        result.woofer_terminal_transfer.push(woofer_transfer);
        result.tweeter_terminal_transfer.push(tweeter_transfer);
    }
    Ok(result)
}

fn transfer_points(frequencies_hz: &[f64], velocities_rms: &[Complex64]) -> Vec<TransferPoint> {
    frequencies_hz
        .iter()
        .zip(velocities_rms)
        .map(|(&freq, &velocity_rms)| {
            // Circuit/lumped phasors are RMS with e^{+jωt}; the BEM input
            // is a peak phasor with e^{-iωt}.
            let bem_peak = velocity_rms.conj() * std::f64::consts::SQRT_2;
            TransferPoint {
                freq,
                magnitude: bem_peak.norm(),
                phase_deg: bem_peak.arg().to_degrees(),
            }
        })
        .collect()
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(default)
}

fn mechano_model(transfers: &DriverTransfers) -> LoudspeakerModel {
    let freq_end = env_f64("MECHANO23_FREQ_END_HZ", 8_000.0);
    let simulation = SimulationConfig {
        freq_start: env_f64("MECHANO23_FREQ_START_HZ", 50.0),
        freq_end,
        freq_points: env_f64("MECHANO23_FREQ_POINTS", 21.0) as usize,
        speed_of_sound: 343.0,
        air_density: 1.204,
        max_mesh_frequency: Some(env_f64("MECHANO23_MAX_MESH_HZ", freq_end)),
    };
    let frequencies = simulation.frequencies();
    LoudspeakerModel {
        cabinet: CabinetDimensions {
            height: CABINET_HEIGHT_M,
            width: CABINET_WIDTH_M,
            depth: CABINET_DEPTH_M,
        },
        woofer: DriverConfig {
            radius: woofer_radius_m(),
            x_offset: 0.0,
            y_offset: WOOFER_Y_M,
            piston_velocity: 0.0,
            transfer_function: Some(transfer_points(
                &frequencies,
                &transfers.woofer_velocity_rms,
            )),
            driver_model: None,
            cone: None,
        },
        tweeter: DriverConfig {
            radius: tweeter_radius_m(),
            x_offset: 0.0,
            y_offset: TWEETER_Y_M,
            piston_velocity: 0.0,
            transfer_function: Some(transfer_points(
                &frequencies,
                &transfers.tweeter_velocity_rms,
            )),
            driver_model: None,
            cone: None,
        },
        additional_drivers: Vec::new(),
        product_design: None,
        separator: SeparatorConfig {
            y_position: 0.012,
            thickness: 0.018,
        },
        // Short integrated horn: 104 mm faceplate, 26 mm effective dome.
        // The 10 mm depth is an explicit approximation from the driver
        // drawing; the BEM therefore should not be treated as an exact
        // high-frequency horn model.
        waveguide: Some(WaveguideConfig {
            horizontal_spline: vec![
                [0.0, 0.052],
                [0.35, 0.040],
                [0.70, 0.024],
                [1.0, tweeter_radius_m()],
            ],
            vertical_spline: vec![
                [0.0, 0.052],
                [0.35, 0.040],
                [0.70, 0.024],
                [1.0, tweeter_radius_m()],
            ],
            depth: 0.010,
        }),
        materials: Default::default(),
        simulation,
    }
}

/// One horizontal/vertical SPL plane keyed by its printed angle label.
#[derive(Debug, Clone)]
struct SplPlane {
    frequencies_hz: Vec<f64>,
    curves: BTreeMap<String, Vec<f64>>,
}

fn parse_spl_plane(path: &Path) -> ExampleResult<SplPlane> {
    let text = std::fs::read_to_string(path)?;
    let mut lines = text.lines();
    let _title = lines.next().ok_or("missing SPL plane title")?;
    let header = lines.next().ok_or("missing SPL plane header")?;
    let labels: Vec<String> = header
        .split('\t')
        .map(|field| field.trim().trim_matches('"').to_owned())
        .filter(|field| !field.is_empty())
        .collect();
    let mut frequencies_hz = Vec::new();
    let mut curves: BTreeMap<String, Vec<f64>> = labels
        .iter()
        .map(|label| (label.clone(), Vec::new()))
        .collect();
    for line in lines {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < labels.len() * 2 {
            continue;
        }
        let Ok(frequency) = fields[0].trim().parse::<f64>() else {
            continue;
        };
        let mut row = Vec::with_capacity(labels.len());
        let mut valid = true;
        for curve in 0..labels.len() {
            match fields[curve * 2 + 1].trim().parse::<f64>() {
                Ok(value) => row.push(value),
                Err(_) => {
                    valid = false;
                    break;
                }
            }
        }
        if valid {
            frequencies_hz.push(frequency);
            for (label, value) in labels.iter().zip(row) {
                curves
                    .get_mut(label)
                    .expect("curve created from header")
                    .push(value);
            }
        }
    }
    if frequencies_hz.len() < 3 || curves.len() < 36 {
        return Err(format!("SPL plane {} is incomplete", path.display()).into());
    }
    Ok(SplPlane {
        frequencies_hz,
        curves,
    })
}

fn interpolate_db(frequencies_hz: &[f64], values: &[f64], target_hz: f64) -> f64 {
    if target_hz <= frequencies_hz[0] {
        return values[0];
    }
    if target_hz >= frequencies_hz[frequencies_hz.len() - 1] {
        return values[values.len() - 1];
    }
    let upper = frequencies_hz.partition_point(|&value| value <= target_hz);
    let f0 = frequencies_hz[upper - 1];
    let f1 = frequencies_hz[upper];
    let weight = (target_hz.ln() - f0.ln()) / (f1.ln() - f0.ln());
    values[upper - 1] + (values[upper] - values[upper - 1]) * weight
}

fn signed_angle_labels() -> Vec<i32> {
    let mut angles: Vec<i32> = (-170..=170).step_by(10).collect();
    angles.push(180);
    angles
}

fn angle_label(angle: i32) -> String {
    if angle == 0 {
        "On-Axis".to_owned()
    } else {
        format!("{angle}°")
    }
}

fn plane_rows_at(plane: &SplPlane, frequencies_hz: &[f64]) -> Vec<Vec<f64>> {
    signed_angle_labels()
        .iter()
        .map(|&angle| {
            let curve = plane
                .curves
                .get(&angle_label(angle))
                .unwrap_or_else(|| panic!("missing measured angle {angle}"));
            frequencies_hz
                .iter()
                .map(|&frequency| interpolate_db(&plane.frequencies_hz, curve, frequency))
                .collect()
        })
        .collect()
}

fn row_for_angle(angle: i32) -> usize {
    signed_angle_labels()
        .iter()
        .position(|&value| value == angle)
        .unwrap_or_else(|| panic!("angle {angle} is not on the 10 degree grid"))
}

fn rows_for_angles(angles: &[i32]) -> Vec<usize> {
    angles.iter().map(|&angle| row_for_angle(angle)).collect()
}

#[derive(Debug, Clone)]
struct CeaCurves {
    on_axis: Vec<f64>,
    listening_window: Vec<f64>,
    early_reflections: Vec<f64>,
    sound_power: Vec<f64>,
    estimated_in_room: Vec<f64>,
}

fn aggregate_cea(
    frequencies_hz: &[f64],
    horizontal_rows: &[Vec<f64>],
    vertical_rows: &[Vec<f64>],
) -> ExampleResult<CeaCurves> {
    let mut full = horizontal_rows.to_vec();
    full.extend(vertical_rows.iter().cloned());
    let row_h = |angle: i32| row_for_angle(angle);
    let row_v = |angle: i32| horizontal_rows.len() + row_for_angle(angle);
    let idx = vec![
        vec![
            row_h(0),
            row_v(-10),
            row_v(10),
            row_h(-10),
            row_h(10),
            row_h(-20),
            row_h(20),
            row_h(-30),
            row_h(30),
        ],
        vec![row_h(0)],
        rows_for_angles(&[-20, -30, -40])
            .iter()
            .map(|&row| horizontal_rows.len() + row)
            .collect(),
        rows_for_angles(&[40, 50, 60])
            .iter()
            .map(|&row| horizontal_rows.len() + row)
            .collect(),
        rows_for_angles(&[0, -10, 10, -20, 20, -30, 30]),
        rows_for_angles(&[-40, 40, -50, 50, -60, 60, -70, 70, -80, 80]),
        rows_for_angles(&[-90, 90, 180]),
        {
            let mut rows: Vec<usize> = (0..horizontal_rows.len()).collect();
            rows.extend((0..vertical_rows.len()).filter_map(|row| {
                let angle = signed_angle_labels()[row];
                if angle == 0 || angle == 180 {
                    None
                } else {
                    Some(horizontal_rows.len() + row)
                }
            }));
            rows
        },
    ];
    let mut weights = vec![0.0; full.len()];
    let assign_weights = |start: usize, rows: &[Vec<f64>], weights: &mut Vec<f64>| {
        for (row, &angle) in signed_angle_labels().iter().enumerate() {
            let polar = angle.unsigned_abs() as f64;
            let ring = (polar - 5.0).max(0.0).to_radians().cos()
                - (polar + 5.0).min(180.0).to_radians().cos();
            let samples = if polar == 0.0 || polar == 180.0 {
                1.0
            } else {
                4.0
            };
            weights[start + row] = ring / samples;
            let _ = rows;
        }
    };
    assign_weights(0, horizontal_rows, &mut weights);
    let horizontal_len = horizontal_rows.len();
    assign_weights(horizontal_len, vertical_rows, &mut weights);
    // Poles receive no separate spherical weight in the combined set.
    weights[row_h(0)] = 0.0;
    weights[row_h(180)] = 0.0;
    weights[row_v(0)] = 0.0;
    weights[row_v(180)] = 0.0;

    let aggregated = try_cea2034(&full, &idx, &weights)
        .map_err(|error| format!("CEA-2034 aggregation failed: {error}"))?;
    let on_axis_row = row_h(0);
    Ok(CeaCurves {
        on_axis: frequencies_hz
            .iter()
            .enumerate()
            .map(|(index, _)| horizontal_rows[on_axis_row][index])
            .collect(),
        listening_window: aggregated[0].clone(),
        early_reflections: aggregated[1].clone(),
        sound_power: aggregated[7].clone(),
        estimated_in_room: aggregated[8].clone(),
    })
}

fn load_vituix_response(
    vituix_dir: &Path,
    driver: &str,
    plane: char,
    signed_angle: i32,
) -> ExampleResult<ResponseTable> {
    let (directory, prefix, extension) = match driver {
        "woofer" => (
            vituix_dir.join("m23_sba_pfcr13/Hmrg100_300_90%"),
            "m23_SBApfcr13",
            "txt",
        ),
        "tweeter" => (vituix_dir.join("m23_scspk/H"), "scspk", "frd"),
        _ => return Err(format!("unknown Vituix driver {driver}").into()),
    };
    let file_angle = match plane {
        'H' => signed_angle.abs(),
        'V' => signed_angle.rem_euclid(360),
        _ => return Err(format!("unknown response plane {plane}").into()),
    };
    let plane_name = if plane == 'H' { "hor" } else { "ver" };
    let path = directory.join(format!("{prefix}_{plane_name}_{file_angle:03}.{extension}"));
    parse_response_table(&path).map_err(|error| {
        format!(
            "cannot load Vituix {} {plane} {signed_angle} response {}: {error}",
            driver,
            path.display()
        )
        .into()
    })
}

fn vituix_curves(
    frequencies_hz: &[f64],
    transfers: &DriverTransfers,
    vituix_dir: &Path,
) -> ExampleResult<CeaCurves> {
    let angles = signed_angle_labels();
    let build_plane = |plane: char| -> ExampleResult<Vec<Vec<f64>>> {
        let mut rows = Vec::with_capacity(angles.len());
        for (row, &angle) in angles.iter().enumerate() {
            let woofer = load_vituix_response(vituix_dir, "woofer", plane, angle)?;
            let tweeter = load_vituix_response(vituix_dir, "tweeter", plane, angle)?;
            let mut curve = Vec::with_capacity(frequencies_hz.len());
            for (index, &frequency) in frequencies_hz.iter().enumerate() {
                let pressure = woofer.pressure_at(frequency)
                    * transfers.woofer_terminal_transfer[index]
                    + tweeter.pressure_at(frequency) * transfers.tweeter_terminal_transfer[index];
                curve.push(20.0 * pressure.norm().max(1.0e-12).log10());
            }
            let _ = row;
            rows.push(curve);
        }
        Ok(rows)
    };
    let horizontal = build_plane('H')?;
    let vertical = build_plane('V')?;
    aggregate_cea(frequencies_hz, &horizontal, &vertical)
}

#[derive(Debug, Clone)]
struct CurveComparison {
    name: String,
    sensitivity_offset_db: f64,
    shape_mae_db: f64,
    shape_rmse_db: f64,
    max_shape_deviation_db: f64,
    raw_mae_db: f64,
    points: usize,
}

fn compare_curves(
    name: &str,
    frequencies_hz: &[f64],
    candidate: &[f64],
    reference: &[f64],
) -> CurveComparison {
    let pairs: Vec<(f64, f64)> = frequencies_hz
        .iter()
        .zip(candidate.iter().zip(reference.iter()))
        .filter(|(frequency, _)| (COMPARISON_MIN_HZ..=COMPARISON_MAX_HZ).contains(&**frequency))
        .map(|(_, (&candidate_value, &reference_value))| (candidate_value, reference_value))
        .collect();
    let offset = pairs
        .iter()
        .map(|(candidate_value, reference_value)| candidate_value - reference_value)
        .sum::<f64>()
        / pairs.len().max(1) as f64;
    let mut shape_absolute = 0.0;
    let mut shape_squared = 0.0;
    let mut raw_absolute = 0.0;
    let mut max_deviation = 0.0f64;
    for (candidate_value, reference_value) in &pairs {
        let raw = candidate_value - reference_value;
        let shape = raw - offset;
        shape_absolute += shape.abs();
        shape_squared += shape * shape;
        raw_absolute += raw.abs();
        max_deviation = max_deviation.max(shape.abs());
    }
    let count = pairs.len().max(1) as f64;
    CurveComparison {
        name: name.to_owned(),
        sensitivity_offset_db: offset,
        shape_mae_db: shape_absolute / count,
        shape_rmse_db: (shape_squared / count).sqrt(),
        max_shape_deviation_db: max_deviation,
        raw_mae_db: raw_absolute / count,
        points: pairs.len(),
    }
}

fn compare_sets(
    left_name: &str,
    left: &CeaCurves,
    right_name: &str,
    right: &CeaCurves,
    frequencies_hz: &[f64],
) -> Vec<CurveComparison> {
    [
        ("On Axis", &left.on_axis, &right.on_axis),
        (
            "Listening Window",
            &left.listening_window,
            &right.listening_window,
        ),
        (
            "Early Reflections",
            &left.early_reflections,
            &right.early_reflections,
        ),
        ("Sound Power", &left.sound_power, &right.sound_power),
        (
            "Estimated In-Room",
            &left.estimated_in_room,
            &right.estimated_in_room,
        ),
    ]
    .into_iter()
    .map(|(curve, left_curve, right_curve)| {
        compare_curves(
            &format!("{curve} ({left_name} vs {right_name})"),
            frequencies_hz,
            left_curve,
            right_curve,
        )
    })
    .collect()
}

fn default_vituix_dir() -> PathBuf {
    std::env::var_os("SONIUM_MECHANO23_VXP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("tmp/mechano_vituix/Mechano-23"))
}

fn write_curves_csv(
    path: &Path,
    frequencies_hz: &[f64],
    sonium: &CeaCurves,
    vituix: &CeaCurves,
    asr: &CeaCurves,
) -> ExampleResult<()> {
    let mut csv = String::from(
        "frequency_hz,sonium_on_axis_db,sonium_listening_window_db,sonium_early_reflections_db,sonium_sound_power_db,sonium_estimated_in_room_db,vituix_on_axis_db,vituix_listening_window_db,vituix_early_reflections_db,vituix_sound_power_db,vituix_estimated_in_room_db,asr_on_axis_db,asr_listening_window_db,asr_early_reflections_db,asr_sound_power_db,asr_estimated_in_room_db\n",
    );
    for (index, &frequency) in frequencies_hz.iter().enumerate() {
        let values = [
            sonium.on_axis[index],
            sonium.listening_window[index],
            sonium.early_reflections[index],
            sonium.sound_power[index],
            sonium.estimated_in_room[index],
            vituix.on_axis[index],
            vituix.listening_window[index],
            vituix.early_reflections[index],
            vituix.sound_power[index],
            vituix.estimated_in_room[index],
            asr.on_axis[index],
            asr.listening_window[index],
            asr.early_reflections[index],
            asr.sound_power[index],
            asr.estimated_in_room[index],
        ];
        csv.push_str(&format!("{frequency:.4}"));
        for value in values {
            csv.push_str(&format!(",{value:.4}"));
        }
        csv.push('\n');
    }
    std::fs::write(path, csv)?;
    Ok(())
}

fn write_summary(
    path: &Path,
    comparisons: &[CurveComparison],
    crossover_points: &[(f64, f64, f64)],
    comparison_max_hz: f64,
) -> ExampleResult<()> {
    let mut text = format!(
        "Mechano23 Tier B comparison (100 Hz-{comparison_max_hz:.0} Hz; one broadband offset removed per curve)\n"
    );
    text.push_str("crossover terminal-voltage magnitudes:\n");
    for (frequency, woofer, tweeter) in crossover_points {
        text.push_str(&format!(
            "  {frequency:7.1} Hz  woofer {woofer:7.3}  tweeter {tweeter:7.3}\n"
        ));
    }
    for comparison in comparisons {
        text.push_str(&format!(
            "{}: offset {:+.2} dB, shape MAE {:.2} dB, RMSE {:.2} dB, max {:.2} dB, raw MAE {:.2} dB ({} points)\n",
            comparison.name,
            comparison.sensitivity_offset_db,
            comparison.shape_mae_db,
            comparison.shape_rmse_db,
            comparison.max_shape_deviation_db,
            comparison.raw_mae_db,
            comparison.points,
        ));
    }
    std::fs::write(path, &text)?;
    print!("{text}");
    Ok(())
}

fn main() -> ExampleResult<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out_dir = args
        .first()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/mechano23-tierb"));
    let spinorama_dir = args.get(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("sonium/crates/sonium-sim-speaker/fixtures/tierb/mechano23")
    });
    let vituix_dir = args
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(default_vituix_dir);
    std::fs::create_dir_all(&out_dir)?;

    let woofer_impedance =
        parse_response_table(&vituix_dir.join("m23_sba_pfcr13/m23_zma_SBApfcr13_incabiinet.txt"))?;
    let tweeter_impedance =
        parse_response_table(&vituix_dir.join("m23_scspk/m23_scspk_tweeter_zma.zma"))?;

    let simulation_probe = mechano_model(&DriverTransfers {
        woofer_velocity_rms: Vec::new(),
        tweeter_velocity_rms: Vec::new(),
        woofer_terminal_transfer: Vec::new(),
        tweeter_terminal_transfer: Vec::new(),
    });
    let frequencies = simulation_probe.simulation.frequencies();
    let transfers = driver_transfers(&frequencies, &woofer_impedance, &tweeter_impedance)?;
    let model = mechano_model(&transfers);
    model.validate()?;

    let horizontal = parse_spl_plane(&spinorama_dir.join("SPL Horizontal.txt"))?;
    let vertical = parse_spl_plane(&spinorama_dir.join("SPL Vertical.txt"))?;
    let asr = aggregate_cea(
        &frequencies,
        &plane_rows_at(&horizontal, &frequencies),
        &plane_rows_at(&vertical, &frequencies),
    )?;
    let vituix = vituix_curves(&frequencies, &transfers, &vituix_dir)?;

    // Evaluate around the tweeter, matching the design/listening axis.
    let eval_points =
        cea2034_output::generate_eval_points([0.0, TWEETER_Y_M, model.baffle_center()[2]]);
    eprintln!(
        "Running Mechano23 Sonium BEM: {} frequencies x {} points...",
        frequencies.len(),
        eval_points.len()
    );
    // Measured on this cabinet (Oct 2026): the default restart-40 /
    // recycle-20 GCRO-DR stalls above ~1 kHz on the λ/8 mesh (12,000-iteration
    // cap, true residuals 1e-3..1e-1), while restart 200 / recycle 50
    // converges at every probed frequency in 50..400 iterations. The larger
    // subspace is therefore this example's production default; the
    // MECHANO23_* variables below still override it for probes.
    let mut solver_config = SolverConfig {
        restart: 200,
        recycle_subspace_size: 50,
        ..SolverConfig::default()
    };
    if std::env::var_os("MECHANO23_DENSE").is_some_and(|value| value == "1") {
        // Force the dense operator path: no ACA far-field approximation.
        solver_config.hmatrix_threshold = usize::MAX;
    }
    solver_config.tolerance = env_f64("MECHANO23_TOLERANCE", solver_config.tolerance);
    solver_config.hmatrix_aca_tolerance = env_f64(
        "MECHANO23_ACA_TOLERANCE",
        solver_config.hmatrix_aca_tolerance,
    );
    solver_config.max_iterations = env_f64(
        "MECHANO23_MAX_ITERATIONS",
        solver_config.max_iterations as f64,
    ) as usize;
    solver_config.restart = env_f64("MECHANO23_RESTART", solver_config.restart as f64) as usize;
    solver_config.recycle_subspace_size = env_f64(
        "MECHANO23_RECYCLE",
        solver_config.recycle_subspace_size as f64,
    ) as usize;
    if let Some(lanes) = std::env::var("MECHANO23_MAX_LANES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&lanes| lanes > 0)
    {
        solver_config.max_frequency_lanes = Some(lanes);
    }
    #[cfg(feature = "cuda")]
    let result = if std::env::var_os("MECHANO23_CUDA").is_some_and(|value| value == "1") {
        // GPU probe path: dense operator assembled and solved on CUDA,
        // accepted only through the crate's own residual/field gates.
        let (result, report) = sonium_sim_speaker::simulation::run_simulation_cuda_with_report(
            &model,
            &eval_points,
            &solver_config,
        )?;
        eprintln!("GPU execution report: {report:?}");
        result
    } else {
        run_simulation_with_config(&model, &eval_points, &solver_config)?
    };
    #[cfg(not(feature = "cuda"))]
    let result = run_simulation_with_config(&model, &eval_points, &solver_config)?;
    let predicted_metrics = cea2034_output::compute_spinorama_metrics(&result)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    let sonium = CeaCurves {
        on_axis: predicted_metrics.on_axis.clone(),
        listening_window: predicted_metrics.listening_window.clone(),
        early_reflections: predicted_metrics.early_reflections.clone(),
        sound_power: predicted_metrics.sound_power.clone(),
        estimated_in_room: predicted_metrics.pir.clone(),
    };

    write_curves_csv(
        &out_dir.join("mechano23_curves.csv"),
        &frequencies,
        &sonium,
        &vituix,
        &asr,
    )?;
    let mut comparisons = Vec::new();
    comparisons.extend(compare_sets("Sonium", &sonium, "ASR", &asr, &frequencies));
    comparisons.extend(compare_sets(
        "Sonium",
        &sonium,
        "Vituix",
        &vituix,
        &frequencies,
    ));
    comparisons.extend(compare_sets("Vituix", &vituix, "ASR", &asr, &frequencies));
    let crossover_points: Vec<(f64, f64, f64)> = frequencies
        .iter()
        .enumerate()
        .filter(|(_, frequency)| (100.0..=8_000.0).contains(*frequency))
        .step_by(4)
        .map(|(index, &frequency)| {
            (
                frequency,
                transfers.woofer_terminal_transfer[index].norm(),
                transfers.tweeter_terminal_transfer[index].norm(),
            )
        })
        .collect();
    let comparison_max_hz = frequencies
        .iter()
        .copied()
        .filter(|frequency| *frequency <= COMPARISON_MAX_HZ)
        .fold(COMPARISON_MIN_HZ, f64::max);
    write_summary(
        &out_dir.join("summary.txt"),
        &comparisons,
        &crossover_points,
        comparison_max_hz,
    )?;
    println!("Wrote {}", out_dir.join("mechano23_curves.csv").display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crossover_has_lowpass_and_highpass_asymptotes() {
        let woofer = ResponseTable {
            frequencies_hz: vec![10.0, 100.0, 1_000.0, 10_000.0],
            magnitudes: vec![4.0, 4.0, 5.0, 8.0],
            phases_deg: vec![0.0, 20.0, 40.0, 60.0],
        };
        let tweeter = ResponseTable {
            frequencies_hz: vec![10.0, 100.0, 1_000.0, 10_000.0],
            magnitudes: vec![6.0, 6.0, 6.0, 8.0],
            phases_deg: vec![0.0, 20.0, 40.0, 60.0],
        };
        let (low_woofer, low_tweeter) = crossover_transfers(
            100.0,
            woofer.impedance_at(100.0),
            tweeter.impedance_at(100.0),
        )
        .unwrap();
        let (high_woofer, high_tweeter) = crossover_transfers(
            8_000.0,
            woofer.impedance_at(8_000.0),
            tweeter.impedance_at(8_000.0),
        )
        .unwrap();
        assert!(low_woofer.norm() > high_woofer.norm());
        assert!(high_tweeter.norm() > low_tweeter.norm());
    }
}
