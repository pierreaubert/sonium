//! End-to-end port tone: coupled unsteady Navier-Stokes snapshots through
//! the 2-D permeable FW-H contour.
//!
//! Large-scale validation T1: runs the real M1 coupled box/driver/port
//! lockstep ([`run_coupled_port`]) on the coarse 2-D slot fixture at a
//! 0.01 V tiny drive (duct Re ≈ 0.6, linear regime), then feeds the
//! retained Taylor-Hood field snapshots to [`analyze_port_noise`] with the
//! permeable-FW-H contour path. Gates:
//!
//! 1. the drive-frequency tone dominates the coherent sub/super-harmonic
//!    bins (tone presence in a real unsteady field),
//! 2. the contour field spreads cylindrically (`1/√r`, Hankel far field)
//!    across three same-direction observers,
//! 3. every spectrum point records the forced permeable-FW-H method.
//!
//! The record analyzed is the last four of ten drive periods at 64 samples
//! per period: the drive, its half, and its tenth multiple all complete
//! integer cycles (bins 2, 4, 40 of 256), so the single-frequency DFTs are
//! leakage-free. The `f_b` free transient sits coherently in the `2f` bin
//! and is reported, not gated.

use std::f64::consts::{LN_2, PI, TAU};
use std::time::Instant;

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::physics::coupled::lumped_circuit::VoiceCoilParams;
use sonium_fem::physics::fluid::stokes::navier_stokes::NavierStokesConfig;
use sonium_fem::physics::fluid::stokes::time_stepping::BdfConfig;
use sonium_fem::spaces::{H1Space, VectorH1Space};
use sonium_sim_speaker_modeling::port_flow::box_driver::BoxDriverModel;
use sonium_sim_speaker_modeling::port_flow::coupled::{CoupledPortConfig, run_coupled_port};
use sonium_sim_speaker_modeling::port_flow::geometry::{SlotPortGeometry, build_slot_port_mesh};
use sonium_sim_speaker_modeling::{
    AcousticMedium, EnclosureSpec, PortNoiseMethod, PortNoiseObserver, PortNoiseSpec, PortSpec,
    analyze_port_noise,
};

const BOX_VOLUME_M3: f64 = 0.005;
const DUCT_LENGTH_M: f64 = 0.1;
const DUCT_HEIGHT_M: f64 = 0.02;
const DRIVE_VOLTS: f64 = 0.01;
const SAMPLES_PER_PERIOD: usize = 64;
const PERIOD_COUNT: usize = 10;
const ANALYZED_PERIODS: usize = 4;

struct QaCheck {
    method: &'static str,
    name: String,
    metric: f64,
    maximum: f64,
    invert: bool,
}

impl QaCheck {
    fn below(method: &'static str, name: String, metric: f64, maximum: f64) -> Self {
        Self {
            method,
            name,
            metric,
            maximum,
            invert: false,
        }
    }

    fn above(method: &'static str, name: String, metric: f64, minimum: f64) -> Self {
        Self {
            method,
            name,
            metric,
            maximum: minimum,
            invert: true,
        }
    }

    fn passed(&self) -> bool {
        if self.invert {
            self.metric >= self.maximum
        } else {
            self.metric <= self.maximum
        }
    }
}

fn medium() -> AcousticMedium {
    AcousticMedium {
        density_kg_per_m3: 1.2,
        sound_speed_m_per_s: 343.0,
        dynamic_viscosity_pa_s: 1.85e-5,
        thermal_diffusivity_m2_per_s: 2.2e-5,
    }
}

fn drive_frequency_hz() -> f64 {
    let enclosure = EnclosureSpec::Vented {
        volume_m3: BOX_VOLUME_M3,
        port: PortSpec {
            radius_m: (DUCT_HEIGHT_M / PI).sqrt(),
            length_m: DUCT_LENGTH_M,
            end_correction_m: 4.0 * LN_2 / PI * DUCT_HEIGHT_M,
            resistance_pa_s_per_m3: 0.0,
        },
        leakage_resistance_pa_s_per_m3: 0.0,
        absorption_resistance_pa_s_per_m3: 0.0,
    };
    enclosure
        .tuning_frequency_hz(medium())
        .expect("lumped tuning")
        .expect("vented tuning")
        / 2.0
}

fn main() {
    let started = Instant::now();
    let mut checks = Vec::new();

    let mesh = build_slot_port_mesh(&SlotPortGeometry {
        duct_length_m: DUCT_LENGTH_M,
        duct_height_m: DUCT_HEIGHT_M,
        chamber_length_m: 0.05,
        chamber_height_m: 0.08,
        mesh_size_m: 0.01,
    })
    .expect("coarse slot mesh");
    let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
    let p_space = H1Space::new(PolynomialDegree::P1);

    let f_drive = drive_frequency_hz();
    let dt = 1.0 / (f_drive * SAMPLES_PER_PERIOD as f64);
    let num_steps = PERIOD_COUNT * SAMPLES_PER_PERIOD;
    let mut bdf = BdfConfig::bdf2(dt, num_steps);
    bdf.output_interval = 1;
    let config = CoupledPortConfig {
        driver: BoxDriverModel {
            mms_kg: 0.02,
            rms_kg_per_s: 1.0,
            kms_n_per_m: 4000.0,
            voice_coil: VoiceCoilParams::linear(6.0, 0.5e-3, 10.0),
            compliance_force_fn: None,
            compliance_potential_fn: None,
            sd_m2: 0.01,
            box_volume_m3: Some(BOX_VOLUME_M3),
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
        bdf,
        voltage_fn: Box::new(move |t| DRIVE_VOLTS * (TAU * f_drive * t).sin()),
        under_relaxation: 1.0,
        exterior_pressure_pa: 0.0,
    };
    let flowed = Instant::now();
    let result = run_coupled_port(&mesh, &v_space, &p_space, &config).expect("coupled port run");
    println!(
        "coupled run: {} steps in {:.1}s, {} snapshots",
        result.steps_completed,
        flowed.elapsed().as_secs_f64(),
        result.field_snapshots.len()
    );
    checks.push(QaCheck::below(
        "M1",
        "coupled steps completed".to_string(),
        (num_steps - result.steps_completed) as f64,
        0.0,
    ));

    // Last four periods: integer drive cycles, leakage-free DFT bins.
    let analyzed = ANALYZED_PERIODS * SAMPLES_PER_PERIOD;
    let snapshots = &result.field_snapshots[result.field_snapshots.len() - analyzed..];
    let frequencies_hz = vec![0.5 * f_drive, f_drive, 2.0 * f_drive, 10.0 * f_drive];
    let spec = PortNoiseSpec {
        method: PortNoiseMethod::PermeableFwh,
        frequencies_hz,
        observers: [4.0, 9.0, 16.0]
            .map(|range| PortNoiseObserver {
                position_m: [range, 0.0, 0.0],
            })
            .to_vec(),
        characteristic_source_length_m: 0.1,
        ..PortNoiseSpec::default()
    };
    let noise = analyze_port_noise(
        &mesh,
        &H1Space::new(PolynomialDegree::P2),
        &H1Space::new(PolynomialDegree::P1),
        snapshots,
        medium(),
        &spec,
    )
    .expect("contour FW-H analysis");

    let near = &noise.observers[0];
    for point in &near.spectrum {
        println!(
            "r=4m f={:7.1} Hz method={:?} p_rms={:.4e} Pa spl={:.1} dB",
            point.frequency_hz, point.method, point.pressure_rms_pa, point.spl_db
        );
    }
    let tone = near.spectrum[1].pressure_rms_pa;
    // Coherent sampling holds leakage-free bins ~1e4 above the floor;
    // 100 catches a broken DFT or wrong-frequency regression with margin.
    checks.push(QaCheck::above(
        "FW-H",
        "tone over subharmonic f/2".to_string(),
        tone / near.spectrum[0].pressure_rms_pa.max(f64::MIN_POSITIVE),
        100.0,
    ));
    checks.push(QaCheck::above(
        "FW-H",
        "tone over 10f bin".to_string(),
        tone / near.spectrum[3].pressure_rms_pa.max(f64::MIN_POSITIVE),
        100.0,
    ));

    // Cylindrical spreading: same-direction observers, Hankel far field.
    let level = |observer: usize| noise.observers[observer].spectrum[1].pressure_rms_pa;
    let ratio_9_4 = level(1) / level(0);
    let ratio_16_9 = level(2) / level(1);
    println!(
        "spreading |p(9)|/|p(4)|={ratio_9_4:.4} (1/sqrt: 0.6667) |p(16)|/|p(9)|={ratio_16_9:.4} (0.75)"
    );
    checks.push(QaCheck::below(
        "FW-H",
        "cylindrical spreading 4m->9m".to_string(),
        (ratio_9_4 - 2.0 / 3.0).abs() / (2.0 / 3.0),
        5e-2,
    ));
    checks.push(QaCheck::below(
        "FW-H",
        "cylindrical spreading 9m->16m".to_string(),
        (ratio_16_9 - 0.75).abs() / 0.75,
        5e-2,
    ));
    checks.push(QaCheck::below(
        "FW-H",
        "forced contour method recorded".to_string(),
        noise
            .observers
            .iter()
            .flat_map(|observer| observer.spectrum.iter())
            .filter(|point| point.method != PortNoiseMethod::PermeableFwh)
            .count() as f64,
        0.0,
    ));

    println!("Port contour-tone validation (f_drive={f_drive:.2} Hz)");
    println!(
        "{:<12} {:<48} {:>12} {:>12}  result",
        "method", "check", "metric", "limit"
    );
    for result in &checks {
        println!(
            "{:<12} {:<48} {:>12.4e} {:>12.4e}  {}",
            result.method,
            result.name,
            result.metric,
            result.maximum,
            if result.passed() { "pass" } else { "fail" }
        );
    }
    println!("total {:.1}s", started.elapsed().as_secs_f64());
    let failures: Vec<_> = checks.iter().filter(|result| !result.passed()).collect();
    if failures.is_empty() {
        println!("SUCCESS");
    } else {
        eprintln!("FAILED: {} of {} QA checks", failures.len(), checks.len());
        std::process::exit(1);
    }
}
