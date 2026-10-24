//! Same-input identity of the PWM step-mean voltage calculation
//! against the reference `PwmVoltageSource.mean_voltages`.
//!
//! Both sides are the raw (uncompensated) asymmetric regular-
//! sampled sine-triangle modulator; the exact interval means of
//! the chopped pole voltages must agree to round-off over
//! commensurate and awkward step intervals alike.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::electromagnetics::rotating_machine::{
    VoltageDrivePoint, VoltageWaveform, step_mean_voltages_v,
};

#[test]
fn pwm_step_means_match_reference_modulator() {
    let data: Value =
        serde_json::from_str(include_str!("fixtures/motor_oracle/pwm_crosscheck.json"))
            .expect("pwm fixture parses");
    let meta = &data["meta"];
    let pole_pairs = meta["pole_pairs"].as_u64().unwrap() as usize;
    let mut max_error_v = 0.0f64;
    for case in data["cases"].as_array().unwrap() {
        let point = VoltageDrivePoint {
            v_phase_peak_v: meta["v_phase_peak_v"].as_f64().unwrap(),
            v_delta_deg: meta["v_delta_deg"].as_f64().unwrap(),
            daxis_deg: meta["daxis_deg"].as_f64().unwrap(),
            rpm: 15_000.0,
            r_phase_ohm: 0.01,
            i_config_rms_a: 60.0,
            waveform: VoltageWaveform::Pwm {
                v_bus_v: case["v_bus_v"].as_f64().unwrap(),
                carriers_per_period: case["carriers"].as_u64().unwrap() as usize,
            },
        };
        for step in case["steps"].as_array().unwrap() {
            let got = step_mean_voltages_v(
                &point,
                pole_pairs,
                step["from_deg"].as_f64().unwrap(),
                step["to_deg"].as_f64().unwrap(),
            );
            for (got, expected) in got.iter().zip(step["mean_v"].as_array().unwrap()) {
                max_error_v = max_error_v.max((got - expected.as_f64().unwrap()).abs());
            }
        }
    }
    eprintln!("pwm oracle: max |dV| = {max_error_v:.3e} V");
    assert!(
        max_error_v < 1.0e-12,
        "PWM step means deviate from reference by {max_error_v} V"
    );
}
