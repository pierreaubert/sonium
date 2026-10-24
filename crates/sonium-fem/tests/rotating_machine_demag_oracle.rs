//! Same-input identity of the demagnetisation ratchet rule against
//! the reference `simulation/demag.py`.
//!
//! The fixture feeds identical element-mean flux densities through
//! Python's `MagnetDemag.update` on the F45SH_120C card curve
//! (175 H values sweeping the knee, at incumbent factors 1.0 and
//! 0.7). The Rust rule must reproduce every factor to round-off:
//! the load-line crossing, the recoil intercept and the monotone
//! ratchet are arithmetic, not discretisation.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::electromagnetics::rotating_machine::MagnetDemagCurve;

#[test]
fn demag_rule_matches_reference_on_shared_inputs() {
    let data: Value =
        serde_json::from_str(include_str!("fixtures/motor_oracle/demag_crosscheck.json"))
            .expect("demag fixture parses");
    let curve_points: Vec<[f64; 2]> = data["meta"]["bh_curve_h_b"]
        .as_array()
        .unwrap()
        .iter()
        .map(|point| [point[0].as_f64().unwrap(), point[1].as_f64().unwrap()])
        .collect();
    let curve = MagnetDemagCurve::new(
        &curve_points,
        data["meta"]["mu_rec"].as_f64().unwrap(),
        data["meta"]["br0_t"].as_f64().unwrap(),
    )
    .expect("curve builds");
    let mut max_error = 0.0f64;
    for case in data["cases"].as_array().unwrap() {
        let br_before = case["br_before"].as_f64().unwrap();
        for (b, expected) in case["b_t"]
            .as_array()
            .unwrap()
            .iter()
            .zip(case["br_after"].as_array().unwrap())
        {
            let got = curve.updated_factor(
                [b[0].as_f64().unwrap(), b[1].as_f64().unwrap()],
                [1.0, 0.0],
                br_before,
            );
            max_error = max_error.max((got - expected.as_f64().unwrap()).abs());
        }
    }
    eprintln!("demag rule oracle: max |dBr| = {max_error:.3e}");
    assert!(
        max_error < 1.0e-12,
        "demag rule deviates from reference by {max_error}"
    );
}
