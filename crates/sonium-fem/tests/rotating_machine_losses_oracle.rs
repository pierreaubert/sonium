//! Loss-model cross-checks against the reference Python implementation.
//!
//! Reference values were produced by `core_loss_surface` and
//! `field_ops.copper_loss_W` of eMotres/Motor_Ai_Simulation on the
//! B15AHV950M record and the GEO_30MM winding (see
//! `tools/motor-oracle/`). The steel curves themselves are read from
//! the exported motor fixture, so the test pins the interpolation and
//! blending arithmetic, not a copy of the table.

// Rust guideline compliant 2026-02-21

use std::path::PathBuf;

use serde_json::Value;
use sonium_fem::physics::electromagnetics::rotating_machine::{
    BertottiCoefficients, CopperLossInput, MeasuredLossSurface, copper_loss_dc,
};

fn fixture() -> Value {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/motor_oracle/motor30.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str(&text).expect("motor fixture parses")
}

fn steel_surface(data: &Value) -> (MeasuredLossSurface, BertottiCoefficients) {
    let steel = &data["materials"]["steel_B15AHV950M"];
    let bertotti = BertottiCoefficients {
        kh: steel["core_loss_kh"].as_f64().expect("kh"),
        kc: steel["core_loss_kc"].as_f64().expect("kc"),
        ke: steel["core_loss_ke"].as_f64().expect("ke"),
    };
    let curves = steel["core_loss_curves"].as_object().expect("curves");
    let mut frequencies = Vec::new();
    let mut tables = Vec::new();
    for (key, curve) in curves {
        let frequency: f64 = key
            .trim_end_matches("Hz")
            .parse()
            .unwrap_or_else(|_| panic!("frequency key {key}"));
        let points: Vec<(f64, f64)> = curve
            .as_array()
            .expect("curve")
            .iter()
            .map(|point| {
                let pair = point.as_array().expect("point");
                (pair[0].as_f64().expect("B"), pair[1].as_f64().expect("P"))
            })
            .collect();
        frequencies.push(frequency);
        tables.push(points);
    }
    let density = steel["density"].as_f64().expect("density");
    let w_per_kg = steel["core_loss_curve_unit"].as_str() == Some("w_per_kg");
    let surface = MeasuredLossSurface::new(&frequencies, &tables, density, w_per_kg, bertotti)
        .expect("surface builds");
    (surface, bertotti)
}

#[test]
fn measured_surface_matches_python_values() {
    let data = fixture();
    let (surface, _) = steel_surface(&data);
    // (B [T], f [Hz], python w_per_m3, python envelope_b [T])
    let cases = [
        (0.5, 50.0, 1869.6, 1.886),
        (1.0, 400.0, 67188.55338, 1.692),
        (1.5, 1000.0, 566803.9525, 1.597),
        (1.9, 1750.0, 1697432.549, 1.44201),
        (0.8, 3333.0, 917848.1443, 1.12514),
        (2.2, 100.0, 53923.5007, 1.88),
    ];
    for (b, f, expected, envelope) in cases {
        let (value, _) = surface.w_per_m3(b, f);
        assert!(
            (value - expected).abs() / expected < 5.0e-4,
            "B={b} f={f}: rust {value} vs python {expected}"
        );
        let env = surface.envelope_b_t(f);
        assert!(
            (env - envelope).abs() / envelope < 5.0e-4,
            "envelope f={f}: rust {env} vs python {envelope}"
        );
    }
}

#[test]
fn copper_loss_matches_python_dc_value() {
    let loss = copper_loss_dc(&CopperLossInput {
        num_slots: 12,
        conductors_per_slot: 6.0,
        nominal_wire_area_m2: 1.0e-6,
        measured_copper_area_m2: Some(72.0e-6),
        stack_length_m: 0.01,
        n_parallel: 1.0,
        i_phase_rms_a: 60.0,
        coil_temp_c: 120.0,
        end_factor: 0.0,
        tooth_width_m: 2.6e-3,
        bundle_width_m: 2.0e-3,
    });
    assert!(
        (loss.power_w - 107.22580717628409).abs() / 107.22580717628409 < 1.0e-9,
        "P_cu {}",
        loss.power_w
    );
    assert!((loss.end_factor - 1.7225663103256523).abs() < 1.0e-12);
    assert!(
        (loss.r_phase_ohm - 0.009928315479285564).abs() / 0.009928315479285564 < 1.0e-9,
        "R_phase {}",
        loss.r_phase_ohm
    );
}
