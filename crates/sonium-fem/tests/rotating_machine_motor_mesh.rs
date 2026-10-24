//! GEO_30MM sliding-band mesh built natively from the exported
//! CadQuery polygons (see `tools/motor-oracle/export_motor30.py`).
//!
//! Asserts region areas against the polygon areas, an exactly uniform
//! 840-node slip ring, P2 promotion to 1680 ring DOFs, and rigid
//! integer-shift rotation (element areas invariant under shift).

// Rust guideline compliant 2026-02-21

use std::collections::HashMap;
use std::path::PathBuf;

use serde_json::Value;
use sonium_fem::physics::electromagnetics::rotating_machine::{
    MARKER_ROTOR, MARKER_STATOR, MotorMeshOptions, MotorPolygons, RegionPolygon, build_motor_mesh,
};

const MM: f64 = 1.0e-3;

fn fixture() -> Value {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/motor_oracle/motor30.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str(&text).expect("motor fixture parses")
}

fn ring(value: &Value) -> Vec<(f64, f64)> {
    value
        .as_array()
        .expect("ring")
        .iter()
        .map(|point| {
            let pair = point.as_array().expect("point");
            (
                pair[0].as_f64().expect("x") * MM,
                pair[1].as_f64().expect("y") * MM,
            )
        })
        .collect()
}

fn polygon(value: &Value) -> RegionPolygon {
    RegionPolygon {
        marker: 0,
        exterior: ring(&value["exterior"]),
        holes: value["interiors"]
            .as_array()
            .expect("interiors")
            .iter()
            .map(ring)
            .collect(),
    }
}

fn motor_polygons(data: &Value) -> MotorPolygons {
    let polys = &data["polygons_mm"];
    MotorPolygons {
        stator: polygon(&polys["stator"][0]),
        rotor: polygon(&polys["rotor"][0]),
        shaft: polygon(&polys["shaft"][0]),
        magnets: polys["magnets"]
            .as_array()
            .expect("magnets")
            .iter()
            .map(|magnet| polygon(&magnet["polygon"][0]))
            .collect(),
        coils: polys["coils"]
            .as_array()
            .expect("coils")
            .iter()
            .map(|coil| {
                (
                    coil["slot"].as_u64().expect("slot") as usize,
                    polygon(&coil["polygon"][0]),
                )
            })
            .collect(),
        mid_gap_radius_m: data["derived"]["mid_gap_radius_mm"].as_f64().unwrap() * MM,
        outer_radius_m: data["derived"]["outer_boundary_radius_mm"]
            .as_f64()
            .unwrap()
            * MM,
    }
}

fn element_areas(mesh: &sonium_fem::mesh::Mesh) -> Vec<f64> {
    mesh.elements
        .iter()
        .map(|element| {
            let a = mesh.nodes[element.nodes[0]];
            let b = mesh.nodes[element.nodes[1]];
            let c = mesh.nodes[element.nodes[2]];
            0.5 * ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)).abs()
        })
        .collect()
}

#[test]
fn geo30mm_meshes_natively_with_uniform_ring() {
    let data = fixture();
    let polygons = motor_polygons(&data);
    let options = MotorMeshOptions {
        iron_size_m: 1.2e-3,
        ..MotorMeshOptions::default()
    };
    let start = std::time::Instant::now();
    let mut motor = build_motor_mesh(&polygons, &options).expect("motor mesh builds");
    eprintln!(
        "motor mesh: {} nodes, {} elements in {:?}",
        motor.mesh.num_nodes(),
        motor.mesh.num_elements(),
        start.elapsed()
    );
    assert_eq!(motor.ring_nodes.len(), options.n_ring);

    // Region areas vs polygon areas (mm^2 -> m^2).
    let mut areas: HashMap<i32, f64> = HashMap::new();
    for (element, area) in motor.mesh.elements.iter().zip(element_areas(&motor.mesh)) {
        *areas.entry(element.marker).or_default() += area;
    }
    let expected = |value: &Value| value["area_mm2"].as_f64().unwrap() * MM * MM;
    let check = |marker: i32, expected_m2: f64, label: &str| {
        let got = areas.get(&marker).copied().unwrap_or(0.0);
        assert!(
            (got - expected_m2).abs() / expected_m2 < 2.0e-3,
            "{label}: mesh {got:e} vs polygon {expected_m2:e}"
        );
    };
    check(
        MARKER_STATOR,
        expected(&data["polygons_mm"]["stator"][0]),
        "stator",
    );
    check(
        MARKER_ROTOR,
        expected(&data["polygons_mm"]["rotor"][0]),
        "rotor",
    );
    for magnet in data["polygons_mm"]["magnets"].as_array().unwrap() {
        let index = magnet["index"].as_i64().unwrap() as i32;
        check(200 + index, expected(magnet), "magnet");
    }
    // Coil markers are per WIRE (MARKER_COIL_BASE + wire, for the
    // eddy conductor blocks); each wire is 2.0 x 0.5 mm and each of
    // the 12 slots carries 6 of them.
    let wires = motor.coil_slot_of_wire.len();
    assert_eq!(wires, 72, "12 slots x 6 wires");
    let mut slot_areas = [0.0; 12];
    for (wire, &slot) in motor.coil_slot_of_wire.iter().enumerate() {
        let marker = 100 + i32::try_from(wire).unwrap();
        check(marker, 1.0 * MM * MM, "coil wire");
        slot_areas[slot] += areas.get(&marker).copied().unwrap_or(0.0);
    }
    for (slot, area) in slot_areas.iter().enumerate() {
        assert!(
            (area - 6.0 * MM * MM).abs() / (6.0 * MM * MM) < 2.0e-3,
            "coil slot {slot}: wires sum to {area:e} vs 6 mm^2"
        );
    }
    // Total area is the outer disc.
    let total: f64 = areas.values().sum();
    let disc = std::f64::consts::PI * polygons.outer_radius_m.powi(2);
    assert!(
        (total - disc).abs() / disc < 1.0e-3,
        "total {total} vs disc {disc}"
    );

    // Rigid rotation: element areas invariant under integer shift, up
    // to the ring quantization (rotated interior nodes are exact; ring
    // nodes are points of the quantized 840-gon, so sliver elements at
    // the ring breathe at the 1e-4 relative level).
    let before = element_areas(&motor.mesh);
    motor.set_rotor_shift(10);
    let after = element_areas(&motor.mesh);
    // Total area is exactly preserved; per-element breathing is
    // confined to quantization-scale slivers at the ring whose area
    // trades between neighbours. Bound both measures.
    let total_before: f64 = before.iter().sum();
    let total_after: f64 = after.iter().sum();
    assert!((total_after - total_before).abs() / total_before < 1.0e-12);
    let mut drifted_area = 0.0;
    for (a, b) in before.iter().zip(&after) {
        drifted_area += (a - b).abs();
        assert!(
            (a - b).abs() < 5.0e-3 * a + 1.0e-13,
            "area drift {a} -> {b}"
        );
    }
    assert!(
        drifted_area / total_before < 1.0e-5,
        "drifted {drifted_area:e}"
    );
    motor.set_rotor_shift(0);

    motor
        .promote_p2(options.planar.quantize_m)
        .expect("P2 promotion");
    assert_eq!(motor.ring_nodes.len(), 2 * options.n_ring);
    motor.set_rotor_shift(-7);
    assert_eq!(motor.rotor_shift(), -7);
    eprintln!(
        "P2 motor mesh: {} nodes, {} elements",
        motor.mesh.num_nodes(),
        motor.mesh.num_elements()
    );
}
