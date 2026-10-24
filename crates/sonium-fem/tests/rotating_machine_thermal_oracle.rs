//! Same-mesh identity check of the steady thermal twin against the
//! reference solver's `solve_steady_thermal` (scikit-fem).
//!
//! The fixture carries the exact mesh, per-element conductivities
//! (isotropic inner band, anisotropic outer band) and the reference
//! nodal temperatures. The Rust assembly samples the cylindrical
//! tensor at the same quadrature points scikit-fem uses, so the
//! fields must agree to solver precision, not merely in trend.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::mesh::{ElementType, Mesh, Point};
use sonium_fem::physics::electromagnetics::rotating_machine::{
    RobinSurface, ThermalProblem, solve_steady_thermal,
};

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "fixtures/motor_oracle/thermal_crosscheck.json"
    ))
    .expect("thermal fixture is valid JSON")
}

fn numbers(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .expect("array")
        .iter()
        .map(|entry| entry.as_f64().expect("number"))
        .collect()
}

#[test]
fn thermal_field_matches_scikit_fem_on_shared_mesh() {
    let data = fixture();
    let mut mesh = Mesh::new(2);
    for node in data["nodes"].as_array().unwrap() {
        mesh.add_node(Point::new_2d(
            node[0].as_f64().unwrap(),
            node[1].as_f64().unwrap(),
        ));
    }
    let tags: Vec<i32> = data["cell_tags"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tag| i32::try_from(tag.as_i64().unwrap()).unwrap())
        .collect();
    for (triangle, &tag) in data["triangles"].as_array().unwrap().iter().zip(&tags) {
        let nodes: Vec<usize> = triangle
            .as_array()
            .unwrap()
            .iter()
            .map(|node| usize::try_from(node.as_i64().unwrap()).unwrap())
            .collect();
        mesh.add_element_with_marker(ElementType::Triangle, nodes, tag);
    }

    let meta = &data["meta"];
    let surfaces: Vec<RobinSurface> = meta["surfaces"]
        .as_array()
        .unwrap()
        .iter()
        .map(|surface| RobinSurface {
            name: surface["name"].as_str().unwrap().to_string(),
            radius_m: surface["radius_m"].as_f64().unwrap(),
            radius_tol_m: 1.0e-9,
            h_w_per_m2k: surface["h"].as_f64().unwrap(),
            t_sink_c: surface["t_sink"].as_f64().unwrap(),
        })
        .collect();
    let problem = ThermalProblem {
        k_tangential_w_per_mk: numbers(&data["k_tangential"]),
        k_radial_w_per_mk: Some(numbers(&data["k_radial"])),
        heat_source_w_per_m3: numbers(&data["q_source"]),
        drop_markers: Vec::new(),
        element_keep: Vec::new(),
        surfaces,
        volume_sinks: Vec::new(),
        stack_length_m: meta["stack_length_m"].as_f64().unwrap(),
    };
    let result = solve_steady_thermal(&mesh, &problem).expect("thermal solve");
    let reference = numbers(&data["temperature_c"]);
    assert_eq!(result.temperature_c.len(), reference.len());
    let max_error = result
        .temperature_c
        .iter()
        .zip(&reference)
        .map(|(got, want)| (got - want).abs())
        .fold(0.0, f64::max);
    eprintln!(
        "thermal oracle: max |dT| = {max_error:.3e} K, generated {:.6} W \
         (reference {:.6}), residual {:.3e} W",
        result.heat_generated_w,
        data["reference"]["q_generated_W"].as_f64().unwrap(),
        result.heat_residual_w
    );
    assert!(
        max_error < 1.0e-6,
        "temperature field deviates from scikit-fem by {max_error} K"
    );
    let reference_generated = data["reference"]["q_generated_W"].as_f64().unwrap();
    assert!(
        (result.heat_generated_w - reference_generated).abs() < 1.0e-9 * reference_generated,
        "generated heat {} vs reference {reference_generated}",
        result.heat_generated_w
    );
    assert!(
        result.heat_residual_w.abs() < 1.0e-9 * result.heat_generated_w,
        "budget residual {} W",
        result.heat_residual_w
    );
}
