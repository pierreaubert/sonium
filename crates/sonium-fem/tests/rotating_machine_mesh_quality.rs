//! Mesh-quality census of the native GEO_30MM sliding-band mesh.
//!
//! Diagnostic gate for iterative/GPU solver viability: the sparse
//! direct backend tolerates near-degenerate elements, Krylov
//! methods do not (see the CUDA parity test). The census tests
//! are ignored diagnostics (run with `--ignored --nocapture`);
//! the final test is the always-on regression tripwire: bit-exact
//! build reproducibility plus the quality floor that contour
//! decimation and the Ruppert circumcenter loop maintain.

// Rust guideline compliant 2026-02-21

use std::path::PathBuf;

use serde_json::Value;
use sonium_fem::physics::electromagnetics::rotating_machine::{
    MotorMeshOptions, MotorPolygons, RegionPolygon, build_motor_mesh,
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

/// (min angle deg, edge ratio, radius ratio R/inr, area, shortest edge)
fn quality(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> (f64, f64, f64, f64, f64) {
    let dist = |p: (f64, f64), q: (f64, f64)| ((p.0 - q.0).powi(2) + (p.1 - q.1).powi(2)).sqrt();
    let (lab, lbc, lca) = (dist(a, b), dist(b, c), dist(c, a));
    let area2 = ((b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)).abs();
    let area = 0.5 * area2;
    let angle = |p: (f64, f64), q: (f64, f64), r: (f64, f64)| -> f64 {
        let u = (q.0 - p.0, q.1 - p.1);
        let v = (r.0 - p.0, r.1 - p.1);
        let cross = (u.0 * v.1 - u.1 * v.0).abs();
        let dot = u.0 * v.0 + u.1 * v.1;
        cross.atan2(dot).to_degrees()
    };
    let min_angle = angle(a, b, c).min(angle(b, c, a)).min(angle(c, a, b));
    let (lmin, lmax) = (lab.min(lbc).min(lca), lab.max(lbc).max(lca));
    let semi = 0.5 * (lab + lbc + lca);
    let inr = if semi > 0.0 { area / semi } else { 0.0 };
    let circumr = if area2 > 0.0 {
        lab * lbc * lca / (2.0 * area2)
    } else {
        f64::INFINITY
    };
    let radius_ratio = if inr > 0.0 {
        circumr / inr
    } else {
        f64::INFINITY
    };
    (min_angle, lmax / lmin.max(1e-300), radius_ratio, area, lmin)
}

#[test]
#[ignore = "diagnostic census; run with --nocapture"]
fn geo30mm_sliver_census() {
    let data = fixture();
    let polygons = motor_polygons(&data);
    let motor = build_motor_mesh(&polygons, &MotorMeshOptions::default()).expect("mesh builds");
    let mesh = &motor.mesh;
    let xy = |i: usize| (mesh.nodes[i].x, mesh.nodes[i].y);

    let mut rows: Vec<(f64, f64, f64, f64, f64, i32, f64)> = Vec::new();
    for element in &mesh.elements {
        let (a, b, c) = (
            xy(element.nodes[0]),
            xy(element.nodes[1]),
            xy(element.nodes[2]),
        );
        let (min_angle, edge_ratio, radius_ratio, area, lmin) = quality(a, b, c);
        let centroid_r = ((a.0 + b.0 + c.0) / 3.0).hypot((a.1 + b.1 + c.1) / 3.0);
        rows.push((
            min_angle,
            edge_ratio,
            radius_ratio,
            area,
            lmin,
            element.marker,
            centroid_r,
        ));
    }
    let n = rows.len();
    let count_below = |deg: f64| rows.iter().filter(|r| r.0 < deg).count();
    eprintln!("elements: {n}");
    for deg in [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0] {
        eprintln!(
            "  min-angle < {deg:4.1} deg: {:6} ({:.2}%)",
            count_below(deg),
            100.0 * count_below(deg) as f64 / n as f64
        );
    }
    let worst_ratio = rows.iter().map(|r| r.2).fold(0.0, f64::max);
    let over = |lim: f64| rows.iter().filter(|r| r.2 > lim).count();
    eprintln!(
        "radius ratio R/r: max {worst_ratio:.3e}; >10: {}, >100: {}, >1000: {}",
        over(10.0),
        over(100.0),
        over(1000.0)
    );
    let mut worst: Vec<_> = rows.iter().collect();
    worst.sort_by(|x, y| x.0.total_cmp(&y.0));
    eprintln!(
        "worst 15 by min angle (deg, edge_ratio, R/r, area mm^2, min_edge um, marker, centroid_r mm):"
    );
    for r in worst.iter().take(15) {
        eprintln!(
            "  {:8.4} {:10.1} {:12.1} {:10.6} {:9.3} {:5} {:7.3}",
            r.0,
            r.1,
            r.2,
            r.3 / (MM * MM),
            r.4 * 1.0e6,
            r.5,
            r.6 / MM
        );
    }
    // Where do the bad ones live? Histogram by centroid radius band.
    let bands = [
        ("shaft r<3.7", 0.0, 3.7e-3),
        ("rotor 3.7-9.0", 3.7e-3, 9.0e-3),
        ("gap 9.0-9.2", 9.0e-3, 9.2e-3),
        ("slots/teeth 9.2-13.5", 9.2e-3, 13.5e-3),
        ("yoke 13.5-15", 13.5e-3, 15.0e-3),
        ("outer air >15", 15.0e-3, f64::INFINITY),
    ];
    for (name, lo, hi) in bands {
        let band: Vec<_> = rows.iter().filter(|r| r.6 >= lo && r.6 < hi).collect();
        let bad = band.iter().filter(|r| r.0 < 10.0).count();
        eprintln!("band {name}: {} elements, {} below 10 deg", band.len(), bad);
    }
}

#[test]
#[ignore = "diagnostic isolation; run with --nocapture"]
fn disk_isolation_census() {
    use sonium_fem::physics::electromagnetics::rotating_machine::{
        PlanarMeshOptions, triangulate_regions,
    };
    let census = |label: &str, mesh: &sonium_fem::mesh::Mesh| {
        let xy = |i: usize| (mesh.nodes[i].x, mesh.nodes[i].y);
        let mut below10 = 0usize;
        let mut min_angle = f64::INFINITY;
        for element in &mesh.elements {
            let (ang, _, _, _, _) = quality(
                xy(element.nodes[0]),
                xy(element.nodes[1]),
                xy(element.nodes[2]),
            );
            if ang < 10.0 {
                below10 += 1;
            }
            min_angle = min_angle.min(ang);
        }
        eprintln!(
            "{label}: {} elements, {} below 10 deg ({:.1}%), global min {min_angle:.3} deg",
            mesh.elements.len(),
            below10,
            100.0 * below10 as f64 / mesh.elements.len() as f64
        );
    };
    for (segments, rounds) in [(256usize, 24usize), (256, 0), (48, 24), (48, 0)] {
        let cover = RegionPolygon {
            marker: 3,
            exterior: (0..segments)
                .map(|k| {
                    let a = 2.0 * std::f64::consts::PI * k as f64 / segments as f64;
                    (3.7e-3 * a.cos(), 3.7e-3 * a.sin())
                })
                .collect(),
            holes: Vec::new(),
        };
        let options = PlanarMeshOptions {
            refinement_rounds: rounds,
            ..PlanarMeshOptions::default()
        };
        let mesh = triangulate_regions(&cover, &[], &|_, _| 9.0e-4, &options).expect("disk meshes");
        census(&format!("disk seg={segments} rounds={rounds}"), &mesh);
    }
}

/// Regression gate: the GEO_30MM mesh build is reproducible to the
/// bit and clears the quality floor the Ruppert loop targets.
///
/// Determinism caught a real bug: greedy contour decimation once
/// scanned a HashMap, so each process kept different chain
/// vertices and produced a different mesh (and a Krylov
/// convergence lottery downstream). Two builds in one process
/// still expose that class: std HashMap seeds differ per instance.
#[test]
fn geo30mm_mesh_is_deterministic_and_above_quality_floor() {
    let data = fixture();
    let polygons = motor_polygons(&data);
    let first = build_motor_mesh(&polygons, &MotorMeshOptions::default()).expect("mesh builds");
    let second = build_motor_mesh(&polygons, &MotorMeshOptions::default()).expect("mesh builds");
    assert_eq!(first.mesh.nodes.len(), second.mesh.nodes.len());
    for (a, b) in first.mesh.nodes.iter().zip(&second.mesh.nodes) {
        assert_eq!(a.x.to_bits(), b.x.to_bits());
        assert_eq!(a.y.to_bits(), b.y.to_bits());
    }
    assert_eq!(first.mesh.elements.len(), second.mesh.elements.len());
    for (a, b) in first.mesh.elements.iter().zip(&second.mesh.elements) {
        assert_eq!(a.nodes, b.nodes);
        assert_eq!(a.marker, b.marker);
    }

    let mesh = &first.mesh;
    let xy = |i: usize| (mesh.nodes[i].x, mesh.nodes[i].y);
    let n = mesh.elements.len();
    let mut below_1 = 0usize;
    let mut below_10 = 0usize;
    let mut worst_ratio = 0.0f64;
    for element in &mesh.elements {
        let (min_angle, _, radius_ratio, _, _) = quality(
            xy(element.nodes[0]),
            xy(element.nodes[1]),
            xy(element.nodes[2]),
        );
        if min_angle < 1.0 {
            below_1 += 1;
        }
        if min_angle < 10.0 {
            below_10 += 1;
        }
        worst_ratio = worst_ratio.max(radius_ratio);
    }
    // Measured on the quality-refined mesh: 0 below 1 deg, 30
    // (0.07 %) below 10 deg (all at sharp slot-insulation corners
    // the loop cannot repair), worst radius ratio 300.
    assert_eq!(below_1, 0, "no element may collapse below 1 degree");
    assert!(
        below_10 * 200 <= n,
        "below-10-degree share {below_10}/{n} exceeds 0.5 %"
    );
    assert!(
        worst_ratio <= 1000.0,
        "worst radius ratio {worst_ratio} exceeds 1000"
    );
}
