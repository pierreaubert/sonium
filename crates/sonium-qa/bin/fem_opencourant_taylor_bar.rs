//! External reference health and native parity gate for taylor-bar.
//!
//! Verifies the checked-in decks match the `sonium_qa::rad` writer, then runs the starter and
//! engine when `OPENCOURANT_EXE` (or legacy `RADISS_EXE`) is set and compares the wall force
//! plateau (rho c A V) and the impact-face motion against the 1D elastic impact closed form.
//!
//! Frame note: `sonium_qa::rad` currently offers no initial-velocity card, so the bar starts at
//! rest while the wall rises at the impact speed. By Galilean equivalence the contact force
//! matches the flying-bar test exactly; the displacement probe uses the laboratory-frame value.
//! The complete copper plastic leg continues to 80 us with isothermal Johnson-Cook
//! flow and validates the full mushroom shape and plastic-strain field against an
//! independent viscous-stabilization reference at the same physical timestamp.
//!
//! Missing references emit a structured optional skip, or fail when
//! SONIUM_QA_REQUIRE_OPENCOURANT=1. External probe health is recorded separately; release
//! success also requires a matching native solve and full named phenomenon.

// Rust guideline compliant 2026-02-21

use sonium_qa::rad::{
    OpenCourantCase, RadElasticMaterial, RadFunction, RadMesh, RadNodeSet, RadProbe, RadWall,
    assert_probe_close, engine_exe, probes_json, run_case, write_engine, write_starter,
};
use std::path::PathBuf;

#[path = "support/taylor_full_fixture.rs"]
pub mod taylor_full_fixture;
#[path = "support/taylor_native.rs"]
pub mod taylor_native;

const NAME: &str = "taylor_bar";
const LENGTH: f64 = 0.0254;
const RADIUS: f64 = 0.003175;
const VELOCITY: f64 = 227.0;
const YOUNGS: f64 = 117.0e9;
const POISSON: f64 = 0.35;
const DENSITY: f64 = 8930.0;

/// Builds the Taylor bar case on the real `sonium_qa::rad` API.
pub fn taylor_bar_case() -> OpenCourantCase {
    let (mut mesh, face_set, face_center) = cylinder_mesh(8, 6);
    let area = std::f64::consts::PI * RADIUS * RADIUS;
    let c = (YOUNGS / DENSITY).sqrt();
    let plateau = DENSITY * c * area * VELOCITY;
    // One microsecond of free approach, then 3 us of contact; the run ends
    // well before the release wave returns (2L/c ~ 14 us).
    let stop = 4.0e-6;
    let travel = VELOCITY * (stop - 1.0e-6);
    // Secondary group covers the whole bar (mushrooming spreads contact past
    // the initial face); the driver node is appended after so it stays out
    // of the contact group.
    let bar_nodes: Vec<usize> = (1..=mesh.num_nodes()).collect();
    let anvil_point = [0.0, 0.0, 0.0];
    let driver = mesh.add_node(anvil_point);
    OpenCourantCase {
        name: NAME.to_string(),
        title: "sonium_qa rad taylor bar elastic impact".to_string(),
        mesh,
        material: RadElasticMaterial {
            youngs_modulus_pa: YOUNGS,
            poisson_ratio: POISSON,
            density_kg_per_m3: DENSITY,
        },
        node_sets: vec![
            RadNodeSet {
                name: "impact face".to_string(),
                nodes: face_set,
            },
            RadNodeSet {
                name: "bar".to_string(),
                nodes: bar_nodes,
            },
        ],
        bcs: Vec::new(),
        functions: vec![RadFunction {
            name: "constant wall velocity".to_string(),
            points: vec![(0.0, 1.0), (stop, 1.0)],
        }],
        cloads: Vec::new(),
        walls: vec![RadWall {
            name: "rising anvil".to_string(),
            normal: [0.0, 0.0, 1.0],
            point_m: anvil_point,
            friction: 0.0,
            velocity_m_per_s: Some([0.0, 0.0, VELOCITY]),
            driver_node: Some(driver),
            secondary_set: Some(1),
            search_distance_m: 0.0,
        }],
        th_nodes: vec![face_center],
        probes: vec![
            RadProbe {
                label: "anvil FNZ plateau".to_string(),
                channel: "FNZ/RWALL/1".to_string(),
                // Proven on the engine: wall channels report the force ON
                // the wall (reaction), so the rising anvil reads negative.
                expected: -plateau,
                abs_tol: 1.0e3,
                rel_tol: 0.25,
            },
            RadProbe {
                label: "impact face DZ".to_string(),
                channel: format!("DZ/{face_center}"),
                expected: travel,
                abs_tol: 1e-9,
                // Measured engine-vs-analytical probe deviations <= 2.8e-2; gate at ~1.4x.
                rel_tol: 0.04,
            },
        ],
        stop_time_s: stop,
        th_period_s: 1.0e-7,
        anim_period_s: 1.0e-6,
    }
}

/// Bricks a cylinder with 1-based connectivity for `RadMesh`.
///
/// Clips an `n` by `n` Cartesian grid per layer to the circular section
/// and projects region-boundary nodes radially; the impact face floats
/// one approach gap above the wall. Returns the mesh, the impact-face
/// node set, and the face-center node id.
pub fn cylinder_mesh(nz: usize, n: usize) -> (RadMesh, Vec<usize>, usize) {
    let gap = VELOCITY * 1.0e-6;
    let step = 2.0 * RADIUS / n as f64;
    let corner = |i: usize| -RADIUS + step * i as f64;
    let mut kept = vec![false; n * n];
    for i in 0..n {
        for j in 0..n {
            kept[i * n + j] = (0..=1).all(|di| {
                (0..=1).all(|dj| {
                    let (x, y) = (corner(i + di), corner(j + dj));
                    x * x + y * y <= RADIUS * RADIUS
                })
            });
        }
    }
    let mut layer_id = vec![0usize; (n + 1) * (n + 1)];
    let mut layer_nodes: Vec<[f64; 2]> = Vec::new();
    for i in 0..=n {
        for j in 0..=n {
            let mut used = false;
            let mut boundary = false;
            for di in 0..=1 {
                for dj in 0..=1 {
                    let (ci, cj) = (i + 1 - di, j + 1 - dj);
                    if ci == 0 || cj == 0 || ci > n || cj > n {
                        continue;
                    }
                    if kept[(ci - 1) * n + (cj - 1)] {
                        used = true;
                    } else {
                        boundary = true;
                    }
                }
            }
            if i == 0 || j == 0 || i == n || j == n {
                boundary = true;
            }
            if !used {
                continue;
            }
            let (mut x, mut y) = (corner(i), corner(j));
            if boundary {
                let r = (x * x + y * y).sqrt();
                if r > 1e-12 {
                    x *= RADIUS / r;
                    y *= RADIUS / r;
                }
            }
            layer_id[i * (n + 1) + j] = layer_nodes.len() + 1;
            layer_nodes.push([x, y]);
        }
    }
    let per_layer = layer_nodes.len();
    let mut nodes = Vec::with_capacity(per_layer * (nz + 1));
    for k in 0..=nz {
        let z = gap + LENGTH * k as f64 / nz as f64;
        for xy in &layer_nodes {
            nodes.push([xy[0], xy[1], z]);
        }
    }
    let mut bricks = Vec::new();
    for k in 0..nz {
        for i in 0..n {
            for j in 0..n {
                if !kept[i * n + j] {
                    continue;
                }
                let at = |di: usize, dj: usize| layer_id[(i + di) * (n + 1) + (j + dj)];
                let base = k * per_layer;
                bricks.push([
                    base + at(0, 0),
                    base + at(1, 0),
                    base + at(1, 1),
                    base + at(0, 1),
                    base + per_layer + at(0, 0),
                    base + per_layer + at(1, 0),
                    base + per_layer + at(1, 1),
                    base + per_layer + at(0, 1),
                ]);
            }
        }
    }
    let face_set: Vec<usize> = (1..=per_layer).collect();
    // Node nearest the section centroid on the impact face.
    let face_center = layer_nodes
        .iter()
        .enumerate()
        .min_by(|a, b| {
            let ra = a.1[0] * a.1[0] + a.1[1] * a.1[1];
            let rb = b.1[0] * b.1[0] + b.1[1] * b.1[1];
            ra.total_cmp(&rb)
        })
        .map(|(index, _)| index + 1)
        .unwrap_or(1);
    (RadMesh { nodes, bricks }, face_set, face_center)
}

fn decks_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("decks/taylor_bar")
}

const CASE_ID: &str = "fem.opencourant.taylor-bar.native-parity.v1";

fn main() {
    let case = taylor_bar_case();
    let dir = decks_dir();
    let starter_path = dir.join("taylor_bar_0000.rad");
    let engine_path = dir.join("taylor_bar_0001.rad");
    let probes_path = dir.join("probes.json");

    let starter = write_starter(&case);
    let engine = write_engine(&case);
    if std::env::var("SONIUM_RAD_WRITE_DECKS").is_ok() {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&starter_path, &starter).unwrap();
        std::fs::write(&engine_path, &engine).unwrap();
        std::fs::write(
            &probes_path,
            probes_json(&case, "none (prove-out deferred)"),
        )
        .unwrap();
        println!("wrote decks/taylor_bar artifacts");
        return;
    }
    // No hand-built decks: the checked-in pair must match the writer.
    let checked_starter = std::fs::read_to_string(&starter_path).unwrap_or_default();
    let checked_engine = std::fs::read_to_string(&engine_path).unwrap_or_default();
    assert_eq!(
        checked_starter, starter,
        "decks/taylor_bar/taylor_bar_0000.rad drifted from the rad writer"
    );
    assert_eq!(
        checked_engine, engine,
        "decks/taylor_bar/taylor_bar_0001.rad drifted from the rad writer"
    );
    assert!(
        probes_path.exists(),
        "decks/taylor_bar/probes.json must be checked in"
    );
    let _ = probes_json(&case, "none (prove-out deferred)");

    let Some(exe) = engine_exe() else {
        sonium_qa::rad::gate::report_missing_engine(CASE_ID);
    };

    // Run in a scratch dir so engine artifacts never pollute the repo.
    let work = std::env::temp_dir().join(format!("sonium_rad_taylor_{}", std::process::id()));
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("taylor_bar_0000.rad"), &starter).unwrap();
    std::fs::write(work.join("taylor_bar_0001.rad"), &engine).unwrap();
    let output = run_case(&work, &exe, &case).unwrap_or_else(|error| {
        sonium_qa::rad::gate::publish_and_exit(sonium_qa::rad::gate::execution_failure_result(
            CASE_ID,
            &error.to_string(),
        ));
    });
    println!("engine: {}", output.engine);
    let mut failed = false;
    // th_to_csv reports wall forces directly (it differentiates the T01
    // impulses), so every probe reads its final channel value.
    for probe in &case.probes {
        match output.history.final_value(&probe.channel) {
            Ok(value) => {
                println!(
                    "probe {}: actual {value:.6e}, expected {:.6e}",
                    probe.label, probe.expected
                );
                if let Err(error) = assert_probe_close(
                    value,
                    probe.expected,
                    probe.abs_tol,
                    probe.rel_tol,
                    &probe.label,
                ) {
                    eprintln!("FAIL: {error}");
                    failed = true;
                }
            }
            Err(error) => {
                eprintln!("FAIL: probe {} unreadable: {error}", probe.label);
                failed = true;
            }
        }
    }
    std::fs::remove_dir_all(&work).ok();
    let result = taylor_native::parity_result(&case, CASE_ID, &output.engine, !failed)
        .unwrap_or_else(|error| {
            sonium_qa::QaResult::failed(CASE_ID, error.to_string())
                .metric("external_reference_health_passed", f64::from(!failed))
                .metric("native_parity_executed", 0.0)
                .reference(&output.engine)
        });
    sonium_qa::rad::gate::publish_and_exit(result);
}
