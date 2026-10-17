//! External reference health and native parity gate for rail-crush.
//!
//! Verifies the checked-in decks match the `sonium_qa::rad` writer, then runs the starter and
//! engine when `OPENCOURANT_EXE` (or legacy `RADISS_EXE`) is set and compares the crusher force
//! (EA delta / L, pre-buckling) and the loaded face motion against the elastic closed form.
//!
//! This original solid-column leg remains elastic. The separate full hollow
//! rail fixture adds plasticity,self-contact and sixty-percent stroke between
//! grips; its construction alone does not establish folding or native parity.
//!
//! Missing references emit a structured optional skip, or fail when
//! SONIUM_QA_REQUIRE_OPENCOURANT=1. External probe health is recorded separately; release
//! success also requires a matching native solve and full named phenomenon.

// Rust guideline compliant 2026-02-21

use sonium_qa::rad::{
    OpenCourantCase, RadElasticMaterial, RadFunction, RadNodeSet, RadProbe, RadWall,
    assert_probe_close, cantilever_brick_mesh, engine_exe, probes_json, run_case, write_engine,
    write_starter,
};
use std::path::PathBuf;

const NAME: &str = "rail_crush";

#[path = "support/rail_full_fixture.rs"]
pub mod rail_full_fixture;

/// Builds the rail crush case on the real `sonium_qa::rad` API.
pub(crate) fn rail_crush_case() -> OpenCourantCase {
    // Short box column along x; 1% quasi-static crush stays elastic and
    // far below the Euler buckling load (5x margin by construction).
    let length = 0.2;
    let section = 0.05;
    let youngs = 200.0e9;
    let (mut mesh, _root_set, _tip_set, tip_node) =
        cantilever_brick_mesh(10, 2, 2, length, section, section);
    let crush = 0.01 * length;
    let speed = 0.1;
    let stop = crush / speed;
    let area = section * section;
    let expected_force = youngs * area * crush / length;
    // Split secondary groups (root/tip halves): a node secondary to
    // both walls trips starter warning 312 (incompatible kinematic
    // conditions). The driver node is appended after so it stays out of
    // both groups.
    let mut root_nodes = Vec::new();
    let mut tip_nodes = Vec::new();
    for (index, node) in mesh.nodes.iter().enumerate() {
        if node[0] <= length / 2.0 {
            root_nodes.push(index + 1);
        } else {
            tip_nodes.push(index + 1);
        }
    }
    let crusher_point = [length, 0.0, 0.0];
    let driver = mesh.add_node(crusher_point);
    OpenCourantCase {
        name: NAME.to_string(),
        title: "sonium_qa rad rail crush elastic compression".to_string(),
        mesh,
        material: RadElasticMaterial {
            youngs_modulus_pa: youngs,
            poisson_ratio: 0.3,
            density_kg_per_m3: 7800.0,
        },
        node_sets: vec![
            RadNodeSet {
                name: "column root".to_string(),
                nodes: root_nodes,
            },
            RadNodeSet {
                name: "column tip".to_string(),
                nodes: tip_nodes,
            },
        ],
        bcs: Vec::new(),
        functions: vec![RadFunction {
            name: "constant crush velocity".to_string(),
            points: vec![(0.0, 1.0), (stop, 1.0)],
        }],
        cloads: Vec::new(),
        walls: vec![
            RadWall {
                name: "fixed anvil".to_string(),
                normal: [1.0, 0.0, 0.0],
                point_m: [0.0, 0.0, 0.0],
                friction: 0.0,
                velocity_m_per_s: None,
                driver_node: None,
                secondary_set: Some(0),
                search_distance_m: 0.0,
            },
            RadWall {
                name: "crusher".to_string(),
                normal: [-1.0, 0.0, 0.0],
                point_m: crusher_point,
                friction: 0.0,
                velocity_m_per_s: Some([-speed, 0.0, 0.0]),
                driver_node: Some(driver),
                secondary_set: Some(1),
                search_distance_m: 0.0,
            },
        ],
        th_nodes: vec![tip_node],
        // Proven on the engine: wall channels report the force ON the
        // wall (reaction). The crusher pushes the column toward -x, so the
        // column pushes the crusher toward +x; the loaded face rides -x.
        probes: vec![
            RadProbe {
                label: "crusher FNX".to_string(),
                channel: "FNX/RWALL/2".to_string(),
                expected: expected_force,
                abs_tol: 1.0e4,
                rel_tol: 0.15,
            },
            RadProbe {
                label: "loaded face DX".to_string(),
                channel: format!("DX/{tip_node}"),
                expected: -crush,
                abs_tol: 1e-9,
                // Measured engine-vs-analytical probe deviations <= 2.7e-2; gate at ~1.5x.
                rel_tol: 0.04,
            },
        ],
        stop_time_s: stop,
        th_period_s: stop / 40.0,
        anim_period_s: stop / 4.0,
    }
}

fn decks_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("decks/rail_crush")
}

const CASE_ID: &str = "fem.opencourant.rail-crush.native-parity.v1";

fn main() {
    let case = rail_crush_case();
    let dir = decks_dir();
    let starter_path = dir.join("rail_crush_0000.rad");
    let engine_path = dir.join("rail_crush_0001.rad");
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
        println!("wrote decks/rail_crush artifacts");
        return;
    }
    // No hand-built decks: the checked-in pair must match the writer.
    let checked_starter = std::fs::read_to_string(&starter_path).unwrap_or_default();
    let checked_engine = std::fs::read_to_string(&engine_path).unwrap_or_default();
    assert_eq!(
        checked_starter, starter,
        "decks/rail_crush/rail_crush_0000.rad drifted from the rad writer"
    );
    assert_eq!(
        checked_engine, engine,
        "decks/rail_crush/rail_crush_0001.rad drifted from the rad writer"
    );
    assert!(
        probes_path.exists(),
        "decks/rail_crush/probes.json must be checked in"
    );
    let _ = probes_json(&case, "none (prove-out deferred)");

    let Some(exe) = engine_exe() else {
        sonium_qa::rad::gate::report_missing_engine(CASE_ID);
    };

    // Run in a scratch dir so engine artifacts never pollute the repo.
    let work = std::env::temp_dir().join(format!("sonium_rad_rail_{}", std::process::id()));
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("rail_crush_0000.rad"), &starter).unwrap();
    std::fs::write(work.join("rail_crush_0001.rad"), &engine).unwrap();
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
    sonium_qa::rad::gate::publish_and_exit(sonium_qa::rad::gate::reference_health_result(
        CASE_ID,
        &output.engine,
        !failed,
    ));
}
