//! External reference health and native parity gate for beam-3d-cantilever.
//!
//! Verifies the checked-in decks match the `sonium_qa::rad` writer, then runs the starter and
//! engine when `OPENCOURANT_EXE` (or legacy `RADISS_EXE`) is set and compares the brick-mesh
//! tip displacement against Euler-Bernoulli beam theory. Case contributed by the beams lane;
//! Worker 1 owns the runner.
//!
//! Missing references emit a structured optional skip, or fail when
//! SONIUM_QA_REQUIRE_OPENCOURANT=1. External probe health is recorded separately; release
//! success also requires a matching native solve and full named phenomenon.

// Rust guideline compliant 2026-02-21

use sonium_qa::rad::{
    OpenCourantCase, RadBoundaryCondition, RadConcentratedLoad, RadElasticMaterial, RadFunction,
    RadNodeSet, RadProbe, assert_probe_close, cantilever_brick_mesh, engine_exe, probes_json,
    run_case, write_engine, write_starter,
};
use std::path::PathBuf;

#[path = "support/cantilever_native.rs"]
mod cantilever_native;

/// Single-sourced 3-D cantilever comparison case (Y tip load, SI units).
pub fn beam_3d_cantilever_case() -> OpenCourantCase {
    let (mesh, root_set, tip_set, tip_node) = cantilever_brick_mesh(10, 2, 2, 1.0, 0.05, 0.05);
    let expected = sonium_qa::beam_euler_cantilever_point(
        1.0,
        1.0,
        200.0e9,
        0.05 * 0.05_f64.powi(3) / 12.0,
        1000.0,
    );
    // /CLOAD applies the scale to EACH node of the set.
    let tip_scale = 1000.0 / tip_set.len() as f64;
    OpenCourantCase {
        name: "beam_3d_cantilever".to_string(),
        title: "sonium_qa rad beam_3d_cantilever quasi-static tip load".to_string(),
        mesh,
        material: RadElasticMaterial {
            youngs_modulus_pa: 200.0e9,
            poisson_ratio: 0.3,
            density_kg_per_m3: 7800.0,
        },
        node_sets: vec![
            RadNodeSet {
                name: "root".to_string(),
                nodes: root_set,
            },
            RadNodeSet {
                name: "tip".to_string(),
                nodes: tip_set,
            },
        ],
        bcs: vec![RadBoundaryCondition {
            name: "clamp root".to_string(),
            set: 0,
            lock: [true, true, true, true, true, true],
        }],
        functions: vec![RadFunction {
            name: "ramp 0 to 1 over stop time".to_string(),
            points: vec![(0.0, 0.0), (0.9, 0.9), (1.0, 1.0)],
        }],
        cloads: vec![RadConcentratedLoad {
            name: "tip load".to_string(),
            set: 1,
            direction: 2,
            function: 0,
            force_scale_n: tip_scale,
        }],
        walls: Vec::new(),
        th_nodes: vec![tip_node],
        probes: vec![RadProbe {
            label: "tip DY".to_string(),
            channel: format!("DY/{tip_node}"),
            expected,
            abs_tol: 1e-9,
            // Measured engine-vs-analytical probe deviation = 2.6e-2; gate at ~1.5x.
            rel_tol: 0.04,
        }],
        stop_time_s: 1.0,
        th_period_s: 0.01,
        anim_period_s: 0.1,
    }
}

fn decks_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("decks/beam_3d_cantilever")
}

const CASE_ID: &str = "fem.opencourant.beam-3d-cantilever.native-parity.v1";

fn main() {
    let case = beam_3d_cantilever_case();
    if std::env::var("REGEN_DECKS").as_deref() == Ok("1") {
        let dir = decks_dir();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("beam_3d_cantilever_0000.rad"),
            write_starter(&case),
        )
        .unwrap();
        std::fs::write(dir.join("beam_3d_cantilever_0001.rad"), write_engine(&case)).unwrap();
        std::fs::write(
            dir.join("probes.json"),
            probes_json(&case, "none (prove-out deferred)"),
        )
        .unwrap();
        println!("regenerated decks for beam_3d_cantilever");
        return;
    }
    let dir = decks_dir();
    let starter_path = dir.join("beam_3d_cantilever_0000.rad");
    let engine_path = dir.join("beam_3d_cantilever_0001.rad");
    let probes_path = dir.join("probes.json");

    // No hand-built decks: the checked-in pair must match the writer.
    let starter = write_starter(&case);
    let engine = write_engine(&case);
    let checked_starter = std::fs::read_to_string(&starter_path).unwrap_or_default();
    let checked_engine = std::fs::read_to_string(&engine_path).unwrap_or_default();
    assert_eq!(
        checked_starter, starter,
        "decks/beam_3d_cantilever/beam_3d_cantilever_0000.rad drifted from the rad writer"
    );
    assert_eq!(
        checked_engine, engine,
        "decks/beam_3d_cantilever/beam_3d_cantilever_0001.rad drifted from the rad writer"
    );
    assert!(
        probes_path.exists(),
        "decks/beam_3d_cantilever/probes.json must be checked in"
    );
    let _ = probes_json(&case, "none (prove-out deferred)");

    let Some(exe) = engine_exe() else {
        sonium_qa::rad::gate::report_missing_engine(CASE_ID);
    };

    // Run in a scratch dir so engine artifacts never pollute the repo.
    let work = std::env::temp_dir().join(format!(
        "sonium_rad_beam_3d_cantilever_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("beam_3d_cantilever_0000.rad"), &starter).unwrap();
    std::fs::write(work.join("beam_3d_cantilever_0001.rad"), &engine).unwrap();
    let output = run_case(&work, &exe, &case).unwrap_or_else(|error| {
        sonium_qa::rad::gate::publish_and_exit(sonium_qa::rad::gate::execution_failure_result(
            CASE_ID,
            &error.to_string(),
        ));
    });
    println!("engine: {}", output.engine);
    let mut failed = false;
    for probe in &case.probes {
        match output.history.final_value(&probe.channel) {
            Ok(actual) => {
                println!(
                    "probe {}: actual {actual:.6e}, expected {:.6e}",
                    probe.label, probe.expected
                );
                if let Err(error) = assert_probe_close(
                    actual,
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
    let result = output
        .history
        .final_value(&case.probes[0].channel)
        .map_err(|error| error.to_string())
        .and_then(|tip| {
            cantilever_native::parity_result(&case, CASE_ID, &output.engine, tip, !failed)
                .map_err(|error| error.to_string())
        })
        .unwrap_or_else(|error| {
            sonium_qa::QaResult::failed(
                CASE_ID,
                format!("native transient validation failed: {error}"),
            )
            .requested_backend("cpu+opencourant")
            .actual_backend("cpu+opencourant")
            .metric("external_reference_executed", 1.0)
            .metric("native_parity_executed", 0.0)
        });
    sonium_qa::rad::gate::publish_and_exit(result);
}
