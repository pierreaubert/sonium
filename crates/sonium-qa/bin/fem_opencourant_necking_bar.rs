//! External reference health and native parity gate for necking-bar.
//!
//! Verifies the checked-in decks match the `sonium_qa::rad` writer, then runs the starter and
//! engine when `OPENCOURANT_EXE` (or legacy `RADISS_EXE`) is set and compares the tip
//! elongation against the elastic bar closed form (FL/EA) at a sub-yield load.
//!
//! The full necking bar has no closed form: this deck plus the Wolfram incremental reference
//! and the Considere/Bridgman checks in `qa-fem-necking-bar` are separate validation
//! legs. The shared complete plastic fixture adds explicit hardening and pulling
//! beyond the elastic regime; its native parity and necking gate remain pending.
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

const NAME: &str = "necking_bar";

#[path = "support/necking_full_fixture.rs"]
pub mod necking_full_fixture;

/// Builds the necking bar case on the real `sonium_qa::rad` API.
pub(crate) fn necking_bar_case() -> OpenCourantCase {
    let length = 0.1;
    let section = 0.01;
    let youngs = 200.0e9;
    // 20 kN holds the bar at 200 MPa: elastic, sub-yield, comparable with
    // the future plastic leg before diffuse necking.
    let force = 20.0e3;
    let (mesh, root_set, tip_set, tip_node) =
        cantilever_brick_mesh(10, 2, 2, length, section, section);
    let expected = force * length / (youngs * section * section);
    let tip_scale = force / tip_set.len() as f64;
    OpenCourantCase {
        name: NAME.to_string(),
        title: "sonium_qa rad necking bar elastic tension".to_string(),
        mesh,
        material: RadElasticMaterial {
            youngs_modulus_pa: youngs,
            poisson_ratio: 0.3,
            density_kg_per_m3: 7800.0,
        },
        node_sets: vec![
            RadNodeSet {
                name: "grip".to_string(),
                nodes: root_set,
            },
            RadNodeSet {
                name: "pull".to_string(),
                nodes: tip_set,
            },
        ],
        bcs: vec![RadBoundaryCondition {
            name: "clamp grip".to_string(),
            set: 0,
            lock: [true, true, true, true, true, true],
        }],
        functions: vec![RadFunction {
            name: "ramp 0 to 1 over stop time".to_string(),
            points: vec![(0.0, 0.0), (0.9, 0.9), (1.0, 1.0)],
        }],
        cloads: vec![RadConcentratedLoad {
            name: "tensile pull".to_string(),
            set: 1,
            direction: 1,
            function: 0,
            force_scale_n: tip_scale,
        }],
        walls: Vec::new(),
        th_nodes: vec![tip_node],
        probes: vec![RadProbe {
            label: "tip DX".to_string(),
            channel: format!("DX/{tip_node}"),
            expected,
            abs_tol: 1e-9,
            // Measured engine-vs-analytical probe deviation = 4.8e-2; gate at ~1.25x.
            rel_tol: 0.06,
        }],
        stop_time_s: 1.0,
        th_period_s: 0.01,
        anim_period_s: 0.1,
    }
}

fn decks_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("decks/necking_bar")
}

const CASE_ID: &str = "fem.opencourant.necking-bar.native-parity.v1";

fn main() {
    let case = necking_bar_case();
    let dir = decks_dir();
    let starter_path = dir.join("necking_bar_0000.rad");
    let engine_path = dir.join("necking_bar_0001.rad");
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
        println!("wrote decks/necking_bar artifacts");
        return;
    }
    // No hand-built decks: the checked-in pair must match the writer.
    let checked_starter = std::fs::read_to_string(&starter_path).unwrap_or_default();
    let checked_engine = std::fs::read_to_string(&engine_path).unwrap_or_default();
    assert_eq!(
        checked_starter, starter,
        "decks/necking_bar/necking_bar_0000.rad drifted from the rad writer"
    );
    assert_eq!(
        checked_engine, engine,
        "decks/necking_bar/necking_bar_0001.rad drifted from the rad writer"
    );
    assert!(
        probes_path.exists(),
        "decks/necking_bar/probes.json must be checked in"
    );
    let _ = probes_json(&case, "none (prove-out deferred)");

    let Some(exe) = engine_exe() else {
        sonium_qa::rad::gate::report_missing_engine(CASE_ID);
    };

    // Run in a scratch dir so engine artifacts never pollute the repo.
    let work = std::env::temp_dir().join(format!("sonium_rad_necking_{}", std::process::id()));
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("necking_bar_0000.rad"), &starter).unwrap();
    std::fs::write(work.join("necking_bar_0001.rad"), &engine).unwrap();
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
    sonium_qa::rad::gate::publish_and_exit(sonium_qa::rad::gate::reference_health_result(
        CASE_ID,
        &output.engine,
        !failed,
    ));
}
