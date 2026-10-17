//! External reference health and native parity gate for beam-torsion.
//!
//! Verifies the checked-in decks match the `sonium_qa::rad` writer, then runs the starter and
//! engine when `OPENCOURANT_EXE` (or legacy `RADISS_EXE`) is set and compares the mid-span
//! corner displacement against the Saint-Venant twist with the exact square-section torsion
//! constant. Case contributed by the beams lane; Worker 1 owns the runner.
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

/// Exact torsion constant of a square bar with side `a`.
///
/// Uses `J/a^4 = 1/3 - (64/pi^5) sum tanh((2n-1) pi/2)/(2n-1)^5`, which
/// converges to machine precision in a handful of terms.
pub fn square_torsion_constant(a: f64) -> f64 {
    let pi = std::f64::consts::PI;
    let mut sum = 0.0;
    for n in 1..=10 {
        let odd: f64 = 2.0 * n as f64 - 1.0;
        sum += ((odd * pi / 2.0).tanh()) / odd.powi(5);
    }
    a.powi(4) * (1.0 / 3.0 - 64.0 / pi.powi(5) * sum)
}

/// Single-sourced square-bar torsion comparison case (SI units).
///
/// A tip force couple applies torque about +X; the probe reads the corner
/// transverse displacement at mid-span, far from root warping restraint and
/// tip load-introduction effects.
pub fn beam_torsion_case() -> OpenCourantCase {
    let nx = 24;
    let ny = 6;
    let nz = 6;
    let length = 2.0;
    let side = 0.05;
    let torque = 500.0;
    let (mesh, root_set, _tip_set, _tip_node) =
        cantilever_brick_mesh(nx, ny, nz, length, side, side);
    let id = |i: usize, j: usize, k: usize| i * (ny + 1) * (nz + 1) + j * (nz + 1) + k + 1;
    let torsion = square_torsion_constant(side);
    let shear = 70.0e9 / (2.0 * (1.0 + 0.33));
    let twist = torque * length / (shear * torsion);
    // Force couple about +X: -Fz at y=-a/2, +Fz at y=+a/2 on the tip face.
    let couple = torque / side;
    let load_a = id(nx, 0, nz / 2);
    let load_b = id(nx, ny, nz / 2);
    // Mid-span corner (y=-a/2, z=-a/2): Duy = -phi*z = +phi*a/2.
    let corner = id(nx / 2, 0, 0);
    let expected = (twist / 2.0) * side / 2.0;
    OpenCourantCase {
        name: "beam_torsion".to_string(),
        title: "sonium_qa rad beam_torsion quasi-static torque".to_string(),
        mesh,
        material: RadElasticMaterial {
            youngs_modulus_pa: 70.0e9,
            poisson_ratio: 0.33,
            density_kg_per_m3: 2700.0,
        },
        node_sets: vec![
            RadNodeSet {
                name: "root".to_string(),
                nodes: root_set,
            },
            RadNodeSet {
                name: "couple low".to_string(),
                nodes: vec![load_a],
            },
            RadNodeSet {
                name: "couple high".to_string(),
                nodes: vec![load_b],
            },
            RadNodeSet {
                name: "probe corner".to_string(),
                nodes: vec![corner],
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
        cloads: vec![
            RadConcentratedLoad {
                name: "couple low".to_string(),
                set: 1,
                direction: 3,
                function: 0,
                force_scale_n: -couple,
            },
            RadConcentratedLoad {
                name: "couple high".to_string(),
                set: 2,
                direction: 3,
                function: 0,
                force_scale_n: couple,
            },
        ],
        walls: Vec::new(),
        th_nodes: vec![corner],
        probes: vec![RadProbe {
            label: "corner DY".to_string(),
            channel: format!("DY/{corner}"),
            expected,
            abs_tol: 1e-9,
            // Measured engine-vs-analytical probe deviation = 2.4e-2; gate at ~1.6x.
            rel_tol: 0.04,
        }],
        stop_time_s: 1.0,
        th_period_s: 0.01,
        anim_period_s: 0.1,
    }
}

fn decks_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("decks/beam_torsion")
}

const CASE_ID: &str = "fem.opencourant.beam-torsion.native-parity.v1";

fn main() {
    let case = beam_torsion_case();
    if std::env::var("REGEN_DECKS").as_deref() == Ok("1") {
        let dir = decks_dir();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("beam_torsion_0000.rad"), write_starter(&case)).unwrap();
        std::fs::write(dir.join("beam_torsion_0001.rad"), write_engine(&case)).unwrap();
        std::fs::write(
            dir.join("probes.json"),
            probes_json(&case, "none (prove-out deferred)"),
        )
        .unwrap();
        println!("regenerated decks for beam_torsion");
        return;
    }
    let dir = decks_dir();
    let starter_path = dir.join("beam_torsion_0000.rad");
    let engine_path = dir.join("beam_torsion_0001.rad");
    let probes_path = dir.join("probes.json");

    // No hand-built decks: the checked-in pair must match the writer.
    let starter = write_starter(&case);
    let engine = write_engine(&case);
    let checked_starter = std::fs::read_to_string(&starter_path).unwrap_or_default();
    let checked_engine = std::fs::read_to_string(&engine_path).unwrap_or_default();
    assert_eq!(
        checked_starter, starter,
        "decks/beam_torsion/beam_torsion_0000.rad drifted from the rad writer"
    );
    assert_eq!(
        checked_engine, engine,
        "decks/beam_torsion/beam_torsion_0001.rad drifted from the rad writer"
    );
    assert!(
        probes_path.exists(),
        "decks/beam_torsion/probes.json must be checked in"
    );
    let _ = probes_json(&case, "none (prove-out deferred)");

    let Some(exe) = engine_exe() else {
        sonium_qa::rad::gate::report_missing_engine(CASE_ID);
    };

    // Run in a scratch dir so engine artifacts never pollute the repo.
    let work = std::env::temp_dir().join(format!("sonium_rad_beam_torsion_{}", std::process::id()));
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("beam_torsion_0000.rad"), &starter).unwrap();
    std::fs::write(work.join("beam_torsion_0001.rad"), &engine).unwrap();
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
        .and_then(|probe| {
            cantilever_native::parity_result(&case, CASE_ID, &output.engine, probe, !failed)
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

#[cfg(test)]
#[test]
fn torsion_preserves_original_geometry_couple_and_probe() {
    let case = beam_torsion_case();
    let mut total_force = [0.0; 3];
    let mut torque_x = 0.0;
    for load in &case.cloads {
        for &node in &case.node_sets[load.set].nodes {
            let point = case.mesh.nodes[node - 1];
            assert_eq!(point[0], 2.0);
            total_force[load.direction - 1] += load.force_scale_n;
            torque_x += point[1] * load.force_scale_n;
        }
    }
    assert_eq!(total_force, [0.0; 3]);
    assert!((torque_x - 500.0).abs() < 1e-12);
    let node: usize = case.probes[0]
        .channel
        .strip_prefix("DY/")
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(case.mesh.nodes[node - 1], [1.0, -0.025, -0.025]);
}
