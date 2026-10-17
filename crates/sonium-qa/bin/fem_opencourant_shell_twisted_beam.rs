//! External reference health and native parity gate for shell-twisted-beam.
//!
//! MacNeal-Harder thin twisted beam (L = 12 in, width 1.1 in, t = 0.32 in, E = 29e6 psi, nu =
//! 0.22, 90-degree twist, 1 lbf tip load) as a QEPH quad shell deck in SI units. This is the
//! head-to-head C3 cross-check of the warp-aware MITC4 membrane fix (Phase-B Worker-3
//! P1-B-W3-0): the legacy flat-facet membrane diverged on this warped-quad geometry while the
//! covariant membrane converges to the in-plane target 0.005424 in within 1% at 32x8 (see
//! `fem_shell_twisted_beam.rs`). The probe replays the in-plane tip deflection, the direction
//! that discriminates the fix.
//!
//! Reference: MacNeal, R. H. and Harder, R. L. (1985), "A proposed standard set of problems to
//! test finite element accuracy", FEAD 1.
//!
//! Verifies the checked-in decks match the `sonium_qa::rad` extended writer, then runs the
//! starter and engine when `OPENCOURANT_EXE` (or legacy `RADISS_EXE`) is set and compares the
//! tip deflection against the MacNeal target. Decks regenerate with `REGEN_DECKS=1`.
//!
//! Missing references emit a structured optional skip, or fail when
//! SONIUM_QA_REQUIRE_OPENCOURANT=1. External probe health is recorded separately; release
//! success also requires a matching native solve and full named phenomenon.

// Rust guideline compliant 2026-02-21

use sonium_qa::rad::{
    OpenCourantCase, RadBoundaryCondition, RadConcentratedLoad, RadElasticMaterial,
    RadExtendedCase, RadFunction, RadMesh, RadNodeSet, RadProbe, RadShellMesh, RadShellProperty,
    RadThRequest, assert_probe_close, assert_probes_payload_eq, engine_exe, extended_probes_json,
    run_extended_case, shel_channel, write_extended_engine, write_extended_starter,
};
use std::path::PathBuf;

#[path = "support/twisted_native.rs"]
pub mod twisted_native;

/// MacNeal-Harder inputs converted once to SI (m, Pa, N).
const LENGTH_M: f64 = 12.0 * 0.0254;
const WIDTH_M: f64 = 1.1 * 0.0254;
const THICK_M: f64 = 0.32 * 0.0254;
const YOUNG_PA: f64 = 29.0e6 * 6_894.757_293_178;
const POISSON: f64 = 0.22;
const DENSITY: f64 = 7_800.0;
const LOAD_N: f64 = 4.448_221_615_260_5;
/// In-plane tip target 0.005424 in, in metres.
const TARGET_IN_M: f64 = 0.005424 * 0.0254;

/// Twisted quad mesh matching the in-process QA convention: the root
/// section spans Y and twists 90 degrees about X toward the tip, so the
/// tip width lies along Z and the in-plane load is +Z there.
struct TwistedMesh {
    nodes: Vec<[f64; 3]>,
    quads: Vec<[usize; 4]>,
    root: Vec<usize>,
    tip_center: usize,
}

fn twisted_beam_mesh(n_x: usize, n_y: usize) -> TwistedMesh {
    assert!(
        n_y.is_multiple_of(2),
        "tip center needs an even across count"
    );
    let total = std::f64::consts::FRAC_PI_2;
    let mut nodes = Vec::with_capacity((n_x + 1) * (n_y + 1));
    for ix in 0..=n_x {
        let x = LENGTH_M * ix as f64 / n_x as f64;
        let twist = total * ix as f64 / n_x as f64;
        for iy in 0..=n_y {
            let s = WIDTH_M * (iy as f64 / n_y as f64 - 0.5);
            nodes.push([x, s * twist.cos(), s * twist.sin()]);
        }
    }
    let id = |ix: usize, iy: usize| ix * (n_y + 1) + iy + 1;
    let mut quads = Vec::with_capacity(n_x * n_y);
    for ix in 0..n_x {
        for iy in 0..n_y {
            quads.push([
                id(ix, iy),
                id(ix + 1, iy),
                id(ix + 1, iy + 1),
                id(ix, iy + 1),
            ]);
        }
    }
    let root: Vec<usize> = (0..=n_y).map(|iy| id(0, iy)).collect();
    let tip_center = id(n_x, n_y / 2);
    TwistedMesh {
        nodes,
        quads,
        root,
        tip_center,
    }
}

/// Build the twisted-beam shell comparison case.
///
/// A 32x6 QEPH cantilever carries the 1 lbf-equivalent tip load in the
/// in-plane (+Z at the tip) direction under a 20 ms ramp with a 40 ms
/// hold. The first bending mode (71 Hz, 14 ms period) rings hard on a
/// fast ramp -- a 5 ms ramp read 1.75x high at the first transient peak
/// -- so the probe averages the hold over two full periods. The deck
/// mesh matches the in-process convergence point (within 1% of target
/// in-process), and the 5% probe tolerance follows the Scordelis and
/// clamped-plate C3 precedent for QEPH-vs-theory twin gaps.
fn shell_twisted_beam_case() -> RadExtendedCase {
    let mesh = twisted_beam_mesh(32, 6);
    let base = OpenCourantCase {
        name: "shell_twisted_beam".to_string(),
        title: "sonium_qa rad twisted beam quasi-static tip load".to_string(),
        mesh: RadMesh {
            nodes: mesh.nodes.clone(),
            bricks: Vec::new(),
        },
        material: RadElasticMaterial {
            youngs_modulus_pa: YOUNG_PA,
            poisson_ratio: POISSON,
            density_kg_per_m3: DENSITY,
        },
        node_sets: vec![
            RadNodeSet {
                name: "clamped root".to_string(),
                nodes: mesh.root,
            },
            RadNodeSet {
                name: "tip center".to_string(),
                nodes: vec![mesh.tip_center],
            },
        ],
        bcs: vec![RadBoundaryCondition {
            name: "clamp root".to_string(),
            set: 0,
            lock: [true, true, true, true, true, true],
        }],
        functions: vec![RadFunction {
            name: "ramp 20 ms then hold".to_string(),
            points: vec![(0.0, 0.0), (0.02, 1.0), (0.06, 1.0)],
        }],
        cloads: vec![RadConcentratedLoad {
            name: "in-plane tip load".to_string(),
            set: 1,
            direction: 3,
            function: 0,
            force_scale_n: LOAD_N,
        }],
        walls: Vec::new(),
        th_nodes: vec![mesh.tip_center],
        probes: vec![
            RadProbe {
                label: "in-plane tip DZ".to_string(),
                channel: format!("DZ/{}", mesh.tip_center),
                expected: TARGET_IN_M,
                abs_tol: 1e-9,
                // Measured engine-vs-analytical probe deviation = 6.4e-3; gate at ~3x.
                rel_tol: 0.02,
            },
            RadProbe {
                label: "root shell THIC".to_string(),
                channel: shel_channel("THIC", 1),
                expected: THICK_M,
                abs_tol: 1e-9,
                rel_tol: 1e-6,
            },
        ],
        stop_time_s: 0.06,
        th_period_s: 1e-4,
        anim_period_s: 6e-3,
    };
    let mut case = RadExtendedCase::wrap_elastic(base);
    case.shells = RadShellMesh {
        quads: mesh.quads,
        tris: Vec::new(),
    };
    case.shell_prop = Some(RadShellProperty::qeph("beam section", THICK_M));
    case.th_shell = RadThRequest::shell_resultants(vec![1]);
    case
}

fn decks_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("decks/shell_twisted_beam")
}

const CASE_ID: &str = "fem.opencourant.shell-twisted-beam.native-parity.v1";

fn main() {
    let case = shell_twisted_beam_case();
    case.validate().expect("shell_twisted_beam must validate");
    if std::env::var("REGEN_DECKS").as_deref() == Ok("1") {
        let dir = decks_dir();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("shell_twisted_beam_0000.rad"),
            write_extended_starter(&case),
        )
        .unwrap();
        std::fs::write(
            dir.join("shell_twisted_beam_0001.rad"),
            write_extended_engine(&case),
        )
        .unwrap();
        std::fs::write(
            dir.join("probes.json"),
            extended_probes_json(&case, "none (prove-out deferred)"),
        )
        .unwrap();
        println!("regenerated decks for shell_twisted_beam");
        return;
    }
    let dir = decks_dir();
    let starter_path = dir.join("shell_twisted_beam_0000.rad");
    let engine_path = dir.join("shell_twisted_beam_0001.rad");
    let probes_path = dir.join("probes.json");

    // No hand-built decks: the checked-in trio must match the writer.
    let starter = write_extended_starter(&case);
    let engine = write_extended_engine(&case);
    let checked_starter = std::fs::read_to_string(&starter_path).unwrap_or_default();
    let checked_engine = std::fs::read_to_string(&engine_path).unwrap_or_default();
    let checked_probes = std::fs::read_to_string(&probes_path).unwrap_or_default();
    assert_eq!(
        checked_starter, starter,
        "decks/shell_twisted_beam/shell_twisted_beam_0000.rad drifted from the rad writer"
    );
    assert_eq!(
        checked_engine, engine,
        "decks/shell_twisted_beam/shell_twisted_beam_0001.rad drifted from the rad writer"
    );
    assert_probes_payload_eq(
        &checked_probes,
        &extended_probes_json(&case, "none (prove-out deferred)"),
        "decks/shell_twisted_beam/probes.json drifted from the rad writer",
    );

    let Some(exe) = engine_exe() else {
        sonium_qa::rad::gate::report_missing_engine(CASE_ID);
    };

    // Run in a scratch dir so engine artifacts never pollute the repo.
    let work = std::env::temp_dir().join(format!(
        "sonium_rad_shell_twisted_beam_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("shell_twisted_beam_0000.rad"), &starter).unwrap();
    std::fs::write(work.join("shell_twisted_beam_0001.rad"), &engine).unwrap();
    let output = run_extended_case(&work, &exe, &case).unwrap_or_else(|error| {
        sonium_qa::rad::gate::publish_and_exit(sonium_qa::rad::gate::execution_failure_result(
            CASE_ID,
            &error.to_string(),
        ));
    });
    println!("engine: {}", output.engine);
    let mut failed = false;
    let mut values = Vec::new();
    for (index, probe) in case.base.probes.iter().enumerate() {
        // The tip DZ averages the hold over two first-mode periods (28 ms
        // window); the raw trace still carries the ramp transient.
        let actual = if index == 0 {
            output.history.channel(&probe.channel).and_then(|values| {
                twisted_native::hold_mean(&output.history.times, values)
                    .map_err(|error| sonium_qa::rad::RadError::Parse(error.to_string()))
            })
        } else {
            output.history.final_value(&probe.channel)
        };
        match actual {
            Ok(value) => {
                values.push(value);
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
    let result = twisted_native::parity_result(&case, CASE_ID, &output.engine, &values, !failed)
        .unwrap_or_else(|error| {
            sonium_qa::QaResult::failed(
                CASE_ID,
                format!("native twisted-shell validation failed: {error}"),
            )
            .requested_backend("cpu+opencourant")
            .actual_backend("cpu+opencourant")
            .metric("external_reference_executed", 1.0)
            .metric("native_parity_executed", 0.0)
        });
    sonium_qa::rad::gate::publish_and_exit(result);
}
