//! Regression gate for the website thin-walled tube crush demo
//! (`qa-website-demos crush`), case `fem.website-crush-regression.v1`.
//!
//! The demo was previously visual-only: a mesh refinement from
//! 20 x 12 to 40 x 24 quads changed the fold mode and drove nodes
//! through the rigid wall because the driver had no wall contact.
//! This gate runs the shipped configuration and the original coarse
//! configuration and asserts the physical invariants that must hold
//! at any resolution: no wall penetration beyond the shell contact
//! offset, hourglass energy below the explicit-code 5% band, an
//! energy account that closes, and a mean crush force inside the
//! Alexander/Wierzbicki analytical band for a circular tube.

#![cfg(feature = "qa-binaries")]

// Rust guideline compliant 2026-02-21

use std::path::PathBuf;
use std::process::Command;

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_qa-website-demos"))
}

fn scratch(test: &str) -> PathBuf {
    let root =
        std::env::var("SONIUM_QA_SCRATCH").map_or_else(|_| std::env::temp_dir(), PathBuf::from);
    let dir = root.join(format!("website_crush_{test}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Runs one crush configuration and returns its parsed metrics.
fn run_crush(test: &str, envs: &[(&str, &str)]) -> serde_json::Value {
    let dir = scratch(test);
    let mut command = Command::new(bin());
    command.arg("crush").arg(&dir).envs(envs.iter().copied());
    let output = command.output().expect("run qa-website-demos crush");
    assert!(
        output.status.success(),
        "crush {test} failed: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    let text = std::fs::read_to_string(dir.join("crush_metrics.json"))
        .expect("crush_metrics.json written");
    serde_json::from_str(&text).expect("parse crush_metrics.json")
}

fn metric(metrics: &serde_json::Value, key: &str) -> f64 {
    metrics[key]
        .as_f64()
        .unwrap_or_else(|| panic!("metric {key}"))
}

/// Invariants every resolution must satisfy.
fn assert_invariants(metrics: &serde_json::Value) {
    // Shell midsurface never below the wall contact offset t/2.
    assert!(
        metric(metrics, "min_z_m") >= 0.0006 - 1.0e-9,
        "wall penetration: min_z {}",
        metric(metrics, "min_z_m")
    );
    // Deepest corrected wall penetration per pass stays microscopic
    // (observed ~2e-6 m; bound is 1/12 of the thickness).
    assert!(metric(metrics, "max_wall_penetration_m") < 1.0e-4);
    // Hourglass energy below 5% of plastic work (explicit-code band,
    // same acceptance as the Radioss golden `hgfrac` probe).
    assert!(
        metric(metrics, "hourglass_fraction") < 0.05,
        "hourglass fraction {}",
        metric(metrics, "hourglass_fraction")
    );
    // Energy account closes, within 5%: platen work plus the work
    // done by contact position projections (estimated with the step
    // force and reported separately, since one-sided projection was
    // found to inject energy invisibly) = plastic + kinetic +
    // contact-removed + hourglass. Elastic storage is the residual.
    let ext = metric(metrics, "external_work_J");
    let inputs = ext + metric(metrics, "self_work_J");
    let accounted = metric(metrics, "plastic_work_J")
        + metric(metrics, "kinetic_final_J")
        + metric(metrics, "wall_ke_removed_J")
        + metric(metrics, "self_ke_removed_J")
        + metric(metrics, "hourglass_energy_J");
    assert!(ext > 0.0 && metric(metrics, "plastic_work_J") > 0.0);
    assert!(
        (inputs - accounted).abs() / ext < 0.05,
        "energy closure: inputs {inputs}, accounted {accounted}"
    );
    // The plastic diagnostic must also track the balance-absorbed
    // energy (inputs minus kinetic/contact/hourglass) within a
    // factor 1.6: the 30 x 18 configuration once dissipated 3.1x
    // the platen work through projection injection.
    let absorbed = inputs
        - metric(metrics, "kinetic_final_J")
        - metric(metrics, "wall_ke_removed_J")
        - metric(metrics, "self_ke_removed_J")
        - metric(metrics, "hourglass_energy_J");
    let ratio = metric(metrics, "plastic_work_J") / absorbed;
    assert!(
        (0.5..=1.6).contains(&ratio),
        "plastic/balance-absorbed ratio {ratio}"
    );
}

/// Alexander/Wierzbicki mean crush force of a circular tube:
/// `Pm = 25.23 * sigma_flow * t^(3/2) * D^(1/2)`, with the flow
/// stress averaged between yield (250 MPa) and the hardened level
/// reached at fold strains (~350 MPa for the demo's 600 MPa linear
/// hardening), i.e. 300 MPa.
fn analytical_mean_force() -> f64 {
    25.23 * 300.0e6 * 0.0012f64.powf(1.5) * 0.09f64.sqrt()
}

#[test]
fn crush_shipped_configuration_satisfies_invariants() {
    let metrics = run_crush("shipped", &[]);
    assert_invariants(&metrics);
    // Mean crush force inside a factor 0.2-3.0 of the analytical
    // estimate. Alexander assumes axisymmetric concertina folding;
    // the demo seeds a 6-lobe diamond mode, whose mean force is
    // systematically lower (observed ratio ~0.3), so the band is a
    // sanity gate against order-of-magnitude error, not a fit.
    let estimate = analytical_mean_force();
    let mean = metric(&metrics, "mean_crush_force_N");
    assert!(
        (0.2 * estimate..=3.0 * estimate).contains(&mean),
        "mean crush force {mean} vs analytical {estimate}"
    );
}

#[test]
fn crush_coarse_configuration_agrees_on_global_response() {
    let shipped = run_crush("agree_shipped", &[]);
    let coarse = run_crush(
        "agree_coarse",
        &[("CRUSH_N_AROUND", "20"), ("CRUSH_N_AXIAL", "12")],
    );
    assert_invariants(&coarse);
    // Fold location is mesh-sensitive (the coarse element exceeds
    // the fold boundary-layer scale), but the global absorbed-energy
    // response must agree within 35%.
    let mean_shipped = metric(&shipped, "mean_crush_force_N");
    let mean_coarse = metric(&coarse, "mean_crush_force_N");
    let rel = (mean_coarse - mean_shipped).abs() / mean_shipped;
    assert!(
        rel < 0.35,
        "coarse/shipped mean force disagreement {rel:.3}: {mean_coarse} vs {mean_shipped}"
    );
}
