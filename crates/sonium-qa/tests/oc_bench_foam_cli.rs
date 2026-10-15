//! CLI integration tests for the foam `qa-oc-bench` arm (Unit 3).
//!
//! Covers the prescribed-motion replay arm (`support/foam_c3.rs`) end to
//! end: deck resolution, native replay, probe gates, and bundle schema.
//! Engine-free tests run everywhere; the engine-arm test runs only when
//! a vendor engine is configured and otherwise skips loudly (the
//! `oc_bench_cli` precedent).

#![cfg(feature = "qa-binaries")]

// Rust guideline compliant 2026-02-21

use sonium_bench::oc_compare::Side;
use std::path::PathBuf;
use std::process::Command;

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_qa-oc-bench"))
}

fn base(test: &str) -> (Command, PathBuf) {
    let dir = std::env::temp_dir().join(format!("oc_bench_{test}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bundle = dir.join("bundle.json");
    let mut command = Command::new(bin());
    command
        .arg("--emit-bundle")
        .arg(&bundle)
        .arg("--scratch-root")
        .arg(dir.join("scratch"))
        .env_remove("OPENCOURANT_EXE")
        .env_remove("RADISS_EXE");
    (command, bundle)
}

fn run_sonium_quick(deck: &str, tag: &str) {
    let (mut command, bundle) = base(tag);
    let output = command
        .arg("--deck")
        .arg(deck)
        .arg("--quick")
        .arg("--sonium-only")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "stdout: {stdout}");
    assert!(stdout.contains("SUCCESS: qa-oc-bench"), "stdout: {stdout}");
    let text = std::fs::read_to_string(&bundle).unwrap();
    let parsed = sonium_bench::oc_compare::parse_bundle_json(&text).unwrap();
    assert_eq!(parsed.schema, sonium_bench::oc_compare::BUNDLE_SCHEMA);
    assert_eq!(parsed.rows.len(), 1);
    assert_eq!(parsed.rows[0].side, Side::Sonium);
    assert_eq!(parsed.rows[0].deck, deck);
    assert_eq!(parsed.rows[0].flavor, "native");
    assert!(parsed.rows[0].correctness_passed);
    assert_eq!(parsed.rows[0].dof, 24);
    assert_eq!(parsed.protocol.repeats, 1);
    assert_eq!(parsed.protocol.warmup, 0);
}

#[test]
fn sonium_quick_foam33_emits_valid_bundle() {
    run_sonium_quick("foam33_cube", "foam33_quick");
}

#[test]
fn sonium_quick_foam76_compression_passes() {
    run_sonium_quick("foam76_compression", "foam76c_quick");
}

#[test]
fn sonium_quick_foam76_tension_passes() {
    run_sonium_quick("foam76_tension", "foam76t_quick");
}

#[test]
fn engine_quick_foam33_when_configured() {
    let exe = ["OPENCOURANT_EXE", "RADISS_EXE"]
        .iter()
        .filter_map(|name| std::env::var(name).ok())
        .find(|value| !value.trim().is_empty());
    let Some(exe) = exe else {
        println!("SKIP engine_quick_foam33_when_configured: no vendor engine configured");
        return;
    };
    let dir = std::env::temp_dir().join(format!("oc_bench_foam33_engine_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bundle = dir.join("bundle.json");
    let output = Command::new(bin())
        .arg("--deck")
        .arg("foam33_cube")
        .arg("--emit-bundle")
        .arg(&bundle)
        .arg("--scratch-root")
        .arg(dir.join("scratch"))
        .arg("--exe")
        .arg(exe)
        .arg("--quick")
        .arg("--engine-only")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "stdout: {stdout}\nstderr: {stderr}"
    );
    assert!(stdout.contains("SUCCESS: qa-oc-bench"), "stdout: {stdout}");
    let text = std::fs::read_to_string(&bundle).unwrap();
    let parsed = sonium_bench::oc_compare::parse_bundle_json(&text).unwrap();
    assert_eq!(parsed.rows.len(), 1);
    assert_eq!(parsed.rows[0].side, Side::Engine);
    assert!(parsed.rows[0].correctness_passed);
}
