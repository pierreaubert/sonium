//! CLI integration tests for `qa-oc-bench` (lane 6A harness).
//!
//! Engine-free tests run everywhere; the engine-arm test runs only
//! when a vendor engine is configured and otherwise skips loudly
//! (the `fem_opencourant_*` legs follow the same precedent).

#![cfg(feature = "qa-binaries")]

// Rust guideline compliant 2026-02-21

use sonium_bench::oc_compare::Side;
use std::path::PathBuf;
use std::process::Command;

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_qa-oc-bench"))
}

fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("oc_bench_{test}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn base(test: &str) -> (Command, PathBuf) {
    let dir = scratch(test);
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

#[test]
fn help_lists_frozen_flags() {
    let output = Command::new(bin()).arg("--help").output().unwrap();
    assert!(output.status.success());
    let text = String::from_utf8_lossy(&output.stdout);
    for flag in [
        "--deck",
        "--emit-bundle",
        "--side",
        "--threads",
        "--repeats",
        "--quick",
        "--sonium-only",
        "--engine-only",
        "--flavor",
        "--mpi-ranks",
    ] {
        assert!(text.contains(flag), "help misses {flag}");
    }
}

#[test]
fn unknown_deck_fails_closed() {
    let (mut command, _) = base("unknown_deck");
    let output = command
        .arg("--deck")
        .arg("no_such_deck")
        .arg("--quick")
        .arg("--sonium-only")
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("FAILED: qa-oc-bench"), "stdout: {stdout}");
}

#[test]
fn contradictory_sides_fail_closed() {
    let (mut command, _) = base("contradictory");
    let output = command
        .arg("--deck")
        .arg("taylor_bar")
        .arg("--sonium-only")
        .arg("--engine-only")
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("FAILED: qa-oc-bench"), "stdout: {stdout}");
}

#[test]
fn engine_arm_without_exe_fails_closed() {
    let (mut command, _) = base("no_exe");
    let output = command
        .arg("--deck")
        .arg("taylor_bar")
        .arg("--quick")
        .arg("--engine-only")
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stdout.contains("FAILED: qa-oc-bench"), "stdout: {stdout}");
    assert!(stderr.contains("--exe"), "stderr: {stderr}");
}

#[test]
fn sonium_arm_rejects_miniqa_cleanly() {
    let (mut command, _) = base("miniqa_sonium");
    let output = command
        .arg("--deck")
        .arg("LOIS/LOI2/BESH/input_44")
        .arg("--quick")
        .arg("--sonium-only")
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("C3 decks only"), "stderr: {stderr}");
}

#[test]
fn sonium_arm_reports_unsupported_card_reason() {
    // N-deck harness: axisym resolves as C3 but has no native arm
    // (Unit 1 proposal: axisymmetric kernel), so the failure names
    // the unsupported card (fast, no timing).
    let (mut command, _) = base("card_reason");
    let output = command
        .arg("--deck")
        .arg("axisym_cylinder")
        .arg("--quick")
        .arg("--sonium-only")
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stdout.contains("FAILED: qa-oc-bench"), "stdout: {stdout}");
    assert!(stderr.contains("/ANALY"), "stderr: {stderr}");
}

#[test]
fn nonconforming_deck_dir_fails_closed_with_reason() {
    // N-deck harness: crash_box_original carries no C3-named starter
    // pair, so resolution names the missing artifact (no timing).
    let (mut command, _) = base("crash_reason");
    let output = command
        .arg("--deck")
        .arg("crash_box_original")
        .arg("--quick")
        .arg("--sonium-only")
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stdout.contains("FAILED: qa-oc-bench"), "stdout: {stdout}");
    assert!(stderr.contains("misses"), "stderr: {stderr}");
    assert!(
        stderr.contains("crash_box_original_0000.rad"),
        "stderr: {stderr}"
    );
}

#[test]
fn sonium_quick_shell_clamped_plate_emits_valid_bundle() {
    // Unit 4 flip: the shell arm now serves shell decks (slow:
    // thin-shell dt bound needs ~100k steps even for --quick).
    let (mut command, bundle) = base("shell_quick");
    let output = command
        .arg("--deck")
        .arg("shell_clamped_plate")
        .arg("--quick")
        .arg("--sonium-only")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "stdout: {stdout}");
    assert!(stdout.contains("SUCCESS: qa-oc-bench"), "stdout: {stdout}");
    let text = std::fs::read_to_string(&bundle).unwrap();
    let parsed = sonium_bench::oc_compare::parse_bundle_json(&text).unwrap();
    assert_eq!(parsed.rows.len(), 1);
    assert_eq!(parsed.rows[0].side, Side::Sonium);
    assert_eq!(parsed.rows[0].deck, "shell_clamped_plate");
    assert!(parsed.rows[0].correctness_passed);
}

#[test]
fn sonium_quick_cantilever_emits_valid_bundle() {
    // Unit 1: the EAS-linear path serves the bending decks.
    let (mut command, bundle) = base("cantilever_quick");
    let output = command
        .arg("--deck")
        .arg("cantilever")
        .arg("--quick")
        .arg("--sonium-only")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "stdout: {stdout}");
    assert!(stdout.contains("SUCCESS: qa-oc-bench"), "stdout: {stdout}");
    let text = std::fs::read_to_string(&bundle).unwrap();
    let parsed = sonium_bench::oc_compare::parse_bundle_json(&text).unwrap();
    assert_eq!(parsed.rows.len(), 1);
    assert_eq!(parsed.rows[0].side, Side::Sonium);
    assert_eq!(parsed.rows[0].deck, "cantilever");
    assert!(parsed.rows[0].correctness_passed);
}

#[test]
fn sonium_quick_foam76_compression_emits_valid_bundle() {
    // Unit 3 flip: the foam replay arm serves the foam decks.
    let (mut command, bundle) = base("foam_quick");
    let output = command
        .arg("--deck")
        .arg("foam76_compression")
        .arg("--quick")
        .arg("--sonium-only")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "stdout: {stdout}");
    assert!(stdout.contains("SUCCESS: qa-oc-bench"), "stdout: {stdout}");
    let text = std::fs::read_to_string(&bundle).unwrap();
    let parsed = sonium_bench::oc_compare::parse_bundle_json(&text).unwrap();
    assert_eq!(parsed.rows.len(), 1);
    assert_eq!(parsed.rows[0].side, Side::Sonium);
    assert_eq!(parsed.rows[0].deck, "foam76_compression");
    assert!(parsed.rows[0].correctness_passed);
    assert_eq!(parsed.rows[0].dof, 24);
}

#[test]
fn sonium_quick_taylor_emits_valid_bundle() {
    let (mut command, bundle) = base("taylor_quick");
    let output = command
        .arg("--deck")
        .arg("taylor_bar")
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
    assert_eq!(parsed.rows[0].deck, "taylor_bar");
    assert_eq!(parsed.rows[0].flavor, "native");
    assert!(parsed.rows[0].correctness_passed);
    assert_eq!(parsed.protocol.repeats, 1);
    assert_eq!(parsed.protocol.warmup, 0);
}

#[test]
fn engine_quick_taylor_when_configured() {
    let exe = ["OPENCOURANT_EXE", "RADISS_EXE"]
        .iter()
        .filter_map(|name| std::env::var(name).ok())
        .find(|value| !value.trim().is_empty());
    let Some(exe) = exe else {
        println!("SKIP engine_quick_taylor_when_configured: no vendor engine configured");
        return;
    };
    let dir = scratch("taylor_engine");
    let bundle = dir.join("bundle.json");
    let output = Command::new(bin())
        .arg("--deck")
        .arg("taylor_bar")
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
