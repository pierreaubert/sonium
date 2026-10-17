//! End-to-end test for `speaker-product-worker`: a tiny product
//! radiation job (300 Hz mesh cap) through the shared Workbench job
//! store must complete and produce finite CEA-2034 curves.

// Rust guideline compliant 2026-02-21

use std::path::PathBuf;
use std::process::Command;

#[test]
fn product_worker_completes_a_small_radiation_job() {
    let scratch = std::env::temp_dir().join(format!(
        "speaker-product-worker-test-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).expect("scratch dir");
    let project_path = scratch.join("project.json");
    std::fs::write(
        &project_path,
        r#"{
          "schema_version": 1,
          "name": "Product worker smoke",
          "studies": [
            {"id": "product-radiation", "kind": "product_radiation", "fidelity": "distributed", "enabled": true}
          ],
          "port": {"radius_m": 0.025, "length_m": 0.15, "end_correction_m": 0.03, "box_volume_m3": 0.012},
          "speaker": {
            "driver": {"re_ohm": 5.8, "le_h": 0.0003, "bl_t_m": 4.5, "mms_kg": 0.008, "cms_m_per_n": 0.0012, "rms_n_s_per_m": 0.5, "sd_m2": 0.0087, "xmax_m": 0.005},
            "enclosure": {"kind": "sealed", "volume_m3": 0.012},
            "drive": {"voltage_vrms": 2.83, "start_hz": 50.0, "end_hz": 500.0, "points": 10},
            "product": {"cabinet_width_m": 0.17, "cabinet_height_m": 0.26, "cabinet_depth_m": 0.24, "driver_y_from_bottom_m": 0.09, "radiation_end_hz": 300.0, "radiation_points": 3}
          }
        }"#,
    )
    .expect("write project");
    let jobs_root = scratch.join("jobs");
    let output = Command::new(env!("CARGO_BIN_EXE_speaker-product-worker"))
        .arg("--jobs-root")
        .arg(&jobs_root)
        .args(["run", "--project"])
        .arg(&project_path)
        .args(["--study", "product-radiation"])
        .output()
        .expect("run product worker");
    assert!(
        output.status.success(),
        "worker failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let record: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("record json");
    assert_eq!(record["state"], "completed", "record: {record}");
    let job_id = record["id"].as_str().expect("job id");
    let result_text =
        std::fs::read_to_string(jobs_root.join(job_id).join("result.json"))
            .expect("result file");
    let result: serde_json::Value =
        serde_json::from_str(&result_text).expect("result json");
    assert_eq!(result["frequencies_hz"].as_array().map(Vec::len), Some(3));
    for curve in ["on_axis_db", "listening_window_db", "sound_power_db"] {
        let values = result[curve].as_array().expect("curve present");
        assert_eq!(values.len(), 3, "{curve} length");
        assert!(
            values
                .iter()
                .all(|value| value.as_f64().is_some_and(f64::is_finite)),
            "{curve} must be finite: {values:?}"
        );
    }
    std::fs::remove_dir_all(&scratch).expect("cleanup");
    let _: PathBuf = project_path;
}
