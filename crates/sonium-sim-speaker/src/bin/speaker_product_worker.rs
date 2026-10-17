//! `speaker-product-worker`: product-radiation jobs for the
//! Workbench job service.
//!
//! The lumped studies live in `workbench-worker`
//! (`sonium-sim-speaker-modeling`); this binary owns the BEM product
//! path so the unified front end can route Product Radiation studies
//! to the crate that owns the solver:
//!
//! ```text
//! speaker-product-worker --jobs-root DIR run --project project.json --study product-radiation
//! ```
//!
//! The job executes through the same `JobStore` contract (hashed
//! immutable snapshot, persisted lifecycle, result document). The
//! model is a rigid-piston cabinet BEM driven by the lumped driver
//! velocity from the document's speaker section; only the workflow
//! driver radiates (the tweeter aperture is an inactive rigid
//! placeholder), which the result document states explicitly.

// Rust guideline compliant 2026-02-21

use std::path::PathBuf;

use sha2::{Digest, Sha256};
use sonium_sim_speaker::cea2034_output::{compute_spinorama_metrics, generate_eval_points};
use sonium_sim_speaker::model::{
    CabinetDimensions, DriverConfig, LoudspeakerModel, SeparatorConfig, SimulationConfig,
    SpeakerMaterialAssignment, TransferPoint,
};
use sonium_sim_speaker::simulation::{SolverConfig, run_simulation_with_config};
use sonium_sim_speaker_modeling::job_runner::{WorkbenchUiProject, evaluate_lumped_driver};
use sonium_sim_speaker_modeling::job_service::{JobState, JobStore};
use sonium_sim_speaker_modeling::workbench::WorkbenchStudyKind;

/// Highest radiated frequency this worker will mesh (λ/8 BEM mesh).
const MAX_RADIATION_HZ: f64 = 4_000.0;

fn main() {
    if let Err(error) = run() {
        eprintln!("speaker-product-worker failed: {error}");
        std::process::exit(1);
    }
}

fn argument(args: &[String], flag: &str) -> Result<String, Box<dyn std::error::Error>> {
    let position = args
        .iter()
        .position(|arg| arg == flag)
        .ok_or_else(|| format!("missing {flag} argument"))?;
    args.get(position + 1)
        .cloned()
        .ok_or_else(|| format!("missing value for {flag}").into())
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let jobs_root = PathBuf::from(argument(&args, "--jobs-root")?);
    let store = JobStore::open(&jobs_root)?;
    match args
        .iter()
        .find(|arg| arg.as_str() == "run" || arg.as_str() == "list")
        .map(String::as_str)
    {
        Some("list") => {
            println!("{}", serde_json::to_string_pretty(&store.list()?)?);
            Ok(())
        }
        Some("run") => {
            let project_path = PathBuf::from(argument(&args, "--project")?);
            let study_id = argument(&args, "--study")?;
            let job_id = argument(&args, "--job-id").ok();
            let snapshot = std::fs::read(&project_path)?;
            let document: WorkbenchUiProject = serde_json::from_slice(&snapshot)?;
            let study = document
                .studies
                .iter()
                .find(|entry| entry.id == study_id)
                .ok_or_else(|| format!("project has no study with id {study_id:?}"))?;
            if study.kind != WorkbenchStudyKind::ProductRadiation {
                return Err(format!(
                    "speaker-product-worker only executes product_radiation studies, got {study_id:?}"
                )
                .into());
            }
            let mut hasher = Sha256::new();
            hasher.update(&snapshot);
            let input_hash = format!("{:x}", hasher.finalize());
            let job_id = job_id.unwrap_or_else(|| format!("{study_id}-{}", &input_hash[..12]));
            let record = store.submit(&job_id, &document.name, &study_id, &snapshot)?;
            if record.state != JobState::Queued {
                println!("{}", serde_json::to_string_pretty(&record)?);
                return Ok(());
            }
            store.transition(&record.id, JobState::Running)?;
            store.beat(&record.id)?;
            match execute(&document) {
                Ok(result) => {
                    store.write_result(&record.id, &result)?;
                    store.transition(&record.id, JobState::Completed)?;
                }
                Err(message) => {
                    store.write_result(&record.id, &serde_json::json!({ "error": message }))?;
                    store.transition(&record.id, JobState::Failed)?;
                }
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&store.load(&record.id)?)?
            );
            Ok(())
        }
        _ => Err(
            "usage: speaker-product-worker --jobs-root DIR run --project FILE --study ID | list"
                .into(),
        ),
    }
}

fn execute(document: &WorkbenchUiProject) -> Result<serde_json::Value, String> {
    let speaker = document.speaker.as_ref().ok_or_else(|| {
        "speaker workflow section is missing; product radiation needs it".to_owned()
    })?;
    let product = speaker.product.as_ref().ok_or_else(|| {
        "speaker.product section is missing; product radiation needs cabinet and radiation settings".to_owned()
    })?;
    if !(product.radiation_end_hz.is_finite()
        && product.radiation_end_hz > speaker.drive.start_hz
        && product.radiation_end_hz <= MAX_RADIATION_HZ)
    {
        return Err(format!(
            "radiation_end_hz must be in (drive start, {MAX_RADIATION_HZ:.0}] Hz, got {}",
            product.radiation_end_hz
        ));
    }
    if !(2..=25).contains(&product.radiation_points) {
        return Err(format!(
            "radiation_points must be in 2..=25, got {}",
            product.radiation_points
        ));
    }
    let ratio = (product.radiation_end_hz / speaker.drive.start_hz)
        .powf(1.0 / (product.radiation_points - 1) as f64);
    let frequencies: Vec<f64> = (0..product.radiation_points)
        .map(|i| speaker.drive.start_hz * ratio.powi(i as i32))
        .collect();
    let lumped = evaluate_lumped_driver(document, &frequencies)?;
    let transfer: Vec<TransferPoint> = lumped
        .iter()
        .map(|point| TransferPoint {
            // The lumped chain uses the e^{+i\u{03c9}t} convention; the
            // BEM pressure phasor uses e^{-i\u{03c9}t}, hence the
            // conjugated (negated) phase.
            freq: point.frequency_hz,
            magnitude: point.velocity_rms_m_per_s.norm() * std::f64::consts::SQRT_2,
            phase_deg: -point.velocity_rms_m_per_s.arg().to_degrees(),
        })
        .collect();
    let inactive: Vec<TransferPoint> = frequencies
        .iter()
        .map(|&freq| TransferPoint {
            freq,
            magnitude: 0.0,
            phase_deg: 0.0,
        })
        .collect();
    let radius = (speaker.driver.sd_m2 / std::f64::consts::PI).sqrt();
    let woofer_y = product.driver_y_from_bottom_m - product.cabinet_height_m / 2.0;
    let tweeter_y = product.cabinet_height_m / 2.0 - 0.035;
    let model = LoudspeakerModel {
        cabinet: CabinetDimensions {
            height: product.cabinet_height_m,
            width: product.cabinet_width_m,
            depth: product.cabinet_depth_m,
        },
        woofer: DriverConfig {
            radius,
            x_offset: 0.0,
            y_offset: woofer_y,
            piston_velocity: 0.0,
            transfer_function: Some(transfer),
            driver_model: None,
            cone: None,
        },
        tweeter: DriverConfig {
            radius: 0.013,
            x_offset: 0.0,
            y_offset: tweeter_y,
            piston_velocity: 0.0,
            transfer_function: Some(inactive),
            driver_model: None,
            cone: None,
        },
        additional_drivers: Vec::new(),
        product_design: None,
        separator: SeparatorConfig {
            // Internal plate midway between the two apertures.
            y_position: (woofer_y + tweeter_y) / 2.0,
            thickness: 0.018,
        },
        waveguide: None,
        materials: SpeakerMaterialAssignment::default(),
        simulation: SimulationConfig {
            freq_start: frequencies[0],
            freq_end: product.radiation_end_hz,
            freq_points: frequencies.len(),
            speed_of_sound: 343.0,
            air_density: 1.204,
            max_mesh_frequency: Some(product.radiation_end_hz),
        },
    };
    let eval_points = generate_eval_points([0.0, 0.0, product.cabinet_depth_m]);
    let solver_config = SolverConfig {
        restart: 200,
        recycle_subspace_size: 50,
        ..SolverConfig::default()
    };
    let result = run_simulation_with_config(&model, &eval_points, &solver_config)
        .map_err(|error| error.to_string())?;
    let metrics = compute_spinorama_metrics(&result).map_err(|error| {
        format!(
            "spinorama aggregation failed: {error}; the CEA-2034 preference score needs a wider radiation grid (widen radiation_end_hz / radiation_points)"
        )
    })?;
    Ok(serde_json::json!({
        "study_kind": "product_radiation",
        "model": "rigid-piston cabinet BEM with lumped driver velocity; single active driver (tweeter aperture inactive)",
        "frequencies_hz": result.frequencies,
        "on_axis_db": metrics.on_axis,
        "listening_window_db": metrics.listening_window,
        "early_reflections_db": metrics.early_reflections,
        "sound_power_db": metrics.sound_power,
        "estimated_in_room_db": metrics.pir,
        "preference_score": metrics.preference_score,
    }))
}
