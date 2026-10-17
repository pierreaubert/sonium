//! Execution side of the Workbench job service.
//!
//! A front end submits its project document once; this module owns
//! what happens next. The document is the workbench UI project (the
//! solver-side importer for that payload): a study tree using the
//! canonical study kinds plus the port settings the UI carries. Jobs
//! execute through [`JobStore`], so every run leaves a hashed input
//! snapshot, a persisted lifecycle, and a result document.
//!
//! Executors exist where the payload fully determines the physics:
//! the lumped enclosure system, the port analysis, and the migrated
//! Driver Lab workflows (lumped driver-linear sweep and large-signal
//! excursion levels) driven by the document's speaker section.
//! Product radiation is routed to `speaker-product-worker` in
//! `sonium-sim-speaker`, which owns the BEM product path. Study kinds
//! whose inputs the UI document does not carry fail the job with a
//! named gap instead of silently substituting a lower-fidelity
//! result.

// Rust guideline compliant 2026-02-21

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::job_service::{JobRecord, JobState, JobStore, JobStoreError};
use crate::port_flow::duct1d::{PortFidelity, cutoff_frequency_hz, select_port_fidelity};
use crate::workbench::{WORKBENCH_SCHEMA_VERSION, WorkbenchStudy, WorkbenchStudyKind};
use crate::{AcousticMedium, EnclosureSpec, PortSpec};

/// Port settings carried by the workbench UI project document.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WorkbenchUiPort {
    /// Port radius in metres.
    pub radius_m: f64,
    /// Port length in metres.
    pub length_m: f64,
    /// Total end correction in metres.
    pub end_correction_m: f64,
    /// Enclosure volume in cubic metres.
    pub box_volume_m3: f64,
}

/// Lumped driver parameters for the speaker workflow studies.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UiDriver {
    /// Voice-coil DC resistance (Ω).
    pub re_ohm: f64,
    /// Voice-coil inductance (H).
    pub le_h: f64,
    /// Force factor Bl (T·m).
    pub bl_t_m: f64,
    /// Moving mass (kg).
    pub mms_kg: f64,
    /// Suspension compliance (m/N).
    pub cms_m_per_n: f64,
    /// Mechanical resistance (N·s/m).
    pub rms_n_s_per_m: f64,
    /// Effective piston area (m²).
    pub sd_m2: f64,
    /// Linear excursion limit, one-way (m).
    pub xmax_m: f64,
}

/// Enclosure family for the speaker workflow studies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UiEnclosureKind {
    /// Sealed enclosure.
    Sealed,
    /// Vented enclosure using the project port settings.
    Vented,
}

/// Enclosure settings for the speaker workflow studies.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UiEnclosure {
    /// Sealed or vented.
    pub kind: UiEnclosureKind,
    /// Net enclosure volume (m³).
    pub volume_m3: f64,
}

/// Drive and sweep settings for the speaker workflow studies.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UiDrive {
    /// Terminal drive voltage (V RMS).
    pub voltage_vrms: f64,
    /// Sweep start frequency (Hz).
    pub start_hz: f64,
    /// Sweep end frequency (Hz).
    pub end_hz: f64,
    /// Number of logarithmically spaced sweep points.
    pub points: usize,
}

/// Cabinet and radiation settings for product-radiation jobs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UiProduct {
    /// Cabinet width (m).
    pub cabinet_width_m: f64,
    /// Cabinet height (m).
    pub cabinet_height_m: f64,
    /// Cabinet depth (m).
    pub cabinet_depth_m: f64,
    /// Driver centre height above the cabinet bottom (m).
    pub driver_y_from_bottom_m: f64,
    /// Highest radiated frequency (Hz), capped by the product worker.
    pub radiation_end_hz: f64,
    /// Number of logarithmically spaced radiated frequencies.
    pub radiation_points: usize,
}

/// Speaker workflow payload: the migrated Driver Lab / Speaker Lab
/// inputs that the lumped and product executors need.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UiSpeakerWorkflow {
    /// Lumped driver parameters.
    pub driver: UiDriver,
    /// Enclosure settings.
    pub enclosure: UiEnclosure,
    /// Drive and sweep settings.
    pub drive: UiDrive,
    /// Product-radiation settings, when the study is used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub product: Option<UiProduct>,
}

/// The workbench UI project document accepted by the job service.
///
/// Study entries deserialize directly into the canonical
/// [`WorkbenchStudy`] type, so study kinds, fidelity labels, and ids
/// cannot drift between the front end and the solver.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkbenchUiProject {
    /// Schema version; must equal [`WORKBENCH_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Human-readable project name.
    pub name: String,
    /// Study tree in execution order.
    pub studies: Vec<WorkbenchStudy>,
    /// Port settings for the port and enclosure studies.
    pub port: WorkbenchUiPort,
    /// Speaker workflow payload for driver/enclosure/product studies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<UiSpeakerWorkflow>,
}

/// Errors that prevent a job from being executed at all.
///
/// Execution failures (including unimplemented study kinds) are job
/// outcomes: the job transitions to `Failed` with the message in its
/// result document, and [`run_job`] returns the final record.
#[derive(Debug, thiserror::Error)]
pub enum JobRunnerError {
    /// Job store operation failed.
    #[error(transparent)]
    Store(#[from] JobStoreError),
    /// The UI document schema version is not supported.
    #[error("unsupported workbench UI schema version {found}, expected {expected}")]
    UnsupportedUiSchema {
        /// Version found in the payload.
        found: u32,
        /// Version supported by this build.
        expected: u32,
    },
    /// The job references a study the project does not contain.
    #[error("project has no study with id {id:?}")]
    StudyNotFound {
        /// Missing study identifier.
        id: String,
    },
}

fn kind_label(kind: WorkbenchStudyKind) -> &'static str {
    match kind {
        WorkbenchStudyKind::Motor => "motor",
        WorkbenchStudyKind::DriverLinear => "driver_linear",
        WorkbenchStudyKind::Breakup => "breakup",
        WorkbenchStudyKind::EnclosureSystem => "enclosure_system",
        WorkbenchStudyKind::ProductRadiation => "product_radiation",
        WorkbenchStudyKind::LargeSignal => "large_signal",
        WorkbenchStudyKind::PortOptimisation => "port_optimisation",
        WorkbenchStudyKind::Optimise => "optimise",
    }
}

fn vented_spec(project: &WorkbenchUiProject) -> Result<EnclosureSpec, String> {
    vented_spec_with_volume(project, project.port.box_volume_m3)
}

fn vented_spec_with_volume(
    project: &WorkbenchUiProject,
    volume_m3: f64,
) -> Result<EnclosureSpec, String> {
    let port = PortSpec {
        radius_m: project.port.radius_m,
        length_m: project.port.length_m,
        end_correction_m: project.port.end_correction_m,
        resistance_pa_s_per_m3: 0.0,
    };
    let spec = EnclosureSpec::Vented {
        volume_m3,
        port,
        leakage_resistance_pa_s_per_m3: 0.0,
        absorption_resistance_pa_s_per_m3: 0.0,
    };
    spec.validate().map_err(|error| error.to_string())?;
    Ok(spec)
}

/// Enclosure for the speaker workflow: the payload's family and
/// volume when present, else the legacy vented port settings.
fn speaker_enclosure(project: &WorkbenchUiProject) -> Result<EnclosureSpec, String> {
    match &project.speaker {
        Some(speaker) => match speaker.enclosure.kind {
            UiEnclosureKind::Sealed => {
                let spec = EnclosureSpec::Sealed {
                    volume_m3: speaker.enclosure.volume_m3,
                    leakage_resistance_pa_s_per_m3: 0.0,
                    absorption_resistance_pa_s_per_m3: 0.0,
                };
                spec.validate().map_err(|error| error.to_string())?;
                Ok(spec)
            }
            UiEnclosureKind::Vented => {
                vented_spec_with_volume(project, speaker.enclosure.volume_m3)
            }
        },
        None => vented_spec(project),
    }
}

fn execute_enclosure_system(project: &WorkbenchUiProject) -> Result<serde_json::Value, String> {
    let spec = speaker_enclosure(project)?;
    let medium = AcousticMedium::standard_air();
    let mut result = serde_json::json!({
        "study_kind": "enclosure_system",
        "volume_m3": match &project.speaker {
            Some(speaker) => speaker.enclosure.volume_m3,
            None => project.port.box_volume_m3,
        },
    });
    match spec {
        EnclosureSpec::Sealed { .. } => {
            result["enclosure"] = serde_json::json!("sealed");
        }
        EnclosureSpec::Vented { .. } => {
            result["enclosure"] = serde_json::json!("vented");
            result["tuning_frequency_hz"] = serde_json::json!(
                spec.tuning_frequency_hz(medium)
                    .map_err(|error| error.to_string())?
                    .ok_or_else(|| "vented enclosure reported no tuning frequency".to_owned())?
            );
        }
        _ => return Err("speaker workflow enclosures are sealed or vented".to_owned()),
    }
    let mut samples = Vec::new();
    for frequency_hz in [20.0, 40.0, 80.0, 160.0, 320.0] {
        let impedance = spec
            .acoustic_impedance(frequency_hz, medium)
            .map_err(|error| error.to_string())?;
        samples.push(serde_json::json!({
            "frequency_hz": frequency_hz,
            "real": impedance.re,
            "imag": impedance.im,
            "magnitude": impedance.norm(),
        }));
    }
    result["acoustic_impedance_samples"] = serde_json::json!(samples);
    Ok(result)
}

/// One frequency of the lumped driver sweep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LumpedDriverPoint {
    /// Frequency (Hz).
    pub frequency_hz: f64,
    /// Terminal impedance (Ω, complex).
    pub impedance_ohm: Complex64,
    /// Diaphragm velocity (m/s, complex RMS phasor).
    pub velocity_rms_m_per_s: Complex64,
    /// Diaphragm excursion magnitude (m).
    pub excursion_m: f64,
}

fn log_grid(start_hz: f64, end_hz: f64, points: usize) -> Result<Vec<f64>, String> {
    if !(start_hz.is_finite() && start_hz > 0.0 && end_hz > start_hz)
        || !(2..=200).contains(&points)
    {
        return Err(format!(
            "sweep grid must satisfy 0 < start < end with 2..=200 points, got {start_hz}..{end_hz} x {points}"
        ));
    }
    let ratio = (end_hz / start_hz).powf(1.0 / (points - 1) as f64);
    Ok((0..points)
        .map(|i| start_hz * ratio.powi(i as i32))
        .collect())
}

/// Evaluates the lumped electromechanical driver chain in its
/// enclosure: voice-coil impedance reflected through the moving
/// system plus the enclosure's acoustic load.
///
/// This is the shared Driver Lab small-signal model; the product
/// worker reuses it for radiation boundary velocities.
///
/// # Errors
///
/// Returns a message when the speaker payload is missing or invalid.
pub fn evaluate_lumped_driver(
    project: &WorkbenchUiProject,
    frequencies: &[f64],
) -> Result<Vec<LumpedDriverPoint>, String> {
    let speaker = project.speaker.as_ref().ok_or_else(|| {
        "speaker workflow section is missing; driver studies need driver, enclosure, and drive settings".to_owned()
    })?;
    let driver = &speaker.driver;
    for (name, value) in [
        ("re_ohm", driver.re_ohm),
        ("bl_t_m", driver.bl_t_m),
        ("mms_kg", driver.mms_kg),
        ("cms_m_per_n", driver.cms_m_per_n),
        ("sd_m2", driver.sd_m2),
    ] {
        if !(value.is_finite() && value > 0.0) {
            return Err(format!(
                "driver {name} must be finite and positive, got {value}"
            ));
        }
    }
    if !(speaker.drive.voltage_vrms.is_finite() && speaker.drive.voltage_vrms > 0.0) {
        return Err("drive voltage_vrms must be finite and positive".to_owned());
    }
    let spec = speaker_enclosure(project)?;
    let medium = AcousticMedium::standard_air();
    let area_sq = driver.sd_m2 * driver.sd_m2;
    let mut points = Vec::with_capacity(frequencies.len());
    for &frequency_hz in frequencies {
        if !(frequency_hz.is_finite() && frequency_hz > 0.0) {
            return Err(format!(
                "sweep frequency must be positive, got {frequency_hz}"
            ));
        }
        let omega = std::f64::consts::TAU * frequency_hz;
        let electrical = Complex64::new(driver.re_ohm, omega * driver.le_h.max(0.0));
        let mechanical = Complex64::new(
            driver.rms_n_s_per_m.max(0.0),
            omega * driver.mms_kg - 1.0 / (omega * driver.cms_m_per_n),
        );
        let load = spec
            .acoustic_impedance(frequency_hz, medium)
            .map_err(|error| error.to_string())?
            * area_sq;
        let total_mechanical = mechanical + load;
        let terminal = electrical + (driver.bl_t_m * driver.bl_t_m) / total_mechanical;
        let current = speaker.drive.voltage_vrms / terminal;
        let velocity = driver.bl_t_m * current / total_mechanical;
        points.push(LumpedDriverPoint {
            frequency_hz,
            impedance_ohm: terminal,
            velocity_rms_m_per_s: velocity,
            excursion_m: velocity.norm() / omega,
        });
    }
    Ok(points)
}

fn execute_driver_linear(project: &WorkbenchUiProject) -> Result<serde_json::Value, String> {
    let speaker = project.speaker.as_ref().ok_or_else(|| {
        "speaker workflow section is missing; driver studies need driver, enclosure, and drive settings".to_owned()
    })?;
    let driver = &speaker.driver;
    let frequencies = log_grid(
        speaker.drive.start_hz,
        speaker.drive.end_hz,
        speaker.drive.points,
    )?;
    let points = evaluate_lumped_driver(project, &frequencies)?;
    let fs_free = 1.0 / (std::f64::consts::TAU * (driver.mms_kg * driver.cms_m_per_n).sqrt());
    let qms = std::f64::consts::TAU * fs_free * driver.mms_kg / driver.rms_n_s_per_m.max(1e-12);
    let qes = std::f64::consts::TAU * fs_free * driver.mms_kg * driver.re_ohm
        / (driver.bl_t_m * driver.bl_t_m);
    let peak = points
        .iter()
        .max_by(|a, b| a.impedance_ohm.norm().total_cmp(&b.impedance_ohm.norm()))
        .ok_or("empty sweep")?;
    let max_excursion = points.iter().fold(0.0_f64, |acc, p| acc.max(p.excursion_m));
    Ok(serde_json::json!({
        "study_kind": "driver_linear",
        "model": "lumped electromechanical driver in enclosure load",
        "free_air_resonance_hz": fs_free,
        "qms": qms,
        "qes": qes,
        "qts": qms * qes / (qms + qes),
        "impedance_peak": {
            "frequency_hz": peak.frequency_hz,
            "magnitude_ohm": peak.impedance_ohm.norm(),
        },
        "max_excursion_mm": max_excursion * 1e3,
        "samples": points.iter().map(|p| serde_json::json!({
            "frequency_hz": p.frequency_hz,
            "impedance_magnitude_ohm": p.impedance_ohm.norm(),
            "impedance_phase_deg": p.impedance_ohm.arg().to_degrees(),
            "velocity_magnitude_m_per_s": p.velocity_rms_m_per_s.norm(),
            "excursion_mm": p.excursion_m * 1e3,
        })).collect::<Vec<_>>(),
    }))
}

fn execute_large_signal(project: &WorkbenchUiProject) -> Result<serde_json::Value, String> {
    let speaker = project.speaker.as_ref().ok_or_else(|| {
        "speaker workflow section is missing; driver studies need driver, enclosure, and drive settings".to_owned()
    })?;
    if !(speaker.driver.xmax_m.is_finite() && speaker.driver.xmax_m > 0.0) {
        return Err("driver xmax_m must be finite and positive for large-signal study".to_owned());
    }
    let frequencies = log_grid(
        speaker.drive.start_hz,
        speaker.drive.end_hz,
        speaker.drive.points,
    )?;
    let points = evaluate_lumped_driver(project, &frequencies)?;
    let peak_per_volt =
        points.iter().map(|p| p.excursion_m).fold(0.0_f64, f64::max) / speaker.drive.voltage_vrms;
    let voltage_at_xmax = speaker.driver.xmax_m / peak_per_volt;
    let levels: Vec<serde_json::Value> = [0.25, 0.5, 1.0, 2.0, 4.0]
        .iter()
        .map(|factor| {
            let voltage = speaker.drive.voltage_vrms * factor;
            serde_json::json!({
                "voltage_vrms": voltage,
                "peak_excursion_mm": peak_per_volt * voltage * 1e3,
                "within_xmax": peak_per_volt * voltage <= speaker.driver.xmax_m,
            })
        })
        .collect();
    Ok(serde_json::json!({
        "study_kind": "large_signal",
        "model": "linear lumped excursion extrapolation; Bl(x)/Le(x,i) variation not included",
        "xmax_mm": speaker.driver.xmax_m * 1e3,
        "voltage_at_xmax_vrms": voltage_at_xmax,
        "levels": levels,
    }))
}

fn execute_port_optimisation(project: &WorkbenchUiProject) -> Result<serde_json::Value, String> {
    let spec = vented_spec(project)?;
    let medium = AcousticMedium::standard_air();
    let tuning_hz = spec
        .tuning_frequency_hz(medium)
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "vented enclosure reported no tuning frequency".to_owned())?;
    let EnclosureSpec::Vented { port, .. } = spec else {
        return Err("port study requires a vented enclosure".to_owned());
    };
    let cutoff_hz =
        cutoff_frequency_hz(medium, port.radius_m).map_err(|error| error.to_string())?;
    let fidelity = match select_port_fidelity(tuning_hz, cutoff_hz) {
        PortFidelity::OneDimensional => "one_dimensional",
        PortFidelity::ResolvedModal => "resolved_modal",
    };
    Ok(serde_json::json!({
        "study_kind": "port_optimisation",
        "tuning_frequency_hz": tuning_hz,
        "duct_cutoff_frequency_hz": cutoff_hz,
        "fidelity_at_tuning": fidelity,
        "port_area_m2": port.area_m2().map_err(|error| error.to_string())?,
        "effective_length_m": port.length_m + port.end_correction_m,
        "box_volume_m3": project.port.box_volume_m3,
    }))
}

fn execute(
    project: &WorkbenchUiProject,
    study: &WorkbenchStudy,
) -> Result<serde_json::Value, String> {
    if !study.enabled {
        return Err(format!(
            "study {:?} is disabled; enable it before running",
            study.id
        ));
    }
    match study.kind {
        WorkbenchStudyKind::EnclosureSystem => execute_enclosure_system(project),
        WorkbenchStudyKind::PortOptimisation => execute_port_optimisation(project),
        WorkbenchStudyKind::DriverLinear => execute_driver_linear(project),
        WorkbenchStudyKind::LargeSignal => execute_large_signal(project),
        WorkbenchStudyKind::ProductRadiation => Err(
            "product radiation is executed by speaker-product-worker (sonium-sim-speaker); \
             route the study to that worker"
                .to_owned(),
        ),
        other => Err(format!(
            "no executor implemented for study kind {}; the job fails closed instead of substituting a lower-fidelity result",
            kind_label(other)
        )),
    }
}

/// Executes one queued job to a terminal state.
///
/// The job must already exist in the store (submitted over the exact
/// project document bytes). Returns the final record for both
/// completed and failed executions; `Err` is reserved for failures
/// that prevent execution entirely.
///
/// # Errors
///
/// Returns [`JobRunnerError`] for store failures, unsupported UI
/// schema versions, or a job whose study is absent from the project.
pub fn run_job(
    store: &JobStore,
    job_id: &str,
    project: &WorkbenchUiProject,
) -> Result<JobRecord, JobRunnerError> {
    if project.schema_version != WORKBENCH_SCHEMA_VERSION {
        return Err(JobRunnerError::UnsupportedUiSchema {
            found: project.schema_version,
            expected: WORKBENCH_SCHEMA_VERSION,
        });
    }
    let record = store.load(job_id)?;
    let study = project
        .studies
        .iter()
        .find(|study| study.id == record.study_id)
        .ok_or_else(|| JobRunnerError::StudyNotFound {
            id: record.study_id.clone(),
        })?;
    store.transition(job_id, JobState::Running)?;
    store.beat(job_id)?;
    match execute(project, study) {
        Ok(result) => {
            store.write_result(job_id, &result)?;
            store.transition(job_id, JobState::Completed)?;
        }
        Err(message) => {
            store.write_result(job_id, &serde_json::json!({ "error": message }))?;
            store.transition(job_id, JobState::Failed)?;
        }
    }
    Ok(store.load(job_id)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job_service::JobState;
    use crate::workbench::{FidelityLabel, WorkbenchStudy};

    fn ui_project() -> WorkbenchUiProject {
        let study = |id: &str, kind: WorkbenchStudyKind| WorkbenchStudy {
            id: id.to_owned(),
            kind,
            fidelity: FidelityLabel::Lumped,
            enabled: true,
        };
        WorkbenchUiProject {
            schema_version: WORKBENCH_SCHEMA_VERSION,
            name: "Reference two-way (sealed)".to_owned(),
            studies: vec![
                study("enclosure-system", WorkbenchStudyKind::EnclosureSystem),
                study("port-optimisation", WorkbenchStudyKind::PortOptimisation),
                study("motor", WorkbenchStudyKind::Motor),
            ],
            port: WorkbenchUiPort {
                radius_m: 0.025,
                length_m: 0.15,
                end_correction_m: 0.03,
                box_volume_m3: 0.02,
            },
            speaker: None,
        }
    }

    fn speaker_project() -> WorkbenchUiProject {
        let mut project = ui_project();
        project.studies.push(WorkbenchStudy {
            id: "driver-linear".to_owned(),
            kind: WorkbenchStudyKind::DriverLinear,
            fidelity: FidelityLabel::Lumped,
            enabled: true,
        });
        project.studies.push(WorkbenchStudy {
            id: "large-signal".to_owned(),
            kind: WorkbenchStudyKind::LargeSignal,
            fidelity: FidelityLabel::Lumped,
            enabled: true,
        });
        project.speaker = Some(UiSpeakerWorkflow {
            driver: UiDriver {
                re_ohm: 5.8,
                le_h: 0.0003,
                bl_t_m: 4.5,
                mms_kg: 0.008,
                cms_m_per_n: 0.0012,
                rms_n_s_per_m: 0.5,
                sd_m2: 0.0087,
                xmax_m: 0.005,
            },
            enclosure: UiEnclosure {
                kind: UiEnclosureKind::Sealed,
                volume_m3: 0.012,
            },
            drive: UiDrive {
                voltage_vrms: 2.83,
                start_hz: 20.0,
                end_hz: 500.0,
                points: 25,
            },
            product: None,
        });
        project
    }

    fn scratch_store(test_name: &str) -> (JobStore, std::path::PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "sonium-job-runner-{test_name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let store = JobStore::open(&root).expect("open scratch store");
        (store, root)
    }

    fn submit_and_run(store: &JobStore, project: &WorkbenchUiProject, study_id: &str) -> JobRecord {
        let snapshot = serde_json::to_vec(project).expect("serialise project");
        let job_id = format!("test-{study_id}");
        store
            .submit(&job_id, &project.name, study_id, &snapshot)
            .expect("submit");
        run_job(store, &job_id, project).expect("run job")
    }

    #[test]
    fn port_job_completes_with_hand_checked_tuning() {
        let (store, root) = scratch_store("port");
        let project = ui_project();
        let record = submit_and_run(&store, &project, "port-optimisation");
        assert_eq!(record.state, JobState::Completed);
        let result_text = std::fs::read_to_string(root.join("test-port-optimisation/result.json"))
            .expect("result file");
        let result: serde_json::Value = serde_json::from_str(&result_text).expect("result json");
        // Independent Helmholtz check: c/(2pi)*sqrt(S/(V*L_eff)) with
        // r=25 mm, L_eff=180 mm, V=20 L gives 40.32 Hz.
        let tuning = result["tuning_frequency_hz"].as_f64().expect("tuning");
        assert!((tuning - 40.32).abs() < 0.05, "tuning {tuning}");
        let cutoff = result["duct_cutoff_frequency_hz"].as_f64().expect("cutoff");
        assert!((cutoff - 4020.9).abs() < 0.5, "cutoff {cutoff}");
        assert_eq!(result["fidelity_at_tuning"], "one_dimensional");
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn enclosure_job_completes_with_impedance_samples() {
        let (store, root) = scratch_store("enclosure");
        let project = ui_project();
        let record = submit_and_run(&store, &project, "enclosure-system");
        assert_eq!(record.state, JobState::Completed);
        let result_text = std::fs::read_to_string(root.join("test-enclosure-system/result.json"))
            .expect("result file");
        let result: serde_json::Value = serde_json::from_str(&result_text).expect("result json");
        assert_eq!(
            result["acoustic_impedance_samples"]
                .as_array()
                .map(Vec::len),
            Some(5)
        );
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn driver_linear_reports_hand_checked_resonance_and_raised_box_peak() {
        let (store, root) = scratch_store("driver-linear");
        let project = speaker_project();
        let record = submit_and_run(&store, &project, "driver-linear");
        assert_eq!(record.state, JobState::Completed);
        let result_text = std::fs::read_to_string(root.join("test-driver-linear/result.json"))
            .expect("result file");
        let result: serde_json::Value = serde_json::from_str(&result_text).expect("result json");
        // fs = 1/(2*pi*sqrt(Mms*Cms)) for Mms=8 g, Cms=1.2 mm/N.
        let fs = result["free_air_resonance_hz"].as_f64().expect("fs");
        assert!((fs - 51.38).abs() < 0.05, "fs {fs}");
        // The 12 L sealed box must raise the impedance peak above fs.
        let peak = result["impedance_peak"]["frequency_hz"]
            .as_f64()
            .expect("peak hz");
        assert!(peak > fs, "in-box peak {peak} must exceed free-air {fs}");
        assert_eq!(
            result["samples"].as_array().map(Vec::len),
            Some(25),
            "sweep sample count"
        );
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn large_signal_voltage_at_xmax_is_consistent_with_levels() {
        let (store, root) = scratch_store("large-signal");
        let project = speaker_project();
        let record = submit_and_run(&store, &project, "large-signal");
        assert_eq!(record.state, JobState::Completed);
        let result_text = std::fs::read_to_string(root.join("test-large-signal/result.json"))
            .expect("result file");
        let result: serde_json::Value = serde_json::from_str(&result_text).expect("result json");
        let voltage_at_xmax = result["voltage_at_xmax_vrms"]
            .as_f64()
            .expect("voltage at xmax");
        // Linear scaling: the level nearest that voltage must peak at
        // approximately xmax (5 mm).
        let levels = result["levels"].as_array().expect("levels");
        let nearest = levels
            .iter()
            .min_by(|a, b| {
                let da = (a["voltage_vrms"].as_f64().unwrap() - voltage_at_xmax).abs();
                let db = (b["voltage_vrms"].as_f64().unwrap() - voltage_at_xmax).abs();
                da.total_cmp(&db)
            })
            .expect("nearest level");
        let ratio = nearest["voltage_vrms"].as_f64().unwrap() / voltage_at_xmax;
        let expected_mm = 5.0 * ratio;
        let actual_mm = nearest["peak_excursion_mm"].as_f64().unwrap();
        assert!(
            (actual_mm - expected_mm).abs() / expected_mm < 0.45,
            "excursion scaling {actual_mm} vs {expected_mm}"
        );
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn sealed_enclosure_job_reports_no_tuning() {
        let (store, root) = scratch_store("sealed");
        let project = speaker_project();
        let record = submit_and_run(&store, &project, "enclosure-system");
        assert_eq!(record.state, JobState::Completed);
        let result_text = std::fs::read_to_string(root.join("test-enclosure-system/result.json"))
            .expect("result file");
        let result: serde_json::Value = serde_json::from_str(&result_text).expect("result json");
        assert_eq!(result["enclosure"], "sealed");
        assert!(result.get("tuning_frequency_hz").is_none());
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn unimplemented_kind_fails_the_job_with_a_named_gap() {
        let (store, root) = scratch_store("unimplemented");
        let project = ui_project();
        let record = submit_and_run(&store, &project, "motor");
        assert_eq!(record.state, JobState::Failed);
        let result_text =
            std::fs::read_to_string(root.join("test-motor/result.json")).expect("result file");
        assert!(result_text.contains("no executor implemented for study kind motor"));
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn unknown_study_and_bad_schema_never_start() {
        let (store, root) = scratch_store("preflight");
        let project = ui_project();
        let snapshot = serde_json::to_vec(&project).expect("serialise");
        store
            .submit("test-missing", &project.name, "nope", &snapshot)
            .expect("submit");
        assert!(matches!(
            run_job(&store, "test-missing", &project),
            Err(JobRunnerError::StudyNotFound { .. })
        ));
        assert_eq!(
            store.load("test-missing").expect("load").state,
            JobState::Queued
        );
        let mut bad = ui_project();
        bad.schema_version = 99;
        assert!(matches!(
            run_job(&store, "test-missing", &bad),
            Err(JobRunnerError::UnsupportedUiSchema { .. })
        ));
        std::fs::remove_dir_all(root).expect("cleanup");
    }
}
