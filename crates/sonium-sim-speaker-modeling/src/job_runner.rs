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
//! the lumped enclosure system and the port analysis. Study kinds
//! whose inputs the UI document does not carry fail the job with a
//! named gap instead of silently substituting a lower-fidelity
//! result.

// Rust guideline compliant 2026-02-21

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
    let port = PortSpec {
        radius_m: project.port.radius_m,
        length_m: project.port.length_m,
        end_correction_m: project.port.end_correction_m,
        resistance_pa_s_per_m3: 0.0,
    };
    let spec = EnclosureSpec::Vented {
        volume_m3: project.port.box_volume_m3,
        port,
        leakage_resistance_pa_s_per_m3: 0.0,
        absorption_resistance_pa_s_per_m3: 0.0,
    };
    spec.validate().map_err(|error| error.to_string())?;
    Ok(spec)
}

fn execute_enclosure_system(project: &WorkbenchUiProject) -> Result<serde_json::Value, String> {
    let spec = vented_spec(project)?;
    let medium = AcousticMedium::standard_air();
    let tuning_hz = spec
        .tuning_frequency_hz(medium)
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "vented enclosure reported no tuning frequency".to_owned())?;
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
    Ok(serde_json::json!({
        "study_kind": "enclosure_system",
        "enclosure": "vented",
        "volume_m3": project.port.box_volume_m3,
        "tuning_frequency_hz": tuning_hz,
        "acoustic_impedance_samples": samples,
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
        }
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
