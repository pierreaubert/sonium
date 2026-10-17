//! Unified job records for the Speaker Workbench service.
//!
//! One job model replaces the three legacy ones (Speaker Lab SQLite
//! jobs, Speaker Studio run directories, and the demo lab shell). A job
//! owns an immutable, hash-identified input snapshot and a status file;
//! cancellation is exact (one job id, one status file), reconnecting
//! never duplicates a job, and a service restart marks every
//! non-terminal job interrupted instead of pretending it finished.

// Rust guideline compliant 2026-02-21

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Lifecycle state of one Workbench job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    /// Input snapshot written; worker not yet started.
    Queued,
    /// Worker is running and heartbeating.
    Running,
    /// Worker finished and published results.
    Completed,
    /// Worker failed; see the job log.
    Failed,
    /// Cancellation was requested and honoured.
    Cancelled,
    /// Service restarted while the job was non-terminal.
    Interrupted,
}

impl JobState {
    /// Returns true for states a job never leaves.
    #[must_use]
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Cancelled | Self::Interrupted
        )
    }
}

/// Immutable record of one Workbench job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobRecord {
    /// Caller-chosen stable job identifier.
    pub id: String,
    /// Project name the job was submitted from.
    pub project_name: String,
    /// Study identifier within the project tree.
    pub study_id: String,
    /// SHA-256 hex digest of the canonical input snapshot.
    pub input_hash: String,
    /// Current lifecycle state.
    pub state: JobState,
    /// Monotonic heartbeat counter from the worker.
    pub heartbeat: u64,
}

impl JobRecord {
    /// Creates a queued job over a canonical input snapshot.
    ///
    /// The snapshot bytes are hashed, never mutated afterwards; two
    /// submissions with identical bytes share an input hash, so a
    /// reconnect can recognise its job instead of duplicating it.
    #[must_use]
    pub fn queued(id: &str, project_name: &str, study_id: &str, input_snapshot: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(input_snapshot);
        Self {
            id: id.to_owned(),
            project_name: project_name.to_owned(),
            study_id: study_id.to_owned(),
            input_hash: format!("{:x}", hasher.finalize()),
            state: JobState::Queued,
            heartbeat: 0,
        }
    }

    /// Advances the lifecycle if the transition is legal.
    ///
    /// Terminal states never transition again; a restart reconciliation
    /// uses [`JobRecord::mark_interrupted_on_restart`] instead.
    ///
    /// # Errors
    ///
    /// Returns [`JobServiceError::IllegalTransition`] for moves out of
    /// a terminal state or backwards moves.
    pub fn transition(&mut self, next: JobState) -> Result<(), JobServiceError> {
        let legal = matches!(
            (self.state, next),
            (JobState::Queued, JobState::Running)
                | (JobState::Queued, JobState::Cancelled)
                | (JobState::Running, JobState::Completed)
                | (JobState::Running, JobState::Failed)
                | (JobState::Running, JobState::Cancelled)
        );
        if !legal {
            return Err(JobServiceError::IllegalTransition {
                from: self.state,
                to: next,
            });
        }
        self.state = next;
        Ok(())
    }

    /// Records one worker heartbeat.
    pub fn beat(&mut self) {
        self.heartbeat = self.heartbeat.saturating_add(1);
    }

    /// Marks a non-terminal job interrupted after a service restart.
    ///
    /// Returns true when the state changed.
    pub fn mark_interrupted_on_restart(&mut self) -> bool {
        if self.state.is_terminal() {
            return false;
        }
        self.state = JobState::Interrupted;
        true
    }
}

/// Directory-backed persistence for Workbench jobs.
///
/// Each job owns one directory under the store root containing the
/// immutable `input.snapshot`, the `status.json` lifecycle record, and
/// a `result.json` once the job reaches a terminal state. Status
/// writes go through a temporary file plus rename, so a crash cannot
/// leave a torn status behind. Opening a store reconciles the
/// previous session: every non-terminal job is marked interrupted
/// rather than left pretending it is still running.
#[derive(Debug, Clone)]
pub struct JobStore {
    root: std::path::PathBuf,
}

/// Errors returned by the Workbench job store.
#[derive(Debug, thiserror::Error)]
pub enum JobStoreError {
    /// A filesystem operation failed.
    #[error("job store io error at {path}: {source}")]
    Io {
        /// Path being operated on.
        path: std::path::PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// A persisted job file could not be decoded.
    #[error("job store decode error at {path}: {message}")]
    Decode {
        /// Path being decoded.
        path: std::path::PathBuf,
        /// Decoder message.
        message: String,
    },
    /// Job lifecycle transition was rejected.
    #[error(transparent)]
    Service(#[from] JobServiceError),
    /// No job with this id exists in the store.
    #[error("unknown job id {0}")]
    UnknownJob(String),
    /// The job id is not a safe single path component.
    #[error("invalid job id {0:?}")]
    InvalidJobId(String),
    /// A job id was resubmitted with different input bytes.
    #[error("job id {id} already exists with a different input snapshot")]
    ConflictingJobId {
        /// Reused job identifier.
        id: String,
    },
}

impl JobStore {
    /// Opens (creating if needed) the store rooted at `root` and
    /// reconciles jobs left non-terminal by a previous session.
    ///
    /// # Errors
    ///
    /// Returns [`JobStoreError`] when the root cannot be created or a
    /// persisted status cannot be read or rewritten.
    pub fn open(root: impl Into<std::path::PathBuf>) -> Result<Self, JobStoreError> {
        let root = root.into();
        std::fs::create_dir_all(&root).map_err(|source| JobStoreError::Io {
            path: root.clone(),
            source,
        })?;
        let store = Self { root };
        store.reconcile_on_open()?;
        Ok(store)
    }

    /// Root directory of the store.
    #[must_use]
    pub fn root(&self) -> &std::path::Path {
        &self.root
    }

    /// Submits a job over an immutable input snapshot.
    ///
    /// Resubmitting the same id with identical bytes returns the
    /// existing record (reconnects never duplicate a job); the same
    /// id with different bytes is a conflict and fails closed.
    ///
    /// # Errors
    ///
    /// Returns [`JobStoreError`] for invalid ids, conflicts, or I/O
    /// failures.
    pub fn submit(
        &self,
        id: &str,
        project_name: &str,
        study_id: &str,
        input_snapshot: &[u8],
    ) -> Result<JobRecord, JobStoreError> {
        validate_job_id(id)?;
        let candidate = JobRecord::queued(id, project_name, study_id, input_snapshot);
        if self.job_dir(id).exists() {
            let existing = self.load(id)?;
            if existing.input_hash != candidate.input_hash {
                return Err(JobStoreError::ConflictingJobId { id: id.to_owned() });
            }
            return Ok(existing);
        }
        let dir = self.job_dir(id);
        std::fs::create_dir_all(&dir).map_err(|source| JobStoreError::Io {
            path: dir.clone(),
            source,
        })?;
        let input_path = dir.join("input.snapshot");
        std::fs::write(&input_path, input_snapshot).map_err(|source| JobStoreError::Io {
            path: input_path,
            source,
        })?;
        self.write_status(&candidate)?;
        Ok(candidate)
    }

    /// Loads the current record for `id`.
    ///
    /// # Errors
    ///
    /// Returns [`JobStoreError::UnknownJob`] when absent, or a decode
    /// or I/O error when the status file is unreadable.
    pub fn load(&self, id: &str) -> Result<JobRecord, JobStoreError> {
        validate_job_id(id)?;
        let path = self.status_path(id);
        let bytes = std::fs::read(&path).map_err(|source| {
            if source.kind() == std::io::ErrorKind::NotFound {
                JobStoreError::UnknownJob(id.to_owned())
            } else {
                JobStoreError::Io {
                    path: path.clone(),
                    source,
                }
            }
        })?;
        serde_json::from_slice(&bytes).map_err(|error| JobStoreError::Decode {
            path,
            message: error.to_string(),
        })
    }

    /// Lists every job in the store, sorted by job id.
    ///
    /// # Errors
    ///
    /// Returns [`JobStoreError`] when the root cannot be scanned or a
    /// status file is unreadable.
    pub fn list(&self) -> Result<Vec<JobRecord>, JobStoreError> {
        let entries = std::fs::read_dir(&self.root).map_err(|source| JobStoreError::Io {
            path: self.root.clone(),
            source,
        })?;
        let mut ids = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|source| JobStoreError::Io {
                path: self.root.clone(),
                source,
            })?;
            if entry
                .file_type()
                .map_err(|source| JobStoreError::Io {
                    path: entry.path(),
                    source,
                })?
                .is_dir()
            {
                ids.push(entry.file_name().to_string_lossy().into_owned());
            }
        }
        ids.sort();
        ids.iter().map(|id| self.load(id)).collect()
    }

    /// Advances a job's lifecycle and persists the new status.
    ///
    /// # Errors
    ///
    /// Returns [`JobStoreError`] when the transition is illegal or
    /// the status cannot be persisted.
    pub fn transition(&self, id: &str, next: JobState) -> Result<JobRecord, JobStoreError> {
        let mut record = self.load(id)?;
        record.transition(next)?;
        self.write_status(&record)?;
        Ok(record)
    }

    /// Records one worker heartbeat and persists the status.
    ///
    /// # Errors
    ///
    /// Returns [`JobStoreError`] when the job is unknown or the
    /// status cannot be persisted.
    pub fn beat(&self, id: &str) -> Result<JobRecord, JobStoreError> {
        let mut record = self.load(id)?;
        record.beat();
        self.write_status(&record)?;
        Ok(record)
    }

    /// Finds a job by project, study, and exact input bytes.
    ///
    /// # Errors
    ///
    /// Returns [`JobStoreError`] when the store cannot be scanned.
    pub fn find_by_input(
        &self,
        project_name: &str,
        study_id: &str,
        input_snapshot: &[u8],
    ) -> Result<Option<JobRecord>, JobStoreError> {
        let probe = JobRecord::queued("probe", project_name, study_id, input_snapshot);
        Ok(self.list()?.into_iter().find(|record| {
            record.project_name == project_name
                && record.study_id == study_id
                && record.input_hash == probe.input_hash
        }))
    }

    /// Writes the terminal result document for a job.
    ///
    /// # Errors
    ///
    /// Returns [`JobStoreError`] when the job is unknown or the
    /// result cannot be encoded or written.
    pub fn write_result(&self, id: &str, result: &serde_json::Value) -> Result<(), JobStoreError> {
        validate_job_id(id)?;
        if !self.job_dir(id).exists() {
            return Err(JobStoreError::UnknownJob(id.to_owned()));
        }
        let path = self.job_dir(id).join("result.json");
        let bytes = serde_json::to_vec_pretty(result).map_err(|error| JobStoreError::Decode {
            path: path.clone(),
            message: error.to_string(),
        })?;
        write_atomically(&path, &bytes)
    }

    fn reconcile_on_open(&self) -> Result<(), JobStoreError> {
        for mut record in self.list()? {
            if record.mark_interrupted_on_restart() {
                self.write_status(&record)?;
            }
        }
        Ok(())
    }

    fn job_dir(&self, id: &str) -> std::path::PathBuf {
        self.root.join(id)
    }

    fn status_path(&self, id: &str) -> std::path::PathBuf {
        self.job_dir(id).join("status.json")
    }

    fn write_status(&self, record: &JobRecord) -> Result<(), JobStoreError> {
        let path = self.status_path(&record.id);
        let bytes = serde_json::to_vec_pretty(record).map_err(|error| JobStoreError::Decode {
            path: path.clone(),
            message: error.to_string(),
        })?;
        write_atomically(&path, &bytes)
    }
}

fn validate_job_id(id: &str) -> Result<(), JobStoreError> {
    let valid = !id.is_empty()
        && id != "."
        && id != ".."
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    if valid {
        Ok(())
    } else {
        Err(JobStoreError::InvalidJobId(id.to_owned()))
    }
}

fn write_atomically(path: &std::path::Path, bytes: &[u8]) -> Result<(), JobStoreError> {
    let temporary = path.with_extension("json.tmp");
    std::fs::write(&temporary, bytes).map_err(|source| JobStoreError::Io {
        path: temporary.clone(),
        source,
    })?;
    std::fs::rename(&temporary, path).map_err(|source| JobStoreError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Errors returned by the Workbench job service.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum JobServiceError {
    /// A lifecycle transition was attempted from a terminal state.
    #[error("illegal job transition from {from:?} to {to:?}")]
    IllegalTransition {
        /// State the job was in.
        from: JobState,
        /// State that was requested.
        to: JobState,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_inputs_share_a_hash() {
        let first = JobRecord::queued("a", "p", "s", b"input");
        let second = JobRecord::queued("b", "p", "s", b"input");
        assert_eq!(first.input_hash, second.input_hash);
        assert_ne!(
            first.input_hash,
            JobRecord::queued("c", "p", "s", b"other").input_hash
        );
    }

    #[test]
    fn lifecycle_and_restart_reconciliation() {
        let mut job = JobRecord::queued("a", "p", "s", b"input");
        job.transition(JobState::Running)
            .expect("queued to running");
        job.beat();
        assert_eq!(job.heartbeat, 1);
        assert!(job.mark_interrupted_on_restart());
        assert_eq!(job.state, JobState::Interrupted);
        assert!(job.transition(JobState::Running).is_err());
        let mut done = JobRecord::queued("b", "p", "s", b"input");
        done.transition(JobState::Running).expect("running");
        done.transition(JobState::Completed).expect("completed");
        assert!(!done.mark_interrupted_on_restart());
    }

    fn scratch_store(test_name: &str) -> (JobStore, std::path::PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "sonium-job-store-{test_name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let store = JobStore::open(&root).expect("open scratch store");
        (store, root)
    }

    #[test]
    fn store_submit_is_idempotent_and_conflicts_fail_closed() {
        let (store, root) = scratch_store("submit");
        let first = store.submit("job-1", "p", "s", b"input").expect("submit");
        let second = store.submit("job-1", "p", "s", b"input").expect("resubmit");
        assert_eq!(first, second);
        assert_eq!(store.list().expect("list").len(), 1);
        assert!(matches!(
            store.submit("job-1", "p", "s", b"different"),
            Err(JobStoreError::ConflictingJobId { .. })
        ));
        assert!(matches!(
            store.submit("../escape", "p", "s", b"input"),
            Err(JobStoreError::InvalidJobId(_))
        ));
        assert_eq!(
            store
                .find_by_input("p", "s", b"input")
                .expect("find")
                .map(|r| r.id),
            Some("job-1".to_owned())
        );
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn store_persists_transitions_and_reconciles_on_reopen() {
        let (store, root) = scratch_store("persist");
        store.submit("job-1", "p", "s", b"input").expect("submit");
        let running = store.transition("job-1", JobState::Running).expect("run");
        assert_eq!(running.state, JobState::Running);
        // A fresh open is a service restart: the running job must be
        // marked interrupted, never silently completed.
        let reopened = JobStore::open(&root).expect("reopen");
        assert_eq!(
            reopened.load("job-1").expect("load").state,
            JobState::Interrupted
        );
        assert!(reopened.transition("job-1", JobState::Completed).is_err());
        std::fs::remove_dir_all(root).expect("cleanup");
    }
}
