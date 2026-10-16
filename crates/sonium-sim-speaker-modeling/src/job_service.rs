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
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled | Self::Interrupted)
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
            return Err(JobServiceError::IllegalTransition { from: self.state, to: next });
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
        job.transition(JobState::Running).expect("queued to running");
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
}
