//! `workbench-worker`: execute Workbench job-service requests.
//!
//! Front ends (the GPUI Speaker Workbench today) submit their UI
//! project document through this binary so job execution stays in
//! the solver workspace:
//!
//! ```text
//! workbench-worker --jobs-root DIR run  --project project.json --study port-optimisation
//! workbench-worker --jobs-root DIR list
//! ```
//!
//! `run` submits the document bytes as the job's immutable snapshot
//! (job id defaults to `<study>-<input-hash-prefix>`, so a reconnect
//! with identical bytes finds the existing job instead of
//! duplicating it), executes a queued job to completion, and prints
//! the final job record as JSON on stdout. `list` prints every job
//! record in the store as a JSON array; opening the store reconciles
//! jobs a previous session left non-terminal (they are reported as
//! interrupted, never as silently finished).

// Rust guideline compliant 2026-02-21

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use sha2::{Digest, Sha256};
use sonium_sim_speaker_modeling::job_runner::{WorkbenchUiProject, run_job};
use sonium_sim_speaker_modeling::job_service::{JobState, JobStore};

#[derive(Parser)]
#[command(
    name = "workbench-worker",
    about = "Execute Workbench job-service requests from front-end project documents"
)]
struct Cli {
    /// Root directory of the job store.
    #[arg(long, default_value = "workbench-jobs")]
    jobs_root: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Submit a UI project document and run one study job.
    Run {
        /// Path to the workbench UI project JSON document.
        #[arg(long)]
        project: PathBuf,
        /// Study identifier within the project tree.
        #[arg(long)]
        study: String,
        /// Job identifier; defaults to `<study>-<input-hash-prefix>`.
        #[arg(long)]
        job_id: Option<String>,
    },
    /// List every job record in the store as JSON.
    List,
}

fn main() {
    if let Err(error) = run(Cli::parse()) {
        eprintln!("workbench-worker failed: {error}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    let store = JobStore::open(&cli.jobs_root)?;
    match cli.command {
        Command::List => {
            println!("{}", serde_json::to_string_pretty(&store.list()?)?);
            Ok(())
        }
        Command::Run {
            project,
            study,
            job_id,
        } => {
            let snapshot = std::fs::read(&project)?;
            let document: WorkbenchUiProject = serde_json::from_slice(&snapshot)?;
            if !document.studies.iter().any(|entry| entry.id == study) {
                return Err(
                    format!("project {:?} has no study with id {study:?}", document.name).into(),
                );
            }
            let mut hasher = Sha256::new();
            hasher.update(&snapshot);
            let input_hash = format!("{:x}", hasher.finalize());
            let job_id = job_id.unwrap_or_else(|| format!("{study}-{}", &input_hash[..12]));
            let record = store.submit(&job_id, &document.name, &study, &snapshot)?;
            let final_record = if record.state == JobState::Queued {
                run_job(&store, &record.id, &document)?
            } else {
                // Terminal or already-running jobs are returned as-is:
                // a reconnect never duplicates or re-runs a job.
                record
            };
            println!("{}", serde_json::to_string_pretty(&final_record)?);
            Ok(())
        }
    }
}
