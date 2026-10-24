//! Mesh2HRTF / NumCalc project driver.
//!
//! Parses a Mesh2HRTF project directory (NC.inp + node + element + evaluation
//! grid files), prepares a `Vec<Element>` with the boundary conditions the
//! project requests, and runs a frequency sweep through `sonium-bem`.
//!
//! Output is a JSON file per ear with `pressure[freq, eval_point]` complex
//! values, matching what the Python `Output2HRTFs` post-processor expects as
//! input. A native AES69 SOFA writer will be layered on top.

pub mod project;
pub mod runner;

pub use project::{NumCalcProject, ProjectError, load_project};
#[cfg(feature = "cuda")]
#[doc(inline)]
pub use runner::run_project_cuda_jacobi;
pub use runner::{SweepExecution, SweepOutput, run_project};
