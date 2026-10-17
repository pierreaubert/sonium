//! Compressible Navier-Stokes smoke and nonuniform viscous accuracy gates.
//!
//! The original uniform-flow record checks free-stream preservation, positive
//! thermodynamics, and Sutherland viscosity. A separate rotating shear-wave
//! record solves a nonuniform exact viscous benchmark on three tetrahedral
//! meshes and checks temporal convergence through the public SSPRK3 solver.
//!
//! The spatial driver supplies exact boundary traces at SSPRK3 stage times.
//! The public solver's temporal test uses fixed traces and self-convergence;
//! it does not claim time-dependent boundary support in the production API.
//!
//! Scope is deliberately limited to those smoke and shear-wave gates: do
//! not cite this binary as validation of time-dependent boundary handling,
//! turbulent or high-Mach flow, or production compressible-NS accuracy.

// Rust guideline compliant 2026-02-21
fn main() {
    sonium_qa::fem_physics_3d::run_compressible_case();
    match compressible_shear::run().and_then(|r| r.to_json_line()) {
        Ok(line) => {
            println!("QA_RESULT: {line}");
            println!("SUCCESS");
        }
        Err(error) => {
            if let Ok(line) =
                sonium_qa::QaResult::failed("fem.compressible-shear-wave-3d.v1", &error)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
            {
                println!("QA_RESULT: {line}");
            }
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}

#[path = "support/compressible_shear.rs"]
mod compressible_shear;
