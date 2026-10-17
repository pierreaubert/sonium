//! Release-backed continuous quadratic Quad4 penetrable-sphere convergence.

// Rust guideline compliant 2026-02-21

#[path = "../../sonium-bem/tests/continuous_quadratic_quad4_transmission_mie_convergence.rs"]
mod packet;

use sonium_qa::QaResult;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.transmission-continuous-quadratic-quad4-mie-convergence.v1";

fn main() {
    let started = Instant::now();
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        packet::run_continuous_quadratic_quad4_transmission_mie_convergence()
    }));
    let mut result = match outcome {
        Ok(metrics) => {
            let refinement_ratio = (metrics.medium_error
                / metrics.coarse_error.max(f64::MIN_POSITIVE))
            .max(metrics.fine_error / metrics.medium_error.max(f64::MIN_POSITIVE));
            QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("coarse_error", metrics.coarse_error)
                .metric("medium_error",metrics.medium_error)
                .metric("medium_residual",metrics.medium_residual).maximum("medium_residual",1e-8)
                .metric("medium_solution_error",metrics.medium_solution_error).maximum("medium_solution_error",1e-8)
                .metric("numerical_solve_executed",3.0).minimum("numerical_solve_executed",3.0)
                .metric("fine_error", metrics.fine_error)
                .metric("field_refinement_ratio", refinement_ratio)
                .metric("coarse_residual", metrics.coarse_residual)
                .metric("fine_residual", metrics.fine_residual)
                .metric("coarse_solution_error", metrics.coarse_solution_error)
                .metric("fine_solution_error", metrics.fine_solution_error)
                .metric("coarse_dofs", metrics.coarse_dofs as f64)
                .metric("fine_dofs", metrics.fine_dofs as f64)
                .metric("coarse_panels", metrics.coarse_panels as f64)
                .metric("fine_panels", metrics.fine_panels as f64)
                .maximum("fine_error", 0.01)
                .maximum("field_refinement_ratio", 0.72)
                .maximum("coarse_residual", 1.0e-8)
                .maximum("fine_residual", 1.0e-8)
                .maximum("coarse_solution_error", 1.0e-8)
                .maximum("fine_solution_error", 1.0e-8)
                .minimum("fine_dofs", (metrics.coarse_dofs + 1) as f64)
                .minimum("fine_panels", (metrics.coarse_panels + 1) as f64)
                .input_checksum(
                    "bem:continuous-q2-quad4:fluid-sphere-420hz:face-resolutions-2-4-8",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:bem-transmission-continuous-q2-quad4-mie-v1",
                )
                .reference("independent FluidSphereTransmission3d exterior pressure")
                .dimension("3-D three-level CPU continuous-Q2 Quad4 transmission; face resolutions 2/4/8")
                .formulation("Galerkin Müller two-medium transmission with shared quadratic Quad4 traces and projected incident data")
                .tolerance(
                    "fine complex field error <= 0.01; both refinement ratios <= 0.72; residual/action <= 1e-8",
                )
                .seed("bem-transmission-continuous-q2-quad4-mie:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX))
        }
        Err(_) => QaResult::failed(CASE_ID, "continuous Quad4 transmission Mie packet panicked")
            .requested_backend("cpu")
            .actual_backend("cpu")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
    };
    if result.status == sonium_qa::QaStatus::Passed
        && let Err(error) = result.release_eligible()
    {
        // Preserve measured errors and provenance when accuracy admission fails.
        result.status = sonium_qa::QaStatus::Failed;
        result.message = Some(error);
    }
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if matches!(result.status, sonium_qa::QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
