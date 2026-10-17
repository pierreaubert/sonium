//! Release-backed coarse JLR paper-tau regularised shakedown (QA tier).
//!
//! The 11,111-step LBM run takes tens of minutes, so it is not a
//! `cargo test` case; this binary executes the shared packet from
//! sonium-sim-jlr and gates its peak lattice speed.

// Rust guideline compliant 2026-02-21

#[allow(dead_code)]
#[path = "../../sonium-sim-jlr/tests/jlr_shakedown.rs"]
mod packet;

use sonium_qa::QaResult;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "sim-jlr.paper-tau-regularized-shakedown.v1";

fn main() {
    let started = Instant::now();
    let outcome = catch_unwind(AssertUnwindSafe(packet::run_coarse_jlr_paper_tau_regularized));
    let mut result = match outcome {
        Ok(max_velocity_lb) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("steps", 11_111.0)
            .minimum("steps", 11_111.0)
            .metric("max_velocity_lb", max_velocity_lb)
            .minimum("max_velocity_lb", 0.005)
            .maximum("max_velocity_lb", 0.35)
            .input_checksum("sim-jlr:coarse-4.5mm:paper-tau:regularized:op1")
            .reference_checksum("analytic/invariant:jlr-paper-tau-regularized-v1")
            .reference("finite stabilised LBM shakedown at the paper operating point")
            .dimension("3-D coarse 4.5 mm JLR lattice; 11,111 steps; plug inlet")
            .formulation("D3Q19 LBM, tau = 0.500005, Smagorinsky 0.1, central-moment regularised collision")
            .tolerance("all velocities finite; peak lattice speed in (0.005, 0.35)")
            .seed("sim-jlr-paper-tau-regularized:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(_) => QaResult::failed(CASE_ID, "paper-tau regularised shakedown panicked")
            .requested_backend("cpu")
            .actual_backend("cpu")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
    };
    if result.status == sonium_qa::QaStatus::Passed
        && let Err(error) = result.release_eligible()
    {
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
