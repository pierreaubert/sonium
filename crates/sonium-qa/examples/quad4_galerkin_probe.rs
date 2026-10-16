//! Diagnostic entry point for the canonical continuous quadratic Quad4 Mie study.

// Rust guideline compliant 2026-02-21

#[path = "../../sonium-bem/tests/continuous_quadratic_quad4_transmission_mie_convergence.rs"]
mod packet;

use packet::run_continuous_quadratic_quad4_transmission_mie_convergence;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let metrics = run_continuous_quadratic_quad4_transmission_mie_convergence();
    println!("{metrics:?}");
    if metrics.fine_error > 0.01
        || metrics.medium_error > 0.65 * metrics.coarse_error
        || metrics.fine_error > 0.65 * metrics.medium_error
        || [
            metrics.coarse_residual,
            metrics.medium_residual,
            metrics.fine_residual,
        ]
        .iter()
        .any(|r| !r.is_finite() || *r > 1e-8)
        || [
            metrics.coarse_solution_error,
            metrics.medium_solution_error,
            metrics.fine_solution_error,
        ]
        .iter()
        .any(|r| !r.is_finite() || *r > 1e-8)
        || metrics.fine_dofs <= metrics.coarse_dofs
        || metrics.fine_panels <= metrics.coarse_panels
    {
        return Err("original Quad4 accuracy/convergence/equation budgets not met".into());
    }
    Ok(())
}
