//! Retained CUDA execution for checked source-to-target treatment handoffs.

// Rust guideline compliant 2026-02-21
use super::*;
use crate::treatment_optimization::CudaMultiFrequencyRoomTreatmentSession;

/// Reanalyze projected and baseline target designs with one retained CUDA session.
///
/// The target session must have been prepared for the target mesh and its own
/// solver configuration. Geometry transfer, dimensional validation and result
/// construction are shared with CPU reanalysis. Both target solves are required.
///
/// # Errors
/// Rejects invalid projection/design dimensions and propagates objective or
/// native solver errors. No partial reanalysis report is returned.
pub fn reanalyze_projected_treatment_cuda(
    projection: &BodyFittedTreatmentProjection,
    source_treatments: &[f64],
    target: &mut CudaMultiFrequencyRoomTreatmentSession<'_>,
    baseline_target_treatments: &[f64],
    objective: &TreatmentObjectiveConfig,
) -> Result<BodyFittedTreatmentReanalysis, TreatmentProjectionError> {
    reanalyze_with_evaluator(
        projection,
        source_treatments,
        target.treatment_count(),
        baseline_target_treatments,
        |design| {
            target
                .evaluate(design, objective)
                .map_err(|error| TreatmentProjectionError::Reanalysis(error.to_string()))
        },
    )
}

/// Admit a projected design after retained CUDA target reanalysis and non-regression checks.
///
/// # Errors
/// Propagates reanalysis errors and rejects invalid tolerances or objective regression.
pub fn promote_projected_treatment_cuda(
    projection: &BodyFittedTreatmentProjection,
    source_treatments: &[f64],
    target: &mut CudaMultiFrequencyRoomTreatmentSession<'_>,
    baseline_target_treatments: &[f64],
    objective: &TreatmentObjectiveConfig,
    non_regression_tolerance: f64,
) -> Result<BodyFittedTreatmentPromotion, TreatmentProjectionError> {
    reanalyze_projected_treatment_cuda(
        projection,
        source_treatments,
        target,
        baseline_target_treatments,
        objective,
    )?
    .promote_non_regressing(non_regression_tolerance)
}

/// Optimize on CUDA and promote through independent retained target-mesh reanalysis.
///
/// Source convergence and the target non-regression gate are shared with the
/// CPU workflow. Source and target sessions retain their own operators,
/// configured preconditioners and warm starts; neither falls back to CPU solves.
///
/// # Errors
/// Rejects unconverged source optimization and propagates projection, target
/// solve and promotion failures. No promotion is returned on any failure.
pub fn optimize_and_promote_body_fitted_treatment_cuda(
    source: &mut CudaMultiFrequencyRoomTreatmentSession<'_>,
    source_config: &MultiFrequencyRoomTreatmentConfig,
    projection: &BodyFittedTreatmentProjection,
    target: &mut CudaMultiFrequencyRoomTreatmentSession<'_>,
    baseline_target_treatments: &[f64],
    target_objective: &TreatmentObjectiveConfig,
    target_non_regression_tolerance: f64,
) -> Result<BodyFittedTreatmentOptimizationResult, TreatmentProjectionError> {
    let source = source.optimize(source_config).map_err(|error| {
        TreatmentProjectionError::Reanalysis(format!("source optimization failed: {error}"))
    })?;
    require_converged_source(&source)?;
    let promotion = promote_projected_treatment_cuda(
        projection,
        &source.treatments,
        target,
        baseline_target_treatments,
        target_objective,
        target_non_regression_tolerance,
    )?;
    Ok(BodyFittedTreatmentOptimizationResult { source, promotion })
}
