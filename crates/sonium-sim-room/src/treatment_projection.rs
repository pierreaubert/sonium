//! Checked geometric transfer of fixed-mesh treatment densities.
//!
//! A body-fitted target mesh generally has a different panel layout from the
//! fixed mesh used by the optimizer. [`BodyFittedTreatmentProjection`] maps a
//! source density field to that target layout with normalized compact-support
//! hat kernels and exposes the exact transpose for adjoint quantities. The
//! operator is deliberately geometry-only: callers must run a target FEM
//! reanalysis before treating the projected field as physically validated.

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
#[doc(inline)]
pub use cuda::{
    optimize_and_promote_body_fitted_treatment_cuda, promote_projected_treatment_cuda,
    reanalyze_projected_treatment_cuda,
};

use crate::treatment_optimization::{
    MultiFrequencyRoomTreatmentConfig, MultiFrequencyRoomTreatmentProblem,
    MultiFrequencyRoomTreatmentResult, TreatmentEvaluation, TreatmentObjectiveConfig,
};

/// Recoverable projection-construction or application error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TreatmentProjectionError {
    /// A required position, radius, or density is invalid.
    InvalidInput(&'static str),
    /// A target panel has no source support inside the configured radius.
    MissingSupport(usize),
    /// A vector length does not match the projection topology.
    DimensionMismatch(&'static str),
    /// The target FEM reanalysis could not be evaluated or validated.
    Reanalysis(String),
}

impl std::fmt::Display for TreatmentProjectionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(message) => {
                write!(formatter, "invalid treatment projection: {message}")
            }
            Self::MissingSupport(index) => {
                write!(
                    formatter,
                    "treatment projection target {index} has no source support"
                )
            }
            Self::DimensionMismatch(message) => {
                write!(
                    formatter,
                    "treatment projection dimension mismatch: {message}"
                )
            }
            Self::Reanalysis(message) => {
                write!(
                    formatter,
                    "treatment projection reanalysis failed: {message}"
                )
            }
        }
    }
}

impl std::error::Error for TreatmentProjectionError {}

/// Geometry-only treatment transfer from fixed-mesh source panels to target
/// body-fitted panels.
#[derive(Debug, Clone)]
pub struct BodyFittedTreatmentProjection {
    /// Compact-support radius in metres.
    pub radius_m: f64,
    weights: Vec<Vec<f64>>,
    max_support_distance_m: f64,
}

impl BodyFittedTreatmentProjection {
    /// Construct a normalized source-to-target hat-kernel projection.
    ///
    /// Every target position must have at least one source panel strictly
    /// within `radius_m`; silently assigning a nearest panel would hide a
    /// geometry mismatch, so unsupported targets are rejected.
    pub fn new(
        source_positions_m: &[[f64; 3]],
        target_positions_m: &[[f64; 3]],
        radius_m: f64,
    ) -> Result<Self, TreatmentProjectionError> {
        if source_positions_m.is_empty() || target_positions_m.is_empty() {
            return Err(TreatmentProjectionError::InvalidInput(
                "source and target positions must be non-empty",
            ));
        }
        if !radius_m.is_finite() || radius_m <= 0.0 {
            return Err(TreatmentProjectionError::InvalidInput(
                "projection radius must be finite and positive",
            ));
        }
        if source_positions_m
            .iter()
            .chain(target_positions_m)
            .flatten()
            .any(|coordinate| !coordinate.is_finite())
        {
            return Err(TreatmentProjectionError::InvalidInput(
                "projection positions must be finite",
            ));
        }
        let radius_squared = radius_m * radius_m;
        if !radius_squared.is_finite() {
            return Err(TreatmentProjectionError::InvalidInput(
                "projection radius is too large",
            ));
        }

        let mut weights = Vec::with_capacity(target_positions_m.len());
        let mut max_support_distance_m = 0.0_f64;
        for (target_index, target) in target_positions_m.iter().enumerate() {
            let mut row = vec![0.0; source_positions_m.len()];
            for (source_index, source) in source_positions_m.iter().enumerate() {
                let distance_squared = target
                    .iter()
                    .zip(source)
                    .map(|(left, right)| (left - right).powi(2))
                    .sum::<f64>();
                if !distance_squared.is_finite() {
                    return Err(TreatmentProjectionError::InvalidInput(
                        "projection position distance is non-finite",
                    ));
                }
                if distance_squared < radius_squared {
                    let distance = distance_squared.sqrt();
                    row[source_index] = 1.0 - distance / radius_m;
                    max_support_distance_m = max_support_distance_m.max(distance);
                }
            }
            let row_sum = row.iter().sum::<f64>();
            if !row_sum.is_finite() || row_sum <= 0.0 {
                return Err(TreatmentProjectionError::MissingSupport(target_index));
            }
            for weight in &mut row {
                *weight /= row_sum;
            }
            weights.push(row);
        }

        let projection = Self {
            radius_m,
            weights,
            max_support_distance_m,
        };
        projection.validate(source_positions_m.len(), target_positions_m.len())?;
        Ok(projection)
    }

    /// Number of source fixed-mesh panels.
    pub fn source_count(&self) -> usize {
        self.weights.first().map_or(0, Vec::len)
    }

    /// Number of target body-fitted panels.
    pub fn target_count(&self) -> usize {
        self.weights.len()
    }

    /// Largest source-to-target distance with nonzero interpolation weight.
    pub fn max_support_distance_m(&self) -> f64 {
        self.max_support_distance_m
    }

    /// Validate dimensions, normalized rows, and finite non-negative weights.
    pub fn validate(
        &self,
        source_count: usize,
        target_count: usize,
    ) -> Result<(), TreatmentProjectionError> {
        if source_count == 0 || target_count == 0 || self.weights.len() != target_count {
            return Err(TreatmentProjectionError::DimensionMismatch(
                "projection topology is empty or has the wrong target count",
            ));
        }
        if !self.radius_m.is_finite() || self.radius_m <= 0.0 {
            return Err(TreatmentProjectionError::InvalidInput(
                "projection radius must be finite and positive",
            ));
        }
        if !self.max_support_distance_m.is_finite() || self.max_support_distance_m < 0.0 {
            return Err(TreatmentProjectionError::InvalidInput(
                "projection support distance must be finite and non-negative",
            ));
        }
        for row in &self.weights {
            if row.len() != source_count
                || row
                    .iter()
                    .any(|weight| !weight.is_finite() || *weight < 0.0)
                || (row.iter().sum::<f64>() - 1.0).abs() > 1.0e-10
            {
                return Err(TreatmentProjectionError::InvalidInput(
                    "projection rows must be finite, non-negative, and normalized",
                ));
            }
        }
        Ok(())
    }

    /// Project source treatment densities to the target body-fitted panels.
    pub fn project(&self, source_treatments: &[f64]) -> Result<Vec<f64>, TreatmentProjectionError> {
        if source_treatments.len() != self.source_count() {
            return Err(TreatmentProjectionError::DimensionMismatch(
                "source treatment count does not match projection",
            ));
        }
        if source_treatments
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(TreatmentProjectionError::InvalidInput(
                "source treatments must be finite and lie in [0, 1]",
            ));
        }
        let target_treatments = self
            .weights
            .iter()
            .map(|row| {
                row.iter()
                    .zip(source_treatments)
                    .map(|(weight, treatment)| weight * treatment)
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        if target_treatments
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(TreatmentProjectionError::InvalidInput(
                "projection produced an invalid target treatment",
            ));
        }
        Ok(target_treatments)
    }

    /// Pull a target physical gradient back to source optimization variables.
    pub fn pullback_gradient(
        &self,
        target_gradient: &[f64],
    ) -> Result<Vec<f64>, TreatmentProjectionError> {
        if target_gradient.len() != self.target_count() {
            return Err(TreatmentProjectionError::DimensionMismatch(
                "target gradient count does not match projection",
            ));
        }
        if target_gradient.iter().any(|value| !value.is_finite()) {
            return Err(TreatmentProjectionError::InvalidInput(
                "target gradient must be finite",
            ));
        }
        let mut source_gradient = vec![0.0; self.source_count()];
        for (row, &gradient) in self.weights.iter().zip(target_gradient) {
            for (source_gradient, weight) in source_gradient.iter_mut().zip(row) {
                *source_gradient += gradient * weight;
            }
        }
        if source_gradient.iter().any(|value| !value.is_finite()) {
            return Err(TreatmentProjectionError::InvalidInput(
                "projection pullback gradient is non-finite",
            ));
        }
        Ok(source_gradient)
    }
}

/// Results from evaluating a projected design on the target FEM problem.
///
/// Projection alone is only a geometric transfer. This report pairs the
/// projected evaluation with an explicitly supplied target-mesh baseline so
/// callers can check whether the transfer preserved the target objective.
#[derive(Debug, Clone)]
pub struct BodyFittedTreatmentReanalysis {
    /// Treatment densities on the target body-fitted panels.
    pub target_treatments: Vec<f64>,
    /// Largest source-to-target support distance used by the transfer.
    pub max_support_distance_m: f64,
    /// Target objective evaluated at the supplied baseline design.
    pub baseline: TreatmentEvaluation,
    /// Target objective evaluated at `target_treatments`.
    pub projected: TreatmentEvaluation,
}

/// A body-fitted treatment admitted after explicit target-FEM reanalysis.
///
/// This is intentionally distinct from [`BodyFittedTreatmentReanalysis`]: a
/// report records what the target solve produced, while this type can only be
/// constructed through [`promote_projected_treatment`] or
/// [`BodyFittedTreatmentReanalysis::promote_non_regressing`]. It is an
/// objective-preserving promotion gate, not a substitute for mesh-convergence
/// or measured-room validation.
#[derive(Debug, Clone)]
pub struct BodyFittedTreatmentPromotion {
    /// Treatment densities on the target body-fitted panels.
    pub target_treatments: Vec<f64>,
    /// Target objective evaluated at the promoted design.
    pub evaluation: TreatmentEvaluation,
    /// Promoted objective minus the explicit target baseline objective.
    pub objective_delta: f64,
    /// Largest source-to-target support distance used by the transfer.
    pub max_support_distance_m: f64,
}

/// Fixed-mesh optimization result and its admitted body-fitted promotion.
///
/// Keeping both results together prevents a caller from publishing the
/// source-mesh optimizer result while silently discarding target reanalysis.
#[derive(Debug, Clone)]
pub struct BodyFittedTreatmentOptimizationResult {
    /// Converged fixed-mesh optimizer result.
    pub source: MultiFrequencyRoomTreatmentResult,
    /// Explicit target-FEM promotion of `source.treatments`.
    pub promotion: BodyFittedTreatmentPromotion,
}

impl BodyFittedTreatmentReanalysis {
    /// Difference between projected and baseline target objective values.
    pub fn objective_delta(&self) -> f64 {
        self.projected.objective - self.baseline.objective
    }

    /// Require the projected target objective not to exceed the baseline.
    ///
    /// `tolerance` is an absolute objective tolerance and must be finite and
    /// non-negative. The check is intentionally separate from reanalysis so
    /// workflows may record regressions instead of rejecting them.
    pub fn require_non_regression(&self, tolerance: f64) -> Result<(), TreatmentProjectionError> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(TreatmentProjectionError::InvalidInput(
                "non-regression tolerance must be finite and non-negative",
            ));
        }
        let delta = self.objective_delta();
        if !delta.is_finite() {
            return Err(TreatmentProjectionError::Reanalysis(
                "baseline or projected objective is non-finite".to_string(),
            ));
        }
        if delta > tolerance {
            return Err(TreatmentProjectionError::Reanalysis(format!(
                "projected objective regressed by {delta:.6e}, tolerance {tolerance:.6e}"
            )));
        }
        Ok(())
    }

    /// Admit this target-FEM result when the projected design does not regress.
    pub fn promote_non_regressing(
        self,
        tolerance: f64,
    ) -> Result<BodyFittedTreatmentPromotion, TreatmentProjectionError> {
        self.require_non_regression(tolerance)?;
        if !self.max_support_distance_m.is_finite() || self.max_support_distance_m < 0.0 {
            return Err(TreatmentProjectionError::InvalidInput(
                "reanalysis support distance must be finite non-negative",
            ));
        }
        let objective_delta = self.objective_delta();
        Ok(BodyFittedTreatmentPromotion {
            target_treatments: self.target_treatments,
            evaluation: self.projected,
            objective_delta,
            max_support_distance_m: self.max_support_distance_m,
        })
    }
}

/// Project source treatments and re-evaluate both designs on a target FEM
/// problem.
///
/// The baseline is evaluated on the target problem exactly as supplied; it is
/// not inferred from the source mesh. This makes mesh-transfer regressions
/// observable and prevents a geometry-only projection from being presented as
/// a physically validated reanalysis.
pub fn reanalyze_projected_treatment(
    projection: &BodyFittedTreatmentProjection,
    source_treatments: &[f64],
    target_problem: &MultiFrequencyRoomTreatmentProblem,
    baseline_target_treatments: &[f64],
    objective: &TreatmentObjectiveConfig,
    max_frequency_lanes: Option<usize>,
) -> Result<BodyFittedTreatmentReanalysis, TreatmentProjectionError> {
    reanalyze_with_evaluator(
        projection,
        source_treatments,
        target_problem.treatment_boundaries().len(),
        baseline_target_treatments,
        |design| {
            target_problem
                .evaluate(design, objective, max_frequency_lanes)
                .map_err(|error| TreatmentProjectionError::Reanalysis(error.to_string()))
        },
    )
}

fn reanalyze_with_evaluator(
    projection: &BodyFittedTreatmentProjection,
    source_treatments: &[f64],
    target_count: usize,
    baseline_target_treatments: &[f64],
    mut evaluate: impl FnMut(&[f64]) -> Result<TreatmentEvaluation, TreatmentProjectionError>,
) -> Result<BodyFittedTreatmentReanalysis, TreatmentProjectionError> {
    let target_treatments = projection.project(source_treatments)?;
    if target_count != projection.target_count() {
        return Err(TreatmentProjectionError::DimensionMismatch(
            "projection target count does not match target FEM treatment count",
        ));
    }
    if baseline_target_treatments.len() != target_count {
        return Err(TreatmentProjectionError::DimensionMismatch(
            "baseline target treatment count does not match target FEM treatment count",
        ));
    }

    let baseline = evaluate(baseline_target_treatments)?;
    let projected = evaluate(&target_treatments)?;

    Ok(BodyFittedTreatmentReanalysis {
        target_treatments,
        max_support_distance_m: projection.max_support_distance_m(),
        baseline,
        projected,
    })
}

/// Reanalyze and admit a fixed-mesh design on an explicit body-fitted target.
///
/// The target problem is solved for both the supplied target baseline and the
/// projected design. Promotion succeeds only when the projected objective is
/// within `non_regression_tolerance` of that baseline. This is the intended
/// post-optimization handoff; callers cannot promote a geometry-only
/// projection by skipping the target solve.
pub fn promote_projected_treatment(
    projection: &BodyFittedTreatmentProjection,
    source_treatments: &[f64],
    target_problem: &MultiFrequencyRoomTreatmentProblem,
    baseline_target_treatments: &[f64],
    objective: &TreatmentObjectiveConfig,
    max_frequency_lanes: Option<usize>,
    non_regression_tolerance: f64,
) -> Result<BodyFittedTreatmentPromotion, TreatmentProjectionError> {
    let reanalysis = reanalyze_projected_treatment(
        projection,
        source_treatments,
        target_problem,
        baseline_target_treatments,
        objective,
        max_frequency_lanes,
    )?;
    reanalysis.promote_non_regressing(non_regression_tolerance)
}

/// Optimize fixed-mesh treatment and promote it through target FEM reanalysis.
///
/// The source optimizer must report convergence. Its treatment vector is then
/// projected, re-solved on `target_problem` against the explicit target
/// baseline, and admitted only when the target objective satisfies the
/// configured non-regression tolerance. Solver failures and unconverged source
/// results are recoverable errors; neither can produce a promotion.
pub fn optimize_and_promote_body_fitted_treatment(
    source_problem: &MultiFrequencyRoomTreatmentProblem,
    source_config: &MultiFrequencyRoomTreatmentConfig,
    projection: &BodyFittedTreatmentProjection,
    target_problem: &MultiFrequencyRoomTreatmentProblem,
    baseline_target_treatments: &[f64],
    target_objective: &TreatmentObjectiveConfig,
    target_max_frequency_lanes: Option<usize>,
    target_non_regression_tolerance: f64,
) -> Result<BodyFittedTreatmentOptimizationResult, TreatmentProjectionError> {
    let source = source_problem.optimize(source_config).map_err(|error| {
        TreatmentProjectionError::Reanalysis(format!("source optimization failed: {error}"))
    })?;
    require_converged_source(&source)?;
    let promotion = promote_projected_treatment(
        projection,
        &source.treatments,
        target_problem,
        baseline_target_treatments,
        target_objective,
        target_max_frequency_lanes,
        target_non_regression_tolerance,
    )?;
    Ok(BodyFittedTreatmentOptimizationResult { source, promotion })
}

fn require_converged_source(
    source: &MultiFrequencyRoomTreatmentResult,
) -> Result<(), TreatmentProjectionError> {
    if !source.converged {
        return Err(TreatmentProjectionError::Reanalysis(
            "source treatment optimization did not converge".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::treatment_optimization::{
        ListenerFunctional, MultiFrequencyRoomTreatmentConfig, MultiFrequencyRoomTreatmentProblem,
        TreatmentBoundary, TreatmentEvaluation, TreatmentFrequencySetup,
        TreatmentObjectiveBreakdown, TreatmentObjectiveKind,
    };
    use num_complex::Complex64;
    use sonium_fem::assembly::{HelmholtzAssembler, MassMatrix, StiffnessMatrix};
    use sonium_fem::solver::{SolverConfig, SolverType};
    use std::collections::HashMap;

    #[test]
    fn projection_interpolates_target_layout_and_preserves_constants() {
        let projection = BodyFittedTreatmentProjection::new(
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            &[[0.25, 0.0, 0.0], [0.75, 0.0, 0.0]],
            1.0,
        )
        .unwrap();
        let target = projection.project(&[0.0, 1.0]).unwrap();
        assert!((target[0] - 0.25).abs() < 1.0e-12);
        assert!((target[1] - 0.75).abs() < 1.0e-12);
        assert_eq!(projection.project(&[0.4, 0.4]).unwrap(), vec![0.4, 0.4]);
        assert_eq!(projection.source_count(), 2);
        assert_eq!(projection.target_count(), 2);
        assert!(projection.max_support_distance_m() <= 0.75);
    }

    #[test]
    fn projection_transpose_is_the_adjoint_of_interpolation() {
        let projection = BodyFittedTreatmentProjection::new(
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            &[[0.25, 0.0, 0.0], [0.75, 0.0, 0.0]],
            1.0,
        )
        .unwrap();
        let source = [0.2, 0.8];
        let target_dual = [1.5, -0.25];
        let target = projection.project(&source).unwrap();
        let source_dual = projection.pullback_gradient(&target_dual).unwrap();
        let lhs = target
            .iter()
            .zip(target_dual)
            .map(|(value, dual)| value * dual)
            .sum::<f64>();
        let rhs = source
            .iter()
            .zip(source_dual)
            .map(|(value, dual)| value * dual)
            .sum::<f64>();
        assert!((lhs - rhs).abs() < 1.0e-12);
    }

    #[test]
    fn projection_rejects_missing_support_and_malformed_values() {
        assert!(matches!(
            BodyFittedTreatmentProjection::new(&[[0.0, 0.0, 0.0]], &[[2.0, 0.0, 0.0]], 1.0,),
            Err(TreatmentProjectionError::MissingSupport(0))
        ));
        assert!(
            BodyFittedTreatmentProjection::new(&[[f64::NAN, 0.0, 0.0]], &[[0.0, 0.0, 0.0]], 1.0,)
                .is_err()
        );
        let projection =
            BodyFittedTreatmentProjection::new(&[[0.0, 0.0, 0.0]], &[[0.0, 0.0, 0.0]], 1.0)
                .unwrap();
        assert!(projection.project(&[f64::NAN]).is_err());
        assert!(projection.project(&[0.2, 0.3]).is_err());
        assert!(projection.pullback_gradient(&[f64::INFINITY]).is_err());
        assert!(projection.pullback_gradient(&[0.2, 0.3]).is_err());
    }

    fn direct_target_problem() -> MultiFrequencyRoomTreatmentProblem {
        let stiffness = StiffnessMatrix {
            rows: vec![0],
            cols: vec![0],
            values: vec![1.0],
            dim: 1,
        };
        let mass = MassMatrix {
            rows: vec![0],
            cols: vec![0],
            values: vec![1.0],
            dim: 1,
        };
        let boundary_a = MassMatrix {
            rows: vec![0],
            cols: vec![0],
            values: vec![1.0],
            dim: 1,
        };
        let boundary_b = MassMatrix {
            rows: vec![0],
            cols: vec![0],
            values: vec![0.5],
            dim: 1,
        };
        let assembler = HelmholtzAssembler::from_matrices(
            &stiffness,
            &mass,
            &[(7, boundary_a), (8, boundary_b)],
        );
        let treatments = vec![
            TreatmentBoundary {
                marker: 7,
                coefficient_direction: Complex64::new(0.0, -0.2),
                area_m2: None,
                cost_per_m2: None,
            },
            TreatmentBoundary {
                marker: 8,
                coefficient_direction: Complex64::new(0.0, -0.1),
                area_m2: None,
                cost_per_m2: None,
            },
        ];
        let setup = TreatmentFrequencySetup {
            frequency_hz: 100.0,
            wavenumber: Complex64::new(1.0, 0.0),
            weight: 1.0,
            base_boundary_coefficients: HashMap::new(),
            rhs: vec![Complex64::new(1.0, 0.0)],
        };
        MultiFrequencyRoomTreatmentProblem::from_helmholtz(
            &assembler,
            &[setup],
            &treatments,
            vec![
                ListenerFunctional::at_node(0),
                ListenerFunctional::at_node(0),
            ],
            SolverConfig {
                solver_type: SolverType::Direct,
                ..SolverConfig::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn reanalysis_compares_projected_design_with_target_baseline() {
        let projection = BodyFittedTreatmentProjection::new(
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            &[[0.25, 0.0, 0.0], [0.75, 0.0, 0.0]],
            1.0,
        )
        .unwrap();
        let problem = direct_target_problem();
        let objective = TreatmentObjectiveConfig::default();
        let report = reanalyze_projected_treatment(
            &projection,
            &[0.0, 1.0],
            &problem,
            &[0.0, 1.0],
            &objective,
            Some(1),
        )
        .unwrap();

        assert_eq!(report.target_treatments, vec![0.25, 0.75]);
        assert!(report.baseline.objective.is_finite());
        assert!(report.projected.objective.is_finite());
        assert_eq!(
            report.objective_delta(),
            report.projected.objective - report.baseline.objective
        );
    }

    #[test]
    fn reanalysis_rejects_topology_mismatch_and_checks_non_regression() {
        let projection =
            BodyFittedTreatmentProjection::new(&[[0.0, 0.0, 0.0]], &[[0.0, 0.0, 0.0]], 1.0)
                .unwrap();
        let problem = direct_target_problem();
        let error = reanalyze_projected_treatment(
            &projection,
            &[0.5],
            &problem,
            &[0.0, 1.0],
            &TreatmentObjectiveConfig::default(),
            Some(1),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            TreatmentProjectionError::DimensionMismatch(_)
        ));

        let report = reanalyze_projected_treatment(
            &BodyFittedTreatmentProjection::new(
                &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                1.0,
            )
            .unwrap(),
            &[0.2, 0.8],
            &problem,
            &[0.2, 0.8],
            &TreatmentObjectiveConfig::default(),
            Some(1),
        )
        .unwrap();
        assert!(report.require_non_regression(1.0e-12).is_ok());
        assert!(report.require_non_regression(-1.0).is_err());
    }

    #[test]
    fn promotion_reanalyzes_target_and_preserves_support_provenance() {
        let projection = BodyFittedTreatmentProjection::new(
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            &[[0.25, 0.0, 0.0], [0.75, 0.0, 0.0]],
            1.0,
        )
        .unwrap();
        let promotion = promote_projected_treatment(
            &projection,
            &[0.0, 1.0],
            &direct_target_problem(),
            &[0.25, 0.75],
            &TreatmentObjectiveConfig::default(),
            Some(1),
            1.0e-12,
        )
        .unwrap();
        assert_eq!(promotion.target_treatments, vec![0.25, 0.75]);
        assert!(promotion.evaluation.objective.is_finite());
        assert!(promotion.objective_delta.abs() <= 1.0e-12);
        assert!((promotion.max_support_distance_m - 0.75).abs() <= 1.0e-12);
    }

    #[test]
    fn promotion_gate_rejects_regressed_reanalysis() {
        let evaluation = |objective| TreatmentEvaluation {
            objective,
            breakdown: TreatmentObjectiveBreakdown::default(),
            gradient: vec![0.0],
        };
        let report = BodyFittedTreatmentReanalysis {
            target_treatments: vec![0.5],
            max_support_distance_m: 0.25,
            baseline: evaluation(1.0),
            projected: evaluation(1.1),
        };
        let error = report.promote_non_regressing(0.05).unwrap_err();
        assert!(matches!(error, TreatmentProjectionError::Reanalysis(_)));
    }

    #[test]
    fn optimization_handoff_requires_convergence_and_reanalyzes_target() {
        let projection = BodyFittedTreatmentProjection::new(
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            &[[0.25, 0.0, 0.0], [0.75, 0.0, 0.0]],
            1.0,
        )
        .unwrap();
        let objective = TreatmentObjectiveConfig {
            kind: TreatmentObjectiveKind::SeatVariance,
            ..TreatmentObjectiveConfig::default()
        };
        let source_config = MultiFrequencyRoomTreatmentConfig {
            max_iterations: 4,
            tolerance: 1.0e-3,
            treatment_fraction: 0.5,
            initial_treatment: 0.25,
            objective: objective.clone(),
            max_frequency_lanes: Some(1),
            ..MultiFrequencyRoomTreatmentConfig::default()
        };
        let result = optimize_and_promote_body_fitted_treatment(
            &direct_target_problem(),
            &source_config,
            &projection,
            &direct_target_problem(),
            &[0.25, 0.25],
            &objective,
            Some(1),
            1.0e-12,
        )
        .unwrap();
        assert!(result.source.converged);
        assert_eq!(result.promotion.target_treatments, vec![0.25, 0.25]);
        assert!(result.promotion.objective_delta.abs() <= 1.0e-12);
    }
}
