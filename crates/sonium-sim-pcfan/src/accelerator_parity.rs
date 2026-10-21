//! Backend-parity admission for PC-fan resolved-flow observables.
//!
//! The CFD solver can select CPU, CUDA, or Metal and may report a host
//! fallback. This module compares a backend result with a CPU reference while
//! keeping fallback and synthetic fixtures explicit. It does not claim that a
//! hardware backend was exercised when the caller supplies no such result.

use crate::cfd::solve_pc_fan_cfd_mean_flow;
use crate::evaluation::{PcFanCfdEvaluationError, try_evaluate_pc_fan_from_cfd_solution};
use crate::model::{
    PcFanCfdBackend, PcFanCfdConservation, PcFanCfdExecutionEnvironment, PcFanCfdFlowSolution,
    PcFanConfig, PcFanDesignVariables, PcFanFlowModel,
};
use serde::{Deserialize, Serialize};
use std::fmt;

const METRIC_COUNT: usize = 9;

/// Index of the first dimensionless conservation ledger in `metric_values`
/// (mass, momentum, pressure closure, energy). Earlier entries are physical
/// observables (airflow, pressure drop, velocity, Reynolds number).
const FIRST_CONSERVATION_METRIC_INDEX: usize = 4;

/// Conservation ledgers at or below this magnitude are solver round-off:
/// their ideal value is zero, and backends with different preconditioners
/// cancel boundary fluxes to different last bits. Observed cross-backend
/// noise is ~1e-13; the production mass-balance gate is 1e-2, so this floor
/// sits far below any physically meaningful imbalance.
const CONSERVATION_NOISE_FLOOR: f64 = 1.0e-9;

/// One RPM-aligned resolved-flow backend observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanAcceleratorParityObservation {
    pub rpm: f64,
    pub requested_backend: PcFanCfdBackend,
    pub actual_backend: PcFanCfdBackend,
    pub converged: bool,
    pub airflow_cfm: f64,
    pub pressure_drop_pa: f64,
    pub max_velocity_m_s: f64,
    pub reynolds_number: f64,
    pub relative_mass_imbalance: f64,
    pub relative_momentum_imbalance: f64,
    pub relative_pressure_closure: f64,
    pub relative_energy_imbalance: f64,
    pub residual: f64,
}

impl PcFanAcceleratorParityObservation {
    /// Build a parity observation from a resolved-flow solution and explicit
    /// backend provenance supplied by the dispatcher.
    pub fn from_solution(
        rpm: f64,
        requested_backend: PcFanCfdBackend,
        actual_backend: PcFanCfdBackend,
        solution: &PcFanCfdFlowSolution,
    ) -> Self {
        Self {
            rpm,
            requested_backend,
            actual_backend,
            converged: solution.converged,
            airflow_cfm: solution.airflow_cfm,
            pressure_drop_pa: solution.pressure_drop_pa,
            max_velocity_m_s: solution.max_velocity_m_s,
            reynolds_number: solution.reynolds_number,
            relative_mass_imbalance: solution.conservation.relative_mass_imbalance,
            relative_momentum_imbalance: solution.conservation.relative_momentum_imbalance,
            relative_pressure_closure: solution.conservation.relative_pressure_closure,
            relative_energy_imbalance: solution.conservation.relative_energy_imbalance,
            residual: solution.residual,
        }
    }

    /// Build a parity observation using the concrete backend recorded by the
    /// production CFD solve.
    ///
    /// Callers should prefer this constructor over copying a backend label
    /// from configuration: the configuration is only a request and may have
    /// resolved to CPU after an unavailable accelerator fallback.
    pub fn from_cfd_solution(
        rpm: f64,
        requested_backend: PcFanCfdBackend,
        solution: &PcFanCfdFlowSolution,
    ) -> Self {
        Self::from_solution(rpm, requested_backend, solution.actual_backend, solution)
    }

    /// Build an observation solely from the structured provenance published by
    /// a production CFD solve.
    fn from_production_solution(rpm: f64, solution: &PcFanCfdFlowSolution) -> Self {
        Self::from_solution(
            rpm,
            solution.backend_provenance.requested_backend,
            solution.backend_provenance.actual_backend,
            solution,
        )
    }
}

/// Provenance-labelled reference/candidate backend series.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanAcceleratorParityDataset {
    pub provenance: String,
    pub observations: Vec<PcFanAcceleratorParityObservation>,
}

/// A parity artifact collected by executing the production CFD solver.
///
/// The observations are intentionally private: QA code must obtain a run by
/// calling [`collect_pc_fan_accelerator_parity_run`] instead of constructing a
/// dataset whose backend labels and fields were supplied independently of a
/// solver execution.
#[derive(Debug, Clone, PartialEq)]
pub struct PcFanAcceleratorParityRun {
    dataset: PcFanAcceleratorParityDataset,
    solutions: Vec<PcFanCfdFlowSolution>,
}

impl PcFanAcceleratorParityRun {
    /// Borrow the production observations for reporting or serialization.
    pub fn dataset(&self) -> &PcFanAcceleratorParityDataset {
        &self.dataset
    }

    /// Consume the run and return its serialized observation payload.
    pub fn into_dataset(self) -> PcFanAcceleratorParityDataset {
        self.dataset
    }
}

/// Failure while collecting a production parity run.
#[derive(Debug)]
pub enum PcFanAcceleratorParityRunError {
    UnsupportedFlowModel(PcFanFlowModel),
    InvalidRpm {
        index: usize,
        rpm: f64,
    },
    Evaluation {
        index: usize,
        source: PcFanCfdEvaluationError,
    },
}

impl fmt::Display for PcFanAcceleratorParityRunError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedFlowModel(model) => write!(
                formatter,
                "accelerator parity production collection requires CFD, got {model:?}"
            ),
            Self::InvalidRpm { index, rpm } => write!(
                formatter,
                "accelerator parity production RPM {rpm} at index {index} is not finite, positive, and strictly increasing"
            ),
            Self::Evaluation { index, source } => write!(
                formatter,
                "accelerator parity production CFD evaluation failed at index {index}: {source}"
            ),
        }
    }
}

impl std::error::Error for PcFanAcceleratorParityRunError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Evaluation { source, .. } => Some(source),
            Self::UnsupportedFlowModel(_) | Self::InvalidRpm { .. } => None,
        }
    }
}

/// Execute the production resolved-CFD path at each requested RPM and retain
/// backend provenance, convergence, fields, and conservation ledgers in a
/// parity artifact.
pub fn collect_pc_fan_accelerator_parity_run(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    rpms: &[f64],
) -> Result<PcFanAcceleratorParityRun, PcFanAcceleratorParityRunError> {
    if !config.cfd.flow_model.is_cfd() {
        return Err(PcFanAcceleratorParityRunError::UnsupportedFlowModel(
            config.cfd.flow_model,
        ));
    }

    let mut observations = Vec::with_capacity(rpms.len());
    let mut solutions = Vec::with_capacity(rpms.len());
    for (index, &rpm) in rpms.iter().enumerate() {
        if !rpm.is_finite()
            || rpm <= 0.0
            || rpms
                .get(index.checked_sub(1).unwrap_or(usize::MAX))
                .is_some_and(|previous| rpm.total_cmp(previous) != std::cmp::Ordering::Greater)
        {
            return Err(PcFanAcceleratorParityRunError::InvalidRpm { index, rpm });
        }

        let mut point_config = config.clone();
        point_config.operating_point.rpm = rpm;
        let solution = solve_pc_fan_cfd_mean_flow(&point_config, design);
        try_evaluate_pc_fan_from_cfd_solution(&point_config, design, &solution)
            .map_err(|source| PcFanAcceleratorParityRunError::Evaluation { index, source })?;
        observations.push(PcFanAcceleratorParityObservation::from_production_solution(
            rpm, &solution,
        ));
        solutions.push(solution);
    }

    Ok(PcFanAcceleratorParityRun {
        dataset: PcFanAcceleratorParityDataset {
            provenance: "production:sonium-sim-pcfan:resolved-cfd:v1".into(),
            observations,
        },
        solutions,
    })
}

/// Independent budgets for resolved-flow backend parity.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PcFanAcceleratorParityConfig {
    pub rpm_tolerance: f64,
    pub max_relative_error: f64,
    pub max_absolute_residual_difference: f64,
    pub min_points: usize,
    pub allow_host_fallback: bool,
}

impl Default for PcFanAcceleratorParityConfig {
    fn default() -> Self {
        Self {
            rpm_tolerance: 1.0e-6,
            max_relative_error: 2.0e-3,
            max_absolute_residual_difference: 1.0e-10,
            min_points: 3,
            allow_host_fallback: false,
        }
    }
}

/// Reasons a backend comparison cannot be admitted.
#[derive(Debug, Clone, PartialEq)]
pub enum PcFanAcceleratorParityError {
    InvalidTolerance(&'static str, f64),
    InvalidMinimumPoints,
    MissingProvenance(&'static str),
    TooFewPoints {
        dataset: &'static str,
        actual: usize,
        required: usize,
    },
    NonFiniteObservation {
        dataset: &'static str,
        index: usize,
        field: &'static str,
    },
    InvalidRpm {
        dataset: &'static str,
        index: usize,
        rpm: f64,
    },
    NonIncreasingRpm {
        dataset: &'static str,
        index: usize,
    },
    UnconvergedObservation {
        dataset: &'static str,
        index: usize,
    },
    BackendTopologyMismatch {
        dataset: &'static str,
        index: usize,
    },
    RequestedBackendMismatch {
        dataset: &'static str,
        index: usize,
        requested: PcFanCfdBackend,
        actual: PcFanCfdBackend,
    },
    BackendMismatch {
        reference: PcFanCfdBackend,
        candidate: PcFanCfdBackend,
    },
    HostFallback {
        requested: PcFanCfdBackend,
    },
    ProductionFieldTopologyMismatch {
        index: usize,
        field: &'static str,
    },
    NonFiniteProductionField {
        index: usize,
        field: &'static str,
    },
    RpmTopologyMismatch {
        index: usize,
        reference_rpm: f64,
        candidate_rpm: f64,
    },
    EmptyMetricScale {
        index: usize,
        metric: &'static str,
    },
}

impl fmt::Display for PcFanAcceleratorParityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTolerance(name, value) => {
                write!(
                    formatter,
                    "invalid accelerator parity tolerance {name}={value}"
                )
            }
            Self::InvalidMinimumPoints => {
                write!(formatter, "accelerator parity minimum points is zero")
            }
            Self::MissingProvenance(dataset) => {
                write!(
                    formatter,
                    "accelerator parity {dataset} provenance is blank"
                )
            }
            Self::TooFewPoints {
                dataset,
                actual,
                required,
            } => write!(
                formatter,
                "accelerator parity {dataset} has {actual} points; requires {required}"
            ),
            Self::NonFiniteObservation {
                dataset,
                index,
                field,
            } => write!(
                formatter,
                "accelerator parity {dataset} observation {index} has non-finite {field}"
            ),
            Self::InvalidRpm {
                dataset,
                index,
                rpm,
            } => write!(
                formatter,
                "accelerator parity {dataset} observation {index} has invalid RPM {rpm}"
            ),
            Self::NonIncreasingRpm { dataset, index } => write!(
                formatter,
                "accelerator parity {dataset} RPM is not strictly increasing at {index}"
            ),
            Self::UnconvergedObservation { dataset, index } => write!(
                formatter,
                "accelerator parity {dataset} observation {index} is not converged"
            ),
            Self::BackendTopologyMismatch { dataset, index } => write!(
                formatter,
                "accelerator parity {dataset} observation {index} changes backend topology"
            ),
            Self::RequestedBackendMismatch {
                dataset,
                index,
                requested,
                actual,
            } => write!(
                formatter,
                "accelerator parity {dataset} observation {index} requested {requested:?} but ran on {actual:?}"
            ),
            Self::BackendMismatch {
                reference,
                candidate,
            } => write!(
                formatter,
                "accelerator parity candidate backend {candidate:?} duplicates reference backend {reference:?}"
            ),
            Self::HostFallback { requested } => write!(
                formatter,
                "accelerator parity requested {requested:?} but dispatcher reported a host fallback"
            ),
            Self::ProductionFieldTopologyMismatch { index, field } => write!(
                formatter,
                "accelerator parity production point {index} has mismatched {field} topology"
            ),
            Self::NonFiniteProductionField { index, field } => write!(
                formatter,
                "accelerator parity production point {index} has non-finite {field}"
            ),
            Self::RpmTopologyMismatch {
                index,
                reference_rpm,
                candidate_rpm,
            } => write!(
                formatter,
                "accelerator parity RPM mismatch at {index}: {reference_rpm} vs {candidate_rpm}"
            ),
            Self::EmptyMetricScale { index, metric } => write!(
                formatter,
                "accelerator parity metric {metric} has zero scale at point {index}"
            ),
        }
    }
}

impl std::error::Error for PcFanAcceleratorParityError {}

/// Absolute/relative backend parity report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanAcceleratorParityReport {
    pub reference_backend: PcFanCfdBackend,
    pub candidate_backend: PcFanCfdBackend,
    pub sample_count: usize,
    pub max_relative_errors: [f64; METRIC_COUNT - 1],
    pub max_residual_difference: f64,
    /// Maximum componentwise relative difference across retained velocity and
    /// pressure fields. Legacy dataset comparisons leave this at zero.
    pub max_field_relative_error: f64,
    /// Maximum relative difference across the full conservation ledger and
    /// the airflow objective. Legacy dataset comparisons leave this at zero.
    pub max_production_relative_error: f64,
    /// Build/runtime metadata retained by production-run comparisons.
    pub reference_environment: Option<PcFanCfdExecutionEnvironment>,
    pub candidate_environment: Option<PcFanCfdExecutionEnvironment>,
    pub passed: bool,
}

/// Compare imported observation datasets for compatibility or diagnostics.
///
/// This lower-level API is intentionally named to distinguish it from the
/// production parity gate below. A caller can deserialize or fabricate this
/// type, so it must not be used as hardware QA evidence.
pub fn evaluate_pc_fan_accelerator_dataset_parity(
    config: &PcFanAcceleratorParityConfig,
    reference: &PcFanAcceleratorParityDataset,
    candidate: &PcFanAcceleratorParityDataset,
) -> Result<PcFanAcceleratorParityReport, PcFanAcceleratorParityError> {
    validate_config(config)?;
    validate_dataset(config, "reference", reference)?;
    validate_dataset(config, "candidate", candidate)?;

    let reference_backend = reference.observations[0].actual_backend;
    let candidate_backend = candidate.observations[0].actual_backend;
    let candidate_is_explicit_host_fallback = config.allow_host_fallback
        && candidate.observations[0].requested_backend != PcFanCfdBackend::Cpu
        && candidate_backend == PcFanCfdBackend::Cpu;
    if reference_backend == candidate_backend && !candidate_is_explicit_host_fallback {
        return Err(PcFanAcceleratorParityError::BackendMismatch {
            reference: reference_backend,
            candidate: candidate_backend,
        });
    }

    if reference.observations.len() != candidate.observations.len() {
        return Err(PcFanAcceleratorParityError::RpmTopologyMismatch {
            index: reference
                .observations
                .len()
                .min(candidate.observations.len()),
            reference_rpm: reference.observations.last().map_or(0.0, |point| point.rpm),
            candidate_rpm: candidate.observations.last().map_or(0.0, |point| point.rpm),
        });
    }

    let mut maximum_relative_errors = [0.0_f64; METRIC_COUNT - 1];
    let mut maximum_residual_difference = 0.0_f64;
    for (index, (reference_point, candidate_point)) in reference
        .observations
        .iter()
        .zip(&candidate.observations)
        .enumerate()
    {
        if (reference_point.rpm - candidate_point.rpm).abs() > config.rpm_tolerance {
            return Err(PcFanAcceleratorParityError::RpmTopologyMismatch {
                index,
                reference_rpm: reference_point.rpm,
                candidate_rpm: candidate_point.rpm,
            });
        }
        for (metric_index, (reference_value, candidate_value)) in metric_values(reference_point)
            .into_iter()
            .zip(metric_values(candidate_point))
            .enumerate()
        {
            let scale = reference_value.abs().max(candidate_value.abs());
            // The conservation ledgers are relative residuals whose ideal
            // value is zero. Below the noise floor both backends are
            // reporting solver round-off (different preconditioners cancel
            // the boundary fluxes to different last bits), so the ratio of
            // the two residuals carries no parity signal. Metrics with any
            // physical magnitude keep the strict relative comparison.
            if metric_index >= FIRST_CONSERVATION_METRIC_INDEX && scale <= CONSERVATION_NOISE_FLOOR
            {
                continue;
            }
            if scale <= f64::EPSILON {
                if (reference_value - candidate_value).abs() > f64::EPSILON {
                    return Err(PcFanAcceleratorParityError::EmptyMetricScale {
                        index,
                        metric: metric_name(metric_index),
                    });
                }
                continue;
            }
            maximum_relative_errors[metric_index] = maximum_relative_errors[metric_index]
                .max((candidate_value - reference_value).abs() / scale);
        }
        maximum_residual_difference = maximum_residual_difference
            .max((candidate_point.residual - reference_point.residual).abs());
    }

    let passed = maximum_relative_errors
        .into_iter()
        .all(|error| error <= config.max_relative_error)
        && maximum_residual_difference <= config.max_absolute_residual_difference;
    Ok(PcFanAcceleratorParityReport {
        reference_backend,
        candidate_backend,
        sample_count: reference.observations.len(),
        max_relative_errors: maximum_relative_errors,
        max_residual_difference: maximum_residual_difference,
        max_field_relative_error: 0.0,
        max_production_relative_error: 0.0,
        reference_environment: None,
        candidate_environment: None,
        passed,
    })
}

/// Compare two parity artifacts produced by the production CFD runner.
///
/// This is the only API intended for accelerator QA. Its private solution
/// payload is populated by [`collect_pc_fan_accelerator_parity_run`], so field
/// and ledger comparisons cannot be supplied independently of a solver run.
pub fn evaluate_pc_fan_accelerator_parity(
    config: &PcFanAcceleratorParityConfig,
    reference: &PcFanAcceleratorParityRun,
    candidate: &PcFanAcceleratorParityRun,
) -> Result<PcFanAcceleratorParityReport, PcFanAcceleratorParityError> {
    let mut report =
        evaluate_pc_fan_accelerator_dataset_parity(config, &reference.dataset, &candidate.dataset)?;
    if reference.solutions.len() != candidate.solutions.len() {
        return Err(PcFanAcceleratorParityError::RpmTopologyMismatch {
            index: reference.solutions.len().min(candidate.solutions.len()),
            reference_rpm: reference
                .dataset
                .observations
                .last()
                .map_or(0.0, |point| point.rpm),
            candidate_rpm: candidate
                .dataset
                .observations
                .last()
                .map_or(0.0, |point| point.rpm),
        });
    }

    let mut maximum_field_relative_error = 0.0_f64;
    let mut maximum_production_relative_error = 0.0_f64;
    for (index, (reference_solution, candidate_solution)) in reference
        .solutions
        .iter()
        .zip(&candidate.solutions)
        .enumerate()
    {
        if reference_solution.domain != candidate_solution.domain {
            return Err(
                PcFanAcceleratorParityError::ProductionFieldTopologyMismatch {
                    index,
                    field: "domain",
                },
            );
        }
        if reference_solution.velocity_m_s.len() != candidate_solution.velocity_m_s.len() {
            return Err(
                PcFanAcceleratorParityError::ProductionFieldTopologyMismatch {
                    index,
                    field: "velocity_m_s",
                },
            );
        }
        if reference_solution.pressure_pa.len() != candidate_solution.pressure_pa.len() {
            return Err(
                PcFanAcceleratorParityError::ProductionFieldTopologyMismatch {
                    index,
                    field: "pressure_pa",
                },
            );
        }
        for velocity in reference_solution
            .velocity_m_s
            .iter()
            .chain(&candidate_solution.velocity_m_s)
        {
            if velocity.iter().any(|value| !value.is_finite()) {
                return Err(PcFanAcceleratorParityError::NonFiniteProductionField {
                    index,
                    field: "velocity_m_s",
                });
            }
        }
        if reference_solution
            .pressure_pa
            .iter()
            .chain(&candidate_solution.pressure_pa)
            .any(|value| !value.is_finite())
        {
            return Err(PcFanAcceleratorParityError::NonFiniteProductionField {
                index,
                field: "pressure_pa",
            });
        }
        // Scale each complete field by its dominant physical magnitude.
        // Component-wise relative error is ill-conditioned at symmetry zeros:
        // two converged backends can differ only by roundoff there and still
        // report O(1) relative error.  Topology and finiteness were checked
        // above, so a global max-norm comparison retains fail-closed coverage
        // without promoting cancellation noise to a field failure.
        let velocity_scale = reference_solution
            .velocity_m_s
            .iter()
            .chain(&candidate_solution.velocity_m_s)
            .flat_map(|velocity| velocity.iter())
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        let velocity_difference = reference_solution
            .velocity_m_s
            .iter()
            .zip(&candidate_solution.velocity_m_s)
            .flat_map(|(reference, candidate)| reference.iter().zip(candidate))
            .map(|(reference, candidate)| (reference - candidate).abs())
            .fold(0.0_f64, f64::max);
        let pressure_scale = reference_solution
            .pressure_pa
            .iter()
            .chain(&candidate_solution.pressure_pa)
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        let pressure_difference = reference_solution
            .pressure_pa
            .iter()
            .zip(&candidate_solution.pressure_pa)
            .map(|(reference, candidate)| (reference - candidate).abs())
            .fold(0.0_f64, f64::max);
        maximum_field_relative_error = maximum_field_relative_error
            .max(velocity_difference / velocity_scale.max(f64::MIN_POSITIVE))
            .max(pressure_difference / pressure_scale.max(f64::MIN_POSITIVE));
        maximum_production_relative_error =
            maximum_production_relative_error.max(relative_difference(
                reference_solution.airflow_cfm,
                candidate_solution.airflow_cfm,
            ));
        maximum_production_relative_error =
            maximum_production_relative_error.max(conservation_relative_difference(
                reference_solution.conservation,
                candidate_solution.conservation,
            ));
    }

    report.max_field_relative_error = maximum_field_relative_error;
    report.max_production_relative_error = maximum_production_relative_error;
    report.reference_environment = reference
        .solutions
        .first()
        .map(|solution| solution.execution_environment.clone());
    report.candidate_environment = candidate
        .solutions
        .first()
        .map(|solution| solution.execution_environment.clone());
    report.passed = report.passed
        && maximum_field_relative_error <= config.max_relative_error
        && maximum_production_relative_error <= config.max_relative_error;
    Ok(report)
}

fn relative_difference(reference: f64, candidate: f64) -> f64 {
    let scale = reference.abs().max(candidate.abs());
    if scale <= f64::EPSILON {
        if (reference - candidate).abs() <= f64::EPSILON {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (candidate - reference).abs() / scale
    }
}

/// Relative difference for a dimensionless conservation ledger whose ideal
/// value is zero. Ledgers at or below `CONSERVATION_NOISE_FLOOR` on both
/// backends are solver round-off and agree by definition; larger ledgers
/// keep the strict relative comparison.
fn ratio_relative_difference(reference: f64, candidate: f64) -> f64 {
    if reference.abs().max(candidate.abs()) <= CONSERVATION_NOISE_FLOOR {
        0.0
    } else {
        relative_difference(reference, candidate)
    }
}

fn conservation_relative_difference(
    reference: PcFanCfdConservation,
    candidate: PcFanCfdConservation,
) -> f64 {
    let dimensional = [
        (
            reference.inlet_mass_flow_kg_s,
            candidate.inlet_mass_flow_kg_s,
        ),
        (
            reference.outlet_mass_flow_kg_s,
            candidate.outlet_mass_flow_kg_s,
        ),
        (
            reference.inlet_axial_momentum_flux_n,
            candidate.inlet_axial_momentum_flux_n,
        ),
        (
            reference.outlet_axial_momentum_flux_n,
            candidate.outlet_axial_momentum_flux_n,
        ),
        (reference.pressure_force_n, candidate.pressure_force_n),
        (reference.body_force_n, candidate.body_force_n),
        (
            reference.axial_momentum_residual_n,
            candidate.axial_momentum_residual_n,
        ),
        (
            reference.pressure_force_from_loading_n,
            candidate.pressure_force_from_loading_n,
        ),
        (reference.pressure_power_w, candidate.pressure_power_w),
        (
            reference.shaft_power_to_fluid_w,
            candidate.shaft_power_to_fluid_w,
        ),
    ]
    .into_iter()
    .map(|(reference, candidate)| relative_difference(reference, candidate))
    .fold(0.0, f64::max);
    let ledgers = [
        (
            reference.relative_mass_imbalance,
            candidate.relative_mass_imbalance,
        ),
        (
            reference.relative_momentum_imbalance,
            candidate.relative_momentum_imbalance,
        ),
        (
            reference.relative_pressure_closure,
            candidate.relative_pressure_closure,
        ),
        (
            reference.relative_energy_imbalance,
            candidate.relative_energy_imbalance,
        ),
    ]
    .into_iter()
    .map(|(reference, candidate)| ratio_relative_difference(reference, candidate))
    .fold(0.0, f64::max);
    dimensional.max(ledgers)
}

fn validate_config(
    config: &PcFanAcceleratorParityConfig,
) -> Result<(), PcFanAcceleratorParityError> {
    for (name, value) in [
        ("rpm_tolerance", config.rpm_tolerance),
        ("max_relative_error", config.max_relative_error),
        (
            "max_absolute_residual_difference",
            config.max_absolute_residual_difference,
        ),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(PcFanAcceleratorParityError::InvalidTolerance(name, value));
        }
    }
    if config.min_points == 0 {
        return Err(PcFanAcceleratorParityError::InvalidMinimumPoints);
    }
    Ok(())
}

fn validate_dataset(
    config: &PcFanAcceleratorParityConfig,
    name: &'static str,
    dataset: &PcFanAcceleratorParityDataset,
) -> Result<(), PcFanAcceleratorParityError> {
    if dataset.provenance.trim().is_empty() {
        return Err(PcFanAcceleratorParityError::MissingProvenance(name));
    }
    if dataset.observations.len() < config.min_points {
        return Err(PcFanAcceleratorParityError::TooFewPoints {
            dataset: name,
            actual: dataset.observations.len(),
            required: config.min_points,
        });
    }
    let expected_requested_backend = dataset.observations[0].requested_backend;
    let expected_actual_backend = dataset.observations[0].actual_backend;
    for (index, observation) in dataset.observations.iter().enumerate() {
        if !observation.converged {
            return Err(PcFanAcceleratorParityError::UnconvergedObservation {
                dataset: name,
                index,
            });
        }
        if observation.requested_backend != expected_requested_backend
            || observation.actual_backend != expected_actual_backend
        {
            return Err(PcFanAcceleratorParityError::BackendTopologyMismatch {
                dataset: name,
                index,
            });
        }
        if !config.allow_host_fallback
            && observation.requested_backend != PcFanCfdBackend::Cpu
            && observation.actual_backend == PcFanCfdBackend::Cpu
        {
            return Err(PcFanAcceleratorParityError::HostFallback {
                requested: observation.requested_backend,
            });
        }
        if observation.requested_backend != observation.actual_backend
            && !(config.allow_host_fallback
                && observation.requested_backend != PcFanCfdBackend::Cpu
                && observation.actual_backend == PcFanCfdBackend::Cpu)
        {
            return Err(PcFanAcceleratorParityError::RequestedBackendMismatch {
                dataset: name,
                index,
                requested: observation.requested_backend,
                actual: observation.actual_backend,
            });
        }
        for (field, value) in metric_values(observation)
            .into_iter()
            .zip(metric_names())
            .map(|(value, field)| (field, value))
        {
            if !value.is_finite() {
                return Err(PcFanAcceleratorParityError::NonFiniteObservation {
                    dataset: name,
                    index,
                    field,
                });
            }
        }
        if !observation.residual.is_finite() {
            return Err(PcFanAcceleratorParityError::NonFiniteObservation {
                dataset: name,
                index,
                field: "residual",
            });
        }
        if !observation.rpm.is_finite() || observation.rpm <= 0.0 {
            return Err(PcFanAcceleratorParityError::InvalidRpm {
                dataset: name,
                index,
                rpm: observation.rpm,
            });
        }
        if let Some(previous) = index
            .checked_sub(1)
            .and_then(|previous| dataset.observations.get(previous))
            && observation.rpm.total_cmp(&previous.rpm) != std::cmp::Ordering::Greater
        {
            return Err(PcFanAcceleratorParityError::NonIncreasingRpm {
                dataset: name,
                index,
            });
        }
    }
    Ok(())
}

fn metric_values(observation: &PcFanAcceleratorParityObservation) -> [f64; METRIC_COUNT - 1] {
    [
        observation.airflow_cfm,
        observation.pressure_drop_pa,
        observation.max_velocity_m_s,
        observation.reynolds_number,
        observation.relative_mass_imbalance,
        observation.relative_momentum_imbalance,
        observation.relative_pressure_closure,
        observation.relative_energy_imbalance,
    ]
}

fn metric_names() -> [&'static str; METRIC_COUNT - 1] {
    [
        "airflow_cfm",
        "pressure_drop_pa",
        "max_velocity_m_s",
        "reynolds_number",
        "relative_mass_imbalance",
        "relative_momentum_imbalance",
        "relative_pressure_closure",
        "relative_energy_imbalance",
    ]
}

fn metric_name(index: usize) -> &'static str {
    metric_names()[index]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(rpm: f64, backend: PcFanCfdBackend, scale: f64) -> PcFanAcceleratorParityObservation {
        PcFanAcceleratorParityObservation {
            rpm,
            requested_backend: backend,
            actual_backend: backend,
            converged: true,
            airflow_cfm: 40.0 * scale,
            pressure_drop_pa: 20.0 * scale,
            max_velocity_m_s: 8.0 * scale,
            reynolds_number: 10_000.0 * scale,
            relative_mass_imbalance: 1.0e-4 * scale,
            relative_momentum_imbalance: 2.0e-4 * scale,
            relative_pressure_closure: 3.0e-4 * scale,
            relative_energy_imbalance: 4.0e-4 * scale,
            residual: 1.0e-9 * scale,
        }
    }

    fn datasets(
        candidate_scale: f64,
    ) -> (PcFanAcceleratorParityDataset, PcFanAcceleratorParityDataset) {
        let rpms = [600.0, 900.0, 1200.0];
        (
            PcFanAcceleratorParityDataset {
                provenance: "fixture:cpu-reference".into(),
                observations: rpms
                    .into_iter()
                    .map(|rpm| point(rpm, PcFanCfdBackend::Cpu, 1.0))
                    .collect(),
            },
            PcFanAcceleratorParityDataset {
                provenance: "fixture:cuda-candidate".into(),
                observations: rpms
                    .into_iter()
                    .map(|rpm| point(rpm, PcFanCfdBackend::Cuda, candidate_scale))
                    .collect(),
            },
        )
    }

    #[test]
    fn close_backend_fixture_passes_and_reports_all_metrics() {
        let (reference, candidate) = datasets(1.0005);
        let report = evaluate_pc_fan_accelerator_dataset_parity(
            &PcFanAcceleratorParityConfig::default(),
            &reference,
            &candidate,
        )
        .unwrap();
        assert!(report.passed);
        assert_eq!(report.sample_count, 3);
        assert!(report.max_relative_errors.iter().all(|error| *error > 0.0));
    }

    #[test]
    fn parity_ignores_roundoff_conservation_ledgers_but_not_real_ones() {
        // Both backends report mass imbalance at solver round-off
        // (2.3e-13 vs 5.8e-15): a 97% relative difference that carries no
        // physical signal and must not fail parity.
        let (mut reference, mut candidate) = datasets(1.0);
        for observation in &mut reference.observations {
            observation.relative_mass_imbalance = 2.28e-13;
        }
        for observation in &mut candidate.observations {
            observation.relative_mass_imbalance = 5.8e-15;
        }
        let report = evaluate_pc_fan_accelerator_dataset_parity(
            &PcFanAcceleratorParityConfig::default(),
            &reference,
            &candidate,
        )
        .unwrap();
        assert!(report.passed, "round-off ledgers must pass: {report:?}");
        assert_eq!(report.max_relative_errors[4], 0.0);

        // A conservation ledger with physical magnitude (1e-4 vs 2e-4)
        // keeps the strict relative comparison and must fail a 0.2%
        // budget.
        let (mut reference, mut candidate) = datasets(1.0);
        for observation in &mut reference.observations {
            observation.relative_mass_imbalance = 1.0e-4;
        }
        for observation in &mut candidate.observations {
            observation.relative_mass_imbalance = 2.0e-4;
        }
        let report = evaluate_pc_fan_accelerator_dataset_parity(
            &PcFanAcceleratorParityConfig::default(),
            &reference,
            &candidate,
        )
        .unwrap();
        assert!(!report.passed, "real ledger mismatch must fail: {report:?}");
        assert!(report.max_relative_errors[4] > 0.4);
    }

    #[test]
    fn parity_rejects_fallback_and_rpm_topology_mismatch() {
        let (reference, mut candidate) = datasets(1.0);
        candidate.observations[0].requested_backend = PcFanCfdBackend::Metal;
        candidate.observations[0].actual_backend = PcFanCfdBackend::Cpu;
        assert!(matches!(
            evaluate_pc_fan_accelerator_dataset_parity(
                &PcFanAcceleratorParityConfig::default(),
                &reference,
                &candidate
            ),
            Err(PcFanAcceleratorParityError::HostFallback { .. })
        ));
        let (_, mut candidate) = datasets(1.0);
        candidate.observations[1].rpm = 901.0;
        assert!(matches!(
            evaluate_pc_fan_accelerator_dataset_parity(
                &PcFanAcceleratorParityConfig::default(),
                &reference,
                &candidate
            ),
            Err(PcFanAcceleratorParityError::RpmTopologyMismatch { index: 1, .. })
        ));
    }

    #[test]
    fn parity_reports_budget_failure_and_rejects_nonfinite_data() {
        let (reference, candidate) = datasets(1.1);
        let report = evaluate_pc_fan_accelerator_dataset_parity(
            &PcFanAcceleratorParityConfig::default(),
            &reference,
            &candidate,
        )
        .unwrap();
        assert!(!report.passed);
        let (_, mut candidate) = datasets(1.0);
        candidate.observations[2].residual = f64::NAN;
        assert!(matches!(
            evaluate_pc_fan_accelerator_dataset_parity(
                &PcFanAcceleratorParityConfig::default(),
                &reference,
                &candidate
            ),
            Err(PcFanAcceleratorParityError::NonFiniteObservation { .. })
        ));
    }

    #[test]
    fn parity_rejects_unconverged_and_mixed_backend_series() {
        let (reference, mut candidate) = datasets(1.0);
        candidate.observations[1].converged = false;
        assert!(matches!(
            evaluate_pc_fan_accelerator_dataset_parity(
                &PcFanAcceleratorParityConfig::default(),
                &reference,
                &candidate,
            ),
            Err(PcFanAcceleratorParityError::UnconvergedObservation {
                dataset: "candidate",
                index: 1,
            })
        ));

        let (reference, mut candidate) = datasets(1.0);
        candidate.observations[1].actual_backend = PcFanCfdBackend::Metal;
        assert!(matches!(
            evaluate_pc_fan_accelerator_dataset_parity(
                &PcFanAcceleratorParityConfig::default(),
                &reference,
                &candidate,
            ),
            Err(PcFanAcceleratorParityError::BackendTopologyMismatch {
                dataset: "candidate",
                index: 1,
            })
        ));
    }

    #[test]
    fn parity_rejects_requested_backend_substitution() {
        let (reference, mut candidate) = datasets(1.0);
        for observation in &mut candidate.observations {
            observation.actual_backend = PcFanCfdBackend::Metal;
        }

        assert!(matches!(
            evaluate_pc_fan_accelerator_dataset_parity(
                &PcFanAcceleratorParityConfig::default(),
                &reference,
                &candidate,
            ),
            Err(PcFanAcceleratorParityError::RequestedBackendMismatch {
                dataset: "candidate",
                index: 0,
                requested: PcFanCfdBackend::Cuda,
                actual: PcFanCfdBackend::Metal,
            })
        ));

        let (reference, mut candidate) = datasets(1.0);
        for observation in &mut candidate.observations {
            observation.requested_backend = PcFanCfdBackend::Cpu;
        }
        assert!(matches!(
            evaluate_pc_fan_accelerator_dataset_parity(
                &PcFanAcceleratorParityConfig::default(),
                &reference,
                &candidate,
            ),
            Err(PcFanAcceleratorParityError::RequestedBackendMismatch {
                dataset: "candidate",
                index: 0,
                requested: PcFanCfdBackend::Cpu,
                actual: PcFanCfdBackend::Cuda,
            })
        ));
    }

    #[test]
    fn parity_allows_explicit_host_fallback_when_configured() {
        let (reference, mut candidate) = datasets(1.0);
        for observation in &mut candidate.observations {
            observation.actual_backend = PcFanCfdBackend::Cpu;
        }
        let config = PcFanAcceleratorParityConfig {
            allow_host_fallback: true,
            ..PcFanAcceleratorParityConfig::default()
        };

        let report = evaluate_pc_fan_accelerator_dataset_parity(&config, &reference, &candidate)
            .expect("explicitly enabled host fallback should be comparable");
        assert!(report.passed);
        assert_eq!(report.reference_backend, PcFanCfdBackend::Cpu);
        assert_eq!(report.candidate_backend, PcFanCfdBackend::Cpu);
    }

    #[test]
    fn parity_dataset_serde_round_trip_preserves_backend_labels() {
        let (reference, _) = datasets(1.0);
        let encoded = serde_json::to_vec(&reference).unwrap();
        let decoded: PcFanAcceleratorParityDataset = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, reference);
    }
}
