//! Persistent per-frequency CUDA state with shared production objectives and MMA.

// Rust guideline compliant 2026-02-21
use super::*;
use std::sync::atomic::Ordering;

/// Retains a forward/adjoint CUDA pair for every frequency of a treatment problem.
///
/// Frequencies execute serially on the supplied backend in sorted problem order;
/// aggregation and cost terms are shared with the CPU implementation. State and
/// basis memory grow with frequency count. Preparation propagates pool/backend
/// allocation failures; no automatic CPU fallback or frequency eviction occurs.
/// Each frequency explicitly lags its preconditioners at the preparation design.
pub struct CudaMultiFrequencyRoomTreatmentSession<'a> {
    problem: &'a MultiFrequencyRoomTreatmentProblem,
    pub(super) frequencies: Vec<CudaTreatmentState<'a>>,
}

impl std::fmt::Debug for CudaMultiFrequencyRoomTreatmentSession<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaMultiFrequencyRoomTreatmentSession")
            .field("frequencies", &self.frequencies.len())
            .field("solver", &self.problem.solver.solver_type)
            .finish_non_exhaustive()
    }
}

impl<'a> CudaMultiFrequencyRoomTreatmentSession<'a> {
    /// Prepare resident configured solves for each frequency at one treatment design.
    ///
    /// # Errors
    /// Rejects invalid designs, unsupported workloads/configurations, and native
    /// setup or allocation failures. Partially prepared state is dropped on error.
    pub fn try_prepare(
        problem: &'a MultiFrequencyRoomTreatmentProblem,
        pool: &'a MemoryPool<CudaBackend>,
        workload: FemWorkload,
        treatments: &[f64],
    ) -> Result<Self, RoomTreatmentError> {
        problem.validate_treatments(treatments)?;
        problem.check_passivity()?;
        let frequencies = problem
            .frequencies
            .iter()
            .map(|frequency| {
                // This temporary host view owns no retained data: the prepared state
                // owns its uploads and only borrows the caller's pool.
                let local = RoomTreatmentProblem {
                    base_operator: frequency.base_operator.clone(),
                    treatment_operators: problem.treatment_operators.clone(),
                    rhs: frequency.rhs.clone(),
                    listeners: problem.listeners.clone(),
                    solver: problem.solver.clone(),
                };
                CudaTreatmentState::try_prepare(&local, pool, workload, treatments)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            problem,
            frequencies,
        })
    }

    /// Return the number of treatment variables in the retained problem.
    pub fn treatment_count(&self) -> usize {
        self.problem.treatment_boundaries().len()
    }

    /// Refresh every frequency's preconditioners before committing any replacement.
    ///
    /// Host numeric assembly/factorization and device uploads are explicit setup
    /// work. Old and new preconditioners coexist until all preparations succeed,
    /// increasing peak memory. Physical operators, bases and Krylov state persist;
    /// certified fields become warm-start candidates for the next evaluation.
    ///
    /// # Errors
    /// Rejects invalid designs or preparation failures without installing any
    /// candidate. Backend failures may require recovery before the next evaluation.
    pub fn try_refresh_preconditioners(
        &mut self,
        treatments: &[f64],
    ) -> Result<(), RoomTreatmentError> {
        self.problem.validate_treatments(treatments)?;
        let candidates = self
            .frequencies
            .iter()
            .zip(&self.problem.frequencies)
            .map(|(state, frequency)| {
                let mut matrix = frequency.base_operator.clone();
                for (&alpha, derivative) in treatments.iter().zip(&self.problem.treatment_operators)
                {
                    for (value, delta) in matrix.values.iter_mut().zip(&derivative.values) {
                        *value += alpha * *delta;
                    }
                }
                state.prepare_preconditioners(&matrix)
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (state, candidate) in self.frequencies.iter_mut().zip(candidates) {
            state.install_preconditioners(candidate);
        }
        Ok(())
    }

    /// Evaluate the band objective with resident forward/adjoint frequency states.
    ///
    /// # Errors
    /// Rejects invalid designs/objectives and propagates native execution failures.
    pub fn evaluate(
        &mut self,
        treatments: &[f64],
        objective: &TreatmentObjectiveConfig,
    ) -> Result<TreatmentEvaluation, RoomTreatmentError> {
        self.evaluate_with_cancel(treatments, objective, None)
    }

    /// Evaluate in deterministic frequency order with cancellation between frequencies.
    ///
    /// Completed frequency states remain reusable after cancellation, but no
    /// partial aggregate is returned. A later call re-evaluates the entire band.
    ///
    /// # Errors
    /// Returns an error on cancellation, invalid input or native execution failure.
    pub fn evaluate_with_cancel(
        &mut self,
        treatments: &[f64],
        objective: &TreatmentObjectiveConfig,
        cancel: Option<&AtomicBool>,
    ) -> Result<TreatmentEvaluation, RoomTreatmentError> {
        self.problem.validate_treatments(treatments)?;
        objective.validate(self.frequencies.len())?;
        let mut evaluations = Vec::with_capacity(self.frequencies.len());
        for (index, state) in self.frequencies.iter_mut().enumerate() {
            if cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
                return Err(solve_error(
                    "CUDA treatment frequency execution was cancelled",
                ));
            }
            let result = state.evaluate(treatments, |pressures| match objective.kind {
                TreatmentObjectiveKind::SeatVariance => {
                    let (value, gradient) = seat_variance_pressures(pressures);
                    ((0.0, value, 0.0), gradient)
                }
                TreatmentObjectiveKind::Production => {
                    production_pressures(pressures, objective, objective.target_level_db(index))
                }
            })?;
            evaluations.push(result);
        }
        if cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
            return Err(solve_error(
                "CUDA treatment frequency execution was cancelled",
            ));
        }
        self.problem
            .aggregate_evaluations(treatments, objective, &evaluations)
    }

    /// Run the shared area-constrained MMA loop using retained CUDA frequency states.
    ///
    /// Frequency lanes are serial on this backend; `max_frequency_lanes` is a CPU
    /// scheduling control. Objective, filtering, decay/spatial constraints and
    /// termination semantics are shared with CPU optimization.
    ///
    /// # Errors
    /// Propagates configuration, constraint, objective and native execution failures.
    pub fn optimize(
        &mut self,
        config: &MultiFrequencyRoomTreatmentConfig,
    ) -> Result<MultiFrequencyRoomTreatmentResult, RoomTreatmentError> {
        let problem = self.problem;
        problem.optimize_with_evaluator(config, |treatments| {
            self.evaluate(treatments, &config.objective)
        })
    }
}
