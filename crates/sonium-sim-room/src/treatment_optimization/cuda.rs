//! Retained forward and adjoint CUDA solves in the room-treatment MMA loop.

// Rust guideline compliant 2026-02-21
use super::*;
use sonium_compute::{
    Backend, Buffer, CudaBackend, MemoryPool, PersistentDeviceCsr, ReductionMode,
};
use sonium_fem::backend::{FemAction, FemPhysics, FemWorkload};
use sonium_fem::solver::device::{ConfiguredFemPreconditioner, CudaConfiguredFemSolveSession};
use sonium_solvers::iterative::DeviceInitialGuess;

fn solve_error(error: impl std::fmt::Display) -> RoomTreatmentError {
    RoomTreatmentError::Solve(error.to_string())
}

struct Phase<'a> {
    solver: CudaConfiguredFemSolveSession<'a>,
    base_values: Buffer<Complex64, CudaBackend>,
    treatment_values: Vec<Buffer<Complex64, CudaBackend>>,
    rhs: Buffer<Complex64, CudaBackend>,
    warm: bool,
}

impl<'a> Phase<'a> {
    fn prepare(
        pool: &'a MemoryPool<CudaBackend>,
        workload: FemWorkload,
        matrix: &CsrMatrix<Complex64>,
        problem: &RoomTreatmentProblem,
    ) -> Result<Self, RoomTreatmentError> {
        let solver = CudaConfiguredFemSolveSession::try_from_csr(
            pool,
            workload,
            &problem.solver,
            matrix,
            &vec![Complex64::default(); matrix.num_rows],
        )
        .map_err(solve_error)?;
        let upload_values = |matrix: &CsrMatrix<Complex64>| {
            let oriented = if workload.action == FemAction::Hermitian {
                conjugate_transpose(matrix)
            } else {
                matrix.clone()
            };
            if solver.session().pattern()
                != (
                    oriented.row_ptrs.as_slice(),
                    oriented.col_indices.as_slice(),
                )
            {
                return Err(RoomTreatmentError::InvalidProblem(
                    "CUDA treatment bases must share the canonical operator pattern",
                ));
            }
            let mut values = pool.try_alloc(oriented.values.len()).map_err(solve_error)?;
            pool.backend()
                .try_copy_h2d(&oriented.values, values.raw_mut())
                .map_err(solve_error)?;
            Ok(values)
        };
        let base_values = upload_values(&problem.base_operator)?;
        let treatment_values = problem
            .treatment_operators
            .iter()
            .map(upload_values)
            .collect::<Result<Vec<_>, _>>()?;
        let mut rhs = pool.try_alloc(matrix.num_rows).map_err(solve_error)?;
        let initial_rhs = if workload.action == FemAction::Forward {
            problem.rhs.clone()
        } else {
            vec![Complex64::default(); matrix.num_rows]
        };
        pool.backend()
            .try_copy_h2d(&initial_rhs, rhs.raw_mut())
            .map_err(solve_error)?;
        Ok(Self {
            base_values,
            treatment_values,
            rhs,
            solver,
            warm: false,
        })
    }

    fn update(&mut self, treatments: &[f64]) -> Result<(), RoomTreatmentError> {
        if treatments.len() != self.treatment_values.len() {
            return Err(RoomTreatmentError::InvalidProblem(
                "CUDA treatment coefficient count changed",
            ));
        }
        let terms: Vec<_> = treatments
            .iter()
            .zip(&self.treatment_values)
            .map(|(&alpha, values)| (Complex64::new(alpha, 0.0), values.raw()))
            .collect();
        self.solver
            .session_mut()
            .try_combine_system_lagged(
                (Complex64::new(1.0, 0.0), self.base_values.raw()),
                &terms,
                (Complex64::new(1.0, 0.0), self.rhs.raw()),
                &[],
            )
            .map_err(solve_error)
    }

    fn solve(&mut self, treatments: &[f64]) -> Result<(), RoomTreatmentError> {
        // Clear eligibility before any fallible work; failed uploads or solves
        // cannot publish a warm start for a later objective evaluation.
        let warm = std::mem::replace(&mut self.warm, false);
        self.update(treatments)?;
        let status = self
            .solver
            .solve(if warm {
                DeviceInitialGuess::PreviousSolution
            } else {
                DeviceInitialGuess::Zero
            })
            .map_err(solve_error)?;
        if !status.converged {
            return Err(RoomTreatmentError::Solve(format!(
                "CUDA treatment solve did not converge: residual {}",
                status.true_residual
            )));
        }
        self.warm = true;
        Ok(())
    }
}

/// Retains configured CUDA forward and adjoint solves across treatment designs.
///
/// The preconditioners are explicitly lagged at the preparation design. Every
/// solve certifies the current original system; nonconvergence returns an error
/// permits explicit preconditioner refresh when the lagged action is unsuitable.
/// Base and treatment operators are uploaded once, then combined on the device.
/// Listener sparse maps, adjoint RHS assembly and treatment sensitivities run
/// on CUDA. Only listener-sized pressure/gradient vectors, treatment scalar
/// reductions and compact solver controls reach the host. MMA remains host work.
/// Both certified fields and the forward RHS remain resident.
/// Resident bases require device storage proportional to the treatment count.
pub struct CudaRoomTreatmentSession<'a> {
    problem: &'a RoomTreatmentProblem,
    state: CudaTreatmentState<'a>,
}

struct CudaTreatmentState<'a> {
    pool: &'a MemoryPool<CudaBackend>,
    forward: Phase<'a>,
    adjoint: Phase<'a>,
    listeners: PersistentDeviceCsr<CudaBackend, Complex64>,
    listener_adjoint: PersistentDeviceCsr<CudaBackend, Complex64>,
    listener_values: Buffer<Complex64, CudaBackend>,
    derivative: PersistentDeviceCsr<CudaBackend, Complex64>,
    derivative_action: Buffer<Complex64, CudaBackend>,
}

impl std::fmt::Debug for CudaRoomTreatmentSession<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaRoomTreatmentSession")
            .field("dofs", &self.problem.base_operator.num_rows)
            .field("solver", &self.problem.solver.solver_type)
            .finish_non_exhaustive()
    }
}

impl<'a> CudaTreatmentState<'a> {
    /// Prepare both configured actions at an explicit initial treatment design.
    ///
    /// `workload` must describe the mesh/space used to build the problem. Its
    /// action is overridden for the forward and Hermitian phases. The problem's
    /// solver configuration is honored; unsupported CUDA algorithms reject.
    ///
    /// # Errors
    /// Rejects invalid designs, unsupported workloads/configurations, invalid
    /// assembly, preconditioner failures and device allocation/upload failures.
    fn try_prepare(
        problem: &RoomTreatmentProblem,
        pool: &'a MemoryPool<CudaBackend>,
        workload: FemWorkload,
        treatments: &[f64],
    ) -> Result<Self, RoomTreatmentError> {
        validate(problem, treatments)?;
        if workload.physics != FemPhysics::Helmholtz {
            return Err(RoomTreatmentError::InvalidProblem(
                "room treatment requires a Helmholtz workload",
            ));
        }
        let matrix = problem.operator(treatments);
        let forward = Phase::prepare(
            pool,
            FemWorkload {
                action: FemAction::Forward,
                ..workload
            },
            &matrix,
            problem,
        )?;
        let adjoint = Phase::prepare(
            pool,
            FemWorkload {
                action: FemAction::Hermitian,
                ..workload
            },
            &matrix,
            problem,
        )?;
        let entries = problem
            .listeners
            .iter()
            .enumerate()
            .flat_map(|(row, listener)| {
                listener
                    .entries
                    .iter()
                    .map(move |&(column, weight)| (row, column, weight))
            })
            .collect();
        let listener_matrix =
            CsrMatrix::from_triplets(problem.listeners.len(), matrix.num_rows, entries);
        let listener_adjoint_matrix = conjugate_transpose(&listener_matrix);
        let upload = |matrix: &CsrMatrix<Complex64>| {
            PersistentDeviceCsr::try_from_host_csr_ref(pool.backend(), matrix.as_host_csr_ref())
                .map_err(solve_error)
        };
        let listeners = upload(&listener_matrix)?;
        let listener_adjoint = upload(&listener_adjoint_matrix)?;
        // Reuse one derivative descriptor and action buffer for every treatment;
        // numeric values come from the already resident forward basis buffers.
        let derivative = upload(&problem.base_operator)?;
        let listener_values = pool
            .try_alloc(problem.listeners.len())
            .map_err(solve_error)?;
        let derivative_action = pool.try_alloc(matrix.num_rows).map_err(solve_error)?;
        Ok(Self {
            pool,
            forward,
            adjoint,
            listeners,
            listener_adjoint,
            listener_values,
            derivative,
            derivative_action,
        })
    }

    fn prepare_preconditioners(
        &self,
        matrix: &CsrMatrix<Complex64>,
    ) -> Result<(ConfiguredFemPreconditioner, ConfiguredFemPreconditioner), RoomTreatmentError>
    {
        let forward = self
            .forward
            .solver
            .try_prepare_preconditioner(matrix)
            .map_err(solve_error)?;
        let adjoint = self
            .adjoint
            .solver
            .try_prepare_preconditioner(&conjugate_transpose(matrix))
            .map_err(solve_error)?;
        Ok((forward, adjoint))
    }

    fn install_preconditioners(
        &mut self,
        candidates: (ConfiguredFemPreconditioner, ConfiguredFemPreconditioner),
    ) {
        self.forward
            .solver
            .session_mut()
            .replace_preconditioner(candidates.0);
        self.adjoint
            .solver
            .session_mut()
            .replace_preconditioner(candidates.1);
    }

    fn evaluate(
        &mut self,
        treatments: &[f64],
        pressure_objective: impl FnOnce(&[Complex64]) -> ((f64, f64, f64), Vec<Complex64>),
    ) -> Result<FrequencyEvaluation, RoomTreatmentError> {
        self.forward.solve(treatments)?;
        let (target, variance, min_level) = self.prepare_adjoint_rhs_with(pressure_objective)?;
        self.adjoint.solve(treatments)?;
        Ok(FrequencyEvaluation {
            target,
            variance,
            min_level,
            gradient: self.treatment_gradient()?,
        })
    }

    #[cfg(test)]
    fn prepare_adjoint_rhs(&mut self) -> Result<f64, RoomTreatmentError> {
        self.prepare_adjoint_rhs_with(|pressures| {
            let (value, gradient) = seat_variance_pressures(pressures);
            ((0.0, value, 0.0), gradient)
        })
        .map(|(_, variance, _)| variance)
    }

    fn prepare_adjoint_rhs_with(
        &mut self,
        pressure_objective: impl FnOnce(&[Complex64]) -> ((f64, f64, f64), Vec<Complex64>),
    ) -> Result<(f64, f64, f64), RoomTreatmentError> {
        let backend = self.pool.backend();
        let state = self
            .forward
            .solver
            .session()
            .solution()
            .ok_or_else(|| solve_error("forward field is not certified"))?;
        self.listeners
            .try_apply(backend, state, self.listener_values.raw_mut())
            .map_err(solve_error)?;
        let mut pressures = vec![Complex64::default(); self.listener_values.len()];
        backend
            .try_copy_d2h(self.listener_values.raw(), &mut pressures)
            .map_err(solve_error)?;
        let (objective, mut pressure_gradient) = pressure_objective(&pressures);
        if !objective.0.is_finite()
            || !objective.1.is_finite()
            || !objective.2.is_finite()
            || pressure_gradient.len() != pressures.len()
            || pressure_gradient
                .iter()
                .any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err(solve_error(
                "non-finite CUDA listener objective or gradient",
            ));
        }
        for value in &mut pressure_gradient {
            *value = -*value;
        }
        backend
            .try_copy_h2d(&pressure_gradient, self.listener_values.raw_mut())
            .map_err(solve_error)?;
        self.listener_adjoint
            .try_apply(
                backend,
                self.listener_values.raw(),
                self.adjoint.rhs.raw_mut(),
            )
            .map_err(solve_error)?;
        Ok(objective)
    }

    fn treatment_gradient(&mut self) -> Result<Vec<f64>, RoomTreatmentError> {
        let backend = self.pool.backend();
        let state = self
            .forward
            .solver
            .session()
            .solution()
            .ok_or_else(|| solve_error("forward field is not certified"))?;
        let adjoint = self
            .adjoint
            .solver
            .session()
            .solution()
            .ok_or_else(|| solve_error("adjoint field is not certified"))?;
        self.forward
            .treatment_values
            .iter()
            .map(|values| {
                self.derivative
                    .try_copy_values_from_device(backend, values.raw())
                    .map_err(solve_error)?;
                self.derivative
                    .try_apply(backend, state, self.derivative_action.raw_mut())
                    .map_err(solve_error)?;
                let product = backend
                    .try_dot_with_mode(ReductionMode::Fast, adjoint, self.derivative_action.raw())
                    .map_err(solve_error)?;
                let gradient = 2.0 * product.re;
                if !gradient.is_finite() {
                    return Err(solve_error("non-finite CUDA treatment gradient"));
                }
                Ok(gradient)
            })
            .collect()
    }
}

impl<'a> CudaRoomTreatmentSession<'a> {
    /// Prepare retained CUDA forward/adjoint solves for a single-frequency problem.
    ///
    /// The workload describes the source mesh and space; the solver configuration
    /// and preparation design determine the explicitly lagged preconditioners.
    ///
    /// # Errors
    /// Rejects unsupported workloads/configurations or invalid designs and
    /// propagates native setup, assembly and allocation failures.
    pub fn try_prepare(
        problem: &'a RoomTreatmentProblem,
        pool: &'a MemoryPool<CudaBackend>,
        workload: FemWorkload,
        treatments: &[f64],
    ) -> Result<Self, RoomTreatmentError> {
        Ok(Self {
            problem,
            state: CudaTreatmentState::try_prepare(problem, pool, workload, treatments)?,
        })
    }

    /// Refresh both preconditioners at a chosen design without rebuilding resident solve state.
    ///
    /// Both replacements are prepared before either is installed. Setup assembles
    /// the design operator and factors preconditioners on the host; matrix bases,
    /// physical operators, Krylov buffers and warm-start fields are retained.
    /// Successful refresh unpublishes fields until the next certified solve.
    ///
    /// # Errors
    /// Rejects invalid designs and propagates setup/upload errors. A failed
    /// preparation leaves both installed actions unchanged; recover the backend
    /// when required before evaluating again.
    pub fn try_refresh_preconditioners(
        &mut self,
        treatments: &[f64],
    ) -> Result<(), RoomTreatmentError> {
        validate(self.problem, treatments)?;
        let matrix = self.problem.operator(treatments);
        let candidates = self.state.prepare_preconditioners(&matrix)?;
        self.state.install_preconditioners(candidates);
        Ok(())
    }

    /// Evaluate seat variance and its gradient with resident fields.
    ///
    /// # Errors
    /// Rejects invalid designs and propagates native update/solve/reduction failures.
    pub fn objective_gradient(
        &mut self,
        treatments: &[f64],
    ) -> Result<(f64, Vec<f64>), RoomTreatmentError> {
        validate(self.problem, treatments)?;
        let result = self.state.evaluate(treatments, |pressures| {
            let (value, gradient) = seat_variance_pressures(pressures);
            ((0.0, value, 0.0), gradient)
        })?;
        Ok((result.variance, result.gradient))
    }

    /// Run the existing MMA treatment loop with persistent CUDA forward/adjoint solves.
    ///
    /// Density filtering, spatial constraints, stopping criteria and result layout
    /// are shared with the CPU implementation.
    ///
    /// # Errors
    /// Propagates optimizer configuration, objective and native solve failures.
    pub fn optimize(
        &mut self,
        config: &RoomTreatmentConfig,
    ) -> Result<RoomTreatmentResult, RoomTreatmentError> {
        let problem = self.problem;
        problem.optimize_with_evaluator(config, |treatments| self.objective_gradient(treatments))
    }
}

fn validate(problem: &RoomTreatmentProblem, treatments: &[f64]) -> Result<(), RoomTreatmentError> {
    if treatments.len() != problem.treatment_operators.len()
        || treatments
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
    {
        return Err(RoomTreatmentError::InvalidProblem(
            "treatment vector has the wrong size or lies outside [0, 1]",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;

mod multifrequency;
#[doc(inline)]
pub use multifrequency::CudaMultiFrequencyRoomTreatmentSession;
