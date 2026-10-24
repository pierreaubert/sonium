//! Retained CUDA solves for admitted, host-assembled FEM systems.
//!
//! Admission runs before the assembly closure. Geometry, forms, constraints and
//! preconditioner setup remain explicit host work. Repeated solves retain CSR,
//! RHS, preconditioner and Krylov storage; numeric device updates reuse them.
//!
//! Taylor–Hood Stokes systems can use FGMRES with
//! [`sonium_compute::preconditioner::SaddlePointJacobiSchurBackend::try_from_host_csr`]
//! in the preparation closure. Supply the constrained matrix and a pressure
//! Schur approximation in the same block ordering. For the symmetric system
//! `[A B; B^T 0]`, the Schur complement is negative; preserve pressure-gauge
//! identity rows separately. The returned preconditioner owns its uploaded
//! matrices, so the closure need not retain the temporary host CSR. This does
//! not move nonlinear assembly or time-history state onto the device.

// Rust guideline compliant 2026-02-21
use crate::backend::{
    FemAction, FemAdmission, FemAdmissionError, FemAssemblyMode, FemPhysics, FemScalar,
    FemWorkload, admit_fem_cuda,
};
use crate::boundary::generic::{BoundaryConditionError, FemSystem, FemSystemError};
use num_complex::Complex64;
use sonium_compute::{
    Backend, BackendError, Buffer, CudaBackend, CudaBuffer, DeviceBufferLike, DevicePreconditioner,
    DeviceScalar, MemoryPool, PcgBackendConfig, PcgBackendError, PersistentDeviceCsr,
};
use sonium_solvers::iterative::{
    BackendKrylovConfig, BackendKrylovError, DeviceGmresSession, DeviceInitialGuess,
    DevicePcgAssumption, DevicePcgSession,
};
use sonium_solvers::{ComplexField, CsrMatrix};
use std::any::TypeId;

mod configured;
#[doc(inline)]
pub use configured::{ConfiguredFemPreconditioner, CudaConfiguredFemSolveSession};

/// Solver convention for a retained assembled FEM system.
#[derive(Debug, Clone, Copy)]
pub enum FemDeviceMethod {
    /// Standard GMRES; the preconditioner must remain linear and fixed during each solve.
    Gmres,
    /// Flexible GMRES, including variable or auxiliary-system preconditioners.
    Fgmres,
    /// PCG; the caller certifies both operator and preconditioner are SPD/HPD.
    Pcg(DevicePcgAssumption),
}

/// Checked failure from assembly, admission, device updates or iteration.
#[derive(Debug, thiserror::Error)]
pub enum FemDeviceSolveError {
    #[error(transparent)]
    Admission(#[from] FemAdmissionError),
    #[error(transparent)]
    System(#[from] FemSystemError),
    #[error(transparent)]
    Space(#[from] crate::spaces::SpaceValidationError),
    #[error(transparent)]
    Boundary(#[from] BoundaryConditionError),
    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error(transparent)]
    Krylov(#[from] Box<BackendKrylovError>),
    #[error(transparent)]
    Pcg(#[from] PcgBackendError),
}
impl From<BackendKrylovError> for FemDeviceSolveError {
    fn from(error: BackendKrylovError) -> Self {
        Self::Krylov(Box::new(error))
    }
}

/// Original-system certification of a retained FEM solve.
#[derive(Debug, Clone, Copy)]
pub struct FemDeviceSolveStatus {
    /// Completed Krylov iterations.
    pub iterations: usize,
    /// Recomputed relative residual, or absolute residual for a zero RHS.
    pub true_residual: f64,
    /// Whether the original-system residual meets the requested tolerance.
    pub converged: bool,
}

enum Iteration<'a, T: DeviceScalar> {
    Gmres(Box<DeviceGmresSession<'a, CudaBackend, T>>),
    Pcg(Box<DevicePcgSession<'a, CudaBackend, T>>),
}

/// Retains an admitted assembled operator, preconditioner, RHS and solver workspace.
///
/// The caller supplies a suitable device preconditioner: shifted auxiliary AMG
/// for indefinite Helmholtz, stationary symmetric AMG for PCG, or field-split/
/// Schur methods for saddle points. This API does not infer those properties.
/// Host assembly and preconditioner setup are not GPU assembly coverage.
/// Matrix/preconditioner backend allocations retain their existing quota policy;
/// only the RHS and Krylov buffers here are explicitly owned by the supplied pool.
pub struct CudaFemSolveSession<'a, T: DeviceScalar, P> {
    pool: &'a MemoryPool<CudaBackend>,
    admission: FemAdmission,
    row_offsets: Vec<usize>,
    columns: Vec<usize>,
    matrix: PersistentDeviceCsr<CudaBackend, T>,
    rhs: Buffer<T, CudaBackend>,
    preconditioner: P,
    iteration: Iteration<'a, T>,
    matrix_valid: bool,
    rhs_valid: bool,
    published: bool,
    guess_available: bool,
}

impl<T: DeviceScalar, P> std::fmt::Debug for CudaFemSolveSession<'_, T, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaFemSolveSession")
            .field("admission", &self.admission)
            .field("dimension", &self.rhs.len())
            .field("has_solution", &self.published)
            .finish_non_exhaustive()
    }
}

fn invalid(reason: &str) -> BackendError {
    BackendError::InvalidArgument {
        operation: "fem_device_session",
        reason: reason.into(),
    }
}

impl<'a, T, P> CudaFemSolveSession<'a, T, P>
where
    T: DeviceScalar<Real = f64> + ComplexField<Real = f64> + Default,
    P: DevicePreconditioner<CudaBackend, T>,
{
    /// Admit, assemble and upload a constrained system once.
    ///
    /// `assemble` supplies the complete matrix and RHS after boundary/constraint
    /// elimination. For transpose/Hermitian requests its RHS is the RHS of that
    /// requested system; the matrix is transformed once during host setup.
    /// `prepare` receives the canonical matrix of the requested action, and must
    /// return a preconditioner on the supplied pool's backend and stream.
    ///
    /// # Errors
    /// Rejects unsupported workloads/scalars/modes before invoking either closure,
    /// including missing native solver primitives at runtime.
    /// PCG for Stokes, malformed/non-finite systems, and backend/setup failures.
    /// Geometry and constraint correctness remain the assembler's contract.
    pub fn try_assemble(
        pool: &'a MemoryPool<CudaBackend>,
        workload: FemWorkload,
        method: FemDeviceMethod,
        assemble: impl FnOnce() -> Result<FemSystem<T>, FemDeviceSolveError>,
        prepare: impl FnOnce(&CsrMatrix<T>) -> Result<P, BackendError>,
    ) -> Result<Self, FemDeviceSolveError> {
        let admission = Self::admit(pool, workload, method)?;
        let system = assemble()?;
        system.validate()?;
        let n = system.num_dofs();
        if n == 0
            || system.matrix.num_cols != n
            || system.dirichlet_dofs.iter().any(|&dof| dof >= n)
            || system
                .matrix
                .values
                .iter()
                .chain(&system.rhs)
                .any(|v| !DeviceScalar::is_finite(v))
        {
            return Err(
                invalid("non-square, empty, non-finite or invalid constrained system").into(),
            );
        }
        let entries = system
            .matrix
            .rows
            .iter()
            .zip(&system.matrix.cols)
            .zip(&system.matrix.values)
            .map(|((&row, &column), &value)| match workload.action {
                FemAction::Forward => (row, column, value),
                FemAction::Transpose => (column, row, value),
                FemAction::Hermitian => (column, row, ComplexField::conj(&value)),
            })
            .collect();
        let canonical = CsrMatrix::from_triplets(n, n, entries);
        Self::upload(pool, admission, method, &canonical, &system.rhs, prepare)
    }

    /// Prepare a retained solve directly from a constrained canonical CSR system.
    ///
    /// Supply the forward matrix after all boundary and constraint elimination.
    /// Rows must contain strictly increasing column indices without duplicates.
    /// Forward preparation borrows values without a COO conversion; transpose
    /// and Hermitian preparation perform one linear-time host transpose. `rhs`
    /// belongs to the requested action and is never transformed. `prepare`
    /// receives that action's matrix, on the supplied pool's backend and stream.
    /// Geometry, constraint validity and suitable preconditioning remain caller
    /// responsibilities. Pattern changes require a new session.
    ///
    /// # Errors
    /// Rejects unsupported workloads/methods, malformed or noncanonical CSR,
    /// empty/nonsquare matrices, invalid RHS lengths and non-finite values before
    /// uploading or invoking `prepare`. Propagates native setup failures.
    pub fn try_from_csr(
        pool: &'a MemoryPool<CudaBackend>,
        workload: FemWorkload,
        method: FemDeviceMethod,
        matrix: &CsrMatrix<T>,
        rhs: &[T],
        prepare: impl FnOnce(&CsrMatrix<T>) -> Result<P, BackendError>,
    ) -> Result<Self, FemDeviceSolveError> {
        let admission = Self::admit(pool, workload, method)?;
        let view = sonium_compute::HostCsrRef::try_new(
            matrix.num_rows,
            matrix.num_cols,
            &matrix.row_ptrs,
            &matrix.col_indices,
            &matrix.values,
        )
        .map_err(|error| invalid(&error.to_string()))?;
        if matrix.num_rows == 0
            || matrix.num_rows != matrix.num_cols
            || rhs.len() != matrix.num_rows
            || matrix
                .values
                .iter()
                .chain(rhs)
                .any(|v| !DeviceScalar::is_finite(v))
            || matrix.row_ptrs.windows(2).any(|range| {
                matrix.col_indices[range[0]..range[1]]
                    .windows(2)
                    .any(|pair| pair[0] >= pair[1])
            })
        {
            return Err(invalid(
                "requires nonempty square canonical CSR and finite matrix/RHS values",
            )
            .into());
        }
        if workload.action == FemAction::Forward {
            return Self::upload(pool, admission, method, matrix, rhs, prepare);
        }
        let transposed = view
            .try_transpose()
            .map_err(|error| invalid(&error.to_string()))?;
        let mut canonical = CsrMatrix {
            num_rows: transposed.num_rows,
            num_cols: transposed.num_cols,
            row_ptrs: transposed.row_ptrs,
            col_indices: transposed.col_indices,
            values: transposed.values,
        };
        if workload.action == FemAction::Hermitian {
            for value in &mut canonical.values {
                *value = ComplexField::conj(value);
            }
        }
        Self::upload(pool, admission, method, &canonical, rhs, prepare)
    }

    fn admit(
        pool: &MemoryPool<CudaBackend>,
        workload: FemWorkload,
        method: FemDeviceMethod,
    ) -> Result<FemAdmission, FemDeviceSolveError> {
        let admission = admit_fem_cuda(workload)?;
        let scalar = if TypeId::of::<T>() == TypeId::of::<f64>() {
            FemScalar::F64
        } else if TypeId::of::<T>() == TypeId::of::<Complex64>() {
            FemScalar::Complex64
        } else {
            return Err(invalid("unsupported FEM scalar").into());
        };
        if workload.scalar != scalar || workload.mode != FemAssemblyMode::Assembled {
            return Err(invalid("scalar mismatch or non-assembled workload").into());
        }
        if workload.physics == FemPhysics::Stokes && matches!(method, FemDeviceMethod::Pcg(_)) {
            return Err(
                invalid("Stokes saddle-point systems require a nonsymmetric solver").into(),
            );
        }
        match method {
            FemDeviceMethod::Gmres => sonium_solvers::iterative::validate_backend_krylov::<_, T>(
                pool.backend(),
                sonium_solvers::iterative::BackendKrylovMethod::Gmres,
                true,
            )?,
            FemDeviceMethod::Fgmres => sonium_solvers::iterative::validate_backend_krylov::<_, T>(
                pool.backend(),
                sonium_solvers::iterative::BackendKrylovMethod::Fgmres,
                true,
            )?,
            FemDeviceMethod::Pcg(_) => {
                sonium_solvers::iterative::validate_backend_pcg::<_, T>(pool.backend(), true)?
            }
        }
        Ok(admission)
    }

    fn upload(
        pool: &'a MemoryPool<CudaBackend>,
        admission: FemAdmission,
        method: FemDeviceMethod,
        canonical: &CsrMatrix<T>,
        host_rhs: &[T],
        prepare: impl FnOnce(&CsrMatrix<T>) -> Result<P, BackendError>,
    ) -> Result<Self, FemDeviceSolveError> {
        let matrix = PersistentDeviceCsr::try_from_host_csr_ref(
            pool.backend(),
            canonical.as_host_csr_ref(),
        )?;
        let preconditioner = prepare(canonical)?;
        let n = canonical.num_rows;
        let mut rhs = pool.try_alloc(n).map_err(BackendError::from)?;
        pool.backend().try_copy_h2d(host_rhs, rhs.raw_mut())?;
        let iteration = match method {
            FemDeviceMethod::Gmres => Iteration::Gmres(Box::new(
                DeviceGmresSession::try_new(pool, n).map_err(BackendError::from)?,
            )),
            FemDeviceMethod::Fgmres => Iteration::Gmres(Box::new(
                DeviceGmresSession::try_new_flexible(pool, n).map_err(BackendError::from)?,
            )),
            FemDeviceMethod::Pcg(assumption) => Iteration::Pcg(Box::new(
                DevicePcgSession::try_new(pool, n, assumption).map_err(BackendError::from)?,
            )),
        };
        Ok(Self {
            pool,
            admission,
            row_offsets: canonical.row_ptrs.clone(),
            columns: canonical.col_indices.clone(),
            matrix,
            rhs,
            preconditioner,
            iteration,
            matrix_valid: true,
            rhs_valid: true,
            published: false,
            guess_available: false,
        })
    }

    /// Return the admission record; host setup remains explicit in this API.
    pub fn admission(&self) -> &FemAdmission {
        &self.admission
    }

    /// Return the fixed CSR pattern in the requested action's ordering.
    ///
    /// Numeric replacements must use this row-offset/column ordering, including
    /// the transpose or conjugation chosen during construction. Pattern, dimension
    /// or constraint-layout changes require a new session.
    pub fn pattern(&self) -> (&[usize], &[usize]) {
        (&self.row_offsets, &self.columns)
    }

    /// Borrow the last certified solution; updates and failed solves unpublish it.
    pub fn solution(&self) -> Option<&CudaBuffer<T>> {
        if !self.published {
            return None;
        }
        match &self.iteration {
            Iteration::Gmres(s) => s.solution(),
            Iteration::Pcg(s) => s.solution(),
        }
    }

    /// Replace resident physical values while explicitly retaining the lagged preconditioner.
    ///
    /// A successful update preserves the last certified vector as a warm-start
    /// candidate, but unpublishes it until another solve certifies the new system.
    /// PCG callers must preserve their SPD/HPD assumption for each replacement.
    ///
    /// # Errors
    /// Rejects length/non-finite values and native failures. A failed copy marks
    /// the operator unusable until a full successful replacement; it also removes
    /// warm-start eligibility. Validation downloads only a compact status word.
    pub fn try_update_values_lagged(
        &mut self,
        values: &CudaBuffer<T>,
    ) -> Result<(), FemDeviceSolveError> {
        self.published = false;
        self.check_values(values, self.columns.len())?;
        self.matrix_valid = false;
        if let Err(error) = self
            .matrix
            .try_copy_values_from_device(self.pool.backend(), values)
        {
            self.guess_available = false;
            return Err(error.into());
        }
        self.matrix_valid = true;
        Ok(())
    }

    /// Combine resident matrix and RHS bases while retaining the lagged preconditioner.
    ///
    /// Each output is `base.0 * base.1 + sum(alpha * term)`. Matrix bases must
    /// follow [`Self::pattern`], including the selected adjoint action and fixed
    /// constraint layout. Supply constraint identity rows only once; material
    /// increments must have zero constrained rows. RHS bases must include the
    /// matching Dirichlet lifting. Callers preserve PCG's SPD/HPD assumptions.
    ///
    /// Success retains warm-start eligibility but unpublishes the old solution.
    /// Any failure invalidates both outputs and the warm start until complete
    /// replacement. No host assembly or bulk transfer occurs; validation scratch
    /// may warm on first use and is retained thereafter.
    ///
    /// # Errors
    /// Rejects invalid lengths, nonfinite coefficients or combined values, and
    /// unavailable native validation. Propagates backend failures; terminal
    /// synchronization failures require recreating the backend and session.
    pub fn try_combine_system_lagged(
        &mut self,
        matrix_base: (T, &CudaBuffer<T>),
        matrix_terms: &[(T, &CudaBuffer<T>)],
        rhs_base: (T, &CudaBuffer<T>),
        rhs_terms: &[(T, &CudaBuffer<T>)],
    ) -> Result<(), FemDeviceSolveError> {
        self.published = false;
        self.matrix_valid = false;
        self.rhs_valid = false;
        let warm = std::mem::replace(&mut self.guess_available, false);
        for (base, terms, length) in [
            (matrix_base, matrix_terms, self.columns.len()),
            (rhs_base, rhs_terms, self.rhs.len()),
        ] {
            if !DeviceScalar::is_finite(&base.0)
                || base.1.len() != length
                || terms.iter().any(|(alpha, values)| {
                    !DeviceScalar::is_finite(alpha) || values.len() != length
                })
            {
                return Err(invalid("invalid resident system combination").into());
            }
        }
        let backend = self.pool.backend();
        self.matrix.try_combine_scaled_values(
            backend,
            matrix_base.0,
            matrix_base.1,
            matrix_terms,
        )?;
        let combined = if let [(beta, y)] = rhs_terms {
            backend.try_linear_combination2_native(
                rhs_base.0,
                rhs_base.1,
                *beta,
                y,
                self.rhs.raw_mut(),
            )?
        } else {
            false
        };
        if !combined {
            backend.try_copy_d2d(rhs_base.1, self.rhs.raw_mut())?;
            backend.try_scale(rhs_base.0, self.rhs.raw_mut())?;
            for &(alpha, values) in rhs_terms {
                backend.try_axpy(alpha, values, self.rhs.raw_mut())?;
            }
        }
        self.check_values(&self.matrix.matrix().values, self.columns.len())?;
        self.check_values(self.rhs.raw(), self.rhs.len())?;
        backend.try_synchronize()?;
        self.matrix_valid = true;
        self.rhs_valid = true;
        self.guess_available = warm;
        Ok(())
    }

    /// Replace the resident right-hand side without reallocating solver buffers.
    ///
    /// # Errors
    /// Rejects non-finite/wrong-sized data or native failures. A failed copy blocks
    /// solving until a complete RHS replacement succeeds.
    pub fn try_update_rhs(&mut self, values: &CudaBuffer<T>) -> Result<(), FemDeviceSolveError> {
        self.published = false;
        self.check_values(values, self.rhs.len())?;
        self.rhs_valid = false;
        if let Err(error) = self.pool.backend().try_copy_d2d(values, self.rhs.raw_mut()) {
            self.guess_available = false;
            return Err(error.into());
        }
        self.rhs_valid = true;
        Ok(())
    }

    fn check_values(&self, values: &CudaBuffer<T>, n: usize) -> Result<(), FemDeviceSolveError> {
        if values.len() != n {
            return Err(invalid("incorrect numeric update length").into());
        }
        if self.pool.backend().try_all_finite_native(values)? != Some(true) {
            return Err(invalid("non-finite values or unavailable native validation").into());
        }
        Ok(())
    }

    /// Replace a prepared preconditioner while retaining matrix and Krylov storage.
    ///
    /// The new preconditioner must use this session's dimension/backend/stream,
    /// and satisfy the selected method's assumptions. Preparation is caller-owned;
    /// a failed preparation need not replace the existing preconditioner.
    pub fn replace_preconditioner(&mut self, preconditioner: P) {
        self.published = false;
        self.preconditioner = preconditioner;
    }

    /// Solve on CUDA and certify the original requested system before publication.
    ///
    /// `require_native` must be true. PCG uses the common tolerance and iteration
    /// limit; restart and exact-residual interval apply to GMRES and FGMRES. All methods
    /// recompute the final original-system residual. Compact control reductions
    /// still reach the host; no full solution is downloaded here.
    ///
    /// # Errors
    /// Rejects invalid controls, incomplete updates and unavailable warm starts;
    /// propagates typed iteration/backend failures. All unsuccessful solves
    /// invalidate publication and warm-start eligibility.
    pub fn solve(
        &mut self,
        initial: DeviceInitialGuess,
        config: &BackendKrylovConfig<f64>,
    ) -> Result<FemDeviceSolveStatus, FemDeviceSolveError> {
        self.published = false;
        let previous = std::mem::replace(&mut self.guess_available, false);
        if !self.matrix_valid
            || !self.rhs_valid
            || (initial == DeviceInitialGuess::PreviousSolution && !previous)
        {
            return Err(invalid("incomplete update or unavailable previous solution").into());
        }
        if !config.require_native
            || config.max_iterations == 0
            || (matches!(&self.iteration, Iteration::Gmres(_)) && config.restart == 0)
            || !config.tolerance.is_finite()
            || config.tolerance <= 0.0
        {
            return Err(invalid("invalid native solver controls").into());
        }
        let status = match &mut self.iteration {
            Iteration::Gmres(s) => {
                let capacity = config.restart.min(config.max_iterations);
                s.try_reserve_cycle(capacity)?;
                self.pool
                    .backend()
                    .try_reserve_krylov_reductions::<T>(capacity)?;
                let result = s.solve(
                    &self.matrix,
                    &self.preconditioner,
                    self.rhs.raw(),
                    initial,
                    config,
                    None,
                )?;
                FemDeviceSolveStatus {
                    iterations: result.iterations,
                    true_residual: result.true_residual,
                    converged: result.converged,
                }
            }
            Iteration::Pcg(s) => {
                let result = s.solve(
                    &self.matrix,
                    &self.preconditioner,
                    self.rhs.raw(),
                    initial,
                    &PcgBackendConfig {
                        max_iterations: config.max_iterations,
                        tolerance: config.tolerance,
                    },
                )?;
                FemDeviceSolveStatus {
                    iterations: result.iterations,
                    true_residual: result.true_residual,
                    converged: result.converged,
                }
            }
        };
        self.published = status.converged;
        self.guess_available = status.converged;
        Ok(status)
    }
}

#[cfg(test)]
mod tests;
