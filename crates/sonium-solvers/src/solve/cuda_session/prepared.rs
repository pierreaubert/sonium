//! Explicit flexible Krylov preparation and preconditioner update policies.

// Rust guideline compliant 2026-02-21
use super::*;
use sonium_compute::BackendError;

impl<'a, T> CudaLinearSolveSession<'a, T>
where
    T: ComplexField + DeviceScalar<Real = <T as ComplexField>::Real>,
    <T as ComplexField>::Real: FromPrimitive + Sync + Send,
{
    /// Prepare retained FGMRES with a caller-selected native device preconditioner.
    ///
    /// `prepare` receives the session backend and canonical physical matrix. It
    /// must return a checked device action on that backend, with matching DOF
    /// ordering and dimensions. Shifted auxiliary AMG or a Schur action can be
    /// supplied without asserting that the physical matrix is SPD. Selecting a
    /// physically suitable preconditioner remains the caller's responsibility.
    /// Native controls cannot inspect arbitrary caller actions for host staging;
    /// callers must validate their action's residency separately.
    ///
    /// FGMRES retains each preconditioned basis vector, so changing/nonlinear
    /// actions are supported. Controls must require native execution. Result
    /// certification always uses the physical matrix, never the auxiliary one.
    ///
    /// # Errors
    /// Rejects invalid controls/matrices and non-native Krylov primitives before
    /// invoking `prepare`. Propagates
    /// preconditioner preparation, upload and allocation failures.
    pub fn try_new_flexible<P>(
        pool: &'a MemoryPool<CudaBackend>,
        matrix: &CsrMatrix<T>,
        controls: &BackendKrylovConfig<<T as DeviceScalar>::Real>,
        prepare: impl FnOnce(&CudaBackend, &CsrMatrix<T>) -> Result<P, BackendError>,
    ) -> Result<Self, LinearSolveError>
    where
        P: DevicePreconditioner<CudaBackend, T> + 'static,
    {
        if !controls.require_native {
            return Err(LinearSolveError::InvalidConfig(
                "public CUDA FGMRES requires native execution".into(),
            ));
        }
        // Gmres supplies only common host certification/control metadata here.
        // A Prepared preconditioner selects the flexible device iteration.
        let config = LinearSolveConfig {
            solver: LinearSolverType::Gmres,
            max_iterations: controls.max_iterations,
            restart: controls.restart,
            tolerance: controls.tolerance.to_f64().ok_or_else(|| {
                LinearSolveError::InvalidConfig("invalid flexible tolerance".into())
            })?,
            exact_residual_check_interval: controls.exact_residual_check_interval,
            verbosity: 0,
        };
        config.validate()?;
        validate_csr_structure(matrix)?;
        validate_linear_inputs(
            matrix,
            &vec![<T as DeviceScalar>::zero(); matrix.num_rows],
            None,
            &config,
        )?;
        crate::iterative::validate_backend_krylov::<_, T>(
            pool.backend(),
            crate::iterative::BackendKrylovMethod::Fgmres,
            true,
        )
        .map_err(backend_error)?;
        let canonical = coalesced_csr(matrix);
        let preconditioner = SessionPreconditioner::Prepared(Box::new(
            prepare(pool.backend(), &canonical).map_err(backend_error)?,
        ));
        Self::from_prepared(
            pool,
            matrix,
            &canonical,
            &config,
            LinearSystemAssumption::General,
            preconditioner,
        )
    }

    /// Update a prepared FGMRES matrix while explicitly retaining its preconditioner.
    ///
    /// Canonical sparsity must stay fixed. The existing preconditioner is an
    /// intentionally lagged action; convergence is certified against the new
    /// physical matrix. Successful updates preserve the previous warm start but
    /// hide its output. A failed upload blocks all solves until a complete update.
    ///
    /// # Errors
    /// Rejects non-prepared sessions or invalid/changed matrix structure.
    /// Propagates backend failures; terminal errors require a new pool/session.
    pub fn try_update_matrix_lagged(
        &mut self,
        matrix: &CsrMatrix<T>,
    ) -> Result<(), LinearSolveError> {
        let previous = self.begin_prepared_update()?;
        let canonical = self.validate_update(matrix)?;
        self.commit_prepared_update(matrix, &canonical, None, previous)
    }

    /// Update a prepared FGMRES matrix and explicitly replace its preconditioner.
    ///
    /// `prepare` receives the canonical new physical matrix. Preparation errors
    /// leave the old numeric system usable with a zero start. After uploads begin,
    /// failures block solving until a full matrix update succeeds. Do not restore
    /// usability with a lagged update unless retaining the current action is intended.
    /// Successful updates preserve the previous solution only as a warm start.
    ///
    /// # Errors
    /// Rejects non-prepared sessions, invalid matrices or changed canonical
    /// sparsity before invoking `prepare`. Propagates setup and backend failures.
    pub fn try_update_matrix_with_preconditioner<P>(
        &mut self,
        matrix: &CsrMatrix<T>,
        prepare: impl FnOnce(&CudaBackend, &CsrMatrix<T>) -> Result<P, BackendError>,
    ) -> Result<(), LinearSolveError>
    where
        P: DevicePreconditioner<CudaBackend, T> + 'static,
    {
        let previous = self.begin_prepared_update()?;
        let canonical = self.validate_update(matrix)?;
        let replacement = SessionPreconditioner::Prepared(Box::new(
            prepare(self.pool.backend(), &canonical).map_err(backend_error)?,
        ));
        self.commit_prepared_update(matrix, &canonical, Some(replacement), previous)
    }

    fn begin_prepared_update(&mut self) -> Result<bool, LinearSolveError> {
        self.published = false;
        let previous = std::mem::replace(&mut self.warm_start_available, false);
        if !self.preconditioner.is_prepared() {
            return Err(LinearSolveError::InvalidConfig(
                "explicit prepared updates require a prepared FGMRES session".into(),
            ));
        }
        Ok(previous)
    }

    fn commit_prepared_update(
        &mut self,
        matrix: &CsrMatrix<T>,
        canonical: &CsrMatrix<T>,
        replacement: Option<SessionPreconditioner<T>>,
        previous: bool,
    ) -> Result<(), LinearSolveError> {
        self.numeric_ready = false;
        self.matrix
            .try_copy_values_from_host(self.pool.backend(), &canonical.values)
            .map_err(backend_error)?;
        self.pool
            .backend()
            .try_synchronize()
            .map_err(backend_error)?;
        if let Some(replacement) = replacement {
            self.preconditioner = replacement;
        }
        self.host_matrix = matrix.clone();
        self.numeric_ready = true;
        self.warm_start_available = previous;
        Ok(())
    }
}
