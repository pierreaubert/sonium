//! Persistent device buffers and workspace for repeated preconditioned GMRES solves.
//!
//! Operators, preconditioners and right-hand sides stay caller-owned on the device.
//! The session performs no host vector upload or download. Small solver control
//! data still follows the underlying Krylov implementation's transfer policy.

// Rust guideline compliant 2026-02-21
use super::*;

/// Initial iterate for a persistent device solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceInitialGuess {
    /// Start from zero, regardless of the preceding solve.
    Zero,
    /// Reuse the last successfully certified solution on the device.
    PreviousSolution,
}

/// Retains solution buffers and Krylov workspace across device GMRES solves.
///
/// Keep the matrix and preconditioner alive alongside this session. Their numeric
/// values may change between solves without recreating the session if dimensions
/// stay fixed. Only converged solutions are published; a failed or unconverged
/// solve invalidates access to the previous solution to prevent stale output use.
/// The pool fixes the backend and memory quota for the session's lifetime.
pub struct DeviceGmresSession<'a, B: Backend, T: DeviceScalar> {
    pool: &'a MemoryPool<B>,
    workspace: BackendKrylovWorkspace<B, T>,
    solution: Buffer<T, B>,
    candidate: Buffer<T, B>,
    valid: bool,
    method: BackendKrylovMethod,
}

impl<B: Backend, T: DeviceScalar> std::fmt::Debug for DeviceGmresSession<'_, B, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceGmresSession")
            .field("dimension", &self.workspace.dimension())
            .field("has_solution", &self.valid)
            .field("method", &self.method)
            .finish_non_exhaustive()
    }
}

impl<'a, B: Backend, T: DeviceScalar> DeviceGmresSession<'a, B, T> {
    /// Allocate persistent solution buffers from the supplied pool.
    ///
    /// Krylov scratch is allocated lazily by the first solve.
    ///
    /// # Errors
    /// Returns a quota or backend allocation error. Partial allocations are freed.
    pub fn try_new(
        pool: &'a MemoryPool<B>,
        dimension: usize,
    ) -> Result<Self, sonium_compute::OomError> {
        Ok(Self {
            pool,
            workspace: BackendKrylovWorkspace::new(dimension),
            solution: pool.try_alloc(dimension)?,
            candidate: pool.try_alloc(dimension)?,
            valid: false,
            method: BackendKrylovMethod::Gmres,
        })
    }

    /// Allocate a persistent flexible GMRES session for changing preconditioners.
    ///
    /// Each Arnoldi preconditioner result is retained until the cycle ends.
    /// This consumes one additional device vector per iteration in the cycle.
    ///
    /// # Errors
    /// Returns a quota or backend allocation error. Partial allocations are freed.
    pub fn try_new_flexible(
        pool: &'a MemoryPool<B>,
        dimension: usize,
    ) -> Result<Self, sonium_compute::OomError> {
        let mut session = Self::try_new(pool, dimension)?;
        session.method = BackendKrylovMethod::Fgmres;
        Ok(session)
    }

    /// Reserve device vectors for a complete cycle and subsequent restart.
    ///
    /// This covers harder subsequent RHS cases even when the first solve exits
    /// early. Small reduction buffers still warm on the first solve and may
    /// resize when cycle lengths change. The published solution is unchanged.
    ///
    /// # Errors
    /// Rejects zero capacity or overflow and propagates allocation failures.
    pub fn try_reserve_cycle(&mut self, iterations: usize) -> Result<(), BackendKrylovError> {
        self.workspace
            .scalar
            .try_reserve_cycle(
                self.pool,
                iterations,
                self.method == BackendKrylovMethod::Fgmres,
            )
            .map_err(|error| map_gmres_error(self.pool.backend(), self.method, error, 0))?;
        if self.workspace.scalar_restart.is_none() {
            self.workspace.scalar_restart = Some(
                self.pool.try_alloc(self.workspace.dimension()).map_err(|error| {
                    map_gmres_error(
                        self.pool.backend(),
                        self.method,
                        GmresBackendError::Allocation {
                            label: "reserved restart iterate",
                            reason: error.to_string(),
                        },
                        0,
                    )
                })?,
            );
        }
        Ok(())
    }

    /// Borrow the certified device solution from the most recent solve.
    #[must_use]
    pub fn solution(&self) -> Option<&B::Buffer<T>> {
        self.valid.then(|| self.solution.raw())
    }

    /// Solve using resident data and the session's selected preconditioning convention.
    ///
    /// A converged candidate replaces the published solution without a vector
    /// copy. Unconverged candidates are never published. Sessions created with
    /// `try_new` require a stationary linear preconditioner; `try_new_flexible`
    /// permits changing or nonlinear preconditioner actions.
    ///
    /// # Errors
    /// Returns checked Krylov errors for invalid inputs, unsupported capabilities,
    /// cancellation, allocation or backend failures. Requesting a previous solution
    /// before a successful solve is invalid. Errors invalidate solution access;
    /// obey the error's context disposition before reusing the backend.
    pub fn solve<A, P>(
        &mut self,
        operator: &A,
        preconditioner: &P,
        rhs: &B::Buffer<T>,
        initial: DeviceInitialGuess,
        config: &BackendKrylovConfig<T::Real>,
        cancellation: Option<&CancellationToken>,
    ) -> Result<BackendKrylovStatus<T::Real>, BackendKrylovError>
    where
        A: DeviceOperator<B, T> + ?Sized,
        P: DevicePreconditioner<B, T> + ?Sized,
    {
        let previous_valid = std::mem::replace(&mut self.valid, false);
        let backend = self.pool.backend();
        if initial == DeviceInitialGuess::PreviousSolution && !previous_valid {
            return Err(BackendKrylovError::InvalidArgument {
                method: self.method,
                reason: "no certified previous device solution".into(),
                diagnostic: BackendKrylovError::diagnostic_for(
                    self.method,
                    backend.capabilities().backend_name,
                    0,
                    BackendOutputDisposition::Preserved,
                    BackendContextDisposition::Reusable,
                ),
            });
        }
        let guess = (initial == DeviceInitialGuess::PreviousSolution).then(|| self.solution.raw());
        let status = try_scalar_backend(
            backend,
            self.pool,
            &mut self.workspace,
            self.method,
            operator,
            preconditioner,
            rhs,
            guess,
            self.candidate.raw_mut(),
            config,
            cancellation,
        )?;
        if status.converged {
            std::mem::swap(&mut self.solution, &mut self.candidate);
            self.valid = true;
        }
        Ok(status)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_compute::{CpuBackend, DeviceCsr, MemoryPoolConfig, PointJacobiBackend};

    #[test]
    fn repeated_preconditioned_solve_and_failed_publication() {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let backend = pool.backend();
        let matrix = DeviceCsr::try_from_host(
            backend,
            2,
            2,
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[4.0, 1.0, 1.0, 3.0],
        )
        .unwrap();
        let preconditioner = PointJacobiBackend::try_from_diagonal(backend, &[4.0, 3.0]).unwrap();
        let mut rhs = pool.try_alloc::<f64>(2).unwrap();
        backend.try_copy_h2d(&[6.0, 7.0], rhs.raw_mut()).unwrap();
        let mut session = DeviceGmresSession::try_new(&pool, 2).unwrap();
        let config = BackendKrylovConfig {
            max_iterations: 8,
            restart: 2,
            tolerance: 1e-12,
            require_native: true,
            exact_residual_check_interval: 0,
        };
        assert!(
            session
                .solve(
                    &matrix,
                    &preconditioner,
                    rhs.raw(),
                    DeviceInitialGuess::PreviousSolution,
                    &config,
                    None
                )
                .is_err()
        );
        let first = session
            .solve(
                &matrix,
                &preconditioner,
                rhs.raw(),
                DeviceInitialGuess::Zero,
                &config,
                None,
            )
            .unwrap();
        assert!(first.converged);
        let mut result = [0.0; 2];
        backend
            .try_copy_d2h(session.solution().unwrap(), &mut result)
            .unwrap();
        assert!((result[0] - 1.0_f64).abs() < 1e-11 && (result[1] - 2.0_f64).abs() < 1e-11);
        let repeated = session
            .solve(
                &matrix,
                &preconditioner,
                rhs.raw(),
                DeviceInitialGuess::PreviousSolution,
                &config,
                None,
            )
            .unwrap();
        assert!(repeated.converged);
        assert_eq!(repeated.iterations, 0);
        backend.try_copy_h2d(&[9.0, 5.0], rhs.raw_mut()).unwrap();
        assert!(
            session
                .solve(
                    &matrix,
                    &preconditioner,
                    rhs.raw(),
                    DeviceInitialGuess::PreviousSolution,
                    &config,
                    None
                )
                .unwrap()
                .converged
        );
        backend
            .try_copy_d2h(session.solution().unwrap(), &mut result)
            .unwrap();
        assert!((result[0] - 2.0).abs() < 1e-11 && (result[1] - 1.0).abs() < 1e-11);
        let short = BackendKrylovConfig {
            max_iterations: 1,
            restart: 1,
            ..config.clone()
        };
        let incomplete = session
            .solve(
                &matrix,
                &preconditioner,
                rhs.raw(),
                DeviceInitialGuess::Zero,
                &short,
                None,
            )
            .unwrap();
        assert!(!incomplete.converged);
        assert!(session.solution().is_none());
        assert!(
            session
                .solve(
                    &matrix,
                    &preconditioner,
                    rhs.raw(),
                    DeviceInitialGuess::Zero,
                    &config,
                    None
                )
                .unwrap()
                .converged
        );
        let cancel = CancellationToken::new();
        cancel.cancel();
        assert!(
            session
                .solve(
                    &matrix,
                    &preconditioner,
                    rhs.raw(),
                    DeviceInitialGuess::PreviousSolution,
                    &config,
                    Some(&cancel)
                )
                .is_err()
        );
        assert!(session.solution().is_none());
        drop(session);
        drop(rhs);
        assert_eq!(pool.bytes_in_use(), 0);
    }

    #[test]
    fn restart_storage_is_reused_without_reusing_stale_iterates() {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let backend = pool.backend();
        let matrix = DeviceCsr::try_from_host(
            backend,
            2,
            2,
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[4.0, 1.0, 1.0, 3.0],
        )
        .unwrap();
        let preconditioner = PointJacobiBackend::try_from_diagonal(backend, &[4.0, 3.0]).unwrap();
        let mut rhs = pool.try_alloc::<f64>(2).unwrap();
        backend.try_copy_h2d(&[6.0, 7.0], rhs.raw_mut()).unwrap();
        let mut session = DeviceGmresSession::try_new(&pool, 2).unwrap();
        session.try_reserve_cycle(1).unwrap();
        assert!(session.workspace.scalar_restart.is_some());
        let config = BackendKrylovConfig {
            max_iterations: 100,
            restart: 1,
            tolerance: 1e-10,
            require_native: true,
            exact_residual_check_interval: 0,
        };
        let first = session
            .solve(
                &matrix,
                &preconditioner,
                rhs.raw(),
                DeviceInitialGuess::Zero,
                &config,
                None,
            )
            .unwrap();
        assert!(first.converged && first.restarts > 0);
        assert!(session.workspace.scalar_restart.is_some());
        let retained = pool.bytes_in_use();
        let peak = pool.high_water_bytes();
        let second = session
            .solve(
                &matrix,
                &preconditioner,
                rhs.raw(),
                DeviceInitialGuess::Zero,
                &config,
                None,
            )
            .unwrap();
        assert!(second.converged);
        assert_eq!(first.iterations, second.iterations);
        assert_eq!(pool.bytes_in_use(), retained);
        assert_eq!(pool.high_water_bytes(), peak);
        // With one iteration the changed RHS must fail to converge. Starting
        // from the stale restart vector instead of zero changes its residual.
        backend.try_copy_h2d(&[9.0, 5.0], rhs.raw_mut()).unwrap();
        let short = BackendKrylovConfig {
            max_iterations: 1,
            ..config
        };
        let reused = session
            .solve(
                &matrix,
                &preconditioner,
                rhs.raw(),
                DeviceInitialGuess::Zero,
                &short,
                None,
            )
            .unwrap();
        let mut fresh = DeviceGmresSession::try_new(&pool, 2).unwrap();
        let reference = fresh
            .solve(
                &matrix,
                &preconditioner,
                rhs.raw(),
                DeviceInitialGuess::Zero,
                &short,
                None,
            )
            .unwrap();
        assert!(!reused.converged);
        assert!((reused.true_residual - reference.true_residual).abs() < 1e-14);
        assert!(
            fresh.workspace.scalar_restart.is_none(),
            "no restart allocation at iteration limit"
        );
        let before_clear = pool.bytes_in_use();
        session.workspace.clear().unwrap();
        assert!(session.workspace.scalar_restart.is_none());
        assert!(pool.bytes_in_use() < before_clear);
    }

    #[test]
    fn constructor_refunds_partial_allocation() {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(16)).unwrap();
        assert!(DeviceGmresSession::<_, f64>::try_new(&pool, 2).is_err());
        assert_eq!(pool.bytes_in_use(), 0);
    }
}
