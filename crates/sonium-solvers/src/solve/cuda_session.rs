//! Retained high-level CUDA solves with explicit host and device result boundaries.

// Rust guideline compliant 2026-02-21
use super::*;
use sonium_compute::{Buffer, CudaBuffer, DeviceBufferLike};

mod preconditioner;
mod prepared;
mod update;
use preconditioner::SessionPreconditioner;
#[doc(inline)]
pub use update::CudaLinearUpdateStatus;

/// Certified device result without downloading the solution vector.
#[derive(Debug, Clone, Copy)]
pub struct CudaLinearSolveStatus {
    /// Number of iterations used by this solve.
    pub iterations: usize,
    /// Recomputed relative residual, or absolute residual for a zero RHS.
    pub residual: f64,
}

enum KrylovSession<'a, T: DeviceScalar> {
    Gmres(Box<DeviceGmresSession<'a, CudaBackend, T>>),
    Pcg(Box<DevicePcgSession<'a, CudaBackend, T>>),
}

/// Retains a CUDA matrix, preconditioner and Krylov workspace across right-hand sides.
///
/// The caller owns the pool and therefore the context and quota. This session
/// supports the facade's explicit GMRES, GMRES+Jacobi, GMRES+ILU, GMRES+AMG, GMRES+Chebyshev, CG and CG+AMG
/// strategies. AMG and CG retain the facade's SPD/HPD admission requirements.
/// Chebyshev retains the CPU facade's real-SPD coefficient restriction (also
/// for complex scalar storage), default polynomial degree and spectral interval.
/// Its spectral estimate is host setup/update work; polynomial application is
/// resident and retains its matrix and workspace across solves and updates.
/// ILU uses the CPU default checked ILU(0) factors with native dependency-level
/// triangular application. Factorization stays on the host; compatible refreshes
/// retain factors and scratch. Long dependency chains require many GPU launches.
/// Schwarz uses the CPU automatic partition policy with overlap depth one and
/// partition-of-unity weights. GMRES+Schwarz retains a stationary action;
/// FGMRES+Schwarz additionally retains the flexible Krylov basis. Both retain
/// packed local ILU factors and sparse gather/assembly maps between solves.
/// Auto resolves once using the CPU size/scalar policy: ILU for small or real
/// systems, FGMRES–Schwarz for large complex systems. The selected strategy is
/// retained across numeric updates. CUDA uses its resident standard/flexible
/// Krylov implementation, not the CPU-only optional pipelined-GMRES variant.
/// [`Self::try_new_flexible`] accepts a caller-prepared device preconditioner for
/// FGMRES, including shifted auxiliary AMG and Schur actions on indefinite systems.
/// Matrix values can be refreshed with [`Self::try_update_matrix`]. Canonical
/// sparsity, strategy and controls are fixed; construct a new session when
/// they change. AMG hierarchy construction remains host work.
///
/// Host solves upload the RHS and download/certify the result. Device solves
/// retain bulk vectors; small Krylov control and validation transfers remain.
/// Workspace warms on first use. Errors invalidate published solution access,
/// including warm starts, until a successful zero-start solve. A failed numeric
/// upload additionally requires a successful matrix update before any solve.
///
/// # Examples
///
/// ```no_run
/// use sonium_compute::{CudaBackend, MemoryPool, MemoryPoolConfig};
/// use sonium_solvers::{CsrMatrix, CudaLinearSolveSession, LinearSolveConfig,
///     LinearSolverType, LinearSystemAssumption};
/// use sonium_solvers::iterative::DeviceInitialGuess;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let pool = MemoryPool::new(CudaBackend::new(0)?, MemoryPoolConfig::default())?;
/// let matrix = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 2.0_f64), (1, 1, 3.0)]);
/// let config = LinearSolveConfig { solver: LinearSolverType::GmresJacobi,
///     ..Default::default() };
/// let mut session = CudaLinearSolveSession::try_new(
///     &pool, &matrix, &config, LinearSystemAssumption::General)?;
/// let first = session.solve_host(&[2.0, 3.0], DeviceInitialGuess::Zero)?;
/// let second = session.solve_host(&[4.0, 6.0], DeviceInitialGuess::PreviousSolution)?;
/// assert!(first.converged && second.converged);
/// # Ok(())
/// # }
/// ```
pub struct CudaLinearSolveSession<'a, T>
where
    T: ComplexField + DeviceScalar<Real = <T as ComplexField>::Real>,
{
    pool: &'a MemoryPool<CudaBackend>,
    host_matrix: CsrMatrix<T>,
    matrix: PersistentDeviceCsr<CudaBackend, T>,
    preconditioner: SessionPreconditioner<T>,
    rhs: Buffer<T, CudaBackend>,
    krylov: KrylovSession<'a, T>,
    config: LinearSolveConfig,
    requested_solver: LinearSolverType,
    controls: BackendKrylovConfig<<T as DeviceScalar>::Real>,
    assumption: LinearSystemAssumption,
    published: bool,
    warm_start_available: bool,
    numeric_ready: bool,
}

impl<T> std::fmt::Debug for CudaLinearSolveSession<'_, T>
where
    T: ComplexField + DeviceScalar<Real = <T as ComplexField>::Real>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaLinearSolveSession")
            .field("dimension", &self.host_matrix.num_rows)
            .field("solver", &self.config.solver)
            .field("requested_solver", &self.requested_solver)
            .field(
                "flexible",
                &(self.preconditioner.is_prepared()
                    || self.config.solver == LinearSolverType::FgmresSchwarz),
            )
            .field("has_solution", &self.published)
            .field("numeric_ready", &self.numeric_ready)
            .field("warm_start_available", &self.warm_start_available)
            .finish_non_exhaustive()
    }
}

pub(super) fn validate_strategy(config: &LinearSolveConfig) -> Result<(), LinearSolveError> {
    if !matches!(
        config.solver,
        LinearSolverType::Auto
            | LinearSolverType::Gmres
            | LinearSolverType::GmresJacobi
            | LinearSolverType::GmresIlu
            | LinearSolverType::GmresSchwarz
            | LinearSolverType::FgmresSchwarz
            | LinearSolverType::GmresAmg
            | LinearSolverType::GmresChebyshev
            | LinearSolverType::Cg
            | LinearSolverType::CgAmg
    ) {
        return Err(LinearSolveError::BackendUnavailable {
            backend: LinearSolveBackend::Cuda,
            reason: format!(
                "high-level CUDA facade supports CG with identity/AMG or GMRES with identity/Jacobi/ILU/AMG/Chebyshev/Schwarz or FGMRES+Schwarz (requested {:?})",
                config.solver
            ),
        });
    }
    Ok(())
}

fn backend_error(error: impl std::fmt::Display) -> LinearSolveError {
    LinearSolveError::BackendFailure {
        backend: LinearSolveBackend::Cuda,
        reason: error.to_string(),
    }
}

impl<'a, T> CudaLinearSolveSession<'a, T>
where
    T: ComplexField + DeviceScalar<Real = <T as ComplexField>::Real>,
    <T as ComplexField>::Real: FromPrimitive + Sync + Send,
{
    /// Prepare retained CUDA state for a fixed matrix and solver configuration.
    ///
    /// Duplicate CSR entries are coalesced for cuSPARSE. The original host
    /// matrix is retained for independent certification of host results.
    ///
    /// # Errors
    /// Rejects unsupported strategies, invalid matrices/controls and unproven
    /// SPD/HPD requirements. Propagates allocation and preconditioner failures.
    /// Runtime primitive admission precedes preconditioner/device preparation.
    pub fn try_new(
        pool: &'a MemoryPool<CudaBackend>,
        matrix: &CsrMatrix<T>,
        config: &LinearSolveConfig,
        assumption: LinearSystemAssumption,
    ) -> Result<Self, LinearSolveError> {
        validate_strategy(config)?;
        config.validate()?;
        // Validate structural dimensions before allocating a dummy setup RHS.
        validate_csr_structure(matrix)?;
        validate_linear_inputs(
            matrix,
            &vec![<T as DeviceScalar>::zero(); matrix.num_rows],
            None,
            config,
        )?;
        if matches!(
            config.solver,
            LinearSolverType::Cg
                | LinearSolverType::CgAmg
                | LinearSolverType::GmresAmg
                | LinearSolverType::GmresChebyshev
        ) {
            validate_spd_system(matrix, config.solver, assumption)?;
        }
        Self::from_validated(pool, matrix, config, assumption)
    }

    // The one-shot facade validates before creating a CUDA context. Reuse that
    // admission here rather than performing bounded SPD factorization twice.
    pub(super) fn from_validated(
        pool: &'a MemoryPool<CudaBackend>,
        matrix: &CsrMatrix<T>,
        config: &LinearSolveConfig,
        assumption: LinearSystemAssumption,
    ) -> Result<Self, LinearSolveError> {
        let requested_solver = config.solver;
        let mut resolved = config.clone();
        resolved.solver = resolve_auto_solver(config.solver, matrix, is_complex_scalar::<T>());
        let config = &resolved;
        if matches!(
            config.solver,
            LinearSolverType::Cg | LinearSolverType::CgAmg
        ) {
            crate::iterative::validate_backend_pcg::<_, T>(pool.backend(), true)
                .map_err(backend_error)?;
        } else {
            crate::iterative::validate_backend_krylov::<_, T>(
                pool.backend(),
                if config.solver == LinearSolverType::FgmresSchwarz {
                    crate::iterative::BackendKrylovMethod::Fgmres
                } else {
                    crate::iterative::BackendKrylovMethod::Gmres
                },
                true,
            )
            .map_err(backend_error)?;
        }
        let canonical = coalesced_csr(matrix);
        let preconditioner =
            SessionPreconditioner::try_new(pool.backend(), &canonical, config.solver)?;
        let mut session =
            Self::from_prepared(pool, matrix, &canonical, config, assumption, preconditioner)?;
        session.requested_solver = requested_solver;
        Ok(session)
    }

    fn from_prepared(
        pool: &'a MemoryPool<CudaBackend>,
        matrix: &CsrMatrix<T>,
        canonical: &CsrMatrix<T>,
        config: &LinearSolveConfig,
        assumption: LinearSystemAssumption,
        preconditioner: SessionPreconditioner<T>,
    ) -> Result<Self, LinearSolveError> {
        let tolerance = <<T as ComplexField>::Real as FromPrimitive>::from_f64(config.tolerance)
            .filter(|value| {
                value.is_finite() && *value > <<T as ComplexField>::Real as Zero>::zero()
            })
            .ok_or_else(|| {
                LinearSolveError::InvalidConfig(
                    "tolerance is not representable by device scalar".into(),
                )
            })?;
        let device_matrix =
            PersistentDeviceCsr::try_from_host_csr_ref(pool.backend(), canonical.as_host_csr_ref())
                .map_err(backend_error)?;
        let rhs = pool.try_alloc(matrix.num_rows).map_err(backend_error)?;
        let krylov = if matches!(
            config.solver,
            LinearSolverType::Cg | LinearSolverType::CgAmg
        ) {
            KrylovSession::Pcg(Box::new(
                DevicePcgSession::try_new(
                    pool,
                    matrix.num_cols,
                    DevicePcgAssumption::PositiveDefinite,
                )
                .map_err(backend_error)?,
            ))
        } else {
            let mut session = if preconditioner.is_prepared()
                || config.solver == LinearSolverType::FgmresSchwarz
            {
                DeviceGmresSession::try_new_flexible(pool, matrix.num_cols)
            } else {
                DeviceGmresSession::try_new(pool, matrix.num_cols)
            }
            .map_err(backend_error)?;
            session
                .try_reserve_cycle(config.restart.min(config.max_iterations))
                .map_err(backend_error)?;
            pool.backend()
                .try_reserve_krylov_reductions::<T>(config.restart.min(config.max_iterations))
                .map_err(backend_error)?;
            KrylovSession::Gmres(Box::new(session))
        };
        Ok(Self {
            pool,
            host_matrix: matrix.clone(),
            matrix: device_matrix,
            preconditioner,
            rhs,
            krylov,
            config: config.clone(),
            requested_solver: config.solver,
            controls: BackendKrylovConfig {
                max_iterations: config.max_iterations,
                restart: config.restart,
                tolerance,
                require_native: true,
                exact_residual_check_interval: config.exact_residual_check_interval,
            },
            assumption,
            published: false,
            warm_start_available: false,
            numeric_ready: true,
        })
    }

    /// Return the requested named strategy, or none for caller-prepared FGMRES.
    #[must_use]
    pub fn requested_strategy(&self) -> Option<LinearSolverType> {
        (!self.preconditioner.is_prepared()).then_some(self.requested_solver)
    }

    /// Return the retained strategy after resolving Auto during construction.
    ///
    /// Returns none for caller-prepared FGMRES, which has no named facade
    /// strategy. Numeric updates retain the selected strategy; reconstruct the
    /// session to re-evaluate Auto after changing selection thresholds.
    #[must_use]
    pub fn selected_strategy(&self) -> Option<LinearSolverType> {
        (!self.preconditioner.is_prepared()).then_some(self.config.solver)
    }

    /// Borrow the last certified solution without downloading its vector.
    #[must_use]
    pub fn solution_device(&self) -> Option<&CudaBuffer<T>> {
        if !self.published {
            return None;
        }
        match &self.krylov {
            KrylovSession::Gmres(session) => session.solution(),
            KrylovSession::Pcg(session) => session.solution(),
        }
    }

    /// Solve a resident RHS and publish only independently certified device output.
    ///
    /// # Errors
    /// Rejects invalid RHS buffers or unavailable warm starts. Nonconvergence
    /// and backend failures invalidate output; recover the backend as required
    /// by its error state before attempting another zero-start solve.
    pub fn solve_device(
        &mut self,
        rhs: &CudaBuffer<T>,
        initial: DeviceInitialGuess,
    ) -> Result<CudaLinearSolveStatus, LinearSolveError> {
        self.run(Some(rhs), initial)
    }

    fn run(
        &mut self,
        rhs: Option<&CudaBuffer<T>>,
        initial: DeviceInitialGuess,
    ) -> Result<CudaLinearSolveStatus, LinearSolveError> {
        self.published = false;
        let previous = std::mem::replace(&mut self.warm_start_available, false);
        if !self.numeric_ready {
            return Err(LinearSolveError::InvalidConfig(
                "CUDA numeric update failed; complete a matrix update before solving".into(),
            ));
        }
        if initial == DeviceInitialGuess::PreviousSolution && !previous {
            return Err(LinearSolveError::InvalidConfig(
                "no certified previous CUDA solution".into(),
            ));
        }
        let rhs = rhs.unwrap_or_else(|| self.rhs.raw());
        // Reject foreign streams before Krylov copies or reductions touch RHS.
        // Sparse-operator validation happens later and cannot protect this boundary.
        if rhs.len() != self.host_matrix.num_rows
            || rhs.raw().is_none_or(|raw| {
                !std::sync::Arc::ptr_eq(raw.stream(), self.pool.backend().stream())
                    || rhs.len().checked_mul(std::mem::size_of::<T>()) != Some(raw.len())
            })
        {
            return Err(LinearSolveError::InvalidConfig(
                "CUDA RHS must have matching dimensions, allocation and session stream".into(),
            ));
        }
        let (iterations, residual, converged) = match &mut self.krylov {
            KrylovSession::Gmres(session) => {
                let result = session
                    .solve(
                        &self.matrix,
                        self.preconditioner.action(),
                        rhs,
                        initial,
                        &self.controls,
                        None,
                    )
                    .map_err(backend_error)?;
                (result.iterations, result.true_residual, result.converged)
            }
            KrylovSession::Pcg(session) => {
                let result = session
                    .solve(
                        &self.matrix,
                        self.preconditioner.action(),
                        rhs,
                        initial,
                        &PcgBackendConfig {
                            max_iterations: self.config.max_iterations,
                            tolerance: self.controls.tolerance,
                        },
                    )
                    .map_err(backend_error)?;
                (result.iterations, result.true_residual, result.converged)
            }
        };
        let residual = residual.to_f64().unwrap_or(f64::INFINITY);
        if !converged || !residual.is_finite() || residual > self.config.tolerance {
            return Err(LinearSolveError::NotConverged {
                iterations,
                residual,
            });
        }
        self.published = true;
        self.warm_start_available = true;
        Ok(CudaLinearSolveStatus {
            iterations,
            residual,
        })
    }

    /// Solve a host RHS and independently certify the downloaded result.
    ///
    /// Context, matrix, preconditioner and device workspace survive this call.
    /// Only the new RHS is uploaded and the solution downloaded. Host result
    /// allocation and host residual/backward-error evaluation remain explicit.
    ///
    /// # Errors
    /// Rejects invalid RHS values/shape or unavailable warm starts. Propagates
    /// backend and certification failures; every error invalidates output.
    pub fn solve_host(
        &mut self,
        rhs: &[T],
        initial: DeviceInitialGuess,
    ) -> Result<LinearSolveResult<T>, LinearSolveError> {
        let result = self.solve_host_inner(rhs, initial);
        if result.is_err() {
            self.published = false;
            self.warm_start_available = false;
        }
        result
    }

    fn solve_host_inner(
        &mut self,
        rhs: &[T],
        initial: DeviceInitialGuess,
    ) -> Result<LinearSolveResult<T>, LinearSolveError> {
        if !self.numeric_ready {
            return Err(LinearSolveError::InvalidConfig(
                "CUDA numeric update failed; complete a matrix update before solving".into(),
            ));
        }
        if rhs.len() != self.host_matrix.num_rows {
            return Err(LinearSolveError::DimensionMismatch {
                operation: "linear solve rhs",
                expected: self.host_matrix.num_rows,
                actual: rhs.len(),
            });
        }
        if let Some(index) = first_non_finite(rhs) {
            return Err(LinearSolveError::NonFiniteInput {
                operation: "rhs",
                index,
            });
        }
        self.pool
            .backend()
            .try_copy_h2d(rhs, self.rhs.raw_mut())
            .map_err(backend_error)?;
        let status = self.run(None, initial)?;
        let output = self
            .solution_device()
            .ok_or(LinearSolveError::NotConverged {
                iterations: status.iterations,
                residual: status.residual,
            })?;
        let mut x = vec![<T as DeviceScalar>::zero(); self.host_matrix.num_cols];
        self.pool
            .backend()
            .try_copy_d2h(output, &mut x)
            .map_err(backend_error)?;
        self.pool
            .backend()
            .try_synchronize()
            .map_err(backend_error)?;
        let residual = true_relative_residual(&self.host_matrix, rhs, &x)?;
        if residual > self.config.tolerance {
            return Err(LinearSolveError::NotConverged {
                iterations: status.iterations,
                residual,
            });
        }
        let backward_error = true_backward_error(&self.host_matrix, rhs, &x)?;
        let result = LinearSolveResult {
            x,
            iterations: status.iterations,
            residual,
            backward_error,
            converged: true,
            status: SolverStatus::Converged,
            attainable_accuracy: AccuracyStatus::Attained,
            assumption: self.assumption,
            assumption_provenance: assumption_provenance(
                &self.host_matrix,
                self.config.solver,
                self.assumption,
            ),
            backend: LinearSolveBackendProvenance::native(LinearSolveBackend::Cuda),
            saddle_assumptions: None,
        };
        result.validate_contract()?;
        Ok(result)
    }
}
