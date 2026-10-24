//! Retained CUDA preparation from the public FEM solver configuration.

// Rust guideline compliant 2026-02-21
use super::*;
use crate::solver::{SolverConfig, SolverType};
use sonium_compute::{
    BackendResult, DeviceAdjointCapability, DevicePreconditionerDiagnosticsSnapshot,
    IdentityDevicePreconditioner, PointJacobiBackend,
};
use sonium_solvers::preconditioners::{
    AdditiveSchwarzPreconditioner, BackendAmgPreconditioner, CudaIluPreconditioner,
    CudaSchwarzPreconditioner,
};

/// Opaque device action prepared from an explicit FEM solver configuration.
pub struct ConfiguredFemPreconditioner {
    action: Box<dyn DevicePreconditioner<CudaBackend, Complex64>>,
}

impl std::fmt::Debug for ConfiguredFemPreconditioner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConfiguredFemPreconditioner")
            .finish_non_exhaustive()
    }
}

impl DevicePreconditioner<CudaBackend, Complex64> for ConfiguredFemPreconditioner {
    fn apply(
        &self,
        backend: &CudaBackend,
        rhs: &CudaBuffer<Complex64>,
        output: &mut CudaBuffer<Complex64>,
    ) {
        self.action.apply(backend, rhs, output);
    }

    fn try_apply(
        &self,
        backend: &CudaBackend,
        rhs: &CudaBuffer<Complex64>,
        output: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        self.action.try_apply(backend, rhs, output)
    }

    fn forward_capability(&self) -> DeviceAdjointCapability {
        self.action.forward_capability()
    }

    fn diagnostics(&self) -> DevicePreconditionerDiagnosticsSnapshot {
        self.action.diagnostics()
    }
}

/// Retains a configured complex FEM operator, preconditioner and GMRES workspace.
///
/// Assembly and preconditioner setup remain host work. Numeric device updates
/// are available through [`Self::session_mut`]; explicitly lagging a
/// preconditioner preserves its configured action until explicitly refreshed.
/// Transpose/Hermitian requests prepare the selected action's canonical matrix,
/// not an assumed adjoint of the forward preconditioner.
#[derive(Debug)]
pub struct CudaConfiguredFemSolveSession<'a> {
    session: CudaFemSolveSession<'a, Complex64, ConfiguredFemPreconditioner>,
    config: SolverConfig,
    controls: BackendKrylovConfig<f64>,
}

impl<'a> CudaConfiguredFemSolveSession<'a> {
    /// Prepare the requested FEM solver once, without substituting solver variants.
    ///
    /// Supports standard GMRES with identity, Jacobi, ILU(0), one-level Schwarz
    /// or AMG. Schwarz honors `schwarz_subdomains` and `schwarz_overlap` exactly.
    /// AMG uses the CPU default hierarchy with native coarse LU; indefinite
    /// problems needing a shifted auxiliary operator should use the preparation
    /// closure on [`CudaFemSolveSession::try_assemble`] instead.
    ///
    /// # Errors
    /// Rejects unsupported solvers, adaptive restart, non-default convergence
    /// cadence, progress printing and invalid controls before assembly. Propagates
    /// workload admission, native primitive, assembly and preconditioner failures.
    pub fn try_assemble(
        pool: &'a MemoryPool<CudaBackend>,
        workload: FemWorkload,
        config: &SolverConfig,
        assemble: impl FnOnce() -> Result<FemSystem<Complex64>, FemDeviceSolveError>,
    ) -> Result<Self, FemDeviceSolveError> {
        let controls = checked_controls(config)?;
        require_preconditioner_primitives(pool.backend(), config)?;
        let session = CudaFemSolveSession::try_assemble(
            pool,
            workload,
            FemDeviceMethod::Gmres,
            assemble,
            |matrix| prepare(pool.backend(), matrix, config),
        )?;
        Ok(Self {
            session,
            config: config.clone(),
            controls,
        })
    }

    /// Prepare the configured CUDA solver directly from constrained canonical CSR.
    ///
    /// This is the retained CUDA entry point for preassembled FEM producers.
    /// It preserves the controls and preconditioner choices of
    /// [`Self::try_assemble`] without converting CSR to COO. Matrix ordering,
    /// requested action and RHS follow [`CudaFemSolveSession::try_from_csr`].
    /// Subsequent device RHS/numeric updates use [`Self::session_mut`].
    ///
    /// # Errors
    /// Rejects unsupported configuration/workloads, malformed or noncanonical
    /// CSR and invalid/non-finite RHS data. Propagates native setup failures.
    pub fn try_from_csr(
        pool: &'a MemoryPool<CudaBackend>,
        workload: FemWorkload,
        config: &SolverConfig,
        matrix: &CsrMatrix<Complex64>,
        rhs: &[Complex64],
    ) -> Result<Self, FemDeviceSolveError> {
        let controls = checked_controls(config)?;
        require_preconditioner_primitives(pool.backend(), config)?;
        let session = CudaFemSolveSession::try_from_csr(
            pool,
            workload,
            FemDeviceMethod::Gmres,
            matrix,
            rhs,
            |matrix| prepare(pool.backend(), matrix, config),
        )?;
        Ok(Self {
            session,
            config: config.clone(),
            controls,
        })
    }

    /// Prepare a replacement action without mutating the retained solve state.
    ///
    /// Supply a finite matrix in the canonical ordering returned by
    /// `session().pattern()`, already transposed/conjugated for the requested
    /// action. Values may describe a chosen lagged or auxiliary system. The
    /// configured solver and Schwarz settings are preserved. Setup/factorization
    /// is host work and allocates a new preconditioner; the old action remains
    /// installed until `session_mut().replace_preconditioner(candidate)`.
    ///
    /// # Errors
    /// Rejects dimension, pattern, value-count and non-finite data before setup.
    /// Propagates preparation/upload errors without changing the installed action,
    /// published solution or warm start; backend errors may require recovery.
    pub fn try_prepare_preconditioner(
        &self,
        canonical: &CsrMatrix<Complex64>,
    ) -> Result<ConfiguredFemPreconditioner, FemDeviceSolveError> {
        let n = self.session.rhs.len();
        if canonical.num_rows != n
            || canonical.num_cols != n
            || self.session.pattern()
                != (
                    canonical.row_ptrs.as_slice(),
                    canonical.col_indices.as_slice(),
                )
            || canonical.values.len() != canonical.col_indices.len()
            || canonical
                .values
                .iter()
                .any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err(invalid("preconditioner replacement requires finite values on the canonical session pattern").into());
        }
        Ok(prepare(
            self.session.pool.backend(),
            canonical,
            &self.config,
        )?)
    }

    /// Rebuild the configured preconditioner while retaining operator and Krylov state.
    ///
    /// The matrix follows [`Self::try_prepare_preconditioner`]'s canonical action
    /// contract. Successful installation unpublishes the solution but preserves
    /// its warm-start eligibility. Physical matrix/RHS values remain unchanged.
    ///
    /// # Errors
    /// Propagates preparation errors without replacing the installed action.
    pub fn try_refresh_preconditioner(
        &mut self,
        canonical: &CsrMatrix<Complex64>,
    ) -> Result<(), FemDeviceSolveError> {
        let candidate = self.try_prepare_preconditioner(canonical)?;
        self.session.replace_preconditioner(candidate);
        Ok(())
    }

    /// Return the retained solver configuration.
    pub fn config(&self) -> &SolverConfig {
        &self.config
    }

    /// Return the retained device session and its certified solution.
    pub fn session(&self) -> &CudaFemSolveSession<'a, Complex64, ConfiguredFemPreconditioner> {
        &self.session
    }

    /// Access resident RHS and numeric updates with the existing invalidation contract.
    pub fn session_mut(
        &mut self,
    ) -> &mut CudaFemSolveSession<'a, Complex64, ConfiguredFemPreconditioner> {
        &mut self.session
    }

    /// Solve using the retained configuration and certify the original system.
    ///
    /// # Errors
    /// Propagates invalid updates, unavailable warm starts and native solver failures.
    pub fn solve(
        &mut self,
        initial: DeviceInitialGuess,
    ) -> Result<FemDeviceSolveStatus, FemDeviceSolveError> {
        self.session.solve(initial, &self.controls)
    }
}

fn require_preconditioner_primitives(
    backend: &CudaBackend,
    config: &SolverConfig,
) -> BackendResult<()> {
    if matches!(
        config.solver_type,
        SolverType::GmresIlu | SolverType::GmresSchwarz
    ) {
        backend.require_native_gs_color_sweep::<Complex64>()?;
    }
    Ok(())
}

fn checked_controls(
    config: &SolverConfig,
) -> Result<BackendKrylovConfig<f64>, FemDeviceSolveError> {
    if !matches!(
        config.solver_type,
        SolverType::Gmres
            | SolverType::GmresJacobi
            | SolverType::GmresIlu
            | SolverType::GmresSchwarz
            | SolverType::GmresAmg
    ) {
        return Err(invalid(
            "requested FEM solver has no equivalent configured CUDA implementation",
        )
        .into());
    }
    let gmres = &config.gmres;
    if gmres.max_iterations == 0
        || gmres.restart == 0
        || !gmres.tolerance.is_finite()
        || gmres.tolerance <= 0.0
    {
        return Err(invalid("invalid configured GMRES controls").into());
    }
    if gmres.min_restart.is_some()
        || gmres.max_restart.is_some()
        || gmres.convergence_check_interval != 1
        || gmres.print_interval != 0
        || config.verbosity != 0
    {
        return Err(invalid("adaptive restart, convergence cadence and progress printing are unsupported on configured CUDA FEM").into());
    }
    if config.solver_type == SolverType::GmresSchwarz && config.schwarz_subdomains == 0 {
        return Err(invalid("Schwarz requires at least one subdomain").into());
    }
    Ok(BackendKrylovConfig {
        max_iterations: gmres.max_iterations,
        restart: gmres.restart,
        tolerance: gmres.tolerance,
        exact_residual_check_interval: gmres.exact_residual_check_interval,
        require_native: true,
    })
}

fn prepare(
    backend: &CudaBackend,
    matrix: &CsrMatrix<Complex64>,
    config: &SolverConfig,
) -> BackendResult<ConfiguredFemPreconditioner> {
    let action: Box<dyn DevicePreconditioner<CudaBackend, Complex64>> = match config.solver_type {
        SolverType::Gmres => Box::new(IdentityDevicePreconditioner),
        SolverType::GmresJacobi => {
            let diagonal: Vec<_> = (0..matrix.num_rows)
                .map(|row| matrix.get(row, row))
                .collect();
            Box::new(PointJacobiBackend::try_from_diagonal(backend, &diagonal)?)
        }
        SolverType::GmresIlu => Box::new(CudaIluPreconditioner::try_from_csr(backend, matrix)?),
        SolverType::GmresSchwarz => {
            let host = AdditiveSchwarzPreconditioner::try_from_csr(
                matrix,
                config.schwarz_subdomains,
                config.schwarz_overlap,
            )
            .map_err(|error| invalid(&error.to_string()))?;
            Box::new(CudaSchwarzPreconditioner::try_from_host(backend, &host)?)
        }
        SolverType::GmresAmg => Box::new(
            BackendAmgPreconditioner::<CudaBackend, Complex64>::try_from_csr_with_cuda_coarse_lu(
                backend,
                matrix,
                sonium_solvers::AmgConfig::default(),
            )?,
        ),
        _ => return Err(invalid("unsupported configured CUDA FEM preconditioner")),
    };
    Ok(ConfiguredFemPreconditioner { action })
}
