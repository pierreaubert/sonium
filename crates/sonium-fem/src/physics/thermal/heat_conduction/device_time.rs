//! Retained CUDA theta steps for fixed linear thermal operators.

// Rust guideline compliant 2026-02-21
use super::ThermalTimeStepper;
use crate::solver::device::{FemDeviceSolveError, FemDeviceSolveStatus};
use sonium_compute::{
    Backend, BackendError, Buffer, CudaBackend, CudaBuffer, DeviceBufferLike, HostCsrRef,
    MemoryPool, PcgBackendConfig, PersistentDeviceCsr,
};
use sonium_solvers::iterative::{DeviceInitialGuess, DevicePcgAssumption, DevicePcgSession};
use sonium_solvers::{
    AmgConfig, AmgPreconditioner, CsrMatrix, preconditioners::BackendAmgPreconditioner,
};

fn invalid(reason: impl Into<String>) -> FemDeviceSolveError {
    BackendError::InvalidArgument {
        operation: "cuda_thermal_time_step",
        reason: reason.into(),
    }
    .into()
}

fn validate_scheme(scheme: &ThermalTimeStepper) -> Result<(), FemDeviceSolveError> {
    let theta = scheme.scheme.theta();
    if !scheme.dt.is_finite()
        || scheme.dt <= 0.0
        || !(1.0 / scheme.dt).is_finite()
        || !theta.is_finite()
        || !(0.0..=1.0).contains(&theta)
    {
        return Err(invalid(
            "positive finite dt and theta in [0,1] are required",
        ));
    }
    Ok(())
}

/// Retains thermal history, sparse operators, symmetric AMG and PCG scratch.
///
/// Matrix assembly, constraint reduction and AMG setup are host work. Subsequent
/// history actions, source combination and solves run on CUDA. A failed step
/// leaves the last committed temperature unchanged; no partial step is exposed.
/// The pool covers history and Krylov buffers, not all legacy sparse allocations.
pub struct CudaThermalTimeStepper<'a> {
    pool: &'a MemoryPool<CudaBackend>,
    scheme: ThermalTimeStepper,
    host_mass: CsrMatrix<f64>,
    host_stiffness: CsrMatrix<f64>,
    effective_pattern: (Vec<usize>, Vec<usize>),
    numeric_ready: bool,
    mass_basis: Buffer<f64, CudaBackend>,
    stiffness_basis: Buffer<f64, CudaBackend>,
    mass: PersistentDeviceCsr<CudaBackend, f64>,
    stiffness: PersistentDeviceCsr<CudaBackend, f64>,
    effective: PersistentDeviceCsr<CudaBackend, f64>,
    preconditioner: BackendAmgPreconditioner<CudaBackend, f64>,
    solver: DevicePcgSession<'a, CudaBackend, f64>,
    temperature: Buffer<f64, CudaBackend>,
    candidate: Buffer<f64, CudaBackend>,
    rhs: Buffer<f64, CudaBackend>,
    scratch: Buffer<f64, CudaBackend>,
    warm: bool,
    steps: usize,
}

impl std::fmt::Debug for CudaThermalTimeStepper<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaThermalTimeStepper")
            .field("scheme", &self.scheme)
            .field("dimension", &self.temperature.len())
            .field("steps", &self.steps)
            .finish_non_exhaustive()
    }
}

impl ThermalTimeStepper {
    /// Prepare repeated CUDA steps with fixed mass, stiffness and time step.
    ///
    /// Supply consistently reduced free-DOF matrices and initial temperatures.
    /// Prescribed-temperature lifting belongs in the caller's source vectors.
    /// `assumption` certifies that the effective operator is SPD. The fixed
    /// preconditioner uses symmetric AMG. Changing matrices or constraints requires
    /// a new session. Use `try_update_scheme` to change dt or theta while retaining
    /// history. No matrix factorization is repeated for an unchanged scheme.
    ///
    /// # Errors
    /// Rejects invalid dt/theta, malformed/non-finite matrices or initial state,
    /// and propagates allocation and backend failures.
    pub fn prepare_cuda<'a>(
        &self,
        pool: &'a MemoryPool<CudaBackend>,
        mass: &CsrMatrix<f64>,
        stiffness: &CsrMatrix<f64>,
        initial_temperature: &[f64],
        assumption: DevicePcgAssumption,
    ) -> Result<CudaThermalTimeStepper<'a>, FemDeviceSolveError> {
        validate_scheme(self)?;
        let n = initial_temperature.len();
        if n == 0 || initial_temperature.iter().any(|v| !v.is_finite()) {
            return Err(invalid("initial temperature must be nonempty and finite"));
        }
        for a in [mass, stiffness] {
            HostCsrRef::try_new(
                a.num_rows,
                a.num_cols,
                &a.row_ptrs,
                &a.col_indices,
                &a.values,
            )
            .map_err(|e| invalid(e.to_string()))?;
            if a.num_rows != n
                || a.num_cols != n
                || a.row_ptrs.first() != Some(&0)
                || a.values.iter().any(|v| !v.is_finite())
            {
                return Err(invalid(
                    "thermal matrices must be matching finite square CSR systems",
                ));
            }
        }
        let effective = self.effective_matrix(mass, stiffness);
        if effective.values.iter().any(|v| !v.is_finite()) {
            return Err(invalid("effective thermal matrix overflow"));
        }
        let aligned = |matrix: &CsrMatrix<f64>| -> Result<Vec<f64>, FemDeviceSolveError> {
            let mut values = vec![0.0; effective.values.len()];
            for row in 0..n {
                let range = effective.row_ptrs[row]..effective.row_ptrs[row + 1];
                for entry in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
                    let offset = effective.col_indices[range.clone()]
                        .binary_search(&matrix.col_indices[entry])
                        .map_err(|_| invalid("effective sparsity omits a thermal basis entry"))?;
                    values[range.start + offset] += matrix.values[entry];
                }
            }
            if values.iter().any(|v| !v.is_finite()) {
                return Err(invalid("thermal numeric basis overflow"));
            }
            Ok(values)
        };
        let mass_values = aligned(mass)?;
        let stiffness_values = aligned(stiffness)?;
        let backend = pool.backend();
        let preconditioner = BackendAmgPreconditioner::try_from_csr(
            backend,
            &effective,
            AmgConfig::for_backend_pcg(),
        )?;
        let mut mass_basis = pool
            .try_alloc(effective.values.len())
            .map_err(BackendError::from)?;
        let mut stiffness_basis = pool
            .try_alloc(effective.values.len())
            .map_err(BackendError::from)?;
        backend.try_copy_h2d(&mass_values, mass_basis.raw_mut())?;
        backend.try_copy_h2d(&stiffness_values, stiffness_basis.raw_mut())?;
        let mut temperature = pool.try_alloc(n).map_err(BackendError::from)?;
        backend.try_copy_h2d(initial_temperature, temperature.raw_mut())?;
        Ok(CudaThermalTimeStepper {
            pool,
            scheme: *self,
            host_mass: mass.clone(),
            host_stiffness: stiffness.clone(),
            effective_pattern: (effective.row_ptrs.clone(), effective.col_indices.clone()),
            numeric_ready: true,
            mass_basis,
            stiffness_basis,
            mass: PersistentDeviceCsr::try_from_host_csr_ref(backend, mass.as_host_csr_ref())?,
            stiffness: PersistentDeviceCsr::try_from_host_csr_ref(
                backend,
                stiffness.as_host_csr_ref(),
            )?,
            effective: PersistentDeviceCsr::try_from_host_csr_ref(
                backend,
                effective.as_host_csr_ref(),
            )?,
            preconditioner,
            solver: DevicePcgSession::try_new(pool, n, assumption).map_err(BackendError::from)?,
            temperature,
            candidate: pool.try_alloc(n).map_err(BackendError::from)?,
            rhs: pool.try_alloc(n).map_err(BackendError::from)?,
            scratch: pool.try_alloc(n).map_err(BackendError::from)?,
            warm: false,
            steps: 0,
        })
    }
}

impl CudaThermalTimeStepper<'_> {
    /// Borrow the initial state or the last successfully committed temperature.
    pub fn temperature(&self) -> &CudaBuffer<f64> {
        self.temperature.raw()
    }

    /// Return the number of successfully committed time steps.
    pub fn steps_completed(&self) -> usize {
        self.steps
    }

    /// Change dt or theta while retaining history, sparse structure and Krylov storage.
    ///
    /// The caller renews the SPD assumption for the new effective operator.
    /// Effective values combine resident bases on CUDA. Host matrix combination
    /// and AMG setup still prepare the preconditioner on each update. Matching AMG
    /// hierarchies refresh device storage; changed hierarchies rebuild only the
    /// preconditioner. Returns whether the preconditioner was rebuilt.
    ///
    /// Validation errors preserve the previous scheme. Once device updates begin, any
    /// failure blocks stepping until a complete update succeeds after backend
    /// recovery. Committed temperatures and the step count never change here.
    /// Retry either the old or new scheme to repair a failed update.
    ///
    /// # Errors
    /// Rejects invalid dt/theta, overflow or changed effective sparsity. Propagates
    /// AMG and backend failures. Terminal failures require a new pool and session.
    pub fn try_update_scheme(
        &mut self,
        scheme: ThermalTimeStepper,
        _assumption: DevicePcgAssumption,
    ) -> Result<bool, FemDeviceSolveError> {
        let warm = std::mem::replace(&mut self.warm, false);
        validate_scheme(&scheme)?;
        let effective = scheme.effective_matrix(&self.host_mass, &self.host_stiffness);
        if effective.values.iter().any(|v| !v.is_finite()) {
            return Err(invalid("effective thermal matrix overflow"));
        }
        if effective.row_ptrs != self.effective_pattern.0
            || effective.col_indices != self.effective_pattern.1
        {
            return Err(invalid(
                "changed effective sparsity requires a new thermal session",
            ));
        }
        let host = AmgPreconditioner::try_from_csr(&effective, AmgConfig::for_backend_pcg())
            .map_err(|e| invalid(e.to_string()))?;
        self.numeric_ready = false;
        let backend = self.pool.backend();
        self.effective.try_combine_scaled_values(
            backend,
            1.0 / scheme.dt,
            self.mass_basis.raw(),
            &[(scheme.scheme.theta(), self.stiffness_basis.raw())],
        )?;
        let rebuilt = !self
            .preconditioner
            .try_refresh_from_host_amg(backend, &host)?;
        if rebuilt {
            self.preconditioner = BackendAmgPreconditioner::try_from_host_amg(backend, &host)?;
        }
        backend.try_synchronize()?;
        self.scheme = scheme;
        self.numeric_ready = true;
        self.warm = warm;
        Ok(rebuilt)
    }

    /// Update the physical matrix while retaining the current symmetric AMG preconditioner.
    ///
    /// This explicit lagging policy avoids host hierarchy construction and bulk
    /// uploads. It can increase PCG iterations when dt/theta changes substantially.
    /// The caller renews the SPD assumption for the new physical operator; the
    /// retained preconditioner stays stationary and symmetric within each solve.
    /// Call `try_update_scheme` whenever fresh AMG is needed.
    ///
    /// Matrix values combine on CUDA and receive a native finite check. Only
    /// compact validation status reaches the host. History and step count are
    /// unchanged. A failed device update blocks stepping and further lagged updates
    /// until a complete `try_update_scheme` succeeds after backend recovery.
    ///
    /// # Errors
    /// Rejects invalid dt/theta, an incomplete previous update, non-finite device
    /// values or unavailable native validation. Propagates backend failures.
    pub fn try_update_scheme_lagged(
        &mut self,
        scheme: ThermalTimeStepper,
        _assumption: DevicePcgAssumption,
    ) -> Result<(), FemDeviceSolveError> {
        let warm = std::mem::replace(&mut self.warm, false);
        if !self.numeric_ready {
            return Err(invalid(
                "incomplete thermal numeric update; refresh the complete scheme",
            ));
        }
        validate_scheme(&scheme)?;
        self.numeric_ready = false;
        let backend = self.pool.backend();
        self.effective.try_combine_scaled_values(
            backend,
            1.0 / scheme.dt,
            self.mass_basis.raw(),
            &[(scheme.scheme.theta(), self.stiffness_basis.raw())],
        )?;
        if backend.try_all_finite_native(&self.effective.matrix().values)? != Some(true) {
            return Err(invalid(
                "effective thermal matrix is non-finite or native validation is unavailable",
            ));
        }
        backend.try_synchronize()?;
        self.scheme = scheme;
        self.numeric_ready = true;
        self.warm = warm;
        Ok(())
    }

    fn check_source(&self, source: &CudaBuffer<f64>) -> Result<(), FemDeviceSolveError> {
        if source.len() != self.temperature.len()
            || source.raw().is_none_or(|raw| {
                !std::sync::Arc::ptr_eq(raw.stream(), self.pool.backend().stream())
                    || source.len().checked_mul(std::mem::size_of::<f64>()) != Some(raw.len())
            })
        {
            return Err(invalid(
                "thermal source length, allocation or stream mismatch",
            ));
        }
        if self.pool.backend().try_all_finite_native(source)? != Some(true) {
            return Err(invalid(
                "source is non-finite or native validation is unavailable",
            ));
        }
        Ok(())
    }

    /// Form the theta history RHS, solve and commit one resident temperature step.
    ///
    /// As in the CPU API, omitted `source_current` contributes zero. For changing
    /// sources with theta below one, pass the actual previous-time load. Small
    /// validation and residual reductions reach the host. Nonconvergence returns
    /// `converged=false` and does not advance history. A successful copy is
    /// synchronized before committing history. Explicit theta may require a CFL
    /// limit; this API does not infer stability from a small linear residual.
    ///
    /// # Errors
    /// Rejects invalid sources/controls and propagates backend/PCG failures.
    /// Recoverable failures allow retry after backend recovery. Terminal context
    /// failures require a new pool and session.
    pub fn step_device(
        &mut self,
        source_next: &CudaBuffer<f64>,
        source_current: Option<&CudaBuffer<f64>>,
        config: &PcgBackendConfig<f64>,
    ) -> Result<FemDeviceSolveStatus, FemDeviceSolveError> {
        let warm = std::mem::replace(&mut self.warm, false);
        if !self.numeric_ready {
            return Err(invalid(
                "incomplete thermal numeric update; retry the complete scheme update",
            ));
        }
        if config.max_iterations == 0 || !config.tolerance.is_finite() || config.tolerance <= 0.0 {
            return Err(invalid("invalid thermal PCG controls"));
        }
        self.check_source(source_next)?;
        if let Some(current) = source_current {
            self.check_source(current)?;
        }
        let backend = self.pool.backend();
        let theta = self.scheme.scheme.theta();
        self.mass
            .try_apply(backend, self.temperature.raw(), self.rhs.raw_mut())?;
        backend.try_scale(1.0 / self.scheme.dt, self.rhs.raw_mut())?;
        if theta != 1.0 {
            self.stiffness
                .try_apply(backend, self.temperature.raw(), self.scratch.raw_mut())?;
            backend.try_axpy(theta - 1.0, self.scratch.raw(), self.rhs.raw_mut())?;
            if let Some(current) = source_current {
                backend.try_axpy(1.0 - theta, current, self.rhs.raw_mut())?;
            }
        }
        backend.try_axpy(theta, source_next, self.rhs.raw_mut())?;
        let status = self.solver.solve(
            &self.effective,
            &self.preconditioner,
            self.rhs.raw(),
            if warm {
                DeviceInitialGuess::PreviousSolution
            } else {
                DeviceInitialGuess::Zero
            },
            config,
        )?;
        if status.converged {
            let next = self
                .solver
                .solution()
                .ok_or_else(|| invalid("missing certified thermal solution"))?;
            backend.try_copy_d2d(next, self.candidate.raw_mut())?;
            backend.try_synchronize()?;
            std::mem::swap(&mut self.temperature, &mut self.candidate);
            self.steps += 1;
            self.warm = true;
        }
        Ok(FemDeviceSolveStatus {
            iterations: status.iterations,
            true_residual: status.true_residual,
            converged: status.converged,
        })
    }
}

#[cfg(test)]
mod tests;
