//! Retained affine tetrahedral pressure solves with shifted AMG and native updates.
//!
//! Geometry, constraints and the auxiliary AMG hierarchy are prepared once on
//! the host. CSR mode assembles numeric values on CUDA; element mode only updates
//! coefficient scales and retains no physical CSR or assembly map. Both the
//! vendor CSR and fused element-action routes use the same FGMRES session and
//! shifted preconditioner. Natural homogeneous Neumann and prescribed pressure
//! are supported. P1/P2 sessions additionally support fixed Robin/Neumann facets.

// Rust guideline compliant 2026-02-21
use super::{
    HelmholtzForm,
    device_tet::{prepare_tet_p1_helmholtz, prepare_tet_p2_helmholtz},
};
use crate::{mesh::Mesh, spaces::H1Space};
use num_complex::Complex64;

mod boundary;
use boundary::TetBoundary;
#[doc(inline)]
pub use boundary::TetRobinBoundary;
use sonium_compute::cuda::{CudaBackend, CudaBuffer};
mod volume;
use sonium_compute::fem_tet::TetAcousticC64;
use sonium_compute::{BackendError, BackendResult, Buffer, DeviceBufferLike, MemoryPool};
use sonium_solvers::iterative::{
    BackendKrylovConfig, BackendKrylovError, BackendKrylovStatus, CancellationToken,
    DeviceGmresSession, DeviceInitialGuess,
};
use sonium_solvers::{AmgConfig, BackendAmgPreconditioner, CsrMatrix};
use volume::TetVolume;

/// Physical operator used by the retained tetrahedral solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TetPressureOperator {
    /// Numeric CSR assembly followed by vendor sparse actions.
    Csr,
    /// Fused element action without physical CSR storage or numeric assembly maps.
    Element,
}

/// Fixed constraints, operator selection and auxiliary AMG configuration.
#[derive(Debug, Clone)]
pub struct TetPressureConfig {
    /// Original H1 pressure DOFs, including prescribed edge DOFs for P2.
    pub dirichlet: Vec<usize>,
    /// Auxiliary operator is K + mass_shift M with the same constraints.
    pub mass_shift: Complex64,
    /// AMG hierarchy and smoother choices for the shifted operator.
    pub amg: AmgConfig,
    /// Physical action implementation; CSR remains the performance baseline.
    pub operator: TetPressureOperator,
}

/// Setup, update and solve failure for a retained tetrahedral pressure session.
#[derive(Debug, thiserror::Error)]
pub enum TetPressureError {
    /// Backend validation, resource or execution failure.
    #[error(transparent)]
    Backend(#[from] BackendError),
    /// Krylov validation, cancellation or execution failure.
    #[error(transparent)]
    Krylov(Box<BackendKrylovError>),
}
impl From<BackendKrylovError> for TetPressureError {
    fn from(error: BackendKrylovError) -> Self {
        Self::Krylov(Box::new(error))
    }
}

/// Retained P1 pressure session on affine tetrahedra.
pub type CudaTetP1PressureSession<'a> = CudaTetPressureSession<'a, 4>;
/// Retained P2 pressure session on affine tetrahedra.
pub type CudaTetP2PressureSession<'a> = CudaTetPressureSession<'a, 10>;

/// Retains tetrahedral volume, constraints, shifted AMG and flexible Krylov storage.
///
/// Changing frequency retains the auxiliary hierarchy as a lagged preconditioner.
/// Call `try_rebuild_preconditioner` when convergence requires a different shift.
/// Solutions are published only after original-system residual certification.
/// Mesh and constraints are immutable. This volume session does not imply
/// support for curved geometry or generic FEM dispatch.
pub struct CudaTetPressureSession<'a, const N: usize> {
    pool: &'a MemoryPool<CudaBackend>,
    host: TetAcousticC64<N>,
    volume: TetVolume<N>,
    preconditioner: BackendAmgPreconditioner<CudaBackend, Complex64>,
    boundary: Option<TetBoundary<'a>>,
    krylov: DeviceGmresSession<'a, CudaBackend, Complex64>,
    lifted: Buffer<Complex64, CudaBackend>,
    mode: TetPressureOperator,
    valid: bool,
    published: bool,
    guess_available: bool,
}
impl<const N: usize> std::fmt::Debug for CudaTetPressureSession<'_, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaTetPressureSession")
            .field("n", &self.host.dimension())
            .field("mode", &self.mode)
            .field("valid", &self.valid)
            .field("published", &self.published)
            .finish_non_exhaustive()
    }
}
fn invalid(reason: impl Into<String>) -> BackendError {
    BackendError::InvalidArgument {
        operation: "cuda_tet_pressure_solve",
        reason: reason.into(),
    }
}
fn square(k: Complex64) -> BackendResult<Complex64> {
    let squared = k * k;
    if [k, squared]
        .iter()
        .any(|z| !z.re.is_finite() || !z.im.is_finite())
    {
        return Err(invalid("wavenumber and its square must be finite"));
    }
    Ok(squared)
}
fn shifted<const N: usize>(
    host: &TetAcousticC64<N>,
    shift: Complex64,
) -> BackendResult<CsrMatrix<Complex64>> {
    let (pattern, values) = host.assemble_csr(Complex64::new(1.0, 0.0), -shift)?;
    let n = host.dimension();
    let mut entries = Vec::with_capacity(values.len());
    for row in 0..n {
        let range = pattern.row_ptrs()[row] as usize..pattern.row_ptrs()[row + 1] as usize;
        for (&column, &value) in pattern.columns()[range.clone()].iter().zip(&values[range]) {
            entries.push((row, column as usize, value));
        }
    }
    CsrMatrix::try_from_triplets(n, n, entries).map_err(|error| invalid(error.to_string()))
}

impl<'a> CudaTetPressureSession<'a, 4> {
    /// Prepare a retained affine P1 volume solve from FEM mesh/space inputs.
    ///
    /// # Errors
    /// Rejects incompatible mesh/space, invalid coefficients/constraints and
    /// propagates backend or AMG setup failures without a CPU solve fallback.
    pub fn try_new(
        pool: &'a MemoryPool<CudaBackend>,
        mesh: &Mesh,
        space: &H1Space,
        form: &HelmholtzForm,
        config: &TetPressureConfig,
    ) -> Result<Self, TetPressureError> {
        Self::admit(pool, false)?;
        let host = prepare_tet_p1_helmholtz(mesh, space, &HelmholtzForm::from_real(1.0))?
            .try_with_dirichlet(&config.dirichlet)?;
        Self::prepare(pool, host, form, config, None)
    }
}
impl<'a> CudaTetPressureSession<'a, 4> {
    /// Prepare a P1 solve with fixed Robin/Neumann facets and prescribed pressure.
    ///
    /// Boundary callbacks run during host setup only. Their action is included
    /// in both the physical operator and shifted AMG. Frequency updates retain
    /// these fixed coefficients; frequency-dependent impedance uses `try_update_boundary`.
    ///
    /// # Errors
    /// Rejects duplicate/missing markers, invalid geometry or coefficients, and
    /// propagates allocation, upload and solver setup failures.
    pub fn try_new_with_boundaries(
        pool: &'a MemoryPool<CudaBackend>,
        mesh: &Mesh,
        space: &H1Space,
        form: &HelmholtzForm,
        config: &TetPressureConfig,
        boundaries: &[TetRobinBoundary<'_>],
    ) -> Result<Self, TetPressureError> {
        Self::admit(pool, !boundaries.is_empty())?;
        let host = prepare_tet_p1_helmholtz(mesh, space, &HelmholtzForm::from_real(1.0))?
            .try_with_dirichlet(&config.dirichlet)?;
        let boundary = TetBoundary::prepare(pool, mesh, space, config, boundaries)?;
        Self::prepare(pool, host, form, config, boundary)
    }
}
impl<'a> CudaTetPressureSession<'a, 10> {
    /// Prepare a P2 solve with fixed Robin/Neumann facets and prescribed pressure.
    ///
    /// Boundary callbacks run during host setup only. Their action is included
    /// in both the physical operator and shifted AMG. Frequency updates retain
    /// these fixed coefficients; frequency-dependent impedance uses `try_update_boundary`.
    ///
    /// # Errors
    /// Rejects duplicate/missing markers, invalid geometry or coefficients, and
    /// propagates allocation, upload and solver setup failures.
    pub fn try_new_with_boundaries(
        pool: &'a MemoryPool<CudaBackend>,
        mesh: &Mesh,
        space: &H1Space,
        form: &HelmholtzForm,
        config: &TetPressureConfig,
        boundaries: &[TetRobinBoundary<'_>],
    ) -> Result<Self, TetPressureError> {
        Self::admit(pool, !boundaries.is_empty())?;
        let host = prepare_tet_p2_helmholtz(mesh, space, &HelmholtzForm::from_real(1.0))?
            .try_with_dirichlet(&config.dirichlet)?;
        let boundary = TetBoundary::prepare(pool, mesh, space, config, boundaries)?;
        Self::prepare(pool, host, form, config, boundary)
    }
}
impl<'a> CudaTetPressureSession<'a, 10> {
    /// Prepare a retained affine P2 volume solve from FEM mesh/space inputs.
    ///
    /// # Errors
    /// Rejects incompatible mesh/space, invalid coefficients/constraints and
    /// propagates backend or AMG setup failures without a CPU solve fallback.
    pub fn try_new(
        pool: &'a MemoryPool<CudaBackend>,
        mesh: &Mesh,
        space: &H1Space,
        form: &HelmholtzForm,
        config: &TetPressureConfig,
    ) -> Result<Self, TetPressureError> {
        Self::admit(pool, false)?;
        let host = prepare_tet_p2_helmholtz(mesh, space, &HelmholtzForm::from_real(1.0))?
            .try_with_dirichlet(&config.dirichlet)?;
        Self::prepare(pool, host, form, config, None)
    }
}
impl<'a, const N: usize> CudaTetPressureSession<'a, N> {
    fn admit(pool: &MemoryPool<CudaBackend>, boundaries: bool) -> Result<(), TetPressureError> {
        sonium_solvers::iterative::validate_backend_krylov::<_, Complex64>(
            pool.backend(),
            sonium_solvers::iterative::BackendKrylovMethod::Fgmres,
            true,
        )?;
        pool.backend().try_prepare_tet_kernels::<N>()?;
        if boundaries {
            pool.backend().try_prepare_quadrature_kernels()?;
        }
        Ok(())
    }

    fn prepare(
        pool: &'a MemoryPool<CudaBackend>,
        host: TetAcousticC64<N>,
        form: &HelmholtzForm,
        config: &TetPressureConfig,
        boundary: Option<TetBoundary<'a>>,
    ) -> Result<Self, TetPressureError> {
        square(form.wavenumber)?;
        let mut auxiliary = shifted(&host, config.mass_shift)?;
        if let Some(boundary) = &boundary {
            auxiliary = boundary.add_to(&auxiliary, pool.backend())?;
        }
        let preconditioner =
            BackendAmgPreconditioner::<CudaBackend, Complex64>::try_from_csr_with_cuda_coarse_lu(
                pool.backend(),
                &auxiliary,
                config.amg.clone(),
            )?;
        let n = host.dimension();
        let volume = TetVolume::prepare(pool, &host, config.operator)?;
        let mut session = Self {
            pool,
            host,
            volume,
            preconditioner,
            boundary,
            krylov: DeviceGmresSession::try_new_flexible(pool, n).map_err(BackendError::from)?,
            lifted: pool.try_alloc(n).map_err(BackendError::from)?,
            mode: config.operator,
            valid: false,
            published: false,
            guess_available: false,
        };
        session.try_update_wavenumber(form.wavenumber)?;
        Ok(session)
    }

    /// Opt into fused CSR Jacobi sweeps in the auxiliary AMG preconditioner.
    ///
    /// Disabled by default. Preserves the hierarchy, damping, sweep counts,
    /// published solution and warm start. No device work or allocation occurs.
    /// Numeric refresh and explicit preconditioner rebuild retain the selection.
    /// Other smoother algorithms are unchanged.
    pub fn set_fused_jacobi_enabled(&mut self, enabled: bool) {
        self.preconditioner.set_fused_jacobi_enabled(enabled);
    }

    /// Return whether fused CSR Jacobi sweeps are explicitly enabled.
    pub fn fused_jacobi_enabled(&self) -> bool {
        self.preconditioner.fused_jacobi_enabled()
    }

    /// Update physical coefficients and numeric CSR while retaining all geometry.
    ///
    /// The auxiliary AMG hierarchy remains lagged. No host numeric matrix is
    /// assembled or transferred. Element mode updates only host launch parameters,
    /// with no device operation. An interrupted update invalidates the operator
    /// until another update succeeds; the previous solution is unpublished.
    ///
    /// # Errors
    /// Rejects non-finite coefficients and propagates native assembly/copy errors.
    pub fn try_update_wavenumber(&mut self, k: Complex64) -> Result<(), TetPressureError> {
        self.published = false;
        let mass = square(k)?;
        self.valid = false;
        let result = self.volume.update(self.pool.backend(), mass);
        if result.is_err() {
            self.guess_available = false;
        }
        result?;
        self.valid = true;
        Ok(())
    }

    /// Rebuild auxiliary AMG for a new mass shift without replacing solve storage.
    ///
    /// This explicit setup step assembles the auxiliary matrix and hierarchy on
    /// the host. Updated boundary values are explicitly downloaded for this setup.
    /// Failure preserves the previous preconditioner but hides output.
    ///
    /// # Errors
    /// Rejects invalid shifts and propagates AMG, allocation and upload failures.
    pub fn try_rebuild_preconditioner(
        &mut self,
        shift: Complex64,
        amg: AmgConfig,
    ) -> Result<(), TetPressureError> {
        self.published = false;
        let mut auxiliary = shifted(&self.host, shift)?;
        if let Some(boundary) = &self.boundary {
            auxiliary = boundary.add_to(&auxiliary, self.pool.backend())?;
        }
        let mut replacement =
            BackendAmgPreconditioner::<CudaBackend, Complex64>::try_from_csr_with_cuda_coarse_lu(
                self.pool.backend(),
                &auxiliary,
                amg,
            )?;
        replacement.set_fused_jacobi_enabled(self.preconditioner.fused_jacobi_enabled());
        self.preconditioner = replacement;
        Ok(())
    }

    /// Refresh a compatible auxiliary hierarchy while retaining its device storage.
    ///
    /// Builds the new hierarchy on the host, including boundary readback when
    /// present. Returns `false` without modifying preconditioning if level or
    /// transfer patterns change; call `try_rebuild_preconditioner` in that case.
    /// Successful refresh retains coarse LU, level operators and Krylov storage.
    /// Output is hidden on every attempt. A partial refresh blocks preconditioner
    /// application until a complete refresh or rebuild succeeds.
    ///
    /// # Errors
    /// Rejects invalid shifts/hierarchies and propagates upload or factorization
    /// failures. Backend recovery may be required before retrying.
    pub fn try_refresh_preconditioner(
        &mut self,
        shift: Complex64,
        amg: AmgConfig,
    ) -> Result<bool, TetPressureError> {
        self.published = false;
        let mut auxiliary = shifted(&self.host, shift)?;
        if let Some(boundary) = &self.boundary {
            auxiliary = boundary.add_to(&auxiliary, self.pool.backend())?;
        }
        let host = sonium_solvers::AmgPreconditioner::try_from_csr(&auxiliary, amg)
            .map_err(|e| invalid(e.to_string()))?;
        Ok(self
            .preconditioner
            .try_refresh_from_host_amg(self.pool.backend(), &host)?)
    }

    /// Coordinates and markers in the resident boundary coefficient sample order.
    pub fn boundary_samples(&self) -> Option<(&[[f64; 3]], &[i32])> {
        self.boundary
            .as_ref()
            .map(|b| (b.points.as_slice(), b.markers.as_slice()))
    }

    /// Update boundary action, prescribed-pressure coupling and sources on CUDA.
    ///
    /// Supply finite alpha/source samples in `boundary_samples` order. Geometry
    /// and constraints stay fixed. Numeric assembly and copies retain all storage;
    /// no host coefficient arrays are transferred. AMG stays lagged until rebuilt.
    /// A successful update unpublishes the old solution but preserves its iterate.
    ///
    /// # Errors
    /// Rejects sessions without boundary maps or incompatible buffers. A failed
    /// update blocks solves until a complete retry succeeds, after backend recovery
    /// if required. Non-finite arithmetic is detected by solve certification.
    pub fn try_update_boundary(
        &mut self,
        alpha: &CudaBuffer<Complex64>,
        source: &CudaBuffer<Complex64>,
    ) -> Result<(), TetPressureError> {
        self.published = false;
        let previous_guess = std::mem::replace(&mut self.guess_available, false);
        let boundary = self
            .boundary
            .as_mut()
            .ok_or_else(|| invalid("boundary maps were not prepared"))?;
        boundary.update(self.pool.backend(), alpha, source)?;
        self.guess_available = previous_guess;
        Ok(())
    }

    /// Borrow the last residual-certified device solution.
    pub fn solution(&self) -> Option<&CudaBuffer<Complex64>> {
        if self.published {
            self.krylov.solution()
        } else {
            None
        }
    }

    /// Lift resident prescribed pressure and solve with retained FGMRES and AMG.
    ///
    /// The RHS is the unconstrained volume load; prescribed free-node entries
    /// are ignored. The first solve reserves Krylov capacity for later solves.
    ///
    /// # Errors
    /// Rejects invalid state, shapes or controls; propagates cancellation and
    /// native errors. Failed/unconverged calls unpublish the previous solution.
    pub fn solve(
        &mut self,
        rhs: &CudaBuffer<Complex64>,
        prescribed: &CudaBuffer<Complex64>,
        initial: DeviceInitialGuess,
        config: &BackendKrylovConfig<f64>,
        cancellation: Option<&CancellationToken>,
    ) -> Result<BackendKrylovStatus<f64>, TetPressureError> {
        self.published = false;
        let had_solution = std::mem::replace(&mut self.guess_available, false);
        if !self.valid
            || self.boundary.as_ref().is_some_and(|b| !b.valid)
            || (initial == DeviceInitialGuess::PreviousSolution && !had_solution)
        {
            return Err(
                invalid("incomplete operator update or unavailable previous solution").into(),
            );
        }
        if rhs.len() != self.host.dimension()
            || prescribed.len() != self.host.dimension()
            || config.max_iterations == 0
            || config.restart == 0
            || !config.tolerance.is_finite()
            || config.tolerance < 0.0
        {
            return Err(invalid("invalid RHS, prescribed values or Krylov controls").into());
        }
        self.krylov
            .try_reserve_cycle(config.restart.min(config.max_iterations))?;
        self.pool
            .backend()
            .try_reserve_krylov_reductions::<Complex64>(
                config.restart.min(config.max_iterations),
            )?;
        self.volume.element().try_lift_rhs_colmajor(
            self.pool.backend(),
            1,
            rhs,
            prescribed,
            self.lifted.raw_mut(),
        )?;
        if let Some(boundary) = &mut self.boundary {
            boundary.lift(self.pool.backend(), prescribed, self.lifted.raw_mut())?;
        }
        let operator = self.volume.action();
        let status = if let Some(boundary) = &self.boundary {
            self.krylov.solve(
                &boundary.sum.try_bind(operator, &boundary.action)?,
                &self.preconditioner,
                self.lifted.raw(),
                initial,
                config,
                cancellation,
            )?
        } else {
            self.krylov.solve(
                operator,
                &self.preconditioner,
                self.lifted.raw(),
                initial,
                config,
                cancellation,
            )?
        };
        self.published = status.converged;
        self.guess_available = status.converged;
        Ok(status)
    }
}

#[cfg(test)]
#[path = "device_tet_solve_tests.rs"]
mod tests;
