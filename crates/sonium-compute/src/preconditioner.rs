//! Device-resident preconditioners.
//!
//! These are small, backend-neutral preconditioners that only need
//! `sonium-compute` primitives. Physics-specific or CPU-only preconditioners
//! stay in `sonium-solvers`.

use crate::gmres::{BackendCsr, BackendPreconditioner};
use crate::operator::{
    DeviceAdjointCapability, PersistentDeviceCsr, validate_csr_inputs,
};
use crate::{
    Backend, DeviceBufferLike, DevicePreconditioner, DeviceScalar,
    PreconditionerFailureInjectionPoint,
};
use num_traits::Float;
use std::sync::Mutex;

mod block_jacobi;
mod chebyshev;
mod matrix;
use matrix::PreconditionerMatrix;
pub mod saddle_point;

pub use block_jacobi::{BlockJacobiBackend, DEFAULT_BLOCK_SIZE};
pub use chebyshev::{ChebyshevBackend, ChebyshevBackendF32};
pub use saddle_point::{
    PressureSchurBackendKind, SaddlePointJacobiSchurBackend, SaddlePointJacobiSchurBackendConfig,
    embed_pressure_schur_with_velocity_identity,
};

/// Inverse-diagonal preconditioner over scalar `T`.
///
/// Stores `1 / A_ii` on-device as a diagonal CSR matrix; `apply` is a single
/// SpMV with that diagonal matrix.
pub struct PointJacobiBackend<B: Backend, T: DeviceScalar = f64> {
    diag_inv: PersistentDeviceCsr<B, T>,
    n: usize,
    workspace: Mutex<Option<B::Buffer<T>>>,
    // Block applications must not evict the scalar scratch used by recovery.
    block_workspace: Mutex<Option<B::Buffer<T>>>,
    runtime_diagnostics: crate::DevicePreconditionerDiagnostics,
}

impl<B: Backend, T: DeviceScalar> PointJacobiBackend<B, T> {
    /// Build from the host-side diagonal of `A`.
    ///
    /// Entries with `|d| <= epsilon(T::Real)` are rejected by the checked
    /// constructor.  A silently regularized row would no longer represent
    /// the requested Jacobi preconditioner.
    pub fn from_diagonal(backend: &B, diag_host: &[T]) -> Self {
        Self::try_from_diagonal(backend, diag_host).expect("point Jacobi construction failed")
    }

    /// Checked variant of [`Self::from_diagonal`].
    pub fn try_from_diagonal(backend: &B, diag_host: &[T]) -> crate::BackendResult<Self> {
        if diag_host.iter().any(|value| !value.is_finite()) {
            return Err(crate::BackendError::InvalidArgument {
                operation: "point_jacobi_constructor",
                reason: "diagonal values must be finite".into(),
            });
        }
        let n = diag_host.len();
        let eps = T::Real::epsilon();
        let mut diag_inv: Vec<T> = Vec::with_capacity(n);
        for (index, &d) in diag_host.iter().enumerate() {
            let abs = d.norm_sqr().sqrt();
            if abs <= eps {
                return Err(crate::BackendError::InvalidArgument {
                    operation: "point_jacobi_constructor",
                    reason: format!("diagonal entry {index} is too small for Jacobi inversion"),
                });
            }
            diag_inv.push(T::one() / d);
        }
        backend.try_inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::DiagonalUpload,
        )?;
        let diag_inv = PersistentDeviceCsr::try_from_host_diagonal(backend, &diag_inv)?;
        backend.try_inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::WorkspaceAllocation,
        )?;
        let workspace = backend
            .raw_alloc::<T>(n)
            .map_err(crate::BackendError::from)?;
        let runtime_diagnostics = crate::DevicePreconditionerDiagnostics::default();
        runtime_diagnostics.record_setup();

        Ok(Self {
            diag_inv,
            n,
            workspace: Mutex::new(Some(workspace)),
            block_workspace: Mutex::new(None),
            runtime_diagnostics,
        })
    }

    /// Update the inverse diagonal in-place for a new matrix value set.
    /// The resident CSR structure and prebaked sparse descriptor are retained.
    pub fn try_update_diagonal(
        &mut self,
        backend: &B,
        diag_host: &[T],
    ) -> crate::BackendResult<()> {
        if diag_host.len() != self.n {
            return Err(crate::BackendError::InvalidArgument {
                operation: "point_jacobi_update",
                reason: format!(
                    "expected diagonal length {}, got {}",
                    self.n,
                    diag_host.len()
                ),
            });
        }
        if diag_host.iter().any(|value| !value.is_finite()) {
            return Err(crate::BackendError::InvalidArgument {
                operation: "point_jacobi_update",
                reason: "diagonal values must be finite".into(),
            });
        }
        backend.try_inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::DiagonalUpload,
        )?;
        let eps = T::Real::epsilon();
        let mut inverse = Vec::with_capacity(self.n);
        for (index, &d) in diag_host.iter().enumerate() {
            if d.norm_sqr().sqrt() <= eps {
                return Err(crate::BackendError::InvalidArgument {
                    operation: "point_jacobi_update",
                    reason: format!("diagonal entry {index} is too small for Jacobi inversion"),
                });
            }
            inverse.push(T::one() / d);
        }
        self.diag_inv.try_copy_values_from_host(backend, &inverse)
    }

    /// Build directly from a host CSR matrix by extracting its diagonal.
    pub fn from_csr_host(
        backend: &B,
        m: usize,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
        values_host: &[T],
    ) -> Self {
        Self::try_from_csr_host(backend, m, row_ptr_host, col_idx_host, values_host)
            .expect("point Jacobi CSR construction failed")
    }

    /// Checked variant of [`Self::from_csr_host`].
    pub fn try_from_csr_host(
        backend: &B,
        m: usize,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
        values_host: &[T],
    ) -> crate::BackendResult<Self> {
        validate_csr_inputs(m, m, row_ptr_host, col_idx_host, values_host)?;

        let mut diag = vec![T::zero(); m];
        for row in 0..m {
            let lo = row_ptr_host[row] as usize;
            let hi = row_ptr_host[row + 1] as usize;
            for k in lo..hi {
                if col_idx_host[k] as usize == row {
                    diag[row] = values_host[k];
                    break;
                }
            }
        }
        Self::try_from_diagonal(backend, &diag)
    }
}

impl<B: Backend, T: DeviceScalar> DevicePreconditioner<B, T> for PointJacobiBackend<B, T> {
    fn apply(&self, backend: &B, r: &B::Buffer<T>, z: &mut B::Buffer<T>) {
        if let Err(error) = self.try_apply(backend, r, z) {
            log::error!(
                target: "sonium_compute::preconditioner",
                "point-Jacobi application failed: {error}"
            );
        }
    }

    fn try_apply(
        &self,
        backend: &B,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        let result = self.try_apply_untracked(backend, r, z);
        match &result {
            Ok(()) => self.runtime_diagnostics.record_apply(),
            Err(error) => self.runtime_diagnostics.record_failure(error),
        }
        result
    }

    fn try_apply_untracked(
        &self,
        backend: &B,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        backend.try_inject_preconditioner_failure(PreconditionerFailureInjectionPoint::Launch)?;
        if r.len() != self.n || z.len() != self.n {
            return Err(crate::BackendError::InvalidArgument {
                operation: "point_jacobi_apply",
                reason: format!(
                    "expected input/output length {}, got {}/{}",
                    self.n,
                    r.len(),
                    z.len()
                ),
            });
        }
        let mut workspace =
            self.workspace
                .lock()
                .map_err(|_| crate::BackendError::OperationFailed {
                    operation: "point_jacobi_apply",
                    reason: "workspace mutex poisoned".into(),
                })?;
        if workspace
            .as_ref()
            .is_none_or(|buffer| buffer.len() != self.n)
        {
            backend.try_inject_preconditioner_failure(
                PreconditionerFailureInjectionPoint::WorkspaceAllocation,
            )?;
            *workspace = Some(
                backend
                    .raw_alloc::<T>(self.n)
                    .map_err(crate::BackendError::from)?,
            );
        }
        let scratch = workspace
            .as_mut()
            .ok_or(crate::BackendError::OperationFailed {
                operation: "point_jacobi_apply",
                reason: "workspace was not initialized".into(),
            })?;
        self.diag_inv.try_apply(backend, r, scratch)?;
        backend.try_copy_d2d(scratch, z)
    }

    fn apply_colmajor(&self, backend: &B, rhs: usize, r: &B::Buffer<T>, z: &mut B::Buffer<T>) {
        if let Err(error) = self.try_apply_colmajor(backend, rhs, r, z) {
            log::error!(
                target: "sonium_compute::preconditioner",
                "point-Jacobi multi-RHS application failed: {error}"
            );
        }
    }

    fn try_apply_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        backend.try_inject_preconditioner_failure(PreconditionerFailureInjectionPoint::Launch)?;
        let result = self.try_apply_colmajor_untracked(backend, rhs, r, z);
        match &result {
            Ok(()) => self.runtime_diagnostics.record_apply(),
            Err(error) => self.runtime_diagnostics.record_failure(error),
        }
        result
    }

    fn try_apply_colmajor_untracked(
        &self,
        backend: &B,
        rhs: usize,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        backend.try_inject_preconditioner_failure(PreconditionerFailureInjectionPoint::Launch)?;
        let len = self
            .n
            .checked_mul(rhs)
            .ok_or(crate::BackendError::InvalidArgument {
                operation: "point_jacobi_apply_colmajor",
                reason: format!("workspace length overflow: n={}, rhs={rhs}", self.n),
            })?;
        if r.len() != len || z.len() != len {
            return Err(crate::BackendError::InvalidArgument {
                operation: "point_jacobi_apply_colmajor",
                reason: format!(
                    "expected input/output length {len}, got {}/{}",
                    r.len(),
                    z.len()
                ),
            });
        }
        let storage = if rhs == 1 {
            &self.workspace
        } else {
            &self.block_workspace
        };
        let mut workspace =
            storage
                .lock()
                .map_err(|_| crate::BackendError::OperationFailed {
                    operation: "point_jacobi_apply_colmajor",
                    reason: "workspace mutex poisoned".into(),
                })?;
        if workspace.as_ref().is_none_or(|buffer| buffer.len() != len) {
            backend.try_inject_preconditioner_failure(
                PreconditionerFailureInjectionPoint::WorkspaceAllocation,
            )?;
            *workspace = Some(
                backend
                    .raw_alloc::<T>(len)
                    .map_err(crate::BackendError::from)?,
            );
        }
        let scratch = workspace
            .as_mut()
            .ok_or(crate::BackendError::OperationFailed {
                operation: "point_jacobi_apply_colmajor",
                reason: "workspace was not initialized".into(),
            })?;
        self.diag_inv.try_apply_colmajor(backend, rhs, r, scratch)?;
        backend.try_copy_d2d(scratch, z)
    }

    fn transpose_capability(&self) -> DeviceAdjointCapability {
        DeviceAdjointCapability::Exact
    }

    fn diagnostics(&self) -> crate::DevicePreconditionerDiagnosticsSnapshot {
        self.runtime_diagnostics.snapshot()
    }

    fn try_apply_transpose(
        &self,
        backend: &B,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        let result = self.try_apply_untracked(backend, r, z);
        match &result {
            Ok(()) => self.runtime_diagnostics.record_transpose(),
            Err(error) => self.runtime_diagnostics.record_transpose_failure(error),
        }
        result
    }

    fn try_apply_hermitian(
        &self,
        _backend: &B,
        _r: &B::Buffer<T>,
        _z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        let error = crate::BackendError::Unsupported {
            backend: "device",
            operation: "point_jacobi_hermitian",
            scalar: "device scalar",
        };
        self.runtime_diagnostics.record_hermitian_failure(&error);
        Err(error)
    }

    fn try_apply_transpose_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        let result = self.try_apply_colmajor_untracked(backend, rhs, r, z);
        match &result {
            Ok(()) => self.runtime_diagnostics.record_transpose(),
            Err(error) => self.runtime_diagnostics.record_transpose_failure(error),
        }
        result
    }

    fn try_apply_hermitian_colmajor(
        &self,
        _backend: &B,
        _rhs: usize,
        _r: &B::Buffer<T>,
        _z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        let error = crate::BackendError::Unsupported {
            backend: "device",
            operation: "point_jacobi_hermitian_colmajor",
            scalar: "device scalar",
        };
        self.runtime_diagnostics.record_hermitian_failure(&error);
        Err(error)
    }
}

/// Polynomial Jacobi preconditioner of degree `K >= 1`.
///
/// Holds a borrowed or owned resident system matrix `A` plus an on-device
/// diagonal CSR storing `1/A_ii`. `apply` runs `K` damped
/// Jacobi-Richardson steps.
pub struct PolynomialJacobiBackend<'a, B: Backend, T: DeviceScalar = f64> {
    a: PreconditionerMatrix<'a, B, T>,
    diag_inv: PersistentDeviceCsr<B, T>,
    degree: usize,
    damping: T,
    workspace: Mutex<PolynomialWorkspace<B, T>>,
    // Scalar recovery must not evict the latest block application scratch.
    block_workspace: Mutex<PolynomialWorkspace<B, T>>,
    runtime_diagnostics: crate::DevicePreconditionerDiagnostics,
}

struct PolynomialWorkspace<B: Backend, T: DeviceScalar> {
    result: B::Buffer<T>,
    rk: B::Buffer<T>,
    dk: B::Buffer<T>,
}

fn ensure_buffer_exact<B: Backend, T: DeviceScalar>(
    backend: &B,
    buffer: &mut B::Buffer<T>,
    len: usize,
) -> crate::BackendResult<()> {
    if buffer.len() != len {
        *buffer = backend
            .raw_alloc::<T>(len)
            .map_err(crate::BackendError::from)?;
    }
    Ok(())
}

pub type PolynomialJacobiBackendF32<'a, B> = PolynomialJacobiBackend<'a, B, f32>;

impl<'a, B: Backend, T: DeviceScalar> PolynomialJacobiBackend<'a, B, T> {
    /// Build from the borrowed device CSR plus the host-side diagonal.
    ///
    /// Entries with `|d| <= epsilon(T::Real)` are rejected rather than
    /// silently regularized.
    pub fn new(
        backend: &'a B,
        a: &'a BackendCsr<B, T>,
        diag_host: &[T],
        degree: usize,
        damping: T,
    ) -> Self {
        Self::try_new(backend, a, diag_host, degree, damping)
            .expect("polynomial Jacobi construction failed")
    }

    /// Checked variant of [`Self::new`].
    pub fn try_new(
        backend: &'a B,
        a: &'a BackendCsr<B, T>,
        diag_host: &[T],
        degree: usize,
        damping: T,
    ) -> crate::BackendResult<Self> {
        Self::try_from_matrix(backend, a.m, diag_host, degree, damping, || {
            PreconditionerMatrix::borrowed(backend, a)
        })
    }

    /// Build from an owned resident matrix, retaining descriptors across numeric updates.
    ///
    /// # Errors
    /// Rejects invalid degree, damping or diagonal and propagates backend failures.
    pub fn try_from_owned_csr(
        backend: &B,
        a: PersistentDeviceCsr<B, T>,
        diag_host: &[T],
        degree: usize,
        damping: T,
    ) -> crate::BackendResult<Self> {
        let n = a.m();
        Self::try_from_matrix(backend, n, diag_host, degree, damping, || {
            Ok(PreconditionerMatrix::Owned {
                matrix: a,
                valid: true,
            })
        })
    }

    /// Refresh owned matrix values and inverse diagonal without replacing device storage.
    ///
    /// Device values must be finite and match the retained sparsity pattern.
    /// Scalar and block scratch remain cached. Validation errors preserve the old
    /// state; a backend failure requires successful refresh before further applies.
    ///
    /// # Errors
    /// Rejects borrowed matrices, invalid shapes/diagonals and backend failures.
    pub fn try_update_numeric(
        &mut self,
        backend: &B,
        values: &B::Buffer<T>,
        diagonal: &[T],
    ) -> crate::BackendResult<()> {
        self.a
            .try_refresh(backend, values, diagonal, &mut self.diag_inv)
    }

    fn try_from_matrix(
        backend: &B,
        rows: usize,
        diag_host: &[T],
        degree: usize,
        damping: T,
        make_matrix: impl FnOnce() -> crate::BackendResult<PreconditionerMatrix<'a, B, T>>,
    ) -> crate::BackendResult<Self> {
        if degree == 0 {
            return Err(crate::BackendError::InvalidArgument {
                operation: "polynomial_jacobi_constructor",
                reason: "polynomial degree must be >= 1".into(),
            });
        }
        if diag_host.len() != rows {
            return Err(crate::BackendError::InvalidArgument {
                operation: "polynomial_jacobi_constructor",
                reason: format!(
                    "diagonal length {} != A row count {}",
                    diag_host.len(),
                    rows
                ),
            });
        }
        if !damping.is_finite() || diag_host.iter().any(|value| !value.is_finite()) {
            return Err(crate::BackendError::InvalidArgument {
                operation: "polynomial_jacobi_constructor",
                reason: "diagonal and damping must be finite".into(),
            });
        }

        let n = diag_host.len();
        let eps = T::Real::epsilon();
        let mut diag_inv: Vec<T> = Vec::with_capacity(n);
        for (index, &d) in diag_host.iter().enumerate() {
            let abs = d.norm_sqr().sqrt();
            if abs <= eps {
                return Err(crate::BackendError::InvalidArgument {
                    operation: "polynomial_jacobi_constructor",
                    reason: format!("diagonal entry {index} is too small for Jacobi inversion"),
                });
            }
            diag_inv.push(T::one() / d);
        }

        backend.try_inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::HierarchyUpload,
        )?;
        let a = make_matrix()?;
        backend.try_inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::DiagonalUpload,
        )?;
        let diag_inv = PersistentDeviceCsr::try_from_host_diagonal(backend, &diag_inv)?;
        backend.try_inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::WorkspaceAllocation,
        )?;
        let workspace = PolynomialWorkspace {
            result: backend
                .raw_alloc::<T>(n)
                .map_err(crate::BackendError::from)?,
            rk: backend
                .raw_alloc::<T>(n)
                .map_err(crate::BackendError::from)?,
            dk: backend
                .raw_alloc::<T>(n)
                .map_err(crate::BackendError::from)?,
        };
        let block_workspace = PolynomialWorkspace {
            result: backend
                .raw_alloc::<T>(0)
                .map_err(crate::BackendError::from)?,
            rk: backend
                .raw_alloc::<T>(0)
                .map_err(crate::BackendError::from)?,
            dk: backend
                .raw_alloc::<T>(0)
                .map_err(crate::BackendError::from)?,
        };
        let runtime_diagnostics = crate::DevicePreconditionerDiagnostics::default();
        runtime_diagnostics.record_setup();

        Ok(Self {
            a,
            diag_inv,
            degree,
            damping,
            workspace: Mutex::new(workspace),
            block_workspace: Mutex::new(block_workspace),
            runtime_diagnostics,
        })
    }

    /// Build directly from the host CSR.
    pub fn from_csr_host(
        backend: &'a B,
        a: &'a BackendCsr<B, T>,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
        values_host: &[T],
        degree: usize,
        damping: T,
    ) -> Self {
        Self::try_from_csr_host(
            backend,
            a,
            row_ptr_host,
            col_idx_host,
            values_host,
            degree,
            damping,
        )
        .expect("polynomial Jacobi CSR construction failed")
    }

    /// Checked variant of [`Self::from_csr_host`].
    pub fn try_from_csr_host(
        backend: &'a B,
        a: &'a BackendCsr<B, T>,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
        values_host: &[T],
        degree: usize,
        damping: T,
    ) -> crate::BackendResult<Self> {
        let m = a.m;
        validate_csr_inputs(m, a.n, row_ptr_host, col_idx_host, values_host)?;

        let mut diag = vec![T::zero(); m];
        for row in 0..m {
            let lo = row_ptr_host[row] as usize;
            let hi = row_ptr_host[row + 1] as usize;
            for k in lo..hi {
                if col_idx_host[k] as usize == row {
                    diag[row] = values_host[k];
                    break;
                }
            }
        }
        Self::try_new(backend, a, &diag, degree, damping)
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn damping(&self) -> T {
        self.damping
    }
}

impl<'a, B: Backend> PolynomialJacobiBackend<'a, B, f32> {
    /// Backward-compatible alias for [`Self::from_csr_host`] on the f32
    /// specialization.
    pub fn from_csr_f32(
        backend: &'a B,
        a: &'a BackendCsr<B, f32>,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
        values_host: &[f32],
        degree: usize,
        damping: f32,
    ) -> Self {
        Self::from_csr_host(
            backend,
            a,
            row_ptr_host,
            col_idx_host,
            values_host,
            degree,
            damping,
        )
    }
}

impl<B: Backend, T: DeviceScalar> BackendPreconditioner<B, T>
    for PolynomialJacobiBackend<'_, B, T>
{
    fn apply(&self, backend: &B, r: &B::Buffer<T>, z: &mut B::Buffer<T>) {
        if let Err(error) = self.try_apply(backend, r, z) {
            log::error!(
                target: "sonium_compute::preconditioner",
                "polynomial-Jacobi application failed: {error}"
            );
        }
    }

    fn try_apply(
        &self,
        backend: &B,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        let result = self.try_apply_untracked(backend, r, z);
        match &result {
            Ok(()) => self.runtime_diagnostics.record_apply(),
            Err(error) => self.runtime_diagnostics.record_failure(error),
        }
        result
    }

    fn try_apply_untracked(
        &self,
        backend: &B,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        backend.try_inject_preconditioner_failure(PreconditionerFailureInjectionPoint::Launch)?;
        self.a.ensure_valid()?;
        let n = self.a.matrix().m;
        if r.len() != n || z.len() != n {
            return Err(crate::BackendError::InvalidArgument {
                operation: "polynomial_jacobi_apply",
                reason: format!(
                    "expected input/output length {n}, got {}/{}",
                    r.len(),
                    z.len()
                ),
            });
        }
        let mut workspace =
            self.workspace
                .lock()
                .map_err(|_| crate::BackendError::OperationFailed {
                    operation: "polynomial_jacobi_apply",
                    reason: "workspace mutex poisoned".into(),
                })?;
        backend.try_inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::WorkspaceAllocation,
        )?;
        ensure_buffer_exact(backend, &mut workspace.result, n)?;
        ensure_buffer_exact(backend, &mut workspace.rk, n)?;
        ensure_buffer_exact(backend, &mut workspace.dk, n)?;
        self.diag_inv.try_apply(backend, r, &mut workspace.result)?;
        backend.try_scale::<T>(self.damping, &mut workspace.result)?;
        if self.degree == 1 {
            return backend.try_copy_d2d(&workspace.result, z);
        }
        for _ in 1..self.degree {
            let PolynomialWorkspace { result, rk, dk } = &mut *workspace;
            self.a.try_apply(backend, result, rk)?;
            backend.try_scale::<T>(-T::one(), rk)?;
            backend.try_axpy::<T>(T::one(), r, rk)?;
            self.diag_inv.try_apply(backend, rk, dk)?;
            backend.try_axpy::<T>(self.damping, dk, result)?;
        }
        backend.try_copy_d2d(&workspace.result, z)
    }

    fn apply_colmajor(&self, backend: &B, rhs: usize, r: &B::Buffer<T>, z: &mut B::Buffer<T>) {
        if let Err(error) = self.try_apply_colmajor(backend, rhs, r, z) {
            log::error!(
                target: "sonium_compute::preconditioner",
                "polynomial-Jacobi multi-RHS application failed: {error}"
            );
        }
    }

    fn try_apply_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        let result = self.try_apply_colmajor_untracked(backend, rhs, r, z);
        match &result {
            Ok(()) => self.runtime_diagnostics.record_apply(),
            Err(error) => self.runtime_diagnostics.record_failure(error),
        }
        result
    }

    fn try_apply_colmajor_untracked(
        &self,
        backend: &B,
        rhs: usize,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        backend.try_inject_preconditioner_failure(PreconditionerFailureInjectionPoint::Launch)?;
        self.a.ensure_valid()?;
        let n = self.a.matrix().m;
        let len = n
            .checked_mul(rhs)
            .ok_or(crate::BackendError::InvalidArgument {
                operation: "polynomial_jacobi_apply_colmajor",
                reason: format!("workspace length overflow: n={n}, rhs={rhs}"),
            })?;
        if r.len() != len || z.len() != len {
            return Err(crate::BackendError::InvalidArgument {
                operation: "polynomial_jacobi_apply_colmajor",
                reason: format!(
                    "expected input/output length {len}, got {}/{}",
                    r.len(),
                    z.len()
                ),
            });
        }
        if rhs == 1 {
            return self.try_apply_untracked(backend, r, z);
        }
        let mut workspace =
            self.block_workspace
                .lock()
                .map_err(|_| crate::BackendError::OperationFailed {
                    operation: "polynomial_jacobi_apply_colmajor",
                    reason: "workspace mutex poisoned".into(),
                })?;
        backend.try_inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::WorkspaceAllocation,
        )?;
        ensure_buffer_exact(backend, &mut workspace.result, len)?;
        ensure_buffer_exact(backend, &mut workspace.rk, len)?;
        ensure_buffer_exact(backend, &mut workspace.dk, len)?;
        self.diag_inv
            .try_apply_colmajor(backend, rhs, r, &mut workspace.result)?;
        backend.try_scale::<T>(self.damping, &mut workspace.result)?;
        if self.degree == 1 {
            return backend.try_copy_d2d(&workspace.result, z);
        }
        for _ in 1..self.degree {
            let PolynomialWorkspace { result, rk, dk } = &mut *workspace;
            self.a
                .try_apply_colmajor(backend, rhs, result, rk)?;
            backend.try_scale::<T>(-T::one(), rk)?;
            backend.try_axpy::<T>(T::one(), r, rk)?;
            self.diag_inv.try_apply_colmajor(backend, rhs, rk, dk)?;
            backend.try_axpy::<T>(self.damping, dk, result)?;
        }
        backend.try_copy_d2d(&workspace.result, z)
    }

    fn diagnostics(&self) -> crate::DevicePreconditionerDiagnosticsSnapshot {
        self.runtime_diagnostics.snapshot()
    }

    fn try_apply_transpose(
        &self,
        _backend: &B,
        _r: &B::Buffer<T>,
        _z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        let error = crate::BackendError::Unsupported {
            backend: "device",
            operation: "polynomial_jacobi_transpose",
            scalar: "device scalar",
        };
        self.runtime_diagnostics.record_transpose_failure(&error);
        Err(error)
    }

    fn try_apply_hermitian(
        &self,
        _backend: &B,
        _r: &B::Buffer<T>,
        _z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        let error = crate::BackendError::Unsupported {
            backend: "device",
            operation: "polynomial_jacobi_hermitian",
            scalar: "device scalar",
        };
        self.runtime_diagnostics.record_hermitian_failure(&error);
        Err(error)
    }

    fn try_apply_transpose_colmajor(
        &self,
        _backend: &B,
        _rhs: usize,
        _r: &B::Buffer<T>,
        _z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        let error = crate::BackendError::Unsupported {
            backend: "device",
            operation: "polynomial_jacobi_transpose_colmajor",
            scalar: "device scalar",
        };
        self.runtime_diagnostics.record_transpose_failure(&error);
        Err(error)
    }

    fn try_apply_hermitian_colmajor(
        &self,
        _backend: &B,
        _rhs: usize,
        _r: &B::Buffer<T>,
        _z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        let error = crate::BackendError::Unsupported {
            backend: "device",
            operation: "polynomial_jacobi_hermitian_colmajor",
            scalar: "device scalar",
        };
        self.runtime_diagnostics.record_hermitian_failure(&error);
        Err(error)
    }
}

#[cfg(test)]
// Fixtures intentionally use the panicking convenience constructors on
// tiny fixed matrices; production paths must use the `try_*` variants.
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::cpu::CpuBackend;
    use crate::gmres::{
        BackendCsr, GmresBackendConfig, IdentityBackendPreconditioner, gmres_backend_preconditioned,
    };
    use crate::scalar::BufferElement;

    fn diag_dominant_5x5() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
        let row_ptr = vec![0, 2, 5, 8, 11, 13];
        let col_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4];
        let values = vec![
            10.0, -1.0, // row 0
            -1.0, 20.0, -1.0, // row 1
            -1.0, 30.0, -1.0, // row 2
            -1.0, 40.0, -1.0, // row 3
            -1.0, 50.0, // row 4
        ];
        (row_ptr, col_idx, values)
    }

    fn make_buf<T>(backend: &CpuBackend, host: &[T]) -> crate::cpu::HostBuffer<T>
    where
        T: BufferElement,
    {
        let mut b = backend.raw_alloc::<T>(host.len()).unwrap();
        backend.copy_h2d(host, &mut b);
        b
    }

    #[test]
    fn point_jacobi_extracts_diagonal_correctly() {
        let backend = CpuBackend::new();
        let (row_ptr, col_idx, values) = diag_dominant_5x5();
        let m = PointJacobiBackend::from_csr_host(&backend, 5, &row_ptr, &col_idx, &values);

        let mut diag_inv = vec![0.0_f64; 5];
        backend.copy_d2h(&m.diag_inv.matrix().values, &mut diag_inv);
        let expected = [0.1, 1.0 / 20.0, 1.0 / 30.0, 1.0 / 40.0, 1.0 / 50.0];
        for (got, want) in diag_inv.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1.0e-12, "got {got} want {want}");
        }
    }

    #[test]
    fn point_jacobi_apply_divides_by_diagonal() {
        let backend = CpuBackend::new();
        let (row_ptr, col_idx, values) = diag_dominant_5x5();
        let m = PointJacobiBackend::from_csr_host(&backend, 5, &row_ptr, &col_idx, &values);
        let r = make_buf(&backend, &[10.0_f64, 20.0, 30.0, 40.0, 50.0]);
        let mut z = backend.raw_alloc::<f64>(5).unwrap();

        m.apply(&backend, &r, &mut z);

        let mut z_host = vec![0.0_f64; 5];
        backend.copy_d2h(&z, &mut z_host);
        for v in &z_host {
            assert!((v - 1.0).abs() < 1.0e-12, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn point_jacobi_apply_colmajor_divides_each_column_by_diagonal() {
        let backend = CpuBackend::new();
        let (row_ptr, col_idx, values) = diag_dominant_5x5();
        let m = PointJacobiBackend::from_csr_host(&backend, 5, &row_ptr, &col_idx, &values);
        let r = make_buf(
            &backend,
            &[
                10.0_f64, 20.0, 30.0, 40.0, 50.0, -10.0, -20.0, -30.0, -40.0, -50.0,
            ],
        );
        let mut z = backend.raw_alloc::<f64>(10).unwrap();

        m.apply_colmajor(&backend, 2, &r, &mut z);

        let mut z_host = vec![0.0_f64; 10];
        backend.copy_d2h(&z, &mut z_host);
        let expected = [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0];
        for (got, want) in z_host.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1.0e-12, "got {got} want {want}");
        }
    }

    #[test]
    fn checked_device_actions_reuse_resident_storage_and_gate_unsupported_adjoints() {
        let backend = CpuBackend::new();
        let (row_ptr, col_idx, values) = diag_dominant_5x5();
        let a = BackendCsr::from_host(&backend, 5, 5, &row_ptr, &col_idx, &values);
        let point = PointJacobiBackend::from_csr_host(&backend, 5, &row_ptr, &col_idx, &values);
        let polynomial = PolynomialJacobiBackend::from_csr_host(
            &backend, &a, &row_ptr, &col_idx, &values, 2, 0.75,
        );
        let chebyshev =
            ChebyshevBackend::from_csr_host(&backend, &a, &row_ptr, &col_idx, &values, 2, 30.0);
        let r = make_buf(&backend, &[1.0_f64, 2.0, 3.0, 4.0, 5.0]);
        let mut z = backend.raw_alloc::<f64>(5).unwrap();
        let r_block = make_buf(
            &backend,
            &[1.0_f64, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, -3.0, -4.0, -5.0],
        );
        let mut z_block = backend.raw_alloc::<f64>(10).unwrap();

        // Construction may upload matrices and allocate workspaces. Once the
        // objects are built, these checked actions use the persistent device
        // handles and setup-owned workspaces.
        point.try_apply(&backend, &r, &mut z).unwrap();
        point
            .try_apply_colmajor(&backend, 2, &r_block, &mut z_block)
            .unwrap();
        polynomial.try_apply(&backend, &r, &mut z).unwrap();
        polynomial
            .try_apply_colmajor(&backend, 2, &r_block, &mut z_block)
            .unwrap();
        chebyshev.try_apply(&backend, &r, &mut z).unwrap();
        chebyshev
            .try_apply_colmajor(&backend, 2, &r_block, &mut z_block)
            .unwrap();
        assert_eq!(point.diagnostics().apply_count, 2);
        assert_eq!(polynomial.diagnostics().apply_count, 2);
        assert_eq!(chebyshev.diagnostics().apply_count, 2);
        assert_eq!(point.diagnostics().setup_count, 1);
        assert_eq!(polynomial.diagnostics().setup_count, 1);
        assert_eq!(chebyshev.diagnostics().setup_count, 1);

        assert_eq!(
            polynomial.hermitian_capability(),
            DeviceAdjointCapability::Unsupported
        );
        let error = polynomial
            .try_apply_hermitian(&backend, &r, &mut z)
            .expect_err("non-Hermitian polynomial action must be rejected");
        assert!(matches!(error, crate::BackendError::Unsupported { .. }));
        assert_eq!(polynomial.diagnostics().hermitian_failures, 1);
        assert_eq!(
            polynomial.diagnostics().first_failure_stage,
            Some(crate::DevicePreconditionerFailureStage::Unknown)
        );

        let identity = IdentityBackendPreconditioner;
        assert_eq!(
            <IdentityBackendPreconditioner as DevicePreconditioner<CpuBackend, f64>>::
                transpose_capability(&identity),
            DeviceAdjointCapability::Exact
        );
        assert_eq!(
            <IdentityBackendPreconditioner as DevicePreconditioner<CpuBackend, f64>>::
                hermitian_capability(&identity),
            DeviceAdjointCapability::Exact
        );
    }

    #[test]
    fn injected_checked_failures_are_typed_transactional_and_retryable() {
        let backend = CpuBackend::new();
        let (row_ptr, col_idx, values) = diag_dominant_5x5();
        let csr = BackendCsr::from_host(&backend, 5, 5, &row_ptr, &col_idx, &values);

        CpuBackend::inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::HierarchyUpload,
        );
        let error = match PolynomialJacobiBackend::try_new(&backend, &csr, &[10.0; 5], 2, 0.75) {
            Err(error) => error,
            Ok(_) => panic!("hierarchy upload fault must cross the checked setup boundary"),
        };
        assert!(matches!(error, crate::BackendError::Transfer { .. }));

        CpuBackend::inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::DiagonalUpload,
        );
        let error =
            match ChebyshevBackend::try_with_lambda_max(&backend, &csr, &[10.0; 5], 2, 30.0, 2.0) {
                Err(error) => error,
                Ok(_) => panic!("diagonal upload fault must cross the checked setup boundary"),
            };
        assert!(matches!(error, crate::BackendError::Transfer { .. }));

        CpuBackend::inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::WorkspaceAllocation,
        );
        let error = match PolynomialJacobiBackend::try_new(&backend, &csr, &[10.0; 5], 2, 0.75) {
            Err(error) => error,
            Ok(_) => panic!("workspace fault must cross the checked setup boundary"),
        };
        assert!(matches!(error, crate::BackendError::Allocation { .. }));

        let polynomial = PolynomialJacobiBackend::try_new(&backend, &csr, &[10.0; 5], 2, 0.75)
            .expect("the one-shot setup faults must not poison a retry");
        let r = make_buf(&backend, &[1.0_f64, 2.0, 3.0, 4.0, 5.0]);
        let mut z = make_buf(&backend, &[91.0_f64; 5]);

        CpuBackend::inject_preconditioner_failure(PreconditionerFailureInjectionPoint::Launch);
        let error = polynomial
            .try_apply(&backend, &r, &mut z)
            .expect_err("launch fault must fail before device work");
        assert!(matches!(error, crate::BackendError::Launch { .. }));
        let mut z_host = [0.0_f64; 5];
        backend.copy_d2h(&z, &mut z_host);
        assert_eq!(z_host, [91.0; 5], "failed apply must not publish output");
        let diagnostics = polynomial.diagnostics();
        assert_eq!(diagnostics.apply_failures, 1);
        assert_eq!(
            diagnostics.first_failure_stage,
            Some(crate::DevicePreconditionerFailureStage::Launch)
        );

        CpuBackend::inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::NonFiniteGeneration,
        );
        let error = polynomial
            .try_apply(&backend, &r, &mut z)
            .expect_err("non-finite generation must be reported before publication");
        assert!(matches!(error, crate::BackendError::InvalidArgument { .. }));
        backend.copy_d2h(&z, &mut z_host);
        assert_eq!(z_host, [91.0; 5], "non-finite output must not be published");

        let nonfinite_polynomial = PolynomialJacobiBackend::try_new(
            &backend, &csr, &[10.0; 5], 2, 0.75,
        )
        .expect("a fresh preconditioner should provide an independent diagnostic first stage");
        assert_eq!(nonfinite_polynomial.diagnostics().first_failure_stage, None);
        let mut nonfinite_output = make_buf(&backend, &[91.0_f64; 5]);
        CpuBackend::inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::NonFiniteGeneration,
        );
        let error = nonfinite_polynomial
            .try_apply(&backend, &r, &mut nonfinite_output)
            .expect_err("non-finite generation must be typed at the first failing stage");
        assert!(matches!(error, crate::BackendError::InvalidArgument { .. }));
        assert_eq!(
            nonfinite_polynomial.diagnostics().first_failure_stage,
            Some(crate::DevicePreconditionerFailureStage::NonFiniteGeneration)
        );

        let device_loss_polynomial =
            PolynomialJacobiBackend::try_new(&backend, &csr, &[10.0; 5], 2, 0.75)
                .expect("device-loss fault must not affect a fresh CPU context");
        let mut device_loss_output = make_buf(&backend, &[91.0_f64; 5]);
        CpuBackend::inject_preconditioner_failure(PreconditionerFailureInjectionPoint::DeviceLoss);
        let error = device_loss_polynomial
            .try_apply(&backend, &r, &mut device_loss_output)
            .expect_err("device loss must be surfaced as a typed failure");
        assert!(matches!(error, crate::BackendError::DeviceLost { .. }));
        assert_eq!(
            device_loss_polynomial.diagnostics().first_failure_stage,
            Some(crate::DevicePreconditionerFailureStage::DeviceLoss)
        );

        polynomial
            .try_apply(&backend, &r, &mut z)
            .expect("retry after a checked failure must succeed");
        backend.copy_d2h(&z, &mut z_host);
        assert_ne!(z_host, [91.0; 5]);

        CpuBackend::inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::Synchronization,
        );
        let error = backend
            .try_synchronize()
            .expect_err("synchronization fault must be surfaced");
        assert!(matches!(error, crate::BackendError::Synchronization { .. }));
        backend
            .try_synchronize()
            .expect("synchronization retry must observe a reusable backend");

        CpuBackend::clear_preconditioner_failure();
    }

    #[test]
    fn point_jacobi_preconditioned_gmres_converges_no_slower_than_identity() {
        let backend = CpuBackend::new();
        let (row_ptr, col_idx, values) = diag_dominant_5x5();
        let a = BackendCsr::from_host(&backend, 5, 5, &row_ptr, &col_idx, &values);
        let b = make_buf::<f64>(&backend, &[1.0, 2.0, 3.0, 4.0, 5.0]);

        let mut out_id = backend.raw_alloc::<f64>(5).unwrap();
        let status_id = gmres_backend_preconditioned(
            &backend,
            &a,
            &IdentityBackendPreconditioner,
            &b,
            None,
            &mut out_id,
            &GmresBackendConfig {
                max_iterations: 100,
                tolerance: 1.0e-12,
            },
        )
        .unwrap();

        let mut out_jac = backend.raw_alloc::<f64>(5).unwrap();
        let pj = PointJacobiBackend::from_csr_host(&backend, 5, &row_ptr, &col_idx, &values);
        let status_jac = gmres_backend_preconditioned(
            &backend,
            &a,
            &pj,
            &b,
            None,
            &mut out_jac,
            &GmresBackendConfig {
                max_iterations: 100,
                tolerance: 1.0e-12,
            },
        )
        .unwrap();

        assert!(status_id.converged && status_jac.converged);
        assert!(
            status_jac.iterations <= status_id.iterations,
            "jacobi {} > identity {}",
            status_jac.iterations,
            status_id.iterations
        );
    }

    #[test]
    fn degree_one_unit_damping_matches_point_jacobi() {
        let backend = CpuBackend::new();
        let (rp, ci, va) = diag_dominant_5x5();
        let csr = BackendCsr::from_host(&backend, 5, 5, &rp, &ci, &va);

        let pj = PointJacobiBackend::from_csr_host(&backend, 5, &rp, &ci, &va);
        let poly = PolynomialJacobiBackend::from_csr_host(&backend, &csr, &rp, &ci, &va, 1, 1.0);

        let r = make_buf(&backend, &[3.0_f64, -7.0, 11.0, 13.0, -2.0]);
        let mut z_point = backend.raw_alloc::<f64>(5).unwrap();
        let mut z_poly = backend.raw_alloc::<f64>(5).unwrap();
        pj.apply(&backend, &r, &mut z_point);
        poly.apply(&backend, &r, &mut z_poly);
        let _ = csr;

        let mut h_point = vec![0.0_f64; 5];
        let mut h_poly = vec![0.0_f64; 5];
        backend.copy_d2h(&z_point, &mut h_point);
        backend.copy_d2h(&z_poly, &mut h_poly);
        for (i, (a, b)) in h_point.iter().zip(h_poly.iter()).enumerate() {
            assert!((a - b).abs() < 1.0e-15, "row {i}: point {a} vs poly {b}");
        }
    }

    #[test]
    fn degree_k_on_identity_returns_input() {
        let backend = CpuBackend::new();
        let rp: Vec<i32> = (0..=5).collect();
        let ci: Vec<i32> = (0..5).collect();
        let va: Vec<f64> = vec![1.0; 5];
        let csr = BackendCsr::from_host(&backend, 5, 5, &rp, &ci, &va);

        for &k in &[1_usize, 2, 4, 8] {
            let poly =
                PolynomialJacobiBackend::from_csr_host(&backend, &csr, &rp, &ci, &va, k, 1.0);
            let r = make_buf(&backend, &[1.5_f64, -2.5, 3.0, 0.0, 4.25]);
            let mut z = backend.raw_alloc::<f64>(5).unwrap();
            poly.apply(&backend, &r, &mut z);
            let mut h = vec![0.0_f64; 5];
            backend.copy_d2h(&z, &mut h);
            for (got, want) in h.iter().zip(&[1.5_f64, -2.5, 3.0, 0.0, 4.25]) {
                assert!((got - want).abs() < 1.0e-15, "k={k}: got {got} want {want}");
            }
        }
    }

    #[test]
    fn polynomial_jacobi_apply_colmajor_matches_column_apply() {
        let backend = CpuBackend::new();
        let (row_ptr, col_idx, values) = diag_dominant_5x5();
        let a = BackendCsr::from_host(&backend, 5, 5, &row_ptr, &col_idx, &values);
        let pc = PolynomialJacobiBackend::from_csr_host(
            &backend, &a, &row_ptr, &col_idx, &values, 2, 0.75,
        );
        let r_host = [1.0_f64, 2.0, 3.0, 4.0, 5.0, -2.0, 1.0, -1.0, 3.0, -4.0];
        let r = make_buf(&backend, &r_host);
        let mut z_multi = backend.raw_alloc::<f64>(10).unwrap();

        pc.apply_colmajor(&backend, 2, &r, &mut z_multi);

        let mut expected = [0.0_f64; 10];
        for col in 0..2 {
            let range = col * 5..(col + 1) * 5;
            let r_col = make_buf(&backend, &r_host[range.clone()]);
            let mut z_col = backend.raw_alloc::<f64>(5).unwrap();
            pc.apply(&backend, &r_col, &mut z_col);
            backend.copy_d2h(&z_col, &mut expected[range]);
        }

        // A block workspace may be reused with a different RHS count.  The
        // resident sparse action requires the active dense layout exactly,
        // so both growth and shrinkage must remain valid.
        let r_three_host = [
            1.0_f64, 2.0, 3.0, 4.0, 5.0, -2.0, 1.0, -1.0, 3.0, -4.0, 0.5, 1.5, -2.5, 3.5, -4.5,
        ];
        let r_three = make_buf(&backend, &r_three_host);
        let mut z_three = backend.raw_alloc::<f64>(15).unwrap();
        pc.try_apply_colmajor(&backend, 3, &r_three, &mut z_three)
            .unwrap();
        pc.try_apply_colmajor(&backend, 2, &r, &mut z_multi)
            .unwrap();

        let mut actual = vec![0.0_f64; 10];
        backend.copy_d2h(&z_multi, &mut actual);
        for (got, want) in actual.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1.0e-12, "got {got} want {want}");
        }
    }

    #[test]
    fn higher_degree_polynomial_converges_no_worse_than_degree_one() {
        let backend = CpuBackend::new();
        let (rp, ci, va) = diag_dominant_5x5();
        let a = BackendCsr::from_host(&backend, 5, 5, &rp, &ci, &va);
        let b = make_buf::<f64>(&backend, &[1.0, 2.0, 3.0, 4.0, 5.0]);

        let mut x_id = backend.raw_alloc::<f64>(5).unwrap();
        let s_id = gmres_backend_preconditioned(
            &backend,
            &a,
            &IdentityBackendPreconditioner,
            &b,
            None,
            &mut x_id,
            &GmresBackendConfig {
                max_iterations: 100,
                tolerance: 1.0e-12,
            },
        )
        .unwrap();

        let poly1 = PolynomialJacobiBackend::from_csr_host(&backend, &a, &rp, &ci, &va, 1, 1.0);
        let mut x_p1 = backend.raw_alloc::<f64>(5).unwrap();
        let s_p1 = gmres_backend_preconditioned(
            &backend,
            &a,
            &poly1,
            &b,
            None,
            &mut x_p1,
            &GmresBackendConfig {
                max_iterations: 100,
                tolerance: 1.0e-12,
            },
        )
        .unwrap();

        let poly3 = PolynomialJacobiBackend::from_csr_host(&backend, &a, &rp, &ci, &va, 3, 1.0);
        let mut x_p3 = backend.raw_alloc::<f64>(5).unwrap();
        let s_p3 = gmres_backend_preconditioned(
            &backend,
            &a,
            &poly3,
            &b,
            None,
            &mut x_p3,
            &GmresBackendConfig {
                max_iterations: 100,
                tolerance: 1.0e-12,
            },
        )
        .unwrap();

        assert!(s_id.converged && s_p1.converged && s_p3.converged);

        let mut h_id = vec![0.0_f64; 5];
        let mut h_p1 = vec![0.0_f64; 5];
        let mut h_p3 = vec![0.0_f64; 5];
        backend.copy_d2h(&x_id, &mut h_id);
        backend.copy_d2h(&x_p1, &mut h_p1);
        backend.copy_d2h(&x_p3, &mut h_p3);
        for i in 0..5 {
            assert!(
                (h_id[i] - h_p1[i]).abs() < 1.0e-9,
                "row {i}: id {} vs poly1 {}",
                h_id[i],
                h_p1[i]
            );
            assert!(
                (h_id[i] - h_p3[i]).abs() < 1.0e-9,
                "row {i}: id {} vs poly3 {}",
                h_id[i],
                h_p3[i]
            );
        }

        assert!(
            s_p1.iterations <= s_id.iterations,
            "poly-1 {} > identity {}",
            s_p1.iterations,
            s_id.iterations
        );
        assert!(
            s_p3.iterations <= s_id.iterations,
            "poly-3 {} > identity {}",
            s_p3.iterations,
            s_id.iterations
        );
    }

    #[test]
    fn checked_polynomial_constructor_rejects_invalid_degree_and_csr() {
        let backend = CpuBackend::new();
        let row_ptr = vec![0, 1, 2, 3];
        let col_idx = vec![0, 1, 2];
        let values = vec![1.0_f64; 3];
        let a = BackendCsr::from_host(&backend, 3, 3, &row_ptr, &col_idx, &values);

        let zero_degree = PolynomialJacobiBackend::try_new(&backend, &a, &values, 0, 1.0);
        assert!(matches!(
            zero_degree,
            Err(crate::BackendError::InvalidArgument { .. })
        ));

        let malformed_row_ptr = vec![0, 2, 1, 3];
        let malformed = PolynomialJacobiBackend::try_from_csr_host(
            &backend,
            &a,
            &malformed_row_ptr,
            &col_idx,
            &values,
            1,
            1.0,
        );
        assert!(matches!(
            malformed,
            Err(crate::BackendError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn checked_polynomial_apply_rejects_wrong_length() {
        let backend = CpuBackend::new();
        let (row_ptr, col_idx, values) = diag_dominant_5x5();
        let a = BackendCsr::from_host(&backend, 5, 5, &row_ptr, &col_idx, &values);
        let poly = PolynomialJacobiBackend::from_csr_host(
            &backend, &a, &row_ptr, &col_idx, &values, 2, 0.75,
        );
        let r = make_buf(&backend, &[1.0_f64; 4]);
        let mut z = backend.raw_alloc::<f64>(5).unwrap();

        let result = poly.try_apply(&backend, &r, &mut z);
        assert!(matches!(
            result,
            Err(crate::BackendError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn legacy_polynomial_apply_is_panic_free_on_wrong_length() {
        let backend = CpuBackend::new();
        let (row_ptr, col_idx, values) = diag_dominant_5x5();
        let a = BackendCsr::from_host(&backend, 5, 5, &row_ptr, &col_idx, &values);
        let poly = PolynomialJacobiBackend::from_csr_host(
            &backend, &a, &row_ptr, &col_idx, &values, 3, 0.75,
        );
        let r = make_buf(&backend, &[1.0_f64; 4]);
        let mut z = backend.raw_alloc::<f64>(5).unwrap();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            poly.apply(&backend, &r, &mut z);
        }));
        assert!(result.is_ok(), "legacy polynomial apply must not panic");
    }

    #[test]
    fn legacy_point_jacobi_apply_colmajor_is_panic_free_on_wrong_length() {
        let backend = CpuBackend::new();
        let (row_ptr, col_idx, values) = diag_dominant_5x5();
        let point = PointJacobiBackend::from_csr_host(&backend, 5, &row_ptr, &col_idx, &values);
        let r = make_buf(&backend, &[1.0_f64; 4]);
        let mut z = backend.raw_alloc::<f64>(5).unwrap();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            point.apply_colmajor(&backend, 2, &r, &mut z);
        }));
        assert!(result.is_ok(), "legacy point-Jacobi apply must not panic");
    }

    #[test]
    fn checked_point_jacobi_rejects_non_finite_diagonal() {
        let backend = CpuBackend::new();
        let result = PointJacobiBackend::try_from_diagonal(&backend, &[1.0, f64::NAN]);
        assert!(matches!(
            result,
            Err(crate::BackendError::InvalidArgument { .. })
        ));

        let result = PointJacobiBackend::try_from_diagonal(&backend, &[1.0, 0.0]);
        assert!(matches!(
            result,
            Err(crate::BackendError::InvalidArgument { .. })
        ));
    }
}
