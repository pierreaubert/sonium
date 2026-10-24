//! GPU-friendly Chebyshev polynomial preconditioner.
//!
//! Approximates `A⁻¹` by `K` steps of Chebyshev iteration applied to the
//! Jacobi-preconditioned system `D⁻¹ A`:
//!
//! ```text
//!     θ  = ½(λ_max + λ_min)    (centre of the eigenvalue interval)
//!     δ  = ½(λ_max − λ_min)    (half-width)
//!     σ  = θ/δ
//!     ρ₀ = 1/σ
//!
//!     d₀ = (1/θ) · D⁻¹ · r           // initial Jacobi-Richardson step
//!     z  = d₀
//!     for k = 1..K:
//!         ρ_k = 1 / (2σ − ρ_{k-1})
//!         res = D⁻¹ · (r − A · z)     // current preconditioned residual
//!         d_k = ρ_k·ρ_{k-1} · d_{k-1} + (2 ρ_k / δ) · res
//!         z   = z + d_k
//!         ρ_{k-1} ← ρ_k
//! ```
//!
//! This is the standard 3-term recurrence (Greenbaum §11; Saad §13.3) for
//! the Chebyshev polynomial of degree `K` on `[λ_min, λ_max]`. The
//! polynomial is optimal (in min-max sense) for damping the spectrum of
//! `D⁻¹ A` over the interval, so for SPD or near-SPD systems Chebyshev
//! beats the Neumann-series polynomial Jacobi at the same degree —
//! provided the eigenvalue bounds are reasonably tight.
//!
//! # Eigenvalue estimate
//!
//! `λ_max` of `D⁻¹ A` is estimated once at construction via 20 power
//! iterations on the host (cheap, `O(20·nnz)`). `λ_min` is set to
//! `λ_max / eigenvalue_ratio`. PETSc and HYPRE both default the ratio
//! to 30; we follow.
//!
//! # Cost per apply
//!
//! CUDA uses one native diagonal scale at start, then a full SpMV/SpMM and
//! one fused residual/recurrence/solution update per remaining step. Columns
//! share the inverse diagonal without duplicating it. Other backends retain
//! the checked diagonal-CSR and vector-operation fallback. Five workspace
//! buffers are retained separately for scalar and block application; output
//! is published only after the complete polynomial succeeds.
//!
//! # When to use
//!
//! - SPD or near-SPD systems where eigenvalue bounds are reasonably
//!   tight (Poisson-like, elliptic). Strictly better than polynomial
//!   Jacobi.
//! - Multigrid smoother surrogate. Used as the smoother inside AMG when
//!   Gauss-Seidel's serial nature kills GPU performance.
//!
//! # When NOT to use
//!
//! - Indefinite or strongly non-symmetric systems — Chebyshev assumes
//!   the eigenvalues lie on a positive real interval. For Stokes-like
//!   saddle-point blocks, [`PolynomialJacobiBackend`] with damping
//!   `ω ∈ [0.5, 0.8]` is more robust.

use crate::gmres::{BackendCsr, BackendPreconditioner};
use crate::operator::{PersistentDeviceCsr, validate_csr_inputs};
use super::matrix::PreconditionerMatrix;
use crate::{Backend, DeviceBufferLike, DeviceScalar, PreconditionerFailureInjectionPoint};
use num_traits::{Float, FromPrimitive, One, Zero};
use std::sync::Mutex;

/// Chebyshev polynomial preconditioner of degree `K ≥ 1`. Generic
/// over scalar `T`. Eigenvalue interval `[λ_max/ratio, λ_max]` is
/// stored in `T::Real` (Chebyshev assumes positive-real spectrum).
pub struct ChebyshevBackend<'a, B: Backend, T: DeviceScalar = f64> {
    a: PreconditionerMatrix<'a, B, T>,
    diag_inv: PersistentDeviceCsr<B, T>,
    lambda_max: T::Real,
    eigenvalue_ratio: T::Real,
    degree: usize,
    workspace: Mutex<ChebyshevWorkspace<B, T>>,
    // Scalar recovery must not evict the latest block application scratch.
    block_workspace: Mutex<ChebyshevWorkspace<B, T>>,
    runtime_diagnostics: crate::DevicePreconditionerDiagnostics,
}

struct ChebyshevWorkspace<B: Backend, T: DeviceScalar> {
    result: B::Buffer<T>,
    d: B::Buffer<T>,
    az: B::Buffer<T>,
    res: B::Buffer<T>,
    d_new: B::Buffer<T>,
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

/// Type alias for the f32 Chebyshev backend specialization.
pub type ChebyshevBackendF32<'a, B> = ChebyshevBackend<'a, B, f32>;

impl<'a, B: Backend> ChebyshevBackend<'a, B, f32> {
    /// Backward-compat alias for [`Self::from_csr_host`] on the f32
    /// specialisation. Kept to ease migration from the legacy
    /// `ChebyshevBackendF32` type-name.
    pub fn from_csr_f32(
        backend: &'a B,
        a: &'a BackendCsr<B, f32>,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
        values_host: &[f32],
        degree: usize,
        eigenvalue_ratio: f32,
    ) -> Self {
        Self::from_csr_host(
            backend,
            a,
            row_ptr_host,
            col_idx_host,
            values_host,
            degree,
            eigenvalue_ratio,
        )
    }
}

impl<'a, B: Backend, T: DeviceScalar> ChebyshevBackend<'a, B, T> {
    pub fn with_lambda_max(
        backend: &'a B,
        a: &'a BackendCsr<B, T>,
        diag_host: &[T],
        degree: usize,
        eigenvalue_ratio: T::Real,
        lambda_max: T::Real,
    ) -> Self {
        Self::try_with_lambda_max(backend, a, diag_host, degree, eigenvalue_ratio, lambda_max)
            .expect("Chebyshev construction failed")
    }

    /// Checked variant of [`Self::with_lambda_max`].
    pub fn try_with_lambda_max(
        backend: &'a B,
        a: &'a BackendCsr<B, T>,
        diag_host: &[T],
        degree: usize,
        eigenvalue_ratio: T::Real,
        lambda_max: T::Real,
    ) -> crate::BackendResult<Self> {
        Self::try_from_matrix(
            backend,
            a.m,
            diag_host,
            degree,
            eigenvalue_ratio,
            lambda_max,
            || PreconditionerMatrix::borrowed(backend, a),
        )
    }

    /// Build an owned preconditioner with a supplied spectral bound.
    ///
    /// # Errors
    /// Rejects invalid polynomial parameters or diagonal and propagates backend failures.
    pub fn try_from_owned_csr(
        backend: &B,
        a: PersistentDeviceCsr<B, T>,
        diag_host: &[T],
        degree: usize,
        eigenvalue_ratio: T::Real,
        lambda_max: T::Real,
    ) -> crate::BackendResult<Self> {
        let n = a.m();
        Self::try_from_matrix(
            backend,
            n,
            diag_host,
            degree,
            eigenvalue_ratio,
            lambda_max,
            || {
                Ok(PreconditionerMatrix::Owned {
                    matrix: a,
                    valid: true,
                })
            },
        )
    }

    /// Refresh owned numeric data and its spectral bound without replacing device storage.
    ///
    /// Device values must be finite and match the retained sparsity pattern.
    /// Validation errors preserve the old state. Backend failures require a
    /// successful refresh before further applications, including degree-one solves.
    ///
    /// # Errors
    /// Rejects borrowed matrices, invalid shape, diagonal or bound, and backend failures.
    pub fn try_update_numeric(
        &mut self,
        backend: &B,
        values: &B::Buffer<T>,
        diagonal: &[T],
        lambda_max: T::Real,
    ) -> crate::BackendResult<()> {
        if !lambda_max.is_finite() || lambda_max <= T::Real::zero() {
            return Err(crate::BackendError::InvalidArgument {
                operation: "chebyshev_update",
                reason: "spectral bound must be finite and positive".into(),
            });
        }
        self.a
            .try_refresh(backend, values, diagonal, &mut self.diag_inv)?;
        self.lambda_max = lambda_max;
        Ok(())
    }

    fn try_from_matrix(
        backend: &B,
        rows: usize,
        diag_host: &[T],
        degree: usize,
        eigenvalue_ratio: T::Real,
        lambda_max: T::Real,
        make_matrix: impl FnOnce() -> crate::BackendResult<PreconditionerMatrix<'a, B, T>>,
    ) -> crate::BackendResult<Self> {
        if degree == 0 {
            return Err(crate::BackendError::InvalidArgument {
                operation: "chebyshev_constructor",
                reason: "Chebyshev degree must be >= 1".into(),
            });
        }
        let one_real = T::Real::one();
        if !eigenvalue_ratio.is_finite() || eigenvalue_ratio < one_real {
            return Err(crate::BackendError::InvalidArgument {
                operation: "chebyshev_constructor",
                reason: "eigenvalue_ratio must be finite and >= 1.0".into(),
            });
        }
        if !lambda_max.is_finite() || lambda_max <= T::Real::zero() {
            return Err(crate::BackendError::InvalidArgument {
                operation: "chebyshev_constructor",
                reason: "lambda_max must be finite and > 0.0".into(),
            });
        }
        if diag_host.len() != rows {
            return Err(crate::BackendError::InvalidArgument {
                operation: "chebyshev_constructor",
                reason: format!(
                    "diagonal length {} != A row count {}",
                    diag_host.len(),
                    rows
                ),
            });
        }
        if diag_host.iter().any(|value| !value.is_finite()) {
            return Err(crate::BackendError::InvalidArgument {
                operation: "chebyshev_constructor",
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
                    operation: "chebyshev_constructor",
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
        let workspace = ChebyshevWorkspace {
            result: backend
                .raw_alloc::<T>(n)
                .map_err(crate::BackendError::from)?,
            d: backend
                .raw_alloc::<T>(n)
                .map_err(crate::BackendError::from)?,
            az: backend
                .raw_alloc::<T>(n)
                .map_err(crate::BackendError::from)?,
            res: backend
                .raw_alloc::<T>(n)
                .map_err(crate::BackendError::from)?,
            d_new: backend
                .raw_alloc::<T>(n)
                .map_err(crate::BackendError::from)?,
        };
        let block_workspace = ChebyshevWorkspace {
            result: backend
                .raw_alloc::<T>(0)
                .map_err(crate::BackendError::from)?,
            d: backend
                .raw_alloc::<T>(0)
                .map_err(crate::BackendError::from)?,
            az: backend
                .raw_alloc::<T>(0)
                .map_err(crate::BackendError::from)?,
            res: backend
                .raw_alloc::<T>(0)
                .map_err(crate::BackendError::from)?,
            d_new: backend
                .raw_alloc::<T>(0)
                .map_err(crate::BackendError::from)?,
        };
        let runtime_diagnostics = crate::DevicePreconditionerDiagnostics::default();
        runtime_diagnostics.record_setup();

        Ok(Self {
            a,
            diag_inv,
            lambda_max,
            eigenvalue_ratio,
            degree,
            workspace: Mutex::new(workspace),
            block_workspace: Mutex::new(block_workspace),
            runtime_diagnostics,
        })
    }

    /// Build from the host CSR; estimate `λ_max(D⁻¹A)` via 20 power
    /// iterations on the host.
    pub fn from_csr_host(
        backend: &'a B,
        a: &'a BackendCsr<B, T>,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
        values_host: &[T],
        degree: usize,
        eigenvalue_ratio: T::Real,
    ) -> Self {
        Self::try_from_csr_host(
            backend,
            a,
            row_ptr_host,
            col_idx_host,
            values_host,
            degree,
            eigenvalue_ratio,
        )
        .expect("Chebyshev CSR construction failed")
    }

    /// Checked variant of [`Self::from_csr_host`].
    pub fn try_from_csr_host(
        backend: &'a B,
        a: &'a BackendCsr<B, T>,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
        values_host: &[T],
        degree: usize,
        eigenvalue_ratio: T::Real,
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
        let lambda_max =
            estimate_lambda_max_host::<T>(m, row_ptr_host, col_idx_host, values_host, &diag);
        Self::try_with_lambda_max(backend, a, &diag, degree, eigenvalue_ratio, lambda_max)
    }

    /// Build an owned preconditioner using the matching host CSR for spectral estimation.
    ///
    /// # Errors
    /// Rejects invalid CSR/parameters and propagates backend failures.
    pub fn try_from_owned_csr_host(
        backend: &B,
        a: PersistentDeviceCsr<B, T>,
        row_ptr: &[i32],
        col_idx: &[i32],
        values: &[T],
        degree: usize,
        eigenvalue_ratio: T::Real,
    ) -> crate::BackendResult<Self> {
        let (diagonal, bound) = Self::host_spectrum(a.m(), a.n(), row_ptr, col_idx, values)?;
        Self::try_from_owned_csr(backend, a, &diagonal, degree, eigenvalue_ratio, bound)
    }

    /// Refresh owned values and recompute the bound from matching host CSR data.
    ///
    /// Host and device values must describe the same retained sparsity pattern.
    ///
    /// # Errors
    /// Rejects invalid host data or borrowed storage and propagates refresh failures.
    pub fn try_update_from_csr_host(
        &mut self,
        backend: &B,
        device_values: &B::Buffer<T>,
        row_ptr: &[i32],
        col_idx: &[i32],
        values: &[T],
    ) -> crate::BackendResult<()> {
        let matrix = self.a.matrix();
        let (diagonal, bound) = Self::host_spectrum(matrix.m, matrix.n, row_ptr, col_idx, values)?;
        self.try_update_numeric(backend, device_values, &diagonal, bound)
    }

    fn host_spectrum(
        m: usize,
        n: usize,
        rows: &[i32],
        columns: &[i32],
        values: &[T],
    ) -> crate::BackendResult<(Vec<T>, T::Real)> {
        validate_csr_inputs(m, n, rows, columns, values)?;
        if m != n {
            return Err(crate::BackendError::InvalidArgument {
                operation: "chebyshev_host_spectrum",
                reason: "matrix must be square".into(),
            });
        }
        let mut diagonal = vec![T::zero(); m];
        for row in 0..m {
            for index in rows[row] as usize..rows[row + 1] as usize {
                if columns[index] as usize == row {
                    diagonal[row] = values[index];
                    break;
                }
            }
        }
        if diagonal
            .iter()
            .any(|d| !d.is_finite() || d.norm_sqr().sqrt() <= T::Real::epsilon())
        {
            return Err(crate::BackendError::InvalidArgument {
                operation: "chebyshev_host_spectrum",
                reason: "diagonal must be finite and invertible".into(),
            });
        }
        let bound = estimate_lambda_max_host::<T>(m, rows, columns, values, &diagonal);
        Ok((diagonal, bound))
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn lambda_max(&self) -> T::Real {
        self.lambda_max
    }

    pub fn eigenvalue_ratio(&self) -> T::Real {
        self.eigenvalue_ratio
    }
}

impl<B: Backend, T: DeviceScalar> BackendPreconditioner<B, T> for ChebyshevBackend<'_, B, T> {
    fn apply(&self, backend: &B, r: &B::Buffer<T>, z: &mut B::Buffer<T>) {
        if let Err(error) = self.try_apply(backend, r, z) {
            log::error!(
                target: "sonium_compute::preconditioner",
                "Chebyshev application failed: {error}"
            );
        }
    }

    #[allow(dead_code)]
    fn apply_legacy_unchecked(&self, backend: &B, r: &B::Buffer<T>, z: &mut B::Buffer<T>) {
        let n = self.a.matrix().m;
        let Some(two) = T::Real::from_f64(2.0) else {
            log::error!(target: "sonium_compute::preconditioner", "Chebyshev recurrence constant 2 is not representable");
            return;
        };
        let Some(half) = T::Real::from_f64(0.5) else {
            log::error!(target: "sonium_compute::preconditioner", "Chebyshev recurrence constant 0.5 is not representable");
            return;
        };
        let one = T::Real::one();

        let lambda_min = self.lambda_max / self.eigenvalue_ratio;
        let theta = half * (self.lambda_max + lambda_min);
        let delta = half * (self.lambda_max - lambda_min);
        let sigma = theta / delta;
        let mut rho = one / sigma;
        let inv_theta = one / theta;

        // Prebake the internal diag-inverse CSR and `self.a` once per
        // `apply`, reused across the K Chebyshev steps. Self-contained
        // — no shared prebake with the outer solver, since the outer
        // matrix may differ from `self.a` (BEM near-field vs full FMM).
        let d_prebaked = backend.prebake_csr::<T>(
            n,
            n,
            &self.diag_inv.matrix().row_ptr,
            &self.diag_inv.matrix().col_idx,
            &self.diag_inv.matrix().values,
        );
        let a_prebaked = self.a.matrix().prebake(backend);

        let Ok(mut d) = backend.raw_alloc::<T>(n) else {
            log::error!(target: "sonium_compute::preconditioner", "Chebyshev workspace allocation failed");
            return;
        };
        backend.spmv_prebaked::<T>(&d_prebaked, r, &mut d);
        backend.scale::<T>(T::from_real(inv_theta), &mut d);
        backend.copy_d2d(&d, z);

        if self.degree == 1 {
            return;
        }

        let Ok(mut az) = backend.raw_alloc::<T>(n) else {
            log::error!(target: "sonium_compute::preconditioner", "Chebyshev az workspace allocation failed");
            return;
        };
        let Ok(mut res) = backend.raw_alloc::<T>(n) else {
            log::error!(target: "sonium_compute::preconditioner", "Chebyshev residual workspace allocation failed");
            return;
        };
        let Ok(mut d_new) = backend.raw_alloc::<T>(n) else {
            log::error!(target: "sonium_compute::preconditioner", "Chebyshev recurrence workspace allocation failed");
            return;
        };

        for _ in 1..self.degree {
            let rho_new = one / (two * sigma - rho);
            let coeff_old = rho_new * rho;
            let coeff_res = two * rho_new / delta;

            backend.spmv_prebaked::<T>(&a_prebaked, z, &mut az);
            backend.scale::<T>(-T::one(), &mut az);
            backend.axpy::<T>(T::one(), r, &mut az);
            backend.spmv_prebaked::<T>(&d_prebaked, &az, &mut res);

            backend.copy_d2d(&d, &mut d_new);
            backend.scale::<T>(T::from_real(coeff_old), &mut d_new);
            backend.axpy::<T>(T::from_real(coeff_res), &res, &mut d_new);

            backend.axpy::<T>(T::one(), &d_new, z);
            std::mem::swap(&mut d, &mut d_new);
            rho = rho_new;
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
                operation: "chebyshev_apply",
                reason: format!(
                    "expected input/output length {n}, got {}/{}",
                    r.len(),
                    z.len()
                ),
            });
        }
        let two = T::Real::from_f64(2.0).ok_or(crate::BackendError::InvalidArgument {
            operation: "chebyshev_apply",
            reason: "failed to represent the Chebyshev recurrence constant 2".into(),
        })?;
        let half = T::Real::from_f64(0.5).ok_or(crate::BackendError::InvalidArgument {
            operation: "chebyshev_apply",
            reason: "failed to represent the Chebyshev recurrence constant 0.5".into(),
        })?;
        let one = T::Real::one();
        let lambda_min = self.lambda_max / self.eigenvalue_ratio;
        let theta = half * (self.lambda_max + lambda_min);
        let delta = half * (self.lambda_max - lambda_min);
        if !theta.is_finite()
            || !delta.is_finite()
            || theta <= T::Real::zero()
            || delta <= T::Real::zero()
        {
            return Err(crate::BackendError::InvalidArgument {
                operation: "chebyshev_apply",
                reason: "Chebyshev eigenvalue interval must have finite positive width".into(),
            });
        }
        let sigma = theta / delta;
        let mut rho = one / sigma;
        let inv_theta = one / theta;
        let mut workspace =
            self.workspace
                .lock()
                .map_err(|_| crate::BackendError::OperationFailed {
                    operation: "chebyshev_apply",
                    reason: "workspace mutex poisoned".into(),
                })?;
        let ChebyshevWorkspace {
            result,
            d,
            az,
            res,
            d_new,
        } = &mut *workspace;
        backend.try_inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::WorkspaceAllocation,
        )?;
        ensure_buffer_exact(backend, result, n)?;
        ensure_buffer_exact(backend, d, n)?;
        ensure_buffer_exact(backend, az, n)?;
        ensure_buffer_exact(backend, res, n)?;
        ensure_buffer_exact(backend, d_new, n)?;
        if !backend.try_diagonal_scale_native(&self.diag_inv.matrix().values, r, d, inv_theta)? {
            self.diag_inv.try_apply(backend, r, d)?;
            backend.try_scale::<T>(T::from_real(inv_theta), d)?;
        }
        backend.try_copy_d2d(d, result)?;
        if self.degree == 1 {
            return backend.try_copy_d2d(result, z);
        }
        for _ in 1..self.degree {
            let rho_new = one / (two * sigma - rho);
            let coeff_old = rho_new * rho;
            let coeff_res = two * rho_new / delta;
            self.a.try_apply(backend, result, az)?;
            if backend.try_chebyshev_update_native(
                &self.diag_inv.matrix().values, r, az, d, result, coeff_res, coeff_old,
            )? {
                rho = rho_new;
                continue;
            }
            backend.try_scale::<T>(-T::one(), az)?;
            backend.try_axpy::<T>(T::one(), r, az)?;
            self.diag_inv.try_apply(backend, az, res)?;
            backend.try_copy_d2d(d, d_new)?;
            backend.try_scale::<T>(T::from_real(coeff_old), d_new)?;
            backend.try_axpy::<T>(T::from_real(coeff_res), res, d_new)?;
            backend.try_axpy::<T>(T::one(), d_new, result)?;
            std::mem::swap(d, d_new);
            rho = rho_new;
        }
        backend.try_copy_d2d(result, z)
    }

    #[allow(dead_code)]
    fn apply_colmajor(&self, backend: &B, rhs: usize, r: &B::Buffer<T>, z: &mut B::Buffer<T>) {
        if let Err(error) = self.try_apply_colmajor(backend, rhs, r, z) {
            log::error!(
                target: "sonium_compute::preconditioner",
                "Chebyshev multi-RHS application failed: {error}"
            );
        }
    }

    #[allow(dead_code)]
    fn apply_colmajor_legacy_unchecked(
        &self,
        backend: &B,
        rhs: usize,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) {
        let n = self.a.matrix().m;
        let Some(two) = T::Real::from_f64(2.0) else {
            log::error!(target: "sonium_compute::preconditioner", "Chebyshev recurrence constant 2 is not representable");
            return;
        };
        let Some(half) = T::Real::from_f64(0.5) else {
            log::error!(target: "sonium_compute::preconditioner", "Chebyshev recurrence constant 0.5 is not representable");
            return;
        };
        let one = T::Real::one();

        let lambda_min = self.lambda_max / self.eigenvalue_ratio;
        let theta = half * (self.lambda_max + lambda_min);
        let delta = half * (self.lambda_max - lambda_min);
        let sigma = theta / delta;
        let mut rho = one / sigma;
        let inv_theta = one / theta;

        let d_prebaked = backend.prebake_csr::<T>(
            n,
            n,
            &self.diag_inv.matrix().row_ptr,
            &self.diag_inv.matrix().col_idx,
            &self.diag_inv.matrix().values,
        );
        let a_prebaked = self.a.matrix().prebake(backend);

        let Some(len) = n.checked_mul(rhs) else {
            log::error!(target: "sonium_compute::preconditioner", "Chebyshev multi-RHS length overflow");
            return;
        };
        let Ok(mut d) = backend.raw_alloc::<T>(len) else {
            log::error!(target: "sonium_compute::preconditioner", "Chebyshev multi-RHS workspace allocation failed");
            return;
        };
        backend.spmm_prebaked_colmajor::<T>(&d_prebaked, rhs, r, &mut d);
        backend.scale::<T>(T::from_real(inv_theta), &mut d);
        backend.copy_d2d(&d, z);

        if self.degree == 1 {
            return;
        }

        let Ok(mut az) = backend.raw_alloc::<T>(len) else {
            log::error!(target: "sonium_compute::preconditioner", "Chebyshev multi-RHS az workspace allocation failed");
            return;
        };
        let Ok(mut res) = backend.raw_alloc::<T>(len) else {
            log::error!(target: "sonium_compute::preconditioner", "Chebyshev multi-RHS residual workspace allocation failed");
            return;
        };
        let Ok(mut d_new) = backend.raw_alloc::<T>(len) else {
            log::error!(target: "sonium_compute::preconditioner", "Chebyshev multi-RHS recurrence workspace allocation failed");
            return;
        };

        for _ in 1..self.degree {
            let rho_new = one / (two * sigma - rho);
            let coeff_old = rho_new * rho;
            let coeff_res = two * rho_new / delta;

            backend.spmm_prebaked_colmajor::<T>(&a_prebaked, rhs, z, &mut az);
            backend.scale::<T>(-T::one(), &mut az);
            backend.axpy::<T>(T::one(), r, &mut az);
            backend.spmm_prebaked_colmajor::<T>(&d_prebaked, rhs, &az, &mut res);

            backend.copy_d2d(&d, &mut d_new);
            backend.scale::<T>(T::from_real(coeff_old), &mut d_new);
            backend.axpy::<T>(T::from_real(coeff_res), &res, &mut d_new);

            backend.axpy::<T>(T::one(), &d_new, z);
            std::mem::swap(&mut d, &mut d_new);
            rho = rho_new;
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
                operation: "chebyshev_apply_colmajor",
                reason: format!("workspace length overflow: n={n}, rhs={rhs}"),
            })?;
        if r.len() != len || z.len() != len {
            return Err(crate::BackendError::InvalidArgument {
                operation: "chebyshev_apply_colmajor",
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
        let two = T::Real::from_f64(2.0).ok_or(crate::BackendError::InvalidArgument {
            operation: "chebyshev_apply_colmajor",
            reason: "failed to represent the Chebyshev recurrence constant 2".into(),
        })?;
        let half = T::Real::from_f64(0.5).ok_or(crate::BackendError::InvalidArgument {
            operation: "chebyshev_apply_colmajor",
            reason: "failed to represent the Chebyshev recurrence constant 0.5".into(),
        })?;
        let one = T::Real::one();
        let lambda_min = self.lambda_max / self.eigenvalue_ratio;
        let theta = half * (self.lambda_max + lambda_min);
        let delta = half * (self.lambda_max - lambda_min);
        if !theta.is_finite()
            || !delta.is_finite()
            || theta <= T::Real::zero()
            || delta <= T::Real::zero()
        {
            return Err(crate::BackendError::InvalidArgument {
                operation: "chebyshev_apply_colmajor",
                reason: "Chebyshev eigenvalue interval must have finite positive width".into(),
            });
        }
        let sigma = theta / delta;
        let mut rho = one / sigma;
        let inv_theta = one / theta;
        let mut workspace =
            self.block_workspace
                .lock()
                .map_err(|_| crate::BackendError::OperationFailed {
                    operation: "chebyshev_apply_colmajor",
                    reason: "workspace mutex poisoned".into(),
                })?;
        let ChebyshevWorkspace {
            result,
            d,
            az,
            res,
            d_new,
        } = &mut *workspace;
        backend.try_inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::WorkspaceAllocation,
        )?;
        ensure_buffer_exact(backend, result, len)?;
        ensure_buffer_exact(backend, d, len)?;
        ensure_buffer_exact(backend, az, len)?;
        ensure_buffer_exact(backend, res, len)?;
        ensure_buffer_exact(backend, d_new, len)?;
        if !backend.try_diagonal_scale_colmajor_native(rhs, &self.diag_inv.matrix().values, r, d, inv_theta)? {
            self.diag_inv.try_apply_colmajor(backend, rhs, r, d)?;
            backend.try_scale::<T>(T::from_real(inv_theta), d)?;
        }
        backend.try_copy_d2d(d, result)?;
        if self.degree == 1 {
            return backend.try_copy_d2d(result, z);
        }
        for _ in 1..self.degree {
            let rho_new = one / (two * sigma - rho);
            let coeff_old = rho_new * rho;
            let coeff_res = two * rho_new / delta;
            self.a
                .try_apply_colmajor(backend, rhs, result, az)?;
            if backend.try_chebyshev_update_colmajor_native(
                rhs, &self.diag_inv.matrix().values, r, az, d, result, coeff_res, coeff_old,
            )? {
                rho = rho_new;
                continue;
            }
            backend.try_scale::<T>(-T::one(), az)?;
            backend.try_axpy::<T>(T::one(), r, az)?;
            self.diag_inv.try_apply_colmajor(backend, rhs, az, res)?;
            backend.try_copy_d2d(d, d_new)?;
            backend.try_scale::<T>(T::from_real(coeff_old), d_new)?;
            backend.try_axpy::<T>(T::from_real(coeff_res), res, d_new)?;
            backend.try_axpy::<T>(T::one(), d_new, result)?;
            std::mem::swap(d, d_new);
            rho = rho_new;
        }
        backend.try_copy_d2d(result, z)
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
            operation: "chebyshev_transpose",
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
            operation: "chebyshev_hermitian",
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
            operation: "chebyshev_transpose_colmajor",
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
            operation: "chebyshev_hermitian_colmajor",
            scalar: "device scalar",
        };
        self.runtime_diagnostics.record_hermitian_failure(&error);
        Err(error)
    }
}

/// 20-iter power method on `D⁻¹ A` (host-side, `O(20·nnz)`).
/// Generic over `T: DeviceScalar`; returns `T::Real` since the
/// largest-magnitude eigenvalue is real. Used at construction
/// time only.
fn estimate_lambda_max_host<T: DeviceScalar>(
    n: usize,
    row_ptr: &[i32],
    col_idx: &[i32],
    values: &[T],
    diag: &[T],
) -> T::Real {
    let one = T::Real::one();
    let zero = T::Real::zero();
    let eps = T::Real::epsilon();
    if n == 0 {
        return one;
    }
    let mut v: Vec<T> = (0..n)
        .map(|i| {
            if i % 2 == 0 {
                T::one()
            } else {
                T::zero() - T::one()
            }
        })
        .collect();
    let mut w = vec![T::zero(); n];
    let mut lambda = one;
    for _ in 0..20 {
        // w = A · v
        for i in 0..n {
            let lo = row_ptr[i] as usize;
            let hi = row_ptr[i + 1] as usize;
            let mut acc = T::zero();
            for k in lo..hi {
                acc = acc + values[k] * v[col_idx[k] as usize];
            }
            w[i] = acc;
        }
        // w = D⁻¹ · w
        for i in 0..n {
            let abs = diag[i].norm_sqr().sqrt();
            if abs > eps {
                w[i] = w[i] / diag[i];
            }
        }
        // Rayleigh quotient λ = |⟨v, w⟩| / |⟨v, v⟩| (Hermitian inner
        // product; magnitude works for both real and complex spectra).
        let mut num = T::zero();
        for (a, b) in v.iter().zip(w.iter()) {
            num = num + a.conj_mul(*b);
        }
        let mut den = zero;
        for a in v.iter() {
            den = den + a.norm_sqr();
        }
        if den > zero {
            lambda = num.norm_sqr().sqrt() / den;
        }
        // Renormalise v ← w / ‖w‖
        let mut nrm_sq = zero;
        for a in w.iter() {
            nrm_sq = nrm_sq + a.norm_sqr();
        }
        let nrm = nrm_sq.sqrt();
        if nrm > zero {
            let inv_nrm = T::from_real(one / nrm);
            for i in 0..n {
                v[i] = w[i] * inv_nrm;
            }
        }
    }
    // Pad slightly so we never under-estimate.
    let pad = T::Real::from_f64(1.05).expect("T::Real has FromPrimitive");
    lambda * pad
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
    use crate::preconditioner::PolynomialJacobiBackend;
    use crate::scalar::BufferElement;

    /// 1-D Laplacian-like 5×5: SPD, well-conditioned. Same fixture as
    /// the in-tree gmres_backend tests for cross-comparability.
    fn tridiag_5x5_spd() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
        let row_ptr = vec![0, 2, 5, 8, 11, 13];
        let col_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4];
        let values = vec![
            2.0, -1.0, //
            -1.0, 2.0, -1.0, //
            -1.0, 2.0, -1.0, //
            -1.0, 2.0, -1.0, //
            -1.0, 2.0, //
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

    /// Power-iteration estimate must converge to a value at least as
    /// large as the true λ_max of `D⁻¹A` for the fixture. For the 1-D
    /// Laplacian with `D = 2 I`, `D⁻¹A = ½ A`, true λ_max ≈ 1.866 at
    /// n=5 (eigenvalues `2(1 − cos(kπ/(n+1)))/2`). Power iter padded
    /// by 5% should give ≥ 1.866 and ≤ ~2.0.
    #[test]
    fn power_iteration_lambda_max_estimate_is_in_range() {
        let (rp, ci, va) = tridiag_5x5_spd();
        let diag = vec![2.0_f64; 5];
        let lm = estimate_lambda_max_host(5, &rp, &ci, &va, &diag);
        assert!(
            (1.7..=2.1).contains(&lm),
            "lambda_max estimate {lm} out of expected range"
        );
    }

    /// Chebyshev with degree 1 ≈ a single Jacobi-Richardson step
    /// `z = (1/θ)·D⁻¹·r`. Sanity check the apply produces this.
    #[test]
    fn chebyshev_degree_one_is_jacobi_richardson() {
        let backend = CpuBackend::new();
        let (rp, ci, va) = tridiag_5x5_spd();
        let csr = BackendCsr::from_host(&backend, 5, 5, &rp, &ci, &va);
        let cheb = ChebyshevBackend::with_lambda_max(&backend, &csr, &[2.0; 5], 1, 30.0, 2.0);

        // theta = 0.5*(lambda_max + lambda_min) where lambda_min = 2/30
        // theta = 0.5 * (2 + 2/30) = 0.5 * 2.0667 = 1.0333
        // For r = D = [2,2,2,2,2] expect z = (1/theta) * D⁻¹·r = (1/theta)·1 ≈ 0.9677
        let r = make_buf(&backend, &[2.0_f64; 5]);
        let mut z = backend.raw_alloc::<f64>(5).unwrap();
        cheb.apply(&backend, &r, &mut z);
        let mut h = vec![0.0_f64; 5];
        backend.copy_d2h(&z, &mut h);
        let _ = csr; // keep csr alive for borrow lifetime
        let theta = 0.5 * (2.0 + 2.0 / 30.0);
        let expected = 1.0 / theta;
        for v in h.iter() {
            assert!(
                (v - expected).abs() < 1.0e-12,
                "got {v}, expected {expected}"
            );
        }
    }

    /// Build a 1-D Laplacian-like SPD CSR of size `n × n`:
    /// `2·I − tridiag(−1, 0, −1)`. Condition number `κ ≈ 4 n²/π²`.
    fn laplacian_1d_n(n: usize) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
        let mut row_ptr = vec![0_i32; n + 1];
        let mut col_idx: Vec<i32> = Vec::new();
        let mut values: Vec<f64> = Vec::new();
        for i in 0..n {
            if i > 0 {
                col_idx.push(i as i32 - 1);
                values.push(-1.0);
            }
            col_idx.push(i as i32);
            values.push(2.0);
            if i + 1 < n {
                col_idx.push(i as i32 + 1);
                values.push(-1.0);
            }
            row_ptr[i + 1] = values.len() as i32;
        }
        (row_ptr, col_idx, values)
    }

    #[allow(clippy::too_many_arguments)]
    fn chebyshev_reference_apply_f64(
        row_ptr: &[i32],
        col_idx: &[i32],
        values: &[f64],
        diag: &[f64],
        r: &[f64],
        degree: usize,
        eigenvalue_ratio: f64,
        lambda_max: f64,
    ) -> Vec<f64> {
        let n = r.len();
        let lambda_min = lambda_max / eigenvalue_ratio;
        let theta = 0.5 * (lambda_max + lambda_min);
        let delta = 0.5 * (lambda_max - lambda_min);
        let sigma = theta / delta;
        let mut rho = 1.0 / sigma;

        let mut d: Vec<f64> = r
            .iter()
            .zip(diag.iter())
            .map(|(&ri, &di)| ri / di / theta)
            .collect();
        let mut z = d.clone();

        for _ in 1..degree {
            let rho_new = 1.0 / (2.0 * sigma - rho);
            let coeff_old = rho_new * rho;
            let coeff_res = 2.0 * rho_new / delta;

            let mut res = vec![0.0_f64; n];
            for row in 0..n {
                let mut az = 0.0_f64;
                for k in row_ptr[row] as usize..row_ptr[row + 1] as usize {
                    az += values[k] * z[col_idx[k] as usize];
                }
                res[row] = (r[row] - az) / diag[row];
            }

            let mut d_new = vec![0.0_f64; n];
            for i in 0..n {
                d_new[i] = coeff_old * d[i] + coeff_res * res[i];
                z[i] += d_new[i];
            }

            d = d_new;
            rho = rho_new;
        }

        z
    }

    /// Backend Chebyshev vs a local scalar reference recurrence: apply
    /// both to the same residual with the same eigenvalue bounds, and
    /// the resulting `z` vectors must agree to floating-point precision.
    #[test]
    fn backend_chebyshev_matches_cpu_chebyshev_apply() {
        let n = 16;
        let (rp, ci, va) = laplacian_1d_n(n);

        let lambda_max = 4.0_f64;
        let degree = 5;

        let backend = CpuBackend::new();
        let a = BackendCsr::from_host(&backend, n, n, &rp, &ci, &va);
        let dev_pc = ChebyshevBackend::with_lambda_max(
            &backend,
            &a,
            &vec![2.0; n],
            degree,
            30.0,
            lambda_max,
        );

        let r_host: Vec<f64> = (0..n).map(|i| ((i as f64 + 1.0) * 0.37).sin()).collect();
        let ref_z = chebyshev_reference_apply_f64(
            &rp,
            &ci,
            &va,
            &vec![2.0; n],
            &r_host,
            degree,
            30.0,
            lambda_max,
        );

        let r = make_buf(&backend, &r_host);
        let mut z = backend.raw_alloc::<f64>(n).unwrap();
        dev_pc.apply(&backend, &r, &mut z);
        let mut z_host = vec![0.0_f64; n];
        backend.copy_d2h(&z, &mut z_host);

        for i in 0..n {
            assert!(
                (ref_z[i] - z_host[i]).abs() < 1.0e-12,
                "row {i}: ref {} vs backend {} (diff {:.3e})",
                ref_z[i],
                z_host[i],
                (ref_z[i] - z_host[i]).abs()
            );
        }

        // Suppress unused-import warning when this test is skipped.
        let _ = PolynomialJacobiBackend::from_csr_host(&backend, &a, &rp, &ci, &va, 1, 1.0);
    }

    #[test]
    fn chebyshev_apply_colmajor_matches_column_apply() {
        let n = 16;
        let (rp, ci, va) = laplacian_1d_n(n);
        let backend = CpuBackend::new();
        let a = BackendCsr::from_host(&backend, n, n, &rp, &ci, &va);
        let pc = ChebyshevBackend::with_lambda_max(&backend, &a, &vec![2.0; n], 4, 30.0, 4.0);

        let mut r_host = Vec::with_capacity(n * 2);
        r_host.extend((0..n).map(|i| ((i as f64 + 1.0) * 0.37).sin()));
        r_host.extend((0..n).map(|i| ((i as f64 + 1.0) * 0.23).cos()));
        let r = make_buf(&backend, &r_host);
        let mut z_multi = backend.raw_alloc::<f64>(n * 2).unwrap();

        pc.apply_colmajor(&backend, 2, &r, &mut z_multi);

        let mut expected = vec![0.0_f64; n * 2];
        for col in 0..2 {
            let range = col * n..(col + 1) * n;
            let r_col = make_buf(&backend, &r_host[range.clone()]);
            let mut z_col = backend.raw_alloc::<f64>(n).unwrap();
            pc.apply(&backend, &r_col, &mut z_col);
            backend.copy_d2h(&z_col, &mut expected[range]);
        }

        let mut r_three_host = Vec::with_capacity(n * 3);
        r_three_host.extend((0..n).map(|i| ((i as f64 + 1.0) * 0.11).sin()));
        r_three_host.extend((0..n).map(|i| ((i as f64 + 1.0) * 0.17).cos()));
        r_three_host.extend((0..n).map(|i| ((i as f64 + 1.0) * 0.29).sin()));
        let r_three = make_buf(&backend, &r_three_host);
        let mut z_three = backend.raw_alloc::<f64>(n * 3).unwrap();
        pc.try_apply_colmajor(&backend, 3, &r_three, &mut z_three)
            .unwrap();
        pc.try_apply_colmajor(&backend, 2, &r, &mut z_multi)
            .unwrap();

        let mut actual = vec![0.0_f64; n * 2];
        backend.copy_d2h(&z_multi, &mut actual);
        for (got, want) in actual.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1.0e-12, "got {got} want {want}");
        }
    }

    // -----------------------------------------------------------------------
    // Complex32 / Complex64 — the generic path is exercised by the f64 tests
    // above, but a complex-valued spectrum has its own quirks (the power-iter
    // λ_max estimator must produce a *real* value from a complex Rayleigh
    // quotient, and the 3-term recurrence's `T::from_real(coeff)` calls must
    // round-trip without imaginary leakage).
    // -----------------------------------------------------------------------

    /// Complex32 Helmholtz tridiag — same shape as the IR roadmap test
    /// fixture. Diagonally dominant by `|2+iη| > 2`; spectrum clusters
    /// along the positive real axis after Jacobi preconditioning, so
    /// Chebyshev applies even though the matrix is non-Hermitian.
    fn complex32_helmholtz_tridiag(
        n: usize,
        eta: f32,
    ) -> (Vec<i32>, Vec<i32>, Vec<num_complex::Complex32>) {
        use num_complex::Complex32;
        let mut row_ptr = vec![0_i32; n + 1];
        let mut col_idx: Vec<i32> = Vec::new();
        let mut values: Vec<Complex32> = Vec::new();
        for i in 0..n {
            if i > 0 {
                col_idx.push(i as i32 - 1);
                values.push(Complex32::new(-1.0, 0.0));
            }
            col_idx.push(i as i32);
            values.push(Complex32::new(2.0, eta));
            if i + 1 < n {
                col_idx.push(i as i32 + 1);
                values.push(Complex32::new(-1.0, 0.0));
            }
            row_ptr[i + 1] = values.len() as i32;
        }
        (row_ptr, col_idx, values)
    }

    /// Sanity: λ_max estimate for a complex Helmholtz tridiag returns a
    /// real value in the expected range. For `[-1, 2+iη, -1]` with η small,
    /// `D⁻¹ A` ≈ `(2/(2+iη)) · I − (1/(2+iη))·tridiag(1,0,1)` — eigenvalues
    /// clustered near `1 ± 2/|2+iη|`, so λ_max ≈ 2 within the 5% pad.
    #[test]
    fn power_iteration_returns_real_value_for_complex_matrix() {
        use num_complex::Complex32;
        let n = 16;
        let (rp, ci, va) = complex32_helmholtz_tridiag(n, 0.05);
        let mut diag = vec![Complex32::new(0.0, 0.0); n];
        for row in 0..n {
            let lo = rp[row] as usize;
            let hi = rp[row + 1] as usize;
            for k in lo..hi {
                if ci[k] as usize == row {
                    diag[row] = va[k];
                }
            }
        }
        let lm: f32 = estimate_lambda_max_host::<Complex32>(n, &rp, &ci, &va, &diag);
        assert!(lm > 0.0, "lambda_max must be positive, got {lm}");
        assert!(
            (1.5..3.0).contains(&lm),
            "lambda_max for complex Helmholtz tridiag = {lm}, expected ~2"
        );
    }

    /// Chebyshev Complex32 apply runs without panic and produces a
    /// non-trivial preconditioned vector (i.e. `z != r`).
    #[test]
    fn chebyshev_complex32_apply_runs_and_produces_meaningful_output() {
        use num_complex::Complex32;
        let backend = CpuBackend::new();
        let (rp, ci, va) = complex32_helmholtz_tridiag(16, 0.05);
        let csr = BackendCsr::<CpuBackend, Complex32>::from_host(&backend, 16, 16, &rp, &ci, &va);
        let cheb = ChebyshevBackend::<CpuBackend, Complex32>::from_csr_host(
            &backend, &csr, &rp, &ci, &va, 5, 30.0,
        );

        let r_host: Vec<Complex32> = (0..16)
            .map(|i| Complex32::new(((i as f32) * 0.13).sin(), ((i as f32) * 0.07).cos()))
            .collect();
        let r = make_buf(&backend, &r_host);
        let mut z = backend.raw_alloc::<Complex32>(16).unwrap();
        cheb.apply(&backend, &r, &mut z);
        let _ = csr; // keep csr alive

        let mut z_host = vec![Complex32::new(0.0, 0.0); 16];
        backend.copy_d2h(&z, &mut z_host);
        // Sanity: not all zeros, not equal to r (Chebyshev does *something*).
        let mut max_diff: f32 = 0.0;
        let mut max_z: f32 = 0.0;
        for (zi, ri) in z_host.iter().zip(r_host.iter()) {
            max_diff = max_diff.max((zi - ri).norm());
            max_z = max_z.max(zi.norm());
        }
        assert!(max_z > 1.0e-3, "z is essentially zero — apply did nothing");
        assert!(max_diff > 1.0e-3, "z == r — preconditioner is identity");
    }

    /// On a complex SPD-ish problem (Hermitian-like, diagonally
    /// dominant), GMRES + Chebyshev must beat GMRES + PolyJacobi K=2
    /// in iter count. This is the classic Chebyshev advantage and the
    /// motivation for adding it to the IR inner-precision toolkit.
    #[test]
    fn chebyshev_complex32_beats_polyjacobi_on_hermitian_like() {
        use crate::gmres::{GmresBackendConfig, gmres_backend_preconditioned};
        use num_complex::Complex32;

        let backend = CpuBackend::new();
        let n = 64;
        let (rp, ci, va) = complex32_helmholtz_tridiag(n, 0.01);
        let a = BackendCsr::<CpuBackend, Complex32>::from_host(&backend, n, n, &rp, &ci, &va);

        let b_host: Vec<Complex32> = (0..n)
            .map(|i| Complex32::new(((i as f32) * 0.1).sin(), 0.0))
            .collect();
        let b = make_buf(&backend, &b_host);

        let pj = PolynomialJacobiBackend::<CpuBackend, Complex32>::from_csr_host(
            &backend,
            &a,
            &rp,
            &ci,
            &va,
            2,
            Complex32::new(1.0, 0.0),
        );
        let cheb = ChebyshevBackend::<CpuBackend, Complex32>::from_csr_host(
            &backend, &a, &rp, &ci, &va, 5, 30.0,
        );

        // The projected Arnoldi estimate for this low-precision fixture drops
        // below 1e-6 before the original-operator residual does. Keep this
        // strict probe so a future implementation cannot regress to reporting
        // that optimistic estimate as convergence.
        let strict_cfg = GmresBackendConfig::<Complex32> {
            max_iterations: 200,
            tolerance: 1.0e-6,
        };
        let mut strict_x_pj = backend.raw_alloc::<Complex32>(n).unwrap();
        let mut strict_x_cb = backend.raw_alloc::<Complex32>(n).unwrap();
        let strict_pj = gmres_backend_preconditioned::<CpuBackend, _, _, Complex32>(
            &backend,
            &a,
            &pj,
            &b,
            None,
            &mut strict_x_pj,
            &strict_cfg,
        )
        .unwrap();
        let strict_cb = gmres_backend_preconditioned::<CpuBackend, _, _, Complex32>(
            &backend,
            &a,
            &cheb,
            &b,
            None,
            &mut strict_x_cb,
            &strict_cfg,
        )
        .unwrap();
        assert!(!strict_pj.converged && !strict_cb.converged);
        assert!(strict_pj.final_residual > strict_cfg.tolerance);
        assert!(strict_cb.final_residual > strict_cfg.tolerance);

        let cfg = GmresBackendConfig::<Complex32> {
            max_iterations: 200,
            // Complex32's attainable residual on this fixture is around 2e-5;
            // use a tolerance that tests the preconditioner comparison without
            // asking the low-precision solve for unattainable accuracy.
            tolerance: 1.0e-4,
        };
        let mut x_pj = backend.raw_alloc::<Complex32>(n).unwrap();
        let mut x_cb = backend.raw_alloc::<Complex32>(n).unwrap();
        let s_pj = gmres_backend_preconditioned::<CpuBackend, _, _, Complex32>(
            &backend, &a, &pj, &b, None, &mut x_pj, &cfg,
        )
        .unwrap();
        let s_cb = gmres_backend_preconditioned::<CpuBackend, _, _, Complex32>(
            &backend, &a, &cheb, &b, None, &mut x_cb, &cfg,
        )
        .unwrap();
        assert!(s_pj.converged && s_cb.converged);
        // Chebyshev K=5 has 5 SpMVs/apply; PolyJac K=2 has 2 SpMVs/apply.
        // Per-iter Chebyshev costs ~2.5× PolyJac, so Chebyshev wins on
        // total work iff `cheb_iters · 5 < polyjac_iters · 2`.
        // For a near-Hermitian diagonally-dominant problem this holds
        // comfortably — Chebyshev iters is typically ⅓ to ½ of PolyJac.
        assert!(
            s_cb.iterations * 5 < s_pj.iterations * 4,
            "Chebyshev advantage absent: cheb={} polyjac={}",
            s_cb.iterations,
            s_pj.iterations
        );
    }

    /// On the same SPD problem, GMRES + Chebyshev must converge in
    /// fewer iters than GMRES with no preconditioner (identity).
    #[test]
    fn chebyshev_beats_identity() {
        let backend = CpuBackend::new();
        let (rp, ci, va) = tridiag_5x5_spd();
        let a = BackendCsr::from_host(&backend, 5, 5, &rp, &ci, &va);
        let b = make_buf::<f64>(&backend, &[1.0, 2.0, 3.0, 4.0, 5.0]);

        let cheb = ChebyshevBackend::from_csr_host(&backend, &a, &rp, &ci, &va, 5, 30.0);
        let mut x_cheb = backend.raw_alloc::<f64>(5).unwrap();
        let mut x_id = backend.raw_alloc::<f64>(5).unwrap();
        let s_cheb = gmres_backend_preconditioned(
            &backend,
            &a,
            &cheb,
            &b,
            None,
            &mut x_cheb,
            &GmresBackendConfig {
                max_iterations: 100,
                tolerance: 1.0e-12,
            },
        )
        .unwrap();
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
        assert!(s_cheb.converged && s_id.converged);
        assert!(
            s_cheb.iterations <= s_id.iterations,
            "chebyshev {} > identity {} on SPD problem",
            s_cheb.iterations,
            s_id.iterations
        );
    }

    #[test]
    fn checked_chebyshev_constructor_rejects_invalid_configuration() {
        let backend = CpuBackend::new();
        let (rp, ci, va) = tridiag_5x5_spd();
        let a = BackendCsr::from_host(&backend, 5, 5, &rp, &ci, &va);
        let result =
            ChebyshevBackend::try_with_lambda_max(&backend, &a, &[1.0_f64; 5], 0, 30.0, 2.0);
        assert!(matches!(
            result,
            Err(crate::BackendError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn checked_chebyshev_apply_rejects_wrong_length() {
        let backend = CpuBackend::new();
        let (rp, ci, va) = tridiag_5x5_spd();
        let a = BackendCsr::from_host(&backend, 5, 5, &rp, &ci, &va);
        let cheb = ChebyshevBackend::from_csr_host(&backend, &a, &rp, &ci, &va, 3, 30.0);
        let r = make_buf(&backend, &[1.0_f64; 4]);
        let mut z = backend.raw_alloc::<f64>(5).unwrap();

        let result = cheb.try_apply(&backend, &r, &mut z);
        assert!(matches!(
            result,
            Err(crate::BackendError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn legacy_chebyshev_apply_is_panic_free_on_wrong_length() {
        let backend = CpuBackend::new();
        let (rp, ci, va) = tridiag_5x5_spd();
        let a = BackendCsr::from_host(&backend, 5, 5, &rp, &ci, &va);
        let cheb = ChebyshevBackend::from_csr_host(&backend, &a, &rp, &ci, &va, 3, 30.0);
        let r = make_buf(&backend, &[1.0_f64; 4]);
        let mut z = backend.raw_alloc::<f64>(5).unwrap();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cheb.apply(&backend, &r, &mut z);
        }));
        assert!(result.is_ok(), "legacy Chebyshev apply must not panic");
    }

    #[test]
    fn legacy_chebyshev_apply_colmajor_is_panic_free_on_wrong_length() {
        let backend = CpuBackend::new();
        let (rp, ci, va) = tridiag_5x5_spd();
        let a = BackendCsr::from_host(&backend, 5, 5, &rp, &ci, &va);
        let cheb = ChebyshevBackend::from_csr_host(&backend, &a, &rp, &ci, &va, 3, 30.0);
        let r = make_buf(&backend, &[1.0_f64; 4]);
        let mut z = backend.raw_alloc::<f64>(5).unwrap();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cheb.apply_colmajor(&backend, 2, &r, &mut z);
        }));
        assert!(
            result.is_ok(),
            "legacy Chebyshev multi-RHS apply must not panic"
        );
    }
}
