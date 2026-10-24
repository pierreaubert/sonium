//! Backend-resident preconditioned Conjugate Gradient (PCG).
//!
//! Device-side counterpart of the host [`pcg`][crate] solver: the iterate,
//! residual, search direction, and `A*p` vectors all stay in backend buffers
//! while host scalars drive the recurrence. Works with any [`Backend`]
//! (CPU, CUDA, Metal) plus any [`BackendOperator`] / [`BackendPreconditioner`]
//! pair — in particular a [`BackendAmgPreconditioner`][amg] uploaded with
//! `from_host_amg`, which closes the GPU CG+AMG solve bridge without staging
//! vectors through the host each iteration.
//!
//! Like the host solver this implements standard (left-)preconditioned CG,
//! which requires a Hermitian positive-definite operator and preconditioner.
//! Indefiniteness surfaces as [`PcgBackendError::Breakdown`], mirroring the
//! host breakdown contract.
//!
//! [amg]: crate::amg

use crate::operator::{DeviceOperator, DevicePreconditioner};
use crate::{Backend, DeviceBufferLike, DeviceScalar, MemoryPool, ReductionMode};
use std::marker::PhantomData;

/// PCG configuration, generic over the scalar type `T`.
/// Tolerance is `T::Real` (real for both real and complex `T`).
#[derive(Debug, Clone)]
pub struct PcgBackendConfig<T: DeviceScalar = f64> {
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Relative residual tolerance for convergence (`||r_k|| / ||b||`).
    pub tolerance: T::Real,
}

impl<T: DeviceScalar> Default for PcgBackendConfig<T> {
    fn default() -> Self {
        use num_traits::Zero;
        Self {
            max_iterations: 200,
            // `DeviceScalar` is extensible.  If a custom real type cannot
            // represent the legacy literal, fail closed; the checked solver
            // rejects non-finite tolerance as invalid input.
            tolerance: <T::Real as num_traits::FromPrimitive>::from_f64(1.0e-8)
                .unwrap_or_else(T::Real::zero),
        }
    }
}

/// Outcome of a backend PCG solve.
#[derive(Debug, Clone)]
pub struct PcgBackendStatus<T: DeviceScalar = f64> {
    /// Iterations performed.
    pub iterations: usize,
    /// Final relative residual `||r|| / ||b||` from the CG recurrence (the
    /// same quantity the host solver reports; not independently recomputed).
    pub final_residual: T::Real,
    /// Whether `final_residual` satisfies the configured tolerance.
    pub converged: bool,
}

/// Errors that can be returned from the solver.
#[derive(Debug, thiserror::Error)]
pub enum PcgBackendError {
    #[error("PCG workspace allocation failed for {label}: {reason}")]
    Allocation { label: &'static str, reason: String },
    #[error("PCG invalid input: {reason}")]
    InvalidArgument { reason: String },
    #[error("right-hand side has zero norm; solution is x = x0")]
    ZeroRhs,
    #[error(
        "PCG breakdown at iteration {iteration}: <p, Ap> collapsed; operator or preconditioner is not positive-definite"
    )]
    Breakdown { iteration: usize },
    #[error("backend operation failed: {source}")]
    Backend { source: crate::BackendError },
}

impl From<crate::BackendError> for PcgBackendError {
    fn from(source: crate::BackendError) -> Self {
        Self::Backend { source }
    }
}

fn alloc_error(label: &'static str, error: crate::OomError) -> PcgBackendError {
    PcgBackendError::Allocation {
        label,
        reason: error.to_string(),
    }
}

/// Retained device scratch for repeated PCG solves: residual `r`,
/// preconditioned residual `z`, search direction `p`, and `q = A*p`.
/// Buffers are allocated through the pool on first use and fully overwritten
/// on every solve, so capacity (not content) carries over between solves.
pub struct PcgBackendWorkspace<B: Backend, T: DeviceScalar> {
    n: usize,
    r: Option<crate::pool::Buffer<T, B>>,
    z: Option<crate::pool::Buffer<T, B>>,
    p: Option<crate::pool::Buffer<T, B>>,
    q: Option<crate::pool::Buffer<T, B>>,
    _phantom: PhantomData<B>,
}

impl<B: Backend, T: DeviceScalar> PcgBackendWorkspace<B, T> {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            r: None,
            z: None,
            p: None,
            q: None,
            _phantom: PhantomData,
        }
    }

    pub fn dimension(&self) -> usize {
        self.n
    }

    /// Number of retained device buffers available for reuse.
    pub fn cached_buffer_count(&self) -> usize {
        [
            self.r.is_some(),
            self.z.is_some(),
            self.p.is_some(),
            self.q.is_some(),
        ]
        .into_iter()
        .filter(|present| *present)
        .count()
    }

    fn ensure(&mut self, pool: &MemoryPool<B>) -> Result<(), PcgBackendError> {
        // `try_alloc` returns zeroed buffers; every buffer below is fully
        // overwritten before any read on each solve regardless.
        if self.r.is_none() {
            self.r = Some(
                pool.try_alloc::<T>(self.n)
                    .map_err(|e| alloc_error("r", e))?,
            );
        }
        if self.z.is_none() {
            self.z = Some(
                pool.try_alloc::<T>(self.n)
                    .map_err(|e| alloc_error("z", e))?,
            );
        }
        if self.p.is_none() {
            self.p = Some(
                pool.try_alloc::<T>(self.n)
                    .map_err(|e| alloc_error("p", e))?,
            );
        }
        if self.q.is_none() {
            self.q = Some(
                pool.try_alloc::<T>(self.n)
                    .map_err(|e| alloc_error("q", e))?,
            );
        }
        Ok(())
    }
}

/// Backend-resident preconditioned Conjugate Gradient.
///
/// `out` holds the solution on return. When `x0` is given it is the initial
/// guess, otherwise the solve starts from zero. Only `b`, `x0`, and `out`
/// cross the caller boundary; all iteration traffic stays on the device
/// except three host-side scalar reductions per step (two Hermitian dots
/// plus the residual norm, evaluated as sqrt(dot) — see the loop). Pass a
/// persistent operator (e.g. [`PersistentDeviceCsr`](crate::PersistentDeviceCsr))
/// for `a`: a plain [`DeviceCsr`](crate::operator::DeviceCsr) re-prebakes
/// its cuSPARSE handle on every application (~7x per-SpMV cost).
#[allow(clippy::too_many_arguments)]
pub fn pcg_backend_preconditioned<B, A, P, T>(
    backend: &B,
    pool: &MemoryPool<B>,
    workspace: &mut PcgBackendWorkspace<B, T>,
    a: &A,
    precond: &P,
    b: &B::Buffer<T>,
    x0: Option<&B::Buffer<T>>,
    out: &mut B::Buffer<T>,
    config: &PcgBackendConfig<T>,
) -> Result<PcgBackendStatus<T>, PcgBackendError>
where
    B: Backend,
    A: DeviceOperator<B, T> + ?Sized,
    P: DevicePreconditioner<B, T> + ?Sized,
    T: DeviceScalar,
{
    use num_traits::{Float, FromPrimitive, Zero};

    let n = a.n();
    if b.len() != n || out.len() != n {
        return Err(PcgBackendError::InvalidArgument {
            reason: format!(
                "operator dimension {n} disagrees with b ({}) / out ({})",
                b.len(),
                out.len()
            ),
        });
    }
    if x0.is_some_and(|x| x.len() != n) {
        return Err(PcgBackendError::InvalidArgument {
            reason: format!("initial guess length disagrees with operator dimension {n}"),
        });
    }
    if workspace.dimension() != n {
        return Err(PcgBackendError::InvalidArgument {
            reason: format!(
                "workspace dimension {} != operator {n}",
                workspace.dimension()
            ),
        });
    }
    if !config.tolerance.is_finite() || config.tolerance < T::Real::zero() {
        return Err(PcgBackendError::InvalidArgument {
            reason: "tolerance must be finite and non-negative".into(),
        });
    }
    workspace.ensure(pool)?;

    let b_norm = backend.try_nrm2_with_mode::<T>(ReductionMode::Fast, b)?;
    if b_norm == T::Real::zero() {
        // Zero right-hand side: the initial guess is already the solution,
        // mirroring the host solver's zero-RHS contract.
        if let Some(x) = x0 {
            backend.try_copy_d2d(x, out)?;
        } else {
            backend.try_scale::<T>(T::zero(), out)?;
        }
        return Err(PcgBackendError::ZeroRhs);
    }

    let r = workspace.r.as_mut().expect("workspace ensured");
    let z = workspace.z.as_mut().expect("workspace ensured");
    let p = workspace.p.as_mut().expect("workspace ensured");
    let q = workspace.q.as_mut().expect("workspace ensured");

    // Initial residual: r = b - A*out.
    if let Some(x) = x0 {
        backend.try_copy_d2d(x, out)?;
        a.try_apply(backend, out, q.raw_mut())?;
        backend.try_copy_d2d(b, r.raw_mut())?;
        backend.try_axpy::<T>(T::zero() - T::one(), q.raw(), r.raw_mut())?;
    } else {
        backend.try_scale::<T>(T::zero(), out)?;
        backend.try_copy_d2d(b, r.raw_mut())?;
    }

    // A retained solution can already satisfy the new system. Check before
    // applying M or forming a zero search direction (which would break down).
    if x0.is_some() {
        let initial_residual =
            backend.try_nrm2_with_mode::<T>(ReductionMode::Fast, r.raw())? / b_norm;
        if initial_residual <= config.tolerance {
            return Ok(PcgBackendStatus {
                iterations: 0,
                final_residual: initial_residual,
                converged: true,
            });
        }
    }

    // Breakdown guard mirrors the host solver: relative to ||b|| with an
    // absolute floor, so tiny-but-valid pivots on scaled problems survive.
    let breakdown_threshold = (b_norm * T::Real::from_f64(1e-20).unwrap_or_else(T::Real::zero))
        .max(T::Real::from_f64(1e-30).unwrap_or_else(T::Real::zero));

    precond.try_apply(backend, r.raw(), z.raw_mut())?;
    backend.try_copy_d2d(z.raw(), p.raw_mut())?;
    let mut rho = backend.try_dot_with_mode::<T>(ReductionMode::Fast, r.raw(), z.raw())?;

    for iter in 0..config.max_iterations {
        a.try_apply(backend, p.raw(), q.raw_mut())?;
        let pq = backend.try_dot_with_mode::<T>(ReductionMode::Fast, p.raw(), q.raw())?;
        if pq.magnitude() < breakdown_threshold {
            return Err(PcgBackendError::Breakdown { iteration: iter });
        }
        let alpha = rho / pq;
        backend.try_axpy::<T>(alpha, p.raw(), out)?;
        backend.try_axpy::<T>(T::zero() - alpha, q.raw(), r.raw_mut())?;

        // Residual norm via sqrt(dot): cuBLAS `nrm2`'s robust multi-pass
        // kernel costs ~6x a dot at 512k rows (225 vs 37us on Blackwell),
        // while the Hermitian dot here equals ||r||^2 for every scalar
        // type. `magnitude().sqrt()` keeps the complex path real-valued.
        // Overflow semantics differ from `nrm2` at |r| ~ 1e154+ (the
        // dot's own finiteness check still guards); non-physical for a
        // converging residual.
        let rr = backend.try_dot_with_mode::<T>(ReductionMode::Fast, r.raw(), r.raw())?;
        let rel_residual = rr.magnitude().sqrt() / b_norm;
        if rel_residual < config.tolerance {
            return Ok(PcgBackendStatus {
                iterations: iter + 1,
                final_residual: rel_residual,
                converged: true,
            });
        }

        precond.try_apply(backend, r.raw(), z.raw_mut())?;
        let rho_new = backend.try_dot_with_mode::<T>(ReductionMode::Fast, r.raw(), z.raw())?;
        let beta = rho_new / rho;
        rho = rho_new;
        backend.try_scale::<T>(beta, p.raw_mut())?;
        backend.try_axpy::<T>(T::one(), z.raw(), p.raw_mut())?;
    }

    let r_norm = backend.try_nrm2_with_mode::<T>(ReductionMode::Fast, r.raw())?;
    Ok(PcgBackendStatus {
        iterations: config.max_iterations,
        final_residual: r_norm / b_norm,
        converged: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuBackend;

    /// Tiny SPD fixture shared by the workspace tests: tridiagonal -1/2/-1.
    /// Index arrays are `i32`, matching [`DeviceCsr::try_from_host`].
    fn fixture(n: usize) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            row_ptr.push(col_idx.len() as i32);
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
        }
        row_ptr.push(col_idx.len() as i32);
        (row_ptr, col_idx, values)
    }

    #[test]
    fn workspace_caches_four_buffers() {
        use crate::MemoryPoolConfig;
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 30)).unwrap();
        let mut ws = PcgBackendWorkspace::<CpuBackend, f64>::new(8);
        assert_eq!(ws.cached_buffer_count(), 0);
        ws.ensure(&pool).unwrap();
        assert_eq!(ws.cached_buffer_count(), 4);
    }

    #[test]
    fn accepts_converged_warm_start_before_search_direction_breakdown() {
        use crate::{DeviceCsr, IdentityDevicePreconditioner, MemoryPoolConfig};
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let backend = pool.backend();
        let (rows, columns, values) = fixture(4);
        let a = DeviceCsr::try_from_host(backend, 4, 4, &rows, &columns, &values).unwrap();
        let mut workspace = PcgBackendWorkspace::<_, f64>::new(4);
        let mut rhs = pool.try_alloc(4).unwrap();
        let mut guess = pool.try_alloc(4).unwrap();
        let mut out = pool.try_alloc(4).unwrap();
        backend
            .try_copy_h2d(&[1.0, 0.0, 0.0, 1.0], rhs.raw_mut())
            .unwrap();
        for perturbation in [0.0, 1e-12] {
            let expected = [1.0 + perturbation, 1.0, 1.0, 1.0];
            backend.try_copy_h2d(&expected, guess.raw_mut()).unwrap();
            let status = pcg_backend_preconditioned(
                backend,
                &pool,
                &mut workspace,
                &a,
                &IdentityDevicePreconditioner,
                rhs.raw(),
                Some(guess.raw()),
                out.raw_mut(),
                &PcgBackendConfig {
                    max_iterations: 4,
                    tolerance: 1e-10,
                },
            )
            .unwrap();
            assert!(status.converged);
            assert_eq!(status.iterations, 0);
            assert!(status.final_residual <= 1e-10);
            let mut observed = [0.0; 4];
            backend.try_copy_d2h(out.raw(), &mut observed).unwrap();
            assert_eq!(observed, expected);
        }
    }

    #[test]
    fn rejects_dimension_mismatch() {
        use crate::MemoryPoolConfig;
        use crate::operator::{DeviceCsr, IdentityDevicePreconditioner};
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 30)).unwrap();
        let cpu = pool.backend();
        let (rp, ci, vals) = fixture(4);
        let a = DeviceCsr::try_from_host(cpu, 4, 4, &rp, &ci, &vals).unwrap();
        let pre = IdentityDevicePreconditioner;
        let mut ws = PcgBackendWorkspace::<CpuBackend, f64>::new(4);
        let b = cpu.raw_alloc::<f64>(3).unwrap();
        let mut out = cpu.raw_alloc::<f64>(4).unwrap();
        let err = pcg_backend_preconditioned(
            cpu,
            &pool,
            &mut ws,
            &a,
            &pre,
            &b,
            None,
            &mut out,
            &PcgBackendConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(err, PcgBackendError::InvalidArgument { .. }));
    }
}
