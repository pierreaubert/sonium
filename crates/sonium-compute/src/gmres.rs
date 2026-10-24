//! Backend-aware GMRES for the GPU-offload path.
//!
//! Unlike `gmres.rs` which operates on `Vec<T>` host
//! buffers, this implementation lives entirely on
//! [`sonium_compute::Backend::Buffer<T>`]. With `CpuBackend` the buffer
//! is a host vector and BLAS-1 ops use the compute CPU backend; with
//! `CudaBackend` they hit cudarc/cuBLAS/cuSPARSE kernels and do not
//! round-trip through the host on the Krylov hot path.
//!
//! The Krylov basis `V` is a `Vec<Backend::Buffer<T>>` with one
//! buffer per column. Givens rotations and the small upper-Hessenberg
//! triangular solve stay on the host (size ≤ `restart`²; trivially
//! cheap).
//!
//! # Scope
//!
//! - **Generic scalar type.** The solver works for every
//!   `T: sonium_compute::DeviceScalar`, including real and complex
//!   f32/f64 pairs supported by the active backend.
//! - **Right-preconditioned** (the convention used by the existing
//!   `gmres_preconditioned_with_guess`). Trial vector update applies
//!   `M⁻¹` once per inner iteration.
//! - **No restart.** `restart` is the inner iteration cap; if not
//!   converged the solver returns `Err(...)`. Callers wanting restart
//!   behaviour can wrap this in a loop. Keeping the implementation
//!   restart-free makes the host-side Hessenberg bookkeeping much
//!   simpler.
//!
//! # Verification
//!
//! See the tests at the bottom of this file: they run real and complex
//! CPU solves and compare the backend operator path against known
//! residual targets. CUDA/Metal-specific coverage lives in
//! `gmres_ir_backend` and the compute backend parity tests.

use crate::{Backend, DeviceBufferLike, DeviceScalar, MemoryPool, ReductionMode};
use std::cell::RefCell;
use std::sync::{Arc, Mutex};

pub use crate::operator::{
    DeviceCsr as BackendCsr, DeviceOperator as BackendOperator,
    DevicePreconditioner as BackendPreconditioner,
    IdentityDevicePreconditioner as IdentityBackendPreconditioner,
};

/// GMRES configuration, generic over the scalar type `T`.
/// Tolerance is `T::Real` (real for both real and complex `T`).
#[derive(Debug, Clone)]
pub struct GmresBackendConfig<T: DeviceScalar = f64> {
    /// Maximum inner iterations; the solver does not restart.
    pub max_iterations: usize,
    /// Relative residual tolerance for convergence (`||r_k|| / ||b||`).
    pub tolerance: T::Real,
}

impl<T: DeviceScalar> Default for GmresBackendConfig<T> {
    fn default() -> Self {
        use num_traits::Zero;
        Self {
            max_iterations: 200,
            // `DeviceScalar` is extensible.  If a custom real type cannot
            // represent the legacy literal, fail closed; the checked solver
            // rejects zero tolerance as typed invalid input.
            tolerance: <T::Real as num_traits::FromPrimitive>::from_f64(1.0e-8)
                .unwrap_or_else(T::Real::zero),
        }
    }
}

/// Outcome of a backend GMRES solve.
#[derive(Debug, Clone)]
pub struct GmresBackendStatus<T: DeviceScalar = f64> {
    pub iterations: usize,
    /// Relative residual predicted by the Hessenberg/Givens recurrence.
    /// This is diagnostic only; `final_residual` is always independently
    /// recomputed from the original operator.
    /// For a zero RHS, residuals are absolute rather than relative.
    pub projected_residual: T::Real,
    /// Relative residual recomputed with the original operator after the
    /// final candidate is formed (not the projected Hessenberg estimate).
    /// For a zero RHS, this is the absolute residual norm.
    pub final_residual: T::Real,
    /// Normwise backward error of the returned candidate.
    pub backward_error: T::Real,
    /// Whether the certified `final_residual` satisfies the configured
    /// tolerance.
    pub converged: bool,
}

/// Errors that can be returned from the solver. Length mismatches are
/// asserts (programming errors); these are runtime conditions.
#[derive(Debug, thiserror::Error)]
pub enum GmresBackendError {
    #[error("GMRES workspace allocation failed for {label}: {reason}")]
    Allocation { label: &'static str, reason: String },
    #[error("GMRES workspace contract failed: {reason}")]
    Workspace { reason: String },
    #[error("right-hand side has zero norm; solution is x = x0")]
    ZeroRhs,
    #[error(
        "Arnoldi vector collapsed to zero at iteration {iteration}; matrix is rank-deficient or starting guess is too good"
    )]
    HappyBreakdown { iteration: usize },
    #[error("backend operation failed: {source}")]
    Backend { source: crate::BackendError },
}

impl From<crate::BackendError> for GmresBackendError {
    fn from(source: crate::BackendError) -> Self {
        Self::Backend { source }
    }
}

trait GmresOwnedBuffer<B: Backend, T: DeviceScalar> {
    fn raw(&self) -> &B::Buffer<T>;
    fn raw_mut(&mut self) -> &mut B::Buffer<T>;
}

struct RawGmresBuffer<B: Backend, T: DeviceScalar>(B::Buffer<T>);

impl<B, T> GmresOwnedBuffer<B, T> for RawGmresBuffer<B, T>
where
    B: Backend,
    T: DeviceScalar,
{
    fn raw(&self) -> &B::Buffer<T> {
        &self.0
    }

    fn raw_mut(&mut self) -> &mut B::Buffer<T> {
        &mut self.0
    }
}

impl<B, T> GmresOwnedBuffer<B, T> for crate::pool::Buffer<T, B>
where
    B: Backend,
    T: DeviceScalar,
{
    fn raw(&self) -> &B::Buffer<T> {
        self.raw()
    }

    fn raw_mut(&mut self) -> &mut B::Buffer<T> {
        self.raw_mut()
    }
}

trait GmresAllocator<B: Backend, T: DeviceScalar> {
    type Buffer: GmresOwnedBuffer<B, T>;

    fn alloc(&self, len: usize, label: &'static str) -> Result<Self::Buffer, GmresBackendError>;
}

struct RawGmresAllocator<'a, B: Backend>(&'a B);

impl<B, T> GmresAllocator<B, T> for RawGmresAllocator<'_, B>
where
    B: Backend,
    T: DeviceScalar,
{
    type Buffer = RawGmresBuffer<B, T>;

    fn alloc(&self, len: usize, label: &'static str) -> Result<Self::Buffer, GmresBackendError> {
        self.0
            .raw_alloc::<T>(len)
            .map(RawGmresBuffer)
            .map_err(|error| GmresBackendError::Allocation {
                label,
                reason: error.to_string(),
            })
    }
}

struct PooledGmresAllocator<'a, B: Backend>(&'a MemoryPool<B>);

impl<B, T> GmresAllocator<B, T> for PooledGmresAllocator<'_, B>
where
    B: Backend,
    T: DeviceScalar,
{
    type Buffer = crate::pool::Buffer<T, B>;

    fn alloc(&self, len: usize, label: &'static str) -> Result<Self::Buffer, GmresBackendError> {
        self.0
            .try_alloc::<T>(len)
            .map_err(|error| GmresBackendError::Allocation {
                label,
                reason: error.to_string(),
            })
    }
}

/// Solve `A x = b` with right-preconditioner `M` and starting guess
/// `x0` (zeros if `None`). All operands live on the device.
///
/// On return `out` holds the (preconditioned) solution. `out` may
/// alias `x0` (the solver allocates an internal scratch). The host
/// only sees the iteration count and final residual scalar.
///
/// # Algorithm
///
/// Right-preconditioned GMRES per Saad §9.3:
///
/// ```text
///   1. r0 = b − A x0
///   2. β = ||r0||₂; v1 = r0 / β
///   3. for j = 1, 2, …, m:
///        w = A · M⁻¹ · v_j
///        for i = 1..=j:        h_{i,j} = ⟨v_i, w⟩;   w -= h_{i,j} · v_i
///        h_{j+1,j} = ||w||;    v_{j+1} = w / h_{j+1,j}
///        update Givens rotations on column j of H
///        residual = |β · ē_{j+1}|
///        if residual / ||b|| < tol: break
///   4. solve H[1..j, 1..j] · y = β·e1 (small triangular system, host)
///   5. x = x0 + M⁻¹ · (V · y)
/// ```
/// Reusable pool-backed scratch for repeated device GMRES solves.
#[derive(Debug)]
pub struct GmresWorkspace<B: Backend, T: DeviceScalar> {
    n: usize,
    cache: Arc<Mutex<Vec<crate::pool::Buffer<T, B>>>>,
    host: Arc<Mutex<GmresHostScratch<T>>>,
    /// Reusable device coefficient/result buffers for native fused
    /// projection and update boundaries.
    device_projection: Option<crate::pool::Buffer<T, B>>,
    device_coefficients: Option<crate::pool::Buffer<T, B>>,
    basis_storage: Vec<RecycledGmresBuffer<B, T>>,
    preconditioned_storage: Vec<RecycledGmresBuffer<B, T>>,
    // Backing storage for the references passed to fused reductions. The
    // workspace entry point clears these references before basis buffers are
    // released, retaining only the allocation capacity between solves.
    basis_refs: Vec<&'static B::Buffer<T>>,
}

/// Host-side Krylov bookkeeping retained between pooled solves.
///
/// The small Hessenberg system is intentionally solved on the host, but its
/// vectors should not be reallocated on every solve.  Keeping this scratch
/// separate from the device-buffer cache lets the workspace reuse both sides
/// of the steady-state GMRES path without changing the public solve result.
#[derive(Debug, Default)]
struct GmresHostScratch<T: DeviceScalar> {
    h: Vec<Vec<T>>,
    h_col: Vec<T>,
    cs: Vec<T>,
    sn: Vec<T>,
    g: Vec<T>,
    dots1: Vec<T>,
    dots2: Vec<T>,
    neg_dots1: Vec<T>,
    neg_dots2: Vec<T>,
    y: Vec<T>,
}

impl<T: DeviceScalar> GmresHostScratch<T> {
    fn prepare(&mut self, max_iter: usize) -> Result<(), GmresBackendError> {
        let allocation_error = |what: &str| GmresBackendError::Backend {
            source: crate::BackendError::Allocation {
                reason: format!("gmres {what} host scratch allocation failed"),
            },
        };
        if self.h.len() < max_iter {
            self.h
                .try_reserve(max_iter - self.h.len())
                .map_err(|_| allocation_error("Hessenberg"))?;
            while self.h.len() < max_iter {
                self.h.push(Vec::new());
            }
        }
        let column_capacity = max_iter
            .checked_add(1)
            .ok_or_else(|| allocation_error("Hessenberg column"))?;
        for column in &mut self.h {
            column.clear();
            column
                .try_reserve(column_capacity)
                .map_err(|_| allocation_error("Hessenberg column"))?;
        }
        let vector_capacity = max_iter
            .checked_add(2)
            .ok_or_else(|| allocation_error("Krylov"))?;
        for vector in [
            &mut self.h_col,
            &mut self.cs,
            &mut self.sn,
            &mut self.g,
            &mut self.dots1,
            &mut self.dots2,
            &mut self.neg_dots1,
            &mut self.neg_dots2,
            &mut self.y,
        ] {
            vector.clear();
            vector
                .try_reserve(vector_capacity)
                .map_err(|_| allocation_error("Krylov"))?;
        }
        Ok(())
    }

    fn clear(&mut self) {
        *self = Self::default();
    }

    fn capacity(&self) -> usize {
        self.h.iter().map(Vec::capacity).sum::<usize>()
            + self.h_col.capacity()
            + self.cs.capacity()
            + self.sn.capacity()
            + self.g.capacity()
            + self.dots1.capacity()
            + self.dots2.capacity()
            + self.neg_dots1.capacity()
            + self.neg_dots2.capacity()
            + self.y.capacity()
    }
}

impl<B: Backend, T: DeviceScalar> GmresWorkspace<B, T> {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            cache: Arc::new(Mutex::new(Vec::new())),
            host: Arc::new(Mutex::new(GmresHostScratch::default())),
            device_projection: None,
            device_coefficients: None,
            basis_storage: Vec::new(),
            preconditioned_storage: Vec::new(),
            basis_refs: Vec::new(),
        }
    }

    pub fn dimension(&self) -> usize {
        self.n
    }

    /// Reserve device vectors for an entire cycle, including flexible basis storage.
    ///
    /// Unlike warming with one solve, this covers later RHS cases that need more
    /// iterations. Use the same pool for reservation and solves. Shorter restart
    /// cycles retain the larger projection buffers; this method
    /// reserves the dimension-sized vectors independently of convergence history.
    ///
    /// # Errors
    /// Rejects zero capacity or overflow and propagates quota/backend failures.
    /// Successfully reserved buffers remain cached if a later allocation fails.
    pub fn try_reserve_cycle(
        &mut self,
        pool: &MemoryPool<B>,
        max_iterations: usize,
        flexible: bool,
    ) -> Result<(), GmresBackendError> {
        let capacity_error = || GmresBackendError::Workspace {
            reason: "invalid GMRES cycle reservation capacity".into(),
        };
        if max_iterations == 0 || self.n == 0 {
            return Err(capacity_error());
        }
        // Up to m+1 Arnoldi vectors, m flexible vectors, and five residual,
        // operator/preconditioner and final-update temporaries coexist.
        let count = max_iterations
            .checked_mul(if flexible { 2 } else { 1 })
            .and_then(|v| v.checked_add(6))
            .ok_or_else(capacity_error)?;
        self.n
            .checked_mul(count)
            .and_then(|v| v.checked_mul(size_of::<T>()))
            .ok_or_else(capacity_error)?;
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| GmresBackendError::Workspace {
                reason: "workspace cache mutex is poisoned".into(),
            })?;
        let needed = count.saturating_sub(cache.iter().filter(|b| b.len() == self.n).count());
        cache
            .try_reserve(needed)
            .map_err(|error| GmresBackendError::Allocation {
                label: "GMRES reserved vector slots",
                reason: error.to_string(),
            })?;
        for _ in 0..needed {
            cache.push(pool.try_alloc::<T>(self.n).map_err(|error| {
                GmresBackendError::Allocation {
                    label: "GMRES reserved cycle vector",
                    reason: error.to_string(),
                }
            })?);
        }
        Ok(())
    }

    /// Number of retained device buffers available for reuse.
    pub fn cached_buffer_count(&self) -> Result<usize, GmresBackendError> {
        self.cache
            .lock()
            .map(|cache| cache.len())
            .map_err(|_| GmresBackendError::Workspace {
                reason: "workspace cache mutex is poisoned".to_string(),
            })
    }

    /// Bytes retained by the reusable workspace.
    pub fn cached_bytes(&self) -> Result<usize, GmresBackendError> {
        self.cache
            .lock()
            .map(|cache| {
                cache.iter().map(|buffer| buffer.byte_size()).sum::<usize>()
                    + self
                        .device_projection
                        .as_ref()
                        .map_or(0, crate::pool::Buffer::byte_size)
                    + self
                        .device_coefficients
                        .as_ref()
                        .map_or(0, crate::pool::Buffer::byte_size)
            })
            .map_err(|_| GmresBackendError::Workspace {
                reason: "workspace cache mutex is poisoned".to_string(),
            })
    }

    /// Capacity of the persistent device-side projection/update scratch.
    /// A nonzero value after warm-up confirms that the solve's coefficient
    /// boundary is backed by reusable device storage rather than a temporary
    /// allocation in the Arnoldi loop.
    pub fn cached_device_reduction_capacity(&self) -> usize {
        self.device_projection
            .as_ref()
            .map_or(0, crate::pool::Buffer::len)
            .min(
                self.device_coefficients
                    .as_ref()
                    .map_or(0, crate::pool::Buffer::len),
            )
    }

    /// Number of host scalar slots retained for Krylov bookkeeping.
    pub fn cached_host_capacity(&self) -> Result<usize, GmresBackendError> {
        self.host
            .lock()
            .map(|scratch| scratch.capacity())
            .map_err(|_| GmresBackendError::Workspace {
                reason: "workspace host scratch mutex is poisoned".to_string(),
            })
    }
    /// Capacity of retained Arnoldi basis-reference storage.
    pub fn cached_basis_reference_capacity(&self) -> usize {
        self.basis_refs.capacity()
    }

    /// Capacity retained for the Arnoldi basis vector slots themselves.
    pub fn cached_basis_slot_capacity(&self) -> usize {
        self.basis_storage.capacity()
    }

    /// Release retained scratch buffers and refund their pool reservations.
    pub fn clear(&mut self) -> Result<(), GmresBackendError> {
        self.basis_storage = Vec::new();
        self.preconditioned_storage = Vec::new();
        self.basis_refs = Vec::new();
        self.device_projection = None;
        self.device_coefficients = None;
        self.cache
            .lock()
            .map(|mut cache| cache.clear())
            .map_err(|_| GmresBackendError::Workspace {
                reason: "workspace cache mutex is poisoned".to_string(),
            })?;
        self.host
            .lock()
            .map(|mut scratch| scratch.clear())
            .map_err(|_| GmresBackendError::Workspace {
                reason: "workspace host scratch mutex is poisoned".to_string(),
            })
    }

    fn ensure_device_reductions(
        &mut self,
        pool: &MemoryPool<B>,
        max_iterations: usize,
    ) -> Result<(), GmresBackendError> {
        // Transfers use only the active prefix, so shorter restart cycles can
        // retain the largest coefficient and projection buffers seen so far.
        if self
            .device_projection
            .as_ref()
            .is_none_or(|buffer| buffer.len() < max_iterations)
        {
            self.device_projection =
                Some(pool.try_alloc::<T>(max_iterations).map_err(|error| {
                    GmresBackendError::Allocation {
                        label: "GMRES device projection",
                        reason: error.to_string(),
                    }
                })?);
        }
        if self
            .device_coefficients
            .as_ref()
            .is_none_or(|buffer| buffer.len() < max_iterations)
        {
            self.device_coefficients =
                Some(pool.try_alloc::<T>(max_iterations).map_err(|error| {
                    GmresBackendError::Allocation {
                        label: "GMRES device coefficients",
                        reason: error.to_string(),
                    }
                })?);
        }
        Ok(())
    }
}

/// Populate a reusable basis-reference slice for one backend operation.
///
/// Keeping the references scoped to the callback lets the caller mutate the
/// owning basis (for example, append the next Arnoldi vector) immediately
/// afterwards while retaining the reference Vec's allocation.
fn with_basis_refs<B, T, Buffer, R, F>(
    references: &mut Vec<&'static B::Buffer<T>>,
    basis: &[Buffer],
    count: usize,
    callback: F,
) -> R
where
    B: Backend,
    T: DeviceScalar,
    Buffer: GmresOwnedBuffer<B, T>,
    F: FnOnce(&[&B::Buffer<T>]) -> R,
{
    references.clear();
    references.extend(basis.iter().take(count).map(|buffer| {
        // SAFETY: the owning basis reserves max_iter + 1 slots and only
        // appends after this helper clears the references. The callback does
        // not retain the slice, so each reference is dead before the basis is
        // mutated or dropped.
        unsafe { &*(buffer.raw() as *const B::Buffer<T>) }
    }));
    let result = callback(references);
    references.clear();
    result
}

/// Temporarily lends a reusable basis-slot vector to one solve while
/// returning its allocation capacity on every exit path. The slot values are
/// cleared on drop, so device buffers still flow back through the allocator's
/// normal cache/RAII path.
struct BasisStorageGuard<'a, T> {
    destination: &'a mut Vec<T>,
    storage: RefCell<Vec<T>>,
}

impl<'a, T> BasisStorageGuard<'a, T> {
    fn new(destination: &'a mut Vec<T>) -> Self {
        Self {
            storage: RefCell::new(std::mem::take(destination)),
            destination,
        }
    }
}

impl<T> Drop for BasisStorageGuard<'_, T> {
    fn drop(&mut self) {
        let mut storage = self.storage.replace(Vec::new());
        storage.clear();
        *self.destination = storage;
    }
}

struct ReusablePooledGmresAllocator<'a, B: Backend, T: DeviceScalar> {
    pool: &'a MemoryPool<B>,
    cache: Arc<Mutex<Vec<crate::pool::Buffer<T, B>>>>,
}

#[derive(Debug)]
struct RecycledGmresBuffer<B: Backend, T: DeviceScalar> {
    buffer: Option<crate::pool::Buffer<T, B>>,
    cache: Arc<Mutex<Vec<crate::pool::Buffer<T, B>>>>,
}

impl<B: Backend, T: DeviceScalar> GmresOwnedBuffer<B, T> for RecycledGmresBuffer<B, T> {
    fn raw(&self) -> &B::Buffer<T> {
        self.buffer
            .as_ref()
            .expect("recycled GMRES buffer accessed after drop")
            .raw()
    }

    fn raw_mut(&mut self) -> &mut B::Buffer<T> {
        self.buffer
            .as_mut()
            .expect("recycled GMRES buffer accessed after drop")
            .raw_mut()
    }
}

impl<B: Backend, T: DeviceScalar> Drop for RecycledGmresBuffer<B, T> {
    fn drop(&mut self) {
        let Some(buffer) = self.buffer.take() else {
            return;
        };
        if let Ok(mut cache) = self.cache.lock() {
            cache.push(buffer);
        }
    }
}

impl<B: Backend, T: DeviceScalar> GmresAllocator<B, T> for ReusablePooledGmresAllocator<'_, B, T> {
    type Buffer = RecycledGmresBuffer<B, T>;

    fn alloc(&self, len: usize, label: &'static str) -> Result<Self::Buffer, GmresBackendError> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| GmresBackendError::Workspace {
                reason: "workspace cache mutex is poisoned".to_string(),
            })?;
        let cached = cache.iter().position(|buffer| buffer.len() == len);
        let buffer = if let Some(index) = cached {
            cache.swap_remove(index)
        } else {
            drop(cache);
            self.pool
                .try_alloc::<T>(len)
                .map_err(|error| GmresBackendError::Allocation {
                    label,
                    reason: error.to_string(),
                })?
        };
        Ok(RecycledGmresBuffer {
            buffer: Some(buffer),
            cache: Arc::clone(&self.cache),
        })
    }
}

/// Solve a backend-buffer GMRES system with a right preconditioner.
pub fn gmres_backend_preconditioned<B, A, P, T>(
    backend: &B,
    a: &A,
    precond: &P,
    b: &B::Buffer<T>,
    x0: Option<&B::Buffer<T>>,
    out: &mut B::Buffer<T>,
    config: &GmresBackendConfig<T>,
) -> Result<GmresBackendStatus<T>, GmresBackendError>
where
    B: Backend,
    T: DeviceScalar,
    A: BackendOperator<B, T> + ?Sized,
    P: BackendPreconditioner<B, T> + ?Sized,
{
    let allocator = RawGmresAllocator(backend);
    let mut host = GmresHostScratch::default();
    let mut basis_refs = Vec::new();
    gmres_backend_preconditioned_alloc(
        backend,
        &allocator,
        a,
        precond,
        b,
        x0,
        out,
        config,
        &mut host,
        &mut basis_refs,
        false,
        None,
        None,
        None,
        None,
    )
}

/// Pool-backed variant of [`gmres_backend_preconditioned`].
///
/// Internal Krylov basis and scratch buffers are allocated through
/// [`MemoryPool`], so repeated solver use is visible to the quota/high-water
/// accounting. Operands still use raw backend-buffer borrows, matching the
/// existing operator/preconditioner traits.
pub fn gmres_backend_preconditioned_pooled<B, A, P, T>(
    pool: &MemoryPool<B>,
    a: &A,
    precond: &P,
    b: &B::Buffer<T>,
    x0: Option<&B::Buffer<T>>,
    out: &mut B::Buffer<T>,
    config: &GmresBackendConfig<T>,
) -> Result<GmresBackendStatus<T>, GmresBackendError>
where
    B: Backend,
    T: DeviceScalar,
    A: BackendOperator<B, T> + ?Sized,
    P: BackendPreconditioner<B, T> + ?Sized,
{
    let allocator = PooledGmresAllocator(pool);
    let mut host = GmresHostScratch::default();
    let mut basis_refs = Vec::new();
    gmres_backend_preconditioned_alloc(
        pool.backend(),
        &allocator,
        a,
        precond,
        b,
        x0,
        out,
        config,
        &mut host,
        &mut basis_refs,
        false,
        None,
        None,
        None,
        None,
    )
}

/// Pool-backed GMRES using reusable device-resident Krylov scratch.
#[allow(clippy::too_many_arguments)]
pub fn gmres_backend_preconditioned_pooled_with_workspace<B, A, P, T>(
    pool: &MemoryPool<B>,
    workspace: &mut GmresWorkspace<B, T>,
    a: &A,
    precond: &P,
    b: &B::Buffer<T>,
    x0: Option<&B::Buffer<T>>,
    out: &mut B::Buffer<T>,
    config: &GmresBackendConfig<T>,
) -> Result<GmresBackendStatus<T>, GmresBackendError>
where
    B: Backend,
    T: DeviceScalar,
    A: BackendOperator<B, T> + ?Sized,
    P: BackendPreconditioner<B, T> + ?Sized,
{
    gmres_backend_workspace(pool, workspace, a, precond, b, x0, out, config, false)
}

/// Pool-backed flexible GMRES retaining each preconditioned Arnoldi vector.
///
/// The preconditioner may change between applications. The solution update uses
/// the saved vectors directly, without applying the preconditioner again.
///
/// # Errors
/// Returns checked allocation, workspace, operator, or backend failures.
pub fn fgmres_backend_preconditioned_pooled_with_workspace<B, A, P, T>(
    pool: &MemoryPool<B>,
    workspace: &mut GmresWorkspace<B, T>,
    a: &A,
    precond: &P,
    b: &B::Buffer<T>,
    x0: Option<&B::Buffer<T>>,
    out: &mut B::Buffer<T>,
    config: &GmresBackendConfig<T>,
) -> Result<GmresBackendStatus<T>, GmresBackendError>
where
    B: Backend,
    T: DeviceScalar,
    A: BackendOperator<B, T> + ?Sized,
    P: BackendPreconditioner<B, T> + ?Sized,
{
    gmres_backend_workspace(pool, workspace, a, precond, b, x0, out, config, true)
}

fn gmres_backend_workspace<B, A, P, T>(
    pool: &MemoryPool<B>,
    workspace: &mut GmresWorkspace<B, T>,
    a: &A,
    precond: &P,
    b: &B::Buffer<T>,
    x0: Option<&B::Buffer<T>>,
    out: &mut B::Buffer<T>,
    config: &GmresBackendConfig<T>,
    flexible: bool,
) -> Result<GmresBackendStatus<T>, GmresBackendError>
where
    B: Backend,
    T: DeviceScalar,
    A: BackendOperator<B, T> + ?Sized,
    P: BackendPreconditioner<B, T> + ?Sized,
{
    if workspace.n != a.n() || b.len() != workspace.n || out.len() != workspace.n {
        return Err(GmresBackendError::Workspace {
            reason: format!(
                "workspace dimension {} does not match operator {}, rhs {}, output {}",
                workspace.n,
                a.n(),
                b.len(),
                out.len()
            ),
        });
    }
    let allocator = ReusablePooledGmresAllocator {
        pool,
        cache: Arc::clone(&workspace.cache),
    };
    workspace.ensure_device_reductions(pool, config.max_iterations)?;
    let mut host = workspace
        .host
        .lock()
        .map_err(|_| GmresBackendError::Workspace {
            reason: "workspace host scratch mutex is poisoned".to_string(),
        })?;
    // Move the reference vector out while the solve runs. This keeps the
    // workspace free of references if an unexpected panic unwinds through a
    // backend implementation; the vector is restored only after it has been
    // cleared.
    let mut basis_refs = std::mem::take(&mut workspace.basis_refs);
    let result = gmres_backend_preconditioned_alloc(
        pool.backend(),
        &allocator,
        a,
        precond,
        b,
        x0,
        out,
        config,
        &mut host,
        &mut basis_refs,
        flexible,
        Some(&mut workspace.preconditioned_storage),
        Some(&mut workspace.basis_storage),
        workspace
            .device_projection
            .as_mut()
            .map(|buffer| buffer.raw_mut()),
        workspace
            .device_coefficients
            .as_mut()
            .map(|buffer| buffer.raw_mut()),
    );
    // Basis references borrow buffers owned by this solve. Clear them before
    // those buffers are returned to the reusable cache while retaining the
    // vector allocation for the next solve.
    basis_refs.clear();
    workspace.basis_refs = basis_refs;
    result
}

#[allow(clippy::too_many_arguments)]
fn gmres_backend_preconditioned_alloc<B, A, P, T, Alloc>(
    backend: &B,
    alloc: &Alloc,
    a: &A,
    precond: &P,
    b: &B::Buffer<T>,
    x0: Option<&B::Buffer<T>>,
    out: &mut B::Buffer<T>,
    config: &GmresBackendConfig<T>,
    host: &mut GmresHostScratch<T>,
    basis_refs: &mut Vec<&'static B::Buffer<T>>,
    flexible: bool,
    preconditioned_storage: Option<&mut Vec<Alloc::Buffer>>,
    basis_storage: Option<&mut Vec<Alloc::Buffer>>,
    mut device_projection: Option<&mut B::Buffer<T>>,
    device_coefficients: Option<&mut B::Buffer<T>>,
) -> Result<GmresBackendStatus<T>, GmresBackendError>
where
    B: Backend,
    T: DeviceScalar,
    A: BackendOperator<B, T> + ?Sized,
    P: BackendPreconditioner<B, T> + ?Sized,
    Alloc: GmresAllocator<B, T>,
{
    use num_traits::{Float, FromPrimitive, One, Zero};

    let n = a.n();
    if b.len() != n || out.len() != n {
        return Err(GmresBackendError::Backend {
            source: crate::BackendError::InvalidArgument {
                operation: "gmres",
                reason: format!(
                    "RHS/output lengths {}/{} must match operator dimension {n}",
                    b.len(),
                    out.len()
                ),
            },
        });
    }
    if config.max_iterations == 0
        || !config.tolerance.is_finite()
        || config.tolerance < T::Real::zero()
    {
        return Err(GmresBackendError::Backend {
            source: crate::BackendError::InvalidArgument {
                operation: "gmres",
                reason:
                    "max_iterations must be positive and tolerance must be finite and non-negative"
                        .into(),
            },
        });
    }

    let max_iter = config.max_iterations;
    let zero_real = T::Real::zero();
    let tol = if config.tolerance < zero_real {
        zero_real
    } else {
        config.tolerance
    };

    // Initial residual r0 = b − A x0. If x0 is None, x0 = 0 so r0 = b.
    let mut r0 = alloc.alloc(n, "r0")?;
    match x0 {
        None => {
            backend.try_scale::<T>(T::zero(), out)?;
            backend.try_copy_d2d(b, r0.raw_mut())?;
        }
        Some(x_init) => {
            if x_init.len() != n {
                return Err(GmresBackendError::Backend {
                    source: crate::BackendError::InvalidArgument {
                        operation: "gmres",
                        reason: format!(
                            "initial guess length {} must match operator dimension {n}",
                            x_init.len()
                        ),
                    },
                });
            }
            backend.try_copy_d2d(x_init, out)?;
            a.try_residual(backend, x_init, b, r0.raw_mut())
                .map_err(GmresBackendError::from)?;
        }
    }

    let b_norm = backend.try_nrm2_with_mode::<T>(ReductionMode::Fast, b)?;
    // Homogeneous systems are valid, including warm starts with nonzero residual.
    // Use absolute convergence when no RHS norm is available for normalization.
    let residual_scale = if b_norm == zero_real {
        T::Real::one()
    } else {
        b_norm
    };
    let beta = backend.try_nrm2_with_mode::<T>(ReductionMode::Fast, r0.raw())?;
    let initial_resid_rel = beta / residual_scale;
    if initial_resid_rel <= tol {
        return Ok(GmresBackendStatus {
            iterations: 0,
            projected_residual: initial_resid_rel,
            final_residual: initial_resid_rel,
            backward_error: initial_resid_rel,
            converged: true,
        });
    }

    // Krylov basis V — vector of device buffers, one per column.
    let required_basis_slots =
        max_iter
            .checked_add(1)
            .ok_or_else(|| GmresBackendError::Backend {
                source: crate::BackendError::Allocation {
                    reason: "gmres Krylov basis capacity overflows usize".into(),
                },
            })?;
    let mut local_basis_storage = Vec::new();
    let basis_destination = basis_storage.unwrap_or(&mut local_basis_storage);
    let basis_guard = BasisStorageGuard::new(basis_destination);
    let current_basis_capacity = basis_guard.storage.borrow().capacity();
    if current_basis_capacity < required_basis_slots {
        basis_guard
            .storage
            .borrow_mut()
            .try_reserve(required_basis_slots - current_basis_capacity)
            .map_err(|_| GmresBackendError::Backend {
                source: crate::BackendError::Allocation {
                    reason: "gmres Krylov basis allocation failed".into(),
                },
            })?;
    }
    let mut local_preconditioned = Vec::new();
    let z_guard =
        BasisStorageGuard::new(preconditioned_storage.unwrap_or(&mut local_preconditioned));
    if flexible {
        z_guard
            .storage
            .borrow_mut()
            .try_reserve(max_iter)
            .map_err(|_| GmresBackendError::Allocation {
                label: "preconditioned_basis",
                reason: "FGMRES basis slot allocation failed".into(),
            })?;
    }
    let v = &basis_guard.storage;
    let mut v0 = alloc.alloc(n, "V[0]")?;
    backend.try_copy_d2d(r0.raw(), v0.raw_mut())?;
    backend.try_scale::<T>(T::from_real(T::Real::one() / beta), v0.raw_mut())?;
    v.borrow_mut().push(v0);

    // Upper-Hessenberg/Givens/RHS scratch is retained by pooled workspaces.
    // The non-pooled entry points pass a fresh scratch value, preserving their
    // existing ownership and allocation behaviour.
    host.prepare(max_iter)?;
    // `prepare` clears these staging vecs, so resize always lands exactly on
    // `max_iter` — matching the exact-size device buffers above, whose
    // D2H/H2D transfers require length equality.
    host.g.push(T::from_real(beta));

    let mut iterations_done = 0_usize;
    let mut final_resid_rel = initial_resid_rel;

    // Workspace buffers.
    let mut w = alloc.alloc(n, "w")?;
    let mut z = alloc.alloc(n, "z")?;

    // Reuse the basis-reference slice across Arnoldi iterations. The basis
    // grows each step, but this backing allocation need not be repeated.
    basis_refs.clear();
    basis_refs
        .try_reserve(max_iter)
        .map_err(|_| GmresBackendError::Backend {
            source: crate::BackendError::Allocation {
                reason: "gmres basis-reference allocation failed".into(),
            },
        })?;

    for j in 0..max_iter {
        {
            let v_ref = v.borrow();
            precond
                .try_apply(backend, v_ref[j].raw(), z.raw_mut())
                .map_err(GmresBackendError::from)?;
        }
        a.try_apply(backend, z.raw(), w.raw_mut())
            .map_err(GmresBackendError::from)?;
        if flexible {
            let mut saved = alloc.alloc(n, "Z[j]")?;
            backend.try_copy_d2d(z.raw(), saved.raw_mut())?;
            z_guard.storage.borrow_mut().push(saved);
        }

        // CGS2 against V[0..=j] using backend-fused basis projections.
        host.h_col.clear();
        host.h_col.resize(j + 2, T::zero());
        let v_ref = v.borrow();
        with_basis_refs(basis_refs, &v_ref, j + 1, |basis_refs| {
            host.dots1.resize(j + 1, T::zero());
            if let Some(projection) = device_projection.as_deref_mut() {
                backend.try_multi_dot_with_mode_device_into::<T>(
                    ReductionMode::Fast,
                    basis_refs,
                    w.raw(),
                    projection,
                )?;
                backend.try_copy_view_d2h(
                    projection,
                    crate::BufferView::new(projection.len(), 0, j + 1, 1)?,
                    &mut host.dots1,
                )?;
                // The native projection is already in the exact coefficient
                // buffer consumed by the fused update. Negating it in place
                // removes the per-Arnoldi-iteration H2D control transfer.
                backend.try_scale(T::zero() - T::one(), projection)?;
                backend.try_axpy_many_device::<T>(projection, basis_refs, w.raw_mut())?;
            } else {
                backend.try_multi_dot_with_mode_into::<T>(
                    ReductionMode::Fast,
                    basis_refs,
                    w.raw(),
                    &mut host.dots1,
                )?;
            }
            host.neg_dots1.clear();
            host.neg_dots1
                .extend(host.dots1.iter().map(|&h_ij| T::zero() - h_ij));
            if device_projection.is_none() {
                backend.try_axpy_many::<T>(&host.neg_dots1, basis_refs, w.raw_mut())?;
            }
            host.h_col[..=j].copy_from_slice(&host.dots1[..=j]);

            host.dots2.resize(j + 1, T::zero());
            if let Some(projection) = device_projection.as_deref_mut() {
                backend.try_multi_dot_with_mode_device_into::<T>(
                    ReductionMode::Fast,
                    basis_refs,
                    w.raw(),
                    projection,
                )?;
                backend.try_copy_view_d2h(
                    projection,
                    crate::BufferView::new(projection.len(), 0, j + 1, 1)?,
                    &mut host.dots2,
                )?;
                backend.try_scale(T::zero() - T::one(), projection)?;
                backend.try_axpy_many_device::<T>(projection, basis_refs, w.raw_mut())?;
            } else {
                backend.try_multi_dot_with_mode_into::<T>(
                    ReductionMode::Fast,
                    basis_refs,
                    w.raw(),
                    &mut host.dots2,
                )?;
            }
            host.neg_dots2.clear();
            host.neg_dots2
                .extend(host.dots2.iter().map(|&h_ij| T::zero() - h_ij));
            if device_projection.is_none() {
                backend.try_axpy_many::<T>(&host.neg_dots2, basis_refs, w.raw_mut())?;
            }
            Ok::<_, GmresBackendError>(())
        })?;
        drop(v_ref);
        for (slot, &dot) in host.h_col.iter_mut().zip(host.dots2.iter()).take(j + 1) {
            *slot = *slot + dot;
        }
        let h_jp1 = backend.try_nrm2_with_mode::<T>(ReductionMode::Fast, w.raw())?;
        host.h_col[j + 1] = T::from_real(h_jp1);
        let invariant_subspace = h_jp1 == zero_real;
        // V[j+1] = w / h_{j+1,j}
        if !invariant_subspace {
            let mut v_next = alloc.alloc(n, "V[j+1]")?;
            backend.try_copy_d2d(w.raw(), v_next.raw_mut())?;
            backend.try_scale::<T>(T::from_real(T::Real::one() / h_jp1), v_next.raw_mut())?;
            v.borrow_mut().push(v_next);
        }

        // Apply previous Givens rotations to column j of H.
        // For complex T, the rotation's lower-left coefficient is
        // `-conj(s)`; for real T `conj(s) = s` so the formula below
        // reduces to the standard real Givens.
        for i in 0..j {
            let s_conj = host.sn[i].conj_mul(T::one()); // = conj(sn[i])
            let temp = host.cs[i] * host.h_col[i] + host.sn[i] * host.h_col[i + 1];
            host.h_col[i + 1] =
                (T::zero() - s_conj) * host.h_col[i] + host.cs[i] * host.h_col[i + 1];
            host.h_col[i] = temp;
        }

        if invariant_subspace {
            let diagonal_norm = host.h_col[j].magnitude();
            if !diagonal_norm.is_finite() {
                return Err(GmresBackendError::Backend {
                    source: crate::BackendError::OperationFailed {
                        operation: "gmres",
                        reason: format!("Hessenberg diagonal at iteration {} is non-finite", j + 1),
                    },
                });
            }
            if diagonal_norm == zero_real {
                return Err(GmresBackendError::HappyBreakdown { iteration: j + 1 });
            }
            host.h[j].clear();
            host.h[j].extend_from_slice(&host.h_col);
            iterations_done = j + 1;
            final_resid_rel = zero_real;
            break;
        }

        // Compute new Givens rotation to zero h_col[j+1]. Canonical
        // complex form: c real (≥ 0), s complex.
        let abs_a = host.h_col[j].magnitude();
        let abs_b = host.h_col[j + 1].magnitude();
        let r = abs_a.hypot(abs_b);

        let (cos_j, sin_j): (T, T) = if r == zero_real {
            (T::one(), T::zero())
        } else if abs_a == zero_real {
            // a = 0: rotate so [0; b] -> [|b|·sign(b); 0].
            // c = 0, s = b / |b|.
            (T::zero(), host.h_col[j + 1] / T::from_real(abs_b))
        } else {
            // c = |a|/r (real)
            let c = T::from_real(abs_a / r);
            // s = (a/|a|) · conj(b) / r
            let tau = host.h_col[j] / T::from_real(abs_a);
            let conj_b = host.h_col[j + 1].conj_mul(T::one()); // = conj(b)
            let s = tau * conj_b / T::from_real(r);
            (c, s)
        };
        host.cs.push(cos_j);
        host.sn.push(sin_j);
        // After this Givens, the new diagonal entry is c·a + s·b which
        // in canonical form has magnitude r and argument equal to a's.
        host.h_col[j] = cos_j * host.h_col[j] + sin_j * host.h_col[j + 1];
        host.h_col[j + 1] = T::zero();

        // Update RHS g: same canonical rotation.
        let g_j = host.g[j];
        let s_conj = sin_j.conj_mul(T::one());
        host.g.push((T::zero() - s_conj) * g_j);
        host.g[j] = cos_j * g_j;

        host.h[j].clear();
        host.h[j].extend_from_slice(&host.h_col);
        iterations_done = j + 1;

        // Residual = |g[j+1]| (note: norm of the complex/real entry).
        let resid_abs = host.g[j + 1].magnitude();
        let resid_rel = resid_abs / residual_scale;
        final_resid_rel = resid_rel;
        if resid_rel <= tol {
            break;
        }
    }

    let converged = final_resid_rel <= tol;
    let k = iterations_done;
    if k == 0 {
        return Ok(GmresBackendStatus {
            iterations: 0,
            projected_residual: final_resid_rel,
            final_residual: final_resid_rel,
            backward_error: final_resid_rel,
            converged,
        });
    }

    // Back-substitution on H[0..k, 0..k] · y = g[0..k]  (T arithmetic).
    host.y.resize(k, T::zero());
    for i in (0..k).rev() {
        let mut sum = host.g[i];
        for j in (i + 1)..k {
            sum = sum - host.h[j][i] * host.y[j];
        }
        host.y[i] = sum / host.h[i][i];
    }

    // Stationary: combine V then apply M. Flexible: combine saved Z directly.
    let mut dx_unprec = alloc.alloc(n, "dx_unprec")?;
    backend.try_scale::<T>(T::zero(), dx_unprec.raw_mut())?;
    let v_ref = if flexible {
        z_guard.storage.borrow()
    } else {
        v.borrow()
    };
    with_basis_refs(basis_refs, &v_ref, k, |basis_refs| {
        if let Some(coefficients) = device_coefficients {
            backend.try_copy_view_h2d(
                &host.y,
                coefficients,
                crate::BufferView::new(coefficients.len(), 0, k, 1)?,
            )?;
            backend.try_axpy_many_device::<T>(coefficients, basis_refs, dx_unprec.raw_mut())
        } else {
            backend.try_axpy_many::<T>(&host.y, basis_refs, dx_unprec.raw_mut())
        }
    })?;
    drop(v_ref);

    if flexible {
        backend.try_axpy::<T>(T::one(), dx_unprec.raw(), out)?;
    } else {
        // A stationary linear preconditioner commutes with the basis combination.
        let mut dx = alloc.alloc(n, "dx")?;
        precond
            .try_apply(backend, dx_unprec.raw(), dx.raw_mut())
            .map_err(GmresBackendError::from)?;
        backend.try_axpy::<T>(T::one(), dx.raw(), out)?;
    }

    // The Hessenberg/Givens estimate is useful for the inner loop, but it
    // is not a sufficient convergence contract: loss of orthogonality and
    // a non-normal/right-preconditioned operator can make it optimistic.
    // Certify the published result against the original operator before
    // returning, reusing the initial-residual buffer to avoid another
    // allocation on the Krylov hot path.
    a.try_residual(backend, out, b, r0.raw_mut())
        .map_err(GmresBackendError::from)?;
    let true_residual = backend.try_nrm2_with_mode::<T>(ReductionMode::Fast, r0.raw())?;
    if !true_residual.is_finite() {
        return Err(GmresBackendError::Backend {
            source: crate::BackendError::OperationFailed {
                operation: "gmres",
                reason: "final residual is non-finite".into(),
            },
        });
    }
    let true_residual_rel = true_residual / residual_scale;
    let converged = true_residual_rel <= tol;
    // Reuse the work vector for ||A*x|| so the backend result carries the
    // same normwise backward-error contract as the host solver.  This is a
    // device operation and does not introduce an intermediate vector
    // transfer on CUDA.
    a.try_apply(backend, out, w.raw_mut())
        .map_err(GmresBackendError::from)?;
    let ax_norm = backend.try_nrm2_with_mode::<T>(ReductionMode::Fast, w.raw())?;
    if !ax_norm.is_finite() {
        return Err(GmresBackendError::Backend {
            source: crate::BackendError::OperationFailed {
                operation: "gmres",
                reason: "backward-error operator norm is non-finite".into(),
            },
        });
    }
    let denominator = (ax_norm + b_norm).max(T::Real::one());
    let backward_error = true_residual / denominator;
    let _: Option<usize> = <usize as FromPrimitive>::from_usize(0);
    Ok(GmresBackendStatus {
        iterations: iterations_done,
        projected_residual: final_resid_rel,
        final_residual: true_residual_rel,
        backward_error,
        converged,
    })
}

#[cfg(test)]
// Fixtures intentionally use the panicking convenience constructors on
// tiny fixed matrices; production paths must use the `try_*` variants.
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::cpu::CpuBackend;

    /// Tridiagonal `[-1, 2, -1]` — SPD, well-conditioned 5×5.
    fn tridiag_5x5() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
        let row_ptr = vec![0, 2, 5, 8, 11, 13];
        let col_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4];
        let values = vec![
            2.0, -1.0, // row 0
            -1.0, 2.0, -1.0, // row 1
            -1.0, 2.0, -1.0, // row 2
            -1.0, 2.0, -1.0, // row 3
            -1.0, 2.0, // row 4
        ];
        (row_ptr, col_idx, values)
    }

    fn make_buf<T>(backend: &CpuBackend, host: &[T]) -> crate::cpu::HostBuffer<T>
    where
        T: crate::scalar::BufferElement,
    {
        let mut b = backend.raw_alloc::<T>(host.len()).unwrap();
        backend.copy_h2d(host, &mut b);
        b
    }

    #[test]
    fn gmres_backend_arnoldi_uses_fused_basis_ops() {
        let source = include_str!("gmres.rs");
        let body = source
            .split("for j in 0..max_iter")
            .nth(1)
            .expect("GMRES Arnoldi loop should exist")
            .split("// Apply previous Givens rotations")
            .next()
            .expect("Arnoldi orthogonalization should precede Givens rotations");

        assert!(
            body.contains("try_multi_dot_with_mode::<T>")
                || body.contains("try_multi_dot_with_mode_into::<T>"),
            "backend GMRES Arnoldi projection should use checked fused multi_dot"
        );
        assert!(
            body.contains("try_axpy_many::<T>"),
            "backend GMRES Arnoldi projection should use checked fused axpy_many"
        );
        assert!(
            !body.contains("backend.dot::<T>"),
            "backend GMRES Arnoldi projection should not use serial backend.dot calls"
        );
    }

    #[test]
    fn gmres_backend_initial_residual_uses_operator_residual_contract() {
        let source = include_str!("gmres.rs");
        let body = source
            .split("match x0")
            .nth(1)
            .expect("initial residual branch should exist")
            .split("let b_norm")
            .next()
            .expect("initial residual branch should precede b_norm computation");

        assert!(
            body.contains(".try_residual(") || body.contains(".residual("),
            "nonzero initial guess should use the DeviceOperator residual contract"
        );
        assert!(
            !body.contains(".scale::<T>(-T::one()"),
            "initial residual should not open-code residual scale"
        );
    }

    #[test]
    fn cpu_backend_gmres_solves_tridiagonal_with_identity_preconditioner() {
        let backend = CpuBackend::new();
        let (row_ptr, col_idx, values) = tridiag_5x5();
        let a = BackendCsr::from_host(&backend, 5, 5, &row_ptr, &col_idx, &values);
        let b_host = [1.0_f64, 0.0, 0.0, 0.0, 1.0];
        let b = make_buf(&backend, &b_host);
        let mut out = backend.raw_alloc::<f64>(5).unwrap();

        let status = gmres_backend_preconditioned(
            &backend,
            &a,
            &IdentityBackendPreconditioner,
            &b,
            None,
            &mut out,
            &GmresBackendConfig {
                max_iterations: 50,
                tolerance: 1.0e-12,
            },
        )
        .expect("GMRES solve");

        assert!(
            status.converged,
            "expected convergence; status = {:?}",
            status
        );
        assert!(status.iterations <= 5, "tridiagonal SPD converges in ≤ n");
        let mut x_host = vec![0.0; 5];
        backend.copy_d2h(&out, &mut x_host);
        // Verify residual: ||b − A x|| / ||b|| < 1e-9
        let mut ax = backend.raw_alloc::<f64>(5).unwrap();
        let a_p = a.prebake(&backend);
        backend.spmv_prebaked::<f64>(&a_p, &out, &mut ax);
        let mut r_host = vec![0.0; 5];
        backend.copy_d2h(&ax, &mut r_host);
        let resid: f64 = b_host
            .iter()
            .zip(r_host.iter())
            .map(|(bi, axi)| (bi - axi).powi(2))
            .sum::<f64>()
            .sqrt();
        let bnorm: f64 = b_host.iter().map(|v| v * v).sum::<f64>().sqrt();
        let true_residual_rel = resid / bnorm;
        assert!(
            true_residual_rel < 1.0e-9,
            "residual not converged: {} ({:?})",
            true_residual_rel,
            x_host
        );
        assert!(
            (status.final_residual - true_residual_rel).abs() < 1.0e-12,
            "reported residual must be the original-operator residual: reported={} true={}",
            status.final_residual,
            true_residual_rel
        );
    }

    #[test]
    fn backend_gmres_givens_rotation_does_not_square_large_finite_values() {
        let backend = CpuBackend::new();
        let scale = f64::MAX / 4.0;
        let a = BackendCsr::from_host(
            &backend,
            2,
            2,
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[2.0 * scale, scale, scale, 2.0 * scale],
        );
        let b = make_buf(&backend, &[1.0, 0.0]);
        let mut out = backend.raw_alloc::<f64>(2).expect("output");

        let status = gmres_backend_preconditioned(
            &backend,
            &a,
            &IdentityBackendPreconditioner,
            &b,
            None,
            &mut out,
            &GmresBackendConfig {
                max_iterations: 1,
                tolerance: 0.5,
            },
        )
        .expect("large finite Hessenberg values must remain representable");
        let mut solution = [0.0; 2];
        backend.copy_d2h(&out, &mut solution);

        assert!(status.converged);
        assert!(status.final_residual.is_finite() && status.final_residual > 0.0);
        assert!(solution.iter().all(|value| value.is_finite()));
        assert!(solution[0] > 0.0);
    }

    #[test]
    fn backend_gmres_reports_rank_deficient_arnoldi_breakdown() {
        let backend = CpuBackend::new();
        let a = BackendCsr::from_host(&backend, 2, 2, &[0, 0, 0], &[], &[]);
        let b = make_buf(&backend, &[1.0, 0.0]);
        let mut out = backend.raw_alloc::<f64>(2).expect("output");

        let error = gmres_backend_preconditioned(
            &backend,
            &a,
            &IdentityBackendPreconditioner,
            &b,
            None,
            &mut out,
            &GmresBackendConfig {
                max_iterations: 2,
                tolerance: 1.0e-12,
            },
        )
        .expect_err("a zero operator cannot solve a non-zero right-hand side");

        assert!(matches!(
            error,
            GmresBackendError::HappyBreakdown { iteration: 1 }
        ));
        let mut solution = [f64::NAN; 2];
        backend.copy_d2h(&out, &mut solution);
        assert!(solution.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn pooled_backend_gmres_counts_internal_workspace() {
        let pool = crate::MemoryPool::new(
            CpuBackend::new(),
            crate::MemoryPoolConfig::with_cap(1 << 20),
        )
        .unwrap();
        let backend = pool.backend();
        let (row_ptr, col_idx, values) = tridiag_5x5();
        let a = BackendCsr::from_host(backend, 5, 5, &row_ptr, &col_idx, &values);
        let b_host = [1.0_f64, 0.0, 0.0, 0.0, 1.0];
        let mut b = pool.try_alloc::<f64>(5).unwrap();
        let mut out = pool.try_alloc::<f64>(5).unwrap();
        backend.copy_h2d(&b_host, b.raw_mut());
        let high_water_before_solve = pool.high_water_bytes();

        let status = gmres_backend_preconditioned_pooled(
            &pool,
            &a,
            &IdentityBackendPreconditioner,
            b.raw(),
            None,
            out.raw_mut(),
            &GmresBackendConfig {
                max_iterations: 50,
                tolerance: 1.0e-12,
            },
        )
        .expect("pooled GMRES solve");

        assert!(status.converged);
        assert!(
            pool.high_water_bytes() > high_water_before_solve,
            "pooled GMRES should charge internal workspace to MemoryPool"
        );
    }

    #[test]
    fn pooled_backend_gmres_workspace_reuses_device_buffers() {
        let pool = crate::MemoryPool::new(
            CpuBackend::new(),
            crate::MemoryPoolConfig::with_cap(1 << 20),
        )
        .unwrap();
        let backend = pool.backend();
        let (row_ptr, col_idx, values) = tridiag_5x5();
        let a = BackendCsr::from_host(backend, 5, 5, &row_ptr, &col_idx, &values);
        let mut b = pool.try_alloc::<f64>(5).unwrap();
        let mut out = pool.try_alloc::<f64>(5).unwrap();
        backend.copy_h2d(&[1.0, 0.0, 0.0, 0.0, 1.0], b.raw_mut());
        let bytes_before_workspace = pool.bytes_in_use();
        let mut workspace = GmresWorkspace::new(5);
        let config = GmresBackendConfig {
            max_iterations: 50,
            tolerance: 1.0e-12,
        };

        let first = gmres_backend_preconditioned_pooled_with_workspace(
            &pool,
            &mut workspace,
            &a,
            &IdentityBackendPreconditioner,
            b.raw(),
            None,
            out.raw_mut(),
            &config,
        )
        .expect("first workspace solve");
        assert!(first.converged);
        let cached_after_first = workspace.cached_buffer_count().unwrap();
        let host_capacity_after_first = workspace.cached_host_capacity().unwrap();
        let device_reduction_capacity_after_first = workspace.cached_device_reduction_capacity();
        let basis_reference_capacity_after_first = workspace.cached_basis_reference_capacity();
        let basis_slot_capacity_after_first = workspace.cached_basis_slot_capacity();
        let high_water_after_first = pool.high_water_bytes();
        let bytes_after_first = pool.bytes_in_use();
        assert!(cached_after_first > 0);
        assert!(host_capacity_after_first > 0);
        assert_eq!(device_reduction_capacity_after_first, 50);
        assert!(basis_reference_capacity_after_first > 0);
        assert!(basis_slot_capacity_after_first > 0);
        assert!(workspace.cached_bytes().unwrap() > 0);
        assert!(bytes_after_first > bytes_before_workspace);

        let second = gmres_backend_preconditioned_pooled_with_workspace(
            &pool,
            &mut workspace,
            &a,
            &IdentityBackendPreconditioner,
            b.raw(),
            None,
            out.raw_mut(),
            &config,
        )
        .expect("second workspace solve");
        assert!(second.converged);
        assert_eq!(
            workspace.cached_host_capacity().unwrap(),
            host_capacity_after_first,
            "pooled GMRES should reuse host Krylov scratch after warm-up"
        );
        assert_eq!(
            workspace.cached_device_reduction_capacity(),
            device_reduction_capacity_after_first,
            "pooled GMRES should reuse device projection/update scratch after warm-up"
        );
        assert_eq!(
            workspace.cached_basis_reference_capacity(),
            basis_reference_capacity_after_first,
            "pooled GMRES should reuse basis-reference storage after warm-up"
        );
        assert_eq!(
            workspace.cached_basis_slot_capacity(),
            basis_slot_capacity_after_first,
            "pooled GMRES should reuse basis-slot storage after warm-up"
        );
        assert_eq!(workspace.cached_buffer_count().unwrap(), cached_after_first);
        assert_eq!(pool.high_water_bytes(), high_water_after_first);
        assert_eq!(pool.bytes_in_use(), bytes_after_first);

        let failed = gmres_backend_preconditioned_pooled_with_workspace(
            &pool,
            &mut workspace,
            &a,
            &IdentityBackendPreconditioner,
            b.raw(),
            None,
            out.raw_mut(),
            &GmresBackendConfig {
                max_iterations: 0,
                tolerance: 1.0e-12,
            },
        );
        assert!(matches!(
            failed,
            Err(GmresBackendError::Backend {
                source: crate::BackendError::InvalidArgument { .. }
            })
        ));
        assert_eq!(
            workspace.cached_basis_reference_capacity(),
            basis_reference_capacity_after_first,
            "typed GMRES errors should restore reusable basis-reference storage"
        );
        assert_eq!(
            workspace.cached_basis_slot_capacity(),
            basis_slot_capacity_after_first,
            "typed GMRES errors should restore reusable basis-slot storage"
        );

        workspace.clear().expect("clear reusable workspace");
        assert_eq!(pool.bytes_in_use(), bytes_before_workspace);
        assert_eq!(workspace.cached_basis_slot_capacity(), 0);
    }

    #[test]
    fn pooled_backend_gmres_propagates_workspace_quota_failure() {
        let pool = crate::MemoryPool::new(
            CpuBackend::new(),
            crate::MemoryPoolConfig::with_cap(2 * 5 * std::mem::size_of::<f64>()),
        )
        .unwrap();
        let backend = pool.backend();
        let (row_ptr, col_idx, values) = tridiag_5x5();
        let a = BackendCsr::from_host(backend, 5, 5, &row_ptr, &col_idx, &values);
        let mut b = pool.try_alloc::<f64>(5).unwrap();
        let mut out = pool.try_alloc::<f64>(5).unwrap();
        backend.copy_h2d(&[1.0, 0.0, 0.0, 0.0, 1.0], b.raw_mut());

        let result = gmres_backend_preconditioned_pooled(
            &pool,
            &a,
            &IdentityBackendPreconditioner,
            b.raw(),
            None,
            out.raw_mut(),
            &GmresBackendConfig::default(),
        );

        match result {
            Err(GmresBackendError::Allocation { label, reason }) => {
                assert_eq!(label, "r0");
                assert!(
                    reason.contains("quota exceeded"),
                    "unexpected error: {reason}"
                );
            }
            other => panic!("expected a typed workspace allocation error, got {other:?}"),
        }
    }

    #[test]
    fn gmres_handles_zero_rhs_with_zero_initial_guess() {
        let backend = CpuBackend::new();
        let (row_ptr, col_idx, values) = tridiag_5x5();
        let a = BackendCsr::from_host(&backend, 5, 5, &row_ptr, &col_idx, &values);
        let b = make_buf::<f64>(&backend, &[0.0; 5]);
        let mut out = backend.raw_alloc::<f64>(5).unwrap();

        let result = gmres_backend_preconditioned(
            &backend,
            &a,
            &IdentityBackendPreconditioner,
            &b,
            None,
            &mut out,
            &GmresBackendConfig::default(),
        );
        let status = result.unwrap();
        assert!(status.converged);
        assert_eq!(status.iterations, 0);
        assert_eq!(status.final_residual, 0.0);
        let mut solution = [1.0; 5];
        backend.copy_d2h(&out, &mut solution);
        assert_eq!(solution, [0.0; 5]);
    }

    #[test]
    fn gmres_zero_rhs_corrects_nonzero_initial_guess() {
        let backend = CpuBackend::new();
        let (row_ptr, col_idx, values) = tridiag_5x5();
        let a = BackendCsr::from_host(&backend, 5, 5, &row_ptr, &col_idx, &values);
        let b = make_buf::<f64>(&backend, &[0.0; 5]);
        let initial = make_buf::<f64>(&backend, &[1.0, -0.5, 0.7, 0.2, -0.9]);
        let mut out = backend.raw_alloc::<f64>(5).unwrap();
        let status = gmres_backend_preconditioned(
            &backend,
            &a,
            &IdentityBackendPreconditioner,
            &b,
            Some(&initial),
            &mut out,
            &GmresBackendConfig {
                max_iterations: 10,
                tolerance: 1e-10,
            },
        )
        .unwrap();
        assert!(status.converged);
        assert!(status.iterations > 0);
        assert!(status.final_residual < 1e-10);
        let mut solution = [1.0; 5];
        backend.copy_d2h(&out, &mut solution);
        assert!(solution.iter().all(|v| v.abs() < 1e-10));
    }

    #[test]
    fn gmres_rejects_invalid_iteration_budget_and_tolerance() {
        let backend = CpuBackend::new();
        let (row_ptr, col_idx, values) = tridiag_5x5();
        let a = BackendCsr::from_host(&backend, 5, 5, &row_ptr, &col_idx, &values);
        let b = make_buf::<f64>(&backend, &[1.0, 0.0, 0.0, 0.0, 1.0]);
        let mut out = backend.raw_alloc::<f64>(5).unwrap();

        for config in [
            GmresBackendConfig {
                max_iterations: 0,
                tolerance: 1.0e-8,
            },
            GmresBackendConfig {
                max_iterations: 5,
                tolerance: f64::NAN,
            },
        ] {
            let result = gmres_backend_preconditioned(
                &backend,
                &a,
                &IdentityBackendPreconditioner,
                &b,
                None,
                &mut out,
                &config,
            );
            assert!(matches!(
                result,
                Err(GmresBackendError::Backend {
                    source: crate::BackendError::InvalidArgument { .. }
                })
            ));
        }
    }

    #[test]
    fn from_host_round_trip_preserves_csr_structure() {
        let backend = CpuBackend::new();
        let (row_ptr, col_idx, values) = tridiag_5x5();
        let a = BackendCsr::from_host(&backend, 5, 5, &row_ptr, &col_idx, &values);

        let mut row_ptr_back = vec![0_i32; row_ptr.len()];
        let mut col_idx_back = vec![0_i32; col_idx.len()];
        let mut values_back = vec![0.0_f64; values.len()];
        backend.copy_d2h(&a.row_ptr, &mut row_ptr_back);
        backend.copy_d2h(&a.col_idx, &mut col_idx_back);
        backend.copy_d2h(&a.values, &mut values_back);
        assert_eq!(row_ptr_back, row_ptr);
        assert_eq!(col_idx_back, col_idx);
        for (a, b) in values_back.iter().zip(values.iter()) {
            assert!((a - b).abs() < 1.0e-15);
        }
    }
}

#[test]
fn gmres_backend_reuses_basis_reference_storage() {
    let source = include_str!("gmres.rs");
    let implementation = source
        .split("#[cfg(test)]")
        .next()
        .expect("GMRES implementation should precede tests");
    let allocator_body = implementation
        .split("fn gmres_backend_preconditioned_alloc")
        .nth(1)
        .expect("GMRES allocator helper should exist");
    assert!(
        allocator_body.contains("basis_refs: &mut Vec<&'static B::Buffer<T>>"),
        "basis references should be supplied by the caller"
    );
    let body = allocator_body
        .split("for j in 0..max_iter")
        .nth(1)
        .expect("GMRES Arnoldi loop should exist")
        .split("// Apply previous Givens rotations")
        .next()
        .expect("basis-reference storage should be in Arnoldi loop");
    assert!(
        body.contains("with_basis_refs"),
        "Arnoldi basis references should be scoped through the reusable helper"
    );
    assert!(
        implementation.matches("with_basis_refs(basis_refs").count() >= 2,
        "Arnoldi and solution update should share the reusable helper"
    );
    assert!(
        !body.contains("let mut basis_refs")
            && !body.contains("Vec<&B::Buffer<T>> = v.iter().take(j + 1)")
            && !body.contains(".take(j + 1).map(|buf| buf.raw()).collect()"),
        "Arnoldi iterations must not allocate a fresh reference Vec"
    );
}

#[cfg(test)]
mod reservation_tests;
