//! Device-backed block GMRES.
//!
//! The operator and preconditioner see a contiguous column-major multi-vector
//! and may use a native SpMM implementation.  The large block Arnoldi basis,
//! orthogonalisation vectors, and least-squares candidate updates remain on
//! the backend, including the initial residual and QR. Native validation and
//! zeroing hooks avoid full-vector initialization transfers when available;
//! other backends use checked host staging. Small QR/Krylov coefficients still
//! cross to the host. Scalar recovery and final solution assembly use device
//! copies, with retained scratch for repeated room FEM and BEM loads.
//!
//! The returned convergence values are always recomputed from `b - A*x` after
//! each block step.  They are therefore true residuals, not only projected
//! block-GMRES estimates.

mod shapes;
use shapes::BlockShape;

use crate::gmres::{
    GmresBackendConfig, GmresBackendError, GmresBackendStatus, GmresWorkspace,
    gmres_backend_preconditioned, gmres_backend_preconditioned_pooled_with_workspace,
};
use crate::operator::{DeviceOperator, DevicePreconditioner};
use crate::{Backend, BufferView, DeviceBufferLike, DeviceScalar, ReductionMode};
use num_traits::{Float, FromPrimitive, One, Zero};

/// Configuration for one unrestarted block-GMRES solve.
#[derive(Debug, Clone)]
pub struct BlockGmresConfig<R = f64>
where
    R: Float + FromPrimitive + Copy,
{
    /// Number of block Arnoldi steps. One step advances all right-hand sides.
    pub max_block_iterations: usize,
    /// Relative true-residual tolerance, applied independently per RHS.
    pub tolerance: R,
}

impl<R> Default for BlockGmresConfig<R>
where
    R: Float + FromPrimitive + Copy,
{
    fn default() -> Self {
        Self {
            max_block_iterations: 30,
            // `FromPrimitive` is fallible for generic `Float` implementations.
            // Keep the default total for scalar types whose representation cannot
            // encode the requested decimal literal instead of panicking during
            // configuration construction.
            tolerance: R::from_f64(1.0e-8).unwrap_or_else(|| R::epsilon().sqrt()),
        }
    }
}

/// Caller-owned scratch for repeated block-GMRES solves with the same backend
/// and scalar family. Device buffers are retained at the exact full-RHS and
/// active-block shapes used by the most recent solve; changing either shape
/// replaces that class of buffers on the next call. Scalar fallback column
/// buffers are retained independently. Use `with_fallback_pool` to retain
/// scalar Krylov scratch as well, or `with_retained_shapes` to also cache
/// previously visited active-block widths until `clear`.
pub struct BlockGmresWorkspace<B: Backend, T: DeviceScalar> {
    n: usize,
    full_len: usize,
    block_len: usize,
    full_x: Option<B::Buffer<T>>,
    full_ax: Option<B::Buffer<T>>,
    block_x: Option<B::Buffer<T>>,
    block_v: Option<B::Buffer<T>>,
    block_z: Option<B::Buffer<T>>,
    block_w: Option<B::Buffer<T>>,
    block_ax: Option<B::Buffer<T>>,
    /// Active block RHS, initial solution, and residual column scratch.  The
    /// first two remain on the backend for the whole solve; `block_column` is
    /// reused when a native backend exposes a contiguous column operation.
    block_b: Option<B::Buffer<T>>,
    block_x_initial: Option<B::Buffer<T>>,
    block_column: Option<B::Buffer<T>>,
    block_residual: Option<B::Buffer<T>>,
    /// Resident block Arnoldi basis.  Each entry contains `block_size`
    /// contiguous columns of length `n`; the flat mirrors are used by the
    /// backend strided projection/update primitives.
    resident_block_len: usize,
    resident_basis: Vec<B::Buffer<T>>,
    resident_z: Vec<B::Buffer<T>>,
    resident_v_flat: Option<B::Buffer<T>>,
    resident_z_flat: Option<B::Buffer<T>>,
    /// Resident projection and update coefficient workspaces.  Their size is
    /// the maximum flattened block basis count, so the inner recurrence can
    /// reuse them for every target column.
    resident_projection: Option<B::Buffer<T>>,
    resident_coefficients: Option<B::Buffer<T>>,
    host: BlockGmresHostWorkspace<T>,
    fallback: BlockFallback<B, T>,
    retain_active_shapes: bool,
    cached_shapes: Vec<BlockShape<B, T>>,
}

struct BlockFallback<B: Backend, T: DeviceScalar> {
    columns: Option<[B::Buffer<T>; 3]>,
    pool: Option<crate::MemoryPool<B>>,
    krylov: GmresWorkspace<B, T>,
}

/// Reusable host-side vectors for the unavoidable staging and residual work
/// around the device-backed block recurrence.  Keeping these buffers with the
/// caller-owned workspace avoids allocating `O(n * rhs)` arrays on every
/// repeated solve.  Arnoldi basis and Hessenberg storage are retained too;
/// their iteration-dependent lengths are cleared and rebuilt without dropping
/// capacity between solves.
struct BlockGmresHostWorkspace<T: DeviceScalar> {
    full_len: usize,
    block_len: usize,
    b_full: Vec<T>,
    x_full: Vec<T>,
    b_norms: Vec<T::Real>,
    residual_norms: Vec<T::Real>,
    active: Vec<usize>,
    best_residuals: Vec<T::Real>,
    /// Krylov basis and Hessenberg storage retained across repeated solves.
    v_blocks: Vec<Vec<T>>,
    z_blocks: Vec<Vec<T>>,
    h_blocks: Vec<Vec<Vec<T>>>,
    h_scalar: Vec<T>,
    rhs_scalar: Vec<T>,
    full_residuals: Vec<T::Real>,
    /// Per-step host temporaries retained across repeated solves.
    w_scratch: Vec<T>,
    ax_scratch: Vec<T>,
    residual_scratch: Vec<T>,
    /// QR and dense small-system scratch retained across Arnoldi steps.
    qr_r: Vec<T>,
    initial_r: Vec<T>,
    block_inner: Vec<T>,
    least_squares_normal: Vec<T>,
    least_squares_rhs: Vec<T>,
    least_squares_solution: Vec<T>,
}

impl<T: DeviceScalar> BlockGmresHostWorkspace<T> {
    fn new() -> Self {
        Self {
            full_len: 0,
            block_len: 0,
            b_full: Vec::new(),
            x_full: Vec::new(),
            b_norms: Vec::new(),
            residual_norms: Vec::new(),
            active: Vec::new(),
            best_residuals: Vec::new(),
            v_blocks: Vec::new(),
            z_blocks: Vec::new(),
            h_blocks: Vec::new(),
            h_scalar: Vec::new(),
            rhs_scalar: Vec::new(),
            full_residuals: Vec::new(),
            w_scratch: Vec::new(),
            ax_scratch: Vec::new(),
            residual_scratch: Vec::new(),
            qr_r: Vec::new(),
            initial_r: Vec::new(),
            block_inner: Vec::new(),
            least_squares_normal: Vec::new(),
            least_squares_rhs: Vec::new(),
            least_squares_solution: Vec::new(),
        }
    }

    fn ensure_full(&mut self, total: usize, rhs: usize) -> Result<(), BlockGmresError> {
        reserve_block_host(&mut self.b_norms, rhs, "block RHS norms")?;
        reserve_block_host(&mut self.residual_norms, rhs, "block residual norms")?;
        reserve_block_host(&mut self.active, rhs, "active RHS indices")?;
        self.full_len = total;
        self.b_norms.resize(rhs, T::Real::zero());
        self.residual_norms.resize(rhs, T::Real::zero());
        self.active.clear();
        Ok(())
    }

    fn ensure_block(&mut self, block_len: usize, block_size: usize) -> Result<(), BlockGmresError> {
        reserve_block_host(&mut self.best_residuals, block_size, "best block residuals")?;
        self.block_len = block_len;
        self.best_residuals.resize(block_size, T::Real::zero());
        Ok(())
    }

    fn clear(&mut self) {
        *self = Self::new();
    }

    fn capacity(&self) -> usize {
        self.b_full.capacity()
            + self.x_full.capacity()
            + self.b_norms.capacity()
            + self.residual_norms.capacity()
            + self.active.capacity()
            + self.best_residuals.capacity()
            + self.v_blocks.iter().map(Vec::capacity).sum::<usize>()
            + self.z_blocks.iter().map(Vec::capacity).sum::<usize>()
            + self
                .h_blocks
                .iter()
                .flat_map(|row| row.iter())
                .map(Vec::capacity)
                .sum::<usize>()
            + self.h_scalar.capacity()
            + self.rhs_scalar.capacity()
            + self.full_residuals.capacity()
            + self.w_scratch.capacity()
            + self.ax_scratch.capacity()
            + self.residual_scratch.capacity()
            + self.qr_r.capacity()
            + self.initial_r.capacity()
            + self.block_inner.capacity()
            + self.least_squares_normal.capacity()
            + self.least_squares_rhs.capacity()
            + self.least_squares_solution.capacity()
    }
}

fn reserve_block_host<T>(
    values: &mut Vec<T>,
    required: usize,
    label: &'static str,
) -> Result<(), BlockGmresError> {
    if required > values.len() {
        values
            .try_reserve_exact(required - values.len())
            .map_err(|error| BlockGmresError::Allocation {
                label,
                reason: error.to_string(),
            })?;
    }
    Ok(())
}

#[cfg(test)]
fn zeroed_block_host<T: Copy + Default>(
    len: usize,
    label: &'static str,
) -> Result<Vec<T>, BlockGmresError> {
    crate::backend::try_zeroed_host::<T>(len, label).map_err(|error| match error {
        crate::BackendError::Allocation { reason } => BlockGmresError::Allocation { label, reason },
        other => BlockGmresError::Backend(other.to_string()),
    })
}

fn reserved_block_host<T>(len: usize, label: &'static str) -> Result<Vec<T>, BlockGmresError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|error| BlockGmresError::Allocation {
            label,
            reason: error.to_string(),
        })?;
    Ok(values)
}

fn push_block_host<T>(
    values: &mut Vec<T>,
    value: T,
    label: &'static str,
) -> Result<(), BlockGmresError> {
    if values.len() == values.capacity() {
        values
            .try_reserve_exact(1)
            .map_err(|error| BlockGmresError::Allocation {
                label,
                reason: error.to_string(),
            })?;
    }
    values.push(value);
    Ok(())
}

fn prepare_block_slot<'a, T: DeviceScalar>(
    blocks: &'a mut Vec<Vec<T>>,
    index: usize,
    len: usize,
    label: &'static str,
) -> Result<&'a mut Vec<T>, BlockGmresError> {
    while blocks.len() <= index {
        push_block_host(blocks, Vec::new(), label)?;
    }
    reserve_block_host(&mut blocks[index], len, label)?;
    blocks[index].resize(len, T::zero());
    Ok(&mut blocks[index])
}

impl<B: Backend, T: DeviceScalar> BlockGmresWorkspace<B, T> {
    /// Create empty scratch for operators of dimension `n`.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            full_len: 0,
            block_len: 0,
            full_x: None,
            full_ax: None,
            block_x: None,
            block_v: None,
            block_z: None,
            block_w: None,
            block_ax: None,
            block_b: None,
            block_x_initial: None,
            block_column: None,
            block_residual: None,
            resident_block_len: 0,
            resident_basis: Vec::new(),
            resident_z: Vec::new(),
            resident_v_flat: None,
            resident_z_flat: None,
            resident_projection: None,
            resident_coefficients: None,
            host: BlockGmresHostWorkspace::new(),
            retain_active_shapes: false,
            cached_shapes: Vec::new(),
            fallback: BlockFallback {
                columns: None,
                pool: None,
                krylov: GmresWorkspace::new(n),
            },
        }
    }

    /// Retain scalar fallback Krylov storage through the supplied pool.
    ///
    /// Pass `pool.backend()` to solves using this workspace. The pool covers
    /// scalar fallback Krylov buffers; block buffers and the three retained
    /// fallback columns continue to use the backend allocator directly.
    pub fn with_fallback_pool(n: usize, pool: &crate::MemoryPool<B>) -> Self {
        let mut workspace = Self::new(n);
        workspace.fallback.pool = Some(pool.clone());
        workspace
    }

    /// Retain active-width buffers as well as pooled scalar-fallback scratch.
    ///
    /// Deflation can visit several active RHS widths. This opt-in constructor
    /// retains each visited width instead of replacing its buffers, trading
    /// memory for allocation reuse. New widths or larger Krylov limits may grow
    /// storage; `clear` releases every cached shape. Block basis buffers retain
    /// their existing backend allocation policy, so pool quota coverage is partial.
    pub fn with_retained_shapes(n: usize, pool: &crate::MemoryPool<B>) -> Self {
        let mut workspace = Self::with_fallback_pool(n, pool);
        workspace.retain_active_shapes = true;
        workspace
    }

    /// Bytes cached in the optional scalar fallback Krylov workspace.
    ///
    /// These bytes are excluded from `device_capacity` and `device_buffer_count`.
    ///
    /// # Errors
    /// Returns an error if the scalar workspace cache was poisoned.
    pub fn fallback_cached_bytes(&self) -> Result<usize, GmresBackendError> {
        self.fallback.krylov.cached_bytes()
    }

    /// Operator dimension this workspace accepts.
    pub fn dimension(&self) -> usize {
        self.n
    }

    /// Number of directly retained buffers, excluding pooled scalar Krylov storage.
    pub fn device_buffer_count(&self) -> usize {
        [
            self.full_x.is_some(),
            self.full_ax.is_some(),
            self.block_x.is_some(),
            self.block_v.is_some(),
            self.block_z.is_some(),
            self.block_w.is_some(),
            self.block_ax.is_some(),
            self.block_b.is_some(),
            self.block_x_initial.is_some(),
            self.block_column.is_some(),
            self.block_residual.is_some(),
            self.resident_v_flat.is_some(),
            self.resident_z_flat.is_some(),
            self.resident_projection.is_some(),
            self.resident_coefficients.is_some(),
        ]
        .into_iter()
        .filter(|present| *present)
        .count()
            + self.resident_basis.len()
            + self.resident_z.len()
            + self.fallback.columns.as_ref().map_or(0, |_| 3)
            + self
                .cached_shapes
                .iter()
                .map(BlockShape::buffer_count)
                .sum::<usize>()
    }

    /// Directly retained scalar slots, excluding pooled scalar Krylov storage.
    pub fn device_capacity(&self) -> usize {
        self.cached_shapes
            .iter()
            .map(BlockShape::capacity)
            .sum::<usize>()
            + self.fallback.columns.as_ref().map_or(0, |columns| {
                columns.iter().map(DeviceBufferLike::len).sum::<usize>()
            })
            + self.full_x.as_ref().map_or(0, |buffer| buffer.len())
            + self.full_ax.as_ref().map_or(0, |buffer| buffer.len())
            + self.block_x.as_ref().map_or(0, |buffer| buffer.len())
            + self.block_v.as_ref().map_or(0, |buffer| buffer.len())
            + self.block_z.as_ref().map_or(0, |buffer| buffer.len())
            + self.block_w.as_ref().map_or(0, |buffer| buffer.len())
            + self.block_ax.as_ref().map_or(0, |buffer| buffer.len())
            + self.block_b.as_ref().map_or(0, |buffer| buffer.len())
            + self
                .block_x_initial
                .as_ref()
                .map_or(0, |buffer| buffer.len())
            + self.block_column.as_ref().map_or(0, |buffer| buffer.len())
            + self
                .block_residual
                .as_ref()
                .map_or(0, |buffer| buffer.len())
            + self
                .resident_basis
                .iter()
                .map(|buffer| buffer.len())
                .sum::<usize>()
            + self
                .resident_z
                .iter()
                .map(|buffer| buffer.len())
                .sum::<usize>()
            + self
                .resident_v_flat
                .as_ref()
                .map_or(0, |buffer| buffer.len())
            + self
                .resident_z_flat
                .as_ref()
                .map_or(0, |buffer| buffer.len())
            + self
                .resident_projection
                .as_ref()
                .map_or(0, |buffer| buffer.len())
            + self
                .resident_coefficients
                .as_ref()
                .map_or(0, |buffer| buffer.len())
    }

    /// Number of retained host scalar slots used by staging and residual work.
    pub fn host_capacity(&self) -> usize {
        self.host.capacity()
    }

    /// Release workspace buffers and shape state.
    ///
    /// The caller's pool may cache released scalar buffers for later reuse.
    pub fn clear(&mut self) {
        self.cached_shapes.clear();
        self.full_len = 0;
        self.block_len = 0;
        self.full_x = None;
        self.full_ax = None;
        self.block_x = None;
        self.block_v = None;
        self.block_z = None;
        self.block_w = None;
        self.block_ax = None;
        self.block_b = None;
        self.block_x_initial = None;
        self.block_column = None;
        self.block_residual = None;
        self.resident_block_len = 0;
        self.resident_basis.clear();
        self.resident_z.clear();
        self.resident_v_flat = None;
        self.resident_z_flat = None;
        self.resident_projection = None;
        self.resident_coefficients = None;
        self.host.clear();
        self.fallback.columns = None;
        self.fallback.krylov = GmresWorkspace::new(self.n);
    }

    fn ensure_full(&mut self, backend: &B, len: usize) -> Result<(), BlockGmresError> {
        if self.block_column.is_none() {
            self.block_column = Some(backend.raw_alloc::<T>(self.n).map_err(|error| {
                BlockGmresError::Allocation {
                    label: "block column scratch",
                    reason: error.to_string(),
                }
            })?);
        }
        if self.full_len == len && self.full_x.is_some() && self.full_ax.is_some() {
            return Ok(());
        }
        let x = backend
            .raw_alloc::<T>(len)
            .map_err(|error| BlockGmresError::Allocation {
                label: "block full solution",
                reason: error.to_string(),
            })?;
        let ax = backend
            .raw_alloc::<T>(len)
            .map_err(|error| BlockGmresError::Allocation {
                label: "block full matvec",
                reason: error.to_string(),
            })?;
        self.full_len = len;
        self.full_x = Some(x);
        self.full_ax = Some(ax);
        Ok(())
    }

    fn ensure_block(&mut self, backend: &B, len: usize) -> Result<(), BlockGmresError> {
        if self.block_len == len
            && self.block_x.is_some()
            && self.block_v.is_some()
            && self.block_z.is_some()
            && self.block_w.is_some()
            && self.block_ax.is_some()
        {
            return Ok(());
        }
        if self.retain_active_shapes {
            self.cached_shapes
                .try_reserve(1)
                .map_err(|error| BlockGmresError::Allocation {
                    label: "retained block shapes",
                    reason: error.to_string(),
                })?;
            if let Some(index) = self
                .cached_shapes
                .iter()
                .position(|shape| shape.block_len == len)
            {
                let cached = self.cached_shapes.swap_remove(index);
                let previous = self.take_block_shape();
                if previous.block_len != 0 {
                    self.cached_shapes.push(previous);
                }
                self.install_block_shape(cached);
                return Ok(());
            }
        }
        let x = backend
            .raw_alloc::<T>(len)
            .map_err(|error| BlockGmresError::Allocation {
                label: "block solution",
                reason: error.to_string(),
            })?;
        let v = backend
            .raw_alloc::<T>(len)
            .map_err(|error| BlockGmresError::Allocation {
                label: "block basis",
                reason: error.to_string(),
            })?;
        let z = backend
            .raw_alloc::<T>(len)
            .map_err(|error| BlockGmresError::Allocation {
                label: "block preconditioned basis",
                reason: error.to_string(),
            })?;
        let w = backend
            .raw_alloc::<T>(len)
            .map_err(|error| BlockGmresError::Allocation {
                label: "block Arnoldi",
                reason: error.to_string(),
            })?;
        let ax = backend
            .raw_alloc::<T>(len)
            .map_err(|error| BlockGmresError::Allocation {
                label: "block residual",
                reason: error.to_string(),
            })?;
        let b = backend
            .raw_alloc::<T>(len)
            .map_err(|error| BlockGmresError::Allocation {
                label: "block RHS",
                reason: error.to_string(),
            })?;
        let x_initial =
            backend
                .raw_alloc::<T>(len)
                .map_err(|error| BlockGmresError::Allocation {
                    label: "block initial solution",
                    reason: error.to_string(),
                })?;
        let residual =
            backend
                .raw_alloc::<T>(len)
                .map_err(|error| BlockGmresError::Allocation {
                    label: "block residual",
                    reason: error.to_string(),
                })?;
        if self.retain_active_shapes && self.block_len != 0 {
            let previous = self.take_block_shape();
            self.cached_shapes.push(previous);
        }
        self.block_len = len;
        self.block_x = Some(x);
        self.block_v = Some(v);
        self.block_z = Some(z);
        self.block_w = Some(w);
        self.block_ax = Some(ax);
        self.block_b = Some(b);
        self.block_x_initial = Some(x_initial);
        self.block_residual = Some(residual);
        Ok(())
    }

    fn ensure_resident_krylov(
        &mut self,
        backend: &B,
        block_len: usize,
        block_size: usize,
        max_blocks: usize,
    ) -> Result<(), BlockGmresError> {
        if self.resident_block_len != block_len {
            self.resident_block_len = block_len;
            self.resident_basis.clear();
            self.resident_z.clear();
            self.resident_v_flat = None;
            self.resident_z_flat = None;
        }

        let flat_len = block_len.checked_mul(max_blocks).ok_or_else(|| {
            BlockGmresError::InvalidArgument("resident block basis size overflow".into())
        })?;
        if self
            .resident_v_flat
            .as_ref()
            .is_none_or(|buffer| buffer.len() < flat_len)
        {
            self.resident_v_flat = Some(backend.raw_alloc::<T>(flat_len).map_err(|error| {
                BlockGmresError::Allocation {
                    label: "resident block basis",
                    reason: error.to_string(),
                }
            })?);
        }
        if self
            .resident_z_flat
            .as_ref()
            .is_none_or(|buffer| buffer.len() < flat_len)
        {
            self.resident_z_flat = Some(backend.raw_alloc::<T>(flat_len).map_err(|error| {
                BlockGmresError::Allocation {
                    label: "resident block preconditioned basis",
                    reason: error.to_string(),
                }
            })?);
        }
        let projection_len = block_size.checked_mul(max_blocks).ok_or_else(|| {
            BlockGmresError::InvalidArgument("resident projection size overflow".into())
        })?;
        if self
            .resident_projection
            .as_ref()
            .is_none_or(|buffer| buffer.len() < projection_len)
        {
            self.resident_projection =
                Some(backend.raw_alloc::<T>(projection_len).map_err(|error| {
                    BlockGmresError::Allocation {
                        label: "resident block projection",
                        reason: error.to_string(),
                    }
                })?);
        }
        if self
            .resident_coefficients
            .as_ref()
            .is_none_or(|buffer| buffer.len() < projection_len)
        {
            self.resident_coefficients =
                Some(backend.raw_alloc::<T>(projection_len).map_err(|error| {
                    BlockGmresError::Allocation {
                        label: "resident block coefficients",
                        reason: error.to_string(),
                    }
                })?);
        }
        while self.resident_basis.len() < max_blocks {
            self.resident_basis
                .push(backend.raw_alloc::<T>(block_len).map_err(|error| {
                    BlockGmresError::Allocation {
                        label: "resident block Arnoldi basis",
                        reason: error.to_string(),
                    }
                })?);
        }
        while self.resident_z.len() < max_blocks {
            self.resident_z
                .push(backend.raw_alloc::<T>(block_len).map_err(|error| {
                    BlockGmresError::Allocation {
                        label: "resident block preconditioned basis",
                        reason: error.to_string(),
                    }
                })?);
        }
        Ok(())
    }
}

/// Per-block and per-RHS result from [`gmres_backend_block_preconditioned`].
#[derive(Debug, Clone)]
pub struct BlockGmresStatus<T: DeviceScalar> {
    /// Number of block Arnoldi steps performed.
    pub block_iterations: usize,
    /// Projected block residual.  The current block implementation promotes
    /// every candidate through the true residual path, so this is equal to
    /// `final_residual` until a backend supplies a separate least-squares
    /// telemetry reduction.
    pub projected_residual: T::Real,
    /// Maximum final relative true residual over all RHS.
    pub final_residual: T::Real,
    /// Maximum normwise backward error over all RHS.
    pub backward_error: T::Real,
    /// True when every RHS passed its tolerance gate.
    pub converged: bool,
    /// Individual residuals and convergence flags, in input RHS order.
    pub rhs: Vec<GmresBackendStatus<T>>,
}

/// Runtime failures specific to the block recurrence.
#[derive(Debug, thiserror::Error)]
pub enum BlockGmresError {
    #[error("invalid block-GMRES argument: {0}")]
    InvalidArgument(String),
    #[error("block-GMRES workspace allocation failed for {label}: {reason}")]
    Allocation { label: &'static str, reason: String },
    #[error("backend GMRES failed: {0}")]
    Backend(String),
    #[error(
        "block Arnoldi rank deficient at block iteration {iteration}: rank {rank} of {block_size}"
    )]
    RankDeficient {
        iteration: usize,
        rank: usize,
        block_size: usize,
    },
    #[error("block least-squares system became singular at block iteration {iteration}")]
    SingularLeastSquares { iteration: usize },
    #[error("block Arnoldi broke down at block iteration {iteration} before convergence")]
    Breakdown { iteration: usize },
}

/// Route host/device transfers through the checked backend boundary.  The
/// legacy `copy_*` methods are intentionally panic-compatible for existing
/// callers, but a checked Krylov solve must surface driver/shape failures as
/// `BlockGmresError` instead of unwinding halfway through a recurrence.
fn copy_h2d_checked<B, T>(
    backend: &B,
    host: &[T],
    device: &mut B::Buffer<T>,
) -> Result<(), BlockGmresError>
where
    B: Backend,
    T: DeviceScalar,
{
    backend
        .try_copy_h2d(host, device)
        .map_err(|error| BlockGmresError::Backend(error.to_string()))
}

fn copy_d2h_checked<B, T>(
    backend: &B,
    device: &B::Buffer<T>,
    host: &mut [T],
) -> Result<(), BlockGmresError>
where
    B: Backend,
    T: DeviceScalar,
{
    backend
        .try_copy_d2h(device, host)
        .map_err(|error| BlockGmresError::Backend(error.to_string()))
}

fn validate_block_input<B: Backend, T: DeviceScalar>(
    backend: &B,
    input: &B::Buffer<T>,
    staging: &mut Vec<T>,
    label: &str,
) -> Result<(), BlockGmresError> {
    let finite = match backend
        .try_all_finite_native(input)
        .map_err(|error| BlockGmresError::Backend(error.to_string()))?
    {
        Some(finite) => finite,
        None => {
            reserve_block_host(staging, input.len(), "input validation")?;
            staging.resize(input.len(), T::zero());
            copy_d2h_checked(backend, input, staging)?;
            staging.iter().all(DeviceScalar::is_finite)
        }
    };
    if finite {
        Ok(())
    } else {
        Err(BlockGmresError::InvalidArgument(format!(
            "block-GMRES {label} contains non-finite values"
        )))
    }
}

fn missing_workspace_buffer(label: &'static str) -> BlockGmresError {
    BlockGmresError::Backend(format!(
        "block-GMRES workspace buffer {label} is not initialized"
    ))
}

fn copy_device_region<B, T>(
    backend: &B,
    source: &B::Buffer<T>,
    source_offset: usize,
    destination: &mut B::Buffer<T>,
    destination_offset: usize,
    len: usize,
) -> Result<(), BlockGmresError>
where
    B: Backend,
    T: DeviceScalar,
{
    let source_view = BufferView::new(source.len(), source_offset, len, 1)
        .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
    let destination_view = BufferView::new(destination.len(), destination_offset, len, 1)
        .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
    backend
        .try_copy_views_d2d(source, source_view, destination, destination_view)
        .map_err(|error| BlockGmresError::Backend(error.to_string()))
}

/// Solve `A X = B` with a shared block Arnoldi basis.
///
/// `b` and `out` contain `rhs` contiguous columns of length `A.n()`.  The
/// preconditioner is right-applied to each Arnoldi block, matching the
/// convention used by the single-RHS backend GMRES implementation.  Exact
/// zero RHS columns are deflated before the block recurrence and are reported
/// as converged with zero residual.
#[allow(clippy::too_many_arguments)]
pub fn gmres_backend_block_preconditioned<B, A, P, T>(
    backend: &B,
    a: &A,
    precond: &P,
    b: &B::Buffer<T>,
    x0: Option<&B::Buffer<T>>,
    out: &mut B::Buffer<T>,
    rhs: usize,
    config: &BlockGmresConfig<T::Real>,
) -> Result<BlockGmresStatus<T>, BlockGmresError>
where
    B: Backend,
    A: DeviceOperator<B, T> + ?Sized,
    P: DevicePreconditioner<B, T> + ?Sized,
    T: DeviceScalar + std::ops::AddAssign,
{
    let mut workspace = BlockGmresWorkspace::new(a.n());
    gmres_backend_block_preconditioned_with_workspace(
        backend,
        a,
        precond,
        b,
        x0,
        out,
        rhs,
        config,
        &mut workspace,
    )
}

/// Block-GMRES solve using caller-owned reusable device scratch.
#[allow(clippy::too_many_arguments)]
pub fn gmres_backend_block_preconditioned_with_workspace<B, A, P, T>(
    backend: &B,
    a: &A,
    precond: &P,
    b: &B::Buffer<T>,
    x0: Option<&B::Buffer<T>>,
    out: &mut B::Buffer<T>,
    rhs: usize,
    config: &BlockGmresConfig<T::Real>,
    workspace: &mut BlockGmresWorkspace<B, T>,
) -> Result<BlockGmresStatus<T>, BlockGmresError>
where
    B: Backend,
    A: DeviceOperator<B, T> + ?Sized,
    P: DevicePreconditioner<B, T> + ?Sized,
    T: DeviceScalar + std::ops::AddAssign,
{
    if let Some(pool) = &workspace.fallback.pool
        && !std::ptr::eq(backend, pool.backend())
    {
        return Err(BlockGmresError::InvalidArgument(
            "block workspace requires its fallback pool's backend instance".into(),
        ));
    }
    if !config.tolerance.is_finite() || config.tolerance < T::Real::zero() {
        return Err(BlockGmresError::InvalidArgument(
            "block-GMRES tolerance must be finite and non-negative".into(),
        ));
    }
    if config.max_block_iterations == 0 {
        return Err(BlockGmresError::InvalidArgument(
            "block-GMRES max_block_iterations must be positive".into(),
        ));
    }
    let n = a.n();
    if workspace.dimension() != n {
        return Err(BlockGmresError::InvalidArgument(format!(
            "block-GMRES workspace dimension mismatch: workspace {}, operator {n}",
            workspace.dimension()
        )));
    }
    if rhs == 0 {
        return Err(BlockGmresError::InvalidArgument(
            "block-GMRES requires at least one RHS".into(),
        ));
    }
    let total = n.checked_mul(rhs).ok_or_else(|| {
        BlockGmresError::InvalidArgument(format!("RHS size overflow: n={n}, rhs={rhs}"))
    })?;
    if b.len() != total {
        return Err(BlockGmresError::InvalidArgument(format!(
            "block RHS length mismatch: got {}, expected {total}",
            b.len()
        )));
    }
    if out.len() != total {
        return Err(BlockGmresError::InvalidArgument(format!(
            "block output length mismatch: got {}, expected {total}",
            out.len()
        )));
    }

    if let Some(x0) = x0
        && x0.len() != total
    {
        return Err(BlockGmresError::InvalidArgument(format!(
            "block initial guess length mismatch: got {}, expected {total}",
            x0.len()
        )));
    }

    // Retain device storage and compact host control state. Full input staging
    // is allocated lazily only if a backend lacks native initialization hooks.
    workspace.ensure_full(backend, total)?;
    workspace.host.ensure_full(total, rhs)?;
    validate_block_input(backend, b, &mut workspace.host.b_full, "RHS")?;
    if let Some(x0) = x0 {
        validate_block_input(backend, x0, &mut workspace.host.x_full, "initial guess")?;
    }

    // Compute one initial residual for the full set.  This also lets us
    // deflate zero RHS columns without forcing a rank-deficient QR block.
    let full_x = workspace
        .full_x
        .as_mut()
        .ok_or_else(|| missing_workspace_buffer("full solution"))?;
    if let Some(x0) = x0 {
        backend
            .try_copy_d2d(x0, full_x)
            .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
    } else if !backend
        .try_zero_native(full_x)
        .map_err(|error| BlockGmresError::Backend(error.to_string()))?
    {
        reserve_block_host(&mut workspace.host.x_full, total, "zero initial guess")?;
        workspace.host.x_full.resize(total, T::zero());
        workspace.host.x_full.fill(T::zero());
        copy_h2d_checked(backend, &workspace.host.x_full, full_x)?;
    }
    a.try_apply_colmajor(
        backend,
        rhs,
        workspace
            .full_x
            .as_ref()
            .ok_or_else(|| missing_workspace_buffer("full solution"))?,
        workspace
            .full_ax
            .as_mut()
            .ok_or_else(|| missing_workspace_buffer("full operator result"))?,
    )
    .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
    let residual = workspace
        .full_ax
        .as_mut()
        .ok_or_else(|| missing_workspace_buffer("full residual"))?;
    backend
        .try_scale(T::zero() - T::one(), residual)
        .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
    backend
        .try_axpy(T::one(), b, residual)
        .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
    let tolerance = config.tolerance.max(T::Real::zero());
    for col in 0..rhs {
        let column = workspace
            .block_column
            .as_mut()
            .ok_or_else(|| missing_workspace_buffer("column scratch"))?;
        copy_device_region(backend, b, col * n, column, 0, n)?;
        workspace.host.b_norms[col] = backend
            .try_nrm2_with_mode(ReductionMode::Fast, column)
            .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
        copy_device_region(backend, residual, col * n, column, 0, n)?;
        workspace.host.residual_norms[col] = backend
            .try_nrm2_with_mode(ReductionMode::Fast, column)
            .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
    }
    workspace.host.active.extend((0..rhs).filter(|&col| {
        let b_norm = workspace.host.b_norms[col];
        let residual_norm = workspace.host.residual_norms[col];
        if b_norm > T::Real::zero() {
            residual_norm / b_norm > tolerance
        } else {
            residual_norm > tolerance
        }
    }));
    let active = std::mem::take(&mut workspace.host.active);
    let b_norms = std::mem::take(&mut workspace.host.b_norms);
    let residual_norms = std::mem::take(&mut workspace.host.residual_norms);
    if active.is_empty() {
        let status = make_status(0, &b_norms, &residual_norms, tolerance, rhs)?;
        workspace.host.active = active;
        workspace.host.b_norms = b_norms;
        workspace.host.residual_norms = residual_norms;
        backend
            .try_copy_d2d(
                workspace
                    .full_x
                    .as_ref()
                    .ok_or_else(|| missing_workspace_buffer("full solution"))?,
                out,
            )
            .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
        return Ok(status);
    }

    let block_size = active.len();
    let block_storage = n.checked_mul(block_size).ok_or_else(|| {
        BlockGmresError::InvalidArgument(format!(
            "active block size overflow: n={n}, block_size={block_size}"
        ))
    })?;
    workspace.ensure_block(backend, block_storage)?;
    workspace.host.ensure_block(block_storage, block_size)?;
    for (active_col, &full_col) in active.iter().enumerate() {
        copy_device_region(
            backend,
            b,
            full_col * n,
            workspace
                .block_b
                .as_mut()
                .ok_or_else(|| missing_workspace_buffer("RHS"))?,
            active_col * n,
            n,
        )?;
        copy_device_region(
            backend,
            workspace
                .full_x
                .as_ref()
                .ok_or_else(|| missing_workspace_buffer("full solution"))?,
            full_col * n,
            workspace
                .block_x_initial
                .as_mut()
                .ok_or_else(|| missing_workspace_buffer("initial solution"))?,
            active_col * n,
            n,
        )?;
        copy_device_region(
            backend,
            workspace
                .full_ax
                .as_ref()
                .ok_or_else(|| missing_workspace_buffer("full residual"))?,
            full_col * n,
            workspace
                .block_w
                .as_mut()
                .ok_or_else(|| missing_workspace_buffer("active residual"))?,
            active_col * n,
            n,
        )?;
    }
    let qr_tolerance = T::Real::epsilon().sqrt();
    let rank = qr_device_block_into(
        backend,
        workspace
            .block_w
            .as_ref()
            .ok_or_else(|| missing_workspace_buffer("active residual"))?,
        workspace
            .block_v
            .as_mut()
            .ok_or_else(|| missing_workspace_buffer("initial basis"))?,
        workspace
            .block_column
            .as_mut()
            .ok_or_else(|| missing_workspace_buffer("column scratch"))?,
        n,
        block_size,
        qr_tolerance,
        &mut workspace.host.qr_r,
        &mut workspace.host.block_inner,
    )?;
    if rank < block_size {
        // A dependent initial residual block contains fewer independent
        // directions than requested RHS columns.  Deflate that block to the
        // scalar path instead of rejecting a perfectly solvable batch.  This
        // fallback can also occur in small residual-correction systems;
        // full-rank batches retain the SpMM/block Krylov path below.
        let result = solve_rank_deficient_columns(
            &mut workspace.fallback,
            backend,
            a,
            precond,
            b,
            workspace
                .full_x
                .as_mut()
                .ok_or_else(|| missing_workspace_buffer("full solution"))?,
            &active,
            n,
            config,
            tolerance,
            out,
        );
        workspace.host.active = active;
        workspace.host.b_norms = b_norms;
        workspace.host.residual_norms = residual_norms;
        return result;
    }

    let max_iterations = config.max_block_iterations.max(1);
    workspace.host.initial_r.clear();
    reserve_block_host(
        &mut workspace.host.initial_r,
        workspace.host.qr_r.len(),
        "initial block QR coefficients",
    )?;
    workspace
        .host
        .initial_r
        .extend_from_slice(&workspace.host.qr_r);
    // Block-Hessenberg storage: rows of block matrices, each row indexed by
    // Arnoldi block and each column by the current block step.
    workspace.host.best_residuals.clear();
    reserve_block_host(
        &mut workspace.host.best_residuals,
        active.len(),
        "best block residuals",
    )?;
    workspace
        .host
        .best_residuals
        .extend(active.iter().map(|&col| {
            if b_norms[col] > T::Real::zero() {
                residual_norms[col] / b_norms[col]
            } else {
                residual_norms[col]
            }
        }));
    workspace.host.h_scalar.clear();
    workspace.host.rhs_scalar.clear();
    let resident_blocks = max_iterations.checked_add(1).ok_or_else(|| {
        BlockGmresError::InvalidArgument("resident block basis count overflow".into())
    })?;
    workspace.ensure_resident_krylov(backend, block_storage, block_size, resident_blocks)?;
    backend
        .try_copy_d2d(
            workspace
                .block_x_initial
                .as_ref()
                .ok_or_else(|| missing_workspace_buffer("initial solution"))?,
            workspace
                .block_x
                .as_mut()
                .ok_or_else(|| missing_workspace_buffer("solution"))?,
        )
        .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
    backend
        .try_copy_d2d(
            workspace
                .block_v
                .as_ref()
                .ok_or_else(|| missing_workspace_buffer("initial basis"))?,
            &mut workspace.resident_basis[0],
        )
        .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
    copy_device_region(
        backend,
        &workspace.resident_basis[0],
        0,
        workspace
            .resident_v_flat
            .as_mut()
            .ok_or_else(|| missing_workspace_buffer("flat basis"))?,
        0,
        block_storage,
    )?;
    let mut last_iterations = 0;

    for iteration in 0..max_iterations {
        last_iterations = iteration + 1;
        precond
            .try_apply_colmajor(
                backend,
                block_size,
                &workspace.resident_basis[iteration],
                &mut workspace.resident_z[iteration],
            )
            .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
        copy_device_region(
            backend,
            &workspace.resident_z[iteration],
            0,
            workspace
                .resident_z_flat
                .as_mut()
                .ok_or_else(|| missing_workspace_buffer("flat preconditioned basis"))?,
            iteration.checked_mul(block_storage).ok_or_else(|| {
                BlockGmresError::InvalidArgument("flat basis offset overflow".into())
            })?,
            block_storage,
        )?;
        a.try_apply_colmajor(
            backend,
            block_size,
            &workspace.resident_z[iteration],
            workspace
                .block_w
                .as_mut()
                .ok_or_else(|| missing_workspace_buffer("operator result"))?,
        )
        .map_err(|error| BlockGmresError::Backend(error.to_string()))?;

        let required_rows = iteration.checked_add(2).ok_or_else(|| {
            BlockGmresError::InvalidArgument("block Hessenberg row count overflow".into())
        })?;
        while workspace.host.h_blocks.len() < required_rows {
            let row_capacity = iteration.checked_add(1).ok_or_else(|| {
                BlockGmresError::InvalidArgument("block Hessenberg row size overflow".into())
            })?;
            let row = reserved_block_host(row_capacity, "block Hessenberg row")?;
            push_block_host(
                &mut workspace.host.h_blocks,
                row,
                "block Hessenberg row list",
            )?;
        }
        let h_block_len = block_size.checked_mul(block_size).ok_or_else(|| {
            BlockGmresError::InvalidArgument("block Hessenberg size overflow".into())
        })?;
        for row in workspace.host.h_blocks.iter_mut().take(required_rows) {
            for column in 0..=iteration {
                let block = prepare_block_slot(row, column, h_block_len, "block Hessenberg row")?;
                block.fill(T::zero());
            }
        }

        // Orthogonalise each W column against the resident, flat V basis.
        // The small projection is read back for the host Hessenberg/control
        // state, but the resident projection buffer is negated in place and
        // consumed directly by the fused update. W and every Arnoldi vector
        // remain device-resident, with no per-iteration coefficient upload.
        let basis_count = (iteration + 1)
            .checked_mul(block_size)
            .ok_or_else(|| BlockGmresError::InvalidArgument("block basis count overflow".into()))?;
        for target_col in 0..block_size {
            copy_device_region(
                backend,
                workspace
                    .block_w
                    .as_ref()
                    .ok_or_else(|| missing_workspace_buffer("operator result"))?,
                target_col.checked_mul(n).ok_or_else(|| {
                    BlockGmresError::InvalidArgument("column offset overflow".into())
                })?,
                workspace
                    .block_column
                    .as_mut()
                    .ok_or_else(|| missing_workspace_buffer("column scratch"))?,
                0,
                n,
            )?;
            reserve_block_host(
                &mut workspace.host.block_inner,
                basis_count,
                "block projection coefficients",
            )?;
            workspace.host.block_inner.resize(basis_count, T::zero());
            if let Some(projection) = workspace.resident_projection.as_mut() {
                backend
                    .try_multi_dot_strided_with_mode_device_into(
                        ReductionMode::Fast,
                        workspace
                            .resident_v_flat
                            .as_ref()
                            .ok_or_else(|| missing_workspace_buffer("flat basis"))?,
                        n,
                        basis_count,
                        workspace
                            .block_column
                            .as_ref()
                            .ok_or_else(|| missing_workspace_buffer("column scratch"))?,
                        projection,
                    )
                    .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
                backend
                    .try_copy_view_d2h(
                        projection,
                        BufferView::new(projection.len(), 0, basis_count, 1)
                            .map_err(|error| BlockGmresError::Backend(error.to_string()))?,
                        &mut workspace.host.block_inner[..basis_count],
                    )
                    .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
            } else {
                backend
                    .try_multi_dot_strided_with_mode_into(
                        ReductionMode::Fast,
                        workspace
                            .resident_v_flat
                            .as_ref()
                            .ok_or_else(|| missing_workspace_buffer("flat basis"))?,
                        n,
                        basis_count,
                        workspace
                            .block_column
                            .as_ref()
                            .ok_or_else(|| missing_workspace_buffer("column scratch"))?,
                        &mut workspace.host.block_inner,
                    )
                    .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
            }
            for row in 0..=iteration {
                let start = row * block_size;
                for left_col in 0..block_size {
                    workspace.host.h_blocks[row][iteration][left_col * block_size + target_col] =
                        workspace.host.block_inner[start + left_col];
                }
            }
            let projection = workspace
                .resident_projection
                .as_mut()
                .ok_or_else(|| missing_workspace_buffer("projection"))?;
            backend
                .try_scale(T::zero() - T::one(), projection)
                .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
            backend
                .try_basis_axpy_strided_device(
                    workspace
                        .resident_v_flat
                        .as_ref()
                        .ok_or_else(|| missing_workspace_buffer("flat basis"))?,
                    n,
                    basis_count,
                    projection,
                    workspace
                        .block_column
                        .as_mut()
                        .ok_or_else(|| missing_workspace_buffer("column scratch"))?,
                )
                .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
            copy_device_region(
                backend,
                workspace
                    .block_column
                    .as_ref()
                    .ok_or_else(|| missing_workspace_buffer("column scratch"))?,
                0,
                workspace
                    .block_w
                    .as_mut()
                    .ok_or_else(|| missing_workspace_buffer("operator result"))?,
                target_col.checked_mul(n).ok_or_else(|| {
                    BlockGmresError::InvalidArgument("column offset overflow".into())
                })?,
                n,
            )?;
        }

        // Modified Gram-Schmidt/CGS2 QR of W on the backend.  The R block is
        // small host scalar state, while Q is written directly into the next
        // resident Arnoldi block.
        workspace.host.qr_r.fill(T::zero());
        let mut next_rank = 0;
        for column in 0..block_size {
            copy_device_region(
                backend,
                workspace
                    .block_w
                    .as_ref()
                    .ok_or_else(|| missing_workspace_buffer("operator result"))?,
                column * n,
                workspace
                    .block_column
                    .as_mut()
                    .ok_or_else(|| missing_workspace_buffer("column scratch"))?,
                0,
                n,
            )?;
            for _ in 0..2 {
                reserve_block_host(
                    &mut workspace.host.block_inner,
                    column,
                    "QR projection coefficients",
                )?;
                workspace.host.block_inner.resize(column, T::zero());
                if let Some(projection) = workspace.resident_projection.as_mut() {
                    backend
                        .try_multi_dot_strided_with_mode_device_into(
                            ReductionMode::Fast,
                            &workspace.resident_basis[iteration + 1],
                            n,
                            column,
                            workspace
                                .block_column
                                .as_ref()
                                .ok_or_else(|| missing_workspace_buffer("column scratch"))?,
                            projection,
                        )
                        .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
                    backend
                        .try_copy_view_d2h(
                            projection,
                            BufferView::new(projection.len(), 0, column, 1)
                                .map_err(|error| BlockGmresError::Backend(error.to_string()))?,
                            &mut workspace.host.block_inner[..column],
                        )
                        .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
                } else {
                    backend
                        .try_multi_dot_strided_with_mode_into(
                            ReductionMode::Fast,
                            &workspace.resident_basis[iteration + 1],
                            n,
                            column,
                            workspace
                                .block_column
                                .as_ref()
                                .ok_or_else(|| missing_workspace_buffer("column scratch"))?,
                            &mut workspace.host.block_inner,
                        )
                        .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
                }
                for (previous, &coefficient) in workspace.host.block_inner.iter().enumerate() {
                    workspace.host.qr_r[previous * block_size + column] += coefficient;
                }
                let projection = workspace
                    .resident_projection
                    .as_mut()
                    .ok_or_else(|| missing_workspace_buffer("projection"))?;
                backend
                    .try_scale(T::zero() - T::one(), projection)
                    .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
                backend
                    .try_basis_axpy_strided_device(
                        &workspace.resident_basis[iteration + 1],
                        n,
                        column,
                        projection,
                        workspace
                            .block_column
                            .as_mut()
                            .ok_or_else(|| missing_workspace_buffer("column scratch"))?,
                    )
                    .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
            }
            let norm = backend
                .try_nrm2_with_mode::<T>(
                    ReductionMode::Fast,
                    workspace
                        .block_column
                        .as_ref()
                        .ok_or_else(|| missing_workspace_buffer("column scratch"))?,
                )
                .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
            if norm > qr_tolerance {
                next_rank += 1;
                workspace.host.qr_r[column * block_size + column] = T::from_real(norm);
                backend
                    .try_scale::<T>(
                        T::from_real(T::Real::one() / norm),
                        workspace
                            .block_column
                            .as_mut()
                            .ok_or_else(|| missing_workspace_buffer("column scratch"))?,
                    )
                    .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
                copy_device_region(
                    backend,
                    workspace
                        .block_column
                        .as_ref()
                        .ok_or_else(|| missing_workspace_buffer("column scratch"))?,
                    0,
                    &mut workspace.resident_basis[iteration + 1],
                    column * n,
                    n,
                )?;
            }
        }
        let breakdown = next_rank < block_size;
        if !breakdown {
            copy_device_region(
                backend,
                &workspace.resident_basis[iteration + 1],
                0,
                workspace
                    .resident_v_flat
                    .as_mut()
                    .ok_or_else(|| missing_workspace_buffer("flat basis"))?,
                (iteration + 1).checked_mul(block_storage).ok_or_else(|| {
                    BlockGmresError::InvalidArgument("flat basis offset overflow".into())
                })?,
                block_storage,
            )?;
            workspace.host.h_blocks[iteration + 1][iteration].copy_from_slice(&workspace.host.qr_r);
        }

        let row_blocks = if breakdown {
            iteration + 1
        } else {
            iteration + 2
        };
        let column_blocks = iteration + 1;
        flatten_h_into(
            &workspace.host.h_blocks,
            row_blocks,
            column_blocks,
            block_size,
            &mut workspace.host.h_scalar,
        )?;
        let block_square = block_size.checked_mul(block_size).ok_or_else(|| {
            BlockGmresError::InvalidArgument("block RHS square size overflow".into())
        })?;
        let rhs_scalar_len = row_blocks
            .checked_mul(block_square)
            .ok_or_else(|| BlockGmresError::InvalidArgument("block RHS size overflow".into()))?;
        reserve_block_host(
            &mut workspace.host.rhs_scalar,
            rhs_scalar_len,
            "block least-squares RHS",
        )?;
        workspace.host.rhs_scalar.resize(rhs_scalar_len, T::zero());
        workspace.host.rhs_scalar[..block_square].copy_from_slice(&workspace.host.initial_r);
        let least_squares_result = least_squares_into(
            &workspace.host.h_scalar,
            row_blocks * block_size,
            column_blocks * block_size,
            &workspace.host.rhs_scalar,
            block_size,
            &mut workspace.host.least_squares_normal,
            &mut workspace.host.least_squares_rhs,
            &mut workspace.host.least_squares_solution,
        );
        match least_squares_result {
            Ok(()) => {}
            Err(LeastSquaresError::Allocation(error)) => return Err(error),
            Err(LeastSquaresError::Singular) => {
                // A later Arnoldi block can lose rank even when the initial
                // residual block was full rank.  The projected normal
                // equations are then singular, so finish the active columns
                // through the checked scalar path instead of publishing a
                // panic or an unusable block iterate.
                let result = solve_rank_deficient_columns(
                    &mut workspace.fallback,
                    backend,
                    a,
                    precond,
                    b,
                    workspace
                        .full_x
                        .as_mut()
                        .ok_or_else(|| missing_workspace_buffer("full solution"))?,
                    &active,
                    n,
                    config,
                    tolerance,
                    out,
                );
                workspace.host.active = active;
                workspace.host.b_norms = b_norms;
                workspace.host.residual_norms = residual_norms;
                return result;
            }
        }

        backend
            .try_copy_d2d(
                workspace
                    .block_x_initial
                    .as_ref()
                    .ok_or_else(|| missing_workspace_buffer("initial solution"))?,
                workspace
                    .block_x
                    .as_mut()
                    .ok_or_else(|| missing_workspace_buffer("solution"))?,
            )
            .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
        let z_basis_count = column_blocks.checked_mul(block_size).ok_or_else(|| {
            BlockGmresError::InvalidArgument("block solution basis count overflow".into())
        })?;
        for rhs_col in 0..block_size {
            copy_device_region(
                backend,
                workspace
                    .block_x
                    .as_ref()
                    .ok_or_else(|| missing_workspace_buffer("solution"))?,
                rhs_col * n,
                workspace
                    .block_column
                    .as_mut()
                    .ok_or_else(|| missing_workspace_buffer("column scratch"))?,
                0,
                n,
            )?;
            reserve_block_host(
                &mut workspace.host.block_inner,
                z_basis_count,
                "block solution coefficients",
            )?;
            workspace.host.block_inner.resize(z_basis_count, T::zero());
            for (basis_index, coefficient) in workspace.host.block_inner.iter_mut().enumerate() {
                *coefficient =
                    workspace.host.least_squares_solution[basis_index * block_size + rhs_col];
            }
            let coefficients = workspace
                .resident_coefficients
                .as_mut()
                .ok_or_else(|| missing_workspace_buffer("resident coefficients"))?;
            backend
                .try_copy_view_h2d(
                    &workspace.host.block_inner[..z_basis_count],
                    coefficients,
                    BufferView::new(coefficients.len(), 0, z_basis_count, 1)
                        .map_err(|error| BlockGmresError::Backend(error.to_string()))?,
                )
                .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
            backend
                .try_basis_axpy_strided_device(
                    workspace
                        .resident_z_flat
                        .as_ref()
                        .ok_or_else(|| missing_workspace_buffer("flat preconditioned basis"))?,
                    n,
                    z_basis_count,
                    workspace
                        .resident_coefficients
                        .as_ref()
                        .ok_or_else(|| missing_workspace_buffer("resident coefficients"))?,
                    workspace
                        .block_column
                        .as_mut()
                        .ok_or_else(|| missing_workspace_buffer("column scratch"))?,
                )
                .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
            copy_device_region(
                backend,
                workspace
                    .block_column
                    .as_ref()
                    .ok_or_else(|| missing_workspace_buffer("column scratch"))?,
                0,
                workspace
                    .block_x
                    .as_mut()
                    .ok_or_else(|| missing_workspace_buffer("solution"))?,
                rhs_col * n,
                n,
            )?;
        }
        a.try_apply_colmajor(
            backend,
            block_size,
            workspace
                .block_x
                .as_ref()
                .ok_or_else(|| missing_workspace_buffer("solution"))?,
            workspace
                .block_ax
                .as_mut()
                .ok_or_else(|| missing_workspace_buffer("operator result"))?,
        )
        .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
        backend
            .try_copy_d2d(
                workspace
                    .block_b
                    .as_ref()
                    .ok_or_else(|| missing_workspace_buffer("RHS"))?,
                workspace
                    .block_residual
                    .as_mut()
                    .ok_or_else(|| missing_workspace_buffer("residual"))?,
            )
            .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
        backend
            .try_axpy(
                T::zero() - T::one(),
                workspace
                    .block_ax
                    .as_ref()
                    .ok_or_else(|| missing_workspace_buffer("operator result"))?,
                workspace
                    .block_residual
                    .as_mut()
                    .ok_or_else(|| missing_workspace_buffer("residual"))?,
            )
            .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
        workspace.host.best_residuals.clear();
        reserve_block_host(
            &mut workspace.host.best_residuals,
            block_size,
            "best block residuals",
        )?;
        for col in 0..block_size {
            copy_device_region(
                backend,
                workspace
                    .block_residual
                    .as_ref()
                    .ok_or_else(|| missing_workspace_buffer("residual"))?,
                col * n,
                workspace
                    .block_column
                    .as_mut()
                    .ok_or_else(|| missing_workspace_buffer("column scratch"))?,
                0,
                n,
            )?;
            let residual_norm = backend
                .try_nrm2_with_mode::<T>(
                    ReductionMode::Fast,
                    workspace
                        .block_column
                        .as_ref()
                        .ok_or_else(|| missing_workspace_buffer("column scratch"))?,
                )
                .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
            let denom = b_norms[active[col]];
            let value = if denom > T::Real::zero() {
                residual_norm / denom
            } else {
                residual_norm
            };
            push_block_host(
                &mut workspace.host.best_residuals,
                value,
                "best block residuals",
            )?;
        }

        if workspace
            .host
            .best_residuals
            .iter()
            .all(|&value| value <= tolerance)
        {
            break;
        }
        if breakdown {
            let result = solve_rank_deficient_columns(
                &mut workspace.fallback,
                backend,
                a,
                precond,
                b,
                workspace
                    .full_x
                    .as_mut()
                    .ok_or_else(|| missing_workspace_buffer("full solution"))?,
                &active,
                n,
                config,
                tolerance,
                out,
            );
            workspace.host.active = active;
            workspace.host.b_norms = b_norms;
            workspace.host.residual_norms = residual_norms;
            return result;
        }
    }

    for (active_col, &full_col) in active.iter().enumerate() {
        copy_device_region(
            backend,
            workspace
                .block_x
                .as_ref()
                .ok_or_else(|| missing_workspace_buffer("solution"))?,
            active_col * n,
            workspace
                .full_x
                .as_mut()
                .ok_or_else(|| missing_workspace_buffer("full solution"))?,
            full_col * n,
            n,
        )?;
    }
    backend
        .try_copy_d2d(
            workspace
                .full_x
                .as_ref()
                .ok_or_else(|| missing_workspace_buffer("full solution"))?,
            out,
        )
        .map_err(|error| BlockGmresError::Backend(error.to_string()))?;

    reserve_block_host(
        &mut workspace.host.full_residuals,
        rhs,
        "block full residuals",
    )?;
    workspace.host.full_residuals.resize(rhs, T::Real::zero());
    for (active_col, &full_col) in active.iter().enumerate() {
        workspace.host.full_residuals[full_col] = workspace.host.best_residuals[active_col];
    }
    workspace.host.active = active;
    workspace.host.b_norms = b_norms;
    workspace.host.residual_norms = residual_norms;
    let status =
        make_status_from_relative(last_iterations, &workspace.host.full_residuals, tolerance)?;
    if status.converged {
        Ok(status)
    } else {
        // A full-rank block may still stagnate before the configured block
        // iteration limit.  Complete the batch with checked scalar solves so
        // non-convergence is reported by the true residual gate rather than
        // returning an unusable partial block iterate.
        solve_rank_deficient_columns(
            &mut workspace.fallback,
            backend,
            a,
            precond,
            b,
            workspace
                .full_x
                .as_mut()
                .ok_or_else(|| missing_workspace_buffer("full solution"))?,
            &workspace.host.active,
            n,
            config,
            tolerance,
            out,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn solve_rank_deficient_columns<B, A, P, T>(
    fallback: &mut BlockFallback<B, T>,
    backend: &B,
    a: &A,
    precond: &P,
    b_full: &B::Buffer<T>,
    x_full: &mut B::Buffer<T>,
    active: &[usize],
    n: usize,
    config: &BlockGmresConfig<T::Real>,
    tolerance: T::Real,
    out: &mut B::Buffer<T>,
) -> Result<BlockGmresStatus<T>, BlockGmresError>
where
    B: Backend,
    A: DeviceOperator<B, T> + ?Sized,
    P: DevicePreconditioner<B, T> + ?Sized,
    T: DeviceScalar + std::ops::AddAssign,
{
    let scalar_config = GmresBackendConfig {
        max_iterations: config.max_block_iterations.max(1),
        tolerance,
    };
    let mut relative_residuals =
        reserved_block_host::<T::Real>(active.len(), "rank-deficient residuals")?;
    relative_residuals.resize(active.len(), T::Real::zero());
    let mut max_iterations = 0usize;
    if fallback.columns.is_none() {
        let allocate = || {
            backend
                .raw_alloc::<T>(n)
                .map_err(|error| BlockGmresError::Allocation {
                    label: "rank-deficient column",
                    reason: error.to_string(),
                })
        };
        fallback.columns = Some([allocate()?, allocate()?, allocate()?]);
    }
    let [b_column, x_initial, x_column] = fallback
        .columns
        .as_mut()
        .ok_or_else(|| missing_workspace_buffer("scalar fallback columns"))?;
    for (active_position, &full_col) in active.iter().enumerate() {
        copy_device_region(backend, b_full, full_col * n, b_column, 0, n)?;
        copy_device_region(backend, x_full, full_col * n, x_initial, 0, n)?;
        let status = if let Some(pool) = &fallback.pool {
            gmres_backend_preconditioned_pooled_with_workspace(
                pool,
                &mut fallback.krylov,
                a,
                precond,
                b_column,
                Some(x_initial),
                x_column,
                &scalar_config,
            )
        } else {
            gmres_backend_preconditioned(
                backend,
                a,
                precond,
                b_column,
                Some(x_initial),
                x_column,
                &scalar_config,
            )
        }
        .map_err(map_rank_deficient_fallback_error)?;
        max_iterations = max_iterations.max(status.iterations);
        relative_residuals[active_position] = status.final_residual;
        copy_device_region(backend, x_column, 0, x_full, full_col * n, n)?;
    }
    backend
        .try_copy_d2d(x_full, out)
        .map_err(|error| BlockGmresError::Backend(error.to_string()))?;

    let full_rhs = b_full.len() / n;
    let mut full_residuals =
        reserved_block_host::<T::Real>(full_rhs, "rank-deficient full residuals")?;
    full_residuals.resize(full_rhs, T::Real::zero());
    for (active_position, &full_col) in active.iter().enumerate() {
        full_residuals[full_col] = relative_residuals[active_position];
    }
    let mut result = make_status_from_relative(max_iterations, &full_residuals, tolerance)?;
    // The scalar fallback's status is already per-RHS; keeping the block
    // result shape allows callers to handle it exactly like a regular batch.
    result.block_iterations = max_iterations;
    Ok(result)
}

fn map_rank_deficient_fallback_error(error: GmresBackendError) -> BlockGmresError {
    match error {
        // A rank-deficient initial block only selects this scalar fallback; it
        // must not rewrite an allocation failure as a numerical-rank diagnosis.
        // Preserve the typed allocation contract so callers can retry or report
        // resource pressure.
        GmresBackendError::Allocation { label, reason } => {
            BlockGmresError::Allocation { label, reason }
        }
        // The remaining scalar-GMRES failures (backend, workspace, zero-RHS,
        // and happy-breakdown) are runtime failures of the fallback, not
        // evidence that the initial block rank was the cause.
        other => BlockGmresError::Backend(other.to_string()),
    }
}

fn make_status<T: DeviceScalar>(
    iterations: usize,
    b_norms: &[T::Real],
    residual_norms: &[T::Real],
    tolerance: T::Real,
    rhs: usize,
) -> Result<BlockGmresStatus<T>, BlockGmresError> {
    let mut residuals = reserved_block_host(rhs, "block status residuals")?;
    residuals.resize(rhs, T::Real::zero());
    for (col, residual) in residuals.iter_mut().enumerate() {
        *residual = if b_norms[col] > T::Real::zero() {
            residual_norms[col] / b_norms[col]
        } else {
            residual_norms[col]
        };
    }
    make_status_from_relative(iterations, &residuals, tolerance)
}

fn make_status_from_relative<T: DeviceScalar>(
    iterations: usize,
    residuals: &[T::Real],
    tolerance: T::Real,
) -> Result<BlockGmresStatus<T>, BlockGmresError> {
    let final_residual = residuals
        .iter()
        .copied()
        .fold(T::Real::zero(), T::Real::max);
    let mut rhs = reserved_block_host(residuals.len(), "block status RHS")?;
    for &residual in residuals {
        push_block_host(
            &mut rhs,
            GmresBackendStatus {
                iterations,
                projected_residual: residual,
                final_residual: residual,
                backward_error: residual,
                converged: residual <= tolerance,
            },
            "block status RHS",
        )?;
    }
    Ok(BlockGmresStatus {
        block_iterations: iterations,
        projected_residual: final_residual,
        final_residual,
        backward_error: final_residual,
        converged: final_residual <= tolerance,
        rhs,
    })
}

// Two-pass block orthogonalization with only the small R factor on the host.
#[expect(
    clippy::too_many_arguments,
    reason = "explicit disjoint device scratch and QR shape"
)]
fn qr_device_block_into<B: Backend, T: DeviceScalar>(
    backend: &B,
    input: &B::Buffer<T>,
    q: &mut B::Buffer<T>,
    column: &mut B::Buffer<T>,
    n: usize,
    block_size: usize,
    tolerance: T::Real,
    r: &mut Vec<T>,
    coefficients: &mut Vec<T>,
) -> Result<usize, BlockGmresError> {
    let count = block_size.checked_mul(block_size).ok_or_else(|| {
        BlockGmresError::InvalidArgument("initial QR coefficient size overflow".into())
    })?;
    reserve_block_host(r, count, "initial QR coefficients")?;
    r.resize(count, T::zero());
    r.fill(T::zero());
    reserve_block_host(coefficients, block_size, "initial QR projection")?;
    let mut rank = 0;
    for col in 0..block_size {
        copy_device_region(backend, input, col * n, column, 0, n)?;
        coefficients.resize(col, T::zero());
        if col > 0 {
            for _ in 0..2 {
                backend
                    .try_multi_dot_strided_with_mode_into(
                        ReductionMode::Fast,
                        q,
                        n,
                        col,
                        column,
                        coefficients,
                    )
                    .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
                for (prev, &value) in coefficients.iter().enumerate() {
                    r[prev * block_size + col] = r[prev * block_size + col] + value;
                }
                backend
                    .try_basis_axpy_strided(T::zero() - T::one(), q, n, col, coefficients, column)
                    .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
            }
        }
        let norm = backend
            .try_nrm2_with_mode(ReductionMode::Fast, column)
            .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
        let scale = if norm > tolerance {
            rank += 1;
            r[col * block_size + col] = T::from_real(norm);
            T::from_real(T::Real::one() / norm)
        } else {
            // A finite rejected column must overwrite any old basis contents.
            T::zero()
        };
        backend
            .try_scale(scale, column)
            .map_err(|error| BlockGmresError::Backend(error.to_string()))?;
        copy_device_region(backend, column, 0, q, col * n, n)?;
    }
    Ok(rank)
}

#[cfg(test)]
fn qr_block_into<T: DeviceScalar>(
    input: &[T],
    n: usize,
    block_size: usize,
    tolerance: T::Real,
    q: &mut Vec<T>,
    r: &mut Vec<T>,
    work: &mut Vec<T>,
) -> Result<usize, BlockGmresError> {
    let q_len = n
        .checked_mul(block_size)
        .ok_or_else(|| BlockGmresError::InvalidArgument("block QR basis size overflow".into()))?;
    let r_len = block_size.checked_mul(block_size).ok_or_else(|| {
        BlockGmresError::InvalidArgument("block QR coefficient size overflow".into())
    })?;
    reserve_block_host(q, q_len, "block QR basis")?;
    reserve_block_host(r, r_len, "block QR coefficients")?;
    reserve_block_host(work, n, "block QR work")?;
    q.resize(q_len, T::zero());
    r.resize(r_len, T::zero());
    work.resize(n, T::zero());
    q.fill(T::zero());
    r.fill(T::zero());
    let mut rank = 0;
    for col in 0..block_size {
        work.copy_from_slice(&input[col * n..(col + 1) * n]);
        for _ in 0..2 {
            for prev in 0..col {
                let coeff = column_dot(q, work, n, block_size, prev);
                r[prev * block_size + col] = r[prev * block_size + col] + coeff;
                for row in 0..n {
                    work[row] = work[row] - q[prev * n + row] * coeff;
                }
            }
        }
        let norm = work
            .iter()
            .map(|value| value.norm_sqr())
            .fold(T::Real::zero(), |sum, value| sum + value)
            .sqrt();
        if norm > tolerance {
            rank += 1;
            r[col * block_size + col] = T::from_real(norm);
            for row in 0..n {
                q[col * n + row] = work[row] / T::from_real(norm);
            }
        }
    }
    Ok(rank)
}

#[cfg(test)]
fn qr_block<T: DeviceScalar>(
    input: &[T],
    n: usize,
    block_size: usize,
    tolerance: T::Real,
) -> Result<(Vec<T>, Vec<T>, usize), BlockGmresError> {
    let mut q = Vec::new();
    let mut r = Vec::new();
    let mut work = Vec::new();
    let rank = qr_block_into(input, n, block_size, tolerance, &mut q, &mut r, &mut work)?;
    Ok((q, r, rank))
}

#[cfg(test)]
fn column_dot<T: DeviceScalar>(q: &[T], w: &[T], n: usize, _block_size: usize, col: usize) -> T {
    (0..n)
        .map(|row| q[col * n + row].conj_mul(w[row]))
        .fold(T::zero(), |sum, value| sum + value)
}

#[cfg(test)]
fn block_inner_into<T: DeviceScalar>(
    left: &[T],
    right: &[T],
    n: usize,
    block_size: usize,
    result: &mut Vec<T>,
) -> Result<(), BlockGmresError> {
    let len = block_size.checked_mul(block_size).ok_or_else(|| {
        BlockGmresError::InvalidArgument("block inner-product size overflow".into())
    })?;
    reserve_block_host(result, len, "block inner product")?;
    result.resize(len, T::zero());
    for left_col in 0..block_size {
        for right_col in 0..block_size {
            result[left_col * block_size + right_col] = (0..n)
                .map(|row| left[left_col * n + row].conj_mul(right[right_col * n + row]))
                .fold(T::zero(), |sum, value| sum + value);
        }
    }
    Ok(())
}

#[cfg(test)]
fn block_inner<T: DeviceScalar>(
    left: &[T],
    right: &[T],
    n: usize,
    block_size: usize,
) -> Result<Vec<T>, BlockGmresError> {
    let mut result = Vec::new();
    block_inner_into(left, right, n, block_size, &mut result)?;
    Ok(result)
}

fn flatten_h_into<T: DeviceScalar>(
    h_blocks: &[Vec<Vec<T>>],
    row_blocks: usize,
    column_blocks: usize,
    block_size: usize,
    flat: &mut Vec<T>,
) -> Result<(), BlockGmresError> {
    let rows = row_blocks.checked_mul(block_size).ok_or_else(|| {
        BlockGmresError::InvalidArgument("block Hessenberg row size overflow".into())
    })?;
    let cols = column_blocks.checked_mul(block_size).ok_or_else(|| {
        BlockGmresError::InvalidArgument("block Hessenberg column size overflow".into())
    })?;
    let len = rows.checked_mul(cols).ok_or_else(|| {
        BlockGmresError::InvalidArgument("block Hessenberg storage size overflow".into())
    })?;
    let block_len = block_size.checked_mul(block_size).ok_or_else(|| {
        BlockGmresError::InvalidArgument("block Hessenberg block size overflow".into())
    })?;
    reserve_block_host(flat, len, "block Hessenberg flattened")?;
    flat.resize(len, T::zero());
    for block_row in 0..row_blocks {
        for block_col in 0..column_blocks {
            let row = h_blocks.get(block_row).ok_or_else(|| {
                BlockGmresError::InvalidArgument(format!(
                    "block Hessenberg row {block_row} is missing (available rows: {})",
                    h_blocks.len()
                ))
            })?;
            let block = row.get(block_col).ok_or_else(|| {
                BlockGmresError::InvalidArgument(format!(
                    "block Hessenberg block ({block_row}, {block_col}) is missing (available columns: {})",
                    row.len()
                ))
            })?;
            if block.len() != block_len {
                return Err(BlockGmresError::InvalidArgument(format!(
                    "block Hessenberg block ({block_row}, {block_col}) has {} entries, expected {}",
                    block.len(),
                    block_len
                )));
            }
            for row in 0..block_size {
                for col in 0..block_size {
                    flat[(block_row * block_size + row) * cols + block_col * block_size + col] =
                        block[row * block_size + col];
                }
            }
        }
    }
    Ok(())
}

/// Solve a small dense least-squares problem using normal equations.
///
/// The dimensions are at most `(restart * block_size)`, so this host-side
/// fallback avoids pulling a dense linear algebra dependency into the backend
/// crate.  Block Arnoldi's orthonormal basis keeps the normal equations small;
/// a pivot gate rejects numerically singular recurrences.
#[derive(Debug)]
enum LeastSquaresError {
    Allocation(BlockGmresError),
    Singular,
}

#[cfg(test)]
fn least_squares<T: DeviceScalar>(
    matrix: &[T],
    rows: usize,
    cols: usize,
    rhs: &[T],
    rhs_cols: usize,
) -> Result<Vec<T>, LeastSquaresError> {
    let normal_len = cols.checked_mul(cols).ok_or_else(|| {
        LeastSquaresError::Allocation(BlockGmresError::InvalidArgument(
            "block least-squares normal size overflow".into(),
        ))
    })?;
    let normal_rhs_len = cols.checked_mul(rhs_cols).ok_or_else(|| {
        LeastSquaresError::Allocation(BlockGmresError::InvalidArgument(
            "block least-squares RHS size overflow".into(),
        ))
    })?;
    let mut normal = zeroed_block_host::<T>(normal_len, "block least-squares normal")
        .map_err(LeastSquaresError::Allocation)?;
    let mut normal_rhs = zeroed_block_host::<T>(normal_rhs_len, "block least-squares RHS")
        .map_err(LeastSquaresError::Allocation)?;
    for row in 0..rows {
        for col in 0..cols {
            let left = matrix[row * cols + col];
            for other in 0..cols {
                normal[col * cols + other] =
                    normal[col * cols + other] + left.conj_mul(matrix[row * cols + other]);
            }
            for rhs_col in 0..rhs_cols {
                normal_rhs[col * rhs_cols + rhs_col] = normal_rhs[col * rhs_cols + rhs_col]
                    + left.conj_mul(rhs[row * rhs_cols + rhs_col]);
            }
        }
    }
    let mut max_pivot = T::Real::zero();
    for col in 0..cols {
        max_pivot = max_pivot.max(normal[col * cols + col].norm_sqr().sqrt());
    }
    let pivot_floor = T::Real::epsilon()
        * max_pivot.max(T::Real::one())
        * T::Real::from_f64(100.0).unwrap_or_else(T::Real::one);

    for pivot in 0..cols {
        let mut pivot_row = pivot;
        let mut pivot_norm = normal[pivot * cols + pivot].norm_sqr();
        for row in pivot + 1..cols {
            let candidate = normal[row * cols + pivot].norm_sqr();
            if candidate > pivot_norm {
                pivot_row = row;
                pivot_norm = candidate;
            }
        }
        if pivot_norm.sqrt() <= pivot_floor {
            return Err(LeastSquaresError::Singular);
        }
        if pivot_row != pivot {
            for col in pivot..cols {
                normal.swap(pivot * cols + col, pivot_row * cols + col);
            }
            for rhs_col in 0..rhs_cols {
                normal_rhs.swap(pivot * rhs_cols + rhs_col, pivot_row * rhs_cols + rhs_col);
            }
        }
        let pivot_value = normal[pivot * cols + pivot];
        for row in pivot + 1..cols {
            let factor = normal[row * cols + pivot] / pivot_value;
            normal[row * cols + pivot] = T::zero();
            for col in pivot + 1..cols {
                normal[row * cols + col] =
                    normal[row * cols + col] - factor * normal[pivot * cols + col];
            }
            for rhs_col in 0..rhs_cols {
                normal_rhs[row * rhs_cols + rhs_col] = normal_rhs[row * rhs_cols + rhs_col]
                    - factor * normal_rhs[pivot * rhs_cols + rhs_col];
            }
        }
    }

    let solution_len = cols.checked_mul(rhs_cols).ok_or_else(|| {
        LeastSquaresError::Allocation(BlockGmresError::InvalidArgument(
            "block least-squares solution size overflow".into(),
        ))
    })?;
    let mut solution = zeroed_block_host::<T>(solution_len, "block least-squares solution")
        .map_err(LeastSquaresError::Allocation)?;
    for row in (0..cols).rev() {
        for rhs_col in 0..rhs_cols {
            let mut value = normal_rhs[row * rhs_cols + rhs_col];
            for col in row + 1..cols {
                value = value - normal[row * cols + col] * solution[col * rhs_cols + rhs_col];
            }
            solution[row * rhs_cols + rhs_col] = value / normal[row * cols + row];
        }
    }
    Ok(solution)
}

/// In-place variant of [`least_squares`] used by the warm block-GMRES path.
///
/// The normal matrix, right-hand side, and solution are retained by the
/// caller-owned workspace.  Only their logical lengths are changed between
/// Arnoldi steps, so the steady-state recurrence does not allocate host
/// vectors for each least-squares solve.
#[allow(clippy::too_many_arguments)]
fn least_squares_into<T: DeviceScalar>(
    matrix: &[T],
    rows: usize,
    cols: usize,
    rhs: &[T],
    rhs_cols: usize,
    normal: &mut Vec<T>,
    normal_rhs: &mut Vec<T>,
    solution: &mut Vec<T>,
) -> Result<(), LeastSquaresError> {
    let matrix_len = rows.checked_mul(cols).ok_or_else(|| {
        LeastSquaresError::Allocation(BlockGmresError::InvalidArgument(
            "block least-squares matrix size overflow".into(),
        ))
    })?;
    let normal_len = cols.checked_mul(cols).ok_or_else(|| {
        LeastSquaresError::Allocation(BlockGmresError::InvalidArgument(
            "block least-squares normal size overflow".into(),
        ))
    })?;
    let normal_rhs_len = cols.checked_mul(rhs_cols).ok_or_else(|| {
        LeastSquaresError::Allocation(BlockGmresError::InvalidArgument(
            "block least-squares RHS size overflow".into(),
        ))
    })?;
    let solution_len = normal_rhs_len;
    if matrix.len() < matrix_len
        || rhs.len()
            < rows.checked_mul(rhs_cols).ok_or_else(|| {
                LeastSquaresError::Allocation(BlockGmresError::InvalidArgument(
                    "block least-squares input RHS size overflow".into(),
                ))
            })?
    {
        return Err(LeastSquaresError::Allocation(
            BlockGmresError::InvalidArgument(
                "block least-squares input buffers are too short".into(),
            ),
        ));
    }
    reserve_block_host(normal, normal_len, "block least-squares normal")
        .map_err(LeastSquaresError::Allocation)?;
    reserve_block_host(normal_rhs, normal_rhs_len, "block least-squares RHS")
        .map_err(LeastSquaresError::Allocation)?;
    reserve_block_host(solution, solution_len, "block least-squares solution")
        .map_err(LeastSquaresError::Allocation)?;
    normal.resize(normal_len, T::zero());
    normal_rhs.resize(normal_rhs_len, T::zero());
    solution.resize(solution_len, T::zero());
    normal.fill(T::zero());
    normal_rhs.fill(T::zero());
    solution.fill(T::zero());

    for row in 0..rows {
        for col in 0..cols {
            let left = matrix[row * cols + col];
            for other in 0..cols {
                normal[col * cols + other] =
                    normal[col * cols + other] + left.conj_mul(matrix[row * cols + other]);
            }
            for rhs_col in 0..rhs_cols {
                normal_rhs[col * rhs_cols + rhs_col] = normal_rhs[col * rhs_cols + rhs_col]
                    + left.conj_mul(rhs[row * rhs_cols + rhs_col]);
            }
        }
    }

    let mut max_pivot = T::Real::zero();
    for col in 0..cols {
        max_pivot = max_pivot.max(normal[col * cols + col].norm_sqr().sqrt());
    }
    let max_pivot = max_pivot.max(T::Real::one());
    let pivot_tolerance =
        max_pivot * T::Real::epsilon() * T::Real::from_f64(100.0).unwrap_or_else(T::Real::one);

    for pivot in 0..cols {
        let (pivot_row, pivot_norm) = (pivot..cols)
            .map(|row| (row, normal[row * cols + pivot].norm_sqr().sqrt()))
            .max_by(|(_, left), (_, right)| {
                left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or((pivot, T::Real::zero()));
        if pivot_norm <= pivot_tolerance {
            return Err(LeastSquaresError::Singular);
        }
        if pivot_row != pivot {
            for col in pivot..cols {
                normal.swap(pivot * cols + col, pivot_row * cols + col);
            }
            for rhs_col in 0..rhs_cols {
                normal_rhs.swap(pivot * rhs_cols + rhs_col, pivot_row * rhs_cols + rhs_col);
            }
        }
        let pivot_value = normal[pivot * cols + pivot];
        for row in pivot + 1..cols {
            let factor = normal[row * cols + pivot] / pivot_value;
            normal[row * cols + pivot] = T::zero();
            for col in pivot + 1..cols {
                normal[row * cols + col] =
                    normal[row * cols + col] - factor * normal[pivot * cols + col];
            }
            for rhs_col in 0..rhs_cols {
                normal_rhs[row * rhs_cols + rhs_col] = normal_rhs[row * rhs_cols + rhs_col]
                    - factor * normal_rhs[pivot * rhs_cols + rhs_col];
            }
        }
    }

    for row in (0..cols).rev() {
        for rhs_col in 0..rhs_cols {
            let mut value = normal_rhs[row * rhs_cols + rhs_col];
            for col in row + 1..cols {
                value = value - normal[row * cols + col] * solution[col * rhs_cols + rhs_col];
            }
            solution[row * rhs_cols + rhs_col] = value / normal[row * cols + row];
        }
    }
    Ok(())
}

#[cfg(test)]
// Fixtures intentionally use the panicking convenience constructors on
// tiny fixed matrices; production paths must use the `try_*` variants.
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::{CpuBackend, DeviceCsr, IdentityDevicePreconditioner};
    use num_complex::{Complex32, Complex64};

    #[test]
    fn block_gmres_default_config_is_finite_without_unchecked_conversion() {
        let config = BlockGmresConfig::<f64>::default();

        assert!(config.tolerance.is_finite());
        assert_eq!(config.tolerance, 1.0e-8);
    }

    #[test]
    fn block_gmres_rejects_invalid_tolerance_before_execution() {
        let backend = CpuBackend::new();
        let matrix = DeviceCsr::from_host(&backend, 1, 1, &[0, 1], &[0], &[2.0_f64]);
        let mut rhs = backend.raw_alloc::<f64>(1).expect("RHS allocation");
        backend.copy_h2d(&[4.0], &mut rhs);

        for tolerance in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
            let mut out = backend.raw_alloc::<f64>(1).expect("output allocation");
            backend.copy_h2d(&[17.0], &mut out);
            let config = BlockGmresConfig {
                max_block_iterations: 1,
                tolerance,
            };
            let error = gmres_backend_block_preconditioned(
                &backend,
                &matrix,
                &IdentityDevicePreconditioner,
                &rhs,
                None,
                &mut out,
                1,
                &config,
            )
            .expect_err("invalid tolerance must be rejected before execution");
            assert!(
                matches!(error, BlockGmresError::InvalidArgument(message) if message.contains("finite"))
            );

            let mut observed = [0.0];
            backend.copy_d2h(&out, &mut observed);
            assert_eq!(observed, [17.0], "invalid tolerance mutated output");
        }
    }

    #[test]
    fn block_gmres_rejects_zero_iteration_budget_before_execution() {
        let backend = CpuBackend::new();
        let matrix = DeviceCsr::from_host(&backend, 1, 1, &[0, 1], &[0], &[2.0_f64]);
        let mut rhs = backend.raw_alloc::<f64>(1).expect("RHS allocation");
        backend.copy_h2d(&[4.0], &mut rhs);
        let mut out = backend.raw_alloc::<f64>(1).expect("output allocation");
        backend.copy_h2d(&[17.0], &mut out);

        let error = gmres_backend_block_preconditioned(
            &backend,
            &matrix,
            &IdentityDevicePreconditioner,
            &rhs,
            None,
            &mut out,
            1,
            &BlockGmresConfig {
                max_block_iterations: 0,
                tolerance: 1.0e-8,
            },
        )
        .expect_err("zero iteration budget must be rejected");

        assert!(
            matches!(error, BlockGmresError::InvalidArgument(message) if message.contains("positive"))
        );
        let mut observed = [0.0];
        backend.copy_d2h(&out, &mut observed);
        assert_eq!(observed, [17.0], "invalid configuration mutated output");
    }

    #[test]
    fn block_gmres_rejects_non_finite_rhs_and_initial_guess() {
        let backend = CpuBackend::new();
        let matrix = DeviceCsr::from_host(&backend, 1, 1, &[0, 1], &[0], &[2.0_f64]);
        let mut rhs = backend.raw_alloc::<f64>(1).expect("RHS allocation");
        let mut out = backend.raw_alloc::<f64>(1).expect("output allocation");
        backend.copy_h2d(&[17.0], &mut out);
        let config = BlockGmresConfig {
            max_block_iterations: 1,
            tolerance: 1.0e-8,
        };

        backend.copy_h2d(&[f64::NAN], &mut rhs);
        let error = gmres_backend_block_preconditioned(
            &backend,
            &matrix,
            &IdentityDevicePreconditioner,
            &rhs,
            None,
            &mut out,
            1,
            &config,
        )
        .expect_err("non-finite RHS must be rejected");
        assert!(
            matches!(error, BlockGmresError::InvalidArgument(message) if message.contains("RHS"))
        );

        let mut finite_rhs = backend.raw_alloc::<f64>(1).expect("finite RHS allocation");
        backend.copy_h2d(&[4.0], &mut finite_rhs);
        let mut initial_guess = backend
            .raw_alloc::<f64>(1)
            .expect("initial guess allocation");
        backend.copy_h2d(&[f64::INFINITY], &mut initial_guess);
        let error = gmres_backend_block_preconditioned(
            &backend,
            &matrix,
            &IdentityDevicePreconditioner,
            &finite_rhs,
            Some(&initial_guess),
            &mut out,
            1,
            &config,
        )
        .expect_err("non-finite initial guess must be rejected");
        assert!(
            matches!(error, BlockGmresError::InvalidArgument(message) if message.contains("initial guess"))
        );

        let mut observed = [0.0];
        backend.copy_d2h(&out, &mut observed);
        assert_eq!(observed, [17.0], "non-finite inputs mutated output");
    }

    #[test]
    fn block_host_capacity_failure_is_typed() {
        assert!(matches!(
            reserved_block_host::<u8>(usize::MAX, "block metadata"),
            Err(BlockGmresError::Allocation {
                label: "block metadata",
                ..
            })
        ));
    }

    #[test]
    fn block_helper_size_failures_are_typed() {
        assert!(matches!(
            qr_block::<f64>(&[], usize::MAX, 2, 1.0e-8),
            Err(BlockGmresError::InvalidArgument(message))
                if message.contains("block QR basis size overflow")
        ));
        assert!(matches!(
            block_inner::<f64>(&[], &[], 1, usize::MAX),
            Err(BlockGmresError::InvalidArgument(message))
                if message.contains("block inner-product size overflow")
        ));
        let mut flattened = Vec::<f64>::new();
        assert!(matches!(
            flatten_h_into(&[], usize::MAX, 1, 2, &mut flattened),
            Err(BlockGmresError::InvalidArgument(message))
                if message.contains("block Hessenberg row size overflow")
        ));
        let malformed_hessenberg = vec![vec![Vec::<f64>::new()]];
        assert!(matches!(
            flatten_h_into(&malformed_hessenberg, 1, 1, 1, &mut flattened),
            Err(BlockGmresError::InvalidArgument(message))
                if message.contains("block Hessenberg block (0, 0) has 0 entries")
        ));
        assert!(matches!(
            least_squares::<f64>(&[], 0, usize::MAX, &[], 1),
            Err(LeastSquaresError::Allocation(BlockGmresError::InvalidArgument(message)))
                if message.contains("block least-squares normal size overflow")
        ));
    }

    #[test]
    fn checked_block_transfers_map_shape_errors_without_panicking() {
        let backend = CpuBackend::new();
        let mut device = backend.raw_alloc::<f64>(2).unwrap();

        let h2d_error = copy_h2d_checked(&backend, &[1.0], &mut device).unwrap_err();
        assert!(
            matches!(h2d_error, BlockGmresError::Backend(reason) if reason.contains("copy_h2d"))
        );

        let mut host = [0.0_f64];
        let d2h_error = copy_d2h_checked(&backend, &device, &mut host).unwrap_err();
        assert!(
            matches!(d2h_error, BlockGmresError::Backend(reason) if reason.contains("copy_d2h"))
        );
    }

    #[test]
    fn block_gmres_solves_two_real_rhs_with_one_operator_path() {
        let backend = CpuBackend::new();
        let matrix = DeviceCsr::from_host(
            &backend,
            2,
            2,
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[4.0_f64, 1.0, 1.0, 3.0],
        );
        let b = backend.raw_alloc::<f64>(4).unwrap();
        let mut b = b;
        backend.copy_h2d(&[9.0, 8.0, 3.0, 7.0], &mut b);
        let mut out = backend.raw_alloc::<f64>(4).unwrap();
        let config = BlockGmresConfig {
            max_block_iterations: 4,
            tolerance: 1.0e-10,
        };
        let status = gmres_backend_block_preconditioned(
            &backend,
            &matrix,
            &IdentityDevicePreconditioner,
            &b,
            None,
            &mut out,
            2,
            &config,
        )
        .unwrap();
        let mut values = [0.0; 4];
        backend.copy_d2h(&out, &mut values);
        assert!(status.converged, "{status:?}");
        assert!(status.rhs.iter().all(|rhs| rhs.converged));
        assert!((values[0] - 19.0 / 11.0).abs() < 1.0e-8);
        assert!((values[1] - 23.0 / 11.0).abs() < 1.0e-8);
        assert!((values[2] - 2.0 / 11.0).abs() < 1.0e-8);
        assert!((values[3] - 25.0 / 11.0).abs() < 1.0e-8);
    }

    #[test]
    fn block_gmres_workspace_reuses_and_clears_device_buffers() {
        let backend = CpuBackend::new();
        let matrix = DeviceCsr::from_host(
            &backend,
            2,
            2,
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[4.0_f64, 1.0, 1.0, 3.0],
        );
        let mut b = backend.raw_alloc::<f64>(4).unwrap();
        backend.copy_h2d(&[9.0, 8.0, 3.0, 7.0], &mut b);
        let mut out = backend.raw_alloc::<f64>(4).unwrap();
        let config = BlockGmresConfig {
            max_block_iterations: 4,
            tolerance: 1.0e-10,
        };
        let mut workspace = BlockGmresWorkspace::new(2);

        let first = gmres_backend_block_preconditioned_with_workspace(
            &backend,
            &matrix,
            &IdentityDevicePreconditioner,
            &b,
            None,
            &mut out,
            2,
            &config,
            &mut workspace,
        )
        .unwrap();
        assert!(first.converged, "{first:?}");
        let count = workspace.device_buffer_count();
        let capacity = workspace.device_capacity();
        let host_capacity = workspace.host_capacity();
        assert!(count > 7, "resident Arnoldi state must be retained");
        assert!(capacity > 0);
        assert!(host_capacity > 0);

        let second = gmres_backend_block_preconditioned_with_workspace(
            &backend,
            &matrix,
            &IdentityDevicePreconditioner,
            &b,
            None,
            &mut out,
            2,
            &config,
            &mut workspace,
        )
        .unwrap();
        assert!(second.converged, "{second:?}");
        assert_eq!(workspace.device_buffer_count(), count);
        assert_eq!(workspace.device_capacity(), capacity);
        assert_eq!(workspace.host_capacity(), host_capacity);

        workspace.clear();
        assert_eq!(workspace.device_buffer_count(), 0);
        assert_eq!(workspace.device_capacity(), 0);
        assert_eq!(workspace.host_capacity(), 0);

        let mut mismatch = BlockGmresWorkspace::new(3);
        assert!(matches!(
            gmres_backend_block_preconditioned_with_workspace(
                &backend,
                &matrix,
                &IdentityDevicePreconditioner,
                &b,
                None,
                &mut out,
                2,
                &config,
                &mut mismatch,
            ),
            Err(BlockGmresError::InvalidArgument(message))
                if message.contains("workspace dimension mismatch")
        ));
    }

    #[test]
    fn block_gmres_supports_f32() {
        let backend = CpuBackend::new();
        let matrix = DeviceCsr::from_host(
            &backend,
            2,
            2,
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[4.0_f32, 1.0, 1.0, 3.0],
        );
        let mut b = backend.raw_alloc::<f32>(4).unwrap();
        backend.copy_h2d(&[9.0, 8.0, 3.0, 7.0], &mut b);
        let mut out = backend.raw_alloc::<f32>(4).unwrap();
        let status = gmres_backend_block_preconditioned(
            &backend,
            &matrix,
            &IdentityDevicePreconditioner,
            &b,
            None,
            &mut out,
            2,
            &BlockGmresConfig {
                max_block_iterations: 4,
                tolerance: 1.0e-5,
            },
        )
        .unwrap();
        let mut values = [0.0_f32; 4];
        backend.copy_d2h(&out, &mut values);
        assert!(status.converged, "{status:?}");
        assert!((values[0] - 19.0 / 11.0).abs() < 1.0e-4);
        assert!((values[1] - 23.0 / 11.0).abs() < 1.0e-4);
        assert!((values[2] - 2.0 / 11.0).abs() < 1.0e-4);
        assert!((values[3] - 25.0 / 11.0).abs() < 1.0e-4);
    }

    #[test]
    fn block_gmres_supports_complex32_and_deflates_zero_rhs() {
        let backend = CpuBackend::new();
        let matrix = DeviceCsr::from_host(
            &backend,
            2,
            2,
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[
                Complex32::new(2.0, 0.5),
                Complex32::new(0.0, 1.0),
                Complex32::new(0.0, -1.0),
                Complex32::new(3.0, 0.0),
            ],
        );
        let mut b = backend.raw_alloc::<Complex32>(4).unwrap();
        backend.copy_h2d(
            &[
                Complex32::new(1.0, 0.0),
                Complex32::new(2.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(0.0, 0.0),
            ],
            &mut b,
        );
        let mut out = backend.raw_alloc::<Complex32>(4).unwrap();
        let status = gmres_backend_block_preconditioned(
            &backend,
            &matrix,
            &IdentityDevicePreconditioner,
            &b,
            None,
            &mut out,
            2,
            &BlockGmresConfig {
                max_block_iterations: 4,
                tolerance: 1.0e-5,
            },
        )
        .unwrap();
        let mut values = [Complex32::new(0.0, 0.0); 4];
        backend.copy_d2h(&out, &mut values);
        assert!(status.converged, "{status:?}");
        assert_eq!(values[2], Complex32::new(0.0, 0.0));
        assert_eq!(values[3], Complex32::new(0.0, 0.0));
    }

    #[test]
    fn block_gmres_supports_complex64() {
        let backend = CpuBackend::new();
        let matrix = DeviceCsr::from_host(
            &backend,
            2,
            2,
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[
                Complex64::new(2.0, 0.5),
                Complex64::new(0.0, 1.0),
                Complex64::new(0.0, -1.0),
                Complex64::new(3.0, 0.0),
            ],
        );
        let mut b = backend.raw_alloc::<Complex64>(4).unwrap();
        backend.copy_h2d(
            &[
                Complex64::new(0.375, -0.25),
                Complex64::new(-2.75, 1.0),
                Complex64::new(-0.375, 1.625),
                Complex64::new(1.5, -1.25),
            ],
            &mut b,
        );
        let mut out = backend.raw_alloc::<Complex64>(4).unwrap();
        let status = gmres_backend_block_preconditioned(
            &backend,
            &matrix,
            &IdentityDevicePreconditioner,
            &b,
            None,
            &mut out,
            2,
            &BlockGmresConfig {
                max_block_iterations: 4,
                tolerance: 1.0e-10,
            },
        )
        .unwrap();
        let mut values = [Complex64::new(0.0, 0.0); 4];
        backend.copy_d2h(&out, &mut values);
        assert!(status.converged, "{status:?}");
        let expected = [
            Complex64::new(0.5, 0.25),
            Complex64::new(-1.0, 0.5),
            Complex64::new(-0.25, 0.75),
            Complex64::new(0.25, -0.5),
        ];
        for (actual, expected) in values.into_iter().zip(expected) {
            assert!((actual - expected).norm() < 1.0e-8);
        }
    }

    #[test]
    fn block_gmres_deflates_dependent_initial_residuals() {
        let backend = CpuBackend::new();
        let matrix = DeviceCsr::from_host(
            &backend,
            2,
            2,
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[2.0_f64, 0.0, 0.0, 3.0],
        );
        let mut b = backend.raw_alloc::<f64>(4).unwrap();
        backend.copy_h2d(&[2.0, 3.0, 4.0, 6.0], &mut b);
        let mut out = backend.raw_alloc::<f64>(4).unwrap();
        let status = gmres_backend_block_preconditioned(
            &backend,
            &matrix,
            &IdentityDevicePreconditioner,
            &b,
            None,
            &mut out,
            2,
            &BlockGmresConfig {
                max_block_iterations: 4,
                tolerance: 1.0e-10,
            },
        )
        .unwrap();
        let mut values = [0.0; 4];
        backend.copy_d2h(&out, &mut values);
        assert!(status.converged, "{status:?}");
        assert_eq!(status.rhs.len(), 2);
        assert!((values[0] - 1.0).abs() < 1.0e-10);
        assert!((values[1] - 1.0).abs() < 1.0e-10);
        assert!((values[2] - 2.0).abs() < 1.0e-10);
        assert!((values[3] - 2.0).abs() < 1.0e-10);
    }

    #[test]
    fn rank_deficient_fallback_preserves_scalar_error_taxonomy() {
        let allocation = map_rank_deficient_fallback_error(GmresBackendError::Allocation {
            label: "rank-deficient RHS",
            reason: "injected allocation failure".into(),
        });
        assert!(matches!(
            allocation,
            BlockGmresError::Allocation {
                label: "rank-deficient RHS",
                reason
            } if reason == "injected allocation failure"
        ));

        let backend = map_rank_deficient_fallback_error(GmresBackendError::Backend {
            source: crate::BackendError::OperationFailed {
                operation: "test_scalar_operator",
                reason: "injected scalar fallback failure".into(),
            },
        });
        assert!(matches!(
            backend,
            BlockGmresError::Backend(message)
                if message.contains("test_scalar_operator")
        ));
    }

    #[test]
    fn complex64_least_squares_keeps_hermitian_products() {
        let matrix = [
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(1.0, -1.0),
            Complex64::new(0.0, -1.0),
        ];
        let rhs = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
        let result = least_squares(&matrix, 2, 1, &rhs, 1).unwrap();
        assert!(result[0].norm() > 0.0);
    }
}

#[cfg(test)]
mod initial_qr_tests;
