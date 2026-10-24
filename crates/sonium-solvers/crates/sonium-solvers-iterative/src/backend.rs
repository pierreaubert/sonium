//! Backend-resident Krylov entry points.
//!
//! The ordinary solver APIs in this crate operate on host slices for
//! compatibility.  This module is the explicit accelerator boundary: all
//! vectors passed to these functions are [`sonium_compute`] device buffers,
//! capability admission happens before the first iteration, and the only
//! values returned to the host are the small convergence/status records and
//! the caller-owned final output buffer.
//!
//! The compute substrate owns the backend kernels and reusable workspaces;
//! this module owns the iterative-family contract and prevents an implicit
//! host-staged fallback from being reported as a CUDA solve.

// The checked error deliberately carries the complete diagnostic tuple by
// value so callers can inspect it without another fallible lookup. Keep the
// API shape stable; the larger error is an intentional trade-off here.
#![allow(clippy::result_large_err)]

#[cfg(test)]
mod block_restart_tests;
#[cfg(test)]
mod flexible_tests;
#[cfg(test)]
mod admission_tests;
mod pcg_session;
mod session;

#[doc(inline)]
pub use pcg_session::{DevicePcgAssumption, DevicePcgSession, DevicePcgStatus, validate_backend_pcg};

#[doc(inline)]
pub use session::{DeviceGmresSession, DeviceInitialGuess};

use num_traits::Float;
use sonium_compute::pool::Buffer;
use sonium_compute::{
    Backend, BackendContextDisposition, BackendOutputDisposition, BlockGmresConfig,
    BlockGmresError, BlockGmresStatus, BlockGmresWorkspace, DeviceBufferLike, DeviceOperator,
    DevicePreconditioner, DeviceScalar, GmresBackendConfig, GmresBackendError, GmresBackendStatus,
    GmresWorkspace, MemoryPool, TransferStatsSnapshot,
    fgmres_backend_preconditioned_pooled_with_workspace,
    gmres_backend_block_preconditioned_with_workspace,
    gmres_backend_preconditioned_pooled_with_workspace,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Public backend methods owned by the iterative crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKrylovMethod {
    Gmres,
    Fgmres,
    BlockGmres,
    BlockFgmres,
}

/// Scalar families represented in the machine-readable capability matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKrylovScalar {
    F32,
    F64,
    Complex32,
    Complex64,
}

/// One method/scalar capability cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendKrylovCapability {
    pub backend: &'static str,
    pub method: BackendKrylovMethod,
    pub scalar: BackendKrylovScalar,
    /// Operator family represented by this capability cell.  The current
    /// backend-resident entry points intentionally advertise CSR SpMV/SpMM
    /// only; arbitrary host callbacks are not silently treated as native.
    pub operator: &'static str,
    pub operator_state: sonium_compute::CapabilityState,
    /// Preconditioner family represented by this capability cell.  Identity
    /// is always native because it performs no backend operation.  Other
    /// preconditioners must publish their own capability before admission.
    pub preconditioner: &'static str,
    pub preconditioner_state: sonium_compute::CapabilityState,
    pub scalar_state: sonium_compute::CapabilityState,
    pub blas1: sonium_compute::CapabilityState,
    pub axpy_many: sonium_compute::CapabilityState,
    pub multi_dot: sonium_compute::CapabilityState,
    pub strided_projection: sonium_compute::CapabilityState,
    pub reduction: sonium_compute::CapabilityState,
    pub csr_spmm: sonium_compute::CapabilityState,
    /// True only when every primitive needed by this method is native or
    /// vendor-native.  A custom `DeviceOperator` must still implement its
    /// checked apply methods without host staging.
    pub native_execution: bool,
}

/// A cloned cancellation flag that can be shared with a caller or callback.
#[derive(Debug, Clone, Default)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// Configuration shared by scalar and block backend-resident solves.
#[derive(Debug, Clone)]
pub struct BackendKrylovConfig<R> {
    pub max_iterations: usize,
    pub restart: usize,
    pub tolerance: R,
    /// Require all Krylov primitives to be native/vendor-native. Setting this
    /// to false is the explicit opt-in for a host-staged compatibility path.
    pub require_native: bool,
    /// The backend solver always performs an exact final residual check. This
    /// interval is retained for API symmetry with host GMRES and documents
    /// the requested progress-check cadence.
    pub exact_residual_check_interval: usize,
}

impl Default for BackendKrylovConfig<f64> {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            restart: 30,
            tolerance: 1.0e-8,
            require_native: true,
            exact_residual_check_interval: 0,
        }
    }
}

impl Default for BackendKrylovConfig<f32> {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            restart: 30,
            tolerance: 1.0e-5,
            require_native: true,
            exact_residual_check_interval: 0,
        }
    }
}

impl<R: Float> BackendKrylovConfig<R> {
    fn validate(&self) -> Result<(), BackendKrylovError> {
        if self.max_iterations == 0 {
            return Err(BackendKrylovError::InvalidArgument {
                method: BackendKrylovMethod::Gmres,
                reason: "max_iterations must be positive".into(),
                diagnostic: BackendKrylovError::diagnostic_for(
                    BackendKrylovMethod::Gmres,
                    "backend",
                    0,
                    BackendOutputDisposition::Preserved,
                    BackendContextDisposition::Reusable,
                ),
            });
        }
        if self.restart == 0 {
            return Err(BackendKrylovError::InvalidArgument {
                method: BackendKrylovMethod::Gmres,
                reason: "restart must be positive".into(),
                diagnostic: BackendKrylovError::diagnostic_for(
                    BackendKrylovMethod::Gmres,
                    "backend",
                    0,
                    BackendOutputDisposition::Preserved,
                    BackendContextDisposition::Reusable,
                ),
            });
        }
        if !self.tolerance.is_finite() || self.tolerance <= R::zero() {
            return Err(BackendKrylovError::InvalidArgument {
                method: BackendKrylovMethod::Gmres,
                reason: "tolerance must be finite and positive".into(),
                diagnostic: BackendKrylovError::diagnostic_for(
                    BackendKrylovMethod::Gmres,
                    "backend",
                    0,
                    BackendOutputDisposition::Preserved,
                    BackendContextDisposition::Reusable,
                ),
            });
        }
        Ok(())
    }
}

/// Stopping reason preserved by every successful backend solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendStoppingReason {
    Converged,
    MaxIterations,
    Cancelled,
    /// A checked backend operation failed before a valid successful status
    /// could be published.  It is distinct from max-iteration termination.
    Failed,
}

/// Complete diagnostic tuple attached to every checked backend failure.
///
/// Iteration metrics are optional because allocation/admission failures occur
/// before a projected or true residual exists.  The fields remain present in
/// serialized QA records as `null`, which is materially different from
/// claiming a zero residual.  `partial_iterate` and
/// `context_disposition` are always populated from the typed backend error.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendKrylovFailureDiagnostic {
    pub method: BackendKrylovMethod,
    pub actual_backend: &'static str,
    pub iterations: usize,
    pub stopping_reason: BackendStoppingReason,
    pub projected_residual: Option<f64>,
    pub true_residual: Option<f64>,
    pub backward_error: Option<f64>,
    pub partial_iterate: BackendOutputDisposition,
    pub context_disposition: BackendContextDisposition,
}

/// Status returned after the final exact residual gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendKrylovStatus<R> {
    pub method: BackendKrylovMethod,
    pub backend: &'static str,
    pub iterations: usize,
    pub restarts: usize,
    pub projected_residual: R,
    pub true_residual: R,
    pub backward_error: R,
    pub converged: bool,
    pub stopping_reason: BackendStoppingReason,
    pub partial_iterate: BackendOutputDisposition,
    pub transfers: TransferStatsSnapshot,
    pub workspace_device_buffers: usize,
    pub workspace_device_capacity: usize,
}

/// Errors at the backend-resident iterative boundary.
#[derive(Debug, thiserror::Error)]
pub enum BackendKrylovError {
    #[error("{method:?} backend capability admission failed on {backend} for {scalar}: {reason}")]
    Capability {
        method: BackendKrylovMethod,
        backend: &'static str,
        scalar: &'static str,
        reason: String,
        diagnostic: BackendKrylovFailureDiagnostic,
    },
    #[error("{method:?} backend argument is invalid: {reason}")]
    InvalidArgument {
        method: BackendKrylovMethod,
        reason: String,
        diagnostic: BackendKrylovFailureDiagnostic,
    },
    #[error("{method:?} backend solve was cancelled on {backend} after {iterations} iterations")]
    Cancelled {
        method: BackendKrylovMethod,
        backend: &'static str,
        iterations: usize,
        diagnostic: BackendKrylovFailureDiagnostic,
    },
    #[error("{method:?} backend GMRES failed on {backend}: {source}")]
    Gmres {
        method: BackendKrylovMethod,
        backend: &'static str,
        source: GmresBackendError,
        diagnostic: BackendKrylovFailureDiagnostic,
    },
    #[error("{method:?} backend block GMRES failed on {backend}: {source}")]
    Block {
        method: BackendKrylovMethod,
        backend: &'static str,
        source: BlockGmresError,
        diagnostic: BackendKrylovFailureDiagnostic,
    },
}

impl BackendKrylovError {
    fn diagnostic_for(
        method: BackendKrylovMethod,
        actual_backend: &'static str,
        iterations: usize,
        partial_iterate: BackendOutputDisposition,
        context_disposition: BackendContextDisposition,
    ) -> BackendKrylovFailureDiagnostic {
        BackendKrylovFailureDiagnostic {
            method,
            actual_backend,
            iterations,
            stopping_reason: BackendStoppingReason::Failed,
            projected_residual: None,
            true_residual: None,
            backward_error: None,
            partial_iterate,
            context_disposition,
        }
    }

    /// Return the complete failure tuple retained by this checked error.
    #[must_use]
    pub fn diagnostic(&self) -> &BackendKrylovFailureDiagnostic {
        match self {
            Self::Capability { diagnostic, .. }
            | Self::InvalidArgument { diagnostic, .. }
            | Self::Cancelled { diagnostic, .. }
            | Self::Gmres { diagnostic, .. }
            | Self::Block { diagnostic, .. } => diagnostic,
        }
    }

    #[must_use]
    pub fn output_disposition(&self) -> BackendOutputDisposition {
        match self {
            Self::Capability { .. } | Self::InvalidArgument { .. } => {
                BackendOutputDisposition::Preserved
            }
            Self::Cancelled { iterations, .. } => {
                if *iterations == 0 {
                    BackendOutputDisposition::Preserved
                } else {
                    BackendOutputDisposition::Unusable
                }
            }
            Self::Gmres { source, .. } => match source {
                GmresBackendError::Backend { source } => source.output_disposition(),
                GmresBackendError::Allocation { .. }
                | GmresBackendError::Workspace { .. }
                | GmresBackendError::ZeroRhs
                | GmresBackendError::HappyBreakdown { .. } => BackendOutputDisposition::Preserved,
            },
            // The block substrate only publishes `out` after its complete
            // candidate is formed. A backend string error is conservatively
            // marked unusable because the legacy block error cannot retain
            // the underlying typed disposition yet.
            Self::Block { source, .. } => match source {
                BlockGmresError::Backend(_) => BackendOutputDisposition::Unusable,
                BlockGmresError::InvalidArgument(_)
                | BlockGmresError::Allocation { .. }
                | BlockGmresError::RankDeficient { .. }
                | BlockGmresError::SingularLeastSquares { .. }
                | BlockGmresError::Breakdown { .. } => BackendOutputDisposition::Preserved,
            },
        }
    }

    #[must_use]
    pub fn context_disposition(&self) -> BackendContextDisposition {
        match self {
            Self::Gmres {
                source: GmresBackendError::Backend { source },
                ..
            } => source.context_disposition(),
            Self::Block {
                source: BlockGmresError::Backend(_),
                ..
            } => BackendContextDisposition::SynchronizeBeforeReuse,
            _ => BackendContextDisposition::Reusable,
        }
    }
}

// The legacy constructor preserves its backend allocation policy. New retained
// callers use `with_pool` so restart and scalar-fallback storage are accounted.
enum BlockRestart<B: Backend, T: DeviceScalar> {
    Pooled(Buffer<T, B>),
    Legacy(B::Buffer<T>),
}

impl<B: Backend, T: DeviceScalar> BlockRestart<B, T> {
    fn raw(&self) -> &B::Buffer<T> {
        match self {
            Self::Pooled(buffer) => buffer.raw(),
            Self::Legacy(buffer) => buffer,
        }
    }
    fn raw_mut(&mut self) -> &mut B::Buffer<T> {
        match self {
            Self::Pooled(buffer) => buffer.raw_mut(),
            Self::Legacy(buffer) => buffer,
        }
    }
}

/// Reusable scalar and block device workspaces.
pub struct BackendKrylovWorkspace<B: Backend, T: DeviceScalar> {
    dimension: usize,
    scalar: GmresWorkspace<B, T>,
    scalar_restart: Option<Buffer<T, B>>,
    block: BlockGmresWorkspace<B, T>,
    block_restart: Option<BlockRestart<B, T>>,
    block_pool: Option<MemoryPool<B>>,
}

impl<B: Backend, T: DeviceScalar> BackendKrylovWorkspace<B, T> {
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            scalar: GmresWorkspace::new(dimension),
            scalar_restart: None,
            block: BlockGmresWorkspace::new(dimension),
            block_restart: None,
            block_pool: None,
        }
    }

    /// Create reusable block scratch with pooled restart and scalar-fallback storage.
    ///
    /// Use `pool.backend()` for every solve with this workspace. The compute
    /// block substrate still owns its existing backend-allocated basis buffers;
    /// this constructor does not imply complete block-workspace quota coverage.
    /// Active-width buffers are cached across deflation; memory grows with the
    /// visited widths until `clear`. Cached restart values are never used as an
    /// implicit initial guess.
    #[must_use]
    pub fn with_pool(dimension: usize, pool: &MemoryPool<B>) -> Self {
        let mut workspace = Self::new(dimension);
        workspace.block = BlockGmresWorkspace::with_retained_shapes(dimension, pool);
        workspace.block_pool = Some(pool.clone());
        workspace
    }

    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn clear(&mut self) -> Result<(), BackendKrylovError> {
        self.scalar
            .clear()
            .map_err(|source| BackendKrylovError::Gmres {
                method: BackendKrylovMethod::Gmres,
                backend: "backend",
                diagnostic: BackendKrylovError::diagnostic_for(
                    BackendKrylovMethod::Gmres,
                    "backend",
                    0,
                    BackendOutputDisposition::Preserved,
                    BackendContextDisposition::Reusable,
                ),
                source,
            })?;
        self.scalar_restart = None;
        self.block.clear();
        self.block_restart = None;
        Ok(())
    }

    fn scalar_metrics(&self) -> (usize, usize) {
        (
            self.scalar.cached_buffer_count().unwrap_or(0)
                + usize::from(self.scalar_restart.is_some()),
            self.scalar.cached_bytes().unwrap_or(0)
                + self.scalar_restart.as_ref().map_or(0, Buffer::byte_size),
        )
    }

    fn block_metrics(&self) -> (usize, usize) {
        (
            self.block.device_buffer_count() + usize::from(self.block_restart.is_some()),
            self.block.device_capacity()
                + self
                    .block_restart
                    .as_ref()
                    .map_or(0, |buffer| buffer.raw().len()),
        )
    }
}

/// Return all method/scalar cells for a backend.
#[must_use]
pub fn backend_krylov_capability_matrix<B: Backend>(backend: &B) -> Vec<BackendKrylovCapability> {
    let mut cells = Vec::with_capacity(16);
    for method in [
        BackendKrylovMethod::Gmres,
        BackendKrylovMethod::Fgmres,
        BackendKrylovMethod::BlockGmres,
        BackendKrylovMethod::BlockFgmres,
    ] {
        cells.push(capability_for::<B, f32>(
            backend,
            method,
            BackendKrylovScalar::F32,
        ));
        cells.push(capability_for::<B, f64>(
            backend,
            method,
            BackendKrylovScalar::F64,
        ));
        cells.push(capability_for::<B, num_complex::Complex32>(
            backend,
            method,
            BackendKrylovScalar::Complex32,
        ));
        cells.push(capability_for::<B, num_complex::Complex64>(
            backend,
            method,
            BackendKrylovScalar::Complex64,
        ));
    }
    cells
}

fn capability_for<B: Backend, T: DeviceScalar>(
    backend: &B,
    method: BackendKrylovMethod,
    scalar: BackendKrylovScalar,
) -> BackendKrylovCapability {
    let caps = backend.capabilities();
    capability_from_report::<T>(&caps, method, scalar)
}

fn capability_from_report<T: DeviceScalar>(
    caps: &sonium_compute::BackendCapabilities,
    method: BackendKrylovMethod,
    scalar: BackendKrylovScalar,
) -> BackendKrylovCapability {
    let scalar_state = caps.blas1.state_for::<T>();
    let block = matches!(
        method,
        BackendKrylovMethod::BlockGmres | BackendKrylovMethod::BlockFgmres
    );
    let operator_state = if block {
        caps.csr_spmm.state_for::<T>()
    } else {
        caps.csr_spmv.state_for::<T>()
    };
    // This matrix describes the identity preconditioner cell.  A concrete
    // device preconditioner remains responsible for its own checked
    // capability admission and failure provenance.
    let preconditioner_state = sonium_compute::CapabilityState::Native;
    let axpy_many = caps.axpy_many.state_for::<T>();
    let multi_dot = caps.multi_dot.state_for::<T>();
    let strided_projection = if block {
        least_native_state(
            caps.multi_dot_strided.state_for::<T>(),
            caps.basis_axpy_strided.state_for::<T>(),
        )
    } else {
        sonium_compute::CapabilityState::Native
    };
    let reduction = caps.reduction.fast.state_for::<T>();
    let csr_spmm = caps.csr_spmm.state_for::<T>();
    let native_execution = [
        scalar_state,
        operator_state,
        preconditioner_state,
        axpy_many,
        multi_dot,
        strided_projection,
        reduction,
        if block {
            csr_spmm
        } else {
            sonium_compute::CapabilityState::Native
        },
    ]
    .into_iter()
    .all(sonium_compute::CapabilityState::is_native);
    BackendKrylovCapability {
        backend: caps.backend_name,
        method,
        scalar,
        operator: if block { "csr_spmm" } else { "csr_spmv" },
        operator_state,
        preconditioner: "identity",
        preconditioner_state,
        scalar_state,
        blas1: scalar_state,
        axpy_many,
        multi_dot,
        strided_projection,
        reduction,
        csr_spmm,
        native_execution,
    }
}

fn least_native_state(
    left: sonium_compute::CapabilityState,
    right: sonium_compute::CapabilityState,
) -> sonium_compute::CapabilityState {
    use sonium_compute::CapabilityState;
    if left == right {
        return left;
    }
    if left == CapabilityState::Unsupported || right == CapabilityState::Unsupported {
        return CapabilityState::Unsupported;
    }
    if left == CapabilityState::HostStagedFallback || right == CapabilityState::HostStagedFallback {
        return CapabilityState::HostStagedFallback;
    }
    if left == CapabilityState::HostSharedFallback || right == CapabilityState::HostSharedFallback {
        return CapabilityState::HostSharedFallback;
    }
    if left == CapabilityState::VendorNative || right == CapabilityState::VendorNative {
        return CapabilityState::VendorNative;
    }
    CapabilityState::Native
}

/// Validate Krylov primitive admission before allocating retained solve state.
///
/// Reads runtime capabilities without device work. This checks the declared
/// CSR/identity cell; arbitrary operator and preconditioner actions remain
/// responsible for their own dependencies and residency contracts.
///
/// # Errors
/// Rejects unsupported primitives, or host fallback when native execution is required.
pub fn validate_backend_krylov<B: Backend, T: DeviceScalar>(
    backend: &B,
    method: BackendKrylovMethod,
    require_native: bool,
) -> Result<(), BackendKrylovError> {
    let scalar = scalar_kind::<T>();
    let cell = capability_for::<B, T>(backend, method, scalar);
    validate_capability_cell::<T>(&cell, require_native)
}

fn validate_capability_cell<T: DeviceScalar>(
    cell: &BackendKrylovCapability,
    require_native: bool,
) -> Result<(), BackendKrylovError> {
    let method = cell.method;
    let available = [
        cell.scalar_state,
        cell.operator_state,
        cell.preconditioner_state,
        cell.axpy_many,
        cell.multi_dot,
        cell.reduction,
    ]
    .into_iter()
    .all(sonium_compute::CapabilityState::is_available)
        && (!matches!(
            method,
            BackendKrylovMethod::BlockGmres | BackendKrylovMethod::BlockFgmres
        ) || [cell.strided_projection, cell.csr_spmm]
            .into_iter()
            .all(sonium_compute::CapabilityState::is_available));
    if !available {
        return Err(BackendKrylovError::Capability {
            method,
            backend: cell.backend,
            scalar: std::any::type_name::<T>(),
            reason: "one or more required backend primitives are unsupported".into(),
            diagnostic: BackendKrylovError::diagnostic_for(
                method,
                cell.backend,
                0,
                BackendOutputDisposition::Preserved,
                BackendContextDisposition::Reusable,
            ),
        });
    }
    ensure_native_execution(
        method,
        cell.backend,
        std::any::type_name::<T>(),
        cell.native_execution,
        require_native,
    )
}

/// Refuse a non-native execution cell when the caller required native
/// execution.
///
/// Split from [`validate_backend_krylov`] so refusal is directly unit-testable:
/// runtime module availability may change a native cell into a host fallback.
fn ensure_native_execution(
    method: BackendKrylovMethod,
    backend: &'static str,
    scalar: &'static str,
    native_execution: bool,
    require_native: bool,
) -> Result<(), BackendKrylovError> {
    if require_native && !native_execution {
        return Err(BackendKrylovError::Capability {
            method,
            backend,
            scalar,
            reason: "the requested native execution cell would use host fallback".into(),
            diagnostic: BackendKrylovError::diagnostic_for(
                method,
                backend,
                0,
                BackendOutputDisposition::Preserved,
                BackendContextDisposition::Reusable,
            ),
        });
    }
    Ok(())
}

fn scalar_kind<T: DeviceScalar>() -> BackendKrylovScalar {
    use std::any::TypeId;
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        BackendKrylovScalar::F32
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        BackendKrylovScalar::F64
    } else if TypeId::of::<T>() == TypeId::of::<num_complex::Complex32>() {
        BackendKrylovScalar::Complex32
    } else {
        BackendKrylovScalar::Complex64
    }
}

fn transfer_delta(
    before: TransferStatsSnapshot,
    after: TransferStatsSnapshot,
) -> TransferStatsSnapshot {
    TransferStatsSnapshot {
        h2d_operations: after.h2d_operations.saturating_sub(before.h2d_operations),
        h2d_bytes: after.h2d_bytes.saturating_sub(before.h2d_bytes),
        d2h_operations: after.d2h_operations.saturating_sub(before.d2h_operations),
        d2h_bytes: after.d2h_bytes.saturating_sub(before.d2h_bytes),
        d2d_operations: after.d2d_operations.saturating_sub(before.d2d_operations),
        d2d_bytes: after.d2d_bytes.saturating_sub(before.d2d_bytes),
    }
}

fn map_gmres_error<B: Backend>(
    backend: &B,
    method: BackendKrylovMethod,
    source: GmresBackendError,
    iterations: usize,
) -> BackendKrylovError {
    let partial_iterate = match &source {
        GmresBackendError::Backend { source } => source.output_disposition(),
        GmresBackendError::Allocation { .. }
        | GmresBackendError::Workspace { .. }
        | GmresBackendError::ZeroRhs
        | GmresBackendError::HappyBreakdown { .. } => BackendOutputDisposition::Preserved,
    };
    let context_disposition = match &source {
        GmresBackendError::Backend { source } => source.context_disposition(),
        GmresBackendError::Allocation { .. }
        | GmresBackendError::Workspace { .. }
        | GmresBackendError::ZeroRhs
        | GmresBackendError::HappyBreakdown { .. } => BackendContextDisposition::Reusable,
    };
    let diagnostic = BackendKrylovError::diagnostic_for(
        method,
        backend.capabilities().backend_name,
        iterations,
        partial_iterate,
        context_disposition,
    );
    BackendKrylovError::Gmres {
        method,
        backend: backend.capabilities().backend_name,
        source,
        diagnostic,
    }
}

fn map_block_error<B: Backend>(
    backend: &B,
    method: BackendKrylovMethod,
    source: BlockGmresError,
    iterations: usize,
) -> BackendKrylovError {
    let (partial_iterate, context_disposition) = match &source {
        BlockGmresError::Backend(_) => (
            BackendOutputDisposition::Unusable,
            BackendContextDisposition::SynchronizeBeforeReuse,
        ),
        BlockGmresError::InvalidArgument(_)
        | BlockGmresError::Allocation { .. }
        | BlockGmresError::RankDeficient { .. }
        | BlockGmresError::SingularLeastSquares { .. }
        | BlockGmresError::Breakdown { .. } => (
            BackendOutputDisposition::Preserved,
            BackendContextDisposition::Reusable,
        ),
    };
    let diagnostic = BackendKrylovError::diagnostic_for(
        method,
        backend.capabilities().backend_name,
        iterations,
        partial_iterate,
        context_disposition,
    );
    BackendKrylovError::Block {
        method,
        backend: backend.capabilities().backend_name,
        source,
        diagnostic,
    }
}

/// Backend-resident restarted GMRES with a fixed right preconditioner.
#[allow(clippy::too_many_arguments)]
pub fn try_gmres_backend<B, A, P, T>(
    backend: &B,
    pool: &MemoryPool<B>,
    workspace: &mut BackendKrylovWorkspace<B, T>,
    a: &A,
    precond: &P,
    b: &B::Buffer<T>,
    x0: Option<&B::Buffer<T>>,
    out: &mut B::Buffer<T>,
    config: &BackendKrylovConfig<T::Real>,
    cancellation: Option<&CancellationToken>,
) -> Result<BackendKrylovStatus<T::Real>, BackendKrylovError>
where
    B: Backend,
    A: DeviceOperator<B, T> + ?Sized,
    P: DevicePreconditioner<B, T> + ?Sized,
    T: DeviceScalar,
{
    try_scalar_backend(
        backend,
        pool,
        workspace,
        BackendKrylovMethod::Gmres,
        a,
        precond,
        b,
        x0,
        out,
        config,
        cancellation,
    )
}

/// Backend-resident FGMRES entry point.
///
/// Retains each preconditioned Arnoldi vector, allowing the checked device
/// preconditioner to change between applications. Restart cycles reuse pooled
/// storage and certify convergence against the original operator.
#[allow(clippy::too_many_arguments)]
pub fn try_fgmres_backend_preconditioned<B, A, P, T>(
    backend: &B,
    pool: &MemoryPool<B>,
    workspace: &mut BackendKrylovWorkspace<B, T>,
    a: &A,
    precond: &P,
    b: &B::Buffer<T>,
    x0: Option<&B::Buffer<T>>,
    out: &mut B::Buffer<T>,
    config: &BackendKrylovConfig<T::Real>,
    cancellation: Option<&CancellationToken>,
) -> Result<BackendKrylovStatus<T::Real>, BackendKrylovError>
where
    B: Backend,
    A: DeviceOperator<B, T> + ?Sized,
    P: DevicePreconditioner<B, T> + ?Sized,
    T: DeviceScalar,
{
    try_scalar_backend(
        backend,
        pool,
        workspace,
        BackendKrylovMethod::Fgmres,
        a,
        precond,
        b,
        x0,
        out,
        config,
        cancellation,
    )
}

#[allow(clippy::too_many_arguments)]
fn try_scalar_backend<B, A, P, T>(
    backend: &B,
    pool: &MemoryPool<B>,
    workspace: &mut BackendKrylovWorkspace<B, T>,
    method: BackendKrylovMethod,
    a: &A,
    precond: &P,
    b: &B::Buffer<T>,
    x0: Option<&B::Buffer<T>>,
    out: &mut B::Buffer<T>,
    config: &BackendKrylovConfig<T::Real>,
    cancellation: Option<&CancellationToken>,
) -> Result<BackendKrylovStatus<T::Real>, BackendKrylovError>
where
    B: Backend,
    A: DeviceOperator<B, T> + ?Sized,
    P: DevicePreconditioner<B, T> + ?Sized,
    T: DeviceScalar,
{
    config.validate().map_err(|error| match error {
        BackendKrylovError::InvalidArgument { reason, .. } => BackendKrylovError::InvalidArgument {
            method,
            reason,
            diagnostic: BackendKrylovError::diagnostic_for(
                method,
                backend.capabilities().backend_name,
                0,
                BackendOutputDisposition::Preserved,
                BackendContextDisposition::Reusable,
            ),
        },
        other => other,
    })?;
    validate_backend_krylov::<B, T>(backend, method, config.require_native)?;
    if workspace.dimension() != a.n() {
        return Err(BackendKrylovError::InvalidArgument {
            method,
            reason: format!(
                "workspace dimension {} != operator {}",
                workspace.dimension(),
                a.n()
            ),
            diagnostic: BackendKrylovError::diagnostic_for(
                method,
                backend.capabilities().backend_name,
                0,
                BackendOutputDisposition::Preserved,
                BackendContextDisposition::Reusable,
            ),
        });
    }
    let backend_name = backend.capabilities().backend_name;
    if cancellation.is_some_and(CancellationToken::is_cancelled) {
        return Err(BackendKrylovError::Cancelled {
            method,
            backend: backend_name,
            iterations: 0,
            diagnostic: BackendKrylovFailureDiagnostic {
                method,
                actual_backend: backend_name,
                iterations: 0,
                stopping_reason: BackendStoppingReason::Cancelled,
                projected_residual: None,
                true_residual: None,
                backward_error: None,
                partial_iterate: BackendOutputDisposition::Preserved,
                context_disposition: BackendContextDisposition::Reusable,
            },
        });
    }
    let before = backend.transfer_stats();
    let mut total_iterations = 0usize;
    let mut restarts = 0usize;
    let mut last = GmresBackendStatus {
        iterations: 0,
        projected_residual: T::Real::infinity(),
        final_residual: T::Real::infinity(),
        backward_error: T::Real::infinity(),
        converged: false,
    };

    while total_iterations < config.max_iterations {
        if cancellation.is_some_and(CancellationToken::is_cancelled) {
            return Err(BackendKrylovError::Cancelled {
                method,
                backend: backend_name,
                iterations: total_iterations,
                diagnostic: BackendKrylovFailureDiagnostic {
                    method,
                    actual_backend: backend_name,
                    iterations: total_iterations,
                    stopping_reason: BackendStoppingReason::Cancelled,
                    projected_residual: None,
                    true_residual: None,
                    backward_error: None,
                    partial_iterate: if total_iterations == 0 {
                        BackendOutputDisposition::Preserved
                    } else {
                        BackendOutputDisposition::Unusable
                    },
                    context_disposition: BackendContextDisposition::Reusable,
                },
            });
        }
        let cycle_limit = config
            .restart
            .min(config.max_iterations.saturating_sub(total_iterations));
        let cycle_config = GmresBackendConfig {
            max_iterations: cycle_limit,
            tolerance: config.tolerance,
        };
        // Cached restart storage contains a previous solve until this solve
        // completes its first cycle. Never use it as an implicit initial guess.
        let initial = if total_iterations == 0 {
            x0
        } else {
            workspace.scalar_restart.as_ref().map(Buffer::raw)
        };
        let solve_cycle = if method == BackendKrylovMethod::Fgmres {
            fgmres_backend_preconditioned_pooled_with_workspace
        } else {
            gmres_backend_preconditioned_pooled_with_workspace
        };
        last = solve_cycle(
            pool,
            &mut workspace.scalar,
            a,
            precond,
            b,
            initial,
            out,
            &cycle_config,
        )
        .map_err(|source| map_gmres_error(backend, method, source, total_iterations))?;
        total_iterations = total_iterations.saturating_add(last.iterations);
        if last.converged {
            return Ok(BackendKrylovStatus {
                method,
                backend: backend_name,
                iterations: total_iterations,
                restarts,
                projected_residual: last.projected_residual,
                true_residual: last.final_residual,
                backward_error: last.backward_error,
                converged: true,
                stopping_reason: BackendStoppingReason::Converged,
                partial_iterate: BackendOutputDisposition::Preserved,
                transfers: transfer_delta(before, backend.transfer_stats()),
                workspace_device_buffers: workspace.scalar_metrics().0,
                workspace_device_capacity: workspace.scalar_metrics().1,
            });
        }
        if last.iterations == 0 || total_iterations >= config.max_iterations {
            break;
        }
        if workspace.scalar_restart.is_none() {
            workspace.scalar_restart = Some(pool.try_alloc::<T>(a.n()).map_err(|error| {
                map_gmres_error(
                    backend,
                    method,
                    GmresBackendError::Allocation {
                        label: "restart_initial",
                        reason: error.to_string(),
                    },
                    total_iterations,
                )
            })?);
        }
        let restart = workspace.scalar_restart.as_mut().ok_or_else(|| {
            BackendKrylovError::InvalidArgument {
                method,
                reason: "restart buffer was not retained after allocation".into(),
                diagnostic: BackendKrylovError::diagnostic_for(
                    method,
                    backend_name,
                    total_iterations,
                    BackendOutputDisposition::Preserved,
                    BackendContextDisposition::Reusable,
                ),
            }
        })?;
        backend
            .try_copy_d2d(out, restart.raw_mut())
            .map_err(|source| {
                map_gmres_error(
                    backend,
                    method,
                    GmresBackendError::from(source),
                    total_iterations,
                )
            })?;
        restarts = restarts.saturating_add(1);
    }

    Ok(BackendKrylovStatus {
        method,
        backend: backend_name,
        iterations: total_iterations,
        restarts,
        projected_residual: last.projected_residual,
        true_residual: last.final_residual,
        backward_error: last.backward_error,
        converged: false,
        stopping_reason: BackendStoppingReason::MaxIterations,
        partial_iterate: BackendOutputDisposition::Preserved,
        transfers: transfer_delta(before, backend.transfer_stats()),
        workspace_device_buffers: workspace.scalar_metrics().0,
        workspace_device_capacity: workspace.scalar_metrics().1,
    })
}

/// Backend-resident block GMRES. `b` and `out` are `n × rhs` column-major
/// device buffers. The block substrate uses SpMM and backend projection
/// primitives after its initial setup and performs an exact residual check for
/// every active RHS before publishing convergence.
#[allow(clippy::too_many_arguments)]
pub fn try_block_gmres_backend<B, A, P, T>(
    backend: &B,
    workspace: &mut BackendKrylovWorkspace<B, T>,
    a: &A,
    precond: &P,
    b: &B::Buffer<T>,
    x0: Option<&B::Buffer<T>>,
    out: &mut B::Buffer<T>,
    rhs: usize,
    config: &BackendKrylovConfig<T::Real>,
    cancellation: Option<&CancellationToken>,
) -> Result<BackendKrylovStatus<T::Real>, BackendKrylovError>
where
    B: Backend,
    A: DeviceOperator<B, T> + ?Sized,
    P: DevicePreconditioner<B, T> + ?Sized,
    T: DeviceScalar + std::ops::AddAssign,
{
    try_block_backend(
        backend,
        workspace,
        BackendKrylovMethod::BlockGmres,
        a,
        precond,
        b,
        x0,
        out,
        rhs,
        config,
        cancellation,
    )
}

/// Backend-resident block FGMRES specialization with a fixed device
/// preconditioner.
#[allow(clippy::too_many_arguments)]
pub fn try_block_fgmres_backend<B, A, P, T>(
    backend: &B,
    workspace: &mut BackendKrylovWorkspace<B, T>,
    a: &A,
    precond: &P,
    b: &B::Buffer<T>,
    x0: Option<&B::Buffer<T>>,
    out: &mut B::Buffer<T>,
    rhs: usize,
    config: &BackendKrylovConfig<T::Real>,
    cancellation: Option<&CancellationToken>,
) -> Result<BackendKrylovStatus<T::Real>, BackendKrylovError>
where
    B: Backend,
    A: DeviceOperator<B, T> + ?Sized,
    P: DevicePreconditioner<B, T> + ?Sized,
    T: DeviceScalar + std::ops::AddAssign,
{
    try_block_backend(
        backend,
        workspace,
        BackendKrylovMethod::BlockFgmres,
        a,
        precond,
        b,
        x0,
        out,
        rhs,
        config,
        cancellation,
    )
}

#[allow(clippy::too_many_arguments)]
fn try_block_backend<B, A, P, T>(
    backend: &B,
    workspace: &mut BackendKrylovWorkspace<B, T>,
    method: BackendKrylovMethod,
    a: &A,
    precond: &P,
    b: &B::Buffer<T>,
    x0: Option<&B::Buffer<T>>,
    out: &mut B::Buffer<T>,
    rhs: usize,
    config: &BackendKrylovConfig<T::Real>,
    cancellation: Option<&CancellationToken>,
) -> Result<BackendKrylovStatus<T::Real>, BackendKrylovError>
where
    B: Backend,
    A: DeviceOperator<B, T> + ?Sized,
    P: DevicePreconditioner<B, T> + ?Sized,
    T: DeviceScalar + std::ops::AddAssign,
{
    config.validate().map_err(|error| match error {
        BackendKrylovError::InvalidArgument { reason, .. } => BackendKrylovError::InvalidArgument {
            method,
            reason,
            diagnostic: BackendKrylovError::diagnostic_for(
                method,
                backend.capabilities().backend_name,
                0,
                BackendOutputDisposition::Preserved,
                BackendContextDisposition::Reusable,
            ),
        },
        other => other,
    })?;
    validate_backend_krylov::<B, T>(backend, method, config.require_native)?;
    if workspace.dimension() != a.n() {
        return Err(BackendKrylovError::InvalidArgument {
            method,
            reason: format!(
                "workspace dimension {} != operator {}",
                workspace.dimension(),
                a.n()
            ),
            diagnostic: BackendKrylovError::diagnostic_for(
                method,
                backend.capabilities().backend_name,
                0,
                BackendOutputDisposition::Preserved,
                BackendContextDisposition::Reusable,
            ),
        });
    }
    let backend_name = backend.capabilities().backend_name;
    if cancellation.is_some_and(CancellationToken::is_cancelled) {
        return Err(BackendKrylovError::Cancelled {
            method,
            backend: backend_name,
            iterations: 0,
            diagnostic: BackendKrylovFailureDiagnostic {
                method,
                actual_backend: backend_name,
                iterations: 0,
                stopping_reason: BackendStoppingReason::Cancelled,
                projected_residual: None,
                true_residual: None,
                backward_error: None,
                partial_iterate: BackendOutputDisposition::Preserved,
                context_disposition: BackendContextDisposition::Reusable,
            },
        });
    }
    let before = backend.transfer_stats();
    let block_len = a.n().checked_mul(rhs).ok_or_else(|| {
        map_block_error(
            backend,
            method,
            BlockGmresError::InvalidArgument("block extent overflows usize".into()),
            0,
        )
    })?;
    // Only a restart completed during this call may become the next initial guess.
    let mut restart_ready = false;
    let mut total_iterations = 0usize;
    let mut restarts = 0usize;
    let mut last = BlockGmresStatus {
        block_iterations: 0,
        projected_residual: T::Real::infinity(),
        final_residual: T::Real::infinity(),
        backward_error: T::Real::infinity(),
        converged: false,
        rhs: Vec::new(),
    };
    while total_iterations < config.max_iterations {
        if cancellation.is_some_and(CancellationToken::is_cancelled) {
            return Err(BackendKrylovError::Cancelled {
                method,
                backend: backend_name,
                iterations: total_iterations,
                diagnostic: BackendKrylovFailureDiagnostic {
                    method,
                    actual_backend: backend_name,
                    iterations: total_iterations,
                    stopping_reason: BackendStoppingReason::Cancelled,
                    projected_residual: None,
                    true_residual: None,
                    backward_error: None,
                    partial_iterate: if total_iterations == 0 {
                        BackendOutputDisposition::Preserved
                    } else {
                        BackendOutputDisposition::Unusable
                    },
                    context_disposition: BackendContextDisposition::Reusable,
                },
            });
        }
        let cycle_limit = config
            .restart
            .min(config.max_iterations.saturating_sub(total_iterations));
        let initial = if restart_ready {
            workspace.block_restart.as_ref().map(BlockRestart::raw)
        } else {
            x0
        };
        last = gmres_backend_block_preconditioned_with_workspace(
            backend,
            a,
            precond,
            b,
            initial,
            out,
            rhs,
            &BlockGmresConfig {
                max_block_iterations: cycle_limit,
                tolerance: config.tolerance,
            },
            &mut workspace.block,
        )
        .map_err(|source| map_block_error(backend, method, source, total_iterations))?;
        total_iterations = total_iterations.saturating_add(last.block_iterations);
        if last.converged {
            break;
        }
        if last.block_iterations == 0 {
            break;
        }
        if total_iterations >= config.max_iterations {
            break;
        }
        if workspace
            .block_restart
            .as_ref()
            .is_none_or(|buffer| buffer.raw().len() != block_len)
        {
            let allocation = if let Some(pool) = &workspace.block_pool {
                pool.try_alloc::<T>(block_len).map(BlockRestart::Pooled)
            } else {
                // Compatibility path: retain the existing backend allocation,
                // now sized for every RHS rather than only the first column.
                backend.raw_alloc::<T>(block_len).map(BlockRestart::Legacy)
            };
            workspace.block_restart = Some(allocation.map_err(|error| {
                map_block_error(
                    backend,
                    method,
                    BlockGmresError::Allocation {
                        label: "block restart initial",
                        reason: error.to_string(),
                    },
                    total_iterations,
                )
            })?);
        }
        let restart = workspace
            .block_restart
            .as_mut()
            .ok_or_else(|| {
                map_block_error(
                    backend,
                    method,
                    BlockGmresError::InvalidArgument("missing block restart storage".into()),
                    total_iterations,
                )
            })?
            .raw_mut();
        backend.try_copy_d2d(out, restart).map_err(|source| {
            map_block_error(
                backend,
                method,
                BlockGmresError::Backend(source.to_string()),
                total_iterations,
            )
        })?;
        restart_ready = true;
        restarts = restarts.saturating_add(1);
    }
    Ok(BackendKrylovStatus {
        method,
        backend: backend_name,
        iterations: total_iterations,
        restarts,
        projected_residual: last.projected_residual,
        true_residual: last.final_residual,
        backward_error: last.backward_error,
        converged: last.converged,
        stopping_reason: if last.converged {
            BackendStoppingReason::Converged
        } else {
            BackendStoppingReason::MaxIterations
        },
        partial_iterate: BackendOutputDisposition::Preserved,
        transfers: transfer_delta(before, backend.transfer_stats()),
        workspace_device_buffers: workspace.block_metrics().0,
        workspace_device_capacity: workspace.block_metrics().1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_compute::{CpuBackend, DeviceCsr, IdentityDevicePreconditioner, MemoryPoolConfig};

    // Fixture intentionally uses the panicking convenience constructor on a
    // tiny fixed matrix; production paths must use the `try_*` variants.
    #[allow(deprecated)]
    fn fixture() -> (CpuBackend, DeviceCsr<CpuBackend, f64>) {
        let backend = CpuBackend;
        let matrix = DeviceCsr::from_host(
            &backend,
            2,
            2,
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[4.0, 1.0, 1.0, 3.0],
        );
        (backend, matrix)
    }

    #[test]
    fn capability_matrix_has_all_iterative_cells() {
        let backend = CpuBackend;
        let cells = backend_krylov_capability_matrix(&backend);
        assert_eq!(cells.len(), 16);
        assert!(cells.iter().all(|cell| cell.native_execution));
        assert!(cells.iter().all(|cell| {
            cell.operator
                == if matches!(
                    cell.method,
                    BackendKrylovMethod::BlockGmres | BackendKrylovMethod::BlockFgmres
                ) {
                    "csr_spmm"
                } else {
                    "csr_spmv"
                }
                && cell.preconditioner == "identity"
                && cell.preconditioner_state.is_native()
        }));
    }

    #[test]
    fn require_native_refuses_non_native_cell() {
        let error = ensure_native_execution(BackendKrylovMethod::Gmres, "cuda", "f64", false, true)
            .expect_err("fallback cell must be refused under require_native");
        match error {
            BackendKrylovError::Capability {
                method,
                backend,
                reason,
                ..
            } => {
                assert_eq!(method, BackendKrylovMethod::Gmres);
                assert_eq!(backend, "cuda");
                assert!(
                    reason.contains("host fallback"),
                    "refusal must name the fallback, got: {reason}"
                );
            }
            other => panic!("expected a Capability refusal, got: {other}"),
        }
    }

    #[test]
    fn native_cell_passes_with_and_without_require_native() {
        for require_native in [false, true] {
            ensure_native_execution(
                BackendKrylovMethod::BlockGmres,
                "cuda",
                "complex64",
                true,
                require_native,
            )
            .expect("native cell must pass");
        }
    }

    #[test]
    fn fallback_cell_passes_when_native_not_required() {
        ensure_native_execution(BackendKrylovMethod::Gmres, "cuda", "f64", false, false)
            .expect("explicit staged opt-in must pass preflight");
    }

    #[test]
    fn checked_failure_diagnostic_is_complete_for_invalid_input() {
        let (backend, matrix) = fixture();
        let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let b = pool.backend().raw_alloc::<f64>(2).unwrap();
        let mut out = pool.backend().raw_alloc::<f64>(2).unwrap();
        let error = try_gmres_backend(
            pool.backend(),
            &pool,
            &mut BackendKrylovWorkspace::new(2),
            &matrix,
            &IdentityDevicePreconditioner,
            &b,
            None,
            &mut out,
            &BackendKrylovConfig {
                max_iterations: 0,
                ..BackendKrylovConfig::default()
            },
            None,
        )
        .expect_err("zero max iterations must be rejected before dispatch");
        let diagnostic = error.diagnostic();
        assert_eq!(diagnostic.method, BackendKrylovMethod::Gmres);
        assert_eq!(diagnostic.actual_backend, "cpu");
        assert_eq!(diagnostic.iterations, 0);
        assert_eq!(diagnostic.stopping_reason, BackendStoppingReason::Failed);
        assert!(diagnostic.projected_residual.is_none());
        assert!(diagnostic.true_residual.is_none());
        assert!(diagnostic.backward_error.is_none());
        assert_eq!(
            diagnostic.partial_iterate,
            BackendOutputDisposition::Preserved
        );
        assert_eq!(
            diagnostic.context_disposition,
            BackendContextDisposition::Reusable
        );
    }

    #[test]
    fn scalar_backend_contract_reports_exact_residual_and_workspace() {
        let (backend, matrix) = fixture();
        let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let b = {
            let mut value = pool.backend().raw_alloc::<f64>(2).unwrap();
            pool.backend().copy_h2d(&[9.0, 8.0], &mut value);
            value
        };
        let mut out = pool.backend().raw_alloc::<f64>(2).unwrap();
        let mut workspace = BackendKrylovWorkspace::new(2);
        let status = try_gmres_backend(
            pool.backend(),
            &pool,
            &mut workspace,
            &matrix,
            &IdentityDevicePreconditioner,
            &b,
            None,
            &mut out,
            &BackendKrylovConfig {
                max_iterations: 4,
                restart: 2,
                tolerance: 1.0e-10,
                require_native: true,
                exact_residual_check_interval: 0,
            },
            None,
        )
        .unwrap();
        assert!(status.converged);
        assert_eq!(status.stopping_reason, BackendStoppingReason::Converged);
        assert!(status.true_residual <= 1.0e-10);
        assert!(status.workspace_device_buffers > 0);
        assert_eq!(status.backend, "cpu");
    }

    #[test]
    fn cancellation_is_reported_before_iteration_zero_and_preserves_output() {
        let (backend, matrix) = fixture();
        let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let b = pool.backend().raw_alloc::<f64>(2).unwrap();
        let mut out = pool.backend().raw_alloc::<f64>(2).unwrap();
        pool.backend().copy_h2d(&[3.0, 4.0], &mut out);
        let token = CancellationToken::new();
        token.cancel();
        let error = try_gmres_backend(
            pool.backend(),
            &pool,
            &mut BackendKrylovWorkspace::new(2),
            &matrix,
            &IdentityDevicePreconditioner,
            &b,
            None,
            &mut out,
            &BackendKrylovConfig::<f64>::default(),
            Some(&token),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            BackendKrylovError::Cancelled { iterations: 0, .. }
        ));
        assert_eq!(
            error.output_disposition(),
            BackendOutputDisposition::Preserved
        );
        assert_eq!(
            error.diagnostic().stopping_reason,
            BackendStoppingReason::Cancelled
        );
        assert_eq!(error.diagnostic().actual_backend, "cpu");
        assert_eq!(error.diagnostic().iterations, 0);
    }
}
