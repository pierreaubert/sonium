//! The [`Backend`] trait — abstraction over compute targets.
//!
//! A `Backend` owns the device context and exposes the BLAS / sparse
//! primitives needed by Krylov solvers, preconditioners, and matrix
//! assembly. Buffers always come out of a [`crate::MemoryPool`], never
//! directly from the backend, so device memory usage stays inside its
//! configured cap.
//!
//! CPU, CUDA, and Metal backends share one trait surface. Each backend reports
//! a machine-readable capability table so callers and CI can tell native
//! device execution apart from intentional host fallbacks.

use crate::bem::{
    BemFieldElementC64, BemFieldPointC64, BemRegularPairIn, BemRegularPairInF64, BemRegularPairOut,
    BemRegularPairOutF64,
};
#[cfg(feature = "cpu")]
use crate::cpu::blas3::trsm::{TrsmSide, TrsmUplo};
use crate::error::{BackendError, BackendResult, OomError};
use crate::scalar::{BufferElement, DeviceScalar, LowerPrecision};
use crate::segmented::{segmented_sum, validate_segmented_offsets};
use crate::tensor_contract::batched_local_contract;
#[cfg(test)]
use num_complex::Complex32;
use num_complex::{Complex, Complex64};
use num_traits::{Float, Zero};
use std::panic::{AssertUnwindSafe, catch_unwind};
#[cfg(any(feature = "cuda", feature = "metal", test))]
use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicUsize, Ordering};

/// One-shot fault-injection points for checked device-preconditioner paths.
///
/// Production backends ignore these points unless a caller explicitly arms
/// their test/fault-injection hook.  Keeping the point vocabulary in the
/// shared backend contract lets CPU tests exercise the same setup and runtime
/// error boundaries as CUDA without pretending that host buffers are CUDA
/// execution evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PreconditionerFailureInjectionPoint {
    /// Failure while uploading a hierarchy matrix or transfer operator.
    HierarchyUpload,
    /// Failure while uploading an inverse diagonal.
    DiagonalUpload,
    /// Failure while allocating a preconditioner workspace.
    WorkspaceAllocation,
    /// Failure entering a configured smoother.
    Smoother,
    /// Failure entering a coarse direct solve.
    CoarseSolve,
    /// Failure immediately before a checked device action dispatches work.
    Launch,
    /// Failure at the checked synchronization boundary.
    Synchronization,
    /// A generated non-finite value is reported by the action.
    NonFiniteGeneration,
    /// The backend reports that its device context was lost.
    DeviceLoss,
}

impl PreconditionerFailureInjectionPoint {
    pub(crate) const fn operation(self) -> &'static str {
        match self {
            Self::HierarchyUpload => "preconditioner_hierarchy_upload",
            Self::DiagonalUpload => "preconditioner_diagonal_upload",
            Self::WorkspaceAllocation => "preconditioner_workspace_allocation",
            Self::Smoother => "preconditioner_smoother",
            Self::CoarseSolve => "preconditioner_coarse_solve",
            Self::Launch => "preconditioner_launch",
            Self::Synchronization => "preconditioner_synchronization",
            Self::NonFiniteGeneration => "preconditioner_nonfinite_generation",
            Self::DeviceLoss => "preconditioner_device_loss",
        }
    }
}

pub(crate) fn injected_preconditioner_error(
    point: PreconditionerFailureInjectionPoint,
    backend: &'static str,
) -> BackendError {
    let operation = point.operation();
    match point {
        PreconditionerFailureInjectionPoint::HierarchyUpload
        | PreconditionerFailureInjectionPoint::DiagonalUpload => BackendError::Transfer {
            backend,
            operation,
            reason: "injected preconditioner upload failure".into(),
        },
        PreconditionerFailureInjectionPoint::WorkspaceAllocation => BackendError::Allocation {
            reason: "injected preconditioner workspace allocation failure".into(),
        },
        PreconditionerFailureInjectionPoint::Smoother
        | PreconditionerFailureInjectionPoint::Launch => BackendError::Launch {
            backend,
            operation,
            reason: "injected preconditioner launch failure".into(),
        },
        PreconditionerFailureInjectionPoint::CoarseSolve => BackendError::OperationFailed {
            operation,
            reason: "injected preconditioner coarse solve failure".into(),
        },
        PreconditionerFailureInjectionPoint::Synchronization => BackendError::Synchronization {
            backend,
            operation,
            reason: "injected preconditioner synchronization failure".into(),
        },
        PreconditionerFailureInjectionPoint::NonFiniteGeneration => BackendError::InvalidArgument {
            operation,
            reason: "injected non-finite preconditioner generation".into(),
        },
        PreconditionerFailureInjectionPoint::DeviceLoss => BackendError::DeviceLost {
            backend,
            operation,
            reason: "injected device loss".into(),
        },
    }
}

/// One-shot fault-injection points for CUDA checked resource and completion
/// boundaries. These hooks are intended for driver-shim and acceptance tests;
/// they are inert until explicitly armed by a caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CudaFailureInjectionPoint {
    /// Fail before reserving the CUDA allocation ledger.
    Allocation,
    /// Fail before creating a native CUDA/cuSPARSE descriptor.
    DescriptorCreation,
    /// Fail before encoding a checked custom-kernel launch.
    Launch,
    /// Fail before a host-to-device transfer mutates the destination.
    TransferH2D,
    /// Fail before a device-to-host transfer mutates the host destination.
    TransferD2H,
    /// Fail before a device-to-device transfer mutates the destination.
    TransferD2D,
    /// Fail before stream synchronization.
    Synchronization,
    /// Fail before recording a completion event.
    EventRecord,
    /// Fail before waiting for a completion event.
    EventWait,
    /// Report terminal device loss at the next checked preflight.
    DeviceLoss,
    /// Retain a typed teardown failure while resources are being destroyed.
    Teardown,
}

/// Real-valued transform implemented by a backend-native plan.
///
/// The variants cover the public DCT/DST forward, inverse, and
/// Euclidean-adjoint operations. Keeping the transform vocabulary in the
/// backend crate avoids making CUDA depend on the higher-level FFT crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RealTransformKind {
    Dct2,
    Dct3,
    Dct2Adjoint,
    Dct3Adjoint,
    Dst2,
    Dst3,
    Dst2Adjoint,
    Dst3Adjoint,
}

impl CudaFailureInjectionPoint {
    #[cfg(feature = "cuda")]
    pub(crate) const fn operation(self) -> &'static str {
        match self {
            Self::Allocation => "cuda_allocation",
            Self::DescriptorCreation => "cuda_descriptor_creation",
            Self::Launch => "cuda_launch",
            Self::TransferH2D => "cuda_copy_h2d",
            Self::TransferD2H => "cuda_copy_d2h",
            Self::TransferD2D => "cuda_copy_d2d",
            Self::Synchronization => "cuda_synchronize",
            Self::EventRecord => "cuda_event_record",
            Self::EventWait => "cuda_event_wait",
            Self::DeviceLoss => "cuda_device_loss",
            Self::Teardown => "cuda_teardown",
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn injected_cuda_failure_error(point: CudaFailureInjectionPoint) -> BackendError {
    let operation = point.operation();
    match point {
        CudaFailureInjectionPoint::Allocation => BackendError::Allocation {
            reason: "injected CUDA allocation failure".into(),
        },
        CudaFailureInjectionPoint::DescriptorCreation | CudaFailureInjectionPoint::Teardown => {
            BackendError::OperationFailed {
                operation,
                reason: "injected CUDA resource failure".into(),
            }
        }
        CudaFailureInjectionPoint::Launch | CudaFailureInjectionPoint::EventRecord => {
            BackendError::Launch {
                backend: "cuda",
                operation,
                reason: "injected CUDA launch/event failure".into(),
            }
        }
        CudaFailureInjectionPoint::TransferH2D
        | CudaFailureInjectionPoint::TransferD2H
        | CudaFailureInjectionPoint::TransferD2D => BackendError::Transfer {
            backend: "cuda",
            operation,
            reason: "injected CUDA transfer failure".into(),
        },
        CudaFailureInjectionPoint::Synchronization | CudaFailureInjectionPoint::EventWait => {
            BackendError::Synchronization {
                backend: "cuda",
                operation,
                reason: "injected CUDA synchronization failure".into(),
            }
        }
        CudaFailureInjectionPoint::DeviceLoss => BackendError::DeviceLost {
            backend: "cuda",
            operation,
            reason: "injected CUDA device loss".into(),
        },
    }
}

/// Snapshot of host/device transfer activity observed by a backend.
///
/// Counts are updated only after a checked transfer has accepted the request
/// (or, for asynchronous device-to-device copies, after the command is
/// encoded). They are intended for bounded QA and performance evidence rather
/// than as a replacement for driver profilers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TransferStatsSnapshot {
    pub h2d_operations: usize,
    pub h2d_bytes: usize,
    pub d2h_operations: usize,
    pub d2h_bytes: usize,
    pub d2d_operations: usize,
    pub d2d_bytes: usize,
}

impl TransferStatsSnapshot {
    /// Return event and byte counts observed after `earlier`.
    pub fn saturating_sub(self, earlier: Self) -> Self {
        Self {
            h2d_operations: self.h2d_operations.saturating_sub(earlier.h2d_operations),
            h2d_bytes: self.h2d_bytes.saturating_sub(earlier.h2d_bytes),
            d2h_operations: self.d2h_operations.saturating_sub(earlier.d2h_operations),
            d2h_bytes: self.d2h_bytes.saturating_sub(earlier.d2h_bytes),
            d2d_operations: self.d2d_operations.saturating_sub(earlier.d2d_operations),
            d2d_bytes: self.d2d_bytes.saturating_sub(earlier.d2d_bytes),
        }
    }
}

/// Snapshot of backend-owned device and host-staging memory.
///
/// Native accelerator backends expose the same counters through this small
/// common contract so bounded workloads can report peak residency without
/// depending on CUDA- or Metal-specific types.  CPU backends return zeros.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BackendMemoryStatsSnapshot {
    pub live_allocation_bytes: usize,
    pub allocation_high_water_bytes: usize,
    pub live_staging_bytes: usize,
    pub staging_high_water_bytes: usize,
}

/// Shared atomic transfer counters used by concrete backends.
#[derive(Debug, Default)]
pub struct BackendTransferStats {
    h2d_operations: AtomicUsize,
    h2d_bytes: AtomicUsize,
    d2h_operations: AtomicUsize,
    d2h_bytes: AtomicUsize,
    d2d_operations: AtomicUsize,
    d2d_bytes: AtomicUsize,
}

impl BackendTransferStats {
    pub const fn new() -> Self {
        Self {
            h2d_operations: AtomicUsize::new(0),
            h2d_bytes: AtomicUsize::new(0),
            d2h_operations: AtomicUsize::new(0),
            d2h_bytes: AtomicUsize::new(0),
            d2d_operations: AtomicUsize::new(0),
            d2d_bytes: AtomicUsize::new(0),
        }
    }

    pub fn snapshot(&self) -> TransferStatsSnapshot {
        TransferStatsSnapshot {
            h2d_operations: self.h2d_operations.load(Ordering::Acquire),
            h2d_bytes: self.h2d_bytes.load(Ordering::Acquire),
            d2h_operations: self.d2h_operations.load(Ordering::Acquire),
            d2h_bytes: self.d2h_bytes.load(Ordering::Acquire),
            d2d_operations: self.d2d_operations.load(Ordering::Acquire),
            d2d_bytes: self.d2d_bytes.load(Ordering::Acquire),
        }
    }

    pub fn reset(&self) {
        self.h2d_operations.store(0, Ordering::Release);
        self.h2d_bytes.store(0, Ordering::Release);
        self.d2h_operations.store(0, Ordering::Release);
        self.d2h_bytes.store(0, Ordering::Release);
        self.d2d_operations.store(0, Ordering::Release);
        self.d2d_bytes.store(0, Ordering::Release);
    }

    pub(crate) fn record_h2d(&self, bytes: usize) {
        self.h2d_operations.fetch_add(1, Ordering::Relaxed);
        self.h2d_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub(crate) fn record_d2h(&self, bytes: usize) {
        self.d2h_operations.fetch_add(1, Ordering::Relaxed);
        self.d2h_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub(crate) fn record_d2d(&self, bytes: usize) {
        self.d2d_operations.fetch_add(1, Ordering::Relaxed);
        self.d2d_bytes.fetch_add(bytes, Ordering::Relaxed);
    }
}

/// Shared admission ledger for allocations owned by a backend.
///
/// The memory pool accounts for caller-visible buffers, but several backend
/// operations also retain device workspaces and deferred scratch buffers. A
/// backend uses this ledger for both classes so every driver allocation is
/// admitted before the driver is called and released when the final owning
/// handle is dropped.
#[cfg(any(feature = "cuda", feature = "metal", test))]
#[derive(Debug)]
pub(crate) struct BackendAllocationLedger {
    pub(crate) live_bytes: AtomicUsize,
    pub(crate) high_water_bytes: AtomicUsize,
    pub(crate) allocation_operations: AtomicUsize,
    cap_bytes: AtomicUsize,
    #[cfg(test)]
    fail_next: AtomicBool,
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
impl Default for BackendAllocationLedger {
    fn default() -> Self {
        Self::with_cap(usize::MAX)
    }
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
impl BackendAllocationLedger {
    pub(crate) fn with_cap(cap_bytes: usize) -> Self {
        Self {
            live_bytes: AtomicUsize::new(0),
            high_water_bytes: AtomicUsize::new(0),
            allocation_operations: AtomicUsize::new(0),
            cap_bytes: AtomicUsize::new(cap_bytes),
            #[cfg(test)]
            fail_next: AtomicBool::new(false),
        }
    }
    /// Inject one backend-driver allocation failure for deterministic tests.
    ///
    /// This fires after the caller reaches the backend allocation seam but
    /// before the live-byte CAS, so a failed native allocation cannot charge
    /// quota and the next attempt remains retryable. Native driver failure
    /// injection is still backend-specific; this hook verifies the shared
    /// accounting and error contract without requiring accelerator hardware.
    #[cfg(test)]
    pub(crate) fn inject_driver_failure(&self) {
        self.fail_next.store(true, Ordering::Release);
    }

    #[cfg(any(feature = "cuda", feature = "metal"))]
    pub(crate) fn set_cap_bytes(&self, cap_bytes: usize) {
        self.cap_bytes.store(cap_bytes, Ordering::Release);
    }

    pub(crate) fn try_reserve(
        self: &Arc<Self>,
        bytes: usize,
    ) -> Result<BackendAllocationGuard, OomError> {
        #[cfg(test)]
        if self.fail_next.swap(false, Ordering::AcqRel) {
            return Err(OomError::DriverOom(
                "injected backend allocation failure".into(),
            ));
        }
        let mut in_use = self.live_bytes.load(Ordering::Relaxed);
        loop {
            let next = in_use.checked_add(bytes).ok_or_else(|| {
                OomError::Invalid(format!(
                    "backend allocation counter overflow: in_use={in_use}, requested={bytes}"
                ))
            })?;
            let cap = self.cap_bytes.load(Ordering::Acquire);
            if next > cap {
                return Err(OomError::QuotaExceeded {
                    requested: bytes,
                    in_use,
                    cap,
                });
            }
            match self.live_bytes.compare_exchange_weak(
                in_use,
                next,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.high_water_bytes.fetch_max(next, Ordering::Relaxed);
                    self.allocation_operations.fetch_add(1, Ordering::Relaxed);
                    return Ok(BackendAllocationGuard {
                        ledger: Arc::clone(self),
                        bytes,
                    });
                }
                Err(observed) => in_use = observed,
            }
        }
    }
}

/// Allocate a zero-initialized host staging vector under the backend
/// allocation ledger.  Native backends use this for temporary host fallbacks
/// and overlap-safe copies so their peak/live accounting includes both device
/// and staging storage.  The reservation is acquired before the `Vec` grows
/// and is released by the returned RAII guard on every success, error, or
/// unwind path.
#[cfg(any(feature = "cuda", feature = "metal", test))]
pub(crate) fn try_host_staging_vec<T: BufferElement>(
    ledger: &Arc<BackendAllocationLedger>,
    len: usize,
    operation: &'static str,
) -> BackendResult<(Vec<T>, BackendAllocationGuard)> {
    let (mut values, reservation) = try_host_staging_capacity(ledger, len, operation)?;
    values.resize(len, T::default());
    Ok((values, reservation))
}

/// Reserve quota for a temporary host vector whose elements do not need to
/// implement [`BufferElement`] (for example, native pointer or synchronization
/// guard arrays). The vector is returned empty but with the requested capacity;
/// callers retain the same typed, retryable allocation failure contract as the
/// zero-initialized staging helper above.
#[cfg(any(feature = "cuda", feature = "metal", test))]
pub(crate) fn try_host_staging_capacity<T>(
    ledger: &Arc<BackendAllocationLedger>,
    capacity: usize,
    operation: &'static str,
) -> BackendResult<(Vec<T>, BackendAllocationGuard)> {
    let bytes = capacity
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| BackendError::InvalidArgument {
            operation,
            reason: "host staging size overflows usize".into(),
        })?;
    let reservation = ledger.try_reserve(bytes).map_err(BackendError::from)?;
    let mut values = Vec::new();
    if values.try_reserve_exact(capacity).is_err() {
        return Err(BackendError::Allocation {
            reason: format!("{operation}: host staging allocation failed for {bytes} bytes"),
        });
    }
    Ok((values, reservation))
}

/// RAII reservation in a [`BackendAllocationLedger`].
#[cfg(any(feature = "cuda", feature = "metal", test))]
#[derive(Debug)]
pub(crate) struct BackendAllocationGuard {
    ledger: Arc<BackendAllocationLedger>,
    bytes: usize,
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
impl BackendAllocationGuard {
    /// Compatibility constructor for tests and already-admitted callers.
    /// Driver-facing paths should use [`BackendAllocationLedger::try_reserve`]
    /// so quota failures are returned instead of panicking.
    #[cfg(all(test, any(feature = "cuda", feature = "metal")))]
    pub(crate) fn new(ledger: Arc<BackendAllocationLedger>, bytes: usize) -> Self {
        ledger
            .try_reserve(bytes)
            .expect("backend allocation guard constructed over quota")
    }
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
impl Drop for BackendAllocationGuard {
    fn drop(&mut self) {
        self.ledger
            .live_bytes
            .fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

/// Convert a panic from a legacy backend implementation into the checked
/// operation contract. Backend implementations predate `try_*` methods and
/// still expose infallible hot-path entry points; checked callers must not
/// inherit those process-aborting failure semantics.
pub(crate) fn catch_backend_panic<T, F>(operation: &'static str, f: F) -> BackendResult<T>
where
    F: FnOnce() -> T,
{
    catch_unwind(AssertUnwindSafe(f)).map_err(|payload| {
        let reason = if let Some(message) = payload.downcast_ref::<&'static str>() {
            (*message).to_string()
        } else if let Some(message) = payload.downcast_ref::<String>() {
            message.clone()
        } else {
            "backend implementation panicked".to_string()
        };
        BackendError::OperationFailed { operation, reason }
    })
}

/// Completion fence returned after asynchronous backend work is submitted.
///
/// A token owns the backend-native completion primitive and must not report
/// completion until all resources referenced by the submitted work are safe
/// to release. CUDA implementations use a stream event; Metal implementations
/// own the committed command buffer and its deferred scratch resources.
pub trait BackendCompletion: Send {
    /// Wait for the submitted work and surface any device/command failure.
    fn wait(&mut self) -> BackendResult<()>;

    /// Cancel a submission when the backend can do so safely.
    ///
    /// The conservative default waits for already-enqueued work: releasing a
    /// buffer before that work reaches its fence would be a use-after-free.
    fn cancel(&mut self) -> BackendResult<()> {
        self.wait()
    }
}

#[derive(Debug, Default)]
struct ImmediateBackendCompletion;

impl BackendCompletion for ImmediateBackendCompletion {
    fn wait(&mut self) -> BackendResult<()> {
        Ok(())
    }
}

mod validate;
#[cfg(feature = "cuda")]
pub(crate) use self::validate::validate_host_csr_structure;
pub(crate) use self::validate::{
    checked_strided_span, gs_color_sweep_host, host_csr_transpose, require_finite_scalar,
    require_precision_capability, require_reduction_capability, require_scalar_capability,
    try_collect_backend, try_zeroed_host, validate_bem_regular_pair_f64_inputs,
    validate_bem_regular_pair_f64_outputs, validate_bem_regular_pair_inputs,
    validate_bem_regular_pair_outputs, validate_gather4_finite_inputs, validate_gemm_args,
    validate_gemv_args, validate_syrk_args, validate_trsm_args,
};

mod capabilities;
pub use self::capabilities::{
    BackendCapabilities, BemCapabilities, CapabilityState, PrecisionCastCapabilities,
    ReductionAccumulation, ReductionCapabilities, ReductionFailureGuarantee, ReductionOrdering,
    ReductionPolicy, ReductionSynchronization, ScalarCapabilities,
};

#[cfg(test)]
mod reduction_contract_tests;

#[cfg(test)]
mod tests;

mod host;
pub(crate) use self::host::{
    host_cast_from_lower, host_cast_to_lower, host_copy_h2d_cast, host_gather4, host_gemm,
    host_gemv, host_syrk, host_trsm,
};
#[cfg(test)]
pub(crate) use self::host::{host_syrk_compute, host_trsm_compute};

mod views;
pub use self::views::{
    BufferView, Transpose, ViewAliasing, ViewDescriptor, ViewLayout,
    checked_fft_strided_view_descriptor, checked_fft_view_descriptor,
};

/// Reduction policy used by deterministic gradient checks and replayable QA.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ReductionMode {
    /// Use the backend's fastest available reduction.
    #[default]
    Fast,
    /// Use a fixed, repeatable reduction path. The current checked backend
    /// implementation promotes this to the stronger deterministic host-order
    /// path on every backend.
    Reproducible,
    /// Use a fixed-order host reduction after one device read.
    Deterministic,
}

/// Row bucket summary produced by CSR SpMV inspectors.
///
/// The thresholds mirror the Metal adaptive kernels: short rows use a
/// thread-per-row strategy, medium rows use one SIMD/simdgroup lane set per
/// row, and long rows use a full cooperative threadgroup/CTA-style row.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CsrRowBucketStats {
    pub rows: usize,
    pub nnz: usize,
    pub empty_rows: usize,
    pub short_rows: usize,
    pub simd_rows: usize,
    pub cta_rows: usize,
    pub max_nnz_per_row: usize,
}

impl CsrRowBucketStats {
    pub const SHORT_ROW_NNZ: usize = 8;
    pub const CTA_ROW_NNZ: usize = 64;

    /// Build row-bucket statistics after validating the CSR row-pointer
    /// contract.  The old infallible helper is retained for compatibility,
    /// but checked CSR/prebake paths should use this method so malformed or
    /// descending `i32` pointers cannot overflow subtraction or silently
    /// produce a misleading inspector.
    pub fn try_from_row_ptr(row_ptr: &[i32]) -> BackendResult<Self> {
        if row_ptr.is_empty() {
            return Ok(Self::default());
        }
        if row_ptr[0] != 0 {
            return Err(BackendError::InvalidArgument {
                operation: "csr_row_buckets",
                reason: "row pointer must start at zero".into(),
            });
        }

        let mut stats = Self {
            rows: row_ptr.len() - 1,
            ..Self::default()
        };
        let mut previous = 0_i32;
        for &current in &row_ptr[1..] {
            if current < previous {
                return Err(BackendError::InvalidArgument {
                    operation: "csr_row_buckets",
                    reason: "row pointers must be nondecreasing".into(),
                });
            }
            let nnz =
                usize::try_from(current - previous).map_err(|_| BackendError::InvalidArgument {
                    operation: "csr_row_buckets",
                    reason: "row pointer difference does not fit usize".into(),
                })?;
            stats.nnz =
                stats
                    .nnz
                    .checked_add(nnz)
                    .ok_or_else(|| BackendError::InvalidArgument {
                        operation: "csr_row_buckets",
                        reason: "nonzero count overflows usize".into(),
                    })?;
            stats.max_nnz_per_row = stats.max_nnz_per_row.max(nnz);
            if nnz == 0 {
                stats.empty_rows += 1;
            } else if nnz < Self::SHORT_ROW_NNZ {
                stats.short_rows += 1;
            } else if nnz < Self::CTA_ROW_NNZ {
                stats.simd_rows += 1;
            } else {
                stats.cta_rows += 1;
            }
            previous = current;
        }
        Ok(stats)
    }

    /// Compatibility summary for trusted callers.  Malformed metadata now
    /// yields an empty summary instead of panicking on signed subtraction;
    /// checked callers get the exact typed error from [`Self::try_from_row_ptr`].
    pub fn from_row_ptr(row_ptr: &[i32]) -> Self {
        Self::try_from_row_ptr(row_ptr).unwrap_or_default()
    }
}

/// Backend-selected execution family for a prebaked CSR SpMV.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsrSpmvFormat {
    Empty,
    PlainCsr,
    RowBucketedCsr,
    CusparseGenericCsr,
}

/// Cached CSR inspector record attached to a prebaked SpMV handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CsrSpmvInspector {
    pub row_buckets: CsrRowBucketStats,
    pub format: CsrSpmvFormat,
    pub workspace_bytes: usize,
    pub algorithm: &'static str,
}

/// One dense GEMV problem in a grouped backend dispatch.
pub struct GemvBatch<'a, B, T: DeviceScalar> {
    pub op: Transpose,
    pub m: usize,
    pub n: usize,
    pub alpha: T,
    pub a: &'a B,
    pub lda: usize,
    pub x: &'a B,
    pub beta: T,
    pub y: &'a mut B,
}

/// One dense GEMM problem in a grouped backend dispatch.
pub struct GemmBatch<'a, B, T: DeviceScalar> {
    pub transa: Transpose,
    pub transb: Transpose,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub alpha: T,
    pub a: &'a B,
    pub lda: usize,
    pub b: &'a B,
    pub ldb: usize,
    pub beta: T,
    pub c: &'a mut B,
    pub ldc: usize,
}

/// Trait satisfied by every device buffer type.
///
/// A buffer carries the element type as a generic parameter and reports its
/// length. Backends are free to choose the concrete representation
/// (`Vec<T>` for CPU, a CUDA device allocation for CUDA, etc.).
pub trait DeviceBufferLike<T: BufferElement>: Send + Sync {
    /// Number of elements in the buffer.
    fn len(&self) -> usize;

    /// Convenience: empty check.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Compute backend abstraction.
///
/// The surface covers memory movement, BLAS-1, BLAS-2/3 dense products, CSR
/// SpMV, fixed-width gather, precision casts, and solver-oriented reductions.
/// Not every backend implements every operation natively; call
/// [`Self::capabilities`] to distinguish device-native paths from explicit
/// host/shared-memory fallbacks.
///
/// # Memory ownership
///
/// `Backend::raw_alloc` is the unsafe lower-level entry point that the
/// [`crate::MemoryPool`] uses after the quota check has passed. Do not call
/// it directly from upper crates — go through `MemoryPool::try_alloc`.
pub trait Backend: Send + Sync + 'static {
    /// Concrete buffer type produced by this backend.
    type Buffer<T: BufferElement>: DeviceBufferLike<T>;

    /// A CSR matrix that has been pre-prepared for repeated SpMV.
    ///
    /// Borrows the source buffers, so they must outlive the prebaked
    /// handle. The cost of building one is roughly equal to a single
    /// `spmv_csr` setup; the win is that subsequent [`Self::spmv_prebaked`]
    /// calls skip the descriptor / workspace creation work.
    ///
    /// On the cudarc-backed `CudaBackend` this caches the
    /// `cusparseSpMatDescr_t`, `cusparseSpMV` workspace allocation, and
    /// inspector metadata. CPU and Metal prebakes cache row-bucket analysis
    /// so sparse kernels can make repeatable format/dispatch choices without
    /// rescanning the sparsity pattern on every call.
    type CsrPrebaked<'a, T: DeviceScalar>: Send + Sync
    where
        Self: 'a;

    /// Static operation capability table for this backend implementation.
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::unsupported(std::any::type_name::<Self>())
    }

    /// Capability of an analytic constant-panel BEM wavenumber-derivative
    /// assembly path.  This is separate from the value pair capability:
    /// callers must not infer derivative support from a value kernel and then
    /// silently finite-difference or stage the operation on the host.
    fn bem_wavenumber_derivative_capability(&self) -> CapabilityState {
        CapabilityState::Unsupported
    }

    /// Capability of the native constant-panel BEM source-collocation x
    /// derivative assembly path.  This is separate from value and wavenumber
    /// derivative capabilities so callers cannot silently finite-difference
    /// or stage a geometry derivative on the host.
    fn bem_source_x_derivative_capability(&self) -> CapabilityState {
        CapabilityState::Unsupported
    }

    /// Capability of the native constant-panel BEM source-collocation y
    /// derivative assembly path.  This is separate from value and other
    /// derivative capabilities so callers cannot silently finite-difference
    /// or stage a geometry derivative on the host.
    fn bem_source_y_derivative_capability(&self) -> CapabilityState {
        CapabilityState::Unsupported
    }

    /// Return checked host/device transfer counters for this backend.
    fn transfer_stats(&self) -> TransferStatsSnapshot {
        TransferStatsSnapshot::default()
    }

    /// Return backend-owned allocation and staging watermarks.
    fn memory_stats(&self) -> BackendMemoryStatsSnapshot {
        BackendMemoryStatsSnapshot::default()
    }

    /// Reset checked host/device transfer counters for a bounded measurement.
    fn reset_transfer_stats(&self) {}

    /// Checked in-place complex FFT entry point for backend-native plans.
    ///
    /// The default is explicitly unsupported. FFT crates can probe the
    /// capability table before calling this method and fall back to their
    /// normative CPU path without staging ambiguity. `fft_len` is the
    /// length of one transform; `buffer.len()` may contain a contiguous
    /// batch of those transforms.
    fn try_fft_in_place<T: BufferElement>(
        &self,
        buffer: &mut Self::Buffer<T>,
        fft_len: usize,
        inverse: bool,
    ) -> BackendResult<()> {
        let _ = (buffer, fft_len, inverse);
        Err(BackendError::Unsupported {
            backend: std::any::type_name::<Self>(),
            operation: "fft",
            scalar: std::any::type_name::<T>(),
        })
    }

    /// Checked in-place FFT over regularly strided, independent transforms.
    ///
    /// The default is unsupported so a backend cannot accidentally advertise
    /// a host-staged implementation as a native strided transform.
    fn try_fft_strided_in_place<T: BufferElement>(
        &self,
        buffer: &mut Self::Buffer<T>,
        fft_len: usize,
        batch: usize,
        stride: usize,
        inverse: bool,
    ) -> BackendResult<()> {
        let _ = (buffer, fft_len, batch, stride, inverse);
        Err(BackendError::Unsupported {
            backend: std::any::type_name::<Self>(),
            operation: "fft_strided",
            scalar: std::any::type_name::<T>(),
        })
    }

    /// Checked out-of-place real-to-half-complex FFT.
    ///
    /// `input` contains `buffer.len() / fft_len` contiguous real transforms;
    /// `output` contains the corresponding `fft_len / 2 + 1` bins per batch.
    /// Backends that implement this natively must not stage through host
    /// memory. The inverse operation below has the matching C2R contract.
    fn try_real_fft_forward<T>(
        &self,
        input: &Self::Buffer<T>,
        output: &mut Self::Buffer<Complex<T>>,
        fft_len: usize,
    ) -> BackendResult<()>
    where
        T: BufferElement,
        Complex<T>: BufferElement,
    {
        let _ = (input, output, fft_len);
        Err(BackendError::Unsupported {
            backend: std::any::type_name::<Self>(),
            operation: "real_fft_forward",
            scalar: std::any::type_name::<T>(),
        })
    }

    /// Checked out-of-place half-complex-to-real FFT. cuFFT and this trait
    /// use the same unnormalised inverse convention, so callers that promise
    /// a round trip apply `1 / fft_len` after the native operation.
    fn try_real_fft_inverse<T>(
        &self,
        input: &Self::Buffer<Complex<T>>,
        output: &mut Self::Buffer<T>,
        fft_len: usize,
    ) -> BackendResult<()>
    where
        T: BufferElement,
        Complex<T>: BufferElement,
    {
        let _ = (input, output, fft_len);
        Err(BackendError::Unsupported {
            backend: std::any::type_name::<Self>(),
            operation: "real_fft_inverse",
            scalar: std::any::type_name::<T>(),
        })
    }

    /// Checked out-of-place real transform. The operation is batched by
    /// contiguous `fft_len`-element rows in both buffers. DCT/DST kernels use
    /// a backend-owned or cached work area, so steady-state calls do not
    /// allocate or transfer data through the host.
    fn try_real_transform<T>(
        &self,
        kind: RealTransformKind,
        input: &Self::Buffer<T>,
        output: &mut Self::Buffer<T>,
        fft_len: usize,
    ) -> BackendResult<()>
    where
        T: BufferElement,
    {
        let _ = (kind, input, output, fft_len);
        Err(BackendError::Unsupported {
            backend: std::any::type_name::<Self>(),
            operation: "real_transform",
            scalar: std::any::type_name::<T>(),
        })
    }

    /// Validate a raw-buffer view before passing its offset and stride to a
    /// device binding. Backends may use the returned metadata directly in a
    /// kernel launch; no pointer arithmetic is performed by this method.
    fn checked_view<T: BufferElement>(
        &self,
        buffer: &Self::Buffer<T>,
        offset: usize,
        len: usize,
        stride: usize,
    ) -> BackendResult<BufferView> {
        BufferView::new(buffer.len(), offset, len, stride)
    }

    /// Validate a view and materialize the complete descriptor required by a
    /// native binding. The backend name is part of the descriptor so a view
    /// cannot be accidentally reused across device contexts.
    fn checked_view_descriptor<T: BufferElement>(
        &self,
        buffer: &Self::Buffer<T>,
        offset: usize,
        len: usize,
        stride: usize,
    ) -> BackendResult<ViewDescriptor> {
        let view = self.checked_view(buffer, offset, len, stride)?;
        view.descriptor::<T>(self.capabilities().backend_name)
    }

    /// Validate and bind a view descriptor with an explicit aliasing policy.
    ///
    /// Backend implementations that launch native kernels with strided views
    /// should use this entry point immediately before constructing a raw
    /// pointer. The default path still stages through checked copies, but the
    /// same descriptor contract now applies to native and fallback adapters.
    fn checked_view_descriptor_with_aliasing<T: BufferElement>(
        &self,
        buffer: &Self::Buffer<T>,
        offset: usize,
        len: usize,
        stride: usize,
        aliasing: ViewAliasing,
    ) -> BackendResult<ViewDescriptor> {
        let view = self.checked_view(buffer, offset, len, stride)?;
        view.descriptor_with_aliasing::<T>(self.capabilities().backend_name, aliasing)
    }

    /// Validate and bind a rank-N descriptor against a backend allocation.
    ///
    /// Native kernels with a logical matrix or tensor view should use this
    /// helper immediately before extracting offsets and strides for launch.
    /// The descriptor remains backend-neutral; the allocation-bound check is
    /// the part that cannot be performed by [`ViewDescriptor::validate`]
    /// alone.
    fn checked_nd_view_descriptor<T: BufferElement>(
        &self,
        buffer: &Self::Buffer<T>,
        shape: &[usize],
        byte_offset: usize,
        element_strides: &[usize],
        aliasing: ViewAliasing,
    ) -> BackendResult<ViewDescriptor> {
        let descriptor = ViewDescriptor::try_new(
            shape,
            byte_offset,
            element_strides,
            ViewLayout::Strided,
            std::any::type_name::<T>(),
            self.capabilities().backend_name,
            aliasing,
        )?;
        descriptor.validate_for_buffer::<T>(buffer.len(), descriptor.device)?;
        Ok(descriptor)
    }

    /// Validate and build a rank-N descriptor with an explicit layout.
    ///
    /// The existing [`Self::checked_nd_view_descriptor`] helper remains the
    /// conservative arbitrary-stride form. Native tensor launch paths that
    /// require contiguous row/column-major storage should use this method so
    /// the requirement is checked before dispatch.
    fn checked_nd_view_descriptor_with_layout<T: BufferElement>(
        &self,
        buffer: &Self::Buffer<T>,
        shape: &[usize],
        byte_offset: usize,
        element_strides: &[usize],
        layout: ViewLayout,
        aliasing: ViewAliasing,
    ) -> BackendResult<ViewDescriptor> {
        let descriptor = ViewDescriptor::try_new(
            shape,
            byte_offset,
            element_strides,
            layout,
            std::any::type_name::<T>(),
            self.capabilities().backend_name,
            aliasing,
        )?;
        descriptor.validate_for_buffer::<T>(buffer.len(), descriptor.device)?;
        Ok(descriptor)
    }

    /// Bind a padded row-major matrix through the same checked descriptor
    /// contract used by rank-N native launches.  `ld` is the physical row
    /// stride; a zero-sized matrix still receives a non-zero metadata stride
    /// so descriptor validation cannot be bypassed by degenerate dimensions.
    fn checked_dense_matrix_descriptor<T: BufferElement>(
        &self,
        buffer: &Self::Buffer<T>,
        rows: usize,
        cols: usize,
        ld: usize,
        aliasing: ViewAliasing,
    ) -> BackendResult<ViewDescriptor> {
        self.checked_nd_view_descriptor_with_layout(
            buffer,
            &[rows, cols],
            0,
            &[ld.max(1), 1],
            ViewLayout::Strided,
            aliasing,
        )
    }

    /// Bind a contiguous vector for a dense operation.  Keeping this helper
    /// beside the matrix binding prevents native backends from validating only
    /// the matrix while passing an unchecked vector pointer to the driver.
    fn checked_dense_vector_descriptor<T: BufferElement>(
        &self,
        buffer: &Self::Buffer<T>,
        aliasing: ViewAliasing,
    ) -> BackendResult<ViewDescriptor> {
        self.checked_nd_view_descriptor_with_layout(
            buffer,
            &[buffer.len()],
            0,
            &[1],
            ViewLayout::RowMajor,
            aliasing,
        )
    }

    /// Validate a rank-N descriptor whose public strides are expressed in
    /// bytes, as native tensor APIs commonly require.
    ///
    /// Typed buffers cannot represent sub-element or misaligned strides, so
    /// every byte stride must be an exact multiple of `size_of::<T>()`.
    fn checked_nd_byte_strided_view_descriptor<T: BufferElement>(
        &self,
        buffer: &Self::Buffer<T>,
        shape: &[usize],
        byte_offset: usize,
        byte_strides: &[usize],
        aliasing: ViewAliasing,
    ) -> BackendResult<ViewDescriptor> {
        let element_size = std::mem::size_of::<T>();
        if element_size == 0 {
            return Err(BackendError::InvalidArgument {
                operation: "view_descriptor",
                reason: "zero-sized buffer elements are not bindable".into(),
            });
        }
        let element_strides = try_collect_backend(
            byte_strides.iter().map(|stride| {
                if !stride.is_multiple_of(element_size) {
                    return Err(BackendError::InvalidArgument {
                        operation: "view_descriptor",
                        reason: format!(
                            "byte stride {stride} is not aligned to element size {element_size}"
                        ),
                    });
                }
                Ok(stride / element_size)
            }),
            byte_strides.len(),
            "view_descriptor",
        )?;
        self.checked_nd_view_descriptor(buffer, shape, byte_offset, &element_strides, aliasing)
    }

    /// Copy a strided device view to host memory after validating its bounds.
    ///
    /// This default is intentionally host-staged so the checked contract is
    /// available to every backend. CUDA and Metal may override it with a
    /// native strided copy without changing view validation semantics.
    fn try_copy_view_d2h<T: BufferElement>(
        &self,
        buffer: &Self::Buffer<T>,
        view: BufferView,
        host: &mut [T],
    ) -> BackendResult<()> {
        self.checked_view_descriptor(buffer, view.offset, view.len, view.stride)?;
        if host.len() != view.len {
            return Err(BackendError::InvalidArgument {
                operation: "copy_view_d2h",
                reason: format!("host length {} != view length {}", host.len(), view.len),
            });
        }
        if view.len == 0 {
            return Ok(());
        }
        let mut full = try_zeroed_host(buffer.len(), "view_copy")?;
        self.try_copy_d2h(buffer, &mut full)?;
        for (destination, source) in host
            .iter_mut()
            .zip((0..view.len).map(|index| view.checked_element_index(index)))
        {
            *destination = full[source?];
        }
        Ok(())
    }

    /// Copy host values into a strided device view after validating bounds.
    ///
    /// The default implementation preserves all non-view elements through a
    /// host-staged round trip; native backends can replace the transfer path.
    fn try_copy_view_h2d<T: BufferElement>(
        &self,
        host: &[T],
        buffer: &mut Self::Buffer<T>,
        view: BufferView,
    ) -> BackendResult<()> {
        self.checked_view_descriptor(buffer, view.offset, view.len, view.stride)?;
        if host.len() != view.len {
            return Err(BackendError::InvalidArgument {
                operation: "copy_view_h2d",
                reason: format!("host length {} != view length {}", host.len(), view.len),
            });
        }
        if view.len == 0 {
            return Ok(());
        }
        let mut full = try_zeroed_host(buffer.len(), "view_copy")?;
        self.try_copy_d2h(buffer, &mut full)?;
        for (source, destination) in host
            .iter()
            .zip((0..view.len).map(|index| view.checked_element_index(index)))
        {
            full[destination?] = *source;
        }
        self.try_copy_h2d(&full, buffer)
    }

    /// Copy views through checked host staging. The staging boundary gives
    /// the operation memmove semantics even when a backend's native copy
    /// primitive has no overlap guarantee; the buffers may therefore be
    /// distinct or aliased without exposing a native aliasing hazard.
    fn try_copy_views_d2d<T: BufferElement>(
        &self,
        source: &Self::Buffer<T>,
        source_view: BufferView,
        destination: &mut Self::Buffer<T>,
        destination_view: BufferView,
    ) -> BackendResult<()> {
        self.checked_view_descriptor(
            source,
            source_view.offset,
            source_view.len,
            source_view.stride,
        )?;
        self.checked_view_descriptor_with_aliasing(
            destination,
            destination_view.offset,
            destination_view.len,
            destination_view.stride,
            ViewAliasing::OverlapSafe,
        )?;
        if source_view.len != destination_view.len {
            return Err(BackendError::InvalidArgument {
                operation: "copy_views_d2d",
                reason: format!(
                    "source view length {} != destination view length {}",
                    source_view.len, destination_view.len
                ),
            });
        }
        if source_view.len == 0 {
            return Ok(());
        }
        let mut staged = try_zeroed_host(source_view.len, "copy_views_d2d")?;
        self.try_copy_view_d2h(source, source_view, &mut staged)?;
        self.try_copy_view_h2d(&staged, destination, destination_view)
    }

    /// Copy between overlapping views in one buffer with defined memmove
    /// semantics. The source is fully staged before the destination is
    /// written, so overlapping and identical views are both supported.
    fn try_copy_view_in_place<T: BufferElement>(
        &self,
        buffer: &mut Self::Buffer<T>,
        source: BufferView,
        destination: BufferView,
    ) -> BackendResult<()> {
        self.checked_view_descriptor(buffer, source.offset, source.len, source.stride)?;
        self.checked_view_descriptor_with_aliasing(
            buffer,
            destination.offset,
            destination.len,
            destination.stride,
            ViewAliasing::OverlapSafe,
        )?;
        if source.len != destination.len {
            return Err(BackendError::InvalidArgument {
                operation: "copy_view_in_place",
                reason: format!(
                    "source view length {} != destination view length {}",
                    source.len, destination.len
                ),
            });
        }
        if source.len == 0 {
            return Ok(());
        }
        let mut staged = try_zeroed_host(source.len, "copy_view_in_place")?;
        self.try_copy_view_d2h(buffer, source, &mut staged)?;
        self.try_copy_view_h2d(&staged, buffer, destination)
    }

    /// Copy a checked rank-N device view to a contiguous host slice.
    ///
    /// Logical iteration is row-major over `descriptor.shape`; physical
    /// addresses are selected by `byte_offset` and `element_strides`.
    fn try_copy_nd_view_d2h<T: BufferElement>(
        &self,
        buffer: &Self::Buffer<T>,
        descriptor: &ViewDescriptor,
        host: &mut [T],
    ) -> BackendResult<()> {
        let device = self.capabilities().backend_name;
        descriptor.validate_for_buffer::<T>(buffer.len(), device)?;
        let count = descriptor.checked_element_count()?;
        if host.len() != count {
            return Err(BackendError::InvalidArgument {
                operation: "copy_nd_view_d2h",
                reason: format!(
                    "host length {} != descriptor element count {count}",
                    host.len()
                ),
            });
        }
        if count == 0 {
            return Ok(());
        }
        let mut full = try_zeroed_host(buffer.len(), "copy_view_h2d")?;
        self.try_copy_d2h(buffer, &mut full)?;
        for (index, destination) in host.iter_mut().enumerate() {
            let source = descriptor.checked_element_index::<T>(index, device)?;
            *destination = full[source];
        }
        Ok(())
    }

    /// Copy a contiguous host slice into a checked rank-N device view.
    ///
    /// A descriptor marked [`ViewAliasing::ReadOnly`] is rejected before the
    /// destination buffer is downloaded or mutated.
    fn try_copy_nd_view_h2d<T: BufferElement>(
        &self,
        host: &[T],
        buffer: &mut Self::Buffer<T>,
        descriptor: &ViewDescriptor,
    ) -> BackendResult<()> {
        let device = self.capabilities().backend_name;
        descriptor.validate_for_buffer::<T>(buffer.len(), device)?;
        if descriptor.aliasing == ViewAliasing::ReadOnly {
            return Err(BackendError::InvalidArgument {
                operation: "copy_nd_view_h2d",
                reason: "destination descriptor is read-only".into(),
            });
        }
        let count = descriptor.checked_element_count()?;
        if host.len() != count {
            return Err(BackendError::InvalidArgument {
                operation: "copy_nd_view_h2d",
                reason: format!(
                    "host length {} != descriptor element count {count}",
                    host.len()
                ),
            });
        }
        if count == 0 {
            return Ok(());
        }
        let mut full = try_zeroed_host(buffer.len(), "view_copy")?;
        self.try_copy_d2h(buffer, &mut full)?;
        for (index, source) in host.iter().enumerate() {
            let destination = descriptor.checked_element_index::<T>(index, device)?;
            full[destination] = *source;
        }
        self.try_copy_h2d(&full, buffer)
    }

    /// Copy equally shaped rank-N views through checked host staging.
    fn try_copy_nd_views_d2d<T: BufferElement>(
        &self,
        source: &Self::Buffer<T>,
        source_descriptor: &ViewDescriptor,
        destination: &mut Self::Buffer<T>,
        destination_descriptor: &ViewDescriptor,
    ) -> BackendResult<()> {
        let device = self.capabilities().backend_name;
        source_descriptor.validate_for_buffer::<T>(source.len(), device)?;
        destination_descriptor.validate_for_buffer::<T>(destination.len(), device)?;
        if destination_descriptor.aliasing == ViewAliasing::ReadOnly {
            return Err(BackendError::InvalidArgument {
                operation: "copy_nd_views_d2d",
                reason: "destination descriptor is read-only".into(),
            });
        }
        let source_count = source_descriptor.checked_element_count()?;
        let destination_count = destination_descriptor.checked_element_count()?;
        if source_count != destination_count {
            return Err(BackendError::InvalidArgument {
                operation: "copy_nd_views_d2d",
                reason: format!(
                    "source descriptor count {source_count} != destination descriptor count {destination_count}"
                ),
            });
        }
        let mut staged = try_zeroed_host(source_count, "copy_nd_views_d2d")?;
        self.try_copy_nd_view_d2h(source, source_descriptor, &mut staged)?;
        self.try_copy_nd_view_h2d(&staged, destination, destination_descriptor)
    }

    /// Copy rank-N views in one allocation with defined memmove semantics.
    fn try_copy_nd_view_in_place<T: BufferElement>(
        &self,
        buffer: &mut Self::Buffer<T>,
        source_descriptor: &ViewDescriptor,
        destination_descriptor: &ViewDescriptor,
    ) -> BackendResult<()> {
        if destination_descriptor.aliasing != ViewAliasing::OverlapSafe {
            return Err(BackendError::InvalidArgument {
                operation: "copy_nd_view_in_place",
                reason: "in-place destination must declare overlap-safe semantics".into(),
            });
        }
        let device = self.capabilities().backend_name;
        source_descriptor.validate_for_buffer::<T>(buffer.len(), device)?;
        destination_descriptor.validate_for_buffer::<T>(buffer.len(), device)?;
        let source_count = source_descriptor.checked_element_count()?;
        let destination_count = destination_descriptor.checked_element_count()?;
        if source_count != destination_count {
            return Err(BackendError::InvalidArgument {
                operation: "copy_nd_view_in_place",
                reason: format!(
                    "source descriptor count {source_count} != destination descriptor count {destination_count}"
                ),
            });
        }
        let mut staged = try_zeroed_host(source_count, "copy_nd_view_in_place")?;
        self.try_copy_nd_view_d2h(buffer, source_descriptor, &mut staged)?;
        self.try_copy_nd_view_h2d(&staged, buffer, destination_descriptor)
    }

    /// Checked host-to-device copy for boundary code and FFI adapters.
    fn try_copy_h2d<T: BufferElement>(
        &self,
        host: &[T],
        dev: &mut Self::Buffer<T>,
    ) -> BackendResult<()> {
        if host.len() != dev.len() {
            return Err(BackendError::InvalidArgument {
                operation: "copy_h2d",
                reason: format!("host length {} != device length {}", host.len(), dev.len()),
            });
        }
        catch_backend_panic("copy_h2d", || self.copy_h2d(host, dev))?;
        self.check_pending_error()
    }

    /// Checked device-to-host copy for boundary code and FFI adapters.
    fn try_copy_d2h<T: BufferElement>(
        &self,
        dev: &Self::Buffer<T>,
        host: &mut [T],
    ) -> BackendResult<()> {
        if host.len() != dev.len() {
            return Err(BackendError::InvalidArgument {
                operation: "copy_d2h",
                reason: format!("host length {} != device length {}", host.len(), dev.len()),
            });
        }
        catch_backend_panic("copy_d2h", || self.copy_d2h(dev, host))?;
        self.check_pending_error()
    }

    /// Checked device-to-device copy for boundary code and FFI adapters.
    fn try_copy_d2d<T: BufferElement>(
        &self,
        src: &Self::Buffer<T>,
        dst: &mut Self::Buffer<T>,
    ) -> BackendResult<()> {
        if src.len() != dst.len() {
            return Err(BackendError::InvalidArgument {
                operation: "copy_d2d",
                reason: format!(
                    "source length {} != destination length {}",
                    src.len(),
                    dst.len()
                ),
            });
        }
        catch_backend_panic("copy_d2d", || self.copy_d2d(src, dst))?;
        self.check_pending_error()
    }

    /// Validate backend-specific dimension limits before a checked dense
    /// GEMV enters a native kernel. The default has no narrowing constraint.
    fn validate_gemv_backend_limits<T: DeviceScalar>(
        &self,
        _op: Transpose,
        _m: usize,
        _n: usize,
        _lda: usize,
    ) -> BackendResult<()> {
        Ok(())
    }

    /// Checked dense GEMV entry point. This is the preferred API for inputs
    /// whose shape comes from a mesh or file; the legacy `gemv` method remains
    /// available for hot paths that have already validated their dimensions.
    #[allow(clippy::too_many_arguments)]
    fn try_gemv<T: DeviceScalar>(
        &self,
        op: Transpose,
        m: usize,
        n: usize,
        alpha: T,
        a: &Self::Buffer<T>,
        lda: usize,
        x: &Self::Buffer<T>,
        beta: T,
        y: &mut Self::Buffer<T>,
    ) -> BackendResult<()> {
        let capabilities = self.capabilities();
        require_scalar_capability::<T>(capabilities.backend_name, "gemv", capabilities.gemv)?;
        validate_gemv_args(op, m, n, a.len(), lda, x.len(), y.len())?;
        let (stored_rows, stored_cols) = match op {
            Transpose::NoTrans => (m, n),
            Transpose::Trans | Transpose::ConjTrans => (n, m),
        };
        self.checked_dense_matrix_descriptor(
            a,
            stored_rows,
            stored_cols,
            lda,
            ViewAliasing::ReadOnly,
        )?;
        self.checked_dense_vector_descriptor(x, ViewAliasing::ReadOnly)?;
        self.checked_dense_vector_descriptor(y, ViewAliasing::Disjoint)?;
        require_finite_scalar("gemv", "alpha", alpha)?;
        require_finite_scalar("gemv", "beta", beta)?;
        self.validate_gemv_backend_limits::<T>(op, m, n, lda)?;
        // Preserve BLAS semantics for a zero-column matrix: there is no
        // matrix product, but the output scaling epilogue still applies.
        // Handling this before backend dispatch keeps CPU, CUDA, and Metal
        // behavior identical and avoids vendor kernels rejecting `n == 0`.
        if m == 0 {
            return Ok(());
        }
        if n == 0 {
            return self.try_scale(beta, y);
        }
        catch_backend_panic("gemv", || self.gemv(op, m, n, alpha, a, lda, x, beta, y))?;
        self.check_pending_error()
    }

    /// Checked dense GEMM entry point.
    #[allow(clippy::too_many_arguments)]
    fn try_gemm<T: DeviceScalar>(
        &self,
        transa: Transpose,
        transb: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: T,
        a: &Self::Buffer<T>,
        lda: usize,
        b: &Self::Buffer<T>,
        ldb: usize,
        beta: T,
        c: &mut Self::Buffer<T>,
        ldc: usize,
    ) -> BackendResult<()> {
        let capabilities = self.capabilities();
        require_scalar_capability::<T>(capabilities.backend_name, "gemm", capabilities.gemm)?;
        let (a_rows, a_cols) = match transa {
            Transpose::NoTrans => (m, k),
            Transpose::Trans | Transpose::ConjTrans => (k, m),
        };
        let (b_rows, b_cols) = match transb {
            Transpose::NoTrans => (k, n),
            Transpose::Trans | Transpose::ConjTrans => (n, k),
        };
        validate_gemm_args(
            transa,
            transb,
            m,
            n,
            k,
            a.len(),
            lda,
            b.len(),
            ldb,
            c.len(),
            ldc,
        )?;
        self.checked_dense_matrix_descriptor(a, a_rows, a_cols, lda, ViewAliasing::ReadOnly)?;
        self.checked_dense_matrix_descriptor(b, b_rows, b_cols, ldb, ViewAliasing::ReadOnly)?;
        self.checked_dense_matrix_descriptor(c, m, n, ldc, ViewAliasing::Disjoint)?;
        require_finite_scalar("gemm", "alpha", alpha)?;
        require_finite_scalar("gemm", "beta", beta)?;
        catch_backend_panic("gemm", || {
            self.gemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
        })?;
        self.check_pending_error()
    }

    /// Allocate a buffer of `len` elements for a pool or backend operation.
    ///
    /// The memory pool reserves its caller-visible quota before invoking this
    /// method. Backend implementations additionally reserve their shared
    /// backend ledger before calling the driver, so direct internal workspaces
    /// and pooled buffers remain bounded by the configured backend cap.
    /// Returning an error leaves both reservations retryable.
    fn raw_alloc<T: BufferElement>(&self, len: usize) -> Result<Self::Buffer<T>, OomError>;

    /// Configure the cap shared by backend-owned allocations.
    ///
    /// Backends with device-side allocation tracking override this hook;
    /// CPU and third-party backends may keep the default unlimited ledger.
    fn set_allocation_cap_bytes(&self, _cap_bytes: usize) {}

    /// Allocate a buffer that may contain uninitialized/undefined values.
    ///
    /// # Safety
    ///
    /// Callers must fully initialize the returned buffer before any operation
    /// can read from it. The default implementation falls back to
    /// [`Self::raw_alloc`], preserving zeroed allocation semantics for
    /// backends that do not provide a cheaper scratch-buffer path.
    unsafe fn raw_alloc_uninitialized<T: BufferElement>(
        &self,
        len: usize,
    ) -> Result<Self::Buffer<T>, OomError> {
        self.raw_alloc::<T>(len)
    }

    /// Total device memory in bytes, used to compute the 80 % default cap.
    ///
    /// CPU returns total host RAM; CUDA returns
    /// `cuMemGetInfo()`'s `total`. Implementations should return `None` only
    /// when discovery is impossible — in that case the pool falls back to
    /// requiring an explicit cap.
    fn total_device_memory_bytes(&self) -> Option<usize>;

    /// Copy host slice into a device buffer (offset 0). Length must match.
    fn copy_h2d<T: BufferElement>(&self, host: &[T], dev: &mut Self::Buffer<T>);

    /// Copy host values into a lower-precision device buffer while casting.
    ///
    /// The default implementation stages a converted host vector so existing
    /// backends keep correct semantics. Backends with host-visible storage can
    /// override this to cast directly into their buffer.
    fn copy_h2d_cast<Upper>(&self, host: &[Upper], dev: &mut Self::Buffer<Upper::Lower>)
    where
        Upper: LowerPrecision,
    {
        if let Err(error) = host_copy_h2d_cast(self, host, dev) {
            log::error!(target: "sonium_compute::backend", "copy_h2d_cast failed: {error}");
        }
    }

    /// Checked host-to-device precision cast.
    fn try_copy_h2d_cast<Upper>(
        &self,
        host: &[Upper],
        dev: &mut Self::Buffer<Upper::Lower>,
    ) -> BackendResult<()>
    where
        Upper: LowerPrecision,
    {
        let capabilities = self.capabilities();
        require_precision_capability::<Upper>(
            capabilities.backend_name,
            "copy_h2d_cast",
            capabilities.precision_cast,
            false,
        )?;
        if host.len() != dev.len() {
            return Err(BackendError::InvalidArgument {
                operation: "copy_h2d_cast",
                reason: format!("host length {} != device length {}", host.len(), dev.len()),
            });
        }
        catch_backend_panic("copy_h2d_cast", || self.copy_h2d_cast(host, dev))?;
        self.check_pending_error()
    }

    /// Copy device buffer back to host slice. Length must match.
    fn copy_d2h<T: BufferElement>(&self, dev: &Self::Buffer<T>, host: &mut [T]);

    /// Device-to-device copy. Lengths must match.
    ///
    /// Used by Krylov solvers to materialise basis-vector copies (e.g.
    /// the new Arnoldi column `V_{k+1}`). On CPU it's a slice copy; on
    /// CUDA/Metal it's a `cudaMemcpyDeviceToDevice` / Metal blit.
    fn copy_d2d<T: BufferElement>(&self, src: &Self::Buffer<T>, dst: &mut Self::Buffer<T>);

    /// Weighted four-point gather:
    /// `out[i] = Σ_{j=0..3} weights[4*i+j] * values[indices[4*i+j]]`.
    ///
    /// This covers fixed-order interpolation workloads, such as P1
    /// tetrahedral receiver sampling, without forcing callers to download the
    /// whole device vector. GPU backends should override this with a device
    /// kernel; the default implementation stages through host memory.
    fn gather4_weighted<T>(
        &self,
        values: &Self::Buffer<T>,
        indices: &Self::Buffer<i32>,
        weights: &Self::Buffer<T::Real>,
        out: &mut Self::Buffer<T>,
    ) where
        T: DeviceScalar,
        T::Real: BufferElement,
    {
        if let Err(error) = host_gather4(self, values, indices, weights, out) {
            log::error!(target: "sonium_compute::backend", "default gather4 fallback rejected request: {error}");
        }
    }

    /// Checked weighted four-point gather with layout validation and panic
    /// containment around legacy/native backend implementations.
    fn try_gather4_weighted<T>(
        &self,
        values: &Self::Buffer<T>,
        indices: &Self::Buffer<i32>,
        weights: &Self::Buffer<T::Real>,
        out: &mut Self::Buffer<T>,
    ) -> BackendResult<()>
    where
        T: DeviceScalar,
        T::Real: BufferElement,
    {
        let capabilities = self.capabilities();
        require_scalar_capability::<T>(
            capabilities.backend_name,
            "gather4_weighted",
            capabilities.gather4_weighted,
        )?;
        if indices.len() != weights.len() || !indices.len().is_multiple_of(4) {
            return Err(BackendError::InvalidArgument {
                operation: "gather4_weighted",
                reason: format!(
                    "indices/weights lengths {}/{} must match and be divisible by four",
                    indices.len(),
                    weights.len()
                ),
            });
        }
        let expected_out = indices.len() / 4;
        if out.len() != expected_out {
            return Err(BackendError::InvalidArgument {
                operation: "gather4_weighted",
                reason: format!("output length {}, expected {expected_out}", out.len()),
            });
        }
        self.validate_gather4_backend_limits::<T>(
            "gather4_weighted",
            values.len(),
            indices.len(),
            out.len(),
        )?;
        validate_gather4_finite_inputs(self, values, weights)?;
        let mut host_indices = try_zeroed_host(indices.len(), "gather4_weighted")?;
        self.try_copy_d2h(indices, &mut host_indices)?;
        for (position, &index) in host_indices.iter().enumerate() {
            if index < 0 || (index as usize) >= values.len() {
                return Err(BackendError::InvalidArgument {
                    operation: "gather4_weighted",
                    reason: format!(
                        "index {index} at position {position} is outside values length {}",
                        values.len()
                    ),
                });
            }
        }
        catch_backend_panic("gather4_weighted", || {
            self.gather4_weighted(values, indices, weights, out)
        })?;
        self.check_pending_error()
    }

    /// Validate dimensions before a checked weighted gather enters a native
    /// kernel. The default has no narrowing constraint.
    fn validate_gather4_backend_limits<T: DeviceScalar>(
        &self,
        _operation: &'static str,
        _values_len: usize,
        _indices_len: usize,
        _output_len: usize,
    ) -> BackendResult<()> {
        Ok(())
    }

    /// Test finiteness without staging the input through host memory.
    ///
    /// Returns `None` when no native implementation exists. A supported
    /// implementation may transfer a compact status word to the host.
    ///
    /// # Errors
    /// Returns checked storage, execution or allocation failures.
    fn try_all_finite_native<T: DeviceScalar>(
        &self,
        _values: &Self::Buffer<T>,
    ) -> BackendResult<Option<bool>> {
        Ok(None)
    }

    /// Zero a buffer natively, including previously non-finite entries.
    ///
    /// Returns false without mutation when unavailable. No host array staging
    /// occurs; multiplying old contents by zero does not satisfy this contract.
    ///
    /// # Errors
    /// Returns checked storage or execution failures.
    fn try_zero_native<T: DeviceScalar>(
        &self,
        _values: &mut Self::Buffer<T>,
    ) -> BackendResult<bool> {
        Ok(false)
    }

    /// Replace a resident matrix action with `rhs - action` in place.
    ///
    /// Returns false without mutation when unavailable. Never stages arrays on
    /// the host. Nonfinite arithmetic follows the scalar subtraction semantics.
    ///
    /// # Errors
    /// Rejects unequal lengths, invalid storage and backend execution failures.
    fn try_residual_update_native<T: DeviceScalar>(
        &self,
        rhs: &Self::Buffer<T>,
        action: &mut Self::Buffer<T>,
    ) -> BackendResult<bool> {
        if rhs.len() != action.len() {
            return Err(BackendError::InvalidArgument {
                operation: "residual_update_native",
                reason: "unequal buffer lengths".into(),
            });
        }
        Ok(false)
    }

    /// Overwrite output with `omega * diagonal * input` on the device.
    ///
    /// Returns false without mutation if unavailable. Never reads old output or
    /// stages arrays on the host, so non-finite scratch contents are overwritten.
    ///
    /// # Errors
    /// Rejects unequal lengths, non-finite damping, or backend execution failures.
    fn try_diagonal_scale_native<T: DeviceScalar>(
        &self,
        diagonal: &Self::Buffer<T>,
        input: &Self::Buffer<T>,
        output: &mut Self::Buffer<T>,
        omega: T::Real,
    ) -> BackendResult<bool> {
        if diagonal.len() != output.len() || input.len() != output.len() || !omega.is_finite() {
            return Err(BackendError::InvalidArgument {
                operation: "diagonal_scale_native",
                reason: "unequal buffer lengths or non-finite damping".into(),
            });
        }
        Ok(false)
    }

    /// Scale column-major vectors by one shared diagonal on the device.
    ///
    /// Each column has `diagonal.len()` entries. Preserves the scalar hook's
    /// overwrite and residency contract; unavailable paths return false unchanged.
    ///
    /// # Errors
    /// Rejects shape overflow, mismatched lengths, non-finite damping or backend failures.
    fn try_diagonal_scale_colmajor_native<T: DeviceScalar>(
        &self,
        rhs: usize,
        diagonal: &Self::Buffer<T>,
        input: &Self::Buffer<T>,
        output: &mut Self::Buffer<T>,
        omega: T::Real,
    ) -> BackendResult<bool> {
        if diagonal.len().checked_mul(rhs) != Some(output.len())
            || input.len() != output.len()
            || !omega.is_finite()
        {
            return Err(BackendError::InvalidArgument {
                operation: "diagonal_scale_colmajor_native",
                reason: "invalid block shape or non-finite damping".into(),
            });
        }
        if rhs == 1 {
            self.try_diagonal_scale_native(diagonal, input, output, omega)
        } else {
            Ok(false)
        }
    }

    /// Try simultaneous Jacobi sweeps on a small resident CSR system.
    ///
    /// Returns false unchanged when unavailable. Never stages arrays on the host.
    /// Malformed resident indices may produce NaNs; callers must certify results.
    ///
    /// # Errors
    /// Rejects inconsistent lengths, nonfinite damping, or backend failures.
    fn try_small_csr_jacobi_native<T: DeviceScalar>(
        &self,
        rows: &Self::Buffer<i32>,
        columns: &Self::Buffer<i32>,
        values: &Self::Buffer<T>,
        inverse: &Self::Buffer<T>,
        b: &Self::Buffer<T>,
        x: &mut Self::Buffer<T>,
        omega: T::Real,
        _sweeps: usize,
    ) -> BackendResult<bool> {
        if x.len().checked_add(1) != Some(rows.len())
            || columns.len() != values.len()
            || inverse.len() != x.len()
            || b.len() != x.len()
            || !omega.is_finite()
        {
            return Err(BackendError::InvalidArgument {
                operation: "small_csr_jacobi_native",
                reason: "inconsistent lengths or nonfinite damping".into(),
            });
        }
        Ok(false)
    }

    /// Try one fused CSR Jacobi sweep into a separate output buffer.
    ///
    /// Computes `output = x + omega * inverse * (b - A*x)`. Returns false
    /// unchanged when unavailable. Inputs remain immutable for the entire sweep.
    /// Malformed resident indices may produce NaNs; certify results separately.
    /// No allocation or host staging is permitted.
    ///
    /// # Errors
    /// Rejects inconsistent lengths, nonfinite damping, or native failures.
    fn try_csr_jacobi_step_native<T: DeviceScalar>(
        &self,
        rows: &Self::Buffer<i32>,
        columns: &Self::Buffer<i32>,
        values: &Self::Buffer<T>,
        inverse: &Self::Buffer<T>,
        b: &Self::Buffer<T>,
        x: &Self::Buffer<T>,
        output: &mut Self::Buffer<T>,
        omega: T::Real,
    ) -> BackendResult<bool> {
        if x.len().checked_add(1) != Some(rows.len())
            || columns.len() != values.len() || inverse.len() != x.len()
            || b.len() != x.len() || output.len() != x.len() || !omega.is_finite()
        {
            return Err(BackendError::InvalidArgument {
                operation: "csr_jacobi_step_native",
                reason: "inconsistent lengths or nonfinite damping".into(),
            });
        }
        Ok(false)
    }

    /// Try a native fused update: x += omega * inverse * (b - ax).
    ///
    /// Returns false without modifying output when no native implementation exists.
    /// Callers can then use ordinary backend operations. This method never stages
    /// arrays through host memory. A failed launch returns an error, not false.
    ///
    /// # Errors
    /// Rejects unequal lengths, non-finite damping, or backend execution failures.
    fn try_jacobi_update_native<T: DeviceScalar>(
        &self,
        inverse: &Self::Buffer<T>,
        b: &Self::Buffer<T>,
        ax: &Self::Buffer<T>,
        x: &mut Self::Buffer<T>,
        omega: T::Real,
    ) -> BackendResult<bool> {
        if inverse.len() != x.len()
            || b.len() != x.len()
            || ax.len() != x.len()
            || !omega.is_finite()
        {
            return Err(BackendError::InvalidArgument {
                operation: "jacobi_update_native",
                reason: "unequal buffer lengths or non-finite damping".into(),
            });
        }
        Ok(false)
    }

    /// Update a Chebyshev direction and solution together on the device.
    ///
    /// Computes `direction = beta * direction + alpha * inverse * (b - ax)`,
    /// then `x += direction`, componentwise. When beta is zero, old direction
    /// contents are not read, including non-finite scratch. Returns false without
    /// mutation if unavailable; never allocates scratch or stages host arrays.
    ///
    /// # Errors
    /// Rejects unequal lengths, non-finite coefficients, or backend execution failures.
    fn try_chebyshev_update_native<T: DeviceScalar>(
        &self,
        inverse: &Self::Buffer<T>,
        b: &Self::Buffer<T>,
        ax: &Self::Buffer<T>,
        direction: &mut Self::Buffer<T>,
        x: &mut Self::Buffer<T>,
        alpha: T::Real,
        beta: T::Real,
    ) -> BackendResult<bool> {
        if inverse.len() != x.len()
            || b.len() != x.len()
            || ax.len() != x.len()
            || direction.len() != x.len()
            || !alpha.is_finite()
            || !beta.is_finite()
        {
            return Err(BackendError::InvalidArgument {
                operation: "chebyshev_update_native",
                reason: "unequal buffer lengths or non-finite coefficients".into(),
            });
        }
        Ok(false)
    }

    /// Update column-major Chebyshev directions and solutions with one shared diagonal.
    ///
    /// Each column has `inverse.len()` entries. Uses the scalar hook's recurrence,
    /// including ignoring old direction contents when beta is zero. Unavailable
    /// paths return false without mutation, allocation or host array staging.
    ///
    /// # Errors
    /// Rejects shape overflow, mismatched lengths, non-finite coefficients or backend failures.
    fn try_chebyshev_update_colmajor_native<T: DeviceScalar>(
        &self,
        rhs: usize,
        inverse: &Self::Buffer<T>,
        b: &Self::Buffer<T>,
        ax: &Self::Buffer<T>,
        direction: &mut Self::Buffer<T>,
        x: &mut Self::Buffer<T>,
        alpha: T::Real,
        beta: T::Real,
    ) -> BackendResult<bool> {
        if inverse.len().checked_mul(rhs) != Some(x.len())
            || b.len() != x.len()
            || ax.len() != x.len()
            || direction.len() != x.len()
            || !alpha.is_finite()
            || !beta.is_finite()
        {
            return Err(BackendError::InvalidArgument {
                operation: "chebyshev_update_colmajor_native",
                reason: "invalid block shape or non-finite coefficients".into(),
            });
        }
        if rhs == 1 {
            self.try_chebyshev_update_native(inverse, b, ax, direction, x, alpha, beta)
        } else {
            Ok(false)
        }
    }

    /// Convert an upper-precision device buffer into its lower-precision
    /// partner.
    ///
    /// The default implementation stages through host memory so every
    /// backend gets correct semantics immediately. GPU backends should
    /// override this with a device kernel to keep iterative refinement data
    /// resident on the accelerator.
    /// One Gauss-Seidel sweep over a single color subset: for each row `i` in
    /// `color_rows`, `x[i] = (1-omega)*x[i] + omega*(b[i] - Σ_{j≠i} a_ij*x[j]) / a_ii`.
    ///
    /// Rows within one color share no edges, so the updates are independent
    /// and may run in parallel; colors must be swept sequentially (forward
    /// then reverse for the symmetric variant). Matches the host
    /// `smooth_multicolor_gauss_seidel` semantics exactly, including the
    /// `|a_ii| <= 1e-15` skip and the missing-diagonal default of 1.
    ///
    /// The default implementation stages through host memory, preserving the
    /// overwrite contract on every backend. Native backends can override
    /// this with a device kernel without changing callers.
    #[allow(clippy::too_many_arguments)]
    fn try_gs_color_sweep<T: DeviceScalar>(
        &self,
        row_ptr: &Self::Buffer<i32>,
        col_idx: &Self::Buffer<i32>,
        values: &Self::Buffer<T>,
        color_rows: &Self::Buffer<i32>,
        omega: T::Real,
        b: &Self::Buffer<T>,
        x: &mut Self::Buffer<T>,
    ) -> BackendResult<()> {
        let capabilities = self.capabilities();
        require_scalar_capability::<T>(
            capabilities.backend_name,
            "gs_color_sweep",
            capabilities.csr_spmv,
        )?;
        let n = x.len();
        if b.len() != n {
            return Err(BackendError::InvalidArgument {
                operation: "gs_color_sweep",
                reason: format!("b length {} != x length {n}", b.len()),
            });
        }
        if row_ptr.len() != n + 1 {
            return Err(BackendError::InvalidArgument {
                operation: "gs_color_sweep",
                reason: format!(
                    "row_ptr length {} != x length + 1 ({})",
                    row_ptr.len(),
                    n + 1
                ),
            });
        }
        let mut host_row_ptr = try_zeroed_host(row_ptr.len(), "gs_color_sweep")?;
        let mut host_col_idx = try_zeroed_host(col_idx.len(), "gs_color_sweep")?;
        let mut host_values = try_zeroed_host(values.len(), "gs_color_sweep")?;
        let mut host_colors = try_zeroed_host(color_rows.len(), "gs_color_sweep")?;
        let mut host_b = try_zeroed_host(b.len(), "gs_color_sweep")?;
        let mut host_x = try_zeroed_host(x.len(), "gs_color_sweep")?;
        catch_backend_panic("gs_color_sweep copy_d2h", || {
            self.copy_d2h(row_ptr, &mut host_row_ptr);
            self.copy_d2h(col_idx, &mut host_col_idx);
            self.copy_d2h(values, &mut host_values);
            self.copy_d2h(color_rows, &mut host_colors);
            self.copy_d2h(b, &mut host_b);
            self.copy_d2h(x, &mut host_x);
        })?;
        gs_color_sweep_host(
            &host_row_ptr,
            &host_col_idx,
            &host_values,
            &host_colors,
            omega,
            &host_b,
            &mut host_x,
        )?;
        catch_backend_panic("gs_color_sweep copy_h2d", || {
            self.copy_h2d(&host_x, x);
        })?;
        self.check_pending_error()
    }

    /// Checked segmented sum over device values using an offset buffer.
    ///
    /// The default implementation stages through host memory, preserving the
    /// fixed-order and overwrite contract on every backend. Native backends
    /// can override this with a device reduction without changing callers.
    fn try_segmented_sum<T>(
        &self,
        values: &Self::Buffer<T>,
        offsets: &Self::Buffer<i32>,
        output: &mut Self::Buffer<T>,
    ) -> BackendResult<()>
    where
        T: DeviceScalar,
    {
        let capabilities = self.capabilities();
        require_scalar_capability::<T>(
            capabilities.backend_name,
            "segmented_sum",
            capabilities.segmented_reduce,
        )?;
        let expected_offsets =
            output
                .len()
                .checked_add(1)
                .ok_or_else(|| BackendError::InvalidArgument {
                    operation: "segmented_sum",
                    reason: "output length overflows offset count".to_string(),
                })?;
        if offsets.len() != expected_offsets {
            return Err(BackendError::InvalidArgument {
                operation: "segmented_sum",
                reason: format!(
                    "offset length {} != output length + 1 ({expected_offsets})",
                    offsets.len()
                ),
            });
        }
        self.validate_segmented_backend_limits::<T>(
            "segmented_sum",
            values.len(),
            offsets.len(),
            output.len(),
        )?;

        let mut host_values = try_zeroed_host(values.len(), "segmented_sum")?;
        let mut host_offsets = try_zeroed_host(offsets.len(), "segmented_sum")?;
        let mut host_output = try_zeroed_host(output.len(), "segmented_sum")?;
        catch_backend_panic("segmented_sum copy_d2h", || {
            self.copy_d2h(values, &mut host_values);
            self.copy_d2h(offsets, &mut host_offsets);
        })?;
        segmented_sum(&host_values, &host_offsets, &mut host_output).map_err(|error| {
            BackendError::InvalidArgument {
                operation: "segmented_sum",
                reason: error.to_string(),
            }
        })?;
        catch_backend_panic("segmented_sum copy_h2d", || {
            self.copy_h2d(&host_output, output);
        })?;
        self.check_pending_error()
    }

    /// Checked segmented sum with an explicit reduction policy.
    ///
    /// `Fast` delegates to the backend's native/fallback implementation.
    /// `Deterministic` performs one checked download, validates all numeric
    /// input, accumulates every segment in fixed logical order, validates the
    /// generated values, and uploads only after the complete result succeeds.
    fn try_segmented_sum_with_mode<T>(
        &self,
        mode: ReductionMode,
        values: &Self::Buffer<T>,
        offsets: &Self::Buffer<i32>,
        output: &mut Self::Buffer<T>,
    ) -> BackendResult<()>
    where
        T: DeviceScalar,
    {
        self.check_pending_error()?;
        let capabilities = self.capabilities();
        require_scalar_capability::<T>(
            capabilities.backend_name,
            "segmented_sum",
            capabilities.segmented_reduce,
        )?;
        require_reduction_capability::<T>(
            capabilities.backend_name,
            "segmented_sum",
            capabilities.reduction,
            mode,
        )?;
        if mode == ReductionMode::Fast {
            self.try_segmented_sum(values, offsets, output)?;
            return self.check_pending_error();
        }

        let expected_offsets =
            output
                .len()
                .checked_add(1)
                .ok_or_else(|| BackendError::InvalidArgument {
                    operation: "segmented_sum",
                    reason: "output length overflows offset count".into(),
                })?;
        if offsets.len() != expected_offsets {
            return Err(BackendError::InvalidArgument {
                operation: "segmented_sum",
                reason: format!(
                    "offset length {} != output length + 1 ({expected_offsets})",
                    offsets.len()
                ),
            });
        }

        let mut host_values = try_zeroed_host(values.len(), "segmented_sum")?;
        let mut host_offsets = try_zeroed_host(offsets.len(), "segmented_sum")?;
        self.try_copy_d2h(values, &mut host_values)?;
        self.try_copy_d2h(offsets, &mut host_offsets)?;
        validate_segmented_offsets(values.len(), &host_offsets, output.len()).map_err(|error| {
            BackendError::InvalidArgument {
                operation: "segmented_sum",
                reason: error.to_string(),
            }
        })?;
        if host_values.iter().any(|value| !value.is_finite()) {
            return Err(BackendError::OperationFailed {
                operation: "segmented_sum",
                reason: "reduction input contains a non-finite value".into(),
            });
        }

        let mut host_output = try_zeroed_host(output.len(), "segmented_sum")?;
        segmented_sum(&host_values, &host_offsets, &mut host_output).map_err(|error| {
            BackendError::InvalidArgument {
                operation: "segmented_sum",
                reason: error.to_string(),
            }
        })?;
        if host_output.iter().any(|value| !value.is_finite()) {
            return Err(BackendError::OperationFailed {
                operation: "segmented_sum",
                reason: "reduction produced a non-finite value".into(),
            });
        }
        self.try_copy_h2d(&host_output, output)?;
        self.check_pending_error()
    }

    /// Validate dimensions before a checked segmented reduction enters a
    /// native backend kernel. The default has no narrowing constraint.
    fn validate_segmented_backend_limits<T: DeviceScalar>(
        &self,
        _operation: &'static str,
        _values_len: usize,
        _offsets_len: usize,
        _output_len: usize,
    ) -> BackendResult<()> {
        Ok(())
    }

    /// Checked batched local contraction over contiguous row-major tensors.
    ///
    /// The default implementation stages through host memory, preserving the
    /// fixed accumulation order and overwrite semantics on every backend.
    /// Native backends can replace this with a batched GEMM or tensor kernel.
    #[allow(clippy::too_many_arguments)]
    fn try_batched_local_contract<T>(
        &self,
        lhs: &Self::Buffer<T>,
        rhs: &Self::Buffer<T>,
        output: &mut Self::Buffer<T>,
        batches: usize,
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> BackendResult<()>
    where
        T: DeviceScalar,
    {
        let capabilities = self.capabilities();
        require_scalar_capability::<T>(
            capabilities.backend_name,
            "batched_local_contract",
            capabilities.batched_local_contract,
        )?;
        let mut host_lhs = try_zeroed_host(lhs.len(), "batched_local_contract")?;
        let mut host_rhs = try_zeroed_host(rhs.len(), "batched_local_contract")?;
        let mut host_output = try_zeroed_host(output.len(), "batched_local_contract")?;
        catch_backend_panic("batched_local_contract copy_d2h", || {
            self.copy_d2h(lhs, &mut host_lhs);
            self.copy_d2h(rhs, &mut host_rhs);
        })?;
        batched_local_contract(
            &host_lhs,
            &host_rhs,
            &mut host_output,
            batches,
            rows,
            inner,
            cols,
        )
        .map_err(|error| BackendError::InvalidArgument {
            operation: "batched_local_contract",
            reason: error.to_string(),
        })?;
        catch_backend_panic("batched_local_contract copy_h2d", || {
            self.copy_h2d(&host_output, output);
        })?;
        self.check_pending_error()
    }

    fn cast_to_lower<Upper>(&self, src: &Self::Buffer<Upper>, dst: &mut Self::Buffer<Upper::Lower>)
    where
        Upper: LowerPrecision,
    {
        if let Err(error) = self.try_cast_to_lower(src, dst) {
            log::error!(target: "sonium_compute::backend", "cast_to_lower failed: {error}");
        }
    }

    /// Checked upper-to-lower device cast.
    fn try_cast_to_lower<Upper>(
        &self,
        src: &Self::Buffer<Upper>,
        dst: &mut Self::Buffer<Upper::Lower>,
    ) -> BackendResult<()>
    where
        Upper: LowerPrecision,
    {
        let capabilities = self.capabilities();
        require_precision_capability::<Upper>(
            capabilities.backend_name,
            "cast_to_lower",
            capabilities.precision_cast,
            false,
        )?;
        if src.len() != dst.len() {
            return Err(BackendError::InvalidArgument {
                operation: "cast_to_lower",
                reason: format!(
                    "source length {} != destination length {}",
                    src.len(),
                    dst.len()
                ),
            });
        }
        host_cast_to_lower(self, src, dst)
    }

    /// Convert a lower-precision device buffer back into its upper-precision
    /// partner.
    ///
    /// See [`Self::cast_to_lower`] for the backend contract.
    fn cast_from_lower<Upper>(
        &self,
        src: &Self::Buffer<Upper::Lower>,
        dst: &mut Self::Buffer<Upper>,
    ) where
        Upper: LowerPrecision,
    {
        if let Err(error) = self.try_cast_from_lower(src, dst) {
            log::error!(target: "sonium_compute::backend", "cast_from_lower failed: {error}");
        }
    }

    /// Checked lower-to-upper device cast.
    fn try_cast_from_lower<Upper>(
        &self,
        src: &Self::Buffer<Upper::Lower>,
        dst: &mut Self::Buffer<Upper>,
    ) -> BackendResult<()>
    where
        Upper: LowerPrecision,
    {
        let capabilities = self.capabilities();
        require_precision_capability::<Upper>(
            capabilities.backend_name,
            "cast_from_lower",
            capabilities.precision_cast,
            true,
        )?;
        if src.len() != dst.len() {
            return Err(BackendError::InvalidArgument {
                operation: "cast_from_lower",
                reason: format!(
                    "source length {} != destination length {}",
                    src.len(),
                    dst.len()
                ),
            });
        }
        host_cast_from_lower(self, src, dst)
    }

    /// Block until all enqueued work on this backend has finished.
    ///
    /// CPU is synchronous and returns immediately; CUDA waits on the default
    /// stream. Device and command-buffer failures are returned to the caller.
    fn synchronize(&self) -> BackendResult<()>;

    /// Fallible synchronization boundary for asynchronous backends.
    fn try_synchronize(&self) -> BackendResult<()> {
        self.try_inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::Synchronization,
        )?;
        catch_backend_panic("synchronize", || self.synchronize())??;
        Ok(())
    }

    /// Check an asynchronous backend's sticky submission-error state without
    /// waiting for all currently queued work. Backends with deferred command
    /// buffers override this so checked operations cannot turn an earlier
    /// encoder/command-buffer failure into an `Ok` neutral result.
    fn check_pending_error(&self) -> BackendResult<()> {
        Ok(())
    }

    /// Consume an explicitly armed one-shot preconditioner fault.
    ///
    /// This is intentionally a no-op for ordinary backends.  CUDA and the
    /// CPU fault harness override it so checked preconditioner setup/action
    /// tests can cover the same failure provenance without changing the
    /// backend's normal execution path.
    fn try_inject_preconditioner_failure(
        &self,
        _point: PreconditionerFailureInjectionPoint,
    ) -> BackendResult<()> {
        Ok(())
    }

    /// Submit a backend-native completion fence without waiting for it.
    ///
    /// CPU and legacy backends use the checked synchronization boundary as a
    /// conservative fallback and return an already-completed token. CUDA and
    /// Metal override this to record/commit a native fence while leaving the
    /// caller free to overlap other host work before [`BackendCompletion::wait`].
    fn try_submit_completion(&self) -> BackendResult<Box<dyn BackendCompletion>> {
        self.try_synchronize()?;
        Ok(Box::new(ImmediateBackendCompletion))
    }

    // ----- BLAS-1 -----

    /// `y <- alpha * x + y`.
    fn axpy<T: DeviceScalar>(&self, alpha: T, x: &Self::Buffer<T>, y: &mut Self::Buffer<T>);

    /// Checked AXPY dispatch. Backends with native status-returning BLAS
    /// should override this so allocation, pointer, and execution failures
    /// remain distinguishable from a legacy panic.
    fn try_axpy<T: DeviceScalar>(
        &self,
        alpha: T,
        x: &Self::Buffer<T>,
        y: &mut Self::Buffer<T>,
    ) -> BackendResult<()> {
        let capabilities = self.capabilities();
        require_scalar_capability::<T>(capabilities.backend_name, "axpy", capabilities.blas1)?;
        if x.len() != y.len() {
            return Err(BackendError::InvalidArgument {
                operation: "axpy",
                reason: format!("vector lengths {} and {} differ", x.len(), y.len()),
            });
        }
        require_finite_scalar("axpy", "alpha", alpha)?;
        self.validate_vector_backend_limits::<T>("axpy", x.len())?;
        catch_backend_panic("axpy", || self.axpy(alpha, x, y))?;
        self.check_pending_error()
    }

    /// Optionally compute `output = alpha*x + beta*y` in one native pass.
    ///
    /// Returns false without modifying output when unsupported. Inputs and output
    /// must have equal lengths and coefficients must be finite. No allocation or
    /// host transfer is permitted. Callers may use copy/scale/AXPY after false.
    ///
    /// # Errors
    /// Reports invalid arguments and native failures. Output is unspecified after
    /// an execution failure; callers must invalidate any dependent solve state.
    fn try_linear_combination2_native<T: DeviceScalar>(
        &self,
        _alpha: T,
        _x: &Self::Buffer<T>,
        _beta: T,
        _y: &Self::Buffer<T>,
        _output: &mut Self::Buffer<T>,
    ) -> BackendResult<bool> {
        Ok(false)
    }

    /// Validate a vector length before a checked BLAS-1 or reduction call
    /// enters a backend kernel. Most backends use `usize` dimensions; native
    /// accelerator kernels may narrow lengths to a device scalar width.
    fn validate_vector_backend_limits<T: DeviceScalar>(
        &self,
        _operation: &'static str,
        _vector_len: usize,
    ) -> BackendResult<()> {
        Ok(())
    }

    /// `y <- y + Σ_i alpha[i] * x_basis[i]`.
    ///
    /// Default impl preserves semantics by calling [`Self::axpy`] once per
    /// basis vector. Backends can override to fuse the update into one pass
    /// or one device dispatch, which is the solution-update hot path for
    /// GMRES/FGMRES after the small Hessenberg solve.
    fn axpy_many<T: DeviceScalar>(
        &self,
        alphas: &[T],
        x_basis: &[&Self::Buffer<T>],
        y: &mut Self::Buffer<T>,
    ) {
        assert_eq!(
            alphas.len(),
            x_basis.len(),
            "axpy_many coefficient/basis length mismatch"
        );
        for (&alpha, x) in alphas.iter().zip(x_basis.iter()) {
            self.axpy(alpha, *x, y);
        }
    }

    /// Checked fused AXPY update. Validates the coefficient/basis contract and
    /// contains legacy backend panics before dispatching the compatibility
    /// implementation.
    fn validate_axpy_many_backend_limits<T: DeviceScalar>(
        &self,
        _operation: &'static str,
        _vector_len: usize,
        _basis_count: usize,
    ) -> BackendResult<()> {
        Ok(())
    }

    fn try_axpy_many<T: DeviceScalar>(
        &self,
        alphas: &[T],
        x_basis: &[&Self::Buffer<T>],
        y: &mut Self::Buffer<T>,
    ) -> BackendResult<()> {
        let capabilities = self.capabilities();
        require_scalar_capability::<T>(
            capabilities.backend_name,
            "axpy_many",
            capabilities.axpy_many,
        )?;
        if alphas.len() != x_basis.len() {
            return Err(BackendError::InvalidArgument {
                operation: "axpy_many",
                reason: format!(
                    "coefficient count {} differs from basis count {}",
                    alphas.len(),
                    x_basis.len()
                ),
            });
        }
        if let Some((index, x)) = x_basis.iter().enumerate().find(|(_, x)| x.len() != y.len()) {
            return Err(BackendError::InvalidArgument {
                operation: "axpy_many",
                reason: format!(
                    "basis vector {index} has length {}, expected {}",
                    x.len(),
                    y.len()
                ),
            });
        }
        for &alpha in alphas {
            require_finite_scalar("axpy_many", "alpha", alpha)?;
        }
        self.validate_axpy_many_backend_limits::<T>("axpy_many", y.len(), x_basis.len())?;
        // The checked fallback deliberately applies one checked AXPY at a
        // time. This avoids the legacy wrapper's assertion and any backend
        // implementation's temporary reference-vector allocation; native
        // CUDA/Metal implementations override this method with fused paths.
        for (&alpha, x) in alphas.iter().zip(x_basis.iter().copied()) {
            self.try_axpy(alpha, x, y)?;
        }
        self.check_pending_error()
    }

    /// Checked fused AXPY update whose coefficients already live in a device
    /// buffer. Native backends can consume the coefficient buffer directly;
    /// the default preserves the contract with one explicitly staged copy.
    fn try_axpy_many_device<T: DeviceScalar>(
        &self,
        alphas: &Self::Buffer<T>,
        x_basis: &[&Self::Buffer<T>],
        y: &mut Self::Buffer<T>,
    ) -> BackendResult<()> {
        if alphas.len() < x_basis.len() {
            return Err(BackendError::InvalidArgument {
                operation: "axpy_many_device",
                reason: format!(
                    "coefficient count {} is smaller than basis count {}",
                    alphas.len(),
                    x_basis.len()
                ),
            });
        }
        let mut host_alphas = try_zeroed_host::<T>(alphas.len(), "axpy_many_device")?;
        self.try_copy_d2h(alphas, &mut host_alphas)?;
        self.try_axpy_many(&host_alphas[..x_basis.len()], x_basis, y)
    }

    /// `Σ conj(x_i) * y_i` — Hermitian inner product. For real types, the
    /// conjugate is a no-op.
    fn dot<T: DeviceScalar>(&self, x: &Self::Buffer<T>, y: &Self::Buffer<T>) -> T;

    /// Dot product with an explicit reproducibility policy.
    fn dot_with_mode<T: DeviceScalar>(
        &self,
        mode: ReductionMode,
        x: &Self::Buffer<T>,
        y: &Self::Buffer<T>,
    ) -> T {
        match mode {
            ReductionMode::Fast => self.dot(x, y),
            ReductionMode::Reproducible | ReductionMode::Deterministic => {
                self.dot_deterministic(x, y)
            }
        }
    }

    /// Fixed-order Hermitian dot product. One host transfer per input is
    /// intentional: deterministic mode is for gradient checks and replayable
    /// QA, not the steady-state Krylov path.
    fn dot_deterministic<T: DeviceScalar>(&self, x: &Self::Buffer<T>, y: &Self::Buffer<T>) -> T {
        self.try_dot_deterministic(x, y).unwrap_or_else(|error| {
            log::error!(target: "sonium_compute::backend", "deterministic dot rejected: {error}");
            T::zero()
        })
    }

    /// Fallible fixed-order Hermitian dot product. Unlike the legacy
    /// `dot_deterministic` entry point this never truncates mismatched
    /// vectors and propagates transfer failures to the caller.
    fn try_dot_deterministic<T: DeviceScalar>(
        &self,
        x: &Self::Buffer<T>,
        y: &Self::Buffer<T>,
    ) -> BackendResult<T> {
        let capabilities = self.capabilities();
        require_reduction_capability::<T>(
            capabilities.backend_name,
            "dot_deterministic",
            capabilities.reduction,
            ReductionMode::Deterministic,
        )?;
        if x.len() != y.len() {
            return Err(BackendError::InvalidArgument {
                operation: "dot_deterministic",
                reason: format!("vector lengths {} and {} differ", x.len(), y.len()),
            });
        }
        if x.len() == 0 {
            return Ok(T::zero());
        }
        let mut x_host = try_zeroed_host(x.len(), "dot_deterministic")?;
        let mut y_host = try_zeroed_host(y.len(), "dot_deterministic")?;
        self.try_copy_d2h(x, &mut x_host)?;
        self.try_copy_d2h(y, &mut y_host)?;
        if x_host
            .iter()
            .chain(y_host.iter())
            .any(|value| !value.is_finite())
        {
            return Err(BackendError::OperationFailed {
                operation: "dot_deterministic",
                reason: "reduction input contains a non-finite value".into(),
            });
        }
        let result = x_host
            .into_iter()
            .zip(y_host)
            .fold(T::zero(), |acc, (x_i, y_i)| acc + x_i.conj_mul(y_i));
        if !result.is_finite() {
            return Err(BackendError::OperationFailed {
                operation: "dot_deterministic",
                reason: "reduction result is non-finite".into(),
            });
        }
        Ok(result)
    }

    /// Fallible reduction dispatch with explicit reproducibility semantics.
    fn try_dot_with_mode<T: DeviceScalar>(
        &self,
        mode: ReductionMode,
        x: &Self::Buffer<T>,
        y: &Self::Buffer<T>,
    ) -> BackendResult<T> {
        let capabilities = self.capabilities();
        require_reduction_capability::<T>(
            capabilities.backend_name,
            "dot",
            capabilities.reduction,
            mode,
        )?;
        if x.len() != y.len() {
            return Err(BackendError::InvalidArgument {
                operation: "dot",
                reason: format!("vector lengths {} and {} differ", x.len(), y.len()),
            });
        }
        if matches!(mode, ReductionMode::Fast) {
            self.validate_vector_backend_limits::<T>("dot", x.len())?;
        }
        match mode {
            ReductionMode::Fast => {
                let result = catch_backend_panic("dot", || self.dot(x, y))?;
                self.check_pending_error()?;
                require_finite_scalar("dot", "result", result)?;
                Ok(result)
            }
            ReductionMode::Reproducible | ReductionMode::Deterministic => {
                self.try_dot_deterministic(x, y)
            }
        }
    }

    /// Multiple dot products against a shared right operand:
    /// `result[i] = ⟨x_basis[i], y⟩` for `i in 0..x_basis.len()`.
    ///
    /// Default impl calls [`Self::dot`] in a loop; backends can fuse
    /// into a single dispatch + single host sync, which is the
    /// performance-critical path for Modified Gram-Schmidt where
    /// `j+1` independent dots block the GPU pipeline `j+1` times.
    fn multi_dot<T: DeviceScalar>(
        &self,
        x_basis: &[&Self::Buffer<T>],
        y: &Self::Buffer<T>,
    ) -> Vec<T> {
        let mut result = match try_zeroed_host::<T>(x_basis.len(), "multi_dot") {
            Ok(result) => result,
            Err(error) => {
                log::error!(
                    target: "sonium_compute::backend",
                    "multi-dot compatibility result allocation failed: {error}"
                );
                return Vec::new();
            }
        };
        for (slot, x) in result.iter_mut().zip(x_basis.iter().copied()) {
            match self.try_dot_with_mode(ReductionMode::Fast, x, y) {
                Ok(value) => *slot = value,
                Err(error) => {
                    log::error!(
                        target: "sonium_compute::backend",
                        "multi-dot compatibility reduction failed: {error}"
                    );
                    return Vec::new();
                }
            }
        }
        result
    }

    /// Multiple dot products with an explicit reproducibility policy.
    fn multi_dot_with_mode<T: DeviceScalar>(
        &self,
        mode: ReductionMode,
        x_basis: &[&Self::Buffer<T>],
        y: &Self::Buffer<T>,
    ) -> Vec<T> {
        match mode {
            ReductionMode::Fast => self.multi_dot(x_basis, y),
            ReductionMode::Reproducible | ReductionMode::Deterministic => {
                let mut result = match try_zeroed_host::<T>(x_basis.len(), "multi_dot") {
                    Ok(result) => result,
                    Err(error) => {
                        log::error!(
                            target: "sonium_compute::backend",
                            "multi-dot compatibility result allocation failed: {error}"
                        );
                        return Vec::new();
                    }
                };
                for (slot, x) in result.iter_mut().zip(x_basis.iter().copied()) {
                    match self.try_dot_deterministic(x, y) {
                        Ok(value) => *slot = value,
                        Err(error) => {
                            log::error!(
                                target: "sonium_compute::backend",
                                "deterministic multi-dot compatibility reduction failed: {error}"
                            );
                            return Vec::new();
                        }
                    }
                }
                result
            }
        }
    }

    /// Fallible multi-dot reduction with explicit shape and reproducibility
    /// validation.
    fn try_multi_dot_with_mode<T: DeviceScalar>(
        &self,
        mode: ReductionMode,
        x_basis: &[&Self::Buffer<T>],
        y: &Self::Buffer<T>,
    ) -> BackendResult<Vec<T>> {
        let capabilities = self.capabilities();
        require_scalar_capability::<T>(
            capabilities.backend_name,
            "multi_dot",
            capabilities.multi_dot,
        )?;
        require_reduction_capability::<T>(
            capabilities.backend_name,
            "multi_dot",
            capabilities.reduction,
            mode,
        )?;
        if let Some((index, x)) = x_basis.iter().enumerate().find(|(_, x)| x.len() != y.len()) {
            return Err(BackendError::InvalidArgument {
                operation: "multi_dot",
                reason: format!(
                    "basis vector {index} has length {}, expected {}",
                    x.len(),
                    y.len()
                ),
            });
        }
        if matches!(mode, ReductionMode::Fast) {
            self.validate_multi_dot_backend_limits::<T>("multi_dot", y.len(), x_basis.len())?;
        }
        match mode {
            ReductionMode::Fast => {
                let mut result = try_zeroed_host(x_basis.len(), "multi_dot")?;
                self.try_multi_dot_with_mode_into(mode, x_basis, y, &mut result)?;
                self.check_pending_error()?;
                if result.iter().any(|value| !value.is_finite()) {
                    return Err(BackendError::OperationFailed {
                        operation: "multi_dot",
                        reason: "reduction result is non-finite".into(),
                    });
                }
                Ok(result)
            }
            ReductionMode::Reproducible | ReductionMode::Deterministic => Ok(try_collect_backend(
                x_basis.iter().map(|x| self.try_dot_deterministic(x, y)),
                x_basis.len(),
                "multi_dot",
            )?),
        }
    }

    /// Fallible multi-dot reduction writing into caller-owned host storage.
    ///
    /// Native implementations can fill the caller's buffer directly, avoiding
    /// a result allocation in Krylov loops.  The default implementation uses a
    /// fallible temporary result so its scalar checked-dot fallback can commit
    /// the caller's buffer atomically after every basis vector succeeds.
    /// Backends with a native fused reduction can override this method while
    /// preserving the same shape, finiteness, and reduction-policy checks.
    fn try_multi_dot_with_mode_into<T: DeviceScalar>(
        &self,
        mode: ReductionMode,
        x_basis: &[&Self::Buffer<T>],
        y: &Self::Buffer<T>,
        out: &mut [T],
    ) -> BackendResult<()> {
        if out.len() != x_basis.len() {
            return Err(BackendError::InvalidArgument {
                operation: "multi_dot_into",
                reason: format!(
                    "output length {} must match basis count {}",
                    out.len(),
                    x_basis.len()
                ),
            });
        }
        // Keep the caller-owned output transactional.  A later basis vector
        // can fail shape, capability, or backend validation after an earlier
        // dot has already completed; writing directly into `out` would leave
        // a partially updated result that callers could mistake for valid.
        let mut results = try_zeroed_host::<T>(x_basis.len(), "multi_dot_into")?;
        for (slot, basis) in results.iter_mut().zip(x_basis.iter().copied()) {
            *slot = self.try_dot_with_mode(mode, basis, y)?;
        }
        out.copy_from_slice(&results);
        Ok(())
    }

    /// Checked fused multi-dot reduction writing to a device buffer. This is
    /// the residency boundary used by device-resident Krylov implementations.
    /// Backends without a native implementation retain correctness with an
    /// explicitly staged host result; native backends must override it when
    /// their capability table advertises device-resident reductions.
    fn try_multi_dot_with_mode_device_into<T: DeviceScalar>(
        &self,
        mode: ReductionMode,
        x_basis: &[&Self::Buffer<T>],
        y: &Self::Buffer<T>,
        out: &mut Self::Buffer<T>,
    ) -> BackendResult<()> {
        if out.len() < x_basis.len() {
            return Err(BackendError::InvalidArgument {
                operation: "multi_dot_device_into",
                reason: format!(
                    "output length {} is smaller than basis count {}",
                    out.len(),
                    x_basis.len()
                ),
            });
        }
        let mut host = try_zeroed_host::<T>(out.len(), "multi_dot_device_into")?;
        self.try_multi_dot_with_mode_into(mode, x_basis, y, &mut host[..x_basis.len()])?;
        self.try_copy_h2d(&host, out)
    }

    /// Validate dimensions before a checked fused multi-dot enters a native
    /// backend kernel. The default has no narrowing constraint.
    fn validate_multi_dot_backend_limits<T: DeviceScalar>(
        &self,
        _operation: &'static str,
        _vector_len: usize,
        _basis_count: usize,
    ) -> BackendResult<()> {
        Ok(())
    }

    /// Multiple dot products against a flat, strided Krylov basis:
    /// `result[j] = ⟨basis[j], y⟩`.
    ///
    /// The `basis` buffer stores `basis_count` contiguous basis vectors, each
    /// of length `vector_len`, i.e. `basis[j * vector_len + i]`.
    /// This avoids building an array of device-buffer references for solvers
    /// whose basis already lives in one flat allocation.
    fn multi_dot_strided<T: DeviceScalar>(
        &self,
        basis: &Self::Buffer<T>,
        vector_len: usize,
        basis_count: usize,
        y: &Self::Buffer<T>,
    ) -> Vec<T> {
        if y.len() != vector_len {
            log::error!(
                target: "sonium_compute::backend",
                "multi_dot_strided rejected y length {} (expected {vector_len})",
                y.len()
            );
            return Vec::new();
        }
        let Some(required) = vector_len.checked_mul(basis_count) else {
            log::error!(
                target: "sonium_compute::backend",
                "multi_dot_strided basis shape {basis_count}x{vector_len} overflows"
            );
            return Vec::new();
        };
        if basis.len() < required {
            log::error!(
                target: "sonium_compute::backend",
                "multi_dot_strided rejected basis length {} (expected at least {required})",
                basis.len()
            );
            return Vec::new();
        }
        if basis_count == 0 {
            return Vec::new();
        }
        let mut basis_host = match try_zeroed_host::<T>(basis.len(), "multi_dot_strided") {
            Ok(values) => values,
            Err(error) => {
                log::error!(
                    target: "sonium_compute::backend",
                    "multi-dot strided basis staging allocation failed: {error}"
                );
                return Vec::new();
            }
        };
        let mut y_host = match try_zeroed_host::<T>(y.len(), "multi_dot_strided") {
            Ok(values) => values,
            Err(error) => {
                log::error!(
                    target: "sonium_compute::backend",
                    "multi-dot strided vector staging allocation failed: {error}"
                );
                return Vec::new();
            }
        };
        if let Err(error) = self.try_copy_d2h(basis, &mut basis_host) {
            log::error!(target: "sonium_compute::backend", "multi_dot_strided basis download failed: {error}");
            return Vec::new();
        }
        if let Err(error) = self.try_copy_d2h(y, &mut y_host) {
            log::error!(target: "sonium_compute::backend", "multi_dot_strided vector download failed: {error}");
            return Vec::new();
        }
        if basis_host.iter().any(|value| !value.is_finite())
            || y_host.iter().any(|value| !value.is_finite())
        {
            log::error!(
                target: "sonium_compute::backend",
                "multi_dot_strided rejected non-finite input"
            );
            return Vec::new();
        }
        let mut result = Vec::new();
        if let Err(error) = result.try_reserve_exact(basis_count) {
            log::error!(
                target: "sonium_compute::backend",
                "multi-dot strided result allocation failed: {error}"
            );
            return Vec::new();
        }
        for col in 0..basis_count {
            let base = col * vector_len;
            let mut acc = T::zero();
            for i in 0..vector_len {
                acc = acc + basis_host[base + i].conj_mul(y_host[i]);
            }
            result.push(acc);
        }
        result
    }

    /// Validate backend-specific limits for flat-basis operations before a
    /// checked path enters a native kernel. Most backends use `usize`
    /// dimensions throughout; accelerator kernels may narrow selected values
    /// to device scalar widths such as `u32`.
    fn validate_strided_backend_limits(
        &self,
        _operation: &'static str,
        _vector_len: usize,
        _basis_count: usize,
    ) -> BackendResult<()> {
        Ok(())
    }

    /// Fallible flat-basis multi-dot with an explicit reduction policy.
    ///
    /// This is the checked counterpart to [`Self::multi_dot_strided`]. It
    /// validates the logical view before any host transfer or backend
    /// dispatch, preserves fixed element order in deterministic mode, and
    /// reports backend panics/non-finite reductions as typed errors.
    fn try_multi_dot_strided_with_mode<T: DeviceScalar>(
        &self,
        mode: ReductionMode,
        basis: &Self::Buffer<T>,
        vector_len: usize,
        basis_count: usize,
        y: &Self::Buffer<T>,
    ) -> BackendResult<Vec<T>> {
        let capabilities = self.capabilities();
        require_scalar_capability::<T>(
            capabilities.backend_name,
            "multi_dot_strided",
            capabilities.multi_dot_strided,
        )?;
        require_reduction_capability::<T>(
            capabilities.backend_name,
            "multi_dot_strided",
            capabilities.reduction,
            mode,
        )?;
        if y.len() != vector_len {
            return Err(BackendError::InvalidArgument {
                operation: "multi_dot_strided",
                reason: format!("y length {}, expected {vector_len}", y.len()),
            });
        }
        let required =
            vector_len
                .checked_mul(basis_count)
                .ok_or_else(|| BackendError::InvalidArgument {
                    operation: "multi_dot_strided",
                    reason: format!("basis shape {basis_count}x{vector_len} overflows"),
                })?;
        if basis.len() < required {
            return Err(BackendError::InvalidArgument {
                operation: "multi_dot_strided",
                reason: format!(
                    "basis length {} is smaller than required {required}",
                    basis.len()
                ),
            });
        }
        self.validate_strided_backend_limits("multi_dot_strided", vector_len, basis_count)?;
        if basis_count == 0 {
            return Ok(Vec::new());
        }

        let result = match mode {
            ReductionMode::Fast => catch_backend_panic("multi_dot_strided", || {
                self.multi_dot_strided(basis, vector_len, basis_count, y)
            })?,
            ReductionMode::Reproducible | ReductionMode::Deterministic => {
                let mut basis_host = try_zeroed_host(basis.len(), "multi_dot_strided")?;
                let mut y_host = try_zeroed_host(y.len(), "multi_dot_strided")?;
                self.try_copy_d2h(basis, &mut basis_host)?;
                self.try_copy_d2h(y, &mut y_host)?;
                if basis_host
                    .iter()
                    .chain(y_host.iter())
                    .any(|value| !value.is_finite())
                {
                    return Err(BackendError::OperationFailed {
                        operation: "multi_dot_strided",
                        reason: "reduction input contains NaN or infinity".into(),
                    });
                }
                (0..basis_count)
                    .map(|column| {
                        let start = column * vector_len;
                        basis_host[start..start + vector_len]
                            .iter()
                            .zip(y_host.iter())
                            .fold(T::zero(), |acc, (&x_i, &y_i)| acc + x_i.conj_mul(y_i))
                    })
                    .collect()
            }
        };
        self.check_pending_error()?;
        if result.len() != basis_count {
            return Err(BackendError::OperationFailed {
                operation: "multi_dot_strided",
                reason: format!(
                    "backend returned {} results for {basis_count} basis vectors",
                    result.len()
                ),
            });
        }
        if result.iter().any(|value| !value.is_finite()) {
            return Err(BackendError::OperationFailed {
                operation: "multi_dot_strided",
                reason: "reduction result is non-finite".into(),
            });
        }
        Ok(result)
    }

    /// Fallible flat-basis multi-dot writing into caller-owned host storage.
    ///
    /// Krylov implementations use this form with workspace-backed storage so
    /// the small projection block does not allocate a fresh `Vec` on every
    /// Arnoldi step.  Native backends may override it to write the reduction
    /// result directly into their persistent staging allocation.
    fn try_multi_dot_strided_with_mode_into<T: DeviceScalar>(
        &self,
        mode: ReductionMode,
        basis: &Self::Buffer<T>,
        vector_len: usize,
        basis_count: usize,
        y: &Self::Buffer<T>,
        out: &mut [T],
    ) -> BackendResult<()> {
        if out.len() != basis_count {
            return Err(BackendError::InvalidArgument {
                operation: "multi_dot_strided_into",
                reason: format!("output length {}, expected {basis_count}", out.len()),
            });
        }
        if basis_count == 0 {
            return Ok(());
        }
        let result =
            self.try_multi_dot_strided_with_mode(mode, basis, vector_len, basis_count, y)?;
        out.copy_from_slice(&result);
        Ok(())
    }

    /// Checked flat-basis multi-dot writing into a reusable device buffer.
    /// Native accelerator implementations can keep the projection block on
    /// the device; the default explicitly stages only this small coefficient
    /// block for backends without a device-output primitive.
    fn try_multi_dot_strided_with_mode_device_into<T: DeviceScalar>(
        &self,
        mode: ReductionMode,
        basis: &Self::Buffer<T>,
        vector_len: usize,
        basis_count: usize,
        y: &Self::Buffer<T>,
        out: &mut Self::Buffer<T>,
    ) -> BackendResult<()> {
        if out.len() < basis_count {
            return Err(BackendError::InvalidArgument {
                operation: "multi_dot_strided_device_into",
                reason: format!(
                    "output length {} is smaller than basis count {basis_count}",
                    out.len()
                ),
            });
        }
        let mut host = try_zeroed_host::<T>(out.len(), "multi_dot_strided_device_into")?;
        self.try_multi_dot_strided_with_mode_into(
            mode,
            basis,
            vector_len,
            basis_count,
            y,
            &mut host[..basis_count],
        )?;
        self.try_copy_h2d(&host, out)
    }

    /// Tiled flat-basis multi-dot: `tiles` blocks cooperate on each dot.
    ///
    /// Opt-in occupancy path for small basis counts on wide GPUs; the 1D
    /// one-block-per-dot grid stays the default. Backends without a tiled
    /// kernel keep the default `Unsupported` body.
    fn try_multi_dot_tiled_with_mode_into<T: DeviceScalar>(
        &self,
        mode: ReductionMode,
        basis: &Self::Buffer<T>,
        vector_len: usize,
        basis_count: usize,
        tiles: usize,
        y: &Self::Buffer<T>,
        out: &mut [T],
    ) -> BackendResult<()> {
        let _ = (mode, basis, vector_len, basis_count, tiles, y, out);
        Err(BackendError::Unsupported {
            backend: "backend",
            operation: "multi_dot_tiled_into",
            scalar: std::any::type_name::<T>(),
        })
    }

    /// Fused flat-basis update: `y <- y + alpha * basis * coeffs`.
    ///
    /// With `alpha = -1`, this is the GMRES orthogonalisation update
    /// `w -= V*h`; with `alpha = 1`, this is the solution update
    /// `x += V*y_small`.
    ///
    /// The basis and destination are a disjoint-input/output contract. The
    /// checked native descriptor marks the destination `Disjoint`; callers
    /// that need overlapping storage must use an operation with explicit
    /// staging, such as [`Self::try_copy_view_in_place`].
    fn basis_axpy_strided<T: DeviceScalar>(
        &self,
        alpha: T,
        basis: &Self::Buffer<T>,
        vector_len: usize,
        basis_count: usize,
        coeffs: &[T],
        y: &mut Self::Buffer<T>,
    ) {
        if coeffs.len() != basis_count {
            log::error!(
                target: "sonium_compute::backend",
                "basis_axpy_strided rejected coefficient length {} (expected {basis_count})",
                coeffs.len()
            );
            return;
        }
        if y.len() != vector_len {
            log::error!(
                target: "sonium_compute::backend",
                "basis_axpy_strided rejected y length {} (expected {vector_len})",
                y.len()
            );
            return;
        }
        let Some(required) = vector_len.checked_mul(basis_count) else {
            log::error!(
                target: "sonium_compute::backend",
                "basis_axpy_strided basis shape {basis_count}x{vector_len} overflows"
            );
            return;
        };
        if basis.len() < required {
            log::error!(
                target: "sonium_compute::backend",
                "basis_axpy_strided rejected basis length {} (expected at least {required})",
                basis.len()
            );
            return;
        }
        if basis_count == 0 || vector_len == 0 {
            return;
        }
        let mut basis_host = match try_zeroed_host(basis.len(), "basis_axpy_strided") {
            Ok(values) => values,
            Err(error) => {
                log::error!(
                    target: "sonium_compute::backend",
                    "basis_axpy_strided basis staging allocation failed: {error}"
                );
                return;
            }
        };
        let mut y_host = match try_zeroed_host(y.len(), "basis_axpy_strided") {
            Ok(values) => values,
            Err(error) => {
                log::error!(
                    target: "sonium_compute::backend",
                    "basis_axpy_strided vector staging allocation failed: {error}"
                );
                return;
            }
        };
        if let Err(error) = self.try_copy_d2h(basis, &mut basis_host) {
            log::error!(target: "sonium_compute::backend", "basis_axpy_strided basis download failed: {error}");
            return;
        }
        if let Err(error) = self.try_copy_d2h(y, &mut y_host) {
            log::error!(target: "sonium_compute::backend", "basis_axpy_strided vector download failed: {error}");
            return;
        }
        if !alpha.is_finite() || coeffs.iter().any(|value| !value.is_finite()) {
            log::error!(
                target: "sonium_compute::backend",
                "basis_axpy_strided rejected non-finite coefficient"
            );
            return;
        }
        for i in 0..vector_len {
            let mut update = T::zero();
            for (col, &coeff) in coeffs.iter().enumerate() {
                update = update + basis_host[col * vector_len + i] * coeff;
            }
            y_host[i] = y_host[i] + alpha * update;
        }
        if let Err(error) = self.try_copy_h2d(&y_host, y) {
            log::error!(target: "sonium_compute::backend", "basis_axpy_strided result upload failed: {error}");
        }
    }

    /// Fallible checked counterpart to [`Self::basis_axpy_strided`]. The basis
    /// and destination must remain disjoint for the native fused update.
    ///
    /// The destination is not usable if the backend reports an error after a
    /// partial update; callers must discard it and retry from a fresh buffer.
    fn try_basis_axpy_strided<T: DeviceScalar>(
        &self,
        alpha: T,
        basis: &Self::Buffer<T>,
        vector_len: usize,
        basis_count: usize,
        coeffs: &[T],
        y: &mut Self::Buffer<T>,
    ) -> BackendResult<()> {
        if coeffs.len() != basis_count {
            return Err(BackendError::InvalidArgument {
                operation: "basis_axpy_strided",
                reason: format!(
                    "coefficient length {}, expected {basis_count}",
                    coeffs.len()
                ),
            });
        }
        if y.len() != vector_len {
            return Err(BackendError::InvalidArgument {
                operation: "basis_axpy_strided",
                reason: format!("y length {}, expected {vector_len}", y.len()),
            });
        }
        let required =
            vector_len
                .checked_mul(basis_count)
                .ok_or_else(|| BackendError::InvalidArgument {
                    operation: "basis_axpy_strided",
                    reason: "basis shape multiplication overflow".into(),
                })?;
        if basis.len() < required {
            return Err(BackendError::InvalidArgument {
                operation: "basis_axpy_strided",
                reason: format!("basis length {}, expected at least {required}", basis.len()),
            });
        }
        self.validate_strided_backend_limits("basis_axpy_strided", vector_len, basis_count)?;
        if !alpha.is_finite() || coeffs.iter().any(|value| !value.is_finite()) {
            return Err(BackendError::OperationFailed {
                operation: "basis_axpy_strided",
                reason: "update coefficient is non-finite".into(),
            });
        }
        let mut basis_host = try_zeroed_host(basis.len(), "basis_axpy_strided")?;
        let mut y_host = try_zeroed_host(y.len(), "basis_axpy_strided")?;
        self.try_copy_d2h(basis, &mut basis_host)?;
        self.try_copy_d2h(y, &mut y_host)?;
        for i in 0..vector_len {
            let mut update = T::zero();
            for (column, &coefficient) in coeffs.iter().enumerate() {
                update = update + basis_host[column * vector_len + i] * coefficient;
            }
            y_host[i] = y_host[i] + alpha * update;
        }
        if y_host.iter().any(|value| !value.is_finite()) {
            return Err(BackendError::OperationFailed {
                operation: "basis_axpy_strided",
                reason: "update produced a non-finite value".into(),
            });
        }
        self.try_copy_h2d(&y_host, y)?;
        Ok(())
    }

    /// Checked flat-basis update whose full coefficients already reside in a
    /// backend buffer: `y <- y + basis * coeffs[..basis_count]`.
    ///
    /// This is the device-resident counterpart used by block Krylov. The
    /// default stages only the small coefficient view and delegates to the
    /// ordinary checked update; CUDA consumes the coefficient buffer directly
    /// in its fused kernel.
    fn try_basis_axpy_strided_device<T: DeviceScalar>(
        &self,
        basis: &Self::Buffer<T>,
        vector_len: usize,
        basis_count: usize,
        coeffs: &Self::Buffer<T>,
        y: &mut Self::Buffer<T>,
    ) -> BackendResult<()> {
        let capabilities = self.capabilities();
        require_scalar_capability::<T>(
            capabilities.backend_name,
            "basis_axpy_strided",
            capabilities.basis_axpy_strided,
        )?;
        let required =
            vector_len
                .checked_mul(basis_count)
                .ok_or_else(|| BackendError::InvalidArgument {
                    operation: "basis_axpy_strided_device",
                    reason: "basis shape multiplication overflow".into(),
                })?;
        if basis.len() < required || y.len() != vector_len || coeffs.len() < basis_count {
            return Err(BackendError::InvalidArgument {
                operation: "basis_axpy_strided_device",
                reason: format!(
                    "invalid lengths: basis {}, vector_len {vector_len}, coeffs {}, y {}",
                    basis.len(),
                    coeffs.len(),
                    y.len()
                ),
            });
        }
        if basis_count == 0 || vector_len == 0 {
            return Ok(());
        }
        let coeff_view = BufferView::new(coeffs.len(), 0, basis_count, 1)?;
        let mut host_coeffs = try_zeroed_host(basis_count, "basis_axpy_strided_device")?;
        self.try_copy_view_d2h(coeffs, coeff_view, &mut host_coeffs)?;
        self.try_basis_axpy_strided(T::one(), basis, vector_len, basis_count, &host_coeffs, y)
    }

    /// Project `y` against a flat basis and subtract the projection in one
    /// policy-visible operation: `y <- y - basis * (basis^H y)`.
    ///
    /// Returns the projection coefficients so Arnoldi implementations can
    /// accumulate the Hessenberg column without recomputing the dots.
    fn basis_project_out_strided<T: DeviceScalar>(
        &self,
        basis: &Self::Buffer<T>,
        vector_len: usize,
        basis_count: usize,
        y: &mut Self::Buffer<T>,
    ) -> Vec<T> {
        match self.try_basis_project_out_strided_with_mode(
            ReductionMode::Fast,
            basis,
            vector_len,
            basis_count,
            y,
        ) {
            Ok(coeffs) => coeffs,
            Err(error) => {
                log::error!(
                    target: "sonium_compute::backend",
                    "basis_project_out_strided failed: {error}"
                );
                Vec::new()
            }
        }
    }

    /// Fallible basis projection with an explicit reduction policy. The
    /// deterministic path reads each checked basis view in a fixed order and
    /// propagates transfer/shape failures instead of silently truncating.
    fn try_basis_project_out_strided_with_mode<T: DeviceScalar>(
        &self,
        mode: ReductionMode,
        basis: &Self::Buffer<T>,
        vector_len: usize,
        basis_count: usize,
        y: &mut Self::Buffer<T>,
    ) -> BackendResult<Vec<T>> {
        let capabilities = self.capabilities();
        require_scalar_capability::<T>(
            capabilities.backend_name,
            "basis_project_out_strided",
            capabilities.basis_project_out_strided,
        )?;
        require_scalar_capability::<T>(
            capabilities.backend_name,
            "basis_project_out_strided/multi_dot",
            capabilities.multi_dot_strided,
        )?;
        require_scalar_capability::<T>(
            capabilities.backend_name,
            "basis_project_out_strided/basis_axpy",
            capabilities.basis_axpy_strided,
        )?;
        require_reduction_capability::<T>(
            capabilities.backend_name,
            "basis_project_out_strided",
            capabilities.reduction,
            mode,
        )?;
        if y.len() != vector_len {
            return Err(BackendError::InvalidArgument {
                operation: "basis_project_out_strided",
                reason: format!("y length {}, expected {vector_len}", y.len()),
            });
        }
        let expected_basis_len =
            vector_len
                .checked_mul(basis_count)
                .ok_or_else(|| BackendError::InvalidArgument {
                    operation: "basis_project_out_strided",
                    reason: "basis length multiplication overflow".into(),
                })?;
        if basis.len() < expected_basis_len {
            return Err(BackendError::InvalidArgument {
                operation: "basis_project_out_strided",
                reason: format!(
                    "basis length {}, expected at least {expected_basis_len}",
                    basis.len()
                ),
            });
        }
        if mode == ReductionMode::Fast {
            // Do not call the legacy projection wrapper here.  It preserves
            // an infallible API by converting transfer failures to an empty
            // result, and its legacy AXPY half can likewise swallow a failed
            // destination upload.  Checked callers must observe both halves
            // of the operation independently.
            let coeffs = self.try_multi_dot_strided_with_mode(
                ReductionMode::Fast,
                basis,
                vector_len,
                basis_count,
                y,
            )?;
            self.try_basis_axpy_strided(
                T::zero() - T::one(),
                basis,
                vector_len,
                basis_count,
                &coeffs,
                y,
            )?;
            self.check_pending_error()?;
            if coeffs.len() != basis_count || coeffs.iter().any(|value| !value.is_finite()) {
                return Err(BackendError::OperationFailed {
                    operation: "basis_project_out_strided",
                    reason: "backend returned an invalid reduction result".into(),
                });
            }
            return Ok(coeffs);
        }

        let mut y_host = try_zeroed_host(vector_len, "basis_project_out_strided")?;
        self.try_copy_d2h(y, &mut y_host)?;
        if y_host.iter().any(|value| !value.is_finite()) {
            return Err(BackendError::OperationFailed {
                operation: "basis_project_out_strided",
                reason: "reduction input contains a non-finite value".into(),
            });
        }
        // Reserve through the checked host-staging seam. A checked projection
        // must report allocation failure instead of allowing infallible Vec
        // growth to panic or expose a partial coefficient vector.
        let mut coeffs = try_zeroed_host::<T>(basis_count, "basis_project_out_strided")?;
        // Reuse one staging vector for every basis column.  The checked path is
        // used by deterministic/reproducible Krylov iterations, so allocating a
        // fresh host vector for each column defeats the steady-state workspace
        // contract (and makes allocation failures depend on basis_count).
        let mut basis_host = try_zeroed_host::<T>(vector_len, "basis_project_out_strided")?;
        for (basis_index, coefficient) in coeffs.iter_mut().enumerate() {
            let view = BufferView::new(basis.len(), basis_index * vector_len, vector_len, 1)?;
            self.try_copy_view_d2h(basis, view, &mut basis_host)?;
            if basis_host.iter().any(|value| !value.is_finite()) {
                return Err(BackendError::OperationFailed {
                    operation: "basis_project_out_strided",
                    reason: "reduction input contains a non-finite value".into(),
                });
            }
            *coefficient = basis_host
                .iter()
                .copied()
                .zip(y_host.iter().copied())
                .fold(T::zero(), |acc, (basis_i, y_i)| acc + basis_i.conj_mul(y_i));
        }
        self.try_basis_axpy_strided(
            T::zero() - T::one(),
            basis,
            vector_len,
            basis_count,
            &coeffs,
            y,
        )?;
        Ok(coeffs)
    }

    /// `sqrt(Σ |x_i|²)`.
    fn nrm2<T: DeviceScalar>(&self, x: &Self::Buffer<T>) -> T::Real;

    /// Norm with an explicit reproducibility policy.
    fn nrm2_with_mode<T: DeviceScalar>(&self, mode: ReductionMode, x: &Self::Buffer<T>) -> T::Real {
        match mode {
            ReductionMode::Fast => self.nrm2(x),
            ReductionMode::Reproducible | ReductionMode::Deterministic => {
                self.nrm2_deterministic(x)
            }
        }
    }

    /// Fallible norm dispatch. Legacy backend implementations may still
    /// expose an infallible `nrm2`; checked callers must not let a driver
    /// panic escape that boundary.
    fn try_nrm2_with_mode<T: DeviceScalar>(
        &self,
        mode: ReductionMode,
        x: &Self::Buffer<T>,
    ) -> BackendResult<T::Real> {
        let capabilities = self.capabilities();
        require_reduction_capability::<T>(
            capabilities.backend_name,
            "nrm2",
            capabilities.reduction,
            mode,
        )?;
        if matches!(mode, ReductionMode::Fast) {
            self.validate_vector_backend_limits::<T>("nrm2", x.len())?;
        }
        let result = match mode {
            ReductionMode::Fast => catch_backend_panic("nrm2", || self.nrm2(x)),
            ReductionMode::Reproducible | ReductionMode::Deterministic => {
                self.try_nrm2_deterministic(x)
            }
        }?;
        self.check_pending_error()?;
        if !result.is_finite() {
            return Err(BackendError::OperationFailed {
                operation: "nrm2",
                reason: "reduction result is non-finite".into(),
            });
        }
        Ok(result)
    }

    /// Fixed-order norm used by deterministic QA and derivative checks.
    fn nrm2_deterministic<T: DeviceScalar>(&self, x: &Self::Buffer<T>) -> T::Real {
        self.try_nrm2_deterministic(x).unwrap_or_else(|error| {
            log::error!(target: "sonium_compute::backend", "deterministic norm rejected: {error}");
            T::Real::zero()
        })
    }

    /// Fallible fixed-order norm. Unlike the legacy compatibility method this
    /// preserves transfer and non-finite-input failures for checked callers.
    fn try_nrm2_deterministic<T: DeviceScalar>(
        &self,
        x: &Self::Buffer<T>,
    ) -> BackendResult<T::Real> {
        let capabilities = self.capabilities();
        require_reduction_capability::<T>(
            capabilities.backend_name,
            "nrm2_deterministic",
            capabilities.reduction,
            ReductionMode::Deterministic,
        )?;
        let mut host = try_zeroed_host(x.len(), "nrm2_deterministic")?;
        self.try_copy_d2h(x, &mut host)?;
        if host.iter().any(|value| !value.is_finite()) {
            return Err(BackendError::OperationFailed {
                operation: "nrm2",
                reason: "reduction input contains a non-finite value".into(),
            });
        }
        let result = crate::scalar::stable_nrm2(&host);
        if !result.is_finite() {
            return Err(BackendError::OperationFailed {
                operation: "nrm2",
                reason: "reduction result is non-finite".into(),
            });
        }
        Ok(result)
    }

    /// `x <- alpha * x` in place.
    fn scale<T: DeviceScalar>(&self, alpha: T, x: &mut Self::Buffer<T>);

    /// Checked in-place scaling. Native backends should override this when
    /// their BLAS call can report execution or pointer failures.
    fn try_scale<T: DeviceScalar>(&self, alpha: T, x: &mut Self::Buffer<T>) -> BackendResult<()> {
        let capabilities = self.capabilities();
        require_scalar_capability::<T>(capabilities.backend_name, "scale", capabilities.blas1)?;
        require_finite_scalar("scale", "alpha", alpha)?;
        self.validate_vector_backend_limits::<T>("scale", x.len())?;
        catch_backend_panic("scale", || self.scale(alpha, x))?;
        self.check_pending_error()
    }

    // ----- BLAS-2 -----

    /// `y := alpha * op(A) * x + beta * y` (dense row-major matrix-vector
    /// product).
    ///
    /// `op(A)` has shape `m × n`:
    /// - For `Transpose::NoTrans`, the *stored* `a` is `m × n` row-major and
    ///   `lda ≥ n`.
    /// - For `Transpose::Trans` / `Transpose::ConjTrans`, the *stored* `a` is
    ///   `n × m` row-major and `lda ≥ m`.
    ///
    /// `x` has length `n`, `y` has length `m`.
    ///
    /// The default is a validated host fallback. Device backends should
    /// override it when a native implementation exists; capability tables
    /// must then report `Native` or `VendorNative` for that path.
    #[allow(clippy::too_many_arguments)]
    fn gemv<T: DeviceScalar>(
        &self,
        op: Transpose,
        m: usize,
        n: usize,
        alpha: T,
        a: &Self::Buffer<T>,
        lda: usize,
        x: &Self::Buffer<T>,
        beta: T,
        y: &mut Self::Buffer<T>,
    ) {
        if let Err(error) = host_gemv(self, op, m, n, alpha, a, lda, x, beta, y) {
            log::error!(target: "sonium_compute::backend", "default GEMV fallback rejected request: {error}");
        }
    }

    /// Grouped dense matrix-vector dispatch.
    ///
    /// The default implementation is a correctness-preserving loop over
    /// [`Self::gemv`]. Device backends can override this with grouped/batched
    /// vendor APIs once the shapes and epilogues justify the extra machinery.
    fn gemv_batched<T: DeviceScalar>(&self, batch: &mut [GemvBatch<'_, Self::Buffer<T>, T>])
    where
        Self: Sized,
    {
        for item in batch {
            self.gemv(
                item.op, item.m, item.n, item.alpha, item.a, item.lda, item.x, item.beta, item.y,
            );
        }
    }

    /// Checked grouped GEMV dispatch.
    ///
    /// Every item is validated before the legacy batch method is entered, so
    /// callers that accept user-controlled shapes can receive a typed error
    /// instead of relying on a backend's infallible compatibility path.
    fn try_gemv_batched<T: DeviceScalar>(
        &self,
        batch: &mut [GemvBatch<'_, Self::Buffer<T>, T>],
    ) -> BackendResult<()>
    where
        Self: Sized,
    {
        let capabilities = self.capabilities();
        require_scalar_capability::<T>(
            capabilities.backend_name,
            "gemv_batched",
            capabilities.gemv,
        )?;
        for item in batch.iter() {
            validate_gemv_args(
                item.op,
                item.m,
                item.n,
                item.a.len(),
                item.lda,
                item.x.len(),
                item.y.len(),
            )?;
            let (stored_rows, stored_cols) = match item.op {
                Transpose::NoTrans => (item.m, item.n),
                Transpose::Trans | Transpose::ConjTrans => (item.n, item.m),
            };
            self.checked_dense_matrix_descriptor(
                item.a,
                stored_rows,
                stored_cols,
                item.lda,
                ViewAliasing::ReadOnly,
            )?;
            self.checked_dense_vector_descriptor(item.x, ViewAliasing::ReadOnly)?;
            self.checked_dense_vector_descriptor(item.y, ViewAliasing::Disjoint)?;
        }
        for item in batch.iter() {
            require_finite_scalar("gemv_batched", "alpha", item.alpha)?;
            require_finite_scalar("gemv_batched", "beta", item.beta)?;
        }
        catch_backend_panic("gemv_batched", || self.gemv_batched(batch))?;
        self.check_pending_error()
    }

    // ----- BLAS-3 -----

    /// `C := alpha * op(A) * op(B) + beta * C` (dense row-major matrix-matrix
    /// product).
    ///
    /// `op(A)` is `m × k`, `op(B)` is `k × n`, and `C` is `m × n` row-major
    /// with stride `ldc ≥ n`. Stored shapes follow the same convention as
    /// [`Self::gemv`].
    #[allow(clippy::too_many_arguments)]
    fn gemm<T: DeviceScalar>(
        &self,
        transa: Transpose,
        transb: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: T,
        a: &Self::Buffer<T>,
        lda: usize,
        b: &Self::Buffer<T>,
        ldb: usize,
        beta: T,
        c: &mut Self::Buffer<T>,
        ldc: usize,
    ) {
        if let Err(error) = host_gemm(
            self, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        ) {
            log::error!(target: "sonium_compute::backend", "default GEMM fallback rejected request: {error}");
        }
    }

    /// Grouped dense matrix-matrix dispatch.
    ///
    /// The default implementation is a loop over [`Self::gemm`]. CUDA can
    /// later override this with cuBLAS grouped-batched GEMM or CUTLASS grouped
    /// scheduling without changing solver/block-operator call sites.
    ///
    /// The default implementation is a validated loop over [`Self::gemm`].
    /// Backends that need grouped vendor dispatch may override this method;
    /// unsupported scalar types are handled by the underlying GEMM contract
    /// rather than by an `unimplemented!` panic.
    fn gemm_batched<T: DeviceScalar>(&self, batch: &mut [GemmBatch<'_, Self::Buffer<T>, T>])
    where
        Self: Sized,
    {
        for item in batch {
            self.gemm(
                item.transa,
                item.transb,
                item.m,
                item.n,
                item.k,
                item.alpha,
                item.a,
                item.lda,
                item.b,
                item.ldb,
                item.beta,
                item.c,
                item.ldc,
            );
        }
    }

    /// Checked grouped GEMM dispatch.
    ///
    /// Every item is validated before the legacy batch method is entered, so
    /// malformed dimensions cannot reach a native launch.
    fn try_gemm_batched<T: DeviceScalar>(
        &self,
        batch: &mut [GemmBatch<'_, Self::Buffer<T>, T>],
    ) -> BackendResult<()>
    where
        Self: Sized,
    {
        let capabilities = self.capabilities();
        require_scalar_capability::<T>(
            capabilities.backend_name,
            "gemm_batched",
            capabilities.gemm,
        )?;
        for item in batch.iter() {
            validate_gemm_args(
                item.transa,
                item.transb,
                item.m,
                item.n,
                item.k,
                item.a.len(),
                item.lda,
                item.b.len(),
                item.ldb,
                item.c.len(),
                item.ldc,
            )?;
            let (a_rows, a_cols) = match item.transa {
                Transpose::NoTrans => (item.m, item.k),
                Transpose::Trans | Transpose::ConjTrans => (item.k, item.m),
            };
            let (b_rows, b_cols) = match item.transb {
                Transpose::NoTrans => (item.k, item.n),
                Transpose::Trans | Transpose::ConjTrans => (item.n, item.k),
            };
            self.checked_dense_matrix_descriptor(
                item.a,
                a_rows,
                a_cols,
                item.lda,
                ViewAliasing::ReadOnly,
            )?;
            self.checked_dense_matrix_descriptor(
                item.b,
                b_rows,
                b_cols,
                item.ldb,
                ViewAliasing::ReadOnly,
            )?;
            self.checked_dense_matrix_descriptor(
                item.c,
                item.m,
                item.n,
                item.ldc,
                ViewAliasing::Disjoint,
            )?;
        }
        for item in batch.iter() {
            require_finite_scalar("gemm_batched", "alpha", item.alpha)?;
            require_finite_scalar("gemm_batched", "beta", item.beta)?;
        }
        catch_backend_panic("gemm_batched", || self.gemm_batched(batch))?;
        self.check_pending_error()
    }

    // ----- Triangular solve (BLAS-3) -----

    /// Triangular solve with multiple right-hand sides: `op(A) * X = alpha * B`
    /// (`Left`) or `X * op(A) = alpha * B` (`Right`), solved in place over `B`.
    ///
    /// `A` holds the `adim x adim` triangle (`adim = m` for `Left`,
    /// `adim = n` for `Right`) with `lda >= adim`; `B` is `m x n` with
    /// `ldb >= n`. The default implementation stages the validated buffers
    /// through the host; native backends should override this method with a
    /// checked device primitive.
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "cpu")]
    fn trsm<T: DeviceScalar>(
        &self,
        side: TrsmSide,
        uplo: TrsmUplo,
        trans: Transpose,
        unit: bool,
        m: usize,
        n: usize,
        alpha: T,
        a: &Self::Buffer<T>,
        lda: usize,
        b: &mut Self::Buffer<T>,
        ldb: usize,
    ) {
        if let Err(error) = host_trsm(self, side, uplo, trans, unit, m, n, alpha, a, lda, b, ldb) {
            log::error!(target: "sonium_compute::backend", "default TRSM fallback rejected request: {error}");
        }
    }

    /// Checked triangular solve dispatch.
    ///
    /// Dimensions, descriptors, and `alpha` are validated before the legacy
    /// solve method is entered, so malformed dimensions cannot reach a
    /// native launch.
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "cpu")]
    fn try_trsm<T: DeviceScalar>(
        &self,
        side: TrsmSide,
        uplo: TrsmUplo,
        trans: Transpose,
        unit: bool,
        m: usize,
        n: usize,
        alpha: T,
        a: &Self::Buffer<T>,
        lda: usize,
        b: &mut Self::Buffer<T>,
        ldb: usize,
    ) -> BackendResult<()>
    where
        Self: Sized,
    {
        let adim = match side {
            TrsmSide::Left => m,
            TrsmSide::Right => n,
        };
        validate_trsm_args(side, uplo, m, n, a.len(), lda, b.len(), ldb)?;
        self.checked_dense_matrix_descriptor(a, adim, adim, lda, ViewAliasing::ReadOnly)?;
        self.checked_dense_matrix_descriptor(b, m, n, ldb, ViewAliasing::Disjoint)?;
        require_finite_scalar("trsm", "alpha", alpha)?;
        if m == 0 || n == 0 {
            return Ok(());
        }
        catch_backend_panic("trsm", || {
            self.trsm(side, uplo, trans, unit, m, n, alpha, a, lda, b, ldb)
        })?;
        self.check_pending_error()
    }

    // ----- Symmetric rank-k update (BLAS-3) -----

    /// Symmetric rank-k update: `C := alpha * op(A) * op(A)^T + beta * C`
    /// over the `uplo` triangle of the `n x n` matrix `C` (`op` is `N` or
    /// `T`; complex `ConjTrans` requests the Hermitian update like the CPU
    /// kernel). `A` is `n x k` for `NoTrans` and `k x n` otherwise.
    ///
    /// The default implementation stages the validated buffers through the
    /// host; native backends should override this method with a checked
    /// device primitive.
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "cpu")]
    fn syrk<T: DeviceScalar>(
        &self,
        uplo: TrsmUplo,
        trans: Transpose,
        n: usize,
        k: usize,
        alpha: T,
        a: &Self::Buffer<T>,
        lda: usize,
        beta: T,
        c: &mut Self::Buffer<T>,
        ldc: usize,
    ) {
        if let Err(error) = host_syrk(self, uplo, trans, n, k, alpha, a, lda, beta, c, ldc) {
            log::error!(target: "sonium_compute::backend", "default SYRK fallback rejected request: {error}");
        }
    }

    /// Checked symmetric rank-k dispatch.
    ///
    /// Dimensions, descriptors, and scalars are validated before the legacy
    /// update method is entered, so malformed dimensions cannot reach a
    /// native launch.
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "cpu")]
    fn try_syrk<T: DeviceScalar>(
        &self,
        uplo: TrsmUplo,
        trans: Transpose,
        n: usize,
        k: usize,
        alpha: T,
        a: &Self::Buffer<T>,
        lda: usize,
        beta: T,
        c: &mut Self::Buffer<T>,
        ldc: usize,
    ) -> BackendResult<()>
    where
        Self: Sized,
    {
        let (a_rows, a_cols) = match trans {
            Transpose::NoTrans => (n, k),
            Transpose::Trans | Transpose::ConjTrans => (k, n),
        };
        validate_syrk_args(trans, n, k, a.len(), lda, c.len(), ldc)?;
        self.checked_dense_matrix_descriptor(a, a_rows, a_cols, lda, ViewAliasing::ReadOnly)?;
        self.checked_dense_matrix_descriptor(c, n, n, ldc, ViewAliasing::Disjoint)?;
        require_finite_scalar("syrk", "alpha", alpha)?;
        require_finite_scalar("syrk", "beta", beta)?;
        // NOTE: no `k == 0` shortcut — a zero inner dimension still scales
        // `C` by `beta`, which both the host core and cuBLAS apply.
        if n == 0 {
            return Ok(());
        }
        catch_backend_panic("syrk", || {
            self.syrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
        })?;
        self.check_pending_error()
    }

    // ----- Sparse -----

    /// Checked CSR transpose: construct CSR(Aᵀ) from a resident CSR matrix.
    ///
    /// The input has `row_ptr.len() == m + 1` and
    /// `col_idx.len() == values.len() == nnz`. The output buffers must have
    /// `out_row_ptr.len() == n + 1` and
    /// `out_col_idx.len() == out_values.len() == nnz`.
    ///
    /// Outputs are separate from the input buffers and from each other. The
    /// default implementation stages the validated metadata and values
    /// through the host; native backends should override this method with a
    /// checked device primitive. No output is modified until all structural
    /// input validation succeeds.
    #[allow(clippy::too_many_arguments)]
    fn try_csr_transpose<T: DeviceScalar>(
        &self,
        m: usize,
        n: usize,
        row_ptr: &Self::Buffer<i32>,
        col_idx: &Self::Buffer<i32>,
        values: &Self::Buffer<T>,
        out_row_ptr: &mut Self::Buffer<i32>,
        out_col_idx: &mut Self::Buffer<i32>,
        out_values: &mut Self::Buffer<T>,
    ) -> BackendResult<()> {
        let capabilities = self.capabilities();
        require_scalar_capability::<T>(
            capabilities.backend_name,
            "csr_transpose",
            capabilities.csr_transpose,
        )?;
        let expected_row_ptr = m
            .checked_add(1)
            .ok_or_else(|| BackendError::InvalidArgument {
                operation: "csr_transpose",
                reason: "row-pointer length overflows m + 1".into(),
            })?;
        let expected_out_row_ptr =
            n.checked_add(1)
                .ok_or_else(|| BackendError::InvalidArgument {
                    operation: "csr_transpose",
                    reason: "output row-pointer length overflows n + 1".into(),
                })?;
        if row_ptr.len() != expected_row_ptr {
            return Err(BackendError::InvalidArgument {
                operation: "csr_transpose",
                reason: format!(
                    "row-pointer length {} != m + 1 ({expected_row_ptr})",
                    row_ptr.len()
                ),
            });
        }
        if col_idx.len() != values.len() {
            return Err(BackendError::InvalidArgument {
                operation: "csr_transpose",
                reason: format!(
                    "column-index length {} != value length {}",
                    col_idx.len(),
                    values.len()
                ),
            });
        }
        if out_row_ptr.len() != expected_out_row_ptr {
            return Err(BackendError::InvalidArgument {
                operation: "csr_transpose",
                reason: format!(
                    "output row-pointer length {} != n + 1 ({expected_out_row_ptr})",
                    out_row_ptr.len()
                ),
            });
        }
        if out_col_idx.len() != col_idx.len() || out_values.len() != values.len() {
            return Err(BackendError::InvalidArgument {
                operation: "csr_transpose",
                reason: format!(
                    "output nnz lengths {}/{} do not match input nnz {}",
                    out_col_idx.len(),
                    out_values.len(),
                    col_idx.len()
                ),
            });
        }

        let mut host_row_ptr = try_zeroed_host(row_ptr.len(), "csr_transpose")?;
        let mut host_col_idx = try_zeroed_host(col_idx.len(), "csr_transpose")?;
        let mut host_values = try_zeroed_host(values.len(), "csr_transpose")?;
        self.try_copy_d2h(row_ptr, &mut host_row_ptr)?;
        self.try_copy_d2h(col_idx, &mut host_col_idx)?;
        self.try_copy_d2h(values, &mut host_values)?;
        let mut host_out_row_ptr = try_zeroed_host(out_row_ptr.len(), "csr_transpose")?;
        let mut host_out_col_idx = try_zeroed_host(out_col_idx.len(), "csr_transpose")?;
        let mut host_out_values = try_zeroed_host(out_values.len(), "csr_transpose")?;
        host_csr_transpose(
            m,
            n,
            &host_row_ptr,
            &host_col_idx,
            &host_values,
            &mut host_out_row_ptr,
            &mut host_out_col_idx,
            &mut host_out_values,
        )?;
        self.try_copy_h2d(&host_out_row_ptr, out_row_ptr)?;
        self.try_copy_h2d(&host_out_col_idx, out_col_idx)?;
        self.try_copy_h2d(&host_out_values, out_values)?;
        self.check_pending_error()
    }

    /// Build a [`Self::CsrPrebaked`] handle for a CSR matrix that will
    /// be hit repeatedly.
    ///
    /// `row_ptr.len() == m + 1`, `col_idx.len() == values.len() == nnz`.
    /// The buffers must outlive the returned handle; that's enforced by
    /// the lifetime parameter.
    ///
    /// Krylov solvers should call this once per system matrix at the top
    /// of the solve, then drive `spmv_prebaked` inside the inner loop.
    fn prebake_csr<'a, T: DeviceScalar>(
        &'a self,
        m: usize,
        n: usize,
        row_ptr: &'a Self::Buffer<i32>,
        col_idx: &'a Self::Buffer<i32>,
        values: &'a Self::Buffer<T>,
    ) -> Self::CsrPrebaked<'a, T>;

    /// Checked CSR descriptor/workspace construction. Backends with native
    /// descriptor APIs should override this to preserve allocation and
    /// driver-status errors instead of relying on panic containment.
    fn try_prebake_csr<'a, T: DeviceScalar>(
        &'a self,
        m: usize,
        n: usize,
        row_ptr: &'a Self::Buffer<i32>,
        col_idx: &'a Self::Buffer<i32>,
        values: &'a Self::Buffer<T>,
    ) -> BackendResult<Self::CsrPrebaked<'a, T>> {
        let capabilities = self.capabilities();
        require_scalar_capability::<T>(
            capabilities.backend_name,
            "prebake_csr",
            capabilities.csr_spmv,
        )?;
        let prebaked = catch_backend_panic("prebake_csr", || {
            self.prebake_csr(m, n, row_ptr, col_idx, values)
        })?;
        self.check_pending_error()?;
        Ok(prebaked)
    }

    /// Backend-specific checked descriptor construction hook. Native
    /// descriptor backends override this to keep driver/library failures in
    /// the checked result instead of using the compatibility stub contract.
    #[doc(hidden)]
    fn try_prebake_csr_native<'a, T: DeviceScalar>(
        &'a self,
        _m: usize,
        _n: usize,
        _row_ptr: &'a Self::Buffer<i32>,
        _col_idx: &'a Self::Buffer<i32>,
        _values: &'a Self::Buffer<T>,
    ) -> BackendResult<Self::CsrPrebaked<'a, T>> {
        Err(BackendError::Unsupported {
            backend: self.capabilities().backend_name,
            operation: "prebake_csr_native",
            scalar: std::any::type_name::<T>(),
        })
    }

    /// `y <- A * x` against a prebaked descriptor.
    ///
    /// `x.len() == n`, `y.len() == m` (where `m`/`n` are those passed to
    /// [`Self::prebake_csr`]). Bounds checking is the caller's
    /// responsibility on the hot path; debug builds may assert.
    fn spmv_prebaked<T: DeviceScalar>(
        &self,
        _prebaked: &Self::CsrPrebaked<'_, T>,
        x: &Self::Buffer<T>,
        y: &mut Self::Buffer<T>,
    );

    /// Checked SpMV dispatch. Native backends should override this when
    /// they can validate descriptor dimensions and preserve driver status.
    fn try_spmv_prebaked<T: DeviceScalar>(
        &self,
        prebaked: &Self::CsrPrebaked<'_, T>,
        x: &Self::Buffer<T>,
        y: &mut Self::Buffer<T>,
    ) -> BackendResult<()> {
        if self.capabilities().csr_spmv.is_unsupported_for::<T>() {
            return Err(BackendError::Unsupported {
                backend: self.capabilities().backend_name,
                operation: "spmv_prebaked",
                scalar: std::any::type_name::<T>(),
            });
        }
        catch_backend_panic("spmv_prebaked", || self.spmv_prebaked(prebaked, x, y))?;
        self.check_pending_error()
    }

    /// Multi-RHS sparse matrix product against a prebaked CSR descriptor:
    /// `Y <- A * X`.
    ///
    /// `X` is an `n × nrhs` dense matrix in column-major family layout with
    /// leading dimension `ldx >= n`; `Y` is `m × nrhs` with `ldy >= m`.
    /// Backends with native sparse-matrix/matrix products should implement
    /// this directly so solvers can express padded or submatrix RHS batches.
    ///
    #[allow(clippy::too_many_arguments)]
    fn spmm_prebaked<T: DeviceScalar>(
        &self,
        _prebaked: &Self::CsrPrebaked<'_, T>,
        _nrhs: usize,
        _x: &Self::Buffer<T>,
        _ldx: usize,
        _y: &mut Self::Buffer<T>,
        _ldy: usize,
    ) {
        log::error!(
            target: "sonium_compute::backend",
            "CSR SpMM is unsupported on backend {} for {}",
            self.capabilities().backend_name,
            std::any::type_name::<T>()
        );
    }

    /// Checked SpMM dispatch. The capability table is consulted before the
    /// legacy void method so callers can distinguish unsupported hardware from
    /// a completed operation.
    #[allow(clippy::too_many_arguments)]
    fn try_spmm_prebaked<T: DeviceScalar>(
        &self,
        _prebaked: &Self::CsrPrebaked<'_, T>,
        nrhs: usize,
        x: &Self::Buffer<T>,
        ldx: usize,
        y: &mut Self::Buffer<T>,
        ldy: usize,
    ) -> BackendResult<()> {
        if nrhs != 0 && (ldx == 0 || ldy == 0) {
            return Err(BackendError::InvalidArgument {
                operation: "spmm_prebaked",
                reason: "leading dimensions must be non-zero for non-empty RHS".into(),
            });
        }
        let x_required = ldx
            .checked_mul(nrhs)
            .ok_or_else(|| BackendError::InvalidArgument {
                operation: "spmm_prebaked",
                reason: "x leading-dimension multiplication overflow".into(),
            })?;
        let y_required = ldy
            .checked_mul(nrhs)
            .ok_or_else(|| BackendError::InvalidArgument {
                operation: "spmm_prebaked",
                reason: "y leading-dimension multiplication overflow".into(),
            })?;
        if x.len() < x_required || y.len() < y_required {
            return Err(BackendError::InvalidArgument {
                operation: "spmm_prebaked",
                reason: format!("buffers too short for nrhs={nrhs}, ldx={ldx}, ldy={ldy}"),
            });
        }
        if self.capabilities().csr_spmm.is_unsupported_for::<T>() {
            return Err(BackendError::Unsupported {
                backend: self.capabilities().backend_name,
                operation: "spmm_prebaked",
                scalar: std::any::type_name::<T>(),
            });
        }
        // A backend that advertises CSR SpMM must override this checked
        // boundary. Calling the legacy void default here would log and return
        // success without writing `y`, which is indistinguishable from a
        // completed operation to a solver.
        Err(BackendError::Unsupported {
            backend: self.capabilities().backend_name,
            operation: "spmm_prebaked",
            scalar: std::any::type_name::<T>(),
        })
    }

    /// Multi-RHS sparse matrix product against a prebaked CSR descriptor:
    /// `Y <- A * X`.
    ///
    /// `X` is an `n × rhs` dense matrix in column-major layout (`ldx = n`),
    /// and `Y` is an `m × rhs` dense matrix in column-major layout
    /// (`ldy = m`). This makes each RHS contiguous, so CPU-style repeated
    /// SpMV and GPU SpMM both have a clear layout contract.
    ///
    fn spmm_prebaked_colmajor<T: DeviceScalar>(
        &self,
        _prebaked: &Self::CsrPrebaked<'_, T>,
        _rhs: usize,
        _x: &Self::Buffer<T>,
        _y: &mut Self::Buffer<T>,
    ) {
        log::error!(
            target: "sonium_compute::backend",
            "column-major CSR SpMM is unsupported on backend {} for {}",
            self.capabilities().backend_name,
            std::any::type_name::<T>()
        );
    }

    /// Checked column-major SpMM dispatch.
    fn try_spmm_prebaked_colmajor<T: DeviceScalar>(
        &self,
        _prebaked: &Self::CsrPrebaked<'_, T>,
        _rhs: usize,
        _x: &Self::Buffer<T>,
        _y: &mut Self::Buffer<T>,
    ) -> BackendResult<()> {
        if self.capabilities().csr_spmm.is_unsupported_for::<T>() {
            return Err(BackendError::Unsupported {
                backend: self.capabilities().backend_name,
                operation: "spmm_prebaked_colmajor",
                scalar: std::any::type_name::<T>(),
            });
        }
        // As with row-major SpMM, the checked method is the capability
        // boundary. A backend must implement it explicitly rather than
        // inheriting the legacy logging-only void default.
        Err(BackendError::Unsupported {
            backend: self.capabilities().backend_name,
            operation: "spmm_prebaked_colmajor",
            scalar: std::any::type_name::<T>(),
        })
    }

    /// One-shot CSR sparse matrix-vector product: `y <- A * x`.
    ///
    /// Default impl prebakes-then-prebaked, so backends only need to
    /// implement [`Self::prebake_csr`] and [`Self::spmv_prebaked`].
    /// Override only if there's a measurable single-call advantage to
    /// inlining the path (e.g. on CPU, skipping the holder allocation).
    #[allow(clippy::too_many_arguments)]
    fn spmv_csr<T: DeviceScalar>(
        &self,
        m: usize,
        n: usize,
        row_ptr: &Self::Buffer<i32>,
        col_idx: &Self::Buffer<i32>,
        values: &Self::Buffer<T>,
        x: &Self::Buffer<T>,
        y: &mut Self::Buffer<T>,
    ) {
        let prebaked = self.prebake_csr::<T>(m, n, row_ptr, col_idx, values);
        self.spmv_prebaked::<T>(&prebaked, x, y);
    }

    /// Checked one-shot CSR SpMV: validates descriptor construction and the
    /// native operation before returning control to the caller.
    #[allow(clippy::too_many_arguments)]
    fn try_spmv_csr<T: DeviceScalar>(
        &self,
        m: usize,
        n: usize,
        row_ptr: &Self::Buffer<i32>,
        col_idx: &Self::Buffer<i32>,
        values: &Self::Buffer<T>,
        x: &Self::Buffer<T>,
        y: &mut Self::Buffer<T>,
    ) -> BackendResult<()> {
        let capabilities = self.capabilities();
        require_scalar_capability::<T>(
            capabilities.backend_name,
            "spmv_csr",
            capabilities.csr_spmv,
        )?;
        // Validate the complete CSR contract before constructing a backend
        // descriptor.  The legacy `prebake_csr` entry point historically
        // assumed trusted mesh data and several implementations index the
        // row/column arrays directly; allowing malformed metadata through
        // here would therefore turn a checked API back into a panic boundary.
        let expected_row_ptr = m
            .checked_add(1)
            .ok_or_else(|| BackendError::InvalidArgument {
                operation: "spmv_csr",
                reason: "row-pointer length overflows m + 1".into(),
            })?;
        if row_ptr.len() != expected_row_ptr {
            return Err(BackendError::InvalidArgument {
                operation: "spmv_csr",
                reason: format!(
                    "row-pointer length {} != m + 1 ({expected_row_ptr})",
                    row_ptr.len()
                ),
            });
        }
        if col_idx.len() != values.len() {
            return Err(BackendError::InvalidArgument {
                operation: "spmv_csr",
                reason: format!(
                    "column-index length {} != value length {}",
                    col_idx.len(),
                    values.len()
                ),
            });
        }
        if x.len() != n || y.len() != m {
            return Err(BackendError::InvalidArgument {
                operation: "spmv_csr",
                reason: format!(
                    "vector lengths x/y {}/{} != matrix dimensions {}/{}",
                    x.len(),
                    y.len(),
                    n,
                    m
                ),
            });
        }

        let mut host_row_ptr = try_zeroed_host(row_ptr.len(), "spmv_csr")?;
        let mut host_col_idx = try_zeroed_host(col_idx.len(), "spmv_csr")?;
        let mut host_values = try_zeroed_host(values.len(), "spmv_csr")?;
        self.try_copy_d2h(row_ptr, &mut host_row_ptr)?;
        self.try_copy_d2h(col_idx, &mut host_col_idx)?;
        self.try_copy_d2h(values, &mut host_values)?;
        let nnz = host_col_idx.len();
        if host_row_ptr[0] != 0
            || host_row_ptr
                .windows(2)
                .any(|window| window[0] < 0 || window[1] < window[0] || (window[1] as usize) > nnz)
            || host_row_ptr[m] < 0
            || host_row_ptr[m] as usize != nnz
        {
            return Err(BackendError::InvalidArgument {
                operation: "spmv_csr",
                reason: "row-pointer values are not monotone or do not terminate at nnz".into(),
            });
        }
        if host_col_idx
            .iter()
            .any(|&column| column < 0 || column as usize >= n)
        {
            return Err(BackendError::InvalidArgument {
                operation: "spmv_csr",
                reason: "column index is outside matrix dimensions".into(),
            });
        }
        if host_values.iter().any(|value| !value.is_finite()) {
            return Err(BackendError::OperationFailed {
                operation: "spmv_csr",
                reason: "matrix values contain NaN or infinity".into(),
            });
        }
        let mut host_x = try_zeroed_host(x.len(), "spmv_csr")?;
        self.try_copy_d2h(x, &mut host_x)?;
        if host_x.iter().any(|value| !value.is_finite()) {
            return Err(BackendError::OperationFailed {
                operation: "spmv_csr",
                reason: "input vector contains NaN or infinity".into(),
            });
        }
        let prebaked = self.try_prebake_csr::<T>(m, n, row_ptr, col_idx, values)?;
        self.try_spmv_prebaked::<T>(&prebaked, x, y)
    }

    /// Unchecked CSR SpMV: same computation as [`try_spmv_csr`] but skips
    /// the host-side full-matrix validation (device-to-host copies plus
    /// monotonicity, column-bounds, and finiteness walks). Keeps the O(1)
    /// dimension checks, then prebakes descriptors per call and runs the
    /// prebaked SpMV.
    ///
    /// The caller guarantees a structurally valid finite CSR matrix and
    /// vector (e.g., validated once via [`try_spmv_csr`]); malformed input
    /// may produce garbage or backend errors instead of a checked
    /// [`BackendError::InvalidArgument`]. For hot loops over a fixed
    /// pattern, prebake once and call [`try_spmv_prebaked`] directly —
    /// this entry still rebuilds descriptors every call.
    fn try_spmv_csr_unchecked<T: DeviceScalar>(
        &self,
        m: usize,
        n: usize,
        row_ptr: &Self::Buffer<i32>,
        col_idx: &Self::Buffer<i32>,
        values: &Self::Buffer<T>,
        x: &Self::Buffer<T>,
        y: &mut Self::Buffer<T>,
    ) -> BackendResult<()> {
        let capabilities = self.capabilities();
        require_scalar_capability::<T>(
            capabilities.backend_name,
            "spmv_csr_unchecked",
            capabilities.csr_spmv,
        )?;
        let expected_row_ptr = m
            .checked_add(1)
            .ok_or_else(|| BackendError::InvalidArgument {
                operation: "spmv_csr_unchecked",
                reason: "row-pointer length overflows m + 1".into(),
            })?;
        if row_ptr.len() != expected_row_ptr {
            return Err(BackendError::InvalidArgument {
                operation: "spmv_csr_unchecked",
                reason: format!(
                    "row-pointer length {} != m + 1 ({expected_row_ptr})",
                    row_ptr.len()
                ),
            });
        }
        if col_idx.len() != values.len() {
            return Err(BackendError::InvalidArgument {
                operation: "spmv_csr_unchecked",
                reason: format!(
                    "column-index length {} != value length {}",
                    col_idx.len(),
                    values.len()
                ),
            });
        }
        if x.len() != n || y.len() != m {
            return Err(BackendError::InvalidArgument {
                operation: "spmv_csr_unchecked",
                reason: format!(
                    "vector lengths x/y {}/{} != matrix dimensions {}/{}",
                    x.len(),
                    y.len(),
                    n,
                    m
                ),
            });
        }
        let prebaked = self.try_prebake_csr::<T>(m, n, row_ptr, col_idx, values)?;
        self.try_spmv_prebaked::<T>(&prebaked, x, y)
    }

    /// Evaluate regular Helmholtz `(G, H, H^T, E)` triangle-pair integrals
    /// using the shared f32 BEM kernel ABI.
    ///
    /// Backends with native kernels should override this method. The default
    /// CPU fallback preserves the public backend surface on platforms where a
    /// device kernel is unavailable or not yet implemented.
    fn bem_regular_pair_c32_batch(
        &self,
        pairs: &[BemRegularPairIn],
        wave_number: f32,
        harmonic_factor: f32,
    ) -> Vec<BemRegularPairOut> {
        crate::bem_cpu::regular_pair_c32_batch_from_gpu_abi(pairs, wave_number, harmonic_factor)
    }

    /// Checked BEM dispatch that rejects host fallback before doing work.
    ///
    /// The legacy infallible method remains available for CPU/reference code.
    /// Device-facing callers should use this method whenever a requested
    /// accelerator must not silently become a host computation.
    fn try_bem_regular_pair_c32_batch(
        &self,
        pairs: &[BemRegularPairIn],
        wave_number: f32,
        harmonic_factor: f32,
    ) -> BackendResult<Vec<BemRegularPairOut>> {
        self.capabilities().require_native_bem_regular_pair()?;
        validate_bem_regular_pair_inputs(pairs, wave_number, harmonic_factor)?;
        let output = catch_backend_panic("bem_regular_pair_c32_batch", || {
            self.bem_regular_pair_c32_batch(pairs, wave_number, harmonic_factor)
        })?;
        self.check_pending_error()?;
        if output.len() != pairs.len() {
            return Err(BackendError::OperationFailed {
                operation: "bem_regular_pair_c32_batch",
                reason: format!(
                    "backend returned {} results for {} input pairs",
                    output.len(),
                    pairs.len()
                ),
            });
        }
        validate_bem_regular_pair_outputs(&output)?;
        Ok(output)
    }

    /// Evaluate regular Helmholtz triangle-pair integrals in the required
    /// double-precision reference format.
    fn bem_regular_pair_c64_batch(
        &self,
        pairs: &[BemRegularPairInF64],
        wave_number: f64,
        harmonic_factor: f64,
    ) -> Vec<BemRegularPairOutF64> {
        crate::bem_cpu::regular_pair_f64_batch(pairs, wave_number, harmonic_factor)
    }

    /// Checked BEM dispatch for the double-precision pair kernel.  A caller
    /// requesting accelerator residency receives a typed error when the
    /// backend has no native Complex64 implementation; it never silently
    /// runs the CPU reference under a CUDA/Metal label.
    fn try_bem_regular_pair_c64_batch(
        &self,
        pairs: &[BemRegularPairInF64],
        wave_number: f64,
        harmonic_factor: f64,
    ) -> BackendResult<Vec<BemRegularPairOutF64>> {
        self.capabilities().require_native_bem_regular_pair_c64()?;
        validate_bem_regular_pair_f64_inputs(pairs, wave_number, harmonic_factor)?;
        let output = catch_backend_panic("bem_regular_pair_c64_batch", || {
            self.bem_regular_pair_c64_batch(pairs, wave_number, harmonic_factor)
        })?;
        self.check_pending_error()?;
        if output.len() != pairs.len() {
            return Err(BackendError::OperationFailed {
                operation: "bem_regular_pair_c64_batch",
                reason: format!(
                    "backend returned {} results for {} input pairs",
                    output.len(),
                    pairs.len()
                ),
            });
        }
        validate_bem_regular_pair_f64_outputs(&output)?;
        Ok(output)
    }

    /// Evaluate a constant-basis exterior field with all geometry, traces,
    /// points, and output buffers already resident on the backend.
    ///
    /// The default is deliberately unsupported. A caller asking for an
    /// accelerator path must not accidentally turn a device field phase into
    /// a host contraction merely because the backend lacks this operation.
    #[allow(clippy::too_many_arguments)]
    fn try_bem_field_c64_device(
        &self,
        elements: &Self::Buffer<BemFieldElementC64>,
        pressure: &Self::Buffer<Complex64>,
        velocity: &Self::Buffer<Complex64>,
        points: &Self::Buffer<BemFieldPointC64>,
        output: &mut Self::Buffer<Complex64>,
        wave_number: f64,
        harmonic_factor: f64,
        pressure_factor: f64,
    ) -> BackendResult<()> {
        let _ = (
            elements,
            pressure,
            velocity,
            points,
            output,
            wave_number,
            harmonic_factor,
            pressure_factor,
        );
        Err(BackendError::Unsupported {
            backend: self.capabilities().backend_name,
            operation: "bem_field_c64_device",
            scalar: "Complex64",
        })
    }
}

#[cfg(test)]
mod checked_projection_allocation_tests {
    use super::{Backend, BackendError, ReductionMode};
    use crate::cpu::CpuBackend;

    #[test]
    fn projection_reports_checked_coefficient_allocation_failure() {
        let backend = CpuBackend::new();
        let basis = backend
            .raw_alloc::<f64>(0)
            .expect("empty basis allocation should succeed");
        let mut y = backend
            .raw_alloc::<f64>(0)
            .expect("empty projection vector allocation should succeed");

        let error = backend
            .try_basis_project_out_strided_with_mode(
                ReductionMode::Deterministic,
                &basis,
                0,
                usize::MAX,
                &mut y,
            )
            .expect_err("coefficient staging overflow must remain typed");
        assert!(matches!(
            error,
            BackendError::Allocation { .. } | BackendError::InvalidArgument { .. }
        ));
    }
}
