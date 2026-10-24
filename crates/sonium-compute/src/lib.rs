//! Compute backend abstraction for sonium.
//!
//! This crate sits below `sonium-solvers`, `sonium-bem`, and `sonium-fem`. It
//! defines:
//!
//! - The [`Backend`] trait — the only thing upper crates depend on. A backend
//!   owns device-side buffers and exposes the BLAS / sparse / FFT primitives
//!   needed by Krylov solvers and matrix assembly.
//! - The [`MemoryPool`] type — the *only* path through which a backend hands
//!   out buffers. It enforces a hard byte cap before delegating to the driver,
//!   so we fail fast with [`OomError::QuotaExceeded`] instead of getting
//!   killed by a CUDA out-of-memory.
//! - A [`cpu`] backend (always available), plus feature-gated `cuda` (mature
//!   cuBLAS/cuSPARSE via cudarc) and `metal`. There is no `vulkan` feature:
//!   the reserved flag was removed until a backend exists, so requesting it
//!   fails the build instead of silently building to nothing.
//!
//! See `crates/sonium-compute/README.md` and `AGENTS.md` for the backend
//! contract, capability table, and profiling workflow.

pub mod backend;
pub mod bem;
pub mod bem_cpu;
pub mod block_krylov;
pub mod block_operator;
pub mod elementwise;
pub mod error;
pub mod explicit_dt;
pub mod explicit_hourglass;
pub mod explicit_restart;
pub mod explicit_scatter;
pub mod explicit_state;
pub mod fem_tet;
pub mod fem_hex;
pub mod quadrature_assembly;
pub mod fmm_cpu;
pub mod gmres;
pub mod gmres_ir;
pub mod host_array;
pub mod host_csr;
pub mod operator;
pub mod operator_sum;
pub mod operator_tape;
pub mod pcg;
pub mod perf;
pub mod pool;
pub mod precision;
pub mod preconditioner;
pub mod scalar;
pub mod scaled_storage;
pub mod scheduler;
pub mod segmented;
pub mod sme2;
pub mod tensor_contract;
pub mod transpose;

#[cfg(feature = "cpu")]
pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "metal")]
pub mod metal;

pub use backend::{
    Backend, BackendCapabilities, BackendCompletion, BackendMemoryStatsSnapshot,
    BackendTransferStats, BemCapabilities, BufferView, CapabilityState, CsrRowBucketStats,
    CsrSpmvFormat, CsrSpmvInspector, CudaFailureInjectionPoint, DeviceBufferLike, GemmBatch,
    GemvBatch, PrecisionCastCapabilities, PreconditionerFailureInjectionPoint, RealTransformKind,
    ReductionAccumulation, ReductionCapabilities, ReductionFailureGuarantee, ReductionMode,
    ReductionOrdering, ReductionPolicy, ReductionSynchronization, ScalarCapabilities,
    TransferStatsSnapshot, Transpose, ViewAliasing, ViewDescriptor, ViewLayout,
    checked_fft_strided_view_descriptor, checked_fft_view_descriptor,
};
pub use bem::{
    BemFieldElementC64, BemFieldPointC64, BemQuadraturePairC64, BemQuadratureSampleC64,
    BemRegularPairIn, BemRegularPairInF64, BemRegularPairOut, BemRegularPairOutF64,
    BemRegularPairOutRealF64,
};
pub use block_krylov::{
    BlockGmresConfig, BlockGmresError, BlockGmresStatus, BlockGmresWorkspace,
    gmres_backend_block_preconditioned, gmres_backend_block_preconditioned_with_workspace,
};
pub use block_operator::{
    BlockLayout, DeviceBlock, DeviceBlockOperator, DeviceBlockWorkspace, SquareDeviceBlock,
};
#[cfg(feature = "cpu")]
pub use cpu::CpuBackend;
/// Portable CPU backend exposed for WebAssembly builds.
#[cfg(feature = "wasm")]
pub type WasmBackend = cpu::CpuBackend;
pub use elementwise::{ElementwiseError, pointwise_mul, scale_real};
pub use error::{
    BackendContextDisposition, BackendError, BackendOutputDisposition, BackendResult, OomError,
};
pub use gmres::{
    BackendCsr, BackendOperator, BackendPreconditioner, GmresBackendConfig, GmresBackendError,
    GmresBackendStatus, GmresWorkspace, IdentityBackendPreconditioner,
    fgmres_backend_preconditioned_pooled_with_workspace, gmres_backend_preconditioned,
    gmres_backend_preconditioned_pooled,
    gmres_backend_preconditioned_pooled_with_workspace,
};
pub use gmres_ir::{
    BackendCsrF32, BackendPreconditionerF32, GmresInnerConfig, GmresInnerError,
    GmresInnerF32Config, GmresInnerF32Error, GmresInnerF32Status, GmresInnerStatus,
    GmresIrAccuracy, GmresIrConfig, GmresIrError, GmresIrStatus, IdentityBackendPreconditionerF32,
    gmres_ir_backend_preconditioned,
};
pub use host_array::{HostMatrix, HostTensor3};
pub use host_csr::{HostCsrError, HostCsrRef, OwnedHostCsr};
pub use operator::{
    AdjointDeviceCsr, DeviceAdjointCapability, DeviceAdjointOperator, DeviceCsr, DeviceOperator,
    DevicePreconditioner, DevicePreconditionerDiagnostics, DevicePreconditionerDiagnosticsSnapshot,
    DevicePreconditionerFailureStage, IdentityDevicePreconditioner, PersistentDeviceCsr,
    PooledDeviceCsr, PrebakedAdjointSpmv, PrebakedSpmv,
};
pub use operator_tape::{
    DeviceOperatorTape, DeviceTapeGradients, OperatorTape, TapeBudget, TapeError, TapeGradients,
    TapeMetrics, TapeNodeMetadata, TapeValueId,
};
pub use pcg::{
    PcgBackendConfig, PcgBackendError, PcgBackendStatus, PcgBackendWorkspace,
    pcg_backend_preconditioned,
};
pub use perf::PerfModel;
pub use pool::{Buffer, MemoryPool, MemoryPoolConfig};
pub use precision::{MixedPrecisionGateError, MixedPrecisionPolicy};
pub use preconditioner::{
    BlockJacobiBackend, ChebyshevBackend, ChebyshevBackendF32, DEFAULT_BLOCK_SIZE,
    PointJacobiBackend, PolynomialJacobiBackend, PolynomialJacobiBackendF32,
};
pub use scalar::{BufferElement, DeviceScalar, LowerPrecision};
pub use scaled_storage::{
    ScaledStorageElement, ScaledStorageMeta, dequantize_scaled_complex32, dequantize_scaled_f32,
    quantize_scaled_complex32, quantize_scaled_f32,
};
pub use scheduler::{
    BackendSubmission, DeviceBudget, DeviceScheduler, DeviceSnapshot, ScheduleError,
    WorkCompletion, WorkLease, WorkRequest,
};
pub use segmented::{SegmentedReductionError, segmented_sum};
pub use tensor_contract::{LocalContractError, batched_local_contract};
pub use transpose::{TransposeError, transpose_2d};

#[cfg(feature = "cuda")]
pub use cuda::{
    CudaAllocationLedger, CudaBackend, CudaDenseJacobi, CudaBuffer, CudaMlfmmC64, CudaMlfmmLevelC64Input,
    CudaMlfmmNearBlockC64Input, CudaMlfmmPhaseTransferC64Input, CudaMlfmmTransferC64Input,
    CudaMlfmmTranslationC64Input, cuda_device_count, try_cuda_device_count,
};
#[cfg(feature = "cuda")]
pub use cuda::{
    CudaBemFieldOperatorC64, CudaBemQuadratureOperatorC64, CudaBemRegularPairOperatorC64,
    CudaDenseBlockC64Input, CudaDenseCholF64, CudaDenseLuC32, CudaDenseLuC64, CudaDenseLuF64,
    CudaDriverResourceStatsSnapshot, CudaGraphReplay, CudaHMatrixC64,
    CudaHMatrixWavenumberUpdateC64, CudaLowRankBlockC64Input, CudaRuntimeStatsSnapshot,
};

#[cfg(feature = "metal")]
pub use metal::{MetalBackend, MetalBuffer};
