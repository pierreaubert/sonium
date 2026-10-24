//! NVIDIA CUDA backend façade.
//!
//! The production NVIDIA backend is built on `cudarc` with official
//! cuBLAS/cuSPARSE/cuSOLVER bindings. The old oxicuda implementation is no
//! longer a selectable sonium-compute feature; keep oxicuda usage confined to
//! benchmark harnesses.

#[cfg(feature = "cuda")]
mod cublas;
#[cfg(feature = "cuda")]
pub mod dg;
#[cfg(feature = "cuda")]
pub mod explicit;
#[cfg(feature = "cuda")]
mod hmatrix;
#[cfg(feature = "cuda")]
pub mod fem_tet;
#[cfg(feature = "cuda")]
pub mod fem_hex;
#[cfg(feature = "cuda")]
pub mod quadrature_assembly;
#[cfg(feature = "cuda")]
mod mlfmm;
#[cfg(feature = "cuda")]
pub use cublas::{
    CudaAllocationLedger, CudaBackend, CudaDenseJacobi, CudaBemFieldOperatorC64, CudaBemQuadratureOperatorC64,
    CudaBemRegularPairOperatorC64, CudaBuffer, CudaDenseCholF64, CudaDenseLuC32, CudaDenseLuC64,
    CudaDenseLuF64, CudaDriverResourceStatsSnapshot, CudaGraphReplay, CudaRuntimeStatsSnapshot,
};
#[cfg(feature = "cuda")]
pub use hmatrix::{
    CudaDenseBlockC64Input, CudaHMatrixC64, CudaHMatrixWavenumberUpdateC64,
    CudaLowRankBlockC64Input,
};
#[cfg(feature = "cuda")]
pub use mlfmm::{
    CudaMlfmmC64, CudaMlfmmLevelC64Input, CudaMlfmmNearBlockC64Input,
    CudaMlfmmPhaseTransferC64Input, CudaMlfmmTransferC64Input, CudaMlfmmTranslationC64Input,
};

use crate::error::OomError;

/// Return the number of CUDA devices visible to the driver.
///
/// The cudarc loader reports missing driver libraries as a panic on some
/// platforms.  Keep that implementation detail behind the compute crate's
/// fallible boundary so language bindings can probe CUDA without risking an
/// interpreter crash.
#[cfg(feature = "cuda")]
pub fn cuda_device_count() -> Result<usize, OomError> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(try_cuda_device_count))
        .map_err(|payload| OomError::Invalid(format!("CUDA driver probe panicked: {payload:?}")))?
}

/// Return the number of CUDA devices through a strict typed boundary.
///
/// Unlike [`cuda_device_count`], this function does not catch panics from a
/// cudarc dynamic-library loader.  It is intended for checked applications
/// that own the process-level panic policy; missing drivers and invalid
/// counts are still returned as `OomError::Invalid`.
#[cfg(feature = "cuda")]
pub fn try_cuda_device_count() -> Result<usize, OomError> {
    cudarc::driver::CudaContext::device_count()
        .map_err(|error| OomError::Invalid(format!("CUDA device-count query failed: {error:?}")))
        .and_then(|count| {
            usize::try_from(count).map_err(|_| {
                OomError::Invalid(format!("CUDA driver returned invalid device count {count}"))
            })
        })
}
