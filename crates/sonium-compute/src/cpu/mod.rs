//! CPU backend — host-only buffers backed by [`Vec<T>`].
//!
//! BLAS-1 (`axpy`, `dot`, `nrm2`, `scale`) routes through the in-house
//! [`crate::cpu::blas1`] kernels — NEON intrinsics on AArch64, AVX2/FMA on
//! x86_64, scalar multi-accumulator fallback elsewhere. SpMV-CSR uses a
//! row-bucket inspector; medium/long real rows route through explicit SIMD
//! reducers when the host ISA supports gathers.
//!
//! ## Why not SME2 here?
//!
//! ARM SME2 accelerates dense matrix-multiply (FMOPA tile ops). The
//! Backend trait is BLAS-1 + sparse, both of which are bandwidth-bound,
//! not compute-bound — SME2 doesn't apply. The SME2 path lives in
//! `sonium-compute::sme2` and is invoked from dense GEMM/GEMV callers.

pub mod blas1;
pub mod blas2;
pub mod blas3;
pub(crate) mod parallelism;
pub(crate) mod spin_pool;
pub mod spmv_simd;

use crate::backend::{
    Backend, BackendCapabilities, BackendTransferStats, BemCapabilities, CapabilityState,
    CsrRowBucketStats, CsrSpmvFormat, CsrSpmvInspector, DeviceBufferLike,
    PrecisionCastCapabilities, PreconditionerFailureInjectionPoint, ReductionCapabilities,
    ScalarCapabilities, TransferStatsSnapshot, Transpose, catch_backend_panic,
    checked_strided_span, host_copy_h2d_cast, injected_preconditioner_error,
};
use crate::bem::{BemRegularPairIn, BemRegularPairInF64, BemRegularPairOut, BemRegularPairOutF64};
use crate::error::{BackendError, BackendResult, OomError};
use crate::scalar::{BufferElement, DeviceScalar, LowerPrecision};
use num_complex::{Complex32, Complex64};
use num_traits::{Float, Zero};
use rayon::prelude::*;
use std::any::TypeId;
use std::cell::Cell;
use std::sync::OnceLock;

/// Disable native CPU kernels for a whole process when collecting scalar
/// reference evidence.  Reading this once prevents a single workload from
/// mixing dispatch modes if its environment is changed concurrently.
#[inline]
pub(crate) fn force_scalar_dispatch() -> bool {
    static FORCE_SCALAR: OnceLock<bool> = OnceLock::new();
    *FORCE_SCALAR.get_or_init(|| {
        matches!(
            std::env::var("SONIUM_COMPUTE_FORCE_SCALAR").as_deref(),
            Ok("1") | Ok("true") | Ok("yes")
        )
    })
}

/// Host buffer — just a `Vec<T>`.
pub struct HostBuffer<T: BufferElement> {
    data: Vec<T>,
}

/// CPU-side CSR descriptor — borrows the source buffers and remembers
/// the dimensions. Construction cost is `O(1)`; the inner-loop SpMV
/// path runs the same scalar code as the legacy `spmv_csr`.
pub struct CpuCsrPrebaked<'a, T: DeviceScalar> {
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) row_ptr: &'a HostBuffer<i32>,
    pub(crate) col_idx: &'a HostBuffer<i32>,
    pub(crate) values: &'a HostBuffer<T>,
    pub(crate) buckets: CpuCsrRowBuckets,
    pub(crate) inspector: CsrSpmvInspector,
}

pub(crate) struct CpuCsrRowBuckets {
    pub(crate) short: Vec<usize>,
    pub(crate) simd: Vec<usize>,
    pub(crate) cta: Vec<usize>,
}

impl CpuCsrRowBuckets {
    fn from_row_ptr(row_ptr: &[i32]) -> Self {
        let rows = row_ptr.len().saturating_sub(1);
        let mut buckets = Self {
            short: Vec::new(),
            simd: Vec::new(),
            cta: Vec::new(),
        };
        for row in 0..rows {
            let nnz = row_ptr[row + 1]
                .checked_sub(row_ptr[row])
                .and_then(|value| usize::try_from(value).ok())
                .unwrap_or(0);
            if nnz == 0 {
                continue;
            }
            if nnz < CsrRowBucketStats::SHORT_ROW_NNZ {
                buckets.short.push(row);
            } else if nnz < CsrRowBucketStats::CTA_ROW_NNZ {
                buckets.simd.push(row);
            } else {
                buckets.cta.push(row);
            }
        }
        buckets
    }
}

impl<T: DeviceScalar> CpuCsrPrebaked<'_, T> {
    pub fn inspector(&self) -> CsrSpmvInspector {
        self.inspector
    }

    pub fn row_bucket_stats(&self) -> CsrRowBucketStats {
        self.inspector.row_buckets
    }

    pub fn bucket_row_counts(&self) -> (usize, usize, usize) {
        (
            self.buckets.short.len(),
            self.buckets.simd.len(),
            self.buckets.cta.len(),
        )
    }
}

unsafe impl<T: DeviceScalar> Send for CpuCsrPrebaked<'_, T> {}
unsafe impl<T: DeviceScalar> Sync for CpuCsrPrebaked<'_, T> {}

impl<T: BufferElement> HostBuffer<T> {
    /// View as a slice.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// View as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T: BufferElement> DeviceBufferLike<T> for HostBuffer<T> {
    fn len(&self) -> usize {
        self.data.len()
    }
}

/// CPU backend. Constructed with [`CpuBackend::new`]. Stateless.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

// CPU buffers are host allocations, but they still use the backend transfer
// contract so residency tests can detect accidental staging. Keep the
// counters thread-local: a process-wide counter makes one test's reset race
// with unrelated parallel CPU tests and makes a per-workload measurement
// meaningless. CPU backends are synchronous, so the thread performing the
// checked copy is the owner of that observation.
thread_local! {
    static CPU_TRANSFER_STATS: BackendTransferStats = const { BackendTransferStats::new() };
}
#[cfg(test)]
thread_local! {
    static CPU_PENDING_ERROR: Cell<bool> = const { Cell::new(false) };
}
thread_local! {
    static CPU_PRECONDITIONER_FAILURE: Cell<Option<PreconditionerFailureInjectionPoint>> =
        const { Cell::new(None) };
}

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }

    /// Arm a one-shot checked preconditioner failure on the current thread.
    ///
    /// This is a deterministic fault harness for the shared backend
    /// contract. It is also useful in downstream tests that need to verify
    /// transactional preconditioner behavior without a CUDA device.
    pub fn inject_preconditioner_failure(point: PreconditionerFailureInjectionPoint) {
        CPU_PRECONDITIONER_FAILURE.with(|failure| failure.set(Some(point)));
    }

    /// Clear an armed CPU preconditioner fault without consuming it.
    pub fn clear_preconditioner_failure() {
        CPU_PRECONDITIONER_FAILURE.with(|failure| failure.set(None));
    }

    #[cfg(test)]
    pub(crate) fn inject_pending_error() {
        CPU_PENDING_ERROR.with(|pending| pending.set(true));
    }

    /// CPU implementation of the shared BEM regular-pair f32 ABI.
    pub fn bem_regular_pair_c32_batch(
        &self,
        pairs: &[BemRegularPairIn],
        wave_number: f32,
        harmonic_factor: f32,
    ) -> Vec<BemRegularPairOut> {
        crate::bem_cpu::regular_pair_c32_batch_from_gpu_abi(pairs, wave_number, harmonic_factor)
    }

    /// CPU implementation of the double-precision BEM pair ABI.
    pub fn bem_regular_pair_c64_batch(
        &self,
        pairs: &[BemRegularPairInF64],
        wave_number: f64,
        harmonic_factor: f64,
    ) -> Vec<BemRegularPairOutF64> {
        crate::bem_cpu::regular_pair_f64_batch(pairs, wave_number, harmonic_factor)
    }
}

// ---------------------------------------------------------------------------
// TypeId-branched BLAS-1 dispatch
//
// The Backend trait is generic over `T: DeviceScalar`. The in-house
// `cpu::blas1` kernels are specialised per-precision (`axpy_f32`,
// `axpy_f64`, `dot_f64`, ...). We branch on TypeId at runtime; the
// branches with no compile-time match are removed by the compiler
// (the cast inside the branch is monomorphised only when `T` matches),
// so this is a zero-cost dispatch in release builds.
// ---------------------------------------------------------------------------

/// SAFETY: caller must have verified `TypeId::of::<T>() == TypeId::of::<U>()`.
/// Reinterprets `&[T]` as `&[U]` — both must be POD with identical layout.
#[inline]
unsafe fn slice_cast<T, U>(s: &[T]) -> &[U] {
    debug_assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<U>());
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const U, s.len()) }
}

#[inline]
unsafe fn slice_cast_mut<T, U>(s: &mut [T]) -> &mut [U] {
    debug_assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<U>());
    unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut U, s.len()) }
}

/// Reinterpret a generic `T` value as `U`. SAFETY: layouts must match.
#[inline]
unsafe fn val_cast<T: Copy, U: Copy>(v: T) -> U {
    debug_assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<U>());
    unsafe { *(&v as *const T as *const U) }
}

// Machine-adaptive SpMV/SpMM gates live in `parallelism`: row (or output)
// counts scale with thread count instead of a fixed crossover. The serial
// and parallel paths accumulate each output row serially within one worker,
// so they are bitwise identical.

#[inline]
fn spmv_csr_row_f32(row_ptr: &[i32], col_idx: &[i32], values: &[f32], x: &[f32], i: usize) -> f32 {
    let start = row_ptr[i] as usize;
    let end = row_ptr[i + 1] as usize;
    let mut sum0 = 0.0_f32;
    let mut sum1 = 0.0_f32;
    let mut sum2 = 0.0_f32;
    let mut sum3 = 0.0_f32;
    let mut k = start;
    while k + 4 <= end {
        sum0 += values[k] * x[col_idx[k] as usize];
        sum1 += values[k + 1] * x[col_idx[k + 1] as usize];
        sum2 += values[k + 2] * x[col_idx[k + 2] as usize];
        sum3 += values[k + 3] * x[col_idx[k + 3] as usize];
        k += 4;
    }
    let mut acc = (sum0 + sum1) + (sum2 + sum3);
    while k < end {
        acc += values[k] * x[col_idx[k] as usize];
        k += 1;
    }
    acc
}

#[inline]
fn spmv_csr_row_f64(row_ptr: &[i32], col_idx: &[i32], values: &[f64], x: &[f64], i: usize) -> f64 {
    let start = row_ptr[i] as usize;
    let end = row_ptr[i + 1] as usize;
    let mut sum0 = 0.0_f64;
    let mut sum1 = 0.0_f64;
    let mut sum2 = 0.0_f64;
    let mut sum3 = 0.0_f64;
    let mut k = start;
    while k + 4 <= end {
        sum0 += values[k] * x[col_idx[k] as usize];
        sum1 += values[k + 1] * x[col_idx[k + 1] as usize];
        sum2 += values[k + 2] * x[col_idx[k + 2] as usize];
        sum3 += values[k + 3] * x[col_idx[k + 3] as usize];
        k += 4;
    }
    let mut acc = (sum0 + sum1) + (sum2 + sum3);
    while k < end {
        acc += values[k] * x[col_idx[k] as usize];
        k += 1;
    }
    acc
}

#[inline]
fn spmv_csr_f32(row_ptr: &[i32], col_idx: &[i32], values: &[f32], x: &[f32], y: &mut [f32]) {
    if parallelism::use_compute_parallel(y.len(), parallelism::SPMV_ROWS_PER_THREAD) {
        y.par_iter_mut().enumerate().for_each(|(i, yi)| {
            *yi = spmv_csr_row_f32(row_ptr, col_idx, values, x, i);
        });
    } else {
        for (i, yi) in y.iter_mut().enumerate() {
            *yi = spmv_csr_row_f32(row_ptr, col_idx, values, x, i);
        }
    }
}

#[inline]
fn spmv_csr_f64(row_ptr: &[i32], col_idx: &[i32], values: &[f64], x: &[f64], y: &mut [f64]) {
    if parallelism::use_compute_parallel(y.len(), parallelism::SPMV_ROWS_PER_THREAD) {
        y.par_iter_mut().enumerate().for_each(|(i, yi)| {
            *yi = spmv_csr_row_f64(row_ptr, col_idx, values, x, i);
        });
    } else {
        for (i, yi) in y.iter_mut().enumerate() {
            *yi = spmv_csr_row_f64(row_ptr, col_idx, values, x, i);
        }
    }
}

#[repr(transparent)]
struct OutputRows<T>(*mut T);

unsafe impl<T: Send> Sync for OutputRows<T> {}

impl<T> OutputRows<T> {
    unsafe fn write(&self, row: usize, value: T) {
        unsafe {
            *self.0.add(row) = value;
        }
    }
}

#[inline]
fn spmv_csr_bucketed_f32(
    buckets: &CpuCsrRowBuckets,
    row_ptr: &[i32],
    col_idx: &[i32],
    values: &[f32],
    x: &[f32],
    y: &mut [f32],
) {
    y.fill(0.0);
    // Each output row is accumulated serially within one worker, so the
    // serial fallback below is bitwise identical to the parallel path.
    let row_count = buckets
        .short
        .len()
        .saturating_add(buckets.simd.len())
        .saturating_add(buckets.cta.len());
    if !parallelism::use_compute_parallel(row_count, parallelism::SPMV_ROWS_PER_THREAD) {
        for &row in buckets.short.iter() {
            y[row] = spmv_csr_row_f32(row_ptr, col_idx, values, x, row);
        }
        for rows in [&buckets.simd, &buckets.cta] {
            for &row in rows.iter() {
                let start = row_ptr[row] as usize;
                let end = row_ptr[row + 1] as usize;
                y[row] = spmv_simd::row_f32(col_idx, values, x, start, end);
            }
        }
        return;
    }
    let out = OutputRows(y.as_mut_ptr());
    buckets.short.par_iter().for_each(|&row| {
        let value = spmv_csr_row_f32(row_ptr, col_idx, values, x, row);
        unsafe { out.write(row, value) };
    });
    for rows in [&buckets.simd, &buckets.cta] {
        rows.par_iter().for_each(|&row| {
            let start = row_ptr[row] as usize;
            let end = row_ptr[row + 1] as usize;
            let value = spmv_simd::row_f32(col_idx, values, x, start, end);
            unsafe { out.write(row, value) };
        });
    }
}

#[inline]
fn spmv_csr_bucketed_f64(
    buckets: &CpuCsrRowBuckets,
    row_ptr: &[i32],
    col_idx: &[i32],
    values: &[f64],
    x: &[f64],
    y: &mut [f64],
) {
    y.fill(0.0);
    // Each output row is accumulated serially within one worker, so the
    // serial fallback below is bitwise identical to the parallel path.
    let row_count = buckets
        .short
        .len()
        .saturating_add(buckets.simd.len())
        .saturating_add(buckets.cta.len());
    if !parallelism::use_compute_parallel(row_count, parallelism::SPMV_ROWS_PER_THREAD) {
        for &row in buckets.short.iter() {
            y[row] = spmv_csr_row_f64(row_ptr, col_idx, values, x, row);
        }
        for rows in [&buckets.simd, &buckets.cta] {
            for &row in rows.iter() {
                let start = row_ptr[row] as usize;
                let end = row_ptr[row + 1] as usize;
                y[row] = spmv_simd::row_f64(col_idx, values, x, start, end);
            }
        }
        return;
    }
    let out = OutputRows(y.as_mut_ptr());
    buckets.short.par_iter().for_each(|&row| {
        let value = spmv_csr_row_f64(row_ptr, col_idx, values, x, row);
        unsafe { out.write(row, value) };
    });
    for rows in [&buckets.simd, &buckets.cta] {
        rows.par_iter().for_each(|&row| {
            let start = row_ptr[row] as usize;
            let end = row_ptr[row + 1] as usize;
            let value = spmv_simd::row_f64(col_idx, values, x, start, end);
            unsafe { out.write(row, value) };
        });
    }
}

impl Backend for CpuBackend {
    type Buffer<T: BufferElement> = HostBuffer<T>;
    type CsrPrebaked<'a, T: DeviceScalar>
        = CpuCsrPrebaked<'a, T>
    where
        Self: 'a;

    fn capabilities(&self) -> BackendCapabilities {
        let backend_name = if cfg!(target_arch = "wasm32") {
            "wasm-cpu"
        } else {
            "cpu"
        };
        BackendCapabilities {
            backend_name,
            blas1: ScalarCapabilities::splat(CapabilityState::Native),
            gemv: ScalarCapabilities::splat(CapabilityState::Native),
            gemm: ScalarCapabilities::splat(CapabilityState::Native),
            trsm: ScalarCapabilities::splat(CapabilityState::Native),
            syrk: ScalarCapabilities::splat(CapabilityState::Native),
            fft: ScalarCapabilities::splat(CapabilityState::Unsupported),
            real_fft: ScalarCapabilities::splat(CapabilityState::Unsupported),
            dct: ScalarCapabilities::splat(CapabilityState::Unsupported),
            dst: ScalarCapabilities::splat(CapabilityState::Unsupported),
            csr_spmv: ScalarCapabilities::splat(CapabilityState::Native),
            csr_spmm: ScalarCapabilities::splat(CapabilityState::Native),
            csr_transpose: ScalarCapabilities::splat(CapabilityState::Native),
            axpy_many: ScalarCapabilities::splat(CapabilityState::Native),
            multi_dot: ScalarCapabilities::splat(CapabilityState::Native),
            multi_dot_strided: ScalarCapabilities::splat(CapabilityState::Native),
            basis_axpy_strided: ScalarCapabilities::splat(CapabilityState::Native),
            basis_project_out_strided: ScalarCapabilities::splat(CapabilityState::Native),
            gather4_weighted: ScalarCapabilities::splat(CapabilityState::Native),
            segmented_reduce: ScalarCapabilities::splat(CapabilityState::Native),
            batched_local_contract: ScalarCapabilities::splat(CapabilityState::Native),
            reduction: ReductionCapabilities::splat(CapabilityState::Native),
            precision_cast: PrecisionCastCapabilities::splat(CapabilityState::Native),
            bem_regular_pair_c32: CapabilityState::Native,
            bem_regular_pair_c64: CapabilityState::Native,
            bem: BemCapabilities {
                regular_pair_c32: CapabilityState::Native,
                regular_pair_c64: CapabilityState::Native,
                near_pair: ScalarCapabilities::new(
                    CapabilityState::Unsupported,
                    CapabilityState::Unsupported,
                    CapabilityState::Unsupported,
                    CapabilityState::Native,
                ),
                singular_pair: ScalarCapabilities::new(
                    CapabilityState::Unsupported,
                    CapabilityState::Unsupported,
                    CapabilityState::Unsupported,
                    CapabilityState::Native,
                ),
                hypersingular_pair: ScalarCapabilities::new(
                    CapabilityState::Unsupported,
                    CapabilityState::Unsupported,
                    CapabilityState::Unsupported,
                    CapabilityState::Native,
                ),
                field_evaluation: ScalarCapabilities::new(
                    CapabilityState::Unsupported,
                    CapabilityState::Unsupported,
                    CapabilityState::Unsupported,
                    CapabilityState::Native,
                ),
                hmatrix_setup: ScalarCapabilities::new(
                    CapabilityState::Unsupported,
                    CapabilityState::Unsupported,
                    CapabilityState::Unsupported,
                    CapabilityState::Native,
                ),
                hmatrix: ScalarCapabilities::new(
                    CapabilityState::Unsupported,
                    CapabilityState::Unsupported,
                    CapabilityState::Unsupported,
                    CapabilityState::Native,
                ),
                fmm_setup: ScalarCapabilities::new(
                    CapabilityState::Unsupported,
                    CapabilityState::Unsupported,
                    CapabilityState::Unsupported,
                    CapabilityState::Native,
                ),
                fmm: ScalarCapabilities::new(
                    CapabilityState::Unsupported,
                    CapabilityState::Unsupported,
                    CapabilityState::Unsupported,
                    CapabilityState::Native,
                ),
            },
        }
    }

    fn transfer_stats(&self) -> TransferStatsSnapshot {
        CPU_TRANSFER_STATS.with(BackendTransferStats::snapshot)
    }

    fn reset_transfer_stats(&self) {
        CPU_TRANSFER_STATS.with(BackendTransferStats::reset);
    }

    fn raw_alloc<T: BufferElement>(&self, len: usize) -> Result<Self::Buffer<T>, OomError> {
        let mut v = Vec::new();
        v.try_reserve_exact(len)
            .map_err(|error| OomError::DriverOom(error.to_string()))?;
        // SAFETY: T is one of f32/f64/Complex32/Complex64/i32/u32/usize per
        // the BufferElement impls in `scalar.rs`; all have a valid all-zero
        // representation.
        unsafe {
            std::ptr::write_bytes(v.as_mut_ptr(), 0, len);
            v.set_len(len);
        }
        Ok(HostBuffer { data: v })
    }

    unsafe fn raw_alloc_uninitialized<T: BufferElement>(
        &self,
        len: usize,
    ) -> Result<Self::Buffer<T>, OomError> {
        let mut storage: Vec<std::mem::MaybeUninit<T>> = Vec::new();
        storage
            .try_reserve_exact(len)
            .map_err(|error| OomError::DriverOom(error.to_string()))?;
        // SAFETY: this method is unsafe because the elements are not
        // initialized. `BufferElement` excludes Drop types, so dropping this
        // vector before initialization is valid; reading before writing is not.
        unsafe {
            storage.set_len(len);
        }
        let mut storage = std::mem::ManuallyDrop::new(storage);
        // SAFETY: `MaybeUninit<T>` has the same layout as `T`, and this
        // unsafe API requires the caller to initialize all elements before
        // reading them.
        let data = unsafe {
            Vec::from_raw_parts(storage.as_mut_ptr().cast::<T>(), len, storage.capacity())
        };
        Ok(HostBuffer { data })
    }

    fn total_device_memory_bytes(&self) -> Option<usize> {
        None
    }

    fn copy_h2d<T: BufferElement>(&self, host: &[T], dev: &mut Self::Buffer<T>) {
        assert_eq!(host.len(), dev.data.len(), "copy_h2d length mismatch");
        dev.data.copy_from_slice(host);
        CPU_TRANSFER_STATS.with(|stats| stats.record_h2d(std::mem::size_of_val(host)));
    }

    fn copy_h2d_cast<Upper>(&self, host: &[Upper], dev: &mut Self::Buffer<Upper::Lower>)
    where
        Upper: LowerPrecision,
    {
        if let Err(error) = host_copy_h2d_cast(self, host, dev) {
            log::error!(target: "sonium_compute::cpu", "copy_h2d_cast failed: {error}");
        }
    }

    fn copy_d2h<T: BufferElement>(&self, dev: &Self::Buffer<T>, host: &mut [T]) {
        assert_eq!(host.len(), dev.data.len(), "copy_d2h length mismatch");
        host.copy_from_slice(&dev.data);
        CPU_TRANSFER_STATS.with(|stats| stats.record_d2h(std::mem::size_of_val(host)));
    }

    fn copy_d2d<T: BufferElement>(&self, src: &Self::Buffer<T>, dst: &mut Self::Buffer<T>) {
        assert_eq!(src.data.len(), dst.data.len(), "copy_d2d length mismatch");
        dst.data.copy_from_slice(&src.data);
        CPU_TRANSFER_STATS
            .with(|stats| stats.record_d2d(std::mem::size_of_val(src.data.as_slice())));
    }

    fn try_copy_view_d2h<T: BufferElement>(
        &self,
        buffer: &Self::Buffer<T>,
        view: crate::BufferView,
        host: &mut [T],
    ) -> BackendResult<()> {
        self.checked_view_descriptor(buffer, view.offset, view.len, view.stride)?;
        if host.len() != view.len {
            return Err(BackendError::InvalidArgument {
                operation: "copy_view_d2h",
                reason: format!("host length {} != view length {}", host.len(), view.len),
            });
        }
        for (index, value) in host.iter_mut().enumerate() {
            *value = buffer.data[view.checked_element_index(index)?];
        }
        CPU_TRANSFER_STATS.with(|stats| stats.record_d2h(std::mem::size_of_val(host)));
        Ok(())
    }

    fn try_copy_view_h2d<T: BufferElement>(
        &self,
        host: &[T],
        buffer: &mut Self::Buffer<T>,
        view: crate::BufferView,
    ) -> BackendResult<()> {
        self.checked_view_descriptor(buffer, view.offset, view.len, view.stride)?;
        if host.len() != view.len {
            return Err(BackendError::InvalidArgument {
                operation: "copy_view_h2d",
                reason: format!("host length {} != view length {}", host.len(), view.len),
            });
        }
        for (index, &value) in host.iter().enumerate() {
            let destination = view.checked_element_index(index)?;
            buffer.data[destination] = value;
        }
        CPU_TRANSFER_STATS.with(|stats| stats.record_h2d(std::mem::size_of_val(host)));
        Ok(())
    }

    fn try_copy_views_d2d<T: BufferElement>(
        &self,
        source: &Self::Buffer<T>,
        source_view: crate::BufferView,
        destination: &mut Self::Buffer<T>,
        destination_view: crate::BufferView,
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
            crate::ViewAliasing::OverlapSafe,
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
        for index in 0..source_view.len {
            let source_index = source_view.checked_element_index(index)?;
            let destination_index = destination_view.checked_element_index(index)?;
            destination.data[destination_index] = source.data[source_index];
        }
        CPU_TRANSFER_STATS
            .with(|stats| stats.record_d2d(source_view.len * std::mem::size_of::<T>()));
        Ok(())
    }

    fn synchronize(&self) -> BackendResult<()> {
        // CPU is synchronous.
        Ok(())
    }

    fn axpy<T: DeviceScalar>(&self, alpha: T, x: &Self::Buffer<T>, y: &mut Self::Buffer<T>) {
        assert_eq!(x.data.len(), y.data.len(), "axpy length mismatch");
        if x.data.is_empty() {
            return;
        }
        // SAFETY: each branch has verified TypeId equality, so the
        // slice/value casts are layout-safe (POD + identical size).
        unsafe {
            if TypeId::of::<T>() == TypeId::of::<f32>() {
                blas1::axpy_f32(
                    val_cast::<T, f32>(alpha),
                    slice_cast::<T, f32>(&x.data),
                    slice_cast_mut::<T, f32>(&mut y.data),
                );
            } else if TypeId::of::<T>() == TypeId::of::<f64>() {
                blas1::axpy_f64(
                    val_cast::<T, f64>(alpha),
                    slice_cast::<T, f64>(&x.data),
                    slice_cast_mut::<T, f64>(&mut y.data),
                );
            } else if TypeId::of::<T>() == TypeId::of::<Complex32>() {
                blas1::axpy_c32(
                    val_cast::<T, Complex32>(alpha),
                    slice_cast::<T, Complex32>(&x.data),
                    slice_cast_mut::<T, Complex32>(&mut y.data),
                );
            } else if TypeId::of::<T>() == TypeId::of::<Complex64>() {
                blas1::axpy_c64(
                    val_cast::<T, Complex64>(alpha),
                    slice_cast::<T, Complex64>(&x.data),
                    slice_cast_mut::<T, Complex64>(&mut y.data),
                );
            } else {
                // i32 / u32 / usize fallback — never used by Krylov but
                // keeps the trait honest.
                for (xi, yi) in x.data.iter().zip(y.data.iter_mut()) {
                    *yi = *yi + alpha * *xi;
                }
            }
        }
    }

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
        if alphas.is_empty() || y.data.is_empty() {
            return;
        }
        for x in x_basis {
            assert_eq!(
                x.data.len(),
                y.data.len(),
                "axpy_many basis length mismatch"
            );
        }
        // SAFETY: TypeId equality verified in each branch; layouts match.
        unsafe {
            if TypeId::of::<T>() == TypeId::of::<f32>() {
                let alphas_f32: Vec<f32> = alphas.iter().map(|a| val_cast::<T, f32>(*a)).collect();
                let basis_refs: Vec<&[f32]> = x_basis
                    .iter()
                    .map(|x| slice_cast::<T, f32>(&x.data))
                    .collect();
                blas1::axpy_many_f32(
                    &alphas_f32,
                    &basis_refs,
                    slice_cast_mut::<T, f32>(&mut y.data),
                );
            } else if TypeId::of::<T>() == TypeId::of::<f64>() {
                let alphas_f64: Vec<f64> = alphas.iter().map(|a| val_cast::<T, f64>(*a)).collect();
                let basis_refs: Vec<&[f64]> = x_basis
                    .iter()
                    .map(|x| slice_cast::<T, f64>(&x.data))
                    .collect();
                blas1::axpy_many_f64(
                    &alphas_f64,
                    &basis_refs,
                    slice_cast_mut::<T, f64>(&mut y.data),
                );
            } else if TypeId::of::<T>() == TypeId::of::<Complex32>() {
                let alphas_c32: Vec<Complex32> = alphas
                    .iter()
                    .map(|a| val_cast::<T, Complex32>(*a))
                    .collect();
                let basis_refs: Vec<&[Complex32]> = x_basis
                    .iter()
                    .map(|x| slice_cast::<T, Complex32>(&x.data))
                    .collect();
                blas1::axpy_many_c32(
                    &alphas_c32,
                    &basis_refs,
                    slice_cast_mut::<T, Complex32>(&mut y.data),
                );
            } else if TypeId::of::<T>() == TypeId::of::<Complex64>() {
                let alphas_c64: Vec<Complex64> = alphas
                    .iter()
                    .map(|a| val_cast::<T, Complex64>(*a))
                    .collect();
                let basis_refs: Vec<&[Complex64]> = x_basis
                    .iter()
                    .map(|x| slice_cast::<T, Complex64>(&x.data))
                    .collect();
                blas1::axpy_many_c64(
                    &alphas_c64,
                    &basis_refs,
                    slice_cast_mut::<T, Complex64>(&mut y.data),
                );
            } else {
                // Fallback for i32/u32/usize — never used by Krylov.
                for (row, yi) in y.data.iter_mut().enumerate() {
                    let mut acc = *yi;
                    for (&alpha, x) in alphas.iter().zip(x_basis.iter()) {
                        acc = acc + alpha * x.data[row];
                    }
                    *yi = acc;
                }
            }
        }
    }

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
                target: "sonium_compute::cpu",
                "basis_axpy_strided rejected coefficient length {} (expected {basis_count})",
                coeffs.len()
            );
            return;
        }
        if y.data.len() != vector_len {
            log::error!(
                target: "sonium_compute::cpu",
                "basis_axpy_strided rejected y length {} (expected {vector_len})",
                y.data.len()
            );
            return;
        }
        let Some(required) = vector_len.checked_mul(basis_count) else {
            log::error!(
                target: "sonium_compute::cpu",
                "basis_axpy_strided basis shape {basis_count}x{vector_len} overflows"
            );
            return;
        };
        if basis.data.len() < required {
            log::error!(
                target: "sonium_compute::cpu",
                "basis_axpy_strided rejected basis length {} (expected at least {required})",
                basis.data.len()
            );
            return;
        }
        if !alpha.is_finite()
            || coeffs.iter().any(|value| !value.is_finite())
            || basis.data[..required]
                .iter()
                .any(|value| !value.is_finite())
            || y.data.iter().any(|value| !value.is_finite())
        {
            log::error!(
                target: "sonium_compute::cpu",
                "basis_axpy_strided rejected non-finite input"
            );
            return;
        }
        if basis_count == 0 || vector_len == 0 {
            return;
        }
        for i in 0..vector_len {
            let mut update = T::zero();
            for (col, &coeff) in coeffs.iter().enumerate() {
                update = update + basis.data[col * vector_len + i] * coeff;
            }
            y.data[i] = y.data[i] + alpha * update;
        }
    }

    fn dot<T: DeviceScalar>(&self, x: &Self::Buffer<T>, y: &Self::Buffer<T>) -> T {
        assert_eq!(x.data.len(), y.data.len(), "dot length mismatch");
        if x.data.is_empty() {
            return T::zero();
        }
        // The Backend trait specifies Hermitian dot `Σ conj(x_i) y_i`
        // — for real types this collapses to plain dot; for complex we
        // need `dotc`.
        unsafe {
            if TypeId::of::<T>() == TypeId::of::<f32>() {
                let r =
                    blas1::dot_f32(slice_cast::<T, f32>(&x.data), slice_cast::<T, f32>(&y.data));
                val_cast::<f32, T>(r)
            } else if TypeId::of::<T>() == TypeId::of::<f64>() {
                let r =
                    blas1::dot_f64(slice_cast::<T, f64>(&x.data), slice_cast::<T, f64>(&y.data));
                val_cast::<f64, T>(r)
            } else if TypeId::of::<T>() == TypeId::of::<Complex32>() {
                let r = blas1::dotc_c32(
                    slice_cast::<T, Complex32>(&x.data),
                    slice_cast::<T, Complex32>(&y.data),
                );
                val_cast::<Complex32, T>(r)
            } else if TypeId::of::<T>() == TypeId::of::<Complex64>() {
                let r = blas1::dotc_c64(
                    slice_cast::<T, Complex64>(&x.data),
                    slice_cast::<T, Complex64>(&y.data),
                );
                val_cast::<Complex64, T>(r)
            } else {
                let mut acc = T::zero();
                for (xi, yi) in x.data.iter().zip(y.data.iter()) {
                    acc = acc + xi.conj_mul(*yi);
                }
                acc
            }
        }
    }

    fn multi_dot_strided<T: DeviceScalar>(
        &self,
        basis: &Self::Buffer<T>,
        vector_len: usize,
        basis_count: usize,
        y: &Self::Buffer<T>,
    ) -> Vec<T> {
        if y.data.len() != vector_len {
            log::error!(
                target: "sonium_compute::cpu",
                "multi_dot_strided rejected y length {} (expected {vector_len})",
                y.data.len()
            );
            return Vec::new();
        }
        let Some(required) = vector_len.checked_mul(basis_count) else {
            log::error!(
                target: "sonium_compute::cpu",
                "multi_dot_strided basis shape {basis_count}x{vector_len} overflows"
            );
            return Vec::new();
        };
        if basis.data.len() < required {
            log::error!(
                target: "sonium_compute::cpu",
                "multi_dot_strided rejected basis length {} (expected at least {required})",
                basis.data.len()
            );
            return Vec::new();
        }
        if basis.data[..required]
            .iter()
            .any(|value| !value.is_finite())
            || y.data.iter().any(|value| !value.is_finite())
        {
            log::error!(
                target: "sonium_compute::cpu",
                "multi_dot_strided rejected non-finite input"
            );
            return Vec::new();
        }
        let mut out = vec![T::zero(); basis_count];
        for (col, out_col) in out.iter_mut().enumerate() {
            let base = col * vector_len;
            let mut acc = T::zero();
            for i in 0..vector_len {
                acc = acc + basis.data[base + i].conj_mul(y.data[i]);
            }
            *out_col = acc;
        }
        out
    }

    fn nrm2<T: DeviceScalar>(&self, x: &Self::Buffer<T>) -> T::Real {
        if x.data.is_empty() {
            return T::Real::zero();
        }
        let result = unsafe {
            if TypeId::of::<T>() == TypeId::of::<f32>() {
                let r = blas1::nrm2_f32(slice_cast::<T, f32>(&x.data));
                val_cast::<f32, T::Real>(r)
            } else if TypeId::of::<T>() == TypeId::of::<f64>() {
                let r = blas1::nrm2_f64(slice_cast::<T, f64>(&x.data));
                val_cast::<f64, T::Real>(r)
            } else if TypeId::of::<T>() == TypeId::of::<Complex32>() {
                let r = blas1::nrm2_c32(slice_cast::<T, Complex32>(&x.data));
                val_cast::<f32, T::Real>(r)
            } else if TypeId::of::<T>() == TypeId::of::<Complex64>() {
                let r = blas1::nrm2_c64(slice_cast::<T, Complex64>(&x.data));
                val_cast::<f64, T::Real>(r)
            } else {
                let mut acc = T::Real::zero();
                for xi in x.data.iter() {
                    acc = acc + xi.norm_sqr();
                }
                acc.sqrt()
            }
        };

        // The vector kernels use a fast sum-of-squares reduction. If a
        // finite component overflows while being squared, recover with the
        // scaled implementation before the checked wrapper classifies the
        // result as a numerical failure.
        if result.is_finite() {
            result
        } else {
            crate::scalar::stable_nrm2(&x.data)
        }
    }

    fn scale<T: DeviceScalar>(&self, alpha: T, x: &mut Self::Buffer<T>) {
        if x.data.is_empty() {
            return;
        }
        unsafe {
            if TypeId::of::<T>() == TypeId::of::<f32>() {
                blas1::scal_f32(
                    val_cast::<T, f32>(alpha),
                    slice_cast_mut::<T, f32>(&mut x.data),
                );
            } else if TypeId::of::<T>() == TypeId::of::<f64>() {
                blas1::scal_f64(
                    val_cast::<T, f64>(alpha),
                    slice_cast_mut::<T, f64>(&mut x.data),
                );
            } else if TypeId::of::<T>() == TypeId::of::<Complex32>() {
                blas1::scal_c32(
                    val_cast::<T, Complex32>(alpha),
                    slice_cast_mut::<T, Complex32>(&mut x.data),
                );
            } else if TypeId::of::<T>() == TypeId::of::<Complex64>() {
                blas1::scal_c64(
                    val_cast::<T, Complex64>(alpha),
                    slice_cast_mut::<T, Complex64>(&mut x.data),
                );
            } else {
                for xi in x.data.iter_mut() {
                    *xi = alpha * *xi;
                }
            }
        }
    }

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
        debug_assert_eq!(x.data.len(), n);
        debug_assert_eq!(y.data.len(), m);
        if m == 0 {
            return;
        }
        // SAFETY: every branch verifies TypeId equality before reinterpreting.
        unsafe {
            if TypeId::of::<T>() == TypeId::of::<f32>() {
                blas2::gemv_f32(
                    op,
                    m,
                    n,
                    val_cast::<T, f32>(alpha),
                    slice_cast::<T, f32>(&a.data),
                    lda,
                    slice_cast::<T, f32>(&x.data),
                    val_cast::<T, f32>(beta),
                    slice_cast_mut::<T, f32>(&mut y.data),
                );
            } else if TypeId::of::<T>() == TypeId::of::<f64>() {
                blas2::gemv_f64(
                    op,
                    m,
                    n,
                    val_cast::<T, f64>(alpha),
                    slice_cast::<T, f64>(&a.data),
                    lda,
                    slice_cast::<T, f64>(&x.data),
                    val_cast::<T, f64>(beta),
                    slice_cast_mut::<T, f64>(&mut y.data),
                );
            } else if TypeId::of::<T>() == TypeId::of::<Complex32>() {
                blas2::gemv_c32(
                    op,
                    m,
                    n,
                    val_cast::<T, Complex32>(alpha),
                    slice_cast::<T, Complex32>(&a.data),
                    lda,
                    slice_cast::<T, Complex32>(&x.data),
                    val_cast::<T, Complex32>(beta),
                    slice_cast_mut::<T, Complex32>(&mut y.data),
                );
            } else if TypeId::of::<T>() == TypeId::of::<Complex64>() {
                blas2::gemv_c64(
                    op,
                    m,
                    n,
                    val_cast::<T, Complex64>(alpha),
                    slice_cast::<T, Complex64>(&a.data),
                    lda,
                    slice_cast::<T, Complex64>(&x.data),
                    val_cast::<T, Complex64>(beta),
                    slice_cast_mut::<T, Complex64>(&mut y.data),
                );
            } else {
                if let Err(error) =
                    crate::backend::host_gemv(self, op, m, n, alpha, a, lda, x, beta, y)
                {
                    log::error!(target: "sonium_compute::cpu", "GEMV fallback rejected request: {error}");
                }
            }
        }
    }

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
        debug_assert!(c.data.len() >= m * ldc.max(n));
        if m == 0 || n == 0 {
            return;
        }
        unsafe {
            if TypeId::of::<T>() == TypeId::of::<f32>() {
                blas3::gemm_f32(
                    transa,
                    transb,
                    m,
                    n,
                    k,
                    val_cast::<T, f32>(alpha),
                    slice_cast::<T, f32>(&a.data),
                    lda,
                    slice_cast::<T, f32>(&b.data),
                    ldb,
                    val_cast::<T, f32>(beta),
                    slice_cast_mut::<T, f32>(&mut c.data),
                    ldc,
                );
            } else if TypeId::of::<T>() == TypeId::of::<f64>() {
                blas3::gemm_f64(
                    transa,
                    transb,
                    m,
                    n,
                    k,
                    val_cast::<T, f64>(alpha),
                    slice_cast::<T, f64>(&a.data),
                    lda,
                    slice_cast::<T, f64>(&b.data),
                    ldb,
                    val_cast::<T, f64>(beta),
                    slice_cast_mut::<T, f64>(&mut c.data),
                    ldc,
                );
            } else if TypeId::of::<T>() == TypeId::of::<Complex32>() {
                blas3::gemm_c32(
                    transa,
                    transb,
                    m,
                    n,
                    k,
                    val_cast::<T, Complex32>(alpha),
                    slice_cast::<T, Complex32>(&a.data),
                    lda,
                    slice_cast::<T, Complex32>(&b.data),
                    ldb,
                    val_cast::<T, Complex32>(beta),
                    slice_cast_mut::<T, Complex32>(&mut c.data),
                    ldc,
                );
            } else if TypeId::of::<T>() == TypeId::of::<Complex64>() {
                blas3::gemm_c64(
                    transa,
                    transb,
                    m,
                    n,
                    k,
                    val_cast::<T, Complex64>(alpha),
                    slice_cast::<T, Complex64>(&a.data),
                    lda,
                    slice_cast::<T, Complex64>(&b.data),
                    ldb,
                    val_cast::<T, Complex64>(beta),
                    slice_cast_mut::<T, Complex64>(&mut c.data),
                    ldc,
                );
            } else {
                if let Err(error) = crate::backend::host_gemm(
                    self, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                ) {
                    log::error!(target: "sonium_compute::cpu", "GEMM fallback rejected request: {error}");
                }
            }
        }
    }

    fn prebake_csr<'a, T: DeviceScalar>(
        &'a self,
        m: usize,
        n: usize,
        row_ptr: &'a Self::Buffer<i32>,
        col_idx: &'a Self::Buffer<i32>,
        values: &'a Self::Buffer<T>,
    ) -> Self::CsrPrebaked<'a, T> {
        let row_buckets = CsrRowBucketStats::from_row_ptr(&row_ptr.data);
        let buckets = CpuCsrRowBuckets::from_row_ptr(&row_ptr.data);
        let format = if m == 0 || row_buckets.nnz == 0 {
            CsrSpmvFormat::Empty
        } else if parallelism::use_compute_parallel(m, parallelism::SPMV_ROWS_PER_THREAD)
            && (!buckets.simd.is_empty() || !buckets.cta.is_empty())
        {
            CsrSpmvFormat::RowBucketedCsr
        } else {
            CsrSpmvFormat::PlainCsr
        };
        CpuCsrPrebaked {
            m,
            n,
            row_ptr,
            col_idx,
            values,
            buckets,
            inspector: CsrSpmvInspector {
                row_buckets,
                format,
                workspace_bytes: 0,
                algorithm: "cpu-csr-row-buckets",
            },
        }
    }

    fn try_prebake_csr<'a, T: DeviceScalar>(
        &'a self,
        m: usize,
        n: usize,
        row_ptr: &'a Self::Buffer<i32>,
        col_idx: &'a Self::Buffer<i32>,
        values: &'a Self::Buffer<T>,
    ) -> BackendResult<Self::CsrPrebaked<'a, T>> {
        let expected_row_ptr_len =
            m.checked_add(1)
                .ok_or_else(|| BackendError::InvalidArgument {
                    operation: "prebake_csr",
                    reason: "row pointer length overflows usize".into(),
                })?;
        if row_ptr.len() != expected_row_ptr_len {
            return Err(BackendError::InvalidArgument {
                operation: "prebake_csr",
                reason: format!(
                    "row pointer length {} != expected {expected_row_ptr_len}",
                    row_ptr.len()
                ),
            });
        }
        if col_idx.len() != values.len() {
            return Err(BackendError::InvalidArgument {
                operation: "prebake_csr",
                reason: format!(
                    "column/value lengths {} and {} differ",
                    col_idx.len(),
                    values.len()
                ),
            });
        }
        let row_buckets = CsrRowBucketStats::try_from_row_ptr(&row_ptr.data)?;
        if row_buckets.nnz != col_idx.len() {
            return Err(BackendError::InvalidArgument {
                operation: "prebake_csr",
                reason: format!(
                    "row-pointer terminal nnz {} != column/value length {}",
                    row_buckets.nnz,
                    col_idx.len()
                ),
            });
        }
        if col_idx
            .data
            .iter()
            .any(|&column| column < 0 || usize::try_from(column).map_or(true, |column| column >= n))
        {
            return Err(BackendError::InvalidArgument {
                operation: "prebake_csr",
                reason: "column index is outside matrix dimensions".into(),
            });
        }
        Ok(self.prebake_csr(m, n, row_ptr, col_idx, values))
    }

    fn spmv_prebaked<T: DeviceScalar>(
        &self,
        prebaked: &Self::CsrPrebaked<'_, T>,
        x: &Self::Buffer<T>,
        y: &mut Self::Buffer<T>,
    ) {
        if prebaked.inspector.format == CsrSpmvFormat::RowBucketedCsr {
            unsafe {
                if TypeId::of::<T>() == TypeId::of::<f32>() {
                    spmv_csr_bucketed_f32(
                        &prebaked.buckets,
                        &prebaked.row_ptr.data,
                        &prebaked.col_idx.data,
                        slice_cast::<T, f32>(&prebaked.values.data),
                        slice_cast::<T, f32>(&x.data),
                        slice_cast_mut::<T, f32>(&mut y.data),
                    );
                    return;
                }
                if TypeId::of::<T>() == TypeId::of::<f64>() {
                    spmv_csr_bucketed_f64(
                        &prebaked.buckets,
                        &prebaked.row_ptr.data,
                        &prebaked.col_idx.data,
                        slice_cast::<T, f64>(&prebaked.values.data),
                        slice_cast::<T, f64>(&x.data),
                        slice_cast_mut::<T, f64>(&mut y.data),
                    );
                    return;
                }
            }
        }
        self.spmv_csr::<T>(
            prebaked.m,
            prebaked.n,
            prebaked.row_ptr,
            prebaked.col_idx,
            prebaked.values,
            x,
            y,
        );
    }

    fn try_spmv_prebaked<T: DeviceScalar>(
        &self,
        prebaked: &Self::CsrPrebaked<'_, T>,
        x: &Self::Buffer<T>,
        y: &mut Self::Buffer<T>,
    ) -> BackendResult<()> {
        let expected_row_ptr_len =
            prebaked
                .m
                .checked_add(1)
                .ok_or_else(|| BackendError::InvalidArgument {
                    operation: "spmv_prebaked",
                    reason: "row-pointer length overflows usize".into(),
                })?;
        let malformed_descriptor = prebaked.row_ptr.data.len() != expected_row_ptr_len
            || prebaked.col_idx.data.len() != prebaked.values.data.len()
            || prebaked.row_ptr.data.first() != Some(&0)
            || prebaked
                .row_ptr
                .data
                .windows(2)
                .any(|window| window[0] < 0 || window[1] < window[0])
            || prebaked
                .row_ptr
                .data
                .last()
                .and_then(|&end| usize::try_from(end).ok())
                != Some(prebaked.values.data.len())
            || prebaked.col_idx.data.iter().any(|&column| {
                column < 0 || usize::try_from(column).map_or(true, |column| column >= prebaked.n)
            });
        if malformed_descriptor {
            return Err(BackendError::InvalidArgument {
                operation: "spmv_prebaked",
                reason: "malformed CSR descriptor".into(),
            });
        }
        if x.len() != prebaked.n || y.len() != prebaked.m {
            return Err(BackendError::InvalidArgument {
                operation: "spmv_prebaked",
                reason: format!(
                    "buffer lengths must match matrix dimensions {}x{}: x={}, y={}",
                    prebaked.m,
                    prebaked.n,
                    x.len(),
                    y.len()
                ),
            });
        }

        // The one-shot checked API validates the matrix and input vector
        // before constructing the descriptor.  Reusing a pre-baked handle
        // must preserve that same finite-input contract; otherwise callers
        // could bypass admission simply by choosing the steady-state path.
        if prebaked.values.data.iter().any(|value| !value.is_finite()) {
            return Err(BackendError::OperationFailed {
                operation: "spmv_prebaked",
                reason: "matrix values contain NaN or infinity".into(),
            });
        }
        if x.data.iter().any(|value| !value.is_finite()) {
            return Err(BackendError::OperationFailed {
                operation: "spmv_prebaked",
                reason: "input vector contains NaN or infinity".into(),
            });
        }

        catch_backend_panic("spmv_prebaked", || self.spmv_prebaked(prebaked, x, y))?;
        self.check_pending_error()
    }

    fn spmm_prebaked<T: DeviceScalar>(
        &self,
        prebaked: &Self::CsrPrebaked<'_, T>,
        nrhs: usize,
        x: &Self::Buffer<T>,
        ldx: usize,
        y: &mut Self::Buffer<T>,
        ldy: usize,
    ) {
        // The checked entry point validates these conditions before dispatch,
        // but this compatibility method is also reachable directly.  Keep
        // malformed legacy calls fail-closed instead of allowing stride
        // arithmetic or slicing below to panic.
        if nrhs != 0 && (ldx < prebaked.n || ldy < prebaked.m) {
            log::error!(
                target: "sonium_compute::cpu",
                "spmm rejected leading dimensions ldx={ldx}, ldy={ldy} for matrix {}x{}",
                prebaked.m,
                prebaked.n
            );
            return;
        }
        let Some(x_required) = checked_strided_span("spmm_prebaked", nrhs, ldx, prebaked.n).ok()
        else {
            log::error!(
                target: "sonium_compute::cpu",
                "spmm rejected overflowing x layout for nrhs={nrhs}, ldx={ldx}"
            );
            return;
        };
        let Some(y_required) = checked_strided_span("spmm_prebaked", nrhs, ldy, prebaked.m).ok()
        else {
            log::error!(
                target: "sonium_compute::cpu",
                "spmm rejected overflowing y layout for nrhs={nrhs}, ldy={ldy}"
            );
            return;
        };
        if x.data.len() < x_required || y.data.len() < y_required {
            log::error!(
                target: "sonium_compute::cpu",
                "spmm rejected short buffers for nrhs={nrhs}, x={}, y={}",
                x.data.len(),
                y.data.len()
            );
            return;
        }
        if nrhs == 0 || prebaked.m == 0 {
            return;
        }

        let row_ptr = &prebaked.row_ptr.data;
        let col_idx = &prebaked.col_idx.data;
        let values = &prebaked.values.data;
        let m = prebaked.m;
        let compute_col = |col: usize, y_col: &mut [T]| {
            let x_col = &x.data[col * ldx..col * ldx + prebaked.n];
            for row in 0..m {
                let start = row_ptr[row] as usize;
                let end = row_ptr[row + 1] as usize;
                let mut acc = T::zero();
                for idx in start..end {
                    acc = acc + values[idx] * x_col[col_idx[idx] as usize];
                }
                y_col[row] = acc;
            }
        };

        // Each output column is accumulated serially within one worker,
        // so both paths are bitwise identical.
        if parallelism::use_compute_parallel(
            m.saturating_mul(nrhs),
            parallelism::SPMM_OUTPUTS_PER_THREAD,
        ) {
            y.data
                .par_chunks_mut(ldy)
                .take(nrhs)
                .enumerate()
                .for_each(|(col, y_col)| compute_col(col, y_col));
        } else {
            for (col, y_col) in y.data.chunks_mut(ldy).take(nrhs).enumerate() {
                compute_col(col, y_col);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn try_spmm_prebaked<T: DeviceScalar>(
        &self,
        prebaked: &Self::CsrPrebaked<'_, T>,
        nrhs: usize,
        x: &Self::Buffer<T>,
        ldx: usize,
        y: &mut Self::Buffer<T>,
        ldy: usize,
    ) -> BackendResult<()> {
        if nrhs != 0 && (ldx < prebaked.n || ldy < prebaked.m) {
            return Err(BackendError::InvalidArgument {
                operation: "spmm_prebaked",
                reason: format!(
                    "leading dimensions ({ldx}, {ldy}) are smaller than matrix ({}, {})",
                    prebaked.n, prebaked.m
                ),
            });
        }
        let x_required = checked_strided_span("spmm_prebaked", nrhs, ldx, prebaked.n)?;
        let y_required = checked_strided_span("spmm_prebaked", nrhs, ldy, prebaked.m)?;
        if x.len() < x_required || y.len() < y_required {
            return Err(BackendError::InvalidArgument {
                operation: "spmm_prebaked",
                reason: format!("buffers are too short for nrhs={nrhs}, ldx={ldx}, ldy={ldy}"),
            });
        }
        if prebaked
            .m
            .checked_add(1)
            .is_none_or(|expected| prebaked.row_ptr.data.len() != expected)
            || prebaked.col_idx.data.len() != prebaked.values.data.len()
            || prebaked
                .row_ptr
                .data
                .windows(2)
                .any(|window| window[0] < 0 || window[1] < window[0])
            || prebaked
                .row_ptr
                .data
                .last()
                .is_none_or(|&end| end as usize != prebaked.values.data.len())
            || prebaked
                .col_idx
                .data
                .iter()
                .any(|&column| column < 0 || column as usize >= prebaked.n)
        {
            return Err(BackendError::InvalidArgument {
                operation: "spmm_prebaked",
                reason: "malformed CSR descriptor".into(),
            });
        }
        if x.data[..x_required].iter().any(|value| !value.is_finite())
            || prebaked.values.data.iter().any(|value| !value.is_finite())
        {
            return Err(BackendError::OperationFailed {
                operation: "spmm_prebaked",
                reason: "input contains NaN or infinity".into(),
            });
        }
        catch_backend_panic("spmm_prebaked", || {
            self.spmm_prebaked(prebaked, nrhs, x, ldx, y, ldy)
        })?;
        if y.data[..y_required].iter().any(|value| !value.is_finite()) {
            return Err(BackendError::OperationFailed {
                operation: "spmm_prebaked",
                reason: "result contains NaN or infinity".into(),
            });
        }
        Ok(())
    }

    fn spmm_prebaked_colmajor<T: DeviceScalar>(
        &self,
        prebaked: &Self::CsrPrebaked<'_, T>,
        rhs: usize,
        x: &Self::Buffer<T>,
        y: &mut Self::Buffer<T>,
    ) {
        self.spmm_prebaked::<T>(prebaked, rhs, x, prebaked.n, y, prebaked.m);
    }

    fn try_spmm_prebaked_colmajor<T: DeviceScalar>(
        &self,
        prebaked: &Self::CsrPrebaked<'_, T>,
        rhs: usize,
        x: &Self::Buffer<T>,
        y: &mut Self::Buffer<T>,
    ) -> BackendResult<()> {
        self.try_spmm_prebaked(prebaked, rhs, x, prebaked.n, y, prebaked.m)
    }

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
        debug_assert_eq!(row_ptr.data.len(), m + 1);
        debug_assert_eq!(x.data.len(), n);
        debug_assert_eq!(y.data.len(), m);
        let _ = n;

        unsafe {
            if TypeId::of::<T>() == TypeId::of::<f32>() {
                spmv_csr_f32(
                    &row_ptr.data,
                    &col_idx.data,
                    slice_cast::<T, f32>(&values.data),
                    slice_cast::<T, f32>(&x.data),
                    slice_cast_mut::<T, f32>(&mut y.data),
                );
                return;
            }
            if TypeId::of::<T>() == TypeId::of::<f64>() {
                spmv_csr_f64(
                    &row_ptr.data,
                    &col_idx.data,
                    slice_cast::<T, f64>(&values.data),
                    slice_cast::<T, f64>(&x.data),
                    slice_cast_mut::<T, f64>(&mut y.data),
                );
                return;
            }
        }

        let row_ptr_slice = &row_ptr.data;
        let col_idx_slice = &col_idx.data;
        let values_slice = &values.data;
        let x_slice = &x.data;
        let compute_row = |i: usize| -> T {
            let start = row_ptr_slice[i] as usize;
            let end = row_ptr_slice[i + 1] as usize;
            (start..end).fold(T::zero(), |acc, k| {
                let j = col_idx_slice[k] as usize;
                acc + values_slice[k] * x_slice[j]
            })
        };
        if parallelism::use_compute_parallel(y.data.len(), parallelism::SPMV_ROWS_PER_THREAD) {
            y.data.par_iter_mut().enumerate().for_each(|(i, yi)| {
                *yi = compute_row(i);
            });
        } else {
            for (i, yi) in y.data.iter_mut().enumerate() {
                *yi = compute_row(i);
            }
        }
    }
    #[cfg(test)]
    fn check_pending_error(&self) -> BackendResult<()> {
        if CPU_PENDING_ERROR.with(|pending| pending.replace(false)) {
            Err(BackendError::OperationFailed {
                operation: "cpu_pending_error",
                reason: "injected pending backend error".into(),
            })
        } else {
            Ok(())
        }
    }

    fn try_inject_preconditioner_failure(
        &self,
        point: PreconditionerFailureInjectionPoint,
    ) -> BackendResult<()> {
        let armed = CPU_PRECONDITIONER_FAILURE.with(|failure| {
            let candidate = failure.get();
            let matches = candidate == Some(point)
                || (point == PreconditionerFailureInjectionPoint::Launch
                    && matches!(
                        candidate,
                        Some(
                            PreconditionerFailureInjectionPoint::NonFiniteGeneration
                                | PreconditionerFailureInjectionPoint::DeviceLoss
                        )
                    ));
            if matches {
                failure.set(None);
                candidate
            } else {
                None
            }
        });
        if let Some(armed) = armed {
            Err(injected_preconditioner_error(armed, "cpu"))
        } else {
            Ok(())
        }
    }

    fn bem_regular_pair_c64_batch(
        &self,
        pairs: &[BemRegularPairInF64],
        wave_number: f64,
        harmonic_factor: f64,
    ) -> Vec<BemRegularPairOutF64> {
        CpuBackend::bem_regular_pair_c64_batch(self, pairs, wave_number, harmonic_factor)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn checked_prebaked_spmv_rejects_nonfinite_inputs() {
        let backend = CpuBackend::new();
        let mut row_ptr = backend.raw_alloc::<i32>(2).expect("row pointer");
        let mut col_idx = backend.raw_alloc::<i32>(1).expect("column index");
        let mut values = backend.raw_alloc::<f64>(1).expect("matrix value");
        let mut x = backend.raw_alloc::<f64>(1).expect("input vector");
        let mut y = backend.raw_alloc::<f64>(1).expect("output vector");

        backend.copy_h2d(&[0, 1], &mut row_ptr);
        backend.copy_h2d(&[0], &mut col_idx);
        backend.copy_h2d(&[1.0], &mut values);
        backend.copy_h2d(&[f64::NAN], &mut x);

        let prebaked = backend
            .try_prebake_csr(1, 1, &row_ptr, &col_idx, &values)
            .expect("valid CSR descriptor");
        let error = backend
            .try_spmv_prebaked(&prebaked, &x, &mut y)
            .expect_err("non-finite input must be rejected");
        assert!(matches!(
            error,
            BackendError::OperationFailed {
                operation: "spmv_prebaked",
                ..
            }
        ));

        backend.copy_h2d(&[1.0], &mut x);
        let mut nonfinite_values = backend
            .raw_alloc::<f64>(1)
            .expect("non-finite matrix value");
        backend.copy_h2d(&[f64::INFINITY], &mut nonfinite_values);
        let nonfinite_prebaked = backend
            .try_prebake_csr(1, 1, &row_ptr, &col_idx, &nonfinite_values)
            .expect("valid CSR structure");
        let error = backend
            .try_spmv_prebaked(&nonfinite_prebaked, &x, &mut y)
            .expect_err("non-finite matrix values must be rejected");
        assert!(matches!(
            error,
            BackendError::OperationFailed {
                operation: "spmv_prebaked",
                ..
            }
        ));
    }

    use super::*;
    use crate::backend::{BufferView, ReductionMode};
    use crate::pool::{MemoryPool, MemoryPoolConfig};
    use approx::assert_relative_eq;

    #[test]
    fn transfer_stats_track_checked_cpu_copies() {
        let backend = CpuBackend::new();
        let before = backend.transfer_stats();
        let mut source = backend.raw_alloc::<f64>(3).expect("source allocation");
        let mut destination = backend.raw_alloc::<f64>(3).expect("destination allocation");
        backend
            .try_copy_h2d(&[1.0, 2.0, 3.0], &mut source)
            .expect("host-to-device copy");
        backend
            .try_copy_d2d(&source, &mut destination)
            .expect("device-to-device copy");
        let mut host = [0.0; 3];
        backend
            .try_copy_d2h(&destination, &mut host)
            .expect("device-to-host copy");
        let after = backend.transfer_stats();
        assert_eq!(host, [1.0, 2.0, 3.0]);
        assert!(after.h2d_operations > before.h2d_operations);
        assert!(after.d2d_operations > before.d2d_operations);
        assert!(after.d2h_operations > before.d2h_operations);
        assert!(after.h2d_bytes >= before.h2d_bytes + 3 * std::mem::size_of::<f64>());
        assert!(after.d2d_bytes >= before.d2d_bytes + 3 * std::mem::size_of::<f64>());
        assert!(after.d2h_bytes >= before.d2h_bytes + 3 * std::mem::size_of::<f64>());
    }

    #[test]
    fn checked_cpu_views_copy_strided_data_without_full_buffer_staging() {
        let backend = CpuBackend::new();
        backend.reset_transfer_stats();
        let mut source = backend.raw_alloc::<f64>(8).expect("source allocation");
        let mut destination = backend.raw_alloc::<f64>(8).expect("destination allocation");
        backend
            .try_copy_h2d(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &mut source)
            .expect("source upload");
        backend
            .try_copy_h2d(&[10.0; 8], &mut destination)
            .expect("destination upload");

        backend
            .try_copy_views_d2d(
                &source,
                BufferView::new(8, 1, 3, 2).expect("source view"),
                &mut destination,
                BufferView::new(8, 2, 3, 1).expect("destination view"),
            )
            .expect("strided device copy");

        let mut result = [0.0; 8];
        backend
            .try_copy_d2h(&destination, &mut result)
            .expect("destination download");
        assert_eq!(result, [10.0, 10.0, 1.0, 3.0, 5.0, 10.0, 10.0, 10.0]);

        let stats = backend.transfer_stats();
        // The CPU counters are process-global because they also cover the
        // backend's compatibility callers; parallel unit tests may add work
        // between the reset and snapshot.  The isolated QA binary checks the
        // exact steady-state counts.
        assert!(stats.d2d_operations >= 1);
        assert!(stats.d2d_bytes >= 3 * std::mem::size_of::<f64>());
        assert!(stats.h2d_operations >= 2);
        assert!(stats.d2h_operations >= 1);
    }

    #[test]
    fn checked_cpu_view_copy_rejects_length_mismatch_without_mutation() {
        let backend = CpuBackend::new();
        let source = backend.raw_alloc::<f32>(4).expect("source allocation");
        let mut destination = backend.raw_alloc::<f32>(4).expect("destination allocation");
        backend
            .try_copy_h2d(&[1.0, 2.0, 3.0, 4.0], &mut destination)
            .expect("destination upload");
        let error = backend
            .try_copy_views_d2d(
                &source,
                BufferView::new(4, 0, 2, 1).expect("source view"),
                &mut destination,
                BufferView::new(4, 0, 1, 1).expect("destination view"),
            )
            .expect_err("mismatched views must fail");
        assert!(matches!(error, BackendError::InvalidArgument { .. }));
        let mut result = [0.0; 4];
        backend
            .try_copy_d2h(&destination, &mut result)
            .expect("destination download");
        assert_eq!(result, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn axpy_f64_via_blas1() {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let mut x = pool.try_alloc::<f64>(4).unwrap();
        let mut y = pool.try_alloc::<f64>(4).unwrap();
        pool.backend().copy_h2d(&[1.0, 2.0, 3.0, 4.0], x.raw_mut());
        pool.backend()
            .copy_h2d(&[10.0, 20.0, 30.0, 40.0], y.raw_mut());
        pool.backend().axpy(2.0_f64, x.raw(), y.raw_mut());
        let mut out = vec![0.0; 4];
        pool.backend().copy_d2h(y.raw(), &mut out);
        assert_relative_eq!(out[0], 12.0);
        assert_relative_eq!(out[3], 48.0);
    }

    #[test]
    fn axpy_f32_via_blas1() {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let mut x = pool.try_alloc::<f32>(8).unwrap();
        let mut y = pool.try_alloc::<f32>(8).unwrap();
        let xs: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let ys: [f32; 8] = [0.5; 8];
        pool.backend().copy_h2d(&xs, x.raw_mut());
        pool.backend().copy_h2d(&ys, y.raw_mut());
        pool.backend().axpy(2.0_f32, x.raw(), y.raw_mut());
        let mut out = vec![0.0_f32; 8];
        pool.backend().copy_d2h(y.raw(), &mut out);
        for (i, v) in out.iter().enumerate() {
            assert_relative_eq!(*v, 0.5 + 2.0 * (i as f32 + 1.0), max_relative = 1e-6);
        }
    }

    #[test]
    fn axpy_many_f64_fuses_linear_combination() {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let mut x0 = pool.try_alloc::<f64>(4).unwrap();
        let mut x1 = pool.try_alloc::<f64>(4).unwrap();
        let mut y = pool.try_alloc::<f64>(4).unwrap();
        pool.backend().copy_h2d(&[1.0, 2.0, 3.0, 4.0], x0.raw_mut());
        pool.backend().copy_h2d(&[5.0, 6.0, 7.0, 8.0], x1.raw_mut());
        pool.backend()
            .copy_h2d(&[10.0, 20.0, 30.0, 40.0], y.raw_mut());

        pool.backend()
            .axpy_many(&[0.5_f64, -2.0], &[x0.raw(), x1.raw()], y.raw_mut());

        let mut out = vec![0.0; 4];
        pool.backend().copy_d2h(y.raw(), &mut out);
        let expected = [0.5, 9.0, 17.5, 26.0];
        for (got, want) in out.iter().zip(expected.iter()) {
            assert_relative_eq!(*got, *want, max_relative = 1e-12);
        }
    }

    #[test]
    fn axpy_many_c32_handles_complex_coefficients() {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let mut x0 = pool.try_alloc::<Complex32>(2).unwrap();
        let mut x1 = pool.try_alloc::<Complex32>(2).unwrap();
        let mut y = pool.try_alloc::<Complex32>(2).unwrap();
        pool.backend().copy_h2d(
            &[Complex32::new(1.0, 2.0), Complex32::new(-3.0, 0.5)],
            x0.raw_mut(),
        );
        pool.backend().copy_h2d(
            &[Complex32::new(0.25, -1.0), Complex32::new(2.0, 3.0)],
            x1.raw_mut(),
        );
        pool.backend().copy_h2d(
            &[Complex32::new(4.0, -2.0), Complex32::new(0.5, 0.75)],
            y.raw_mut(),
        );

        let alphas = [Complex32::new(0.5, -0.25), Complex32::new(-1.0, 0.5)];
        pool.backend()
            .axpy_many(&alphas, &[x0.raw(), x1.raw()], y.raw_mut());

        let mut out = vec![Complex32::default(); 2];
        pool.backend().copy_d2h(y.raw(), &mut out);
        let expected = [Complex32::new(5.25, -0.125), Complex32::new(-4.375, -0.25)];
        for (got, want) in out.iter().zip(expected.iter()) {
            assert_relative_eq!(got.re, want.re, max_relative = 1e-6);
            assert_relative_eq!(got.im, want.im, max_relative = 1e-6);
        }
    }

    #[test]
    fn dot_real_via_blas1() {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let mut x = pool.try_alloc::<f64>(3).unwrap();
        let mut y = pool.try_alloc::<f64>(3).unwrap();
        pool.backend().copy_h2d(&[1.0, 2.0, 3.0], x.raw_mut());
        pool.backend().copy_h2d(&[4.0, 5.0, 6.0], y.raw_mut());
        let d = pool.backend().dot(x.raw(), y.raw());
        assert_relative_eq!(d, 32.0);
    }

    #[test]
    fn nrm2_f32_via_blas1() {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let mut x = pool.try_alloc::<f32>(3).unwrap();
        pool.backend().copy_h2d(&[3.0_f32, 4.0, 0.0], x.raw_mut());
        let n = pool.backend().nrm2(x.raw());
        assert_relative_eq!(n, 5.0_f32, max_relative = 1e-6);
    }

    #[test]
    fn dotc_complex_hermitian() {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let mut x = pool.try_alloc::<Complex64>(2).unwrap();
        let mut y = pool.try_alloc::<Complex64>(2).unwrap();
        pool.backend().copy_h2d(
            &[Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)],
            x.raw_mut(),
        );
        pool.backend().copy_h2d(
            &[Complex64::new(3.0, 0.0), Complex64::new(0.0, 1.0)],
            y.raw_mut(),
        );
        // dotc = conj(x) . y
        // = conj(1+i)*(3) + conj(2-i)*(i)
        // = (1-i)*3 + (2+i)*i
        // = 3 - 3i + 2i + i*i
        // = 3 - 3i + 2i - 1
        // = 2 - i
        let d = pool.backend().dot(x.raw(), y.raw());
        assert_relative_eq!(d.re, 2.0, max_relative = 1e-12);
        assert_relative_eq!(d.im, -1.0, max_relative = 1e-12);
    }

    #[test]
    fn scale_f64() {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let mut x = pool.try_alloc::<f64>(4).unwrap();
        pool.backend().copy_h2d(&[1.0, 2.0, 3.0, 4.0], x.raw_mut());
        pool.backend().scale(2.5_f64, x.raw_mut());
        let mut out = vec![0.0; 4];
        pool.backend().copy_d2h(x.raw(), &mut out);
        assert_eq!(out, vec![2.5, 5.0, 7.5, 10.0]);
    }

    #[test]
    fn spmv_diag() {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let mut row = pool.try_alloc::<i32>(4).unwrap();
        let mut col = pool.try_alloc::<i32>(3).unwrap();
        let mut val = pool.try_alloc::<f64>(3).unwrap();
        let mut x = pool.try_alloc::<f64>(3).unwrap();
        let mut y = pool.try_alloc::<f64>(3).unwrap();
        pool.backend().copy_h2d(&[0, 1, 2, 3], row.raw_mut());
        pool.backend().copy_h2d(&[0, 1, 2], col.raw_mut());
        pool.backend().copy_h2d(&[2.0, 3.0, 4.0], val.raw_mut());
        pool.backend().copy_h2d(&[1.0, 1.0, 1.0], x.raw_mut());
        pool.backend()
            .spmv_csr(3, 3, row.raw(), col.raw(), val.raw(), x.raw(), y.raw_mut());
        let mut out = vec![0.0; 3];
        pool.backend().copy_d2h(y.raw(), &mut out);
        assert_eq!(out, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn spmm_prebaked_colmajor_matches_repeated_spmv() {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let m = 3usize;
        let n = 3usize;
        let rhs = 2usize;
        let row_host = vec![0_i32, 2, 4, 6];
        let col_host = vec![0_i32, 1, 1, 2, 0, 2];
        let val_host = vec![2.0_f64, -1.0, 3.0, 0.5, -2.0, 4.0];
        let x_host = vec![1.0_f64, 2.0, -1.0, 0.5, -0.25, 3.0];
        let mut row = pool.try_alloc::<i32>(row_host.len()).unwrap();
        let mut col = pool.try_alloc::<i32>(col_host.len()).unwrap();
        let mut val = pool.try_alloc::<f64>(val_host.len()).unwrap();
        let mut x = pool.try_alloc::<f64>(x_host.len()).unwrap();
        let mut y = pool.try_alloc::<f64>(m * rhs).unwrap();
        pool.backend().copy_h2d(&row_host, row.raw_mut());
        pool.backend().copy_h2d(&col_host, col.raw_mut());
        pool.backend().copy_h2d(&val_host, val.raw_mut());
        pool.backend().copy_h2d(&x_host, x.raw_mut());

        let prebaked = pool
            .backend()
            .prebake_csr::<f64>(m, n, row.raw(), col.raw(), val.raw());
        pool.backend()
            .spmm_prebaked_colmajor(&prebaked, rhs, x.raw(), y.raw_mut());

        let mut want = vec![0.0_f64; m * rhs];
        for col_id in 0..rhs {
            let x_col = &x_host[col_id * n..(col_id + 1) * n];
            for row_id in 0..m {
                let start = row_host[row_id] as usize;
                let end = row_host[row_id + 1] as usize;
                for idx in start..end {
                    want[col_id * m + row_id] += val_host[idx] * x_col[col_host[idx] as usize];
                }
            }
        }
        let mut got = vec![0.0_f64; m * rhs];
        pool.backend().copy_d2h(y.raw(), &mut got);
        assert_eq!(got, want);
    }

    #[test]
    fn spmm_prebaked_supports_strided_layout() {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let m = 3usize;
        let n = 3usize;
        let rhs = 2usize;
        let ldx = 5usize;
        let ldy = 4usize;
        let row_host = vec![0_i32, 2, 4, 6];
        let col_host = vec![0_i32, 1, 1, 2, 0, 2];
        let val_host = vec![2.0_f64, -1.0, 3.0, 0.5, -2.0, 4.0];
        let x_host = vec![1.0_f64, 2.0, -1.0, 99.0, 99.0, 0.5, -0.25, 3.0, 77.0, 77.0];
        let mut row = pool.try_alloc::<i32>(row_host.len()).unwrap();
        let mut col = pool.try_alloc::<i32>(col_host.len()).unwrap();
        let mut val = pool.try_alloc::<f64>(val_host.len()).unwrap();
        let mut x = pool.try_alloc::<f64>(x_host.len()).unwrap();
        let mut y = pool.try_alloc::<f64>(ldy * rhs).unwrap();
        pool.backend().copy_h2d(&row_host, row.raw_mut());
        pool.backend().copy_h2d(&col_host, col.raw_mut());
        pool.backend().copy_h2d(&val_host, val.raw_mut());
        pool.backend().copy_h2d(&x_host, x.raw_mut());

        let prebaked = pool
            .backend()
            .prebake_csr::<f64>(m, n, row.raw(), col.raw(), val.raw());
        pool.backend()
            .spmm_prebaked(&prebaked, rhs, x.raw(), ldx, y.raw_mut(), ldy);

        let mut want = vec![0.0_f64; ldy * rhs];
        for col_id in 0..rhs {
            let x_col = &x_host[col_id * ldx..col_id * ldx + n];
            for row_id in 0..m {
                let start = row_host[row_id] as usize;
                let end = row_host[row_id + 1] as usize;
                for idx in start..end {
                    want[col_id * ldy + row_id] += val_host[idx] * x_col[col_host[idx] as usize];
                }
            }
        }
        let mut got = vec![0.0_f64; ldy * rhs];
        pool.backend().copy_d2h(y.raw(), &mut got);
        assert_eq!(got, want);
    }

    #[test]
    fn checked_spmm_accepts_padded_final_stride_without_overread() {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let row_host = vec![0_i32, 1, 2];
        let col_host = vec![0_i32, 1];
        let val_host = vec![2.0_f64, 3.0];
        let mut row = pool.try_alloc::<i32>(row_host.len()).unwrap();
        let mut col = pool.try_alloc::<i32>(col_host.len()).unwrap();
        let mut val = pool.try_alloc::<f64>(val_host.len()).unwrap();
        let mut x = pool.try_alloc::<f64>(5).unwrap();
        let mut y = pool.try_alloc::<f64>(4).unwrap();
        pool.backend().copy_h2d(&row_host, row.raw_mut());
        pool.backend().copy_h2d(&col_host, col.raw_mut());
        pool.backend().copy_h2d(&val_host, val.raw_mut());
        // n=2, rhs=2, ldx=3: the last logical entry is at index 4, not 5.
        pool.backend()
            .copy_h2d(&[1.0, 2.0, 99.0, 4.0, 5.0], x.raw_mut());
        let prebaked = pool
            .backend()
            .prebake_csr::<f64>(2, 2, row.raw(), col.raw(), val.raw());

        pool.backend()
            .try_spmm_prebaked(&prebaked, 2, x.raw(), 3, y.raw_mut(), 2)
            .expect("valid padded layout should be accepted");
        let mut got = vec![0.0; 4];
        pool.backend().copy_d2h(y.raw(), &mut got);
        assert_eq!(got, vec![2.0, 6.0, 8.0, 15.0]);
    }

    #[test]
    fn legacy_spmm_rejects_bad_layout_without_panicking_or_mutating() {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let mut row = pool.try_alloc::<i32>(2).unwrap();
        let mut col = pool.try_alloc::<i32>(1).unwrap();
        let mut val = pool.try_alloc::<f64>(1).unwrap();
        let mut x = pool.try_alloc::<f64>(1).unwrap();
        let mut y = pool.try_alloc::<f64>(1).unwrap();
        pool.backend().copy_h2d(&[0, 1], row.raw_mut());
        pool.backend().copy_h2d(&[0], col.raw_mut());
        pool.backend().copy_h2d(&[2.0], val.raw_mut());
        pool.backend().copy_h2d(&[1.0], x.raw_mut());
        pool.backend().copy_h2d(&[42.0], y.raw_mut());
        let prebaked = pool
            .backend()
            .prebake_csr::<f64>(1, 1, row.raw(), col.raw(), val.raw());

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // ldx=0 violates the legacy entry point's layout contract.
            pool.backend()
                .spmm_prebaked(&prebaked, 1, x.raw(), 0, y.raw_mut(), 1);
        }));
        assert!(result.is_ok(), "legacy SpMM must fail closed, not panic");

        let mut got = [0.0];
        pool.backend().copy_d2h(y.raw(), &mut got);
        assert_eq!(got, [42.0]);
    }

    #[test]
    fn checked_spmm_rejects_bad_layout_before_mutating_output() {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let mut row = pool.try_alloc::<i32>(2).unwrap();
        let mut col = pool.try_alloc::<i32>(1).unwrap();
        let mut val = pool.try_alloc::<f64>(1).unwrap();
        let mut x = pool.try_alloc::<f64>(1).unwrap();
        let mut y = pool.try_alloc::<f64>(1).unwrap();
        pool.backend().copy_h2d(&[0, 1], row.raw_mut());
        pool.backend().copy_h2d(&[0], col.raw_mut());
        pool.backend().copy_h2d(&[2.0], val.raw_mut());
        pool.backend().copy_h2d(&[1.0], x.raw_mut());
        pool.backend().copy_h2d(&[42.0], y.raw_mut());
        let prebaked = pool
            .backend()
            .prebake_csr::<f64>(1, 1, row.raw(), col.raw(), val.raw());

        let error = pool
            .backend()
            .try_spmm_prebaked(&prebaked, 1, x.raw(), 0, y.raw_mut(), 1)
            .expect_err("zero leading dimension must be rejected");
        assert!(matches!(error, BackendError::InvalidArgument { .. }));
        let mut got = vec![0.0];
        pool.backend().copy_d2h(y.raw(), &mut got);
        assert_eq!(got, vec![42.0]);
    }

    #[test]
    fn flat_basis_helpers_match_serial_math() {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let vector_len = 4usize;
        let basis_count = 3usize;
        let basis_host = vec![
            1.0_f64, 2.0, 0.0, -1.0, -2.0, 0.5, 1.0, 3.0, 0.25, -1.5, 2.0, 0.75,
        ];
        let y_host = vec![2.0_f64, -1.0, 0.5, 4.0];
        let coeffs = vec![0.5_f64, -2.0, 1.25];
        let mut basis = pool.try_alloc::<f64>(basis_host.len()).unwrap();
        let mut y = pool.try_alloc::<f64>(y_host.len()).unwrap();
        pool.backend().copy_h2d(&basis_host, basis.raw_mut());
        pool.backend().copy_h2d(&y_host, y.raw_mut());

        let dots = pool
            .backend()
            .multi_dot_strided(basis.raw(), vector_len, basis_count, y.raw());
        let want_dots: Vec<f64> = (0..basis_count)
            .map(|col_id| {
                (0..vector_len)
                    .map(|i| basis_host[col_id * vector_len + i] * y_host[i])
                    .sum()
            })
            .collect();
        assert_eq!(dots, want_dots);

        pool.backend().basis_axpy_strided(
            -1.0,
            basis.raw(),
            vector_len,
            basis_count,
            &coeffs,
            y.raw_mut(),
        );
        let mut got = vec![0.0_f64; vector_len];
        pool.backend().copy_d2h(y.raw(), &mut got);
        let mut want = y_host;
        for i in 0..vector_len {
            for col_id in 0..basis_count {
                want[i] -= basis_host[col_id * vector_len + i] * coeffs[col_id];
            }
        }
        assert_eq!(got, want);
    }

    #[test]
    fn prebaked_spmv_caches_row_bucket_inspector() {
        // The prebaked format resolves through the machine-adaptive gate,
        // so this test pins the gate hermetically: hold the intra-crate env
        // lock (the gate-selection and forced-path tests mutate these keys
        // on other threads), force serial, and restore afterwards. Serial
        // always selects PlainCsr on any machine, regardless of thread
        // count or ambient environment.
        let _env_guard = parallelism::env_test_lock();
        let prev_threshold = std::env::var_os("SONIUM_COMPUTE_PAR_THRESHOLD");
        let prev_serial = std::env::var_os("SONIUM_SERIAL");
        // SAFETY: holding the intra-crate env lock; restored below.
        unsafe {
            std::env::remove_var("SONIUM_COMPUTE_PAR_THRESHOLD");
            std::env::set_var("SONIUM_SERIAL", "1");
        }
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let row_host = vec![0, 0, 3, 11, 80];
        let col_host: Vec<i32> = (0..80).map(|i| i % 4).collect();
        let val_host = vec![1.0_f32; 80];
        let mut row = pool.try_alloc::<i32>(row_host.len()).unwrap();
        let mut col = pool.try_alloc::<i32>(col_host.len()).unwrap();
        let mut val = pool.try_alloc::<f32>(val_host.len()).unwrap();
        pool.backend().copy_h2d(&row_host, row.raw_mut());
        pool.backend().copy_h2d(&col_host, col.raw_mut());
        pool.backend().copy_h2d(&val_host, val.raw_mut());

        let prebaked = pool
            .backend()
            .prebake_csr(4, 4, row.raw(), col.raw(), val.raw());
        let stats = prebaked.row_bucket_stats();
        assert_eq!(stats.rows, 4);
        assert_eq!(stats.nnz, 80);
        assert_eq!(stats.empty_rows, 1);
        assert_eq!(stats.short_rows, 1);
        assert_eq!(stats.simd_rows, 1);
        assert_eq!(stats.cta_rows, 1);
        assert_eq!(prebaked.bucket_row_counts(), (1, 1, 1));
        assert_eq!(prebaked.inspector().format, CsrSpmvFormat::PlainCsr);
        assert_eq!(prebaked.inspector().algorithm, "cpu-csr-row-buckets");
        // SAFETY: restoring the gate overrides saved above.
        unsafe {
            match prev_threshold {
                Some(value) => std::env::set_var("SONIUM_COMPUTE_PAR_THRESHOLD", value),
                None => std::env::remove_var("SONIUM_COMPUTE_PAR_THRESHOLD"),
            }
            match prev_serial {
                Some(value) => std::env::set_var("SONIUM_SERIAL", value),
                None => std::env::remove_var("SONIUM_SERIAL"),
            }
        }
    }

    #[test]
    fn prebaked_spmv_uses_bucketed_real_path() {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 24)).unwrap();
        // Size the problem above the machine-adaptive parallel gate so
        // this test exercises the bucketed parallel path on any machine.
        // The gate override is forced hermetically (saved/restored) so
        // `SONIUM_SERIAL=1` full-suite runs cannot flip the format. The
        // intra-crate env lock excludes concurrent writers (notably the
        // gate-selection test, which asserts the selected path).
        let _env_guard = parallelism::env_test_lock();
        let prev_threshold = std::env::var_os("SONIUM_COMPUTE_PAR_THRESHOLD");
        let prev_serial = std::env::var_os("SONIUM_SERIAL");
        // SAFETY: holding the intra-crate env lock; restored below.
        unsafe {
            std::env::remove_var("SONIUM_SERIAL");
            std::env::set_var("SONIUM_COMPUTE_PAR_THRESHOLD", "1");
        }
        let threads = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .max(2);
        let n = threads * parallelism::SPMV_ROWS_PER_THREAD + 17;
        let mut row_host = Vec::with_capacity(n + 1);
        let mut col_host = Vec::new();
        let mut val_host = Vec::new();
        row_host.push(0_i32);
        for row in 0..n {
            let nnz = if row % 17 == 0 {
                70
            } else if row % 5 == 0 {
                9
            } else if row % 7 == 0 {
                0
            } else {
                3
            };
            for lane in 0..nnz {
                col_host.push(((row + lane * 13) % n) as i32);
                val_host.push(((row + lane + 1) as f32 * 0.001).sin());
            }
            row_host.push(col_host.len() as i32);
        }
        let x_host: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).cos()).collect();

        let mut row = pool.try_alloc::<i32>(row_host.len()).unwrap();
        let mut col = pool.try_alloc::<i32>(col_host.len()).unwrap();
        let mut val = pool.try_alloc::<f32>(val_host.len()).unwrap();
        let mut x = pool.try_alloc::<f32>(x_host.len()).unwrap();
        let mut y_prebaked = pool.try_alloc::<f32>(n).unwrap();
        let mut y_plain = pool.try_alloc::<f32>(n).unwrap();
        pool.backend().copy_h2d(&row_host, row.raw_mut());
        pool.backend().copy_h2d(&col_host, col.raw_mut());
        pool.backend().copy_h2d(&val_host, val.raw_mut());
        pool.backend().copy_h2d(&x_host, x.raw_mut());

        let prebaked = pool
            .backend()
            .prebake_csr(n, n, row.raw(), col.raw(), val.raw());
        assert_eq!(prebaked.inspector().format, CsrSpmvFormat::RowBucketedCsr);
        pool.backend()
            .spmv_prebaked(&prebaked, x.raw(), y_prebaked.raw_mut());
        pool.backend().spmv_csr(
            n,
            n,
            row.raw(),
            col.raw(),
            val.raw(),
            x.raw(),
            y_plain.raw_mut(),
        );

        let mut got = vec![0.0_f32; n];
        let mut want = vec![0.0_f32; n];
        pool.backend().copy_d2h(y_prebaked.raw(), &mut got);
        pool.backend().copy_d2h(y_plain.raw(), &mut want);
        // Cross-kernel comparison (SIMD row reducers vs scalar
        // accumulation), not a path-equivalence check: f32 summation
        // order legitimately differs at ~1e-6 relative on long rows, so
        // the tolerance is 1e-5. Serial/parallel path equivalence is
        // bitwise by construction (one output row per worker).
        for (g, w) in got.iter().zip(want.iter()) {
            assert_relative_eq!(g, w, max_relative = 1e-5, epsilon = 1e-5);
        }
        // SAFETY: restoring the hermetic gate override saved above.
        unsafe {
            match prev_threshold {
                Some(value) => std::env::set_var("SONIUM_COMPUTE_PAR_THRESHOLD", value),
                None => std::env::remove_var("SONIUM_COMPUTE_PAR_THRESHOLD"),
            }
            match prev_serial {
                Some(value) => std::env::set_var("SONIUM_SERIAL", value),
                None => std::env::remove_var("SONIUM_SERIAL"),
            }
        }
    }

    #[test]
    fn spmv_cpu_uses_parallel_row_distribution() {
        let source = include_str!("mod.rs");
        assert!(
            source.contains("par_iter_mut().enumerate()"),
            "CPU CSR SpMV should distribute independent output rows with rayon"
        );
    }

    #[test]
    fn spmv_cpu_real_path_breaks_accumulator_dependency_chain() {
        let source = include_str!("mod.rs");
        let backend_impl = source
            .split("impl Backend for CpuBackend")
            .nth(1)
            .expect("CPU backend impl should exist")
            .split("#[cfg(test)]")
            .next()
            .expect("backend impl should precede tests");

        assert!(
            backend_impl.contains("spmv_csr_f32") && backend_impl.contains("spmv_csr_f64"),
            "CPU CSR SpMV should use specialized real-valued kernels"
        );
        assert!(
            backend_impl.contains("spmv_csr_bucketed_f32")
                && backend_impl.contains("spmv_csr_bucketed_f64"),
            "prebaked CPU CSR SpMV should consume cached row buckets for real-valued kernels"
        );
        assert!(
            source.contains("spmv_simd::row_f32") && source.contains("spmv_simd::row_f64"),
            "medium/long CPU CSR buckets should route through explicit SIMD row reducers"
        );
        assert!(
            source.contains("sum0") && source.contains("sum1"),
            "real CSR SpMV should use multiple accumulators instead of one scalar reduction chain"
        );
    }

    #[test]
    fn complex_nrm2_does_not_route_through_complex_dot() {
        let source = include_str!("mod.rs");
        let nrm2_body = source
            .split("fn nrm2<T: DeviceScalar>")
            .nth(1)
            .expect("nrm2 implementation should exist")
            .split("fn scale<T: DeviceScalar>")
            .next()
            .expect("nrm2 body should be followed by scale");
        assert!(
            !nrm2_body.contains("dotc_c32") && !nrm2_body.contains("dotc_c64"),
            "complex nrm2 should use a real norm reduction instead of complex dotc"
        );
    }

    #[test]
    fn cpu_uninitialized_alloc_skips_zero_fill() {
        let source = include_str!("mod.rs");
        let backend_impl = source
            .split("impl Backend for CpuBackend")
            .nth(1)
            .expect("CPU backend impl should exist")
            .split("#[cfg(test)]")
            .next()
            .expect("backend impl should precede tests");
        let body = backend_impl
            .split("fn raw_alloc_uninitialized")
            .nth(1)
            .expect("CPU backend should provide an uninitialized allocation path")
            .split("fn total_device_memory_bytes")
            .next()
            .expect("raw_alloc_uninitialized should precede total_device_memory_bytes");

        assert!(
            !body.contains("write_bytes"),
            "uninitialized CPU allocation should not zero-fill scratch buffers"
        );
    }

    /// Big-buffer parity test — exercises the SIMD path of `cpu::blas1`
    /// against a hand-rolled scalar reference. Confirms we're not
    /// dropping accuracy or overrunning buffers.
    #[test]
    fn axpy_f64_simd_path_parity() {
        let n = 4096;
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 24)).unwrap();
        let mut x = pool.try_alloc::<f64>(n).unwrap();
        let mut y = pool.try_alloc::<f64>(n).unwrap();
        let xs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.0123).sin()).collect();
        let ys: Vec<f64> = (0..n).map(|i| (i as f64 * 0.0177).cos()).collect();
        pool.backend().copy_h2d(&xs, x.raw_mut());
        pool.backend().copy_h2d(&ys, y.raw_mut());
        pool.backend().axpy(0.7_f64, x.raw(), y.raw_mut());
        let mut out = vec![0.0; n];
        pool.backend().copy_d2h(y.raw(), &mut out);
        for (i, v) in out.iter().enumerate() {
            let ref_v = ys[i] + 0.7 * xs[i];
            assert_relative_eq!(*v, ref_v, max_relative = 1e-12);
        }
    }

    #[test]
    fn checked_views_and_copies_reject_invalid_shapes() {
        assert_eq!(
            BufferView::new(16, 2, 4, 3).unwrap(),
            BufferView {
                offset: 2,
                len: 4,
                stride: 3,
            }
        );
        assert!(BufferView::new(16, 2, 4, 0).is_err());
        assert!(BufferView::new(16, 14, 2, 2).is_err());

        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let mut dev = pool.try_alloc::<f64>(2).unwrap();
        let error = pool.backend().try_copy_h2d(&[1.0_f64], dev.raw_mut());
        assert!(matches!(
            error,
            Err(crate::BackendError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn checked_dense_dispatch_rejects_invalid_shape_without_mutation() {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let mut a = pool.try_alloc::<f64>(4).unwrap();
        let mut x = pool.try_alloc::<f64>(1).unwrap();
        let mut y = pool.try_alloc::<f64>(2).unwrap();
        pool.backend().copy_h2d(&[1.0, 0.0, 0.0, 1.0], a.raw_mut());
        pool.backend().copy_h2d(&[3.0], x.raw_mut());
        pool.backend().copy_h2d(&[7.0, 11.0], y.raw_mut());
        let error = pool.backend().try_gemv(
            Transpose::NoTrans,
            2,
            2,
            1.0_f64,
            a.raw(),
            2,
            x.raw(),
            0.0,
            y.raw_mut(),
        );
        assert!(matches!(
            error,
            Err(crate::BackendError::InvalidArgument { .. })
        ));
        let mut host = [0.0; 2];
        pool.backend().copy_d2h(y.raw(), &mut host);
        assert_eq!(host, [7.0, 11.0]);
    }

    #[test]
    fn checked_prebaked_spmv_rejects_short_buffers_without_mutation() {
        let backend = CpuBackend::new();
        let mut row_ptr = backend.raw_alloc::<i32>(3).expect("row pointer");
        let mut col_idx = backend.raw_alloc::<i32>(1).expect("column index");
        let mut values = backend.raw_alloc::<f64>(1).expect("matrix values");
        let mut x = backend.raw_alloc::<f64>(1).expect("short input vector");
        let mut y = backend.raw_alloc::<f64>(2).expect("output vector");

        backend.copy_h2d(&[0, 1, 1], &mut row_ptr);
        backend.copy_h2d(&[0], &mut col_idx);
        backend.copy_h2d(&[2.0], &mut values);
        backend.copy_h2d(&[3.0], &mut x);
        backend.copy_h2d(&[7.0, 11.0], &mut y);

        let prebaked = backend
            .try_prebake_csr(2, 2, &row_ptr, &col_idx, &values)
            .expect("valid CSR descriptor");
        let error = backend
            .try_spmv_prebaked(&prebaked, &x, &mut y)
            .expect_err("short input vector must be rejected");
        assert!(matches!(error, BackendError::InvalidArgument { .. }));

        let mut got = [0.0; 2];
        backend.copy_d2h(&y, &mut got);
        assert_eq!(got, [7.0, 11.0]);

        let mut full_x = backend.raw_alloc::<f64>(2).expect("full input vector");
        let mut short_y = backend.raw_alloc::<f64>(1).expect("short output vector");
        backend.copy_h2d(&[3.0, 5.0], &mut full_x);
        backend.copy_h2d(&[13.0], &mut short_y);
        let error = backend
            .try_spmv_prebaked(&prebaked, &full_x, &mut short_y)
            .expect_err("short output vector must be rejected");
        assert!(matches!(error, BackendError::InvalidArgument { .. }));
        let mut short_y_host = [0.0; 1];
        backend.copy_d2h(&short_y, &mut short_y_host);
        assert_eq!(short_y_host, [13.0]);
    }

    #[test]
    fn checked_prebaked_spmv_rejects_malformed_descriptor_without_mutation() {
        let backend = CpuBackend::new();
        let mut row_ptr = backend.raw_alloc::<i32>(3).expect("row pointer");
        let mut col_idx = backend.raw_alloc::<i32>(1).expect("column index");
        let mut values = backend.raw_alloc::<f64>(1).expect("matrix values");
        let mut x = backend.raw_alloc::<f64>(2).expect("input vector");
        let mut y = backend.raw_alloc::<f64>(2).expect("output vector");

        backend.copy_h2d(&[0, 1, 0], &mut row_ptr);
        backend.copy_h2d(&[0], &mut col_idx);
        backend.copy_h2d(&[2.0], &mut values);
        backend.copy_h2d(&[3.0, 5.0], &mut x);
        backend.copy_h2d(&[7.0, 11.0], &mut y);

        let prebaked = backend.prebake_csr(2, 2, &row_ptr, &col_idx, &values);
        let error = backend
            .try_spmv_prebaked(&prebaked, &x, &mut y)
            .expect_err("malformed CSR descriptor must be rejected");
        assert!(matches!(error, BackendError::InvalidArgument { .. }));

        let mut got = [0.0; 2];
        backend.copy_d2h(&y, &mut got);
        assert_eq!(got, [7.0, 11.0]);
    }

    #[test]
    fn checked_prebaked_spmv_rejects_padded_input_without_mutation() {
        let backend = CpuBackend::new();
        let mut row_ptr = backend.raw_alloc::<i32>(3).expect("row pointer");
        let mut col_idx = backend.raw_alloc::<i32>(1).expect("column index");
        let mut values = backend.raw_alloc::<f64>(1).expect("matrix values");
        let mut x = backend.raw_alloc::<f64>(3).expect("padded input vector");
        let mut y = backend.raw_alloc::<f64>(2).expect("output vector");

        backend.copy_h2d(&[0, 1, 1], &mut row_ptr);
        backend.copy_h2d(&[0], &mut col_idx);
        backend.copy_h2d(&[2.0], &mut values);
        backend.copy_h2d(&[3.0, 5.0, f64::NAN], &mut x);
        backend.copy_h2d(&[7.0, 11.0], &mut y);

        let prebaked = backend
            .try_prebake_csr(2, 2, &row_ptr, &col_idx, &values)
            .expect("valid CSR descriptor");
        let error = backend
            .try_spmv_prebaked(&prebaked, &x, &mut y)
            .expect_err("padded input must be rejected for SpMV");
        assert!(matches!(error, BackendError::InvalidArgument { .. }));

        let mut got = [0.0; 2];
        backend.copy_d2h(&y, &mut got);
        assert_eq!(got, [7.0, 11.0]);
    }

    #[test]
    fn checked_gather_and_cast_dispatch_reject_invalid_layouts() {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let values = pool.try_alloc::<f64>(4).unwrap();
        let indices = pool.try_alloc::<i32>(3).unwrap();
        let weights = pool.try_alloc::<f64>(3).unwrap();
        let mut out = pool.try_alloc::<f64>(1).unwrap();
        assert!(matches!(
            pool.backend().try_gather4_weighted(
                values.raw(),
                indices.raw(),
                weights.raw(),
                out.raw_mut(),
            ),
            Err(crate::BackendError::InvalidArgument { .. })
        ));

        let src = pool.try_alloc::<f64>(2).unwrap();
        let mut dst = pool.try_alloc::<f32>(1).unwrap();
        assert!(matches!(
            pool.backend()
                .try_cast_to_lower::<f64>(src.raw(), dst.raw_mut()),
            Err(crate::BackendError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn deterministic_reductions_use_the_explicit_policy() {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let mut x = pool.try_alloc::<f64>(4).unwrap();
        let mut y = pool.try_alloc::<f64>(4).unwrap();
        pool.backend().copy_h2d(&[1.0, 2.0, 3.0, 4.0], x.raw_mut());
        pool.backend().copy_h2d(&[4.0, 3.0, 2.0, 1.0], y.raw_mut());
        assert_eq!(
            pool.backend()
                .dot_with_mode(ReductionMode::Deterministic, x.raw(), y.raw()),
            20.0
        );
        assert_eq!(
            pool.backend()
                .nrm2_with_mode(ReductionMode::Deterministic, x.raw()),
            30.0_f64.sqrt()
        );
    }

    #[test]
    fn oversized_allocations_return_typed_oom_instead_of_panicking() {
        let backend = CpuBackend;
        let error = match backend.raw_alloc::<f64>(usize::MAX) {
            Ok(_) => panic!("oversized allocation must be fallible"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            OomError::DriverOom(_) | OomError::Invalid(_)
        ));

        // The uninitialised scratch path must enforce the same contract.
        let error = match unsafe { backend.raw_alloc_uninitialized::<f64>(usize::MAX) } {
            Ok(_) => panic!("oversized uninitialised allocation must be fallible"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            OomError::DriverOom(_) | OomError::Invalid(_)
        ));
    }
}
