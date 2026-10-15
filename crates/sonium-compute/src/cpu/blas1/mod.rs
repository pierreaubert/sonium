//! In-house BLAS-1 kernels.
//!
//! Exposes the level-1 functions sonium uses: `axpy`, `scal`, `dot`/`dotc`,
//! `nrm2` for f32, f64, Complex32, Complex64. Each is dispatched at compile
//! time:
//!
//! - On `target_arch = "aarch64"` we use NEON intrinsics (always available
//!   on AArch64 — no runtime feature check).
//! - On `target_arch = "x86_64"` we runtime-dispatch to AVX2/FMA kernels
//!   when the host supports them, otherwise use the scalar reference.
//! - On every other target we fall back to the multi-accumulator scalar
//!   reference.
//!
//! Surface is intentionally narrow — every function listed here has a real
//! call site in `sonium-compute::cpu`. New BLAS-1 ops should be added as
//! they are needed, with a parity test in the bench's correctness suite.

pub mod scalar;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(target_arch = "x86_64")]
use num_complex::{Complex32, Complex64};

#[cfg(target_arch = "x86_64")]
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use super::spin_pool::SpinPool;

#[cfg(target_arch = "aarch64")]
use num_complex::{Complex32, Complex64};

// Re-export the active implementation flat into this module's namespace.
// `cpu/mod.rs` and the benchmark both call e.g. `blas1::axpy_f32(...)`
// without caring which path is compiled in.
//
// On aarch64, real-typed `axpy`, `dot`, `nrm2`, and `scal` go through a
// thin dispatcher below that picks SME2 streaming-SVE for long vectors
// (n >= ~4096) and NEON for short ones. Complex BLAS-1 goes straight to NEON.
#[cfg(target_arch = "aarch64")]
pub use neon::{axpy_c32, axpy_c64, dotc_c32, dotc_c64, nrm2_c32, nrm2_c64, scal_c32, scal_c64};

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub use scalar::{
    apply_ab_c32_row, apply_ab_c64_row, axpy_c32, axpy_c64, axpy_f32, axpy_f64, axpy_many_c32,
    axpy_many_c64, axpy_many_f32, axpy_many_f64, dlassq_row_update_f32, dlassq_row_update_f64,
    dot_f32, dot_f64, dotc_c32, dotc_c64, find_non_finite_f32, find_non_finite_f64, ger_update_c32,
    ger_update_c64, ger_update_f32, ger_update_f64, max_norm_checked_c32, max_norm_checked_c64,
    max_norm_f32, max_norm_f64, nrm2_c32, nrm2_c64, nrm2_f32, nrm2_f64, pivot_scan_c32,
    pivot_scan_c64, pivot_scan_f32, pivot_scan_f64, scal_c32, scal_c64, scal_f32, scal_f64,
};

// ---------- x86_64 AVX2/FMA-aware dispatch ----------

/// Total traffic (x read + y read + y write) from which `axpy` fans out
/// over the thread pools. Elementwise, so chunking is bitwise identical
/// to the serial kernel. 256 KB: the spin pool dispatches in ~0.3µs, so
/// even 384 KB vectors (n = 16384 f64/complex32) win parallel; below
/// that the serial AVX2 kernel wins (f32 n = 16384 stays serial at
/// 0.58x). Serial at n = 262144 trails threaded OpenBLAS 8-15x.
#[cfg(target_arch = "x86_64")]
const AXPY_PAR_THRESHOLD_BYTES: usize = 1 << 18;

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len());
    if avx2::is_available() {
        if x.len() * 12 >= AXPY_PAR_THRESHOLD_BYTES {
            if let Some(spin) = spin_pool_for_blas1() {
                spin_axpy_f32(alpha, x, y, spin);
                return;
            }
            if let Some(pool) = spread_pool_for_blas1() {
                axpy_broadcast_f32(alpha, x, y, pool);
                return;
            }
            let chunk = dot_chunk_els(x.len());
            // SAFETY: chunks are disjoint; each runs the AVX2 kernel.
            y.par_chunks_mut(chunk)
                .zip(x.par_chunks(chunk))
                .for_each(|(yy, xx)| unsafe { avx2::axpy_f32(alpha, xx, yy) });
            return;
        }
        unsafe { avx2::axpy_f32(alpha, x, y) }
    } else {
        scalar::axpy_f32(alpha, x, y)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn axpy_f64(alpha: f64, x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    if avx2::is_available() {
        if x.len() * 24 >= AXPY_PAR_THRESHOLD_BYTES {
            if let Some(spin) = spin_pool_for_blas1() {
                spin_axpy_f64(alpha, x, y, spin);
                return;
            }
            if let Some(pool) = spread_pool_for_blas1() {
                axpy_broadcast_f64(alpha, x, y, pool);
                return;
            }
            let chunk = dot_chunk_els(x.len());
            // SAFETY: chunks are disjoint; each runs the AVX2 kernel.
            y.par_chunks_mut(chunk)
                .zip(x.par_chunks(chunk))
                .for_each(|(yy, xx)| unsafe { avx2::axpy_f64(alpha, xx, yy) });
            return;
        }
        unsafe { avx2::axpy_f64(alpha, x, y) }
    } else {
        scalar::axpy_f64(alpha, x, y)
    }
}

/// Four simultaneous dot products sharing one left operand (see
/// `avx2::dot4_f32`): the Cholesky TRSM row kernel's inner loop.
/// Dispatches to AVX2 when available, else four scalar dots.
#[inline]
pub fn dot4_f32(l: &[f32], x: [&[f32]; 4]) -> [f32; 4] {
    #[cfg(target_arch = "x86_64")]
    if avx2::is_available() {
        return unsafe { avx2::dot4_f32(l, x[0], x[1], x[2], x[3]) };
    }
    [
        scalar::dot_f32(l, x[0]),
        scalar::dot_f32(l, x[1]),
        scalar::dot_f32(l, x[2]),
        scalar::dot_f32(l, x[3]),
    ]
}

/// f64 twin of [`dot4_f32`].
#[inline]
pub fn dot4_f64(l: &[f64], x: [&[f64]; 4]) -> [f64; 4] {
    #[cfg(target_arch = "x86_64")]
    if avx2::is_available() {
        return unsafe { avx2::dot4_f64(l, x[0], x[1], x[2], x[3]) };
    }
    [
        scalar::dot_f64(l, x[0]),
        scalar::dot_f64(l, x[1]),
        scalar::dot_f64(l, x[2]),
        scalar::dot_f64(l, x[3]),
    ]
}

/// Fused rank-1 update `block[i*bs+j] += alpha*mults[i*ms] * pivot[j]`
/// (f32). Single-dispatch replacement for a row loop of [`axpy_f32`] calls
/// over the same rows (LU panel factorization, `alpha = -1`). For
/// `alpha = ±1` bitwise identical to that loop on the AVX2 path.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn ger_update_f32(
    alpha: f32,
    mults: &[f32],
    mult_stride: usize,
    pivot: &[f32],
    block: &mut [f32],
    block_stride: usize,
    rows: usize,
    cols: usize,
) {
    if avx2::is_available() {
        unsafe {
            avx2::ger_update_f32(
                alpha,
                mults,
                mult_stride,
                pivot,
                block,
                block_stride,
                rows,
                cols,
            )
        }
    } else {
        scalar::ger_update_f32(
            alpha,
            mults,
            mult_stride,
            pivot,
            block,
            block_stride,
            rows,
            cols,
        )
    }
}

/// Pivot scan `max |base[(first+i)*stride]|` over `len` rows (f32).
/// Vectorized idamax for the LU panel pivot search. Bitwise-identical
/// index/value to the scalar loop on the AVX2 path (first maximum wins,
/// NaN/+Inf initial pivot keeps its row).
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn pivot_scan_f32(base: &[f32], stride: usize, first: usize, len: usize) -> (usize, f32) {
    if avx2::is_available() {
        unsafe { avx2::pivot_scan_f32(base, stride, first, len) }
    } else {
        scalar::pivot_scan_f32(base, stride, first, len)
    }
}

/// Pivot scan `max |base[(first+i)*stride]|` over `len` rows (f64).
/// Same bit-for-bit contract as [`pivot_scan_f32`].
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn pivot_scan_f64(base: &[f64], stride: usize, first: usize, len: usize) -> (usize, f64) {
    if avx2::is_available() {
        unsafe { avx2::pivot_scan_f64(base, stride, first, len) }
    } else {
        scalar::pivot_scan_f64(base, stride, first, len)
    }
}

/// Fused rank-1 update `block[i*bs+j] += alpha*mults[i*ms] * pivot[j]`
/// (f64). Single-dispatch replacement for a row loop of [`axpy_f64`] calls
/// over the same rows (LU panel factorization, `alpha = -1`). For
/// `alpha = ±1` bitwise identical to that loop on the AVX2 path.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn ger_update_f64(
    alpha: f64,
    mults: &[f64],
    mult_stride: usize,
    pivot: &[f64],
    block: &mut [f64],
    block_stride: usize,
    rows: usize,
    cols: usize,
) {
    if avx2::is_available() {
        unsafe {
            avx2::ger_update_f64(
                alpha,
                mults,
                mult_stride,
                pivot,
                block,
                block_stride,
                rows,
                cols,
            )
        }
    } else {
        scalar::ger_update_f64(
            alpha,
            mults,
            mult_stride,
            pivot,
            block,
            block_stride,
            rows,
            cols,
        )
    }
}

/// Diagnostic hook: `SONIUM_COMPLEX_SCALAR=1` forces the scalar reference
/// for the eight complex scan/GER/norm/epilogue dispatches below (A/B
/// against the AVX2 kernels in one binary; the pre-kernel behavior was
/// exactly this scalar path via the `ComplexField` defaults and the inline
/// 4M combine loop).
#[cfg(target_arch = "x86_64")]
#[inline]
fn complex_scalar_forced() -> bool {
    static FORCED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FORCED.get_or_init(|| std::env::var("SONIUM_COMPLEX_SCALAR").is_ok_and(|v| v != "0"))
}

/// Complex64 pivot scan: `max base[(first+i)*stride].norm()` (true
/// modulus) over `len` rows, first maximum wins. AVX2 `sqrt(re²+im²)` fold
/// when available, scalar `hypot` reference otherwise (identical except
/// sub-ulp ties; see `avx2::pivot_scan_c64`). LU panel pivot search.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn pivot_scan_c64(base: &[Complex64], stride: usize, first: usize, len: usize) -> (usize, f64) {
    if avx2::is_available() && !complex_scalar_forced() {
        unsafe { avx2::pivot_scan_c64(base, stride, first, len) }
    } else {
        scalar::pivot_scan_c64(base, stride, first, len)
    }
}

/// Complex32 twin of [`pivot_scan_c64`].
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn pivot_scan_c32(base: &[Complex32], stride: usize, first: usize, len: usize) -> (usize, f32) {
    if avx2::is_available() && !complex_scalar_forced() {
        unsafe { avx2::pivot_scan_c32(base, stride, first, len) }
    } else {
        scalar::pivot_scan_c32(base, stride, first, len)
    }
}

/// Fused complex rank-1 update `block[i*bs+j] += alpha*mults[i*ms] *
/// pivot[j]` (Complex64). AVX2 permute-based kernel when available, scalar
/// axpy-per-row reference otherwise. LU panel leaf.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn ger_update_c64(
    alpha: Complex64,
    mults: &[Complex64],
    mult_stride: usize,
    pivot: &[Complex64],
    block: &mut [Complex64],
    block_stride: usize,
    rows: usize,
    cols: usize,
) {
    if avx2::is_available() && !complex_scalar_forced() {
        unsafe {
            avx2::ger_update_c64(
                alpha,
                mults,
                mult_stride,
                pivot,
                block,
                block_stride,
                rows,
                cols,
            )
        }
    } else {
        scalar::ger_update_c64(
            alpha,
            mults,
            mult_stride,
            pivot,
            block,
            block_stride,
            rows,
            cols,
        )
    }
}

/// Fused complex entry-validation fold `(max norm, all_finite)` in one pass
/// (Complex64). AVX2 `hadd`/`sqrt` fold when available, scalar `hypot`
/// reference otherwise (flag bit-exact, max identical except sub-ulp ties;
/// see `avx2::max_norm_checked_c64`). LU entry scan.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn max_norm_checked_c64(values: &[Complex64]) -> (f64, bool) {
    if avx2::is_available() && !complex_scalar_forced() {
        unsafe { avx2::max_norm_checked_c64(values) }
    } else {
        scalar::max_norm_checked_c64(values)
    }
}

/// Complex32 twin of [`max_norm_checked_c64`].
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn max_norm_checked_c32(values: &[Complex32]) -> (f32, bool) {
    if avx2::is_available() && !complex_scalar_forced() {
        unsafe { avx2::max_norm_checked_c32(values) }
    } else {
        scalar::max_norm_checked_c32(values)
    }
}

/// 4M epilogue row `C := α·(R + i·I) + β·C` (Complex64) over one contiguous
/// output row. AVX2 unpack-based kernel when available, scalar reference
/// otherwise (bitwise-identical). Complex-GEMM combine pass.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn apply_ab_c64_row(
    r: &[f64],
    imag: &[f64],
    c: &mut [Complex64],
    alpha: Complex64,
    beta: Complex64,
    beta_zero: bool,
) {
    if avx2::is_available() && !complex_scalar_forced() {
        unsafe { avx2::apply_ab_c64_row(r, imag, c, alpha, beta, beta_zero) }
    } else {
        scalar::apply_ab_c64_row(r, imag, c, alpha, beta, beta_zero)
    }
}

/// Complex32 twin of [`apply_ab_c64_row`].
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn apply_ab_c32_row(
    r: &[f32],
    imag: &[f32],
    c: &mut [Complex32],
    alpha: Complex32,
    beta: Complex32,
    beta_zero: bool,
) {
    if avx2::is_available() && !complex_scalar_forced() {
        unsafe { avx2::apply_ab_c32_row(r, imag, c, alpha, beta, beta_zero) }
    } else {
        scalar::apply_ab_c32_row(r, imag, c, alpha, beta, beta_zero)
    }
}

/// 3M epilogue row (Complex64): [`apply_ab_c64_row`] with the 3M
/// intermediates (`R = Ar·Br`, `Q = Ai·Bi`, `I = (Ar+Ai)·(Br+Bi)`).
/// AVX2 kernel when available, scalar reference otherwise
/// (bitwise-identical).
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn apply_ab3_c64_row(
    r: &[f64],
    q: &[f64],
    imag: &[f64],
    c: &mut [Complex64],
    alpha: Complex64,
    beta: Complex64,
    beta_zero: bool,
) {
    if avx2::is_available() && !complex_scalar_forced() {
        unsafe { avx2::apply_ab3_c64_row(r, q, imag, c, alpha, beta, beta_zero) }
    } else {
        scalar::apply_ab3_c64_row(r, q, imag, c, alpha, beta, beta_zero)
    }
}

/// Complex32 twin of [`apply_ab3_c64_row`].
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn apply_ab3_c32_row(
    r: &[f32],
    q: &[f32],
    imag: &[f32],
    c: &mut [Complex32],
    alpha: Complex32,
    beta: Complex32,
    beta_zero: bool,
) {
    if avx2::is_available() && !complex_scalar_forced() {
        unsafe { avx2::apply_ab3_c32_row(r, q, imag, c, alpha, beta, beta_zero) }
    } else {
        scalar::apply_ab3_c32_row(r, q, imag, c, alpha, beta, beta_zero)
    }
}

/// Complex32 twin of [`ger_update_c64`].
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn ger_update_c32(
    alpha: Complex32,
    mults: &[Complex32],
    mult_stride: usize,
    pivot: &[Complex32],
    block: &mut [Complex32],
    block_stride: usize,
    rows: usize,
    cols: usize,
) {
    if avx2::is_available() && !complex_scalar_forced() {
        unsafe {
            avx2::ger_update_c32(
                alpha,
                mults,
                mult_stride,
                pivot,
                block,
                block_stride,
                rows,
                cols,
            )
        }
    } else {
        scalar::ger_update_c32(
            alpha,
            mults,
            mult_stride,
            pivot,
            block,
            block_stride,
            rows,
            cols,
        )
    }
}

/// Max `|v|` over f32 lanes (`0` when empty). Vectorized entry scale fold.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn max_norm_f32(values: &[f32]) -> f32 {
    if avx2::is_available() {
        unsafe { avx2::max_norm_f32(values) }
    } else {
        scalar::max_norm_f32(values)
    }
}

/// Max `|v|` over f64 lanes (`0` when empty). Same contract as [`max_norm_f32`].
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn max_norm_f64(values: &[f64]) -> f64 {
    if avx2::is_available() {
        unsafe { avx2::max_norm_f64(values) }
    } else {
        scalar::max_norm_f64(values)
    }
}

/// Fused entry-validation fold `(max |v|, all_finite)` in one pass (f32).
/// Same max value as [`max_norm_f32`] when the flag holds; unspecified
/// otherwise. Callers rescan for the first bad index on `false` (rare).
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn max_norm_checked_f32(values: &[f32]) -> (f32, bool) {
    if avx2::is_available() {
        unsafe { avx2::max_norm_checked_f32(values) }
    } else {
        scalar::max_norm_checked_f32(values)
    }
}

/// Fused entry-validation fold `(max |v|, all_finite)` in one pass (f64).
/// Same contract as [`max_norm_checked_f32`].
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn max_norm_checked_f64(values: &[f64]) -> (f64, bool) {
    if avx2::is_available() {
        unsafe { avx2::max_norm_checked_f64(values) }
    } else {
        scalar::max_norm_checked_f64(values)
    }
}

/// First index of a non-finite lane (NaN or ±inf), f32.
/// Vectorized entry/end-scan for LU/Cholesky validation.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn find_non_finite_f32(values: &[f32]) -> Option<usize> {
    if avx2::is_available() {
        unsafe { avx2::find_non_finite_f32(values) }
    } else {
        scalar::find_non_finite_f32(values)
    }
}

/// First index of a non-finite lane (NaN or ±inf), f64.
/// Same contract as [`find_non_finite_f32`].
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn find_non_finite_f64(values: &[f64]) -> Option<usize> {
    if avx2::is_available() {
        unsafe { avx2::find_non_finite_f64(values) }
    } else {
        scalar::find_non_finite_f64(values)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn axpy_c32(alpha: Complex32, x: &[Complex32], y: &mut [Complex32]) {
    debug_assert_eq!(x.len(), y.len());
    if avx2::is_available() {
        if x.len() * 24 >= AXPY_PAR_THRESHOLD_BYTES {
            if let Some(spin) = spin_pool_for_blas1() {
                spin_axpy_c32(alpha, x, y, spin);
                return;
            }
            if let Some(pool) = spread_pool_for_blas1() {
                axpy_broadcast_c32(alpha, x, y, pool);
                return;
            }
            let chunk = dot_chunk_els(x.len());
            // SAFETY: chunks are disjoint; each runs the AVX2 kernel.
            y.par_chunks_mut(chunk)
                .zip(x.par_chunks(chunk))
                .for_each(|(yy, xx)| unsafe { avx2::axpy_c32(alpha, xx, yy) });
            return;
        }
        unsafe { avx2::axpy_c32(alpha, x, y) }
    } else {
        scalar::axpy_c32(alpha, x, y)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn axpy_c64(alpha: Complex64, x: &[Complex64], y: &mut [Complex64]) {
    debug_assert_eq!(x.len(), y.len());
    if avx2::is_available() {
        if x.len() * 48 >= AXPY_PAR_THRESHOLD_BYTES {
            if let Some(spin) = spin_pool_for_blas1() {
                spin_axpy_c64(alpha, x, y, spin);
                return;
            }
            if let Some(pool) = spread_pool_for_blas1() {
                axpy_broadcast_c64(alpha, x, y, pool);
                return;
            }
            let chunk = dot_chunk_els(x.len());
            // SAFETY: chunks are disjoint; each runs the AVX2 kernel.
            y.par_chunks_mut(chunk)
                .zip(x.par_chunks(chunk))
                .for_each(|(yy, xx)| unsafe { avx2::axpy_c64(alpha, xx, yy) });
            return;
        }
        unsafe { avx2::axpy_c64(alpha, x, y) }
    } else {
        scalar::axpy_c64(alpha, x, y)
    }
}

/// Total traffic (x read + x write) from which `scal` fans out over
/// rayon. Elementwise, so chunking is bitwise identical to the serial
/// kernel. 32 MB keeps both 1M vectors serial (8/16 MB traffic, where the
/// serial kernel already wins) while DRAM-scale vectors fan out (serial
/// trails threaded OpenBLAS ~4.4-5.7x at n = 64M without this fan-out).
#[cfg(target_arch = "x86_64")]
const SCAL_PAR_THRESHOLD_BYTES: usize = 1 << 25;

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn scal_f32(alpha: f32, x: &mut [f32]) {
    if avx2::is_available() {
        if x.len() * 8 >= SCAL_PAR_THRESHOLD_BYTES {
            if let Some(spin) = spin_pool_for_blas1() {
                spin_scal_f32(alpha, x, spin);
                return;
            }
            if let Some(pool) = spread_pool_for_blas1() {
                scal_broadcast_f32(alpha, x, pool);
                return;
            }
            let chunk = dot_chunk_els(x.len());
            // SAFETY: chunks are disjoint; each runs the AVX2 kernel.
            x.par_chunks_mut(chunk)
                .for_each(|xx| unsafe { avx2::scal_f32(alpha, xx) });
            return;
        }
        unsafe { avx2::scal_f32(alpha, x) }
    } else {
        scalar::scal_f32(alpha, x)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn scal_f64(alpha: f64, x: &mut [f64]) {
    if avx2::is_available() {
        if x.len() * 16 >= SCAL_PAR_THRESHOLD_BYTES {
            if let Some(spin) = spin_pool_for_blas1() {
                spin_scal_f64(alpha, x, spin);
                return;
            }
            if let Some(pool) = spread_pool_for_blas1() {
                scal_broadcast_f64(alpha, x, pool);
                return;
            }
            let chunk = dot_chunk_els(x.len());
            // SAFETY: chunks are disjoint; each runs the AVX2 kernel.
            x.par_chunks_mut(chunk)
                .for_each(|xx| unsafe { avx2::scal_f64(alpha, xx) });
            return;
        }
        unsafe { avx2::scal_f64(alpha, x) }
    } else {
        scalar::scal_f64(alpha, x)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn scal_c32(alpha: Complex32, x: &mut [Complex32]) {
    if avx2::is_available() {
        if x.len() * 16 >= SCAL_PAR_THRESHOLD_BYTES {
            if let Some(spin) = spin_pool_for_blas1() {
                spin_scal_c32(alpha, x, spin);
                return;
            }
            if let Some(pool) = spread_pool_for_blas1() {
                scal_broadcast_c32(alpha, x, pool);
                return;
            }
            let chunk = dot_chunk_els(x.len());
            // SAFETY: chunks are disjoint; each runs the AVX2 kernel.
            x.par_chunks_mut(chunk)
                .for_each(|xx| unsafe { avx2::scal_c32(alpha, xx) });
            return;
        }
        unsafe { avx2::scal_c32(alpha, x) }
    } else {
        scalar::scal_c32(alpha, x)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn scal_c64(alpha: Complex64, x: &mut [Complex64]) {
    if avx2::is_available() {
        if x.len() * 32 >= SCAL_PAR_THRESHOLD_BYTES {
            if let Some(spin) = spin_pool_for_blas1() {
                spin_scal_c64(alpha, x, spin);
                return;
            }
            if let Some(pool) = spread_pool_for_blas1() {
                scal_broadcast_c64(alpha, x, pool);
                return;
            }
            let chunk = dot_chunk_els(x.len());
            // SAFETY: chunks are disjoint; each runs the AVX2 kernel.
            x.par_chunks_mut(chunk)
                .for_each(|xx| unsafe { avx2::scal_c64(alpha, xx) });
            return;
        }
        unsafe { avx2::scal_c64(alpha, x) }
    } else {
        scalar::scal_c64(alpha, x)
    }
}

/// Total traffic (x + y) from which `dot` fans out over the thread
/// pools. Dot is streaming (bandwidth-bound); the spin pool dispatches
/// in ~0.3µs, so 512 KB already wins parallel, and chunked partial sums
/// only perturb rounding (~1e-7 relative), inside the 1e-5-scale parity
/// tolerances. Unit tests top out at n = 4099, far below. Scaling past
/// ~8 threads is placement-sensitive (pinned threads hold parity with
/// OpenBLAS; unpinned all-core runs trail it) — a threading runtime
/// concern, not a kernel gap.
#[cfg(target_arch = "x86_64")]
const DOT_PAR_THRESHOLD_BYTES: usize = 1 << 19;
/// Minimum elements per worker chunk (512 KB for f64: L2-friendly).
#[cfg(target_arch = "x86_64")]
const DOT_MIN_CHUNK_ELS: usize = 1 << 16;

#[cfg(target_arch = "x86_64")]
#[inline]
fn dot_chunk_els(n: usize) -> usize {
    let workers = rayon::current_num_threads().max(1);
    (n / workers.saturating_mul(4).max(1)).max(DOT_MIN_CHUNK_ELS)
}

/// Pointer wrappers for the spread-broadcast closures below (same shape as
/// `blas2::avx2`'s, which are private to that module).
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct SendConstPtr<T>(*const T);
#[cfg(target_arch = "x86_64")]
unsafe impl<T> Send for SendConstPtr<T> {}
#[cfg(target_arch = "x86_64")]
unsafe impl<T> Sync for SendConstPtr<T> {}
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct SendMutPtr<T>(*mut T);
#[cfg(target_arch = "x86_64")]
unsafe impl<T> Send for SendMutPtr<T> {}
#[cfg(target_arch = "x86_64")]
unsafe impl<T> Sync for SendMutPtr<T> {}

/// Spread pool for MT BLAS-1 (`dot`/`axpy`/`scal`, real and complex), if
/// the broadcast arm applies: default-on via the shared spread-pool
/// switch (`SONIUM_GEMV_PINNED=0` restores legacy stealing `par_chunks`;
/// read per call so same-process A/B can interleave both arms), top-level
/// calls only (broadcast from inside a pool risks deadlock), and only
/// when the fixed-width spread pool fits the ambient pool (else fall
/// through to the legacy path, no oversubscription).
///
/// Why: stealing `par_chunks` reshuffles the chunk→CPU map every rep, so
/// L3-resident vectors (ddot n = 1M: 16 MB) pay fabric/DRAM (~150 GB/s)
/// instead of local slices (OpenBLAS ~550 GB/s; our ST kernel is at
/// parity, 0.84x, so this is purely placement). Static index-addressed
/// bands settle slice-local across reps — the same mechanism as the
/// GEMV NoTrans/Trans broadcast ships. Partial-sum order differs from
/// the legacy path, so agreement is within tolerance, not bitwise.
#[cfg(target_arch = "x86_64")]
#[inline]
fn spread_pool_for_blas1() -> Option<&'static rayon::ThreadPool> {
    let pinned = std::env::var("SONIUM_GEMV_PINNED")
        .map(|v| v != "0")
        .unwrap_or(true);
    if !pinned || rayon::current_thread_index().is_some() {
        return None;
    }
    let pool = crate::cpu::blas3::spread_pool()?;
    if pool.current_num_threads() > rayon::current_num_threads() {
        return None;
    }
    Some(pool)
}

/// Spin pool for MT BLAS-1, tried before the rayon spread broadcast:
/// rayon broadcast costs ~5µs at 16 workers (empty-body median), which
/// dominates L3-resident vectors, while the spin pool dispatches in
/// ~0.1-0.3µs. Default-on with `SONIUM_BLAS1_SPIN=0` restoring the rayon
/// path for same-binary A/B; top-level calls only (a broadcast from a
/// rayon or spin worker would deadlock), and only when the spin width
/// fits the ambient pool (else fall through, no oversubscription).
#[cfg(target_arch = "x86_64")]
#[inline]
fn spin_pool_for_blas1() -> Option<&'static SpinPool> {
    let enabled = std::env::var("SONIUM_BLAS1_SPIN")
        .map(|v| v != "0")
        .unwrap_or(true);
    if !enabled || rayon::current_thread_index().is_some() || SpinPool::on_worker() {
        return None;
    }
    let pool = super::spin_pool::spin_pool()?;
    if pool.width() > rayon::current_num_threads() {
        return None;
    }
    Some(pool)
}

/// `axpy` over spin-pool bands (f32). Elementwise, so bitwise identical
/// to the serial kernel and the rayon paths.
#[cfg(target_arch = "x86_64")]
#[inline]
fn spin_axpy_f32(alpha: f32, x: &[f32], y: &mut [f32], pool: &SpinPool) {
    let x_send = SendConstPtr(x.as_ptr());
    let y_send = SendMutPtr(y.as_mut_ptr());
    pool.broadcast_bands_k(
        x.len(),
        sonium_topology::bandwidth::stream_width(),
        &|_, t0, t1| {
            let x_send: SendConstPtr<f32> = x_send;
            let y_send: SendMutPtr<f32> = y_send;
            // SAFETY: bands are disjoint in-bounds slices; `broadcast_bands`
            // joins before returning. AVX2 checked by caller.
            unsafe {
                let xb = std::slice::from_raw_parts(x_send.0.add(t0), t1 - t0);
                let yb = std::slice::from_raw_parts_mut(y_send.0.add(t0), t1 - t0);
                avx2::axpy_f32(alpha, xb, yb);
            }
        },
    );
}

/// `axpy` over spin-pool bands (f64). Same contract as
/// [`spin_axpy_f32`].
#[cfg(target_arch = "x86_64")]
#[inline]
fn spin_axpy_f64(alpha: f64, x: &[f64], y: &mut [f64], pool: &SpinPool) {
    let x_send = SendConstPtr(x.as_ptr());
    let y_send = SendMutPtr(y.as_mut_ptr());
    pool.broadcast_bands_k(
        x.len(),
        sonium_topology::bandwidth::stream_width(),
        &|_, t0, t1| {
            let x_send: SendConstPtr<f64> = x_send;
            let y_send: SendMutPtr<f64> = y_send;
            // SAFETY: bands are disjoint in-bounds slices; `broadcast_bands`
            // joins before returning. AVX2 checked by caller.
            unsafe {
                let xb = std::slice::from_raw_parts(x_send.0.add(t0), t1 - t0);
                let yb = std::slice::from_raw_parts_mut(y_send.0.add(t0), t1 - t0);
                avx2::axpy_f64(alpha, xb, yb);
            }
        },
    );
}

/// `axpy` over spin-pool bands (Complex32). Same contract as
/// [`spin_axpy_f32`].
#[cfg(target_arch = "x86_64")]
#[inline]
fn spin_axpy_c32(alpha: Complex32, x: &[Complex32], y: &mut [Complex32], pool: &SpinPool) {
    let x_send = SendConstPtr(x.as_ptr());
    let y_send = SendMutPtr(y.as_mut_ptr());
    pool.broadcast_bands_k(
        x.len(),
        sonium_topology::bandwidth::stream_width(),
        &|_, t0, t1| {
            let x_send: SendConstPtr<Complex32> = x_send;
            let y_send: SendMutPtr<Complex32> = y_send;
            // SAFETY: bands are disjoint in-bounds slices; `broadcast_bands`
            // joins before returning. AVX2 checked by caller.
            unsafe {
                let xb = std::slice::from_raw_parts(x_send.0.add(t0), t1 - t0);
                let yb = std::slice::from_raw_parts_mut(y_send.0.add(t0), t1 - t0);
                avx2::axpy_c32(alpha, xb, yb);
            }
        },
    );
}

/// `axpy` over spin-pool bands (Complex64). Same contract as
/// [`spin_axpy_f32`].
#[cfg(target_arch = "x86_64")]
#[inline]
fn spin_axpy_c64(alpha: Complex64, x: &[Complex64], y: &mut [Complex64], pool: &SpinPool) {
    let x_send = SendConstPtr(x.as_ptr());
    let y_send = SendMutPtr(y.as_mut_ptr());
    pool.broadcast_bands_k(
        x.len(),
        sonium_topology::bandwidth::stream_width(),
        &|_, t0, t1| {
            let x_send: SendConstPtr<Complex64> = x_send;
            let y_send: SendMutPtr<Complex64> = y_send;
            // SAFETY: bands are disjoint in-bounds slices; `broadcast_bands`
            // joins before returning. AVX2 checked by caller.
            unsafe {
                let xb = std::slice::from_raw_parts(x_send.0.add(t0), t1 - t0);
                let yb = std::slice::from_raw_parts_mut(y_send.0.add(t0), t1 - t0);
                avx2::axpy_c64(alpha, xb, yb);
            }
        },
    );
}

/// `scal` over spin-pool bands (f32). Elementwise, so bitwise identical
/// to the serial kernel and the rayon paths.
#[cfg(target_arch = "x86_64")]
#[inline]
fn spin_scal_f32(alpha: f32, x: &mut [f32], pool: &SpinPool) {
    let x_send = SendMutPtr(x.as_mut_ptr());
    pool.broadcast_bands_k(
        x.len(),
        sonium_topology::bandwidth::stream_width(),
        &|_, t0, t1| {
            let x_send: SendMutPtr<f32> = x_send;
            // SAFETY: bands are disjoint in-bounds slices; `broadcast_bands`
            // joins before returning. AVX2 checked by caller.
            unsafe {
                let xb = std::slice::from_raw_parts_mut(x_send.0.add(t0), t1 - t0);
                avx2::scal_f32(alpha, xb);
            }
        },
    );
}

/// `scal` over spin-pool bands (f64). Same contract as
/// [`spin_scal_f32`].
#[cfg(target_arch = "x86_64")]
#[inline]
fn spin_scal_f64(alpha: f64, x: &mut [f64], pool: &SpinPool) {
    let x_send = SendMutPtr(x.as_mut_ptr());
    pool.broadcast_bands_k(
        x.len(),
        sonium_topology::bandwidth::stream_width(),
        &|_, t0, t1| {
            let x_send: SendMutPtr<f64> = x_send;
            // SAFETY: bands are disjoint in-bounds slices; `broadcast_bands`
            // joins before returning. AVX2 checked by caller.
            unsafe {
                let xb = std::slice::from_raw_parts_mut(x_send.0.add(t0), t1 - t0);
                avx2::scal_f64(alpha, xb);
            }
        },
    );
}

/// `scal` over spin-pool bands (Complex32). Same contract as
/// [`spin_scal_f32`].
#[cfg(target_arch = "x86_64")]
#[inline]
fn spin_scal_c32(alpha: Complex32, x: &mut [Complex32], pool: &SpinPool) {
    let x_send = SendMutPtr(x.as_mut_ptr());
    pool.broadcast_bands_k(
        x.len(),
        sonium_topology::bandwidth::stream_width(),
        &|_, t0, t1| {
            let x_send: SendMutPtr<Complex32> = x_send;
            // SAFETY: bands are disjoint in-bounds slices; `broadcast_bands`
            // joins before returning. AVX2 checked by caller.
            unsafe {
                let xb = std::slice::from_raw_parts_mut(x_send.0.add(t0), t1 - t0);
                avx2::scal_c32(alpha, xb);
            }
        },
    );
}

/// `scal` over spin-pool bands (Complex64). Same contract as
/// [`spin_scal_f32`].
#[cfg(target_arch = "x86_64")]
#[inline]
fn spin_scal_c64(alpha: Complex64, x: &mut [Complex64], pool: &SpinPool) {
    let x_send = SendMutPtr(x.as_mut_ptr());
    pool.broadcast_bands_k(
        x.len(),
        sonium_topology::bandwidth::stream_width(),
        &|_, t0, t1| {
            let x_send: SendMutPtr<Complex64> = x_send;
            // SAFETY: bands are disjoint in-bounds slices; `broadcast_bands`
            // joins before returning. AVX2 checked by caller.
            unsafe {
                let xb = std::slice::from_raw_parts_mut(x_send.0.add(t0), t1 - t0);
                avx2::scal_c64(alpha, xb);
            }
        },
    );
}

/// `dot` over spin-pool bands (f32): band partials reduce in slice order
/// on the caller. Order differs from the serial kernel, so agreement is
/// within tolerance, not bitwise (same as the rayon paths).
#[cfg(target_arch = "x86_64")]
#[inline]
fn spin_dot_f32(x: &[f32], y: &[f32], pool: &SpinPool) -> f32 {
    let mut partials = vec![0.0f32; pool.width()];
    let x_send = SendConstPtr(x.as_ptr());
    let y_send = SendConstPtr(y.as_ptr());
    let p_send = SendMutPtr(partials.as_mut_ptr());
    pool.broadcast_bands_k(
        x.len(),
        sonium_topology::bandwidth::stream_width(),
        &|w, t0, t1| {
            let x_send: SendConstPtr<f32> = x_send;
            let y_send: SendConstPtr<f32> = y_send;
            let p_send: SendMutPtr<f32> = p_send;
            // SAFETY: band slices are in bounds; `partials[w]` is owned by
            // worker `w`; `broadcast_bands` joins before returning. AVX2
            // checked by caller.
            unsafe {
                let xb = std::slice::from_raw_parts(x_send.0.add(t0), t1 - t0);
                let yb = std::slice::from_raw_parts(y_send.0.add(t0), t1 - t0);
                *p_send.0.add(w) = avx2::dot_f32(xb, yb);
            }
        },
    );
    partials.iter().sum()
}

/// `dot` over spin-pool bands (f64). Same contract as
/// [`spin_dot_f32`].
#[cfg(target_arch = "x86_64")]
#[inline]
fn spin_dot_f64(x: &[f64], y: &[f64], pool: &SpinPool) -> f64 {
    let mut partials = vec![0.0f64; pool.width()];
    let x_send = SendConstPtr(x.as_ptr());
    let y_send = SendConstPtr(y.as_ptr());
    let p_send = SendMutPtr(partials.as_mut_ptr());
    pool.broadcast_bands_k(
        x.len(),
        sonium_topology::bandwidth::stream_width(),
        &|w, t0, t1| {
            let x_send: SendConstPtr<f64> = x_send;
            let y_send: SendConstPtr<f64> = y_send;
            let p_send: SendMutPtr<f64> = p_send;
            // SAFETY: band slices are in bounds; `partials[w]` is owned by
            // worker `w`; `broadcast_bands` joins before returning. AVX2
            // checked by caller.
            unsafe {
                let xb = std::slice::from_raw_parts(x_send.0.add(t0), t1 - t0);
                let yb = std::slice::from_raw_parts(y_send.0.add(t0), t1 - t0);
                *p_send.0.add(w) = avx2::dot_f64(xb, yb);
            }
        },
    );
    partials.iter().sum()
}

/// `dotc` over spin-pool bands (Complex32). Same contract as
/// [`spin_dot_f32`].
#[cfg(target_arch = "x86_64")]
#[inline]
fn spin_dotc_c32(x: &[Complex32], y: &[Complex32], pool: &SpinPool) -> Complex32 {
    let mut partials = vec![Complex32::new(0.0, 0.0); pool.width()];
    let x_send = SendConstPtr(x.as_ptr());
    let y_send = SendConstPtr(y.as_ptr());
    let p_send = SendMutPtr(partials.as_mut_ptr());
    pool.broadcast_bands_k(
        x.len(),
        sonium_topology::bandwidth::stream_width(),
        &|w, t0, t1| {
            let x_send: SendConstPtr<Complex32> = x_send;
            let y_send: SendConstPtr<Complex32> = y_send;
            let p_send: SendMutPtr<Complex32> = p_send;
            // SAFETY: band slices are in bounds; `partials[w]` is owned by
            // worker `w`; `broadcast_bands` joins before returning. AVX2
            // checked by caller.
            unsafe {
                let xb = std::slice::from_raw_parts(x_send.0.add(t0), t1 - t0);
                let yb = std::slice::from_raw_parts(y_send.0.add(t0), t1 - t0);
                *p_send.0.add(w) = avx2::dotc_c32(xb, yb);
            }
        },
    );
    partials.iter().sum()
}

/// `dotc` over spin-pool bands (Complex64). Same contract as
/// [`spin_dot_f32`].
#[cfg(target_arch = "x86_64")]
#[inline]
fn spin_dotc_c64(x: &[Complex64], y: &[Complex64], pool: &SpinPool) -> Complex64 {
    let mut partials = vec![Complex64::new(0.0, 0.0); pool.width()];
    let x_send = SendConstPtr(x.as_ptr());
    let y_send = SendConstPtr(y.as_ptr());
    let p_send = SendMutPtr(partials.as_mut_ptr());
    pool.broadcast_bands_k(
        x.len(),
        sonium_topology::bandwidth::stream_width(),
        &|w, t0, t1| {
            let x_send: SendConstPtr<Complex64> = x_send;
            let y_send: SendConstPtr<Complex64> = y_send;
            let p_send: SendMutPtr<Complex64> = p_send;
            // SAFETY: band slices are in bounds; `partials[w]` is owned by
            // worker `w`; `broadcast_bands` joins before returning. AVX2
            // checked by caller.
            unsafe {
                let xb = std::slice::from_raw_parts(x_send.0.add(t0), t1 - t0);
                let yb = std::slice::from_raw_parts(y_send.0.add(t0), t1 - t0);
                *p_send.0.add(w) = avx2::dotc_c64(xb, yb);
            }
        },
    );
    partials.iter().sum()
}

/// `dot` over static spread bands (f32): worker `w` owns elements
/// `[w*epb, (w+1)*epb)`; partials reduce in slice order on the caller.
#[cfg(target_arch = "x86_64")]
#[inline]
fn dot_broadcast_f32(x: &[f32], y: &[f32], pool: &rayon::ThreadPool) -> f32 {
    let n = x.len();
    let width = pool.current_num_threads().max(1);
    let bands = width.min(n).max(1);
    let els_per_band = n.div_ceil(bands);
    let x_send = SendConstPtr(x.as_ptr());
    let y_send = SendConstPtr(y.as_ptr());
    let mut partials = vec![0.0f32; width];
    let p_send = SendMutPtr(partials.as_mut_ptr());
    pool.broadcast(|ctx| {
        let x_send: SendConstPtr<f32> = x_send;
        let y_send: SendConstPtr<f32> = y_send;
        let p_send: SendMutPtr<f32> = p_send;
        let w = ctx.index();
        let t0 = w.saturating_mul(els_per_band).min(n);
        let t1 = t0.saturating_add(els_per_band).min(n);
        if t0 < t1 {
            // SAFETY: band slices are in bounds; `partials[w]` is owned
            // by worker `w`; `broadcast` joins before returning, so the
            // borrows cannot outlive the call. AVX2 checked by caller.
            unsafe {
                let xb = std::slice::from_raw_parts(x_send.0.add(t0), t1 - t0);
                let yb = std::slice::from_raw_parts(y_send.0.add(t0), t1 - t0);
                *p_send.0.add(w) = avx2::dot_f32(xb, yb);
            }
        }
    });
    partials.iter().sum()
}

/// `dot` over static spread bands (f64). Same contract as
/// [`dot_broadcast_f32`].
#[cfg(target_arch = "x86_64")]
#[inline]
fn dot_broadcast_f64(x: &[f64], y: &[f64], pool: &rayon::ThreadPool) -> f64 {
    let n = x.len();
    let width = pool.current_num_threads().max(1);
    let bands = width.min(n).max(1);
    let els_per_band = n.div_ceil(bands);
    let x_send = SendConstPtr(x.as_ptr());
    let y_send = SendConstPtr(y.as_ptr());
    let mut partials = vec![0.0f64; width];
    let p_send = SendMutPtr(partials.as_mut_ptr());
    pool.broadcast(|ctx| {
        let x_send: SendConstPtr<f64> = x_send;
        let y_send: SendConstPtr<f64> = y_send;
        let p_send: SendMutPtr<f64> = p_send;
        let w = ctx.index();
        let t0 = w.saturating_mul(els_per_band).min(n);
        let t1 = t0.saturating_add(els_per_band).min(n);
        if t0 < t1 {
            // SAFETY: band slices are in bounds; `partials[w]` is owned
            // by worker `w`; `broadcast` joins before returning, so the
            // borrows cannot outlive the call. AVX2 checked by caller.
            unsafe {
                let xb = std::slice::from_raw_parts(x_send.0.add(t0), t1 - t0);
                let yb = std::slice::from_raw_parts(y_send.0.add(t0), t1 - t0);
                *p_send.0.add(w) = avx2::dot_f64(xb, yb);
            }
        }
    });
    partials.iter().sum()
}

/// `dotc` over static spread bands (Complex32). Same contract as
/// [`dot_broadcast_f32`].
#[cfg(target_arch = "x86_64")]
#[inline]
fn dot_broadcast_c32(x: &[Complex32], y: &[Complex32], pool: &rayon::ThreadPool) -> Complex32 {
    let n = x.len();
    let width = pool.current_num_threads().max(1);
    let bands = width.min(n).max(1);
    let els_per_band = n.div_ceil(bands);
    let x_send = SendConstPtr(x.as_ptr());
    let y_send = SendConstPtr(y.as_ptr());
    let mut partials = vec![Complex32::new(0.0, 0.0); width];
    let p_send = SendMutPtr(partials.as_mut_ptr());
    pool.broadcast(|ctx| {
        let x_send: SendConstPtr<Complex32> = x_send;
        let y_send: SendConstPtr<Complex32> = y_send;
        let p_send: SendMutPtr<Complex32> = p_send;
        let w = ctx.index();
        let t0 = w.saturating_mul(els_per_band).min(n);
        let t1 = t0.saturating_add(els_per_band).min(n);
        if t0 < t1 {
            // SAFETY: band slices are in bounds; `partials[w]` is owned
            // by worker `w`; `broadcast` joins before returning, so the
            // borrows cannot outlive the call. AVX2 checked by caller.
            unsafe {
                let xb = std::slice::from_raw_parts(x_send.0.add(t0), t1 - t0);
                let yb = std::slice::from_raw_parts(y_send.0.add(t0), t1 - t0);
                *p_send.0.add(w) = avx2::dotc_c32(xb, yb);
            }
        }
    });
    partials.iter().sum()
}

/// `axpy` over static spread bands (f32): worker `w` owns `y[t0..t1]`.
/// Elementwise, so bitwise identical to the legacy stealing path.
#[cfg(target_arch = "x86_64")]
#[inline]
fn axpy_broadcast_f32(alpha: f32, x: &[f32], y: &mut [f32], pool: &rayon::ThreadPool) {
    let n = x.len();
    let width = pool.current_num_threads().max(1);
    let bands = width.min(n).max(1);
    let els_per_band = n.div_ceil(bands);
    let x_send = SendConstPtr(x.as_ptr());
    let y_send = SendMutPtr(y.as_mut_ptr());
    pool.broadcast(|ctx| {
        let x_send: SendConstPtr<f32> = x_send;
        let y_send: SendMutPtr<f32> = y_send;
        let w = ctx.index();
        let t0 = w.saturating_mul(els_per_band).min(n);
        let t1 = t0.saturating_add(els_per_band).min(n);
        if t0 < t1 {
            // SAFETY: band slices are in bounds and disjoint across
            // workers; `broadcast` joins before returning. AVX2 checked
            // by caller.
            unsafe {
                let xb = std::slice::from_raw_parts(x_send.0.add(t0), t1 - t0);
                let yb = std::slice::from_raw_parts_mut(y_send.0.add(t0), t1 - t0);
                avx2::axpy_f32(alpha, xb, yb);
            }
        }
    });
}

/// `axpy` over static spread bands (f64). Same contract as
/// [`axpy_broadcast_f32`].
#[cfg(target_arch = "x86_64")]
#[inline]
fn axpy_broadcast_f64(alpha: f64, x: &[f64], y: &mut [f64], pool: &rayon::ThreadPool) {
    let n = x.len();
    let width = pool.current_num_threads().max(1);
    let bands = width.min(n).max(1);
    let els_per_band = n.div_ceil(bands);
    let x_send = SendConstPtr(x.as_ptr());
    let y_send = SendMutPtr(y.as_mut_ptr());
    pool.broadcast(|ctx| {
        let x_send: SendConstPtr<f64> = x_send;
        let y_send: SendMutPtr<f64> = y_send;
        let w = ctx.index();
        let t0 = w.saturating_mul(els_per_band).min(n);
        let t1 = t0.saturating_add(els_per_band).min(n);
        if t0 < t1 {
            // SAFETY: band slices are in bounds and disjoint across
            // workers; `broadcast` joins before returning. AVX2 checked
            // by caller.
            unsafe {
                let xb = std::slice::from_raw_parts(x_send.0.add(t0), t1 - t0);
                let yb = std::slice::from_raw_parts_mut(y_send.0.add(t0), t1 - t0);
                avx2::axpy_f64(alpha, xb, yb);
            }
        }
    });
}

/// `axpy` over static spread bands (Complex32). Same contract as
/// [`axpy_broadcast_f32`].
#[cfg(target_arch = "x86_64")]
#[inline]
fn axpy_broadcast_c32(
    alpha: Complex32,
    x: &[Complex32],
    y: &mut [Complex32],
    pool: &rayon::ThreadPool,
) {
    let n = x.len();
    let width = pool.current_num_threads().max(1);
    let bands = width.min(n).max(1);
    let els_per_band = n.div_ceil(bands);
    let x_send = SendConstPtr(x.as_ptr());
    let y_send = SendMutPtr(y.as_mut_ptr());
    pool.broadcast(|ctx| {
        let x_send: SendConstPtr<Complex32> = x_send;
        let y_send: SendMutPtr<Complex32> = y_send;
        let w = ctx.index();
        let t0 = w.saturating_mul(els_per_band).min(n);
        let t1 = t0.saturating_add(els_per_band).min(n);
        if t0 < t1 {
            // SAFETY: band slices are in bounds and disjoint across
            // workers; `broadcast` joins before returning. AVX2 checked
            // by caller.
            unsafe {
                let xb = std::slice::from_raw_parts(x_send.0.add(t0), t1 - t0);
                let yb = std::slice::from_raw_parts_mut(y_send.0.add(t0), t1 - t0);
                avx2::axpy_c32(alpha, xb, yb);
            }
        }
    });
}

/// `axpy` over static spread bands (Complex64). Same contract as
/// [`axpy_broadcast_f32`].
#[cfg(target_arch = "x86_64")]
#[inline]
fn axpy_broadcast_c64(
    alpha: Complex64,
    x: &[Complex64],
    y: &mut [Complex64],
    pool: &rayon::ThreadPool,
) {
    let n = x.len();
    let width = pool.current_num_threads().max(1);
    let bands = width.min(n).max(1);
    let els_per_band = n.div_ceil(bands);
    let x_send = SendConstPtr(x.as_ptr());
    let y_send = SendMutPtr(y.as_mut_ptr());
    pool.broadcast(|ctx| {
        let x_send: SendConstPtr<Complex64> = x_send;
        let y_send: SendMutPtr<Complex64> = y_send;
        let w = ctx.index();
        let t0 = w.saturating_mul(els_per_band).min(n);
        let t1 = t0.saturating_add(els_per_band).min(n);
        if t0 < t1 {
            // SAFETY: band slices are in bounds and disjoint across
            // workers; `broadcast` joins before returning. AVX2 checked
            // by caller.
            unsafe {
                let xb = std::slice::from_raw_parts(x_send.0.add(t0), t1 - t0);
                let yb = std::slice::from_raw_parts_mut(y_send.0.add(t0), t1 - t0);
                avx2::axpy_c64(alpha, xb, yb);
            }
        }
    });
}

/// `dotc` over static spread bands (Complex64). Same contract as
/// [`dot_broadcast_f32`].
#[cfg(target_arch = "x86_64")]
#[inline]
fn dot_broadcast_c64(x: &[Complex64], y: &[Complex64], pool: &rayon::ThreadPool) -> Complex64 {
    let n = x.len();
    let width = pool.current_num_threads().max(1);
    let bands = width.min(n).max(1);
    let els_per_band = n.div_ceil(bands);
    let x_send = SendConstPtr(x.as_ptr());
    let y_send = SendConstPtr(y.as_ptr());
    let mut partials = vec![Complex64::new(0.0, 0.0); width];
    let p_send = SendMutPtr(partials.as_mut_ptr());
    pool.broadcast(|ctx| {
        let x_send: SendConstPtr<Complex64> = x_send;
        let y_send: SendConstPtr<Complex64> = y_send;
        let p_send: SendMutPtr<Complex64> = p_send;
        let w = ctx.index();
        let t0 = w.saturating_mul(els_per_band).min(n);
        let t1 = t0.saturating_add(els_per_band).min(n);
        if t0 < t1 {
            // SAFETY: band slices are in bounds; `partials[w]` is owned
            // by worker `w`; `broadcast` joins before returning, so the
            // borrows cannot outlive the call. AVX2 checked by caller.
            unsafe {
                let xb = std::slice::from_raw_parts(x_send.0.add(t0), t1 - t0);
                let yb = std::slice::from_raw_parts(y_send.0.add(t0), t1 - t0);
                *p_send.0.add(w) = avx2::dotc_c64(xb, yb);
            }
        }
    });
    partials.iter().sum()
}

/// `scal` over static spread bands (f32): worker `w` owns `x[t0..t1]`.
/// Elementwise, so bitwise identical to the legacy stealing path.
#[cfg(target_arch = "x86_64")]
#[inline]
fn scal_broadcast_f32(alpha: f32, x: &mut [f32], pool: &rayon::ThreadPool) {
    let n = x.len();
    let width = pool.current_num_threads().max(1);
    let bands = width.min(n).max(1);
    let els_per_band = n.div_ceil(bands);
    let x_send = SendMutPtr(x.as_mut_ptr());
    pool.broadcast(|ctx| {
        let x_send: SendMutPtr<f32> = x_send;
        let w = ctx.index();
        let t0 = w.saturating_mul(els_per_band).min(n);
        let t1 = t0.saturating_add(els_per_band).min(n);
        if t0 < t1 {
            // SAFETY: band slices are in bounds and disjoint across
            // workers; `broadcast` joins before returning. AVX2 checked
            // by caller.
            unsafe {
                let xb = std::slice::from_raw_parts_mut(x_send.0.add(t0), t1 - t0);
                avx2::scal_f32(alpha, xb);
            }
        }
    });
}

/// `scal` over static spread bands (f64). Same contract as
/// [`scal_broadcast_f32`].
#[cfg(target_arch = "x86_64")]
#[inline]
fn scal_broadcast_f64(alpha: f64, x: &mut [f64], pool: &rayon::ThreadPool) {
    let n = x.len();
    let width = pool.current_num_threads().max(1);
    let bands = width.min(n).max(1);
    let els_per_band = n.div_ceil(bands);
    let x_send = SendMutPtr(x.as_mut_ptr());
    pool.broadcast(|ctx| {
        let x_send: SendMutPtr<f64> = x_send;
        let w = ctx.index();
        let t0 = w.saturating_mul(els_per_band).min(n);
        let t1 = t0.saturating_add(els_per_band).min(n);
        if t0 < t1 {
            // SAFETY: band slices are in bounds and disjoint across
            // workers; `broadcast` joins before returning. AVX2 checked
            // by caller.
            unsafe {
                let xb = std::slice::from_raw_parts_mut(x_send.0.add(t0), t1 - t0);
                avx2::scal_f64(alpha, xb);
            }
        }
    });
}

/// `scal` over static spread bands (Complex32). Same contract as
/// [`scal_broadcast_f32`].
#[cfg(target_arch = "x86_64")]
#[inline]
fn scal_broadcast_c32(alpha: Complex32, x: &mut [Complex32], pool: &rayon::ThreadPool) {
    let n = x.len();
    let width = pool.current_num_threads().max(1);
    let bands = width.min(n).max(1);
    let els_per_band = n.div_ceil(bands);
    let x_send = SendMutPtr(x.as_mut_ptr());
    pool.broadcast(|ctx| {
        let x_send: SendMutPtr<Complex32> = x_send;
        let w = ctx.index();
        let t0 = w.saturating_mul(els_per_band).min(n);
        let t1 = t0.saturating_add(els_per_band).min(n);
        if t0 < t1 {
            // SAFETY: band slices are in bounds and disjoint across
            // workers; `broadcast` joins before returning. AVX2 checked
            // by caller.
            unsafe {
                let xb = std::slice::from_raw_parts_mut(x_send.0.add(t0), t1 - t0);
                avx2::scal_c32(alpha, xb);
            }
        }
    });
}

/// `scal` over static spread bands (Complex64). Same contract as
/// [`scal_broadcast_f32`].
#[cfg(target_arch = "x86_64")]
#[inline]
fn scal_broadcast_c64(alpha: Complex64, x: &mut [Complex64], pool: &rayon::ThreadPool) {
    let n = x.len();
    let width = pool.current_num_threads().max(1);
    let bands = width.min(n).max(1);
    let els_per_band = n.div_ceil(bands);
    let x_send = SendMutPtr(x.as_mut_ptr());
    pool.broadcast(|ctx| {
        let x_send: SendMutPtr<Complex64> = x_send;
        let w = ctx.index();
        let t0 = w.saturating_mul(els_per_band).min(n);
        let t1 = t0.saturating_add(els_per_band).min(n);
        if t0 < t1 {
            // SAFETY: band slices are in bounds and disjoint across
            // workers; `broadcast` joins before returning. AVX2 checked
            // by caller.
            unsafe {
                let xb = std::slice::from_raw_parts_mut(x_send.0.add(t0), t1 - t0);
                avx2::scal_c64(alpha, xb);
            }
        }
    });
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    if x.len() * 8 >= DOT_PAR_THRESHOLD_BYTES && avx2::is_available() {
        if let Some(spin) = spin_pool_for_blas1() {
            return spin_dot_f32(x, y, spin);
        }
        if let Some(pool) = spread_pool_for_blas1() {
            return dot_broadcast_f32(x, y, pool);
        }
        let chunk = dot_chunk_els(x.len());
        // SAFETY: chunks are disjoint; each runs the AVX2 kernel.
        return x
            .par_chunks(chunk)
            .zip(y.par_chunks(chunk))
            .map(|(a, b)| unsafe { avx2::dot_f32(a, b) })
            .sum();
    }
    if avx2::is_available() {
        unsafe { avx2::dot_f32(x, y) }
    } else {
        scalar::dot_f32(x, y)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn dot_f64(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    if x.len() * 16 >= DOT_PAR_THRESHOLD_BYTES && avx2::is_available() {
        if let Some(spin) = spin_pool_for_blas1() {
            return spin_dot_f64(x, y, spin);
        }
        if let Some(pool) = spread_pool_for_blas1() {
            return dot_broadcast_f64(x, y, pool);
        }
        let chunk = dot_chunk_els(x.len());
        // SAFETY: chunks are disjoint; each runs the AVX2 kernel.
        return x
            .par_chunks(chunk)
            .zip(y.par_chunks(chunk))
            .map(|(a, b)| unsafe { avx2::dot_f64(a, b) })
            .sum();
    }
    if avx2::is_available() {
        unsafe { avx2::dot_f64(x, y) }
    } else {
        scalar::dot_f64(x, y)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn dotc_c32(x: &[Complex32], y: &[Complex32]) -> Complex32 {
    debug_assert_eq!(x.len(), y.len());
    if x.len() * 16 >= DOT_PAR_THRESHOLD_BYTES && avx2::is_available() {
        if let Some(spin) = spin_pool_for_blas1() {
            return spin_dotc_c32(x, y, spin);
        }
        if let Some(pool) = spread_pool_for_blas1() {
            return dot_broadcast_c32(x, y, pool);
        }
        let chunk = dot_chunk_els(x.len());
        // SAFETY: chunks are disjoint; each runs the AVX2 kernel.
        return x
            .par_chunks(chunk)
            .zip(y.par_chunks(chunk))
            .map(|(a, b)| unsafe { avx2::dotc_c32(a, b) })
            .reduce(|| Complex32::new(0.0, 0.0), |a, b| a + b);
    }
    if avx2::is_available() {
        unsafe { avx2::dotc_c32(x, y) }
    } else {
        scalar::dotc_c32(x, y)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn dotc_c64(x: &[Complex64], y: &[Complex64]) -> Complex64 {
    debug_assert_eq!(x.len(), y.len());
    if x.len() * 32 >= DOT_PAR_THRESHOLD_BYTES && avx2::is_available() {
        if let Some(spin) = spin_pool_for_blas1() {
            return spin_dotc_c64(x, y, spin);
        }
        if let Some(pool) = spread_pool_for_blas1() {
            return dot_broadcast_c64(x, y, pool);
        }
        let chunk = dot_chunk_els(x.len());
        // SAFETY: chunks are disjoint; each runs the AVX2 kernel.
        return x
            .par_chunks(chunk)
            .zip(y.par_chunks(chunk))
            .map(|(a, b)| unsafe { avx2::dotc_c64(a, b) })
            .reduce(|| Complex64::new(0.0, 0.0), |a, b| a + b);
    }
    if avx2::is_available() {
        unsafe { avx2::dotc_c64(x, y) }
    } else {
        scalar::dotc_c64(x, y)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn nrm2_f32(x: &[f32]) -> f32 {
    if avx2::is_available() {
        unsafe { avx2::nrm2_f32(x) }
    } else {
        scalar::nrm2_f32(x)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn nrm2_f64(x: &[f64]) -> f64 {
    if avx2::is_available() {
        unsafe { avx2::nrm2_f64(x) }
    } else {
        scalar::nrm2_f64(x)
    }
}

/// Fused `dlassq` row accumulation (f64): fold one row of magnitudes
/// into per-column `(scale, sums)`, returning row finiteness. AVX2
/// 4-wide on x86_64 (bitwise identical to the scalar reference —
/// vectorizes across independent columns, keeping per-column op
/// order); scalar elsewhere.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn dlassq_row_update_f64(row: &[f64], scales: &mut [f64], sums: &mut [f64]) -> bool {
    debug_assert_eq!(row.len(), scales.len());
    debug_assert_eq!(row.len(), sums.len());
    if avx2::is_available() {
        unsafe { avx2::dlassq_row_update_f64(row, scales, sums) }
    } else {
        scalar::dlassq_row_update_f64(row, scales, sums)
    }
}

/// f32 twin of [`dlassq_row_update_f64`]: same contract, 8-wide.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn dlassq_row_update_f32(row: &[f32], scales: &mut [f32], sums: &mut [f32]) -> bool {
    debug_assert_eq!(row.len(), scales.len());
    debug_assert_eq!(row.len(), sums.len());
    if avx2::is_available() {
        unsafe { avx2::dlassq_row_update_f32(row, scales, sums) }
    } else {
        scalar::dlassq_row_update_f32(row, scales, sums)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn nrm2_c32(x: &[Complex32]) -> f32 {
    if avx2::is_available() {
        unsafe { avx2::nrm2_c32(x) }
    } else {
        scalar::nrm2_c32(x)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn nrm2_c64(x: &[Complex64]) -> f64 {
    if avx2::is_available() {
        unsafe { avx2::nrm2_c64(x) }
    } else {
        scalar::nrm2_c64(x)
    }
}

// ---------- aarch64 SME2-aware dispatch for real BLAS-1 ----------

#[cfg(target_arch = "aarch64")]
use crate::sme2::blas1 as sme2_blas1;

/// `y <- alpha * x + y` (f32). Picks SME2 streaming-SVE for long vectors,
/// NEON for short. Crossover at `sme2_blas1::min_len()` (~4096 elements,
/// overridable via `SONIUM_STREAMING_SVE_MIN_LEN`).
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]) {
    if sme2_blas1::streaming_sve_wins(x.len(), false) {
        sme2_blas1::axpy_f32(alpha, x, y);
    } else {
        neon::axpy_f32(alpha, x, y);
    }
}

/// `y <- alpha * x + y` (f64). f64 streaming-SVE requires `FEAT_SME_F64F64`;
/// `sme2_blas1::streaming_sve_wins` checks that cap and falls back to NEON
/// if it's missing.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn axpy_f64(alpha: f64, x: &[f64], y: &mut [f64]) {
    if sme2_blas1::streaming_sve_wins(x.len(), true) {
        sme2_blas1::axpy_f64(alpha, x, y);
    } else {
        neon::axpy_f64(alpha, x, y);
    }
}

/// Pivot scan `max |base[(first+i)*stride]|` over `len` rows (f32).
/// No NEON kernel yet: scalar fallback.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn pivot_scan_f32(base: &[f32], stride: usize, first: usize, len: usize) -> (usize, f32) {
    scalar::pivot_scan_f32(base, stride, first, len)
}

/// Fused `dlassq` row accumulation (f32). No NEON kernel yet: scalar
/// fallback (exact reference semantics).
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn dlassq_row_update_f32(row: &[f32], scales: &mut [f32], sums: &mut [f32]) -> bool {
    scalar::dlassq_row_update_f32(row, scales, sums)
}

/// Fused `dlassq` row accumulation (f64). No NEON kernel yet: scalar
/// fallback (exact reference semantics).
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn dlassq_row_update_f64(row: &[f64], scales: &mut [f64], sums: &mut [f64]) -> bool {
    scalar::dlassq_row_update_f64(row, scales, sums)
}

/// Pivot scan `max |base[(first+i)*stride]|` over `len` rows (f64).
/// No NEON kernel yet: scalar fallback.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn pivot_scan_f64(base: &[f64], stride: usize, first: usize, len: usize) -> (usize, f64) {
    scalar::pivot_scan_f64(base, stride, first, len)
}

/// First index of a non-finite lane (f32). No NEON kernel yet: scalar fallback.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn find_non_finite_f32(values: &[f32]) -> Option<usize> {
    scalar::find_non_finite_f32(values)
}

/// First index of a non-finite lane (f64). No NEON kernel yet: scalar fallback.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn find_non_finite_f64(values: &[f64]) -> Option<usize> {
    scalar::find_non_finite_f64(values)
}

/// Max `|v|` over f32 lanes. No NEON kernel yet: scalar fallback.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn max_norm_f32(values: &[f32]) -> f32 {
    scalar::max_norm_f32(values)
}

/// Max `|v|` over f64 lanes. No NEON kernel yet: scalar fallback.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn max_norm_f64(values: &[f64]) -> f64 {
    scalar::max_norm_f64(values)
}

/// Fused entry-validation fold, f32. No NEON kernel yet: scalar fallback.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn max_norm_checked_f32(values: &[f32]) -> (f32, bool) {
    scalar::max_norm_checked_f32(values)
}

/// Fused entry-validation fold, f64. No NEON kernel yet: scalar fallback.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn max_norm_checked_f64(values: &[f64]) -> (f64, bool) {
    scalar::max_norm_checked_f64(values)
}

/// Fused rank-1 update `block[i*bs+j] += alpha*mults[i*ms] * pivot[j]`
/// (f32). No NEON kernel yet: scalar fallback (still a single dispatch per
/// panel column instead of one axpy per row).
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn ger_update_f32(
    alpha: f32,
    mults: &[f32],
    mult_stride: usize,
    pivot: &[f32],
    block: &mut [f32],
    block_stride: usize,
    rows: usize,
    cols: usize,
) {
    scalar::ger_update_f32(
        alpha,
        mults,
        mult_stride,
        pivot,
        block,
        block_stride,
        rows,
        cols,
    )
}

/// Fused rank-1 update `block[i*bs+j] += alpha*mults[i*ms] * pivot[j]`
/// (f64). No NEON kernel yet: scalar fallback (still a single dispatch per
/// panel column instead of one axpy per row).
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn ger_update_f64(
    alpha: f64,
    mults: &[f64],
    mult_stride: usize,
    pivot: &[f64],
    block: &mut [f64],
    block_stride: usize,
    rows: usize,
    cols: usize,
) {
    scalar::ger_update_f64(
        alpha,
        mults,
        mult_stride,
        pivot,
        block,
        block_stride,
        rows,
        cols,
    )
}

/// Complex64 pivot scan (true modulus, first wins). No NEON kernel yet:
/// scalar fallback.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn pivot_scan_c64(base: &[Complex64], stride: usize, first: usize, len: usize) -> (usize, f64) {
    scalar::pivot_scan_c64(base, stride, first, len)
}

/// Complex32 twin of [`pivot_scan_c64`]. No NEON kernel yet: scalar fallback.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn pivot_scan_c32(base: &[Complex32], stride: usize, first: usize, len: usize) -> (usize, f32) {
    scalar::pivot_scan_c32(base, stride, first, len)
}

/// Fused complex rank-1 update (Complex64). No NEON kernel yet: scalar
/// fallback (still a single dispatch per panel column).
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn ger_update_c64(
    alpha: Complex64,
    mults: &[Complex64],
    mult_stride: usize,
    pivot: &[Complex64],
    block: &mut [Complex64],
    block_stride: usize,
    rows: usize,
    cols: usize,
) {
    scalar::ger_update_c64(
        alpha,
        mults,
        mult_stride,
        pivot,
        block,
        block_stride,
        rows,
        cols,
    )
}

/// Complex64 entry-validation fold. No NEON kernel yet: scalar fallback.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn max_norm_checked_c64(values: &[Complex64]) -> (f64, bool) {
    scalar::max_norm_checked_c64(values)
}

/// Complex32 twin of [`max_norm_checked_c64`]. No NEON kernel yet: scalar fallback.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn max_norm_checked_c32(values: &[Complex32]) -> (f32, bool) {
    scalar::max_norm_checked_c32(values)
}

/// 4M epilogue row (Complex64). No NEON kernel yet: scalar fallback.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn apply_ab_c64_row(
    r: &[f64],
    imag: &[f64],
    c: &mut [Complex64],
    alpha: Complex64,
    beta: Complex64,
    beta_zero: bool,
) {
    scalar::apply_ab_c64_row(r, imag, c, alpha, beta, beta_zero)
}

/// 4M epilogue row (Complex32). No NEON kernel yet: scalar fallback.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn apply_ab_c32_row(
    r: &[f32],
    imag: &[f32],
    c: &mut [Complex32],
    alpha: Complex32,
    beta: Complex32,
    beta_zero: bool,
) {
    scalar::apply_ab_c32_row(r, imag, c, alpha, beta, beta_zero)
}

/// 3M epilogue row (Complex64). No NEON kernel yet: scalar fallback.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn apply_ab3_c64_row(
    r: &[f64],
    q: &[f64],
    imag: &[f64],
    c: &mut [Complex64],
    alpha: Complex64,
    beta: Complex64,
    beta_zero: bool,
) {
    scalar::apply_ab3_c64_row(r, q, imag, c, alpha, beta, beta_zero)
}

/// 3M epilogue row (Complex32). No NEON kernel yet: scalar fallback.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn apply_ab3_c32_row(
    r: &[f32],
    q: &[f32],
    imag: &[f32],
    c: &mut [Complex32],
    alpha: Complex32,
    beta: Complex32,
    beta_zero: bool,
) {
    scalar::apply_ab3_c32_row(r, q, imag, c, alpha, beta, beta_zero)
}

/// Complex32 twin of [`ger_update_c64`]. No NEON kernel yet: scalar fallback.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn ger_update_c32(
    alpha: Complex32,
    mults: &[Complex32],
    mult_stride: usize,
    pivot: &[Complex32],
    block: &mut [Complex32],
    block_stride: usize,
    rows: usize,
    cols: usize,
) {
    scalar::ger_update_c32(
        alpha,
        mults,
        mult_stride,
        pivot,
        block,
        block_stride,
        rows,
        cols,
    )
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
    if sme2_blas1::streaming_sve_wins(x.len(), false) {
        sme2_blas1::dot_f32(x, y)
    } else {
        neon::dot_f32(x, y)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn dot_f64(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "dot_f64 vectors must have equal length");
    if sme2_blas1::streaming_sve_wins(x.len(), true) {
        sme2_blas1::dot_f64(x, y)
    } else {
        neon::dot_f64(x, y)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn scal_f32(alpha: f32, x: &mut [f32]) {
    if sme2_blas1::streaming_sve_wins(x.len(), false) {
        sme2_blas1::scal_f32(alpha, x);
    } else {
        neon::scal_f32(alpha, x);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn scal_f64(alpha: f64, x: &mut [f64]) {
    if sme2_blas1::streaming_sve_wins(x.len(), true) {
        sme2_blas1::scal_f64(alpha, x);
    } else {
        neon::scal_f64(alpha, x);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn nrm2_f32(x: &[f32]) -> f32 {
    if sme2_blas1::streaming_sve_wins(x.len(), false) {
        sme2_blas1::nrm2_f32(x)
    } else {
        neon::nrm2_f32(x)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn nrm2_f64(x: &[f64]) -> f64 {
    if sme2_blas1::streaming_sve_wins(x.len(), true) {
        sme2_blas1::nrm2_f64(x)
    } else {
        neon::nrm2_f64(x)
    }
}

// ---------- axpy_many: y[i] += Σ_bb alphas[bb] * x_basis[bb][i] ----------

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn axpy_many_f32(alphas: &[f32], x_basis: &[&[f32]], y: &mut [f32]) {
    neon::axpy_many_f32(alphas, x_basis, y);
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn axpy_many_f64(alphas: &[f64], x_basis: &[&[f64]], y: &mut [f64]) {
    neon::axpy_many_f64(alphas, x_basis, y);
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn axpy_many_c32(alphas: &[Complex32], x_basis: &[&[Complex32]], y: &mut [Complex32]) {
    neon::axpy_many_c32(alphas, x_basis, y);
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn axpy_many_c64(alphas: &[Complex64], x_basis: &[&[Complex64]], y: &mut [Complex64]) {
    neon::axpy_many_c64(alphas, x_basis, y);
}

// x86_64 AVX2 dispatch for axpy_many

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn axpy_many_f32(alphas: &[f32], x_basis: &[&[f32]], y: &mut [f32]) {
    if avx2::is_available() {
        unsafe { avx2::axpy_many_f32(alphas, x_basis, y) }
    } else {
        scalar::axpy_many_f32(alphas, x_basis, y)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn axpy_many_f64(alphas: &[f64], x_basis: &[&[f64]], y: &mut [f64]) {
    if avx2::is_available() {
        unsafe { avx2::axpy_many_f64(alphas, x_basis, y) }
    } else {
        scalar::axpy_many_f64(alphas, x_basis, y)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn axpy_many_c32(alphas: &[Complex32], x_basis: &[&[Complex32]], y: &mut [Complex32]) {
    if avx2::is_available() {
        unsafe { avx2::axpy_many_c32(alphas, x_basis, y) }
    } else {
        scalar::axpy_many_c32(alphas, x_basis, y)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn axpy_many_c64(alphas: &[Complex64], x_basis: &[&[Complex64]], y: &mut [Complex64]) {
    if avx2::is_available() {
        unsafe { avx2::axpy_many_c64(alphas, x_basis, y) }
    } else {
        scalar::axpy_many_c64(alphas, x_basis, y)
    }
}

// ---------- correctness tests ----------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num_complex::{Complex32, Complex64};

    fn ramp_f32(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * 0.0123).sin()).collect()
    }

    fn ramp_f64(n: usize) -> Vec<f64> {
        (0..n).map(|i| (i as f64 * 0.0123).sin()).collect()
    }

    fn ramp_c32(n: usize) -> Vec<Complex32> {
        (0..n)
            .map(|i| Complex32::new((i as f32 * 0.0123).sin(), (i as f32 * 0.0177).cos()))
            .collect()
    }

    fn ramp_c64(n: usize) -> Vec<Complex64> {
        (0..n)
            .map(|i| Complex64::new((i as f64 * 0.0123).sin(), (i as f64 * 0.0177).cos()))
            .collect()
    }

    fn cases() -> [usize; 7] {
        [1, 3, 7, 16, 17, 1024, 4099]
    }

    fn dot4_cases() -> [usize; 8] {
        [0, 1, 3, 7, 8, 9, 33, 130]
    }

    #[test]
    fn dot4_f32_matches_scalar() {
        for &n in &dot4_cases() {
            let l = ramp_f32(n.max(1));
            let (x0, x1, x2, x3) = (
                ramp_f32(n.max(1)),
                ramp_f32(n.max(1))
                    .iter()
                    .map(|v| v * 1.7)
                    .collect::<Vec<_>>(),
                ramp_f32(n.max(1))
                    .iter()
                    .map(|v| v - 0.3)
                    .collect::<Vec<_>>(),
                ramp_f32(n.max(1))
                    .iter()
                    .map(|v| v * 0.13)
                    .collect::<Vec<_>>(),
            );
            let (l, x0, x1, x2, x3) = (&l[..n], &x0[..n], &x1[..n], &x2[..n], &x3[..n]);
            let got = dot4_f32(l, [x0, x1, x2, x3]);
            let want = [
                scalar::dot_f32(l, x0),
                scalar::dot_f32(l, x1),
                scalar::dot_f32(l, x2),
                scalar::dot_f32(l, x3),
            ];
            for (a, b) in got.iter().zip(want.iter()) {
                assert_relative_eq!(a, b, max_relative = 1e-5, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn dot4_f64_matches_scalar() {
        for &n in &dot4_cases() {
            let l = ramp_f64(n.max(1));
            let (x0, x1, x2, x3) = (
                ramp_f64(n.max(1)),
                ramp_f64(n.max(1))
                    .iter()
                    .map(|v| v * 1.7)
                    .collect::<Vec<_>>(),
                ramp_f64(n.max(1))
                    .iter()
                    .map(|v| v - 0.3)
                    .collect::<Vec<_>>(),
                ramp_f64(n.max(1))
                    .iter()
                    .map(|v| v * 0.13)
                    .collect::<Vec<_>>(),
            );
            let (l, x0, x1, x2, x3) = (&l[..n], &x0[..n], &x1[..n], &x2[..n], &x3[..n]);
            let got = dot4_f64(l, [x0, x1, x2, x3]);
            let want = [
                scalar::dot_f64(l, x0),
                scalar::dot_f64(l, x1),
                scalar::dot_f64(l, x2),
                scalar::dot_f64(l, x3),
            ];
            for (a, b) in got.iter().zip(want.iter()) {
                assert_relative_eq!(a, b, max_relative = 1e-12, epsilon = 1e-14);
            }
        }
    }

    #[test]
    fn axpy_f32_matches_scalar() {
        for &n in &cases() {
            let x = ramp_f32(n);
            let mut y_ours: Vec<f32> = ramp_f32(n).iter().map(|v| v + 0.5).collect();
            let mut y_ref = y_ours.clone();
            axpy_f32(0.7, &x, &mut y_ours);
            scalar::axpy_f32(0.7, &x, &mut y_ref);
            for (a, b) in y_ours.iter().zip(y_ref.iter()) {
                assert_relative_eq!(a, b, max_relative = 1e-5, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn axpy_f64_matches_scalar() {
        for &n in &cases() {
            let x = ramp_f64(n);
            let mut y_ours: Vec<f64> = ramp_f64(n).iter().map(|v| v + 0.5).collect();
            let mut y_ref = y_ours.clone();
            axpy_f64(0.7, &x, &mut y_ours);
            scalar::axpy_f64(0.7, &x, &mut y_ref);
            for (a, b) in y_ours.iter().zip(y_ref.iter()) {
                assert_relative_eq!(a, b, max_relative = 1e-12);
            }
        }
    }

    /// Above-threshold `axpy` (rayon fan-out on x86_64) matches scalar
    /// within tolerance and is bitwise deterministic across calls
    /// (elementwise chunks, no reduction order).
    #[test]
    fn axpy_par_threshold_matches_scalar() {
        let n = 1_500_000; // 18 MB f32 / 36 MB f64 traffic: above the 16 MB gate
        let x32 = ramp_f32(n);
        let y32_init: Vec<f32> = ramp_f32(n).iter().map(|v| v + 0.5).collect();
        let mut y32 = y32_init.clone();
        let mut y32_ref = y32_init.clone();
        axpy_f32(0.7, &x32, &mut y32);
        scalar::axpy_f32(0.7, &x32, &mut y32_ref);
        for (a, b) in y32.iter().zip(y32_ref.iter()) {
            assert_relative_eq!(a, b, max_relative = 1e-5, epsilon = 1e-6);
        }
        let mut y32_again = y32_init.clone();
        axpy_f32(0.7, &x32, &mut y32_again);
        assert_eq!(y32, y32_again);

        let x64 = ramp_f64(n);
        let y64_init: Vec<f64> = ramp_f64(n).iter().map(|v| v + 0.5).collect();
        let mut y64 = y64_init.clone();
        let mut y64_ref = y64_init.clone();
        axpy_f64(0.7, &x64, &mut y64);
        scalar::axpy_f64(0.7, &x64, &mut y64_ref);
        for (a, b) in y64.iter().zip(y64_ref.iter()) {
            assert_relative_eq!(a, b, max_relative = 1e-12);
        }
        let mut y64_again = y64_init.clone();
        axpy_f64(0.7, &x64, &mut y64_again);
        assert_eq!(y64, y64_again);
        // Broadcast and legacy arms are bitwise identical (elementwise,
        // no reduction order): set the shared spread switch explicitly
        // per arm. Robust to parallel tests flipping the env — any
        // interleaving still compares identical elementwise results.
        unsafe { std::env::set_var("SONIUM_GEMV_PINNED", "1") };
        let mut y32_b = y32_init.clone();
        axpy_f32(0.7, &x32, &mut y32_b);
        unsafe { std::env::set_var("SONIUM_GEMV_PINNED", "0") };
        let mut y32_l = y32_init.clone();
        axpy_f32(0.7, &x32, &mut y32_l);
        unsafe { std::env::remove_var("SONIUM_GEMV_PINNED") };
        assert_eq!(y32_b, y32_l);
    }

    /// Above-threshold `dot` matches scalar within tolerance on BOTH the
    /// spread-broadcast arm (`SONIUM_GEMV_PINNED=1`, x86_64 default) and
    /// the legacy stealing arm (`=0`). Each arm is asserted against
    /// scalar (never arm-vs-arm: partial-sum order differs), so parallel
    /// tests flipping the process-global env cannot fail it.
    #[test]
    fn dot_par_arms_match_scalar() {
        let n = 1_000_000; // 8 MB f32 / 16 MB f64 traffic: above the 4 MB gate
        let x32 = ramp_f32(n);
        let y32: Vec<f32> = ramp_f32(n).iter().map(|v| v + 0.25).collect();
        let x64 = ramp_f64(n);
        let y64: Vec<f64> = ramp_f64(n).iter().map(|v| v + 0.25).collect();
        let ref32 = scalar::dot_f32(&x32, &y32);
        let ref64 = scalar::dot_f64(&x64, &y64);
        for pinned in ["1", "0"] {
            unsafe { std::env::set_var("SONIUM_GEMV_PINNED", pinned) };
            assert_relative_eq!(dot_f32(&x32, &y32), ref32, max_relative = 1e-5);
            assert_relative_eq!(dot_f64(&x64, &y64), ref64, max_relative = 1e-12);
        }
        unsafe { std::env::remove_var("SONIUM_GEMV_PINNED") };
    }

    /// Above-threshold `scal` (MT fan-out on x86_64) matches scalar
    /// within tolerance, is bitwise deterministic, and agrees bitwise
    /// across the broadcast/legacy arms (elementwise, no reduction).
    #[test]
    fn scal_par_threshold_matches_scalar() {
        let n = 5_000_000; // 40 MB f32 / 80 MB f64 traffic: above the 32 MB gate
        let x32_init: Vec<f32> = ramp_f32(n).iter().map(|v| v + 0.5).collect();
        let mut x32 = x32_init.clone();
        let mut x32_ref = x32_init.clone();
        scal_f32(0.7, &mut x32);
        scalar::scal_f32(0.7, &mut x32_ref);
        for (a, b) in x32.iter().zip(x32_ref.iter()) {
            assert_relative_eq!(a, b, max_relative = 1e-5, epsilon = 1e-6);
        }
        let mut x32_again = x32_init.clone();
        scal_f32(0.7, &mut x32_again);
        assert_eq!(x32, x32_again);

        let x64_init: Vec<f64> = ramp_f64(n).iter().map(|v| v + 0.5).collect();
        let mut x64 = x64_init.clone();
        let mut x64_ref = x64_init.clone();
        scal_f64(0.7, &mut x64);
        scalar::scal_f64(0.7, &mut x64_ref);
        for (a, b) in x64.iter().zip(x64_ref.iter()) {
            assert_relative_eq!(a, b, max_relative = 1e-12);
        }
        unsafe { std::env::set_var("SONIUM_GEMV_PINNED", "1") };
        let mut x64_b = x64_init.clone();
        scal_f64(0.7, &mut x64_b);
        unsafe { std::env::set_var("SONIUM_GEMV_PINNED", "0") };
        let mut x64_l = x64_init.clone();
        scal_f64(0.7, &mut x64_l);
        unsafe { std::env::remove_var("SONIUM_GEMV_PINNED") };
        assert_eq!(x64_b, x64_l);
    }

    fn ger_reference_f32(mults: &[f32], ms: usize, pivot: &[f32], block: &mut [f32], bs: usize) {
        let rows = mults.len() / ms;
        let cols = pivot.len();
        for i in 0..rows {
            axpy_f32(-mults[i * ms], pivot, &mut block[i * bs..i * bs + cols]);
        }
    }

    fn ger_reference_f64(mults: &[f64], ms: usize, pivot: &[f64], block: &mut [f64], bs: usize) {
        let rows = mults.len() / ms;
        let cols = pivot.len();
        for i in 0..rows {
            axpy_f64(-mults[i * ms], pivot, &mut block[i * bs..i * bs + cols]);
        }
    }

    #[test]
    fn ger_update_f32_matches_axpy_loop() {
        // Strided mults (panel column) and padded block stride, plus odd
        // widths that hit the vector tail. (100, 5) is the LU tall-skinny
        // panel geometry the four-row unroll targets.
        for &(rows, cols) in &[(1, 1), (3, 7), (16, 15), (33, 31), (100, 5)] {
            let bs = cols + 2;
            let ms = 3;
            let mults: Vec<f32> = (0..rows * ms).map(|i| 0.01 * (i as f32 + 1.0)).collect();
            let pivot: Vec<f32> = (0..cols).map(|j| 0.03 * (j as f32) - 0.2).collect();
            let mut ours: Vec<f32> = (0..rows * bs)
                .map(|i| 0.05 * ((i % 11) as f32) - 0.1)
                .collect();
            let mut reference = ours.clone();
            ger_update_f32(-1.0, &mults, ms, &pivot, &mut ours, bs, rows, cols);
            ger_reference_f32(&mults, ms, &pivot, &mut reference, bs);
            for (a, b) in ours.iter().zip(reference.iter()) {
                assert_relative_eq!(a, b, max_relative = 1e-5, epsilon = 1e-6);
            }
        }
    }

    fn pivot_reference_f32(base: &[f32], stride: usize, first: usize, len: usize) -> (usize, f32) {
        let mut row = first;
        let mut val = base[first * stride].abs();
        for i in 1..len {
            let v = base[(first + i) * stride].abs();
            if v > val {
                val = v;
                row = first + i;
            }
        }
        (row, val)
    }

    fn pivot_reference_f64(base: &[f64], stride: usize, first: usize, len: usize) -> (usize, f64) {
        let mut row = first;
        let mut val = base[first * stride].abs();
        for i in 1..len {
            let v = base[(first + i) * stride].abs();
            if v > val {
                val = v;
                row = first + i;
            }
        }
        (row, val)
    }

    #[test]
    fn pivot_scan_matches_scalar_bitwise() {
        // Strided columns, odd lengths hitting vector tails, ties (first
        // must win), negative values, and non-finite corners.
        let cases_f64: Vec<(Vec<f64>, usize, usize, usize)> = vec![
            (vec![1.0], 1, 0, 1),
            (vec![-3.0, 1.0, 2.0], 1, 0, 3),
            // Tie between scanned rows 1 and 3 (|9| == |-9|): row 1 wins.
            (vec![0.5, 0.0, 9.0, 0.0, 1.0, 0.0, -9.0, 0.0], 2, 1, 3),
            // Negative maximum magnitude.
            ((0..40).map(|i| -0.5 * (i as f64)).collect(), 1, 3, 33),
            // Stride with padding, max in tail.
            (
                (0..64).map(|i| ((i * 7) % 13) as f64 - 6.0).collect(),
                3,
                2,
                17,
            ),
            // NaN initial pivot keeps row 0.
            (vec![f64::NAN, 1.0, 2.0, 3.0], 1, 0, 4),
            // +Inf initial pivot keeps row 0.
            (vec![f64::INFINITY, 1.0, 2.0], 1, 0, 3),
            // NaN later value never wins.
            (vec![1.0, f64::NAN, 0.5, 0.25], 1, 0, 4),
            // -Inf initial: finite values still win.
            (vec![f64::NEG_INFINITY, -1.0, -2.0], 1, 0, 3),
            // All equal: first row wins.
            (vec![2.0; 20], 1, 5, 11),
            // Cross-block tie: max at rows 7 (lane 3) and 8 (lane 0).
            // Lane order alone would keep row 8; the row tie-break
            // must return 7.
            (
                vec![1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 5.0, 5.0, 1.0, 1.0, 1.0],
                1,
                0,
                12,
            ),
        ];
        for (base, stride, first, len) in &cases_f64 {
            let (r0, v0) = pivot_scan_f64(base, *stride, *first, *len);
            let (r1, v1) = pivot_reference_f64(base, *stride, *first, *len);
            assert_eq!(
                r0, r1,
                "row mismatch stride={stride} first={first} len={len}"
            );
            assert!(
                v0 == v1 || (v0.is_nan() && v1.is_nan()),
                "value mismatch stride={stride} first={first} len={len}: {v0} vs {v1}"
            );
        }
        let cases_f32: Vec<(Vec<f32>, usize, usize, usize)> = vec![
            (vec![1.0], 1, 0, 1),
            (vec![0.5, 0.0, 9.0, 0.0, 1.0, 0.0, -9.0, 0.0], 2, 1, 3),
            (
                (0..64).map(|i| ((i * 7) % 13) as f32 - 6.0).collect(),
                3,
                2,
                17,
            ),
            (vec![f32::NAN, 1.0, 2.0, 3.0], 1, 0, 4),
            (vec![f32::INFINITY, 1.0, 2.0], 1, 0, 3),
            (vec![1.0, f32::NAN, 0.5, 0.25], 1, 0, 4),
            (vec![f32::NEG_INFINITY, -1.0, -2.0], 1, 0, 3),
            (vec![2.0; 24], 1, 5, 15),
            // Cross-block tie: max at rows 7 (lane 7) and 8 (lane 0).
            (
                vec![
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                ],
                1,
                0,
                16,
            ),
        ];
        for (base, stride, first, len) in &cases_f32 {
            let (r0, v0) = pivot_scan_f32(base, *stride, *first, *len);
            let (r1, v1) = pivot_reference_f32(base, *stride, *first, *len);
            assert_eq!(
                r0, r1,
                "row mismatch stride={stride} first={first} len={len}"
            );
            assert!(
                v0 == v1 || (v0.is_nan() && v1.is_nan()),
                "value mismatch stride={stride} first={first} len={len}: {v0} vs {v1}"
            );
        }
    }

    #[test]
    fn max_norm_checked_matches_split_passes() {
        // Finite ramps across block/tail sizes: same max as `max_norm`,
        // flag holds. Non-finite corners (each position class: block,
        // tail, first, last): flag drops; the max is unspecified there.
        for &n in &[0usize, 1, 3, 7, 8, 9, 16, 17, 100, 1024, 4099] {
            let v64 = ramp_f64(n.max(1))[..n].to_vec();
            let (m64, ok64) = max_norm_checked_f64(&v64);
            assert!(ok64, "finite ramp flagged bad, n={n}");
            assert_eq!(m64, max_norm_f64(&v64), "max mismatch f64, n={n}");
            let v32: Vec<f32> = v64.iter().map(|&x| x as f32).collect();
            let (m32, ok32) = max_norm_checked_f32(&v32);
            assert!(ok32, "finite ramp flagged bad, n={n}");
            assert_eq!(m32, max_norm_f32(&v32), "max mismatch f32, n={n}");
        }
        // Empty input: zero max, finite.
        assert_eq!(max_norm_checked_f64(&[]), (0.0, true));
        assert_eq!(max_norm_checked_f32(&[]), (0.0, true));
        // Non-finite at block, tail, first, and last positions.
        for &n in &[9usize, 17, 100, 1025] {
            for &bad in &[f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                for &pos in &[0usize, 1, 7, 8, n - 2, n - 1] {
                    let mut v = ramp_f64(n);
                    v[pos] = bad;
                    let (_, ok) = max_norm_checked_f64(&v);
                    assert!(!ok, "missed {bad} at {pos}/{n}");
                    // The rescan contract the LU entry path relies on:
                    // `find_non_finite` locates exactly this index.
                    assert_eq!(find_non_finite_f64(&v), Some(pos));
                    let mut w2 = ramp_f32(n);
                    w2[pos] = bad as f32;
                    let (_, ok2) = max_norm_checked_f32(&w2);
                    assert!(!ok2, "missed f32 {bad} at {pos}/{n}");
                    assert_eq!(find_non_finite_f32(&w2), Some(pos));
                }
            }
        }
        // -0.0 and subnormals are finite; max sees through the sign.
        let edge = vec![-0.0f64, -3.0, 2.0, f64::MIN_POSITIVE / 2.0];
        assert_eq!(max_norm_checked_f64(&edge), (3.0, true));
        // Complementary exponents whose OR forges phantom all-ones
        // (0x409 | 0x3FE == 0x7FF): all finite, flag must hold. This is
        // the LU entry-matrix shape (large diagonal, small fractions)
        // that caught the raw-field OR bug.
        let phantom64 = vec![1152.0f64, 0.5, -0.25, 2048.0, 0.1, -1152.0, 3.0, 0.02, 7.0];
        assert_eq!(max_norm_checked_f64(&phantom64), (2048.0, true));
        let phantom32 = vec![
            1152.0f32, 0.5, -0.25, 2048.0, 0.1, -1152.0, 3.0, 0.02, 7.0, 1.0,
        ];
        assert_eq!(max_norm_checked_f32(&phantom32), (2048.0, true));
    }

    #[test]
    fn ger_update_f64_matches_axpy_loop() {
        for &(rows, cols) in &[(1, 1), (3, 7), (16, 15), (33, 31)] {
            let bs = cols + 2;
            let ms = 3;
            let mults: Vec<f64> = (0..rows * ms).map(|i| 0.01 * (i as f64 + 1.0)).collect();
            let pivot: Vec<f64> = (0..cols).map(|j| 0.03 * (j as f64) - 0.2).collect();
            let mut ours: Vec<f64> = (0..rows * bs)
                .map(|i| 0.05 * ((i % 11) as f64) - 0.1)
                .collect();
            let mut reference = ours.clone();
            ger_update_f64(-1.0, &mults, ms, &pivot, &mut ours, bs, rows, cols);
            ger_reference_f64(&mults, ms, &pivot, &mut reference, bs);
            for (a, b) in ours.iter().zip(reference.iter()) {
                assert_relative_eq!(a, b, max_relative = 1e-12);
            }
        }
    }

    #[test]
    fn axpy_c32_matches_scalar() {
        let alpha = Complex32::new(0.3, -0.7);
        for &n in &cases() {
            let x = ramp_c32(n);
            let mut y_ours: Vec<Complex32> = ramp_c32(n);
            let mut y_ref = y_ours.clone();
            axpy_c32(alpha, &x, &mut y_ours);
            scalar::axpy_c32(alpha, &x, &mut y_ref);
            for (a, b) in y_ours.iter().zip(y_ref.iter()) {
                assert_relative_eq!(a.re, b.re, max_relative = 1e-5, epsilon = 1e-6);
                assert_relative_eq!(a.im, b.im, max_relative = 1e-5, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn axpy_c64_matches_scalar() {
        let alpha = Complex64::new(0.3, -0.7);
        for &n in &cases() {
            let x = ramp_c64(n);
            let mut y_ours: Vec<Complex64> = ramp_c64(n);
            let mut y_ref = y_ours.clone();
            axpy_c64(alpha, &x, &mut y_ours);
            scalar::axpy_c64(alpha, &x, &mut y_ref);
            for (a, b) in y_ours.iter().zip(y_ref.iter()) {
                assert_relative_eq!(a.re, b.re, max_relative = 1e-12);
                assert_relative_eq!(a.im, b.im, max_relative = 1e-12);
            }
        }
    }

    #[test]
    fn scal_complex_matches_scalar() {
        for &n in &cases() {
            let alpha = Complex32::new(0.4, -1.1);
            let mut a_ours = ramp_c32(n);
            let mut a_ref = a_ours.clone();
            scal_c32(alpha, &mut a_ours);
            scalar::scal_c32(alpha, &mut a_ref);
            for (a, b) in a_ours.iter().zip(a_ref.iter()) {
                assert_relative_eq!(a.re, b.re, max_relative = 1e-5, epsilon = 1e-6);
                assert_relative_eq!(a.im, b.im, max_relative = 1e-5, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn dot_real_matches_scalar() {
        for &n in &cases() {
            let x = ramp_f64(n);
            let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.041).cos()).collect();
            let a = dot_f64(&x, &y);
            let b = scalar::dot_f64(&x, &y);
            assert_relative_eq!(a, b, max_relative = 1e-12, epsilon = 1e-12);
        }
    }

    #[test]
    fn dotc_complex_hermitian_matches_scalar() {
        for &n in &cases() {
            let x = ramp_c32(n);
            let y = ramp_c32(n);
            let a = dotc_c32(&x, &y);
            let b = scalar::dotc_c32(&x, &y);
            assert_relative_eq!(a.re, b.re, max_relative = 1e-4, epsilon = 1e-5);
            assert_relative_eq!(a.im, b.im, max_relative = 1e-4, epsilon = 1e-5);
        }
    }

    #[test]
    fn nrm2_matches_scalar() {
        for &n in &cases() {
            let x = ramp_f64(n);
            assert_relative_eq!(nrm2_f64(&x), scalar::nrm2_f64(&x), max_relative = 1e-12);

            let xc = ramp_c64(n);
            assert_relative_eq!(nrm2_c64(&xc), scalar::nrm2_c64(&xc), max_relative = 1e-12);
        }
    }

    #[test]
    fn axpy_many_c32_matches_scalar() {
        let alphas = [Complex32::new(0.5, -0.25), Complex32::new(-1.0, 0.5)];
        for &n in &cases() {
            let x0 = ramp_c32(n);
            let x1 = ramp_c32(n);
            let mut y_neon = ramp_c32(n);
            let mut y_scalar = y_neon.clone();
            let basis_neon = vec![&x0[..], &x1[..]];
            let basis_scalar = vec![&x0[..], &x1[..]];
            axpy_many_c32(&alphas, &basis_neon, &mut y_neon);
            scalar::axpy_many_c32(&alphas, &basis_scalar, &mut y_scalar);
            for (a, b) in y_neon.iter().zip(y_scalar.iter()) {
                assert_relative_eq!(a.re, b.re, max_relative = 1e-5, epsilon = 1e-6);
                assert_relative_eq!(a.im, b.im, max_relative = 1e-5, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn axpy_many_c64_matches_scalar() {
        let alphas = [Complex64::new(0.5, -0.25), Complex64::new(-1.0, 0.5)];
        for &n in &cases() {
            let x0 = ramp_c64(n);
            let x1 = ramp_c64(n);
            let mut y_neon = ramp_c64(n);
            let mut y_scalar = y_neon.clone();
            let basis_neon = vec![&x0[..], &x1[..]];
            let basis_scalar = vec![&x0[..], &x1[..]];
            axpy_many_c64(&alphas, &basis_neon, &mut y_neon);
            scalar::axpy_many_c64(&alphas, &basis_scalar, &mut y_scalar);
            for (a, b) in y_neon.iter().zip(y_scalar.iter()) {
                assert_relative_eq!(a.re, b.re, max_relative = 1e-12);
                assert_relative_eq!(a.im, b.im, max_relative = 1e-12);
            }
        }
    }

    /// Complex pivot scan: dispatched kernel reproduces the scalar
    /// reference's index exactly (first maximum wins, ties included) with
    /// the value matching to `hypot`-vs-`sqrt`-of-squares tolerance.
    #[test]
    fn pivot_scan_c64_matches_scalar() {
        for &(len, stride, first) in &[
            (1usize, 7usize, 0usize),
            (3, 5, 1),
            (4, 3, 0),
            (100, 11, 3),
            (1000, 37, 17),
        ] {
            let mut v = vec![Complex64::new(0.0, 0.0); (first + len) * stride + stride];
            for i in 0..len {
                // Scaled so the planted modulus-5 tie below is the unique max.
                v[(first + i) * stride] =
                    Complex64::new((i as f64 * 0.0123).sin() * 3.5, (i as f64 * 0.0177).cos());
            }
            if len > 10 {
                // Exact tie: (5,0) and (3,4) both have modulus 5; first wins.
                v[(first + 3) * stride] = Complex64::new(5.0, 0.0);
                v[(first + 7) * stride] = Complex64::new(3.0, 4.0);
            }
            let (rs, vs) = scalar::pivot_scan_c64(&v, stride, first, len);
            let (ra, va) = pivot_scan_c64(&v, stride, first, len);
            assert_eq!(rs, ra, "len={len} stride={stride} first={first}");
            assert_relative_eq!(vs, va, max_relative = 1e-12);
            if len > 10 {
                assert_eq!(ra, first + 3);
            }
        }
        // Non-finite edges: initial NaN keeps its row, later +Inf wins.
        let mut v = vec![Complex64::new(1.0, 1.0); 64];
        v[0] = Complex64::new(f64::NAN, 0.0);
        assert_eq!(pivot_scan_c64(&v, 1, 0, 64).0, 0);
        v[0] = Complex64::new(1.0, 1.0);
        v[9] = Complex64::new(1.0, f64::INFINITY);
        assert_eq!(pivot_scan_c64(&v, 1, 0, 64).0, 9);
    }

    /// Complex32 twin of [`pivot_scan_c64_matches_scalar`].
    #[test]
    fn pivot_scan_c32_matches_scalar() {
        for &(len, stride, first) in &[
            (1usize, 7usize, 0usize),
            (7, 5, 1),
            (8, 3, 0),
            (100, 11, 3),
            (1000, 37, 17),
        ] {
            let mut v = vec![Complex32::new(0.0, 0.0); (first + len) * stride + stride];
            for i in 0..len {
                // Scaled so the planted modulus-5 tie below is the unique max.
                v[(first + i) * stride] = Complex32::new(
                    (i as f32 * 0.0123).sin() * 3.0,
                    (i as f32 * 0.0177).cos() * 3.0,
                );
            }
            if len > 10 {
                v[(first + 2) * stride] = Complex32::new(5.0, 0.0);
                v[(first + 6) * stride] = Complex32::new(3.0, 4.0);
            }
            let (rs, vs) = scalar::pivot_scan_c32(&v, stride, first, len);
            let (ra, va) = pivot_scan_c32(&v, stride, first, len);
            assert_eq!(rs, ra, "len={len} stride={stride} first={first}");
            assert_relative_eq!(vs, va, max_relative = 1e-5);
            if len > 10 {
                assert_eq!(ra, first + 2);
            }
        }
    }

    /// Complex GER: dispatched kernel matches the scalar axpy-per-row
    /// reference to rounding tolerance (documented non-bitwise: FMA-style
    /// accumulation vs scalar order).
    #[test]
    fn ger_update_c64_matches_scalar() {
        for &(rows, cols, ms, bs) in &[
            (1usize, 1usize, 1usize, 1usize),
            (3, 5, 2, 7),
            (100, 31, 1, 40),
            (999, 17, 3, 19),
        ] {
            for alpha in [Complex64::new(-1.0, 0.0), Complex64::new(0.7, -0.3)] {
                let mults: Vec<Complex64> = (0..rows * ms + 1)
                    .map(|i| Complex64::new((i as f64 * 0.013).sin(), (i as f64 * 0.007).cos()))
                    .collect();
                let pivot: Vec<Complex64> = (0..cols)
                    .map(|i| Complex64::new((i as f64 * 0.019).cos(), (i as f64 * 0.011).sin()))
                    .collect();
                let mut b1: Vec<Complex64> = (0..rows * bs)
                    .map(|i| Complex64::new((i as f64 * 0.023).sin(), 0.0))
                    .collect();
                let mut b2 = b1.clone();
                scalar::ger_update_c64(alpha, &mults, ms, &pivot, &mut b1, bs, rows, cols);
                ger_update_c64(alpha, &mults, ms, &pivot, &mut b2, bs, rows, cols);
                let scale = b1.iter().map(|x| x.norm()).fold(0.0, f64::max).max(1.0);
                for (a, b) in b1.iter().zip(b2.iter()) {
                    assert_relative_eq!(a.re, b.re, max_relative = 1e-12, epsilon = 1e-12 * scale);
                    assert_relative_eq!(a.im, b.im, max_relative = 1e-12, epsilon = 1e-12 * scale);
                }
            }
        }
    }

    /// Complex32 twin of [`ger_update_c64_matches_scalar`].
    #[test]
    fn ger_update_c32_matches_scalar() {
        for &(rows, cols) in &[(3usize, 5usize), (100, 31), (999, 17)] {
            let mults: Vec<Complex32> = (0..rows)
                .map(|i| Complex32::new((i as f32 * 0.013).sin(), (i as f32 * 0.007).cos()))
                .collect();
            let pivot: Vec<Complex32> = (0..cols)
                .map(|i| Complex32::new((i as f32 * 0.019).cos(), (i as f32 * 0.011).sin()))
                .collect();
            let mut b1: Vec<Complex32> = (0..rows * cols)
                .map(|i| Complex32::new((i as f32 * 0.023).sin(), 0.0))
                .collect();
            let mut b2 = b1.clone();
            scalar::ger_update_c32(
                Complex32::new(-1.0, 0.0),
                &mults,
                1,
                &pivot,
                &mut b1,
                cols,
                rows,
                cols,
            );
            ger_update_c32(
                Complex32::new(-1.0, 0.0),
                &mults,
                1,
                &pivot,
                &mut b2,
                cols,
                rows,
                cols,
            );
            let scale = b1.iter().map(|x| x.norm()).fold(0.0, f32::max).max(1.0);
            for (a, b) in b1.iter().zip(b2.iter()) {
                assert_relative_eq!(a.re, b.re, max_relative = 1e-5, epsilon = 1e-5 * scale);
                assert_relative_eq!(a.im, b.im, max_relative = 1e-5, epsilon = 1e-5 * scale);
            }
        }
    }

    /// Complex entry-validation fold: flag bit-exact (finite and
    /// NaN/Inf-bearing inputs), max to `hypot`-vs-`sqrt`-of-squares
    /// tolerance.
    #[test]
    fn max_norm_checked_c64_matches_scalar() {
        for &n in &[0usize, 1, 3, 7, 16, 1000, 4099] {
            let v = ramp_c64(n.max(1));
            let (bs, oks) = scalar::max_norm_checked_c64(&v[..n]);
            let (ba, oka) = max_norm_checked_c64(&v[..n]);
            assert_eq!(oks, oka, "flag n={n}");
            assert_eq!(oks, true);
            assert_relative_eq!(bs, ba, max_relative = 1e-12);
        }
        let mut v = ramp_c64(100);
        v[37] = Complex64::new(f64::NAN, 1.0);
        assert_eq!(max_norm_checked_c64(&v).1, false);
        v[37] = Complex64::new(1.0, f64::INFINITY);
        assert_eq!(max_norm_checked_c64(&v).1, false);
        v[37] = Complex64::new(1.0, 1.0);
        assert_eq!(max_norm_checked_c64(&v).1, true);
    }

    /// 4M epilogue row: dispatched kernel is bitwise-identical to the
    /// scalar reference across α/β combinations (Schur accumulate,
    /// overwrite, general) and tail widths.
    #[test]
    fn apply_ab_c64_row_matches_scalar() {
        for &n in &[0usize, 1, 3, 4, 5, 37, 1000] {
            let r: Vec<f64> = (0..n).map(|i| (i as f64 * 0.013).sin()).collect();
            let imag: Vec<f64> = (0..n).map(|i| (i as f64 * 0.017).cos()).collect();
            let c0: Vec<Complex64> = (0..n)
                .map(|i| Complex64::new((i as f64 * 0.019).sin(), (i as f64 * 0.023).cos()))
                .collect();
            for (alpha, beta) in [
                (Complex64::new(-1.0, 0.0), Complex64::new(1.0, 0.0)),
                (Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)),
                (Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)),
                (Complex64::new(0.3, -0.2), Complex64::new(0.5, 0.1)),
            ] {
                let beta_zero = beta.re == 0.0 && beta.im == 0.0;
                let mut c1 = c0.clone();
                let mut c2 = c0.clone();
                scalar::apply_ab_c64_row(&r, &imag, &mut c1, alpha, beta, beta_zero);
                apply_ab_c64_row(&r, &imag, &mut c2, alpha, beta, beta_zero);
                for (a, b) in c1.iter().zip(c2.iter()) {
                    assert_eq!(
                        a.re.to_bits(),
                        b.re.to_bits(),
                        "re n={n} a={alpha:?} b={beta:?}"
                    );
                    assert_eq!(
                        a.im.to_bits(),
                        b.im.to_bits(),
                        "im n={n} a={alpha:?} b={beta:?}"
                    );
                }
            }
        }
    }

    /// Complex32 twin of [`apply_ab_c64_row_matches_scalar`].
    #[test]
    fn apply_ab_c32_row_matches_scalar() {
        for &n in &[0usize, 1, 7, 8, 9, 37, 1000] {
            let r: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin()).collect();
            let imag: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017).cos()).collect();
            let c0: Vec<Complex32> = (0..n)
                .map(|i| Complex32::new((i as f32 * 0.019).sin(), (i as f32 * 0.023).cos()))
                .collect();
            for (alpha, beta) in [
                (Complex32::new(-1.0, 0.0), Complex32::new(1.0, 0.0)),
                (Complex32::new(1.0, 0.0), Complex32::new(0.0, 0.0)),
                (Complex32::new(1.0, 0.0), Complex32::new(1.0, 0.0)),
                (Complex32::new(0.3, -0.2), Complex32::new(0.5, 0.1)),
            ] {
                let beta_zero = beta.re == 0.0 && beta.im == 0.0;
                let mut c1 = c0.clone();
                let mut c2 = c0.clone();
                scalar::apply_ab_c32_row(&r, &imag, &mut c1, alpha, beta, beta_zero);
                apply_ab_c32_row(&r, &imag, &mut c2, alpha, beta, beta_zero);
                for (a, b) in c1.iter().zip(c2.iter()) {
                    assert_eq!(
                        a.re.to_bits(),
                        b.re.to_bits(),
                        "re n={n} a={alpha:?} b={beta:?}"
                    );
                    assert_eq!(
                        a.im.to_bits(),
                        b.im.to_bits(),
                        "im n={n} a={alpha:?} b={beta:?}"
                    );
                }
            }
        }
    }

    /// Complex32 twin of [`max_norm_checked_c64_matches_scalar`].
    #[test]
    fn max_norm_checked_c32_matches_scalar() {
        for &n in &[0usize, 1, 3, 15, 16, 1000, 4099] {
            let v = ramp_c32(n.max(1));
            let (bs, oks) = scalar::max_norm_checked_c32(&v[..n]);
            let (ba, oka) = max_norm_checked_c32(&v[..n]);
            assert_eq!(oks, oka, "flag n={n}");
            assert_eq!(oks, true);
            assert_relative_eq!(bs, ba, max_relative = 1e-5);
        }
        let mut v = ramp_c32(100);
        v[37] = Complex32::new(f32::NAN, 1.0);
        assert_eq!(max_norm_checked_c32(&v).1, false);
    }

    /// Fused dlassq row update: dispatched kernel is bitwise-identical to
    /// the scalar reference across widths (vector/tail alignment), prior
    /// state shapes, and adversarial magnitudes (±0, subnormals, ±inf,
    /// NaN, overflow-scale ratios).
    #[test]
    fn dlassq_row_update_f64_matches_scalar_bitwise() {
        let pool = [
            0.0,
            -0.0,
            1.0,
            -2.5,
            0.5,
            1e-300,
            1e300,
            f64::MAX,
            f64::MIN_POSITIVE,
            5e-324,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        for &n in &[0usize, 1, 3, 4, 5, 7, 8, 9, 16, 17, 33, 1000] {
            for (case, &(sc0, su0)) in [(0.0, 1.0), (1.0, 1.0), (1e200, 2.5), (1e-200, 0.0)]
                .iter()
                .enumerate()
            {
                let row: Vec<f64> = (0..n).map(|i| pool[(i + case * 5) % pool.len()]).collect();
                let mut sc1 = vec![sc0; n];
                let mut su1 = vec![su0; n];
                let mut sc2 = sc1.clone();
                let mut su2 = su1.clone();
                let ok1 = scalar::dlassq_row_update_f64(&row, &mut sc1, &mut su1);
                let ok2 = dlassq_row_update_f64(&row, &mut sc2, &mut su2);
                assert_eq!(ok1, ok2, "flag n={n} case={case}");
                for c in 0..n {
                    assert_eq!(
                        sc1[c].to_bits(),
                        sc2[c].to_bits(),
                        "scale n={n} case={case} c={c}"
                    );
                    assert_eq!(
                        su1[c].to_bits(),
                        su2[c].to_bits(),
                        "sums n={n} case={case} c={c}"
                    );
                }
            }
        }
    }

    /// f32 twin of [`dlassq_row_update_f64_matches_scalar_bitwise`].
    #[test]
    fn dlassq_row_update_f32_matches_scalar_bitwise() {
        let pool = [
            0.0,
            -0.0,
            1.0,
            -2.5,
            0.5,
            1e-30,
            1e30,
            f32::MAX,
            f32::MIN_POSITIVE,
            1e-45,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        for &n in &[0usize, 1, 3, 7, 8, 9, 15, 16, 17, 33, 1000] {
            for (case, &(sc0, su0)) in [(0.0, 1.0), (1.0, 1.0), (1e19, 2.5), (1e-19, 0.0)]
                .iter()
                .enumerate()
            {
                let row: Vec<f32> = (0..n).map(|i| pool[(i + case * 5) % pool.len()]).collect();
                let mut sc1 = vec![sc0; n];
                let mut su1 = vec![su0; n];
                let mut sc2 = sc1.clone();
                let mut su2 = su1.clone();
                let ok1 = scalar::dlassq_row_update_f32(&row, &mut sc1, &mut su1);
                let ok2 = dlassq_row_update_f32(&row, &mut sc2, &mut su2);
                assert_eq!(ok1, ok2, "flag n={n} case={case}");
                for c in 0..n {
                    assert_eq!(
                        sc1[c].to_bits(),
                        sc2[c].to_bits(),
                        "scale n={n} case={case} c={c}"
                    );
                    assert_eq!(
                        su1[c].to_bits(),
                        su2[c].to_bits(),
                        "sums n={n} case={case} c={c}"
                    );
                }
            }
        }
    }

    /// x86_64: the raw AVX2 kernels (bypassing dispatch) reproduce the
    /// scalar reference bitwise — guards the masking contract directly.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn dlassq_row_update_avx2_matches_scalar_bitwise() {
        if !avx2::is_available() {
            return;
        }
        let pool64 = [
            0.0,
            -0.0,
            1.0,
            1e-300,
            1e300,
            f64::MAX,
            5e-324,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        for &n in &[1usize, 3, 4, 5, 8, 12, 17, 100] {
            let row: Vec<f64> = (0..n).map(|i| pool64[i % pool64.len()]).collect();
            for &sc0 in &[0.0, 1.0, 1e200] {
                let mut sc1 = vec![sc0; n];
                let mut su1 = vec![1.0; n];
                let mut sc2 = sc1.clone();
                let mut su2 = su1.clone();
                let ok1 = scalar::dlassq_row_update_f64(&row, &mut sc1, &mut su1);
                let ok2 = unsafe { avx2::dlassq_row_update_f64(&row, &mut sc2, &mut su2) };
                assert_eq!(ok1, ok2, "f64 flag n={n} sc0={sc0}");
                for c in 0..n {
                    assert_eq!(sc1[c].to_bits(), sc2[c].to_bits(), "f64 scale n={n} c={c}");
                    assert_eq!(su1[c].to_bits(), su2[c].to_bits(), "f64 sums n={n} c={c}");
                }
            }
        }
        let pool32 = [
            0.0,
            -0.0,
            1.0,
            1e-30,
            1e30,
            f32::MAX,
            1e-45,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        for &n in &[1usize, 3, 7, 8, 9, 16, 24, 100] {
            let row: Vec<f32> = (0..n).map(|i| pool32[i % pool32.len()]).collect();
            for &sc0 in &[0.0, 1.0, 1e19] {
                let mut sc1 = vec![sc0; n];
                let mut su1 = vec![1.0; n];
                let mut sc2 = sc1.clone();
                let mut su2 = su1.clone();
                let ok1 = scalar::dlassq_row_update_f32(&row, &mut sc1, &mut su1);
                let ok2 = unsafe { avx2::dlassq_row_update_f32(&row, &mut sc2, &mut su2) };
                assert_eq!(ok1, ok2, "f32 flag n={n} sc0={sc0}");
                for c in 0..n {
                    assert_eq!(sc1[c].to_bits(), sc2[c].to_bits(), "f32 scale n={n} c={c}");
                    assert_eq!(su1[c].to_bits(), su2[c].to_bits(), "f32 sums n={n} c={c}");
                }
            }
        }
    }
}
