//! In-house BLAS-3 kernels.
//!
//! `gemm` for `f32`, `f64`, `Complex32`, `Complex64`, row-major, with the
//! BLAS-style `(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C,
//! ldc)` signature.
//!
//! Real-valued (`f32`, `f64`) calls route through the SME2 micro-kernels in
//! [`crate::sme2`]. Transposed B is first packed into a row-major `k x n`
//! scratch panel so the micro-kernel still sees a `NoTrans` right operand.
//! Complex calls use the 3M decomposition in [`complex3m`], packing any
//! transposed/conjugated operands into pooled row-major `NoTrans` panels.
//!
//! On macOS, eligible `(NoTrans, NoTrans)` calls at ≥ 32³ FLOPs route to
//! Accelerate `cblas_?gemm` instead: real kernels above the documented
//! thresholds, and complex `cgemm`/`zgemm` for non-skewed output shapes
//! (skewed `m:n` ratios > 4 stay on 4M-SME2 — measured faster there).

#![allow(clippy::too_many_arguments)]

pub mod complex3m;
pub mod complex_native;
pub mod complex_native32;
pub mod complex_think;
pub mod scalar;
pub mod syrk;
pub mod trsm;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(target_arch = "aarch64")]
pub mod neon;

use crate::backend::Transpose;
use crate::sme2;
use num_complex::{Complex32, Complex64};
use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicU8, Ordering};

/// Whether the SME2 GEMM micro-kernels are usable on this host.
/// On Apple Silicon without SME2 (M1/M2/M3) the in-house real GEMM
/// fallback is a scalar triple loop (~2 GF/s measured on M1 vs
/// ~200 GF/s for Accelerate), so macOS dispatch routes around it
/// through Accelerate instead (see `accelerate::*_t`). On SME2 hosts
/// (M4+) this returns true and every historical routing is unchanged.
#[allow(dead_code)]
pub(crate) fn sme2_available() -> bool {
    crate::sme2::detect::Sme2Caps::detect().has_sme2
}

#[cfg(target_os = "macos")]
pub(crate) mod accelerate {
    use crate::backend::Transpose;
    use num_complex::{Complex32, Complex64};

    const CBLAS_ROW_MAJOR: i32 = 101;
    const CBLAS_NO_TRANS: i32 = 111;
    const CBLAS_TRANS: i32 = 112;
    const CBLAS_CONJ_TRANS: i32 = 113;

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn cblas_sgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
        fn cblas_dgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f64,
            a: *const f64,
            lda: i32,
            b: *const f64,
            ldb: i32,
            beta: f64,
            c: *mut f64,
            ldc: i32,
        );
        // Complex scalars and matrices are passed as pointers to interleaved
        // `[re, im]` pairs — layout-compatible with `num_complex::Complex*`
        // (repr(C), re then im).
        fn cblas_cgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: *const f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: *const f32,
            c: *mut f32,
            ldc: i32,
        );
        fn cblas_zgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: *const f64,
            a: *const f64,
            lda: i32,
            b: *const f64,
            ldb: i32,
            beta: *const f64,
            c: *mut f64,
            ldc: i32,
        );
        fn cblas_sgemv(
            order: i32,
            trans: i32,
            m: i32,
            n: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            x: *const f32,
            incx: i32,
            beta: f32,
            y: *mut f32,
            incy: i32,
        );
        fn cblas_dgemv(
            order: i32,
            trans: i32,
            m: i32,
            n: i32,
            alpha: f64,
            a: *const f64,
            lda: i32,
            x: *const f64,
            incx: i32,
            beta: f64,
            y: *mut f64,
            incy: i32,
        );
    }

    #[inline]
    fn matrix_fits(len: usize, rows: usize, cols: usize, ld: usize) -> bool {
        rows == 0
            || cols == 0
            || (cols <= ld && (rows - 1).saturating_mul(ld).saturating_add(cols) <= len)
    }

    #[inline]
    fn eligible(
        m: usize,
        n: usize,
        k: usize,
        a_len: usize,
        lda: usize,
        b_len: usize,
        ldb: usize,
        c_len: usize,
        ldc: usize,
    ) -> bool {
        let enough_work = m.saturating_mul(n).saturating_mul(k) >= 32 * 32 * 32;
        enough_work
            && (m | n | k | lda | ldb | ldc) <= i32::MAX as usize
            && matrix_fits(a_len, m, k, lda)
            && matrix_fits(b_len, k, n, ldb)
            && matrix_fits(c_len, m, n, ldc)
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn sgemm(
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &[f32],
        lda: usize,
        b: &[f32],
        ldb: usize,
        beta: f32,
        c: &mut [f32],
        ldc: usize,
    ) -> bool {
        if !eligible(m, n, k, a.len(), lda, b.len(), ldb, c.len(), ldc) {
            return false;
        }
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                m as i32,
                n as i32,
                k as i32,
                alpha,
                a.as_ptr(),
                lda as i32,
                b.as_ptr(),
                ldb as i32,
                beta,
                c.as_mut_ptr(),
                ldc as i32,
            );
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn dgemm(
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a: &[f64],
        lda: usize,
        b: &[f64],
        ldb: usize,
        beta: f64,
        c: &mut [f64],
        ldc: usize,
    ) -> bool {
        let min_output_dim = m.min(n);
        if min_output_dim == 0 || m.max(n) > min_output_dim.saturating_mul(4) {
            return false;
        }
        if !eligible(m, n, k, a.len(), lda, b.len(), ldb, c.len(), ldc) {
            return false;
        }
        unsafe {
            cblas_dgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                m as i32,
                n as i32,
                k as i32,
                alpha,
                a.as_ptr(),
                lda as i32,
                b.as_ptr(),
                ldb as i32,
                beta,
                c.as_mut_ptr(),
                ldc as i32,
            );
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn cgemm(
        m: usize,
        n: usize,
        k: usize,
        alpha: Complex32,
        a: &[Complex32],
        lda: usize,
        b: &[Complex32],
        ldb: usize,
        beta: Complex32,
        c: &mut [Complex32],
        ldc: usize,
    ) -> bool {
        // Measured on M4 Pro (blas3_compare gemm_c32, 2026-07): cgemm beats
        // the 4M-SME2 path 2.1x at 64^3, 2.4x at 256^3, 1.2x at 1024^3, but
        // LOSES 1.1x on the skewed 2048x128x2048 BEM shape — same guard as
        // dgemm keeps skewed outputs on the in-house kernel.
        let min_output_dim = m.min(n);
        if min_output_dim == 0 || m.max(n) > min_output_dim.saturating_mul(4) {
            return false;
        }
        if !eligible(m, n, k, a.len(), lda, b.len(), ldb, c.len(), ldc) {
            return false;
        }
        unsafe {
            cblas_cgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                m as i32,
                n as i32,
                k as i32,
                &alpha as *const Complex32 as *const f32,
                a.as_ptr() as *const f32,
                lda as i32,
                b.as_ptr() as *const f32,
                ldb as i32,
                &beta as *const Complex32 as *const f32,
                c.as_mut_ptr() as *mut f32,
                ldc as i32,
            );
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn zgemm(
        m: usize,
        n: usize,
        k: usize,
        alpha: Complex64,
        a: &[Complex64],
        lda: usize,
        b: &[Complex64],
        ldb: usize,
        beta: Complex64,
        c: &mut [Complex64],
        ldc: usize,
    ) -> bool {
        // Same skew guard as cgemm/dgemm — see the measured numbers there.
        let min_output_dim = m.min(n);
        if min_output_dim == 0 || m.max(n) > min_output_dim.saturating_mul(4) {
            return false;
        }
        if !eligible(m, n, k, a.len(), lda, b.len(), ldb, c.len(), ldc) {
            return false;
        }
        unsafe {
            cblas_zgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                m as i32,
                n as i32,
                k as i32,
                &alpha as *const Complex64 as *const f64,
                a.as_ptr() as *const f64,
                lda as i32,
                b.as_ptr() as *const f64,
                ldb as i32,
                &beta as *const Complex64 as *const f64,
                c.as_mut_ptr() as *mut f64,
                ldc as i32,
            );
        }
        true
    }

    // ---- Trans-aware variants for hosts without SME2 -----------------
    //
    // Only called when `sme2_available()` is false (M1/M2/M3): there the
    // in-house fallback behind every transposed or skewed shape is the
    // scalar triple loop (~1.5-2 GF/s on M1), while cblas trans flags
    // run at full AMX rate (130-160x faster at 512^3, 99x on the skewed
    // 2048x128x2048 BEM shape; measured 2026-10-15). The work floor is
    // far below the NN wrappers' 32^3: even a 64^3 transposed GEMM pays
    // ~130us scalar vs ~10us through cblas.

    fn cblas_trans(op: Transpose, complex: bool) -> i32 {
        match op {
            Transpose::NoTrans => CBLAS_NO_TRANS,
            Transpose::Trans => CBLAS_TRANS,
            Transpose::ConjTrans => {
                if complex {
                    CBLAS_CONJ_TRANS
                } else {
                    CBLAS_TRANS
                }
            }
        }
    }

    #[inline]
    fn stored_fits(
        transa: Transpose,
        transb: Transpose,
        m: usize,
        n: usize,
        k: usize,
        a_len: usize,
        lda: usize,
        b_len: usize,
        ldb: usize,
        c_len: usize,
        ldc: usize,
    ) -> bool {
        if m == 0 || n == 0 {
            return false;
        }
        let (a_rows, a_cols) = match transa {
            Transpose::NoTrans => (m, k),
            _ => (k, m),
        };
        let (b_rows, b_cols) = match transb {
            Transpose::NoTrans => (k, n),
            _ => (n, k),
        };
        (m | n | k | lda | ldb | ldc) <= i32::MAX as usize
            && matrix_fits(a_len, a_rows, a_cols, lda)
            && matrix_fits(b_len, b_rows, b_cols, ldb)
            && matrix_fits(c_len, m, n, ldc)
            && m.saturating_mul(n).saturating_mul(k) >= 8192
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sgemm_t(
        transa: Transpose,
        transb: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &[f32],
        lda: usize,
        b: &[f32],
        ldb: usize,
        beta: f32,
        c: &mut [f32],
        ldc: usize,
    ) -> bool {
        if !stored_fits(transa, transb, m, n, k, a.len(), lda, b.len(), ldb, c.len(), ldc) {
            return false;
        }
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                cblas_trans(transa, false),
                cblas_trans(transb, false),
                m as i32,
                n as i32,
                k as i32,
                alpha,
                a.as_ptr(),
                lda as i32,
                b.as_ptr(),
                ldb as i32,
                beta,
                c.as_mut_ptr(),
                ldc as i32,
            );
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dgemm_t(
        transa: Transpose,
        transb: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a: &[f64],
        lda: usize,
        b: &[f64],
        ldb: usize,
        beta: f64,
        c: &mut [f64],
        ldc: usize,
    ) -> bool {
        if !stored_fits(transa, transb, m, n, k, a.len(), lda, b.len(), ldb, c.len(), ldc) {
            return false;
        }
        unsafe {
            cblas_dgemm(
                CBLAS_ROW_MAJOR,
                cblas_trans(transa, false),
                cblas_trans(transb, false),
                m as i32,
                n as i32,
                k as i32,
                alpha,
                a.as_ptr(),
                lda as i32,
                b.as_ptr(),
                ldb as i32,
                beta,
                c.as_mut_ptr(),
                ldc as i32,
            );
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cgemm_t(
        transa: Transpose,
        transb: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: Complex32,
        a: &[Complex32],
        lda: usize,
        b: &[Complex32],
        ldb: usize,
        beta: Complex32,
        c: &mut [Complex32],
        ldc: usize,
    ) -> bool {
        if !stored_fits(transa, transb, m, n, k, a.len(), lda, b.len(), ldb, c.len(), ldc) {
            return false;
        }
        unsafe {
            cblas_cgemm(
                CBLAS_ROW_MAJOR,
                cblas_trans(transa, true),
                cblas_trans(transb, true),
                m as i32,
                n as i32,
                k as i32,
                &alpha as *const Complex32 as *const f32,
                a.as_ptr() as *const f32,
                lda as i32,
                b.as_ptr() as *const f32,
                ldb as i32,
                &beta as *const Complex32 as *const f32,
                c.as_mut_ptr() as *mut f32,
                ldc as i32,
            );
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub fn zgemm_t(
        transa: Transpose,
        transb: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: Complex64,
        a: &[Complex64],
        lda: usize,
        b: &[Complex64],
        ldb: usize,
        beta: Complex64,
        c: &mut [Complex64],
        ldc: usize,
    ) -> bool {
        if !stored_fits(transa, transb, m, n, k, a.len(), lda, b.len(), ldb, c.len(), ldc) {
            return false;
        }
        unsafe {
            cblas_zgemm(
                CBLAS_ROW_MAJOR,
                cblas_trans(transa, true),
                cblas_trans(transb, true),
                m as i32,
                n as i32,
                k as i32,
                &alpha as *const Complex64 as *const f64,
                a.as_ptr() as *const f64,
                lda as i32,
                b.as_ptr() as *const f64,
                ldb as i32,
                &beta as *const Complex64 as *const f64,
                c.as_mut_ptr() as *mut f64,
                ldc as i32,
            );
        }
        true
    }

    /// NoTrans GEMV for hosts without SME2 (see `blas2`): the SME2
    /// scalar fallback runs at ~9 GB/s on M1 vs ~116 GB/s for cblas at
    /// 1024x1024. Contiguous vectors only (incx = incy = 1).
    #[allow(clippy::too_many_arguments)]
    pub fn sgemv_n(
        m: usize,
        n: usize,
        alpha: f32,
        a: &[f32],
        lda: usize,
        x: &[f32],
        beta: f32,
        y: &mut [f32],
    ) -> bool {
        if m == 0 || n == 0 || (m | n | lda) > i32::MAX as usize {
            return false;
        }
        if !matrix_fits(a.len(), m, n, lda) || x.len() < n || y.len() < m {
            return false;
        }
        unsafe {
            cblas_sgemv(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                m as i32,
                n as i32,
                alpha,
                a.as_ptr(),
                lda as i32,
                x.as_ptr(),
                1,
                beta,
                y.as_mut_ptr(),
                1,
            );
        }
        true
    }

    /// f64 twin of [`sgemv_n`].
    #[allow(clippy::too_many_arguments)]
    pub fn dgemv_n(
        m: usize,
        n: usize,
        alpha: f64,
        a: &[f64],
        lda: usize,
        x: &[f64],
        beta: f64,
        y: &mut [f64],
    ) -> bool {
        if m == 0 || n == 0 || (m | n | lda) > i32::MAX as usize {
            return false;
        }
        if !matrix_fits(a.len(), m, n, lda) || x.len() < n || y.len() < m {
            return false;
        }
        unsafe {
            cblas_dgemv(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                m as i32,
                n as i32,
                alpha,
                a.as_ptr(),
                lda as i32,
                x.as_ptr(),
                1,
                beta,
                y.as_mut_ptr(),
                1,
            );
        }
        true
    }
}

thread_local! {
    pub(crate) static F32_B_PACK: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    pub(crate) static F64_B_PACK: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    pub(crate) static F32_A_PANELS: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    pub(crate) static F64_A_PANELS: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static C32_A_PACK: RefCell<Vec<Complex32>> = const { RefCell::new(Vec::new()) };
    static C64_A_PACK: RefCell<Vec<Complex64>> = const { RefCell::new(Vec::new()) };
}

/// Worker count of the small-work pool (see [`scope_for_macs`]): 48 is the
/// measured sweet spot — every worker's share stays above the dispatch
/// fixed costs on sub-512M-MAC scopes, while the 128-wide pool's
/// barrier/SMT coordination costs more than the extra cores return below
/// that size (48 beats 32 by ~17% on 268M-MAC passes, ties it on tiny
/// updates, and both double the 128-wide rate).
/// Worker count of the tiny-work pool (see [`scope_for_macs`]): 16 is the
/// measured sweet spot for sub-64M-MAC scopes — a 1024²×32 thin-`k` update
/// peaks at 111–120 GF/s on 12–24 workers and falls to ~83 GF/s on 48+,
/// while 201M-MAC scopes still scale to 48. Like the small pool, the tiny
/// pool only engages on wide machines (see [`workers_for_macs`]).
pub(crate) const TINY_THREADS: usize = 16;

/// Per-scope multiply-accumulate ceiling for the tiny-work pool: scopes
/// under this run on [`TINY_THREADS`] workers instead of the small-work or
/// global pool. Placed at 64M MACs from A/B probes: every tested shape at
/// or below 64M MACs wins big or ties on 16 workers (400³ 136 → 219
/// GF/s, 384³ 121 → 198, 1024²×32 83 → 116, 1024²×48 134 → 149,
/// 1024²×60 ties 154 → 159), while every tested shape at 67M+ prefers 48
/// (1024²×64 170 → 197, 512²×256 227 → 266, 1024²×96 175 → 250).
pub(crate) const TINY_MAX_MACS: usize = 64_000_000;

/// Per-scope multiply-accumulate ceiling for the hot-workers pool: scopes
/// under this run on the compact-topology pool (the wide L3-neighbour set,
/// up to 16 workers since the 2026-10-02 four-domain widening) instead of
/// the tiny/small pools. Rationale: short
/// scopes never heat the parked cores their wide-pool tasks land on
/// (schedutil parks idle cores near 2.2GHz vs 4.3GHz boosted, and the wall
/// is the slowest task), so concentrating on the already-hot L3-neighbour
/// cores beats spreading — 64x960x64 GEMM 0.16 -> 0.05ms, SYRK-960
/// 0.44 -> 0.26ms, TRSM-right-960x64 0.080 -> 0.052ms free-mask. Longer
/// scopes heat their cores mid-flight, so the width-tuned tiny/small tiers
/// keep everything above this line (see the placement note on
/// [`HOT_MAX_MACS`]).
/// `SONIUM_HOT_MAX_MACS` overrides (diagnostic/A-B, read per call like the
/// other gates).
///
/// Placed at 8M MACs: below this a scope's bands cannot fill the tiny
/// pool (16 workers) — e.g. 64x960x64 GEMM (3.9M, 4 bands) and SYRK at
/// n <= 362 — so concentrating wins big (0.19 -> 0.067ms, 0.17 -> 0.11ms
/// medians). At and above it the width-tuned tiers win once warm (256^3
/// GEMM regressed 0.13 -> 0.25ms on the 8-wide hot pool: 16 bands want
/// 16 workers, not 8), so they keep their routing. NOTE 2026-10-02:
/// the hot pool widened to 16 with the four-domain compact change, so
/// that parenthetical predates the current width — the 8M ceiling
/// itself is untouched (hot tier is inert at RAYON=16 anyway) and wants
/// a re-tune before any ceiling move.
pub(crate) const HOT_MAX_MACS: usize = 8_000_000;

/// Diagnostic tuning gate: `SONIUM_HOT_MAX_MACS=n` overrides
/// [`HOT_MAX_MACS`] (parsed once per scope-routing call).
#[inline]
fn hot_max_macs() -> usize {
    std::env::var("SONIUM_HOT_MAX_MACS")
        .ok()
        .and_then(|text| text.parse::<usize>().ok())
        .filter(|macs| *macs > 0)
        .unwrap_or(HOT_MAX_MACS)
}

/// Whether a scope of `macs` multiply-accumulates takes the hot-workers
/// pool: top-level calls only (an ordinary thread — inside any pool the
/// enclosing width already governs and re-installing would deadlock),
/// wide machines only (small machines never hop, as in
/// [`workers_for_macs`]), no active [`pool_floor_small`] guard (floored
/// solves stay monochromatic on tiny/small), and only when the compact
/// topology formed a pool to borrow. Results are bitwise identical on
/// every pool (see [`scope_for_macs`]).
#[inline]
pub(crate) fn hot_tier_applies(macs: usize) -> bool {
    rayon::current_thread_index().is_none()
        && rayon::current_num_threads() > SMALL_THREADS
        && POOL_FLOOR.load(Ordering::Relaxed) == 0
        && macs < hot_max_macs()
        && compact_pool(true).is_some()
}

/// Worker count of the hot-workers pool (the borrowed compact width).
/// Only meaningful when [`hot_tier_applies`] holds; kept beside it so the
/// router and the sizers cannot disagree.
#[inline]
fn hot_tier_workers() -> usize {
    compact_pool(true)
        .map(|pool| pool.current_num_threads().max(1))
        .unwrap_or(SMALL_THREADS)
}

pub(crate) const SMALL_THREADS: usize = 48;

/// Per-scope multiply-accumulate ceiling for the small-work pool: scopes
/// under this run on [`SMALL_THREADS`] workers instead of the global pool.
/// Sized so a 1024^3 GEMM (268M MACs per Kc pass) routes small while a
/// 2048^3 GEMM (1.07B per pass) stays global.
pub(crate) const SMALL_MAX_MACS: usize = 512_000_000;

thread_local! {
    static IN_SMALL_POOL: Cell<bool> = const { Cell::new(false) };
    static IN_TINY_POOL: Cell<bool> = const { Cell::new(false) };
    static IN_MIRROR_POOL: Cell<bool> = const { Cell::new(false) };
}

/// Compact single-L3-domain pool for factorizations whose working set
/// fits in L3 (see [`super::affinity::compact_cpus`]): keeping the whole
/// panel/TRSM/update sequence on one CCX holds the trailing matrix
/// L3-resident across panels (dgetrf-1024 ~11.7ms vs ~14.6 spread).
/// `None` when topology gave no compact set or the build failed — the
/// zone then falls back to the mirror pool. Sized by the compact set
/// (measured shape: 4 workers on 4 cores), not the ambient pool.
static COMPACT_POOL: std::sync::LazyLock<Option<rayon::ThreadPool>> =
    std::sync::LazyLock::new(|| {
        let cpus = super::affinity::compact_cpus()?;
        if cpus.is_empty() {
            return None;
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(cpus.len())
            .thread_name(|i| format!("sonium-compact-{i}"))
            .start_handler(super::affinity::pin_compact_worker)
            .build()
            .ok()
    });

/// Wide compact pool over [`super::affinity::COMPACT_WIDE_GROUPS`]
/// L3 domains (see [`super::affinity::compact_cpus_wide`]):
/// compute-dense rank-64 SYRK
/// wants the extra domains' slices and FMA units (spotrf-1024 7.3 ->
/// 5.9ms vs one CCX at two domains; dpotrf-1024 -6% more at four),
/// while memory-bound rank-32 LU stays narrow. `None` when the wide
/// set coincides with the narrow set — callers share [`COMPACT_POOL`]
/// then, so no duplicate pool is ever built.
static COMPACT_POOL_WIDE: std::sync::LazyLock<Option<rayon::ThreadPool>> =
    std::sync::LazyLock::new(|| {
        let cpus = super::affinity::compact_cpus_wide()?;
        if cpus.is_empty() {
            return None;
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(cpus.len())
            .thread_name(|i| format!("sonium-compact-wide-{i}"))
            .start_handler(super::affinity::pin_compact_wide_worker)
            .build()
            .ok()
    });

/// The compact pool for `wide`: the wide pool when built, else the
/// narrow pool (still compact, better than the mirror fallback).
fn compact_pool(wide: bool) -> Option<&'static rayon::ThreadPool> {
    if wide {
        COMPACT_POOL_WIDE.as_ref().or(COMPACT_POOL.as_ref())
    } else {
        COMPACT_POOL.as_ref()
    }
}

static SMALL_POOL: std::sync::LazyLock<rayon::ThreadPool> = std::sync::LazyLock::new(|| {
    // Packed sequential placement (see `affinity`): spreading won cold
    // single-SYRK calls but loses factorizations 2.3x (dpotrf-1024 8T
    // packed 4.4-4.5 vs spread 9.9-11.0) — cross-panel L3 reuse of the
    // trailing matrix beats conflict avoidance; do not re-spread without
    // factorization-level evidence. Standalone GEMM-MT re-tested too
    // (2026-09-27, GEMM-only spread router, reverted): sq1024-MT16
    // spread-vs-packed ties (-5% median, +/-10% noise, 4 rounds) after a
    // taskset mirage (OB confinement artifact + outlier luck) briefly
    // suggested 2x — packed stands for GEMM as well.
    rayon::ThreadPoolBuilder::new()
        .num_threads(SMALL_THREADS)
        .thread_name(|i| format!("sonium-small-{i}"))
        .start_handler(super::affinity::pin_worker)
        .build()
        .expect("small-work rayon pool")
});

static TINY_POOL: std::sync::LazyLock<rayon::ThreadPool> = std::sync::LazyLock::new(|| {
    rayon::ThreadPoolBuilder::new()
        .num_threads(TINY_THREADS)
        .thread_name(|i| format!("sonium-tiny-{i}"))
        .start_handler(super::affinity::pin_worker)
        .build()
        .expect("tiny-work rayon pool")
});

/// Ambient-sized mirror pool behind [`scope_pinned`]: same worker count the
/// `rayon::scope` fallback would have used (the ambient global pool size,
/// so `RAYON_NUM_THREADS` caps and app pool sizings are respected), but
/// with pinned workers. Sized once at first top-level scope: init from an
/// ordinary thread reads the ambient global size; init from inside a pool
/// (only possible via a nested top-level-looking call) reads that pool's
/// size, which is the right width there too.
static MIRROR_POOL: std::sync::LazyLock<rayon::ThreadPool> = std::sync::LazyLock::new(|| {
    let ambient = rayon::current_num_threads().max(1);
    // Mask-aware sizing: under a cpuset/mask (containers, numactl,
    // taskset) an ambient-sized pool oversubscribes the allowed CPUs
    // (e.g. 128 threads on 8 cores); size by the inherited mask instead.
    // The min keeps explicit `RAYON_NUM_THREADS` caps and small-pool
    // inits working as before; ambient unmasked processes are unchanged.
    let n = super::affinity::mask_cpu_count()
        .map(|m| ambient.min(m))
        .unwrap_or(ambient)
        .max(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .thread_name(|i| format!("sonium-mirror-{i}"))
        .start_handler(super::affinity::pin_worker)
        .build()
        .expect("mirror rayon pool")
});

/// Spread pool for streaming kernels (currently GEMV NoTrans): workers
/// pinned one per L3 slice (see [`super::affinity::spread_cpus`]) so each
/// worker's band stays in its own slice across reps. Callers use
/// [`rayon::ThreadPool::broadcast`] (exactly one invocation per worker,
/// indexed by [`rayon::WorkerContext::index`]) rather than scopes: work
/// stealing would rotate bands across slices every rep and defeat the
/// point (measured: band→CPU map fully reshuffles rep to rep on the
/// global pool). `None` when topology gave no spread set — callers fall
/// back to a raw scope. Sized by the spread set, not the ambient pool.
fn build_spread_pool(num_threads: usize, tag: &'static str) -> Option<rayon::ThreadPool> {
    let cpus = super::affinity::spread_cpus()?;
    if cpus.is_empty() {
        return None;
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .thread_name(move |i| format!("sonium-spread-{tag}-{i}"))
        .start_handler(super::affinity::pin_spread_worker)
        .build()
        .ok()
}

static SPREAD_POOL: std::sync::LazyLock<Option<rayon::ThreadPool>> =
    std::sync::LazyLock::new(|| {
        let cpus = super::affinity::spread_cpus()?;
        build_spread_pool(cpus.len(), "full")
    });

/// Ambient-sized spread pools (see [`spread_pool_for_ambient`]): same
/// pinning as the full pool, so worker `i` lands on the same spread CPU
/// as full-pool worker `i` (`pin_spread_worker` indexes into the same
/// set). Only built when a sub-width ambient pool actually calls.
static SPREAD_POOL_W2: std::sync::LazyLock<Option<rayon::ThreadPool>> =
    std::sync::LazyLock::new(|| build_spread_pool(2, "w2"));
static SPREAD_POOL_W4: std::sync::LazyLock<Option<rayon::ThreadPool>> =
    std::sync::LazyLock::new(|| build_spread_pool(4, "w4"));
static SPREAD_POOL_W8: std::sync::LazyLock<Option<rayon::ThreadPool>> =
    std::sync::LazyLock::new(|| build_spread_pool(8, "w8"));

/// The spread pool for streaming kernels, if one could be built.
pub(crate) fn spread_pool() -> Option<&'static rayon::ThreadPool> {
    SPREAD_POOL.as_ref()
}

/// Spread pool sized to the ambient pool: the largest cached width at or
/// below `min(ambient, full_width)`, so broadcast dispatch keeps working
/// when the caller caps threads below the full spread width instead of
/// falling back to the stealing raw scope (which reshuffles bands across
/// L3 slices every rep and plateaus ~4 threads: sq4k-f32 NoTrans MT-8
/// measured 9-12x slower than MT-16 broadcast on the same box).
/// Ambient widths at/above full return the full pool (identical to the
/// legacy width check there); odd widths floor to 2/4/8; ambient < 2
/// returns `None` (serial-ish anyway). `SONIUM_GEMV_AMBIENT=0` restores
/// the legacy rule (full pool only when the ambient pool could fill it)
/// for same-binary A/B; default-on.
pub(crate) fn spread_pool_for_ambient() -> Option<&'static rayon::ThreadPool> {
    let full = SPREAD_POOL.as_ref()?;
    let full_w = full.current_num_threads().max(1);
    let ambient = rayon::current_num_threads().max(1);
    if ambient >= full_w {
        return Some(full);
    }
    let legacy_only = std::env::var("SONIUM_GEMV_AMBIENT")
        .map(|v| v == "0")
        .unwrap_or(false);
    if legacy_only {
        return None;
    }
    let cap = ambient.min(full_w);
    if cap >= 8 {
        SPREAD_POOL_W8.as_ref()
    } else if cap >= 4 {
        SPREAD_POOL_W4.as_ref()
    } else if cap >= 2 {
        SPREAD_POOL_W2.as_ref()
    } else {
        None
    }
}

/// Process-wide pool floor (see [`pool_floor_small`]): `0` routes every
/// scope by size alone, `1` lifts tiny-eligible scopes onto the small pool.
/// A plain [`AtomicU8`] (not thread-local) so both the scope router on the
/// caller thread and the task sizers on pool worker threads see the same
/// value — pool choice and task sizing can never disagree. Only ever a
/// performance hint: every pool computes bitwise-identical results.
static POOL_FLOOR: AtomicU8 = AtomicU8::new(0);

/// RAII guard pinning this thread's solve to one pool: while alive, scopes
/// that would take the tiny pool take the small pool instead, so a solve
/// whose scopes straddle the tiny ceiling stays monochromatic. Motivation:
/// alternating tiny and small scopes in one solve (e.g. TRSM-1024: twelve
/// small scopes plus three tiny trailing updates) runs ~17% slower than
/// either pool alone — consistent with per-switch package power-state
/// transition costs on wide machines — while all-tiny (TRSM-512/640) and
/// all-small solves run at full rate. Dropping the guard restores the
/// previous floor (nesting-safe); concurrent solves on other threads
/// observe the floor too, so treat it as a solve-wide hint, not a lock.
pub(crate) struct PoolFloorGuard {
    prev: u8,
}

/// Pin tiny-eligible scopes onto the small pool until the guard drops.
/// Only floors upward (tiny → small): small/global routing is unchanged,
/// and results are bitwise identical on every pool.
pub(crate) fn pool_floor_small() -> PoolFloorGuard {
    PoolFloorGuard {
        prev: POOL_FLOOR.swap(1, Ordering::SeqCst),
    }
}

impl Drop for PoolFloorGuard {
    fn drop(&mut self) {
        POOL_FLOOR.store(self.prev, Ordering::SeqCst);
    }
}

/// Worker count of the pool [`scope_for_macs`] will pick for `macs`
/// multiply-accumulates: the single predictor its callers' task sizers
/// mirror (see `rows_per_task` / `cols_per_task`). Kept as one function so
/// the router and the sizers cannot disagree: both gate pool hops on
/// `global > SMALL_THREADS` (small machines never hop), both prefer the hot
/// pool below [`HOT_MAX_MACS`] at top level (see [`hot_tier_applies`]), the
/// tiny pool below [`TINY_MAX_MACS`], the small pool below
/// [`SMALL_MAX_MACS`], and the global pool otherwise. Inside a capped pool
/// `current_num_threads` is already the pool size, so the gates fail and
/// the sizers see the executing pool's width. An active [`pool_floor_small`]
/// guard lifts the tiny answer to the small pool here too.
pub(crate) fn workers_for_macs(macs: usize) -> usize {
    let global = rayon::current_num_threads().max(1);
    if hot_tier_applies(macs) {
        return hot_tier_workers();
    }
    if global > SMALL_THREADS && macs < TINY_MAX_MACS && POOL_FLOOR.load(Ordering::Relaxed) == 0 {
        TINY_THREADS
    } else if global > SMALL_THREADS && macs < SMALL_MAX_MACS {
        SMALL_THREADS
    } else {
        global
    }
}

/// Run `op` in a `rayon::scope` on pinned workers.
///
/// Top-level calls (an ordinary thread, not inside any pool) install the
/// ambient-sized [`MIRROR_POOL`]: same worker count the plain
/// `rayon::scope` fallback would have used, but with pinned workers, so
/// FMA-bound bands stop sharing physical cores with SMT siblings (see
/// [`super::affinity`]). Nested calls — inside our pools (tracked per
/// thread), the ambient global pool, or an app pool — run inline on the
/// enclosing pool, exactly as before: stealing the scope onto the mirror
/// would abandon the enclosing pool's workers mid-solve. Results are
/// bitwise identical on every pool: scope placement never changes numbers.
pub fn scope_pinned<'scope, OP>(op: OP)
where
    OP: for<'a> FnOnce(&'a rayon::Scope<'scope>) + Send,
{
    let nested = IN_SMALL_POOL.with(|c| c.get())
        || IN_TINY_POOL.with(|c| c.get())
        || IN_MIRROR_POOL.with(|c| c.get())
        || rayon::current_thread_index().is_some();
    if nested {
        rayon::scope(op);
        return;
    }
    MIRROR_POOL.install(|| {
        IN_MIRROR_POOL.with(|c| c.set(true));
        rayon::scope(op);
        IN_MIRROR_POOL.with(|c| c.set(false));
    });
}

/// Run `f` once per band over `nbands` bands.
///
/// The spin-pool path broadcasts one closure to pinned workers
/// (~0.3µs dispatch in the bursting steady state), each worker running
/// a contiguous run of bands; bands are caller-mapped (a band index
/// means whatever rows the caller derives from it), so factorization
/// panel loops keep their exact banding while shedding the ~70µs
/// per-panel `par_iter`/scope dispatch. Falls back to one task per
/// band on a pinned scope when the spin pool is unavailable, when
/// broadcasting from a spin-pool worker, or when the pool is narrower
/// than the ambient rayon width (wide unmasked runs keep their
/// workers). Results are identical on both paths: band contents never
/// depend on the executing worker.
///
/// # Examples
///
/// ```rust
/// use sonium_compute::cpu::blas3::broadcast_bands;
/// use std::sync::atomic::{AtomicUsize, Ordering};
///
/// let hits: Vec<AtomicUsize> = (0..64).map(|_| AtomicUsize::new(0)).collect();
/// broadcast_bands(64, &|b: usize| {
///     hits[b].fetch_add(1, Ordering::Relaxed);
/// });
/// assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
/// ```
///
/// # Panics
///
/// A band closure's panic resumes on the caller after every worker
/// finished (both paths). Spin-pool worker callers take the scope
/// fallback (broadcasting to their own pool would wait on itself
/// forever); rayon-pool callers take the spin path safely, since the
/// spin workers are separate threads.
pub fn broadcast_bands<F: Fn(usize) + Sync>(nbands: usize, f: &F) {
    if nbands == 0 {
        return;
    }
    // Nesting guard first: a spin worker broadcasting to its own pool
    // would deadlock (see `SpinPool::broadcast`).
    if !super::spin_pool::SpinPool::on_worker()
        && let Some(pool) = super::spin_pool::spin_pool()
    {
        // Spin only when the pool covers the ambient width: fewer
        // workers than the rayon fallback would use loses parallelism
        // on wide unmasked runs. (BLAS-1's rule is
        // oversubscription-only because bandwidth saturates; these
        // bands are compute-bound, so width matters.)
        if pool.width() >= rayon::current_num_threads().max(1) {
            let per = nbands.div_ceil(pool.width());
            pool.broadcast(&|w: usize| {
                let b0 = (w * per).min(nbands);
                let b1 = (b0 + per).min(nbands);
                for b in b0..b1 {
                    f(b);
                }
            });
            return;
        }
    }
    scope_pinned(|s| {
        for b in 0..nbands {
            s.spawn(move |_| f(b));
        }
    });
}

/// Run `op` to completion on a pool worker while the caller blocks.
///
/// Serial solve phases run on the calling thread at whatever frequency
/// that core happens to hold: under `schedutil` an idle core sits near
/// 2.2GHz vs 4.3GHz boosted, doubling serial time (dgetrf-1024 with a
/// 128-CPU mask: caller on cold core 21.0ms, roaming 15.6ms, confined to
/// hot cores 10.2ms — the whole delta is caller-core frequency). Pool
/// workers stay hot from the parallel phases, so serial work runs at full
/// rate there. Nested calls (already on a worker of any pool) run inline:
/// a worker blocking on its own pool would deadlock, and workers are
/// already hot — the funnel only helps ordinary, possibly cold, calling
/// threads. Panics propagate to the caller. Costs one spawn round-trip
/// (~µs); parallel scopes inside still fan out via [`scope_pinned`].
pub fn run_on_worker<R, F>(op: F) -> R
where
    R: Send + 'static,
    F: FnOnce() -> R + Send + 'static,
{
    if rayon::current_thread_index().is_some() {
        return op();
    }
    let (tx, rx) = std::sync::mpsc::channel();
    MIRROR_POOL.spawn(move || {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(op));
        let _ = tx.send(r);
    });
    match rx.recv() {
        Ok(Ok(value)) => value,
        Ok(Err(payload)) => std::panic::resume_unwind(payload),
        Err(_) => panic!("sonium worker terminated while running a funneled task"),
    }
}

/// Run `op` on the mirror pool via `install` (borrow) instead of
/// [`run_on_worker`]'s `spawn`: the caller joins the pool while blocked,
/// so serial phases run on the (usually hot) caller and the pool gains a
/// worker instead of parking the caller on a channel. Same pool, same
/// worker count, same static task splits as the spawn path — only the
/// launching thread differs, so the Cholesky mirror path (its only
/// caller) reproduces the spawn path's factors bitwise: same kernels,
/// same per-element order, no cross-task reductions. Nested calls run
/// inline, exactly as in [`run_on_worker`]. Panics propagate to the
/// caller. No `IN_MIRROR_POOL` flag (same as [`run_compact`]: pool
/// membership already routes nested scopes inline).
///
/// Rationale (2026-10-02, `cholphases` process-interleaved A/B,
/// RAYON=16): install beats spawn 8/8 rounds (~-15% at dpotrf-1024 on
/// identical 16 CPUs) — the serial panel runs 2x slower on the spawned
/// worker and TRSM phases cost +40%, consistent with a cold/parked
/// spawn target under `schedutil` turning straggler.
pub fn run_mirror<R, F>(op: F) -> R
where
    R: Send,
    F: FnOnce() -> R + Send,
{
    if rayon::current_thread_index().is_some() {
        return op();
    }
    MIRROR_POOL.install(op)
}

/// Whether the compact pool applies to a working set of `est_bytes`:
/// enabled (no `SONIUM_COMPACT=0`), topology formed the requested pool,
/// and the set fits in L3 (with headroom for packs/scratch). Drivers
/// check this to decide whether to clone inside [`run_compact`] (local
/// first-touch) or outside for the [`run_on_worker`] fallback. `wide`
/// selects the multi-domain pool (Cholesky) vs the single-domain pool
/// (LU); a missing wide pool falls back to narrow (see [`compact_pool`]).
pub fn compact_applies(est_bytes: u64, wide: bool) -> bool {
    if std::env::var("SONIUM_COMPACT")
        .map(|v| v == "0")
        .unwrap_or(false)
    {
        return false;
    }
    let Some(l3) = super::affinity::l3_bytes() else {
        return false;
    };
    if compact_pool(wide).is_none() {
        return false;
    }
    // Factor plus packs/scratch stay resident: 3/4 of L3.
    est_bytes > 0 && est_bytes <= l3 * 3 / 4
}

/// Run `op` like [`run_on_worker`], but on the compact pool when it
/// applies: serial phases run on one hot worker and every parallel scope
/// inside stays on the same L3 domain(s) with no router changes (pool
/// membership already routes scopes inline, and the task sizers see the
/// compact width via `current_num_threads`). Unlike [`run_on_worker`]
/// this borrows through the pool (`install`, not `spawn`), so callers
/// clone/work inside and first-touch binds pages to the compact NUMA
/// node — cloning outside would bind them to the caller's node. `wide`
/// selects the multi-domain pool (Cholesky's compute-dense SYRK) vs the
/// single-domain pool (LU's memory-bound updates). Nested calls run
/// inline, exactly as in [`run_on_worker`]. Panics propagate.
/// `SONIUM_COMPACT=0` forces the mirror-pool path.
pub fn run_compact<R, F>(est_bytes: u64, wide: bool, op: F) -> R
where
    R: Send,
    F: FnOnce() -> R + Send,
{
    if rayon::current_thread_index().is_some() {
        return op();
    }
    if !compact_applies(est_bytes, wide) {
        // No funnel here: drivers branch on `compact_applies` first and
        // keep the `run_on_worker` funnel on the fallback path, so this
        // is defensive-only (pool vanished between check and use).
        return op();
    }
    let pool = compact_pool(wide).expect("compact pool checked above");
    pool.install(op)
}

/// Run `op` on the tiny-work pool (16 pinned workers) with the
/// `IN_TINY` flag set, so nested scopes route inline with 16-wide
/// sizings and zero per-call pool installs — one install per
/// factorization instead of ~200 nested hops (each hop is a cross-pool
/// install + wake + join, ~10-30µs). Mirrors the [`scope_for_macs`]
/// tiny branch (flag set on the executing worker, so scope bodies
/// spawned there route inline). Nested calls run inline. Pool choice
/// can change task geometry (hence TRSM band splits) at last-ulp
/// level; factorizations keep same pivots with equal residuals.
/// Used by the LU driver (see `SONIUM_LU_TINY`).
pub fn run_tiny<R, F>(op: F) -> R
where
    R: Send,
    F: FnOnce() -> R + Send,
{
    if rayon::current_thread_index().is_some() {
        return op();
    }
    TINY_POOL.install(|| {
        IN_TINY_POOL.with(|c| c.set(true));
        let r = op();
        IN_TINY_POOL.with(|c| c.set(false));
        r
    })
}

/// Dispatch a `rayon::scope` whose tasks spawn further parallel scopes.
/// The whole tree stays on one pool: when the machine is wide, nested
/// children land on the small pool (see [`scope_for_macs`]), so a
/// 128-wide outer scope multiplexing 48-wide inner scopes collapses
/// (SYRK 2048 drops 0.95x to 0.65x) while an all-48 tree holds the rate.
/// Pass the same `nest_outer`-style flag the scope body uses to decide
/// whether children go parallel: when children stay serial, the outer
/// scope keeps the default pool.
pub(crate) fn scope_for_nested_outer<'scope, F>(nest_outer: bool, f: F)
where
    F: for<'a> FnOnce(&'a rayon::Scope<'scope>) + Send,
{
    let in_small = IN_SMALL_POOL.with(|c| c.get());
    let in_tiny = IN_TINY_POOL.with(|c| c.get());
    if nest_outer && !in_small && !in_tiny && rayon::current_num_threads() > SMALL_THREADS {
        SMALL_POOL.install(|| {
            IN_SMALL_POOL.with(|c| c.set(true));
            rayon::scope(f);
            IN_SMALL_POOL.with(|c| c.set(false));
        });
        return;
    }
    scope_pinned(f);
}

/// Dispatch a `rayon::scope` on the pool that fits the work: scopes under
/// [`SMALL_MAX_MACS`] MACs run on the capped small-work pool instead of the
/// global one, and scopes under [`TINY_MAX_MACS`] MACs run on the even
/// smaller tiny-work pool. Past ~48 workers the pool-wide barrier,
/// work-stealing coordination, and SMT-sibling contention cost more than
/// extra cores return on small scopes (1024^3 GEMM halves going 48 to 128
/// threads while OpenBLAS stays flat); capping recovers the 32-thread
/// rate. Below 64M MACs even 48 workers oversubscribe (1024²×32 peaks at
/// 111–120 GF/s on 16 workers, ~83 GF/s on 48+), so the tiny tier recovers
/// the 16-thread rate there.
///
/// Results are unchanged: GEMM tasks own disjoint `C` elements with the
/// same per-element accumulation order at any worker count, TRSM bands
/// solve independent right-hand sides (no cross-band reduction), and pack
/// tasks are pure copies — so the worker count never changes the numbers.
/// Reentrancy-safe: scopes already running on a capped pool (tracked per
/// thread) stay on their current pool instead of nesting installs, and
/// pools with [`SMALL_THREADS`] or fewer workers skip the hop entirely —
/// so small machines and nested calls behave as before. The pool choice
/// always agrees with [`workers_for_macs`]. An active [`pool_floor_small`]
/// guard lifts the tiny branch to the small pool here too.
pub(crate) fn scope_for_macs<'scope, OP>(macs: usize, op: OP)
where
    OP: for<'a> FnOnce(&'a rayon::Scope<'scope>) + Send,
{
    let in_small = IN_SMALL_POOL.with(|c| c.get());
    let in_tiny = IN_TINY_POOL.with(|c| c.get());
    // Hot-workers tier first (see `hot_tier_applies`): short top-level
    // scopes concentrate on L3-neighbour hot cores instead of spraying
    // parked ones. The install borrows the compact pool without setting an
    // `IN_*` flag — nested calls inside still route inline (their
    // `current_num_threads` is the hot width, so every hop gate fails and
    // the sizers see the executing width), exactly as in the compact
    // factorization pools.
    if hot_tier_applies(macs) {
        let pool = compact_pool(true).expect("hot tier checked above");
        pool.install(|| rayon::scope(op));
        return;
    }
    if !in_small && !in_tiny && rayon::current_num_threads() > SMALL_THREADS {
        if macs < TINY_MAX_MACS && POOL_FLOOR.load(Ordering::Relaxed) == 0 {
            TINY_POOL.install(|| {
                IN_TINY_POOL.with(|c| c.set(true));
                rayon::scope(op);
                IN_TINY_POOL.with(|c| c.set(false));
            });
            return;
        }
        if macs < SMALL_MAX_MACS {
            SMALL_POOL.install(|| {
                IN_SMALL_POOL.with(|c| c.set(true));
                rayon::scope(op);
                IN_SMALL_POOL.with(|c| c.set(false));
            });
            return;
        }
    }
    scope_pinned(op);
}

/// Borrow a thread-local pack buffer, growing to `need` on first call.
///
/// Reentrancy: the borrow is held while `f` runs (including the SME2 GEMM
/// and its internal `rayon::scope`s), and a rayon worker waiting in one of
/// those scopes may steal another GEMM onto the same thread, re-entering
/// this pool slot. `try_borrow_mut` detects that case and falls back to a
/// freshly allocated buffer owned by the nested call — one allocation on
/// the nested path instead of panicking with "RefCell already borrowed".
/// The hot non-nested path is unchanged.
/// 64B pack-slice alignment (shipped default-on,
/// `SONIUM_GEMM_ALIGN64=0` restores the 16B `Vec` base): our `Vec`
/// pack buffers are 16B-aligned, so ~50% of the sweep's 32B vector
/// loads split cache lines; OB's 64B panels never split. Same-process
/// ST A/B 2026-09-28 (alternating order, OB ref, 3 seeds ×
/// k512/1024/2048): aligned 0.947-0.960x, 9/9 cells, bitwise exact.
/// Read per call so same-process A/B can interleave arms (~50 ns).
#[inline]
pub(crate) fn align64_on() -> bool {
    std::env::var("SONIUM_GEMM_ALIGN64")
        .map(|v| v != "0")
        .unwrap_or(true)
}

/// Raw spare-capacity twin of [`align64_slice`] for the M-split
/// panel's uninitialised-write pattern: reserves (never zero-fills —
/// the pack overwrites) and returns the 64B-aligned base pointer
/// into the spare capacity. Callers must write `need` elements
/// before any read.
pub(crate) fn align64_spare_ptr<T>(buf: &mut Vec<T>, need: usize) -> *mut T {
    if !align64_on() {
        buf.reserve(need);
        return buf.as_mut_ptr();
    }
    let el = core::mem::size_of::<T>().max(1);
    buf.reserve(need + 64 / el);
    let mis = (buf.as_ptr() as usize) % 64;
    let off = if mis == 0 { 0 } else { (64 - mis) / el };
    // SAFETY: `off + need <= capacity` by the reservation.
    unsafe { buf.as_mut_ptr().add(off) }
}

/// Grow `buf` to `need` elements and lend a 64B-aligned `need`-slice
/// (see [`align64_on`]; offset wastes at most 63 bytes). Used by all
/// pack-buffer lenders (B tiles, A packs/panels) so every vectorised
/// pack store and sweep load is line-aligned.
pub(crate) fn align64_slice<T: Copy + Default>(buf: &mut Vec<T>, need: usize) -> &mut [T] {
    if !align64_on() {
        if buf.len() < need {
            buf.resize(need, T::default());
        }
        return &mut buf[..need];
    }
    let el = core::mem::size_of::<T>().max(1);
    let slack = 64 / el;
    if buf.len() < need + slack {
        buf.resize(need + slack, T::default());
    }
    let mis = (buf.as_ptr() as usize) % 64;
    let off = if mis == 0 { 0 } else { (64 - mis) / el };
    &mut buf[off..off + need]
}

pub(crate) fn with_buf<T, F, R>(
    cell: &'static std::thread::LocalKey<RefCell<Vec<T>>>,
    need: usize,
    f: F,
) -> R
where
    T: Copy + Default + 'static,
    F: FnOnce(&mut [T]) -> R,
{
    cell.with(|c| match c.try_borrow_mut() {
        Ok(mut buf) => {
            let dst = align64_slice(&mut buf, need);
            f(dst)
        }
        Err(_) => {
            let mut buf = Vec::new();
            let dst = align64_slice(&mut buf, need);
            f(dst)
        }
    })
}

#[inline]
fn bget_f32(b: &[f32], ldb: usize, p: usize, j: usize, op: Transpose) -> f32 {
    match op {
        Transpose::NoTrans => b[p * ldb + j],
        Transpose::Trans | Transpose::ConjTrans => b[j * ldb + p],
    }
}

#[inline]
fn bget_f64(b: &[f64], ldb: usize, p: usize, j: usize, op: Transpose) -> f64 {
    match op {
        Transpose::NoTrans => b[p * ldb + j],
        Transpose::Trans | Transpose::ConjTrans => b[j * ldb + p],
    }
}

#[inline]
fn cget_c32(a: &[Complex32], lda: usize, i: usize, p: usize, op: Transpose) -> Complex32 {
    match op {
        Transpose::NoTrans => a[i * lda + p],
        Transpose::Trans => a[p * lda + i],
        Transpose::ConjTrans => a[p * lda + i].conj(),
    }
}

#[inline]
fn cget_c64(a: &[Complex64], lda: usize, i: usize, p: usize, op: Transpose) -> Complex64 {
    match op {
        Transpose::NoTrans => a[i * lda + p],
        Transpose::Trans => a[p * lda + i],
        Transpose::ConjTrans => a[p * lda + i].conj(),
    }
}

/// Minimum pack elements (`rows × cols` of the packed buffer) before the
/// transpose-pack fans out over rayon. Below this the serial loop wins.
/// A transposed pack reads its source strided (one fresh cache line/TLB
/// entry per element at large strides), so single-threaded packing caps at
/// ~1–2 GB/s — threading it is worth ~8x on multi-MB packs.
const PACK_PAR_MIN_ELS: usize = 65_536;

/// Pointer wrappers for `rayon::scope` pack dispatch: disjoint `out` row
/// bands, read-only source.
#[derive(Clone, Copy)]
struct PackConstPtr<T>(*const T);
unsafe impl<T> Send for PackConstPtr<T> {}
unsafe impl<T> Sync for PackConstPtr<T> {}
#[derive(Clone, Copy)]
struct PackMutPtr<T>(*mut T);
unsafe impl<T> Send for PackMutPtr<T> {}
unsafe impl<T> Sync for PackMutPtr<T> {}

/// Parallel pack driver: `out[r * cols + c] = src[idx(r, c)]` over row
/// bands. The index closure hoists the transpose/conjugation op out of the
/// per-element loop; callers keep their serial loop below the size gate.
fn pack_parallel<T, F>(rows: usize, cols: usize, src: &[T], out: &mut [T], idx: F)
where
    T: Copy + Send + Sync,
    F: Fn(usize, usize) -> usize + Send + Sync,
{
    debug_assert!(rows.saturating_mul(cols) <= out.len());
    let workers = rayon::current_num_threads().max(1);
    let tasks = (rows / 64).clamp(1, workers.saturating_mul(4).max(1));
    let s_send = PackConstPtr(src.as_ptr());
    let slen = src.len();
    let o_send = PackMutPtr(out.as_mut_ptr());
    // Shared reference: `&F` is `Copy`, so every spawned task can hold it
    // without moving `idx` (`F: Sync` carries across threads).
    let idx_ref = &idx;
    rayon::scope(|s| {
        let mut r0 = 0;
        while r0 < rows {
            let r1 = (r0 + rows.div_ceil(tasks)).min(rows);
            s.spawn(move |_| {
                let s_send: PackConstPtr<T> = s_send;
                let o_send: PackMutPtr<T> = o_send;
                // SAFETY: task owns `out` rows `[r0, r1)`; source
                // read-only. `idx` stays in bounds per the caller contract.
                unsafe {
                    let src_slice = core::slice::from_raw_parts(s_send.0, slen);
                    let out_slice =
                        core::slice::from_raw_parts_mut(o_send.0.add(r0 * cols), (r1 - r0) * cols);
                    for r in r0..r1 {
                        for c in 0..cols {
                            out_slice[(r - r0) * cols + c] = src_slice[idx_ref(r, c)];
                        }
                    }
                }
            });
            r0 = r1;
        }
    });
}

fn pack_b_f32(b: &[f32], ldb: usize, n: usize, k: usize, op: Transpose, out: &mut [f32]) {
    // Vectorised 8×8 transpose-pack on x86 (see `pack_trans_f64` for the
    // rationale). Used off-x86 and without AVX2; on x86 the kernels gather
    // `Trans` tiles/panels directly (no pre-pack).
    // SAFETY: `out` holds `k * n` elements and `b` the `k × n` source rows
    // per the caller contract; the kernel only reads/writes those ranges.
    // Bands below rebase both pointers so each task runs the same kernel on
    // its own packed-row range (packed rows are contiguous; source rows
    // rebase by the same offset since `dst[R][C] = src[C][R]`).
    #[cfg(target_arch = "x86_64")]
    if !matches!(op, Transpose::NoTrans) && crate::cpu::blas3::avx2::is_available() {
        let workers = rayon::current_num_threads().max(1);
        let tasks = workers.min(k.div_ceil(64)).max(1);
        if tasks == 1 {
            unsafe {
                crate::cpu::blas3::avx2::pack_trans_f32(b.as_ptr(), ldb, k, n, out.as_mut_ptr());
            }
            return;
        }
        let s_send = PackConstPtr(b.as_ptr());
        let o_send = PackMutPtr(out.as_mut_ptr());
        rayon::scope(|s| {
            let mut r0 = 0;
            while r0 < k {
                // Multiple of 8 rows keeps every band but the last on the
                // vector path; the kernel's scalar tails stay correct anyway.
                let mut r1 = (r0 + k.div_ceil(tasks)).min(k);
                if r1 < k {
                    r1 -= r1 % 8;
                }
                if r1 <= r0 {
                    r1 = (r0 + 8).min(k);
                }
                s.spawn(move |_| {
                    let s_send: PackConstPtr<f32> = s_send;
                    let o_send: PackMutPtr<f32> = o_send;
                    // SAFETY: task owns packed rows `[r0, r1)`; source
                    // read-only. Rebasing preserves the kernel contract.
                    unsafe {
                        crate::cpu::blas3::avx2::pack_trans_f32(
                            s_send.0.add(r0),
                            ldb,
                            r1 - r0,
                            n,
                            o_send.0.add(r0 * n),
                        );
                    }
                });
                r0 = r1;
            }
        });
        return;
    }
    if k.saturating_mul(n) >= PACK_PAR_MIN_ELS {
        match op {
            Transpose::NoTrans => pack_parallel(k, n, b, out, |p, j| p * ldb + j),
            Transpose::Trans | Transpose::ConjTrans => {
                pack_parallel(k, n, b, out, |p, j| j * ldb + p)
            }
        }
        return;
    }
    for p in 0..k {
        for j in 0..n {
            out[p * n + j] = bget_f32(b, ldb, p, j, op);
        }
    }
}

fn pack_b_f64(b: &[f64], ldb: usize, n: usize, k: usize, op: Transpose, out: &mut [f64]) {
    // Vectorised 8×4 transpose-pack on x86 (order of magnitude faster than
    // the scalar strided loop below). Used off-x86 and without AVX2; on
    // x86 the kernels gather `Trans` tiles/panels directly (no pre-pack).
    // `ConjTrans` == `Trans` for f64.
    // SAFETY: `out` holds `k * n` elements and `b` the `k × n` source rows
    // per the caller contract; the kernel only reads/writes those ranges.
    // Bands below rebase both pointers so each task runs the same kernel on
    // its own packed-row range (packed rows are contiguous; source rows
    // rebase by the same offset since `dst[R][C] = src[C][R]`).
    #[cfg(target_arch = "x86_64")]
    if !matches!(op, Transpose::NoTrans) && crate::cpu::blas3::avx2::is_available() {
        let workers = rayon::current_num_threads().max(1);
        let tasks = workers.min(k.div_ceil(64)).max(1);
        if tasks == 1 {
            unsafe {
                crate::cpu::blas3::avx2::pack_trans_f64(b.as_ptr(), ldb, k, n, out.as_mut_ptr());
            }
            return;
        }
        let s_send = PackConstPtr(b.as_ptr());
        let o_send = PackMutPtr(out.as_mut_ptr());
        rayon::scope(|s| {
            let mut r0 = 0;
            while r0 < k {
                // Multiple of 4 rows keeps every band but the last on the
                // vector path; the kernel's scalar tails stay correct anyway.
                let mut r1 = (r0 + k.div_ceil(tasks)).min(k);
                if r1 < k {
                    r1 -= r1 % 4;
                }
                if r1 <= r0 {
                    r1 = (r0 + 4).min(k);
                }
                s.spawn(move |_| {
                    let s_send: PackConstPtr<f64> = s_send;
                    let o_send: PackMutPtr<f64> = o_send;
                    // SAFETY: task owns packed rows `[r0, r1)`; source
                    // read-only. Rebasing preserves the kernel contract.
                    unsafe {
                        crate::cpu::blas3::avx2::pack_trans_f64(
                            s_send.0.add(r0),
                            ldb,
                            r1 - r0,
                            n,
                            o_send.0.add(r0 * n),
                        );
                    }
                });
                r0 = r1;
            }
        });
        return;
    }
    if k.saturating_mul(n) >= PACK_PAR_MIN_ELS {
        match op {
            Transpose::NoTrans => pack_parallel(k, n, b, out, |p, j| p * ldb + j),
            Transpose::Trans | Transpose::ConjTrans => {
                pack_parallel(k, n, b, out, |p, j| j * ldb + p)
            }
        }
        return;
    }
    for p in 0..k {
        for j in 0..n {
            out[p * n + j] = bget_f64(b, ldb, p, j, op);
        }
    }
}

/// Shared `A^T` pack for the multi-threaded trans-A path (see
/// `pack_a_mt`): `out[i * k + p] = a[p * lda + i]`, packed once and read
/// by every GEMM band task. Per-task strided gathers re-read `A` once
/// per band (`m * k` cache lines per task); the shared pack reads each
/// line once total. Bands mirror [`pack_b_f64`] (pointer rebasing keeps
/// the kernel contract `dst[R][C] = src[C * lda + R]`).
/// SAFETY: `out` holds `m * k` elements; `a` holds the `k x m` source
/// rows per the caller contract.
pub(crate) fn pack_a_trans_f64(a: &[f64], lda: usize, m: usize, k: usize, out: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::blas3::avx2::is_available() {
        let workers = rayon::current_num_threads().max(1);
        let tasks = workers.min(m.div_ceil(64)).max(1);
        if tasks == 1 {
            unsafe {
                crate::cpu::blas3::avx2::pack_trans_f64(a.as_ptr(), lda, m, k, out.as_mut_ptr());
            }
            return;
        }
        let s_send = PackConstPtr(a.as_ptr());
        let o_send = PackMutPtr(out.as_mut_ptr());
        rayon::scope(|s| {
            let mut r0 = 0;
            while r0 < m {
                // Multiple of 4 rows keeps every band but the last on the
                // vector path; the kernel's scalar tails stay correct anyway.
                let mut r1 = (r0 + m.div_ceil(tasks)).min(m);
                if r1 < m {
                    r1 -= r1 % 4;
                }
                if r1 <= r0 {
                    r1 = (r0 + 4).min(m);
                }
                s.spawn(move |_| {
                    let s_send: PackConstPtr<f64> = s_send;
                    let o_send: PackMutPtr<f64> = o_send;
                    // SAFETY: task owns packed rows `[r0, r1)`; source
                    // read-only. Rebasing preserves the kernel contract.
                    unsafe {
                        crate::cpu::blas3::avx2::pack_trans_f64(
                            s_send.0.add(r0),
                            lda,
                            r1 - r0,
                            k,
                            o_send.0.add(r0 * k),
                        );
                    }
                });
                r0 = r1;
            }
        });
        return;
    }
    if m.saturating_mul(k) >= PACK_PAR_MIN_ELS {
        pack_parallel(m, k, a, out, |r, c| c * lda + r);
        return;
    }
    for r in 0..m {
        for c in 0..k {
            out[r * k + c] = a[c * lda + r];
        }
    }
}

/// f32 twin of [`pack_a_trans_f64`]: shared `A^T` pack for the
/// multi-threaded trans-A path, same row-band split (see the f64
/// helper for the contract). The f32 MT driver packs serially without
/// this; the parallel pack is what lets the footprint crossover drop
/// (serial-pack cost kept the f32 gate at 16 MiB).
/// SAFETY: `out` holds `m * k` elements; `a` holds the `k x m` source
/// rows per the caller contract.
pub(crate) fn pack_a_trans_f32(a: &[f32], lda: usize, m: usize, k: usize, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::blas3::avx2::is_available() {
        let workers = rayon::current_num_threads().max(1);
        let tasks = workers.min(m.div_ceil(64)).max(1);
        if tasks == 1 {
            unsafe {
                crate::cpu::blas3::avx2::pack_trans_f32(a.as_ptr(), lda, m, k, out.as_mut_ptr());
            }
            return;
        }
        let s_send = PackConstPtr(a.as_ptr());
        let o_send = PackMutPtr(out.as_mut_ptr());
        rayon::scope(|s| {
            let mut r0 = 0;
            while r0 < m {
                // Multiple of 4 rows keeps every band but the last on the
                // vector path; the kernel's scalar tails stay correct anyway.
                let mut r1 = (r0 + m.div_ceil(tasks)).min(m);
                if r1 < m {
                    r1 -= r1 % 4;
                }
                if r1 <= r0 {
                    r1 = (r0 + 4).min(m);
                }
                s.spawn(move |_| {
                    let s_send: PackConstPtr<f32> = s_send;
                    let o_send: PackMutPtr<f32> = o_send;
                    // SAFETY: task owns packed rows `[r0, r1)`; source
                    // read-only. Rebasing preserves the kernel contract.
                    unsafe {
                        crate::cpu::blas3::avx2::pack_trans_f32(
                            s_send.0.add(r0),
                            lda,
                            r1 - r0,
                            k,
                            o_send.0.add(r0 * k),
                        );
                    }
                });
                r0 = r1;
            }
        });
        return;
    }
    if m.saturating_mul(k) >= PACK_PAR_MIN_ELS {
        pack_parallel(m, k, a, out, |r, c| c * lda + r);
        return;
    }
    for r in 0..m {
        for c in 0..k {
            out[r * k + c] = a[c * lda + r];
        }
    }
}

/// Cache-blocked out-of-place transpose-pack: `out[p*cols+j] =
/// src[j*ld+p]` in 8×8 tiles. The naive p-outer loop strides by `ld`
/// elements on every read — at band shapes (1KB+ strides) each read
/// wastes a cache line (195µs for a 1024×64 c64 pack); tiling keeps both
/// working sets L1-resident (52µs). Bitwise identical (pure copies).
/// Covers the serial `Trans` arms below; `ConjTrans` (value conjugation)
/// and the parallel path keep their loops.
fn pack_transpose_blocked<T: Copy>(src: &[T], ld: usize, rows: usize, cols: usize, out: &mut [T]) {
    pack_transpose_blocked_range(src, ld, cols, out, 0, rows);
}

/// Row-range half of [`pack_transpose_blocked`]: transposes output rows
/// `r0..r1` (`out[p*cols+j] = src[j*ld+p]`) in 8×8 tiles. Absolute
/// indexing (no sub-slice) so parallel bands share one helper.
fn pack_transpose_blocked_range<T: Copy>(
    src: &[T],
    ld: usize,
    cols: usize,
    out: &mut [T],
    r0: usize,
    r1: usize,
) {
    let mut p0 = r0;
    while p0 < r1 {
        let p1 = (p0 + 8).min(r1);
        let mut j0 = 0;
        while j0 < cols {
            let j1 = (j0 + 8).min(cols);
            for p in p0..p1 {
                let orow = p * cols;
                for j in j0..j1 {
                    out[orow + j] = src[j * ld + p];
                }
            }
            j0 = j1;
        }
        p0 = p1;
    }
}

/// Parallel blocked transpose-pack for large `Trans` operands. The
/// generic [`pack_parallel`] bands over rows with a naive inner loop —
/// at band shapes (rows=k=64) that spawns one task doing a fully-strided
/// serial transpose (~200µs for 64K complex els inside a scope). This
/// bands over 8-row multiples (tiles stay whole) with the blocked inner
/// loop, sized by total elements. Bitwise identical (pure copies).
fn pack_transpose_parallel<T: Copy + Send + Sync>(
    src: &[T],
    ld: usize,
    rows: usize,
    cols: usize,
    out: &mut [T],
) {
    let els = rows.saturating_mul(cols);
    let workers = rayon::current_num_threads().max(1);
    let tasks = (els / 4096).clamp(1, workers.saturating_mul(4).max(1));
    let bands = tasks.min(rows.div_ceil(8).max(1));
    if bands <= 1 {
        pack_transpose_blocked(src, ld, rows, cols, out);
        return;
    }
    let s_send = PackConstPtr(src.as_ptr());
    let slen = src.len();
    let o_send = PackMutPtr(out.as_mut_ptr());
    let olen = out.len();
    let rows_per = (rows.div_ceil(bands).div_ceil(8) * 8).max(8);
    rayon::scope(|s| {
        let mut r0 = 0;
        while r0 < rows {
            let r1 = (r0 + rows_per).min(rows);
            s.spawn(move |_| {
                let s_send: PackConstPtr<T> = s_send;
                let o_send: PackMutPtr<T> = o_send;
                // SAFETY: band owns `out` rows `[r0, r1)`; source
                // read-only. Bounds per the caller contract.
                unsafe {
                    let src_slice = core::slice::from_raw_parts(s_send.0, slen);
                    let out_slice = core::slice::from_raw_parts_mut(o_send.0, olen);
                    pack_transpose_blocked_range(src_slice, ld, cols, out_slice, r0, r1);
                }
            });
            r0 = r1;
        }
    });
}

fn pack_a_c32(
    a: &[Complex32],
    lda: usize,
    m: usize,
    k: usize,
    op: Transpose,
    out: &mut [Complex32],
) {
    // NB: ConjTrans conjugates values (not just addresses), which the
    // index driver cannot express — it stays on the serial loop below.
    if m.saturating_mul(k) >= PACK_PAR_MIN_ELS && !matches!(op, Transpose::ConjTrans) {
        match op {
            Transpose::NoTrans => pack_parallel(m, k, a, out, |i, p| i * lda + p),
            Transpose::Trans => pack_transpose_parallel(a, lda, m, k, out),
            Transpose::ConjTrans => unreachable!(),
        }
        return;
    }
    if matches!(op, Transpose::Trans) {
        pack_transpose_blocked(a, lda, m, k, out);
        return;
    }
    for i in 0..m {
        for p in 0..k {
            out[i * k + p] = cget_c32(a, lda, i, p, op);
        }
    }
}

fn pack_a_c64(
    a: &[Complex64],
    lda: usize,
    m: usize,
    k: usize,
    op: Transpose,
    out: &mut [Complex64],
) {
    if m.saturating_mul(k) >= PACK_PAR_MIN_ELS && !matches!(op, Transpose::ConjTrans) {
        match op {
            Transpose::NoTrans => pack_parallel(m, k, a, out, |i, p| i * lda + p),
            Transpose::Trans => pack_transpose_parallel(a, lda, m, k, out),
            Transpose::ConjTrans => unreachable!(),
        }
        return;
    }
    if matches!(op, Transpose::Trans) {
        pack_transpose_blocked(a, lda, m, k, out);
        return;
    }
    for i in 0..m {
        for p in 0..k {
            out[i * k + p] = cget_c64(a, lda, i, p, op);
        }
    }
}

pub fn gemm_f32(
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if !matches!(transb, Transpose::NoTrans) && crate::cpu::blas3::avx2::is_available() {
        // SAFETY: AVX2+FMA detected; the `Trans` source layout is part of
        // the caller contract (same as the pre-pack this replaces — it
        // never validated the unpacked source either). The kernel gathers
        // `Trans` tiles/panels itself — no pre-pack, so a transposed band
        // GEMM pays one strided tile read per task instead of a
        // full-matrix transpose-pack plus a second read.
        unsafe {
            crate::cpu::blas3::avx2::sgemm(
                m,
                n,
                k,
                alpha,
                a,
                lda,
                b,
                ldb,
                beta,
                c,
                ldc,
                !matches!(transa, Transpose::NoTrans),
                true,
            );
        }
        return;
    }
    // Without SME2, skewed NoTrans shapes the NN wrapper below would
    // reject degrade to the scalar triple loop; Accelerate takes them.
    // Transposed shapes stay in-house: the NEON fallback accumulates
    // each element in sequential k order, and grouped TRSM/SYRK
    // comparisons rely on that shape independence.
    #[cfg(target_os = "macos")]
    if !sme2_available()
        && matches!(transa, Transpose::NoTrans)
        && matches!(transb, Transpose::NoTrans)
        && accelerate::sgemm_t(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    {
        return;
    }
    if matches!(transb, Transpose::NoTrans) {
        #[cfg(target_os = "macos")]
        if matches!(transa, Transpose::NoTrans)
            && accelerate::sgemm(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
        {
            return;
        }
        let trans_a = !matches!(transa, Transpose::NoTrans);
        sme2::gemm::sgemm(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_a);
        return;
    }
    with_buf(&F32_B_PACK, k * n, |bp| {
        pack_b_f32(b, ldb, n, k, transb, bp);
        let trans_a = !matches!(transa, Transpose::NoTrans);
        sme2::gemm::sgemm(m, n, k, alpha, a, lda, bp, n, beta, c, ldc, trans_a);
    });
}

/// Serial [`gemm_f32`] (no `rayon::scope` anywhere in the call tree):
/// for callers that already fan out over threads (SYRK bands) or overlap
/// serial work with a parallel GEMM (LU lookahead). Off x86 (or without
/// AVX2) it is the regular entry — same as today. Results are bitwise
/// identical to [`gemm_f32`]: same packs, same sweep order per element.
///
/// Bounds follow the [`gemm_f32`] contract.
pub fn gemm_f32_serial(
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::blas3::avx2::is_available() {
        // SAFETY: AVX2+FMA detected; bounds per the caller contract. The
        // kernel gathers `Trans` tiles/panels itself — no pre-pack.
        unsafe {
            let trans_a = !matches!(transa, Transpose::NoTrans);
            let trans_b = !matches!(transb, Transpose::NoTrans);
            crate::cpu::blas3::avx2::sgemm_serial(
                m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_a, trans_b,
            );
        }
        return;
    }
    gemm_f32(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

pub fn gemm_f64(
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    beta: f64,
    c: &mut [f64],
    ldc: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if !matches!(transb, Transpose::NoTrans) && crate::cpu::blas3::avx2::is_available() {
        // SAFETY: AVX2+FMA detected; the `Trans` source layout is part of
        // the caller contract (same as the pre-pack this replaces — it
        // never validated the unpacked source either). The kernel gathers
        // `Trans` tiles/panels itself — no pre-pack, so a transposed band
        // GEMM pays one strided tile read per task instead of a
        // full-matrix transpose-pack plus a second read.
        unsafe {
            crate::cpu::blas3::avx2::dgemm(
                m,
                n,
                k,
                alpha,
                a,
                lda,
                b,
                ldb,
                beta,
                c,
                ldc,
                !matches!(transa, Transpose::NoTrans),
                true,
            );
        }
        return;
    }
    // Without SME2, skewed NoTrans shapes the NN wrapper's skew guard
    // rejects degrade to the scalar triple loop; Accelerate takes
    // them. Transposed shapes stay in-house (see `gemm_f32`).
    #[cfg(target_os = "macos")]
    if !sme2_available()
        && matches!(transa, Transpose::NoTrans)
        && matches!(transb, Transpose::NoTrans)
        && accelerate::dgemm_t(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    {
        return;
    }
    if matches!(transb, Transpose::NoTrans) {
        #[cfg(target_os = "macos")]
        if matches!(transa, Transpose::NoTrans)
            && accelerate::dgemm(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
        {
            return;
        }
        let trans_a = !matches!(transa, Transpose::NoTrans);
        sme2::gemm::dgemm(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_a);
        return;
    }
    with_buf(&F64_B_PACK, k * n, |bp| {
        pack_b_f64(b, ldb, n, k, transb, bp);
        let trans_a = !matches!(transa, Transpose::NoTrans);
        sme2::gemm::dgemm(m, n, k, alpha, a, lda, bp, n, beta, c, ldc, trans_a);
    });
}

/// Serial [`gemm_f64`] (no `rayon::scope` anywhere in the call tree):
/// for callers that already fan out over threads (SYRK bands) or overlap
/// serial work with a parallel GEMM (LU lookahead). Off x86 (or without
/// AVX2) it is the regular entry — same as today. Results are bitwise
/// identical to [`gemm_f64`]: same packs, same sweep order per element.
///
/// Bounds follow the [`gemm_f64`] contract.
pub fn gemm_f64_serial(
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    beta: f64,
    c: &mut [f64],
    ldc: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::blas3::avx2::is_available() {
        // SAFETY: AVX2+FMA detected; bounds per the caller contract. The
        // kernel gathers `Trans` tiles/panels itself — no pre-pack.
        unsafe {
            let trans_a = !matches!(transa, Transpose::NoTrans);
            let trans_b = !matches!(transb, Transpose::NoTrans);
            crate::cpu::blas3::avx2::dgemm_serial(
                m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_a, trans_b,
            );
        }
        return;
    }
    gemm_f64(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

pub fn gemm_c32(
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: Complex32,
    a: &[Complex32],
    lda: usize,
    b: &[Complex32],
    ldb: usize,
    beta: Complex32,
    c: &mut [Complex32],
    ldc: usize,
) {
    // Without SME2, skewed NoTrans shapes fall to the AVX2-tuned
    // native sweep running scalar on aarch64 (~4x slower than cblas at
    // the skewed BEM shape); Accelerate takes them. Transposed shapes
    // stay on the native sweep, whose per-element k order is
    // shape-independent.
    #[cfg(target_os = "macos")]
    if !sme2_available()
        && matches!(transa, Transpose::NoTrans)
        && matches!(transb, Transpose::NoTrans)
        && accelerate::cgemm_t(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    {
        return;
    }
    if matches!(transa, Transpose::NoTrans) && matches!(transb, Transpose::NoTrans) {
        #[cfg(target_os = "macos")]
        if accelerate::cgemm(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) {
            return;
        }
        // Native thin-k path first (Schur accumulate / overwrite pairs);
        // falls through to 4M for general shapes and scalars.
        if complex_think::cgemm_think(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) {
            return;
        }
        // Native panel-pack path (general alpha/beta, k-chunked).
        if complex_native32::cgemm_native(
            m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, false, false,
        ) {
            return;
        }
        complex3m::gemm_c32_nn(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        return;
    }
    // Transposed-B runs the native sweep directly (see `gemm_c64`).
    let trans_b = !matches!(transb, Transpose::NoTrans);
    let conj_b = matches!(transb, Transpose::ConjTrans);
    if matches!(transa, Transpose::NoTrans) {
        complex_native32::cgemm_native(
            m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_b, conj_b,
        );
        return;
    }
    with_buf(&C32_A_PACK, m * k, |ap| {
        pack_a_c32(a, lda, m, k, transa, ap);
        complex_native32::cgemm_native(
            m, n, k, alpha, ap, k, b, ldb, beta, c, ldc, trans_b, conj_b,
        );
    });
}

pub fn gemm_c64(
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: Complex64,
    a: &[Complex64],
    lda: usize,
    b: &[Complex64],
    ldb: usize,
    beta: Complex64,
    c: &mut [Complex64],
    ldc: usize,
) {
    // See `gemm_c32`: without SME2, skewed NoTrans shapes route
    // through Accelerate; transposed shapes stay on the native sweep.
    #[cfg(target_os = "macos")]
    if !sme2_available()
        && matches!(transa, Transpose::NoTrans)
        && matches!(transb, Transpose::NoTrans)
        && accelerate::zgemm_t(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    {
        return;
    }
    if matches!(transa, Transpose::NoTrans) && matches!(transb, Transpose::NoTrans) {
        #[cfg(target_os = "macos")]
        if accelerate::zgemm(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) {
            return;
        }
        // Native thin-k path first (Schur accumulate / overwrite pairs);
        // falls through to 4M for general shapes and scalars.
        if complex_think::zgemm_think(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) {
            return;
        }
        // Native panel-pack path (overwrite only, full-k for now).
        if complex_native::zgemm_native(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, false, false)
        {
            return;
        }
        complex3m::gemm_c64_nn(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        return;
    }
    // Transposed-B runs the native sweep directly: each task packs the
    // B-blocks it consumes (fused, worker-local), so no whole-matrix
    // pre-pack crosses the interconnect (the old pre-pack cost ~200us of
    // strided reads plus scattered re-reads per SYRK-panel call). Only a
    // transposed A still packs (the sweep reads A directly); NoTrans
    // operands are never copied. Native always handles these calls.
    let trans_b = !matches!(transb, Transpose::NoTrans);
    let conj_b = matches!(transb, Transpose::ConjTrans);
    if matches!(transa, Transpose::NoTrans) {
        complex_native::zgemm_native(
            m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_b, conj_b,
        );
        return;
    }
    with_buf(&C64_A_PACK, m * k, |ap| {
        pack_a_c64(a, lda, m, k, transa, ap);
        complex_native::zgemm_native(m, n, k, alpha, ap, k, b, ldb, beta, c, ldc, trans_b, conj_b);
    });
}

/// Serial [`gemm_c64`] (no `rayon::scope` anywhere in the call tree):
/// the native sweep run single-threaded, for callers that already fan
/// out over threads or overlap serial work with a parallel GEMM (LU
/// lookahead serial block update). A transposed `A` packs first; a
/// transposed `B` is consumed by the sweep (same as `gemm_c64`).
/// Same per-element op order as the parallel sweep.
/// Bounds follow the [`gemm_c64`] contract.
pub fn gemm_c64_serial(
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: Complex64,
    a: &[Complex64],
    lda: usize,
    b: &[Complex64],
    ldb: usize,
    beta: Complex64,
    c: &mut [Complex64],
    ldc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        for row in c.chunks_mut(ldc).take(m) {
            for slot in row.iter_mut().take(n) {
                *slot = beta * *slot;
            }
        }
        return;
    }
    if matches!(transa, Transpose::NoTrans) && matches!(transb, Transpose::NoTrans) {
        complex_native::sweep_serial_c64(
            m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, false, false,
        );
        return;
    }
    // Same no-B-pack structure as `gemm_c64` (serial sweep).
    let trans_b = !matches!(transb, Transpose::NoTrans);
    let conj_b = matches!(transb, Transpose::ConjTrans);
    if matches!(transa, Transpose::NoTrans) {
        complex_native::sweep_serial_c64(
            m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_b, conj_b,
        );
        return;
    }
    with_buf(&C64_A_PACK, m * k, |ap| {
        pack_a_c64(a, lda, m, k, transa, ap);
        complex_native::sweep_serial_c64(
            m, n, k, alpha, ap, k, b, ldb, beta, c, ldc, trans_b, conj_b,
        );
    });
}

/// Serial [`gemm_c32`]: `f32` twin of [`gemm_c64_serial`].
pub fn gemm_c32_serial(
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: Complex32,
    a: &[Complex32],
    lda: usize,
    b: &[Complex32],
    ldb: usize,
    beta: Complex32,
    c: &mut [Complex32],
    ldc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        for row in c.chunks_mut(ldc).take(m) {
            for slot in row.iter_mut().take(n) {
                *slot = beta * *slot;
            }
        }
        return;
    }
    if matches!(transa, Transpose::NoTrans) && matches!(transb, Transpose::NoTrans) {
        complex_native32::sweep_serial_c32(
            m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, false, false,
        );
        return;
    }
    let trans_b = !matches!(transb, Transpose::NoTrans);
    let conj_b = matches!(transb, Transpose::ConjTrans);
    if matches!(transa, Transpose::NoTrans) {
        complex_native32::sweep_serial_c32(
            m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_b, conj_b,
        );
        return;
    }
    with_buf(&C32_A_PACK, m * k, |ap| {
        pack_a_c32(a, lda, m, k, transa, ap);
        complex_native32::sweep_serial_c32(
            m, n, k, alpha, ap, k, b, ldb, beta, c, ldc, trans_b, conj_b,
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rayon::prelude::*;

    /// Parallel-pack parity: packs above `PACK_PAR_MIN_ELS` must match the
    /// serial pack elementwise for every op and all six pack routines
    /// (real A/B, complex A/B with conjugation). The rayon row-band split
    /// only partitions the output, so this guards the dispatch and the
    /// index closures against mixups.
    #[test]
    fn pack_transpose_blocked_matches_naive() {
        // Blocked serial + parallel transpose-pack vs the naive loop:
        // bitwise identical (pure copies). Odd shapes exercise tails;
        // the 1024x64 band shape covers the SYRK-panel regime.
        // (rows, cols, ld): rows <= ld (p indexes src columns); the
        // 64x1024 band shape covers the SYRK-panel regime.
        for (rows, cols, ld) in [
            (64usize, 1024usize, 64usize),
            (300, 300, 307),
            (48, 512, 64),
            (7, 13, 16),
        ] {
            let src: Vec<u64> = (0..cols * ld)
                .map(|i| {
                    (i as u64)
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407)
                })
                .collect();
            let mut expect = vec![0u64; rows * cols];
            for p in 0..rows {
                for j in 0..cols {
                    expect[p * cols + j] = src[j * ld + p];
                }
            }
            let mut got = vec![0u64; rows * cols];
            pack_transpose_blocked(&src, ld, rows, cols, &mut got);
            assert_eq!(got, expect, "blocked rows={rows} cols={cols}");
            let mut gotp = vec![0u64; rows * cols];
            pack_transpose_parallel(&src, ld, rows, cols, &mut gotp);
            assert_eq!(gotp, expect, "parallel rows={rows} cols={cols}");
        }
    }

    #[test]
    fn pack_parallel_matches_serial() {
        // 300 × 300 = 90k ≥ gate; odd strides exercise tails.
        let (rows, cols, ld) = (300usize, 300usize, 307usize);
        let r32: Vec<f32> = (0..rows * ld).map(|i| (i as f32 * 0.013).sin()).collect();
        let r64: Vec<f64> = (0..rows * ld).map(|i| (i as f64 * 0.013).sin()).collect();
        let c32: Vec<Complex32> = (0..rows * ld)
            .map(|i| Complex32::new((i as f32 * 0.013).sin(), (i as f32 * 0.007).cos()))
            .collect();
        let c64: Vec<Complex64> = (0..rows * ld)
            .map(|i| Complex64::new((i as f64 * 0.013).sin(), (i as f64 * 0.007).cos()))
            .collect();
        for op in [Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
            let mut o32 = vec![0.0f32; rows * cols];
            pack_b_f32(&r32, ld, cols, rows, op, &mut o32);
            let mut o64 = vec![0.0f64; rows * cols];
            pack_b_f64(&r64, ld, cols, rows, op, &mut o64);
            let mut oa32 = vec![Complex32::new(0.0, 0.0); rows * cols];
            pack_a_c32(&c32, ld, rows, cols, op, &mut oa32);
            let mut oa64 = vec![Complex64::new(0.0, 0.0); rows * cols];
            pack_a_c64(&c64, ld, rows, cols, op, &mut oa64);
            for p in 0..rows {
                for j in 0..cols {
                    assert_relative_eq!(
                        o32[p * cols + j],
                        bget_f32(&r32, ld, p, j, op),
                        epsilon = 0.0
                    );
                    assert_relative_eq!(
                        o64[p * cols + j],
                        bget_f64(&r64, ld, p, j, op),
                        epsilon = 0.0
                    );
                    assert_eq!(oa32[p * cols + j], cget_c32(&c32, ld, p, j, op));
                    assert_eq!(oa64[p * cols + j], cget_c64(&c64, ld, p, j, op));
                }
            }
        }
    }

    /// Deterministic reentrancy repro: on a single-threaded rayon pool, a
    /// task running inside a `rayon::scope` executes on the same worker
    /// thread that already holds the scratch-pool borrow. Pre-fix the inner
    /// `borrow_mut` panicked with "RefCell already borrowed"; the pool now
    /// falls back to a fresh allocation for the nested call.
    #[test]
    fn scratch_pool_is_reentrant_across_rayon_scope() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("single-thread test pool");

        pool.install(|| {
            with_buf(&F64_B_PACK, 16, |outer| {
                outer.fill(1.0);
                let outer_ptr = outer.as_ptr() as usize;

                rayon::scope(|scope| {
                    scope.spawn(move |_| {
                        with_buf(&F64_B_PACK, 8, |inner| {
                            assert_ne!(
                                inner.as_ptr() as usize,
                                outer_ptr,
                                "nested call must not alias the active outer buffer"
                            );
                            inner.fill(2.0);
                        });
                    });
                });

                assert_eq!(outer, &[1.0; 16]);
            });
        });
    }

    /// Nested-GEMM stress: an outer `par_iter` runs complex GEMMs whose inner
    /// 4M decomposition itself fans out over rayon (m above the SME2
    /// parallel-M threshold). Workers waiting in the inner `rayon::scope`
    /// steal outer iterations onto the same thread, re-entering the
    /// thread-local scratch pools while they are still checked out. Pre-fix
    /// this panicked with "RefCell already borrowed"; the pools now fall back
    /// to a fresh allocation on reentrant calls. Asserts no panic and exact
    /// parity with a top-level call.
    #[test]
    fn nested_gemm_c64_stress_no_refcell_panic() {
        let m = 96;
        let n = 80;
        let k = 72;
        let alpha = Complex64::new(0.7, -0.3);
        let beta = Complex64::new(0.0, 0.0);
        let a: Vec<Complex64> = (0..m * k)
            .map(|i| Complex64::new((i as f64 * 0.013).sin(), (i as f64 * 0.007).cos()))
            .collect();
        let b: Vec<Complex64> = (0..k * n)
            .map(|i| Complex64::new((i as f64 * 0.011).cos(), (i as f64 * 0.017).sin()))
            .collect();

        let mut want = vec![Complex64::new(0.0, 0.0); m * n];
        gemm_c64(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            alpha,
            &a,
            k,
            &b,
            n,
            beta,
            &mut want,
            n,
        );

        // 32 rounds x 64 parallel calls. The width must exceed the thread
        // count so a worker waiting in an inner `rayon::scope` still has
        // outer iterations left to steal onto its own thread — that is what
        // re-enters the thread-local pools mid-borrow.
        for _ in 0..32 {
            let mut outs: Vec<Vec<Complex64>> = (0..64)
                .map(|_| vec![Complex64::new(0.0, 0.0); m * n])
                .collect();
            outs.par_iter_mut().for_each(|c| {
                gemm_c64(
                    Transpose::NoTrans,
                    Transpose::NoTrans,
                    m,
                    n,
                    k,
                    alpha,
                    &a,
                    k,
                    &b,
                    n,
                    beta,
                    c,
                    n,
                );
            });
            for got in &outs {
                for (g, w) in got.iter().zip(want.iter()) {
                    assert_relative_eq!(g.re, w.re, max_relative = 1e-12, epsilon = 1e-12);
                    assert_relative_eq!(g.im, w.im, max_relative = 1e-12, epsilon = 1e-12);
                }
            }
        }
    }

    /// Same stress shape through the real TransB path, which packs B into the
    /// thread-local `F64_B_PACK` scratch before calling the SME2 kernel.
    #[test]
    fn nested_gemm_f64_transb_stress_no_refcell_panic() {
        let m = 96;
        let n = 80;
        let k = 72;
        let a: Vec<f64> = (0..m * k).map(|i| (i as f64 * 0.013).sin()).collect();
        // Stored n x k for TransB.
        let b: Vec<f64> = (0..n * k).map(|i| (i as f64 * 0.017).cos()).collect();

        let mut want = vec![0.0f64; m * n];
        gemm_f64(
            Transpose::NoTrans,
            Transpose::Trans,
            m,
            n,
            k,
            1.0,
            &a,
            k,
            &b,
            k,
            0.0,
            &mut want,
            n,
        );

        for _ in 0..32 {
            let mut outs: Vec<Vec<f64>> = (0..64).map(|_| vec![0.0f64; m * n]).collect();
            outs.par_iter_mut().for_each(|c| {
                gemm_f64(
                    Transpose::NoTrans,
                    Transpose::Trans,
                    m,
                    n,
                    k,
                    1.0,
                    &a,
                    k,
                    &b,
                    k,
                    0.0,
                    c,
                    n,
                );
            });
            for got in &outs {
                for (g, w) in got.iter().zip(want.iter()) {
                    assert_relative_eq!(g, w, max_relative = 1e-12, epsilon = 1e-12);
                }
            }
        }
    }

    /// Numerical parity between the Accelerate `zgemm` dispatch (macOS,
    /// eligible `(NoTrans, NoTrans)` sizes) and the in-house 4M path, at an
    /// odd size with non-trivial alpha/beta.
    #[cfg(target_os = "macos")]
    #[test]
    fn gemm_c64_accelerate_zgemm_matches_4m() {
        let n = 257;
        let alpha = Complex64::new(0.7, -0.3);
        let beta = Complex64::new(0.25, 0.5);
        let a: Vec<Complex64> = (0..n * n)
            .map(|i| Complex64::new((i as f64 * 0.011).sin(), (i as f64 * 0.007).cos()))
            .collect();
        let b: Vec<Complex64> = (0..n * n)
            .map(|i| Complex64::new((i as f64 * 0.005).cos(), (i as f64 * 0.019).sin()))
            .collect();
        let c0: Vec<Complex64> = (0..n * n)
            .map(|i| Complex64::new(0.2 - i as f64 * 1e-6, -0.1 + i as f64 * 2e-6))
            .collect();

        // 257^3 >= 32^3 and square → dispatch routes to Accelerate zgemm.
        let mut got = c0.clone();
        gemm_c64(
            Transpose::NoTrans,
            Transpose::NoTrans,
            n,
            n,
            n,
            alpha,
            &a,
            n,
            &b,
            n,
            beta,
            &mut got,
            n,
        );

        let mut want = c0;
        complex3m::gemm_c64_nn(n, n, n, alpha, &a, n, &b, n, beta, &mut want, n);

        for (g, w) in got.iter().zip(want.iter()) {
            assert_relative_eq!(g.re, w.re, max_relative = 1e-5, epsilon = 1e-8);
            assert_relative_eq!(g.im, w.im, max_relative = 1e-5, epsilon = 1e-8);
        }
    }

    /// Same parity gate for Complex32/cgemm.
    #[cfg(target_os = "macos")]
    #[test]
    fn gemm_c32_accelerate_cgemm_matches_4m() {
        let n = 257;
        let alpha = Complex32::new(0.7, -0.3);
        let beta = Complex32::new(0.25, 0.5);
        let a: Vec<Complex32> = (0..n * n)
            .map(|i| Complex32::new((i as f32 * 0.011).sin(), (i as f32 * 0.007).cos()))
            .collect();
        let b: Vec<Complex32> = (0..n * n)
            .map(|i| Complex32::new((i as f32 * 0.005).cos(), (i as f32 * 0.019).sin()))
            .collect();
        let c0: Vec<Complex32> = (0..n * n)
            .map(|i| Complex32::new(0.2 - i as f32 * 1e-6, -0.1 + i as f32 * 2e-6))
            .collect();

        let mut got = c0.clone();
        gemm_c32(
            Transpose::NoTrans,
            Transpose::NoTrans,
            n,
            n,
            n,
            alpha,
            &a,
            n,
            &b,
            n,
            beta,
            &mut got,
            n,
        );

        let mut want = c0;
        complex3m::gemm_c32_nn(n, n, n, alpha, &a, n, &b, n, beta, &mut want, n);

        for (g, w) in got.iter().zip(want.iter()) {
            assert_relative_eq!(g.re, w.re, max_relative = 1e-3, epsilon = 1e-4);
            assert_relative_eq!(g.im, w.im, max_relative = 1e-3, epsilon = 1e-4);
        }
    }

    #[test]
    fn real_gemm_dispatch_threshold_matches_scalar_with_beta() {
        let size = 32;

        let a32: Vec<f32> = (0..size * size)
            .map(|i| (i as f32 * 0.013 + 0.2).sin())
            .collect();
        let b32: Vec<f32> = (0..size * size)
            .map(|i| (i as f32 * 0.017 + 0.7).cos())
            .collect();
        let initial32: Vec<f32> = (0..size * size).map(|i| i as f32 * 0.001).collect();
        let mut got32 = initial32.clone();
        let mut want32 = initial32;
        gemm_f32(
            Transpose::NoTrans,
            Transpose::NoTrans,
            size,
            size,
            size,
            0.61,
            &a32,
            size,
            &b32,
            size,
            0.37,
            &mut got32,
            size,
        );
        scalar::gemm_f32(
            Transpose::NoTrans,
            Transpose::NoTrans,
            size,
            size,
            size,
            0.61,
            &a32,
            size,
            &b32,
            size,
            0.37,
            &mut want32,
            size,
        );
        for (&got, &want) in got32.iter().zip(&want32) {
            assert_relative_eq!(got, want, max_relative = 3e-4, epsilon = 3e-5);
        }

        let a64: Vec<f64> = (0..size * size)
            .map(|i| (i as f64 * 0.013 + 0.2).sin())
            .collect();
        let b64: Vec<f64> = (0..size * size)
            .map(|i| (i as f64 * 0.017 + 0.7).cos())
            .collect();
        let initial64: Vec<f64> = (0..size * size).map(|i| i as f64 * 0.001).collect();
        let mut got64 = initial64.clone();
        let mut want64 = initial64;
        gemm_f64(
            Transpose::NoTrans,
            Transpose::NoTrans,
            size,
            size,
            size,
            0.61,
            &a64,
            size,
            &b64,
            size,
            0.37,
            &mut got64,
            size,
        );
        scalar::gemm_f64(
            Transpose::NoTrans,
            Transpose::NoTrans,
            size,
            size,
            size,
            0.61,
            &a64,
            size,
            &b64,
            size,
            0.37,
            &mut want64,
            size,
        );
        for (&got, &want) in got64.iter().zip(&want64) {
            assert_relative_eq!(got, want, max_relative = 2e-11, epsilon = 2e-12);
        }
    }

    #[test]
    fn gemm_f64_basic() {
        // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = vec![1.0_f64, 2.0, 3.0, 4.0];
        let b = vec![5.0_f64, 6.0, 7.0, 8.0];
        let mut c = vec![0.0_f64; 4];
        gemm_f64(
            Transpose::NoTrans,
            Transpose::NoTrans,
            2,
            2,
            2,
            1.0,
            &a,
            2,
            &b,
            2,
            0.0,
            &mut c,
            2,
        );
        assert_relative_eq!(c[0], 19.0, max_relative = 1e-12);
        assert_relative_eq!(c[1], 22.0, max_relative = 1e-12);
        assert_relative_eq!(c[2], 43.0, max_relative = 1e-12);
        assert_relative_eq!(c[3], 50.0, max_relative = 1e-12);
    }

    #[test]
    fn gemm_f64_transa() {
        // op(A) = A^T = [[1,3],[2,4]], B = [[5,6],[7,8]] → [[26,30],[38,44]]
        let a = vec![1.0_f64, 2.0, 3.0, 4.0]; // stored 2x2
        let b = vec![5.0_f64, 6.0, 7.0, 8.0];
        let mut c = vec![0.0_f64; 4];
        gemm_f64(
            Transpose::Trans,
            Transpose::NoTrans,
            2,
            2,
            2,
            1.0,
            &a,
            2,
            &b,
            2,
            0.0,
            &mut c,
            2,
        );
        assert_relative_eq!(c[0], 26.0, max_relative = 1e-12);
        assert_relative_eq!(c[1], 30.0, max_relative = 1e-12);
        assert_relative_eq!(c[2], 38.0, max_relative = 1e-12);
        assert_relative_eq!(c[3], 44.0, max_relative = 1e-12);
    }

    #[test]
    fn gemm_f32_transb_uses_fast_path_and_matches_scalar() {
        let m = 3;
        let n = 4;
        let k = 5;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.13).sin()).collect();
        let b: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.17).cos()).collect();
        let c0: Vec<f32> = (0..m * n).map(|i| i as f32 * 0.01).collect();
        let mut got = c0.clone();
        let mut want = c0;
        gemm_f32(
            Transpose::NoTrans,
            Transpose::Trans,
            m,
            n,
            k,
            0.7,
            &a,
            k,
            &b,
            k,
            0.25,
            &mut got,
            n,
        );
        scalar::gemm_f32(
            Transpose::NoTrans,
            Transpose::Trans,
            m,
            n,
            k,
            0.7,
            &a,
            k,
            &b,
            k,
            0.25,
            &mut want,
            n,
        );
        for (g, w) in got.iter().zip(want.iter()) {
            assert_relative_eq!(g, w, max_relative = 1e-5, epsilon = 1e-6);
        }
    }

    #[test]
    fn gemm_f64_transb_uses_fast_path_and_matches_scalar() {
        let m = 3;
        let n = 4;
        let k = 5;
        let a: Vec<f64> = (0..m * k).map(|i| (i as f64 * 0.13).sin()).collect();
        let b: Vec<f64> = (0..n * k).map(|i| (i as f64 * 0.17).cos()).collect();
        let c0: Vec<f64> = (0..m * n).map(|i| i as f64 * 0.01).collect();
        let mut got = c0.clone();
        let mut want = c0;
        gemm_f64(
            Transpose::ConjTrans,
            Transpose::Trans,
            m,
            n,
            k,
            0.7,
            &a,
            m,
            &b,
            k,
            0.25,
            &mut got,
            n,
        );
        scalar::gemm_f64(
            Transpose::ConjTrans,
            Transpose::Trans,
            m,
            n,
            k,
            0.7,
            &a,
            m,
            &b,
            k,
            0.25,
            &mut want,
            n,
        );
        for (g, w) in got.iter().zip(want.iter()) {
            assert_relative_eq!(g, w, max_relative = 1e-12);
        }
    }

    #[test]
    fn small_pool_routing_matches_scalar() {
        // `scope_for_macs` routes sub-ceiling scopes to the capped pool;
        // the worker count must never change the numbers. These shapes
        // clear the parallel threshold but stay under the small-pool
        // ceiling, so they run the N-split on the small pool on wide
        // machines and on the global pool elsewhere — same result either
        // way (checked against the scalar nest).
        for (m, n, k) in [(256usize, 256, 128), (512, 128, 96)] {
            let a: Vec<f64> = (0..m * k).map(|i| (i as f64 * 0.013).sin()).collect();
            let b: Vec<f64> = (0..k * n).map(|i| (i as f64 * 0.017).cos()).collect();
            let c0: Vec<f64> = (0..m * n).map(|i| i as f64 * 0.001).collect();
            for transb in [Transpose::NoTrans, Transpose::Trans] {
                // `b` holds `k * n == n * k` elements; only the leading
                // dimension changes with the transpose flag.
                let ldb = match transb {
                    Transpose::NoTrans => n,
                    _ => k,
                };
                let mut got = c0.clone();
                gemm_f64(
                    Transpose::NoTrans,
                    transb,
                    m,
                    n,
                    k,
                    0.61,
                    &a,
                    k,
                    &b,
                    ldb,
                    0.37,
                    &mut got,
                    n,
                );
                let mut want = c0.clone();
                scalar::gemm_f64(
                    Transpose::NoTrans,
                    transb,
                    m,
                    n,
                    k,
                    0.61,
                    &a,
                    k,
                    &b,
                    ldb,
                    0.37,
                    &mut want,
                    n,
                );
                for (g, w) in got.iter().zip(want.iter()) {
                    assert_relative_eq!(g, w, max_relative = 2e-11, epsilon = 2e-12);
                }
            }
            let a32: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.013).sin()).collect();
            let b32: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.017).cos()).collect();
            let c032: Vec<f32> = (0..m * n).map(|i| i as f32 * 0.001).collect();
            let mut got32 = c032.clone();
            gemm_f32(
                Transpose::NoTrans,
                Transpose::NoTrans,
                m,
                n,
                k,
                0.61,
                &a32,
                k,
                &b32,
                n,
                0.37,
                &mut got32,
                n,
            );
            let mut want32 = c032;
            scalar::gemm_f32(
                Transpose::NoTrans,
                Transpose::NoTrans,
                m,
                n,
                k,
                0.61,
                &a32,
                k,
                &b32,
                n,
                0.37,
                &mut want32,
                n,
            );
            for (g, w) in got32.iter().zip(&want32) {
                assert_relative_eq!(g, w, max_relative = 3e-4, epsilon = 3e-5);
            }
        }
    }

    #[test]
    fn scope_pinned_computes_and_nests() {
        // The scope body runs (exact values) and a scope opened on a pool
        // worker runs inline on the enclosing pool instead of deadlocking
        // or migrating: 0+..+7 = 28 either way.
        let total = &std::sync::Mutex::new(0u64);
        scope_pinned(|s| {
            for chunk in 0..8u64 {
                s.spawn(move |_| {
                    scope_pinned(|inner| {
                        inner.spawn(|_| {
                            *total.lock().unwrap() += chunk;
                        });
                    });
                });
            }
        });
        assert_eq!(*total.lock().unwrap(), 28);
    }

    #[test]
    fn run_on_worker_returns_nests_and_propagates_panic() {
        // Value identity through the funnel.
        assert_eq!(run_on_worker(|| 6 * 7), 42);
        // Nested (already on a worker): inline, no deadlock.
        scope_pinned(|s| {
            s.spawn(|_| assert_eq!(run_on_worker(|| 1 + 1), 2));
        });
        // Worker panics resume on the caller.
        let caught = std::panic::catch_unwind(|| run_on_worker(|| panic!("probe")));
        assert!(caught.is_err());
    }

    #[test]
    fn run_mirror_returns_nests_and_propagates_panic() {
        // Install-funnel twin of the spawn test above: value identity,
        // nested-inline (no deadlock), panics resume on the caller.
        assert_eq!(run_mirror(|| 6 * 7), 42);
        scope_pinned(|s| {
            s.spawn(|_| assert_eq!(run_mirror(|| 1 + 1), 2));
        });
        let caught = std::panic::catch_unwind(|| run_mirror(|| panic!("probe")));
        assert!(caught.is_err());
        // Same pool as the spawn funnel: identical width on both.
        let via_install = run_mirror(|| rayon::current_num_threads().max(1));
        let via_spawn = run_on_worker(|| rayon::current_num_threads().max(1));
        assert_eq!(via_install, via_spawn);
        // Borrows instead of spawning: no 'static bound, so a
        // stack borrow runs through the funnel.
        let lent = [1u32, 2, 3];
        assert_eq!(run_mirror(|| lent.iter().sum::<u32>()), 6);
    }

    #[test]
    fn broadcast_bands_covers_exactly_once() {
        // Every band runs exactly once however bands split across
        // workers (fewer, equal, and more bands than workers).
        use std::sync::atomic::{AtomicUsize, Ordering};
        for nbands in [1usize, 3, 16, 17, 100] {
            let hits: Vec<AtomicUsize> = (0..nbands).map(|_| AtomicUsize::new(0)).collect();
            broadcast_bands(nbands, &|b: usize| {
                hits[b].fetch_add(1, Ordering::Relaxed);
            });
            for (b, hit) in hits.iter().enumerate() {
                assert_eq!(hit.load(Ordering::Relaxed), 1, "band {b} of {nbands}");
            }
        }
    }

    #[test]
    fn broadcast_bands_zero_is_noop() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = AtomicUsize::new(0);
        broadcast_bands(0, &|_: usize| {
            calls.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn broadcast_bands_panic_resumes_on_caller() {
        // A panicking band aborts the broadcast with the worker's
        // payload once every worker finished; the pool stays usable.
        use std::panic::{AssertUnwindSafe, catch_unwind};
        use std::sync::atomic::{AtomicUsize, Ordering};
        let result = catch_unwind(AssertUnwindSafe(|| {
            broadcast_bands(32, &|b: usize| {
                if b == 0 {
                    panic!("intentional band panic");
                }
            });
        }));
        assert!(result.is_err(), "caller must observe the panic");
        let calls = AtomicUsize::new(0);
        broadcast_bands(32, &|_: usize| {
            calls.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(calls.load(Ordering::Relaxed), 32);
    }

    #[test]
    fn broadcast_bands_from_spin_worker_takes_fallback() {
        // The nesting guard routes spin-worker callers to the scope
        // fallback instead of deadlocking on their own pool.
        use std::sync::atomic::{AtomicUsize, Ordering};
        let Some(pool) = super::super::spin_pool::spin_pool() else {
            return;
        };
        let calls = AtomicUsize::new(0);
        let calls_ref = &calls;
        pool.broadcast(&|_: usize| {
            assert!(super::super::spin_pool::SpinPool::on_worker());
            broadcast_bands(16, &|_: usize| {
                calls_ref.fetch_add(1, Ordering::Relaxed);
            });
        });
        assert_eq!(
            calls.load(Ordering::Relaxed),
            16 * pool.width(),
            "every worker's fallback broadcast runs all bands"
        );
    }

    #[test]
    fn worker_affinity_pins_only_the_spawned_thread() {
        // Pinning runs on a dedicated thread the test owns, so the shared
        // test-runner threads are never restricted. Either affinity is
        // active and the mask is exactly the predicted singleton, or it
        // is disabled/unavailable and the pin is a verified no-op.
        let mask = std::thread::spawn(|| {
            crate::cpu::affinity::pin_worker(0);
            crate::cpu::affinity::current_mask()
        })
        .join()
        .expect("affinity probe thread");
        match crate::cpu::affinity::pinned_cpu(0) {
            Some(cpu) => assert_eq!(mask, vec![cpu]),
            None => {}
        }
    }

    /// `align64_slice` lends a 64B-aligned `need`-slice that grows
    /// the buffer, preserves existing content, and wastes at most 63
    /// bytes; the spare-ptr twin aligns into uninitialised capacity.
    /// Holds the crate env lock (the helper reads the kill-switch).
    #[test]
    fn align64_slice_properties() {
        let _guard = crate::cpu::parallelism::env_test_lock();
        // SAFETY: the kill-switch is ours alone in this test; the lock
        // serialises env flips process-wide.
        unsafe { std::env::remove_var("SONIUM_GEMM_ALIGN64") };
        for need in [1usize, 7, 16, 100, 1000, 65536] {
            let mut buf = vec![7.0f32; 13];
            let ptr = {
                let dst = align64_slice(&mut buf, need);
                assert_eq!(dst.len(), need);
                dst.as_ptr() as usize
            };
            assert_eq!(ptr % 64, 0, "need={need}");
            assert!(buf.len() >= need && buf.len() <= need + 16);
            assert!(buf[..13.min(buf.len())].iter().all(|&x| x == 7.0));
            let mut spare: Vec<f32> = Vec::new();
            let p = align64_spare_ptr(&mut spare, need);
            assert_eq!((p as usize) % 64, 0, "spare need={need}");
            assert!(spare.capacity() >= need);
        }
        // SAFETY: restoring default (unset).
        unsafe { std::env::remove_var("SONIUM_GEMM_ALIGN64") };
    }

    #[test]
    fn scope_for_macs_deep_nesting_terminates() {
        // Nested scopes must not consume one pool thread per level (pool
        // starvation would deadlock past `SMALL_THREADS` depth): a scope
        // opened while its thread already serves the small pool runs on
        // the current pool instead of nesting installs.
        fn nest(depth: usize) {
            if depth == 0 {
                return;
            }
            scope_for_macs(1, |_| nest(depth - 1));
        }
        SMALL_POOL.install(|| nest(SMALL_THREADS + 8));
        assert!(!IN_SMALL_POOL.with(|c| c.get()));
    }

    #[test]
    fn run_compact_returns_value_on_both_paths() {
        // est = 0 never applies (mirror path); u64::MAX exceeds any L3
        // (fallback unless the box has exabytes of L3). Both compute.
        assert_eq!(run_compact(0, false, || 42u32), 42);
        assert_eq!(run_compact(u64::MAX, true, || 7u32), 7);
        // Nested calls run inline instead of deadlocking.
        assert_eq!(run_on_worker(|| run_compact(1 << 20, false, || 9u32)), 9);
    }

    #[test]
    fn run_compact_uses_compact_pool_when_it_applies() {
        let est = 1u64 << 20; // 1 MiB fits any real L3
        let threads = run_compact(est, false, || rayon::current_num_threads().max(1));
        if compact_applies(est, false) {
            let len = crate::cpu::affinity::compact_cpus()
                .map(|c| c.len())
                .unwrap_or(0);
            assert!(len > 0);
            assert_eq!(threads, len);
        } else {
            assert_eq!(threads, rayon::current_num_threads().max(1));
        }
    }

    #[test]
    fn run_compact_wide_uses_wide_pool_when_built() {
        let est = 1u64 << 20; // 1 MiB fits any real L3
        let threads = run_compact(est, true, || rayon::current_num_threads().max(1));
        if compact_applies(est, true) {
            // Wide pool when built, else the narrow pool (see
            // `compact_pool`); either way a compact width, and on a
            // multi-domain box exactly the wide set.
            let wide_len = crate::cpu::affinity::compact_cpus_wide().map(|c| c.len());
            let narrow_len = crate::cpu::affinity::compact_cpus()
                .map(|c| c.len())
                .unwrap_or(0);
            match wide_len {
                Some(len) => assert_eq!(threads, len),
                None => assert_eq!(threads, narrow_len.max(1)),
            }
        } else {
            assert_eq!(threads, rayon::current_num_threads().max(1));
        }
    }

    #[test]
    fn gemm_c64_conjtrans() {
        // A = [[1+i, 2], [3, 4-i]], op = C
        // A^H = [[1-i, 3], [2, 4+i]]
        // B = I → C = A^H
        let a = vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, -1.0),
        ];
        let b = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let mut c = vec![Complex64::new(0.0, 0.0); 4];
        gemm_c64(
            Transpose::ConjTrans,
            Transpose::NoTrans,
            2,
            2,
            2,
            Complex64::new(1.0, 0.0),
            &a,
            2,
            &b,
            2,
            Complex64::new(0.0, 0.0),
            &mut c,
            2,
        );
        assert_relative_eq!(c[0].re, 1.0, max_relative = 1e-12);
        assert_relative_eq!(c[0].im, -1.0, max_relative = 1e-12);
        assert_relative_eq!(c[1].re, 3.0, max_relative = 1e-12);
        assert_relative_eq!(c[2].re, 2.0, max_relative = 1e-12);
        assert_relative_eq!(c[3].re, 4.0, max_relative = 1e-12);
        assert_relative_eq!(c[3].im, 1.0, max_relative = 1e-12);
    }

    #[test]
    fn gemm_c32_transposed_operands_use_4m_and_match_scalar() {
        let m = 3;
        let n = 4;
        let k = 5;
        let a: Vec<Complex32> = (0..k * m)
            .map(|i| Complex32::new((i as f32 * 0.11).sin(), (i as f32 * 0.07).cos()))
            .collect();
        let b: Vec<Complex32> = (0..n * k)
            .map(|i| Complex32::new((i as f32 * 0.05).cos(), (i as f32 * 0.19).sin()))
            .collect();
        let c0 = vec![Complex32::new(0.2, -0.1); m * n];
        let mut got = c0.clone();
        let mut want = c0;
        let alpha = Complex32::new(0.7, -0.3);
        let beta = Complex32::new(0.25, 0.5);
        gemm_c32(
            Transpose::ConjTrans,
            Transpose::Trans,
            m,
            n,
            k,
            alpha,
            &a,
            m,
            &b,
            k,
            beta,
            &mut got,
            n,
        );
        scalar::gemm_c32(
            Transpose::ConjTrans,
            Transpose::Trans,
            m,
            n,
            k,
            alpha,
            &a,
            m,
            &b,
            k,
            beta,
            &mut want,
            n,
        );
        for (g, w) in got.iter().zip(want.iter()) {
            assert_relative_eq!(g.re, w.re, max_relative = 1e-4, epsilon = 1e-5);
            assert_relative_eq!(g.im, w.im, max_relative = 1e-4, epsilon = 1e-5);
        }
    }

    #[test]
    fn gemm_c64_transposed_operands_use_4m_and_match_scalar() {
        let m = 3;
        let n = 4;
        let k = 5;
        let a: Vec<Complex64> = (0..k * m)
            .map(|i| Complex64::new((i as f64 * 0.11).sin(), (i as f64 * 0.07).cos()))
            .collect();
        let b: Vec<Complex64> = (0..n * k)
            .map(|i| Complex64::new((i as f64 * 0.05).cos(), (i as f64 * 0.19).sin()))
            .collect();
        let c0 = vec![Complex64::new(0.2, -0.1); m * n];
        let mut got = c0.clone();
        let mut want = c0;
        let alpha = Complex64::new(0.7, -0.3);
        let beta = Complex64::new(0.25, 0.5);
        gemm_c64(
            Transpose::ConjTrans,
            Transpose::Trans,
            m,
            n,
            k,
            alpha,
            &a,
            m,
            &b,
            k,
            beta,
            &mut got,
            n,
        );
        scalar::gemm_c64(
            Transpose::ConjTrans,
            Transpose::Trans,
            m,
            n,
            k,
            alpha,
            &a,
            m,
            &b,
            k,
            beta,
            &mut want,
            n,
        );
        for (g, w) in got.iter().zip(want.iter()) {
            assert_relative_eq!(g.re, w.re, max_relative = 1e-12);
            assert_relative_eq!(g.im, w.im, max_relative = 1e-12);
        }
    }

    #[test]
    fn complex3m_c64_all_transpose_combinations_match_scalar() {
        let cases = [
            (
                3usize,
                5usize,
                4usize,
                Transpose::NoTrans,
                Transpose::NoTrans,
            ),
            (4, 3, 5, Transpose::NoTrans, Transpose::Trans),
            (5, 4, 3, Transpose::Trans, Transpose::NoTrans),
            (3, 4, 5, Transpose::Trans, Transpose::ConjTrans),
            (4, 5, 3, Transpose::ConjTrans, Transpose::Trans),
            (5, 3, 4, Transpose::ConjTrans, Transpose::ConjTrans),
        ];
        for (case, &(m, n, k, transa, transb)) in cases.iter().enumerate() {
            let (a_rows, a_cols) = match transa {
                Transpose::NoTrans => (m, k),
                Transpose::Trans | Transpose::ConjTrans => (k, m),
            };
            let (b_rows, b_cols) = match transb {
                Transpose::NoTrans => (k, n),
                Transpose::Trans | Transpose::ConjTrans => (n, k),
            };
            let lda = a_cols + 1;
            let ldb = b_cols + 2;
            let ldc = n + 1;
            let a: Vec<Complex64> = (0..a_rows * lda)
                .map(|index| {
                    Complex64::new(
                        (index as f64 * 0.17 + case as f64).sin(),
                        (index as f64 * 0.11 - case as f64).cos(),
                    )
                })
                .collect();
            let b: Vec<Complex64> = (0..b_rows * ldb)
                .map(|index| {
                    Complex64::new(
                        (index as f64 * 0.13 - 0.4).cos(),
                        (index as f64 * 0.07 + 0.2).sin(),
                    )
                })
                .collect();
            let initial: Vec<Complex64> = (0..m * ldc)
                .map(|index| Complex64::new(0.2 + index as f64 * 0.01, -0.3))
                .collect();
            let alpha = Complex64::new(0.7, -0.25);
            let beta = Complex64::new(-0.2, 0.4);
            let mut four_m = initial.clone();
            let mut scalar_result = initial;
            gemm_c64(
                transa,
                transb,
                m,
                n,
                k,
                alpha,
                &a,
                lda,
                &b,
                ldb,
                beta,
                &mut four_m,
                ldc,
            );
            scalar::gemm_c64(
                transa,
                transb,
                m,
                n,
                k,
                alpha,
                &a,
                lda,
                &b,
                ldb,
                beta,
                &mut scalar_result,
                ldc,
            );
            for row in 0..m {
                for col in 0..n {
                    let actual = four_m[row * ldc + col];
                    let expected = scalar_result[row * ldc + col];
                    assert_relative_eq!(
                        actual.re,
                        expected.re,
                        max_relative = 2.0e-12,
                        epsilon = 2.0e-12
                    );
                    assert_relative_eq!(
                        actual.im,
                        expected.im,
                        max_relative = 2.0e-12,
                        epsilon = 2.0e-12
                    );
                }
            }
        }
    }

    #[test]
    fn complex3m_c32_all_transpose_combinations_match_scalar() {
        let cases = [
            (
                3usize,
                5usize,
                4usize,
                Transpose::NoTrans,
                Transpose::NoTrans,
            ),
            (4, 3, 5, Transpose::NoTrans, Transpose::Trans),
            (5, 4, 3, Transpose::Trans, Transpose::NoTrans),
            (3, 4, 5, Transpose::Trans, Transpose::ConjTrans),
            (4, 5, 3, Transpose::ConjTrans, Transpose::Trans),
            (5, 3, 4, Transpose::ConjTrans, Transpose::ConjTrans),
        ];
        for (case, &(m, n, k, transa, transb)) in cases.iter().enumerate() {
            let (a_rows, a_cols) = match transa {
                Transpose::NoTrans => (m, k),
                Transpose::Trans | Transpose::ConjTrans => (k, m),
            };
            let (b_rows, b_cols) = match transb {
                Transpose::NoTrans => (k, n),
                Transpose::Trans | Transpose::ConjTrans => (n, k),
            };
            let lda = a_cols + 1;
            let ldb = b_cols + 2;
            let ldc = n + 1;
            let a: Vec<Complex32> = (0..a_rows * lda)
                .map(|index| {
                    Complex32::new(
                        (index as f32 * 0.17 + case as f32).sin(),
                        (index as f32 * 0.11 - case as f32).cos(),
                    )
                })
                .collect();
            let b: Vec<Complex32> = (0..b_rows * ldb)
                .map(|index| {
                    Complex32::new(
                        (index as f32 * 0.13 - 0.4).cos(),
                        (index as f32 * 0.07 + 0.2).sin(),
                    )
                })
                .collect();
            let initial: Vec<Complex32> = (0..m * ldc)
                .map(|index| Complex32::new(0.2 + index as f32 * 0.01, -0.3))
                .collect();
            let alpha = Complex32::new(0.7, -0.25);
            let beta = Complex32::new(-0.2, 0.4);
            let mut four_m = initial.clone();
            let mut scalar_result = initial;
            gemm_c32(
                transa,
                transb,
                m,
                n,
                k,
                alpha,
                &a,
                lda,
                &b,
                ldb,
                beta,
                &mut four_m,
                ldc,
            );
            scalar::gemm_c32(
                transa,
                transb,
                m,
                n,
                k,
                alpha,
                &a,
                lda,
                &b,
                ldb,
                beta,
                &mut scalar_result,
                ldc,
            );
            for row in 0..m {
                for col in 0..n {
                    let actual = four_m[row * ldc + col];
                    let expected = scalar_result[row * ldc + col];
                    assert_relative_eq!(
                        actual.re,
                        expected.re,
                        max_relative = 2.0e-4,
                        epsilon = 3.0e-5
                    );
                    assert_relative_eq!(
                        actual.im,
                        expected.im,
                        max_relative = 2.0e-4,
                        epsilon = 3.0e-5
                    );
                }
            }
        }
    }
}
