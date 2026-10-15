//! Native complex thin-`k` GEMM (`C := α·A·B + β·C`, `NoTrans` only).
//!
//! Single-pass alternative to the 4M decomposition ([`super::complex4m`])
//! for thin-`k` shapes (LU Schur updates, tall-skinny multiplies). 4M pays
//! per call: deinterleave passes over `A`/`B`, four real-GEMM dispatches
//! (four packs, four rayon scopes), two `m × n` scratch tiles, and a scalar
//! combine epilogue. This kernel instead packs `B` once into split
//! real/imag halves, streams `A` rows through a 2×2-complex AVX2
//! micro-kernel with split accumulators, and folds the epilogue into the
//! tile store — one `C` pass instead of 4M's half-dozen.
//!
//! Gated to the two hot `(α, β)` pairs: Schur accumulate (`α = −1`,
//! `β = 1`) and overwrite (`α = 1`, `β = 0`). Anything else (or `k` above
//! [`THIN_K_MAX`]) returns `false` and the caller falls back to 4M, so
//! general-GEMM users never see this path.
//!
//! Rounding: accumulation runs over split FMA chains in `k` order per
//! 2-row band, so results match the serial sweep bit-for-bit across
//! parallel/serial decomposition, but differ from 4M/cblas in the last
//! ulp (different summation order — same class as any blocked GEMM).

#![allow(unsafe_op_in_unsafe_fn)]

use num_complex::{Complex32, Complex64};
use rayon::prelude::*;
use std::cell::RefCell;

thread_local! {
    static THINK_BR: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static THINK_BI: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

/// Maximum `k` intercepted: the `nb = 32` complex LU panel width. Wider
/// `k` belongs to the panel-pack native path — at `k = 64` (the QR panel
/// width) the native 6×4 sweep beats this kernel's `n`-strided B pack by
/// ~1.35x on Schur shapes (2048×2048×64: 14.5 vs 19.5 ms, interleaved
/// A/B 2026-10-15), while at `k = 32` the thin-k path still wins the LU
/// Schur updates it was built for.
pub const THIN_K_MAX: usize = 32;

/// Work floor (complex MACs): the native kernel only engages for large
/// parallel shapes. The serial micro-kernel is broadcast/load port-bound
/// (~14 GF) and loses to the serial real kernels everywhere, so 4M keeps
/// serial, single-threaded, and small shapes — and mid-size parallel
/// shapes too: the hand-rolled band sweep shows placement-luck variance
/// in LU context below ~8M MACs (a 1M floor loses ~2ms on zgetrf-512, a
/// 3M floor still loses ~1.7ms there, both while winning at 1024). At 9M+
/// the per-call sweep wins are large and robust. Tuned by sweep +
/// end-to-end A/B (see `zsweep`).
pub const THIN_MIN_MACS: usize = 9_000_000;

/// Target complex MACs per row-band task (same sizing rationale as the
/// real kernel's task target: enough waves on wide machines, spawn cost
/// negligible).
const THIN_TASK_MACS: usize = 250_000;

/// B-pack deinterleave floor (`k * n` elements): above it the pack fans
/// out over the pool, below it the serial loop keeps dispatch off the
/// critical path. Serial pack costs ~7ns/el on the caller; the pool
/// dispatch costs ~20us, so breakeven sits near 3K els and 8K holds
/// 2.5x margin. LU-Schur shapes always clear it (`k * n >= 48 * 433`).
const THIN_PACK_PAR_MIN_ELS: usize = 8_000;

/// B-pack prefetch distance (`k` steps ahead) in the AVX2 micro-kernels:
/// the pack streams from L3 (~50-70 cycles) while a 4-row step retires
/// ~8 cycles, so 8 steps keeps the stream fed. Guarded (`p + DIST < k`)
/// so the pointer math never leaves the pack.
#[cfg(target_arch = "x86_64")]
const THINK_PREFETCH_DIST: usize = 8;

/// Diagnostic hook: `SONIUM_COMPLEX_THINK=0` disables the native path
/// (clean A/B against 4M in one binary).
#[inline]
fn think_disabled() -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var("SONIUM_COMPLEX_THINK").is_ok_and(|v| v == "0"))
}

/// Thin-`k` complex GEMM. Returns `true` when the call was handled (always
/// for gated shapes, including empty ones), `false` to fall back to 4M.
pub fn zgemm_think(
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
    if m == 0 || n == 0 {
        return true;
    }
    if think_disabled() || k == 0 || k > THIN_K_MAX || m < 2 || n < 2 {
        return false;
    }
    // Hot pairs only (see module docs).
    let schur = alpha.re == -1.0 && alpha.im == 0.0 && beta.re == 1.0 && beta.im == 0.0;
    let overwrite = alpha.re == 1.0 && alpha.im == 0.0 && beta.re == 0.0 && beta.im == 0.0;
    if !(schur || overwrite) {
        return false;
    }
    // Parallel-only: serial/small/single-threaded shapes stay on 4M.
    if !(rayon::current_num_threads() > 1
        && m >= 32
        && m.saturating_mul(n).saturating_mul(k) >= THIN_MIN_MACS)
    {
        return false;
    }
    super::with_buf(&THINK_BR, k * n, |br| {
        super::with_buf(&THINK_BI, k * n, |bi| {
            // B deinterleave into split halves (shipped 2026-10-02): the
            // serial pack cost ~7ns/el on the caller — 0.34ms at T=1000
            // k=48, the entire mid-size thin-k gap vs OB (T=1000 1.52x,
            // T=500 1.47x, exact T-proportional overhead). The pool pack
            // (~20us dispatch + work/width) wins above the floor; the
            // serial loop keeps small shapes dispatch-free below it.
            // `with_buf` yields exactly `k * n` els, so the chunk zip
            // covers all `k` rows with no remainder.
            if k.saturating_mul(n) >= THIN_PACK_PAR_MIN_ELS {
                br.par_chunks_mut(n)
                    .zip(bi.par_chunks_mut(n))
                    .enumerate()
                    .for_each(|(p, (br_row, bi_row))| {
                        for (j, z) in b[p * ldb..p * ldb + n].iter().enumerate() {
                            br_row[j] = z.re;
                            bi_row[j] = z.im;
                        }
                    });
            } else {
                for p in 0..k {
                    for j in 0..n {
                        let z = b[p * ldb + j];
                        br[p * n + j] = z.re;
                        bi[p * n + j] = z.im;
                    }
                }
            }
            // SAFETY: bands partition C rows disjointly; A/B are read-only
            // (B pack owned by this call, alive across the scope).
            unsafe {
                let macs = m.saturating_mul(n).saturating_mul(k);
                // The gate already ensured a parallel shape; re-check for
                // safety (direct callers aside, the serial band is the
                // fallback, not the fast path — see THIN_MIN_MACS).
                let parallel = rayon::current_num_threads() > 1 && macs >= THIN_MIN_MACS && m >= 32;
                if !parallel {
                    zthink_band(
                        0,
                        m,
                        m,
                        n,
                        k,
                        schur,
                        a.as_ptr(),
                        lda,
                        br.as_ptr(),
                        bi.as_ptr(),
                        c.as_mut_ptr(),
                        ldc,
                    );
                    return;
                }
                let workers = super::workers_for_macs(macs.saturating_mul(8));
                let per_row = n.saturating_mul(k).max(1);
                let mut band_rows = (THIN_TASK_MACS.div_ceil(per_row)).max(16).min(m);
                band_rows &= !1;
                if band_rows < 2 {
                    band_rows = 2.min(m & !1);
                }
                // Keep at least `workers` bands when rows allow.
                let min_bands = workers.min(m / 2);
                if min_bands > 1 {
                    let fit = (m / min_bands) & !1;
                    if fit >= 4 {
                        band_rows = band_rows.min(fit);
                    }
                }
                // Pointers cross the scope as `usize` (disjoint closure
                // capture would otherwise grab the raw fields, defeating
                // the Send wrappers); bands partition C rows disjointly.
                let ap = a.as_ptr() as usize;
                let brp = br.as_ptr() as usize;
                let bip = bi.as_ptr() as usize;
                let cp = c.as_mut_ptr() as usize;
                super::scope_pinned(|s| {
                    let mut r0 = 0;
                    while r0 < m {
                        let r1 = (r0 + band_rows).min(m);
                        s.spawn(move |_| {
                            zthink_band(
                                r0,
                                r1,
                                m,
                                n,
                                k,
                                schur,
                                ap as *const Complex64,
                                lda,
                                brp as *const f64,
                                bip as *const f64,
                                cp as *mut Complex64,
                                ldc,
                            );
                        });
                        r0 = r1;
                    }
                });
            }
        });
    });
    true
}

/// Serial sweep over rows `[r0, r1)`: 2×2-complex AVX2 tiles, scalar tails.
#[inline]
unsafe fn zthink_band(
    r0: usize,
    r1: usize,
    _m: usize,
    n: usize,
    k: usize,
    schur: bool,
    a: *const Complex64,
    lda: usize,
    br: *const f64,
    bi: *const f64,
    c: *mut Complex64,
    ldc: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::blas1::avx2::is_available() {
        // SAFETY: same contract as the caller (disjoint band rows, live
        // buffers, AVX2+FMA checked above).
        unsafe {
            zthink_band_avx2(r0, r1, n, k, schur, a, lda, br, bi, c, ldc);
        }
        return;
    }
    // SAFETY: same contract as the caller.
    unsafe {
        zthink_band_scalar(r0, r1, n, k, schur, a, lda, br, bi, c, ldc);
    }
}

/// Portable reference: same tile decomposition, scalar complex FMA order.
unsafe fn zthink_band_scalar(
    r0: usize,
    r1: usize,
    n: usize,
    k: usize,
    schur: bool,
    a: *const Complex64,
    lda: usize,
    br: *const f64,
    bi: *const f64,
    c: *mut Complex64,
    ldc: usize,
) {
    for i in r0..r1 {
        for j in 0..n {
            let mut sr = 0.0;
            let mut si = 0.0;
            for p in 0..k {
                let av = *a.add(i * lda + p);
                let brp = *br.add(p * n + j);
                let bip = *bi.add(p * n + j);
                sr += av.re * brp - av.im * bip;
                si += av.re * bip + av.im * brp;
            }
            let out = if schur {
                let prev = *c.add(i * ldc + j);
                Complex64::new(prev.re - sr, prev.im - si)
            } else {
                Complex64::new(sr, si)
            };
            *c.add(i * ldc + j) = out;
        }
    }
}

/// AVX2 2×2-complex micro-kernel with split accumulators. Per `k` step and
/// row pair: 2 `B` loads, 4 broadcasts, 8 FMAs for 4 complex MACs (32 real
/// flops). Epilogue interleaves via unpack + `permute2f128` resequence
/// (same lane-ordering discipline as the 4M epilogue rows).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn zthink_band_avx2(
    r0: usize,
    r1: usize,
    n: usize,
    k: usize,
    schur: bool,
    a: *const Complex64,
    lda: usize,
    br: *const f64,
    bi: *const f64,
    c: *mut Complex64,
    ldc: usize,
) {
    use core::arch::x86_64::*;
    let af = a as *const f64;
    let cf = c as *mut f64;
    let sign = _mm256_set1_pd(-0.0);
    // Full-width tiles: the 4-lane accumulators cover 4 columns per step
    // (lanes 2..3 hold cols j+2..j+3, stored via the 0x31 half — leaving
    // them idle would waste half the FMA throughput AND over-read the
    // pack at the row end).
    let n_main = n & !3;
    let mut i = r0;
    // 4-row tiles (main path, shipped 2026-10-02): 8 accumulator chains
    // hide the B-pack L3 latency (the 2-row form's 4 chains stall at
    // ~40% of the port limit — the worker sweep scales linearly, so the
    // kernel is latency-bound, not bandwidth-bound) and each pack
    // stream feeds 4 rows (half the B traffic). 12 live ymm (8 accs +
    // 2 B + 2 broadcasts) fit the AVX2 regfile. Per-element op order
    // matches the 2-row path exactly (k-order accumulation into one
    // accumulator per element), so results are bit-identical.
    while i + 4 <= r1 {
        let mut j = 0;
        while j + 4 <= n_main {
            let mut re0 = _mm256_setzero_pd();
            let mut im0 = _mm256_setzero_pd();
            let mut re1 = _mm256_setzero_pd();
            let mut im1 = _mm256_setzero_pd();
            let mut re2 = _mm256_setzero_pd();
            let mut im2 = _mm256_setzero_pd();
            let mut re3 = _mm256_setzero_pd();
            let mut im3 = _mm256_setzero_pd();
            for p in 0..k {
                let bro = _mm256_loadu_pd(br.add(p * n + j));
                let bio = _mm256_loadu_pd(bi.add(p * n + j));
                if p + THINK_PREFETCH_DIST < k {
                    _mm_prefetch(
                        br.add((p + THINK_PREFETCH_DIST) * n + j) as *const i8,
                        _MM_HINT_T0,
                    );
                    _mm_prefetch(
                        bi.add((p + THINK_PREFETCH_DIST) * n + j) as *const i8,
                        _MM_HINT_T0,
                    );
                }
                let ar0 = _mm256_broadcast_sd(&*af.add((i * lda + p) * 2));
                let ai0 = _mm256_broadcast_sd(&*af.add((i * lda + p) * 2 + 1));
                let ar1 = _mm256_broadcast_sd(&*af.add(((i + 1) * lda + p) * 2));
                let ai1 = _mm256_broadcast_sd(&*af.add(((i + 1) * lda + p) * 2 + 1));
                let ar2 = _mm256_broadcast_sd(&*af.add(((i + 2) * lda + p) * 2));
                let ai2 = _mm256_broadcast_sd(&*af.add(((i + 2) * lda + p) * 2 + 1));
                let ar3 = _mm256_broadcast_sd(&*af.add(((i + 3) * lda + p) * 2));
                let ai3 = _mm256_broadcast_sd(&*af.add(((i + 3) * lda + p) * 2 + 1));
                re0 = _mm256_fmadd_pd(ar0, bro, _mm256_fnmadd_pd(ai0, bio, re0));
                im0 = _mm256_fmadd_pd(ar0, bio, _mm256_fmadd_pd(ai0, bro, im0));
                re1 = _mm256_fmadd_pd(ar1, bro, _mm256_fnmadd_pd(ai1, bio, re1));
                im1 = _mm256_fmadd_pd(ar1, bio, _mm256_fmadd_pd(ai1, bro, im1));
                re2 = _mm256_fmadd_pd(ar2, bro, _mm256_fnmadd_pd(ai2, bio, re2));
                im2 = _mm256_fmadd_pd(ar2, bio, _mm256_fmadd_pd(ai2, bro, im2));
                re3 = _mm256_fmadd_pd(ar3, bro, _mm256_fnmadd_pd(ai3, bio, re3));
                im3 = _mm256_fmadd_pd(ar3, bio, _mm256_fmadd_pd(ai3, bro, im3));
            }
            if schur {
                re0 = _mm256_xor_pd(re0, sign);
                im0 = _mm256_xor_pd(im0, sign);
                re1 = _mm256_xor_pd(re1, sign);
                im1 = _mm256_xor_pd(im1, sign);
                re2 = _mm256_xor_pd(re2, sign);
                im2 = _mm256_xor_pd(im2, sign);
                re3 = _mm256_xor_pd(re3, sign);
                im3 = _mm256_xor_pd(im3, sign);
            }
            let lo0 = _mm256_unpacklo_pd(re0, im0);
            let hi0 = _mm256_unpackhi_pd(re0, im0);
            let ab00 = _mm256_permute2f128_pd(lo0, hi0, 0x20);
            let ab01 = _mm256_permute2f128_pd(lo0, hi0, 0x31);
            let lo1 = _mm256_unpacklo_pd(re1, im1);
            let hi1 = _mm256_unpackhi_pd(re1, im1);
            let ab10 = _mm256_permute2f128_pd(lo1, hi1, 0x20);
            let ab11 = _mm256_permute2f128_pd(lo1, hi1, 0x31);
            let lo2 = _mm256_unpacklo_pd(re2, im2);
            let hi2 = _mm256_unpackhi_pd(re2, im2);
            let ab20 = _mm256_permute2f128_pd(lo2, hi2, 0x20);
            let ab21 = _mm256_permute2f128_pd(lo2, hi2, 0x31);
            let lo3 = _mm256_unpacklo_pd(re3, im3);
            let hi3 = _mm256_unpackhi_pd(re3, im3);
            let ab30 = _mm256_permute2f128_pd(lo3, hi3, 0x20);
            let ab31 = _mm256_permute2f128_pd(lo3, hi3, 0x31);
            if schur {
                let c00 = _mm256_loadu_pd(cf.add((i * ldc + j) * 2));
                let c01 = _mm256_loadu_pd(cf.add((i * ldc + j) * 2 + 4));
                let c10 = _mm256_loadu_pd(cf.add(((i + 1) * ldc + j) * 2));
                let c11 = _mm256_loadu_pd(cf.add(((i + 1) * ldc + j) * 2 + 4));
                let c20 = _mm256_loadu_pd(cf.add(((i + 2) * ldc + j) * 2));
                let c21 = _mm256_loadu_pd(cf.add(((i + 2) * ldc + j) * 2 + 4));
                let c30 = _mm256_loadu_pd(cf.add(((i + 3) * ldc + j) * 2));
                let c31 = _mm256_loadu_pd(cf.add(((i + 3) * ldc + j) * 2 + 4));
                _mm256_storeu_pd(cf.add((i * ldc + j) * 2), _mm256_add_pd(c00, ab00));
                _mm256_storeu_pd(cf.add((i * ldc + j) * 2 + 4), _mm256_add_pd(c01, ab01));
                _mm256_storeu_pd(cf.add(((i + 1) * ldc + j) * 2), _mm256_add_pd(c10, ab10));
                _mm256_storeu_pd(
                    cf.add(((i + 1) * ldc + j) * 2 + 4),
                    _mm256_add_pd(c11, ab11),
                );
                _mm256_storeu_pd(cf.add(((i + 2) * ldc + j) * 2), _mm256_add_pd(c20, ab20));
                _mm256_storeu_pd(
                    cf.add(((i + 2) * ldc + j) * 2 + 4),
                    _mm256_add_pd(c21, ab21),
                );
                _mm256_storeu_pd(cf.add(((i + 3) * ldc + j) * 2), _mm256_add_pd(c30, ab30));
                _mm256_storeu_pd(
                    cf.add(((i + 3) * ldc + j) * 2 + 4),
                    _mm256_add_pd(c31, ab31),
                );
            } else {
                _mm256_storeu_pd(cf.add((i * ldc + j) * 2), ab00);
                _mm256_storeu_pd(cf.add((i * ldc + j) * 2 + 4), ab01);
                _mm256_storeu_pd(cf.add(((i + 1) * ldc + j) * 2), ab10);
                _mm256_storeu_pd(cf.add(((i + 1) * ldc + j) * 2 + 4), ab11);
                _mm256_storeu_pd(cf.add(((i + 2) * ldc + j) * 2), ab20);
                _mm256_storeu_pd(cf.add(((i + 2) * ldc + j) * 2 + 4), ab21);
                _mm256_storeu_pd(cf.add(((i + 3) * ldc + j) * 2), ab30);
                _mm256_storeu_pd(cf.add(((i + 3) * ldc + j) * 2 + 4), ab31);
            }
            j += 4;
        }
        // Column tail (n % 4): scalar accumulation for all four rows of
        // the tile (same per-element order as the portable reference).
        for jj in n_main..n {
            for r in 0..4 {
                let ii = i + r;
                let mut sr = 0.0;
                let mut si = 0.0;
                for p in 0..k {
                    let av = *a.add(ii * lda + p);
                    let brp = *br.add(p * n + jj);
                    let bip = *bi.add(p * n + jj);
                    sr += av.re * brp - av.im * bip;
                    si += av.re * bip + av.im * brp;
                }
                let out = if schur {
                    let prev = *c.add(ii * ldc + jj);
                    Complex64::new(prev.re - sr, prev.im - si)
                } else {
                    Complex64::new(sr, si)
                };
                *c.add(ii * ldc + jj) = out;
            }
        }
        i += 4;
    }
    // 2-row cleanup for the leftover pair (bands are even-rowed, so at
    // most one pair remains; the odd-row tail below still applies).
    while i + 2 <= r1 {
        let mut j = 0;
        while j + 4 <= n_main {
            let mut re0 = _mm256_setzero_pd();
            let mut im0 = _mm256_setzero_pd();
            let mut re1 = _mm256_setzero_pd();
            let mut im1 = _mm256_setzero_pd();
            for p in 0..k {
                let bro = _mm256_loadu_pd(br.add(p * n + j));
                let bio = _mm256_loadu_pd(bi.add(p * n + j));
                let ar0 = _mm256_broadcast_sd(&*af.add((i * lda + p) * 2));
                let ai0 = _mm256_broadcast_sd(&*af.add((i * lda + p) * 2 + 1));
                let ar1 = _mm256_broadcast_sd(&*af.add(((i + 1) * lda + p) * 2));
                let ai1 = _mm256_broadcast_sd(&*af.add(((i + 1) * lda + p) * 2 + 1));
                re0 = _mm256_fmadd_pd(ar0, bro, _mm256_fnmadd_pd(ai0, bio, re0));
                im0 = _mm256_fmadd_pd(ar0, bio, _mm256_fmadd_pd(ai0, bro, im0));
                re1 = _mm256_fmadd_pd(ar1, bro, _mm256_fnmadd_pd(ai1, bio, re1));
                im1 = _mm256_fmadd_pd(ar1, bio, _mm256_fmadd_pd(ai1, bro, im1));
            }
            if schur {
                re0 = _mm256_xor_pd(re0, sign);
                im0 = _mm256_xor_pd(im0, sign);
                re1 = _mm256_xor_pd(re1, sign);
                im1 = _mm256_xor_pd(im1, sign);
            }
            // Row 0: cols j..j+1 via the 0x20 half, j+2..j+3 via 0x31.
            let lo0 = _mm256_unpacklo_pd(re0, im0);
            let hi0 = _mm256_unpackhi_pd(re0, im0);
            let ab00 = _mm256_permute2f128_pd(lo0, hi0, 0x20);
            let ab01 = _mm256_permute2f128_pd(lo0, hi0, 0x31);
            let lo1 = _mm256_unpacklo_pd(re1, im1);
            let hi1 = _mm256_unpackhi_pd(re1, im1);
            let ab10 = _mm256_permute2f128_pd(lo1, hi1, 0x20);
            let ab11 = _mm256_permute2f128_pd(lo1, hi1, 0x31);
            if schur {
                let c00 = _mm256_loadu_pd(cf.add((i * ldc + j) * 2));
                let c01 = _mm256_loadu_pd(cf.add((i * ldc + j) * 2 + 4));
                let c10 = _mm256_loadu_pd(cf.add(((i + 1) * ldc + j) * 2));
                let c11 = _mm256_loadu_pd(cf.add(((i + 1) * ldc + j) * 2 + 4));
                _mm256_storeu_pd(cf.add((i * ldc + j) * 2), _mm256_add_pd(c00, ab00));
                _mm256_storeu_pd(cf.add((i * ldc + j) * 2 + 4), _mm256_add_pd(c01, ab01));
                _mm256_storeu_pd(cf.add(((i + 1) * ldc + j) * 2), _mm256_add_pd(c10, ab10));
                _mm256_storeu_pd(
                    cf.add(((i + 1) * ldc + j) * 2 + 4),
                    _mm256_add_pd(c11, ab11),
                );
            } else {
                _mm256_storeu_pd(cf.add((i * ldc + j) * 2), ab00);
                _mm256_storeu_pd(cf.add((i * ldc + j) * 2 + 4), ab01);
                _mm256_storeu_pd(cf.add(((i + 1) * ldc + j) * 2), ab10);
                _mm256_storeu_pd(cf.add(((i + 1) * ldc + j) * 2 + 4), ab11);
            }
            j += 4;
        }
        // Column tail (n % 4): scalar accumulation for both rows of the
        // pair (same per-element order as the portable reference).
        for jj in n_main..n {
            for r in 0..2 {
                let ii = i + r;
                let mut sr = 0.0;
                let mut si = 0.0;
                for p in 0..k {
                    let av = *a.add(ii * lda + p);
                    let brp = *br.add(p * n + jj);
                    let bip = *bi.add(p * n + jj);
                    sr += av.re * brp - av.im * bip;
                    si += av.re * bip + av.im * brp;
                }
                let out = if schur {
                    let prev = *c.add(ii * ldc + jj);
                    Complex64::new(prev.re - sr, prev.im - si)
                } else {
                    Complex64::new(sr, si)
                };
                *c.add(ii * ldc + jj) = out;
            }
        }
        i += 2;
    }
    // Row tail (m odd): portable scalar over all columns (rare; keeps the
    // vector path free of masking).
    zthink_band_scalar(i, r1, n, k, schur, a, lda, br, bi, c, ldc);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn ref_zgemm(
        m: usize,
        n: usize,
        k: usize,
        alpha: Complex64,
        a: &[Complex64],
        lda: usize,
        b: &[Complex64],
        ldb: usize,
        beta: Complex64,
        c0: &[Complex64],
        ldc: usize,
    ) -> Vec<Complex64> {
        let mut c = c0.to_vec();
        for i in 0..m {
            for j in 0..n {
                let mut s = Complex64::new(0.0, 0.0);
                for p in 0..k {
                    s += a[i * lda + p] * b[p * ldb + j];
                }
                c[i * ldc + j] = alpha * s + beta * c0[i * ldc + j];
            }
        }
        c
    }

    fn ramp_c64(n: usize, f: f64, g: f64) -> Vec<Complex64> {
        (0..n)
            .map(|i| Complex64::new((i as f64 * 0.0123 * f).sin(), (i as f64 * 0.0177 * g).cos()))
            .collect()
    }

    /// Gated pairs (Schur accumulate, overwrite) match the naive reference
    /// on even, odd, strided, and tiny shapes (tails included).
    #[test]
    fn think_gated_pairs_match_reference() {
        let shapes = [
            (130usize, 129usize, 32usize, 130usize, 129usize, 150usize),
            (7, 9, 5, 7, 9, 9),
            (16, 16, 1, 16, 16, 16),
            (33, 33, 64, 64, 64, 40),
            (100, 3, 17, 100, 17, 3),
            (3, 100, 17, 17, 100, 100),
        ];
        for (alpha, beta) in [
            (Complex64::new(-1.0, 0.0), Complex64::new(1.0, 0.0)),
            (Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)),
        ] {
            for &(m, n, k, lda, ldb, ldc) in &shapes {
                let a = ramp_c64(m * lda, 1.0, 1.0);
                let b = ramp_c64(k * ldb, 1.7, 0.6);
                let c0 = ramp_c64(m * ldc, 0.5, 1.3);
                let mut c = c0.clone();
                // Bypass the gate: call the band sweep directly so small
                // shapes exercise the tails even though the gate rejects
                // them (fall-through correctness is covered by the 4M
                // tests; this pins the micro-kernel + tails).
                super::super::with_buf(&THINK_BR, k * n, |br| {
                    super::super::with_buf(&THINK_BI, k * n, |bi| {
                        for p in 0..k {
                            for j in 0..n {
                                let z = b[p * ldb + j];
                                br[p * n + j] = z.re;
                                bi[p * n + j] = z.im;
                            }
                        }
                        unsafe {
                            zthink_band(
                                0,
                                m,
                                m,
                                n,
                                k,
                                alpha.re == -1.0,
                                a.as_ptr(),
                                lda,
                                br.as_ptr(),
                                bi.as_ptr(),
                                c.as_mut_ptr(),
                                ldc,
                            );
                        }
                    });
                });
                let want = ref_zgemm(m, n, k, alpha, &a, lda, &b, ldb, beta, &c0, ldc);
                for i in 0..m {
                    for (g, w) in c[i * ldc..i * ldc + n]
                        .iter()
                        .zip(want[i * ldc..i * ldc + n].iter())
                    {
                        assert_relative_eq!(g.re, w.re, max_relative = 1e-9, epsilon = 1e-9);
                        assert_relative_eq!(g.im, w.im, max_relative = 1e-9, epsilon = 1e-9);
                    }
                }
            }
        }
    }

    /// Gate: general scalars, wide k, and small shapes fall back to 4M
    /// (`false`); empty shapes are handled (`true`).
    #[test]
    fn think_gate_falls_back_off_shape() {
        let a = ramp_c64(64 * 8, 1.0, 1.0);
        let b = ramp_c64(8 * 64, 1.0, 1.0);
        let mut c = ramp_c64(64 * 64, 1.0, 1.0);
        // General alpha: fall back.
        assert!(!zgemm_think(
            64,
            64,
            8,
            Complex64::new(0.3, -0.2),
            &a,
            8,
            &b,
            64,
            Complex64::new(1.0, 0.0),
            &mut c,
            64,
        ));
        // Wide k: fall back.
        let aw = ramp_c64(64 * 128, 1.0, 1.0);
        let bw = ramp_c64(128 * 64, 1.0, 1.0);
        assert!(!zgemm_think(
            64,
            64,
            128,
            Complex64::new(-1.0, 0.0),
            &aw,
            128,
            &bw,
            64,
            Complex64::new(1.0, 0.0),
            &mut c,
            64,
        ));
        // Empty: handled.
        assert!(zgemm_think(
            0,
            64,
            8,
            Complex64::new(-1.0, 0.0),
            &a,
            8,
            &b,
            64,
            Complex64::new(1.0, 0.0),
            &mut c,
            64,
        ));
    }
}

// ---------- Complex32 twin ----------

thread_local! {
    static THINK_BR32: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static THINK_BI32: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Complex32 twin of [`zgemm_think`]: same gate (hot pairs, `k <= 64`,
/// parallel shapes at/above [`THIN_MIN_MACS`]), same `SONIUM_COMPLEX_THINK`
/// kill switch.
pub fn cgemm_think(
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
    use num_complex::Complex32 as C;
    if m == 0 || n == 0 {
        return true;
    }
    if think_disabled() || k == 0 || k > THIN_K_MAX || m < 2 || n < 2 {
        return false;
    }
    let schur = alpha.re == -1.0 && alpha.im == 0.0 && beta.re == 1.0 && beta.im == 0.0;
    let overwrite = alpha.re == 1.0 && alpha.im == 0.0 && beta.re == 0.0 && beta.im == 0.0;
    if !(schur || overwrite) {
        return false;
    }
    if !(rayon::current_num_threads() > 1
        && m >= 32
        && m.saturating_mul(n).saturating_mul(k) >= THIN_MIN_MACS)
    {
        return false;
    }
    super::with_buf(&THINK_BR32, k * n, |br| {
        super::with_buf(&THINK_BI32, k * n, |bi| {
            // Parallel B deinterleave (see `zgemm_think`): same serial
            // cost (~7ns/el) dominated the mid-size c32 Schur cells.
            if k.saturating_mul(n) >= THIN_PACK_PAR_MIN_ELS {
                br.par_chunks_mut(n)
                    .zip(bi.par_chunks_mut(n))
                    .enumerate()
                    .for_each(|(p, (br_row, bi_row))| {
                        for (j, z) in b[p * ldb..p * ldb + n].iter().enumerate() {
                            br_row[j] = z.re;
                            bi_row[j] = z.im;
                        }
                    });
            } else {
                for p in 0..k {
                    for j in 0..n {
                        let z = b[p * ldb + j];
                        br[p * n + j] = z.re;
                        bi[p * n + j] = z.im;
                    }
                }
            }
            unsafe {
                let macs = m.saturating_mul(n).saturating_mul(k);
                let parallel = rayon::current_num_threads() > 1 && macs >= THIN_MIN_MACS && m >= 32;
                if !parallel {
                    cthink_band(
                        0,
                        m,
                        m,
                        n,
                        k,
                        schur,
                        a.as_ptr(),
                        lda,
                        br.as_ptr(),
                        bi.as_ptr(),
                        c.as_mut_ptr(),
                        ldc,
                    );
                    return;
                }
                let workers = super::workers_for_macs(macs.saturating_mul(8));
                let per_row = n.saturating_mul(k).max(1);
                let mut band_rows = (THIN_TASK_MACS.div_ceil(per_row)).max(16).min(m);
                band_rows &= !1;
                if band_rows < 2 {
                    band_rows = 2.min(m & !1);
                }
                let min_bands = workers.min(m / 2);
                if min_bands > 1 {
                    let fit = (m / min_bands) & !1;
                    if fit >= 4 {
                        band_rows = band_rows.min(fit);
                    }
                }
                let ap = a.as_ptr() as usize;
                let brp = br.as_ptr() as usize;
                let bip = bi.as_ptr() as usize;
                let cp = c.as_mut_ptr() as usize;
                super::scope_pinned(|s| {
                    let mut r0 = 0;
                    while r0 < m {
                        let r1 = (r0 + band_rows).min(m);
                        s.spawn(move |_| {
                            cthink_band(
                                r0,
                                r1,
                                m,
                                n,
                                k,
                                schur,
                                ap as *const C,
                                lda,
                                brp as *const f32,
                                bip as *const f32,
                                cp as *mut C,
                                ldc,
                            );
                        });
                        r0 = r1;
                    }
                });
            }
        });
    });
    true
}

/// Serial sweep over rows `[r0, r1)` (Complex32).
#[inline]
unsafe fn cthink_band(
    r0: usize,
    r1: usize,
    _m: usize,
    n: usize,
    k: usize,
    schur: bool,
    a: *const Complex32,
    lda: usize,
    br: *const f32,
    bi: *const f32,
    c: *mut Complex32,
    ldc: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::blas1::avx2::is_available() {
        // SAFETY: same contract as the caller (disjoint band rows, live
        // buffers, AVX2+FMA checked above).
        unsafe {
            cthink_band_avx2(r0, r1, n, k, schur, a, lda, br, bi, c, ldc);
        }
        return;
    }
    // SAFETY: same contract as the caller.
    unsafe {
        cthink_band_scalar(r0, r1, n, k, schur, a, lda, br, bi, c, ldc);
    }
}

/// Portable reference (Complex32).
unsafe fn cthink_band_scalar(
    r0: usize,
    r1: usize,
    n: usize,
    k: usize,
    schur: bool,
    a: *const Complex32,
    lda: usize,
    br: *const f32,
    bi: *const f32,
    c: *mut Complex32,
    ldc: usize,
) {
    use num_complex::Complex32 as C;
    for i in r0..r1 {
        for j in 0..n {
            let mut sr = 0.0;
            let mut si = 0.0;
            for p in 0..k {
                let av = *a.add(i * lda + p);
                let brp = *br.add(p * n + j);
                let bip = *bi.add(p * n + j);
                sr += av.re * brp - av.im * bip;
                si += av.re * bip + av.im * brp;
            }
            let out = if schur {
                let prev = *c.add(i * ldc + j);
                C::new(prev.re - sr, prev.im - si)
            } else {
                C::new(sr, si)
            };
            *c.add(i * ldc + j) = out;
        }
    }
}

/// AVX2 2×2-complex micro-kernel (Complex32): 4 values per vector, pair
/// swap via `_mm256_permute_ps(_, 0xB1)` where interleaved, same
/// unpack + `permute2f128` resequence discipline as the c64 epilogue.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn cthink_band_avx2(
    r0: usize,
    r1: usize,
    n: usize,
    k: usize,
    schur: bool,
    a: *const Complex32,
    lda: usize,
    br: *const f32,
    bi: *const f32,
    c: *mut Complex32,
    ldc: usize,
) {
    use core::arch::x86_64::*;
    use num_complex::Complex32 as C;
    let af = a as *const f32;
    let cf = c as *mut f32;
    let sign = _mm256_set1_ps(-0.0);
    // Full-width tiles: the 8-lane accumulators cover 8 columns per step
    // (ab0 = cols j..j+3, ab1 = j+4..j+7). A narrower step would overlap
    // tiles (double accumulate) and overrun the row end.
    let n_main = n & !7;
    let mut i = r0;
    // 4-row tiles (main path, shipped 2026-10-02): c32 twin of the c64
    // 4-row form — 8 accumulator chains hide the B-pack L3 latency and
    // each pack stream feeds 4 rows (half the B traffic). 12 live ymm
    // fit the AVX2 regfile; per-element op order matches the 2-row path
    // exactly, so results are bit-identical.
    while i + 4 <= r1 {
        let mut j = 0;
        while j + 8 <= n_main {
            let mut re0 = _mm256_setzero_ps();
            let mut im0 = _mm256_setzero_ps();
            let mut re1 = _mm256_setzero_ps();
            let mut im1 = _mm256_setzero_ps();
            let mut re2 = _mm256_setzero_ps();
            let mut im2 = _mm256_setzero_ps();
            let mut re3 = _mm256_setzero_ps();
            let mut im3 = _mm256_setzero_ps();
            for p in 0..k {
                // 8-wide B halves cover 8 columns; A broadcasts per row.
                let bro = _mm256_loadu_ps(br.add(p * n + j));
                let bio = _mm256_loadu_ps(bi.add(p * n + j));
                if p + THINK_PREFETCH_DIST < k {
                    _mm_prefetch(
                        br.add((p + THINK_PREFETCH_DIST) * n + j) as *const i8,
                        _MM_HINT_T0,
                    );
                    _mm_prefetch(
                        bi.add((p + THINK_PREFETCH_DIST) * n + j) as *const i8,
                        _MM_HINT_T0,
                    );
                }
                let ar0 = _mm256_broadcast_ss(&*af.add((i * lda + p) * 2));
                let ai0 = _mm256_broadcast_ss(&*af.add((i * lda + p) * 2 + 1));
                let ar1 = _mm256_broadcast_ss(&*af.add(((i + 1) * lda + p) * 2));
                let ai1 = _mm256_broadcast_ss(&*af.add(((i + 1) * lda + p) * 2 + 1));
                let ar2 = _mm256_broadcast_ss(&*af.add(((i + 2) * lda + p) * 2));
                let ai2 = _mm256_broadcast_ss(&*af.add(((i + 2) * lda + p) * 2 + 1));
                let ar3 = _mm256_broadcast_ss(&*af.add(((i + 3) * lda + p) * 2));
                let ai3 = _mm256_broadcast_ss(&*af.add(((i + 3) * lda + p) * 2 + 1));
                re0 = _mm256_fmadd_ps(ar0, bro, _mm256_fnmadd_ps(ai0, bio, re0));
                im0 = _mm256_fmadd_ps(ar0, bio, _mm256_fmadd_ps(ai0, bro, im0));
                re1 = _mm256_fmadd_ps(ar1, bro, _mm256_fnmadd_ps(ai1, bio, re1));
                im1 = _mm256_fmadd_ps(ar1, bio, _mm256_fmadd_ps(ai1, bro, im1));
                re2 = _mm256_fmadd_ps(ar2, bro, _mm256_fnmadd_ps(ai2, bio, re2));
                im2 = _mm256_fmadd_ps(ar2, bio, _mm256_fmadd_ps(ai2, bro, im2));
                re3 = _mm256_fmadd_ps(ar3, bro, _mm256_fnmadd_ps(ai3, bio, re3));
                im3 = _mm256_fmadd_ps(ar3, bio, _mm256_fmadd_ps(ai3, bro, im3));
            }
            // 4 rows × 8 cols: two interleaved stores per row.
            for r in 0..4 {
                let (re, im) = if r == 0 {
                    (re0, im0)
                } else if r == 1 {
                    (re1, im1)
                } else if r == 2 {
                    (re2, im2)
                } else {
                    (re3, im3)
                };
                let (re, im) = if schur {
                    (_mm256_xor_ps(re, sign), _mm256_xor_ps(im, sign))
                } else {
                    (re, im)
                };
                let lo = _mm256_unpacklo_ps(re, im);
                let hi = _mm256_unpackhi_ps(re, im);
                let ab0 = _mm256_permute2f128_ps(lo, hi, 0x20);
                let ab1 = _mm256_permute2f128_ps(lo, hi, 0x31);
                let row = (i + r) * ldc + j;
                if schur {
                    let c0 = _mm256_loadu_ps(cf.add(row * 2));
                    let c1 = _mm256_loadu_ps(cf.add(row * 2 + 8));
                    _mm256_storeu_ps(cf.add(row * 2), _mm256_add_ps(c0, ab0));
                    _mm256_storeu_ps(cf.add(row * 2 + 8), _mm256_add_ps(c1, ab1));
                } else {
                    _mm256_storeu_ps(cf.add(row * 2), ab0);
                    _mm256_storeu_ps(cf.add(row * 2 + 8), ab1);
                }
            }
            j += 8;
        }
        // Column tail (n % 8): scalar for all four rows of the tile.
        for jj in n_main..n {
            for r in 0..4 {
                let ii = i + r;
                let mut sr = 0.0;
                let mut si = 0.0;
                for p in 0..k {
                    let av = *a.add(ii * lda + p);
                    let brp = *br.add(p * n + jj);
                    let bip = *bi.add(p * n + jj);
                    sr += av.re * brp - av.im * bip;
                    si += av.re * bip + av.im * brp;
                }
                let out = if schur {
                    let prev = *c.add(ii * ldc + jj);
                    C::new(prev.re - sr, prev.im - si)
                } else {
                    C::new(sr, si)
                };
                *c.add(ii * ldc + jj) = out;
            }
        }
        i += 4;
    }
    // 2-row cleanup for the leftover pair (bands are even-rowed, so at
    // most one pair remains; the odd-row tail below still applies).
    while i + 2 <= r1 {
        let mut j = 0;
        while j + 8 <= n_main {
            let mut re0 = _mm256_setzero_ps();
            let mut im0 = _mm256_setzero_ps();
            let mut re1 = _mm256_setzero_ps();
            let mut im1 = _mm256_setzero_ps();
            for p in 0..k {
                // 4-wide B halves cover 4 columns; A broadcasts per row.
                let bro = _mm256_loadu_ps(br.add(p * n + j));
                let bio = _mm256_loadu_ps(bi.add(p * n + j));
                let ar0 = _mm256_broadcast_ss(&*af.add((i * lda + p) * 2));
                let ai0 = _mm256_broadcast_ss(&*af.add((i * lda + p) * 2 + 1));
                let ar1 = _mm256_broadcast_ss(&*af.add(((i + 1) * lda + p) * 2));
                let ai1 = _mm256_broadcast_ss(&*af.add(((i + 1) * lda + p) * 2 + 1));
                re0 = _mm256_fmadd_ps(ar0, bro, _mm256_fnmadd_ps(ai0, bio, re0));
                im0 = _mm256_fmadd_ps(ar0, bio, _mm256_fmadd_ps(ai0, bro, im0));
                re1 = _mm256_fmadd_ps(ar1, bro, _mm256_fnmadd_ps(ai1, bio, re1));
                im1 = _mm256_fmadd_ps(ar1, bio, _mm256_fmadd_ps(ai1, bro, im1));
            }
            // 2 rows × 4 cols: two interleaved stores per row.
            for r in 0..2 {
                let (re, im) = if r == 0 { (re0, im0) } else { (re1, im1) };
                let (re, im) = if schur {
                    (_mm256_xor_ps(re, sign), _mm256_xor_ps(im, sign))
                } else {
                    (re, im)
                };
                let lo = _mm256_unpacklo_ps(re, im);
                let hi = _mm256_unpackhi_ps(re, im);
                let ab0 = _mm256_permute2f128_ps(lo, hi, 0x20);
                let ab1 = _mm256_permute2f128_ps(lo, hi, 0x31);
                let row = (i + r) * ldc + j;
                if schur {
                    let c0 = _mm256_loadu_ps(cf.add(row * 2));
                    let c1 = _mm256_loadu_ps(cf.add(row * 2 + 8));
                    _mm256_storeu_ps(cf.add(row * 2), _mm256_add_ps(c0, ab0));
                    _mm256_storeu_ps(cf.add(row * 2 + 8), _mm256_add_ps(c1, ab1));
                } else {
                    _mm256_storeu_ps(cf.add(row * 2), ab0);
                    _mm256_storeu_ps(cf.add(row * 2 + 8), ab1);
                }
            }
            j += 8;
        }
        // Column tail (n % 8): scalar for both rows of the pair.
        for jj in n_main..n {
            for r in 0..2 {
                let ii = i + r;
                let mut sr = 0.0;
                let mut si = 0.0;
                for p in 0..k {
                    let av = *a.add(ii * lda + p);
                    let brp = *br.add(p * n + jj);
                    let bip = *bi.add(p * n + jj);
                    sr += av.re * brp - av.im * bip;
                    si += av.re * bip + av.im * brp;
                }
                let out = if schur {
                    let prev = *c.add(ii * ldc + jj);
                    C::new(prev.re - sr, prev.im - si)
                } else {
                    C::new(sr, si)
                };
                *c.add(ii * ldc + jj) = out;
            }
        }
        i += 2;
    }
    // Row tail (m odd): portable scalar over all columns.
    cthink_band_scalar(i, r1, n, k, schur, a, lda, br, bi, c, ldc);
}

#[cfg(test)]
mod tests_c32 {
    use super::*;
    use approx::assert_relative_eq;
    use num_complex::Complex32;

    fn ramp_c32(n: usize, f: f32, g: f32) -> Vec<Complex32> {
        (0..n)
            .map(|i| Complex32::new((i as f32 * 0.0123 * f).sin(), (i as f32 * 0.0177 * g).cos()))
            .collect()
    }

    fn ref_cgemm(
        m: usize,
        n: usize,
        k: usize,
        alpha: Complex32,
        a: &[Complex32],
        lda: usize,
        b: &[Complex32],
        ldb: usize,
        beta: Complex32,
        c0: &[Complex32],
        ldc: usize,
    ) -> Vec<Complex32> {
        let mut c = c0.to_vec();
        for i in 0..m {
            for j in 0..n {
                let mut s = Complex32::new(0.0, 0.0);
                for p in 0..k {
                    s += a[i * lda + p] * b[p * ldb + j];
                }
                c[i * ldc + j] = alpha * s + beta * c0[i * ldc + j];
            }
        }
        c
    }

    /// c32 twin coverage: even, odd, narrow, strided (tails included).
    #[test]
    fn cthink_gated_pairs_match_reference() {
        let shapes = [
            (130usize, 129usize, 32usize, 130usize, 129usize, 150usize),
            (7, 9, 5, 9, 9, 9),
            (16, 16, 1, 16, 16, 16),
            (100, 3, 17, 100, 17, 3),
            (3, 100, 17, 17, 100, 100),
        ];
        for (alpha, beta) in [
            (Complex32::new(-1.0, 0.0), Complex32::new(1.0, 0.0)),
            (Complex32::new(1.0, 0.0), Complex32::new(0.0, 0.0)),
        ] {
            for &(m, n, k, lda, ldb, ldc) in &shapes {
                let a: Vec<Complex32> = ramp_c32(m * lda, 1.0, 1.0);
                let b: Vec<Complex32> = ramp_c32(k * ldb, 1.7, 0.6);
                let c0: Vec<Complex32> = ramp_c32(m * ldc, 0.5, 1.3);
                let mut c = c0.clone();
                super::super::with_buf(&THINK_BR32, k * n, |br| {
                    super::super::with_buf(&THINK_BI32, k * n, |bi| {
                        for p in 0..k {
                            for j in 0..n {
                                let z = b[p * ldb + j];
                                br[p * n + j] = z.re;
                                bi[p * n + j] = z.im;
                            }
                        }
                        unsafe {
                            cthink_band(
                                0,
                                m,
                                m,
                                n,
                                k,
                                alpha.re == -1.0,
                                a.as_ptr(),
                                lda,
                                br.as_ptr(),
                                bi.as_ptr(),
                                c.as_mut_ptr(),
                                ldc,
                            );
                        }
                    });
                });
                let want = ref_cgemm(m, n, k, alpha, &a, lda, &b, ldb, beta, &c0, ldc);
                for i in 0..m {
                    for (g, w) in c[i * ldc..i * ldc + n]
                        .iter()
                        .zip(want[i * ldc..i * ldc + n].iter())
                    {
                        assert_relative_eq!(g.re, w.re, max_relative = 1e-4, epsilon = 1e-5);
                        assert_relative_eq!(g.im, w.im, max_relative = 1e-4, epsilon = 1e-5);
                    }
                }
            }
        }
    }
}
