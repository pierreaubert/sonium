//! Native complex GEMM micro-kernels (AVX2 + FMA).
//!
//! Splits each complex multiply into re/im lanes fed by a **panel-major**
//! B-pack: for panel `jp` (4 columns) and depth `p`, the pack stores the
//! 4-wide vector contiguously, so the k-loop streams `br`/`bi` forward
//! (compare [`complex_think`](super::complex_think), whose `br[p * n + j]`
//! layout strides by `n` per step and loses L1 residency on big-k).
//!
//! Layout, f64 per buffer (c64): `br[(jp * k + p) * 4 + jj]`, same for `bi`.
//! The 6x4 AVX2 micro-kernel holds 12 accumulators (6 re + 6 im YMM):
//! 12 FMA chains cover Zen2's 5-cycle latency (8 chains stall ~1
//! cycle/FMA — the same finding the real `micro_f64_full6` kernel ships).
//! 6-wide tiles use pad-and-copy scratch so edge rows/cols run through
//! the SAME kernel (a scalar tail regressed parallel 256^3 to 3.95x
//! through 4 tail rows).
//!
//! Status (2026-10-01): c64 parallel band sweep, overwrite (alpha=1, beta=0)
//! epilogue first; general alpha/beta arms, k-blocking, serial path, and the
//! c32 twin follow behind this core.

use num_complex::Complex64;
use std::cell::RefCell;

use super::{scope_pinned, with_buf, workers_for_macs};

/// c64 micro-tile: rows x cols per AVX2 step. 6x4: 12 FMA chains
/// cover Zen2's 5-cycle latency (8 chains stall ~1 cycle/FMA — the same
/// finding the real `micro_f64_full6` kernel ships).
pub(crate) const MR: usize = 6;
pub(crate) const NR: usize = 4;

/// Parallel sweep only above this many MACs; smaller tiles run the
/// serial sweep. Unlike the thin-k path (whose shared pack loses below
/// 9M MACs in LU context), the fused sweep parallelizes small updates
/// profitably: floor 100K beats 9M by 30% on zgetrf-512 (13.7 vs 18.5ms)
/// and 11% on zgetrf-1024 (49.3 vs 55.3ms), 3/3 interleaved; floors
/// 10K/100K/1M tie, floor 0 loses on micro-tile scope overhead.
pub(crate) const NATIVE_PAR_MACS: usize = 100_000;

thread_local! {
    static NATIVE_BR: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static NATIVE_BI: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

/// Min-dim gate routing fat cubes to 3M (see [`zgemm_native`]):
/// 3M iff min-dim strictly EXCEEDS this (only min-dim == 1024 shapes
/// ride native; 1536^3+ stay phased).
const C64_3M_MIN_DIM: usize = 1024;

/// Gated entry: general `alpha`/`beta`, any `k` (k-chunked), parallel
/// above [`NATIVE_PAR_MACS`], serial below. Returns whether the call was
/// handled (mirrors the thin-k convention).
#[allow(clippy::too_many_arguments)]
pub(crate) fn zgemm_native(
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
    trans_b: bool,
    conj_b: bool,
) -> bool {
    // 3M on fat cubes (shipped 2026-10-02): the native 6x4 kernel runs
    // ~92% of FMA-peak in real flops, but 3M does 25% fewer of them and
    // the vendor zgemm matches 3 x vendor dgemm time (3M-family), so
    // native 4M-math cannot win there (c64-2048^3 1.32x). A/B
    // (interleaved, masked 16T): 512^3 native +23%, 768^3 native +7%,
    // 1024^3 3M +7%, 2048^3 3M +15% (116.4 -> 98.4ms, 1.32x ->
    // 1.04x). Thin-n 2048x512x2048 (2.1G MACs) still prefers native
    // (+9%), so the gate is min-dim, not MACs. Accuracy: 3M-vs-naive
    // 1.5e-16 vs native 1.3e-16 (relative, 1024^3 general alpha/beta)
    // — identical rounding floors. `SONIUM_C64_FORCE3M=1` forces 3M
    // (diagnostic/A-B).
    // Re-tuned 2026-10-06: the Oct-2 crossover is stale — native now
    // beats phased 3M at 1024^3 ~9% (interleaved same-process 16T,
    // 16 rounds, P/N 1.088 [1.065, 1.095]; phased pays ~3.8ms
    // split+epilogue traffic over 3x dgemm). Phased still wins 1536^3
    // (+12%) and 2048^3 (+29%), so the comparison is strict.
    // 3M is NoTrans-only; transposed-B always runs the native sweep.
    // `SONIUM_C64_FORCENATIVE=1` forces the native sweep (diagnostic/A-B
    // twin of `SONIUM_C64_FORCE3M`).
    if !trans_b
        && std::env::var_os("SONIUM_C64_FORCENATIVE").is_none()
        && (m.min(n).min(k) > C64_3M_MIN_DIM || std::env::var_os("SONIUM_C64_FORCE3M").is_some())
    {
        super::complex3m::gemm_c64_nn(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        return true;
    }
    zgemm_native_run(
        m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_b, conj_b,
    );
    true
}

/// Native c64 GEMM driver: parallel sweep above the floor, serial below.
/// `trans_b` consumes `B` transposed (`n x ldb`, `ConjTrans` adds `conj_b`),
/// skipping the whole-matrix pre-pack; otherwise `B` is `k x ldb`.
///
/// # Panics (debug)
/// Debug-asserts the slice minima for the `m x lda` / `k x ldb`
/// (`n x ldb` transposed) / `m x ldc` footprints.
#[allow(clippy::too_many_arguments)]
pub(crate) fn zgemm_native_run(
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
    trans_b: bool,
    conj_b: bool,
) {
    // Tight footprint minima: `(rows - 1) * ld + cols` (a packed `m x k`
    // A with `lda > k`, or a C sub-block view, legitimately holds less
    // than `rows * ld`).
    debug_assert!(m == 0 || a.len() >= (m - 1).saturating_mul(lda).saturating_add(k));
    if trans_b {
        debug_assert!(n == 0 || b.len() >= (n - 1).saturating_mul(ldb).saturating_add(k));
    } else {
        debug_assert!(k == 0 || b.len() >= (k - 1).saturating_mul(ldb).saturating_add(n));
    }
    debug_assert!(m == 0 || c.len() >= (m - 1).saturating_mul(ldc).saturating_add(n));
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        // C = beta * C (no chunks run).
        for row in c.chunks_mut(ldc).take(m) {
            for slot in row.iter_mut().take(n) {
                *slot = beta * *slot;
            }
        }
        return;
    }
    if m.saturating_mul(n).saturating_mul(k) >= NATIVE_PAR_MACS {
        sweep_bands_c64(
            m, n, k, a, lda, b, ldb, c, ldc, alpha, beta, trans_b, conj_b,
        );
    } else {
        sweep_serial_c64(
            m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_b, conj_b,
        );
    }
}

/// Serial sweep: one row band (`0..full_m`) plus bottom tasks, run
/// directly on the caller (no scope). Same task fns as the parallel path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sweep_serial_c64(
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
    trans_b: bool,
    conj_b: bool,
) {
    let full_m = m / MR * MR;
    let full_n = n / NR * NR;
    let npanels = n.div_ceil(NR);
    let r_tail = m - full_m;
    let mut apad = vec![Complex64::new(0.0, 0.0); MR * k];
    for rr in 0..r_tail {
        apad[rr * k..(rr + 1) * k]
            .copy_from_slice(&a[(full_m + rr) * lda..(full_m + rr) * lda + k]);
    }
    // SAFETY: single-threaded; rectangles partition C disjointly by the
    // same construction as the parallel sweep.
    unsafe {
        let mut b0 = 0;
        while b0 < npanels {
            let b1 = (b0 + PB_PANELS).min(npanels);
            if full_m > 0 {
                band_c64(
                    0,
                    full_m,
                    n,
                    full_n,
                    b0,
                    b1,
                    k,
                    a.as_ptr(),
                    lda,
                    b.as_ptr(),
                    ldb,
                    c.as_mut_ptr(),
                    ldc,
                    alpha,
                    beta,
                    trans_b,
                    conj_b,
                );
            }
            if r_tail > 0 {
                bottom_task_c64(
                    full_m,
                    r_tail,
                    apad.as_ptr(),
                    n,
                    b0,
                    b1,
                    k,
                    b.as_ptr(),
                    ldb,
                    c.as_mut_ptr(),
                    ldc,
                    alpha,
                    beta,
                    trans_b,
                    conj_b,
                );
            }
            b0 += PB_PANELS;
        }
    }
}

/// Pack B rows `[k0, k0 + kc)` for panels `[b0, b1)` into worker-local
/// `br`/`bi` at block-relative offsets (`(jp - b0) * kc * 4`). Each task
/// packs the block it consumes (fused: produced + consumed on the same
/// core, so the pack never crosses the interconnect — a caller-side shared
/// pack cost 0.3+ ms of cross-chiplet C2C per 256^3 call). A partial tail
/// panel in range is zero-padded. `trans_b` reads `B` transposed
/// (`b[j*ldb+p]`, the SYRK-panel layout) instead of packing a whole
/// transposed copy up front; `conj_b` additionally negates the imaginary
/// part (`ConjTrans`, free inside the split).
#[allow(clippy::too_many_arguments)]
fn pack_block_c64(
    n: usize,
    k0: usize,
    kc: usize,
    b: &[Complex64],
    ldb: usize,
    b0: usize,
    b1: usize,
    br: &mut [f64],
    bi: &mut [f64],
    trans_b: bool,
    conj_b: bool,
) {
    debug_assert!(br.len() >= (b1 - b0) * kc * NR);
    debug_assert!(bi.len() >= (b1 - b0) * kc * NR);
    if trans_b {
        pack_block_c64_trans(n, k0, kc, b, ldb, b0, b1, br, bi, conj_b);
        return;
    }
    let full = n / NR * NR;
    #[cfg(target_arch = "x86_64")]
    let simd =
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let simd = false;
    for jp in b0..b1.min(full / NR) {
        let jc = jp * NR;
        let base = (jp - b0) * kc * NR;
        if simd {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: `jc + 4 <= n` (full panel), rows `k0..k0 + kc`
            // readable, `base + kc * 4` in bounds by the caller
            // contract; AVX2+FMA detected above.
            unsafe {
                pack_panel_avx2(kc, b.as_ptr().add(k0 * ldb + jc), ldb, br, bi, base, NR);
            }
        } else {
            for p in 0..kc {
                let row = &b[(k0 + p) * ldb + jc..(k0 + p) * ldb + jc + NR];
                let dst = base + p * NR;
                for (jj, bv) in row.iter().enumerate() {
                    br[dst + jj] = bv.re;
                    bi[dst + jj] = bv.im;
                }
            }
        }
    }
    let tail_panel = full / NR;
    if full < n && b0 <= tail_panel && tail_panel < b1 {
        let base = (tail_panel - b0) * kc * NR;
        br[base..base + kc * NR].fill(0.0);
        bi[base..base + kc * NR].fill(0.0);
        for p in 0..kc {
            let row = &b[(k0 + p) * ldb + full..(k0 + p) * ldb + n];
            let dst = base + p * NR;
            for (jj, bv) in row.iter().enumerate() {
                br[dst + jj] = bv.re;
                bi[dst + jj] = bv.im;
            }
        }
    }
}

/// Transposed half of [`pack_block_c64`]: same `br`/`bi` layout, but
/// element `(p, j)` reads `b[j*ldb + k0+p]` (`B` is `n × ldb`,
/// `k0+p < ldb`). Strided scalar loop (the AVX2 deinterleave needs
/// contiguous rows); the block is small and consumed hot by the calling
/// task, so this still beats a whole-matrix pre-pack plus re-read.
/// `conj_b` negates the imaginary part (`ConjTrans`).
#[allow(clippy::too_many_arguments)]
fn pack_block_c64_trans(
    n: usize,
    k0: usize,
    kc: usize,
    b: &[Complex64],
    ldb: usize,
    b0: usize,
    b1: usize,
    br: &mut [f64],
    bi: &mut [f64],
    conj_b: bool,
) {
    let full = n / NR * NR;
    for jp in b0..b1.min(full / NR) {
        let jc = jp * NR;
        let base = (jp - b0) * kc * NR;
        for p in 0..kc {
            let dst = base + p * NR;
            for jj in 0..NR {
                let bv = b[(jc + jj) * ldb + k0 + p];
                br[dst + jj] = bv.re;
                bi[dst + jj] = if conj_b { -bv.im } else { bv.im };
            }
        }
    }
    let tail_panel = full / NR;
    if full < n && b0 <= tail_panel && tail_panel < b1 {
        let base = (tail_panel - b0) * kc * NR;
        br[base..base + kc * NR].fill(0.0);
        bi[base..base + kc * NR].fill(0.0);
        for p in 0..kc {
            let dst = base + p * NR;
            for jj in 0..n - full {
                let bv = b[(full + jj) * ldb + k0 + p];
                br[dst + jj] = bv.re;
                bi[dst + jj] = if conj_b { -bv.im } else { bv.im };
            }
        }
    }
}

/// Strip copyback: fold a 6x4 scratch tile's live `r x cw` corner into C.
/// First chunk applies `beta` (`C = S + beta * C`, plain copy when
/// `beta == 0`); later chunks accumulate (`C += S`).
///
/// # Safety
/// `dst` spans `r` rows of stride `ldc` (`cw <= 4` columns each, writable,
/// exclusively owned); `scratch` holds the 6 x 4 tile.
#[allow(clippy::too_many_arguments)]
unsafe fn strip_copyback_c64(
    dst: *mut Complex64,
    ldc: usize,
    r: usize,
    cw: usize,
    scratch: &[Complex64; MR * NR],
    beta: Complex64,
    first: bool,
) {
    debug_assert!(r <= MR && cw <= NR);
    let apply_beta = first && beta != Complex64::new(0.0, 0.0);
    for rr in 0..r {
        for jj in 0..cw {
            // SAFETY: `rr < r`, `jj < cw` in bounds per the contract.
            unsafe {
                let slot = dst.add(rr * ldc + jj);
                let s = scratch[rr * NR + jj];
                *slot = if first {
                    if apply_beta { s + beta * *slot } else { s }
                } else {
                    *slot + s
                };
            }
        }
    }
}

/// AVX2 deinterleave pack of 4 panel columns: per depth `p`, two
/// 32-byte loads of 4 complex, `unpacklo`/`unpackhi` + `permute4x64` into
/// re/im vectors, two stores (~8 vector ops per 64 B).
///
/// # Safety
/// `bcol` points at B column `jc` (`jc + 4 <= n`, `k` rows of stride
/// `ldb` readable); `br`/`bi` hold `base + (k - 1) * stride + 4`; caller
/// verified AVX2+FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pack_panel_avx2(
    k: usize,
    bcol: *const Complex64,
    ldb: usize,
    br: &mut [f64],
    bi: &mut [f64],
    base: usize,
    stride: usize,
) {
    use std::arch::x86_64::*;
    let bf = bcol as *const f64;
    for p in 0..k {
        // SAFETY: row `p` of the 4 live columns in bounds per the contract.
        let pair01 = unsafe { _mm256_loadu_pd(bf.add(p * ldb * 2)) };
        let pair23 = unsafe { _mm256_loadu_pd(bf.add(p * ldb * 2 + 4)) };
        let t0 = _mm256_unpacklo_pd(pair01, pair23);
        let t1 = _mm256_unpackhi_pd(pair01, pair23);
        // t0 = [r0,r2,r1,r3] -> [r0,r1,r2,r3]; same lane fix for imag.
        let rev = _mm256_permute4x64_pd(t0, 0xD8);
        let imv = _mm256_permute4x64_pd(t1, 0xD8);
        let dst = base + p * stride;
        // SAFETY: `dst + 4` in bounds per the contract.
        unsafe {
            _mm256_storeu_pd(br.as_mut_ptr().add(dst), rev);
            _mm256_storeu_pd(bi.as_mut_ptr().add(dst), imv);
        }
    }
}

/// Panels per 2D task block: an 8-panel B-block (32 KB per 128-chunk).
/// 8 panels cut 256^3 band count (redundancy 5x -> 2x) while big-k bands
/// stay A-cap-bound (unchanged there). Serial sweep only; the parallel
/// sweep sizes blocks for its shape (see [`pb_panels_par`]).
const PB_PANELS: usize = 8;

/// Parallel-sweep panels per 2D task block: legacy 8 panels, except
/// narrow-tall shapes (`m >= 4n`, `n <= 128`, `npanels > 8`) take two
/// blocks (`pb = npanels / 2`, B-block ~128 KB). Tall-narrow sweeps
/// re-read A once per block over a large A (2048x128x2048: 4 blocks =
/// 4x63 MB), and halving the block count halves that traffic while the
/// band slack policy keeps two waves of tasks (16 bands x 2 blocks);
/// measured 0.93x interleaved 4/4 there. One block is a step too far
/// (B-pack redundancy doubles: 1.05x vs two), and wider-n talls
/// (n = 256) plus all squares keep small legacy blocks (L1D-fit
/// B-blocks win there — a traffic-balanced `sqrt(2Wn/m)` rule was
/// trialled and lost 11% on square c32 by tripling B-packs).
/// `SONIUM_CNATIVE_PB=n` overrides (diagnostic/A-B; shared with c32;
/// read once per process).
fn pb_panels_par(npanels: usize, m: usize, n: usize) -> usize {
    static PB: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        std::env::var("SONIUM_CNATIVE_PB")
            .ok()
            .and_then(|text| text.parse::<usize>().ok())
            .filter(|pb| *pb > 0)
            .unwrap_or(0)
    });
    if *PB > 0 {
        return npanels.min(*PB);
    }
    if npanels > 8 && n <= 128 && m >= 4 * n {
        return npanels.div_ceil(2).max(1);
    }
    PB_PANELS.min(npanels.max(1))
}
/// Max A working set per row band (band_rows x k complex): 4 MB/band
/// passes big-k at 0.90x, 8 MB/band fails at 1.10x (equal task counts).
const NATIVE_BAND_A_CAP: usize = 4 * 1024 * 1024;

/// Tall-narrow A budget per band (see the L3-fit rule in
/// [`sweep_bands_c64`]): ~1.4 MB/band (48 bands at the 64 MB target)
/// keeps four concurrent bands per 16 MB L3 slice resident with
/// headroom for B/C and box noise.
const TALL_A_PER_BAND: usize = 1_400_000;

/// k-chunk depth: 1024 (was 128). Each extra chunk re-reads and
/// re-writes all of C through the general epilogue arm (chunks 1+), and
/// under L3/DRAM contention those C passes dominate: cutting 1024^3
/// from 8 chunks to 1 wins ~16% hot steady-state (22.8 -> 19.2ms, 6/6
/// interleaved iters loaded box) and tall 2048x128x2048 from 16 to 2
/// chunks wins ~5% (11.4 -> 10.8ms, 6/6). The Oct-2 "k=256 overflows
/// L1, costs 256^3 15%" precedent is stale on current code: KC=1024
/// wins 256^3 too (-8%, 2/2) since L2 absorbs the re-reads while the
/// C passes shrink. KC=2048 (single-chunk k=2048) loses tall +5% (2 MB
/// B-blocks thrash contended L3), so the depth caps at 1024.
/// `SONIUM_C64_KC=n` overrides (diagnostic/A-B; read once per process).
const KC: usize = 1024;

/// Effective k-chunk depth (see [`KC`]).
fn kc_now() -> usize {
    static KC_OV: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        std::env::var("SONIUM_C64_KC")
            .ok()
            .and_then(|text| text.parse::<usize>().ok())
            .filter(|kc| *kc > 0)
            .unwrap_or(0)
    });
    if *KC_OV > 0 { *KC_OV } else { KC }
}

/// Sweep C in 2D (row band x panel block) tasks over `scope_pinned`.
/// Each task packs its B-block fused (worker-local) and tiles it; bottom
/// (`m % 6`) rows join as padded tasks over the same blocks.
#[allow(clippy::too_many_arguments)]
fn sweep_bands_c64(
    m: usize,
    n: usize,
    k: usize,
    a: &[Complex64],
    lda: usize,
    b: &[Complex64],
    ldb: usize,
    c: &mut [Complex64],
    ldc: usize,
    alpha: Complex64,
    beta: Complex64,
    trans_b: bool,
    conj_b: bool,
) {
    let full_m = m / MR * MR;
    let full_n = n / NR * NR;
    let macs = m.saturating_mul(n).saturating_mul(k);
    let workers = workers_for_macs(macs.saturating_mul(8));
    // Fat bands (~1 per worker): fused pack redundancy scales with the band
    // count (each band's tasks re-pack all blocks), so thin task-MACS bands
    // (85 x 2 MB on a 1024 cell) lose to one-band-per-worker (17 x 2 MB).
    // Bands stay L2-resident regardless (A-band + B-block <= ~390 KB).
    // Band policy: fused-pack B re-read redundancy scales WITH the band
    // count (256^3: 16 bands = 1.51x, 4 bands = 1.18x), but each band's A
    // working set (band_rows x k complex) must fit its cache share
    // (big-k: 4 MB/band = 0.90x, 8 MB/band = 1.10x at equal task counts).
    // So: bands cover the A-cap AND keep >= 1 task per worker.
    let npanels = n.div_ceil(NR);
    let pb = pb_panels_par(npanels, m, n);
    let nblocks = npanels.div_ceil(pb);
    let a_cap_bands = m
        .saturating_mul(k)
        .saturating_mul(16)
        .div_ceil(NATIVE_BAND_A_CAP)
        .max(1);
    // Two waves of tasks (2x workers): one exact wave (16 zero-slack
    // tasks on 256^3) collapses to the slowest task's makespan; two
    // waves let stealing hide stragglers (4 bands = 1.18x vs 16 bands
    // = 1.51x on 256^3; over-feed only past ~36 tasks). Big-k stays
    // A-cap governed, tiny stays MR-capped, so only mid sizes move.
    let slack_bands = (2 * workers).div_ceil(nblocks).max(1);
    let base_bands = a_cap_bands.max(slack_bands).min(full_m / MR).max(1);
    // Tall-narrow L3-fit rule (same shape gate as [`pb_panels_par`]): the
    // 4 MB A-cap sizes bands for single-worker reuse, but 16 packed
    // workers (4 per 16 MB L3 slice) concurrently demand 4 bands per
    // slice — 16 MB of A alone at 2048x128x2048, thrashing the slice
    // before the second block re-reads it (c32's half-bytes fit, which
    // is why only c64 tall lags). Cap A at ~1.4 MB/band so the
    // concurrent per-slice set fits with headroom for B/C and box
    // noise; the extra bands also re-steal faster around preempted
    // workers (floor shift + tight spread, not just tail-trimming).
    // Clamped to [16, 64]: small talls keep the default, huge ones keep
    // pack bounded. Measured (interleaved 16T same-binary A/B, 10 reps
    // x 6 rounds/arm): 48 bands beats 16 by -9.6% loaded (6/6) and
    // -5.7% quiet (5/6); 32 bands -6.5% loaded (6/6), -1.2% quiet
    // (3/6); 24 bands neutral. 48 also beats 32 head-to-head 11/12.
    // `SONIUM_C64_TBANDS=n` overrides the count exactly (diagnostic/A-B;
    // read per call for same-binary arms).
    let tall_narrow = npanels > 8 && n <= 128 && m >= 4 * n;
    let tbb = std::env::var("SONIUM_C64_TBANDS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0);
    let nbands = if !tall_narrow {
        base_bands
    } else {
        match tbb {
            Some(tb) => tb.min(full_m / MR).max(1),
            None => {
                let a_bands = m
                    .saturating_mul(k)
                    .saturating_mul(16)
                    .div_ceil(TALL_A_PER_BAND)
                    .clamp(16, 64);
                base_bands.max(a_bands).min(full_m / MR).max(1)
            }
        }
    };
    // Round DOWN to a multiple of MR: the sliver band this leaves adds
    // small tasks that fill schedule gaps (rounding up to exact waves
    // starves: 16 zero-slack tasks collapse 256^3 to 2.0x on this box).
    let band_rows = if full_m == 0 {
        m
    } else {
        (full_m / nbands / MR * MR).max(MR)
    };
    // Zero-padded scratch for the bottom rows (shared read-only across
    // bottom tasks; 6 x k complex, at most 24 KB at k <= 256).
    let r_tail = m - full_m;
    let mut apad = vec![Complex64::new(0.0, 0.0); MR * k];
    for rr in 0..r_tail {
        apad[rr * k..(rr + 1) * k]
            .copy_from_slice(&a[(full_m + rr) * lda..(full_m + rr) * lda + k]);
    }
    // Pointers cross the scope as `usize` (raw pointers are not Send);
    // tasks partition C rectangles disjointly so the cast-back is sound.
    let a_ptr = a.as_ptr() as usize;
    let b_ptr = b.as_ptr() as usize;
    let c_ptr = c.as_mut_ptr() as usize;
    let apad_ptr = apad.as_ptr() as usize;
    // SAFETY: main tasks partition rows `0..full_m` x all panels, bottom
    // tasks rows `full_m..m` x all panels; each task touches only its own
    // C rectangle plus shared read-only A/B/scratch. Bounds hold by
    // construction (`r1 <= full_m`, `b1 <= npanels`, `p < k`).
    scope_pinned(|s| {
        let mut r0 = 0;
        while r0 < full_m {
            let r1 = (r0 + band_rows).min(full_m);
            let mut b0 = 0;
            while b0 < npanels {
                let b1 = (b0 + pb).min(npanels);
                s.spawn(move |_| {
                    // SAFETY: see outer comment; per-task (row, panel)
                    // rectangle is exclusive.
                    unsafe {
                        band_c64(
                            r0,
                            r1,
                            n,
                            full_n,
                            b0,
                            b1,
                            k,
                            a_ptr as *const Complex64,
                            lda,
                            b_ptr as *const Complex64,
                            ldb,
                            c_ptr as *mut Complex64,
                            ldc,
                            alpha,
                            beta,
                            trans_b,
                            conj_b,
                        );
                    }
                });
                b0 += pb;
            }
            r0 = r1;
        }
        // Bottom rows (if any) as padded tasks over the same blocks (small
        // per-block bottoms steal-distribute as gap fillers; one fat bottom
        // task occupies a whole worker and tails 256^3).
        if r_tail > 0 {
            let mut b0 = 0;
            while b0 < npanels {
                let b1 = (b0 + pb).min(npanels);
                s.spawn(move |_| {
                    // SAFETY: see outer comment; bottom rows x this block.
                    unsafe {
                        bottom_task_c64(
                            full_m,
                            r_tail,
                            apad_ptr as *const Complex64,
                            n,
                            b0,
                            b1,
                            k,
                            b_ptr as *const Complex64,
                            ldb,
                            c_ptr as *mut Complex64,
                            ldc,
                            alpha,
                            beta,
                            trans_b,
                            conj_b,
                        );
                    }
                });
                b0 += pb;
            }
        }
    });
}

/// One 2D task: rows `r0..r1` (multiple of 6) times panels `b0..b1`,
/// over k-chunks of [`KC`]. Packs each B-block chunk into worker-local
/// bufs first (fused: produced and consumed on the same core), then tiles
/// with `beta` on chunk 0 and accumulate after. Covers the `n % 4` right
/// strip when the block contains the tail panel.
///
/// # Safety
/// `r1 <= full_m`, `r0 % 6 == 0`, `r1 % 6 == 0`, `b1 <= n.div_ceil(4)`,
/// `a` holds `m x lda`, `b` holds `k x ldb`, `c` holds `m x ldc`, and the
/// caller guarantees no other task touches the
/// `(r0..r1) x (b0 * 4..b1 * 4)` C rectangle.
#[allow(clippy::too_many_arguments)]
unsafe fn band_c64(
    r0: usize,
    r1: usize,
    n: usize,
    full_n: usize,
    b0: usize,
    b1: usize,
    k: usize,
    a: *const Complex64,
    lda: usize,
    b: *const Complex64,
    ldb: usize,
    c: *mut Complex64,
    ldc: usize,
    alpha: Complex64,
    beta: Complex64,
    trans_b: bool,
    conj_b: bool,
) {
    debug_assert!(r0.is_multiple_of(MR) && r1.is_multiple_of(MR));
    // SAFETY: `b` spans `k x ldb` (`n x ldb` transposed) per the caller contract.
    let blen = if trans_b { n * ldb } else { k * ldb };
    let bslice = unsafe { std::slice::from_raw_parts(b, blen) };
    let tail_panel = full_n / NR;
    let has_tail = full_n < n && b0 <= tail_panel && tail_panel < b1;
    let mut ck0 = 0;
    let mut first = true;
    while ck0 < k {
        let kc = (k - ck0).min(kc_now());
        let eff_beta = if first {
            beta
        } else {
            Complex64::new(1.0, 0.0)
        };
        with_buf(&NATIVE_BR, (b1 - b0) * kc * NR, |br| {
            with_buf(&NATIVE_BI, (b1 - b0) * kc * NR, |bi| {
                pack_block_c64(n, ck0, kc, bslice, ldb, b0, b1, br, bi, trans_b, conj_b);
                // Rows outer within the block: each row-tile's A chunk
                // stays L1-resident across the block's panels while the
                // B-block re-reads from LOCAL L2 (worker-packed). Panels-
                // outer instead re-reads the SHARED A band per panel
                // (16-32x traffic, possibly cross-NUMA).
                let jc_end = (b1 * NR).min(full_n);
                let mut i = r0;
                while i < r1 {
                    let mut jc = b0 * NR;
                    while jc < jc_end {
                        // SAFETY: rows `i..i+6` (cols `ck0..ck0 + kc`)
                        // valid, panel packed at block-relative
                        // `jc / 4 - b0`, C tile `(i, jc)` inside this
                        // task's exclusive rectangle — all per the caller
                        // contract and loop bounds.
                        unsafe {
                            micro_tile_6x4(
                                a.add(i * lda + ck0),
                                lda,
                                jc / NR - b0,
                                kc,
                                br,
                                bi,
                                c.add(i * ldc + jc),
                                ldc,
                                alpha,
                                eff_beta,
                                true,
                            );
                        }
                        jc += NR;
                    }
                    i += MR;
                }
            })
        });
        // Right strip (tail panel only): tile `(alpha, 0)` per chunk into
        // stack scratch, fold into C per chunk (`beta` on chunk 0).
        if has_tail {
            with_buf(&NATIVE_BR, kc * NR, |br| {
                with_buf(&NATIVE_BI, kc * NR, |bi| {
                    pack_block_c64(
                        n,
                        ck0,
                        kc,
                        bslice,
                        ldb,
                        tail_panel,
                        tail_panel + 1,
                        br,
                        bi,
                        trans_b,
                        conj_b,
                    );
                    let cw = n - full_n;
                    let mut i = r0;
                    while i < r1 {
                        let mut cscratch = [Complex64::new(0.0, 0.0); MR * NR];
                        // SAFETY: rows `i..i+6` (cols `ck0..ck0 + kc`)
                        // valid, tail panel packed at relative 0, scratch
                        // holds 6 x 4 — per the tile contract.
                        unsafe {
                            micro_tile_6x4(
                                a.add(i * lda + ck0),
                                lda,
                                0,
                                kc,
                                br,
                                bi,
                                cscratch.as_mut_ptr(),
                                NR,
                                alpha,
                                Complex64::new(0.0, 0.0),
                                false,
                            );
                            // SAFETY: `cw <= 3` live columns inside this
                            // task's exclusive rectangle; source is scratch.
                            strip_copyback_c64(
                                c.add(i * ldc + full_n),
                                ldc,
                                MR,
                                cw,
                                &cscratch,
                                beta,
                                first,
                            );
                        }
                        i += MR;
                    }
                })
            });
        }
        first = false;
        ck0 += kc;
    }
    // Drain NT-store write-combining buffers before the scope join makes
    // this task's C tiles visible cross-thread (thread sync alone does
    // not drain WC).
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_sfence();
    }
}

/// One bottom-strip 2D task: the `r` live rows (of 6 zero-padded
/// scratch-A rows) times panels `b0..b1`, over k-chunks. Tiles
/// `(alpha, 0)` per chunk into scratch, folds into C per chunk (`beta` on
/// chunk 0).
///
/// # Safety
/// `apad` spans 6 rows of stride `k` (first `r` live); `b` holds `k x ldb`;
/// `c` holds `m x ldc`; the caller guarantees no other task touches rows
/// `full_m..full_m + r` in columns `b0 * 4..b1 * 4`.
#[allow(clippy::too_many_arguments)]
unsafe fn bottom_task_c64(
    full_m: usize,
    r: usize,
    apad: *const Complex64,
    n: usize,
    b0: usize,
    b1: usize,
    k: usize,
    b: *const Complex64,
    ldb: usize,
    c: *mut Complex64,
    ldc: usize,
    alpha: Complex64,
    beta: Complex64,
    trans_b: bool,
    conj_b: bool,
) {
    debug_assert!(r > 0 && r < MR);
    // SAFETY: `b` spans `k x ldb` (`n x ldb` transposed) per the caller contract.
    let blen = if trans_b { n * ldb } else { k * ldb };
    let bslice = unsafe { std::slice::from_raw_parts(b, blen) };
    let mut ck0 = 0;
    let mut first = true;
    while ck0 < k {
        let kc = (k - ck0).min(kc_now());
        with_buf(&NATIVE_BR, (b1 - b0) * kc * NR, |br| {
            with_buf(&NATIVE_BI, (b1 - b0) * kc * NR, |bi| {
                pack_block_c64(n, ck0, kc, bslice, ldb, b0, b1, br, bi, trans_b, conj_b);
                let jc_end = (b1 * NR).min(n);
                let mut jc = b0 * NR;
                while jc < jc_end {
                    let mut cscratch = [Complex64::new(0.0, 0.0); MR * NR];
                    // SAFETY: scratch A holds 6 x k (cols `ck0..ck0 + kc`
                    // live), panel packed at block-relative `jc / 4 - b0`
                    // (tail zero-padded), scratch C holds 6 x 4 — per the
                    // tile contract.
                    unsafe {
                        micro_tile_6x4(
                            apad.add(ck0),
                            k,
                            jc / NR - b0,
                            kc,
                            br,
                            bi,
                            cscratch.as_mut_ptr(),
                            NR,
                            alpha,
                            Complex64::new(0.0, 0.0),
                            false,
                        );
                        let cw = (n - jc).min(NR);
                        // SAFETY: `cw` live columns inside this task's
                        // exclusive rectangle; source is scratch.
                        strip_copyback_c64(
                            c.add(full_m * ldc + jc),
                            ldc,
                            r,
                            cw,
                            &cscratch,
                            beta,
                            first,
                        );
                    }
                    jc += NR;
                }
            })
        });
        first = false;
        ck0 += kc;
    }
    // Drain NT-store write-combining buffers (see `band_c64`).
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_sfence();
    }
}

/// 6x4 micro-tile `tile = alpha * A6xkc * Bpanel + beta * tile`:
/// dispatch to the AVX2 kernel when the CPU has it, else the scalar twin.
/// Pointer-based so edge strips can pass scratch buffers (`a6`/`tile`
/// need not sit inside the source matrices). `streaming` permits
/// non-temporal C stores in the no-C-read arms (overwrite, beta=0):
/// true for tiles written directly to C (skips the write-allocate RFO),
/// false for stack scratch tiles (must stay L1-resident for copyback).
///
/// # Safety
/// `a6` spans 6 valid rows of stride `lda` (elements `r * lda + p` for
/// `r < 6`, `p < k` readable); `br`/`bi` contain panel `panel`
/// (`(panel * k + p) * 4 + jj` in bounds for `p < k`, `jj < 4`); `tile`
/// spans a 6 x 4 block of stride `ldc` (readable unless `beta == 0`),
/// exclusively owned.
#[allow(clippy::too_many_arguments)]
unsafe fn micro_tile_6x4(
    a6: *const Complex64,
    lda: usize,
    panel: usize,
    k: usize,
    br: &[f64],
    bi: &[f64],
    tile: *mut Complex64,
    ldc: usize,
    alpha: Complex64,
    beta: Complex64,
    streaming: bool,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: feature-gated; bounds per the caller contract.
            unsafe {
                micro_tile_6x4_avx2(a6, lda, panel, k, br, bi, tile, ldc, alpha, beta, streaming);
            }
            return;
        }
    }
    // SAFETY: same contract; scalar twin needs no CPU features.
    unsafe {
        micro_tile_6x4_scalar(a6, lda, panel, k, br, bi, tile, ldc, alpha, beta);
    }
}

/// Scalar 6x4 twin over the panel pack (fallback + non-x86_64).
///
/// # Safety
/// Same contract as [`micro_tile_6x4`].
#[allow(clippy::too_many_arguments)]
unsafe fn micro_tile_6x4_scalar(
    a6: *const Complex64,
    lda: usize,
    panel: usize,
    k: usize,
    br: &[f64],
    bi: &[f64],
    tile: *mut Complex64,
    ldc: usize,
    alpha: Complex64,
    beta: Complex64,
) {
    let base = panel * k * NR;
    let overwrite = alpha == Complex64::new(1.0, 0.0) && beta == Complex64::new(0.0, 0.0);
    for r in 0..MR {
        for jj in 0..NR {
            let mut acc = Complex64::new(0.0, 0.0);
            for p in 0..k {
                // SAFETY: `r < 6`, `p < k` per the caller contract.
                let av = unsafe { *a6.add(r * lda + p) };
                let bvr = br[base + p * NR + jj];
                let bvi = bi[base + p * NR + jj];
                acc.re += av.re * bvr - av.im * bvi;
                acc.im += av.re * bvi + av.im * bvr;
            }
            // SAFETY: tile exclusively owned, indices in bounds.
            unsafe {
                let slot = tile.add(r * ldc + jj);
                *slot = if overwrite {
                    acc
                } else if beta == Complex64::new(0.0, 0.0) {
                    // beta == 0 must not read C (poisoned C would
                    // propagate NaN through `0 * *slot`); the AVX2
                    // kernel has the same dedicated arm.
                    alpha * acc
                } else {
                    alpha * acc + beta * *slot
                };
            }
        }
    }
}

/// AVX2 + FMA 6x4 micro-kernel: 12 YMM accumulators (6 re + 6 im) so
/// 12 independent FMA chains cover Zen2's 5-cycle latency (see the
/// `micro_f64_full6` note). The B panel streams forward (stride-1) while
/// each A scalar broadcasts across its row's 4 columns. Register budget:
/// 12 accumulators + 2 B vectors + 2 transient A broadcasts = 16 YMM.
/// Row pointers advance add-only (no imul in the k-loop). Epilogue:
/// overwrite fast path (`alpha == 1`, `beta == 0`, no C read), else the
/// general `alpha * acc + beta * C` arm (C deinterleaved per row; the arm
/// costs ~18 vector ops/row against 3000+ in a full k-chunk).
///
/// # Safety
/// Same contract as [`micro_tile_6x4`], plus the caller must have verified
/// `avx2` + `fma` support (done in [`micro_tile_6x4`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn micro_tile_6x4_avx2(
    a6: *const Complex64,
    lda: usize,
    panel: usize,
    k: usize,
    br: &[f64],
    bi: &[f64],
    tile: *mut Complex64,
    ldc: usize,
    alpha: Complex64,
    beta: Complex64,
    streaming: bool,
) {
    use std::arch::x86_64::*;
    // A as f64 lanes: row `r` starts at `r * lda * 2` past `a6`, advances
    // 2/step.
    // SAFETY: row bases in bounds (6 valid rows per the caller contract).
    let af = a6 as *const f64;
    // Opaque row bases: without the barrier LLVM merges the six row
    // inductions into one dependent `leaq` chain (~5 serial cycles per
    // k-step feeding the broadcasts; read off the asm 2026-10-02, cost
    // ~1.3x on the c32 twin's NC bands). `black_box` keeps six parallel
    // add-chains at zero emitted cost. If this barrier is ever removed,
    // re-check the k-loop asm for chained `leaq`s before believing the
    // rate.
    let (mut ap0, mut ap1, mut ap2, mut ap3, mut ap4, mut ap5) = unsafe {
        (
            std::hint::black_box(af),
            std::hint::black_box(af.add(lda * 2)),
            std::hint::black_box(af.add(2 * lda * 2)),
            std::hint::black_box(af.add(3 * lda * 2)),
            std::hint::black_box(af.add(4 * lda * 2)),
            std::hint::black_box(af.add(5 * lda * 2)),
        )
    };
    let base = panel * k * NR;
    // SAFETY: `base + k * 4 <= br.len()` by the pack contract.
    let mut brp = unsafe { br.as_ptr().add(base) };
    let mut bip = unsafe { bi.as_ptr().add(base) };
    // 12 accumulators: 6 rows x (re/im).
    let mut re0 = _mm256_setzero_pd();
    let mut re1 = _mm256_setzero_pd();
    let mut re2 = _mm256_setzero_pd();
    let mut re3 = _mm256_setzero_pd();
    let mut re4 = _mm256_setzero_pd();
    let mut re5 = _mm256_setzero_pd();
    let mut im0 = _mm256_setzero_pd();
    let mut im1 = _mm256_setzero_pd();
    let mut im2 = _mm256_setzero_pd();
    let mut im3 = _mm256_setzero_pd();
    let mut im4 = _mm256_setzero_pd();
    let mut im5 = _mm256_setzero_pd();
    let mut p = 0;
    while p < k {
        // SAFETY: `brp[0..4]` / `bip[0..4]` in bounds: `p < k` steps of 4
        // from a `k * 4` panel.
        let bv_re = unsafe { _mm256_loadu_pd(brp) };
        let bv_im = unsafe { _mm256_loadu_pd(bip) };
        macro_rules! row {
            ($re:ident, $im:ident, $ap:ident) => {{
                let ar = unsafe { _mm256_broadcast_sd(&*$ap) };
                let ai = unsafe { _mm256_broadcast_sd(&*$ap.add(1)) };
                $ap = unsafe { $ap.add(2) };
                $re = _mm256_fnmadd_pd(ai, bv_im, _mm256_fmadd_pd(ar, bv_re, $re));
                $im = _mm256_fmadd_pd(ai, bv_re, _mm256_fmadd_pd(ar, bv_im, $im));
            }};
        }
        row!(re0, im0, ap0);
        row!(re1, im1, ap1);
        row!(re2, im2, ap2);
        row!(re3, im3, ap3);
        row!(re4, im4, ap4);
        row!(re5, im5, ap5);
        // SAFETY: advance stays in bounds (`p + 1 <= k` steps of 4).
        brp = unsafe { brp.add(NR) };
        bip = unsafe { bip.add(NR) };
        p += 1;
    }
    // Epilogue: interleave re/im lanes into complex pairs (overwrite),
    // the `beta == 0` arm (alpha-scale, no C read), or the general
    // `alpha * acc + beta * C` arm.
    let overwrite = alpha == Complex64::new(1.0, 0.0) && beta == Complex64::new(0.0, 0.0);
    let beta0 = beta == Complex64::new(0.0, 0.0);
    let cf = tile as *mut f64;
    // Hoisted scalar broadcasts for the general arm.
    let var = _mm256_set1_pd(alpha.re);
    let vai = _mm256_set1_pd(alpha.im);
    let vbr = _mm256_set1_pd(beta.re);
    let vbi = _mm256_set1_pd(beta.im);
    macro_rules! row {
        ($re:ident, $im:ident, $rr:expr) => {{
            // SAFETY: row base in bounds (row `rr`, stride `ldc`).
            let dst = unsafe { cf.add(($rr * ldc) * 2) };
            if overwrite {
                let lo = _mm256_unpacklo_pd($re, $im);
                let hi = _mm256_unpackhi_pd($re, $im);
                let ab0 = _mm256_permute2f128_pd(lo, hi, 0x20);
                let ab1 = _mm256_permute2f128_pd(lo, hi, 0x31);
                // SAFETY: tile rows exclusively owned, 4 columns in bounds.
                // NT when streaming+aligned (C never read here): skips the
                // write-allocate RFO (second store inherits alignment:
                // +4 f64 = +32 bytes).
                unsafe {
                    if streaming && (dst as usize) & 31 == 0 {
                        _mm256_stream_pd(dst, ab0);
                        _mm256_stream_pd(dst.add(4), ab1);
                    } else {
                        _mm256_storeu_pd(dst, ab0);
                        _mm256_storeu_pd(dst.add(4), ab1);
                    }
                }
            } else if beta0 {
                // No C read (beta == 0 annihilates it): out = alpha * acc.
                let mut out_re = _mm256_fmadd_pd(var, $re, _mm256_setzero_pd());
                out_re = _mm256_fnmadd_pd(vai, $im, out_re);
                let mut out_im = _mm256_fmadd_pd(var, $im, _mm256_setzero_pd());
                out_im = _mm256_fmadd_pd(vai, $re, out_im);
                let lo = _mm256_unpacklo_pd(out_re, out_im);
                let hi = _mm256_unpackhi_pd(out_re, out_im);
                let ab0 = _mm256_permute2f128_pd(lo, hi, 0x20);
                let ab1 = _mm256_permute2f128_pd(lo, hi, 0x31);
                // SAFETY: same NT rule as the overwrite arm (C unread here).
                unsafe {
                    if streaming && (dst as usize) & 31 == 0 {
                        _mm256_stream_pd(dst, ab0);
                        _mm256_stream_pd(dst.add(4), ab1);
                    } else {
                        _mm256_storeu_pd(dst, ab0);
                        _mm256_storeu_pd(dst.add(4), ab1);
                    }
                }
            } else {
                // SAFETY: tile readable (`beta != 0` or general alpha),
                // rows exclusively owned, 4 columns in bounds.
                let (cr, ci) = unsafe {
                    let ab0 = _mm256_loadu_pd(dst);
                    let ab1 = _mm256_loadu_pd(dst.add(4));
                    let t0 = _mm256_unpacklo_pd(ab0, ab1);
                    let t1 = _mm256_unpackhi_pd(ab0, ab1);
                    (
                        _mm256_permute4x64_pd(t0, 0xD8),
                        _mm256_permute4x64_pd(t1, 0xD8),
                    )
                };
                // out_re = ar*re - ai*im + br*cr - bi*ci
                let mut out_re = _mm256_fmadd_pd(var, $re, _mm256_setzero_pd());
                out_re = _mm256_fnmadd_pd(vai, $im, out_re);
                out_re = _mm256_fmadd_pd(vbr, cr, out_re);
                out_re = _mm256_fnmadd_pd(vbi, ci, out_re);
                // out_im = ar*im + ai*re + br*ci + bi*cr
                let mut out_im = _mm256_fmadd_pd(var, $im, _mm256_setzero_pd());
                out_im = _mm256_fmadd_pd(vai, $re, out_im);
                out_im = _mm256_fmadd_pd(vbr, ci, out_im);
                out_im = _mm256_fmadd_pd(vbi, cr, out_im);
                let lo = _mm256_unpacklo_pd(out_re, out_im);
                let hi = _mm256_unpackhi_pd(out_re, out_im);
                let ab0 = _mm256_permute2f128_pd(lo, hi, 0x20);
                let ab1 = _mm256_permute2f128_pd(lo, hi, 0x31);
                unsafe {
                    _mm256_storeu_pd(dst, ab0);
                    _mm256_storeu_pd(dst.add(4), ab1);
                }
            }
        }};
    }
    row!(re0, im0, 0);
    row!(re1, im1, 1);
    row!(re2, im2, 2);
    row!(re3, im3, 3);
    row!(re4, im4, 4);
    row!(re5, im5, 5);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(
        m: usize,
        n: usize,
        k: usize,
        alpha: Complex64,
        a: &[Complex64],
        b: &[Complex64],
        beta: Complex64,
        c0: &[Complex64],
    ) -> Vec<Complex64> {
        let mut c = vec![Complex64::new(0.0, 0.0); m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = Complex64::new(0.0, 0.0);
                for p in 0..k {
                    acc += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = alpha * acc + beta * c0[i * n + j];
            }
        }
        c
    }

    fn det_data(rows: usize, cols: usize, seed: u64) -> Vec<Complex64> {
        (0..rows * cols)
            .map(|t| {
                let x = (t as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(seed);
                let re = ((x >> 11) % 2000) as f64 / 1000.0 - 1.0;
                let im = ((x >> 27) % 2000) as f64 / 1000.0 - 1.0;
                Complex64::new(re, im)
            })
            .collect()
    }

    fn check_close(got: &[Complex64], want: &[Complex64]) {
        assert_eq!(got.len(), want.len());
        for (t, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            let err = (g - w).norm();
            let tol = 1e-9 * (1.0 + w.norm());
            assert!(err <= tol, "t={t} got={g:?} want={w:?} err={err}");
        }
    }

    /// Serial-sweep correctness over odd shapes (m/n tails), general
    /// scalars, and multi-chunk k.
    #[test]
    fn native_serial_matches_naive() {
        let alpha = Complex64::new(0.7, -0.3);
        let beta = Complex64::new(0.3, 0.1);
        for &(m, n, k) in &[
            (1, 1, 1),
            (7, 5, 9),
            (16, 16, 16),
            (33, 17, 65),
            (5, 40, 7),
            (14, 9, 11),
            (30, 30, 300),
            (45, 50, 260),
        ] {
            let a = det_data(m, k, 11);
            let b = det_data(k, n, 77);
            let c0 = det_data(m, n, 99);
            let want = naive(m, n, k, alpha, &a, &b, beta, &c0);
            let mut got = c0.clone();
            zgemm_native_run(
                m, n, k, alpha, &a, k, &b, n, beta, &mut got, n, false, false,
            );
            check_close(&got, &want);
        }
    }

    /// beta = 0 with general alpha exercises the no-C-read epilogue arm
    /// (overwrite arm needs alpha == 1 too, so this is distinct).
    #[test]
    fn native_beta0_scales_without_c_read() {
        let alpha = Complex64::new(0.7, 0.1);
        let beta = Complex64::new(0.0, 0.0);
        for &(m, n, k) in &[(33, 17, 65), (45, 50, 260), (300, 260, 200)] {
            let a = det_data(m, k, 31);
            let b = det_data(k, n, 32);
            // Poison C: the beta0 arm must not read it (any C read
            // propagates NaN into `got` and fails check_close).
            let c0 = vec![Complex64::new(f64::NAN, f64::NAN); m * n];
            let mut got = c0.clone();
            zgemm_native_run(
                m, n, k, alpha, &a, k, &b, n, beta, &mut got, n, false, false,
            );
            let zeros = vec![Complex64::new(0.0, 0.0); m * n];
            let want = naive(m, n, k, alpha, &a, &b, beta, &zeros);
            check_close(&got, &want);
        }
    }

    /// k = 0 scales C by beta.
    #[test]
    fn native_k0_scales_c() {
        let alpha = Complex64::new(0.7, -0.3);
        let beta = Complex64::new(0.3, 0.1);
        let c0 = det_data(4, 4, 3);
        let mut got = c0.clone();
        zgemm_native_run(
            4,
            4,
            0,
            alpha,
            &[],
            0,
            &[],
            0,
            beta,
            &mut got,
            4,
            false,
            false,
        );
        let want: Vec<Complex64> = c0.iter().map(|&v| beta * v).collect();
        check_close(&got, &want);
    }

    /// Direct-call correctness with padded row strides and general scalars.
    #[test]
    fn native_matches_naive_strided() {
        let alpha = Complex64::new(-1.0, 0.5);
        let beta = Complex64::new(1.0, -0.25);
        let (m, n, k) = (30, 22, 140);
        let (lda, ldb, ldc) = (k + 3, n + 5, n + 7);
        let pack_a: Vec<Complex64> = det_data(m, k, 5);
        let pack_b: Vec<Complex64> = det_data(k, n, 9);
        let pack_c: Vec<Complex64> = det_data(m, n, 13);
        let mut a = vec![Complex64::new(0.0, 0.0); m * lda];
        let mut b = vec![Complex64::new(0.0, 0.0); k * ldb];
        let mut c = vec![Complex64::new(0.0, 0.0); m * ldc];
        for i in 0..m {
            a[i * lda..i * lda + k].copy_from_slice(&pack_a[i * k..(i + 1) * k]);
            c[i * ldc..i * ldc + n].copy_from_slice(&pack_c[i * n..(i + 1) * n]);
        }
        for p in 0..k {
            b[p * ldb..p * ldb + n].copy_from_slice(&pack_b[p * n..(p + 1) * n]);
        }
        let want = naive(m, n, k, alpha, &pack_a, &pack_b, beta, &pack_c);
        zgemm_native_run(
            m, n, k, alpha, &a, lda, &b, ldb, beta, &mut c, ldc, false, false,
        );
        let got: Vec<Complex64> = (0..m)
            .flat_map(|i| c[i * ldc..i * ldc + n].to_vec())
            .collect();
        check_close(&got, &want);
    }

    /// Parallel sweep (above the floor) matches naive on general scalars
    /// with multi-chunk k.
    #[test]
    fn native_parallel_matches_naive() {
        let alpha = Complex64::new(0.7, 0.3);
        let beta = Complex64::new(0.2, -0.4);
        let (m, n, k) = (300usize, 260usize, 200usize);
        assert!(m * n * k >= NATIVE_PAR_MACS);
        let a = det_data(m, k, 21);
        let b = det_data(k, n, 22);
        let c0 = det_data(m, n, 23);
        let want = naive(m, n, k, alpha, &a, &b, beta, &c0);
        let mut got = c0.clone();
        let handled = zgemm_native(
            m, n, k, alpha, &a, k, &b, n, beta, &mut got, n, false, false,
        );
        assert!(handled);
        check_close(&got, &want);
    }

    /// Forced-3M routing (below the min-dim gate, via
    /// `SONIUM_C64_FORCE3M`) matches naive on general scalars: the gate
    /// only selects the algorithm, both arms share the naive contract.
    #[test]
    fn forced_3m_matches_naive() {
        use crate::cpu::parallelism::env_test_lock;
        let _guard = env_test_lock();
        let alpha = Complex64::new(0.7, 0.3);
        let beta = Complex64::new(0.2, -0.4);
        let (m, n, k) = (96usize, 84usize, 72usize);
        assert!(m.min(n).min(k) < C64_3M_MIN_DIM);
        let a = det_data(m, k, 31);
        let b = det_data(k, n, 32);
        let c0 = det_data(m, n, 33);
        let want = naive(m, n, k, alpha, &a, &b, beta, &c0);
        unsafe {
            std::env::set_var("SONIUM_C64_FORCE3M", "1");
        }
        let mut got = c0.clone();
        let handled = zgemm_native(
            m, n, k, alpha, &a, k, &b, n, beta, &mut got, n, false, false,
        );
        unsafe {
            std::env::remove_var("SONIUM_C64_FORCE3M");
        }
        assert!(handled);
        check_close(&got, &want);
    }
}
