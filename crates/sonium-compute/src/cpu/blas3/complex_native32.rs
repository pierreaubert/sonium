//! Native complex-single GEMM micro-kernels (AVX2 + FMA).
//!
//! f32 twin of [`complex_native`](super::complex_native): the same
//! panel-major B-pack and fused band sweep, with a 6x8 micro-tile (an f32
//! YMM lane holds 8 values, so 6 rows x 8 cols uses the same 12
//! accumulators + 2 B vectors + 2 broadcasts = 16 YMM budget as c64's
//! 6x4 tile at twice the flops/step).
//!
//! Layout, f32 per buffer: `br[(jp * k + p) * 8 + jj]`, same for `bi`.
//! 6-wide tiles use pad-and-copy scratch so edge rows/cols run through
//! the SAME kernel (c64 measured a scalar tail regressing parallel 256^3
//! to 3.95x through 4 tail rows).

use num_complex::Complex32;
use std::cell::RefCell;

use super::{scope_pinned, with_buf, workers_for_macs};

/// c32 micro-tile: rows x cols per AVX2 step. 6x8: 12 FMA chains cover
/// Zen2's 5-cycle latency with half the broadcasts/flop of a 6x4 tile.
const MR: usize = 6;
const NR: usize = 8;

/// Parallel sweep only above this many MACs; smaller tiles run the
/// serial sweep. Mirrors the c64 floor (100K beats 9M by 30% on
/// zgetrf-512 there; the sweep shape is identical).
const NATIVE_PAR_MACS_32: usize = 100_000;

thread_local! {
    static NATIVE_BR32: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static NATIVE_BI32: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Gated entry: general `alpha`/`beta`, any `k` (k-chunked), parallel
/// above the floor, serial below. Returns whether the call was handled
/// (mirrors the thin-k convention).
#[allow(clippy::too_many_arguments)]
pub(crate) fn cgemm_native(
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
    trans_b: bool,
    conj_b: bool,
) -> bool {
    cgemm_native_run(
        m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_b, conj_b,
    );
    true
}

/// Native c32 GEMM driver: parallel sweep above the floor, serial below.
/// No transpose support (callers pack first).
///
/// # Panics (debug)
/// Debug-asserts the slice minima for the `m x lda` / `k x ldb` / `m x ldc`
/// footprints.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cgemm_native_run(
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
    if m.saturating_mul(n).saturating_mul(k) >= NATIVE_PAR_MACS_32 {
        sweep_bands_c32(
            m, n, k, a, lda, b, ldb, c, ldc, alpha, beta, trans_b, conj_b,
        );
    } else {
        sweep_serial_c32(
            m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_b, conj_b,
        );
    }
}

/// Serial sweep: one row band (`0..full_m`) plus bottom tasks, run
/// directly on the caller (no scope). Same task fns as the parallel path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sweep_serial_c32(
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
    trans_b: bool,
    conj_b: bool,
) {
    let npanels = n.div_ceil(NR);
    let pb = pb_panels(npanels);
    sweep_blocks_c32(
        m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_b, conj_b, pb,
    );
}

/// Serial sweep with explicit panel blocking: the [`sweep_serial_c32`]
/// body factored so tests can assert `pb`-invariance directly (the
/// production `pb` comes from the read-once [`pb_panels`] gate, which
/// cannot be flipped per test in-process). Every C tile is computed by
/// one `micro_tile_6x8` call however panels group into blocks, so all
/// `pb >= 1` agree bitwise.
#[allow(clippy::too_many_arguments)]
fn sweep_blocks_c32(
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
    trans_b: bool,
    conj_b: bool,
    pb: usize,
) {
    let pb = pb.max(1);
    let full_m = m / MR * MR;
    let full_n = n / NR * NR;
    let npanels = n.div_ceil(NR);
    let r_tail = m - full_m;
    let mut apad = vec![Complex32::new(0.0, 0.0); MR * k];
    for rr in 0..r_tail {
        apad[rr * k..(rr + 1) * k]
            .copy_from_slice(&a[(full_m + rr) * lda..(full_m + rr) * lda + k]);
    }
    // SAFETY: single-threaded; rectangles partition C disjointly by the
    // same construction as the parallel sweep.
    unsafe {
        let mut b0 = 0;
        while b0 < npanels {
            let b1 = (b0 + pb).min(npanels);
            if full_m > 0 {
                band_c32(
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
                bottom_task_c32(
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
            b0 += pb;
        }
    }
}

/// Pack B rows `[k0, k0 + kc)` for panels `[b0, b1)` into worker-local
/// `br`/`bi` at block-relative offsets (`(jp - b0) * kc * 8`). Each task
/// packs the block it consumes (fused: produced + consumed on the same
/// core, so the pack never crosses the interconnect). A partial tail
/// panel in range is zero-padded.
#[allow(clippy::too_many_arguments)]
fn pack_block_c32(
    n: usize,
    k0: usize,
    kc: usize,
    b: &[Complex32],
    ldb: usize,
    b0: usize,
    b1: usize,
    br: &mut [f32],
    bi: &mut [f32],
    trans_b: bool,
    conj_b: bool,
) {
    debug_assert!(br.len() >= (b1 - b0) * kc * NR);
    debug_assert!(bi.len() >= (b1 - b0) * kc * NR);
    if trans_b {
        pack_block_c32_trans(n, k0, kc, b, ldb, b0, b1, br, bi, conj_b);
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
            // SAFETY: `jc + 8 <= n` (full panel), rows `k0..k0 + kc`
            // readable, `base + kc * 8` in bounds by the caller
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

/// Transposed half of [`pack_block_c32`]: same `br`/`bi` layout, but
/// element `(p, j)` reads `b[j*ldb + k0+p]` (`B` is `n × ldb`).
/// `conj_b` negates the imaginary part (`ConjTrans`).
#[allow(clippy::too_many_arguments)]
fn pack_block_c32_trans(
    n: usize,
    k0: usize,
    kc: usize,
    b: &[Complex32],
    ldb: usize,
    b0: usize,
    b1: usize,
    br: &mut [f32],
    bi: &mut [f32],
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

/// Strip copyback: fold a 6x8 scratch tile's live `r x cw` corner into C.
/// First chunk applies `beta` (`C = S + beta * C`, plain copy when
/// `beta == 0`); later chunks accumulate (`C += S`).
///
/// # Safety
/// `dst` spans `r` rows of stride `ldc` (`cw <= 8` columns each, writable,
/// exclusively owned); `scratch` holds the 6 x 8 tile.
#[allow(clippy::too_many_arguments)]
unsafe fn strip_copyback_c32(
    dst: *mut Complex32,
    ldc: usize,
    r: usize,
    cw: usize,
    scratch: &[Complex32; MR * NR],
    beta: Complex32,
    first: bool,
) {
    debug_assert!(r <= MR && cw <= NR);
    let apply_beta = first && beta != Complex32::new(0.0, 0.0);
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

/// AVX2 deinterleave pack of 8 panel columns: per depth `p`, two 32-byte
/// loads of 8 complex, `shuffle` (0x88/0xDD pick re/im within each
/// 128-bit lane) + `permutevar8x32` (lane fix `[0,1,4,5,2,3,6,7]`) into
/// re/im vectors, two stores (~8 vector ops per 64 B).
///
/// # Safety
/// `bcol` points at B column `jc` (`jc + 8 <= n`, `k` rows of stride
/// `ldb` readable); `br`/`bi` hold `base + (k - 1) * stride + 8`; caller
/// verified AVX2+FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pack_panel_avx2(
    k: usize,
    bcol: *const Complex32,
    ldb: usize,
    br: &mut [f32],
    bi: &mut [f32],
    base: usize,
    stride: usize,
) {
    use std::arch::x86_64::*;
    // Lane fix for the shuffle output `[r0,r1,r4,r5,r2,r3,r6,r7]`.
    let idx = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
    let bf = bcol as *const f32;
    for p in 0..k {
        // SAFETY: row `p` of the 8 live columns in bounds per the contract.
        let v0 = unsafe { _mm256_loadu_ps(bf.add(p * ldb * 2)) };
        let v1 = unsafe { _mm256_loadu_ps(bf.add(p * ldb * 2 + 8)) };
        let rev = _mm256_permutevar8x32_ps(_mm256_shuffle_ps(v0, v1, 0x88), idx);
        let imv = _mm256_permutevar8x32_ps(_mm256_shuffle_ps(v0, v1, 0xDD), idx);
        let dst = base + p * stride;
        // SAFETY: `dst + 8` in bounds per the contract.
        unsafe {
            _mm256_storeu_ps(br.as_mut_ptr().add(dst), rev);
            _mm256_storeu_ps(bi.as_mut_ptr().add(dst), imv);
        }
    }
}

/// Panels per 2D task block, by panel count: narrow shapes (<= 8 panels,
/// e.g. 64^3) use one 8-panel block (few tasks: scope+spawn is 80% of a
/// 30us cell), wider shapes use 2-panel blocks (more blocks -> fewer
/// bands at the same task count -> ~1x fused-pack redundancy on 256^3).
/// Serial sweep only; the parallel sweep sizes blocks for its worker
/// count (see [`pb_panels_par`]).
/// Serial-sweep panels per n-block: 8 wide (matches c64 `PB_PANELS`;
/// narrow shapes keep their single block via the `min`). The old wide
/// value 2 ran 60 n-blocks per 960-column SYRK band (4 TLS borrows +
/// pack call + sweep setup each, ~1ms of block overhead per 976x48
/// HERK cell serially); 8 cuts that 4x while the B-block (2 bufs x 8
/// panels x kc x 8 f32 = 24KB at kc=48) stays L1-resident. Pure
/// regrouping: every C tile is still computed by one `micro_tile_6x8`
/// call, so results are bitwise identical at any `pb`.
/// `SONIUM_C32_PB=n` overrides (diagnostic/A-B; read once per process —
/// unlike the per-call gates, this sits below the per-band serial entry
/// at ~60 calls/cell, where a `getenv` per call would cost ~3%).
/// Values < 1 fall back to 8.
fn pb_panels(npanels: usize) -> usize {
    static PB: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        std::env::var("SONIUM_C32_PB")
            .ok()
            .and_then(|text| text.parse::<usize>().ok())
            .filter(|pb| *pb > 0)
            .unwrap_or(8)
    });
    npanels.min(*PB)
}

/// Parallel-sweep panels per 2D task block: narrow shapes (<= 8 panels)
/// keep one block (minimal dispatch, as serial), wider shapes cap the
/// block count at ~2x workers (two waves of mains plus their bottom
/// gap-fillers). Fixed pb=2 spawns 8-16x workers on wide shapes (128+
/// tasks on a 158us 128x1024x64 cell) and the sequential spawn section
/// plus steal/join volume dominate the wall (ST ties OB 0.97x while MT
/// loses 1.28x on identical kernels). Bigger blocks also halve the
/// shared-A re-reads per doubling (each block streams the full A band).
/// B-pack redundancy rises with the band count the slack policy adds as
/// blocks shrink, but the pack is ~1% — spawns are the wall.
///
/// Narrow-tall exception (mirrors c64): `m >= 4n`, `n <= 128` shapes
/// take two blocks (`pb = npanels / 2`, B-block ~128 KB). The 2W cap
/// over-splits them (2048x128x2048: 8 blocks = 8x32 MB of A re-reads);
/// two blocks halve that traffic while the band slack policy keeps two
/// waves of tasks (16 bands x 2 blocks); measured 0.94x interleaved
/// 4/4 there. Wider-n talls (n = 256) and squares keep the legacy cap
/// (small blocks win there).
///
/// Fat-square exception: `512 <= m == n <= 1024` with `k` in the same
/// box take two-panel blocks (a 4W cap, like the tall rule's coarser
/// sibling in the other direction: small B-blocks win here). The 2W
/// cap under-splits mid squares (1024^3: pb 4 -> 2 wins 0.945x
/// interleaved x3; 768^3: pb 3 -> 2 wins 0.96x x3 — L1D-fit B-blocks
/// at 32 KB). Squares-only: rect-fat 2048x1024 loses 1.03x at the
/// 4W width, and 2048^3 loses 1.015x (hence the 1024 cap); 512^3 is
/// a no-op (both rules yield 2).
/// `SONIUM_CNATIVE_PB=n` overrides (diagnostic/A-B; shared with c64).
fn pb_panels_par(npanels: usize, workers: usize, m: usize, n: usize, k: usize) -> usize {
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
    if npanels <= 8 {
        return 8;
    }
    if n <= 128 && m >= 4 * n {
        return npanels.div_ceil(2).max(1);
    }
    if m == n && (512..=1024).contains(&m) && (512..=1024).contains(&k) {
        let target = (4 * workers).max(1);
        return npanels.div_ceil(target).max(2);
    }
    let target = (2 * workers).max(1);
    npanels.div_ceil(target).max(2)
}
/// Max A working set per row band (band_rows x k complex): mirrors the
/// c64 4 MB cap (4 MB/band passes big-k there at equal task counts).
const NATIVE_BAND_A_CAP: usize = 4 * 1024 * 1024;

/// k-chunk depth: 1024 (was 256). Each extra chunk re-reads and
/// re-writes all of C through the general epilogue arm (chunks 1+), and
/// under L3/DRAM contention those C passes dominate: cutting 1024^3
/// from 4 chunks to 1 wins ~11% (6.6 -> 5.9ms, 2/2 interleaved rounds
/// plus a monotonic 512 midpoint at 6.2ms), while tall 2048x128x2048
/// (8 -> 2 chunks) is flat at ~+1% (3.50 -> 3.53ms, inside cell
/// noise). Same mechanism as the banked c64 KC=1024 win. The B-block
/// (2 bufs x pb panels x kc x 8 f32 = 128 KB at pb=2, 512 KB at pb=8)
/// and the A 6-row chunk (49 KB) spill out of L1D but stay L2-resident
/// (512 KB/core), which absorbs the re-reads. KC=2048 loses tall +2.5%
/// (1 MB B-blocks spill L2), so the depth caps at 1024.
/// `SONIUM_C32_KC=n` overrides (diagnostic/A-B; read once per process).
const KC: usize = 1024;

/// Effective k-chunk depth (see [`KC`]).
fn kc_now() -> usize {
    static KC_OV: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        std::env::var("SONIUM_C32_KC")
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
fn sweep_bands_c32(
    m: usize,
    n: usize,
    k: usize,
    a: &[Complex32],
    lda: usize,
    b: &[Complex32],
    ldb: usize,
    c: &mut [Complex32],
    ldc: usize,
    alpha: Complex32,
    beta: Complex32,
    trans_b: bool,
    conj_b: bool,
) {
    let full_m = m / MR * MR;
    let full_n = n / NR * NR;
    let macs = m.saturating_mul(n).saturating_mul(k);
    let workers = workers_for_macs(macs.saturating_mul(8));
    // Band policy (mirrors c64): fused-pack B re-read redundancy scales
    // WITH the band count, but each band's A working set must fit its
    // cache share. So: bands cover the A-cap AND keep >= 1 task/worker.
    let npanels = n.div_ceil(NR);
    let pb = pb_panels_par(npanels, workers, m, n, k);
    let nblocks = npanels.div_ceil(pb);
    let a_cap_bands = m
        .saturating_mul(k)
        .saturating_mul(8)
        .div_ceil(NATIVE_BAND_A_CAP)
        .max(1);
    // Two waves of tasks (2x workers): one exact wave (16 zero-slack
    // tasks on 256^3) collapses to the slowest task's makespan with 12+
    // workers idle; two waves let stealing hide stragglers. Big-k stays
    // A-cap governed, tiny stays MR-capped, so only mid sizes move.
    let slack_bands = (2 * workers).div_ceil(nblocks).max(1);
    let nbands = a_cap_bands.max(slack_bands).min(full_m / MR).max(1);
    // Round DOWN to a multiple of MR: the sliver band this leaves adds
    // small tasks that fill schedule gaps (exact-wave schedules with
    // zero slack collapse on this box).
    let band_rows = if full_m == 0 {
        m
    } else {
        (full_m / nbands / MR * MR).max(MR)
    };
    // Zero-padded scratch for the bottom rows (shared read-only across
    // bottom tasks; 6 x k complex32, at most 12 KB at k <= 256).
    let r_tail = m - full_m;
    let mut apad = vec![Complex32::new(0.0, 0.0); MR * k];
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
                        band_c32(
                            r0,
                            r1,
                            n,
                            full_n,
                            b0,
                            b1,
                            k,
                            a_ptr as *const Complex32,
                            lda,
                            b_ptr as *const Complex32,
                            ldb,
                            c_ptr as *mut Complex32,
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
                        bottom_task_c32(
                            full_m,
                            r_tail,
                            apad_ptr as *const Complex32,
                            n,
                            b0,
                            b1,
                            k,
                            b_ptr as *const Complex32,
                            ldb,
                            c_ptr as *mut Complex32,
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
/// with `beta` on chunk 0 and accumulate after. Covers the `n % 8` right
/// strip when the block contains the tail panel.
///
/// # Safety
/// `r1 <= full_m`, `r0 % 6 == 0`, `r1 % 6 == 0`, `b1 <= n.div_ceil(8)`,
/// `a` holds `m x lda`, `b` holds `k x ldb`, `c` holds `m x ldc`, and the
/// caller guarantees no other task touches the
/// `(r0..r1) x (b0 * 8..b1 * 8)` C rectangle.
#[allow(clippy::too_many_arguments)]
unsafe fn band_c32(
    r0: usize,
    r1: usize,
    n: usize,
    full_n: usize,
    b0: usize,
    b1: usize,
    k: usize,
    a: *const Complex32,
    lda: usize,
    b: *const Complex32,
    ldb: usize,
    c: *mut Complex32,
    ldc: usize,
    alpha: Complex32,
    beta: Complex32,
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
            Complex32::new(1.0, 0.0)
        };
        with_buf(&NATIVE_BR32, (b1 - b0) * kc * NR, |br| {
            with_buf(&NATIVE_BI32, (b1 - b0) * kc * NR, |bi| {
                pack_block_c32(n, ck0, kc, bslice, ldb, b0, b1, br, bi, trans_b, conj_b);
                // Rows outer within the block: each row-tile's A chunk
                // stays L1-resident across the block's panels while the
                // B-block re-reads from LOCAL L2 (worker-packed). Panels-
                // outer instead re-reads the SHARED A band per panel
                // (16-32x traffic, possibly cross-NUMA); rows-outer cuts
                // shared A traffic to ~1x per chunk with better (row-major
                // sequential) C locality.
                let jc_end = (b1 * NR).min(full_n);
                let mut i = r0;
                while i < r1 {
                    let mut jc = b0 * NR;
                    while jc < jc_end {
                        // SAFETY: rows `i..i+6` (cols `ck0..ck0 + kc`)
                        // valid, panel packed at block-relative
                        // `jc / 8 - b0`, C tile `(i, jc)` inside this
                        // task's exclusive rectangle — all per the caller
                        // contract and loop bounds.
                        unsafe {
                            micro_tile_6x8(
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
            with_buf(&NATIVE_BR32, kc * NR, |br| {
                with_buf(&NATIVE_BI32, kc * NR, |bi| {
                    pack_block_c32(
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
                        let mut cscratch = [Complex32::new(0.0, 0.0); MR * NR];
                        // SAFETY: rows `i..i+6` (cols `ck0..ck0 + kc`)
                        // valid, tail panel packed at relative 0, scratch
                        // holds 6 x 8 — per the tile contract.
                        unsafe {
                            micro_tile_6x8(
                                a.add(i * lda + ck0),
                                lda,
                                0,
                                kc,
                                br,
                                bi,
                                cscratch.as_mut_ptr(),
                                NR,
                                alpha,
                                Complex32::new(0.0, 0.0),
                                false,
                            );
                            // SAFETY: `cw <= 7` live columns inside this
                            // task's exclusive rectangle; source is scratch.
                            strip_copyback_c32(
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
/// `full_m..full_m + r` in columns `b0 * 8..b1 * 8`.
#[allow(clippy::too_many_arguments)]
unsafe fn bottom_task_c32(
    full_m: usize,
    r: usize,
    apad: *const Complex32,
    n: usize,
    b0: usize,
    b1: usize,
    k: usize,
    b: *const Complex32,
    ldb: usize,
    c: *mut Complex32,
    ldc: usize,
    alpha: Complex32,
    beta: Complex32,
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
        with_buf(&NATIVE_BR32, (b1 - b0) * kc * NR, |br| {
            with_buf(&NATIVE_BI32, (b1 - b0) * kc * NR, |bi| {
                pack_block_c32(n, ck0, kc, bslice, ldb, b0, b1, br, bi, trans_b, conj_b);
                let jc_end = (b1 * NR).min(n);
                let mut jc = b0 * NR;
                while jc < jc_end {
                    let mut cscratch = [Complex32::new(0.0, 0.0); MR * NR];
                    // SAFETY: scratch A holds 6 x k (cols `ck0..ck0 + kc`
                    // live), panel packed at block-relative `jc / 8 - b0`
                    // (tail zero-padded), scratch C holds 6 x 8 — per the
                    // tile contract.
                    unsafe {
                        micro_tile_6x8(
                            apad.add(ck0),
                            k,
                            jc / NR - b0,
                            kc,
                            br,
                            bi,
                            cscratch.as_mut_ptr(),
                            NR,
                            alpha,
                            Complex32::new(0.0, 0.0),
                            false,
                        );
                        let cw = (n - jc).min(NR);
                        // SAFETY: `cw` live columns inside this task's
                        // exclusive rectangle; source is scratch.
                        strip_copyback_c32(
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
    // Drain NT-store write-combining buffers (see `band_c32`).
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_sfence();
    }
}

/// 6x8 micro-tile `tile = alpha * A6xkc * Bpanel + beta * tile`:
/// dispatch to the AVX2 kernel when the CPU has it, else the scalar twin.
/// Pointer-based so edge strips can pass scratch buffers (`a6`/`tile`
/// need not sit inside the source matrices). `streaming` permits
/// non-temporal C stores in the no-C-read arms (overwrite, beta=0):
/// true for tiles written directly to C (skips the write-allocate RFO,
/// which costs ~100us on a far-NUMA 256^3 C), false for stack scratch
/// tiles (must stay L1-resident for the scalar copyback).
///
/// # Safety
/// `a6` spans 6 valid rows of stride `lda` (elements `r * lda + p` for
/// `r < 6`, `p < k` readable); `br`/`bi` contain panel `panel`
/// (`(panel * k + p) * 8 + jj` in bounds for `p < k`, `jj < 8`); `tile`
/// spans a 6 x 8 block of stride `ldc` (readable unless `beta == 0`),
/// exclusively owned.
#[allow(clippy::too_many_arguments)]
unsafe fn micro_tile_6x8(
    a6: *const Complex32,
    lda: usize,
    panel: usize,
    k: usize,
    br: &[f32],
    bi: &[f32],
    tile: *mut Complex32,
    ldc: usize,
    alpha: Complex32,
    beta: Complex32,
    streaming: bool,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: feature-gated; bounds per the caller contract.
            unsafe {
                micro_tile_6x8_avx2(a6, lda, panel, k, br, bi, tile, ldc, alpha, beta, streaming);
            }
            return;
        }
    }
    // SAFETY: same contract; scalar twin needs no CPU features.
    unsafe {
        micro_tile_6x8_scalar(a6, lda, panel, k, br, bi, tile, ldc, alpha, beta);
    }
}

/// Scalar 6x8 twin over the panel pack (fallback + non-x86_64).
///
/// # Safety
/// Same contract as [`micro_tile_6x8`].
#[allow(clippy::too_many_arguments)]
unsafe fn micro_tile_6x8_scalar(
    a6: *const Complex32,
    lda: usize,
    panel: usize,
    k: usize,
    br: &[f32],
    bi: &[f32],
    tile: *mut Complex32,
    ldc: usize,
    alpha: Complex32,
    beta: Complex32,
) {
    let base = panel * k * NR;
    let overwrite = alpha == Complex32::new(1.0, 0.0) && beta == Complex32::new(0.0, 0.0);
    for r in 0..MR {
        for jj in 0..NR {
            let mut acc = Complex32::new(0.0, 0.0);
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
                } else if beta == Complex32::new(0.0, 0.0) {
                    // beta == 0 must not read C; see the c64 twin.
                    alpha * acc
                } else {
                    alpha * acc + beta * *slot
                };
            }
        }
    }
}

/// AVX2 + FMA 6x8 micro-kernel: 12 YMM accumulators (6 re + 6 im) so
/// 12 independent FMA chains cover Zen2's 5-cycle latency. The B panel
/// streams forward (stride-1) while each A scalar broadcasts across its
/// row's 8 columns. Register budget: 12 accumulators + 2 B vectors + 2
/// transient A broadcasts = 16 YMM. Row pointers advance add-only (no
/// imul in the k-loop). Epilogue: overwrite fast path (`alpha == 1`,
/// `beta == 0`, no C read), else the general `alpha * acc + beta * C`
/// arm (C deinterleaved per row via shuffle + permutevar).
///
/// # Safety
/// Same contract as [`micro_tile_6x8`], plus the caller must have verified
/// `avx2` + `fma` support (done in [`micro_tile_6x8`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn micro_tile_6x8_avx2(
    a6: *const Complex32,
    lda: usize,
    panel: usize,
    k: usize,
    br: &[f32],
    bi: &[f32],
    tile: *mut Complex32,
    ldc: usize,
    alpha: Complex32,
    beta: Complex32,
    streaming: bool,
) {
    use std::arch::x86_64::*;
    // A as f32 lanes: row `r` starts at `r * lda * 2` past `a6`, advances
    // 2/step.
    // SAFETY: row bases in bounds (6 valid rows per the caller contract).
    let af = a6 as *const f32;
    // Opaque row bases: without the barrier LLVM merges the six row
    // inductions into one dependent `leaq` chain (~5 serial cycles per
    // k-step feeding the broadcasts; read off the asm 2026-10-02, cost
    // ~1.3x on NC bands). `black_box` keeps six parallel add-chains at
    // zero emitted cost. If this barrier is ever removed, re-check the
    // k-loop asm for chained `leaq`s before believing the rate.
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
    // SAFETY: `base + k * 8 <= br.len()` by the pack contract.
    let mut brp = unsafe { br.as_ptr().add(base) };
    let mut bip = unsafe { bi.as_ptr().add(base) };
    // 12 accumulators: 6 rows x (re/im).
    let mut re0 = _mm256_setzero_ps();
    let mut re1 = _mm256_setzero_ps();
    let mut re2 = _mm256_setzero_ps();
    let mut re3 = _mm256_setzero_ps();
    let mut re4 = _mm256_setzero_ps();
    let mut re5 = _mm256_setzero_ps();
    let mut im0 = _mm256_setzero_ps();
    let mut im1 = _mm256_setzero_ps();
    let mut im2 = _mm256_setzero_ps();
    let mut im3 = _mm256_setzero_ps();
    let mut im4 = _mm256_setzero_ps();
    let mut im5 = _mm256_setzero_ps();
    // Single-step k-loop, deliberately not manually unrolled: x4's
    // 96-FMA body overflows the backend scheduler window (RS/ROB) and
    // runs 2x slower than x1 on Zen2 (256^3: 0.41 vs 0.26 unmasked).
    let mut p = 0;
    while p < k {
        // SAFETY: `brp[0..8]` / `bip[0..8]` in bounds: `p < k` steps of 8
        // from a `k * 8` panel.
        let bv_re = unsafe { _mm256_loadu_ps(brp) };
        let bv_im = unsafe { _mm256_loadu_ps(bip) };
        macro_rules! row {
            ($re:ident, $im:ident, $ap:ident) => {{
                let ar = unsafe { _mm256_broadcast_ss(&*$ap) };
                let ai = unsafe { _mm256_broadcast_ss(&*$ap.add(1)) };
                $ap = unsafe { $ap.add(2) };
                $re = _mm256_fnmadd_ps(ai, bv_im, _mm256_fmadd_ps(ar, bv_re, $re));
                $im = _mm256_fmadd_ps(ai, bv_re, _mm256_fmadd_ps(ar, bv_im, $im));
            }};
        }
        row!(re0, im0, ap0);
        row!(re1, im1, ap1);
        row!(re2, im2, ap2);
        row!(re3, im3, ap3);
        row!(re4, im4, ap4);
        row!(re5, im5, ap5);
        // SAFETY: advance stays in bounds (`p + 1 <= k` steps of 8).
        brp = unsafe { brp.add(NR) };
        bip = unsafe { bip.add(NR) };
        p += 1;
    }
    // Epilogue: interleave re/im lanes into complex pairs (overwrite),
    // the `beta == 0` arm (alpha-scale, no C read), or the general
    // `alpha * acc + beta * C` arm.
    let overwrite = alpha == Complex32::new(1.0, 0.0) && beta == Complex32::new(0.0, 0.0);
    let beta0 = beta == Complex32::new(0.0, 0.0);
    // Schur arms (LU/QR/Cholesky trailing updates): real `alpha = ±1`,
    // `beta = 1` folds to one add/subtract per component (see the c64
    // twin in `complex_native`).
    let schur_add = alpha == Complex32::new(1.0, 0.0) && beta == Complex32::new(1.0, 0.0);
    let schur_sub = alpha == Complex32::new(-1.0, 0.0) && beta == Complex32::new(1.0, 0.0);
    let cf = tile as *mut f32;
    // Hoisted scalar broadcasts for the general arm, plus the C-read
    // deinterleave lane fix (same `[0,1,4,5,2,3,6,7]` as the pack).
    let var = _mm256_set1_ps(alpha.re);
    let vai = _mm256_set1_ps(alpha.im);
    let vbr = _mm256_set1_ps(beta.re);
    let vbi = _mm256_set1_ps(beta.im);
    let idx = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
    macro_rules! row {
        ($re:ident, $im:ident, $rr:expr) => {{
            // SAFETY: row base in bounds (row `rr`, stride `ldc`).
            let dst = unsafe { cf.add(($rr * ldc) * 2) };
            if overwrite {
                let lo = _mm256_unpacklo_ps($re, $im);
                let hi = _mm256_unpackhi_ps($re, $im);
                let ab0 = _mm256_permute2f128_ps(lo, hi, 0x20);
                let ab1 = _mm256_permute2f128_ps(lo, hi, 0x31);
                // SAFETY: tile rows exclusively owned, 8 columns in bounds.
                // NT when streaming+aligned (C never read here): skips the
                // write-allocate RFO (second store inherits alignment:
                // +8 f32 = +32 bytes).
                unsafe {
                    if streaming && (dst as usize) & 31 == 0 {
                        _mm256_stream_ps(dst, ab0);
                        _mm256_stream_ps(dst.add(8), ab1);
                    } else {
                        _mm256_storeu_ps(dst, ab0);
                        _mm256_storeu_ps(dst.add(8), ab1);
                    }
                }
            } else if beta0 {
                // No C read (beta == 0 annihilates it): out = alpha * acc.
                // Saves the C-stream + deinterleave (measured -12% on
                // 256^3 general-vs-overwrite delta).
                let mut out_re = _mm256_fmadd_ps(var, $re, _mm256_setzero_ps());
                out_re = _mm256_fnmadd_ps(vai, $im, out_re);
                let mut out_im = _mm256_fmadd_ps(var, $im, _mm256_setzero_ps());
                out_im = _mm256_fmadd_ps(vai, $re, out_im);
                let lo = _mm256_unpacklo_ps(out_re, out_im);
                let hi = _mm256_unpackhi_ps(out_re, out_im);
                let ab0 = _mm256_permute2f128_ps(lo, hi, 0x20);
                let ab1 = _mm256_permute2f128_ps(lo, hi, 0x31);
                // SAFETY: same NT rule as the overwrite arm (C unread here).
                unsafe {
                    if streaming && (dst as usize) & 31 == 0 {
                        _mm256_stream_ps(dst, ab0);
                        _mm256_stream_ps(dst.add(8), ab1);
                    } else {
                        _mm256_storeu_ps(dst, ab0);
                        _mm256_storeu_ps(dst.add(8), ab1);
                    }
                }
            } else if schur_add || schur_sub {
                // Interleaved merge (see the c64 twin): interleave the
                // accumulator and add/subtract against C as stored,
                // skipping the C deinterleave.
                let lo = _mm256_unpacklo_ps($re, $im);
                let hi = _mm256_unpackhi_ps($re, $im);
                let ab0 = _mm256_permute2f128_ps(lo, hi, 0x20);
                let ab1 = _mm256_permute2f128_ps(lo, hi, 0x31);
                // SAFETY: tile readable (`beta == 1`), rows exclusively
                // owned, 8 columns in bounds.
                unsafe {
                    let c0 = _mm256_loadu_ps(dst);
                    let c1 = _mm256_loadu_ps(dst.add(8));
                    let (o0, o1) = if schur_add {
                        (_mm256_add_ps(c0, ab0), _mm256_add_ps(c1, ab1))
                    } else {
                        (_mm256_sub_ps(c0, ab0), _mm256_sub_ps(c1, ab1))
                    };
                    _mm256_storeu_ps(dst, o0);
                    _mm256_storeu_ps(dst.add(8), o1);
                }
            } else {
                // SAFETY: tile readable (`beta != 0` or general alpha),
                // rows exclusively owned, 8 columns in bounds.
                let (cr, ci) = unsafe {
                    let ab0 = _mm256_loadu_ps(dst);
                    let ab1 = _mm256_loadu_ps(dst.add(8));
                    (
                        _mm256_permutevar8x32_ps(_mm256_shuffle_ps(ab0, ab1, 0x88), idx),
                        _mm256_permutevar8x32_ps(_mm256_shuffle_ps(ab0, ab1, 0xDD), idx),
                    )
                };
                // out_re = ar*re - ai*im + br*cr - bi*ci
                let mut out_re = _mm256_fmadd_ps(var, $re, _mm256_setzero_ps());
                out_re = _mm256_fnmadd_ps(vai, $im, out_re);
                out_re = _mm256_fmadd_ps(vbr, cr, out_re);
                out_re = _mm256_fnmadd_ps(vbi, ci, out_re);
                // out_im = ar*im + ai*re + br*ci + bi*cr
                let mut out_im = _mm256_fmadd_ps(var, $im, _mm256_setzero_ps());
                out_im = _mm256_fmadd_ps(vai, $re, out_im);
                out_im = _mm256_fmadd_ps(vbr, ci, out_im);
                out_im = _mm256_fmadd_ps(vbi, cr, out_im);
                let lo = _mm256_unpacklo_ps(out_re, out_im);
                let hi = _mm256_unpackhi_ps(out_re, out_im);
                let ab0 = _mm256_permute2f128_ps(lo, hi, 0x20);
                let ab1 = _mm256_permute2f128_ps(lo, hi, 0x31);
                unsafe {
                    _mm256_storeu_ps(dst, ab0);
                    _mm256_storeu_ps(dst.add(8), ab1);
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
        alpha: Complex32,
        a: &[Complex32],
        b: &[Complex32],
        beta: Complex32,
        c0: &[Complex32],
    ) -> Vec<Complex32> {
        let mut c = vec![Complex32::new(0.0, 0.0); m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = Complex32::new(0.0, 0.0);
                for p in 0..k {
                    acc += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = alpha * acc + beta * c0[i * n + j];
            }
        }
        c
    }

    fn det_data(rows: usize, cols: usize, seed: u64) -> Vec<Complex32> {
        (0..rows * cols)
            .map(|t| {
                let x = (t as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(seed);
                let re = ((x >> 11) % 2000) as f32 / 1000.0 - 1.0;
                let im = ((x >> 27) % 2000) as f32 / 1000.0 - 1.0;
                Complex32::new(re, im)
            })
            .collect()
    }

    fn check_close(got: &[Complex32], want: &[Complex32]) {
        assert_eq!(got.len(), want.len());
        for (t, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            let err = (g - w).norm();
            // f32 blocked accumulation vs serial naive over k <= 300 terms.
            let tol = 1e-4 * (1.0 + w.norm());
            assert!(err <= tol, "t={t} got={g:?} want={w:?} err={err}");
        }
    }

    /// Serial-sweep correctness over odd shapes (m/n tails), general
    /// scalars, and multi-chunk k.
    #[test]
    fn native_serial_matches_naive() {
        let alpha = Complex32::new(0.7, -0.3);
        let beta = Complex32::new(0.3, 0.1);
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
            cgemm_native_run(
                m, n, k, alpha, &a, k, &b, n, beta, &mut got, n, false, false,
            );
            check_close(&got, &want);
        }
    }

    /// beta = 0 with general alpha exercises the no-C-read epilogue arm
    /// (overwrite arm needs alpha == 1 too, so this is distinct).
    #[test]
    fn native_beta0_scales_without_c_read() {
        let alpha = Complex32::new(0.7, 0.1);
        let beta = Complex32::new(0.0, 0.0);
        for &(m, n, k) in &[(33, 17, 65), (45, 50, 260), (300, 260, 200)] {
            let a = det_data(m, k, 31);
            let b = det_data(k, n, 32);
            // Poison C: the beta0 arm must not read it (any C read
            // propagates NaN into `got` and fails check_close).
            let c0 = vec![Complex32::new(f32::NAN, f32::NAN); m * n];
            let mut got = c0.clone();
            cgemm_native_run(
                m, n, k, alpha, &a, k, &b, n, beta, &mut got, n, false, false,
            );
            let zeros = vec![Complex32::new(0.0, 0.0); m * n];
            let want = naive(m, n, k, alpha, &a, &b, beta, &zeros);
            check_close(&got, &want);
        }
    }

    /// k = 0 scales C by beta.
    #[test]
    fn native_k0_scales_c() {
        let alpha = Complex32::new(0.7, -0.3);
        let beta = Complex32::new(0.3, 0.1);
        let c0 = det_data(4, 4, 3);
        let mut got = c0.clone();
        cgemm_native_run(
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
        let want: Vec<Complex32> = c0.iter().map(|&v| beta * v).collect();
        check_close(&got, &want);
    }

    /// Direct-call correctness with padded row strides and general scalars.
    #[test]
    fn native_matches_naive_strided() {
        let alpha = Complex32::new(-1.0, 0.5);
        let beta = Complex32::new(1.0, -0.25);
        let (m, n, k) = (30, 22, 140);
        let (lda, ldb, ldc) = (k + 3, n + 5, n + 7);
        let pack_a: Vec<Complex32> = det_data(m, k, 5);
        let pack_b: Vec<Complex32> = det_data(k, n, 9);
        let pack_c: Vec<Complex32> = det_data(m, n, 13);
        let mut a = vec![Complex32::new(0.0, 0.0); m * lda];
        let mut b = vec![Complex32::new(0.0, 0.0); k * ldb];
        let mut c = vec![Complex32::new(0.0, 0.0); m * ldc];
        for i in 0..m {
            a[i * lda..i * lda + k].copy_from_slice(&pack_a[i * k..(i + 1) * k]);
            c[i * ldc..i * ldc + n].copy_from_slice(&pack_c[i * n..(i + 1) * n]);
        }
        for p in 0..k {
            b[p * ldb..p * ldb + n].copy_from_slice(&pack_b[p * n..(p + 1) * n]);
        }
        let want = naive(m, n, k, alpha, &pack_a, &pack_b, beta, &pack_c);
        cgemm_native_run(
            m, n, k, alpha, &a, lda, &b, ldb, beta, &mut c, ldc, false, false,
        );
        let got: Vec<Complex32> = (0..m)
            .flat_map(|i| c[i * ldc..i * ldc + n].to_vec())
            .collect();
        check_close(&got, &want);
    }

    /// Parallel sweep (above the floor) matches naive on general scalars
    /// with multi-chunk k.
    #[test]
    fn native_parallel_matches_naive() {
        let alpha = Complex32::new(0.7, 0.3);
        let beta = Complex32::new(0.2, -0.4);
        let (m, n, k) = (300usize, 260usize, 200usize);
        assert!(m * n * k >= NATIVE_PAR_MACS_32);
        let a = det_data(m, k, 21);
        let b = det_data(k, n, 22);
        let c0 = det_data(m, n, 23);
        let want = naive(m, n, k, alpha, &a, &b, beta, &c0);
        let mut got = c0.clone();
        let handled = cgemm_native(
            m, n, k, alpha, &a, k, &b, n, beta, &mut got, n, false, false,
        );
        assert!(handled);
        check_close(&got, &want);
    }

    /// Serial-sweep panel blocking is bitwise-invariant: every C tile is
    /// computed by one micro-tile call however panels group into blocks,
    /// so all `pb` values (one panel, the shipped 8, oversized)
    /// agree exactly — including row/column tails and the transposed-B
    /// arm. Guards the `SONIUM_C32_PB` gate.
    #[test]
    fn serial_pb_bitwise_agreement() {
        let alpha = Complex32::new(-1.0, 0.0);
        let beta = Complex32::new(1.0, 0.0);
        // (m, n, k, trans_b): SYRK-band shape, tails, big-k, transposed-B.
        for (m, n, k, trans_b) in [
            (32usize, 960usize, 48usize, false),
            (30usize, 100usize, 48usize, false),
            (7usize, 13usize, 16usize, false),
            (64usize, 64usize, 256usize, false),
            (32usize, 200usize, 48usize, true),
        ] {
            let a = det_data(m, k, 31);
            let b = det_data(if trans_b { n } else { k }, if trans_b { k } else { n }, 32);
            let c0 = det_data(m, n, 33);
            let mut base = c0.clone();
            sweep_blocks_c32(
                m,
                n,
                k,
                alpha,
                &a,
                k,
                &b,
                if trans_b { k } else { n },
                beta,
                &mut base,
                n,
                trans_b,
                false,
                8,
            );
            for pb in [1usize, 2, 3, 7, 16, 64, 1024] {
                let mut got = c0.clone();
                sweep_blocks_c32(
                    m,
                    n,
                    k,
                    alpha,
                    &a,
                    k,
                    &b,
                    if trans_b { k } else { n },
                    beta,
                    &mut got,
                    n,
                    trans_b,
                    false,
                    pb,
                );
                assert_eq!(got, base, "m={m} n={n} k={k} trans_b={trans_b} pb={pb}");
            }
        }
    }
}
