//! AVX2/FMA GEMV kernels for x86_64 (`NoTrans`).
//!
//! The SME2 GEMV path falls back to a single-accumulator scalar loop on
//! x86_64 hosts. These kernels close that gap: each row dot runs four
//! FMA accumulators over 16 (f64) / 32 (f32) elements per iteration, and
//! large matrices fan out over rayon row bands.
//!
//! Contract mirrors the scalar fallback exactly, including `beta`
//! edge cases: `beta == 0` overwrites `y` without reading it (so NaN
//! entries do not propagate), `alpha == 0` scales `y` by `beta` without
//! touching `A`/`x`.
//!
//! SAFETY: callers must ensure the host has AVX2 + FMA (guard with
//! [`crate::cpu::blas1::avx2::is_available`]).

use core::arch::x86_64::*;

use crate::cpu::blas1::avx2::{hsum_pd, hsum_ps, is_available};

/// Pointer wrappers for `rayon::scope` dispatch: disjoint `y` row bands,
/// read-only `A`/`x`.
#[derive(Clone, Copy)]
struct SendConstPtr<T>(*const T);
unsafe impl<T> Send for SendConstPtr<T> {}
unsafe impl<T> Sync for SendConstPtr<T> {}
#[derive(Clone, Copy)]
struct SendMutPtr<T>(*mut T);
unsafe impl<T> Send for SendMutPtr<T> {}
unsafe impl<T> Sync for SendMutPtr<T> {}

/// Minimum `m * n` before paying scope spawn/join, and minimum rows for a
/// parallel split at all. Below this the serial sweep wins.
const PAR_MIN_MN: usize = 262_144;
const PAR_MIN_M: usize = 32;
/// Rows per rayon task on the legacy raw-scope path (see [`gemv_pinned`]).
const BAND_ROWS: usize = 64;

#[inline]
fn use_parallel(m: usize, n: usize) -> bool {
    m >= PAR_MIN_M && m.saturating_mul(n) >= PAR_MIN_MN
}

/// Spread-broadcast dispatch for GEMV MT, NoTrans and Trans (shipped
/// default-on; `SONIUM_GEMV_PINNED=0` restores the legacy raw-
/// `rayon::scope` on both). Read per call so same-process A/B can
/// interleave both arms.
///
/// Why: the raw scope plateaus at ~4 threads (4096² f64: 1T 6.3ms, 4T
/// 2.4ms, 16T 2.4ms, 32T 2.1ms) while OpenBLAS scales to 0.2ms — a
/// placement diagnostic showed the band→CPU map fully reshuffling every
/// rep, so each band's 8MB lands in a different CCX's L3 slice than the
/// last rep and every rep pays fabric/DRAM (~56GB/s) instead of local
/// slices (~470GB/s). Broadcast on spread-pinned workers makes the
/// mapping static by construction.
///
/// Shipped 2026-09-27 on quiet-box same-process A/B (16T, bitwise 0e0):
/// broadcast beats raw on every NoTrans shape (sq1k 1.5x, sq4k 2.2-2.6x,
/// sq8k 1.4-1.5x, tall 2.1-3.5x), taking NoTrans MT from 1.05-9.0x vs
/// OpenBLAS to 0.84-3.05x (sq8k now wins outright). Nested calls fall
/// back to the raw scope (see call sites); sub-width ambient pools
/// broadcast on ambient-sized spread pools (2/4/8, floored; see
/// `spread_pool_for_ambient`), `SONIUM_GEMV_AMBIENT=0` restores the
/// legacy full-width-only rule for A/B.
///
/// NOTE: an earlier pinned variant (BLAS-3 pool router + worker-count
/// bands via scopes) was falsified first — it lost squares 0.5-0.7x
/// (compact pooling starves streaming bandwidth) — because scopes still
/// steal. The broadcast (no stealing, index-addressed bands) is the
/// load-bearing half, not just the pinning. Results are bitwise
/// identical either way (asserted in-probe and in
/// `avx2_gemv_notrans_pinned_matches_scalar`).
#[inline]
fn gemv_pinned() -> bool {
    std::env::var("SONIUM_GEMV_PINNED")
        .map(|v| v != "0")
        .unwrap_or(true)
}

/// One f64 row dot with four FMA accumulators.
///
/// NOTE (2026-09-27): an eight-chain twin (mirroring the shipped f32
/// change below) ties 4-chain on every MT shape (min-identical on 7
/// shapes; ST unmeasured): f64 dots look bandwidth-capped (0.125
/// flops/byte) so extra chains buy nothing, while f32 (0.25
/// flops/byte) is latency/MLP-bound and gains 1.1-1.35x long-n MT.
/// Keep 4 here.
#[target_feature(enable = "avx2,fma")]
unsafe fn row_dot_f64(a_row: *const f64, x: *const f64, n: usize) -> f64 {
    // SAFETY: outer `unsafe fn` contract plus `#[target_feature]`: AVX2/FMA live, slices in bounds.
    unsafe {
        let mut a0 = _mm256_setzero_pd();
        let mut a1 = _mm256_setzero_pd();
        let mut a2 = _mm256_setzero_pd();
        let mut a3 = _mm256_setzero_pd();
        let mut j = 0;
        while j + 16 <= n {
            a0 = _mm256_fmadd_pd(_mm256_loadu_pd(a_row.add(j)), _mm256_loadu_pd(x.add(j)), a0);
            a1 = _mm256_fmadd_pd(
                _mm256_loadu_pd(a_row.add(j + 4)),
                _mm256_loadu_pd(x.add(j + 4)),
                a1,
            );
            a2 = _mm256_fmadd_pd(
                _mm256_loadu_pd(a_row.add(j + 8)),
                _mm256_loadu_pd(x.add(j + 8)),
                a2,
            );
            a3 = _mm256_fmadd_pd(
                _mm256_loadu_pd(a_row.add(j + 12)),
                _mm256_loadu_pd(x.add(j + 12)),
                a3,
            );
            j += 16;
        }
        let mut acc = hsum_pd(_mm256_add_pd(_mm256_add_pd(a0, a1), _mm256_add_pd(a2, a3)));
        // Vector-accumulated spill tier: one hsum at the end. A per-iter
        // `acc += hsum(mul)` serializes on ~10cy hsum latency and the
        // scalar loop-carried dep (~30x slower per el than the main
        // loop; measured 1.57x whole-shape loss on n=256 f32).
        let mut t0 = _mm256_setzero_pd();
        while j + 4 <= n {
            t0 = _mm256_fmadd_pd(_mm256_loadu_pd(x.add(j)), _mm256_loadu_pd(a_row.add(j)), t0);
            j += 4;
        }
        acc += hsum_pd(t0);
        while j < n {
            acc += *a_row.add(j) * *x.add(j);
            j += 1;
        }
        acc
    }
}

/// One f32 row dot with eight FMA accumulators: 4 chains cover 4/5
/// FMA/cy of latency; 8 double the chains and the concurrent A-misses
/// (measured 1.1-1.35x on long-n MT). Tolerance-equal results — the
/// 8-way tree reorders rounding. f64 keeps 4 (bandwidth-capped; see
/// the NOTE on [`row_dot_f64`]).
#[target_feature(enable = "avx2,fma")]
unsafe fn row_dot_f32(a_row: *const f32, x: *const f32, n: usize) -> f32 {
    // SAFETY: outer `unsafe fn` contract plus `#[target_feature]`: AVX2/FMA live, slices in bounds.
    unsafe {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        let mut a4 = _mm256_setzero_ps();
        let mut a5 = _mm256_setzero_ps();
        let mut a6 = _mm256_setzero_ps();
        let mut a7 = _mm256_setzero_ps();
        let mut j = 0;
        while j + 64 <= n {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_row.add(j)), _mm256_loadu_ps(x.add(j)), a0);
            a1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a_row.add(j + 8)),
                _mm256_loadu_ps(x.add(j + 8)),
                a1,
            );
            a2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a_row.add(j + 16)),
                _mm256_loadu_ps(x.add(j + 16)),
                a2,
            );
            a3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a_row.add(j + 24)),
                _mm256_loadu_ps(x.add(j + 24)),
                a3,
            );
            a4 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a_row.add(j + 32)),
                _mm256_loadu_ps(x.add(j + 32)),
                a4,
            );
            a5 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a_row.add(j + 40)),
                _mm256_loadu_ps(x.add(j + 40)),
                a5,
            );
            a6 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a_row.add(j + 48)),
                _mm256_loadu_ps(x.add(j + 48)),
                a6,
            );
            a7 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a_row.add(j + 56)),
                _mm256_loadu_ps(x.add(j + 56)),
                a7,
            );
            j += 64;
        }
        let lo = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        let hi = _mm256_add_ps(_mm256_add_ps(a4, a5), _mm256_add_ps(a6, a7));
        let mut acc = hsum_ps(_mm256_add_ps(lo, hi));
        // Vector-accumulated spill tier (see the NOTE on the f64 tier):
        // per-iter `acc += hsum(mul)` serializes on hsum latency.
        let mut t0 = _mm256_setzero_ps();
        while j + 8 <= n {
            t0 = _mm256_fmadd_ps(_mm256_loadu_ps(x.add(j)), _mm256_loadu_ps(a_row.add(j)), t0);
            j += 8;
        }
        acc += hsum_ps(t0);
        while j < n {
            acc += *a_row.add(j) * *x.add(j);
            j += 1;
        }
        acc
    }
}

/// Two interleaved f32 row dots (7 chains each, x shared): 14 FMA
/// chains in flight raise concurrent A-misses against L3/fabric far
/// past the single-row 8 (the OOO window holds one big row, so
/// explicit pairing is the only way up), and sharing each x-load
/// across both rows halves x-traffic. Seven — not eight — per row:
/// 8+8 needs 17 live vectors (16 accumulators + x temp) and LLVM
/// spills accs into the LD-bound loop, eating the MLP gain; 7+7 fits
/// 15 registers exactly and still saturates miss buffers.
/// Tolerance-equal per row to two [`row_dot_f32`] calls (different
/// chain count reorders rounding). f32-only: f64 dots are
/// bandwidth-capped (see the NOTE on [`row_dot_f64`]).
#[target_feature(enable = "avx2,fma")]
unsafe fn row_dot2_f32(
    a0_row: *const f32,
    a1_row: *const f32,
    x: *const f32,
    n: usize,
) -> (f32, f32) {
    // SAFETY: outer `unsafe fn` contract plus `#[target_feature]`: AVX2/FMA live, slices in bounds.
    unsafe {
        let mut b0 = _mm256_setzero_ps();
        let mut b1 = _mm256_setzero_ps();
        let mut b2 = _mm256_setzero_ps();
        let mut b3 = _mm256_setzero_ps();
        let mut b4 = _mm256_setzero_ps();
        let mut b5 = _mm256_setzero_ps();
        let mut b6 = _mm256_setzero_ps();
        let mut c0 = _mm256_setzero_ps();
        let mut c1 = _mm256_setzero_ps();
        let mut c2 = _mm256_setzero_ps();
        let mut c3 = _mm256_setzero_ps();
        let mut c4 = _mm256_setzero_ps();
        let mut c5 = _mm256_setzero_ps();
        let mut c6 = _mm256_setzero_ps();
        let mut j = 0;
        while j + 56 <= n {
            macro_rules! lane {
                ($b:ident, $c:ident, $o:expr) => {{
                    let xv = _mm256_loadu_ps(x.add(j + $o));
                    $b = _mm256_fmadd_ps(xv, _mm256_loadu_ps(a0_row.add(j + $o)), $b);
                    $c = _mm256_fmadd_ps(xv, _mm256_loadu_ps(a1_row.add(j + $o)), $c);
                }};
            }
            lane!(b0, c0, 0);
            lane!(b1, c1, 8);
            lane!(b2, c2, 16);
            lane!(b3, c3, 24);
            lane!(b4, c4, 32);
            lane!(b5, c5, 40);
            lane!(b6, c6, 48);
            j += 56;
        }
        let b0123 = _mm256_add_ps(_mm256_add_ps(b0, b1), _mm256_add_ps(b2, b3));
        let b456 = _mm256_add_ps(_mm256_add_ps(b4, b5), b6);
        let c0123 = _mm256_add_ps(_mm256_add_ps(c0, c1), _mm256_add_ps(c2, c3));
        let c456 = _mm256_add_ps(_mm256_add_ps(c4, c5), c6);
        let mut acc0 = hsum_ps(_mm256_add_ps(b0123, b456));
        let mut acc1 = hsum_ps(_mm256_add_ps(c0123, c456));
        // Vector-accumulated spill tier (see the NOTE on the f64 tier):
        // per-iter `acc += hsum(mul)` serializes on hsum latency. The
        // 56-wide main loop always leaves a remainder on power-of-2 n,
        // so this tier is hot, not a corner.
        let mut t0 = _mm256_setzero_ps();
        let mut t1 = _mm256_setzero_ps();
        while j + 8 <= n {
            let xv = _mm256_loadu_ps(x.add(j));
            t0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(a0_row.add(j)), t0);
            t1 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(a1_row.add(j)), t1);
            j += 8;
        }
        acc0 += hsum_ps(t0);
        acc1 += hsum_ps(t1);
        while j < n {
            let xv = *x.add(j);
            acc0 += xv * *a0_row.add(j);
            acc1 += xv * *a1_row.add(j);
            j += 1;
        }
        (acc0, acc1)
    }
}

/// f64 twin of [`row_dot2_f32`]: two interleaved dots, 7 chains each,
/// x shared (28-wide main loop). 15 live vectors, spill-free.
/// Tolerance-equal per row to two [`row_dot_f64`] calls.
#[target_feature(enable = "avx2,fma")]
unsafe fn row_dot2_f64(
    a0_row: *const f64,
    a1_row: *const f64,
    x: *const f64,
    n: usize,
) -> (f64, f64) {
    // SAFETY: outer `unsafe fn` contract plus `#[target_feature]`: AVX2/FMA live, slices in bounds.
    unsafe {
        let mut b0 = _mm256_setzero_pd();
        let mut b1 = _mm256_setzero_pd();
        let mut b2 = _mm256_setzero_pd();
        let mut b3 = _mm256_setzero_pd();
        let mut b4 = _mm256_setzero_pd();
        let mut b5 = _mm256_setzero_pd();
        let mut b6 = _mm256_setzero_pd();
        let mut c0 = _mm256_setzero_pd();
        let mut c1 = _mm256_setzero_pd();
        let mut c2 = _mm256_setzero_pd();
        let mut c3 = _mm256_setzero_pd();
        let mut c4 = _mm256_setzero_pd();
        let mut c5 = _mm256_setzero_pd();
        let mut c6 = _mm256_setzero_pd();
        let mut j = 0;
        while j + 28 <= n {
            macro_rules! lane {
                ($b:ident, $c:ident, $o:expr) => {{
                    let xv = _mm256_loadu_pd(x.add(j + $o));
                    $b = _mm256_fmadd_pd(xv, _mm256_loadu_pd(a0_row.add(j + $o)), $b);
                    $c = _mm256_fmadd_pd(xv, _mm256_loadu_pd(a1_row.add(j + $o)), $c);
                }};
            }
            lane!(b0, c0, 0);
            lane!(b1, c1, 4);
            lane!(b2, c2, 8);
            lane!(b3, c3, 12);
            lane!(b4, c4, 16);
            lane!(b5, c5, 20);
            lane!(b6, c6, 24);
            j += 28;
        }
        let b0123 = _mm256_add_pd(_mm256_add_pd(b0, b1), _mm256_add_pd(b2, b3));
        let b456 = _mm256_add_pd(_mm256_add_pd(b4, b5), b6);
        let c0123 = _mm256_add_pd(_mm256_add_pd(c0, c1), _mm256_add_pd(c2, c3));
        let c456 = _mm256_add_pd(_mm256_add_pd(c4, c5), c6);
        let mut acc0 = hsum_pd(_mm256_add_pd(b0123, b456));
        let mut acc1 = hsum_pd(_mm256_add_pd(c0123, c456));
        // Vector-accumulated spill tier (see the NOTE on the f64
        // single-row tier): per-iter hsum would serialize.
        let mut t0 = _mm256_setzero_pd();
        let mut t1 = _mm256_setzero_pd();
        while j + 4 <= n {
            let xv = _mm256_loadu_pd(x.add(j));
            t0 = _mm256_fmadd_pd(xv, _mm256_loadu_pd(a0_row.add(j)), t0);
            t1 = _mm256_fmadd_pd(xv, _mm256_loadu_pd(a1_row.add(j)), t1);
            j += 4;
        }
        acc0 += hsum_pd(t0);
        acc1 += hsum_pd(t1);
        while j < n {
            let xv = *x.add(j);
            acc0 += xv * *a0_row.add(j);
            acc1 += xv * *a1_row.add(j);
            j += 1;
        }
        (acc0, acc1)
    }
}

/// Four interleaved f32 row dots (3 chains each, x shared): one x-load
/// feeds 4 FMAs, quartering the x-refill traffic that caps the
/// single-row (8 chains, full x per row) and paired (14 chains, x per 2
/// rows) kernels on L3-resident shapes (x re-streams from L2 every row
/// and shares L1D fill ports with A; pairing's measured 1.07-1.26x is
/// the x-halving paying off). 12 accumulators + x temp = 13 live
/// vectors, spill-free (4x4 would spill like 8+8 did). 12 total A-miss
/// chains ≈ the pair's 14, so this is strictly narrower only in
/// per-row MLP. Tolerance-equal per row to [`row_dot_f32`].
#[target_feature(enable = "avx2,fma")]
unsafe fn row_dot4_f32(
    a0_row: *const f32,
    a1_row: *const f32,
    a2_row: *const f32,
    a3_row: *const f32,
    x: *const f32,
    n: usize,
) -> (f32, f32, f32, f32) {
    // SAFETY: outer `unsafe fn` contract plus `#[target_feature]`: AVX2/FMA live, slices in bounds.
    unsafe {
        let mut b0 = _mm256_setzero_ps();
        let mut b1 = _mm256_setzero_ps();
        let mut b2 = _mm256_setzero_ps();
        let mut c0 = _mm256_setzero_ps();
        let mut c1 = _mm256_setzero_ps();
        let mut c2 = _mm256_setzero_ps();
        let mut d0 = _mm256_setzero_ps();
        let mut d1 = _mm256_setzero_ps();
        let mut d2 = _mm256_setzero_ps();
        let mut e0 = _mm256_setzero_ps();
        let mut e1 = _mm256_setzero_ps();
        let mut e2 = _mm256_setzero_ps();
        let mut j = 0;
        while j + 24 <= n {
            macro_rules! lane {
                ($b:ident, $c:ident, $d:ident, $e:ident, $o:expr) => {{
                    let xv = _mm256_loadu_ps(x.add(j + $o));
                    $b = _mm256_fmadd_ps(xv, _mm256_loadu_ps(a0_row.add(j + $o)), $b);
                    $c = _mm256_fmadd_ps(xv, _mm256_loadu_ps(a1_row.add(j + $o)), $c);
                    $d = _mm256_fmadd_ps(xv, _mm256_loadu_ps(a2_row.add(j + $o)), $d);
                    $e = _mm256_fmadd_ps(xv, _mm256_loadu_ps(a3_row.add(j + $o)), $e);
                }};
            }
            lane!(b0, c0, d0, e0, 0);
            lane!(b1, c1, d1, e1, 8);
            lane!(b2, c2, d2, e2, 16);
            j += 24;
        }
        let mut acc0 = hsum_ps(_mm256_add_ps(_mm256_add_ps(b0, b1), b2));
        let mut acc1 = hsum_ps(_mm256_add_ps(_mm256_add_ps(c0, c1), c2));
        let mut acc2 = hsum_ps(_mm256_add_ps(_mm256_add_ps(d0, d1), d2));
        let mut acc3 = hsum_ps(_mm256_add_ps(_mm256_add_ps(e0, e1), e2));
        // Vector-accumulated spill tier (see the NOTE on the f64
        // single-row tier): per-iter hsum would serialize.
        let mut t0 = _mm256_setzero_ps();
        let mut t1 = _mm256_setzero_ps();
        let mut t2 = _mm256_setzero_ps();
        let mut t3 = _mm256_setzero_ps();
        while j + 8 <= n {
            let xv = _mm256_loadu_ps(x.add(j));
            t0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(a0_row.add(j)), t0);
            t1 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(a1_row.add(j)), t1);
            t2 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(a2_row.add(j)), t2);
            t3 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(a3_row.add(j)), t3);
            j += 8;
        }
        acc0 += hsum_ps(t0);
        acc1 += hsum_ps(t1);
        acc2 += hsum_ps(t2);
        acc3 += hsum_ps(t3);
        while j < n {
            let xv = *x.add(j);
            acc0 += xv * *a0_row.add(j);
            acc1 += xv * *a1_row.add(j);
            acc2 += xv * *a2_row.add(j);
            acc3 += xv * *a3_row.add(j);
            j += 1;
        }
        (acc0, acc1, acc2, acc3)
    }
}

/// f64 twin of [`row_dot4_f32`]: four interleaved dots, 3 chains each,
/// x shared (12-wide main loop). 13 live vectors, spill-free.
/// Tolerance-equal per row to [`row_dot_f64`].
#[target_feature(enable = "avx2,fma")]
unsafe fn row_dot4_f64(
    a0_row: *const f64,
    a1_row: *const f64,
    a2_row: *const f64,
    a3_row: *const f64,
    x: *const f64,
    n: usize,
) -> (f64, f64, f64, f64) {
    // SAFETY: outer `unsafe fn` contract plus `#[target_feature]`: AVX2/FMA live, slices in bounds.
    unsafe {
        let mut b0 = _mm256_setzero_pd();
        let mut b1 = _mm256_setzero_pd();
        let mut b2 = _mm256_setzero_pd();
        let mut c0 = _mm256_setzero_pd();
        let mut c1 = _mm256_setzero_pd();
        let mut c2 = _mm256_setzero_pd();
        let mut d0 = _mm256_setzero_pd();
        let mut d1 = _mm256_setzero_pd();
        let mut d2 = _mm256_setzero_pd();
        let mut e0 = _mm256_setzero_pd();
        let mut e1 = _mm256_setzero_pd();
        let mut e2 = _mm256_setzero_pd();
        let mut j = 0;
        while j + 12 <= n {
            macro_rules! lane {
                ($b:ident, $c:ident, $d:ident, $e:ident, $o:expr) => {{
                    let xv = _mm256_loadu_pd(x.add(j + $o));
                    $b = _mm256_fmadd_pd(xv, _mm256_loadu_pd(a0_row.add(j + $o)), $b);
                    $c = _mm256_fmadd_pd(xv, _mm256_loadu_pd(a1_row.add(j + $o)), $c);
                    $d = _mm256_fmadd_pd(xv, _mm256_loadu_pd(a2_row.add(j + $o)), $d);
                    $e = _mm256_fmadd_pd(xv, _mm256_loadu_pd(a3_row.add(j + $o)), $e);
                }};
            }
            lane!(b0, c0, d0, e0, 0);
            lane!(b1, c1, d1, e1, 4);
            lane!(b2, c2, d2, e2, 8);
            j += 12;
        }
        let mut acc0 = hsum_pd(_mm256_add_pd(_mm256_add_pd(b0, b1), b2));
        let mut acc1 = hsum_pd(_mm256_add_pd(_mm256_add_pd(c0, c1), c2));
        let mut acc2 = hsum_pd(_mm256_add_pd(_mm256_add_pd(d0, d1), d2));
        let mut acc3 = hsum_pd(_mm256_add_pd(_mm256_add_pd(e0, e1), e2));
        // Vector-accumulated spill tier (see the NOTE on the f64
        // single-row tier): per-iter hsum would serialize.
        let mut t0 = _mm256_setzero_pd();
        let mut t1 = _mm256_setzero_pd();
        let mut t2 = _mm256_setzero_pd();
        let mut t3 = _mm256_setzero_pd();
        while j + 4 <= n {
            let xv = _mm256_loadu_pd(x.add(j));
            t0 = _mm256_fmadd_pd(xv, _mm256_loadu_pd(a0_row.add(j)), t0);
            t1 = _mm256_fmadd_pd(xv, _mm256_loadu_pd(a1_row.add(j)), t1);
            t2 = _mm256_fmadd_pd(xv, _mm256_loadu_pd(a2_row.add(j)), t2);
            t3 = _mm256_fmadd_pd(xv, _mm256_loadu_pd(a3_row.add(j)), t3);
            j += 4;
        }
        acc0 += hsum_pd(t0);
        acc1 += hsum_pd(t1);
        acc2 += hsum_pd(t2);
        acc3 += hsum_pd(t3);
        while j < n {
            let xv = *x.add(j);
            acc0 += xv * *a0_row.add(j);
            acc1 += xv * *a1_row.add(j);
            acc2 += xv * *a2_row.add(j);
            acc3 += xv * *a3_row.add(j);
            j += 1;
        }
        (acc0, acc1, acc2, acc3)
    }
}

/// Quad-dot gate: default-on at `n >= 256` (level 2);
/// `SONIUM_GEMV_DOT4=1` narrows to `n >= 1024` (same gate as pairing),
/// `=0` keeps the legacy pair/single dispatch everywhere. Read per call
/// so same-process A/B can interleave arms. Quad wins only where
/// x-refill traffic binds (MT bands; measured MT wins-or-ties on every
/// cell, ST loses up to 1.36x on x-L1-resident shapes) — callers pass
/// `allow_quad=false` on the serial path and wherever fewer than 2
/// workers run (single-threaded scopes, width-1 broadcasts under
/// masks).
#[inline]
fn dot4_gate() -> u8 {
    match std::env::var("SONIUM_GEMV_DOT4") {
        Ok(v) if v.as_bytes().first() == Some(&b'0') => 0,
        Ok(v) if v.as_bytes().first() == Some(&b'1') => 1,
        _ => 2,
    }
}

/// Row-band sweep for rows `[r0, r1)`: `y[i] = alpha·(A[i]·x) + beta·y[i]`.
/// Shared by the serial loop and each rayon task.
#[target_feature(enable = "avx2,fma")]
unsafe fn sweep_f64(
    r0: usize,
    r1: usize,
    n: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    x: *const f64,
    beta: f64,
    y: *mut f64,
    allow_quad: bool,
) {
    // SAFETY: outer `unsafe fn` contract: `a` rows and `y[r0..r1]` are
    // addressable, `x` spans `n`.
    unsafe {
        // Rows pair into `row_dot2_f64` (14 chains, x shared) on
        // long rows only (`n >= 1024`, same gate as f32): the n-sweep
        // loses 1.11-1.57x at n<=384, wins 1.10x at n=512 but ties
        // n=768 (non-monotonic territory), and wins 1.19-1.23x solid
        // at n>=1024. Conservative gate keeps the proven wins; the
        // n=512 win is a follow-up (needs MT validation + the 768
        // dip understood). A lone trailing row keeps single-row.
        let pair = n >= 1024;
        let gate = dot4_gate();
        // Quad dots only where x-refill traffic binds (MT bands):
        // single-threaded sweeps are per-row-MLP-bound and lose on
        // quads (measured to 1.36x), so the serial path passes
        // `allow_quad=false` and stays on the proven pair/single.
        let quad = allow_quad && (gate == 2 && n >= 256 || gate == 1 && n >= 1024);
        // Canonical quad alignment: quads always cover rows [4k, 4k+4),
        // so 4-aligned band splits compute bitwise-identical rows (all
        // production bandings are 4-aligned: broadcast rounds up, raw
        // scope uses 64-row bands, serial starts at 0). A hypothetical
        // misaligned band drains its head (<=3 rows) single first, so
        // its interior quads stay canonical; rows spanning the cut
        // itself are tolerance-equal only (no tiling can reproduce a
        // quad whose rows live on two bands). Skipped unless quads are
        // active, so legacy dispatch is untouched.
        let q0 = r0.next_multiple_of(4).min(r1);
        let mut i = r0;
        if beta == 0.0 {
            if quad {
                while i < q0 {
                    *y.add(i) = alpha * row_dot_f64(a.add(i * lda), x, n);
                    i += 1;
                }
            }
            while quad && i + 3 < r1 {
                let (d0, d1, d2, d3) = row_dot4_f64(
                    a.add(i * lda),
                    a.add((i + 1) * lda),
                    a.add((i + 2) * lda),
                    a.add((i + 3) * lda),
                    x,
                    n,
                );
                *y.add(i) = alpha * d0;
                *y.add(i + 1) = alpha * d1;
                *y.add(i + 2) = alpha * d2;
                *y.add(i + 3) = alpha * d3;
                i += 4;
            }
            while pair && i + 1 < r1 {
                let (d0, d1) = row_dot2_f64(a.add(i * lda), a.add((i + 1) * lda), x, n);
                *y.add(i) = alpha * d0;
                *y.add(i + 1) = alpha * d1;
                i += 2;
            }
            for i in i..r1 {
                *y.add(i) = alpha * row_dot_f64(a.add(i * lda), x, n);
            }
        } else {
            if quad {
                while i < q0 {
                    let yi = y.add(i);
                    *yi = alpha * row_dot_f64(a.add(i * lda), x, n) + beta * *yi;
                    i += 1;
                }
            }
            while quad && i + 3 < r1 {
                let (d0, d1, d2, d3) = row_dot4_f64(
                    a.add(i * lda),
                    a.add((i + 1) * lda),
                    a.add((i + 2) * lda),
                    a.add((i + 3) * lda),
                    x,
                    n,
                );
                let y0 = y.add(i);
                let y1 = y.add(i + 1);
                let y2 = y.add(i + 2);
                let y3 = y.add(i + 3);
                *y0 = alpha * d0 + beta * *y0;
                *y1 = alpha * d1 + beta * *y1;
                *y2 = alpha * d2 + beta * *y2;
                *y3 = alpha * d3 + beta * *y3;
                i += 4;
            }
            while pair && i + 1 < r1 {
                let (d0, d1) = row_dot2_f64(a.add(i * lda), a.add((i + 1) * lda), x, n);
                let y0 = y.add(i);
                let y1 = y.add(i + 1);
                *y0 = alpha * d0 + beta * *y0;
                *y1 = alpha * d1 + beta * *y1;
                i += 2;
            }
            for i in i..r1 {
                let yi = y.add(i);
                *yi = alpha * row_dot_f64(a.add(i * lda), x, n) + beta * *yi;
            }
        }
    }
}

/// f32 twin of [`sweep_f64`].
#[target_feature(enable = "avx2,fma")]
unsafe fn sweep_f32(
    r0: usize,
    r1: usize,
    n: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    x: *const f32,
    beta: f32,
    y: *mut f32,
    allow_quad: bool,
) {
    // SAFETY: outer `unsafe fn` contract: `a` rows and `y[r0..r1]` are
    // addressable, `x` spans `n`.
    unsafe {
        // Rows pair into `row_dot2_f32` (14 chains, x shared) on
        // long rows only (`n >= 1024`): below the crossover pairing
        // loses 1.13-1.22x (n=128..512, single-row already at the
        // streaming roof; the pair zigzag only costs), above it wins
        // 1.07-1.26x (n=1024..2048; n=768 ties). A lone trailing row
        // keeps the single-row kernel. Tolerance-equal per row to the
        // unpaired loop (7+7 chains reorder rounding).
        let pair = n >= 1024;
        let gate = dot4_gate();
        // Quad dots only where x-refill traffic binds (MT bands; see
        // the NOTE on the f64 sweep): the serial path passes
        // `allow_quad=false`.
        let quad = allow_quad && (gate == 2 && n >= 256 || gate == 1 && n >= 1024);
        // Canonical quad alignment (see the NOTE on the f64 sweep):
        // 4-aligned bands quad from the band start; a hypothetical
        // misaligned head (<=3 rows) drains single first.
        let q0 = r0.next_multiple_of(4).min(r1);
        let mut i = r0;
        if beta == 0.0 {
            if quad {
                while i < q0 {
                    *y.add(i) = alpha * row_dot_f32(a.add(i * lda), x, n);
                    i += 1;
                }
            }
            while quad && i + 3 < r1 {
                let (d0, d1, d2, d3) = row_dot4_f32(
                    a.add(i * lda),
                    a.add((i + 1) * lda),
                    a.add((i + 2) * lda),
                    a.add((i + 3) * lda),
                    x,
                    n,
                );
                *y.add(i) = alpha * d0;
                *y.add(i + 1) = alpha * d1;
                *y.add(i + 2) = alpha * d2;
                *y.add(i + 3) = alpha * d3;
                i += 4;
            }
            while pair && i + 1 < r1 {
                let (d0, d1) = row_dot2_f32(a.add(i * lda), a.add((i + 1) * lda), x, n);
                *y.add(i) = alpha * d0;
                *y.add(i + 1) = alpha * d1;
                i += 2;
            }
            for i in i..r1 {
                *y.add(i) = alpha * row_dot_f32(a.add(i * lda), x, n);
            }
        } else {
            if quad {
                while i < q0 {
                    let yi = y.add(i);
                    *yi = alpha * row_dot_f32(a.add(i * lda), x, n) + beta * *yi;
                    i += 1;
                }
            }
            while quad && i + 3 < r1 {
                let (d0, d1, d2, d3) = row_dot4_f32(
                    a.add(i * lda),
                    a.add((i + 1) * lda),
                    a.add((i + 2) * lda),
                    a.add((i + 3) * lda),
                    x,
                    n,
                );
                let y0 = y.add(i);
                let y1 = y.add(i + 1);
                let y2 = y.add(i + 2);
                let y3 = y.add(i + 3);
                *y0 = alpha * d0 + beta * *y0;
                *y1 = alpha * d1 + beta * *y1;
                *y2 = alpha * d2 + beta * *y2;
                *y3 = alpha * d3 + beta * *y3;
                i += 4;
            }
            while pair && i + 1 < r1 {
                let (d0, d1) = row_dot2_f32(a.add(i * lda), a.add((i + 1) * lda), x, n);
                let y0 = y.add(i);
                let y1 = y.add(i + 1);
                *y0 = alpha * d0 + beta * *y0;
                *y1 = alpha * d1 + beta * *y1;
                i += 2;
            }
            for i in i..r1 {
                let yi = y.add(i);
                *yi = alpha * row_dot_f32(a.add(i * lda), x, n) + beta * *yi;
            }
        }
    }
}

/// Fused fan-out tile: folds rows `[j0, j1)` into `priv_y[0..clen]` with one
/// call per task instead of one BLAS-1 `axpy` per row. The per-row call
/// (dispatch check + call/ret) costs ~50ns, which dominates once the tile
/// loop is short: at 8192² the tiled fan-out issues 65k calls (~3ms of pure
/// overhead). Fusing keeps the identical arithmetic in raw AVX2 loop form.
#[target_feature(enable = "avx2,fma")]
unsafe fn sweep_fanout_tile_f64(
    j0: usize,
    j1: usize,
    i0: usize,
    clen: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    x: *const f64,
    priv_y: *mut f64,
) {
    // SAFETY: outer `unsafe fn` contract: rows `j0..j1` carry `clen`
    // elements from column `i0` (`lda >= i0 + clen`), `x[j]` readable,
    // `priv_y[0..clen]` writable.
    unsafe {
        for j in j0..j1 {
            let xj = _mm256_broadcast_sd(&*x.add(j));
            let xj = _mm256_mul_pd(xj, _mm256_set1_pd(alpha));
            let row = a.add(j * lda + i0);
            let mut k = 0;
            // Four-lane unroll (mirrors the blas1 axpy kernel): four
            // independent load-FMA-store chains per iteration. The
            // single-chain loop this replaced measured ~1.4x slower per
            // element than axpy (fused serial fan-out lost 0.71x to the
            // axpy-per-row loop it was meant to beat). Same op per
            // element in the same order — bitwise identical.
            while k + 16 <= clen {
                let a0 = _mm256_loadu_pd(priv_y.add(k));
                let a1 = _mm256_loadu_pd(priv_y.add(k + 4));
                let a2 = _mm256_loadu_pd(priv_y.add(k + 8));
                let a3 = _mm256_loadu_pd(priv_y.add(k + 12));
                let v0 = _mm256_loadu_pd(row.add(k));
                let v1 = _mm256_loadu_pd(row.add(k + 4));
                let v2 = _mm256_loadu_pd(row.add(k + 8));
                let v3 = _mm256_loadu_pd(row.add(k + 12));
                _mm256_storeu_pd(priv_y.add(k), _mm256_fmadd_pd(xj, v0, a0));
                _mm256_storeu_pd(priv_y.add(k + 4), _mm256_fmadd_pd(xj, v1, a1));
                _mm256_storeu_pd(priv_y.add(k + 8), _mm256_fmadd_pd(xj, v2, a2));
                _mm256_storeu_pd(priv_y.add(k + 12), _mm256_fmadd_pd(xj, v3, a3));
                k += 16;
            }
            while k + 4 <= clen {
                let acc = _mm256_loadu_pd(priv_y.add(k));
                let av = _mm256_loadu_pd(row.add(k));
                _mm256_storeu_pd(priv_y.add(k), _mm256_fmadd_pd(xj, av, acc));
                k += 4;
            }
            while k < clen {
                *priv_y.add(k) += alpha * *x.add(j) * *row.add(k);
                k += 1;
            }
        }
    }
}

/// f32 twin of [`sweep_fanout_tile_f64`].
#[target_feature(enable = "avx2,fma")]
unsafe fn sweep_fanout_tile_f32(
    j0: usize,
    j1: usize,
    i0: usize,
    clen: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    x: *const f32,
    priv_y: *mut f32,
) {
    // SAFETY: outer `unsafe fn` contract, bounds as above.
    unsafe {
        for j in j0..j1 {
            let xj = _mm256_broadcast_ss(&*x.add(j));
            let xj = _mm256_mul_ps(xj, _mm256_set1_ps(alpha));
            let row = a.add(j * lda + i0);
            let mut k = 0;
            // Four-lane unroll (mirrors the blas1 axpy kernel and the f64
            // twin above): the single-chain loop measured ~1.4x slower
            // per element than axpy. Bitwise identical.
            while k + 32 <= clen {
                let a0 = _mm256_loadu_ps(priv_y.add(k));
                let a1 = _mm256_loadu_ps(priv_y.add(k + 8));
                let a2 = _mm256_loadu_ps(priv_y.add(k + 16));
                let a3 = _mm256_loadu_ps(priv_y.add(k + 24));
                let v0 = _mm256_loadu_ps(row.add(k));
                let v1 = _mm256_loadu_ps(row.add(k + 8));
                let v2 = _mm256_loadu_ps(row.add(k + 16));
                let v3 = _mm256_loadu_ps(row.add(k + 24));
                _mm256_storeu_ps(priv_y.add(k), _mm256_fmadd_ps(xj, v0, a0));
                _mm256_storeu_ps(priv_y.add(k + 8), _mm256_fmadd_ps(xj, v1, a1));
                _mm256_storeu_ps(priv_y.add(k + 16), _mm256_fmadd_ps(xj, v2, a2));
                _mm256_storeu_ps(priv_y.add(k + 24), _mm256_fmadd_ps(xj, v3, a3));
                k += 32;
            }
            while k + 8 <= clen {
                let acc = _mm256_loadu_ps(priv_y.add(k));
                let av = _mm256_loadu_ps(row.add(k));
                _mm256_storeu_ps(priv_y.add(k), _mm256_fmadd_ps(xj, av, acc));
                k += 8;
            }
            while k < clen {
                *priv_y.add(k) += alpha * *x.add(j) * *row.add(k);
                k += 1;
            }
        }
    }
}

/// Spawn one sweep task per `rows_per_band` row band into `s` (f64).
/// Bands own disjoint `y` ranges; row dots are independent so the band
/// count never changes the numbers (bitwise identical at any split).
///
/// # Safety
///
/// Pointers addressable per the [`gemv_notrans_f64`] contract; AVX2 + FMA
/// live.
unsafe fn spawn_notrans_bands_f64(
    s: &rayon::Scope<'_>,
    m: usize,
    rows_per_band: usize,
    n: usize,
    alpha: f64,
    a: SendConstPtr<f64>,
    lda: usize,
    x: SendConstPtr<f64>,
    beta: f64,
    y: SendMutPtr<f64>,
    allow_quad: bool,
) {
    let mut t0 = 0;
    while t0 < m {
        let t1 = (t0 + rows_per_band).min(m);
        s.spawn(move |_| {
            let a: SendConstPtr<f64> = a;
            let x: SendConstPtr<f64> = x;
            let y: SendMutPtr<f64> = y;
            // SAFETY: disjoint `y` band; AVX2 checked by caller.
            unsafe {
                sweep_f64(t0, t1, n, alpha, a.0, lda, x.0, beta, y.0, allow_quad);
            }
        });
        t0 = t1;
    }
}

/// f32 twin of [`spawn_notrans_bands_f64`].
///
/// # Safety
///
/// Pointers addressable per the [`gemv_notrans_f32`] contract; AVX2 + FMA
/// live.
unsafe fn spawn_notrans_bands_f32(
    s: &rayon::Scope<'_>,
    m: usize,
    rows_per_band: usize,
    n: usize,
    alpha: f32,
    a: SendConstPtr<f32>,
    lda: usize,
    x: SendConstPtr<f32>,
    beta: f32,
    y: SendMutPtr<f32>,
    allow_quad: bool,
) {
    let mut t0 = 0;
    while t0 < m {
        let t1 = (t0 + rows_per_band).min(m);
        s.spawn(move |_| {
            let a: SendConstPtr<f32> = a;
            let x: SendConstPtr<f32> = x;
            let y: SendMutPtr<f32> = y;
            // SAFETY: disjoint `y` band; AVX2 checked by caller.
            unsafe {
                sweep_f32(t0, t1, n, alpha, a.0, lda, x.0, beta, y.0, allow_quad);
            }
        });
        t0 = t1;
    }
}

/// `y[0..m] := alpha * A * x + beta * y[0..m]`, `A` row-major `m × n`.
///
/// # Safety
///
/// AVX2 + FMA required; slice bounds as in the scalar fallback
/// (`a.len() >= (m-1)*lda + n` elements addressable, `x.len() >= n`,
/// `y.len() >= m`).
pub unsafe fn gemv_notrans_f64(
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) {
    debug_assert!(is_available());
    if m == 0 {
        return;
    }
    if alpha == 0.0 || n == 0 {
        let y_active = &mut y[..m];
        if beta == 0.0 {
            for yi in y_active.iter_mut() {
                *yi = 0.0;
            }
        } else if beta != 1.0 {
            for yi in y_active.iter_mut() {
                *yi *= beta;
            }
        }
        return;
    }
    if use_parallel(m, n) {
        // SAFETY: tasks own disjoint `y` bands; `A`/`x` are read-only.
        let a_send = SendConstPtr(a.as_ptr());
        let x_send = SendConstPtr(x.as_ptr());
        let y_send = SendMutPtr(y.as_mut_ptr());
        // Spread broadcast (see `gemv_pinned`): exactly one band per
        // spread worker, indexed by worker — the same rows on the same
        // pinned worker every rep, so each band stays in its own L3
        // slice. The pool is ambient-sized (see
        // `spread_pool_for_ambient`), so sub-width callers broadcast
        // too; falls back to the raw scope when nested (broadcast from
        // inside a pool risks deadlock), below 2 ambient threads, or
        // when no spread pool could be built (topology unreadable).
        if gemv_pinned()
            && rayon::current_thread_index().is_none()
            && let Some(pool) = crate::cpu::blas3::spread_pool_for_ambient()
        {
            let width = pool.current_num_threads().max(1);
            let bands = width.min(m).max(1);
            // Multiple-of-4 band width: every band starts on a 4-row
            // boundary, so quad dots never span a band edge and any
            // banding computes bitwise-identical rows (see the sweep
            // NOTE). Rounded UP (a tail worker may idle) rather than
            // down (a fat tail band would unbalance); power-of-2 m
            // (all production shapes) is already aligned, so this is
            // free there.
            let rows_per_band = m.div_ceil(bands).next_multiple_of(4).max(4);
            pool.broadcast(|ctx| {
                // Rebind whole structs (not `.0` fields) so the closure
                // captures `Send + Sync` wrappers, not raw pointers.
                let a_send: SendConstPtr<f64> = a_send;
                let x_send: SendConstPtr<f64> = x_send;
                let y_send: SendMutPtr<f64> = y_send;
                let w = ctx.index();
                let t0 = w.saturating_mul(rows_per_band).min(m);
                let t1 = t0.saturating_add(rows_per_band).min(m);
                if t0 < t1 {
                    // SAFETY: worker `w` owns `y[t0..t1]` (bands are
                    // disjoint by construction); `A`/`x` read-only;
                    // AVX2 checked by caller. `broadcast` joins before
                    // returning, so the borrows cannot outlive the call.
                    unsafe {
                        sweep_f64(
                            t0,
                            t1,
                            n,
                            alpha,
                            a_send.0,
                            lda,
                            x_send.0,
                            beta,
                            y_send.0,
                            width > 1,
                        );
                    }
                }
            });
            return;
        }
        rayon::scope(|s| unsafe {
            // Quad dots only with >1 worker: a single-threaded scope is
            // MLP-bound like the serial path and loses on quads.
            let allow_quad = rayon::current_num_threads() > 1;
            spawn_notrans_bands_f64(
                s, m, BAND_ROWS, n, alpha, a_send, lda, x_send, beta, y_send, allow_quad,
            );
        });
        return;
    }
    // SAFETY: AVX2 checked by caller; full bounds hold.
    unsafe {
        sweep_f64(
            0,
            m,
            n,
            alpha,
            a.as_ptr(),
            lda,
            x.as_ptr(),
            beta,
            y.as_mut_ptr(),
            false,
        );
    }
}

/// Whether the transposed fan-out below pays off: enough total work and
/// enough columns to split. Sized at 1M elements (gemvtransmt MT-vs-serial
/// sweep 2026-09-27: MT loses 1.4x at 512², wins 1.3-1.75x at 1024² and
/// 1.8-6x at 1536²+): below, dispatch + privatisation cost more than
/// threading returns; above, the serial axpy loop additionally falls off
/// a local-L3-slice cliff (single-threaded `A` streams past ~16MB serve
/// from remote slices/DRAM, while MT bands stay slice-local), so MT wins
/// up to 6-8x there. `m, n >= 256` keeps degenerate-narrow shapes serial.
#[inline]
pub(crate) fn trans_parallel_worthwhile(m: usize, n: usize) -> bool {
    m >= 256 && n >= 256 && m.saturating_mul(n) >= 1_048_576
}

/// MT-vs-serial dispatch for the transposed fan-out: the worthwhile
/// gate above, unless `SONIUM_GEMV_TRANS_MT` forces a side (`1` = always
/// the privatised MT path, `0` = always the serial axpy loop). Default-off
/// measurement override — unset leaves dispatch untouched. Added
/// 2026-09-27 to A/B the small-shape MT hole (sq1kT runs serial at
/// ~0.13ms while OB threads to ~0.03ms): forcing MT shows whether our
/// tiled scheme or the gate itself is at fault. Read per call so
/// same-process A/B can interleave both arms.
/// Pure MT-vs-serial routing for the transposed fan-out (unit-testable;
/// see [`trans_use_mt`]). `mt_override` is the parsed
/// `SONIUM_GEMV_TRANS_MT` (`Some(true)`/`Some(false)` force a side).
fn trans_routing(m: usize, n: usize, threads: usize, mt_override: Option<bool>) -> bool {
    if let Some(force) = mt_override {
        return force;
    }
    // Single-threaded solves stay serial: the tiled MT path pays
    // partials alloc + task spawn + reduce for one worker (pure
    // overhead — serial wins 1.15-1.74x at 1T on every measured
    // square trans shape, both types). Mirrors TRSM's gate.
    trans_parallel_worthwhile(m, n) && threads > 1
}

#[inline]
fn trans_use_mt(m: usize, n: usize) -> bool {
    let mt_override = match std::env::var("SONIUM_GEMV_TRANS_MT").as_deref() {
        Ok("1") => Some(true),
        Ok("0") => Some(false),
        _ => None,
    };
    trans_routing(m, n, rayon::current_num_threads().max(1), mt_override)
}

/// Trans-MT 1x/4x task-budget boundary in matrix bytes: below 32 MiB
/// the problem takes one task per worker, at or above it four per worker. The
/// old 2M-element line put 1536² (2.36M els) on 4x, where rows/block
/// halves (64→48) while dispatch/partials/reduce quadruple: f32-1536
/// trailed 1.3-1.45x at 4x and reaches parity at 1x (forced-1X/4X
/// scratch-build A/B 2026-09-27). The byte (not element) line keeps
/// precision behavior aligned: f32 flips at 8M els (1536²/2048²/rects
/// all measured wins at 1x: 1536²→parity, 2048² −15% ours-absolute,
/// r1k4kT 1.36→1.20x, r4k1kT 1.30→0.93x), f64 at 4M els (1536²
/// helped 1.7-1.8x→1.5-1.65x; 2048² 1x-vs-4x tie so 4x is harmless;
/// rects/4k+ unchanged at 4x). sq1kT stays 1x, sq4kT+ stays 4x both
/// precisions (the 1x-regression evidence for large shapes is untouched).
#[cfg(target_arch = "x86_64")]
const TRANS_1X_MAX_BYTES: usize = 32 << 20;

/// Whether Trans-MT takes the 1x task budget for an `m × n` problem with
/// `elsize`-byte elements. Pure so the boundary is unit-testable.
#[cfg(target_arch = "x86_64")]
#[inline]
fn trans_tasks_1x(m: usize, n: usize, elsize: usize) -> bool {
    // Strict `<`: exactly-32MiB shapes (f64 2048² / 4.2M-el rects) stay
    // 4x (measured tie-or-win there), everything below goes 1x.
    m.saturating_mul(n).saturating_mul(elsize) < TRANS_1X_MAX_BYTES
}

/// `y[0..m] := alpha * Aᵀ * x + beta * y[0..m]` with `A` stored `n × m`
/// row-major (leading dimension `lda >= m`).
///
/// Columns are split into blocks; each worker accumulates its block into a
/// private buffer (no `xj == 0` skips, so NaN propagation matches the
/// serial reference) and the partials reduce into `y` in block order.
/// Summation order differs from the serial fan-out — rounding-level only,
/// inside the 1e-12 parity tolerances.
///
/// # Safety
///
/// AVX2 + FMA required; `a` holds `n` rows of stride `lda`,
/// `x.len() >= n`, `y.len() >= m`.
pub unsafe fn gemv_trans_f64(
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) {
    debug_assert!(is_available());
    if m == 0 {
        return;
    }
    if beta == 0.0 {
        for v in y[..m].iter_mut() {
            *v = 0.0;
        }
    } else if beta != 1.0 {
        for v in y[..m].iter_mut() {
            *v *= beta;
        }
    }
    if alpha == 0.0 {
        return;
    }
    if !trans_use_mt(m, n) {
        // NOTE (2026-09-27): a fused serial fan-out (one
        // `sweep_fanout_tile` call over `y` itself instead of one axpy
        // per row) was trialled behind `SONIUM_GEMV_TRANS_FUSED` and
        // reverted — interleaved ST A/B (reproduced 2/2) measured
        // shape-schizophrenic results (1024²T 1.14-1.24x wins, 2048²T
        // ties, 512x2048T 0.68-0.93x losses, 2048x512T splits by
        // precision). The 2-3.7x vendor gap needs register-blocked
        // multi-row fan-out, not dispatch fusion; do not retry
        // single-row fusion. (The 4x tile unroll shipped separately.)
        // Four-row fused fan-out: `y` stays in registers across 4
        // rows (4x fewer y-stores than one axpy per row at the same
        // A-traffic). A runtime-`k` loop version ties the axpy loop
        // (per-row broadcast/loop overhead eats the fusion gain), so
        // the rows are unrolled by construction here; the <4 tail
        // keeps single axpys. Bitwise identical to the axpy loop.
        let mut j = 0;
        while j + 4 <= n {
            let c0 = alpha * x[j];
            let c1 = alpha * x[j + 1];
            let c2 = alpha * x[j + 2];
            let c3 = alpha * x[j + 3];
            // SAFETY: AVX2+FMA required by this fn's contract
            // (`debug_assert!(is_available())` above); all slices
            // in-bounds by construction (`lda >= m` per contract).
            unsafe {
                crate::cpu::blas1::avx2::fanout_4row_f64(
                    c0,
                    c1,
                    c2,
                    c3,
                    &a[j * lda..j * lda + m],
                    &a[(j + 1) * lda..(j + 1) * lda + m],
                    &a[(j + 2) * lda..(j + 2) * lda + m],
                    &a[(j + 3) * lda..(j + 3) * lda + m],
                    &mut y[..m],
                );
            }
            j += 4;
        }
        for j in j..n {
            let xj = alpha * x[j];
            crate::cpu::blas1::axpy_f64(xj, &a[j * lda..j * lda + m], &mut y[..m]);
        }
        return;
    }
    // 2D-tiled privatised fan-out: the output is split into `CHUNK`-element
    // tiles so each task's private band (8 KiB) stays L1-resident across its
    // rows, while `A` is still read exactly once. An untiled band spans the
    // whole `y` and spills out of L1 on every row (~2x slower at large `m`).
    const CHUNK: usize = 1024;
    let workers = rayon::current_num_threads().max(1);
    // Task budget: one task per worker on small shapes (see
    // `trans_tasks_1x`), four per worker above. Bands are uniform, so on
    // small shapes oversubscription buys no balance and 4x tasks only
    // multiply dispatch, partials, and reduce cost; on large shapes the
    // extra tasks win back more than they cost (sq4kT/sq8kT regress
    // 1.2-1.8x at 1x — A/B 2026-09-27), plausibly by keeping each task's
    // concurrent `A` stream fittable when the unpinned pool clusters on
    // few L3 slices.
    let tasks = if trans_tasks_1x(m, n, core::mem::size_of::<f64>()) {
        workers.max(2)
    } else {
        workers.saturating_mul(4).max(2)
    };
    let nchunks = m.div_ceil(CHUNK).max(1);
    let blocks = (n / 32).clamp(1, tasks.div_ceil(nchunks)).max(1);
    // One flat partials buffer (`blocks` × `nchunks` bands of `CHUNK`, the
    // last chunk's tail padding staying zero); each task owns one band and
    // the ordered reduce runs after the scope joins.
    let mut partials = vec![0.0f64; blocks * nchunks * CHUNK];
    // SAFETY: task `(b, c)` writes only band `b * nchunks + c`; `A`/`x` are
    // read-only.
    let a_send = SendConstPtr(a.as_ptr());
    let x_send = SendConstPtr(x.as_ptr());
    let p_send = SendMutPtr(partials.as_mut_ptr());
    // Spread-broadcast dispatch (see `gemv_pinned`): static cells per
    // pinned worker instead of the stealing raw scope. Shipped
    // 2026-09-27 on same-binary A/B (16T): broadcast beats scope on
    // every Trans shape (sq1kT 1.6-1.7x, sq4kT 1.9-5.9x, sq8kT
    // 1.4-2.5x), taking Trans MT from 1.06-2.9x vs OpenBLAS to
    // 0.22-1.47x (parity-or-better). Same mechanism as NoTrans:
    // stealing migrates bands across L3 slices every rep; static
    // bands settle slice-local (sq4kT-f64 hot hits 361GB/s
    // aggregate L3 vs scope's 61GB/s).
    if gemv_pinned()
        && rayon::current_thread_index().is_none()
        && let Some(pool) = crate::cpu::blas3::spread_pool_for_ambient()
    {
        let width = pool.current_num_threads().max(1);
        let ncells = blocks * nchunks;
        let cells_per = ncells.div_ceil(width).max(1);
        let jrows = n.div_ceil(blocks).max(1);
        pool.broadcast(|ctx| {
            let a_send: SendConstPtr<f64> = a_send;
            let x_send: SendConstPtr<f64> = x_send;
            let p_send: SendMutPtr<f64> = p_send;
            let w = ctx.index();
            let c0 = w.saturating_mul(cells_per).min(ncells);
            let c1 = (w.saturating_add(1).saturating_mul(cells_per)).min(ncells);
            for cell in c0..c1 {
                let b = cell / nchunks;
                let c = cell % nchunks;
                let j0 = b.saturating_mul(jrows).min(n);
                let j1 = (b.saturating_add(1).saturating_mul(jrows)).min(n);
                if j0 >= j1 {
                    continue;
                }
                let i0 = c * CHUNK;
                let clen = (m - i0).min(CHUNK);
                // SAFETY: worker `w` owns cells `c0..c1` (disjoint by
                // construction); bands zeroed via `vec![0.0; ..]`.
                // `broadcast` joins before returning.
                unsafe {
                    sweep_fanout_tile_f64(
                        j0,
                        j1,
                        i0,
                        clen,
                        alpha,
                        a_send.0,
                        lda,
                        x_send.0,
                        p_send.0.add(cell * CHUNK),
                    );
                }
            }
        });
    } else {
        rayon::scope(|s| {
            for c in 0..nchunks {
                let i0 = c * CHUNK;
                let clen = (m - i0).min(CHUNK);
                let mut j0 = 0;
                let mut b = 0;
                while j0 < n {
                    let j1 = (j0 + n.div_ceil(blocks)).min(n);
                    s.spawn(move |_| {
                        let a_send: SendConstPtr<f64> = a_send;
                        let x_send: SendConstPtr<f64> = x_send;
                        let p_send: SendMutPtr<f64> = p_send;
                        // SAFETY: task `(b, c)` owns band `b * nchunks + c`
                        // (zeroed via `vec![0.0; ..]`); rows `j0..j1` carry
                        // `clen` elements from column `i0` (`lda >= m`).
                        // One fused call per task — not one `axpy` per row.
                        unsafe {
                            sweep_fanout_tile_f64(
                                j0,
                                j1,
                                i0,
                                clen,
                                alpha,
                                a_send.0,
                                lda,
                                x_send.0,
                                p_send.0.add((b * nchunks + c) * CHUNK),
                            );
                        }
                    });
                    b += 1;
                    j0 = j1;
                }
            }
        });
    }
    for c in 0..nchunks {
        let i0 = c * CHUNK;
        let clen = (m - i0).min(CHUNK);
        let y_chunk = &mut y[i0..i0 + clen];
        for b in 0..blocks {
            crate::cpu::blas1::axpy_f64(
                1.0,
                &partials[(b * nchunks + c) * CHUNK..(b * nchunks + c) * CHUNK + clen],
                y_chunk,
            );
        }
    }
}

/// f32 twin of [`gemv_trans_f64`].
///
/// # Safety
///
/// AVX2 + FMA required; slice bounds as above.
pub unsafe fn gemv_trans_f32(
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    x: &[f32],
    beta: f32,
    y: &mut [f32],
) {
    debug_assert!(is_available());
    if m == 0 {
        return;
    }
    if beta == 0.0 {
        for v in y[..m].iter_mut() {
            *v = 0.0;
        }
    } else if beta != 1.0 {
        for v in y[..m].iter_mut() {
            *v *= beta;
        }
    }
    if alpha == 0.0 {
        return;
    }
    if !trans_use_mt(m, n) {
        // Four-row fused fan-out (f32 twin of the f64 driver above);
        // the <4 tail keeps single axpys. Bitwise identical to the
        // axpy loop.
        let mut j = 0;
        while j + 4 <= n {
            let c0 = alpha * x[j];
            let c1 = alpha * x[j + 1];
            let c2 = alpha * x[j + 2];
            let c3 = alpha * x[j + 3];
            // SAFETY: AVX2+FMA required by this fn's contract
            // (`debug_assert!(is_available())` above); all slices
            // in-bounds by construction (`lda >= m` per contract).
            unsafe {
                crate::cpu::blas1::avx2::fanout_4row_f32(
                    c0,
                    c1,
                    c2,
                    c3,
                    &a[j * lda..j * lda + m],
                    &a[(j + 1) * lda..(j + 1) * lda + m],
                    &a[(j + 2) * lda..(j + 2) * lda + m],
                    &a[(j + 3) * lda..(j + 3) * lda + m],
                    &mut y[..m],
                );
            }
            j += 4;
        }
        for j in j..n {
            let xj = alpha * x[j];
            crate::cpu::blas1::axpy_f32(xj, &a[j * lda..j * lda + m], &mut y[..m]);
        }
        return;
    }
    // 2D-tiled privatised fan-out, mirroring `gemv_trans_f64` above
    // (2048 `f32` = 8 KiB per band, L1-resident).
    const CHUNK: usize = 2048;
    let workers = rayon::current_num_threads().max(1);
    // Task sizing mirrors `gemv_trans_f64` above (1x/4x budget by size).
    let tasks = if trans_tasks_1x(m, n, core::mem::size_of::<f32>()) {
        workers.max(2)
    } else {
        workers.saturating_mul(4).max(2)
    };
    let nchunks = m.div_ceil(CHUNK).max(1);
    let blocks = (n / 32).clamp(1, tasks.div_ceil(nchunks)).max(1);
    let mut partials = vec![0.0f32; blocks * nchunks * CHUNK];
    // SAFETY: task `(b, c)` writes only band `b * nchunks + c`; `A`/`x` are
    // read-only.
    let a_send = SendConstPtr(a.as_ptr());
    let x_send = SendConstPtr(x.as_ptr());
    let p_send = SendMutPtr(partials.as_mut_ptr());
    // Spread-broadcast dispatch (f32 twin of the f64 arm above;
    // see `gemv_pinned` for the ship rationale).
    if gemv_pinned()
        && rayon::current_thread_index().is_none()
        && let Some(pool) = crate::cpu::blas3::spread_pool_for_ambient()
    {
        let width = pool.current_num_threads().max(1);
        let ncells = blocks * nchunks;
        let cells_per = ncells.div_ceil(width).max(1);
        let jrows = n.div_ceil(blocks).max(1);
        pool.broadcast(|ctx| {
            let a_send: SendConstPtr<f32> = a_send;
            let x_send: SendConstPtr<f32> = x_send;
            let p_send: SendMutPtr<f32> = p_send;
            let w = ctx.index();
            let c0 = w.saturating_mul(cells_per).min(ncells);
            let c1 = (w.saturating_add(1).saturating_mul(cells_per)).min(ncells);
            for cell in c0..c1 {
                let b = cell / nchunks;
                let c = cell % nchunks;
                let j0 = b.saturating_mul(jrows).min(n);
                let j1 = (b.saturating_add(1).saturating_mul(jrows)).min(n);
                if j0 >= j1 {
                    continue;
                }
                let i0 = c * CHUNK;
                let clen = (m - i0).min(CHUNK);
                // SAFETY: as in the f64 twin above.
                unsafe {
                    sweep_fanout_tile_f32(
                        j0,
                        j1,
                        i0,
                        clen,
                        alpha,
                        a_send.0,
                        lda,
                        x_send.0,
                        p_send.0.add(cell * CHUNK),
                    );
                }
            }
        });
    } else {
        rayon::scope(|s| {
            for c in 0..nchunks {
                let i0 = c * CHUNK;
                let clen = (m - i0).min(CHUNK);
                let mut j0 = 0;
                let mut b = 0;
                while j0 < n {
                    let j1 = (j0 + n.div_ceil(blocks)).min(n);
                    s.spawn(move |_| {
                        let a_send: SendConstPtr<f32> = a_send;
                        let x_send: SendConstPtr<f32> = x_send;
                        let p_send: SendMutPtr<f32> = p_send;
                        // SAFETY: as in the `f64` twin above — one fused call
                        // per task, not one `axpy` per row.
                        unsafe {
                            sweep_fanout_tile_f32(
                                j0,
                                j1,
                                i0,
                                clen,
                                alpha,
                                a_send.0,
                                lda,
                                x_send.0,
                                p_send.0.add((b * nchunks + c) * CHUNK),
                            );
                        }
                    });
                    b += 1;
                    j0 = j1;
                }
            }
        });
    }
    for c in 0..nchunks {
        let i0 = c * CHUNK;
        let clen = (m - i0).min(CHUNK);
        let y_chunk = &mut y[i0..i0 + clen];
        for b in 0..blocks {
            crate::cpu::blas1::axpy_f32(
                1.0,
                &partials[(b * nchunks + c) * CHUNK..(b * nchunks + c) * CHUNK + clen],
                y_chunk,
            );
        }
    }
}

/// f32 twin of [`gemv_notrans_f64`].
///
/// # Safety
///
/// AVX2 + FMA required; slice bounds as in the scalar fallback.
pub unsafe fn gemv_notrans_f32(
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    x: &[f32],
    beta: f32,
    y: &mut [f32],
) {
    debug_assert!(is_available());
    if m == 0 {
        return;
    }
    if alpha == 0.0 || n == 0 {
        let y_active = &mut y[..m];
        if beta == 0.0 {
            for yi in y_active.iter_mut() {
                *yi = 0.0;
            }
        } else if beta != 1.0 {
            for yi in y_active.iter_mut() {
                *yi *= beta;
            }
        }
        return;
    }
    if use_parallel(m, n) {
        // SAFETY: tasks own disjoint `y` bands; `A`/`x` are read-only.
        let a_send = SendConstPtr(a.as_ptr());
        let x_send = SendConstPtr(x.as_ptr());
        let y_send = SendMutPtr(y.as_mut_ptr());
        // Spread broadcast: f32 twin of the f64 branch above.
        if gemv_pinned()
            && rayon::current_thread_index().is_none()
            && let Some(pool) = crate::cpu::blas3::spread_pool_for_ambient()
        {
            let width = pool.current_num_threads().max(1);
            let bands = width.min(m).max(1);
            // Multiple-of-4 band width (see the f64 NOTE above): quad
            // dots never span a band edge; free on power-of-2 m.
            let rows_per_band = m.div_ceil(bands).next_multiple_of(4).max(4);
            pool.broadcast(|ctx| {
                // Rebind whole structs (not `.0` fields) so the closure
                // captures `Send + Sync` wrappers, not raw pointers.
                let a_send: SendConstPtr<f32> = a_send;
                let x_send: SendConstPtr<f32> = x_send;
                let y_send: SendMutPtr<f32> = y_send;
                let w = ctx.index();
                let t0 = w.saturating_mul(rows_per_band).min(m);
                let t1 = t0.saturating_add(rows_per_band).min(m);
                if t0 < t1 {
                    // SAFETY: as in the f64 twin above.
                    unsafe {
                        sweep_f32(
                            t0,
                            t1,
                            n,
                            alpha,
                            a_send.0,
                            lda,
                            x_send.0,
                            beta,
                            y_send.0,
                            width > 1,
                        );
                    }
                }
            });
            return;
        }
        rayon::scope(|s| unsafe {
            // Quad dots only with >1 worker (see the f64 NOTE above).
            let allow_quad = rayon::current_num_threads() > 1;
            spawn_notrans_bands_f32(
                s, m, BAND_ROWS, n, alpha, a_send, lda, x_send, beta, y_send, allow_quad,
            );
        });
        return;
    }
    // SAFETY: AVX2 checked by caller; full bounds hold.
    unsafe {
        sweep_f32(
            0,
            m,
            n,
            alpha,
            a.as_ptr(),
            lda,
            x.as_ptr(),
            beta,
            y.as_mut_ptr(),
            false,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::blas2::scalar;
    use crate::cpu::parallelism::env_test_lock;

    fn check_f64(m: usize, n: usize, lda: usize, alpha: f64, beta: f64) {
        if !is_available() {
            return;
        }
        assert!(lda >= n);
        let a: Vec<f64> = (0..m * lda).map(|i| (i as f64 * 0.013).sin()).collect();
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.017).cos()).collect();
        let y0: Vec<f64> = (0..m).map(|i| (i as f64 * 0.011).sin()).collect();
        let mut got = y0.clone();
        unsafe {
            gemv_notrans_f64(m, n, alpha, &a, lda, &x, beta, &mut got);
        }
        let mut want = y0;
        scalar::gemv_f64(
            crate::backend::Transpose::NoTrans,
            m,
            n,
            alpha,
            &a,
            lda,
            &x,
            beta,
            &mut want,
        );
        for (g, w) in got.iter().zip(want.iter()) {
            let tol = 1e-12 * w.abs().max(1.0);
            assert!(
                (g - w).abs() <= tol,
                "mismatch {g} vs {w} at ({m},{n},lda={lda})"
            );
        }
    }

    fn check_f32(m: usize, n: usize, lda: usize, alpha: f32, beta: f32) {
        if !is_available() {
            return;
        }
        assert!(lda >= n);
        let a: Vec<f32> = (0..m * lda).map(|i| (i as f32 * 0.013).sin()).collect();
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017).cos()).collect();
        let y0: Vec<f32> = (0..m).map(|i| (i as f32 * 0.011).sin()).collect();
        let mut got = y0.clone();
        unsafe {
            gemv_notrans_f32(m, n, alpha, &a, lda, &x, beta, &mut got);
        }
        let mut want = y0;
        scalar::gemv_f32(
            crate::backend::Transpose::NoTrans,
            m,
            n,
            alpha,
            &a,
            lda,
            &x,
            beta,
            &mut want,
        );
        for (g, w) in got.iter().zip(want.iter()) {
            let tol = 1e-5 * w.abs().max(1.0);
            assert!(
                (g - w).abs() <= tol,
                "mismatch {g} vs {w} at ({m},{n},lda={lda})"
            );
        }
    }

    #[test]
    fn sweep_f32_pairing_bitwise() {
        if !is_available() {
            return;
        }
        // Quad arms below flip SONIUM_GEMV_DOT4: serialize with the
        // other env-flipping tests.
        let _guard = env_test_lock();
        // Paired rows must match per-row sweeps (all tails) to f32 dot
        // tolerance (7+7 chains reorder rounding vs the 8-chain single
        // row, so bitwise identity is not claimed). Compare one full
        // sweep (paired) against per-row sweeps.
        // SAFETY: AVX2+FMA gated above; all buffers sized per contract.
        unsafe {
            // Big-n cases (n>=1024) exercise the paired path through the
            // gate; small-n covers tails (tier + scalar) on both arms.
            for &(m, n) in &[(7, 100), (64, 300), (130, 17), (6, 1100), (5, 1031)] {
                let lda = n + 3;
                let a: Vec<f32> = (0..m * lda)
                    .map(|i| ((i * 37) as f32 * 0.001).sin())
                    .collect();
                let x: Vec<f32> = (0..n).map(|i| ((i * 91) as f32 * 0.001).cos()).collect();
                for &beta in &[0.0f32, 0.35] {
                    let y0: Vec<f32> = (0..m).map(|i| (i as f32 * 0.13 + 1.0).sin()).collect();
                    let mut paired = y0.clone();
                    sweep_f32(
                        0,
                        m,
                        n,
                        0.7,
                        a.as_ptr(),
                        lda,
                        x.as_ptr(),
                        beta,
                        paired.as_mut_ptr(),
                        false,
                    );
                    let mut single = y0.clone();
                    for r in 0..m {
                        sweep_f32(
                            r,
                            r + 1,
                            n,
                            0.7,
                            a.as_ptr(),
                            lda,
                            x.as_ptr(),
                            beta,
                            single.as_mut_ptr(),
                            false,
                        );
                    }
                    let tol = 1e-5 * (n as f32).sqrt().max(1.0);
                    for (p, s) in paired.iter().zip(single.iter()) {
                        assert!(
                            (p - s).abs() <= tol,
                            "pairing differs at (m={m}, n={n}, beta={beta}): {p} vs {s}"
                        );
                    }
                    // Quad dispatch (gate =1 and =2) vs the same per-row
                    // reference: tolerance-equal (3+3+3+3 chains reorder
                    // rounding vs the 8-chain single row). Env-flipped per
                    // arm under env_test_lock; per-row sweeps are single-row
                    // regardless of gate (no quad/pair fits one row).
                    for gate in ["1", "2"] {
                        // env_test_lock serializes env flips (see above).
                        std::env::set_var("SONIUM_GEMV_DOT4", gate);
                        let mut quad = y0.clone();
                        sweep_f32(
                            0,
                            m,
                            n,
                            0.7,
                            a.as_ptr(),
                            lda,
                            x.as_ptr(),
                            beta,
                            quad.as_mut_ptr(),
                            true,
                        );
                        for (q, s) in quad.iter().zip(single.iter()) {
                            assert!(
                                (q - s).abs() <= tol,
                                "quad differs at (m={m}, n={n}, beta={beta}, gate={gate}): {q} vs {s}"
                            );
                        }
                    }
                    // As above; restores the legacy dispatch.
                    std::env::remove_var("SONIUM_GEMV_DOT4");
                }
            }
        }
    }

    #[test]
    fn sweep_f64_pairing_tolerance() {
        if !is_available() {
            return;
        }
        // Quad arms below flip SONIUM_GEMV_DOT4: serialize with the
        // other env-flipping tests.
        let _guard = env_test_lock();
        // f64 twin of the f32 pairing gate. Big-n cases (n>=1024)
        // exercise the paired path through the gate; small-n covers
        // tails on both arms.
        // SAFETY: AVX2+FMA gated above; all buffers sized per contract.
        unsafe {
            for &(m, n) in &[(7, 100), (64, 300), (6, 1100), (5, 1031)] {
                let lda = n + 3;
                let a: Vec<f64> = (0..m * lda)
                    .map(|i| ((i * 37) as f64 * 0.001).sin())
                    .collect();
                let x: Vec<f64> = (0..n).map(|i| ((i * 91) as f64 * 0.001).cos()).collect();
                for &beta in &[0.0f64, 0.35] {
                    let y0: Vec<f64> = (0..m).map(|i| (i as f64 * 0.13 + 1.0).sin()).collect();
                    let mut paired = y0.clone();
                    sweep_f64(
                        0,
                        m,
                        n,
                        0.7,
                        a.as_ptr(),
                        lda,
                        x.as_ptr(),
                        beta,
                        paired.as_mut_ptr(),
                        false,
                    );
                    let mut single = y0.clone();
                    for r in 0..m {
                        sweep_f64(
                            r,
                            r + 1,
                            n,
                            0.7,
                            a.as_ptr(),
                            lda,
                            x.as_ptr(),
                            beta,
                            single.as_mut_ptr(),
                            false,
                        );
                    }
                    let tol = 1e-12 * (n as f64).sqrt().max(1.0);
                    for (p, s) in paired.iter().zip(single.iter()) {
                        assert!(
                            (p - s).abs() <= tol,
                            "pairing differs at (m={m}, n={n}, beta={beta}): {p} vs {s}"
                        );
                    }
                    // Quad dispatch (gate =1 and =2) vs the same per-row
                    // reference: tolerance-equal (3+3+3+3 chains reorder
                    // rounding vs the 8-chain single row). Env-flipped per
                    // arm under env_test_lock; per-row sweeps are single-row
                    // regardless of gate (no quad/pair fits one row).
                    for gate in ["1", "2"] {
                        // env_test_lock serializes env flips (see above).
                        std::env::set_var("SONIUM_GEMV_DOT4", gate);
                        let mut quad = y0.clone();
                        sweep_f64(
                            0,
                            m,
                            n,
                            0.7,
                            a.as_ptr(),
                            lda,
                            x.as_ptr(),
                            beta,
                            quad.as_mut_ptr(),
                            true,
                        );
                        for (q, s) in quad.iter().zip(single.iter()) {
                            assert!(
                                (q - s).abs() <= tol,
                                "quad differs at (m={m}, n={n}, beta={beta}, gate={gate}): {q} vs {s}"
                            );
                        }
                    }
                    // As above; restores the legacy dispatch.
                    std::env::remove_var("SONIUM_GEMV_DOT4");
                }
            }
        }
    }

    /// Quad tiling is split-invariant on 4-aligned cuts: splits at
    /// rows divisible by 4 (36, 100, 104 — all production bandings
    /// are 4-aligned by construction) match the unsplit sweep
    /// BITWISE, both precisions and both quad gates; misaligned cuts
    /// (38, 101) span quads and are tolerance-equal only. Without the
    /// canonical-alignment pre-drain the aligned arms group rows
    /// differently and differ in the last ulp.
    #[test]
    fn sweep_quad_split_bitwise() {
        if !is_available() {
            return;
        }
        let _guard = env_test_lock();
        for gate in ["1", "2"] {
            // SAFETY: env_test_lock serializes env flips (see above).
            unsafe {
                std::env::set_var("SONIUM_GEMV_DOT4", gate);
            }
            // SAFETY: AVX2+FMA gated above; all buffers sized per contract.
            unsafe {
                for &(m, n, lda) in &[(200, 300, 303), (200, 1100, 1103)] {
                    let a64: Vec<f64> = (0..m * lda).map(|i| (i as f64 * 0.013).sin()).collect();
                    let x64: Vec<f64> = (0..n).map(|i| (i as f64 * 0.017).cos()).collect();
                    let a32: Vec<f32> = (0..m * lda).map(|i| (i as f32 * 0.013).sin()).collect();
                    let x32: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017).cos()).collect();
                    for &beta in &[0.0, 0.35] {
                        let y064: Vec<f64> = (0..m).map(|i| (i as f64 * 0.011).sin()).collect();
                        let y032: Vec<f32> = (0..m).map(|i| (i as f32 * 0.011).sin()).collect();
                        let mut whole64 = y064.clone();
                        sweep_f64(
                            0,
                            m,
                            n,
                            0.7,
                            a64.as_ptr(),
                            lda,
                            x64.as_ptr(),
                            beta,
                            whole64.as_mut_ptr(),
                            true,
                        );
                        let mut whole32 = y032.clone();
                        sweep_f32(
                            0,
                            m,
                            n,
                            0.7,
                            a32.as_ptr(),
                            lda,
                            x32.as_ptr(),
                            beta as f32,
                            whole32.as_mut_ptr(),
                            true,
                        );
                        for &cut in &[36, 100, 104] {
                            let mut split64 = y064.clone();
                            sweep_f64(
                                0,
                                cut,
                                n,
                                0.7,
                                a64.as_ptr(),
                                lda,
                                x64.as_ptr(),
                                beta,
                                split64.as_mut_ptr(),
                                true,
                            );
                            sweep_f64(
                                cut,
                                m,
                                n,
                                0.7,
                                a64.as_ptr(),
                                lda,
                                x64.as_ptr(),
                                beta,
                                split64.as_mut_ptr(),
                                true,
                            );
                            assert_eq!(
                                whole64, split64,
                                "f64 split differs at (m={m}, n={n}, beta={beta}, cut={cut}, gate={gate})"
                            );
                            let mut split32 = y032.clone();
                            sweep_f32(
                                0,
                                cut,
                                n,
                                0.7,
                                a32.as_ptr(),
                                lda,
                                x32.as_ptr(),
                                beta as f32,
                                split32.as_mut_ptr(),
                                true,
                            );
                            sweep_f32(
                                cut,
                                m,
                                n,
                                0.7,
                                a32.as_ptr(),
                                lda,
                                x32.as_ptr(),
                                beta as f32,
                                split32.as_mut_ptr(),
                                true,
                            );
                            assert_eq!(
                                whole32, split32,
                                "f32 split differs at (m={m}, n={n}, beta={beta}, cut={cut}, gate={gate})"
                            );
                        }
                        // Misaligned cuts span quads: tolerance-equal only.
                        let tol64 = 1e-12 * (n as f64).sqrt().max(1.0);
                        let tol32 = 1e-5 * (n as f32).sqrt().max(1.0);
                        for &cut in &[38, 101] {
                            let mut split64 = y064.clone();
                            sweep_f64(
                                0,
                                cut,
                                n,
                                0.7,
                                a64.as_ptr(),
                                lda,
                                x64.as_ptr(),
                                beta,
                                split64.as_mut_ptr(),
                                true,
                            );
                            sweep_f64(
                                cut,
                                m,
                                n,
                                0.7,
                                a64.as_ptr(),
                                lda,
                                x64.as_ptr(),
                                beta,
                                split64.as_mut_ptr(),
                                true,
                            );
                            for (w, s) in whole64.iter().zip(split64.iter()) {
                                assert!(
                                    (w - s).abs() <= tol64,
                                    "f64 misaligned split exceeds tol at (m={m}, n={n}, beta={beta}, cut={cut}, gate={gate}): {w} vs {s}"
                                );
                            }
                            let mut split32 = y032.clone();
                            sweep_f32(
                                0,
                                cut,
                                n,
                                0.7,
                                a32.as_ptr(),
                                lda,
                                x32.as_ptr(),
                                beta as f32,
                                split32.as_mut_ptr(),
                                true,
                            );
                            sweep_f32(
                                cut,
                                m,
                                n,
                                0.7,
                                a32.as_ptr(),
                                lda,
                                x32.as_ptr(),
                                beta as f32,
                                split32.as_mut_ptr(),
                                true,
                            );
                            for (w, s) in whole32.iter().zip(split32.iter()) {
                                assert!(
                                    (w - s).abs() <= tol32,
                                    "f32 misaligned split exceeds tol at (m={m}, n={n}, beta={beta}, cut={cut}, gate={gate}): {w} vs {s}"
                                );
                            }
                        }
                    }
                }
            }
        }
        // SAFETY: as above; restores the default gate.
        unsafe {
            std::env::remove_var("SONIUM_GEMV_DOT4");
        }
    }

    #[test]
    fn avx2_gemv_notrans_matches_scalar() {
        for &(m, n, lda) in &[(1, 1, 1), (5, 37, 40), (64, 64, 64), (130, 17, 24)] {
            for &(alpha, beta) in &[(1.0, 0.0), (0.7, 0.35), (0.0, 0.5), (1.5, 1.0)] {
                check_f64(m, n, lda, alpha, beta);
                check_f32(m, n, lda, alpha as f32, beta as f32);
            }
        }
    }

    #[test]
    fn avx2_gemv_notrans_parallel_matches_scalar() {
        // 600 × 500 = 300K flops ≥ threshold: exercises the rayon bands,
        // including a row tail (600 % 64) and a column tail (500 % 16).
        for &(alpha, beta) in &[(0.9, 0.25), (1.0, 0.0)] {
            check_f64(600, 500, 500, alpha, beta);
            check_f32(600, 500, 500, alpha as f32, beta as f32);
        }
    }

    fn check_trans_f64(m: usize, n: usize, lda: usize, alpha: f64, beta: f64) {
        if !is_available() {
            return;
        }
        assert!(lda >= m);
        // Stored `n` rows of stride `lda >= m` (output length).
        let a: Vec<f64> = (0..n * lda).map(|i| (i as f64 * 0.013).sin()).collect();
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.017).cos()).collect();
        let y0: Vec<f64> = (0..m).map(|i| (i as f64 * 0.011).sin()).collect();
        let mut got = y0.clone();
        unsafe {
            gemv_trans_f64(m, n, alpha, &a, lda, &x, beta, &mut got);
        }
        let mut want = y0;
        scalar::gemv_f64(
            crate::backend::Transpose::Trans,
            m,
            n,
            alpha,
            &a,
            lda,
            &x,
            beta,
            &mut want,
        );
        for (g, w) in got.iter().zip(want.iter()) {
            let tol = 1e-12 * w.abs().max(1.0);
            assert!(
                (g - w).abs() <= tol,
                "mismatch {g} vs {w} at ({m},{n},lda={lda})"
            );
        }
    }

    fn check_trans_f32(m: usize, n: usize, lda: usize, alpha: f32, beta: f32) {
        if !is_available() {
            return;
        }
        assert!(lda >= m);
        let a: Vec<f32> = (0..n * lda).map(|i| (i as f32 * 0.013).sin()).collect();
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017).cos()).collect();
        let y0: Vec<f32> = (0..m).map(|i| (i as f32 * 0.011).sin()).collect();
        let mut got = y0.clone();
        unsafe {
            gemv_trans_f32(m, n, alpha, &a, lda, &x, beta, &mut got);
        }
        let mut want = y0;
        scalar::gemv_f32(
            crate::backend::Transpose::Trans,
            m,
            n,
            alpha,
            &a,
            lda,
            &x,
            beta,
            &mut want,
        );
        // f32 reduction-order error grows with column count (~n·eps);
        // 1e-4 is the honest bound at n ≈ 2000.
        for (g, w) in got.iter().zip(want.iter()) {
            let tol = 1e-4 * w.abs().max(1.0);
            assert!(
                (g - w).abs() <= tol,
                "mismatch {g} vs {w} at ({m},{n},lda={lda})"
            );
        }
    }

    #[test]
    fn avx2_gemv_trans_matches_scalar() {
        for &(m, n, lda) in &[(1, 1, 1), (9, 40, 40), (64, 70, 72)] {
            for &(alpha, beta) in &[(1.0, 0.0), (0.6, 0.4), (0.0, 0.5), (2.0, 1.0)] {
                check_trans_f64(m, n, lda, alpha, beta);
                check_trans_f32(m, n, lda, alpha as f32, beta as f32);
            }
        }
    }

    #[test]
    fn avx2_gemv_trans_parallel_matches_scalar() {
        // 2400 × 2200 = 5.3M ≥ threshold: exercises block splitting,
        // including a column-block tail.
        for &(alpha, beta) in &[(0.8, 0.3), (1.0, 0.0)] {
            check_trans_f64(2400, 2200, 2400, alpha, beta);
            check_trans_f32(2400, 2200, 2400, alpha as f32, beta as f32);
        }
    }

    /// `SONIUM_GEMV_TRANS_MT` override: forces the MT fan-out on a
    /// below-gate shape (512²) and the serial loop on an above-gate
    /// shape (2400×2200), both vs scalar. Also asserts the dispatch
    /// itself flips (not just path correctness) and that unset restores
    /// the gate default (serial below 1M elements, MT at/above).
    #[test]
    fn avx2_gemv_trans_mt_override_matches_scalar() {
        if !is_available() {
            return;
        }
        let _guard = env_test_lock();
        // SAFETY: `env_test_lock` serializes env flips; the switch is read per
        // call on the calling thread only, and no other thread reads it.
        unsafe {
            std::env::set_var("SONIUM_GEMV_TRANS_MT", "1");
        }
        assert!(trans_use_mt(512, 512));
        assert!(trans_use_mt(10, 10));
        check_trans_f64(512, 512, 512, 0.8, 0.3);
        check_trans_f32(512, 512, 512, 0.8, 0.3);
        // SAFETY: as above; forces the serial arm.
        unsafe {
            std::env::set_var("SONIUM_GEMV_TRANS_MT", "0");
        }
        assert!(!trans_use_mt(2400, 2200));
        assert!(!trans_use_mt(8192, 8192));
        check_trans_f64(2400, 2200, 2400, 0.8, 0.3);
        check_trans_f32(2400, 2200, 2400, 0.8, 0.3);
        // SAFETY: as above; restores gate-default dispatch.
        unsafe {
            std::env::remove_var("SONIUM_GEMV_TRANS_MT");
        }
        assert!(!trans_use_mt(512, 512));
        // Ambient pool width varies (1-thread harnesses stay serial),
        // so the positive MT assertions use the pure rule directly.
        assert!(trans_routing(1024, 1024, 16, None));
        assert!(trans_routing(2400, 2200, 16, None));
    }

    #[test]
    fn trans_routing_gate() {
        // Worthwhile + MT threads take the tiled path.
        assert!(trans_routing(1024, 1024, 16, None));
        assert!(trans_routing(2400, 2200, 2, None));
        // One thread stays serial even when worthwhile (measured win).
        assert!(!trans_routing(1024, 1024, 1, None));
        assert!(!trans_routing(8192, 8192, 1, None));
        // Below the worthwhile floor stays serial at any width.
        assert!(!trans_routing(512, 512, 16, None));
        assert!(!trans_routing(100, 100, 16, None));
        // Env override forces either side.
        assert!(trans_routing(10, 10, 1, Some(true)));
        assert!(!trans_routing(8192, 8192, 16, Some(false)));
    }
    /// Trans fan-out tile vs scalar, both precisions: beta-scale `y`,
    /// then one full-range tile over `y` itself. Covers the 4x-unrolled
    /// tile loop (tails, non-tight `lda`, NaN propagation) that the
    /// parallel fan-out path runs per task. (A fused *serial* arm using
    /// this exact sequence was trialled and reverted for
    /// shape-schizophrenic performance — see the NOTE in
    /// `gemv_trans_f64`.)
    #[test]
    fn avx2_gemv_trans_tile_matches_scalar() {
        if !is_available() {
            return;
        }
        for &(m, n, lda) in &[(1, 1, 1), (9, 40, 40), (130, 65, 136)] {
            for &(alpha, beta) in &[(1.0, 0.0), (0.6, 0.4), (2.0, 1.0)] {
                let a64: Vec<f64> = (0..n * lda).map(|i| (i as f64 * 0.013).sin()).collect();
                let x64: Vec<f64> = (0..n).map(|i| (i as f64 * 0.017).cos()).collect();
                let y064: Vec<f64> = (0..m).map(|i| (i as f64 * 0.011).sin()).collect();
                let mut got64 = y064.clone();
                unsafe {
                    if beta == 0.0 {
                        for v in got64[..m].iter_mut() {
                            *v = 0.0;
                        }
                    } else if beta != 1.0 {
                        for v in got64[..m].iter_mut() {
                            *v *= beta;
                        }
                    }
                    sweep_fanout_tile_f64(
                        0,
                        n,
                        0,
                        m,
                        alpha,
                        a64.as_ptr(),
                        lda,
                        x64.as_ptr(),
                        got64.as_mut_ptr(),
                    );
                }
                let mut want64 = y064;
                scalar::gemv_f64(
                    crate::backend::Transpose::Trans,
                    m,
                    n,
                    alpha,
                    &a64,
                    lda,
                    &x64,
                    beta,
                    &mut want64,
                );
                for (g, w) in got64.iter().zip(want64.iter()) {
                    let tol = 1e-12 * w.abs().max(1.0);
                    assert!(
                        (g - w).abs() <= tol,
                        "f64 mismatch {g} vs {w} at ({m},{n},lda={lda})"
                    );
                }
                let a32: Vec<f32> = (0..n * lda).map(|i| (i as f32 * 0.013).sin()).collect();
                let x32: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017).cos()).collect();
                let y032: Vec<f32> = (0..m).map(|i| (i as f32 * 0.011).sin()).collect();
                let mut got32 = y032.clone();
                unsafe {
                    let beta32 = beta as f32;
                    if beta32 == 0.0 {
                        for v in got32[..m].iter_mut() {
                            *v = 0.0;
                        }
                    } else if beta32 != 1.0 {
                        for v in got32[..m].iter_mut() {
                            *v *= beta32;
                        }
                    }
                    sweep_fanout_tile_f32(
                        0,
                        n,
                        0,
                        m,
                        alpha as f32,
                        a32.as_ptr(),
                        lda,
                        x32.as_ptr(),
                        got32.as_mut_ptr(),
                    );
                }
                let mut want32 = y032;
                scalar::gemv_f32(
                    crate::backend::Transpose::Trans,
                    m,
                    n,
                    alpha as f32,
                    &a32,
                    lda,
                    &x32,
                    beta as f32,
                    &mut want32,
                );
                for (g, w) in got32.iter().zip(want32.iter()) {
                    let tol = 1e-4 * w.abs().max(1.0);
                    assert!(
                        (g - w).abs() <= tol,
                        "f32 mismatch {g} vs {w} at ({m},{n},lda={lda})"
                    );
                }
            }
        }
        // NaN entry must propagate (the tile has no zero-skips).
        let (m, n, lda) = (64, 70, 72);
        let mut a: Vec<f64> = (0..n * lda).map(|i| (i as f64 * 0.013).sin()).collect();
        a[3 * lda + 5] = f64::NAN;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.017).cos()).collect();
        let mut got = vec![0.0f64; m];
        unsafe {
            sweep_fanout_tile_f64(
                0,
                n,
                0,
                m,
                1.0,
                a.as_ptr(),
                lda,
                x.as_ptr(),
                got.as_mut_ptr(),
            );
        }
        assert!(got[5].is_nan(), "NaN entry must propagate to y[5]");
    }

    /// Pinned NoTrans path (the `SONIUM_GEMV_PINNED=1` arm) vs scalar:
    /// exercises the spread broadcast through the public entry. Row dots
    /// are independent so any banding matches — including bitwise, which
    /// the second half asserts against the raw scope arm.
    #[test]
    fn avx2_gemv_notrans_pinned_matches_scalar() {
        if !is_available() {
            return;
        }
        let _guard = env_test_lock();
        // SAFETY: `env_test_lock` serializes env flips; the switch is read per
        // call on the calling thread only, and no other thread reads it.
        unsafe {
            std::env::set_var("SONIUM_GEMV_PINNED", "1");
        }
        // Parallel shape (also crosses a row tail at 600 % 64 on the raw
        // arm) plus a small shape that stays serial either way.
        for &(m, n, lda) in &[(600, 500, 500), (130, 17, 24)] {
            for &(alpha, beta) in &[(0.9, 0.25), (1.0, 0.0)] {
                check_f64(m, n, lda, alpha, beta);
                check_f32(m, n, lda, alpha as f32, beta as f32);
            }
        }
        // Bitwise raw-vs-broadcast agreement on the parallel shape.
        // NOTE: broadcast is default-on, so the raw arm needs an
        // explicit `=0` — `remove_var` would run broadcast twice and
        // the assertion would prove nothing.
        let (m, n, lda) = (600usize, 500usize, 500usize);
        let a64: Vec<f64> = (0..m * lda).map(|i| (i as f64 * 0.013).sin()).collect();
        let x64: Vec<f64> = (0..n).map(|i| (i as f64 * 0.017).cos()).collect();
        let y064: Vec<f64> = (0..m).map(|i| (i as f64 * 0.011).sin()).collect();
        let mut pin64 = y064.clone();
        let mut raw64 = y064.clone();
        unsafe {
            gemv_notrans_f64(m, n, 0.9, &a64, lda, &x64, 0.25, &mut pin64);
        }
        // SAFETY: as above; selects the legacy raw-scope arm.
        unsafe {
            std::env::set_var("SONIUM_GEMV_PINNED", "0");
        }
        unsafe {
            gemv_notrans_f64(m, n, 0.9, &a64, lda, &x64, 0.25, &mut raw64);
        }
        assert_eq!(pin64, raw64, "broadcast must match raw scope bitwise");
        // SAFETY: as above; restores the default-on state.
        unsafe {
            std::env::remove_var("SONIUM_GEMV_PINNED");
        }
    }

    /// Ambient-sized spread-pool selection is width-deterministic (no
    /// timing): inside a width-`w` pool the selected pool never exceeds
    /// `w` (no oversubscription) and floors to a cached 2/4/8/full
    /// width; at/above full width it is the full pool (legacy-identical
    /// there); `SONIUM_GEMV_AMBIENT=0` restores the legacy full-only
    /// rule (`None` below full width).
    #[test]
    fn spread_pool_for_ambient_floors_width() {
        use crate::cpu::blas3::spread_pool_for_ambient;
        let _guard = env_test_lock();
        // SAFETY: `env_test_lock` serializes env flips (see above).
        unsafe {
            std::env::remove_var("SONIUM_GEMV_AMBIENT");
        }
        let full_w = match crate::cpu::blas3::spread_pool() {
            Some(p) => p.current_num_threads().max(1),
            None => return, // No topology: nothing selectable anywhere.
        };
        let selected_in = |w: usize| -> Option<usize> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(w)
                .build()
                .unwrap();
            pool.install(|| spread_pool_for_ambient().map(|p| p.current_num_threads()))
        };
        for w in [1, 2, 3, 4, 6, 8, 12, 16, 32] {
            let sel = selected_in(w);
            match sel {
                None => assert!(w < 2, "ambient {w}: unexpected None"),
                Some(sw) => {
                    assert!(sw <= w, "ambient {w}: pool {sw} oversubscribes");
                    assert!(sw <= full_w, "ambient {w}: pool {sw} exceeds full {full_w}");
                    assert!(
                        sw == full_w || sw == 2 || sw == 4 || sw == 8,
                        "ambient {w}: pool {sw} is not a cached width"
                    );
                    if w >= full_w {
                        assert_eq!(sw, full_w, "ambient {w}: must take the full pool");
                    }
                }
            }
            // Exact floors on a standard 16-wide spread box.
            if full_w == 16 {
                let expect = match w {
                    1 => None,
                    2 | 3 => Some(2),
                    4 | 6 => Some(4),
                    8 | 12 => Some(8),
                    _ => Some(16),
                };
                assert_eq!(sel, expect, "ambient {w}: wrong floor");
            }
        }
        // Legacy rule: full-only below full width.
        // SAFETY: as above; selects the legacy arm.
        unsafe {
            std::env::set_var("SONIUM_GEMV_AMBIENT", "0");
        }
        for w in [1, 2, 4, 8] {
            assert_eq!(selected_in(w), None, "ambient {w}: legacy must be None");
        }
        assert_eq!(
            selected_in(full_w.max(16)),
            Some(full_w),
            "legacy keeps the full pool at/above full width"
        );
        // SAFETY: as above; restores the default-on state.
        unsafe {
            std::env::remove_var("SONIUM_GEMV_AMBIENT");
        }
    }

    /// Sub-width broadcast (ambient 8 → W8 pool) matches the raw scope
    /// bitwise: same argument as the full-width pinned test (row dots
    /// are banding-independent), exercised through the public entry.
    #[test]
    fn avx2_gemv_notrans_ambient8_matches_raw() {
        if !is_available() {
            return;
        }
        if crate::cpu::blas3::spread_pool().is_none() {
            return;
        }
        let _guard = env_test_lock();
        // SAFETY: `env_test_lock` serializes env flips (see above).
        unsafe {
            std::env::set_var("SONIUM_GEMV_PINNED", "1");
            std::env::remove_var("SONIUM_GEMV_AMBIENT");
        }
        let (m, n, lda) = (600usize, 500usize, 500usize);
        let a: Vec<f64> = (0..m * lda).map(|i| (i as f64 * 0.013).sin()).collect();
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.017).cos()).collect();
        let y0: Vec<f64> = (0..m).map(|i| (i as f64 * 0.011).sin()).collect();
        let pool8 = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        let mut got8 = y0.clone();
        pool8.install(|| unsafe {
            gemv_notrans_f64(m, n, 0.9, &a, lda, &x, 0.25, &mut got8);
        });
        // SAFETY: as above; selects the legacy raw-scope arm.
        unsafe {
            std::env::set_var("SONIUM_GEMV_PINNED", "0");
        }
        let mut raw = y0.clone();
        pool8.install(|| unsafe {
            gemv_notrans_f64(m, n, 0.9, &a, lda, &x, 0.25, &mut raw);
        });
        assert_eq!(
            got8, raw,
            "ambient-8 broadcast must match raw scope bitwise"
        );
        // SAFETY: as above; restores the default-on state.
        unsafe {
            std::env::remove_var("SONIUM_GEMV_PINNED");
        }
    }

    /// Concurrent top-level broadcasts against the shared spread pool:
    /// one per OS thread, all on the parallel shape. Proves the pool does
    /// not deadlock under concurrent `broadcast` (a hang would fail the
    /// suite) and that every caller gets its own correct result.
    #[test]
    fn avx2_gemv_notrans_concurrent_broadcast_matches() {
        if !is_available() {
            return;
        }
        let _guard = env_test_lock();
        // SAFETY: `env_test_lock` serializes env flips; held across the joins
        // so no sibling test can flip the switch mid-broadcast.
        unsafe {
            std::env::set_var("SONIUM_GEMV_PINNED", "1");
        }
        let (m, n, lda) = (600usize, 500usize, 500usize);
        let a: std::sync::Arc<Vec<f64>> =
            std::sync::Arc::new((0..m * lda).map(|i| (i as f64 * 0.013).sin()).collect());
        let x: std::sync::Arc<Vec<f64>> =
            std::sync::Arc::new((0..n).map(|i| (i as f64 * 0.017).cos()).collect());
        let y0: std::sync::Arc<Vec<f64>> =
            std::sync::Arc::new((0..m).map(|i| (i as f64 * 0.011).sin()).collect());
        // Single-threaded reference through the same public entry.
        let mut expected = y0.to_vec();
        unsafe {
            gemv_notrans_f64(m, n, 0.9, &a, lda, &x, 0.25, &mut expected);
        }
        let mut handles = Vec::new();
        for _ in 0..4 {
            let (a, x, y0, expected) = (a.clone(), x.clone(), y0.clone(), expected.clone());
            handles.push(std::thread::spawn(move || {
                let mut got = y0.to_vec();
                unsafe {
                    gemv_notrans_f64(m, n, 0.9, &a, lda, &x, 0.25, &mut got);
                }
                assert_eq!(got, expected);
            }));
        }
        for h in handles {
            h.join().expect("broadcast worker thread panicked");
        }
        // SAFETY: as above; restores the default-on state.
        unsafe {
            std::env::remove_var("SONIUM_GEMV_PINNED");
        }
    }

    /// Trans-MT 1x/4x byte-boundary pins: f32 flips at 8M els, f64 at
    /// 4M els (32 MiB). sq1kT stays 1x / sq4kT+ stays 4x both precisions.
    #[test]
    fn trans_tasks_1x_boundary() {
        // f32 (4 B/el): 1536²/2048²/4.2M-el rects take 1x (measured wins).
        assert!(trans_tasks_1x(1536, 1536, 4));
        assert!(trans_tasks_1x(2048, 2048, 4));
        assert!(trans_tasks_1x(1024, 4096, 4));
        assert!(!trans_tasks_1x(4096, 4096, 4));
        // f64 (8 B/el): 1536² takes 1x; 2048²/rects/4k+ take 4x.
        assert!(trans_tasks_1x(1536, 1536, 8));
        assert!(!trans_tasks_1x(2048, 2048, 8));
        assert!(!trans_tasks_1x(1024, 4096, 8));
        assert!(!trans_tasks_1x(4096, 4096, 8));
        // sq1kT stays 1x both precisions; saturates (no overflow).
        assert!(trans_tasks_1x(1024, 1024, 4));
        assert!(trans_tasks_1x(1024, 1024, 8));
        assert!(!trans_tasks_1x(usize::MAX, usize::MAX, 8));
    }

    /// Trans broadcast-vs-scope agreement: the `SONIUM_GEMV_PINNED=1`
    /// (default) arm must match the legacy raw-scope arm bitwise (same
    /// (b, c) bands, same partials layout, same ordered reduce).
    /// Shapes engage the MT path with multi-chunk (f64, m=1100 over
    /// CHUNK=1024) and multi-block (n=2000) grids.
    #[test]
    fn avx2_gemv_trans_pinned_matches_scope() {
        if !is_available() {
            return;
        }
        let _guard = env_test_lock();
        for &(m, n, lda) in &[(1100usize, 1000usize, 1100usize), (600, 2000, 600)] {
            let a64: Vec<f64> = (0..n * lda).map(|i| (i as f64 * 0.013).sin()).collect();
            let x64: Vec<f64> = (0..n).map(|i| (i as f64 * 0.017).cos()).collect();
            let y064: Vec<f64> = (0..m).map(|i| (i as f64 * 0.011).sin()).collect();
            let a32: Vec<f32> = a64.iter().map(|&v| v as f32).collect();
            let x32: Vec<f32> = x64.iter().map(|&v| v as f32).collect();
            let y032: Vec<f32> = y064.iter().map(|&v| v as f32).collect();
            // SAFETY: `env_test_lock` serializes env flips; read per call on
            // the calling thread only.
            unsafe {
                std::env::set_var("SONIUM_GEMV_PINNED", "1");
            }
            let (mut pin64, mut pin32) = (y064.clone(), y032.clone());
            unsafe {
                gemv_trans_f64(m, n, 0.9, &a64, lda, &x64, 0.25, &mut pin64);
                gemv_trans_f32(m, n, 0.9, &a32, lda, &x32, 0.25, &mut pin32);
            }
            // SAFETY: as above; selects the legacy raw-scope arm.
            unsafe {
                std::env::set_var("SONIUM_GEMV_PINNED", "0");
            }
            let (mut raw64, mut raw32) = (y064.clone(), y032.clone());
            unsafe {
                gemv_trans_f64(m, n, 0.9, &a64, lda, &x64, 0.25, &mut raw64);
                gemv_trans_f32(m, n, 0.9, &a32, lda, &x32, 0.25, &mut raw32);
            }
            assert_eq!(
                pin64, raw64,
                "trans broadcast must match scope bitwise (f64 {m}x{n})"
            );
            assert_eq!(
                pin32, raw32,
                "trans broadcast must match scope bitwise (f32 {m}x{n})"
            );
        }
        // SAFETY: as above; restores the default-on state.
        unsafe {
            std::env::remove_var("SONIUM_GEMV_PINNED");
        }
    }

    #[test]
    fn avx2_gemv_trans_propagates_nan() {
        if !is_available() {
            return;
        }
        // Large enough to engage the parallel fan-out.
        let (m, n, lda) = (2400, 2200, 2400);
        let mut a: Vec<f64> = (0..n * lda).map(|i| (i as f64 * 0.013).sin()).collect();
        a[3 * lda + 5] = f64::NAN;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.017).cos()).collect();
        let y0 = vec![1.0f64; m];
        let mut got = y0.clone();
        unsafe {
            gemv_trans_f64(m, n, 1.0, &a, lda, &x, 0.0, &mut got);
        }
        assert!(got[5].is_nan(), "NaN entry must propagate to y[5]");
        // beta == 0 overwrites y without reading it: pre-existing NaN
        // in y must not survive.
        let mut got2 = vec![f64::NAN; m];
        let a_clean: Vec<f64> = (0..n * lda).map(|i| (i as f64 * 0.013).sin()).collect();
        unsafe {
            gemv_trans_f64(m, n, 1.0, &a_clean, lda, &x, 0.0, &mut got2);
        }
        assert!(got2.iter().all(|v| !v.is_nan()));
    }
}
