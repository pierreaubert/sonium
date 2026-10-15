//! x86_64 AVX2/FMA dense GEMM kernels.
//!
//! Dependency-free replacement for the scalar triple loop that
//! `sme2::gemm::{sgemm,dgemm}` falls back to on x86_64 hosts (Linux has no
//! system-BLAS linkage; macOS x86_64 routes eligible `(NoTrans, NoTrans)`
//! calls to Accelerate before ever reaching here).
//!
//! Design — a BLIS-lite register-blocked kernel:
//!
//! - **Micro-panel** `MR × NR` = 4 × 16 (f32) / 4 × 8 (f64): 8 YMM
//!   accumulators, one broadcast of `A[i,p]` driving two FMAs per inner-k
//!   step. Column tails go through `maskload`/`maskstore`; row tails run
//!   the same panel with `mb < MR`.
//! - **Kc/Nc cache blocking** — the B band `kc × nc` (≤ 512 KB) stays
//!   L2-resident while row panels of A stream past it; A is re-read only
//!   `n / NC` times. Kc blocking accumulates partial sums into C with
//!   `beta = 1` after the first pass, mirroring the SME2 driver.
//! - **No packing** — B rows are consumed directly (`loadu`); A broadcasts
//!   handle both `NoTrans` (`A[i*lda + p]`) and `Trans` (`A[p*lda + i]`)
//!   via (row, col) stride pairs so the inner loop never branches on the
//!   transpose flag.
//!
//! Serial only: x86_64 is validated through Rosetta 2 in this workspace and
//! is not the production target; the SME2 driver's rayon machinery is not
//! replicated here.
//!
//! Rosetta caveat: `is_x86_feature_detected!` reports no AVX2 under
//! Rosetta 2 even on macOS 26, which *does* translate AVX2/FMA — the CPUID
//! bits are masked. The dispatch therefore conservatively takes the scalar
//! fallback under Rosetta; the kernels themselves were validated there by
//! forcing the gate (4-16x over scalar, parity tests pass). Real AVX2
//! x86_64 hosts detect normally.

#![cfg(target_arch = "x86_64")]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use core::arch::x86_64::*;
use std::cell::RefCell;

thread_local! {
    /// Scratch for transpose-A packing below (grows once per thread;
    /// nested-call fallback mirrors the `with_buf` pool in `super`).
    static DGEMM_A_PACK: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    /// f32 twin of [`DGEMM_A_PACK`].
    static SGEMM_A_PACK: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Below this `m × k` footprint the pack traffic exceeds the strided-A
/// microkernel loss; the direct path stands.
const DGEMM_TRANSPOSE_PACK_MIN: usize = 4096;

/// Pack strided `A^T` (stored `k × m`, leading dim `lda`) into contiguous
/// row-major `m × k` and run `f` on it. Same values the microkernel
/// would broadcast from strided storage, so results are identical.
/// Callers must have checked [`is_available`] (the vector pack needs AVX2).
unsafe fn with_packed_a_trans_f64(
    m: usize,
    k: usize,
    a: *const f64,
    lda: usize,
    f: impl FnOnce(&[f64]),
) {
    DGEMM_A_PACK.with(|cell| match cell.try_borrow_mut() {
        Ok(mut buf) => {
            let dst = super::align64_slice(&mut buf, m * k);
            // Vectorised 8×4 pack. NOTE (2026-09-27): this measures a
            // tie vs the scalar element loop it replaced (ST 1024³T
            // 1.08x both ways) — the full-matrix pack is bandwidth-bound
            // at these footprints, not instruction-bound. Kept for
            // symmetry with the f32 twin, not for speed.
            pack_trans_f64(a, lda, m, k, dst.as_mut_ptr());
            f(dst);
        }
        Err(_) => {
            let mut buf = Vec::new();
            let dst = super::align64_slice(&mut buf, m * k);
            pack_trans_f64(a, lda, m, k, dst.as_mut_ptr());
            f(dst);
        }
    });
}

/// Whether the N-split shared `A^T` pack applies (f32): default-on
/// for `A` footprints at/above 4 MiB (re-tuned 2026-09-28 after the
/// parallel pack landed — the serial-pack crossover was 16 MiB; with
/// the ~0.5 ms parallel pack the crossover drops into (1 MiB, 4 MiB]:
/// pack/gather 1.42x/1.47x/1.26x/1.28x at 4 MiB (sq1kT, 4/4),
/// 1.36-1.62x at 8-64 MiB (4/4 each), but only 1.07x/1.01x at 1 MiB
/// (sq512T) and 1.00x/0.93x at 256 KiB (sq256T); conservative at the
/// proven side). `SONIUM_GEMM_APACK=0` forces the legacy gather arm
/// for A/B, `=2` forces the pack arm (crossover experiments; the
/// tuning above predates narrow bands + align64, which shifted the
/// tradeoff toward packing). Read per call so same-process A/B can
/// interleave arms.
#[inline]
fn apack_on(m: usize, k: usize) -> bool {
    let v = std::env::var("SONIUM_GEMM_APACK").unwrap_or_default();
    if v.as_bytes().first() == Some(&b'0') {
        return false;
    }
    if v.as_bytes().first() == Some(&b'2') {
        return true;
    }
    m.saturating_mul(k) >= 1_048_576 // 4 MiB / 4 B
}

/// f32 twin of [`with_packed_a_trans_f64`]: pack strided `A^T` into
/// contiguous row-major `m × k` via the vectorised 8×8 [`pack_trans_f32`]
/// and run `f` on it. Callers must have checked [`is_available`].
unsafe fn with_packed_a_trans_f32(
    m: usize,
    k: usize,
    a: *const f32,
    lda: usize,
    f: impl FnOnce(&[f32]),
) {
    SGEMM_A_PACK.with(|cell| match cell.try_borrow_mut() {
        Ok(mut buf) => {
            let dst = super::align64_slice(&mut buf, m * k);
            pack_trans_f32(a, lda, m, k, dst.as_mut_ptr());
            f(dst);
        }
        Err(_) => {
            let mut buf = Vec::new();
            let dst = super::align64_slice(&mut buf, m * k);
            pack_trans_f32(a, lda, m, k, dst.as_mut_ptr());
            f(dst);
        }
    });
}

/// Parallel pack for the f32 MT apack arm (shipped default-on,
/// `SONIUM_GEMM_APACK_PAR=0` restores the serial pack): the serial
/// [`pack_trans_f32`] costs ~4 ms at 16 MiB while the parallel
/// [`super::pack_a_trans_f32`] costs ~0.5 ms (gemmatab PAR A/B
/// 2026-09-28, rotated order, maxdiff 0e0: par/serial 1.29x/1.25x at
/// sq2kT, 1.14x/1.09x at sq4kT; narrow-n shapes tie 0.97-1.00x —
/// they take M-split where the N-split-hooked pack never engages,
/// so immunity is by construction). Read per call for same-process A/B.
#[inline]
fn apack_par() -> bool {
    std::env::var("SONIUM_GEMM_APACK_PAR")
        .map(|v| v != "0")
        .unwrap_or(true)
}

/// MT-only twin of [`with_packed_a_trans_f32`]: lend the TLS buffer and
/// pack via the parallel [`super::pack_a_trans_f32`] when `parallel`
/// (the MT driver), else the serial kernel (legacy arm for A/B). Safe:
/// `a` is a shared borrow and the parallel pack joins before `f` runs.
fn with_packed_a_trans_f32_mt(
    m: usize,
    k: usize,
    a: &[f32],
    lda: usize,
    parallel: bool,
    f: impl FnOnce(&[f32]),
) {
    SGEMM_A_PACK.with(|cell| match cell.try_borrow_mut() {
        Ok(mut buf) => {
            let dst = super::align64_slice(&mut buf, m * k);
            if parallel {
                super::pack_a_trans_f32(a, lda, m, k, dst);
            } else {
                // SAFETY: AVX2 checked by the caller; `dst` holds `m*k`.
                unsafe {
                    pack_trans_f32(a.as_ptr(), lda, m, k, dst.as_mut_ptr());
                }
            }
            f(dst);
        }
        Err(_) => {
            let mut buf = Vec::new();
            let dst = super::align64_slice(&mut buf, m * k);
            if parallel {
                super::pack_a_trans_f32(a, lda, m, k, dst);
            } else {
                // SAFETY: as above; fallback buffer holds `m*k`.
                unsafe {
                    pack_trans_f32(a.as_ptr(), lda, m, k, dst.as_mut_ptr());
                }
            }
            f(dst);
        }
    });
}

#[inline]
pub fn is_available() -> bool {
    !crate::cpu::force_scalar_dispatch()
        && std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
}

const MR: usize = 4;

/// Vectorised transpose-pack (f64): `dst[r * cols + c] = src[c * ldb + r]`
/// for an `rows × cols` packed panel from a `cols × rows` row-major source.
///
/// The scalar element loop this replaces reads one fresh cache line per
/// element at large strides (~1–2 GB/s); 8×4 AVX2 blocks (8 contiguous
/// loads, in-register transpose, 8 contiguous stores) run an order of
/// magnitude faster. Every `NoTrans/Trans` GEMM and SYRK off-diagonal pays
/// one full pack, so this is on the hot path. Remainder rows/columns go
/// scalar; values are exact copies either way, so numerics never change.
#[target_feature(enable = "avx2")]
pub unsafe fn pack_trans_f64(src: *const f64, ldb: usize, rows: usize, cols: usize, dst: *mut f64) {
    let r8 = rows & !3;
    let c8 = cols & !7;
    let mut r = 0;
    while r < r8 {
        let mut c = 0;
        while c < c8 {
            // 8×4 block: one contiguous 4-wide load per source row
            // (`src` rows `c..c+8`, columns `r..r+4`), then the
            // in-register transpose below turns the 8 row-vectors into 4
            // packed-row vectors for contiguous stores.
            let s0 = _mm256_loadu_pd(src.add(c * ldb + r));
            let s1 = _mm256_loadu_pd(src.add((c + 1) * ldb + r));
            let s2 = _mm256_loadu_pd(src.add((c + 2) * ldb + r));
            let s3 = _mm256_loadu_pd(src.add((c + 3) * ldb + r));
            let s4 = _mm256_loadu_pd(src.add((c + 4) * ldb + r));
            let s5 = _mm256_loadu_pd(src.add((c + 5) * ldb + r));
            let s6 = _mm256_loadu_pd(src.add((c + 6) * ldb + r));
            let s7 = _mm256_loadu_pd(src.add((c + 7) * ldb + r));
            let t0 = _mm256_unpacklo_pd(s0, s1);
            let t1 = _mm256_unpackhi_pd(s0, s1);
            let t2 = _mm256_unpacklo_pd(s2, s3);
            let t3 = _mm256_unpackhi_pd(s2, s3);
            let t4 = _mm256_unpacklo_pd(s4, s5);
            let t5 = _mm256_unpackhi_pd(s4, s5);
            let t6 = _mm256_unpacklo_pd(s6, s7);
            let t7 = _mm256_unpackhi_pd(s6, s7);
            // Output rows `r..r+3` hold source columns `c..c+7` at source
            // rows `r..r+3`: packed row `r+j` is `[S(c+i, r+j)]` for `i` in
            // `0..8`; each store pair below is two halves of one such row.
            _mm256_storeu_pd(dst.add(r * cols + c), _mm256_permute2f128_pd(t0, t2, 0x20));
            _mm256_storeu_pd(
                dst.add(r * cols + c + 4),
                _mm256_permute2f128_pd(t4, t6, 0x20),
            );
            _mm256_storeu_pd(
                dst.add((r + 1) * cols + c),
                _mm256_permute2f128_pd(t1, t3, 0x20),
            );
            _mm256_storeu_pd(
                dst.add((r + 1) * cols + c + 4),
                _mm256_permute2f128_pd(t5, t7, 0x20),
            );
            _mm256_storeu_pd(
                dst.add((r + 2) * cols + c),
                _mm256_permute2f128_pd(t0, t2, 0x31),
            );
            _mm256_storeu_pd(
                dst.add((r + 2) * cols + c + 4),
                _mm256_permute2f128_pd(t4, t6, 0x31),
            );
            _mm256_storeu_pd(
                dst.add((r + 3) * cols + c),
                _mm256_permute2f128_pd(t1, t3, 0x31),
            );
            _mm256_storeu_pd(
                dst.add((r + 3) * cols + c + 4),
                _mm256_permute2f128_pd(t5, t7, 0x31),
            );
            c += 8;
        }
        // Remainder columns for these 4 rows.
        for rr in r..r + 4 {
            for cc in c8..cols {
                *dst.add(rr * cols + cc) = *src.add(cc * ldb + rr);
            }
        }
        r += 4;
    }
    // Remainder rows.
    for rr in r8..rows {
        for cc in 0..cols {
            *dst.add(rr * cols + cc) = *src.add(cc * ldb + rr);
        }
    }
}

/// Vectorised transpose-pack (f32): `dst[r * cols + c] = src[c * ldb + r]`
/// for an `rows × cols` packed panel from a `cols × rows` row-major source.
///
/// f32 twin of [`pack_trans_f64`]: 8×8 AVX2 blocks (8 contiguous loads,
/// in-register transpose, 8 contiguous stores) instead of the scalar
/// element loop, which reads one fresh cache line per element at large
/// strides. Remainder rows/columns go scalar; values are exact copies
/// either way, so numerics never change.
#[target_feature(enable = "avx2")]
pub unsafe fn pack_trans_f32(src: *const f32, ldb: usize, rows: usize, cols: usize, dst: *mut f32) {
    let r8 = rows & !7;
    let c8 = cols & !7;
    let mut r = 0;
    while r < r8 {
        let mut c = 0;
        while c < c8 {
            // 8×8 block: one contiguous 8-wide load per source row
            // (`src` rows `c..c+8`, columns `r..r+8`), then the
            // unpack/shuffle below turns the 8 row-vectors into 8
            // packed-row vectors for contiguous stores.
            let v0 = _mm256_loadu_ps(src.add(c * ldb + r));
            let v1 = _mm256_loadu_ps(src.add((c + 1) * ldb + r));
            let v2 = _mm256_loadu_ps(src.add((c + 2) * ldb + r));
            let v3 = _mm256_loadu_ps(src.add((c + 3) * ldb + r));
            let v4 = _mm256_loadu_ps(src.add((c + 4) * ldb + r));
            let v5 = _mm256_loadu_ps(src.add((c + 5) * ldb + r));
            let v6 = _mm256_loadu_ps(src.add((c + 6) * ldb + r));
            let v7 = _mm256_loadu_ps(src.add((c + 7) * ldb + r));
            let u0 = _mm256_unpacklo_ps(v0, v1);
            let u1 = _mm256_unpackhi_ps(v0, v1);
            let u2 = _mm256_unpacklo_ps(v2, v3);
            let u3 = _mm256_unpackhi_ps(v2, v3);
            let u4 = _mm256_unpacklo_ps(v4, v5);
            let u5 = _mm256_unpackhi_ps(v4, v5);
            let u6 = _mm256_unpacklo_ps(v6, v7);
            let u7 = _mm256_unpackhi_ps(v6, v7);
            // `w` pairs hold the low/high column quads of two packed rows
            // each; the permutes below concatenate the halves.
            let w0 = _mm256_shuffle_ps(u0, u2, 0x44);
            let w1 = _mm256_shuffle_ps(u4, u6, 0x44);
            let w2 = _mm256_shuffle_ps(u0, u2, 0xEE);
            let w3 = _mm256_shuffle_ps(u4, u6, 0xEE);
            let w4 = _mm256_shuffle_ps(u1, u3, 0x44);
            let w5 = _mm256_shuffle_ps(u5, u7, 0x44);
            let w6 = _mm256_shuffle_ps(u1, u3, 0xEE);
            let w7 = _mm256_shuffle_ps(u5, u7, 0xEE);
            _mm256_storeu_ps(dst.add(r * cols + c), _mm256_permute2f128_ps(w0, w1, 0x20));
            _mm256_storeu_ps(
                dst.add((r + 4) * cols + c),
                _mm256_permute2f128_ps(w0, w1, 0x31),
            );
            _mm256_storeu_ps(
                dst.add((r + 1) * cols + c),
                _mm256_permute2f128_ps(w2, w3, 0x20),
            );
            _mm256_storeu_ps(
                dst.add((r + 5) * cols + c),
                _mm256_permute2f128_ps(w2, w3, 0x31),
            );
            _mm256_storeu_ps(
                dst.add((r + 2) * cols + c),
                _mm256_permute2f128_ps(w4, w5, 0x20),
            );
            _mm256_storeu_ps(
                dst.add((r + 6) * cols + c),
                _mm256_permute2f128_ps(w4, w5, 0x31),
            );
            _mm256_storeu_ps(
                dst.add((r + 3) * cols + c),
                _mm256_permute2f128_ps(w6, w7, 0x20),
            );
            _mm256_storeu_ps(
                dst.add((r + 7) * cols + c),
                _mm256_permute2f128_ps(w6, w7, 0x31),
            );
            c += 8;
        }
        // Remainder columns for these 8 rows.
        for rr in r..r + 8 {
            for cc in c8..cols {
                *dst.add(rr * cols + cc) = *src.add(cc * ldb + rr);
            }
        }
        r += 8;
    }
    // Remainder rows.
    for rr in r8..rows {
        for cc in 0..cols {
            *dst.add(rr * cols + cc) = *src.add(cc * ldb + rr);
        }
    }
}

/// Pointer wrappers that are `Send + Sync` for `rayon::scope` worker
/// dispatch. Each task writes a disjoint C row band; A/B are read-only.
#[derive(Clone, Copy)]
struct SendConstPtr<T>(*const T);
unsafe impl<T> Send for SendConstPtr<T> {}
unsafe impl<T> Sync for SendConstPtr<T> {}
#[derive(Clone, Copy)]
struct SendMutPtr<T>(*mut T);
unsafe impl<T> Send for SendMutPtr<T> {}
unsafe impl<T> Sync for SendMutPtr<T> {}

/// Minimum total multiply-accumulates (`m * n * k`) before paying
/// `rayon::scope` spawn/join overhead. Below this the serial sweep wins
/// (e.g. 64^3 = 262k MACs stays serial; 256^3 = 16.7M goes parallel).
const PAR_MIN_MACS: usize = 4_000_000;

/// Single-pass floor: shapes with `k <= kc` run in one Kc pass, so a
/// parallel call pays exactly one scope-join round and one `B` pack per
/// task — no per-pass barrier or re-pack amplification. Measurement basis:
/// OpenBLAS parallelises thin-`k` updates (62x512x32: 2.1x over its own
/// serial; 124x992x32: 2.6x) while our 4M floor left them serial at
/// 0.34-0.53x. The serial kernel itself is at parity (0.84-1.11x
/// single-threaded), so the gap is the missing parallelism. Floor tried
/// at 1M first: 124x992x32 won 2.2x, but 62x512x32 (1.0M) lost 16% and
/// 128^3 (2.1M) flipped bimodally between a 1.9x win and a loss — at
/// 1-2M MACs the 1M-per-tile floor leaves 1-2 tasks on 8 workers, so
/// spawn/wakeup placement luck dominates. 3M keeps the consistent wins
/// (3.9M+) and leaves the overhead-dominated regime serial.
const PAR_MIN_MACS_SINGLE_PASS: usize = 3_000_000;

/// Parallel-entry floor for a shape: the single-pass floor when the whole
/// `k` fits one Kc panel, the full floor otherwise.
#[inline]
fn par_min_macs(k: usize, kc: usize) -> usize {
    if k <= kc {
        PAR_MIN_MACS_SINGLE_PASS
    } else {
        PAR_MIN_MACS
    }
}

/// Decide whether the M-row sweep parallelises. Partitioning over disjoint
/// C row bands keeps per-element accumulation order identical to the serial
/// sweep, so results are bitwise identical either way.
///
/// The task decomposition re-reads the shared operand per task (B panel per
/// row task, A panel per column task): across cores that traffic is parallel
/// L3, but on a single thread it is pure serial waste (measured 2 GB vs
/// 8 MB of B traffic at 1024^3) — so one thread always takes the serial
/// nest, whose per-tile streaming reads each operand once.
///
/// Tall-thin-k exception (2026-10-01): `m >= 4n`, `k <= 64` shapes
/// parallelise from 2M MACs instead of the 3M single-pass floor. The
/// 3M floor exists because 1-2M-MAC squares flip bimodally (see
/// [`PAR_MIN_MACS_SINGLE_PASS`]), but tall thin-k calls are nowhere
/// near that regime: their serial nest runs at ~25 GFLOPS on strided
/// narrow bands (1000x48x48: 0.181ms serial) while the M-split does
/// ~0.10ms — the serial/strided disease, not dispatch luck. M-split
/// only (this gate); N-split/2D keep the portfolio floor.
/// `SONIUM_GEMM_TALLFLOOR=0` disables it (diagnostic/A-B fallback).
#[inline]
fn use_parallel(m: usize, n: usize, k: usize, kc: usize) -> bool {
    if rayon::current_num_threads() <= 1 || m < MR * 2 {
        return false;
    }
    let macs = m.saturating_mul(n).saturating_mul(k);
    if macs >= par_min_macs(k, kc) {
        return true;
    }
    // Tall-thin-k M-split floor (see above): 2M MACs, `m >= 4n` aspect,
    // `k <= 64` thin (single Kc pass guaranteed: `kc >= 256` for both
    // precisions, so no multi-pass dispatch multiplies the small call).
    // Division form of the aspect test (overflow-proof for huge `n`).
    if std::env::var("SONIUM_GEMM_TALLFLOOR")
        .map(|v| v == "0")
        .unwrap_or(false)
    {
        return false;
    }
    k > 0 && k <= 64 && m / 4 >= n && macs >= 2_000_000
}

/// Target multiply-accumulates per rayon task. A 250K-MAC task is ~10-25µs
/// of AVX2/FMA work per core: small enough to yield 2+ waves of tasks on
/// 64-128 threads for thin shapes (measured 2.1x faster than 4M-MAC tasks
/// at 32 threads on `96x1024x512`), large enough that scope spawn/wakeup
/// (~µs) and the per-task B-tile pack stay negligible. Sizing by work
/// rather than by row count keeps thin-K updates (e.g. LU Schur complements
/// with k = block width) from drowning in 3µs tasks.
const TASK_TARGET_MACS: usize = 250000;

/// Rows per rayon task: enough rows that the task covers roughly
/// [`TASK_TARGET_MACS`] over its `n × k` tile, symmetric to
/// [`cols_per_task`] — a four-micro-panel floor (whole-panel multiples,
/// pool-filling capped) cures the same degenerate-narrow disease on the
/// row axis (`1984x32x1984` runs 0.16x at one panel per task).
/// (A sub-[`MIN_TILE_MACS`] coarsening was trialled and reverted:
/// `896x64x128` runs identical times at 7 tasks and 32 tasks, so that
/// shape is sweep-rate-bound at narrow `n`, not dispatch-bound — the
/// row count is not the lever there.)
#[inline]
fn rows_per_task(m: usize, n: usize, k: usize) -> usize {
    let per_row = n.saturating_mul(k).max(1);
    let by_work = TASK_TARGET_MACS.div_ceil(per_row);
    let macs = m.saturating_mul(n).saturating_mul(k);
    let workers = super::workers_for_macs(macs);
    let round_down = |w: usize| (w / MR * MR).max(MR);
    let wide = (4 * MR).min(round_down(m.div_ceil(workers).max(MR)));
    round_down(by_work.max(wide)).min(m.max(1))
}

/// Decide whether the N-column sweep parallelises. Taken when the output is
/// at least a quarter as wide as tall (`4n > m`): each task streaming its
/// own packed `B` tile reads `B` once total from an L2-resident buffer,
/// where the M-split would re-read the full `B` band from L3 once per row
/// band (the thin-`m` collapse: 14–26x behind OpenBLAS at `m = 64`; and
/// 2 GB of re-reads even at square 1024^3). Disjoint C column bands keep
/// per-element accumulation order identical to the serial nest, so results
/// are bitwise identical either way. Also covers `m < MR * 2`, which the
/// M-split leaves serial.
///
/// NOTE: an earlier revision additionally gated on `B` L3-residency (M-split
/// below 64 MB), theorising contiguous re-reads beat strided single reads;
/// A/B measurement showed the opposite (N 2.2x faster at `64×2048×2048`),
/// so the gate was removed — traffic volume dominates access pattern here
/// because the packed `B` is hot across L3 slices.
/// `SONIUM_GEMM_M_SPLIT` escape hatch: when set, the parallel GEMM entries
/// skip the 2D and N-split paths and take the M-split (row bands). The
/// M-split packs one `B` panel shared read-only by all row tasks, while
/// the N-split re-reads the full `A` panel once per column band (8 MB of
/// re-reads on a 3 MB 512³ problem); under memory-bandwidth contention
/// the N-split's re-reads collapse small-square throughput (512³ f32:
/// 263 GF/s loaded vs 445 quiet) while the M-split holds. Interleaved
/// square-M-vs-N A/B (8 threads, f32): M wins 512³-2048³ by 10-40% every round;
/// the one N win (1024³, power-of-2 panel stride) was 4 KiB set aliasing,
/// fixed by the padded panel stride (`panel_stride`): padded M wins
/// f32-1024³ by 29% and f64-512³ by 30%; f64-2048³ M wins 19% too.
/// Residual: only f64-1024³ still N +8% (reason open). Wide-pool A/B vetoes a default
/// flip: M wins pinned-8 (+20%) but loses unpinned-8/16/32/128 (-11% to
/// -52%) — the shared panel needs cache affinity (one L3 domain), which
/// no thread count can promise; the N-split's private tiles are
/// migration-proof. So the gate stays default-off for pinned/narrow
/// deployments. Read per call, same as the `SONIUM_TRSM_FUSED` gates.
#[inline]
fn force_m_split() -> bool {
    std::env::var("SONIUM_GEMM_M_SPLIT").is_ok()
}

/// Diagnostic probe gate: `SONIUM_MTALL_RPT=n` overrides the M-split row
/// width (rounded down to `MR`). Read per pass for same-process A/B.
#[inline]
fn mtall_rpt_override() -> Option<usize> {
    std::env::var("SONIUM_MTALL_RPT")
        .ok()?
        .parse::<usize>()
        .ok()
        .filter(|v| *v > 0)
}

/// Diagnostic probe gate: `SONIUM_MTALL_PRIVB=0/1` forces the M-split
/// shared/private `B` panel. Read per pass for same-process A/B.
#[inline]
fn mtall_privb_override() -> Option<bool> {
    match std::env::var("SONIUM_MTALL_PRIVB").ok()?.as_str() {
        "0" => Some(false),
        "1" => Some(true),
        _ => None,
    }
}

/// Diagnostic probe gate: `SONIUM_MTALL_KC=n` overrides the M-split pass
/// width. Read per pass for same-process A/B.
#[inline]
fn mtall_kc_override() -> Option<usize> {
    std::env::var("SONIUM_MTALL_KC")
        .ok()?
        .parse::<usize>()
        .ok()
        .filter(|v| *v > 0)
}

/// Fused-pass M-split gate (shipped default-on, `SONIUM_GEMM_MFUSE=0`
/// restores the pass-per-scope loop): one scope whose row-band tasks loop
/// over all Kc passes instead of one scope-join per pass. Same packs, same
/// per-element sweep order — bitwise identical to the legacy loop. Tall
/// shapes pay a pool install + spawn round + join barrier per pass (8
/// passes at 2048x128x2048); fusing removes 7 of each with identical
/// traffic. Requires a uniform privb pass grid (see [`m_fuse_kc`); tails
/// and shared-panel shapes keep the legacy loop). Read once per call.
#[inline]
fn m_fuse_on() -> bool {
    std::env::var("SONIUM_GEMM_MFUSE")
        .map(|v| v != "0")
        .unwrap_or(true)
}

/// Fused-pass M-split pass width (pure over injected panel/budget sizes so
/// the dispatch is unit-testable): the uniform privb `kc` when every pass
/// takes the private panel at the same width, else `None` (legacy loop).
/// Mirrors the per-pass [`privb_kc_cap`] fit check with `elsize`-scaled
/// arithmetic, plus the no-tail requirement (`k % kc == 0`: a short tail
/// pass would pack/sweep at a different width). `use_pack` needs no check:
/// a privb-fitting panel (`kc * panel_n * elsize <= budget <= 256 KiB`) is
/// far below [`PANEL_MAX_ELS`]. Callers additionally gate on NoTrans-B
/// (`want_privb`) and [`m_fuse_on`].
fn m_fuse_kc(k: usize, kc0: usize, panel_n: usize, elsize: usize, budget: usize) -> Option<usize> {
    if k == 0 || kc0 == 0 || elsize == 0 {
        return None;
    }
    let kc_try = kc0.min((budget / elsize / panel_n.max(1)).max(128));
    let fit = kc_try >= 128 && kc_try.saturating_mul(panel_n).saturating_mul(elsize) <= budget;
    if !fit {
        return None;
    }
    if !k.is_multiple_of(kc_try) {
        return None;
    }
    Some(kc_try)
}

/// 2D-veto bypass for thin `k` (measured 2026-09-27, shipped
/// default-off diagnostic): forcing the 2D grid past the `ct*2 > bands_n`
/// traffic veto wins uniformly at `k = 64` (0.7-0.8x -> 0.9-1.1x vs
/// OpenBLAS across all 14 trailing widths) but is neutral at `k = 128`
/// and untested on square/large-`k` shapes the veto protects — so it
/// stays a diagnostic gate, not a default. A conditioned default
/// (bypass when `k <= 64`, B tiles L1-resident) needs portfolio-wide A/B
/// first. LU (`k = 128` Schur) unaffected either way: full dgetrf-2048
/// ties at 40.0/40.0ms.
#[inline]
fn force_2d() -> bool {
    std::env::var_os("SONIUM_GEMM_FORCE_2D").is_some()
}

/// Pack-once-shared `A^T` for the multi-threaded trans-A path
/// (shipped default-on, `SONIUM_GEMM_PACK_A_MT=0` restores the gather
/// path): pack `A^T` once into a shared buffer and recurse down the
/// `NoTrans` path instead of letting every band task gather `A` strided
/// (one cache line per element, re-read per band). The packed tiles feed
/// the sweep the same elements in the same order, so results match the
/// gather path bit-for-bit (locked by
/// `avx2_trans_a_mt_pack_matches_gather_bitwise`). Read per call, same
/// as the other GEMM gates.
#[inline]
fn pack_a_mt() -> bool {
    std::env::var("SONIUM_GEMM_PACK_A_MT")
        .map(|v| v != "0")
        .unwrap_or(true)
}

/// TLS pack buffer for the MT pack arm (shipped default-on,
/// `SONIUM_GEMM_PACK_A_MT_TLS=0` restores the fresh-`vec!`): the fresh
/// `vec!` faults ~2000 pages per call at sq1k (faults scale with `m*k`)
/// while the TLS buffer is pre-faulted after warmup (gemmatab64 A/B
/// 2026-09-28, rotated order, maxdiff 0e0: TLS/fresh 1.23x/1.34x at
/// sq1kT, 1.20x/1.23x at sq2kT, 1.13x/1.18x at n2k-m1k; gate-excluded
/// sq512T control ties 1.00x/0.98x). Read per call for same-process A/B.
#[inline]
fn pack_a_mt_tls() -> bool {
    std::env::var("SONIUM_GEMM_PACK_A_MT_TLS")
        .map(|v| v != "0")
        .unwrap_or(true)
}

/// Lower work gate (MACs) for [`pack_a_mt`]: below this too few bands
/// re-read `A` for the shared pack to pay off (A/B 2026-09-27).
const PACK_A_MT_MIN_MACS: usize = 100_000_000;

/// Upper size gate (elements) for [`pack_a_mt`]: above this the shared
/// pack buffer itself becomes an L3 liability.
const PACK_A_MT_MAX_ELS: usize = 8_000_000;

/// Row floor for [`pack_a_mt`]: below this the per-band gather hits
/// L3-resident lines ~free while the shared pack pays a full extra DRAM
/// pass, so gather wins (tapackm interleaved A/B 2026-09-27, n=k=2048:
/// gather/pack 0.92x at m=32, 0.77x at m=64, 0.76x at m=128, 0.87x at
/// m=256, then pack wins 1.16x at m=512; k=1984 control identical).
/// Conservative (crossover lies in (256, 512]) — pack only where proven.
const PACK_A_MT_MIN_M: usize = 512;

/// Column floor for [`pack_a_mt`]: the pack cost amortizes over the
/// N-split bands re-reading `A`, so narrow-`n` shapes lose. Re-tuned
/// 2026-09-28 after wide-Kc shipped (the kc=256 crossover predates it):
/// force-pack vs gather A/B, both orders, OFF brackets (dsolo solos):
/// n=512 wins 0.79x (1024x512x1024: 3.14 vs 3.97ms), n=768 wins
/// 0.72-0.86x (1024x768x1024: 4.50 vs 6.29ms; 768^3: 2.59-2.75 vs
/// 3.03-3.34ms), n=1024/2048 win as before; n=256 still loses
/// (prior 0.76-0.81x, unretested — floor stays above it). Crossover
/// now in (256, 512] — conservative at the proven side. CAUTION
/// (prior tuning): the first 4 runs without order rotation showed
/// n=512 winning 3/4 via warmup bias; the unbiased reruns reversed
/// it — the floor below is re-proven, not inherited.
const PACK_A_MT_MIN_N: usize = 512;

/// `A`-footprint floor (elements) for [`pack_a_mt`]: at small `A` the
/// gather hits L3-resident lines ~free while the pack pays a full
/// extra pass (512^3T: force-pack ties 1.00-1.02x, 1.037 vs
/// 1.038ms). 4 MiB mirrors the f32 `apack_on` crossover; 768^3T
/// (4.5 MiB) wins 0.86x above it.
const PACK_A_MT_MIN_A_ELS: usize = 524_288;

/// Shape clauses of the MT pack gate: enough rows ([`PACK_A_MT_MIN_M`])
/// and columns ([`PACK_A_MT_MIN_N`]), a pack buffer in
/// [`PACK_A_MT_MIN_A_ELS`]`..=`[`PACK_A_MT_MAX_ELS`], and work over
/// [`PACK_A_MT_MIN_MACS`]. Pure so the dispatch is unit-testable without
/// running a solve.
fn pack_a_mt_shape(m: usize, n: usize, k: usize) -> bool {
    m >= PACK_A_MT_MIN_M
        && n >= PACK_A_MT_MIN_N
        && m * k >= PACK_A_MT_MIN_A_ELS
        && m * k <= PACK_A_MT_MAX_ELS
        && m.saturating_mul(n).saturating_mul(k) >= PACK_A_MT_MIN_MACS
}

/// Rank-`k` M-split rule: single-`Kc`-pass updates with a small shared
/// `B` panel take the row-band path instead of 2D/N. Measured 1.35-1.5x
/// on `992²×32` / `1920²×32` NT/NT (137→186, 145→215 GF/s): the 2D/N
/// decompositions spill the beta=1 `C` tiles while row bands keep one
/// shared `B` pack and L2-resident `C` tiles. Narrow by construction
/// (`k <= 128`, panel `<= 4 MB` f64 — see [`RANKK_MAX_PANEL_ELS`]): the
/// wide-pool veto on the M-split default (64 MB+ shared panels losing
/// cache affinity unpinned) cannot apply — a migrated task re-fetches a
/// few megabytes at most, and the ceiling sits an order of magnitude
/// below the veto. `n >= 128` keeps
/// degenerate-narrow shapes on their tuned paths. Row-count gate in two
/// clauses (interleaved M-vs-N medians, two runs each, f32 and f64):
/// `m >= 640` is the measured crossover at `n = 992, k = 32` in f64
/// (m = 512 ties at +4/-4%, m = 640 wins +11/+10%, m >= 704 wins
/// +7..+21%, m <= 448 ties-or-loses; f32 ties at m = 640-992 within
/// ±2%, so the arm is harmless but unproven there and retained for
/// f64); `m >= 512 && n >= 512` covers the
/// `k = 128` TRSM-update regime (n = 1024: m >= 512 wins +9..+29% f32,
/// +7..+20% f64, m = 384 ties-or-loses; n = 512: m >= 512 wins
/// +13/+14% f32, +4/+8% f64; n = 256: m = 512 loses -4..-17%, m = 640
/// mixed, m = 896 wins — hence the `n >= 512` cut). Below the crossover
/// the default parallel path is faster; above it one shared `B` pack
/// amortizes the pack cost.
/// `SONIUM_GEMM_NO_RANKK_M` disables it (diagnostic/A-B fallback, read
/// per call like the other gates).
///
/// Shared-panel ceiling: the `k × n` panel holds at most this many
/// elements. Placed at 512K (4 MB f64, 2 MB f32) from the
/// dgetrf-4096 Schur cliff: the old 128K ceiling (1 MB) cut the
/// biggest trailing panels (`T > 2730` at `k = 48`) back to the N-split,
/// whose per-band `A` re-reads run them at 250-300 GF/s vs 400+ on the
/// shared-pack row bands; raising the ceiling recovers the fast path
/// (Schur-4096 sequence 162 → 119 ms, 1.37x, mirror-consistent) with no
/// upper cliff through 8 MB and no regression on the newly admitted
/// thin-`k` shapes (`4096²×64` +30%, `2048²×128` +7%, `4096²×128` ties,
/// all still ahead of OpenBLAS). The pack stays L3-resident and shared
/// read-only, so raising further only risks the unpinned wide-pool
/// migration the M-split veto guards — re-measure there before lifting.
const RANKK_MAX_PANEL_ELS: usize = 524_288;

/// Diagnostic tuning gate: `SONIUM_RANKK_CAP=n` overrides
/// [`RANKK_MAX_PANEL_ELS`] (parsed once per routing call).
#[inline]
fn rankk_max_panel_els() -> usize {
    std::env::var("SONIUM_RANKK_CAP")
        .ok()
        .and_then(|text| text.parse::<usize>().ok())
        .filter(|cap| *cap > 0)
        .unwrap_or(RANKK_MAX_PANEL_ELS)
}

#[inline]
fn use_m_split_rank_k(m: usize, n: usize, k: usize, kc: usize) -> bool {
    if std::env::var_os("SONIUM_GEMM_NO_RANKK_M").is_some() {
        return false;
    }
    rayon::current_num_threads() > 1
        && k > 0
        && k <= kc.min(128)
        && k.saturating_mul(n) <= rankk_max_panel_els()
        && (m >= 640 || (m >= 512 && n >= 512))
        && n >= 128
        && m.saturating_mul(n).saturating_mul(k) >= PAR_MIN_MACS
}

fn use_parallel_n(m: usize, n: usize, k: usize, kc: usize) -> bool {
    // Taken when the output is at least a quarter as wide as tall. The
    // original `n > m` gate assumed row bands share `B` cheaply, but the
    // M-split re-reads the full `B` band per row task from L3 (2 GB of
    // re-reads at 1024^3, where the serial nest reads 8 MB): N-split tasks
    // instead stream L2-resident packed `B` tiles (256 KB) and win even on
    // square shapes (1024^3 at 32 threads: 312 → 596 GF/s, past OpenBLAS at
    // 574). Genuinely tall outputs (`m >= 4n`) stay on the M-split, where
    // per-task `A` re-reads would dominate instead; tall shapes measure
    // 0.87–1.11x there. Single thread takes the serial nest (see
    // `use_parallel`).
    rayon::current_num_threads() > 1
        && 4 * n > m
        && n >= 32
        && m.saturating_mul(n).saturating_mul(k) >= par_min_macs(k, kc)
}

/// Columns per rayon task for the N-split path, sized by work like
/// [`rows_per_task`]. The floor is four micro-panels (`4 * nr`), capped by
/// the pool-filling width: narrower bands need no alignment (the sweep
/// masks narrow `nb`), but sub-`4 * nr` sweeps waste the per-task
/// tile-pack setup and re-read `A` once per task, while an uncapped floor
/// would starve a wide pool (64 tasks on 128 threads halved large-GEMM
/// throughput).
///
/// NOTE: this returns the geometry width only. The fixed-cost band cap
/// ([`cap_bands_by_work`]) applies at the N-split drivers, not here: the
/// 2D gate's `bands_n` veto must keep comparing against geometry (capping
/// here flipped thin-k shapes off the 2D route by shrinking `bands_n`).
/// `workers` is the executing scope's width (usually
/// [`super::workers_for_macs`]; the CCD-compact route passes the compact
/// width) so sizer and router agree.
#[inline]
fn cols_per_task(m: usize, n: usize, k: usize, nr: usize, workers: usize) -> usize {
    let per_col = m.saturating_mul(k).max(1);
    let by_work = TASK_TARGET_MACS.div_ceil(per_col);
    // Degenerate-narrow floor: sweeps much narrower than four micro-panels
    // waste the per-task tile-pack setup and re-read `A` once per task
    // (`64x2048x2048` runs 0.35x at 8 wide, 0.69x at 32 wide). Capped so
    // the floor never starves the engaged pool: `n / workers` is the
    // widest task width that still keeps every worker busy. The pool here
    // is the same one `scope_for_macs` will pick (see
    // `workers_for_macs`; full-`k` MACs below a ceiling imply per-pass
    // MACs below it too), so a capped-pool route still sees its full pool
    // and a global route still sees a full global pool. Both the floor
    // and the final width round down to whole
    // micro-panels: a masked remainder tail costs more than its columns
    // (`512^3` drops 1.17x -> 0.69x going 8 wide to 8+3 wide).
    let workers = workers.max(1);
    let round_down = |w: usize| (w / nr * nr).max(nr);
    let wide = (4 * nr).min(round_down(n.div_ceil(workers).max(nr)));
    // Small-pool floor: `wide` caps at four micro-panels, leaving big
    // shapes with dozens of narrow sweeps per pass on a small pool
    // (2048^3: 64 tasks of 32-wide sweeps on 8 workers, while M-split
    // runs 0.66x-vs-0.51x there). Floor the width at two waves' worth of
    // columns so sweeps widen as the pool shrinks; wide pools are
    // unaffected (`n / 2w` rounds below `4 * nr` there), and thin axes
    // still floor at micro-panel granularity via `round_down`.
    let pool_floor = round_down(n.div_ceil(2 * workers.max(1)).max(nr));
    round_down(by_work.max(wide).max(pool_floor)).min(n.max(1))
}

/// f32 N-split narrow-band task floor: bands may narrow until tasks
/// reach this many MACs (~8 MFLOP, ~80µs — spawn + pack setup stays a
/// few percent). The legacy [`F32_MIN_TASK_MAC`] (150M) forbade the
/// L1-resident bands below and cost sq1k ~13%.
const F32_NARROW_TASK_MAC: usize = 4_000_000;
/// f32 N-split narrow-band wave cap: at most this many full pool waves
/// (serial `spawn` cost grows per task; 8 waves measured fine at
/// m512n2k, tighter caps untested — do not raise without shape data).
const F32_NARROW_MAX_WAVES: usize = 8;

/// f32 N-split narrow-band count (pure over an injected L3 size so the
/// dispatch is unit-testable): the narrowest bands (`nc = NR`, B tile
/// and multi-pass C revisit L1-resident) subject to A-fit (`bands × A
/// ≤ L3` — past L3 the re-reads hit DRAM and lose, sq2k 32-band
/// 1.10-1.16x), full pool waves (partial waves strand workers:
/// sq768 24-band 1.10x vs 16/48), the spawn wave cap, and the task
/// floor. Pool-fill wins over A-fit when they conflict (fewer bands
/// than workers idles cores, worse than DRAM). Returns a band count
/// `>= 1`, `<= geometry`.
fn f32_narrow_bands(m: usize, n: usize, k: usize, workers: usize, l3: usize) -> usize {
    let w = workers.max(1);
    let geom = n.div_ceil(F32_NR).max(1);
    let a_bytes = m.saturating_mul(k).saturating_mul(4).max(1);
    let fit = (l3 / a_bytes).max(1);
    let macs = m.saturating_mul(n).saturating_mul(k);
    let floor = (macs / F32_NARROW_TASK_MAC.max(1)).max(1);
    let cap = F32_NARROW_MAX_WAVES * w;
    let target = geom.min(fit).min(cap).min(floor);
    // Snap to full waves, preferring up (narrower) when fit, cap, and
    // floor all allow; else down (down may breach fit/floor only via
    // the pool-fill clamp below — intended).
    let up = target.div_ceil(w) * w;
    let down = (target / w).max(1) * w;
    let bands = if up <= fit && up <= cap && up <= floor {
        up
    } else {
        down
    };
    bands.min(geom).max(w.min(geom)).max(1)
}

/// Skip predicate for the narrow-band rule (pure, unit-tested): on a
/// single pass (`k <= kc_step`) with the legacy B tile already
/// L1-resident there is no B-gain and no C-revisit gain, only extra
/// per-task setup (k128 narrow 1.036x bracketed, both arms B-L1).
/// `legacy_nc` is the legacy chain's band width; the B footprint uses
/// the actual pass depth `min(k, kc_step)`, not the allocation.
fn f32_narrow_skip_1pass_l1(k: usize, kc_step: usize, legacy_nc: usize) -> bool {
    const L1_BYTES: usize = 32 * 1024;
    k <= kc_step && k.saturating_mul(legacy_nc).saturating_mul(4) <= L1_BYTES
}

/// f32 N-split narrow-band width (shipped default-on,
/// `SONIUM_GEMM_NARROW=0` restores the legacy chain): bands A/B
/// 2026-09-28 (OB-bracketed rounds, 16T, bitwise ndiff=0 everywhere):
/// sq1kT 0.76x, sq1kNT 0.82x, sq768 0.84-0.86x, r1k512N 0.65x,
/// m512n2k 0.72x, k256 0.78x, sq512NT 0.86x, r1k512T 0.91x;
/// sq2k/sq512T/thin-m tie (1.00-1.02x, within the ±2% meter).
/// Gated to fat shapes (`m >= 512`, like `l3_fit_cpt`); thin-m keeps
/// legacy. `None` when gated out, killed, single-pass-L1 (see
/// [`f32_narrow_skip_1pass_l1`]), or L3 unreadable (all keep legacy).
#[inline]
fn f32_narrow_cpt(
    m: usize,
    n: usize,
    k: usize,
    workers: usize,
    legacy_cpt: usize,
) -> Option<usize> {
    if m < 512 {
        return None;
    }
    if std::env::var("SONIUM_GEMM_NARROW")
        .map(|v| v == "0")
        .unwrap_or(false)
    {
        return None;
    }
    if f32_narrow_skip_1pass_l1(k, f32_kc_nsplit(k), legacy_cpt) {
        return None;
    }
    let l3 = sonium_topology::affinity::l3_total_bytes()?;
    let bands = f32_narrow_bands(m, n, k, workers, l3);
    Some((n.div_ceil(bands).div_ceil(F32_NR) * F32_NR).max(F32_NR))
}

/// f64 N-split narrow-band task floor: mirrors [`F32_NARROW_TASK_MAC`]
/// (same MAC count; the f64 kernel runs ~half rate so tasks run ~2x
/// longer — spawn + pack setup stays a few percent either way).
const F64_NARROW_TASK_MAC: usize = 4_000_000;
/// f64 N-split narrow-band wave cap: mirrors [`F32_NARROW_MAX_WAVES`].
const F64_NARROW_MAX_WAVES: usize = 8;
/// f64 N-split narrow-band A-fit budget, in units of total L3: the L1
/// recipe (32 KB B tile) needs 2x the bands of f32 at 2x the A bytes
/// (4x nominal traffic: 128 x 8 MB = 1 GB at sq1k), and measured sq1k
/// wins 0.86x there (sharing: the 8 MB A fits each CCX L3, so band
/// re-reads are L3 hits, not the nominal DRAM). 8x keeps sq1k (1 GB)
/// with margin (the A cap in [`f64_narrow_cpt`] vetoes sq2k outright:
/// 8 GB nominal at 256 bands loses 1.44x, and even 64 bands lose
/// 1.16x there).
const F64_NARROW_FIT_L3_MULT: usize = 8;

/// f64 N-split narrow-band count (pure over an injected L3 size so the
/// dispatch is unit-testable): mirrors [`f32_narrow_bands`] with f64
/// floors (`NR = 8`, 8-byte elements) and the [`F64_NARROW_FIT_L3_MULT`]
/// fit budget. Returns a band count `>= 1`, `<= geometry`.
fn f64_narrow_bands(m: usize, n: usize, k: usize, workers: usize, l3: usize) -> usize {
    let w = workers.max(1);
    let geom = n.div_ceil(F64_NR).max(1);
    let a_bytes = m.saturating_mul(k).saturating_mul(8).max(1);
    let fit = (l3.saturating_mul(F64_NARROW_FIT_L3_MULT) / a_bytes).max(1);
    let macs = m.saturating_mul(n).saturating_mul(k);
    let floor = (macs / F64_NARROW_TASK_MAC.max(1)).max(1);
    let cap = F64_NARROW_MAX_WAVES * w;
    let target = geom.min(fit).min(cap).min(floor);
    // Snap to full waves, preferring up (narrower) when fit, cap, and
    // floor all allow; else down (down may breach fit/floor only via
    // the pool-fill clamp below — intended).
    let up = target.div_ceil(w) * w;
    let down = (target / w).max(1) * w;
    let bands = if up <= fit && up <= cap && up <= floor {
        up
    } else {
        down
    };
    bands.min(geom).max(w.min(geom)).max(1)
}

/// Skip predicate for the f64 narrow-band rule (pure, unit-tested):
/// mirrors [`f32_narrow_skip_1pass_l1`] with 8-byte elements.
fn f64_narrow_skip_1pass_l1(k: usize, kc_step: usize, legacy_nc: usize) -> bool {
    const L1_BYTES: usize = 32 * 1024;
    k <= kc_step && k.saturating_mul(legacy_nc).saturating_mul(8) <= L1_BYTES
}

/// f64 N-split narrow-band A-footprint cap: narrow bands re-read A
/// once per band, and past one CCX L3 slice the re-reads stop being
/// L3 hits (sq2k: A = 32 MB, 64 bands lose 1.16x interleaved x3;
/// sq1k: A = 8 MB, 128 bands win 0.86x). 16 MB keeps the win domain
/// (8 MB and below) with margin while the fit veto sizes the middle.
const F64_NARROW_MAX_A_BYTES: usize = 16 * 1024 * 1024;

/// f64 N-split narrow-band width (shipped default-on,
/// `SONIUM_GEMM_DNARROW=0` restores the legacy chain): f64 twin of
/// [`f32_narrow_cpt`]. Band-width A/B 2026-10-06 (OB-bracketed,
/// interleaved x3, 16T): sq1k cpt8 0.86x vs legacy cpt32 (4.56 vs 5.33
/// ms; gap 1.20x -> 1.05x), r512n2k 0.92x, r256n1k ~0.9x; sq2k keeps
/// legacy via the A cap (64 bands lose 1.16x), thin-m via the row
/// gate. Gated to `m >= 256` (256-wide rows win measured; thin-m
/// loses 1.23x at cpt8 — spawn/task economics) and `A <= 16 MB` (see
/// [`F64_NARROW_MAX_A_BYTES`]); the 256 guard keeps legacy via the
/// floor + single-pass-L1 skip (both provoke `None` there). `None`
/// when gated out, killed, single-pass-L1, or L3 unreadable (all keep
/// legacy).
#[inline]
fn f64_narrow_cpt(
    m: usize,
    n: usize,
    k: usize,
    workers: usize,
    legacy_cpt: usize,
) -> Option<usize> {
    if m < 256 {
        return None;
    }
    if m.saturating_mul(k).saturating_mul(8) > F64_NARROW_MAX_A_BYTES {
        return None;
    }
    if std::env::var("SONIUM_GEMM_DNARROW")
        .map(|v| v == "0")
        .unwrap_or(false)
    {
        return None;
    }
    if f64_narrow_skip_1pass_l1(k, f64_kc_nsplit(n, k), legacy_cpt) {
        return None;
    }
    let l3 = sonium_topology::affinity::l3_total_bytes()?;
    let bands = f64_narrow_bands(m, n, k, workers, l3);
    Some((n.div_ceil(bands).div_ceil(F64_NR) * F64_NR).max(F64_NR))
}

/// L3-fit N-split band width (shipped 2026-09-27). The N-split re-reads
/// `A` once per band: past total L3 the re-reads self-evict, while the
/// most bands that fit win. Rule: most bands with bands × A ≤ L3, at
/// least pool-fill, at most geometry (16-element floor) and min-task
/// economics. Immune A/B (rotated order + swapped addresses + balanced
/// predecessors, 12 reps, quiet box): sq2048 32→16 bands wins f32
/// 1.10-1.14x / f64 1.05-1.08x (4/4 runs); sq1024/sq512/thin shapes
/// unchanged (rule converges to legacy or gates out); f64-bemskinny
/// 16→8 ties. Gated to fat-mid-K (`m >= 512`, `k >= 256`):
/// thin-m/narrow-K keep the legacy floors. `None` when gated out or L3
/// unreadable (both keep legacy sizing).
#[inline]
fn l3_fit_cpt(
    m: usize,
    n: usize,
    k: usize,
    elsize: usize,
    nr: usize,
    workers: usize,
    min_task_mac: usize,
) -> Option<usize> {
    if m < 512 || k < 256 {
        return None;
    }
    let l3 = sonium_topology::affinity::l3_total_bytes()?;
    let a_bytes = m.saturating_mul(k).saturating_mul(elsize).max(1);
    let fit = (l3 / a_bytes).max(1);
    let macs = m.saturating_mul(n).saturating_mul(k);
    let work = (macs / min_task_mac.max(1)).max(1);
    let geom = n.div_ceil(16).max(1);
    let bands = fit.min(work).max(workers.max(1)).min(geom);
    // Snap the band count down to full pool waves (shipped,
    // `SONIUM_GEMM_WAVE=0` restores the legacy count): the
    // NR-rounding below can collapse the count to a partial wave
    // (d768: 30 -> cpt 32 -> 24 bands = 1.5 waves on 16 workers:
    // T12->T16 stalls +3%, T24 jumps 1.53x to one wave; snapping to
    // 16 bands wins 0.80-0.85x both orders), stranding workers
    // exactly like the `cap_bands_by_work` partial-wave note.
    // Snapping down widens bands (fewer A restreams); the pool floor
    // keeps at least one full wave, and the geometry clamp keeps the
    // geometry-bound case (bands < workers, e.g. narrow-n) exactly at
    // the legacy count instead of oversubscribing. Band count never
    // changes fp grouping (each element keeps its task and pass
    // structure), so this is bitwise-neutral. Read per call for
    // same-process A/B.
    let bands = if std::env::var("SONIUM_GEMM_WAVE")
        .map(|v| v == "0")
        .unwrap_or(false)
    {
        bands
    } else {
        ((bands / workers.max(1)).max(1) * workers.max(1)).min(geom)
    };
    // Floor-round the band width to whole micro-panels (shipped,
    // `SONIUM_GEMM_CPTFLR=0` restores ceiling-rounding): ceil can
    // collapse the snapped count back to a partial wave (n=260:
    // 16 planned -> cpt 17 -> 24 -> 11 bands, 5 workers idle,
    // d1kn260 3.34 vs 2.15 at n=256), while floor keeps all planned
    // bands full and leaves the remainder to one narrow tail band in
    // a near-empty second wave (trial 2026-09-28, 16T solos:
    // d1kn260 3.34 -> 2.00, 0.60x, knob-off restores 3.22;
    // d1kn252/256, d768, d1k controls flat).
    let ceil = std::env::var("SONIUM_GEMM_CPTFLR")
        .map(|v| v == "0")
        .unwrap_or(false);
    if ceil {
        return Some((n.div_ceil(bands).div_ceil(nr) * nr).max(nr));
    }
    Some((n.div_ceil(bands) / nr * nr).max(nr))
}

/// Minimum per-task work for the N-split path: per-task overhead (~0.3ms
/// at 512^3 — B-tile checkout, A-panel refetch) dominates small tasks
/// (task-time medians 0.69ms for 16.8 MFLOP at 16 bands vs 1.1ms for
/// 33.5 MFLOP at 8 bands), so bands below this size lose more to fixed
/// cost than they gain from extra waves. Pinned by the 512^3 sweep
/// (16 -> 8 bands: 0.47x -> 0.61x); large shapes already exceed it
/// (1024^3 stays 16 bands).
///
/// Band-count override trialled 2026-09-27 via a temporary env gate
/// (removed): forcing 8/16/32/64 bands at f64 `1920²×128` (default ~30)
/// moves 148/206/191/214 GF/s vs OpenBLAS ~330 — flat everywhere except
/// 8-band underfill (half the pool idle), so neither A re-read volume
/// (16x vs 64x) nor wave tails explain the thin-k scaling gap. Do not
/// retry band-count tuning for this; the lever is elsewhere (packed
/// operands or per-thread blocking, untested).
const MIN_TASK_MAC: usize = 15_000_000;

/// f32 N-split per-task floor (2026-09-26): the f32 kernel runs ~2x the
/// f64 rate, so N-split bands go L3-traffic-bound (each band re-streams
/// the full `A` slab per Kc pass) long before the f64 path does —
/// halving 1024^3 from 16 to 8 bands saves ~128 MB of re-reads,
/// measured 3.5 -> 2.8ms interleaved x3 at 8T (beats OpenBLAS 3.40).
/// f64 is compute-bound there (6.5x scaling) and ties at 8 bands, so its
/// floor stays at [`MIN_TASK_MAC`]. Value: any floor putting 1024^3's
/// `want` in [1,15] snaps to 8 bands on 8 workers; 150M keeps 2048^3
/// (want 57 -> 32 bands, geometry-capped) and small shapes (geometry-
/// and pool-floors bind first) exactly where they were — verified ties
/// at 256^3/512^3 f32+f64. f32-2048^3 at fewer bands was trialled
/// 2026-09-26 via a temporary force-gate (removed): 8 hurts (24.0 vs
/// 22.5), 16/64 tie the default 32 (22.5-22.8) — fewer bands only win
/// while per-task C-tiles stay L2-resident (512KB at 8x1024^2, 2MB at
/// 8x2048^2 thrashes). A footprint-based band rule (count from tile
/// size, not just MACs) is the follow-up; traffic math alone does not
/// predict 2048.
const F32_MIN_TASK_MAC: usize = 150_000_000;

/// Narrow an N-split band count (widen bands) until each task covers
/// [`MIN_TASK_MAC`], never below pool width (underfill: 192^3 loses at
/// 3 bands on 8 workers) and never above what geometry yields. The count
/// snaps down to full pool waves: a partial wave strands workers (13
/// bands on 8 workers measured worse than both 8 and 16 at 512^3).
/// Applied at the N-split drivers — not inside [`cols_per_task`], whose
/// value feeds the 2D gate's `bands_n` veto (see its note). Returns the
/// replacement column width, rounded up to whole micro-panels.
/// Band-count audit 2026-09-26 (temporary print probe, removed): portfolio
/// counts on 8 workers are 256:8, 512:8, 1024 f32:8 / f64:16, 2048:16 —
/// every one keeps per-task C-tiles at or under ~1MB (128x1024 f32,
/// 64x1024 f64, 128x2048), which is why the MAC floor and the geometry
/// agree there. A footprint-first rule would only differ on shapes
/// outside this envelope (e.g. f64-2048 at 2MB tiles); changing the
/// two-wave pool floor to reach them is untested — do not generalize
/// from the 1024 result without shape-by-shape A/B.
#[inline]
fn cap_bands_by_work(
    n: usize,
    macs: usize,
    geom_bands: usize,
    nr: usize,
    min_task_mac: usize,
) -> usize {
    let workers = super::workers_for_macs(macs).max(1);
    let want = geom_bands.min(macs.div_ceil(min_task_mac).max(1));
    let bands = (want / workers).max(1) * workers;
    let bands = bands.min(geom_bands).max(workers.min(geom_bands)).max(1);
    (n.div_ceil(bands).div_ceil(nr) * nr).max(nr).min(n.max(1))
}

/// 2D-split gate (single-pass thin-`k`): taken when partitioning both the
/// row and column axes beats the N-split's full-height column bands. The
/// N-split streams the whole `A` panel (`m × k`) from L3 once per column
/// band — 64 MB of re-reads at `1984×1024×64` — while a 2D grid with `ct`
/// column bands reads each `A` row `ct` times. Capping the band width so
/// the private `B` tile (`k × c`) stays L1-resident (`c <= 4096/k` f64
/// elements) and requiring at most half the N-split's bands (with at least
/// two column bands) keeps the B stream in L1 while
/// cutting `A` traffic several-fold. Single-pass only (`k <= kc`): each
/// task packs its `B` tile once and sweeps once, so there is no per-pass
/// re-pack. Tasks own disjoint `C` rectangles swept in serial order, so
/// results are bitwise identical to the serial nest either way.
fn use_parallel_2d(
    m: usize,
    n: usize,
    k: usize,
    nr: usize,
    kc: usize,
    l1_els: usize,
    workers: usize,
) -> bool {
    if k > kc || k == 0 || m < 2 * MR || n < 2 * nr {
        return false;
    }
    if m.saturating_mul(n).saturating_mul(k) < par_min_macs(k, kc) {
        return false;
    }
    if (m / MR).saturating_mul(n / nr) < 2 * workers.max(1) {
        return false;
    }
    let c_l1 = (l1_els / k.max(1) / nr).max(1) * nr;
    let ct = n.div_ceil(c_l1.min(n));
    if ct < 2 {
        return false;
    }
    let bands_n = n.div_ceil(cols_per_task(m, n, k, nr, workers).max(1));
    if !force_2d() && ct * 2 > bands_n {
        return false;
    }
    // Tile coverage after panel rounding: the grid must fill two waves of
    // the engaged pool. A single row band (`rt == 1`) is fine when the
    // column bands alone cover the pool — it halves the N-split's bands
    // with zero `B`-pack redundancy across row bands.
    let (r, c) = grid_2d(m, n, k, nr, l1_els, workers);
    m.div_ceil(r).saturating_mul(n.div_ceil(c)) >= 2 * workers.max(1)
}

/// 2D grid sizer matching [`use_parallel_2d`]: column bands capped so the
/// private `B` tile stays L1-resident, row bands sized to fill two waves
/// of the engaged pool. Widths round down to whole micro-panels (masked
/// remainder tails cost more than their columns); the edge bands absorb
/// the remainders.
fn grid_2d(
    m: usize,
    n: usize,
    k: usize,
    nr: usize,
    l1_els: usize,
    workers: usize,
) -> (usize, usize) {
    let c_l1 = (l1_els / k.max(1) / nr).max(1) * nr;
    let ct = n.div_ceil(c_l1.min(n)).max(1);
    let rt = (2 * workers.max(1)).div_ceil(ct).clamp(1, (m / MR).max(1));
    let round_down_mr = |w: usize| (w / MR * MR).max(MR).min(m.max(1));
    let round_down_nr = |w: usize| (w / nr * nr).max(nr).min(n.max(1));
    let mut rpt = round_down_mr(m.div_ceil(rt).max(MR));
    let mut cpt = round_down_nr(n.div_ceil(ct).max(nr));
    // Guarantee pool-filling: shrink the coarser axis (in panel units)
    // until the grid covers the engaged pool.
    while m.div_ceil(rpt).saturating_mul(n.div_ceil(cpt)) < workers.max(1) && (rpt > MR || cpt > nr)
    {
        if m.div_ceil(rpt) / MR >= n.div_ceil(cpt) / nr && rpt > MR {
            rpt = round_down_mr(rpt / 2);
        } else if cpt > nr {
            cpt = round_down_nr(cpt / 2);
        } else {
            rpt = round_down_mr(rpt / 2);
        }
    }
    (rpt, cpt)
}

/// Minimum GEMM tile work: a rayon scope spawn costs ~3µs (measured
/// 0.13–0.29ms spawning 64 tasks at `384×512×128`), so sub-million-MAC
/// tiles spend more spawning than computing while adding join traffic.
/// Large shapes already exceed this via `TASK_TARGET_MACS`, so the floor
/// only ever widens small-call tiles — and the [`widen_to_min_tile`]
/// pool-fill cap keeps that widening from stranding workers.
const MIN_TILE_MACS: usize = 1_000_000;

/// Widen a GEMM task width so each tile covers at least [`MIN_TILE_MACS`]:
/// `per_unit` is the pass MACs per row/column (`m × kc` for N-split
/// columns, `n × kc` for M-split rows) and `align` the micro-panel width
/// rows/columns round to (`MR`/`NR`: partial panels cost more than their
/// elements). Never past the axis length, and never so wide that the bands
/// underfill the engaged pool: `min_bands` is the executing width (the
/// [`super::workers_for_macs`] value for the scope's own MACs, so router
/// and sizer agree). One full wave is the optimum — within a wave thinner
/// bands only shorten the longest task, past it each extra band adds a
/// per-task spawn to every wave — so the widen caps at the widest aligned
/// width that still yields `min_bands` bands (64x960x64 widened to 4 bands
/// on 8 workers runs half idle; capped it fills all 8).
///
/// WORK-CAP FALSIFIED 2026-10-01 (temp hook, reverted): capping the
/// band mandate at one band per [`MIN_TILE_MACS`] of pass work (5 fat
/// bands for 4.6M MACs at 2000x48x48 instead of 17 thin ones) loses
/// 0.134 -> 0.202ms there — within a wave, thinner bands' shorter
/// longest-task beats fatter bands' saved spawns, and the 16+1 tail
/// costs less than the fattening. (Wide-n N-split liked it: 2000x1952x48
/// 1.46 -> 1.39ms via fewer A re-reads — but N-split has its own
/// cap_bands_by_work for that; do not generalize the M-split loss.)
#[inline]
fn widen_to_min_tile(
    len: usize,
    per_unit: usize,
    align: usize,
    width: usize,
    min_bands: usize,
) -> usize {
    let need = MIN_TILE_MACS.div_ceil(per_unit.max(1));
    let round_up = |w: usize| (w.div_ceil(align) * align).max(align).min(len.max(1));
    let widened = width.max(round_up(need)).min(len.max(1));
    if min_bands <= 1 {
        return widened;
    }
    // Round down so the band count stays at or above `min_bands`; a
    // degenerate axis (shorter than `align * min_bands`) still yields its
    // maximum aligned bands.
    let fill = (len.div_ceil(min_bands) / align * align).max(align);
    widened.min(fill).max(align).min(len.max(1))
}

// Private-B-panel L2 budget: a per-task panel must fit this share of
// the 512KB L2 (headroom for the task's A/C streams). Per-precision
// (interleaved 2026-10-02): f32 fires at half budget (n=256/256KB
// panels regressed +61% at full budget: 64 tasks x pack writes exceed
// the shared path's L3 re-read savings), while f64 needs the full
// budget (its kc0=512 shared panels are 4x bigger, so n=128/kc=256
// still wins -16%).
const PRIVB_L2_BUDGET_F32: usize = 128 * 1024;
const PRIVB_L2_BUDGET_F64: usize = 256 * 1024;

/// Largest `kc` whose `kc × panel_n` private panel fits `budget`
/// (floored at 128 so the k-loop
/// keeps its unrolled rate; callers treat sub-128 as shared-panel).
fn privb_kc_cap<T>(panel_n: usize, budget: usize) -> usize {
    (budget / core::mem::size_of::<T>() / panel_n.max(1)).max(128)
}

// f32: 16-col panels (2 × YMM), B tile capped at 256 × 256 × 4 B = 256 KB:
// a 512 KB tile exactly fills the 512 KB L2 and thrashes against the A/C
// streams (measured slower on every shape at 1/32/128 threads).
const F32_NR: usize = 16;
/// f32 full-tile row height for [`micro_f32_full6`]: 6 rows × 16 cols for
/// 12 independent FMA chains (f32 twin of [`F64_MR6`]).
const F32_MR6: usize = 6;
const F32_KC: usize = 256;
const F32_NC: usize = 256;

/// N-split Kc width (f32, shipped, `SONIUM_GEMM_KC=256` restores the
/// legacy 256): 512 at `k >= 768`, else 256. Halving the pass count
/// halves the N-split whole-A restream per task and the C-band revisit;
/// the wider B tile (512x256 = L2-size) costs L2-residence, which wins
/// once C-revisit savings dominate (sq1kT +3-8%, sq2kT/NT +3-8%,
/// r1k512T/NT +6-16%, sq768NT +3-9%, sq512NT +14%, m512n2kT +7-8%;
/// sq1kNT/sq768T tie; sq512T loses 0.90x (small-C: revisit free, pure
/// B-spill cost — hence the `k >= 768` floor); kc1024 loses NT/sq2k
/// (1.16-1.45x, B-tile L2-spill dominates) and is not shipped. Pass
/// boundaries change fp grouping (same terms — matches scalar within
/// the standard f32 tolerance, like the single-pass 2D path; not
/// bitwise vs Kc=256). Read per call for same-process A/B.
const F32_KC_WIDE: usize = 512;
const F32_KC_WIDE_MIN_K: usize = 768;

#[inline]
fn f32_kc_nsplit(k: usize) -> usize {
    if std::env::var("SONIUM_GEMM_KC")
        .map(|v| v == "256")
        .unwrap_or(false)
    {
        return F32_KC;
    }
    if k >= F32_KC_WIDE_MIN_K {
        F32_KC_WIDE
    } else {
        F32_KC
    }
}
/// f32 serial A-panel row height: `Mc × Kc` panels of 256 × 256 × 4 B =
/// 256 KB (same L2 budget as the B tile) for the serial A-pack route.
const F32_MC: usize = 256;

// f64: 8-col panels (2 × YMM), B tile capped at 256 × 128 × 8 B = 256 KB,
// same L2 sizing as f32 (measured faster everywhere, e.g. 1024^3 single
// 31.9 → 44.0 GF/s).
const F64_NR: usize = 8;
/// f64 full-tile row height for [`micro_f64_full6`]: 6 rows × 8 cols gives
/// 12 independent FMA chains (see the kernel docs). Tails (`mb < 6`) stay
/// on the 4-wide kernels via [`micro_f64`], whose accumulator array is
/// sized for this height.
const F64_MR6: usize = 6;
const F64_KC: usize = 256;
/// M-split Kc: deeper K-blocking halves the pass/dispatch count on tall
/// shapes, where the packed-B band is shared by every row task so the
/// bigger band stays L3-resident (`1984x32x1984` runs 0.56x at 256,
/// 1.25x at 512). N-split keeps 256: its per-task tiles are private, and
/// 512-deep tiles spill narrow tasks out of L1 (`128x1024x512` drops
/// 0.85x to 0.57x).
const F64_KC_M: usize = 512;
/// N-split Kc width (f64, shipped, `SONIUM_GEMM_DKC=256` restores the
/// legacy 256): 512 at `768 <= k <= 1152` and `n >= 512`, else 256.
/// Halving the pass count halves the N-split whole-A restream per task
/// and the C-band revisit (d1kNT +12%, d1kT +10%, d768NT +5%,
/// 256x1024x1024 +10%, 512x2048x1024 +10%, all reproduced both orders;
/// pass boundaries change fp grouping — maxdiff vs OpenBLAS ~6e-15).
/// The `k <= 1152` cap excludes the measured loss domain (1024x1024x2048
/// 1.03-1.07x, 2048^3 1.06-1.24x — the 512 KB B tile fills the 512 KB L2
/// and thrashes against the A/C streams once the pass working set exceeds
/// the L3 slice) and the `n >= 512` floor excludes narrow-n (1024x256x1024
/// 1.02-1.08x — few bands, B-spill cost dominates). Read per call for
/// same-process A/B.
const F64_KC_WIDE: usize = 512;
const F64_KC_WIDE_MIN_K: usize = 768;
const F64_KC_WIDE_MAX_K: usize = 1152;
const F64_KC_WIDE_MIN_N: usize = 512;
#[inline]
fn f64_kc_nsplit(n: usize, k: usize) -> usize {
    if std::env::var("SONIUM_GEMM_DKC")
        .map(|v| v == "256")
        .unwrap_or(false)
    {
        return F64_KC;
    }
    if (F64_KC_WIDE_MIN_K..=F64_KC_WIDE_MAX_K).contains(&k) && n >= F64_KC_WIDE_MIN_N {
        F64_KC_WIDE
    } else {
        F64_KC
    }
}
const F64_NC: usize = 128;
/// f64 serial A-panel row height: 128 × 256 × 8 B = 256 KB (same L2
/// budget as the B tile) for the serial A-pack route.
const F64_MC: usize = 128;

/// Serial A-panel pack route (shipped default-on,
/// `SONIUM_GEMM_ST_APACK=0` restores the legacy nest): when `A`
/// overflows one L3 slice the serial nest's per-(jc, pc) `A` re-reads
/// (128 MiB at sq2k) self-evict; packing `Mc × kc` panels once per
/// (pc, ic) and caching them across jc-tiles cuts that to ~16 MiB
/// (STAP A/B 3/3: f32 1.017x/1.025x/1.030x, f64 1.045x/1.033x/1.012x
/// at sq2k; sub-slice controls tie 0.995-1.000x). NoTrans-A only
/// (Trans has its own serial pre-pack); all-blocking-multi (`m > Mc`,
/// `k > Kc`, `n > Nc`) so every pack amortizes; `A >= slice` (else
/// re-reads hit L3 ~free). Read per call for same-process A/B.
#[inline]
fn st_apack_on() -> bool {
    std::env::var("SONIUM_GEMM_ST_APACK")
        .map(|v| v != "0")
        .unwrap_or(true)
}

/// Shape clauses of the serial A-pack gate: blocked-multi in every
/// axis plus an `A` footprint at/over one L3 slice (`None` slice =
/// status quo). Pure over an injected slice size so the dispatch is
/// unit-testable without running a solve.
fn st_apack_shape(
    m: usize,
    n: usize,
    k: usize,
    mc: usize,
    kc: usize,
    nc: usize,
    elsize: usize,
    slice: Option<u64>,
) -> bool {
    let slice = match slice {
        Some(s) => s as usize,
        None => return false,
    };
    m > mc && n > nc && k > kc && m.saturating_mul(k).saturating_mul(elsize) >= slice
}

/// Lane mask for the low 8 f32 lanes holding `count` valid columns.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn mask_ps(count: usize) -> __m256i {
    _mm256_cmpgt_epi32(
        _mm256_set1_epi32(count as i32),
        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
    )
}

/// Lane mask for the low 4 f64 lanes holding `count` valid columns.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn mask_pd(count: usize) -> __m256i {
    _mm256_cmpgt_epi64(
        _mm256_set1_epi64x(count as i64),
        _mm256_setr_epi64x(0, 1, 2, 3),
    )
}

/// Full-panel (`mb == MR`) f32 fast path with scalar accumulators.
///
/// Same operation and per-element accumulation order as
/// [`micro_f32`]; see [`micro_f64_full`] for why the stack array had to
/// go (hot-loop spills against 16 ymm registers).
#[target_feature(enable = "avx2,fma")]
#[inline(never)]
unsafe fn micro_f32_full<const FULL: bool, const SKIP_BETA_MUL: bool>(
    nb: usize,
    kc: usize,
    alpha: f32,
    a_ptr: *const f32,
    a_row_stride: usize,
    a_col_stride: usize,
    b_ptr: *const f32,
    ldb: usize,
    beta: f32,
    c_ptr: *mut f32,
    ldc: usize,
) {
    let (mask_lo, mask_hi) = if FULL {
        (_mm256_setzero_si256(), _mm256_setzero_si256())
    } else {
        (mask_ps(nb.min(8)), mask_ps(nb.saturating_sub(8)))
    };
    let hi_off = nb.min(8);

    let mut acc00 = _mm256_setzero_ps();
    let mut acc01 = _mm256_setzero_ps();
    let mut acc10 = _mm256_setzero_ps();
    let mut acc11 = _mm256_setzero_ps();
    let mut acc20 = _mm256_setzero_ps();
    let mut acc21 = _mm256_setzero_ps();
    let mut acc30 = _mm256_setzero_ps();
    let mut acc31 = _mm256_setzero_ps();

    let mut ap0 = a_ptr;
    let mut ap1 = a_ptr.add(a_row_stride);
    let mut ap2 = a_ptr.add(a_row_stride * 2);
    let mut ap3 = a_ptr.add(a_row_stride * 3);
    let mut bp = b_ptr;
    let mut p = 0;
    while p < kc {
        let (bv_lo, bv_hi) = if FULL {
            (_mm256_loadu_ps(bp), _mm256_loadu_ps(bp.add(hi_off)))
        } else {
            (
                _mm256_maskload_ps(bp, mask_lo),
                _mm256_maskload_ps(bp.add(hi_off), mask_hi),
            )
        };
        let a0 = _mm256_set1_ps(*ap0);
        ap0 = ap0.add(a_col_stride);
        let a1 = _mm256_set1_ps(*ap1);
        ap1 = ap1.add(a_col_stride);
        let a2 = _mm256_set1_ps(*ap2);
        ap2 = ap2.add(a_col_stride);
        let a3 = _mm256_set1_ps(*ap3);
        ap3 = ap3.add(a_col_stride);
        acc00 = _mm256_fmadd_ps(a0, bv_lo, acc00);
        acc01 = _mm256_fmadd_ps(a0, bv_hi, acc01);
        acc10 = _mm256_fmadd_ps(a1, bv_lo, acc10);
        acc11 = _mm256_fmadd_ps(a1, bv_hi, acc11);
        acc20 = _mm256_fmadd_ps(a2, bv_lo, acc20);
        acc21 = _mm256_fmadd_ps(a2, bv_hi, acc21);
        acc30 = _mm256_fmadd_ps(a3, bv_lo, acc30);
        acc31 = _mm256_fmadd_ps(a3, bv_hi, acc31);
        bp = bp.add(ldb);
        p += 1;
    }

    let va = _mm256_set1_ps(alpha);
    let vb = _mm256_set1_ps(beta);
    let rows = [
        (c_ptr, acc00, acc01),
        (c_ptr.add(ldc), acc10, acc11),
        (c_ptr.add(ldc * 2), acc20, acc21),
        (c_ptr.add(ldc * 3), acc30, acc31),
    ];
    for &(crow, al, ah) in rows.iter() {
        let (out_lo, out_hi) = if beta == 0.0 {
            (_mm256_mul_ps(va, al), _mm256_mul_ps(va, ah))
        } else {
            let (c_lo, c_hi) = if FULL {
                (_mm256_loadu_ps(crow), _mm256_loadu_ps(crow.add(hi_off)))
            } else {
                (
                    _mm256_maskload_ps(crow, mask_lo),
                    _mm256_maskload_ps(crow.add(hi_off), mask_hi),
                )
            };
            // beta == 1.0 scaling is exact (see `micro_f32_full6`).
            let (bl, bh) = if SKIP_BETA_MUL {
                (c_lo, c_hi)
            } else {
                (_mm256_mul_ps(vb, c_lo), _mm256_mul_ps(vb, c_hi))
            };
            (_mm256_fmadd_ps(va, al, bl), _mm256_fmadd_ps(va, ah, bh))
        };
        if FULL {
            _mm256_storeu_ps(crow, out_lo);
            _mm256_storeu_ps(crow.add(hi_off), out_hi);
        } else {
            _mm256_maskstore_ps(crow, mask_lo, out_lo);
            _mm256_maskstore_ps(crow.add(hi_off), mask_hi, out_hi);
        }
    }
}

/// f32 6×16 full-tile micro-panel: 12 independent FMA chains instead of
/// the 4×16 kernel's 8 (same Zen2 latency arithmetic as
/// [`micro_f64_full6`]). Register budget: 12 accumulators + 2 B vectors +
/// 1 reused A broadcast = 15 YMM. Same values and accumulation order as
/// [`micro_f32_full`], so results are bitwise identical; only the
/// full-tile case exists (tails stay on the 4-wide kernels).
/// `#[inline(never)]` like [`micro_f32_full`]: the wider frame must not
/// dissolve into the sweep body (register spills there).
/// `SKIP_BETA_MUL`: skip the `beta * C` multiply for `beta == 1.0`
/// (scaling by 1.0 is exact, bitwise identical). Shaves ~5 cycles of
/// epilogue chain + port pressure per tile (~2% on thin-k beta=1,
/// A/B 2026-09-27).
/// Const-dispatched (not a runtime branch: a runtime flag + opaque
/// call inside this frame collapses codegen ~2x, measured 2026-09-27
/// — register budget is 15/16 YMM, regalloc spills).
/// Single-pointer A addressing (shipped 2026-09-28, ~1% ST, bitwise):
/// one running `ap` advanced by `a_col_stride` per step plus five
/// loop-invariant row offsets in GPRs (free `[base + off]` AGU
/// addressing) instead of six pointers each advanced per step — 4
/// fewer integer adds per k-step (28→24 x86 instructions for the same
/// 12 FMAs; same-process A/B 17/18 cells 0.975-1.006x). `a_offs`
/// points at 5 caller-computed row offsets (elements); passing them
/// through memory (not deriving them here) keeps the progression
/// opaque so LSR cannot regenerate the six-pointer loop (verified in
/// asm) — robust across LLVM versions, unlike a `black_box` (which
/// also spills in the prologue).
#[target_feature(enable = "avx2,fma")]
#[inline(never)]
unsafe fn micro_f32_full6<const SKIP_BETA_MUL: bool>(
    kc: usize,
    alpha: f32,
    a_ptr: *const f32,
    a_offs: *const usize,
    a_col_stride: usize,
    b_ptr: *const f32,
    ldb: usize,
    beta: f32,
    c_ptr: *mut f32,
    ldc: usize,
) {
    let mut acc00 = _mm256_setzero_ps();
    let mut acc01 = _mm256_setzero_ps();
    let mut acc10 = _mm256_setzero_ps();
    let mut acc11 = _mm256_setzero_ps();
    let mut acc20 = _mm256_setzero_ps();
    let mut acc21 = _mm256_setzero_ps();
    let mut acc30 = _mm256_setzero_ps();
    let mut acc31 = _mm256_setzero_ps();
    let mut acc40 = _mm256_setzero_ps();
    let mut acc41 = _mm256_setzero_ps();
    let mut acc50 = _mm256_setzero_ps();
    let mut acc51 = _mm256_setzero_ps();

    // Caller-computed row offsets (opaque loads — see above); the
    // single running pointer advances by `a_col_stride` per step.
    let off1 = *a_offs;
    let off2 = *a_offs.add(1);
    let off3 = *a_offs.add(2);
    let off4 = *a_offs.add(3);
    let off5 = *a_offs.add(4);
    // NOTE (reverted trials 2026-09-26): a k×2 unroll and an L2-stream
    // prefetch (`bp + 4*ldb`), alone and combined, all measure zero by the
    // within-run OB ratio protocol (65% small-`k`, 74% fat-`k` either
    // way). An early +34% read was boost artifact — absolute GF comparisons
    // across runs are meaningless on this box; always normalize against
    // the in-run OB rate. The loop is neither overhead- nor stream-bound
    // at this shape.
    // NOTE (falsified 2026-09-27): a beta-gated C-read prefetch here
    // (six T0 of the tile's C rows ahead of the k-loop, same-binary
    // A/B) loses 6-8% x3 on thin-k beta=1 — the epilogue C-miss theory
    // is wrong (or T0 evicts the B stream); do not retry T0 here.
    let mut ap = a_ptr;
    let mut bp = b_ptr;
    let mut p = 0;
    while p < kc {
        let bv_lo = _mm256_loadu_ps(bp);
        let bv_hi = _mm256_loadu_ps(bp.add(8));
        let a0 = _mm256_set1_ps(*ap);
        acc00 = _mm256_fmadd_ps(a0, bv_lo, acc00);
        acc01 = _mm256_fmadd_ps(a0, bv_hi, acc01);
        let a1 = _mm256_set1_ps(*ap.add(off1));
        acc10 = _mm256_fmadd_ps(a1, bv_lo, acc10);
        acc11 = _mm256_fmadd_ps(a1, bv_hi, acc11);
        let a2 = _mm256_set1_ps(*ap.add(off2));
        acc20 = _mm256_fmadd_ps(a2, bv_lo, acc20);
        acc21 = _mm256_fmadd_ps(a2, bv_hi, acc21);
        let a3 = _mm256_set1_ps(*ap.add(off3));
        acc30 = _mm256_fmadd_ps(a3, bv_lo, acc30);
        acc31 = _mm256_fmadd_ps(a3, bv_hi, acc31);
        let a4 = _mm256_set1_ps(*ap.add(off4));
        acc40 = _mm256_fmadd_ps(a4, bv_lo, acc40);
        acc41 = _mm256_fmadd_ps(a4, bv_hi, acc41);
        let a5 = _mm256_set1_ps(*ap.add(off5));
        acc50 = _mm256_fmadd_ps(a5, bv_lo, acc50);
        acc51 = _mm256_fmadd_ps(a5, bv_hi, acc51);
        ap = ap.add(a_col_stride);
        bp = bp.add(ldb);
        p += 1;
    }

    let va = _mm256_set1_ps(alpha);
    let vb = _mm256_set1_ps(beta);
    let c0 = c_ptr;
    let c1 = c_ptr.add(ldc);
    let c2 = c_ptr.add(ldc * 2);
    let c3 = c_ptr.add(ldc * 3);
    let c4 = c_ptr.add(ldc * 4);
    let c5 = c_ptr.add(ldc * 5);
    let rows = [
        (c0, acc00, acc01),
        (c1, acc10, acc11),
        (c2, acc20, acc21),
        (c3, acc30, acc31),
        (c4, acc40, acc41),
        (c5, acc50, acc51),
    ];
    for &(crow, al, ah) in rows.iter() {
        let (out_lo, out_hi) = if beta == 0.0 {
            (_mm256_mul_ps(va, al), _mm256_mul_ps(va, ah))
        } else {
            let c_lo = _mm256_loadu_ps(crow);
            let c_hi = _mm256_loadu_ps(crow.add(8));
            let (bl, bh) = if SKIP_BETA_MUL {
                (c_lo, c_hi)
            } else {
                (_mm256_mul_ps(vb, c_lo), _mm256_mul_ps(vb, c_hi))
            };
            (_mm256_fmadd_ps(va, al, bl), _mm256_fmadd_ps(va, ah, bh))
        };
        _mm256_storeu_ps(crow, out_lo);
        _mm256_storeu_ps(crow.add(8), out_hi);
    }
}

/// f32 micro-panel: `C[0..mb, 0..nb] := alpha * A·B + beta * C` over `kc`
/// inner steps. `FULL` selects unmasked B/C column accesses (nb == 16).
///
/// `a_ptr` points at `A[i0, p_start]`-equivalent storage; element (r, p) is
/// at `a_ptr + r * a_row_stride + p * a_col_stride`, which covers both
/// NoTrans (lda, 1) and Trans (1, lda) sources. `b_ptr` points at
/// `B[p_start, j0]` with row stride `ldb`; `c_ptr` at `C[i0, j0]`.
#[target_feature(enable = "avx2,fma")]
unsafe fn micro_f32<const FULL: bool, const SKIP_BETA_MUL: bool>(
    mb: usize,
    nb: usize,
    kc: usize,
    alpha: f32,
    a_ptr: *const f32,
    a_row_stride: usize,
    a_col_stride: usize,
    b_ptr: *const f32,
    ldb: usize,
    beta: f32,
    c_ptr: *mut f32,
    ldc: usize,
) {
    let (mask_lo, mask_hi) = if FULL {
        (_mm256_setzero_si256(), _mm256_setzero_si256())
    } else {
        (mask_ps(nb.min(8)), mask_ps(nb.saturating_sub(8)))
    };
    // Offset of the second vector within a panel row. `nb.min(8)` keeps the
    // pointer in-bounds when the high vector is entirely masked off (nb ≤ 8
    // and a tight `ldb`), where `brow.add(8)` could leave the allocation.
    let hi_off = nb.min(8);

    let mut acc = [[_mm256_setzero_ps(); 2]; F32_MR6];
    for p in 0..kc {
        let brow = b_ptr.add(p * ldb);
        let (bv_lo, bv_hi) = if FULL {
            (_mm256_loadu_ps(brow), _mm256_loadu_ps(brow.add(hi_off)))
        } else {
            (
                _mm256_maskload_ps(brow, mask_lo),
                _mm256_maskload_ps(brow.add(hi_off), mask_hi),
            )
        };
        for (r, acc_r) in acc.iter_mut().enumerate().take(mb) {
            let av = _mm256_set1_ps(*a_ptr.add(r * a_row_stride + p * a_col_stride));
            acc_r[0] = _mm256_fmadd_ps(av, bv_lo, acc_r[0]);
            acc_r[1] = _mm256_fmadd_ps(av, bv_hi, acc_r[1]);
        }
    }

    let va = _mm256_set1_ps(alpha);
    let vb = _mm256_set1_ps(beta);
    for (r, acc_r) in acc.iter().enumerate().take(mb) {
        let crow = c_ptr.add(r * ldc);
        let (out_lo, out_hi) = if beta == 0.0 {
            (_mm256_mul_ps(va, acc_r[0]), _mm256_mul_ps(va, acc_r[1]))
        } else {
            let (c_lo, c_hi) = if FULL {
                (_mm256_loadu_ps(crow), _mm256_loadu_ps(crow.add(hi_off)))
            } else {
                (
                    _mm256_maskload_ps(crow, mask_lo),
                    _mm256_maskload_ps(crow.add(hi_off), mask_hi),
                )
            };
            // beta == 1.0 scaling is exact (see `micro_f32_full6`).
            let (bl, bh) = if SKIP_BETA_MUL {
                (c_lo, c_hi)
            } else {
                (_mm256_mul_ps(vb, c_lo), _mm256_mul_ps(vb, c_hi))
            };
            (
                _mm256_fmadd_ps(va, acc_r[0], bl),
                _mm256_fmadd_ps(va, acc_r[1], bh),
            )
        };
        if FULL {
            _mm256_storeu_ps(crow, out_lo);
            _mm256_storeu_ps(crow.add(hi_off), out_hi);
        } else {
            _mm256_maskstore_ps(crow, mask_lo, out_lo);
            _mm256_maskstore_ps(crow.add(hi_off), mask_hi, out_hi);
        }
    }
}

/// Full-panel (`mb == MR`) f64 fast path with scalar accumulators.
///
/// Same operation and per-element accumulation order as
/// [`micro_f64`], but the 8 accumulators are named locals instead of a
/// stack array iterated with a runtime `take(mb)`: that abstraction kept
/// the array in memory and the hot loop ran load/FMA/store against stack
/// slots (~40% of per-core peak). `#[inline(never)]` keeps this loop in
/// its own register-allocation frame instead of dissolved into the
/// sweep's giant inlined body, which spilled B vectors to the stack.
#[target_feature(enable = "avx2,fma")]
#[inline(never)]
unsafe fn micro_f64_full<const FULL: bool, const SKIP_BETA_MUL: bool>(
    nb: usize,
    kc: usize,
    alpha: f64,
    a_ptr: *const f64,
    a_row_stride: usize,
    a_col_stride: usize,
    b_ptr: *const f64,
    ldb: usize,
    beta: f64,
    c_ptr: *mut f64,
    ldc: usize,
) {
    let (mask_lo, mask_hi) = if FULL {
        (_mm256_setzero_si256(), _mm256_setzero_si256())
    } else {
        (mask_pd(nb.min(4)), mask_pd(nb.saturating_sub(4)))
    };
    let hi_off = nb.min(4);

    let mut acc00 = _mm256_setzero_pd();
    let mut acc01 = _mm256_setzero_pd();
    let mut acc10 = _mm256_setzero_pd();
    let mut acc11 = _mm256_setzero_pd();
    let mut acc20 = _mm256_setzero_pd();
    let mut acc21 = _mm256_setzero_pd();
    let mut acc30 = _mm256_setzero_pd();
    let mut acc31 = _mm256_setzero_pd();

    // Row pointers advanced by `a_col_stride` per step (add-only addressing).
    let mut ap0 = a_ptr;
    let mut ap1 = a_ptr.add(a_row_stride);
    let mut ap2 = a_ptr.add(a_row_stride * 2);
    let mut ap3 = a_ptr.add(a_row_stride * 3);
    let mut bp = b_ptr;
    let mut p = 0;
    while p < kc {
        let (bv_lo, bv_hi) = if FULL {
            (_mm256_loadu_pd(bp), _mm256_loadu_pd(bp.add(hi_off)))
        } else {
            (
                _mm256_maskload_pd(bp, mask_lo),
                _mm256_maskload_pd(bp.add(hi_off), mask_hi),
            )
        };
        let a0 = _mm256_set1_pd(*ap0);
        ap0 = ap0.add(a_col_stride);
        let a1 = _mm256_set1_pd(*ap1);
        ap1 = ap1.add(a_col_stride);
        let a2 = _mm256_set1_pd(*ap2);
        ap2 = ap2.add(a_col_stride);
        let a3 = _mm256_set1_pd(*ap3);
        ap3 = ap3.add(a_col_stride);
        acc00 = _mm256_fmadd_pd(a0, bv_lo, acc00);
        acc01 = _mm256_fmadd_pd(a0, bv_hi, acc01);
        acc10 = _mm256_fmadd_pd(a1, bv_lo, acc10);
        acc11 = _mm256_fmadd_pd(a1, bv_hi, acc11);
        acc20 = _mm256_fmadd_pd(a2, bv_lo, acc20);
        acc21 = _mm256_fmadd_pd(a2, bv_hi, acc21);
        acc30 = _mm256_fmadd_pd(a3, bv_lo, acc30);
        acc31 = _mm256_fmadd_pd(a3, bv_hi, acc31);
        bp = bp.add(ldb);
        p += 1;
    }

    let va = _mm256_set1_pd(alpha);
    let vb = _mm256_set1_pd(beta);
    // (row base, acc_lo, acc_hi) triples; same beta == 0 fast path as micro_f64.
    let c0 = c_ptr;
    let c1 = c_ptr.add(ldc);
    let c2 = c_ptr.add(ldc * 2);
    let c3 = c_ptr.add(ldc * 3);
    let rows = [
        (c0, acc00, acc01),
        (c1, acc10, acc11),
        (c2, acc20, acc21),
        (c3, acc30, acc31),
    ];
    for &(crow, al, ah) in rows.iter() {
        let (out_lo, out_hi) = if beta == 0.0 {
            (_mm256_mul_pd(va, al), _mm256_mul_pd(va, ah))
        } else {
            let (c_lo, c_hi) = if FULL {
                (_mm256_loadu_pd(crow), _mm256_loadu_pd(crow.add(hi_off)))
            } else {
                (
                    _mm256_maskload_pd(crow, mask_lo),
                    _mm256_maskload_pd(crow.add(hi_off), mask_hi),
                )
            };
            // beta == 1.0 scaling is exact (see `micro_f32_full6`).
            let (bl, bh) = if SKIP_BETA_MUL {
                (c_lo, c_hi)
            } else {
                (_mm256_mul_pd(vb, c_lo), _mm256_mul_pd(vb, c_hi))
            };
            (_mm256_fmadd_pd(va, al, bl), _mm256_fmadd_pd(va, ah, bh))
        };
        if FULL {
            _mm256_storeu_pd(crow, out_lo);
            _mm256_storeu_pd(crow.add(hi_off), out_hi);
        } else {
            _mm256_maskstore_pd(crow, mask_lo, out_lo);
            _mm256_maskstore_pd(crow.add(hi_off), mask_hi, out_hi);
        }
    }
}

/// f64 6×8 full-tile micro-panel: 12 independent FMA chains instead of the
/// 4×8 kernel's 8. Zen2 FMA latency is 5 cycles at 2 FMAs/cycle, so 8
/// chains issue one FMA per chain every 4 cycles and stall ~1 cycle per
/// FMA (~20% — the measured single-thread gap to OpenBLAS); 12 chains
/// issue every 6 cycles and fully cover the latency. Register budget:
/// 12 accumulators + 2 B vectors + 1 reused A broadcast = 15 YMM (A row
/// pointers and the B/C pointers live in GPRs). The broadcast is consumed
/// row-by-row (one live `av` at a time) rather than held for all rows.
/// Same values and accumulation order as [`micro_f64_full`], so results
/// are bitwise identical; only the full-tile case exists (tails stay on
/// the 4-wide kernels).
#[target_feature(enable = "avx2,fma")]
unsafe fn micro_f64_full6<const SKIP_BETA_MUL: bool>(
    kc: usize,
    alpha: f64,
    a_ptr: *const f64,
    a_row_stride: usize,
    a_col_stride: usize,
    b_ptr: *const f64,
    ldb: usize,
    beta: f64,
    c_ptr: *mut f64,
    ldc: usize,
) {
    let mut acc00 = _mm256_setzero_pd();
    let mut acc01 = _mm256_setzero_pd();
    let mut acc10 = _mm256_setzero_pd();
    let mut acc11 = _mm256_setzero_pd();
    let mut acc20 = _mm256_setzero_pd();
    let mut acc21 = _mm256_setzero_pd();
    let mut acc30 = _mm256_setzero_pd();
    let mut acc31 = _mm256_setzero_pd();
    let mut acc40 = _mm256_setzero_pd();
    let mut acc41 = _mm256_setzero_pd();
    let mut acc50 = _mm256_setzero_pd();
    let mut acc51 = _mm256_setzero_pd();

    // NOTE (reverted trial 2026-09-26): two 3-row passes regress 18.2 →
    // 15.4 GF/s — halving the register pressure is not worth re-reading
    // the B micro-panel from L2 twice. The kernel is stream-bound, not
    // spill-bound (see the prefetch below).
    // Row pointers advanced by `a_col_stride` per step (add-only addressing).
    let mut ap0 = a_ptr;
    let mut ap1 = a_ptr.add(a_row_stride);
    let mut ap2 = a_ptr.add(a_row_stride * 2);
    let mut ap3 = a_ptr.add(a_row_stride * 3);
    let mut ap4 = a_ptr.add(a_row_stride * 4);
    let mut ap5 = a_ptr.add(a_row_stride * 5);
    let mut bp = b_ptr;
    let mut p = 0;
    // NOTE (falsified 2026-09-27): same beta-gated C-prefetch trial as
    // the f32 twin, same 6-8% loss — do not retry T0 here.
    while p < kc {
        // NOTE (reverted trial 2026-09-26): an L2-stream prefetch
        // (`bp + 4*ldb`) regresses 18.2 → 17.6 GF/s serial and hurts
        // in-situ dpotrf (3.8 → 4.0ms) — extra traffic/pollution across
        // workers sharing L3, no latency win. The hardware streamer
        // already covers this walk.
        let bv_lo = _mm256_loadu_pd(bp);
        let bv_hi = _mm256_loadu_pd(bp.add(4));
        let a0 = _mm256_set1_pd(*ap0);
        ap0 = ap0.add(a_col_stride);
        acc00 = _mm256_fmadd_pd(a0, bv_lo, acc00);
        acc01 = _mm256_fmadd_pd(a0, bv_hi, acc01);
        let a1 = _mm256_set1_pd(*ap1);
        ap1 = ap1.add(a_col_stride);
        acc10 = _mm256_fmadd_pd(a1, bv_lo, acc10);
        acc11 = _mm256_fmadd_pd(a1, bv_hi, acc11);
        let a2 = _mm256_set1_pd(*ap2);
        ap2 = ap2.add(a_col_stride);
        acc20 = _mm256_fmadd_pd(a2, bv_lo, acc20);
        acc21 = _mm256_fmadd_pd(a2, bv_hi, acc21);
        let a3 = _mm256_set1_pd(*ap3);
        ap3 = ap3.add(a_col_stride);
        acc30 = _mm256_fmadd_pd(a3, bv_lo, acc30);
        acc31 = _mm256_fmadd_pd(a3, bv_hi, acc31);
        let a4 = _mm256_set1_pd(*ap4);
        ap4 = ap4.add(a_col_stride);
        acc40 = _mm256_fmadd_pd(a4, bv_lo, acc40);
        acc41 = _mm256_fmadd_pd(a4, bv_hi, acc41);
        let a5 = _mm256_set1_pd(*ap5);
        ap5 = ap5.add(a_col_stride);
        acc50 = _mm256_fmadd_pd(a5, bv_lo, acc50);
        acc51 = _mm256_fmadd_pd(a5, bv_hi, acc51);
        bp = bp.add(ldb);
        p += 1;
    }

    let va = _mm256_set1_pd(alpha);
    let vb = _mm256_set1_pd(beta);
    // (row base, acc_lo, acc_hi) triples; same beta == 0 fast path as micro_f64.
    let c0 = c_ptr;
    let c1 = c_ptr.add(ldc);
    let c2 = c_ptr.add(ldc * 2);
    let c3 = c_ptr.add(ldc * 3);
    let c4 = c_ptr.add(ldc * 4);
    let c5 = c_ptr.add(ldc * 5);
    let rows = [
        (c0, acc00, acc01),
        (c1, acc10, acc11),
        (c2, acc20, acc21),
        (c3, acc30, acc31),
        (c4, acc40, acc41),
        (c5, acc50, acc51),
    ];
    for &(crow, al, ah) in rows.iter() {
        let (out_lo, out_hi) = if beta == 0.0 {
            (_mm256_mul_pd(va, al), _mm256_mul_pd(va, ah))
        } else {
            let c_lo = _mm256_loadu_pd(crow);
            let c_hi = _mm256_loadu_pd(crow.add(4));
            // beta == 1.0 scaling is exact (see `micro_f32_full6`).
            let (bl, bh) = if SKIP_BETA_MUL {
                (c_lo, c_hi)
            } else {
                (_mm256_mul_pd(vb, c_lo), _mm256_mul_pd(vb, c_hi))
            };
            (_mm256_fmadd_pd(va, al, bl), _mm256_fmadd_pd(va, ah, bh))
        };
        _mm256_storeu_pd(crow, out_lo);
        _mm256_storeu_pd(crow.add(4), out_hi);
    }
}

/// f64 micro-panel — same structure as [`micro_f32`] with 8-col panels.
#[target_feature(enable = "avx2,fma")]
unsafe fn micro_f64<const FULL: bool, const SKIP_BETA_MUL: bool>(
    mb: usize,
    nb: usize,
    kc: usize,
    alpha: f64,
    a_ptr: *const f64,
    a_row_stride: usize,
    a_col_stride: usize,
    b_ptr: *const f64,
    ldb: usize,
    beta: f64,
    c_ptr: *mut f64,
    ldc: usize,
) {
    let (mask_lo, mask_hi) = if FULL {
        (_mm256_setzero_si256(), _mm256_setzero_si256())
    } else {
        (mask_pd(nb.min(4)), mask_pd(nb.saturating_sub(4)))
    };
    // See `micro_f32` — keeps the high-vector pointer in-bounds when it is
    // entirely masked off on a tight `ldb`.
    let hi_off = nb.min(4);

    let mut acc = [[_mm256_setzero_pd(); 2]; F64_MR6];
    for p in 0..kc {
        let brow = b_ptr.add(p * ldb);
        let (bv_lo, bv_hi) = if FULL {
            (_mm256_loadu_pd(brow), _mm256_loadu_pd(brow.add(hi_off)))
        } else {
            (
                _mm256_maskload_pd(brow, mask_lo),
                _mm256_maskload_pd(brow.add(hi_off), mask_hi),
            )
        };
        for (r, acc_r) in acc.iter_mut().enumerate().take(mb) {
            let av = _mm256_set1_pd(*a_ptr.add(r * a_row_stride + p * a_col_stride));
            acc_r[0] = _mm256_fmadd_pd(av, bv_lo, acc_r[0]);
            acc_r[1] = _mm256_fmadd_pd(av, bv_hi, acc_r[1]);
        }
    }

    let va = _mm256_set1_pd(alpha);
    let vb = _mm256_set1_pd(beta);
    for (r, acc_r) in acc.iter().enumerate().take(mb) {
        let crow = c_ptr.add(r * ldc);
        let (out_lo, out_hi) = if beta == 0.0 {
            (_mm256_mul_pd(va, acc_r[0]), _mm256_mul_pd(va, acc_r[1]))
        } else {
            let (c_lo, c_hi) = if FULL {
                (_mm256_loadu_pd(crow), _mm256_loadu_pd(crow.add(hi_off)))
            } else {
                (
                    _mm256_maskload_pd(crow, mask_lo),
                    _mm256_maskload_pd(crow.add(hi_off), mask_hi),
                )
            };
            // beta == 1.0 scaling is exact (see `micro_f32_full6`).
            let (bl, bh) = if SKIP_BETA_MUL {
                (c_lo, c_hi)
            } else {
                (_mm256_mul_pd(vb, c_lo), _mm256_mul_pd(vb, c_hi))
            };
            (
                _mm256_fmadd_pd(va, acc_r[0], bl),
                _mm256_fmadd_pd(va, acc_r[1], bh),
            )
        };
        if FULL {
            _mm256_storeu_pd(crow, out_lo);
            _mm256_storeu_pd(crow.add(hi_off), out_hi);
        } else {
            _mm256_maskstore_pd(crow, mask_lo, out_lo);
            _mm256_maskstore_pd(crow.add(hi_off), mask_hi, out_hi);
        }
    }
}

/// Copy the `(pc, kc) x (jc, nc)` tile of row-major `B` (stride `ldb`)
/// into `dst` as `kc` contiguous rows of width `nc`.
///
/// The packed tile defeats L1D set-aliasing: power-of-two `ldb` strides map
/// every source row into the same handful of L1 sets, so the micro-kernel's
/// 4 A-rows + 2 B-vectors constantly evict each other (measured 20.4 GF/s
/// at 1024^3 f64); the packed tile streams sequentially and is HW-prefetch
/// friendly (34+ GF/s on the same math). Values are exact copies and the
/// sweep keeps its accumulation order, so results are bitwise identical.
#[inline]
unsafe fn pack_b_tile_f32(
    dst: *mut f32,
    src: *const f32,
    ldb: usize,
    jc: usize,
    nc: usize,
    pc: usize,
    kc: usize,
) {
    for p in 0..kc {
        core::ptr::copy_nonoverlapping(src.add((pc + p) * ldb + jc), dst.add(p * nc), nc);
    }
}

/// f64 twin of [`pack_b_tile_f32`].
#[inline]
unsafe fn pack_b_tile_f64(
    dst: *mut f64,
    src: *const f64,
    ldb: usize,
    jc: usize,
    nc: usize,
    pc: usize,
    kc: usize,
) {
    for p in 0..kc {
        core::ptr::copy_nonoverlapping(src.add((pc + p) * ldb + jc), dst.add(p * nc), nc);
    }
}

/// Pack the full `(pc, kc) x n` band of `B` into `dst` (`kc` contiguous rows
/// of width `n`), one row-copy per source row, parallelised over row chunks
/// when the band is large. The band is packed once per Kc pass and shared
/// read-only by every row task; per-task tile packing here would multiply
/// pack traffic by the row-task count.
#[inline]
unsafe fn pack_b_panel_f32(
    dst: *mut f32,
    src: *const f32,
    ldb: usize,
    n: usize,
    pc: usize,
    kc: usize,
    ldp: usize,
) {
    // Rows per pack task: enough row-copies (~`n` elements each) that scope
    // dispatch (~µs) stays negligible.
    const ROWS_PER_TASK: usize = 32;
    let dst_w = SendMutPtr(dst);
    let src_w = SendConstPtr(src);
    if kc.saturating_mul(n) >= PACK_PAR_MIN_ELS {
        // SAFETY: tasks write disjoint `dst` row ranges and read disjoint
        // `src` rows; the scope joins before either pointer is reused.
        rayon::scope(|s| {
            let mut p0 = 0;
            while p0 < kc {
                let p1 = (p0 + ROWS_PER_TASK).min(kc);
                s.spawn(move |_| {
                    let dst: SendMutPtr<f32> = dst_w;
                    let src: SendConstPtr<f32> = src_w;
                    for p in p0..p1 {
                        core::ptr::copy_nonoverlapping(
                            src.0.add((pc + p) * ldb),
                            dst.0.add(p * ldp),
                            n,
                        );
                    }
                });
                p0 = p1;
            }
        });
    } else {
        for p in 0..kc {
            core::ptr::copy_nonoverlapping(src.add((pc + p) * ldb), dst.add(p * ldp), n);
        }
    }
}

/// f64 twin of [`pack_b_panel_f32`].
#[inline]
unsafe fn pack_b_panel_f64(
    dst: *mut f64,
    src: *const f64,
    ldb: usize,
    n: usize,
    pc: usize,
    kc: usize,
    ldp: usize,
) {
    // Rows per pack task: enough row-copies (~`n` elements each) that scope
    // dispatch (~µs) stays negligible.
    const ROWS_PER_TASK: usize = 32;
    let dst_w = SendMutPtr(dst);
    let src_w = SendConstPtr(src);
    if kc.saturating_mul(n) >= PACK_PAR_MIN_ELS {
        // SAFETY: tasks write disjoint `dst` row ranges and read disjoint
        // `src` rows; the scope joins before either pointer is reused.
        rayon::scope(|s| {
            let mut p0 = 0;
            while p0 < kc {
                let p1 = (p0 + ROWS_PER_TASK).min(kc);
                s.spawn(move |_| {
                    let dst: SendMutPtr<f64> = dst_w;
                    let src: SendConstPtr<f64> = src_w;
                    for p in p0..p1 {
                        core::ptr::copy_nonoverlapping(
                            src.0.add((pc + p) * ldb),
                            dst.0.add(p * ldp),
                            n,
                        );
                    }
                });
                p0 = p1;
            }
        });
    } else {
        for p in 0..kc {
            core::ptr::copy_nonoverlapping(src.add((pc + p) * ldb), dst.add(p * ldp), n);
        }
    }
}

/// Minimum band elements before the panel pack parallelises over row chunks.
/// Below this a serial copy (~tens of µs) is cheaper than rayon dispatch.
const PACK_PAR_MIN_ELS: usize = 65_536;

/// 8x8-blocked transpose core shared by the panel packs below: the naive
/// j-outer/q-inner order strides the stores by `ldp` over the whole
/// `kc × width` panel and thrashes L1D at fat `k` (thin-`m` `Trans`-B
/// through the M-split panel pack ran ~0.5x before blocking). Same
/// elements, different visit order — bitwise identical output.
#[inline]
unsafe fn pack_trans_chunk_f32(
    dst: *mut f32,
    src: *const f32,
    ldb: usize,
    ldp: usize,
    j0: usize,
    j1: usize,
    pc: usize,
    kc: usize,
) {
    const QR: usize = 8;
    const JR: usize = 8;
    let mut j = j0;
    while j + JR <= j1 {
        let mut q = 0;
        while q + QR <= kc {
            for jj in 0..JR {
                for qq in 0..QR {
                    *dst.add((q + qq) * ldp + j + jj) = *src.add((j + jj) * ldb + pc + q + qq);
                }
            }
            q += QR;
        }
        for qq in q..kc {
            for jj in 0..JR {
                *dst.add(qq * ldp + j + jj) = *src.add((j + jj) * ldb + pc + qq);
            }
        }
        j += JR;
    }
    for j in j..j1 {
        for q in 0..kc {
            *dst.add(q * ldp + j) = *src.add(j * ldb + pc + q);
        }
    }
}

/// f64 twin of [`pack_trans_chunk_f32`].
#[inline]
unsafe fn pack_trans_chunk_f64(
    dst: *mut f64,
    src: *const f64,
    ldb: usize,
    ldp: usize,
    j0: usize,
    j1: usize,
    pc: usize,
    kc: usize,
) {
    const QR: usize = 8;
    const JR: usize = 8;
    let mut j = j0;
    while j + JR <= j1 {
        let mut q = 0;
        while q + QR <= kc {
            for jj in 0..JR {
                for qq in 0..QR {
                    *dst.add((q + qq) * ldp + j + jj) = *src.add((j + jj) * ldb + pc + q + qq);
                }
            }
            q += QR;
        }
        for qq in q..kc {
            for jj in 0..JR {
                *dst.add(qq * ldp + j + jj) = *src.add((j + jj) * ldb + pc + qq);
            }
        }
        j += JR;
    }
    for j in j..j1 {
        for q in 0..kc {
            *dst.add(q * ldp + j) = *src.add(j * ldb + pc + q);
        }
    }
}

/// Transposed-gather twin of [`pack_b_tile_f32`]: packs the `(pc, kc)` row
/// block of `op(B)` (`k × n`) over columns `[jc, jc + nc)` from a `Trans`
/// source (`n × k` row-major, stride `ldb`) into the same row-major tile
/// layout the sweep consumes (element `(q, j)` gathers source row `jc + j`
/// at offset `pc + q`). Delegates to the vectorised 8×8 AVX2
/// [`pack_trans_f32`] with offset pointers (exact remap: `dst[q * nc +
/// j] = src[(jc + j) * ldb + pc + q]`); the scalar 8×8-blocked gather
/// this replaces cost ~20% of a thin-`m` band GEMM (32×1024×256:
/// 0.232ms NT vs 0.187 NN with the identical sweep). Exact copies
/// either way, so numerics never change. All callers already run under
/// `avx2,fma`. Caveat (see the f64 twin): the 8×8 core re-touches source
/// lines across `r` slices, so a 4 KiB source row pitch (`ldb` a multiple
/// of 1024) aliases one L1D set and may thrash — verify any `k ≥ 1024`
/// trans-B shape after touching this.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn pack_b_tile_f32_trans(
    dst: *mut f32,
    src: *const f32,
    ldb: usize,
    jc: usize,
    nc: usize,
    pc: usize,
    kc: usize,
) {
    // SAFETY: per the caller contract (`jc + nc <= n`, `pc + kc <= k`,
    // `dst` holds `kc * nc`); `pack_trans_f32` reads/writes exactly the
    // remapped ranges.
    unsafe {
        pack_trans_f32(src.add(jc * ldb + pc), ldb, kc, nc, dst);
    }
}

/// f64 twin of [`pack_b_tile_f32_trans`]. Cache-blocked scalar 8x8 (kept
/// scalar deliberately): delegating to the vectorised 8×4 AVX2
/// [`pack_trans_f64`] regressed thin-`m` bands hard (f64 32×2048×512:
/// 1.10x → 1.94x vs OpenBLAS) — its r-outer/c-inner order re-touches
/// every source line across `r` slices, and at `ldb = k = 512` (4 KiB
/// row pitch) all 8 concurrent lines alias one L1D set and thrash, while
/// the scalar 8×8 streams each full line once. Do not re-vectorise
/// without a line-once visit order.
#[inline]
unsafe fn pack_b_tile_f64_trans(
    dst: *mut f64,
    src: *const f64,
    ldb: usize,
    jc: usize,
    nc: usize,
    pc: usize,
    kc: usize,
) {
    const QR: usize = 8;
    const JR: usize = 8;
    let mut j = 0;
    while j + JR <= nc {
        let mut q = 0;
        while q + QR <= kc {
            for jj in 0..JR {
                for qq in 0..QR {
                    *dst.add((q + qq) * nc + j + jj) = *src.add((jc + j + jj) * ldb + pc + q + qq);
                }
            }
            q += QR;
        }
        for qq in q..kc {
            for jj in 0..JR {
                *dst.add(qq * nc + j + jj) = *src.add((jc + j + jj) * ldb + pc + qq);
            }
        }
        j += JR;
    }
    for j in j..nc {
        for q in 0..kc {
            *dst.add(q * nc + j) = *src.add((jc + j) * ldb + pc + q);
        }
    }
}

/// Transposed-gather twin of [`pack_b_panel_f32`]: packs the `(pc, kc)` row
/// block of `op(B)` over all `n` columns from a `Trans` source into the
/// same `kc × n` panel layout, parallelised over column chunks when large.
/// `panel[q * n + j] = src[j * ldb + pc + q]`.
#[inline]
unsafe fn pack_b_panel_f32_trans(
    dst: *mut f32,
    src: *const f32,
    ldb: usize,
    n: usize,
    pc: usize,
    kc: usize,
    ldp: usize,
) {
    const COLS_PER_TASK: usize = 32;
    let dst_w = SendMutPtr(dst);
    let src_w = SendConstPtr(src);
    if kc.saturating_mul(n) >= PACK_PAR_MIN_ELS {
        // SAFETY: tasks write disjoint `dst` column ranges; the scope joins
        // before either pointer is reused.
        rayon::scope(|s| {
            let mut j0 = 0;
            while j0 < n {
                let j1 = (j0 + COLS_PER_TASK).min(n);
                s.spawn(move |_| {
                    let dst: SendMutPtr<f32> = dst_w;
                    let src: SendConstPtr<f32> = src_w;
                    pack_trans_chunk_f32(dst.0, src.0, ldb, ldp, j0, j1, pc, kc);
                });
                j0 = j1;
            }
        });
    } else {
        pack_trans_chunk_f32(dst, src, ldb, ldp, 0, n, pc, kc);
    }
}

/// f64 twin of [`pack_b_panel_f32_trans`].
#[inline]
unsafe fn pack_b_panel_f64_trans(
    dst: *mut f64,
    src: *const f64,
    ldb: usize,
    n: usize,
    pc: usize,
    kc: usize,
    ldp: usize,
) {
    const COLS_PER_TASK: usize = 32;
    let dst_w = SendMutPtr(dst);
    let src_w = SendConstPtr(src);
    if kc.saturating_mul(n) >= PACK_PAR_MIN_ELS {
        // SAFETY: tasks write disjoint `dst` column ranges; the scope joins
        // before either pointer is reused.
        rayon::scope(|s| {
            let mut j0 = 0;
            while j0 < n {
                let j1 = (j0 + COLS_PER_TASK).min(n);
                s.spawn(move |_| {
                    let dst: SendMutPtr<f64> = dst_w;
                    let src: SendConstPtr<f64> = src_w;
                    pack_trans_chunk_f64(dst.0, src.0, ldb, ldp, j0, j1, pc, kc);
                });
                j0 = j1;
            }
        });
    } else {
        pack_trans_chunk_f64(dst, src, ldb, ldp, 0, n, pc, kc);
    }
}

/// Maximum packed-panel elements for the M-split shared band. Beyond this
/// the transient panel rivals the source matrix itself, so the sweep reads
/// `B` in place (correct, just L1-unfriendly) instead of doubling memory.
const PANEL_MAX_ELS: usize = 67_108_864;

/// Padded row stride for the M-split shared `B` panel. When `n` elements
/// span an exact multiple of 2 KiB, consecutive panel rows cycle just 2
/// L1 sets (32-line pitch over 64 sets) and the sweep stalls on set
/// conflicts (measured: forced-M d1kn256 runs 2.77ms vs 2.13ms at
/// n=260 with more flops; the 4 KiB case loses square-1024 f32/f64 by
/// 16-75% while non-power-of-2 strides up to 1088 win by 14-28%). A
/// 16-element pad breaks the alignment (272x8 = 2176 B = 34-line pitch
/// cycles 32 sets); any other stride returns `n`, so unaffected shapes
/// pay nothing. Layout-only: values identical, bitwise-neutral.
#[inline]
fn panel_stride(n: usize, elsize: usize) -> usize {
    if n.saturating_mul(elsize).is_multiple_of(2048) {
        n + 16
    } else {
        n
    }
}

/// Row-band sweep for one `(jc, pc)` tile: rows `[r0, r1)` of C.
/// Shared by the serial loop and each rayon task; per-element accumulation
/// order is identical in both, so dispatch never changes numerics.
#[target_feature(enable = "avx2,fma")]
unsafe fn sweep_f32(
    r0: usize,
    r1: usize,
    // Full matrix height, kept for the band-within-matrix contract (the
    // tail cap below keys off `r1`, not `m`).
    _m: usize,
    jc: usize,
    nc: usize,
    pc: usize,
    kc: usize,
    alpha: f32,
    a: *const f32,
    a_row_stride: usize,
    a_col_stride: usize,
    b: *const f32,
    ldb: usize,
    // See `sweep_f64`: packed tiles/bands rebase the `B` origin.
    b_pc: usize,
    b_jc: usize,
    beta_eff: f32,
    c: *mut f32,
    ldc: usize,
) {
    // Row offsets for the 6-wide kernel (constant per sweep; passed
    // by pointer so LSR cannot fold them — see the kernel docs).
    let a_offs = [
        a_row_stride,
        a_row_stride * 2,
        a_row_stride * 3,
        a_row_stride * 4,
        a_row_stride * 5,
    ];
    let mut i0 = r0;
    while i0 < r1 {
        // 6-row full tiles feed the 12-chain kernel below; any remainder
        // (< 6 rows) falls through to the 4-wide kernels. The tail cap
        // keys off the BAND remainder (`r1 - i0`): see the `sweep_f64`
        // overshoot note — 6-stepping breaks 4-aligned band invariants.
        if r1 - i0 >= F32_MR6 {
            let mut j0 = jc;
            while j0 < jc + nc {
                let nb = (jc + nc - j0).min(F32_NR);
                let a_ptr = a.add(i0 * a_row_stride + pc * a_col_stride);
                let b_ptr = b.add(b_pc * ldb + (j0 - b_jc));
                let c_ptr = c.add(i0 * ldc + j0);
                if nb == F32_NR {
                    // Const-dispatched beta == 1.0 multiply skip: exact,
                    // bitwise identical; each monomorphization branch-free.
                    if beta_eff == 1.0 {
                        micro_f32_full6::<true>(
                            kc,
                            alpha,
                            a_ptr,
                            a_offs.as_ptr(),
                            a_col_stride,
                            b_ptr,
                            ldb,
                            beta_eff,
                            c_ptr,
                            ldc,
                        );
                    } else {
                        micro_f32_full6::<false>(
                            kc,
                            alpha,
                            a_ptr,
                            a_offs.as_ptr(),
                            a_col_stride,
                            b_ptr,
                            ldb,
                            beta_eff,
                            c_ptr,
                            ldc,
                        );
                    }
                } else if beta_eff == 1.0 {
                    micro_f32::<false, true>(
                        F32_MR6,
                        nb,
                        kc,
                        alpha,
                        a_ptr,
                        a_row_stride,
                        a_col_stride,
                        b_ptr,
                        ldb,
                        beta_eff,
                        c_ptr,
                        ldc,
                    );
                } else {
                    micro_f32::<false, false>(
                        F32_MR6,
                        nb,
                        kc,
                        alpha,
                        a_ptr,
                        a_row_stride,
                        a_col_stride,
                        b_ptr,
                        ldb,
                        beta_eff,
                        c_ptr,
                        ldc,
                    );
                }
                j0 += F32_NR;
            }
            i0 += F32_MR6;
            continue;
        }
        let mb = (r1 - i0).min(MR);
        let mut j0 = jc;
        while j0 < jc + nc {
            let nb = (jc + nc - j0).min(F32_NR);
            let a_ptr = a.add(i0 * a_row_stride + pc * a_col_stride);
            let b_ptr = b.add(b_pc * ldb + (j0 - b_jc));
            let c_ptr = c.add(i0 * ldc + j0);
            if mb == MR && nb == F32_NR {
                if beta_eff == 1.0 {
                    micro_f32_full::<true, true>(
                        nb,
                        kc,
                        alpha,
                        a_ptr,
                        a_row_stride,
                        a_col_stride,
                        b_ptr,
                        ldb,
                        beta_eff,
                        c_ptr,
                        ldc,
                    );
                } else {
                    micro_f32_full::<true, false>(
                        nb,
                        kc,
                        alpha,
                        a_ptr,
                        a_row_stride,
                        a_col_stride,
                        b_ptr,
                        ldb,
                        beta_eff,
                        c_ptr,
                        ldc,
                    );
                }
            } else if nb == F32_NR {
                if beta_eff == 1.0 {
                    micro_f32::<true, true>(
                        mb,
                        nb,
                        kc,
                        alpha,
                        a_ptr,
                        a_row_stride,
                        a_col_stride,
                        b_ptr,
                        ldb,
                        beta_eff,
                        c_ptr,
                        ldc,
                    );
                } else {
                    micro_f32::<true, false>(
                        mb,
                        nb,
                        kc,
                        alpha,
                        a_ptr,
                        a_row_stride,
                        a_col_stride,
                        b_ptr,
                        ldb,
                        beta_eff,
                        c_ptr,
                        ldc,
                    );
                }
            } else if mb == MR {
                if beta_eff == 1.0 {
                    micro_f32_full::<false, true>(
                        nb,
                        kc,
                        alpha,
                        a_ptr,
                        a_row_stride,
                        a_col_stride,
                        b_ptr,
                        ldb,
                        beta_eff,
                        c_ptr,
                        ldc,
                    );
                } else {
                    micro_f32_full::<false, false>(
                        nb,
                        kc,
                        alpha,
                        a_ptr,
                        a_row_stride,
                        a_col_stride,
                        b_ptr,
                        ldb,
                        beta_eff,
                        c_ptr,
                        ldc,
                    );
                }
            } else if beta_eff == 1.0 {
                micro_f32::<false, true>(
                    mb,
                    nb,
                    kc,
                    alpha,
                    a_ptr,
                    a_row_stride,
                    a_col_stride,
                    b_ptr,
                    ldb,
                    beta_eff,
                    c_ptr,
                    ldc,
                );
            } else {
                micro_f32::<false, false>(
                    mb,
                    nb,
                    kc,
                    alpha,
                    a_ptr,
                    a_row_stride,
                    a_col_stride,
                    b_ptr,
                    ldb,
                    beta_eff,
                    c_ptr,
                    ldc,
                );
            }
            j0 += F32_NR;
        }
        i0 += MR;
    }
}

/// Full-tile row band for [`sweep_f64`]: `ntiles` contiguous `MR × NR`
/// tiles sharing one call frame. The `#[inline(never)]` micro-kernel costs
/// ~11cy per call in frame setup/dispatch; at thin `k` that is ~30% of the
/// solve (amortized to noise at large `k`). This fused frame now serves
/// only sub-6-row regions (see the gate in [`sweep_f64`]): everywhere 6
/// rows fit, the 12-chain 6-row kernel wins outright. Fusing a row of
/// tiles into one frame removes it;
/// the tile body is the exact `micro_f64_full::<true, false>` sequence
/// (bitwise identical), with the alpha/beta broadcasts hoisted across
/// tiles. `a_ptr` is the shared 4-row `A` block start (re-read per tile
/// from L1); `b_ptr`/`c_ptr` advance `NR` per tile. Caller guarantees all
/// tiles full and `ntiles >= 1`.
#[target_feature(enable = "avx2,fma")]
#[inline(never)]
unsafe fn rowband_f64_full<const SKIP_BETA_MUL: bool>(
    ntiles: usize,
    kc: usize,
    alpha: f64,
    a_ptr: *const f64,
    a_row_stride: usize,
    a_col_stride: usize,
    b_ptr: *const f64,
    ldb: usize,
    beta: f64,
    c_ptr: *mut f64,
    ldc: usize,
) {
    use core::arch::x86_64::*;
    // SAFETY: same contract as `micro_f64_full::<true, false>` per tile; the
    // caller guarantees `ntiles` full tiles of addressable rows.
    unsafe {
        let va = _mm256_set1_pd(alpha);
        let vb = _mm256_set1_pd(beta);
        let mut bp = b_ptr;
        let mut cp = c_ptr;
        let mut t = 0;
        while t < ntiles {
            let mut acc00 = _mm256_setzero_pd();
            let mut acc01 = _mm256_setzero_pd();
            let mut acc10 = _mm256_setzero_pd();
            let mut acc11 = _mm256_setzero_pd();
            let mut acc20 = _mm256_setzero_pd();
            let mut acc21 = _mm256_setzero_pd();
            let mut acc30 = _mm256_setzero_pd();
            let mut acc31 = _mm256_setzero_pd();
            let mut ap0 = a_ptr;
            let mut ap1 = a_ptr.add(a_row_stride);
            let mut ap2 = a_ptr.add(a_row_stride * 2);
            let mut ap3 = a_ptr.add(a_row_stride * 3);
            let mut bq = bp;
            let mut p = 0;
            while p < kc {
                let bv_lo = _mm256_loadu_pd(bq);
                let bv_hi = _mm256_loadu_pd(bq.add(4));
                let a0 = _mm256_set1_pd(*ap0);
                ap0 = ap0.add(a_col_stride);
                let a1 = _mm256_set1_pd(*ap1);
                ap1 = ap1.add(a_col_stride);
                let a2 = _mm256_set1_pd(*ap2);
                ap2 = ap2.add(a_col_stride);
                let a3 = _mm256_set1_pd(*ap3);
                ap3 = ap3.add(a_col_stride);
                acc00 = _mm256_fmadd_pd(a0, bv_lo, acc00);
                acc01 = _mm256_fmadd_pd(a0, bv_hi, acc01);
                acc10 = _mm256_fmadd_pd(a1, bv_lo, acc10);
                acc11 = _mm256_fmadd_pd(a1, bv_hi, acc11);
                acc20 = _mm256_fmadd_pd(a2, bv_lo, acc20);
                acc21 = _mm256_fmadd_pd(a2, bv_hi, acc21);
                acc30 = _mm256_fmadd_pd(a3, bv_lo, acc30);
                acc31 = _mm256_fmadd_pd(a3, bv_hi, acc31);
                bq = bq.add(ldb);
                p += 1;
            }
            let c0 = cp;
            let c1 = cp.add(ldc);
            let c2 = cp.add(ldc * 2);
            let c3 = cp.add(ldc * 3);
            // (row, lo, hi) triples; same beta == 0 fast path as the micro.
            let rows = [
                (c0, acc00, acc01),
                (c1, acc10, acc11),
                (c2, acc20, acc21),
                (c3, acc30, acc31),
            ];
            for &(crow, al, ah) in rows.iter() {
                let (out_lo, out_hi) = if beta == 0.0 {
                    (_mm256_mul_pd(va, al), _mm256_mul_pd(va, ah))
                } else {
                    let c_lo = _mm256_loadu_pd(crow);
                    let c_hi = _mm256_loadu_pd(crow.add(4));
                    // beta == 1.0 scaling is exact (see `micro_f32_full6`).
                    let (bl, bh) = if SKIP_BETA_MUL {
                        (c_lo, c_hi)
                    } else {
                        (_mm256_mul_pd(vb, c_lo), _mm256_mul_pd(vb, c_hi))
                    };
                    (_mm256_fmadd_pd(va, al, bl), _mm256_fmadd_pd(va, ah, bh))
                };
                _mm256_storeu_pd(crow, out_lo);
                _mm256_storeu_pd(crow.add(4), out_hi);
            }
            bp = bp.add(F64_NR);
            cp = cp.add(F64_NR);
            t += 1;
        }
    }
}

/// f64 twin of [`sweep_f32`].
#[target_feature(enable = "avx2,fma")]
unsafe fn sweep_f64(
    r0: usize,
    r1: usize,
    // Full matrix height, kept for the band-within-matrix contract (the
    // tail cap below keys off `r1`, not `m`).
    _m: usize,
    jc: usize,
    nc: usize,
    pc: usize,
    kc: usize,
    alpha: f64,
    a: *const f64,
    a_row_stride: usize,
    a_col_stride: usize,
    b: *const f64,
    ldb: usize,
    // `B` operand addressing: element `(p, j0)` is at
    // `b + b_pc * ldb + (j0 - b_jc)`. In-place reads use `b_pc = pc`,
    // `b_jc = 0`; packed tiles/bands rebase the origin (`b_pc = 0` with
    // `b_jc = jc` for a tile, `b_jc = 0` for a full-width band).
    b_pc: usize,
    b_jc: usize,
    beta_eff: f64,
    c: *mut f64,
    ldc: usize,
) {
    // Rowband fast path (see `rowband_f64_full`): the whole region splits
    // into full `MR × NR` tiles — one fused call per 4-row band instead of
    // one micro call per tile. Kept only for sub-6-row regions at small
    // pass-`k` (`kc <= 64`): the 12-chain 6-row kernel in the general loop
    // below beats the fused 4-wide frame everywhere 6 rows are available
    // (thin-`k` SYRK 0.3x -> 1.2x vs OpenBLAS), while the fused frame still
    // wins where no 6-row tile fits (4x256x32: 22.8 vs 17.0 GF/s).
    //
    // Process-once gate read: `sweep_f64` runs tens of thousands of times
    // per large factorize, and every `std::env::var_os` takes the global
    // environment lock plus an allocation (~56ns uncontended, worse with
    // 16 workers hammering it — measurable milliseconds per dpotrf-2048).
    // No test mutates this variable (unlike the per-call M_SPLIT-style
    // gates), so caching it once is behavior-identical everywhere.
    static ROWBAND_ALLOWED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let rowband_allowed =
        *ROWBAND_ALLOWED.get_or_init(|| std::env::var_os("SONIUM_GEMM_NOROWBAND").is_none());
    if rowband_allowed
        && kc <= 64
        && r1 - r0 < F64_MR6
        && (r1 - r0).is_multiple_of(MR)
        && nc.is_multiple_of(F64_NR)
    {
        // SAFETY: bounds per the caller contract; the modulo guards make
        // every band/tile full.
        unsafe {
            let mut i0 = r0;
            while i0 < r1 {
                let a_ptr = a.add(i0 * a_row_stride + pc * a_col_stride);
                let b_ptr = b.add(b_pc * ldb + (jc - b_jc));
                let c_ptr = c.add(i0 * ldc + jc);
                if beta_eff == 1.0 {
                    rowband_f64_full::<true>(
                        nc / F64_NR,
                        kc,
                        alpha,
                        a_ptr,
                        a_row_stride,
                        a_col_stride,
                        b_ptr,
                        ldb,
                        beta_eff,
                        c_ptr,
                        ldc,
                    );
                } else {
                    rowband_f64_full::<false>(
                        nc / F64_NR,
                        kc,
                        alpha,
                        a_ptr,
                        a_row_stride,
                        a_col_stride,
                        b_ptr,
                        ldb,
                        beta_eff,
                        c_ptr,
                        ldc,
                    );
                }
                i0 += MR;
            }
        }
        return;
    }
    let mut i0 = r0;
    while i0 < r1 {
        // 6-row full tiles feed the 12-chain kernel below; any remainder
        // (< 6 rows) falls through to the 4-wide kernels, whose
        // accumulator array is sized for `mb <= F64_MR6`.
        if r1 - i0 >= F64_MR6 {
            let mut j0 = jc;
            while j0 < jc + nc {
                let nb = (jc + nc - j0).min(F64_NR);
                let a_ptr = a.add(i0 * a_row_stride + pc * a_col_stride);
                let b_ptr = b.add(b_pc * ldb + (j0 - b_jc));
                let c_ptr = c.add(i0 * ldc + j0);
                if nb == F64_NR {
                    // Const-dispatched beta == 1.0 multiply skip: exact,
                    // bitwise identical; each monomorphization branch-free.
                    if beta_eff == 1.0 {
                        micro_f64_full6::<true>(
                            kc,
                            alpha,
                            a_ptr,
                            a_row_stride,
                            a_col_stride,
                            b_ptr,
                            ldb,
                            beta_eff,
                            c_ptr,
                            ldc,
                        );
                    } else {
                        micro_f64_full6::<false>(
                            kc,
                            alpha,
                            a_ptr,
                            a_row_stride,
                            a_col_stride,
                            b_ptr,
                            ldb,
                            beta_eff,
                            c_ptr,
                            ldc,
                        );
                    }
                } else if beta_eff == 1.0 {
                    micro_f64::<false, true>(
                        F64_MR6,
                        nb,
                        kc,
                        alpha,
                        a_ptr,
                        a_row_stride,
                        a_col_stride,
                        b_ptr,
                        ldb,
                        beta_eff,
                        c_ptr,
                        ldc,
                    );
                } else {
                    micro_f64::<false, false>(
                        F64_MR6,
                        nb,
                        kc,
                        alpha,
                        a_ptr,
                        a_row_stride,
                        a_col_stride,
                        b_ptr,
                        ldb,
                        beta_eff,
                        c_ptr,
                        ldc,
                    );
                }
                j0 += F64_NR;
            }
            i0 += F64_MR6;
            continue;
        }
        // Cap by the BAND remainder (`r1 - i0`), not the matrix height:
        // 6-stepping above breaks the 4-aligned band invariant the old
        // `(m - i0)` cap relied on, and overshooting `r1` would trample
        // the neighbouring task's rows.
        let mb = (r1 - i0).min(MR);
        let mut j0 = jc;
        while j0 < jc + nc {
            let nb = (jc + nc - j0).min(F64_NR);
            let a_ptr = a.add(i0 * a_row_stride + pc * a_col_stride);
            let b_ptr = b.add(b_pc * ldb + (j0 - b_jc));
            let c_ptr = c.add(i0 * ldc + j0);
            if mb == MR && nb == F64_NR {
                if beta_eff == 1.0 {
                    micro_f64_full::<true, true>(
                        nb,
                        kc,
                        alpha,
                        a_ptr,
                        a_row_stride,
                        a_col_stride,
                        b_ptr,
                        ldb,
                        beta_eff,
                        c_ptr,
                        ldc,
                    );
                } else {
                    micro_f64_full::<true, false>(
                        nb,
                        kc,
                        alpha,
                        a_ptr,
                        a_row_stride,
                        a_col_stride,
                        b_ptr,
                        ldb,
                        beta_eff,
                        c_ptr,
                        ldc,
                    );
                }
            } else if nb == F64_NR {
                if beta_eff == 1.0 {
                    micro_f64::<true, true>(
                        mb,
                        nb,
                        kc,
                        alpha,
                        a_ptr,
                        a_row_stride,
                        a_col_stride,
                        b_ptr,
                        ldb,
                        beta_eff,
                        c_ptr,
                        ldc,
                    );
                } else {
                    micro_f64::<true, false>(
                        mb,
                        nb,
                        kc,
                        alpha,
                        a_ptr,
                        a_row_stride,
                        a_col_stride,
                        b_ptr,
                        ldb,
                        beta_eff,
                        c_ptr,
                        ldc,
                    );
                }
            } else if mb == MR {
                if beta_eff == 1.0 {
                    micro_f64_full::<false, true>(
                        nb,
                        kc,
                        alpha,
                        a_ptr,
                        a_row_stride,
                        a_col_stride,
                        b_ptr,
                        ldb,
                        beta_eff,
                        c_ptr,
                        ldc,
                    );
                } else {
                    micro_f64_full::<false, false>(
                        nb,
                        kc,
                        alpha,
                        a_ptr,
                        a_row_stride,
                        a_col_stride,
                        b_ptr,
                        ldb,
                        beta_eff,
                        c_ptr,
                        ldc,
                    );
                }
            } else if beta_eff == 1.0 {
                micro_f64::<false, true>(
                    mb,
                    nb,
                    kc,
                    alpha,
                    a_ptr,
                    a_row_stride,
                    a_col_stride,
                    b_ptr,
                    ldb,
                    beta_eff,
                    c_ptr,
                    ldc,
                );
            } else {
                micro_f64::<false, false>(
                    mb,
                    nb,
                    kc,
                    alpha,
                    a_ptr,
                    a_row_stride,
                    a_col_stride,
                    b_ptr,
                    ldb,
                    beta_eff,
                    c_ptr,
                    ldc,
                );
            }
            j0 += F64_NR;
        }
        i0 += MR;
    }
}

/// `C := beta * C` over the `m × n` sub-matrix (stride `ldc`). beta == 1 is
/// a no-op; beta == 0 zeroes so uninitialised/NaN C is not propagated.
unsafe fn scale_c_f32(m: usize, n: usize, beta: f32, c: *mut f32, ldc: usize) {
    if beta == 1.0 {
        return;
    }
    for i in 0..m {
        for j in 0..n {
            let cij = c.add(i * ldc + j);
            *cij = if beta == 0.0 { 0.0 } else { beta * *cij };
        }
    }
}

unsafe fn scale_c_f64(m: usize, n: usize, beta: f64, c: *mut f64, ldc: usize) {
    if beta == 1.0 {
        return;
    }
    for i in 0..m {
        for j in 0..n {
            let cij = c.add(i * ldc + j);
            *cij = if beta == 0.0 { 0.0 } else { beta * *cij };
        }
    }
}

/// `C := alpha * op(A) * op(B) + beta * C` (f32, row-major).
/// Same contract as `sme2::gemm::sgemm`: bounds are pre-checked by the
/// caller; `trans_a` selects the stored-A layout (k × m when true) and
/// `trans_b` the stored-B layout (n × k when true, in which case the sweep
/// consumes transposed-gather tiles/panels — no separate pre-pack).
#[target_feature(enable = "avx2,fma")]
/// N-split task spawner (f32): tasks own column bands `[u0, u1)` across
/// all rows, with Kc passes fused per task. Extracted so the Trans-A
/// shared-pack arm and the legacy strided arm share one body — they
/// differ only in the `A` base/strides (`packed, k, 1` vs `a, 1, lda`).
/// Same values, same sweep order: bitwise identical either way.
#[allow(clippy::too_many_arguments)]
unsafe fn spawn_n_tasks_f32(
    s: &rayon::Scope<'_>,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a_ptr: *const f32,
    a_row_stride: usize,
    a_col_stride: usize,
    b_ptr: *const f32,
    ldb: usize,
    trans_b: bool,
    beta: f32,
    c_ptr: *mut f32,
    ldc: usize,
    cpt: usize,
) {
    // SAFETY: each task sweeps columns `[u0, u1)` across all rows —
    // disjoint C bands, read-only A/B.
    let a_send = SendConstPtr(a_ptr);
    let b_send = SendConstPtr(b_ptr);
    let c_send = SendMutPtr(c_ptr);
    let mut u0 = 0;
    while u0 < n {
        let u1 = (u0 + cpt).min(n);
        s.spawn(move |_| {
            let a_send: SendConstPtr<f32> = a_send;
            let b_send: SendConstPtr<f32> = b_send;
            let c_send: SendMutPtr<f32> = c_send;
            // Thread-local packed-B tile buffer (see the f64
            // N-split: per-task `Vec` mmaps serialise the pool).
            // Wide-Kc passes at k >= 768 (see `f32_kc_nsplit`).
            let kc_step = f32_kc_nsplit(k);
            super::with_buf(&super::F32_B_PACK, kc_step * F32_NC, |tile| {
                // Kc passes fused into the task (see above): one
                // spawn covers all passes, no barrier between.
                let mut pc = 0;
                while pc < k {
                    let kc = (k - pc).min(kc_step);
                    let beta_eff = if pc == 0 { beta } else { 1.0 };
                    let mut jc = u0;
                    while jc < u1 {
                        let nc = (jc + F32_NC).min(u1) - jc;
                        // SAFETY: pack overwrites `kc * nc`
                        // elements before the sweep reads them
                        // (transposed gather for a `Trans`
                        // source).
                        if trans_b {
                            pack_b_tile_f32_trans(tile.as_mut_ptr(), b_send.0, ldb, jc, nc, pc, kc);
                        } else {
                            pack_b_tile_f32(tile.as_mut_ptr(), b_send.0, ldb, jc, nc, pc, kc);
                        }
                        sweep_f32(
                            0,
                            m,
                            m,
                            jc,
                            nc,
                            pc,
                            kc,
                            alpha,
                            a_send.0,
                            a_row_stride,
                            a_col_stride,
                            tile.as_ptr(),
                            nc,
                            0,
                            jc,
                            beta_eff,
                            c_send.0,
                            ldc,
                        );
                        jc += F32_NC;
                    }
                    pc += kc;
                }
            });
        });
        u0 = u1;
    }
}

pub unsafe fn sgemm(
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
    trans_a: bool,
    trans_b: bool,
) {
    if m == 0 || n == 0 {
        return;
    }
    if alpha == 0.0 || k == 0 {
        scale_c_f32(m, n, beta, c.as_mut_ptr(), ldc);
        return;
    }
    // Serial-only pre-pack (f32 twin of the f64 gate in `dgemm`):
    // transposed-A micro-panels broadcast from lda-strided storage (one
    // cache line per element); packing `A^T` once into a contiguous
    // buffer lets the NoTrans sweep stream it. Single recursion,
    // terminates (inner call is NoTrans). Results identical either way
    // (same values, same sweep order).
    if trans_a && m * k >= DGEMM_TRANSPOSE_PACK_MIN && rayon::current_num_threads() == 1 {
        with_packed_a_trans_f32(m, k, a.as_ptr(), lda, |packed| {
            sgemm(
                m, n, k, alpha, packed, k, b, ldb, beta, c, ldc, false, trans_b,
            );
        });
        return;
    }
    // NOTE (2026-09-27): an MT pack-once-shared `A^T` gate mirroring
    // the f64 one was trialled here and reverted — quiet-box 32T A/B/A
    // measured gather at parity-or-better on every trans-A shape (f32
    // 1024³T: off 1.15x vs on 1.20x/1.41x; thin-m ties). The shared
    // pack pays a full extra DRAM pass over `A` while the per-band
    // strided gathers hit L3-resident lines ~free, and the gate only
    // fires when `A` fits in L3 — exactly where packing cannot win.
    // Do not retry without addressing the extra-pass traffic (e.g.
    // in-kernel micropanel packs like the vendor, or an uninitialized
    // pack buffer to halve write traffic).
    let (a_row_stride, a_col_stride) = if trans_a { (1, lda) } else { (lda, 1) };
    // 2D-split path: tasks own disjoint C row-band × column-band rectangles
    // (see `use_parallel_2d`). Single-pass thin-`k` only: each task packs
    // its `B` tile once and sweeps its rows once, with no per-pass re-pack.
    // Row-splitting cuts the N-split's per-band `A` re-reads while the
    // L1-capped band width keeps the `B` stream L1-resident. Per-element
    // accumulation order matches the serial nest exactly.
    {
        let macs = m.saturating_mul(n).saturating_mul(k);
        let workers = super::workers_for_macs(macs);
        if !force_m_split()
            && !use_m_split_rank_k(m, n, k, F32_KC)
            && use_parallel_2d(m, n, k, F32_NR, F32_KC, 8192, workers)
        {
            let (rpt, cpt) = grid_2d(m, n, k, F32_NR, 8192, workers);
            // SAFETY: each task sweeps rows `[t0, t1)` × columns `[u0, u1)` —
            // disjoint C rectangles, read-only A/B.
            let a_send = SendConstPtr(a.as_ptr());
            let b_send = SendConstPtr(b.as_ptr());
            let c_send = SendMutPtr(c.as_mut_ptr());
            super::scope_for_macs(macs, |s| {
                let mut t0 = 0;
                while t0 < m {
                    let t1 = (t0 + rpt).min(m);
                    let mut u0 = 0;
                    while u0 < n {
                        let u1 = (u0 + cpt).min(n);
                        s.spawn(move |_| {
                            let a_send: SendConstPtr<f32> = a_send;
                            let b_send: SendConstPtr<f32> = b_send;
                            let c_send: SendMutPtr<f32> = c_send;
                            // Thread-local packed-B tile buffer (see the
                            // N-split: per-task `Vec` mmaps serialise the pool).
                            super::with_buf(&super::F32_B_PACK, F32_KC * F32_NC, |tile| {
                                let mut jc = u0;
                                while jc < u1 {
                                    let nc = (jc + F32_NC).min(u1) - jc;
                                    // SAFETY: pack overwrites `kc * nc`
                                    // elements before the sweep reads them.
                                    if trans_b {
                                        pack_b_tile_f32_trans(
                                            tile.as_mut_ptr(),
                                            b_send.0,
                                            ldb,
                                            jc,
                                            nc,
                                            0,
                                            k,
                                        );
                                    } else {
                                        pack_b_tile_f32(
                                            tile.as_mut_ptr(),
                                            b_send.0,
                                            ldb,
                                            jc,
                                            nc,
                                            0,
                                            k,
                                        );
                                    }
                                    sweep_f32(
                                        t0,
                                        t1,
                                        m,
                                        jc,
                                        nc,
                                        0,
                                        k,
                                        alpha,
                                        a_send.0,
                                        a_row_stride,
                                        a_col_stride,
                                        tile.as_ptr(),
                                        nc,
                                        0,
                                        jc,
                                        beta,
                                        c_send.0,
                                        ldc,
                                    );
                                    jc += F32_NC;
                                }
                            });
                        });
                        u0 = u1;
                    }
                    t0 = t1;
                }
            });
            return;
        }
    }
    // N-split path: tasks own disjoint C column bands and sweep all rows.
    // Preferred when the output is wider than tall so the big operand `B`
    // is read once total instead of once per row band. One scope covers all
    // Kc passes: bands are disjoint, so passes need no barrier — each task
    // runs its own pass loop (thin-`m` paid one spawn-join round per pass
    // before, e.g. 8 rounds at k = 2048). Per-element accumulation order
    // still matches the serial pc-ascending nest exactly.
    if !force_m_split() && !use_m_split_rank_k(m, n, k, F32_KC) && use_parallel_n(m, n, k, F32_KC) {
        let kc0 = k.min(F32_KC);
        let macs_n = m.saturating_mul(n).saturating_mul(kc0);
        // NOTE (2026-09-28): a CCD-compact N-split route (1-CCD pool at
        // A+C <= 8 MiB) shipped and was reverted: order-fair ABBA shows
        // it wins only square-896/1024 (+10-35%) and loses ~45-58%
        // everywhere else (sq512/sq768/rects) — the 8-vs-16-core deficit
        // beats the 1-CCD sharing gain except at one resonance. See
        // perf/PERF.md; do not re-ship without a proven win-domain gate.
        let workers = super::workers_for_macs(macs_n);
        let cpt = cols_per_task(
            m,
            n,
            k,
            F32_NR,
            super::workers_for_macs(m.saturating_mul(n).saturating_mul(k)),
        );
        // Floor tile size (see `widen_to_min_tile`): fixed for all passes
        // so every pass reuses the same band partition (sized on a full Kc
        // pass; the tail pass just reuses it).
        // Pool-fill floor (see `widen_to_min_tile`): the executing scope's
        // own width, so router and sizer agree.
        let cpt = widen_to_min_tile(n, m.saturating_mul(kc0), F32_NR, cpt, workers);
        // Fixed-cost band cap (see `cap_bands_by_work`): applied here at
        // the driver so `cols_per_task` keeps returning geometry for the
        // 2D gate's `bands_n` veto.
        let cpt = {
            let macs = m.saturating_mul(n).saturating_mul(k);
            cap_bands_by_work(
                n,
                macs,
                n.div_ceil(cpt.max(1)).max(1),
                F32_NR,
                F32_MIN_TASK_MAC,
            )
        };
        // L3-fit rule wins when it applies (see `l3_fit_cpt`), else
        // legacy sizing.
        let cpt = l3_fit_cpt(
            m,
            n,
            k,
            core::mem::size_of::<f32>(),
            F32_NR,
            workers,
            F32_MIN_TASK_MAC,
        )
        .unwrap_or(cpt);
        // Narrow-band rule wins when it applies (see
        // `f32_narrow_cpt`), else the legacy/L3-fit sizing above.
        let cpt = f32_narrow_cpt(m, n, k, workers, cpt).unwrap_or(cpt);

        // Trans-A shared pack (see `apack_on`): pack the whole strided
        // `A^T` once (TLS-cached) so every N-task streams it instead
        // of re-gathering per (task, slab, tile). Bitwise identical
        // (same values, same sweep order); the gather arm covers
        // L3-resident `A` where the extra pass is pure overhead.
        if trans_a && apack_on(m, k) {
            // SAFETY: AVX2 checked by the caller; bounds pre-checked.
            unsafe {
                with_packed_a_trans_f32_mt(m, k, a, lda, apack_par(), |packed| {
                    super::scope_for_macs(macs_n, |s| {
                        spawn_n_tasks_f32(
                            s,
                            m,
                            n,
                            k,
                            alpha,
                            packed.as_ptr(),
                            k,
                            1,
                            b.as_ptr(),
                            ldb,
                            trans_b,
                            beta,
                            c.as_mut_ptr(),
                            ldc,
                            cpt,
                        );
                    });
                });
            }
        } else {
            super::scope_for_macs(macs_n, |s| {
                // SAFETY: task bodies as documented on the spawner.
                unsafe {
                    spawn_n_tasks_f32(
                        s,
                        m,
                        n,
                        k,
                        alpha,
                        a.as_ptr(),
                        a_row_stride,
                        a_col_stride,
                        b.as_ptr(),
                        ldb,
                        trans_b,
                        beta,
                        c.as_mut_ptr(),
                        ldc,
                        cpt,
                    );
                }
            });
        }
        return;
    }
    // Parallel path: one scope-join per Kc pass over the FULL n range, so a
    // thin-K update pays one wakeup per pass instead of one per N tile.
    // Tasks own disjoint C row bands and sweep every N tile in serial jc
    // order; per-element accumulation order matches the serial nest exactly.
    if use_parallel(m, n, k, F32_KC) {
        let rpt = rows_per_task(m, n, k);
        // Fused-pass M-split (see `m_fuse_on`): one scope, each row-band
        // task loops over all Kc passes. Same packs, same sweep order per
        // element — bitwise identical to the pass-per-scope loop below.
        // Pool sizing uses per-pass MACs so routing matches the legacy
        // loop exactly (see the N-split fused scope, same convention).
        if m_fuse_on() && !trans_b {
            let kc0 = mtall_kc_override()
                .map(|kc| k.min(kc))
                .unwrap_or(k.min(F32_KC));
            let panel_n = panel_stride(n, core::mem::size_of::<f32>());
            let privb_allowed = mtall_privb_override().unwrap_or(true);
            if privb_allowed
                && let Some(kc) = m_fuse_kc(
                    k,
                    kc0,
                    panel_n,
                    core::mem::size_of::<f32>(),
                    PRIVB_L2_BUDGET_F32,
                )
            {
                let workers = super::workers_for_macs(m.saturating_mul(n).saturating_mul(kc));
                let rpt = widen_to_min_tile(m, n.saturating_mul(kc), MR, rpt, workers);
                // Probe override (see `mtall_rpt_override`): MR-aligned, clamped.
                let rpt = mtall_rpt_override()
                    .map(|r| (r / MR * MR).max(MR).min(m.max(1)))
                    .unwrap_or(rpt);
                let a_send = SendConstPtr(a.as_ptr());
                let b_src = SendConstPtr(b.as_ptr());
                let c_send = SendMutPtr(c.as_mut_ptr());
                // SAFETY: each task owns rows `[t0, t1)` across all passes —
                // disjoint C bands, read-only A/B; per-element pass order
                // matches the pass-per-scope loop exactly.
                super::scope_for_macs(m.saturating_mul(n).saturating_mul(kc), |s| {
                    let mut t0 = 0;
                    while t0 < m {
                        let t1 = (t0 + rpt).min(m);
                        s.spawn(move |_| {
                            let a_send: SendConstPtr<f32> = a_send;
                            let b_src: SendConstPtr<f32> = b_src;
                            let c_send: SendMutPtr<f32> = c_send;
                            // One TLS checkout for all passes (each pass
                            // pack overwrites the panel fully).
                            super::with_buf(&super::F32_B_PACK, kc * panel_n, |tile| {
                                let mut pc = 0;
                                while pc < k {
                                    let beta_eff = if pc == 0 { beta } else { 1.0 };
                                    // SAFETY: pack overwrites `kc * n` before
                                    // the sweep reads; TLS buffer outlives task.
                                    pack_b_panel_f32(
                                        tile.as_mut_ptr(),
                                        b_src.0,
                                        ldb,
                                        n,
                                        pc,
                                        kc,
                                        panel_n,
                                    );
                                    let mut jc = 0;
                                    while jc < n {
                                        let nc = (n - jc).min(F32_NC);
                                        sweep_f32(
                                            t0,
                                            t1,
                                            m,
                                            jc,
                                            nc,
                                            pc,
                                            kc,
                                            alpha,
                                            a_send.0,
                                            a_row_stride,
                                            a_col_stride,
                                            tile.as_ptr(),
                                            panel_n,
                                            0,
                                            0,
                                            beta_eff,
                                            c_send.0,
                                            ldc,
                                        );
                                        jc += F32_NC;
                                    }
                                    pc += kc;
                                }
                            });
                        });
                        t0 = t1;
                    }
                });
                return;
            }
        }
        // Packed-B band buffer, capacity reused across Kc passes.
        let mut panel: Vec<f32> = Vec::new();
        let mut pc = 0;
        while pc < k {
            // Pass width stays F32_KC: wider passes (512/1024) trialled
            // 2026-09-26 on 2048x128x2048 via a temporary env gate
            // (removed) — one noisy -8% then 1 tie + 2 losses interleaved.
            // The narrow-n deficit is micro-tile efficiency, not passes.
            let kc0 = mtall_kc_override()
                .map(|kc| (k - pc).min(kc))
                .unwrap_or((k - pc).min(F32_KC));
            let beta_eff = if pc == 0 { beta } else { 1.0 };
            // Padded panel stride (see `panel_stride`): breaks 2/4 KiB row
            // aliasing on power-of-2 widths; identity otherwise. Hoisted
            // (n-only) so the privb L2-fit gate below can use it.
            let panel_n = panel_stride(n, core::mem::size_of::<f32>());
            // Private B-panel: each M-split row task packs its own
            // panel (TLS, L2-resident) instead of sharing the L3 panel
            // (thin-N wins -10%: 64 tasks x L3 re-reads saturate the
            // shared panel). NoTrans panels only (transposed packs
            // re-read strided source per task: untested, stays shared).
            // `kc` is capped so the panel fits its L2 budget (see
            // `PRIVB_L2_BUDGET_F32`).
            let want_privb = !trans_b;
            let kc_try = if want_privb {
                kc0.min(privb_kc_cap::<f32>(panel_n, PRIVB_L2_BUDGET_F32))
            } else {
                kc0
            };
            // Shapes whose panel cannot fit the L2 budget at `kc >= 128`
            // stay on the shared panel with the unmodified `kc0`.
            let privb_fit = kc_try >= 128
                && kc_try
                    .saturating_mul(panel_n)
                    .saturating_mul(core::mem::size_of::<f32>())
                    <= PRIVB_L2_BUDGET_F32;
            let kc = if want_privb && privb_fit { kc_try } else { kc0 };
            // Floor tile size (see `widen_to_min_tile`).
            let workers = super::workers_for_macs(m.saturating_mul(n).saturating_mul(kc));
            let rpt = widen_to_min_tile(m, n.saturating_mul(kc), MR, rpt, workers);
            // Probe override (see `mtall_rpt_override`): MR-aligned, clamped.
            let rpt = mtall_rpt_override()
                .map(|r| (r / MR * MR).max(MR).min(m.max(1)))
                .unwrap_or(rpt);
            // Packed-B band shared read-only by every row task (see
            // `pack_b_panel_f32`); in-place reads stay for giant bands.
            // A `Trans` source always packs (transposed gather): in-place
            // reads would see the untransposed layout.
            let use_pack = trans_b || kc.saturating_mul(n) <= PANEL_MAX_ELS;
            // 64B-aligned panel base (see `align64_spare_ptr`); raw
            // offset into `panel`'s capacity, no zero-fill (the pack
            // overwrites before reads).
            let privb = use_pack
                && mtall_privb_override()
                    .map(|p| p && want_privb)
                    .unwrap_or(want_privb && privb_fit);
            let mut panel_ptr = panel.as_mut_ptr();
            if use_pack && !privb {
                panel.clear();
                panel_ptr = super::align64_spare_ptr(&mut panel, kc * panel_n);
                // SAFETY: pack overwrites all `kc * n` elements before any
                // task reads them; the scope joins before `panel` drops.
                // Both layouts produce the same `kc × n` panel of `op(B)`.
                if trans_b {
                    pack_b_panel_f32_trans(panel_ptr, b.as_ptr(), ldb, n, pc, kc, panel_n);
                } else {
                    pack_b_panel_f32(panel_ptr, b.as_ptr(), ldb, n, pc, kc, panel_n);
                }
            }
            // SAFETY: each task sweeps rows `[t0, t1)` across all N tiles —
            // disjoint C bands, read-only A/B. The scope joins before the
            // next Kc pass, so accumulation order per element matches serial.
            let a_send = SendConstPtr(a.as_ptr());
            let b_send = SendConstPtr(if use_pack { panel_ptr } else { b.as_ptr() });
            let b_src = SendConstPtr(b.as_ptr());
            let c_send = SendMutPtr(c.as_mut_ptr());
            super::scope_for_macs(m.saturating_mul(n).saturating_mul(kc), |s| {
                let mut t0 = 0;
                while t0 < m {
                    let t1 = (t0 + rpt).min(m);
                    s.spawn(move |_| {
                        let a_send: SendConstPtr<f32> = a_send;
                        let b_send: SendConstPtr<f32> = b_send;
                        let b_src: SendConstPtr<f32> = b_src;
                        let c_send: SendMutPtr<f32> = c_send;
                        // Private panel pack (TLS, worker-local L2)
                        // instead of the shared panel. Same pack fn,
                        // same values.
                        if privb {
                            super::with_buf(&super::F32_B_PACK, kc * panel_n, |tile| {
                                // SAFETY: pack overwrites `kc * n` before
                                // the sweep reads; TLS buffer outlives the
                                // task; same contract as the shared pack.
                                pack_b_panel_f32(
                                    tile.as_mut_ptr(),
                                    b_src.0,
                                    ldb,
                                    n,
                                    pc,
                                    kc,
                                    panel_n,
                                );
                                let mut jc = 0;
                                while jc < n {
                                    let nc = (n - jc).min(F32_NC);
                                    sweep_f32(
                                        t0,
                                        t1,
                                        m,
                                        jc,
                                        nc,
                                        pc,
                                        kc,
                                        alpha,
                                        a_send.0,
                                        a_row_stride,
                                        a_col_stride,
                                        tile.as_ptr(),
                                        panel_n,
                                        0,
                                        0,
                                        beta_eff,
                                        c_send.0,
                                        ldc,
                                    );
                                    jc += F32_NC;
                                }
                            });
                        } else {
                            let mut jc = 0;
                            while jc < n {
                                let nc = (n - jc).min(F32_NC);
                                sweep_f32(
                                    t0,
                                    t1,
                                    m,
                                    jc,
                                    nc,
                                    pc,
                                    kc,
                                    alpha,
                                    a_send.0,
                                    a_row_stride,
                                    a_col_stride,
                                    b_send.0,
                                    if use_pack { panel_n } else { ldb },
                                    if use_pack { 0 } else { pc },
                                    0,
                                    beta_eff,
                                    c_send.0,
                                    ldc,
                                );
                                jc += F32_NC;
                            }
                        }
                    });
                    t0 = t1;
                }
            });
            pc += kc;
        }
        return;
    }

    // Serial nest shared with [`sgemm_serial`]: per-tile streaming reads
    // each operand once (see that function for the nested-parallelism
    // rationale).
    unsafe {
        sgemm_serial(
            m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_a, trans_b,
        );
    }
}

/// Serial `sgemm` nest (no `rayon::scope`): for callers that already fan
/// out over threads (SYRK/TRSM bands). A tile-sized GEMM above
/// [`PAR_MIN_MACS`] would otherwise spawn its own N-/M-split scopes from
/// inside the outer scope — dozens of nested scopes whose per-task
/// thread-local buffer checkouts miss the pool and mmap, dwarfing the
/// kernel work. Per-element accumulation order matches the parallel
/// paths exactly, so results are bitwise identical either way.
///
/// Bounds are pre-checked by the caller; `trans_a` selects the stored-A
/// layout (k × m when true), `trans_b` the stored-B layout (n × k when
/// true — transposed-gather tiles, no pre-pack).
#[target_feature(enable = "avx2,fma")]
/// Serial NoTrans GEMM with packed `A` panels (see [`st_apack_on`]):
/// pc-outer / jc / ic-inner nest; each `Mc × kc` panel is packed once
/// per (pc, ic) at the first jc-tile and cached in TLS across the
/// remaining jc-tiles, so slice-overflowing `A` streams once per pass
/// instead of once per (jc, pc). B-tile traffic matches the legacy
/// nest (one pack per (pc, jc)). Per-element pass order is pc-ascending
/// like the legacy nest, so results are bitwise identical.
unsafe fn sgemm_serial_apack(
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
    trans_b: bool,
) {
    let npanels = m.div_ceil(F32_MC);
    super::with_buf(&super::F32_A_PANELS, npanels * F32_MC * F32_KC, |panels| {
        super::with_buf(&super::F32_B_PACK, F32_KC * F32_NC, |tile| {
            let mut pc = 0;
            while pc < k {
                let kc = (k - pc).min(F32_KC);
                let beta_eff = if pc == 0 { beta } else { 1.0 };
                let mut jc = 0;
                while jc < n {
                    let nc = (n - jc).min(F32_NC);
                    // SAFETY: pack overwrites `kc * nc` before the
                    // sweep reads them (same contract as the nest).
                    if trans_b {
                        pack_b_tile_f32_trans(tile.as_mut_ptr(), b.as_ptr(), ldb, jc, nc, pc, kc);
                    } else {
                        pack_b_tile_f32(tile.as_mut_ptr(), b.as_ptr(), ldb, jc, nc, pc, kc);
                    }
                    let mut ic = 0;
                    let mut pi = 0;
                    while ic < m {
                        let mc = (m - ic).min(F32_MC);
                        // SAFETY: panel `pi` owns
                        // `Mc*KC` elements at its offset; the pack
                        // writes `mc * kc` of them.
                        let panel = panels.as_mut_ptr().add(pi * F32_MC * F32_KC);
                        if jc == 0 {
                            // Rows [ic, ic+mc) × cols [pc, pc+kc]
                            // become contiguous `mc × kc` (the tile
                            // packer with rows/cols swapped).
                            pack_b_tile_f32(panel, a.as_ptr(), lda, pc, kc, ic, mc);
                        }
                        // SAFETY: panel-relative sweep (`pc = 0`,
                        // row stride `kc`); `C` rebased to row `ic`.
                        sweep_f32(
                            0,
                            mc,
                            mc,
                            jc,
                            nc,
                            0,
                            kc,
                            alpha,
                            panel,
                            kc,
                            1,
                            tile.as_ptr(),
                            nc,
                            0,
                            jc,
                            beta_eff,
                            c.as_mut_ptr().add(ic * ldc),
                            ldc,
                        );
                        ic += mc;
                        pi += 1;
                    }
                    jc += nc;
                }
                pc += kc;
            }
        });
    });
}

pub(crate) unsafe fn sgemm_serial(
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
    trans_a: bool,
    trans_b: bool,
) {
    if m == 0 || n == 0 {
        return;
    }
    if alpha == 0.0 || k == 0 {
        scale_c_f32(m, n, beta, c.as_mut_ptr(), ldc);
        return;
    }
    // Pure-dimension pre-check first (see the f64 twin): never evaluate
    // `l3_bytes()` for shapes the apack path rejects on dims alone.
    if st_apack_on()
        && !trans_a
        && m > F32_MC
        && n > F32_NC
        && k > F32_KC
        && st_apack_shape(
            m,
            n,
            k,
            F32_MC,
            F32_KC,
            F32_NC,
            core::mem::size_of::<f32>(),
            sonium_topology::affinity::l3_bytes(),
        )
    {
        sgemm_serial_apack(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_b);
        return;
    }
    let (a_row_stride, a_col_stride) = if trans_a { (1, lda) } else { (lda, 1) };
    // Thread-local tile buffer (see the N-split tasks): a per-call
    // `Vec` mmaps on every call, which dominates when a blocked
    // solver (TRSM/SYRK/LU) issues dozens of small GEMMs per solve.
    super::with_buf(&super::F32_B_PACK, F32_KC * F32_NC, |tile| {
        let mut jc = 0;
        while jc < n {
            let nc = (n - jc).min(F32_NC);
            let mut pc = 0;
            while pc < k {
                let kc = (k - pc).min(F32_KC);
                let beta_eff = if pc == 0 { beta } else { 1.0 };
                // SAFETY: pack overwrites `kc * nc` elements before the sweep
                // reads them. The transposed gather produces the same tile
                // elements from a `Trans` source — no pre-pack needed.
                if trans_b {
                    pack_b_tile_f32_trans(tile.as_mut_ptr(), b.as_ptr(), ldb, jc, nc, pc, kc);
                } else {
                    pack_b_tile_f32(tile.as_mut_ptr(), b.as_ptr(), ldb, jc, nc, pc, kc);
                }
                sweep_f32(
                    0,
                    m,
                    m,
                    jc,
                    nc,
                    pc,
                    kc,
                    alpha,
                    a.as_ptr(),
                    a_row_stride,
                    a_col_stride,
                    tile.as_ptr(),
                    nc,
                    0,
                    jc,
                    beta_eff,
                    c.as_mut_ptr(),
                    ldc,
                );
                pc += kc;
            }
            jc += nc;
        }
    });
}

/// `C := alpha * op(A) * op(B) + beta * C` (f64). See [`sgemm`].
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dgemm(
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
    trans_a: bool,
    trans_b: bool,
) {
    if m == 0 || n == 0 {
        return;
    }
    if alpha == 0.0 || k == 0 {
        scale_c_f64(m, n, beta, c.as_mut_ptr(), ldc);
        return;
    }
    // Serial-only pre-pack: the A^T pack pays off for the serial sweep
    // (ST map 09-26: pack beats gather on all 5 trans-A shapes, e.g.
    // 256x2048x2048T 1.13x vs 1.42x) but loses under threads (MT map:
    // gather beats pack everywhere, e.g. 32x300x2048T 1.60x vs 2.59x,
    // 256x2048x2048T 0.82x vs 1.26x) — the serial pack plus the recursed
    // path cost more than strided streaming saves. Results identical
    // either way (probe maxdiff 0e0 both regimes).
    if trans_a && m * k >= DGEMM_TRANSPOSE_PACK_MIN && rayon::current_num_threads() == 1 {
        // Transposed-A micro-panels broadcast from lda-strided storage
        // (one cache line per element); packing A^T once into a
        // contiguous buffer lets the NoTrans path stream it (~3x on
        // 32x300x299). Single recursion, terminates (inner call is
        // NoTrans).
        with_packed_a_trans_f64(m, k, a.as_ptr(), lda, |packed| {
            dgemm(
                m, n, k, alpha, packed, k, b, ldb, beta, c, ldc, false, trans_b,
            );
        });
        return;
    }
    // Multi-threaded pack-once-shared `A^T` (see `pack_a_mt`): one
    // shared pack replaces every band task's strided `A` gather. Single
    // recursion, terminates (inner call is `NoTrans`).
    if trans_a && pack_a_mt() && rayon::current_num_threads() > 1 && pack_a_mt_shape(m, n, k) {
        // TLS-buffer pack (see `pack_a_mt_tls`): same parallel pack
        // and NoTrans recursion, only the buffer is pre-faulted TLS
        // instead of a fresh `vec!`. No reentrancy: the recursed
        // NoTrans call never touches `DGEMM_A_PACK`.
        if pack_a_mt_tls() {
            DGEMM_A_PACK.with(|cell| match cell.try_borrow_mut() {
                Ok(mut buf) => {
                    if buf.len() < m * k {
                        buf.resize(m * k, 0.0);
                    }
                    let dst = &mut buf[..m * k];
                    super::pack_a_trans_f64(a, lda, m, k, dst);
                    dgemm(m, n, k, alpha, dst, k, b, ldb, beta, c, ldc, false, trans_b);
                }
                Err(_) => {
                    let mut packed = vec![0.0f64; m * k];
                    super::pack_a_trans_f64(a, lda, m, k, &mut packed);
                    dgemm(
                        m, n, k, alpha, &packed, k, b, ldb, beta, c, ldc, false, trans_b,
                    );
                }
            });
            return;
        }
        let mut packed = vec![0.0f64; m * k];
        super::pack_a_trans_f64(a, lda, m, k, &mut packed);
        dgemm(
            m, n, k, alpha, &packed, k, b, ldb, beta, c, ldc, false, trans_b,
        );
        return;
    }
    let (a_row_stride, a_col_stride) = if trans_a { (1, lda) } else { (lda, 1) };
    // 2D-split path: tasks own disjoint C row-band × column-band rectangles
    // (see `use_parallel_2d`). Single-pass thin-`k` only: each task packs
    // its `B` tile once and sweeps its rows once, with no per-pass re-pack.
    // Row-splitting cuts the N-split's per-band `A` re-reads while the
    // L1-capped band width keeps the `B` stream L1-resident. Per-element
    // accumulation order matches the serial nest exactly.
    {
        let macs = m.saturating_mul(n).saturating_mul(k);
        let workers = super::workers_for_macs(macs);
        if !force_m_split()
            && !use_m_split_rank_k(m, n, k, F64_KC)
            && use_parallel_2d(m, n, k, F64_NR, F64_KC, 4096, workers)
        {
            let (rpt, cpt) = grid_2d(m, n, k, F64_NR, 4096, workers);
            // SAFETY: each task sweeps rows `[t0, t1)` × columns `[u0, u1)` —
            // disjoint C rectangles, read-only A/B.
            let a_send = SendConstPtr(a.as_ptr());
            let b_send = SendConstPtr(b.as_ptr());
            let c_send = SendMutPtr(c.as_mut_ptr());
            super::scope_for_macs(macs, |s| {
                let mut t0 = 0;
                while t0 < m {
                    let t1 = (t0 + rpt).min(m);
                    let mut u0 = 0;
                    while u0 < n {
                        let u1 = (u0 + cpt).min(n);
                        s.spawn(move |_| {
                            let a_send: SendConstPtr<f64> = a_send;
                            let b_send: SendConstPtr<f64> = b_send;
                            let c_send: SendMutPtr<f64> = c_send;
                            // Thread-local packed-B tile buffer (see the
                            // N-split: per-task `Vec` mmaps serialise the pool).
                            super::with_buf(&super::F64_B_PACK, F64_KC * F64_NC, |tile| {
                                let mut jc = u0;
                                while jc < u1 {
                                    let nc = (jc + F64_NC).min(u1) - jc;
                                    // SAFETY: pack overwrites `kc * nc`
                                    // elements before the sweep reads them.
                                    if trans_b {
                                        pack_b_tile_f64_trans(
                                            tile.as_mut_ptr(),
                                            b_send.0,
                                            ldb,
                                            jc,
                                            nc,
                                            0,
                                            k,
                                        );
                                    } else {
                                        pack_b_tile_f64(
                                            tile.as_mut_ptr(),
                                            b_send.0,
                                            ldb,
                                            jc,
                                            nc,
                                            0,
                                            k,
                                        );
                                    }
                                    sweep_f64(
                                        t0,
                                        t1,
                                        m,
                                        jc,
                                        nc,
                                        0,
                                        k,
                                        alpha,
                                        a_send.0,
                                        a_row_stride,
                                        a_col_stride,
                                        tile.as_ptr(),
                                        nc,
                                        0,
                                        jc,
                                        beta,
                                        c_send.0,
                                        ldc,
                                    );
                                    jc += F64_NC;
                                }
                            });
                        });
                        u0 = u1;
                    }
                    t0 = t1;
                }
            });
            return;
        }
    }
    // N-split path: tasks own disjoint C column bands and sweep all rows
    // (mirror of the `sgemm` N-split above with `F64_NC` tiles). One scope
    // covers all Kc passes: bands are disjoint, so passes need no barrier —
    // each task runs its own pass loop (thin-`m` paid one spawn-join round
    // per pass before, e.g. 8 rounds at k = 2048). Per-element accumulation
    // order still matches the serial pc-ascending nest exactly.
    if !force_m_split() && !use_m_split_rank_k(m, n, k, F64_KC) && use_parallel_n(m, n, k, F64_KC) {
        let cpt = cols_per_task(
            m,
            n,
            k,
            F64_NR,
            super::workers_for_macs(m.saturating_mul(n).saturating_mul(k)),
        );
        // Floor tile size (see `widen_to_min_tile`): fixed for all passes
        // so every pass reuses the same band partition (sized on a full Kc
        // pass; the tail pass just reuses it).
        let kc0 = k.min(F64_KC);
        // Pool-fill floor (see `widen_to_min_tile`): the executing scope's
        // own width, so router and sizer agree.
        let workers = super::workers_for_macs(m.saturating_mul(n).saturating_mul(kc0));
        let cpt = widen_to_min_tile(n, m.saturating_mul(kc0), F64_NR, cpt, workers);
        // Fixed-cost band cap (see `cap_bands_by_work`): applied here at
        // the driver so `cols_per_task` keeps returning geometry for the
        // 2D gate's `bands_n` veto.
        let cpt = {
            let macs = m.saturating_mul(n).saturating_mul(k);
            cap_bands_by_work(n, macs, n.div_ceil(cpt.max(1)).max(1), F64_NR, MIN_TASK_MAC)
        };
        // L3-fit rule wins when it applies (see `l3_fit_cpt`), else
        // legacy sizing.
        let cpt = l3_fit_cpt(
            m,
            n,
            k,
            core::mem::size_of::<f64>(),
            F64_NR,
            workers,
            MIN_TASK_MAC,
        )
        .unwrap_or(cpt);
        // Narrow-band rule wins when it applies (see
        // `f64_narrow_cpt`), else the legacy/L3-fit sizing above.
        let cpt = f64_narrow_cpt(m, n, k, workers, cpt).unwrap_or(cpt);

        // SAFETY: each task sweeps columns `[u0, u1)` across all rows —
        // disjoint C bands, read-only A/B.
        let a_send = SendConstPtr(a.as_ptr());
        let b_send = SendConstPtr(b.as_ptr());
        let c_send = SendMutPtr(c.as_mut_ptr());
        super::scope_for_macs(m.saturating_mul(n).saturating_mul(kc0), |s| {
            let mut u0 = 0;
            while u0 < n {
                let u1 = (u0 + cpt).min(n);
                s.spawn(move |_| {
                    let a_send: SendConstPtr<f64> = a_send;
                    let b_send: SendConstPtr<f64> = b_send;
                    let c_send: SendMutPtr<f64> = c_send;
                    // Packed-B tile buffer shared with the calling
                    // thread's pool (thread-local, reused across passes
                    // and calls): a per-task `Vec` here mmaps/munmaps
                    // ~512 KB per pass, which serialises all tasks on
                    // the allocator and TLB shootdowns.
                    // Wide-Kc passes in the gated domain (see `f64_kc_nsplit`).
                    let kc_step = f64_kc_nsplit(n, k);
                    super::with_buf(&super::F64_B_PACK, kc_step * F64_NC, |tile| {
                        // Kc passes fused into the task (see above): one
                        // spawn covers all passes, no barrier between.
                        let mut pc = 0;
                        while pc < k {
                            let kc = (k - pc).min(kc_step);
                            let beta_eff = if pc == 0 { beta } else { 1.0 };
                            let mut jc = u0;
                            while jc < u1 {
                                let nc = (jc + F64_NC).min(u1) - jc;
                                // SAFETY: pack overwrites `kc * nc`
                                // elements before the sweep reads them
                                // (transposed gather for a `Trans`
                                // source).
                                if trans_b {
                                    pack_b_tile_f64_trans(
                                        tile.as_mut_ptr(),
                                        b_send.0,
                                        ldb,
                                        jc,
                                        nc,
                                        pc,
                                        kc,
                                    );
                                } else {
                                    pack_b_tile_f64(
                                        tile.as_mut_ptr(),
                                        b_send.0,
                                        ldb,
                                        jc,
                                        nc,
                                        pc,
                                        kc,
                                    );
                                }
                                sweep_f64(
                                    0,
                                    m,
                                    m,
                                    jc,
                                    nc,
                                    pc,
                                    kc,
                                    alpha,
                                    a_send.0,
                                    a_row_stride,
                                    a_col_stride,
                                    tile.as_ptr(),
                                    nc,
                                    0,
                                    jc,
                                    beta_eff,
                                    c_send.0,
                                    ldc,
                                );
                                jc += F64_NC;
                            }
                            pc += kc;
                        }
                    });
                });
                u0 = u1;
            }
        });
        return;
    }
    // Parallel path: one scope-join per Kc pass over the FULL n range, so a
    // thin-K update pays one wakeup per pass instead of one per N tile.
    // Tasks own disjoint C row bands and sweep every N tile in serial jc
    // order; per-element accumulation order matches the serial nest exactly.
    if use_parallel(m, n, k, F64_KC) {
        let rpt = rows_per_task(m, n, k);
        // Fused-pass M-split (see `m_fuse_on`): f64 twin of the `sgemm`
        // fused scope. Bitwise identical to the pass-per-scope loop.
        // Per-pass pool MACs match the legacy loop (N-split convention).
        if m_fuse_on() && !trans_b {
            let kc0 = mtall_kc_override()
                .map(|kc| k.min(kc))
                .unwrap_or(k.min(F64_KC_M));
            let panel_n = panel_stride(n, core::mem::size_of::<f64>());
            let privb_allowed = mtall_privb_override().unwrap_or(true);
            if privb_allowed
                && let Some(kc) = m_fuse_kc(
                    k,
                    kc0,
                    panel_n,
                    core::mem::size_of::<f64>(),
                    PRIVB_L2_BUDGET_F64,
                )
            {
                let workers = super::workers_for_macs(m.saturating_mul(n).saturating_mul(kc));
                let rpt = widen_to_min_tile(m, n.saturating_mul(kc), MR, rpt, workers);
                // Probe override (see `mtall_rpt_override`): MR-aligned, clamped.
                let rpt = mtall_rpt_override()
                    .map(|r| (r / MR * MR).max(MR).min(m.max(1)))
                    .unwrap_or(rpt);
                let a_send = SendConstPtr(a.as_ptr());
                let b_src = SendConstPtr(b.as_ptr());
                let c_send = SendMutPtr(c.as_mut_ptr());
                // SAFETY: each task owns rows `[t0, t1)` across all passes —
                // disjoint C bands, read-only A/B; per-element pass order
                // matches the pass-per-scope loop exactly.
                super::scope_for_macs(m.saturating_mul(n).saturating_mul(kc), |s| {
                    let mut t0 = 0;
                    while t0 < m {
                        let t1 = (t0 + rpt).min(m);
                        s.spawn(move |_| {
                            let a_send: SendConstPtr<f64> = a_send;
                            let b_src: SendConstPtr<f64> = b_src;
                            let c_send: SendMutPtr<f64> = c_send;
                            // One TLS checkout for all passes (each pass
                            // pack overwrites the panel fully).
                            super::with_buf(&super::F64_B_PACK, kc * panel_n, |tile| {
                                let mut pc = 0;
                                while pc < k {
                                    let beta_eff = if pc == 0 { beta } else { 1.0 };
                                    // SAFETY: pack overwrites `kc * n` before
                                    // the sweep reads; TLS buffer outlives task.
                                    pack_b_panel_f64(
                                        tile.as_mut_ptr(),
                                        b_src.0,
                                        ldb,
                                        n,
                                        pc,
                                        kc,
                                        panel_n,
                                    );
                                    let mut jc = 0;
                                    while jc < n {
                                        let nc = (n - jc).min(F64_NC);
                                        sweep_f64(
                                            t0,
                                            t1,
                                            m,
                                            jc,
                                            nc,
                                            pc,
                                            kc,
                                            alpha,
                                            a_send.0,
                                            a_row_stride,
                                            a_col_stride,
                                            tile.as_ptr(),
                                            panel_n,
                                            0,
                                            0,
                                            beta_eff,
                                            c_send.0,
                                            ldc,
                                        );
                                        jc += F64_NC;
                                    }
                                    pc += kc;
                                }
                            });
                        });
                        t0 = t1;
                    }
                });
                return;
            }
        }
        // Packed-B band buffer, capacity reused across Kc passes (reserve
        // keeps the mapping; re-allocating per pass would mmap/munmap it).
        let mut panel: Vec<f64> = Vec::new();
        let mut pc = 0;
        while pc < k {
            let kc0 = mtall_kc_override()
                .map(|kc| (k - pc).min(kc))
                .unwrap_or((k - pc).min(F64_KC_M));
            let beta_eff = if pc == 0 { beta } else { 1.0 };
            // Padded panel stride hoisted (n-only, see `sgemm`).
            let panel_n = panel_stride(n, core::mem::size_of::<f64>());
            // Private B-panel (see `sgemm`): `f64` twin.
            let want_privb = !trans_b;
            let kc_try = if want_privb {
                kc0.min(privb_kc_cap::<f64>(panel_n, PRIVB_L2_BUDGET_F64))
            } else {
                kc0
            };
            let privb_fit = kc_try >= 128
                && kc_try
                    .saturating_mul(panel_n)
                    .saturating_mul(core::mem::size_of::<f64>())
                    <= PRIVB_L2_BUDGET_F64;
            let kc = if want_privb && privb_fit { kc_try } else { kc0 };
            // Floor tile size (see `widen_to_min_tile`): the executing
            // scope's own width, so router and sizer agree.
            let workers = super::workers_for_macs(m.saturating_mul(n).saturating_mul(kc));
            let rpt = widen_to_min_tile(m, n.saturating_mul(kc), MR, rpt, workers);
            // Probe override (see `mtall_rpt_override`): MR-aligned, clamped.
            let rpt = mtall_rpt_override()
                .map(|r| (r / MR * MR).max(MR).min(m.max(1)))
                .unwrap_or(rpt);
            // Packed-B band shared read-only by every row task (see
            // `pack_b_panel_f64`): one parallel copy per pass instead of one
            // strided L1-thrashing re-read per task. In-place reads stay for
            // bands that would rival the source matrix in size.
            // A `Trans` source always packs (transposed gather): in-place
            // reads would see the untransposed layout.
            let use_pack = trans_b || kc.saturating_mul(n) <= PANEL_MAX_ELS;
            // Private panel only when it fits (see `sgemm`).
            let privb = use_pack
                && mtall_privb_override()
                    .map(|p| p && want_privb)
                    .unwrap_or(want_privb && privb_fit);
            if use_pack && !privb {
                panel.clear();
                panel.reserve(kc * panel_n);
                // SAFETY: pack overwrites all `kc * n` elements before any
                // task reads them; the scope joins before `panel` drops.
                // Both layouts produce the same `kc × n` panel of `op(B)`.
                if trans_b {
                    pack_b_panel_f64_trans(panel.as_mut_ptr(), b.as_ptr(), ldb, n, pc, kc, panel_n);
                } else {
                    pack_b_panel_f64(panel.as_mut_ptr(), b.as_ptr(), ldb, n, pc, kc, panel_n);
                }
            }
            // SAFETY: each task sweeps rows `[t0, t1)` across all N tiles —
            // disjoint C bands, read-only A/B. The scope joins before the
            // next Kc pass, so accumulation order per element matches serial.
            let a_send = SendConstPtr(a.as_ptr());
            let b_send = SendConstPtr(if use_pack { panel.as_ptr() } else { b.as_ptr() });
            let b_src = SendConstPtr(b.as_ptr());
            let c_send = SendMutPtr(c.as_mut_ptr());
            super::scope_for_macs(m.saturating_mul(n).saturating_mul(kc), |s| {
                let mut t0 = 0;
                while t0 < m {
                    let t1 = (t0 + rpt).min(m);
                    s.spawn(move |_| {
                        let a_send: SendConstPtr<f64> = a_send;
                        let b_send: SendConstPtr<f64> = b_send;
                        let b_src: SendConstPtr<f64> = b_src;
                        let c_send: SendMutPtr<f64> = c_send;
                        if privb {
                            super::with_buf(&super::F64_B_PACK, kc * panel_n, |tile| {
                                // SAFETY: same contract as the `sgemm`
                                // twin (pack overwrites before reads).
                                pack_b_panel_f64(
                                    tile.as_mut_ptr(),
                                    b_src.0,
                                    ldb,
                                    n,
                                    pc,
                                    kc,
                                    panel_n,
                                );
                                let mut jc = 0;
                                while jc < n {
                                    let nc = (n - jc).min(F64_NC);
                                    sweep_f64(
                                        t0,
                                        t1,
                                        m,
                                        jc,
                                        nc,
                                        pc,
                                        kc,
                                        alpha,
                                        a_send.0,
                                        a_row_stride,
                                        a_col_stride,
                                        tile.as_ptr(),
                                        panel_n,
                                        0,
                                        0,
                                        beta_eff,
                                        c_send.0,
                                        ldc,
                                    );
                                    jc += F64_NC;
                                }
                            });
                        } else {
                            let mut jc = 0;
                            while jc < n {
                                let nc = (n - jc).min(F64_NC);
                                sweep_f64(
                                    t0,
                                    t1,
                                    m,
                                    jc,
                                    nc,
                                    pc,
                                    kc,
                                    alpha,
                                    a_send.0,
                                    a_row_stride,
                                    a_col_stride,
                                    b_send.0,
                                    if use_pack { panel_n } else { ldb },
                                    if use_pack { 0 } else { pc },
                                    0,
                                    beta_eff,
                                    c_send.0,
                                    ldc,
                                );
                                jc += F64_NC;
                            }
                        }
                    });
                    t0 = t1;
                }
            });
            pc += kc;
        }
        return;
    }

    // Serial nest shared with [`dgemm_serial`]: per-tile streaming reads
    // each operand once (see that function for the nested-parallelism
    // rationale).
    unsafe {
        dgemm_serial(
            m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_a, trans_b,
        );
    }
}

/// Serial `dgemm` nest (no `rayon::scope`): for callers that already fan
/// out over threads (SYRK/TRSM bands). A tile-sized GEMM above
/// [`PAR_MIN_MACS`] would otherwise spawn its own N-/M-split scopes from
/// inside the outer scope — dozens of nested scopes whose per-task
/// thread-local buffer checkouts miss the pool and mmap, dwarfing the
/// kernel work. Per-element accumulation order matches the parallel
/// paths exactly, so results are bitwise identical either way.
///
/// Bounds are pre-checked by the caller; `trans_a` selects the stored-A
/// layout (k × m when true), `trans_b` the stored-B layout (n × k when
/// true — transposed-gather tiles, no pre-pack).
/// f64 twin of [`sgemm_serial_apack`]: pc-outer / jc / ic-inner nest
/// with cached `Mc × kc` `A` panels (see [`st_apack_on`]). Bitwise
/// identical to the legacy nest (pc-ascending per-element order).
unsafe fn dgemm_serial_apack(
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
    trans_b: bool,
) {
    let npanels = m.div_ceil(F64_MC);
    super::with_buf(&super::F64_A_PANELS, npanels * F64_MC * F64_KC, |panels| {
        super::with_buf(&super::F64_B_PACK, F64_KC * F64_NC, |tile| {
            let mut pc = 0;
            while pc < k {
                let kc = (k - pc).min(F64_KC);
                let beta_eff = if pc == 0 { beta } else { 1.0 };
                let mut jc = 0;
                while jc < n {
                    let nc = (n - jc).min(F64_NC);
                    // SAFETY: pack overwrites `kc * nc` before the
                    // sweep reads them (same contract as the nest).
                    if trans_b {
                        pack_b_tile_f64_trans(tile.as_mut_ptr(), b.as_ptr(), ldb, jc, nc, pc, kc);
                    } else {
                        pack_b_tile_f64(tile.as_mut_ptr(), b.as_ptr(), ldb, jc, nc, pc, kc);
                    }
                    let mut ic = 0;
                    let mut pi = 0;
                    while ic < m {
                        let mc = (m - ic).min(F64_MC);
                        // SAFETY: panel `pi` owns
                        // `Mc*KC` elements at its offset; the pack
                        // writes `mc * kc` of them.
                        let panel = panels.as_mut_ptr().add(pi * F64_MC * F64_KC);
                        if jc == 0 {
                            // Rows [ic, ic+mc) × cols [pc, pc+kc]
                            // become contiguous `mc × kc` (the tile
                            // packer with rows/cols swapped).
                            pack_b_tile_f64(panel, a.as_ptr(), lda, pc, kc, ic, mc);
                        }
                        // SAFETY: panel-relative sweep (`pc = 0`,
                        // row stride `kc`); `C` rebased to row `ic`.
                        sweep_f64(
                            0,
                            mc,
                            mc,
                            jc,
                            nc,
                            0,
                            kc,
                            alpha,
                            panel,
                            kc,
                            1,
                            tile.as_ptr(),
                            nc,
                            0,
                            jc,
                            beta_eff,
                            c.as_mut_ptr().add(ic * ldc),
                            ldc,
                        );
                        ic += mc;
                        pi += 1;
                    }
                    jc += nc;
                }
                pc += kc;
            }
        });
    });
}

#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dgemm_serial(
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
    trans_a: bool,
    trans_b: bool,
) {
    if m == 0 || n == 0 {
        return;
    }
    if alpha == 0.0 || k == 0 {
        scale_c_f64(m, n, beta, c.as_mut_ptr(), ldc);
        return;
    }
    // Pure-dimension pre-check first: `l3_bytes()` must not even be
    // evaluated for shapes the apack path rejects on dims alone —
    // thin serial calls never qualify (2026-09-28: the uncached
    // sysfs read here cost ~40% on thin serial Schur updates).
    if st_apack_on()
        && !trans_a
        && m > F64_MC
        && n > F64_NC
        && k > F64_KC
        && st_apack_shape(
            m,
            n,
            k,
            F64_MC,
            F64_KC,
            F64_NC,
            core::mem::size_of::<f64>(),
            sonium_topology::affinity::l3_bytes(),
        )
    {
        dgemm_serial_apack(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_b);
        return;
    }
    let (a_row_stride, a_col_stride) = if trans_a { (1, lda) } else { (lda, 1) };
    // Thread-local tile buffer (see the N-split tasks): a per-call
    // `Vec` mmaps on every call, which dominates when a blocked
    // solver (TRSM/SYRK/LU) issues dozens of small GEMMs per solve.
    super::with_buf(&super::F64_B_PACK, F64_KC * F64_NC, |tile| {
        let mut jc = 0;
        while jc < n {
            let nc = (n - jc).min(F64_NC);
            let mut pc = 0;
            while pc < k {
                let kc = (k - pc).min(F64_KC);
                let beta_eff = if pc == 0 { beta } else { 1.0 };
                // SAFETY: pack overwrites `kc * nc` elements before the sweep
                // reads them. The transposed gather produces the same tile
                // elements from a `Trans` source — no pre-pack needed.
                if trans_b {
                    pack_b_tile_f64_trans(tile.as_mut_ptr(), b.as_ptr(), ldb, jc, nc, pc, kc);
                } else {
                    pack_b_tile_f64(tile.as_mut_ptr(), b.as_ptr(), ldb, jc, nc, pc, kc);
                }
                sweep_f64(
                    0,
                    m,
                    m,
                    jc,
                    nc,
                    pc,
                    kc,
                    alpha,
                    a.as_ptr(),
                    a_row_stride,
                    a_col_stride,
                    tile.as_ptr(),
                    nc,
                    0,
                    jc,
                    beta_eff,
                    c.as_mut_ptr(),
                    ldc,
                );
                pc += kc;
            }
            jc += nc;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Transpose;
    use crate::cpu::blas3::scalar;
    use crate::cpu::parallelism::env_test_lock;
    use approx::assert_relative_eq;

    /// Forced M-split (`SONIUM_GEMM_M_SPLIT`) agrees with the default
    /// decomposition: same kernels, different grouping, so agreement is
    /// expected to tolerance. Locks the previously untested square-M route.
    /// Shapes cover the unpadded panel (384²×512) and the 4 KiB-padded
    /// panel (256×1024×256: padded in both precisions), all above
    /// `PAR_MIN_MACS` so both take a parallel path. The gate is
    /// process-global, but this is its only writer among tests, so
    /// set/remove here is safe (same contract as the `SONIUM_TRSM_FUSED`
    /// test).
    #[test]
    fn gemm_m_split_matches_default_square() {
        if !is_available() {
            return;
        }
        let _guard = env_test_lock();
        for (m, n, k) in [
            (384usize, 384usize, 512usize),
            (256usize, 1024usize, 256usize),
        ] {
            // `Trans` B is stored `n × k`; exercises the transposed-gather
            // panel packs (and their padded stride) too.
            for trans_b in [false, true] {
                let (b_rows, b_cols, ldb) = if trans_b { (n, k, k) } else { (k, n, n) };
                let a_f32: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.013).sin()).collect();
                let b_f32: Vec<f32> = (0..b_rows * b_cols)
                    .map(|i| (i as f32 * 0.017).cos())
                    .collect();
                let c0_f32: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.011).sin()).collect();
                let a_f64: Vec<f64> = a_f32.iter().map(|&x| x as f64).collect();
                let b_f64: Vec<f64> = b_f32.iter().map(|&x| x as f64).collect();
                let c0_f64: Vec<f64> = c0_f32.iter().map(|&x| x as f64).collect();
                let mut default32 = c0_f32.clone();
                let mut default64 = c0_f64.clone();
                unsafe {
                    sgemm(
                        m,
                        n,
                        k,
                        0.7,
                        &a_f32,
                        k,
                        &b_f32,
                        ldb,
                        0.35,
                        &mut default32,
                        n,
                        false,
                        trans_b,
                    );
                    dgemm(
                        m,
                        n,
                        k,
                        0.7,
                        &a_f64,
                        k,
                        &b_f64,
                        ldb,
                        0.35,
                        &mut default64,
                        n,
                        false,
                        trans_b,
                    );
                    // SAFETY: only writer of `SONIUM_GEMM_M_SPLIT` among tests;
                    // concurrent `std::env::var` reads elsewhere serialize on the
                    // internal lock, and every GEMM test asserts to tolerance.
                    std::env::set_var("SONIUM_GEMM_M_SPLIT", "1");
                }
                let mut forced32 = c0_f32.clone();
                let mut forced64 = c0_f64.clone();
                unsafe {
                    sgemm(
                        m,
                        n,
                        k,
                        0.7,
                        &a_f32,
                        k,
                        &b_f32,
                        ldb,
                        0.35,
                        &mut forced32,
                        n,
                        false,
                        trans_b,
                    );
                    dgemm(
                        m,
                        n,
                        k,
                        0.7,
                        &a_f64,
                        k,
                        &b_f64,
                        ldb,
                        0.35,
                        &mut forced64,
                        n,
                        false,
                        trans_b,
                    );
                    std::env::remove_var("SONIUM_GEMM_M_SPLIT");
                }
                for (g, w) in forced32.iter().zip(default32.iter()) {
                    assert_relative_eq!(g, w, epsilon = 1e-4);
                }
                for (g, w) in forced64.iter().zip(default64.iter()) {
                    assert_relative_eq!(g, w, epsilon = 1e-11);
                }
            }
        }
    }

    /// L3-fit band rule (`l3_fit_cpt`): gates (thin-m/narrow-K keep
    /// legacy) plus machine-portable invariants (relative to the real
    /// `l3_total_bytes`, so any topology passes): `cpt` is an
    /// `nr`-multiple ≥ 16 elements, bands fill the pool, fit L3, and
    /// respect min-task economics (one more band would break fit or
    /// economics, unless at the geometry ceiling).
    #[test]
    fn gemm_l3_fit_cpt_gates_and_invariants() {
        // Gates: thin-m and narrow-K keep legacy sizing.
        assert_eq!(l3_fit_cpt(100, 1024, 1024, 4, 16, 16, 150_000_000), None);
        assert_eq!(l3_fit_cpt(1024, 1024, 100, 4, 16, 16, 150_000_000), None);
        assert_eq!(l3_fit_cpt(32, 2048, 2048, 4, 16, 16, 150_000_000), None);
        let Some(l3) = sonium_topology::affinity::l3_total_bytes() else {
            return;
        };
        assert!(l3 > 0 && l3 % 1024 == 0, "sane L3 total: {l3}");
        for (m, n, k, el, nr, workers, min_task) in [
            (
                2048usize,
                2048usize,
                2048usize,
                4usize,
                16usize,
                16usize,
                150_000_000usize,
            ),
            (2048, 2048, 2048, 8, 8, 16, 15_000_000),
            (1024, 1024, 1024, 4, 16, 16, 150_000_000),
            (512, 512, 512, 8, 8, 16, 15_000_000),
            (2048, 128, 2048, 8, 8, 16, 15_000_000),
        ] {
            let cpt = l3_fit_cpt(m, n, k, el, nr, workers, min_task).unwrap();
            assert_eq!(cpt % nr, 0, "cpt nr-aligned at {m}x{n}x{k}");
            assert!(cpt >= 16, "cpt 16-element floor at {m}x{n}x{k}");
            let bands = n.div_ceil(cpt);
            let geom = n.div_ceil(16);
            assert!(bands <= geom, "bands {bands} within geometry {geom}");
            let a_bytes = m * k * el;
            let macs = m * n * k;
            let fit = (l3 / a_bytes).max(1);
            let work = (macs / min_task).max(1);
            // nr-rounding only narrows bands, so fit + economics hold
            // whenever the pool-fill floor is not binding.
            if fit.min(work) >= workers.min(geom) {
                assert!(bands * a_bytes <= l3, "bands×A fits L3");
                assert!(macs / bands >= min_task.min(macs), "min-task economics");
            }
        }
    }

    /// Narrow-band counts on a 256 MiB L3 with 16 workers: narrowest
    /// (`nc = 16`) where A-fit allows (sq1k 64, sq512 32, sq768 48,
    /// r1k512 32, m512n2k 128), A-fit-capped where not (sq2k 16),
    /// full waves only (a 24 target snaps down to 16 when 32 would
    /// breach fit), pool-fill on tiny-n (never below geometry).
    #[test]
    fn gemm_f32_narrow_bands_audit() {
        const L3: usize = 256 << 20;
        for (m, n, k, want) in [
            (1024usize, 1024usize, 1024usize, 64usize),
            (2048, 2048, 2048, 16),
            (512, 512, 512, 32),
            (768, 768, 768, 48),
            (1024, 512, 1024, 32),
            (512, 2048, 1024, 128),
        ] {
            assert_eq!(f32_narrow_bands(m, n, k, 16, L3), want, "{m}x{n}x{k}");
        }
        // Partial-wave target snaps down when up would breach A-fit:
        // 768²×768 at 16 workers with a 96 MiB L3 (fit 42): target 42
        // (up 48 > fit) -> 32.
        assert_eq!(f32_narrow_bands(768, 768, 768, 16, 96 << 20), 32);
        // Tiny-n: geometry binds (4 bands), pool-fill cannot exceed it.
        assert_eq!(f32_narrow_bands(1024, 64, 1024, 16, L3), 4);
        // Single-pass-L1 skip: k128 legacy (kc 256, nc 64, B 32 KiB)
        // skips; k256 (B 64 KiB) and multi-pass sq1k do not.
        assert!(f32_narrow_skip_1pass_l1(128, 256, 64));
        assert!(!f32_narrow_skip_1pass_l1(256, 256, 64));
        assert!(!f32_narrow_skip_1pass_l1(1024, 512, 64));
        // Full-wave invariant on a shape sweep (16 workers).
        for (m, n, k) in [
            (1024usize, 1024usize, 1024usize),
            (2048, 2048, 2048),
            (512, 512, 512),
            (768, 768, 768),
            (1024, 512, 1024),
            (512, 2048, 1024),
            (1024, 257, 1024),
            (1024, 1000, 1024),
        ] {
            let bands = f32_narrow_bands(m, n, k, 16, L3);
            let geom = n.div_ceil(16);
            assert!(bands <= geom, "bands {bands} within geometry {geom}");
            assert!(
                bands.is_multiple_of(16) || bands == geom,
                "bands {bands} full-wave or geometry-capped"
            );
        }
    }

    /// f64 narrow-band counts on a 256 MiB L3 with 16 workers (mirrors
    /// `gemm_f32_narrow_bands_audit`): narrowest (`nc = 8`) where the
    /// 8x-L3 fit allows (sq1k 128, sq512 32, sq768 96, r256n1k 64,
    /// r1kn256 32), fit-capped where not (sq2k 64 — the A cap vetoes
    /// sq2k at the `cpt` level above this), full waves only, pool-fill
    /// on tiny-n (never below geometry).
    #[test]
    fn gemm_f64_narrow_bands_audit() {
        const L3: usize = 256 << 20;
        for (m, n, k, want) in [
            (1024usize, 1024usize, 1024usize, 128usize),
            (2048, 2048, 2048, 64),
            (512, 512, 512, 32),
            (768, 768, 768, 96),
            (256, 1024, 1024, 64),
            (1024, 256, 1024, 32),
        ] {
            assert_eq!(f64_narrow_bands(m, n, k, 16, L3), want, "{m}x{n}x{k}");
        }
        // Partial-wave target snaps down when up would breach fit:
        // 768²×768 at 16 workers with a 24 MiB L3 (fit 42): target 42
        // (up 48 > fit) -> 32.
        assert_eq!(f64_narrow_bands(768, 768, 768, 16, 24 << 20), 32);
        // Tiny-n: geometry binds (8 bands), pool-fill cannot exceed it.
        assert_eq!(f64_narrow_bands(1024, 64, 1024, 16, L3), 8);
        // Single-pass-L1 skip: k128 legacy (kc 256, nc 32, B 32 KiB)
        // skips; k256 (B 64 KiB) and multi-pass sq1k do not.
        assert!(f64_narrow_skip_1pass_l1(128, 256, 32));
        assert!(!f64_narrow_skip_1pass_l1(256, 256, 32));
        assert!(!f64_narrow_skip_1pass_l1(1024, 512, 32));
        // Full-wave invariant on a shape sweep (16 workers).
        for (m, n, k) in [
            (1024usize, 1024usize, 1024usize),
            (2048, 2048, 2048),
            (512, 512, 512),
            (768, 768, 768),
            (256, 1024, 1024),
            (1024, 256, 1024),
            (1024, 257, 1024),
            (1024, 1000, 1024),
        ] {
            let bands = f64_narrow_bands(m, n, k, 16, L3);
            let geom = n.div_ceil(8);
            assert!(bands <= geom, "bands {bands} within geometry {geom}");
            assert!(
                bands.is_multiple_of(16) || bands == geom,
                "bands {bands} full-wave or geometry-capped"
            );
        }
    }

    /// Rank-`k` M-split route (`use_m_split_rank_k`) agrees with the
    /// default decomposition on shapes above the gate: `704x992x32`
    /// (Schur/LU trailing update), `896x1024x128` (TRSM `nb = 128`
    /// update, second gate clause), and `640x2048x96` (newly admitted by
    /// the raised panel ceiling: `k*n = 196K` elements), all with
    /// alpha = -1, beta = 1.
    /// Default takes the shared-`B`-pack row-band path,
    /// `SONIUM_GEMM_NO_RANKK_M=1` the default parallel path. Same kernels,
    /// different grouping, so agreement is expected to tolerance; locks
    /// the routes the threshold admits. The gate is process-global, but
    /// this is its only writer among tests, so set/remove here is safe
    /// (same contract as the `SONIUM_GEMM_M_SPLIT` test).
    /// Single-threaded runners take the same path in both runs and pass
    /// trivially.
    #[test]
    fn gemm_rankk_m_split_matches_off_path() {
        if !is_available() {
            return;
        }
        let _guard = env_test_lock();
        for (m, n, k) in [
            (704usize, 992usize, 32usize),
            (896usize, 1024usize, 128usize),
            (640usize, 2048usize, 96usize),
        ] {
            let a_f32: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.013).sin()).collect();
            let b_f32: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.017).cos()).collect();
            let c0_f32: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.011).sin()).collect();
            let a_f64: Vec<f64> = a_f32.iter().map(|&x| x as f64).collect();
            let b_f64: Vec<f64> = b_f32.iter().map(|&x| x as f64).collect();
            let c0_f64: Vec<f64> = c0_f32.iter().map(|&x| x as f64).collect();
            let mut default32 = c0_f32.clone();
            let mut default64 = c0_f64.clone();
            unsafe {
                sgemm(
                    m,
                    n,
                    k,
                    -1.0,
                    &a_f32,
                    k,
                    &b_f32,
                    n,
                    1.0,
                    &mut default32,
                    n,
                    false,
                    false,
                );
                dgemm(
                    m,
                    n,
                    k,
                    -1.0,
                    &a_f64,
                    k,
                    &b_f64,
                    n,
                    1.0,
                    &mut default64,
                    n,
                    false,
                    false,
                );
                // SAFETY: only writer of `SONIUM_GEMM_NO_RANKK_M` among tests;
                // concurrent `std::env::var` reads elsewhere serialize on the
                // internal lock, and every GEMM test asserts to tolerance.
                std::env::set_var("SONIUM_GEMM_NO_RANKK_M", "1");
            }
            let mut off32 = c0_f32.clone();
            let mut off64 = c0_f64.clone();
            unsafe {
                sgemm(
                    m, n, k, -1.0, &a_f32, k, &b_f32, n, 1.0, &mut off32, n, false, false,
                );
                dgemm(
                    m, n, k, -1.0, &a_f64, k, &b_f64, n, 1.0, &mut off64, n, false, false,
                );
                std::env::remove_var("SONIUM_GEMM_NO_RANKK_M");
            }
            for (g, w) in default32.iter().zip(off32.iter()) {
                assert_relative_eq!(g, w, epsilon = 1e-4);
            }
            for (g, w) in default64.iter().zip(off64.iter()) {
                assert_relative_eq!(g, w, epsilon = 1e-11);
            }
        }
    }

    /// Single-pass parallel floor: `124x992x32` (3.9M MACs) sits below
    /// `PAR_MIN_MACS` but fits one Kc panel, so the M and N gates must
    /// admit it (the 2D gate's grid checks are pool-sized and not pinned
    /// here). Parity against the scalar reference guards the newly
    /// newly admitted parallel route against races. `16^3` stays serial
    /// under any thread count (4K MACs < single-pass floor). The positive
    /// asserts need more than one rayon worker, hence the early return.
    #[test]
    fn small_single_pass_parallel_parity() {
        if !is_available() || rayon::current_num_threads() <= 1 {
            return;
        }
        let (m, n, k) = (124usize, 992usize, 32usize);
        assert!(
            use_parallel(m, n, k, F64_KC),
            "single-pass shape must engage the parallel path"
        );
        assert!(
            use_parallel_n(m, n, k, F64_KC),
            "single-pass shape must engage the N-split path"
        );
        assert!(
            !use_parallel(16, 16, 16, F64_KC),
            "sub-floor shape must stay serial"
        );
        // 1M-floor trial fallout (see `PAR_MIN_MACS_SINGLE_PASS`): 1-2M
        // MACs lose or flip bimodally (62x512x32 -16%, 128^3 win/loss),
        // so both stay serial under any thread count.
        assert!(
            !use_parallel(62, 512, 32, F64_KC),
            "1M-MAC shape must stay serial"
        );
        assert!(
            !use_parallel(128, 128, 128, F64_KC),
            "2.1M-MAC shape must stay serial"
        );
        let a32: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.0013).sin()).collect();
        let b32: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.0017).cos()).collect();
        let c032: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.0011).sin()).collect();
        let mut got32 = c032.clone();
        let mut want32 = c032;
        unsafe {
            sgemm(
                m, n, k, -1.0, &a32, k, &b32, n, 1.0, &mut got32, n, false, false,
            );
        }
        scalar::gemm_f32(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            -1.0,
            &a32,
            k,
            &b32,
            n,
            1.0,
            &mut want32,
            n,
        );
        for (g, w) in got32.iter().zip(want32.iter()) {
            assert_relative_eq!(g, w, max_relative = 1e-4, epsilon = 1e-5);
        }

        let a64: Vec<f64> = (0..m * k).map(|i| (i as f64 * 0.0013).sin()).collect();
        let b64: Vec<f64> = (0..k * n).map(|i| (i as f64 * 0.0017).cos()).collect();
        let c064: Vec<f64> = (0..m * n).map(|i| (i as f64 * 0.0011).sin()).collect();
        let mut got64 = c064.clone();
        let mut want64 = c064;
        unsafe {
            dgemm(
                m, n, k, -1.0, &a64, k, &b64, n, 1.0, &mut got64, n, false, false,
            );
        }
        scalar::gemm_f64(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            -1.0,
            &a64,
            k,
            &b64,
            n,
            1.0,
            &mut want64,
            n,
        );
        for (g, w) in got64.iter().zip(want64.iter()) {
            assert_relative_eq!(g, w, max_relative = 1e-11, epsilon = 1e-12);
        }
    }

    /// Non-final row bands must not write past `r1`: 6-row stepping broke
    /// the old 4-aligned band invariant, and a `(m - i0)` tail cap once
    /// let a height-8 band trample its neighbour's rows (caught only at
    /// high thread counts by the M-split parity test). Sentinel design:
    /// C is NaN-filled, a `[0, 8)` band of a 16-row problem runs with
    /// beta = 0, and rows `8..` must stay NaN while rows `0..8` match a
    /// full reference GEMM. Deterministic and single-threaded — the
    /// overshoot wrote correct values, so only the sentinel (not value
    /// comparison) catches it.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn sweep_band_r1_check() {
        if !is_available() {
            return;
        }
        // k = 96 > 64 keeps the rowband fast path off so the general
        // 6-stepping loop (not the 4-wide rowband) runs this band.
        let (m, n, k) = (16usize, 16usize, 96usize);
        let a: Vec<f64> = (0..m * k).map(|i| (i as f64 * 0.013).sin()).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i as f64 * 0.017).cos()).collect();
        let mut band = vec![f64::NAN; m * n];
        sweep_f64(
            0,
            8,
            m,
            0,
            n,
            0,
            k,
            1.0,
            a.as_ptr(),
            k,
            1,
            b.as_ptr(),
            n,
            0,
            0,
            0.0,
            band.as_mut_ptr(),
            n,
        );
        for (i, v) in band.iter().enumerate() {
            if i / n < 8 {
                assert!(v.is_finite(), "band row lost at index {i}");
            } else {
                assert!(v.is_nan(), "band overshoot wrote index {i}: {v}");
            }
        }
        let mut full = vec![0.0; m * n];
        unsafe {
            dgemm(m, n, k, 1.0, &a, k, &b, n, 0.0, &mut full, n, false, false);
        }
        for i in 0..8 * n {
            assert_relative_eq!(band[i], full[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn sweep_f64_nonfinal_band_respects_r1() {
        // SAFETY: x86_64-only module; `sweep_band_r1_check` gates on
        // AVX2/FMA availability and only touches its own buffers.
        unsafe {
            sweep_band_r1_check();
        }
    }

    /// f32 twin of the `sweep_f64` sentinel above: a `[0, 8)` band of a
    /// 16-row problem must leave rows `8..` NaN while rows `0..8` match a
    /// full reference GEMM. Guards the 6-stepping tail cap in `sweep_f32`.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn sweep_band_r1_check_f32() {
        if !is_available() {
            return;
        }
        let (m, n, k) = (16usize, 16usize, 96usize);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.013).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.017).cos()).collect();
        let mut band = vec![f32::NAN; m * n];
        sweep_f32(
            0,
            8,
            m,
            0,
            n,
            0,
            k,
            1.0,
            a.as_ptr(),
            k,
            1,
            b.as_ptr(),
            n,
            0,
            0,
            0.0,
            band.as_mut_ptr(),
            n,
        );
        for (i, v) in band.iter().enumerate() {
            if i / n < 8 {
                assert!(v.is_finite(), "band row lost at index {i}");
            } else {
                assert!(v.is_nan(), "band overshoot wrote index {i}: {v}");
            }
        }
        let mut full = vec![0.0; m * n];
        unsafe {
            sgemm(m, n, k, 1.0, &a, k, &b, n, 0.0, &mut full, n, false, false);
        }
        for i in 0..8 * n {
            assert_relative_eq!(band[i], full[i], epsilon = 1e-4);
        }
    }

    #[test]
    fn sweep_f32_nonfinal_band_respects_r1() {
        // SAFETY: x86_64-only module; `sweep_band_r1_check_f32` gates on
        // AVX2/FMA availability and only touches its own buffers.
        unsafe {
            sweep_band_r1_check_f32();
        }
    }

    /// Odd sizes exercise the row tail (37 % 4), the column tail (41 % 16 /
    /// 41 % 8), and the Kc loop stays single-pass; non-tight leading dims
    /// guard against stride mixups. Non-trivial alpha/beta, TransA variant
    /// included.
    #[test]
    fn avx2_sgemm_matches_scalar_with_tails() {
        if !is_available() {
            return;
        }
        let (m, n, k) = (37, 41, 29);
        let lda = k + 5;
        let ldb = n + 7;
        let ldc = n + 3;
        let a: Vec<f32> = (0..m * lda).map(|i| (i as f32 * 0.013).sin()).collect();
        let b: Vec<f32> = (0..k * ldb).map(|i| (i as f32 * 0.017).cos()).collect();
        let c0: Vec<f32> = (0..m * ldc).map(|i| (i as f32 * 0.011).sin()).collect();
        let mut got = c0.clone();
        let mut want = c0;
        unsafe {
            sgemm(
                m, n, k, 0.7, &a, lda, &b, ldb, 0.35, &mut got, ldc, false, false,
            );
        }
        scalar::gemm_f32(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            0.7,
            &a,
            lda,
            &b,
            ldb,
            0.35,
            &mut want,
            ldc,
        );
        for i in 0..m {
            for j in 0..n {
                assert_relative_eq!(
                    got[i * ldc + j],
                    want[i * ldc + j],
                    max_relative = 1e-4,
                    epsilon = 1e-5
                );
            }
        }
    }

    #[test]
    fn avx2_dgemm_matches_scalar_with_tails_and_trans_a() {
        if !is_available() {
            return;
        }
        let (m, n, k) = (37, 41, 29);
        // TransA: stored A is k × m.
        let lda = m + 5;
        let ldb = n + 7;
        let ldc = n + 3;
        let a: Vec<f64> = (0..k * lda).map(|i| (i as f64 * 0.013).sin()).collect();
        let b: Vec<f64> = (0..k * ldb).map(|i| (i as f64 * 0.017).cos()).collect();
        let c0: Vec<f64> = (0..m * ldc).map(|i| (i as f64 * 0.011).sin()).collect();
        let mut got = c0.clone();
        let mut want = c0;
        unsafe {
            dgemm(
                m, n, k, -0.4, &a, lda, &b, ldb, 0.85, &mut got, ldc, true, false,
            );
        }
        scalar::gemm_f64(
            Transpose::Trans,
            Transpose::NoTrans,
            m,
            n,
            k,
            -0.4,
            &a,
            lda,
            &b,
            ldb,
            0.85,
            &mut want,
            ldc,
        );
        for i in 0..m {
            for j in 0..n {
                assert_relative_eq!(
                    got[i * ldc + j],
                    want[i * ldc + j],
                    max_relative = 1e-11,
                    epsilon = 1e-12
                );
            }
        }
    }

    /// Cross the Kc cap (k > F32_KC = 256) so the multi-pass beta=1
    /// accumulation path runs, plus a beta = 0 first pass.
    #[test]
    fn avx2_sgemm_kc_multi_pass_matches_scalar() {
        if !is_available() {
            return;
        }
        let (m, n, k) = (9, 33, 600);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.0013).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.0017).cos()).collect();
        let c0: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.0011).sin()).collect();
        let mut got = c0.clone();
        let mut want = c0;
        unsafe {
            sgemm(m, n, k, 0.9, &a, k, &b, n, 0.0, &mut got, n, false, false);
        }
        scalar::gemm_f32(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            0.9,
            &a,
            k,
            &b,
            n,
            0.0,
            &mut want,
            n,
        );
        for (g, w) in got.iter().zip(want.iter()) {
            assert_relative_eq!(g, w, max_relative = 1e-4, epsilon = 1e-5);
        }
    }

    /// MT pack shape gate: thin-m gathers, m>=512 packs (tapackm
    /// crossover, n=k=2048); narrow-n gathers below 512 (n=256 loses
    /// per the original tuning; n>=512 wins per the wide-Kc re-tune:
    /// 1024x512x1024 packs at 0.79x, 768^3 at 0.86x); small-A
    /// gathers (512^3T ties 1.00x); and the MAC/buffer clauses still
    /// reject degenerate calls.
    #[test]
    fn pack_a_mt_shape_gates_thin_m() {
        assert!(!pack_a_mt_shape(32, 2048, 2048));
        assert!(!pack_a_mt_shape(256, 2048, 2048));
        assert!(pack_a_mt_shape(512, 2048, 2048));
        assert!(!pack_a_mt_shape(1024, 256, 1024));
        assert!(pack_a_mt_shape(1024, 512, 1024));
        assert!(pack_a_mt_shape(768, 768, 768));
        assert!(!pack_a_mt_shape(512, 512, 512));
        assert!(pack_a_mt_shape(1024, 1024, 1024));
        assert!(!pack_a_mt_shape(512, 8, 8));
        assert!(!pack_a_mt_shape(8192, 8192, 8192));
    }

    /// MT trans-A vs the scalar reference, both precisions. The 768^3
    /// shape clears the f64 MT pack gates (`m`/`n` over the floors,
    /// `m*k` in `PACK_A_MT_MIN_A_ELS..=PACK_A_MT_MAX_ELS`, MACs over
    /// `PACK_A_MT_MIN_MACS`) so f64 runs the pack-once-shared recursion
    /// under the test pool while f32 runs the strided-gather MT path
    /// (2.25 MB `A` under its 4 MB pack gate). Non-tight `lda` guards
    /// the pack/gather indexing.
    #[test]
    fn avx2_trans_a_matches_scalar_mt() {
        if !is_available() {
            return;
        }
        let (m, n, k) = (768usize, 768usize, 768usize);
        assert!(
            pack_a_mt_shape(m, n, k),
            "test size must engage the pack paths"
        );
        let lda = m + 7;
        // f32 MT gather path (1 MB `A` under the 4 MB pack gate).
        let a32: Vec<f32> = (0..k * lda).map(|i| (i as f32 * 0.0013).sin()).collect();
        let b32: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.0017).cos()).collect();
        let c032: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.0011).sin()).collect();
        let mut got32 = c032.clone();
        let mut want32 = c032;
        unsafe {
            sgemm(
                m, n, k, 0.9, &a32, lda, &b32, n, 0.35, &mut got32, n, true, false,
            );
        }
        scalar::gemm_f32(
            Transpose::Trans,
            Transpose::NoTrans,
            m,
            n,
            k,
            0.9,
            &a32,
            lda,
            &b32,
            n,
            0.35,
            &mut want32,
            n,
        );
        for (g, w) in got32.iter().zip(want32.iter()) {
            assert_relative_eq!(g, w, max_relative = 1e-4, epsilon = 1e-5);
        }
        // f64 MT pack path.
        let a64: Vec<f64> = (0..k * lda).map(|i| (i as f64 * 0.0013).sin()).collect();
        let b64: Vec<f64> = (0..k * n).map(|i| (i as f64 * 0.0017).cos()).collect();
        let c064: Vec<f64> = (0..m * n).map(|i| (i as f64 * 0.0011).sin()).collect();
        let mut got64 = c064.clone();
        let mut want64 = c064;
        unsafe {
            dgemm(
                m, n, k, -0.4, &a64, lda, &b64, n, 0.85, &mut got64, n, true, false,
            );
        }
        scalar::gemm_f64(
            Transpose::Trans,
            Transpose::NoTrans,
            m,
            n,
            k,
            -0.4,
            &a64,
            lda,
            &b64,
            n,
            0.85,
            &mut want64,
            n,
        );
        for (g, w) in got64.iter().zip(want64.iter()) {
            assert_relative_eq!(g, w, max_relative = 1e-11, epsilon = 1e-12);
        }
    }

    /// f64 MT pack-once-shared vs the strided-gather arm on the 768^3
    /// trans-A shape: both arms must agree bit-for-bit (the packed
    /// tiles feed the sweep the same elements in the same order).
    /// Arms selected by explicit `=1`/`=0` — `remove_var` would run the
    /// default-on pack twice and prove nothing (same trap as the GEMV
    /// pinned test).
    #[test]
    fn avx2_trans_a_mt_pack_matches_gather_bitwise() {
        if !is_available() {
            return;
        }
        let (m, n, k) = (768usize, 768usize, 768usize);
        assert!(
            pack_a_mt_shape(m, n, k),
            "test size must engage the MT pack gate"
        );
        // Lock AFTER the engagement assert: asserting under the guard
        // would poison the crate-wide env lock on failure and cascade
        // into every other env test.
        let _guard = env_test_lock();
        let lda = m + 7;
        let a64: Vec<f64> = (0..k * lda).map(|i| (i as f64 * 0.0013).sin()).collect();
        let b64: Vec<f64> = (0..k * n).map(|i| (i as f64 * 0.0017).cos()).collect();
        let c064: Vec<f64> = (0..m * n).map(|i| (i as f64 * 0.0011).sin()).collect();
        let mut packed = c064.clone();
        let mut gathered = c064;
        // SAFETY: only writer of `SONIUM_GEMM_PACK_A_MT` among tests;
        // concurrent `std::env::var` reads elsewhere serialize on the
        // internal lock, and every GEMM test asserts to tolerance.
        unsafe {
            std::env::set_var("SONIUM_GEMM_PACK_A_MT", "1");
        }
        unsafe {
            dgemm(
                m,
                n,
                k,
                -0.4,
                &a64,
                lda,
                &b64,
                n,
                0.85,
                &mut packed,
                n,
                true,
                false,
            );
        }
        // SAFETY: as above; selects the gather arm.
        unsafe {
            std::env::set_var("SONIUM_GEMM_PACK_A_MT", "0");
        }
        unsafe {
            dgemm(
                m,
                n,
                k,
                -0.4,
                &a64,
                lda,
                &b64,
                n,
                0.85,
                &mut gathered,
                n,
                true,
                false,
            );
        }
        // SAFETY: as above; restores the default-on state.
        unsafe {
            std::env::remove_var("SONIUM_GEMM_PACK_A_MT");
        }
        assert_eq!(packed, gathered, "MT pack must match gather bitwise");
    }

    /// f64 MT pack TLS buffer vs the fresh-`vec!` buffer on the same
    /// 768^3 trans-A shape: both buffers must agree bit-for-bit
    /// (same parallel pack, same recursion, only the allocation differs).
    /// Arms selected by explicit `=1`/`=0` on `SONIUM_GEMM_PACK_A_MT_TLS`
    /// with `SONIUM_GEMM_PACK_A_MT=1` fixed; both vars restored after.
    #[test]
    fn avx2_trans_a_mt_tls_pack_matches_fresh_bitwise() {
        if !is_available() {
            return;
        }
        let (m, n, k) = (768usize, 768usize, 768usize);
        assert!(
            pack_a_mt_shape(m, n, k),
            "test size must engage the MT pack gate"
        );
        // Lock after the engagement assert (see the pack-vs-gather
        // test: asserting under the guard poisons the env lock).
        let _guard = env_test_lock();
        let lda = m + 7;
        let a64: Vec<f64> = (0..k * lda).map(|i| (i as f64 * 0.0013).sin()).collect();
        let b64: Vec<f64> = (0..k * n).map(|i| (i as f64 * 0.0017).cos()).collect();
        let c064: Vec<f64> = (0..m * n).map(|i| (i as f64 * 0.0011).sin()).collect();
        let mut tls = c064.clone();
        let mut fresh = c064;
        // SAFETY: only writer of the pack env vars among tests (same
        // convention as the pack-vs-gather test above).
        unsafe {
            std::env::set_var("SONIUM_GEMM_PACK_A_MT", "1");
            std::env::set_var("SONIUM_GEMM_PACK_A_MT_TLS", "1");
        }
        unsafe {
            dgemm(
                m, n, k, -0.4, &a64, lda, &b64, n, 0.85, &mut tls, n, true, false,
            );
        }
        unsafe {
            std::env::set_var("SONIUM_GEMM_PACK_A_MT_TLS", "0");
        }
        unsafe {
            dgemm(
                m, n, k, -0.4, &a64, lda, &b64, n, 0.85, &mut fresh, n, true, false,
            );
        }
        unsafe {
            std::env::remove_var("SONIUM_GEMM_PACK_A_MT");
            std::env::remove_var("SONIUM_GEMM_PACK_A_MT_TLS");
        }
        assert_eq!(tls, fresh, "TLS pack must match fresh-vec pack bitwise");
    }

    /// Fused-pass M-split gate audit (see [`m_fuse_kc`]): fuses the tall
    /// privb shapes at the legacy pass width (f32 tall `kc = 256`, f64
    /// tall `kc = 256` via the L2 cap shrinking `kc0 = 512`), two-pass
    /// shapes fuse, and tails / single passes / oversize panels /
    /// degenerate inputs keep the legacy loop.
    #[test]
    fn gemm_m_fuse_kc_audit() {
        // f32 2048x128x2048 (scored tall cell): kc0 256 fits the 128 KiB
        // budget exactly (256*128*4), no tail.
        assert_eq!(
            m_fuse_kc(2048, 256, panel_stride(128, 4), 4, PRIVB_L2_BUDGET_F32),
            Some(256)
        );
        // f64 2048x128x2048: kc0 512 shrinks to the 256 KiB cap (256),
        // which fits exactly (256*128*8), no tail.
        assert_eq!(
            m_fuse_kc(2048, 512, panel_stride(128, 8), 8, PRIVB_L2_BUDGET_F64),
            Some(256)
        );
        // Two-pass shape fuses at the same width.
        assert_eq!(
            m_fuse_kc(512, 256, panel_stride(128, 4), 4, PRIVB_L2_BUDGET_F32),
            Some(256)
        );
        // Kc tail keeps the legacy loop (short final pass).
        assert_eq!(
            m_fuse_kc(2000, 256, panel_stride(128, 4), 4, PRIVB_L2_BUDGET_F32),
            None
        );
        // Single pass keeps legacy (nothing to fuse; kc_try < 128 floor).
        assert_eq!(
            m_fuse_kc(100, 100, panel_stride(128, 4), 4, PRIVB_L2_BUDGET_F32),
            None
        );
        // Oversize panel (n = 1024, padded stride 1040): kc_try 128 needs
        // 532 KiB > 128 KiB budget — legacy shared panel.
        assert_eq!(
            m_fuse_kc(2048, 256, panel_stride(1024, 4), 4, PRIVB_L2_BUDGET_F32),
            None
        );
        // Degenerate inputs never fuse.
        assert_eq!(
            m_fuse_kc(0, 256, panel_stride(128, 4), 4, PRIVB_L2_BUDGET_F32),
            None
        );
        assert_eq!(
            m_fuse_kc(2048, 0, panel_stride(128, 4), 4, PRIVB_L2_BUDGET_F32),
            None
        );
        assert_eq!(
            m_fuse_kc(2048, 256, panel_stride(128, 4), 0, PRIVB_L2_BUDGET_F32),
            None
        );
    }

    /// f32 fused-pass M-split vs the pass-per-scope loop on a tall privb
    /// shape (1024x64x1024, 4 passes): both must agree bit-for-bit (same
    /// packs, same per-element sweep order). Arms selected by unset/`=0`
    /// — the switch is default-on. Runs both trans-A variants (the fused
    /// task gathers strided A exactly like the legacy task).
    #[test]
    fn avx2_m_fuse_matches_legacy_bitwise_f32() {
        if !is_available() {
            return;
        }
        let (m, n, k) = (1024usize, 64usize, 1024usize);
        assert!(
            use_parallel(m, n, k, F32_KC),
            "test size must engage the M-split path"
        );
        assert!(
            !use_parallel_n(m, n, k, F32_KC),
            "test size must skip the N-split path"
        );
        assert!(
            m_fuse_kc(k, k.min(F32_KC), panel_stride(n, 4), 4, PRIVB_L2_BUDGET_F32).is_some(),
            "test size must engage the fused M-split gate"
        );
        // Lock AFTER the engagement asserts (see the MT-pack test:
        // asserting under the guard poisons the crate-wide env lock).
        let _guard = env_test_lock();
        for trans_a in [false, true] {
            let (a_len, lda) = if trans_a { (k * m, m) } else { (m * k, k) };
            let a32: Vec<f32> = (0..a_len).map(|i| (i as f32 * 0.0013).sin()).collect();
            let b32: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.0017).cos()).collect();
            let c032: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.0011).sin()).collect();
            let mut fused = c032.clone();
            let mut legacy = c032;
            // SAFETY: only writer of `SONIUM_GEMM_MFUSE` among tests;
            // fused/legacy are bitwise identical, so a concurrent reader
            // observes identical numbers either way.
            unsafe {
                std::env::remove_var("SONIUM_GEMM_MFUSE");
            }
            unsafe {
                sgemm(
                    m, n, k, 0.9, &a32, lda, &b32, n, 0.35, &mut fused, n, trans_a, false,
                );
            }
            unsafe {
                std::env::set_var("SONIUM_GEMM_MFUSE", "0");
            }
            unsafe {
                sgemm(
                    m,
                    n,
                    k,
                    0.9,
                    &a32,
                    lda,
                    &b32,
                    n,
                    0.35,
                    &mut legacy,
                    n,
                    trans_a,
                    false,
                );
            }
            unsafe {
                std::env::remove_var("SONIUM_GEMM_MFUSE");
            }
            assert_eq!(
                fused, legacy,
                "fused M-split must match legacy bitwise (trans_a = {trans_a})"
            );
        }
    }

    /// f64 twin of [`avx2_m_fuse_matches_legacy_bitwise_f32`] on the same
    /// 1024x64x1024 shape (2 passes at the fused `kc = 512`): fused and
    /// legacy agree bit-for-bit under both trans-A variants.
    #[test]
    fn avx2_m_fuse_matches_legacy_bitwise_f64() {
        if !is_available() {
            return;
        }
        let (m, n, k) = (1024usize, 64usize, 1024usize);
        assert!(
            use_parallel(m, n, k, F64_KC),
            "test size must engage the M-split path"
        );
        assert!(
            !use_parallel_n(m, n, k, F64_KC),
            "test size must skip the N-split path"
        );
        assert_eq!(
            m_fuse_kc(
                k,
                k.min(F64_KC_M),
                panel_stride(n, 8),
                8,
                PRIVB_L2_BUDGET_F64
            ),
            Some(512),
            "test size must engage the fused M-split gate at kc = 512"
        );
        let _guard = env_test_lock();
        for trans_a in [false, true] {
            let (a_len, lda) = if trans_a { (k * m, m) } else { (m * k, k) };
            let a64: Vec<f64> = (0..a_len).map(|i| (i as f64 * 0.0013).sin()).collect();
            let b64: Vec<f64> = (0..k * n).map(|i| (i as f64 * 0.0017).cos()).collect();
            let c064: Vec<f64> = (0..m * n).map(|i| (i as f64 * 0.0011).sin()).collect();
            let mut fused = c064.clone();
            let mut legacy = c064;
            // SAFETY: only writer of `SONIUM_GEMM_MFUSE` among tests (same
            // convention as the f32 twin above).
            unsafe {
                std::env::remove_var("SONIUM_GEMM_MFUSE");
            }
            unsafe {
                dgemm(
                    m, n, k, -0.4, &a64, lda, &b64, n, 0.85, &mut fused, n, trans_a, false,
                );
            }
            unsafe {
                std::env::set_var("SONIUM_GEMM_MFUSE", "0");
            }
            unsafe {
                dgemm(
                    m,
                    n,
                    k,
                    -0.4,
                    &a64,
                    lda,
                    &b64,
                    n,
                    0.85,
                    &mut legacy,
                    n,
                    trans_a,
                    false,
                );
            }
            unsafe {
                std::env::remove_var("SONIUM_GEMM_MFUSE");
            }
            assert_eq!(
                fused, legacy,
                "fused M-split must match legacy bitwise (trans_a = {trans_a})"
            );
        }
    }

    /// f32 N-split shared `A^T` pack vs the strided-gather arm on the
    /// same 2048³ trans-A shape (16 MiB `A`, above the pack gate):
    /// both arms must agree bit-for-bit (the packed buffer feeds the
    /// sweep the same elements in the same order). Arms selected by
    /// unset/`=0` — the switch is default-on. Shape asserts the
    /// N-split gate (the only hooked path) and the footprint gate;
    /// other paths never read the switch.
    #[test]
    fn avx2_trans_a_n_pack_matches_gather_bitwise() {
        if !is_available() {
            return;
        }
        let _guard = env_test_lock();
        let (m, n, k) = (2048usize, 2048usize, 2048usize);
        assert!(
            use_parallel_n(m, n, k, F32_KC),
            "test size must engage the N-split path"
        );
        assert!(apack_on(m, k), "test size must engage the pack gate");
        assert!(!apack_on(512, 512), "1 MiB A must keep gathers");
        assert!(apack_on(1024, 1024), "4 MiB A must pack");
        assert!(apack_on(3072, 1024), "12 MiB A must pack");
        let lda = m + 7;
        let a32: Vec<f32> = (0..k * lda).map(|i| (i as f32 * 0.0013).sin()).collect();
        let b32: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.0017).cos()).collect();
        let c032: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.0011).sin()).collect();
        let mut packed = c032.clone();
        let mut gathered = c032;
        // SAFETY: only writer of `SONIUM_GEMM_APACK` among tests (same
        // convention as the f64 MT-pack test above).
        unsafe {
            std::env::remove_var("SONIUM_GEMM_APACK");
        }
        unsafe {
            sgemm(
                m,
                n,
                k,
                0.9,
                &a32,
                lda,
                &b32,
                n,
                0.35,
                &mut packed,
                n,
                true,
                false,
            );
        }
        // SAFETY: as above; selects the gather arm.
        unsafe {
            std::env::set_var("SONIUM_GEMM_APACK", "0");
        }
        unsafe {
            sgemm(
                m,
                n,
                k,
                0.9,
                &a32,
                lda,
                &b32,
                n,
                0.35,
                &mut gathered,
                n,
                true,
                false,
            );
        }
        // SAFETY: as above; restores the default-on state.
        unsafe {
            std::env::remove_var("SONIUM_GEMM_APACK");
        }
        assert_eq!(packed, gathered, "N-split pack must match gather bitwise");
    }

    /// f32 N-split parallel pack vs the serial pack on the 2048×1024×2048
    /// trans-A shape (16 MiB `A`, above the pack gate): both packs must
    /// agree bit-for-bit (same elements, same order — only the row-band
    /// split differs). Arms selected by explicit `=1`/`=0` on
    /// `SONIUM_GEMM_APACK_PAR` with `SONIUM_GEMM_APACK` unset (default
    /// on); the PAR var is removed after.
    #[test]
    fn avx2_trans_a_n_par_pack_matches_serial_bitwise() {
        if !is_available() {
            return;
        }
        let _guard = env_test_lock();
        let (m, n, k) = (2048usize, 1024usize, 2048usize);
        assert!(
            use_parallel_n(m, n, k, F32_KC),
            "test size must engage the N-split path"
        );
        assert!(apack_on(m, k), "test size must engage the pack gate");
        let lda = m + 7;
        let a32: Vec<f32> = (0..k * lda).map(|i| (i as f32 * 0.0013).sin()).collect();
        let b32: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.0017).cos()).collect();
        let c032: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.0011).sin()).collect();
        let mut par = c032.clone();
        let mut serial = c032;
        // SAFETY: only writer of `SONIUM_GEMM_APACK_PAR` among tests
        // (same convention as the pack-vs-gather test above).
        unsafe {
            std::env::set_var("SONIUM_GEMM_APACK_PAR", "1");
        }
        unsafe {
            sgemm(
                m, n, k, 0.9, &a32, lda, &b32, n, 0.35, &mut par, n, true, false,
            );
        }
        unsafe {
            std::env::set_var("SONIUM_GEMM_APACK_PAR", "0");
        }
        unsafe {
            sgemm(
                m,
                n,
                k,
                0.9,
                &a32,
                lda,
                &b32,
                n,
                0.35,
                &mut serial,
                n,
                true,
                false,
            );
        }
        unsafe {
            std::env::remove_var("SONIUM_GEMM_APACK_PAR");
        }
        assert_eq!(par, serial, "parallel pack must match serial bitwise");
    }

    /// Serial A-panel nest vs the legacy serial nest on 2048×1024×2048
    /// f32 (16 MiB `A`, over one slice): both nests must agree
    /// bit-for-bit (same pc-ascending per-element order, only the tile
    /// grouping differs). Arms selected by explicit `=1`/`=0` on
    /// `SONIUM_GEMM_ST_APACK`; skipped when the slice size is
    /// unreadable (the route never engages then). Calls
    /// [`sgemm_serial`] directly so the (wide) test pool can't route
    /// the shape down a parallel path.
    #[test]
    fn avx2_st_apack_matches_legacy_bitwise() {
        if !is_available() {
            return;
        }
        let _guard = env_test_lock();
        let slice = sonium_topology::affinity::l3_bytes();
        if slice.is_none() {
            return;
        }
        let (m, n, k) = (2048usize, 1024usize, 2048usize);
        assert!(
            st_apack_shape(m, n, k, F32_MC, F32_KC, F32_NC, 4, slice),
            "test size must engage the ST-A-pack gate"
        );
        assert!(
            !st_apack_shape(1024, 1024, 1024, F32_MC, F32_KC, F32_NC, 4, slice),
            "sq1k must keep the legacy nest"
        );
        let lda = k + 5;
        let ldb = n + 3;
        let ldc = n + 7;
        let a32: Vec<f32> = (0..m * lda).map(|i| (i as f32 * 0.0013).sin()).collect();
        let b32: Vec<f32> = (0..k * ldb).map(|i| (i as f32 * 0.0017).cos()).collect();
        let c032: Vec<f32> = (0..m * ldc).map(|i| (i as f32 * 0.0011).sin()).collect();
        let mut panel = c032.clone();
        let mut legacy = c032;
        // SAFETY: only writer of `SONIUM_GEMM_ST_APACK` among tests.
        unsafe {
            std::env::set_var("SONIUM_GEMM_ST_APACK", "1");
        }
        unsafe {
            sgemm_serial(
                m, n, k, 0.9, &a32, lda, &b32, ldb, 0.35, &mut panel, ldc, false, false,
            );
        }
        unsafe {
            std::env::set_var("SONIUM_GEMM_ST_APACK", "0");
        }
        unsafe {
            sgemm_serial(
                m,
                n,
                k,
                0.9,
                &a32,
                lda,
                &b32,
                ldb,
                0.35,
                &mut legacy,
                ldc,
                false,
                false,
            );
        }
        unsafe {
            std::env::remove_var("SONIUM_GEMM_ST_APACK");
        }
        assert_eq!(panel, legacy, "panel nest must match legacy bitwise");
    }

    /// f64 twin of [`avx2_st_apack_matches_legacy_bitwise`] on
    /// 2048×1024×2048 (32 MiB `A`, over one slice): panel nest vs
    /// legacy nest, bitwise identical. Same skip rules.
    #[test]
    fn avx2_st_apack_matches_legacy_bitwise_f64() {
        if !is_available() {
            return;
        }
        let _guard = env_test_lock();
        let slice = sonium_topology::affinity::l3_bytes();
        if slice.is_none() {
            return;
        }
        let (m, n, k) = (2048usize, 1024usize, 2048usize);
        assert!(
            st_apack_shape(m, n, k, F64_MC, F64_KC, F64_NC, 8, slice),
            "test size must engage the ST-A-pack gate"
        );
        assert!(
            !st_apack_shape(1024, 1024, 1024, F64_MC, F64_KC, F64_NC, 8, slice),
            "sq1k must keep the legacy nest"
        );
        let lda = k + 5;
        let ldb = n + 3;
        let ldc = n + 7;
        let a64: Vec<f64> = (0..m * lda).map(|i| (i as f64 * 0.0013).sin()).collect();
        let b64: Vec<f64> = (0..k * ldb).map(|i| (i as f64 * 0.0017).cos()).collect();
        let c064: Vec<f64> = (0..m * ldc).map(|i| (i as f64 * 0.0011).sin()).collect();
        let mut panel = c064.clone();
        let mut legacy = c064;
        // SAFETY: only writer of `SONIUM_GEMM_ST_APACK` among tests.
        unsafe {
            std::env::set_var("SONIUM_GEMM_ST_APACK", "1");
        }
        unsafe {
            dgemm_serial(
                m, n, k, -0.4, &a64, lda, &b64, ldb, 0.85, &mut panel, ldc, false, false,
            );
        }
        unsafe {
            std::env::set_var("SONIUM_GEMM_ST_APACK", "0");
        }
        unsafe {
            dgemm_serial(
                m,
                n,
                k,
                -0.4,
                &a64,
                lda,
                &b64,
                ldb,
                0.85,
                &mut legacy,
                ldc,
                false,
                false,
            );
        }
        unsafe {
            std::env::remove_var("SONIUM_GEMM_ST_APACK");
        }
        assert_eq!(panel, legacy, "panel nest must match legacy bitwise");
    }

    /// f32 N-split wide-Kc (512, default at k >= 768) vs the scalar
    /// reference on 512x512x768 Trans and NoTrans: pass boundaries
    /// change fp parenthesization (same terms, different grouping —
    /// like the single-pass 2D path), so widths cannot be
    /// bitwise-equal; both must match scalar within the standard f32
    /// tolerance instead (same class as
    /// `avx2_sgemm_kc_multi_pass_matches_scalar`).
    #[test]
    fn avx2_kc512_matches_scalar() {
        if !is_available() {
            return;
        }
        for trans_a in [true, false] {
            let (m, n, k) = (512usize, 512usize, 768usize);
            assert!(
                use_parallel_n(m, n, k, F32_KC),
                "test size must engage the N-split path"
            );
            assert_eq!(f32_kc_nsplit(k), F32_KC_WIDE, "test k must engage wide-Kc");
            let lda = if trans_a { m + 5 } else { k + 3 };
            let alen = lda * if trans_a { k } else { m };
            let a32: Vec<f32> = (0..alen).map(|i| (i as f32 * 0.0013).sin()).collect();
            let b32: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.0017).cos()).collect();
            let c032: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.0011).sin()).collect();
            let mut got = c032.clone();
            let mut want = c032;
            let sta = if trans_a {
                Transpose::Trans
            } else {
                Transpose::NoTrans
            };
            unsafe {
                sgemm(
                    m, n, k, 0.9, &a32, lda, &b32, n, 0.35, &mut got, n, trans_a, false,
                );
            }
            scalar::gemm_f32(
                sta,
                Transpose::NoTrans,
                m,
                n,
                k,
                0.9,
                &a32,
                lda,
                &b32,
                n,
                0.35,
                &mut want,
                n,
            );
            for (g, w) in got.iter().zip(want.iter()) {
                assert_relative_eq!(g, w, max_relative = 1e-4, epsilon = 1e-5);
            }
        }
    }

    /// f64 N-split wide-Kc (512, default at 768 <= k <= 1152 and n >= 512)
    /// vs the scalar reference on 512x512x768 Trans and NoTrans: pass
    /// boundaries change fp parenthesization (same terms, different
    /// grouping — like the f32 twin), so the widths cannot be
    /// bitwise-equal; both must match scalar within the standard f64
    /// tolerance instead.
    #[test]
    fn avx2_dkc512_matches_scalar() {
        if !is_available() {
            return;
        }
        for trans_a in [true, false] {
            let (m, n, k) = (512usize, 512usize, 768usize);
            assert!(
                use_parallel_n(m, n, k, F64_KC),
                "test size must engage the N-split path"
            );
            assert_eq!(
                f64_kc_nsplit(n, k),
                F64_KC_WIDE,
                "test shape must engage wide-Kc"
            );
            let lda = if trans_a { m + 5 } else { k + 3 };
            let alen = lda * if trans_a { k } else { m };
            let a64: Vec<f64> = (0..alen).map(|i| (i as f64 * 0.0013).sin()).collect();
            let b64: Vec<f64> = (0..k * n).map(|i| (i as f64 * 0.0017).cos()).collect();
            let c064: Vec<f64> = (0..m * n).map(|i| (i as f64 * 0.0011).sin()).collect();
            let mut got = c064.clone();
            let mut want = c064;
            let sta = if trans_a {
                Transpose::Trans
            } else {
                Transpose::NoTrans
            };
            unsafe {
                dgemm(
                    m, n, k, 0.9, &a64, lda, &b64, n, 0.35, &mut got, n, trans_a, false,
                );
            }
            scalar::gemm_f64(
                sta,
                Transpose::NoTrans,
                m,
                n,
                k,
                0.9,
                &a64,
                lda,
                &b64,
                n,
                0.35,
                &mut want,
                n,
            );
            for (g, w) in got.iter().zip(want.iter()) {
                assert_relative_eq!(g, w, max_relative = 1e-9, epsilon = 1e-12);
            }
        }
    }

    /// Wave-snapped band counts are bitwise-neutral: 768^3 f64 with the
    /// shipped snap (16 bands) vs `SONIUM_GEMM_WAVE=0` (24 bands) must
    /// agree elementwise — each C element keeps its task and pass
    /// structure, only the column partition moves.
    #[test]
    fn avx2_wave_snap_bitwise_neutral() {
        if !is_available() {
            return;
        }
        if sonium_topology::affinity::l3_total_bytes().is_none() {
            return;
        }
        let _guard = env_test_lock();
        // A 16-thread pool: the ambient test pool sizes bands for 128
        // threads (where the snap is a no-op); 16 reproduces the
        // 24-band partial wave the snap fixes.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(16)
            .build()
            .unwrap();
        pool.install(|| {
            let (m, n, k) = (768usize, 768usize, 768usize);
            assert!(
                use_parallel_n(m, n, k, F64_KC),
                "test size must engage the N-split path"
            );
            // SAFETY: only writer of `SONIUM_GEMM_WAVE` among tests;
            // the lock serializes env-mutating tests.
            unsafe {
                std::env::remove_var("SONIUM_GEMM_WAVE");
            }
            let workers = crate::cpu::blas3::workers_for_macs(
                m.saturating_mul(n).saturating_mul(k.min(F64_KC)),
            );
            assert_eq!(workers, 16, "test pool must size 16 bands");
            let snapped = l3_fit_cpt(m, n, k, 8, F64_NR, workers, MIN_TASK_MAC);
            unsafe {
                std::env::set_var("SONIUM_GEMM_WAVE", "0");
            }
            let legacy = l3_fit_cpt(m, n, k, 8, F64_NR, workers, MIN_TASK_MAC);
            assert_ne!(snapped, legacy, "test shape must change band counts");
            let a64: Vec<f64> = (0..m * k).map(|i| (i as f64 * 0.0013).sin()).collect();
            let b64: Vec<f64> = (0..k * n).map(|i| (i as f64 * 0.0017).cos()).collect();
            let c064: Vec<f64> = (0..m * n).map(|i| (i as f64 * 0.0011).sin()).collect();
            let mut got_legacy = c064.clone();
            unsafe {
                dgemm(
                    m,
                    n,
                    k,
                    0.9,
                    &a64,
                    k,
                    &b64,
                    n,
                    0.35,
                    &mut got_legacy,
                    n,
                    false,
                    false,
                );
                std::env::remove_var("SONIUM_GEMM_WAVE");
            }
            let mut got_snapped = c064.clone();
            unsafe {
                dgemm(
                    m,
                    n,
                    k,
                    0.9,
                    &a64,
                    k,
                    &b64,
                    n,
                    0.35,
                    &mut got_snapped,
                    n,
                    false,
                    false,
                );
            }
            assert_eq!(got_snapped, got_legacy, "wave snap must be bitwise-neutral");
        });
    }

    /// Floor-rounded band widths are bitwise-neutral: 1024x260x1024 f64
    /// with the shipped floor (cpt 16, 17 bands) vs
    /// `SONIUM_GEMM_CPTFLR=0` (cpt 24, 11 bands) must agree
    /// elementwise — same argument as the wave snap (only the column
    /// partition moves). The exact cpt values are machine-portable:
    /// the pool floor binds the planned count to 16 on any L3 size
    /// (`geom` 17 < 32 keeps the snap at one wave).
    #[test]
    fn avx2_cpt_floor_bitwise_neutral() {
        if !is_available() {
            return;
        }
        if sonium_topology::affinity::l3_total_bytes().is_none() {
            return;
        }
        let _guard = env_test_lock();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(16)
            .build()
            .unwrap();
        pool.install(|| {
            let (m, n, k) = (1024usize, 260usize, 1024usize);
            assert!(
                use_parallel_n(m, n, k, F64_KC),
                "test size must engage the N-split path"
            );
            // SAFETY: only writer of `SONIUM_GEMM_CPTFLR` among tests;
            // the lock serializes env-mutating tests.
            unsafe {
                std::env::remove_var("SONIUM_GEMM_CPTFLR");
            }
            let workers = crate::cpu::blas3::workers_for_macs(
                m.saturating_mul(n).saturating_mul(k.min(F64_KC)),
            );
            assert_eq!(workers, 16, "test pool must size 16 bands");
            let floored = l3_fit_cpt(m, n, k, 8, F64_NR, workers, MIN_TASK_MAC);
            assert_eq!(floored, Some(16), "floor must give cpt 16");
            unsafe {
                std::env::set_var("SONIUM_GEMM_CPTFLR", "0");
            }
            let ceiled = l3_fit_cpt(m, n, k, 8, F64_NR, workers, MIN_TASK_MAC);
            assert_eq!(ceiled, Some(24), "ceil must give cpt 24");
            let a64: Vec<f64> = (0..m * k).map(|i| (i as f64 * 0.0013).sin()).collect();
            let b64: Vec<f64> = (0..k * n).map(|i| (i as f64 * 0.0017).cos()).collect();
            let c064: Vec<f64> = (0..m * n).map(|i| (i as f64 * 0.0011).sin()).collect();
            let mut got_ceiled = c064.clone();
            unsafe {
                dgemm(
                    m,
                    n,
                    k,
                    0.9,
                    &a64,
                    k,
                    &b64,
                    n,
                    0.35,
                    &mut got_ceiled,
                    n,
                    false,
                    false,
                );
                std::env::remove_var("SONIUM_GEMM_CPTFLR");
            }
            let mut got_floored = c064.clone();
            unsafe {
                dgemm(
                    m,
                    n,
                    k,
                    0.9,
                    &a64,
                    k,
                    &b64,
                    n,
                    0.35,
                    &mut got_floored,
                    n,
                    false,
                    false,
                );
            }
            assert_eq!(got_floored, got_ceiled, "cpt floor must be bitwise-neutral");
        });
    }

    /// Serial trans-A pre-pack vs the scalar reference, both precisions.
    /// A 1-thread pool install forces the
    /// `rayon::current_num_threads() == 1` gate; the odd shape crosses
    /// the vector-pack tails.
    #[test]
    fn avx2_trans_a_serial_pack_matches_scalar() {
        if !is_available() {
            return;
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| {
            assert_eq!(rayon::current_num_threads(), 1);
            let (m, n, k) = (133usize, 129usize, 130usize);
            assert!(
                m * k >= DGEMM_TRANSPOSE_PACK_MIN,
                "test size must engage the serial pre-pack"
            );
            let lda = m + 3;
            let a32: Vec<f32> = (0..k * lda).map(|i| (i as f32 * 0.0013).sin()).collect();
            let b32: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.0017).cos()).collect();
            let c032: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.0011).sin()).collect();
            let mut got32 = c032.clone();
            let mut want32 = c032;
            unsafe {
                sgemm(
                    m, n, k, 0.9, &a32, lda, &b32, n, 0.35, &mut got32, n, true, false,
                );
            }
            scalar::gemm_f32(
                Transpose::Trans,
                Transpose::NoTrans,
                m,
                n,
                k,
                0.9,
                &a32,
                lda,
                &b32,
                n,
                0.35,
                &mut want32,
                n,
            );
            for (g, w) in got32.iter().zip(want32.iter()) {
                assert_relative_eq!(g, w, max_relative = 1e-4, epsilon = 1e-5);
            }
            let a64: Vec<f64> = (0..k * lda).map(|i| (i as f64 * 0.0013).sin()).collect();
            let b64: Vec<f64> = (0..k * n).map(|i| (i as f64 * 0.0017).cos()).collect();
            let c064: Vec<f64> = (0..m * n).map(|i| (i as f64 * 0.0011).sin()).collect();
            let mut got64 = c064.clone();
            let mut want64 = c064;
            unsafe {
                dgemm(
                    m, n, k, -0.4, &a64, lda, &b64, n, 0.85, &mut got64, n, true, false,
                );
            }
            scalar::gemm_f64(
                Transpose::Trans,
                Transpose::NoTrans,
                m,
                n,
                k,
                -0.4,
                &a64,
                lda,
                &b64,
                n,
                0.85,
                &mut want64,
                n,
            );
            for (g, w) in got64.iter().zip(want64.iter()) {
                assert_relative_eq!(g, w, max_relative = 1e-11, epsilon = 1e-12);
            }
        });
    }

    /// `trans_b` gather path: `B` stored `n × k` row-major, consumed as
    /// `k × n` via transposed-gather tiles — no pre-pack. Odd sizes cross
    /// the N-split, M-split (tall `m2`), and serial tails; non-tight
    /// strides guard the gather indexing.
    #[test]
    fn avx2_trans_b_matches_scalar() {
        if !is_available() {
            return;
        }
        // (m, n, k, lda, ldb, ldc): wide N-split (odd tails), tall
        // M-split, serial. All parallel shapes clear `PAR_MIN_MACS`.
        let shapes = [
            (67usize, 513usize, 300usize, 305usize, 307usize, 517usize),
            (300usize, 40usize, 400usize, 405usize, 407usize, 43usize),
            (37usize, 41usize, 29usize, 34usize, 31usize, 44usize),
        ];
        for (m, n, k, lda, ldb, ldc) in shapes {
            // B stored n × k.
            let a32: Vec<f32> = (0..m * lda).map(|i| (i as f32 * 0.0013).sin()).collect();
            let b32: Vec<f32> = (0..n * ldb).map(|i| (i as f32 * 0.0017).cos()).collect();
            let c032: Vec<f32> = (0..m * ldc).map(|i| (i as f32 * 0.0011).sin()).collect();
            let mut got32 = c032.clone();
            let mut want32 = c032;
            unsafe {
                sgemm(
                    m, n, k, 0.9, &a32, lda, &b32, ldb, 0.35, &mut got32, ldc, false, true,
                );
            }
            scalar::gemm_f32(
                Transpose::NoTrans,
                Transpose::Trans,
                m,
                n,
                k,
                0.9,
                &a32,
                lda,
                &b32,
                ldb,
                0.35,
                &mut want32,
                ldc,
            );
            for i in 0..m {
                for j in 0..n {
                    assert_relative_eq!(
                        got32[i * ldc + j],
                        want32[i * ldc + j],
                        max_relative = 1e-4,
                        epsilon = 1e-5
                    );
                }
            }

            let a64: Vec<f64> = (0..m * lda).map(|i| (i as f64 * 0.0013).sin()).collect();
            let b64: Vec<f64> = (0..n * ldb).map(|i| (i as f64 * 0.0017).cos()).collect();
            let c064: Vec<f64> = (0..m * ldc).map(|i| (i as f64 * 0.0011).sin()).collect();
            let mut got64 = c064.clone();
            let mut want64 = c064;
            unsafe {
                dgemm(
                    m, n, k, -0.4, &a64, lda, &b64, ldb, 0.85, &mut got64, ldc, false, true,
                );
            }
            scalar::gemm_f64(
                Transpose::NoTrans,
                Transpose::Trans,
                m,
                n,
                k,
                -0.4,
                &a64,
                lda,
                &b64,
                ldb,
                0.85,
                &mut want64,
                ldc,
            );
            for i in 0..m {
                for j in 0..n {
                    assert_relative_eq!(
                        got64[i * ldc + j],
                        want64[i * ldc + j],
                        max_relative = 1e-11,
                        epsilon = 1e-12
                    );
                }
            }
        }
    }

    /// Parallel-path parity: a size above `PAR_MIN_MACS` with row tails
    /// (134 % 4), column tails (130 % 16 / 130 % 8), and a multi-pass Kc
    /// loop (k = 300 > 256) must match the scalar reference. The rayon row
    /// partition keeps per-element accumulation order identical to serial,
    /// so this also guards the parallel dispatch against races.
    #[test]
    fn avx2_parallel_matches_scalar_with_tails_and_kc_passes() {
        if !is_available() {
            return;
        }
        let (m, n, k) = (134, 130, 300);
        assert!(
            use_parallel(m, n, k, F32_KC),
            "test size must engage the parallel path"
        );
        let a32: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.0013).sin()).collect();
        let b32: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.0017).cos()).collect();
        let c032: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.0011).sin()).collect();
        let mut got32 = c032.clone();
        let mut want32 = c032;
        unsafe {
            sgemm(
                m, n, k, 0.9, &a32, k, &b32, n, 0.35, &mut got32, n, false, false,
            );
        }
        scalar::gemm_f32(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            0.9,
            &a32,
            k,
            &b32,
            n,
            0.35,
            &mut want32,
            n,
        );
        for (g, w) in got32.iter().zip(want32.iter()) {
            assert_relative_eq!(g, w, max_relative = 1e-4, epsilon = 1e-5);
        }

        let a64: Vec<f64> = (0..m * k).map(|i| (i as f64 * 0.0013).sin()).collect();
        let b64: Vec<f64> = (0..k * n).map(|i| (i as f64 * 0.0017).cos()).collect();
        let c064: Vec<f64> = (0..m * n).map(|i| (i as f64 * 0.0011).sin()).collect();
        let mut got64 = c064.clone();
        let mut want64 = c064;
        unsafe {
            dgemm(
                m, n, k, -0.4, &a64, k, &b64, n, 0.85, &mut got64, n, false, false,
            );
        }
        scalar::gemm_f64(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            -0.4,
            &a64,
            k,
            &b64,
            n,
            0.85,
            &mut want64,
            n,
        );
        for (g, w) in got64.iter().zip(want64.iter()) {
            assert_relative_eq!(g, w, max_relative = 1e-11, epsilon = 1e-12);
        }
    }

    /// N-split parity: thin-`m` shapes above `PAR_MIN_MACS` with row tails,
    /// column tails, and multi-pass Kc must match the scalar reference.
    /// Covers both a mid-size N-gated shape and the M-gate-excluded regime
    /// (`m < MR * 2`), which the M-split leaves serial. Column-band tasks
    /// keep accumulation order identical to serial, so this also guards
    /// the N dispatch against races. Both shapes satisfy `n > m` with
    /// enough work, so they take the N path, not M.
    #[test]
    fn avx2_nsplit_matches_scalar_with_tails() {
        if !is_available() {
            return;
        }
        for (m, n, k) in [(66, 2048, 4096), (4, 4096, 2048)] {
            assert!(
                use_parallel_n(m, n, k, F32_KC),
                "test size ({m},{n},{k}) must engage the N-split path"
            );
            let a32: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.0013).sin()).collect();
            let b32: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.0017).cos()).collect();
            let c032: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.0011).sin()).collect();
            let mut got32 = c032.clone();
            let mut want32 = c032;
            unsafe {
                sgemm(
                    m, n, k, 0.9, &a32, k, &b32, n, 0.35, &mut got32, n, false, false,
                );
            }
            scalar::gemm_f32(
                Transpose::NoTrans,
                Transpose::NoTrans,
                m,
                n,
                k,
                0.9,
                &a32,
                k,
                &b32,
                n,
                0.35,
                &mut want32,
                n,
            );
            for (g, w) in got32.iter().zip(want32.iter()) {
                assert_relative_eq!(g, w, max_relative = 1e-4, epsilon = 1e-5);
            }

            let a64: Vec<f64> = (0..m * k).map(|i| (i as f64 * 0.0013).sin()).collect();
            let b64: Vec<f64> = (0..k * n).map(|i| (i as f64 * 0.0017).cos()).collect();
            let c064: Vec<f64> = (0..m * n).map(|i| (i as f64 * 0.0011).sin()).collect();
            let mut got64 = c064.clone();
            let mut want64 = c064;
            unsafe {
                dgemm(
                    m, n, k, -0.4, &a64, k, &b64, n, 0.85, &mut got64, n, false, false,
                );
            }
            scalar::gemm_f64(
                Transpose::NoTrans,
                Transpose::NoTrans,
                m,
                n,
                k,
                -0.4,
                &a64,
                k,
                &b64,
                n,
                0.85,
                &mut want64,
                n,
            );
            for (g, w) in got64.iter().zip(want64.iter()) {
                assert_relative_eq!(g, w, max_relative = 1e-11, epsilon = 1e-12);
            }
        }
    }

    /// Large shapes engage the packed-B path. Odd extents exercise
    /// kc/nc/row/column tails across multiple tiles; non-tight leading
    /// dims guard stride mixups between the pack layout and the direct
    /// path. (Ported from the direct-x86 opt track; the gate constants
    /// have since moved into the parallel packing machinery above, but
    /// the coverage is still the packed path at these sizes.)
    #[test]
    fn avx2_dgemm_packed_matches_scalar_with_tails() {
        if !is_available() {
            return;
        }
        let (m, n, k) = (402, 388, 376);
        let lda = k + 3;
        let ldb = n + 5;
        let ldc = n + 7;
        let a: Vec<f64> = (0..m * lda).map(|i| (i as f64 * 0.013).sin()).collect();
        let b: Vec<f64> = (0..k * ldb).map(|i| (i as f64 * 0.017).cos()).collect();
        let c0: Vec<f64> = (0..m * ldc).map(|i| (i as f64 * 0.011).sin()).collect();
        let mut got = c0.clone();
        let mut want = c0;
        unsafe {
            dgemm(
                m, n, k, 0.7, &a, lda, &b, ldb, 0.35, &mut got, ldc, false, false,
            );
        }
        scalar::gemm_f64(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            0.7,
            &a,
            lda,
            &b,
            ldb,
            0.35,
            &mut want,
            ldc,
        );
        for i in 0..m {
            for j in 0..n {
                assert_relative_eq!(
                    got[i * ldc + j],
                    want[i * ldc + j],
                    max_relative = 1e-11,
                    epsilon = 1e-12
                );
            }
        }
    }

    /// f32 packed-B path with tails and non-tight strides, TransA variant.
    #[test]
    fn avx2_sgemm_packed_matches_scalar_with_tails() {
        if !is_available() {
            return;
        }
        // ~0.97 Gflop exercises the pack path; odd extents hit tails.
        let (m, n, k) = (822, 781, 759);
        // TransA: stored A is k × m.
        let lda = m + 3;
        let ldb = n + 5;
        let ldc = n + 7;
        let a: Vec<f32> = (0..k * lda).map(|i| (i as f32 * 0.013).sin()).collect();
        let b: Vec<f32> = (0..k * ldb).map(|i| (i as f32 * 0.017).cos()).collect();
        let c0: Vec<f32> = (0..m * ldc).map(|i| (i as f32 * 0.011).sin()).collect();
        let mut got = c0.clone();
        let mut want = c0;
        unsafe {
            sgemm(
                m, n, k, -0.4, &a, lda, &b, ldb, 0.85, &mut got, ldc, true, false,
            );
        }
        scalar::gemm_f32(
            Transpose::Trans,
            Transpose::NoTrans,
            m,
            n,
            k,
            -0.4,
            &a,
            lda,
            &b,
            ldb,
            0.85,
            &mut want,
            ldc,
        );
        for i in 0..m {
            for j in 0..n {
                assert_relative_eq!(
                    got[i * ldc + j],
                    want[i * ldc + j],
                    max_relative = 1e-4,
                    epsilon = 1e-5
                );
            }
        }
    }

    /// Degenerate alpha = 0 must zero/scale C without reading A or B.
    #[test]
    fn avx2_dgemm_alpha_zero_scales_c() {
        if !is_available() {
            return;
        }
        let (m, n, k) = (5, 6, 4);
        let a = vec![f64::NAN; m * k];
        let b = vec![f64::NAN; k * n];
        let mut c = vec![2.0f64; m * n];
        unsafe {
            dgemm(m, n, k, 0.0, &a, k, &b, n, 0.0, &mut c, n, false, false);
        }
        assert!(c.iter().all(|&v| v == 0.0));
    }

    /// `pack_trans_f64` against the scalar element loop, with dimensions
    /// that defeat blocking (primes around the 8-wide blocks) and a padded
    /// source stride. Exact bitwise equality: packing copies values.
    #[test]
    fn pack_trans_f64_matches_scalar() {
        if !is_available() {
            return;
        }
        // `ldb >= rows`: the source is `cols × rows` with a valid stride.
        for (rows, cols, ldb) in [
            (8, 8, 8),
            (13, 19, 23),
            (64, 65, 65),
            (1, 100, 100),
            (100, 1, 100),
        ] {
            let src: Vec<f64> = (0..(cols - 1) * ldb + rows)
                .map(|i| (i as f64 * 0.031).sin())
                .collect();
            let mut got = vec![0.0; rows * cols];
            let mut want = vec![0.0; rows * cols];
            unsafe {
                pack_trans_f64(src.as_ptr(), ldb, rows, cols, got.as_mut_ptr());
            }
            for r in 0..rows {
                for c in 0..cols {
                    want[r * cols + c] = src[c * ldb + r];
                }
            }
            assert_eq!(got, want, "rows={rows} cols={cols} ldb={ldb}");
        }
    }

    /// `pack_trans_f32` against the scalar element loop. Dimensions defeat
    /// the 8-wide blocks (primes) plus a padded source stride and a
    /// tall-skinny panel (the SYRK `k × n` shape). Exact bitwise equality:
    /// packing copies values.
    #[test]
    fn pack_trans_f32_matches_scalar() {
        if !is_available() {
            return;
        }
        // `ldb >= rows`: the source is `cols × rows` with a valid stride.
        for (rows, cols, ldb) in [
            (8, 8, 8),
            (13, 19, 23),
            (64, 65, 65),
            (48, 544, 1024),
            (1, 100, 100),
            (100, 1, 100),
        ] {
            let src: Vec<f32> = (0..(cols - 1) * ldb + rows)
                .map(|i| (i as f32 * 0.031).sin())
                .collect();
            let mut got = vec![0.0; rows * cols];
            let mut want = vec![0.0; rows * cols];
            unsafe {
                pack_trans_f32(src.as_ptr(), ldb, rows, cols, got.as_mut_ptr());
            }
            for r in 0..rows {
                for c in 0..cols {
                    want[r * cols + c] = src[c * ldb + r];
                }
            }
            assert_eq!(got, want, "rows={rows} cols={cols} ldb={ldb}");
        }
    }
}
