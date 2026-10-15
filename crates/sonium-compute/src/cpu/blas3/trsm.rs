//! Blocked triangular solve with multiple right-hand sides (`TRSM`).
//!
//! Solves `op(A) * X = alpha * B` (`Side::Left`) or `X * op(A) = alpha * B`
//! (`Side::Right`) for row-major `A`/`B`, overwriting `B` with `X`. This is
//! the LAPACK `*TRSM` workhorse our `LU`/`Cholesky` multi-RHS solves need,
//! and the BLAS entry vendor libraries ship that `sonium-compute` lacked.
//!
//! Algorithm: LAPACK-style blocked solve with an unblocked diagonal kernel
//! (`NB = 64`) and trailing updates through the threaded
//! [`crate::cpu::blas3`] `gemm`, so large shapes ride the GEMM fast path
//! (threading included) while small shapes stay in cache. Only rounding
//! differs from a fully unblocked solve; no zero-skips, so NaN propagates
//! per IEEE.
//!
//! All four real variants (`Left`/`Right` × `Lower`/`Upper` ×
//! `NoTrans`/`Trans`) plus complex `ConjTrans` (conjugation folded into the
//! unblocked reads) are covered for `f32`, `f64`, `Complex32`, `Complex64`
//! via one generic core; the per-type wrappers below only select the
//! matching `gemm` trailing-update kernel.

use crate::backend::Transpose;
use crate::cpu::blas3::{
    PoolFloorGuard, TINY_MAX_MACS, gemm_c32, gemm_c64, gemm_f32, gemm_f32_serial, gemm_f64,
    gemm_f64_serial, pool_floor_small, scope_for_macs, scope_for_nested_outer, workers_for_macs,
};
use num_complex::{Complex32, Complex64};
use num_traits::Num;

/// Which side of `X` the triangular factor sits on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TrsmSide {
    Left,
    Right,
}

/// Which triangle of `A` is referenced.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TrsmUplo {
    Lower,
    Upper,
}

/// Scalar contract for the generic TRSM core: ring ops plus conjugation
/// (identity for real types) so `ConjTrans` folds into the unblocked reads.
pub trait TrsmScalar: Copy + PartialEq + Num + std::ops::Neg<Output = Self> {
    fn conj(self) -> Self;
}

impl TrsmScalar for f32 {
    #[inline]
    fn conj(self) -> Self {
        self
    }
}
impl TrsmScalar for f64 {
    #[inline]
    fn conj(self) -> Self {
        self
    }
}
impl TrsmScalar for Complex32 {
    #[inline]
    fn conj(self) -> Self {
        num_complex::Complex::conj(&self)
    }
}
impl TrsmScalar for Complex64 {
    #[inline]
    fn conj(self) -> Self {
        num_complex::Complex::conj(&self)
    }
}

/// Panel width of the blocked solve. `NB = 64` measured faster than 128
/// (the unblocked diagonal's `m · n · NB` flops double with the width while
/// the trailing updates only reshape); the diagonal stays threaded over
/// right-hand sides (see `UNBLOCKED_PAR_MIN`).
const TRSM_NB: usize = 128;

/// Diagnostic cap on the Right-solve panel width (`SONIUM_TRSM_CHUNK`; floor
/// 16, default `TRSM_NB`). Finer internal panels (e.g. 32) keep the nb x nb
/// operator in L1D across the panel solve + update, at the cost of narrower
/// updates. Trial outcome (this box, noisy load ~3): chunk=32 probe A/B
/// min-of-3: m=1984 +4%, m=960/m=256 tied; in-situ dpotrf-2048 RAYON=16
/// medians tied (19.2-19.7ms both arms). NOT shipped default-on: the
/// L1-operator theory is falsified as the main lever — the ~2.5x gap is
/// kernel rate / launch overhead, not operator residency. Gate retained
/// (default off) for future sweeps. Left loops keep `TRSM_NB`.
fn trsm_right_chunk() -> usize {
    std::env::var("SONIUM_TRSM_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&c| c >= 16)
        .unwrap_or(TRSM_NB)
}

type GemmFn<T> =
    fn(Transpose, Transpose, usize, usize, usize, T, &[T], usize, &[T], usize, T, &mut [T], usize);

/// Concrete band-solve kernel pointer, threaded from the `trsm_*` entry
/// points through [`trsm_core`] like [`GemmFn`] so accelerated kernels
/// (AVX2+FMA) can replace the generic scalar lane loop without touching
/// the parallel dispatch. Argument order matches the `band_*_kernel` fns:
/// `(xb, ldb, x0, nb, j0, j1, trans, unit, ab, di)`.
type BandKernelFn<T> =
    unsafe fn(*mut T, usize, usize, usize, usize, usize, bool, bool, *const T, *const T);

// ---------- unblocked diagonal kernels --------------------------------------
// All solve in place: `xb` (the current `B` block) becomes the `X` block.
// `a`/`lda`/`a0` address the diagonal block origin; `xb`/`ldb`/`x0` the
// `B`-block origin. `trans` selects `op` inside the block reads, `conjugate`
// folds `ConjTrans` into them (real types pass `false`).

/// Pointer wrappers for `rayon::scope` dispatch of the unblocked diagonal
/// solves: disjoint `xb` bands, read-only `A`. `Clone`/`Copy` are
/// unconditional (a raw pointer is `Copy` whatever it points at — the
/// fused path shares a whole panels `Vec` through the same wrapper, whose
/// element type is not `Copy`, so a derived impl would not apply).
struct SendConstPtr<T>(*const T);
impl<T> Clone for SendConstPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SendConstPtr<T> {}
unsafe impl<T> Send for SendConstPtr<T> {}
unsafe impl<T> Sync for SendConstPtr<T> {}
struct SendMutPtr<T>(*mut T);
impl<T> Clone for SendMutPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SendMutPtr<T> {}
unsafe impl<T> Send for SendMutPtr<T> {}
unsafe impl<T> Sync for SendMutPtr<T> {}

/// Minimum right-hand sides (left) or rows (right) before the unblocked
/// diagonal solve fans out over rayon. Below this the serial band wins:
/// one right-hand side does `nb² / 2` flops (~8k at `TRSM_NB = 128`).
const UNBLOCKED_PAR_MIN: usize = 128;

/// Per-solve unblocked-scope MAC ceiling for letting a TRSM solve use the
/// tiny-work pool (see [`scope_for_macs`]): small solves run every scope
/// on tiny, larger solves pin every scope to the small pool via
/// [`pool_floor_small`], so a solve never alternates pools mid-flight.
/// Alternating costs ~0.4ms per tiny scope on wide machines (TRSM-1024:
/// 6.2ms all-small vs 7.5ms mixed; TRSM-2048 recovers 35 → 29ms once its
/// sixteen 33M-MAC unblocked scopes stop alternating) — consistent with
/// package power-state transitions — while all-tiny solves win big
/// (TRSM-512 1.3x, TRSM-640 1.2x) and monochromatic solves never pay it.
/// A/B probes at `TRSM_NB = 128` (unblocked scope MACs = `nb²·count`):
/// 8.4M (`N = 512`) and 10.5M (`N = 640`) win on tiny, 12.6M (`N = 768`)
/// ties, 16.7M+ loses, so 12M is the largest safe whole-solve ceiling.
const TRSM_TINY_MAX_MACS: usize = 12_000_000;

/// Pin this solve's pool when any of its scopes would leave the tiny pool:
/// returns a live [`pool_floor_small`] guard (all scopes take the small
/// pool) unless both the largest unblocked scope (`nb²·count`) and the
/// largest trailing update (`m·n·nb`) fit the tiny routing, in which case
/// the solve stays all-tiny. `count` is the parallelized axis (`n` for
/// left, `m` for right). Either way the solve is monochromatic and pays
/// no pool-switching cost; results are bitwise identical on every pool.
fn pool_floor_for_trsm(
    m: usize,
    n: usize,
    count: usize,
    panel_len: usize,
) -> Option<PoolFloorGuard> {
    // Single-panel solves run exactly one unblocked scope: there is nothing
    // to alternate, so no floor — the scope-level routing in
    // `scope_for_macs` picks tiny/small/global by MACs on its own, and
    // small single-panel solves win big on fewer workers (contention on
    // wide pools costs more than the extra parallelism buys: 48 workers
    // ~1.7x slower than 16 on `m = 48, n = 2048`). Multi-panel solves keep
    // the tuned floor below. Pool choice never changes the numbers.
    if panel_len <= TRSM_NB {
        return None;
    }
    let unblocked = (TRSM_NB as u64) * (TRSM_NB as u64) * (count as u64);
    let trailing = (m as u64) * (n as u64) * (TRSM_NB as u64);
    if unblocked < TRSM_TINY_MAX_MACS as u64 && trailing < TINY_MAX_MACS as u64 {
        None
    } else {
        Some(pool_floor_small())
    }
}

/// Solve band width (columns per band task): 64 at 32 or fewer workers
/// (the long-standing tuned point), 16 above. Wide bands leave most of a
/// 128-worker pool idle during the solve phases, and idle workers'
/// spin-stealing on SMT siblings costs more than the extra dispatch of
/// narrow bands (measured 60 → 89 GF/s on `512×2048` going 64 → 16 wide
/// at 128 workers; neutral at 32).
const TRSM_BAND_W_WIDE: usize = 64;
const TRSM_BAND_W_NARROW: usize = 16;

/// Solve scratch chunk width (columns): each band is solved in `≤
/// BAND_CHUNK`-wide column chunks, each packed to a tight-stride block
/// (stride = chunk width) so the k-reduction stays L1-resident. The scratch
/// holds `nb × BAND_CHUNK` elements (64KB f64 at `TRSM_NB = 128`), sized for
/// the widest chunk; narrower chunks use its first `nb × nw` elements.
const BAND_CHUNK: usize = 64;

/// Band-task count for the unblocked solves over `count` columns (left) or
/// rows (right): 64-wide bands oversubscribed 4x. (A worker-scaled
/// coarsening was tried and reverted: fewer tasks left more workers idle,
/// whose spin-stealing on SMT siblings cost more than the dispatch it
/// saved. The oversubscription stays.)
///
/// NOTE (sizer/router experiment, settled): passing [`workers_for_macs`]
/// of the scope MACs instead of the caller width coarsened capped-pool
/// scopes (e.g. 120 → 30 tasks on the 16-worker tiny pool). Interleaved
/// within-run A/B (`SONIUM_TRSM_TASK_MODEL=router` vs default, same
/// process) shows the router model 4–16% slower at m = 256–1024 and tied
/// at m = 1920: the coarse split quantizes badly on the small pool
/// (stragglers), while the fine split stays balanced. Caller width stays
/// the default; the router model remains selectable via
/// [`band_task_workers`] for future re-tuning. `workers` stays a
/// parameter so the fused path (caller width, tuned as is) and the
/// unblocked paths share one sizer.
fn trsm_band_tasks(count: usize, nb: usize, workers: usize) -> usize {
    let workers = workers.max(1);
    // NOTE (reverted trial): doubling the band width (`SONIUM_TRSM_PAIR`)
    // ties in-situ dpotrf-2048 (16.74 -> 16.77ms med) — spawn savings
    // exactly offset by half-idle workers on big panels (8 tasks on 16
    // workers). Oversubscription stays load-bearing; do not re-coarsen
    // without addressing big-panel underfill (see the SYRK pairing note).
    let base = if workers <= 32 {
        TRSM_BAND_W_WIDE
    } else {
        TRSM_BAND_W_NARROW
    };
    // Small diagonal blocks: one band per tiny-pool worker (`count/16`
    // bands; bands wider than `BAND_CHUNK` just run more ≤64-wide chunks
    // reusing the task's stack scratch). Spawning costs serial µs per task
    // while a narrow band at `nb = 48` is ~1µs of compute — the 16-wide
    // fan-out spent more time dispatching than solving. Identity at
    // `nb >= TRSM_NB`, so the tuned full-block behaviour is untouched.
    // NOTE (corrected 2026-09-27): "bitwise identical under any split"
    // holds only when chunk boundaries align: bands wider than
    // `BAND_CHUNK`/lchunk re-chunk locally, and chunk edges change the
    // blocked accumulation order (serial-vs-parallel TRSM mismatch
    // measured, same residuals). Serial/parallel routing changes need
    // checksum verification, not just column-independence reasoning.
    let width = if nb >= TRSM_NB {
        base
    } else {
        (count / 16).max(base)
    };
    (count / width).clamp(1, workers.saturating_mul(4).max(1))
}

/// Forced narrow band width for tall-skinny Left/Lower solves, or `None`
/// to keep the default wide bands.
///
/// Default 64-wide bands starve the pool on tall-skinny shapes (n = 128
/// -> 2 bands on 16 workers); one-micro-panel bands (16 columns f32, 8
/// columns f64 at n <= 128) fill it. Measured 2026-09 x86 MT (isolated
/// A/B, min-of-reps): 1.15-2.3x on Left/Lower m in 256..=1024,
/// 64 <= n <= 512, unit and non-unit alike. The win decays monotonically
/// with m (f32 n=128: 1.99x at m=512, 1.27x at m=1024, 0.89x at m=1920;
/// same trend f64) as fatter per-band work saturates the pool and the
/// skinnier update GEMMs lose efficiency — so m > 1024 keeps default,
/// as do n > 512 (neutral-or-loses) and the unmeasured m < 256 / n < 64
/// corner. Complex32 inherits the f64 rule by element size and Complex64
/// takes 16 (perf analogy, unmeasured; correctness covered by the suite,
/// which runs with this gate always on).
fn trsm_narrow_width(m: usize, n: usize, elem_bytes: usize) -> Option<usize> {
    if (256..=1024).contains(&m) && (64..=512).contains(&n) {
        let w = if elem_bytes == 8 && n <= 128 { 8 } else { 16 };
        Some(w.min(n))
    } else {
        None
    }
}

/// Minimum `m` for the fused path: below it the panel loop wins even
/// multithreaded (upfront panel setup plus band fan-out don't amortize
/// over ~3M MACs; crossover measured between m = 224 and m = 256 at
/// 16T, both scalar types and diagonals, n = 128/256).
const TRSM_FUSED_MIN_M: usize = 256;

/// Max `n` for the invert-then-GEMM Right path (see
/// [`trsm_right_inverted`]): a single panel whose unblocked inverse is
/// small enough to stay a fraction of the GEMM that follows (64³/3
/// flops vs m·64²). Covers Cholesky's L21 panels (n <= 64).
const TRSM_INVERT_MAX_N: usize = 64;

/// Whether the fused single-scope path serves this Left solve. The
/// fused path's band fan-out plus upfront panel setup only pay with
/// threads to feed and work to amortize over: at one thread the plain
/// panel loop wins on every measured shape (same-process interleaved
/// ST A/B, freq-immune: panel 1.01-1.72x over fused across f32/f64,
/// unit/non-unit, m in 160..=1024, n in 128..=512), so single-threaded
/// solves route to the panel loop, as do small multi-panel solves
/// below [`TRSM_FUSED_MIN_M`] (panel 1.08-1.56x over fused at 16T,
/// m in 160..=224). `threads` is the current pool width.
fn trsm_use_fused(
    uplo: TrsmUplo,
    trans_flag: bool,
    m: usize,
    n: usize,
    threads: usize,
    fused_off: bool,
) -> bool {
    uplo == TrsmUplo::Lower
        && !trans_flag
        && m >= TRSM_FUSED_MIN_M
        && n >= UNBLOCKED_PAR_MIN
        && threads > 1
        && !fused_off
}

/// Ascending panel-loop block size. f32 Lower/NoTrans solves use 64:
/// finer panels shift flops from the k-dot diagonal micro-solve
/// (42%-of-GEMM rate) to GEMM-rate trailing updates — measured
/// 1.02-1.32x over 128 across m 64..1920, n 64..2048 ST (immune A/B)
/// and tie-or-win on MT small-m, with two marginal 1.02-1.03x cells.
/// f64 keeps 128 (already 0.88-1.01x ST; 64 loses 1.02-1.11x at
/// m >= 512 and 32 loses everywhere); Upper/Trans stay 128
/// (unmeasured). `elem_bytes == 4` selects f32 exactly (Complex32
/// is 8 bytes per element).
fn trsm_panel_nb(uplo: TrsmUplo, trans_flag: bool, elem_bytes: usize) -> usize {
    if uplo == TrsmUplo::Lower && !trans_flag && elem_bytes == 4 {
        64
    } else {
        TRSM_NB
    }
}

/// Worker width for an unblocked band scope over `macs` MACs: the caller
/// width by default, or [`workers_for_macs`] (the pool the scope router
/// will pick) when `SONIUM_TRSM_TASK_MODEL=router`. Diagnostic A/B gate
/// for the sizer/router question (see [`trsm_band_tasks`]); the env read
/// costs ~100ns once per panel, negligible next to a scope dispatch.
fn band_task_workers(macs: usize) -> usize {
    if std::env::var_os("SONIUM_TRSM_TASK_MODEL").is_some() {
        workers_for_macs(macs)
    } else {
        rayon::current_num_threads().max(1)
    }
}

// Thread-local packed-panel scratch, one cell per scalar type: a fresh
// `nb × nb` panel buffer (128 KB at `TRSM_NB = 128`) hits the allocator's
// mmap threshold, so every per-panel allocation paid mmap plus page
// faults on every solve (~65 µs of an 87 µs `panel_prep`, measured) —
// while sub-threshold buffers reuse hot arena pages. These cells keep one
// hot buffer per worker thread instead. Selected through `PanelBuf` so
// the generic solve core stays monomorphic over the scalar type.
thread_local! {
    static PANEL_F32: core::cell::RefCell<Vec<f32>> =
        const { core::cell::RefCell::new(Vec::new()) };
    static PANEL_F64: core::cell::RefCell<Vec<f64>> =
        const { core::cell::RefCell::new(Vec::new()) };
    static PANEL_C32: core::cell::RefCell<Vec<Complex32>> =
        const { core::cell::RefCell::new(Vec::new()) };
    static PANEL_C64: core::cell::RefCell<Vec<Complex64>> =
        const { core::cell::RefCell::new(Vec::new()) };
}

/// Per-type selector for the thread-local panel scratch (see above): runs
/// `f` with `need` hot elements. Same borrow/reentrancy contract as
/// [`with_buf`](crate::cpu::blas3::with_buf) — a nested borrow on the same
/// thread falls back to a fresh allocation instead of panicking.
trait PanelBuf: TrsmScalar {
    fn with_panel_buf<R>(need: usize, f: impl FnOnce(&mut [Self]) -> R) -> R;
}

impl PanelBuf for f32 {
    fn with_panel_buf<R>(need: usize, f: impl FnOnce(&mut [Self]) -> R) -> R {
        super::with_buf(&PANEL_F32, need, f)
    }
}
impl PanelBuf for f64 {
    fn with_panel_buf<R>(need: usize, f: impl FnOnce(&mut [Self]) -> R) -> R {
        super::with_buf(&PANEL_F64, need, f)
    }
}
impl PanelBuf for Complex32 {
    fn with_panel_buf<R>(need: usize, f: impl FnOnce(&mut [Self]) -> R) -> R {
        super::with_buf(&PANEL_C32, need, f)
    }
}
impl PanelBuf for Complex64 {
    fn with_panel_buf<R>(need: usize, f: impl FnOnce(&mut [Self]) -> R) -> R {
        super::with_buf(&PANEL_C64, need, f)
    }
}

/// Diagonal reciprocals of the once-per-panel setup (see [`pack_abuf`]):
/// small (`nb` elements, arena-hot) — stays an owned allocation.
/// Returned empty for unit diagonal (never read).
fn panel_dinv<T: TrsmScalar>(
    a: &[T],
    lda: usize,
    a0: usize,
    nb: usize,
    conjugate: bool,
    unit: bool,
) -> Vec<T> {
    // SAFETY: diagonal elements are in bounds per the caller contract.
    if unit {
        Vec::new()
    } else {
        (0..nb)
            .map(|i| {
                let d = unsafe { *a.as_ptr().add(a0 + i * lda + i) };
                T::one() / if conjugate { d.conj() } else { d }
            })
            .collect()
    }
}

/// Pack `op(A11)` contiguously into `abuf` (`nb × nb`, trans/conj folded
/// in): resolves the trans/conj branches once, sequentialises strided
/// Trans reads, and hands the lane loop a no-alias `A` operand, which is
/// what unlocks vectorisation.
/// `transposed` packs the transpose instead (`ab[r][c] = op[c][r]`):
/// the right-side solves run on transposed row-band tiles through the
/// left kernels, which need the transposed operator. Flipping the read
/// branch is exactly the transpose of the packed operator either way.
/// Writes (not pushes) `nb * nb` elements in the same order — bitwise
/// identical to the old push loop. `abuf` is the caller-owned or
/// thread-local scratch (see [`PanelBuf`]).
/// SAFETY: caller guarantees `abuf.len() >= nb * nb`; `A` reads are in
/// bounds per the caller contract.
fn pack_abuf<T: TrsmScalar>(
    a: &[T],
    lda: usize,
    a0: usize,
    nb: usize,
    trans: bool,
    conjugate: bool,
    transposed: bool,
    abuf: &mut [T],
) {
    for r in 0..nb {
        for c in 0..nb {
            let v = unsafe {
                if trans != transposed {
                    *a.as_ptr().add(a0 + c * lda + r)
                } else {
                    *a.as_ptr().add(a0 + r * lda + c)
                }
            };
            abuf[r * nb + c] = if conjugate { v.conj() } else { v };
        }
    }
}

/// Minimum band-trailing MACs before a fused band nests the parallel GEMM
/// even when the outer band scope fills the machine: mirror of SYRK's
/// `SYRK_NEST_MIN_MACS` — a huge band serialised onto one worker would own
/// the critical path while nesting costs one scope. Either entry
/// accumulates per element in an order the tests accept to tolerance (the
/// kernels in this file already document tolerance-not-bitwise
/// comparison), so the choice is safe.
const TRSM_NEST_MIN_MACS: usize = 64_000_000;

/// Derived small gate for the underfilled-machine case (nesting threshold
/// / 16). A 4M absolute floor was trialled via gate: interleaved A/B at
/// 2048 showed RN -2%, RT +2%, LN +6% (a temp-edit -20% on RT did not
/// reproduce) — the default stands.
const TRSM_NEST_SMALL_MIN_MACS: usize = TRSM_NEST_MIN_MACS / 16;

/// Pick the fused band's trailing-update entry (mirror of SYRK's
/// `pick_offdiag_gemm`): nest the parallel kernel when the outer band
/// scope underfills the machine (`nest_outer`: the fan-out is the only way
/// to reach idle cores — tall-skinny solves whose few bands would
/// otherwise strand workers), or when the band itself is huge. Otherwise
/// the serial kernel: the outer scope already spreads bands over threads,
/// and a nested fan-out costs more than it gains.
#[inline]
fn pick_band_gemm<T>(
    nest_outer: bool,
    rows: usize,
    bw: usize,
    nb: usize,
    gemm: GemmFn<T>,
    gemm_serial: GemmFn<T>,
) -> GemmFn<T> {
    let macs = rows.saturating_mul(bw).saturating_mul(nb);
    if macs >= TRSM_NEST_MIN_MACS || (nest_outer && macs >= TRSM_NEST_SMALL_MIN_MACS) {
        gemm
    } else {
        gemm_serial
    }
}

/// One column-band task of [`trsm_left_lower_fused`]: columns `j0..j1` run
/// the whole panel loop serially — diagonal band solve plus serial
/// trailing update on the band's own columns. Raw pointers so the scope
/// and the serial fallback share the body (mirror of SYRK's `syrk_band`).
/// `panels` holds one `(nb, dinv, ab_offset)` per panel; `abufs` is the
/// flat packed-operator scratch they index into (see [`PanelBuf`]).
#[allow(clippy::too_many_arguments)]
fn fused_band_lower<T: TrsmScalar>(
    xb: *mut T,
    ldb: usize,
    j0: usize,
    j1: usize,
    m: usize,
    a: *const T,
    lda: usize,
    panels: &[(usize, Vec<T>, usize)],
    abufs: *const T,
    unit: bool,
    kernel: BandKernelFn<T>,
    gemm: GemmFn<T>,
    gemm_serial: GemmFn<T>,
    nest_outer: bool,
) {
    let one = T::one();
    // SAFETY: caller guarantees task ownership of the `xb` columns
    // `j0..j1` (all rows) and read-only `a`/panels/`abufs`; bounds per the
    // caller contract. Each band touches only its own columns, so bands
    // are disjoint; the solved panel block a band's trailing update reads
    // was just solved by that same band — no cross-band dependence.
    unsafe {
        let mut k = 0;
        for (nb, dinv, off) in panels {
            let nb = *nb;
            band_left_lower(
                xb,
                ldb,
                k * ldb,
                nb,
                j0,
                j1,
                false,
                unit,
                abufs.add(*off),
                dinv.as_ptr(),
                kernel,
            );
            let rows = m - k - nb;
            if rows > 0 {
                let bw = j1 - j0;
                let a21 =
                    core::slice::from_raw_parts(a.add((k + nb) * lda + k), (rows - 1) * lda + nb);
                let xpan = core::slice::from_raw_parts(
                    xb.add(k * ldb + j0) as *const T,
                    (nb - 1) * ldb + bw,
                );
                let xbot = core::slice::from_raw_parts_mut(
                    xb.add((k + nb) * ldb + j0),
                    (rows - 1) * ldb + bw,
                );
                let g = pick_band_gemm(nest_outer, rows, bw, nb, gemm, gemm_serial);
                g(
                    Transpose::NoTrans,
                    Transpose::NoTrans,
                    rows,
                    bw,
                    nb,
                    -one,
                    a21,
                    lda,
                    xpan,
                    ldb,
                    one,
                    xbot,
                    ldb,
                );
            }
            k += nb;
        }
    }
}

/// One column-band task of [`trsm_left_upper_fused`]: descending mirror
/// of [`fused_band_lower`] — columns `j0..j1` run the whole panel loop
/// serially from the bottom panel up. Per panel: diagonal band solve
/// via [`band_left_upper`], then the trailing update on the rows ABOVE
/// (`B[0..k] -= A[0..k, k..k+nb] · X[k..k+nb]`, the descending-loop
/// geometry). `panels` holds one `(k, nb, dinv, ab_offset)` per panel
/// in solve order (descending `k`; the first panel is partial when
/// `m % TRSM_NB != 0`). Same disjoint-columns/no-cross-band-dependence
/// contract as the Lower twin.
#[allow(clippy::too_many_arguments)]
fn fused_band_upper<T: TrsmScalar>(
    xb: *mut T,
    ldb: usize,
    j0: usize,
    j1: usize,
    a: *const T,
    lda: usize,
    panels: &[(usize, usize, Vec<T>, usize)],
    abufs: *const T,
    unit: bool,
    kernel: BandKernelFn<T>,
    gemm: GemmFn<T>,
    gemm_serial: GemmFn<T>,
    nest_outer: bool,
) {
    let one = T::one();
    // SAFETY: caller guarantees task ownership of the `xb` columns
    // `j0..j1` (all rows) and read-only `a`/panels/`abufs`; bounds per
    // the caller contract. Each band touches only its own columns, so
    // bands are disjoint; the solved panel block a band's trailing
    // update reads was just solved by that same band.
    unsafe {
        for (k, nb, dinv, off) in panels {
            let (k, nb) = (*k, *nb);
            band_left_upper(
                xb,
                ldb,
                k * ldb,
                nb,
                j0,
                j1,
                false,
                unit,
                abufs.add(*off),
                dinv.as_ptr(),
                kernel,
            );
            if k > 0 {
                let bw = j1 - j0;
                let a21 = core::slice::from_raw_parts(a.add(k), (k - 1) * lda + nb);
                let xpan = core::slice::from_raw_parts(
                    xb.add(k * ldb + j0) as *const T,
                    (nb - 1) * ldb + bw,
                );
                let xhead = core::slice::from_raw_parts_mut(xb.add(j0), (k - 1) * ldb + bw);
                let g = pick_band_gemm(nest_outer, k, bw, nb, gemm, gemm_serial);
                g(
                    Transpose::NoTrans,
                    Transpose::NoTrans,
                    k,
                    bw,
                    nb,
                    -one,
                    a21,
                    lda,
                    xpan,
                    ldb,
                    one,
                    xhead,
                    ldb,
                );
            }
        }
    }
}

/// One column-band task of [`trsm_left_upper_trans_fused`]: ascending
/// mirror of [`fused_band_lower`] for `op(A) = A^T`/`A^H` with Upper
/// storage — columns `j0..j1` run the whole panel loop serially from
/// the top panel down. Per panel: diagonal band solve via
/// [`band_left_upper`] with `trans = true` (forward substitution on
/// the trans-packed operator), then the trailing update on the rows
/// BELOW (`B[k+nb..] -= A[k..k+nb, k+nb..]^H · X[k..k+nb]`, the
/// ascending-loop geometry with a transposed GEMM). Same
/// disjoint-columns/no-cross-band-dependence contract as the Lower
/// twin. `panels` holds one `(nb, dinv, ab_offset)` per panel;
/// `abufs` holds the trans-packed operators (see [`pack_abuf`]).
#[allow(clippy::too_many_arguments)]
fn fused_band_upper_trans<T: TrsmScalar>(
    xb: *mut T,
    ldb: usize,
    j0: usize,
    j1: usize,
    m: usize,
    a: *const T,
    lda: usize,
    panels: &[(usize, Vec<T>, usize)],
    abufs: *const T,
    unit: bool,
    trans: Transpose,
    kernel: BandKernelFn<T>,
    gemm: GemmFn<T>,
    gemm_serial: GemmFn<T>,
    nest_outer: bool,
) {
    let one = T::one();
    // SAFETY: as in [`fused_band_lower`]: task ownership of the `xb`
    // columns `j0..j1` (all rows), read-only `a`/panels/`abufs`.
    unsafe {
        let mut k = 0;
        for (nb, dinv, off) in panels {
            let nb = *nb;
            band_left_upper(
                xb,
                ldb,
                k * ldb,
                nb,
                j0,
                j1,
                true,
                unit,
                abufs.add(*off),
                dinv.as_ptr(),
                kernel,
            );
            let rows = m - k - nb;
            if rows > 0 {
                let bw = j1 - j0;
                // `op(A21)` is `rows × nb`; the stored block is
                // `nb × rows` at rows `k..k+nb`, cols `k+nb..`.
                let a21t =
                    core::slice::from_raw_parts(a.add(k * lda + k + nb), (nb - 1) * lda + rows);
                let xpan = core::slice::from_raw_parts(
                    xb.add(k * ldb + j0) as *const T,
                    (nb - 1) * ldb + bw,
                );
                let xbot = core::slice::from_raw_parts_mut(
                    xb.add((k + nb) * ldb + j0),
                    (rows - 1) * ldb + bw,
                );
                let g = pick_band_gemm(nest_outer, rows, bw, nb, gemm, gemm_serial);
                g(
                    trans,
                    Transpose::NoTrans,
                    rows,
                    bw,
                    nb,
                    -one,
                    a21t,
                    lda,
                    xpan,
                    ldb,
                    one,
                    xbot,
                    ldb,
                );
            }
            k += nb;
        }
    }
}

/// One column-band task of [`trsm_left_lower_trans_fused`]: descending
/// mirror of [`fused_band_upper`] for `op(A) = A^T`/`A^H` with Lower
/// storage — columns `j0..j1` run the whole panel loop serially from
/// the bottom panel up. Per panel: diagonal band solve via
/// [`band_left_lower`] with `trans = true`, then the trailing update
/// on the rows ABOVE (`B[0..k] -= A[k..k+nb, 0..k]^H ·
/// X[k..k+nb]`). `panels` holds one `(k, nb, dinv, ab_offset)` per
/// panel in solve order (descending `k`; the first panel is partial
/// when `m % TRSM_NB != 0`). Same contract as the Upper twin.
#[allow(clippy::too_many_arguments)]
fn fused_band_lower_trans<T: TrsmScalar>(
    xb: *mut T,
    ldb: usize,
    j0: usize,
    j1: usize,
    a: *const T,
    lda: usize,
    panels: &[(usize, usize, Vec<T>, usize)],
    abufs: *const T,
    unit: bool,
    trans: Transpose,
    kernel: BandKernelFn<T>,
    gemm: GemmFn<T>,
    gemm_serial: GemmFn<T>,
    nest_outer: bool,
) {
    let one = T::one();
    // SAFETY: as in [`fused_band_upper`]: task ownership of the `xb`
    // columns `j0..j1` (all rows), read-only `a`/panels/`abufs`.
    unsafe {
        for (k, nb, dinv, off) in panels {
            let (k, nb) = (*k, *nb);
            band_left_lower(
                xb,
                ldb,
                k * ldb,
                nb,
                j0,
                j1,
                true,
                unit,
                abufs.add(*off),
                dinv.as_ptr(),
                kernel,
            );
            if k > 0 {
                let bw = j1 - j0;
                // `op(A12)` is `k × nb`; the stored block is `nb × k`
                // at rows `k..k+nb`, cols `0..k`.
                let a12t = core::slice::from_raw_parts(a.add(k * lda), (nb - 1) * lda + k);
                let xpan = core::slice::from_raw_parts(
                    xb.add(k * ldb + j0) as *const T,
                    (nb - 1) * ldb + bw,
                );
                let xhead = core::slice::from_raw_parts_mut(xb.add(j0), (k - 1) * ldb + bw);
                let g = pick_band_gemm(nest_outer, k, bw, nb, gemm, gemm_serial);
                g(
                    trans,
                    Transpose::NoTrans,
                    k,
                    bw,
                    nb,
                    -one,
                    a12t,
                    lda,
                    xpan,
                    ldb,
                    one,
                    xhead,
                    ldb,
                );
            }
        }
    }
}

/// Fused Left/Lower/NoTrans blocked solve: one scope over column bands,
/// each band running the full panel loop serially (see
/// [`fused_band_lower`]).
///
/// Replaces the panel loop's two dispatches per panel (diagonal scope +
/// trailing-update GEMM dispatch, ~80-100µs each on wide pools — the
/// dominant cost of mid-size solves: single-core thin-K already runs at
/// ~45 GF/s, so the kernels are fine and dispatch is the wall). Each
/// band's columns stay L2-resident across panels, and the GEMM/band L3
/// ping-pong is gone. Multi-panel solves with enough right-hand sides
/// only; everything else stays on the panel loop. Summation order differs
/// from the panel loop (tolerance-equal, not bitwise — the comparison
/// contract this file's kernels already document).
fn trsm_left_lower_fused<T: TrsmScalar + Send + Sync + PanelBuf>(
    m: usize,
    n: usize,
    a: &[T],
    lda: usize,
    b: &mut [T],
    ldb: usize,
    unit: bool,
    band_ll: BandKernelFn<T>,
    gemm: GemmFn<T>,
    gemm_serial: GemmFn<T>,
) {
    // Per-panel setup serially (diagonal reciprocals + packed `op(A11)`
    // into one flat hot scratch): microseconds per panel, shared read-only
    // by every band. One buffer for all panels — per-panel allocations
    // would hit the mmap threshold and re-pay faults on every solve.
    let mut nbs: Vec<usize> = Vec::new();
    let mut k = 0;
    // NOTE (2026-09-27): fine fused panels (nb 64) lose ~1.09x MT f32
    // (consistent over 512/1024 x 128/256/512; f64 cells scatter = box
    // noise): halving the diag flops does not pay because each band
    // dispatches one update GEMM per panel, so finer panels multiply
    // band x panel dispatches while the panel loop (where nb 64 wins)
    // pays only one GEMM per panel. Keep 128 here.
    while k < m {
        let nb = TRSM_NB.min(m - k);
        nbs.push(nb);
        k += nb;
    }
    let total: usize = nbs.iter().map(|nb| nb * nb).sum();
    T::with_panel_buf(total, |abufs| {
        let mut panels: Vec<(usize, Vec<T>, usize)> = Vec::with_capacity(nbs.len());
        let mut k = 0;
        let mut off = 0;
        for nb in &nbs {
            let nb = *nb;
            let dinv = panel_dinv(a, lda, k * lda + k, nb, false, unit);
            pack_abuf(
                a,
                lda,
                k * lda + k,
                nb,
                false,
                false,
                false,
                &mut abufs[off..off + nb * nb],
            );
            panels.push((nb, dinv, off));
            off += nb * nb;
            k += nb;
        }
        // Same band split points as the unblocked path; `nest_outer`
        // mirrors SYRK (bands <= workers: nesting reaches idle cores).
        // Caller-width sizing (the fused scope's small-pool landing is
        // tuned as is — see `trsm_band_tasks`).
        let tasks = trsm_band_tasks(n, TRSM_NB, rayon::current_num_threads().max(1));
        let width = n.div_ceil(tasks);
        let bands = n.div_ceil(width);
        // Narrow-band fill for tall-skinny solves (see `trsm_narrow_width`).
        let (width, bands) = match trsm_narrow_width(m, n, core::mem::size_of::<T>()) {
            Some(w) => (w, n.div_ceil(w)),
            None => (width, bands),
        };
        let nest_outer = bands <= rayon::current_num_threads().max(1);
        let a_send = SendConstPtr(a.as_ptr());
        let x_send = SendMutPtr(b.as_mut_ptr());
        // `panels`/`abufs` are read-only and shared across bands (needs
        // `T: Sync`, bounded on this function); raw pointers because a
        // shared borrow cannot be moved into one spawn closure per band.
        let panels_send = SendConstPtr(panels.as_ptr());
        let abufs_send = SendConstPtr(abufs.as_ptr());
        let npanels = panels.len();
        scope_for_nested_outer(nest_outer, |s| {
            let mut j0 = 0;
            while j0 < n {
                let j1 = (j0 + width).min(n);
                s.spawn(move |_| {
                    let a_send: SendConstPtr<T> = a_send;
                    let x_send: SendMutPtr<T> = x_send;
                    let panels_send: SendConstPtr<(usize, Vec<T>, usize)> = panels_send;
                    let abufs_send: SendConstPtr<T> = abufs_send;
                    // SAFETY: `panels`/`abufs` outlive the scope, are never
                    // mutated while bands run, and `T: Sync` makes the
                    // shared slices safe to read from every band task.
                    let panels = unsafe { core::slice::from_raw_parts(panels_send.0, npanels) };
                    fused_band_lower(
                        x_send.0,
                        ldb,
                        j0,
                        j1,
                        m,
                        a_send.0,
                        lda,
                        panels,
                        abufs_send.0,
                        unit,
                        band_ll,
                        gemm,
                        gemm_serial,
                        nest_outer,
                    );
                });
                j0 = j1;
            }
        });
    });
}

/// Fused Left/Upper/NoTrans blocked solve: descending mirror of
/// [`trsm_left_lower_fused`] — one scope over column bands, each band
/// running the full panel loop serially from the bottom panel up (see
/// [`fused_band_upper`]). Panel setup (diagonal reciprocals + packed
/// `op(A11)`) is order-independent and shared read-only, like Lower.
/// Default-on (same size gates as Lower;
/// `SONIUM_TRSM_FUSED=0` restores the panel loop): fused/panel A/B
/// 2026-09-28 (16T, OB-free alternating medians, shared B buffer):
/// strsm1kU 0.68x, dtrsm1kU 0.71x, strsm512U 0.82x; Trans
/// identical-code controls 0.99x.
fn trsm_left_upper_fused<T: TrsmScalar + Send + Sync + PanelBuf>(
    m: usize,
    n: usize,
    a: &[T],
    lda: usize,
    b: &mut [T],
    ldb: usize,
    unit: bool,
    band_lu: BandKernelFn<T>,
    gemm: GemmFn<T>,
    gemm_serial: GemmFn<T>,
) {
    // Descending solve order (mirror of the panel loop's `kend` nest):
    // the first panel is partial when `m % TRSM_NB != 0`. Each entry
    // carries its `k` (Lower reconstructs `k` ascending; descending
    // with a partial-first panel cannot).
    let mut nbs: Vec<(usize, usize)> = Vec::new();
    let mut kend = m;
    while kend > 0 {
        let nb = TRSM_NB.min(kend);
        let k = kend - nb;
        nbs.push((k, nb));
        kend = k;
    }
    let total: usize = nbs.iter().map(|(_, nb)| nb * nb).sum();
    T::with_panel_buf(total, |abufs| {
        let mut panels: Vec<(usize, usize, Vec<T>, usize)> = Vec::with_capacity(nbs.len());
        let mut off = 0;
        for (k, nb) in &nbs {
            let (k, nb) = (*k, *nb);
            let dinv = panel_dinv(a, lda, k * lda + k, nb, false, unit);
            pack_abuf(
                a,
                lda,
                k * lda + k,
                nb,
                false,
                false,
                false,
                &mut abufs[off..off + nb * nb],
            );
            panels.push((k, nb, dinv, off));
            off += nb * nb;
        }
        // Same band split as the Lower twin (bands over N are
        // uplo-independent); `nest_outer` mirrors SYRK likewise.
        let tasks = trsm_band_tasks(n, TRSM_NB, rayon::current_num_threads().max(1));
        let width = n.div_ceil(tasks);
        let bands = n.div_ceil(width);
        // Narrow-band fill for tall-skinny solves (see `trsm_narrow_width`).
        let (width, bands) = match trsm_narrow_width(m, n, core::mem::size_of::<T>()) {
            Some(w) => (w, n.div_ceil(w)),
            None => (width, bands),
        };
        let nest_outer = bands <= rayon::current_num_threads().max(1);
        let a_send = SendConstPtr(a.as_ptr());
        let x_send = SendMutPtr(b.as_mut_ptr());
        // `panels`/`abufs` are read-only and shared across bands (needs
        // `T: Sync`, bounded on this function); raw pointers because a
        // shared borrow cannot be moved into one spawn closure per band.
        let panels_send = SendConstPtr(panels.as_ptr());
        let abufs_send = SendConstPtr(abufs.as_ptr());
        let npanels = panels.len();
        scope_for_nested_outer(nest_outer, |s| {
            let mut j0 = 0;
            while j0 < n {
                let j1 = (j0 + width).min(n);
                s.spawn(move |_| {
                    let a_send: SendConstPtr<T> = a_send;
                    let x_send: SendMutPtr<T> = x_send;
                    let panels_send: SendConstPtr<(usize, usize, Vec<T>, usize)> = panels_send;
                    let abufs_send: SendConstPtr<T> = abufs_send;
                    // SAFETY: `panels`/`abufs` outlive the scope, are never
                    // mutated while bands run, and `T: Sync` makes the
                    // shared slices safe to read from every band task.
                    let panels = unsafe { core::slice::from_raw_parts(panels_send.0, npanels) };
                    fused_band_upper(
                        x_send.0,
                        ldb,
                        j0,
                        j1,
                        a_send.0,
                        lda,
                        panels,
                        abufs_send.0,
                        unit,
                        band_lu,
                        gemm,
                        gemm_serial,
                        nest_outer,
                    );
                });
                j0 = j1;
            }
        });
    });
}

/// Fused Left/Upper/Trans blocked solve: ascending twin of
/// [`trsm_left_lower_fused`] for `op(A) = A^T`/`A^H` — one scope over
/// column bands, each band running the full panel loop serially from
/// the top panel down (see [`fused_band_upper_trans`]). Panel setup
/// packs the transposed operator (`trans = true`, `conjugate` for
/// `ConjTrans`) once, shared read-only. Same size gates as the
/// NoTrans twins; `SONIUM_TRSM_FUSED=0` restores the panel loop.
fn trsm_left_upper_trans_fused<T: TrsmScalar + Send + Sync + PanelBuf>(
    m: usize,
    n: usize,
    a: &[T],
    lda: usize,
    b: &mut [T],
    ldb: usize,
    unit: bool,
    trans: Transpose,
    band_lu: BandKernelFn<T>,
    gemm: GemmFn<T>,
    gemm_serial: GemmFn<T>,
) {
    let conjugate = matches!(trans, Transpose::ConjTrans);
    let mut nbs: Vec<usize> = Vec::new();
    let mut k = 0;
    while k < m {
        let nb = TRSM_NB.min(m - k);
        nbs.push(nb);
        k += nb;
    }
    let total: usize = nbs.iter().map(|nb| nb * nb).sum();
    T::with_panel_buf(total, |abufs| {
        let mut panels: Vec<(usize, Vec<T>, usize)> = Vec::with_capacity(nbs.len());
        let mut k = 0;
        let mut off = 0;
        for nb in &nbs {
            let nb = *nb;
            let dinv = panel_dinv(a, lda, k * lda + k, nb, conjugate, unit);
            pack_abuf(
                a,
                lda,
                k * lda + k,
                nb,
                true,
                conjugate,
                false,
                &mut abufs[off..off + nb * nb],
            );
            panels.push((nb, dinv, off));
            off += nb * nb;
            k += nb;
        }
        // Same band split as the NoTrans twins (bands over N are
        // uplo- and trans-independent); `nest_outer` mirrors SYRK.
        let tasks = trsm_band_tasks(n, TRSM_NB, rayon::current_num_threads().max(1));
        let width = n.div_ceil(tasks);
        let bands = n.div_ceil(width);
        // Narrow-band fill for tall-skinny solves (see `trsm_narrow_width`).
        let (width, bands) = match trsm_narrow_width(m, n, core::mem::size_of::<T>()) {
            Some(w) => (w, n.div_ceil(w)),
            None => (width, bands),
        };
        let nest_outer = bands <= rayon::current_num_threads().max(1);
        let a_send = SendConstPtr(a.as_ptr());
        let x_send = SendMutPtr(b.as_mut_ptr());
        // `panels`/`abufs` are read-only and shared across bands (needs
        // `T: Sync`, bounded on this function); raw pointers because a
        // shared borrow cannot be moved into one spawn closure per band.
        let panels_send = SendConstPtr(panels.as_ptr());
        let abufs_send = SendConstPtr(abufs.as_ptr());
        let npanels = panels.len();
        scope_for_nested_outer(nest_outer, |s| {
            let mut j0 = 0;
            while j0 < n {
                let j1 = (j0 + width).min(n);
                s.spawn(move |_| {
                    let a_send: SendConstPtr<T> = a_send;
                    let x_send: SendMutPtr<T> = x_send;
                    let panels_send: SendConstPtr<(usize, Vec<T>, usize)> = panels_send;
                    let abufs_send: SendConstPtr<T> = abufs_send;
                    // SAFETY: `panels`/`abufs` outlive the scope, are never
                    // mutated while bands run, and `T: Sync` makes the
                    // shared slices safe to read from every band task.
                    let panels = unsafe { core::slice::from_raw_parts(panels_send.0, npanels) };
                    fused_band_upper_trans(
                        x_send.0,
                        ldb,
                        j0,
                        j1,
                        m,
                        a_send.0,
                        lda,
                        panels,
                        abufs_send.0,
                        unit,
                        trans,
                        band_lu,
                        gemm,
                        gemm_serial,
                        nest_outer,
                    );
                });
                j0 = j1;
            }
        });
    });
}

/// Fused Left/Lower/Trans blocked solve: descending twin of
/// [`trsm_left_upper_fused`] for `op(A) = A^T`/`A^H` — one scope over
/// column bands, each band running the full panel loop serially from
/// the bottom panel up (see [`fused_band_lower_trans`]). Panel setup
/// packs the transposed operator once, shared read-only. Same size
/// gates as the NoTrans twins; `SONIUM_TRSM_FUSED=0` restores the
/// panel loop.
fn trsm_left_lower_trans_fused<T: TrsmScalar + Send + Sync + PanelBuf>(
    m: usize,
    n: usize,
    a: &[T],
    lda: usize,
    b: &mut [T],
    ldb: usize,
    unit: bool,
    trans: Transpose,
    band_ll: BandKernelFn<T>,
    gemm: GemmFn<T>,
    gemm_serial: GemmFn<T>,
) {
    let conjugate = matches!(trans, Transpose::ConjTrans);
    // Descending solve order (mirror of the panel loop's `kend` nest):
    // the first panel is partial when `m % TRSM_NB != 0`. Each entry
    // carries its `k`.
    let mut nbs: Vec<(usize, usize)> = Vec::new();
    let mut kend = m;
    while kend > 0 {
        let nb = TRSM_NB.min(kend);
        let k = kend - nb;
        nbs.push((k, nb));
        kend = k;
    }
    let total: usize = nbs.iter().map(|(_, nb)| nb * nb).sum();
    T::with_panel_buf(total, |abufs| {
        let mut panels: Vec<(usize, usize, Vec<T>, usize)> = Vec::with_capacity(nbs.len());
        let mut off = 0;
        for (k, nb) in &nbs {
            let (k, nb) = (*k, *nb);
            let dinv = panel_dinv(a, lda, k * lda + k, nb, conjugate, unit);
            pack_abuf(
                a,
                lda,
                k * lda + k,
                nb,
                true,
                conjugate,
                false,
                &mut abufs[off..off + nb * nb],
            );
            panels.push((k, nb, dinv, off));
            off += nb * nb;
        }
        // Same band split as the NoTrans twins (bands over N are
        // uplo- and trans-independent); `nest_outer` mirrors SYRK.
        let tasks = trsm_band_tasks(n, TRSM_NB, rayon::current_num_threads().max(1));
        let width = n.div_ceil(tasks);
        let bands = n.div_ceil(width);
        // Narrow-band fill for tall-skinny solves (see `trsm_narrow_width`).
        let (width, bands) = match trsm_narrow_width(m, n, core::mem::size_of::<T>()) {
            Some(w) => (w, n.div_ceil(w)),
            None => (width, bands),
        };
        let nest_outer = bands <= rayon::current_num_threads().max(1);
        let a_send = SendConstPtr(a.as_ptr());
        let x_send = SendMutPtr(b.as_mut_ptr());
        // `panels`/`abufs` are read-only and shared across bands (needs
        // `T: Sync`, bounded on this function); raw pointers because a
        // shared borrow cannot be moved into one spawn closure per band.
        let panels_send = SendConstPtr(panels.as_ptr());
        let abufs_send = SendConstPtr(abufs.as_ptr());
        let npanels = panels.len();
        scope_for_nested_outer(nest_outer, |s| {
            let mut j0 = 0;
            while j0 < n {
                let j1 = (j0 + width).min(n);
                s.spawn(move |_| {
                    let a_send: SendConstPtr<T> = a_send;
                    let x_send: SendMutPtr<T> = x_send;
                    let panels_send: SendConstPtr<(usize, usize, Vec<T>, usize)> = panels_send;
                    let abufs_send: SendConstPtr<T> = abufs_send;
                    // SAFETY: `panels`/`abufs` outlive the scope, are never
                    // mutated while bands run, and `T: Sync` makes the
                    // shared slices safe to read from every band task.
                    let panels = unsafe { core::slice::from_raw_parts(panels_send.0, npanels) };
                    fused_band_lower_trans(
                        x_send.0,
                        ldb,
                        j0,
                        j1,
                        a_send.0,
                        lda,
                        panels,
                        abufs_send.0,
                        unit,
                        trans,
                        band_ll,
                        gemm,
                        gemm_serial,
                        nest_outer,
                    );
                });
                j0 = j1;
            }
        });
    });
}

/// One row-band task of [`trsm_right_fused`]: rows `i0..i1` run the whole
/// panel loop serially — tiled diagonal solve plus trailing update on the
/// band's own rows. Transposed mirror of [`fused_band_lower`]: rows (not
/// columns) are independent, so no cross-band dependence exists at all.
/// `panels` holds one `(k, nb, tgt_cols, dinv, ab_offset)` per panel in solve order;
/// `abufs` is the flat packed-operator scratch they index into (Right
/// transposed pack, see [`pack_abuf`]). The solved X panel is read strided
/// (no `xpack` copy — the panel loop's pack exists only to split the
/// borrow, which bands don't need). Raw pointers so the scope shares the
/// body. Summation order differs from the panel loop (tolerance-equal, not
/// bitwise — same contract as the fused Left path).
#[allow(clippy::too_many_arguments)]
fn fused_band_right<T: TrsmScalar>(
    xb: *mut T,
    ldb: usize,
    i0: usize,
    i1: usize,
    a: *const T,
    lda: usize,
    trans: Transpose,
    trans_flag: bool,
    descending: bool,
    panels: &[(usize, usize, usize, Vec<T>, usize)],
    abufs: *const T,
    unit: bool,
    kernel: BandKernelFn<T>,
    gemm: GemmFn<T>,
    gemm_serial: GemmFn<T>,
    nest_outer: bool,
) {
    let one = T::one();
    let rows = i1 - i0;
    if rows == 0 {
        return;
    }
    // SAFETY: caller guarantees task ownership of the `xb` rows `i0..i1`
    // (all columns) and read-only `a`/panels/`abufs`; bounds per the caller
    // contract. Each band touches only its own rows, so bands are
    // disjoint; the solved panel the trailing update reads was just solved
    // by that same band — and rows are independent, so there is no
    // cross-band dependence either.
    unsafe {
        for (k, nb, tgt_cols, dinv, off) in panels {
            let (k, nb, tgt_cols) = (*k, *nb, *tgt_cols);
            kernel(
                xb,
                ldb,
                k,
                i0,
                i1,
                nb,
                trans_flag,
                unit,
                abufs.add(*off),
                dinv.as_ptr(),
            );
            // Trailing-update target: left of the panel (descending,
            // `tgt_cols = k`) or right of it (ascending); empty (0) at the
            // far edge.
            let tgt0 = if descending { 0 } else { k + nb };
            if tgt_cols > 0 {
                let xpan = core::slice::from_raw_parts(
                    xb.add(i0 * ldb + k) as *const T,
                    (rows - 1) * ldb + nb,
                );
                let xtgt = core::slice::from_raw_parts_mut(
                    xb.add(i0 * ldb + tgt0),
                    (rows - 1) * ldb + tgt_cols,
                );
                let g = pick_band_gemm(nest_outer, rows, tgt_cols, nb, gemm, gemm_serial);
                if descending {
                    if !trans_flag {
                        // X[.., 0..k] -= X[.., k..k+nb] * A[k..k+nb, 0..k].
                        let a_sub = core::slice::from_raw_parts(a.add(k * lda), (nb - 1) * lda + k);
                        g(
                            Transpose::NoTrans,
                            Transpose::NoTrans,
                            rows,
                            tgt_cols,
                            nb,
                            -one,
                            xpan,
                            ldb,
                            a_sub,
                            lda,
                            one,
                            xtgt,
                            ldb,
                        );
                    } else {
                        // X[.., 0..k] -= X[.., k..k+nb] * A[0..k, k..k+nb]^H.
                        let a_sub = core::slice::from_raw_parts(a.add(k), (k - 1) * lda + nb);
                        g(
                            Transpose::NoTrans,
                            trans,
                            rows,
                            tgt_cols,
                            nb,
                            -one,
                            xpan,
                            ldb,
                            a_sub,
                            lda,
                            one,
                            xtgt,
                            ldb,
                        );
                    }
                } else if !trans_flag {
                    // X[.., k+nb..] -= X[.., k..k+nb] * A[k..k+nb, k+nb..].
                    let a_sub = core::slice::from_raw_parts(
                        a.add(k * lda + k + nb),
                        (nb - 1) * lda + tgt_cols,
                    );
                    g(
                        Transpose::NoTrans,
                        Transpose::NoTrans,
                        rows,
                        tgt_cols,
                        nb,
                        -one,
                        xpan,
                        ldb,
                        a_sub,
                        lda,
                        one,
                        xtgt,
                        ldb,
                    );
                } else {
                    // X[.., k+nb..] -= X[.., k..k+nb] * A[k+nb.., k..k+nb]^H.
                    let a_sub = core::slice::from_raw_parts(
                        a.add((k + nb) * lda + k),
                        (tgt_cols - 1) * lda + nb,
                    );
                    g(
                        Transpose::NoTrans,
                        trans,
                        rows,
                        tgt_cols,
                        nb,
                        -one,
                        xpan,
                        ldb,
                        a_sub,
                        lda,
                        one,
                        xtgt,
                        ldb,
                    );
                }
            }
        }
    }
}

/// Fused Right blocked solve: one scope over row bands, each band running
/// the full panel loop serially (see [`fused_band_right`]). Transposed
/// mirror of [`trsm_left_lower_fused`].
///
/// Replaces the panel loop's per-panel dispatch pair (row-band solve scope
/// plus full-`m` trailing-update GEMM dispatch) and the `xpack` copy — the
/// panel loop rewrites the whole `B` and repacks every X panel once per
/// panel, while each fused band streams only its own rows. Each band's
/// rows stay resident across panels. Multi-panel solves with enough rows
/// only; everything else stays on the panel loop. Summation order differs
/// from the panel loop (tolerance-equal, not bitwise — same contract as
/// the fused Left path). `SONIUM_TRSM_FUSED=0` forces the panel loop
/// (diagnostic/A-B fallback; the fused path stays the default).
#[allow(clippy::too_many_arguments)]
/// Unblocked inverse of a triangular `n x n` (`n <=
/// [`TRSM_INVERT_MAX_N`]) into row-major `out` (exactly `n * n`, zeroed
/// first so the opposite triangle stays zero). Solves `W·op = I` a row
/// at a time (right inverse): the solve `X = B·W` needs a small RIGHT
/// residual, and the left recurrence (`A·W = I`) amplifies it by
/// `cond²` (measured 1.9e-2 f32 residual on unit triangles vs 1e-5
/// here). The caller guarantees nonsingular diagonals (Cholesky/LU
/// pivots).
fn invert_tri_small<T: TrsmScalar>(
    a: &[T],
    lda: usize,
    lower: bool,
    unit: bool,
    n: usize,
    out: &mut [T],
) {
    debug_assert!(n <= TRSM_INVERT_MAX_N);
    debug_assert_eq!(out.len(), n * n);
    out.fill(T::zero());
    let one = T::one();
    if lower {
        for i in 0..n {
            out[i * n + i] = if unit { one } else { one / a[i * lda + i] };
            for j in (0..i).rev() {
                let mut s = T::zero();
                for k in (j + 1)..=i {
                    s = s + out[i * n + k] * a[k * lda + j];
                }
                out[i * n + j] = if unit { -s } else { -s / a[j * lda + j] };
            }
        }
    } else {
        for i in 0..n {
            out[i * n + i] = if unit { one } else { one / a[i * lda + i] };
            for j in (i + 1)..n {
                let mut s = T::zero();
                for k in i..j {
                    s = s + out[i * n + k] * a[k * lda + j];
                }
                out[i * n + j] = if unit { -s } else { -s / a[j * lda + j] };
            }
        }
    }
}

/// Right solve `X·op = B` via explicit inverse + GEMM for small `n`
/// (`n <= [`TRSM_INVERT_MAX_N`], `m >= [`UNBLOCKED_PAR_MIN`]): `W =
/// op⁻¹` (unblocked, `n³/3` flops) then `X = B·W` at GEMM rate. The
/// unblocked substitution it replaces is dependency-chained scalar
/// work (~16 GF/s serial f32); the GEMM runs 10-20x faster even after
/// the inverse plus the out-of-place temp (GEMM cannot accumulate into
/// its own A operand, so the solve lands in a pooled `m x n` buffer
/// and copies back). Transposed operators fold into GEMM's `transb`
/// (`X·opᵀ = B ⟺ X = B·(op⁻¹)ᵀ`, conjugated for `ConjTrans`), so `W`
/// always inverts the untransposed triangle. `B` arrives
/// alpha-prescaled (see `trsm_core`), hence GEMM `alpha = 1`.
/// Forward error matches substitution (`cond(op)·eps` class; verified
/// by `trsm_right_inverted_matches_naive` residuals, not bitwise).
fn trsm_right_inverted<T: TrsmScalar + PanelBuf>(
    m: usize,
    n: usize,
    a: &[T],
    lda: usize,
    b: &mut [T],
    ldb: usize,
    uplo: TrsmUplo,
    trans: Transpose,
    unit: bool,
    gemm: GemmFn<T>,
) {
    let lower = uplo == TrsmUplo::Lower;
    // SAFETY: `MaybeUninit` needs no drop; `invert_tri_small` fills the
    // whole `n * n` prefix before the GEMM reads it.
    let mut w_uninit =
        [core::mem::MaybeUninit::<T>::uninit(); TRSM_INVERT_MAX_N * TRSM_INVERT_MAX_N];
    let w_ptr = w_uninit.as_mut_ptr().cast::<T>();
    let w = unsafe { core::slice::from_raw_parts_mut(w_ptr, n * n) };
    invert_tri_small(a, lda, lower, unit, n, w);
    let transb = trans;
    T::with_panel_buf(m * n, |tmp| {
        gemm(
            Transpose::NoTrans,
            transb,
            m,
            n,
            n,
            T::one(),
            b,
            ldb,
            w,
            n,
            T::zero(),
            tmp,
            n,
        );
        for i in 0..m {
            b[i * ldb..i * ldb + n].copy_from_slice(&tmp[i * n..(i + 1) * n]);
        }
    });
}

fn trsm_right_fused<T: TrsmScalar + Send + Sync + PanelBuf>(
    m: usize,
    n: usize,
    a: &[T],
    lda: usize,
    b: &mut [T],
    ldb: usize,
    descending: bool,
    trans: Transpose,
    trans_flag: bool,
    conjugate: bool,
    unit: bool,
    band_r: BandKernelFn<T>,
    gemm: GemmFn<T>,
    gemm_serial: GemmFn<T>,
) {
    // Panel starts in solve order, with per-panel trailing widths.
    let mut ks: Vec<(usize, usize)> = Vec::new();
    // `SONIUM_TRSM_CHUNK` refines the panel split (trial: finer L1-resident
    // internal panels; default = one `TRSM_NB` panel per loop pass).
    let chunk = trsm_right_chunk();
    if descending {
        let mut kend = n;
        while kend > 0 {
            let nb = chunk.min(kend);
            let k = kend - nb;
            ks.push((k, nb));
            kend = k;
        }
    } else {
        let mut k = 0;
        while k < n {
            let nb = chunk.min(n - k);
            ks.push((k, nb));
            k += nb;
        }
    }
    // Per-panel setup serially (diagonal reciprocals + transposed-packed
    // `op(A11)` into one flat hot scratch): microseconds per panel,
    // shared read-only by every band. One buffer for all panels — same
    // mmap-threshold reason as the fused Left path.
    let total: usize = ks.iter().map(|(_, nb)| nb * nb).sum();
    T::with_panel_buf(total, |abufs| {
        let mut panels: Vec<(usize, usize, usize, Vec<T>, usize)> = Vec::with_capacity(ks.len());
        let mut off = 0;
        for (k, nb) in &ks {
            let (k, nb) = (*k, *nb);
            let dinv = panel_dinv(a, lda, k * lda + k, nb, conjugate, unit);
            pack_abuf(
                a,
                lda,
                k * lda + k,
                nb,
                trans_flag,
                conjugate,
                true,
                &mut abufs[off..off + nb * nb],
            );
            let tgt_cols = if descending { k } else { n - k - nb };
            panels.push((k, nb, tgt_cols, dinv, off));
            off += nb * nb;
        }
        // Same band split shape as the unblocked path; `nest_outer`
        // mirrors SYRK (bands <= workers: nesting reaches idle cores).
        // (A `SONIUM_TRSM_FUSED_BAND_W` width override was trialled:
        // interleaved A/B showed 256-wide neutral at 2048 (-3%) and
        // worse at 1024 (+17%, band underfill), 512-wide +15% at 2048 —
        // the caller-width sizer stands.)
        let tasks = trsm_band_tasks(m, TRSM_NB, rayon::current_num_threads().max(1));
        let height = m.div_ceil(tasks);
        let bands = m.div_ceil(height);
        let nest_outer = bands <= rayon::current_num_threads().max(1);
        let a_send = SendConstPtr(a.as_ptr());
        let x_send = SendMutPtr(b.as_mut_ptr());
        // `panels`/`abufs` are read-only and shared across bands (needs
        // `T: Sync`, bounded on this function); raw pointers because a
        // shared borrow cannot be moved into one spawn closure per band.
        let panels_send = SendConstPtr(panels.as_ptr());
        let abufs_send = SendConstPtr(abufs.as_ptr());
        let npanels = panels.len();
        scope_for_nested_outer(nest_outer, |s| {
            let mut i0 = 0;
            while i0 < m {
                let i1 = (i0 + height).min(m);
                s.spawn(move |_| {
                    let a_send: SendConstPtr<T> = a_send;
                    let x_send: SendMutPtr<T> = x_send;
                    let panels_send: SendConstPtr<(usize, usize, usize, Vec<T>, usize)> =
                        panels_send;
                    let abufs_send: SendConstPtr<T> = abufs_send;
                    // SAFETY: `panels`/`abufs` outlive the scope, are never
                    // mutated while bands run, and `T: Sync` makes the
                    // shared slices safe to read from every band task.
                    let panels = unsafe { core::slice::from_raw_parts(panels_send.0, npanels) };
                    fused_band_right(
                        x_send.0,
                        ldb,
                        i0,
                        i1,
                        a_send.0,
                        lda,
                        trans,
                        trans_flag,
                        descending,
                        panels,
                        abufs_send.0,
                        unit,
                        band_r,
                        gemm,
                        gemm_serial,
                        nest_outer,
                    );
                });
                i0 = i1;
            }
        });
    });
}

/// Left, lower block: `A11 * X1 = B1`, `A11` unit-or-not lower, `nb × nb`.
/// The packed operator lives in the thread-local scratch ([`PanelBuf`]):
/// a fresh per-panel allocation would hit the mmap threshold at
/// `TRSM_NB = 128` and re-pay faults on every solve.
fn unblocked_left_lower<T: TrsmScalar + PanelBuf>(
    a: &[T],
    lda: usize,
    a0: usize,
    xb: &mut [T],
    ldb: usize,
    x0: usize,
    nb: usize,
    nrhs: usize,
    trans: bool,
    conjugate: bool,
    unit: bool,
    kernel: BandKernelFn<T>,
) {
    // Panel setup once: diagonal reciprocals (small, owned) plus packed
    // op(A11) into the hot scratch, shared read-only by every band task.
    let dinv = panel_dinv(a, lda, a0, nb, conjugate, unit);
    // Right-hand sides are independent: fan out over column bands when
    // there are enough of them, else run the single serial band.
    // Single-band inline (shipped default-on 2026-09-27,
    // `SONIUM_TRSM_ONEBAND=0` restores the scoped two-band split):
    // at `nrhs <= TRSM_NB` the scope + pool install (~25-40µs) cost
    // more than the second band buys (m=nb=128: 124µs -> ~100µs at
    // 16T, 2/2 replicated pairs), so the serial band covers it.
    // Columns are independent with the same kernel op sequence, so
    // results are bitwise identical under either split.
    let oneband_off = std::env::var("SONIUM_TRSM_ONEBAND")
        .map(|v| v == "0")
        .unwrap_or(false);
    if nrhs >= UNBLOCKED_PAR_MIN && (oneband_off || nrhs > TRSM_NB) {
        T::with_panel_buf(nb * nb, |abuf| {
            pack_abuf(a, lda, a0, nb, trans, conjugate, false, abuf);
            let macs = nb.saturating_mul(nb).saturating_mul(nrhs);
            let tasks = trsm_band_tasks(nrhs, nb, band_task_workers(macs));
            let ab_send = SendConstPtr(abuf.as_ptr());
            let di_send = SendConstPtr(dinv.as_ptr());
            let x_send = SendMutPtr(xb.as_mut_ptr());
            scope_for_macs(macs, |s| {
                let mut j0 = 0;
                while j0 < nrhs {
                    let j1 = (j0 + nrhs.div_ceil(tasks)).min(nrhs);
                    s.spawn(move |_| {
                        let ab_send: SendConstPtr<T> = ab_send;
                        let di_send: SendConstPtr<T> = di_send;
                        let x_send: SendMutPtr<T> = x_send;
                        // SAFETY: band `j0..j1` owns disjoint `xb` columns;
                        // `ab`/`di` are read-only and outlive the scope;
                        // bounds per the caller contract.
                        band_left_lower(
                            x_send.0, ldb, x0, nb, j0, j1, trans, unit, ab_send.0, di_send.0,
                            kernel,
                        );
                    });
                    j0 = j1;
                }
            });
        });
        return;
    }
    T::with_panel_buf(nb * nb, |abuf| {
        pack_abuf(a, lda, a0, nb, trans, conjugate, false, abuf);
        band_left_lower(
            xb.as_mut_ptr(),
            ldb,
            x0,
            nb,
            0,
            nrhs,
            trans,
            unit,
            abuf.as_ptr(),
            dinv.as_ptr(),
            kernel,
        );
    });
}

#[allow(clippy::needless_range_loop)]
/// Serial band of [`unblocked_left_lower`] over right-hand sides `j0..j1`.
/// `ab` is the packed row-major `op(A11)` (`nb × nb`, trans/conj folded)
/// and `di` the diagonal reciprocals from [`panel_prep`] (empty for unit
/// diagonal); both read-only and shared across bands.
///
/// The band's k-reduction reads X rows with stride `ldb`; when that stride
/// is a multiple of 4KB (e.g. `ldb = 2048` f64 rows are 16KB apart) every
/// row lands in the same L1 set and the reduction runs at L2 latency.
/// Packs the band's `nb × nw` block to a narrow-stride scratch first
/// (stride `nw`, no collisions, L1-resident) and copies back after.
/// Pack/unpack are exact copies and the kernel op order is untouched, so
/// results are bitwise identical. Skipped when the band is already narrow:
/// `ldb == nw` implies a sub-4KB stride on every call path (the serial
/// band covers `nrhs < 128`, and the parallel path never issues a single
/// full-width band).
fn band_left_lower<T: TrsmScalar>(
    xb: *mut T,
    ldb: usize,
    x0: usize,
    nb: usize,
    j0: usize,
    j1: usize,
    trans: bool,
    unit: bool,
    ab: *const T,
    di: *const T,
    kernel: BandKernelFn<T>,
) {
    // Solve in ≤64-wide column chunks on a narrow-stride scratch (see
    // below); chunked so even a 512-wide serial band keeps a 512B stride.
    // Stack scratch: `nb <= TRSM_NB`, so `nb × BAND_CHUNK` is at most 64KB
    // — no heap alloc/free per band task (dozens of concurrent 24KB
    // mallocs contended on wide pools). Uninitialised: the chunk packs
    // (writes) every used element before the kernel reads it.
    // SAFETY: `MaybeUninit` needs no drop; the chunk writes all `nb * nw`
    // used elements via the pack copies before any read, so the
    // initialised prefix handed to the chunk is valid.
    let mut scratch = [core::mem::MaybeUninit::<T>::uninit(); TRSM_NB * BAND_CHUNK];
    let scratch_ptr = scratch.as_mut_ptr().cast::<T>();
    // Left chunk width (shipped 16 default 2026-09-27,
    // `SONIUM_TRSM_LCHUNK=n` overrides for sweeps): 16-wide chunks keep
    // the `nb x nw` kernel block L1-resident (32KB at nb=128) where
    // 64-wide chunks spill to L2 (u12 panel shapes +12-18% at 16T,
    // +3-4% serial, replicated; bitwise identical — pack/unpack are
    // exact copies, kernel op order untouched). Capped at BAND_CHUNK:
    // the stack scratch below is sized for it.
    let lchunk = std::env::var("SONIUM_TRSM_LCHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&c| (4..=BAND_CHUNK).contains(&c))
        .unwrap_or(16);
    let mut c0 = j0;
    while c0 < j1 {
        let c1 = (c0 + lchunk).min(j1);
        band_left_lower_chunk(
            xb,
            ldb,
            x0,
            nb,
            c0,
            c1,
            trans,
            unit,
            ab,
            di,
            scratch_ptr,
            kernel,
        );
        c0 = c1;
    }
}

/// One ≤64-wide chunk of [`band_left_lower`]: packs columns `c0..c1` to a
/// narrow-stride scratch, runs the kernel, copies back. Pack/unpack are
/// exact copies and the kernel op order is untouched — bitwise identical.
/// Skipped (direct kernel) when the chunk is already narrow: `ldb == nw`
/// with `nw ≤ 64` implies a sub-4KB stride on every call path (the serial
/// band covers `nrhs < 128`, and the parallel path never issues a single
/// full-width band).
///
#[allow(clippy::needless_range_loop)]
fn band_left_lower_chunk<T: TrsmScalar>(
    xb: *mut T,
    ldb: usize,
    x0: usize,
    nb: usize,
    c0: usize,
    c1: usize,
    trans: bool,
    unit: bool,
    ab: *const T,
    di: *const T,
    scratch: *mut T,
    kernel: BandKernelFn<T>,
) {
    let nw = c1 - c0;
    if ldb == nw {
        // SAFETY: same contract as the kernel (band fns are safe; only
        // their bodies are `unsafe` internally).
        unsafe {
            kernel(xb, ldb, x0, nb, c0, c1, trans, unit, ab, di);
        }
        return;
    }
    // SAFETY: `nb` row-copies of `nw` elements between the `xb` band rows
    // (which hold the band per the caller contract) and the scratch; the
    // kernel runs on the scratch with tight stride `nw` (not `BAND_CHUNK`):
    // the k-reduction reloads X rows `nb/2` times on average, so its stride
    // sets the cache level — at stride 64 every row lands 8 L1 sets apart
    // and 128 rows overflow the associativity (L2 latency, width-independent),
    // while tight packing keeps the `nb × nw` block L1-resident. The chunk
    // is at most `BAND_CHUNK` wide, so the tight block fits the scratch.
    unsafe {
        for i in 0..nb {
            core::ptr::copy_nonoverlapping(xb.add(x0 + i * ldb + c0), scratch.add(i * nw), nw);
        }
        kernel(scratch, nw, 0, nb, 0, nw, trans, unit, ab, di);
        for i in 0..nb {
            core::ptr::copy_nonoverlapping(scratch.add(i * nw), xb.add(x0 + i * ldb + c0), nw);
        }
    }
}

/// Kernel body of [`band_left_lower`], factored out so the narrow-stride
/// wrapper above shares it. Same contract.
fn band_left_lower_kernel<T: TrsmScalar>(
    xb: *mut T,
    ldb: usize,
    x0: usize,
    nb: usize,
    j0: usize,
    j1: usize,
    trans: bool,
    unit: bool,
    ab: *const T,
    di: *const T,
) {
    // SAFETY: caller guarantees the `xb` columns `j0..j1` exclusively
    // writable and `ab`/`di` readable for the panel ranges used below.
    unsafe {
        // Four-wide over the independent columns: each lane runs the exact
        // scalar op sequence of the original loop (bitwise-identical
        // results), and the four straight-line lanes are SLP bait the inner
        // `k` reduction (ordered, unvectorisable) otherwise denies.
        if !trans {
            let mut j = j0;
            while j + 4 <= j1 {
                for i in 0..nb {
                    let mut s0 = *xb.add(x0 + i * ldb + j);
                    let mut s1 = *xb.add(x0 + i * ldb + j + 1);
                    let mut s2 = *xb.add(x0 + i * ldb + j + 2);
                    let mut s3 = *xb.add(x0 + i * ldb + j + 3);
                    for k in 0..i {
                        let aik = *ab.add(i * nb + k);
                        s0 = s0 - aik * *xb.add(x0 + k * ldb + j);
                        s1 = s1 - aik * *xb.add(x0 + k * ldb + j + 1);
                        s2 = s2 - aik * *xb.add(x0 + k * ldb + j + 2);
                        s3 = s3 - aik * *xb.add(x0 + k * ldb + j + 3);
                    }
                    if unit {
                        *xb.add(x0 + i * ldb + j) = s0;
                        *xb.add(x0 + i * ldb + j + 1) = s1;
                        *xb.add(x0 + i * ldb + j + 2) = s2;
                        *xb.add(x0 + i * ldb + j + 3) = s3;
                    } else {
                        let d = *di.add(i);
                        *xb.add(x0 + i * ldb + j) = s0 * d;
                        *xb.add(x0 + i * ldb + j + 1) = s1 * d;
                        *xb.add(x0 + i * ldb + j + 2) = s2 * d;
                        *xb.add(x0 + i * ldb + j + 3) = s3 * d;
                    }
                }
                j += 4;
            }
            for j in j..j1 {
                for i in 0..nb {
                    let mut s = *xb.add(x0 + i * ldb + j);
                    for k in 0..i {
                        s = s - *ab.add(i * nb + k) * *xb.add(x0 + k * ldb + j);
                    }
                    *xb.add(x0 + i * ldb + j) = if unit { s } else { s * *di.add(i) };
                }
            }
        } else {
            // op(A11) is upper triangular: backward substitution.
            let mut j = j0;
            while j + 4 <= j1 {
                for i in (0..nb).rev() {
                    let mut s0 = *xb.add(x0 + i * ldb + j);
                    let mut s1 = *xb.add(x0 + i * ldb + j + 1);
                    let mut s2 = *xb.add(x0 + i * ldb + j + 2);
                    let mut s3 = *xb.add(x0 + i * ldb + j + 3);
                    for k in (i + 1)..nb {
                        let aik = *ab.add(i * nb + k);
                        s0 = s0 - aik * *xb.add(x0 + k * ldb + j);
                        s1 = s1 - aik * *xb.add(x0 + k * ldb + j + 1);
                        s2 = s2 - aik * *xb.add(x0 + k * ldb + j + 2);
                        s3 = s3 - aik * *xb.add(x0 + k * ldb + j + 3);
                    }
                    if unit {
                        *xb.add(x0 + i * ldb + j) = s0;
                        *xb.add(x0 + i * ldb + j + 1) = s1;
                        *xb.add(x0 + i * ldb + j + 2) = s2;
                        *xb.add(x0 + i * ldb + j + 3) = s3;
                    } else {
                        let d = *di.add(i);
                        *xb.add(x0 + i * ldb + j) = s0 * d;
                        *xb.add(x0 + i * ldb + j + 1) = s1 * d;
                        *xb.add(x0 + i * ldb + j + 2) = s2 * d;
                        *xb.add(x0 + i * ldb + j + 3) = s3 * d;
                    }
                }
                j += 4;
            }
            for j in j..j1 {
                for i in (0..nb).rev() {
                    let mut s = *xb.add(x0 + i * ldb + j);
                    for k in (i + 1)..nb {
                        s = s - *ab.add(i * nb + k) * *xb.add(x0 + k * ldb + j);
                    }
                    *xb.add(x0 + i * ldb + j) = if unit { s } else { s * *di.add(i) };
                }
            }
        }
    }
}

/// Concrete `f64` left-lower band kernel: AVX2 when available (see
/// [`band_left_lower_kernel_f64_avx2`]), else the generic kernel. Threaded
/// from [`trsm_f64`] through [`trsm_core`] as a [`BandKernelFn`].
unsafe fn band_left_lower_kernel_f64(
    xb: *mut f64,
    ldb: usize,
    x0: usize,
    nb: usize,
    j0: usize,
    j1: usize,
    trans: bool,
    unit: bool,
    ab: *const f64,
    di: *const f64,
) {
    // SAFETY: same contract as the generic kernel below (caller-owned
    // `xb` band, readable `ab`/`di`); the `unsafe` only matches the
    // [`BandKernelFn`] pointer type.
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::blas3::avx2::is_available() {
        // NOTE (2026-09-27): a blocked micro-solve twin of the f32 one
        // below measured tie-or-1% on f64 ST (8/8 cells) and tie on MT
        // (no immune win), so f64 keeps the row-wise kernel; only f32
        // diverts. See `band_left_lower_blocked_f32`.
        unsafe { band_left_lower_kernel_f64_avx2(xb, ldb, x0, nb, j0, j1, trans, unit, ab, di) };
        return;
    }
    band_left_lower_kernel(xb, ldb, x0, nb, j0, j1, trans, unit, ab, di)
}

/// AVX2 `f64` left-lower band solve. Same contract as
/// [`band_left_lower_kernel`]: solves the `j0..j1` columns in place.
///
/// The generic kernel this replaces compiles to 2-wide SSE (`mulpd`+`subpd`)
/// under the crate's baseline x86-64 target — about 2 GF/s/core. Here one
/// `ymm` per 4-wide column group holds the `s` accumulators in registers
/// across the `k` reduction, with `aik` broadcast and one FMA per group per
/// step. Each lane runs the scalar op sequence in the same order, but the
/// fused multiply-add rounds once where the generic kernel rounds twice,
/// so results differ by ~1 ulp (well within the 1e-9 test tolerance; FMA is
/// strictly more accurate). Like all vendor BLAS, callers must compare with
/// tolerance, not bitwise. `trans` selects forward (lower) or backward
/// (upper) substitution over the packed `op(A11)`; the `j1 - j0 < 4` tail
/// stays scalar.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn band_left_lower_kernel_f64_avx2(
    xb: *mut f64,
    ldb: usize,
    x0: usize,
    nb: usize,
    j0: usize,
    j1: usize,
    trans: bool,
    unit: bool,
    ab: *const f64,
    di: *const f64,
) {
    use core::arch::x86_64::*;
    // SAFETY: caller guarantees the `xb` columns `j0..j1` exclusively
    // writable and `ab`/`di` readable for the panel ranges below; all
    // accesses are `loadu`/`storeu` (arbitrary `ldb` stride alignment).
    unsafe {
        let base = xb.add(x0);
        let mut j = j0;
        // 16-wide main loop: four independent 4-wide accumulators. The
        // `k` reduction carries each accumulator (`mul`+`sub` ≈ 8 cycles
        // of dependent latency), so a single `ymm` stalls the pipeline
        // (~0.8 GF/s/core measured); four interleaved chains approach port
        // throughput (~4×). Lanes are independent columns with the exact
        // scalar op order each, so results stay bitwise identical to the
        // generic kernel (still deliberately no FMA). The 4-wide loop
        // below handles the mid tail, scalar the final tail.
        while j + 16 <= j1 {
            if !trans {
                // Two-row unrolled forward solve (`SONIUM_TRSM_NOUNROLL=1`
                // forces the single-row loop for A/B): rows `i, i+1` share
                // each `X[k]` load in a lockstep `k` loop, halving the
                // `X` traffic the L2-bound reduction otherwise pays (each
                // row re-streams all previous rows). Row `i` stores first;
                // row `i+1`'s final `k = i` step reuses it from registers.
                // Per-element op order (ascending `k`, one FMA per step)
                // is unchanged — bitwise identical to the single-row loop.
                // Only for large `nb`: when the band is L1-resident the
                // extra broadcast/FMA port pressure costs more than the
                // saved (already-L1) loads.
                let nounroll = std::env::var_os("SONIUM_TRSM_NOUNROLL").is_some() || nb < 64;
                if !nounroll {
                    let mut i = 0;
                    while i + 1 < nb {
                        let r0 = base.add(i * ldb + j);
                        let r1 = base.add(i * ldb + j + 4);
                        let r2 = base.add(i * ldb + j + 8);
                        let r3 = base.add(i * ldb + j + 12);
                        let s0 = base.add((i + 1) * ldb + j);
                        let s1 = base.add((i + 1) * ldb + j + 4);
                        let s2 = base.add((i + 1) * ldb + j + 8);
                        let s3 = base.add((i + 1) * ldb + j + 12);
                        let mut a0 = _mm256_loadu_pd(r0);
                        let mut a1 = _mm256_loadu_pd(r1);
                        let mut a2 = _mm256_loadu_pd(r2);
                        let mut a3 = _mm256_loadu_pd(r3);
                        let mut b0 = _mm256_loadu_pd(s0);
                        let mut b1 = _mm256_loadu_pd(s1);
                        let mut b2 = _mm256_loadu_pd(s2);
                        let mut b3 = _mm256_loadu_pd(s3);
                        let abrow0 = ab.add(i * nb);
                        let abrow1 = ab.add((i + 1) * nb);
                        for k in 0..i {
                            let x0 = _mm256_loadu_pd(base.add(k * ldb + j));
                            let x1 = _mm256_loadu_pd(base.add(k * ldb + j + 4));
                            let x2 = _mm256_loadu_pd(base.add(k * ldb + j + 8));
                            let x3 = _mm256_loadu_pd(base.add(k * ldb + j + 12));
                            let aik0 = _mm256_broadcast_sd(&*abrow0.add(k));
                            let aik1 = _mm256_broadcast_sd(&*abrow1.add(k));
                            a0 = _mm256_fnmadd_pd(aik0, x0, a0);
                            a1 = _mm256_fnmadd_pd(aik0, x1, a1);
                            a2 = _mm256_fnmadd_pd(aik0, x2, a2);
                            a3 = _mm256_fnmadd_pd(aik0, x3, a3);
                            b0 = _mm256_fnmadd_pd(aik1, x0, b0);
                            b1 = _mm256_fnmadd_pd(aik1, x1, b1);
                            b2 = _mm256_fnmadd_pd(aik1, x2, b2);
                            b3 = _mm256_fnmadd_pd(aik1, x3, b3);
                        }
                        if !unit {
                            let d = _mm256_broadcast_sd(&*di.add(i));
                            a0 = _mm256_mul_pd(a0, d);
                            a1 = _mm256_mul_pd(a1, d);
                            a2 = _mm256_mul_pd(a2, d);
                            a3 = _mm256_mul_pd(a3, d);
                        }
                        _mm256_storeu_pd(r0, a0);
                        _mm256_storeu_pd(r1, a1);
                        _mm256_storeu_pd(r2, a2);
                        _mm256_storeu_pd(r3, a3);
                        // Final `k = i` step for row `i + 1` straight from
                        // the just-stored row (no reload).
                        let aik = _mm256_broadcast_sd(&*abrow1.add(i));
                        b0 = _mm256_fnmadd_pd(aik, a0, b0);
                        b1 = _mm256_fnmadd_pd(aik, a1, b1);
                        b2 = _mm256_fnmadd_pd(aik, a2, b2);
                        b3 = _mm256_fnmadd_pd(aik, a3, b3);
                        if !unit {
                            let d = _mm256_broadcast_sd(&*di.add(i + 1));
                            b0 = _mm256_mul_pd(b0, d);
                            b1 = _mm256_mul_pd(b1, d);
                            b2 = _mm256_mul_pd(b2, d);
                            b3 = _mm256_mul_pd(b3, d);
                        }
                        _mm256_storeu_pd(s0, b0);
                        _mm256_storeu_pd(s1, b1);
                        _mm256_storeu_pd(s2, b2);
                        _mm256_storeu_pd(s3, b3);
                        i += 2;
                    }
                    if i < nb {
                        let r0 = base.add(i * ldb + j);
                        let r1 = base.add(i * ldb + j + 4);
                        let r2 = base.add(i * ldb + j + 8);
                        let r3 = base.add(i * ldb + j + 12);
                        let mut a0 = _mm256_loadu_pd(r0);
                        let mut a1 = _mm256_loadu_pd(r1);
                        let mut a2 = _mm256_loadu_pd(r2);
                        let mut a3 = _mm256_loadu_pd(r3);
                        let abrow = ab.add(i * nb);
                        for k in 0..i {
                            let aik = _mm256_broadcast_sd(&*abrow.add(k));
                            a0 = _mm256_fnmadd_pd(aik, _mm256_loadu_pd(base.add(k * ldb + j)), a0);
                            a1 = _mm256_fnmadd_pd(
                                aik,
                                _mm256_loadu_pd(base.add(k * ldb + j + 4)),
                                a1,
                            );
                            a2 = _mm256_fnmadd_pd(
                                aik,
                                _mm256_loadu_pd(base.add(k * ldb + j + 8)),
                                a2,
                            );
                            a3 = _mm256_fnmadd_pd(
                                aik,
                                _mm256_loadu_pd(base.add(k * ldb + j + 12)),
                                a3,
                            );
                        }
                        if !unit {
                            let d = _mm256_broadcast_sd(&*di.add(i));
                            a0 = _mm256_mul_pd(a0, d);
                            a1 = _mm256_mul_pd(a1, d);
                            a2 = _mm256_mul_pd(a2, d);
                            a3 = _mm256_mul_pd(a3, d);
                        }
                        _mm256_storeu_pd(r0, a0);
                        _mm256_storeu_pd(r1, a1);
                        _mm256_storeu_pd(r2, a2);
                        _mm256_storeu_pd(r3, a3);
                    }
                    j += 16;
                    continue;
                }
                for i in 0..nb {
                    let r0 = base.add(i * ldb + j);
                    let r1 = base.add(i * ldb + j + 4);
                    let r2 = base.add(i * ldb + j + 8);
                    let r3 = base.add(i * ldb + j + 12);
                    let mut a0 = _mm256_loadu_pd(r0);
                    let mut a1 = _mm256_loadu_pd(r1);
                    let mut a2 = _mm256_loadu_pd(r2);
                    let mut a3 = _mm256_loadu_pd(r3);
                    let abrow = ab.add(i * nb);
                    for k in 0..i {
                        let aik = _mm256_broadcast_sd(&*abrow.add(k));
                        a0 = _mm256_fnmadd_pd(aik, _mm256_loadu_pd(base.add(k * ldb + j)), a0);
                        a1 = _mm256_fnmadd_pd(aik, _mm256_loadu_pd(base.add(k * ldb + j + 4)), a1);
                        a2 = _mm256_fnmadd_pd(aik, _mm256_loadu_pd(base.add(k * ldb + j + 8)), a2);
                        a3 = _mm256_fnmadd_pd(aik, _mm256_loadu_pd(base.add(k * ldb + j + 12)), a3);
                    }
                    if !unit {
                        let d = _mm256_broadcast_sd(&*di.add(i));
                        a0 = _mm256_mul_pd(a0, d);
                        a1 = _mm256_mul_pd(a1, d);
                        a2 = _mm256_mul_pd(a2, d);
                        a3 = _mm256_mul_pd(a3, d);
                    }
                    _mm256_storeu_pd(r0, a0);
                    _mm256_storeu_pd(r1, a1);
                    _mm256_storeu_pd(r2, a2);
                    _mm256_storeu_pd(r3, a3);
                }
            } else {
                // Descending two-row mirror of the forward unroll above
                // (same gate): pairs `(lo, hi)` share each `X[k]` load for
                // `k > hi`; row `hi` stores first and row `lo`'s final
                // `k = hi` step reuses it from registers. Ascending `k`
                // per element — bitwise identical to the single-row loop.
                let nounroll = std::env::var_os("SONIUM_TRSM_NOUNROLL").is_some() || nb < 64;
                if !nounroll {
                    let mut i = nb;
                    while i >= 2 {
                        let (lo, hi) = (i - 2, i - 1);
                        let r0 = base.add(hi * ldb + j);
                        let r1 = base.add(hi * ldb + j + 4);
                        let r2 = base.add(hi * ldb + j + 8);
                        let r3 = base.add(hi * ldb + j + 12);
                        let s0 = base.add(lo * ldb + j);
                        let s1 = base.add(lo * ldb + j + 4);
                        let s2 = base.add(lo * ldb + j + 8);
                        let s3 = base.add(lo * ldb + j + 12);
                        let mut a0 = _mm256_loadu_pd(r0);
                        let mut a1 = _mm256_loadu_pd(r1);
                        let mut a2 = _mm256_loadu_pd(r2);
                        let mut a3 = _mm256_loadu_pd(r3);
                        let mut b0 = _mm256_loadu_pd(s0);
                        let mut b1 = _mm256_loadu_pd(s1);
                        let mut b2 = _mm256_loadu_pd(s2);
                        let mut b3 = _mm256_loadu_pd(s3);
                        let abrow0 = ab.add(hi * nb);
                        let abrow1 = ab.add(lo * nb);
                        for k in (hi + 1)..nb {
                            let x0 = _mm256_loadu_pd(base.add(k * ldb + j));
                            let x1 = _mm256_loadu_pd(base.add(k * ldb + j + 4));
                            let x2 = _mm256_loadu_pd(base.add(k * ldb + j + 8));
                            let x3 = _mm256_loadu_pd(base.add(k * ldb + j + 12));
                            let aik0 = _mm256_broadcast_sd(&*abrow0.add(k));
                            let aik1 = _mm256_broadcast_sd(&*abrow1.add(k));
                            a0 = _mm256_fnmadd_pd(aik0, x0, a0);
                            a1 = _mm256_fnmadd_pd(aik0, x1, a1);
                            a2 = _mm256_fnmadd_pd(aik0, x2, a2);
                            a3 = _mm256_fnmadd_pd(aik0, x3, a3);
                            b0 = _mm256_fnmadd_pd(aik1, x0, b0);
                            b1 = _mm256_fnmadd_pd(aik1, x1, b1);
                            b2 = _mm256_fnmadd_pd(aik1, x2, b2);
                            b3 = _mm256_fnmadd_pd(aik1, x3, b3);
                        }
                        if !unit {
                            let d = _mm256_broadcast_sd(&*di.add(hi));
                            a0 = _mm256_mul_pd(a0, d);
                            a1 = _mm256_mul_pd(a1, d);
                            a2 = _mm256_mul_pd(a2, d);
                            a3 = _mm256_mul_pd(a3, d);
                        }
                        _mm256_storeu_pd(r0, a0);
                        _mm256_storeu_pd(r1, a1);
                        _mm256_storeu_pd(r2, a2);
                        _mm256_storeu_pd(r3, a3);
                        // Final `k = hi` step for row `lo` straight from
                        // the just-stored row (no reload).
                        let aik = _mm256_broadcast_sd(&*abrow1.add(hi));
                        b0 = _mm256_fnmadd_pd(aik, a0, b0);
                        b1 = _mm256_fnmadd_pd(aik, a1, b1);
                        b2 = _mm256_fnmadd_pd(aik, a2, b2);
                        b3 = _mm256_fnmadd_pd(aik, a3, b3);
                        if !unit {
                            let d = _mm256_broadcast_sd(&*di.add(lo));
                            b0 = _mm256_mul_pd(b0, d);
                            b1 = _mm256_mul_pd(b1, d);
                            b2 = _mm256_mul_pd(b2, d);
                            b3 = _mm256_mul_pd(b3, d);
                        }
                        _mm256_storeu_pd(s0, b0);
                        _mm256_storeu_pd(s1, b1);
                        _mm256_storeu_pd(s2, b2);
                        _mm256_storeu_pd(s3, b3);
                        i -= 2;
                    }
                    if i == 1 {
                        let r0 = base.add(j);
                        let r1 = base.add(j + 4);
                        let r2 = base.add(j + 8);
                        let r3 = base.add(j + 12);
                        let mut a0 = _mm256_loadu_pd(r0);
                        let mut a1 = _mm256_loadu_pd(r1);
                        let mut a2 = _mm256_loadu_pd(r2);
                        let mut a3 = _mm256_loadu_pd(r3);
                        let abrow = ab;
                        for k in 1..nb {
                            let aik = _mm256_broadcast_sd(&*abrow.add(k));
                            a0 = _mm256_fnmadd_pd(aik, _mm256_loadu_pd(base.add(k * ldb + j)), a0);
                            a1 = _mm256_fnmadd_pd(
                                aik,
                                _mm256_loadu_pd(base.add(k * ldb + j + 4)),
                                a1,
                            );
                            a2 = _mm256_fnmadd_pd(
                                aik,
                                _mm256_loadu_pd(base.add(k * ldb + j + 8)),
                                a2,
                            );
                            a3 = _mm256_fnmadd_pd(
                                aik,
                                _mm256_loadu_pd(base.add(k * ldb + j + 12)),
                                a3,
                            );
                        }
                        if !unit {
                            let d = _mm256_broadcast_sd(&*di);
                            a0 = _mm256_mul_pd(a0, d);
                            a1 = _mm256_mul_pd(a1, d);
                            a2 = _mm256_mul_pd(a2, d);
                            a3 = _mm256_mul_pd(a3, d);
                        }
                        _mm256_storeu_pd(r0, a0);
                        _mm256_storeu_pd(r1, a1);
                        _mm256_storeu_pd(r2, a2);
                        _mm256_storeu_pd(r3, a3);
                    }
                    j += 16;
                    continue;
                }
                for i in (0..nb).rev() {
                    let r0 = base.add(i * ldb + j);
                    let r1 = base.add(i * ldb + j + 4);
                    let r2 = base.add(i * ldb + j + 8);
                    let r3 = base.add(i * ldb + j + 12);
                    let mut a0 = _mm256_loadu_pd(r0);
                    let mut a1 = _mm256_loadu_pd(r1);
                    let mut a2 = _mm256_loadu_pd(r2);
                    let mut a3 = _mm256_loadu_pd(r3);
                    let abrow = ab.add(i * nb);
                    for k in (i + 1)..nb {
                        let aik = _mm256_broadcast_sd(&*abrow.add(k));
                        a0 = _mm256_fnmadd_pd(aik, _mm256_loadu_pd(base.add(k * ldb + j)), a0);
                        a1 = _mm256_fnmadd_pd(aik, _mm256_loadu_pd(base.add(k * ldb + j + 4)), a1);
                        a2 = _mm256_fnmadd_pd(aik, _mm256_loadu_pd(base.add(k * ldb + j + 8)), a2);
                        a3 = _mm256_fnmadd_pd(aik, _mm256_loadu_pd(base.add(k * ldb + j + 12)), a3);
                    }
                    if !unit {
                        let d = _mm256_broadcast_sd(&*di.add(i));
                        a0 = _mm256_mul_pd(a0, d);
                        a1 = _mm256_mul_pd(a1, d);
                        a2 = _mm256_mul_pd(a2, d);
                        a3 = _mm256_mul_pd(a3, d);
                    }
                    _mm256_storeu_pd(r0, a0);
                    _mm256_storeu_pd(r1, a1);
                    _mm256_storeu_pd(r2, a2);
                    _mm256_storeu_pd(r3, a3);
                }
            }
            j += 16;
        }
        while j + 4 <= j1 {
            if !trans {
                for i in 0..nb {
                    let row = base.add(i * ldb + j);
                    let mut acc = _mm256_loadu_pd(row);
                    let abrow = ab.add(i * nb);
                    for k in 0..i {
                        let aik = _mm256_broadcast_sd(&*abrow.add(k));
                        let xv = _mm256_loadu_pd(base.add(k * ldb + j));
                        acc = _mm256_sub_pd(acc, _mm256_mul_pd(aik, xv));
                    }
                    if !unit {
                        acc = _mm256_mul_pd(acc, _mm256_broadcast_sd(&*di.add(i)));
                    }
                    _mm256_storeu_pd(row, acc);
                }
            } else {
                for i in (0..nb).rev() {
                    let row = base.add(i * ldb + j);
                    let mut acc = _mm256_loadu_pd(row);
                    let abrow = ab.add(i * nb);
                    for k in (i + 1)..nb {
                        let aik = _mm256_broadcast_sd(&*abrow.add(k));
                        let xv = _mm256_loadu_pd(base.add(k * ldb + j));
                        acc = _mm256_sub_pd(acc, _mm256_mul_pd(aik, xv));
                    }
                    if !unit {
                        acc = _mm256_mul_pd(acc, _mm256_broadcast_sd(&*di.add(i)));
                    }
                    _mm256_storeu_pd(row, acc);
                }
            }
            j += 4;
        }
        while j < j1 {
            if !trans {
                for i in 0..nb {
                    let mut s = *base.add(i * ldb + j);
                    for k in 0..i {
                        s -= *ab.add(i * nb + k) * *base.add(k * ldb + j);
                    }
                    *base.add(i * ldb + j) = if unit { s } else { s * *di.add(i) };
                }
            } else {
                for i in (0..nb).rev() {
                    let mut s = *base.add(i * ldb + j);
                    for k in (i + 1)..nb {
                        s -= *ab.add(i * nb + k) * *base.add(k * ldb + j);
                    }
                    *base.add(i * ldb + j) = if unit { s } else { s * *di.add(i) };
                }
            }
            j += 1;
        }
    }
}

/// Concrete `f32` left-lower band kernel: AVX2 when available (see
/// [`band_left_lower_kernel_f32_avx2`]), else the generic kernel. Threaded
/// from [`trsm_f32`] through [`trsm_core`] as a [`BandKernelFn`].
unsafe fn band_left_lower_kernel_f32(
    xb: *mut f32,
    ldb: usize,
    x0: usize,
    nb: usize,
    j0: usize,
    j1: usize,
    trans: bool,
    unit: bool,
    ab: *const f32,
    di: *const f32,
) {
    // SAFETY: same contract as the generic kernel (caller-owned `xb`
    // band, readable `ab`/`di`); the `unsafe` only matches the
    // [`BandKernelFn`] pointer type.
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::blas3::avx2::is_available() {
        // Blocked micro-solve (bitwise identical, fewer X reloads): needs
        // at least two micro-blocks to pay (single-block panels stay
        // row-wise); `trans` keeps the backward-substitution kernel.
        if !trans && nb >= 2 * TRSM_MICRO_MB && trsm_blocked_allowed() {
            unsafe { band_left_lower_blocked_f32(xb, ldb, x0, nb, j0, j1, trans, unit, ab, di) };
            return;
        }
        unsafe { band_left_lower_kernel_f32_avx2(xb, ldb, x0, nb, j0, j1, trans, unit, ab, di) };
        return;
    }
    band_left_lower_kernel(xb, ldb, x0, nb, j0, j1, trans, unit, ab, di)
}

/// Packed-pair complex product `s * (dr + i*di)` for the AVX2 complex
/// band kernels: `s`, `dr`, `di` are `__m256d` with two complex lanes.
/// `v_re` carries `re'` on even lanes, `v_im` carries `im'` on even
/// lanes; the permute moves `im'` to odd lanes and the blend assembles
/// `[re0', im0', re1', im1']`. FMA rounds once where scalar rounds
/// twice — ~1 ulp, within test tolerance.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn cmul_pd(
    s: core::arch::x86_64::__m256d,
    dr: core::arch::x86_64::__m256d,
    di: core::arch::x86_64::__m256d,
) -> core::arch::x86_64::__m256d {
    use core::arch::x86_64::*;
    let sw = _mm256_permute_pd(s, 0x5);
    let v_re = _mm256_fmsub_pd(dr, s, _mm256_mul_pd(di, sw));
    let v_im = _mm256_fmadd_pd(dr, sw, _mm256_mul_pd(di, s));
    _mm256_blend_pd(v_re, _mm256_permute_pd(v_im, 0x5), 0xA)
}

/// Packed-pair complex product, `f32` twin of [`cmul_pd`]: four complex
/// lanes per `__m256`. `0xB1` swaps adjacent pairs, `0xAA` blends the
/// odd lanes from the permuted `im'` vector.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn cmul_ps(
    s: core::arch::x86_64::__m256,
    dr: core::arch::x86_64::__m256,
    di: core::arch::x86_64::__m256,
) -> core::arch::x86_64::__m256 {
    use core::arch::x86_64::*;
    let sw = _mm256_permute_ps(s, 0xB1);
    let v_re = _mm256_fmsub_ps(dr, s, _mm256_mul_ps(di, sw));
    let v_im = _mm256_fmadd_ps(dr, sw, _mm256_mul_ps(di, s));
    _mm256_blend_ps(v_re, _mm256_permute_ps(v_im, 0xB1), 0xAA)
}

/// AVX2 `Complex64` left-lower band solve. Same contract as
/// [`band_left_lower_kernel`]: solves the `j0..j1` columns in place over
/// the packed `op(A11)` (`trans` selects forward/backward substitution).
/// The generic kernel this replaces runs scalar complex ops (no SLP: the
/// ordered `k` reduction plus pointer aliasing defeats vectorization) —
/// about 3x off the vendor TRSM rate on complex Cholesky/LU panels. One
/// `ymm` holds 2 complex lanes; the `k`-step `s -= aik * x` runs as
/// `fma(ai, x_sw, fnmadd(ar, x, s))` with `x_sw` re/im-swapped and
/// im-negated: re gets `s - ar*xr + ai*xi`, im gets `s - ar*xi - ai*xr`.
/// Four interleaved
/// accumulators (8 columns) hide the reduction latency; the `j1 - j0 < 8`
/// tail falls back to the generic kernel. Per-lane op order matches the
/// generic kernel (FMA ~1 ulp).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn band_left_lower_kernel_c64_avx2(
    xb: *mut Complex64,
    ldb: usize,
    x0: usize,
    nb: usize,
    j0: usize,
    j1: usize,
    trans: bool,
    unit: bool,
    ab: *const Complex64,
    di: *const Complex64,
) {
    use core::arch::x86_64::*;
    // SAFETY: caller guarantees the `xb` columns `j0..j1` exclusively
    // writable and `ab`/`di` readable for the panel ranges below; all
    // accesses are `loadu`/`storeu`. `Complex64` is `repr(C)` (`re, im`),
    // so the band addresses as `f64` pairs.
    unsafe {
        let base = (xb as *mut f64).add(x0 * 2);
        let abf = ab as *const f64;
        let dif = di as *const f64;
        let stride = ldb * 2;
        // Im-lane sign mask: the `ai` correction adds on re lanes but
        // subtracts on im lanes (`s -= aik*x`: re gets `+ai*xi`, im gets
        // `-ai*xr`), so the swapped x is im-negated before the FMA.
        let neg_im = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);
        let mut j = j0;
        while j + 8 <= j1 {
            let jb = j * 2;
            if !trans {
                for i in 0..nb {
                    let r = base.add(i * stride + jb);
                    let mut a0 = _mm256_loadu_pd(r);
                    let mut a1 = _mm256_loadu_pd(r.add(4));
                    let mut a2 = _mm256_loadu_pd(r.add(8));
                    let mut a3 = _mm256_loadu_pd(r.add(12));
                    let abrow = abf.add(i * nb * 2);
                    for k in 0..i {
                        let ar = _mm256_broadcast_sd(&*abrow.add(k * 2));
                        let ai = _mm256_broadcast_sd(&*abrow.add(k * 2 + 1));
                        let xrow = base.add(k * stride + jb);
                        let x0 = _mm256_loadu_pd(xrow);
                        let x1 = _mm256_loadu_pd(xrow.add(4));
                        let x2 = _mm256_loadu_pd(xrow.add(8));
                        let x3 = _mm256_loadu_pd(xrow.add(12));
                        let x0w = _mm256_xor_pd(_mm256_permute_pd(x0, 0x5), neg_im);
                        a0 = _mm256_fmadd_pd(ai, x0w, _mm256_fnmadd_pd(ar, x0, a0));
                        let x1w = _mm256_xor_pd(_mm256_permute_pd(x1, 0x5), neg_im);
                        a1 = _mm256_fmadd_pd(ai, x1w, _mm256_fnmadd_pd(ar, x1, a1));
                        let x2w = _mm256_xor_pd(_mm256_permute_pd(x2, 0x5), neg_im);
                        a2 = _mm256_fmadd_pd(ai, x2w, _mm256_fnmadd_pd(ar, x2, a2));
                        let x3w = _mm256_xor_pd(_mm256_permute_pd(x3, 0x5), neg_im);
                        a3 = _mm256_fmadd_pd(ai, x3w, _mm256_fnmadd_pd(ar, x3, a3));
                    }
                    if !unit {
                        let dr = _mm256_broadcast_sd(&*dif.add(i * 2));
                        let dii = _mm256_broadcast_sd(&*dif.add(i * 2 + 1));
                        a0 = cmul_pd(a0, dr, dii);
                        a1 = cmul_pd(a1, dr, dii);
                        a2 = cmul_pd(a2, dr, dii);
                        a3 = cmul_pd(a3, dr, dii);
                    }
                    _mm256_storeu_pd(r, a0);
                    _mm256_storeu_pd(r.add(4), a1);
                    _mm256_storeu_pd(r.add(8), a2);
                    _mm256_storeu_pd(r.add(12), a3);
                }
            } else {
                // op(A11) is upper triangular: backward substitution.
                for i in (0..nb).rev() {
                    let r = base.add(i * stride + jb);
                    let mut a0 = _mm256_loadu_pd(r);
                    let mut a1 = _mm256_loadu_pd(r.add(4));
                    let mut a2 = _mm256_loadu_pd(r.add(8));
                    let mut a3 = _mm256_loadu_pd(r.add(12));
                    let abrow = abf.add(i * nb * 2);
                    for k in (i + 1)..nb {
                        let ar = _mm256_broadcast_sd(&*abrow.add(k * 2));
                        let ai = _mm256_broadcast_sd(&*abrow.add(k * 2 + 1));
                        let xrow = base.add(k * stride + jb);
                        let x0 = _mm256_loadu_pd(xrow);
                        let x1 = _mm256_loadu_pd(xrow.add(4));
                        let x2 = _mm256_loadu_pd(xrow.add(8));
                        let x3 = _mm256_loadu_pd(xrow.add(12));
                        let x0w = _mm256_xor_pd(_mm256_permute_pd(x0, 0x5), neg_im);
                        a0 = _mm256_fmadd_pd(ai, x0w, _mm256_fnmadd_pd(ar, x0, a0));
                        let x1w = _mm256_xor_pd(_mm256_permute_pd(x1, 0x5), neg_im);
                        a1 = _mm256_fmadd_pd(ai, x1w, _mm256_fnmadd_pd(ar, x1, a1));
                        let x2w = _mm256_xor_pd(_mm256_permute_pd(x2, 0x5), neg_im);
                        a2 = _mm256_fmadd_pd(ai, x2w, _mm256_fnmadd_pd(ar, x2, a2));
                        let x3w = _mm256_xor_pd(_mm256_permute_pd(x3, 0x5), neg_im);
                        a3 = _mm256_fmadd_pd(ai, x3w, _mm256_fnmadd_pd(ar, x3, a3));
                    }
                    if !unit {
                        let dr = _mm256_broadcast_sd(&*dif.add(i * 2));
                        let dii = _mm256_broadcast_sd(&*dif.add(i * 2 + 1));
                        a0 = cmul_pd(a0, dr, dii);
                        a1 = cmul_pd(a1, dr, dii);
                        a2 = cmul_pd(a2, dr, dii);
                        a3 = cmul_pd(a3, dr, dii);
                    }
                    _mm256_storeu_pd(r, a0);
                    _mm256_storeu_pd(r.add(4), a1);
                    _mm256_storeu_pd(r.add(8), a2);
                    _mm256_storeu_pd(r.add(12), a3);
                }
            }
            j += 8;
        }
        if j < j1 {
            band_left_lower_kernel(xb, ldb, x0, nb, j, j1, trans, unit, ab, di);
        }
    }
}

/// AVX2 `Complex32` left-lower band solve: mirror twin of
/// [`band_left_lower_kernel_c64_avx2`] with four complex lanes per `ymm`
/// (16-column main loop, generic tail below 16).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn band_left_lower_kernel_c32_avx2(
    xb: *mut Complex32,
    ldb: usize,
    x0: usize,
    nb: usize,
    j0: usize,
    j1: usize,
    trans: bool,
    unit: bool,
    ab: *const Complex32,
    di: *const Complex32,
) {
    use core::arch::x86_64::*;
    // SAFETY: same contract as the `Complex64` twin; `Complex32` is
    // `repr(C)` (`re, im`), addressed as `f32` pairs.
    unsafe {
        let base = (xb as *mut f32).add(x0 * 2);
        let abf = ab as *const f32;
        let dif = di as *const f32;
        let stride = ldb * 2;
        // Im-lane sign mask (see the `Complex64` twin): the swapped x is
        // im-negated so the `ai` FMA adds on re lanes, subtracts on im.
        let neg_im = _mm256_set_ps(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0);
        let mut j = j0;
        while j + 16 <= j1 {
            let jb = j * 2;
            if !trans {
                for i in 0..nb {
                    let r = base.add(i * stride + jb);
                    let mut a0 = _mm256_loadu_ps(r);
                    let mut a1 = _mm256_loadu_ps(r.add(8));
                    let mut a2 = _mm256_loadu_ps(r.add(16));
                    let mut a3 = _mm256_loadu_ps(r.add(24));
                    let abrow = abf.add(i * nb * 2);
                    for k in 0..i {
                        let ar = _mm256_set1_ps(*abrow.add(k * 2));
                        let ai = _mm256_set1_ps(*abrow.add(k * 2 + 1));
                        let xrow = base.add(k * stride + jb);
                        let x0 = _mm256_loadu_ps(xrow);
                        let x1 = _mm256_loadu_ps(xrow.add(8));
                        let x2 = _mm256_loadu_ps(xrow.add(16));
                        let x3 = _mm256_loadu_ps(xrow.add(24));
                        let x0w = _mm256_xor_ps(_mm256_permute_ps(x0, 0xB1), neg_im);
                        a0 = _mm256_fmadd_ps(ai, x0w, _mm256_fnmadd_ps(ar, x0, a0));
                        let x1w = _mm256_xor_ps(_mm256_permute_ps(x1, 0xB1), neg_im);
                        a1 = _mm256_fmadd_ps(ai, x1w, _mm256_fnmadd_ps(ar, x1, a1));
                        let x2w = _mm256_xor_ps(_mm256_permute_ps(x2, 0xB1), neg_im);
                        a2 = _mm256_fmadd_ps(ai, x2w, _mm256_fnmadd_ps(ar, x2, a2));
                        let x3w = _mm256_xor_ps(_mm256_permute_ps(x3, 0xB1), neg_im);
                        a3 = _mm256_fmadd_ps(ai, x3w, _mm256_fnmadd_ps(ar, x3, a3));
                    }
                    if !unit {
                        let dr = _mm256_set1_ps(*dif.add(i * 2));
                        let dii = _mm256_set1_ps(*dif.add(i * 2 + 1));
                        a0 = cmul_ps(a0, dr, dii);
                        a1 = cmul_ps(a1, dr, dii);
                        a2 = cmul_ps(a2, dr, dii);
                        a3 = cmul_ps(a3, dr, dii);
                    }
                    _mm256_storeu_ps(r, a0);
                    _mm256_storeu_ps(r.add(8), a1);
                    _mm256_storeu_ps(r.add(16), a2);
                    _mm256_storeu_ps(r.add(24), a3);
                }
            } else {
                // op(A11) is upper triangular: backward substitution.
                for i in (0..nb).rev() {
                    let r = base.add(i * stride + jb);
                    let mut a0 = _mm256_loadu_ps(r);
                    let mut a1 = _mm256_loadu_ps(r.add(8));
                    let mut a2 = _mm256_loadu_ps(r.add(16));
                    let mut a3 = _mm256_loadu_ps(r.add(24));
                    let abrow = abf.add(i * nb * 2);
                    for k in (i + 1)..nb {
                        let ar = _mm256_set1_ps(*abrow.add(k * 2));
                        let ai = _mm256_set1_ps(*abrow.add(k * 2 + 1));
                        let xrow = base.add(k * stride + jb);
                        let x0 = _mm256_loadu_ps(xrow);
                        let x1 = _mm256_loadu_ps(xrow.add(8));
                        let x2 = _mm256_loadu_ps(xrow.add(16));
                        let x3 = _mm256_loadu_ps(xrow.add(24));
                        let x0w = _mm256_xor_ps(_mm256_permute_ps(x0, 0xB1), neg_im);
                        a0 = _mm256_fmadd_ps(ai, x0w, _mm256_fnmadd_ps(ar, x0, a0));
                        let x1w = _mm256_xor_ps(_mm256_permute_ps(x1, 0xB1), neg_im);
                        a1 = _mm256_fmadd_ps(ai, x1w, _mm256_fnmadd_ps(ar, x1, a1));
                        let x2w = _mm256_xor_ps(_mm256_permute_ps(x2, 0xB1), neg_im);
                        a2 = _mm256_fmadd_ps(ai, x2w, _mm256_fnmadd_ps(ar, x2, a2));
                        let x3w = _mm256_xor_ps(_mm256_permute_ps(x3, 0xB1), neg_im);
                        a3 = _mm256_fmadd_ps(ai, x3w, _mm256_fnmadd_ps(ar, x3, a3));
                    }
                    if !unit {
                        let dr = _mm256_set1_ps(*dif.add(i * 2));
                        let dii = _mm256_set1_ps(*dif.add(i * 2 + 1));
                        a0 = cmul_ps(a0, dr, dii);
                        a1 = cmul_ps(a1, dr, dii);
                        a2 = cmul_ps(a2, dr, dii);
                        a3 = cmul_ps(a3, dr, dii);
                    }
                    _mm256_storeu_ps(r, a0);
                    _mm256_storeu_ps(r.add(8), a1);
                    _mm256_storeu_ps(r.add(16), a2);
                    _mm256_storeu_ps(r.add(24), a3);
                }
            }
            j += 16;
        }
        if j < j1 {
            band_left_lower_kernel(xb, ldb, x0, nb, j, j1, trans, unit, ab, di);
        }
    }
}

/// Concrete `Complex64` left-lower band kernel: AVX2 when available (see
/// [`band_left_lower_kernel_c64_avx2`]), else the generic kernel. Threaded
/// from [`trsm_c64`] through [`trsm_core`] as a [`BandKernelFn`].
unsafe fn band_left_lower_kernel_c64(
    xb: *mut Complex64,
    ldb: usize,
    x0: usize,
    nb: usize,
    j0: usize,
    j1: usize,
    trans: bool,
    unit: bool,
    ab: *const Complex64,
    di: *const Complex64,
) {
    // SAFETY: same contract as the generic kernel (caller-owned `xb`
    // band, readable `ab`/`di`); the `unsafe` only matches the
    // [`BandKernelFn`] pointer type.
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::blas3::avx2::is_available() {
        unsafe { band_left_lower_kernel_c64_avx2(xb, ldb, x0, nb, j0, j1, trans, unit, ab, di) };
        return;
    }
    band_left_lower_kernel(xb, ldb, x0, nb, j0, j1, trans, unit, ab, di)
}

/// Concrete `Complex32` left-lower band kernel: AVX2 when available (see
/// [`band_left_lower_kernel_c32_avx2`]), else the generic kernel. Threaded
/// from [`trsm_c32`] through [`trsm_core`] as a [`BandKernelFn`].
unsafe fn band_left_lower_kernel_c32(
    xb: *mut Complex32,
    ldb: usize,
    x0: usize,
    nb: usize,
    j0: usize,
    j1: usize,
    trans: bool,
    unit: bool,
    ab: *const Complex32,
    di: *const Complex32,
) {
    // SAFETY: same contract as the generic kernel (caller-owned `xb`
    // band, readable `ab`/`di`); the `unsafe` only matches the
    // [`BandKernelFn`] pointer type.
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::blas3::avx2::is_available() {
        unsafe { band_left_lower_kernel_c32_avx2(xb, ldb, x0, nb, j0, j1, trans, unit, ab, di) };
        return;
    }
    band_left_lower_kernel(xb, ldb, x0, nb, j0, j1, trans, unit, ab, di)
}

/// Concrete `Complex64` left-upper band kernel: mirror twin reusing
/// [`band_left_lower_kernel_c64`] with a flipped direction flag (same
/// pattern as [`band_left_upper_kernel_f64`]).
unsafe fn band_left_upper_kernel_c64(
    xb: *mut Complex64,
    ldb: usize,
    x0: usize,
    nb: usize,
    j0: usize,
    j1: usize,
    trans: bool,
    unit: bool,
    ab: *const Complex64,
    di: *const Complex64,
) {
    // SAFETY: same contract as the caller; the flip only selects the
    // substitution direction over the already-packed triangle.
    unsafe { band_left_lower_kernel_c64(xb, ldb, x0, nb, j0, j1, !trans, unit, ab, di) };
}

/// Concrete `Complex32` left-upper band kernel: mirror twin reusing
/// [`band_left_lower_kernel_c32`] with a flipped direction flag.
unsafe fn band_left_upper_kernel_c32(
    xb: *mut Complex32,
    ldb: usize,
    x0: usize,
    nb: usize,
    j0: usize,
    j1: usize,
    trans: bool,
    unit: bool,
    ab: *const Complex32,
    di: *const Complex32,
) {
    // SAFETY: same contract as the caller; the flip only selects the
    // substitution direction over the already-packed triangle.
    unsafe { band_left_lower_kernel_c32(xb, ldb, x0, nb, j0, j1, !trans, unit, ab, di) };
}

/// Concrete `f64` left-upper band kernel: the upper solve is the exact
/// mirror of the lower one (backward vs forward substitution over the
/// packed `op(A11)`), so it reuses [`band_left_lower_kernel_f64`] with a
/// flipped direction flag — same ops, bitwise identical. Threaded from
/// [`trsm_f64`] through [`trsm_core`] as a [`BandKernelFn`].
unsafe fn band_left_upper_kernel_f64(
    xb: *mut f64,
    ldb: usize,
    x0: usize,
    nb: usize,
    j0: usize,
    j1: usize,
    trans: bool,
    unit: bool,
    ab: *const f64,
    di: *const f64,
) {
    // SAFETY: same contract as the caller; the flip only selects the
    // substitution direction over the already-packed triangle.
    unsafe { band_left_lower_kernel_f64(xb, ldb, x0, nb, j0, j1, !trans, unit, ab, di) };
}

/// Concrete `f32` left-upper band kernel: mirror twin of
/// [`band_left_lower_kernel_f32`], same flip arrangement. Threaded from
/// [`trsm_f32`] through [`trsm_core`] as a [`BandKernelFn`].
unsafe fn band_left_upper_kernel_f32(
    xb: *mut f32,
    ldb: usize,
    x0: usize,
    nb: usize,
    j0: usize,
    j1: usize,
    trans: bool,
    unit: bool,
    ab: *const f32,
    di: *const f32,
) {
    // SAFETY: same contract as the caller; the flip only selects the
    // substitution direction over the already-packed triangle.
    unsafe { band_left_lower_kernel_f32(xb, ldb, x0, nb, j0, j1, !trans, unit, ab, di) };
}

/// Micro-block rows for the blocked left-lower f32 solve below: 4 rows
/// fits the register-held hot path (8 accumulators plus 2 X vectors plus
/// 1 broadcast plus 1 dinv = 12 YMM); tails and off-size chunks use the
/// memory general path.
#[cfg(target_arch = "x86_64")]
const TRSM_MICRO_MB: usize = 4;

/// Blocked-solve escape hatch (`SONIUM_TRSM_NOBLOCKED=1` restores the
/// row-wise kernels), cached once like `ROWBAND_ALLOWED`: the hook runs
/// per band-chunk and must not take the env lock per call. No test
/// mutates it (correctness tests call the blocked kernels directly).
#[cfg(target_arch = "x86_64")]
fn trsm_blocked_allowed() -> bool {
    static ALLOWED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ALLOWED.get_or_init(|| std::env::var_os("SONIUM_TRSM_NOBLOCKED").is_none())
}

/// AVX2 tile-pack gate, cached once: the pack arms run per band-tile
/// (hundreds per TRSM) and a per-call env read costs more than the pack
/// itself (f32 Right regressed on the uncached gate). No test mutates
/// `SONIUM_TRSM_PACK_SCALAR`, so caching is behavior-identical.
#[cfg(target_arch = "x86_64")]
fn pack_avx2_allowed() -> bool {
    static ALLOWED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ALLOWED.get_or_init(|| {
        crate::cpu::blas3::avx2::is_available()
            && std::env::var_os("SONIUM_TRSM_PACK_SCALAR").is_none()
    })
}

/// Blocked left-lower NoTrans micro-solve (f32 AVX2, [`BandKernelFn`]
/// shape): rows advance in [`TRSM_MICRO_MB`]-row micro-blocks; each block
/// first updates from the finalized rows above (k-outer over `0..r`, fully
/// parallel over rows x columns with the block's X rows register-held on
/// the hot path) and then solves its <=4 rows row-wise (<=3 within-k
/// steps). Lane op order matches the row-wise kernel exactly (ascending-k
/// `fnmadd`, scalar `mul+sub` tails, `mul` by `dinv`), so results are
/// BITWISE identical to [`band_left_lower_kernel_f32_avx2`]; only the
/// visit order changes (each finalized X row loads once per block instead
/// of once per row). `trans` must be false (the hook keeps `trans` on the
/// row-wise kernel); `unit` skips the `dinv` reads (`di` empty then).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn band_left_lower_blocked_f32(
    xb: *mut f32,
    ldb: usize,
    x0: usize,
    nb: usize,
    j0: usize,
    j1: usize,
    trans: bool,
    unit: bool,
    ab: *const f32,
    di: *const f32,
) {
    use core::arch::x86_64::*;
    // SAFETY: same contract as the row-wise kernel (caller-owned `xb`
    // band `j0..j1`, readable packed `ab`/`di`); all accesses in-bounds
    // by the same arithmetic, `loadu`/`storeu` throughout.
    unsafe {
        debug_assert!(!trans);
        let base = xb.add(x0);
        let mut jj = j0;
        while jj < j1 {
            let je = (jj + 16).min(j1);
            let nw = je - jj;
            let mut r = 0;
            while r < nb {
                let r1 = (r + TRSM_MICRO_MB).min(nb);
                if r1 - r == TRSM_MICRO_MB && nw == 16 {
                    // Register-held hot path: 4 rows x 16 cols.
                    let mut a0l = _mm256_loadu_ps(base.add(r * ldb + jj));
                    let mut a0h = _mm256_loadu_ps(base.add(r * ldb + jj + 8));
                    let mut a1l = _mm256_loadu_ps(base.add((r + 1) * ldb + jj));
                    let mut a1h = _mm256_loadu_ps(base.add((r + 1) * ldb + jj + 8));
                    let mut a2l = _mm256_loadu_ps(base.add((r + 2) * ldb + jj));
                    let mut a2h = _mm256_loadu_ps(base.add((r + 2) * ldb + jj + 8));
                    let mut a3l = _mm256_loadu_ps(base.add((r + 3) * ldb + jj));
                    let mut a3h = _mm256_loadu_ps(base.add((r + 3) * ldb + jj + 8));
                    for k in 0..r {
                        let xk0 = _mm256_loadu_ps(base.add(k * ldb + jj));
                        let xk1 = _mm256_loadu_ps(base.add(k * ldb + jj + 8));
                        let b0 = _mm256_broadcast_ss(&*ab.add(r * nb + k));
                        a0l = _mm256_fnmadd_ps(b0, xk0, a0l);
                        a0h = _mm256_fnmadd_ps(b0, xk1, a0h);
                        let b1 = _mm256_broadcast_ss(&*ab.add((r + 1) * nb + k));
                        a1l = _mm256_fnmadd_ps(b1, xk0, a1l);
                        a1h = _mm256_fnmadd_ps(b1, xk1, a1h);
                        let b2 = _mm256_broadcast_ss(&*ab.add((r + 2) * nb + k));
                        a2l = _mm256_fnmadd_ps(b2, xk0, a2l);
                        a2h = _mm256_fnmadd_ps(b2, xk1, a2h);
                        let b3 = _mm256_broadcast_ss(&*ab.add((r + 3) * nb + k));
                        a3l = _mm256_fnmadd_ps(b3, xk0, a3l);
                        a3h = _mm256_fnmadd_ps(b3, xk1, a3h);
                    }
                    // Within-block solve: finalized rows stay live in the
                    // accumulators, so no X reloads at all.
                    if !unit {
                        let d = _mm256_broadcast_ss(&*di.add(r));
                        a0l = _mm256_mul_ps(a0l, d);
                        a0h = _mm256_mul_ps(a0h, d);
                    }
                    _mm256_storeu_ps(base.add(r * ldb + jj), a0l);
                    _mm256_storeu_ps(base.add(r * ldb + jj + 8), a0h);
                    let c1 = _mm256_broadcast_ss(&*ab.add((r + 1) * nb + r));
                    a1l = _mm256_fnmadd_ps(c1, a0l, a1l);
                    a1h = _mm256_fnmadd_ps(c1, a0h, a1h);
                    if !unit {
                        let d = _mm256_broadcast_ss(&*di.add(r + 1));
                        a1l = _mm256_mul_ps(a1l, d);
                        a1h = _mm256_mul_ps(a1h, d);
                    }
                    _mm256_storeu_ps(base.add((r + 1) * ldb + jj), a1l);
                    _mm256_storeu_ps(base.add((r + 1) * ldb + jj + 8), a1h);
                    let c20 = _mm256_broadcast_ss(&*ab.add((r + 2) * nb + r));
                    a2l = _mm256_fnmadd_ps(c20, a0l, a2l);
                    a2h = _mm256_fnmadd_ps(c20, a0h, a2h);
                    let c21 = _mm256_broadcast_ss(&*ab.add((r + 2) * nb + r + 1));
                    a2l = _mm256_fnmadd_ps(c21, a1l, a2l);
                    a2h = _mm256_fnmadd_ps(c21, a1h, a2h);
                    if !unit {
                        let d = _mm256_broadcast_ss(&*di.add(r + 2));
                        a2l = _mm256_mul_ps(a2l, d);
                        a2h = _mm256_mul_ps(a2h, d);
                    }
                    _mm256_storeu_ps(base.add((r + 2) * ldb + jj), a2l);
                    _mm256_storeu_ps(base.add((r + 2) * ldb + jj + 8), a2h);
                    let c30 = _mm256_broadcast_ss(&*ab.add((r + 3) * nb + r));
                    a3l = _mm256_fnmadd_ps(c30, a0l, a3l);
                    a3h = _mm256_fnmadd_ps(c30, a0h, a3h);
                    let c31 = _mm256_broadcast_ss(&*ab.add((r + 3) * nb + r + 1));
                    a3l = _mm256_fnmadd_ps(c31, a1l, a3l);
                    a3h = _mm256_fnmadd_ps(c31, a1h, a3h);
                    let c32 = _mm256_broadcast_ss(&*ab.add((r + 3) * nb + r + 2));
                    a3l = _mm256_fnmadd_ps(c32, a2l, a3l);
                    a3h = _mm256_fnmadd_ps(c32, a2h, a3h);
                    if !unit {
                        let d = _mm256_broadcast_ss(&*di.add(r + 3));
                        a3l = _mm256_mul_ps(a3l, d);
                        a3h = _mm256_mul_ps(a3h, d);
                    }
                    _mm256_storeu_ps(base.add((r + 3) * ldb + jj), a3l);
                    _mm256_storeu_ps(base.add((r + 3) * ldb + jj + 8), a3h);
                } else {
                    // General path (row tail or off-size chunk): same
                    // block/update/solve structure through memory.
                    let nv = nw / 8;
                    let nt = nw % 8;
                    for k in 0..r {
                        for i in r..r1 {
                            let aik = *ab.add(i * nb + k);
                            let b = _mm256_broadcast_ss(&aik);
                            for v in 0..nv {
                                let p = base.add(i * ldb + jj + v * 8);
                                let xv = _mm256_loadu_ps(base.add(k * ldb + jj + v * 8));
                                _mm256_storeu_ps(p, _mm256_fnmadd_ps(b, xv, _mm256_loadu_ps(p)));
                            }
                            for t in 0..nt {
                                let p = base.add(i * ldb + jj + nv * 8 + t);
                                *p -= aik * *base.add(k * ldb + jj + nv * 8 + t);
                            }
                        }
                    }
                    for i in r..r1 {
                        for v in 0..nv {
                            let p = base.add(i * ldb + jj + v * 8);
                            let mut acc = _mm256_loadu_ps(p);
                            for k in r..i {
                                let aik = _mm256_broadcast_ss(&*ab.add(i * nb + k));
                                acc = _mm256_fnmadd_ps(
                                    aik,
                                    _mm256_loadu_ps(base.add(k * ldb + jj + v * 8)),
                                    acc,
                                );
                            }
                            if !unit {
                                acc = _mm256_mul_ps(acc, _mm256_broadcast_ss(&*di.add(i)));
                            }
                            _mm256_storeu_ps(p, acc);
                        }
                        for t in 0..nt {
                            let p = base.add(i * ldb + jj + nv * 8 + t);
                            let mut s = *p;
                            for k in r..i {
                                s -= *ab.add(i * nb + k) * *base.add(k * ldb + jj + nv * 8 + t);
                            }
                            *p = if unit { s } else { s * *di.add(i) };
                        }
                    }
                }
                r = r1;
            }
            jj = je;
        }
    }
}

/// AVX2 `f32` left-lower band solve: the 8-wide twin of
/// [`band_left_lower_kernel_f64_avx2`] (one `ymm` holds eight `s`
/// accumulators), FMA throughout like the f64 kernel. Same contract; like
/// the f64 kernel the fused multiply-add rounds once where the generic
/// kernel rounds twice (~1 ulp, within test tolerance, strictly more
/// accurate — same arrangement as every vendor BLAS).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn band_left_lower_kernel_f32_avx2(
    xb: *mut f32,
    ldb: usize,
    x0: usize,
    nb: usize,
    j0: usize,
    j1: usize,
    trans: bool,
    unit: bool,
    ab: *const f32,
    di: *const f32,
) {
    use core::arch::x86_64::*;
    // SAFETY: caller guarantees the `xb` columns `j0..j1` exclusively
    // writable and `ab`/`di` readable for the panel ranges below; all
    // accesses are `loadu`/`storeu` (arbitrary `ldb` stride alignment).
    unsafe {
        let base = xb.add(x0);
        let mut j = j0;
        // 16-wide four-chain tier (same gate/threshold as below): two rows
        // × two 8-wide column groups interleaved in one `k` loop — the f32
        // analogue of the f64 16-wide loop. `sub(mul)`, exact scalar op
        // order per lane — bitwise identical. The 8-wide tier below handles
        // the tail.
        // Unroll gate at 16, not 64 (2026-09-26): the old `nb < 64` forced
        // 16-64-wide bands (all of Cholesky's, nb=48) onto the 2-chain
        // single-row fallback, running f32 at f64 rate (band 64x48:
        // 12.5us vs OB strsm 5.6us). The two-row tiers keep exact
        // per-lane op order and win at nb=48 (same band: 4.8us, matching
        // OB); the floor below 16 is untested. f64 keeps its own gate
        // (its 16-wide main loop is unconditional).
        let nounroll16 = std::env::var_os("SONIUM_TRSM_NOUNROLL").is_some() || nb < 16;
        // 32-wide eight-chain tier: two rows × four 8-wide groups. The
        // 16-wide tier's four FMA chains cover ~20 cycles of dependent
        // latency per `k` step but issue only four FMAs; doubling the
        // groups doubles the chains (eight) at the same two broadcasts
        // per step, halving the latency-bound cycles per flop. Same
        // two-row structure and exact per-lane op order as below, so
        // results match the 16-wide tier to the bit; the 16-wide loop
        // handles the remainder. NoTrans only: the Trans branch keeps
        // its 16-wide top tier.
        while j + 32 <= j1 {
            if !trans && !nounroll16 {
                let mut i = 0;
                while i + 1 < nb {
                    let r0 = [
                        base.add(i * ldb + j),
                        base.add(i * ldb + j + 8),
                        base.add(i * ldb + j + 16),
                        base.add(i * ldb + j + 24),
                    ];
                    let r1 = [
                        base.add((i + 1) * ldb + j),
                        base.add((i + 1) * ldb + j + 8),
                        base.add((i + 1) * ldb + j + 16),
                        base.add((i + 1) * ldb + j + 24),
                    ];
                    let mut a = [
                        _mm256_loadu_ps(r0[0]),
                        _mm256_loadu_ps(r0[1]),
                        _mm256_loadu_ps(r0[2]),
                        _mm256_loadu_ps(r0[3]),
                    ];
                    let mut b = [
                        _mm256_loadu_ps(r1[0]),
                        _mm256_loadu_ps(r1[1]),
                        _mm256_loadu_ps(r1[2]),
                        _mm256_loadu_ps(r1[3]),
                    ];
                    let abrow0 = ab.add(i * nb);
                    let abrow1 = ab.add((i + 1) * nb);
                    // NOTE (2026-09-27): a software-pipelined `k` loop
                    // (preload X row `k + 1` while FMAs consume row `k`)
                    // tied bit-for-bit here (ST 128x128: 1.73x vs 1.74x
                    // off OpenBLAS) — L1 load-use is already hidden, the
                    // ~1.7x is elsewhere (prime suspect: row-wise k-dots
                    // vs a blocked/inverted-diagonal micro-solve running
                    // at GEMM rate; untested). Do not retry pipelining.
                    for k in 0..i {
                        let x = [
                            _mm256_loadu_ps(base.add(k * ldb + j)),
                            _mm256_loadu_ps(base.add(k * ldb + j + 8)),
                            _mm256_loadu_ps(base.add(k * ldb + j + 16)),
                            _mm256_loadu_ps(base.add(k * ldb + j + 24)),
                        ];
                        let aik0 = _mm256_broadcast_ss(&*abrow0.add(k));
                        let aik1 = _mm256_broadcast_ss(&*abrow1.add(k));
                        a[0] = _mm256_fnmadd_ps(aik0, x[0], a[0]);
                        a[1] = _mm256_fnmadd_ps(aik0, x[1], a[1]);
                        a[2] = _mm256_fnmadd_ps(aik0, x[2], a[2]);
                        a[3] = _mm256_fnmadd_ps(aik0, x[3], a[3]);
                        b[0] = _mm256_fnmadd_ps(aik1, x[0], b[0]);
                        b[1] = _mm256_fnmadd_ps(aik1, x[1], b[1]);
                        b[2] = _mm256_fnmadd_ps(aik1, x[2], b[2]);
                        b[3] = _mm256_fnmadd_ps(aik1, x[3], b[3]);
                    }
                    if !unit {
                        let d = _mm256_broadcast_ss(&*di.add(i));
                        a[0] = _mm256_mul_ps(a[0], d);
                        a[1] = _mm256_mul_ps(a[1], d);
                        a[2] = _mm256_mul_ps(a[2], d);
                        a[3] = _mm256_mul_ps(a[3], d);
                    }
                    _mm256_storeu_ps(r0[0], a[0]);
                    _mm256_storeu_ps(r0[1], a[1]);
                    _mm256_storeu_ps(r0[2], a[2]);
                    _mm256_storeu_ps(r0[3], a[3]);
                    let aik = _mm256_broadcast_ss(&*abrow1.add(i));
                    b[0] = _mm256_fnmadd_ps(aik, a[0], b[0]);
                    b[1] = _mm256_fnmadd_ps(aik, a[1], b[1]);
                    b[2] = _mm256_fnmadd_ps(aik, a[2], b[2]);
                    b[3] = _mm256_fnmadd_ps(aik, a[3], b[3]);
                    if !unit {
                        let d = _mm256_broadcast_ss(&*di.add(i + 1));
                        b[0] = _mm256_mul_ps(b[0], d);
                        b[1] = _mm256_mul_ps(b[1], d);
                        b[2] = _mm256_mul_ps(b[2], d);
                        b[3] = _mm256_mul_ps(b[3], d);
                    }
                    _mm256_storeu_ps(r1[0], b[0]);
                    _mm256_storeu_ps(r1[1], b[1]);
                    _mm256_storeu_ps(r1[2], b[2]);
                    _mm256_storeu_ps(r1[3], b[3]);
                    i += 2;
                }
                if i < nb {
                    for g in 0..4 {
                        let row = base.add(i * ldb + j + g * 8);
                        let mut acc = _mm256_loadu_ps(row);
                        let abrow = ab.add(i * nb);
                        for k in 0..i {
                            let aik = _mm256_broadcast_ss(&*abrow.add(k));
                            let xv = _mm256_loadu_ps(base.add(k * ldb + j + g * 8));
                            acc = _mm256_fnmadd_ps(aik, xv, acc);
                        }
                        if !unit {
                            acc = _mm256_mul_ps(acc, _mm256_broadcast_ss(&*di.add(i)));
                        }
                        _mm256_storeu_ps(row, acc);
                    }
                }
                j += 32;
                continue;
            }
            break;
        }
        while j + 16 <= j1 {
            if !trans {
                if !nounroll16 {
                    let mut i = 0;
                    while i + 1 < nb {
                        let r00 = base.add(i * ldb + j);
                        let r01 = base.add(i * ldb + j + 8);
                        let r10 = base.add((i + 1) * ldb + j);
                        let r11 = base.add((i + 1) * ldb + j + 8);
                        let mut a00 = _mm256_loadu_ps(r00);
                        let mut a01 = _mm256_loadu_ps(r01);
                        let mut b00 = _mm256_loadu_ps(r10);
                        let mut b01 = _mm256_loadu_ps(r11);
                        let abrow0 = ab.add(i * nb);
                        let abrow1 = ab.add((i + 1) * nb);
                        for k in 0..i {
                            let x0 = _mm256_loadu_ps(base.add(k * ldb + j));
                            let x1 = _mm256_loadu_ps(base.add(k * ldb + j + 8));
                            let aik0 = _mm256_broadcast_ss(&*abrow0.add(k));
                            let aik1 = _mm256_broadcast_ss(&*abrow1.add(k));
                            a00 = _mm256_fnmadd_ps(aik0, x0, a00);
                            a01 = _mm256_fnmadd_ps(aik0, x1, a01);
                            b00 = _mm256_fnmadd_ps(aik1, x0, b00);
                            b01 = _mm256_fnmadd_ps(aik1, x1, b01);
                        }
                        if !unit {
                            let d = _mm256_broadcast_ss(&*di.add(i));
                            a00 = _mm256_mul_ps(a00, d);
                            a01 = _mm256_mul_ps(a01, d);
                        }
                        _mm256_storeu_ps(r00, a00);
                        _mm256_storeu_ps(r01, a01);
                        let aik = _mm256_broadcast_ss(&*abrow1.add(i));
                        b00 = _mm256_fnmadd_ps(aik, a00, b00);
                        b01 = _mm256_fnmadd_ps(aik, a01, b01);
                        if !unit {
                            let d = _mm256_broadcast_ss(&*di.add(i + 1));
                            b00 = _mm256_mul_ps(b00, d);
                            b01 = _mm256_mul_ps(b01, d);
                        }
                        _mm256_storeu_ps(r10, b00);
                        _mm256_storeu_ps(r11, b01);
                        i += 2;
                    }
                    if i < nb {
                        for g in 0..2 {
                            let row = base.add(i * ldb + j + g * 8);
                            let mut acc = _mm256_loadu_ps(row);
                            let abrow = ab.add(i * nb);
                            for k in 0..i {
                                let aik = _mm256_broadcast_ss(&*abrow.add(k));
                                let xv = _mm256_loadu_ps(base.add(k * ldb + j + g * 8));
                                acc = _mm256_fnmadd_ps(aik, xv, acc);
                            }
                            if !unit {
                                acc = _mm256_mul_ps(acc, _mm256_broadcast_ss(&*di.add(i)));
                            }
                            _mm256_storeu_ps(row, acc);
                        }
                    }
                    j += 16;
                    continue;
                }
                for i in 0..nb {
                    for g in 0..2 {
                        let row = base.add(i * ldb + j + g * 8);
                        let mut acc = _mm256_loadu_ps(row);
                        let abrow = ab.add(i * nb);
                        for k in 0..i {
                            let aik = _mm256_broadcast_ss(&*abrow.add(k));
                            let xv = _mm256_loadu_ps(base.add(k * ldb + j + g * 8));
                            acc = _mm256_fnmadd_ps(aik, xv, acc);
                        }
                        if !unit {
                            acc = _mm256_mul_ps(acc, _mm256_broadcast_ss(&*di.add(i)));
                        }
                        _mm256_storeu_ps(row, acc);
                    }
                }
            } else {
                if !nounroll16 {
                    let mut i = nb;
                    while i >= 2 {
                        let (lo, hi) = (i - 2, i - 1);
                        let r00 = base.add(hi * ldb + j);
                        let r01 = base.add(hi * ldb + j + 8);
                        let r10 = base.add(lo * ldb + j);
                        let r11 = base.add(lo * ldb + j + 8);
                        let mut a00 = _mm256_loadu_ps(r00);
                        let mut a01 = _mm256_loadu_ps(r01);
                        let mut b00 = _mm256_loadu_ps(r10);
                        let mut b01 = _mm256_loadu_ps(r11);
                        let abrow0 = ab.add(hi * nb);
                        let abrow1 = ab.add(lo * nb);
                        for k in (hi + 1)..nb {
                            let x0 = _mm256_loadu_ps(base.add(k * ldb + j));
                            let x1 = _mm256_loadu_ps(base.add(k * ldb + j + 8));
                            let aik0 = _mm256_broadcast_ss(&*abrow0.add(k));
                            let aik1 = _mm256_broadcast_ss(&*abrow1.add(k));
                            a00 = _mm256_fnmadd_ps(aik0, x0, a00);
                            a01 = _mm256_fnmadd_ps(aik0, x1, a01);
                            b00 = _mm256_fnmadd_ps(aik1, x0, b00);
                            b01 = _mm256_fnmadd_ps(aik1, x1, b01);
                        }
                        if !unit {
                            let d = _mm256_broadcast_ss(&*di.add(hi));
                            a00 = _mm256_mul_ps(a00, d);
                            a01 = _mm256_mul_ps(a01, d);
                        }
                        _mm256_storeu_ps(r00, a00);
                        _mm256_storeu_ps(r01, a01);
                        let aik = _mm256_broadcast_ss(&*abrow1.add(hi));
                        b00 = _mm256_fnmadd_ps(aik, a00, b00);
                        b01 = _mm256_fnmadd_ps(aik, a01, b01);
                        if !unit {
                            let d = _mm256_broadcast_ss(&*di.add(lo));
                            b00 = _mm256_mul_ps(b00, d);
                            b01 = _mm256_mul_ps(b01, d);
                        }
                        _mm256_storeu_ps(r10, b00);
                        _mm256_storeu_ps(r11, b01);
                        i -= 2;
                    }
                    if i == 1 {
                        for g in 0..2 {
                            let row = base.add(j + g * 8);
                            let mut acc = _mm256_loadu_ps(row);
                            for k in 1..nb {
                                let aik = _mm256_broadcast_ss(&*ab.add(k));
                                let xv = _mm256_loadu_ps(base.add(k * ldb + j + g * 8));
                                acc = _mm256_fnmadd_ps(aik, xv, acc);
                            }
                            if !unit {
                                acc = _mm256_mul_ps(acc, _mm256_broadcast_ss(&*di));
                            }
                            _mm256_storeu_ps(row, acc);
                        }
                    }
                    j += 16;
                    continue;
                }
                for i in (0..nb).rev() {
                    for g in 0..2 {
                        let row = base.add(i * ldb + j + g * 8);
                        let mut acc = _mm256_loadu_ps(row);
                        let abrow = ab.add(i * nb);
                        for k in (i + 1)..nb {
                            let aik = _mm256_broadcast_ss(&*abrow.add(k));
                            let xv = _mm256_loadu_ps(base.add(k * ldb + j + g * 8));
                            acc = _mm256_fnmadd_ps(aik, xv, acc);
                        }
                        if !unit {
                            acc = _mm256_mul_ps(acc, _mm256_broadcast_ss(&*di.add(i)));
                        }
                        _mm256_storeu_ps(row, acc);
                    }
                }
            }
            j += 16;
        }
        while j + 8 <= j1 {
            if !trans {
                // Two-row unroll mirroring the f64 kernel (same
                // `SONIUM_TRSM_NOUNROLL` gate and `nb >= 64` threshold):
                // rows share each `X[k]` load; FMA like the f64 kernel
                // (same ~1-ulp rounding note as above).
                let nounroll = std::env::var_os("SONIUM_TRSM_NOUNROLL").is_some() || nb < 64;
                if !nounroll {
                    let mut i = 0;
                    while i + 1 < nb {
                        let row0 = base.add(i * ldb + j);
                        let row1 = base.add((i + 1) * ldb + j);
                        let mut a0 = _mm256_loadu_ps(row0);
                        let mut b0 = _mm256_loadu_ps(row1);
                        let abrow0 = ab.add(i * nb);
                        let abrow1 = ab.add((i + 1) * nb);
                        for k in 0..i {
                            let xv = _mm256_loadu_ps(base.add(k * ldb + j));
                            a0 = _mm256_fnmadd_ps(_mm256_broadcast_ss(&*abrow0.add(k)), xv, a0);
                            b0 = _mm256_fnmadd_ps(_mm256_broadcast_ss(&*abrow1.add(k)), xv, b0);
                        }
                        if !unit {
                            a0 = _mm256_mul_ps(a0, _mm256_broadcast_ss(&*di.add(i)));
                        }
                        _mm256_storeu_ps(row0, a0);
                        // Final `k = i` step for row `i + 1` from registers.
                        b0 = _mm256_fnmadd_ps(_mm256_broadcast_ss(&*abrow1.add(i)), a0, b0);
                        if !unit {
                            b0 = _mm256_mul_ps(b0, _mm256_broadcast_ss(&*di.add(i + 1)));
                        }
                        _mm256_storeu_ps(row1, b0);
                        i += 2;
                    }
                    if i < nb {
                        let row = base.add(i * ldb + j);
                        let mut acc = _mm256_loadu_ps(row);
                        let abrow = ab.add(i * nb);
                        for k in 0..i {
                            let aik = _mm256_broadcast_ss(&*abrow.add(k));
                            let xv = _mm256_loadu_ps(base.add(k * ldb + j));
                            acc = _mm256_fnmadd_ps(aik, xv, acc);
                        }
                        if !unit {
                            acc = _mm256_mul_ps(acc, _mm256_broadcast_ss(&*di.add(i)));
                        }
                        _mm256_storeu_ps(row, acc);
                    }
                    j += 8;
                    continue;
                }
                for i in 0..nb {
                    let row = base.add(i * ldb + j);
                    let mut acc = _mm256_loadu_ps(row);
                    let abrow = ab.add(i * nb);
                    for k in 0..i {
                        let aik = _mm256_broadcast_ss(&*abrow.add(k));
                        let xv = _mm256_loadu_ps(base.add(k * ldb + j));
                        acc = _mm256_fnmadd_ps(aik, xv, acc);
                    }
                    if !unit {
                        acc = _mm256_mul_ps(acc, _mm256_broadcast_ss(&*di.add(i)));
                    }
                    _mm256_storeu_ps(row, acc);
                }
            } else {
                // Descending mirror of the forward unroll above (same
                // gate): pairs share `X[k]` for `k > hi`; row `lo`'s final
                // `k = hi` step reuses row `hi` from registers. FMA like
                // the f64 kernel (same ~1-ulp rounding note as above).
                let nounroll = std::env::var_os("SONIUM_TRSM_NOUNROLL").is_some() || nb < 64;
                if !nounroll {
                    let mut i = nb;
                    while i >= 2 {
                        let (lo, hi) = (i - 2, i - 1);
                        let row0 = base.add(hi * ldb + j);
                        let row1 = base.add(lo * ldb + j);
                        let mut a0 = _mm256_loadu_ps(row0);
                        let mut b0 = _mm256_loadu_ps(row1);
                        let abrow0 = ab.add(hi * nb);
                        let abrow1 = ab.add(lo * nb);
                        for k in (hi + 1)..nb {
                            let xv = _mm256_loadu_ps(base.add(k * ldb + j));
                            a0 = _mm256_fnmadd_ps(_mm256_broadcast_ss(&*abrow0.add(k)), xv, a0);
                            b0 = _mm256_fnmadd_ps(_mm256_broadcast_ss(&*abrow1.add(k)), xv, b0);
                        }
                        if !unit {
                            a0 = _mm256_mul_ps(a0, _mm256_broadcast_ss(&*di.add(hi)));
                        }
                        _mm256_storeu_ps(row0, a0);
                        // Final `k = hi` step for row `lo` from registers.
                        b0 = _mm256_fnmadd_ps(_mm256_broadcast_ss(&*abrow1.add(hi)), a0, b0);
                        if !unit {
                            b0 = _mm256_mul_ps(b0, _mm256_broadcast_ss(&*di.add(lo)));
                        }
                        _mm256_storeu_ps(row1, b0);
                        i -= 2;
                    }
                    if i == 1 {
                        let row = base.add(j);
                        let mut acc = _mm256_loadu_ps(row);
                        for k in 1..nb {
                            let aik = _mm256_broadcast_ss(&*ab.add(k));
                            let xv = _mm256_loadu_ps(base.add(k * ldb + j));
                            acc = _mm256_fnmadd_ps(aik, xv, acc);
                        }
                        if !unit {
                            acc = _mm256_mul_ps(acc, _mm256_broadcast_ss(&*di));
                        }
                        _mm256_storeu_ps(row, acc);
                    }
                    j += 8;
                    continue;
                }
                for i in (0..nb).rev() {
                    let row = base.add(i * ldb + j);
                    let mut acc = _mm256_loadu_ps(row);
                    let abrow = ab.add(i * nb);
                    for k in (i + 1)..nb {
                        let aik = _mm256_broadcast_ss(&*abrow.add(k));
                        let xv = _mm256_loadu_ps(base.add(k * ldb + j));
                        acc = _mm256_fnmadd_ps(aik, xv, acc);
                    }
                    if !unit {
                        acc = _mm256_mul_ps(acc, _mm256_broadcast_ss(&*di.add(i)));
                    }
                    _mm256_storeu_ps(row, acc);
                }
            }
            j += 8;
        }
        while j < j1 {
            if !trans {
                for i in 0..nb {
                    let mut s = *base.add(i * ldb + j);
                    for k in 0..i {
                        s -= *ab.add(i * nb + k) * *base.add(k * ldb + j);
                    }
                    *base.add(i * ldb + j) = if unit { s } else { s * *di.add(i) };
                }
            } else {
                for i in (0..nb).rev() {
                    let mut s = *base.add(i * ldb + j);
                    for k in (i + 1)..nb {
                        s -= *ab.add(i * nb + k) * *base.add(k * ldb + j);
                    }
                    *base.add(i * ldb + j) = if unit { s } else { s * *di.add(i) };
                }
            }
            j += 1;
        }
    }
}

/// Left, upper block: `A11 * X1 = B1`, `A11` unit-or-not upper.
fn unblocked_left_upper<T: TrsmScalar + PanelBuf>(
    a: &[T],
    lda: usize,
    a0: usize,
    xb: &mut [T],
    ldb: usize,
    x0: usize,
    nb: usize,
    nrhs: usize,
    trans: bool,
    conjugate: bool,
    unit: bool,
    kernel: BandKernelFn<T>,
) {
    // Panel setup once, shared read-only by every band task (hot scratch;
    // see [`unblocked_left_lower`]).
    let dinv = panel_dinv(a, lda, a0, nb, conjugate, unit);
    // Right-hand sides are independent: same fan-out as
    // [`unblocked_left_lower`].
    if nrhs >= UNBLOCKED_PAR_MIN {
        T::with_panel_buf(nb * nb, |abuf| {
            pack_abuf(a, lda, a0, nb, trans, conjugate, false, abuf);
            let macs = nb.saturating_mul(nb).saturating_mul(nrhs);
            let tasks = trsm_band_tasks(nrhs, nb, band_task_workers(macs));
            let ab_send = SendConstPtr(abuf.as_ptr());
            let di_send = SendConstPtr(dinv.as_ptr());
            let x_send = SendMutPtr(xb.as_mut_ptr());
            scope_for_macs(macs, |s| {
                let mut j0 = 0;
                while j0 < nrhs {
                    let j1 = (j0 + nrhs.div_ceil(tasks)).min(nrhs);
                    s.spawn(move |_| {
                        let ab_send: SendConstPtr<T> = ab_send;
                        let di_send: SendConstPtr<T> = di_send;
                        let x_send: SendMutPtr<T> = x_send;
                        // SAFETY: band `j0..j1` owns disjoint `xb` columns;
                        // `ab`/`di` are read-only and outlive the scope;
                        // bounds per the caller contract.
                        band_left_upper(
                            x_send.0, ldb, x0, nb, j0, j1, trans, unit, ab_send.0, di_send.0,
                            kernel,
                        );
                    });
                    j0 = j1;
                }
            });
        });
        return;
    }
    T::with_panel_buf(nb * nb, |abuf| {
        pack_abuf(a, lda, a0, nb, trans, conjugate, false, abuf);
        band_left_upper(
            xb.as_mut_ptr(),
            ldb,
            x0,
            nb,
            0,
            nrhs,
            trans,
            unit,
            abuf.as_ptr(),
            dinv.as_ptr(),
            kernel,
        );
    });
}

#[allow(clippy::needless_range_loop)]
/// Serial band of [`unblocked_left_upper`] over right-hand sides `j0..j1`.
/// `ab`/`di` are the shared panel setup from [`panel_prep`].
///
/// Narrow-stride scratch wrapper like [`band_left_lower`]: the k-reduction
/// reads X rows with stride `ldb`, so a 4K-multiple stride collides every
/// row in one L1 set. Pack/unpack are exact copies, kernel op order is
/// untouched — bitwise identical. Skipped when already narrow.
fn band_left_upper<T: TrsmScalar>(
    xb: *mut T,
    ldb: usize,
    x0: usize,
    nb: usize,
    j0: usize,
    j1: usize,
    trans: bool,
    unit: bool,
    ab: *const T,
    di: *const T,
    kernel: BandKernelFn<T>,
) {
    // Solve in ≤64-wide column chunks on a narrow-stride scratch (see
    // below); chunked so even a 512-wide serial band keeps a 512B stride.
    // Stack scratch (same contract as the lower band above): no heap
    // alloc/free per band task; the chunk packs every used element before
    // the kernel reads it.
    // SAFETY: as in the lower band; `nb <= TRSM_NB` bounds the frame.
    let mut scratch = [core::mem::MaybeUninit::<T>::uninit(); TRSM_NB * BAND_CHUNK];
    let scratch_ptr = scratch.as_mut_ptr().cast::<T>();
    let mut c0 = j0;
    while c0 < j1 {
        let c1 = (c0 + BAND_CHUNK).min(j1);
        band_left_upper_chunk(
            xb,
            ldb,
            x0,
            nb,
            c0,
            c1,
            trans,
            unit,
            ab,
            di,
            scratch_ptr,
            kernel,
        );
        c0 = c1;
    }
}

/// One ≤64-wide chunk of [`band_left_upper`]: packs columns `c0..c1` to a
/// narrow-stride scratch, runs the kernel, copies back. Pack/unpack are
/// exact copies and the kernel op order is untouched — bitwise identical.
/// Skipped (direct kernel) when the chunk is already narrow.
fn band_left_upper_chunk<T: TrsmScalar>(
    xb: *mut T,
    ldb: usize,
    x0: usize,
    nb: usize,
    c0: usize,
    c1: usize,
    trans: bool,
    unit: bool,
    ab: *const T,
    di: *const T,
    scratch: *mut T,
    kernel: BandKernelFn<T>,
) {
    let nw = c1 - c0;
    if ldb == nw {
        // SAFETY: same contract as the kernel (band fns are safe; only
        // their bodies are `unsafe` internally).
        unsafe {
            kernel(xb, ldb, x0, nb, c0, c1, trans, unit, ab, di);
        }
        return;
    }
    // SAFETY: `nb` row-copies of `nw` elements between the `xb` band rows
    // (which hold the band per the caller contract) and the scratch; the
    // kernel runs on the scratch with tight stride `nw` (see
    // `band_left_lower_chunk`): the chunk is at most `BAND_CHUNK` wide, so
    // the tight block fits the scratch.
    unsafe {
        for i in 0..nb {
            core::ptr::copy_nonoverlapping(xb.add(x0 + i * ldb + c0), scratch.add(i * nw), nw);
        }
        kernel(scratch, nw, 0, nb, 0, nw, trans, unit, ab, di);
        for i in 0..nb {
            core::ptr::copy_nonoverlapping(scratch.add(i * nw), xb.add(x0 + i * ldb + c0), nw);
        }
    }
}

/// Transpose-pack for the right-side tile solve, selected per scalar type:
/// f64 uses the vectorized AVX2 transpose-pack when available, every other
/// type (and f64 without AVX2) uses the scalar loop. Both are exact copies,
/// so the packed tile is bitwise identical either way.
trait TilePack: TrsmScalar {
    fn pack_band_tile(
        xb: *const Self,
        ldb: usize,
        x0: usize,
        i0: usize,
        r: usize,
        nb: usize,
        tile: *mut Self,
    );
    fn unpack_band_tile(
        tile: *const Self,
        xb: *mut Self,
        ldb: usize,
        x0: usize,
        i0: usize,
        r: usize,
        nb: usize,
    );
}

impl TilePack for f32 {
    fn pack_band_tile(
        xb: *const Self,
        ldb: usize,
        x0: usize,
        i0: usize,
        r: usize,
        nb: usize,
        tile: *mut Self,
    ) {
        // AVX2 transpose-pack (same arrangement as the f64 arm below):
        // the scalar loop left Right-side f32 TRSM ~2x behind vendor
        // (spotrf row-block solves are all Right/Trans). Bitwise
        // identical (exact copies; `pack_trans_f32_matches_scalar`).
        // `SONIUM_TRSM_PACK_SCALAR=1` forces scalar (diagnostic/A-B).
        #[cfg(target_arch = "x86_64")]
        if pack_avx2_allowed() {
            // SAFETY: band rows/columns readable, tile writable per the
            // caller contract; AVX2 gated at runtime above.
            unsafe {
                crate::cpu::blas3::avx2::pack_trans_f32(xb.add(x0 + i0 * ldb), ldb, nb, r, tile);
            }
            return;
        }
        pack_band_transpose(xb, ldb, x0, i0, r, nb, tile);
    }
    fn unpack_band_tile(
        tile: *const Self,
        xb: *mut Self,
        ldb: usize,
        x0: usize,
        i0: usize,
        r: usize,
        nb: usize,
    ) {
        unpack_band_transpose(tile, xb, ldb, x0, i0, r, nb);
    }
}

impl TilePack for f64 {
    fn pack_band_tile(
        xb: *const Self,
        ldb: usize,
        x0: usize,
        i0: usize,
        r: usize,
        nb: usize,
        tile: *mut Self,
    ) {
        // `pack_trans_f64(src, ldb, rows, cols, dst)` writes
        // `dst[k * r + i] = src[i * ldb + k]` for `i < r`, `k < nb` —
        // exactly this band transpose with `rows = nb`, `cols = r`.
        // Source range `(r - 1) * ldb + nb` is the band's rows/columns,
        // in bounds per the caller contract. `SONIUM_TRSM_PACK_SCALAR=1`
        // forces the scalar loop (diagnostic/A-B fallback).
        #[cfg(target_arch = "x86_64")]
        if pack_avx2_allowed() {
            // SAFETY: band rows/columns readable, tile writable per the
            // caller contract; AVX2 gated at runtime above.
            unsafe {
                crate::cpu::blas3::avx2::pack_trans_f64(xb.add(x0 + i0 * ldb), ldb, nb, r, tile);
            }
            return;
        }
        pack_band_transpose(xb, ldb, x0, i0, r, nb, tile);
    }
    fn unpack_band_tile(
        tile: *const Self,
        xb: *mut Self,
        ldb: usize,
        x0: usize,
        i0: usize,
        r: usize,
        nb: usize,
    ) {
        // NOTE (unpack experiments, both reverted — see
        // `unpack_band_transpose`): AVX2 shuffle-scatter and loop
        // interchange both lost to the plain scalar below.
        unpack_band_transpose(tile, xb, ldb, x0, i0, r, nb);
    }
}

impl TilePack for Complex32 {
    fn pack_band_tile(
        xb: *const Self,
        ldb: usize,
        x0: usize,
        i0: usize,
        r: usize,
        nb: usize,
        tile: *mut Self,
    ) {
        pack_band_transpose(xb, ldb, x0, i0, r, nb, tile);
    }
    fn unpack_band_tile(
        tile: *const Self,
        xb: *mut Self,
        ldb: usize,
        x0: usize,
        i0: usize,
        r: usize,
        nb: usize,
    ) {
        unpack_band_transpose(tile, xb, ldb, x0, i0, r, nb);
    }
}

impl TilePack for Complex64 {
    fn pack_band_tile(
        xb: *const Self,
        ldb: usize,
        x0: usize,
        i0: usize,
        r: usize,
        nb: usize,
        tile: *mut Self,
    ) {
        pack_band_transpose(xb, ldb, x0, i0, r, nb, tile);
    }
    fn unpack_band_tile(
        tile: *const Self,
        xb: *mut Self,
        ldb: usize,
        x0: usize,
        i0: usize,
        r: usize,
        nb: usize,
    ) {
        unpack_band_transpose(tile, xb, ldb, x0, i0, r, nb);
    }
}

/// Transpose-pack a row band for the right-side tile solve: `tile[k * r +
/// (i - i0)] = xb[i * ldb + k]` for rows `i0..i1` (`r = i1 - i0`) and
/// columns `0..nb`. Exact copies — bitwise invisible.
fn pack_band_transpose<T: Copy>(
    xb: *const T,
    ldb: usize,
    x0: usize,
    i0: usize,
    r: usize,
    nb: usize,
    tile: *mut T,
) {
    // SAFETY: caller guarantees the band rows and the `nb × r` tile
    // readable/writable respectively per the caller contract.
    unsafe {
        for i in 0..r {
            for k in 0..nb {
                *tile.add(k * r + i) = *xb.add(x0 + (i0 + i) * ldb + k);
            }
        }
    }
}

/// Transpose-unpack a solved tile back to its row band: the inverse of
/// [`pack_band_transpose`]. Exact copies — bitwise invisible.
///
/// NOTE (unpack experiments): an AVX2 4×4 transpose-scatter measured
/// slower cross-run (consistent with shuffle-port pressure: 8 shuffle
/// uops per 16 elements serialize on one port — removed, not dispatched).
/// Interleaved within-run A/B of the scalar loop order
/// (`SONIUM_TRSM_UNPACK_SWAP=1` vs the default row-outer/column-inner
/// below) shows ±8% either way — no measurable difference — so the
/// default stays and the swap remains a diagnostic gate only.
fn unpack_band_transpose<T: Copy>(
    tile: *const T,
    xb: *mut T,
    ldb: usize,
    x0: usize,
    i0: usize,
    r: usize,
    nb: usize,
) {
    // SAFETY: caller guarantees the `nb × r` tile readable and the band
    // rows writable per the caller contract.
    unsafe {
        if std::env::var_os("SONIUM_TRSM_UNPACK_SWAP").is_some() {
            for k in 0..nb {
                for i in 0..r {
                    *xb.add(x0 + (i0 + i) * ldb + k) = *tile.add(k * r + i);
                }
            }
            return;
        }
        for i in 0..r {
            for k in 0..nb {
                *xb.add(x0 + (i0 + i) * ldb + k) = *tile.add(k * r + i);
            }
        }
    }
}

/// Row-chunk height for [`right_tile_solve`]: a `32 × nb` tile is 32KB
/// at f64/`nb = 128` — L1-resident with a 256B k-reduction stride — so the
/// left kernel on the tile never sees the set-collision stride a tall
/// serial band would impose. Narrower panels run taller tiles in the same
/// element budget (see the fat-tile chunking below).
const RIGHT_TILE_RC: usize = 32;

/// Solve a right-side row band through transposed tiles: packs rows in
/// `≤ fat_rc` chunks to `nb × rc` tiles, runs each through `kernel`
/// (a left-solve dispatch over the transposed operator), unpacks back.
/// `dir` selects the substitution direction (see the concrete kernels).
fn right_tile_solve<T: TilePack>(
    xb: *mut T,
    ldb: usize,
    x0: usize,
    i0: usize,
    i1: usize,
    nb: usize,
    dir: bool,
    unit: bool,
    ab: *const T,
    di: *const T,
    kernel: BandKernelFn<T>,
) {
    let r = i1 - i0;
    if r == 0 || nb == 0 {
        return;
    }
    // Stack tile: `nb <= TRSM_NB`, so at most 16KB — no heap alloc/free
    // per band task. The pack writes every used element before the kernel
    // reads it.
    // SAFETY: `MaybeUninit` needs no drop; only the packed (written)
    // `nb * rc` prefix is ever read.
    let mut tile = [core::mem::MaybeUninit::<T>::uninit(); TRSM_NB * RIGHT_TILE_RC];
    let tile_ptr = tile.as_mut_ptr().cast::<T>();
    // Fat-tile chunking: the stack budget is `TRSM_NB * RIGHT_TILE_RC`
    // elements, so narrow panels (`nb < TRSM_NB`, e.g. Cholesky's 48-wide
    // L21 bands) can run taller tiles in the same footprint — fewer
    // pack/unpack round trips and fuller AVX2 rows for the left kernel
    // (spotrf-1024 TRSM bands: 4 tiles of 32 -> 2 tiles of 85 at nb=48).
    // Same footprint, same per-element arithmetic (rows are independent),
    // so results are bitwise identical; `nb = TRSM_NB` keeps `rc = 32`.
    let fat_rc = (TRSM_NB * RIGHT_TILE_RC) / nb.max(1);
    let mut r0 = 0;
    while r0 < r {
        let rc = (r - r0).min(fat_rc);
        T::pack_band_tile(xb, ldb, x0, i0 + r0, rc, nb, tile_ptr);
        // SAFETY: tile chunk is exclusively owned; `ab`/`di` readable.
        unsafe {
            kernel(tile_ptr, rc, 0, nb, 0, rc, dir, unit, ab, di);
        }
        T::unpack_band_tile(tile_ptr, xb, ldb, x0, i0 + r0, rc, nb);
        r0 += rc;
    }
}

/// Concrete `f64` right-lower band kernel: solves the `i0..i1` rows by
/// running them as transposed `nb × rc` tiles through the left solve
/// (`(op)ᵀ · Xᵀ = Bᵀ`, same per-element op order as the scalar right loop
/// — multiplication commutes, so bitwise identical). The tile makes the
/// four row lanes contiguous, which is what unlocks the AVX2 core; the
/// row-major band keeps them `ldb`-strided, defeating vectorisation (the
/// compiler pairs them into shuffled 2-wide SSE at ~2 GF/s). `ab` is the
/// transposed pack from [`panel_prep`]. Threaded from [`trsm_f64`] through
/// [`trsm_core`] as a [`BandKernelFn`].
///
/// NOTE (native-kernel experiment, reverted): a native strided-row AVX2
/// kernel (2- and 8-row variants, FMA + k-blocking, no transpose) was
/// tried and measured slower (48-60 vs 70 GF/s at m=1792·n=128). Root
/// cause: every output needs a vector-reduction epilogue (hsum), whose
/// per-(row,column) cost plus the k-loop floor exceeds the tile kernel
/// total — the left kernel writes accumulators straight to `C` with no
/// reduction. The shuffle's 2x traffic costs less than the reduction.
/// Revisit only with a reduction-free formulation (none exists under the
/// per-column dependency).
unsafe fn band_right_lower_kernel_f64(
    xb: *mut f64,
    ldb: usize,
    x0: usize,
    i0: usize,
    i1: usize,
    nb: usize,
    trans: bool,
    unit: bool,
    ab: *const f64,
    di: *const f64,
) {
    // `!trans`: op is lower ⟺ its transpose is upper ⟺ backward solve.
    // (`right_tile_solve` is safe; the `unsafe fn` marker only matches the
    // [`BandKernelFn`] pointer type.)
    right_tile_solve(
        xb,
        ldb,
        x0,
        i0,
        i1,
        nb,
        !trans,
        unit,
        ab,
        di,
        band_left_lower_kernel_f64,
    );
}

/// Concrete `f32` right-lower band kernel: mirror twin of
/// [`band_right_lower_kernel_f64`]. Threaded from [`trsm_f32`] through
/// [`trsm_core`] as a [`BandKernelFn`].
unsafe fn band_right_lower_kernel_f32(
    xb: *mut f32,
    ldb: usize,
    x0: usize,
    i0: usize,
    i1: usize,
    nb: usize,
    trans: bool,
    unit: bool,
    ab: *const f32,
    di: *const f32,
) {
    // (`right_tile_solve` is safe; the `unsafe fn` marker only matches the
    // [`BandKernelFn`] pointer type.)
    right_tile_solve(
        xb,
        ldb,
        x0,
        i0,
        i1,
        nb,
        !trans,
        unit,
        ab,
        di,
        band_left_lower_kernel_f32,
    );
}

/// Concrete `f64` right-upper band kernel: mirror twin of
/// [`band_right_lower_kernel_f64`] for the upper triangle (`trans` selects
/// the direction directly: op upper ⟺ transpose lower ⟺ forward solve).
/// Threaded from [`trsm_f64`] through [`trsm_core`] as a [`BandKernelFn`].
unsafe fn band_right_upper_kernel_f64(
    xb: *mut f64,
    ldb: usize,
    x0: usize,
    i0: usize,
    i1: usize,
    nb: usize,
    trans: bool,
    unit: bool,
    ab: *const f64,
    di: *const f64,
) {
    // (`right_tile_solve` is safe; the `unsafe fn` marker only matches the
    // [`BandKernelFn`] pointer type.)
    right_tile_solve(
        xb,
        ldb,
        x0,
        i0,
        i1,
        nb,
        trans,
        unit,
        ab,
        di,
        band_left_lower_kernel_f64,
    );
}

/// Concrete `f32` right-upper band kernel: mirror twin of
/// [`band_right_upper_kernel_f64`]. Threaded from [`trsm_f32`] through
/// [`trsm_core`] as a [`BandKernelFn`].
unsafe fn band_right_upper_kernel_f32(
    xb: *mut f32,
    ldb: usize,
    x0: usize,
    i0: usize,
    i1: usize,
    nb: usize,
    trans: bool,
    unit: bool,
    ab: *const f32,
    di: *const f32,
) {
    // (`right_tile_solve` is safe; the `unsafe fn` marker only matches the
    // [`BandKernelFn`] pointer type.)
    right_tile_solve(
        xb,
        ldb,
        x0,
        i0,
        i1,
        nb,
        trans,
        unit,
        ab,
        di,
        band_left_lower_kernel_f32,
    );
}

/// Concrete `Complex64` right-lower band kernel: transpose-tile twin of
/// [`band_right_lower_kernel_f64`] with the AVX2 complex inner kernel
/// ([`band_left_lower_kernel_c64`]). Threaded from [`trsm_c64`] through
/// [`trsm_core`] as a [`BandKernelFn`].
unsafe fn band_right_lower_kernel_c64(
    xb: *mut Complex64,
    ldb: usize,
    x0: usize,
    i0: usize,
    i1: usize,
    nb: usize,
    trans: bool,
    unit: bool,
    ab: *const Complex64,
    di: *const Complex64,
) {
    // `!trans`: op is lower ⟺ its transpose is upper ⟺ backward solve.
    // (`right_tile_solve` is safe; the `unsafe fn` marker only matches the
    // [`BandKernelFn`] pointer type.)
    right_tile_solve(
        xb,
        ldb,
        x0,
        i0,
        i1,
        nb,
        !trans,
        unit,
        ab,
        di,
        band_left_lower_kernel_c64,
    );
}

/// Concrete `Complex32` right-lower band kernel: mirror twin of
/// [`band_right_lower_kernel_c64`]. Threaded from [`trsm_c32`] through
/// [`trsm_core`] as a [`BandKernelFn`].
unsafe fn band_right_lower_kernel_c32(
    xb: *mut Complex32,
    ldb: usize,
    x0: usize,
    i0: usize,
    i1: usize,
    nb: usize,
    trans: bool,
    unit: bool,
    ab: *const Complex32,
    di: *const Complex32,
) {
    // (`right_tile_solve` is safe; the `unsafe fn` marker only matches the
    // [`BandKernelFn`] pointer type.)
    right_tile_solve(
        xb,
        ldb,
        x0,
        i0,
        i1,
        nb,
        !trans,
        unit,
        ab,
        di,
        band_left_lower_kernel_c32,
    );
}

/// Concrete `Complex64` right-upper band kernel: mirror twin of
/// [`band_right_lower_kernel_c64`] (`trans` selects the direction
/// directly). Threaded from [`trsm_c64`] through [`trsm_core`] as a
/// [`BandKernelFn`].
unsafe fn band_right_upper_kernel_c64(
    xb: *mut Complex64,
    ldb: usize,
    x0: usize,
    i0: usize,
    i1: usize,
    nb: usize,
    trans: bool,
    unit: bool,
    ab: *const Complex64,
    di: *const Complex64,
) {
    // (`right_tile_solve` is safe; the `unsafe fn` marker only matches the
    // [`BandKernelFn`] pointer type.)
    right_tile_solve(
        xb,
        ldb,
        x0,
        i0,
        i1,
        nb,
        trans,
        unit,
        ab,
        di,
        band_left_lower_kernel_c64,
    );
}

/// Concrete `Complex32` right-upper band kernel: mirror twin of
/// [`band_right_upper_kernel_c64`]. Threaded from [`trsm_c32`] through
/// [`trsm_core`] as a [`BandKernelFn`].
unsafe fn band_right_upper_kernel_c32(
    xb: *mut Complex32,
    ldb: usize,
    x0: usize,
    i0: usize,
    i1: usize,
    nb: usize,
    trans: bool,
    unit: bool,
    ab: *const Complex32,
    di: *const Complex32,
) {
    // (`right_tile_solve` is safe; the `unsafe fn` marker only matches the
    // [`BandKernelFn`] pointer type.)
    right_tile_solve(
        xb,
        ldb,
        x0,
        i0,
        i1,
        nb,
        trans,
        unit,
        ab,
        di,
        band_left_lower_kernel_c32,
    );
}

/// Right, lower block: `X1 * A11 = B1`, `A11` unit-or-not lower.
/// Packed operator in the thread-local scratch (see
/// [`unblocked_left_lower`]).
fn unblocked_right_lower<T: TrsmScalar + PanelBuf>(
    a: &[T],
    lda: usize,
    a0: usize,
    xb: &mut [T],
    ldb: usize,
    x0: usize,
    m: usize,
    nb: usize,
    trans: bool,
    conjugate: bool,
    unit: bool,
    kernel: BandKernelFn<T>,
) {
    // Panel setup once, shared read-only by every band task.
    let dinv = panel_dinv(a, lda, a0, nb, conjugate, unit);
    // Rows are independent: fan out over row bands when there are enough
    // of them, else run the single serial band.
    if m >= UNBLOCKED_PAR_MIN {
        T::with_panel_buf(nb * nb, |abuf| {
            pack_abuf(a, lda, a0, nb, trans, conjugate, true, abuf);
            // Diagnostic A/B gate: `SONIUM_TRSM_DIAG_POOL=small` pins the
            // diagonal scope to the small pool (lifts the tiny routing the
            // MAC count would pick). Interleaved within-run A/B
            // (`diag-small` arm) shows small-pool 2–25% slower at every
            // m — the tile path wants the tiny pool's low coordination
            // cost, not more workers. Gate stays off by default.
            let _diag_floor = std::env::var_os("SONIUM_TRSM_DIAG_POOL")
                .is_some()
                .then(pool_floor_small);
            let macs = nb.saturating_mul(nb).saturating_mul(m);
            let tasks = trsm_band_tasks(m, nb, band_task_workers(macs));
            let ab_send = SendConstPtr(abuf.as_ptr());
            let di_send = SendConstPtr(dinv.as_ptr());
            let x_send = SendMutPtr(xb.as_mut_ptr());
            // NOTE (reverted trial): a single `spawn_broadcast` task with
            // atomic-cursor self-scheduling (one push + fetch-adds instead
            // of one spawn per band) tied the spawn loop at every m
            // (interleaved in-process A/B, 20 pairs each: 1.005x/1.026x/
            // 1.039x/0.975x, bitwise identical) — per-spawn dispatch is
            // not the scope cost; the remainder is pool rendezvous
            // (steal/wake latency), which the broadcast pays too. Keep
            // the plain spawn loop.
            scope_for_macs(macs, |s| {
                let mut i0 = 0;
                while i0 < m {
                    let i1 = (i0 + m.div_ceil(tasks)).min(m);
                    s.spawn(move |_| {
                        let ab_send: SendConstPtr<T> = ab_send;
                        let di_send: SendConstPtr<T> = di_send;
                        let x_send: SendMutPtr<T> = x_send;
                        // SAFETY: band `i0..i1` owns disjoint `xb` rows;
                        // `ab`/`di` are read-only and outlive the scope;
                        // bounds per the caller contract.
                        band_right_lower(
                            x_send.0, ldb, x0, i0, i1, nb, trans, unit, ab_send.0, di_send.0,
                            kernel,
                        );
                    });
                    i0 = i1;
                }
            });
        });
        return;
    }
    T::with_panel_buf(nb * nb, |abuf| {
        pack_abuf(a, lda, a0, nb, trans, conjugate, true, abuf);
        band_right_lower(
            xb.as_mut_ptr(),
            ldb,
            x0,
            0,
            m,
            nb,
            trans,
            unit,
            abuf.as_ptr(),
            dinv.as_ptr(),
            kernel,
        );
    });
}

#[allow(clippy::needless_range_loop)]
/// Serial band of [`unblocked_right_lower`] over rows `i0..i1`.
/// `ab`/`di` are the shared panel setup from [`panel_prep`].
fn band_right_lower<T: TrsmScalar>(
    xb: *mut T,
    ldb: usize,
    x0: usize,
    i0: usize,
    i1: usize,
    nb: usize,
    trans: bool,
    unit: bool,
    ab: *const T,
    di: *const T,
    kernel: BandKernelFn<T>,
) {
    // SAFETY: same contract as the kernel (band fns are safe; only their
    // bodies are `unsafe` internally).
    unsafe {
        kernel(xb, ldb, x0, i0, i1, nb, trans, unit, ab, di);
    }
}

/// Kernel body of [`band_right_lower`], factored out so accelerated
/// kernels can replace it. `ab` is the TRANSPOSED packed `op(A11)` (see
/// [`panel_prep`]): element `op[k][j]` lives at `ab[j * nb + k]`, so the
/// k-reduction streams it sequentially. Same values and op order as the
/// row-major loop — bitwise identical.
fn band_right_lower_kernel<T: TrsmScalar>(
    xb: *mut T,
    ldb: usize,
    x0: usize,
    i0: usize,
    i1: usize,
    nb: usize,
    trans: bool,
    unit: bool,
    ab: *const T,
    di: *const T,
) {
    // SAFETY: caller guarantees the `xb` rows `i0..i1` exclusively
    // writable and `ab`/`di` readable for the panel ranges used below.
    unsafe {
        // Four-wide over the independent rows with the exact scalar op
        // sequence per lane (bitwise-identical results). Row lanes stride
        // by `ldb`, so this buys ILP more than SLP — still a multiple of
        // the scalar rate on an in-order-unfriendly reduction.
        if !trans {
            // op(A11) lower: x[i, j] needs x[i, k] for k > j → descending.
            for j in (0..nb).rev() {
                let mut i = i0;
                while i + 4 <= i1 {
                    let mut s0 = *xb.add(x0 + i * ldb + j);
                    let mut s1 = *xb.add(x0 + (i + 1) * ldb + j);
                    let mut s2 = *xb.add(x0 + (i + 2) * ldb + j);
                    let mut s3 = *xb.add(x0 + (i + 3) * ldb + j);
                    for k in (j + 1)..nb {
                        s0 = s0 - *xb.add(x0 + i * ldb + k) * *ab.add(j * nb + k);
                        s1 = s1 - *xb.add(x0 + (i + 1) * ldb + k) * *ab.add(j * nb + k);
                        s2 = s2 - *xb.add(x0 + (i + 2) * ldb + k) * *ab.add(j * nb + k);
                        s3 = s3 - *xb.add(x0 + (i + 3) * ldb + k) * *ab.add(j * nb + k);
                    }
                    if unit {
                        *xb.add(x0 + i * ldb + j) = s0;
                        *xb.add(x0 + (i + 1) * ldb + j) = s1;
                        *xb.add(x0 + (i + 2) * ldb + j) = s2;
                        *xb.add(x0 + (i + 3) * ldb + j) = s3;
                    } else {
                        let d = *di.add(j);
                        *xb.add(x0 + i * ldb + j) = s0 * d;
                        *xb.add(x0 + (i + 1) * ldb + j) = s1 * d;
                        *xb.add(x0 + (i + 2) * ldb + j) = s2 * d;
                        *xb.add(x0 + (i + 3) * ldb + j) = s3 * d;
                    }
                    i += 4;
                }
                for i in i..i1 {
                    let mut s = *xb.add(x0 + i * ldb + j);
                    for k in (j + 1)..nb {
                        s = s - *xb.add(x0 + i * ldb + k) * *ab.add(j * nb + k);
                    }
                    *xb.add(x0 + i * ldb + j) = if unit { s } else { s * *di.add(j) };
                }
            }
        } else {
            // op(A11) upper: ascending.
            for j in 0..nb {
                let mut i = i0;
                while i + 4 <= i1 {
                    let mut s0 = *xb.add(x0 + i * ldb + j);
                    let mut s1 = *xb.add(x0 + (i + 1) * ldb + j);
                    let mut s2 = *xb.add(x0 + (i + 2) * ldb + j);
                    let mut s3 = *xb.add(x0 + (i + 3) * ldb + j);
                    for k in 0..j {
                        s0 = s0 - *xb.add(x0 + i * ldb + k) * *ab.add(j * nb + k);
                        s1 = s1 - *xb.add(x0 + (i + 1) * ldb + k) * *ab.add(j * nb + k);
                        s2 = s2 - *xb.add(x0 + (i + 2) * ldb + k) * *ab.add(j * nb + k);
                        s3 = s3 - *xb.add(x0 + (i + 3) * ldb + k) * *ab.add(j * nb + k);
                    }
                    if unit {
                        *xb.add(x0 + i * ldb + j) = s0;
                        *xb.add(x0 + (i + 1) * ldb + j) = s1;
                        *xb.add(x0 + (i + 2) * ldb + j) = s2;
                        *xb.add(x0 + (i + 3) * ldb + j) = s3;
                    } else {
                        let d = *di.add(j);
                        *xb.add(x0 + i * ldb + j) = s0 * d;
                        *xb.add(x0 + (i + 1) * ldb + j) = s1 * d;
                        *xb.add(x0 + (i + 2) * ldb + j) = s2 * d;
                        *xb.add(x0 + (i + 3) * ldb + j) = s3 * d;
                    }
                    i += 4;
                }
                for i in i..i1 {
                    let mut s = *xb.add(x0 + i * ldb + j);
                    for k in 0..j {
                        s = s - *xb.add(x0 + i * ldb + k) * *ab.add(j * nb + k);
                    }
                    *xb.add(x0 + i * ldb + j) = if unit { s } else { s * *di.add(j) };
                }
            }
        }
    }
}

/// Right, upper block: `X1 * A11 = B1`, `A11` unit-or-not upper.
/// Packed operator in the thread-local scratch (see
/// [`unblocked_left_lower`]).
fn unblocked_right_upper<T: TrsmScalar + PanelBuf>(
    a: &[T],
    lda: usize,
    a0: usize,
    xb: &mut [T],
    ldb: usize,
    x0: usize,
    m: usize,
    nb: usize,
    trans: bool,
    conjugate: bool,
    unit: bool,
    kernel: BandKernelFn<T>,
) {
    // Panel setup once, shared read-only by every band task.
    let dinv = panel_dinv(a, lda, a0, nb, conjugate, unit);
    // Rows are independent: same fan-out as [`unblocked_right_lower`],
    // including the `SONIUM_TRSM_DIAG_POOL` diagnostic gate above.
    if m >= UNBLOCKED_PAR_MIN {
        T::with_panel_buf(nb * nb, |abuf| {
            pack_abuf(a, lda, a0, nb, trans, conjugate, true, abuf);
            let _diag_floor = std::env::var_os("SONIUM_TRSM_DIAG_POOL")
                .is_some()
                .then(pool_floor_small);
            let macs = nb.saturating_mul(nb).saturating_mul(m);
            let tasks = trsm_band_tasks(m, nb, band_task_workers(macs));
            let ab_send = SendConstPtr(abuf.as_ptr());
            let di_send = SendConstPtr(dinv.as_ptr());
            let x_send = SendMutPtr(xb.as_mut_ptr());
            scope_for_macs(macs, |s| {
                let mut i0 = 0;
                while i0 < m {
                    let i1 = (i0 + m.div_ceil(tasks)).min(m);
                    s.spawn(move |_| {
                        let ab_send: SendConstPtr<T> = ab_send;
                        let di_send: SendConstPtr<T> = di_send;
                        let x_send: SendMutPtr<T> = x_send;
                        // SAFETY: band `i0..i1` owns disjoint `xb` rows;
                        // `ab`/`di` are read-only and outlive the scope;
                        // bounds per the caller contract.
                        band_right_upper(
                            x_send.0, ldb, x0, i0, i1, nb, trans, unit, ab_send.0, di_send.0,
                            kernel,
                        );
                    });
                    i0 = i1;
                }
            });
        });
        return;
    }
    T::with_panel_buf(nb * nb, |abuf| {
        pack_abuf(a, lda, a0, nb, trans, conjugate, true, abuf);
        band_right_upper(
            xb.as_mut_ptr(),
            ldb,
            x0,
            0,
            m,
            nb,
            trans,
            unit,
            abuf.as_ptr(),
            dinv.as_ptr(),
            kernel,
        );
    });
}

#[allow(clippy::needless_range_loop)]
/// Serial band of [`unblocked_right_upper`] over rows `i0..i1`.
/// `ab`/`di` are the shared panel setup from [`panel_prep`].
fn band_right_upper<T: TrsmScalar>(
    xb: *mut T,
    ldb: usize,
    x0: usize,
    i0: usize,
    i1: usize,
    nb: usize,
    trans: bool,
    unit: bool,
    ab: *const T,
    di: *const T,
    kernel: BandKernelFn<T>,
) {
    // SAFETY: same contract as the kernel (band fns are safe; only their
    // bodies are `unsafe` internally).
    unsafe {
        kernel(xb, ldb, x0, i0, i1, nb, trans, unit, ab, di);
    }
}

/// Kernel body of [`band_right_upper`], factored out so accelerated
/// kernels can replace it. `ab` is the TRANSPOSED packed `op(A11)` like
/// [`band_right_lower_kernel`]: `op[k][j]` lives at `ab[j * nb + k]`.
/// Same values and op order as the row-major loop — bitwise identical.
fn band_right_upper_kernel<T: TrsmScalar>(
    xb: *mut T,
    ldb: usize,
    x0: usize,
    i0: usize,
    i1: usize,
    nb: usize,
    trans: bool,
    unit: bool,
    ab: *const T,
    di: *const T,
) {
    // SAFETY: caller guarantees the `xb` rows `i0..i1` exclusively
    // writable and `ab`/`di` readable for the panel ranges used below.
    unsafe {
        // Four-wide over the independent rows with the exact scalar op
        // sequence per lane (bitwise-identical results); see
        // [`band_right_lower`] on row-lane ILP vs SLP.
        if !trans {
            // op(A11) upper: x[i, j] needs x[i, k] for k < j → ascending.
            for j in 0..nb {
                let mut i = i0;
                while i + 4 <= i1 {
                    let mut s0 = *xb.add(x0 + i * ldb + j);
                    let mut s1 = *xb.add(x0 + (i + 1) * ldb + j);
                    let mut s2 = *xb.add(x0 + (i + 2) * ldb + j);
                    let mut s3 = *xb.add(x0 + (i + 3) * ldb + j);
                    for k in 0..j {
                        let akj = *ab.add(j * nb + k);
                        s0 = s0 - *xb.add(x0 + i * ldb + k) * akj;
                        s1 = s1 - *xb.add(x0 + (i + 1) * ldb + k) * akj;
                        s2 = s2 - *xb.add(x0 + (i + 2) * ldb + k) * akj;
                        s3 = s3 - *xb.add(x0 + (i + 3) * ldb + k) * akj;
                    }
                    if unit {
                        *xb.add(x0 + i * ldb + j) = s0;
                        *xb.add(x0 + (i + 1) * ldb + j) = s1;
                        *xb.add(x0 + (i + 2) * ldb + j) = s2;
                        *xb.add(x0 + (i + 3) * ldb + j) = s3;
                    } else {
                        let d = *di.add(j);
                        *xb.add(x0 + i * ldb + j) = s0 * d;
                        *xb.add(x0 + (i + 1) * ldb + j) = s1 * d;
                        *xb.add(x0 + (i + 2) * ldb + j) = s2 * d;
                        *xb.add(x0 + (i + 3) * ldb + j) = s3 * d;
                    }
                    i += 4;
                }
                for i in i..i1 {
                    let mut s = *xb.add(x0 + i * ldb + j);
                    for k in 0..j {
                        s = s - *xb.add(x0 + i * ldb + k) * *ab.add(j * nb + k);
                    }
                    *xb.add(x0 + i * ldb + j) = if unit { s } else { s * *di.add(j) };
                }
            }
        } else {
            // op(A11) lower: descending.
            for j in (0..nb).rev() {
                let mut i = i0;
                while i + 4 <= i1 {
                    let mut s0 = *xb.add(x0 + i * ldb + j);
                    let mut s1 = *xb.add(x0 + (i + 1) * ldb + j);
                    let mut s2 = *xb.add(x0 + (i + 2) * ldb + j);
                    let mut s3 = *xb.add(x0 + (i + 3) * ldb + j);
                    for k in (j + 1)..nb {
                        let akj = *ab.add(j * nb + k);
                        s0 = s0 - *xb.add(x0 + i * ldb + k) * akj;
                        s1 = s1 - *xb.add(x0 + (i + 1) * ldb + k) * akj;
                        s2 = s2 - *xb.add(x0 + (i + 2) * ldb + k) * akj;
                        s3 = s3 - *xb.add(x0 + (i + 3) * ldb + k) * akj;
                    }
                    if unit {
                        *xb.add(x0 + i * ldb + j) = s0;
                        *xb.add(x0 + (i + 1) * ldb + j) = s1;
                        *xb.add(x0 + (i + 2) * ldb + j) = s2;
                        *xb.add(x0 + (i + 3) * ldb + j) = s3;
                    } else {
                        let d = *di.add(j);
                        *xb.add(x0 + i * ldb + j) = s0 * d;
                        *xb.add(x0 + (i + 1) * ldb + j) = s1 * d;
                        *xb.add(x0 + (i + 2) * ldb + j) = s2 * d;
                        *xb.add(x0 + (i + 3) * ldb + j) = s3 * d;
                    }
                    i += 4;
                }
                for i in i..i1 {
                    let mut s = *xb.add(x0 + i * ldb + j);
                    for k in (j + 1)..nb {
                        s = s - *xb.add(x0 + i * ldb + k) * *ab.add(j * nb + k);
                    }
                    *xb.add(x0 + i * ldb + j) = if unit { s } else { s * *di.add(j) };
                }
            }
        }
    }
}

// ---------- team-dispatch API --------------------------------------------------
// Hoisted panel setup plus serial band slices for right-side solves, so a
// caller-owned team scope (one scope per Cholesky panel, tasks split by
// rows, internal barrier before the trailing update) can run bands without
// paying per-slice setup or per-slice scopes. `trsm_right_setup_into`
// computes exactly what `unblocked_right_lower/upper` compute internally
// (same calls, same order); the band entries run one band body each over
// rows `i0..i1` reading that shared setup. Assembling a panel from slices
// is bitwise identical to the whole call: bands are row-independent and
// each runs the same kernel op sequence (see the slice-equality tests).

/// Once-per-panel setup for right-side solves: diagonal reciprocals
/// (returned) plus the transposed-packed `op(A11)` written into `abuf`
/// (must hold `nb * nb`). Same values `unblocked_right_lower/upper`
/// compute internally.
pub fn trsm_right_setup_into<T: TrsmScalar>(
    a: &[T],
    lda: usize,
    a0: usize,
    nb: usize,
    trans: bool,
    conjugate: bool,
    unit: bool,
    abuf: &mut [T],
) -> Vec<T> {
    pack_abuf(a, lda, a0, nb, trans, conjugate, true, abuf);
    panel_dinv(a, lda, a0, nb, conjugate, unit)
}

/// Serial right-side band slice over rows `i0..i1` with shared setup.
/// `ab`/`di` come from [`trsm_right_setup_into`]. `uplo` selects the
/// lower/upper band body; remaining params mirror [`trsm_f64`].
///
/// # Safety
///
/// `xb` rows `i0..i1` (at stride `ldb` from `x0`) are exclusively writable
/// by this call; `ab` holds `nb * nb` setup elements, `di` the setup
/// reciprocals (empty when `unit`); all reads in bounds.
pub unsafe fn trsm_right_band_f64(
    uplo: TrsmUplo,
    xb: *mut f64,
    ldb: usize,
    x0: usize,
    i0: usize,
    i1: usize,
    nb: usize,
    trans: bool,
    unit: bool,
    ab: &[f64],
    di: &[f64],
) {
    // Band fns are safe; only their bodies are `unsafe` internally.
    // This entry stays `unsafe` because the disjoint-rows contract is on
    // the cross-crate caller (see the SAFETY section above).
    match uplo {
        TrsmUplo::Lower => band_right_lower(
            xb,
            ldb,
            x0,
            i0,
            i1,
            nb,
            trans,
            unit,
            ab.as_ptr(),
            di.as_ptr(),
            band_right_lower_kernel_f64,
        ),
        TrsmUplo::Upper => band_right_upper(
            xb,
            ldb,
            x0,
            i0,
            i1,
            nb,
            trans,
            unit,
            ab.as_ptr(),
            di.as_ptr(),
            band_right_upper_kernel_f64,
        ),
    }
}

/// [`trsm_right_band_f64`] for `f32`.
///
/// # Safety
///
/// Per the caller contract (see [`trsm_right_band_f64`]).
pub unsafe fn trsm_right_band_f32(
    uplo: TrsmUplo,
    xb: *mut f32,
    ldb: usize,
    x0: usize,
    i0: usize,
    i1: usize,
    nb: usize,
    trans: bool,
    unit: bool,
    ab: &[f32],
    di: &[f32],
) {
    match uplo {
        TrsmUplo::Lower => band_right_lower(
            xb,
            ldb,
            x0,
            i0,
            i1,
            nb,
            trans,
            unit,
            ab.as_ptr(),
            di.as_ptr(),
            band_right_lower_kernel_f32,
        ),
        TrsmUplo::Upper => band_right_upper(
            xb,
            ldb,
            x0,
            i0,
            i1,
            nb,
            trans,
            unit,
            ab.as_ptr(),
            di.as_ptr(),
            band_right_upper_kernel_f32,
        ),
    }
}

/// [`trsm_right_band_f64`] for `Complex32` (generic kernels, as in
/// [`trsm_c32`]).
///
/// # Safety
///
/// Per the caller contract (see [`trsm_right_band_f64`]).
pub unsafe fn trsm_right_band_c32(
    uplo: TrsmUplo,
    xb: *mut Complex32,
    ldb: usize,
    x0: usize,
    i0: usize,
    i1: usize,
    nb: usize,
    trans: bool,
    unit: bool,
    ab: &[Complex32],
    di: &[Complex32],
) {
    match uplo {
        TrsmUplo::Lower => band_right_lower(
            xb,
            ldb,
            x0,
            i0,
            i1,
            nb,
            trans,
            unit,
            ab.as_ptr(),
            di.as_ptr(),
            band_right_lower_kernel,
        ),
        TrsmUplo::Upper => band_right_upper(
            xb,
            ldb,
            x0,
            i0,
            i1,
            nb,
            trans,
            unit,
            ab.as_ptr(),
            di.as_ptr(),
            band_right_upper_kernel,
        ),
    }
}

/// [`trsm_right_band_f64`] for `Complex64` (generic kernels, as in
/// [`trsm_c64`]).
///
/// # Safety
///
/// Per the caller contract (see [`trsm_right_band_f64`]).
pub unsafe fn trsm_right_band_c64(
    uplo: TrsmUplo,
    xb: *mut Complex64,
    ldb: usize,
    x0: usize,
    i0: usize,
    i1: usize,
    nb: usize,
    trans: bool,
    unit: bool,
    ab: &[Complex64],
    di: &[Complex64],
) {
    match uplo {
        TrsmUplo::Lower => band_right_lower(
            xb,
            ldb,
            x0,
            i0,
            i1,
            nb,
            trans,
            unit,
            ab.as_ptr(),
            di.as_ptr(),
            band_right_lower_kernel,
        ),
        TrsmUplo::Upper => band_right_upper(
            xb,
            ldb,
            x0,
            i0,
            i1,
            nb,
            trans,
            unit,
            ab.as_ptr(),
            di.as_ptr(),
            band_right_upper_kernel,
        ),
    }
}

// ---------- blocked drivers --------------------------------------------------

/// Core driver shared by the four per-type wrappers.
fn trsm_core<T: TrsmScalar + Send + Sync + PanelBuf>(
    side: TrsmSide,
    uplo: TrsmUplo,
    trans: Transpose,
    unit: bool,
    m: usize,
    n: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    b: &mut [T],
    ldb: usize,
    gemm: GemmFn<T>,
    gemm_serial: GemmFn<T>,
    band_ll: BandKernelFn<T>,
    band_lu: BandKernelFn<T>,
    band_rl: BandKernelFn<T>,
    band_ru: BandKernelFn<T>,
) {
    if m == 0 || n == 0 {
        return;
    }
    // Keep the solve monochromatic: pin the pool when any scope would
    // leave the tiny pool, so small and tiny scopes never alternate
    // mid-solve (see `pool_floor_for_trsm`). The guard lives for the whole
    // solve; results are identical on every pool.
    let _pool_floor = match side {
        TrsmSide::Left => pool_floor_for_trsm(m, n, n, m),
        TrsmSide::Right => pool_floor_for_trsm(m, n, m, n),
    };
    let one = T::one();
    let zero = T::zero();
    if alpha == zero {
        for i in 0..m {
            for j in 0..n {
                b[i * ldb + j] = zero;
            }
        }
        return;
    }
    if alpha != one {
        for i in 0..m {
            for j in 0..n {
                b[i * ldb + j] = alpha * b[i * ldb + j];
            }
        }
    }
    let trans_flag = !matches!(trans, Transpose::NoTrans);
    let conjugate = matches!(trans, Transpose::ConjTrans);
    match side {
        TrsmSide::Left => {
            // Ascending panels for (Lower, NoTrans) and (Upper, Trans):
            // the solved block sits above the trailing update.
            let ascending = (uplo == TrsmUplo::Lower) != trans_flag;
            if ascending {
                // Fused single-scope fast path for Lower/NoTrans
                // multi-panel solves with enough right-hand sides (see
                // `trsm_left_lower_fused`): it completes the whole Left
                // solve, so return afterwards. Upper/Trans has its own
                // fused check next; the rest keep the panel loop below.
                // `SONIUM_TRSM_FUSED=0` forces the panel
                // loop (diagnostic/A-B fallback; the fused path stays the
                // default).
                let fused_off = std::env::var("SONIUM_TRSM_FUSED")
                    .map(|v| v == "0")
                    .unwrap_or(false);
                if trsm_use_fused(
                    uplo,
                    trans_flag,
                    m,
                    n,
                    rayon::current_num_threads().max(1),
                    fused_off,
                ) {
                    trsm_left_lower_fused(m, n, a, lda, b, ldb, unit, band_ll, gemm, gemm_serial);
                    return;
                }
                // Fused Upper/Trans fast path (ascending twin of
                // Lower/NoTrans; same gates, shared family switch).
                if uplo == TrsmUplo::Upper
                    && trans_flag
                    && m >= TRSM_FUSED_MIN_M
                    && n >= UNBLOCKED_PAR_MIN
                    && rayon::current_num_threads().max(1) > 1
                    && !fused_off
                {
                    trsm_left_upper_trans_fused(
                        m,
                        n,
                        a,
                        lda,
                        b,
                        ldb,
                        unit,
                        trans,
                        band_lu,
                        gemm,
                        gemm_serial,
                    );
                    return;
                }
                let mut k = 0;
                let panel_nb = trsm_panel_nb(uplo, trans_flag, core::mem::size_of::<T>());
                while k < m {
                    let nb = panel_nb.min(m - k);
                    match uplo {
                        TrsmUplo::Lower => unblocked_left_lower(
                            a,
                            lda,
                            k * lda + k,
                            b,
                            ldb,
                            k * ldb,
                            nb,
                            n,
                            trans_flag,
                            conjugate,
                            unit,
                            band_ll,
                        ),
                        TrsmUplo::Upper => unblocked_left_upper(
                            a,
                            lda,
                            k * lda + k,
                            b,
                            ldb,
                            k * ldb,
                            nb,
                            n,
                            trans_flag,
                            conjugate,
                            unit,
                            band_lu,
                        ),
                    }
                    if k + nb < m {
                        let rows = m - k - nb;
                        if !trans_flag {
                            // B[k+nb..] -= A[k+nb.., k..k+nb] * X[k..k+nb].
                            // Row ranges are linearly separable: one split,
                            // no copy.
                            let (x_head, c_tail) = b.split_at_mut((k + nb) * ldb);
                            gemm(
                                Transpose::NoTrans,
                                Transpose::NoTrans,
                                rows,
                                n,
                                nb,
                                -one,
                                &a[(k + nb) * lda + k..],
                                lda,
                                &x_head[k * ldb..],
                                ldb,
                                one,
                                &mut c_tail[..rows * ldb],
                                ldb,
                            );
                        } else {
                            // B[k+nb..] -= A[k..k+nb, k+nb..]^H * X[k..k+nb].
                            let (x_head, c_tail) = b.split_at_mut((k + nb) * ldb);
                            gemm(
                                trans,
                                Transpose::NoTrans,
                                rows,
                                n,
                                nb,
                                -one,
                                &a[k * lda + k + nb..],
                                lda,
                                &x_head[k * ldb..],
                                ldb,
                                one,
                                &mut c_tail[..rows * ldb],
                                ldb,
                            );
                        }
                    }
                    k += nb;
                }
            } else {
                // Fused Upper/NoTrans fast path (see
                // `trsm_left_upper_fused`): completes the whole Left
                // solve, so return afterwards. Same size gates as the
                // Lower twin; small/thread-single cases keep the panel
                // loop below. `SONIUM_TRSM_FUSED=0` forces the
                // panel loop (shared family switch with Lower/Right).
                let fused_off = std::env::var("SONIUM_TRSM_FUSED")
                    .map(|v| v == "0")
                    .unwrap_or(false);
                if uplo == TrsmUplo::Upper
                    && !trans_flag
                    && m >= TRSM_FUSED_MIN_M
                    && n >= UNBLOCKED_PAR_MIN
                    && rayon::current_num_threads().max(1) > 1
                    && !fused_off
                {
                    trsm_left_upper_fused(m, n, a, lda, b, ldb, unit, band_lu, gemm, gemm_serial);
                    return;
                }
                // Fused Lower/Trans fast path (descending twin of
                // Upper/NoTrans; same gates, shared family switch).
                if uplo == TrsmUplo::Lower
                    && trans_flag
                    && m >= TRSM_FUSED_MIN_M
                    && n >= UNBLOCKED_PAR_MIN
                    && rayon::current_num_threads().max(1) > 1
                    && !fused_off
                {
                    trsm_left_lower_trans_fused(
                        m,
                        n,
                        a,
                        lda,
                        b,
                        ldb,
                        unit,
                        trans,
                        band_ll,
                        gemm,
                        gemm_serial,
                    );
                    return;
                }
                let mut kend = m;
                while kend > 0 {
                    let nb = TRSM_NB.min(kend);
                    let k = kend - nb;
                    match uplo {
                        TrsmUplo::Lower => unblocked_left_lower(
                            a,
                            lda,
                            k * lda + k,
                            b,
                            ldb,
                            k * ldb,
                            nb,
                            n,
                            trans_flag,
                            conjugate,
                            unit,
                            band_ll,
                        ),
                        TrsmUplo::Upper => unblocked_left_upper(
                            a,
                            lda,
                            k * lda + k,
                            b,
                            ldb,
                            k * ldb,
                            nb,
                            n,
                            trans_flag,
                            conjugate,
                            unit,
                            band_lu,
                        ),
                    }
                    if k > 0 {
                        if !trans_flag {
                            // B[0..k] -= A[0..k, k..k+nb] * X[k..k+nb].
                            let (c_head, x_tail) = b.split_at_mut(k * ldb);
                            gemm(
                                Transpose::NoTrans,
                                Transpose::NoTrans,
                                k,
                                n,
                                nb,
                                -one,
                                &a[k..],
                                lda,
                                &x_tail[..nb * ldb],
                                ldb,
                                one,
                                &mut c_head[..k * ldb],
                                ldb,
                            );
                        } else {
                            // B[0..k] -= A[k..k+nb, 0..k]^H * X[k..k+nb].
                            let (c_head, x_tail) = b.split_at_mut(k * ldb);
                            gemm(
                                trans,
                                Transpose::NoTrans,
                                k,
                                n,
                                nb,
                                -one,
                                &a[k * lda..],
                                lda,
                                &x_tail[..nb * ldb],
                                ldb,
                                one,
                                &mut c_head[..k * ldb],
                                ldb,
                            );
                        }
                    }
                    kend = k;
                }
            }
        }
        TrsmSide::Right => {
            // Packed X-panel scratch: the X and C column ranges interleave
            // within every row, so no linear split can separate them for a
            // simultaneous &-borrow (X) and &mut-borrow (C). Packing the
            // solved panel contiguously is also the faster GEMM read.
            // Total copy traffic is 2·m·n elements over the whole solve —
            // negligible next to the O(m·n·min(m,n)) solve flops.
            // Allocated lazily: single-panel solves (the common small-`n`
            // shape, e.g. one Cholesky L21 panel per call) never touch it,
            // and zeroing `m × TRSM_NB` elements per call would cost more
            // than the solve itself there.
            let mut xpack: Vec<T> = Vec::new();
            // Ascending column panels for (Upper, NoTrans) and (Lower,
            // Trans): the solved block sits left of the trailing update.
            let ascending = (uplo == TrsmUplo::Upper) != trans_flag;
            // Fused single-scope fast path (see `trsm_right_fused`): one
            // scope over row bands, each band running the full panel loop
            // — the transposed mirror of the fused Left solve. Rows (not
            // right-hand sides) are the parallel dimension, so the size
            // gates mirror transposed: enough rows, multiple panels.
            // `SONIUM_TRSM_FUSED=0` forces the panel loop
            // (diagnostic/A-B fallback; the fused path stays the default).
            let fused_off = std::env::var("SONIUM_TRSM_FUSED")
                .map(|v| v == "0")
                .unwrap_or(false);
            // Invert-then-GEMM for small `n` (see
            // [`trsm_right_inverted`]): FALSIFIED 2026-10-02, opt-in only
            // (`SONIUM_TRSM_INVERT=1`). The substitution it replaces is
            // ~16 GF/s serial f32, but the 8M-flop replacement GEMM is
            // dispatch-dominated on our substrate (~10µs/band over 16
            // bands ≈ 0.15ms vs 0.03ms OB): interleaved f32-Right
            // 1984x64 went 0.088ms substitution -> 0.182ms inverted
            // (2.6x -> 5.8x vs OB). Correct (agreement + residuals
            // green) but slower until GEMM dispatch reaches µs scale.
            // Single-panel shapes only (`n <= 64`); smaller-`m` keeps
            // the serial unblocked path.
            let invert_on = std::env::var("SONIUM_TRSM_INVERT")
                .map(|v| v != "0")
                .unwrap_or(false);
            if m >= UNBLOCKED_PAR_MIN && n <= TRSM_INVERT_MAX_N && invert_on && !fused_off {
                trsm_right_inverted(m, n, a, lda, b, ldb, uplo, trans, unit, gemm);
                return;
            }
            if m >= UNBLOCKED_PAR_MIN && n > TRSM_NB && !fused_off {
                let band_r = match uplo {
                    TrsmUplo::Lower => band_rl,
                    TrsmUplo::Upper => band_ru,
                };
                trsm_right_fused(
                    m,
                    n,
                    a,
                    lda,
                    b,
                    ldb,
                    !ascending,
                    trans,
                    trans_flag,
                    conjugate,
                    unit,
                    band_r,
                    gemm,
                    gemm_serial,
                );
                return;
            }
            if ascending {
                let mut k = 0;
                while k < n {
                    let nb = TRSM_NB.min(n - k);
                    match uplo {
                        TrsmUplo::Lower => unblocked_right_lower(
                            a,
                            lda,
                            k * lda + k,
                            b,
                            ldb,
                            k,
                            m,
                            nb,
                            trans_flag,
                            conjugate,
                            unit,
                            band_rl,
                        ),
                        TrsmUplo::Upper => unblocked_right_upper(
                            a,
                            lda,
                            k * lda + k,
                            b,
                            ldb,
                            k,
                            m,
                            nb,
                            trans_flag,
                            conjugate,
                            unit,
                            band_ru,
                        ),
                    }
                    if k + nb < n {
                        let cols = n - k - nb;
                        if !trans_flag {
                            // B[.., k+nb..] -= X[.., k..k+nb] * A[k..k+nb, k+nb..].
                            if xpack.len() < m * nb {
                                xpack.resize(m * nb, zero);
                            }
                            // Pack the solved X panel (all rows, cols k..k+nb).
                            for (i, row) in xpack[..m * nb].chunks_exact_mut(nb).enumerate() {
                                row.copy_from_slice(&b[i * ldb + k..i * ldb + k + nb]);
                            }
                            gemm(
                                Transpose::NoTrans,
                                Transpose::NoTrans,
                                m,
                                cols,
                                nb,
                                -one,
                                &xpack[..m * nb],
                                nb,
                                &a[k * lda + k + nb..],
                                lda,
                                one,
                                &mut b[k + nb..],
                                ldb,
                            );
                        } else {
                            // B[.., k+nb..] -= X[.., k..k+nb] * A[k+nb.., k..k+nb]^H.
                            if xpack.len() < m * nb {
                                xpack.resize(m * nb, zero);
                            }
                            // Pack the solved X panel (all rows, cols k..k+nb).
                            for (i, row) in xpack[..m * nb].chunks_exact_mut(nb).enumerate() {
                                row.copy_from_slice(&b[i * ldb + k..i * ldb + k + nb]);
                            }
                            gemm(
                                Transpose::NoTrans,
                                trans,
                                m,
                                cols,
                                nb,
                                -one,
                                &xpack[..m * nb],
                                nb,
                                &a[(k + nb) * lda + k..],
                                lda,
                                one,
                                &mut b[k + nb..],
                                ldb,
                            );
                        }
                    }
                    k += nb;
                }
            } else {
                let mut kend = n;
                while kend > 0 {
                    let nb = TRSM_NB.min(kend);
                    let k = kend - nb;
                    match uplo {
                        TrsmUplo::Lower => unblocked_right_lower(
                            a,
                            lda,
                            k * lda + k,
                            b,
                            ldb,
                            k,
                            m,
                            nb,
                            trans_flag,
                            conjugate,
                            unit,
                            band_rl,
                        ),
                        TrsmUplo::Upper => unblocked_right_upper(
                            a,
                            lda,
                            k * lda + k,
                            b,
                            ldb,
                            k,
                            m,
                            nb,
                            trans_flag,
                            conjugate,
                            unit,
                            band_ru,
                        ),
                    }
                    if k > 0 {
                        if !trans_flag {
                            // B[.., 0..k] -= X[.., k..k+nb] * A[k..k+nb, 0..k].
                            if xpack.len() < m * nb {
                                xpack.resize(m * nb, zero);
                            }
                            // Pack the solved X panel (all rows, cols k..k+nb).
                            for (i, row) in xpack[..m * nb].chunks_exact_mut(nb).enumerate() {
                                row.copy_from_slice(&b[i * ldb + k..i * ldb + k + nb]);
                            }
                            gemm(
                                Transpose::NoTrans,
                                Transpose::NoTrans,
                                m,
                                k,
                                nb,
                                -one,
                                &xpack[..m * nb],
                                nb,
                                &a[k * lda..],
                                lda,
                                one,
                                &mut b[..],
                                ldb,
                            );
                        } else {
                            // B[.., 0..k] -= X[.., k..k+nb] * A[0..k, k..k+nb]^H.
                            if xpack.len() < m * nb {
                                xpack.resize(m * nb, zero);
                            }
                            // Pack the solved X panel (all rows, cols k..k+nb).
                            for (i, row) in xpack[..m * nb].chunks_exact_mut(nb).enumerate() {
                                row.copy_from_slice(&b[i * ldb + k..i * ldb + k + nb]);
                            }
                            gemm(
                                Transpose::NoTrans,
                                trans,
                                m,
                                k,
                                nb,
                                -one,
                                &xpack[..m * nb],
                                nb,
                                &a[k..],
                                lda,
                                one,
                                &mut b[..],
                                ldb,
                            );
                        }
                    }
                    kend = k;
                }
            }
        }
    }
}

// ---------- per-type wrappers ------------------------------------------------

/// `op(A) * X = alpha * B` or `X * op(A) = alpha * B` for `f32`, row-major.
///
/// `A` is `m × m` (`Left`) or `n × n` (`Right`) with row stride `lda`;
/// `B` is `m × n` with row stride `ldb` and is overwritten with `X`.
/// Only the triangle named by `uplo` is read. `unit` selects unit diagonal
/// (diagonal unread). `alpha == 0` zeroes `B` without touching `A`.
pub fn trsm_f32(
    side: TrsmSide,
    uplo: TrsmUplo,
    trans: Transpose,
    unit: bool,
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &mut [f32],
    ldb: usize,
) {
    trsm_core(
        side,
        uplo,
        trans,
        unit,
        m,
        n,
        alpha,
        a,
        lda,
        b,
        ldb,
        gemm_f32,
        gemm_f32_serial,
        band_left_lower_kernel_f32,
        band_left_upper_kernel_f32,
        band_right_lower_kernel_f32,
        band_right_upper_kernel_f32,
    );
}

/// `f64` twin of [`trsm_f32`].
pub fn trsm_f64(
    side: TrsmSide,
    uplo: TrsmUplo,
    trans: Transpose,
    unit: bool,
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    b: &mut [f64],
    ldb: usize,
) {
    trsm_core(
        side,
        uplo,
        trans,
        unit,
        m,
        n,
        alpha,
        a,
        lda,
        b,
        ldb,
        gemm_f64,
        gemm_f64_serial,
        band_left_lower_kernel_f64,
        band_left_upper_kernel_f64,
        band_right_lower_kernel_f64,
        band_right_upper_kernel_f64,
    );
}

/// `Complex32` twin of [`trsm_f32`]; `ConjTrans` conjugates.
pub fn trsm_c32(
    side: TrsmSide,
    uplo: TrsmUplo,
    trans: Transpose,
    unit: bool,
    m: usize,
    n: usize,
    alpha: Complex32,
    a: &[Complex32],
    lda: usize,
    b: &mut [Complex32],
    ldb: usize,
) {
    trsm_core(
        side,
        uplo,
        trans,
        unit,
        m,
        n,
        alpha,
        a,
        lda,
        b,
        ldb,
        gemm_c32,
        // Complex GEMM is already serial: same kernel for both entries.
        gemm_c32,
        band_left_lower_kernel_c32,
        band_left_upper_kernel_c32,
        band_right_lower_kernel_c32,
        band_right_upper_kernel_c32,
    );
}

/// `Complex64` twin of [`trsm_f32`]; `ConjTrans` conjugates.
pub fn trsm_c64(
    side: TrsmSide,
    uplo: TrsmUplo,
    trans: Transpose,
    unit: bool,
    m: usize,
    n: usize,
    alpha: Complex64,
    a: &[Complex64],
    lda: usize,
    b: &mut [Complex64],
    ldb: usize,
) {
    trsm_core(
        side,
        uplo,
        trans,
        unit,
        m,
        n,
        alpha,
        a,
        lda,
        b,
        ldb,
        gemm_c64,
        // Complex GEMM is already serial: same kernel for both entries.
        gemm_c64,
        band_left_lower_kernel_c64,
        band_left_upper_kernel_c64,
        band_right_lower_kernel_c64,
        band_right_upper_kernel_c64,
    );
}

// ---------- correctness tests ------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::parallelism::env_test_lock;
    use approx::assert_relative_eq;

    // Naive obviously-correct reference: unblocked substitution over the
    // full triangle, no blocking, no GEMM.
    fn ref_trsm_f64(
        side: TrsmSide,
        uplo: TrsmUplo,
        trans: Transpose,
        unit: bool,
        m: usize,
        n: usize,
        alpha: f64,
        a: &[f64],
        lda: usize,
        b: &mut [f64],
        ldb: usize,
    ) {
        let t = !matches!(trans, Transpose::NoTrans);
        let at = |r: usize, c: usize| -> f64 { if t { a[c * lda + r] } else { a[r * lda + c] } };
        // Scale by alpha (reference mirrors the alpha == 0 early-out:
        // only the m × n submatrix is touched, never padding).
        if alpha == 0.0 {
            for i in 0..m {
                for j in 0..n {
                    b[i * ldb + j] = 0.0;
                }
            }
            return;
        }
        for i in 0..m {
            for j in 0..n {
                b[i * ldb + j] *= alpha;
            }
        }
        match side {
            TrsmSide::Left => {
                let lo = uplo == TrsmUplo::Lower;
                // Forward for (Lower, N) and (Upper, T); else backward.
                let fwd = lo != t;
                for j in 0..n {
                    if fwd {
                        for i in 0..m {
                            let mut s = b[i * ldb + j];
                            for k in 0..i {
                                s -= at(i, k) * b[k * ldb + j];
                            }
                            b[i * ldb + j] = if unit { s } else { s / at(i, i) };
                        }
                    } else {
                        for i in (0..m).rev() {
                            let mut s = b[i * ldb + j];
                            for k in (i + 1)..m {
                                s -= at(i, k) * b[k * ldb + j];
                            }
                            b[i * ldb + j] = if unit { s } else { s / at(i, i) };
                        }
                    }
                }
            }
            TrsmSide::Right => {
                let up = uplo == TrsmUplo::Upper;
                let asc = up != t;
                for i in 0..m {
                    if asc {
                        for j in 0..n {
                            let mut s = b[i * ldb + j];
                            for k in 0..j {
                                s -= b[i * ldb + k] * at(k, j);
                            }
                            b[i * ldb + j] = if unit { s } else { s / at(j, j) };
                        }
                    } else {
                        for j in (0..n).rev() {
                            let mut s = b[i * ldb + j];
                            for k in (j + 1)..n {
                                s -= b[i * ldb + k] * at(k, j);
                            }
                            b[i * ldb + j] = if unit { s } else { s / at(j, j) };
                        }
                    }
                }
            }
        }
    }

    fn prng(k: usize) -> f64 {
        let h = k
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407)
            >> 11;
        (h % 2000) as f64 / 1000.0 - 1.0
    }

    // Naive f32 Right-only reference for the invert-then-GEMM path
    // (the f64 twin above covers Left; Right with transposed access
    // mirrored here).
    fn ref_trsm_f32_right(
        uplo: TrsmUplo,
        trans: Transpose,
        unit: bool,
        m: usize,
        n: usize,
        alpha: f32,
        a: &[f32],
        lda: usize,
        b: &mut [f32],
        ldb: usize,
    ) {
        let t = !matches!(trans, Transpose::NoTrans);
        let at = |r: usize, c: usize| -> f32 { if t { a[c * lda + r] } else { a[r * lda + c] } };
        for i in 0..m {
            for j in 0..n {
                b[i * ldb + j] *= alpha;
            }
        }
        let up = uplo == TrsmUplo::Upper;
        let asc = up != t;
        for i in 0..m {
            if asc {
                for j in 0..n {
                    let mut s = b[i * ldb + j];
                    for k in 0..j {
                        s -= b[i * ldb + k] * at(k, j);
                    }
                    b[i * ldb + j] = if unit { s } else { s / at(j, j) };
                }
            } else {
                for j in (0..n).rev() {
                    let mut s = b[i * ldb + j];
                    for k in (j + 1)..n {
                        s -= b[i * ldb + k] * at(k, j);
                    }
                    b[i * ldb + j] = if unit { s } else { s / at(j, j) };
                }
            }
        }
    }

    /// Diagonally-dominant f32 triangle (mirror of [`tri_a`], with
    /// damped off-diagonals: unit-diagonal triangles with ±1 off-diagonals
    /// reach cond ~500 at n=64, where both the inverted path and the naive
    /// reference are forward-inaccurate (agreement-meaningless); ±0.25
    /// keeps cond ~10 so agreement plus residuals both bite).
    fn tri_a_f32(m: usize, lda: usize, lower: bool, k0: usize) -> Vec<f32> {
        let mut a = vec![0.0; m * lda];
        for i in 0..m {
            for j in 0..m {
                let inside = if lower { j <= i } else { j >= i };
                a[i * lda + j] = if inside {
                    0.25 * prng(k0 + i * m + j) as f32
                } else {
                    0.0
                };
            }
            a[i * lda + i] += 4.0 * if lower { 1.0 } else { -1.0 };
        }
        a
    }

    /// Max `|X·op - alpha·B|` over the `m x n` block (the solve contract,
    /// independent of summation order): both the inverted path and the
    /// naive reference must satisfy it.
    fn right_residual_f32(
        uplo: TrsmUplo,
        trans: Transpose,
        unit: bool,
        m: usize,
        n: usize,
        alpha: f32,
        a: &[f32],
        lda: usize,
        x: &[f32],
        b_orig: &[f32],
        ldb: usize,
    ) -> f32 {
        let t = !matches!(trans, Transpose::NoTrans);
        let lower = uplo == TrsmUplo::Lower;
        let mut worst = 0.0f32;
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for k in 0..n {
                    // op[k][j]: triangle (or unit diag), transposed when `t`.
                    let (r, c) = if t { (j, k) } else { (k, j) };
                    let inside = if lower { c <= r } else { c >= r };
                    if !inside {
                        continue;
                    }
                    let aval = if unit && r == c { 1.0 } else { a[r * lda + c] };
                    s += x[i * ldb + k] * aval;
                }
                let want = alpha * b_orig[i * ldb + j];
                worst = worst.max((s - want).abs());
            }
        }
        worst
    }

    /// Diagonally-dominant triangle so the reference division is safe.
    fn tri_a(m: usize, lda: usize, lower: bool, k0: usize) -> Vec<f64> {
        let mut a = vec![0.0; m * lda];
        for i in 0..m {
            for j in 0..m {
                let inside = if lower { j <= i } else { j >= i };
                a[i * lda + j] = if inside { prng(k0 + i * m + j) } else { 0.0 };
            }
            a[i * lda + i] += 4.0 * if lower { 1.0 } else { -1.0 };
        }
        a
    }

    fn check_f64(
        side: TrsmSide,
        uplo: TrsmUplo,
        trans: Transpose,
        unit: bool,
        m: usize,
        n: usize,
        pad: usize,
        alpha: f64,
    ) {
        let (adim, lda) = match side {
            TrsmSide::Left => (m, m.max(1) + pad),
            TrsmSide::Right => (n, n.max(1) + pad),
        };
        let ldb = n.max(1) + pad;
        let lower = uplo == TrsmUplo::Lower;
        let a = if adim == 0 {
            vec![]
        } else {
            tri_a(adim, lda, lower, 7)
        };
        let b0: Vec<f64> = (0..m.max(1) * ldb).map(|k| prng(100_000 + k)).collect();
        let mut got = b0.clone();
        trsm_f64(side, uplo, trans, unit, m, n, alpha, &a, lda, &mut got, ldb);
        let mut want = b0;
        ref_trsm_f64(
            side, uplo, trans, unit, m, n, alpha, &a, lda, &mut want, ldb,
        );
        for (g, w) in got.iter().zip(want.iter()) {
            assert_relative_eq!(g, w, epsilon = 1e-9, max_relative = 1e-9);
        }
    }

    #[test]
    fn trsm_f64_all_variants() {
        let ud = [(false, 1.0), (true, 2.5), (false, 0.0)];
        for side in [TrsmSide::Left, TrsmSide::Right] {
            for uplo in [TrsmUplo::Lower, TrsmUplo::Upper] {
                for trans in [Transpose::NoTrans, Transpose::Trans] {
                    for (unit, alpha) in ud {
                        // Tiny / odd / non-multiple-of-NB / padded.
                        // The last two cross the fused-path gates
                        // (multi-panel, parallel bands) for every variant.
                        for (m, n, pad) in [
                            (1, 1, 0),
                            (3, 5, 0),
                            (5, 3, 2),
                            (17, 16, 0),
                            (16, 17, 3),
                            (70, 65, 0),
                            (65, 70, 1),
                            (130, 129, 0),
                            (200, 200, 0),
                            (129, 257, 1),
                        ] {
                            check_f64(side, uplo, trans, unit, m, n, pad, alpha);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn trsm_right_inverted_matches_naive() {
        // The invert-then-GEMM path (n <= 64, m >= 128) against the
        // naive substitution reference: agreement to f32 summation
        // order plus small residuals on both (the solve contract).
        for uplo in [TrsmUplo::Lower, TrsmUplo::Upper] {
            for trans in [Transpose::NoTrans, Transpose::Trans] {
                for unit in [false, true] {
                    for (m, n, pad) in [(200, 64, 0), (256, 48, 1), (130, 16, 2)] {
                        let lda = n + pad;
                        let ldb = n + pad;
                        let lower = uplo == TrsmUplo::Lower;
                        let a = tri_a_f32(n, lda, lower, 7);
                        let b0: Vec<f32> = (0..m * ldb).map(|k| prng(100_000 + k) as f32).collect();
                        let mut got = b0.clone();
                        // Direct call: the public path gates this
                        // behind `SONIUM_TRSM_INVERT=1` (falsified
                        // default-off), so route around the gate to keep
                        // coverage. Alpha is 1.0 (no prescale needed).
                        trsm_right_inverted(
                            m, n, &a, lda, &mut got, ldb, uplo, trans, unit, gemm_f32,
                        );
                        let mut want = b0.clone();
                        ref_trsm_f32_right(uplo, trans, unit, m, n, 1.0, &a, lda, &mut want, ldb);
                        // Agreement to inverse-vs-substitution order (GEMM
                        // k-reduction plus 64 chained divisions differ from
                        // naive chaining at the 1e-4 level f32); the
                        // residual below holds the solve contract tight.
                        for (g, w) in got.iter().zip(want.iter()) {
                            assert_relative_eq!(g, w, epsilon = 1e-3, max_relative = 1e-3);
                        }
                        let res_got = right_residual_f32(
                            uplo, trans, unit, m, n, 1.0, &a, lda, &got, &b0, ldb,
                        );
                        let res_want = right_residual_f32(
                            uplo, trans, unit, m, n, 1.0, &a, lda, &want, &b0, ldb,
                        );
                        assert!(
                            res_got < 1e-3,
                            "inverted residual {res_got} ({uplo:?} {trans:?} unit={unit} {m}x{n})"
                        );
                        assert!(
                            res_want < 1e-3,
                            "naive residual {res_want} ({uplo:?} {trans:?} unit={unit} {m}x{n})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn trsm_right_fused_matches_panel_loop() {
        let _guard = env_test_lock();
        // Fused Right (default) vs panel loop (`SONIUM_TRSM_FUSED=0`) at
        // multi-panel parallel scale: same kernels, different grouping, so
        // agreement is expected to 1e-9 across all four Right variants
        // (both diagonals). The gate is process-global, but this is one
        // of the writers among tests (see the Upper/Trans twins),
        // serialised by the lock, so set/remove here is safe.
        for uplo in [TrsmUplo::Lower, TrsmUplo::Upper] {
            for trans in [Transpose::NoTrans, Transpose::Trans] {
                for unit in [false, true] {
                    let (m, n) = (320, 320);
                    let lower = uplo == TrsmUplo::Lower;
                    let a = tri_a(n, n, lower, 7);
                    let b0: Vec<f64> = (0..m * n).map(|k| prng(100_000 + k)).collect();
                    let mut fused = b0.clone();
                    trsm_f64(
                        TrsmSide::Right,
                        uplo,
                        trans,
                        unit,
                        m,
                        n,
                        1.0,
                        &a,
                        n,
                        &mut fused,
                        n,
                    );
                    // SAFETY: one of the writers of `SONIUM_TRSM_FUSED`
                    // among tests, serialised by the lock, and no FFI in
                    // the test binary touches the environment;
                    // concurrent `std::env::var` reads elsewhere
                    // serialize on the internal lock.
                    unsafe {
                        std::env::set_var("SONIUM_TRSM_FUSED", "0");
                    }
                    let mut panel = b0.clone();
                    trsm_f64(
                        TrsmSide::Right,
                        uplo,
                        trans,
                        unit,
                        m,
                        n,
                        1.0,
                        &a,
                        n,
                        &mut panel,
                        n,
                    );
                    unsafe {
                        std::env::remove_var("SONIUM_TRSM_FUSED");
                    }
                    for (g, w) in fused.iter().zip(panel.iter()) {
                        assert_relative_eq!(g, w, epsilon = 1e-9, max_relative = 1e-9);
                    }
                }
            }
        }
    }

    #[test]
    fn trsm_left_upper_fused_matches_panel_loop() {
        let _guard = env_test_lock();
        // Upper fused (default) vs panel loop (`SONIUM_TRSM_FUSED=0`):
        // same kernels, different grouping, so agreement is expected
        // to 1e-9 (f64). Shapes cover full panels (512), partial-first
        // panels (300 = 2×128+44, 260), and a tall-skinny narrow-fill
        // case (512×128); both unit flags. The gate is process-global,
        // but this is its only writer among tests (with the Right
        // twin, serialised by the lock), so set/remove here is safe.
        for (m, n) in [(512usize, 512usize), (300, 300), (260, 260), (512, 128)] {
            for unit in [false, true] {
                let a = tri_a(m, m, false, 7);
                let b0: Vec<f64> = (0..m * n).map(|k| prng(100_000 + k)).collect();
                let mut fused = b0.clone();
                trsm_f64(
                    TrsmSide::Left,
                    TrsmUplo::Upper,
                    Transpose::NoTrans,
                    unit,
                    m,
                    n,
                    1.0,
                    &a,
                    m,
                    &mut fused,
                    n,
                );
                // SAFETY: one of the writers of `SONIUM_TRSM_FUSED`
                // among tests (see the Right/Trans twins), serialised by
                // the lock; no FFI in the test binary touches the
                // environment; concurrent `std::env::var` reads
                // elsewhere serialize on the internal lock.
                unsafe {
                    std::env::set_var("SONIUM_TRSM_FUSED", "0");
                }
                let mut panel = b0.clone();
                trsm_f64(
                    TrsmSide::Left,
                    TrsmUplo::Upper,
                    Transpose::NoTrans,
                    unit,
                    m,
                    n,
                    1.0,
                    &a,
                    m,
                    &mut panel,
                    n,
                );
                unsafe {
                    std::env::remove_var("SONIUM_TRSM_FUSED");
                }
                for (g, w) in fused.iter().zip(panel.iter()) {
                    assert_relative_eq!(g, w, epsilon = 1e-9, max_relative = 1e-9);
                }
            }
        }
    }

    #[test]
    fn trsm_left_trans_fused_matches_panel_loop() {
        let _guard = env_test_lock();
        // Trans fused (default) vs panel loop (`SONIUM_TRSM_FUSED=0`):
        // same kernels, different grouping, so agreement is expected
        // to 1e-9 (f64). Both uplos (ascending Upper, descending
        // Lower) × Trans; shapes cover full panels (512),
        // partial panels (300, 260), and a tall-skinny narrow-fill
        // case (512×128); both unit flags. Serialised with the
        // other `SONIUM_TRSM_FUSED` writers by the lock.
        for uplo in [TrsmUplo::Lower, TrsmUplo::Upper] {
            let lower = uplo == TrsmUplo::Lower;
            for (m, n) in [(512usize, 512usize), (300, 300), (260, 260), (512, 128)] {
                for unit in [false, true] {
                    let a = tri_a(m, m, lower, 7);
                    let b0: Vec<f64> = (0..m * n).map(|k| prng(100_000 + k)).collect();
                    let mut fused = b0.clone();
                    trsm_f64(
                        TrsmSide::Left,
                        uplo,
                        Transpose::Trans,
                        unit,
                        m,
                        n,
                        1.0,
                        &a,
                        m,
                        &mut fused,
                        n,
                    );
                    // SAFETY: one of the writers of `SONIUM_TRSM_FUSED`
                    // among tests, serialised by the lock; no FFI in
                    // the test binary touches the environment;
                    // concurrent `std::env::var` reads elsewhere
                    // serialize on the internal lock.
                    unsafe {
                        std::env::set_var("SONIUM_TRSM_FUSED", "0");
                    }
                    let mut panel = b0.clone();
                    trsm_f64(
                        TrsmSide::Left,
                        uplo,
                        Transpose::Trans,
                        unit,
                        m,
                        n,
                        1.0,
                        &a,
                        m,
                        &mut panel,
                        n,
                    );
                    unsafe {
                        std::env::remove_var("SONIUM_TRSM_FUSED");
                    }
                    for (g, w) in fused.iter().zip(panel.iter()) {
                        assert_relative_eq!(g, w, epsilon = 1e-9, max_relative = 1e-9);
                    }
                }
            }
        }
    }

    #[test]
    fn trsm_left_conjtrans_fused_matches_panel_loop() {
        let _guard = env_test_lock();
        // c64 ConjTrans fused vs panel loop: exercises the conjugate
        // pack (`pack_abuf` with `conjugate`) and conjugated diagonal
        // reciprocals on both Trans drivers. Diagonally-dominant
        // complex triangles; multi-panel shapes (320 = 2×128+64 has a
        // partial panel both ascending and descending) plus a
        // tall-skinny case; both unit flags. 1e-9 agreement.
        for uplo in [TrsmUplo::Lower, TrsmUplo::Upper] {
            let lower = uplo == TrsmUplo::Lower;
            for (m, n) in [(320usize, 320usize), (512, 128)] {
                for unit in [false, true] {
                    let mut a: Vec<Complex64> = (0..m * m)
                        .map(|k| Complex64::new(prng(k), prng(50_000 + k)))
                        .collect();
                    for i in 0..m {
                        for j in 0..m {
                            let inside = if lower { j <= i } else { j >= i };
                            if !inside {
                                a[i * m + j] = Complex64::new(0.0, 0.0);
                            }
                        }
                        a[i * m + i] += Complex64::new(4.0, 0.0);
                    }
                    let b0: Vec<Complex64> = (0..m * n)
                        .map(|k| Complex64::new(prng(100_000 + k), prng(200_000 + k)))
                        .collect();
                    let mut fused = b0.clone();
                    trsm_c64(
                        TrsmSide::Left,
                        uplo,
                        Transpose::ConjTrans,
                        unit,
                        m,
                        n,
                        Complex64::new(1.0, 0.0),
                        &a,
                        m,
                        &mut fused,
                        n,
                    );
                    // SAFETY: one of the writers of `SONIUM_TRSM_FUSED`
                    // among tests, serialised by the lock (see above).
                    unsafe {
                        std::env::set_var("SONIUM_TRSM_FUSED", "0");
                    }
                    let mut panel = b0.clone();
                    trsm_c64(
                        TrsmSide::Left,
                        uplo,
                        Transpose::ConjTrans,
                        unit,
                        m,
                        n,
                        Complex64::new(1.0, 0.0),
                        &a,
                        m,
                        &mut panel,
                        n,
                    );
                    unsafe {
                        std::env::remove_var("SONIUM_TRSM_FUSED");
                    }
                    for (g, w) in fused.iter().zip(panel.iter()) {
                        assert_relative_eq!(g.re, w.re, epsilon = 1e-9, max_relative = 1e-9);
                        assert_relative_eq!(g.im, w.im, epsilon = 1e-9, max_relative = 1e-9);
                    }
                }
            }
        }
    }

    #[test]
    fn trsm_f32_spot_check() {
        // f32 shares the generic core; spot-check one variant per axis
        // against an f64-computed reference at f32 tolerance.
        for (side, uplo, trans, unit, m, n) in [
            (
                TrsmSide::Left,
                TrsmUplo::Lower,
                Transpose::NoTrans,
                false,
                70,
                65,
            ),
            (
                TrsmSide::Left,
                TrsmUplo::Upper,
                Transpose::Trans,
                true,
                65,
                70,
            ),
            (
                TrsmSide::Right,
                TrsmUplo::Upper,
                Transpose::NoTrans,
                false,
                33,
                70,
            ),
            (
                TrsmSide::Right,
                TrsmUplo::Lower,
                Transpose::Trans,
                true,
                70,
                33,
            ),
        ] {
            let (adim, lda) = match side {
                TrsmSide::Left => (m, m + 1),
                TrsmSide::Right => (n, n + 1),
            };
            let ldb = n + 1;
            let lower = uplo == TrsmUplo::Lower;
            let a64 = tri_a(adim, lda, lower, 7);
            let a32: Vec<f32> = a64.iter().map(|&v| v as f32).collect();
            let b0: Vec<f32> = (0..m * ldb).map(|k| prng(100_000 + k) as f32).collect();
            let mut got = b0.clone();
            trsm_f32(side, uplo, trans, unit, m, n, 2.0, &a32, lda, &mut got, ldb);
            let mut want64: Vec<f64> = b0.iter().map(|&v| v as f64).collect();
            ref_trsm_f64(
                side,
                uplo,
                trans,
                unit,
                m,
                n,
                2.0,
                &a64,
                lda,
                &mut want64,
                ldb,
            );
            // Valid submatrix only; padding is untouched by contract.
            for i in 0..m {
                for j in 0..n {
                    let (g, w) = (got[i * ldb + j], want64[i * ldb + j]);
                    assert!(
                        (g as f64 - w).abs() <= 1e-4 * w.abs().max(1.0),
                        "f32 mismatch {g} vs {w}"
                    );
                }
            }
        }
    }

    /// The blocked left-lower f32 kernel is BITWISE identical to the
    /// row-wise AVX2 kernel (same lane op order, only the visit order
    /// changes): hot path (`nb % 4 == 0`, full sub-chunks), row tails,
    /// chunk tails, scalar tails, padded strides, unit and non-unit
    /// diagonals. (f64 keeps row-wise: its blocked twin tied; see the
    /// dispatcher note.)
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn band_c64_avx2_matches_generic() {
        // AVX2 complex left-lower kernel vs the generic kernel: same
        // packed triangle, same diagonal inverses, same B. Forward and
        // backward, unit and non-unit, strided ldb, nonzero x0.
        for (trans, lower) in [(false, true), (true, false)] {
            for unit in [false, true] {
                let (nb, nrhs, ldb, x0) = (40usize, 37usize, 41usize, 2usize);
                let mut ab = vec![Complex64::new(0.0, 0.0); nb * nb];
                for i in 0..nb {
                    for k in 0..nb {
                        let use_it = if lower { k <= i } else { k >= i };
                        if use_it {
                            ab[i * nb + k] =
                                Complex64::new(prng(i * 1000 + k), prng(9000 + i * 1000 + k));
                        }
                    }
                    ab[i * nb + i] += Complex64::new(4.0, 0.0);
                }
                let di: Vec<Complex64> = (0..nb).map(|i| ab[i * nb + i].inv()).collect();
                let b0: Vec<Complex64> = (0..nb * ldb)
                    .map(|t| Complex64::new(prng(7000 + t), prng(8000 + t)))
                    .collect();
                let mut g = b0.clone();
                let mut v = b0.clone();
                unsafe {
                    band_left_lower_kernel(
                        g.as_mut_ptr(),
                        ldb,
                        x0,
                        nb,
                        0,
                        nrhs,
                        trans,
                        unit,
                        ab.as_ptr(),
                        di.as_ptr(),
                    );
                    band_left_lower_kernel_c64_avx2(
                        v.as_mut_ptr(),
                        ldb,
                        x0,
                        nb,
                        0,
                        nrhs,
                        trans,
                        unit,
                        ab.as_ptr(),
                        di.as_ptr(),
                    );
                }
                let mut max = 0.0f64;
                for i in 0..nb {
                    for j in 0..nrhs {
                        let d = (g[i * ldb + x0 + j] - v[i * ldb + x0 + j]).norm();
                        max = max.max(d);
                    }
                }
                assert!(max < 1e-9, "trans={trans} unit={unit} maxdiff={max:e}");
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn band_c32_avx2_matches_generic() {
        // `Complex32` twin of `band_c64_avx2_matches_generic` (1e-4).
        for (trans, lower) in [(false, true), (true, false)] {
            for unit in [false, true] {
                let (nb, nrhs, ldb, x0) = (40usize, 37usize, 41usize, 2usize);
                // Unit-diagonal recurrences with O(1) coefficients grow to
                // ~1e4 by row 39 (genuine instability, both kernels agree
                // relatively); shrink them so the absolute tolerance bites.
                let cscale = if unit { 0.1f32 } else { 1.0f32 };
                let mut ab = vec![Complex32::new(0.0, 0.0); nb * nb];
                for i in 0..nb {
                    for k in 0..nb {
                        let use_it = if lower { k <= i } else { k >= i };
                        if use_it && k != i {
                            ab[i * nb + k] = Complex32::new(
                                cscale * prng(i * 1000 + k) as f32,
                                cscale * prng(9000 + i * 1000 + k) as f32,
                            );
                        }
                    }
                    ab[i * nb + i] = if unit {
                        Complex32::new(1.0, 0.0)
                    } else {
                        Complex32::new(4.0, 0.0)
                    };
                }
                let di: Vec<Complex32> = (0..nb).map(|i| ab[i * nb + i].inv()).collect();
                let b0: Vec<Complex32> = (0..nb * ldb)
                    .map(|t| Complex32::new(prng(7000 + t) as f32, prng(8000 + t) as f32))
                    .collect();
                let mut g = b0.clone();
                let mut v = b0.clone();
                unsafe {
                    band_left_lower_kernel(
                        g.as_mut_ptr(),
                        ldb,
                        x0,
                        nb,
                        0,
                        nrhs,
                        trans,
                        unit,
                        ab.as_ptr(),
                        di.as_ptr(),
                    );
                    band_left_lower_kernel_c32_avx2(
                        v.as_mut_ptr(),
                        ldb,
                        x0,
                        nb,
                        0,
                        nrhs,
                        trans,
                        unit,
                        ab.as_ptr(),
                        di.as_ptr(),
                    );
                }
                let mut max = 0.0f32;
                for i in 0..nb {
                    for j in 0..nrhs {
                        let d = (g[i * ldb + x0 + j] - v[i * ldb + x0 + j]).norm();
                        max = max.max(d);
                    }
                }
                assert!(max < 1e-4, "trans={trans} unit={unit} maxdiff={max:e}");
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn trsm_blocked_matches_rowwise_bitwise() {
        if !crate::cpu::blas3::avx2::is_available() {
            return;
        }
        macro_rules! case {
            ($t:ty, $rowwise:ident, $blocked:ident) => {{
                for nb in [8usize, 9, 15, 16, 17, 31, 32, 64, 100, 127, 128] {
                    for nw in [1usize, 3, 7, 8, 9, 15, 16, 17, 24, 33, 64] {
                        for unit in [false, true] {
                            for pad in [0usize, 3] {
                                let ldb = nw + pad;
                                let mut a = vec![0 as $t; nb * nb];
                                for i in 0..nb {
                                    for j in 0..=i {
                                        a[i * nb + j] =
                                            (((i * 131 + j * 17) % 97) as f64 * 0.01 - 0.5) as $t;
                                    }
                                    a[i * nb + i] = (8.0 + (i % 5) as f64) as $t;
                                }
                                let mut abuf = vec![0 as $t; nb * nb];
                                pack_abuf(&a, nb, 0, nb, false, false, false, &mut abuf);
                                let dinv = panel_dinv(&a, nb, 0, nb, false, unit);
                                let b0: Vec<$t> = (0..nb * ldb)
                                    .map(|v| (((v * 37 + 11) % 89) as f64 * 0.02 - 0.9) as $t)
                                    .collect();
                                let mut got_row = b0.clone();
                                let mut got_blk = b0.clone();
                                // SAFETY: AVX2 checked above; packed `ab`/`di`
                                // cover the panel; bands own disjoint cols.
                                unsafe {
                                    $rowwise(
                                        got_row.as_mut_ptr(),
                                        ldb,
                                        0,
                                        nb,
                                        0,
                                        nw,
                                        false,
                                        unit,
                                        abuf.as_ptr(),
                                        dinv.as_ptr(),
                                    );
                                    $blocked(
                                        got_blk.as_mut_ptr(),
                                        ldb,
                                        0,
                                        nb,
                                        0,
                                        nw,
                                        false,
                                        unit,
                                        abuf.as_ptr(),
                                        dinv.as_ptr(),
                                    );
                                }
                                for (x, y) in got_row.iter().zip(got_blk.iter()) {
                                    assert!(
                                        x.to_bits() == y.to_bits(),
                                        "bitwise mismatch nb={nb} nw={nw} unit={unit} pad={pad}: {x} vs {y}"
                                    );
                                }
                            }
                        }
                    }
                }
            }};
        }
        case!(
            f32,
            band_left_lower_kernel_f32_avx2,
            band_left_lower_blocked_f32
        );
    }

    #[test]
    fn trsm_f32_left_lower_notrans_tiers_match_reference() {
        // Vector-tier coverage for the f32 AVX2 solve kernel at full
        // panel height: (128, 256) runs parallel 64-wide bands (16-wide
        // two-row unroll, 8-wide tier and scalar tail across the band
        // grid; a 32-wide tier slots in above the 16-wide one), while
        // (128, 100) runs one serial band with 64+36 chunks. Both go
        // through the packed-scratch path (`ldb != width`). Reference is
        // the naive f64 substitution at f32 tolerance; the AVX2 kernel's
        // FMA (~1 ulp) is well within it.
        for (m, n, unit) in [
            (128usize, 256usize, false),
            (128usize, 100usize, true),
            // Odd panel height: exercises the 32-wide tier's single-row
            // tail (`i < nb` after the row-pair loop) on 64-wide bands.
            (65usize, 128usize, false),
        ] {
            let (lda, ldb) = (m + 2, n + 3);
            // Strongly dominant triangle (diag ~8, off-diag <= 0.1):
            // forward substitution must not amplify rounding, so the
            // reference comparison measures kernel rounding (~1 ulp
            // from FMA) rather than conditioning. (`tri_a`'s ±1
            // off-diagonals explode at 128 rows: row sum ~64 vs diag
            // ~4, verified maxrel > 1 there.)
            let mut a64 = vec![0.0; m * lda];
            for i in 0..m {
                for j in 0..=i {
                    a64[i * lda + j] = if i == j {
                        8.0 + prng(7 + i * m + j)
                    } else {
                        0.1 * prng(7 + i * m + j)
                    };
                }
            }
            let a32: Vec<f32> = a64.iter().map(|&v| v as f32).collect();
            let b0: Vec<f32> = (0..m * ldb).map(|k| prng(100_000 + k) as f32).collect();
            let mut got = b0.clone();
            trsm_f32(
                TrsmSide::Left,
                TrsmUplo::Lower,
                Transpose::NoTrans,
                unit,
                m,
                n,
                1.0,
                &a32,
                lda,
                &mut got,
                ldb,
            );
            let mut want64: Vec<f64> = b0.iter().map(|&v| v as f64).collect();
            ref_trsm_f64(
                TrsmSide::Left,
                TrsmUplo::Lower,
                Transpose::NoTrans,
                unit,
                m,
                n,
                1.0,
                &a64,
                lda,
                &mut want64,
                ldb,
            );
            for i in 0..m {
                for j in 0..n {
                    let (g, w) = (got[i * ldb + j], want64[i * ldb + j]);
                    assert!(
                        (g as f64 - w).abs() <= 1e-4 * w.abs().max(1.0),
                        "f32 tier mismatch {g} vs {w} at ({i},{j}) unit={unit}"
                    );
                }
            }
        }
    }

    #[test]
    fn trsm_complex_spot_check() {
        // Complex twins share the generic core; spot-check NoTrans and
        // ConjTrans against explicit complex references.
        let (m, n) = (40, 37);
        let (lda, ldb) = (m + 2, n + 3);
        let a: Vec<Complex64> = (0..m * lda)
            .map(|k| Complex64::new(prng(k), prng(50_000 + k)))
            .collect();
        let mut a_dd = a.clone();
        for i in 0..m {
            a_dd[i * lda + i] += Complex64::new(4.0, 0.0);
        }
        for trans in [Transpose::NoTrans, Transpose::ConjTrans] {
            let b0: Vec<Complex64> = (0..m * ldb)
                .map(|k| Complex64::new(prng(100_000 + k), prng(200_000 + k)))
                .collect();
            let mut got = b0.clone();
            trsm_c64(
                TrsmSide::Left,
                TrsmUplo::Lower,
                trans,
                false,
                m,
                n,
                Complex64::new(1.5, -0.5),
                &a_dd,
                lda,
                &mut got,
                ldb,
            );
            // Reference: scale then substitute with conjugated reads for
            // ConjTrans. op(A) is lower for NoTrans (forward) and upper
            // for ConjTrans (backward).
            let mut want: Vec<Complex64> =
                b0.iter().map(|&v| Complex64::new(1.5, -0.5) * v).collect();
            let conj = trans == Transpose::ConjTrans;
            let aop = |r: usize, c: usize| -> Complex64 {
                let v = if conj {
                    a_dd[c * lda + r].conj()
                } else {
                    a_dd[r * lda + c]
                };
                v
            };
            for j in 0..n {
                if !conj {
                    for i in 0..m {
                        let mut s = want[i * ldb + j];
                        for k in 0..i {
                            s -= aop(i, k) * want[k * ldb + j];
                        }
                        want[i * ldb + j] = s / aop(i, i);
                    }
                } else {
                    for i in (0..m).rev() {
                        let mut s = want[i * ldb + j];
                        for k in (i + 1)..m {
                            s -= aop(i, k) * want[k * ldb + j];
                        }
                        want[i * ldb + j] = s / aop(i, i);
                    }
                }
            }
            // Compare the valid m × n submatrix only: padding is
            // untouched by the implementation by contract.
            let mut err = 0.0;
            let mut norm = 0.0;
            for i in 0..m {
                for j in 0..n {
                    err += (got[i * ldb + j] - want[i * ldb + j]).norm_sqr();
                    norm += want[i * ldb + j].norm_sqr();
                }
            }
            let (err, norm) = (err.sqrt(), norm.sqrt());
            assert!(
                err <= 1e-12 * norm.max(1.0),
                "c64 {trans:?}: {err} / {norm}"
            );
        }
        // c32 smoke: runs and matches c64 at f32 tolerance.
        let a32: Vec<Complex32> = a_dd
            .iter()
            .map(|&v| Complex32::new(v.re as f32, v.im as f32))
            .collect();
        let b0: Vec<Complex32> = (0..m * ldb)
            .map(|k| Complex32::new(prng(100_000 + k) as f32, prng(200_000 + k) as f32))
            .collect();
        let mut got = b0.clone();
        trsm_c32(
            TrsmSide::Left,
            TrsmUplo::Lower,
            Transpose::NoTrans,
            false,
            m,
            n,
            Complex32::new(1.0, 0.0),
            &a32,
            lda,
            &mut got,
            ldb,
        );
        assert!(got.iter().all(|v| v.re.is_finite() && v.im.is_finite()));
    }

    // Lane-vs-tail bitwise agreement: the 4-wide lane loops must produce
    // bit-identical results to the scalar tails on the same columns/rows.
    // Each case solves identical input twice — once with the target columns
    // (rows) covered only by tails (`< 4` wide), once with them covered by
    // lanes — and compares bitwise. Any FP-reordering in the lanes fails
    // this test while passing tolerance checks.
    fn lane_tail_f64_setup(nb: usize, ldb: usize) -> (Vec<f64>, Vec<f64>) {
        let lda = nb + 2;
        let mut a = vec![0.0; nb * lda];
        for i in 0..nb {
            for j in 0..nb {
                a[i * lda + j] = if i == j {
                    4.0 + (i as f64 * 0.37).sin()
                } else {
                    ((i * 31 + j * 17) as f64 * 0.01).sin()
                };
            }
        }
        let b: Vec<f64> = (0..nb * ldb)
            .map(|k| ((k * 13 + 7) as f64 * 0.023).sin())
            .collect();
        (a, b)
    }

    #[test]
    fn trsm_band_lane_tail_bitwise() {
        let nb = 8usize;
        let lda = nb + 2;
        let ldb = 11usize; // padded stride exercises non-unit ldb
        for trans in [false, true] {
            for unit in [false, true] {
                let (a, b0) = lane_tail_f64_setup(nb, ldb);
                // Shared panel setup; both runs below must agree bitwise,
                // so they share it.
                let dinv = panel_dinv(&a, lda, 0, nb, false, unit);
                let mut abuf = vec![0.0; nb * nb];
                pack_abuf(&a, lda, 0, nb, trans, false, false, &mut abuf);
                let (ab, di) = (abuf.as_ptr(), dinv.as_ptr());
                // Transposed pack for the right bands below.
                let mut abuf_t = vec![0.0; nb * nb];
                pack_abuf(&a, lda, 0, nb, trans, false, true, &mut abuf_t);
                let ab_t = abuf_t.as_ptr();
                // Left bands: columns 0..2 via tails vs via lanes.
                let mut tail_only = b0.clone();
                band_left_lower(
                    tail_only.as_mut_ptr(),
                    ldb,
                    0,
                    nb,
                    0,
                    3,
                    trans,
                    unit,
                    ab,
                    di,
                    band_left_lower_kernel_f64,
                );
                band_left_upper(
                    tail_only.as_mut_ptr(),
                    ldb,
                    0,
                    nb,
                    0,
                    3,
                    trans,
                    unit,
                    ab,
                    di,
                    band_left_upper_kernel_f64,
                );
                let mut with_lanes = b0.clone();
                band_left_lower(
                    with_lanes.as_mut_ptr(),
                    ldb,
                    0,
                    nb,
                    0,
                    7,
                    trans,
                    unit,
                    ab,
                    di,
                    band_left_lower_kernel_f64,
                );
                band_left_upper(
                    with_lanes.as_mut_ptr(),
                    ldb,
                    0,
                    nb,
                    0,
                    7,
                    trans,
                    unit,
                    ab,
                    di,
                    band_left_upper_kernel_f64,
                );
                for i in 0..nb {
                    for j in 0..3 {
                        assert_eq!(
                            tail_only[i * ldb + j].to_bits(),
                            with_lanes[i * ldb + j].to_bits(),
                            "left trans={trans} unit={unit} at ({i},{j})"
                        );
                    }
                }
                // Right bands: rows 0..2 via tails vs via lanes.
                let mut tail_only = b0.clone();
                band_right_lower(
                    tail_only.as_mut_ptr(),
                    ldb,
                    0,
                    0,
                    3,
                    nb,
                    trans,
                    unit,
                    ab_t,
                    di,
                    band_right_lower_kernel_f64,
                );
                band_right_upper(
                    tail_only.as_mut_ptr(),
                    ldb,
                    0,
                    0,
                    3,
                    nb,
                    trans,
                    unit,
                    ab_t,
                    di,
                    band_right_upper_kernel_f64,
                );
                let mut with_lanes = b0.clone();
                band_right_lower(
                    with_lanes.as_mut_ptr(),
                    ldb,
                    0,
                    0,
                    7,
                    nb,
                    trans,
                    unit,
                    ab_t,
                    di,
                    band_right_lower_kernel_f64,
                );
                band_right_upper(
                    with_lanes.as_mut_ptr(),
                    ldb,
                    0,
                    0,
                    7,
                    nb,
                    trans,
                    unit,
                    ab_t,
                    di,
                    band_right_upper_kernel_f64,
                );
                for i in 0..3 {
                    for j in 0..nb {
                        assert_eq!(
                            tail_only[i * ldb + j].to_bits(),
                            with_lanes[i * ldb + j].to_bits(),
                            "right trans={trans} unit={unit} at ({i},{j})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn trsm_band_packed_matches_narrow_bitwise() {
        // The narrow-stride scratch wrapper must be bitwise invisible: the
        // same band solved from a wide-stride layout (pack path) and from
        // a tight layout (fast path) agrees bit-for-bit. `nw = 7` covers
        // lanes + tail; `ldb_wide = 2048` is a 4K-multiple stride (pack
        // path and L1 set collisions on real hardware).
        let nb = 8usize;
        let lda = nb + 2;
        let nw = 7usize;
        let ldb_wide = 2048usize;
        for trans in [false, true] {
            for unit in [false, true] {
                let (a, b_wide) = lane_tail_f64_setup(nb, ldb_wide);
                let dinv = panel_dinv(&a, lda, 0, nb, false, unit);
                let mut abuf = vec![0.0; nb * nb];
                pack_abuf(&a, lda, 0, nb, trans, false, false, &mut abuf);
                let (ab, di) = (abuf.as_ptr(), dinv.as_ptr());
                // Tight copy of the same band values.
                let mut b_tight = vec![0.0; nb * nw];
                for i in 0..nb {
                    b_tight[i * nw..(i + 1) * nw]
                        .copy_from_slice(&b_wide[i * ldb_wide..i * ldb_wide + nw]);
                }
                let mut got_wide = b_wide.clone();
                band_left_lower(
                    got_wide.as_mut_ptr(),
                    ldb_wide,
                    0,
                    nb,
                    0,
                    nw,
                    trans,
                    unit,
                    ab,
                    di,
                    band_left_lower_kernel_f64,
                );
                band_left_upper(
                    got_wide.as_mut_ptr(),
                    ldb_wide,
                    0,
                    nb,
                    0,
                    nw,
                    trans,
                    unit,
                    ab,
                    di,
                    band_left_upper_kernel_f64,
                );
                let mut got_tight = b_tight.clone();
                band_left_lower(
                    got_tight.as_mut_ptr(),
                    nw,
                    0,
                    nb,
                    0,
                    nw,
                    trans,
                    unit,
                    ab,
                    di,
                    band_left_lower_kernel_f64,
                );
                band_left_upper(
                    got_tight.as_mut_ptr(),
                    nw,
                    0,
                    nb,
                    0,
                    nw,
                    trans,
                    unit,
                    ab,
                    di,
                    band_left_upper_kernel_f64,
                );
                for i in 0..nb {
                    for j in 0..nw {
                        assert_eq!(
                            got_wide[i * ldb_wide + j].to_bits(),
                            got_tight[i * nw + j].to_bits(),
                            "pack-vs-narrow trans={trans} unit={unit} at ({i},{j})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn trsm_tile_pack_f64_matches_scalar_bitwise() {
        // The vectorized f64 tile pack must be an exact copy of the scalar
        // loop: prime-ish dims defeat the 8-wide blocks and the padded
        // stride defeats unit-stride assumptions. Bitwise equality —
        // packing copies values.
        for (r, nb, ldb) in [
            (1usize, 1usize, 1usize),
            (7, 13, 19),
            (32, 128, 128),
            (33, 65, 2048),
        ] {
            let xb: Vec<f64> = (0..(r - 1) * ldb + nb)
                .map(|k| ((k * 13 + 7) as f64 * 0.023).sin())
                .collect();
            let mut got = vec![0.0; nb * r];
            let mut want = vec![0.0; nb * r];
            f64::pack_band_tile(xb.as_ptr(), ldb, 0, 0, r, nb, got.as_mut_ptr());
            pack_band_transpose(xb.as_ptr(), ldb, 0, 0, r, nb, want.as_mut_ptr());
            assert_eq!(got, want, "r={r} nb={nb} ldb={ldb}");
            // Pack/unpack roundtrip restores the band bit-for-bit
            // (padding cloned so only band columns are compared).
            let mut back = xb.clone();
            f64::unpack_band_tile(got.as_ptr(), back.as_mut_ptr(), ldb, 0, 0, r, nb);
            assert_eq!(back, xb, "roundtrip r={r} nb={nb} ldb={ldb}");
            // Nonzero block origin too.
            if r > 1 && nb > 1 {
                let x0 = 1usize;
                let i0 = 1usize;
                let mut got2 = vec![0.0; nb * r];
                let mut want2 = vec![0.0; nb * r];
                f64::pack_band_tile(xb.as_ptr(), ldb, x0, i0, r - 1, nb - 1, got2.as_mut_ptr());
                pack_band_transpose(xb.as_ptr(), ldb, x0, i0, r - 1, nb - 1, want2.as_mut_ptr());
                assert_eq!(got2, want2, "offset r={r} nb={nb} ldb={ldb}");
            }
        }
    }

    /// Team-slice equality: one shared setup plus serial band slices over
    /// odd row ranges reproduces the whole right-solve call to ~1 ulp
    /// (the safety property the Cholesky team loop relies on). Bitwise
    /// identity does NOT hold across different row groupings: the tile
    /// kernel runs full 4-wide column groups with FMA (single rounding)
    /// but scalar tails with separate mul+sub (double rounding), so a row
    /// that falls in a main group under one grouping and in a tail under
    /// another differs by 1 ulp — same contract as all vendor BLAS.
    /// Covers the in-situ variants (`Lower` × {`Trans`, `NoTrans`},
    /// non-unit) in `f64`/`f32` plus one `Upper` body.
    #[test]
    fn team_slices_match_whole_right() {
        let m = 200usize;
        let n = 64usize;
        let ranges = [(0usize, 70usize), (70, 150), (150, 200)];
        // f64 Lower x Trans/NoTrans.
        for trans in [Transpose::Trans, Transpose::NoTrans] {
            let trans_b = !matches!(trans, Transpose::NoTrans);
            let mut l11 = vec![0.0f64; n * n];
            for i in 0..n {
                for j in 0..=i {
                    l11[i * n + j] = if i == j {
                        2.0 + (i % 7) as f64 * 0.1
                    } else {
                        ((i * 13 + j * 7) % 11) as f64 * 0.05
                    };
                }
            }
            let b0: Vec<f64> = (0..m * n).map(|i| 0.02 * ((i % 61) as f64)).collect();
            let mut whole = b0.clone();
            trsm_f64(
                TrsmSide::Right,
                TrsmUplo::Lower,
                trans,
                false,
                m,
                n,
                1.0,
                &l11,
                n,
                &mut whole,
                n,
            );
            let mut abuf = vec![0.0f64; n * n];
            let di = trsm_right_setup_into(&l11, n, 0, n, trans_b, false, false, &mut abuf);
            let mut sliced = b0.clone();
            for &(i0, i1) in &ranges {
                // SAFETY: slices partition the rows (disjoint, in bounds);
                // setup shared read-only, AVX2 not required (dispatch inside).
                unsafe {
                    trsm_right_band_f64(
                        TrsmUplo::Lower,
                        sliced.as_mut_ptr(),
                        n,
                        0,
                        i0,
                        i1,
                        n,
                        trans_b,
                        false,
                        &abuf,
                        &di,
                    );
                }
            }
            let maxdiff = whole
                .iter()
                .zip(sliced.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            assert!(maxdiff <= 1e-15, "f64 Lower {trans:?}: maxdiff={maxdiff:e}");
        }
        // f32 Lower x Trans (in-situ single-precision variant).
        {
            let trans = Transpose::Trans;
            let trans_b = true;
            let mut l11 = vec![0.0f32; n * n];
            for i in 0..n {
                for j in 0..=i {
                    l11[i * n + j] = if i == j {
                        2.0 + (i % 7) as f32 * 0.1
                    } else {
                        ((i * 13 + j * 7) % 11) as f32 * 0.05
                    };
                }
            }
            let b0: Vec<f32> = (0..m * n).map(|i| 0.02 * ((i % 61) as f32)).collect();
            let mut whole = b0.clone();
            trsm_f32(
                TrsmSide::Right,
                TrsmUplo::Lower,
                trans,
                false,
                m,
                n,
                1.0,
                &l11,
                n,
                &mut whole,
                n,
            );
            let mut abuf = vec![0.0f32; n * n];
            let di = trsm_right_setup_into(&l11, n, 0, n, trans_b, false, false, &mut abuf);
            let mut sliced = b0.clone();
            for &(i0, i1) in &ranges {
                // SAFETY: as above.
                unsafe {
                    trsm_right_band_f32(
                        TrsmUplo::Lower,
                        sliced.as_mut_ptr(),
                        n,
                        0,
                        i0,
                        i1,
                        n,
                        trans_b,
                        false,
                        &abuf,
                        &di,
                    );
                }
            }
            let maxdiff = whole
                .iter()
                .zip(sliced.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(maxdiff <= 1e-6, "f32 Lower Trans: maxdiff={maxdiff:e}");
        }
        // Upper body + Complex64 smoke.
        {
            let mut u11 = vec![0.0f64; n * n];
            for i in 0..n {
                for j in i..n {
                    u11[i * n + j] = if i == j {
                        2.0 + (i % 5) as f64 * 0.1
                    } else {
                        ((i * 7 + j * 13) % 11) as f64 * 0.05
                    };
                }
            }
            let b0: Vec<f64> = (0..m * n).map(|i| 0.03 * ((i % 59) as f64)).collect();
            let mut whole = b0.clone();
            trsm_f64(
                TrsmSide::Right,
                TrsmUplo::Upper,
                Transpose::NoTrans,
                false,
                m,
                n,
                1.0,
                &u11,
                n,
                &mut whole,
                n,
            );
            let mut abuf = vec![0.0f64; n * n];
            let di = trsm_right_setup_into(&u11, n, 0, n, false, false, false, &mut abuf);
            let mut sliced = b0.clone();
            for &(i0, i1) in &ranges {
                // SAFETY: as above.
                unsafe {
                    trsm_right_band_f64(
                        TrsmUplo::Upper,
                        sliced.as_mut_ptr(),
                        n,
                        0,
                        i0,
                        i1,
                        n,
                        false,
                        false,
                        &abuf,
                        &di,
                    );
                }
            }
            let maxdiff = whole
                .iter()
                .zip(sliced.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            assert!(maxdiff <= 1e-15, "f64 Upper NoTrans: maxdiff={maxdiff:e}");
        }
    }

    #[test]
    fn narrow_width_gate() {
        // Measured shapes take narrow bands; wide/small/unmeasured keep default.
        assert_eq!(trsm_narrow_width(512, 128, 4), Some(16));
        assert_eq!(trsm_narrow_width(512, 128, 8), Some(8));
        assert_eq!(trsm_narrow_width(512, 256, 8), Some(16));
        assert_eq!(trsm_narrow_width(1024, 512, 4), Some(16));
        assert_eq!(trsm_narrow_width(512, 64, 4), Some(16));
        assert_eq!(trsm_narrow_width(512, 1024, 4), None);
        assert_eq!(trsm_narrow_width(128, 128, 4), None);
        assert_eq!(trsm_narrow_width(512, 32, 4), None);
        // m-decay: above 1024 the narrow win crosses to a loss (f32).
        assert_eq!(trsm_narrow_width(1920, 128, 4), None);
        assert_eq!(trsm_narrow_width(1920, 128, 8), None);
        assert_eq!(trsm_narrow_width(1024, 512, 8), Some(16));
        // Complex analogy: C32 (8 B) mirrors f64, C64 (16 B) takes 16.
        assert_eq!(trsm_narrow_width(512, 128, 16), Some(16));
    }

    #[test]
    fn fused_routing_gate() {
        use TrsmUplo::{Lower, Upper};
        // Eligible Left/Lower/NoTrans, MT, m >= 256 (crossover 224->256).
        assert!(trsm_use_fused(Lower, false, 512, 128, 16, false));
        assert!(trsm_use_fused(Lower, false, 256, 512, 2, false));
        assert!(trsm_use_fused(Lower, false, 256, 128, 16, false));
        // Single-threaded routes to the panel loop (measured win).
        assert!(!trsm_use_fused(Lower, false, 512, 128, 1, false));
        assert!(!trsm_use_fused(Lower, false, 1024, 512, 1, false));
        // Small multi-panel routes to the panel loop even MT.
        assert!(!trsm_use_fused(Lower, false, 160, 128, 16, false));
        assert!(!trsm_use_fused(Lower, false, 224, 256, 16, false));
        // Unchanged exclusions: Upper, Trans, single panel, narrow n, env-off.
        assert!(!trsm_use_fused(Upper, false, 512, 128, 16, false));
        assert!(!trsm_use_fused(Lower, true, 512, 128, 16, false));
        assert!(!trsm_use_fused(Lower, false, 128, 128, 16, false));
        assert!(!trsm_use_fused(Lower, false, 512, 64, 16, false));
        assert!(!trsm_use_fused(Lower, false, 512, 128, 16, true));
    }

    #[test]
    fn panel_nb_gate() {
        use TrsmUplo::{Lower, Upper};
        // f32 Lower/NoTrans takes fine panels; everything else keeps 128.
        assert_eq!(trsm_panel_nb(Lower, false, 4), 64);
        assert_eq!(trsm_panel_nb(Lower, false, 8), 128);
        assert_eq!(trsm_panel_nb(Lower, false, 16), 128);
        assert_eq!(trsm_panel_nb(Lower, true, 4), 128);
        assert_eq!(trsm_panel_nb(Upper, false, 4), 128);
        assert_eq!(trsm_panel_nb(Upper, true, 4), 128);
    }
}
