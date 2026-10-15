//! Dense GEMM driven by SME2 FMOPA outer-product micro-kernels.
//!
//! Computes `C := alpha * A * B + beta * C` where A is m × k, B is k × n,
//! C is m × n, all row-major with explicit leading dimensions.
//!
//! Design (follows arxiv 2512.21473v1 — Deng et al., "Demystifying ARM SME
//! to Optimize General Matrix Multiplications"). The pipeline is:
//!
//! 1. **Single-tile FMOPA kernel** (`sonium_{s,d}gemm_panel_sme2`) —
//!    SVL_T × SVL_T outer product per inner-k step into one ZA tile.
//!    At SVL=512: 256 f32 FMAs or 64 f64 FMAs per instruction.
//! 2. **Multi-tile (4-tile) FMOPA kernel** (`sonium_{s,d}gemm_panel4_sme2`)
//!    — one A column drives FOUR FMOPAs into ZA0..ZA3; B side uses a
//!    grouped multi-vector `LD1W {Zn-Zn+3}` (~900 GB/s vs ~230 GB/s for
//!    four single-Z loads). 1024 f32 FMAs or 256 f64 FMAs per inner step.
//! 3. **Outer Kc + Nc cache blocking** — adaptive: each is engaged only
//!    when the unblocked B-band working set would overflow the M4
//!    P-cluster's ~8 MB effective L2. Inside blocking, the first Kc pass
//!    uses caller's beta; subsequent passes use beta=1 to accumulate
//!    partial sums into C.
//! 4. **A-panel + B-panel packing** — A is packed column-major within each
//!    row-panel so the inner-k loop reads A columns with one contiguous
//!    LD1; B is packed into Nr-wide column panels so the 4-tile kernel's
//!    `mul vl` offsets land on contiguous rows. Both packs reuse
//!    thread-local growth buffers (no per-call allocation).
//! 5. **Column tail fallback** — when the trailing N cols don't fill a
//!    full 4*SVL_T panel they go through the single-tile kernel with
//!    predicated B loads. Same for row tail (mb < SVL_T) and Kc tail.
//!
//! Public entry points: [`sgemm`] (f32) and [`dgemm`] (f64). On x86_64 hosts
//! with AVX2+FMA they dispatch to the register-blocked kernels in
//! [`crate::cpu::blas3::avx2`]; everywhere else without SME2 they fall back
//! to a scalar reference. See [`super::gemm_bf16`] for the bf16-input
//! variant.

#![allow(clippy::too_many_arguments)]
#![cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]

use super::detect::{Sme2Caps, l2_budget_bytes};
use crate::{BackendError, BackendResult};

// ---- External assembly kernels (linked from sme2_kernels.S) ----

#[cfg(target_arch = "aarch64")]
unsafe extern "C" {
    fn sonium_sgemm_panel_sme2(
        mb: usize,
        nb: usize,
        k: usize,
        alpha: f32,
        packed_a: *const f32,
        b: *const f32,
        ldb: usize,
        beta: f32,
        c: *mut f32,
        ldc: usize,
    );
    fn sonium_sgemm_panel4_sme2(
        mb: usize,
        nb: usize,
        k: usize,
        alpha: f32,
        packed_a: *const f32,
        b: *const f32,
        ldb: usize,
        beta: f32,
        c: *mut f32,
        ldc: usize,
    );
    fn sonium_dgemm_panel_sme2(
        mb: usize,
        nb: usize,
        k: usize,
        alpha: f64,
        packed_a: *const f64,
        b: *const f64,
        ldb: usize,
        beta: f64,
        c: *mut f64,
        ldc: usize,
    );
    fn sonium_dgemm_panel4_sme2(
        mb: usize,
        nb: usize,
        k: usize,
        alpha: f64,
        packed_a: *const f64,
        b: *const f64,
        ldb: usize,
        beta: f64,
        c: *mut f64,
        ldc: usize,
    );
    fn sonium_dgemm_panel8_sme2(
        mb: usize,
        nb: usize,
        k: usize,
        alpha: f64,
        packed_a: *const f64,
        b: *const f64,
        ldb: usize,
        beta: f64,
        c: *mut f64,
        ldc: usize,
    );
}

// Multi-tile micro-kernel width: 4 col-tiles per panel = 4*SVL_T columns.
// At SVL=512 this is 64 for f32 and 32 for f64.
const TILES_PER_PANEL: usize = 4;

// f64 has 8 ZA tiles (vs 4 for f32), so dgemm has a wider second-stage
// fast path that uses ALL of them — 8 col-tiles per panel = 8*SVL_D
// columns. At SVL=512 this is 64. The driver runs the 8-tile kernel for
// the panel8-aligned bulk of N, then falls back to the 4-tile kernel for
// the in-between (4-tile-aligned but not 8-tile-aligned) remainder, then
// to single-tile for the column tail.
const DGEMM_TILES_PER_PANEL_8: usize = 8;

// Minimum M dimension before paying rayon thread-spawn overhead. The fast
// path is "one full row-panel per worker", so the right threshold is
// `rayon::current_num_threads() * svl_T`. At SVL=512 on the 4-core M4
// P-cluster default this is 64 for f32 (svl_s=16) and 32 for f64
// (svl_d=8). The compile-time constant is the documented fallback when
// SVL detection isn't available. Not OnceLock-cached because svl_s and
// svl_d would clobber each other and the multiplication is one cycle.
const PARALLEL_M_THRESHOLD_DEFAULT: usize = 64;

#[inline]
fn parallel_m_threshold(svl_t: usize) -> usize {
    let n = rayon::current_num_threads();
    if n == 0 || svl_t == 0 {
        PARALLEL_M_THRESHOLD_DEFAULT
    } else {
        n * svl_t
    }
}

/// Validate one row-major matrix view before entering a native GEMM kernel.
///
/// Checked entry points must reject malformed dimensions without relying on
/// the assertions retained by the legacy infallible wrappers. A zero-extent
/// matrix has no addressable elements, so its required span is zero, while
/// the leading dimension still has to cover the logical column count.
#[inline]
fn validate_matrix_bounds(
    operation: &'static str,
    name: &'static str,
    rows: usize,
    cols: usize,
    len: usize,
    ld: usize,
) -> BackendResult<()> {
    if cols > ld {
        return Err(BackendError::InvalidArgument {
            operation,
            reason: format!("{name} leading dimension {ld} < columns {cols}"),
        });
    }
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    let required = (rows - 1)
        .checked_mul(ld)
        .and_then(|offset| offset.checked_add(cols))
        .ok_or_else(|| BackendError::InvalidArgument {
            operation,
            reason: format!("{name} matrix span overflows usize"),
        })?;
    if len < required {
        return Err(BackendError::InvalidArgument {
            operation,
            reason: format!("{name} buffer length {len} too small for required {required}"),
        });
    }
    Ok(())
}

#[inline]
pub(crate) fn validate_gemm_bounds(
    operation: &'static str,
    trans_a: bool,
    m: usize,
    n: usize,
    k: usize,
    a_len: usize,
    lda: usize,
    b_len: usize,
    ldb: usize,
    c_len: usize,
    ldc: usize,
) -> BackendResult<()> {
    let (a_rows, a_cols) = if trans_a { (k, m) } else { (m, k) };
    validate_matrix_bounds(operation, "A", a_rows, a_cols, a_len, lda)?;
    validate_matrix_bounds(operation, "B", k, n, b_len, ldb)?;
    validate_matrix_bounds(operation, "C", m, n, c_len, ldc)
}

// Verify that an (m × n) sub-matrix with leading dimension `ld` fits
// inside `slice_len` row-major elements. Same bound formula for A
// (rows=m, cols=k), B (rows=k, cols=n), and C (rows=m, cols=n).
//
// The kernel reads element `[(m-1) * ld + (n-1)]` at worst; the slice
// therefore needs at least `(m-1) * ld + n` elements (or zero if `m`
// or `n` is zero — both degenerate cases return early before any load).
#[inline]
fn check_matrix_bounds(slice_len: usize, rows: usize, cols: usize, ld: usize, name: &str) {
    if rows == 0 || cols == 0 {
        return;
    }
    assert!(
        cols <= ld,
        "{name}: cols ({cols}) exceeds leading dimension ({ld})"
    );
    let last = (rows - 1).saturating_mul(ld).saturating_add(cols);
    assert!(
        last <= slice_len,
        "{name}: slice len {slice_len} < required {last} for rows={rows}, cols={cols}, ld={ld}"
    );
}

// Outer-K block size. Engaged only when the unblocked B-band working set
// (K*N*sizeof) would overflow the M4 P-cluster's ~8 MB effective L2 — see
// `choose_kc`. When the unblocked path is L2-friendly we set Kc=K (single
// pass, no partial-sum re-touch of C). When blocked, Kc is capped here.
const SGEMM_KC_CAP: usize = 512;
const DGEMM_KC_CAP: usize = 256;
// Outer N-block cap. With caps Kc=KC_CAP, sizeof=4/8 the packed_B working
// set per (j_c, p_c) tile is Kc * Nc * sizeof. Capping at 2048 leaves ~4
// MB of L2 headroom for packed_A, the C row band, and the still-cold A/B
// rows being streamed — which matters at the L2 boundary where setting
// Nc to the full L2 budget would thrash. (Earlier 4096 cap put us at
// exactly 8 MB = entire budget; benches were OK at that size but the
// margin was uncomfortable.)
const SGEMM_NC_CAP: usize = 2048;
const DGEMM_NC_CAP: usize = 2048;

fn choose_kc(k: usize, n: usize, elem_bytes: usize, cap: usize) -> usize {
    let unblocked_bytes = k.saturating_mul(n).saturating_mul(elem_bytes);
    if unblocked_bytes <= l2_budget_bytes() {
        k // unblocked — packed-A is tiny and B band fits in L2
    } else {
        cap.min(k)
    }
}

fn choose_nc(k: usize, n: usize, elem_bytes: usize, cap: usize) -> usize {
    // Mirror the Kc rule: skip outer Nc blocking when the whole B band fits
    // in L2. Once blocked, cap at NC_CAP so packed_B per (j_c, p_c) tile
    // stays L2-friendly.
    let unblocked_bytes = k.saturating_mul(n).saturating_mul(elem_bytes);
    if unblocked_bytes <= l2_budget_bytes() {
        n
    } else {
        cap.min(n)
    }
}

// Thread-local pack workspaces. Reused across calls so a small GEMM doesn't
// pay a fresh allocation on every invocation (matters at ≤256³ where the
// kernel itself is sub-100µs). The Vecs grow monotonically; each kernel call
// `resize`s only when the existing capacity is insufficient.
//
// SGEMM/DGEMM_PACKED_B live on the calling thread and are written serially
// before each rayon::scope (one packer, many readers). The cached Vec is moved
// out of the RefCell while in use, keeping its allocation stable without
// holding a TLS borrow across the scope. This is important when nested Rayon
// work re-enters GEMM on the same worker.
//
// SGEMM/DGEMM_LOCAL_PA live on EVERY thread, so each rayon worker has its
// own packed_A buffer. They amortise the per-task allocation that the
// previous version did with `vec![0; pa_need]` inside every spawn — at
// m=1024, that was 64+ mallocs per (j_c, p_c) iteration.
use std::cell::RefCell;
use std::thread::LocalKey;
thread_local! {
    static SGEMM_PACKED_B: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static DGEMM_PACKED_B: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static SGEMM_LOCAL_PA: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static DGEMM_LOCAL_PA: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

// Take ownership of a cached pack buffer, growing it on first use, and return
// it after the closure finishes. Moving the Vec does not move its allocation,
// so raw pointers used by rayon workers remain stable across the scope.
//
// Crucially, no RefCell borrow is held while `f` runs. A Rayon worker waiting
// in a nested scope may execute another GEMM on the same thread; that
// reentrant call sees an empty cache and allocates a separate buffer instead
// of panicking with "RefCell already borrowed".
fn with_pack_buffer<T, F, R>(cell: &'static LocalKey<RefCell<Vec<T>>>, need: usize, f: F) -> R
where
    T: Copy + num_traits::Zero + 'static,
    F: FnOnce(&mut [T]) -> R,
{
    cell.with(|c| {
        let mut buf = std::mem::take(&mut *c.borrow_mut());
        if buf.len() < need {
            buf.resize(need, T::zero());
        }
        let result = f(&mut buf[..need]);
        c.replace(buf);
        result
    })
}

// Same trick on the per-worker packed_A buffer. Each rayon task body wraps
// itself in this so the first call per worker grows the Vec; later calls
// re-use the existing allocation.
//
// Reentrancy: a rayon worker waiting in a nested `rayon::scope` may steal
// another GEMM onto the same thread. `try_borrow_mut` detects that case and
// falls back to a freshly allocated buffer owned by this call, so the nested
// call pays one allocation instead of panicking with
// "RefCell already borrowed". The hot non-nested path is unchanged.
fn with_local_pa<T, F, R>(cell: &'static LocalKey<RefCell<Vec<T>>>, need: usize, f: F) -> R
where
    T: Copy + num_traits::Zero + 'static,
    F: FnOnce(&mut [T]) -> R,
{
    cell.with(|c| match c.try_borrow_mut() {
        Ok(mut buf) => {
            if buf.len() < need {
                buf.resize(need, T::zero());
            }
            f(&mut buf[..need])
        }
        Err(_) => {
            let mut buf = vec![T::zero(); need];
            f(&mut buf)
        }
    })
}

// Pointer newtypes that are `Send + Sync` for rayon::scope worker dispatch.
// Each worker dereferences these into disjoint row-bands of C / read-only
// borrows of B and packed_B — see the SAFETY block above each `rayon::scope`
// for the disjointness argument. `Sync` is needed because rayon::scope's
// outer closure must be `Send`, and the outer closure captures these by
// reference before each spawn copies them into workers by value.
#[derive(Copy, Clone)]
struct SendMutPtr<T>(*mut T);
#[derive(Copy, Clone)]
struct SendConstPtr<T>(*const T);
// SAFETY: callers ensure each spawned task writes to a disjoint sub-range
// of the underlying allocation; no two tasks ever touch the same byte.
// Const pointers are read-only — multi-threaded reads are always sound.
unsafe impl<T> Send for SendMutPtr<T> {}
unsafe impl<T> Sync for SendMutPtr<T> {}
unsafe impl<T> Send for SendConstPtr<T> {}
unsafe impl<T> Sync for SendConstPtr<T> {}

// One row-panel of the GEMM micro-kernel: pack A into `packed_a`, run the
// 4-tile fast path over the packed B panels, then handle the column tail
// with the single-tile kernel. This is the shared body between sgemm and
// dgemm, and between their serial and parallel call sites — the four
// previous copy-pasted versions all collapse here.
//
// Generic over T (f32/f64) plus the two kernel fn pointers, which lets a
// single body cover both precisions without runtime indirection cost: the
// fn-pointer call site monomorphises to a direct call after inlining when
// the caller passes a known-constant `extern "C"` function item.
//
// # Safety
//
// - `a` must reference at least `(i0+mb-1)*lda + p_start+kc` valid elements.
// - `b_ptr` must be the head of an array with at least
//   `(p_start+kc-1)*ldb + j_c+nc_end` valid elements.
// - `packed_b_ptr` must reference at least
//   `(nc_full/panel_nr) * (kc*panel_nr)` valid elements packed by
//   `pack_b_panels`.
// - `c_ptr.add(i0*ldc + j)` for `j in j_c..j_c+nc_end` must be valid for
//   writes, and no other task may write the same byte concurrently.
// - The asm kernels assume streaming-SVE/SME2 is available; callers must
//   have verified `Sme2Caps::has_sme2` (and `has_f64` for dgemm).
/// Pack-A function pointer used by the row-panel processors. Picks between
/// `pack_a_panel_kc` (NoTrans) and `pack_a_panel_kc_trans` (Trans/ConjTrans
/// on real types) at the dispatcher level so the kernel inner loop never
/// branches.
type PackAFn<T> = fn(&[T], usize, usize, usize, usize, usize, usize, &mut [T]);

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn process_row_panel<T: Copy + num_traits::Zero>(
    panel4_kernel: unsafe extern "C" fn(
        usize,
        usize,
        usize,
        T,
        *const T,
        *const T,
        usize,
        T,
        *mut T,
        usize,
    ),
    panel1_kernel: unsafe extern "C" fn(
        usize,
        usize,
        usize,
        T,
        *const T,
        *const T,
        usize,
        T,
        *mut T,
        usize,
    ),
    pack_a_fn: PackAFn<T>,
    mb: usize,
    i0: usize,
    j_c: usize,
    nc_full: usize,
    nc_end: usize,
    p_start: usize,
    kc: usize,
    svl: usize,
    panel_nr: usize,
    alpha: T,
    beta_eff: T,
    lda: usize,
    ldb: usize,
    ldc: usize,
    a: &[T],
    b_ptr: *const T,
    packed_a: &mut [T],
    packed_b_ptr: *const T,
    c_ptr: *mut T,
) {
    pack_a_fn(a, lda, i0, mb, p_start, kc, svl, packed_a);

    // 4-tile fast path over the fully-packed B panels.
    let mut panel_idx = 0usize;
    while panel_idx * panel_nr < nc_full {
        let j0 = j_c + panel_idx * panel_nr;
        let pb_off = panel_idx * (kc * panel_nr);
        unsafe {
            panel4_kernel(
                mb,
                panel_nr,
                kc,
                alpha,
                packed_a.as_ptr(),
                packed_b_ptr.add(pb_off),
                panel_nr,
                beta_eff,
                c_ptr.add(i0 * ldc + j0),
                ldc,
            );
        }
        panel_idx += 1;
    }

    // Column tail — reads B directly with the caller's ldb (unpacked).
    let mut j0 = j_c + nc_full;
    while j0 < nc_end {
        let nb = (nc_end - j0).min(svl);
        unsafe {
            panel1_kernel(
                mb,
                nb,
                kc,
                alpha,
                packed_a.as_ptr(),
                b_ptr.add(p_start * ldb + j0),
                ldb,
                beta_eff,
                c_ptr.add(i0 * ldc + j0),
                ldc,
            );
        }
        j0 += svl;
    }
}

// Three-tier row-panel processor used by dgemm. Identical structure to
// `process_row_panel` but with an extra "big" multi-tile tier inserted
// before the "small" tier. f64 gets this because it has 8 ZA.D tiles
// (8-tile fast path) on top of the universal 4-tile path; f32 only has 4
// ZA.S tiles so its multi-tile path is single-tier and uses the simpler
// helper above.
//
// Cols are processed in this order:
//   [j_c                              .. j_c + nc_full_big       ) — panel_big_kernel
//   [j_c + nc_full_big                .. j_c + nc_full_big + nc_full_small) — panel_small_kernel
//   [j_c + nc_full_big + nc_full_small .. nc_end                  ) — panel_tail_kernel (svl wide)
//
// `packed_b_big_ptr` and `packed_b_small_ptr` reference DISJOINT regions
// of the same `packed_b` buffer (the caller packs each region with the
// matching panel_nr stride).
//
// # Safety
//
// Same contract as `process_row_panel`. `nc_full_big` must be a multiple
// of `panel_nr_big`; `nc_full_small` must be a multiple of `panel_nr_small`.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn process_row_panel_3tier<T: Copy + num_traits::Zero>(
    panel_big_kernel: unsafe extern "C" fn(
        usize,
        usize,
        usize,
        T,
        *const T,
        *const T,
        usize,
        T,
        *mut T,
        usize,
    ),
    panel_small_kernel: unsafe extern "C" fn(
        usize,
        usize,
        usize,
        T,
        *const T,
        *const T,
        usize,
        T,
        *mut T,
        usize,
    ),
    panel_tail_kernel: unsafe extern "C" fn(
        usize,
        usize,
        usize,
        T,
        *const T,
        *const T,
        usize,
        T,
        *mut T,
        usize,
    ),
    pack_a_fn: PackAFn<T>,
    mb: usize,
    i0: usize,
    j_c: usize,
    nc_full_big: usize,
    nc_full_small: usize,
    nc_end: usize,
    p_start: usize,
    kc: usize,
    svl: usize,
    panel_nr_big: usize,
    panel_nr_small: usize,
    alpha: T,
    beta_eff: T,
    lda: usize,
    ldb: usize,
    ldc: usize,
    a: &[T],
    b_ptr: *const T,
    packed_a: &mut [T],
    packed_b_big_ptr: *const T,
    packed_b_small_ptr: *const T,
    c_ptr: *mut T,
) {
    pack_a_fn(a, lda, i0, mb, p_start, kc, svl, packed_a);

    // Tier 1: big multi-tile (8-tile for dgemm).
    let mut panel_idx = 0usize;
    while panel_idx * panel_nr_big < nc_full_big {
        let j0 = j_c + panel_idx * panel_nr_big;
        let pb_off = panel_idx * (kc * panel_nr_big);
        unsafe {
            panel_big_kernel(
                mb,
                panel_nr_big,
                kc,
                alpha,
                packed_a.as_ptr(),
                packed_b_big_ptr.add(pb_off),
                panel_nr_big,
                beta_eff,
                c_ptr.add(i0 * ldc + j0),
                ldc,
            );
        }
        panel_idx += 1;
    }

    // Tier 2: small multi-tile (4-tile for dgemm) — picks up any
    // 4-aligned-but-not-8-aligned cols left after tier 1.
    let small_base = j_c + nc_full_big;
    let mut panel_idx = 0usize;
    while panel_idx * panel_nr_small < nc_full_small {
        let j0 = small_base + panel_idx * panel_nr_small;
        let pb_off = panel_idx * (kc * panel_nr_small);
        unsafe {
            panel_small_kernel(
                mb,
                panel_nr_small,
                kc,
                alpha,
                packed_a.as_ptr(),
                packed_b_small_ptr.add(pb_off),
                panel_nr_small,
                beta_eff,
                c_ptr.add(i0 * ldc + j0),
                ldc,
            );
        }
        panel_idx += 1;
    }

    // Tier 3: column tail — single-tile, reads B directly (unpacked).
    let mut j0 = j_c + nc_full_big + nc_full_small;
    while j0 < nc_end {
        let nb = (nc_end - j0).min(svl);
        unsafe {
            panel_tail_kernel(
                mb,
                nb,
                kc,
                alpha,
                packed_a.as_ptr(),
                b_ptr.add(p_start * ldb + j0),
                ldb,
                beta_eff,
                c_ptr.add(i0 * ldc + j0),
                ldc,
            );
        }
        j0 += svl;
    }
}

/// Checked `C := alpha * op(A) * B + beta * C` (f32, row-major).
///
/// This is the entry point for caller-controlled shapes. It performs the
/// same span admission used by the backend GEMM contract and returns a typed
/// [`BackendError::InvalidArgument`] before any output mutation. The legacy
/// [`sgemm`] wrapper retains its historical panic-on-invalid-input behavior.
#[allow(clippy::too_many_arguments)]
pub fn try_sgemm(
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
) -> BackendResult<()> {
    validate_gemm_bounds(
        "sgemm",
        trans_a,
        m,
        n,
        k,
        a.len(),
        lda,
        b.len(),
        ldb,
        c.len(),
        ldc,
    )?;
    sgemm(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_a);
    Ok(())
}

/// `C := alpha * op(A) * B + beta * C` (f32, row-major).
///
/// `trans_a = false` ⇒ A is m×k row-major with `lda >= k` (NoTrans).
/// `trans_a = true`  ⇒ stored A is k×m row-major with `lda >= m`, applied
///                      as A^T. ConjTrans on a real type is identical
///                      to Trans.
///
/// B is always read as NoTrans (k×n row-major, `ldb >= n`); the TransB
/// path stays on the scalar fallback for now — uncommon in BEM/FEM, and
/// pre-packing the column tail for TransB adds complexity we'll only pay
/// for when there's a measured hot path. Use `cpu_blas3::scalar::gemm_f32`
/// for those combinations.
#[allow(clippy::too_many_arguments)]
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
    trans_a: bool,
) {
    // Bounds checks run for ALL paths (SME2 and scalar) so callers get the
    // same Rust-side panic regardless of host. Trans A is stored k×m so
    // the bound check rows/cols swap.
    if trans_a {
        check_matrix_bounds(a.len(), k, m, lda, "sgemm A^T");
    } else {
        check_matrix_bounds(a.len(), m, k, lda, "sgemm A");
    }
    check_matrix_bounds(b.len(), k, n, ldb, "sgemm B");
    check_matrix_bounds(c.len(), m, n, ldc, "sgemm C");

    let caps = Sme2Caps::detect();
    if !caps.has_sme2 {
        return sgemm_fallback(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_a);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        sgemm_fallback(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_a);
    }
    #[cfg(target_arch = "aarch64")]
    {
        let svl_s = caps.svl_bits / 32;
        if svl_s == 0 {
            return sgemm_scalar(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_a);
        }

        // Degenerate cases: when there is no FMA work to do, GEMM reduces
        // to `C := beta * C`. scale_c handles beta == 1.0 as a no-op.
        if k == 0 || alpha == 0.0 {
            scale_c(m, n, beta, c, ldc);
            return;
        }

        let panel_nr = TILES_PER_PANEL * svl_s; // 64 at SVL=512
        let kc_max = choose_kc(k, n, core::mem::size_of::<f32>(), SGEMM_KC_CAP);
        let nc_max = choose_nc(k, n, core::mem::size_of::<f32>(), SGEMM_NC_CAP);
        let nc_max_full = (nc_max / panel_nr) * panel_nr;

        // Each row-panel writes to non-overlapping rows of C, so no locks
        // needed. Below threshold, rayon's spawn cost dominates the work.
        let use_parallel = m >= parallel_m_threshold(svl_s);

        let pb_need = (kc_max * nc_max_full).max(1);
        let pa_need = svl_s * kc_max;

        // Pack-A helper picked once per call; the kernel inner loop reads
        // the packed buffer with one layout regardless of the source.
        let pack_a_fn: PackAFn<f32> = if trans_a {
            pack_a_panel_kc_trans::<f32>
        } else {
            pack_a_panel_kc::<f32>
        };

        // Take the calling thread's cached packed_B out of its RefCell for
        // the whole gemm call (it is put back afterwards). Workers read it
        // through a raw pointer; owning the Vec keeps the allocation alive
        // and at a stable address across every `rayon::scope` below, and a
        // nested GEMM re-entering on this thread falls back to its own
        // buffer instead of panicking on the TLS borrow.
        with_pack_buffer(&SGEMM_PACKED_B, pb_need, |packed_b| {
            let mut j_c = 0usize;
            while j_c < n {
                let nc = (n - j_c).min(nc_max);
                let nc_end = j_c + nc;
                let nc_full = (nc / panel_nr) * panel_nr;

                let mut p_start = 0usize;
                while p_start < k {
                    let kc = (k - p_start).min(kc_max);
                    let beta_eff = if p_start == 0 { beta } else { 1.0f32 };

                    if nc_full > 0 {
                        pack_b_panels(b, ldb, p_start, j_c, kc, panel_nr, nc_full, packed_b);
                    }

                    if use_parallel {
                        // SAFETY: every spawned task processes a distinct
                        // row-panel `[i0, i0+mb)` where `i0` advances by
                        // `svl_s` and `mb <= svl_s`, so the panels never
                        // overlap. The only mutable aliasing is on `C`,
                        // and each task only writes into its own row-band
                        // `c[i0*ldc .. (i0+mb)*ldc]`. `packed_b` and `a`
                        // are read-only borrows held for the duration of
                        // the scope. `rayon::scope` joins before we mutate
                        // either of them.
                        let c_send = SendMutPtr(c.as_mut_ptr());
                        let b_send = SendConstPtr(b.as_ptr());
                        let pb_send = SendConstPtr(packed_b.as_ptr());

                        rayon::scope(|s| {
                            let mut i0 = 0usize;
                            while i0 < m {
                                let mb = (m - i0).min(svl_s);
                                s.spawn(move |_| {
                                    // Force whole-struct capture of the
                                    // wrappers (defeats disjoint-capture,
                                    // which would otherwise grab the bare
                                    // raw-ptr `.0` field and refuse to ship
                                    // it across threads).
                                    let c_send: SendMutPtr<f32> = c_send;
                                    let b_send: SendConstPtr<f32> = b_send;
                                    let pb_send: SendConstPtr<f32> = pb_send;
                                    with_local_pa(&SGEMM_LOCAL_PA, pa_need, |packed_a| {
                                        // SAFETY: see the SAFETY block on
                                        // `process_row_panel`. Bounds were
                                        // pre-checked by `check_matrix_bounds`
                                        // at the top of `sgemm`.
                                        unsafe {
                                            process_row_panel::<f32>(
                                                sonium_sgemm_panel4_sme2,
                                                sonium_sgemm_panel_sme2,
                                                pack_a_fn,
                                                mb,
                                                i0,
                                                j_c,
                                                nc_full,
                                                nc_end,
                                                p_start,
                                                kc,
                                                svl_s,
                                                panel_nr,
                                                alpha,
                                                beta_eff,
                                                lda,
                                                ldb,
                                                ldc,
                                                a,
                                                b_send.0,
                                                packed_a,
                                                pb_send.0,
                                                c_send.0,
                                            );
                                        }
                                    });
                                });
                                i0 += svl_s;
                            }
                        });
                    } else {
                        // Serial path: the calling thread does every row-panel
                        // in turn, reusing one packed_A buffer for the lot.
                        with_local_pa(&SGEMM_LOCAL_PA, pa_need, |packed_a| {
                            let c_ptr = c.as_mut_ptr();
                            let b_ptr = b.as_ptr();
                            let packed_b_ptr = packed_b.as_ptr();
                            let mut i0 = 0usize;
                            while i0 < m {
                                let mb = (m - i0).min(svl_s);
                                // SAFETY: bounds pre-checked; no aliasing
                                // (single-threaded).
                                unsafe {
                                    process_row_panel::<f32>(
                                        sonium_sgemm_panel4_sme2,
                                        sonium_sgemm_panel_sme2,
                                        pack_a_fn,
                                        mb,
                                        i0,
                                        j_c,
                                        nc_full,
                                        nc_end,
                                        p_start,
                                        kc,
                                        svl_s,
                                        panel_nr,
                                        alpha,
                                        beta_eff,
                                        lda,
                                        ldb,
                                        ldc,
                                        a,
                                        b_ptr,
                                        packed_a,
                                        packed_b_ptr,
                                        c_ptr,
                                    );
                                }
                                i0 += svl_s;
                            }
                        });
                    }

                    p_start += kc;
                }
                j_c += nc;
            }
        });
    }
}

/// Checked `C := alpha * op(A) * B + beta * C` (f64, row-major).
///
/// This validates every matrix span before dispatch. Use this API when any
/// dimension or leading dimension originates outside the crate; the legacy
/// [`dgemm`] wrapper preserves its panic-on-invalid-input behavior.
#[allow(clippy::too_many_arguments)]
pub fn try_dgemm(
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
) -> BackendResult<()> {
    validate_gemm_bounds(
        "dgemm",
        trans_a,
        m,
        n,
        k,
        a.len(),
        lda,
        b.len(),
        ldb,
        c.len(),
        ldc,
    )?;
    dgemm(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_a);
    Ok(())
}

/// `C := alpha * op(A) * B + beta * C` (f64, row-major). See [`sgemm`]
/// docs for the `trans_a` semantics — same shape for f64.
#[allow(clippy::too_many_arguments)]
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
    trans_a: bool,
) {
    if trans_a {
        check_matrix_bounds(a.len(), k, m, lda, "dgemm A^T");
    } else {
        check_matrix_bounds(a.len(), m, k, lda, "dgemm A");
    }
    check_matrix_bounds(b.len(), k, n, ldb, "dgemm B");
    check_matrix_bounds(c.len(), m, n, ldc, "dgemm C");

    let caps = Sme2Caps::detect();
    if !caps.has_f64 {
        return dgemm_fallback(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_a);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dgemm_fallback(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_a);
    }
    #[cfg(target_arch = "aarch64")]
    {
        let svl_d = caps.svl_bits / 64;
        if svl_d == 0 {
            return dgemm_scalar(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_a);
        }

        // Degenerate cases — see sgemm for the rationale.
        if k == 0 || alpha == 0.0 {
            scale_c(m, n, beta, c, ldc);
            return;
        }

        let panel_nr_8 = DGEMM_TILES_PER_PANEL_8 * svl_d; // 64 at SVL=512
        let panel_nr_4 = TILES_PER_PANEL * svl_d; // 32 at SVL=512
        let kc_max = choose_kc(k, n, core::mem::size_of::<f64>(), DGEMM_KC_CAP);
        let nc_max = choose_nc(k, n, core::mem::size_of::<f64>(), DGEMM_NC_CAP);
        // packed_B working set must hold both an 8-tile aligned region
        // AND a residual 4-tile aligned region for the same Nc tile —
        // sized for the worst case where everything is 4-aligned but not
        // 8-aligned (= nc_max rounded down to panel_nr_4).
        let nc_max_full_4 = (nc_max / panel_nr_4) * panel_nr_4;

        let use_parallel = m >= parallel_m_threshold(svl_d);

        let pb_need = (kc_max * nc_max_full_4).max(1);
        let pa_need = svl_d * kc_max;

        let pack_a_fn: PackAFn<f64> = if trans_a {
            pack_a_panel_kc_trans::<f64>
        } else {
            pack_a_panel_kc::<f64>
        };

        with_pack_buffer(&DGEMM_PACKED_B, pb_need, |packed_b| {
            let mut j_c = 0usize;
            while j_c < n {
                let nc = (n - j_c).min(nc_max);
                let nc_end = j_c + nc;
                // 8-tile bulk: largest multiple of panel_nr_8 in nc.
                let nc_full_8 = (nc / panel_nr_8) * panel_nr_8;
                // 4-tile remainder: take what's left of (nc - nc_full_8)
                // that's still panel_nr_4-aligned.
                let nc_rem = nc - nc_full_8;
                let nc_full_4 = (nc_rem / panel_nr_4) * panel_nr_4;

                let mut p_start = 0usize;
                while p_start < k {
                    let kc = (k - p_start).min(kc_max);
                    let beta_eff = if p_start == 0 { beta } else { 1.0f64 };

                    // Pack both regions of B into disjoint sub-ranges of
                    // packed_b. The 8-tile region occupies indices
                    // [0, kc * nc_full_8); the 4-tile region picks up at
                    // `kc * nc_full_8`.
                    if nc_full_8 > 0 {
                        pack_b_panels(
                            b,
                            ldb,
                            p_start,
                            j_c,
                            kc,
                            panel_nr_8,
                            nc_full_8,
                            &mut packed_b[..kc * nc_full_8],
                        );
                    }
                    if nc_full_4 > 0 {
                        let pb_4_base = kc * nc_full_8;
                        pack_b_panels(
                            b,
                            ldb,
                            p_start,
                            j_c + nc_full_8,
                            kc,
                            panel_nr_4,
                            nc_full_4,
                            &mut packed_b[pb_4_base..pb_4_base + kc * nc_full_4],
                        );
                    }
                    let pb_4_off = kc * nc_full_8;

                    if use_parallel {
                        // SAFETY: see sgemm above — disjoint row-panels,
                        // read-only borrows of `a`/`packed_b`, scope joins
                        // before either is mutated.
                        let c_send = SendMutPtr(c.as_mut_ptr());
                        let b_send = SendConstPtr(b.as_ptr());
                        let pb_send = SendConstPtr(packed_b.as_ptr());

                        rayon::scope(|s| {
                            let mut i0 = 0usize;
                            while i0 < m {
                                let mb = (m - i0).min(svl_d);
                                s.spawn(move |_| {
                                    // See sgemm for the disjoint-capture rationale.
                                    let c_send: SendMutPtr<f64> = c_send;
                                    let b_send: SendConstPtr<f64> = b_send;
                                    let pb_send: SendConstPtr<f64> = pb_send;
                                    with_local_pa(&DGEMM_LOCAL_PA, pa_need, |packed_a| {
                                        // SAFETY: see `process_row_panel_3tier` docs.
                                        unsafe {
                                            process_row_panel_3tier::<f64>(
                                                sonium_dgemm_panel8_sme2,
                                                sonium_dgemm_panel4_sme2,
                                                sonium_dgemm_panel_sme2,
                                                pack_a_fn,
                                                mb,
                                                i0,
                                                j_c,
                                                nc_full_8,
                                                nc_full_4,
                                                nc_end,
                                                p_start,
                                                kc,
                                                svl_d,
                                                panel_nr_8,
                                                panel_nr_4,
                                                alpha,
                                                beta_eff,
                                                lda,
                                                ldb,
                                                ldc,
                                                a,
                                                b_send.0,
                                                packed_a,
                                                pb_send.0,
                                                pb_send.0.add(pb_4_off),
                                                c_send.0,
                                            );
                                        }
                                    });
                                });
                                i0 += svl_d;
                            }
                        });
                    } else {
                        with_local_pa(&DGEMM_LOCAL_PA, pa_need, |packed_a| {
                            let c_ptr = c.as_mut_ptr();
                            let b_ptr = b.as_ptr();
                            let packed_b_8_ptr = packed_b.as_ptr();
                            // SAFETY: stays within the borrow of packed_b.
                            let packed_b_4_ptr = unsafe { packed_b_8_ptr.add(pb_4_off) };
                            let mut i0 = 0usize;
                            while i0 < m {
                                let mb = (m - i0).min(svl_d);
                                // SAFETY: bounds pre-checked; no aliasing.
                                unsafe {
                                    process_row_panel_3tier::<f64>(
                                        sonium_dgemm_panel8_sme2,
                                        sonium_dgemm_panel4_sme2,
                                        sonium_dgemm_panel_sme2,
                                        pack_a_fn,
                                        mb,
                                        i0,
                                        j_c,
                                        nc_full_8,
                                        nc_full_4,
                                        nc_end,
                                        p_start,
                                        kc,
                                        svl_d,
                                        panel_nr_8,
                                        panel_nr_4,
                                        alpha,
                                        beta_eff,
                                        lda,
                                        ldb,
                                        ldc,
                                        a,
                                        b_ptr,
                                        packed_a,
                                        packed_b_8_ptr,
                                        packed_b_4_ptr,
                                        c_ptr,
                                    );
                                }
                                i0 += svl_d;
                            }
                        });
                    }

                    p_start += kc;
                }
                j_c += nc;
            }
        });
    }
}

// ---- Pack helpers ----

// Pack a row-panel of A into a column-major sub-buffer that the FMOPA
// kernels consume: `packed[p * mr + r] = A[i0 + r, p_start + p]` for
// p in 0..kc. Rows r in mb..mr are zeroed so the FMOPA row predicate
// can stay `ptrue` (zero-padded rows contribute 0 to the outer product).
//
// NoTrans: stored A is m×k row-major with `A[i,p] = a[i*lda + p]`.
#[inline]
fn pack_a_panel_kc<T: Copy + num_traits::Zero>(
    a: &[T],
    lda: usize,
    i0: usize,
    mb: usize,
    p_start: usize,
    kc: usize,
    mr: usize,
    packed: &mut [T],
) {
    for p in 0..kc {
        let off = p * mr;
        for r in 0..mb {
            packed[off + r] = a[(i0 + r) * lda + p_start + p];
        }
        for r in mb..mr {
            packed[off + r] = T::zero();
        }
    }
}

// Trans variant: stored A is k×m row-major with `A^T[i,p] = a[p*lda + i]`.
// Produces the SAME packed layout as `pack_a_panel_kc`, so downstream
// kernels are unchanged — only the source indexing differs.
//
// Real types only here (f32/f64). ConjTrans on real == Trans (the conjugate
// is a no-op). For complex types the Trans/ConjTrans path stays on the
// scalar fallback in cpu/blas3/scalar.rs until task #16 (native complex
// FMOPA) lands; we'd then need to conjugate at pack time for ConjTrans,
// which is cheap but only meaningful with a complex micro-kernel.
#[inline]
fn pack_a_panel_kc_trans<T: Copy + num_traits::Zero>(
    a: &[T],
    lda: usize,
    i0: usize,
    mb: usize,
    p_start: usize,
    kc: usize,
    mr: usize,
    packed: &mut [T],
) {
    for p in 0..kc {
        let off = p * mr;
        let row_stored = p_start + p; // logical column of A^T → row of stored A
        let base = row_stored * lda;
        for r in 0..mb {
            packed[off + r] = a[base + i0 + r];
        }
        for r in mb..mr {
            packed[off + r] = T::zero();
        }
    }
}

// Pack a band of B into Nr-wide panels in panel-of-Nr layout — exactly
// what the 4-tile kernel reads with `ldb=panel_nr` and `mul vl` offsets.
//
// Layout:  for panel_idx pi, base = pi * (kc * panel_nr); row p occupies
// `packed[base + p*panel_nr .. base + (p+1)*panel_nr]` with values
// `B[p_start + p, j_offset + pi*panel_nr + 0..panel_nr]`.
#[inline]
fn pack_b_panels<T: Copy>(
    b: &[T],
    ldb: usize,
    p_start: usize,
    j_offset: usize,
    kc: usize,
    panel_nr: usize,
    n_full: usize,
    packed: &mut [T],
) {
    let panel_count = n_full / panel_nr;
    for panel_idx in 0..panel_count {
        let j0 = j_offset + panel_idx * panel_nr;
        let panel_off = panel_idx * kc * panel_nr;
        for p in 0..kc {
            let src = (p_start + p) * ldb + j0;
            let dst = panel_off + p * panel_nr;
            packed[dst..dst + panel_nr].copy_from_slice(&b[src..src + panel_nr]);
        }
    }
}

// `C := beta * C` over the (m, n) sub-matrix of `c` (stride `ldc`).
// beta == 1.0 is a no-op; beta == 0.0 zeroes (so we don't propagate NaNs
// from any uninitialized C the caller may have passed in).
fn scale_c<T>(m: usize, n: usize, beta: T, c: &mut [T], ldc: usize)
where
    T: Copy + PartialEq + num_traits::Zero + num_traits::One + core::ops::MulAssign,
{
    if beta == T::one() {
        return;
    }
    if beta == T::zero() {
        for i in 0..m {
            for j in 0..n {
                c[i * ldc + j] = T::zero();
            }
        }
    } else {
        for i in 0..m {
            for j in 0..n {
                c[i * ldc + j] *= beta;
            }
        }
    }
}

// ---- Scalar reference implementations (used on non-SME2 hosts) ----

// Non-SME2 fallback dispatch: the AVX2/FMA register-blocked kernel in
// `cpu::blas3::avx2` on x86_64 hosts that support it, the NEON
// register-blocked kernel in `cpu::blas3::neon` on aarch64, the scalar
// triple loop everywhere else. Numerical behaviour is identical up to
// FMA contraction/reassociation.
#[allow(clippy::too_many_arguments)]
fn sgemm_fallback(
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
) {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::blas3::avx2::is_available() {
        // SAFETY: AVX2+FMA detected above; matrix bounds were pre-checked
        // by the public entry points before dispatching here.
        return unsafe {
            crate::cpu::blas3::avx2::sgemm(
                m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_a, false,
            )
        };
    }
    #[cfg(target_arch = "aarch64")]
    if crate::cpu::blas3::neon::is_available() {
        // SAFETY: NEON is baseline on aarch64; bounds pre-checked by
        // the public entry points before dispatching here.
        return unsafe {
            crate::cpu::blas3::neon::sgemm(
                m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_a, false,
            )
        };
    }
    sgemm_scalar(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_a);
}

#[allow(clippy::too_many_arguments)]
fn dgemm_fallback(
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
) {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::blas3::avx2::is_available() {
        // SAFETY: AVX2+FMA detected above; bounds pre-checked by the caller.
        return unsafe {
            crate::cpu::blas3::avx2::dgemm(
                m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_a, false,
            )
        };
    }
    #[cfg(target_arch = "aarch64")]
    if crate::cpu::blas3::neon::is_available() {
        // SAFETY: as in `sgemm_fallback`.
        return unsafe {
            crate::cpu::blas3::neon::dgemm(
                m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_a, false,
            )
        };
    }
    dgemm_scalar(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_a);
}

#[allow(clippy::too_many_arguments)]
fn sgemm_scalar(
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
) {
    scale_c(m, n, beta, c, ldc);
    if alpha == 0.0 || k == 0 {
        return;
    }
    if trans_a {
        // stored A is k×m; A^T[i,p] = a[p*lda + i]
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0_f32;
                for p in 0..k {
                    sum += a[p * lda + i] * b[p * ldb + j];
                }
                c[i * ldc + j] += alpha * sum;
            }
        }
    } else {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0_f32;
                for p in 0..k {
                    sum += a[i * lda + p] * b[p * ldb + j];
                }
                c[i * ldc + j] += alpha * sum;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn dgemm_scalar(
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
) {
    scale_c(m, n, beta, c, ldc);
    if alpha == 0.0 || k == 0 {
        return;
    }
    if trans_a {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0_f64;
                for p in 0..k {
                    sum += a[p * lda + i] * b[p * ldb + j];
                }
                c[i * ldc + j] += alpha * sum;
            }
        }
    } else {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0_f64;
                for p in 0..k {
                    sum += a[i * lda + p] * b[p * ldb + j];
                }
                c[i * ldc + j] += alpha * sum;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::with_pack_buffer;
    use std::cell::RefCell;

    thread_local! {
        static TEST_PACK_BUFFER: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    }

    #[test]
    fn pack_buffer_is_reentrant_across_rayon_scope() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("single-thread test pool");

        pool.install(|| {
            with_pack_buffer(&TEST_PACK_BUFFER, 8, |outer| {
                outer.fill(1.0);
                let outer_ptr = outer.as_ptr() as usize;

                rayon::scope(|scope| {
                    scope.spawn(move |_| {
                        with_pack_buffer(&TEST_PACK_BUFFER, 4, |inner| {
                            assert_ne!(
                                inner.as_ptr() as usize,
                                outer_ptr,
                                "nested GEMM must not reuse the active outer allocation"
                            );
                            inner.fill(2.0);
                        });
                    });
                });

                assert_eq!(outer, &[1.0; 8]);
            });
        });
    }

    /// Nested-GEMM stress: an outer `rayon::join` runs parallel-path GEMMs
    /// (m above the parallel-M threshold, so every call fans out over
    /// `rayon::scope`). Workers waiting in an inner scope steal outer work
    /// onto the same thread, re-entering the thread-local pack buffers.
    /// Pre-fix that reentrancy panicked with "RefCell already borrowed";
    /// nested calls now fall back to a fresh allocation. Asserts no panic
    /// and parity with top-level calls.
    #[test]
    fn nested_gemm_stress_matches_top_level() {
        use super::{dgemm, sgemm};
        use approx::assert_relative_eq;

        let m = 96;
        let n = 80;
        let k = 72;

        let a32: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.013).sin()).collect();
        let b32: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.017).cos()).collect();
        let mut want32 = vec![0.0f32; m * n];
        sgemm(m, n, k, 1.0, &a32, k, &b32, n, 0.0, &mut want32, n, false);

        let a64: Vec<f64> = (0..m * k).map(|i| (i as f64 * 0.013).sin()).collect();
        let b64: Vec<f64> = (0..k * n).map(|i| (i as f64 * 0.017).cos()).collect();
        let mut want64 = vec![0.0f64; m * n];
        dgemm(m, n, k, 1.0, &a64, k, &b64, n, 0.0, &mut want64, n, false);

        for _ in 0..256 {
            let (mut c1, mut c2) = (vec![0.0f32; m * n], vec![0.0f32; m * n]);
            rayon::join(
                || sgemm(m, n, k, 1.0, &a32, k, &b32, n, 0.0, &mut c1, n, false),
                || sgemm(m, n, k, 1.0, &a32, k, &b32, n, 0.0, &mut c2, n, false),
            );
            for (got, w) in c1.iter().chain(&c2).zip(want32.iter().cycle()) {
                assert_relative_eq!(got, w, max_relative = 1e-5, epsilon = 1e-6);
            }

            let (mut d1, mut d2) = (vec![0.0f64; m * n], vec![0.0f64; m * n]);
            rayon::join(
                || dgemm(m, n, k, 1.0, &a64, k, &b64, n, 0.0, &mut d1, n, false),
                || dgemm(m, n, k, 1.0, &a64, k, &b64, n, 0.0, &mut d2, n, false),
            );
            for (got, w) in d1.iter().chain(&d2).zip(want64.iter().cycle()) {
                assert_relative_eq!(got, w, max_relative = 1e-12, epsilon = 1e-12);
            }
        }
    }
}
