//! Stockham auto-sort mixed-radix FFT driver.
//!
//! Operates on a single FFT of length `n = ∏ radices`. The driver
//! ping-pongs between two equal-size buffers; the input lives in the
//! user buffer on entry and the output is left in the user buffer on
//! exit, with at most one extra copy when the radix count is odd.
//!
//! The stage indexing scheme follows the standard mixed-radix
//! decimation-in-time Stockham factorisation:
//!
//! ```text
//!   for each radix r in radices:
//!       m_new = m · r           // sub-FFT size after this stage
//!       s     = n / m_new       // number of independent groups
//!       for g in 0..s:
//!           for l in 0..m:
//!               read  r-tuple at positions g·m + l + k·(n/r) for k ∈ [0,r)
//!               apply twiddle    w_N^(l·k·s)                   for k ∈ [1,r)
//!               apply radix-r DFT
//!               write r-tuple to positions g·m_new + a·m + l   for a ∈ [0,r)
//!       m = m_new
//! ```
//!
//! `w_N = exp(dir_sign · 2π i / N)` is the direction-signed root of
//! unity, materialised in [`crate::twiddles::TwiddleTable`].

#![allow(dead_code)]

use num_complex::Complex;
use std::cell::RefCell;

use crate::CpuFftDispatch;
use crate::codelets::scalar::{radix2, radix3, radix4, radix5, radix8};
use crate::scalar::FftScalar;
use crate::twiddles::StageTwiddles;

/// Assert the layout compatibility that every `transmute`-on-`TypeId`
/// dispatch below relies on: `TypeId` equality proves `T = U`, but the
/// transmutes additionally require identical size and alignment of both the
/// complex lanes and the twiddle tables. A future layout change to
/// [`StageTwiddles`] or [`Complex`] then fails here, at the call site,
/// instead of silently miscompiling downstream SIMD loads.
fn assert_transmute_layout<T: FftScalar, U: FftScalar>() {
    assert!(
        std::mem::size_of::<StageTwiddles<T>>() == std::mem::size_of::<StageTwiddles<U>>(),
        "StageTwiddles layout changed; transmute dispatch is unsound"
    );
    assert!(
        std::mem::align_of::<StageTwiddles<T>>() == std::mem::align_of::<StageTwiddles<U>>(),
        "StageTwiddles alignment changed; transmute dispatch is unsound"
    );
    assert!(
        std::mem::size_of::<Complex<T>>() == std::mem::size_of::<Complex<U>>(),
        "Complex layout changed; transmute dispatch is unsound"
    );
    assert!(
        std::mem::align_of::<Complex<T>>() == std::mem::align_of::<Complex<U>>(),
        "Complex alignment changed; transmute dispatch is unsound"
    );
}

thread_local! {
    static PAIR_TMP0_256: RefCell<Vec<Complex<f64>>> = const { RefCell::new(Vec::new()) };
    static PAIR_TMP1_256: RefCell<Vec<Complex<f64>>> = const { RefCell::new(Vec::new()) };
    static PAIR_SCRATCH_256: RefCell<Vec<Complex<f64>>> = const { RefCell::new(Vec::new()) };
    static PAIR_TMP0_1024: RefCell<Vec<Complex<f64>>> = const { RefCell::new(Vec::new()) };
    static PAIR_TMP1_1024: RefCell<Vec<Complex<f64>>> = const { RefCell::new(Vec::new()) };
    static PAIR_SCRATCH_1024: RefCell<Vec<Complex<f64>>> = const { RefCell::new(Vec::new()) };
    static PAIR_TMP0_4096: RefCell<Vec<Complex<f64>>> = const { RefCell::new(Vec::new()) };
    static PAIR_TMP1_4096: RefCell<Vec<Complex<f64>>> = const { RefCell::new(Vec::new()) };
    static PAIR_SCRATCH_4096: RefCell<Vec<Complex<f64>>> = const { RefCell::new(Vec::new()) };
    // Generic pair buffers cover every f64 plan whose stages are radix-2/4.
    // The fixed-size fused paths above remain useful for their extra stage
    // fusion, while this cache removes the old length whitelist from the
    // vector-across-batch fallback.
    static PAIR_TMP0_GENERIC: RefCell<Vec<Complex<f64>>> = const { RefCell::new(Vec::new()) };
    static PAIR_TMP1_GENERIC: RefCell<Vec<Complex<f64>>> = const { RefCell::new(Vec::new()) };
    static PAIR_SCRATCH_GENERIC: RefCell<Vec<Complex<f64>>> = const { RefCell::new(Vec::new()) };
    #[cfg(target_arch = "aarch64")]
    static PAIR_TMP0_GENERIC_F32: RefCell<Vec<Complex<f32>>> = const { RefCell::new(Vec::new()) };
    #[cfg(target_arch = "aarch64")]
    static PAIR_TMP1_GENERIC_F32: RefCell<Vec<Complex<f32>>> = const { RefCell::new(Vec::new()) };
    #[cfg(target_arch = "aarch64")]
    static PAIR_SCRATCH_GENERIC_F32: RefCell<Vec<Complex<f32>>> = const { RefCell::new(Vec::new()) };
    #[cfg(target_arch = "x86_64")]
    static PAIR_TMP0_AVX2: RefCell<Vec<Complex<f64>>> = const { RefCell::new(Vec::new()) };
    #[cfg(target_arch = "x86_64")]
    static PAIR_TMP1_AVX2: RefCell<Vec<Complex<f64>>> = const { RefCell::new(Vec::new()) };
    #[cfg(target_arch = "x86_64")]
    static PAIR_SCRATCH_AVX2: RefCell<Vec<Complex<f64>>> = const { RefCell::new(Vec::new()) };
    #[cfg(target_arch = "x86_64")]
    static PAIR_TMP0_AVX2_F32: RefCell<Vec<Complex<f32>>> = const { RefCell::new(Vec::new()) };
    #[cfg(target_arch = "x86_64")]
    static PAIR_TMP1_AVX2_F32: RefCell<Vec<Complex<f32>>> = const { RefCell::new(Vec::new()) };
    #[cfg(target_arch = "x86_64")]
    static PAIR_SCRATCH_AVX2_F32: RefCell<Vec<Complex<f32>>> = const { RefCell::new(Vec::new()) };
}

/// A planner-time group of consecutive Stockham stages.
///
/// Sections keep the intermediate sub-FFT working set bounded while a plan
/// walks through its stages. They are metadata only at execution time: the
/// ranges are borrowed from the plan and never allocated on the hot path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StageSection {
    pub(crate) first_stage: usize,
    pub(crate) end_stage: usize,
    pub(crate) working_set_bytes: usize,
}

const AARCH64_L1D_BYTES: usize = 128 * 1024;
const X86_L1D_BYTES: usize = 32 * 1024;
const PORTABLE_SECTION_BYTES: usize = 32 * 1024;

/// Return the conservative per-plan cache budget used for Stockham sections.
///
/// The value follows the documented M4 L1D size on AArch64 and a conservative
/// 32 KiB L1D budget elsewhere. It is deliberately a planner policy rather
/// than a compile-time requirement; future device-specific autotuning can
/// replace this function without changing the stage contract.
#[inline]
pub(crate) fn section_cache_budget_bytes() -> usize {
    if cfg!(target_arch = "aarch64") {
        AARCH64_L1D_BYTES
    } else if cfg!(target_arch = "x86_64") {
        X86_L1D_BYTES
    } else {
        PORTABLE_SECTION_BYTES
    }
}

/// Build contiguous stage sections from the mixed-radix plan.
///
/// A section grows while the largest intermediate sub-FFT footprint remains
/// within the architecture budget. A single stage larger than the budget is
/// retained as an oversized section so every stage remains represented and
/// execution never changes the transform graph.
pub(crate) fn plan_stage_sections<T: FftScalar>(
    n: usize,
    stages: &[StageTwiddles<T>],
) -> Vec<StageSection> {
    if stages.is_empty() {
        return Vec::new();
    }
    let budget = section_cache_budget_bytes();
    let element_bytes = std::mem::size_of::<Complex<T>>();
    let mut sections = Vec::new();
    let mut first_stage = 0;
    let mut m = 1usize;
    let mut section_working_set = 0usize;

    for (stage_index, stage_tw) in stages.iter().enumerate() {
        m = m.saturating_mul(stage_tw.radix);
        let working_set_bytes = m
            .saturating_mul(2)
            .saturating_mul(element_bytes)
            .min(n.saturating_mul(2).saturating_mul(element_bytes));
        if stage_index > first_stage && working_set_bytes > budget {
            sections.push(StageSection {
                first_stage,
                end_stage: stage_index,
                working_set_bytes: section_working_set,
            });
            first_stage = stage_index;
            section_working_set = 0;
        }
        section_working_set = section_working_set.max(working_set_bytes);
    }

    sections.push(StageSection {
        first_stage,
        end_stage: stages.len(),
        working_set_bytes: section_working_set,
    });
    sections
}

/// Run the Stockham FFT for a single length-`n` transform.
///
/// `buffer.len() == scratch.len() == n` must hold and `n` must equal the
/// product of `radices`. On exit `buffer` holds the transform and the
/// content of `scratch` is unspecified.
pub fn run_single<T: FftScalar>(
    stages: &[StageTwiddles<T>],
    dir_sign: T,
    buffer: &mut [Complex<T>],
    scratch: &mut [Complex<T>],
) {
    run_single_with_dispatch(stages, dir_sign, buffer, scratch, CpuFftDispatch::Auto);
}

pub(crate) fn run_single_with_dispatch<T: FftScalar>(
    stages: &[StageTwiddles<T>],
    dir_sign: T,
    buffer: &mut [Complex<T>],
    scratch: &mut [Complex<T>],
    dispatch: CpuFftDispatch,
) {
    run_single_grouped(stages, dir_sign, buffer, scratch, &[], dispatch);
}

fn run_single_grouped<T: FftScalar>(
    stages: &[StageTwiddles<T>],
    dir_sign: T,
    buffer: &mut [Complex<T>],
    scratch: &mut [Complex<T>],
    sections: &[StageSection],
    dispatch: CpuFftDispatch,
) {
    let n = buffer.len();
    assert_eq!(
        scratch.len(),
        n,
        "Stockham scratch length must equal buffer length"
    );
    if n <= 1 || stages.is_empty() {
        return;
    }
    assert_eq!(
        stages.iter().map(|s| s.radix).product::<usize>(),
        n,
        "Stockham stage radices must multiply to buffer length"
    );

    // Choose where the *first* stage reads from so the *last* stage
    // writes into `buffer`. Odd stage count => start in scratch.
    let parity = stages.len() % 2;
    let (mut src, mut dst): (&mut [Complex<T>], &mut [Complex<T>]) = if parity == 0 {
        (buffer, scratch)
    } else {
        scratch.copy_from_slice(buffer);
        (scratch, buffer)
    };

    if sections.is_empty() {
        for stage_tw in stages {
            stage(n, src, dst, stage_tw, dir_sign, dispatch);
            std::mem::swap(&mut src, &mut dst);
        }
    } else {
        debug_assert_eq!(sections.first().map(|s| s.first_stage), Some(0));
        debug_assert_eq!(sections.last().map(|s| s.end_stage), Some(stages.len()));
        for section in sections {
            debug_assert!(section.first_stage < section.end_stage);
            for stage_tw in &stages[section.first_stage..section.end_stage] {
                stage(n, src, dst, stage_tw, dir_sign, dispatch);
                std::mem::swap(&mut src, &mut dst);
            }
        }
    }
    // After the last swap, `src` holds the output; for a correctly
    // chosen parity it aliases `buffer`. The `dst` view (scratch) is
    // discarded.
    debug_assert!(std::ptr::eq(src.as_ptr(), buffer.as_ptr()));
}

/// Process a batched transform — a flat buffer of `batch · n` complex
/// elements, each contiguous `n`-chunk is one FFT.
pub fn run_batched<T: FftScalar>(
    stages: &[StageTwiddles<T>],
    dir_sign: T,
    n: usize,
    buffer: &mut [Complex<T>],
    scratch: &mut [Complex<T>],
) {
    run_batched_with_sections(stages, dir_sign, n, buffer, scratch, &[]);
}

pub(crate) fn run_batched_with_sections<T: FftScalar>(
    stages: &[StageTwiddles<T>],
    dir_sign: T,
    n: usize,
    buffer: &mut [Complex<T>],
    scratch: &mut [Complex<T>],
    sections: &[StageSection],
) {
    run_batched_with_dispatch(
        stages,
        dir_sign,
        n,
        buffer,
        scratch,
        sections,
        CpuFftDispatch::Auto,
    );
}

pub(crate) fn run_batched_with_dispatch<T: FftScalar>(
    stages: &[StageTwiddles<T>],
    dir_sign: T,
    n: usize,
    buffer: &mut [Complex<T>],
    scratch: &mut [Complex<T>],
    sections: &[StageSection],
    dispatch: CpuFftDispatch,
) {
    debug_assert!(scratch.len() >= n);
    if n == 0 {
        return;
    }
    assert!(scratch.len() >= n, "Stockham scratch buffer is too short");
    assert_eq!(
        buffer.len() % n,
        0,
        "Stockham buffer length must be a multiple of plan length"
    );
    let scratch = &mut scratch[..n];

    if try_run_batched_f64_pairs(stages, dir_sign, n, buffer, scratch, dispatch) {
        return;
    }

    if try_run_batched_f64_audio(stages, dir_sign, n, buffer, scratch, dispatch) {
        return;
    }

    if try_run_batched_f32_audio(stages, dir_sign, n, buffer, scratch, dispatch) {
        return;
    }

    for chunk in buffer.chunks_exact_mut(n) {
        run_single_grouped(stages, dir_sign, chunk, scratch, sections, dispatch);
    }
}

#[inline(always)]
fn try_run_batched_f64_audio<T: FftScalar>(
    stages: &[StageTwiddles<T>],
    dir_sign: T,
    n: usize,
    buffer: &mut [Complex<T>],
    scratch: &mut [Complex<T>],
    dispatch: CpuFftDispatch,
) -> bool {
    if !dispatch.allows_simd()
        || !matches!(n, 64 | 128 | 256 | 512 | 1024 | 16384)
        || std::any::TypeId::of::<T>() != std::any::TypeId::of::<f64>()
        || !stages.iter().all(|s| matches!(s.radix, 2 | 4))
    {
        return false;
    }

    // Statically assert layout compatibility so the transmutes below remain
    // sound if StageTwiddles or Complex ever change their representations.
    assert!(std::mem::size_of::<StageTwiddles<T>>() == std::mem::size_of::<StageTwiddles<f64>>());
    assert!(std::mem::align_of::<StageTwiddles<T>>() == std::mem::align_of::<StageTwiddles<f64>>());
    assert!(std::mem::size_of::<Complex<T>>() == std::mem::size_of::<Complex<f64>>());
    assert!(std::mem::align_of::<Complex<T>>() == std::mem::align_of::<Complex<f64>>());

    unsafe {
        let stages_f64: &[StageTwiddles<f64>] =
            std::mem::transmute::<&[StageTwiddles<T>], &[StageTwiddles<f64>]>(stages);
        let buffer_f64: &mut [Complex<f64>] =
            std::mem::transmute::<&mut [Complex<T>], &mut [Complex<f64>]>(buffer);
        let scratch_f64: &mut [Complex<f64>] =
            std::mem::transmute::<&mut [Complex<T>], &mut [Complex<f64>]>(scratch);
        let dir_sign_f64 = *(&dir_sign as *const T as *const f64);
        for chunk in buffer_f64.chunks_exact_mut(n) {
            match n {
                64 => run_single_f64_64(stages_f64, dir_sign_f64, chunk, scratch_f64),
                128 => run_single_f64_128(stages_f64, dir_sign_f64, chunk, scratch_f64),
                256 => run_single_f64_256(stages_f64, dir_sign_f64, chunk, scratch_f64),
                512 => run_single_f64_512(stages_f64, dir_sign_f64, chunk, scratch_f64),
                1024 => run_single_f64_1024(stages_f64, dir_sign_f64, chunk, scratch_f64),
                16384 => run_single_f64_16384(stages_f64, dir_sign_f64, chunk, scratch_f64),
                _ => run_single_f64_audio(stages_f64, dir_sign_f64, chunk, scratch_f64),
            }
        }
    }
    true
}

/// Fused single-transform fast path for f32 audio sizes on x86 AVX2:
/// fused first pair + generic middle stage(s) + fused final pair. Pure
/// radix-2/4 lengths only; everything else keeps the generic driver.
#[inline(always)]
fn try_run_batched_f32_audio<T: FftScalar>(
    stages: &[StageTwiddles<T>],
    dir_sign: T,
    n: usize,
    buffer: &mut [Complex<T>],
    scratch: &mut [Complex<T>],
    dispatch: CpuFftDispatch,
) -> bool {
    if !dispatch.allows_simd()
        || !cfg!(target_arch = "x86_64")
        || !matches!(n, 512 | 1024)
        || std::any::TypeId::of::<T>() != std::any::TypeId::of::<f32>()
        || !stages.iter().all(|s| matches!(s.radix, 2 | 4))
    {
        return false;
    }

    assert_transmute_layout::<T, f32>();
    // SAFETY: TypeId equality proves T = f32; the layout asserts above pin
    // the transmute targets.
    unsafe {
        let stages_f32: &[StageTwiddles<f32>] =
            std::mem::transmute::<&[StageTwiddles<T>], &[StageTwiddles<f32>]>(stages);
        let buffer_f32: &mut [Complex<f32>] =
            std::mem::transmute::<&mut [Complex<T>], &mut [Complex<f32>]>(buffer);
        let scratch_f32: &mut [Complex<f32>] =
            std::mem::transmute::<&mut [Complex<T>], &mut [Complex<f32>]>(scratch);
        let dir_sign_f32 = *(&dir_sign as *const T as *const f32);
        for chunk in buffer_f32.chunks_exact_mut(n) {
            match n {
                512 => run_single_f32_512(stages_f32, dir_sign_f32, chunk, scratch_f32, dispatch),
                _ => run_single_f32_1024(stages_f32, dir_sign_f32, chunk, scratch_f32, dispatch),
            }
        }
    }
    true
}

#[inline(always)]
fn run_single_f32_512(
    stages: &[StageTwiddles<f32>],
    dir_sign: f32,
    buffer: &mut [Complex<f32>],
    scratch: &mut [Complex<f32>],
    dispatch: CpuFftDispatch,
) {
    debug_assert_eq!(buffer.len(), 512);
    debug_assert_eq!(stages.len(), 5);
    // Three-pass AVX2 sequence: fused first pair, one generic stage,
    // fused final pair. Falls back to copy + three stages without AVX2.
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        // SAFETY: runtime detection proves the target features required
        // by the AVX2/FMA kernels.
        unsafe {
            crate::wide_avx2::fused_first2_f32(buffer, scratch, &stages[0], &stages[1], dir_sign);
        }
        stage(512, scratch, buffer, &stages[2], dir_sign, dispatch);
        // SAFETY: same detection as above.
        unsafe {
            crate::wide_avx2::fused_final_pair_f32(buffer, &stages[3], &stages[4], dir_sign);
        }
        return;
    }
    // Odd stage count: start in scratch so the fused final pair lands in
    // `buffer` (same parity convention as the generic driver).
    scratch.copy_from_slice(buffer);
    stage(512, scratch, buffer, &stages[0], dir_sign, dispatch);
    stage(512, buffer, scratch, &stages[1], dir_sign, dispatch);
    stage(512, scratch, buffer, &stages[2], dir_sign, dispatch);
    // SAFETY: the caller gated on AVX2 + FMA detection.
    unsafe {
        crate::wide_avx2::fused_final_pair_f32(buffer, &stages[3], &stages[4], dir_sign);
    }
}

#[inline(always)]
fn run_single_f32_1024(
    stages: &[StageTwiddles<f32>],
    dir_sign: f32,
    buffer: &mut [Complex<f32>],
    scratch: &mut [Complex<f32>],
    dispatch: CpuFftDispatch,
) {
    debug_assert_eq!(buffer.len(), 1024);
    debug_assert_eq!(stages.len(), 5);
    // Three-pass AVX2 sequence: fused first pair, one generic stage,
    // fused final pair. Falls back to copy + three stages without AVX2.
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        // SAFETY: runtime detection proves the target features required
        // by the AVX2/FMA kernels.
        unsafe {
            crate::wide_avx2::fused_first2_f32(buffer, scratch, &stages[0], &stages[1], dir_sign);
        }
        stage(1024, scratch, buffer, &stages[2], dir_sign, dispatch);
        // SAFETY: same detection as above.
        unsafe {
            crate::wide_avx2::fused_final_pair_f32(buffer, &stages[3], &stages[4], dir_sign);
        }
        return;
    }
    scratch.copy_from_slice(buffer);
    stage(1024, scratch, buffer, &stages[0], dir_sign, dispatch);
    stage(1024, buffer, scratch, &stages[1], dir_sign, dispatch);
    stage(1024, scratch, buffer, &stages[2], dir_sign, dispatch);
    // SAFETY: the caller gated on AVX2 + FMA detection.
    unsafe {
        crate::wide_avx2::fused_final_pair_f32(buffer, &stages[3], &stages[4], dir_sign);
    }
}

#[inline(always)]
fn try_run_batched_f64_pairs<T: FftScalar>(
    stages: &[StageTwiddles<T>],
    dir_sign: T,
    n: usize,
    buffer: &mut [Complex<T>],
    _scratch: &mut [Complex<T>],
    dispatch: CpuFftDispatch,
) -> bool {
    let batch = buffer.len() / n;
    if !dispatch.allows_simd()
        || batch < 2
        || n < 2
        || !stages.iter().all(|s| matches!(s.radix, 2 | 4))
    {
        return false;
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            assert_transmute_layout::<T, f64>();
            let stages_f64: &[StageTwiddles<f64>] =
                std::mem::transmute::<&[StageTwiddles<T>], &[StageTwiddles<f64>]>(stages);
            let buffer_f64: &mut [Complex<f64>] =
                std::mem::transmute::<&mut [Complex<T>], &mut [Complex<f64>]>(buffer);
            let dir_sign_f64 = *(&dir_sign as *const T as *const f64);
            if n == 256 {
                run_batched_f64_256_pairs(stages_f64, dir_sign_f64, buffer_f64);
                return true;
            }
            if n == 1024 {
                run_batched_f64_1024_pairs(stages_f64, dir_sign_f64, buffer_f64);
                return true;
            }
            if n == 4096 {
                run_batched_f64_4096_pairs(stages_f64, dir_sign_f64, buffer_f64);
                return true;
            }
            run_batched_f64_generic_pairs(stages_f64, dir_sign_f64, n, buffer_f64);
            return true;
        }
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            assert_transmute_layout::<T, f32>();
            let stages_f32: &[StageTwiddles<f32>] =
                std::mem::transmute::<&[StageTwiddles<T>], &[StageTwiddles<f32>]>(stages);
            let buffer_f32: &mut [Complex<f32>] =
                std::mem::transmute::<&mut [Complex<T>], &mut [Complex<f32>]>(buffer);
            let dir_sign_f32 = *(&dir_sign as *const T as *const f32);
            run_batched_f32_neon_pairs(stages_f32, dir_sign_f32, n, buffer_f32);
            return true;
        }
        false
    }

    #[cfg(target_arch = "x86_64")]
    if dispatch.allows_simd()
        && std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
    {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            assert_transmute_layout::<T, f64>();
            unsafe {
                let stages_f64: &[StageTwiddles<f64>] =
                    std::mem::transmute::<&[StageTwiddles<T>], &[StageTwiddles<f64>]>(stages);
                let buffer_f64: &mut [Complex<f64>] =
                    std::mem::transmute::<&mut [Complex<T>], &mut [Complex<f64>]>(buffer);
                run_batched_f64_avx2_pairs(stages_f64, dir_sign_f64::<T>(dir_sign), n, buffer_f64);
                return true;
            }
        }
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            assert_transmute_layout::<T, f32>();
            unsafe {
                let stages_f32: &[StageTwiddles<f32>] =
                    std::mem::transmute::<&[StageTwiddles<T>], &[StageTwiddles<f32>]>(stages);
                let buffer_f32: &mut [Complex<f32>] =
                    std::mem::transmute::<&mut [Complex<T>], &mut [Complex<f32>]>(buffer);
                run_batched_f32_avx2_pairs(stages_f32, dir_sign_f32::<T>(dir_sign), n, buffer_f32);
                return true;
            }
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (stages, dir_sign, n, buffer, _scratch);
    }

    #[cfg(not(target_arch = "aarch64"))]
    false
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn dir_sign_f64<T: FftScalar>(dir_sign: T) -> f64 {
    // The caller has already established that T is f64 by TypeId above.
    unsafe { *(&dir_sign as *const T as *const f64) }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn dir_sign_f32<T: FftScalar>(dir_sign: T) -> f32 {
    // The caller has already established that T is f32 by TypeId above.
    unsafe { *(&dir_sign as *const T as *const f32) }
}

/// Process f64 batches in pairs with AVX2 lanes representing the two
/// independent transforms. The temporary interleaving is kept in thread-local
/// buffers so the steady-state path remains allocation-free.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn run_batched_f64_avx2_pairs(
    stages: &[StageTwiddles<f64>],
    dir_sign: f64,
    n: usize,
    buffer: &mut [Complex<f64>],
) {
    let batch = buffer.len() / n;
    PAIR_TMP0_AVX2.with(|tmp0_cell| {
        PAIR_TMP1_AVX2.with(|tmp1_cell| {
            PAIR_SCRATCH_AVX2.with(|scratch_cell| {
                let mut tmp0_buf = tmp0_cell.borrow_mut();
                let mut tmp1_buf = tmp1_cell.borrow_mut();
                let mut scratch_buf = scratch_cell.borrow_mut();
                tmp0_buf.resize(2 * n, Complex::new(0.0, 0.0));
                tmp1_buf.resize(2 * n, Complex::new(0.0, 0.0));
                scratch_buf.resize(n, Complex::new(0.0, 0.0));
                let tmp0 = &mut tmp0_buf[..2 * n];
                let tmp1 = &mut tmp1_buf[..2 * n];

                for pair_idx in 0..batch / 2 {
                    let base0 = 2 * pair_idx * n;
                    let base1 = base0 + n;
                    for l in 0..n {
                        tmp0[2 * l] = buffer[base0 + l];
                        tmp0[2 * l + 1] = buffer[base1 + l];
                    }

                    let mut src_is_tmp0 = true;
                    for stage_tw in stages {
                        unsafe {
                            if src_is_tmp0 {
                                if stage_tw.radix == 2 {
                                    crate::wide_avx2::stage_radix2_f64_batched2(
                                        n / stage_tw.radix,
                                        &*tmp0,
                                        tmp1,
                                        stage_tw,
                                    );
                                } else {
                                    crate::wide_avx2::stage_radix4_f64_batched2(
                                        n / stage_tw.radix,
                                        &*tmp0,
                                        tmp1,
                                        stage_tw,
                                        dir_sign,
                                    );
                                }
                            } else if stage_tw.radix == 2 {
                                crate::wide_avx2::stage_radix2_f64_batched2(
                                    n / stage_tw.radix,
                                    &*tmp1,
                                    tmp0,
                                    stage_tw,
                                );
                            } else {
                                crate::wide_avx2::stage_radix4_f64_batched2(
                                    n / stage_tw.radix,
                                    &*tmp1,
                                    tmp0,
                                    stage_tw,
                                    dir_sign,
                                );
                            }
                        }
                        src_is_tmp0 = !src_is_tmp0;
                    }

                    let result = if src_is_tmp0 { &*tmp0 } else { &*tmp1 };
                    for l in 0..n {
                        buffer[base0 + l] = result[2 * l];
                        buffer[base1 + l] = result[2 * l + 1];
                    }
                }

                if batch % 2 == 1 {
                    let last = &mut buffer[(batch - 1) * n..batch * n];
                    run_single(stages, dir_sign, last, &mut scratch_buf[..n]);
                }
            });
        });
    });
}

/// Process f32 batches in pairs with AVX2 lanes representing the two
/// independent transforms. The interleaving is kept in thread-local buffers
/// so steady-state execution remains allocation-free after warmup.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn run_batched_f32_avx2_pairs(
    stages: &[StageTwiddles<f32>],
    dir_sign: f32,
    n: usize,
    buffer: &mut [Complex<f32>],
) {
    let batch = buffer.len() / n;
    PAIR_TMP0_AVX2_F32.with(|tmp0_cell| {
        PAIR_TMP1_AVX2_F32.with(|tmp1_cell| {
            PAIR_SCRATCH_AVX2_F32.with(|scratch_cell| {
                let mut tmp0_buf = tmp0_cell.borrow_mut();
                let mut tmp1_buf = tmp1_cell.borrow_mut();
                let mut scratch_buf = scratch_cell.borrow_mut();
                tmp0_buf.resize(2 * n, Complex::new(0.0, 0.0));
                tmp1_buf.resize(2 * n, Complex::new(0.0, 0.0));
                scratch_buf.resize(n, Complex::new(0.0, 0.0));
                let tmp0 = &mut tmp0_buf[..2 * n];
                let tmp1 = &mut tmp1_buf[..2 * n];

                for pair_idx in 0..batch / 2 {
                    let base0 = 2 * pair_idx * n;
                    let base1 = base0 + n;
                    for l in 0..n {
                        tmp0[2 * l] = buffer[base0 + l];
                        tmp0[2 * l + 1] = buffer[base1 + l];
                    }

                    let mut src_is_tmp0 = true;
                    for stage_tw in stages {
                        unsafe {
                            if src_is_tmp0 {
                                if stage_tw.radix == 2 {
                                    crate::wide_avx2::stage_radix2_f32_batched2(
                                        n / stage_tw.radix,
                                        &*tmp0,
                                        tmp1,
                                        stage_tw,
                                    );
                                } else {
                                    crate::wide_avx2::stage_radix4_f32_batched2(
                                        n / stage_tw.radix,
                                        &*tmp0,
                                        tmp1,
                                        stage_tw,
                                        dir_sign,
                                    );
                                }
                            } else if stage_tw.radix == 2 {
                                crate::wide_avx2::stage_radix2_f32_batched2(
                                    n / stage_tw.radix,
                                    &*tmp1,
                                    tmp0,
                                    stage_tw,
                                );
                            } else {
                                crate::wide_avx2::stage_radix4_f32_batched2(
                                    n / stage_tw.radix,
                                    &*tmp1,
                                    tmp0,
                                    stage_tw,
                                    dir_sign,
                                );
                            }
                        }
                        src_is_tmp0 = !src_is_tmp0;
                    }

                    let result = if src_is_tmp0 { &*tmp0 } else { &*tmp1 };
                    for l in 0..n {
                        buffer[base0 + l] = result[2 * l];
                        buffer[base1 + l] = result[2 * l + 1];
                    }
                }

                if batch % 2 == 1 {
                    let last = &mut buffer[(batch - 1) * n..batch * n];
                    run_single(stages, dir_sign, last, &mut scratch_buf[..n]);
                }
            });
        });
    });
}

/// Process f32 batches in pairs with AArch64 NEON lanes representing the two
/// independent transforms. Thread-local scratch keeps the steady-state path
/// allocation-free after the first call on each worker thread.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn run_batched_f32_neon_pairs(
    stages: &[StageTwiddles<f32>],
    dir_sign: f32,
    n: usize,
    buffer: &mut [Complex<f32>],
) {
    let batch = buffer.len() / n;
    PAIR_TMP0_GENERIC_F32.with(|tmp0_cell| {
        PAIR_TMP1_GENERIC_F32.with(|tmp1_cell| {
            PAIR_SCRATCH_GENERIC_F32.with(|scratch_cell| {
                let mut tmp0_buf = tmp0_cell.borrow_mut();
                let mut tmp1_buf = tmp1_cell.borrow_mut();
                let mut scratch_buf = scratch_cell.borrow_mut();
                tmp0_buf.resize(2 * n, Complex::new(0.0, 0.0));
                tmp1_buf.resize(2 * n, Complex::new(0.0, 0.0));
                scratch_buf.resize(n, Complex::new(0.0, 0.0));
                let tmp0 = &mut tmp0_buf[..2 * n];
                let tmp1 = &mut tmp1_buf[..2 * n];

                for pair_idx in 0..batch / 2 {
                    let base0 = 2 * pair_idx * n;
                    let base1 = base0 + n;
                    for l in 0..n {
                        tmp0[2 * l] = buffer[base0 + l];
                        tmp0[2 * l + 1] = buffer[base1 + l];
                    }

                    let mut src_is_tmp0 = true;
                    for stage_tw in stages {
                        unsafe {
                            if src_is_tmp0 {
                                if stage_tw.radix == 2 {
                                    crate::wide_neon::stage_radix2_f32_batched2(
                                        n / stage_tw.radix,
                                        &*tmp0,
                                        tmp1,
                                        stage_tw,
                                    );
                                } else {
                                    crate::wide_neon::stage_radix4_f32_batched2(
                                        n / stage_tw.radix,
                                        &*tmp0,
                                        tmp1,
                                        stage_tw,
                                        dir_sign,
                                    );
                                }
                            } else if stage_tw.radix == 2 {
                                crate::wide_neon::stage_radix2_f32_batched2(
                                    n / stage_tw.radix,
                                    &*tmp1,
                                    tmp0,
                                    stage_tw,
                                );
                            } else {
                                crate::wide_neon::stage_radix4_f32_batched2(
                                    n / stage_tw.radix,
                                    &*tmp1,
                                    tmp0,
                                    stage_tw,
                                    dir_sign,
                                );
                            }
                        }
                        src_is_tmp0 = !src_is_tmp0;
                    }

                    let result = if src_is_tmp0 { &*tmp0 } else { &*tmp1 };
                    for l in 0..n {
                        buffer[base0 + l] = result[2 * l];
                        buffer[base1 + l] = result[2 * l + 1];
                    }
                }

                if batch % 2 == 1 {
                    let last = &mut buffer[(batch - 1) * n..batch * n];
                    run_single(stages, dir_sign, last, &mut scratch_buf[..n]);
                }
            });
        });
    });
}

/// Process all radix-2/4 f64 batches in pairs on AArch64.
///
/// The fixed-size routines below fuse stage groups for common audio sizes.
/// This generic path handles the rest of the power-of-two family by using the
/// already-audited interleaved pair stages. The first transpose makes the two
/// independent FFTs SIMD lanes; the final transpose restores the public
/// contiguous-batch layout. Temporary buffers live in thread-local caches, so
/// steady-state execution remains allocation-free.
#[inline(always)]
#[cfg(target_arch = "aarch64")]
fn run_batched_f64_generic_pairs(
    stages: &[StageTwiddles<f64>],
    dir_sign: f64,
    n: usize,
    buffer: &mut [Complex<f64>],
) {
    let batch = buffer.len() / n;
    debug_assert!(n >= 2);
    debug_assert_eq!(buffer.len() % n, 0);

    PAIR_TMP0_GENERIC.with(|tmp0_cell| {
        PAIR_TMP1_GENERIC.with(|tmp1_cell| {
            PAIR_SCRATCH_GENERIC.with(|scratch_cell| {
                let mut tmp0_buf = tmp0_cell.borrow_mut();
                let mut tmp1_buf = tmp1_cell.borrow_mut();
                let mut scratch_buf = scratch_cell.borrow_mut();
                if tmp0_buf.len() < 2 * n {
                    tmp0_buf.resize(2 * n, Complex::new(0.0, 0.0));
                }
                if tmp1_buf.len() < 2 * n {
                    tmp1_buf.resize(2 * n, Complex::new(0.0, 0.0));
                }
                if scratch_buf.len() < n {
                    scratch_buf.resize(n, Complex::new(0.0, 0.0));
                }
                let tmp0 = &mut tmp0_buf[..2 * n];
                let tmp1 = &mut tmp1_buf[..2 * n];

                for pair_idx in 0..(batch / 2) {
                    let i0 = pair_idx * 2;
                    let i1 = i0 + 1;
                    let (left, right) = buffer.split_at_mut(i1 * n);
                    let src0 = &left[i0 * n..i0 * n + n];
                    let src1 = &right[..n];
                    unsafe {
                        crate::wide_neon::transpose_pair_in(src0, src1, tmp0);
                    }

                    let mut src_is_tmp0 = true;
                    for stage_tw in stages {
                        unsafe {
                            if src_is_tmp0 {
                                if stage_tw.radix == 2 {
                                    crate::wide_neon::stage_radix2_f64_batched2(
                                        n / stage_tw.radix,
                                        &*tmp0,
                                        tmp1,
                                        stage_tw,
                                    );
                                } else {
                                    crate::wide_neon::stage_radix4_f64_batched2(
                                        n / stage_tw.radix,
                                        &*tmp0,
                                        tmp1,
                                        stage_tw,
                                        dir_sign,
                                    );
                                }
                            } else if stage_tw.radix == 2 {
                                crate::wide_neon::stage_radix2_f64_batched2(
                                    n / stage_tw.radix,
                                    &*tmp1,
                                    tmp0,
                                    stage_tw,
                                );
                            } else {
                                crate::wide_neon::stage_radix4_f64_batched2(
                                    n / stage_tw.radix,
                                    &*tmp1,
                                    tmp0,
                                    stage_tw,
                                    dir_sign,
                                );
                            }
                        }
                        src_is_tmp0 = !src_is_tmp0;
                    }

                    let (left, right) = buffer.split_at_mut(i1 * n);
                    let dst0 = &mut left[i0 * n..i0 * n + n];
                    let dst1 = &mut right[..n];
                    unsafe {
                        if src_is_tmp0 {
                            crate::wide_neon::transpose_pair_out(&*tmp0, dst0, dst1);
                        } else {
                            crate::wide_neon::transpose_pair_out(&*tmp1, dst0, dst1);
                        }
                    }
                }

                if batch % 2 == 1 {
                    let last = &mut buffer[(batch - 1) * n..];
                    run_single(stages, dir_sign, last, &mut scratch_buf[..n]);
                }
            });
        });
    });
}

#[inline(always)]
#[cfg(target_arch = "aarch64")]
fn run_batched_f64_256_pairs(
    stages: &[StageTwiddles<f64>],
    dir_sign: f64,
    buffer: &mut [Complex<f64>],
) {
    let n = 256usize;
    let batch = buffer.len() / n;
    debug_assert_eq!(buffer.len() % n, 0);
    debug_assert_eq!(stages.len(), 4);
    debug_assert_eq!(stages.iter().map(|s| s.radix).product::<usize>(), n);

    PAIR_TMP0_256.with(|c0| {
        PAIR_TMP1_256.with(|c1| {
            let mut tmp0_buf = c0.borrow_mut();
            let mut tmp1_buf = c1.borrow_mut();
            if tmp0_buf.len() < 2 * n {
                tmp0_buf.resize(2 * n, Complex::new(0.0, 0.0));
            }
            if tmp1_buf.len() < 2 * n {
                tmp1_buf.resize(2 * n, Complex::new(0.0, 0.0));
            }
            let tmp0 = &mut tmp0_buf[..2 * n];
            let tmp1 = &mut tmp1_buf[..2 * n];

            for pair_idx in 0..(batch / 2) {
                let i0 = pair_idx * 2;
                let i1 = pair_idx * 2 + 1;

                unsafe {
                    // Stage 0 (m=1, radix-4) fused with transpose-in to planar
                    {
                        let (left, right) = buffer.split_at_mut(i1 * n);
                        let src0 = &left[i0 * n..i0 * n + n];
                        let src1 = &right[..n];
                        crate::wide_neon::fused_first_stage_radix4_m1_transpose_in_pair_planar(
                            src0, src1, tmp0, dir_sign,
                        );
                    }

                    // Stage 1 (m=4, radix-4) on planar data
                    crate::wide_neon::stage_radix4_f64_batched2_planar(
                        64, &*tmp0, tmp1, &stages[1], dir_sign,
                    );

                    // Stages 2+3 fused with transpose-out back to original chunks
                    {
                        let (left, right) = buffer.split_at_mut(i1 * n);
                        let dst0 = &mut left[i0 * n..i0 * n + n];
                        let dst1 = &mut right[..n];
                        crate::wide_neon::fused_final_radix4x4_256_transpose_out_pair_planar(
                            &*tmp1, &stages[2], &stages[3], dir_sign, dst0, dst1,
                        );
                    }
                }
            }
        });
    });

    // If batch is odd, process the last FFT with the scalar path.
    if batch % 2 == 1 {
        let last = &mut buffer[(batch - 1) * n..];
        PAIR_SCRATCH_256.with(|c| {
            let mut scratch_buf = c.borrow_mut();
            if scratch_buf.len() < n {
                scratch_buf.resize(n, Complex::new(0.0, 0.0));
            }
            run_single_f64_256(stages, dir_sign, last, &mut scratch_buf[..n]);
        });
    }
}

#[inline(always)]
#[cfg(target_arch = "aarch64")]
fn run_batched_f64_1024_pairs(
    stages: &[StageTwiddles<f64>],
    dir_sign: f64,
    buffer: &mut [Complex<f64>],
) {
    let n = 1024usize;
    let batch = buffer.len() / n;
    debug_assert_eq!(buffer.len() % n, 0);
    debug_assert_eq!(stages.len(), 5);
    debug_assert_eq!(stages.iter().map(|s| s.radix).product::<usize>(), n);

    PAIR_TMP0_1024.with(|c0| {
        PAIR_TMP1_1024.with(|c1| {
            let mut tmp0_buf = c0.borrow_mut();
            let mut tmp1_buf = c1.borrow_mut();
            if tmp0_buf.len() < 2 * n {
                tmp0_buf.resize(2 * n, Complex::new(0.0, 0.0));
            }
            if tmp1_buf.len() < 2 * n {
                tmp1_buf.resize(2 * n, Complex::new(0.0, 0.0));
            }
            let tmp0 = &mut tmp0_buf[..2 * n];
            let tmp1 = &mut tmp1_buf[..2 * n];

            for pair_idx in 0..(batch / 2) {
                let i0 = pair_idx * 2;
                let i1 = pair_idx * 2 + 1;

                unsafe {
                    // Fused first two stages (radix4x4) with transpose-in to planar
                    {
                        let (left, right) = buffer.split_at_mut(i1 * n);
                        let src0 = &left[i0 * n..i0 * n + n];
                        let src1 = &right[..n];
                        crate::wide_neon::fused_first_stage_radix4_m1_transpose_in_pair_planar(
                            src0, src1, tmp0, dir_sign,
                        );
                    }

                    // Stage 1 (m=4, radix-4) on planar data
                    crate::wide_neon::stage_radix4_f64_batched2_planar(
                        256, &*tmp0, tmp1, &stages[1], dir_sign,
                    );

                    // Stage 2 (m=16, radix-4) back into the first planar
                    // scratch buffer before the fused final stages.
                    crate::wide_neon::stage_radix4_f64_batched2_planar(
                        256, &*tmp1, tmp0, &stages[2], dir_sign,
                    );

                    // Stages 3+4 fused with transpose-out
                    {
                        let (left, right) = buffer.split_at_mut(i1 * n);
                        let dst0 = &mut left[i0 * n..i0 * n + n];
                        let dst1 = &mut right[..n];
                        crate::wide_neon::fused_final_radix4x4_1024_transpose_out_pair_planar(
                            &*tmp0, &stages[3], &stages[4], dir_sign, dst0, dst1,
                        );
                    }
                }
            }
        });
    });

    if batch % 2 == 1 {
        let last = &mut buffer[(batch - 1) * n..];
        PAIR_SCRATCH_1024.with(|c| {
            let mut scratch_buf = c.borrow_mut();
            if scratch_buf.len() < n {
                scratch_buf.resize(n, Complex::new(0.0, 0.0));
            }
            run_single_f64_1024(stages, dir_sign, last, &mut scratch_buf[..n]);
        });
    }
}

#[inline(always)]
#[cfg(target_arch = "aarch64")]
fn run_batched_f64_4096_pairs(
    stages: &[StageTwiddles<f64>],
    dir_sign: f64,
    buffer: &mut [Complex<f64>],
) {
    let n = 4096usize;
    let batch = buffer.len() / n;
    debug_assert_eq!(buffer.len() % n, 0);
    debug_assert_eq!(stages.len(), 6);
    debug_assert_eq!(stages.iter().map(|s| s.radix).product::<usize>(), n);

    PAIR_TMP0_4096.with(|c0| {
        PAIR_TMP1_4096.with(|c1| {
            let mut tmp0_buf = c0.borrow_mut();
            let mut tmp1_buf = c1.borrow_mut();
            if tmp0_buf.len() < 2 * n {
                tmp0_buf.resize(2 * n, Complex::new(0.0, 0.0));
            }
            if tmp1_buf.len() < 2 * n {
                tmp1_buf.resize(2 * n, Complex::new(0.0, 0.0));
            }
            let tmp0 = &mut tmp0_buf[..2 * n];
            let tmp1 = &mut tmp1_buf[..2 * n];

            for pair_idx in 0..(batch / 2) {
                let i0 = pair_idx * 2;
                let i1 = pair_idx * 2 + 1;

                unsafe {
                    // Stage 0 (m=1, radix-4) fused with transpose-in to planar
                    {
                        let (left, right) = buffer.split_at_mut(i1 * n);
                        let src0 = &left[i0 * n..i0 * n + n];
                        let src1 = &right[..n];
                        crate::wide_neon::fused_first_stage_radix4_m1_transpose_in_pair_planar(
                            src0, src1, tmp0, dir_sign,
                        );
                    }

                    // Stage 1 (m=4, radix-4) on planar data
                    crate::wide_neon::stage_radix4_f64_batched2_planar(
                        1024, &*tmp0, tmp1, &stages[1], dir_sign,
                    );

                    // Stage 2 (m=16, radix-4) on planar data
                    crate::wide_neon::stage_radix4_f64_batched2_planar(
                        1024, &*tmp1, tmp0, &stages[2], dir_sign,
                    );

                    // Stage 3 (m=64, radix-4) on planar data
                    crate::wide_neon::stage_radix4_f64_batched2_planar(
                        1024, &*tmp0, tmp1, &stages[3], dir_sign,
                    );

                    // Stages 4+5 fused with transpose-out
                    {
                        let (left, right) = buffer.split_at_mut(i1 * n);
                        let dst0 = &mut left[i0 * n..i0 * n + n];
                        let dst1 = &mut right[..n];
                        crate::wide_neon::fused_final_radix4x4_4096_transpose_out_pair_planar(
                            &*tmp1, &stages[4], &stages[5], dir_sign, dst0, dst1, tmp0,
                        );
                    }
                }
            }
        });
    });

    if batch % 2 == 1 {
        let last = &mut buffer[(batch - 1) * n..];
        PAIR_SCRATCH_4096.with(|c| {
            let mut scratch_buf = c.borrow_mut();
            if scratch_buf.len() < n {
                scratch_buf.resize(n, Complex::new(0.0, 0.0));
            }
            run_single_f64_4096(stages, dir_sign, last, &mut scratch_buf[..n]);
        });
    }
}

#[inline(always)]
fn run_single_f64_512(
    stages: &[StageTwiddles<f64>],
    dir_sign: f64,
    buffer: &mut [Complex<f64>],
    scratch: &mut [Complex<f64>],
) {
    debug_assert_eq!(buffer.len(), 512);
    debug_assert_eq!(scratch.len(), 512);
    debug_assert_eq!(stages.len(), 5);
    debug_assert_eq!(stages.iter().map(|s| s.radix).product::<usize>(), 512);

    stage_radix4x4_f64_first2(128, buffer, scratch, &stages[0], &stages[1], dir_sign);
    stage_radix4_f64_direct(128, scratch, buffer, &stages[2], dir_sign);
    stage_radix4_then_radix2_f64_final_512(buffer, &stages[3], &stages[4], dir_sign);
}

/// In-place fused radix-4 + radix-2 final two stages for 512-point FFT.
/// Mirrors `stage_radix4_then_radix2_f64_final` (2048) but with 512-specific
/// strides so the scratch->buffer pass of stage 4 is eliminated.
#[inline(always)]
fn stage_radix4_then_radix2_f64_512(
    buffer: &mut [Complex<f64>],
    radix4_stage: &StageTwiddles<f64>,
    radix2_stage: &StageTwiddles<f64>,
    dir_sign: f64,
) {
    debug_assert_eq!(buffer.len(), 512);
    debug_assert_eq!(radix4_stage.radix, 4);
    debug_assert_eq!(radix4_stage.m, 64);
    debug_assert_eq!(radix4_stage.s, 2);
    debug_assert_eq!(radix2_stage.radix, 2);
    debug_assert_eq!(radix2_stage.m, 256);
    debug_assert_eq!(radix2_stage.s, 1);

    if fused_final_pair_f64_x86(buffer, radix4_stage, radix2_stage, dir_sign) {
        return;
    }

    let tw = radix2_stage.k_slice(1);
    for l in 0..64 {
        let (a0, a1, a2, a3) = stage_radix4_point_f64(buffer, 0, l, 128, radix4_stage, dir_sign);
        let (b0, b1, b2, b3) = stage_radix4_point_f64(buffer, 64, l, 128, radix4_stage, dir_sign);

        let (y0, y1) = radix2(a0, b0 * tw[l]);
        buffer[l] = y0;
        buffer[256 + l] = y1;
        let (y0, y1) = radix2(a1, b1 * tw[64 + l]);
        buffer[64 + l] = y0;
        buffer[320 + l] = y1;
        let (y0, y1) = radix2(a2, b2 * tw[128 + l]);
        buffer[128 + l] = y0;
        buffer[384 + l] = y1;
        let (y0, y1) = radix2(a3, b3 * tw[192 + l]);
        buffer[192 + l] = y0;
        buffer[448 + l] = y1;
    }
}

fn run_single_f64_1024(
    stages: &[StageTwiddles<f64>],
    dir_sign: f64,
    buffer: &mut [Complex<f64>],
    scratch: &mut [Complex<f64>],
) {
    debug_assert_eq!(buffer.len(), 1024);
    debug_assert_eq!(scratch.len(), 1024);
    debug_assert_eq!(stages.len(), 5);
    debug_assert_eq!(stages.iter().map(|s| s.radix).product::<usize>(), 1024);

    stage_radix4x4_f64_first2(256, buffer, scratch, &stages[0], &stages[1], dir_sign);
    stage_radix4_f64_direct(256, scratch, buffer, &stages[2], dir_sign);
    stage_radix4_then_radix4_f64_final_1024(buffer, &stages[3], &stages[4], dir_sign);
}

/// In-place fused radix-4 + radix-4 final two stages for 1024-point FFT.
/// Mirrors `stage_radix4_then_radix4_f64_final` (4096) but with 1024-specific
/// strides so the scratch->buffer pass of stage 3 is eliminated.
fn stage_radix4_then_radix4_f64_1024(
    buffer: &mut [Complex<f64>],
    radix4_stage: &StageTwiddles<f64>,
    final_stage: &StageTwiddles<f64>,
    dir_sign: f64,
) {
    debug_assert_eq!(buffer.len(), 1024);
    debug_assert_eq!(radix4_stage.radix, 4);
    debug_assert_eq!(radix4_stage.m, 64);
    debug_assert_eq!(radix4_stage.s, 4);
    debug_assert_eq!(final_stage.radix, 4);
    debug_assert_eq!(final_stage.m, 256);
    debug_assert_eq!(final_stage.s, 1);

    if fused_final_pair_f64_x86(buffer, radix4_stage, final_stage, dir_sign) {
        return;
    }

    let tw1 = final_stage.k_slice(1);
    let tw2 = final_stage.k_slice(2);
    let tw3 = final_stage.k_slice(3);
    for l in 0..64 {
        let (a0, a1, a2, a3) = stage_radix4_point_f64(buffer, 0, l, 256, radix4_stage, dir_sign);
        let (b0, b1, b2, b3) = stage_radix4_point_f64(buffer, 64, l, 256, radix4_stage, dir_sign);
        let (c0, c1, c2, c3) = stage_radix4_point_f64(buffer, 128, l, 256, radix4_stage, dir_sign);
        let (d0, d1, d2, d3) = stage_radix4_point_f64(buffer, 192, l, 256, radix4_stage, dir_sign);

        let (y0, y1, y2, y3) = radix4(a0, b0 * tw1[l], c0 * tw2[l], d0 * tw3[l], dir_sign);
        buffer[l] = y0;
        buffer[256 + l] = y1;
        buffer[512 + l] = y2;
        buffer[768 + l] = y3;
        let (y0, y1, y2, y3) = radix4(
            a1,
            b1 * tw1[64 + l],
            c1 * tw2[64 + l],
            d1 * tw3[64 + l],
            dir_sign,
        );
        buffer[64 + l] = y0;
        buffer[320 + l] = y1;
        buffer[576 + l] = y2;
        buffer[832 + l] = y3;
        let (y0, y1, y2, y3) = radix4(
            a2,
            b2 * tw1[128 + l],
            c2 * tw2[128 + l],
            d2 * tw3[128 + l],
            dir_sign,
        );
        buffer[128 + l] = y0;
        buffer[384 + l] = y1;
        buffer[640 + l] = y2;
        buffer[896 + l] = y3;
        let (y0, y1, y2, y3) = radix4(
            a3,
            b3 * tw1[192 + l],
            c3 * tw2[192 + l],
            d3 * tw3[192 + l],
            dir_sign,
        );
        buffer[192 + l] = y0;
        buffer[448 + l] = y1;
        buffer[704 + l] = y2;
        buffer[960 + l] = y3;
    }
}

#[inline(always)]
fn run_single_f64_4096(
    stages: &[StageTwiddles<f64>],
    dir_sign: f64,
    buffer: &mut [Complex<f64>],
    scratch: &mut [Complex<f64>],
) {
    debug_assert_eq!(buffer.len(), 4096);
    debug_assert_eq!(scratch.len(), 4096);
    debug_assert_eq!(stages.len(), 6);
    debug_assert_eq!(stages.iter().map(|s| s.radix).product::<usize>(), 4096);

    stage_radix4_f64_direct(1024, buffer, scratch, &stages[0], dir_sign);
    stage_radix4_f64_direct(1024, scratch, buffer, &stages[1], dir_sign);
    stage_radix4_f64_direct(1024, buffer, scratch, &stages[2], dir_sign);
    stage_radix4_f64_direct(1024, scratch, buffer, &stages[3], dir_sign);
    stage_radix4_then_radix4_f64_final(buffer, &stages[4], &stages[5], dir_sign);
}

#[inline(always)]
fn run_single_f64_64(
    stages: &[StageTwiddles<f64>],
    dir_sign: f64,
    buffer: &mut [Complex<f64>],
    scratch: &mut [Complex<f64>],
) {
    debug_assert_eq!(buffer.len(), 64);
    debug_assert_eq!(scratch.len(), 64);
    debug_assert_eq!(stages.len(), 3);
    debug_assert_eq!(stages.iter().map(|s| s.radix).product::<usize>(), 64);

    stage_radix4x4_f64_first2(16, buffer, scratch, &stages[0], &stages[1], dir_sign);
    stage_radix4_f64_direct(16, scratch, buffer, &stages[2], dir_sign);
}

#[inline(always)]
fn run_single_f64_128(
    stages: &[StageTwiddles<f64>],
    dir_sign: f64,
    buffer: &mut [Complex<f64>],
    scratch: &mut [Complex<f64>],
) {
    debug_assert_eq!(buffer.len(), 128);
    debug_assert_eq!(scratch.len(), 128);
    debug_assert_eq!(stages.len(), 4);
    debug_assert_eq!(stages.iter().map(|s| s.radix).product::<usize>(), 128);

    stage_radix4_f64_direct(32, buffer, scratch, &stages[0], dir_sign);
    stage_radix4_f64_direct(32, scratch, buffer, &stages[1], dir_sign);
    stage_radix4_then_radix2_f64_final_128(buffer, &stages[2], &stages[3], dir_sign);
}

#[inline(always)]
fn run_single_f64_256(
    stages: &[StageTwiddles<f64>],
    dir_sign: f64,
    buffer: &mut [Complex<f64>],
    scratch: &mut [Complex<f64>],
) {
    debug_assert_eq!(buffer.len(), 256);
    debug_assert_eq!(scratch.len(), 256);
    debug_assert_eq!(stages.len(), 4);
    debug_assert_eq!(stages.iter().map(|s| s.radix).product::<usize>(), 256);

    stage_radix4_f64_direct(64, buffer, scratch, &stages[0], dir_sign);
    stage_radix4_f64_direct(64, scratch, buffer, &stages[1], dir_sign);
    stage_radix4_then_radix4_f64_final_256(buffer, &stages[2], &stages[3], dir_sign);
}

#[inline(always)]
fn run_single_f64_16384(
    stages: &[StageTwiddles<f64>],
    dir_sign: f64,
    buffer: &mut [Complex<f64>],
    scratch: &mut [Complex<f64>],
) {
    debug_assert_eq!(buffer.len(), 16384);
    debug_assert_eq!(scratch.len(), 16384);
    debug_assert_eq!(stages.len(), 7);
    debug_assert_eq!(stages.iter().map(|s| s.radix).product::<usize>(), 16384);

    stage_radix4x4_f64_first2(4096, buffer, scratch, &stages[0], &stages[1], dir_sign);
    stage_radix4_f64_direct(4096, scratch, buffer, &stages[2], dir_sign);
    stage_radix4_f64_direct(4096, buffer, scratch, &stages[3], dir_sign);
    stage_radix4_f64_direct(4096, scratch, buffer, &stages[4], dir_sign);
    stage_radix4_f64_direct(4096, buffer, scratch, &stages[5], dir_sign);
    stage_radix4_f64_s1_direct(scratch, buffer, &stages[6], dir_sign);
}

#[inline(always)]
fn run_single_f64_audio(
    stages: &[StageTwiddles<f64>],
    dir_sign: f64,
    buffer: &mut [Complex<f64>],
    scratch: &mut [Complex<f64>],
) {
    let n = buffer.len();
    debug_assert_eq!(scratch.len(), n);
    debug_assert_eq!(stages.iter().map(|s| s.radix).product::<usize>(), n);

    let (mut src, mut dst): (&mut [Complex<f64>], &mut [Complex<f64>]) =
        if stages.len().is_multiple_of(2) {
            (buffer, scratch)
        } else {
            scratch.copy_from_slice(buffer);
            (scratch, buffer)
        };

    for stage_tw in stages {
        stage_f64_audio(n, src, dst, stage_tw, dir_sign);
        std::mem::swap(&mut src, &mut dst);
    }
    debug_assert!(std::ptr::eq(src.as_ptr(), buffer.as_ptr()));
}

#[inline(always)]
fn stage_f64_audio(
    n: usize,
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    stage_tw: &StageTwiddles<f64>,
    dir_sign: f64,
) {
    let stride = n / stage_tw.radix;
    match stage_tw.radix {
        2 => stage_radix2_f64_direct(stride, src, dst, stage_tw),
        4 if stage_tw.m != 1 && stage_tw.s == 1 => {
            stage_radix4_f64_s1_direct(src, dst, stage_tw, dir_sign)
        }
        4 => stage_radix4_f64_direct(stride, src, dst, stage_tw, dir_sign),
        _ => unreachable!("audio f64 fast path supports only radix-2/4"),
    }
}

/// x86 AVX2/FMA entry for the fused final pair; returns `true` when the
/// pair was executed by the vector kernel.
#[inline(always)]
fn fused_final_pair_f64_x86(
    buffer: &mut [Complex<f64>],
    stage_a: &StageTwiddles<f64>,
    stage_b: &StageTwiddles<f64>,
    dir_sign: f64,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        // SAFETY: runtime detection proves the target features required
        // by the AVX2/FMA kernel.
        unsafe {
            crate::wide_avx2::fused_final_pair_f64(buffer, stage_a, stage_b, dir_sign);
        }
        return true;
    }
    let _ = (buffer, stage_a, stage_b, dir_sign);
    false
}

/// x86 AVX2/FMA entry for the fused first pair; returns `true` when the
/// pair was executed by the vector kernel.
#[inline(always)]
fn fused_first2_f64_x86(
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    stage0: &StageTwiddles<f64>,
    stage1: &StageTwiddles<f64>,
    dir_sign: f64,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        // SAFETY: runtime detection proves the target features required
        // by the AVX2/FMA kernel.
        unsafe {
            crate::wide_avx2::fused_first2_f64(src, dst, stage0, stage1, dir_sign);
        }
        return true;
    }
    let _ = (src, dst, stage0, stage1, dir_sign);
    false
}

#[inline(always)]
fn stage_radix4x4_f64_first2(
    stride: usize,
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    stage0: &StageTwiddles<f64>,
    stage1: &StageTwiddles<f64>,
    dir_sign: f64,
) {
    debug_assert_eq!(stage0.radix, 4);
    debug_assert_eq!(stage0.m, 1);
    debug_assert_eq!(stage1.radix, 4);
    debug_assert_eq!(stage1.m, 4);
    debug_assert_eq!(stage0.s, stage1.s * 4);
    debug_assert_eq!(stride, stage0.s);

    if fused_first2_f64_x86(src, dst, stage0, stage1, dir_sign) {
        return;
    }

    let groups = stage1.s;
    let tw1 = stage1.k_slice(1);
    let tw2 = stage1.k_slice(2);
    let tw3 = stage1.k_slice(3);

    let w10 = tw1[0];
    let w11 = tw1[1];
    let w12 = tw1[2];
    let w13 = tw1[3];
    let w20 = tw2[0];
    let w21 = tw2[1];
    let w22 = tw2[2];
    let w23 = tw2[3];
    let w30 = tw3[0];
    let w31 = tw3[1];
    let w32 = tw3[2];
    let w33 = tw3[3];

    for g in 0..groups {
        let h0 = g;
        let h1 = g + groups;
        let h2 = g + 2 * groups;
        let h3 = g + 3 * groups;

        let (a00, a01, a02, a03) = radix4(
            src[h0],
            src[h0 + stride],
            src[h0 + 2 * stride],
            src[h0 + 3 * stride],
            dir_sign,
        );
        let (a10, a11, a12, a13) = radix4(
            src[h1],
            src[h1 + stride],
            src[h1 + 2 * stride],
            src[h1 + 3 * stride],
            dir_sign,
        );
        let (a20, a21, a22, a23) = radix4(
            src[h2],
            src[h2 + stride],
            src[h2 + 2 * stride],
            src[h2 + 3 * stride],
            dir_sign,
        );
        let (a30, a31, a32, a33) = radix4(
            src[h3],
            src[h3 + stride],
            src[h3 + 2 * stride],
            src[h3 + 3 * stride],
            dir_sign,
        );

        let base = g * 16;
        stage_radix4x4_store_l(dst, base, 0, a00, a10, a20, a30, w10, w20, w30, dir_sign);
        stage_radix4x4_store_l(dst, base, 1, a01, a11, a21, a31, w11, w21, w31, dir_sign);
        stage_radix4x4_store_l(dst, base, 2, a02, a12, a22, a32, w12, w22, w32, dir_sign);
        stage_radix4x4_store_l(dst, base, 3, a03, a13, a23, a33, w13, w23, w33, dir_sign);
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn stage_radix4x4_store_l(
    dst: &mut [Complex<f64>],
    base: usize,
    l: usize,
    x0: Complex<f64>,
    x1: Complex<f64>,
    x2: Complex<f64>,
    x3: Complex<f64>,
    tw1: Complex<f64>,
    tw2: Complex<f64>,
    tw3: Complex<f64>,
    dir_sign: f64,
) {
    let (y0, y1, y2, y3) = radix4(x0, x1 * tw1, x2 * tw2, x3 * tw3, dir_sign);
    dst[base + l] = y0;
    dst[base + 4 + l] = y1;
    dst[base + 8 + l] = y2;
    dst[base + 12 + l] = y3;
}

fn stage_radix4_then_radix4_f64_final(
    buffer: &mut [Complex<f64>],
    radix4_stage: &StageTwiddles<f64>,
    final_stage: &StageTwiddles<f64>,
    dir_sign: f64,
) {
    debug_assert_eq!(buffer.len(), 4096);
    debug_assert_eq!(radix4_stage.radix, 4);
    debug_assert_eq!(radix4_stage.m, 256);
    debug_assert_eq!(radix4_stage.s, 4);
    debug_assert_eq!(final_stage.radix, 4);
    debug_assert_eq!(final_stage.m, 1024);
    debug_assert_eq!(final_stage.s, 1);

    if fused_final_pair_f64_x86(buffer, radix4_stage, final_stage, dir_sign) {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        crate::wide_neon::fused_final_radix4x4_4096_in_place(
            buffer,
            radix4_stage,
            final_stage,
            dir_sign,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let tw1 = final_stage.k_slice(1);
        let tw2 = final_stage.k_slice(2);
        let tw3 = final_stage.k_slice(3);
        for l in 0..256 {
            let (a0, a1, a2, a3) =
                stage_radix4_point_f64(buffer, 0, l, 1024, radix4_stage, dir_sign);
            let (b0, b1, b2, b3) =
                stage_radix4_point_f64(buffer, 256, l, 1024, radix4_stage, dir_sign);
            let (c0, c1, c2, c3) =
                stage_radix4_point_f64(buffer, 512, l, 1024, radix4_stage, dir_sign);
            let (d0, d1, d2, d3) =
                stage_radix4_point_f64(buffer, 768, l, 1024, radix4_stage, dir_sign);

            stage_radix4_radix4_store(buffer, l, a0, b0, c0, d0, tw1[l], tw2[l], tw3[l], dir_sign);
            stage_radix4_radix4_store(
                buffer,
                256 + l,
                a1,
                b1,
                c1,
                d1,
                tw1[256 + l],
                tw2[256 + l],
                tw3[256 + l],
                dir_sign,
            );
            stage_radix4_radix4_store(
                buffer,
                512 + l,
                a2,
                b2,
                c2,
                d2,
                tw1[512 + l],
                tw2[512 + l],
                tw3[512 + l],
                dir_sign,
            );
            stage_radix4_radix4_store(
                buffer,
                768 + l,
                a3,
                b3,
                c3,
                d3,
                tw1[768 + l],
                tw2[768 + l],
                tw3[768 + l],
                dir_sign,
            );
        }
    }
}

#[inline(always)]
fn stage_radix4_point_f64(
    src: &[Complex<f64>],
    base: usize,
    l: usize,
    stride: usize,
    stage_tw: &StageTwiddles<f64>,
    dir_sign: f64,
) -> (Complex<f64>, Complex<f64>, Complex<f64>, Complex<f64>) {
    let tw1 = stage_tw.k_slice(1);
    let tw2 = stage_tw.k_slice(2);
    let tw3 = stage_tw.k_slice(3);
    radix4(
        src[base + l],
        src[base + l + stride] * tw1[l],
        src[base + l + 2 * stride] * tw2[l],
        src[base + l + 3 * stride] * tw3[l],
        dir_sign,
    )
}

#[inline(always)]
fn stage_radix4_radix2_store(
    dst: &mut [Complex<f64>],
    l: usize,
    a: Complex<f64>,
    b: Complex<f64>,
    tw: Complex<f64>,
) {
    let (y0, y1) = radix2(a, b * tw);
    dst[l] = y0;
    dst[1024 + l] = y1;
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn stage_radix4_radix4_store(
    dst: &mut [Complex<f64>],
    l: usize,
    x0: Complex<f64>,
    x1: Complex<f64>,
    x2: Complex<f64>,
    x3: Complex<f64>,
    tw1: Complex<f64>,
    tw2: Complex<f64>,
    tw3: Complex<f64>,
    dir_sign: f64,
) {
    let (y0, y1, y2, y3) = radix4(x0, x1 * tw1, x2 * tw2, x3 * tw3, dir_sign);
    dst[l] = y0;
    dst[1024 + l] = y1;
    dst[2048 + l] = y2;
    dst[3072 + l] = y3;
}

#[inline(always)]
fn stage_radix4_then_radix2_f64_final_128(
    buffer: &mut [Complex<f64>],
    radix4_stage: &StageTwiddles<f64>,
    radix2_stage: &StageTwiddles<f64>,
    dir_sign: f64,
) {
    debug_assert_eq!(buffer.len(), 128);
    debug_assert_eq!(radix4_stage.radix, 4);
    debug_assert_eq!(radix4_stage.m, 16);
    debug_assert_eq!(radix4_stage.s, 2);
    debug_assert_eq!(radix2_stage.radix, 2);
    debug_assert_eq!(radix2_stage.m, 64);
    debug_assert_eq!(radix2_stage.s, 1);

    if fused_final_pair_f64_x86(buffer, radix4_stage, radix2_stage, dir_sign) {
        return;
    }

    let tw = radix2_stage.k_slice(1);
    for l in 0..16 {
        let (a0, a1, a2, a3) = stage_radix4_point_f64(buffer, 0, l, 32, radix4_stage, dir_sign);
        let (b0, b1, b2, b3) = stage_radix4_point_f64(buffer, 16, l, 32, radix4_stage, dir_sign);

        stage_radix4_radix2_store_128(buffer, l, a0, b0, tw[l]);
        stage_radix4_radix2_store_128(buffer, 16 + l, a1, b1, tw[16 + l]);
        stage_radix4_radix2_store_128(buffer, 32 + l, a2, b2, tw[32 + l]);
        stage_radix4_radix2_store_128(buffer, 48 + l, a3, b3, tw[48 + l]);
    }
}

#[inline(always)]
fn stage_radix4_then_radix4_f64_final_256(
    buffer: &mut [Complex<f64>],
    radix4_stage: &StageTwiddles<f64>,
    final_stage: &StageTwiddles<f64>,
    dir_sign: f64,
) {
    debug_assert_eq!(buffer.len(), 256);
    debug_assert_eq!(radix4_stage.radix, 4);
    debug_assert_eq!(radix4_stage.m, 16);
    debug_assert_eq!(radix4_stage.s, 4);
    debug_assert_eq!(final_stage.radix, 4);
    debug_assert_eq!(final_stage.m, 64);
    debug_assert_eq!(final_stage.s, 1);

    if fused_final_pair_f64_x86(buffer, radix4_stage, final_stage, dir_sign) {
        return;
    }

    let tw1 = final_stage.k_slice(1);
    let tw2 = final_stage.k_slice(2);
    let tw3 = final_stage.k_slice(3);
    for l in 0..16 {
        let (a0, a1, a2, a3) = stage_radix4_point_f64(buffer, 0, l, 64, radix4_stage, dir_sign);
        let (b0, b1, b2, b3) = stage_radix4_point_f64(buffer, 16, l, 64, radix4_stage, dir_sign);
        let (c0, c1, c2, c3) = stage_radix4_point_f64(buffer, 32, l, 64, radix4_stage, dir_sign);
        let (d0, d1, d2, d3) = stage_radix4_point_f64(buffer, 48, l, 64, radix4_stage, dir_sign);

        stage_radix4_radix4_store_256(buffer, l, a0, b0, c0, d0, tw1[l], tw2[l], tw3[l], dir_sign);
        stage_radix4_radix4_store_256(
            buffer,
            16 + l,
            a1,
            b1,
            c1,
            d1,
            tw1[16 + l],
            tw2[16 + l],
            tw3[16 + l],
            dir_sign,
        );
        stage_radix4_radix4_store_256(
            buffer,
            32 + l,
            a2,
            b2,
            c2,
            d2,
            tw1[32 + l],
            tw2[32 + l],
            tw3[32 + l],
            dir_sign,
        );
        stage_radix4_radix4_store_256(
            buffer,
            48 + l,
            a3,
            b3,
            c3,
            d3,
            tw1[48 + l],
            tw2[48 + l],
            tw3[48 + l],
            dir_sign,
        );
    }
}

#[inline(always)]
fn stage_radix4_then_radix4_f64_final_1024(
    buffer: &mut [Complex<f64>],
    radix4_stage: &StageTwiddles<f64>,
    final_stage: &StageTwiddles<f64>,
    dir_sign: f64,
) {
    debug_assert_eq!(buffer.len(), 1024);
    debug_assert_eq!(radix4_stage.radix, 4);
    debug_assert_eq!(radix4_stage.m, 64);
    debug_assert_eq!(radix4_stage.s, 4);
    debug_assert_eq!(final_stage.radix, 4);
    debug_assert_eq!(final_stage.m, 256);
    debug_assert_eq!(final_stage.s, 1);

    if fused_final_pair_f64_x86(buffer, radix4_stage, final_stage, dir_sign) {
        return;
    }

    let tw1 = final_stage.k_slice(1);
    let tw2 = final_stage.k_slice(2);
    let tw3 = final_stage.k_slice(3);
    for l in 0..64 {
        let (a0, a1, a2, a3) = stage_radix4_point_f64(buffer, 0, l, 256, radix4_stage, dir_sign);
        let (b0, b1, b2, b3) = stage_radix4_point_f64(buffer, 64, l, 256, radix4_stage, dir_sign);
        let (c0, c1, c2, c3) = stage_radix4_point_f64(buffer, 128, l, 256, radix4_stage, dir_sign);
        let (d0, d1, d2, d3) = stage_radix4_point_f64(buffer, 192, l, 256, radix4_stage, dir_sign);

        stage_radix4_radix4_store_1024(buffer, l, a0, b0, c0, d0, tw1[l], tw2[l], tw3[l], dir_sign);
        stage_radix4_radix4_store_1024(
            buffer,
            64 + l,
            a1,
            b1,
            c1,
            d1,
            tw1[64 + l],
            tw2[64 + l],
            tw3[64 + l],
            dir_sign,
        );
        stage_radix4_radix4_store_1024(
            buffer,
            128 + l,
            a2,
            b2,
            c2,
            d2,
            tw1[128 + l],
            tw2[128 + l],
            tw3[128 + l],
            dir_sign,
        );
        stage_radix4_radix4_store_1024(
            buffer,
            192 + l,
            a3,
            b3,
            c3,
            d3,
            tw1[192 + l],
            tw2[192 + l],
            tw3[192 + l],
            dir_sign,
        );
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn stage_radix4_radix4_store_1024(
    dst: &mut [Complex<f64>],
    l: usize,
    x0: Complex<f64>,
    x1: Complex<f64>,
    x2: Complex<f64>,
    x3: Complex<f64>,
    tw1: Complex<f64>,
    tw2: Complex<f64>,
    tw3: Complex<f64>,
    dir_sign: f64,
) {
    let (y0, y1, y2, y3) = radix4(x0, x1 * tw1, x2 * tw2, x3 * tw3, dir_sign);
    dst[l] = y0;
    dst[256 + l] = y1;
    dst[512 + l] = y2;
    dst[768 + l] = y3;
}

#[inline(always)]
fn stage_radix4_radix2_store_128(
    dst: &mut [Complex<f64>],
    l: usize,
    a: Complex<f64>,
    b: Complex<f64>,
    tw: Complex<f64>,
) {
    let (y0, y1) = radix2(a, b * tw);
    dst[l] = y0;
    dst[64 + l] = y1;
}

#[inline(always)]
fn stage_radix4_then_radix2_f64_final_512(
    buffer: &mut [Complex<f64>],
    radix4_stage: &StageTwiddles<f64>,
    radix2_stage: &StageTwiddles<f64>,
    dir_sign: f64,
) {
    debug_assert_eq!(buffer.len(), 512);
    debug_assert_eq!(radix4_stage.radix, 4);
    debug_assert_eq!(radix4_stage.m, 64);
    debug_assert_eq!(radix4_stage.s, 2);
    debug_assert_eq!(radix2_stage.radix, 2);
    debug_assert_eq!(radix2_stage.m, 256);
    debug_assert_eq!(radix2_stage.s, 1);

    if fused_final_pair_f64_x86(buffer, radix4_stage, radix2_stage, dir_sign) {
        return;
    }

    let tw = radix2_stage.k_slice(1);
    for l in 0..64 {
        let (a0, a1, a2, a3) = stage_radix4_point_f64(buffer, 0, l, 128, radix4_stage, dir_sign);
        let (b0, b1, b2, b3) = stage_radix4_point_f64(buffer, 64, l, 128, radix4_stage, dir_sign);

        stage_radix4_radix2_store_512(buffer, l, a0, b0, tw[l]);
        stage_radix4_radix2_store_512(buffer, 64 + l, a1, b1, tw[64 + l]);
        stage_radix4_radix2_store_512(buffer, 128 + l, a2, b2, tw[128 + l]);
        stage_radix4_radix2_store_512(buffer, 192 + l, a3, b3, tw[192 + l]);
    }
}

#[inline(always)]
fn stage_radix4_radix2_store_512(
    dst: &mut [Complex<f64>],
    l: usize,
    a: Complex<f64>,
    b: Complex<f64>,
    tw: Complex<f64>,
) {
    let (y0, y1) = radix2(a, b * tw);
    dst[l] = y0;
    dst[256 + l] = y1;
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn stage_radix4_radix4_store_256(
    dst: &mut [Complex<f64>],
    l: usize,
    x0: Complex<f64>,
    x1: Complex<f64>,
    x2: Complex<f64>,
    x3: Complex<f64>,
    tw1: Complex<f64>,
    tw2: Complex<f64>,
    tw3: Complex<f64>,
    dir_sign: f64,
) {
    let (y0, y1, y2, y3) = radix4(x0, x1 * tw1, x2 * tw2, x3 * tw3, dir_sign);
    dst[l] = y0;
    dst[64 + l] = y1;
    dst[128 + l] = y2;
    dst[192 + l] = y3;
}

#[inline(always)]
fn stage_radix2_f64_direct(
    stride: usize,
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    stage_tw: &StageTwiddles<f64>,
) {
    if stage_tw.s == 1 {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: runtime detection proves the target features
            // required by the AVX2/FMA stage.
            unsafe {
                crate::wide_avx2::stage_radix2_f64_aos(stride, src, dst, stage_tw);
            }
            return;
        }
        stage_radix2_f64_s1(src, dst, stage_tw);
        return;
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        crate::wide_neon::stage_radix2_f64(stride, src, dst, stage_tw);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe {
                crate::wide_avx2::stage_radix2_f64_aos(stride, src, dst, stage_tw);
            }
            return;
        }

        stage_radix2::<f64>(stride, src, dst, stage_tw, CpuFftDispatch::Auto);
    }
}

#[inline(always)]
fn stage_radix4_f64_direct(
    stride: usize,
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    stage_tw: &StageTwiddles<f64>,
    dir_sign: f64,
) {
    if stage_tw.m == 1 {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: runtime detection proves the target features
            // required by the AVX2/FMA stage.
            unsafe {
                crate::wide_avx2::stage_radix4_f64_m1(src, dst, stage_tw, dir_sign);
            }
            return;
        }
        stage_radix4_f64_m1(stride, src, dst, stage_tw.s, dir_sign);
        return;
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        crate::wide_neon::stage_radix4_f64(stride, src, dst, stage_tw, dir_sign);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe {
                crate::wide_avx2::stage_radix4_f64_aos(stride, src, dst, stage_tw, dir_sign);
            }
            return;
        }

        stage_radix4::<f64>(stride, src, dst, stage_tw, dir_sign, CpuFftDispatch::Auto);
    }
}

#[inline(always)]
fn stage_radix4_f64_s1_direct(
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    stage_tw: &StageTwiddles<f64>,
    dir_sign: f64,
) {
    debug_assert_eq!(stage_tw.s, 1);

    #[cfg(target_arch = "aarch64")]
    unsafe {
        crate::wide_neon::stage_radix4_f64_s1(src, dst, stage_tw, dir_sign);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe {
                crate::wide_avx2::stage_radix4_f64_s1_aos(src, dst, stage_tw, dir_sign);
            }
            return;
        }

        stage_radix4::<f64>(
            stage_tw.m,
            src,
            dst,
            stage_tw,
            dir_sign,
            CpuFftDispatch::Auto,
        );
    }
}

#[inline(always)]
fn stage_radix2_f64_s1(
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    stage_tw: &StageTwiddles<f64>,
) {
    let m = stage_tw.m;
    let tw1 = stage_tw.k_slice(1);
    for l in 0..m {
        let a = src[l];
        let b = if l == 0 {
            src[l + m]
        } else {
            src[l + m] * tw1[l]
        };
        let (y0, y1) = radix2(a, b);
        dst[l] = y0;
        dst[m + l] = y1;
    }
}

#[inline(always)]
fn stage_radix4_f64_m1(
    stride: usize,
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    groups: usize,
    dir_sign: f64,
) {
    let mut g = 0usize;
    while g + 4 <= groups {
        let (y00, y01, y02, y03) = radix4(
            src[g],
            src[g + stride],
            src[g + 2 * stride],
            src[g + 3 * stride],
            dir_sign,
        );
        let (y10, y11, y12, y13) = radix4(
            src[g + 1],
            src[g + 1 + stride],
            src[g + 1 + 2 * stride],
            src[g + 1 + 3 * stride],
            dir_sign,
        );
        let (y20, y21, y22, y23) = radix4(
            src[g + 2],
            src[g + 2 + stride],
            src[g + 2 + 2 * stride],
            src[g + 2 + 3 * stride],
            dir_sign,
        );
        let (y30, y31, y32, y33) = radix4(
            src[g + 3],
            src[g + 3 + stride],
            src[g + 3 + 2 * stride],
            src[g + 3 + 3 * stride],
            dir_sign,
        );

        let base_out = g * 4;
        dst[base_out] = y00;
        dst[base_out + 1] = y01;
        dst[base_out + 2] = y02;
        dst[base_out + 3] = y03;
        dst[base_out + 4] = y10;
        dst[base_out + 5] = y11;
        dst[base_out + 6] = y12;
        dst[base_out + 7] = y13;
        dst[base_out + 8] = y20;
        dst[base_out + 9] = y21;
        dst[base_out + 10] = y22;
        dst[base_out + 11] = y23;
        dst[base_out + 12] = y30;
        dst[base_out + 13] = y31;
        dst[base_out + 14] = y32;
        dst[base_out + 15] = y33;
        g += 4;
    }
    while g < groups {
        let x0 = src[g];
        let x1 = src[g + stride];
        let x2 = src[g + 2 * stride];
        let x3 = src[g + 3 * stride];
        let (y0, y1, y2, y3) = radix4(x0, x1, x2, x3, dir_sign);
        let base_out = g * 4;
        dst[base_out] = y0;
        dst[base_out + 1] = y1;
        dst[base_out + 2] = y2;
        dst[base_out + 3] = y3;
        g += 1;
    }
}

fn stage<T: FftScalar>(
    n: usize,
    src: &[Complex<T>],
    dst: &mut [Complex<T>],
    stage_tw: &StageTwiddles<T>,
    dir_sign: T,
    dispatch: CpuFftDispatch,
) {
    assert_eq!(
        src.len(),
        n,
        "Stockham source length must equal plan length"
    );
    assert_eq!(
        dst.len(),
        n,
        "Stockham destination length must equal plan length"
    );
    assert!(stage_tw.radix >= 2, "Stockham radix must be at least 2");
    assert_eq!(
        stage_tw.m * stage_tw.radix * stage_tw.s,
        n,
        "Stockham stage dimensions are inconsistent"
    );
    let stride = n / stage_tw.radix;
    match stage_tw.radix {
        2 => stage_radix2(stride, src, dst, stage_tw, dispatch),
        3 => stage_radix3(stride, src, dst, stage_tw, dir_sign),
        4 => stage_radix4(stride, src, dst, stage_tw, dir_sign, dispatch),
        5 => stage_radix5(stride, src, dst, stage_tw, dir_sign),
        8 => stage_radix8(stride, src, dst, stage_tw, dir_sign, dispatch),
        r => panic!("unsupported radix {r}; supported: 2, 3, 4, 5, 8"),
    }
}

#[inline(always)]
fn stage_radix2<T: FftScalar>(
    stride: usize,
    src: &[Complex<T>],
    dst: &mut [Complex<T>],
    stage_tw: &StageTwiddles<T>,
    dispatch: CpuFftDispatch,
) {
    let m = stage_tw.m;
    let s = stage_tw.s;

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if dispatch.allows_simd() && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        // SAFETY: TypeId equality proves T = f32 and the module is compiled
        // only when the wasm `simd128` target feature is enabled.
        unsafe {
            let src_f32: &[Complex<f32>] =
                std::mem::transmute::<&[Complex<T>], &[Complex<f32>]>(src);
            let dst_f32: &mut [Complex<f32>] =
                std::mem::transmute::<&mut [Complex<T>], &mut [Complex<f32>]>(dst);
            let tw_f32: &StageTwiddles<f32> =
                std::mem::transmute::<&StageTwiddles<T>, &StageTwiddles<f32>>(stage_tw);
            assert_transmute_layout::<T, f32>();
            crate::wasm_simd::stage_radix2_f32(stride, src_f32, dst_f32, tw_f32);
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    if dispatch.allows_simd() && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        // SAFETY: TypeId equality proves T = f64; transmutes are layout-
        // identical between Complex<T> and Complex<f64> under the bound.
        unsafe {
            let src_f64: &[Complex<f64>] =
                std::mem::transmute::<&[Complex<T>], &[Complex<f64>]>(src);
            let dst_f64: &mut [Complex<f64>] =
                std::mem::transmute::<&mut [Complex<T>], &mut [Complex<f64>]>(dst);
            let tw_f64: &StageTwiddles<f64> =
                std::mem::transmute::<&StageTwiddles<T>, &StageTwiddles<f64>>(stage_tw);
            assert_transmute_layout::<T, f64>();
            crate::wide_neon::stage_radix2_f64(stride, src_f64, dst_f64, tw_f64);
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if dispatch.allows_simd()
        && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>()
        && std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
    {
        // SAFETY: TypeId equality proves T = f64; runtime detection proves
        // the target features required by the AVX2/FMA stage.
        unsafe {
            let src_f64: &[Complex<f64>] =
                std::mem::transmute::<&[Complex<T>], &[Complex<f64>]>(src);
            let dst_f64: &mut [Complex<f64>] =
                std::mem::transmute::<&mut [Complex<T>], &mut [Complex<f64>]>(dst);
            let tw_f64: &StageTwiddles<f64> =
                std::mem::transmute::<&StageTwiddles<T>, &StageTwiddles<f64>>(stage_tw);
            assert_transmute_layout::<T, f64>();
            crate::wide_avx2::stage_radix2_f64_aos(stride, src_f64, dst_f64, tw_f64);
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if dispatch.allows_simd()
        && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        && std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
    {
        // SAFETY: TypeId equality proves T = f32; runtime detection proves
        // the target features required by the AVX2/FMA stage.
        unsafe {
            let src_f32: &[Complex<f32>] =
                std::mem::transmute::<&[Complex<T>], &[Complex<f32>]>(src);
            let dst_f32: &mut [Complex<f32>] =
                std::mem::transmute::<&mut [Complex<T>], &mut [Complex<f32>]>(dst);
            let tw_f32: &StageTwiddles<f32> =
                std::mem::transmute::<&StageTwiddles<T>, &StageTwiddles<f32>>(stage_tw);
            assert_transmute_layout::<T, f32>();
            crate::wide_avx2::stage_radix2_f32_aos(stride, src_f32, dst_f32, tw_f32);
        }
        return;
    }

    let m_new = m * 2;
    let tw1 = stage_tw.k_slice(1);
    for g in 0..s {
        let base_in = g * m;
        let base_out = g * m_new;
        for l in 0..m {
            let a = src[base_in + l];
            // Skip the identity multiply at l=0 (tw1[0] = 1+0i). The
            // branch is predicted not-taken after the first iteration;
            // the saved complex-mul (~6 f64 ops) is a measurable win at
            // small `m` where the whole stage stays scalar.
            let b = if l == 0 {
                src[base_in + l + stride]
            } else {
                src[base_in + l + stride] * tw1[l]
            };
            let (y0, y1) = radix2(a, b);
            dst[base_out + l] = y0;
            dst[base_out + m + l] = y1;
        }
    }
}

#[inline(always)]
fn stage_radix3<T: FftScalar>(
    stride: usize,
    src: &[Complex<T>],
    dst: &mut [Complex<T>],
    stage_tw: &StageTwiddles<T>,
    dir_sign: T,
) {
    let m = stage_tw.m;
    let s = stage_tw.s;
    let m_new = m * 3;
    let tw1 = stage_tw.k_slice(1);
    let tw2 = stage_tw.k_slice(2);
    for g in 0..s {
        let base_in = g * m;
        let base_out = g * m_new;
        for l in 0..m {
            let x0 = src[base_in + l];
            let (x1, x2) = if l == 0 {
                (src[base_in + l + stride], src[base_in + l + 2 * stride])
            } else {
                (
                    src[base_in + l + stride] * tw1[l],
                    src[base_in + l + 2 * stride] * tw2[l],
                )
            };
            let (y0, y1, y2) = radix3(x0, x1, x2, dir_sign);
            dst[base_out + l] = y0;
            dst[base_out + m + l] = y1;
            dst[base_out + 2 * m + l] = y2;
        }
    }
}

#[inline(always)]
fn stage_radix4<T: FftScalar>(
    stride: usize,
    src: &[Complex<T>],
    dst: &mut [Complex<T>],
    stage_tw: &StageTwiddles<T>,
    dir_sign: T,
    dispatch: CpuFftDispatch,
) {
    let m = stage_tw.m;
    let s = stage_tw.s;

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if dispatch.allows_simd() && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        // SAFETY: TypeId equality proves T = f32 and the module is compiled
        // only when the wasm `simd128` target feature is enabled.
        unsafe {
            let src_f32: &[Complex<f32>] =
                std::mem::transmute::<&[Complex<T>], &[Complex<f32>]>(src);
            let dst_f32: &mut [Complex<f32>] =
                std::mem::transmute::<&mut [Complex<T>], &mut [Complex<f32>]>(dst);
            let tw_f32: &StageTwiddles<f32> =
                std::mem::transmute::<&StageTwiddles<T>, &StageTwiddles<f32>>(stage_tw);
            let dir_f32 = *(&dir_sign as *const T as *const f32);
            assert_transmute_layout::<T, f32>();
            crate::wasm_simd::stage_radix4_f32(stride, src_f32, dst_f32, tw_f32, dir_f32);
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    if dispatch.allows_simd() && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        // SAFETY: TypeId match proves T = f64; layouts are identical.
        unsafe {
            let src_f64: &[Complex<f64>] =
                std::mem::transmute::<&[Complex<T>], &[Complex<f64>]>(src);
            let dst_f64: &mut [Complex<f64>] =
                std::mem::transmute::<&mut [Complex<T>], &mut [Complex<f64>]>(dst);
            let tw_f64: &StageTwiddles<f64> =
                std::mem::transmute::<&StageTwiddles<T>, &StageTwiddles<f64>>(stage_tw);
            let dir_f64: f64 = *(&dir_sign as *const T as *const f64);
            assert_transmute_layout::<T, f64>();
            crate::wide_neon::stage_radix4_f64(stride, src_f64, dst_f64, tw_f64, dir_f64);
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    if dispatch.allows_simd() && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        // SAFETY: TypeId match proves T = f32; layouts are identical.
        unsafe {
            let src_f32: &[Complex<f32>] =
                std::mem::transmute::<&[Complex<T>], &[Complex<f32>]>(src);
            let dst_f32: &mut [Complex<f32>] =
                std::mem::transmute::<&mut [Complex<T>], &mut [Complex<f32>]>(dst);
            let tw_f32: &StageTwiddles<f32> =
                std::mem::transmute::<&StageTwiddles<T>, &StageTwiddles<f32>>(stage_tw);
            let dir_f32: f32 = *(&dir_sign as *const T as *const f32);
            assert_transmute_layout::<T, f32>();
            crate::wide_neon::stage_radix4_f32(stride, src_f32, dst_f32, tw_f32, dir_f32);
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if dispatch.allows_simd()
        && stage_tw.m == 1
        && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>()
        && std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
    {
        // SAFETY: TypeId match proves T = f64; runtime detection proves the
        // target features required by the AVX2/FMA stage.
        unsafe {
            let src_f64: &[Complex<f64>] =
                std::mem::transmute::<&[Complex<T>], &[Complex<f64>]>(src);
            let dst_f64: &mut [Complex<f64>] =
                std::mem::transmute::<&mut [Complex<T>], &mut [Complex<f64>]>(dst);
            let tw_f64: &StageTwiddles<f64> =
                std::mem::transmute::<&StageTwiddles<T>, &StageTwiddles<f64>>(stage_tw);
            let dir_f64: f64 = *(&dir_sign as *const T as *const f64);
            assert_transmute_layout::<T, f64>();
            crate::wide_avx2::stage_radix4_f64_m1(src_f64, dst_f64, tw_f64, dir_f64);
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if dispatch.allows_simd()
        && stage_tw.m == 1
        && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        && std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
    {
        // SAFETY: TypeId match proves T = f32; runtime detection proves the
        // target features required by the AVX2/FMA stage.
        unsafe {
            let src_f32: &[Complex<f32>] =
                std::mem::transmute::<&[Complex<T>], &[Complex<f32>]>(src);
            let dst_f32: &mut [Complex<f32>] =
                std::mem::transmute::<&mut [Complex<T>], &mut [Complex<f32>]>(dst);
            let tw_f32: &StageTwiddles<f32> =
                std::mem::transmute::<&StageTwiddles<T>, &StageTwiddles<f32>>(stage_tw);
            let dir_f32 = *(&dir_sign as *const T as *const f32);
            assert_transmute_layout::<T, f32>();
            crate::wide_avx2::stage_radix4_f32_m1(src_f32, dst_f32, tw_f32, dir_f32);
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if dispatch.allows_simd()
        && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>()
        && std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
    {
        // SAFETY: TypeId match proves T = f64; runtime detection proves the
        // target features required by the AVX2/FMA stage.
        unsafe {
            let src_f64: &[Complex<f64>] =
                std::mem::transmute::<&[Complex<T>], &[Complex<f64>]>(src);
            let dst_f64: &mut [Complex<f64>] =
                std::mem::transmute::<&mut [Complex<T>], &mut [Complex<f64>]>(dst);
            let tw_f64: &StageTwiddles<f64> =
                std::mem::transmute::<&StageTwiddles<T>, &StageTwiddles<f64>>(stage_tw);
            let dir_f64: f64 = *(&dir_sign as *const T as *const f64);
            assert_transmute_layout::<T, f64>();
            crate::wide_avx2::stage_radix4_f64_aos(stride, src_f64, dst_f64, tw_f64, dir_f64);
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if dispatch.allows_simd()
        && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        && std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
    {
        // SAFETY: TypeId match proves T = f32; runtime detection proves the
        // target features required by the AVX2/FMA stage.
        unsafe {
            let src_f32: &[Complex<f32>] =
                std::mem::transmute::<&[Complex<T>], &[Complex<f32>]>(src);
            let dst_f32: &mut [Complex<f32>] =
                std::mem::transmute::<&mut [Complex<T>], &mut [Complex<f32>]>(dst);
            let tw_f32: &StageTwiddles<f32> =
                std::mem::transmute::<&StageTwiddles<T>, &StageTwiddles<f32>>(stage_tw);
            let dir_f32 = *(&dir_sign as *const T as *const f32);
            assert_transmute_layout::<T, f32>();
            crate::wide_avx2::stage_radix4_f32_aos(stride, src_f32, dst_f32, tw_f32, dir_f32);
        }
        return;
    }

    let m_new = m * 4;
    let tw1 = stage_tw.k_slice(1);
    let tw2 = stage_tw.k_slice(2);
    let tw3 = stage_tw.k_slice(3);
    for g in 0..s {
        let base_in = g * m;
        let base_out = g * m_new;
        for l in 0..m {
            let x0 = src[base_in + l];
            let (x1, x2, x3) = if l == 0 {
                (
                    src[base_in + l + stride],
                    src[base_in + l + 2 * stride],
                    src[base_in + l + 3 * stride],
                )
            } else {
                (
                    src[base_in + l + stride] * tw1[l],
                    src[base_in + l + 2 * stride] * tw2[l],
                    src[base_in + l + 3 * stride] * tw3[l],
                )
            };
            let (y0, y1, y2, y3) = radix4(x0, x1, x2, x3, dir_sign);
            dst[base_out + l] = y0;
            dst[base_out + m + l] = y1;
            dst[base_out + 2 * m + l] = y2;
            dst[base_out + 3 * m + l] = y3;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn stage_radix8<T: FftScalar>(
    stride: usize,
    src: &[Complex<T>],
    dst: &mut [Complex<T>],
    stage_tw: &StageTwiddles<T>,
    dir_sign: T,
    dispatch: CpuFftDispatch,
) {
    let m = stage_tw.m;
    let s = stage_tw.s;

    #[cfg(target_arch = "x86_64")]
    if dispatch.allows_simd()
        && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>()
        && std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
    {
        // SAFETY: TypeId match proves T = f64; runtime detection proves the
        // target features required by the AVX2/FMA stage.
        unsafe {
            let src_f64: &[Complex<f64>] =
                std::mem::transmute::<&[Complex<T>], &[Complex<f64>]>(src);
            let dst_f64: &mut [Complex<f64>] =
                std::mem::transmute::<&mut [Complex<T>], &mut [Complex<f64>]>(dst);
            let tw_f64: &StageTwiddles<f64> =
                std::mem::transmute::<&StageTwiddles<T>, &StageTwiddles<f64>>(stage_tw);
            let dir_f64: f64 = *(&dir_sign as *const T as *const f64);
            assert_transmute_layout::<T, f64>();
            if stage_tw.m == 1 {
                crate::wide_avx2::stage_radix8_f64_m1(src_f64, dst_f64, tw_f64, dir_f64);
            } else {
                crate::wide_avx2::stage_radix8_f64_aos(stride, src_f64, dst_f64, tw_f64, dir_f64);
            }
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if dispatch.allows_simd()
        && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        && std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
    {
        // SAFETY: TypeId match proves T = f32; runtime detection proves the
        // target features required by the AVX2/FMA stage.
        unsafe {
            let src_f32: &[Complex<f32>] =
                std::mem::transmute::<&[Complex<T>], &[Complex<f32>]>(src);
            let dst_f32: &mut [Complex<f32>] =
                std::mem::transmute::<&mut [Complex<T>], &mut [Complex<f32>]>(dst);
            let tw_f32: &StageTwiddles<f32> =
                std::mem::transmute::<&StageTwiddles<T>, &StageTwiddles<f32>>(stage_tw);
            let dir_f32 = *(&dir_sign as *const T as *const f32);
            assert_transmute_layout::<T, f32>();
            if stage_tw.m == 1 {
                crate::wide_avx2::stage_radix8_f32_m1(src_f32, dst_f32, tw_f32, dir_f32);
            } else {
                crate::wide_avx2::stage_radix8_f32_aos(stride, src_f32, dst_f32, tw_f32, dir_f32);
            }
        }
        return;
    }

    let m_new = m * 8;
    let tw = [
        stage_tw.k_slice(1),
        stage_tw.k_slice(2),
        stage_tw.k_slice(3),
        stage_tw.k_slice(4),
        stage_tw.k_slice(5),
        stage_tw.k_slice(6),
        stage_tw.k_slice(7),
    ];
    for g in 0..s {
        let base_in = g * m;
        let base_out = g * m_new;
        for l in 0..m {
            let x0 = src[base_in + l];
            let mut x = [x0; 8];
            if l == 0 {
                for k in 1..8 {
                    x[k] = src[base_in + l + k * stride];
                }
            } else {
                for k in 1..8 {
                    x[k] = src[base_in + l + k * stride] * tw[k - 1][l];
                }
            }
            let (y0, y1, y2, y3, y4, y5, y6, y7) =
                radix8(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], dir_sign);
            dst[base_out + l] = y0;
            dst[base_out + m + l] = y1;
            dst[base_out + 2 * m + l] = y2;
            dst[base_out + 3 * m + l] = y3;
            dst[base_out + 4 * m + l] = y4;
            dst[base_out + 5 * m + l] = y5;
            dst[base_out + 6 * m + l] = y6;
            dst[base_out + 7 * m + l] = y7;
        }
    }
}

#[inline(always)]
fn stage_radix5<T: FftScalar>(
    stride: usize,
    src: &[Complex<T>],
    dst: &mut [Complex<T>],
    stage_tw: &StageTwiddles<T>,
    dir_sign: T,
) {
    let m = stage_tw.m;
    let s = stage_tw.s;
    let m_new = m * 5;
    let tw1 = stage_tw.k_slice(1);
    let tw2 = stage_tw.k_slice(2);
    let tw3 = stage_tw.k_slice(3);
    let tw4 = stage_tw.k_slice(4);
    for g in 0..s {
        let base_in = g * m;
        let base_out = g * m_new;
        for l in 0..m {
            let x0 = src[base_in + l];
            let (x1, x2, x3, x4) = if l == 0 {
                (
                    src[base_in + l + stride],
                    src[base_in + l + 2 * stride],
                    src[base_in + l + 3 * stride],
                    src[base_in + l + 4 * stride],
                )
            } else {
                (
                    src[base_in + l + stride] * tw1[l],
                    src[base_in + l + 2 * stride] * tw2[l],
                    src[base_in + l + 3 * stride] * tw3[l],
                    src[base_in + l + 4 * stride] * tw4[l],
                )
            };
            let (y0, y1, y2, y3, y4) = radix5(x0, x1, x2, x3, x4, dir_sign);
            dst[base_out + l] = y0;
            dst[base_out + m + l] = y1;
            dst[base_out + 2 * m + l] = y2;
            dst[base_out + 3 * m + l] = y3;
            dst[base_out + 4 * m + l] = y4;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FftDirection;
    use crate::factor;
    use approx::assert_relative_eq;
    use num_complex::{Complex32, Complex64};
    use std::f64::consts::TAU;

    fn naive_dft(input: &[Complex64], dir_sign: f64) -> Vec<Complex64> {
        let n = input.len();
        let mut out = vec![Complex64::new(0.0, 0.0); n];
        for (k, ok) in out.iter_mut().enumerate() {
            let mut acc = Complex64::new(0.0, 0.0);
            for (j, ij) in input.iter().enumerate() {
                let theta = dir_sign * TAU * (j as f64) * (k as f64) / (n as f64);
                acc += ij * Complex64::new(theta.cos(), theta.sin());
            }
            *ok = acc;
        }
        out
    }

    fn assert_close(got: &[Complex64], want: &[Complex64], tol: f64) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want.iter()) {
            assert_relative_eq!(g.re, w.re, max_relative = tol, epsilon = tol);
            assert_relative_eq!(g.im, w.im, max_relative = tol, epsilon = tol);
        }
    }

    fn assert_close32(got: &[Complex32], want: &[Complex32], tol: f32) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want.iter()) {
            assert_relative_eq!(g.re, w.re, max_relative = tol, epsilon = tol);
            assert_relative_eq!(g.im, w.im, max_relative = tol, epsilon = tol);
        }
    }

    /// Explicit radix-8-first factorisation for tests that target the
    /// radix-8 stages specifically, independent of the planner default.
    fn radix8_radices(n: usize) -> Vec<usize> {
        let mut v = Vec::new();
        let mut rem = n;
        while rem.is_multiple_of(8) {
            v.push(8);
            rem /= 8;
        }
        while rem.is_multiple_of(4) {
            v.push(4);
            rem /= 4;
        }
        if rem == 2 {
            v.push(2);
        }
        assert_eq!(v.iter().product::<usize>(), n, "n={n} is not 8-smooth");
        v
    }

    #[test]
    fn radix8_stages_match_naive_and_scalar() {
        for &n in &[64usize, 512, 1024, 4096] {
            let radices = radix8_radices(n);
            for &dir_sign in &[-1.0f64, 1.0] {
                let direction = if dir_sign < 0.0 {
                    FftDirection::Forward
                } else {
                    FftDirection::Inverse
                };
                let stages =
                    crate::twiddles::build_stage_twiddles::<f64>(n, &radices, direction);
                let input = random(0x8BAD_0000 + n as u64, n);

                let mut expected = input.clone();
                let mut scratch = vec![Complex64::new(0.0, 0.0); n];
                run_single_scalar_f64(&stages, dir_sign, &mut expected, &mut scratch);
                if n == 64 {
                    assert_close(
                        &expected,
                        &naive_dft(&input, dir_sign),
                        (n as f64 * 1e-13).max(1e-12),
                    );
                }

                let mut actual = input.clone();
                let mut scratch = vec![Complex64::new(0.0, 0.0); n];
                run_single(&stages, dir_sign, &mut actual, &mut scratch);
                assert_close(&actual, &expected, (n as f64 * 1e-12).max(1e-12));

                // f32: AVX dispatch must agree with forced scalar dispatch.
                let stages32 =
                    crate::twiddles::build_stage_twiddles::<f32>(n, &radices, direction);
                let input32: Vec<Complex32> = input
                    .iter()
                    .map(|v| Complex32::new(v.re as f32, v.im as f32))
                    .collect();
                let mut scalar32 = input32.clone();
                let mut scratch32 = vec![Complex32::new(0.0, 0.0); n];
                run_single_with_dispatch(
                    &stages32,
                    dir_sign as f32,
                    &mut scalar32,
                    &mut scratch32,
                    CpuFftDispatch::Scalar,
                );
                let mut auto32 = input32;
                let mut scratch32 = vec![Complex32::new(0.0, 0.0); n];
                run_single_with_dispatch(
                    &stages32,
                    dir_sign as f32,
                    &mut auto32,
                    &mut scratch32,
                    CpuFftDispatch::Auto,
                );
                assert_close32(&auto32, &scalar32, (n as f32 * 5.0e-6).max(2.0e-5));
            }
        }
    }

    /// Explicit radix-4-first factorisation (plus a residual radix-2) for
    /// tests that target the radix-2/4 batched-pair kernels specifically,
    /// independent of the planner's arch-dependent default factorisation.
    fn radix4_radices(n: usize) -> Vec<usize> {
        let mut v = Vec::new();
        let mut rem = n;
        while rem.is_multiple_of(4) {
            v.push(4);
            rem /= 4;
        }
        if rem == 2 {
            v.push(2);
        }
        assert_eq!(v.iter().product::<usize>(), n, "n={n} is not 4-smooth");
        v
    }

    fn random(seed: u64, n: usize) -> Vec<Complex64> {
        let mut s = seed
            .wrapping_mul(2862933555777941757)
            .wrapping_add(3037000493);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let re = ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let im = ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
                Complex64::new(re, im)
            })
            .collect()
    }

    fn run_for_len(n: usize, dir: FftDirection) {
        let strategy = factor::plan_length(n);
        let radices = match strategy {
            factor::Strategy::MixedRadix(v) => v,
            factor::Strategy::Bluestein { .. } => return, // not tested here
        };
        let stages = crate::twiddles::build_stage_twiddles::<f64>(n.max(1), &radices, dir);
        let dir_sign = match dir {
            FftDirection::Forward => -1.0,
            FftDirection::Inverse => 1.0,
        };

        let input = random(n as u64 * 7919, n);
        let want = naive_dft(&input, dir_sign);
        let mut buf = input.clone();
        let mut scratch = vec![Complex64::new(0.0, 0.0); n];
        run_single(&stages, dir_sign, &mut buf, &mut scratch);
        let tol = (n as f64 * 1e-13).max(1e-12);
        assert_close(&buf, &want, tol);
    }

    #[test]
    fn forward_smooth_lengths() {
        // 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 24, 25, 27, 30, 32,
        // 36, 40, 45, 48, 60, 64, 75, 80, 100, 120, 125, 128, 144, 240,
        // 256, 360, 480, 720, 1024
        for &n in &[
            1usize, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48,
            60, 64, 75, 80, 100, 120, 125, 128, 144, 240, 256, 360, 480, 720, 1024,
        ] {
            run_for_len(n, FftDirection::Forward);
        }
    }

    #[test]
    fn inverse_smooth_lengths() {
        for &n in &[1usize, 2, 3, 4, 5, 8, 9, 16, 25, 27, 64, 125, 1024] {
            run_for_len(n, FftDirection::Inverse);
        }
    }

    #[test]
    fn radix4_f64_dispatch_matches_scalar_stage() {
        let m = 7;
        let s = 3;
        let n = m * 4 * s;
        let stride = n / 4;
        let input = random(0x07AD_4F64, n);

        for &dir_sign in &[-1.0, 1.0] {
            let mut flat = Vec::with_capacity(3 * m);
            for k in 1..=3 {
                for l in 0..m {
                    let theta = dir_sign * TAU * (k * l * s) as f64 / n as f64;
                    let (sin, cos) = theta.sin_cos();
                    flat.push(Complex64::new(cos, sin));
                }
            }
            let stage_tw = crate::twiddles::StageTwiddles::from_parts(4, m, s, flat);
            let mut scalar = vec![Complex64::new(0.0, 0.0); n];
            let mut dispatched = vec![Complex64::new(0.0, 0.0); n];
            stage_radix4::<f64>(
                stride,
                &input,
                &mut scalar,
                &stage_tw,
                dir_sign,
                CpuFftDispatch::Auto,
            );
            stage_radix4_f64_direct(stride, &input, &mut dispatched, &stage_tw, dir_sign);
            assert_close(&dispatched, &scalar, 2.0e-13);
        }
    }

    #[test]
    fn round_trip_normalizes() {
        let n = 60;
        let radices = match factor::plan_length(n) {
            factor::Strategy::MixedRadix(v) => v,
            _ => panic!(),
        };
        let stages_fwd =
            crate::twiddles::build_stage_twiddles::<f64>(n, &radices, FftDirection::Forward);
        let stages_inv =
            crate::twiddles::build_stage_twiddles::<f64>(n, &radices, FftDirection::Inverse);
        let input = random(42, n);
        let mut buf = input.clone();
        let mut scratch = vec![Complex64::new(0.0, 0.0); n];
        run_single(&stages_fwd, -1.0, &mut buf, &mut scratch);
        run_single(&stages_inv, 1.0, &mut buf, &mut scratch);
        // Inverse is unnormalised: buf = N * input.
        let inv_n = 1.0 / n as f64;
        for (b, x) in buf.iter().zip(input.iter()) {
            assert_relative_eq!(b.re * inv_n, x.re, max_relative = 1e-12, epsilon = 1e-12);
            assert_relative_eq!(b.im * inv_n, x.im, max_relative = 1e-12, epsilon = 1e-12);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn batched_pair_256_matches_scalar() {
        let n = 256;
        let radices = match factor::plan_length(n) {
            factor::Strategy::MixedRadix(v) => v,
            _ => panic!(),
        };
        let stages =
            crate::twiddles::build_stage_twiddles::<f64>(n, &radices, FftDirection::Forward);
        let input = random(42, n * 2);
        let mut scalar = input.clone();
        let mut pair = input.clone();
        let mut scratch = vec![Complex64::new(0.0, 0.0); n];
        run_single_f64_256(&stages, -1.0, &mut scalar[..n], &mut scratch);
        run_single_f64_256(&stages, -1.0, &mut scalar[n..], &mut scratch);
        run_batched_f64_256_pairs(&stages, -1.0, &mut pair);
        let tol = (n as f64 * 1e-13).max(1e-12);
        assert_close(&scalar, &pair, tol);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn generic_batched_pairs_match_scalar_for_unfused_lengths() {
        for &(n, batch) in &[(8usize, 2usize), (32, 3), (128, 2), (512, 3)] {
            let radices = match factor::plan_length(n) {
                factor::Strategy::MixedRadix(v) => v,
                _ => panic!("power-of-two length must use Stockham"),
            };
            let stages =
                crate::twiddles::build_stage_twiddles::<f64>(n, &radices, FftDirection::Forward);
            let input = random(0xBADC0FFE + n as u64 + batch as u64, n * batch);
            let mut expected = input.clone();
            let mut actual = input;
            let mut scratch = vec![Complex64::new(0.0, 0.0); n];
            for chunk in expected.chunks_exact_mut(n) {
                run_single(&stages, -1.0, chunk, &mut scratch);
            }
            run_batched(&stages, -1.0, n, &mut actual, &mut scratch);
            assert_close(&actual, &expected, (n as f64 * 1e-13).max(1e-12));
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_batched_pairs_match_scalar_for_radix2_radix4_lengths() {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            return;
        }
        for &(n, batch) in &[
            (4usize, 2usize),
            (8, 3),
            (16, 2),
            (32, 3),
            (64, 3),
            (128, 2),
            (256, 3),
            (512, 2),
            (1024, 3),
            (2048, 2),
            (4096, 3),
            (8192, 2),
            (16384, 3),
        ] {
            let radices = radix4_radices(n);
            for direction in [FftDirection::Forward, FftDirection::Inverse] {
                let stages = crate::twiddles::build_stage_twiddles::<f64>(n, &radices, direction);
                let dir_sign = match direction {
                    FftDirection::Forward => -1.0,
                    FftDirection::Inverse => 1.0,
                };
                let input = random(
                    0xA5A2_0000_u64 + n as u64 + batch as u64 + dir_sign as u64,
                    n * batch,
                );
                let mut expected = input.clone();
                for chunk in expected.chunks_exact_mut(n) {
                    let mut scratch = vec![Complex64::new(0.0, 0.0); n];
                    run_single(&stages, dir_sign, chunk, &mut scratch);
                }
                let mut actual = input;
                let mut scratch = vec![Complex64::new(0.0, 0.0); n];
                run_batched(&stages, dir_sign, n, &mut actual, &mut scratch);
                assert_close(&actual, &expected, (n as f64 * 1e-13).max(1e-12));
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn run_single_scalar_f64(
        stages: &[StageTwiddles<f64>],
        dir_sign: f64,
        buffer: &mut [Complex64],
        scratch: &mut [Complex64],
    ) {
        assert_eq!(scratch.len(), buffer.len());
        if buffer.len() <= 1 || stages.is_empty() {
            return;
        }
        let (mut src, mut dst) = if stages.len().is_multiple_of(2) {
            (buffer, scratch)
        } else {
            scratch.copy_from_slice(buffer);
            (scratch, buffer)
        };
        for stage_tw in stages {
            let m = stage_tw.m;
            let s = stage_tw.s;
            let radix = stage_tw.radix;
            let stride = src.len() / radix;
            let m_new = m * radix;
            for g in 0..s {
                let base_in = g * m;
                let base_out = g * m_new;
                for l in 0..m {
                    let x0 = src[base_in + l];
                    match radix {
                        2 => {
                            let tw1 = stage_tw.k_slice(1);
                            let x1 = if l == 0 {
                                src[base_in + l + stride]
                            } else {
                                src[base_in + l + stride] * tw1[l]
                            };
                            let (y0, y1) = radix2(x0, x1);
                            dst[base_out + l] = y0;
                            dst[base_out + m + l] = y1;
                        }
                        4 => {
                            let tw1 = stage_tw.k_slice(1);
                            let tw2 = stage_tw.k_slice(2);
                            let tw3 = stage_tw.k_slice(3);
                            let x1 = if l == 0 {
                                src[base_in + l + stride]
                            } else {
                                src[base_in + l + stride] * tw1[l]
                            };
                            let x2 = if l == 0 {
                                src[base_in + l + 2 * stride]
                            } else {
                                src[base_in + l + 2 * stride] * tw2[l]
                            };
                            let x3 = if l == 0 {
                                src[base_in + l + 3 * stride]
                            } else {
                                src[base_in + l + 3 * stride] * tw3[l]
                            };
                            let (y0, y1, y2, y3) = radix4(x0, x1, x2, x3, dir_sign);
                            dst[base_out + l] = y0;
                            dst[base_out + m + l] = y1;
                            dst[base_out + 2 * m + l] = y2;
                            dst[base_out + 3 * m + l] = y3;
                        }
                        8 => {
                            let mut x = [x0; 8];
                            for k in 1..8 {
                                let twk = stage_tw.k_slice(k);
                                x[k] = if l == 0 {
                                    src[base_in + l + k * stride]
                                } else {
                                    src[base_in + l + k * stride] * twk[l]
                                };
                            }
                            let ys =
                                radix8(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], dir_sign);
                            dst[base_out + l] = ys.0;
                            dst[base_out + m + l] = ys.1;
                            dst[base_out + 2 * m + l] = ys.2;
                            dst[base_out + 3 * m + l] = ys.3;
                            dst[base_out + 4 * m + l] = ys.4;
                            dst[base_out + 5 * m + l] = ys.5;
                            dst[base_out + 6 * m + l] = ys.6;
                            dst[base_out + 7 * m + l] = ys.7;
                        }
                        radix => panic!("scalar AVX2 oracle does not support radix {radix}"),
                    }
                }
            }
            std::mem::swap(&mut src, &mut dst);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_fused_paths_match_forced_scalar_for_every_audio_size() {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            return;
        }
        for &n in &[64usize, 128, 256, 512, 1024, 2048, 4096, 16384] {
            let radices = match factor::plan_length(n) {
                factor::Strategy::MixedRadix(radices) => radices,
                factor::Strategy::Bluestein { .. } => panic!("audio size must be smooth"),
            };
            let input = random(0xA2_0000 + n as u64, n);
            for &dir_sign in &[-1.0, 1.0] {
                let direction = if dir_sign < 0.0 {
                    FftDirection::Forward
                } else {
                    FftDirection::Inverse
                };
                let stages = crate::twiddles::build_stage_twiddles::<f64>(n, &radices, direction);
                let mut expected = input.clone();
                let mut scalar_scratch = vec![Complex64::new(0.0, 0.0); n];
                run_single_scalar_f64(&stages, dir_sign, &mut expected, &mut scalar_scratch);
                let mut actual = input.clone();
                let mut actual_scratch = vec![Complex64::new(0.0, 0.0); n];
                run_batched(&stages, dir_sign, n, &mut actual, &mut actual_scratch);
                assert_close(&actual, &expected, (n as f64 * 2.0e-12).max(2.0e-12));
            }
        }
    }

    #[test]
    fn section_plan_groups_consecutive_stages_with_cache_bounds() {
        let n = 16_384;
        let radices = match factor::plan_length(n) {
            factor::Strategy::MixedRadix(v) => v,
            _ => panic!("power-of-two length must use Stockham"),
        };
        let stages =
            crate::twiddles::build_stage_twiddles::<f64>(n, &radices, FftDirection::Forward);
        let sections = plan_stage_sections(n, &stages);

        assert!(!sections.is_empty());
        assert_eq!(sections.first().unwrap().first_stage, 0);
        assert_eq!(sections.last().unwrap().end_stage, stages.len());
        for pair in sections.windows(2) {
            assert_eq!(pair[0].end_stage, pair[1].first_stage);
        }
        for section in sections {
            assert!(section.first_stage < section.end_stage);
            assert!(
                section.working_set_bytes <= section_cache_budget_bytes()
                    || section.end_stage - section.first_stage == 1
            );
        }
    }

    #[test]
    fn sectioned_execution_matches_unsectioned_stockham() {
        let n = 4096;
        let radices = match factor::plan_length(n) {
            factor::Strategy::MixedRadix(v) => v,
            _ => panic!("power-of-two length must use Stockham"),
        };
        let stages =
            crate::twiddles::build_stage_twiddles::<f64>(n, &radices, FftDirection::Forward);
        let sections = plan_stage_sections(n, &stages);
        let input = random(0x05EC_710A, n * 3);
        let mut unsectioned = input.clone();
        let mut sectioned = input;
        let mut scratch_a = vec![Complex64::new(0.0, 0.0); n];
        let mut scratch_b = vec![Complex64::new(0.0, 0.0); n];
        run_batched(&stages, -1.0, n, &mut unsectioned, &mut scratch_a);
        run_batched_with_sections(&stages, -1.0, n, &mut sectioned, &mut scratch_b, &sections);
        assert_close(&sectioned, &unsectioned, 1.0e-12);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn batched_f32_pairs_match_scalar_for_radix2_radix4_lengths() {
        #[cfg(target_arch = "x86_64")]
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            return;
        }
        for &(n, batch) in &[(4usize, 2usize), (8, 3), (16, 2), (32, 3), (8192, 2)] {
            let radices = radix4_radices(n);
            for direction in [FftDirection::Forward, FftDirection::Inverse] {
                let stages = crate::twiddles::build_stage_twiddles::<f32>(n, &radices, direction);
                let dir_sign = match direction {
                    FftDirection::Forward => -1.0_f32,
                    FftDirection::Inverse => 1.0_f32,
                };
                let input: Vec<Complex32> = random(
                    0xF32A_0000_u64 + n as u64 + batch as u64 + dir_sign as u64,
                    n * batch,
                )
                .into_iter()
                .map(|value| Complex32::new(value.re as f32, value.im as f32))
                .collect();
                let mut expected = input.clone();
                for chunk in expected.chunks_exact_mut(n) {
                    let mut scratch = vec![Complex32::new(0.0, 0.0); n];
                    run_single(&stages, dir_sign, chunk, &mut scratch);
                }
                let mut actual = input;
                let mut scratch = vec![Complex32::new(0.0, 0.0); n];
                run_batched(&stages, dir_sign, n, &mut actual, &mut scratch);
                assert_close32(&actual, &expected, (n as f32 * 5.0e-6).max(2.0e-5));
            }
        }
    }
}
