//! AArch64 NEON register-blocked GEMM — the non-SME2 fallback twin of
//! the AVX2 kernel in [`super::avx2`].
//!
//! On Apple Silicon without SME2 (M1/M2/M3) the SME2 dispatch in
//! `sme2::gemm` used to land on a scalar triple loop (~2 GF/s on M1);
//! this kernel restores a vectorised path there. Two properties are
//! contractual, not incidental:
//!
//! - Per-element accumulation is a single sequential FMA chain over
//!   `k`, identical in the vector tiles and the scalar tails (tails use
//!   `mul_add`). The rounding of one output element therefore does not
//!   depend on the matrix shape or tile position — TRSM/SYRK grouped
//!   comparisons rely on this, and it is why transposed shapes route
//!   here rather than to a reassociating vendor kernel.
//! - `beta == 0` never reads `C`, matching the BLAS contract (poisoned
//!   `C` must not propagate NaN).
//!
//! Parallelism splits output rows into contiguous bands; each element
//! is still accumulated by exactly one worker in `k` order, so the
//! band count never changes results.

#![cfg(target_arch = "aarch64")]
#![allow(clippy::too_many_arguments)]

use core::arch::aarch64::*;

/// NEON is part of the AArch64 base ISA; only the process-wide scalar
/// escape hatch can disable this path.
#[inline]
pub fn is_available() -> bool {
    !crate::cpu::force_scalar_dispatch()
}

const MR: usize = 4;
const NR64: usize = 8;
const NR32: usize = 16;

/// MACs below which row-banding costs more than it returns.
const PAR_MIN_MACS: usize = 262_144;

#[derive(Clone, Copy)]
struct SendConst<T>(*const T);
unsafe impl<T> Send for SendConst<T> {}
unsafe impl<T> Sync for SendConst<T> {}
#[derive(Clone, Copy)]
struct SendMut<T>(*mut T);
unsafe impl<T> Send for SendMut<T> {}
unsafe impl<T> Sync for SendMut<T> {}

#[inline]
fn a_elem<T: Copy>(a: *const T, lda: usize, i: usize, p: usize, trans_a: bool) -> T {
    // SAFETY: caller checked the stored-matrix bounds for (i, p).
    unsafe { if trans_a { *a.add(p * lda + i) } else { *a.add(i * lda + p) } }
}

fn scale_c_f64(m: usize, n: usize, beta: f64, c: *mut f64, ldc: usize) {
    for i in 0..m {
        // SAFETY: caller checked the C bounds.
        unsafe {
            let row = c.add(i * ldc);
            if beta == 0.0 {
                for j in 0..n {
                    *row.add(j) = 0.0;
                }
            } else if beta != 1.0 {
                for j in 0..n {
                    *row.add(j) *= beta;
                }
            }
        }
    }
}

fn scale_c_f32(m: usize, n: usize, beta: f32, c: *mut f32, ldc: usize) {
    for i in 0..m {
        // SAFETY: caller checked the C bounds.
        unsafe {
            let row = c.add(i * ldc);
            if beta == 0.0 {
                for j in 0..n {
                    *row.add(j) = 0.0;
                }
            } else if beta != 1.0 {
                for j in 0..n {
                    *row.add(j) *= beta;
                }
            }
        }
    }
}

/// Scalar contract-complete fallback for `trans_b` (the SME2 shim never
/// passes it; this keeps the entry honest for direct callers).
fn gemm_f64_scalar_tb(
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
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 || alpha == 0.0 {
        scale_c_f64(m, n, beta, c.as_mut_ptr(), ldc);
        return;
    }
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                let av = a_elem(a.as_ptr(), lda, i, p, trans_a);
                // SAFETY: bounds checked by the public entry points.
                let bv = unsafe { *b.as_ptr().add(j * ldb + p) };
                acc = av.mul_add(bv, acc);
            }
            let slot = &mut c[i * ldc + j];
            *slot = if beta == 0.0 {
                alpha * acc
            } else {
                alpha.mul_add(acc, beta * *slot)
            };
        }
    }
}

/// f32 twin of [`gemm_f64_scalar_tb`].
fn gemm_f32_scalar_tb(
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
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 || alpha == 0.0 {
        scale_c_f32(m, n, beta, c.as_mut_ptr(), ldc);
        return;
    }
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                let av = a_elem(a.as_ptr(), lda, i, p, trans_a);
                // SAFETY: bounds checked by the public entry points.
                let bv = unsafe { *b.as_ptr().add(j * ldb + p) };
                acc = av.mul_add(bv, acc);
            }
            let slot = &mut c[i * ldc + j];
            *slot = if beta == 0.0 {
                alpha * acc
            } else {
                alpha.mul_add(acc, beta * *slot)
            };
        }
    }
}

/// Compute rows `i0..i1` of `C := alpha * op(A) * B + beta * C` (f64).
/// `B` is always NoTrans here.
#[allow(clippy::too_many_arguments)]
fn gemm_f64_rows(
    i0: usize,
    i1: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    b: *const f64,
    ldb: usize,
    beta: f64,
    c: *mut f64,
    ldc: usize,
    trans_a: bool,
) {
    let mut i = i0;
    while i < i1 {
        let rows = (i1 - i).min(MR);
        let mut j = 0usize;
        // SAFETY: all pointer arithmetic stays inside the caller-checked
        // matrix bounds; vector loads cover at most NR64 columns at a
        // time and the tails below handle the remainder scalar-wise.
        unsafe {
            while j + NR64 <= n {
                let mut acc = [[vdupq_n_f64(0.0); NR64 / 2]; MR];
                for p in 0..k {
                    let bp = b.add(p * ldb + j);
                    let bv = [
                        vld1q_f64(bp),
                        vld1q_f64(bp.add(2)),
                        vld1q_f64(bp.add(4)),
                        vld1q_f64(bp.add(6)),
                    ];
                    for (r, acc_r) in acc.iter_mut().enumerate().take(rows) {
                        let av = a_elem(a, lda, i + r, p, trans_a);
                        for (t, acc_v) in acc_r.iter_mut().enumerate() {
                            *acc_v = vfmaq_n_f64(*acc_v, bv[t], av);
                        }
                    }
                }
                for (r, acc_r) in acc.iter().enumerate().take(rows) {
                    let cp = c.add((i + r) * ldc + j);
                    for (t, acc_v) in acc_r.iter().enumerate() {
                        let slot = cp.add(t * 2);
                        let val = if beta == 0.0 {
                            vmulq_n_f64(*acc_v, alpha)
                        } else {
                            vfmaq_n_f64(vmulq_n_f64(vld1q_f64(slot), beta), *acc_v, alpha)
                        };
                        vst1q_f64(slot, val);
                    }
                }
                j += NR64;
            }
            // Column tail: pairs, then a single scalar column. Every
            // element keeps the same sequential-k FMA chain.
            while j < n {
                let width = (n - j).min(2);
                for r in 0..rows {
                    for jj in 0..width {
                        let mut acc = 0.0f64;
                        for p in 0..k {
                            let av = a_elem(a, lda, i + r, p, trans_a);
                            acc = av.mul_add(*b.add(p * ldb + j + jj), acc);
                        }
                        let slot = c.add((i + r) * ldc + j + jj);
                        *slot = if beta == 0.0 {
                            alpha * acc
                        } else {
                            alpha.mul_add(acc, beta * *slot)
                        };
                    }
                }
                j += width;
            }
        }
        i += rows;
    }
}

/// f32 twin of [`gemm_f64_rows`].
#[allow(clippy::too_many_arguments)]
fn gemm_f32_rows(
    i0: usize,
    i1: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    beta: f32,
    c: *mut f32,
    ldc: usize,
    trans_a: bool,
) {
    let mut i = i0;
    while i < i1 {
        let rows = (i1 - i).min(MR);
        let mut j = 0usize;
        // SAFETY: as in `gemm_f64_rows`.
        unsafe {
            while j + NR32 <= n {
                let mut acc = [[vdupq_n_f32(0.0); NR32 / 4]; MR];
                for p in 0..k {
                    let bp = b.add(p * ldb + j);
                    let bv = [
                        vld1q_f32(bp),
                        vld1q_f32(bp.add(4)),
                        vld1q_f32(bp.add(8)),
                        vld1q_f32(bp.add(12)),
                    ];
                    for (r, acc_r) in acc.iter_mut().enumerate().take(rows) {
                        let av = a_elem(a, lda, i + r, p, trans_a);
                        for (t, acc_v) in acc_r.iter_mut().enumerate() {
                            *acc_v = vfmaq_n_f32(*acc_v, bv[t], av);
                        }
                    }
                }
                for (r, acc_r) in acc.iter().enumerate().take(rows) {
                    let cp = c.add((i + r) * ldc + j);
                    for (t, acc_v) in acc_r.iter().enumerate() {
                        let slot = cp.add(t * 4);
                        let val = if beta == 0.0 {
                            vmulq_n_f32(*acc_v, alpha)
                        } else {
                            vfmaq_n_f32(vmulq_n_f32(vld1q_f32(slot), beta), *acc_v, alpha)
                        };
                        vst1q_f32(slot, val);
                    }
                }
                j += NR32;
            }
            while j < n {
                let width = (n - j).min(4);
                for r in 0..rows {
                    for jj in 0..width {
                        let mut acc = 0.0f32;
                        for p in 0..k {
                            let av = a_elem(a, lda, i + r, p, trans_a);
                            acc = av.mul_add(*b.add(p * ldb + j + jj), acc);
                        }
                        let slot = c.add((i + r) * ldc + j + jj);
                        *slot = if beta == 0.0 {
                            alpha * acc
                        } else {
                            alpha.mul_add(acc, beta * *slot)
                        };
                    }
                }
                j += width;
            }
        }
        i += rows;
    }
}

/// `C := alpha * op(A) * B + beta * C` (f32, row-major). Same contract
/// as `sme2::gemm::sgemm`: bounds are pre-checked by the entry points.
///
/// # Safety
/// The slices must satisfy the GEMM stored-matrix bounds for the given
/// shapes and leading dimensions.
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
    if trans_b {
        gemm_f32_scalar_tb(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_a);
        return;
    }
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 || alpha == 0.0 {
        scale_c_f32(m, n, beta, c.as_mut_ptr(), ldc);
        return;
    }
    let macs = m.saturating_mul(n).saturating_mul(k);
    let threads = rayon::current_num_threads().max(1);
    if macs < PAR_MIN_MACS || threads < 2 || m < MR * 2 {
        gemm_f32_rows(0, m, n, k, alpha, a.as_ptr(), lda, b.as_ptr(), ldb, beta, c.as_mut_ptr(), ldc, trans_a);
        return;
    }
    let bands = threads.min(m / MR).max(1);
    let chunk = m.div_ceil(bands);
    let ap = SendConst(a.as_ptr());
    let bp = SendConst(b.as_ptr());
    let cp = SendMut(c.as_mut_ptr());
    rayon::scope(|s| {
        let mut start = 0usize;
        while start < m {
            let end = (start + chunk).min(m);
            s.spawn(move |_| {
                // Force whole-struct capture of the wrappers (disjoint
                // capture would grab the bare raw-pointer fields).
                let ap: SendConst<f32> = ap;
                let bp: SendConst<f32> = bp;
                let cp: SendMut<f32> = cp;
                // Bands partition C rows disjointly; A/B are
                // read-only for the whole scope.
                gemm_f32_rows(start, end, n, k, alpha, ap.0, lda, bp.0, ldb, beta, cp.0, ldc, trans_a);
            });
            start = end;
        }
    });
}

/// f64 twin of [`sgemm`].
///
/// # Safety
/// Same contract as [`sgemm`].
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
    if trans_b {
        gemm_f64_scalar_tb(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_a);
        return;
    }
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 || alpha == 0.0 {
        scale_c_f64(m, n, beta, c.as_mut_ptr(), ldc);
        return;
    }
    let macs = m.saturating_mul(n).saturating_mul(k);
    let threads = rayon::current_num_threads().max(1);
    if macs < PAR_MIN_MACS || threads < 2 || m < MR * 2 {
        gemm_f64_rows(0, m, n, k, alpha, a.as_ptr(), lda, b.as_ptr(), ldb, beta, c.as_mut_ptr(), ldc, trans_a);
        return;
    }
    let bands = threads.min(m / MR).max(1);
    let chunk = m.div_ceil(bands);
    let ap = SendConst(a.as_ptr());
    let bp = SendConst(b.as_ptr());
    let cp = SendMut(c.as_mut_ptr());
    rayon::scope(|s| {
        let mut start = 0usize;
        while start < m {
            let end = (start + chunk).min(m);
            s.spawn(move |_| {
                // Force whole-struct capture; see `sgemm`.
                let ap: SendConst<f64> = ap;
                let bp: SendConst<f64> = bp;
                let cp: SendMut<f64> = cp;
                // Bands partition C rows disjointly; see `sgemm`.
                gemm_f64_rows(start, end, n, k, alpha, ap.0, lda, bp.0, ldb, beta, cp.0, ldc, trans_a);
            });
            start = end;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_f64(
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a: &[f64],
        lda: usize,
        b: &[f64],
        ldb: usize,
        beta: f64,
        c0: &[f64],
        ldc: usize,
        trans_a: bool,
    ) -> Vec<f64> {
        let mut c = c0.to_vec();
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for p in 0..k {
                    acc += a_elem(a.as_ptr(), lda, i, p, trans_a) * b[p * ldb + j];
                }
                c[i * ldc + j] = alpha * acc + beta * c[i * ldc + j];
            }
        }
        c
    }

    #[test]
    fn dgemm_matches_naive_incl_tails_and_transa() {
        for &(m, n, k) in &[(1usize, 1usize, 1usize), (3, 5, 7), (13, 27, 11), (37, 33, 21), (64, 48, 32)] {
            for trans_a in [false, true] {
                let (ar, ac) = if trans_a { (k, m) } else { (m, k) };
                let a: Vec<f64> = (0..ar * ac).map(|t| ((t * 37 % 101) as f64 - 50.0) / 53.0).collect();
                let b: Vec<f64> = (0..k * n).map(|t| ((t * 17 % 89) as f64 - 44.0) / 61.0).collect();
                let c0: Vec<f64> = (0..m * n).map(|t| ((t * 7 % 31) as f64 - 15.0) / 29.0).collect();
                let mut got = c0.clone();
                // SAFETY: shapes and strides are consistent by construction.
                unsafe {
                    dgemm(m, n, k, 0.7, &a, ac, &b, n, -0.3, &mut got, n, trans_a, false);
                }
                let want = naive_f64(m, n, k, 0.7, &a, ac, &b, n, -0.3, &c0, n, trans_a);
                for (t, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                    assert!((g - w).abs() <= 1e-12 * (1.0 + w.abs()), "t={t} m={m} n={n} k={k} trans_a={trans_a}: {g} vs {w}");
                }
            }
        }
    }

    #[test]
    fn dgemm_beta0_does_not_read_c() {
        let (m, n, k) = (13usize, 27usize, 11usize);
        let a: Vec<f64> = (0..m * k).map(|t| (t as f64) / 37.0 - 1.0).collect();
        let b: Vec<f64> = (0..k * n).map(|t| (t as f64) / 41.0 - 1.0).collect();
        let mut got = vec![f64::NAN; m * n];
        // SAFETY: shapes and strides are consistent by construction.
        unsafe {
            dgemm(m, n, k, 1.0, &a, k, &b, n, 0.0, &mut got, n, false, false);
        }
        assert!(got.iter().all(|v| v.is_finite()), "beta=0 read poisoned C");
    }

    #[test]
    fn sgemm_matches_naive() {
        let (m, n, k) = (21usize, 35usize, 17usize);
        let a: Vec<f32> = (0..m * k).map(|t| ((t * 13 % 67) as f32 - 33.0) / 43.0).collect();
        let b: Vec<f32> = (0..k * n).map(|t| ((t * 11 % 59) as f32 - 29.0) / 47.0).collect();
        let c0: Vec<f32> = (0..m * n).map(|t| ((t * 5 % 23) as f32 - 11.0) / 19.0).collect();
        let mut got = c0.clone();
        // SAFETY: shapes and strides are consistent by construction.
        unsafe {
            sgemm(m, n, k, -0.6, &a, k, &b, n, 0.4, &mut got, n, false, false);
        }
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a[i * k + p] * b[p * n + j];
                }
                let want = (-0.6f32).mul_add(acc, 0.4 * c0[i * n + j]);
                let g = got[i * n + j];
                assert!((g - want).abs() <= 1e-4 * (1.0 + want.abs()), "({i},{j}): {g} vs {want}");
            }
        }
    }
}
