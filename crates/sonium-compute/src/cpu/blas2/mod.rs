//! In-house BLAS-2 kernels.
//!
//! Exposes `gemv` for `f32`, `f64`, `Complex32`, `Complex64`. Row-major
//! storage, BLAS-style `(op, m, n, alpha, A, lda, x, beta, y)` signature
//! parameterised by [`crate::backend::Transpose`].
//!
//! The real-valued (`f32`, `f64`) `NoTrans` path routes through the SME2
//! micro-kernels in [`crate::sme2`] when available; real `Trans`/`ConjTrans`
//! fan out into BLAS-1 axpys. Complex kernels come in two flavours:
//! [`complex_native`] is the dispatched production path and works directly on
//! interleaved storage with NEON (aarch64) / AVX2+FMA (x86_64) kernels;
//! [`complex4m`] remains available as an independent 4-real reference path.

#![allow(clippy::too_many_arguments)]

pub mod complex4m;
pub mod complex_native;
pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

use crate::backend::Transpose;
use crate::cpu::blas1;
use crate::sme2;
use num_complex::{Complex32, Complex64};

// ---------- Trans gemv via row-of-A axpys ---------------------------------
// The row-major Trans contract has `A_stored` laid out as `n × m` with
// row stride `lda ≥ m`, so row `j` of stored `A` is contiguous in memory
// of length `m`. Then
//
//   y[i] = Σ_j stored_A[j, i] * x[j]
//        = Σ_j (alpha * x[j]) * stored_A[j, :][i]
//
// is a series of `n` real `axpy` operations of the row vectors into `y`.
// We hand the inner loops to the NEON kernels in [`crate::cpu::blas1`]
// so the Trans path stops depending on LLVM auto-vectorisation.

fn gemv_trans_f32(
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    x: &[f32],
    beta: f32,
    y: &mut [f32],
) {
    debug_assert!(lda >= m);
    // A zero-row result has no output elements and therefore no matrix
    // rows to read.  In particular, do not form empty slices at
    // `j * lda` below: a valid `(m = 0, n > 0)` transpose request may have
    // an empty matrix buffer and a padded leading dimension.
    if m == 0 {
        return;
    }
    // NOTE: the serial fan-out below preserves exact summation order.
    // On x86_64 with AVX2 the privatized fan-out in `avx2::gemv_trans_f32`
    // takes over instead (serial for small shapes, parallel blocks above
    // threshold): block partials reduce in order, so only rounding differs
    // (inside the 1e-12 parity tolerances), and NaN still propagates (no
    // skips). It owns beta/alpha handling end to end, so this delegation
    // sits before the beta scaling below — scaling here as well would
    // apply beta twice.
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::blas1::avx2::is_available() {
        // SAFETY: availability checked; slice bounds per `validate`
        // at the call sites and the assertions above.
        unsafe {
            avx2::gemv_trans_f32(m, n, alpha, a, lda, x, beta, y);
        }
        return;
    }
    if beta == 0.0 {
        for v in y.iter_mut() {
            *v = 0.0;
        }
    } else if beta != 1.0 {
        for v in y.iter_mut() {
            *v *= beta;
        }
    }
    if alpha == 0.0 {
        return;
    }
    for j in 0..n {
        // NOTE: no `xj == 0.0` skip here. Skipping would turn `0 * NaN`
        // matrix entries into silently dropped terms instead of
        // propagating NaN like the reference accumulation below.
        let xj = alpha * x[j];
        blas1::axpy_f32(xj, &a[j * lda..j * lda + m], y);
    }
}

fn gemv_trans_f64(
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) {
    debug_assert!(lda >= m);
    if m == 0 {
        return;
    }
    // Same AVX2 delegation as the f32 path above (before beta scaling:
    // the AVX2 kernel owns beta/alpha end to end).
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::blas1::avx2::is_available() {
        // SAFETY: availability checked; slice bounds per the assertions
        // above and `validate` at the call sites.
        unsafe {
            avx2::gemv_trans_f64(m, n, alpha, a, lda, x, beta, y);
        }
        return;
    }
    if beta == 0.0 {
        for v in y.iter_mut() {
            *v = 0.0;
        }
    } else if beta != 1.0 {
        for v in y.iter_mut() {
            *v *= beta;
        }
    }
    if alpha == 0.0 {
        return;
    }
    for j in 0..n {
        // NOTE: no `xj == 0.0` skip here; see the f32 path above for why
        // (NaN propagation must match the reference accumulation).
        let xj = alpha * x[j];
        blas1::axpy_f64(xj, &a[j * lda..j * lda + m], y);
    }
}

/// `y := alpha * op(A) * x + beta * y` for `f32`.
pub fn gemv_f32(
    op: Transpose,
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    x: &[f32],
    beta: f32,
    y: &mut [f32],
) {
    // Fast path: NoTrans → SME2 sgemv (which now honours arbitrary beta).
    if matches!(op, Transpose::NoTrans) {
        // Without SME2 the SME2 entry is a single-accumulator scalar
        // loop (~9 GB/s on M1); Accelerate runs ~13x faster there.
        #[cfg(target_os = "macos")]
        if !crate::cpu::blas3::sme2_available()
            && crate::cpu::blas3::accelerate::sgemv_n(m, n, alpha, a, lda, x, beta, y)
        {
            return;
        }
        sme2::gemv::sgemv(m, n, alpha, a, lda, x, beta, y);
        return;
    }
    // Trans/ConjTrans (equivalent for real types) → axpy fan-out path.
    gemv_trans_f32(m, n, alpha, a, lda, x, beta, y);
}

/// `y := alpha * op(A) * x + beta * y` for `f64`.
pub fn gemv_f64(
    op: Transpose,
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) {
    if matches!(op, Transpose::NoTrans) {
        // See `gemv_f32`: without SME2, route through Accelerate.
        #[cfg(target_os = "macos")]
        if !crate::cpu::blas3::sme2_available()
            && crate::cpu::blas3::accelerate::dgemv_n(m, n, alpha, a, lda, x, beta, y)
        {
            return;
        }
        sme2::gemv::dgemv(m, n, alpha, a, lda, x, beta, y);
        return;
    }
    gemv_trans_f64(m, n, alpha, a, lda, x, beta, y);
}

/// `y := alpha * op(A) * x + beta * y` for `Complex32`.
pub fn gemv_c32(
    op: Transpose,
    m: usize,
    n: usize,
    alpha: Complex32,
    a: &[Complex32],
    lda: usize,
    x: &[Complex32],
    beta: Complex32,
    y: &mut [Complex32],
) {
    // Keep the public dispatch on the native interleaved kernel. The 4M
    // implementation is retained for independent parity testing, but it
    // deinterleaves into scratch and cannot provide the native x86_64
    // AVX2/FMA contract.
    complex_native::gemv_c32(op, m, n, alpha, a, lda, x, beta, y);
}

/// `y := alpha * op(A) * x + beta * y` for `Complex64`.
pub fn gemv_c64(
    op: Transpose,
    m: usize,
    n: usize,
    alpha: Complex64,
    a: &[Complex64],
    lda: usize,
    x: &[Complex64],
    beta: Complex64,
    y: &mut [Complex64],
) {
    complex_native::gemv_c64(op, m, n, alpha, a, lda, x, beta, y);
}

// ---------- correctness tests ----------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn ref_gemv_f64(
        op: Transpose,
        m: usize,
        n: usize,
        alpha: f64,
        a: &[f64],
        lda: usize,
        x: &[f64],
        beta: f64,
        y_in: &[f64],
    ) -> Vec<f64> {
        let mut y = vec![0.0; m];
        for i in 0..m {
            let mut acc = 0.0;
            for j in 0..n {
                let aij = match op {
                    Transpose::NoTrans => a[i * lda + j],
                    Transpose::Trans | Transpose::ConjTrans => a[j * lda + i],
                };
                acc += aij * x[j];
            }
            y[i] = alpha * acc + beta * y_in[i];
        }
        y
    }

    #[test]
    fn gemv_f64_notrans_matches_ref() {
        let m = 5;
        let n = 4;
        let a: Vec<f64> = (0..m * n).map(|k| (k as f64) * 0.1).collect();
        let x: Vec<f64> = (0..n).map(|k| (k as f64) * 0.3 - 0.5).collect();
        let y0: Vec<f64> = (0..m).map(|k| (k as f64) * 0.2 + 1.0).collect();
        let mut y = y0.clone();
        gemv_f64(Transpose::NoTrans, m, n, 0.7, &a, n, &x, 0.5, &mut y);
        let expected = ref_gemv_f64(Transpose::NoTrans, m, n, 0.7, &a, n, &x, 0.5, &y0);
        for (a, b) in y.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, max_relative = 1e-12);
        }
    }

    #[test]
    fn gemv_f64_trans_matches_ref() {
        // A stored as n×m (n=4 rows, m=5 cols), op = T → result is m×1.
        let m = 5;
        let n = 4;
        let a: Vec<f64> = (0..n * m).map(|k| (k as f64) * 0.1).collect();
        let x: Vec<f64> = (0..n).map(|k| (k as f64) * 0.3 - 0.5).collect();
        let y0: Vec<f64> = (0..m).map(|k| (k as f64) * 0.2 + 1.0).collect();
        let mut y = y0.clone();
        gemv_f64(Transpose::Trans, m, n, 0.7, &a, m, &x, 0.5, &mut y);
        let expected = ref_gemv_f64(Transpose::Trans, m, n, 0.7, &a, m, &x, 0.5, &y0);
        for (a, b) in y.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, max_relative = 1e-12);
        }
    }

    #[test]
    fn gemv_trans_propagates_nan_with_zero_x() {
        // `0 * NaN` in a matrix row must propagate NaN like the reference
        // accumulation; the old `xj == 0.0` skip silently dropped it.
        let m = 2;
        let n = 2;
        let a = vec![1.0_f64, f64::NAN, 3.0, 4.0];
        let x = vec![0.0_f64, 1.0];
        let y0 = vec![0.0_f64, 0.0];
        let mut y = y0.clone();
        gemv_f64(Transpose::Trans, m, n, 1.0, &a, m, &x, 0.0, &mut y);
        let expected = ref_gemv_f64(Transpose::Trans, m, n, 1.0, &a, m, &x, 0.0, &y0);
        assert_eq!(y.len(), expected.len());
        for (got, want) in y.iter().zip(expected.iter()) {
            if want.is_nan() {
                assert!(got.is_nan(), "expected NaN, got {got}");
            } else {
                assert_relative_eq!(got, want, max_relative = 1e-12);
            }
        }
        assert!(y[1].is_nan());
    }

    #[test]
    fn real_transpose_zero_rows_is_a_noop() {
        // For transpose GEMV, `m` is the output length.  The matrix has
        // `n` stored rows and zero stored columns, so its valid span is
        // empty even when `lda` is padded.  This used to panic while slicing
        // the second empty row in the axpy fan-out loop.
        let x32 = [1.0_f32, -2.0, 3.0];
        let mut y32: [f32; 0] = [];
        gemv_f32(
            Transpose::Trans,
            0,
            x32.len(),
            2.0,
            &[],
            4,
            &x32,
            3.0,
            &mut y32,
        );

        let x64 = [1.0_f64, -2.0, 3.0];
        let mut y64: [f64; 0] = [];
        gemv_f64(
            Transpose::ConjTrans,
            0,
            x64.len(),
            2.0,
            &[],
            4,
            &x64,
            3.0,
            &mut y64,
        );
    }

    #[test]
    fn gemv_c32_conjtrans_hermitian() {
        // A = [[1+i, 2], [3, 4-i]], op = C → A^H = [[1-i, 3], [2, 4+i]].
        let a = vec![
            Complex32::new(1.0, 1.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(4.0, -1.0),
        ];
        let x = vec![Complex32::new(1.0, 0.0), Complex32::new(0.0, 1.0)];
        let mut y = vec![Complex32::new(0.0, 0.0); 2];
        gemv_c32(
            Transpose::ConjTrans,
            2,
            2,
            Complex32::new(1.0, 0.0),
            &a,
            2,
            &x,
            Complex32::new(0.0, 0.0),
            &mut y,
        );
        // y[0] = (1-i)*1 + 3*i = 1 - i + 3i = 1 + 2i
        // y[1] = 2*1 + (4+i)*i = 2 + 4i - 1 = 1 + 4i
        assert_relative_eq!(y[0].re, 1.0, max_relative = 1e-6);
        assert_relative_eq!(y[0].im, 2.0, max_relative = 1e-6);
        assert_relative_eq!(y[1].re, 1.0, max_relative = 1e-6);
        assert_relative_eq!(y[1].im, 4.0, max_relative = 1e-6);
    }

    #[test]
    fn complex4m_matches_complex_native_for_all_transpose_modes() {
        let m = 7;
        let n = 5;
        let lda = 8;
        let a32: Vec<Complex32> = (0..m * lda)
            .map(|index| {
                let row = index / lda;
                let column = index % lda;
                if column < n {
                    Complex32::new(
                        0.13 * (row as f32 + 1.0) - 0.07 * column as f32,
                        -0.11 + 0.05 * (row + column) as f32,
                    )
                } else {
                    Complex32::new(99.0, -77.0)
                }
            })
            .collect();
        let a64: Vec<Complex64> = a32
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect();
        let alpha32 = Complex32::new(0.7, -0.4);
        let beta32 = Complex32::new(-0.2, 0.3);
        let alpha64 = Complex64::new(alpha32.re as f64, alpha32.im as f64);
        let beta64 = Complex64::new(beta32.re as f64, beta32.im as f64);

        for op in [Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
            // The BLAS interface keeps `m`/`n` as the output/input vector
            // lengths for every operation; transpose callers provide the
            // stored matrix with `lda >= m`.
            let input_len = n;
            let output_len = m;
            let x32: Vec<Complex32> = (0..input_len)
                .map(|index| Complex32::new(0.31 + index as f32 * 0.17, -0.2 + index as f32 * 0.09))
                .collect();
            let x64: Vec<Complex64> = x32
                .iter()
                .map(|value| Complex64::new(value.re as f64, value.im as f64))
                .collect();
            let y032: Vec<Complex32> = (0..output_len)
                .map(|index| Complex32::new(-0.4 + index as f32 * 0.08, 0.2 - index as f32 * 0.06))
                .collect();
            let y064: Vec<Complex64> = y032
                .iter()
                .map(|value| Complex64::new(value.re as f64, value.im as f64))
                .collect();

            let mut four_m32 = y032.clone();
            let mut native32 = y032;
            complex4m::gemv_c32(op, m, n, alpha32, &a32, lda, &x32, beta32, &mut four_m32);
            complex_native::gemv_c32(op, m, n, alpha32, &a32, lda, &x32, beta32, &mut native32);
            for (four_m, native) in four_m32.iter().zip(native32.iter()) {
                assert_relative_eq!(four_m.re, native.re, max_relative = 2e-5, epsilon = 2e-5);
                assert_relative_eq!(four_m.im, native.im, max_relative = 2e-5, epsilon = 2e-5);
            }

            let mut four_m64 = y064.clone();
            let mut native64 = y064;
            complex4m::gemv_c64(op, m, n, alpha64, &a64, lda, &x64, beta64, &mut four_m64);
            complex_native::gemv_c64(op, m, n, alpha64, &a64, lda, &x64, beta64, &mut native64);
            for (four_m, native) in four_m64.iter().zip(native64.iter()) {
                assert_relative_eq!(four_m.re, native.re, max_relative = 1e-12, epsilon = 1e-12);
                assert_relative_eq!(four_m.im, native.im, max_relative = 1e-12, epsilon = 1e-12);
            }
        }
    }
}
