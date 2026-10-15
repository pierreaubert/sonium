//! Generic scalar radix-N butterflies for `Complex<T>` where `T: FftScalar`.
//!
//! Each codelet is `#[inline(always)]` so the Stockham driver can fuse the
//! butterfly into its stage loop. `dir_sign` is `-1` for forward (DFT) or
//! `+1` for inverse (IDFT) and selects the sign of the imaginary part of
//! the radix's intrinsic roots of unity. Twiddle multiplies between
//! stages are pre-baked into the input `x_k` arguments by the driver.

// Literals below are deliberately written with 17 significant digits so
// they round to the nearest representable f64; clippy's excessive
// precision lint would have us truncate to 16 digits, which is sometimes
// 1 ulp off the true value and causes drift in compounded radix-N stages.
#![allow(clippy::excessive_precision, clippy::type_complexity)]

use num_complex::Complex;
use num_traits::Float;

use crate::FftScalar;

#[inline(always)]
fn cplx_i_rot<T: Float>(a: Complex<T>) -> Complex<T> {
    // Multiply by +i: (re, im) -> (-im, re)
    Complex::new(-a.im, a.re)
}

/// Radix-2 butterfly. No internal twiddles (the radix-2 roots are ±1).
#[inline(always)]
pub fn radix2<T: FftScalar>(x0: Complex<T>, x1: Complex<T>) -> (Complex<T>, Complex<T>) {
    (x0 + x1, x0 - x1)
}

/// Radix-3 butterfly. Internal roots-of-unity: 1, w, w² where
/// w = exp(dir_sign · 2π i / 3).
#[inline(always)]
pub fn radix3<T: FftScalar>(
    x0: Complex<T>,
    x1: Complex<T>,
    x2: Complex<T>,
    dir_sign: T,
) -> (Complex<T>, Complex<T>, Complex<T>) {
    // sqrt(3)/2 = 0.86602540378443864676…  (17 sig digits round to nearest f64).
    let sqrt3_half = T::from_f64(0.866_025_403_784_438_65).unwrap();
    let half = T::from_f64(0.5).unwrap();
    let sum12 = x1 + x2;
    let diff12 = x1 - x2;
    let y0 = x0 + sum12;
    let mid = x0 - sum12.scale(half);
    let i_diff = cplx_i_rot(diff12);
    let twist = i_diff.scale(dir_sign * sqrt3_half);
    let y1 = mid + twist;
    let y2 = mid - twist;
    (y0, y1, y2)
}

/// Radix-4 butterfly. Internal roots: 1, ±i, -1 (depending on direction).
#[inline(always)]
pub fn radix4<T: FftScalar>(
    x0: Complex<T>,
    x1: Complex<T>,
    x2: Complex<T>,
    x3: Complex<T>,
    dir_sign: T,
) -> (Complex<T>, Complex<T>, Complex<T>, Complex<T>) {
    // First level: pair (x0,x2) and (x1,x3) with radix-2 each.
    let a = x0 + x2;
    let b = x0 - x2;
    let c = x1 + x3;
    let d = x1 - x3;
    // Second level: combine. Twiddle on `d` is ±i (sign = dir_sign).
    let id = cplx_i_rot(d).scale(dir_sign);
    let y0 = a + c;
    let y2 = a - c;
    let y1 = b + id;
    let y3 = b - id;
    (y0, y1, y2, y3)
}

/// Radix-8 butterfly via the even/odd radix-4 decomposition:
/// `Y[k] = E[k] + w8^k·O[k]`, `Y[k+4] = E[k] − w8^k·O[k]` where `E`/`O`
/// are the radix-4 DFTs of the even/odd inputs and
/// `w8 = exp(dir_sign · 2π i / 8)`, so `w8^k ∈ {1, (1+dir·i)/√2, dir·i,
/// (−1+dir·i)/√2}`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn radix8<T: FftScalar>(
    x0: Complex<T>,
    x1: Complex<T>,
    x2: Complex<T>,
    x3: Complex<T>,
    x4: Complex<T>,
    x5: Complex<T>,
    x6: Complex<T>,
    x7: Complex<T>,
    dir_sign: T,
) -> (
    Complex<T>,
    Complex<T>,
    Complex<T>,
    Complex<T>,
    Complex<T>,
    Complex<T>,
    Complex<T>,
    Complex<T>,
) {
    let (e0, e1, e2, e3) = radix4(x0, x2, x4, x6, dir_sign);
    let (o0, o1, o2, o3) = radix4(x1, x3, x5, x7, dir_sign);
    // √2/2 = 0.70710678118654752440… (17 sig digits round to nearest f64).
    let c = T::from_f64(std::f64::consts::FRAC_1_SQRT_2).unwrap();
    let o1 = o1 * Complex::new(c, dir_sign * c);
    let o2 = cplx_i_rot(o2).scale(dir_sign);
    let o3 = o3 * Complex::new(-c, dir_sign * c);
    (
        e0 + o0,
        e1 + o1,
        e2 + o2,
        e3 + o3,
        e0 - o0,
        e1 - o1,
        e2 - o2,
        e3 - o3,
    )
}

/// Radix-5 butterfly. Internal roots: 1, w, w², w³, w⁴ where
/// w = exp(dir_sign · 2π i / 5). Implemented with Rader's reduction
/// constants (two real multiplies for cos, two for sin).
#[inline(always)]
pub fn radix5<T: FftScalar>(
    x0: Complex<T>,
    x1: Complex<T>,
    x2: Complex<T>,
    x3: Complex<T>,
    x4: Complex<T>,
    dir_sign: T,
) -> (Complex<T>, Complex<T>, Complex<T>, Complex<T>, Complex<T>) {
    // cos(2π/5) = ( √5 − 1)/4 = 0.30901699437494742410…
    // cos(4π/5) = −(√5 + 1)/4 = −0.80901699437494742410…
    // sin(2π/5) =  √(10+2√5)/4 = 0.95105651629515357212…
    // sin(4π/5) =  √(10−2√5)/4 = 0.58778525229247312917…
    // Written with 17 significant digits so the literal rounds to the
    // nearest f64 of the true value (16-digit literals are 1 ulp off and
    // compounded over many radix-5 stages drift to ~1e-11).
    let c1 = T::from_f64(0.309_016_994_374_947_42).unwrap();
    let c2 = T::from_f64(-0.809_016_994_374_947_42).unwrap();
    let s1 = T::from_f64(0.951_056_516_295_153_57).unwrap();
    let s2 = T::from_f64(0.587_785_252_292_473_13).unwrap();

    let t1 = x1 + x4;
    let t2 = x2 + x3;
    let t3 = x1 - x4;
    let t4 = x2 - x3;
    let y0 = x0 + t1 + t2;

    // Real symmetric part
    let z1 = x0 + t1.scale(c1) + t2.scale(c2);
    let z2 = x0 + t1.scale(c2) + t2.scale(c1);
    // Imaginary antisymmetric part (rotated by i, sign flipped per direction).
    let w1 = cplx_i_rot(t3.scale(s1) + t4.scale(s2)).scale(dir_sign);
    let w2 = cplx_i_rot(t3.scale(s2) - t4.scale(s1)).scale(dir_sign);

    let y1 = z1 + w1;
    let y4 = z1 - w1;
    let y2 = z2 + w2;
    let y3 = z2 - w2;
    (y0, y1, y2, y3, y4)
}

/// Helper trait extension: scale a complex by a real, branch-free.
trait ComplexScale<T: Float>: Sized {
    fn scale(self, s: T) -> Self;
}

impl<T: Float> ComplexScale<T> for Complex<T> {
    #[inline(always)]
    fn scale(self, s: T) -> Self {
        Complex::new(self.re * s, self.im * s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FftScalar;
    use approx::assert_relative_eq;
    use num_complex::Complex64;
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

    fn assert_close(a: &[Complex64], b: &[Complex64]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_relative_eq!(x.re, y.re, max_relative = 1e-12, epsilon = 1e-12);
            assert_relative_eq!(x.im, y.im, max_relative = 1e-12, epsilon = 1e-12);
        }
    }

    fn r(seed: u64, n: usize) -> Vec<Complex64> {
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

    #[test]
    fn radix2_matches_dft_forward() {
        let inp = r(1, 2);
        let (y0, y1) = radix2(inp[0], inp[1]);
        let got = [y0, y1];
        let want = naive_dft(&inp, -1.0);
        assert_close(&got, &want);
    }

    #[test]
    fn radix3_matches_dft_forward() {
        let inp = r(2, 3);
        let (y0, y1, y2) = radix3::<f64>(inp[0], inp[1], inp[2], -1.0);
        let got = [y0, y1, y2];
        let want = naive_dft(&inp, -1.0);
        assert_close(&got, &want);
    }

    #[test]
    fn radix3_matches_dft_inverse() {
        let inp = r(2, 3);
        let (y0, y1, y2) = radix3::<f64>(inp[0], inp[1], inp[2], 1.0);
        let got = [y0, y1, y2];
        let want = naive_dft(&inp, 1.0);
        assert_close(&got, &want);
    }

    #[test]
    fn radix4_matches_dft_forward() {
        let inp = r(3, 4);
        let (y0, y1, y2, y3) = radix4::<f64>(inp[0], inp[1], inp[2], inp[3], -1.0);
        let got = [y0, y1, y2, y3];
        let want = naive_dft(&inp, -1.0);
        assert_close(&got, &want);
    }

    #[test]
    fn radix4_matches_dft_inverse() {
        let inp = r(3, 4);
        let (y0, y1, y2, y3) = radix4::<f64>(inp[0], inp[1], inp[2], inp[3], 1.0);
        let got = [y0, y1, y2, y3];
        let want = naive_dft(&inp, 1.0);
        assert_close(&got, &want);
    }

    #[test]
    fn radix5_matches_dft_forward() {
        let inp = r(4, 5);
        let (y0, y1, y2, y3, y4) = radix5::<f64>(inp[0], inp[1], inp[2], inp[3], inp[4], -1.0);
        let got = [y0, y1, y2, y3, y4];
        let want = naive_dft(&inp, -1.0);
        assert_close(&got, &want);
    }

    #[test]
    fn radix5_matches_dft_inverse() {
        let inp = r(4, 5);
        let (y0, y1, y2, y3, y4) = radix5::<f64>(inp[0], inp[1], inp[2], inp[3], inp[4], 1.0);
        let got = [y0, y1, y2, y3, y4];
        let want = naive_dft(&inp, 1.0);
        assert_close(&got, &want);
    }

    // Marker test: ensures FftScalar implements for f32/f64.
    #[test]
    fn fftscalar_impls() {
        fn _check<T: FftScalar>() {}
        _check::<f32>();
        _check::<f64>();
    }
}
