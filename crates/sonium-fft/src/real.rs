//! Real-input FFT plans and half-spectrum packing.
//!
//! The forward transform uses the same unnormalised complex DFT convention as
//! [`crate::FftPlan`]. For a real input only bins `0..=N/2` are stored. The
//! inverse reconstructs the Hermitian spectrum and applies `1/N`, so
//! `inverse(forward(x)) == x` within floating-point error.

use num_complex::Complex;
use std::sync::Arc;

use crate::FftDirection;
use crate::FftPlan;
use crate::cpu::planner::CpuFftPlanner;
use crate::factor::PlanLengthError;
use crate::scalar::FftScalar;

/// A cached real-input transform plan.
pub struct RealFftPlan<T: FftScalar> {
    n: usize,
    forward: Arc<dyn FftPlan<T>>,
    inverse: Arc<dyn FftPlan<T>>,
    packed_forward: Option<Arc<dyn FftPlan<T>>>,
    packed_inverse: Option<Arc<dyn FftPlan<T>>>,
    /// Plan-time packed-recombination factors, one per output bin:
    /// forward `(sin θ, -cos θ)` with `θ = -2πk/n`, inverse
    /// `(-sin φ, cos φ)` with `φ = 2πk/n`. Hoisting these out of the
    /// per-call loops removes one `sincos` per bin per transform.
    /// Empty for odd `n` (the packed path is unused there).
    forward_factors: Vec<Complex<T>>,
    inverse_factors: Vec<Complex<T>>,
}

/// Build the packed-recombination factor tables for length `n`.
fn build_recombination_factors<T: FftScalar>(n: usize) -> (Vec<Complex<T>>, Vec<Complex<T>>) {
    if !n.is_multiple_of(2) {
        return (Vec::new(), Vec::new());
    }
    let m = n / 2;
    let mut forward = Vec::with_capacity(m + 1);
    for k in 0..=m {
        let theta = -std::f64::consts::TAU * k as f64 / n as f64;
        let (sin, cos) = sonium_fastmath::sincos_scalar(theta);
        forward.push(Complex::new(
            T::from_f64(sin).unwrap(),
            T::from_f64(-cos).unwrap(),
        ));
    }
    let mut inverse = Vec::with_capacity(m);
    for k in 0..m {
        let theta = std::f64::consts::TAU * k as f64 / n as f64;
        let (sin, cos) = sonium_fastmath::sincos_scalar(theta);
        inverse.push(Complex::new(
            T::from_f64(-sin).unwrap(),
            T::from_f64(cos).unwrap(),
        ));
    }
    (forward, inverse)
}

impl<T: FftScalar> RealFftPlan<T> {
    /// Build a real plan using a private CPU planner.
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "real FFT length must be positive");
        let planner = CpuFftPlanner::<T>::new();
        Self::with_planner(n, &planner)
    }

    /// Checked real-plan construction for externally supplied lengths.
    pub fn try_new(n: usize) -> Result<Self, PlanLengthError> {
        let planner = CpuFftPlanner::<T>::new();
        Self::try_with_planner(n, &planner)
    }

    /// Build a real plan from an existing CPU planner so complex and real
    /// plans share its cache.
    pub fn with_planner(n: usize, planner: &CpuFftPlanner<T>) -> Self {
        assert!(n > 0, "real FFT length must be positive");
        let (forward_factors, inverse_factors) = build_recombination_factors(n);
        Self {
            n,
            forward: planner.plan(n, FftDirection::Forward),
            inverse: planner.plan(n, FftDirection::Inverse),
            packed_forward: n
                .is_multiple_of(2)
                .then(|| planner.plan(n / 2, FftDirection::Forward)),
            packed_inverse: n
                .is_multiple_of(2)
                .then(|| planner.plan(n / 2, FftDirection::Inverse)),
            forward_factors,
            inverse_factors,
        }
    }

    /// Checked real-plan construction using a shared planner cache.
    pub fn try_with_planner(n: usize, planner: &CpuFftPlanner<T>) -> Result<Self, PlanLengthError> {
        if n == 0 {
            return Err(PlanLengthError::ZeroLength);
        }
        let forward = planner.try_plan(n, FftDirection::Forward)?;
        let inverse = planner.try_plan(n, FftDirection::Inverse)?;
        let packed_forward = if n.is_multiple_of(2) {
            Some(planner.try_plan(n / 2, FftDirection::Forward)?)
        } else {
            None
        };
        let packed_inverse = if n.is_multiple_of(2) {
            Some(planner.try_plan(n / 2, FftDirection::Inverse)?)
        } else {
            None
        };
        let (forward_factors, inverse_factors) = build_recombination_factors(n);
        Ok(Self {
            n,
            forward,
            inverse,
            packed_forward,
            packed_inverse,
            forward_factors,
            inverse_factors,
        })
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Number of stored bins, including DC and the Nyquist bin when present.
    pub fn spectrum_len(&self) -> usize {
        self.n / 2 + 1
    }

    /// Complex workspace required by the allocation-free methods.
    pub fn scratch_len(&self) -> usize {
        if let (Some(forward), Some(inverse)) = (&self.packed_forward, &self.packed_inverse) {
            self.n / 2 + forward.scratch_len().max(inverse.scratch_len())
        } else {
            self.n + self.forward.scratch_len().max(self.inverse.scratch_len())
        }
    }

    /// Forward real transform, allocating temporary workspace.
    pub fn forward(&self, input: &[T]) -> Vec<Complex<T>> {
        let mut output = vec![Complex::new(T::zero(), T::zero()); self.spectrum_len()];
        let mut scratch = vec![Complex::new(T::zero(), T::zero()); self.scratch_len()];
        self.forward_with_scratch(input, &mut output, &mut scratch);
        output
    }

    /// Inverse real transform, allocating temporary workspace.
    pub fn inverse(&self, input: &[Complex<T>]) -> Vec<T> {
        let mut output = vec![T::zero(); self.n];
        let mut scratch = vec![Complex::new(T::zero(), T::zero()); self.scratch_len()];
        self.inverse_with_scratch(input, &mut output, &mut scratch);
        output
    }

    /// Allocation-free forward transform for one real signal.
    pub fn forward_with_scratch(
        &self,
        input: &[T],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        assert_eq!(input.len(), self.n, "real FFT input length mismatch");
        assert_eq!(
            output.len(),
            self.spectrum_len(),
            "real FFT output length mismatch"
        );
        assert!(
            scratch.len() >= self.scratch_len(),
            "real FFT scratch is too short"
        );
        if let Some(packed) = &self.packed_forward {
            let m = self.n / 2;
            let (work, fft_scratch) = scratch.split_at_mut(m);
            let (pairs, _) = input.as_chunks::<2>();
            for (dst, pair) in work.iter_mut().zip(pairs) {
                *dst = Complex::new(pair[0], pair[1]);
            }
            packed.process_with_scratch(work, &mut fft_scratch[..packed.scratch_len()]);
            let scale = T::from_f64(0.5).unwrap();
            // Endpoints k = 0 and k = m (Nyquist) both draw on work[0];
            // the interior uses direct indices. (The previous form paid
            // two integer modulos per bin in this hot loop.)
            let recombine = |a: Complex<T>, b_raw: Complex<T>, factor: Complex<T>| {
                let b = Complex::new(b_raw.re, -b_raw.im);
                let sum = a + b;
                let diff = a - b;
                // -i exp(i theta) = sin(theta) - i cos(theta), precomputed.
                (sum + factor * diff) * scale
            };
            output[0] = recombine(work[0], work[0], self.forward_factors[0]);
            // AVX2 kernels handle the interior bins in mirror pairs on
            // x86_64; they return the first middle index they did not
            // cover, and the scalar tail finishes that middle range.
            let tail_start = recombine_forward_avx(work, &self.forward_factors, output);
            for k in tail_start..=(m - tail_start) {
                output[k] = recombine(work[k], work[m - k], self.forward_factors[k]);
            }
            if m > 0 {
                output[m] = recombine(work[0], work[0], self.forward_factors[m]);
            }
        } else {
            let (work, fft_scratch) = scratch.split_at_mut(self.n);
            for (dst, &sample) in work.iter_mut().zip(input) {
                *dst = Complex::new(sample, T::zero());
            }
            self.forward
                .process_with_scratch(work, &mut fft_scratch[..self.forward.scratch_len()]);
            output.copy_from_slice(&work[..self.spectrum_len()]);
        }
    }

    /// Allocation-free inverse transform for one Hermitian half-spectrum.
    pub fn inverse_with_scratch(
        &self,
        input: &[Complex<T>],
        output: &mut [T],
        scratch: &mut [Complex<T>],
    ) {
        assert_eq!(
            input.len(),
            self.spectrum_len(),
            "real FFT spectrum length mismatch"
        );
        assert_eq!(output.len(), self.n, "real FFT output length mismatch");
        assert!(
            scratch.len() >= self.scratch_len(),
            "real FFT scratch is too short"
        );
        if let Some(packed) = &self.packed_inverse {
            let m = self.n / 2;
            let (work, fft_scratch) = scratch.split_at_mut(m);
            let x0 = input[0].re;
            let xm = input[m].re;
            work[0] = Complex::new(
                (x0 + xm) * T::from_f64(0.5).unwrap(),
                (x0 - xm) * T::from_f64(0.5).unwrap(),
            );
            // AVX2 kernels handle the interior bins in mirror pairs on
            // x86_64; the scalar tail finishes the middle range.
            let tail_start = recombine_inverse_avx(input, &self.inverse_factors, work);
            for k in tail_start..=(m - tail_start) {
                let x = input[k];
                let mirror = Complex::new(input[m - k].re, -input[m - k].im);
                let sum = x + mirror;
                let diff = x - mirror;
                // i exp(+i theta) = -sin(theta) + i cos(theta), precomputed.
                let factor = self.inverse_factors[k];
                work[k] = (sum + factor * diff) * T::from_f64(0.5).unwrap();
            }
            packed.process_with_scratch(work, &mut fft_scratch[..packed.scratch_len()]);
            let scale = T::from_f64(2.0 / self.n as f64).unwrap();
            for j in 0..m {
                output[2 * j] = work[j].re * scale;
                output[2 * j + 1] = work[j].im * scale;
            }
        } else {
            let (work, fft_scratch) = scratch.split_at_mut(self.n);
            for (dst, &value) in work.iter_mut().zip(input) {
                *dst = value;
            }
            let half = self.n / 2;
            work[0].im = T::zero();
            for k in (half + 1)..self.n {
                let source = self.n - k;
                work[k] = Complex::new(work[source].re, -work[source].im);
            }
            self.inverse
                .process_with_scratch(work, &mut fft_scratch[..self.inverse.scratch_len()]);
            let scale = T::from_f64(1.0 / self.n as f64).unwrap();
            for (dst, value) in output.iter_mut().zip(work.iter()) {
                *dst = value.re * scale;
            }
        }
    }

    /// Forward a contiguous batch of real signals using one reusable scratch
    /// buffer.
    pub fn forward_batched_with_scratch(
        &self,
        input: &[T],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        assert_eq!(
            input.len() % self.n,
            0,
            "real FFT input is not batched by N"
        );
        assert_eq!(
            output.len() % self.spectrum_len(),
            0,
            "real FFT output is not batched"
        );
        assert_eq!(input.len() / self.n, output.len() / self.spectrum_len());
        for (samples, spectrum) in input
            .chunks_exact(self.n)
            .zip(output.chunks_exact_mut(self.spectrum_len()))
        {
            self.forward_with_scratch(samples, spectrum, scratch);
        }
    }

    /// Inverse a contiguous batch of half-spectra using one reusable scratch
    /// buffer.
    pub fn inverse_batched_with_scratch(
        &self,
        input: &[Complex<T>],
        output: &mut [T],
        scratch: &mut [Complex<T>],
    ) {
        assert_eq!(
            input.len() % self.spectrum_len(),
            0,
            "real FFT spectrum is not batched"
        );
        assert_eq!(
            output.len() % self.n,
            0,
            "real FFT output is not batched by N"
        );
        assert_eq!(input.len() / self.spectrum_len(), output.len() / self.n);
        for (spectrum, samples) in input
            .chunks_exact(self.spectrum_len())
            .zip(output.chunks_exact_mut(self.n))
        {
            self.inverse_with_scratch(spectrum, samples, scratch);
        }
    }
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{__m256, __m256d};

/// AoS complex multiply for two interleaved `Complex<f64>` lanes.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn cmul_aos_f64(a: __m256d, b: __m256d) -> __m256d {
    use std::arch::x86_64::*;
    unsafe {
        let a_re = _mm256_shuffle_pd(a, a, 0x00);
        let a_im = _mm256_shuffle_pd(a, a, 0x0F);
        let b_swap = _mm256_shuffle_pd(b, b, 0x05);
        let p1 = _mm256_mul_pd(a_re, b);
        let p2 = _mm256_mul_pd(a_im, b_swap);
        _mm256_addsub_pd(p1, p2)
    }
}

/// AoS complex multiply for four interleaved `Complex<f32>` lanes.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn cmul_aos_f32(a: __m256, b: __m256) -> __m256 {
    use std::arch::x86_64::*;
    unsafe {
        let a_re = _mm256_shuffle_ps(a, a, 0xA0);
        let a_im = _mm256_shuffle_ps(a, a, 0xF5);
        let b_swap = _mm256_shuffle_ps(b, b, 0xB1);
        let p1 = _mm256_mul_ps(a_re, b);
        let p2 = _mm256_mul_ps(a_im, b_swap);
        _mm256_addsub_ps(p1, p2)
    }
}

/// Forward recombination over the interior bins `1..m`, computing each
/// bin and its mirror in one vector pass. Returns the first middle index
/// not covered; the caller finishes that range scalar-ly. Returns 1 when
/// no vector path applied (non-x86, no AVX2/FMA, or other scalar types).
///
/// The mirror identity `factor[m - k] == conj(factor[k])` (with the
/// factor layout built by [`build_recombination_factors`]) lets one pass
/// emit both `output[k]` and `output[m - k]`.
fn recombine_forward_avx<T: FftScalar>(
    work: &[Complex<T>],
    factors: &[Complex<T>],
    output: &mut [Complex<T>],
) -> usize {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            assert_eq!(std::mem::size_of::<Complex<T>>(), std::mem::size_of::<Complex<f64>>());
            assert_eq!(std::mem::align_of::<Complex<T>>(), std::mem::align_of::<Complex<f64>>());
            // SAFETY: TypeId equality proves T = f64 and the layout
            // asserts above pin the transmute targets; AVX2+FMA were just
            // detected at runtime.
            unsafe {
                let work64: &[Complex<f64>] =
                    std::mem::transmute::<&[Complex<T>], &[Complex<f64>]>(work);
                let factors64: &[Complex<f64>] =
                    std::mem::transmute::<&[Complex<T>], &[Complex<f64>]>(factors);
                let output64: &mut [Complex<f64>] =
                    std::mem::transmute::<&mut [Complex<T>], &mut [Complex<f64>]>(output);
                return recombine_forward_f64_avx(work64, factors64, output64);
            }
        }
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            assert_eq!(std::mem::size_of::<Complex<T>>(), std::mem::size_of::<Complex<f32>>());
            assert_eq!(std::mem::align_of::<Complex<T>>(), std::mem::align_of::<Complex<f32>>());
            // SAFETY: TypeId equality proves T = f32 and the layout
            // asserts above pin the transmute targets; AVX2+FMA were just
            // detected at runtime.
            unsafe {
                let work32: &[Complex<f32>] =
                    std::mem::transmute::<&[Complex<T>], &[Complex<f32>]>(work);
                let factors32: &[Complex<f32>] =
                    std::mem::transmute::<&[Complex<T>], &[Complex<f32>]>(factors);
                let output32: &mut [Complex<f32>] =
                    std::mem::transmute::<&mut [Complex<T>], &mut [Complex<f32>]>(output);
                return recombine_forward_f32_avx(work32, factors32, output32);
            }
        }
    }
    1
}

/// AVX2 forward recombination for f64: two bins and their mirrors per
/// iteration. See [`recombine_forward_avx`] for the contract.
///
/// # Safety
///
/// The caller must provide AVX2 and FMA at runtime. `factors.len()` and
/// `output.len()` must both be at least `work.len() + 1`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn recombine_forward_f64_avx(
    work: &[Complex<f64>],
    factors: &[Complex<f64>],
    output: &mut [Complex<f64>],
) -> usize {
    use std::arch::x86_64::*;
    let m = work.len();
    unsafe {
        let conj_mask = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);
        let half = _mm256_set1_pd(0.5);
        let mut k = 1usize;
        while 2 * (k + 1) < m {
            let a = _mm256_loadu_pd(work.as_ptr().add(k) as *const f64);
            let b_rev = _mm256_loadu_pd(work.as_ptr().add(m - k - 1) as *const f64);
            let b = _mm256_permute2f128_pd(b_rev, b_rev, 0x01);
            let f = _mm256_loadu_pd(factors.as_ptr().add(k) as *const f64);
            let conj_a = _mm256_xor_pd(a, conj_mask);
            let conj_b = _mm256_xor_pd(b, conj_mask);
            let s1 = _mm256_add_pd(a, conj_b);
            let d1 = _mm256_sub_pd(a, conj_b);
            let out_a = _mm256_mul_pd(_mm256_add_pd(s1, cmul_aos_f64(f, d1)), half);
            let s2 = _mm256_add_pd(b, conj_a);
            let d2 = _mm256_sub_pd(b, conj_a);
            let f_conj = _mm256_xor_pd(f, conj_mask);
            let out_b = _mm256_mul_pd(_mm256_add_pd(s2, cmul_aos_f64(f_conj, d2)), half);
            _mm256_storeu_pd(output.as_mut_ptr().add(k) as *mut f64, out_a);
            let out_b_rev = _mm256_permute2f128_pd(out_b, out_b, 0x01);
            _mm256_storeu_pd(output.as_mut_ptr().add(m - k - 1) as *mut f64, out_b_rev);
            k += 2;
        }
        k
    }
}

/// AVX2 forward recombination for f32: four bins and their mirrors per
/// iteration. See [`recombine_forward_avx`] for the contract.
///
/// # Safety
///
/// The caller must provide AVX2 and FMA at runtime. `factors.len()` and
/// `output.len()` must both be at least `work.len() + 1`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn recombine_forward_f32_avx(
    work: &[Complex<f32>],
    factors: &[Complex<f32>],
    output: &mut [Complex<f32>],
) -> usize {
    use std::arch::x86_64::*;
    let m = work.len();
    unsafe {
        let conj_mask = _mm256_set_ps(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0);
        let half = _mm256_set1_ps(0.5);
        let mut k = 1usize;
        while 2 * (k + 3) < m {
            let a = _mm256_loadu_ps(work.as_ptr().add(k) as *const f32);
            let b_rev = _mm256_loadu_ps(work.as_ptr().add(m - k - 3) as *const f32);
            let b = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(b_rev), 0x1B));
            let f = _mm256_loadu_ps(factors.as_ptr().add(k) as *const f32);
            let conj_a = _mm256_xor_ps(a, conj_mask);
            let conj_b = _mm256_xor_ps(b, conj_mask);
            let s1 = _mm256_add_ps(a, conj_b);
            let d1 = _mm256_sub_ps(a, conj_b);
            let out_a = _mm256_mul_ps(_mm256_add_ps(s1, cmul_aos_f32(f, d1)), half);
            let s2 = _mm256_add_ps(b, conj_a);
            let d2 = _mm256_sub_ps(b, conj_a);
            let f_conj = _mm256_xor_ps(f, conj_mask);
            let out_b = _mm256_mul_ps(_mm256_add_ps(s2, cmul_aos_f32(f_conj, d2)), half);
            _mm256_storeu_ps(output.as_mut_ptr().add(k) as *mut f32, out_a);
            let out_b_rev =
                _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(out_b), 0x1B));
            _mm256_storeu_ps(output.as_mut_ptr().add(m - k - 3) as *mut f32, out_b_rev);
            k += 4;
        }
        k
    }
}

/// Inverse recombination over the interior bins `1..m`, computing each
/// bin and its mirror in one vector pass. Returns the first middle index
/// not covered (1 when no vector path applied). The mirror identity
/// `factor[m - k] == conj(factor[k])` also holds for the inverse factor
/// layout, and the mirror bin satisfies
/// `work[m - k] = conj((sum - factor[k] * diff) * 0.5)` in terms of the
/// bin-`k` sum/difference, which is what the kernels emit.
fn recombine_inverse_avx<T: FftScalar>(
    input: &[Complex<T>],
    factors: &[Complex<T>],
    work: &mut [Complex<T>],
) -> usize {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            assert_eq!(std::mem::size_of::<Complex<T>>(), std::mem::size_of::<Complex<f64>>());
            assert_eq!(std::mem::align_of::<Complex<T>>(), std::mem::align_of::<Complex<f64>>());
            // SAFETY: TypeId equality proves T = f64 and the layout
            // asserts above pin the transmute targets; AVX2+FMA were just
            // detected at runtime.
            unsafe {
                let input64: &[Complex<f64>] =
                    std::mem::transmute::<&[Complex<T>], &[Complex<f64>]>(input);
                let factors64: &[Complex<f64>] =
                    std::mem::transmute::<&[Complex<T>], &[Complex<f64>]>(factors);
                let work64: &mut [Complex<f64>] =
                    std::mem::transmute::<&mut [Complex<T>], &mut [Complex<f64>]>(work);
                return recombine_inverse_f64_avx(input64, factors64, work64);
            }
        }
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            assert_eq!(std::mem::size_of::<Complex<T>>(), std::mem::size_of::<Complex<f32>>());
            assert_eq!(std::mem::align_of::<Complex<T>>(), std::mem::align_of::<Complex<f32>>());
            // SAFETY: TypeId equality proves T = f32 and the layout
            // asserts above pin the transmute targets; AVX2+FMA were just
            // detected at runtime.
            unsafe {
                let input32: &[Complex<f32>] =
                    std::mem::transmute::<&[Complex<T>], &[Complex<f32>]>(input);
                let factors32: &[Complex<f32>] =
                    std::mem::transmute::<&[Complex<T>], &[Complex<f32>]>(factors);
                let work32: &mut [Complex<f32>] =
                    std::mem::transmute::<&mut [Complex<T>], &mut [Complex<f32>]>(work);
                return recombine_inverse_f32_avx(input32, factors32, work32);
            }
        }
    }
    1
}

/// AVX2 inverse recombination for f64: two bins and their mirrors per
/// iteration. See [`recombine_inverse_avx`] for the contract.
///
/// # Safety
///
/// The caller must provide AVX2 and FMA at runtime. `input.len()` must be
/// at least `work.len() + 1` and `factors.len()` at least `work.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn recombine_inverse_f64_avx(
    input: &[Complex<f64>],
    factors: &[Complex<f64>],
    work: &mut [Complex<f64>],
) -> usize {
    use std::arch::x86_64::*;
    let m = work.len();
    unsafe {
        let conj_mask = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);
        let half = _mm256_set1_pd(0.5);
        let mut k = 1usize;
        while 2 * (k + 1) < m {
            let a = _mm256_loadu_pd(input.as_ptr().add(k) as *const f64);
            let b_rev = _mm256_loadu_pd(input.as_ptr().add(m - k - 1) as *const f64);
            let b = _mm256_permute2f128_pd(b_rev, b_rev, 0x01);
            let f = _mm256_loadu_pd(factors.as_ptr().add(k) as *const f64);
            let conj_b = _mm256_xor_pd(b, conj_mask);
            let sum = _mm256_add_pd(a, conj_b);
            let diff = _mm256_sub_pd(a, conj_b);
            let prod = cmul_aos_f64(f, diff);
            let out_a = _mm256_mul_pd(_mm256_add_pd(sum, prod), half);
            let out_b = _mm256_xor_pd(_mm256_mul_pd(_mm256_sub_pd(sum, prod), half), conj_mask);
            _mm256_storeu_pd(work.as_mut_ptr().add(k) as *mut f64, out_a);
            let out_b_rev = _mm256_permute2f128_pd(out_b, out_b, 0x01);
            _mm256_storeu_pd(work.as_mut_ptr().add(m - k - 1) as *mut f64, out_b_rev);
            k += 2;
        }
        k
    }
}

/// AVX2 inverse recombination for f32: four bins and their mirrors per
/// iteration. See [`recombine_inverse_avx`] for the contract.
///
/// # Safety
///
/// The caller must provide AVX2 and FMA at runtime. `input.len()` must be
/// at least `work.len() + 1` and `factors.len()` at least `work.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn recombine_inverse_f32_avx(
    input: &[Complex<f32>],
    factors: &[Complex<f32>],
    work: &mut [Complex<f32>],
) -> usize {
    use std::arch::x86_64::*;
    let m = work.len();
    unsafe {
        let conj_mask = _mm256_set_ps(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0);
        let half = _mm256_set1_ps(0.5);
        let mut k = 1usize;
        while 2 * (k + 3) < m {
            let a = _mm256_loadu_ps(input.as_ptr().add(k) as *const f32);
            let b_rev = _mm256_loadu_ps(input.as_ptr().add(m - k - 3) as *const f32);
            let b = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(b_rev), 0x1B));
            let f = _mm256_loadu_ps(factors.as_ptr().add(k) as *const f32);
            let conj_b = _mm256_xor_ps(b, conj_mask);
            let sum = _mm256_add_ps(a, conj_b);
            let diff = _mm256_sub_ps(a, conj_b);
            let prod = cmul_aos_f32(f, diff);
            let out_a = _mm256_mul_ps(_mm256_add_ps(sum, prod), half);
            let out_b = _mm256_xor_ps(_mm256_mul_ps(_mm256_sub_ps(sum, prod), half), conj_mask);
            _mm256_storeu_ps(work.as_mut_ptr().add(k) as *mut f32, out_a);
            let out_b_rev =
                _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(out_b), 0x1B));
            _mm256_storeu_ps(work.as_mut_ptr().add(m - k - 3) as *mut f32, out_b_rev);
            k += 4;
        }
        k
    }
}

/// Planner facade for real transforms.
pub struct RealFftPlanner<T: FftScalar> {
    inner: CpuFftPlanner<T>,
}

impl<T: FftScalar> Default for RealFftPlanner<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FftScalar> RealFftPlanner<T> {
    pub fn new() -> Self {
        Self {
            inner: CpuFftPlanner::new(),
        }
    }

    pub fn plan_fft_forward(&self, n: usize) -> Arc<RealFftPlan<T>> {
        Arc::new(RealFftPlan::with_planner(n, &self.inner))
    }

    pub fn plan_fft_inverse(&self, n: usize) -> Arc<RealFftPlan<T>> {
        self.plan_fft_forward(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num_complex::Complex64;

    #[test]
    fn real_round_trip_even_and_odd() {
        for &n in &[1usize, 2, 3, 8, 9, 15, 16, 31] {
            let plan = RealFftPlan::<f64>::new(n);
            let input: Vec<f64> = (0..n)
                .map(|i| (0.13 * i as f64).sin() + 0.2 * i as f64)
                .collect();
            let spectrum = plan.forward(&input);
            let output = plan.inverse(&spectrum);
            for (actual, expected) in output.iter().zip(input.iter()) {
                assert_relative_eq!(actual, expected, max_relative = 1e-10, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn real_batched_scratch_matches_allocating_path() {
        let n = 12;
        let plan = RealFftPlan::<f32>::new(n);
        let input: Vec<f32> = (0..n * 3).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut output = vec![Complex::new(0.0, 0.0); plan.spectrum_len() * 3];
        let mut scratch = vec![Complex::new(0.0, 0.0); plan.scratch_len()];
        plan.forward_batched_with_scratch(&input, &mut output, &mut scratch);
        for (chunk, samples) in output
            .chunks_exact(plan.spectrum_len())
            .zip(input.chunks_exact(n))
        {
            assert_eq!(chunk, plan.forward(samples).as_slice());
        }
    }

    #[test]
    fn packed_forward_matches_direct_real_dft() {
        for &n in &[2usize, 4, 6, 8, 16] {
            let plan = RealFftPlan::<f64>::new(n);
            let input: Vec<f64> = (0..n).map(|i| (0.19 * i as f64).cos()).collect();
            let spectrum = plan.forward(&input);
            for (k, spectrum_value) in spectrum.iter().enumerate().take(n / 2 + 1) {
                let mut expected = Complex64::new(0.0, 0.0);
                for (j, &sample) in input.iter().enumerate() {
                    let theta = -std::f64::consts::TAU * j as f64 * k as f64 / n as f64;
                    let (sin, cos) = theta.sin_cos();
                    expected += Complex64::new(cos, sin) * sample;
                }
                assert_relative_eq!(spectrum_value.re, expected.re, epsilon = 1e-10);
                assert_relative_eq!(spectrum_value.im, expected.im, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn real_parseval_uses_half_spectrum_weights() {
        let n = 32;
        let plan = RealFftPlan::<f64>::new(n);
        let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.31).sin()).collect();
        let spectrum = plan.forward(&input);
        let mut spectral_energy = spectrum[0].norm_sqr() + spectrum[n / 2].norm_sqr();
        for value in &spectrum[1..n / 2] {
            spectral_energy += 2.0 * value.norm_sqr();
        }
        let time_energy: f64 = input.iter().map(|value| value * value).sum();
        assert_relative_eq!(spectral_energy / n as f64, time_energy, epsilon = 1e-10);
    }
}
