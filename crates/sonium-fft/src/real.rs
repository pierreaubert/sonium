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
            for j in 0..m {
                work[j] = Complex::new(input[2 * j], input[2 * j + 1]);
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
            for k in 1..m {
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
            for k in 1..m {
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
