//! Cache-friendly Bailey six-step decomposition for large transforms.
//!
//! For `N = N1*N2`, the implementation reshapes samples as
//! `n1 + N1*n2`, performs the `N2` transforms, applies the Cooley–Tukey
//! twiddle, then performs the `N1` transforms and writes natural output order
//! `k2 + N2*k1`. A full intermediate plane plus one axis buffer is supplied by
//! the caller, so steady-state execution is allocation-free.

use num_complex::Complex;
use std::f64::consts::TAU;
use std::sync::Arc;
use std::sync::Mutex;

use crate::FftDirection;
use crate::FftPlan;
use crate::cpu::planner::CpuFftPlanner;
use crate::scalar::FftScalar;
use sonium_compute::transpose_2d;

/// Six-step plan for a single transform length, with contiguous batching.
pub struct SixStepFftPlan<T: FftScalar> {
    n: usize,
    n1: usize,
    n2: usize,
    direction: FftDirection,
    axis_n1: Arc<dyn FftPlan<T>>,
    axis_n2: Arc<dyn FftPlan<T>>,
    /// Cooley-Tukey twiddles `W_N^(n1_index * k2)` at
    /// `twiddles[n1_index * n2 + k2]`, precomputed once at plan time with a
    /// batched `sincos` so steady-state execution performs no trigonometry
    /// and no `from_f64` conversions.
    twiddles: Vec<Complex<T>>,
    process_scratch: Mutex<Vec<Complex<T>>>,
}

impl<T: FftScalar> SixStepFftPlan<T> {
    pub fn new(n: usize, direction: FftDirection) -> Self {
        assert!(n > 0, "six-step FFT length must be positive");
        let planner = CpuFftPlanner::<T>::new();
        Self::with_planner(n, direction, &planner)
    }

    pub fn with_planner(n: usize, direction: FftDirection, planner: &CpuFftPlanner<T>) -> Self {
        assert!(n > 0, "six-step FFT length must be positive");
        let n1 = nearest_divisor(n);
        let n2 = n / n1;
        let axis_n1 = planner.plan_baseline(n1, direction);
        let axis_n2 = planner.plan_baseline(n2, direction);
        let axis_work_len = (n1 + axis_n1.scratch_len()).max(n2 + axis_n2.scratch_len());
        Self {
            n,
            n1,
            n2,
            direction,
            axis_n1,
            axis_n2,
            twiddles: build_six_step_twiddles(n, n1, n2, direction),
            process_scratch: Mutex::new(vec![
                Complex::new(T::zero(), T::zero());
                n + axis_work_len
            ]),
        }
    }

    pub fn factors(&self) -> (usize, usize) {
        (self.n1, self.n2)
    }

    pub fn scratch_len(&self) -> usize {
        self.n + self.axis_work_len()
    }

    fn axis_work_len(&self) -> usize {
        (self.n1 + self.axis_n1.scratch_len()).max(self.n2 + self.axis_n2.scratch_len())
    }

    fn process_one(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        let (work, axis) = scratch.split_at_mut(self.n);
        transpose_2d(buffer, work, self.n2, self.n1)
            .expect("six-step scratch dimensions are validated at construction");
        // The axis plans can themselves be Bluestein plans. Keep their
        // transform buffer and scratch in one caller-owned region so the
        // six-step wrapper preserves its allocation-free contract for every
        // factorisation, not only smooth axes.

        // First axis: N2-point transforms for each n1 coordinate, followed by
        // the Cooley–Tukey twiddle W_N^(n1*k2) from the plan-time table.
        let (phase_scratch, axis_fft_scratch) = axis.split_at_mut(self.n2);
        for n1_index in 0..self.n1 {
            phase_scratch[..self.n2]
                .copy_from_slice(&work[n1_index * self.n2..(n1_index + 1) * self.n2]);
            self.axis_n2.process_with_scratch(
                &mut phase_scratch[..self.n2],
                &mut axis_fft_scratch[..self.axis_n2.scratch_len()],
            );
            let twiddle_row = &self.twiddles[n1_index * self.n2..(n1_index + 1) * self.n2];
            for (slot, (value, twiddle)) in work[n1_index * self.n2..(n1_index + 1) * self.n2]
                .iter_mut()
                .zip(phase_scratch.iter().zip(twiddle_row.iter()))
            {
                *slot = *value * *twiddle;
            }
        }

        // Second axis: transpose so each k2 vector is contiguous, transform
        // rows, then transpose back into the caller's row-major layout.
        transpose_2d(work, buffer, self.n1, self.n2)
            .expect("six-step transpose dimensions are validated at construction");
        let (phase_scratch, axis_fft_scratch) = axis.split_at_mut(self.n1);
        for k2 in 0..self.n2 {
            phase_scratch[..self.n1].copy_from_slice(&buffer[k2 * self.n1..(k2 + 1) * self.n1]);
            self.axis_n1.process_with_scratch(
                &mut phase_scratch[..self.n1],
                &mut axis_fft_scratch[..self.axis_n1.scratch_len()],
            );
            for k1 in 0..self.n1 {
                work[k2 * self.n1 + k1] = phase_scratch[k1];
            }
        }
        transpose_2d(work, buffer, self.n2, self.n1)
            .expect("six-step transpose dimensions are validated at construction");
    }
}

impl<T: FftScalar> FftPlan<T> for SixStepFftPlan<T> {
    fn len(&self) -> usize {
        self.n
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn process(&self, buffer: &mut [Complex<T>]) {
        if buffer.is_empty() {
            return;
        }
        assert_eq!(
            buffer.len() % self.n,
            0,
            "buffer length must be multiple plan length"
        );
        let batch = buffer.len() / self.n;
        if let Some(workers) = crate::cpu::plan::parallel_workers(batch, buffer.len()) {
            // Large batch: one scoped worker per chunk group, each with
            // owned scratch. This keeps the shared `process_scratch` mutex
            // off the hot path; `process_one` only reads shared plan state.
            let scratch_len = self.scratch_len();
            let counts = crate::cpu::plan::split_batch(batch, workers);
            let mut rest = buffer;
            std::thread::scope(|scope| {
                for (index, count) in counts.into_iter().enumerate() {
                    let (head, tail) = rest.split_at_mut(count * self.n);
                    rest = tail;
                    scope.spawn(move || {
                        sonium_topology::affinity::pin_physical_worker(index);
                        let mut local = vec![Complex::new(T::zero(), T::zero()); scratch_len];
                        for chunk in head.chunks_exact_mut(self.n) {
                            self.process_one(chunk, &mut local);
                        }
                    });
                }
            });
            return;
        }
        let mut scratch = self
            .process_scratch
            .lock()
            .expect("six-step process scratch poisoned");
        for chunk in buffer.chunks_exact_mut(self.n) {
            self.process_one(chunk, &mut scratch[..]);
        }
    }

    fn scratch_len(&self) -> usize {
        self.scratch_len()
    }

    fn process_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        if buffer.is_empty() {
            return;
        }
        assert_eq!(
            buffer.len() % self.n,
            0,
            "buffer length must be multiple plan length"
        );
        assert!(
            scratch.len() >= self.scratch_len(),
            "six-step scratch is too short"
        );
        for chunk in buffer.chunks_exact_mut(self.n) {
            self.process_one(chunk, scratch);
        }
    }
}

fn nearest_divisor(n: usize) -> usize {
    let mut candidate = (n as f64).sqrt() as usize;
    candidate = candidate.max(1);
    while candidate > 1 && !n.is_multiple_of(candidate) {
        candidate -= 1;
    }
    candidate
}

/// Precompute the `n1 * n2` Cooley-Tukey twiddles `W_N^(n1_index * k2)` at
/// plan time, batched through one SIMD `sincos` sweep instead of per-element
/// scalar trigonometry at execution time.
fn build_six_step_twiddles<T: FftScalar>(
    n: usize,
    n1: usize,
    n2: usize,
    direction: FftDirection,
) -> Vec<Complex<T>> {
    let sign = match direction {
        FftDirection::Forward => -1.0,
        FftDirection::Inverse => 1.0,
    };
    let mut thetas = Vec::with_capacity(n);
    for n1_index in 0..n1 {
        for k2 in 0..n2 {
            thetas.push(sign * TAU * n1_index as f64 * k2 as f64 / n as f64);
        }
    }
    let mut sins = vec![0.0_f64; n];
    let mut coss = vec![0.0_f64; n];
    crate::sincos::fill_sincos(&thetas, &mut sins, &mut coss);
    sins.into_iter()
        .zip(coss)
        .map(|(sin, cos)| {
            Complex::new(
                T::from_f64(cos).expect("FFT scalar must represent cosine values"),
                T::from_f64(sin).expect("FFT scalar must represent sine values"),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num_complex::Complex64;

    #[test]
    fn six_step_matches_round_trip_and_direct_plan() {
        for &n in &[1usize, 2, 6, 15, 32, 63, 77, 128] {
            let plan = SixStepFftPlan::<f64>::new(n, FftDirection::Forward);
            let inverse = SixStepFftPlan::<f64>::new(n, FftDirection::Inverse);
            let reference = CpuFftPlanner::<f64>::new().plan(n, FftDirection::Forward);
            let input: Vec<Complex64> = (0..n)
                .map(|i| Complex64::new((i as f64 * 0.11).sin(), (i as f64 * 0.07).cos()))
                .collect();
            let mut actual = input.clone();
            let mut expected = input.clone();
            let mut scratch = vec![Complex64::new(0.0, 0.0); plan.scratch_len()];
            plan.process_with_scratch(&mut actual, &mut scratch);
            reference.process(&mut expected);
            for (got, want) in actual.iter().zip(expected.iter()) {
                assert_relative_eq!(got.re, want.re, epsilon = 1e-10, max_relative = 1e-10);
                assert_relative_eq!(got.im, want.im, epsilon = 1e-10, max_relative = 1e-10);
            }
            inverse.process_with_scratch(&mut actual, &mut scratch);
            for (got, want) in actual.iter().zip(input.iter()) {
                assert_relative_eq!(got.re / n as f64, want.re, epsilon = 1e-10);
                assert_relative_eq!(got.im / n as f64, want.im, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn six_step_prime_axes_use_caller_scratch() {
        let plan = SixStepFftPlan::<f64>::new(77, FftDirection::Forward);
        assert_eq!(plan.factors(), (7, 11));
        let reference = CpuFftPlanner::<f64>::new().plan(77, FftDirection::Forward);
        let input: Vec<Complex64> = (0..77)
            .map(|i| Complex64::new((i as f64 * 0.19).sin(), (i as f64 * 0.07).cos()))
            .collect();
        let mut actual = input.clone();
        let mut expected = input;
        let mut scratch = vec![Complex64::new(0.0, 0.0); plan.scratch_len()];
        plan.process_with_scratch(&mut actual, &mut scratch);
        reference.process(&mut expected);
        for (got, want) in actual.iter().zip(expected) {
            assert_relative_eq!(got.re, want.re, epsilon = 1e-10, max_relative = 1e-10);
            assert_relative_eq!(got.im, want.im, epsilon = 1e-10, max_relative = 1e-10);
        }
    }

    #[test]
    fn cpu_planner_selects_six_step_above_cache_threshold() {
        let n = 131_072;
        let planner = CpuFftPlanner::<f32>::new();
        let forward = planner.plan(n, FftDirection::Forward);
        let inverse = planner.plan(n, FftDirection::Inverse);
        let mut buffer = vec![Complex::<f32>::new(0.0, 0.0); n];
        buffer[17] = Complex::new(1.0, -0.25);
        let mut scratch = vec![Complex::<f32>::new(0.0, 0.0); forward.scratch_len()];
        forward.process_with_scratch(&mut buffer, &mut scratch);
        inverse.process_with_scratch(&mut buffer, &mut scratch);
        assert!((buffer[17].re / n as f32 - 1.0).abs() < 2e-4);
        assert!((buffer[17].im / n as f32 + 0.25).abs() < 2e-4);
        for (index, value) in buffer.iter().enumerate() {
            if index != 17 {
                assert!(value.norm() / (n as f32) < 2e-4);
            }
        }
    }

    #[test]
    fn plan_time_twiddles_match_direct_trigonometry() {
        for direction in [FftDirection::Forward, FftDirection::Inverse] {
            let plan = SixStepFftPlan::<f64>::new(77, direction);
            let (n1, n2) = plan.factors();
            assert_eq!(plan.twiddles.len(), 77);
            let sign = match direction {
                FftDirection::Forward => -1.0,
                FftDirection::Inverse => 1.0,
            };
            for n1_index in 0..n1 {
                for k2 in 0..n2 {
                    let theta = sign * TAU * n1_index as f64 * k2 as f64 / 77.0;
                    let (sin, cos) = theta.sin_cos();
                    let table = plan.twiddles[n1_index * n2 + k2];
                    // fastmath polynomial vs libm agree to ~1e-13; keep the
                    // bound loose enough for the SIMD path on any target.
                    assert!(
                        (table.re - cos).abs() < 1e-12 && (table.im - sin).abs() < 1e-12,
                        "twiddle mismatch at ({n1_index}, {k2})"
                    );
                }
            }
        }
    }

    #[test]
    fn hoisted_twiddles_agree_with_baseline_at_scale() {
        // 20000 = 125 x 160: large enough that the plan-time table (not a
        // handful of scalar trig calls) dominates the twiddle path, but both
        // axes stay Stockham so the test is fast in debug builds too.
        let n = 20_000;
        let plan = SixStepFftPlan::<f64>::new(n, FftDirection::Forward);
        assert_eq!(plan.factors(), (125, 160));
        let planner = CpuFftPlanner::<f64>::new();
        let reference = planner.plan_baseline(n, FftDirection::Forward);
        let input: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.013).sin(), (i as f64 * 0.029).cos()))
            .collect();
        let mut actual = input.clone();
        let mut expected = input.clone();
        let mut scratch = vec![Complex64::new(0.0, 0.0); plan.scratch_len()];
        plan.process_with_scratch(&mut actual, &mut scratch);
        reference.process(&mut expected);
        let scale = expected
            .iter()
            .map(|value| value.norm())
            .fold(1.0_f64, f64::max);
        for (got, want) in actual.iter().zip(expected.iter()) {
            assert!(
                (got - want).norm() <= 1e-9 * scale,
                "six-step/baseline mismatch: {got} vs {want}"
            );
        }
    }

    #[test]
    fn process_reuses_owned_scratch_without_growth() {
        let plan = SixStepFftPlan::<f64>::new(128, FftDirection::Forward);
        let (initial_ptr, initial_capacity) = {
            let scratch = plan
                .process_scratch
                .lock()
                .expect("six-step process scratch poisoned");
            (scratch.as_ptr(), scratch.capacity())
        };
        let mut input = vec![Complex64::new(0.0, 0.0); 128];
        plan.process(&mut input);
        plan.process(&mut input);
        let scratch = plan
            .process_scratch
            .lock()
            .expect("six-step process scratch poisoned");
        assert_eq!(scratch.as_ptr(), initial_ptr);
        assert_eq!(scratch.capacity(), initial_capacity);
    }
}
