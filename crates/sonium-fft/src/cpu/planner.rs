//! [`CpuFftPlanner`] — caches plans by `(length, direction)` so building
//! `n` plans for the same shape is O(1) after the first.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::FftDirection;
use crate::cpu::plan::{BluesteinFftPlan, StockhamFftPlan};
use crate::factor::{PlanLengthError, Strategy, plan_length, try_plan_length};
use crate::scalar::FftScalar;
use crate::six_step::SixStepFftPlan;
use crate::{FftPlan, FftPlanError};

type PlanCache<T> = RwLock<HashMap<(usize, FftDirection), Arc<dyn FftPlan<T>>>>;

/// Length above which the planner auto-selects the six-step decomposition
/// instead of the ping-pong Stockham driver. On x86_64, criterion
/// measurements (`fft_compare`) show the Stockham driver ahead of the
/// current six-step implementation at every measured size (six-step
/// trails by 1.7× or more at 4096–4M points, more for f32), so the
/// crossover sits at the edge of the measured range. Other architectures
/// keep the historical 64K crossover. `SixStepFftPlan` remains available as
/// an explicit plan choice on every architecture.
#[cfg(target_arch = "x86_64")]
const SIX_STEP_THRESHOLD: usize = 1 << 24;
#[cfg(not(target_arch = "x86_64"))]
const SIX_STEP_THRESHOLD: usize = 64 * 1024;

/// CPU SIMD dispatch policy for FFT plans.
///
/// `Auto` selects the best runtime implementation available on the target.
/// `Scalar` is a deterministic portability/reference mode. `Avx2` requests
/// the x86 AVX2/FMA path and falls back to scalar when those features are not
/// available, which makes it safe to use in cross-target evidence harnesses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuFftDispatch {
    Auto,
    Scalar,
    Avx2,
}

impl CpuFftDispatch {
    /// Resolve the policy to the implementation that this target can execute.
    pub fn effective(self) -> Self {
        match self {
            Self::Scalar => Self::Scalar,
            Self::Auto => {
                #[cfg(target_arch = "x86_64")]
                {
                    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
                    {
                        Self::Avx2
                    } else {
                        Self::Scalar
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    Self::Auto
                }
            }
            Self::Avx2 => {
                #[cfg(target_arch = "x86_64")]
                {
                    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
                    {
                        Self::Avx2
                    } else {
                        Self::Scalar
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    Self::Scalar
                }
            }
        }
    }

    pub(crate) fn allows_simd(self) -> bool {
        match self {
            Self::Scalar => false,
            Self::Auto => {
                #[cfg(target_arch = "aarch64")]
                {
                    true
                }
                #[cfg(target_arch = "x86_64")]
                {
                    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
                }
                #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
                {
                    false
                }
            }
            Self::Avx2 => self.effective() == Self::Avx2,
        }
    }

    /// Stable label suitable for machine-readable evidence artifacts.
    pub fn name(self) -> &'static str {
        match self.effective() {
            Self::Auto => "neon",
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
        }
    }
}

/// A complex FFT plan together with its Hermitian Euclidean adjoint.
///
/// Native complex FFTs are unnormalised. Consequently, the adjoint of a
/// forward plan is an unnormalised inverse plan, and conversely. This is not
/// the same as a normalised inverse: callers doing a round trip still divide
/// the result by [`Self::len`].
///
/// The pair shares the planner cache. [`Self::scratch_len`] is sufficient for
/// either operation, so a caller can keep one scratch arena for forward and
/// reverse-mode execution.
#[derive(Clone)]
pub struct FftPlanWithAdjoint<T: FftScalar> {
    plan: Arc<dyn FftPlan<T>>,
    adjoint: Arc<dyn FftPlan<T>>,
}

impl<T: FftScalar> FftPlanWithAdjoint<T> {
    /// Transform length in complex samples.
    pub fn len(&self) -> usize {
        self.plan.len()
    }

    /// Whether this plan has no transform samples.
    ///
    /// FFT planners construct positive-length plans, so this is normally
    /// `false`; keep the query consistent with [`Self::len`] for any custom
    /// plan implementation.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Direction of the primary transform.
    pub fn direction(&self) -> FftDirection {
        self.plan.direction()
    }

    /// Largest scratch requirement of the primary transform and its adjoint.
    #[must_use]
    pub fn scratch_len(&self) -> usize {
        self.plan.scratch_len().max(self.adjoint.scratch_len())
    }

    /// Apply the primary unnormalised transform.
    pub fn process(&self, buffer: &mut [num_complex::Complex<T>]) {
        self.plan.process(buffer);
    }

    /// Checked primary transform using the native plan contract.
    pub fn try_process(&self, buffer: &mut [num_complex::Complex<T>]) -> Result<(), FftPlanError> {
        self.plan.try_process(buffer)
    }

    /// Apply the Hermitian Euclidean adjoint of [`Self::process`].
    pub fn process_adjoint(&self, buffer: &mut [num_complex::Complex<T>]) {
        self.adjoint.process(buffer);
    }

    /// Checked Hermitian-adjoint transform using the native plan contract.
    pub fn try_process_adjoint(
        &self,
        buffer: &mut [num_complex::Complex<T>],
    ) -> Result<(), FftPlanError> {
        self.adjoint.try_process(buffer)
    }

    /// Apply the primary transform using caller-owned scratch storage.
    pub fn process_with_scratch(
        &self,
        buffer: &mut [num_complex::Complex<T>],
        scratch: &mut [num_complex::Complex<T>],
    ) {
        assert!(
            scratch.len() >= self.scratch_len(),
            "FFT adjoint-pair scratch is too short"
        );
        self.plan.process_with_scratch(buffer, scratch);
    }

    /// Checked primary transform with caller-owned scratch.
    pub fn try_process_with_scratch(
        &self,
        buffer: &mut [num_complex::Complex<T>],
        scratch: &mut [num_complex::Complex<T>],
    ) -> Result<(), FftPlanError> {
        if scratch.len() < self.scratch_len() {
            return Err(FftPlanError::ScratchTooShort {
                required: self.scratch_len(),
                actual: scratch.len(),
            });
        }
        self.plan.try_process_with_scratch(buffer, scratch)
    }

    /// Apply the Hermitian adjoint using caller-owned scratch storage.
    pub fn process_adjoint_with_scratch(
        &self,
        buffer: &mut [num_complex::Complex<T>],
        scratch: &mut [num_complex::Complex<T>],
    ) {
        assert!(
            scratch.len() >= self.scratch_len(),
            "FFT adjoint-pair scratch is too short"
        );
        self.adjoint.process_with_scratch(buffer, scratch);
    }

    /// Checked Hermitian-adjoint transform with caller-owned scratch.
    pub fn try_process_adjoint_with_scratch(
        &self,
        buffer: &mut [num_complex::Complex<T>],
        scratch: &mut [num_complex::Complex<T>],
    ) -> Result<(), FftPlanError> {
        if scratch.len() < self.scratch_len() {
            return Err(FftPlanError::ScratchTooShort {
                required: self.scratch_len(),
                actual: scratch.len(),
            });
        }
        self.adjoint.try_process_with_scratch(buffer, scratch)
    }
}

pub struct CpuFftPlanner<T: FftScalar> {
    cache: PlanCache<T>,
    dispatch: CpuFftDispatch,
}

impl<T: FftScalar> Default for CpuFftPlanner<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FftScalar> CpuFftPlanner<T> {
    pub fn new() -> Self {
        Self::with_dispatch(CpuFftDispatch::Auto)
    }

    /// Construct a planner with an explicit SIMD dispatch policy.
    pub fn with_dispatch(dispatch: CpuFftDispatch) -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            dispatch,
        }
    }

    /// Return the effective implementation selected for this planner.
    pub fn dispatch(&self) -> CpuFftDispatch {
        self.dispatch.effective()
    }

    /// Get or build a plan of length `n` in the given direction.
    ///
    /// This is a panicking convenience for trusted, statically known sizes.
    /// Prefer [`Self::try_plan`] for externally supplied or fallible sizes:
    /// it reports zero lengths and Bluestein-convolution overflow as
    /// [`PlanLengthError`] instead of panicking.
    pub fn plan(&self, n: usize, direction: FftDirection) -> Arc<dyn FftPlan<T>> {
        self.try_plan(n, direction)
            .unwrap_or_else(|error| panic!("cannot build FFT plan: {error}"))
    }

    /// Checked plan construction for untrusted or externally supplied sizes.
    ///
    /// The compatibility [`Self::plan`] method retains its historical panic
    /// behavior, while this entry point rejects zero lengths and arithmetic
    /// overflow before touching the cache or allocating a plan.
    pub fn try_plan(
        &self,
        n: usize,
        direction: FftDirection,
    ) -> Result<Arc<dyn FftPlan<T>>, PlanLengthError> {
        try_plan_length(n)?;
        if let Some(p) = self
            .cache
            .read()
            .expect("planner cache poisoned")
            .get(&(n, direction))
        {
            return Ok(p.clone());
        }

        // Build outside the lock so first-time requests for unrelated shapes
        // can proceed concurrently. A racing duplicate build is harmless;
        // the write-side lookup below canonicalises the returned Arc.
        let plan = if n > SIX_STEP_THRESHOLD {
            Arc::new(SixStepFftPlan::with_planner(n, direction, self)) as Arc<dyn FftPlan<T>>
        } else {
            self.plan_baseline(n, direction)
        };
        let mut guard = self.cache.write().expect("planner cache poisoned");
        if let Some(existing) = guard.get(&(n, direction)) {
            return Ok(existing.clone());
        }
        guard.insert((n, direction), plan.clone());
        Ok(plan)
    }

    /// Get or build a complex FFT plan and its Hermitian Euclidean adjoint.
    ///
    /// The returned adjoint is the opposite-direction *unnormalised* plan.
    /// It therefore obeys `<F(x), y> = <x, Fᴴ(y)>` under the complex Euclidean
    /// inner product, whereas round-trip reconstruction requires division by
    /// `n` after the opposite-direction transform.
    pub fn plan_with_adjoint(&self, n: usize, direction: FftDirection) -> FftPlanWithAdjoint<T> {
        self.try_plan_with_adjoint(n, direction)
            .unwrap_or_else(|error| panic!("cannot build FFT plan: {error}"))
    }

    /// Checked construction of a complex FFT plan and its Hermitian adjoint.
    pub fn try_plan_with_adjoint(
        &self,
        n: usize,
        direction: FftDirection,
    ) -> Result<FftPlanWithAdjoint<T>, PlanLengthError> {
        Ok(FftPlanWithAdjoint {
            plan: self.try_plan(n, direction)?,
            adjoint: self.try_plan(n, direction.opposite())?,
        })
    }

    /// Build a non-six-step axis plan for composite decompositions. This is
    /// crate-visible so a six-step plan cannot recursively select itself for
    /// an inconvenient prime factor.
    pub(crate) fn plan_baseline(&self, n: usize, direction: FftDirection) -> Arc<dyn FftPlan<T>> {
        match plan_length(n) {
            Strategy::MixedRadix(radices) => Arc::new(StockhamFftPlan::<T>::new(
                n,
                radices,
                direction,
                self.dispatch,
            )),
            Strategy::Bluestein { .. } => {
                Arc::new(BluesteinFftPlan::<T>::new(n, direction, self.dispatch))
            }
        }
    }

    pub fn plan_forward(&self, n: usize) -> Arc<dyn FftPlan<T>> {
        self.plan(n, FftDirection::Forward)
    }

    pub fn plan_inverse(&self, n: usize) -> Arc<dyn FftPlan<T>> {
        self.plan(n, FftDirection::Inverse)
    }

    /// Checked forward-plan construction.
    pub fn try_plan_forward(&self, n: usize) -> Result<Arc<dyn FftPlan<T>>, PlanLengthError> {
        self.try_plan(n, FftDirection::Forward)
    }

    /// Checked inverse-plan construction.
    pub fn try_plan_inverse(&self, n: usize) -> Result<Arc<dyn FftPlan<T>>, PlanLengthError> {
        self.try_plan(n, FftDirection::Inverse)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    fn inner_product(left: &[Complex<f64>], right: &[Complex<f64>]) -> Complex<f64> {
        left.iter()
            .zip(right)
            .map(|(left, right)| left.conj() * right)
            .sum()
    }

    #[test]
    fn checked_planner_rejects_zero_length() {
        let planner = CpuFftPlanner::<f64>::new();
        assert!(matches!(
            planner.try_plan(0, FftDirection::Forward),
            Err(PlanLengthError::ZeroLength)
        ));
    }

    #[test]
    fn planner_matches_independent_dft_and_parseval() {
        for n in [3, 6, 10, 11, 13] {
            let input: Vec<_> = (0..n)
                .map(|index| {
                    Complex::new(
                        (0.37 * index as f64).sin() + 0.2,
                        (0.19 * index as f64).cos() - 0.1,
                    )
                })
                .collect();
            let expected: Vec<_> = (0..n)
                .map(|k| {
                    (0..n).fold(Complex::new(0.0, 0.0), |sum, j| {
                        let angle = -std::f64::consts::TAU * (j * k) as f64 / n as f64;
                        sum + input[j] * Complex::from_polar(1.0, angle)
                    })
                })
                .collect();

            let planner = CpuFftPlanner::<f64>::new();
            let forward = planner.plan_forward(n);
            let mut spectrum = input.clone();
            forward.process(&mut spectrum);
            let scale = expected
                .iter()
                .chain(&spectrum)
                .map(|value| value.norm())
                .fold(1.0_f64, f64::max);
            for (actual, reference) in spectrum.iter().zip(expected) {
                assert!((actual - reference).norm() <= 5e-12 * scale, "n={n}");
            }

            let time_energy: f64 = input.iter().map(Complex::norm_sqr).sum();
            let frequency_energy: f64 =
                spectrum.iter().map(Complex::norm_sqr).sum::<f64>() / n as f64;
            assert!((time_energy - frequency_energy).abs() <= 5e-12 * time_energy.max(1.0));

            let inverse = planner.plan_inverse(n);
            inverse.process(&mut spectrum);
            for (actual, reference) in spectrum.iter().zip(input) {
                assert!(
                    (actual / n as f64 - reference).norm() <= 5e-12 * scale,
                    "n={n}"
                );
            }
        }
    }

    #[test]
    fn paired_plan_applies_the_complex_hermitian_adjoint() {
        let planner = CpuFftPlanner::<f64>::new();
        for n in [15, 60, 97] {
            let x: Vec<_> = (0..n)
                .map(|index| {
                    Complex::new(
                        (0.37 * index as f64).sin() + 0.2,
                        (0.19 * index as f64).cos() - 0.1,
                    )
                })
                .collect();
            let y: Vec<_> = (0..n)
                .map(|index| {
                    Complex::new(
                        (0.11 * index as f64).cos() - 0.3,
                        (0.41 * index as f64).sin() + 0.1,
                    )
                })
                .collect();
            for direction in [FftDirection::Forward, FftDirection::Inverse] {
                let pair = planner.try_plan_with_adjoint(n, direction).unwrap();
                assert_eq!(pair.len(), n);
                assert!(!pair.is_empty());
                assert_eq!(pair.len(), n);
                assert_eq!(pair.direction(), direction);

                let mut transformed_x = x.clone();
                let mut adjoint_y = y.clone();
                let mut scratch = vec![Complex::new(0.0, 0.0); pair.scratch_len()];
                pair.process_with_scratch(&mut transformed_x, &mut scratch);
                pair.process_adjoint_with_scratch(&mut adjoint_y, &mut scratch);

                let residual =
                    (inner_product(&transformed_x, &y) - inner_product(&x, &adjoint_y)).norm();
                assert!(
                    residual < n as f64 * 2e-10,
                    "n={n}, direction={direction:?}"
                );
            }
        }
    }
}
