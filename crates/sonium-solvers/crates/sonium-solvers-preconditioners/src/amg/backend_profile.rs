//! Optional synchronized wall-time attribution for backend multigrid cycles.

// Rust guideline compliant 2026-02-21
use super::{BackendAmgPreconditioner, ComplexField};
use sonium_compute::{Backend, BackendResult, DeviceScalar};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Nonoverlapping stages within one multigrid level visit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum BackendAmgPhase {
    /// Initial relaxation, including the zero-start optimization.
    PreSmooth,
    /// Matrix action and subtraction forming the fine residual.
    Residual,
    /// Fine-to-coarse residual transfer.
    Restriction,
    /// Coarse-to-fine transfer and correction accumulation.
    Prolongation,
    /// Relaxation after the coarse correction.
    PostSmooth,
    /// Direct coarse solution or configured coarse smoothing fallback.
    CoarseSolve,
}

const PHASES: [BackendAmgPhase; 6] = [
    BackendAmgPhase::PreSmooth,
    BackendAmgPhase::Residual,
    BackendAmgPhase::Restriction,
    BackendAmgPhase::Prolongation,
    BackendAmgPhase::PostSmooth,
    BackendAmgPhase::CoarseSolve,
];

/// Completed synchronized phase measurements for one hierarchy level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BackendAmgPhaseProfile {
    /// Zero-based hierarchy level, starting at the finest matrix.
    pub level: usize,
    /// Measured stage, excluding recursive visits to other levels.
    pub phase: BackendAmgPhase,
    /// Number of successful phase completions, including W/F revisits.
    pub calls: u64,
    /// Synchronized wall time, including host dispatch and completion overhead.
    pub elapsed_ms: f64,
}

#[derive(Default)]
pub(super) struct PhaseCounters {
    calls: AtomicU64,
    nanoseconds: AtomicU64,
}

impl<B, T> BackendAmgPreconditioner<B, T>
where
    B: Backend,
    T: ComplexField<Real = <T as DeviceScalar>::Real> + DeviceScalar,
{
    /// Opt into native multi-sweep Jacobi on small levels when available.
    ///
    /// Disabled by default. Preserves weights, sweep counts and hierarchy.
    /// Unsupported sizes/backends use the existing resident operations. Launch
    /// failures propagate. No buffers are allocated. Numeric refresh retains
    /// this choice; newly built or checkpoint-restored hierarchies start disabled.
    pub fn set_small_jacobi_enabled(&mut self, enabled: bool) {
        self.small_jacobi_enabled = enabled;
    }

    /// Return whether native small-level Jacobi is explicitly enabled.
    pub fn small_jacobi_enabled(&self) -> bool {
        self.small_jacobi_enabled
    }

    /// Opt into fused CSR Jacobi sweeps using retained scratch storage.
    ///
    /// Disabled by default. Keeps the hierarchy, damping and sweep counts fixed.
    /// Each sweep reads the old iterate and writes separate scratch, then swaps
    /// buffers. Unsupported backends retain existing operations. If small-level
    /// multi-sweep Jacobi is also enabled, it takes precedence where supported.
    /// Numeric refresh retains this choice; rebuilding starts disabled.
    pub fn set_fused_jacobi_enabled(&mut self, enabled: bool) {
        self.fused_jacobi_enabled = enabled;
    }

    /// Return whether fused CSR Jacobi sweeps are explicitly enabled.
    pub fn fused_jacobi_enabled(&self) -> bool {
        self.fused_jacobi_enabled
    }

    /// Enable synchronized per-level diagnostics, resetting all phase measurements.
    ///
    /// Disabled by default. Enabling adds completion barriers before and after
    /// each stage and perturbs execution; these are not CUDA event timings.
    /// Disabling releases the host counters and adds no synchronization.
    /// No device buffers are allocated. Rebuilt/checkpoint-restored hierarchies
    /// start disabled; numeric refresh retains the current profiling state.
    pub fn set_phase_profiling(&mut self, enabled: bool) {
        self.phase_profile = enabled.then(|| {
            (0..self.levels.len())
                .map(|_| std::array::from_fn(|_| PhaseCounters::default()))
                .collect()
        });
    }

    /// Snapshot successful phases without reading device data or synchronizing.
    ///
    /// Returns an empty list when disabled, otherwise six entries per level.
    /// Failed applications may have completed earlier phases; this snapshot
    /// is not a certification that the enclosing application succeeded.
    pub fn phase_profile(&self) -> Vec<BackendAmgPhaseProfile> {
        let Some(levels) = &self.phase_profile else {
            return Vec::new();
        };
        levels
            .iter()
            .enumerate()
            .flat_map(|(level, counters)| {
                PHASES
                    .into_iter()
                    .zip(counters)
                    .map(move |(phase, counter)| BackendAmgPhaseProfile {
                        level,
                        phase,
                        calls: counter.calls.load(Ordering::Relaxed),
                        elapsed_ms: counter.nanoseconds.load(Ordering::Relaxed) as f64 / 1e6,
                    })
            })
            .collect()
    }

    pub(super) fn profile_start(&self, backend: &B) -> BackendResult<Option<Instant>> {
        if self.phase_profile.is_none() {
            return Ok(None);
        }
        backend.try_synchronize()?;
        Ok(Some(Instant::now()))
    }

    pub(super) fn profile_finish(
        &self,
        backend: &B,
        level: usize,
        phase: BackendAmgPhase,
        start: Option<Instant>,
    ) -> BackendResult<()> {
        let Some(start) = start else { return Ok(()) };
        backend.try_synchronize()?;
        if let Some(levels) = &self.phase_profile {
            let counter = &levels[level][phase as usize];
            counter.nanoseconds.fetch_add(
                start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                Ordering::Relaxed,
            );
            counter.calls.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::AmgConfig;
    use num_complex::Complex64;
    use sonium_compute::{
        CudaBackend, DevicePreconditioner, MemoryPool, MemoryPoolConfig,
        PreconditionerFailureInjectionPoint,
    };
    use sonium_solvers_sparse::CsrMatrix;

    #[test]
    fn native_phase_profile_coarse_failure_is_not_a_completed_phase() {
        let backend = match CudaBackend::new(0) {
            Ok(backend) => backend,
            Err(error) => {
                assert!(
                    std::env::var("SONIUM_REQUIRE_CUDA").is_err(),
                    "CUDA required: {error}"
                );
                return;
            }
        };
        let pool = MemoryPool::new(backend, MemoryPoolConfig::default()).unwrap();
        let backend = pool.backend();
        let matrix = CsrMatrix::from_triplets(
            3,
            3,
            vec![
                (0, 0, Complex64::new(2.0, 0.1)),
                (1, 1, Complex64::new(3.0, 0.2)),
                (2, 2, Complex64::new(4.0, 0.3)),
            ],
        );
        let mut amg =
            BackendAmgPreconditioner::<CudaBackend, Complex64>::try_from_csr_with_cuda_coarse_lu(
                backend,
                &matrix,
                AmgConfig::default(),
            )
            .unwrap();
        assert_eq!(amg.num_levels(), 1);
        amg.set_phase_profiling(true);
        let mut rhs = pool.try_alloc(3).unwrap();
        let mut output = pool.try_alloc(3).unwrap();
        backend.try_copy_h2d(&matrix.values, rhs.raw_mut()).unwrap();
        backend.inject_preconditioner_failure(PreconditionerFailureInjectionPoint::CoarseSolve);
        assert!(amg.try_apply(backend, rhs.raw(), output.raw_mut()).is_err());
        assert!(amg.phase_profile().iter().all(|p| p.calls == 0));
        backend.recover_after_error().unwrap();
        amg.try_apply(backend, rhs.raw(), output.raw_mut()).unwrap();
        let profile = amg.phase_profile();
        assert_eq!(profile.iter().map(|p| p.calls).sum::<u64>(), 1);
        assert_eq!(
            profile
                .iter()
                .find(|p| p.phase == BackendAmgPhase::CoarseSolve)
                .unwrap()
                .calls,
            1
        );
        let mut actual = vec![Complex64::default(); 3];
        backend.try_copy_d2h(output.raw(), &mut actual).unwrap();
        assert!(
            actual
                .iter()
                .all(|x| (*x - Complex64::new(1.0, 0.0)).norm() < 1e-12)
        );
        amg.set_phase_profiling(false);
        assert!(amg.phase_profile().is_empty());
    }
}
