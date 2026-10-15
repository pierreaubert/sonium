//! Concrete CPU FFT plan implementations.
//!
//! Two flavours, picked by the planner based on length:
//!
//! - [`StockhamFftPlan`] for smooth lengths (factors of 2/3/4/5).
//! - [`BluesteinFftPlan`] for everything else.
//!
//! Both implement [`crate::FftPlan`] over `T: FftScalar` and handle
//! batched input where the buffer length is an integer multiple of the
//! single-FFT length.

use num_complex::Complex;
use std::cell::UnsafeCell;
use std::hint::spin_loop;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::FftDirection;
use crate::FftPlan;
use crate::bluestein::BluesteinPlan;
use crate::cpu::planner::CpuFftDispatch;
use crate::scalar::FftScalar;
use crate::stockham::{self, StageSection};
use crate::twiddles::{StageTwiddles, build_stage_twiddles};

/// Batch-parallelism gates for the owned-scratch `process` path.
///
/// Below either bound the serial loop wins: one FFT at these sizes is
/// microseconds, and spawning/synchronising threads per call costs more than
/// the transform itself. Above both bounds the batch is split across
/// `std::thread::scope` workers, each with owned scratch, so the shared
/// [`PlanScratch`] spinlock leaves the hot path and threads never contend.
/// The parallel path may allocate per-worker scratch; the serial path below
/// the thresholds stays allocation-free (see `tests/no_alloc_execution.rs`).
///
/// A fixed element count cannot be right on every machine: the point where
/// scoped-thread dispatch pays off moves with thread count. The gate is
/// therefore `total_elems >= PAR_ELEMS_PER_THREAD * threads` (with
/// `SONIUM_FFT_PAR_THRESHOLD=<n>` as an absolute total-element override
/// for tuning and forced-path tests, and `SONIUM_SERIAL=1` as a
/// process-wide serial escape hatch). The batch floor of 2 is structural —
/// parallelism needs at least two chunks to split — not a machine
/// threshold.
///
/// `PAR_ELEMS_PER_THREAD` is a starting point, not a calibrated value:
/// it reproduces the legacy 65_536-element crossover on an 8-thread
/// machine. Tune it with benches on the target machine.
const PAR_MIN_BATCH: usize = 2;
const PAR_ELEMS_PER_THREAD: usize = 8_192;

/// Process-wide snapshot of the batch-parallelism environment switches.
///
/// `std::env::var` allocates on every call, which would break the
/// allocation-free serial `process` contract asserted by
/// `tests/no_alloc_execution.rs` — and waste a lookup per FFT regardless.
/// The switches are therefore read once (first gate evaluation) and cached;
/// changing them mid-process has no effect, matching the workspace
/// `SONIUM_*` convention.
struct BatchParGate {
    force_serial: bool,
    threshold_override: Option<usize>,
    /// Worker budget: packed physical CPUs (one thread per physical
    /// core, SMT tail excluded) from `sonium-topology`; falls back to
    /// `available_parallelism` when topology is unavailable (affinity
    /// disabled, non-Linux). Workers pin to these CPUs in order, so a
    /// parallel batch packs NUMA/CCD/CCX-local like sonium-compute's
    /// pinned pools instead of spraying over every hardware thread.
    threads: usize,
}

static BATCH_PAR_GATE: std::sync::OnceLock<BatchParGate> = std::sync::OnceLock::new();

fn batch_par_gate() -> &'static BatchParGate {
    BATCH_PAR_GATE.get_or_init(|| BatchParGate {
        force_serial: std::env::var("SONIUM_SERIAL")
            .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false),
        threshold_override: std::env::var("SONIUM_FFT_PAR_THRESHOLD")
            .ok()
            .and_then(|value| value.parse::<usize>().ok()),
        threads: sonium_topology::affinity::physical_cpu_count()
            .or_else(|| {
                std::thread::available_parallelism()
                    .map(|count| count.get())
                    .ok()
            })
            .unwrap_or(1),
    })
}

/// Pure gate decision over explicit configuration. Kept `pub(crate)` and
/// allocation-free so unit tests can assert the selection hermetically
/// (independent of process environment and of [`BATCH_PAR_GATE`] init order).
pub(crate) fn parallel_workers_with(
    batch: usize,
    total_elems: usize,
    force_serial: bool,
    threshold_override: Option<usize>,
    threads: usize,
) -> Option<usize> {
    if force_serial {
        return None;
    }
    if let Some(threshold) = threshold_override {
        if total_elems < threshold || batch < PAR_MIN_BATCH {
            return None;
        }
    } else if threads < 2
        || batch < PAR_MIN_BATCH
        || total_elems < PAR_ELEMS_PER_THREAD.saturating_mul(threads).max(1)
    {
        return None;
    }
    let workers = threads.min(batch);
    (workers >= 2).then_some(workers)
}

/// Worker count for a batch of `batch` chunks totalling `total_elems`
/// elements, or `None` when the batch stays serial. Allocation-free: all
/// environment configuration is read once into [`BATCH_PAR_GATE`].
pub(crate) fn parallel_workers(batch: usize, total_elems: usize) -> Option<usize> {
    let gate = batch_par_gate();
    parallel_workers_with(
        batch,
        total_elems,
        gate.force_serial,
        gate.threshold_override,
        gate.threads,
    )
}

/// Evenly split `batch` chunks over `workers` threads; counts sum to `batch`.
pub(crate) fn split_batch(batch: usize, workers: usize) -> Vec<usize> {
    let base = batch / workers;
    let extra = batch % workers;
    (0..workers)
        .map(|worker| base + usize::from(worker < extra))
        .collect()
}

/// Mixed-radix Stockham plan for a smooth length.
pub struct StockhamFftPlan<T: FftScalar> {
    n: usize,
    direction: FftDirection,
    stages: Vec<StageTwiddles<T>>,
    sections: Vec<StageSection>,
    dir_sign: T,
    scratch: PlanScratch<T>,
    dispatch: CpuFftDispatch,
}

impl<T: FftScalar> StockhamFftPlan<T> {
    pub(crate) fn new(
        n: usize,
        radices: Vec<usize>,
        direction: FftDirection,
        dispatch: CpuFftDispatch,
    ) -> Self {
        // Read the batch-parallelism environment/topology once here: the
        // std env and available-parallelism queries allocate on first use,
        // and steady-state `process` must stay allocation-free.
        let _ = batch_par_gate();
        let stages = build_stage_twiddles::<T>(n.max(1), &radices, direction);
        let sections = stockham::plan_stage_sections::<T>(n.max(1), &stages);
        let dir_sign = match direction {
            FftDirection::Forward => T::from_f64(-1.0).unwrap(),
            FftDirection::Inverse => T::from_f64(1.0).unwrap(),
        };
        Self {
            n,
            direction,
            stages,
            sections,
            dir_sign,
            scratch: PlanScratch::new(vec![Complex::new(T::zero(), T::zero()); n]),
            dispatch,
        }
    }
}

impl<T: FftScalar> FftPlan<T> for StockhamFftPlan<T> {
    fn len(&self) -> usize {
        self.n
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn process(&self, buffer: &mut [Complex<T>]) {
        if self.n == 0 {
            return;
        }
        assert_eq!(
            buffer.len() % self.n,
            0,
            "buffer length must be a multiple of plan length"
        );
        let batch = buffer.len() / self.n;
        if let Some(workers) = parallel_workers(batch, buffer.len()) {
            // Large batch: one scoped worker per chunk group, each with
            // owned scratch. `run_batched_*` reuses a single scratch region
            // serially, so contiguous whole-chunk ranges stay correct when
            // split across workers.
            let counts = split_batch(batch, workers);
            let mut rest = buffer;
            std::thread::scope(|scope| {
                for (index, count) in counts.into_iter().enumerate() {
                    let (head, tail) = rest.split_at_mut(count * self.n);
                    rest = tail;
                    scope.spawn(move || {
                        sonium_topology::affinity::pin_physical_worker(index);
                        let mut local = vec![Complex::new(T::zero(), T::zero()); self.n];
                        stockham::run_batched_with_dispatch(
                            &self.stages,
                            self.dir_sign,
                            self.n,
                            head,
                            &mut local,
                            &self.sections,
                            self.dispatch,
                        );
                    });
                }
            });
            return;
        }
        let mut scratch = self.scratch.lock();
        stockham::run_batched_with_dispatch(
            &self.stages,
            self.dir_sign,
            self.n,
            buffer,
            &mut scratch,
            &self.sections,
            self.dispatch,
        );
    }

    fn scratch_len(&self) -> usize {
        self.n
    }

    fn cache_section_count(&self) -> usize {
        self.sections.len()
    }

    fn cache_section_budget_bytes(&self) -> usize {
        stockham::section_cache_budget_bytes()
    }

    fn process_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        if self.n == 0 {
            return;
        }
        assert!(scratch.len() >= self.n);
        assert_eq!(buffer.len() % self.n, 0);
        stockham::run_batched_with_dispatch(
            &self.stages,
            self.dir_sign,
            self.n,
            buffer,
            scratch,
            &self.sections,
            self.dispatch,
        );
    }
}

/// Bluestein chirp-Z plan for arbitrary length.
pub struct BluesteinFftPlan<T: FftScalar> {
    inner: BluesteinPlan<T>,
    scratch: PlanScratch<T>,
}

impl<T: FftScalar> BluesteinFftPlan<T> {
    pub(crate) fn new(n: usize, direction: FftDirection, dispatch: CpuFftDispatch) -> Self {
        // Same one-time gate initialization as the Stockham plan so the
        // first `process` call never pays the env/topology allocation.
        let _ = batch_par_gate();
        let inner = BluesteinPlan::<T>::new_with_dispatch(n, direction, dispatch);
        let scratch_len = inner.scratch_len();
        Self {
            inner,
            scratch: PlanScratch::new(vec![Complex::new(T::zero(), T::zero()); scratch_len]),
        }
    }
}

impl<T: FftScalar> FftPlan<T> for BluesteinFftPlan<T> {
    fn len(&self) -> usize {
        self.inner.n
    }

    fn direction(&self) -> FftDirection {
        self.inner.direction
    }

    fn process(&self, buffer: &mut [Complex<T>]) {
        let n = self.inner.n;
        if n == 0 {
            return;
        }
        assert_eq!(buffer.len() % n, 0);
        let batch = buffer.len() / n;
        if let Some(workers) = parallel_workers(batch, buffer.len()) {
            // Large prime/non-smooth batch: chunks are independent, so each
            // scoped worker drains a whole-chunk group with owned scratch.
            let scratch_len = self.inner.scratch_len();
            let counts = split_batch(batch, workers);
            let mut rest = buffer;
            std::thread::scope(|scope| {
                for (index, count) in counts.into_iter().enumerate() {
                    let (head, tail) = rest.split_at_mut(count * n);
                    rest = tail;
                    scope.spawn(move || {
                        sonium_topology::affinity::pin_physical_worker(index);
                        let mut local = vec![Complex::new(T::zero(), T::zero()); scratch_len];
                        for chunk in head.chunks_exact_mut(n) {
                            self.inner.process(chunk, &mut local);
                        }
                    });
                }
            });
            return;
        }
        let mut scratch = self.scratch.lock();
        for chunk in buffer.chunks_exact_mut(n) {
            self.inner.process(chunk, &mut scratch);
        }
    }

    fn scratch_len(&self) -> usize {
        self.inner.scratch_len()
    }

    fn process_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        let n = self.inner.n;
        if n == 0 {
            return;
        }
        assert!(scratch.len() >= self.inner.scratch_len());
        assert_eq!(buffer.len() % n, 0);
        for chunk in buffer.chunks_exact_mut(n) {
            self.inner.process(chunk, scratch);
        }
    }
}

struct PlanScratch<T: FftScalar> {
    locked: AtomicBool,
    buffer: UnsafeCell<Vec<Complex<T>>>,
}

unsafe impl<T: FftScalar> Sync for PlanScratch<T> {}

impl<T: FftScalar> PlanScratch<T> {
    fn new(buffer: Vec<Complex<T>>) -> Self {
        Self {
            locked: AtomicBool::new(false),
            buffer: UnsafeCell::new(buffer),
        }
    }

    fn lock(&self) -> PlanScratchGuard<'_, T> {
        while self
            .locked
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            spin_loop();
        }
        PlanScratchGuard { scratch: self }
    }
}

struct PlanScratchGuard<'a, T: FftScalar> {
    scratch: &'a PlanScratch<T>,
}

impl<T: FftScalar> Deref for PlanScratchGuard<'_, T> {
    type Target = Vec<Complex<T>>;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.scratch.buffer.get() }
    }
}

impl<T: FftScalar> DerefMut for PlanScratchGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.scratch.buffer.get() }
    }
}

impl<T: FftScalar> Drop for PlanScratchGuard<'_, T> {
    fn drop(&mut self) {
        self.scratch.locked.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_batch_partitions_every_chunk() {
        for (batch, workers) in [(8usize, 2), (64, 8), (65, 8), (9, 4), (8, 8)] {
            let counts = split_batch(batch, workers);
            assert_eq!(counts.len(), workers);
            assert_eq!(counts.iter().sum::<usize>(), batch);
            let (min, max) = (
                counts.iter().min().copied().unwrap(),
                counts.iter().max().copied().unwrap(),
            );
            // Even split: worker loads differ by at most one chunk.
            assert!(max - min <= 1);
            assert!(min >= 1);
        }
    }

    #[test]
    fn small_batches_stay_serial_and_large_batches_parallelise() {
        // Hermetic by construction: the pure gate takes explicit
        // configuration, so process env and OnceLock init order cannot leak
        // in (and no env keys are touched — race-free under parallel test
        // threads).
        for threads in [1usize, 2, 8] {
            // Structural floor: fewer than two chunks can never split, on
            // any machine, at any size.
            assert_eq!(
                parallel_workers_with(0, usize::MAX, false, None, threads),
                None
            );
            assert_eq!(
                parallel_workers_with(1, usize::MAX, false, None, threads),
                None
            );
            // Forced serial always wins.
            assert_eq!(
                parallel_workers_with(usize::MAX, usize::MAX, true, None, threads),
                None
            );
            // Below the machine-scaled element gate the allocation-free
            // serial path wins.
            let below = PAR_ELEMS_PER_THREAD
                .saturating_mul(threads.max(2))
                .saturating_sub(1);
            assert_eq!(parallel_workers_with(2, below, false, None, threads), None);
            // At the gate a multi-threaded machine parallelises; a
            // single-threaded machine serialises — both are correct.
            let at = PAR_ELEMS_PER_THREAD.saturating_mul(threads.max(2)).max(1);
            match parallel_workers_with(threads.max(2), at, false, None, threads) {
                None => assert_eq!(
                    threads, 1,
                    "serial above the gates requires a single-threaded machine"
                ),
                Some(workers) => assert!(workers >= 2),
            }
            // Explicit override: below-threshold serial, at-threshold split.
            assert_eq!(
                parallel_workers_with(4, 999, false, Some(1_000), threads),
                None
            );
            match parallel_workers_with(4, 1_000, false, Some(1_000), threads) {
                None => assert_eq!(
                    threads, 1,
                    "serial at an explicit threshold needs one thread"
                ),
                Some(workers) => assert!(workers >= 2),
            }
        }
    }
}
