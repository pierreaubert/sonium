//! Machine-adaptive rayon gates for assembly merges.
//!
//! A fixed work-item threshold cannot be right on every machine: the point
//! where pool dispatch pays off moves with thread count, clock speed, and
//! memory bandwidth. Every parallel gate in this crate therefore resolves
//! through [`ParallelPolicy`]:
//!
//! - default: `work_items >= items_per_thread * gate_threads`, where
//!   `gate_threads` is the effective rayon pool size capped at
//!   [`MAX_PARALLEL_GATE_THREADS`]. Single-threaded pools always run
//!   serially;
//! - `SONIUM_FEM_PAR_THRESHOLD=<n>`: absolute work-item override for tuning,
//!   benches, and forced-path tests (units are the call site's work items —
//!   triplets for [`MERGE_TRIPLETS_PER_THREAD`]);
//! - `SONIUM_SERIAL=1`: process-wide serial escape hatch for deterministic
//!   QA runs.
//!
//! The per-thread grain below is a starting point, not a calibrated value:
//! tune it with benches on the target machine.

use crate::assembly::TripletMatrix;
use rayon::prelude::*;

/// Triplets per thread above which
/// [`assemble_bilinear`](crate::assembly::generic::assemble_bilinear) (and the
/// per-element variant) merge per-element triplet lists on rayon workers
/// instead of pushing them serially. A push is a few stores, so the grain is
/// set higher than per-cell gates elsewhere in the workspace.
pub const MERGE_TRIPLETS_PER_THREAD: usize = 131_072;

/// Machine-adaptive parallelization policy.
#[derive(Debug, Clone, Copy)]
pub struct ParallelPolicy {
    threads: usize,
    force_serial: bool,
    threshold_override: Option<usize>,
}

impl ParallelPolicy {
    /// Resolve from the process environment (see module docs).
    pub fn from_env() -> Self {
        // Size to the effective rayon pool, not the machine: pool dispatch
        // pays off relative to the workers that will run it. Sizing to
        // machine parallelism instead prices mid-size meshes out of
        // threading on many-core machines even when the pool is small.
        // `RAYON_NUM_THREADS` is honored because the global pool sizes
        // itself from it at first use.
        Self {
            threads: rayon::current_num_threads(),
            force_serial: serial_override(),
            threshold_override: env_threshold("SONIUM_FEM_PAR_THRESHOLD"),
        }
    }

    /// Explicit policy without environment reads (tests, benches, callers
    /// that already resolved the machine topology).
    pub fn with_threads(threads: usize) -> Self {
        Self {
            threads: threads.max(1),
            force_serial: false,
            threshold_override: None,
        }
    }

    /// Deterministic serial policy for QA runs.
    pub fn serial() -> Self {
        Self {
            threads: 1,
            force_serial: true,
            threshold_override: None,
        }
    }

    /// True when `work_items` of `items_per_thread`-grained work merit rayon.
    pub fn decide(&self, work_items: usize, items_per_thread: usize) -> bool {
        if self.force_serial || self.threads < 2 {
            return false;
        }
        if let Some(threshold) = self.threshold_override {
            return work_items >= threshold;
        }
        let gate_threads = self.threads.min(MAX_PARALLEL_GATE_THREADS);
        work_items >= items_per_thread.saturating_mul(gate_threads).max(1)
    }

    /// Worker thread count this policy assumes.
    pub fn threads(&self) -> usize {
        self.threads
    }
}

/// Convenience gate: [`ParallelPolicy::from_env`] + `decide`.
pub fn use_parallel(work_items: usize, items_per_thread: usize) -> bool {
    ParallelPolicy::from_env().decide(work_items, items_per_thread)
}

/// Elements per thread above which element-tabulation loops (bilinear,
/// linear, region-routed, nonlinear) parallelize over elements. Per-element
/// tabulation plus quadrature is microsecond-scale work, so the grain keeps
/// pool dispatch amortized; small meshes stay serial.
pub const ELEMENTS_PER_THREAD: usize = 256;

/// Maximum thread count the [`ParallelPolicy::decide`] gate scales with.
///
/// An uncapped `items_per_thread * threads` gate prices mid-size meshes out
/// of threading on many-core machines: with 128 pool threads the assembly
/// gate reaches 32768 elements, so a 10374-tet mesh assembles serially even
/// though the work scales well (measured 22.5s serial vs 1.2s parallel for
/// one P2 BDF assembly on a 128-thread Threadripper). Sixteen grains of
/// microsecond-scale work still dwarf pool-dispatch overhead, so the cap
/// keeps the amortization intent while letting mid-size meshes use the full
/// pool through work stealing. Pools with 16 or fewer threads behave exactly
/// as before.
pub const MAX_PARALLEL_GATE_THREADS: usize = 16;

/// Pointwise values per thread above which dense pointwise combines
/// (prebaked coefficient application, Helmholtz shift assembly) parallelize.
/// Per-value work is a handful of FMAs, so the grain is set high.
pub const POINTWISE_VALUES_PER_THREAD: usize = 131_072;

/// Collect `f(i)` for `i in 0..n`, in index order, using rayon when
/// [`use_parallel`] selects parallel for `n` items of `items_per_thread`
/// grain and a serial loop otherwise. Both paths visit indices in order, so
/// results are identical; this is the auto-select-serial gate for every
/// element-tabulation and pointwise-combine loop in assembly.
pub fn collect_indexed<T, F>(n: usize, items_per_thread: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Send + Sync,
{
    if use_parallel(n, items_per_thread) {
        (0..n).into_par_iter().map(f).collect()
    } else {
        (0..n).map(f).collect()
    }
}

/// Collect `f(i)` for `i in 0..n` where `f` may skip indices, in index
/// order, with the same serial auto-select as [`collect_indexed`]. Used by
/// selected-cell (`*_where`) assembly.
pub fn filter_collect_indexed<T, F>(n: usize, items_per_thread: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> Option<T> + Send + Sync,
{
    if use_parallel(n, items_per_thread) {
        (0..n).into_par_iter().filter_map(f).collect()
    } else {
        (0..n).filter_map(f).collect()
    }
}

/// Merge per-element triplet lists into a global [`TripletMatrix`].
///
/// Large merges shard consecutive elements across rayon workers, each shard
/// concatenating its elements' lists into a thread-local buffer; shards are
/// then pushed in element order, so the push sequence — and therefore the
/// duplicate-summation order in [`TripletMatrix::to_csr`] — is bitwise
/// identical to the serial merge. Small merges stay serial: the gate is the
/// machine-adaptive [`ParallelPolicy`], never a fixed element count.
pub fn merge_element_triplets<T>(
    element_triplets: Vec<Vec<(usize, usize, T)>>,
    n_rows: usize,
    n_cols: usize,
    total_entries: usize,
) -> TripletMatrix<T>
where
    T: Copy + Default + std::ops::AddAssign + Send + Sync,
{
    let policy = ParallelPolicy::from_env();
    let n_elems = element_triplets.len();
    let n_shards = policy.threads().min(n_elems.max(1)).max(1);
    if n_shards < 2 || !policy.decide(total_entries, MERGE_TRIPLETS_PER_THREAD) {
        let mut global = TripletMatrix::with_capacity(n_rows, n_cols, total_entries);
        for triplets in element_triplets {
            for (row, col, value) in triplets {
                global.push(row, col, value);
            }
        }
        return global;
    }
    let chunk = n_elems.div_ceil(n_shards);
    let shards: Vec<Vec<(usize, usize, T)>> = element_triplets
        .par_chunks(chunk)
        .map(|elems| {
            let mut buf = Vec::with_capacity(elems.iter().map(Vec::len).sum());
            for triplets in elems {
                buf.extend_from_slice(triplets);
            }
            buf
        })
        .collect();
    let mut global = TripletMatrix::with_capacity(n_rows, n_cols, total_entries);
    for shard in shards {
        for (row, col, value) in shard {
            global.push(row, col, value);
        }
    }
    global
}

fn serial_override() -> bool {
    std::env::var("SONIUM_SERIAL")
        .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

fn env_threshold(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serial_policy_never_parallelizes() {
        assert!(!ParallelPolicy::serial().decide(usize::MAX, 1));
    }

    #[test]
    fn single_thread_never_parallelizes() {
        assert!(!ParallelPolicy::with_threads(1).decide(usize::MAX, 1));
    }

    #[test]
    fn decision_scales_with_threads() {
        let eight = ParallelPolicy::with_threads(8);
        assert!(eight.decide(8 * MERGE_TRIPLETS_PER_THREAD, MERGE_TRIPLETS_PER_THREAD));
        assert!(!eight.decide(8 * MERGE_TRIPLETS_PER_THREAD - 1, MERGE_TRIPLETS_PER_THREAD));
    }

    #[test]
    fn decision_caps_gate_threads_on_many_core_pools() {
        let wide = ParallelPolicy::with_threads(128);
        let gate = MAX_PARALLEL_GATE_THREADS * ELEMENTS_PER_THREAD;
        assert!(wide.decide(gate, ELEMENTS_PER_THREAD));
        assert!(!wide.decide(gate - 1, ELEMENTS_PER_THREAD));
        // Narrow pools keep the exact linear gate.
        let four = ParallelPolicy::with_threads(4);
        assert!(four.decide(4 * ELEMENTS_PER_THREAD, ELEMENTS_PER_THREAD));
        assert!(!four.decide(4 * ELEMENTS_PER_THREAD - 1, ELEMENTS_PER_THREAD));
    }
}
