//! Machine-adaptive rayon gates for CPU kernels.
//!
//! A fixed row count cannot be right on every machine: the point where
//! pool dispatch pays off moves with thread count and per-row cost. Every
//! parallel gate in the CPU backend therefore resolves through
//! [`use_compute_parallel`]:
//!
//! - default: `work_items >= items_per_thread * threads`, where `threads`
//!   is [`std::thread::available_parallelism`]. Single-threaded processes
//!   always run serially;
//! - `SONIUM_COMPUTE_PAR_THRESHOLD=<n>`: absolute work-item override for
//!   tuning, benches, and forced-path tests (units are the call site's
//!   work items — matrix rows for [`SPMV_ROWS_PER_THREAD`], output
//!   elements for [`SPMM_OUTPUTS_PER_THREAD`], BEM pairs for
//!   [`BEM_PAIRS_PER_THREAD`]);
//! - `SONIUM_SERIAL=1`: process-wide serial escape hatch for deterministic
//!   QA runs.
//!
//! All gated loops parallelize over *independent outputs* (one matrix
//! row, one BEM pair per worker, each accumulated serially within), so
//! the serial and parallel paths are bitwise identical — the gate never
//! affects numerics, only scheduling. Reduction loops (e.g. the
//! transpose-GEMV axpy fan-out in `blas2`, where workers would share one
//! accumulator) stay serial by design: sharding them would make the
//! summation order — and therefore results — machine-dependent.
//!
//! The per-thread grains are starting points, not calibrated values: tune
//! them with benches on the target machine.

/// Matrix rows per thread above which CSR SpMV parallelizes over rows.
/// Calibrated so a 4-thread machine reproduces the legacy 1024-row
/// crossover; wider machines scale the crossover up instead of
/// oversubscribing small problems.
pub(crate) const SPMV_ROWS_PER_THREAD: usize = 256;

/// Output elements per thread above which CSR SpMM parallelizes over
/// right-hand-side columns.
pub(crate) const SPMM_OUTPUTS_PER_THREAD: usize = 256;

/// BEM regular pairs per thread above which [`crate::bem_cpu::regular_pair_batch`]
/// bands the pair list. Each pair runs a 4-point Helmholtz quadrature
/// with transcendental kernel evaluations.
pub(crate) const BEM_PAIRS_PER_THREAD: usize = 512;

/// Machine-adaptive parallel gate. See the module docs.
pub(crate) fn use_compute_parallel(work_items: usize, items_per_thread: usize) -> bool {
    if serial_override() {
        return false;
    }
    if let Some(threshold) = env_threshold("SONIUM_COMPUTE_PAR_THRESHOLD") {
        return work_items >= threshold;
    }
    let threads = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    if threads < 2 {
        return false;
    }
    work_items >= items_per_thread.saturating_mul(threads).max(1)
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

/// Intra-crate lock serializing tests that mutate the gate env keys.
/// Process env is shared by all test threads; tests that only *read* the
/// keys need no lock (both paths are bitwise identical), but tests that
/// *write* them must hold this guard across the whole mutation window,
/// or a concurrent write flips the asserted path (bisect 2026-09-28:
/// `gemm_m_split`'s `M_SPLIT=1` window overlapped `par_pack`'s two arms
/// and flipped routing between them, breaking the bitwise assert).
/// This is the single crate-wide serialiser for ALL `SONIUM_*`-flipping
/// lib tests (GEMM/GEMV/TRSM/SYRK/SpMV); the former module-local locks
/// in `blas2::avx2`/`syrk` could not see each other and were folded in.
/// The lock itself lives in `sonium-topology` so env-mutating tests in
/// every crate of a test binary mutually exclude. Switches are read per
/// call on the calling thread only (rayon workers never read them), so
/// holding this across set/call/restore is sound.
#[cfg(test)]
pub(crate) fn env_test_lock() -> std::sync::MutexGuard<'static, ()> {
    sonium_topology::env_test_lock()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serial_hatch_forces_serial() {
        // This test asserts path *selection*, so it must exclude
        // concurrent writers of the same keys (see `env_test_lock`).
        let _guard = super::env_test_lock();
        // SAFETY: holding the intra-crate env lock; restored below.
        unsafe {
            let prev_threshold = std::env::var_os("SONIUM_COMPUTE_PAR_THRESHOLD");
            let prev_serial = std::env::var_os("SONIUM_SERIAL");
            std::env::remove_var("SONIUM_COMPUTE_PAR_THRESHOLD");
            std::env::remove_var("SONIUM_SERIAL");

            let threads = std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1);
            if threads >= 2 {
                assert!(!use_compute_parallel(1, SPMV_ROWS_PER_THREAD));
                assert!(use_compute_parallel(
                    threads * SPMV_ROWS_PER_THREAD,
                    SPMV_ROWS_PER_THREAD
                ));
                assert!(!use_compute_parallel(
                    threads * SPMV_ROWS_PER_THREAD - 1,
                    SPMV_ROWS_PER_THREAD
                ));
            }
            std::env::set_var("SONIUM_SERIAL", "1");
            assert!(!use_compute_parallel(usize::MAX, 1));
            std::env::remove_var("SONIUM_SERIAL");
            std::env::set_var("SONIUM_COMPUTE_PAR_THRESHOLD", "1");
            assert!(use_compute_parallel(1, usize::MAX));

            match prev_threshold {
                Some(value) => std::env::set_var("SONIUM_COMPUTE_PAR_THRESHOLD", value),
                None => std::env::remove_var("SONIUM_COMPUTE_PAR_THRESHOLD"),
            }
            match prev_serial {
                Some(value) => std::env::set_var("SONIUM_SERIAL", value),
                None => std::env::remove_var("SONIUM_SERIAL"),
            }
        }
    }
}
