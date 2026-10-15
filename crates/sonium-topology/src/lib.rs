//! Machine placement for parallel work: topology discovery, CPU-set
//! selection, thread pinning, and bandwidth-width calibration.
//!
//! Parallel kernels care about three machine facts no OS scheduler
//! exposes directly: which CPUs share a NUMA node / CCD / LLC domain,
//! which of them are physical cores rather than SMT siblings, and how
//! many streaming threads saturate DRAM. This crate answers all three
//! from portable sysfs topology handles (NUMA `node*` links, cache
//! `shared_cpu_list`, `topology/*`) with no vendor or CPUID dispatch,
//! so the same code packs correctly on Threadripper, Xeon, and EPYC
//! across generations.
//!
//! Consumers today: `sonium-compute` (rayon pools, spin pool, BLAS
//! widths) and `sonium-fft` (batch dispatch worker budget + pinning).
//! ML executors are expected to join them — placement queries are
//! therefore kept crate-public and consumer-agnostic, while
//! consumer-specific policy (pool shapes, gates) stays with consumers.
//!
//! Everything is best-effort and Linux-only: pin failures are ignored,
//! other platforms degrade to scheduler placement, and the inherited
//! affinity mask (containers/cgroups/taskset/numactl) is always
//! respected. Placement never changes numerics. Escape hatches:
//! `SONIUM_SET_AFFINITY=0` (scheduler placement), `SONIUM_TOPO_PACK=0`
//! (legacy numeric order), `SONIUM_STREAM_WIDTH=n`, `SONIUM_BW_CALIB=0`.

pub mod affinity;
pub mod bandwidth;

/// Serialize tests that mutate `SONIUM_*` environment keys, across every
/// crate in the process (a consumer's test binary links this crate's
/// copy, so all env-mutating tests in that binary exclude each other).
/// Switches are read per call on the calling thread only, so holding
/// this across set/call/restore is sound.
#[doc(hidden)]
pub fn env_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    ENV_LOCK.lock().expect("SONIUM env-test lock poisoned")
}
