//! Spin-then-park broadcast pool for microsecond-scale kernels.
//!
//! Rayon broadcast costs ~5µs at 16 workers on the x86 box (measured
//! empty-body medians; p90 ~13µs), which dominates BLAS-1 calls on
//! L3-resident vectors. This pool keeps one pinned worker per spread CPU
//! spinning on a generation counter, so dispatch is a single atomic bump
//! (~0.1-0.3µs) in the steady state, with parking past a spin window so
//! idle phases burn no CPU. Workers rendezvous through per-worker
//! generation slots; the submitter holds a mutex for the whole call, so
//! concurrent broadcasts serialize instead of racing the job slot.
//!
//! Used as the primary multi-thread path for BLAS-1 (`axpy`/`dot`/`scal`)
//! with the rayon spread broadcast kept as fallback. Top-level calls only:
//! broadcasting from inside a pool worker would deadlock, so call sites
//! must check [`SpinPool::on_worker`] first (mirroring the existing rayon
//! `current_thread_index` guard).

// Rust guideline compliant 2026-02-21

use std::cell::UnsafeCell;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Spin iterations before a worker parks (`hint::spin_loop` is ~20-50ns,
/// so this is a ~0.2-1ms burst window: Krylov-style BLAS-1 bursts never
/// park, while inter-solve gaps do). Lower values reintroduce wake
/// latency into bursts; higher values burn idle CPU longer.
const SPIN_ITERS: u32 = 20_000;

/// One worker's completion stamp, cache-line isolated so the submitter's
/// completion scan never bounces lines between workers.
#[repr(align(64))]
struct Slot {
    stamp: AtomicU64,
}

/// Type-erased job: the submitter's closure behind a raw pointer plus a
/// monomorphized adapter that restores the concrete type.
///
/// Only ever written under the submit mutex and read by workers after
/// observing the matching generation, so no data race is possible.
struct JobDesc {
    data: *const (),
    call: unsafe fn(*const (), usize),
}

// SAFETY: `data`/`call` cross threads only through the slot protocol;
// the pointed-to closure is `Sync` by the `broadcast` bound.
unsafe impl Send for JobDesc {}

/// Restore a type-erased broadcast closure and invoke it.
///
/// # Safety
///
/// `data` must be the `&F` the submitter stored for the current
/// generation, which the slot protocol keeps alive and unmodified until
/// every worker reports done.
unsafe fn call_adapter<F: Fn(usize)>(data: *const (), index: usize) {
    unsafe { (*data.cast::<F>())(index) }
}

thread_local! {
    /// Set while running as a [`SpinPool`] worker; used for the nesting
    /// guard ([`SpinPool::on_worker`]).
    static ON_WORKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Pool pointer for worker thread entry (same shape as the BLAS-1
/// `SendConstPtr` wrappers).
#[derive(Clone, Copy)]
struct SendPool(*const SpinPool);
// SAFETY: the pool is heap-stable and outlives every worker, which are
// joined in `Drop` and in the spawn-failure path before anyone can
// deallocate it.
unsafe impl Send for SendPool {}

/// Fixed-width pool broadcasting one closure to every worker.
///
/// Workers pin like the rayon spread pool (one per L3 group) and spin on
/// a generation counter, parking past [`SPIN_ITERS`] iterations. A
/// broadcast publishes one job descriptor, bumps the generation, unparks
/// only the workers flagged parked (zero syscalls in the steady state),
/// then spins until every per-worker slot reports the new generation.
///
/// # Panics
///
/// A panicking worker closure is caught, joined like a normal completion,
/// and resumed on the broadcasting thread once every worker finished, so
/// a broadcast either runs the closure everywhere or panics with the
/// worker's payload. Broadcasting from a pool worker would deadlock and
/// is a programming error (debug assertion).
pub(crate) struct SpinPool {
    /// Unpark handles, one per worker (index-addressed like `slots`).
    threads: Vec<std::thread::Thread>,
    /// Join handles, taken by `Drop` (mutex: `Drop` already holds the
    /// submit guard while joining, so this cannot borrow `&mut self`).
    handles: Mutex<Option<Vec<std::thread::JoinHandle<()>>>>,
    width: usize,
    /// Current generation; workers run the job slot when it changes.
    generation: AtomicU64,
    /// Set by `Drop`; workers exit instead of running the next job.
    shutdown: AtomicBool,
    /// Serializes submitters; held from job publish to all-slots-done so
    /// the single job slot is never raced and stays stable throughout.
    submit: Mutex<()>,
    /// Per-worker parked flags (see the pairing argument in `broadcast`).
    parked: Vec<AtomicBool>,
    /// Per-worker completion stamps.
    slots: Vec<Slot>,
    /// Current job, valid only between publish and all-slots-done.
    job: UnsafeCell<JobDesc>,
    /// A worker's panic payload, resumed on the submitter after joining.
    panic_slot: Mutex<Option<Box<dyn std::any::Any + Send>>>,
}

// SAFETY: `threads`/`handles`/`width` are immutable after construction;
// atomics and mutexes are `Sync` by construction; `job` is guarded by
// the submit mutex plus the generation/slot protocol (workers deref it
// only for the generation the submitter published, and the submitter
// holds the mutex until every slot reports that generation).
unsafe impl Sync for SpinPool {}
unsafe impl Send for SpinPool {}

impl SpinPool {
    /// Spawn `width` workers pinned like spread-pool workers.
    ///
    /// Returns `None` when any worker thread fails to spawn, in which
    /// case the caller falls back to the rayon broadcast path. Boxed so
    /// the worker threads' raw pointer stays valid; the pool joins every
    /// worker on drop.
    pub(crate) fn new(width: usize) -> Option<Box<Self>> {
        let width = width.max(1);
        let mut pool = Box::new(Self {
            threads: Vec::with_capacity(width),
            handles: Mutex::new(Some(Vec::with_capacity(width))),
            width,
            generation: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
            submit: Mutex::new(()),
            parked: (0..width).map(|_| AtomicBool::new(false)).collect(),
            slots: (0..width)
                .map(|_| Slot {
                    stamp: AtomicU64::new(0),
                })
                .collect(),
            job: UnsafeCell::new(JobDesc {
                data: std::ptr::null(),
                call: call_adapter::<fn(usize)>,
            }),
            panic_slot: Mutex::new(None),
        });
        // SAFETY: heap-stable for the pool's whole life; workers only
        // deref it for live generations, and `Drop` joins every worker
        // before deallocation.
        let raw = SendPool(&*pool as *const Self);
        for index in 0..width {
            let handle = std::thread::Builder::new()
                .name(format!("sonium-spin-{index}"))
                // SAFETY: `raw` outlives every worker (joined in `Drop`
                // and in the spawn-failure path below before returning).
                .spawn(move || {
                    let raw: SendPool = raw;
                    unsafe { worker_loop(&*raw.0, index) }
                });
            let Ok(handle) = handle else {
                pool.shutdown_and_join();
                return None;
            };
            pool.threads.push(handle.thread().clone());
            if let Some(handles) = pool.handles.lock().unwrap().as_mut() {
                handles.push(handle);
            }
        }
        Some(pool)
    }

    /// Pool width (worker count).
    pub(crate) fn width(&self) -> usize {
        self.width
    }

    /// Whether the calling thread is a pool worker (any pool).
    ///
    /// Call sites check this before broadcasting: a broadcast from a
    /// worker would wait on itself forever.
    pub(crate) fn on_worker() -> bool {
        ON_WORKER.get()
    }

    /// Run `f(worker_index)` once on every worker, returning when all
    /// finished. The closure must be `Sync`: every worker invokes it
    /// concurrently through a shared reference.
    ///
    /// # Panics
    ///
    /// Resumes a worker closure's panic on the caller after joining every
    /// worker. Debug-asserts against nested use (broadcasting from a pool
    /// worker would deadlock).
    pub(crate) fn broadcast<F: Fn(usize) + Sync>(&self, f: &F) {
        debug_assert!(!Self::on_worker(), "nested SpinPool broadcast");
        // Poison only arises from a propagated worker panic (resumed
        // below while the guard is held); the pool state stays consistent,
        // so a poisoned mutex is re-entered rather than fatal.
        let _guard = self
            .submit
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert!(
            !self.shutdown.load(Ordering::SeqCst),
            "broadcast on a shut-down SpinPool"
        );
        // SAFETY: the submit mutex excludes concurrent publishers, and it
        // is held until every slot reports this generation, so the job
        // slot is stable from publish to join. `f` is borrowed from the
        // caller and outlives the join.
        unsafe {
            *self.job.get() = JobDesc {
                data: f as *const F as *const (),
                call: call_adapter::<F>,
            };
        }
        let current = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        // Wake only flagged workers (none in the bursting steady state).
        // Pairing argument (all `SeqCst`): a worker sets its flag and then
        // re-reads the generation before parking. If its flag-set precedes
        // this scan, it is unparked (the permit is sticky, so an
        // unpark-before-park still releases it). Otherwise this bump
        // precedes its re-read, so it observes the new generation and
        // never parks. No wakeup is lost either way.
        for (thread, flag) in self.threads.iter().zip(self.parked.iter()) {
            if flag.load(Ordering::SeqCst) {
                thread.unpark();
            }
        }
        for slot in &self.slots {
            while slot.stamp.load(Ordering::Acquire) != current {
                std::hint::spin_loop();
            }
        }
        if let Some(payload) = self
            .panic_slot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
        {
            resume_unwind(payload);
        }
    }

    /// Run `f(worker, t0, t1)` over `n` elements split into at most `k`
    /// contiguous bands owned by workers `0..k` (empty when `n == 0`).
    /// `k` clamps into `[1, width]`; workers at or past `k` return
    /// without invoking `f` but stay in the generation protocol, so the
    /// steady-state cost is unchanged. Streaming callers pass the
    /// bandwidth-calibrated width (see `bandwidth::stream_width`) so
    /// saturated kernels stop waking cores past the knee; compute-bound
    /// callers keep the full width.
    pub(crate) fn broadcast_bands_k<F: Fn(usize, usize, usize) + Sync>(
        &self,
        n: usize,
        k: usize,
        f: &F,
    ) {
        let k = k.clamp(1, self.width);
        let bands = k.min(n).max(1);
        let els_per_band = n.div_ceil(bands);
        self.broadcast(&|w: usize| {
            if w >= k {
                return;
            }
            let t0 = w.saturating_mul(els_per_band).min(n);
            let t1 = t0.saturating_add(els_per_band).min(n);
            if t0 < t1 {
                f(w, t0, t1);
            }
        });
    }

    /// Signal exit, release every worker, and join all join handles.
    fn shutdown_and_join(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        self.generation.fetch_add(1, Ordering::SeqCst);
        for thread in &self.threads {
            thread.unpark();
        }
        if let Some(handles) = self.handles.lock().unwrap().take() {
            for handle in handles {
                let _ = handle.join();
            }
        }
    }
}

impl Drop for SpinPool {
    fn drop(&mut self) {
        // Serialize with an in-flight broadcast (which holds `submit`
        // until every worker finished) before signalling shutdown, so no
        // worker can observe shutdown mid-job.
        let _guard = self
            .submit
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.shutdown_and_join();
    }
}

/// Worker main loop: run the job slot whenever the generation advances,
/// else spin briefly and then park until unparked or bumped.
fn worker_loop(pool: &SpinPool, index: usize) {
    sonium_topology::affinity::pin_spread_worker(index);
    ON_WORKER.set(true);
    let mut last = 0u64;
    let mut spins = 0u32;
    loop {
        let current = pool.generation.load(Ordering::SeqCst);
        if current != last {
            if pool.shutdown.load(Ordering::SeqCst) {
                break;
            }
            // SAFETY: a changed generation means the submitter published
            // the job slot before bumping (same-thread program order into
            // `SeqCst`) and holds the submit mutex (hence the slot) until
            // this worker's slot reports `gen` below.
            let outcome = catch_unwind(AssertUnwindSafe(|| unsafe {
                let job = &*pool.job.get();
                (job.call)(job.data, index);
            }));
            if let Err(payload) = outcome {
                *pool
                    .panic_slot
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(payload);
            }
            pool.slots[index].stamp.store(current, Ordering::SeqCst);
            last = current;
            spins = 0;
            continue;
        }
        spins += 1;
        if spins < SPIN_ITERS {
            std::hint::spin_loop();
            continue;
        }
        pool.parked[index].store(true, Ordering::SeqCst);
        if pool.generation.load(Ordering::SeqCst) != last {
            pool.parked[index].store(false, Ordering::SeqCst);
            spins = 0;
            continue;
        }
        std::thread::park();
        pool.parked[index].store(false, Ordering::SeqCst);
        spins = 0;
    }
}

/// Process-wide pool sized like the rayon spread pool, if one could be
/// built. `None` (thread spawn failure) sends callers down the existing
/// rayon broadcast path.
pub(crate) fn spin_pool() -> Option<&'static SpinPool> {
    static POOL: std::sync::LazyLock<Option<Box<SpinPool>>> = std::sync::LazyLock::new(|| {
        let width = sonium_topology::affinity::spread_cpus().map(|cpus| cpus.len())?;
        SpinPool::new(width)
    });
    POOL.as_deref()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    #[test]
    fn bands_cover_exactly_once() {
        let pool = SpinPool::new(8).expect("spawn 8 workers");
        let n = 1000usize;
        let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
        pool.broadcast_bands_k(n, 8, &|_, t0, t1| {
            for i in t0..t1 {
                hits[i].fetch_add(1, AtomicOrdering::Relaxed);
            }
        });
        for (i, hit) in hits.iter().enumerate() {
            assert_eq!(hit.load(AtomicOrdering::Relaxed), 1, "index {i}");
        }
    }

    #[test]
    fn broadcast_runs_every_worker_once() {
        let pool = SpinPool::new(4).expect("spawn 4 workers");
        let seen = AtomicUsize::new(0);
        pool.broadcast(&|w: usize| {
            seen.fetch_or(1 << w, AtomicOrdering::Relaxed);
        });
        assert_eq!(seen.load(AtomicOrdering::Relaxed), 0b1111);
    }

    #[test]
    fn empty_bands_are_noop() {
        let pool = SpinPool::new(4).expect("spawn 4 workers");
        let calls = AtomicUsize::new(0);
        pool.broadcast_bands_k(0, 4, &|_, _, _| {
            calls.fetch_add(1, AtomicOrdering::Relaxed);
        });
        assert_eq!(calls.load(AtomicOrdering::Relaxed), 0);
    }

    #[test]
    fn bands_k_covers_exactly_once_on_first_k_workers() {
        // Subset bands (streaming width): every index runs exactly once,
        // only workers below `k` invoke the closure, and `k` past the
        // width clamps to the full pool.
        let pool = SpinPool::new(8).expect("spawn 8 workers");
        let n = 1000usize;
        let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
        let seen = AtomicUsize::new(0);
        pool.broadcast_bands_k(n, 3, &|w: usize, t0: usize, t1: usize| {
            seen.fetch_or(1 << w, AtomicOrdering::Relaxed);
            for i in t0..t1 {
                hits[i].fetch_add(1, AtomicOrdering::Relaxed);
            }
        });
        for (i, hit) in hits.iter().enumerate() {
            assert_eq!(hit.load(AtomicOrdering::Relaxed), 1, "index {i}");
        }
        assert_eq!(seen.load(AtomicOrdering::Relaxed), 0b111);
        // Clamp: k above the width behaves like the full pool.
        let seen = AtomicUsize::new(0);
        pool.broadcast_bands_k(n, 100, &|w: usize, _, _| {
            seen.fetch_or(1 << w, AtomicOrdering::Relaxed);
        });
        assert_eq!(seen.load(AtomicOrdering::Relaxed), 0xff);
    }

    #[test]
    fn worker_panic_resumes_on_caller_and_pool_survives() {
        let pool = SpinPool::new(4).expect("spawn 4 workers");
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            pool.broadcast(&|w: usize| {
                if w == 0 {
                    panic!("intentional worker panic");
                }
            });
        }));
        assert!(result.is_err(), "caller must observe the panic");
        // The pool stays usable (poison-tolerant submit mutex).
        let seen = AtomicUsize::new(0);
        pool.broadcast(&|w: usize| {
            seen.fetch_or(1 << w, AtomicOrdering::Relaxed);
        });
        assert_eq!(seen.load(AtomicOrdering::Relaxed), 0b1111);
    }

    #[test]
    fn parked_workers_wake() {
        let pool = SpinPool::new(4).expect("spawn 4 workers");
        pool.broadcast(&|_: usize| {});
        // Spin window is sub-millisecond; 100ms guarantees parked workers.
        std::thread::sleep(std::time::Duration::from_millis(100));
        let seen = AtomicUsize::new(0);
        pool.broadcast(&|w: usize| {
            seen.fetch_or(1 << w, AtomicOrdering::Relaxed);
        });
        assert_eq!(seen.load(AtomicOrdering::Relaxed), 0b1111);
    }

    #[test]
    fn concurrent_broadcasts_serialize() {
        let pool = SpinPool::new(4).expect("spawn 4 workers");
        let pool_ref = &pool;
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(move || {
                    for _ in 0..25 {
                        let seen = AtomicUsize::new(0);
                        pool_ref.broadcast(&|w: usize| {
                            seen.fetch_or(1 << w, AtomicOrdering::Relaxed);
                        });
                        assert_eq!(seen.load(AtomicOrdering::Relaxed), 0b1111);
                    }
                });
            }
        });
    }

    #[test]
    fn repeated_create_drop_joins_cleanly() {
        for _ in 0..10 {
            let pool = SpinPool::new(4).expect("spawn 4 workers");
            pool.broadcast(&|_: usize| {});
        }
    }

    #[test]
    fn on_worker_detects_nesting() {
        let pool = SpinPool::new(2).expect("spawn 2 workers");
        assert!(!SpinPool::on_worker());
        let nested = AtomicUsize::new(0);
        pool.broadcast(&|_: usize| {
            if SpinPool::on_worker() {
                nested.fetch_add(1, AtomicOrdering::Relaxed);
            }
        });
        assert_eq!(nested.load(AtomicOrdering::Relaxed), 2);
        assert!(!SpinPool::on_worker());
    }
}
