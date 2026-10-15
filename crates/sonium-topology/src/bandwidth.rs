//! Streaming-width calibration: how many workers saturate DRAM?
//!
//! Bandwidth-bound kernels (copy, axpy, dot, streaming SpMV/GEMV) stop
//! scaling once the memory controllers saturate — past that knee extra
//! workers burn cores and frequency budget for nothing, while too few
//! leave bandwidth on the table. The knee moves with the machine (8 slow
//! DDR5 channels vs HBM vs 16 MRDIMM channels differ ~5x), and channel
//! counts are not enumerable from userspace, so the width is measured,
//! not detected: [`stream_width`] sweeps a pinned parallel copy over
//! thread counts once per process and keeps the smallest width within
//! 5% of the observed peak.
//!
//! Consumers: the BLAS-1 spin broadcasts (the highest-frequency
//! streaming traffic) band over the first [`stream_width`] spin workers.
//! The rayon spread path (GEMV) still uses the full spread set — routing
//! it through the calibrated width is follow-up work (it needs a new
//! pool width plus selection-rule updates).
//!
//! Cost: one ~10-100ms sweep on the first streaming call (a few hundred
//! MB of transient traffic — sized past total L3 so it measures DRAM,
//! not cache). Result cached process-wide. `SONIUM_STREAM_WIDTH=n`
//! pins the width exactly (diagnostic/A-B, read per call like the other
//! gates); `SONIUM_BW_CALIB=0` skips the sweep and keeps the full spread
//! width (legacy behavior).

use std::sync::LazyLock;
use std::time::Instant;

/// Fraction of peak bandwidth the calibrated width must reach: the
/// smallest swept width at or above it wins. 5% headroom absorbs run to
/// run noise (shared test runners, DVFS ramps) so the knee is stable;
/// tighter (98%+) keeps straggler cores past saturation, looser (90%-)
/// risks leaving real bandwidth unused on wide machines.
const KNEE_FRACTION: f64 = 0.95;

/// Timed copy iterations per swept width; the minimum wins. Two is
/// enough to shed first-iteration page-cache and frequency-ramp effects
/// while keeping the one-time sweep short.
const ITERS_PER_WIDTH: usize = 2;

/// Per-array sweep size bounds. Traffic per iteration is two arrays
/// (copy kernel: one read stream, one write stream), which must exceed
/// total L3 — otherwise the sweep times the cache, not DRAM. 64 MB
/// floors small-L3 machines; 256 MB caps the transient allocation
/// (512 MB traffic still doubles the largest current L3 totals).
const MIN_ARRAY_BYTES: usize = 64 << 20;
const MAX_ARRAY_BYTES: usize = 256 << 20;

/// Spread-set size, or 1 when topology is unavailable (spin paths never
/// run then — this only keeps the arithmetic total).
static SPREAD_LEN: LazyLock<usize> = LazyLock::new(|| {
    crate::affinity::spread_cpus()
        .map(|cpus| cpus.len())
        .unwrap_or(1)
        .max(1)
});

/// Calibrated streaming width, clamped into `[1, spread]`. Falls back
/// to the full spread width (legacy behavior) when the sweep cannot run.
static CALIBRATED: LazyLock<usize> =
    LazyLock::new(|| calibrate().unwrap_or(*SPREAD_LEN).clamp(1, *SPREAD_LEN));

/// Worker count for streaming broadcasts: the calibrated knee (see the
/// module docs), the `SONIUM_STREAM_WIDTH` override, or the full spread
/// width when calibration is disabled. Always within `[1, spread]`.
pub fn stream_width() -> usize {
    if let Some(n) = env_override() {
        return n.clamp(1, *SPREAD_LEN);
    }
    if calib_disabled() {
        return *SPREAD_LEN;
    }
    *CALIBRATED
}

/// Exact-width override; unparsable values fall through to calibration.
fn env_override() -> Option<usize> {
    std::env::var("SONIUM_STREAM_WIDTH")
        .ok()?
        .parse::<usize>()
        .ok()
}

fn calib_disabled() -> bool {
    std::env::var("SONIUM_BW_CALIB")
        .map(|v| v == "0")
        .unwrap_or(false)
}

/// Widths to sweep for a spread set of `len`: powers of two plus the
/// full width itself (covers non-power-of-two topologies like 12-slice
/// Granite Rapids). Always starts at 1 and ends at `len`.
fn sweep_widths(len: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut w = 1usize;
    while w < len {
        out.push(w);
        w *= 2;
    }
    out.push(len.max(1));
    out
}

/// Per-array sweep bytes: total L3 clamped into
/// [`MIN_ARRAY_BYTES`]`..=`[`MAX_ARRAY_BYTES`] (see the bounds' docs).
fn calib_array_bytes() -> usize {
    let total_l3 = crate::affinity::l3_total_bytes().unwrap_or(32 << 20);
    total_l3.clamp(MIN_ARRAY_BYTES, MAX_ARRAY_BYTES)
}

/// Smallest width reaching [`KNEE_FRACTION`] of the peak. `None` for
/// empty or all-zero input (callers fall back to the spread width).
fn pick_knee(widths: &[usize], gbps: &[f64]) -> Option<usize> {
    let best = gbps.iter().copied().fold(0.0f64, f64::max);
    if best <= 0.0 {
        return None;
    }
    widths
        .iter()
        .zip(gbps.iter())
        .find(|(_, g)| **g >= best * KNEE_FRACTION)
        .map(|(w, _)| *w)
}

/// Run the copy sweep over the spread set and pick the knee. `None`
/// when topology is unavailable or no timing is positive.
fn calibrate() -> Option<usize> {
    let cpus = crate::affinity::spread_cpus()?;
    if cpus.is_empty() {
        return None;
    }
    let widths = sweep_widths(cpus.len());
    let n = calib_array_bytes() / size_of::<u64>();
    if n == 0 {
        return None;
    }
    let mut src = vec![0u64; n];
    let mut dst = vec![0u64; n];
    // Parallel first-touch at full width: pages distribute across the
    // spread workers' nodes, mirroring real streaming use (and warming
    // the threads/pinning before timing starts).
    touch_bands(&mut src, &mut dst, cpus.len());
    let mut gbps = Vec::with_capacity(widths.len());
    for &w in &widths {
        let secs = time_copy(&src, &mut dst, w);
        if secs <= 0.0 {
            return None;
        }
        let traffic = (n * size_of::<u64>() * 2) as f64;
        gbps.push(traffic / secs / 1e9);
    }
    pick_knee(&widths, &gbps)
}

/// Initialize both arrays in parallel over `w` spread-pinned workers
/// (see [`calibrate`]).
fn touch_bands(src: &mut [u64], dst: &mut [u64], w: usize) {
    let chunk = src.len().div_ceil(w.max(1));
    std::thread::scope(|s| {
        for (i, (sc, dc)) in src.chunks_mut(chunk).zip(dst.chunks_mut(chunk)).enumerate() {
            s.spawn(move || {
                crate::affinity::pin_spread_worker(i);
                for (k, s) in sc.iter_mut().enumerate() {
                    *s = k as u64;
                }
                dc.fill(0);
            });
        }
    });
}

/// Fastest of [`ITERS_PER_WIDTH`] parallel copies over `w` spread-pinned
/// workers, in seconds. A barrier aligns worker starts so spawn skew
/// does not pollute the short (millisecond-scale) runs.
fn time_copy(src: &[u64], dst: &mut [u64], w: usize) -> f64 {
    let mut min = f64::INFINITY;
    for _ in 0..ITERS_PER_WIDTH {
        let chunk = src.len().div_ceil(w.max(1));
        let barrier = std::sync::Barrier::new(w + 1);
        // Set after the barrier, read after the scope joins every
        // worker, so the elapsed time is the full parallel copy
        // (main-thread arrival skew ~µs against ms-scale runs).
        let mut t0 = None;
        std::thread::scope(|s| {
            for (i, (sc, dc)) in src.chunks(chunk).zip(dst.chunks_mut(chunk)).enumerate() {
                let barrier = &barrier;
                s.spawn(move || {
                    crate::affinity::pin_spread_worker(i);
                    barrier.wait();
                    dc.copy_from_slice(sc);
                });
            }
            barrier.wait();
            t0 = Some(Instant::now());
        });
        min = min.min(
            t0.expect("barrier passed, timestamp set")
                .elapsed()
                .as_secs_f64(),
        );
    }
    min
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sweep_covers_powers_of_two_plus_full_width() {
        assert_eq!(sweep_widths(16), vec![1, 2, 4, 8, 16]);
        assert_eq!(sweep_widths(12), vec![1, 2, 4, 8, 12]);
        assert_eq!(sweep_widths(5), vec![1, 2, 4, 5]);
        assert_eq!(sweep_widths(2), vec![1, 2]);
        assert_eq!(sweep_widths(1), vec![1]);
    }

    #[test]
    fn knee_is_first_width_within_5pct_of_peak() {
        // Saturating curve: width 4 first reaches 95% of the 20.1 peak.
        assert_eq!(
            pick_knee(&[1, 2, 4, 8, 16], &[10.0, 19.0, 20.0, 20.1, 20.0]),
            Some(4)
        );
        // Still scaling at the top: keep the full width.
        assert_eq!(pick_knee(&[1, 2, 4], &[10.0, 20.0, 30.0]), Some(4));
        // Single width always wins when positive.
        assert_eq!(pick_knee(&[8], &[15.0]), Some(8));
        // Degenerate input falls back (callers use the spread width).
        assert_eq!(pick_knee(&[1, 2], &[0.0, 0.0]), None);
        assert_eq!(pick_knee(&[], &[]), None);
    }

    #[test]
    fn stream_width_stays_in_range() {
        // True by construction under any env (override clamps, fallback
        // is the spread width) — guards future refactors, not values.
        let spread = crate::affinity::spread_cpus()
            .map(|c| c.len())
            .unwrap_or(1)
            .max(1);
        let w = stream_width();
        assert!((1..=spread).contains(&w), "width {w} outside 1..={spread}");
    }

    #[test]
    fn width_override_pins_exactly() {
        // Env-mutating test: hold the crate-wide SONIUM_* lock across
        // the whole set/call/restore window (see `env_test_lock`).
        let _guard = super::super::env_test_lock();
        // SAFETY: holding the intra-crate env lock; restored below.
        unsafe {
            let prev = std::env::var_os("SONIUM_STREAM_WIDTH");
            std::env::set_var("SONIUM_STREAM_WIDTH", "2");
            let spread = crate::affinity::spread_cpus()
                .map(|c| c.len())
                .unwrap_or(1)
                .max(1);
            assert_eq!(stream_width(), 2.min(spread));
            match prev {
                Some(value) => std::env::set_var("SONIUM_STREAM_WIDTH", value),
                None => std::env::remove_var("SONIUM_STREAM_WIDTH"),
            }
        }
    }
}
