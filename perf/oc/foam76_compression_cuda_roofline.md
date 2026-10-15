# foam76_compression CUDA roofline (Phase G0 proxy, 2026-10-15)

Verdict: **GPU not recommended for this deck.** A persistent-kernel proxy of
the native LAW76 replay runs **103x slower than 1-thread CPU** (8.53 s vs
0.083 s sonium native, 15x slower than the 0.57 s engine serial). Limiter:
**strictly serial chain on ratioed FP64**, not bandwidth. No full CUDA port
is warranted; this proxy report stands as the deck's distance-to-roofline
record.

Method: G0 proxy (see `oc-opt-20261015.md` methodology). No `sonium.cuda`
explicit arm exists, and `ncu` counters are unavailable on this box
(`ERR_NVGPUCTRPERM`: needs root `NVreg_RestrictProfilingToAdminHost=0`), so
the roofline rests on device-timed walls, measured machine ceilings, and
differential timing instead of counter replay. Raw outputs and sources sit
beside this file (`foam76_compression_cuda_*.txt`,
`foam76_compression_cuda_*.cu`).

## Machine ceilings (RTX PRO 6000 Blackwell, CC 12.0, 188 SMs, 128 MB L2)

Re-measured same-day, device-timed (necking machine source):

| Quantity | Value |
| -------- | ----- |
| Empty kernel launch | 6.85 us |
| Launch+sync roundtrip (host-stepped floor) | 8.27 us/step |
| 4 KB H2D+D2H pinned roundtrip | 14.8 us |
| H2D / D2H pinned BW (256 MB) | 22.0 / 28.0 GB/s |
| STREAM triad (DRAM ceiling) | 2160 GB/s |
| FP64 MAD (compute ceiling) | 479.9 GFLOP/s |
| FP64 ridge point | 0.22 FLOP/byte |
| Clocks under proxy load | 2880 MHz SM |

(necking-day values were 2161 GB/s / 543.6 GFLOP/s; the FP64 gap is
clock/thermal variance. Same-day ceilings apply below.)

Host-stepped verdict first: one step needs a launch+sync (8.3 us) before
any compute against a 2.70 us/step CPU budget -- the floor alone is 3x
over budget. Only a persistent on-device loop could compete, so the
proxy is exactly that design.

## Proxy design (deck-faithful persistent kernel)

`foam76_compression_cuda_proxy.cu`: 1 block x 1 thread, device-side loop
over the exact 30720-substep replay (corner-split path, -0.04 mm
load/hold/unload/hold), logarithmic axial increments, SAMP76 quadratic
return (trial, von Mises check, 64-iteration bisection, numerical
tangent with 12 extra solves when yielding -- the same 14-solve step
the CPU runs), midpoint stress-work booking, FP64 throughout. State
~100 bytes, L2-resident. Fixed discretization identical to the CPU
replay (no adaptive driver exists on this path).

Fidelity checks (measured, not assumed): final (work, sxx, ep) =
(1.028509e-3, 3.121125e5, 8.917500e-4) **bit-identical to the CPU
replay** at every printed digit; N=1 first step is elastic with the
correct compressive sign; bit-identical repeats; yielding fraction
3666/30720 (12%) matches the path's yield windows. Physics validated
BEFORE timing.

## Roofline numbers

Per-step proxy wall (device-timed, warmed clocks, 2 identical repeats
at 4 ppm): **277.73 us/step**. Full-deck wall is measured, not
projected (8.53 s -- no 45-minute hog question arises): **8.532 s**.

| Quantity | Value |
| -------- | ----- |
| Work estimate | ~131 MFLOP total, ~4.3 kFLOP/step (+-30%) |
| Achieved throughput | 0.0154 GFLOP/s |
| Distance to FP64 ceiling | **~31000x** (0.0032%) |
| Traffic estimate | ~100 B state, L2-resident (~3 MB total) |
| Arithmetic intensity | ~44 FLOP/byte (200x above ridge: compute-side) |
| Empty-force floor (loop+path+booking) | 4.70 us/step (1.7%) |
| Force-math share | **98.3%** (single-thread FP64 latency-bound) |
| Launch config | 1 thread / 188 SMs (starved by construction) |

Perfect-port bound (kills the "bad kernel" objection without writing
G1): the replay is a strictly serial chain -- each substep commits
state the next one reads -- so available parallelism is exactly 1.
One SM's FP64 stream (479.9/188 = 2.55 GFLOP/s) needs ~1536 threads
to fill; a single latency-bound thread with deep dependent chains
(trial->bisection->tangent) delivers ~1% of it (~0.03 GFLOP/s),
against the CPU's effective 1.58 GFLOP/s on the same chain. A
zero-overhead port still floors ~50x above CPU. Measured proxy is
103x short (2.70 us/step CPU).

## Reading

- The deck is latency-bound at every level: 1 active thread, 1 block
  of 188 SMs, FP64 pipes fed by one thread with dependent chains on
  64:1-ratioed hardware. There is nothing to parallelize: the physics
  is one material point marched serially.
- Bandwidth is irrelevant here (L2-resident, compute-side of ridge by
  200x). The G1 full-port gate for serial decks: proxy must project
  within striking distance of CPU. foam76_compression fails 103x short.
- Reproduce: `nvcc -O2 -arch=native -o foam76c_proxy
  foam76_compression_cuda_proxy.cu && ./foam76c_proxy 30720` (9 s,
  idle GPU; first cold-clocks run reads high, then stable).
