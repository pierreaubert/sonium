# foam76_tension CUDA roofline (Phase G0 proxy, 2026-10-15)

Verdict: **GPU not recommended for this deck.** A persistent-kernel proxy of
the native LAW76 replay runs **103x slower than 1-thread CPU** (6.92 s vs
0.067 s sonium native, 12x slower than the 0.58 s engine serial). Limiter:
**strictly serial chain on ratioed FP64**, not bandwidth. No full CUDA port
is warranted; this proxy report stands as the deck's distance-to-roofline
record.

Method: G0 proxy (see `oc-opt-20261015.md` methodology). No `sonium.cuda`
explicit arm exists, and `ncu` counters are unavailable on this box
(`ERR_NVGPUCTRPERM`: needs root `NVreg_RestrictProfilingToAdminHost=0`), so
the roofline rests on device-timed walls, measured machine ceilings, and
differential timing instead of counter replay. Raw outputs and sources sit
beside this file (`foam76_tension_cuda_*.txt`,
`foam76_tension_cuda_*.cu`).

## Machine ceilings (RTX PRO 6000 Blackwell, CC 12.0, 188 SMs, 128 MB L2)

Re-measured same-day, device-timed (necking machine source):

| Quantity | Value |
| -------- | ----- |
| Empty kernel launch | 6.85 us |
| Launch+sync roundtrip (host-stepped floor) | 8.27 us/step |
| 4 KB H2D+D2H pinned roundtrip | 14.8 us |
| H2D pinned BW (256 MB) | 22.0 GB/s |
| STREAM triad (DRAM ceiling) | 2160 GB/s |
| FP64 MAD (compute ceiling) | 479.9 GFLOP/s |
| FP64 ridge point | 0.22 FLOP/byte |

Host-stepped verdict first: one step needs a launch+sync (8.3 us) before
any compute against a 2.19 us/step CPU budget -- the floor alone is 3.8x
over budget. Only a persistent on-device loop could compete, so the
proxy is exactly that design.

## Proxy design (deck-faithful persistent kernel)

`foam76_tension_cuda_proxy.cu`: same kernel as the compression twin, 1
block x 1 thread over the exact 30720-substep replay with the +0.02 mm
tension path, SAMP76 quadratic return with the full 14-solve yielding
step, midpoint work booking, FP64 throughout. State ~100 bytes,
L2-resident.

Fidelity checks (measured, not assumed): final (work, sxx, ep) =
(3.991494e-4, -2.315152e5, 6.614719e-4) **bit-identical to the CPU
replay**; N=1 first step elastic with the correct tensile sign;
bit-identical repeats; yielding fraction 2924/30720 (9.5%). Physics
validated BEFORE timing.

## Roofline numbers

Per-step proxy wall (device-timed, warmed clocks, 3 repeats within
21 ppm): **225.24 us/step**. Full-deck wall measured: **6.919 s**.

| Quantity | Value |
| -------- | ----- |
| Work estimate | ~106 MFLOP total, ~3.5 kFLOP/step (+-30%) |
| Achieved throughput | 0.0153 GFLOP/s |
| Distance to FP64 ceiling | **~31000x** (0.0032%) |
| Traffic estimate | ~100 B state, L2-resident (~3 MB total) |
| Arithmetic intensity | ~35 FLOP/byte (160x above ridge: compute-side) |
| Empty-force floor (loop+path+booking) | 4.70 us/step (2.1%) |
| Force-math share | **97.9%** (single-thread FP64 latency-bound) |
| Launch config | 1 thread / 188 SMs (starved by construction) |

Perfect-port bound: the replay is a strictly serial chain
(parallelism exactly 1). A single latency-bound GPU thread delivers
~1% of one SM's 2.55 GFLOP/s FP64 stream (~0.03 GFLOP/s) against the
CPU's effective 1.58 GFLOP/s on the same chain (106 MFLOP / 0.067 s),
so a zero-overhead port still floors ~50x above CPU. Measured proxy
is 103x short (2.19 us/step CPU) -- the same ratio as the
compression twin, as expected for one kernel with fewer yielding
steps.

## Reading

- Same limiter as the compression twin: one thread, dependent
  chains, ratioed FP64. Nothing to parallelize.
- Bandwidth is irrelevant (L2-resident, compute-side by 160x). The
  G1 gate fails 103x short.
- Reproduce: `nvcc -O2 -arch=native -o foam76t_proxy
  foam76_tension_cuda_proxy.cu && ./foam76t_proxy 30720` (7 s, idle
  GPU; run once to warm clocks, then stable).
