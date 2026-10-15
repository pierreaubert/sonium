# foam33_cube CUDA roofline (Phase G0 proxy, 2026-10-15)

Verdict: **GPU not recommended for this deck.** A persistent-kernel proxy of
the native LAW33 replay runs **260x slower than 1-thread CPU** (11.63 s vs
0.045 s sonium native, 22x slower than the 0.54 s engine serial). Limiter:
**strictly serial chain on ratioed FP64**, not bandwidth. No full CUDA port
is warranted; this proxy report stands as the deck's distance-to-roofline
record.

Method: G0 proxy (see `oc-opt-20261015.md` methodology). No `sonium.cuda`
explicit arm exists, and `ncu` counters are unavailable on this box
(`ERR_NVGPUCTRPERM`: needs root `NVreg_RestrictProfilingToAdminHost=0`), so
the roofline rests on device-timed walls, measured machine ceilings, and
differential timing instead of counter replay. Raw outputs and sources sit
beside this file (`foam33_cube_cuda_*.txt`, `foam33_cube_cuda_*.cu`).

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
any compute against a 1.46 us/step CPU budget -- the floor alone is 5.7x
over budget. Only a persistent on-device loop could compete, so the
proxy is exactly that design.

## Proxy design (deck-faithful persistent kernel)

`foam33_cube_cuda_proxy.cu`: 1 block x 1 thread over the exact
30720-substep replay (-0.5 mm compression path), logarithmic axial
increments with the geometrical volume ratio, LAW33 plain card
(trapped-air pressure, principal cap via 12-sweep Jacobi eigensolve,
numerical tangent with 12 extra solves when yielding), midpoint work
booking, FP64 throughout. State ~100 bytes, L2-resident.

Fidelity checks (measured, not assumed): final (work, sxx) =
(5.326268e-2, 5.0e5) **bit-identical to the CPU replay**; N=1 first
step elastic with the correct compressive sign; bit-identical
repeats; yielding fraction 19508/30720 (63%). Physics validated
BEFORE timing. (The ep tally is informational: the engine PLAS
variable stays zero on this fixture.)

## Roofline numbers

Per-step proxy wall (device-timed, warmed clocks, 2 repeats within
55 ppm): **378.69 us/step**. Full-deck wall measured: **11.633 s**.

| Quantity | Value |
| -------- | ----- |
| Work estimate | ~653 MFLOP total, ~21 kFLOP/step (+-30%) |
| Achieved throughput | 0.056 GFLOP/s |
| Distance to FP64 ceiling | **~8500x** (0.012%) |
| Traffic estimate | ~100 B state, L2-resident (~3 MB total) |
| Arithmetic intensity | ~200 FLOP/byte (900x above ridge: compute-side) |
| Empty-force floor (loop+path+booking) | 4.70 us/step (1.2%) |
| Force-math share | **98.8%** (single-thread FP64 latency-bound) |
| Launch config | 1 thread / 188 SMs (starved by construction) |

Perfect-port bound: strictly serial chain (parallelism exactly 1). A
single latency-bound GPU thread delivers ~1% of one SM's 2.55 GFLOP/s
FP64 stream (~0.03 GFLOP/s) against the CPU's effective 14.6 GFLOP/s
on the same chain (653 MFLOP / 0.045 s), so a zero-overhead port
still floors orders of magnitude above CPU. Measured proxy is 260x
short (1.46 us/step CPU) -- worse than the foam76 twins because the
Jacobi solve is heavier per substep and yields on 63% of the path.

## Reading

- Same limiter as the foam76 twins, amplified: one thread, heavier
  dependent chains (Jacobi eigensolves), ratioed FP64. Nothing to
  parallelize.
- Bandwidth is irrelevant (L2-resident, compute-side by 900x). The
  G1 gate fails 260x short.
- Reproduce: `nvcc -O2 -arch=native -o foam33_proxy
  foam33_cube_cuda_proxy.cu && ./foam33_proxy 30720` (12 s, idle
  GPU; run once to warm clocks, then stable).
