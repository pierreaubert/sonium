# rail_crush CUDA roofline (Phase G0 proxy, 2026-10-15)

Verdict: **GPU not recommended for this deck.** A persistent-kernel proxy of
the native explicit step runs **54x slower than 1-thread CPU** (projected
16.5 s vs 0.303 s sonium native, 30x slower than the 0.55 s engine serial).
Limiter: **parallelism starvation + ratioed FP64**, not bandwidth. No full
CUDA port is warranted; this proxy report stands as the deck's
distance-to-roofline record.

Method: G0 proxy (see `oc-opt-20261015.md` methodology). No `sonium.cuda`
explicit arm exists, and `ncu` counters are unavailable on this box
(`ERR_NVGPUCTRPERM`: needs root `NVreg_RestrictProfilingToAdminHost=0`), so
the roofline rests on device-timed walls, measured machine ceilings, and
differential timing instead of counter replay. Raw outputs and sources sit
beside this file (`rail_crush_cuda_proxy.txt`, `rail_crush_cuda_proxy.cu`).

## Machine ceilings (RTX PRO 6000 Blackwell, CC 12.0, 188 SMs, 128 MB L2)

Measured fresh for this deck, device-timed (necking machine source,
same box/GPU; matches the necking numbers within noise):

| Quantity | Value |
| -------- | ----- |
| Empty kernel launch | 4.70 us |
| Launch+sync roundtrip (host-stepped floor) | 7.97 us/step |
| 4 KB H2D+D2H pinned roundtrip | 15.1 us |
| H2D / D2H pinned BW (256 MB) | 26.5 / 28.6 GB/s |
| STREAM triad (DRAM ceiling) | 2160 GB/s |
| FP64 MAD (compute ceiling) | 543.9 GFLOP/s |
| FP64 ridge point | 0.25 FLOP/byte |

Host-stepped verdict first: one step needs a launch+sync (8.0 us) plus a
~17 KB state roundtrip (~15 us small-transfer floor) before any compute,
i.e. a ~23 us/step floor against a 46.4 us/step CPU budget. Only a
persistent on-device time loop could compete, so the proxy is exactly
that design.

## Proxy design (deck-faithful persistent kernel)

`rail_crush_cuda_proxy.cu`: 1 block x 128 threads, device-side loop over
N steps, exact 10x2x2 rail topology (99 nodes, 40 hex8 bricks,
0.2 x 0.05 x 0.05 m column), 8-color element ordering (5 disjoint
elements per color, `__syncthreads` between colors), per-Gauss
NeoHookean in u-form (F = I + u*dN, det, log, P = mu*F +
(lambda*lnJ - mu)*F^-T, forces -= residual scatter), position-Verlet
update (drift, forces at end-of-step state, full-step velocity — one
force evaluation per step like `ExplicitStepDriver::step`), two
exact-multiplier rigid walls (fixed anvil at x = 0, crusher at
x = 0.2 - 0.1*t, persistent engagement + one inelastic adoption per
transition, mirroring `rigid_wall_multiplier`), FP64 throughout. State
16.6 KB, L2-resident. Fixed dt at 0.8x the native mean step (the native
adaptive driver's unlimited shrinkage is a stabilizer a fixed-step proxy
lacks).

Fidelity checks (measured, not assumed): N=1 forces are bit-exact zero
(u-form gives exact F = I at rest; an F-from-coords draft left mu*eps
residue that engaged the touching anvil one step early) with the
expected 0+9 bootstrap engagement; N=2000 tip dx = -4.900e-4 vs
quasi-static -4.902e-4 (0.05%) and crusher FNX = 1.222e6 vs 1.226e6
(0.3%); N=8160 stop-time tip dx = -2.000e-3 vs `probes.json` -2.0e-3
(0.01%) and crusher FNX = 5.016e6 vs 5.0e6 (0.3%, tolerance 15%);
anvil/crusher balance within 0.4%; face-only 9+9 engagement; two N=2000
runs bit-identical; linear 2k->8.16k scaling (4.078x wall over 4.08x
steps). Cross-check against the native arm itself (`--quick` sonium cell,
6528 steps, dt_max 3.404e-6 s): native FNX 5.040e6 vs proxy 5.016e6
(0.5%), native DX -1.99973e-3 vs proxy -1.99976e-3 (0.001%). Physics
validated BEFORE timing, triangulated three ways (proxy vs native vs
engine-golden probes).

## Roofline numbers

Per-step proxy wall (device-timed, 3 timed runs): **2523.9 us/step**
(spread +-0.05%). Full-deck projection (6528 sonium cycles, proven
linear): **16.5 s**.

| Quantity | Value |
| -------- | ----- |
| Work estimate | ~150 kFLOP/step (~0.98 GFLOP total, +-30%) |
| Achieved throughput | 0.059 GFLOP/s |
| Distance to FP64 ceiling | **~9200x** (0.011%) |
| Traffic estimate | ~30 KB/step, L2-resident (~0.2 GB total) |
| Arithmetic intensity | ~5 FLOP/byte (20x above ridge: compute-side) |
| Empty-force floor (loop+barriers+update+wall checks) | 54.9 us/step (2.2%) |
| Force-math share | **97.8%** (single-warp FP64 latency-bound) |
| Launch config | 1 block / 188 SMs (0.5% SM occupancy by construction) |

Perfect-port bound (kills the "bad kernel" objection without writing G1):
the deck exposes at most ~518-way parallelism (320 Gauss points + 99
nodes + 99 wall slaves), i.e. one SM. One SM's FP64 stream is 543.9/188
= 2.89 GFLOP/s, so a zero-overhead port floors at 150 kFLOP / 2.89
GFLOP/s = **51.9 us/step** -- still 1.1x slower than the CPU's 46.4
us/step. The CPU's effective 3.2 GFLOP/s already exceeds one Blackwell
SM's FP64 share. Measured proxy is 49x above even that floor (latency,
barriers, div/log).

## Reading

- The deck is occupancy/latency-bound at every level: 5 active threads of
  128 in force phases, 1 block of 188 SMs, FP64 pipes fed by one warp with
  deep dependent chains (F->det->inv->P->scatter) on 64:1-ratioed hardware.
- Bandwidth is irrelevant here (L2-resident, compute-side of ridge by 20x);
  porting effort should go to bigger meshes first, where many blocks can
  hide FP64 latency. The G1 full-port gate for small decks: proxy must
  project within striking distance of CPU. rail_crush fails 54x short.
- The serial wall loop on thread 0 is conservative for the GPU (a G1 port
  would parallelize it); it costs ~5 kFLOP of the ~150 kFLOP step, so it
  cannot change the verdict class.
- Reproduce: `nvcc -O2 -arch=native -o rail_proxy
  rail_crush_cuda_proxy.cu && ./rail_proxy 2000` (5 s, idle GPU).
