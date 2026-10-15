# necking_bar CUDA roofline (Phase G0 proxy, 2026-10-15)

Verdict: **GPU not recommended for this deck.** A persistent-kernel proxy of
the native explicit step runs **48x slower than 1-thread CPU** (projected
2730 s vs 56.8 s sonium native, 25x slower than the 111 s engine serial).
Limiter: **parallelism starvation + ratioed FP64**, not bandwidth. No full
CUDA port is warranted; this proxy report stands as the deck's
distance-to-roofline record.

Method: G0 proxy (see `oc-opt-20261015.md` methodology). No `sonium.cuda`
explicit arm exists, and `ncu` counters are unavailable on this box
(`ERR_NVGPUCTRPERM`: needs root `NVreg_RestrictProfilingToAdminHost=0`), so
the roofline rests on device-timed walls, measured machine ceilings, and
differential timing instead of counter replay. Raw outputs and sources sit
beside this file (`necking_bar_cuda_*.txt`, `necking_bar_cuda_*.cu`).

## Machine ceilings (RTX PRO 6000 Blackwell, CC 12.0, 188 SMs, 128 MB L2)

Measured, device-timed (`necking_bar_cuda_machine.cu`):

| Quantity | Value |
| -------- | ----- |
| Empty kernel launch | 2.84 us |
| Launch+sync roundtrip (host-stepped floor) | 7.14 us/step |
| 4 KB H2D+D2H pinned roundtrip | 12.6 us |
| H2D / D2H pinned BW (256 MB) | 26.5 / 26.9 GB/s |
| STREAM triad (DRAM ceiling) | 2161 GB/s |
| FP64 MAD (compute ceiling) | 543.6 GFLOP/s |
| FP64 ridge point | 0.25 FLOP/byte |
| Clocks under proxy load | 2887 MHz SM / 13365 MHz mem |

Host-stepped verdict first: one step needs a launch+sync (7.1 us) plus a
~5 KB state roundtrip (~13 us) before any compute, i.e. a ~20 us/step floor
against a 43.5 us/step CPU budget. Only a persistent on-device time loop
could compete, so the proxy is exactly that design.

## Proxy design (deck-faithful persistent kernel)

`necking_bar_cuda_proxy.cu`: 1 block x 128 threads, device-side loop over
N steps, exact 10x2x2 necking topology (99 nodes, 40 hex8 bricks), 8-color
element ordering (5 disjoint elements per color, `__syncthreads` between
colors), per-Gauss NeoHookean (F, det, log, stress, scatter), Verlet update,
tip cloads with the deck ramp, root pinning, FP64 throughout. State 7.9 KB,
L2-resident. Fixed dt at 0.8x the native mean step (the native adaptive
driver's unlimited shrinkage is a stabilizer a fixed-step proxy lacks).

Fidelity checks (measured, not assumed): N=1 forces are zero to residue
(1.5e-8 N, no prestress); N=2000 tip dx = 1.0e-7 vs quasi-static 1.4e-7;
N=20000 tip dx = 1.3e-6 vs quasi-static 1.4e-6; bit-identical repeats;
linear 2k->20k scaling (9.997x). Physics validated BEFORE timing (the
debugging history matters: a sign-convention slip ran 6% faster while
producing NaN -- non-physical states skew special-function throughput).

## Roofline numbers

Per-step proxy wall (device-timed, 5 repeats incl. 20k): **2091.2 us/step**
(spread +-0.05%). Full-deck projection (1,305,578 sonium cycles, proven
linear -- no 45-minute shared-GPU hog run): **2730 s**.

| Quantity | Value |
| -------- | ----- |
| Work estimate | ~150 kFLOP/step (~196 GFLOP total, +-30%) |
| Achieved throughput | 0.072 GFLOP/s |
| Distance to FP64 ceiling | **~7600x** (0.013%) |
| Traffic estimate | ~14 KB/step, L2-resident (~18 GB total) |
| Arithmetic intensity | ~11 FLOP/byte (43x above ridge: compute-side) |
| Empty-force floor (loop+barriers+update) | 3.8 us/step (0.2%) |
| Force-math share | **99.8%** (single-warp FP64 latency-bound) |
| Launch config | 1 block / 188 SMs (0.5% SM occupancy by construction) |

Perfect-port bound (kills the "bad kernel" objection without writing G1):
the deck exposes at most ~419-way parallelism (320 Gauss points + 99
nodes), i.e. one SM. One SM's FP64 stream is 543.6/188 = 2.89 GFLOP/s, so
a zero-overhead port floors at 150 kFLOP / 2.89 GFLOP/s = **52 us/step** --
still 1.2x slower than the CPU's 43.5 us/step. The CPU's effective
3.4 GFLOP/s already exceeds one Blackwell SM's FP64 share. Measured proxy
is 40x above even that floor (latency, barriers, div/log).

## Reading

- The deck is occupancy/latency-bound at every level: 5 active threads of
  128 in force phases, 1 block of 188 SMs, FP64 pipes fed by one warp with
  deep dependent chains (F->det->inv->P->scatter) on 64:1-ratioed hardware.
- Bandwidth is irrelevant here (L2-resident, compute-side of ridge by 43x);
  porting effort should go to bigger meshes first, where many blocks can
  hide FP64 latency. The G1 full-port gate for small decks: proxy must
  project within striking distance of CPU. necking_bar fails 48x short.
- Reproduce: `nvcc -O2 -arch=native -o necking_proxy
  necking_bar_cuda_proxy.cu && ./necking_proxy 2000` (4 s, idle GPU).
