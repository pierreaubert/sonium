# beam_3d_cantilever CUDA roofline (Phase G0 proxy, 2026-10-15)

Verdict: **GPU not recommended for this deck.** A persistent-kernel proxy of
the native explicit step runs **53x slower than 1-thread CPU** (161 s vs
3.05 s sonium native EAS). Limiter: **parallelism starvation + ratioed
FP64**, not bandwidth. No full CUDA port is warranted; this proxy report
stands as the deck's distance-to-roofline record.

Method: G0 proxy (see `oc-opt-20261015.md` methodology). No `sonium.cuda`
explicit arm exists, and `ncu` counters are unavailable on this box
(`ERR_NVGPUCTRPERM`: needs root `NVreg_RestrictProfilingToAdminHost=0`), so
the roofline rests on device-timed walls, measured machine ceilings, and
differential timing instead of counter replay. Raw outputs and sources sit
beside this file (`beam_3d_cantilever_cuda_proxy.txt`,
`beam_3d_cantilever_cuda_proxy.cu`).

This deck shares the cantilever mesh exactly (10x2x2, 99 nodes, 40 bricks);
only the tip load differs (1000 N in Y vs 100 N in Z). The proxy is the same
persistent design with the load axis retargeted; per-step walls agree with
cantilever to 0.4%.

## Machine ceilings (RTX PRO 6000 Blackwell, CC 12.0, 188 SMs, 128 MB L2)

Measured same-box same-GPU (`necking_bar_cuda_machine.cu`, 2026-10-15):

| Quantity | Value |
| -------- | ----- |
| Launch+sync roundtrip (host-stepped floor) | 7.14 us/step |
| STREAM triad (DRAM ceiling) | 2161 GB/s |
| FP64 MAD (compute ceiling) | 543.6 GFLOP/s |
| FP64 ridge point | 0.25 FLOP/byte |

Host-stepped verdict first: one step needs a launch+sync (7.1 us) before any
compute against an 11.7 us/step CPU budget. Only a persistent on-device time
loop could compete, so the proxy is exactly that design.

## Proxy design (deck-faithful persistent kernel)

`beam_3d_cantilever_cuda_proxy.cu`: 1 block x 128 threads, device-side loop
over N steps, exact 10x2x2 mesh topology (99 nodes, 40 hex8 bricks of
0.1 x 0.025 x 0.025 m), 8-color element ordering (5 disjoint elements per
color, `__syncthreads` between colors), one host-built EAS-condensed 24x24
stiffness applied as a dgemv per element, Verlet update, tip cloads
(1000 N in Y with the deck FUNCT ramp), root pinning via zero force rows
exactly like the native arm, FP64 throughout. State ~8 KB, L2-resident.
Fixed dt at 0.8x the native mean step (3.06 us).

Fidelity checks (measured, not assumed): N=1 forces are zero (no
prestress); the host stiffness checksum matches the cantilever build
exactly (same brick); N=20000 tip dy = 1.941402e-4, exactly 10x the
cantilever N=20000 tip (linear-physics confirmation); full-N tip dy =
3.123639e-3 vs CPU sonium 3.123846e-3 (0.007%).

## Roofline numbers

Per-step proxy wall (device-timed): **494.7 us/step** (full-N run;
N=20000 short run agrees at 492.6 us/step). Full-deck run to t = 1.0 s
(326,371 steps): **161 s**.

| Quantity | Value |
| -------- | ----- |
| Work estimate | ~94 kFLOP/step (~31 GFLOP total) |
| Achieved throughput | 0.19 GFLOP/s |
| Distance to FP64 ceiling | **~2800x** (0.035%) |
| Traffic estimate | ~25 KB/step, L2-resident (~8 GB total) |
| Arithmetic intensity | ~3.8 FLOP/byte (15x above ridge: compute-side) |
| Empty-force floor (loop+barriers+update) | 2.9 us/step (0.6%) |
| Force-math share | **99.4%** (single-warp FP64 latency-bound) |
| Launch config | 1 block / 188 SMs (0.5% SM occupancy by construction) |

Perfect-port bound: the deck exposes at most ~139-way parallelism
(40 elements + 99 nodes), i.e. one SM at 2.89 GFLOP/s, so a zero-overhead
port floors at 94 kFLOP / 2.89 GFLOP/s = **32.5 us/step** -- still 2.8x
slower than the CPU's 11.7 us/step. The CPU's effective 8.0 GFLOP/s already
exceeds one Blackwell SM's FP64 share. The G1 gate fails; no port follows.

## Reading

- Same limiter class as cantilever and necking_bar: occupancy/latency-bound
  (5 active threads of 128, 1 block of 188 SMs), bandwidth irrelevant
  (L2-resident, compute-side of ridge by 15x).
- Reproduce: `nvcc -O2 -arch=native -o beam_proxy
  beam_3d_cantilever_cuda_proxy.cu && ./beam_proxy 20000` (10 s, idle GPU).
