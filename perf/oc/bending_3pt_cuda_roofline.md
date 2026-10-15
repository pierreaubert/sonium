# bending_3pt CUDA roofline (Phase G0 proxy, 2026-10-15)

Verdict: **GPU not recommended for this deck.** A persistent-kernel proxy of
the native explicit step runs **30x slower than 1-thread CPU** (185 s vs
6.16 s sonium native EAS, 7.3x slower than the 25.3 s engine serial).
Limiter: **parallelism starvation + ratioed FP64**, not bandwidth. No full
CUDA port is warranted; this proxy report stands as the deck's
distance-to-roofline record.

Method: G0 proxy (see `oc-opt-20261015.md` methodology). No `sonium.cuda`
explicit arm exists, and `ncu` counters are unavailable on this box
(`ERR_NVGPUCTRPERM`: needs root `NVreg_RestrictProfilingToAdminHost=0`), so
the roofline rests on device-timed walls, measured machine ceilings, and
differential timing instead of counter replay. Raw outputs and sources sit
beside this file (`bending_3pt_cuda_proxy.txt`,
`bending_3pt_cuda_proxy.cu`).

## Machine ceilings (RTX PRO 6000 Blackwell, CC 12.0, 188 SMs, 128 MB L2)

Measured same-box same-GPU (`necking_bar_cuda_machine.cu`, 2026-10-15):

| Quantity | Value |
| -------- | ----- |
| Launch+sync roundtrip (host-stepped floor) | 7.14 us/step |
| STREAM triad (DRAM ceiling) | 2161 GB/s |
| FP64 MAD (compute ceiling) | 543.6 GFLOP/s |
| FP64 ridge point | 0.25 FLOP/byte |

Host-stepped verdict first: one step needs a launch+sync (7.1 us) before any
compute against a 23.6 us/step CPU budget. Only a persistent on-device time
loop could compete, so the proxy is exactly that design.

## Proxy design (deck-faithful persistent kernel)

`bending_3pt_cuda_proxy.cu`: 1 block x 128 threads, device-side loop over
N steps, exact 20x2x2 bending topology (189 nodes, 80 hex8 bricks of
0.05 x 0.025 x 0.025 m), 8-color element ordering (10 disjoint elements per
color, `__syncthreads` between colors), one host-built EAS-condensed 24x24
stiffness applied as a dgemv per element (all deck bricks are congruent),
Verlet update, center cloads (100 N in Z with the deck FUNCT ramp), pin and
roller support lines via zero force rows exactly like the native arm, FP64
throughout. State ~14 KB, L2-resident. Fixed dt at 0.8x the native mean
step (3.06 us).

Fidelity checks (measured, not assumed): N=1 forces are zero (no
prestress); host stiffness rigid-shift null residual 2.1e-6 against 1.8e9
entries; N=20000 center dz = 1.227e-6 vs quasi-static 1.226e-6 (+0.1%);
full-N center dz = 2.005924e-5 vs CPU sonium 2.006208e-5 (0.014%).

## Roofline numbers

Per-step proxy wall (device-timed): **567.1 us/step** (full-N run; N=20000
short run agrees at 567.2 us/step). Full-deck run to t = 1.0 s
(326,371 steps): **185 s**.

| Quantity | Value |
| -------- | ----- |
| Work estimate | ~189 kFLOP/step (~62 GFLOP total) |
| Achieved throughput | 0.33 GFLOP/s |
| Distance to FP64 ceiling | **~1600x** (0.06%) |
| Traffic estimate | ~50 KB/step, L2-resident (~16 GB total) |
| Arithmetic intensity | ~3.8 FLOP/byte (15x above ridge: compute-side) |
| Empty-force floor (loop+barriers+update) | 6.7 us/step (1.2%) |
| Force-math share | **98.8%** (single-warp FP64 latency-bound) |
| Launch config | 1 block / 188 SMs (0.5% SM occupancy by construction) |

Perfect-port bound: the deck exposes at most ~269-way parallelism
(80 elements + 189 nodes), i.e. about two SMs at ~5.8 GFLOP/s combined, so
a zero-overhead port floors at 189 kFLOP / 5.8 GFLOP/s = **~33 us/step** --
still 1.4x slower than the CPU's 23.6 us/step. The CPU's effective
8.0 GFLOP/s (same EAS-dgemv kernel as cantilever, at 2x the work) already
exceeds the deck's SM-share ceiling. The G1 gate fails; no port follows.

## Reading

- Twice the elements of cantilever at only 15% more wall per step (567 vs
  493 us): 10 parallel elements per color fill warps better than 5. But the
  deck still occupies one block of 188 SMs, so latency dominates either way.
- Bandwidth is irrelevant here (L2-resident, compute-side of ridge by 15x);
  the empty-force differential confirms 98.8% of the wall is force math.
- Reproduce: `nvcc -O2 -arch=native -o bend_proxy
  bending_3pt_cuda_proxy.cu && ./bend_proxy 20000` (11 s, idle GPU).
