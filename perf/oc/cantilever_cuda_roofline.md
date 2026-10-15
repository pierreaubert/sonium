# cantilever CUDA roofline (Phase G0 proxy, 2026-10-15)

Verdict: **GPU not recommended for this deck.** A persistent-kernel proxy of
the native explicit step runs **53x slower than 1-thread CPU** (161 s vs
3.06 s sonium native EAS, 7.8x slower than the 20.7 s engine serial).
Limiter: **parallelism starvation + ratioed FP64**, not bandwidth. No full
CUDA port is warranted; this proxy report stands as the deck's
distance-to-roofline record.

Method: G0 proxy (see `oc-opt-20261015.md` methodology). No `sonium.cuda`
explicit arm exists, and `ncu` counters are unavailable on this box
(`ERR_NVGPUCTRPERM`: needs root `NVreg_RestrictProfilingToAdminHost=0`), so
the roofline rests on device-timed walls, measured machine ceilings, and
differential timing instead of counter replay. Raw outputs and sources sit
beside this file (`cantilever_cuda_proxy.txt`, `cantilever_cuda_proxy.cu`).

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

`cantilever_cuda_proxy.cu`: 1 block x 128 threads, device-side loop over
N steps, exact 10x2x2 cantilever topology (99 nodes, 40 hex8 bricks of
0.1 x 0.025 x 0.025 m), 8-color element ordering (5 disjoint elements per
color, `__syncthreads` between colors), one host-built EAS-condensed 24x24
stiffness applied as a dgemv per element (all deck bricks are congruent),
Verlet update, tip cloads (100 N with the deck FUNCT ramp), root pinning via
zero force rows exactly like the native arm, FP64 throughout. State ~8 KB,
L2-resident. Fixed dt at 0.8x the native mean step (3.06 us).

Fidelity checks (measured, not assumed): the host stiffness reproduces the
Rust `eas_linear_stiffness` entries exactly (ke0/ke100/ke575 bit-identical,
rigid-shift null residual 2.4e-6 against 3.5e9 entries); N=1 forces are zero
(no prestress); N=20000 tip dz = 1.94e-5 vs quasi-static 1.92e-5 (+1%);
full-N tip dz = 3.123639e-4 vs CPU sonium 3.123846e-4 (0.007%) and vs the EAS
static replica 3.1355e-4 (-0.4%); bit-identical tip across repeats; linear
2k->20k scaling (9.997x).

## Roofline numbers

Per-step proxy wall (device-timed, 2 full-N repeats): **492.5/494.6 us/step**
(spread 0.4%). Full-deck run to t = 1.0 s (326,371 steps): **161 s**.

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

Perfect-port bound (kills the "bad kernel" objection without writing G1):
the deck exposes at most ~139-way parallelism (40 elements + 99 nodes),
i.e. one SM. One SM's FP64 stream is 543.6/188 = 2.89 GFLOP/s, so
a zero-overhead port floors at 94 kFLOP / 2.89 GFLOP/s = **32.5 us/step** --
still 2.8x slower than the CPU's 11.7 us/step. The CPU's effective
8.0 GFLOP/s already exceeds one Blackwell SM's FP64 share. Measured proxy
is 15x above even that floor (latency, barriers, dgemv trip counts).

## Reading

- The deck is occupancy/latency-bound at every level: 5 active threads of
  128 in force phases, 1 block of 188 SMs, FP64 pipes fed by one warp with
  dependent dgemv chains on 64:1-ratioed hardware.
- Bandwidth is irrelevant here (L2-resident, compute-side of ridge by 15x);
  porting effort should go to bigger meshes first, where many blocks can
  hide FP64 latency. The G1 full-port gate for small decks: proxy must
  project within striking distance of CPU. cantilever fails 53x short
  (and 2.8x short even with zero overhead).
- Reproduce: `nvcc -O2 -arch=native -o canti_proxy
  cantilever_cuda_proxy.cu && ./canti_proxy 20000` (10 s, idle GPU).
