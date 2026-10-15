# taylor_plastic CUDA roofline (Phase G0 proxy, 2026-10-15)

Verdict: **GPU not recommended for this deck.** A persistent-kernel proxy of
the native JC explicit run (780 nodes, 540 bricks, 123 steps) runs **10.5x
slower than 1-thread CPU** (0.917 s vs 0.087 s sonium native, 16x slower
than the 0.056 s engine serial). Limiter: **divergent FP64 return math on
~84 threads of one SM**, not bandwidth. No full CUDA port is warranted;
this proxy report stands as the deck's distance-to-roofline record. (Closest
dynamics deck of the campaign after the thick cylinder; the bound below
quantifies the remaining headroom.)

Method: G0 proxy (see `oc-opt-20261015.md` methodology). No `sonium.cuda`
explicit arm exists, and `ncu` counters are unavailable on this box
(`ERR_NVGPUCTRPERM`: needs root `NVreg_RestrictProfilingToAdminHost=0`), so
the roofline rests on device-timed walls, measured machine ceilings, and
differential timing instead of counter replay. Raw outputs and sources sit
beside this file (`taylor_plastic_cuda_*.txt`,
`taylor_plastic_cuda_*.cu`).

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

Host-stepped verdict first: a launch+sync per step (8.3 us) fits inside
the 707 us/step CPU budget, so the floor alone does not kill a GPU port
here -- unlike the smaller decks. The proxy is still a persistent
on-device loop (the design a real port would use), so the comparison is
fair, not floor-bound.

## Proxy design (deck-faithful persistent kernel)

`taylor_plastic_cuda_proxy.cu`: 1 block x 128 threads over the exact deck
mesh (780 nodes, 540 bricks, deck masses, greedy 9-color ordering with
`__syncthreads` between colors), per element per step: current-config
centroid Jacobian + gradients, uniform strain increment, quadratic-only
bulk viscosity, Johnson-Cook rated radial return (bracket + 80-iteration
bisection, raw step rate -- the deck's fcut collapses smoothing),
V B^T (sigma - q I) scatter, Flanagan-Belytschko hourglass, erosion at
ep >= 0.18 with force zeroing; rising penalty anvil (227 m/s,
k = 0.5 E Le) with the overdamped dashpot; velocity-Verlet at fixed
dt = stop/123 (the native rides its dt_max cap: 123 steps).

Fidelity checks (measured, not assumed) at the 123-step deck:

| Probe | Proxy | Native | Match |
| ----- | ----- | ------ | ----- |
| impact face DZ | 6.641264e-4 | 6.639737e-4 | 0.02% |
| impact brick PLAS | 1.820981e-1 | 1.804519e-1 | 0.9% |
| anvil FNZ plateau | -3.973061e4 | -4.048910e4 | 1.9% |
| eroded | 45/540 | 45/540 | exact |
| max ep | 1.820981e-1 | 1.821363e-1 | 0.02% |

N=1 first step is pre-contact (all zeros, correct: the bar hovers
0.227 mm over the anvil); bit-identical repeats. Physics validated
BEFORE timing. Documented simplifications: fixed dt (native adapts
around the same mean), consistent-tangent FLOP booked in the estimate
instead of computed (~2% of the return), no TH/energy bookkeeping.

## Roofline numbers

Per-step proxy wall (device-timed, 2 repeats within 50 ppm):
**7452 us/step**. Full deck measured, not projected: **0.917 s**.

| Quantity | Value |
| -------- | ----- |
| Work estimate | ~1.6 MFLOP/step (+-40%; yielding ~7.3k, elastic ~1.4k per element) |
| Achieved throughput | 0.21 GFLOP/s |
| Distance to FP64 ceiling | **~2300x** (0.044%) |
| Traffic estimate | ~150 KB state, L2-resident (~0.5 MB/step over passes) |
| Arithmetic intensity | ~3 FLOP/byte (14x above ridge: compute-side) |
| Empty-force floor (wall+update+loop) | 35.1 us/step (0.47%) |
| Force-math share | **99.5%** (JC bisection + pow/log on few threads) |
| Launch config | 1 block x 128, <=84 active, 9 color phases |

Perfect-port bound (kills the "bad kernel" objection without writing
G1): the deck offers ~84-way element parallelism (max color) plus a
780-wide update phase -- about one SM's worth of threads. Element
work (1.6 MFLOP/step) at 100% of one SM's FP64 stream (2.55 GFLOP/s
same-day) floors at 627 us vs the CPU's 707 us: parity needs >=89% of
one SM sustained. That is implausible for this loop (data-dependent
bracket exits, 80-iteration bisections with pow/log per trip,
elastic/yielding warp divergence, 9 color syncs, update/wall phases).
A realistic multi-block G1 (2-3x over this 1-block proxy) projects
~3-5x short of CPU. Measured proxy is 10.5x short (707 us/step CPU).

## Reading

- The deck is return-math-bound: 99.5% of the proxy wall is element
  force math on <=84 threads, dominated by ~83 JC flow evals
  (pow + log each) per yielding element-step. Bandwidth is irrelevant
  (L2-resident, compute-side by 14x).
- The G1 gate fails 10.5x short with ~3-5x realistic headroom --
  closest dynamics deck of the campaign, but not within striking
  distance. (Note the CPU bar itself: sonium 0.087 s already trails
  the engine's 0.056 s serial here -- the JC return is expensive on
  both platforms.)
- Amusing differential detail: the EMPTY build reproduces the DZ
  kinematics (6.80e-4 vs 6.64e-4) because the impact face rides the
  anvil wall, while FNZ collapses 7x without element resistance --
  the split cleanly isolates force math from wall-driven motion.
- Reproduce: `nvcc -O2 -arch=native -o taylor_proxy
  taylor_plastic_cuda_proxy.cu && ./taylor_proxy 123` (1 s, idle GPU).
