# hertz_sphere CUDA roofline (Phase G0 proxy, 2026-10-15)

Verdict: **GPU not recommended for this deck.** A persistent-kernel proxy of
the native explicit step runs **12.6x slower than 1-thread CPU** (116.2 s vs
9.2 s sonium native over the full 105056 steps, 4.5x slower than the 25.6 s
engine serial). Limiter: **parallelism starvation + ratioed FP64**, same as
necking_bar/rail_crush. No full CUDA port is warranted; this proxy report
stands as the deck's distance-to-roofline record.

Method: G0 proxy per `oc-opt-20261015.md` (parent-authored; closes the last
G0 gap on DONE decks). No `sonium.cuda` explicit arm exists and `ncu`
counters are unavailable (`ERR_NVGPUCTRPERM`), so device-timed walls +
measured ceilings + differential timing. Console:
`hertz_sphere_cuda_proxy.txt` (script sections plus a full-N direct block);
source: `hertz_sphere_cuda_proxy.cu`; machine via shared `cuda_machine.cu`.

## Proxy design (deck-faithful persistent kernel)

1 block x 256 threads, device-side loop, FP64. Runtime `.rad` parse (486
nodes, 304 hex8, LAW1 E200G/nu.3/rho7800, 277+208 wall slaves), host-built
linear K per element (24x24, 1.4 MB in global memory — exceeds the 64 KB
constant limit), greedy 12-coloring for race-free scatter, velocity-Verlet
with lumped masses, bottom plane z>=0 fixed + top plane z<=0.1-1e-4*t,
fixed dt = 0.1/105056. Contact models the native exact multiplier as
velocity+position projection (documented deviation; engagement nearly static
at 0.1% strain).

Fidelity (measured): N=1 bit-exact zero; full-N DZ/247 = -9.999905e-06 vs
-1e-5 expected (0.001% — top pole rides the plane); 1106.5 us/step stable
across N=100/2000/full (0.1% spread).

## Roofline numbers

Machine (shared `cuda_machine.cu`, re-measured this run): STREAM 2158 GB/s,
FP64 541.2 GFLOP/s (ridge 0.25 FLOP/byte), launch+sync 6.8 us, 4 KB
roundtrip 11.8 us.

| Quantity | Value |
| -------- | ----- |
| Work estimate | ~350 kFLOP/step, 304 dgemv(24x24) (~37 GFLOP total, +-30%) |
| Achieved throughput | 0.32 GFLOP/s |
| Distance to FP64 ceiling | **~1700x** (0.06%) |
| Traffic | K stream 1.4 MB + state, L2-resident; compute-side of ridge |
| Empty-force floor | 8.7 us/step (0.8%) |
| Force-math share | **99.2%** (single-block FP64 latency-bound) |
| Launch config | 1 block / 188 SMs, 12 colors x ~25 threads |

Perfect-port bound: 350 kFLOP at one SM's 2.88 GFLOP/s FP64 share floors at
121 us/step vs CPU 87.6 us/step — an ideal port still loses 1.4x. The CPU's
effective 4.0 GFLOP/s exceeds one Blackwell SM's FP64 share.

## Reading

- Milder ratio than necking (48x)/rail (54x) only because 304 elements fill
  12 colors x ~25 threads instead of 8x5 — still one block, still starved.
- Reproduce: `bash scripts/perf_oc_gpu.sh hertz_sphere perf/oc` (~15 s) plus
  `./hertz_proxy 105056` for the full-N wall (~2 min, idle GPU).
