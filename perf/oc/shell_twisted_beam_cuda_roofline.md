# shell_twisted_beam G0 CUDA proxy roofline (Unit 4, preliminary)

Date: 2026-10-15. Box: `x86_64` Threadripper PRO 3995WX, GPU
`NVIDIA RTX PRO 6000 Blackwell` (CC 12.0), CUDA 13.3.
Machine profile: Unit 5 `necking_bar_cuda_machine.txt` (same box/GPU):
STREAM triad 2161 GB/s, FP64 MAD 543.6 GFLOP/s, 188 SMs, 128 MB L2.
Proxy: `sonium/perf/oc/shell_twisted_beam_cuda_proxy.cu`
(`nvcc -O2 -arch=native`; EMPTY via `-DEMPTY_FORCE`).
Console: `sonium/perf/oc/shell_twisted_beam_cuda_console.txt`
(direct equivalent of `perf_oc_gpu.sh` sections; the script resolves
paths into the parent tree, so Unit 4 ran the same `nvcc` + `./proxy`
commands directly and archived the outputs verbatim).
GPU was idle (`nvidia-smi` 0%, no compute procs) before every run below.

## Fidelity (before timing)

- N=1: tip 0, max|f| 0, no NaN — zero prestress, clean smoke.
- Full N=300000 (60 ms at dt 2.0e-7 s): tip hold mean (t >= 32 ms,
  140001 samples) 1.364595e-4 m vs native bench 1.350630e-4 m
  (`tmp/u4-twisted.log`, sonium-only quick, correct): **reldiff 1.03%**.
  Same linear CST/DKT physics, independent assembly — validated.
- During bring-up the proxy read 4.5x high; the DKT `a/d`
  coefficients carried spurious squares (fixed, re-validated above).
  The 1.03% residual is dt/rounding (proxy dt 2.0e-7 vs native
  2.015e-7) plus mesh-table rounding (%.6e, ~1e-9 relative).

## Timing (device events)

| run | wall | per step |
| --- | --- | --- |
| N=2000 | 968.6 ms | 484.3 us |
| N=300000 | 145538.7 ms | 485.1 us |
| EMPTY N=2000 | 115.1 ms | 57.5 us |
| EMPTY N=300000 | 17147.2 ms | 57.2 us |

Linear in N (2000-step vs full-N agree to 0.2%). Force assembly
differential: ~427 us/step; loop/update overhead: ~57 us/step.

## Roofline

- FLOP/step (host-counted FP64): 2 Verlet assemblies x 384 tris x
  ~350 FLOP (CST ~100 + DKT ~140 + frame maps ~110) + update ~17 kFLOP
  = **~286 kFLOP/step**.
- Bytes/step (cold DRAM): tri constants ~178 kB + u/m/f ~33 kB
  = **~200 kB/step** => ~1.4 FLOP/byte (memory-bound on paper).
- Achieved: 0.59 GFLOP/s (0.1% of the 543.6 FP64 ceiling),
  0.41 GB/s (0.02% of the 2161 GB/s ceiling).
- The kernel sits nowhere near either roof: 1 block x 256 threads on
  188 SMs (0.5% occupancy), `__syncthreads` x 14/step, and
  `tri_force_one` stack arrays spilling to local memory. It is
  latency/occupancy-bound — the deck (1386 DOFs) cannot fill the GPU.
- CPU native (same physics, 1 thread): 29.05 s / 297827 steps =
  97.5 us/step. **The proxy is 5.0x SLOWER than one CPU thread.**

## G0 gate verdict: FAIL — no G1 port

Even a 10x kernel optimization (register-resident unrolling,
multi-block colors) reaches only ~2x one CPU thread, and the deck
has no more parallelism to give (384 tris). A credible G1 needs
multi-deck batching or larger meshes (OHT at 3900 quads is the first
candidate worth modeling), not a port of this deck. Unit 4 ships no
G1 for the twisted beam. (`ncu` SpeedOfLight was not attempted: this
box needs root `NVreg_RestrictProfilingToAdminHost=0`; device-timed
walls stand per the script's own fallback.)
