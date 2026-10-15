# OC GPU machine profile + G0 proxy method (RTX PRO 6000 Blackwell)

Shared Phase-G reference for oc-opt units 1–5: the measured machine
ceilings every deck roofline reports against, and the G0 proxy-first
method every deck follows. Owned by Unit 5 (fluids); other units
consume it read-only.

## Machine (reference box)

- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, compute
  capability 12.0, 188 SMs, 1536 max threads/SM, 100 KB shared/SM,
  128 MB L2, 96 GB global (95.0 reported), 512-bit bus.
- Host: AMD Ryzen Threadripper PRO 3995WX 64-core, 251 GB RAM.
- CUDA 13.3 (`nvcc`/`ncu`/`nsys` present). `ncu` counters are
  root-gated on this box (`ERR_NVGPUCTRPERM`, needs
  `NVreg_RestrictProfilingToAdminHost=0`); until an admin sets it,
  device-timed walls + differential evidence are the roofline
  method, not a stopgap.

## Measured ceilings (necking_bar G0, 2026-10-15)

Source: `perf/oc/necking_bar_cuda_machine.cu`, console
`perf/oc/necking_bar_cuda_machine.txt` (device-timed, GPU idle).
Reproduce: `nvcc -O2 -arch=native -o machine
perf/oc/<deck>_cuda_machine.cu && ./machine` (seconds).

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

Derived constants other units reuse without re-measuring: one SM's
FP64 stream share is 543.6/188 = 2.89 GFLOP/s; the host-stepped
floor is ~7 us/step launch+sync plus the deck's state-roundtrip
transfer before any compute. Decks whose CPU budget is tens of
us/step must be persistent-kernel designs to compete at all.

## G0 proxy-first method (all decks)

Before porting any deck's time loop to CUDA (G1), each deck files a
G0 proxy report. Steps, in order:

1. **Machine profile.** Reuse the ceilings above; decks with an
   unlike transfer regime (state far from 4–8 KB) re-run the machine
   probe at their size and note the delta. Do not re-tune the
   ceilings per deck.
2. **Persistent-kernel proxy at exact deck topology.**
   `perf/oc/<deck>_cuda_proxy.cu`: one persistent kernel holds the
   whole step loop on-device (no per-step launch/transfer), mirrors
   the native algorithm's op mix (same element/Gauss work, same
   update, same precision), and compiles an `-DEMPTY_FORCE`
   differential that removes the force/math phase but keeps
   loop + barriers + update. The empty build splits the wall into
   floor vs math with no counters.
3. **Validate physics BEFORE timing.** Fidelity checks are measured,
   not assumed: zero-state residue, short-run quantities vs a native
   or analytic reference, bit-identical repeats, proven-linear
   short→long scaling. A proxy that runs fast on non-physical states
   (NaN, wrong sign convention) proves nothing — mirror the native
   force-assembly sign convention, run fixed dt below the native mean
   step (the native adaptive driver's shrinkage is itself a
   stabilizer), and project full-N walls from proven-linear short
   runs rather than hogging the shared GPU.
4. **Perfect-port bound.** Deck parallelism (elements + nodes +
   Gauss points) vs one SM's FP64 share bounds the best possible
   port without writing G1. If a zero-overhead port still floors
   above CPU, the deck fails the G1 gate with the bound as evidence.
5. **G1 gate.** The proxy must project within striking distance of
   CPU, else no G1 follows and the G0 report stands as that deck's
   distance-to-roofline record (necking_bar precedent: 48x short,
   no G1).

## Evidence layout (per deck)

Checked in under `perf/oc/` (parent mirrors `sonium/perf/oc/` to
root `simperf/perf/oc/`):

- `<deck>_cuda_proxy.cu` — the G0 proxy (deck unit authors it;
  required; `perf_oc_gpu.sh` fails closed without it).
- `<deck>_cuda_machine.cu` — deck machine profile (optional; else
  the shared `cuda_machine.cu` when one lands, else the script
  notes "Unit 5 profile applies" and the ceilings above govern).
- `<deck>_cuda_proxy.txt` — console archive written by
  `bash sonium/scripts/perf_oc_gpu.sh <deck> sonium/perf/oc`
  (machine, N=1 fidelity, N=steps device-timed, EMPTY
  differential, optional best-effort `ncu`).
- `<deck>_cuda_roofline.md` — the deck's distance-to-roofline
  report (deck unit authors it; the script archives it from
  `PERF_OC_GPU_REPORT` but never fabricates it).

Run the script only on an idle GPU (it refuses at util > 5% or
with foreign compute procs unless `PERF_OC_GPU_FORCE=1`, and
`--quick`-style short runs are the norm: default 2000 steps).
`ncu` capture is best-effort (`PERF_OC_GPU_NCU=1`); the
root-blocked failure mode is documented, not fatal.

## Roofline report contract (per deck)

Each `<deck>_cuda_roofline.md` opens with a verdict line (GPU
recommended or not, with the projected ratio vs CPU), then states:
machine-ceiling numbers used, proxy design (topology, launch
config, op mix, deviations), fidelity evidence (measured values
vs references), roofline numbers (per-step device wall, full-deck
projection, work/traffic estimates with uncertainty, achieved
throughput, distance to ceiling, AI vs ridge, empty-force floor
share, occupancy), the perfect-port bound, a reading (limiter,
what would change it), and a one-line reproduce command.
Reference implementation: `necking_bar_cuda_roofline.md`.
