# ns_lid_cavity CUDA roofline (G0)

Verdict: **GPU NOT recommended for a faithful port of the cavity linear
solve.** The persistent-kernel GMRES(30)+ILU(0) proxy runs one native
linear-solver iteration in 5.39 ms (device time) vs 0.65 ms native CPU
(8.3x short); the projected full-deck solve phase is 66.0 s vs 8.0 s CPU
(73.5 s vs 15.8 s with the CPU assembly included, 4.7x short). The limiter
is level-scheduled triangular latency (522 sequential wavefronts, ~48 rows
mean width, ~10 us per cooperative grid sync), not flops or bytes. The
existing CUDA-backend speedup (11.64 s vs 15.6 s, 1.34x, deck README) comes
from its algorithmic change (polynomial/Chebyshev, no triangular solve),
which this report reconciles rather than contradicts. No G1 follows for the
faithful port; this G0 stands as the deck's distance-to-roofline record.

## Machine ceilings used

Deck re-run of the shared Unit-5 probe (`ns_lid_cavity_cuda_machine.cu`,
console below): STREAM 2158 GB/s, FP64 543.8 GFLOP/s, launch+sync 6.8 us,
4 KB roundtrip 11.7 us — within 0.2% (ceilings) and ~10% (latencies) of the
canonical necking numbers (2161 GB/s, 543.6 GFLOP/s, 7.1 us, 12.6 us) in
`sonium/perf/oc-gpu-machine.md`, which govern. Box: RTX PRO 6000 Blackwell,
CC 12.0, 188 SMs, 128 MB L2. `ncu` is root-blocked here (`ERR_NVGPUCTRPERM`,
attempt documented in console run); the wall split comes from the
empty-force differential, not counters.

## Proxy design

`ns_lid_cavity_cuda_proxy.cu`: one cooperative persistent kernel holds the
whole GMRES(30) loop on-device at the exact deck topology (65x65 P1 nodes,
4225 pressure + 8450 velocity dofs, n = 12,675, nnz = 261,513). The host
builds the exact first-Picard Oseen system analytically (closed-form P1
stencils plus the native 3-point quadrature for the u-dependent terms
against the BC-imposed iterate, per-QP tau, ILU(0) with the native pivot
rule plus triangular level schedules). The device loop mirrors the native
algorithm phase by phase: SpMV, level-wavefront ILU pair, batched CGS with
DGKS reorthogonalisation, Givens, restart every 30, least-squares x-update.
`-DEMPTY_FORCE` compiles out SpMV+ILU but keeps loop, MGS, syncs and update
for the floor-vs-math split. Deviations (all documented): host-side ILU
setup (native refactors per Picard on CPU — optimistic for GPU); no Picard
outer loop (projection multiplies one-solve cost by the native 12,240 total
GMRES iterations); FP64 throughout like native.

## Fidelity evidence (measured)

- Operator: all 261,513 entries match a native MatrixMarket dump to
  <= 1.8e-17 (patterns exact, rhs to 1.7e-18). The dump diff caught two
  real modeling slips during development (BC-imposed iterate, EMPTY-mode
  NaN); both fixed and re-verified.
- N=1: device residual 2.300935e-01 vs host serial replica
  2.300935195775e-01 (7 digits).
- N=330 (native's stopping point): device true residual 8.36e-9, under the
  1e-8 native exit tolerance; centerline u and p match the native
  first-Picard state divided by the 0.7 Picard relaxation at all 9 stations
  (max |Du| = 6.3e-6, max |Dp| = 1.9e-10).
- Estimate-crossing differs (native 330, proxy 1235) while true-residual
  convergence coincides at 330: the preconditioned estimate runs ~100x
  above truth (the ||M^-1|| effect) and its floor is roundoff-dominated, so
  the crossing point is noise; the converged states agree (above).
- Determinism: bit-identical repeats in every run (identical=1), EMPTY
  included (finite zeros, no NaN).
- Linearity: 5389.4 us/iter at N=2000 vs 5416.2 us/iter at N=20000 (+0.5%);
  N=50/330/1000/2000 samples agree within 0.4%.

## Roofline numbers

Per-GMRES-iteration device wall: 5.39 ms (event-timed, steady state).
Native CPU per-iteration: 0.65 ms (8.0 s solve phase / 12,240 iters).
Ratio: 8.3x short. Full-deck projection: 12,240 x 5.39 ms = 66.0 s solve
phase (vs 8.0 s CPU), 73.5 s with the untouched 7.5 s CPU assembly (vs
15.8 s wall, 4.7x short). The projection is optimistic for GPU (ignores
Picard-loop host traffic and per-step ILU setup).

Work per iteration (closed form): SpMV 0.52 + ILU pair 0.51 + batched CGS
~1.66 + restart amortised ~0.06 = ~2.75 MFLOP. Traffic is L2-resident
(working set ~15 MB vs 128 MB L2; DRAM ~0). Achieved: 0.51 GFLOP/s, 0.09%
of the 543.6 GFLOP/s FP64 ceiling. The wall is not flops or bytes: each
iteration executes ~535 cooperative grid syncs (522 triangular wavefront
levels + ~13 MGS/reduction syncs), and both modes agree on ~10 us per sync
(FULL 5.39 ms / 535; EMPTY 143 us / ~14), so the device wall is grid-sync
latency almost exactly.

Empty-force split: EMPTY 142.6 us/iter (2.6% floor: loop + MGS + syncs +
update); SpMV+ILU math 5.25 ms/iter (97.4%), of which the SpMV is ~1.5 us
of traffic and the rest is the 522 wavefront syncs. Occupancy in the
dominant triangular phases is ~0.3% (mean wavefront width ~48 rows of
16,384 slots); SpMV/CGS phases are full-occupancy and L2-resident.

## Perfect-port bound

The ILU factors admit no parallelism beyond their wavefront DAG: 261 + 261
sequential levels at this deck's 12,675 rows. At the measured ~10 us per
cooperative sync, any level-scheduled triangular port floors at ~5.2 ms per
iteration vs 0.65 ms CPU — 8x short before any other cost. The proxy
achieves this bound (5.25 ms math share), so no implementation of the
faithful port beats it. A no-sync fantasy (2.75 MFLOP at one SM's 2.89
GFLOP/s share = 0.95 ms/iter) still shorts 0.65 ms CPU. G1 gate: FAIL.

## Reading

Limiter: triangular-solve dependencies (261+261 wavefronts at mean width
~48) serialized through ~10 us cooperative syncs. What would change it: a
preconditioner without triangular solves (polynomial/Chebyshev — the
existing CUDA backend's route to 1.34x), or a much larger deck where
wavefronts widen past the sync floor (the deck is fixed n = 64, so not
applicable). Assembly (7.5 s, 47% of the CPU wall) is outside any
solve-port's reach. Recommendation: no G1 faithful port; any future GPU
work on this deck must change the algorithm (as the existing backend
already does) rather than port GMRES+ILU.

## Reproduce

```
cd <worktree>/sonium && nvcc -O2 -arch=native -o /tmp/cav \
  perf/oc/ns_lid_cavity_cuda_proxy.cu && /tmp/cav 330      # fidelity point
bash scripts/perf_oc_gpu.sh ns_lid_cavity perf/oc 2000     # full console
```

Console: `ns_lid_cavity_cuda_proxy.txt` (machine, N=1, N=2000, EMPTY;
N=20000 linearity and ncu-block runs in the report text). All numbers
preliminary single-cell diagnostics on the shared box.
