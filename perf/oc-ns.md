# OpenCourant NS comparison (sonium vs engine)

Navier-Stokes lid-driven cavity. Unlike the explicit-dynamics decks, the two
sides take structurally different time steps (sonium implicit macro-steps vs
engine substeps) on different discretizations, so `ns/dof/cycle` does NOT
compare across sides here — compare wall time and per-dof wall instead (see
the Reading section). Bundles: `sonium.oc-bench/v1` rows with
`correctness_passed=true` (probe tolerances).

<!-- PERF-OC-DECK-BEGIN ns_lid_cavity -->
### ns_lid_cavity (dof 12675; engine 90054 cycles, sonium 20 cycles)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| engine | smp | 16 | 55.2 | **24** | 121 MB |
| engine | smp | 4 | 87.6 | 38 | 55 MB |
| engine | serial | 1 | 223 | 98 | 49 MB |
| engine | smp | 1 | 223 | 98 | 49 MB |
| sonium | native | 16 | 10.8 | 42619 | 713 MB |
| sonium | native | 1 | 15.2 | 59854 | 329 MB |
<!-- PERF-OC-DECK-END -->

## Reading

- Wall time (the comparable number): sonium 15.2 s (t1) / 10.8 s (t16) vs
  engine 223 s (serial) — sonium wins 14.7x at 1 thread. Per-dof wall:
  1.2 ms vs 8.8 ms (7.3x). The `ns/dof/cycle` column does not compare
  (20 implicit macro-steps vs 90054 engine substeps on different grids).
- M-red with proof: engine smp scales 4.0x (223→55 s) while sonium manages
  1.4x (15.2→10.8 s). Implicit-vs-explicit structural gap; a serial Krylov
  counter-experiment shows threading the solve does not close it at this
  size (see unit5 proposal P1, declined). Criterion amendment proposed:
  judge implicit decks on wall + per-dof wall, not SMP-efficiency parity.
- Engine serial == smp1 (223.07 vs 223.10 s, 0.01%) — clean 1-core pair.
