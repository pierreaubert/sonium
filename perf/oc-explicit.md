# OpenCourant explicit-dynamics comparison (Phase 6, 2026-10-11; plastic extension, 2026-10-12)

Sonium native explicit vs the OpenCourant engine on the 6-deck
subset (4 elastic + 2 plastic), measured with `qa-oc-bench`
(`sonium.oc-bench/v1` bundles). Raw evidence: `perf/oc/*.json`
(49 cells). Every published row has `correctness_passed=true`;
any failing row is unpublished by policy (none failed in the
final matrix).

## Fingerprint

- Box: AMD Ryzen Threadripper PRO 3995WX 64-core, 128 threads,
  251 GB RAM, Linux x86_64. No CUDA explicit arm exists in
  sonium (`sonium-fem/cuda` only enables compute BLAS), so the
  sonium flavor is `native` only; CUDA is N/A, not measured.
- Sonium: release build (`cargo build --release -p sonium-qa
  --features qa-binaries --bin qa-oc-bench`), this worktree.
- Engine: `~/src_3rdparties/OpenCourant/exec`
  (`engine_linux64_gf` serial/SMP via `-nt`, `..._ompi` via
  `mpirun -np`); starter always serial, excluded from `wall_s`.
- Protocol: 3 timed repeats + 1 warmup, median wall wins —
  except `hertz_sphere` (1 repeat, 0 warmup: the sonium arm
  needs ~360 s/repeat, so full protocol would cost ~1 h/cell;
  the reduced protocol is recorded in each hertz bundle's
  `protocol` field). All runs sequential on an otherwise quiet
  box (a first pass under load avg 47-77 was discarded and
  re-run quiet; quiet taylor sonium t1 = 0.027 s vs 0.034 s
  loaded).
- Regen: `just perf-oc` runs the full matrix
  (`scripts/perf_oc_matrix.sh`, serial by design: engine runs
  race under concurrent load) and republishes these tables
  (`scripts/perf_oc_publish.py`); `perf-oc-elastic`,
  `perf-oc-plastic`, and `perf-oc-deck <deck>` run subsets.
  Tables live between `PERF-OC-DECK` markers and sort by
  ns/dof/cycle; prose around them is hand-maintained.

## Equivalence basis (from 6A, frozen in PHASE6_CONTRACTS.md)

Same checked-in deck pair both arms (dof 675/297/297/1455
exact on elastic, 2340/1872 on plastic), same material cards
and stop time, same walls/BCs/loads. Differs: engine LAW1-QEPH
vs native Neo-Hookean Q1 bricks (small-strain identical),
engine LAW2/LAW36 vs native `hex_plastic_forces` (Johnson-Cook
/ tabulated yield, same E/nu/rho), engine contact vs native
exact-multiplier rigid walls (zero contact stiffness, no
contact CFL limit; quasi-static decks keep an explicit penalty
cap on the plastic arm only), own adaptive dt per side (cycle
counts differ; normalized by ns/dof/cycle). Native plastic runs with linear
bulk viscosity off (c1 = 0): the compression-only kink
chatters under velocity-Verlet on quasi-static ramps
(55%/step growth from roundoff); the C1-smooth quadratic term
stays on for shocks. Probe sampling mirrors each deck's
engine leg (taylor_plastic FNZ plateau mean, thick-cylinder
transient DX + residual means).

Wave-4 additions (2026-10-15; 6A-record pending): bending decks
(cantilever, beam_3d_cantilever, bending_3pt) run a condensed
linear EAS stiffness (locking-free; full Q1 captures only ~39%
of tip deflection) at second-order error — probe-verified to
0.2–2.4% on all three. beam_torsion/beam_warping run the same
precomputed-stiffness linear path as hertz (0.1% strains,
probe-verified). Shell decks run a new native DKT12/BCIZ +
linear-triangle membrane arm with exact-mapped curved geometry
and velocity-projected walls. Foam decks compare engine
full-dynamics against a native material-point replay of the
same mesh/cards/path/stop (work-conjugate probes agree to
4e-6; own stepping per side, replay-vs-dynamics caveat).
Still deferred with measured reasons: axisym (no native
axisymmetric arm — engine rows only), beam_curved_arch (native
fails probes: EAS cures shear but curved-membrane locking
remains, 12% off — engine row only), crash_box_original (no C3
trio), foam_guided_rods (no dynamic engine leg).

## Results

`ns/dof/cycle = wall_s * 1e9 / (cycles * dof)`. Best in each
deck column is bold.

<!-- PERF-OC-DECK-BEGIN taylor_bar -->
### taylor_bar (dof 675; engine 24 cycles, sonium 21 cycles)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| sonium | native | 64 | 0.00291 | **205** | 4 MB |
| sonium | native | 1 | 0.00306 | 216 | 4 MB |
| sonium | native | 16 | 0.00308 | 217 | 4 MB |
| engine | smp | 1 | 0.0336 | 2071 | 39 MB |
| engine | serial | 1 | 0.0342 | 2113 | 39 MB |
| engine | smp | 4 | 0.0453 | 2795 | 44 MB |
| engine | smp | 16 | 0.0852 | 5258 | 108 MB |
| engine | mpi | 2 | 0.625 | 38558 | 72 MB |
| engine | mpi | 4 | 0.667 | 41181 | 69 MB |
<!-- PERF-OC-DECK-END -->

Engine numbers are setup-dominated (24 cycles in 33 ms; the
~40 ms in-binary setup confound documented by 6A). MPI pays
~0.7 s of launcher/rank overhead for 24 cycles.

<!-- PERF-OC-DECK-BEGIN rail_crush -->
### rail_crush (dof 297; engine 6898 cycles, sonium 6528 cycles)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| sonium | native | 16 | 0.289 | **149** | 5 MB |
| sonium | native | 1 | 0.303 | 156 | 5 MB |
| engine | smp | 1 | 0.546 | 266 | 39 MB |
| engine | serial | 1 | 0.55 | 269 | 39 MB |
| engine | smp | 4 | 1.01 | 495 | 44 MB |
| engine | mpi | 2 | 1.18 | 576 | 71 MB |
| engine | mpi | 4 | 1.24 | 604 | 69 MB |
| engine | smp | 16 | 1.83 | 894 | 109 MB |
<!-- PERF-OC-DECK-END -->

<!-- PERF-OC-DECK-BEGIN necking_bar -->
### necking_bar (dof 297; engine 1372839 cycles, sonium 1305578 cycles)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| sonium | native | 1 | 56.8 | **147** | 4 MB |
| sonium | native | 16 | 59 | 152 | 4 MB |
| engine | smp | 1 | 100 | 246 | 39 MB |
| engine | mpi | 4 | 104 | 255 | 69 MB |
| engine | mpi | 2 | 110 | 269 | 71 MB |
| engine | serial | 1 | 111 | 272 | 39 MB |
| engine | smp | 4 | 187 | 459 | 44 MB |
| engine | smp | 16 | 321 | 788 | 108 MB |
<!-- PERF-OC-DECK-END -->

Necking serial is a 2026-10-15 same-protocol rerun (103/107/111/113 s
per repeat, median 111 s): the 2026-10-13 serial cell at 164 s ran an
identical command to smp1 yet came out 63% slower, so it was discarded
as box-contaminated. Serial vs smp1 are now 10% apart across days
(same-day smp1 rerun pending a quiet box).

<!-- PERF-OC-DECK-BEGIN hertz_sphere -->
### hertz_sphere (dof 1455; engine 137513 cycles, sonium 105056 cycles; 1 repeat)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| sonium | native | 16 | 9.11 | **60** | 13 MB |
| sonium | native | 1 | 9.2 | 60 | 13 MB |
| engine | mpi | 4 | 16.6 | 83 | 69 MB |
| engine | mpi | 2 | 21.3 | 106 | 72 MB |
| engine | serial | 1 | 25.6 | 128 | 39 MB |
| engine | smp | 1 | 25.6 | 128 | 39 MB |
| engine | smp | 4 | 27.9 | 140 | 44 MB |
| engine | smp | 16 | 43.3 | 216 | 107 MB |
<!-- PERF-OC-DECK-END -->

<!-- PERF-OC-DECK-BEGIN plasticity_thick_cylinder -->
### plasticity_thick_cylinder (dof 1872; engine 5188 cycles, sonium 4704 cycles)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| sonium | native | 16 | 1.02 | **116** | 5 MB |
| sonium | native | 1 | 1.03 | 117 | 5 MB |
| engine | smp | 1 | 1.19 | 123 | 39 MB |
| engine | serial | 1 | 1.2 | 123 | 39 MB |
| engine | smp | 4 | 1.26 | 130 | 47 MB |
| engine | mpi | 4 | 1.36 | 140 | 71 MB |
| engine | mpi | 2 | 1.48 | 153 | 72 MB |
| engine | smp | 16 | 1.83 | 189 | 112 MB |
<!-- PERF-OC-DECK-END -->

<!-- PERF-OC-DECK-BEGIN taylor_plastic -->
### taylor_plastic (dof 2340; engine 53 cycles, sonium 123 cycles)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| sonium | native | 16 | 0.0859 | **298** | 5 MB |
| sonium | native | 1 | 0.0863 | 300 | 5 MB |
| engine | serial | 1 | 0.0539 | 434 | 40 MB |
| engine | smp | 1 | 0.0545 | 439 | 40 MB |
| engine | smp | 4 | 0.0572 | 461 | 47 MB |
| engine | smp | 16 | 0.0995 | 802 | 111 MB |
| engine | mpi | 2 | 0.625 | 5041 | 72 MB |
| engine | mpi | 4 | 0.673 | 5425 | 69 MB |
<!-- PERF-OC-DECK-END -->

<!-- PERF-OC-DECK-BEGIN axisym_cylinder -->
### axisym_cylinder (dof 297; engine 14778 cycles, sonium 0 cycles)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| engine | serial | 1 | 0.969 | **221** | 38 MB |
| engine | smp | 1 | 0.977 | 223 | 38 MB |
| engine | mpi | 2 | 1.69 | 385 | 71 MB |
| engine | mpi | 4 | 1.73 | 395 | 69 MB |
| engine | smp | 4 | 2.06 | 469 | 43 MB |
| engine | smp | 16 | 3.64 | 830 | 107 MB |
<!-- PERF-OC-DECK-END -->

Axisym publishes engine rows only (no native axisymmetric arm —
proposal open). Engine serial == smp1 to 1%.

<!-- PERF-OC-DECK-BEGIN beam_3d_cantilever -->
### beam_3d_cantilever (dof 297; engine 274500 cycles, sonium 261116 cycles)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| sonium | native | 16 | 2.94 | **38** | 5 MB |
| sonium | native | 1 | 2.95 | 38 | 5 MB |
| engine | serial | 1 | 20.1 | 246 | 39 MB |
| engine | smp | 1 | 20.1 | 246 | 39 MB |
| engine | mpi | 4 | 21.4 | 263 | 69 MB |
| engine | mpi | 2 | 22.4 | 275 | 71 MB |
| engine | smp | 4 | 37.5 | 460 | 43 MB |
| engine | smp | 16 | 64.3 | 788 | 108 MB |
<!-- PERF-OC-DECK-END -->

<!-- PERF-OC-DECK-BEGIN beam_curved_arch -->
### beam_curved_arch (dof 567; engine 278184 cycles, sonium 0 cycles; 1 repeat)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| engine | serial | 1 | 24.1 | **153** | 38 MB |
<!-- PERF-OC-DECK-END -->

Arch publishes the engine row only: native fails probes (curved-membrane
locking; EAS cures shear, 50%→12% off, still outside tol — see basis).

<!-- PERF-OC-DECK-BEGIN beam_torsion -->
### beam_torsion (dof 3675; engine 868758 cycles, sonium 826377 cycles)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| engine | mpi | 4 | 148 | **46** | 69 MB |
| sonium | native | 1 | 198 | 65 | 10 MB |
| sonium | native | 16 | 198 | 65 | 10 MB |
| engine | mpi | 2 | 229 | 72 | 72 MB |
| engine | smp | 4 | 249 | 78 | 45 MB |
| engine | smp | 16 | 288 | 90 | 110 MB |
| engine | smp | 1 | 361 | 113 | 41 MB |
| engine | serial | 1 | 372 | 117 | 41 MB |
<!-- PERF-OC-DECK-END -->

Torsion M-red: engine smp16 1.25x (MPI r4 2.5x) vs sonium 1.00x
(serial LAW1 path) — same threading proposal as warping. S wins
1.8x on the precomputed-stiffness path.

<!-- PERF-OC-DECK-BEGIN beam_warping -->
### beam_warping (dof 8091; engine 686616 cycles, sonium 652789 cycles)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| engine | mpi | 4 | 178 | **32** | 70 MB |
| engine | smp | 16 | 229 | 41 | 111 MB |
| engine | smp | 4 | 286 | 51 | 47 MB |
| engine | mpi | 2 | 301 | 54 | 73 MB |
| sonium | native | 1 | 300 | 57 | 15 MB |
| sonium | native | 16 | 302 | 57 | 15 MB |
| engine | smp | 1 | 554 | 100 | 42 MB |
| engine | serial | 1 | 577 | 104 | 43 MB |
<!-- PERF-OC-DECK-END -->

Warping note: both 1-core engine cells ramp ~25% across repeats
(519→663 s serial; systematic, not a burst — the 3.2x MPI ladder
is coherent, medians stand). M-red: engine smp16 2.41x vs
sonium 1.00x (serial LAW1 path).

<!-- PERF-OC-DECK-BEGIN bending_3pt -->
### bending_3pt (dof 567; engine 274486 cycles, sonium 261116 cycles)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| sonium | native | 1 | 5.8 | **39** | 5 MB |
| sonium | native | 16 | 5.83 | 39 | 5 MB |
| engine | mpi | 4 | 22.8 | 146 | 69 MB |
| engine | smp | 1 | 24.1 | 155 | 39 MB |
| engine | serial | 1 | 24.2 | 156 | 39 MB |
| engine | mpi | 2 | 25.3 | 162 | 72 MB |
| engine | smp | 4 | 43.1 | 277 | 43 MB |
| engine | smp | 16 | 71 | 456 | 108 MB |
<!-- PERF-OC-DECK-END -->

<!-- PERF-OC-DECK-BEGIN cantilever -->
### cantilever (dof 297; engine 274487 cycles, sonium 261116 cycles)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| sonium | native | 1 | 2.93 | **38** | 5 MB |
| sonium | native | 16 | 2.93 | 38 | 5 MB |
| engine | smp | 1 | 20 | 245 | 39 MB |
| engine | serial | 1 | 20.1 | 246 | 39 MB |
| engine | mpi | 4 | 21.4 | 263 | 69 MB |
| engine | mpi | 2 | 22.4 | 275 | 70 MB |
| engine | smp | 4 | 36.9 | 452 | 43 MB |
| engine | smp | 16 | 63.4 | 778 | 108 MB |
<!-- PERF-OC-DECK-END -->

Bending decks run the condensed linear EAS stiffness (see basis):
cantilever 6.5x, beam_3d 6.5x, bending 4.0x over engine serial.

<!-- PERF-OC-DECK-BEGIN foam33_cube -->
### foam33_cube (dof 24; engine 5820 cycles, sonium 30720 cycles)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| sonium | native | 1 | 0.044 | **60** | 4 MB |
| sonium | native | 16 | 0.044 | 60 | 4 MB |
| engine | smp | 1 | 0.522 | 3738 | 38 MB |
| engine | serial | 1 | 0.524 | 3751 | 38 MB |
| engine | smp | 4 | 0.831 | 5950 | 49 MB |
| engine | mpi | 2 | 1.17 | 8346 | 71 MB |
| engine | mpi | 4 | 1.22 | 8766 | 69 MB |
| engine | smp | 16 | 1.42 | 10168 | 126 MB |
<!-- PERF-OC-DECK-END -->

<!-- PERF-OC-DECK-BEGIN foam76_compression -->
### foam76_compression (dof 24; engine 5683 cycles, sonium 30720 cycles)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| sonium | native | 16 | 0.0827 | **112** | 4 MB |
| sonium | native | 1 | 0.0827 | 112 | 4 MB |
| engine | serial | 1 | 0.562 | 4119 | 39 MB |
| engine | smp | 1 | 0.567 | 4161 | 39 MB |
| engine | smp | 4 | 0.906 | 6642 | 49 MB |
| engine | mpi | 2 | 1.21 | 8874 | 71 MB |
| engine | mpi | 4 | 1.26 | 9242 | 69 MB |
| engine | smp | 16 | 1.45 | 10624 | 126 MB |
<!-- PERF-OC-DECK-END -->

<!-- PERF-OC-DECK-BEGIN foam76_tension -->
### foam76_tension (dof 24; engine 5671 cycles, sonium 30720 cycles)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| sonium | native | 1 | 0.066 | **90** | 4 MB |
| sonium | native | 16 | 0.0663 | 90 | 4 MB |
| engine | smp | 1 | 0.562 | 4126 | 39 MB |
| engine | serial | 1 | 0.57 | 4188 | 39 MB |
| engine | smp | 4 | 0.885 | 6499 | 49 MB |
| engine | mpi | 2 | 1.21 | 8921 | 71 MB |
| engine | mpi | 4 | 1.27 | 9344 | 69 MB |
| engine | smp | 16 | 1.44 | 10596 | 126 MB |
<!-- PERF-OC-DECK-END -->

Foam caveat: sonium rows are a material-point replay of the deck
cards (30720 substeps), engine rows full dynamics — same mesh,
cards, path, stop; work-conjugate probes agree to 4e-6. Read as
constitutive-update cost at deck scale, not like-for-like
dynamics (6A-record pending).

<!-- PERF-OC-DECK-BEGIN shell_clamped_plate -->
### shell_clamped_plate (dof 771; engine 234205 cycles, sonium 903526 cycles)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| engine | mpi | 4 | 22.2 | **123** | 69 MB |
| engine | mpi | 2 | 26.8 | 148 | 72 MB |
| sonium | native | 1 | 104 | 150 | 52 MB |
| sonium | native | 16 | 104 | 150 | 52 MB |
| engine | serial | 1 | 27.8 | 154 | 38 MB |
| engine | smp | 1 | 27.8 | 154 | 38 MB |
| engine | smp | 4 | 40.3 | 223 | 44 MB |
| engine | smp | 16 | 68.2 | 378 | 106 MB |
<!-- PERF-OC-DECK-END -->

Clamped plate is the thinnest S win (1.03x); it wins vs SMP16
(2.5x) but engine MPI-best beats it (123 vs 150 ns) — native dt
bound takes 3.9x engine's cycles on thin shells.

<!-- PERF-OC-DECK-BEGIN shell_oht_elastic -->
### shell_oht_elastic (dof 12192; engine 32317 cycles, sonium 144362 cycles; 1 repeat)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| engine | serial | 1 | 31.8 | **81** | 44 MB |
| sonium | native | 1 | 305 | 173 | 38 MB |
<!-- PERF-OC-DECK-END -->

<!-- PERF-OC-DECK-BEGIN shell_scordelis_lo -->
### shell_scordelis_lo (dof 3267; engine 28808 cycles, sonium 379930 cycles; 1 repeat)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| engine | serial | 1 | 10.6 | **113** | 39 MB |
| sonium | native | 1 | 387 | 312 | 20 MB |
<!-- PERF-OC-DECK-END -->

OHT/scordelis are finding-grade quick records (S-red 0.47x/0.36x):
native shell dt bound needs 4.5x/13x engine's cycles (clamped
plate 3.9x, still S-green at 1.03x). Probes pass; mass-scaling
proposal open (physics change, 6A review).

<!-- PERF-OC-DECK-BEGIN shell_twisted_beam -->
### shell_twisted_beam (dof 693; engine 72748 cycles, sonium 297827 cycles)

| Side | Flavor | Threads | wall_s | ns/dof/cycle | peak RSS |
| ---- | ------ | ------: | -----: | -----------: | -------: |
| sonium | native | 16 | 27.4 | **133** | 19 MB |
| sonium | native | 1 | 27.9 | 135 | 19 MB |
| engine | mpi | 4 | 7.09 | 141 | 69 MB |
| engine | mpi | 2 | 8.27 | 164 | 72 MB |
| engine | smp | 1 | 8.81 | 175 | 38 MB |
| engine | serial | 1 | 8.87 | 176 | 38 MB |
| engine | smp | 4 | 12.9 | 256 | 43 MB |
| engine | smp | 16 | 22.1 | 437 | 107 MB |
<!-- PERF-OC-DECK-END -->

## Reading

- Per-dof-per-cycle, sonium wins 14 of 16 benchable decks against
  engine serial (wave-4 numbers): taylor 9.8x, rail 1.7x, necking
  1.9x, hertz 2.1x, plastic taylor 1.5x, cylinder 1.05x,
  cantilever 6.5x, beam_3d 6.5x, bending 4.0x, torsion 1.8x,
  warping 1.8x, twisted 1.3x, clamped 1.03x, foam 37–63x
  (material-point replay vs full dynamics — comparison with
  caveat, not like-for-like). Hertz flipped from a 2.6x deficit:
  the multiplier-contact refactor removed the penalty dt cap
  (786k → 105k cycles) and hertz runs a precomputed linear
  stiffness (no per-step Jacobians/determinants/logarithms),
  taking it from 320 to 60 ns/dof/cycle with the DZ probe exact
  to 7 digits. Taylor went from 127 to 21 cycles at the bulk CFL
  step. S-red: oht 0.47x and scordelis 0.36x (quick-record,
  finding-grade — native dt bound needs 3–13x more cycles than
  engine; mass-scaling proposal open).
- Wall-clock: sonium hertz takes 9.1 s vs engine 25.6 s (2.8x;
  fewer native cycles too, 105k vs 137k) and sonium taylor
  takes 0.003 s vs engine 0.034 s (11x). Plastic taylor goes
  the other way (engine 0.054 s vs sonium 0.084 s: 53 vs 123
  cycles at a cheaper sonium per-cycle cost), while the
  cylinder runs fewer native cycles than engine (4704 vs
  5188), so sonium wins wall there too (1.01 s vs 1.19 s).
  Necking: 56.8 s vs 111 s serial (1.9x; 1.31M vs 1.37M
  cycles).
- Scaling splits by deck size. On small decks sonium t1 ~= t16
  (serial force loop by design; 1.00x) while engine SMP is flat
  to sharply negative, so sonium wins on efficiency trivially.
  On the two biggest decks the engine DOES scale — torsion
  1.25x, warping 2.41x at smp16; MPI r4 reaches 2.5x/3.2x —
  while sonium holds 1.00x: M-red on both (owner-lane proposal:
  thread the LAW1 elastic explicit path; the plastic path
  already carries the rayon pattern). At 16 cores sonium wins
  15 of 16 decks per-cycle against engine SMP16 (only warping
  loses, 0.72x); engine MPI-best additionally beats sonium on
  torsion, clamped, and twisted — shown for context, outside
  the registered t16-vs-smp16 head-to-head.
- Memory: sonium peaks at 4-13 MB per cell (hertz highest;
  its precomputed linear stiffness is the largest native
  buffer). An earlier fixed 83 MB reading on every cell was a
  measurement artifact: `getrusage(RUSAGE_SELF)` keeps the
  launcher's `ru_maxrss` across `fork` + `exec`, so `cargo run`
  reported cargo's own peak; the harness now reads the
  per-process `VmHWM` from `/proc/self/status` on Linux.
  Engine 38-112 MB per cell (MPI RSS covers the launcher only
  and under-reports ranks — 6A caveat, restated).
- Engine serial == smp1 within 2% on every wave-4 deck (same-day
  pairs; the two cells run byte-identical commands, so any gap
  above ~5% is box noise, not signal). Necking is the exception
  at 10% across days (serial rerun 2026-10-15, smp1 2026-10-13).
  Warping 1-core engine cells ramp ~25% across repeats in both
  serial and smp1 (systematic, coherent with the 3.2x MPI
  ladder — medians stand, see the deck note).

## Rerun

```sh
export TMPDIR=$HOME/src/all_of_sonium/tmp \
  SONIUM_QA_SCRATCH=$HOME/src/all_of_sonium/tmp \
  OPENCOURANT_EXE=$HOME/src_3rdparties/OpenCourant/exec \
  OPENCOURANT_MINIQA=$HOME/src_3rdparties/OpenCourant/OpenCourant/qa-tests/miniqa \
  LD_LIBRARY_PATH=$HOME/src_3rdparties/OpenCourant/extlib/hm_reader/linux64 \
  RAD_CFG_PATH=$HOME/src_3rdparties/OpenCourant/hm_cfg_files
just perf-oc-deck taylor_bar   # one deck; perf-oc / -elastic / -plastic for sets
```

Run cells sequentially on a quiet box; discard any run with
sustained foreign load (compare against `perf/oc/`).
