# sonium-sim-jlr

Replication of Melchior, Weibel, Reviol & von Harbou, "Validation of
single-phase fluid dynamics in a Lattice Boltzmann simulated JLR",
Int. J. Heat Fluid Flow 121 (2026) 110519
(DOI 10.1016/j.ijheatfluidflow.2026.110519), as a native Sonium simulation.
Plan: `reviews/jlr-20261015.md` in the meta-workspace.

## What it does

Simulates the RPTU laboratory jet loop reactor (Ø165 mm × 1640 mm,
draft tube 100 mm ID × 1200 mm, top nozzle 36→12 mm, baffle plate
Ø145 mm) in single-phase water at the paper's three specific power
inputs (2.243 / 4.038 / 5.833 kW/m³) with a D3Q19 lattice Boltzmann
solver: BGK collision, Smagorinsky LES (Cs = 0.1), halfway bounce-back
walls, equilibrium plug inlet with 1 s ramp, zero-gradient outlet,
and a two-level refined patch (one-way macroscopic nesting) for the
draft-tube inlet region. Outputs the annular-gap mean velocity profile
at the measuring height, the annular volume flow, the circulation
number Ncirc, and the characteristic Reynolds numbers, compared
against the paper's Tables 4.1 / 4.2 / A.4 (`fixtures/`, see
`fixtures/PROVENANCE.md`).

## Layout

- `src/lbm.rs` — D3Q19 core (CPU, f64, rayon), Guo forcing, moving walls.
- `src/cuda.rs` — CUDA f32 kernel (NVRTC via cudarc), `--features cuda`.
- `src/geometry.rs` — analytic voxelised reactor from paper Table A.1.
- `src/refinement.rs` — two-level one-way nested refinement driver.
- `src/case.rs` — operating points, run loop, observable extraction.
- `src/compare.rs` — replication gates vs the paper's published values.
- `src/bin/jlr_run.rs` — case runner writing a JSON bundle.
- `src/bin/jlr_bench.rs` — MLUPS benchmark (CPU / CUDA).
- `tests/verification.rs` — Poiseuille, Taylor–Green, cavity, annulus
  area, refinement continuity (independent of the JLR case).
- `tests/cuda_parity.rs` — CPU↔GPU channel parity (cuda feature).

## Run

```sh
# coarse shakedown (CPU)
cargo run --release -p sonium-sim-jlr --bin jlr-run -- \
  --point 1 --dx-mm 4.5 --dt-s 3.6e-5 --total-time-s 2.0 \
  --averaging-start-s 1.0 --out shakedown.json

# production configuration (CUDA, uniform level-I grid)
cargo run --release -p sonium-sim-jlr --features cuda --bin jlr-run -- \
  --cuda --point 1 --out jlr_phi4.json

# re-check a bundle against the paper fixtures
cargo run --release -p sonium-qa --features qa-binaries --bin qa-jlr-lbm -- jlr_phi4.json
```

## Measured performance (this workspace's RTX PRO 6000 Blackwell)

- CUDA f32, full level-I grid (112×112×1059, 13.28M cells, τ = 0.500005,
  LES + central-moment collision): 594 MLUPS → a 20 s case (5.0M steps)
  projects to ≈31 h, vs the paper's <54 h on an RTX 4090 (their grid
  also carries the refined level-II patch; ours is uniform — see
  deviations). The plain BGK kernel measures 1038 MLUPS on the same
  grid; the cascaded operator costs the difference.
- CPU f64 rayon at Δx = 3 mm: ≈113 MLUPS (coarse shakedowns only).

## Stability finding (measured, 2026-10-16)

The first solver versions blew up in every configuration at the
paper's τ = 0.500005 — coarse or production grid, SRT or regularised
BGK, equilibrium or extrapolated inlet, body force or jet. The root
cause was not the collision model: the D3Q19 opposite-direction table
did not match the diagonal ordering of the velocity arrays, so every
diagonal bounce-back reflected populations into the wrong direction
(mass leaked at walls from the first wall contact; a closed-box
conservation test isolated it). `tests/verification.rs` now pins the
table's self-consistency (opposites, involution, weight isotropy),
and `tests/mass_conservation.rs` pins closed-box and inlet/outlet
mass balance. With the table fixed, the central-moment (cascaded)
collision with Smagorinsky LES runs the coarse JLR at the paper's τ
for 60000+ steps with physical jet velocities and <2% density drift,
and is the production collision; BGK and regularised BGK remain
available (`--no-central-moment`, `--no-regularized`) for comparison.
M-Star does not publish its collision model, so the cascaded operator
is also the closest documented relative of what the authors ran.

## First validation result (2026-10-16, Δx = 3 mm, φ = 4.038)

A full 8 s run at Δx = 3 mm (Δt = 1.6e-5, average 4–8 s, 557 MLUPS,
bundle in the workspace scratch `tmp/jlr/runs/dx3_phi4.json`) is stable
and mass-conserving, but the circulation is far from developed:
Ncirc = 0.098 vs the paper's 3.18, annular velocities ≈0.03–0.07 m/s
vs ≈0.5 measured. `qa-jlr-lbm` correctly fails this bundle. The
interpretation is under-resolved jet entrainment at 3 mm (nozzle =
4 cells) plus a short spin-up (8 s ≈ 1–2 loop turnovers); it is not a
solver-stability or mass-balance defect (both pinned by tests). The
production-resolution run (Δx = 1.5 mm, 20 s, φ = 4.038) is the
decisive comparison and is launched from this crate's `jlr-run`
binary; expect ≈31 h on the workspace Blackwell at the measured
594 MLUPS. Treat any circulation number quoted below production
resolution as a lower bound, not a replication result.

## Deviations from the paper's setup (all deliberate, all documented)

- Collision: central-moment (cascaded) with cumulant fourth-order
  targets instead of the paper's stated BGK (above).
- M-Star's inlet/outlet realisation and refinement interface are
  undocumented (the authors state this for the boundaries). We use an
  equilibrium plug inlet / zero-gradient outlet and one-way macroscopic
  nesting for refinement (a two-way equilibrium overwrite was tested
  and acts as a momentum plug — see `src/refinement.rs`).
- CUDA production runs use the uniform level-I grid; the refined patch
  is available on the CPU path and through `RefinedSolver`.
- Dimensions the paper never prints (support struts, nozzle cone
  details, outlet patch) are configurable interpretations, marked
  `derived` in `fixtures/geometry_table_a1.csv`.
- BGK halfway bounce-back carries a τ-dependent effective wall shift:
  measured Poiseuille centreline deviation 1.0% at H = 48 (test gate
  1.5%); Taylor–Green viscosity recovery is the tighter check.
