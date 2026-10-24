# Sonium

Sonium is a Rust workspace for multi-physics simulation: boundary-element (BEM) and finite-element (FEM)
solvers, simulation applications, and the numerical infrastructure underneath them (vectorised math, FFTs,
mesh generation, iterative/direct solvers, WASM deployment). It can also do shape and tolopology optimisation.

- List of supported physics is in [PHYSICS.md](crates/sonium-fem/PHYSICS.md).
- License: GPL V3, as declared in `Cargo.toml`.
- Repository: <https://github.com/pierreaubert/sonium>
- Website: <https://sonium.spinorama.org>

Who is Sonium for or not? Sonium
- is easy to install (no dependencies except CUDA driver) and fully open-source.
- reports performance by workload and platform in the [benchmark records](perf/README.md).
- is accurate (extensive QA with automatic comparisong to many solvers and or analytical solutions, see QA).
- is design to make it easy to add a new physics module.
- works well with agents.

## This repo vs the full checkout

This repository is the **open-source core**. It is developed inside a larger
local checkout (`all_of_sonium`) that also holds closed-source / local-only
companions which are **not** part of this repo: some apps and for now all the ML
parts with PINN, surrogate etc.

### Dependency layers

```
Layer 0 — math primitives : fastmath
Layer 1 — foundations     : wave, xem, solvers (+ family), mesher
Layer 1.5 — transforms    : fft, audio
Layer 2 — PDE solvers     : bem, fem, coupling, meshlang
Layer 3 — applications    : sim-*, roomeq, wasm
Side    — qa, bench       : validation and performance harnesses
```

### Math, transforms, backends

| Crate | What it is |
|---|---|
| `sonium-fastmath` | Vectorised sincos f32/f64 (NEON, AVX2, AVX-512, WASM SIMD, scalar fallback) |
| `sonium-wave` | Analytical wave/Helmholtz solutions (plane waves, scattering, Mie) — the test oracle |
| `sonium-xem` | Shared BEM/FEM room-acoustics types, room geometry, JSON/SOFA I/O |
| `sonium-compute` | Compute backend abstraction (CPU / CUDA, Metal+Vulkan planned) with device-memory quotas |
| `sonium-fft` | Batched complex/real FFT and STFT over `sonium-compute`, plus a rustfft 6.x adapter |
| `sonium-audio` | Audio DSP: IIR/FIR filters, RIR analysis, analog coloration, differentiable DSP |

### Solvers

| Crate | What it is |
|---|---|
| `sonium-solvers` | Facade: high-level solve, mixed-precision and differentiable APIs |
| `sonium-solvers-core` | Traits (`ComplexField`, `LinearOperator`, `Preconditioner`) + dense types |
| `sonium-solvers-sys` | Runtime detection, NUMA topology, lane executor |
| `sonium-solvers-sparse` | Sparse formats (CSR, BSR, triplet) and SpMV plans |
| `sonium-solvers-eigen` | Eigensolvers (LOBPCG) and adjoint sensitivities |
| `sonium-solvers-direct` | Dense/sparse LU/Cholesky, multifrontal BLR |
| `sonium-solvers-preconditioners` | Jacobi, ILU/ILUT/HILU, AMG, Schwarz, Chebyshev, saddle-point |
| `sonium-solvers-iterative` | Krylov methods: GMRES family, CG, BiCGSTAB, IDR(s), MINRES, TFQMR |
| `sonium-solvers-optimization` | BFGS/L-BFGS(-B), Nelder-Mead, SQP/SLSQP, MMA, IPM, CMA-ES, BOBYQA |
| `sonium-solvers-nonlinear` | Newton-Raphson variants, Anderson acceleration |
| `sonium-solvers-time-integration` | Newmark-beta, generalized-alpha, explicit schemes, adjoint checkpointing |

All solvers are generic over f32/f64/Complex32/Complex64 via `ComplexField`.

### PDE solvers and meshes

| Crate | What it is |
|---|---|
| `sonium-bem` | BEM acoustic scattering: dense/FMM assembly, Burton-Miller, Duffy singular integration, sweeps |
| `sonium-fem` | Multi-physics FEM: generic assembly, H1/H(curl)/H(div), pluggable physics modules (see `crates/sonium-fem/AGENTS.md`) |
| `sonium-mesher` | Mesh generation, tetrahedralization, partitioning, 3D convex hull |
| `sonium-coupling` | Interior/exterior (FEM/BEM) coupling: trace operators, interface power ledgers, convention contracts |
| meshlang (`-ast`, `-parser`, `-compiler`) | MeshLang geometry language: AST → Gmsh `.geo` / Python API |

### Simulation applications

| Crate | What it is |
|---|---|
| `sonium-sim-room` | Room measurement-data generation for RoomEQ testing (FEM + optional dense BLR TBEM at low frequencies) |
| `sonium-sim-speaker` | Loudspeaker BEM simulation, mesh generation, CEA-2034 output, optimisation |
| `sonium-sim-speaker-modeling` | Electro-vibroacoustic driver/enclosure models, incl. nonlinear port flow |
| `sonium-sim-pcfan` | PC-fan aeroacoustic optimisation workflow |
| `sonium-sim-propellers` | Shrouded / toroidal / serrated propeller aeroacoustics |
| `sonium-sim-structural-mechanics` | Structural-mechanics bracket examples on FEM |
| `sonium-roomeq` | RoomEQ configuration types and CEA-2034 spinorama scoring |
| `sonium-demo-workers` | Release-mode solver workers for the native demos (non-default member) |
| `sonium-wasm` | Room acoustic simulator for browsers (non-default, WASM-only) |

### QA and bench

| Crate | What it is |
|---|---|
| `sonium-qa` | Standalone accuracy-validation binaries (`qa-bem-*`, `qa-fem-*`, `qa-solvers-*`, …) with Wolfram-derived goldens |
| `sonium-bench` | Vendor-comparison and performance harnesses |

## Requirements

- Rust **stable** via rustup (`rust-toolchain.toml` pins the components:
  cargo, clippy, rustfmt, rust-analyzer, rust-src, …).
- [`just`](https://github.com/casey/just) — task runner for the repo's
  recipes. The `Justfile` prefers the `mbx` cargo wrapper when installed
  and falls back to plain `cargo`.
- Python 3 — helper/analysis scripts under `scripts/` and QA tooling.
- Optional: NVIDIA CUDA (loaded dynamically via `cudarc`, so the workspace
  compiles without a toolkit); `ffmpeg`/`blender`/pyvista only for the
  website-gallery pipeline (local checkout).

## Quickstart

```bash
git clone https://github.com/pierreaubert/sonium.git
cd sonium

cargo check          # fast typecheck, default members
cargo clippy         # must be warning-free before merging
cargo test -p sonium-solvers
```

With `just` (see `just --list` for the ~80 recipes):

```bash
just check           # cargo check (defaults)
just check-all       # cargo check --workspace
just clippy
just test            # release tests, cli features
just test-all        # full workspace, all targets
just ntest           # same via cargo-nextest
```

Run the simulators:

```bash
just run-speaker     # sonium-sim-speaker BEM sim + CEA-2034
just run-room        # generate-roomeq-data helper
just run-pcfan       # pcfan-opt optimiser
```

Handy binaries: `numcalc` and `output2sofa` (BEM), `wav2csv` (audio),
`qa-bem-*` / `qa-fem-*` / `qa-solvers-*` (accuracy gates).

Build notes: the `dev` profile already sets `opt-level = 2` (numerical
kernels are 5–50× slower at `opt-level = 0`, at the cost of incremental
compilation — use `cargo check` for iteration); `release`/`bench` use
`opt-level = 3` with thin LTO, and a `profiling` profile gives
release + line tables without LTO for perf work.

## Testing and QA

- Unit/integration tests live per crate and validate against analytical
  references from `sonium-wave` (BEM scattering, FEM physics modules,
  solver families over all four scalar types).
- Keep default `cargo test` paths unit-scale; large scaling cases belong in
  benches or QA binaries.
- `just validate-simd` runs the cross-architecture SIMD parity checks for
  `fastmath`/`compute` (x86_64 + native) — required before merging any
  change to SIMD kernels or range-reduction code.
- The `[qa]` recipe group (~50 recipes) drives manifest-based accuracy
  gates, Wolfram cross-checks, golden comparisons, and release gates;
  start with `just qa-smoke`.

## Feature flags

Rayon parallelism is always on. Optional features:

| Flag | Effect |
|---|---|
| `cli` | Command-line binary support (BEM, FEM, `sonium-benchmark`) |
| `mshio` | Gmsh `.msh` I/O (FEM, forwarded to `sonium-mesher`) |
| `autograd` | Autodiff for topology optimisation (FEM) |
| `meshlang` | MeshLang geometry compiler (`sonium-mesher`) |

## Performance philosophy

1. **Right algorithm first** — e.g. Helmholtz FEM wants FGMRES with a
   shifted-Laplacian/AMG preconditioner; dense BEM under ~1000 unknowns
   wants direct LU; SPD systems want CG + AMG. Wrong algorithm, no answer.
2. **Parallelism for large problems** — rayon assembly/SpMV above ~1000
   DOFs; serial below.
3. **Memory awareness** — sparse CSR by default, FMM/H-matrix compression
   for large BEM, mixed precision where applicable. For CUDA, budget
   against free VRAM, never host RAM.

Details, per-platform records, and methodology: `perf/PERF.md` and `perf/`. You
can visualise the same data at https://sonium.spinorama.org/perf/

## Key APIs

- `sonium-solvers-core` (re-exported by `sonium-solvers`): `ComplexField`,
  `LinearOperator<T>` (with the multi-vector `apply_block` contract),
  `Preconditioner<T>` (with adjoint-capable transpose/Hermitian applies).
- `sonium-fem`: `BilinearForm<T>` / `LinearForm<T>` (implemented per physics module),
  `FunctionSpace`, `CooOperator`. New physics = implement the form traits;
  assembly, BCs, solvers, and multigrid stay untouched.

## Docs index

| File | Content |
|---|---|
| `AGENTS.md` | Contributor/agent guide: commands, architecture, traits, testing |
| `crates/sonium-fem/AGENTS.md` | FEM physics-module catalogue and layer rules |
| `REFERENCES.md` | Literature and standards index (ISO, IEC, AES69, …) backing the code |
| `perf/PERF.md` + `perf/` | Performance methodology and per-platform records |
| `gotchas.md` | Working agreements (pre-production, CUDA budgets, test scale) |
| `Justfile` | Executable docs — `just --list`, grouped `test`/`qa`/`bench`/`run`/`perf` |
