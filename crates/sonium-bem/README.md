# sonium-bem: Boundary Element Method Library

A high-performance, memory-efficient Boundary Element Method (BEM) library in Rust for solving acoustic scattering and room acoustics problems.

## Overview

This crate provides a pure Rust BEM implementation with Fast Multipole Method (FMM) acceleration and multiple solver options. The library is designed to be:

- **Reusable**: General-purpose BEM solver for acoustic problems
- **Memory-efficient**: Careful memory management for large-scale problems
- **Parallel**: Uses Rayon for data parallelism
- **Validation-oriented**: Analytical, parity and convergence tests; coverage
  and runtime qualification vary by operator/backend combination

## Features

The optional `meshlang` feature enables MeshLang compilation and Gmsh mesh
loading through `sonium-mesher`; it is used by the end-to-end BEM smoke gate.

### Current implementation

#### Core BEM
- Pure Rust BEM implementation with Burton-Miller formulation
- Collocation method with adaptive Gaussian quadrature
- Support for triangular and quadrilateral elements
- Near-singular and singular integration handling

#### Solvers
- **Direct solver**: LU factorization for small problems (N < 1000)
- **GMRES + ILU**: Iterative solver with ILU preconditioning
- **FMM + GMRES**: Fast Multipole Method with O(N log N) complexity
- **FMM + GMRES + near-field ILU**: FMM with the checked near-field ILU path
  where the operator contract admits it
- **FMM + Batched BLAS**: Optimized FMM using batched matrix operations (recommended for large problems)
- Unsupported FMM preconditioners are rejected by checked admission; there is
  no silent identity or hierarchical fallback.

#### Room Acoustics Application
- Room geometry parsing (rectangular and L-shaped rooms)
- Multi-source acoustic simulations
- Frequency sweep with parallel processing
- SPL computation at listening positions
- Spatial field visualization (pressure slices)

#### Optimizations
- Batched BLAS operations for efficient matvec
- Pre-allocated workspaces to minimize allocations
- Rayon-based parallel assembly and solving
- Frequency-adaptive mesh refinement

### Additional implemented paths and limits

- `AssemblyMethod` exposes TBEM, H-matrix/ACA, SLFMM and MLFMM; directional
  MLFMM is experimental. Selection and measured crossover notes live in
  `src/core/bem_solver.rs` and are not universal platform timing guarantees.
- `core::coupling::CoupledTraceOperator` composes conservative acoustic/interior
  traces through `LinearOperator` without a dependency on FEM. Application
  geometry, material coupling and end-to-end validation remain separate work.
- Optional `cuda` and `metal` features enable selected native operations through
  `sonium-compute`, not an entirely GPU-resident BEM solve. Generic wgpu solver
  acceleration is not implemented by these features.

## Mathematical Background

The solver-wide sign, normal, phasor-amplitude, and transpose/Hermitian rules
are documented in [`CONVENTIONS.md`](CONVENTIONS.md). XEM-driven entry points
validate those rules before assembly; compatibility constructors use the same
canonical acoustic convention.

### Boundary Element Method

The Boundary Element Method solves boundary value problems by reformulating them as boundary integral equations. For acoustic scattering problems, we solve the Helmholtz equation:

```
nabla^2 p + k^2 p = 0
```

where:
- `p` is the acoustic pressure
- `k = 2*pi*f/c` is the wave number
- `f` is frequency, `c` is speed of sound

### Burton-Miller Formulation

We use the **Burton-Miller formulation** to avoid spurious resonances at interior eigenfrequencies:

```
integral_dOmega [G(x,y) dp/dn(y) - dG(x,y)/dn(y) p(y)] dS(y) +
i*alpha integral_dOmega [dG(x,y)/dn(x) dp/dn(y) - d^2G(x,y)/(dn(x)dn(y)) p(y)] dS(y) = p_inc(x)
```

where:
- `G(x,y) = exp(ik|x-y|) / (4*pi*|x-y|)` is the Helmholtz Green's function
- `alpha` is a coupling parameter (typically `1/k`)
- `dOmega` is the boundary surface
- `p_inc` is the incident field

### Fast Multipole Method

For large problems (N > 1000 elements), the **Single-Level Fast Multipole Method (SLFMM)** reduces complexity from O(N^2) to approximately O(N log N) by:

1. **Hierarchical decomposition**: Octree spatial subdivision
2. **Far-field approximation**: Multipole expansions for distant interactions
3. **Translation operators**: Efficient propagation of expansions

The FMM implementation includes:
- Adaptive octree construction
- T-matrices (element DOFs to multipole expansion)
- D-matrices (translation between clusters)
- S-matrices (local expansion to field DOFs)
- Batched BLAS operations for efficient matvec

## Solver Selection Guide

| System Size | Solver Method | Configuration |
|-------------|---------------|---------------|
| N < 1000 | Direct LU | `"direct"` |
| N < 5000 | GMRES + ILU | `"gmres+ilu"` |
| N < 20000 | FMM + GMRES + near-field ILU | `"fmm+gmres+ilu"` when admitted |
| N > 20000 | FMM + Batched BLAS | `"fmm+batched"` or `"fmm+gmres+batched"` |

These size ranges are historical heuristics, not per-platform benchmark
results. Use the checked near-field ILU path only when admitted; the method
name does not imply a hierarchical preconditioner. Compare time and peak memory
at the same physical error before choosing a solver.

## Validation case matrix

Independent-analytic coverage per assembly family (rigid-sphere Mie series
from `sonium-wave`, not parity against dense). All rows below run in the
default `cargo test -p sonium-bem` gate unless marked nightly.

| Frequency | Mesh / order | Formulation | Backend | Accuracy evidence |
|---|---|---|---|---|
| ka 0.25–0.75 (5-pt sweep) | default sphere discretization, constant panels | Burton–Miller | TBEM, H-matrix, SLFMM, MLFMM | scattered power vs Mie ≤ 12%; backscatter parity vs dense ≤ 5e-4 (`sphere_reciprocity_passivity::bem_sphere_frequency_sweep_matches_mie_v1`) |
| ka 0.5 | default sphere discretization, constant panels | Burton–Miller | TBEM | scattered power vs Mie ≤ 8%; field pattern vs Mie L2 ≤ 15%, pointwise ≤ 25%; reciprocity ≤ 2% (`sphere_reciprocity_passivity`) |
| ka 0.5 | default sphere discretization, constant panels | regular + Burton–Miller | H-matrix, SLFMM, MLFMM | scattered power vs Mie ≤ 8%; parity vs dense ≤ 1e-4; reciprocity ≤ 2% (`bem_sphere_backend_formulation_matrix_v1`) |
| convergence series | refined spheres / Quad4 | Burton–Miller, Galerkin–Müller, transmission | TBEM, Galerkin, higher-order | `continuous_quadratic_mie_convergence`, `galerkin_muller_mie_convergence`, `penetrable_sphere_mie_convergence` in the test gate; `continuous_quadratic_quad4_transmission_mie_convergence` in the QA tier (case `bem.transmission-continuous-quadratic-quad4-mie-convergence.v1`, ~48 min sweep) |
| production resolution | 320-element icosphere and above | Burton–Miller | SLFMM, MLFMM | nightly only (`#[ignore]` slow gates, see TESTING.md); same Mie/power checks at higher cost |

Not yet established: time/memory to equal error against an external BEM
implementation (e.g. Bempp); high-`kD` directional-MLFMM accuracy beyond the
experimental `DirectionalMlfmm` path; browser/Windows backend execution. The
size ranges in the solver-selection table remain heuristics until measured
against this matrix per platform.

## Usage

### Command-line binaries

`numcalc` is a minimal replacement for the Mesh2HRTF NumCalc CLI: it reads a
Mesh2HRTF project (`NC.inp`), runs the BEM frequency sweep, and writes
`be.out.json` with per-frequency surface and evaluation-grid pressure.
`output2sofa` converts that JSON to SOFA/HRTF output.

```bash
# Build
cargo build --release -p sonium-bem --features cli --bin numcalc

# Run a Mesh2HRTF project sweep (writes <project>/be.out.json by default)
./target/release/numcalc -- path/to/Project/NC.inp --out path/to/be.out.json

# Convert a sweep to SOFA (requires the `sofa` feature at build time)
cargo build --release -p sonium-bem --features cli,sofa --bin output2sofa
./target/release/output2sofa path/to/be.out.json --out path/to/output.sofa
```

Room simulation itself is a library API (`room_acoustics`), not a JSON-config
binary; see `examples/room_simulator_config.rs` (`rectangular`, `lshaped`,
`multi-source`) for the configurable room-simulation entry points.

### Room-simulation configuration

Room simulations are configured in code through `room_acoustics`
(`RectangularRoom`, `Source`, `FmmSolverConfig`, `solve_bem_fmm_gmres_ilu`;
see the Library API example below and `examples/room_simulator_config.rs`).
There is no JSON-config room-simulator binary in this crate. Solver-method
selection for sweeps uses the `AssemblyMethod` / `SolverMethod` builder API
on `BemSolver` (`src/core/bem_solver.rs`), not method-name strings; the
`"fmm+gmres+ilu"` style names below are shorthand used in prose and sweep
tooling.

### Solver Methods

Method shorthand used in prose (configured via the `AssemblyMethod` /
`SolverMethod` builder API on `BemSolver`, not parsed from strings):

- `"direct"`: `AssemblyMethod::Tbem` + `SolverMethod::Direct` (LU)
- `"gmres+ilu"`: dense assembly + `SolverMethod::Gmres` with ILU preconditioning
- `"fmm+gmres"`: `AssemblyMethod::Slfmm`/`Mlfmm` + `SolverMethod::Gmres`
  (no preconditioning)
- `"fmm+gmres+ilu"`: FMM assembly + GMRES with the checked near-field ILU path
- `"fmm+batched"` or `"fmm+gmres+batched"`: FMM assembly with batched BLAS matvec

### Library API

```rust
use sonium_bem::room_acoustics::{
    FmmSolverConfig, Point3D, RectangularRoom, Source, solve_bem_fmm_gmres_ilu,
};
use sonium_mesher::RoomSurfaceMeshing;

let room = RectangularRoom::new(4.0, 5.0, 2.5);
let sources = vec![Source::omnidirectional(Point3D::new(1.0, 1.0, 1.0), 1.0)];

// Solve at a specific frequency
let frequency = 200.0;
let k = 2.0 * std::f64::consts::PI * frequency / 343.0;
let mesh = room.generate_mesh(2);

let solution = solve_bem_fmm_gmres_ilu(
    &mesh,
    &sources,
    k,
    frequency,
    &FmmSolverConfig::default(),
    1000,  // max iterations
    50,    // restart
    1e-6,  // tolerance
)?;
```

### Batched BLAS Operations

For large-scale problems, use the batched BLAS solver:

```rust
use sonium_bem::core::solver::{
    gmres_solve_fmm_batched_with_ilu, GmresConfig
};
use sonium_bem::room_acoustics::build_fmm_system;

// Build FMM system
let (fmm_system, elements, nodes) = build_fmm_system(
    &mesh, &sources, k, frequency, &fmm_config
)?;

// Solve with batched operations
let gmres_config = GmresConfig {
    max_iterations: 1000,
    restart: 100,
    tolerance: 1e-6,
    print_interval: 1,
    ..GmresConfig::default()
};

let result = gmres_solve_fmm_batched_with_ilu(
    &fmm_system,
    &fmm_system.rhs,
    &gmres_config,
);

println!("Converged: {}, Iterations: {}", result.converged, result.iterations);
```

## Architecture

```
sonium-bem/
├── src/
│   ├── lib.rs                  # Public API (flat re-exports per domain)
│   ├── bin/
│   │   ├── numcalc.rs          # Mesh2HRTF-compatible CLI (needs `cli`)
│   │   └── output2sofa.rs      # SOFA/HRTF conversion (needs `cli`, `sofa`)
│   ├── core/
│   │   ├── bem_solver.rs       # High-level BemSolver/BemProblem API
│   │   ├── types.rs            # PhysicsParams, mesh, BCs, conventions
│   │   ├── coupling.rs         # CoupledTraceOperator (no FEM dependency)
│   │   ├── assembly/           # tbem, hmatrix, slfmm, mlfmm, galerkin, …
│   │   ├── solver/             # frequency_sweep, fmm_interface, batched_blas
│   │   ├── integration/        # regular/singular (Duffy), Gauss, Lebedev
│   │   ├── mesh/               # mesh types, generators, octree
│   │   └── postprocess/        # pressure and far-field evaluation
│   ├── room_acoustics/         # Room-simulation library API
│   ├── materials.rs            # Impedance/admittance models (try_* checked)
│   └── numcalc/                # Mesh2HRTF project loading and sweep runner
├── tests/                      # Integration tests (Mie, parity, contracts)
├── benches/                    # Criterion benchmarks (incl. fmm_slow_validation)
└── examples/                   # Library usage examples (room simulator, diagnostics)
```

Analytical references live in the `sonium-wave` crate and are re-exported
for validation. `CONVENTIONS.md` documents the single acoustic convention
(`e^{-iwt}`, `G=e^{ikr}/4pr`, outward normals).

## References

### Primary Research

1. **Burton, A.J., & Miller, G.F. (1971)**
   "The application of integral equation methods to the numerical solution of some exterior boundary-value problems"
   *Proceedings of the Royal Society of London A*, vol. 323, pp. 201-210
   [DOI: 10.1098/rspa.1971.0097](https://doi.org/10.1098/rspa.1971.0097)

2. **Gumerov, N.A., & Duraiswami, R. (2009)**
   "Fast multipole methods for the Helmholtz equation in three dimensions"
   *Elsevier Series in Electromagnetism*
   ISBN: 978-0080531595

3. **Sauter, S.A., & Schwab, C. (2011)**
   "Boundary Element Methods"
   *Springer Series in Computational Mathematics*, vol. 39
   ISBN: 978-3-540-68092-5

4. **Marburg, S., & Nolte, B. (2008)**
   "Computational Acoustics of Noise Propagation in Fluids - Finite and Boundary Element Methods"
   *Springer*
   ISBN: 978-3-540-77447-1

## Build Instructions

### Prerequisites

```bash
# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# BLAS library (platform-specific)
# macOS: Accelerate framework (included)
# Linux: OpenBLAS
sudo apt install libopenblas-dev
# Windows: Intel MKL or OpenBLAS
```

### Build

```bash
# Build library and binary
cargo build --release

# Run tests
cargo test --release

# Build documentation
cargo doc --open
```

## Performance Notes

### Memory Efficiency

- **Batched workspaces**: Pre-allocated buffers for FMM matvec
- **Sparse near-field**: Only store near-field blocks
- **Parallel assembly**: Rayon-based parallel matrix construction

### Solver Selection

- For N < 1000: Benchmark the direct solver as a candidate
- For 1000 < N < 20000: FMM+GMRES with the checked near-field ILU path when
  its validity conditions pass
- For N > 20000: FMM+Batched BLAS for reduced memory pressure

### Parallel Execution

```bash
# Control thread count with Rayon
RAYON_NUM_THREADS=8 ./target/release/numcalc -- path/to/Project/NC.inp
```

## License

BUSL-1.1 (workspace license; see the repository root).

## Citation

```bibtex
@software{sonium_bem,
  title = {sonium-bem: Rust Boundary Element Method Library},
  author = {Sonium Contributors},
  year = {2026},
  url = {https://github.com/pierreaubert/sonium/tree/main/crates/sonium-bem}
}
```
