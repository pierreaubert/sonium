# AGENTS.md

## Build & Test Commands

```bash
# Build (default-members only, excludes WASM crates)
cargo check
cargo clippy

# Build a specific crate
cargo check -p sonium-solvers
cargo clippy -p sonium-solvers

# Build/test the focused solver crates (the facade remains the compatibility API)
cargo check -p sonium-solvers-core
cargo test -p sonium-solvers-iterative
cargo test -p sonium-solvers-preconditioners

# Run the complete split solver family
for c in sonium-solvers-core sonium-solvers-sys sonium-solvers-sparse sonium-solvers-eigen sonium-solvers-direct sonium-solvers-preconditioners sonium-solvers-iterative sonium-solvers-optimization sonium-solvers-nonlinear sonium-solvers-time-integration sonium-solvers; do
  cargo test -p $c
done

# Run all tests for a crate
cargo test -p sonium-solvers
cargo test -p sonium-bem
cargo test -p sonium-fem

# Run a single test
cargo test -p sonium-solvers test_name

# Run benchmarks (criterion)
cargo bench -p sonium-solvers --bench solver_bench
cargo bench -p sonium-bem --bench bem_solver_benchmark
cargo bench -p sonium-fem --bench helmholtz_3d_scaling

# Run binaries
cargo run -p sonium-bem --bin roomsim -- [args]
cargo run -p sonium-fem --bin roomsim-fem -- [args]
cargo run -p sonium-sim-room --bin generate-roomeq-data -- [args]
cargo run -p sonium-solvers --features cli --bin sonium-benchmark -- [args]
```

Always run `cargo check` then `cargo clippy` before considering work done. Validate tests in the specific crate you modified.

### Cross-architecture SIMD validation

`sonium-fastmath` and `sonium-compute::bem_cpu` ship both aarch64 (NEON)
and x86_64 (AVX2 + FMA) SIMD kernels. The aarch64 path is the daily
driver on this machine; the x86_64 path is validated through Rosetta 2,
which emulates AVX2 + FMA on Apple silicon.

```bash
just validate-simd     # check-x86 + test-x86 + native aarch64 SIMD tests
```

Run this before merging any change that touches a
`#[cfg(target_arch = "x86_64")]` block in either crate, or any
`sonium-fastmath` polynomial / range-reduction code. The recipe asserts
numerical parity vs scalar on both architectures within the documented
ULP envelope.

## Architecture

Sonium is a Rust workspace for computational acoustics simulation. Edition 2024, stable toolchain.

### Dependency Layers

```
Layer 0 - Math primitives
  sonium-fastmath      Vectorised sincos f64 (NEON 2-lane today, AVX2/scalar fallback).
                       Cody-Waite range reduction + Sleef u10 minimax polynomials.
                       Consumed by sonium-compute (BEM Helmholtz kernel) and any
                       future hot path that needs `exp(i·k·r)`-style work.

Layer 1 - Foundations
  sonium-wave          Analytical solutions (plane waves, cylinder/sphere scattering, Mie theory)
  sonium-xem-common    Shared types between BEM/FEM (room geometry, source config, JSON I/O)
  sonium-solvers       Compatibility facade over sonium-solvers-{core,sys,sparse,
                       eigen,direct,preconditioners,iterative,optimization,
                       nonlinear,time-integration}; it also owns high-level solve,
                       mixed-precision and differentiable APIs.
                       All solvers generic over f32/f64/Complex32/Complex64.
  sonium-mesher        Mesh generation (tetrahedralization, partitioning), 3D convex hull (convexhull/)

Layer 1.5 - Transforms
  sonium-fft           Batched complex FFT (mixed-radix Stockham + Bluestein) on top of
                       sonium-compute. CPU only in M1; Metal / cuFFT land in M3/M4.
                       Ships a rustfft 6.x drop-in adapter (rustfft_compat::*).
                       Generic over f32 / f64; Complex<T> buffers.

Layer 2 - PDE Solvers
  sonium-bem           Boundary Element Method: traditional + FMM assembly, Burton-Miller formulation,
                       singular integration (Duffy), frequency sweeps, batched BLAS
  sonium-fem           Finite Element Method: multi-physics, generic assembly, layered architecture.
                       See sonium-fem/AGENTS.md for the full physics module list.

Layer 3 - Applications
  sonium-sim-room      Generates room measurement data (17 scenarios) for RoomEQ optimizer testing
  sonium-sim-speaker   Loudspeaker BEM simulation, CEA-2034 output, and optimisation
  sonium-sim-speaker-modeling
                       Driver, enclosure, and resolved nonlinear port-flow models
  sonium-sim-motor     Showcase mini-app for the rotating-machine stack
                       (GEO_30MM operating points, gamma sweep/MTPA,
                       thermal and rotor-stress checks; `motor-sim` CLI)

Layer 4 - Quality Assurance (sibling `sonium-accuracy` workspace)
  sonium-qa            Standalone validation binaries (qa-bem-*, qa-fem-*, qa-solvers-*).
                       Run them from ../sonium-accuracy.
                       `qa-speaker-port-nonlinear` owns the normal 2D
                       pressure-traction Poiseuille, tiny-drive lumped-limit,
                       compression/energy, and THD gates; `SPEAKER_PORT_3D=1`
                       additionally enables the nightly 3D pipe-flow and exact
                       complex-reference gates. The 2D path is validated at a
                       converged timestep; the 3D runtime/reference-convergence
                       gates remain incomplete and must not be reported passed.
```

### Feature Flags

Rayon parallelism is always-on. Remaining optional features:

- **`cli`**: Command-line binary support (BEM, FEM, sonium-benchmark in solvers)
- **`mshio`**: Gmsh .msh file support (FEM, forwarded to sonium-mesher)
- **`autograd`**: Automatic differentiation for topology optimization (FEM only)
- **`meshlang`**: MeshLang geometry compiler (sonium-mesher)

### Key Traits (sonium-solvers-core, re-exported by sonium-solvers)

- `ComplexField` — abstraction over f32, f64, Complex32, Complex64 (and f16). All solvers are generic over this trait.
- `LinearOperator<T>` — generic matrix-vector product interface used by all iterative solvers; `apply_block` (default per-column, dense wrappers override with one `mat_mul_nn` GEMM) is the multi-vector contract used by block GMRES
- `Preconditioner<T>` — preconditioning interface (Jacobi, ILU, ILUT, HILU, AMG, block Jacobi, Schwarz); also `apply_block` for multi-vector application, plus adjoint-capable `apply_transpose`/`apply_hermitian` (default `apply_hermitian` conjugates through `apply_transpose`; ILU and Jacobi override with exact Mᵀ/Mᴴ solves) for adjoint systems like Aᴴλ = g

### Key Traits (sonium-fem)

- `BilinearForm<T>` / `LinearForm<T>` — physics-specific weak form integrands. Each physics module implements these.
- `FunctionSpace` — DOF mapping and basis function tabulation (H1, H(curl), H(div))
- `CooOperator` — COO sparse matrix interface for multigrid (physics-agnostic)

### FEM Architecture (sonium-fem)

The FEM crate has a layered architecture where physics modules are orthogonal to infrastructure:

```
Layer 0: Geometry       mesh/, basis/, quadrature/
Layer 1: Forms          forms/traits.rs  (BilinearForm<T>, LinearForm<T>)
Layer 2: Assembly       assembly/generic.rs, prebaked.rs  (physics-agnostic)
Layer 3: BCs            boundary/generic.rs  (FemSystem<T>, Dirichlet/Neumann/Robin)
Layer 4: Solver         solver/  (iterative/direct)
Layer 5: Physics        physics/*/forms.rs  (implement form traits, zero infra coupling)
```

Adding a new physics module only requires implementing BilinearForm<T>. No changes to assembly, boundary, solver, or multigrid.

### External Dependencies

Two crates are imported from a sibling repo (`../sotf/`):
- `autoeq` — RoomEQ configuration types (used by sonium-sim-room)
- `math-delaunay` — Delaunay triangulation for mesh generation (used by sonium-bem)

### Testing Strategy

Tests validate against analytical solutions (sonium-wave provides reference). BEM tests compare 1D/2D/3D scattering against exact results. FEM tests validate all physics modules: acoustics (Helmholtz, viscothermal, impedance, ray tracing, metrics), elasticity, Maxwell, electrostatic transducers, lumped circuit, and nonlinear solvers. The datagen pipeline test runs a full BEM solve and verifies output config structure. Solver tests cover all iterative methods with f32/f64/Complex32/Complex64 types.

### Performance Philosophy

Performance is critical. Priorities in order:

1. **Algorithmic correctness first** — use the right solver and the right preconditioner for the physics. An unpreconditioned GMRES on an indefinite Helmholtz system will fail; a shifted-Laplacian + ILU preconditioner makes it trivial. Wrong algorithm = wrong answer or no answer, regardless of how fast it runs.
   - Helmholtz (FEM): FGMRES + shifted-Laplacian ILU or AMG-preconditioned GMRES
   - BEM dense (N<1000): direct LU
   - BEM medium (N<5000): GMRES + ILU
   - BEM large (N>5000): FMM + GMRES + ILU
   - SPD systems (elasticity, Laplacian): CG + AMG
   - SPD/HPD sparse direct: simplicial Cholesky + nested-dissection ordering
     (~2x less fill and flops than nonsymmetric LU)
   - Saddle-point (Stokes): block preconditioner with Schur complement
   - Frequency sweeps: PrebakedAssembler for O(nnz) re-assembly

2. **Parallelism for large problems** — use rayon parallel assembly and parallel SpMV when the problem is large enough to amortize thread overhead. For small problems (<1000 DOFs), serial is faster. Thread count is tuned for memory bandwidth saturation (~4 threads on M4).

3. **Memory awareness** — memory varies widely per machine. Sparse CSR is default. For BEM, full dense matrices only for N<1000; above that, FMM or H-matrix compression is mandatory. NUMA-aware partitioning for large FEM meshes. Mixed-precision (f32 inner solve, f64 outer) halves memory for iterative solvers when applicable.

### Performance Notes

- AMG preconditioner targets ~1.4x of PETSc performance
- Sparse matrix (CSR) supports NUMA-aware partitioning, SIMD SpMV (AVX2/NEON)
- FEM PrebakedAssembler enables O(nnz) frequency sweep re-assembly (no sparsity pattern reconstruction)
