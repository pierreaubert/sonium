# sonium-fem (lib: `sonium_fem`)

Multi-physics Finite Element Method solver with a layered, physics-agnostic architecture.

## Architecture

```
Layer 0: Geometry    mesh/, basis/, quadrature/
Layer 1: Forms       forms/traits.rs  (BilinearForm<T>, LinearForm<T>, NonlinearForm<T>)
Layer 2: Assembly    assembly/generic.rs, prebaked.rs, triplet.rs  (physics-agnostic)
Layer 3: BCs         boundary/generic.rs  (FemSystem<T>, Dirichlet/Neumann/Robin)
Layer 4: Solver      solver/  (GMRES, CG, AMG, shifted-Laplacian, WaveHoltz, Schwarz-PML)
Layer 5: Physics     physics/<group>/*/forms.rs  (each implements form traits independently)
```

Local forms reuse physics-independent assembly infrastructure. A new coupled or
transient domain also needs appropriate unknown spaces, boundary conditions,
nonlinear/history operators, solver admission and validation; implementing
`BilinearForm<T>` alone does not supply those contracts.

## Physics Modules

### Acoustics (`physics/acoustics/pressure/`)
- **Frequency domain**: Helmholtz equation, PrebakedAssembler for O(nnz) frequency sweeps
- **Time domain**: Newmark-β implicit, explicit leapfrog
- **Viscothermal**: Equivalent fluid model (LRF) for narrow tubes/slits, thermal BLI
- **Boundary conditions**: Wall, impedance (Delany-Bazley, Miki porous models), waveguide ports with multimode expansion, radiation (Sommerfeld, BGT 2nd order), Floquet periodic
- **Sources**: Normal velocity/acceleration/displacement, monopole point sources
- **Room acoustics**: Ray tracing (Möller-Trumbore), ISO 9613-1 atmosphere model, acoustic diffusion equation, metrics (RT60, EDT, C80, D50, Ts)
- **Post-processing**: Particle velocity, acoustic intensity, SPL, transmission loss, absorption coefficient
- **Solvers**: WaveHoltz, Schwarz-PML, neural multigrid

### Elasticity (`physics/solid/elasticity/`)
- Linear isotropic (plane stress/strain, 3D)
- Viscoelastic (structural damping with loss factor)
- Predefined materials: steel, aluminum, rubber, loudspeaker surround/spider

### Electrostatic (`physics/electromagnetics/electrostatic.rs`)
- Parallel-plate gap: force, capacitance, coupling factor, pull-in voltage
- Linearised electrostatic-mechanical coupling for condenser microphones
- Nonlinear large-signal dynamics

### Electroacoustics (`physics/acoustics/electroacoustics/`)
- Fluid-structure coupling: block system assembly (pressure + displacement)
- Multi-domain transducer solver: electrodynamic (voice coil) and electrostatic (condenser)
- Lumped parameter extraction: Thiele-Small parameters, radiation impedance, effective area
- Cabinet panel radiation (`cabinet_radiation.rs`): simply-supported plate
  modal model; `modal_displacements`/`surface_displacement` expose the
  per-mode displacement X_mn (same pressure-driven convention as
  `surface_velocity`) for structural constraint evaluation

### Lumped Circuit (`physics/coupled/lumped_circuit/`)
- Voice coil: linear + nonlinear (displacement-dependent BL, L, reluctance force)
- Acoustic elements: compliance, mass, resistance, Helmholtz resonator
- Two-port transfer matrix networks: tube, expansion chamber, cascade

### Other Physics
- **Maxwell** (`physics/electromagnetics/maxwell/`): Curl-curl + edge mass, H(curl) Nedelec elements
- **Magnetostatics** (`physics/electromagnetics/magnetostatics/`): Scalar potential, permanent magnet sources, BL extraction
- **Eddy current** (`physics/electromagnetics/eddy_current/`): Frequency-domain A-formulation
- **Rotating machine** (`physics/electromagnetics/rotating_machine/`): Planar nonlinear `A_z` (P1/P2, chord Newton), native CDT sliding-band motor mesh, current/voltage/PWM drive transients, solid-conductor eddy blocks, demagnetisation ratchet, Bertotti/measured-surface iron + copper losses, anisotropic thermal twin, centrifugal rotor stress. Validated against the eMotres/Motor_Ai_Simulation Python reference (same-mesh identity oracles + GEO_30MM operating-point pins in `tests/rotating_machine_*`)
- **Thermal** (`physics/thermal/heat_conduction/`): Conduction, advection, convection, radiation
- **Nonlinear solid** (`physics/solid/nonlinear_solid/`): Hyperelasticity, St. Venant-Kirchhoff
- **Beam** (`physics/solid/beam/`): Euler-Bernoulli beam theory
- **Shell** (`physics/solid/shell/`): Kirchhoff-Love, Reissner-Mindlin
- **Plasticity** (`physics/solid/plasticity/`): Return mapping algorithms

This list is selective, not an absence inventory. `physics/mod.rs` also exports
flow, turbulence, transport, contact, FSI, porous-flow, multiphysics and other
reference modules. Module presence is not production-scale validation or
support for every backend, element order and derivative combination.

## Adding a New Physics Module

1. Create `physics/<group>/my_domain/forms.rs` — implement `BilinearForm<T>` and/or `LinearForm<T>`
2. Create `physics/<group>/my_domain/materials.rs` — material property structs
3. (Optional) Create `physics/<group>/my_domain/assembler.rs` — bridge to `PrebakedAssembler<N>` for frequency sweeps

The form only needs to implement `integrate_point()`. The generic assembler handles element loops, quadrature, DOF scattering, and parallelization automatically.

## Features

- `cli` (default) — Command-line support, including `mshio`
- `mshio` — Gmsh `.msh` loading through `sonium-mesher`
- `meshlang` — MeshLang compilation plus Gmsh loading through `sonium-mesher`
- `cuda`, `metal` — Forward backend features to `sonium-compute`; workload
  admission remains governed by `src/backend.rs`, not feature presence alone

Rayon is an unconditional dependency. There are no `native`, `parallel` or
`autograd` Cargo features; use `Cargo.toml` as the feature authority.

## Testing

```bash
cargo test -p sonium-fem --lib
cargo check -p sonium-fem && cargo clippy -p sonium-fem
```

## Dependencies

- `sonium-solvers` — Linear/nonlinear solvers, preconditioners, sparse matrices
- `sonium-wave` — Analytical solutions for validation
- `sonium-xem` — Shared room geometry types
- `sonium-compute` — Dense host containers (`HostMatrix`, `HostTensor3`) and backend abstraction
- `num-complex` — Complex number support
