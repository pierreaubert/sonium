# Changelog

All notable changes to the OpenCourant-parity crates (`sonium-qa`,
`sonium-fem`, `sonium-compute`, `sonium-bench`). Dates are UTC.

## 2026-10-17 — Coordinated minor release

Bumps: every crate in the `sonium` workspace received a minor version bump (patch reset to zero). Crate changelogs were released or introduced to match, and crates that had no `README.md` gained one. No API break is implied by the coordinated bump itself; per-crate entries carry the substantive notes.

## 2026-10-13 — Follow-ups: RSS measurement, wolfram flake, QA rerun

No version bumps (harness + QA-only changes).

- `qa-oc-bench` sonium-arm peak RSS now reads `VmHWM` from
  `/proc/self/status` on Linux: `getrusage(RUSAGE_SELF)` kept
  the launcher's `ru_maxrss` across `fork` + `exec`, so every
  `cargo run` cell reported cargo's ~83 MB peak instead of the
  binary's own footprint (true: 4-13 MB per cell; proven with a
  120 MB balloon parent). Sonium matrix cells republished;
  ns/dof/cycle and correctness unchanged.
- `wolfram_audio` ETXTBSY flake fixed: fixture scripts now
  live in per-process temp dirs (pid-suffixed, matching the
  sibling harnesses), and engine launch retries a bounded
  number of times on `ExecutableFileBusy` — std `Command`
  spawns of a script intermittently return ETXTBSY under
  thread contention (~10% per spawn; raw fork+execve: 0%).
  Verified 0/48 concurrent test processes (was 44/48) and
  0/40 suite runs (was 6/30).
- `just qa-opencourant` re-run clean after the perf edits:
  24 passed, 0 failed, 1 by-design skip. (A first run showed
  one leg failing to compile `sonium-fem` mid-run under a
  concurrent worktree edit; the clean rerun passes it.)

## 2026-10-12 — Perf goal: hertz beats engine (linear bulk)

No version bumps (harness-only changes).

- Precomputed linear Q1 stiffness for small-strain LAW1 decks
  (`linear_hex_stiffness`, `qa-oc-bench`): one dgemv per
  element replaces the finite-strain kernel (Jacobians,
  determinants, logarithms at 8 Gauss points), proved
  equivalent to Neo-Hookean at small strain by unit test.
- Gated to `hertz_sphere`: taylor_bar impact (~10% strain)
  misses DZ by 5.5% under linearization, so finite-strain
  stays the default elsewhere (`eval_law1_internal_finite`).
- Hertz native: 9.1 s / 60 ns/dof/cycle, beating engine serial
  (25.6 s / 128 ns) 2.8x wall and 2.1x per-cycle; DZ exact to
  7 digits. Matrix republished (taylor/hertz/rail/necking).
- Taylor native: 0.003 s / 216 ns/dof/cycle at 21 bulk-CFL
  cycles (was 0.027 s / 317 ns at 127 penalty-capped cycles),
  9.8x per-cycle over engine serial; probes green again after
  the multiplier-contact refactor converged.

## 2026-10-12 — Perf reporting: OC plastic matrix + regen plumbing

No version bumps (harness + QA-only changes, no library API breaks).

### OpenCourant explicit matrix extended to plasticity (49 cells)

- Native LAW2/LAW36 arm (`qa-oc-bench`) green on `taylor_plastic`
  and `plasticity_thick_cylinder` (both arms, all probes).
- Linear bulk viscosity off (c1 = 0) in the plastic arm: the
  compression-only kink chattered under velocity-Verlet on
  quasi-static ramps (55%/step from roundoff into a 13 mm limit
  cycle); the C1-smooth quadratic term stays on for shocks.
- Probe sampling mirrors each deck's engine leg: taylor_plastic
  FNZ plateau mean, thick-cylinder transient DX (50 MPa crossing
  + dilatational transit) and residual means.
- `th_to_csv` vendor workaround: bare filename when the T01 sits
  in the run dir (absolute paths over ~80 chars SIGFPE the
  converter, CommitID 33e68517); unblocks the thick-cylinder leg.
- `law36_caps_at_tabulated_yield` expectation corrected to the
  constrained uniaxial-strain mark (mean stress K·1% over the
  capped deviator, not uniaxial stress).

### Perf regen tooling

- `scripts/perf_oc_matrix.sh` runs the (deck × side × flavor ×
  threads) matrix serially; `scripts/perf_oc_publish.py`
  republishes `perf/oc-explicit.md` tables (failing rows stay
  unpublished by policy).
- Just recipes: `perf-oc` (full 6 decks), `perf-oc-elastic`,
  `perf-oc-plastic`, `perf-oc-deck <deck>`; `PERF_OC_QUICK=1`
  smoke protocol, hertz stays quick by default (cost).
- Solver reports moved under `perf/` (`perf/PERF.md`,
  `perf/ITERATIVE_PERF.json`) with `just perf-solvers` regen.

## 2026-10-12 — Phase 7: audit follow-ups + OpenCourant naming

Bumps: `sonium-qa` 0.1.5 to 0.1.6, `sonium-fem` 0.6.0 to 0.6.1,
`sonium-compute` 0.1.6 to 0.1.7, `sonium-bench` 0.2.0 to 0.2.1.
No API breaks: `RadCase` stays as a deprecated alias of
`OpenCourantCase`, and `RADISS_EXE` stays as a legacy fallback
for `OPENCOURANT_EXE`.

### Naming unification (radioss/openradioss to OpenCourant)

- 15 QA binaries renamed `fem_radioss_*` to `fem_opencourant_*`
  (`qa-fem-opencourant-*`); `RadCase` renamed to
  `OpenCourantCase` with a deprecated alias.
- Engine resolution prefers `OPENCOURANT_EXE`, falls back to
  `RADISS_EXE`; skip/defer messages updated.
- Vendor facts stay verbatim: `#RADIOSS` deck magics, the
  `OpenRadioss` engine banner word, golden `engine` stamps,
  `hm_cfg_files` sections, manual titles, the `.rad`
  extension, and `rad.*` file-format module names.

### Audit follow-ups (one lane each)

- 7A explicit plasticity: generic `hex_explicit_forces<M>`
  material wiring plus `hex_plastic.rs` (Johnson-Cook/plastic
  update, erosion via `ductile::ErosionSet`); Taylor
  bar with contact + plasticity + erosion holds energy
  (drift 1.7e-5) and stable dt.
- 7B explicit shells: `shell/explicit.rs` central-difference
  shell dynamics on the existing element/mass/elastoplastic
  kernels; Ishell 12/24 + Ish3n 30/31 flags; thick-shell
  build/defer verdict recorded with dated defers.
- 7C native laws + rupture: `foam_explicit.rs` native foam
  LAW driver (LOI50 corotational full-stress replay, LOI76
  TRUE-strain path, LOI68 identities), `Law66Shell` port,
  `coupled.rs` native coupled rupture solve (LAW36 +
  FAIL_TAB + Wierzbicki with golden erosion stamps).
  Formulation bands tightened from 0.05 to measured 0.002
  (2x headroom over the 0.001 knee, re-measure after regen).
- 7D kernels + replays + FUNCT_PYTHON: `q4_*`
  Flanagan-Belytschko kernels mirroring hexa8 with a W-tier
  triple; `--series` + `--replay` for SPH (194 channels /
  8730 samples) and ALE (93 channels / 74400 samples) at
  vendor tolerance; native-vs-engine fidelity gap honestly
  reported (native replay exits 1 by design); `pyo3` 0.23
  `/FUNCT_PYTHON` evaluation behind the `funct-python`
  feature (off builds print a loud `DEFERRED`).

### QA runner

- New `just qa-opencourant` recipe: runs the OpenCourant
  subset in `--release` (engine-free native/golden legs,
  `sonium-qa` lib suites with and without `funct-python`,
  and the 15 live-engine legs, which skip loudly without
  `OPENCOURANT_EXE`). Fails on any `FAILED` leg or suite;
  skips never fail.

### Gate evidence (coordinator-observed, real engine)

- Legs: taylor-plastic, shell-explicit, foam-native (6/6),
  rupture-native, `qa-fem-sph`, `qa-fem-ale` all exit 0
  `SUCCESS`; `qa-fem-opencourant-rail-crush` PASS + SUCCESS
  against `exec/engine_linux64_gf`.
- Suites: qa lib 225/0 (240/0 with `funct-python`), compute
  629+, fem 1753+, bench green; rustfmt clean on all new
  files; QA inventory regenerated and fresh.
- `qa_contract` is 54/55: the one failure is pre-existing
  on HEAD (proven via `git show`) with 25 committed 5%
  gates plus scanner false positives (`5.0e-3`, `5.0e-5`,
  `0.0508186` mass values); Phase 7 adds zero new flags.
- `just clippy` is red only on 2 peer-owned Stokes
  collapsible-ifs (0 mentions of Phase 7 files); the fem
  `#[ignore]`d Poiseuille test is peer-owned nightly
  scope. Both recorded as follow-ups, not Phase 7 scope.

### Standing deferrals

- XFEM/IGE (since Phase 3); P5C3 `/FUNCT_PYTHON` on
  default builds (loud deferral until `--features
  funct-python`); native SPH/ALE formulation fidelity vs
  the engine (machinery proven, physics is future work);
  dated 7C small items (LAW36 leg edit, PENTA6 fixture,
  LOI50/LOI76/LAW68 goldens beyond the legs, `Fad_exp`).
