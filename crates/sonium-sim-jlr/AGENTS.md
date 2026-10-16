# sonium-sim-jlr (lib: `sonium_sim_jlr`, version: 0.1.0)

Replication of Melchior et al. 2026 (IJHFF 121, 110519) JLR LBM study.
Plan: `reviews/jlr-20261015.md` (meta-workspace root).

## Invariants

- Paper values live only in `fixtures/` with `PROVENANCE.md`; never edit
  a fixture to match a run. Derived dimensions stay marked `derived`.
- Verification ladder (`tests/verification.rs`) must pass before any
  JLR result is quoted; CUDA numbers require `tests/cuda_parity.rs`.
- Production claims require a stored bundle re-checked by `qa-jlr-lbm`
  (sonium-qa, `--features qa-binaries`, `--release`).
- The paper's molecular tau is 0.500005. Production collision is the
  central-moment (cascaded) operator; do not change collision models
  silently — it is a documented deviation requiring a plan update.
- The D3Q19 opposite-direction table must stay consistent with the
  velocity arrays (pinned by `d3q19_tables_are_self_consistent`): a
  mismatch corrupts every diagonal bounce-back and presents as wall
  blow-up and mass leak, not as an obvious table error.
- Refinement coupling is one-way macroscopic nesting (see
  `src/refinement.rs` header); do not reintroduce per-step coarse
  patch overwrites — the channel test shows they plug the flow.
