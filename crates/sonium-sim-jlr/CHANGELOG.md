# Changelog

## 0.2.0

### Changed

- Bumped the package version from 0.1.0 to 0.2.0 as part of the coordinated minor release.

## 0.1.0 — 2026-10-16

- Initial replication crate for Melchior et al. 2026 (JLR LBM):
  D3Q19 BGK + Smagorinsky LES core (CPU f64), CUDA f32 kernel with
  CPU parity gate, analytic JLR voxel geometry, two-level one-way
  nested refinement, case runner + MLUPS benchmark, paper fixtures
  (Tables A.4/4.1/4.2/A.1, digitised Fig. 4.3), and the `qa-jlr-lbm`
  validation binary in sonium-qa.
