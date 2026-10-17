# Changelog

## 0.2.0

### Changed

- Bumped the package version from 0.1.8 to 0.2.0 as part of the coordinated minor release.

## 0.1.7

### Fixed

- Emit valid, complete multi-operand CSG entity lists in both GEO and Python backends.
- Keep geometry-local bindings scoped to their geometry and reject unsafe multi-statement
  function inlining explicitly.
- Replace runtime `-1` tag sentinels with stable dimension-qualified entity identities.
- Lower surface CSG and compound declarations with their inferred source dimensions.
- Recurse through nested Python `else if` nodes without revisiting their parent.
- Preserve multi-entity CSG results, topology aliases, sweeps, lofts, fillets, chamfers,
  physical groups, mesh fields, and selector-derived boundaries across both backends.
- Add canonical IR, deterministic artifact metadata, provenance/sensitivity manifests,
  sandboxed imports, incremental caches, WASM preview support, real-Gmsh validation,
  benchmarks, and compiler fuzz assets.
