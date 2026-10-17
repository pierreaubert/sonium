# Changelog

## 0.3.0

### Changed

- Bumped the package version from 0.2.1 to 0.3.0 as part of the coordinated minor release.

## 0.2.0

### Added

- Explicit AST schema v2 envelopes, bounded deserialization, and a tested migration from the
  historical raw v1 `SourceFile` representation.
- Lossless source identity, comments/docstrings, stable source-derived node IDs, transform
  provenance, and exact spans for literal and parenthesized expressions.
- Typed design variables, distributions, objectives, constraints, design/physical/material
  regions, boundaries, interfaces, coordinate systems, topology/level-set provenance,
  manufacturing annotations, and anisotropic mesh metrics.
- A canonical backend-neutral geometry IR with dimension-qualified entity IDs, validation,
  optimization/sensitivity metadata, and deterministic structural hashes.
- Deterministic JSON Schema, TypeScript, and Python artifact generation via
  `cargo run -p meshlang-ast --example generate_tooling -- <format>`.
- Enum-generated tooling inventories and exhaustive visitor matches, so new variants cannot be
  silently omitted.

### Changed

- `Unit::parse` now returns `Result<Unit, UnitParseError>` and supports products, quotients,
  signed integer exponents, parentheses, derived SI units, and explicit dimensionless units.
- `Expr::Literal` and `Expr::Paren` now carry `LiteralExpr`/`ParenExpr` payloads with exact spans.
- `SourceFile` includes serde-defaulted source metadata, comments, and provenance.
- `ResolvedType` exposes entity dimensions and treats both surfaces and solids as CSG operands.

### Compatibility

- Persisted v1 AST JSON must be loaded through `VersionedAst::from_json_slice`; migration is
  explicit and future schema versions are rejected.
- This is a Rust API breaking release because silent unit parsing and span-less expression
  payloads were removed.

## 0.1.2

### Known Issues

- `Unit::parse` silently returns `None` for compound units like `kg*m` or units with exponents.
  No diagnostic is emitted when a unit-like token is unrecognized.
