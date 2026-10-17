# sonium-materials

Unified, versioned material records for Sonium. The crate replaces
scattered per-physics material structs with one `Material` record
carrying provenance, units, tolerances, and curves, plus fail-closed
adapters back to the FEM, magnetics, panel, thermal, porous-acoustic,
speaker-BEM, and cabinet-design material types.

## Scope

- One versioned record for elastic, magnetic, thermal, porous, and
  panel material data, with explicit provenance classes.
- Allocation-free curve interpolation on the hot path (binary search
  over borrowed slices); pack I/O, checksums, and validation stay on
  cold paths.
- Computed screening data (for example OPTIMADE or Materials
  Project drafts) remains explicitly labelled and is never promoted
  silently to measured datasheet truth.

The core is pure Rust. Network fetchers for external databases are
outer tools; this crate only validates the drafts they produce.

## Development

```bash
cargo test -p sonium-materials
```

See `CHANGELOG.md` for release history.
