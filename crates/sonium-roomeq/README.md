# sonium-roomeq

RoomEQ configuration types and CEA-2034 spinorama scoring for
Sonium. The crate replaces the external `autoeq` dependency with a
minimal, ndarray-free implementation owned by the workspace.

## Scope

- `cea2034`: CTA-2034 curve aggregation and scoring — on-axis,
  listening window, early reflections, sound power, and estimated
  in-room response with the standard group/weight definitions.
- `roomeq`: RoomEQ configuration types shared by measurement,
  optimisation, and export workflows.

Consumers that need identical spinorama definitions (speaker BEM,
workbench studies, reporting) should route through this crate rather
than re-implementing the aggregation.

## Development

```bash
cargo test -p sonium-roomeq
```

See `CHANGELOG.md` for release history.
