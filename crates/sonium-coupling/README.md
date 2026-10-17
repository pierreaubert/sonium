# sonium-coupling

Reusable interior/exterior FEM–BEM coupling orchestration for
Sonium. The crate sits above `sonium-fem` and `sonium-bem` and owns
the numeric coupling seam: monolithic trace operators, interface
power ledgers, aperture-trace contracts, and interior acoustic
reduced-order models.

## Scope

- `monolithic`: product-independent interior/exterior acoustic
  operators with compressed exterior blocks and field-split
  preconditioning.
- `power`: interface power computed independently on each side of
  an aperture, plus the conservation audit tying them together.
- `trace`: validated aperture-trace schemas shared by interior and
  exterior builders.
- `acoustic_rom`: rigid-wall and Craig–Bampton interior modal bases
  with static correction for coupled sweeps.

Product geometry, driver networks, and measurement acceptance stay
in the simulation crates; this crate is the reusable coupling layer
below them.

## Development

```bash
cargo test -p sonium-coupling
```

See `CHANGELOG.md` for release history.
