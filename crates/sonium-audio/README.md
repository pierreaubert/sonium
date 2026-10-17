# sonium-audio

Audio DSP building blocks for Sonium: IIR and FIR filters, filter
design, room-impulse-response analysis, analog coloration, and
differentiable DSP primitives.

## Scope

- `iir`: biquads, parametric EQ, state-variable filters, Kautz
  systems, and IIR crossovers.
- `fir`: windowed-sinc FIR design, target-response design, and FIR
  crossovers.
- `rir`: room-response analysis helpers.
- `analog` and `autodiff`: analog-style coloration and differentiable
  DSP building blocks.

The crate is intentionally ndarray-free and keeps its public surface
focused on reusable DSP primitives rather than application policy.

## Development

```bash
cargo test -p sonium-audio
```

See `CHANGELOG.md` for release history.
