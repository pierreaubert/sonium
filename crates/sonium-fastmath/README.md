# sonium-fastmath

Vectorised math primitives for the Sonium workspace, centred on
paired sine/cosine evaluation for phase, phasor, and twiddle work
in `sonium-compute`, `sonium-fft`, and acoustic kernels.

## Scope

- Two-term Cody–Waite range reduction with Sleef-style minimax
  polynomials for the fast path, a three-term reduction for
  mid-range inputs, and a Payne–Hanek slow path for large finite
  values.
- Lane backends for NEON, AVX2/AVX-512, WASM SIMD, and a scalar
  fallback with identical contracts.
- Deterministic backend selection, with environment overrides for
  reproducible baselines (`SONIUM_FASTMATH_FORCE_SCALAR=1`,
  `SONIUM_FASTMATH_FORCE_AVX2=1`).

Non-finite inputs delegate to the standard library so edge-case
behaviour is never silently changed by the fast path.

## Development

```bash
cargo test -p sonium-fastmath
```

See `CHANGELOG.md` for release history.
