# CUDA (NVIDIA GPU, Linux) — FFT

_Seeded 2026-10-15 (not yet measured; regenerate with `just perf-cell cuda fft`)._

- status: seed

## Results

Single unnormalised forward transforms; gap = sonium ÷ rustfft (1.00× is parity). The non-vendor arm is bonus context (— where the harness has no such arm).

Not yet measured on this platform: every cell below is a placeholder until a real `just perf-cell cuda fft` run overwrites this file. The fft harness has no GPU arm, so this cell records the host-CPU comparison.

### `fft_complex_f32`

| size | sonium | rustfft | accelerate | gap |
|---|---|---|---|---|
| 512 | — | — | — | — |
| 1024 | — | — | — | — |
| 2048 | — | — | — | — |
| 4096 | — | — | — | — |
| 1048576 | — | — | — | — |
| 2097152 | — | — | — | — |
| 4194304 | — | — | — | — |

### `fft_complex_f64`

| size | sonium | rustfft | accelerate | gap |
|---|---|---|---|---|
| 512 | — | — | — | — |
| 1024 | — | — | — | — |
| 2048 | — | — | — | — |
| 4096 | — | — | — | — |
| 1048576 | — | — | — | — |
| 2097152 | — | — | — | — |
| 4194304 | — | — | — | — |

### `fft_real_f32`

| size | sonium | rustfft | accelerate | gap |
|---|---|---|---|---|
| 512 | — | — | — | — |
| 1024 | — | — | — | — |
| 2048 | — | — | — | — |
| 4096 | — | — | — | — |
| 1048576 | — | — | — | — |
| 2097152 | — | — | — | — |
| 4194304 | — | — | — | — |

### `fft_real_f64`

| size | sonium | rustfft | accelerate | gap |
|---|---|---|---|---|
| 512 | — | — | — | — |
| 1024 | — | — | — | — |
| 2048 | — | — | — | — |
| 4096 | — | — | — | — |
| 1048576 | — | — | — | — |
| 2097152 | — | — | — | — |
| 4194304 | — | — | — | — |
