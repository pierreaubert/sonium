# x86-64 (Linux) — FFT

_Generated 2026-10-15T11:45:44+00:00 from commit `28926daf16f6-dirty`._

- status: measured
- host: Linux 6.8.0-146-generic x86_64 · 128 logical CPUs · mem 251 GB
- rustc: rustc 1.99.0 (b940084d7 2026-09-28)

## Commands

```text
mbx bench -p sonium-fft --bench fft_compare
```
## Results

Single unnormalised forward transforms; gap = sonium ÷ rustfft (1.00× is parity). The non-vendor arm is bonus context (— where the harness has no such arm).

### `fft_complex_f32`

| size | sonium | rustfft | accelerate | gap |
|---|---|---|---|---|
| 512 | 689.4 ns | 428.8 ns | — | 1.61× |
| 1024 | 1.32 µs | 1.05 µs | — | 1.26× |
| 2048 | 2.99 µs | 2.37 µs | — | 1.26× |
| 4096 | 7.06 µs | 5.69 µs | — | 1.24× |
| 1048576 | 6.378 ms | 4.106 ms | — | 1.55× |
| 2097152 | 17.025 ms | 9.328 ms | — | 1.83× |
| 4194304 | 34.090 ms | 20.810 ms | — | 1.64× |

### `fft_complex_f64`

| size | sonium | rustfft | accelerate | gap |
|---|---|---|---|---|
| 512 | 1.14 µs | 830.5 ns | — | 1.37× |
| 1024 | 2.26 µs | 2.12 µs | — | 1.06× |
| 2048 | 9.81 µs | 4.89 µs | — | 2.01× |
| 4096 | 18.13 µs | 11.15 µs | — | 1.63× |
| 1048576 | 13.890 ms | 8.890 ms | — | 1.56× |
| 2097152 | 34.186 ms | 19.091 ms | — | 1.79× |
| 4194304 | 69.465 ms | 40.887 ms | — | 1.70× |

### `fft_real_f32`

| size | sonium | rustfft | accelerate | gap |
|---|---|---|---|---|
| 512 | 2.07 µs | 428.4 ns | — | 4.84× |
| 1024 | 4.24 µs | 938.8 ns | — | 4.51× |
| 2048 | 8.33 µs | 2.05 µs | — | 4.06× |
| 4096 | 17.24 µs | 4.35 µs | — | 3.96× |
| 1048576 | 5.591 ms | 2.163 ms | — | 2.58× |
| 2097152 | 12.405 ms | 5.010 ms | — | 2.48× |
| 4194304 | 31.120 ms | 11.019 ms | — | 2.82× |

### `fft_real_f64`

| size | sonium | rustfft | accelerate | gap |
|---|---|---|---|---|
| 512 | 2.29 µs | 587.2 ns | — | 3.90× |
| 1024 | 4.61 µs | 1.36 µs | — | 3.39× |
| 2048 | 9.33 µs | 3.17 µs | — | 2.95× |
| 4096 | 21.14 µs | 6.84 µs | — | 3.09× |
| 1048576 | 8.481 ms | 4.455 ms | — | 1.90× |
| 2097152 | 21.429 ms | 9.196 ms | — | 2.33× |
| 4194304 | 49.032 ms | 21.166 ms | — | 2.32× |

