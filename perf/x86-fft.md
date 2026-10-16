# x86-64 (Linux) — FFT

_Generated 2026-10-15T14:53:44+00:00 from commit `ca156e4f1fed-dirty`._

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
| 512 | 523.8 ns | 447.7 ns | — | 1.17× |
| 1024 | 1.16 µs | 1.41 µs | — | 0.83× |
| 2048 | 3.16 µs | 2.37 µs | — | 1.33× |
| 4096 | 7.42 µs | 6.42 µs | — | 1.15× |
| 1048576 | 6.640 ms | 4.360 ms | — | 1.52× |
| 2097152 | 18.247 ms | 9.923 ms | — | 1.84× |
| 4194304 | 36.143 ms | 21.846 ms | — | 1.65× |

### `fft_complex_f64`

| size | sonium | rustfft | accelerate | gap |
|---|---|---|---|---|
| 512 | 1.04 µs | 857.6 ns | — | 1.21× |
| 1024 | 2.35 µs | 2.17 µs | — | 1.09× |
| 2048 | 5.82 µs | 4.96 µs | — | 1.17× |
| 4096 | 22.80 µs | 11.21 µs | — | 2.03× |
| 1048576 | 13.823 ms | 9.904 ms | — | 1.40× |
| 2097152 | 34.305 ms | 23.774 ms | — | 1.44× |
| 4194304 | 76.670 ms | 47.794 ms | — | 1.60× |

### `fft_real_f32`

| size | sonium | rustfft | accelerate | gap |
|---|---|---|---|---|
| 512 | 464.1 ns | 456.4 ns | — | 1.02× |
| 1024 | 793.9 ns | 988.3 ns | — | 0.80× |
| 2048 | 1.59 µs | 2.09 µs | — | 0.76× |
| 4096 | 3.81 µs | 4.44 µs | — | 0.86× |
| 1048576 | 2.123 ms | 2.235 ms | — | 0.95× |
| 2097152 | 6.221 ms | 5.154 ms | — | 1.21× |
| 4194304 | 19.310 ms | 11.131 ms | — | 1.73× |

### `fft_real_f64`

| size | sonium | rustfft | accelerate | gap |
|---|---|---|---|---|
| 512 | 752.9 ns | 609.1 ns | — | 1.24× |
| 1024 | 1.54 µs | 1.40 µs | — | 1.10× |
| 2048 | 3.08 µs | 3.24 µs | — | 0.95× |
| 4096 | 7.52 µs | 7.01 µs | — | 1.07× |
| 1048576 | 6.443 ms | 4.432 ms | — | 1.45× |
| 2097152 | 16.642 ms | 9.970 ms | — | 1.67× |
| 4194304 | 41.008 ms | 21.869 ms | — | 1.88× |

