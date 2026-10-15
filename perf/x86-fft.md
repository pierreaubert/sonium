# x86-64 (Linux) — FFT

_Generated 2026-10-15T07:28:21+00:00 from commit `39fde79e789e-dirty`._

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
| 512 | 3.61 µs | 431.0 ns | — | 8.37× |
| 1024 | 7.46 µs | 1.04 µs | — | 7.16× |
| 2048 | 16.50 µs | 2.38 µs | — | 6.93× |
| 4096 | 34.71 µs | 5.71 µs | — | 6.08× |
| 1048576 | 37.163 ms | 4.288 ms | — | 8.67× |
| 2097152 | 74.080 ms | 9.576 ms | — | 7.74× |
| 4194304 | 149.796 ms | 21.792 ms | — | 6.87× |

### `fft_complex_f64`

| size | sonium | rustfft | accelerate | gap |
|---|---|---|---|---|
| 512 | 2.57 µs | 916.4 ns | — | 2.81× |
| 1024 | 3.86 µs | 2.15 µs | — | 1.79× |
| 2048 | 10.53 µs | 4.93 µs | — | 2.13× |
| 4096 | 22.82 µs | 11.39 µs | — | 2.00× |
| 1048576 | 27.106 ms | 9.502 ms | — | 2.85× |
| 2097152 | 61.091 ms | 20.456 ms | — | 2.99× |
| 4194304 | 125.441 ms | 42.791 ms | — | 2.93× |

### `fft_real_f32`

| size | sonium | rustfft | accelerate | gap |
|---|---|---|---|---|
| 512 | 5.37 µs | 445.0 ns | — | 12.06× |
| 1024 | 11.11 µs | 940.2 ns | — | 11.81× |
| 2048 | 22.49 µs | 3.64 µs | — | 6.18× |
| 4096 | 83.33 µs | 8.21 µs | — | 10.15× |
| 1048576 | 42.176 ms | 4.224 ms | — | 9.98× |
| 2097152 | 88.209 ms | 11.715 ms | — | 7.53× |
| 4194304 | 174.891 ms | 20.173 ms | — | 8.67× |

### `fft_real_f64`

| size | sonium | rustfft | accelerate | gap |
|---|---|---|---|---|
| 512 | 4.85 µs | 606.1 ns | — | 8.01× |
| 1024 | 9.81 µs | 1.39 µs | — | 7.05× |
| 2048 | 18.69 µs | 3.14 µs | — | 5.95× |
| 4096 | 41.11 µs | 7.35 µs | — | 5.59× |
| 1048576 | 22.070 ms | 4.697 ms | — | 4.70× |
| 2097152 | 43.881 ms | 10.043 ms | — | 4.37× |
| 4194304 | 95.474 ms | 21.840 ms | — | 4.37× |

