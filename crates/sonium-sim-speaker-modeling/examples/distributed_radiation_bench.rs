//! Distributed-radiation hot-path benchmark: DtN matvec (`p̄ = D·v`).
//!
//! The DtN application is the per-frequency hot path of the distributed
//! moving-assembly → BEM handoff (dense `m × m` complex matvec, reused
//! buffers, Rayon lanes inside `DtnMap::apply_into`). This bench reports,
//! for aperture sizes 64/256/1024 and 1/2/4 Rayon threads: wall time,
//! effective memory bandwidth (matrix bytes read per apply), and GFLOP/s
//! (8 flops per complex multiply-add), i.e. the roofline coordinates for
//! this memory-bound kernel. It is a CPU bench; the GPU gate is the
//! existing `sonium-sim-speaker` capability matrix (coupled DtN matvec is
//! `Unsupported` there, so no GPU number is claimed).
//!
//! Run with:
//! `cargo run --release -p sonium-sim-speaker-modeling --example distributed_radiation_bench`

use num_complex::Complex64;
use sonium_sim_speaker_modeling::fem_bem_coupling::DtnMap;
use std::time::Instant;

fn bench(n: usize, threads: usize, repeats: usize) -> (f64, f64, f64) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool");
    let mut matrix = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            matrix[i * n + j] = Complex64::new(
                ((i * 31 + j * 17) % 101) as f64 * 0.01,
                ((i * 13 + j * 7) % 89) as f64 * 0.01,
            );
        }
    }
    let dtn = DtnMap { frequency_hz: 1_000.0, n, matrix };
    let velocity: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new(1.0 + 0.001 * i as f64, 0.0))
        .collect();
    let mut out = vec![Complex64::new(0.0, 0.0); n];
    pool.install(|| {
        dtn.apply_into(&velocity, &mut out);
        let start = Instant::now();
        for _ in 0..repeats {
            dtn.apply_into(&velocity, &mut out);
            std::hint::black_box(&out);
        }
        let seconds = start.elapsed().as_secs_f64() / repeats as f64;
        let bytes = (n * n * 16) as f64;
        let flops = 8.0 * (n * n) as f64;
        (seconds * 1.0e9, bytes / seconds / 1.0e9, flops / seconds / 1.0e9)
    })
}

fn main() {
    println!("DtN matvec bench (complex128, matrix bytes read per apply)");
    println!("{:>6} {:>8} {:>12} {:>12} {:>12}", "m", "threads", "ns/apply", "GB/s", "GFLOP/s");
    for n in [64usize, 256, 1024] {
        for threads in [1usize, 2, 4] {
            let repeats = if n >= 1024 { 20 } else { 200 };
            let (ns, gbs, gflops) = bench(n, threads, repeats);
            println!("{n:>6} {threads:>8} {ns:>12.0} {gbs:>12.2} {gflops:>12.2}");
        }
    }
}
