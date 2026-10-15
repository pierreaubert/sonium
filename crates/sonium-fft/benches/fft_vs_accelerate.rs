//! Benchmark sonium-fft against Apple Accelerate/vDSP complex DFT.
//!
//! Run on macOS:
//! `cargo bench -p sonium-fft --bench fft_vs_accelerate`

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_os = "macos")]
use criterion::BenchmarkId;
#[cfg(not(target_arch = "wasm32"))]
use criterion::{Criterion, criterion_group, criterion_main};
#[cfg(target_os = "macos")]
use num_complex::Complex64;
#[cfg(target_os = "macos")]
use sonium_fft::rustfft_compat::FftPlanner as OurFftPlanner;

#[cfg(target_os = "macos")]
fn sizes() -> &'static [usize] {
    &[64, 256, 1024, 2048, 4096, 16384]
}

#[cfg(target_os = "macos")]
fn audio_sizes() -> &'static [usize] {
    &[256, 1024, 2048, 4096]
}

#[cfg(target_os = "macos")]
fn make_input(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|i| Complex64::new((i as f64 * 0.0123).sin(), (i as f64 * 0.0177).cos()))
        .collect()
}

#[cfg(target_os = "macos")]
mod accelerate {
    use super::*;
    use std::ffi::c_void;
    use std::ptr;

    const VDSP_DFT_FORWARD: i32 = 1;
    const VDSP_DFT_INTERLEAVED_COMPLEX_TO_COMPLEX: bool = false;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub(super) struct DspDoubleComplex {
        real: f64,
        imag: f64,
    }

    type VdspLength = usize;
    type VdspDftDirection = i32;
    type VdspDftRealToComplex = bool;
    type VdspDftInterleavedSetupD = *mut c_void;

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn vDSP_DFT_Interleaved_CreateSetupD(
            previous: VdspDftInterleavedSetupD,
            length: VdspLength,
            direction: VdspDftDirection,
            real_to_complex: VdspDftRealToComplex,
        ) -> VdspDftInterleavedSetupD;

        fn vDSP_DFT_Interleaved_ExecuteD(
            setup: VdspDftInterleavedSetupD,
            input: *const DspDoubleComplex,
            output: *mut DspDoubleComplex,
        );

        fn vDSP_DFT_Interleaved_DestroySetupD(setup: VdspDftInterleavedSetupD);
    }

    pub struct VdspDft {
        setup: VdspDftInterleavedSetupD,
    }

    impl VdspDft {
        pub fn new(n: usize) -> Option<Self> {
            let setup = unsafe {
                vDSP_DFT_Interleaved_CreateSetupD(
                    ptr::null_mut(),
                    n,
                    VDSP_DFT_FORWARD,
                    VDSP_DFT_INTERLEAVED_COMPLEX_TO_COMPLEX,
                )
            };
            (!setup.is_null()).then_some(Self { setup })
        }

        pub fn process(&self, input: &[DspDoubleComplex], output: &mut [DspDoubleComplex]) {
            debug_assert_eq!(input.len(), output.len());
            unsafe {
                vDSP_DFT_Interleaved_ExecuteD(self.setup, input.as_ptr(), output.as_mut_ptr());
            }
        }
    }

    impl Drop for VdspDft {
        fn drop(&mut self) {
            unsafe {
                vDSP_DFT_Interleaved_DestroySetupD(self.setup);
            }
        }
    }

    pub fn make_input(input: &[Complex64]) -> Vec<DspDoubleComplex> {
        input
            .iter()
            .map(|c| DspDoubleComplex {
                real: c.re,
                imag: c.im,
            })
            .collect()
    }
}

#[cfg(target_os = "macos")]
fn bench_single_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("single_f64");
    for &n in sizes() {
        let input = make_input(n);

        let mut ours = OurFftPlanner::<f64>::new();
        let p_ours = ours.plan_fft_forward(n);
        g.bench_with_input(BenchmarkId::new("sonium-fft", n), &n, |b, _| {
            let mut buf = input.clone();
            b.iter(|| {
                buf.copy_from_slice(&input);
                p_ours.process(&mut buf);
                std::hint::black_box(&buf);
            });
        });

        if let Some(p_vdsp) = accelerate::VdspDft::new(n) {
            let input_vdsp = accelerate::make_input(&input);
            g.bench_with_input(BenchmarkId::new("accelerate-vdsp", n), &n, |b, _| {
                let mut buf = input_vdsp.clone();
                let mut out = vec![Default::default(); n];
                b.iter(|| {
                    buf.copy_from_slice(&input_vdsp);
                    p_vdsp.process(&buf, &mut out);
                    std::hint::black_box(&out);
                });
            });
        }
    }
    g.finish();
}

#[cfg(target_os = "macos")]
fn bench_audio_rate_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("audio_rate_f64");
    for &n in audio_sizes() {
        let input = make_input(n);

        let mut ours = OurFftPlanner::<f64>::new();
        let p_ours = ours.plan_fft_forward(n);
        g.bench_with_input(BenchmarkId::new("sonium-fft", n), &n, |b, _| {
            let mut buf = input.clone();
            b.iter(|| {
                buf.copy_from_slice(&input);
                p_ours.process(&mut buf);
                std::hint::black_box(&buf);
            });
        });

        if let Some(p_vdsp) = accelerate::VdspDft::new(n) {
            let input_vdsp = accelerate::make_input(&input);
            g.bench_with_input(BenchmarkId::new("accelerate-vdsp", n), &n, |b, _| {
                let mut buf = input_vdsp.clone();
                let mut out = vec![Default::default(); n];
                b.iter(|| {
                    buf.copy_from_slice(&input_vdsp);
                    p_vdsp.process(&buf, &mut out);
                    std::hint::black_box(&out);
                });
            });
        }
    }
    g.finish();
}

#[cfg(target_os = "macos")]
fn bench_batched_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("batched_f64_x16");
    let batch = 16usize;
    for &n in audio_sizes() {
        let input = make_input(n * batch);

        let mut ours = OurFftPlanner::<f64>::new();
        let p_ours = ours.plan_fft_forward(n);
        g.bench_with_input(BenchmarkId::new("sonium-fft", n), &n, |b, _| {
            let mut buf = input.clone();
            b.iter(|| {
                buf.copy_from_slice(&input);
                p_ours.process(&mut buf);
                std::hint::black_box(&buf);
            });
        });

        if let Some(p_vdsp) = accelerate::VdspDft::new(n) {
            let input_vdsp = accelerate::make_input(&input);
            g.bench_with_input(BenchmarkId::new("accelerate-vdsp", n), &n, |b, _| {
                let mut buf = input_vdsp.clone();
                let mut out = vec![Default::default(); n * batch];
                b.iter(|| {
                    buf.copy_from_slice(&input_vdsp);
                    for (input_chunk, output_chunk) in
                        buf.chunks_exact(n).zip(out.chunks_exact_mut(n))
                    {
                        p_vdsp.process(input_chunk, output_chunk);
                    }
                    std::hint::black_box(&out);
                });
            });
        }
    }
    g.finish();
}

#[cfg(target_os = "macos")]
fn bench_plan_build_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("plan_build_f64");
    for &n in sizes() {
        g.bench_with_input(BenchmarkId::new("sonium-fft", n), &n, |b, &n| {
            b.iter(|| {
                let mut planner = OurFftPlanner::<f64>::new();
                let p = planner.plan_fft_forward(n);
                std::hint::black_box(p);
            });
        });

        g.bench_with_input(BenchmarkId::new("accelerate-vdsp", n), &n, |b, &n| {
            b.iter(|| {
                let p = accelerate::VdspDft::new(n).expect("vDSP DFT setup unavailable");
                std::hint::black_box(p);
            });
        });
    }
    g.finish();
}

#[cfg(all(not(target_os = "macos"), not(target_arch = "wasm32")))]
fn bench_single_f64(_c: &mut Criterion) {}

#[cfg(all(not(target_os = "macos"), not(target_arch = "wasm32")))]
fn bench_audio_rate_f64(_c: &mut Criterion) {}

#[cfg(all(not(target_os = "macos"), not(target_arch = "wasm32")))]
fn bench_batched_f64(_c: &mut Criterion) {}

#[cfg(all(not(target_os = "macos"), not(target_arch = "wasm32")))]
fn bench_plan_build_f64(_c: &mut Criterion) {}

#[cfg(not(target_arch = "wasm32"))]
criterion_group!(
    benches,
    bench_single_f64,
    bench_audio_rate_f64,
    bench_batched_f64,
    bench_plan_build_f64
);
#[cfg(not(target_arch = "wasm32"))]
criterion_main!(benches);
