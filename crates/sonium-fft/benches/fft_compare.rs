//! FFT comparison: sonium-fft vs rustfft/realfft vs Apple Accelerate.
//!
//! Groups `fft_complex_f32`, `fft_complex_f64`, `fft_real_f32`,
//! `fft_real_f64`; one single forward transform per timing at audio/small
//! sizes (512, 1024, 2048, 4096) and large sizes (1 Mi, 2 Mi, 4 Mi points).
//!
//! Arms:
//!
//! - **sonium** — native [`sonium_fft::FftPlanner`] / [`sonium_fft::RealFftPlanner`]
//!   with caller-provided scratch (steady-state, allocation-free).
//! - **rustfft** — `rustfft` (complex) and `realfft` (real, same author,
//!   rustfft-backed) with caller-provided scratch. Recorded on every
//!   platform; the vendor column on Linux.
//! - **accelerate** — Apple Accelerate `vDSP_DFT_Interleaved`
//!   complex-to-complex and real-to-complex DFT (macOS only; the vendor
//!   column on Apple Silicon).
//!
//! All arms time unnormalised forward DFTs (sign `-1`, no `1/N`), so no
//! rescaling enters the timed path. A setup-time agreement check compares
//! each vendor arm against sonium on the same input and prints a warning
//! to stderr when spectra disagree beyond rounding (a smoke signal, not a
//! gate — correctness gates live in the test suite).
//!
//! This bench feeds `perf/<platform>-fft.md` via `scripts/perf_cells.py`
//! (`just perf-cell <platform> fft`). Filter with criterion args, e.g.
//! `-- fft_complex_f64` for one group or `-- 512` for one size everywhere.
//!
//! Run: `cargo bench -p sonium-fft --bench fft_compare`.

// Rust guideline compliant 2026-02-21

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
#[cfg(not(target_arch = "wasm32"))]
use num_complex::{Complex32, Complex64};
#[cfg(not(target_arch = "wasm32"))]
use sonium_fft::{FftDirection, FftPlanner, RealFftPlanner};

/// Audio/small sizes: one STFT frame each.
#[cfg(not(target_arch = "wasm32"))]
const AUDIO_SIZES: &[usize] = &[512, 1024, 2048, 4096];

/// Large sizes: full-signal / convolution-grade transforms.
#[cfg(not(target_arch = "wasm32"))]
const LARGE_SIZES: &[usize] = &[1 << 20, 1 << 21, 1 << 22];

#[cfg(not(target_arch = "wasm32"))]
fn all_sizes() -> Vec<usize> {
    AUDIO_SIZES
        .iter()
        .chain(LARGE_SIZES.iter())
        .copied()
        .collect()
}

#[cfg(not(target_arch = "wasm32"))]
fn make_complex_input_f64(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|i| Complex64::new((i as f64 * 0.0123).sin(), (i as f64 * 0.0177).cos()))
        .collect()
}

#[cfg(not(target_arch = "wasm32"))]
fn make_complex_input_f32(n: usize) -> Vec<Complex32> {
    (0..n)
        .map(|i| Complex32::new((i as f32 * 0.0123).sin(), (i as f32 * 0.0177).cos()))
        .collect()
}

#[cfg(not(target_arch = "wasm32"))]
fn make_real_input_f64(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64 * 0.0123).sin()).collect()
}

#[cfg(not(target_arch = "wasm32"))]
fn make_real_input_f32(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * 0.0123).sin()).collect()
}

/// Max absolute difference between two spectra; `None` on length mismatch.
#[cfg(not(target_arch = "wasm32"))]
fn max_abs_diff(left: &[Complex64], right: &[Complex64]) -> Option<f64> {
    if left.len() != right.len() {
        return None;
    }
    Some(
        left.iter()
            .zip(right.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max),
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn max_abs_diff_f32(left: &[Complex32], right: &[Complex32]) -> Option<f32> {
    if left.len() != right.len() {
        return None;
    }
    Some(
        left.iter()
            .zip(right.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f32::max),
    )
}

/// Warn (never fail) when a vendor arm disagrees with sonium beyond
/// rounding. Unnormalised forward DFT magnitudes grow with `n`, so the
/// budget scales linearly.
#[cfg(not(target_arch = "wasm32"))]
fn check_complex_f64(
    group: &str,
    n: usize,
    vendor: &str,
    ours: &[Complex64],
    theirs: &[Complex64],
) {
    let budget = n as f64 * 1.0e-12;
    match max_abs_diff(ours, theirs) {
        Some(diff) if diff <= budget => {}
        other => eprintln!(
            "fft_compare WARNING: {group}/{n} sonium-vs-{vendor} max-abs-diff {other:?} exceeds budget {budget:.3e}"
        ),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn check_complex_f32(
    group: &str,
    n: usize,
    vendor: &str,
    ours: &[Complex32],
    theirs: &[Complex32],
) {
    let budget = n as f32 * 1.0e-6;
    match max_abs_diff_f32(ours, theirs) {
        Some(diff) if diff <= budget => {}
        other => eprintln!(
            "fft_compare WARNING: {group}/{n} sonium-vs-{vendor} max-abs-diff {other:?} exceeds budget {budget:.3e}"
        ),
    }
}

// ---------------------------------------------------------------------
// Accelerate FFI (macOS only).
//
// `vDSP_DFT_Interleaved_*`: interleaved complex DFT for arbitrary
// lengths, complex-to-complex (`real_to_complex = false`) and
// real-to-complex (`real_to_complex = true`). Direction: forward = 1.
// ---------------------------------------------------------------------

#[cfg(target_os = "macos")]
#[allow(non_snake_case)]
mod accelerate {
    use std::ffi::c_void;
    use std::ptr;

    use num_complex::{Complex32, Complex64};

    const VDSP_DFT_FORWARD: i32 = 1;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct DspComplex {
        pub real: f32,
        pub imag: f32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct DspDoubleComplex {
        pub real: f64,
        pub imag: f64,
    }

    type VdspLength = usize;
    type VdspDftDirection = i32;
    type VdspDftRealToComplex = bool;
    type VdspDftInterleavedSetup = *mut c_void;
    type VdspDftInterleavedSetupD = *mut c_void;

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn vDSP_DFT_Interleaved_CreateSetup(
            previous: VdspDftInterleavedSetup,
            length: VdspLength,
            direction: VdspDftDirection,
            real_to_complex: VdspDftRealToComplex,
        ) -> VdspDftInterleavedSetup;

        fn vDSP_DFT_Interleaved_Execute(
            setup: VdspDftInterleavedSetup,
            input: *const DspComplex,
            output: *mut DspComplex,
        );

        fn vDSP_DFT_Interleaved_DestroySetup(setup: VdspDftInterleavedSetup);

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
        setup: VdspDftInterleavedSetup,
    }

    impl VdspDft {
        pub fn new_complex(n: usize) -> Option<Self> {
            let setup = unsafe {
                vDSP_DFT_Interleaved_CreateSetup(ptr::null_mut(), n, VDSP_DFT_FORWARD, false)
            };
            (!setup.is_null()).then_some(Self { setup })
        }

        pub fn new_real(n: usize) -> Option<Self> {
            let setup = unsafe {
                vDSP_DFT_Interleaved_CreateSetup(ptr::null_mut(), n, VDSP_DFT_FORWARD, true)
            };
            (!setup.is_null()).then_some(Self { setup })
        }

        pub fn process(&self, input: &[DspComplex], output: &mut [DspComplex]) {
            debug_assert_eq!(input.len(), output.len());
            unsafe {
                vDSP_DFT_Interleaved_Execute(self.setup, input.as_ptr(), output.as_mut_ptr());
            }
        }

        /// Real-to-complex transform of `n` real samples. The output
        /// buffer holds `n` complex slots (an upper bound over the
        /// vendor's real-output packing: DC..Nyquist always land in the
        /// first `n/2 + 1` slots); over-allocation keeps the call safe
        /// however the vendor packs the redundant half.
        pub fn process_real(&self, input: &[f32], output: &mut [DspComplex]) {
            debug_assert_eq!(output.len(), input.len());
            unsafe {
                vDSP_DFT_Interleaved_Execute(
                    self.setup,
                    input.as_ptr() as *const DspComplex,
                    output.as_mut_ptr(),
                );
            }
        }
    }

    impl Drop for VdspDft {
        fn drop(&mut self) {
            unsafe {
                vDSP_DFT_Interleaved_DestroySetup(self.setup);
            }
        }
    }

    pub struct VdspDftD {
        setup: VdspDftInterleavedSetupD,
    }

    impl VdspDftD {
        pub fn new_complex(n: usize) -> Option<Self> {
            let setup = unsafe {
                vDSP_DFT_Interleaved_CreateSetupD(ptr::null_mut(), n, VDSP_DFT_FORWARD, false)
            };
            (!setup.is_null()).then_some(Self { setup })
        }

        pub fn new_real(n: usize) -> Option<Self> {
            let setup = unsafe {
                vDSP_DFT_Interleaved_CreateSetupD(ptr::null_mut(), n, VDSP_DFT_FORWARD, true)
            };
            (!setup.is_null()).then_some(Self { setup })
        }

        pub fn process(&self, input: &[DspDoubleComplex], output: &mut [DspDoubleComplex]) {
            debug_assert_eq!(input.len(), output.len());
            unsafe {
                vDSP_DFT_Interleaved_ExecuteD(self.setup, input.as_ptr(), output.as_mut_ptr());
            }
        }

        /// Real-to-complex transform; see [`VdspDft::process_real`] for
        /// the output-sizing rationale.
        pub fn process_real(&self, input: &[f64], output: &mut [DspDoubleComplex]) {
            debug_assert_eq!(output.len(), input.len());
            unsafe {
                vDSP_DFT_Interleaved_ExecuteD(
                    self.setup,
                    input.as_ptr() as *const DspDoubleComplex,
                    output.as_mut_ptr(),
                );
            }
        }
    }

    impl Drop for VdspDftD {
        fn drop(&mut self) {
            unsafe {
                vDSP_DFT_Interleaved_DestroySetupD(self.setup);
            }
        }
    }

    pub fn to_interleaved_f32(input: &[Complex32]) -> Vec<DspComplex> {
        input
            .iter()
            .map(|c| DspComplex {
                real: c.re,
                imag: c.im,
            })
            .collect()
    }

    pub fn to_interleaved_f64(input: &[Complex64]) -> Vec<DspDoubleComplex> {
        input
            .iter()
            .map(|c| DspDoubleComplex {
                real: c.re,
                imag: c.im,
            })
            .collect()
    }

    pub fn from_interleaved_f32(output: &[DspComplex]) -> Vec<Complex32> {
        output
            .iter()
            .map(|c| Complex32::new(c.real, c.imag))
            .collect()
    }

    pub fn from_interleaved_f64(output: &[DspDoubleComplex]) -> Vec<Complex64> {
        output
            .iter()
            .map(|c| Complex64::new(c.real, c.imag))
            .collect()
    }
}

// ---------------------------------------------------------------------
// Complex transforms.
// ---------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
fn bench_complex_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("fft_complex_f64");
    for n in all_sizes() {
        g.throughput(Throughput::Elements(n as u64));
        let input = make_complex_input_f64(n);

        let planner = FftPlanner::<f64>::new();
        let ours = planner.plan(n, FftDirection::Forward);
        let mut our_scratch = vec![Complex64::new(0.0, 0.0); ours.scratch_len()];
        let mut ours_out = input.clone();
        ours.process_with_scratch(&mut ours_out, &mut our_scratch);

        g.bench_with_input(BenchmarkId::new("sonium", n), &n, |b, _| {
            let mut buf = input.clone();
            b.iter(|| {
                buf.copy_from_slice(&input);
                ours.process_with_scratch(&mut buf, &mut our_scratch);
                std::hint::black_box(&buf);
            });
        });

        let mut theirs_planner = rustfft::FftPlanner::<f64>::new();
        let theirs = theirs_planner.plan_fft_forward(n);
        let mut their_scratch = vec![Complex64::new(0.0, 0.0); theirs.get_inplace_scratch_len()];
        let mut theirs_out = input.clone();
        theirs.process_with_scratch(&mut theirs_out, &mut their_scratch);
        check_complex_f64("fft_complex_f64", n, "rustfft", &ours_out, &theirs_out);

        g.bench_with_input(BenchmarkId::new("rustfft", n), &n, |b, _| {
            let mut buf = input.clone();
            b.iter(|| {
                buf.copy_from_slice(&input);
                theirs.process_with_scratch(&mut buf, &mut their_scratch);
                std::hint::black_box(&buf);
            });
        });

        #[cfg(target_os = "macos")]
        if let Some(vdsp) = accelerate::VdspDftD::new_complex(n) {
            let input_vdsp = accelerate::to_interleaved_f64(&input);
            let mut vdsp_out = vec![Default::default(); n];
            vdsp.process(&input_vdsp, &mut vdsp_out);
            check_complex_f64(
                "fft_complex_f64",
                n,
                "accelerate",
                &ours_out,
                &accelerate::from_interleaved_f64(&vdsp_out),
            );
            g.bench_with_input(BenchmarkId::new("accelerate", n), &n, |b, _| {
                let mut buf = input_vdsp.clone();
                let mut out = vec![Default::default(); n];
                b.iter(|| {
                    buf.copy_from_slice(&input_vdsp);
                    vdsp.process(&buf, &mut out);
                    std::hint::black_box(&out);
                });
            });
        }
    }
    g.finish();
}

#[cfg(not(target_arch = "wasm32"))]
fn bench_complex_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("fft_complex_f32");
    for n in all_sizes() {
        g.throughput(Throughput::Elements(n as u64));
        let input = make_complex_input_f32(n);

        let planner = FftPlanner::<f32>::new();
        let ours = planner.plan(n, FftDirection::Forward);
        let mut our_scratch = vec![Complex32::new(0.0, 0.0); ours.scratch_len()];
        let mut ours_out = input.clone();
        ours.process_with_scratch(&mut ours_out, &mut our_scratch);

        g.bench_with_input(BenchmarkId::new("sonium", n), &n, |b, _| {
            let mut buf = input.clone();
            b.iter(|| {
                buf.copy_from_slice(&input);
                ours.process_with_scratch(&mut buf, &mut our_scratch);
                std::hint::black_box(&buf);
            });
        });

        let mut theirs_planner = rustfft::FftPlanner::<f32>::new();
        let theirs = theirs_planner.plan_fft_forward(n);
        let mut their_scratch = vec![Complex32::new(0.0, 0.0); theirs.get_inplace_scratch_len()];
        let mut theirs_out = input.clone();
        theirs.process_with_scratch(&mut theirs_out, &mut their_scratch);
        check_complex_f32("fft_complex_f32", n, "rustfft", &ours_out, &theirs_out);

        g.bench_with_input(BenchmarkId::new("rustfft", n), &n, |b, _| {
            let mut buf = input.clone();
            b.iter(|| {
                buf.copy_from_slice(&input);
                theirs.process_with_scratch(&mut buf, &mut their_scratch);
                std::hint::black_box(&buf);
            });
        });

        #[cfg(target_os = "macos")]
        if let Some(vdsp) = accelerate::VdspDft::new_complex(n) {
            let input_vdsp = accelerate::to_interleaved_f32(&input);
            let mut vdsp_out = vec![Default::default(); n];
            vdsp.process(&input_vdsp, &mut vdsp_out);
            check_complex_f32(
                "fft_complex_f32",
                n,
                "accelerate",
                &ours_out,
                &accelerate::from_interleaved_f32(&vdsp_out),
            );
            g.bench_with_input(BenchmarkId::new("accelerate", n), &n, |b, _| {
                let mut buf = input_vdsp.clone();
                let mut out = vec![Default::default(); n];
                b.iter(|| {
                    buf.copy_from_slice(&input_vdsp);
                    vdsp.process(&buf, &mut out);
                    std::hint::black_box(&out);
                });
            });
        }
    }
    g.finish();
}

// ---------------------------------------------------------------------
// Real transforms.
// ---------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
fn bench_real_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("fft_real_f64");
    for n in all_sizes() {
        g.throughput(Throughput::Elements(n as u64));
        let input = make_real_input_f64(n);
        let spectrum_len = n / 2 + 1;

        let planner = RealFftPlanner::<f64>::new();
        let ours = planner.plan_fft_forward(n);
        let mut our_scratch = vec![Complex64::new(0.0, 0.0); ours.scratch_len()];
        let mut ours_out = vec![Complex64::new(0.0, 0.0); spectrum_len];
        ours.forward_with_scratch(&input, &mut ours_out, &mut our_scratch);

        g.bench_with_input(BenchmarkId::new("sonium", n), &n, |b, _| {
            let mut out = vec![Complex64::new(0.0, 0.0); spectrum_len];
            b.iter(|| {
                // `forward_with_scratch` borrows its input immutably, so no
                // refresh copy is needed; this is the call's true cost.
                ours.forward_with_scratch(&input, &mut out, &mut our_scratch);
                std::hint::black_box(&out);
            });
        });

        let mut theirs_planner = realfft::RealFftPlanner::<f64>::new();
        let theirs = theirs_planner.plan_fft_forward(n);
        let mut their_scratch = theirs.make_scratch_vec();
        let mut theirs_in = input.clone();
        let mut theirs_out = theirs.make_output_vec();
        theirs
            .process_with_scratch(&mut theirs_in, &mut theirs_out, &mut their_scratch)
            .expect("realfft forward must succeed");
        check_complex_f64("fft_real_f64", n, "rustfft", &ours_out, &theirs_out);

        g.bench_with_input(BenchmarkId::new("rustfft", n), &n, |b, _| {
            let mut buf = input.clone();
            let mut out = theirs.make_output_vec();
            b.iter(|| {
                // realfft consumes its input as scratch; refresh per iteration.
                buf.copy_from_slice(&input);
                theirs
                    .process_with_scratch(&mut buf, &mut out, &mut their_scratch)
                    .expect("realfft forward must succeed");
                std::hint::black_box(&out);
            });
        });

        #[cfg(target_os = "macos")]
        if let Some(vdsp) = accelerate::VdspDftD::new_real(n) {
            let mut vdsp_out = vec![Default::default(); n];
            vdsp.process_real(&input, &mut vdsp_out);
            let vdsp_spectrum = accelerate::from_interleaved_f64(&vdsp_out);
            check_complex_f64(
                "fft_real_f64",
                n,
                "accelerate",
                &ours_out,
                &vdsp_spectrum[..spectrum_len],
            );
            g.bench_with_input(BenchmarkId::new("accelerate", n), &n, |b, _| {
                let mut buf = input.clone();
                let mut out = vec![Default::default(); n];
                b.iter(|| {
                    buf.copy_from_slice(&input);
                    vdsp.process_real(&buf, &mut out);
                    std::hint::black_box(&out);
                });
            });
        }
    }
    g.finish();
}

#[cfg(not(target_arch = "wasm32"))]
fn bench_real_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("fft_real_f32");
    for n in all_sizes() {
        g.throughput(Throughput::Elements(n as u64));
        let input = make_real_input_f32(n);
        let spectrum_len = n / 2 + 1;

        let planner = RealFftPlanner::<f32>::new();
        let ours = planner.plan_fft_forward(n);
        let mut our_scratch = vec![Complex32::new(0.0, 0.0); ours.scratch_len()];
        let mut ours_out = vec![Complex32::new(0.0, 0.0); spectrum_len];
        ours.forward_with_scratch(&input, &mut ours_out, &mut our_scratch);

        g.bench_with_input(BenchmarkId::new("sonium", n), &n, |b, _| {
            let mut out = vec![Complex32::new(0.0, 0.0); spectrum_len];
            b.iter(|| {
                ours.forward_with_scratch(&input, &mut out, &mut our_scratch);
                std::hint::black_box(&out);
            });
        });

        let mut theirs_planner = realfft::RealFftPlanner::<f32>::new();
        let theirs = theirs_planner.plan_fft_forward(n);
        let mut their_scratch = theirs.make_scratch_vec();
        let mut theirs_in = input.clone();
        let mut theirs_out = theirs.make_output_vec();
        theirs
            .process_with_scratch(&mut theirs_in, &mut theirs_out, &mut their_scratch)
            .expect("realfft forward must succeed");
        check_complex_f32("fft_real_f32", n, "rustfft", &ours_out, &theirs_out);

        g.bench_with_input(BenchmarkId::new("rustfft", n), &n, |b, _| {
            let mut buf = input.clone();
            let mut out = theirs.make_output_vec();
            b.iter(|| {
                buf.copy_from_slice(&input);
                theirs
                    .process_with_scratch(&mut buf, &mut out, &mut their_scratch)
                    .expect("realfft forward must succeed");
                std::hint::black_box(&out);
            });
        });

        #[cfg(target_os = "macos")]
        if let Some(vdsp) = accelerate::VdspDft::new_real(n) {
            let mut vdsp_out = vec![Default::default(); n];
            vdsp.process_real(&input, &mut vdsp_out);
            let vdsp_spectrum = accelerate::from_interleaved_f32(&vdsp_out);
            check_complex_f32(
                "fft_real_f32",
                n,
                "accelerate",
                &ours_out,
                &vdsp_spectrum[..spectrum_len],
            );
            g.bench_with_input(BenchmarkId::new("accelerate", n), &n, |b, _| {
                let mut buf = input.clone();
                let mut out = vec![Default::default(); n];
                b.iter(|| {
                    buf.copy_from_slice(&input);
                    vdsp.process_real(&buf, &mut out);
                    std::hint::black_box(&out);
                });
            });
        }
    }
    g.finish();
}

#[cfg(not(target_arch = "wasm32"))]
criterion_group!(
    benches,
    bench_complex_f64,
    bench_complex_f32,
    bench_real_f64,
    bench_real_f32
);
#[cfg(not(target_arch = "wasm32"))]
criterion_main!(benches);
