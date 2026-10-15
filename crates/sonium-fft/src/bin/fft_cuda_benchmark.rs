//! Emit versioned CUDA FFT evidence.
//!
//! The benchmark is deliberately fail-closed: it only measures plans whose
//! execution class is `VendorNative`, and it never turns an unavailable CUDA
//! context into a CPU timing.  Run it with `SONIUM_REQUIRE_CUDA=1` in CI when
//! a real cuFFT result is required.

use libloading::{Library, Symbol};
use num_complex::{Complex, Complex64};
use sonium_compute::{
    Backend, BufferElement, CudaBackend, CudaBuffer, CudaRuntimeStatsSnapshot, OomError,
};
use sonium_fft::{
    BackendDctPlanner, BackendDstPlanner, BackendFftPlanner, BackendRealFftPlanner,
    FftCancellationToken, FftDirection, FftExecution, FftPlanner, FftScalar, LinearConvolutionPlan,
    RealFftPlan, dct2, dct2_adjoint, dct3, dct3_adjoint, dst2, dst2_adjoint, dst3, dst3_adjoint,
};
use std::any::TypeId;
use std::ffi::c_void;
use std::fmt::Write as _;
use std::sync::{Arc, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const SCHEMA_VERSION: u32 = 1;
const SAMPLES: usize = 24;

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    dtype: &'static str,
    n: usize,
    batch: usize,
    stride: usize,
}

type RealOperation<T> = fn(&[T], &mut [T]);
type DeviceRealOperation<T, U> =
    dyn Fn(&CudaBuffer<T>, &mut CudaBuffer<U>) -> sonium_compute::BackendResult<()>;

struct DeviceRealCase<T: BufferElement, U: BufferElement> {
    case: Case,
    tolerance: f64,
    plan_ns: u128,
    execution: FftExecution,
    input: Vec<T>,
    expected: Vec<U>,
    execute: Box<DeviceRealOperation<T, U>>,
    error_fn: fn(&[U], &[U]) -> f64,
    expected_kernel_launches: usize,
}

#[derive(Clone)]
struct Row {
    case: Case,
    status: &'static str,
    execution: &'static str,
    correctness_passed: bool,
    throughput_gate_passed: bool,
    cache_reuse_gate_passed: bool,
    max_abs_error: Option<f64>,
    error: Option<String>,
    plan_ns: Option<u128>,
    cold_ns: Option<u128>,
    samples_ns: Vec<u128>,
    stats: Option<CudaRuntimeStatsSnapshot>,
    /// Raw-cuFFT reference measurement for complex cases (see
    /// `cufft_reference`); `None` where no cuFFT equivalent exists
    /// (real/DCT/DST rows) or the reference was not measured.
    cufft: Option<CufftRef>,
    cufft_error: Option<String>,
}

/// Raw-cuFFT reference timings for one complex case, measured with the
/// same H2D-outside / exec+synchronize discipline as the sonium samples.
/// `overhead_ratio` (sonium median / this median) is derived at emission.
#[derive(Clone)]
struct CufftRef {
    plan_ns: u128,
    cold_ns: u128,
    samples_ns: Vec<u128>,
    max_abs_error: f64,
}

fn required_cuda() -> bool {
    std::env::var("SONIUM_REQUIRE_CUDA")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}

fn json_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

fn json_string(value: &str) -> String {
    format!("\"{}\"", json_escape(value))
}

fn json_optional_string(value: Option<&str>) -> String {
    value.map_or_else(|| "null".to_owned(), json_string)
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = std::process::Command::new(program)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    (!value.is_empty()).then_some(value)
}

fn cuda_hardware() -> (
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
) {
    let query = command_output(
        "nvidia-smi",
        &[
            "--query-gpu=name,driver_version,clocks.sm,clocks.mem",
            "--format=csv,noheader,nounits",
        ],
    );
    let fields = query
        .as_deref()
        .and_then(|line| line.lines().next())
        .map(|line| line.split(',').map(str::trim).collect::<Vec<_>>())
        .unwrap_or_default();
    let device = fields.first().copied().map(str::to_owned);
    let driver = fields.get(1).copied().map(str::to_owned);
    let sm_clock = fields.get(2).copied().map(str::to_owned);
    let memory_clock = fields.get(3).copied().map(str::to_owned);
    (device, driver, sm_clock, memory_clock)
}

fn cuda_toolkit() -> Option<String> {
    if let Ok(version) = std::env::var("CUDARC_CUDA_VERSION") {
        return Some(version);
    }
    command_output("nvcc", &["--version"]).and_then(|output| {
        output.lines().find_map(|line| {
            let release = line.split("release ").nth(1)?;
            Some(release.split(',').next()?.trim().to_owned())
        })
    })
}

fn compiler() -> Option<String> {
    command_output("rustc", &["-Vv"])
}

fn percentile(samples: &[u128], numerator: usize, denominator: usize) -> u128 {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    sorted[((sorted.len() - 1) * numerator) / denominator]
}

fn median(samples: &[u128]) -> u128 {
    percentile(samples, 50, 100)
}

fn max_error<T: FftScalar>(actual: &[Complex<T>], expected: &[Complex<T>]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| {
            let re = (actual.re - expected.re)
                .abs()
                .to_f64()
                .unwrap_or(f64::INFINITY);
            let im = (actual.im - expected.im)
                .abs()
                .to_f64()
                .unwrap_or(f64::INFINITY);
            re.max(im)
        })
        .fold(0.0, f64::max)
}

fn max_real_error<T: FftScalar>(actual: &[T], expected: &[T]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| {
            (*actual - *expected)
                .abs()
                .to_f64()
                .unwrap_or(f64::INFINITY)
        })
        .fold(0.0, f64::max)
}

fn make_real_input<T: FftScalar>(len: usize, seed: u64) -> Vec<T> {
    make_input::<T>(len, seed)
        .into_iter()
        .map(|value| value.re)
        .collect()
}

fn real_reference<T: FftScalar>(
    input: &[T],
    n: usize,
    batch: usize,
    operation: fn(&[T], &mut [T]),
) -> Vec<T> {
    let mut output = vec![T::zero(); input.len()];
    for batch_index in 0..batch {
        let start = batch_index * n;
        operation(&input[start..start + n], &mut output[start..start + n]);
    }
    output
}

fn make_input<T: FftScalar>(len: usize, seed: u64) -> Vec<Complex<T>> {
    let mut state = 0xC0FFEE_u64 ^ seed.rotate_left(17) ^ len as u64;
    (0..len)
        .map(|index| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let re = ((state as f64 / u64::MAX as f64) - 0.5) + index as f64 * 1.0e-7;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let im = (state as f64 / u64::MAX as f64) - 0.5;
            Complex::new(
                T::from_f64(re).expect("f64 is representable by benchmark scalar"),
                T::from_f64(im).expect("f64 is representable by benchmark scalar"),
            )
        })
        .collect()
}

fn cpu_reference<T: FftScalar>(
    input: &[Complex<T>],
    n: usize,
    batch: usize,
    stride: usize,
) -> Vec<Complex<T>> {
    let planner = FftPlanner::<T>::new();
    let plan = planner.plan(n, FftDirection::Forward);
    let mut output = input.to_vec();
    for batch_index in 0..batch {
        let start = batch_index * stride;
        plan.process(&mut output[start..start + n]);
    }
    output
}

fn execute<T: FftScalar>(
    plan: &sonium_fft::DeviceFftPlan<CudaBackend, T>,
    buffer: &mut <CudaBackend as Backend>::Buffer<Complex<T>>,
    batch: usize,
    stride: usize,
) -> sonium_compute::BackendResult<()>
where
    Complex<T>: sonium_compute::BufferElement,
{
    if stride == plan.len() {
        plan.process(buffer)
    } else {
        plan.process_strided(buffer, batch, stride)
    }
}

// ---------------------------------------------------------------------------
// Raw cuFFT reference, loaded dynamically (mirroring the FFTW reference
// test), so the artifact can attribute sonium's device time between the
// vendor kernel and our wrapper (plan cache, workspace, dispatch). The
// reference intentionally uses cuFFT defaults (auto-allocation on, no
// explicit work area): that is what a typical raw-cuFFT caller pays, and
// the asymmetry is what the comparison should expose — sonium's explicit
// workspace management shows up in its own `stats` block instead.
// Timing discipline mirrors the sonium samples exactly (H2D outside the
// timed region, exec + stream synchronize inside), so medians compare
// like for like via `overhead_ratio` at emission.
// ---------------------------------------------------------------------------

type CufftHandle = i32;
type CufftResult = i32;
type CufftComplex = [f32; 2];
type CufftDoubleComplex = [f64; 2];
type CufftPlanMany = unsafe extern "C" fn(
    plan: *mut CufftHandle,
    rank: i32,
    n: *const i32,
    inembed: *const i32,
    istride: i32,
    idist: i32,
    onembed: *const i32,
    ostride: i32,
    odist: i32,
    fft_type: i32,
    batch: i32,
) -> CufftResult;
type CufftSetStream = unsafe extern "C" fn(plan: CufftHandle, stream: *mut c_void) -> CufftResult;
type CufftExecC2C = unsafe extern "C" fn(
    plan: CufftHandle,
    input: *mut CufftComplex,
    output: *mut CufftComplex,
    direction: i32,
) -> CufftResult;
type CufftExecZ2Z = unsafe extern "C" fn(
    plan: CufftHandle,
    input: *mut CufftDoubleComplex,
    output: *mut CufftDoubleComplex,
    direction: i32,
) -> CufftResult;
type CufftDestroy = unsafe extern "C" fn(plan: CufftHandle) -> CufftResult;

const CUFFT_SUCCESS: CufftResult = 0;
const CUFFT_C2C: i32 = 0x29;
const CUFFT_Z2Z: i32 = 0x69;
const CUFFT_FORWARD: i32 = -1;

struct CufftApi {
    _library: Library,
    plan_many: CufftPlanMany,
    set_stream: CufftSetStream,
    exec_c2c: CufftExecC2C,
    exec_z2z: CufftExecZ2Z,
    destroy: CufftDestroy,
}

impl CufftApi {
    fn load() -> Result<Self, String> {
        #[cfg(target_os = "windows")]
        const LIBRARIES: &[&str] = &["cufft64_12.dll", "cufft64_11.dll", "cufft64_10.dll"];
        #[cfg(target_os = "macos")]
        const LIBRARIES: &[&str] = &["libcufft.dylib"];
        #[cfg(all(unix, not(target_os = "macos")))]
        const LIBRARIES: &[&str] = &["libcufft.so.12", "libcufft.so.11", "libcufft.so"];

        let mut last_error = String::from("no cuFFT shared library found");
        for name in LIBRARIES {
            // SAFETY: loading a named optional shared library is the intended
            // dynamic-loading boundary; symbols are copied into function
            // pointers while the Library is retained in `Self`.
            let Ok(library) = (unsafe { Library::new(*name) }) else {
                continue;
            };
            // SAFETY: cuFFT exports these symbols with the declared C ABI.
            let symbols = unsafe {
                (
                    load_symbol::<CufftPlanMany>(&library, b"cufftPlanMany\0"),
                    load_symbol::<CufftSetStream>(&library, b"cufftSetStream\0"),
                    load_symbol::<CufftExecC2C>(&library, b"cufftExecC2C\0"),
                    load_symbol::<CufftExecZ2Z>(&library, b"cufftExecZ2Z\0"),
                    load_symbol::<CufftDestroy>(&library, b"cufftDestroy\0"),
                )
            };
            match symbols {
                (Ok(plan_many), Ok(set_stream), Ok(exec_c2c), Ok(exec_z2z), Ok(destroy)) => {
                    return Ok(Self {
                        _library: library,
                        plan_many,
                        set_stream,
                        exec_c2c,
                        exec_z2z,
                        destroy,
                    });
                }
                _ => last_error = format!("cuFFT library {name} is missing required symbols"),
            }
        }
        Err(last_error)
    }
}

unsafe fn load_symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T, String> {
    // SAFETY: the caller verifies the symbol's ABI through the requested type
    // and retains `library` for the lifetime of the copied function pointer.
    unsafe {
        library
            .get::<T>(name)
            .map(|symbol: Symbol<'_, T>| *symbol)
            .map_err(|error| error.to_string())
    }
}

fn cufft_api() -> Result<&'static Arc<CufftApi>, String> {
    static API: OnceLock<Result<Arc<CufftApi>, String>> = OnceLock::new();
    API.get_or_init(|| CufftApi::load().map(Arc::new))
        .as_ref()
        .map_err(|error| error.clone())
}

/// Destroys the reference plan on scope exit, including error paths.
struct CufftPlanGuard<'a> {
    api: &'a CufftApi,
    handle: CufftHandle,
}

impl Drop for CufftPlanGuard<'_> {
    fn drop(&mut self) {
        if self.handle != 0 {
            // SAFETY: the handle came from `cufftPlanMany` on this API and
            // is destroyed exactly once here.
            unsafe {
                (self.api.destroy)(self.handle);
            }
        }
    }
}

/// Measure raw cuFFT on the same device buffer, layout, and stream as the
/// sonium plan under test (in-place, `embed = [n]`, `dist = stride`).
/// Runs after the sonium runtime-stats snapshot, so it cannot perturb
/// the cache-reuse gates. Errors are reported, never fail the row: the
/// reference is evidence, not a gate.
fn cufft_reference<T: FftScalar>(
    backend: &Arc<CudaBackend>,
    case: Case,
    device: &mut <CudaBackend as Backend>::Buffer<Complex<T>>,
    input: &[Complex<T>],
    expected: &[Complex<T>],
) -> Result<CufftRef, String>
where
    Complex<T>: BufferElement,
{
    use cudarc::driver::DevicePtrMut as _;

    let api = cufft_api()?.as_ref();
    let n_i32 = i32::try_from(case.n).map_err(|_| "case length exceeds cuFFT range".to_owned())?;
    let batch_i32 =
        i32::try_from(case.batch).map_err(|_| "case batch exceeds cuFFT range".to_owned())?;
    let dist_i32 =
        i32::try_from(case.stride).map_err(|_| "case stride exceeds cuFFT range".to_owned())?;
    let embed = [n_i32];
    let is_f32 = TypeId::of::<T>() == TypeId::of::<f32>();
    let fft_type = if is_f32 { CUFFT_C2C } else { CUFFT_Z2Z };

    // The SyncOnDrop guard is scoped out before the timed loops: the
    // owning `CudaBuffer` cannot be freed or reallocated while
    // `cufft_reference` holds `&mut device`, and every launch is drained
    // by `try_synchronize` before return, so the raw pointer stays valid
    // for exactly the executions below (same contract compute's wrapper
    // relies on within one call).
    let ptr = {
        let (ptr, _guard) = device
            .raw_mut()
            .ok_or_else(|| "empty device buffer".to_owned())?
            .device_ptr_mut(backend.stream());
        ptr
    } as *mut c_void;

    let mut handle: CufftHandle = 0;
    let plan_start = Instant::now();
    // SAFETY: `handle` outlives the call; embed arrays are valid for the
    // duration of the call; the plan is destroyed by the guard below.
    let status = unsafe {
        (api.plan_many)(
            &mut handle,
            1,
            embed.as_ptr(),
            embed.as_ptr(),
            1,
            dist_i32,
            embed.as_ptr(),
            1,
            dist_i32,
            fft_type,
            batch_i32,
        )
    };
    if status != CUFFT_SUCCESS {
        return Err(format!("cufftPlanMany status {status}"));
    }
    let plan_ns = plan_start.elapsed().as_nanos();
    let _plan = CufftPlanGuard { api, handle };
    // SAFETY: the handle was just created; the backend stream outlives it.
    let status = unsafe { (api.set_stream)(handle, backend.stream().cu_stream() as *mut c_void) };
    if status != CUFFT_SUCCESS {
        return Err(format!("cufftSetStream status {status}"));
    }

    let exec = |api: &CufftApi| -> Result<(), String> {
        // SAFETY: the plan is bound to the backend stream, the buffer is
        // the live device allocation behind `ptr` (pointer guard held by
        // the caller until after the final synchronize), and element
        // layout matches the `Complex<T>` (re, im) pair layout cuFFT
        // expects for the selected type.
        let status = if is_f32 {
            unsafe {
                (api.exec_c2c)(
                    handle,
                    ptr as *mut CufftComplex,
                    ptr as *mut CufftComplex,
                    CUFFT_FORWARD,
                )
            }
        } else {
            unsafe {
                (api.exec_z2z)(
                    handle,
                    ptr as *mut CufftDoubleComplex,
                    ptr as *mut CufftDoubleComplex,
                    CUFFT_FORWARD,
                )
            }
        };
        if status != CUFFT_SUCCESS {
            return Err(format!("cufftExec status {status}"));
        }
        Ok(())
    };

    backend
        .try_copy_h2d(input, device)
        .map_err(|error| error.to_string())?;
    backend
        .try_synchronize()
        .map_err(|error| error.to_string())?;
    let cold_start = Instant::now();
    exec(api)?;
    backend
        .try_synchronize()
        .map_err(|error| error.to_string())?;
    let cold_ns = cold_start.elapsed().as_nanos();

    let mut samples_ns = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        backend
            .try_copy_h2d(input, device)
            .map_err(|error| error.to_string())?;
        backend
            .try_synchronize()
            .map_err(|error| error.to_string())?;
        let start = Instant::now();
        exec(api)?;
        backend
            .try_synchronize()
            .map_err(|error| error.to_string())?;
        samples_ns.push(start.elapsed().as_nanos());
    }
    let mut actual = vec![Complex::<T>::default(); expected.len()];
    backend
        .try_copy_d2h(device, &mut actual)
        .map_err(|error| error.to_string())?;
    Ok(CufftRef {
        plan_ns,
        cold_ns,
        samples_ns,
        max_abs_error: max_error(&actual, expected),
    })
}

fn benchmark_input_case<T: FftScalar>(
    backend: &Arc<CudaBackend>,
    case: Case,
    tolerance: f64,
    input: Vec<Complex<T>>,
) -> Row
where
    Complex<T>: sonium_compute::BufferElement,
{
    let planner = BackendFftPlanner::<CudaBackend, T>::new(backend.clone());
    let plan_start = Instant::now();
    let plan = planner.plan(case.n, FftDirection::Forward);
    let plan_ns = plan_start.elapsed().as_nanos();
    if plan.execution() != FftExecution::VendorNative {
        return Row {
            case,
            status: "unsupported",
            execution: "host-staged-fallback",
            correctness_passed: false,
            throughput_gate_passed: false,
            cache_reuse_gate_passed: false,
            max_abs_error: None,
            error: Some("backend did not select vendor-native cuFFT execution".to_owned()),
            plan_ns: Some(plan_ns),
            cold_ns: None,
            samples_ns: Vec::new(),
            stats: None,
            cufft: None,
            cufft_error: None,
        };
    }

    let buffer_len = case
        .batch
        .saturating_sub(1)
        .saturating_mul(case.stride)
        .saturating_add(case.n);
    let expected = cpu_reference(&input, case.n, case.batch, case.stride);
    let mut device = match backend.raw_alloc::<Complex<T>>(buffer_len) {
        Ok(buffer) => buffer,
        Err(error) => return allocation_error(case, plan_ns, error),
    };

    // (cold_ns, samples, stats, max_err, cache_reuse, cufft, cufft_error)
    type SteadyMeasurement = (
        u128,
        Vec<u128>,
        CudaRuntimeStatsSnapshot,
        f64,
        bool,
        Option<CufftRef>,
        Option<String>,
    );
    let result = (|| -> Result<SteadyMeasurement, String> {
        backend
            .try_copy_h2d(&input, &mut device)
            .map_err(|error| error.to_string())?;
        backend
            .try_synchronize()
            .map_err(|error| error.to_string())?;
        backend.reset_runtime_stats();
        let planned_high_water = backend.memory_stats().allocation_high_water_bytes;

        let cold_start = Instant::now();
        execute(&plan, &mut device, case.batch, case.stride).map_err(|error| error.to_string())?;
        backend
            .try_synchronize()
            .map_err(|error| error.to_string())?;
        let cold_ns = cold_start.elapsed().as_nanos();

        let mut samples = Vec::with_capacity(SAMPLES);
        let mut cache_reuse = true;
        for _ in 0..SAMPLES {
            backend
                .try_copy_h2d(&input, &mut device)
                .map_err(|error| error.to_string())?;
            backend
                .try_synchronize()
                .map_err(|error| error.to_string())?;
            let start = Instant::now();
            execute(&plan, &mut device, case.batch, case.stride)
                .map_err(|error| error.to_string())?;
            backend
                .try_synchronize()
                .map_err(|error| error.to_string())?;
            cache_reuse &= backend.memory_stats().allocation_high_water_bytes == planned_high_water;
            samples.push(start.elapsed().as_nanos());
        }
        let mut actual = vec![Complex::<T>::default(); buffer_len];
        backend
            .try_copy_d2h(&device, &mut actual)
            .map_err(|error| error.to_string())?;
        let stats = backend.runtime_stats();
        let error = max_error(&actual, &expected);
        cache_reuse = cache_reuse
            && stats.library_operations == SAMPLES + 1
            && stats.allocation_high_water_bytes == planned_high_water
            && stats.synchronization_calls > 2 * SAMPLES;
        // Reference runs last: its own allocations and library calls must
        // not perturb the stats snapshot or cache-reuse gates above.
        let (cufft, cufft_error) =
            match cufft_reference(backend, case, &mut device, &input, &expected) {
                Ok(reference) => (Some(reference), None),
                Err(error) => (None, Some(error)),
            };
        Ok((
            cold_ns,
            samples,
            stats,
            error,
            cache_reuse,
            cufft,
            cufft_error,
        ))
    })();

    match result {
        Ok((cold_ns, samples, stats, error, cache_reuse, cufft, cufft_error)) => Row {
            case,
            status: if error <= tolerance
                && !samples.is_empty()
                && samples.iter().all(|sample| *sample > 0)
                && cache_reuse
            {
                "passed"
            } else {
                "failed"
            },
            execution: "vendor-native",
            correctness_passed: error <= tolerance,
            throughput_gate_passed: !samples.is_empty()
                && samples.iter().all(|sample| *sample > 0)
                && stats.library_operations > 0,
            cache_reuse_gate_passed: cache_reuse,
            max_abs_error: Some(error),
            error: (error > tolerance).then(|| format!("max error {error} exceeds {tolerance}")),
            plan_ns: Some(plan_ns),
            cold_ns: Some(cold_ns),
            samples_ns: samples,
            stats: Some(stats),
            cufft,
            cufft_error,
        },
        Err(error) => Row {
            case,
            status: "failed",
            execution: "vendor-native",
            correctness_passed: false,
            throughput_gate_passed: false,
            cache_reuse_gate_passed: false,
            max_abs_error: None,
            error: Some(error),
            plan_ns: Some(plan_ns),
            cold_ns: None,
            samples_ns: Vec::new(),
            stats: Some(backend.runtime_stats()),
            cufft: None,
            cufft_error: None,
        },
    }
}

fn benchmark_case<T: FftScalar>(backend: &Arc<CudaBackend>, case: Case, tolerance: f64) -> Row
where
    Complex<T>: sonium_compute::BufferElement,
{
    let buffer_len = case
        .batch
        .saturating_sub(1)
        .saturating_mul(case.stride)
        .saturating_add(case.n);
    benchmark_input_case(
        backend,
        case,
        tolerance,
        make_input::<T>(buffer_len, case.n as u64 ^ case.batch as u64),
    )
}

fn benchmark_device_real_case<T, U>(backend: &Arc<CudaBackend>, spec: DeviceRealCase<T, U>) -> Row
where
    T: BufferElement,
    U: BufferElement,
{
    let DeviceRealCase {
        case,
        tolerance,
        plan_ns,
        execution,
        input,
        expected,
        execute,
        error_fn,
        expected_kernel_launches,
    } = spec;
    if execution != FftExecution::VendorNative {
        return Row {
            case,
            status: "unsupported",
            execution: "host-staged-fallback",
            correctness_passed: false,
            throughput_gate_passed: false,
            cache_reuse_gate_passed: false,
            max_abs_error: None,
            error: Some("backend did not select vendor-native execution".to_owned()),
            plan_ns: Some(plan_ns),
            cold_ns: None,
            samples_ns: Vec::new(),
            stats: None,
            cufft: None,
            cufft_error: None,
        };
    }

    let mut input_device = match backend.raw_alloc::<T>(input.len()) {
        Ok(buffer) => buffer,
        Err(error) => return allocation_error(case, plan_ns, error),
    };
    let mut output_device = match backend.raw_alloc::<U>(expected.len()) {
        Ok(buffer) => buffer,
        Err(error) => return allocation_error(case, plan_ns, error),
    };

    let result = (|| -> Result<(u128, Vec<u128>, CudaRuntimeStatsSnapshot, f64, bool), String> {
        backend
            .try_copy_h2d(&input, &mut input_device)
            .map_err(|error| error.to_string())?;
        backend
            .try_synchronize()
            .map_err(|error| error.to_string())?;
        backend.reset_runtime_stats();

        let cold_start = Instant::now();
        execute(&input_device, &mut output_device).map_err(|error| error.to_string())?;
        backend
            .try_synchronize()
            .map_err(|error| error.to_string())?;
        let cold_ns = cold_start.elapsed().as_nanos();
        let planned_high_water = backend.memory_stats().allocation_high_water_bytes;

        let mut samples = Vec::with_capacity(SAMPLES);
        let mut cache_reuse = true;
        for _ in 0..SAMPLES {
            backend
                .try_copy_h2d(&input, &mut input_device)
                .map_err(|error| error.to_string())?;
            backend
                .try_synchronize()
                .map_err(|error| error.to_string())?;
            let start = Instant::now();
            execute(&input_device, &mut output_device).map_err(|error| error.to_string())?;
            backend
                .try_synchronize()
                .map_err(|error| error.to_string())?;
            cache_reuse &= backend.memory_stats().allocation_high_water_bytes == planned_high_water;
            samples.push(start.elapsed().as_nanos());
        }
        let mut actual = vec![U::default(); expected.len()];
        backend
            .try_copy_d2h(&output_device, &mut actual)
            .map_err(|error| error.to_string())?;
        let stats = backend.runtime_stats();
        let error = error_fn(&actual, &expected);
        cache_reuse = cache_reuse
            && stats.library_operations == SAMPLES + 1
            && stats.kernel_launches == expected_kernel_launches * (SAMPLES + 1)
            && stats.allocation_high_water_bytes == planned_high_water
            && stats.synchronization_calls > 2 * SAMPLES;
        Ok((cold_ns, samples, stats, error, cache_reuse))
    })();

    match result {
        Ok((cold_ns, samples, stats, error, cache_reuse)) => {
            let passed = error <= tolerance
                && !samples.is_empty()
                && samples.iter().all(|sample| *sample > 0)
                && cache_reuse;
            Row {
                case,
                status: if passed { "passed" } else { "failed" },
                execution: "vendor-native",
                correctness_passed: error <= tolerance,
                throughput_gate_passed: !samples.is_empty()
                    && samples.iter().all(|sample| *sample > 0)
                    && stats.library_operations > 0,
                cache_reuse_gate_passed: cache_reuse,
                max_abs_error: Some(error),
                error: (!passed).then(|| format!("real transform gate failed (max error {error})")),
                plan_ns: Some(plan_ns),
                cold_ns: Some(cold_ns),
                samples_ns: samples,
                stats: Some(stats),
                cufft: None,
                cufft_error: None,
            }
        }
        Err(error) => Row {
            case,
            status: "failed",
            execution: "vendor-native",
            correctness_passed: false,
            throughput_gate_passed: false,
            cache_reuse_gate_passed: false,
            max_abs_error: None,
            error: Some(error),
            plan_ns: Some(plan_ns),
            cold_ns: None,
            samples_ns: Vec::new(),
            stats: Some(backend.runtime_stats()),
            cufft: None,
            cufft_error: None,
        },
    }
}

fn benchmark_real_fft_case<T: FftScalar + BufferElement>(
    backend: &Arc<CudaBackend>,
    case: Case,
    tolerance: f64,
) -> Row
where
    Complex<T>: BufferElement,
{
    let planner = BackendRealFftPlanner::<CudaBackend, T>::new(backend.clone());
    let plan_start = Instant::now();
    let plan = planner.plan_fft_forward(case.n);
    let plan_ns = plan_start.elapsed().as_nanos();
    let input = make_real_input::<T>(case.n * case.batch, case.n as u64 ^ case.batch as u64);
    let cpu_plan = RealFftPlan::<T>::new(case.n);
    let expected = input
        .chunks_exact(case.n)
        .flat_map(|chunk| cpu_plan.forward(chunk))
        .collect::<Vec<_>>();
    benchmark_device_real_case(
        backend,
        DeviceRealCase {
            case,
            tolerance,
            plan_ns,
            execution: plan.execution(),
            input,
            expected,
            execute: {
                let plan = plan.clone();
                Box::new(move |input, output| plan.process(input, output))
            },
            error_fn: max_error::<T>,
            expected_kernel_launches: 0,
        },
    )
}

fn benchmark_dct_operation<T: FftScalar + BufferElement>(
    backend: &Arc<CudaBackend>,
    case: Case,
    tolerance: f64,
    operation: usize,
) -> Row
where
{
    let planner = BackendDctPlanner::<CudaBackend, T>::new(backend.clone());
    let plan_start = Instant::now();
    let plan = planner.plan(case.n);
    let plan_ns = plan_start.elapsed().as_nanos();
    let names = if case.dtype == "Real32" {
        [
            "dct2_f32",
            "dct3_f32",
            "dct2_adjoint_f32",
            "dct3_adjoint_f32",
        ]
    } else {
        [
            "dct2_f64",
            "dct3_f64",
            "dct2_adjoint_f64",
            "dct3_adjoint_f64",
        ]
    };
    let (name, reference): (&'static str, RealOperation<T>) = match operation {
        0 => (names[0], dct2::<T>),
        1 => (names[1], dct3::<T>),
        2 => (names[2], dct2_adjoint::<T>),
        _ => (names[3], dct3_adjoint::<T>),
    };
    let input = make_real_input::<T>(case.n * case.batch, operation as u64 ^ case.n as u64);
    let expected = real_reference(&input, case.n, case.batch, reference);
    let case = Case { name, ..case };
    benchmark_device_real_case(
        backend,
        DeviceRealCase {
            case,
            tolerance,
            plan_ns,
            execution: plan.execution(),
            input,
            expected,
            execute: {
                let plan = plan.clone();
                Box::new(move |input, output| match operation {
                    0 => plan.forward(input, output),
                    1 => plan.inverse(input, output),
                    2 => plan.forward_adjoint(input, output),
                    _ => plan.inverse_adjoint(input, output),
                })
            },
            error_fn: max_real_error::<T>,
            expected_kernel_launches: 2,
        },
    )
}

fn benchmark_dst_operation<T: FftScalar + BufferElement>(
    backend: &Arc<CudaBackend>,
    case: Case,
    tolerance: f64,
    operation: usize,
) -> Row
where
{
    let planner = BackendDstPlanner::<CudaBackend, T>::new(backend.clone());
    let plan_start = Instant::now();
    let plan = planner.plan(case.n);
    let plan_ns = plan_start.elapsed().as_nanos();
    let names = if case.dtype == "Real32" {
        [
            "dst2_f32",
            "dst3_f32",
            "dst2_adjoint_f32",
            "dst3_adjoint_f32",
        ]
    } else {
        [
            "dst2_f64",
            "dst3_f64",
            "dst2_adjoint_f64",
            "dst3_adjoint_f64",
        ]
    };
    let (name, reference): (&'static str, RealOperation<T>) = match operation {
        0 => (names[0], dst2::<T>),
        1 => (names[1], dst3::<T>),
        2 => (names[2], dst2_adjoint::<T>),
        _ => (names[3], dst3_adjoint::<T>),
    };
    let input = make_real_input::<T>(case.n * case.batch, operation as u64 ^ case.n as u64);
    let expected = real_reference(&input, case.n, case.batch, reference);
    let case = Case { name, ..case };
    benchmark_device_real_case(
        backend,
        DeviceRealCase {
            case,
            tolerance,
            plan_ns,
            execution: plan.execution(),
            input,
            expected,
            execute: {
                let plan = plan.clone();
                Box::new(move |input, output| match operation {
                    0 => plan.forward(input, output),
                    1 => plan.inverse(input, output),
                    2 => plan.forward_adjoint(input, output),
                    _ => plan.inverse_adjoint(input, output),
                })
            },
            error_fn: max_real_error::<T>,
            expected_kernel_launches: 2,
        },
    )
}

/// Benchmark the complete CUDA convolution pipeline: two cuFFT forwards, a
/// device-resident batched 1x1 contraction for pointwise complex products, and
/// one cuFFT inverse. The contraction maps one complex product to one local
/// matrix multiplication, so no spectrum is staged through the host.
fn benchmark_cuda_convolution(backend: &Arc<CudaBackend>) -> Row {
    let lhs_len = 256;
    let rhs_len = 129;
    let fft_len = 512;
    let case = Case {
        name: "consumer_linear_convolution_c64",
        dtype: "Complex64",
        n: fft_len,
        batch: 1,
        stride: fft_len,
    };
    let planner = BackendFftPlanner::<CudaBackend, f64>::new(backend.clone());
    let plan_start = Instant::now();
    let forward = planner.plan(fft_len, FftDirection::Forward);
    let inverse = planner.plan(fft_len, FftDirection::Inverse);
    let plan_ns = plan_start.elapsed().as_nanos();
    if forward.execution() != FftExecution::VendorNative
        || inverse.execution() != FftExecution::VendorNative
    {
        return Row {
            case,
            status: "unsupported",
            execution: "host-staged-fallback",
            correctness_passed: false,
            throughput_gate_passed: false,
            cache_reuse_gate_passed: false,
            max_abs_error: None,
            error: Some("convolution FFT plans are not vendor-native".to_owned()),
            plan_ns: Some(plan_ns),
            cold_ns: None,
            samples_ns: Vec::new(),
            stats: None,
            cufft: None,
            cufft_error: None,
        };
    }
    if backend.capabilities().batched_local_contract.c64 != sonium_compute::CapabilityState::Native
    {
        return Row {
            case,
            status: "unsupported",
            execution: "host-staged-fallback",
            correctness_passed: false,
            throughput_gate_passed: false,
            cache_reuse_gate_passed: false,
            max_abs_error: None,
            error: Some("CUDA pointwise contraction kernel is unavailable".to_owned()),
            plan_ns: Some(plan_ns),
            cold_ns: None,
            samples_ns: Vec::new(),
            stats: None,
            cufft: None,
            cufft_error: None,
        };
    }

    let lhs = make_input::<f64>(lhs_len, 0xC011);
    let rhs = make_input::<f64>(rhs_len, 0xC022);
    let mut lhs_padded = vec![Complex64::default(); fft_len];
    let mut rhs_padded = vec![Complex64::default(); fft_len];
    lhs_padded[..lhs_len].copy_from_slice(&lhs);
    rhs_padded[..rhs_len].copy_from_slice(&rhs);
    let cpu_plan = LinearConvolutionPlan::<f64>::new(lhs_len, rhs_len);
    let expected_short = cpu_plan.process(&lhs, &rhs);
    let mut expected = vec![Complex64::default(); fft_len];
    expected[..expected_short.len()].copy_from_slice(&expected_short);

    let mut lhs_device = match backend.raw_alloc::<Complex64>(fft_len) {
        Ok(buffer) => buffer,
        Err(error) => return allocation_error(case, plan_ns, error),
    };
    let mut rhs_device = match backend.raw_alloc::<Complex64>(fft_len) {
        Ok(buffer) => buffer,
        Err(error) => return allocation_error(case, plan_ns, error),
    };
    let mut product_device = match backend.raw_alloc::<Complex64>(fft_len) {
        Ok(buffer) => buffer,
        Err(error) => return allocation_error(case, plan_ns, error),
    };

    let run = |lhs_device: &mut <CudaBackend as Backend>::Buffer<Complex64>,
               rhs_device: &mut <CudaBackend as Backend>::Buffer<Complex64>,
               product_device: &mut <CudaBackend as Backend>::Buffer<Complex64>| {
        forward.process(lhs_device)?;
        forward.process(rhs_device)?;
        backend.try_batched_local_contract(
            lhs_device,
            rhs_device,
            product_device,
            fft_len,
            1,
            1,
            1,
        )?;
        inverse.process(product_device)?;
        backend.try_scale(Complex64::new(1.0 / fft_len as f64, 0.0), product_device)
    };
    let result = (|| -> Result<(u128, Vec<u128>, CudaRuntimeStatsSnapshot, f64, bool), String> {
        backend
            .try_copy_h2d(&lhs_padded, &mut lhs_device)
            .map_err(|error| error.to_string())?;
        backend
            .try_copy_h2d(&rhs_padded, &mut rhs_device)
            .map_err(|error| error.to_string())?;
        backend
            .try_synchronize()
            .map_err(|error| error.to_string())?;
        backend.reset_runtime_stats();
        let planned_high_water = backend.memory_stats().allocation_high_water_bytes;

        let cold_start = Instant::now();
        run(&mut lhs_device, &mut rhs_device, &mut product_device)
            .map_err(|error| error.to_string())?;
        backend
            .try_synchronize()
            .map_err(|error| error.to_string())?;
        let cold_ns = cold_start.elapsed().as_nanos();

        let mut samples = Vec::with_capacity(SAMPLES);
        let mut cache_reuse = true;
        for _ in 0..SAMPLES {
            backend
                .try_copy_h2d(&lhs_padded, &mut lhs_device)
                .map_err(|error| error.to_string())?;
            backend
                .try_copy_h2d(&rhs_padded, &mut rhs_device)
                .map_err(|error| error.to_string())?;
            backend
                .try_synchronize()
                .map_err(|error| error.to_string())?;
            let start = Instant::now();
            run(&mut lhs_device, &mut rhs_device, &mut product_device)
                .map_err(|error| error.to_string())?;
            backend
                .try_synchronize()
                .map_err(|error| error.to_string())?;
            cache_reuse &= backend.memory_stats().allocation_high_water_bytes == planned_high_water;
            samples.push(start.elapsed().as_nanos());
        }
        let mut actual = vec![Complex64::default(); fft_len];
        backend
            .try_copy_d2h(&product_device, &mut actual)
            .map_err(|error| error.to_string())?;
        let stats = backend.runtime_stats();
        let error = max_error(&actual, &expected);
        cache_reuse = cache_reuse
            && stats.library_operations == 4 * (SAMPLES + 1)
            && stats.kernel_launches > SAMPLES
            && stats.allocation_high_water_bytes == planned_high_water
            && stats.synchronization_calls > 2 * SAMPLES;
        Ok((cold_ns, samples, stats, error, cache_reuse))
    })();
    match result {
        Ok((cold_ns, samples, stats, error, cache_reuse)) => {
            let passed = error <= 5.0e-9 && !samples.is_empty() && cache_reuse;
            Row {
                case,
                status: if passed { "passed" } else { "failed" },
                execution: "vendor-native",
                correctness_passed: error <= 5.0e-9,
                throughput_gate_passed: !samples.is_empty()
                    && samples.iter().all(|sample| *sample > 0)
                    && stats.kernel_launches > 0,
                cache_reuse_gate_passed: cache_reuse,
                max_abs_error: Some(error),
                error: (!passed).then(|| format!("convolution gate failed (max error {error})")),
                plan_ns: Some(plan_ns),
                cold_ns: Some(cold_ns),
                samples_ns: samples,
                stats: Some(stats),
                cufft: None,
                cufft_error: None,
            }
        }
        Err(error) => Row {
            case,
            status: "failed",
            execution: "vendor-native",
            correctness_passed: false,
            throughput_gate_passed: false,
            cache_reuse_gate_passed: false,
            max_abs_error: None,
            error: Some(error),
            plan_ns: Some(plan_ns),
            cold_ns: None,
            samples_ns: Vec::new(),
            stats: Some(backend.runtime_stats()),
            cufft: None,
            cufft_error: None,
        },
    }
}

/// Build the complex frames used by the consumer gate. The framing and Hann
/// window are the same work a real STFT analysis performs before its C2C
/// transform; only the device-resident FFT portion is benchmarked here.
fn stft_windowed_frames() -> Vec<Complex64> {
    const FFT_SIZE: usize = 64;
    const HOP_SIZE: usize = 32;
    const FRAMES: usize = 8;
    let signal_len = (FRAMES - 1) * HOP_SIZE + FFT_SIZE;
    let signal = make_input::<f64>(signal_len, 0x57_46_54);
    let mut frames = vec![Complex64::default(); FFT_SIZE * FRAMES];
    for frame_index in 0..FRAMES {
        for sample_index in 0..FFT_SIZE {
            let phase = std::f64::consts::TAU * sample_index as f64 / FFT_SIZE as f64;
            let window = 0.5 - 0.5 * phase.cos();
            frames[frame_index * FFT_SIZE + sample_index] =
                signal[frame_index * HOP_SIZE + sample_index] * window;
        }
    }
    frames
}

fn stft_case() -> Case {
    Case {
        name: "consumer_stft_windowed_batch_c64",
        dtype: "Complex64",
        n: 64,
        batch: 8,
        stride: 64,
    }
}

#[derive(Default)]
struct CancellationSummary {
    regular_c32: bool,
    regular_c64: bool,
    strided_c64: bool,
    pre_cancel_rejected: bool,
    post_submit_cancelled: bool,
    completion_fence_drained: bool,
    plan_reused: bool,
    error: Option<String>,
    completion_events: usize,
    completion_waits: usize,
}

fn cancellation_case<T: FftScalar + BufferElement>(
    backend: &Arc<CudaBackend>,
    plan: &sonium_fft::DeviceFftPlan<CudaBackend, T>,
    input: &[Complex<T>],
    batch: usize,
    stride: usize,
) -> Result<(bool, bool, bool), String>
where
    Complex<T>: BufferElement,
{
    let mut buffer = backend
        .raw_alloc::<Complex<T>>(input.len())
        .map_err(|error| format!("cancellation allocation failed: {error}"))?;
    backend
        .try_copy_h2d(input, &mut buffer)
        .map_err(|error| error.to_string())?;
    backend
        .try_synchronize()
        .map_err(|error| error.to_string())?;

    let pre_cancel = FftCancellationToken::new();
    pre_cancel.cancel();
    let pre_error = if stride == plan.len() {
        plan.process_cancellable(&mut buffer, &pre_cancel)
    } else {
        plan.process_strided_cancellable(&mut buffer, batch, stride, &pre_cancel)
    }
    .expect_err("pre-cancelled CUDA FFT must be rejected");
    if !pre_error.is_cancelled() {
        return Err(format!("pre-cancel returned an untyped error: {pre_error}"));
    }
    let mut unchanged = vec![Complex::<T>::default(); input.len()];
    backend
        .try_copy_d2h(&buffer, &mut unchanged)
        .map_err(|error| error.to_string())?;
    let pre_preserved = max_error(&unchanged, input) == 0.0;
    if !pre_preserved {
        return Err("pre-cancel mutated the destination".to_owned());
    }

    backend
        .try_copy_h2d(input, &mut buffer)
        .map_err(|error| error.to_string())?;
    backend
        .try_synchronize()
        .map_err(|error| error.to_string())?;
    let post_cancel = FftCancellationToken::new();
    let request = post_cancel.clone();
    let post_error = if stride == plan.len() {
        plan.process_cancellable_with_hook(&mut buffer, &post_cancel, move || request.cancel())
    } else {
        plan.process_strided_cancellable_with_hook(
            &mut buffer,
            batch,
            stride,
            &post_cancel,
            move || request.cancel(),
        )
    }
    .expect_err("post-submit CUDA FFT cancellation must be reported");
    if !post_error.is_cancelled() {
        return Err(format!(
            "post-submit cancel returned an untyped error: {post_error}"
        ));
    }

    // A cancelled in-place destination is discarded. Re-uploading the input
    // and using the same plan proves the completion fence drained the native
    // operation and that its cached workspace remains reusable.
    backend
        .try_copy_h2d(input, &mut buffer)
        .map_err(|error| error.to_string())?;
    backend
        .try_synchronize()
        .map_err(|error| error.to_string())?;
    let clear = FftCancellationToken::new();
    if stride == plan.len() {
        plan.process_cancellable(&mut buffer, &clear)
    } else {
        plan.process_strided_cancellable(&mut buffer, batch, stride, &clear)
    }
    .map_err(|error| error.to_string())?;
    let mut actual = vec![Complex::<T>::default(); input.len()];
    backend
        .try_copy_d2h(&buffer, &mut actual)
        .map_err(|error| error.to_string())?;
    let expected = cpu_reference(input, plan.len(), batch, stride);
    let reused_correctly = max_error(&actual, &expected)
        <= if TypeId::of::<T>() == TypeId::of::<f32>() {
            5.0e-3
        } else {
            5.0e-11
        };
    Ok((true, true, reused_correctly))
}

fn benchmark_cancellation(backend: &Arc<CudaBackend>) -> CancellationSummary {
    let mut summary = CancellationSummary::default();
    summary.pre_cancel_rejected = true;
    summary.post_submit_cancelled = true;
    summary.plan_reused = true;
    backend.reset_runtime_stats();
    let c32_plan = BackendFftPlanner::<CudaBackend, f32>::new(backend.clone())
        .plan(1024, FftDirection::Forward);
    let c64_plan = BackendFftPlanner::<CudaBackend, f64>::new(backend.clone())
        .plan(1024, FftDirection::Forward);
    let strided_plan = BackendFftPlanner::<CudaBackend, f64>::new(backend.clone())
        .plan(256, FftDirection::Forward);
    if c32_plan.execution() != FftExecution::VendorNative
        || c64_plan.execution() != FftExecution::VendorNative
        || strided_plan.execution() != FftExecution::VendorNative
    {
        summary.error = Some("cancellation plans were not vendor-native".to_owned());
        return summary;
    }

    let c32_input = make_input::<f32>(1024 * 2, 0xCA32);
    let c64_input = make_input::<f64>(1024 * 2, 0xCA64);
    let strided_input = make_input::<f64>(256 + 320, 0xCA5D);
    let c32 = cancellation_case(backend, &c32_plan, &c32_input, 2, 1024);
    let c64 = cancellation_case(backend, &c64_plan, &c64_input, 2, 1024);
    let strided = cancellation_case(backend, &strided_plan, &strided_input, 2, 320);
    match c32 {
        Ok((pre, post, reused)) => {
            summary.regular_c32 = reused;
            summary.pre_cancel_rejected &= pre;
            summary.post_submit_cancelled &= post;
            summary.plan_reused &= reused;
        }
        Err(error) => summary.error = Some(format!("Complex32: {error}")),
    }
    match c64 {
        Ok((pre, post, reused)) => {
            summary.regular_c64 = reused;
            summary.pre_cancel_rejected &= pre;
            summary.post_submit_cancelled &= post;
            summary.plan_reused &= reused;
        }
        Err(error) => summary.error = Some(format!("Complex64: {error}")),
    }
    match strided {
        Ok((pre, post, reused)) => {
            summary.strided_c64 = reused;
            summary.pre_cancel_rejected &= pre;
            summary.post_submit_cancelled &= post;
            summary.plan_reused &= reused;
        }
        Err(error) => summary.error = Some(format!("strided Complex64: {error}")),
    }
    let stats = backend.runtime_stats();
    summary.completion_events = stats.completion_events;
    summary.completion_waits = stats.completion_waits;
    summary.completion_fence_drained = stats.completion_events >= 6 && stats.completion_waits >= 6;
    if summary.error.is_none()
        && summary.regular_c32
        && summary.regular_c64
        && summary.strided_c64
        && summary.pre_cancel_rejected
        && summary.post_submit_cancelled
        && summary.completion_fence_drained
        && summary.plan_reused
    {
        return summary;
    }
    if summary.error.is_none() {
        summary.error = Some("cancellation cleanup/reuse gate failed".to_owned());
    }
    summary
}

fn cancellation_json(summary: &CancellationSummary) -> String {
    let status = summary.error.is_none()
        && summary.regular_c32
        && summary.regular_c64
        && summary.strided_c64
        && summary.pre_cancel_rejected
        && summary.post_submit_cancelled
        && summary.completion_fence_drained
        && summary.plan_reused;
    format!(
        "{{\"status\":{},\"regular_c32\":{},\"regular_c64\":{},\"strided_c64\":{},\"pre_cancel_rejected\":{},\"post_submit_cancelled\":{},\"completion_fence_drained\":{},\"plan_reused\":{},\"completion_events\":{},\"completion_waits\":{},\"destination_disposition\":\"discarded_after_submitted_cancel\",\"error\":{}}}",
        json_string(if status { "passed" } else { "failed" }),
        summary.regular_c32,
        summary.regular_c64,
        summary.strided_c64,
        summary.pre_cancel_rejected,
        summary.post_submit_cancelled,
        summary.completion_fence_drained,
        summary.plan_reused,
        summary.completion_events,
        summary.completion_waits,
        json_optional_string(summary.error.as_deref()),
    )
}

fn record_row(rows: &mut Vec<Row>, status: &mut &'static str, row: Row) {
    if row.status == "failed" || (required_cuda() && row.status != "passed") {
        *status = "failed";
    }
    rows.push(row);
}

fn allocation_error(case: Case, plan_ns: u128, error: OomError) -> Row {
    Row {
        case,
        status: "failed",
        execution: "vendor-native",
        correctness_passed: false,
        throughput_gate_passed: false,
        cache_reuse_gate_passed: false,
        max_abs_error: None,
        error: Some(format!("device allocation failed: {error}")),
        plan_ns: Some(plan_ns),
        cold_ns: None,
        samples_ns: Vec::new(),
        stats: None,
        cufft: None,
        cufft_error: None,
    }
}

fn unsupported_cells() -> &'static [&'static str] {
    &[
        "high-level-real-stft",
        "multidimensional-2d",
        "multidimensional-3d",
        "unsupported-dtype",
        "unsupported-layout",
    ]
}

/// Emit the row-level execution accounting separately from the detailed rows.
/// A required run can therefore be checked without treating an unavailable
/// row or an explicitly unsupported cell as a successful CPU fallback.
fn matrix_json(rows: &[Row]) -> String {
    let executed = rows
        .iter()
        .filter(|row| matches!(row.status, "passed" | "failed"))
        .count();
    let passed = rows.iter().filter(|row| row.status == "passed").count();
    let failed = rows.iter().filter(|row| row.status == "failed").count();
    let skipped = rows
        .iter()
        .filter(|row| row.status == "unavailable")
        .count();
    let unsupported = rows
        .iter()
        .filter(|row| row.status == "unsupported")
        .count();
    format!(
        "{{\"required_cuda\":{},\"fail_closed\":true,\"rows_total\":{},\"executed\":{},\"passed\":{},\"failed\":{},\"skipped\":{},\"unsupported_rows\":{},\"unsupported_cells\":{}}}",
        required_cuda(),
        rows.len(),
        executed,
        passed,
        failed,
        skipped,
        unsupported,
        unsupported_cells().len(),
    )
}

fn stats_json(stats: Option<CudaRuntimeStatsSnapshot>) -> String {
    let Some(stats) = stats else {
        return "null".to_owned();
    };
    format!(
        "{{\"kernel_launches\":{},\"library_operations\":{},\"synchronization_calls\":{},\"failed_synchronizations\":{},\"completion_events\":{},\"completion_waits\":{},\"h2d_operations\":{},\"h2d_bytes\":{},\"d2h_operations\":{},\"d2h_bytes\":{},\"d2d_operations\":{},\"d2d_bytes\":{},\"live_allocation_bytes\":{},\"allocation_high_water_bytes\":{},\"live_staging_bytes\":{},\"staging_high_water_bytes\":{}}}",
        stats.kernel_launches,
        stats.library_operations,
        stats.synchronization_calls,
        stats.failed_synchronizations,
        stats.completion_events,
        stats.completion_waits,
        stats.transfer.h2d_operations,
        stats.transfer.h2d_bytes,
        stats.transfer.d2h_operations,
        stats.transfer.d2h_bytes,
        stats.transfer.d2d_operations,
        stats.transfer.d2d_bytes,
        stats.live_allocation_bytes,
        stats.allocation_high_water_bytes,
        stats.live_staging_bytes,
        stats.staging_high_water_bytes,
    )
}

/// `cufft_reference` value for one row: measured object, unavailable
/// object (load/exec error), or `null` where no cuFFT equivalent exists.
fn cufft_reference_json(row: &Row) -> String {
    match (&row.cufft, &row.cufft_error) {
        (Some(reference), _) => {
            let cufft_median = median(&reference.samples_ns) as f64;
            let sonium_median = median(&row.samples_ns) as f64;
            // Ratio > 1 means sonium's wrapper is slower than raw cuFFT.
            let ratio = if cufft_median > 0.0 && sonium_median > 0.0 {
                format!(
                    "{}",
                    (sonium_median / cufft_median * 1000.0).round() / 1000.0
                )
            } else {
                "null".to_owned()
            };
            format!(
                "{{\"status\":\"measured\",\"allocation\":\"auto\",\"plan_ns\":{},\"cold_ns\":{},\"median_ns\":{},\"p95_ns\":{},\"max_abs_error\":{},\"overhead_ratio\":{}}}",
                reference.plan_ns,
                reference.cold_ns,
                median(&reference.samples_ns),
                percentile(&reference.samples_ns, 95, 100),
                reference.max_abs_error,
                ratio,
            )
        }
        (None, Some(error)) => format!(
            "{{\"status\":\"unavailable\",\"error\":{}}}",
            json_string(error)
        ),
        (None, None) => "null".to_owned(),
    }
}

fn row_json(row: &Row) -> String {
    let stride = (row.case.stride != row.case.n).then_some(row.case.stride);
    let samples = row
        .samples_ns
        .iter()
        .map(u128::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let (median_ns, p95_ns, p99_ns, ci_low_ns, ci_high_ns) = if row.samples_ns.is_empty() {
        (
            "null".to_owned(),
            "null".to_owned(),
            "null".to_owned(),
            "null".to_owned(),
            "null".to_owned(),
        )
    } else {
        (
            median(&row.samples_ns).to_string(),
            percentile(&row.samples_ns, 95, 100).to_string(),
            percentile(&row.samples_ns, 99, 100).to_string(),
            percentile(&row.samples_ns, 5, 200).to_string(),
            percentile(&row.samples_ns, 195, 200).to_string(),
        )
    };
    let (bytes_per_second, transforms_per_second) = if row.samples_ns.is_empty() {
        ("null".to_owned(), "null".to_owned())
    } else {
        let median = median(&row.samples_ns) as f64;
        let bytes_per_sample = match row.case.dtype {
            "Complex32" => 8.0,
            "Real32" => 4.0,
            "Complex64" => 16.0,
            "Real64" => 8.0,
            _ => 1.0,
        };
        let bytes = row.case.n as f64 * row.case.batch as f64 * bytes_per_sample;
        (
            (bytes * 1.0e9 / median).to_string(),
            (row.case.batch as f64 * 1.0e9 / median).to_string(),
        )
    };
    format!(
        "{{\"case\":{},\"dtype\":{},\"n\":{},\"batch\":{},\"stride\":{},\"status\":{},\"execution\":{},\"correctness_passed\":{},\"throughput_gate_passed\":{},\"cache_reuse_gate_passed\":{},\"max_abs_error\":{},\"error\":{},\"plan_ns\":{},\"cold_ns\":{},\"samples_ns\":[{}],\"median_ns\":{},\"p95_ns\":{},\"p99_ns\":{},\"ci95_low_ns\":{},\"ci95_high_ns\":{},\"ci95_method\":\"empirical_order_stat_95\",\"bytes_per_second\":{},\"transforms_per_second\":{},\"cufft_reference\":{},\"stats\":{}}}",
        json_string(row.case.name),
        json_string(row.case.dtype),
        row.case.n,
        row.case.batch,
        stride.map_or_else(|| "null".to_owned(), |value| value.to_string()),
        json_string(row.status),
        json_string(row.execution),
        row.correctness_passed,
        row.throughput_gate_passed,
        row.cache_reuse_gate_passed,
        row.max_abs_error
            .map_or_else(|| "null".to_owned(), |value| value.to_string()),
        json_optional_string(row.error.as_deref()),
        row.plan_ns
            .map_or_else(|| "null".to_owned(), |value| value.to_string()),
        row.cold_ns
            .map_or_else(|| "null".to_owned(), |value| value.to_string()),
        samples,
        median_ns,
        p95_ns,
        p99_ns,
        ci_low_ns,
        ci_high_ns,
        bytes_per_second,
        transforms_per_second,
        cufft_reference_json(row),
        stats_json(row.stats),
    )
}

fn unavailable_row(case: Case, error: &str) -> Row {
    Row {
        case,
        status: "unavailable",
        execution: "not-run",
        correctness_passed: false,
        throughput_gate_passed: false,
        cache_reuse_gate_passed: false,
        max_abs_error: None,
        error: Some(error.to_owned()),
        plan_ns: None,
        cold_ns: None,
        samples_ns: Vec::new(),
        stats: None,
        cufft: None,
        cufft_error: None,
    }
}

fn unavailable_cancellation(error: &str) -> String {
    format!(
        "{{\"status\":\"unavailable\",\"regular_c32\":false,\"regular_c64\":false,\"strided_c64\":false,\"pre_cancel_rejected\":false,\"post_submit_cancelled\":false,\"completion_fence_drained\":false,\"plan_reused\":false,\"completion_events\":0,\"completion_waits\":0,\"destination_disposition\":\"not-run\",\"error\":{}}}",
        json_string(error),
    )
}

fn cases() -> [Case; 8] {
    [
        Case {
            name: "smooth_single_c32",
            dtype: "Complex32",
            n: 64,
            batch: 1,
            stride: 64,
        },
        Case {
            name: "smooth_batched_c32",
            dtype: "Complex32",
            n: 1024,
            batch: 8,
            stride: 1024,
        },
        Case {
            name: "prime_c32",
            dtype: "Complex32",
            n: 257,
            batch: 4,
            stride: 257,
        },
        Case {
            name: "strided_c32",
            dtype: "Complex32",
            n: 128,
            batch: 8,
            stride: 160,
        },
        Case {
            name: "smooth_single_c64",
            dtype: "Complex64",
            n: 64,
            batch: 1,
            stride: 64,
        },
        Case {
            name: "smooth_batched_c64",
            dtype: "Complex64",
            n: 1024,
            batch: 8,
            stride: 1024,
        },
        Case {
            name: "prime_c64",
            dtype: "Complex64",
            n: 257,
            batch: 4,
            stride: 257,
        },
        Case {
            name: "strided_c64",
            dtype: "Complex64",
            n: 128,
            batch: 8,
            stride: 160,
        },
    ]
}

fn main() {
    let (device, driver, sm_clock, memory_clock) = cuda_hardware();
    let mut rows = Vec::new();
    let mut status = "passed";
    // True only if every complex input case (the ones with a cuFFT
    // equivalent) produced a raw-cuFFT reference measurement.
    let mut cufft_available = false;
    let (unavailable_error, cancellation) = match CudaBackend::new_default() {
        Ok(backend) => {
            let backend = Arc::new(backend);
            cufft_available = true;
            for case in cases() {
                let row = if case.dtype == "Complex32" {
                    benchmark_case::<f32>(&backend, case, 5.0e-3)
                } else {
                    benchmark_case::<f64>(&backend, case, 5.0e-11)
                };
                if row.status == "failed" || (required_cuda() && row.status != "passed") {
                    status = "failed";
                }
                cufft_available &= row.cufft.is_some();
                rows.push(row);
            }
            let real_fft32 = Case {
                name: "real_fft_forward_f32",
                dtype: "Real32",
                n: 1024,
                batch: 8,
                stride: 1024,
            };
            let real_fft64 = Case {
                name: "real_fft_forward_f64",
                dtype: "Real64",
                n: 1024,
                batch: 8,
                stride: 1024,
            };
            record_row(
                &mut rows,
                &mut status,
                benchmark_real_fft_case::<f32>(&backend, real_fft32, 5.0e-3),
            );
            record_row(
                &mut rows,
                &mut status,
                benchmark_real_fft_case::<f64>(&backend, real_fft64, 5.0e-10),
            );

            let dct32 = Case {
                name: "dct2_f32",
                dtype: "Real32",
                n: 256,
                batch: 8,
                stride: 256,
            };
            let dct64 = Case {
                name: "dct2_f64",
                dtype: "Real64",
                n: 256,
                batch: 8,
                stride: 256,
            };
            let dst32 = Case {
                name: "dst2_f32",
                dtype: "Real32",
                n: 256,
                batch: 8,
                stride: 256,
            };
            let dst64 = Case {
                name: "dst2_f64",
                dtype: "Real64",
                n: 256,
                batch: 8,
                stride: 256,
            };
            for operation in 0..4 {
                record_row(
                    &mut rows,
                    &mut status,
                    benchmark_dct_operation::<f32>(&backend, dct32, 5.0e-3, operation),
                );
                record_row(
                    &mut rows,
                    &mut status,
                    benchmark_dct_operation::<f64>(&backend, dct64, 5.0e-10, operation),
                );
                record_row(
                    &mut rows,
                    &mut status,
                    benchmark_dst_operation::<f32>(&backend, dst32, 5.0e-3, operation),
                );
                record_row(
                    &mut rows,
                    &mut status,
                    benchmark_dst_operation::<f64>(&backend, dst64, 5.0e-10, operation),
                );
            }
            let consumer =
                benchmark_input_case(&backend, stft_case(), 5.0e-11, stft_windowed_frames());
            if consumer.status == "failed" || (required_cuda() && consumer.status != "passed") {
                status = "failed";
            }
            cufft_available &= consumer.cufft.is_some();
            rows.push(consumer);
            let convolution = benchmark_cuda_convolution(&backend);
            if convolution.status == "failed" || (required_cuda() && convolution.status != "passed")
            {
                status = "failed";
            }
            rows.push(convolution);
            let summary = benchmark_cancellation(&backend);
            if summary.error.is_some() {
                status = "failed";
            }
            (None, cancellation_json(&summary))
        }
        Err(error) => {
            let error = error.to_string();
            status = "unavailable";
            for case in cases() {
                rows.push(unavailable_row(case, &error));
            }
            rows.push(unavailable_row(stft_case(), &error));
            rows.push(unavailable_row(
                Case {
                    name: "consumer_linear_convolution_c64",
                    dtype: "Complex64",
                    n: 512,
                    batch: 1,
                    stride: 512,
                },
                &error,
            ));
            (Some(error.clone()), unavailable_cancellation(&error))
        }
    };

    let mut output = String::new();
    let captured_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs());
    let _ = write!(
        output,
        "{{\"schema_version\":{},\"artifact\":\"sonium-fft-cuda-benchmark\",\"captured_at_unix_s\":{},\"status\":{},\"compiler\":{},\"cuda_toolkit\":{},\"hardware\":{{\"device\":{},\"driver\":{},\"sm_clock_mhz\":{},\"memory_clock_mhz\":{}}},\"target\":{{\"os\":{},\"arch\":{}}},\"energy_joules\":null,\"energy_source\":null,\"matrix\":{},\"cancellation\":{},\"cufft_reference_status\":{},\"unsupported_cells\":[{}],\"rows\":[{}]}}",
        SCHEMA_VERSION,
        captured_at,
        json_string(status),
        json_optional_string(compiler().as_deref()),
        json_optional_string(cuda_toolkit().as_deref()),
        json_optional_string(device.as_deref()),
        json_optional_string(driver.as_deref()),
        json_optional_string(sm_clock.as_deref()),
        json_optional_string(memory_clock.as_deref()),
        json_string(std::env::consts::OS),
        json_string(std::env::consts::ARCH),
        matrix_json(&rows),
        cancellation,
        json_string(if cufft_available {
            "available"
        } else {
            "unavailable"
        }),
        unsupported_cells()
            .iter()
            .map(|cell| json_string(cell))
            .collect::<Vec<_>>()
            .join(","),
        rows.iter().map(row_json).collect::<Vec<_>>().join(","),
    );
    if let Ok(path) = std::env::var("SONIUM_EVIDENCE_OUTPUT") {
        std::fs::write(&path, &output).unwrap_or_else(|error| {
            eprintln!("failed to write CUDA evidence artifact {path}: {error}");
            std::process::exit(2);
        });
    } else {
        println!("{output}");
    }

    if required_cuda() && unavailable_error.is_some() {
        eprintln!("CUDA evidence required but the CUDA backend is unavailable");
        std::process::exit(1);
    }
    if status == "failed" {
        std::process::exit(1);
    }
}
