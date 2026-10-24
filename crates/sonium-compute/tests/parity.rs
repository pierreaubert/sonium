//! CPU ↔ GPU parity tests.
//!
//! For each enabled GPU backend, we run the same BLAS-1 + CSR SpMV ops on
//! the CPU backend and the GPU backend with identical inputs and assert the
//! results agree to within a small tolerance. Tests skip gracefully (with a
//! visible message) when the GPU backend cannot be initialised on the host —
//! this is the expected outcome for the CUDA backend on macOS or any
//! Linux/CI box without an NVIDIA driver, and for the Metal backend on
//! non-Apple targets.
//!
//! Tolerance is loose (1e-5 relative) because GPU kernels typically use
//! parallel reductions whose summation order differs from a sequential CPU
//! loop; correctness is what matters here, not bit-exactness.

#![allow(dead_code)]
// GPU-parity CSR fixtures use the checked `try_*` constructors, matching the
// production contract.

use num_complex::{Complex32, Complex64};
#[cfg(feature = "cuda")]
use sonium_compute::Transpose;
use sonium_compute::{
    Backend, CapabilityState, CpuBackend, GemmBatch, GemvBatch, MemoryPool, MemoryPoolConfig,
};

const REL_TOL: f64 = 1e-5;

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}

#[cfg(feature = "cuda")]
macro_rules! skip_or_require_cuda {
    ($label:expr, $err:expr) => {{
        if crate::env_flag("SONIUM_REQUIRE_CUDA") {
            panic!(
                "CUDA required by SONIUM_REQUIRE_CUDA=1 but unavailable for {}: {}",
                $label, $err
            );
        } else {
            eprintln!("CUDA unavailable, skipping {}: {}", $label, $err);
            return;
        }
    }};
}

#[cfg(feature = "metal")]
macro_rules! skip_or_require_metal {
    ($label:expr, $err:expr) => {{
        if crate::env_flag("SONIUM_REQUIRE_METAL") {
            panic!(
                "Metal required by SONIUM_REQUIRE_METAL=1 but unavailable for {}: {}",
                $label, $err
            );
        } else {
            eprintln!("Metal unavailable, skipping {}: {}", $label, $err);
            return;
        }
    }};
}

/// Reference inputs shared across CPU and GPU runs.
struct Inputs {
    x: Vec<f64>,
    y: Vec<f64>,
    alpha: f64,
    // CSR: a tridiagonal 5x5 matrix with values [-1, 2, -1] per row.
    rows: usize,
    cols: usize,
    row_ptr: Vec<i32>,
    col_idx: Vec<i32>,
    values: Vec<f64>,
    spx: Vec<f64>,
}

fn make_inputs() -> Inputs {
    let x: Vec<f64> = (0..16).map(|i| (i as f64) * 0.5 - 3.0).collect();
    let y: Vec<f64> = (0..16).map(|i| 1.0 + (i as f64) * 0.25).collect();
    // Tridiagonal Laplacian on 5 nodes (rows = cols = 5).
    let row_ptr = vec![0, 2, 5, 8, 11, 13];
    let col_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4];
    let values = vec![
        2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0,
    ];
    let spx = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    Inputs {
        x,
        y,
        alpha: 1.7,
        rows: 5,
        cols: 5,
        row_ptr,
        col_idx,
        values,
        spx,
    }
}

/// Compute reference (CPU) outputs for all five ops.
fn cpu_reference(inp: &Inputs) -> (Vec<f64>, f64, f64, Vec<f64>, Vec<f64>) {
    let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();

    // axpy: y_out = alpha * x + y
    let mut x = pool.try_alloc::<f64>(inp.x.len()).unwrap();
    let mut y = pool.try_alloc::<f64>(inp.y.len()).unwrap();
    pool.backend().copy_h2d(&inp.x, x.raw_mut());
    pool.backend().copy_h2d(&inp.y, y.raw_mut());
    pool.backend().axpy(inp.alpha, x.raw(), y.raw_mut());
    let mut axpy_out = vec![0.0_f64; inp.y.len()];
    pool.backend().copy_d2h(y.raw(), &mut axpy_out);

    // dot, nrm2 of original x
    let mut x2 = pool.try_alloc::<f64>(inp.x.len()).unwrap();
    let mut y2 = pool.try_alloc::<f64>(inp.y.len()).unwrap();
    pool.backend().copy_h2d(&inp.x, x2.raw_mut());
    pool.backend().copy_h2d(&inp.y, y2.raw_mut());
    let dot = pool.backend().dot(x2.raw(), y2.raw());
    let nrm = pool.backend().nrm2(x2.raw());

    // scale: alpha * x
    pool.backend().scale(inp.alpha, x2.raw_mut());
    let mut scale_out = vec![0.0_f64; inp.x.len()];
    pool.backend().copy_d2h(x2.raw(), &mut scale_out);

    // spmv: A * spx
    let mut row = pool.try_alloc::<i32>(inp.row_ptr.len()).unwrap();
    let mut col = pool.try_alloc::<i32>(inp.col_idx.len()).unwrap();
    let mut val = pool.try_alloc::<f64>(inp.values.len()).unwrap();
    let mut spx = pool.try_alloc::<f64>(inp.spx.len()).unwrap();
    let mut spy = pool.try_alloc::<f64>(inp.rows).unwrap();
    pool.backend().copy_h2d(&inp.row_ptr, row.raw_mut());
    pool.backend().copy_h2d(&inp.col_idx, col.raw_mut());
    pool.backend().copy_h2d(&inp.values, val.raw_mut());
    pool.backend().copy_h2d(&inp.spx, spx.raw_mut());
    pool.backend().spmv_csr(
        inp.rows,
        inp.cols,
        row.raw(),
        col.raw(),
        val.raw(),
        spx.raw(),
        spy.raw_mut(),
    );
    let mut spmv_out = vec![0.0_f64; inp.rows];
    pool.backend().copy_d2h(spy.raw(), &mut spmv_out);

    (axpy_out, dot, nrm, scale_out, spmv_out)
}

fn run_on_backend<B: Backend>(
    pool: &MemoryPool<B>,
    inp: &Inputs,
) -> (Vec<f64>, f64, f64, Vec<f64>, Vec<f64>) {
    let mut x = pool.try_alloc::<f64>(inp.x.len()).unwrap();
    let mut y = pool.try_alloc::<f64>(inp.y.len()).unwrap();
    pool.backend().copy_h2d(&inp.x, x.raw_mut());
    pool.backend().copy_h2d(&inp.y, y.raw_mut());
    pool.backend().axpy(inp.alpha, x.raw(), y.raw_mut());
    let mut axpy_out = vec![0.0_f64; inp.y.len()];
    pool.backend().copy_d2h(y.raw(), &mut axpy_out);

    let mut x2 = pool.try_alloc::<f64>(inp.x.len()).unwrap();
    let mut y2 = pool.try_alloc::<f64>(inp.y.len()).unwrap();
    pool.backend().copy_h2d(&inp.x, x2.raw_mut());
    pool.backend().copy_h2d(&inp.y, y2.raw_mut());
    let dot = pool.backend().dot(x2.raw(), y2.raw());
    let nrm = pool.backend().nrm2(x2.raw());

    pool.backend().scale(inp.alpha, x2.raw_mut());
    let mut scale_out = vec![0.0_f64; inp.x.len()];
    pool.backend().copy_d2h(x2.raw(), &mut scale_out);

    let mut row = pool.try_alloc::<i32>(inp.row_ptr.len()).unwrap();
    let mut col = pool.try_alloc::<i32>(inp.col_idx.len()).unwrap();
    let mut val = pool.try_alloc::<f64>(inp.values.len()).unwrap();
    let mut spx = pool.try_alloc::<f64>(inp.spx.len()).unwrap();
    let mut spy = pool.try_alloc::<f64>(inp.rows).unwrap();
    pool.backend().copy_h2d(&inp.row_ptr, row.raw_mut());
    pool.backend().copy_h2d(&inp.col_idx, col.raw_mut());
    pool.backend().copy_h2d(&inp.values, val.raw_mut());
    pool.backend().copy_h2d(&inp.spx, spx.raw_mut());
    pool.backend().spmv_csr(
        inp.rows,
        inp.cols,
        row.raw(),
        col.raw(),
        val.raw(),
        spx.raw(),
        spy.raw_mut(),
    );
    let mut spmv_out = vec![0.0_f64; inp.rows];
    pool.backend().copy_d2h(spy.raw(), &mut spmv_out);

    (axpy_out, dot, nrm, scale_out, spmv_out)
}

fn assert_vec_close(label: &str, actual: &[f64], expected: &[f64]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch ({} vs {})",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let scale = e.abs().max(1.0);
        let diff = (a - e).abs();
        assert!(
            diff <= REL_TOL * scale,
            "{label}[{i}]: gpu={a}, cpu={e}, |diff|={diff}, tol={}",
            REL_TOL * scale
        );
    }
}

fn assert_complex64_vec_close(label: &str, actual: &[Complex64], expected: &[Complex64]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch ({} vs {})",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let scale = e.norm().max(1.0);
        let diff = (a - e).norm();
        assert!(
            diff <= REL_TOL * scale,
            "{label}[{i}]: gpu={a}, cpu={e}, |diff|={diff}, tol={}",
            REL_TOL * scale
        );
    }
}

fn assert_close(label: &str, actual: f64, expected: f64) {
    let scale = expected.abs().max(1.0);
    let diff = (actual - expected).abs();
    assert!(
        diff <= REL_TOL * scale,
        "{label}: gpu={actual}, cpu={expected}, |diff|={diff}, tol={}",
        REL_TOL * scale
    );
}

fn assert_parity(
    backend_name: &str,
    cpu: &(Vec<f64>, f64, f64, Vec<f64>, Vec<f64>),
    gpu: &(Vec<f64>, f64, f64, Vec<f64>, Vec<f64>),
) {
    eprintln!("--- {backend_name} parity ---");
    assert_vec_close(&format!("{backend_name}/axpy"), &gpu.0, &cpu.0);
    assert_close(&format!("{backend_name}/dot"), gpu.1, cpu.1);
    assert_close(&format!("{backend_name}/nrm2"), gpu.2, cpu.2);
    assert_vec_close(&format!("{backend_name}/scale"), &gpu.3, &cpu.3);
    assert_vec_close(&format!("{backend_name}/spmv"), &gpu.4, &cpu.4);
}

#[test]
fn cpu_capabilities_match_native_contract() {
    let caps = CpuBackend::new().capabilities();
    assert_eq!(caps.backend_name, "cpu");
    assert_eq!(caps.blas1.f32, CapabilityState::Native);
    assert_eq!(caps.blas1.f64, CapabilityState::Native);
    assert_eq!(caps.gemv.c32, CapabilityState::Native);
    assert_eq!(caps.gemm.c64, CapabilityState::Native);
    assert_eq!(caps.csr_spmv.f64, CapabilityState::Native);
    assert_eq!(caps.csr_transpose.f64, CapabilityState::Native);
    assert_eq!(caps.precision_cast.c64_to_c32, CapabilityState::Native);
    assert_eq!(caps.bem_regular_pair_c32, CapabilityState::Native);
    assert_eq!(caps.bem_regular_pair_c64, CapabilityState::Native);
    assert_eq!(caps.bem.near_pair.c64, CapabilityState::Native);
    assert_eq!(caps.bem.hmatrix_setup.c64, CapabilityState::Native);
    assert_eq!(caps.bem.hmatrix.c64, CapabilityState::Native);
    // Hierarchical operators are Complex64-only: the remaining scalar
    // cells stay unsupported unconditionally (no staging fallback).
    for (label, table) in [
        ("hmatrix_setup", &caps.bem.hmatrix_setup),
        ("hmatrix", &caps.bem.hmatrix),
        ("fmm_setup", &caps.bem.fmm_setup),
        ("fmm", &caps.bem.fmm),
    ] {
        assert_eq!(table.f32, CapabilityState::Unsupported, "{label}.f32");
        assert_eq!(table.f64, CapabilityState::Unsupported, "{label}.f64");
        assert_eq!(table.c32, CapabilityState::Unsupported, "{label}.c32");
    }
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_capabilities_make_fallbacks_visible() {
    use sonium_compute::CudaBackend;

    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => skip_or_require_cuda!("capability contract test", e),
    };
    let caps = backend.capabilities();
    assert_eq!(caps.backend_name, "cuda");
    // cuFFT exposes complex-to-complex plans only.  Real scalar FFT
    // requests are rejected before plan creation, so their capability
    // cells must stay unsupported even when complex FFT is available.
    assert_eq!(caps.fft.f32, CapabilityState::Unsupported);
    assert_eq!(caps.fft.f64, CapabilityState::Unsupported);
    assert_eq!(caps.fft.c32, caps.fft.c64);
    assert!(matches!(
        caps.fft.c32,
        CapabilityState::VendorNative | CapabilityState::Unsupported
    ));
    assert_eq!(caps.blas1.c64, CapabilityState::VendorNative);
    assert_eq!(caps.gemv.f64, CapabilityState::VendorNative);
    assert_eq!(caps.gemm.c32, CapabilityState::VendorNative);
    assert_eq!(caps.csr_spmv.c64, CapabilityState::VendorNative);
    assert_eq!(caps.csr_transpose.c64, CapabilityState::VendorNative);
    assert_ne!(caps.bem_regular_pair_c32, CapabilityState::Unsupported);
    assert!(matches!(
        caps.bem_regular_pair_c64,
        CapabilityState::Native | CapabilityState::Unsupported
    ));
    assert_eq!(caps.bem.regular_pair_c32, caps.bem_regular_pair_c32);
    assert_eq!(caps.bem.regular_pair_c64, caps.bem_regular_pair_c64);
    assert_eq!(
        caps.bem.field_evaluation.c64,
        if backend.has_bem_regular_pair_c64() {
            CapabilityState::Native
        } else {
            CapabilityState::Unsupported
        },
        "CUDA field capability must track the loaded Complex64 field module"
    );
    let expected_hierarchical = if backend.has_fmm_c64() {
        CapabilityState::Native
    } else {
        CapabilityState::Unsupported
    };
    assert_eq!(caps.bem.hmatrix.c64, expected_hierarchical);
    assert_eq!(
        caps.bem.hmatrix_setup.c64,
        if backend.has_fmm_c64() {
            CapabilityState::HostSharedFallback
        } else {
            CapabilityState::Unsupported
        }
    );
    assert_eq!(caps.bem.fmm.c64, expected_hierarchical);
    assert_eq!(
        caps.bem.fmm_setup.c64,
        if backend.has_fmm_c64() {
            CapabilityState::HostSharedFallback
        } else {
            CapabilityState::Unsupported
        }
    );
    // Hierarchical operators are Complex64-only: the remaining scalar
    // cells stay unsupported unconditionally (no staging fallback).
    for (label, table) in [
        ("hmatrix_setup", &caps.bem.hmatrix_setup),
        ("hmatrix", &caps.bem.hmatrix),
        ("fmm_setup", &caps.bem.fmm_setup),
        ("fmm", &caps.bem.fmm),
    ] {
        assert_eq!(table.f32, CapabilityState::Unsupported, "{label}.f32");
        assert_eq!(table.f64, CapabilityState::Unsupported, "{label}.f64");
        assert_eq!(table.c32, CapabilityState::Unsupported, "{label}.c32");
    }
    assert_ne!(
        caps.multi_dot_strided.f64,
        CapabilityState::Unsupported,
        "CUDA must expose either native or host-staged strided multi-dot"
    );
    if env_flag("SONIUM_REQUIRE_CUDA") {
        assert_eq!(caps.multi_dot.f64, CapabilityState::Native);
        assert_eq!(caps.multi_dot_strided.f64, CapabilityState::Native);
        assert_eq!(caps.basis_axpy_strided.f64, CapabilityState::Native);
        assert_eq!(caps.precision_cast.f64_to_f32, CapabilityState::Native);
        assert_eq!(caps.gather4_weighted.c32, CapabilityState::Native);
    } else {
        assert_ne!(caps.multi_dot.f64, CapabilityState::Unsupported);
        assert_ne!(caps.precision_cast.f64_to_f32, CapabilityState::Unsupported);
        assert_ne!(caps.gather4_weighted.c32, CapabilityState::Unsupported);
    }
}

#[test]
#[cfg(feature = "metal")]
fn metal_capabilities_encode_mixed_precision_policy() {
    use sonium_compute::MetalBackend;

    let backend = match MetalBackend::new() {
        Ok(b) => b,
        Err(e) => skip_or_require_metal!("capability contract test", e),
    };
    let caps = backend.capabilities();
    assert_eq!(caps.backend_name, "metal");
    assert_eq!(caps.blas1.f32, CapabilityState::Native);
    assert_eq!(caps.blas1.f64, CapabilityState::HostSharedFallback);
    assert_eq!(caps.gemv.c32, CapabilityState::Native);
    // Metal's Stockham implementation accepts Complex32 only.  The
    // capability table must not advertise host fallback for real or
    // Complex64 buffers that the checked dispatch rejects.
    assert_eq!(caps.fft.f32, CapabilityState::Unsupported);
    assert_eq!(caps.fft.c32, CapabilityState::Native);
    assert_eq!(caps.fft.f64, CapabilityState::Unsupported);
    assert_eq!(caps.fft.c64, CapabilityState::Unsupported);
    assert_eq!(caps.gemm.f32, CapabilityState::HostSharedFallback);
    assert_eq!(caps.gemm.c32, CapabilityState::HostSharedFallback);
    assert_eq!(caps.csr_spmv.c64, CapabilityState::HostSharedFallback);
    assert_eq!(caps.csr_transpose.c64, CapabilityState::HostStagedFallback);
    assert_eq!(caps.multi_dot.f32, CapabilityState::Native);
    assert_eq!(caps.multi_dot.c32, CapabilityState::Native);
    assert_eq!(caps.multi_dot.c64, CapabilityState::HostSharedFallback);
    assert_eq!(caps.multi_dot_strided.f32, CapabilityState::Native);
    assert_eq!(caps.multi_dot_strided.c32, CapabilityState::Native);
    assert_eq!(caps.basis_axpy_strided.c32, CapabilityState::Native);
    assert_eq!(
        caps.multi_dot_strided.f64,
        CapabilityState::HostSharedFallback
    );
    assert_eq!(
        caps.multi_dot_strided.c64,
        CapabilityState::HostSharedFallback
    );
    assert_eq!(
        caps.basis_axpy_strided.f64,
        CapabilityState::HostSharedFallback
    );
    assert_eq!(
        caps.basis_axpy_strided.c64,
        CapabilityState::HostSharedFallback
    );
    assert_eq!(caps.gather4_weighted.f32, CapabilityState::Native);
    assert_eq!(caps.gather4_weighted.c32, CapabilityState::Native);
    assert_eq!(
        caps.gather4_weighted.c64,
        CapabilityState::HostStagedFallback
    );
    assert_eq!(
        caps.precision_cast.f64_to_f32,
        CapabilityState::HostSharedFallback
    );
    assert_eq!(
        caps.precision_cast.c32_to_c64,
        CapabilityState::HostSharedFallback
    );
    assert_eq!(caps.bem_regular_pair_c32, CapabilityState::Native);
    // Metal has no native hierarchical actions: forward tables are fully
    // unsupported and only Complex64 setup stages through the host.
    for (label, table) in [("hmatrix", &caps.bem.hmatrix), ("fmm", &caps.bem.fmm)] {
        assert_eq!(table.f32, CapabilityState::Unsupported, "{label}.f32");
        assert_eq!(table.f64, CapabilityState::Unsupported, "{label}.f64");
        assert_eq!(table.c32, CapabilityState::Unsupported, "{label}.c32");
        assert_eq!(table.c64, CapabilityState::Unsupported, "{label}.c64");
    }
    for (label, table) in [
        ("hmatrix_setup", &caps.bem.hmatrix_setup),
        ("fmm_setup", &caps.bem.fmm_setup),
    ] {
        assert_eq!(table.f32, CapabilityState::Unsupported, "{label}.f32");
        assert_eq!(table.f64, CapabilityState::Unsupported, "{label}.f64");
        assert_eq!(table.c32, CapabilityState::Unsupported, "{label}.c32");
        assert_eq!(
            table.c64,
            CapabilityState::HostSharedFallback,
            "{label}.c64"
        );
    }
}

#[test]
fn cpu_gather4_weighted_complex64_matches_expected() {
    use num_complex::Complex64;

    let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
    let values = vec![
        Complex64::new(1.0, 0.5),
        Complex64::new(2.0, -1.0),
        Complex64::new(-1.0, 3.0),
        Complex64::new(4.0, 2.0),
    ];
    let indices = vec![0, 1, 2, 3, 3, 2, 1, 0];
    let weights = vec![0.25, 0.25, 0.25, 0.25, 1.0, 0.0, 0.0, 0.0];
    let expected = [Complex64::new(1.5, 1.125), Complex64::new(4.0, 2.0)];

    let mut values_dev = pool.try_alloc::<Complex64>(values.len()).unwrap();
    let mut indices_dev = pool.try_alloc::<i32>(indices.len()).unwrap();
    let mut weights_dev = pool.try_alloc::<f64>(weights.len()).unwrap();
    let mut out_dev = pool.try_alloc::<Complex64>(expected.len()).unwrap();
    pool.backend().copy_h2d(&values, values_dev.raw_mut());
    pool.backend().copy_h2d(&indices, indices_dev.raw_mut());
    pool.backend().copy_h2d(&weights, weights_dev.raw_mut());
    pool.backend().gather4_weighted(
        values_dev.raw(),
        indices_dev.raw(),
        weights_dev.raw(),
        out_dev.raw_mut(),
    );

    let mut out = vec![Complex64::new(0.0, 0.0); expected.len()];
    pool.backend().copy_d2h(out_dev.raw(), &mut out);
    for i in 0..expected.len() {
        assert_close(&format!("cpu/gather4[{i}].re"), out[i].re, expected[i].re);
        assert_close(&format!("cpu/gather4[{i}].im"), out[i].im, expected[i].im);
    }
}

#[test]
fn cpu_batched_gemv_gemm_match_individual_dispatch() {
    let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();

    let a = vec![1.0_f64, 2.0, 3.0, 4.0, 0.5, -1.0];
    let x = vec![0.25_f64, -0.5, 2.0];
    let y0 = vec![1.0_f64, -2.0];
    let mut a_dev = pool.try_alloc::<f64>(a.len()).unwrap();
    let mut x_dev = pool.try_alloc::<f64>(x.len()).unwrap();
    let mut y_batch = pool.try_alloc::<f64>(y0.len()).unwrap();
    let mut y_ref = pool.try_alloc::<f64>(y0.len()).unwrap();
    pool.backend().copy_h2d(&a, a_dev.raw_mut());
    pool.backend().copy_h2d(&x, x_dev.raw_mut());
    pool.backend().copy_h2d(&y0, y_batch.raw_mut());
    pool.backend().copy_h2d(&y0, y_ref.raw_mut());

    {
        let mut batch = [GemvBatch {
            op: sonium_compute::Transpose::NoTrans,
            m: 2,
            n: 3,
            alpha: 0.75,
            a: a_dev.raw(),
            lda: 3,
            x: x_dev.raw(),
            beta: 0.25,
            y: y_batch.raw_mut(),
        }];
        pool.backend().gemv_batched(&mut batch);
    }
    pool.backend().gemv(
        sonium_compute::Transpose::NoTrans,
        2,
        3,
        0.75,
        a_dev.raw(),
        3,
        x_dev.raw(),
        0.25,
        y_ref.raw_mut(),
    );
    let mut got = vec![0.0; y0.len()];
    let mut want = vec![0.0; y0.len()];
    pool.backend().copy_d2h(y_batch.raw(), &mut got);
    pool.backend().copy_d2h(y_ref.raw(), &mut want);
    assert_vec_close("cpu/gemv_batched", &got, &want);

    let b = vec![0.25_f64, 1.5, -2.0, 0.75, 1.0, -0.5];
    let c0 = vec![0.5_f64, -1.0, 2.0, 1.5];
    let mut b_dev = pool.try_alloc::<f64>(b.len()).unwrap();
    let mut c_batch = pool.try_alloc::<f64>(c0.len()).unwrap();
    let mut c_ref = pool.try_alloc::<f64>(c0.len()).unwrap();
    pool.backend().copy_h2d(&b, b_dev.raw_mut());
    pool.backend().copy_h2d(&c0, c_batch.raw_mut());
    pool.backend().copy_h2d(&c0, c_ref.raw_mut());

    {
        let mut batch = [GemmBatch {
            transa: sonium_compute::Transpose::NoTrans,
            transb: sonium_compute::Transpose::NoTrans,
            m: 2,
            n: 2,
            k: 3,
            alpha: 1.25,
            a: a_dev.raw(),
            lda: 3,
            b: b_dev.raw(),
            ldb: 2,
            beta: 0.5,
            c: c_batch.raw_mut(),
            ldc: 2,
        }];
        pool.backend().gemm_batched(&mut batch);
    }
    pool.backend().gemm(
        sonium_compute::Transpose::NoTrans,
        sonium_compute::Transpose::NoTrans,
        2,
        2,
        3,
        1.25,
        a_dev.raw(),
        3,
        b_dev.raw(),
        2,
        0.5,
        c_ref.raw_mut(),
        2,
    );
    let mut got = vec![0.0; c0.len()];
    let mut want = vec![0.0; c0.len()];
    pool.backend().copy_d2h(c_batch.raw(), &mut got);
    pool.backend().copy_d2h(c_ref.raw(), &mut want);
    assert_vec_close("cpu/gemm_batched", &got, &want);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_batched_gemv_gemm_match_individual_dispatch() {
    use sonium_compute::CudaBackend;

    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => skip_or_require_cuda!("batched gemv/gemm parity", e),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();

    let a0 = vec![1.0_f64, 2.0, 3.0, 4.0, 0.5, -1.0];
    let a1 = vec![-0.25_f64, 0.75, 1.25, -1.5, 2.0, 0.5];
    let x0 = vec![0.25_f64, -0.5, 2.0];
    let x1 = vec![1.5_f64, 0.25, -0.75];
    let y0 = vec![1.0_f64, -2.0];
    let y1 = vec![0.5_f64, 1.25];

    let mut a0_dev = pool.try_alloc::<f64>(a0.len()).unwrap();
    let mut a1_dev = pool.try_alloc::<f64>(a1.len()).unwrap();
    let mut x0_dev = pool.try_alloc::<f64>(x0.len()).unwrap();
    let mut x1_dev = pool.try_alloc::<f64>(x1.len()).unwrap();
    let mut y0_batch = pool.try_alloc::<f64>(y0.len()).unwrap();
    let mut y1_batch = pool.try_alloc::<f64>(y1.len()).unwrap();
    let mut y0_ref = pool.try_alloc::<f64>(y0.len()).unwrap();
    let mut y1_ref = pool.try_alloc::<f64>(y1.len()).unwrap();
    pool.backend().copy_h2d(&a0, a0_dev.raw_mut());
    pool.backend().copy_h2d(&a1, a1_dev.raw_mut());
    pool.backend().copy_h2d(&x0, x0_dev.raw_mut());
    pool.backend().copy_h2d(&x1, x1_dev.raw_mut());
    pool.backend().copy_h2d(&y0, y0_batch.raw_mut());
    pool.backend().copy_h2d(&y1, y1_batch.raw_mut());
    pool.backend().copy_h2d(&y0, y0_ref.raw_mut());
    pool.backend().copy_h2d(&y1, y1_ref.raw_mut());

    {
        let mut batch = [
            GemvBatch {
                op: sonium_compute::Transpose::NoTrans,
                m: 2,
                n: 3,
                alpha: 0.75,
                a: a0_dev.raw(),
                lda: 3,
                x: x0_dev.raw(),
                beta: 0.25,
                y: y0_batch.raw_mut(),
            },
            GemvBatch {
                op: sonium_compute::Transpose::NoTrans,
                m: 2,
                n: 3,
                alpha: 0.75,
                a: a1_dev.raw(),
                lda: 3,
                x: x1_dev.raw(),
                beta: 0.25,
                y: y1_batch.raw_mut(),
            },
        ];
        pool.backend().gemv_batched(&mut batch);
    }
    pool.backend().gemv(
        sonium_compute::Transpose::NoTrans,
        2,
        3,
        0.75,
        a0_dev.raw(),
        3,
        x0_dev.raw(),
        0.25,
        y0_ref.raw_mut(),
    );
    pool.backend().gemv(
        sonium_compute::Transpose::NoTrans,
        2,
        3,
        0.75,
        a1_dev.raw(),
        3,
        x1_dev.raw(),
        0.25,
        y1_ref.raw_mut(),
    );
    let mut got = vec![0.0; y0.len()];
    let mut want = vec![0.0; y0.len()];
    pool.backend().copy_d2h(y0_batch.raw(), &mut got);
    pool.backend().copy_d2h(y0_ref.raw(), &mut want);
    assert_vec_close("cuda/gemv_batched[0]", &got, &want);
    pool.backend().copy_d2h(y1_batch.raw(), &mut got);
    pool.backend().copy_d2h(y1_ref.raw(), &mut want);
    assert_vec_close("cuda/gemv_batched[1]", &got, &want);

    let b0 = vec![0.25_f64, 1.5, -2.0, 0.75, 1.0, -0.5];
    let b1 = vec![1.0_f64, -0.25, 0.5, 2.0, -1.5, 0.75];
    let c0 = vec![0.5_f64, -1.0, 2.0, 1.5];
    let c1 = vec![-0.75_f64, 0.25, 1.25, -2.0];
    let mut b0_dev = pool.try_alloc::<f64>(b0.len()).unwrap();
    let mut b1_dev = pool.try_alloc::<f64>(b1.len()).unwrap();
    let mut c0_batch = pool.try_alloc::<f64>(c0.len()).unwrap();
    let mut c1_batch = pool.try_alloc::<f64>(c1.len()).unwrap();
    let mut c0_ref = pool.try_alloc::<f64>(c0.len()).unwrap();
    let mut c1_ref = pool.try_alloc::<f64>(c1.len()).unwrap();
    pool.backend().copy_h2d(&b0, b0_dev.raw_mut());
    pool.backend().copy_h2d(&b1, b1_dev.raw_mut());
    pool.backend().copy_h2d(&c0, c0_batch.raw_mut());
    pool.backend().copy_h2d(&c1, c1_batch.raw_mut());
    pool.backend().copy_h2d(&c0, c0_ref.raw_mut());
    pool.backend().copy_h2d(&c1, c1_ref.raw_mut());

    {
        let mut batch = [
            GemmBatch {
                transa: sonium_compute::Transpose::NoTrans,
                transb: sonium_compute::Transpose::NoTrans,
                m: 2,
                n: 2,
                k: 3,
                alpha: 1.25,
                a: a0_dev.raw(),
                lda: 3,
                b: b0_dev.raw(),
                ldb: 2,
                beta: 0.5,
                c: c0_batch.raw_mut(),
                ldc: 2,
            },
            GemmBatch {
                transa: sonium_compute::Transpose::NoTrans,
                transb: sonium_compute::Transpose::NoTrans,
                m: 2,
                n: 2,
                k: 3,
                alpha: 1.25,
                a: a1_dev.raw(),
                lda: 3,
                b: b1_dev.raw(),
                ldb: 2,
                beta: 0.5,
                c: c1_batch.raw_mut(),
                ldc: 2,
            },
        ];
        pool.backend().gemm_batched(&mut batch);
    }
    pool.backend().gemm(
        sonium_compute::Transpose::NoTrans,
        sonium_compute::Transpose::NoTrans,
        2,
        2,
        3,
        1.25,
        a0_dev.raw(),
        3,
        b0_dev.raw(),
        2,
        0.5,
        c0_ref.raw_mut(),
        2,
    );
    pool.backend().gemm(
        sonium_compute::Transpose::NoTrans,
        sonium_compute::Transpose::NoTrans,
        2,
        2,
        3,
        1.25,
        a1_dev.raw(),
        3,
        b1_dev.raw(),
        2,
        0.5,
        c1_ref.raw_mut(),
        2,
    );
    let mut got = vec![0.0; c0.len()];
    let mut want = vec![0.0; c0.len()];
    pool.backend().copy_d2h(c0_batch.raw(), &mut got);
    pool.backend().copy_d2h(c0_ref.raw(), &mut want);
    assert_vec_close("cuda/gemm_batched[0]", &got, &want);
    pool.backend().copy_d2h(c1_batch.raw(), &mut got);
    pool.backend().copy_d2h(c1_ref.raw(), &mut want);
    assert_vec_close("cuda/gemm_batched[1]", &got, &want);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_batched_gemv_transpose_and_fallback_match_individual_dispatch() {
    use sonium_compute::{CudaBackend, Transpose};

    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => skip_or_require_cuda!("batched gemv transpose parity", e),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();

    let a0 = vec![1.0_f64, 2.0, 3.0, 4.0, 0.5, -1.0];
    let a1 = vec![-0.25_f64, 0.75, 1.25, -1.5, 2.0, 0.5];
    let x0 = vec![0.25_f64, -0.5];
    let x1 = vec![1.5_f64, 0.25];
    let y0 = vec![1.0_f64, -2.0, 0.5];
    let y1 = vec![0.5_f64, 1.25, -0.75];

    let mut a0_dev = pool.try_alloc::<f64>(a0.len()).unwrap();
    let mut a1_dev = pool.try_alloc::<f64>(a1.len()).unwrap();
    let mut x0_dev = pool.try_alloc::<f64>(x0.len()).unwrap();
    let mut x1_dev = pool.try_alloc::<f64>(x1.len()).unwrap();
    let mut y0_batch = pool.try_alloc::<f64>(y0.len()).unwrap();
    let mut y1_batch = pool.try_alloc::<f64>(y1.len()).unwrap();
    let mut y0_ref = pool.try_alloc::<f64>(y0.len()).unwrap();
    let mut y1_ref = pool.try_alloc::<f64>(y1.len()).unwrap();
    pool.backend().copy_h2d(&a0, a0_dev.raw_mut());
    pool.backend().copy_h2d(&a1, a1_dev.raw_mut());
    pool.backend().copy_h2d(&x0, x0_dev.raw_mut());
    pool.backend().copy_h2d(&x1, x1_dev.raw_mut());
    pool.backend().copy_h2d(&y0, y0_batch.raw_mut());
    pool.backend().copy_h2d(&y1, y1_batch.raw_mut());
    pool.backend().copy_h2d(&y0, y0_ref.raw_mut());
    pool.backend().copy_h2d(&y1, y1_ref.raw_mut());

    {
        let mut batch = [
            GemvBatch {
                op: Transpose::Trans,
                m: 2,
                n: 3,
                alpha: 0.75,
                a: a0_dev.raw(),
                lda: 3,
                x: x0_dev.raw(),
                beta: 0.25,
                y: y0_batch.raw_mut(),
            },
            GemvBatch {
                op: Transpose::Trans,
                m: 2,
                n: 3,
                alpha: 0.75,
                a: a1_dev.raw(),
                lda: 3,
                x: x1_dev.raw(),
                beta: 0.25,
                y: y1_batch.raw_mut(),
            },
        ];
        pool.backend().gemv_batched(&mut batch);
    }
    pool.backend().gemv(
        Transpose::Trans,
        2,
        3,
        0.75,
        a0_dev.raw(),
        3,
        x0_dev.raw(),
        0.25,
        y0_ref.raw_mut(),
    );
    pool.backend().gemv(
        Transpose::Trans,
        2,
        3,
        0.75,
        a1_dev.raw(),
        3,
        x1_dev.raw(),
        0.25,
        y1_ref.raw_mut(),
    );

    let mut got = vec![0.0; y0.len()];
    let mut want = vec![0.0; y0.len()];
    pool.backend().copy_d2h(y0_batch.raw(), &mut got);
    pool.backend().copy_d2h(y0_ref.raw(), &mut want);
    assert_vec_close("cuda/gemv_batched_trans[0]", &got, &want);
    pool.backend().copy_d2h(y1_batch.raw(), &mut got);
    pool.backend().copy_d2h(y1_ref.raw(), &mut want);
    assert_vec_close("cuda/gemv_batched_trans[1]", &got, &want);

    let x_notrans = vec![0.25_f64, -0.5, 2.0];
    let y_notrans = vec![1.0_f64, -2.0];
    let mut x_notrans_dev = pool.try_alloc::<f64>(x_notrans.len()).unwrap();
    let mut y_notrans_batch = pool.try_alloc::<f64>(y_notrans.len()).unwrap();
    let mut y_notrans_ref = pool.try_alloc::<f64>(y_notrans.len()).unwrap();
    pool.backend().copy_h2d(&x_notrans, x_notrans_dev.raw_mut());
    pool.backend()
        .copy_h2d(&y_notrans, y_notrans_batch.raw_mut());
    pool.backend().copy_h2d(&y_notrans, y_notrans_ref.raw_mut());
    pool.backend().copy_h2d(&y1, y1_batch.raw_mut());
    pool.backend().copy_h2d(&y1, y1_ref.raw_mut());

    {
        let mut batch = [
            GemvBatch {
                op: Transpose::NoTrans,
                m: 2,
                n: 3,
                alpha: 0.5,
                a: a0_dev.raw(),
                lda: 3,
                x: x_notrans_dev.raw(),
                beta: -0.25,
                y: y_notrans_batch.raw_mut(),
            },
            GemvBatch {
                op: Transpose::Trans,
                m: 2,
                n: 3,
                alpha: 0.75,
                a: a1_dev.raw(),
                lda: 3,
                x: x1_dev.raw(),
                beta: 0.25,
                y: y1_batch.raw_mut(),
            },
        ];
        pool.backend().gemv_batched(&mut batch);
    }
    pool.backend().gemv(
        Transpose::NoTrans,
        2,
        3,
        0.5,
        a0_dev.raw(),
        3,
        x_notrans_dev.raw(),
        -0.25,
        y_notrans_ref.raw_mut(),
    );
    pool.backend().gemv(
        Transpose::Trans,
        2,
        3,
        0.75,
        a1_dev.raw(),
        3,
        x1_dev.raw(),
        0.25,
        y1_ref.raw_mut(),
    );
    let mut got_notrans = vec![0.0; y_notrans.len()];
    let mut want_notrans = vec![0.0; y_notrans.len()];
    pool.backend()
        .copy_d2h(y_notrans_batch.raw(), &mut got_notrans);
    pool.backend()
        .copy_d2h(y_notrans_ref.raw(), &mut want_notrans);
    assert_vec_close(
        "cuda/gemv_batched_heterogeneous[0]",
        &got_notrans,
        &want_notrans,
    );
    pool.backend().copy_d2h(y1_batch.raw(), &mut got);
    pool.backend().copy_d2h(y1_ref.raw(), &mut want);
    assert_vec_close("cuda/gemv_batched_heterogeneous[1]", &got, &want);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_complex_batched_gemv_gemm_match_individual_dispatch() {
    use sonium_compute::{CudaBackend, Transpose};

    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => skip_or_require_cuda!("complex batched gemv/gemm parity", e),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();

    let a0 = vec![
        Complex64::new(1.0, 0.25),
        Complex64::new(2.0, -0.5),
        Complex64::new(3.0, 0.75),
        Complex64::new(4.0, -1.0),
        Complex64::new(0.5, 0.5),
        Complex64::new(-1.0, 0.25),
    ];
    let a1 = vec![
        Complex64::new(-0.25, 0.5),
        Complex64::new(0.75, -0.25),
        Complex64::new(1.25, 1.0),
        Complex64::new(-1.5, 0.75),
        Complex64::new(2.0, -0.5),
        Complex64::new(0.5, 0.25),
    ];
    let x0 = vec![
        Complex64::new(0.25, 0.5),
        Complex64::new(-0.5, 0.25),
        Complex64::new(2.0, -0.75),
    ];
    let x1 = vec![
        Complex64::new(1.5, -0.25),
        Complex64::new(0.25, 0.5),
        Complex64::new(-0.75, 1.0),
    ];
    let y0 = vec![Complex64::new(1.0, -0.25), Complex64::new(-2.0, 0.5)];
    let y1 = vec![Complex64::new(0.5, 0.75), Complex64::new(1.25, -0.5)];

    let mut a0_dev = pool.try_alloc::<Complex64>(a0.len()).unwrap();
    let mut a1_dev = pool.try_alloc::<Complex64>(a1.len()).unwrap();
    let mut x0_dev = pool.try_alloc::<Complex64>(x0.len()).unwrap();
    let mut x1_dev = pool.try_alloc::<Complex64>(x1.len()).unwrap();
    let mut y0_batch = pool.try_alloc::<Complex64>(y0.len()).unwrap();
    let mut y1_batch = pool.try_alloc::<Complex64>(y1.len()).unwrap();
    let mut y0_ref = pool.try_alloc::<Complex64>(y0.len()).unwrap();
    let mut y1_ref = pool.try_alloc::<Complex64>(y1.len()).unwrap();
    pool.backend().copy_h2d(&a0, a0_dev.raw_mut());
    pool.backend().copy_h2d(&a1, a1_dev.raw_mut());
    pool.backend().copy_h2d(&x0, x0_dev.raw_mut());
    pool.backend().copy_h2d(&x1, x1_dev.raw_mut());
    pool.backend().copy_h2d(&y0, y0_batch.raw_mut());
    pool.backend().copy_h2d(&y1, y1_batch.raw_mut());
    pool.backend().copy_h2d(&y0, y0_ref.raw_mut());
    pool.backend().copy_h2d(&y1, y1_ref.raw_mut());

    {
        let mut batch = [
            GemvBatch {
                op: Transpose::NoTrans,
                m: 2,
                n: 3,
                alpha: Complex64::new(0.75, -0.25),
                a: a0_dev.raw(),
                lda: 3,
                x: x0_dev.raw(),
                beta: Complex64::new(0.25, 0.5),
                y: y0_batch.raw_mut(),
            },
            GemvBatch {
                op: Transpose::NoTrans,
                m: 2,
                n: 3,
                alpha: Complex64::new(0.75, -0.25),
                a: a1_dev.raw(),
                lda: 3,
                x: x1_dev.raw(),
                beta: Complex64::new(0.25, 0.5),
                y: y1_batch.raw_mut(),
            },
        ];
        pool.backend().gemv_batched(&mut batch);
    }
    pool.backend().gemv(
        Transpose::NoTrans,
        2,
        3,
        Complex64::new(0.75, -0.25),
        a0_dev.raw(),
        3,
        x0_dev.raw(),
        Complex64::new(0.25, 0.5),
        y0_ref.raw_mut(),
    );
    pool.backend().gemv(
        Transpose::NoTrans,
        2,
        3,
        Complex64::new(0.75, -0.25),
        a1_dev.raw(),
        3,
        x1_dev.raw(),
        Complex64::new(0.25, 0.5),
        y1_ref.raw_mut(),
    );
    let mut got = vec![Complex64::new(0.0, 0.0); y0.len()];
    let mut want = vec![Complex64::new(0.0, 0.0); y0.len()];
    pool.backend().copy_d2h(y0_batch.raw(), &mut got);
    pool.backend().copy_d2h(y0_ref.raw(), &mut want);
    assert_complex64_vec_close("cuda/complex_gemv_batched[0]", &got, &want);
    pool.backend().copy_d2h(y1_batch.raw(), &mut got);
    pool.backend().copy_d2h(y1_ref.raw(), &mut want);
    assert_complex64_vec_close("cuda/complex_gemv_batched[1]", &got, &want);

    let b0 = vec![
        Complex64::new(0.25, 0.5),
        Complex64::new(1.5, -0.25),
        Complex64::new(-2.0, 0.75),
        Complex64::new(0.75, 0.5),
        Complex64::new(1.0, -1.0),
        Complex64::new(-0.5, 0.25),
    ];
    let b1 = vec![
        Complex64::new(1.0, -0.25),
        Complex64::new(-0.25, 0.75),
        Complex64::new(0.5, -0.5),
        Complex64::new(2.0, 0.25),
        Complex64::new(-1.5, 1.0),
        Complex64::new(0.75, -0.75),
    ];
    let c0 = vec![
        Complex64::new(0.5, 0.25),
        Complex64::new(-1.0, 0.75),
        Complex64::new(2.0, -0.5),
        Complex64::new(1.5, 0.5),
    ];
    let c1 = vec![
        Complex64::new(-0.75, 0.25),
        Complex64::new(0.25, -0.5),
        Complex64::new(1.25, 0.75),
        Complex64::new(-2.0, -0.25),
    ];
    let mut b0_dev = pool.try_alloc::<Complex64>(b0.len()).unwrap();
    let mut b1_dev = pool.try_alloc::<Complex64>(b1.len()).unwrap();
    let mut c0_batch = pool.try_alloc::<Complex64>(c0.len()).unwrap();
    let mut c1_batch = pool.try_alloc::<Complex64>(c1.len()).unwrap();
    let mut c0_ref = pool.try_alloc::<Complex64>(c0.len()).unwrap();
    let mut c1_ref = pool.try_alloc::<Complex64>(c1.len()).unwrap();
    pool.backend().copy_h2d(&b0, b0_dev.raw_mut());
    pool.backend().copy_h2d(&b1, b1_dev.raw_mut());
    pool.backend().copy_h2d(&c0, c0_batch.raw_mut());
    pool.backend().copy_h2d(&c1, c1_batch.raw_mut());
    pool.backend().copy_h2d(&c0, c0_ref.raw_mut());
    pool.backend().copy_h2d(&c1, c1_ref.raw_mut());

    {
        let mut batch = [
            GemmBatch {
                transa: Transpose::NoTrans,
                transb: Transpose::NoTrans,
                m: 2,
                n: 2,
                k: 3,
                alpha: Complex64::new(1.25, -0.5),
                a: a0_dev.raw(),
                lda: 3,
                b: b0_dev.raw(),
                ldb: 2,
                beta: Complex64::new(0.5, 0.25),
                c: c0_batch.raw_mut(),
                ldc: 2,
            },
            GemmBatch {
                transa: Transpose::NoTrans,
                transb: Transpose::NoTrans,
                m: 2,
                n: 2,
                k: 3,
                alpha: Complex64::new(1.25, -0.5),
                a: a1_dev.raw(),
                lda: 3,
                b: b1_dev.raw(),
                ldb: 2,
                beta: Complex64::new(0.5, 0.25),
                c: c1_batch.raw_mut(),
                ldc: 2,
            },
        ];
        pool.backend().gemm_batched(&mut batch);
    }
    pool.backend().gemm(
        Transpose::NoTrans,
        Transpose::NoTrans,
        2,
        2,
        3,
        Complex64::new(1.25, -0.5),
        a0_dev.raw(),
        3,
        b0_dev.raw(),
        2,
        Complex64::new(0.5, 0.25),
        c0_ref.raw_mut(),
        2,
    );
    pool.backend().gemm(
        Transpose::NoTrans,
        Transpose::NoTrans,
        2,
        2,
        3,
        Complex64::new(1.25, -0.5),
        a1_dev.raw(),
        3,
        b1_dev.raw(),
        2,
        Complex64::new(0.5, 0.25),
        c1_ref.raw_mut(),
        2,
    );
    let mut got = vec![Complex64::new(0.0, 0.0); c0.len()];
    let mut want = vec![Complex64::new(0.0, 0.0); c0.len()];
    pool.backend().copy_d2h(c0_batch.raw(), &mut got);
    pool.backend().copy_d2h(c0_ref.raw(), &mut want);
    assert_complex64_vec_close("cuda/complex_gemm_batched[0]", &got, &want);
    pool.backend().copy_d2h(c1_batch.raw(), &mut got);
    pool.backend().copy_d2h(c1_ref.raw(), &mut want);
    assert_complex64_vec_close("cuda/complex_gemm_batched[1]", &got, &want);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_spmm_prebaked_colmajor_matches_repeated_spmv() {
    use sonium_compute::CudaBackend;

    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => skip_or_require_cuda!("SpMM parity", e),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();

    let m = 3usize;
    let n = 3usize;
    let rhs = 2usize;
    let row_ptr = vec![0_i32, 2, 4, 6];
    let col_idx = vec![0_i32, 1, 1, 2, 0, 2];
    let values = vec![2.0_f64, -1.0, 3.0, 0.5, -2.0, 4.0];
    // Column-major 3 x 2 dense RHS: first column then second column.
    let x = vec![1.0_f64, 2.0, -1.0, 0.5, -0.25, 3.0];

    let mut row_dev = pool.try_alloc::<i32>(row_ptr.len()).unwrap();
    let mut col_dev = pool.try_alloc::<i32>(col_idx.len()).unwrap();
    let mut val_dev = pool.try_alloc::<f64>(values.len()).unwrap();
    let mut x_dev = pool.try_alloc::<f64>(x.len()).unwrap();
    let mut y_spmm = pool.try_alloc::<f64>(m * rhs).unwrap();
    let mut y_ref = pool.try_alloc::<f64>(m * rhs).unwrap();
    pool.backend().copy_h2d(&row_ptr, row_dev.raw_mut());
    pool.backend().copy_h2d(&col_idx, col_dev.raw_mut());
    pool.backend().copy_h2d(&values, val_dev.raw_mut());
    pool.backend().copy_h2d(&x, x_dev.raw_mut());
    pool.backend()
        .copy_h2d(&vec![0.0_f64; m * rhs], y_spmm.raw_mut());
    pool.backend()
        .copy_h2d(&vec![0.0_f64; m * rhs], y_ref.raw_mut());

    let prebaked =
        pool.backend()
            .prebake_csr::<f64>(m, n, row_dev.raw(), col_dev.raw(), val_dev.raw());
    pool.backend()
        .spmm_prebaked_colmajor(&prebaked, rhs, x_dev.raw(), y_spmm.raw_mut());

    let mut expected = vec![0.0_f64; m * rhs];
    for col in 0..rhs {
        let x_col = &x[col * n..(col + 1) * n];
        for row in 0..m {
            let start = row_ptr[row] as usize;
            let end = row_ptr[row + 1] as usize;
            let mut acc = 0.0;
            for idx in start..end {
                acc += values[idx] * x_col[col_idx[idx] as usize];
            }
            expected[col * m + row] = acc;
        }
    }
    pool.backend().copy_h2d(&expected, y_ref.raw_mut());

    let mut got = vec![0.0; m * rhs];
    let mut want = vec![0.0; m * rhs];
    pool.backend().copy_d2h(y_spmm.raw(), &mut got);
    pool.backend().copy_d2h(y_ref.raw(), &mut want);
    assert_vec_close("cuda/spmm_prebaked_colmajor", &got, &want);
}

/// Row-split (`ALG2`) SpMM path (`nrhs >= 16`) matches the CPU reference:
/// locks the algorithm-selection branch the `rhs = 2` test above never
/// reaches (different cuSPARSE algorithm, different summation order, so
/// tolerance — not bitwise — comparison).
#[test]
#[cfg(feature = "cuda")]
fn cuda_spmm_prebaked_colmajor_wide_rhs_matches_cpu_reference() {
    use sonium_compute::CudaBackend;

    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => skip_or_require_cuda!("SpMM wide-RHS parity", e),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();

    let m = 3usize;
    let n = 3usize;
    let rhs = 16usize;
    let row_ptr = vec![0_i32, 2, 4, 6];
    let col_idx = vec![0_i32, 1, 1, 2, 0, 2];
    let values = vec![2.0_f64, -1.0, 3.0, 0.5, -2.0, 4.0];
    // Column-major 3 x 16 dense RHS: deterministic pseudo-random columns.
    let x: Vec<f64> = (0..n * rhs)
        .map(|i| ((i * 2654435761usize % 1000) as f64) / 1000.0 - 0.5)
        .collect();

    let mut row_dev = pool.try_alloc::<i32>(row_ptr.len()).unwrap();
    let mut col_dev = pool.try_alloc::<i32>(col_idx.len()).unwrap();
    let mut val_dev = pool.try_alloc::<f64>(values.len()).unwrap();
    let mut x_dev = pool.try_alloc::<f64>(x.len()).unwrap();
    let mut y_spmm = pool.try_alloc::<f64>(m * rhs).unwrap();
    let mut y_ref = pool.try_alloc::<f64>(m * rhs).unwrap();
    pool.backend().copy_h2d(&row_ptr, row_dev.raw_mut());
    pool.backend().copy_h2d(&col_idx, col_dev.raw_mut());
    pool.backend().copy_h2d(&values, val_dev.raw_mut());
    pool.backend().copy_h2d(&x, x_dev.raw_mut());
    pool.backend()
        .copy_h2d(&vec![0.0_f64; m * rhs], y_spmm.raw_mut());
    pool.backend()
        .copy_h2d(&vec![0.0_f64; m * rhs], y_ref.raw_mut());

    let prebaked =
        pool.backend()
            .prebake_csr::<f64>(m, n, row_dev.raw(), col_dev.raw(), val_dev.raw());
    pool.backend()
        .spmm_prebaked_colmajor(&prebaked, rhs, x_dev.raw(), y_spmm.raw_mut());

    let mut expected = vec![0.0_f64; m * rhs];
    for col in 0..rhs {
        let x_col = &x[col * n..(col + 1) * n];
        for row in 0..m {
            let start = row_ptr[row] as usize;
            let end = row_ptr[row + 1] as usize;
            let mut acc = 0.0;
            for idx in start..end {
                acc += values[idx] * x_col[col_idx[idx] as usize];
            }
            expected[col * m + row] = acc;
        }
    }
    pool.backend().copy_h2d(&expected, y_ref.raw_mut());

    let mut got = vec![0.0; m * rhs];
    let mut want = vec![0.0; m * rhs];
    pool.backend().copy_d2h(y_spmm.raw(), &mut got);
    pool.backend().copy_d2h(y_ref.raw(), &mut want);
    assert_vec_close("cuda/spmm_prebaked_colmajor_wide_rhs", &got, &want);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_csr_transpose_matches_cpu_reference() {
    use sonium_compute::{CapabilityState, CudaBackend};

    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => skip_or_require_cuda!("CSR transpose parity", e),
    };
    assert_eq!(
        backend.capabilities().csr_transpose.f64,
        CapabilityState::VendorNative
    );
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
    let mut row_ptr = pool.try_alloc::<i32>(3).unwrap();
    let mut col_idx = pool.try_alloc::<i32>(3).unwrap();
    let mut values = pool.try_alloc::<f64>(3).unwrap();
    let mut out_row_ptr = pool.try_alloc::<i32>(4).unwrap();
    let mut out_col_idx = pool.try_alloc::<i32>(3).unwrap();
    let mut out_values = pool.try_alloc::<f64>(3).unwrap();
    pool.backend().copy_h2d(&[0, 2, 3], row_ptr.raw_mut());
    pool.backend().copy_h2d(&[2, 0, 1], col_idx.raw_mut());
    pool.backend()
        .copy_h2d(&[10.0, 20.0, 30.0], values.raw_mut());
    pool.backend()
        .try_csr_transpose(
            2,
            3,
            row_ptr.raw(),
            col_idx.raw(),
            values.raw(),
            out_row_ptr.raw_mut(),
            out_col_idx.raw_mut(),
            out_values.raw_mut(),
        )
        .expect("CUDA CSR transpose");

    let mut got_row_ptr = [0; 4];
    let mut got_col_idx = [0; 3];
    let mut got_values = [0.0; 3];
    pool.backend().copy_d2h(out_row_ptr.raw(), &mut got_row_ptr);
    pool.backend().copy_d2h(out_col_idx.raw(), &mut got_col_idx);
    pool.backend().copy_d2h(out_values.raw(), &mut got_values);
    assert_eq!(got_row_ptr, [0, 1, 2, 3]);
    assert_eq!(got_col_idx, [0, 1, 0]);
    assert_eq!(got_values, [20.0, 30.0, 10.0]);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_graph_replays_stable_native_kernel_segment() {
    use sonium_compute::CudaBackend;

    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => skip_or_require_cuda!("CUDA graph replay", e),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();

    let m = 5usize;
    let x = vec![1.0_f64, 2.0, 4.0, 8.0, 16.0];

    let mut x_dev = pool.try_alloc::<f64>(x.len()).unwrap();
    let mut y_dev = pool.try_alloc::<f32>(m).unwrap();
    pool.backend().copy_h2d(&x, x_dev.raw_mut());
    pool.backend().copy_h2d(&vec![0.0_f32; m], y_dev.raw_mut());

    let graph = pool
        .backend()
        .try_capture_graph_checked(|backend| {
            // cuSPARSE's generic SpMV descriptor path is not stream-capture
            // compatible on every supported driver.  Keep this fixture on a
            // capture-safe native kernel; callers can still use the checked
            // capture API to reject unsupported descriptor sequences.
            backend.try_cast_to_lower(x_dev.raw(), y_dev.raw_mut())
        })
        .expect("CUDA graph capture should succeed");

    graph
        .try_launch()
        .expect("CUDA graph launch should succeed");
    graph
        .try_launch()
        .expect("CUDA graph relaunch should succeed");
    pool.backend()
        .try_synchronize()
        .expect("backend synchronization failed");

    let expected: Vec<f32> = x.iter().map(|&value| value as f32).collect();
    let mut got = vec![0.0_f32; m];
    pool.backend().copy_d2h(y_dev.raw(), &mut got);
    assert_vec_close_f32("cuda/graph_cast", &got, &expected);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_graph_replays_device_resident_krylov_update() {
    use sonium_compute::CudaBackend;

    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => skip_or_require_cuda!("CUDA graph Krylov update", e),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();

    // Two resident Arnoldi vectors and their resident projection coefficients.
    // The captured command is the same fused device update used by block
    // Krylov: y <- y + V * coefficients. No coefficient download/upload or
    // host synchronization occurs inside the graph callback or either replay.
    let vector_len = 5usize;
    let basis_count = 2usize;
    let mut basis = pool.try_alloc::<f64>(vector_len * basis_count).unwrap();
    let mut coefficients = pool.try_alloc::<f64>(basis_count).unwrap();
    let mut y = pool.try_alloc::<f64>(vector_len).unwrap();
    pool.backend().copy_h2d(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        basis.raw_mut(),
    );
    pool.backend()
        .copy_h2d(&[2.0_f64, -1.0], coefficients.raw_mut());
    pool.backend()
        .copy_h2d(&vec![0.0_f64; vector_len], y.raw_mut());

    let graph = pool
        .backend()
        .try_capture_graph_checked(|backend| {
            backend.try_basis_axpy_strided_device(
                basis.raw(),
                vector_len,
                basis_count,
                coefficients.raw(),
                y.raw_mut(),
            )
        })
        .expect("CUDA graph capture should succeed for native Krylov update");

    graph
        .try_launch()
        .expect("CUDA graph launch should succeed");
    graph
        .try_launch()
        .expect("CUDA graph relaunch should succeed");
    pool.backend()
        .try_synchronize()
        .expect("backend synchronization failed");

    let mut got = vec![0.0_f64; vector_len];
    pool.backend().copy_d2h(y.raw(), &mut got);
    assert_vec_close(
        "cuda/graph_device_resident_krylov_update",
        &got,
        &[-16.0, -32.0, -48.0, -64.0, -80.0],
    );
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_graph_replays_bem_fem_device_resident_sequence() {
    use sonium_compute::{BemRegularPairInF64, BemRegularPairOutF64, CudaBackend};

    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => skip_or_require_cuda!("CUDA BEM/FEM graph replay", e),
    };
    if !backend.has_bem_regular_pair_c64() {
        if env_flag("SONIUM_REQUIRE_CUDA") {
            panic!("CUDA Complex64 BEM regular-pair kernel is unavailable");
        }
        eprintln!("CUDA Complex64 BEM regular-pair kernel unavailable, skipping");
        return;
    }

    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
    let backend = pool.backend();
    let pair_host = BemRegularPairInF64::from_triangle(
        [0.0, 0.0, 4.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    );
    let mut pairs = pool.try_alloc::<BemRegularPairInF64>(1).unwrap();
    let mut output = pool.try_alloc::<BemRegularPairOutF64>(1).unwrap();
    let mut reference_output = pool.try_alloc::<BemRegularPairOutF64>(1).unwrap();
    backend.copy_h2d(&[pair_host], pairs.raw_mut());
    backend.copy_h2d(&[BemRegularPairOutF64::default()], output.raw_mut());
    backend.copy_h2d(
        &[BemRegularPairOutF64::default()],
        reference_output.raw_mut(),
    );

    // Establish the independent resident BEM reference before capture.  The
    // graph itself must only enqueue native work and must not allocate,
    // synchronize, or stage the pair through the host.
    backend
        .try_bem_regular_pair_c64_device(pairs.raw(), reference_output.raw_mut(), 1.7, -1.0)
        .unwrap();
    backend.try_synchronize().unwrap();
    let mut expected_pair = [BemRegularPairOutF64::default()];
    backend.copy_d2h(reference_output.raw(), &mut expected_pair);

    let vector_len = 5usize;
    let basis_count = 2usize;
    let mut basis = pool.try_alloc::<f64>(vector_len * basis_count).unwrap();
    let mut coefficients = pool.try_alloc::<f64>(basis_count).unwrap();
    let mut y = pool.try_alloc::<f64>(vector_len).unwrap();
    backend.copy_h2d(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        basis.raw_mut(),
    );
    backend.copy_h2d(&[2.0_f64, -1.0], coefficients.raw_mut());
    backend.copy_h2d(&[0.0_f64; 5], y.raw_mut());

    let graph = backend
        .try_capture_graph_checked(|backend| {
            backend.try_bem_regular_pair_c64_device(pairs.raw(), output.raw_mut(), 1.7, -1.0)?;
            // This device-resident basis update is the FEM/Krylov assembly
            // bridge: it consumes resident basis and coefficient buffers and
            // queues immediately after BEM on the same captured stream.
            backend.try_basis_axpy_strided_device(
                basis.raw(),
                vector_len,
                basis_count,
                coefficients.raw(),
                y.raw_mut(),
            )?;
            Ok(())
        })
        .expect("resident BEM/FEM graph capture should succeed");

    backend.reset_runtime_stats();
    graph.try_launch().expect("first BEM/FEM graph replay");
    graph.try_launch().expect("second BEM/FEM graph replay");
    let queued = backend.runtime_stats();
    assert_eq!(queued.kernel_launches, 2);
    assert_eq!(queued.transfer.h2d_operations, 0);
    assert_eq!(queued.transfer.d2h_operations, 0);
    backend
        .try_synchronize()
        .expect("BEM/FEM graph replay synchronization");

    let mut got_pair = [BemRegularPairOutF64::default()];
    let mut got_y = vec![0.0_f64; vector_len];
    backend.copy_d2h(output.raw(), &mut got_pair);
    backend.copy_d2h(y.raw(), &mut got_y);
    for (name, actual, expected) in [
        ("G", got_pair[0].g, expected_pair[0].g),
        ("H", got_pair[0].h, expected_pair[0].h),
        ("H^T", got_pair[0].ht, expected_pair[0].ht),
        ("E", got_pair[0].e, expected_pair[0].e),
    ] {
        assert!(
            (actual - expected).norm() <= 2.0e-11 * expected.norm().max(1.0),
            "replayed BEM {name} differs: actual={actual:?}, expected={expected:?}"
        );
    }
    assert_vec_close(
        "cuda/graph_bem_fem_device_resident_sequence",
        &got_y,
        &[-16.0, -32.0, -48.0, -64.0, -80.0],
    );
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_scalar_gmres_workspace_has_constant_steady_state_counters() {
    use sonium_compute::{
        DeviceCsr, GmresBackendConfig, GmresWorkspace, IdentityBackendPreconditioner,
        PersistentDeviceCsr, gmres_backend_preconditioned_pooled_with_workspace,
    };

    let backend = match sonium_compute::CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => skip_or_require_cuda!("CUDA scalar GMRES residency", e),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
    let matrix = PersistentDeviceCsr::try_from_device_csr(
        pool.backend(),
        DeviceCsr::try_from_host(
            pool.backend(),
            2,
            2,
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[4.0_f64, 1.0, 1.0, 3.0],
        )
        .expect("fixture CSR upload"),
    )
    .unwrap();
    let mut rhs = pool.try_alloc::<f64>(2).unwrap();
    let mut output = pool.try_alloc::<f64>(2).unwrap();
    pool.backend().copy_h2d(&[1.0, 2.0], rhs.raw_mut());
    let mut workspace = GmresWorkspace::new(2);
    let config = GmresBackendConfig {
        max_iterations: 8,
        tolerance: 1.0e-12,
    };

    pool.backend().reset_runtime_stats();
    let first_start = pool.backend().begin_runtime_stats();
    let first = gmres_backend_preconditioned_pooled_with_workspace(
        &pool,
        &mut workspace,
        &matrix,
        &IdentityBackendPreconditioner,
        rhs.raw(),
        None,
        output.raw_mut(),
        &config,
    )
    .expect("first CUDA scalar GMRES solve");
    assert!(first.converged, "first CUDA scalar GMRES solve: {first:?}");
    let first_stats = pool.backend().runtime_stats_since(first_start);
    let first_buffers = workspace.cached_buffer_count().unwrap();
    let first_bytes = workspace.cached_bytes().unwrap();

    let second_start = pool.backend().begin_runtime_stats();
    let second = gmres_backend_preconditioned_pooled_with_workspace(
        &pool,
        &mut workspace,
        &matrix,
        &IdentityBackendPreconditioner,
        rhs.raw(),
        None,
        output.raw_mut(),
        &config,
    )
    .expect("second CUDA scalar GMRES solve");
    assert!(
        second.converged,
        "second CUDA scalar GMRES solve: {second:?}"
    );
    let second_stats = pool.backend().runtime_stats_since(second_start);

    assert_eq!(workspace.cached_buffer_count().unwrap(), first_buffers);
    assert_eq!(workspace.cached_bytes().unwrap(), first_bytes);
    assert_eq!(
        second_stats.allocation_operations, 0,
        "first={first_stats:?}, second={second_stats:?}"
    );
    assert_eq!(second_stats.transfer, first_stats.transfer);
    assert_eq!(
        second_stats.synchronization_calls,
        first_stats.synchronization_calls
    );
    assert_eq!(second_stats.kernel_launches, first_stats.kernel_launches);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_block_gmres_workspace_has_constant_steady_state_counters() {
    use sonium_compute::{
        BlockGmresConfig, BlockGmresWorkspace, DeviceCsr, IdentityDevicePreconditioner,
        PersistentDeviceCsr, gmres_backend_block_preconditioned_with_workspace,
    };

    let backend = match sonium_compute::CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => skip_or_require_cuda!("CUDA block GMRES residency", e),
    };
    let matrix = PersistentDeviceCsr::try_from_device_csr(
        &backend,
        DeviceCsr::try_from_host(
            &backend,
            2,
            2,
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[4.0_f64, 1.0, 1.0, 3.0],
        )
        .expect("fixture CSR upload"),
    )
    .unwrap();
    let mut rhs = backend.raw_alloc::<f64>(4).unwrap();
    let mut output = backend.raw_alloc::<f64>(4).unwrap();
    backend.copy_h2d(&[9.0, 8.0, 3.0, 7.0], &mut rhs);
    let mut workspace = BlockGmresWorkspace::new(2);
    let config = BlockGmresConfig {
        max_block_iterations: 4,
        tolerance: 1.0e-10,
    };

    backend.reset_runtime_stats();
    let first_start = backend.begin_runtime_stats();
    let first = gmres_backend_block_preconditioned_with_workspace(
        &backend,
        &matrix,
        &IdentityDevicePreconditioner,
        &rhs,
        None,
        &mut output,
        2,
        &config,
        &mut workspace,
    )
    .expect("first CUDA block GMRES solve");
    assert!(first.converged, "first CUDA block GMRES solve: {first:?}");
    let first_stats = backend.runtime_stats_since(first_start);
    let first_buffers = workspace.device_buffer_count();
    let first_capacity = workspace.device_capacity();

    let second_start = backend.begin_runtime_stats();
    let second = gmres_backend_block_preconditioned_with_workspace(
        &backend,
        &matrix,
        &IdentityDevicePreconditioner,
        &rhs,
        None,
        &mut output,
        2,
        &config,
        &mut workspace,
    )
    .expect("second CUDA block GMRES solve");
    assert!(
        second.converged,
        "second CUDA block GMRES solve: {second:?}"
    );
    let second_stats = backend.runtime_stats_since(second_start);

    assert_eq!(workspace.device_buffer_count(), first_buffers);
    assert_eq!(workspace.device_capacity(), first_capacity);
    assert_eq!(
        second_stats.allocation_operations, 0,
        "first={first_stats:?}, second={second_stats:?}"
    );
    assert_eq!(second_stats.transfer, first_stats.transfer);
    assert_eq!(
        second_stats.synchronization_calls,
        first_stats.synchronization_calls
    );
    assert_eq!(second_stats.kernel_launches, first_stats.kernel_launches);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_graph_submission_holds_and_refunds_scheduler_lease() {
    use sonium_compute::{BackendError, CudaBackend, DeviceBudget, DeviceScheduler, WorkRequest};

    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => skip_or_require_cuda!("CUDA scheduled graph replay", e),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
    let mut x_dev = pool.try_alloc::<f64>(5).unwrap();
    let mut y_dev = pool.try_alloc::<f32>(5).unwrap();
    pool.backend()
        .copy_h2d(&[1.0, 2.0, 4.0, 8.0, 16.0], x_dev.raw_mut());
    pool.backend().copy_h2d(&[0.0_f32; 5], y_dev.raw_mut());
    let graph = pool
        .backend()
        .try_capture_graph_checked(|backend| {
            backend.try_cast_to_lower(x_dev.raw(), y_dev.raw_mut())
        })
        .expect("capture-safe native graph");

    let scheduler = DeviceScheduler::new([
        DeviceBudget::new("cuda0", 64 << 20, 1),
        DeviceBudget::new("cuda1", 64 << 20, 1),
    ])
    .expect("valid CUDA scheduler budgets");
    let submission = scheduler
        .submit_backend(
            WorkRequest::new(4096, 1),
            pool.backend(),
            |_device, _backend| {
                graph.try_launch()?;
                Ok::<_, BackendError>(())
            },
        )
        .expect("scheduled graph submission");
    let pending = scheduler.snapshots().expect("pending scheduler state");
    assert_eq!(
        pending
            .iter()
            .map(|snapshot| snapshot.active)
            .sum::<usize>(),
        1
    );
    submission.complete().expect("scheduled graph completion");

    let cancelled = scheduler
        .submit_backend(
            WorkRequest::new(4096, 1),
            pool.backend(),
            |_device, _backend| {
                graph.try_launch()?;
                Ok::<_, BackendError>(())
            },
        )
        .expect("second scheduled graph submission");
    cancelled.cancel().expect("scheduled graph cancellation");
    let finished = scheduler.snapshots().expect("finished scheduler state");
    assert_eq!(
        finished
            .iter()
            .map(|snapshot| snapshot.bytes_in_use)
            .sum::<usize>(),
        0
    );
    assert_eq!(
        finished
            .iter()
            .map(|snapshot| snapshot.active)
            .sum::<usize>(),
        0
    );
    assert_eq!(
        finished
            .iter()
            .map(|snapshot| snapshot.completed)
            .sum::<usize>(),
        1
    );
    assert_eq!(
        finished
            .iter()
            .map(|snapshot| snapshot.cancelled)
            .sum::<usize>(),
        1
    );
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_graph_submission_spawn_replays_and_refunds_scheduler_lease() {
    use sonium_compute::{BackendError, CudaBackend, DeviceBudget, DeviceScheduler, WorkRequest};

    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => skip_or_require_cuda!("CUDA spawned graph replay", e),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
    let mut x_dev = pool.try_alloc::<f64>(5).unwrap();
    let mut y_dev = pool.try_alloc::<f32>(5).unwrap();
    pool.backend()
        .copy_h2d(&[1.0, 2.0, 4.0, 8.0, 16.0], x_dev.raw_mut());
    pool.backend().copy_h2d(&[0.0_f32; 5], y_dev.raw_mut());
    pool.backend()
        .try_synchronize()
        .expect("graph spawn fixture upload");
    let graph = pool
        .backend()
        .try_capture_graph_checked(|backend| {
            backend.try_cast_to_lower(x_dev.raw(), y_dev.raw_mut())
        })
        .expect("capture-safe native graph");
    let baseline = pool.backend().runtime_stats();

    let scheduler = DeviceScheduler::new([DeviceBudget::new("cuda0", 64 << 20, 1)])
        .expect("valid single-device scheduler");
    let first = scheduler
        .submit_backend(
            WorkRequest::new(4096, 1),
            pool.backend(),
            |_device, _backend| {
                graph.try_launch()?;
                Ok::<_, BackendError>(())
            },
        )
        .expect("spawned graph submission");
    let first_result = first.spawn().join().expect("spawned graph thread");
    first_result.expect("spawned graph completion");

    let second = scheduler
        .submit_backend(
            WorkRequest::new(4096, 1),
            pool.backend(),
            |_device, _backend| {
                graph.try_launch()?;
                Ok::<_, BackendError>(())
            },
        )
        .expect("replayed spawned graph submission");
    second
        .spawn()
        .join()
        .expect("replayed graph thread")
        .expect("replayed graph completion");
    let replay_stats = pool.backend().runtime_stats_since(baseline);
    assert_eq!(replay_stats.kernel_launches, 2);
    assert_eq!(replay_stats.completion_events, 2);
    assert_eq!(replay_stats.completion_waits, 2);
    assert_eq!(replay_stats.transfer.h2d_operations, 0);
    assert_eq!(replay_stats.transfer.d2h_operations, 0);
    assert_eq!(
        pool.backend().runtime_stats().live_allocation_bytes,
        baseline.live_allocation_bytes
    );
    assert_eq!(
        pool.backend().runtime_stats().driver_resources,
        baseline.driver_resources
    );

    let mut got = [0.0_f32; 5];
    pool.backend().copy_d2h(y_dev.raw(), &mut got);
    assert_vec_close_f32("cuda/graph_spawn_replay", &got, &[1.0, 2.0, 4.0, 8.0, 16.0]);
    let finished = scheduler.snapshots().expect("finished scheduler state")[0].clone();
    assert_eq!(finished.bytes_in_use, 0);
    assert_eq!(finished.active, 0);
    assert_eq!(finished.completed, 2);
    assert_eq!(finished.cancelled, 0);
    assert_eq!(finished.failed, 0);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_graph_submission_device_loss_refunds_and_blocks_readmission() {
    use sonium_compute::{
        BackendError, CudaBackend, CudaFailureInjectionPoint, DeviceBudget, DeviceScheduler,
        WorkRequest,
    };

    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => skip_or_require_cuda!("CUDA scheduled graph device loss", e),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
    let mut x_dev = pool.try_alloc::<f64>(5).unwrap();
    let mut y_dev = pool.try_alloc::<f32>(5).unwrap();
    pool.backend()
        .copy_h2d(&[1.0, 2.0, 4.0, 8.0, 16.0], x_dev.raw_mut());
    pool.backend().copy_h2d(&[0.0_f32; 5], y_dev.raw_mut());
    let graph = pool
        .backend()
        .try_capture_graph_checked(|backend| {
            backend.try_cast_to_lower(x_dev.raw(), y_dev.raw_mut())
        })
        .expect("capture-safe native graph");

    let scheduler = DeviceScheduler::new([DeviceBudget::new("cuda0", 64 << 20, 1)])
        .expect("valid single-device scheduler budget");
    let submission = scheduler
        .submit_backend(
            WorkRequest::new(4096, 1),
            pool.backend(),
            |_device, _backend| {
                graph.try_launch()?;
                Ok::<_, BackendError>(())
            },
        )
        .expect("scheduled graph submission");

    // The shim fault is consumed by the asynchronous CUDA event token, not
    // by the enqueue path. The scheduler must still refund the reservation and
    // mark the device unavailable until explicit restoration.
    pool.backend()
        .inject_failure(CudaFailureInjectionPoint::DeviceLoss);
    assert!(matches!(
        submission.complete(),
        Err(sonium_compute::ScheduleError::DeviceLost { .. })
    ));
    let lost = scheduler.snapshots().expect("device-loss scheduler state")[0].clone();
    assert_eq!(lost.bytes_in_use, 0);
    assert_eq!(lost.active, 0);
    assert_eq!(lost.device_lost, 1);
    assert!(lost.lost);
    assert!(matches!(
        scheduler.acquire_on(0, WorkRequest::new(1, 1)),
        Err(sonium_compute::ScheduleError::DeviceUnavailable { .. })
    ));

    scheduler
        .restore_device(0)
        .expect("restore idle lost device");
    scheduler
        .acquire_on(0, WorkRequest::new(64 << 20, 1))
        .expect("restored device accepts full-budget work")
        .release()
        .expect("restored lease release");
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_graph_submission_launch_failure_refunds_and_reuses_lease() {
    use sonium_compute::{
        BackendError, CudaBackend, CudaFailureInjectionPoint, DeviceBudget, DeviceScheduler,
        ScheduleError, WorkRequest,
    };

    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => skip_or_require_cuda!("CUDA scheduled graph launch failure", e),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
    let mut x_dev = pool.try_alloc::<f64>(5).unwrap();
    let mut y_dev = pool.try_alloc::<f32>(5).unwrap();
    pool.backend()
        .copy_h2d(&[1.0, 2.0, 4.0, 8.0, 16.0], x_dev.raw_mut());
    pool.backend().copy_h2d(&[0.0_f32; 5], y_dev.raw_mut());
    let graph = pool
        .backend()
        .try_capture_graph_checked(|backend| {
            backend.try_cast_to_lower(x_dev.raw(), y_dev.raw_mut())
        })
        .expect("capture-safe native graph");

    let scheduler = DeviceScheduler::new([DeviceBudget::new("cuda0", 64 << 20, 1)])
        .expect("valid single-device scheduler budget");
    pool.backend()
        .inject_failure(CudaFailureInjectionPoint::Launch);
    let failure = match scheduler.submit_backend(
        WorkRequest::new(4096, 1),
        pool.backend(),
        |_device, _backend| {
            graph.try_launch()?;
            Ok::<_, BackendError>(())
        },
    ) {
        Ok(_) => panic!("injected graph launch failure must refund admission"),
        Err(error) => error,
    };
    assert!(matches!(failure, ScheduleError::BackendFailure { .. }));
    let failed = scheduler.snapshots().expect("failed scheduler state")[0].clone();
    assert_eq!(failed.bytes_in_use, 0);
    assert_eq!(failed.active, 0);
    assert_eq!(failed.failed, 1);

    // Launch injection is recoverable after the backend's checked error
    // boundary synchronizes/clears its first-error slot. The same lease and
    // graph can then be admitted and completed normally.
    pool.backend()
        .recover_after_error()
        .expect("recoverable graph launch failure");
    let submission = scheduler
        .submit_backend(
            WorkRequest::new(4096, 1),
            pool.backend(),
            |_device, _backend| {
                graph.try_launch()?;
                Ok::<_, BackendError>(())
            },
        )
        .expect("graph should be reusable after launch recovery");
    submission.complete().expect("recovered graph completion");
    let finished = scheduler.snapshots().expect("reused scheduler state")[0].clone();
    assert_eq!(finished.bytes_in_use, 0);
    assert_eq!(finished.active, 0);
    assert_eq!(finished.failed, 1);
    assert_eq!(finished.completed, 1);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_matches_cpu() {
    use sonium_compute::CudaBackend;
    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => skip_or_require_cuda!("parity test", e),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
    let inp = make_inputs();
    let cpu = cpu_reference(&inp);
    let gpu = run_on_backend(&pool, &inp);
    assert_parity("cuda", &cpu, &gpu);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_multi_dot_f64_matches_cpu() {
    use sonium_compute::CudaBackend;

    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => skip_or_require_cuda!("multi_dot parity", e),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
    let n = 257usize;
    let k = 5usize;
    let y_host: Vec<f64> = (0..n).map(|i| (i as f64 * 0.07).sin()).collect();
    let basis_host: Vec<Vec<f64>> = (0..k)
        .map(|j| {
            (0..n)
                .map(|i| ((i + 1) as f64).ln() + j as f64 * 0.125)
                .collect()
        })
        .collect();

    let mut y = pool.try_alloc::<f64>(n).unwrap();
    pool.backend().copy_h2d(&y_host, y.raw_mut());
    let mut basis = Vec::with_capacity(k);
    for host in &basis_host {
        let mut buf = pool.try_alloc::<f64>(n).unwrap();
        pool.backend().copy_h2d(host, buf.raw_mut());
        basis.push(buf);
    }
    let basis_refs: Vec<_> = basis.iter().map(|b| b.raw()).collect();
    let got = pool.backend().multi_dot(&basis_refs, y.raw());

    let want: Vec<f64> = basis_host
        .iter()
        .map(|x| x.iter().zip(y_host.iter()).map(|(a, b)| a * b).sum())
        .collect();
    assert_vec_close("cuda/multi_dot_f64", &got, &want);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_multi_dot_complex64_matches_cpu_dotc() {
    use num_complex::Complex64;
    use sonium_compute::CudaBackend;

    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => skip_or_require_cuda!("complex multi_dot parity", e),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
    let n = 193usize;
    let k = 4usize;
    let y_host: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new((i as f64 * 0.11).sin(), (i as f64 * 0.05).cos()))
        .collect();
    let basis_host: Vec<Vec<Complex64>> = (0..k)
        .map(|j| {
            (0..n)
                .map(|i| {
                    Complex64::new(
                        ((i + 1) as f64).ln() + j as f64 * 0.2,
                        (i as f64 * 0.03 + j as f64).sin(),
                    )
                })
                .collect()
        })
        .collect();

    let mut y = pool.try_alloc::<Complex64>(n).unwrap();
    pool.backend().copy_h2d(&y_host, y.raw_mut());
    let mut basis = Vec::with_capacity(k);
    for host in &basis_host {
        let mut buf = pool.try_alloc::<Complex64>(n).unwrap();
        pool.backend().copy_h2d(host, buf.raw_mut());
        basis.push(buf);
    }
    let basis_refs: Vec<_> = basis.iter().map(|b| b.raw()).collect();
    let got = pool.backend().multi_dot(&basis_refs, y.raw());

    let want: Vec<Complex64> = basis_host
        .iter()
        .map(|x| x.iter().zip(y_host.iter()).map(|(a, b)| a.conj() * b).sum())
        .collect();

    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_close(&format!("cuda/multi_dot_c64[{i}].re"), g.re, w.re);
        assert_close(&format!("cuda/multi_dot_c64[{i}].im"), g.im, w.im);
    }
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_multi_dot_strided_matches_cpu() {
    use sonium_compute::CudaBackend;

    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => skip_or_require_cuda!("strided multi_dot parity", e),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
    let n = 257usize;
    let k = 5usize;
    let y_host: Vec<f64> = (0..n).map(|i| (i as f64 * 0.09).cos()).collect();
    let basis_host: Vec<f64> = (0..k)
        .flat_map(|j| (0..n).map(move |i| ((i + 1) as f64).sqrt() + j as f64 * 0.125))
        .collect();

    let mut y = pool.try_alloc::<f64>(n).unwrap();
    pool.backend().copy_h2d(&y_host, y.raw_mut());
    let mut basis = pool.try_alloc::<f64>(n * k).unwrap();
    pool.backend().copy_h2d(&basis_host, basis.raw_mut());
    let got = pool.backend().multi_dot_strided(basis.raw(), n, k, y.raw());

    let want: Vec<f64> = (0..k)
        .map(|col| {
            basis_host[col * n..(col + 1) * n]
                .iter()
                .zip(y_host.iter())
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect();
    assert_vec_close("cuda/multi_dot_strided_f64", &got, &want);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_multi_dot_strided_complex64_matches_cpu_dotc() {
    use sonium_compute::CudaBackend;

    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => skip_or_require_cuda!("complex strided multi_dot parity", e),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
    let n = 193usize;
    let k = 4usize;
    let y_host: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new((i as f64 * 0.11).sin(), (i as f64 * 0.05).cos()))
        .collect();
    let basis_host: Vec<Complex64> = (0..k)
        .flat_map(|j| {
            (0..n).map(move |i| {
                Complex64::new(
                    ((i + 1) as f64).ln() + j as f64 * 0.2,
                    (i as f64 * 0.03 + j as f64).sin(),
                )
            })
        })
        .collect();

    let mut y = pool.try_alloc::<Complex64>(n).unwrap();
    pool.backend().copy_h2d(&y_host, y.raw_mut());
    let mut basis = pool.try_alloc::<Complex64>(n * k).unwrap();
    pool.backend().copy_h2d(&basis_host, basis.raw_mut());
    let got = pool.backend().multi_dot_strided(basis.raw(), n, k, y.raw());

    let want: Vec<Complex64> = (0..k)
        .map(|col| {
            basis_host[col * n..(col + 1) * n]
                .iter()
                .zip(y_host.iter())
                .map(|(a, b)| a.conj() * b)
                .sum()
        })
        .collect();
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_close(&format!("cuda/multi_dot_strided_c64[{i}].re"), g.re, w.re);
        assert_close(&format!("cuda/multi_dot_strided_c64[{i}].im"), g.im, w.im);
    }
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_basis_axpy_strided_f64_matches_cpu() {
    use sonium_compute::CudaBackend;

    let backend = match CudaBackend::new_default() {
        Ok(backend) => backend,
        Err(error) => skip_or_require_cuda!("basis AXPY strided f64 parity", error),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
    let vector_len = 257usize;
    let basis_count = 4usize;
    let basis_host: Vec<f64> = (0..basis_count)
        .flat_map(|basis_index| {
            (0..vector_len).map(move |i| ((i + 1) as f64 * 0.031).sin() + basis_index as f64 * 0.17)
        })
        .collect();
    let coeffs = [0.7_f64, -0.25, 1.2, -0.4];
    let alpha = -0.6_f64;
    let mut y_host: Vec<f64> = (0..vector_len).map(|i| (i as f64 * 0.13).cos()).collect();
    let mut basis = pool.try_alloc::<f64>(basis_host.len()).unwrap();
    let mut y = pool.try_alloc::<f64>(vector_len).unwrap();
    pool.backend().copy_h2d(&basis_host, basis.raw_mut());
    pool.backend().copy_h2d(&y_host, y.raw_mut());
    pool.backend().basis_axpy_strided(
        alpha,
        basis.raw(),
        vector_len,
        basis_count,
        &coeffs,
        y.raw_mut(),
    );
    for (i, value) in y_host.iter_mut().enumerate() {
        for basis_index in 0..basis_count {
            *value += alpha * coeffs[basis_index] * basis_host[basis_index * vector_len + i];
        }
    }
    let mut got = vec![0.0_f64; vector_len];
    pool.backend().copy_d2h(y.raw(), &mut got);
    assert_vec_close("cuda/basis_axpy_strided_f64", &got, &y_host);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_axpy_many_f64_matches_cpu() {
    use sonium_compute::CudaBackend;

    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => skip_or_require_cuda!("axpy_many f64 parity", e),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
    let n = 257usize;
    let basis_host: Vec<Vec<f64>> = (0..4)
        .map(|j| {
            (0..n)
                .map(|i| ((i + 1) as f64).ln() + j as f64 * 0.25)
                .collect()
        })
        .collect();
    let alphas = [0.7_f64, -0.25, 1.2, -0.4];
    let mut y_host: Vec<f64> = (0..n).map(|i| (i as f64 * 0.13).sin()).collect();

    let mut y = pool.try_alloc::<f64>(n).unwrap();
    pool.backend().copy_h2d(&y_host, y.raw_mut());
    let mut basis = Vec::with_capacity(basis_host.len());
    for host in &basis_host {
        let mut buf = pool.try_alloc::<f64>(n).unwrap();
        pool.backend().copy_h2d(host, buf.raw_mut());
        basis.push(buf);
    }
    let basis_refs: Vec<_> = basis.iter().map(|b| b.raw()).collect();
    pool.backend().axpy_many(&alphas, &basis_refs, y.raw_mut());
    for (basis_idx, x) in basis_host.iter().enumerate() {
        for (yi, xi) in y_host.iter_mut().zip(x.iter()) {
            *yi += alphas[basis_idx] * *xi;
        }
    }

    let mut got = vec![0.0_f64; n];
    pool.backend().copy_d2h(y.raw(), &mut got);
    assert_vec_close("cuda/axpy_many_f64", &got, &y_host);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_basis_axpy_strided_c64_matches_cpu() {
    use sonium_compute::CudaBackend;

    let backend = match CudaBackend::new_default() {
        Ok(backend) => backend,
        Err(error) => skip_or_require_cuda!("basis AXPY strided c64 parity", error),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
    let vector_len = 193usize;
    let basis_count = 3usize;
    let basis_host: Vec<Complex64> = (0..basis_count)
        .flat_map(|basis_index| {
            (0..vector_len).map(move |i| {
                Complex64::new(
                    (i as f64 * 0.041).sin() + basis_index as f64 * 0.07,
                    (i as f64 * 0.023).cos() - basis_index as f64 * 0.03,
                )
            })
        })
        .collect();
    let coeffs = [
        Complex64::new(0.7, -0.2),
        Complex64::new(-0.4, 0.3),
        Complex64::new(1.1, 0.15),
    ];
    let alpha = Complex64::new(-0.6, 0.15);
    let mut y_host: Vec<Complex64> = (0..vector_len)
        .map(|i| Complex64::new((i as f64 * 0.071).cos(), (i as f64 * 0.053).sin()))
        .collect();
    let mut basis = pool.try_alloc::<Complex64>(basis_host.len()).unwrap();
    let mut y = pool.try_alloc::<Complex64>(vector_len).unwrap();
    pool.backend().copy_h2d(&basis_host, basis.raw_mut());
    pool.backend().copy_h2d(&y_host, y.raw_mut());
    pool.backend().basis_axpy_strided(
        alpha,
        basis.raw(),
        vector_len,
        basis_count,
        &coeffs,
        y.raw_mut(),
    );
    for (i, value) in y_host.iter_mut().enumerate() {
        for basis_index in 0..basis_count {
            *value += alpha * coeffs[basis_index] * basis_host[basis_index * vector_len + i];
        }
    }
    let mut got = vec![Complex64::new(0.0, 0.0); vector_len];
    pool.backend().copy_d2h(y.raw(), &mut got);
    for (i, (actual, expected)) in got.iter().zip(y_host.iter()).enumerate() {
        assert_close(
            &format!("cuda/basis_axpy_strided_c64[{i}].re"),
            actual.re,
            expected.re,
        );
        assert_close(
            &format!("cuda/basis_axpy_strided_c64[{i}].im"),
            actual.im,
            expected.im,
        );
    }
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_axpy_many_complex32_matches_cpu() {
    use sonium_compute::CudaBackend;

    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => skip_or_require_cuda!("axpy_many c32 parity", e),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
    let n = 193usize;
    let basis_host: Vec<Vec<Complex32>> = (0..3)
        .map(|j| {
            (0..n)
                .map(|i| {
                    Complex32::new(
                        (i as f32 * 0.09 + j as f32).sin(),
                        (i as f32 * 0.05 + j as f32 * 0.2).cos(),
                    )
                })
                .collect()
        })
        .collect();
    let alphas = [
        Complex32::new(0.7, -0.2),
        Complex32::new(-0.25, 0.4),
        Complex32::new(1.1, 0.15),
    ];
    let mut y_host: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new((i as f32 * 0.17).sin(), (i as f32 * 0.11).cos()))
        .collect();

    let mut y = pool.try_alloc::<Complex32>(n).unwrap();
    pool.backend().copy_h2d(&y_host, y.raw_mut());
    let mut basis = Vec::with_capacity(basis_host.len());
    for host in &basis_host {
        let mut buf = pool.try_alloc::<Complex32>(n).unwrap();
        pool.backend().copy_h2d(host, buf.raw_mut());
        basis.push(buf);
    }
    let basis_refs: Vec<_> = basis.iter().map(|b| b.raw()).collect();
    pool.backend().axpy_many(&alphas, &basis_refs, y.raw_mut());
    for (basis_idx, x) in basis_host.iter().enumerate() {
        for (yi, xi) in y_host.iter_mut().zip(x.iter()) {
            *yi += alphas[basis_idx] * *xi;
        }
    }

    let mut got = vec![Complex32::new(0.0, 0.0); n];
    pool.backend().copy_d2h(y.raw(), &mut got);
    for (i, (g, w)) in got.iter().zip(y_host.iter()).enumerate() {
        close_c32(&format!("cuda/axpy_many_c32[{i}]"), *g, *w);
    }
}

fn close_f32(label: &str, a: f32, b: f32) {
    let tol = b.abs().max(1.0) * 1e-5;
    let d = (a - b).abs();
    assert!(d <= tol, "{label}: gpu={a}, cpu={b}, tol={tol}");
}

fn close_f64(label: &str, a: f64, b: f64) {
    let tol = b.abs().max(1.0) * 1e-5;
    let d = (a - b).abs();
    assert!(d <= tol, "{label}: gpu={a}, cpu={b}, tol={tol}");
}

fn close_c32(label: &str, a: Complex32, b: Complex32) {
    close_f32(&format!("{label}.re"), a.re, b.re);
    close_f32(&format!("{label}.im"), a.im, b.im);
}

fn close_c64(label: &str, a: Complex64, b: Complex64) {
    close_f64(&format!("{label}.re"), a.re, b.re);
    close_f64(&format!("{label}.im"), a.im, b.im);
}

fn assert_vec_close_f32(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch ({} vs {})",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let scale = e.abs().max(1.0);
        let diff = (a - e).abs();
        assert!(
            diff <= REL_TOL as f32 * scale,
            "{label}[{i}]: gpu={a}, cpu={e}, |diff|={diff}, tol={}",
            REL_TOL as f32 * scale
        );
    }
}

/// CPU ↔ GPU parity for `Complex32` / `Complex64`.
#[cfg(feature = "cuda")]
mod complex {
    use super::*;
    use sonium_compute::CudaBackend;

    /// 4×4 tridiagonal Laplacian — real values stored in the complex
    /// type so cuSPARSE goes through the C/Z SpMV path even with no
    /// imaginary parts.
    const ROW_PTR: [i32; 5] = [0, 2, 5, 8, 10];
    const COL_IDX: [i32; 10] = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    const SPMAT_VALS: [f32; 10] = [2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0];

    fn direct_dft(input: &[Complex64], inverse: bool) -> Vec<Complex64> {
        let n = input.len();
        let sign = if inverse { 1.0 } else { -1.0 };
        (0..n)
            .map(|k| {
                input
                    .iter()
                    .enumerate()
                    .map(|(j, &value)| {
                        let angle = sign * std::f64::consts::TAU * (j * k) as f64 / n as f64;
                        value * Complex64::new(angle.cos(), angle.sin())
                    })
                    .sum()
            })
            .collect()
    }

    fn direct_batched_dft_c32(
        input: &[Complex32],
        fft_len: usize,
        inverse: bool,
    ) -> Vec<Complex32> {
        input
            .chunks_exact(fft_len)
            .flat_map(|chunk| {
                let as_c64: Vec<_> = chunk
                    .iter()
                    .map(|&value| Complex64::new(value.re as f64, value.im as f64))
                    .collect();
                direct_dft(&as_c64, inverse)
                    .into_iter()
                    .map(|value| Complex32::new(value.re as f32, value.im as f32))
            })
            .collect()
    }

    fn direct_batched_dft_c64(
        input: &[Complex64],
        fft_len: usize,
        inverse: bool,
    ) -> Vec<Complex64> {
        input
            .chunks_exact(fft_len)
            .flat_map(|chunk| direct_dft(chunk, inverse))
            .collect()
    }

    #[test]
    fn cuda_fft_complex32_matches_reference_and_preserves_unscaled_inverse() {
        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend,
            Err(error) => skip_or_require_cuda!("cuFFT Complex32 parity", error),
        };
        let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
        let fft_len = 8;
        let input: Vec<_> = (0..16)
            .map(|i| {
                let x = i as f32 * 0.37;
                Complex32::new(x.sin(), (x * 0.71).cos())
            })
            .collect();
        let expected_forward = direct_batched_dft_c32(&input, fft_len, false);
        let expected_inverse = direct_batched_dft_c32(&expected_forward, fft_len, true);

        let mut buffer = pool.try_alloc::<Complex32>(input.len()).unwrap();
        pool.backend().copy_h2d(&input, buffer.raw_mut());
        if let Err(error) = pool
            .backend()
            .try_fft_in_place(buffer.raw_mut(), fft_len, false)
        {
            skip_or_require_cuda!("cuFFT Complex32 parity", error);
        }
        let mut actual_forward = vec![Complex32::new(0.0, 0.0); input.len()];
        pool.backend().copy_d2h(buffer.raw(), &mut actual_forward);
        for (i, (&actual, &expected)) in actual_forward
            .iter()
            .zip(expected_forward.iter())
            .enumerate()
        {
            close_c32(&format!("cuda/fft_c32/forward[{i}]"), actual, expected);
        }

        if let Err(error) = pool
            .backend()
            .try_fft_in_place(buffer.raw_mut(), fft_len, true)
        {
            skip_or_require_cuda!("cuFFT Complex32 inverse parity", error);
        }
        let mut actual_inverse = vec![Complex32::new(0.0, 0.0); input.len()];
        pool.backend().copy_d2h(buffer.raw(), &mut actual_inverse);
        for (i, (&actual, &expected)) in actual_inverse
            .iter()
            .zip(expected_inverse.iter())
            .enumerate()
        {
            close_c32(&format!("cuda/fft_c32/inverse[{i}]"), actual, expected);
        }
    }

    #[test]
    fn cuda_fft_complex64_matches_reference_and_preserves_unscaled_inverse() {
        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend,
            Err(error) => skip_or_require_cuda!("cuFFT Complex64 parity", error),
        };
        let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
        let fft_len = 8;
        let input: Vec<_> = (0..16)
            .map(|i| {
                let x = i as f64 * 0.37;
                Complex64::new(x.sin(), (x * 0.71).cos())
            })
            .collect();
        let expected_forward = direct_batched_dft_c64(&input, fft_len, false);
        let expected_inverse = direct_batched_dft_c64(&expected_forward, fft_len, true);

        let mut buffer = pool.try_alloc::<Complex64>(input.len()).unwrap();
        pool.backend().copy_h2d(&input, buffer.raw_mut());
        if let Err(error) = pool
            .backend()
            .try_fft_in_place(buffer.raw_mut(), fft_len, false)
        {
            skip_or_require_cuda!("cuFFT Complex64 parity", error);
        }
        let mut actual_forward = vec![Complex64::new(0.0, 0.0); input.len()];
        pool.backend().copy_d2h(buffer.raw(), &mut actual_forward);
        for (i, (&actual, &expected)) in actual_forward
            .iter()
            .zip(expected_forward.iter())
            .enumerate()
        {
            close_c64(&format!("cuda/fft_c64/forward[{i}]"), actual, expected);
        }

        if let Err(error) = pool
            .backend()
            .try_fft_in_place(buffer.raw_mut(), fft_len, true)
        {
            skip_or_require_cuda!("cuFFT Complex64 inverse parity", error);
        }
        let mut actual_inverse = vec![Complex64::new(0.0, 0.0); input.len()];
        pool.backend().copy_d2h(buffer.raw(), &mut actual_inverse);
        for (i, (&actual, &expected)) in actual_inverse
            .iter()
            .zip(expected_inverse.iter())
            .enumerate()
        {
            close_c64(&format!("cuda/fft_c64/inverse[{i}]"), actual, expected);
        }
    }

    #[test]
    fn cuda_gather4_weighted_complex32_matches_cpu() {
        let gpu_backend = match CudaBackend::new_default() {
            Ok(b) => b,
            Err(e) => skip_or_require_cuda!("gather4 c32 parity", e),
        };
        let cpu_pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let gpu_pool = MemoryPool::new(gpu_backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();

        let values: Vec<Complex32> = vec![
            Complex32::new(1.0, 0.5),
            Complex32::new(2.0, -1.0),
            Complex32::new(-1.0, 3.0),
            Complex32::new(4.0, 2.0),
            Complex32::new(-2.0, -0.5),
        ];
        let indices = vec![0, 1, 2, 3, 4, 3, 1, 0, 2, 2, 2, 2];
        let weights = vec![
            0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.25, 0.25, 0.0, 1.0, 0.0, 0.0, 0.0,
        ];

        let cpu = run_gather_c32(&cpu_pool, &values, &indices, &weights);
        let gpu = run_gather_c32(&gpu_pool, &values, &indices, &weights);
        for i in 0..cpu.len() {
            close_c32(&format!("gather4/c32[{i}]"), gpu[i], cpu[i]);
        }
    }

    #[test]
    fn cuda_gather4_weighted_complex64_matches_cpu() {
        let gpu_backend = match CudaBackend::new_default() {
            Ok(b) => b,
            Err(e) => skip_or_require_cuda!("gather4 parity", e),
        };
        let cpu_pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let gpu_pool = MemoryPool::new(gpu_backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();

        let values: Vec<Complex64> = vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(2.0, -1.0),
            Complex64::new(-1.0, 3.0),
            Complex64::new(4.0, 2.0),
            Complex64::new(-2.0, -0.5),
        ];
        let indices = vec![0, 1, 2, 3, 4, 3, 1, 0, 2, 2, 2, 2];
        let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.25, 0.25, 0.0, 1.0, 0.0, 0.0, 0.0];

        let cpu = run_gather_c64(&cpu_pool, &values, &indices, &weights);
        let gpu = run_gather_c64(&gpu_pool, &values, &indices, &weights);
        for i in 0..cpu.len() {
            close_c64(&format!("gather4/c64[{i}]"), gpu[i], cpu[i]);
        }
    }

    fn run_gather_c32<B: Backend>(
        pool: &MemoryPool<B>,
        values: &[Complex32],
        indices: &[i32],
        weights: &[f32],
    ) -> Vec<Complex32> {
        let mut values_dev = pool.try_alloc::<Complex32>(values.len()).unwrap();
        let mut indices_dev = pool.try_alloc::<i32>(indices.len()).unwrap();
        let mut weights_dev = pool.try_alloc::<f32>(weights.len()).unwrap();
        let mut out_dev = pool.try_alloc::<Complex32>(indices.len() / 4).unwrap();
        pool.backend().copy_h2d(values, values_dev.raw_mut());
        pool.backend().copy_h2d(indices, indices_dev.raw_mut());
        pool.backend().copy_h2d(weights, weights_dev.raw_mut());
        pool.backend().gather4_weighted(
            values_dev.raw(),
            indices_dev.raw(),
            weights_dev.raw(),
            out_dev.raw_mut(),
        );
        let mut out = vec![Complex32::new(0.0, 0.0); indices.len() / 4];
        pool.backend().copy_d2h(out_dev.raw(), &mut out);
        out
    }

    fn run_gather_c64<B: Backend>(
        pool: &MemoryPool<B>,
        values: &[Complex64],
        indices: &[i32],
        weights: &[f64],
    ) -> Vec<Complex64> {
        let mut values_dev = pool.try_alloc::<Complex64>(values.len()).unwrap();
        let mut indices_dev = pool.try_alloc::<i32>(indices.len()).unwrap();
        let mut weights_dev = pool.try_alloc::<f64>(weights.len()).unwrap();
        let mut out_dev = pool.try_alloc::<Complex64>(indices.len() / 4).unwrap();
        pool.backend().copy_h2d(values, values_dev.raw_mut());
        pool.backend().copy_h2d(indices, indices_dev.raw_mut());
        pool.backend().copy_h2d(weights, weights_dev.raw_mut());
        pool.backend().gather4_weighted(
            values_dev.raw(),
            indices_dev.raw(),
            weights_dev.raw(),
            out_dev.raw_mut(),
        );
        let mut out = vec![Complex64::new(0.0, 0.0); indices.len() / 4];
        pool.backend().copy_d2h(out_dev.raw(), &mut out);
        out
    }

    #[test]
    fn cuda_device_precision_casts_roundtrip() {
        let gpu_backend = match CudaBackend::new_default() {
            Ok(b) => b,
            Err(e) => skip_or_require_cuda!("precision-cast parity", e),
        };
        let pool = MemoryPool::new(gpu_backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();

        let real: Vec<f64> = vec![-3.25, -1.5, -0.125, 0.0, 0.25, 1.75, 8.5];
        let mut real_upper = pool.try_alloc::<f64>(real.len()).unwrap();
        let mut real_lower = pool.try_alloc::<f32>(real.len()).unwrap();
        let mut real_roundtrip = pool.try_alloc::<f64>(real.len()).unwrap();
        pool.backend().copy_h2d(&real, real_upper.raw_mut());
        pool.backend()
            .cast_to_lower::<f64>(real_upper.raw(), real_lower.raw_mut());
        pool.backend()
            .cast_from_lower::<f64>(real_lower.raw(), real_roundtrip.raw_mut());

        let mut real_lower_host = vec![0.0_f32; real.len()];
        let mut real_roundtrip_host = vec![0.0_f64; real.len()];
        pool.backend()
            .copy_d2h(real_lower.raw(), &mut real_lower_host);
        pool.backend()
            .copy_d2h(real_roundtrip.raw(), &mut real_roundtrip_host);
        for i in 0..real.len() {
            close_f32(
                &format!("cast/f64_to_f32[{i}]"),
                real_lower_host[i],
                real[i] as f32,
            );
            close_f64(
                &format!("cast/f32_to_f64[{i}]"),
                real_roundtrip_host[i],
                real[i] as f32 as f64,
            );
        }

        let complex: Vec<Complex64> = (0..real.len())
            .map(|i| Complex64::new(real[i], real[real.len() - 1 - i] * -0.5))
            .collect();
        let mut complex_upper = pool.try_alloc::<Complex64>(complex.len()).unwrap();
        let mut complex_lower = pool.try_alloc::<Complex32>(complex.len()).unwrap();
        let mut complex_roundtrip = pool.try_alloc::<Complex64>(complex.len()).unwrap();
        pool.backend().copy_h2d(&complex, complex_upper.raw_mut());
        pool.backend()
            .cast_to_lower::<Complex64>(complex_upper.raw(), complex_lower.raw_mut());
        pool.backend()
            .cast_from_lower::<Complex64>(complex_lower.raw(), complex_roundtrip.raw_mut());

        let mut complex_lower_host = vec![Complex32::new(0.0, 0.0); complex.len()];
        let mut complex_roundtrip_host = vec![Complex64::new(0.0, 0.0); complex.len()];
        pool.backend()
            .copy_d2h(complex_lower.raw(), &mut complex_lower_host);
        pool.backend()
            .copy_d2h(complex_roundtrip.raw(), &mut complex_roundtrip_host);
        for i in 0..complex.len() {
            let lower_expected = Complex32::new(complex[i].re as f32, complex[i].im as f32);
            let roundtrip_expected =
                Complex64::new(lower_expected.re as f64, lower_expected.im as f64);
            close_c32(
                &format!("cast/c64_to_c32[{i}]"),
                complex_lower_host[i],
                lower_expected,
            );
            close_c64(
                &format!("cast/c32_to_c64[{i}]"),
                complex_roundtrip_host[i],
                roundtrip_expected,
            );
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_dense_lu_c32_solves_small_system() {
        let gpu_backend = match CudaBackend::new_default() {
            Ok(b) => b,
            Err(e) => skip_or_require_cuda!("dense LU c32 parity", e),
        };
        let pool = MemoryPool::new(gpu_backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();

        // Column-major storage for:
        // [ 4       2 - i   ]
        // [ 1       3 + 0.5i]
        let a_col_major = vec![
            Complex32::new(4.0, 0.0),
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, -1.0),
            Complex32::new(3.0, 0.5),
        ];
        let expected = [Complex32::new(1.0, -2.0), Complex32::new(-0.5, 0.25)];
        let rhs = vec![Complex32::new(3.25, -7.0), Complex32::new(-0.625, -1.5)];

        let lu = pool
            .backend()
            .dense_lu_c32_from_column_major(2, &a_col_major)
            .unwrap();
        let mut rhs_dev = pool.try_alloc::<Complex32>(rhs.len()).unwrap();
        let mut out_dev = pool.try_alloc::<Complex32>(rhs.len()).unwrap();
        pool.backend().copy_h2d(&rhs, rhs_dev.raw_mut());
        pool.backend()
            .dense_lu_c32_solve(&lu, rhs_dev.raw(), out_dev.raw_mut())
            .unwrap();

        let mut out = vec![Complex32::new(0.0, 0.0); rhs.len()];
        pool.backend().copy_d2h(out_dev.raw(), &mut out);
        for i in 0..expected.len() {
            close_c32(&format!("dense_lu_c32[{i}]"), out[i], expected[i]);
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_dense_lu_c64_solves_forward_and_hermitian_systems() {
        let gpu_backend = match CudaBackend::new_default() {
            Ok(b) => b,
            Err(e) => skip_or_require_cuda!("dense LU c64 parity", e),
        };
        let pool = MemoryPool::new(gpu_backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();

        let a_col_major = vec![
            Complex64::new(4.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 0.5),
        ];
        let expected = [Complex64::new(1.0, -2.0), Complex64::new(-0.5, 0.25)];
        let rhs = vec![Complex64::new(3.25, -7.0), Complex64::new(-0.625, -1.5)];
        let lu = pool
            .backend()
            .dense_lu_c64_from_column_major(2, &a_col_major)
            .unwrap();
        let mut rhs_dev = pool.try_alloc::<Complex64>(rhs.len()).unwrap();
        let mut out_dev = pool.try_alloc::<Complex64>(rhs.len()).unwrap();
        pool.backend().copy_h2d(&rhs, rhs_dev.raw_mut());
        pool.backend()
            .dense_lu_c64_solve(&lu, rhs_dev.raw(), out_dev.raw_mut(), Transpose::NoTrans)
            .unwrap();

        let mut out = vec![Complex64::new(0.0, 0.0); rhs.len()];
        pool.backend().copy_d2h(out_dev.raw(), &mut out);
        for i in 0..expected.len() {
            close_c64(&format!("dense_lu_c64[{i}]"), out[i], expected[i]);
        }

        // Solve the conjugate-transposed system against a separately formed
        // right-hand side, exercising cuSOLVER's CUBLAS_OP_C path.
        let hermitian_rhs = vec![Complex64::new(5.0, -1.0), Complex64::new(2.0, 0.5)];
        let mut hermitian_rhs_dev = pool.try_alloc::<Complex64>(2).unwrap();
        let mut hermitian_out_dev = pool.try_alloc::<Complex64>(2).unwrap();
        pool.backend()
            .copy_h2d(&hermitian_rhs, hermitian_rhs_dev.raw_mut());
        pool.backend()
            .dense_lu_c64_solve(
                &lu,
                hermitian_rhs_dev.raw(),
                hermitian_out_dev.raw_mut(),
                Transpose::ConjTrans,
            )
            .unwrap();
        let mut actual = vec![Complex64::new(0.0, 0.0); 2];
        pool.backend()
            .copy_d2h(hermitian_out_dev.raw(), &mut actual);
        let a_h = [
            [a_col_major[0].conj(), a_col_major[1].conj()],
            [a_col_major[2].conj(), a_col_major[3].conj()],
        ];
        let determinant = a_h[0][0] * a_h[1][1] - a_h[0][1] * a_h[1][0];
        let expected_hermitian = [
            (hermitian_rhs[0] * a_h[1][1] - a_h[0][1] * hermitian_rhs[1]) / determinant,
            (a_h[0][0] * hermitian_rhs[1] - hermitian_rhs[0] * a_h[1][0]) / determinant,
        ];
        for i in 0..2 {
            close_c64(
                &format!("dense_lu_c64_hermitian[{i}]"),
                actual[i],
                expected_hermitian[i],
            );
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_dense_lu_f64_solves_and_transposed_system() {
        let gpu_backend = match CudaBackend::new_default() {
            Ok(b) => b,
            Err(e) => skip_or_require_cuda!("dense LU f64 parity", e),
        };
        let pool = MemoryPool::new(gpu_backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();

        // Column-major storage for [[4, 2], [1, 3]] (det = 10).
        let a_col_major = vec![4.0, 1.0, 2.0, 3.0];
        // x = [1, -2] gives b = Ax = [0, -5].
        let expected = [1.0, -2.0];
        let rhs = vec![0.0, -5.0];
        let lu = pool
            .backend()
            .dense_lu_f64_from_column_major(2, &a_col_major)
            .unwrap();
        let mut rhs_dev = pool.try_alloc::<f64>(rhs.len()).unwrap();
        let mut out_dev = pool.try_alloc::<f64>(rhs.len()).unwrap();
        pool.backend().copy_h2d(&rhs, rhs_dev.raw_mut());
        pool.backend()
            .dense_lu_f64_solve(&lu, rhs_dev.raw(), out_dev.raw_mut(), Transpose::NoTrans)
            .unwrap();

        let mut out = vec![0.0; rhs.len()];
        pool.backend().copy_d2h(out_dev.raw(), &mut out);
        for i in 0..expected.len() {
            close_f64(&format!("dense_lu_f64[{i}]"), out[i], expected[i]);
        }

        // Transposed system A^T y = c with y = [2, 1], c = [9, 7].
        let trans_rhs = vec![9.0, 7.0];
        let mut trans_rhs_dev = pool.try_alloc::<f64>(2).unwrap();
        let mut trans_out_dev = pool.try_alloc::<f64>(2).unwrap();
        pool.backend().copy_h2d(&trans_rhs, trans_rhs_dev.raw_mut());
        pool.backend()
            .dense_lu_f64_solve(
                &lu,
                trans_rhs_dev.raw(),
                trans_out_dev.raw_mut(),
                Transpose::Trans,
            )
            .unwrap();
        let mut actual = vec![0.0; 2];
        pool.backend().copy_d2h(trans_out_dev.raw(), &mut actual);
        for (i, want) in [2.0, 1.0].iter().enumerate() {
            close_f64(&format!("dense_lu_f64_trans[{i}]"), actual[i], *want);
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_dense_cholesky_f64_solves_spd_system() {
        let gpu_backend = match CudaBackend::new_default() {
            Ok(b) => b,
            Err(e) => skip_or_require_cuda!("dense Cholesky f64 parity", e),
        };
        let pool = MemoryPool::new(gpu_backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();

        // Column-major storage for [[4, 2], [2, 3]] (SPD, det = 8).
        let a_col_major = vec![4.0, 2.0, 2.0, 3.0];
        // x = [1, -2] gives b = Ax = [0, -4].
        let expected = [1.0, -2.0];
        let rhs = vec![0.0, -4.0];
        let chol = pool
            .backend()
            .dense_cholesky_f64_from_column_major(2, &a_col_major)
            .unwrap();
        let mut rhs_dev = pool.try_alloc::<f64>(rhs.len()).unwrap();
        let mut out_dev = pool.try_alloc::<f64>(rhs.len()).unwrap();
        pool.backend().copy_h2d(&rhs, rhs_dev.raw_mut());
        pool.backend()
            .dense_cholesky_f64_solve(&chol, rhs_dev.raw(), out_dev.raw_mut())
            .unwrap();

        let mut out = vec![0.0; rhs.len()];
        pool.backend().copy_d2h(out_dev.raw(), &mut out);
        for i in 0..expected.len() {
            close_f64(&format!("dense_chol_f64[{i}]"), out[i], expected[i]);
        }
    }

    #[test]
    fn cuda_matches_cpu_complex32() {
        let gpu_backend = match CudaBackend::new_default() {
            Ok(b) => b,
            Err(e) => skip_or_require_cuda!("complex32 parity", e),
        };
        let cpu_pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let gpu_pool = MemoryPool::new(gpu_backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();

        // Inputs with non-trivial imaginary parts so `dotc` and the C
        // BLAS-1 routines exercise the conjugating path.
        let n = 8usize;
        let x: Vec<Complex32> = (0..n)
            .map(|i| Complex32::new(0.5 * i as f32 - 1.5, 0.1 * i as f32))
            .collect();
        let y: Vec<Complex32> = (0..n)
            .map(|i| Complex32::new(0.25 * i as f32 + 0.75, -0.2 * i as f32))
            .collect();
        let alpha = Complex32::new(1.7, 0.3);
        let values: Vec<Complex32> = SPMAT_VALS.iter().map(|&v| Complex32::new(v, 0.0)).collect();
        let spx: Vec<Complex32> = (0..4)
            .map(|i| Complex32::new(i as f32 + 1.0, 0.05 * i as f32))
            .collect();

        let cpu = run_c32(&cpu_pool, &x, &y, alpha, &values, &spx);
        let gpu = run_c32(&gpu_pool, &x, &y, alpha, &values, &spx);

        for i in 0..cpu.0.len() {
            close_c32(&format!("c32/axpy[{i}]"), gpu.0[i], cpu.0[i]);
        }
        close_c32("c32/dot", gpu.1, cpu.1);
        close_f32("c32/nrm2", gpu.2, cpu.2);
        for i in 0..cpu.3.len() {
            close_c32(&format!("c32/scale[{i}]"), gpu.3[i], cpu.3[i]);
        }
        for i in 0..cpu.4.len() {
            close_c32(&format!("c32/spmv[{i}]"), gpu.4[i], cpu.4[i]);
        }
    }

    fn run_c32<B: Backend>(
        pool: &MemoryPool<B>,
        x: &[Complex32],
        y: &[Complex32],
        alpha: Complex32,
        values: &[Complex32],
        spx: &[Complex32],
    ) -> (
        Vec<Complex32>,
        Complex32,
        f32,
        Vec<Complex32>,
        Vec<Complex32>,
    ) {
        let mut xb = pool.try_alloc::<Complex32>(x.len()).unwrap();
        let mut yb = pool.try_alloc::<Complex32>(y.len()).unwrap();
        pool.backend().copy_h2d(x, xb.raw_mut());
        pool.backend().copy_h2d(y, yb.raw_mut());
        pool.backend().axpy(alpha, xb.raw(), yb.raw_mut());
        let mut axpy_out = vec![Complex32::new(0.0, 0.0); y.len()];
        pool.backend().copy_d2h(yb.raw(), &mut axpy_out);

        let mut x2 = pool.try_alloc::<Complex32>(x.len()).unwrap();
        let mut y2 = pool.try_alloc::<Complex32>(y.len()).unwrap();
        pool.backend().copy_h2d(x, x2.raw_mut());
        pool.backend().copy_h2d(y, y2.raw_mut());
        let dot = pool.backend().dot(x2.raw(), y2.raw());
        let nrm = pool.backend().nrm2(x2.raw());
        pool.backend().scale(alpha, x2.raw_mut());
        let mut scale_out = vec![Complex32::new(0.0, 0.0); x.len()];
        pool.backend().copy_d2h(x2.raw(), &mut scale_out);

        let mut rb = pool.try_alloc::<i32>(ROW_PTR.len()).unwrap();
        let mut cb = pool.try_alloc::<i32>(COL_IDX.len()).unwrap();
        let mut vb = pool.try_alloc::<Complex32>(values.len()).unwrap();
        let mut spxb = pool.try_alloc::<Complex32>(spx.len()).unwrap();
        let mut spyb = pool.try_alloc::<Complex32>(4).unwrap();
        pool.backend().copy_h2d(&ROW_PTR, rb.raw_mut());
        pool.backend().copy_h2d(&COL_IDX, cb.raw_mut());
        pool.backend().copy_h2d(values, vb.raw_mut());
        pool.backend().copy_h2d(spx, spxb.raw_mut());
        pool.backend().spmv_csr(
            4,
            4,
            rb.raw(),
            cb.raw(),
            vb.raw(),
            spxb.raw(),
            spyb.raw_mut(),
        );
        let mut spmv_out = vec![Complex32::new(0.0, 0.0); 4];
        pool.backend().copy_d2h(spyb.raw(), &mut spmv_out);

        (axpy_out, dot, nrm, scale_out, spmv_out)
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_matches_cpu_complex64() {
        let gpu_backend = match CudaBackend::new_default() {
            Ok(b) => b,
            Err(e) => skip_or_require_cuda!("complex64 parity", e),
        };
        let cpu_pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let gpu_pool = MemoryPool::new(gpu_backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();

        let n = 8usize;
        let x: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(0.5 * i as f64 - 1.5, 0.1 * i as f64))
            .collect();
        let y: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(0.25 * i as f64 + 0.75, -0.2 * i as f64))
            .collect();
        let alpha = Complex64::new(1.7, 0.3);
        let values: Vec<Complex64> = SPMAT_VALS
            .iter()
            .map(|&v| Complex64::new(v as f64, 0.0))
            .collect();
        let spx: Vec<Complex64> = (0..4)
            .map(|i| Complex64::new(i as f64 + 1.0, 0.05 * i as f64))
            .collect();

        let cpu = run_c64(&cpu_pool, &x, &y, alpha, &values, &spx);
        let gpu = run_c64(&gpu_pool, &x, &y, alpha, &values, &spx);

        for i in 0..cpu.0.len() {
            close_c64(&format!("c64/axpy[{i}]"), gpu.0[i], cpu.0[i]);
        }
        close_c64("c64/dot", gpu.1, cpu.1);
        close_f64("c64/nrm2", gpu.2, cpu.2);
        for i in 0..cpu.3.len() {
            close_c64(&format!("c64/scale[{i}]"), gpu.3[i], cpu.3[i]);
        }
        for i in 0..cpu.4.len() {
            close_c64(&format!("c64/spmv[{i}]"), gpu.4[i], cpu.4[i]);
        }
    }

    fn run_c64<B: Backend>(
        pool: &MemoryPool<B>,
        x: &[Complex64],
        y: &[Complex64],
        alpha: Complex64,
        values: &[Complex64],
        spx: &[Complex64],
    ) -> (
        Vec<Complex64>,
        Complex64,
        f64,
        Vec<Complex64>,
        Vec<Complex64>,
    ) {
        let mut xb = pool.try_alloc::<Complex64>(x.len()).unwrap();
        let mut yb = pool.try_alloc::<Complex64>(y.len()).unwrap();
        pool.backend().copy_h2d(x, xb.raw_mut());
        pool.backend().copy_h2d(y, yb.raw_mut());
        pool.backend().axpy(alpha, xb.raw(), yb.raw_mut());
        let mut axpy_out = vec![Complex64::new(0.0, 0.0); y.len()];
        pool.backend().copy_d2h(yb.raw(), &mut axpy_out);

        let mut x2 = pool.try_alloc::<Complex64>(x.len()).unwrap();
        let mut y2 = pool.try_alloc::<Complex64>(y.len()).unwrap();
        pool.backend().copy_h2d(x, x2.raw_mut());
        pool.backend().copy_h2d(y, y2.raw_mut());
        let dot = pool.backend().dot(x2.raw(), y2.raw());
        let nrm = pool.backend().nrm2(x2.raw());
        pool.backend().scale(alpha, x2.raw_mut());
        let mut scale_out = vec![Complex64::new(0.0, 0.0); x.len()];
        pool.backend().copy_d2h(x2.raw(), &mut scale_out);

        let mut rb = pool.try_alloc::<i32>(ROW_PTR.len()).unwrap();
        let mut cb = pool.try_alloc::<i32>(COL_IDX.len()).unwrap();
        let mut vb = pool.try_alloc::<Complex64>(values.len()).unwrap();
        let mut spxb = pool.try_alloc::<Complex64>(spx.len()).unwrap();
        let mut spyb = pool.try_alloc::<Complex64>(4).unwrap();
        pool.backend().copy_h2d(&ROW_PTR, rb.raw_mut());
        pool.backend().copy_h2d(&COL_IDX, cb.raw_mut());
        pool.backend().copy_h2d(values, vb.raw_mut());
        pool.backend().copy_h2d(spx, spxb.raw_mut());
        pool.backend().spmv_csr(
            4,
            4,
            rb.raw(),
            cb.raw(),
            vb.raw(),
            spxb.raw(),
            spyb.raw_mut(),
        );
        let mut spmv_out = vec![Complex64::new(0.0, 0.0); 4];
        pool.backend().copy_d2h(spyb.raw(), &mut spmv_out);

        (axpy_out, dot, nrm, scale_out, spmv_out)
    }
}

#[test]
#[cfg(feature = "metal")]
fn metal_matches_cpu_f32() {
    use sonium_compute::MetalBackend;
    let backend = match MetalBackend::new() {
        Ok(b) => b,
        Err(e) => skip_or_require_metal!("f32 parity test", e),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();

    let n = 16usize;
    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 3.0).collect();
    let y: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.25).collect();
    let alpha: f32 = 1.7;

    let mut xb = pool.try_alloc::<f32>(n).unwrap();
    let mut yb = pool.try_alloc::<f32>(n).unwrap();
    pool.backend().copy_h2d(&x, xb.raw_mut());
    pool.backend().copy_h2d(&y, yb.raw_mut());
    pool.backend().axpy(alpha, xb.raw(), yb.raw_mut());
    let mut axpy_out = vec![0.0_f32; n];
    pool.backend().copy_d2h(yb.raw(), &mut axpy_out);

    let mut x2 = pool.try_alloc::<f32>(n).unwrap();
    let mut y2 = pool.try_alloc::<f32>(n).unwrap();
    pool.backend().copy_h2d(&x, x2.raw_mut());
    pool.backend().copy_h2d(&y, y2.raw_mut());
    let dot = pool.backend().dot(x2.raw(), y2.raw());
    let nrm = pool.backend().nrm2(x2.raw());

    pool.backend().scale(alpha, x2.raw_mut());
    let mut scale_out = vec![0.0_f32; n];
    pool.backend().copy_d2h(x2.raw(), &mut scale_out);

    // Reference
    let mut axpy_ref = y.clone();
    for (xi, yi) in x.iter().zip(axpy_ref.iter_mut()) {
        *yi += alpha * xi;
    }
    let dot_ref: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let nrm_ref: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    let scale_ref: Vec<f32> = x.iter().map(|v| alpha * v).collect();

    assert_vec_close_f32("metal_f32/axpy", &axpy_out, &axpy_ref);
    assert_close("metal_f32/dot", dot as f64, dot_ref as f64);
    assert_close("metal_f32/nrm2", nrm as f64, nrm_ref as f64);
    assert_vec_close_f32("metal_f32/scale", &scale_out, &scale_ref);
}

#[test]
#[cfg(feature = "metal")]
fn metal_matches_cpu_complex32() {
    use num_complex::Complex32;
    use sonium_compute::MetalBackend;

    let backend = match MetalBackend::new() {
        Ok(b) => b,
        Err(e) => skip_or_require_metal!("c32 parity test", e),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();

    let n = 8usize;
    let x: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new(0.5 * i as f32 - 1.5, 0.1 * i as f32))
        .collect();
    let y: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new(0.25 * i as f32 + 0.75, -0.2 * i as f32))
        .collect();
    let alpha = Complex32::new(1.7, 0.3);

    let mut xb = pool.try_alloc::<Complex32>(n).unwrap();
    let mut yb = pool.try_alloc::<Complex32>(n).unwrap();
    pool.backend().copy_h2d(&x, xb.raw_mut());
    pool.backend().copy_h2d(&y, yb.raw_mut());
    pool.backend().axpy(alpha, xb.raw(), yb.raw_mut());
    let mut axpy_out = vec![Complex32::new(0.0, 0.0); n];
    pool.backend().copy_d2h(yb.raw(), &mut axpy_out);

    let mut x2 = pool.try_alloc::<Complex32>(n).unwrap();
    let mut y2 = pool.try_alloc::<Complex32>(n).unwrap();
    pool.backend().copy_h2d(&x, x2.raw_mut());
    pool.backend().copy_h2d(&y, y2.raw_mut());
    let dot = pool.backend().dot(x2.raw(), y2.raw());
    let nrm = pool.backend().nrm2(x2.raw());

    pool.backend().scale(alpha, x2.raw_mut());
    let mut scale_out = vec![Complex32::new(0.0, 0.0); n];
    pool.backend().copy_d2h(x2.raw(), &mut scale_out);

    // Reference
    let mut axpy_ref = y.clone();
    for (xi, yi) in x.iter().zip(axpy_ref.iter_mut()) {
        *yi += alpha * xi;
    }
    let mut dot_ref = Complex32::new(0.0, 0.0);
    for (xi, yi) in x.iter().zip(y.iter()) {
        dot_ref += xi.conj() * yi;
    }
    let nrm_ref: f32 = x
        .iter()
        .map(|c| c.re * c.re + c.im * c.im)
        .sum::<f32>()
        .sqrt();
    let scale_ref: Vec<Complex32> = x.iter().map(|v| alpha * v).collect();

    for i in 0..n {
        close_c32(&format!("metal_c32/axpy[{i}]"), axpy_out[i], axpy_ref[i]);
    }
    close_c32("metal_c32/dot", dot, dot_ref);
    close_f32("metal_c32/nrm2", nrm, nrm_ref);
    for i in 0..n {
        close_c32(&format!("metal_c32/scale[{i}]"), scale_out[i], scale_ref[i]);
    }
}

#[test]
#[cfg(feature = "metal")]
fn metal_multi_dot_f32_matches_cpu() {
    use sonium_compute::MetalBackend;

    let backend = match MetalBackend::new() {
        Ok(b) => b,
        Err(e) => skip_or_require_metal!("multi_dot parity", e),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
    let n = 257usize;
    let k = 5usize;
    let y_host: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).sin()).collect();
    let basis_host: Vec<Vec<f32>> = (0..k)
        .map(|j| {
            (0..n)
                .map(|i| ((i + 1) as f32).ln() + j as f32 * 0.125)
                .collect()
        })
        .collect();

    let mut y = pool.try_alloc::<f32>(n).unwrap();
    pool.backend().copy_h2d(&y_host, y.raw_mut());
    let mut basis = Vec::with_capacity(k);
    for host in &basis_host {
        let mut buf = pool.try_alloc::<f32>(n).unwrap();
        pool.backend().copy_h2d(host, buf.raw_mut());
        basis.push(buf);
    }
    let basis_refs: Vec<_> = basis.iter().map(|b| b.raw()).collect();
    let got = pool.backend().multi_dot(&basis_refs, y.raw());

    let want: Vec<f32> = basis_host
        .iter()
        .map(|x| x.iter().zip(y_host.iter()).map(|(a, b)| a * b).sum())
        .collect();
    assert_vec_close_f32("metal/multi_dot_f32", &got, &want);
}

#[test]
#[cfg(feature = "metal")]
fn metal_multi_dot_strided_matches_cpu() {
    use sonium_compute::MetalBackend;

    let backend = match MetalBackend::new() {
        Ok(b) => b,
        Err(e) => skip_or_require_metal!("strided multi_dot parity", e),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
    let n = 257usize;
    let k = 5usize;
    let y_host: Vec<f32> = (0..n).map(|i| (i as f32 * 0.09).cos()).collect();
    let basis_host: Vec<f32> = (0..k)
        .flat_map(|j| (0..n).map(move |i| ((i + 1) as f32).sqrt() + j as f32 * 0.125))
        .collect();
    let mut y = pool.try_alloc::<f32>(n).unwrap();
    pool.backend().copy_h2d(&y_host, y.raw_mut());
    let mut basis = pool.try_alloc::<f32>(n * k).unwrap();
    pool.backend().copy_h2d(&basis_host, basis.raw_mut());
    let got = pool.backend().multi_dot_strided(basis.raw(), n, k, y.raw());
    let want: Vec<f32> = (0..k)
        .map(|col| {
            basis_host[col * n..(col + 1) * n]
                .iter()
                .zip(y_host.iter())
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect();
    assert_vec_close_f32("metal/multi_dot_strided_f32", &got, &want);

    let y_complex_host: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new((i as f32 * 0.11).sin(), (i as f32 * 0.05).cos()))
        .collect();
    let basis_complex_host: Vec<Complex32> = (0..k)
        .flat_map(|j| {
            (0..n).map(move |i| {
                Complex32::new(
                    ((i + 1) as f32).ln() + j as f32 * 0.2,
                    (i as f32 * 0.03 + j as f32).sin(),
                )
            })
        })
        .collect();
    let mut y_complex = pool.try_alloc::<Complex32>(n).unwrap();
    pool.backend()
        .copy_h2d(&y_complex_host, y_complex.raw_mut());
    let mut basis_complex = pool.try_alloc::<Complex32>(n * k).unwrap();
    pool.backend()
        .copy_h2d(&basis_complex_host, basis_complex.raw_mut());
    let got_complex = pool
        .backend()
        .multi_dot_strided(basis_complex.raw(), n, k, y_complex.raw());
    let want_complex: Vec<Complex32> = (0..k)
        .map(|col| {
            basis_complex_host[col * n..(col + 1) * n]
                .iter()
                .zip(y_complex_host.iter())
                .map(|(a, b)| a.conj() * b)
                .sum()
        })
        .collect();
    for (i, (got, want)) in got_complex.iter().zip(want_complex.iter()).enumerate() {
        close_c32(&format!("metal/multi_dot_strided_c32[{i}]"), *got, *want);
    }
}

#[test]
#[cfg(feature = "metal")]
fn metal_spmm_prebaked_colmajor_matches_repeated_spmv() {
    use sonium_compute::MetalBackend;

    let backend = match MetalBackend::new() {
        Ok(b) => b,
        Err(e) => skip_or_require_metal!("SpMM parity", e),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();

    let m = 3usize;
    let n = 3usize;
    let rhs = 2usize;
    let row_ptr = vec![0_i32, 2, 4, 6];
    let col_idx = vec![0_i32, 1, 1, 2, 0, 2];
    let values = vec![2.0_f32, -1.0, 3.0, 0.5, -2.0, 4.0];
    let x = vec![1.0_f32, 2.0, -1.0, 0.5, -0.25, 3.0];

    let mut row_dev = pool.try_alloc::<i32>(row_ptr.len()).unwrap();
    let mut col_dev = pool.try_alloc::<i32>(col_idx.len()).unwrap();
    let mut val_dev = pool.try_alloc::<f32>(values.len()).unwrap();
    let mut x_dev = pool.try_alloc::<f32>(x.len()).unwrap();
    let mut y_spmm = pool.try_alloc::<f32>(m * rhs).unwrap();
    pool.backend().copy_h2d(&row_ptr, row_dev.raw_mut());
    pool.backend().copy_h2d(&col_idx, col_dev.raw_mut());
    pool.backend().copy_h2d(&values, val_dev.raw_mut());
    pool.backend().copy_h2d(&x, x_dev.raw_mut());

    let prebaked =
        pool.backend()
            .prebake_csr::<f32>(m, n, row_dev.raw(), col_dev.raw(), val_dev.raw());
    pool.backend()
        .spmm_prebaked_colmajor(&prebaked, rhs, x_dev.raw(), y_spmm.raw_mut());

    let mut want = vec![0.0_f32; m * rhs];
    for col in 0..rhs {
        let x_col = &x[col * n..(col + 1) * n];
        for row in 0..m {
            let start = row_ptr[row] as usize;
            let end = row_ptr[row + 1] as usize;
            for idx in start..end {
                want[col * m + row] += values[idx] * x_col[col_idx[idx] as usize];
            }
        }
    }

    let mut got = vec![0.0_f32; m * rhs];
    pool.backend().copy_d2h(y_spmm.raw(), &mut got);
    assert_vec_close_f32("metal/spmm_prebaked_colmajor", &got, &want);
}

#[test]
#[cfg(feature = "metal")]
fn metal_gather4_f32_matches_cpu() {
    use sonium_compute::MetalBackend;

    let backend = match MetalBackend::new() {
        Ok(b) => b,
        Err(e) => skip_or_require_metal!("gather4 parity", e),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();

    let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let indices = vec![0, 1, 2, 3, 4, 3, 1, 0, 2, 2, 2, 2];
    let weights = vec![
        0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.25, 0.25, 0.0, 1.0, 0.0, 0.0, 0.0,
    ];
    let expected = vec![
        0.1 * 1.0 + 0.2 * 2.0 + 0.3 * 3.0 + 0.4 * 4.0,
        0.5 * 5.0 + 0.25 * 4.0 + 0.25 * 2.0 + 0.0 * 1.0,
        1.0 * 3.0 + 0.0 * 3.0 + 0.0 * 3.0 + 0.0 * 3.0,
    ];

    let mut values_dev = pool.try_alloc::<f32>(values.len()).unwrap();
    let mut indices_dev = pool.try_alloc::<i32>(indices.len()).unwrap();
    let mut weights_dev = pool.try_alloc::<f32>(weights.len()).unwrap();
    let mut out_dev = pool.try_alloc::<f32>(expected.len()).unwrap();
    pool.backend().copy_h2d(&values, values_dev.raw_mut());
    pool.backend().copy_h2d(&indices, indices_dev.raw_mut());
    pool.backend().copy_h2d(&weights, weights_dev.raw_mut());
    pool.backend().gather4_weighted(
        values_dev.raw(),
        indices_dev.raw(),
        weights_dev.raw(),
        out_dev.raw_mut(),
    );

    let mut out = vec![0.0_f32; expected.len()];
    pool.backend().copy_d2h(out_dev.raw(), &mut out);
    assert_vec_close_f32("metal/gather4_f32", &out, &expected);
}

#[test]
#[cfg(feature = "metal")]
fn metal_gather4_c32_matches_cpu() {
    use num_complex::Complex32;
    use sonium_compute::MetalBackend;

    let backend = match MetalBackend::new() {
        Ok(b) => b,
        Err(e) => skip_or_require_metal!("gather4 c32 parity", e),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();

    let values: Vec<Complex32> = vec![
        Complex32::new(1.0, 0.5),
        Complex32::new(2.0, -1.0),
        Complex32::new(-1.0, 3.0),
        Complex32::new(4.0, 2.0),
        Complex32::new(-2.0, -0.5),
    ];
    let indices = vec![0, 1, 2, 3, 4, 3, 1, 0, 2, 2, 2, 2];
    let weights = vec![
        0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.25, 0.25, 0.0, 1.0, 0.0, 0.0, 0.0,
    ];

    let mut values_dev = pool.try_alloc::<Complex32>(values.len()).unwrap();
    let mut indices_dev = pool.try_alloc::<i32>(indices.len()).unwrap();
    let mut weights_dev = pool.try_alloc::<f32>(weights.len()).unwrap();
    let mut out_dev = pool.try_alloc::<Complex32>(indices.len() / 4).unwrap();
    pool.backend().copy_h2d(&values, values_dev.raw_mut());
    pool.backend().copy_h2d(&indices, indices_dev.raw_mut());
    pool.backend().copy_h2d(&weights, weights_dev.raw_mut());
    pool.backend().gather4_weighted(
        values_dev.raw(),
        indices_dev.raw(),
        weights_dev.raw(),
        out_dev.raw_mut(),
    );

    let mut out = vec![Complex32::new(0.0, 0.0); indices.len() / 4];
    pool.backend().copy_d2h(out_dev.raw(), &mut out);

    // CPU reference
    let cpu_pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
    let mut values_cpu = cpu_pool.try_alloc::<Complex32>(values.len()).unwrap();
    let mut indices_cpu = cpu_pool.try_alloc::<i32>(indices.len()).unwrap();
    let mut weights_cpu = cpu_pool.try_alloc::<f32>(weights.len()).unwrap();
    let mut out_cpu = cpu_pool.try_alloc::<Complex32>(indices.len() / 4).unwrap();
    cpu_pool.backend().copy_h2d(&values, values_cpu.raw_mut());
    cpu_pool.backend().copy_h2d(&indices, indices_cpu.raw_mut());
    cpu_pool.backend().copy_h2d(&weights, weights_cpu.raw_mut());
    cpu_pool.backend().gather4_weighted(
        values_cpu.raw(),
        indices_cpu.raw(),
        weights_cpu.raw(),
        out_cpu.raw_mut(),
    );
    let mut expected = vec![Complex32::new(0.0, 0.0); indices.len() / 4];
    cpu_pool.backend().copy_d2h(out_cpu.raw(), &mut expected);

    for i in 0..expected.len() {
        close_c32(&format!("metal/gather4_c32[{i}]"), out[i], expected[i]);
    }
}

#[test]
#[cfg(feature = "metal")]
fn metal_fmm_c32_phase_and_compact_indices_match_cpu() {
    use num_complex::Complex32;
    use sonium_compute::MetalBackend;

    let backend = match MetalBackend::new() {
        Ok(b) => b,
        Err(e) => skip_or_require_metal!("FMM c32 compact parity", e),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();

    let src = vec![
        Complex32::new(0.5, -1.0),
        Complex32::new(2.0, 0.25),
        Complex32::new(-0.5, 0.5),
        Complex32::new(1.5, -0.25),
        Complex32::new(0.75, 2.0),
    ];
    let phase = vec![
        Complex32::new(1.0, 0.5),
        Complex32::new(-0.25, 0.75),
        Complex32::new(0.0, -2.0),
        Complex32::new(3.0, 1.0),
        Complex32::new(-1.0, 0.0),
    ];
    let base = vec![
        Complex32::new(0.1, 0.2),
        Complex32::new(0.2, 0.3),
        Complex32::new(0.3, 0.4),
        Complex32::new(0.4, 0.5),
        Complex32::new(0.5, 0.6),
    ];
    let active = vec![4_i32, 2, 0];

    let mut src_dev = pool.try_alloc::<Complex32>(src.len()).unwrap();
    let mut phase_dev = pool.try_alloc::<Complex32>(phase.len()).unwrap();
    let mut dst_dev = pool.try_alloc::<Complex32>(base.len()).unwrap();
    let mut active_dev = pool.try_alloc::<i32>(active.len()).unwrap();
    let mut compact_dev = pool.try_alloc::<Complex32>(active.len()).unwrap();
    let mut scatter_dev = pool.try_alloc::<Complex32>(base.len()).unwrap();
    pool.backend().copy_h2d(&src, src_dev.raw_mut());
    pool.backend().copy_h2d(&phase, phase_dev.raw_mut());
    pool.backend().copy_h2d(&base, dst_dev.raw_mut());
    pool.backend().copy_h2d(&active, active_dev.raw_mut());
    pool.backend().copy_h2d(&base, scatter_dev.raw_mut());

    pool.backend()
        .fmm_phase_mul_accumulate_c32(dst_dev.raw_mut(), phase_dev.raw(), src_dev.raw());
    pool.backend().fmm_compact_indexed_gather_c32(
        src_dev.raw(),
        active_dev.raw(),
        compact_dev.raw_mut(),
    );
    pool.backend().fmm_compact_indexed_scatter_accumulate_c32(
        compact_dev.raw(),
        active_dev.raw(),
        scatter_dev.raw_mut(),
    );

    let mut phase_out = vec![Complex32::new(0.0, 0.0); base.len()];
    let mut compact_out = vec![Complex32::new(0.0, 0.0); active.len()];
    let mut scatter_out = vec![Complex32::new(0.0, 0.0); base.len()];
    pool.backend().copy_d2h(dst_dev.raw(), &mut phase_out);
    pool.backend().copy_d2h(compact_dev.raw(), &mut compact_out);
    pool.backend().copy_d2h(scatter_dev.raw(), &mut scatter_out);

    for i in 0..base.len() {
        close_c32(
            &format!("metal/fmm_phase_c32[{i}]"),
            phase_out[i],
            base[i] + phase[i] * src[i],
        );
    }
    for (compact_idx, &src_idx) in active.iter().enumerate() {
        close_c32(
            &format!("metal/fmm_compact_gather_c32[{compact_idx}]"),
            compact_out[compact_idx],
            src[src_idx as usize],
        );
    }
    let mut scatter_expected = base.clone();
    for (compact_idx, &dst_idx) in active.iter().enumerate() {
        scatter_expected[dst_idx as usize] += src[active[compact_idx] as usize];
    }
    for i in 0..base.len() {
        close_c32(
            &format!("metal/fmm_compact_scatter_c32[{i}]"),
            scatter_out[i],
            scatter_expected[i],
        );
    }
}

#[test]
#[cfg(feature = "metal")]
fn metal_fmm_c32_indexed_dense_matvec_accumulate_matches_cpu() {
    use num_complex::Complex32;
    use sonium_compute::MetalBackend;

    let backend = match MetalBackend::new() {
        Ok(b) => b,
        Err(e) => skip_or_require_metal!("FMM indexed dense matvec parity", e),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();

    let rows = 2usize;
    let cols = 3usize;
    let lda = 3usize;
    let matrix = vec![
        Complex32::new(1.0, 0.0),
        Complex32::new(0.0, 1.0),
        Complex32::new(2.0, -1.0),
        Complex32::new(-0.5, 0.25),
        Complex32::new(0.75, 0.5),
        Complex32::new(1.25, -0.75),
    ];
    let src = vec![
        Complex32::new(10.0, 0.0),
        Complex32::new(1.0, 2.0),
        Complex32::new(-3.0, 0.5),
        Complex32::new(0.25, -1.5),
    ];
    let src_indices = vec![2_i32, 0, 3];
    let dst_indices = vec![4_i32, 1];
    let base = vec![Complex32::new(0.2, -0.1); 6];

    let mut matrix_dev = pool.try_alloc::<Complex32>(matrix.len()).unwrap();
    let mut src_dev = pool.try_alloc::<Complex32>(src.len()).unwrap();
    let mut src_idx_dev = pool.try_alloc::<i32>(src_indices.len()).unwrap();
    let mut dst_dev = pool.try_alloc::<Complex32>(base.len()).unwrap();
    let mut dst_idx_dev = pool.try_alloc::<i32>(dst_indices.len()).unwrap();
    pool.backend().copy_h2d(&matrix, matrix_dev.raw_mut());
    pool.backend().copy_h2d(&src, src_dev.raw_mut());
    pool.backend().copy_h2d(&src_indices, src_idx_dev.raw_mut());
    pool.backend().copy_h2d(&base, dst_dev.raw_mut());
    pool.backend().copy_h2d(&dst_indices, dst_idx_dev.raw_mut());

    pool.backend().fmm_indexed_dense_matvec_accumulate_c32(
        matrix_dev.raw(),
        rows,
        cols,
        lda,
        src_dev.raw(),
        src_idx_dev.raw(),
        dst_dev.raw_mut(),
        dst_idx_dev.raw(),
    );

    let mut got = vec![Complex32::new(0.0, 0.0); base.len()];
    pool.backend().copy_d2h(dst_dev.raw(), &mut got);

    let mut expected = base;
    for row in 0..rows {
        let mut acc = Complex32::new(0.0, 0.0);
        for col in 0..cols {
            acc += matrix[row * lda + col] * src[src_indices[col] as usize];
        }
        expected[dst_indices[row] as usize] += acc;
    }
    for i in 0..got.len() {
        close_c32(
            &format!("metal/fmm_indexed_dense_matvec_c32[{i}]"),
            got[i],
            expected[i],
        );
    }
}

#[test]
#[cfg(feature = "metal")]
fn metal_precision_casts_roundtrip() {
    use sonium_compute::MetalBackend;

    let backend = match MetalBackend::new() {
        Ok(b) => b,
        Err(e) => skip_or_require_metal!("precision-cast parity", e),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();

    let real: Vec<f64> = vec![-3.25, -1.5, -0.125, 0.0, 0.25, 1.75, 8.5];
    let mut real_upper = pool.try_alloc::<f64>(real.len()).unwrap();
    let mut real_lower = pool.try_alloc::<f32>(real.len()).unwrap();
    let mut real_roundtrip = pool.try_alloc::<f64>(real.len()).unwrap();
    pool.backend().copy_h2d(&real, real_upper.raw_mut());
    pool.backend()
        .cast_to_lower::<f64>(real_upper.raw(), real_lower.raw_mut());
    pool.backend()
        .cast_from_lower::<f64>(real_lower.raw(), real_roundtrip.raw_mut());

    let mut real_lower_host = vec![0.0_f32; real.len()];
    let mut real_roundtrip_host = vec![0.0_f64; real.len()];
    pool.backend()
        .copy_d2h(real_lower.raw(), &mut real_lower_host);
    pool.backend()
        .copy_d2h(real_roundtrip.raw(), &mut real_roundtrip_host);
    for i in 0..real.len() {
        close_f32(
            &format!("metal_cast/f64_to_f32[{i}]"),
            real_lower_host[i],
            real[i] as f32,
        );
        close_f64(
            &format!("metal_cast/f32_to_f64[{i}]"),
            real_roundtrip_host[i],
            real[i] as f32 as f64,
        );
    }

    let complex: Vec<Complex64> = (0..real.len())
        .map(|i| Complex64::new(real[i], real[real.len() - 1 - i] * -0.5))
        .collect();
    let mut complex_upper = pool.try_alloc::<Complex64>(complex.len()).unwrap();
    let mut complex_lower = pool.try_alloc::<Complex32>(complex.len()).unwrap();
    let mut complex_roundtrip = pool.try_alloc::<Complex64>(complex.len()).unwrap();
    pool.backend().copy_h2d(&complex, complex_upper.raw_mut());
    pool.backend()
        .cast_to_lower::<Complex64>(complex_upper.raw(), complex_lower.raw_mut());
    pool.backend()
        .cast_from_lower::<Complex64>(complex_lower.raw(), complex_roundtrip.raw_mut());

    let mut complex_lower_host = vec![Complex32::new(0.0, 0.0); complex.len()];
    let mut complex_roundtrip_host = vec![Complex64::new(0.0, 0.0); complex.len()];
    pool.backend()
        .copy_d2h(complex_lower.raw(), &mut complex_lower_host);
    pool.backend()
        .copy_d2h(complex_roundtrip.raw(), &mut complex_roundtrip_host);
    for i in 0..complex.len() {
        let lower_expected = Complex32::new(complex[i].re as f32, complex[i].im as f32);
        let roundtrip_expected = Complex64::new(lower_expected.re as f64, lower_expected.im as f64);
        close_c32(
            &format!("metal_cast/c64_to_c32[{i}]"),
            complex_lower_host[i],
            lower_expected,
        );
        close_c64(
            &format!("metal_cast/c32_to_c64[{i}]"),
            complex_roundtrip_host[i],
            roundtrip_expected,
        );
    }
}

#[test]
#[cfg(feature = "metal")]
fn metal_matches_cpu() {
    use sonium_compute::MetalBackend;
    let backend = match MetalBackend::new() {
        Ok(b) => b,
        Err(e) => skip_or_require_metal!("parity test", e),
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
    let inp = make_inputs();
    let cpu = cpu_reference(&inp);
    let gpu = run_on_backend(&pool, &inp);
    assert_parity("metal", &cpu, &gpu);
}

#[cfg(feature = "cuda")]
mod cuda_failure_contract {
    use super::*;
    use sonium_compute::{
        BackendError, CudaBackend, CudaFailureInjectionPoint, MemoryPool, MemoryPoolConfig,
        OomError,
    };

    fn assert_recoverable_failure(backend: &CudaBackend, error: &BackendError) {
        assert_eq!(
            backend.pending_error().expect("pending CUDA error query"),
            Some(error.clone()),
            "the first checked CUDA failure must be retained exactly"
        );
        backend
            .recover_after_error()
            .expect("recoverable CUDA failure must synchronize and clear");
        assert_eq!(
            backend.pending_error().expect("cleared CUDA error query"),
            None
        );
    }

    #[test]
    fn checked_resource_failures_preserve_outputs_refund_quota_and_retry() {
        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend,
            Err(error) => skip_or_require_cuda!("CUDA resource failure contract", error),
        };
        let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(16 << 20))
            .expect("CUDA resource-contract pool");

        // The allocation hook fires before either ledger is charged. A
        // failed MemoryPool reservation must be refundable and retryable.
        let pool_bytes = pool.bytes_in_use();
        let backend_bytes = pool.backend().runtime_stats().live_allocation_bytes;
        pool.backend()
            .inject_failure(CudaFailureInjectionPoint::Allocation);
        let error = pool
            .try_alloc::<f64>(4)
            .expect_err("injected allocation failure must be typed");
        assert!(
            matches!(error, OomError::DriverOom(reason) if reason.contains("injected CUDA allocation failure"))
        );
        assert_eq!(pool.bytes_in_use(), pool_bytes);
        assert_eq!(
            pool.backend().runtime_stats().live_allocation_bytes,
            backend_bytes
        );
        let retry = pool
            .try_alloc::<f64>(4)
            .expect("allocation must remain retryable after injection");
        drop(retry);

        let mut source = pool.try_alloc::<f64>(2).expect("source buffer");
        let mut destination = pool.try_alloc::<f64>(2).expect("destination buffer");
        let source_values = [1.25, -2.5];
        let destination_values = [7.0, -3.0];
        pool.backend()
            .try_copy_h2d(&source_values, source.raw_mut())
            .expect("source upload");
        pool.backend()
            .try_copy_h2d(&destination_values, destination.raw_mut())
            .expect("destination upload");

        // H2D rejection occurs before the native copy and leaves the device
        // destination unchanged. The pending error is recoverable.
        pool.backend()
            .inject_failure(CudaFailureInjectionPoint::TransferH2D);
        let error = pool
            .backend()
            .try_copy_h2d(&[-9.0, 11.0], source.raw_mut())
            .expect_err("injected H2D failure must be typed");
        assert!(matches!(
            error,
            BackendError::Transfer {
                operation: "cuda_copy_h2d",
                ..
            }
        ));
        assert_recoverable_failure(pool.backend(), &error);
        let mut observed = [0.0; 2];
        pool.backend()
            .try_copy_d2h(source.raw(), &mut observed)
            .expect("source readback after H2D failure");
        assert_eq!(observed, source_values);

        // D2H rejection leaves the caller-owned host destination untouched.
        let mut host_destination = [91.0, -37.0];
        pool.backend()
            .inject_failure(CudaFailureInjectionPoint::TransferD2H);
        let error = pool
            .backend()
            .try_copy_d2h(source.raw(), &mut host_destination)
            .expect_err("injected D2H failure must be typed");
        assert!(matches!(
            error,
            BackendError::Transfer {
                operation: "cuda_copy_d2h",
                ..
            }
        ));
        assert_recoverable_failure(pool.backend(), &error);
        assert_eq!(host_destination, [91.0, -37.0]);

        // D2D rejection leaves the device destination untouched.
        pool.backend()
            .inject_failure(CudaFailureInjectionPoint::TransferD2D);
        let error = pool
            .backend()
            .try_copy_d2d(source.raw(), destination.raw_mut())
            .expect_err("injected D2D failure must be typed");
        assert!(matches!(
            error,
            BackendError::Transfer {
                operation: "cuda_copy_d2d",
                ..
            }
        ));
        assert_recoverable_failure(pool.backend(), &error);
        let mut observed = [0.0; 2];
        pool.backend()
            .try_copy_d2h(destination.raw(), &mut observed)
            .expect("destination readback after D2D failure");
        assert_eq!(observed, destination_values);

        // Event-record failure uses the same recoverable launch boundary and
        // must not create a completion token that could falsely succeed.
        pool.backend()
            .inject_failure(CudaFailureInjectionPoint::EventRecord);
        let error = match pool.backend().try_submit_completion() {
            Ok(_) => panic!("injected event-record failure must be typed"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            BackendError::Launch {
                operation: "cuda_event_record",
                ..
            }
        ));
        assert_recoverable_failure(pool.backend(), &error);
    }

    #[test]
    fn checked_descriptor_failure_is_typed_and_retryable() {
        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend,
            Err(error) => skip_or_require_cuda!("CUDA descriptor failure contract", error),
        };
        let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(16 << 20))
            .expect("CUDA descriptor-contract pool");
        let mut row_ptr = pool.try_alloc::<i32>(3).expect("row pointer");
        let mut col_idx = pool.try_alloc::<i32>(2).expect("column index");
        let mut values = pool.try_alloc::<f32>(2).expect("CSR values");
        pool.backend()
            .try_copy_h2d(&[0, 1, 2], row_ptr.raw_mut())
            .expect("row-pointer upload");
        pool.backend()
            .try_copy_h2d(&[0, 1], col_idx.raw_mut())
            .expect("column-index upload");
        pool.backend()
            .try_copy_h2d(&[2.0, 3.0], values.raw_mut())
            .expect("value upload");

        pool.backend()
            .inject_failure(CudaFailureInjectionPoint::DescriptorCreation);
        let error =
            match pool
                .backend()
                .try_prebake_csr(2, 2, row_ptr.raw(), col_idx.raw(), values.raw())
            {
                Ok(_) => panic!("injected descriptor failure must be typed"),
                Err(error) => error,
            };
        assert!(matches!(
            error,
            BackendError::OperationFailed {
                operation: "cuda_descriptor_creation",
                ..
            }
        ));
        assert_recoverable_failure(pool.backend(), &error);

        // The failed descriptor did not poison the context or charge a
        // workspace leak; the same valid CSR can be constructed afterwards.
        let prebaked = pool
            .backend()
            .try_prebake_csr(2, 2, row_ptr.raw(), col_idx.raw(), values.raw())
            .expect("descriptor construction must be retryable");
        drop(prebaked);
    }

    #[test]
    fn terminal_synchronization_device_loss_and_event_wait_are_not_reusable() {
        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend,
            Err(error) => skip_or_require_cuda!("CUDA terminal failure contract", error),
        };
        let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(16 << 20))
            .expect("CUDA terminal-contract pool");

        pool.backend()
            .inject_failure(CudaFailureInjectionPoint::Synchronization);
        let error = pool
            .backend()
            .try_synchronize()
            .expect_err("injected synchronization failure must be typed");
        assert!(matches!(
            error,
            BackendError::Synchronization {
                operation: "cuda_synchronize",
                ..
            }
        ));
        assert_eq!(
            pool.backend().pending_error().expect("pending sync error"),
            Some(error.clone())
        );
        let recovery = pool
            .backend()
            .recover_after_error()
            .expect_err("synchronization failure requires backend recreation");
        assert_eq!(recovery, error);

        // Use a fresh context for each terminal fault class. A real device
        // loss or failed event wait must never be made to look retryable by a
        // subsequent operation on the poisoned instance.
        drop(pool);
        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend,
            Err(error) => skip_or_require_cuda!("CUDA device-loss contract", error),
        };
        let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(16 << 20))
            .expect("CUDA device-loss pool");
        pool.backend()
            .inject_failure(CudaFailureInjectionPoint::DeviceLoss);
        let error = pool
            .backend()
            .try_synchronize()
            .expect_err("injected device loss must be typed");
        assert!(matches!(
            error,
            BackendError::DeviceLost {
                operation: "cuda_device_loss",
                ..
            }
        ));
        assert_eq!(
            pool.backend()
                .recover_after_error()
                .expect_err("device loss is terminal"),
            error
        );

        drop(pool);
        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend,
            Err(error) => skip_or_require_cuda!("CUDA event-wait contract", error),
        };
        let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(16 << 20))
            .expect("CUDA event-wait pool");
        let mut completion = pool
            .backend()
            .try_submit_completion()
            .expect("completion event creation");
        let resources = pool.backend().runtime_stats().driver_resources;
        assert_eq!(resources.live_contexts, 1);
        assert_eq!(resources.live_streams, 1);
        assert!(resources.live_dynamic_libraries >= 4);
        assert!(resources.live_library_handles >= 3);
        assert!(
            resources.initialization_baseline_free_bytes >= resources.initialization_free_bytes
        );
        assert_eq!(resources.live_completion_events, 1);
        pool.backend()
            .inject_failure(CudaFailureInjectionPoint::EventWait);
        let error = completion
            .wait()
            .expect_err("injected event wait failure must be typed");
        assert!(matches!(
            error,
            BackendError::Synchronization {
                operation: "cuda_event_wait",
                ..
            }
        ));
        assert_eq!(
            pool.backend()
                .recover_after_error()
                .expect_err("event wait is terminal"),
            error
        );
        drop(completion);
        assert_eq!(
            pool.backend()
                .runtime_stats()
                .driver_resources
                .live_completion_events,
            0,
            "dropping an abandoned completion must release its driver-event census"
        );
    }
}

// ---------------------------------------------------------------------------
// BEM regular-pair Helmholtz kernel parity (Phase 1.3 of bem-step1.md)
//
// Compares the GPU MSL `sonium_bem_regular_pair_c32` kernel against an
// inlined `f64` CPU reference that mirrors `sonium-bem`'s
// `regular_integration` in its `iforie == true` (single-subelement,
// no-adaptive-subdivision) branch. The reference is duplicated here
// rather than imported because `sonium-bem` depends on `sonium-compute`,
// not the other way around — the test must stay inside this crate's
// dependency cone.
// ---------------------------------------------------------------------------

mod bem_regular_pair_parity {
    use num_complex::Complex64;
    use sonium_compute::{Backend, BackendError, BemRegularPairIn, BemRegularPairInF64};
    use std::f64::consts::PI;

    /// 4-point Tri3 quadrature, weights pre-scaled by 0.5 (reference
    /// triangle area). Must match `SONIUM_BEM_TRI4_QUAD` in
    /// `kernels.metal` and `GAUCORWEI_TR4` in `gauss.rs` (which is
    /// scaled by `AREA_SCALE = 0.5` at lookup time).
    const TRI4_QUAD: [(f64, f64, f64); 4] = [
        (0.333333333333333, 0.333333333333333, -0.281250),
        (0.6, 0.2, 0.260416666666667),
        (0.2, 0.6, 0.260416666666667),
        (0.2, 0.2, 0.260416666666667),
    ];

    fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }
    fn norm(a: [f64; 3]) -> f64 {
        (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
    }
    fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
    fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }
    fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
        [a[0] * s, a[1] * s, a[2] * s]
    }
    fn axpy(a: [f64; 3], t: f64, b: [f64; 3]) -> [f64; 3] {
        [a[0] + t * b[0], a[1] + t * b[1], a[2] + t * b[2]]
    }

    /// `f64` reference for the four Helmholtz kernel integrals over a
    /// single Tri3 field element with the source point well-separated
    /// from the element (so a single 4-point Gauss quadrature suffices).
    /// `harmonic_factor` is `±1`.
    fn cpu_regular_pair_reference(
        source_point: [f64; 3],
        source_normal: [f64; 3],
        v0: [f64; 3],
        v1: [f64; 3],
        v2: [f64; 3],
        wave_number: f64,
        harmonic_factor: f64,
    ) -> (Complex64, Complex64, Complex64, Complex64) {
        let wavruim = harmonic_factor * wave_number;
        let k2 = wave_number * wave_number;

        let e1 = sub(v1, v0);
        let e2 = sub(v2, v0);
        let raw_n = cross(e1, e2);
        let jac = norm(raw_n);
        let n_y = if jac > 1e-15 {
            scale(raw_n, 1.0 / jac)
        } else {
            [0.0; 3]
        };

        let mut g = Complex64::new(0.0, 0.0);
        let mut h = Complex64::new(0.0, 0.0);
        let mut ht = Complex64::new(0.0, 0.0);
        let mut e = Complex64::new(0.0, 0.0);

        for &(xi, eta, w) in TRI4_QUAD.iter() {
            let n0 = 1.0 - xi - eta;
            let crd = axpy(axpy(scale(v0, n0), xi, v1), eta, v2);
            let wga = w * jac;
            let diff = sub(crd, source_point);
            let r = norm(diff);
            if r < 1e-15 {
                continue;
            }
            let ur = scale(diff, 1.0 / r);
            let re1 = wavruim * r;
            let re2 = wga / (4.0 * PI * r);
            let zg = Complex64::new(re1.cos() * re2, re1.sin() * re2);
            let z1 = Complex64::new(-1.0 / r, wavruim);
            let zhh_base = zg * z1;
            let r1h = dot(ur, n_y);
            let r2h = -dot(ur, source_normal);
            let zhh = zhh_base * r1h;
            let zht = zhh_base * r2h;
            let rq = r1h * r2h;
            let nx_dot_ny = dot(source_normal, n_y);
            let dq = r * r;
            let ze_factor = Complex64::new(
                (3.0 / dq - k2) * rq + nx_dot_ny / dq,
                -wavruim / r * (3.0 * rq + nx_dot_ny),
            );
            let ze = zg * ze_factor;
            g += zg;
            h += zhh;
            ht += zht;
            e += ze;
        }
        (g, h, ht, e)
    }

    /// A small batch of well-separated Tri3 pairs that exercises a range
    /// of source-to-field distances and orientations. All distances are
    /// ≥ 3× element size so the 4-point Gauss rule is the right one
    /// (matches `optimal_quadrature_order(_, _) == 2`).
    #[allow(clippy::type_complexity)]
    fn fixture_pairs() -> Vec<(
        BemRegularPairIn,
        [f64; 3],
        [f64; 3],
        [f64; 3],
        [f64; 3],
        [f64; 3],
    )> {
        let triangles = [
            ([0.0, 0.0, 0.0_f64], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ([2.0, 0.0, 0.0], [3.0, 0.0, 0.5], [2.0, 1.0, 0.5]),
            ([-1.5, -1.0, 1.0], [-0.5, -1.0, 1.2], [-1.0, 0.0, 1.5]),
            ([0.5, 2.0, -1.0], [1.5, 2.5, -1.0], [0.7, 3.0, -0.6]),
        ];

        let sources = [
            ([5.0_f64, 5.0, 5.0], [0.0, 0.0, 1.0_f64]),
            ([0.0, 0.0, 4.0], [0.0, 1.0, 0.0]),
            ([-3.0, 4.0, 2.0], [0.5_f64.sqrt(), 0.5_f64.sqrt(), 0.0]),
        ];

        let mut out = Vec::new();
        for (sp, sn) in sources.iter() {
            for (v0, v1, v2) in triangles.iter() {
                let pair = BemRegularPairIn::from_triangle(
                    [sp[0] as f32, sp[1] as f32, sp[2] as f32],
                    [sn[0] as f32, sn[1] as f32, sn[2] as f32],
                    [v0[0] as f32, v0[1] as f32, v0[2] as f32],
                    [v1[0] as f32, v1[1] as f32, v1[2] as f32],
                    [v2[0] as f32, v2[1] as f32, v2[2] as f32],
                );
                out.push((pair, *sp, *sn, *v0, *v1, *v2));
            }
        }
        out
    }

    fn rel_err(cpu: Complex64, gpu_re: f32, gpu_im: f32) -> f64 {
        let g = Complex64::new(gpu_re as f64, gpu_im as f64);
        let diff = (cpu - g).norm();
        let denom = cpu.norm().max(1e-30);
        diff / denom
    }

    #[test]
    fn cpu_bem_regular_pair_backend_surface_matches_reference() {
        use sonium_compute::CpuBackend;

        let backend = CpuBackend::new();
        let pairs_full = fixture_pairs();
        let pairs: Vec<BemRegularPairIn> = pairs_full.iter().map(|(p, ..)| *p).collect();
        let out = Backend::bem_regular_pair_c32_batch(&backend, &pairs, 1.0, -1.0);

        assert_eq!(out.len(), pairs.len());
        for (i, ((_, sp, sn, v0, v1, v2), got)) in pairs_full.iter().zip(out.iter()).enumerate() {
            let (cg, ch, cht, ce) = cpu_regular_pair_reference(*sp, *sn, *v0, *v1, *v2, 1.0, -1.0);
            let tol = 1e-4_f64;
            let eg = rel_err(cg, got.g.re, got.g.im);
            let eh = rel_err(ch, got.h.re, got.h.im);
            let eht = rel_err(cht, got.ht.re, got.ht.im);
            let ee = rel_err(ce, got.e.re, got.e.im);
            assert!(
                eg < tol && eh < tol && eht < tol && ee < tol,
                "CPU backend pair {i} exceeds tol {tol:.0e}: \
                 G={eg:.3e} H={eh:.3e} H^T={eht:.3e} E={ee:.3e}"
            );
        }
    }

    #[test]
    fn cpu_bem_regular_pair_c64_backend_surface_matches_reference() {
        use sonium_compute::CpuBackend;

        let backend = CpuBackend::new();
        let pairs_full = fixture_pairs();
        let pairs: Vec<BemRegularPairInF64> = pairs_full
            .iter()
            .map(|(_, sp, sn, v0, v1, v2)| {
                BemRegularPairInF64::from_triangle(*sp, *sn, *v0, *v1, *v2)
            })
            .collect();
        let out = backend
            .try_bem_regular_pair_c64_batch(&pairs, 1.0, -1.0)
            .expect("CPU Complex64 BEM pair evaluation");

        assert_eq!(out.len(), pairs.len());
        for (i, ((_, sp, sn, v0, v1, v2), got)) in pairs_full.iter().zip(out.iter()).enumerate() {
            let expected = cpu_regular_pair_reference(*sp, *sn, *v0, *v1, *v2, 1.0, -1.0);
            for (name, actual, reference) in [
                ("G", got.g, expected.0),
                ("H", got.h, expected.1),
                ("H^T", got.ht, expected.2),
                ("E", got.e, expected.3),
            ] {
                let scale = reference.norm().max(1.0);
                assert!(
                    (actual - reference).norm() <= 1.0e-11 * scale,
                    "CPU Complex64 pair {i} {name} differs: actual={actual:?}, reference={reference:?}"
                );
            }
        }
    }

    #[test]
    fn checked_bem_regular_pair_c64_rejects_non_finite_input() {
        use sonium_compute::CpuBackend;

        let mut pair = BemRegularPairInF64::from_triangle(
            [5.0, 5.0, 5.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        );
        pair.source_point[0] = f64::NAN;
        let error = CpuBackend::new()
            .try_bem_regular_pair_c64_batch(&[pair], 1.0, 1.0)
            .expect_err("non-finite geometry must be rejected");
        assert!(matches!(
            error,
            BackendError::OperationFailed {
                operation: "bem_regular_pair_c64_batch",
                ..
            }
        ));
    }

    #[test]
    fn checked_bem_regular_pair_rejects_degenerate_geometry() {
        use sonium_compute::CpuBackend;

        let pair = BemRegularPairInF64::from_triangle(
            [5.0, 5.0, 5.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        );
        let error = CpuBackend::new()
            .try_bem_regular_pair_c64_batch(&[pair], 1.0, 1.0)
            .expect_err("degenerate geometry must not become a zero-filled success");
        assert!(matches!(
            error,
            BackendError::InvalidArgument {
                operation: "bem_regular_pair_c64_batch",
                ..
            }
        ));
    }

    #[test]
    #[cfg(feature = "metal")]
    fn metal_bem_regular_pair_matches_cpu() {
        use sonium_compute::MetalBackend;

        let backend = match MetalBackend::new() {
            Ok(b) => b,
            Err(err) => skip_or_require_metal!("BEM regular-pair parity", err),
        };

        let pairs_full = fixture_pairs();
        let pairs: Vec<BemRegularPairIn> = pairs_full.iter().map(|(p, ..)| *p).collect();

        // Sweep across a reasonable Helmholtz `kr` range — wave_number
        // chosen so that with characteristic distance ≈ 4-7 we cover
        // `kr ∈ [0.4, 14]`, which spans the typical kD audit cases.
        let cases = [
            (0.1_f64, 1.0_f64), // low frequency, harmonic_factor = +1
            (1.0, 1.0),         // mid
            (2.0, -1.0),        // mid, opposite sign convention
        ];

        for (k, hf) in cases.iter() {
            let gpu_out =
                Backend::bem_regular_pair_c32_batch(&backend, &pairs, *k as f32, *hf as f32);

            assert_eq!(gpu_out.len(), pairs.len());

            for (i, ((_, sp, sn, v0, v1, v2), gpu)) in
                pairs_full.iter().zip(gpu_out.iter()).enumerate()
            {
                let (cg, ch, cht, ce) =
                    cpu_regular_pair_reference(*sp, *sn, *v0, *v1, *v2, *k, *hf);

                // f32 GPU vs f64 CPU on a Helmholtz kernel:
                //   - per-quadrature-point evaluation in f32 → ~1e-7 ULP
                //   - 4 quadrature accumulations → ~5e-7 worst-case
                //   - condition number ~ kr / (4πr)
                // Empirically: well inside 1e-4 across the case sweep.
                let tol = 1e-4_f64;
                let eg = rel_err(cg, gpu.g.re, gpu.g.im);
                let eh = rel_err(ch, gpu.h.re, gpu.h.im);
                let eht = rel_err(cht, gpu.ht.re, gpu.ht.im);
                let ee = rel_err(ce, gpu.e.re, gpu.e.im);
                assert!(
                    eg < tol && eh < tol && eht < tol && ee < tol,
                    "pair {i} (k={k}, hf={hf}) exceeds tol {tol:.0e}: \
                     G={eg:.3e} H={eh:.3e} H^T={eht:.3e} E={ee:.3e}"
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_bem_regular_pair_matches_cpu() {
        use sonium_compute::CudaBackend;

        let backend = match CudaBackend::new_default() {
            Ok(b) => b,
            Err(err) => skip_or_require_cuda!("BEM regular-pair parity", err),
        };
        if !backend.has_bem_regular_pair_c32() {
            if super::env_flag("SONIUM_REQUIRE_CUDA") {
                panic!("CUDA Complex32 BEM regular-pair kernel is unavailable");
            }
            eprintln!("CUDA Complex32 BEM regular-pair kernel unavailable, skipping");
            return;
        }

        let pairs_full = fixture_pairs();
        let pairs: Vec<BemRegularPairIn> = pairs_full.iter().map(|(p, ..)| *p).collect();
        let cases = [(0.1_f64, 1.0_f64), (1.0, 1.0), (2.0, -1.0)];

        for (k, hf) in cases.iter() {
            let gpu_out = backend
                .try_bem_regular_pair_c32_batch(&pairs, *k as f32, *hf as f32)
                .expect("native CUDA Complex32 BEM regular-pair operation must succeed");

            assert_eq!(gpu_out.len(), pairs.len());

            for (i, ((_, sp, sn, v0, v1, v2), gpu)) in
                pairs_full.iter().zip(gpu_out.iter()).enumerate()
            {
                let (cg, ch, cht, ce) =
                    cpu_regular_pair_reference(*sp, *sn, *v0, *v1, *v2, *k, *hf);

                let tol = 1e-4_f64;
                let eg = rel_err(cg, gpu.g.re, gpu.g.im);
                let eh = rel_err(ch, gpu.h.re, gpu.h.im);
                let eht = rel_err(cht, gpu.ht.re, gpu.ht.im);
                let ee = rel_err(ce, gpu.e.re, gpu.e.im);
                assert!(
                    eg < tol && eh < tol && eht < tol && ee < tol,
                    "CUDA pair {i} (k={k}, hf={hf}) exceeds tol {tol:.0e}: \
                     G={eg:.3e} H={eh:.3e} H^T={eht:.3e} E={ee:.3e}"
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_bem_regular_pair_c32_rejects_degenerate_geometry() {
        use sonium_compute::CudaBackend;

        let backend = match CudaBackend::new_default() {
            Ok(b) => b,
            Err(err) => skip_or_require_cuda!("CUDA Complex32 BEM input validation", err),
        };
        if !backend.has_bem_regular_pair_c32() {
            if super::env_flag("SONIUM_REQUIRE_CUDA") {
                panic!("CUDA Complex32 BEM regular-pair kernel is unavailable");
            }
            eprintln!("CUDA Complex32 BEM regular-pair kernel unavailable, skipping");
            return;
        }

        let invalid = BemRegularPairIn::from_triangle(
            [5.0, 5.0, 5.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        );
        let error = backend
            .try_bem_regular_pair_c32_batch(&[invalid], 1.0, 1.0)
            .expect_err("degenerate geometry must fail before the native CUDA launch");
        assert!(matches!(
            error,
            BackendError::InvalidArgument {
                operation: "bem_regular_pair_c32_batch",
                ..
            }
        ));
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_bem_regular_pair_c64_matches_cpu_and_records_device_work() {
        use sonium_compute::CudaBackend;

        let backend = match CudaBackend::new_default() {
            Ok(b) => b,
            Err(err) => skip_or_require_cuda!("CUDA Complex64 BEM regular-pair parity", err),
        };
        if !backend.has_bem_regular_pair_c64() {
            if super::env_flag("SONIUM_REQUIRE_CUDA") {
                panic!("CUDA Complex64 BEM regular-pair kernel is unavailable");
            }
            eprintln!("CUDA Complex64 BEM regular-pair kernel unavailable, skipping");
            return;
        }

        let pairs_full = fixture_pairs();
        let pairs: Vec<BemRegularPairInF64> = pairs_full
            .iter()
            .map(|(_, sp, sn, v0, v1, v2)| {
                BemRegularPairInF64::from_triangle(*sp, *sn, *v0, *v1, *v2)
            })
            .collect();
        backend.reset_runtime_stats();
        let gpu_out = backend
            .try_bem_regular_pair_c64_batch(&pairs, 1.0, -1.0)
            .expect("CUDA Complex64 BEM pair evaluation");
        let stats = backend.runtime_stats();
        assert!(
            stats.kernel_launches >= 1,
            "BEM pair must launch a CUDA kernel"
        );
        assert!(stats.transfer.h2d_operations >= 1);
        assert!(stats.transfer.d2h_operations >= 1);

        for (i, ((_, sp, sn, v0, v1, v2), got)) in pairs_full.iter().zip(gpu_out.iter()).enumerate()
        {
            let expected = cpu_regular_pair_reference(*sp, *sn, *v0, *v1, *v2, 1.0, -1.0);
            for (name, actual, reference) in [
                ("G", got.g, expected.0),
                ("H", got.h, expected.1),
                ("H^T", got.ht, expected.2),
                ("E", got.e, expected.3),
            ] {
                let scale = reference.norm().max(1.0);
                assert!(
                    (actual - reference).norm() <= 2.0e-11 * scale,
                    "CUDA Complex64 pair {i} {name} differs: actual={actual:?}, reference={reference:?}"
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_bem_regular_pair_c64_strided_views_match_cpu_without_staging() {
        use sonium_compute::{
            BemRegularPairOutF64, BufferView, CudaBackend, MemoryPool, MemoryPoolConfig,
        };

        let backend = match CudaBackend::new_default() {
            Ok(b) => b,
            Err(err) => skip_or_require_cuda!("CUDA Complex64 BEM strided views", err),
        };
        if !backend.has_bem_regular_pair_c64() {
            if super::env_flag("SONIUM_REQUIRE_CUDA") {
                panic!("CUDA Complex64 BEM regular-pair kernel is unavailable");
            }
            eprintln!("CUDA Complex64 BEM regular-pair kernel unavailable, skipping");
            return;
        }
        let pool =
            MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 20)).expect("CUDA BEM pool");
        let backend = pool.backend();
        let pairs_full = fixture_pairs();
        let logical_pairs = pairs_full
            .iter()
            .map(|(_, sp, sn, v0, v1, v2)| {
                BemRegularPairInF64::from_triangle(*sp, *sn, *v0, *v1, *v2)
            })
            .collect::<Vec<_>>();
        // Twelve logical records at offset 1 and stride 2 occupy backing
        // elements 1..=23.  Keep the allocation and descriptor bounds
        // consistent; the view validator must reject undersized storage.
        let backing_len = 1 + (logical_pairs.len() - 1) * 2 + 1;
        let pair_view = BufferView::new(backing_len, 1, logical_pairs.len(), 2).unwrap();
        let output_view = BufferView::new(backing_len, 1, logical_pairs.len(), 2).unwrap();
        let mut pair_backing = vec![BemRegularPairInF64::default(); backing_len];
        for (logical, pair) in logical_pairs.iter().enumerate() {
            pair_backing[pair_view.checked_element_index(logical).unwrap()] = *pair;
        }
        let mut pairs = pool
            .try_alloc::<BemRegularPairInF64>(pair_backing.len())
            .unwrap();
        let mut output = pool.try_alloc::<BemRegularPairOutF64>(backing_len).unwrap();
        backend.copy_h2d(&pair_backing, pairs.raw_mut());
        backend.reset_runtime_stats();
        backend
            .try_bem_regular_pair_c64_device_view(
                pairs.raw(),
                pair_view,
                output.raw_mut(),
                output_view,
                1.0,
                -1.0,
            )
            .expect("native CUDA BEM strided pair operation");
        let queued = backend.runtime_stats();
        assert_eq!(queued.kernel_launches, 1);
        assert_eq!(queued.transfer.h2d_operations, 0);
        assert_eq!(queued.transfer.d2h_operations, 0);
        backend.try_synchronize().unwrap();
        let mut output_backing = vec![BemRegularPairOutF64::default(); backing_len];
        backend.copy_d2h(output.raw(), &mut output_backing);
        for (logical, (_, sp, sn, v0, v1, v2)) in pairs_full.iter().enumerate() {
            let expected = cpu_regular_pair_reference(*sp, *sn, *v0, *v1, *v2, 1.0, -1.0);
            let got = output_backing[output_view.checked_element_index(logical).unwrap()];
            for (name, actual, reference) in [
                ("G", got.g, expected.0),
                ("H", got.h, expected.1),
                ("H^T", got.ht, expected.2),
                ("E", got.e, expected.3),
            ] {
                let scale = reference.norm().max(1.0);
                assert!(
                    (actual - reference).norm() <= 2.0e-11 * scale,
                    "CUDA Complex64 strided pair {logical} {name} differs: actual={actual:?}, reference={reference:?}"
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "metal")]
    fn metal_bem_regular_pair_empty_batch_is_noop() {
        use sonium_compute::MetalBackend;

        let backend = match MetalBackend::new() {
            Ok(b) => b,
            Err(err) => skip_or_require_metal!("BEM regular-pair empty test", err),
        };
        let out = backend.bem_regular_pair_c32_batch(&[], 1.0, 1.0);
        assert!(out.is_empty(), "empty input must produce empty output");
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_bem_regular_pair_empty_batch_is_noop() {
        use sonium_compute::CudaBackend;

        let backend = match CudaBackend::new_default() {
            Ok(b) => b,
            Err(err) => skip_or_require_cuda!("BEM regular-pair empty test", err),
        };
        let out = backend.bem_regular_pair_c32_batch(&[], 1.0, 1.0);
        assert!(out.is_empty(), "empty input must produce empty output");
    }
}

#[cfg(feature = "cuda")]
mod cuda_fmm_residency {
    use super::*;
    use num_complex::Complex64;
    use sonium_compute::{
        CudaBackend, CudaDenseBlockC64Input, CudaHMatrixC64, CudaLowRankBlockC64Input, MemoryPool,
        MemoryPoolConfig,
    };

    #[test]
    fn complex64_fmm_primitives_stay_on_device_between_upload_and_readback() {
        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend,
            Err(error) => skip_or_require_cuda!("CUDA Complex64 FMM residency", error),
        };
        if !backend.has_fmm_c64() {
            if super::env_flag("SONIUM_REQUIRE_CUDA") {
                panic!("CUDA Complex64 FMM kernel module is unavailable");
            }
            eprintln!("CUDA Complex64 FMM kernel module unavailable, skipping");
            return;
        }

        let pool =
            MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 20)).expect("CUDA FMM pool");
        let backend = pool.backend();

        let phase_host = vec![
            Complex64::new(2.0, 0.5),
            Complex64::new(-1.0, 0.25),
            Complex64::new(0.5, -2.0),
            Complex64::new(1.25, 1.0),
        ];
        let src_host = vec![
            Complex64::new(1.0, -1.0),
            Complex64::new(2.0, 0.5),
            Complex64::new(-0.5, 3.0),
            Complex64::new(4.0, -2.0),
        ];
        let mut phase = pool.try_alloc::<Complex64>(phase_host.len()).unwrap();
        let mut src = pool.try_alloc::<Complex64>(src_host.len()).unwrap();
        backend.copy_h2d(&phase_host, phase.raw_mut());
        backend.copy_h2d(&src_host, src.raw_mut());

        let mut phase_dst = pool.try_alloc::<Complex64>(4).unwrap();
        let mut gather_dst = pool.try_alloc::<Complex64>(3).unwrap();
        let mut scatter_dst = pool.try_alloc::<Complex64>(4).unwrap();
        let mut dense_dst = pool.try_alloc::<Complex64>(5).unwrap();
        let mut scatter_dense_dst = pool.try_alloc::<Complex64>(5).unwrap();
        let matrix_host = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(99.0, 0.0),
            Complex64::new(-1.0, 0.5),
            Complex64::new(0.5, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(77.0, 0.0),
        ];
        // Repeated/non-monotone source indices are valid for an indexed
        // translation and must not be treated as a contiguous view.
        let src_indices_host = vec![3_i32, 1, 1];
        let dst_indices_host = vec![2_i32, 4];
        let gather_indices_host = vec![2_i32, 0, 3];
        let mut matrix = pool.try_alloc::<Complex64>(matrix_host.len()).unwrap();
        let mut src_indices = pool.try_alloc::<i32>(src_indices_host.len()).unwrap();
        let mut dst_indices = pool.try_alloc::<i32>(dst_indices_host.len()).unwrap();
        let mut gather_indices = pool.try_alloc::<i32>(gather_indices_host.len()).unwrap();
        backend.copy_h2d(&matrix_host, matrix.raw_mut());
        backend.copy_h2d(&src_indices_host, src_indices.raw_mut());
        backend.copy_h2d(&dst_indices_host, dst_indices.raw_mut());
        backend.copy_h2d(&gather_indices_host, gather_indices.raw_mut());

        // No operation below is allowed to stage its vectors through the
        // host.  Publication readback happens only after all five kernels are
        // queued on the same stream.
        backend.reset_runtime_stats();
        backend
            .try_fmm_phase_mul_accumulate_c64(phase_dst.raw_mut(), phase.raw(), src.raw())
            .unwrap();
        backend
            .try_fmm_compact_indexed_gather_c64(
                src.raw(),
                gather_indices.raw(),
                gather_dst.raw_mut(),
            )
            .unwrap();
        backend
            .try_fmm_compact_indexed_scatter_accumulate_c64(
                gather_dst.raw(),
                gather_indices.raw(),
                scatter_dst.raw_mut(),
            )
            .unwrap();
        backend
            .try_fmm_indexed_dense_matvec_accumulate_c64(
                matrix.raw(),
                2,
                3,
                4,
                src.raw(),
                src_indices.raw(),
                dense_dst.raw_mut(),
                dst_indices.raw(),
            )
            .unwrap();
        backend
            .try_fmm_dense_matvec_scatter_accumulate_c64(
                matrix.raw(),
                2,
                3,
                4,
                src.raw(),
                scatter_dense_dst.raw_mut(),
                dst_indices.raw(),
            )
            .unwrap();

        let queued = backend.runtime_stats();
        assert_eq!(queued.transfer.h2d_operations, 0);
        assert_eq!(queued.transfer.d2h_operations, 0);
        assert_eq!(queued.kernel_launches, 5);

        let mut phase_out = vec![Complex64::new(0.0, 0.0); 4];
        let mut gather_out = vec![Complex64::new(0.0, 0.0); 3];
        let mut scatter_out = vec![Complex64::new(0.0, 0.0); 4];
        let mut dense_out = vec![Complex64::new(0.0, 0.0); 5];
        let mut scatter_dense_out = vec![Complex64::new(0.0, 0.0); 5];
        backend.copy_d2h(phase_dst.raw(), &mut phase_out);
        backend.copy_d2h(gather_dst.raw(), &mut gather_out);
        backend.copy_d2h(scatter_dst.raw(), &mut scatter_out);
        backend.copy_d2h(dense_dst.raw(), &mut dense_out);
        backend.copy_d2h(scatter_dense_dst.raw(), &mut scatter_dense_out);

        assert_eq!(
            phase_out,
            phase_host
                .iter()
                .zip(src_host.iter())
                .map(|(phase, src)| *phase * *src)
                .collect::<Vec<_>>()
        );
        assert_eq!(gather_out, vec![src_host[2], src_host[0], src_host[3]]);
        assert_eq!(
            scatter_out,
            vec![
                src_host[0],
                Complex64::new(0.0, 0.0),
                src_host[2],
                src_host[3]
            ]
        );
        assert_eq!(
            dense_out,
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                matrix_host[0] * src_host[3]
                    + matrix_host[1] * src_host[1]
                    + matrix_host[2] * src_host[1],
                Complex64::new(0.0, 0.0),
                matrix_host[4] * src_host[3]
                    + matrix_host[5] * src_host[1]
                    + matrix_host[6] * src_host[1],
            ]
        );
        assert_eq!(
            scatter_dense_out,
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                matrix_host[0] * src_host[0]
                    + matrix_host[1] * src_host[1]
                    + matrix_host[2] * src_host[2],
                Complex64::new(0.0, 0.0),
                matrix_host[4] * src_host[0]
                    + matrix_host[5] * src_host[1]
                    + matrix_host[6] * src_host[2],
            ]
        );
    }

    #[test]
    fn complex64_fmm_compact_shapes_are_rejected_before_dispatch() {
        use sonium_compute::BackendError;

        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend,
            Err(error) => skip_or_require_cuda!("CUDA FMM shape validation", error),
        };
        if !backend.has_fmm_c64() {
            if super::env_flag("SONIUM_REQUIRE_CUDA") {
                panic!("CUDA Complex64 FMM kernel module is unavailable");
            }
            eprintln!("CUDA Complex64 FMM kernel module unavailable, skipping");
            return;
        }

        let pool =
            MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 20)).expect("CUDA FMM pool");
        let src = pool.try_alloc::<Complex64>(2).unwrap();
        let indices = pool.try_alloc::<i32>(1).unwrap();
        let mut dst = pool.try_alloc::<Complex64>(2).unwrap();
        let error = pool
            .backend()
            .try_fmm_compact_indexed_gather_c64(src.raw(), indices.raw(), dst.raw_mut())
            .expect_err("mismatched compact gather shapes must be typed errors");
        assert!(matches!(
            error,
            BackendError::InvalidArgument {
                operation: "fmm_compact_indexed_gather_c64",
                ..
            }
        ));
    }

    #[test]
    fn complex64_fmm_strided_views_match_cpu_and_keep_residency() {
        use sonium_compute::{BufferView, CudaBackend, MemoryPool, MemoryPoolConfig};

        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend,
            Err(error) => skip_or_require_cuda!("CUDA FMM strided views", error),
        };
        if !backend.has_fmm_c64() {
            if super::env_flag("SONIUM_REQUIRE_CUDA") {
                panic!("CUDA Complex64 FMM kernel module is unavailable");
            }
            eprintln!("CUDA Complex64 FMM kernel module unavailable, skipping");
            return;
        }

        let pool =
            MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 20)).expect("CUDA FMM pool");
        let backend = pool.backend();
        let zero = Complex64::new(0.0, 0.0);
        let phase_host = (0..8)
            .map(|index| Complex64::new(index as f64 + 1.0, 0.25 * index as f64))
            .collect::<Vec<_>>();
        let src_host = (0..8)
            .map(|index| Complex64::new(0.5 + index as f64, -0.125 * index as f64))
            .collect::<Vec<_>>();
        let mut phase = pool.try_alloc::<Complex64>(8).unwrap();
        let mut src = pool.try_alloc::<Complex64>(8).unwrap();
        let mut phase_dst = pool.try_alloc::<Complex64>(8).unwrap();
        let mut gather_dst = pool.try_alloc::<Complex64>(8).unwrap();
        let mut scatter_dst = pool.try_alloc::<Complex64>(8).unwrap();
        let mut dense_dst = pool.try_alloc::<Complex64>(8).unwrap();
        let mut scatter_dense_dst = pool.try_alloc::<Complex64>(8).unwrap();
        backend.copy_h2d(&phase_host, phase.raw_mut());
        backend.copy_h2d(&src_host, src.raw_mut());
        backend.copy_h2d(&vec![zero; 8], phase_dst.raw_mut());
        backend.copy_h2d(&vec![zero; 8], gather_dst.raw_mut());
        backend.copy_h2d(&vec![zero; 8], scatter_dst.raw_mut());
        backend.copy_h2d(&vec![zero; 8], dense_dst.raw_mut());
        backend.copy_h2d(&vec![zero; 8], scatter_dense_dst.raw_mut());

        let values = BufferView::new(8, 1, 3, 2).unwrap();
        let indices_view = BufferView::new(8, 1, 3, 2).unwrap();
        let dense_indices_view = BufferView::new(8, 1, 2, 2).unwrap();
        let mut gather_indices = pool.try_alloc::<i32>(8).unwrap();
        let mut scatter_indices = pool.try_alloc::<i32>(8).unwrap();
        let mut dense_src_indices = pool.try_alloc::<i32>(8).unwrap();
        let mut dense_dst_indices = pool.try_alloc::<i32>(8).unwrap();
        let mut index_host = vec![0_i32; 8];
        index_host[1] = 2;
        index_host[3] = 0;
        index_host[5] = 1;
        backend.copy_h2d(&index_host, gather_indices.raw_mut());
        // Scatter consumes compact entries in source order and places them at
        // the supplied destination indices.  Use the inverse permutation of
        // the gather map here so this independent phase produces the
        // deliberately non-identity reference [v1, v2, v0].
        let mut scatter_index_host = vec![0_i32; 8];
        scatter_index_host[1] = 1;
        scatter_index_host[3] = 2;
        scatter_index_host[5] = 0;
        backend.copy_h2d(&scatter_index_host, scatter_indices.raw_mut());
        let mut src_index_host = vec![0_i32; 8];
        src_index_host[1] = 2;
        src_index_host[3] = 0;
        backend.copy_h2d(&src_index_host, dense_src_indices.raw_mut());
        let mut dst_index_host = vec![0_i32; 8];
        dst_index_host[1] = 0;
        dst_index_host[3] = 2;
        backend.copy_h2d(&dst_index_host, dense_dst_indices.raw_mut());

        let mut matrix_host = vec![zero; 12];
        matrix_host[2] = Complex64::new(1.0, 0.5);
        matrix_host[3] = Complex64::new(2.0, -0.5);
        matrix_host[5] = Complex64::new(-1.0, 0.25);
        matrix_host[6] = Complex64::new(0.75, 1.0);
        let mut matrix = pool.try_alloc::<Complex64>(matrix_host.len()).unwrap();
        backend.copy_h2d(&matrix_host, matrix.raw_mut());

        // Every specialized operation below binds non-contiguous views.  No
        // host transfer is permitted until all five native launches are
        // queued, proving that the descriptor path is device-resident.
        backend.reset_runtime_stats();
        backend
            .try_fmm_phase_mul_accumulate_c64_view(
                phase_dst.raw_mut(),
                values,
                phase.raw(),
                values,
                src.raw(),
                values,
            )
            .unwrap();
        backend
            .try_fmm_compact_indexed_gather_c64_view(
                src.raw(),
                values,
                gather_indices.raw(),
                indices_view,
                gather_dst.raw_mut(),
                values,
            )
            .unwrap();
        backend
            .try_fmm_compact_indexed_scatter_accumulate_c64_view(
                gather_dst.raw(),
                values,
                scatter_indices.raw(),
                indices_view,
                scatter_dst.raw_mut(),
                values,
            )
            .unwrap();
        backend
            .try_fmm_indexed_dense_matvec_accumulate_c64_view(
                matrix.raw(),
                2,
                2,
                2,
                3,
                src.raw(),
                values,
                dense_src_indices.raw(),
                dense_indices_view,
                dense_dst.raw_mut(),
                values,
                dense_dst_indices.raw(),
                dense_indices_view,
            )
            .unwrap();
        backend
            .try_fmm_dense_matvec_scatter_accumulate_c64_view(
                matrix.raw(),
                2,
                2,
                2,
                3,
                src.raw(),
                values,
                scatter_dense_dst.raw_mut(),
                values,
                dense_dst_indices.raw(),
                dense_indices_view,
            )
            .unwrap();
        let queued = backend.runtime_stats();
        assert_eq!(queued.kernel_launches, 5);
        assert_eq!(queued.transfer.h2d_operations, 0);
        assert_eq!(queued.transfer.d2h_operations, 0);

        backend.try_synchronize().unwrap();
        let mut phase_out = vec![zero; 8];
        let mut gather_out = vec![zero; 8];
        let mut scatter_out = vec![zero; 8];
        let mut dense_out = vec![zero; 8];
        let mut scatter_dense_out = vec![zero; 8];
        backend.copy_d2h(phase_dst.raw(), &mut phase_out);
        backend.copy_d2h(gather_dst.raw(), &mut gather_out);
        backend.copy_d2h(scatter_dst.raw(), &mut scatter_out);
        backend.copy_d2h(dense_dst.raw(), &mut dense_out);
        backend.copy_d2h(scatter_dense_dst.raw(), &mut scatter_dense_out);

        let logical_values = [src_host[1], src_host[3], src_host[5]];
        for logical in 0..3 {
            let physical = values.checked_element_index(logical).unwrap();
            assert_eq!(
                phase_out[physical],
                phase_host[physical] * src_host[physical],
                "strided phase view element {logical}"
            );
            assert_eq!(
                gather_out[physical],
                logical_values[index_host[1 + logical * 2] as usize],
                "strided gather view element {logical}"
            );
        }
        let expected_scatter = [logical_values[1], logical_values[2], logical_values[0]];
        for logical in 0..3 {
            let physical = values.checked_element_index(logical).unwrap();
            assert_eq!(scatter_out[physical], expected_scatter[logical]);
        }

        let expected_row0 = matrix_host[2] * logical_values[2] + matrix_host[3] * logical_values[0];
        let expected_row1 = matrix_host[5] * logical_values[2] + matrix_host[6] * logical_values[0];
        assert_eq!(dense_out[1], expected_row0);
        assert_eq!(dense_out[5], expected_row1);
        let expected_dense_scatter_row0 =
            matrix_host[2] * logical_values[0] + matrix_host[3] * logical_values[1];
        let expected_dense_scatter_row1 =
            matrix_host[5] * logical_values[0] + matrix_host[6] * logical_values[1];
        assert_eq!(scatter_dense_out[1], expected_dense_scatter_row0);
        assert_eq!(scatter_dense_out[5], expected_dense_scatter_row1);
    }

    #[test]
    fn complex64_mlfmm_resident_forward_transpose_and_hermitian_parity() {
        use sonium_compute::{
            CudaBackend, CudaMlfmmC64, CudaMlfmmLevelC64Input, CudaMlfmmTransferC64Input,
            CudaMlfmmTranslationC64Input, MemoryPool, MemoryPoolConfig,
        };

        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend,
            Err(error) => skip_or_require_cuda!("CUDA Complex64 MLFMM residency", error),
        };
        if !backend.has_fmm_c64() {
            if super::env_flag("SONIUM_REQUIRE_CUDA") {
                panic!("CUDA Complex64 FMM kernel module is unavailable");
            }
            eprintln!("CUDA Complex64 FMM kernel module unavailable, skipping");
            return;
        }

        fn c(re: f64, im: f64) -> Complex64 {
            Complex64::new(re, im)
        }
        fn matrix_action(
            matrix: &[Complex64],
            rows: usize,
            cols: usize,
            input: &[Complex64],
            transpose: bool,
            conjugate: bool,
        ) -> Vec<Complex64> {
            if transpose {
                (0..cols)
                    .map(|column| {
                        (0..rows)
                            .map(|row| {
                                let mut value = matrix[row * cols + column];
                                if conjugate {
                                    value = value.conj();
                                }
                                value * input[row]
                            })
                            .sum()
                    })
                    .collect()
            } else {
                (0..rows)
                    .map(|row| {
                        (0..cols)
                            .map(|column| {
                                let mut value = matrix[row * cols + column];
                                if conjugate {
                                    value = value.conj();
                                }
                                value * input[column]
                            })
                            .sum()
                    })
                    .collect()
            }
        }
        fn phase_action(
            input: &[Complex64],
            phase: &[Complex64],
            output_len: usize,
            conjugate: bool,
        ) -> Vec<Complex64> {
            let mut output = vec![Complex64::new(0.0, 0.0); output_len];
            let count = phase.len().min(input.len()).min(output_len);
            for point in 0..count {
                let coefficient = if conjugate {
                    phase[point].conj()
                } else {
                    phase[point]
                };
                output[point] = coefficient * input[point];
            }
            output
        }
        #[allow(clippy::too_many_arguments)]
        fn expected(
            input: &[Complex64],
            t_leaf: &[Complex64],
            s_leaf: &[Complex64],
            t_phase: &[Complex64],
            s_phase: &[Complex64],
            diagonal: &[Complex64],
            transpose: bool,
            conjugate: bool,
        ) -> Vec<Complex64> {
            if transpose {
                let leaf = matrix_action(s_leaf, 2, 3, input, true, conjugate);
                let root = phase_action(&leaf, s_phase, 2, conjugate);
                let translated = root
                    .iter()
                    .zip(diagonal)
                    .map(|(value, diagonal)| {
                        value
                            * if conjugate {
                                diagonal.conj()
                            } else {
                                *diagonal
                            }
                    })
                    .collect::<Vec<_>>();
                let child = phase_action(&translated, t_phase, 3, conjugate);
                matrix_action(t_leaf, 3, 2, &child, true, conjugate)
            } else {
                let leaf = matrix_action(t_leaf, 3, 2, input, false, conjugate);
                let root = phase_action(&leaf, t_phase, 2, conjugate);
                let translated = root
                    .iter()
                    .zip(diagonal)
                    .map(|(value, diagonal)| {
                        value
                            * if conjugate {
                                diagonal.conj()
                            } else {
                                *diagonal
                            }
                    })
                    .collect::<Vec<_>>();
                let child = phase_action(&translated, s_phase, 3, conjugate);
                matrix_action(s_leaf, 2, 3, &child, false, conjugate)
            }
        }

        let t_leaf = vec![
            c(1.0, 0.0),
            c(2.0, 1.0),
            c(-1.0, 0.5),
            c(0.25, -1.0),
            c(0.5, 2.0),
            c(-0.75, 0.2),
        ];
        let t_root = vec![
            c(0.5, -0.25),
            c(1.0, 0.5),
            c(-0.2, 0.3),
            c(1.5, 0.1),
            c(-0.4, 0.75),
            c(0.8, -0.6),
        ];
        let s_root = vec![
            c(1.0, 0.0),
            c(0.2, -0.5),
            c(-0.5, 0.75),
            c(1.3, 0.2),
            c(0.1, -0.4),
            c(-0.8, 0.6),
        ];
        let s_leaf = vec![
            c(0.7, 0.1),
            c(-0.3, 0.5),
            c(1.2, -0.2),
            c(-0.4, 0.8),
            c(0.9, -0.6),
            c(0.2, 0.3),
        ];
        let diagonal = vec![c(1.2, 0.3), c(-0.7, 0.5)];
        let t_phase = vec![c(0.75, -0.1), c(-1.2, 0.4)];
        let s_phase = vec![c(1.1, 0.3), c(-0.4, -0.2)];
        let input_host = vec![c(0.2, -0.3), c(-0.4, 0.7)];
        let level =
            |t_matrix, t_rows, t_cols, s_matrix, s_rows, s_cols, points| CudaMlfmmLevelC64Input {
                sphere_points: points,
                t_transfers: vec![CudaMlfmmTransferC64Input {
                    matrix: t_matrix,
                    src_indices: (0..t_cols).collect(),
                    dst_indices: (0..t_rows).collect(),
                }],
                s_transfers: vec![CudaMlfmmTransferC64Input {
                    matrix: s_matrix,
                    src_indices: (0..s_cols).collect(),
                    dst_indices: (0..s_rows).collect(),
                }],
                t_phase_transfers: Vec::new(),
                s_phase_transfers: Vec::new(),
                translations: Vec::new(),
            };
        let mut levels = vec![
            level(t_root.clone(), 2, 3, s_root.clone(), 3, 2, 2),
            level(t_leaf.clone(), 3, 2, s_leaf.clone(), 2, 3, 3),
        ];
        levels[0].translations.push(CudaMlfmmTranslationC64Input {
            source_cluster: 0,
            field_cluster: 0,
            diagonal: diagonal.clone(),
        });
        levels[0]
            .t_phase_transfers
            .push(sonium_compute::CudaMlfmmPhaseTransferC64Input {
                source_cluster: 0,
                destination_cluster: 0,
                phase: t_phase.clone(),
            });
        levels[0]
            .s_phase_transfers
            .push(sonium_compute::CudaMlfmmPhaseTransferC64Input {
                source_cluster: 0,
                destination_cluster: 0,
                phase: s_phase.clone(),
            });

        let pool =
            MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 20)).expect("CUDA MLFMM pool");
        let operator = CudaMlfmmC64::new(pool.clone(), 2, levels, Vec::new())
            .expect("valid resident MLFMM hierarchy");
        let backend = pool.backend();
        let mut input = pool.try_alloc::<Complex64>(2).unwrap();
        let mut output = pool.try_alloc::<Complex64>(2).unwrap();
        backend.copy_h2d(&input_host, input.raw_mut());

        for (transpose, expected_output) in [
            (
                Transpose::NoTrans,
                expected(
                    &input_host,
                    &t_leaf,
                    &s_leaf,
                    &t_phase,
                    &s_phase,
                    &diagonal,
                    false,
                    false,
                ),
            ),
            (
                Transpose::Trans,
                expected(
                    &input_host,
                    &t_leaf,
                    &s_leaf,
                    &t_phase,
                    &s_phase,
                    &diagonal,
                    true,
                    false,
                ),
            ),
            (
                Transpose::ConjTrans,
                expected(
                    &input_host,
                    &t_leaf,
                    &s_leaf,
                    &t_phase,
                    &s_phase,
                    &diagonal,
                    true,
                    true,
                ),
            ),
        ] {
            // Warm each orientation, then measure reuse with counter deltas;
            // resetting runtime stats does not reset lifetime allocation counts.
            operator
                .try_apply_device(&input, &mut output, transpose)
                .unwrap();
            for raw in [false, true] {
                let before = backend.begin_runtime_stats();
                if raw {
                    operator
                        .try_apply_raw(input.raw(), output.raw_mut(), transpose)
                        .unwrap();
                } else {
                    operator
                        .try_apply_device(&input, &mut output, transpose)
                        .expect("resident MLFMM action");
                }
                let stats = backend.runtime_stats_since(before);
                assert_eq!(stats.allocation_operations, 0);
                assert_eq!(stats.transfer.h2d_operations, 0);
                assert_eq!(stats.transfer.d2h_operations, 0);
                assert!(stats.kernel_launches > 0);
                let mut actual = vec![Complex64::new(0.0, 0.0); 2];
                backend.copy_d2h(output.raw(), &mut actual);
                for (actual, expected) in actual.iter().zip(&expected_output) {
                    assert!(
                        (*actual - *expected).norm() < 1.0e-11,
                        "{actual:?} != {expected:?}"
                    );
                }
            }
        }
        let foreign_pool = MemoryPool::new(
            CudaBackend::new_default().unwrap(),
            MemoryPoolConfig::with_cap(1 << 20),
        )
        .unwrap();
        let foreign = foreign_pool.try_alloc::<Complex64>(2).unwrap();
        let mut foreign_output = foreign_pool.try_alloc::<Complex64>(2).unwrap();
        let mut short = pool.try_alloc::<Complex64>(1).unwrap();
        let block_input = pool.try_alloc::<Complex64>(4).unwrap();
        let mut block_output = pool.try_alloc::<Complex64>(4).unwrap();
        let before = backend.begin_runtime_stats();
        for action in [Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
            assert!(
                operator
                    .try_apply_raw(foreign.raw(), output.raw_mut(), action)
                    .is_err()
            );
            assert!(
                operator
                    .try_apply_raw(input.raw(), foreign_output.raw_mut(), action)
                    .is_err()
            );
            assert!(
                operator
                    .try_apply_raw(input.raw(), short.raw_mut(), action)
                    .is_err()
            );
        }
        assert!(
            sonium_compute::DeviceOperator::try_apply(
                &operator,
                foreign_pool.backend(),
                input.raw(),
                output.raw_mut()
            )
            .is_err()
        );
        assert!(
            sonium_compute::DeviceOperator::try_residual(
                &operator,
                backend,
                input.raw(),
                foreign.raw(),
                output.raw_mut()
            )
            .is_err()
        );
        assert!(matches!(
            sonium_compute::DeviceOperator::try_apply_colmajor(
                &operator,
                backend,
                2,
                block_input.raw(),
                block_output.raw_mut()
            ),
            Err(sonium_compute::BackendError::Unsupported { .. })
        ));
        let stats = backend.runtime_stats_since(before);
        assert_eq!(
            stats.kernel_launches + stats.library_operations + stats.allocation_operations,
            0
        );
        assert_eq!(
            stats.transfer.h2d_bytes + stats.transfer.d2h_bytes + stats.transfer.d2d_bytes,
            0
        );
        sonium_compute::DeviceOperator::try_apply(&operator, backend, input.raw(), output.raw_mut())
            .unwrap();
        let sentinel = vec![Complex64::new(12.0, -3.0); 2];
        backend.try_copy_h2d(&sentinel, output.raw_mut()).unwrap();
        backend.inject_failure(sonium_compute::CudaFailureInjectionPoint::Launch);
        assert!(
            operator
                .try_apply_raw(input.raw(), output.raw_mut(), Transpose::ConjTrans)
                .is_err()
        );
        backend.recover_after_error().unwrap();
        let mut preserved = vec![Complex64::default(); 2];
        backend.try_copy_d2h(output.raw(), &mut preserved).unwrap();
        assert_eq!(preserved, sentinel);
        backend.inject_failure(sonium_compute::CudaFailureInjectionPoint::Synchronization);
        assert!(
            operator
                .try_apply_raw(input.raw(), output.raw_mut(), Transpose::NoTrans)
                .is_err()
        );
        assert!(
            backend.recover_after_error().is_err(),
            "synchronization failure requires a new context"
        );
    }

    #[test]
    fn complex64_hmatrix_forward_transpose_and_hermitian_reuse_device_blocks() {
        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend,
            Err(error) => skip_or_require_cuda!("CUDA Complex64 H-matrix residency", error),
        };
        if !backend.has_fmm_c64() {
            if super::env_flag("SONIUM_REQUIRE_CUDA") {
                panic!("CUDA Complex64 FMM kernel module is unavailable");
            }
            eprintln!("CUDA Complex64 FMM kernel module unavailable, skipping");
            return;
        }

        let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 20))
            .expect("CUDA H-matrix pool");
        let hmatrix = CudaHMatrixC64::new(
            pool.clone(),
            4,
            vec![CudaDenseBlockC64Input {
                row_indices: vec![0, 2],
                col_indices: vec![1, 3],
                matrix: vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(2.0, 1.0),
                    Complex64::new(-1.0, 0.5),
                    Complex64::new(0.25, -2.0),
                ],
            }],
            vec![CudaLowRankBlockC64Input {
                row_indices: vec![1, 3],
                col_indices: vec![0, 2],
                u: vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)],
                v: vec![Complex64::new(1.0, -1.0), Complex64::new(0.5, 2.0)],
                rank: 1,
            }],
        )
        .expect("valid device H-matrix upload");

        let x_host = vec![
            Complex64::new(0.5, -1.0),
            Complex64::new(1.0, 2.0),
            Complex64::new(-2.0, 0.25),
            Complex64::new(0.75, -0.5),
        ];
        let mut x = pool.try_alloc::<Complex64>(4).unwrap();
        let mut forward = pool.try_alloc::<Complex64>(4).unwrap();
        let mut transpose = pool.try_alloc::<Complex64>(4).unwrap();
        let mut hermitian = pool.try_alloc::<Complex64>(4).unwrap();
        pool.backend().copy_h2d(&x_host, x.raw_mut());

        pool.backend().reset_runtime_stats();
        hmatrix.try_apply(&x, &mut forward).unwrap();
        hmatrix.try_apply_transpose(&x, &mut transpose).unwrap();
        hmatrix.try_apply_hermitian(&x, &mut hermitian).unwrap();
        let queued = pool.backend().runtime_stats();
        assert_eq!(queued.transfer.h2d_operations, 0);
        assert_eq!(queued.transfer.d2h_operations, 0);
        assert_eq!(queued.kernel_launches, 9);

        let mut forward_out = vec![Complex64::new(0.0, 0.0); 4];
        let mut transpose_out = vec![Complex64::new(0.0, 0.0); 4];
        let mut hermitian_out = vec![Complex64::new(0.0, 0.0); 4];
        pool.backend().copy_d2h(forward.raw(), &mut forward_out);
        pool.backend().copy_d2h(transpose.raw(), &mut transpose_out);
        pool.backend().copy_d2h(hermitian.raw(), &mut hermitian_out);

        let dense = [
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 1.0),
            Complex64::new(-1.0, 0.5),
            Complex64::new(0.25, -2.0),
        ];
        let u = [Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)];
        let v = [Complex64::new(1.0, -1.0), Complex64::new(0.5, 2.0)];
        let projection = v[0] * x_host[0] + v[1] * x_host[2];
        let low_forward = [u[0] * projection, u[1] * projection];
        let mut expected_forward = vec![Complex64::new(0.0, 0.0); 4];
        expected_forward[0] = dense[0] * x_host[1] + dense[1] * x_host[3];
        expected_forward[2] = dense[2] * x_host[1] + dense[3] * x_host[3];
        expected_forward[1] = low_forward[0];
        expected_forward[3] = low_forward[1];

        // Krylov consumes raw buffers through DeviceOperator, not the pooled API.
        let before = pool.backend().begin_runtime_stats();
        sonium_compute::DeviceOperator::try_apply(
            &hmatrix,
            pool.backend(),
            x.raw(),
            forward.raw_mut(),
        )
        .unwrap();
        sonium_compute::DeviceOperator::try_residual(
            &hmatrix,
            pool.backend(),
            x.raw(),
            forward.raw(),
            transpose.raw_mut(),
        )
        .unwrap();
        let stats = pool.backend().runtime_stats_since(before);
        assert_eq!(stats.allocation_operations, 0);
        assert_eq!(stats.transfer.h2d_bytes, 0);
        assert_eq!(stats.transfer.d2h_bytes, 0);
        let mut residual = vec![Complex64::new(0.0, 0.0); 4];
        pool.backend()
            .try_copy_d2h(transpose.raw(), &mut residual)
            .unwrap();
        assert!(residual.iter().all(|v| v.norm() < 1e-12));
        let other_backend = CudaBackend::new_default().unwrap();
        assert!(
            sonium_compute::DeviceOperator::try_apply(
                &hmatrix,
                &other_backend,
                x.raw(),
                forward.raw_mut(),
            )
            .is_err()
        );
        let short = pool.try_alloc::<Complex64>(3).unwrap();
        assert!(
            sonium_compute::DeviceOperator::try_residual(
                &hmatrix,
                pool.backend(),
                x.raw(),
                short.raw(),
                transpose.raw_mut(),
            )
            .is_err()
        );
        pool.backend()
            .inject_failure(sonium_compute::CudaFailureInjectionPoint::Launch);
        assert!(
            sonium_compute::DeviceOperator::try_residual(
                &hmatrix,
                pool.backend(),
                x.raw(),
                forward.raw(),
                transpose.raw_mut(),
            )
            .is_err()
        );
        pool.backend().recover_after_error().unwrap();
        pool.backend()
            .try_copy_d2h(transpose.raw(), &mut residual)
            .unwrap();
        assert!(residual.iter().all(|v| v.norm() < 1e-12));

        let transpose_projection = u[0] * x_host[1] + u[1] * x_host[3];
        let mut expected_transpose = vec![Complex64::new(0.0, 0.0); 4];
        expected_transpose[1] = dense[0] * x_host[0] + dense[2] * x_host[2];
        expected_transpose[3] = dense[1] * x_host[0] + dense[3] * x_host[2];
        expected_transpose[0] = v[0] * transpose_projection;
        expected_transpose[2] = v[1] * transpose_projection;

        let hermitian_projection = u[0].conj() * x_host[1] + u[1].conj() * x_host[3];
        let mut expected_hermitian = vec![Complex64::new(0.0, 0.0); 4];
        expected_hermitian[1] = dense[0].conj() * x_host[0] + dense[2].conj() * x_host[2];
        expected_hermitian[3] = dense[1].conj() * x_host[0] + dense[3].conj() * x_host[2];
        expected_hermitian[0] = v[0].conj() * hermitian_projection;
        expected_hermitian[2] = v[1].conj() * hermitian_projection;

        // Raw transpose/Hermitian adapters are the retained Krylov boundary.
        let before = pool.backend().begin_runtime_stats();
        hmatrix.try_apply_transpose_raw(x.raw(), transpose.raw_mut()).unwrap();
        hmatrix.try_apply_hermitian_raw(x.raw(), hermitian.raw_mut()).unwrap();
        let stats = pool.backend().runtime_stats_since(before);
        assert_eq!(stats.allocation_operations, 0);
        assert_eq!(stats.transfer.h2d_bytes + stats.transfer.d2h_bytes, 0);
        let mut raw_transpose = vec![Complex64::default(); 4];
        let mut raw_hermitian = vec![Complex64::default(); 4];
        pool.backend().try_copy_d2h(transpose.raw(), &mut raw_transpose).unwrap();
        pool.backend().try_copy_d2h(hermitian.raw(), &mut raw_hermitian).unwrap();
        for (actual, expected) in [(&raw_transpose, &expected_transpose), (&raw_hermitian, &expected_hermitian)] {
            assert!(actual.iter().zip(expected).all(|(a,b)| (*a-*b).norm() < 1e-12));
        }
        let foreign_pool = MemoryPool::new(other_backend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let mut foreign = foreign_pool.try_alloc::<Complex64>(4).unwrap();
        let before = pool.backend().begin_runtime_stats();
        assert!(hmatrix.try_apply_transpose_raw(foreign.raw(), transpose.raw_mut()).is_err());
        assert!(hmatrix.try_apply_hermitian_raw(x.raw(), foreign.raw_mut()).is_err());
        assert!(hmatrix.try_apply_transpose_raw(short.raw(), transpose.raw_mut()).is_err());
        let stats = pool.backend().runtime_stats_since(before);
        assert_eq!(stats.kernel_launches + stats.allocation_operations, 0);
        assert_eq!(stats.transfer.h2d_bytes + stats.transfer.d2h_bytes, 0);

        let forward_inner_product: Complex64 = x_host
            .iter()
            .zip(&forward_out)
            .map(|(x, value)| x.conj() * value)
            .sum();
        let adjoint_inner_product: Complex64 = hermitian_out
            .iter()
            .zip(&x_host)
            .map(|(value, x)| value.conj() * x)
            .sum();
        assert!(
            (forward_inner_product - adjoint_inner_product).norm() <= 1.0e-12,
            "device H-matrix Hermitian action violates the adjoint identity: forward={forward_inner_product:?}, adjoint={adjoint_inner_product:?}"
        );

        for (actual, expected) in [
            (forward_out, expected_forward),
            (transpose_out, expected_transpose),
            (hermitian_out, expected_hermitian),
        ] {
            for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (*actual - *expected).norm() <= 1.0e-12,
                    "H-matrix entry {index} differs: actual={actual:?}, expected={expected:?}"
                );
            }
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda_field_residency {
    use super::*;
    use sonium_compute::{
        Backend, BackendError, BemFieldElementC64, BemFieldPointC64, CudaBackend,
        CudaBemFieldOperatorC64, CudaFailureInjectionPoint, MemoryPool, MemoryPoolConfig,
    };

    fn reference_field(
        point: [f64; 3],
        element: BemFieldElementC64,
        wave_number: f64,
        harmonic_factor: f64,
        pressure_factor: f64,
    ) -> Complex64 {
        let e1 = [
            element.v1[0] - element.v0[0],
            element.v1[1] - element.v0[1],
            element.v1[2] - element.v0[2],
        ];
        let e2 = [
            element.v2[0] - element.v0[0],
            element.v2[1] - element.v0[1],
            element.v2[2] - element.v0[2],
        ];
        let normal = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let jacobian = normal.iter().map(|value| value * value).sum::<f64>().sqrt();
        let normal = normal.map(|value| value / jacobian);
        let xis = [
            0.333333333333333,
            0.797426985353087,
            0.101286507323456,
            0.101286507323456,
            0.470142064105115,
            0.059715871789770,
            0.470142064105115,
        ];
        let etas = [
            0.333333333333333,
            0.101286507323456,
            0.797426985353087,
            0.101286507323456,
            0.059715871789770,
            0.470142064105115,
            0.470142064105115,
        ];
        let weights = [
            0.1125,
            0.0629695902724135,
            0.0629695902724135,
            0.0629695902724135,
            0.066197076394253,
            0.066197076394253,
            0.066197076394253,
        ];
        let wavruim = wave_number * harmonic_factor;
        let mut result = Complex64::new(0.0, 0.0);
        for ((&xi, &eta), &weight) in xis.iter().zip(&etas).zip(&weights) {
            let n0 = 1.0 - xi - eta;
            let y = [
                n0 * element.v0[0] + xi * element.v1[0] + eta * element.v2[0],
                n0 * element.v0[1] + xi * element.v1[1] + eta * element.v2[1],
                n0 * element.v0[2] + xi * element.v1[2] + eta * element.v2[2],
            ];
            let r_vec = [y[0] - point[0], y[1] - point[1], y[2] - point[2]];
            let r = r_vec.iter().map(|value| value * value).sum::<f64>().sqrt();
            let inv_r = 1.0 / r;
            let drdn_y = r_vec
                .iter()
                .zip(normal.iter())
                .map(|(r, n)| r * n)
                .sum::<f64>()
                * inv_r;
            let surface_weight = jacobian * weight;
            let green = Complex64::from_polar(
                surface_weight / (4.0 * std::f64::consts::PI * r),
                wavruim * r,
            );
            let derivative = green * Complex64::new(-inv_r, wavruim);
            result += element.pressure * derivative * drdn_y;
            if element.velocity.norm_sqr() > 1.0e-30 {
                result -= element.velocity * Complex64::new(0.0, pressure_factor) * green;
            }
        }
        result
    }

    #[test]
    fn complex64_field_operator_reuses_device_geometry_and_matches_cpu_rule() {
        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend,
            Err(error) => skip_or_require_cuda!("CUDA Complex64 field residency", error),
        };
        let pool =
            MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 20)).expect("CUDA field pool");
        let element = BemFieldElementC64::from_triangle(
            [-0.7, -0.3, 0.0],
            [0.8, -0.2, 0.1],
            [-0.1, 0.9, -0.15],
            Complex64::new(1.2, -0.4),
            Complex64::new(-0.15, 0.25),
        );
        let wave_number = 2.3;
        let harmonic_factor = -1.0;
        let pressure_factor = 3.7;
        let operator = CudaBemFieldOperatorC64::new(
            pool.clone(),
            vec![element],
            wave_number,
            harmonic_factor,
            pressure_factor,
        )
        .expect("valid CUDA field operator");
        let points_host = vec![
            BemFieldPointC64::new([0.3, -0.4, 1.8]),
            BemFieldPointC64::new([-0.8, 0.2, 1.3]),
            BemFieldPointC64::new([0.1, 0.6, 2.1]),
        ];
        let mut points = pool
            .try_alloc::<BemFieldPointC64>(points_host.len())
            .unwrap();
        let mut output = pool.try_alloc::<Complex64>(points_host.len()).unwrap();
        pool.backend().copy_h2d(&points_host, points.raw_mut());

        // Both actions below use the same uploaded geometry/traces and point
        // buffer.  Only the final explicit readback is allowed to transfer.
        pool.backend().reset_runtime_stats();
        operator.try_apply_device(&points, &mut output).unwrap();
        operator.try_apply_device(&points, &mut output).unwrap();
        let queued = pool.backend().runtime_stats();
        assert_eq!(queued.transfer.h2d_operations, 0);
        assert_eq!(queued.transfer.d2h_operations, 0);
        assert_eq!(queued.kernel_launches, 2);

        let mut actual = vec![Complex64::new(0.0, 0.0); points_host.len()];
        pool.backend().copy_d2h(output.raw(), &mut actual);
        for (index, (actual, point)) in actual.iter().zip(points_host.iter()).enumerate() {
            let expected = reference_field(
                point.position,
                element,
                wave_number,
                harmonic_factor,
                pressure_factor,
            );
            assert!(
                (*actual - expected).norm() <= 3.0e-12 * expected.norm().max(1.0),
                "field value {index} differs: actual={actual:?}, expected={expected:?}"
            );
        }
    }

    #[test]
    fn field_launch_failure_preserves_output_and_releases_workspace() {
        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend,
            Err(error) => skip_or_require_cuda!("CUDA field launch-failure contract", error),
        };
        let pool =
            MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 20)).expect("field pool");
        let baseline = pool.bytes_in_use();
        {
            let operator = CudaBemFieldOperatorC64::new(
                pool.clone(),
                vec![BemFieldElementC64::from_triangle(
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    Complex64::new(1.0, -0.25),
                    Complex64::new(0.1, 0.05),
                )],
                1.7,
                -1.0,
                2.3,
            )
            .expect("valid field operator");
            let points_host = vec![BemFieldPointC64::new([0.2, 0.3, 1.5])];
            let mut points = pool.try_alloc(points_host.len()).expect("point buffer");
            pool.backend().copy_h2d(&points_host, points.raw_mut());
            let sentinel = vec![Complex64::new(7.0, -3.0)];
            let mut output = pool.try_alloc(1).expect("output buffer");
            pool.backend().copy_h2d(&sentinel, output.raw_mut());

            pool.backend()
                .inject_failure(CudaFailureInjectionPoint::Launch);
            let error = operator
                .try_apply_device(&points, &mut output)
                .expect_err("injected launch failure must be typed");
            assert!(matches!(
                error,
                BackendError::Launch {
                    operation: "cuda_launch",
                    ..
                }
            ));
            assert_eq!(pool.backend().runtime_stats().kernel_launches, 0);
            pool.backend()
                .recover_after_error()
                .expect("recoverable launch failure");

            let mut actual = vec![Complex64::default()];
            pool.backend().copy_d2h(output.raw(), &mut actual);
            assert_eq!(
                actual, sentinel,
                "failed field launch must not publish output"
            );
        }
        assert_eq!(
            pool.bytes_in_use(),
            baseline,
            "field operator failure must release resident workspace on drop"
        );
    }
}

#[cfg(feature = "cuda")]
mod cuda_quadrature_residency {
    use num_complex::Complex64;
    use sonium_compute::{
        Backend, BackendError, BemQuadraturePairC64, BemQuadratureSampleC64, BemRegularPairInF64,
        CudaBackend, CudaBemQuadratureOperatorC64, CudaBemRegularPairOperatorC64,
        CudaFailureInjectionPoint, MemoryPool, MemoryPoolConfig,
    };

    fn reference_pair(
        pair: BemQuadraturePairC64,
        samples: &[BemQuadratureSampleC64],
        wave_number: f64,
        harmonic_factor: f64,
    ) -> sonium_compute::BemRegularPairOutF64 {
        let mut result = sonium_compute::BemRegularPairOutF64::default();
        let source = pair.source_point;
        let source_normal = pair.source_normal;
        let wavruim = wave_number * harmonic_factor;
        for sample in
            &samples[pair.sample_offset as usize..(pair.sample_offset + pair.sample_count) as usize]
        {
            let diff = [
                sample.position[0] - source[0],
                sample.position[1] - source[1],
                sample.position[2] - source[2],
            ];
            let r = diff.iter().map(|value| value * value).sum::<f64>().sqrt();
            let unit = diff.map(|value| value / r);
            let scaled_weight = sample.weight * sample.basis_weight;
            let green = Complex64::from_polar(
                scaled_weight / (4.0 * std::f64::consts::PI * r),
                wavruim * r,
            );
            if sample.kind == BemQuadratureSampleC64::KIND_HYPERSINGULAR_EDGE {
                let factor =
                    Complex64::from_polar(1.0 / (4.0 * std::f64::consts::PI * r), wavruim * r)
                        * Complex64::new(-1.0 / r, wavruim);
                let cross = [
                    factor
                        * (unit[1] * sample.edge_direction[2] - unit[2] * sample.edge_direction[1]),
                    factor
                        * (unit[2] * sample.edge_direction[0] - unit[0] * sample.edge_direction[2]),
                    factor
                        * (unit[0] * sample.edge_direction[1] - unit[1] * sample.edge_direction[0]),
                ];
                result.e += cross
                    .into_iter()
                    .zip(source_normal)
                    .map(|(value, normal)| value * normal)
                    .sum::<Complex64>()
                    * scaled_weight;
                continue;
            }

            let field_normal = sample.field_normal;
            let r_dot_ny = unit
                .iter()
                .zip(field_normal)
                .map(|(value, normal)| value * normal)
                .sum::<f64>();
            let r_dot_nx = -unit
                .iter()
                .zip(source_normal)
                .map(|(value, normal)| value * normal)
                .sum::<f64>();
            let normal_dot = source_normal
                .iter()
                .zip(field_normal)
                .map(|(a, b)| a * b)
                .sum::<f64>();
            let factor = Complex64::new(-1.0 / r, wavruim);
            result.g += green;
            result.h += green * factor * r_dot_ny;
            result.ht += green * factor * r_dot_nx;
            match sample.kind {
                BemQuadratureSampleC64::KIND_FULL => {
                    let rq = r_dot_ny * r_dot_nx;
                    let r2 = r * r;
                    result.e += green
                        * Complex64::new(
                            (3.0 / r2 - wave_number * wave_number) * rq + normal_dot / r2,
                            -wavruim / r * (3.0 * rq + normal_dot),
                        );
                }
                BemQuadratureSampleC64::KIND_SINGULAR_BULK => {
                    result.e += green * (wave_number * wave_number * normal_dot);
                }
                BemQuadratureSampleC64::KIND_GH_ONLY => {}
                _ => unreachable!(),
            }
        }
        result
    }

    #[test]
    fn complex64_quadrature_pair_operator_covers_bulk_singular_and_edge_channels() {
        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend,
            Err(error) => skip_or_require_cuda!("CUDA Complex64 quadrature pair", error),
        };
        let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 20))
            .expect("CUDA quadrature pool");
        let samples = vec![
            BemQuadratureSampleC64::full([1.0, 0.2, 0.7], [0.0, 0.0, 1.0], 0.7, 1.2),
            BemQuadratureSampleC64::singular_bulk([0.3, -0.4, 0.9], [0.0, 0.0, 1.0], 0.4, -0.8),
            BemQuadratureSampleC64::hypersingular_edge([0.4, 0.8, 0.6], [1.0, 0.0, 0.0], 0.25, 1.0),
            BemQuadratureSampleC64::gh_only([-0.2, 0.6, 1.1], [0.0, 0.0, 1.0], 0.35, 0.5),
        ];
        let pairs = vec![BemQuadraturePairC64::new(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            0,
            samples.len() as u64,
        )];
        let operator = CudaBemQuadratureOperatorC64::new(
            pool.clone(),
            pairs.clone(),
            samples.clone(),
            2.1,
            -1.0,
        )
        .expect("valid quadrature pair operator");
        let mut output = pool
            .try_alloc::<sonium_compute::BemRegularPairOutF64>(1)
            .unwrap();
        pool.backend().reset_runtime_stats();
        operator.try_apply_device(&mut output).unwrap();
        operator.try_apply_device(&mut output).unwrap();
        let queued = pool.backend().runtime_stats();
        assert_eq!(queued.transfer.h2d_operations, 0);
        assert_eq!(queued.transfer.d2h_operations, 0);
        assert_eq!(queued.kernel_launches, 2);

        let mut actual = vec![sonium_compute::BemRegularPairOutF64::default(); 1];
        pool.backend().copy_d2h(output.raw(), &mut actual);
        let expected = reference_pair(pairs[0], &samples, 2.1, -1.0);
        for (name, got, want) in [
            ("g", actual[0].g, expected.g),
            ("h", actual[0].h, expected.h),
            ("ht", actual[0].ht, expected.ht),
            ("e", actual[0].e, expected.e),
        ] {
            assert!(
                (got - want).norm() <= 2.0e-12,
                "{name}: got {got:?}, want {want:?}"
            );
        }
    }

    #[test]
    fn quadrature_pair_constructor_rejects_invalid_ranges_before_upload() {
        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend,
            Err(error) => skip_or_require_cuda!("CUDA quadrature pair validation", error),
        };
        let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 20))
            .expect("CUDA quadrature pool");
        let error = match CudaBemQuadratureOperatorC64::new(
            pool,
            vec![BemQuadraturePairC64::new([0.0; 3], [0.0, 0.0, 1.0], 1, 1)],
            vec![BemQuadratureSampleC64::default()],
            1.0,
            -1.0,
        ) {
            Ok(_) => panic!("out-of-range sample span must be rejected"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            sonium_compute::BackendError::InvalidArgument {
                operation: "cuda_bem_quadrature_operator",
                ..
            }
        ));
    }

    #[test]
    fn regular_pair_operator_reuses_device_geometry_and_matches_batch() {
        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend,
            Err(error) => skip_or_require_cuda!("CUDA regular-pair residency", error),
        };
        let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 20))
            .expect("CUDA regular-pair pool");
        let pairs = vec![
            BemRegularPairInF64::from_triangle(
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [-0.7, -0.3, 1.0],
                [0.8, -0.2, 1.1],
                [-0.1, 0.9, 1.2],
            ),
            BemRegularPairInF64::from_triangle(
                [0.2, -0.1, 0.3],
                [0.0, 1.0, 0.0],
                [1.2, -0.4, 1.4],
                [1.8, 0.1, 1.6],
                [1.1, 0.8, 1.5],
            ),
        ];
        let wave_number = 2.3;
        let harmonic_factor = -1.0;
        let expected = pool
            .backend()
            .try_bem_regular_pair_c64_batch(&pairs, wave_number, harmonic_factor)
            .expect("regular-pair batch reference");
        let operator =
            CudaBemRegularPairOperatorC64::new(pool.clone(), pairs, wave_number, harmonic_factor)
                .expect("valid regular-pair operator");
        let mut output = pool
            .try_alloc::<sonium_compute::BemRegularPairOutF64>(operator.len())
            .unwrap();

        pool.backend().reset_runtime_stats();
        operator.try_apply_device(&mut output).unwrap();
        let after_first = pool.backend().runtime_stats();
        operator.try_apply_device(&mut output).unwrap();
        let after_second = pool.backend().runtime_stats();
        assert_eq!(after_second.transfer.h2d_operations, 0);
        assert_eq!(after_second.transfer.d2h_operations, 0);
        assert_eq!(after_second.kernel_launches, 2);
        assert_eq!(
            after_second.allocation_high_water_bytes, after_first.allocation_high_water_bytes,
            "repeated regular-pair actions must not allocate"
        );

        let mut actual = vec![sonium_compute::BemRegularPairOutF64::default(); operator.len()];
        pool.backend().copy_d2h(output.raw(), &mut actual);
        for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
            for (name, got, want) in [
                ("g", got.g, want.g),
                ("h", got.h, want.h),
                ("ht", got.ht, want.ht),
                ("e", got.e, want.e),
            ] {
                assert!(
                    (got - want).norm() <= 2.0e-12 * want.norm().max(1.0),
                    "pair {index} {name}: got {got:?}, want {want:?}"
                );
            }
        }
    }

    #[test]
    fn regular_pair_launch_failure_preserves_output_and_releases_workspace() {
        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend,
            Err(error) => skip_or_require_cuda!("CUDA pair launch-failure contract", error),
        };
        let pool =
            MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 20)).expect("pair pool");
        let baseline = pool.bytes_in_use();
        {
            let pair = BemRegularPairInF64::from_triangle(
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [2.0, -0.5, 0.2],
                [2.7, -0.4, 0.3],
                [2.1, 0.4, 0.5],
            );
            let operator = CudaBemRegularPairOperatorC64::new(pool.clone(), vec![pair], 1.7, -1.0)
                .expect("valid regular-pair operator");
            let sentinel = vec![sonium_compute::BemRegularPairOutF64 {
                g: Complex64::new(7.0, -3.0),
                h: Complex64::new(5.0, -2.0),
                ht: Complex64::new(4.0, -1.0),
                e: Complex64::new(2.0, -6.0),
            }];
            let mut output = pool.try_alloc(1).expect("output buffer");
            pool.backend().copy_h2d(&sentinel, output.raw_mut());

            pool.backend()
                .inject_failure(CudaFailureInjectionPoint::Launch);
            let error = operator
                .try_apply_device(&mut output)
                .expect_err("injected launch failure must be typed");
            assert!(matches!(
                error,
                BackendError::Launch {
                    operation: "cuda_launch",
                    ..
                }
            ));
            assert_eq!(pool.backend().runtime_stats().kernel_launches, 0);
            pool.backend()
                .recover_after_error()
                .expect("recoverable launch failure");

            let mut actual = vec![sonium_compute::BemRegularPairOutF64::default()];
            pool.backend().copy_d2h(output.raw(), &mut actual);
            for (name, got, want) in [
                ("g", actual[0].g, sentinel[0].g),
                ("h", actual[0].h, sentinel[0].h),
                ("ht", actual[0].ht, sentinel[0].ht),
                ("e", actual[0].e, sentinel[0].e),
            ] {
                assert_eq!(got, want, "failed pair launch must not publish {name}");
            }
        }
        assert_eq!(
            pool.bytes_in_use(),
            baseline,
            "pair operator failure must release resident workspace on drop"
        );
    }
}

// ---------------------------------------------------------------------------
// BLAS-2 / BLAS-3 parity (gemv, gemm)
// ---------------------------------------------------------------------------

mod blas23_parity {
    use super::*;
    use sonium_compute::Transpose;

    /// 3×4 row-major f64 matrix.
    const M_F64: usize = 3;
    const N_F64: usize = 4;
    fn a_f64() -> Vec<f64> {
        (0..M_F64 * N_F64).map(|k| (k as f64) * 0.1 - 0.5).collect()
    }
    fn x_n_f64() -> Vec<f64> {
        (0..N_F64).map(|k| (k as f64) * 0.3 + 1.0).collect()
    }
    fn x_m_f64() -> Vec<f64> {
        (0..M_F64).map(|k| (k as f64) * 0.4 - 0.2).collect()
    }
    fn y_m_f64() -> Vec<f64> {
        (0..M_F64).map(|k| 1.0 - (k as f64) * 0.1).collect()
    }
    fn y_n_f64() -> Vec<f64> {
        (0..N_F64).map(|k| 0.5 + (k as f64) * 0.07).collect()
    }

    /// Run gemv on any backend and return the resulting y vector.
    #[allow(clippy::too_many_arguments)]
    fn run_gemv_f64<B: Backend>(
        pool: &MemoryPool<B>,
        op: Transpose,
        m: usize,
        n: usize,
        alpha: f64,
        a_host: &[f64],
        lda: usize,
        x_host: &[f64],
        beta: f64,
        y_host: &[f64],
    ) -> Vec<f64> {
        let mut a = pool.try_alloc::<f64>(a_host.len()).unwrap();
        let mut x = pool.try_alloc::<f64>(x_host.len()).unwrap();
        let mut y = pool.try_alloc::<f64>(y_host.len()).unwrap();
        pool.backend().copy_h2d(a_host, a.raw_mut());
        pool.backend().copy_h2d(x_host, x.raw_mut());
        pool.backend().copy_h2d(y_host, y.raw_mut());
        pool.backend()
            .gemv(op, m, n, alpha, a.raw(), lda, x.raw(), beta, y.raw_mut());
        let mut out = vec![0.0_f64; y_host.len()];
        pool.backend().copy_d2h(y.raw(), &mut out);
        out
    }

    #[test]
    fn cpu_gemv_f64_notrans_matches_hand_ref() {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let a = a_f64();
        let x = x_n_f64();
        let y0 = y_m_f64();
        let got = run_gemv_f64(
            &pool,
            Transpose::NoTrans,
            M_F64,
            N_F64,
            0.7,
            &a,
            N_F64,
            &x,
            0.5,
            &y0,
        );
        // Hand reference.
        let mut want = vec![0.0; M_F64];
        for i in 0..M_F64 {
            let mut s = 0.0;
            for j in 0..N_F64 {
                s += a[i * N_F64 + j] * x[j];
            }
            want[i] = 0.7 * s + 0.5 * y0[i];
        }
        assert_vec_close("cpu/gemv_f64_n", &got, &want);
    }

    #[test]
    fn cpu_gemv_f64_trans_matches_hand_ref() {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        // Stored as 3x4. Trans: op(A) is 4x3, x has 3 entries, y has 4.
        let a = a_f64();
        let x = x_m_f64();
        let y0 = y_n_f64();
        let got = run_gemv_f64(
            &pool,
            Transpose::Trans,
            N_F64,
            M_F64,
            1.0,
            &a,
            N_F64,
            &x,
            0.0,
            &y0,
        );
        let mut want = vec![0.0; N_F64];
        for j in 0..N_F64 {
            let mut s = 0.0;
            for i in 0..M_F64 {
                s += a[i * N_F64 + j] * x[i];
            }
            want[j] = s;
        }
        assert_vec_close("cpu/gemv_f64_t", &got, &want);
    }

    #[test]
    fn cpu_gemv_c64_conjtrans_matches_hand_ref() {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let m_stored = 3;
        let n_stored = 2;
        // op = C: op(A) is n_stored x m_stored = 2x3.
        let a: Vec<Complex64> = (0..m_stored * n_stored)
            .map(|k| Complex64::new((k as f64) * 0.3, 1.0 - (k as f64) * 0.2))
            .collect();
        let x: Vec<Complex64> = (0..m_stored)
            .map(|k| Complex64::new(1.0 + (k as f64) * 0.1, (k as f64) * 0.4))
            .collect();
        let y0: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n_stored];

        let mut a_dev = pool.try_alloc::<Complex64>(a.len()).unwrap();
        let mut x_dev = pool.try_alloc::<Complex64>(x.len()).unwrap();
        let mut y_dev = pool.try_alloc::<Complex64>(y0.len()).unwrap();
        pool.backend().copy_h2d(&a, a_dev.raw_mut());
        pool.backend().copy_h2d(&x, x_dev.raw_mut());
        pool.backend().copy_h2d(&y0, y_dev.raw_mut());
        pool.backend().gemv(
            Transpose::ConjTrans,
            n_stored,
            m_stored,
            Complex64::new(1.0, 0.0),
            a_dev.raw(),
            n_stored,
            x_dev.raw(),
            Complex64::new(0.0, 0.0),
            y_dev.raw_mut(),
        );
        let mut got = vec![Complex64::new(0.0, 0.0); n_stored];
        pool.backend().copy_d2h(y_dev.raw(), &mut got);

        // y[j] = sum_i conj(A[i,j]) * x[i] for j in 0..n_stored.
        for j in 0..n_stored {
            let mut s = Complex64::new(0.0, 0.0);
            for i in 0..m_stored {
                s += a[i * n_stored + j].conj() * x[i];
            }
            close_c64(&format!("cpu/gemv_c64_h[{j}]"), got[j], s);
        }
    }

    /// 2×3 · 3×4 = 2×4 gemm reference.
    fn run_gemm_f64<B: Backend>(
        pool: &MemoryPool<B>,
        m: usize,
        n: usize,
        k: usize,
        a: &[f64],
        b: &[f64],
    ) -> Vec<f64> {
        let mut a_d = pool.try_alloc::<f64>(a.len()).unwrap();
        let mut b_d = pool.try_alloc::<f64>(b.len()).unwrap();
        let mut c_d = pool.try_alloc::<f64>(m * n).unwrap();
        pool.backend().copy_h2d(a, a_d.raw_mut());
        pool.backend().copy_h2d(b, b_d.raw_mut());
        pool.backend().gemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            1.0,
            a_d.raw(),
            k,
            b_d.raw(),
            n,
            0.0,
            c_d.raw_mut(),
            n,
        );
        let mut out = vec![0.0_f64; m * n];
        pool.backend().copy_d2h(c_d.raw(), &mut out);
        out
    }

    #[test]
    fn cpu_gemm_f64_basic_matches_hand_ref() {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let m = 2;
        let n = 4;
        let k = 3;
        let a: Vec<f64> = (0..m * k).map(|i| 0.1 + (i as f64) * 0.2).collect();
        let b: Vec<f64> = (0..k * n).map(|i| 1.0 - (i as f64) * 0.05).collect();
        let got = run_gemm_f64(&pool, m, n, k, &a, &b);
        let mut want = vec![0.0_f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                want[i * n + j] = s;
            }
        }
        assert_vec_close("cpu/gemm_f64", &got, &want);
    }

    // CUDA parity (skipped if CUDA driver unavailable).
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_gemv_f64_matches_cpu() {
        use sonium_compute::CudaBackend;
        let backend = match CudaBackend::new_default() {
            Ok(b) => b,
            Err(e) => skip_or_require_cuda!("gemv parity", e),
        };
        let cpu_pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let gpu_pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
        let a = a_f64();
        let x = x_n_f64();
        let y0 = y_m_f64();
        let cpu_y = run_gemv_f64(
            &cpu_pool,
            Transpose::NoTrans,
            M_F64,
            N_F64,
            0.7,
            &a,
            N_F64,
            &x,
            0.5,
            &y0,
        );
        let gpu_y = run_gemv_f64(
            &gpu_pool,
            Transpose::NoTrans,
            M_F64,
            N_F64,
            0.7,
            &a,
            N_F64,
            &x,
            0.5,
            &y0,
        );
        assert_vec_close("cuda/gemv_f64_n", &gpu_y, &cpu_y);

        let x_t = x_m_f64();
        let y_t = y_n_f64();
        let cpu_t = run_gemv_f64(
            &cpu_pool,
            Transpose::Trans,
            N_F64,
            M_F64,
            1.0,
            &a,
            N_F64,
            &x_t,
            0.0,
            &y_t,
        );
        let gpu_t = run_gemv_f64(
            &gpu_pool,
            Transpose::Trans,
            N_F64,
            M_F64,
            1.0,
            &a,
            N_F64,
            &x_t,
            0.0,
            &y_t,
        );
        assert_vec_close("cuda/gemv_f64_t", &gpu_t, &cpu_t);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_gemv_c64_conjtrans_matches_cpu() {
        use sonium_compute::CudaBackend;
        let backend = match CudaBackend::new_default() {
            Ok(b) => b,
            Err(e) => skip_or_require_cuda!("gemv_c64 parity", e),
        };
        let m_stored = 3;
        let n_stored = 2;
        let a: Vec<Complex64> = (0..m_stored * n_stored)
            .map(|k| Complex64::new((k as f64) * 0.3, 1.0 - (k as f64) * 0.2))
            .collect();
        let x: Vec<Complex64> = (0..m_stored)
            .map(|k| Complex64::new(1.0 + (k as f64) * 0.1, (k as f64) * 0.4))
            .collect();

        let cpu_pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let gpu_pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();

        let run = |pool: &dyn Fn() -> Vec<Complex64>| pool();
        let _ = run; // placeholder; we inline both runs below for the two backend types.

        let cpu_y = {
            let mut a_d = cpu_pool.try_alloc::<Complex64>(a.len()).unwrap();
            let mut x_d = cpu_pool.try_alloc::<Complex64>(x.len()).unwrap();
            let mut y_d = cpu_pool.try_alloc::<Complex64>(n_stored).unwrap();
            cpu_pool.backend().copy_h2d(&a, a_d.raw_mut());
            cpu_pool.backend().copy_h2d(&x, x_d.raw_mut());
            cpu_pool.backend().gemv(
                Transpose::ConjTrans,
                n_stored,
                m_stored,
                Complex64::new(1.0, 0.0),
                a_d.raw(),
                n_stored,
                x_d.raw(),
                Complex64::new(0.0, 0.0),
                y_d.raw_mut(),
            );
            let mut out = vec![Complex64::new(0.0, 0.0); n_stored];
            cpu_pool.backend().copy_d2h(y_d.raw(), &mut out);
            out
        };
        let gpu_y = {
            let mut a_d = gpu_pool.try_alloc::<Complex64>(a.len()).unwrap();
            let mut x_d = gpu_pool.try_alloc::<Complex64>(x.len()).unwrap();
            let mut y_d = gpu_pool.try_alloc::<Complex64>(n_stored).unwrap();
            gpu_pool.backend().copy_h2d(&a, a_d.raw_mut());
            gpu_pool.backend().copy_h2d(&x, x_d.raw_mut());
            gpu_pool.backend().gemv(
                Transpose::ConjTrans,
                n_stored,
                m_stored,
                Complex64::new(1.0, 0.0),
                a_d.raw(),
                n_stored,
                x_d.raw(),
                Complex64::new(0.0, 0.0),
                y_d.raw_mut(),
            );
            let mut out = vec![Complex64::new(0.0, 0.0); n_stored];
            gpu_pool.backend().copy_d2h(y_d.raw(), &mut out);
            out
        };

        for (i, (g, c)) in gpu_y.iter().zip(cpu_y.iter()).enumerate() {
            close_c64(&format!("cuda/gemv_c64_h[{i}]"), *g, *c);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_gemm_f64_matches_cpu() {
        use sonium_compute::CudaBackend;
        let backend = match CudaBackend::new_default() {
            Ok(b) => b,
            Err(e) => skip_or_require_cuda!("gemm parity", e),
        };
        let m = 2;
        let n = 4;
        let k = 3;
        let a: Vec<f64> = (0..m * k).map(|i| 0.1 + (i as f64) * 0.2).collect();
        let b: Vec<f64> = (0..k * n).map(|i| 1.0 - (i as f64) * 0.05).collect();
        let cpu_pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let gpu_pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();
        let cpu_c = run_gemm_f64(&cpu_pool, m, n, k, &a, &b);
        let gpu_c = run_gemm_f64(&gpu_pool, m, n, k, &a, &b);
        assert_vec_close("cuda/gemm_f64", &gpu_c, &cpu_c);
    }

    // Metal parity (on macOS only; tests against CPU on the same host).
    #[cfg(feature = "metal")]
    #[test]
    fn metal_gemv_gemm_matches_cpu() {
        use sonium_compute::MetalBackend;
        let backend = match MetalBackend::new() {
            Ok(b) => b,
            Err(e) => skip_or_require_metal!("gemv/gemm parity", e),
        };
        let cpu_pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let gpu_pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20)).unwrap();

        // gemv f64 NoTrans
        let a = a_f64();
        let x = x_n_f64();
        let y0 = y_m_f64();
        let cpu_y = run_gemv_f64(
            &cpu_pool,
            Transpose::NoTrans,
            M_F64,
            N_F64,
            0.7,
            &a,
            N_F64,
            &x,
            0.5,
            &y0,
        );
        let gpu_y = run_gemv_f64(
            &gpu_pool,
            Transpose::NoTrans,
            M_F64,
            N_F64,
            0.7,
            &a,
            N_F64,
            &x,
            0.5,
            &y0,
        );
        assert_vec_close("metal/gemv_f64_n", &gpu_y, &cpu_y);

        // gemm f64
        let m = 2;
        let n = 4;
        let k = 3;
        let am: Vec<f64> = (0..m * k).map(|i| 0.1 + (i as f64) * 0.2).collect();
        let bm: Vec<f64> = (0..k * n).map(|i| 1.0 - (i as f64) * 0.05).collect();
        let cpu_c = run_gemm_f64(&cpu_pool, m, n, k, &am, &bm);
        let gpu_c = run_gemm_f64(&gpu_pool, m, n, k, &am, &bm);
        assert_vec_close("metal/gemm_f64", &gpu_c, &cpu_c);
    }
}
