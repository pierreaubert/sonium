//! Standalone scalar/block Chebyshev parity and warm application measurements.

// Rust guideline compliant 2026-02-21
#![cfg(feature = "cuda")]
use num_complex::{Complex32, Complex64};
use num_traits::{FromPrimitive, ToPrimitive};
use sonium_compute::{
    Backend, ChebyshevBackend, CpuBackend, CudaBackend, DeviceCsr, DevicePreconditioner,
    DeviceScalar, MemoryPool, MemoryPoolConfig,
};

fn check<T: DeviceScalar>(pool: &MemoryPool<CudaBackend>, diagonal: T, off: T, tolerance: f64) {
    let timing = std::env::var("SONIUM_CHEBYSHEV_BENCH").as_deref() == Ok("1");
    let n = if timing { 4096 } else { 257 };
    let real = |v| T::Real::from_f64(v).unwrap();
    let scalar = |v| T::from_real(real(v));
    let mut rows = vec![0];
    let mut cols = Vec::new();
    let mut values = Vec::new();
    for i in 0..n {
        if i > 0 {
            cols.push(i as i32 - 1);
            values.push(off);
        }
        cols.push(i as i32);
        values.push(diagonal);
        if i + 1 < n {
            cols.push(i as i32 + 1);
            values.push(off);
        }
        rows.push(cols.len() as i32);
    }
    let backend = pool.backend();
    let cpu = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(1 << 26)).unwrap();
    let a = DeviceCsr::try_from_host(backend, n, n, &rows, &cols, &values).unwrap();
    let cpu_a = DeviceCsr::try_from_host(cpu.backend(), n, n, &rows, &cols, &values).unwrap();
    for degree in [1, 4] {
        let pre = ChebyshevBackend::try_with_lambda_max(
            backend,
            &a,
            &vec![diagonal; n],
            degree,
            real(3.),
            real(1.8),
        )
        .unwrap();
        let reference = ChebyshevBackend::try_with_lambda_max(
            cpu.backend(),
            &cpu_a,
            &vec![diagonal; n],
            degree,
            real(3.),
            real(1.8),
        )
        .unwrap();
        let mut single_r = pool.try_alloc::<T>(n).unwrap();
        let mut single_z = pool.try_alloc::<T>(n).unwrap();
        backend
            .try_copy_h2d(&vec![scalar(1.); n], single_r.raw_mut())
            .unwrap();
        let widths: &[usize] = if timing { &[1, 3] } else { &[0, 1, 3] };
        for &width in widths {
            let len = n * width;
            let mut r = pool.try_alloc::<T>(len).unwrap();
            let mut z = pool.try_alloc::<T>(len).unwrap();
            let mut cpu_r = cpu.try_alloc::<T>(len).unwrap();
            let mut cpu_z = cpu.try_alloc::<T>(len).unwrap();
            for pass in 0..3 {
                let rhs: Vec<_> = (0..len)
                    .map(|i| scalar(((i + pass) as f64).sin()) + off)
                    .collect();
                backend.try_copy_h2d(&rhs, r.raw_mut()).unwrap();
                backend
                    .try_copy_h2d(&vec![scalar(f64::NAN); len], z.raw_mut())
                    .unwrap();
                cpu.backend().try_copy_h2d(&rhs, cpu_r.raw_mut()).unwrap();
                reference
                    .try_apply_colmajor(cpu.backend(), width, cpu_r.raw(), cpu_z.raw_mut())
                    .unwrap();
                let before = backend.runtime_stats();
                pre.try_apply_colmajor(backend, width, r.raw(), z.raw_mut())
                    .unwrap();
                pre.try_apply(backend, single_r.raw(), single_z.raw_mut())
                    .unwrap();
                pre.try_apply_colmajor(backend, width, r.raw(), z.raw_mut())
                    .unwrap();
                let after = backend.runtime_stats();
                if pass > 0 {
                    assert_eq!(after.allocation_operations, before.allocation_operations);
                    assert_eq!(after.transfer.h2d_bytes, before.transfer.h2d_bytes);
                    assert_eq!(after.transfer.d2h_bytes, before.transfer.d2h_bytes);
                }
                let mut actual = vec![T::zero(); len];
                let mut expected = actual.clone();
                backend.try_copy_d2h(z.raw(), &mut actual).unwrap();
                cpu.backend()
                    .try_copy_d2h(cpu_z.raw(), &mut expected)
                    .unwrap();
                for (a, e) in actual.iter().zip(expected) {
                    assert!(
                        (*a - e).magnitude().to_f64().unwrap()
                            < tolerance * (1. + e.magnitude().to_f64().unwrap())
                    );
                }
            }
            if timing {
                let mut samples = Vec::new();
                for _ in 0..5 {
                    backend.try_synchronize().unwrap();
                    let start = std::time::Instant::now();
                    for _ in 0..20 {
                        pre.try_apply_colmajor(backend, width, r.raw(), z.raw_mut())
                            .unwrap();
                    }
                    backend.try_synchronize().unwrap();
                    samples.push(start.elapsed().as_secs_f64() / 20.);
                }
                samples.sort_by(f64::total_cmp);
                eprintln!(
                    "CHEBYSHEV_APPLY scalar={} n={n} degree={degree} width={width} median_seconds={} samples={samples:?}",
                    std::any::type_name::<T>(),
                    samples[2]
                );
            }
        }
    }
}

#[test]
fn native_chebyshev_retained_scalar_and_block() {
    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            assert!(std::env::var_os("SONIUM_REQUIRE_CUDA").is_none(), "{e}");
            eprintln!("SKIP native standalone Chebyshev: {e}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 27)).unwrap();
    check(&pool, 3.2_f32, -0.4, 3e-6);
    check(&pool, 3.2_f64, -0.4, 3e-13);
    check(
        &pool,
        Complex32::new(3.2, 0.2),
        Complex32::new(-0.4, 0.1),
        3e-6,
    );
    check(
        &pool,
        Complex64::new(3.2, 0.2),
        Complex64::new(-0.4, 0.1),
        3e-13,
    );
}
