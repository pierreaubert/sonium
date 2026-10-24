//! Verify the complete backend PCG cycle, including its approximate coarse solve.

// Rust guideline compliant 2026-02-21
use super::*;
use num_complex::Complex64;
use sonium_compute::{Backend, CpuBackend, DevicePreconditioner, MemoryPool, MemoryPoolConfig};

fn check_cycle<B: Backend>(pool: &MemoryPool<B>, complex: bool) {
    let n = 48;
    let mut entries = Vec::new();
    for i in 0..n {
        entries.push((i, i, Complex64::new(4.0 + (i % 3) as f64, 0.0)));
        if i + 1 < n {
            let value = Complex64::new(-1.0, if complex { 0.3 } else { 0.0 });
            entries.extend([(i, i + 1, value), (i + 1, i, value.conj())]);
        }
    }
    let host = AmgPreconditioner::try_from_csr(
        &CsrMatrix::from_triplets(n, n, entries),
        AmgConfig {
            coarse_size: 4,
            ..AmgConfig::for_backend_pcg()
        },
    )
    .unwrap();
    assert!(host.levels.len() > 1);
    if complex {
        assert!(
            host.levels
                .iter()
                .filter_map(|level| level.prolongation.as_ref())
                .any(|p| p.values.iter().any(|v| v.im.abs() > 1e-10))
        );
    }
    let backend = pool.backend();
    let amg = BackendAmgPreconditioner::try_from_host_amg(backend, &host).unwrap();
    let restored = BackendAmgPreconditioner::try_from_checkpoint_bytes(
        backend,
        &amg.to_checkpoint_bytes().unwrap(),
    )
    .unwrap();
    let mut rhs = pool.try_alloc::<Complex64>(n).unwrap();
    let mut out = pool.try_alloc::<Complex64>(n).unwrap();
    let mut dense = vec![Complex64::default(); n * n];
    for column in 0..n {
        let mut unit = vec![Complex64::default(); n];
        unit[column] = Complex64::new(1.0, 0.0);
        backend.try_copy_h2d(&unit, rhs.raw_mut()).unwrap();
        amg.try_apply(backend, rhs.raw(), out.raw_mut()).unwrap();
        let before = backend.transfer_stats();
        amg.try_apply(backend, rhs.raw(), out.raw_mut()).unwrap();
        let after = backend.transfer_stats();
        assert_eq!(after.h2d_bytes, before.h2d_bytes);
        assert_eq!(after.d2h_bytes, before.d2h_bytes);
        let mut actual = vec![Complex64::default(); n];
        backend.try_copy_d2h(out.raw(), &mut actual).unwrap();
        for row in 0..n {
            dense[row * n + column] = actual[row];
        }
        restored
            .try_apply(backend, rhs.raw(), out.raw_mut())
            .unwrap();
        let mut replay = vec![Complex64::default(); n];
        backend.try_copy_d2h(out.raw(), &mut replay).unwrap();
        for (a, b) in actual.iter().zip(replay) {
            assert!((*a - b).norm() < 1e-12);
        }
    }
    assert_eq!(amg.diagnostics().runtime.fallback_count, 0);
    // Form the entire applied inverse, check Hermiticity, then independently
    // factor it. Positive pivots test definiteness beyond selected dot products.
    for i in 0..n {
        for j in 0..n {
            assert!((dense[i * n + j] - dense[j * n + i].conj()).norm() < 2e-12);
        }
    }
    let mut lower = vec![Complex64::default(); n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut value = dense[i * n + j];
            for k in 0..j {
                value -= lower[i * n + k] * lower[j * n + k].conj();
            }
            if i == j {
                assert!(
                    value.re > 1e-12 && value.im.abs() < 2e-12,
                    "nonpositive pivot: {value}"
                );
                lower[i * n + j] = Complex64::new(value.re.sqrt(), 0.0);
            } else {
                lower[i * n + j] = value / lower[j * n + j];
            }
        }
    }
}

#[test]
fn backend_pcg_cycle_is_hpd_and_checkpoint_stable() {
    let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(1 << 25)).unwrap();
    check_cycle(&pool, false);
    check_cycle(&pool, true);
}

#[cfg(feature = "cuda")]
#[test]
fn native_backend_pcg_cycle_is_hpd_and_resident() {
    let backend = match sonium_compute::CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                "CUDA required: {error}"
            );
            eprintln!("SKIP CUDA PCG cycle: {error}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 25)).unwrap();
    check_cycle(&pool, false);
    check_cycle(&pool, true);
}
