//! Independent row-norm checks for backend L1-Jacobi smoothing.

// Rust guideline compliant 2026-02-21
use super::*;
use num_complex::Complex64;
use sonium_compute::{Backend, CpuBackend, MemoryPool, MemoryPoolConfig};

fn check_l1<B: Backend>(pool: &MemoryPool<B>, complex: bool) {
    let backend = pool.backend();
    let mut entries = Vec::new();
    for i in 0..32 {
        entries.push((i, i, Complex64::new(3.0, if complex { 0.4 } else { 0.0 })));
        if i + 1 < 32 {
            let value = Complex64::new(-1.0, if complex { 0.2 } else { 0.0 });
            entries.extend([(i, i + 1, value), (i + 1, i, value)]);
        }
    }
    let host = AmgPreconditioner::try_from_csr(
        &CsrMatrix::from_triplets(32, 32, entries),
        AmgConfig {
            smoother: AmgSmoother::L1Jacobi,
            // Deliberately distinct from L1's unit damping.
            jacobi_weight: 0.17,
            coarse_size: 2,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(host.levels.len() > 1);
    let device = BackendAmgPreconditioner::try_from_host_amg(backend, &host).unwrap();
    let restored = BackendAmgPreconditioner::try_from_checkpoint_bytes(
        backend,
        &device.to_checkpoint_bytes().unwrap(),
    )
    .unwrap();
    for amg in [&device, &restored] {
        let mut work = amg.workspace.lock().unwrap();
        for (level_index, level) in host.levels.iter().enumerate() {
            let n = level.num_dofs;
            let b: Vec<_> = (0..n)
                .map(|i| Complex64::new((i as f64).sin(), 0.3))
                .collect();
            let initial: Vec<_> = (0..n)
                .map(|i| Complex64::new(0.2 * i as f64, -0.1))
                .collect();
            let mut expected = initial.clone();
            // Independent simultaneous row-norm relaxation, not the host smoother.
            for _ in 0..3 {
                let previous = expected.clone();
                for row in 0..n {
                    let mut ax = Complex64::default();
                    let mut norm = 0.0;
                    for j in level.matrix.row_ptrs[row]..level.matrix.row_ptrs[row + 1] {
                        let value = level.matrix.values[j];
                        ax += value * previous[level.matrix.col_indices[j]];
                        norm += value.norm();
                    }
                    expected[row] += (b[row] - ax) / norm;
                }
            }
            let mut rhs = pool.try_alloc::<Complex64>(n).unwrap();
            let mut x = pool.try_alloc::<Complex64>(n).unwrap();
            backend.try_copy_h2d(&b, rhs.raw_mut()).unwrap();
            backend.try_copy_h2d(&initial, x.raw_mut()).unwrap();
            amg.smooth(
                backend,
                level_index,
                &mut work.levels[level_index],
                x.raw_mut(),
                rhs.raw(),
                3,
            )
            .unwrap();
            let mut actual = vec![Complex64::default(); n];
            backend.try_copy_d2h(x.raw(), &mut actual).unwrap();
            for (a, e) in actual.iter().zip(expected) {
                assert!((*a - e).norm() < 2e-12 * (1.0 + e.norm()), "{a} != {e}");
            }
        }
    }
}

#[test]
fn backend_l1_scaling_matches_row_norms_at_every_level_and_after_restore() {
    let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(1 << 24)).unwrap();
    check_l1(&pool, false);
    check_l1(&pool, true);
}

#[cfg(feature = "cuda")]
#[test]
fn native_l1_scaling_matches_row_norms_at_every_level_and_after_restore() {
    let backend = match sonium_compute::CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                "CUDA required: {e}"
            );
            eprintln!("SKIP CUDA L1 smoothing: {e}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 24)).unwrap();
    check_l1(&pool, false);
    check_l1(&pool, true);
}
