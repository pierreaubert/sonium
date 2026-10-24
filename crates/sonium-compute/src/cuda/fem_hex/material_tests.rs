//! Native checks for absolute material lowering and retained reference semantics.

// Rust guideline compliant 2026-02-21
use super::*;

#[test]
fn native_material_updates_preserve_reference_and_residency() {
    let backend = match CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                "CUDA required: {error}"
            );
            eprintln!("SKIP CUDA material update: {error}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, crate::MemoryPoolConfig::with_cap(1 << 25)).unwrap();
    let backend = pool.backend();
    for (n, q) in [(2, 3), (3, 2), (4, 5), (8, 8)] {
        let reference = crate::fem_hex::tests::fixture(n, q)
            .try_with_dirichlet(&[0, 3])
            .unwrap();
        let mut op = CudaHexPartialC64::try_new(pool.clone(), &reference, 2).unwrap();
        let count = op.material_sample_count();
        let unit = vec![Complex64::new(1.0, 0.0); count];
        let mut diffusion = upload(&pool, &unit).unwrap();
        let mut mass = upload(&pool, &unit).unwrap();
        assert!(
            op.try_update_material_samples(diffusion.raw(), mass.raw())
                .is_err()
        );
        op.try_prepare_material_updates().unwrap();
        let before = backend.runtime_stats();
        op.try_prepare_material_updates().unwrap();
        assert_eq!(
            backend.runtime_stats().allocation_operations,
            before.allocation_operations
        );
        assert_eq!(
            backend.runtime_stats().transfer.d2d_bytes,
            before.transfer.d2d_bytes
        );
        let scale_d = Complex64::new(0.7, -0.2);
        let scale_m = Complex64::new(-0.8, 0.3);
        op.try_set_scales(scale_d, scale_m).unwrap();
        let input: Vec<_> = (0..2 * reference.dimension())
            .map(|i| Complex64::new((i as f64 * 0.3).sin(), (i as f64 * 0.2).cos()))
            .collect();
        let x = upload(&pool, &input).unwrap();
        let mut y = pool.try_alloc::<Complex64>(input.len()).unwrap();
        // Zero -> heterogeneous complex -> unit checks absolute, noncumulative updates.
        for update in 0..3 {
            let ds: Vec<_> = (0..count)
                .map(|i| match update {
                    0 => Complex64::default(),
                    1 => Complex64::new(0.8 + (i % 7) as f64 * 0.1, (i % 3) as f64 * 0.02),
                    _ => Complex64::new(1.0, 0.0),
                })
                .collect();
            let ms: Vec<_> = ds
                .iter()
                .enumerate()
                .map(|(i, d)| {
                    if update == 1 {
                        *d * Complex64::new(0.5, (i % 5) as f64 * 0.03)
                    } else {
                        *d
                    }
                })
                .collect();
            backend.try_copy_h2d(&ds, diffusion.raw_mut()).unwrap();
            backend.try_copy_h2d(&ms, mass.raw_mut()).unwrap();
            if update == 1 {
                // Raw replacements and idempotent preparation cannot redefine the reference.
                let zeros =
                    upload(&pool, &vec![Complex64::default(); reference.data.len()]).unwrap();
                op.try_update_quadrature_data(zeros.raw()).unwrap();
                op.try_prepare_material_updates().unwrap();
                backend.inject_failure(crate::CudaFailureInjectionPoint::Launch);
                assert!(
                    op.try_update_material_samples(diffusion.raw(), mass.raw())
                        .is_err()
                );
                backend.recover_after_error().unwrap();
                assert!(
                    op.try_apply_action(backend, 2, x.raw(), y.raw_mut(), Transpose::NoTrans)
                        .is_err()
                );
            }
            let before = backend.runtime_stats();
            op.try_update_material_samples(diffusion.raw(), mass.raw())
                .unwrap();
            let after = backend.runtime_stats();
            assert_eq!(after.allocation_operations, before.allocation_operations);
            assert_eq!(after.kernel_launches, before.kernel_launches + 1);
            assert_eq!(after.transfer.h2d_bytes, before.transfer.h2d_bytes);
            assert_eq!(after.transfer.d2h_bytes, before.transfer.d2h_bytes);
            assert_eq!(after.transfer.d2d_bytes, before.transfer.d2d_bytes);
            let wrong = pool.try_alloc::<Complex64>(count - 1).unwrap();
            assert!(
                op.try_update_material_samples(wrong.raw(), mass.raw())
                    .is_err()
            );
            assert!(
                op.try_update_material_samples(diffusion.raw(), wrong.raw())
                    .is_err()
            );
            let mut expected = reference.clone();
            for (point, values) in expected.data.as_chunks_mut::<7>().0.iter_mut().enumerate() {
                values[0] *= ms[point] * scale_m;
                for value in &mut values[1..] {
                    *value *= ds[point] * scale_d;
                }
            }
            for action in [Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
                op.try_apply_action(backend, 2, x.raw(), y.raw_mut(), action)
                    .unwrap();
                let mut actual = vec![Complex64::default(); input.len()];
                backend.try_copy_d2h(y.raw(), &mut actual).unwrap();
                for column in 0..2 {
                    let start = column * reference.dimension();
                    let end = start + reference.dimension();
                    let mut target = vec![Complex64::default(); reference.dimension()];
                    expected
                        .apply(&input[start..end], &mut target, action)
                        .unwrap();
                    for (a, e) in actual[start..end].iter().zip(target) {
                        assert!(
                            (*a - e).norm() < 1e-9 * (1.0 + e.norm()),
                            "n={n}, q={q}, update={update}: {a} != {e}"
                        );
                    }
                }
            }
        }
    }
}
