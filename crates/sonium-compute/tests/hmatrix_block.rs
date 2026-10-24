//! Native compressed block actions use indexed views without full-vector copies.
#![cfg(feature = "cuda")]

// Rust guideline compliant 2026-02-21
use num_complex::Complex64 as C;
use sonium_compute::{
    Backend, CudaBackend, CudaDenseBlockC64Input, CudaHMatrixC64, CudaLowRankBlockC64Input,
    DeviceOperator, MemoryPool, MemoryPoolConfig,
};

#[test]
fn native_hmatrix_block_orientations_and_validation() {
    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(error) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                "{error}"
            );
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::default()).unwrap();
    let matrix: Vec<_> = (0..9)
        .map(|i| C::new((i + 1) as f64, 0.2 * i as f64 - 0.7))
        .collect();
    let operator = CudaHMatrixC64::new(
        pool.clone(),
        3,
        vec![CudaDenseBlockC64Input {
            row_indices: vec![0, 1],
            col_indices: vec![0, 1, 2],
            matrix: matrix[..6].to_vec(),
        }],
        vec![CudaLowRankBlockC64Input {
            row_indices: vec![2],
            col_indices: vec![0, 1, 2],
            rank: 2,
            u: vec![C::new(1.0, 0.0), C::new(0.0, 2.0)],
            v: matrix[6..]
                .iter()
                .flat_map(|v| [*v * 0.5, *v / C::new(0.0, 4.0)])
                .collect(),
        }],
    )
    .unwrap();
    for width in [1, 3, 2] {
        let input: Vec<_> = (0..3 * width)
            .map(|i| C::new(0.3 * i as f64, 1.0 - 0.17 * i as f64))
            .collect();
        let mut x = pool.try_alloc(input.len()).unwrap();
        let mut y = pool.try_alloc(input.len()).unwrap();
        pool.backend().try_copy_h2d(&input, x.raw_mut()).unwrap();
        for kind in 0..3 {
            let before = pool.backend().begin_runtime_stats();
            match kind {
                0 => operator.try_apply_colmajor(pool.backend(), width, x.raw(), y.raw_mut()),
                1 => operator.try_apply_transpose_colmajor_raw(width, x.raw(), y.raw_mut()),
                _ => operator.try_apply_hermitian_colmajor_raw(width, x.raw(), y.raw_mut()),
            }
            .unwrap();
            let stats = pool.backend().runtime_stats_since(before);
            assert_eq!(stats.allocation_operations, 0, "{stats:?}");
            assert_eq!(stats.transfer.h2d_bytes, 0, "{stats:?}");
            assert_eq!(stats.transfer.d2h_bytes, 0, "{stats:?}");
            assert_eq!(stats.transfer.d2d_bytes, 0, "{stats:?}");
            let mut got = vec![C::default(); input.len()];
            pool.backend().try_copy_d2h(y.raw(), &mut got).unwrap();
            for col in 0..width {
                for row in 0..3 {
                    let expected: C = (0..3)
                        .map(|j| {
                            let value = match kind {
                                0 => matrix[row * 3 + j],
                                1 => matrix[j * 3 + row],
                                _ => matrix[j * 3 + row].conj(),
                            };
                            value * input[col * 3 + j]
                        })
                        .sum();
                    assert!((got[col * 3 + row] - expected).norm() < 1e-12);
                }
            }
        }
        assert!(
            operator
                .try_apply_colmajor(pool.backend(), width + 1, x.raw(), y.raw_mut())
                .is_err()
        );
        assert!(
            operator
                .try_apply_transpose_colmajor_raw(usize::MAX, x.raw(), y.raw_mut())
                .is_err()
        );
    }
    let foreign = MemoryPool::new(
        CudaBackend::new_default().unwrap(),
        MemoryPoolConfig::default(),
    )
    .unwrap();
    let x = foreign.try_alloc::<C>(6).unwrap();
    let mut y = pool.try_alloc::<C>(6).unwrap();
    let local = pool.try_alloc::<C>(6).unwrap();
    let mut foreign_output = foreign.try_alloc::<C>(6).unwrap();
    let before = pool.backend().begin_runtime_stats();
    assert!(
        operator
            .try_apply_colmajor(pool.backend(), 2, x.raw(), y.raw_mut())
            .is_err()
    );
    assert!(
        operator
            .try_apply_transpose_colmajor_raw(2, x.raw(), y.raw_mut())
            .is_err()
    );
    assert!(
        operator
            .try_apply_hermitian_colmajor_raw(2, x.raw(), y.raw_mut())
            .is_err()
    );
    assert!(
        operator
            .try_apply_colmajor(pool.backend(), 2, local.raw(), foreign_output.raw_mut())
            .is_err()
    );
    assert!(
        operator
            .try_apply_transpose_colmajor_raw(2, local.raw(), foreign_output.raw_mut())
            .is_err()
    );
    assert!(
        operator
            .try_apply_hermitian_colmajor_raw(2, local.raw(), foreign_output.raw_mut())
            .is_err()
    );
    assert!(
        operator
            .try_apply_colmajor(foreign.backend(), 2, local.raw(), y.raw_mut())
            .is_err()
    );
    let stats = pool.backend().runtime_stats_since(before);
    assert_eq!(stats.allocation_operations, 0);
    assert_eq!(stats.kernel_launches, 0);
    assert_eq!(stats.library_operations, 0);
    assert_eq!(
        stats.transfer.h2d_bytes + stats.transfer.d2h_bytes + stats.transfer.d2d_bytes,
        0
    );
}
