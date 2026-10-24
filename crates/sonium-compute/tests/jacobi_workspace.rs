//! Scalar recovery must not evict the block preconditioner workspace.

// Rust guideline compliant 2026-02-21
use num_complex::{Complex32, Complex64};
use num_traits::{Float, FromPrimitive};
use sonium_compute::{Backend, CpuBackend, DevicePreconditioner, DeviceScalar, PointJacobiBackend};

fn check<B: Backend, T: DeviceScalar>(backend: &B, allocations: impl Fn() -> Option<usize>) {
    let real = |v| T::Real::from_f64(v).unwrap();
    let scalar = |v| T::from_real(real(v));
    let n = 257;
    let mut preconditioner =
        PointJacobiBackend::try_from_diagonal(backend, &vec![scalar(2.0); n]).unwrap();
    let mut scalar_rhs = backend.raw_alloc::<T>(n).unwrap();
    let mut scalar_out = backend.raw_alloc::<T>(n).unwrap();
    let mut block_rhs = backend.raw_alloc::<T>(2 * n).unwrap();
    let mut block_out = backend.raw_alloc::<T>(2 * n).unwrap();
    backend
        .try_copy_h2d(&vec![scalar(6.0); n], &mut scalar_rhs)
        .unwrap();
    backend
        .try_copy_h2d(&vec![scalar(6.0); 2 * n], &mut block_rhs)
        .unwrap();
    let mut baseline = None;
    for diagonal in [2.0, 3.0, 4.0] {
        preconditioner
            .try_update_diagonal(backend, &vec![scalar(diagonal); n])
            .unwrap();
        for _ in 0..2 {
            preconditioner
                .try_apply_colmajor(backend, 2, &block_rhs, &mut block_out)
                .unwrap();
            preconditioner
                .try_apply(backend, &scalar_rhs, &mut scalar_out)
                .unwrap();
            preconditioner
                .try_apply_colmajor(backend, 1, &scalar_rhs, &mut scalar_out)
                .unwrap();
        }
        if let (Some(before), Some(after)) = (baseline, allocations()) {
            assert_eq!(before, after, "Jacobi lost retained scalar/block storage");
        }
        baseline = allocations();
        for (buffer, len) in [(&scalar_out, n), (&block_out, 2 * n)] {
            let mut actual = vec![T::zero(); len];
            backend.try_copy_d2h(buffer, &mut actual).unwrap();
            assert!(
                actual
                    .iter()
                    .all(|&v| (v - scalar(6.0 / diagonal)).norm_sqr().sqrt() < real(1e-5))
            );
        }
    }
}

#[test]
fn cpu_jacobi_alternates_scalar_and_block_after_update() {
    let backend = CpuBackend::new();
    check::<_, f32>(&backend, || None);
    check::<_, f64>(&backend, || None);
    check::<_, Complex32>(&backend, || None);
    check::<_, Complex64>(&backend, || None);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_jacobi_keeps_scalar_and_block_storage_after_update() {
    let backend = match sonium_compute::CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            assert!(
                std::env::var_os("SONIUM_REQUIRE_CUDA").is_none(),
                "CUDA required: {error}"
            );
            return;
        }
    };
    // A failed first block allocation preserves output and the scalar cache.
    let preconditioner = PointJacobiBackend::try_from_diagonal(&backend, &[2.0_f64; 3]).unwrap();
    let mut rhs = backend.raw_alloc::<f64>(6).unwrap();
    let mut output = backend.raw_alloc::<f64>(6).unwrap();
    backend.try_copy_h2d(&[6.0; 6], &mut rhs).unwrap();
    backend.try_copy_h2d(&[91.0; 6], &mut output).unwrap();
    backend.inject_preconditioner_failure(
        sonium_compute::PreconditionerFailureInjectionPoint::WorkspaceAllocation,
    );
    assert!(
        preconditioner
            .try_apply_colmajor(&backend, 2, &rhs, &mut output)
            .is_err()
    );
    let mut actual = [0.0; 6];
    backend.try_copy_d2h(&output, &mut actual).unwrap();
    assert_eq!(actual, [91.0; 6]);
    preconditioner
        .try_apply_colmajor(&backend, 2, &rhs, &mut output)
        .unwrap();
    backend.try_copy_d2h(&output, &mut actual).unwrap();
    assert_eq!(actual, [3.0; 6]);
    let allocations = || Some(backend.runtime_stats().allocation_operations);
    check::<_, f32>(&backend, allocations);
    check::<_, f64>(&backend, allocations);
    check::<_, Complex32>(&backend, allocations);
    check::<_, Complex64>(&backend, allocations);
}

fn check_polynomial_storage<B: Backend, T: DeviceScalar>(
    backend: &B,
    allocations: impl Fn() -> Option<usize> + Copy,
) {
    use sonium_compute::{ChebyshevBackend, DeviceCsr, PolynomialJacobiBackend};
    let scalar = |v| T::from_real(T::Real::from_f64(v).unwrap());
    let n = 257;
    let mut rows = vec![0_i32];
    let mut columns = Vec::new();
    let mut values = Vec::new();
    let diagonal: Vec<_> = (0..n).map(|i| scalar(2.0 + (i % 3) as f64)).collect();
    for (i, &d) in diagonal.iter().enumerate() {
        if i > 0 {
            columns.push((i - 1) as i32);
            values.push(scalar(-0.1));
        }
        columns.push(i as i32);
        values.push(d);
        if i + 1 < n {
            columns.push((i + 1) as i32);
            values.push(scalar(-0.1));
        }
        rows.push(columns.len() as i32);
    }
    let matrix = DeviceCsr::try_from_host(backend, n, n, &rows, &columns, &values).unwrap();
    let polynomial =
        PolynomialJacobiBackend::try_new(backend, &matrix, &diagonal, 3, scalar(0.7)).unwrap();
    let chebyshev = ChebyshevBackend::try_with_lambda_max(
        backend,
        &matrix,
        &diagonal,
        3,
        T::Real::from_f64(1.5).unwrap(),
        T::Real::from_f64(3.0).unwrap(),
    )
    .unwrap();
    for preconditioner in [&polynomial as &dyn DevicePreconditioner<B, T>, &chebyshev] {
        let mut single_rhs = backend.raw_alloc::<T>(n).unwrap();
        let mut single_out = backend.raw_alloc::<T>(n).unwrap();
        let mut block_rhs = backend.raw_alloc::<T>(2 * n).unwrap();
        let mut block_out = backend.raw_alloc::<T>(2 * n).unwrap();
        backend
            .try_copy_h2d(&vec![scalar(2.0); n], &mut single_rhs)
            .unwrap();
        let mut rhs = vec![scalar(2.0); n];
        rhs.extend(vec![scalar(6.0); n]);
        backend.try_copy_h2d(&rhs, &mut block_rhs).unwrap();
        let mut baseline = None;
        for _ in 0..3 {
            preconditioner
                .try_apply_colmajor(backend, 2, &block_rhs, &mut block_out)
                .unwrap();
            preconditioner
                .try_apply(backend, &single_rhs, &mut single_out)
                .unwrap();
            preconditioner
                .try_apply_colmajor(backend, 1, &single_rhs, &mut single_out)
                .unwrap();
            preconditioner
                .try_apply_colmajor(backend, 2, &block_rhs, &mut block_out)
                .unwrap();
            if let (Some(before), Some(after)) = (baseline, allocations()) {
                assert_eq!(
                    before, after,
                    "scalar recovery evicted polynomial block scratch"
                );
            }
            baseline = allocations();
            let mut single = vec![T::zero(); n];
            let mut block = vec![T::zero(); 2 * n];
            backend.try_copy_d2h(&single_out, &mut single).unwrap();
            backend.try_copy_d2h(&block_out, &mut block).unwrap();
            for i in 0..n {
                for column in 0..2 {
                    let expected = single[i] * scalar(if column == 0 { 1.0 } else { 3.0 });
                    assert!(
                        (block[column * n + i] - expected).norm_sqr().sqrt()
                            < T::Real::from_f64(1e-5).unwrap()
                    );
                }
            }
        }
    }
}

#[test]
fn cpu_polynomials_alternate_scalar_and_block() {
    let backend = CpuBackend::new();
    check_polynomial_storage::<_, f32>(&backend, || None);
    check_polynomial_storage::<_, f64>(&backend, || None);
    check_polynomial_storage::<_, Complex32>(&backend, || None);
    check_polynomial_storage::<_, Complex64>(&backend, || None);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_polynomials_retain_scalar_and_block_storage() {
    let backend = match sonium_compute::CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            assert!(
                std::env::var_os("SONIUM_REQUIRE_CUDA").is_none(),
                "CUDA required: {error}"
            );
            return;
        }
    };
    let allocations = || Some(backend.runtime_stats().allocation_operations);
    check_polynomial_storage::<_, f32>(&backend, allocations);
    check_polynomial_storage::<_, f64>(&backend, allocations);
    check_polynomial_storage::<_, Complex32>(&backend, allocations);
    check_polynomial_storage::<_, Complex64>(&backend, allocations);
}
