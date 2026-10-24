//! Retained complex-double CUDA coarse factors for backend AMG.

// Rust guideline compliant 2026-02-21
use super::{
    AmgConfig, AmgPreconditioner, BackendAmgDirectSolver, BackendAmgPreconditioner,
    decode_backend_checkpoint,
};
use num_complex::Complex64;
use sonium_compute::{
    Backend, BackendError, BackendResult, CudaBackend, CudaBuffer, CudaDenseLuC64, OomError,
    Transpose,
};
use sonium_solvers_sparse::CsrMatrix;

struct CudaC64CoarseLu {
    lu: CudaDenseLuC64,
}

fn failure(operation: &'static str, error: OomError) -> BackendError {
    match error {
        OomError::Invalid(reason) => BackendError::OperationFailed { operation, reason },
        error => BackendError::Allocation {
            reason: format!("{operation}: {error}"),
        },
    }
}

impl BackendAmgDirectSolver<CudaBackend, Complex64> for CudaC64CoarseLu {
    fn supports_numeric_refresh(&self) -> bool {
        true
    }

    fn refresh(
        &mut self,
        backend: &CudaBackend,
        matrix: &CsrMatrix<Complex64>,
    ) -> BackendResult<()> {
        let dense = coarse_values(matrix)?;
        backend
            .dense_lu_c64_refactor_column_major(&mut self.lu, &dense)
            .map_err(|e| failure("cuda_amg_c64_coarse_refresh", e))
    }

    fn solve(
        &self,
        backend: &CudaBackend,
        b: &CudaBuffer<Complex64>,
        x: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<bool> {
        backend
            .dense_lu_c64_solve(&self.lu, b, x, Transpose::NoTrans)
            .map_err(|e| failure("cuda_amg_c64_coarse_solve", e))?;
        Ok(true)
    }
}

impl BackendAmgPreconditioner<CudaBackend, Complex64> {
    /// Build complex-double device AMG with retained cuSOLVER coarse factors.
    ///
    /// Hierarchy construction is host setup. Each subsequent cycle retains
    /// matrices, vectors and coarse factors on CUDA. No regularization or CPU
    /// solve fallback is implicit: singular coarse factorizations return errors.
    /// Raw vendor allocations follow the existing backend AMG allocation policy;
    /// this constructor does not promise MemoryPool quota accounting.
    ///
    /// # Errors
    /// Returns invalid hierarchy/configuration, allocation, or factorization errors.
    pub fn try_from_csr_with_cuda_coarse_lu(
        backend: &CudaBackend,
        matrix: &CsrMatrix<Complex64>,
        config: AmgConfig,
    ) -> BackendResult<Self> {
        let host = AmgPreconditioner::try_from_csr(matrix, config).map_err(|e| {
            BackendError::InvalidArgument {
                operation: "amg_setup",
                reason: e.to_string(),
            }
        })?;
        let coarse = build_coarse(backend, &host)?;
        Self::try_from_host_amg_with_direct(backend, &host, Some(coarse))
    }

    /// Restore complex-double device AMG and rebuild checkpointed coarse factors.
    ///
    /// Uses the portable hierarchy checkpoint, not serialized driver handles.
    /// Checkpoints without direct coarse state preserve that configuration.
    ///
    /// # Errors
    /// Rejects invalid/scalar-mismatched checkpoints and propagates setup failures.
    pub fn try_from_checkpoint_bytes_with_cuda_coarse_lu(
        backend: &CudaBackend,
        bytes: &[u8],
    ) -> BackendResult<Self> {
        let (payload, has_direct) = decode_backend_checkpoint::<CudaBackend, Complex64>(bytes)?
            .unwrap_or_else(|| (bytes.to_vec(), false));
        let host =
            AmgPreconditioner::<Complex64>::from_checkpoint_bytes(&payload).map_err(|e| {
                BackendError::InvalidArgument {
                    operation: "amg_checkpoint_restore",
                    reason: e.to_string(),
                }
            })?;
        let coarse = if has_direct {
            Some(build_coarse(backend, &host)?)
        } else {
            None
        };
        Self::try_from_host_amg_with_direct(backend, &host, coarse)
    }
}

fn build_coarse(
    backend: &CudaBackend,
    host: &AmgPreconditioner<Complex64>,
) -> BackendResult<Box<dyn BackendAmgDirectSolver<CudaBackend, Complex64>>> {
    backend.try_inject_preconditioner_failure(
        sonium_compute::PreconditionerFailureInjectionPoint::CoarseSolve,
    )?;
    let matrix = &host
        .levels
        .last()
        .ok_or_else(|| BackendError::InvalidArgument {
            operation: "amg_setup",
            reason: "missing coarse level".into(),
        })?
        .matrix;
    let dense = coarse_values(matrix)?;
    let lu = backend
        .dense_lu_c64_from_column_major(matrix.num_rows, &dense)
        .map_err(|e| failure("cuda_amg_c64_coarse_factor", e))?;
    Ok(Box::new(CudaC64CoarseLu { lu }))
}

fn coarse_values(matrix: &CsrMatrix<Complex64>) -> BackendResult<Vec<Complex64>> {
    let n = matrix.num_rows;
    let count = n
        .checked_mul(n)
        .filter(|_| n != 0 && n == matrix.num_cols && i32::try_from(n).is_ok())
        .ok_or_else(|| BackendError::InvalidArgument {
            operation: "amg_setup",
            reason: "invalid coarse matrix dimensions".into(),
        })?;
    let mut dense = vec![Complex64::default(); count];
    for row in 0..n {
        for i in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            dense[matrix.col_indices[i] * n + row] += matrix.values[i];
        }
    }
    Ok(dense)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_compute::{Backend, DevicePreconditioner};

    #[test]
    fn native_c64_coarse_checkpoint_and_repeated_action() {
        let backend = match CudaBackend::new_default() {
            Ok(b) => b,
            Err(e) => {
                assert!(
                    std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                    "CUDA required: {e}"
                );
                eprintln!("SKIP complex-double CUDA coarse AMG: {e}");
                return;
            }
        };
        let n = 12;
        let mut entries = Vec::new();
        for i in 0..n {
            entries.push((i, i, Complex64::new(3.0, 0.4)));
            if i + 1 < n {
                entries.push((i, i + 1, Complex64::new(-1.0, 0.1)));
                entries.push((i + 1, i, Complex64::new(-1.0, 0.1)));
            }
        }
        let matrix = CsrMatrix::from_triplets(n, n, entries);
        // One level makes the independent dense solve residual an exact check
        // of the coarse path, rather than an approximate-cycle convergence test.
        let config = AmgConfig {
            coarse_size: n,
            ..Default::default()
        };
        let amg =
            BackendAmgPreconditioner::<CudaBackend, Complex64>::try_from_csr_with_cuda_coarse_lu(
                &backend, &matrix, config,
            )
            .unwrap();
        assert!(amg.diagnostics().has_coarse_direct_solver);
        let checkpoint = amg.to_checkpoint_bytes().unwrap();
        assert!(
            BackendAmgPreconditioner::<CudaBackend, Complex64>::try_from_checkpoint_bytes(
                &backend,
                &checkpoint
            )
            .is_err()
        );
        let restored = BackendAmgPreconditioner::<CudaBackend,Complex64>::try_from_checkpoint_bytes_with_cuda_coarse_lu(&backend,&checkpoint).unwrap();
        assert!(restored.diagnostics().has_coarse_direct_solver);
        let mut rhs = backend.raw_alloc::<Complex64>(n).unwrap();
        let mut out = backend.raw_alloc::<Complex64>(n).unwrap();
        for source in 0..3 {
            let exact: Vec<_> = (0..n)
                .map(|i| Complex64::new((i + source) as f64 * 0.1, 0.2))
                .collect();
            let mut b = vec![Complex64::default(); n];
            for (row, v) in b.iter_mut().enumerate() {
                for i in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
                    *v += matrix.values[i] * exact[matrix.col_indices[i]];
                }
            }
            backend.try_copy_h2d(&b, &mut rhs).unwrap();
            for p in [&amg, &restored] {
                p.try_apply(&backend, &rhs, &mut out).unwrap();
                backend.try_synchronize().unwrap();
                let before = backend.runtime_stats();
                p.try_apply(&backend, &rhs, &mut out).unwrap();
                let after = backend.runtime_stats();
                assert_eq!(after.allocation_operations, before.allocation_operations);
                assert_eq!(after.transfer.h2d_bytes, before.transfer.h2d_bytes);
                assert_eq!(after.transfer.d2h_bytes, before.transfer.d2h_bytes);
                let mut actual = vec![Complex64::default(); n];
                backend.try_copy_d2h(&out, &mut actual).unwrap();
                for (a, b) in actual.iter().zip(&exact) {
                    assert!((*a - *b).norm() < 1e-11);
                }
            }
        }
    }
}

#[cfg(test)]
mod refresh_tests;
