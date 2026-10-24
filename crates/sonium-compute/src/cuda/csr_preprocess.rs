//! Optional retained cuSPARSE analysis for repeated fixed-pattern sparse actions.

// Rust guideline compliant 2026-02-21
use super::*;

pub(super) type CsrPreprocess = unsafe extern "C" fn(
    cusparse_sys::cusparseHandle_t,
    cusparse_sys::cusparseOperation_t,
    *const std::ffi::c_void,
    cusparse_sys::cusparseConstSpMatDescr_t,
    cusparse_sys::cusparseConstDnVecDescr_t,
    *const std::ffi::c_void,
    cusparse_sys::cusparseDnVecDescr_t,
    cusparse_sys::cudaDataType,
    cusparse_sys::cusparseSpMVAlg_t,
    *mut std::ffi::c_void,
) -> cusparse_sys::cusparseStatus_t;

impl CudaBackend {
    /// Enable retained CSR preprocessing before preparing persistent operators.
    ///
    /// Disabled by default pending workload-specific measurements. The first
    /// nonempty action preprocesses each descriptor once using its retained
    /// workspace; subsequent vector and numeric-value changes reuse that analysis.
    /// Sparsity changes require a new descriptor. No additional buffer is allocated.
    ///
    /// # Errors
    /// Rejects pending backend failures or a runtime without the preprocessing API.
    pub fn try_with_csr_preprocessing(mut self, enabled: bool) -> BackendResult<Self> {
        self.check_pending_error()?;
        self.csr_preprocess = if enabled {
            // SAFETY: cudarc retains this loaded library for process lifetime.
            // The signature matches NVIDIA's generic SpMV preprocessing C ABI.
            let symbol =
                unsafe { cusparse_sys::culib().get::<CsrPreprocess>(b"cusparseSpMV_preprocess\0") }
                    .map_err(|error| BackendError::OperationFailed {
                        operation: "csr_preprocessing_admission",
                        reason: format!("runtime lacks cusparseSpMV_preprocess: {error}"),
                    })?;
            Some(*symbol)
        } else {
            None
        };
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DeviceCsr, MemoryPoolConfig, PersistentDeviceCsr};

    #[test]
    fn native_csr_preprocess_reuses_analysis_after_numeric_updates() {
        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend.try_with_csr_preprocessing(true).unwrap(),
            Err(error) => {
                assert!(
                    std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                    "{error}"
                );
                return;
            }
        };
        let pool = MemoryPool::new(backend, MemoryPoolConfig::default()).unwrap();
        let backend = pool.backend();
        macro_rules! check {
            ($t:ty, $a:expr, $b:expr, $tol:expr) => {{
                let rows = [0, 2, 2, 5];
                let columns = [0, 3, 0, 1, 2];
                let mut values: Vec<$t> = vec![$a, $b, $b, $a, $b];
                let matrix =
                    DeviceCsr::try_from_host(backend, 3, 4, &rows, &columns, &values).unwrap();
                let mut operator =
                    PersistentDeviceCsr::try_from_device_csr(backend, matrix).unwrap();
                for step in 0..3 {
                    let host_x: Vec<$t> = vec![$a, $b, $a, $b];
                    // Distinct vectors verify safe descriptor retargeting.
                    let mut x = pool.try_alloc::<$t>(4).unwrap();
                    let mut y = pool.try_alloc::<$t>(3).unwrap();
                    backend.try_copy_h2d(&host_x, x.raw_mut()).unwrap();
                    if step > 0 {
                        for value in &mut values {
                            *value = *value + $b;
                        }
                        operator
                            .try_copy_values_from_host(backend, &values)
                            .unwrap();
                    }
                    if step == 0 {
                        backend.inject_failure(CudaFailureInjectionPoint::Launch);
                        assert!(operator.try_apply(backend, x.raw(), y.raw_mut()).is_err());
                        backend.recover_after_error().unwrap();
                    }
                    let before = backend.begin_runtime_stats();
                    operator.try_apply(backend, x.raw(), y.raw_mut()).unwrap();
                    backend.try_synchronize().unwrap();
                    let stats = backend.runtime_stats_since(before);
                    assert_eq!(stats.library_operations, if step == 0 { 2 } else { 1 });
                    assert_eq!(stats.allocation_operations, 0);
                    assert_eq!(stats.transfer.h2d_bytes + stats.transfer.d2h_bytes, 0);
                    let mut actual = vec![<$t as DeviceScalar>::zero(); 3];
                    backend.try_copy_d2h(y.raw(), &mut actual).unwrap();
                    for row in 0..3 {
                        let mut expected = <$t as DeviceScalar>::zero();
                        for j in rows[row]..rows[row + 1] {
                            expected = expected
                                + values[j as usize] * host_x[columns[j as usize] as usize];
                        }
                        assert!((actual[row] - expected).magnitude() < $tol);
                    }
                    let mut block_x = pool.try_alloc::<$t>(8).unwrap();
                    let mut block_y = pool.try_alloc::<$t>(6).unwrap();
                    backend
                        .try_copy_h2d(
                            &[host_x.as_slice(), host_x.as_slice()].concat(),
                            block_x.raw_mut(),
                        )
                        .unwrap();
                    operator
                        .try_apply_colmajor(backend, 2, block_x.raw(), block_y.raw_mut())
                        .unwrap();
                    let mut block_actual = vec![<$t as DeviceScalar>::zero(); 6];
                    backend
                        .try_copy_d2h(block_y.raw(), &mut block_actual)
                        .unwrap();
                    for (i, value) in block_actual.iter().enumerate() {
                        assert!((*value - actual[i % 3]).magnitude() < $tol);
                    }
                }
            }};
        }
        check!(f32, 2.0, -0.5, 1e-5);
        check!(f64, 2.0, -0.5, 1e-13);
        check!(
            Complex32,
            Complex32::new(2.0, 0.25),
            Complex32::new(-0.5, 0.5),
            1e-5
        );
        check!(
            Complex64,
            Complex64::new(2.0, 0.25),
            Complex64::new(-0.5, 0.5),
            1e-13
        );
    }
}
