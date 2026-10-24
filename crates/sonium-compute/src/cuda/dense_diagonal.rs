//! Checked native inverse-diagonal extraction for retained dense operators.

// Rust guideline compliant 2026-02-21
use super::*;

impl CudaBackend {
    /// Extract and invert a resident square matrix's diagonal.
    ///
    /// Accepts row-major or column-major storage with `leading_dimension >= n`.
    /// Both use the same diagonal offsets. Only the diagonal is validated;
    /// off-diagonal entries are not examined. No bulk host transfer occurs.
    /// The caller owns output storage. On error its contents are unspecified
    /// and must not be published as a valid preconditioner.
    ///
    /// # Errors
    /// Rejects inconsistent lengths, foreign streams, unsupported scalar types,
    /// unavailable kernels, non-finite or epsilon-small diagonal entries, and
    /// device failures. Native finite certification transfers one compact flag.
    pub fn try_dense_inverse_diagonal<T: DeviceScalar>(
        &self,
        matrix: &CudaBuffer<T>,
        n: usize,
        leading_dimension: usize,
        inverse: &mut CudaBuffer<T>,
    ) -> BackendResult<()> {
        const OP: &str = "dense_inverse_diagonal";
        let invalid = |reason: &str| BackendError::InvalidArgument {
            operation: OP,
            reason: reason.into(),
        };
        self.check_pending_error()?;
        if leading_dimension < n
            || n.checked_mul(leading_dimension) != Some(matrix.len())
            || inverse.len() != n
        {
            return Err(invalid("inconsistent matrix or inverse dimensions"));
        }
        for buffer in [matrix, &*inverse] {
            let bytes = buffer
                .len()
                .checked_mul(std::mem::size_of::<T>())
                .ok_or_else(|| invalid("buffer byte count overflow"))?;
            match buffer.raw() {
                Some(raw) if raw.len() == bytes && Arc::ptr_eq(raw.stream(), &self.stream) => {}
                None if buffer.len() == 0 => {}
                _ => return Err(invalid("invalid allocation or foreign stream")),
            }
        }
        let count = u32::try_from(n).map_err(|_| invalid("dimension exceeds CUDA index range"))?;
        let ld =
            u64::try_from(leading_dimension).map_err(|_| invalid("leading dimension overflow"))?;
        if n == 0 {
            return Ok(());
        }
        let kernels = self
            .local_kernels
            .as_ref()
            .ok_or_else(|| invalid("native local kernels unavailable"))?;
        if self.validation_kernels.is_none() {
            return Err(invalid("native validation unavailable"));
        }
        let id = TypeId::of::<T>();
        let function = if id == TypeId::of::<f32>() {
            &kernels.inverse_diagonal_f32
        } else if id == TypeId::of::<f64>() {
            &kernels.inverse_diagonal_f64
        } else if id == TypeId::of::<Complex32>() {
            &kernels.inverse_diagonal_c32
        } else if id == TypeId::of::<Complex64>() {
            &kernels.inverse_diagonal_c64
        } else {
            return Err(invalid("unsupported scalar type"));
        };
        self.try_custom_kernel_preflight()?;
        let source = matrix
            .raw()
            .ok_or_else(|| invalid("missing matrix storage"))?;
        let destination = inverse
            .raw_mut()
            .ok_or_else(|| invalid("missing inverse storage"))?;
        self.record_custom_kernel_launch();
        // SAFETY: TypeId selects the matching scalar ABI. Checked n*ld extents,
        // ld>=n and raw byte lengths bound every i*(ld+1) access. Both buffers
        // belong to this stream; exclusive output borrow prevents input aliasing.
        unsafe {
            self.stream
                .launch_builder(function)
                .arg(source)
                .arg(destination)
                .arg(&count)
                .arg(&ld)
                .launch(LaunchConfig::for_num_elems(count))
                .map_err(|error| self.custom_kernel_error(OP, format!("{error:?}")))?;
        }
        if self.try_all_finite_native(inverse)? != Some(true) {
            return Err(invalid("non-finite or epsilon-small diagonal"));
        }
        Ok(())
    }
}

/// Retained native point-Jacobi preconditioning for a changing dense matrix.
///
/// Inverse-diagonal storage is allocated through the caller's pool once.
/// Refresh validates the new diagonal on CUDA without downloading it. Any
/// failed refresh disables application until a subsequent successful refresh.
/// This is a point preconditioner, not an ILU or block inverse.
pub struct CudaDenseJacobi<'a, T: DeviceScalar> {
    backend: &'a CudaBackend,
    inverse: crate::Buffer<T, CudaBackend>,
    ready: bool,
}

impl<T: DeviceScalar> std::fmt::Debug for CudaDenseJacobi<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaDenseJacobi")
            .field("dimension", &self.inverse.len())
            .field("ready", &self.ready)
            .finish_non_exhaustive()
    }
}

impl<'a, T: DeviceScalar> CudaDenseJacobi<'a, T> {
    /// Allocate and prepare the inverse diagonal of a resident dense matrix.
    ///
    /// # Errors
    /// Rejects invalid dimensions, quota exhaustion and the same inputs or
    /// backend failures as `CudaBackend::try_dense_inverse_diagonal`.
    pub fn try_new(
        pool: &'a crate::MemoryPool<CudaBackend>,
        matrix: &CudaBuffer<T>,
        n: usize,
        leading_dimension: usize,
    ) -> BackendResult<Self> {
        if leading_dimension < n || n.checked_mul(leading_dimension) != Some(matrix.len()) {
            return Err(jacobi_error("inconsistent matrix dimensions"));
        }
        let mut result = Self {
            backend: pool.backend(),
            inverse: pool.try_alloc::<T>(n).map_err(BackendError::from)?,
            ready: false,
        };
        result.try_update(matrix, leading_dimension)?;
        Ok(result)
    }

    /// Refresh numeric values without replacing inverse-diagonal storage.
    ///
    /// # Errors
    /// Propagates diagonal validation and CUDA errors. Every failure disables
    /// application, including shape and stream errors, until a successful update.
    pub fn try_update(
        &mut self,
        matrix: &CudaBuffer<T>,
        leading_dimension: usize,
    ) -> BackendResult<()> {
        self.ready = false;
        self.backend.try_dense_inverse_diagonal(
            matrix,
            self.inverse.len(),
            leading_dimension,
            self.inverse.raw_mut(),
        )?;
        self.ready = true;
        Ok(())
    }
}

fn jacobi_error(reason: &str) -> BackendError {
    BackendError::InvalidArgument {
        operation: "cuda_dense_jacobi",
        reason: reason.into(),
    }
}

impl<T: DeviceScalar> crate::DevicePreconditioner<CudaBackend, T> for CudaDenseJacobi<'_, T> {
    fn apply(&self, backend: &CudaBackend, input: &CudaBuffer<T>, output: &mut CudaBuffer<T>) {
        if let Err(error) = self.try_apply(backend, input, output) {
            log::error!("native dense Jacobi failed: {error}");
        }
    }

    fn try_apply(
        &self,
        backend: &CudaBackend,
        input: &CudaBuffer<T>,
        output: &mut CudaBuffer<T>,
    ) -> BackendResult<()> {
        self.try_apply_colmajor(backend, 1, input, output)
    }

    fn apply_colmajor(
        &self,
        backend: &CudaBackend,
        rhs: usize,
        input: &CudaBuffer<T>,
        output: &mut CudaBuffer<T>,
    ) {
        if let Err(error) = self.try_apply_colmajor(backend, rhs, input, output) {
            log::error!("native dense block Jacobi failed: {error}");
        }
    }

    fn try_apply_colmajor(
        &self,
        backend: &CudaBackend,
        rhs: usize,
        input: &CudaBuffer<T>,
        output: &mut CudaBuffer<T>,
    ) -> BackendResult<()> {
        use num_traits::One;
        if !self.ready {
            return Err(jacobi_error("numeric refresh required"));
        }
        if !backend.try_diagonal_scale_colmajor_native(
            rhs,
            self.inverse.raw(),
            input,
            output,
            T::Real::one(),
        )? {
            return Err(jacobi_error("native diagonal scaling unavailable"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MemoryPool, MemoryPoolConfig};

    #[test]
    fn native_dense_jacobi_refresh_and_blocks() {
        use crate::DevicePreconditioner;
        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend,
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
                let mut values = vec![<$t>::default(); 6];
                values[0] = $a;
                values[4] = $b;
                let mut matrix = pool.try_alloc::<$t>(6).unwrap();
                let mut input = pool.try_alloc::<$t>(4).unwrap();
                let mut output = pool.try_alloc::<$t>(4).unwrap();
                let rhs = [$a, $b, $b, $a];
                backend.try_copy_h2d(&values, matrix.raw_mut()).unwrap();
                backend.try_copy_h2d(&rhs, input.raw_mut()).unwrap();
                let mut preconditioner =
                    CudaDenseJacobi::try_new(&pool, matrix.raw(), 2, 3).unwrap();
                for multiplier in [false, true, false] {
                    values[0] = if multiplier { $a + $a } else { $a };
                    values[4] = if multiplier { $b + $b } else { $b };
                    backend.try_copy_h2d(&values, matrix.raw_mut()).unwrap();
                    let before = backend.begin_runtime_stats();
                    preconditioner.try_update(matrix.raw(), 3).unwrap();
                    preconditioner
                        .try_apply_colmajor(backend, 2, input.raw(), output.raw_mut())
                        .unwrap();
                    backend.try_synchronize().unwrap();
                    let stats = backend.runtime_stats_since(before);
                    assert_eq!(stats.allocation_operations, 0);
                    assert_eq!(stats.transfer.h2d_bytes, 0);
                    let mut actual = [<$t>::default(); 4];
                    backend.try_copy_d2h(output.raw(), &mut actual).unwrap();
                    for i in 0..4 {
                        let expected = rhs[i] / values[(i % 2) * 4];
                        assert!(
                            <$t as DeviceScalar>::norm_sqr(&(actual[i] - expected)) < $tol * $tol
                        );
                    }
                }
                values[0] = <$t>::default();
                backend.try_copy_h2d(&values, matrix.raw_mut()).unwrap();
                assert!(preconditioner.try_update(matrix.raw(), 3).is_err());
                assert!(
                    preconditioner
                        .try_apply_colmajor(backend, 2, input.raw(), output.raw_mut())
                        .is_err()
                );
                values[0] = $a;
                backend.try_copy_h2d(&values, matrix.raw_mut()).unwrap();
                backend.inject_failure(crate::CudaFailureInjectionPoint::Launch);
                assert!(preconditioner.try_update(matrix.raw(), 3).is_err());
                backend.recover_after_error().unwrap();
                assert!(
                    preconditioner
                        .try_apply_colmajor(backend, 2, input.raw(), output.raw_mut())
                        .is_err()
                );
                preconditioner.try_update(matrix.raw(), 3).unwrap();
                preconditioner
                    .try_apply_colmajor(backend, 2, input.raw(), output.raw_mut())
                    .unwrap();
            }};
        }
        check!(f32, 2.0, -4.0, 1e-6);
        check!(f64, 2.0, -4.0, 1e-14);
        check!(
            Complex32,
            Complex32::new(2.0, 2.0),
            Complex32::new(0.0, -4.0),
            1e-6
        );
        check!(
            Complex64,
            Complex64::new(2.0, 2.0),
            Complex64::new(0.0, -4.0),
            1e-14
        );
    }

    #[test]
    fn native_dense_inverse_diagonal() {
        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend,
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
            ($t:ty, $a:expr, $b:expr, $ea:expr, $eb:expr, $tol:expr) => {{
                let mut values = vec![<$t>::default(); 6];
                values[0] = $a;
                values[4] = $b;
                let mut matrix = pool.try_alloc::<$t>(6).unwrap();
                let mut inverse = pool.try_alloc::<$t>(2).unwrap();
                backend.try_copy_h2d(&values, matrix.raw_mut()).unwrap();
                backend
                    .try_dense_inverse_diagonal(matrix.raw(), 2, 3, inverse.raw_mut())
                    .unwrap();
                let before = backend.begin_runtime_stats();
                backend
                    .try_dense_inverse_diagonal(matrix.raw(), 2, 3, inverse.raw_mut())
                    .unwrap();
                let stats = backend.runtime_stats_since(before);
                assert_eq!(stats.allocation_operations, 0);
                assert_eq!(stats.transfer.h2d_bytes, 0);
                let mut result = vec![<$t>::default(); 2];
                backend.try_copy_d2h(inverse.raw(), &mut result).unwrap();
                assert!(<$t as DeviceScalar>::norm_sqr(&(result[0] - $ea)) < $tol * $tol);
                assert!(<$t as DeviceScalar>::norm_sqr(&(result[1] - $eb)) < $tol * $tol);
                values[0] = <$t>::default();
                backend.try_copy_h2d(&values, matrix.raw_mut()).unwrap();
                assert!(
                    backend
                        .try_dense_inverse_diagonal(matrix.raw(), 2, 3, inverse.raw_mut())
                        .is_err()
                );
                assert!(
                    backend
                        .try_dense_inverse_diagonal(matrix.raw(), 2, 1, inverse.raw_mut())
                        .is_err()
                );
            }};
        }
        let mut matrix = pool.try_alloc::<Complex64>(1).unwrap();
        let mut inverse = pool.try_alloc::<Complex64>(1).unwrap();
        backend
            .try_copy_h2d(&[Complex64::new(1e300, 1e300)], matrix.raw_mut())
            .unwrap();
        backend
            .try_dense_inverse_diagonal(matrix.raw(), 1, 1, inverse.raw_mut())
            .unwrap();
        let mut output = [Complex64::default()];
        backend.try_copy_d2h(inverse.raw(), &mut output).unwrap();
        assert!((output[0].re / 5e-301 - 1.0).abs() < 1e-14);
        assert!((output[0].im / -5e-301 - 1.0).abs() < 1e-14);
        for invalid in [
            Complex64::new(f64::NAN, 0.0),
            Complex64::new(f64::INFINITY, 0.0),
            Complex64::new(f64::EPSILON, 0.0),
        ] {
            backend.try_copy_h2d(&[invalid], matrix.raw_mut()).unwrap();
            assert!(
                backend
                    .try_dense_inverse_diagonal(matrix.raw(), 1, 1, inverse.raw_mut())
                    .is_err()
            );
        }
        check!(f32, 2.0, -4.0, 0.5, -0.25, 1e-6);
        check!(f64, 2.0, -4.0, 0.5, -0.25, 1e-14);
        check!(
            Complex32,
            Complex32::new(2.0, 2.0),
            Complex32::new(0.0, -4.0),
            Complex32::new(0.25, -0.25),
            Complex32::new(0.0, 0.25),
            1e-6
        );
        check!(
            Complex64,
            Complex64::new(2.0, 2.0),
            Complex64::new(0.0, -4.0),
            Complex64::new(0.25, -0.25),
            Complex64::new(0.0, 0.25),
            1e-14
        );
    }
}
