//! Fused resident numeric combinations for fixed-pattern FEM updates.

// Rust guideline compliant 2026-02-21
use super::*;

impl CudaBackend {
    /// Select fused numeric combinations before constructing retained solver sessions.
    ///
    /// Disabled by default: measured numeric-update savings have not established
    /// a consistent complete-solve benefit. When disabled, the optional hook
    /// returns false and callers retain device copy/scale/AXPY operations.
    /// This supports controlled comparisons and workload-specific tuning without
    /// changing solver, preconditioner, precision or host/device boundaries.
    #[must_use]
    pub fn with_fused_numeric_combinations(mut self, enabled: bool) -> Self {
        self.fused_numeric_combinations = enabled;
        self
    }

    pub(super) fn try_linear_combination2<T: DeviceScalar>(
        &self,
        alpha: T,
        x: &CudaBuffer<T>,
        beta: T,
        y: &CudaBuffer<T>,
        output: &mut CudaBuffer<T>,
    ) -> BackendResult<bool> {
        const OP: &str = "linear_combination2";
        let invalid = |reason: &str| BackendError::InvalidArgument {
            operation: OP,
            reason: reason.into(),
        };
        self.check_pending_error()?;
        let n = output.len();
        if x.len() != n || y.len() != n || !alpha.is_finite() || !beta.is_finite() {
            return Err(invalid("inconsistent lengths or nonfinite coefficients"));
        }
        for buffer in [x, y, &*output] {
            let valid = match buffer.raw() {
                Some(raw) => {
                    n.checked_mul(std::mem::size_of::<T>()) == Some(raw.len())
                        && Arc::ptr_eq(raw.stream(), &self.stream)
                }
                None => n == 0,
            };
            if !valid {
                return Err(invalid("invalid allocation or foreign stream"));
            }
        }
        if n == 0 {
            return Ok(true);
        }
        if !self.fused_numeric_combinations {
            return Ok(false);
        }
        let count = u32::try_from(n).map_err(|_| invalid("length exceeds CUDA kernel limit"))?;
        let Some(kernels) = &self.local_kernels else {
            return Ok(false);
        };
        // Coefficients use a uniform four-double ABI; f32 values convert exactly.
        fn parts<T: DeviceScalar>(value: &T) -> Option<(f64, f64)> {
            let value = value as &dyn std::any::Any;
            if let Some(v) = value.downcast_ref::<f32>() {
                Some((f64::from(*v), 0.0))
            } else if let Some(v) = value.downcast_ref::<f64>() {
                Some((*v, 0.0))
            } else if let Some(v) = value.downcast_ref::<Complex32>() {
                Some((f64::from(v.re), f64::from(v.im)))
            } else {
                value.downcast_ref::<Complex64>().map(|v| (v.re, v.im))
            }
        }
        let (Some((ar, ai)), Some((br, bi))) = (parts(&alpha), parts(&beta)) else {
            return Ok(false);
        };
        let id = TypeId::of::<T>();
        let function = if id == TypeId::of::<f32>() {
            &kernels.combination_f32
        } else if id == TypeId::of::<f64>() {
            &kernels.combination_f64
        } else if id == TypeId::of::<Complex32>() {
            &kernels.combination_c32
        } else {
            &kernels.combination_c64
        };
        self.try_custom_kernel_preflight()?;
        let x = x.raw().ok_or_else(|| invalid("missing x"))?;
        let y = y.raw().ok_or_else(|| invalid("missing y"))?;
        let output = output.raw_mut().ok_or_else(|| invalid("missing output"))?;
        self.record_custom_kernel_launch();
        // SAFETY: TypeId selects the buffer ABI; all byte extents and streams
        // were checked. Each bounded lane writes one output, whose mutable borrow
        // excludes input aliasing. The coefficient ABI is always four doubles.
        unsafe {
            self.stream
                .launch_builder(function)
                .arg(x)
                .arg(y)
                .arg(output)
                .arg(&count)
                .arg(&ar)
                .arg(&ai)
                .arg(&br)
                .arg(&bi)
                .launch(LaunchConfig::for_num_elems(count))
                .map_err(|error| self.custom_kernel_error(OP, format!("{error:?}")))?;
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MemoryPool, MemoryPoolConfig};

    #[test]
    fn native_linear_combination2_disabled_preserves_output() {
        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend.with_fused_numeric_combinations(false),
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
        let x = pool.try_alloc::<f64>(2).unwrap();
        let y = pool.try_alloc::<f64>(2).unwrap();
        let mut output = pool.try_alloc::<f64>(2).unwrap();
        backend.try_copy_h2d(&[7.0, 8.0], output.raw_mut()).unwrap();
        let before = backend.begin_runtime_stats();
        assert!(
            !backend
                .try_linear_combination2_native(1.0, x.raw(), 2.0, y.raw(), output.raw_mut())
                .unwrap()
        );
        let stats = backend.runtime_stats_since(before);
        assert_eq!(stats.kernel_launches, 0);
        assert_eq!(stats.allocation_operations, 0);
        assert_eq!(
            stats.transfer.h2d_bytes + stats.transfer.d2h_bytes + stats.transfer.d2d_bytes,
            0
        );
        let mut actual = [0.0; 2];
        backend.try_copy_d2h(output.raw(), &mut actual).unwrap();
        assert_eq!(actual, [7.0, 8.0]);
    }

    #[test]
    fn native_linear_combination2() {
        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend.with_fused_numeric_combinations(true),
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
                for n in [0, 1, 31, 256, 257, 100003] {
                    let a: $t = $a;
                    let b: $t = $b;
                    let x: Vec<$t> = (0..n).map(|i| if i % 2 == 0 { a } else { b }).collect();
                    let y: Vec<$t> = (0..n).map(|i| if i % 3 == 0 { b } else { a }).collect();
                    let mut dx = pool.try_alloc::<$t>(n).unwrap();
                    let mut dy = pool.try_alloc::<$t>(n).unwrap();
                    let mut out = pool.try_alloc::<$t>(n).unwrap();
                    backend.try_copy_h2d(&x, dx.raw_mut()).unwrap();
                    backend.try_copy_h2d(&y, dy.raw_mut()).unwrap();
                    let before = backend.begin_runtime_stats();
                    assert!(
                        backend
                            .try_linear_combination2_native(a, dx.raw(), b, dy.raw(), out.raw_mut())
                            .unwrap()
                    );
                    backend.try_synchronize().unwrap();
                    let stats = backend.runtime_stats_since(before);
                    assert_eq!(stats.allocation_operations, 0);
                    assert_eq!(stats.transfer.h2d_bytes, 0);
                    assert_eq!(stats.transfer.d2h_bytes, 0);
                    assert_eq!(stats.transfer.d2d_bytes, 0);
                    assert_eq!(stats.kernel_launches, usize::from(n > 0));
                    let mut actual = vec![<$t as DeviceScalar>::zero(); n];
                    backend.try_copy_d2h(out.raw(), &mut actual).unwrap();
                    for i in 0..n {
                        assert!((actual[i] - (a * x[i] + b * y[i])).magnitude() < $tol);
                    }
                    let wrong = pool.try_alloc::<$t>(n + 1).unwrap();
                    assert!(
                        backend
                            .try_linear_combination2_native(
                                a,
                                wrong.raw(),
                                b,
                                dy.raw(),
                                out.raw_mut()
                            )
                            .is_err()
                    );
                }
            }};
        }
        check!(f32, 1.25, -0.75, 1e-6);
        check!(f64, 1.25, -0.75, 1e-14);
        check!(
            Complex32,
            Complex32::new(1.25, 0.5),
            Complex32::new(-0.75, 0.25),
            1e-6
        );
        check!(
            Complex64,
            Complex64::new(1.25, 0.5),
            Complex64::new(-0.75, 0.25),
            1e-14
        );
        let mut x = pool.try_alloc::<f64>(1).unwrap();
        let y = pool.try_alloc::<f64>(1).unwrap();
        let mut out = pool.try_alloc::<f64>(1).unwrap();
        assert!(
            backend
                .try_linear_combination2_native(f64::NAN, x.raw(), 1.0, y.raw(), out.raw_mut())
                .is_err()
        );
        let foreign = MemoryPool::new(
            CudaBackend::new_default().unwrap(),
            MemoryPoolConfig::default(),
        )
        .unwrap();
        let foreign_y = foreign.try_alloc::<f64>(1).unwrap();
        assert!(
            backend
                .try_linear_combination2_native(1.0, x.raw(), 1.0, foreign_y.raw(), out.raw_mut())
                .is_err()
        );
        backend.inject_failure(crate::CudaFailureInjectionPoint::Launch);
        assert!(
            backend
                .try_linear_combination2_native(1.0, x.raw(), 1.0, y.raw(), out.raw_mut())
                .is_err()
        );
        backend.recover_after_error().unwrap();
        // A zero coefficient must not conceal nonfinite input from later validation.
        backend.try_copy_h2d(&[f64::NAN], x.raw_mut()).unwrap();
        assert!(
            backend
                .try_linear_combination2_native(0.0, x.raw(), 1.0, y.raw(), out.raw_mut())
                .unwrap()
        );
        let mut result = [0.0];
        backend.try_copy_d2h(out.raw(), &mut result).unwrap();
        assert!(result[0].is_nan());
    }
}
