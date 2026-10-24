//! Fused CSR residual and weighted Jacobi sweeps with immutable old iterates.

// Rust guideline compliant 2026-02-21
use super::*;

impl CudaBackend {
    pub(super) fn try_csr_jacobi_step<T: DeviceScalar>(
        &self,
        rows: &CudaBuffer<i32>,
        columns: &CudaBuffer<i32>,
        values: &CudaBuffer<T>,
        inverse: &CudaBuffer<T>,
        b: &CudaBuffer<T>,
        x: &CudaBuffer<T>,
        output: &mut CudaBuffer<T>,
        omega: T::Real,
    ) -> BackendResult<bool> {
        const OP: &str = "csr_jacobi_step";
        let invalid = |reason: &str| BackendError::InvalidArgument {
            operation: OP,
            reason: reason.into(),
        };
        self.check_pending_error()?;
        let n = x.len();
        if n.checked_add(1) != Some(rows.len())
            || columns.len() != values.len()
            || inverse.len() != n
            || b.len() != n
            || output.len() != n
        {
            return Err(invalid("inconsistent CSR/vector dimensions"));
        }
        let weight = num_traits::ToPrimitive::to_f64(&omega)
            .filter(|v| v.is_finite())
            .ok_or_else(|| invalid("nonfinite damping"))?;
        fn valid<T: BufferElement>(backend: &CudaBackend, buffer: &CudaBuffer<T>) -> bool {
            match buffer.raw() {
                Some(raw) => {
                    buffer.len().checked_mul(std::mem::size_of::<T>()) == Some(raw.len())
                        && Arc::ptr_eq(raw.stream(), &backend.stream)
                }
                None => buffer.len() == 0,
            }
        }
        if !valid(self, rows)
            || !valid(self, columns)
            || [values, inverse, b, x, &*output]
                .iter()
                .any(|v| !valid(self, v))
        {
            return Err(invalid("invalid allocation or foreign stream"));
        }
        if n == 0 {
            return Ok(true);
        }
        // Native CSR columns and row offsets are signed 32-bit indices.
        if n > i32::MAX as usize || values.len() > i32::MAX as usize || values.is_empty() {
            return Ok(false);
        }
        let Some(kernels) = &self.local_kernels else {
            return Ok(false);
        };
        let id = TypeId::of::<T>();
        let function = if id == TypeId::of::<f32>() {
            &kernels.csr_jacobi_f32
        } else if id == TypeId::of::<f64>() {
            &kernels.csr_jacobi_f64
        } else if id == TypeId::of::<Complex32>() {
            &kernels.csr_jacobi_c32
        } else if id == TypeId::of::<Complex64>() {
            &kernels.csr_jacobi_c64
        } else {
            return Ok(false);
        };
        self.try_custom_kernel_preflight()?;
        let count = n as u32;
        let entries = values.len() as u32;
        let rows = rows.raw().ok_or_else(|| invalid("missing rows"))?;
        let columns = columns.raw().ok_or_else(|| invalid("missing columns"))?;
        let values = values.raw().ok_or_else(|| invalid("missing values"))?;
        let inverse = inverse.raw().ok_or_else(|| invalid("missing inverse"))?;
        let b = b.raw().ok_or_else(|| invalid("missing RHS"))?;
        let x = x.raw().ok_or_else(|| invalid("missing iterate"))?;
        let output = output.raw_mut().ok_or_else(|| invalid("missing output"))?;
        self.record_custom_kernel_launch();
        // SAFETY: TypeId selects the ABI; extents and streams are checked. Four
        // lanes own a row, all lanes participate in shuffles, and every CSR index
        // is checked before dereference. Output cannot alias immutable inputs.
        unsafe {
            self.stream
                .launch_builder(function)
                .arg(rows)
                .arg(columns)
                .arg(values)
                .arg(inverse)
                .arg(b)
                .arg(x)
                .arg(output)
                .arg(&count)
                .arg(&entries)
                .arg(&weight)
                .launch(LaunchConfig {
                    grid_dim: (count.div_ceil(64), 1, 1),
                    block_dim: (256, 1, 1), // Four lanes per row, 64 rows per block.
                    shared_mem_bytes: 0,
                })
                .map_err(|error| self.custom_kernel_error(OP, format!("{error:?}")))?;
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryPoolConfig;

    #[test]
    fn native_fused_csr_jacobi_sweeps() {
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
            ($t:ty, $diag:expr, $off:expr, $tol:expr) => {{
                for n in [0, 1, 31, 63, 64, 65, 257, 2049, 16385] {
                    let diagonal: $t = $diag;
                    let off: $t = $off;
                    let mut rows = vec![0i32];
                    let mut cols = Vec::new();
                    let mut values = Vec::new();
                    for i in 0..n {
                        if i > 0 {
                            cols.push((i - 1) as i32);
                            values.push(off);
                        }
                        cols.push(i as i32);
                        values.push(diagonal);
                        if i + 1 < n {
                            cols.push((i + 1) as i32);
                            values.push(off);
                        }
                        rows.push(values.len() as i32);
                    }
                    let inverse = vec![<$t as DeviceScalar>::one() / diagonal; n];
                    let rhs = vec![diagonal; n];
                    let mut expected: Vec<$t> = (0..n)
                        .map(|i| if i % 2 == 0 { off } else { diagonal })
                        .collect();
                    let mut dr = pool.try_alloc::<i32>(rows.len()).unwrap();
                    let mut dc = pool.try_alloc::<i32>(cols.len()).unwrap();
                    let mut dv = pool.try_alloc::<$t>(values.len()).unwrap();
                    let mut di = pool.try_alloc::<$t>(n).unwrap();
                    let mut db = pool.try_alloc::<$t>(n).unwrap();
                    let mut dx = pool.try_alloc::<$t>(n).unwrap();
                    let mut out = pool.try_alloc::<$t>(n).unwrap();
                    backend.try_copy_h2d(&rows, dr.raw_mut()).unwrap();
                    backend.try_copy_h2d(&cols, dc.raw_mut()).unwrap();
                    backend.try_copy_h2d(&values, dv.raw_mut()).unwrap();
                    backend.try_copy_h2d(&inverse, di.raw_mut()).unwrap();
                    backend.try_copy_h2d(&rhs, db.raw_mut()).unwrap();
                    backend.try_copy_h2d(&expected, dx.raw_mut()).unwrap();
                    for _ in 0..3 {
                        let old = expected.clone();
                        let mut device_old = vec![<$t as DeviceScalar>::zero(); n];
                        backend.try_copy_d2h(dx.raw(), &mut device_old).unwrap();
                        let before = backend.begin_runtime_stats();
                        assert!(
                            backend
                                .try_csr_jacobi_step_native(
                                    dr.raw(),
                                    dc.raw(),
                                    dv.raw(),
                                    di.raw(),
                                    db.raw(),
                                    dx.raw(),
                                    out.raw_mut(),
                                    0.6
                                )
                                .unwrap()
                        );
                        backend.try_synchronize().unwrap();
                        let stats = backend.runtime_stats_since(before);
                        assert_eq!(stats.kernel_launches, usize::from(n > 0));
                        assert_eq!(stats.allocation_operations, 0);
                        assert_eq!(
                            stats.transfer.h2d_bytes
                                + stats.transfer.d2h_bytes
                                + stats.transfer.d2d_bytes,
                            0
                        );
                        for row in 0..n {
                            let mut ax = <$t as DeviceScalar>::zero();
                            for j in rows[row]..rows[row + 1] {
                                ax += values[j as usize] * old[cols[j as usize] as usize];
                            }
                            expected[row] = old[row] + inverse[row] * (rhs[row] - ax) * 0.6;
                        }
                        let mut unchanged = vec![<$t as DeviceScalar>::zero(); n];
                        backend.try_copy_d2h(dx.raw(), &mut unchanged).unwrap();
                        assert_eq!(unchanged, device_old);
                        let mut actual = unchanged;
                        backend.try_copy_d2h(out.raw(), &mut actual).unwrap();
                        for (a, b) in actual.iter().zip(&expected) {
                            assert!((*a - *b).magnitude() < $tol, "n={n}: {a:?} != {b:?}");
                        }
                        std::mem::swap(&mut dx, &mut out);
                    }
                }
            }};
        }
        check!(f32, 3.2, -0.5, 2e-6);
        check!(f64, 3.2, -0.5, 1e-13);
        check!(
            Complex32,
            Complex32::new(3.2, 0.25),
            Complex32::new(-0.5, 0.1),
            2e-6
        );
        check!(
            Complex64,
            Complex64::new(3.2, 0.25),
            Complex64::new(-0.5, 0.1),
            1e-13
        );
        let mut rows = pool.try_alloc::<i32>(4).unwrap();
        let mut cols = pool.try_alloc::<i32>(3).unwrap();
        let mut values = pool.try_alloc::<f64>(3).unwrap();
        let mut x = pool.try_alloc::<f64>(3).unwrap();
        let mut output = pool.try_alloc::<f64>(3).unwrap();
        backend.try_copy_h2d(&[0, 1, 2, 3], rows.raw_mut()).unwrap();
        backend.try_copy_h2d(&[0, -1, 5], cols.raw_mut()).unwrap();
        backend.try_copy_h2d(&[1.0; 3], values.raw_mut()).unwrap();
        backend.try_copy_h2d(&[2.0; 3], x.raw_mut()).unwrap();
        let run = |out: &mut CudaBuffer<f64>, weight| {
            backend.try_csr_jacobi_step_native(
                rows.raw(),
                cols.raw(),
                values.raw(),
                values.raw(),
                values.raw(),
                x.raw(),
                out,
                weight,
            )
        };
        assert!(run(output.raw_mut(), f64::NAN).is_err());
        assert!(run(output.raw_mut(), 0.5).unwrap());
        let mut actual = [0.0; 3];
        backend.try_copy_d2h(output.raw(), &mut actual).unwrap();
        assert!(actual[0].is_finite() && actual[1].is_nan() && actual[2].is_nan());
        backend.inject_failure(CudaFailureInjectionPoint::Launch);
        assert!(run(output.raw_mut(), 0.5).is_err());
        backend.recover_after_error().unwrap();
        assert!(run(output.raw_mut(), 0.5).unwrap());
        let mut bad_rows = pool.try_alloc::<i32>(4).unwrap();
        backend
            .try_copy_h2d(&[0, 1, 4, 3], bad_rows.raw_mut())
            .unwrap();
        assert!(
            backend
                .try_csr_jacobi_step_native(
                    bad_rows.raw(),
                    cols.raw(),
                    values.raw(),
                    values.raw(),
                    values.raw(),
                    x.raw(),
                    output.raw_mut(),
                    0.5
                )
                .unwrap()
        );
        backend.try_copy_d2h(output.raw(), &mut actual).unwrap();
        assert!(actual[1].is_nan() && actual[2].is_nan());
        let foreign = MemoryPool::new(
            CudaBackend::new_default().unwrap(),
            MemoryPoolConfig::default(),
        )
        .unwrap();
        let mut foreign_output = foreign.try_alloc::<f64>(3).unwrap();
        assert!(run(foreign_output.raw_mut(), 0.5).is_err());
        let wrong = pool.try_alloc::<f64>(4).unwrap();
        assert!(
            backend
                .try_csr_jacobi_step_native(
                    rows.raw(),
                    cols.raw(),
                    values.raw(),
                    values.raw(),
                    values.raw(),
                    wrong.raw(),
                    output.raw_mut(),
                    0.5
                )
                .is_err()
        );
    }
}
