//! Opt-in single-block Jacobi sweeps for small resident CSR systems.

// Rust guideline compliant 2026-02-21
use super::*;

impl CudaBackend {
    /// Apply simultaneous weighted Jacobi sweeps in one CUDA block.
    ///
    /// Supports square CSR systems up to 256 rows and up to 32 sweeps. Returns
    /// false without changing `x` outside that range or when kernels/types are
    /// unavailable. Zero sweeps are a no-op. The caller supplies inverse diagonal
    /// values; no diagonal extraction, allocation or host transfer occurs.
    /// Malformed resident CSR indices produce an all-NaN iterate without unsafe
    /// access. Nonfinite arithmetic propagates; certify convergence separately.
    ///
    /// # Errors
    /// Rejects inconsistent lengths, byte extents, foreign streams, nonfinite
    /// weights and CUDA failures. On backend failure, output is unspecified.
    pub fn try_small_csr_jacobi<T: DeviceScalar>(
        &self,
        rows: &CudaBuffer<i32>,
        columns: &CudaBuffer<i32>,
        values: &CudaBuffer<T>,
        inverse: &CudaBuffer<T>,
        b: &CudaBuffer<T>,
        x: &mut CudaBuffer<T>,
        omega: T::Real,
        sweeps: usize,
    ) -> BackendResult<bool> {
        const OP: &str = "small_csr_jacobi";
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
        {
            return Err(invalid("inconsistent CSR or vector dimensions"));
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
            || [values, inverse, b, &*x].iter().any(|v| !valid(self, v))
        {
            return Err(invalid("invalid allocation or foreign stream"));
        }
        if n == 0 || sweeps == 0 {
            return Ok(true);
        }
        // A single block holds the iterate; cap sweep work to keep launch latency bounded.
        if n > 256 || sweeps > 32 || values.is_empty() || values.len() > i32::MAX as usize {
            return Ok(false);
        }
        let Some(kernels) = self.local_kernels.as_ref() else {
            return Ok(false);
        };
        let id = TypeId::of::<T>();
        let function = if id == TypeId::of::<f32>() {
            &kernels.small_jacobi_f32
        } else if id == TypeId::of::<f64>() {
            &kernels.small_jacobi_f64
        } else if id == TypeId::of::<Complex32>() {
            &kernels.small_jacobi_c32
        } else if id == TypeId::of::<Complex64>() {
            &kernels.small_jacobi_c64
        } else {
            return Ok(false);
        };
        self.try_custom_kernel_preflight()?;
        let rows = rows.raw().ok_or_else(|| invalid("missing rows"))?;
        let columns = columns.raw().ok_or_else(|| invalid("missing columns"))?;
        let values = values.raw().ok_or_else(|| invalid("missing values"))?;
        let inverse = inverse.raw().ok_or_else(|| invalid("missing inverse"))?;
        let b = b.raw().ok_or_else(|| invalid("missing RHS"))?;
        let nnz = columns.len() / std::mem::size_of::<i32>();
        let x = x.raw_mut().ok_or_else(|| invalid("missing output"))?;
        let count = n as u32;
        let entries = nnz as u32;
        let sweeps = sweeps as u32;
        self.record_custom_kernel_launch();
        // SAFETY: TypeId selects the scalar ABI; all extents and streams match.
        // One block with four lanes per row owns x; whole warps include padding.
        // At most 1024 threads are launched. The kernel checks every CSR offset
        // and column before dereferencing, and barriers separate iterate reads/writes.
        unsafe {
            self.stream
                .launch_builder(function)
                .arg(rows)
                .arg(columns)
                .arg(values)
                .arg(inverse)
                .arg(b)
                .arg(x)
                .arg(&count)
                .arg(&entries)
                .arg(&sweeps)
                .arg(&weight)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: ((count * 4).div_ceil(32) * 32, 1, 1),
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
    use crate::{CudaFailureInjectionPoint, MemoryPool, MemoryPoolConfig};

    #[test]
    fn native_small_csr_jacobi() {
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
            ($t:ty, $diagonal:expr, $off:expr, $w:expr, $tol:expr) => {{
                for n in [1, 31, 32, 33, 157, 256, 257] {
                    let mut rows = vec![0i32];
                    let mut columns = Vec::new();
                    let mut values = Vec::new();
                    for i in 0..n {
                        if i > 0 {
                            columns.push((i - 1) as i32);
                            values.push($off);
                        }
                        columns.push(i as i32);
                        values.push($diagonal);
                        if i + 1 < n {
                            columns.push((i + 1) as i32);
                            values.push($off);
                        }
                        rows.push(columns.len() as i32);
                    }
                    let inverse = vec![<$t as DeviceScalar>::one() / $diagonal; n];
                    let rhs: Vec<$t> = (0..n)
                        .map(|i| if i % 2 == 0 { $diagonal } else { $off })
                        .collect();
                    let mut dr = pool.try_alloc::<i32>(rows.len()).unwrap();
                    let mut dc = pool.try_alloc::<i32>(columns.len()).unwrap();
                    let mut da = pool.try_alloc::<$t>(values.len()).unwrap();
                    let mut di = pool.try_alloc::<$t>(n).unwrap();
                    let mut db = pool.try_alloc::<$t>(n).unwrap();
                    let mut dx = pool.try_alloc::<$t>(n).unwrap();
                    backend.try_copy_h2d(&rows, dr.raw_mut()).unwrap();
                    backend.try_copy_h2d(&columns, dc.raw_mut()).unwrap();
                    backend.try_copy_h2d(&values, da.raw_mut()).unwrap();
                    backend.try_copy_h2d(&inverse, di.raw_mut()).unwrap();
                    backend.try_copy_h2d(&rhs, db.raw_mut()).unwrap();
                    for zero_start in [true, false] {
                        for sweeps in [0, 1, 2, 5, 32, 33] {
                            let initial = if zero_start {
                                vec![<$t>::default(); n]
                            } else {
                                rhs.clone()
                            };
                            backend.try_copy_h2d(&initial, dx.raw_mut()).unwrap();
                            let before = backend.begin_runtime_stats();
                            let used = backend
                                .try_small_csr_jacobi(
                                    dr.raw(),
                                    dc.raw(),
                                    da.raw(),
                                    di.raw(),
                                    db.raw(),
                                    dx.raw_mut(),
                                    $w,
                                    sweeps,
                                )
                                .unwrap();
                            let stats = backend.runtime_stats_since(before);
                            assert_eq!(stats.allocation_operations, 0);
                            assert_eq!(stats.transfer.h2d_bytes, 0);
                            assert_eq!(stats.transfer.d2h_bytes, 0);
                            let supported = sweeps == 0 || (n <= 256 && sweeps <= 32);
                            assert_eq!(used, supported);
                            assert_eq!(stats.kernel_launches, usize::from(supported && sweeps > 0));
                            let mut expected = initial;
                            if supported {
                                for _ in 0..sweeps {
                                    let old = expected.clone();
                                    for i in 0..n {
                                        let mut sum = <$t>::default();
                                        for j in rows[i] as usize..rows[i + 1] as usize {
                                            sum += values[j] * old[columns[j] as usize];
                                        }
                                        expected[i] = old[i] + inverse[i] * (rhs[i] - sum) * $w;
                                    }
                                }
                            }
                            let mut actual = vec![<$t>::default(); n];
                            backend.try_copy_d2h(dx.raw(), &mut actual).unwrap();
                            for (a, e) in actual.iter().zip(expected) {
                                assert!(
                                    <$t as DeviceScalar>::norm_sqr(&(*a - e)) < $tol * $tol,
                                    "n={n} sweeps={sweeps}"
                                );
                            }
                        }
                    }
                    if n == 33 {
                        // Corrupt offsets must never permit out-of-bounds reads,
                        // including negative, descending and oversized ranges.
                        for (index, offset) in [(0, 1), (1, -1), (2, 0), (2, i32::MAX), (n, 0)] {
                            let mut malformed = rows.clone();
                            malformed[index] = offset;
                            backend.try_copy_h2d(&malformed, dr.raw_mut()).unwrap();
                            backend.try_copy_h2d(&rhs, dx.raw_mut()).unwrap();
                            assert!(
                                backend
                                    .try_small_csr_jacobi(
                                        dr.raw(),
                                        dc.raw(),
                                        da.raw(),
                                        di.raw(),
                                        db.raw(),
                                        dx.raw_mut(),
                                        $w,
                                        2
                                    )
                                    .unwrap()
                            );
                            assert_eq!(
                                backend.try_all_finite_native(dx.raw()).unwrap(),
                                Some(false)
                            );
                        }
                        backend.try_copy_h2d(&rows, dr.raw_mut()).unwrap();
                        columns[0] = n as i32;
                        backend.try_copy_h2d(&columns, dc.raw_mut()).unwrap();
                        assert!(
                            backend
                                .try_small_csr_jacobi(
                                    dr.raw(),
                                    dc.raw(),
                                    da.raw(),
                                    di.raw(),
                                    db.raw(),
                                    dx.raw_mut(),
                                    $w,
                                    2
                                )
                                .unwrap()
                        );
                        assert_eq!(
                            backend.try_all_finite_native(dx.raw()).unwrap(),
                            Some(false)
                        );
                        columns[0] = 0;
                        backend.try_copy_h2d(&columns, dc.raw_mut()).unwrap();
                        backend.inject_failure(CudaFailureInjectionPoint::Launch);
                        assert!(
                            backend
                                .try_small_csr_jacobi(
                                    dr.raw(),
                                    dc.raw(),
                                    da.raw(),
                                    di.raw(),
                                    db.raw(),
                                    dx.raw_mut(),
                                    $w,
                                    2
                                )
                                .is_err()
                        );
                        backend.recover_after_error().unwrap();
                        backend.try_copy_h2d(&rhs, dx.raw_mut()).unwrap();
                        assert!(
                            backend
                                .try_small_csr_jacobi(
                                    dr.raw(),
                                    dc.raw(),
                                    da.raw(),
                                    di.raw(),
                                    db.raw(),
                                    dx.raw_mut(),
                                    $w,
                                    2
                                )
                                .unwrap()
                        );
                        assert_eq!(backend.try_all_finite_native(dx.raw()).unwrap(), Some(true));
                    }
                }
            }};
        }
        check!(f32, 2.0f32, -0.2f32, 0.6f32, 2e-6f32);
        check!(f64, 2.0f64, -0.2f64, 0.6f64, 2e-14f64);
        check!(
            Complex32,
            Complex32::new(2.0, 0.2),
            Complex32::new(-0.2, 0.1),
            0.6f32,
            2e-6f32
        );
        check!(
            Complex64,
            Complex64::new(2.0, 0.2),
            Complex64::new(-0.2, 0.1),
            0.6f64,
            2e-14f64
        );
    }
}
