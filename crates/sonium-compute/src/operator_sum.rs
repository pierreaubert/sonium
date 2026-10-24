//! Retained sums of device operators for volume and boundary contributions.

// Rust guideline compliant 2026-02-21
use crate::pool::Buffer;
use crate::{
    Backend, BackendError, BackendResult, DeviceAdjointOperator, DeviceBufferLike, DeviceOperator,
    DeviceScalar, MemoryPool,
};
use std::sync::{Arc, Mutex};

struct Workspace<B: Backend, T: DeviceScalar> {
    rhs: usize,
    first: Buffer<T, B>,
    second: Buffer<T, B>,
}

/// Sum two square device operators with scratch allocated during setup.
///
/// Scalar application is always prepared; `rhs_counts` selects additional
/// column-major block sizes. No allocation or host transfer is introduced by
/// this wrapper during application. Each child must itself provide native
/// actions for the requested sizes to keep the whole sum device resident.
/// The caller must use the exact backend borrowed from the setup pool.
/// Output is published only after both child actions and accumulation succeed;
/// a final device-copy failure follows the backend's output disposition.
pub struct DeviceSum<
    'a,
    B: Backend,
    T: DeviceScalar,
    A: DeviceOperator<B, T> + ?Sized,
    C: DeviceOperator<B, T> + ?Sized,
> {
    pool: &'a MemoryPool<B>,
    left: &'a A,
    right: &'a C,
    dimension: usize,
    workspace: Arc<Mutex<Vec<Workspace<B, T>>>>,
}
fn invalid(reason: impl Into<String>) -> BackendError {
    BackendError::InvalidArgument {
        operation: "device_sum",
        reason: reason.into(),
    }
}
impl<
    'a,
    B: Backend,
    T: DeviceScalar,
    A: DeviceOperator<B, T> + ?Sized,
    C: DeviceOperator<B, T> + ?Sized,
> DeviceSum<'a, B, T, A, C>
{
    /// Prepare scalar and selected block-RHS sums using pooled scratch buffers.
    ///
    /// # Errors
    /// Rejects incompatible dimensions, zero block counts, overflow or quota
    /// exhaustion. Partial setup allocations are returned to the pool.
    pub fn try_new(
        pool: &'a MemoryPool<B>,
        left: &'a A,
        right: &'a C,
        rhs_counts: &[usize],
    ) -> BackendResult<Self> {
        DeviceSumWorkspace::try_new(pool, left.n(), rhs_counts)?.try_bind(left, right)
    }

    fn run(
        &self,
        backend: &B,
        rhs: usize,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
        alpha: T,
        action: impl FnOnce(&mut B::Buffer<T>, &mut B::Buffer<T>) -> BackendResult<()>,
    ) -> BackendResult<()> {
        let len = self
            .dimension
            .checked_mul(rhs)
            .ok_or_else(|| invalid("RHS size overflow"))?;
        if !std::ptr::eq(backend, self.pool.backend()) || x.len() != len || y.len() != len {
            return Err(invalid("backend or vector shape mismatch"));
        }
        backend.check_pending_error()?;
        if rhs == 0 {
            return Ok(());
        }
        let mut workspaces =
            self.workspace
                .try_lock()
                .map_err(|_| BackendError::OperationFailed {
                    operation: "device_sum",
                    reason: "workspace is already in use or poisoned".into(),
                })?;
        let w = workspaces
            .iter_mut()
            .find(|w| w.rhs == rhs)
            .ok_or_else(|| invalid("RHS count was not prepared"))?;
        action(w.first.raw_mut(), w.second.raw_mut())?;
        backend.try_axpy(alpha, w.second.raw(), w.first.raw_mut())?;
        backend.try_copy_d2d(w.first.raw(), y)
    }
}
/// Retained sum scratch that can be rebound to caller-owned operators.
///
/// Binding validates dimensions but performs no allocation or transfer. Views
/// share one mutex-protected scratch allocation. Concurrent or recursive actions
/// on that allocation return an error instead of waiting. Retain it alongside operators to
/// avoid self-referential storage in higher-level solve sessions.
pub struct DeviceSumWorkspace<'a, B: Backend, T: DeviceScalar> {
    pool: &'a MemoryPool<B>,
    dimension: usize,
    workspace: Arc<Mutex<Vec<Workspace<B, T>>>>,
}

impl<B: Backend, T: DeviceScalar> std::fmt::Debug for DeviceSumWorkspace<'_, B, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceSumWorkspace")
            .field("dimension", &self.dimension)
            .finish_non_exhaustive()
    }
}

impl<'a, B: Backend, T: DeviceScalar> DeviceSumWorkspace<'a, B, T> {
    /// Allocate scalar and selected block sum scratch from the supplied pool.
    ///
    /// # Errors
    /// Rejects zero dimensions/counts, overflow and quota or backend failures.
    pub fn try_new(
        pool: &'a MemoryPool<B>,
        dimension: usize,
        rhs_counts: &[usize],
    ) -> BackendResult<Self> {
        if dimension == 0 || rhs_counts.contains(&0) {
            return Err(invalid("zero dimension or RHS count"));
        }
        let mut counts = vec![1];
        counts.extend_from_slice(rhs_counts);
        counts.sort_unstable();
        counts.dedup();
        let mut workspace = Vec::with_capacity(counts.len());
        for rhs in counts {
            let len = dimension
                .checked_mul(rhs)
                .ok_or_else(|| invalid("RHS size overflow"))?;
            workspace.push(Workspace {
                rhs,
                first: pool.try_alloc(len)?,
                second: pool.try_alloc(len)?,
            });
        }
        Ok(Self {
            pool,
            dimension,
            workspace: Arc::new(Mutex::new(workspace)),
        })
    }

    /// Bind two operators to the retained scratch without allocating.
    ///
    /// # Errors
    /// Rejects operators whose dimensions differ from the prepared workspace.
    pub fn try_bind<'b, A, C>(
        &self,
        left: &'b A,
        right: &'b C,
    ) -> BackendResult<DeviceSum<'b, B, T, A, C>>
    where
        'a: 'b,
        A: DeviceOperator<B, T> + ?Sized,
        C: DeviceOperator<B, T> + ?Sized,
    {
        if left.n() != self.dimension || right.n() != self.dimension {
            return Err(invalid(
                "operator dimensions differ from prepared workspace",
            ));
        }
        Ok(DeviceSum {
            pool: self.pool,
            left,
            right,
            dimension: self.dimension,
            workspace: Arc::clone(&self.workspace),
        })
    }
}

impl<
    B: Backend,
    T: DeviceScalar,
    A: DeviceOperator<B, T> + ?Sized,
    C: DeviceOperator<B, T> + ?Sized,
> DeviceOperator<B, T> for DeviceSum<'_, B, T, A, C>
{
    fn n(&self) -> usize {
        self.dimension
    }
    fn apply(&self, backend: &B, x: &B::Buffer<T>, y: &mut B::Buffer<T>) {
        if let Err(e) = self.try_apply(backend, x, y) {
            log::error!("device sum failed: {e}");
        }
    }
    fn try_apply(&self, backend: &B, x: &B::Buffer<T>, y: &mut B::Buffer<T>) -> BackendResult<()> {
        self.try_apply_colmajor(backend, 1, x, y)
    }
    fn try_apply_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> BackendResult<()> {
        self.run(backend, rhs, x, y, T::one(), |a, b| {
            if rhs == 1 {
                self.left.try_apply(backend, x, a)?;
                self.right.try_apply(backend, x, b)
            } else {
                self.left.try_apply_colmajor(backend, rhs, x, a)?;
                self.right.try_apply_colmajor(backend, rhs, x, b)
            }
        })
    }
    fn try_residual(
        &self,
        backend: &B,
        x: &B::Buffer<T>,
        b: &B::Buffer<T>,
        out: &mut B::Buffer<T>,
    ) -> BackendResult<()> {
        if b.len() != self.dimension {
            return Err(invalid("residual RHS shape mismatch"));
        }
        self.run(backend, 1, x, out, -T::one(), |a, c| {
            self.left.try_residual(backend, x, b, a)?;
            self.right.try_apply(backend, x, c)
        })
    }
}
impl<
    B: Backend,
    T: DeviceScalar,
    A: DeviceAdjointOperator<B, T> + ?Sized,
    C: DeviceAdjointOperator<B, T> + ?Sized,
> DeviceAdjointOperator<B, T> for DeviceSum<'_, B, T, A, C>
{
    fn try_apply_hermitian(
        &self,
        backend: &B,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> BackendResult<()> {
        self.try_apply_hermitian_colmajor(backend, 1, x, y)
    }
    fn try_apply_hermitian_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> BackendResult<()> {
        self.run(backend, rhs, x, y, T::one(), |a, b| {
            if rhs == 1 {
                self.left.try_apply_hermitian(backend, x, a)?;
                self.right.try_apply_hermitian(backend, x, b)
            } else {
                self.left.try_apply_hermitian_colmajor(backend, rhs, x, a)?;
                self.right.try_apply_hermitian_colmajor(backend, rhs, x, b)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AdjointDeviceCsr, CpuBackend, HostCsrRef, MemoryPoolConfig};
    use num_complex::Complex64 as C;

    fn exercise<B: Backend>(backend: B, counters: impl Fn(&B) -> [u64; 3]) {
        let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let b = pool.backend();
        let a = [
            C::new(2.0, 0.4),
            C::new(-1.0, 0.2),
            C::new(0.3, -0.1),
            C::new(3.0, 0.5),
        ];
        let c = [
            C::new(0.7, 0.1),
            C::new(0.2, 0.3),
            C::new(-0.4, 0.2),
            C::new(0.6, -0.2),
        ];
        let a_device = AdjointDeviceCsr::try_from_host_csr_ref(
            b,
            HostCsrRef::try_new(2, 2, &[0, 2, 4], &[0, 1, 0, 1], &a).unwrap(),
        )
        .unwrap();
        let c_device = AdjointDeviceCsr::try_from_host_csr_ref(
            b,
            HostCsrRef::try_new(2, 2, &[0, 2, 4], &[0, 1, 0, 1], &c).unwrap(),
        )
        .unwrap();
        let left = a_device.try_prebake(b).unwrap();
        let right = c_device.try_prebake(b).unwrap();
        let scratch = DeviceSumWorkspace::try_new(&pool, 2, &[3, 3]).unwrap();
        let sum = scratch.try_bind(&left, &right).unwrap();
        for rhs in [1, 3] {
            let input: Vec<_> = (0..2 * rhs)
                .map(|i| C::new(0.3 + i as f64 * 0.2, -0.4))
                .collect();
            let mut x = pool.try_alloc(input.len()).unwrap();
            let mut y = pool.try_alloc(input.len()).unwrap();
            b.try_copy_h2d(&input, x.raw_mut()).unwrap();
            for adjoint in [false, true] {
                for _ in 0..2 {
                    if adjoint {
                        sum.try_apply_hermitian_colmajor(b, rhs, x.raw(), y.raw_mut())
                            .unwrap();
                    } else {
                        sum.try_apply_colmajor(b, rhs, x.raw(), y.raw_mut())
                            .unwrap();
                    }
                }
                let before = counters(b);
                let sum = scratch.try_bind(&left, &right).unwrap();
                if adjoint {
                    sum.try_apply_hermitian_colmajor(b, rhs, x.raw(), y.raw_mut())
                        .unwrap();
                } else {
                    sum.try_apply_colmajor(b, rhs, x.raw(), y.raw_mut())
                        .unwrap();
                }
                let after = counters(b);
                assert_eq!(after, before);
                let mut actual = vec![C::default(); input.len()];
                b.try_copy_d2h(y.raw(), &mut actual).unwrap();
                for col in 0..rhs {
                    for row in 0..2 {
                        let expected: C = (0..2)
                            .map(|j| {
                                let v = if adjoint {
                                    (a[2 * j + row] + c[2 * j + row]).conj()
                                } else {
                                    a[2 * row + j] + c[2 * row + j]
                                };
                                v * input[2 * col + j]
                            })
                            .sum();
                        assert!((actual[2 * col + row] - expected).norm() < 1e-12);
                    }
                }
            }
            if rhs == 1 {
                sum.try_residual(b, x.raw(), x.raw(), y.raw_mut()).unwrap();
                let mut actual = vec![C::default(); 2];
                b.try_copy_d2h(y.raw(), &mut actual).unwrap();
                for row in 0..2 {
                    let expected = input[row]
                        - (0..2)
                            .map(|j| (a[2 * row + j] + c[2 * row + j]) * input[j])
                            .sum::<C>();
                    assert!((actual[row] - expected).norm() < 1e-12);
                }
            }
        }
        let mut x = pool.try_alloc::<C>(4).unwrap();
        let mut y = pool.try_alloc::<C>(4).unwrap();
        let sentinel = vec![C::new(17.0, 2.0); 4];
        b.try_copy_h2d(&sentinel, x.raw_mut()).unwrap();
        b.try_copy_h2d(&sentinel, y.raw_mut()).unwrap();
        assert!(sum.try_apply_colmajor(b, 2, x.raw(), y.raw_mut()).is_err());
        let mut actual = sentinel.clone();
        b.try_copy_d2h(y.raw(), &mut actual).unwrap();
        assert_eq!(actual, sentinel);
    }

    #[test]
    fn retained_sum_forward_adjoint_blocks_and_residual() {
        exercise(CpuBackend, |_| [0; 3]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn native_retained_sum() {
        match crate::CudaBackend::new_default() {
            Ok(b) => exercise(b, |backend| {
                let s = backend.runtime_stats();
                [
                    s.allocation_operations as u64,
                    s.transfer.h2d_bytes as u64,
                    s.transfer.d2h_bytes as u64,
                ]
            }),
            Err(e) => {
                assert!(
                    std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                    "CUDA required: {e}"
                );
                eprintln!("SKIP CUDA operator sum: {e}");
            }
        }
    }

    struct Failing;
    impl DeviceOperator<CpuBackend, f64> for Failing {
        fn n(&self) -> usize {
            2
        }
        fn apply(
            &self,
            _: &CpuBackend,
            _: &<CpuBackend as Backend>::Buffer<f64>,
            _: &mut <CpuBackend as Backend>::Buffer<f64>,
        ) {
            unreachable!()
        }
        fn try_apply(
            &self,
            b: &CpuBackend,
            x: &<CpuBackend as Backend>::Buffer<f64>,
            y: &mut <CpuBackend as Backend>::Buffer<f64>,
        ) -> BackendResult<()> {
            b.try_copy_d2d(x, y)?;
            Err(invalid("injected child failure"))
        }
    }
    #[test]
    fn child_failure_does_not_publish_partial_sum() {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1024)).unwrap();
        let left = crate::DeviceCsr::try_from_host_diagonal(pool.backend(), &[2.0, 3.0]).unwrap();
        let sum = DeviceSum::try_new(&pool, &left, &Failing, &[]).unwrap();
        let mut x = pool.try_alloc::<f64>(2).unwrap();
        let mut y = pool.try_alloc::<f64>(2).unwrap();
        pool.backend()
            .try_copy_h2d(&[1.0, 2.0], x.raw_mut())
            .unwrap();
        pool.backend()
            .try_copy_h2d(&[17.0, 19.0], y.raw_mut())
            .unwrap();
        assert!(sum.try_apply(pool.backend(), x.raw(), y.raw_mut()).is_err());
        let mut actual = vec![0.0; 2];
        pool.backend().try_copy_d2h(y.raw(), &mut actual).unwrap();
        assert_eq!(actual, vec![17.0, 19.0]);
    }

    #[test]
    fn recursive_workspace_use_fails_without_publishing_and_can_recover() {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1024)).unwrap();
        let b = pool.backend();
        let diagonal = crate::DeviceCsr::try_from_host_diagonal(b, &[2.0, 3.0]).unwrap();
        let scratch = DeviceSumWorkspace::try_new(&pool, 2, &[]).unwrap();
        let inner = scratch.try_bind(&diagonal, &diagonal).unwrap();
        let outer = scratch.try_bind(&inner, &diagonal).unwrap();
        let mut x = pool.try_alloc::<f64>(2).unwrap();
        let mut y = pool.try_alloc::<f64>(2).unwrap();
        b.try_copy_h2d(&[1.0, 2.0], x.raw_mut()).unwrap();
        b.try_copy_h2d(&[17.0, 19.0], y.raw_mut()).unwrap();
        assert!(outer.try_apply(b, x.raw(), y.raw_mut()).is_err());
        let mut actual = [0.0; 2];
        b.try_copy_d2h(y.raw(), &mut actual).unwrap();
        assert_eq!(actual, [17.0, 19.0]);
        inner.try_apply(b, x.raw(), y.raw_mut()).unwrap();
        b.try_copy_d2h(y.raw(), &mut actual).unwrap();
        assert_eq!(actual, [4.0, 12.0]);
    }
}
