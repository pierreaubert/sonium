//! Resident numeric quadrature assembly with fixed topology and real geometry weights.

// Rust guideline compliant 2026-02-21
use crate::cuda::{CudaBackend, CudaBuffer};
use crate::quadrature_assembly::{QuadratureAssemblyMap, invalid};
use crate::{
    Backend, BackendError, BackendResult, Buffer, BufferElement, DeviceBufferLike, MemoryPool,
};
use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use num_complex::Complex64;
use std::sync::Arc;

const OP: &str = "cuda_quadrature_assembly";
const SOURCE: &str = include_str!("kernels/quadrature_assembly.cu");

impl CudaBackend {
    /// Prepare and retain numeric quadrature kernels before map uploads.
    ///
    /// Reuses this backend's module. Driver module storage is outside pooled
    /// arrays; this call allocates no map buffers and transfers no samples.
    ///
    /// # Errors
    /// Rejects graph capture and propagates pending, compilation and symbol errors.
    pub fn try_prepare_quadrature_kernels(&self) -> BackendResult<()> {
        self.quadrature_function().map(|_| ())
    }

    fn quadrature_function(&self) -> BackendResult<Arc<CudaFunction>> {
        self.check_pending_error()?;
        if self.stream().capture_status().map_err(failed)?
            != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
        {
            return Err(invalid("prepare quadrature kernels before capture"));
        }
        let mut cache = self.quadrature_kernel.lock().map_err(failed)?;
        if let Some(kernel) = cache.as_ref() {
            return Ok(kernel.clone());
        }
        let ptx = crate::backend::catch_backend_panic(OP, || cudarc::nvrtc::compile_ptx(SOURCE))?
            .map_err(failed)?;
        let module = self.context().load_module(ptx).map_err(failed)?;
        let kernel = Arc::new(module.load_function("sonium_quadrature_assemble_c64").map_err(failed)?);
        *cache = Some(kernel.clone());
        Ok(kernel)
    }
}

fn failed(error: impl std::fmt::Debug) -> BackendError {
    BackendError::OperationFailed {
        operation: OP,
        reason: format!("{error:?}"),
    }
}
fn upload<T: BufferElement>(
    pool: &MemoryPool<CudaBackend>,
    values: &[T],
) -> BackendResult<Buffer<T, CudaBackend>> {
    let mut buffer = pool.try_alloc(values.len())?;
    pool.backend().try_copy_h2d(values, buffer.raw_mut())?;
    Ok(buffer)
}
fn raw<T: BufferElement>(buffer: &Buffer<T, CudaBackend>) -> BackendResult<&CudaSlice<u8>> {
    buffer
        .raw()
        .raw()
        .ok_or_else(|| invalid("missing map storage"))
}

/// Retained GPU reduction map for boundary matrix or load updates.
///
/// One launch assembles a nonempty output with no allocation or host transfer.
/// Coefficients are resident samples supplied by the caller. This performs
/// numeric integration/accumulation, not geometry setup or callback evaluation.
pub struct CudaQuadratureAssembly {
    pool: MemoryPool<CudaBackend>,
    samples: usize,
    outputs: usize,
    offsets: Buffer<u32, CudaBackend>,
    indices: Buffer<u32, CudaBackend>,
    weights: Buffer<f64, CudaBackend>,
    kernel: Arc<CudaFunction>,
}
impl std::fmt::Debug for CudaQuadratureAssembly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaQuadratureAssembly")
            .field("samples", &self.samples)
            .field("outputs", &self.outputs)
            .finish_non_exhaustive()
    }
}
impl CudaQuadratureAssembly {
    /// Upload a validated reduction map before repeated numeric updates.
    ///
    /// # Errors
    /// Rejects graph capture and propagates allocation, upload and JIT failures.
    pub fn try_new(
        pool: MemoryPool<CudaBackend>,
        map: &QuadratureAssemblyMap,
    ) -> BackendResult<Self> {
        let b = pool.backend();
        b.check_pending_error()?;
        if b.stream().capture_status().map_err(failed)?
            != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
        {
            return Err(invalid("prepare quadrature maps before capture"));
        }
        let kernel = b.quadrature_function()?;
        Ok(Self {
            samples: map.sample_count(),
            outputs: map.output_count(),
            offsets: upload(&pool, &map.offsets)?,
            // Sentinel storage supplies valid kernel pointers for maps with no terms.
            indices: upload(
                &pool,
                if map.indices.is_empty() {
                    &[0]
                } else {
                    &map.indices
                },
            )?,
            weights: upload(
                &pool,
                if map.weights.is_empty() {
                    &[0.0]
                } else {
                    &map.weights
                },
            )?,
            kernel,
            pool,
        })
    }

    /// Assemble resident coefficients into retained CSR values or a load vector.
    ///
    /// Output and input must have the map's exact lengths and owning stream.
    /// Non-finite coefficients propagate; callers must reject invalid physical
    /// data or validate the resulting solve. No implicit host validation occurs.
    /// Empty output maps are checked no-ops.
    ///
    /// # Errors
    /// Rejects shape, storage and stream mismatches before launch. Device failures
    /// preserve the backend's output and context recovery dispositions.
    pub fn try_assemble(
        &self,
        coefficients: &CudaBuffer<Complex64>,
        output: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        let b = self.pool.backend();
        for (buffer, len) in [(coefficients, self.samples), (&*output, self.outputs)] {
            if buffer.len() != len {
                return Err(invalid("quadrature buffer length mismatch"));
            }
            if len == 0 {
                continue;
            }
            let storage = buffer
                .raw()
                .ok_or_else(|| invalid("missing quadrature buffer storage"))?;
            let bytes = len
                .checked_mul(size_of::<Complex64>())
                .ok_or_else(|| invalid("quadrature byte overflow"))?;
            if storage.len() != bytes || !Arc::ptr_eq(storage.stream(), b.stream()) {
                return Err(invalid("quadrature buffer storage or stream mismatch"));
            }
        }
        b.check_pending_error()?;
        if self.outputs == 0 {
            return Ok(());
        }
        b.try_custom_kernel_preflight()?;
        let count = self.outputs as u32;
        let input = coefficients
            .raw()
            .ok_or_else(|| invalid("missing coefficients"))?;
        let output = output.raw_mut().ok_or_else(|| invalid("missing output"))?;
        b.record_custom_kernel_launch();
        // SAFETY: immutable map arrays were validated for sample/output bounds.
        // All caller storage lengths, byte sizes and stream identity were checked.
        // The ABI matches the included kernel; cudarc tracks stream dependencies.
        unsafe {
            b.stream()
                .launch_builder(&self.kernel)
                .arg(raw(&self.offsets)?)
                .arg(raw(&self.indices)?)
                .arg(raw(&self.weights)?)
                .arg(input)
                .arg(output)
                .arg(&count)
                .launch(LaunchConfig::for_num_elems(count))
                .map_err(|e| b.custom_kernel_error(OP, format!("{e:?}")))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_quadrature_updates_are_resident_and_checked() {
        let b = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                assert!(
                    std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                    "CUDA required: {e}"
                );
                eprintln!("SKIP CUDA quadrature assembly: {e}");
                return;
            }
        };
        let pool = MemoryPool::new(b, crate::MemoryPoolConfig::with_cap(1 << 22)).unwrap();
        let b = pool.backend();
        assert!(b.quadrature_kernel.lock().unwrap().is_none());
        let before = b.begin_runtime_stats();
        let first = b.quadrature_function().unwrap();
        b.try_prepare_quadrature_kernels().unwrap();
        let second = b.quadrature_function().unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        let stats = b.runtime_stats_since(before);
        assert_eq!(stats.allocation_operations, 0);
        assert_eq!(stats.kernel_launches + stats.library_operations, 0);
        assert_eq!(stats.transfer.h2d_bytes + stats.transfer.d2h_bytes, 0);
        let terms: Vec<_> = (0..259)
            .filter(|row| row % 7 != 0)
            .flat_map(|row| (0..13).map(move |q| (row, (q + row) % 17, (q as f64 - 5.0) * 0.125)))
            .collect();
        for outputs in [0, 259] {
            let map = QuadratureAssemblyMap::try_new(
                17,
                outputs,
                if outputs == 0 { &[] } else { &terms },
            )
            .unwrap();
            let device = CudaQuadratureAssembly::try_new(pool.clone(), &map).unwrap();
            let mut coefficients = pool.try_alloc::<Complex64>(17).unwrap();
            let mut out = pool.try_alloc::<Complex64>(outputs).unwrap();
            for update in 0..3 {
                let values: Vec<_> = (0..17)
                    .map(|q| Complex64::new((q + update) as f64 * 0.1, -(q as f64) * 0.3))
                    .collect();
                b.try_copy_h2d(&values, coefficients.raw_mut()).unwrap();
                let before = b.runtime_stats();
                device
                    .try_assemble(coefficients.raw(), out.raw_mut())
                    .unwrap();
                let after = b.runtime_stats();
                assert_eq!(after.allocation_operations, before.allocation_operations);
                assert_eq!(after.transfer.h2d_bytes, before.transfer.h2d_bytes);
                assert_eq!(after.transfer.d2h_bytes, before.transfer.d2h_bytes);
                assert_eq!(
                    after.kernel_launches - before.kernel_launches,
                    if outputs == 0 { 0 } else { 1 }
                );
                let mut actual = vec![Complex64::default(); outputs];
                b.try_copy_d2h(out.raw(), &mut actual).unwrap();
                for (got, want) in actual.iter().zip(map.try_assemble(&values).unwrap()) {
                    assert!((*got - want).norm() < 2e-12);
                }
            }
            let short = pool.try_alloc::<Complex64>(16).unwrap();
            let sentinel = vec![Complex64::new(17.0, -2.0); outputs];
            b.try_copy_h2d(&sentinel, out.raw_mut()).unwrap();
            assert!(device.try_assemble(short.raw(), out.raw_mut()).is_err());
            let mut actual = vec![Complex64::default(); outputs];
            b.try_copy_d2h(out.raw(), &mut actual).unwrap();
            assert_eq!(actual, sentinel);
        }
        let empty = QuadratureAssemblyMap::try_new(17, 3, &[]).unwrap();
        let device = CudaQuadratureAssembly::try_new(pool.clone(), &empty).unwrap();
        let mut coefficients = pool.try_alloc::<Complex64>(17).unwrap();
        b.try_copy_h2d(&[Complex64::new(f64::NAN, 0.0); 17], coefficients.raw_mut())
            .unwrap();
        let mut out = pool.try_alloc::<Complex64>(3).unwrap();
        device
            .try_assemble(coefficients.raw(), out.raw_mut())
            .unwrap();
        let mut actual = [Complex64::new(1.0, 0.0); 3];
        b.try_copy_d2h(out.raw(), &mut actual).unwrap();
        assert_eq!(actual, [Complex64::default(); 3]);
    }
}
