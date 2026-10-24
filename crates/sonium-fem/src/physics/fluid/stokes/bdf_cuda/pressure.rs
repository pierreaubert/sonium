//! Pressure AMG with optional native restriction and extension of BDF vectors.

// Rust guideline compliant 2026-02-21
use sonium_compute::{
    Backend, BackendError, BackendResult, Buffer, BufferView, CudaBackend, CudaBuffer,
    DeviceBufferLike, DevicePreconditioner, MemoryPool,
};
use sonium_solvers::{
    AmgConfig, AmgPreconditioner, AmgSmoother, CsrMatrix, preconditioners::BackendAmgPreconditioner,
};
use std::sync::{Arc, Mutex};

struct PressureVectors {
    input: Buffer<f64, CudaBackend>,
    output: Buffer<f64, CudaBackend>,
    extended: Buffer<f64, CudaBackend>,
}

struct PressureState {
    action: BackendAmgPreconditioner<CudaBackend, f64>,
    n: usize,
    nv: usize,
    vectors: Option<PressureVectors>,
}

#[derive(Clone)]
pub(super) struct PressureAmg(Arc<Mutex<PressureState>>);

fn host(
    n: usize,
    nv: usize,
    pressure: &CsrMatrix<f64>,
    restricted: bool,
) -> Result<AmgPreconditioner<f64>, String> {
    let embedded = (!restricted).then(|| {
        let (rows, columns, values) =
            sonium_compute::preconditioner::embed_pressure_schur_with_velocity_identity(
                n,
                nv,
                pressure.as_host_csr_ref(),
            );
        CsrMatrix::from_raw_parts(n, n, rows, columns, values)
    });
    AmgPreconditioner::try_from_csr(
        embedded.as_ref().unwrap_or(pressure),
        AmgConfig {
            smoother: AmgSmoother::Chebyshev,
            num_pre_smooth: 2,
            num_post_smooth: 1,
            ..Default::default()
        },
    )
    .map_err(|e| e.to_string())
}

impl PressureAmg {
    pub(super) fn new(
        pool: &MemoryPool<CudaBackend>,
        n: usize,
        nv: usize,
        pressure: &CsrMatrix<f64>,
        restricted: bool,
    ) -> Result<Self, String> {
        Self::new_with_layout(pool, n, nv, pressure, restricted)
    }

    #[cfg(test)]
    pub(super) fn new_restricted(
        pool: &MemoryPool<CudaBackend>,
        n: usize,
        nv: usize,
        pressure: &CsrMatrix<f64>,
    ) -> Result<Self, String> {
        Self::new_with_layout(pool, n, nv, pressure, true)
    }

    fn new_with_layout(
        pool: &MemoryPool<CudaBackend>,
        n: usize,
        nv: usize,
        pressure: &CsrMatrix<f64>,
        restricted: bool,
    ) -> Result<Self, String> {
        if nv > n || pressure.num_rows != n - nv || pressure.num_cols != n - nv {
            return Err("pressure AMG dimensions do not match BDF field layout".into());
        }
        let action = BackendAmgPreconditioner::try_from_host_amg(
            pool.backend(),
            &host(n, nv, pressure, restricted)?,
        )
        .map_err(|e| e.to_string())?;
        let vectors = if restricted {
            let mut extended = pool.try_alloc(n).map_err(|e| e.to_string())?;
            if !pool
                .backend()
                .try_zero_native::<f64>(extended.raw_mut())
                .map_err(|e| e.to_string())?
            {
                return Err("native pressure extension initialization is unavailable".into());
            }
            Some(PressureVectors {
                input: pool.try_alloc(n - nv).map_err(|e| e.to_string())?,
                output: pool.try_alloc(n - nv).map_err(|e| e.to_string())?,
                extended,
            })
        } else {
            None
        };
        Ok(Self(Arc::new(Mutex::new(PressureState {
            action,
            n,
            nv,
            vectors,
        }))))
    }

    pub(super) fn refresh(
        &mut self,
        backend: &CudaBackend,
        n: usize,
        nv: usize,
        pressure: &CsrMatrix<f64>,
    ) -> Result<bool, String> {
        let mut state = self
            .0
            .lock()
            .map_err(|_| "pressure AMG mutex poisoned".to_string())?;
        if state.n != n || state.nv != nv {
            return Ok(false);
        }
        let candidate = host(n, nv, pressure, state.vectors.is_some())?;
        state
            .action
            .try_refresh_from_host_amg(backend, &candidate)
            .map_err(|e| e.to_string())
    }

    #[cfg(test)]
    pub(super) fn diagnostics(&self) -> sonium_solvers::preconditioners::BackendAmgDiagnostics {
        self.0.lock().unwrap().action.diagnostics()
    }
}

fn invalid(reason: &str) -> BackendError {
    BackendError::InvalidArgument {
        operation: "bdf_pressure_amg",
        reason: reason.into(),
    }
}

impl DevicePreconditioner<CudaBackend, f64> for PressureAmg {
    fn apply(&self, backend: &CudaBackend, rhs: &CudaBuffer<f64>, output: &mut CudaBuffer<f64>) {
        if let Err(error) = self.try_apply(backend, rhs, output) {
            log::error!("BDF pressure AMG failed: {error}");
        }
    }

    fn try_apply(
        &self,
        backend: &CudaBackend,
        rhs: &CudaBuffer<f64>,
        output: &mut CudaBuffer<f64>,
    ) -> BackendResult<()> {
        self.try_apply_colmajor(backend, 1, rhs, output)
    }

    fn try_apply_colmajor(
        &self,
        backend: &CudaBackend,
        count: usize,
        rhs: &CudaBuffer<f64>,
        output: &mut CudaBuffer<f64>,
    ) -> BackendResult<()> {
        let mut state = self.0.lock().map_err(|_| BackendError::OperationFailed {
            operation: "bdf_pressure_amg",
            reason: "pressure AMG mutex poisoned".into(),
        })?;
        let PressureState {
            action,
            n,
            nv,
            vectors,
        } = &mut *state;
        let len = n
            .checked_mul(count)
            .ok_or_else(|| invalid("block vector length overflow"))?;
        if rhs.len() != len || output.len() != len {
            return Err(invalid("block vectors do not match BDF field layout"));
        }
        let Some(PressureVectors {
            input,
            output: pressure_output,
            extended,
        }) = vectors.as_mut()
        else {
            // The enclosing Schur action masks velocity entries, preserving the
            // original small-system embedded action without adapter copies.
            return action.try_apply_colmajor(backend, count, rhs, output);
        };
        let np = *n - *nv;
        for column in 0..count {
            backend.try_copy_views_d2d(
                rhs,
                BufferView::contiguous(column * *n + *nv, np),
                input.raw_mut(),
                BufferView::contiguous(0, np),
            )?;
            action.try_apply(backend, input.raw(), pressure_output.raw_mut())?;
            // Velocity slots were zeroed at setup and are never written again.
            // Every pressure slot is replaced before publishing this column.
            backend.try_copy_views_d2d(
                pressure_output.raw(),
                BufferView::contiguous(0, np),
                extended.raw_mut(),
                BufferView::contiguous(*nv, np),
            )?;
            backend.try_copy_views_d2d(
                extended.raw(),
                BufferView::contiguous(0, *n),
                output,
                BufferView::contiguous(column * *n, *n),
            )?;
        }
        Ok(())
    }
}
