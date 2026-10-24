//! Opt-in synchronized phase attribution for the CUDA FEM benchmark.
//!
//! These boundaries perturb execution and include launch/synchronization costs.
//! They are not CUDA event timings or a replacement for normal solve timings.

// Rust guideline compliant 2026-02-21
use num_complex::Complex64;
use serde_json::{Value, json};
use sonium_compute::{
    Backend, BackendResult, CudaBackend, CudaBuffer, DeviceAdjointCapability, DeviceOperator,
    DevicePreconditioner, DevicePreconditionerDiagnosticsSnapshot,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[derive(Default)]
struct Phase {
    calls: AtomicU64,
    nanoseconds: AtomicU64,
}

impl Phase {
    fn measure(
        &self,
        backend: &CudaBackend,
        action: impl FnOnce() -> BackendResult<()>,
    ) -> BackendResult<()> {
        // Drain earlier work before attributing this action. The outer solve
        // measurement retains that drain time in its unclassified remainder.
        backend.try_synchronize()?;
        let start = Instant::now();
        action()?;
        backend.try_synchronize()?;
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.nanoseconds.fetch_add(
            start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        Ok(())
    }

    fn milliseconds(&self) -> f64 {
        self.nanoseconds.load(Ordering::Relaxed) as f64 / 1e6
    }

    fn snapshot(&self) -> Value {
        json!({"calls": self.calls.load(Ordering::Relaxed), "synchronized_wall_ms": self.milliseconds()})
    }
}

pub(super) struct ProfileOperator<'a> {
    inner: &'a dyn DeviceOperator<CudaBackend, Complex64>,
    apply: Phase,
    residual: Phase,
}

impl<'a> ProfileOperator<'a> {
    pub(super) fn new(inner: &'a dyn DeviceOperator<CudaBackend, Complex64>) -> Self {
        Self {
            inner,
            apply: Phase::default(),
            residual: Phase::default(),
        }
    }
}

impl DeviceOperator<CudaBackend, Complex64> for ProfileOperator<'_> {
    fn n(&self) -> usize {
        self.inner.n()
    }

    fn apply(
        &self,
        backend: &CudaBackend,
        x: &CudaBuffer<Complex64>,
        y: &mut CudaBuffer<Complex64>,
    ) {
        if let Err(error) = self.try_apply(backend, x, y) {
            log::error!("profile operator failed: {error}");
        }
    }

    fn try_apply(
        &self,
        backend: &CudaBackend,
        x: &CudaBuffer<Complex64>,
        y: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        self.apply
            .measure(backend, || self.inner.try_apply(backend, x, y))
    }

    fn try_residual(
        &self,
        backend: &CudaBackend,
        x: &CudaBuffer<Complex64>,
        b: &CudaBuffer<Complex64>,
        out: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        // Preserve specialized residual implementations rather than rebuilding
        // a residual from apply/axpy and accidentally changing the measured path.
        self.residual
            .measure(backend, || self.inner.try_residual(backend, x, b, out))
    }
}

pub(super) struct ProfilePreconditioner<'a> {
    inner: &'a dyn DevicePreconditioner<CudaBackend, Complex64>,
    apply: Phase,
}

impl<'a> ProfilePreconditioner<'a> {
    pub(super) fn new(inner: &'a dyn DevicePreconditioner<CudaBackend, Complex64>) -> Self {
        Self {
            inner,
            apply: Phase::default(),
        }
    }

    pub(super) fn report(&self, operator: &ProfileOperator<'_>, total_ms: f64) -> Value {
        let accounted = self.apply.milliseconds()
            + operator.apply.milliseconds()
            + operator.residual.milliseconds();
        json!({
            "method":"synchronized_wall_boundaries",
            "perturbs_execution":true,
            "total_ms":total_ms,
            "operator":operator.apply.snapshot(),
            "residual":operator.residual.snapshot(),
            "preconditioner":self.apply.snapshot(),
            "other_solve_work_ms":total_ms-accounted,
            "other_includes":"Krylov algebra, reductions, control transfers, validation, host work and unattributed synchronization; not pure orthogonalization time"
        })
    }
}

impl DevicePreconditioner<CudaBackend, Complex64> for ProfilePreconditioner<'_> {
    fn apply(
        &self,
        backend: &CudaBackend,
        r: &CudaBuffer<Complex64>,
        z: &mut CudaBuffer<Complex64>,
    ) {
        if let Err(error) = self.try_apply(backend, r, z) {
            log::error!("profile preconditioner failed: {error}");
        }
    }

    fn try_apply(
        &self,
        backend: &CudaBackend,
        r: &CudaBuffer<Complex64>,
        z: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        self.apply
            .measure(backend, || self.inner.try_apply(backend, r, z))
    }

    fn forward_capability(&self) -> DeviceAdjointCapability {
        self.inner.forward_capability()
    }

    fn diagnostics(&self) -> DevicePreconditionerDiagnosticsSnapshot {
        self.inner.diagnostics()
    }
}
