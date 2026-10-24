//! Fixed tetrahedral boundary setup and retained lifting storage.

// Rust guideline compliant 2026-02-21
use super::*;
use crate::physics::acoustics::pressure::device_boundary::prepare_tet_robin_quadratures;
use crate::spaces::FunctionSpace;
use sonium_compute::cuda::quadrature_assembly::CudaQuadratureAssembly;
use sonium_compute::operator_sum::DeviceSumWorkspace;
use sonium_compute::{Backend, PersistentDeviceCsr};

/// Host callbacks for a fixed marked Robin or Neumann boundary.
///
/// Integrates alpha*u*v and source*v. Zero alpha specifies a Neumann load.
pub struct TetRobinBoundary<'a> {
    /// Boundary marker in the mesh.
    pub marker: i32,
    /// Coefficient multiplying the boundary pressure.
    pub alpha: &'a dyn Fn(f64, f64, f64) -> Complex64,
    /// Prescribed natural boundary source.
    pub source: &'a dyn Fn(f64, f64, f64) -> Complex64,
}
impl std::fmt::Debug for TetRobinBoundary<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TetRobinBoundary")
            .field("marker", &self.marker)
            .finish_non_exhaustive()
    }
}

pub(super) struct TetBoundary<'a> {
    host: CsrMatrix<Complex64>,
    pub(super) valid: bool,
    changed: bool,
    pub(super) points: Vec<[f64; 3]>,
    pub(super) markers: Vec<i32>,
    action_map: CudaQuadratureAssembly,
    coupling_map: CudaQuadratureAssembly,
    load_map: CudaQuadratureAssembly,
    action_values: Buffer<Complex64, CudaBackend>,
    coupling_values: Buffer<Complex64, CudaBackend>,
    load_values: Buffer<Complex64, CudaBackend>,
    pub(super) action: PersistentDeviceCsr<CudaBackend, Complex64>,
    coupling: Option<PersistentDeviceCsr<CudaBackend, Complex64>>,
    load: Buffer<Complex64, CudaBackend>,
    scratch: Buffer<Complex64, CudaBackend>,
    pub(super) sum: DeviceSumWorkspace<'a, CudaBackend, Complex64>,
}

fn entries(a: &CsrMatrix<Complex64>, out: &mut Vec<(usize, usize, Complex64)>) {
    for row in 0..a.num_rows {
        for j in a.row_ptrs[row]..a.row_ptrs[row + 1] {
            out.push((row, a.col_indices[j], a.values[j]));
        }
    }
}

impl<'a> TetBoundary<'a> {
    pub(super) fn prepare(
        pool: &'a MemoryPool<CudaBackend>,
        mesh: &Mesh,
        space: &H1Space,
        config: &TetPressureConfig,
        boundaries: &[TetRobinBoundary<'_>],
    ) -> BackendResult<Option<Self>> {
        if boundaries.is_empty() {
            return Ok(None);
        }
        let n = space.num_global_dofs(mesh);
        let plan = prepare_tet_robin_quadratures(
            mesh,
            space,
            &boundaries.iter().map(|b| b.marker).collect::<Vec<_>>(),
            &config.dirichlet,
        )?;
        let mut alpha = Vec::with_capacity(plan.points().len());
        let mut source = Vec::with_capacity(plan.points().len());
        for (point, marker) in plan.points().iter().zip(plan.markers()) {
            let boundary = boundaries
                .iter()
                .find(|b| b.marker == *marker)
                .ok_or_else(|| invalid("missing boundary callback"))?;
            let a = (boundary.alpha)(point[0], point[1], point[2]);
            let g = (boundary.source)(point[0], point[1], point[2]);
            if [a.re, a.im, g.re, g.im].into_iter().any(|v| !v.is_finite()) {
                return Err(invalid("non-finite boundary callback"));
            }
            alpha.push(a);
            source.push(g);
        }
        let mut host = plan.action().template().clone();
        host.values = plan.action().values().try_assemble(&alpha)?;
        let mut coupling = plan.coupling().template().clone();
        coupling.values = plan.coupling().values().try_assemble(&alpha)?;
        let load = plan.load().try_assemble(&source)?;
        let action_values = pool
            .try_alloc(host.values.len())
            .map_err(BackendError::from)?;
        let coupling_values = pool
            .try_alloc(coupling.values.len())
            .map_err(BackendError::from)?;
        let action =
            PersistentDeviceCsr::try_from_host_csr_ref(pool.backend(), host.as_host_csr_ref())?;
        let coupling = if coupling.values.is_empty() {
            None
        } else {
            Some(PersistentDeviceCsr::try_from_host_csr_ref(
                pool.backend(),
                coupling.as_host_csr_ref(),
            )?)
        };
        let mut device_load = pool.try_alloc(n).map_err(BackendError::from)?;
        pool.backend().try_copy_h2d(&load, device_load.raw_mut())?;
        Ok(Some(Self {
            host,
            valid: true,
            changed: false,
            points: plan.points().to_vec(),
            markers: plan.markers().to_vec(),
            action_map: CudaQuadratureAssembly::try_new(pool.clone(), plan.action().values())?,
            coupling_map: CudaQuadratureAssembly::try_new(pool.clone(), plan.coupling().values())?,
            load_map: CudaQuadratureAssembly::try_new(pool.clone(), plan.load())?,
            action_values,
            coupling_values,
            load_values: pool.try_alloc(n).map_err(BackendError::from)?,
            action,
            coupling,
            load: device_load,
            scratch: pool.try_alloc(n).map_err(BackendError::from)?,
            sum: DeviceSumWorkspace::try_new(pool, n, &[])?,
        }))
    }

    pub(super) fn add_to(
        &self,
        volume: &CsrMatrix<Complex64>,
        backend: &CudaBackend,
    ) -> BackendResult<CsrMatrix<Complex64>> {
        let mut terms = Vec::with_capacity(volume.values.len() + self.host.values.len());
        entries(volume, &mut terms);
        if !self.valid {
            return Err(invalid("boundary update is incomplete"));
        }
        if self.changed {
            let mut current = self.host.clone();
            backend.try_copy_d2h(self.action_values.raw(), &mut current.values)?;
            entries(&current, &mut terms);
        } else {
            entries(&self.host, &mut terms);
        }
        CsrMatrix::try_from_triplets(volume.num_rows, volume.num_cols, terms)
            .map_err(|e| invalid(e.to_string()))
    }

    pub(super) fn update(
        &mut self,
        backend: &CudaBackend,
        alpha: &CudaBuffer<Complex64>,
        source: &CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        self.valid = false;
        self.action_map
            .try_assemble(alpha, self.action_values.raw_mut())?;
        self.coupling_map
            .try_assemble(alpha, self.coupling_values.raw_mut())?;
        self.load_map
            .try_assemble(source, self.load_values.raw_mut())?;
        self.action
            .try_copy_values_from_device(backend, self.action_values.raw())?;
        if let Some(coupling) = &mut self.coupling {
            coupling.try_copy_values_from_device(backend, self.coupling_values.raw())?;
        }
        backend.try_copy_d2d(self.load_values.raw(), self.load.raw_mut())?;
        self.changed = true;
        self.valid = true;
        Ok(())
    }

    pub(super) fn lift(
        &mut self,
        backend: &CudaBackend,
        prescribed: &CudaBuffer<Complex64>,
        lifted: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        // Boundary load and coupling have zero constrained rows; identity rows
        // supplied by volume lifting remain unchanged, including intersections.
        backend.try_axpy(Complex64::new(1.0, 0.0), self.load.raw(), lifted)?;
        if let Some(coupling) = &self.coupling {
            coupling.try_apply(backend, prescribed, self.scratch.raw_mut())?;
            backend.try_axpy(Complex64::new(-1.0, 0.0), self.scratch.raw(), lifted)?;
        }
        Ok(())
    }
}
