//! Retained CUDA pressure solves with hex volume, boundary and auxiliary AMG state.
//!
//! Setup is host assembly followed by device upload. Repeated scalar solves
//! retain all operators and scratch; only Krylov control scalars cross the host
//! boundary. Mesh and constraints are fixed. Wavenumber changes reuse geometry;
//! optional retained boundary maps support changing Robin/source samples on device.
//! Resident weighted volume coefficients can also be replaced. Updates retain
//! AMG as an explicit lagged preconditioner until explicitly refreshed or rebuilt.

// Rust guideline compliant 2026-02-21
use super::{
    HelmholtzForm,
    device_boundary::{prepare_hex_robin, prepare_hex_robin_quadratures},
    device_hex::{
        prepare_hex_helmholtz, prepare_hex_helmholtz_with_points, prepare_hex_lor_shifted_laplacian,
    },
};
use crate::{mesh::Mesh, spaces::H1Space};
use num_complex::Complex64;
use sonium_compute::cuda::quadrature_assembly::CudaQuadratureAssembly;
use sonium_compute::cuda::{CudaBackend, CudaBuffer, fem_hex::CudaHexPartialC64};
use sonium_compute::operator_sum::DeviceSumWorkspace;
use sonium_compute::{
    Backend, BackendError, BackendResult, Buffer, DeviceBufferLike, DeviceOperator, MemoryPool,
    PersistentDeviceCsr,
};
use sonium_solvers::iterative::{
    BackendKrylovConfig, BackendKrylovError, BackendKrylovStatus, CancellationToken,
    DeviceGmresSession, DeviceInitialGuess,
};
use sonium_solvers::{AmgConfig, BackendAmgPreconditioner, CsrMatrix};
use std::collections::HashSet;

#[cfg(test)]
mod tests;

/// Host callbacks for a fixed marked Robin or Neumann boundary.
///
/// Integrates alpha*u*v and source*v. Use zero alpha for a Neumann load.
/// The callbacks are evaluated only during construction and need not outlive it.
pub struct HexRobinBoundary<'a> {
    pub marker: i32,
    pub alpha: &'a dyn Fn(f64, f64, f64) -> Complex64,
    pub source: &'a dyn Fn(f64, f64, f64) -> Complex64,
}
impl std::fmt::Debug for HexRobinBoundary<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HexRobinBoundary")
            .field("marker", &self.marker)
            .finish_non_exhaustive()
    }
}

/// Fixed geometry, constraint and auxiliary-preconditioner choices.
#[derive(Debug, Clone)]
pub struct HexPressureConfig {
    /// Volume and boundary quadrature points per axis, from 1 through 8.
    pub quadrature_points: usize,
    /// Original H1 DOFs with prescribed pressure, including edge/face DOFs.
    pub dirichlet: Vec<usize>,
    /// Auxiliary low-order operator is K + mass_shift M, plus Robin action.
    pub mass_shift: Complex64,
    /// Explicit AMG choices; convergence remains workload dependent.
    pub amg: AmgConfig,
}

/// Typed setup, device and Krylov errors for retained hex solves.
#[derive(Debug, thiserror::Error)]
pub enum HexPressureError {
    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error(transparent)]
    Krylov(Box<BackendKrylovError>),
}
impl From<BackendKrylovError> for HexPressureError {
    fn from(error: BackendKrylovError) -> Self {
        Self::Krylov(Box::new(error))
    }
}

struct Boundary {
    action: Option<PersistentDeviceCsr<CudaBackend, Complex64>>,
    coupling: Option<PersistentDeviceCsr<CudaBackend, Complex64>>,
    load: Buffer<Complex64, CudaBackend>,
}

struct BoundaryUpdates {
    points: Vec<[f64; 3]>,
    markers: Vec<i32>,
    action: CudaQuadratureAssembly,
    coupling: CudaQuadratureAssembly,
    load: CudaQuadratureAssembly,
    action_values: Buffer<Complex64, CudaBackend>,
    coupling_values: Buffer<Complex64, CudaBackend>,
    load_values: Buffer<Complex64, CudaBackend>,
}

/// Retains a complete Q1/Q2/Q3 CUDA pressure solve across right-hand sides.
///
/// Uses partial-assembly volume actions and low-order refined shifted AMG with
/// native complex coarse LU. Host-prepared Robin matrices are added to both
/// operators. Published solutions are device buffers certified against the
/// original constrained system. A failed preparation or solve hides any previous
/// solution; use a zero initial guess to recover after correcting input errors.
/// Vendor AMG allocations retain their existing accounting limitations.
pub struct CudaHexPressureSession<'a> {
    pool: &'a MemoryPool<CudaBackend>,
    volume: CudaHexPartialC64,
    volume_points: Option<Vec<[f64; 3]>>,
    boundary: Option<Boundary>,
    boundary_updates: Option<BoundaryUpdates>,
    boundary_valid: bool,
    sum: DeviceSumWorkspace<'a, CudaBackend, Complex64>,
    preconditioner: BackendAmgPreconditioner<CudaBackend, Complex64>,
    krylov: DeviceGmresSession<'a, CudaBackend, Complex64>,
    load_rhs: Buffer<Complex64, CudaBackend>,
    lifted_rhs: Buffer<Complex64, CudaBackend>,
    coupling_rhs: Buffer<Complex64, CudaBackend>,
    published: bool,
    guess_available: bool,
    wavenumber: Complex64,
}
impl std::fmt::Debug for CudaHexPressureSession<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaHexPressureSession")
            .field("dimension", &self.volume.n())
            .field("published", &self.published)
            .finish_non_exhaustive()
    }
}
fn invalid(reason: impl Into<String>) -> BackendError {
    BackendError::InvalidArgument {
        operation: "cuda_hex_pressure_solve",
        reason: reason.into(),
    }
}
fn checked_wavenumber_square(wavenumber: Complex64) -> BackendResult<Complex64> {
    let square = wavenumber * wavenumber;
    if [wavenumber.re, wavenumber.im, square.re, square.im]
        .into_iter()
        .any(|v| !v.is_finite())
    {
        return Err(invalid("wavenumber and its square must be finite"));
    }
    Ok(square)
}
fn upload_matrix(
    pool: &MemoryPool<CudaBackend>,
    matrix: &CsrMatrix<Complex64>,
) -> BackendResult<Option<PersistentDeviceCsr<CudaBackend, Complex64>>> {
    if matrix.values.is_empty() {
        return Ok(None);
    }
    Ok(Some(PersistentDeviceCsr::try_from_host_csr_ref(
        pool.backend(),
        matrix.as_host_csr_ref(),
    )?))
}
fn entries(matrix: &CsrMatrix<Complex64>, target: &mut Vec<(usize, usize, Complex64)>) {
    for row in 0..matrix.num_rows {
        for j in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            target.push((row, matrix.col_indices[j], matrix.values[j]));
        }
    }
}

impl<'a> CudaHexPressureSession<'a> {
    /// Prepare volume, fixed boundary data, auxiliary AMG and retained solve storage.
    ///
    /// # Errors
    /// Rejects invalid mesh/space/quadrature/constraints, duplicate or missing
    /// boundary markers and invalid coefficients. Propagates typed allocation,
    /// upload, factorization and device failures. No CPU solve fallback is used.
    pub fn try_new(
        pool: &'a MemoryPool<CudaBackend>,
        mesh: &Mesh,
        space: &H1Space,
        form: &HelmholtzForm,
        config: &HexPressureConfig,
        boundaries: &[HexRobinBoundary<'_>],
    ) -> Result<Self, HexPressureError> {
        Self::prepare(pool, mesh, space, form, config, boundaries, false)
    }

    /// Prepare a session supporting resident material, Robin coefficient and source updates.
    ///
    /// Retains weighted volume reference data, sample positions, boundary maps and scratch in addition to
    /// the fixed session storage. AMG remains built from the initial boundary:
    /// subsequent solves use it as a lagged preconditioner and certify residuals
    /// against the updated system. Large changes may require a session rebuild
    /// for convergence. Geometry and constraints remain fixed.
    ///
    /// # Errors
    /// Rejects empty boundaries and the same invalid setup inputs as `try_new`;
    /// also propagates mapping, allocation, upload and kernel compilation errors.
    pub fn try_new_updatable(
        pool: &'a MemoryPool<CudaBackend>,
        mesh: &Mesh,
        space: &H1Space,
        form: &HelmholtzForm,
        config: &HexPressureConfig,
        boundaries: &[HexRobinBoundary<'_>],
    ) -> Result<Self, HexPressureError> {
        Self::prepare(pool, mesh, space, form, config, boundaries, true)
    }

    fn prepare(
        pool: &'a MemoryPool<CudaBackend>,
        mesh: &Mesh,
        space: &H1Space,
        form: &HelmholtzForm,
        config: &HexPressureConfig,
        boundaries: &[HexRobinBoundary<'_>],
        updatable: bool,
    ) -> Result<Self, HexPressureError> {
        let mass_scale = checked_wavenumber_square(form.wavenumber)?;
        sonium_solvers::iterative::validate_backend_krylov::<_, Complex64>(
            pool.backend(),
            sonium_solvers::iterative::BackendKrylovMethod::Fgmres,
            true,
        )
        .map_err(|error| invalid(error.to_string()))?;
        pool.backend().try_prepare_hex_kernels()?;
        if updatable {
            pool.backend().try_prepare_quadrature_kernels()?;
        }
        // Retain K-M even when the initial k is zero, so later k² changes never
        // divide by an initial coefficient or require re-uploading geometry.
        let canonical = HelmholtzForm::from_real(1.0);
        let (host, volume_points) = if updatable {
            let (host, points) = prepare_hex_helmholtz_with_points(
                mesh,
                space,
                &canonical,
                config.quadrature_points,
            )?;
            (host, Some(points))
        } else {
            (
                prepare_hex_helmholtz(mesh, space, &canonical, config.quadrature_points)?,
                None,
            )
        };
        let host = host.try_with_dirichlet(&config.dirichlet)?;
        let n = host.dimension();
        let mut seen = HashSet::new();
        let mut action = Vec::new();
        let mut coupling = Vec::new();
        let mut load = vec![Complex64::default(); n];
        let plan = if updatable {
            Some(prepare_hex_robin_quadratures(
                mesh,
                space,
                &boundaries.iter().map(|b| b.marker).collect::<Vec<_>>(),
                config.quadrature_points,
                &config.dirichlet,
            )?)
        } else {
            None
        };
        let mut mapped_matrices = None;
        if let Some(plan) = &plan {
            let mut alpha = Vec::with_capacity(plan.points().len());
            let mut source = Vec::with_capacity(plan.points().len());
            for (p, marker) in plan.points().iter().zip(plan.markers()) {
                let boundary = boundaries
                    .iter()
                    .find(|b| b.marker == *marker)
                    .ok_or_else(|| invalid("missing boundary callback"))?;
                let a = (boundary.alpha)(p[0], p[1], p[2]);
                let g = (boundary.source)(p[0], p[1], p[2]);
                if [a.re, a.im, g.re, g.im].into_iter().any(|v| !v.is_finite()) {
                    return Err(invalid("non-finite boundary callback").into());
                }
                alpha.push(a);
                source.push(g);
            }
            let mut a = plan.action().template().clone();
            let mut c = plan.coupling().template().clone();
            a.values = plan.action().values().try_assemble(&alpha)?;
            c.values = plan.coupling().values().try_assemble(&alpha)?;
            load = plan.load().try_assemble(&source)?;
            entries(&a, &mut action);
            mapped_matrices = Some((a, c));
        } else {
            for boundary in boundaries {
                if !seen.insert(boundary.marker)
                    || !mesh.boundaries.iter().any(|b| b.marker == boundary.marker)
                {
                    return Err(invalid("duplicate or missing Robin marker").into());
                }
                let prepared = prepare_hex_robin(
                    mesh,
                    space,
                    boundary.marker,
                    boundary.alpha,
                    boundary.source,
                    config.quadrature_points,
                    &config.dirichlet,
                )?;
                entries(&prepared.matrix, &mut action);
                entries(&prepared.dirichlet_coupling, &mut coupling);
                for (a, b) in load.iter_mut().zip(prepared.rhs) {
                    *a += b;
                }
            }
        }
        let auxiliary =
            prepare_hex_lor_shifted_laplacian(mesh, space, config.mass_shift, &config.dirichlet)?;
        let mut shifted_entries = action.clone();
        entries(&auxiliary, &mut shifted_entries);
        let shifted = CsrMatrix::try_from_triplets(n, n, shifted_entries)
            .map_err(|e| invalid(e.to_string()))?;
        let preconditioner =
            BackendAmgPreconditioner::<CudaBackend, Complex64>::try_from_csr_with_cuda_coarse_lu(
                pool.backend(),
                &shifted,
                config.amg.clone(),
            )?;
        let mut volume = CudaHexPartialC64::try_new(pool.clone(), &host, 1)?;
        if updatable {
            volume.try_prepare_material_updates()?;
        }
        volume.try_set_scales(Complex64::new(1.0, 0.0), mass_scale)?;
        let boundary = if boundaries.is_empty() {
            None
        } else {
            let (action, coupling) = if let Some(matrices) = mapped_matrices {
                matrices
            } else {
                (
                    CsrMatrix::try_from_triplets(n, n, action)
                        .map_err(|e| invalid(e.to_string()))?,
                    CsrMatrix::try_from_triplets(n, n, coupling)
                        .map_err(|e| invalid(e.to_string()))?,
                )
            };
            let mut device_load = pool.try_alloc(n).map_err(BackendError::from)?;
            pool.backend().try_copy_h2d(&load, device_load.raw_mut())?;
            Some(Boundary {
                action: upload_matrix(pool, &action)?,
                coupling: upload_matrix(pool, &coupling)?,
                load: device_load,
            })
        };
        let boundary_updates = if let Some(plan) = plan {
            Some(BoundaryUpdates {
                points: plan.points().to_vec(),
                markers: plan.markers().to_vec(),
                action: CudaQuadratureAssembly::try_new(pool.clone(), plan.action().values())?,
                coupling: CudaQuadratureAssembly::try_new(pool.clone(), plan.coupling().values())?,
                load: CudaQuadratureAssembly::try_new(pool.clone(), plan.load())?,
                action_values: pool
                    .try_alloc(plan.action().values().output_count())
                    .map_err(BackendError::from)?,
                coupling_values: pool
                    .try_alloc(plan.coupling().values().output_count())
                    .map_err(BackendError::from)?,
                load_values: pool.try_alloc(n).map_err(BackendError::from)?,
            })
        } else {
            None
        };
        Ok(Self {
            pool,
            volume,
            volume_points,
            boundary,
            boundary_updates,
            boundary_valid: true,
            sum: DeviceSumWorkspace::try_new(pool, n, &[])?,
            preconditioner,
            krylov: DeviceGmresSession::try_new_flexible(pool, n).map_err(BackendError::from)?,
            load_rhs: pool.try_alloc(n).map_err(BackendError::from)?,
            lifted_rhs: pool.try_alloc(n).map_err(BackendError::from)?,
            coupling_rhs: pool.try_alloc(n).map_err(BackendError::from)?,
            published: false,
            guess_available: false,
            wavenumber: form.wavenumber,
        })
    }

    /// Return the currently selected complex wavenumber.
    pub fn wavenumber(&self) -> Complex64 {
        self.wavenumber
    }

    /// Opt into cooperative native Jacobi on eligible AMG levels.
    ///
    /// Disabled by default; benefits depend on the hierarchy and workload.
    /// Preserves smoothing weights, sweep counts, certified output and warm
    /// starts. No device work or allocation occurs. Numeric refresh and explicit
    /// preconditioner rebuild retain this choice. Other smoothers are unchanged.
    pub fn set_small_jacobi_enabled(&mut self, enabled: bool) {
        self.preconditioner.set_small_jacobi_enabled(enabled);
    }

    /// Return whether cooperative small-level Jacobi is explicitly enabled.
    pub fn small_jacobi_enabled(&self) -> bool {
        self.preconditioner.small_jacobi_enabled()
    }

    /// Opt into fused CSR Jacobi sweeps in the auxiliary AMG preconditioner.
    ///
    /// Disabled by default. Preserves the hierarchy, damping, sweep counts,
    /// published solution and warm start. No device work or allocation occurs.
    /// Numeric refresh and explicit preconditioner rebuild retain the selection.
    /// Other smoother algorithms are unchanged.
    pub fn set_fused_jacobi_enabled(&mut self, enabled: bool) {
        self.preconditioner.set_fused_jacobi_enabled(enabled);
    }

    /// Return whether fused CSR Jacobi sweeps are explicitly enabled.
    pub fn fused_jacobi_enabled(&self) -> bool {
        self.preconditioner.fused_jacobi_enabled()
    }

    /// Physical coordinates in the required volume material sample order.
    ///
    /// Returns none for fixed sessions. Upload once when a device material law
    /// needs coordinates; this metadata query does not transfer data.
    pub fn volume_samples(&self) -> Option<&[[f64; 3]]> {
        self.volume_points.as_deref()
    }

    /// Update scalar diffusion and mass multipliers from resident material samples.
    ///
    /// Samples follow `volume_samples`. They multiply the original canonical K-M
    /// geometry, independently of prior material or raw volume-data updates. The
    /// selected k² still scales mass, and boundary terms are updated separately.
    /// One launch replaces weighted volume data without allocation or copies.
    /// AMG remains lagged until explicitly rebuilt. Values/products must be finite.
    ///
    /// # Errors
    /// Rejects fixed sessions, invalid buffers and graph capture. A failed update
    /// clears publication/warm-start access; a failed launch also blocks operator
    /// actions until a successful replacement, subject to backend recovery.
    pub fn try_update_material_samples(
        &mut self,
        diffusion: &CudaBuffer<Complex64>,
        mass: &CudaBuffer<Complex64>,
    ) -> Result<(), HexPressureError> {
        let previous_guess = std::mem::replace(&mut self.guess_available, false);
        self.published = false;
        self.volume.try_update_material_samples(diffusion, mass)?;
        self.guess_available = previous_guess;
        Ok(())
    }

    /// Replace canonical volume quadrature data while retaining solve and boundary state.
    ///
    /// Supply weighted K-M data in `HexPartialC64::quadrature_data` order, using
    /// the constructor's mesh, space and quadrature. In particular, mass entries
    /// include the minus sign but not k²; the selected wavenumber still supplies
    /// that factor. Diffusion entries include the geometry metric. Raw material
    /// samples need device lowering before this call. All entries must be finite.
    /// Uses one device copy without allocation or host staging. AMG is lagged;
    /// explicitly rebuild it when coefficient changes impair convergence.
    /// A successful update retains the old iterate but unpublishes its solution.
    ///
    /// # Errors
    /// Rejects invalid buffers and graph capture. Any failed update clears the
    /// warm-start/publication state; a failed copy also blocks operator actions
    /// until a successful full replacement, following backend recovery rules.
    pub fn try_update_volume_data(
        &mut self,
        values: &CudaBuffer<Complex64>,
    ) -> Result<(), HexPressureError> {
        let previous_guess = std::mem::replace(&mut self.guess_available, false);
        self.published = false;
        self.volume.try_update_quadrature_data(values)?;
        self.guess_available = previous_guess;
        Ok(())
    }

    /// Rebuild auxiliary AMG while retaining the physical operator and Krylov state.
    ///
    /// The caller supplies a host auxiliary matrix in the original H1 numbering,
    /// with matching prescribed-DOF elimination. For example, use
    /// `prepare_hex_lor_shifted_laplacian` with a new shift when lagged AMG becomes
    /// ineffective. This is explicit host hierarchy setup followed by upload and
    /// native coarse factorization, not a resident multilevel numeric refresh.
    /// The auxiliary matrix never replaces the physical residual operator.
    ///
    /// # Errors
    /// Rejects dimension mismatches and propagates AMG validation, allocation or
    /// factorization failures. The previous preconditioner and solution state are
    /// retained on failure, subject to the backend's context recovery disposition.
    pub fn try_rebuild_preconditioner(
        &mut self,
        auxiliary: &CsrMatrix<Complex64>,
        config: AmgConfig,
    ) -> Result<(), HexPressureError> {
        if auxiliary.num_rows != self.volume.n() || auxiliary.num_cols != self.volume.n() {
            return Err(invalid("auxiliary preconditioner dimension mismatch").into());
        }
        let mut replacement =
            BackendAmgPreconditioner::<CudaBackend, Complex64>::try_from_csr_with_cuda_coarse_lu(
                self.pool.backend(),
                auxiliary,
                config,
            )?;
        replacement.set_small_jacobi_enabled(self.preconditioner.small_jacobi_enabled());
        replacement.set_fused_jacobi_enabled(self.preconditioner.fused_jacobi_enabled());
        self.preconditioner = replacement;
        Ok(())
    }

    /// Refresh compatible auxiliary AMG while retaining hierarchy and coarse-factor storage.
    ///
    /// The host auxiliary matrix must use the physical H1 DOF ordering and
    /// prescribed-DOF elimination. Hierarchy construction remains host work.
    /// Returns `false` before device mutation when hierarchy patterns change;
    /// use `try_rebuild_preconditioner` then. The physical operator is unchanged.
    /// Every attempt hides published output but retains the iterate as a warm start.
    /// Partial uploads or factorization failures invalidate the preconditioner
    /// until a complete refresh or rebuild succeeds.
    ///
    /// # Errors
    /// Rejects dimensions or invalid host hierarchies and propagates backend
    /// failures. Backend recovery may be required before retrying.
    pub fn try_refresh_preconditioner(
        &mut self,
        auxiliary: &CsrMatrix<Complex64>,
        config: AmgConfig,
    ) -> Result<bool, HexPressureError> {
        self.published = false;
        if auxiliary.num_rows != self.volume.n() || auxiliary.num_cols != self.volume.n() {
            return Err(invalid("auxiliary preconditioner dimension mismatch").into());
        }
        let host = sonium_solvers::AmgPreconditioner::try_from_csr(auxiliary, config)
            .map_err(|error| invalid(error.to_string()))?;
        Ok(self
            .preconditioner
            .try_refresh_from_host_amg(self.pool.backend(), &host)?)
    }

    /// Change the volume wavenumber while retaining geometry, boundaries and AMG.
    ///
    /// The operator becomes K-k²M with the existing constraints. This changes
    /// only launch parameters, with no kernel, allocation or array transfer.
    /// Boundary/source frequency dependence must be updated separately. AMG is
    /// an explicit lagged preconditioner; refresh or rebuild it if it ceases to
    /// be effective. The old iterate remains available as a warm start, but is no
    /// longer published as a certified solution to the new system.
    ///
    /// # Errors
    /// Rejects non-finite wavenumbers/squares and pending backend errors without
    /// changing the selected operator or its publication state.
    pub fn try_update_wavenumber(&mut self, wavenumber: Complex64) -> Result<(), HexPressureError> {
        let scale = checked_wavenumber_square(wavenumber)?;
        self.volume
            .try_set_scales(Complex64::new(1.0, 0.0), scale)?;
        self.wavenumber = wavenumber;
        self.published = false;
        Ok(())
    }

    /// Coordinates and markers in the required resident boundary sample order.
    ///
    /// Returns none for a fixed-boundary session. Upload coordinates once when a
    /// device material law needs them; querying this metadata performs no transfer.
    pub fn boundary_samples(&self) -> Option<(&[[f64; 3]], &[i32])> {
        self.boundary_updates
            .as_ref()
            .map(|u| (u.points.as_slice(), u.markers.as_slice()))
    }

    /// Rebuild numeric boundary action, lifting and load from resident samples.
    ///
    /// Performs at most three assembly launches and three device copies, with no
    /// host staging or warm allocation. `alpha` and `source` follow the order from
    /// `boundary_samples`. Coefficients must be finite; arithmetic does not clip
    /// or download them. AMG stays unchanged as an explicit lagged preconditioner.
    /// The old solution becomes unpublished, but remains an available initial
    /// guess after a successful update. Residual certification uses the new data.
    ///
    /// # Errors
    /// Rejects fixed sessions and invalid buffers. A failed numeric update invalidates
    /// both boundary state and solution access; successfully repeat the full update
    /// before solving, following backend context recovery when required.
    pub fn try_update_boundary(
        &mut self,
        alpha: &CudaBuffer<Complex64>,
        source: &CudaBuffer<Complex64>,
    ) -> Result<(), HexPressureError> {
        let previous_guess = std::mem::replace(&mut self.guess_available, false);
        self.published = false;
        let updates = self
            .boundary_updates
            .as_mut()
            .ok_or_else(|| invalid("boundary updates were not prepared"))?;
        self.boundary_valid = false;
        let boundary = self
            .boundary
            .as_mut()
            .ok_or_else(|| invalid("missing boundary state"))?;
        updates
            .action
            .try_assemble(alpha, updates.action_values.raw_mut())?;
        updates
            .coupling
            .try_assemble(alpha, updates.coupling_values.raw_mut())?;
        updates
            .load
            .try_assemble(source, updates.load_values.raw_mut())?;
        let b = self.pool.backend();
        if let Some(action) = &mut boundary.action {
            action.try_copy_values_from_device(b, updates.action_values.raw())?;
        }
        if let Some(coupling) = &mut boundary.coupling {
            coupling.try_copy_values_from_device(b, updates.coupling_values.raw())?;
        }
        b.try_copy_d2d(updates.load_values.raw(), boundary.load.raw_mut())?;
        self.boundary_valid = true;
        self.guess_available = previous_guess;
        Ok(())
    }

    /// Borrow the last certified solution, or none after a failed/unconverged call.
    pub fn solution(&self) -> Option<&CudaBuffer<Complex64>> {
        if self.published {
            self.krylov.solution()
        } else {
            None
        }
    }

    /// Solve from resident volume loads and prescribed nodal pressure values.
    ///
    /// `rhs` contains volume loads before elimination; the current Robin/Neumann
    /// source is added exactly once. `prescribed` has one entry per original DOF;
    /// free entries are ignored. Both buffers must belong to the setup backend.
    /// The session lifts nonzero constraints on device and publishes only a
    /// converged solution. The first solve reserves the complete basis capacity;
    /// subsequent solves reuse it independently of earlier iteration counts.
    /// Small reduction buffers can resize for changed restart-cycle lengths.
    /// Convergence controls may transfer small scalars/Hessenberg data, but no
    /// bulk RHS or solution staging occurs.
    ///
    /// # Errors
    /// Returns typed invalid input, cancellation, backend and Krylov errors.
    /// After a backend failure follow its context-recovery disposition before reuse.
    pub fn solve(
        &mut self,
        rhs: &CudaBuffer<Complex64>,
        prescribed: &CudaBuffer<Complex64>,
        initial: DeviceInitialGuess,
        config: &BackendKrylovConfig<f64>,
        cancellation: Option<&CancellationToken>,
    ) -> Result<BackendKrylovStatus<f64>, HexPressureError> {
        let had_solution = std::mem::replace(&mut self.guess_available, false);
        self.published = false;
        if !self.boundary_valid {
            return Err(invalid("boundary update is incomplete").into());
        }
        if rhs.len() != self.volume.n() || prescribed.len() != self.volume.n() {
            return Err(invalid("RHS or prescribed vector dimension mismatch").into());
        }
        if initial == DeviceInitialGuess::PreviousSolution && !had_solution {
            return Err(invalid("no certified previous solution").into());
        }
        if config.max_iterations == 0
            || config.restart == 0
            || !config.tolerance.is_finite()
            || config.tolerance < 0.0
        {
            return Err(invalid("invalid Krylov capacity or tolerance").into());
        }
        self.krylov
            .try_reserve_cycle(config.restart.min(config.max_iterations))?;
        self.pool
            .backend()
            .try_reserve_krylov_reductions::<Complex64>(
                config.restart.min(config.max_iterations),
            )?;
        let backend = self.pool.backend();
        backend.try_copy_d2d(rhs, self.load_rhs.raw_mut())?;
        if let Some(boundary) = &self.boundary {
            backend.try_axpy(
                Complex64::new(1.0, 0.0),
                boundary.load.raw(),
                self.load_rhs.raw_mut(),
            )?;
        }
        self.volume.try_lift_rhs_colmajor(
            backend,
            1,
            self.load_rhs.raw(),
            prescribed,
            self.lifted_rhs.raw_mut(),
        )?;
        if let Some(coupling) = self.boundary.as_ref().and_then(|b| b.coupling.as_ref()) {
            coupling.try_apply(backend, prescribed, self.coupling_rhs.raw_mut())?;
            backend.try_axpy(
                Complex64::new(-1.0, 0.0),
                self.coupling_rhs.raw(),
                self.lifted_rhs.raw_mut(),
            )?;
        }
        let status = if let Some(action) = self.boundary.as_ref().and_then(|b| b.action.as_ref()) {
            let operator = self.sum.try_bind(&self.volume, action)?;
            self.krylov.solve(
                &operator,
                &self.preconditioner,
                self.lifted_rhs.raw(),
                initial,
                config,
                cancellation,
            )?
        } else {
            self.krylov.solve(
                &self.volume,
                &self.preconditioner,
                self.lifted_rhs.raw(),
                initial,
                config,
                cancellation,
            )?
        };
        self.published = status.converged;
        self.guess_available = status.converged;
        Ok(status)
    }
}
