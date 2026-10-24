//! Resident electric fields and integral observables from scalar potential.

// Rust guideline compliant 2026-02-21
use super::CapacitorObservables;
use crate::mesh::Mesh;
use crate::physics::electromagnetics::magnetostatics::CudaMagneticCentroidField;
use crate::solver::device::FemDeviceSolveError;
use crate::spaces::H1Space;
use sonium_compute::{
    Backend, BackendError, Buffer, CudaBackend, CudaBuffer, DeviceBufferLike, MemoryPool,
    PersistentDeviceCsr, ReductionMode,
};
use sonium_solvers::CsrMatrix;

fn invalid(reason: &str) -> FemDeviceSolveError {
    BackendError::InvalidArgument {
        operation: "cuda_electrostatic_observables",
        reason: reason.into(),
    }
    .into()
}

/// Retains an unconstrained stiffness operator for electrostatic integral observables.
///
/// Evaluates `energy = epsilon * V^T K V / 2` using resident potentials. Only
/// compact validation/reduction scalars reach the host. The caller supplies the
/// original Laplacian, not the matrix after Dirichlet elimination. Relative
/// material weights may be included in K, with epsilon supplying the reference
/// permittivity. Alternatively, use [`Self::try_from_stiffness`] for an operator
/// already weighted by absolute permittivity. This does not compute
/// conductor-resolved charges or validate the physical solution's PDE residual.
pub struct CudaElectrostaticObservables<'a> {
    pool: &'a MemoryPool<CudaBackend>,
    laplacian: PersistentDeviceCsr<CudaBackend, f64>,
    action: Buffer<f64, CudaBackend>,
    permittivity: f64,
    numeric_ready: bool,
}

impl std::fmt::Debug for CudaElectrostaticObservables<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaElectrostaticObservables")
            .field("dimension", &self.action.len())
            .field("permittivity", &self.permittivity)
            .field("numeric_ready", &self.numeric_ready)
            .finish_non_exhaustive()
    }
}

impl<'a> CudaElectrostaticObservables<'a> {
    /// Upload an unconstrained reference Laplacian and retain reusable action storage.
    ///
    /// Host assembly remains setup work. The pool covers the action vector;
    /// sparse descriptors/storage retain their existing allocation policy.
    /// The matrix may include relative dielectric weights; `permittivity` is
    /// the fixed positive reference scale applied to every evaluation.
    ///
    /// # Errors
    /// Rejects empty/nonsquare/nonfinite matrices, invalid permittivity, unavailable
    /// native primitives and backend allocation or upload failures.
    pub fn try_new(
        pool: &'a MemoryPool<CudaBackend>,
        unit_laplacian: &CsrMatrix<f64>,
        permittivity: f64,
    ) -> Result<Self, FemDeviceSolveError> {
        if !permittivity.is_finite()
            || permittivity <= 0.0
            || unit_laplacian.num_rows == 0
            || unit_laplacian.num_rows != unit_laplacian.num_cols
            || unit_laplacian.values.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(
                "requires a finite square unit Laplacian and positive permittivity",
            ));
        }
        let caps = pool.backend().capabilities();
        for (operation, state) in [
            ("electrostatic_csr_spmv", caps.csr_spmv.state_for::<f64>()),
            ("electrostatic_blas1", caps.blas1.state_for::<f64>()),
            (
                "electrostatic_fast_reduction",
                caps.reduction.fast.state_for::<f64>(),
            ),
        ] {
            if !state.is_native() {
                return Err(BackendError::Unsupported {
                    backend: caps.backend_name,
                    operation,
                    scalar: "f64",
                }
                .into());
            }
        }
        Ok(Self {
            pool,
            laplacian: PersistentDeviceCsr::try_from_host_csr_ref(
                pool.backend(),
                unit_laplacian.as_host_csr_ref(),
            )?,
            action: pool
                .try_alloc(unit_laplacian.num_rows)
                .map_err(BackendError::from)?,
            permittivity,
            numeric_ready: true,
        })
    }

    /// Prepare observables from an unconstrained, absolute-permittivity-weighted stiffness matrix.
    ///
    /// Supports heterogeneous linear dielectrics. The caller must assemble
    /// positive material coefficients and preserve the unconstrained energy
    /// operator; this constructor does not prove positive semidefiniteness.
    /// No additional permittivity scale is applied during evaluation.
    ///
    /// # Errors
    /// Rejects invalid matrices and unavailable native primitives; propagates
    /// allocation, upload and descriptor preparation failures.
    pub fn try_from_stiffness(
        pool: &'a MemoryPool<CudaBackend>,
        stiffness: &CsrMatrix<f64>,
    ) -> Result<Self, FemDeviceSolveError> {
        Self::try_new(pool, stiffness, 1.0)
    }

    /// Replace resident material-weighted values without rebuilding sparse storage.
    ///
    /// Values must follow the original CSR ordering and material normalization.
    /// Geometry, sparsity and reference permittivity remain fixed. The caller
    /// must preserve the physical energy operator and update the solved potential
    /// consistently. All failures disable evaluation until a successful complete
    /// replacement; no previous matrix is silently reused after a failed update.
    ///
    /// # Errors
    /// Rejects wrong-sized or nonfinite values and unavailable native validation.
    /// Propagates copy and synchronization failures. Only compact validation
    /// status reaches the host; numeric values stay on the device.
    pub fn try_update_values(
        &mut self,
        values: &CudaBuffer<f64>,
    ) -> Result<(), FemDeviceSolveError> {
        self.numeric_ready = false;
        let backend = self.pool.backend();
        if values.len() != self.laplacian.nnz() {
            return Err(invalid("incorrect material update length"));
        }
        if backend.try_all_finite_native(values)? != Some(true) {
            return Err(invalid(
                "nonfinite material values or unavailable native validation",
            ));
        }
        self.laplacian
            .try_copy_values_from_device(backend, values)?;
        backend.try_synchronize()?;
        self.numeric_ready = true;
        Ok(())
    }

    /// Combine resident material bases without rebuilding the energy operator.
    ///
    /// Sets values to `base_scale * base + sum(alpha * term)`, with the same
    /// ordering and normalization as construction. The caller preserves positive
    /// material coefficients and consistency with the solved physical system.
    /// Failures disable evaluation until a successful replacement or combination.
    ///
    /// # Errors
    /// Rejects invalid sizes/scalars and nonfinite combined values. Propagates
    /// backend failures; no bulk values are downloaded or uploaded.
    pub fn try_combine_values(
        &mut self,
        base_scale: f64,
        base: &CudaBuffer<f64>,
        terms: &[(f64, &CudaBuffer<f64>)],
    ) -> Result<(), FemDeviceSolveError> {
        self.numeric_ready = false;
        let backend = self.pool.backend();
        self.laplacian
            .try_combine_scaled_values(backend, base_scale, base, terms)?;
        if backend.try_all_finite_native(&self.laplacian.matrix().values)? != Some(true) {
            return Err(invalid(
                "nonfinite material combination or unavailable native validation",
            ));
        }
        backend.try_synchronize()?;
        self.numeric_ready = true;
        Ok(())
    }

    /// Evaluate energy, capacitance and signed charge without downloading potential.
    ///
    /// Caller supplies the voltage difference between the two conductors. Output
    /// is returned only after finite/nonnegative validation; failures leave scratch
    /// unspecified, and a subsequent successful call overwrites it completely.
    ///
    /// # Errors
    /// Rejects incomplete material updates, invalid buffers, nonfinite potential, invalid voltage or squared
    /// voltage outside the positive finite range, negative/nonfinite energy,
    /// observable overflow, and backend failures. No host-array fallback occurs.
    pub fn evaluate(
        &mut self,
        potential: &CudaBuffer<f64>,
        voltage_difference: f64,
    ) -> Result<CapacitorObservables, FemDeviceSolveError> {
        if !self.numeric_ready {
            return Err(invalid(
                "material update incomplete; replace values before evaluation",
            ));
        }
        let voltage_squared = voltage_difference * voltage_difference;
        if !voltage_squared.is_finite()
            || voltage_squared <= 0.0
            || potential.len() != self.action.len()
        {
            return Err(invalid("invalid voltage difference or potential dimension"));
        }
        let backend = self.pool.backend();
        if backend.try_all_finite_native(potential)? != Some(true) {
            return Err(invalid(
                "nonfinite potential or unavailable native validation",
            ));
        }
        self.laplacian
            .try_apply(backend, potential, self.action.raw_mut())?;
        let quadratic =
            backend.try_dot_with_mode(ReductionMode::Fast, potential, self.action.raw())?;
        let energy_j = 0.5 * self.permittivity * quadratic;
        let capacitance_f = 2.0 * energy_j / voltage_squared;
        let charge_c = capacitance_f * voltage_difference;
        if quadratic < 0.0
            || !quadratic.is_finite()
            || !energy_j.is_finite()
            || !capacitance_f.is_finite()
            || !charge_c.is_finite()
        {
            return Err(invalid("negative or nonfinite electrostatic observables"));
        }
        backend.try_synchronize()?;
        Ok(CapacitorObservables {
            energy_j,
            capacitance_f,
            charge_c,
        })
    }
}

/// Retained centroid sampling of `E = -grad(V)` on affine P1/P2 elements.
///
/// Geometry and sparse gradients are prepared on the host once. Evaluations
/// consume resident potentials and retain device output, with compact finite
/// validation transfers only. Triangle output has two components per element;
/// tetrahedral output has three. This is not arbitrary-point interpolation.
#[derive(Debug)]
pub struct CudaElectrostaticCentroidField<'a> {
    gradient: CudaMagneticCentroidField<'a>,
}

impl<'a> CudaElectrostaticCentroidField<'a> {
    /// Prepare resident electric-field recovery for affine triangles or tetrahedra.
    ///
    /// # Errors
    /// Rejects unsupported spaces/geometry and propagates allocation or backend errors.
    pub fn try_new(
        pool: &'a MemoryPool<CudaBackend>,
        mesh: &Mesh,
        space: &H1Space,
    ) -> Result<Self, FemDeviceSolveError> {
        // The shared gradient implementation evaluates Br - mu*grad(phi).
        // Unit weight and zero offset give the electric field directly.
        Ok(Self {
            gradient: CudaMagneticCentroidField::try_new(
                pool,
                mesh,
                space,
                &vec![1.0; mesh.elements.len()],
                &vec![[0.0; 3]; mesh.elements.len()],
            )?,
        })
    }

    /// Evaluate the electric field from a resident nodal potential.
    ///
    /// # Errors
    /// Rejects invalid buffers/nonfinite fields and propagates backend errors.
    /// Any failed evaluation hides the previously published field until recovery.
    pub fn evaluate(&mut self, potential: &CudaBuffer<f64>) -> Result<(), FemDeviceSolveError> {
        self.gradient.evaluate(potential)
    }

    /// Borrow the last successfully evaluated field.
    pub fn field(&self) -> Option<&CudaBuffer<f64>> {
        self.gradient.field()
    }

    /// Return the number of components per element.
    pub fn spatial_dimension(&self) -> usize {
        self.gradient.spatial_dimension()
    }
}
