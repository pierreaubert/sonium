//! Retained CUDA centroid fields for affine scalar-potential elements.

// Rust guideline compliant 2026-02-21
use crate::basis::PolynomialDegree;
use crate::mesh::{ElementType, Mesh};
use crate::solver::device::FemDeviceSolveError;
use crate::spaces::{FunctionSpace, H1Space};
use sonium_compute::{
    Backend, BackendError, Buffer, CudaBackend, CudaBuffer, DeviceBufferLike, MemoryPool,
    PersistentDeviceCsr,
};
use sonium_solvers::CsrMatrix;

fn invalid(reason: impl Into<String>) -> FemDeviceSolveError {
    BackendError::InvalidArgument {
        operation: "cuda_magnetic_centroid_field",
        reason: reason.into(),
    }
    .into()
}

/// Retains the permeability-weighted gradient and remanence for CUDA field recovery.
///
/// Evaluates `B = Br - mu grad(phi)` at each affine element centroid. Output is
/// element-major with two components for triangles or three for tetrahedra,
/// matching the in-plane convention of `compute_b_field`. Geometry and gradient
/// preparation are host work. Repeated evaluations retain sparse descriptors and
/// buffers; only compact finite-validation status reaches the host.
///
/// Mesh, space and permeability are fixed. Rebuild when they change. Remanence
/// can be replaced on-device. The pool covers vector buffers, not all legacy
/// sparse storage. This is centroid sampling, not arbitrary field interpolation.
pub struct CudaMagneticCentroidField<'a> {
    pool: &'a MemoryPool<CudaBackend>,
    gradient: PersistentDeviceCsr<CudaBackend, f64>,
    remanence: Buffer<f64, CudaBackend>,
    field: Buffer<f64, CudaBackend>,
    dimension: usize,
    remanence_ready: bool,
    published: bool,
}
impl std::fmt::Debug for CudaMagneticCentroidField<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaMagneticCentroidField")
            .field("dimension", &self.dimension)
            .field("elements", &(self.field.len() / self.dimension))
            .field("published", &self.published)
            .finish_non_exhaustive()
    }
}
impl<'a> CudaMagneticCentroidField<'a> {
    /// Prepare centroid recovery for affine H1 P1/P2 triangles or tetrahedra.
    ///
    /// Supply positive scalar permeability and one remanence vector per element.
    /// Triangle recovery uses the first two remanence components. Mixed element
    /// families and curved geometry require a separate evaluator.
    ///
    /// # Errors
    /// Rejects unsupported spaces/geometry, malformed connectivity, degenerate
    /// elements, non-finite data and material lengths. Propagates backend failures.
    pub fn try_new(
        pool: &'a MemoryPool<CudaBackend>,
        mesh: &Mesh,
        space: &H1Space,
        permeability: &[f64],
        remanence: &[[f64; 3]],
    ) -> Result<Self, FemDeviceSolveError> {
        let element = mesh
            .elements
            .first()
            .ok_or_else(|| invalid("empty mesh"))?
            .element_type;
        let (dimension, q) = match element {
            ElementType::Triangle => (2, [1.0 / 3.0, 1.0 / 3.0, 0.0]),
            ElementType::Tetrahedron => (3, [0.25; 3]),
            _ => {
                return Err(invalid(
                    "only affine triangles and tetrahedra are supported",
                ));
            }
        };
        if !matches!(space.degree, PolynomialDegree::P1 | PolynomialDegree::P2)
            || mesh
                .elements
                .iter()
                .any(|e| e.element_type != element || e.nodes.len() != element.num_vertices())
        {
            return Err(invalid("requires uniform affine geometry and H1 P1/P2"));
        }
        if permeability.len() != mesh.elements.len()
            || remanence.len() != mesh.elements.len()
            || permeability.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || remanence.iter().flatten().any(|v| !v.is_finite())
        {
            return Err(invalid(
                "one positive permeability and finite remanence per element are required",
            ));
        }
        space.validate(mesh)?;
        let n = space.num_global_dofs(mesh);
        let rows = mesh
            .elements
            .len()
            .checked_mul(dimension)
            .ok_or_else(|| invalid("field size overflow"))?;
        let mut triplets = Vec::new();
        let mut br = Vec::with_capacity(rows);
        for e in 0..mesh.elements.len() {
            let tab = space.tabulate(mesh, e, q[0], q[1], q[2]);
            if !tab.det_j.is_finite()
                || tab.det_j <= 0.0
                || tab.derivatives.iter().any(|v| !v.is_finite())
            {
                return Err(invalid("non-finite or degenerate centroid geometry"));
            }
            let dofs = space.local_to_global(mesh, e);
            for (d, &remanent) in remanence[e][..dimension].iter().enumerate() {
                br.push(remanent);
                for (i, &dof) in dofs.iter().enumerate() {
                    let value = permeability[e] * tab.grad(i, d);
                    if !value.is_finite() {
                        return Err(invalid("permeability-weighted gradient overflow"));
                    }
                    triplets.push((e * dimension + d, dof, value));
                }
            }
        }
        let matrix = CsrMatrix::from_triplets(rows, n, triplets);
        let gradient =
            PersistentDeviceCsr::try_from_host_csr_ref(pool.backend(), matrix.as_host_csr_ref())?;
        let mut remanence = pool.try_alloc(rows).map_err(BackendError::from)?;
        pool.backend().try_copy_h2d(&br, remanence.raw_mut())?;
        Ok(Self {
            pool,
            gradient,
            remanence,
            field: pool.try_alloc(rows).map_err(BackendError::from)?,
            dimension,
            remanence_ready: true,
            published: false,
        })
    }

    /// Return the number of field components per element.
    pub fn spatial_dimension(&self) -> usize {
        self.dimension
    }

    /// Borrow the last successfully evaluated field, if still valid.
    pub fn field(&self) -> Option<&CudaBuffer<f64>> {
        self.published.then(|| self.field.raw())
    }

    fn validate_buffer(
        &self,
        buffer: &CudaBuffer<f64>,
        len: usize,
    ) -> Result<(), FemDeviceSolveError> {
        if buffer.len() != len
            || buffer.raw().is_none_or(|raw| {
                !std::sync::Arc::ptr_eq(raw.stream(), self.pool.backend().stream())
                    || len.checked_mul(std::mem::size_of::<f64>()) != Some(raw.len())
            })
        {
            return Err(invalid("buffer length, allocation or stream mismatch"));
        }
        if self.pool.backend().try_all_finite_native(buffer)? != Some(true) {
            return Err(invalid("non-finite input or unavailable native validation"));
        }
        Ok(())
    }

    /// Replace resident element-major remanence without reallocating storage.
    ///
    /// # Errors
    /// Rejects incompatible/non-finite input. A failed copy blocks evaluation
    /// until a full replacement succeeds after backend recovery.
    pub fn try_update_remanence(
        &mut self,
        remanence: &CudaBuffer<f64>,
    ) -> Result<(), FemDeviceSolveError> {
        self.published = false;
        self.validate_buffer(remanence, self.remanence.len())?;
        self.remanence_ready = false;
        self.pool
            .backend()
            .try_copy_d2d(remanence, self.remanence.raw_mut())?;
        self.pool.backend().try_synchronize()?;
        self.remanence_ready = true;
        Ok(())
    }

    /// Recover centroid fields directly from a resident scalar-potential solution.
    ///
    /// The caller supplies a physical solution; finite validation does not certify
    /// its PDE residual. Failed evaluation unpublishes output. Retry after recovery
    /// for recoverable backend errors; terminal failures require a new session.
    ///
    /// # Errors
    /// Rejects invalid buffers or incomplete remanence updates. Propagates backend
    /// errors and rejects non-finite output before publication.
    pub fn evaluate(&mut self, potential: &CudaBuffer<f64>) -> Result<(), FemDeviceSolveError> {
        self.published = false;
        if !self.remanence_ready {
            return Err(invalid("incomplete remanence update"));
        }
        self.validate_buffer(potential, self.gradient.n())?;
        let backend = self.pool.backend();
        self.gradient.try_residual(
            backend,
            potential,
            self.remanence.raw(),
            self.field.raw_mut(),
        )?;
        if backend.try_all_finite_native(self.field.raw())? != Some(true) {
            return Err(invalid("non-finite field or unavailable native validation"));
        }
        backend.try_synchronize()?;
        self.published = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
