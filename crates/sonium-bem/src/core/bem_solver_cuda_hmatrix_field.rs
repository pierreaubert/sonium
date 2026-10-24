//! Retained off-surface fields for compressed scattering solves.

// Rust guideline compliant 2026-02-21
use super::{BemError, Complex64, CudaBuffer, CudaHMatrixBemSession, failed};
pub(super) use crate::core::postprocess::pressure::CudaBemFieldWorkspace as FieldWorkspace;

impl CudaHMatrixBemSession<'_> {
    /// Prepare fixed off-surface points with native field storage.
    ///
    /// Uses seven-point constant-Tri3 quadrature. Near-surface accuracy requires
    /// independent convergence checks. Failed preparation removes the old plan.
    ///
    /// # Errors
    /// Rejects invalid geometry, points, unsupported traces and backend failures.
    pub fn prepare_field_points(&mut self, points: &[[f64; 3]]) -> Result<(), BemError> {
        self.field = None;
        self.field = Some(FieldWorkspace::try_new(
            self.pool,
            &self.metadata.elements,
            &self.metadata.nodes,
            points,
        )?);
        Ok(())
    }

    /// Evaluate total pressure without downloading certified surface traces.
    ///
    /// # Errors
    /// Requires a certified solution and prepared points. Propagates native
    /// field validation and backend failures.
    pub fn evaluate_pressure_device(&mut self) -> Result<&CudaBuffer<Complex64>, BemError> {
        if !self.published {
            return Err(failed(
                "no certified H-matrix solution for field evaluation",
            ));
        }
        let field = self
            .field
            .as_mut()
            .ok_or_else(|| failed("field points are not prepared"))?;
        let pressure = self
            .krylov
            .solution()
            .ok_or_else(|| failed("missing certified iterate"))?;
        field.try_evaluate(
            self.pool.backend(),
            pressure,
            &self.metadata.physics,
            self.sources.raw(),
            self.source_count,
        )
    }
}
