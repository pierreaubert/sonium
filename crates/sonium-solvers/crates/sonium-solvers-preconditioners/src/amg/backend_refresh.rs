//! Checked numeric replacement for retained AMG hierarchies.

// Rust guideline compliant 2026-02-21
use super::*;
use sonium_compute::{BackendError, BackendResult, DeviceScalar};

fn same_pattern<T: ComplexField>(a: &CsrMatrix<T>, b: &CsrMatrix<T>) -> bool {
    a.num_rows == b.num_rows
        && a.num_cols == b.num_cols
        && a.row_ptrs == b.row_ptrs
        && a.col_indices == b.col_indices
}

fn same_optional_pattern<T: ComplexField>(
    a: Option<&CsrMatrix<T>>,
    b: Option<&CsrMatrix<T>>,
) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => same_pattern(a, b),
        (None, None) => true,
        _ => false,
    }
}

impl<B, T> BackendAmgPreconditioner<B, T>
where
    B: sonium_compute::Backend,
    T: ComplexField<Real = <T as DeviceScalar>::Real> + DeviceScalar,
{
    pub(super) fn ensure_numeric_valid(&self) -> BackendResult<()> {
        if !self.numeric_valid {
            return Err(BackendError::InvalidArgument {
                operation: "amg_numeric_state",
                reason: "numeric update failed; refresh the complete hierarchy before use".into(),
            });
        }
        Ok(())
    }

    /// Refresh a matching host hierarchy without replacing device buffers or descriptors.
    ///
    /// Returns `false` before mutation if any matrix/transfer pattern, level size
    /// or coloring differs; the caller must rebuild in that case. Host hierarchy
    /// construction remains the caller's responsibility. All level operators,
    /// transfer values, smoother scaling, spectral bounds and configuration are
    /// refreshed together. A successful update also replaces the portable checkpoint.
    ///
    /// Validation errors preserve the old action. A backend failure invalidates
    /// scalar/block application and checkpoint export until a complete refresh
    /// succeeds. Retrying the previous host hierarchy restores its entire state.
    ///
    /// # Errors
    /// Rejects invalid host hierarchies, non-finite scaling and direct coarse solvers
    /// without a numeric refresh contract. Complex-double CUDA LU retains factor
    /// storage; complex-single LU currently requires a rebuild. Propagates checkpoint,
    /// factorization and backend failures.
    pub fn try_refresh_from_host_amg(
        &mut self,
        backend: &B,
        host: &AmgPreconditioner<T>,
    ) -> BackendResult<bool> {
        let invalid = |reason: String| BackendError::InvalidArgument {
            operation: "amg_numeric_refresh",
            reason,
        };
        if self
            .coarse_direct
            .as_ref()
            .is_some_and(|coarse| !coarse.supports_numeric_refresh())
        {
            return Err(BackendError::Unsupported {
                backend: type_name::<B>(),
                operation: "amg_numeric_refresh_direct_coarse",
                scalar: type_name::<T>(),
            });
        }
        host.validate_generated_setup()
            .map_err(|e| invalid(e.to_string()))?;
        let (payload, _) = decode_backend_checkpoint::<B, T>(&self.checkpoint_bytes)?
            .unwrap_or_else(|| (self.checkpoint_bytes.clone(), false));
        let previous = AmgPreconditioner::<T>::from_checkpoint_bytes(&payload)
            .map_err(|e| invalid(e.to_string()))?;
        if previous.levels.len() != host.levels.len()
            || previous.levels.iter().zip(&host.levels).any(|(a, b)| {
                a.num_dofs != b.num_dofs
                    || !same_pattern(&a.matrix, &b.matrix)
                    || !same_optional_pattern(a.prolongation.as_ref(), b.prolongation.as_ref())
                    || !same_optional_pattern(a.restriction.as_ref(), b.restriction.as_ref())
                    || a.color_sets != b.color_sets
            })
        {
            return Ok(false);
        }
        let scalings: Vec<Vec<T>> = host
            .levels
            .iter()
            .map(|level| {
                if host.config.smoother == AmgSmoother::L1Jacobi {
                    level
                        .l1_diag
                        .iter()
                        .map(|&v| <T as ComplexField>::from_real(v).inv())
                        .collect()
                } else {
                    level.diag_inv.clone()
                }
            })
            .collect();
        if scalings
            .iter()
            .flatten()
            .any(|v| !DeviceScalar::is_finite(v))
        {
            return Err(invalid("smoother scaling must be finite".into()));
        }
        let payload = host
            .to_checkpoint_bytes()
            .map_err(|e| invalid(e.to_string()))?;
        let checkpoint = backend_checkpoint_bytes::<B, T>(&payload, self.coarse_direct.is_some())?;
        self.numeric_valid = false;
        for ((device, host), scaling) in self.levels.iter_mut().zip(&host.levels).zip(scalings) {
            device
                .matrix
                .try_copy_values_from_host(backend, &host.matrix.values)?;
            if let (Some(device), Some(host)) =
                (device.prolongation.as_mut(), host.prolongation.as_ref())
            {
                device.try_copy_values_from_host(backend, &host.values)?;
            }
            if let (Some(device), Some(host)) =
                (device.restriction.as_mut(), host.restriction.as_ref())
            {
                device.try_copy_values_from_host(backend, &host.values)?;
            }
            backend.try_inject_preconditioner_failure(
                sonium_compute::PreconditionerFailureInjectionPoint::DiagonalUpload,
            )?;
            device
                .diag_inv
                .try_copy_values_from_host(backend, &scaling)?;
            device.lambda_max = host.lambda_max;
        }
        if let Some(coarse) = self.coarse_direct.as_mut() {
            backend.try_inject_preconditioner_failure(
                sonium_compute::PreconditionerFailureInjectionPoint::CoarseSolve,
            )?;
            coarse.refresh(
                backend,
                &host.levels.last().expect("validated hierarchy").matrix,
            )?;
        }
        backend.try_synchronize()?;
        self.config = host.config.clone();
        self.checkpoint_bytes = checkpoint;
        self.numeric_valid = true;
        Ok(true)
    }
}
