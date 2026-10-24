//! Fixed-pattern numeric updates preserve physical CSR and Krylov storage.

// Rust guideline compliant 2026-02-21
use super::*;

/// Storage changes made by a successful numeric matrix update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaLinearUpdateStatus {
    /// Whether changed hierarchy, factor structure or Schwarz membership required new storage.
    ///
    /// False means identity was retained or preconditioner values were refreshed
    /// in place. Physical CSR descriptors and Krylov buffers are always retained.
    pub preconditioner_rebuilt: bool,
}

impl<T> CudaLinearSolveSession<'_, T>
where
    T: ComplexField + DeviceScalar<Real = <T as ComplexField>::Real>,
    <T as ComplexField>::Real: FromPrimitive + Sync + Send,
{
    /// Replace matrix values and refresh preconditioning at fixed canonical sparsity.
    ///
    /// Revalidates the matrix and the original SPD/HPD assumption. Identity and
    /// Jacobi/Chebyshev retain storage; Chebyshev recomputes its host spectral bound.
    /// AMG rebuilds the host hierarchy and refreshes device
    /// storage when every level/transfer pattern matches, otherwise replacing
    /// only the device preconditioner. Physical CSR and Krylov storage survive.
    /// ILU refactors on the host, retaining device storage when factor patterns match.
    /// Schwarz also retains maps when membership and weights match; changed
    /// partitions, weights or factor patterns rebuild only its preconditioner.
    /// Duplicate input entries may change layout if coalesced sparsity matches.
    ///
    /// A successful update hides the old solution but preserves it as a warm
    /// start. Any failure hides output and discards warm-start availability.
    /// Validation failures leave the previous matrix usable with a zero start.
    /// Once uploads begin, a failure blocks all solves until a complete update
    /// succeeds. Recover backend errors first, then retry the old or new matrix.
    /// Terminal backend failures require a new pool and session.
    ///
    /// # Errors
    /// Rejects changed canonical sparsity, dimensions, nonfinite values or failed
    /// SPD/HPD admission. Propagates preconditioner, allocation and transfer errors.
    pub fn try_update_matrix(
        &mut self,
        matrix: &CsrMatrix<T>,
    ) -> Result<CudaLinearUpdateStatus, LinearSolveError> {
        self.published = false;
        let previous_warm_start = std::mem::replace(&mut self.warm_start_available, false);
        if self.preconditioner.is_prepared() {
            return Err(LinearSolveError::InvalidConfig(
                "prepared FGMRES requires try_update_matrix_lagged or try_update_matrix_with_preconditioner".into(),
            ));
        }
        let canonical = self.validate_update(matrix)?;
        // Host preparation rejects coefficients outside the real-SPD contract
        // before physical uploads and computes the new spectral bound once.
        let chebyshev_bound = if self.config.solver == LinearSolverType::GmresChebyshev {
            Some(try_chebyshev_preconditioner(&canonical, ChebyshevConfig::default())?.lambda_max())
        } else {
            None
        };
        self.numeric_ready = false;
        self.matrix
            .try_copy_values_from_host(self.pool.backend(), &canonical.values)
            .map_err(backend_error)?;
        let preconditioner_rebuilt = self.preconditioner.refresh(
            self.pool.backend(),
            &canonical,
            &self.matrix.matrix().values,
            self.config.solver,
            chebyshev_bound,
        )?;
        self.pool
            .backend()
            .try_synchronize()
            .map_err(backend_error)?;
        self.host_matrix = matrix.clone();
        self.numeric_ready = true;
        self.warm_start_available = previous_warm_start;
        Ok(CudaLinearUpdateStatus {
            preconditioner_rebuilt,
        })
    }
    pub(super) fn validate_update(
        &self,
        matrix: &CsrMatrix<T>,
    ) -> Result<CsrMatrix<T>, LinearSolveError> {
        validate_csr_structure(matrix)?;
        validate_linear_inputs(
            matrix,
            &vec![<T as DeviceScalar>::zero(); matrix.num_rows],
            None,
            &self.config,
        )?;
        if matches!(
            self.config.solver,
            LinearSolverType::Cg
                | LinearSolverType::CgAmg
                | LinearSolverType::GmresAmg
                | LinearSolverType::GmresChebyshev
        ) {
            validate_spd_system(matrix, self.config.solver, self.assumption)?;
        }
        let canonical = coalesced_csr(matrix);
        let previous = coalesced_csr(&self.host_matrix);
        if canonical.num_rows != previous.num_rows
            || canonical.num_cols != previous.num_cols
            || canonical.row_ptrs != previous.row_ptrs
            || canonical.col_indices != previous.col_indices
        {
            return Err(LinearSolveError::InvalidConfig(
                "CUDA numeric update requires unchanged canonical CSR sparsity; construct a new session".into(),
            ));
        }
        Ok(canonical)
    }
}
