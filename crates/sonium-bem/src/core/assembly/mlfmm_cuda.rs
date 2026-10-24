//! Retained CUDA solves for an admitted non-directional MLFMM hierarchy.

// Rust guideline compliant 2026-02-21
use super::MlfmmSystem;
use crate::core::bem_solver::BemError;
use num_complex::Complex64;
use sonium_compute::{
    Backend, BackendError, BackendResult, CudaBackend, CudaBuffer, CudaMlfmmC64, DeviceBufferLike,
    DeviceOperator, DevicePreconditioner, MemoryPool, PersistentDeviceCsr, Transpose,
};
use sonium_solvers::iterative::{
    BackendKrylovConfig, BackendKrylovStatus, DeviceGmresSession, DeviceInitialGuess,
};
use sonium_solvers::{CsrMatrix, TripletMatrix};

/// Explicit local preconditioning for the uploaded MLFMM system.
#[derive(Debug, Clone, Copy)]
pub enum CudaMlfmmPreconditioner {
    /// Inverse of the near-field self-block diagonal.
    Jacobi,
    /// Exact inverses of disjoint leaf self-blocks.
    LeafBlockJacobi,
}

/// Fixed MLFMM operator, oriented local inverse and reusable CUDA GMRES storage.
///
/// Tree construction and local factorization are host setup. Repeated RHS
/// solves keep vectors on CUDA. The certified residual is against the uploaded
/// MLFMM approximation; it is not an independent uncompressed TBEM certificate.
/// Reprepare after any matrix, hierarchy or orientation change. Existing sparse
/// descriptor allocations are outside the pool quota.
pub struct CudaMlfmmSolveSession<'a> {
    pool: &'a MemoryPool<CudaBackend>,
    operator: CudaMlfmmC64,
    inverse: Inverse,
    action: Transpose,
    controls: BackendKrylovConfig<f64>,
    krylov: DeviceGmresSession<'a, CudaBackend, Complex64>,
    published: bool,
}

impl std::fmt::Debug for CudaMlfmmSolveSession<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaMlfmmSolveSession")
            .field("dofs", &self.operator.num_dofs())
            .field("action", &self.action)
            .field("published", &self.published)
            .finish_non_exhaustive()
    }
}

fn failed(error: impl std::fmt::Display) -> BemError {
    BemError::SolverFailed(error.to_string())
}

impl MlfmmSystem {
    /// Prepare a persistent solve without materializing the global dense matrix.
    ///
    /// The supplied controls must request native execution. Only leaf self-blocks
    /// are inverted; the full operator retains its near/far representation.
    /// `try_to_cuda` admission rejects directional plans and repeated leaf DOFs.
    ///
    /// # Errors
    /// Rejects invalid controls, unsupported hierarchies, missing/overlapping or
    /// singular self-blocks, non-finite coefficients, allocation and CUDA failures.
    pub fn try_prepare_cuda_solve<'a>(
        &self,
        pool: &'a MemoryPool<CudaBackend>,
        action: Transpose,
        preconditioner: CudaMlfmmPreconditioner,
        controls: BackendKrylovConfig<f64>,
    ) -> Result<CudaMlfmmSolveSession<'a>, BemError> {
        if self.num_dofs == 0
            || !controls.require_native
            || controls.restart == 0
            || controls.max_iterations == 0
            || !controls.tolerance.is_finite()
            || controls.tolerance <= 0.0
        {
            return Err(BemError::InvalidParameters(
                "invalid native MLFMM solver controls".into(),
            ));
        }
        if let Some(error) = pool.backend().pending_error().map_err(failed)? {
            return Err(failed(error));
        }
        sonium_solvers::iterative::validate_backend_krylov::<_, Complex64>(
            pool.backend(),
            sonium_solvers::iterative::BackendKrylovMethod::Gmres,
            true,
        ).map_err(failed)?;
        let host_inverse = inverse_blocks(self, action, preconditioner)?;
        let operator = self.try_to_cuda(pool.clone()).map_err(failed)?;
        let inverse = Inverse(
            PersistentDeviceCsr::try_from_host_csr_ref(
                pool.backend(),
                host_inverse.as_host_csr_ref(),
            )
            .map_err(failed)?,
        );
        let mut krylov = DeviceGmresSession::try_new(pool, self.num_dofs).map_err(failed)?;
        let cycle = controls.restart.min(controls.max_iterations);
        krylov.try_reserve_cycle(cycle).map_err(failed)?;
        pool.backend()
            .try_reserve_krylov_reductions::<Complex64>(cycle)
            .map_err(failed)?;
        Ok(CudaMlfmmSolveSession {
            pool,
            operator,
            inverse,
            action,
            controls,
            krylov,
            published: false,
        })
    }
}

impl CudaMlfmmSolveSession<'_> {
    /// Borrow the last certified device solution.
    pub fn solution_device(&self) -> Option<&CudaBuffer<Complex64>> {
        self.published.then(|| self.krylov.solution()).flatten()
    }

    /// Solve a resident RHS with the prepared operator and local preconditioner.
    ///
    /// # Errors
    /// Rejects invalid RHS extents/streams, non-finite values, unavailable warm
    /// starts, nonconvergence and backend failures. Errors invalidate publication
    /// and warm-start eligibility. Recover the backend before a zero-start retry;
    /// terminal synchronization failures require a new pool and session.
    pub fn solve_device(
        &mut self,
        rhs: &CudaBuffer<Complex64>,
        initial: DeviceInitialGuess,
    ) -> Result<BackendKrylovStatus<f64>, BemError> {
        let previous = std::mem::replace(&mut self.published, false);
        if initial == DeviceInitialGuess::PreviousSolution && !previous {
            return Err(failed("no certified previous MLFMM solution"));
        }
        let backend = self.pool.backend();
        if rhs.len() != self.operator.num_dofs()
            || rhs.raw().is_none_or(|raw| {
                !std::sync::Arc::ptr_eq(raw.stream(), backend.stream())
                    || rhs.len().checked_mul(std::mem::size_of::<Complex64>()) != Some(raw.len())
            })
        {
            return Err(failed("MLFMM RHS dimension, allocation or stream mismatch"));
        }
        if backend.try_all_finite_native(rhs).map_err(failed)? != Some(true) {
            return Err(failed("non-finite RHS or native validation unavailable"));
        }
        let operator = Oriented {
            operator: &self.operator,
            backend,
            action: self.action,
        };
        let status = self
            .krylov
            .solve(&operator, &self.inverse, rhs, initial, &self.controls, None)
            .map_err(failed)?;
        if !status.converged {
            return Err(failed(format!("MLFMM GMRES did not converge: {status:?}")));
        }
        self.published = true;
        Ok(status)
    }
}

struct Inverse(PersistentDeviceCsr<CudaBackend, Complex64>);
impl DevicePreconditioner<CudaBackend, Complex64> for Inverse {
    fn apply(
        &self,
        backend: &CudaBackend,
        r: &CudaBuffer<Complex64>,
        z: &mut CudaBuffer<Complex64>,
    ) {
        if let Err(error) = self.try_apply(backend, r, z) {
            log::error!("MLFMM inverse failed: {error}");
        }
    }
    fn try_apply(
        &self,
        backend: &CudaBackend,
        r: &CudaBuffer<Complex64>,
        z: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        self.0.try_apply(backend, r, z)
    }
}

struct Oriented<'a> {
    operator: &'a CudaMlfmmC64,
    backend: &'a CudaBackend,
    action: Transpose,
}
impl DeviceOperator<CudaBackend, Complex64> for Oriented<'_> {
    fn n(&self) -> usize {
        self.operator.num_dofs()
    }
    fn apply(
        &self,
        backend: &CudaBackend,
        x: &CudaBuffer<Complex64>,
        y: &mut CudaBuffer<Complex64>,
    ) {
        if let Err(error) = self.try_apply(backend, x, y) {
            log::error!("MLFMM action failed: {error}");
        }
    }
    fn try_apply(
        &self,
        backend: &CudaBackend,
        x: &CudaBuffer<Complex64>,
        y: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        if !std::sync::Arc::ptr_eq(backend.stream(), self.backend.stream()) {
            return Err(BackendError::InvalidArgument {
                operation: "mlfmm_solve",
                reason: "solver stream mismatch".into(),
            });
        }
        self.operator.try_apply_raw(x, y, self.action)
    }
    fn try_residual(
        &self,
        backend: &CudaBackend,
        x: &CudaBuffer<Complex64>,
        b: &CudaBuffer<Complex64>,
        out: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        self.try_apply(backend, x, out)?;
        backend.try_scale(Complex64::new(-1.0, 0.0), out)?;
        backend.try_axpy(Complex64::new(1.0, 0.0), b, out)
    }
}

fn inverse_blocks(
    system: &MlfmmSystem,
    action: Transpose,
    kind: CudaMlfmmPreconditioner,
) -> Result<CsrMatrix<Complex64>, BemError> {
    let mut result = TripletMatrix::new(system.num_dofs, system.num_dofs);
    let mut covered = vec![false; system.num_dofs];
    let mut push = |row, col, value: Complex64| match action {
        Transpose::NoTrans => result.push(row, col, value),
        Transpose::Trans => result.push(col, row, value),
        Transpose::ConjTrans => result.push(col, row, value.conj()),
    };
    for block in system
        .near_matrix
        .iter()
        .filter(|b| b.source_cluster == b.field_cluster)
    {
        let dofs = system
            .leaf_dof_indices
            .get(block.source_cluster)
            .ok_or_else(|| failed("invalid self-block cluster"))?;
        if block.coefficients.nrows() != dofs.len() || block.coefficients.ncols() != dofs.len() {
            return Err(failed("MLFMM self-block shape mismatch"));
        }
        for &dof in dofs {
            let used = covered
                .get_mut(dof)
                .ok_or_else(|| failed("self-block DOF outside system"))?;
            if std::mem::replace(used, true) {
                return Err(failed("overlapping MLFMM self-blocks"));
            }
        }
        for i in 0..dofs.len() {
            for j in 0..dofs.len() {
                let v = block.coefficients[[i, j]];
                if !v.re.is_finite() || !v.im.is_finite() {
                    return Err(failed("non-finite self-block"));
                }
            }
        }
        match kind {
            CudaMlfmmPreconditioner::Jacobi => {
                for (i, &dof) in dofs.iter().enumerate() {
                    let v = block.coefficients[[i, i]];
                    if v.norm() == 0.0 {
                        return Err(failed("zero MLFMM diagonal"));
                    }
                    push(dof, dof, v.inv());
                }
            }
            CudaMlfmmPreconditioner::LeafBlockJacobi => {
                let factor = sonium_solvers::lu_factorize(&block.coefficients).map_err(failed)?;
                let mut unit = vec![Complex64::default(); dofs.len()];
                let mut column = unit.clone();
                for (j, &col) in dofs.iter().enumerate() {
                    unit[j] = Complex64::new(1.0, 0.0);
                    factor.solve_into(&unit, &mut column).map_err(failed)?;
                    unit[j] = Complex64::default();
                    for (i, &row) in dofs.iter().enumerate() {
                        push(row, col, column[i]);
                    }
                }
            }
        }
    }
    if covered.iter().any(|v| !*v) {
        return Err(failed("missing MLFMM leaf self-blocks"));
    }
    let inverse = result.try_to_csr().map_err(failed)?;
    if inverse
        .values
        .iter()
        .any(|v| !v.re.is_finite() || !v.im.is_finite())
    {
        return Err(failed("non-finite MLFMM inverse"));
    }
    Ok(inverse)
}

#[cfg(test)]
#[path = "mlfmm_cuda_tests.rs"]
mod tests;
