//! Resident arbitrary-RHS solves for each orientation of a compressed BEM system.

// Rust guideline compliant 2026-02-21
use super::*;
use sonium_compute::{BackendError, DeviceOperator};

#[path = "bem_solver_cuda_hmatrix_block.rs"]
mod block;
#[doc(inline)]
pub use block::CudaHMatrixBlockSolveReport;

/// Orientation fixed when preparing a compressed linear solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaHMatrixAction {
    /// Solve A x = b.
    Forward,
    /// Solve A-transpose x = b without conjugation.
    Transpose,
    /// Solve A-Hermitian x = b.
    Hermitian,
}

/// Retains a compressed operator and oriented preconditioner for resident RHS solves.
///
/// Geometry, material, frequency, formulation and action remain fixed. Repeated
/// RHS solves reuse H-matrix factors, sparse inverse blocks and Krylov storage.
/// The result is a linear-system vector, not a physical scattering solution:
/// this type deliberately exposes no surface-pressure or field-output API.
/// Host setup and optional exact TBEM certification follow the scattering
/// session's policy; disabling exact certification retains device vectors.
#[derive(Debug)]
pub struct CudaHMatrixLinearSession<'a> {
    inner: CudaHMatrixBemSession<'a>,
    action: CudaHMatrixAction,
    block: Option<block::BlockState>,
}

impl BemSolver {
    /// Prepare a compressed solve after admitting native block GMRES dependencies.
    ///
    /// Use this entry point when multi-RHS execution is intended: block projection
    /// and sparse multi-vector capabilities are checked before host hierarchy and
    /// inverse construction. The resulting session also supports scalar solves.
    /// RHS width remains dynamic; block workspace and active-width caches warm on
    /// first use, as documented by [`CudaHMatrixLinearSession::solve_block_device`].
    ///
    /// # Errors
    /// Rejects pending context errors and unavailable native block dependencies,
    /// then propagates the geometry, compression and setup errors of
    /// [`Self::prepare_cuda_hmatrix_linear`].
    pub fn prepare_cuda_hmatrix_block<'a>(
        &self,
        problem: &BemProblem,
        pool: &'a MemoryPool<CudaBackend>,
        preconditioner: CudaHMatrixPreconditioner,
        action: CudaHMatrixAction,
    ) -> Result<CudaHMatrixLinearSession<'a>, BemError> {
        admit_cuda_krylov(
            pool.backend(),
            sonium_solvers::iterative::BackendKrylovMethod::BlockGmres,
        )?;
        self.prepare_cuda_hmatrix_linear(problem, pool, preconditioner, action)
    }

    /// Prepare a retained forward, transpose or Hermitian compressed CUDA solve.
    ///
    /// Uses the same constant-panel H-matrix producer as `prepare_cuda_hmatrix`.
    /// The chosen diagonal/cluster inverse is transposed or conjugate-transposed
    /// during host setup to match the system. The action cannot change afterward.
    /// Incident-source metadata is not interpreted as an adjoint RHS; callers
    /// supply an already-resident vector to `solve_device`.
    ///
    /// # Errors
    /// Rejects unsupported geometry/basis/assembly/solver settings, unresolved
    /// rank caps, singular preconditioner blocks, budgets without estimates,
    /// unavailable native kernels and backend failures.
    pub fn prepare_cuda_hmatrix_linear<'a>(
        &self,
        problem: &BemProblem,
        pool: &'a MemoryPool<CudaBackend>,
        preconditioner: CudaHMatrixPreconditioner,
        action: CudaHMatrixAction,
    ) -> Result<CudaHMatrixLinearSession<'a>, BemError> {
        Ok(CudaHMatrixLinearSession {
            inner: self.prepare_cuda_hmatrix_for_action(problem, pool, preconditioner, action)?,
            action,
            block: None,
        })
    }
}

impl CudaHMatrixLinearSession<'_> {
    /// Return the fixed system orientation.
    pub fn action(&self) -> CudaHMatrixAction {
        self.action
    }

    /// Return the linear-system dimension.
    pub fn num_dofs(&self) -> usize {
        self.inner.rhs.len()
    }

    /// Return the number of compressed far-field blocks.
    pub fn low_rank_blocks(&self) -> usize {
        self.inner.low_rank_blocks
    }

    /// Borrow the last certified linear-system solution on CUDA.
    pub fn solution_device(&self) -> Option<&CudaBuffer<Complex64>> {
        self.inner.solution_device()
    }

    /// Solve an already-resident RHS without changing the prepared operator.
    ///
    /// Native validation downloads a compact finite-value status. Optional exact
    /// TBEM certification downloads the solution and RHS and performs host work;
    /// its configured tolerance applies to the requested orientation.
    ///
    /// # Errors
    /// Rejects wrong dimensions, allocation extents, streams, non-finite RHS,
    /// unavailable warm starts, nonconvergence, failed exact certification and
    /// backend failures. Every error unpublishes output and warm-start eligibility;
    /// recovery requires a successful zero-start solve after backend recovery.
    pub fn solve_device(
        &mut self,
        rhs: &CudaBuffer<Complex64>,
        initial: DeviceInitialGuess,
    ) -> Result<CudaHMatrixSolveReport, BemError> {
        if let Some(block) = &mut self.block {
            block.published = false;
        }
        let s = &mut self.inner;
        let previous = std::mem::replace(&mut s.published, false);
        if initial == DeviceInitialGuess::PreviousSolution && !previous {
            return Err(failed("no certified previous H-matrix linear solution"));
        }
        let backend = s.pool.backend();
        if rhs.len() != s.rhs.len()
            || rhs.raw().is_none_or(|raw| {
                !std::sync::Arc::ptr_eq(raw.stream(), backend.stream())
                    || rhs.len().checked_mul(std::mem::size_of::<Complex64>()) != Some(raw.len())
            })
        {
            return Err(failed(
                "H-matrix RHS must match dimension, allocation and session stream",
            ));
        }
        if backend.try_all_finite_native(rhs).map_err(failed)? != Some(true) {
            return Err(failed(
                "non-finite H-matrix RHS or unavailable native validation",
            ));
        }
        let operator = OrientedOperator {
            operator: &s.operator,
            backend,
            action: self.action,
        };
        let status = s
            .krylov
            .solve(
                &operator,
                &s.preconditioner,
                rhs,
                initial,
                &s.controls,
                None,
            )
            .map_err(failed)?;
        if !status.converged {
            return Err(failed(format!(
                "compressed linear GMRES did not converge: {status:?}"
            )));
        }
        let exact_relative_residual = if let Some(tolerance) = s.exact_tolerance {
            let mut x = vec![Complex64::default(); s.rhs.len()];
            let mut b = x.clone();
            backend
                .try_copy_d2h(
                    s.krylov
                        .solution()
                        .ok_or_else(|| failed("missing certified iterate"))?,
                    &mut x,
                )
                .map_err(failed)?;
            backend.try_copy_d2h(rhs, &mut b).map_err(failed)?;
            let ax = exact_action(&s.metadata, s.beta_scale, self.action, &x)?;
            let norm = b.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
            let residual = ax
                .iter()
                .zip(b)
                .map(|(a, b)| (*a - b).norm_sqr())
                .sum::<f64>()
                .sqrt()
                / if norm > 0.0 { norm } else { 1.0 };
            if !residual.is_finite() || residual > tolerance {
                return Err(BemError::ResidualCheckFailed {
                    label: "cuda-hmatrix-linear-independent-exact".into(),
                    residual,
                    tolerance,
                });
            }
            Some(residual)
        } else {
            None
        };
        s.published = true;
        Ok(CudaHMatrixSolveReport {
            iterations: status.iterations,
            compressed_relative_residual: status.true_residual,
            exact_relative_residual,
        })
    }
}

pub(super) fn orient_inverse(
    inverse: CsrMatrix<Complex64>,
    action: CudaHMatrixAction,
) -> Result<CsrMatrix<Complex64>, BemError> {
    if action == CudaHMatrixAction::Forward {
        return Ok(inverse);
    }
    let mut entries = Vec::with_capacity(inverse.values.len());
    for row in 0..inverse.num_rows {
        for j in inverse.row_ptrs[row]..inverse.row_ptrs[row + 1] {
            let value = inverse.values[j];
            entries.push((
                inverse.col_indices[j],
                row,
                if action == CudaHMatrixAction::Hermitian {
                    value.conj()
                } else {
                    value
                },
            ));
        }
    }
    CsrMatrix::try_from_triplets(inverse.num_rows, inverse.num_cols, entries).map_err(failed)
}

fn exact_action(
    metadata: &BemSolution,
    beta_scale: f64,
    action: CudaHMatrixAction,
    x: &[Complex64],
) -> Result<Vec<Complex64>, BemError> {
    let apply_h = |v: &[Complex64]| {
        apply_tbem_stabilized_exact_hermitian(
            &metadata.elements,
            &metadata.nodes,
            &metadata.physics,
            1.0,
            beta_scale,
            v,
        )
        .map_err(failed)
    };
    match action {
        CudaHMatrixAction::Forward => apply_tbem_stabilized_exact(
            &metadata.elements,
            &metadata.nodes,
            &metadata.physics,
            1.0,
            beta_scale,
            x,
        )
        .map_err(failed),
        CudaHMatrixAction::Hermitian => apply_h(x),
        CudaHMatrixAction::Transpose => {
            let conjugated: Vec<_> = x.iter().map(|v| v.conj()).collect();
            Ok(apply_h(&conjugated)?
                .into_iter()
                .map(|v| v.conj())
                .collect())
        }
    }
}

struct OrientedOperator<'a> {
    operator: &'a CudaHMatrixC64,
    backend: &'a CudaBackend,
    action: CudaHMatrixAction,
}

impl DeviceOperator<CudaBackend, Complex64> for OrientedOperator<'_> {
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
            log::error!("CUDA H-matrix oriented action failed: {error}");
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
                operation: "cuda_hmatrix_oriented_operator",
                reason: "solver stream mismatch".into(),
            });
        }
        match self.action {
            CudaHMatrixAction::Forward => DeviceOperator::try_apply(self.operator, backend, x, y),
            CudaHMatrixAction::Transpose => self.operator.try_apply_transpose_raw(x, y),
            CudaHMatrixAction::Hermitian => self.operator.try_apply_hermitian_raw(x, y),
        }
    }

    fn try_apply_colmajor(
        &self,
        backend: &CudaBackend,
        rhs: usize,
        x: &CudaBuffer<Complex64>,
        y: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        if !std::sync::Arc::ptr_eq(backend.stream(), self.backend.stream()) {
            return Err(BackendError::InvalidArgument {
                operation: "cuda_hmatrix_oriented_block",
                reason: "solver stream mismatch".into(),
            });
        }
        match self.action {
            CudaHMatrixAction::Forward => self.operator.try_apply_colmajor(backend, rhs, x, y),
            CudaHMatrixAction::Transpose => {
                self.operator.try_apply_transpose_colmajor_raw(rhs, x, y)
            }
            CudaHMatrixAction::Hermitian => {
                self.operator.try_apply_hermitian_colmajor_raw(rhs, x, y)
            }
        }
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

#[cfg(test)]
#[path = "bem_solver_cuda_hmatrix_linear_tests.rs"]
mod tests;
