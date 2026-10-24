//! Retained multi-RHS solves for fixed compressed BEM operators.

// Rust guideline compliant 2026-02-21
use super::*;
use sonium_compute::{BufferView, ReductionMode};
use sonium_solvers::iterative::{BackendKrylovWorkspace, try_block_gmres_backend};

/// Independently certified residuals for a compressed block solve.
#[derive(Debug, Clone)]
pub struct CudaHMatrixBlockSolveReport {
    /// Block iterations, including any scalar recovery counted by the solver.
    pub iterations: usize,
    /// Number of restart transitions.
    pub restarts: usize,
    /// Per-column residuals against the compressed operator.
    /// Zero RHS columns use an absolute residual norm.
    pub compressed_relative_residuals: Vec<f64>,
    /// Optional per-column uncompressed TBEM residuals, computed on the host.
    pub exact_relative_residuals: Option<Vec<f64>>,
}

pub(super) struct BlockState {
    rhs_count: usize,
    output: Buffer<Complex64, CudaBackend>,
    initial: Buffer<Complex64, CudaBackend>,
    residual: Buffer<Complex64, CudaBackend>,
    column: Buffer<Complex64, CudaBackend>,
    krylov: BackendKrylovWorkspace<CudaBackend, Complex64>,
    pub(super) published: bool,
}

impl std::fmt::Debug for BlockState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockState")
            .field("rhs_count", &self.rhs_count)
            .field("published", &self.published)
            .finish_non_exhaustive()
    }
}

impl CudaHMatrixLinearSession<'_> {
    /// Borrow the last certified column-major block solution.
    ///
    /// Scalar solves and failed block calls invalidate this publication.
    pub fn block_solution_device(&self) -> Option<&CudaBuffer<Complex64>> {
        self.block
            .as_ref()
            .filter(|b| b.published)
            .map(|b| b.output.raw())
    }

    /// Solve resident RHS columns with retained compressed block GMRES.
    ///
    /// The RHS has `num_dofs() * rhs_count` entries in column-major order.
    /// The operator and oriented inverse remain fixed. Changing RHS width
    /// replaces block storage; previous-solution starts require the same width.
    /// H-matrix columns execute sequentially with resident scratch; the inverse
    /// uses sparse matrix-matrix multiplication. First-use active widths may
    /// allocate and are cached thereafter. Pool quota coverage excludes the
    /// existing block basis and sparse descriptor allocation policy.
    /// Optional exact TBEM certification downloads the full RHS and solution.
    ///
    /// # Errors
    /// Rejects invalid extents, streams, non-finite data, unavailable warm starts,
    /// allocation failures, nonconvergence and failed residual certification.
    /// Any error invalidates published output and previous-solution eligibility.
    pub fn solve_block_device(
        &mut self,
        rhs: &CudaBuffer<Complex64>,
        rhs_count: usize,
        initial: DeviceInitialGuess,
    ) -> Result<CudaHMatrixBlockSolveReport, BemError> {
        self.inner.published = false;
        let previous = self.block.as_mut().is_some_and(|b| {
            let published = std::mem::replace(&mut b.published, false);
            published && b.rhs_count == rhs_count
        });
        let s = &self.inner;
        let n = s.rhs.len();
        let len = n
            .checked_mul(rhs_count)
            .filter(|_| rhs_count > 0)
            .ok_or_else(|| failed("invalid block RHS count or extent"))?;
        let backend = s.pool.backend();
        // Scalar-prepared sessions may request blocks later. Admit the extra
        // dependencies before native validation or block workspace allocation.
        admit_cuda_krylov(
            backend,
            sonium_solvers::iterative::BackendKrylovMethod::BlockGmres,
        )?;
        if rhs.len() != len
            || rhs.raw().is_none_or(|raw| {
                !std::sync::Arc::ptr_eq(raw.stream(), backend.stream())
                    || len.checked_mul(std::mem::size_of::<Complex64>()) != Some(raw.len())
            })
        {
            return Err(failed("block RHS dimension, allocation or stream mismatch"));
        }
        if initial == DeviceInitialGuess::PreviousSolution && !previous {
            return Err(failed("no certified previous block with this RHS width"));
        }
        if backend.try_all_finite_native(rhs).map_err(failed)? != Some(true) {
            return Err(failed(
                "non-finite block RHS or unavailable native validation",
            ));
        }
        if self.block.as_ref().is_none_or(|b| b.rhs_count != rhs_count) {
            self.block = Some(BlockState {
                rhs_count,
                output: s.pool.try_alloc(len).map_err(failed)?,
                initial: s.pool.try_alloc(len).map_err(failed)?,
                residual: s.pool.try_alloc(len).map_err(failed)?,
                column: s.pool.try_alloc(n).map_err(failed)?,
                krylov: BackendKrylovWorkspace::with_pool(n, s.pool),
                published: false,
            });
        }
        let block = self
            .block
            .as_mut()
            .ok_or_else(|| failed("missing block workspace"))?;
        let x0 = if initial == DeviceInitialGuess::PreviousSolution {
            backend
                .try_copy_d2d(block.output.raw(), block.initial.raw_mut())
                .map_err(failed)?;
            Some(block.initial.raw())
        } else {
            None
        };
        let operator = OrientedOperator {
            operator: &s.operator,
            backend,
            action: self.action,
        };
        let status = try_block_gmres_backend(
            backend,
            &mut block.krylov,
            &operator,
            &s.preconditioner,
            rhs,
            x0,
            block.output.raw_mut(),
            rhs_count,
            &s.controls,
            None,
        )
        .map_err(failed)?;
        if !status.converged {
            return Err(failed(format!(
                "compressed block GMRES did not converge: {status:?}"
            )));
        }
        // Certify each original RHS independently of the recurrence's status.
        operator
            .try_apply_colmajor(
                backend,
                rhs_count,
                block.output.raw(),
                block.residual.raw_mut(),
            )
            .map_err(failed)?;
        backend
            .try_scale(Complex64::new(-1.0, 0.0), block.residual.raw_mut())
            .map_err(failed)?;
        backend
            .try_axpy(Complex64::new(1.0, 0.0), rhs, block.residual.raw_mut())
            .map_err(failed)?;
        let mut residuals = Vec::with_capacity(rhs_count);
        for j in 0..rhs_count {
            let view = BufferView::contiguous(j * n, n);
            let column = BufferView::contiguous(0, n);
            backend
                .try_copy_views_d2d(rhs, view, block.column.raw_mut(), column)
                .map_err(failed)?;
            let norm = backend
                .try_nrm2_with_mode(ReductionMode::Fast, block.column.raw())
                .map_err(failed)?;
            backend
                .try_copy_views_d2d(block.residual.raw(), view, block.column.raw_mut(), column)
                .map_err(failed)?;
            let residual = backend
                .try_nrm2_with_mode(ReductionMode::Fast, block.column.raw())
                .map_err(failed)?
                / if norm > 0.0 { norm } else { 1.0 };
            if !residual.is_finite() || residual > s.controls.tolerance {
                return Err(BemError::ResidualCheckFailed {
                    label: format!("cuda-hmatrix-block-column-{j}"),
                    residual,
                    tolerance: s.controls.tolerance,
                });
            }
            residuals.push(residual);
        }
        let exact_residuals = if let Some(tolerance) = s.exact_tolerance {
            let mut x = vec![Complex64::default(); len];
            let mut b = x.clone();
            backend
                .try_copy_d2h(block.output.raw(), &mut x)
                .map_err(failed)?;
            backend.try_copy_d2h(rhs, &mut b).map_err(failed)?;
            let mut values = Vec::with_capacity(rhs_count);
            for (j, (x, b)) in x.chunks_exact(n).zip(b.chunks_exact(n)).enumerate() {
                let ax = exact_action(&s.metadata, s.beta_scale, self.action, x)?;
                let norm = b.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
                let residual = ax
                    .iter()
                    .zip(b)
                    .map(|(a, b)| (*a - *b).norm_sqr())
                    .sum::<f64>()
                    .sqrt()
                    / if norm > 0.0 { norm } else { 1.0 };
                if !residual.is_finite() || residual > tolerance {
                    return Err(BemError::ResidualCheckFailed {
                        label: format!("cuda-hmatrix-block-exact-column-{j}"),
                        residual,
                        tolerance,
                    });
                }
                values.push(residual);
            }
            Some(values)
        } else {
            None
        };
        block.published = true;
        Ok(CudaHMatrixBlockSolveReport {
            iterations: status.iterations,
            restarts: status.restarts,
            compressed_relative_residuals: residuals,
            exact_relative_residuals: exact_residuals,
        })
    }
}
