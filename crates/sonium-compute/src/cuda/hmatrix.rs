//! Device-resident Complex64 hierarchical matrix actions.
//!
//! The BEM crate owns tree construction and compression policy.  This module
//! owns the reusable device representation of the resulting dense and
//! low-rank blocks. Scalar actions accumulate into persistent device scratch and
//! publish to the caller-owned output with one device-to-device copy. Block
//! actions use indexed column views and shared rank scratch without full-vector
//! copies; discard partially written block output after an error. The only
//! host-to-device traffic is the one-time upload performed by
//! [`CudaHMatrixC64::new`].

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use num_complex::Complex64;

use crate::backend::{Backend, BufferView};
use crate::cuda::{CudaBackend, CudaBuffer};
use crate::error::{BackendError, BackendResult};
use crate::pool::{Buffer, MemoryPool};
use crate::{DeviceBufferLike, DeviceOperator};

/// Host description of one dense hierarchical block.
#[derive(Debug, Clone)]
pub struct CudaDenseBlockC64Input {
    /// Global output indices for the block rows.
    pub row_indices: Vec<usize>,
    /// Global input indices for the block columns.
    pub col_indices: Vec<usize>,
    /// Row-major block values.
    pub matrix: Vec<Complex64>,
}

/// Host description of one low-rank hierarchical block.
///
/// The block reconstructs as `A[i, j] = sum_l u[i, l] * v[j, l]` without
/// conjugation.  The factor arrays are row-major with shapes
/// `(row_indices.len(), rank)` and `(col_indices.len(), rank)`.
#[derive(Debug, Clone)]
pub struct CudaLowRankBlockC64Input {
    /// Global output indices represented by rows of `u`.
    pub row_indices: Vec<usize>,
    /// Global input indices represented by rows of `v`.
    pub col_indices: Vec<usize>,
    /// Row-major `U` factor.
    pub u: Vec<Complex64>,
    /// Row-major `V` factor.
    pub v: Vec<Complex64>,
    /// Numerical rank of the factorization.
    pub rank: usize,
}

struct DeviceDenseBlockC64 {
    row_indices: Buffer<i32, CudaBackend>,
    col_indices: Buffer<i32, CudaBackend>,
    forward: Buffer<Complex64, CudaBackend>,
    transpose: Buffer<Complex64, CudaBackend>,
    hermitian: Buffer<Complex64, CudaBackend>,
    rows: usize,
    cols: usize,
}

struct DeviceLowRankBlockC64 {
    row_indices: Buffer<i32, CudaBackend>,
    col_indices: Buffer<i32, CudaBackend>,
    u: Buffer<Complex64, CudaBackend>,
    v_transpose: Buffer<Complex64, CudaBackend>,
    v: Buffer<Complex64, CudaBackend>,
    u_transpose: Buffer<Complex64, CudaBackend>,
    hermitian_v: Buffer<Complex64, CudaBackend>,
    hermitian_u_transpose: Buffer<Complex64, CudaBackend>,
    rows: usize,
    cols: usize,
    rank: usize,
}

struct CudaHMatrixWorkspace {
    input: Buffer<Complex64, CudaBackend>,
    projection: Buffer<Complex64, CudaBackend>,
    rank_indices: Buffer<i32, CudaBackend>,
    logical_indices: Buffer<i32, CudaBackend>,
    result: Buffer<Complex64, CudaBackend>,
}

/// Device-resident hierarchical matrix representation for Complex64.
///
/// Dense near-field blocks and low-rank far-field factors are uploaded once.
/// Forward, algebraic-transpose, and Hermitian actions then reuse those
/// buffers and a rank-sized projection workspace without staging vectors or
/// block data through the host.
pub struct CudaHMatrixC64 {
    pool: MemoryPool<CudaBackend>,
    num_dofs: usize,
    dense_blocks: Vec<DeviceDenseBlockC64>,
    low_rank_blocks: Vec<DeviceLowRankBlockC64>,
    workspace: Mutex<CudaHMatrixWorkspace>,
}

/// Resident first-order wavenumber update for a compressed Complex64
/// H-matrix.
///
/// The value and analytic fixed-geometry derivative operators are uploaded
/// once.  [`Self::try_apply`] evaluates
/// `A(k0 + delta_k) x ≈ A(k0)x + delta_k A'(k0)x` entirely on the device and
/// publishes only after both resident actions and the device combination have
/// completed.  The caller must repeat the exact `base_wavenumber` token used
/// during construction; this prevents a stale derivative from silently being
/// used with a rebuilt value operator.
pub struct CudaHMatrixWavenumberUpdateC64 {
    value: CudaHMatrixC64,
    derivative: CudaHMatrixC64,
    pool: MemoryPool<CudaBackend>,
    base_wavenumber: f64,
    value_output: Buffer<Complex64, CudaBackend>,
    derivative_output: Buffer<Complex64, CudaBackend>,
    combined_output: Buffer<Complex64, CudaBackend>,
}

impl DeviceOperator<CudaBackend, Complex64> for CudaHMatrixC64 {
    fn n(&self) -> usize {
        self.num_dofs
    }

    fn apply(
        &self,
        backend: &CudaBackend,
        x: &CudaBuffer<Complex64>,
        y: &mut CudaBuffer<Complex64>,
    ) {
        if let Err(error) = DeviceOperator::try_apply(self, backend, x, y) {
            log::error!("CUDA H-matrix action failed: {error}");
        }
    }

    fn try_apply(
        &self,
        backend: &CudaBackend,
        x: &CudaBuffer<Complex64>,
        y: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        if !Arc::ptr_eq(backend.stream(), self.pool.backend().stream()) {
            return Err(BackendError::InvalidArgument {
                operation: "cuda_hmatrix_operator",
                reason: "operator and solver must share the CUDA stream".into(),
            });
        }
        self.try_apply_raw_kind(x, y, HMatrixAction::Forward)
    }

    fn try_apply_colmajor(
        &self,
        backend: &CudaBackend,
        rhs: usize,
        x: &CudaBuffer<Complex64>,
        y: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        if !Arc::ptr_eq(backend.stream(), self.pool.backend().stream()) {
            return Err(BackendError::InvalidArgument {
                operation: "cuda_hmatrix_block",
                reason: "operator and solver must share the CUDA stream".into(),
            });
        }
        self.try_apply_colmajor_kind(rhs, x, y, HMatrixAction::Forward)
    }

    fn try_residual(
        &self,
        backend: &CudaBackend,
        x: &CudaBuffer<Complex64>,
        b: &CudaBuffer<Complex64>,
        out: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        if b.len() != self.num_dofs {
            return Err(BackendError::InvalidArgument {
                operation: "cuda_hmatrix_residual",
                reason: "RHS dimension mismatch".into(),
            });
        }
        DeviceOperator::try_apply(self, backend, x, out)?;
        backend.try_scale(Complex64::new(-1.0, 0.0), out)?;
        backend.try_axpy(Complex64::new(1.0, 0.0), b, out)
    }
}

impl CudaHMatrixWavenumberUpdateC64 {
    /// Join resident value and derivative operators for one fixed geometry.
    ///
    /// Both operators must have the same global dimension.  The supplied
    /// pool is also used for persistent combination buffers, so warm updates
    /// do not allocate or stage structure.
    pub fn new(
        value: CudaHMatrixC64,
        derivative: CudaHMatrixC64,
        pool: MemoryPool<CudaBackend>,
        base_wavenumber: f64,
    ) -> BackendResult<Self> {
        if value.num_dofs != derivative.num_dofs {
            return Err(BackendError::InvalidArgument {
                operation: "cuda_hmatrix_wavenumber_update",
                reason: format!(
                    "value and derivative dimensions differ ({} vs {})",
                    value.num_dofs, derivative.num_dofs
                ),
            });
        }
        if !base_wavenumber.is_finite() {
            return Err(BackendError::InvalidArgument {
                operation: "cuda_hmatrix_wavenumber_update",
                reason: "base wavenumber must be finite".into(),
            });
        }
        let value_output = pool.try_alloc(value.num_dofs).map_err(BackendError::from)?;
        let derivative_output = pool.try_alloc(value.num_dofs).map_err(BackendError::from)?;
        let combined_output = pool.try_alloc(value.num_dofs).map_err(BackendError::from)?;
        Ok(Self {
            value,
            derivative,
            pool,
            base_wavenumber,
            value_output,
            derivative_output,
            combined_output,
        })
    }

    /// Global dimension of the resident update operator.
    pub fn num_dofs(&self) -> usize {
        self.value.num_dofs
    }

    /// Base wavenumber token captured when the resident pair was built.
    pub fn base_wavenumber(&self) -> f64 {
        self.base_wavenumber
    }

    /// Apply the resident first-order wavenumber update to device vectors.
    ///
    /// The caller-provided output is untouched when metadata validation or a
    /// value/derivative launch fails.  A successful update publishes from a
    /// persistent device combination buffer after synchronization.
    pub fn try_apply(
        &mut self,
        input: &Buffer<Complex64, CudaBackend>,
        output: &mut Buffer<Complex64, CudaBackend>,
        base_wavenumber: f64,
        delta_wavenumber: f64,
    ) -> BackendResult<()> {
        let operation = "cuda_hmatrix_wavenumber_update";
        if input.len() != self.num_dofs() || output.len() != self.num_dofs() {
            return Err(BackendError::InvalidArgument {
                operation,
                reason: format!(
                    "input/output lengths must equal operator dimension {} (got {}/{})",
                    self.num_dofs(),
                    input.len(),
                    output.len()
                ),
            });
        }
        if !base_wavenumber.is_finite() || !delta_wavenumber.is_finite() {
            return Err(BackendError::InvalidArgument {
                operation,
                reason: "wavenumber metadata and update must be finite".into(),
            });
        }
        let token_tolerance = 1.0e-12
            * self
                .base_wavenumber
                .abs()
                .max(base_wavenumber.abs())
                .max(1.0);
        if (base_wavenumber - self.base_wavenumber).abs() > token_tolerance {
            return Err(BackendError::InvalidArgument {
                operation,
                reason: format!(
                    "stale base wavenumber metadata: expected {:.17e}, got {:.17e}",
                    self.base_wavenumber, base_wavenumber
                ),
            });
        }
        if !(self.base_wavenumber + delta_wavenumber).is_finite() {
            return Err(BackendError::InvalidArgument {
                operation,
                reason: "updated wavenumber overflows finite range".into(),
            });
        }

        self.value.try_apply(input, &mut self.value_output)?;
        self.derivative
            .try_apply(input, &mut self.derivative_output)?;
        let backend = self.pool.backend();
        backend.try_copy_d2d(self.value_output.raw(), self.combined_output.raw_mut())?;
        backend.try_axpy(
            Complex64::new(delta_wavenumber, 0.0),
            self.derivative_output.raw(),
            self.combined_output.raw_mut(),
        )?;
        backend.try_synchronize()?;
        backend.try_copy_d2d(self.combined_output.raw(), output.raw_mut())
    }
}

#[derive(Clone, Copy)]
enum HMatrixAction {
    Forward,
    Transpose,
    Hermitian,
}

impl CudaHMatrixC64 {
    /// Upload a hierarchical matrix and allocate its reusable projection
    /// workspace.  Invalid indices, shapes, non-finite coefficients, and
    /// integer-range overflow are rejected before any block is published.
    pub fn new(
        pool: MemoryPool<CudaBackend>,
        num_dofs: usize,
        dense_blocks: Vec<CudaDenseBlockC64Input>,
        low_rank_blocks: Vec<CudaLowRankBlockC64Input>,
    ) -> BackendResult<Self> {
        let mut dense = Vec::with_capacity(dense_blocks.len());
        for (block, input) in dense_blocks.into_iter().enumerate() {
            let rows = input.row_indices.len();
            let cols = input.col_indices.len();
            let expected = rows.checked_mul(cols).ok_or_else(|| {
                invalid_block(
                    "cuda_hmatrix_dense_block",
                    block,
                    "matrix shape overflows usize",
                )
            })?;
            if input.matrix.len() != expected {
                return Err(invalid_block(
                    "cuda_hmatrix_dense_block",
                    block,
                    format!(
                        "matrix length {} does not match {rows}x{cols}",
                        input.matrix.len()
                    ),
                ));
            }
            let row_indices = checked_indices(
                num_dofs,
                &input.row_indices,
                "cuda_hmatrix_dense_block",
                block,
                "row",
            )?;
            let col_indices = checked_indices(
                num_dofs,
                &input.col_indices,
                "cuda_hmatrix_dense_block",
                block,
                "column",
            )?;
            validate_finite(&input.matrix, "cuda_hmatrix_dense_block", block)?;
            let transpose = transpose_matrix(&input.matrix, rows, cols, false);
            let hermitian = transpose_matrix(&input.matrix, rows, cols, true);
            dense.push(DeviceDenseBlockC64 {
                row_indices: upload(&pool, &row_indices, "dense row indices")?,
                col_indices: upload(&pool, &col_indices, "dense column indices")?,
                forward: upload(&pool, &input.matrix, "dense forward block")?,
                transpose: upload(&pool, &transpose, "dense transpose block")?,
                hermitian: upload(&pool, &hermitian, "dense Hermitian block")?,
                rows,
                cols,
            });
        }

        let mut low_rank = Vec::with_capacity(low_rank_blocks.len());
        let mut max_rank = 0usize;
        for (block, input) in low_rank_blocks.into_iter().enumerate() {
            let rows = input.row_indices.len();
            let cols = input.col_indices.len();
            let expected_u = rows.checked_mul(input.rank).ok_or_else(|| {
                invalid_block(
                    "cuda_hmatrix_low_rank_block",
                    block,
                    "U shape overflows usize",
                )
            })?;
            let expected_v = cols.checked_mul(input.rank).ok_or_else(|| {
                invalid_block(
                    "cuda_hmatrix_low_rank_block",
                    block,
                    "V shape overflows usize",
                )
            })?;
            if input.u.len() != expected_u || input.v.len() != expected_v {
                return Err(invalid_block(
                    "cuda_hmatrix_low_rank_block",
                    block,
                    format!(
                        "factor lengths U={} (expected {expected_u}), V={} (expected {expected_v})",
                        input.u.len(),
                        input.v.len()
                    ),
                ));
            }
            let row_indices = checked_indices(
                num_dofs,
                &input.row_indices,
                "cuda_hmatrix_low_rank_block",
                block,
                "row",
            )?;
            let col_indices = checked_indices(
                num_dofs,
                &input.col_indices,
                "cuda_hmatrix_low_rank_block",
                block,
                "column",
            )?;
            validate_finite(&input.u, "cuda_hmatrix_low_rank_block", block)?;
            validate_finite(&input.v, "cuda_hmatrix_low_rank_block", block)?;
            let v_transpose = transpose_matrix(&input.v, cols, input.rank, false);
            let u_transpose = transpose_matrix(&input.u, rows, input.rank, false);
            let hermitian_v: Vec<Complex64> = input.v.iter().map(Complex64::conj).collect();
            let hermitian_u_transpose = transpose_matrix(&input.u, rows, input.rank, true);
            max_rank = max_rank.max(input.rank);
            low_rank.push(DeviceLowRankBlockC64 {
                row_indices: upload(&pool, &row_indices, "low-rank row indices")?,
                col_indices: upload(&pool, &col_indices, "low-rank column indices")?,
                u: upload(&pool, &input.u, "low-rank U factor")?,
                v_transpose: upload(&pool, &v_transpose, "low-rank V transpose")?,
                v: upload(&pool, &input.v, "low-rank V factor")?,
                u_transpose: upload(&pool, &u_transpose, "low-rank U transpose")?,
                hermitian_v: upload(&pool, &hermitian_v, "low-rank Hermitian V factor")?,
                hermitian_u_transpose: upload(
                    &pool,
                    &hermitian_u_transpose,
                    "low-rank Hermitian U transpose",
                )?,
                rows,
                cols,
                rank: input.rank,
            });
        }

        let rank_indices: Vec<i32> = (0..max_rank)
            .map(|index| {
                i32::try_from(index).map_err(|_| BackendError::InvalidArgument {
                    operation: "cuda_hmatrix_workspace",
                    reason: "rank exceeds signed CUDA index range".into(),
                })
            })
            .collect::<Result<_, _>>()?;
        let logical_indices: Vec<i32> = (0..num_dofs)
            .map(|index| {
                i32::try_from(index).map_err(|_| BackendError::InvalidArgument {
                    operation: "cuda_hmatrix_workspace",
                    reason: "operator dimension exceeds signed CUDA index range".into(),
                })
            })
            .collect::<Result<_, _>>()?;
        let workspace = CudaHMatrixWorkspace {
            input: pool.try_alloc(num_dofs).map_err(BackendError::from)?,
            projection: upload(
                &pool,
                &vec![Complex64::new(0.0, 0.0); max_rank],
                "FMM projection",
            )?,
            rank_indices: upload(&pool, &rank_indices, "FMM rank indices")?,
            logical_indices: upload(&pool, &logical_indices, "FMM logical indices")?,
            result: pool.try_alloc(num_dofs).map_err(BackendError::from)?,
        };

        Ok(Self {
            pool,
            num_dofs,
            dense_blocks: dense,
            low_rank_blocks: low_rank,
            workspace: Mutex::new(workspace),
        })
    }

    /// Global dimension of the operator.
    pub fn num_dofs(&self) -> usize {
        self.num_dofs
    }

    /// Apply `A*x` into a zeroed-and-accumulated device output.
    pub fn try_apply(
        &self,
        input: &Buffer<Complex64, CudaBackend>,
        output: &mut Buffer<Complex64, CudaBackend>,
    ) -> BackendResult<()> {
        self.try_apply_kind(input, output, HMatrixAction::Forward)
    }

    /// Apply the algebraic transpose `A^T*x` into a device output.
    pub fn try_apply_transpose(
        &self,
        input: &Buffer<Complex64, CudaBackend>,
        output: &mut Buffer<Complex64, CudaBackend>,
    ) -> BackendResult<()> {
        self.try_apply_kind(input, output, HMatrixAction::Transpose)
    }

    /// Apply the Hermitian transpose `A^H*x` into a device output.
    pub fn try_apply_hermitian(
        &self,
        input: &Buffer<Complex64, CudaBackend>,
        output: &mut Buffer<Complex64, CudaBackend>,
    ) -> BackendResult<()> {
        self.try_apply_kind(input, output, HMatrixAction::Hermitian)
    }

    /// Apply algebraic transpose to borrowed solver buffers without host staging.
    ///
    /// # Errors
    /// Rejects mismatched lengths, allocation extents or streams and backend failures.
    pub fn try_apply_transpose_raw(
        &self,
        input: &CudaBuffer<Complex64>,
        output: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        self.try_apply_raw_kind(input, output, HMatrixAction::Transpose)
    }

    /// Apply Hermitian transpose to borrowed solver buffers without host staging.
    ///
    /// # Errors
    /// Rejects mismatched lengths, allocation extents or streams and backend failures.
    pub fn try_apply_hermitian_raw(
        &self,
        input: &CudaBuffer<Complex64>,
        output: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        self.try_apply_raw_kind(input, output, HMatrixAction::Hermitian)
    }

    /// Apply transpose to resident column-major RHS blocks.
    ///
    /// Columns use indexed views and retained rank scratch without vector copies or allocation.
    /// A backend error can leave completed output columns; discard output on error.
    ///
    /// # Errors
    /// Rejects invalid extents, foreign streams and backend failures.
    pub fn try_apply_transpose_colmajor_raw(
        &self,
        rhs: usize,
        input: &CudaBuffer<Complex64>,
        output: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        self.try_apply_colmajor_kind(rhs, input, output, HMatrixAction::Transpose)
    }

    /// Apply Hermitian transpose to resident column-major RHS blocks.
    ///
    /// Columns use indexed views and retained rank scratch without vector copies or allocation.
    /// A backend error can leave completed output columns; discard output on error.
    ///
    /// # Errors
    /// Rejects invalid extents, foreign streams and backend failures.
    pub fn try_apply_hermitian_colmajor_raw(
        &self,
        rhs: usize,
        input: &CudaBuffer<Complex64>,
        output: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        self.try_apply_colmajor_kind(rhs, input, output, HMatrixAction::Hermitian)
    }

    fn try_apply_colmajor_kind(
        &self,
        rhs: usize,
        input: &CudaBuffer<Complex64>,
        output: &mut CudaBuffer<Complex64>,
        action: HMatrixAction,
    ) -> BackendResult<()> {
        let operation = "cuda_hmatrix_block";
        let len = self
            .num_dofs
            .checked_mul(rhs)
            .ok_or_else(|| BackendError::InvalidArgument {
                operation,
                reason: "block extent overflow".into(),
            })?;
        let backend = self.pool.backend();
        for buffer in [input, &*output] {
            if buffer.len() != len
                || (len > 0
                    && buffer.raw().is_none_or(|raw| {
                        !Arc::ptr_eq(raw.stream(), backend.stream())
                            || len.checked_mul(std::mem::size_of::<Complex64>()) != Some(raw.len())
                    }))
            {
                return Err(BackendError::InvalidArgument {
                    operation,
                    reason: "block dimension, allocation or stream mismatch".into(),
                });
            }
        }
        if len == 0 {
            return Ok(());
        }
        backend.try_fmm_action_preflight()?;
        let mut workspace = self
            .workspace
            .lock()
            .map_err(|_| BackendError::OperationFailed {
                operation,
                reason: "workspace mutex poisoned".into(),
            })?;
        // Block output has always been discard-on-error. Accumulate directly
        // into column views rather than staging full input/result vectors.
        backend.try_fmm_zero_c64_view(output, BufferView::contiguous(0, len))?;
        for j in 0..rhs {
            let column = BufferView::contiguous(j * self.num_dofs, self.num_dofs);
            for block in &self.dense_blocks {
                let (matrix, rows, cols, row_indices, col_indices) = match action {
                    HMatrixAction::Forward => (
                        block.forward.raw(),
                        block.rows,
                        block.cols,
                        block.row_indices.raw(),
                        block.col_indices.raw(),
                    ),
                    HMatrixAction::Transpose => (
                        block.transpose.raw(),
                        block.cols,
                        block.rows,
                        block.col_indices.raw(),
                        block.row_indices.raw(),
                    ),
                    HMatrixAction::Hermitian => (
                        block.hermitian.raw(),
                        block.cols,
                        block.rows,
                        block.col_indices.raw(),
                        block.row_indices.raw(),
                    ),
                };
                backend.try_fmm_indexed_dense_matvec_accumulate_c64_view(
                    matrix,
                    0,
                    rows,
                    cols,
                    cols,
                    input,
                    column,
                    col_indices,
                    BufferView::contiguous(0, cols),
                    output,
                    column,
                    row_indices,
                    BufferView::contiguous(0, rows),
                )?;
            }
            for block in &self.low_rank_blocks {
                if block.rank == 0 {
                    continue;
                }
                let (left, right, rows, cols, row_indices, col_indices) = match action {
                    HMatrixAction::Forward => (
                        block.u.raw(),
                        block.v_transpose.raw(),
                        block.rows,
                        block.cols,
                        block.row_indices.raw(),
                        block.col_indices.raw(),
                    ),
                    HMatrixAction::Transpose => (
                        block.v.raw(),
                        block.u_transpose.raw(),
                        block.cols,
                        block.rows,
                        block.col_indices.raw(),
                        block.row_indices.raw(),
                    ),
                    HMatrixAction::Hermitian => (
                        block.hermitian_v.raw(),
                        block.hermitian_u_transpose.raw(),
                        block.cols,
                        block.rows,
                        block.col_indices.raw(),
                        block.row_indices.raw(),
                    ),
                };
                let rank = BufferView::contiguous(0, block.rank);
                let CudaHMatrixWorkspace {
                    projection,
                    rank_indices,
                    ..
                } = &mut *workspace;
                backend.try_fmm_zero_c64_view(projection.raw_mut(), rank)?;
                backend.try_fmm_indexed_dense_matvec_accumulate_c64_view(
                    right,
                    0,
                    block.rank,
                    cols,
                    cols,
                    input,
                    column,
                    col_indices,
                    BufferView::contiguous(0, cols),
                    projection.raw_mut(),
                    rank,
                    rank_indices.raw(),
                    rank,
                )?;
                backend.try_fmm_dense_matvec_scatter_accumulate_c64_view(
                    left,
                    0,
                    rows,
                    block.rank,
                    block.rank,
                    projection.raw(),
                    rank,
                    output,
                    column,
                    row_indices,
                    BufferView::contiguous(0, rows),
                )?;
            }
        }
        Ok(())
    }

    /// Apply `A*x` where both vectors are checked one-dimensional views into
    /// caller-owned device buffers.  The hierarchy and all action workspaces
    /// remain resident; only the logical RHS is gathered and the result is
    /// scattered on the device.
    pub fn try_apply_view(
        &self,
        input: &Buffer<Complex64, CudaBackend>,
        input_view: BufferView,
        output: &mut Buffer<Complex64, CudaBackend>,
        output_view: BufferView,
    ) -> BackendResult<()> {
        self.try_apply_kind_view(
            input,
            input_view,
            output,
            output_view,
            HMatrixAction::Forward,
        )
    }

    /// Apply `A^T*x` to checked device-buffer views without host staging.
    pub fn try_apply_transpose_view(
        &self,
        input: &Buffer<Complex64, CudaBackend>,
        input_view: BufferView,
        output: &mut Buffer<Complex64, CudaBackend>,
        output_view: BufferView,
    ) -> BackendResult<()> {
        self.try_apply_kind_view(
            input,
            input_view,
            output,
            output_view,
            HMatrixAction::Transpose,
        )
    }

    /// Apply `A^H*x` to checked device-buffer views without host staging.
    pub fn try_apply_hermitian_view(
        &self,
        input: &Buffer<Complex64, CudaBackend>,
        input_view: BufferView,
        output: &mut Buffer<Complex64, CudaBackend>,
        output_view: BufferView,
    ) -> BackendResult<()> {
        self.try_apply_kind_view(
            input,
            input_view,
            output,
            output_view,
            HMatrixAction::Hermitian,
        )
    }

    fn try_apply_kind(
        &self,
        input: &Buffer<Complex64, CudaBackend>,
        output: &mut Buffer<Complex64, CudaBackend>,
        action: HMatrixAction,
    ) -> BackendResult<()> {
        self.try_apply_raw_kind(input.raw(), output.raw_mut(), action)
    }

    fn try_apply_raw_kind(
        &self,
        input: &CudaBuffer<Complex64>,
        output: &mut CudaBuffer<Complex64>,
        action: HMatrixAction,
    ) -> BackendResult<()> {
        let operation = action.operation();
        if input.len() != self.num_dofs || output.len() != self.num_dofs {
            return Err(BackendError::InvalidArgument {
                operation,
                reason: format!(
                    "input/output lengths must equal operator dimension {} (got {}/{})",
                    self.num_dofs,
                    input.len(),
                    output.len()
                ),
            });
        }
        let backend = self.pool.backend();
        if self.num_dofs > 0 {
            for buffer in [input, &*output] {
                if buffer.raw().is_none_or(|raw| {
                    !Arc::ptr_eq(raw.stream(), backend.stream())
                        || buffer.len().checked_mul(std::mem::size_of::<Complex64>())
                            != Some(raw.len())
                }) {
                    return Err(BackendError::InvalidArgument {
                        operation,
                        reason: "buffer allocation and stream must match the H-matrix".into(),
                    });
                }
            }
        }
        backend.try_fmm_action_preflight()?;
        let mut workspace = self
            .workspace
            .lock()
            .map_err(|_| BackendError::OperationFailed {
                operation: "cuda_hmatrix_workspace",
                reason: "workspace mutex poisoned".into(),
            })?;
        self.execute_action(backend, Some(input), action, &mut workspace)?;
        let result = &workspace.result;
        backend.try_copy_d2d(result.raw(), output)
    }

    fn try_apply_kind_view(
        &self,
        input: &Buffer<Complex64, CudaBackend>,
        input_view: BufferView,
        output: &mut Buffer<Complex64, CudaBackend>,
        output_view: BufferView,
        action: HMatrixAction,
    ) -> BackendResult<()> {
        let operation = action.operation_view();
        if input_view.len != self.num_dofs || output_view.len != self.num_dofs {
            return Err(BackendError::InvalidArgument {
                operation,
                reason: format!(
                    "input/output view lengths must equal operator dimension {} (got {}/{})",
                    self.num_dofs, input_view.len, output_view.len
                ),
            });
        }
        validate_view(operation, input.len(), input_view, "input")?;
        validate_view(operation, output.len(), output_view, "output")?;
        let backend = self.pool.backend();
        backend.try_fmm_action_preflight()?;
        let mut workspace = self
            .workspace
            .lock()
            .map_err(|_| BackendError::OperationFailed {
                operation: "cuda_hmatrix_workspace",
                reason: "workspace mutex poisoned".into(),
            })?;
        let logical_view = BufferView::contiguous(0, self.num_dofs);
        {
            let workspace = &mut *workspace;
            backend.try_fmm_compact_indexed_gather_c64_view(
                input.raw(),
                input_view,
                workspace.logical_indices.raw(),
                logical_view,
                workspace.input.raw_mut(),
                logical_view,
            )?;
        }
        self.execute_action(backend, None, action, &mut workspace)?;
        // This native zero touches only the logical output view, preserving
        // all caller padding before the resident result is scattered into it.
        let workspace = &mut *workspace;
        backend.try_fmm_zero_c64_view(output.raw_mut(), output_view)?;
        backend.try_fmm_compact_indexed_scatter_accumulate_c64_view(
            workspace.result.raw(),
            logical_view,
            workspace.logical_indices.raw(),
            logical_view,
            output.raw_mut(),
            output_view,
        )
    }

    fn execute_action(
        &self,
        backend: &CudaBackend,
        external_input: Option<&CudaBuffer<Complex64>>,
        action: HMatrixAction,
        workspace: &mut CudaHMatrixWorkspace,
    ) -> BackendResult<()> {
        // Accumulate into persistent device scratch and publish only after
        // every block dispatch has been accepted. A failed launch therefore
        // cannot destroy a caller-owned output that may be reused for retry.
        let CudaHMatrixWorkspace {
            input: resident_input,
            projection,
            rank_indices,
            result,
            ..
        } = workspace;
        let input = external_input.unwrap_or(resident_input.raw());
        backend.try_scale(Complex64::new(0.0, 0.0), result.raw_mut())?;

        for block in &self.dense_blocks {
            match action {
                HMatrixAction::Forward => backend.try_fmm_indexed_dense_matvec_accumulate_c64(
                    block.forward.raw(),
                    block.rows,
                    block.cols,
                    block.cols,
                    input,
                    block.col_indices.raw(),
                    result.raw_mut(),
                    block.row_indices.raw(),
                )?,
                HMatrixAction::Transpose => backend.try_fmm_indexed_dense_matvec_accumulate_c64(
                    block.transpose.raw(),
                    block.cols,
                    block.rows,
                    block.rows,
                    input,
                    block.row_indices.raw(),
                    result.raw_mut(),
                    block.col_indices.raw(),
                )?,
                HMatrixAction::Hermitian => backend.try_fmm_indexed_dense_matvec_accumulate_c64(
                    block.hermitian.raw(),
                    block.cols,
                    block.rows,
                    block.rows,
                    input,
                    block.row_indices.raw(),
                    result.raw_mut(),
                    block.col_indices.raw(),
                )?,
            }
        }

        for block in &self.low_rank_blocks {
            if block.rank == 0 {
                continue;
            }
            backend.try_scale(Complex64::new(0.0, 0.0), projection.raw_mut())?;
            match action {
                HMatrixAction::Forward => {
                    backend.try_fmm_indexed_dense_matvec_accumulate_c64(
                        block.v_transpose.raw(),
                        block.rank,
                        block.cols,
                        block.cols,
                        input,
                        block.col_indices.raw(),
                        projection.raw_mut(),
                        rank_indices.raw(),
                    )?;
                    backend.try_fmm_dense_matvec_scatter_accumulate_c64(
                        block.u.raw(),
                        block.rows,
                        block.rank,
                        block.rank,
                        projection.raw(),
                        result.raw_mut(),
                        block.row_indices.raw(),
                    )?;
                }
                HMatrixAction::Transpose => {
                    backend.try_fmm_indexed_dense_matvec_accumulate_c64(
                        block.u_transpose.raw(),
                        block.rank,
                        block.rows,
                        block.rows,
                        input,
                        block.row_indices.raw(),
                        projection.raw_mut(),
                        rank_indices.raw(),
                    )?;
                    backend.try_fmm_dense_matvec_scatter_accumulate_c64(
                        block.v.raw(),
                        block.cols,
                        block.rank,
                        block.rank,
                        projection.raw(),
                        result.raw_mut(),
                        block.col_indices.raw(),
                    )?;
                }
                HMatrixAction::Hermitian => {
                    backend.try_fmm_indexed_dense_matvec_accumulate_c64(
                        block.hermitian_u_transpose.raw(),
                        block.rank,
                        block.rows,
                        block.rows,
                        input,
                        block.row_indices.raw(),
                        projection.raw_mut(),
                        rank_indices.raw(),
                    )?;
                    backend.try_fmm_dense_matvec_scatter_accumulate_c64(
                        block.hermitian_v.raw(),
                        block.cols,
                        block.rank,
                        block.rank,
                        projection.raw(),
                        result.raw_mut(),
                        block.col_indices.raw(),
                    )?;
                }
            }
        }
        Ok(())
    }
}

impl HMatrixAction {
    fn operation(self) -> &'static str {
        match self {
            Self::Forward => "cuda_hmatrix_apply",
            Self::Transpose => "cuda_hmatrix_apply_transpose",
            Self::Hermitian => "cuda_hmatrix_apply_hermitian",
        }
    }

    fn operation_view(self) -> &'static str {
        match self {
            Self::Forward => "cuda_hmatrix_apply_view",
            Self::Transpose => "cuda_hmatrix_apply_transpose_view",
            Self::Hermitian => "cuda_hmatrix_apply_hermitian_view",
        }
    }
}

fn validate_view(
    operation: &'static str,
    buffer_len: usize,
    view: BufferView,
    label: &'static str,
) -> BackendResult<()> {
    let end = view
        .checked_end_exclusive()
        .map_err(|error| BackendError::InvalidArgument {
            operation,
            reason: format!("{label} view metadata is invalid: {error}"),
        })?;
    if end > buffer_len {
        return Err(BackendError::InvalidArgument {
            operation,
            reason: format!("{label} view end {end} exceeds buffer length {buffer_len}"),
        });
    }
    view.descriptor::<Complex64>("cuda")
        .map_err(|error| BackendError::InvalidArgument {
            operation,
            reason: format!("{label} view descriptor is invalid: {error}"),
        })?;
    Ok(())
}

fn invalid_block(operation: &'static str, block: usize, reason: impl Into<String>) -> BackendError {
    BackendError::InvalidArgument {
        operation,
        reason: format!("block {block}: {}", reason.into()),
    }
}

fn checked_indices(
    num_dofs: usize,
    indices: &[usize],
    operation: &'static str,
    block: usize,
    side: &'static str,
) -> BackendResult<Vec<i32>> {
    let converted: Vec<i32> = indices
        .iter()
        .enumerate()
        .map(|(position, &index)| {
            if index >= num_dofs {
                return Err(invalid_block(
                    operation,
                    block,
                    format!("{side} index {index} at position {position} exceeds {num_dofs}"),
                ));
            }
            i32::try_from(index).map_err(|_| {
                invalid_block(
                    operation,
                    block,
                    format!("{side} index {index} exceeds signed CUDA index range"),
                )
            })
        })
        .collect::<Result<_, _>>()?;
    let mut seen = HashSet::with_capacity(converted.len());
    if let Some((position, index)) = converted
        .iter()
        .enumerate()
        .find(|(_, index)| !seen.insert(**index))
    {
        return Err(invalid_block(
            operation,
            block,
            format!("{side} index {index} is duplicated at position {position}"),
        ));
    }
    Ok(converted)
}

fn validate_finite(
    values: &[Complex64],
    operation: &'static str,
    block: usize,
) -> BackendResult<()> {
    values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
        .map_or(Ok(()), |(index, _)| {
            Err(invalid_block(
                operation,
                block,
                format!("coefficient {index} is non-finite"),
            ))
        })
}

fn transpose_matrix(
    matrix: &[Complex64],
    rows: usize,
    cols: usize,
    conjugate: bool,
) -> Vec<Complex64> {
    let mut result = vec![Complex64::new(0.0, 0.0); rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            let value = matrix[row * cols + col];
            result[col * rows + row] = if conjugate { value.conj() } else { value };
        }
    }
    result
}

fn upload<T: crate::scalar::BufferElement>(
    pool: &MemoryPool<CudaBackend>,
    values: &[T],
    label: &'static str,
) -> BackendResult<Buffer<T, CudaBackend>> {
    let mut buffer = pool.try_alloc(values.len()).map_err(BackendError::from)?;
    pool.backend().try_copy_h2d(values, buffer.raw_mut())?;
    let _ = label;
    Ok(buffer)
}
